# Copyright (c) 2026, AMD.

"""256×256 tile MFMA GEMM with explicit LDS/MFMA interleaving via
``rocdl.sched_*`` directives — EXPERIMENTAL, **WORSE than K=16 baseline**.

Same kernel structure as ``gemm_gfx950_256x256.py`` (K=16 per iter,
4-wave WG, 128×128 register tile per wave, 8×8 MFMA grid). The only
change: inside the consume body we emit ``rocdl.sched_mfma`` /
``rocdl.sched_dsrd`` / ``rocdl.sched_barrier`` directives every 8
MFMAs, trying to force the AMDGPU backend to interleave LDS reads
with MFMA compute in alternating groups.

Result (MI355X, f16, time per call, smaller = better):
    shape         256_K16      256_sched    delta
    1024³         102.5        112.8        +10.1% slower
    2048³         210.8        224.4         +6.5% slower
    4096³         455.2        503.1        +10.5% slower

Why it failed: ``sched_group_barrier`` pins the compiler to the
order we emit it in. Our consume body emits 64 A/B frag loads first,
then 64 MFMAs — the sched directives applied over this order don't
produce better interleaving than the LLVM scheduler's default,
they just block LLVM from reordering across the barriers. This is
the opposite of what we wanted.

The right shape to get 256×256 competitive with hipBLASLt's
MT256x256x64 kernel probably requires a different register
allocation strategy (software-pipelined A/B load + MFMA, not
interleaving hints), or a full rewrite that emits loads and MFMAs
interleaved in source order without needing sched hints.

Reference: FlyDSL's preshuffle_gemm.py uses this pattern on a
differently-shaped hot loop where it does help.
"""

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu as _gpu, range_constexpr, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch


_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 16
_FRAG_A = 4
_FRAG_B = 4
_FRAG_C = 4
_WAVE_ROWS = 8
_WAVE_COLS = 8

_A_TILE_BYTES = 256 * 16 * 2
_B_TILE_BYTES = 16 * 256 * 2


def _align(x, a):
    return (x + a - 1) & ~(a - 1)


def _build_gemm_256x256_sched(*, M, N, K, dtype_str, arch):
    assert M % 256 == 0 and N % 256 == 0 and K % _MFMA_K == 0
    assert dtype_str in ("f16", "bf16")

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_256x256_sched_{dtype_str}_{M}_{N}_{K}_smem",
    )
    a_off0 = _align(allocator.ptr, 16); allocator.ptr = a_off0 + _A_TILE_BYTES
    b_off0 = _align(allocator.ptr, 16); allocator.ptr = b_off0 + _B_TILE_BYTES
    a_off1 = _align(allocator.ptr, 16); allocator.ptr = a_off1 + _A_TILE_BYTES
    b_off1 = _align(allocator.ptr, 16); allocator.ptr = b_off1 + _B_TILE_BYTES

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y
        tid = fx.thread_idx.x

        wave_id = tid // fx.Int32(64)
        lane_in_wave = tid % fx.Int32(64)
        lane_row = lane_in_wave % fx.Int32(16)
        lane_k_group = lane_in_wave // fx.Int32(16)

        wave_row = wave_id // fx.Int32(2)
        wave_col = wave_id % fx.Int32(2)

        in_elem_type = T.f16 if dtype_str == "f16" else T.bf16

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        base_ptr = allocator.get_base()
        s_A = [
            SmemPtr(base_ptr, a_off0, in_elem_type, shape=(256, 16)),
            SmemPtr(base_ptr, a_off1, in_elem_type, shape=(256, 16)),
        ]
        s_B = [
            SmemPtr(base_ptr, b_off0, in_elem_type, shape=(16, 256)),
            SmemPtr(base_ptr, b_off1, in_elem_type, shape=(16, 256)),
        ]
        for sp in (*s_A, *s_B):
            sp.get()

        ca_h8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), in_elem_type)
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        h8_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(8, 1), fx.AddressSpace.Register)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        h8_lay = fx.make_layout(8, 1)
        reg_lay = fx.make_layout(1, 1)

        def _load_h8(div_vec, vec_idx):
            r = fx.memref_alloca(h8_reg_ty, h8_lay)
            fx.copy_atom_call(ca_h8, fx.slice(div_vec, (None, vec_idx)), r)
            return fx.memref_load_vec(r)

        def _store_f(div, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            ts = _vfull(1, Float32(val), Float32)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_f, r, fx.slice(div, (None, idx)))

        def _idx(i):
            return ArithValue(i).index_cast(T.index) if hasattr(i, "index_cast") else fx.Index(i)

        m_base_wg = bid_m * fx.Int32(256)
        n_base_wg = bid_n * fx.Int32(256)

        a_load_row = tid

        def _a_row_in_tile(r):
            return wave_row * fx.Int32(128) + fx.Int32(r * 16) + lane_row

        def _b_col_in_tile(c):
            return wave_col * fx.Int32(128) + fx.Int32(c * 16) + lane_row

        acc_ty = T.vec(_FRAG_C, T.f32)

        def _zero_acc():
            zs = []
            for _ in range_constexpr(_FRAG_C):
                zs.append(arith.constant(0.0, type=T.f32))
            return vector.from_elements(acc_ty, zs)

        accs = [[_zero_acc() for _ in range_constexpr(_WAVE_COLS)]
                for _ in range_constexpr(_WAVE_ROWS)]

        def _stage_load(k_tile_scalar, stage):
            k_base = fx.Int32(k_tile_scalar * _MFMA_K)
            a_hbm_row = m_base_wg + a_load_row
            a_row_slice = fx.slice(A_buf, (a_hbm_row, None))
            a_div_v = fx.logical_divide(a_row_slice, h8_lay)
            for k_vec in range_constexpr(2):
                a_hbm_vec_idx = fx.Int32(k_tile_scalar * 2) + fx.Int32(k_vec)
                a_chunk = _load_h8(a_div_v, a_hbm_vec_idx)
                for i in range_constexpr(8):
                    s_A[stage].store(
                        a_chunk[i].ir_value(),
                        [_idx(a_load_row),
                         _idx(fx.Int32(k_vec * 8) + fx.Int32(i))],
                    )
            for pos_off in range_constexpr(2):
                linear_pos = tid * fx.Int32(2) + fx.Int32(pos_off)
                b_k = linear_pos // fx.Int32(32)
                b_n_vec8 = linear_pos % fx.Int32(32)
                b_hbm_k = k_base + b_k
                b_k_slice = fx.slice(B_buf, (b_hbm_k, None))
                b_div_v = fx.logical_divide(b_k_slice, h8_lay)
                b_hbm_vec_idx = bid_n * fx.Int32(256 // 8) + b_n_vec8
                b_chunk = _load_h8(b_div_v, b_hbm_vec_idx)
                for i in range_constexpr(8):
                    s_B[stage].store(
                        b_chunk[i].ir_value(),
                        [_idx(b_k),
                         _idx(b_n_vec8 * fx.Int32(8) + fx.Int32(i))],
                    )

        def _consume(stage, accs):
            # Load all A fragments (8 × vec4 = 32 scalar LDS reads).
            a_frags = []
            for r in range_constexpr(_WAVE_ROWS):
                a_vals = []
                a_row_l = _a_row_in_tile(r)
                for i in range_constexpr(_FRAG_A):
                    k_in_lds = lane_k_group * fx.Int32(_FRAG_A) + fx.Int32(i)
                    a_vals.append(s_A[stage].load([_idx(a_row_l), _idx(k_in_lds)]))
                a_frags.append(vector.from_elements(T.vec(_FRAG_A, in_elem_type), a_vals))

            # Load all B fragments (8 × 4 scalar = 32 LDS reads).
            b_frags = []
            for c in range_constexpr(_WAVE_COLS):
                b_vals = []
                b_col_l = _b_col_in_tile(c)
                for i in range_constexpr(_FRAG_B):
                    k_in_lds = lane_k_group * fx.Int32(_FRAG_B) + fx.Int32(i)
                    b_vals.append(s_B[stage].load([_idx(k_in_lds), _idx(b_col_l)]))
                b_frags.append(vector.from_elements(T.vec(_FRAG_B, in_elem_type), b_vals))

            def _mfma(a, b, acc):
                if dtype_str == "bf16":
                    a_i = vector.bitcast(T.vec(_FRAG_A, T.i16), a)
                    b_i = vector.bitcast(T.vec(_FRAG_B, T.i16), b)
                    return fx.rocdl.mfma_f32_16x16x16bf16_1k(
                        acc_ty, [a_i, b_i, acc, 0, 0, 0],
                    )
                return fx.rocdl.mfma_f32_16x16x16f16(acc_ty, [a, b, acc, 0, 0, 0])

            # --- Scheduling hints ---
            # 64 total ds_read (32 A + 32 B), 64 MFMAs.
            # 8 scheduling groups of (8 dsrd + 8 mfma): the backend
            # emits 8 LDS reads followed by 8 MFMAs, then a barrier,
            # repeating 8 times. This interleaves LDS latency with
            # MFMA compute — LDS reads of group N+1 proceed while
            # MFMAs of group N are still in the pipe.
            fx.rocdl.sched_barrier(0)  # clear any prior region

            # Emit MFMAs with sched directives interleaved.
            idx_mfma = 0
            for r in range_constexpr(_WAVE_ROWS):
                for c in range_constexpr(_WAVE_COLS):
                    accs[r][c] = _mfma(a_frags[r], b_frags[c], accs[r][c])
                    idx_mfma += 1
                    # Every 8 MFMAs, insert a sched group
                    if idx_mfma % 8 == 0:
                        fx.rocdl.sched_mfma(8)
                        fx.rocdl.sched_dsrd(8)
                        fx.rocdl.sched_barrier(0)

            return accs

        k_tiles = K // _MFMA_K

        _stage_load(0, 0)
        _gpu.barrier()

        for k_tile in range_constexpr(k_tiles):
            cur = k_tile & 1
            if k_tile < k_tiles - 1:
                _stage_load(k_tile + 1, 1 - cur)
            accs = _consume(cur, accs)
            _gpu.barrier()

        wave_m_base = m_base_wg + wave_row * fx.Int32(128)
        wave_n_base = n_base_wg + wave_col * fx.Int32(128)

        def _store_subtile(acc, row_base, col_base):
            for i in range_constexpr(_FRAG_C):
                out_row = row_base + lane_k_group * fx.Int32(_FRAG_C) + fx.Int32(i)
                out_col = col_base + lane_row
                row_c = fx.slice(C_buf, (out_row, None))
                c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))
                val_i = vector.extract(acc, static_position=[i], dynamic_position=[])
                _store_f(c_div, out_col, val_i)

        for r in range_constexpr(_WAVE_ROWS):
            for c in range_constexpr(_WAVE_COLS):
                _store_subtile(
                    accs[r][c],
                    wave_m_base + fx.Int32(r * 16),
                    wave_n_base + fx.Int32(c * 16),
                )

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, C).launch(
            grid=(M // 256, N // 256, 1), block=(256, 1, 1), stream=stream,
        )

    return launch


_kernel_cache: dict = {}
_DTYPE2STR = {torch.float16: "f16", torch.bfloat16: "bf16"}


def _compile(M, N, K, dtype_str, arch):
    key = (M, N, K, dtype_str, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_256x256_sched(M=M, N=N, K=K, dtype_str=dtype_str, arch=arch)
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_gemm_256x256_sched_out",
    mutates_args=("out",),
    schema="(Tensor a, Tensor b, Tensor(a0!) out) -> ()",
)
def _gemm_256x256_sched_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    assert a.is_cuda and b.is_cuda and out.is_cuda
    assert a.dtype in (torch.float16, torch.bfloat16) and a.dtype == b.dtype
    assert out.dtype == torch.float32
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and M % 256 == 0 and N % 256 == 0 and K % 16 == 0
    assert out.shape == (M, N)
    _compile(M, N, K, _DTYPE2STR[a.dtype], get_rocm_arch())(a, b, out)


@_gemm_256x256_sched_out.register_fake
def _gemm_256x256_sched_out_fake(a, b, out):
    return None


def gemm_256x256_sched(A: Tensor, B: Tensor) -> Tensor:
    """256×256 tile GEMM with explicit sched_mfma/sched_dsrd hints."""
    assert A.is_cuda and B.is_cuda
    assert A.dtype == B.dtype and A.dtype in (torch.float16, torch.bfloat16)
    M, K = A.shape
    _, N = B.shape
    out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    _gemm_256x256_sched_out(A, B, out)
    return out


__all__ = ["gemm_256x256_sched"]
