# Copyright (c) 2026, AMD.

"""4-wave 64×64 MFMA GEMM with cooperative LDS sharing + ping-pong prefetch.

Extends ``gemm_gfx950_4wave_lds.py`` with 2-stage LDS buffering:

  Prologue: cooperative HBM→LDS of the k=0 tile into stage 0.
  gpu.barrier

  for k in 0..K_tiles-2:
      cur = k & 1, nxt = 1 - cur
      issue HBM→LDS of tile k+1 into stage `nxt`
      read current MFMA fragments from stage `cur` LDS
      4 waves × 4 MFMAs each = 16 MFMAs per lane overlap with the issue
      gpu.barrier — stage `nxt` becomes `cur` next iter

  Last iter: MFMAs on final stage without prefetch.

LDS budget: 2 stages × (64×16 A + 16×64 B) = 8 KiB per workgroup.

Targets the 256-1024 mid-range where the single-stage cooperative LDS
kernel (``gemm_gfx950_4wave_lds.py``) lost 0.68-0.84× vs non-LDS
4-wave — the single barrier between load and consume phases serialises
HBM bandwidth against MFMA compute. Ping-pong overlaps them.

Scope: f16 × f16 → f32; M, N multiples of 64, K multiple of 16. bf16
follow-up.
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

_A_TILE_BYTES = 64 * 16 * 2   # per stage (f16)
_B_TILE_BYTES = 16 * 64 * 2


def _align(x, a):
    return (x + a - 1) & ~(a - 1)


def _build_gemm_64x64_4wave_lds_pp(*, M, N, K, dtype_str, arch):
    assert M % 64 == 0 and N % 64 == 0 and K % _MFMA_K == 0
    assert dtype_str in ("f16", "bf16")

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=(
            f"quack_amd_gemm_64x64_4wave_lds_pp_{dtype_str}_{M}_{N}_{K}_smem"
        ),
    )
    # Two stages × (A tile + B tile).
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
            SmemPtr(base_ptr, a_off0, in_elem_type, shape=(64, 16)),
            SmemPtr(base_ptr, a_off1, in_elem_type, shape=(64, 16)),
        ]
        s_B = [
            SmemPtr(base_ptr, b_off0, in_elem_type, shape=(16, 64)),
            SmemPtr(base_ptr, b_off1, in_elem_type, shape=(16, 64)),
        ]
        for sp in (*s_A, *s_B):
            sp.get()

        ca_h4 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), in_elem_type)
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        h4_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(4, 1), fx.AddressSpace.Register)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        h4_lay = fx.make_layout(4, 1)
        reg_lay = fx.make_layout(1, 1)

        def _load_h4(div_vec, vec_idx):
            r = fx.memref_alloca(h4_reg_ty, h4_lay)
            fx.copy_atom_call(ca_h4, fx.slice(div_vec, (None, vec_idx)), r)
            return fx.memref_load_vec(r)

        def _store_f(div, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            ts = _vfull(1, Float32(val), Float32)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_f, r, fx.slice(div, (None, idx)))

        def _idx(i):
            return ArithValue(i).index_cast(T.index) if hasattr(i, "index_cast") else fx.Index(i)

        m_base_wg = bid_m * fx.Int32(64)
        n_base_wg = bid_n * fx.Int32(64)

        # Cooperative load thread mappings.
        a_load_row = tid // fx.Int32(4)
        a_load_k_vec = tid % fx.Int32(4)
        b_load_k = tid // fx.Int32(16)
        b_load_n_vec = tid % fx.Int32(16)

        # Per-wave MFMA fragment coords.
        a_row_top_l = wave_row * fx.Int32(32) + lane_row
        a_row_bot_l = wave_row * fx.Int32(32) + fx.Int32(16) + lane_row
        b_col_left_l = wave_col * fx.Int32(32) + lane_row
        b_col_right_l = wave_col * fx.Int32(32) + fx.Int32(16) + lane_row

        def _stage_load(k_tile_scalar, stage):
            """Cooperative HBM→LDS copy for one k-tile."""
            k_base = fx.Int32(k_tile_scalar * _MFMA_K)
            # A[m_base_wg:+64, k_base:+16]
            a_hbm_row = m_base_wg + a_load_row
            a_row_slice = fx.slice(A_buf, (a_hbm_row, None))
            a_div_v = fx.logical_divide(a_row_slice, h4_lay)
            a_hbm_vec_idx = fx.Int32(k_tile_scalar * 4) + a_load_k_vec
            a_chunk = _load_h4(a_div_v, a_hbm_vec_idx)
            for i in range_constexpr(4):
                s_A[stage].store(
                    a_chunk[i].ir_value(),
                    [_idx(a_load_row),
                     _idx(a_load_k_vec * fx.Int32(4) + fx.Int32(i))],
                )
            # B[k_base:+16, n_base_wg:+64]
            b_hbm_k = k_base + b_load_k
            b_k_slice = fx.slice(B_buf, (b_hbm_k, None))
            b_div_v = fx.logical_divide(b_k_slice, h4_lay)
            b_hbm_vec_idx = bid_n * fx.Int32(64 // 4) + b_load_n_vec
            b_chunk = _load_h4(b_div_v, b_hbm_vec_idx)
            for i in range_constexpr(4):
                s_B[stage].store(
                    b_chunk[i].ir_value(),
                    [_idx(b_load_k),
                     _idx(b_load_n_vec * fx.Int32(4) + fx.Int32(i))],
                )

        def _consume(stage, acc00, acc01, acc10, acc11):
            """MFMA compute on a fully-loaded LDS stage."""
            a_top_vals = []
            a_bot_vals = []
            for i in range_constexpr(_FRAG_A):
                k_in_lds = lane_k_group * fx.Int32(_FRAG_A) + fx.Int32(i)
                a_top_vals.append(s_A[stage].load([_idx(a_row_top_l), _idx(k_in_lds)]))
                a_bot_vals.append(s_A[stage].load([_idx(a_row_bot_l), _idx(k_in_lds)]))
            a_top = vector.from_elements(T.vec(_FRAG_A, in_elem_type), a_top_vals)
            a_bot = vector.from_elements(T.vec(_FRAG_A, in_elem_type), a_bot_vals)

            b_left_vals = []
            b_right_vals = []
            for i in range_constexpr(_FRAG_B):
                k_in_lds = lane_k_group * fx.Int32(_FRAG_B) + fx.Int32(i)
                b_left_vals.append(s_B[stage].load([_idx(k_in_lds), _idx(b_col_left_l)]))
                b_right_vals.append(s_B[stage].load([_idx(k_in_lds), _idx(b_col_right_l)]))
            b_left = vector.from_elements(T.vec(_FRAG_B, in_elem_type), b_left_vals)
            b_right = vector.from_elements(T.vec(_FRAG_B, in_elem_type), b_right_vals)

            def _mfma(a, b, acc):
                if dtype_str == "bf16":
                    a_i = vector.bitcast(T.vec(_FRAG_A, T.i16), a)
                    b_i = vector.bitcast(T.vec(_FRAG_B, T.i16), b)
                    return fx.rocdl.mfma_f32_16x16x16bf16_1k(
                        acc_ty, [a_i, b_i, acc, 0, 0, 0],
                    )
                return fx.rocdl.mfma_f32_16x16x16f16(acc_ty, [a, b, acc, 0, 0, 0])

            acc00 = _mfma(a_top, b_left, acc00)
            acc01 = _mfma(a_top, b_right, acc01)
            acc10 = _mfma(a_bot, b_left, acc10)
            acc11 = _mfma(a_bot, b_right, acc11)
            return acc00, acc01, acc10, acc11

        acc_ty = T.vec(_FRAG_C, T.f32)

        def _zero_acc():
            zs = []
            for _ in range_constexpr(_FRAG_C):
                zs.append(arith.constant(0.0, type=T.f32))
            return vector.from_elements(acc_ty, zs)

        acc00 = _zero_acc()
        acc01 = _zero_acc()
        acc10 = _zero_acc()
        acc11 = _zero_acc()

        k_tiles = K // _MFMA_K

        # --- Prologue: load tile 0 into stage 0 ---
        _stage_load(0, 0)
        _gpu.barrier()

        # --- Main loop: prefetch next while consuming current ---
        for k_tile in range_constexpr(k_tiles):
            cur = k_tile & 1
            if k_tile < k_tiles - 1:
                _stage_load(k_tile + 1, 1 - cur)
            acc00, acc01, acc10, acc11 = _consume(cur, acc00, acc01, acc10, acc11)
            _gpu.barrier()

        # --- Store C ---
        m_base = m_base_wg + wave_row * fx.Int32(32)
        n_base = n_base_wg + wave_col * fx.Int32(32)

        def _store_subtile(acc, row_base, col_base):
            for i in range_constexpr(_FRAG_C):
                out_row = row_base + lane_k_group * fx.Int32(_FRAG_C) + fx.Int32(i)
                out_col = col_base + lane_row
                row_c = fx.slice(C_buf, (out_row, None))
                c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))
                val_i = vector.extract(acc, static_position=[i], dynamic_position=[])
                _store_f(c_div, out_col, val_i)

        _store_subtile(acc00, m_base,                 n_base)
        _store_subtile(acc01, m_base,                 n_base + fx.Int32(16))
        _store_subtile(acc10, m_base + fx.Int32(16),  n_base)
        _store_subtile(acc11, m_base + fx.Int32(16),  n_base + fx.Int32(16))

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, C).launch(
            grid=(M // 64, N // 64, 1), block=(256, 1, 1), stream=stream,
        )

    return launch


_kernel_cache: dict = {}
_DTYPE2STR = {torch.float16: "f16", torch.bfloat16: "bf16"}


def _compile(M, N, K, dtype_str, arch):
    key = (M, N, K, dtype_str, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_64x64_4wave_lds_pp(M=M, N=N, K=K, dtype_str=dtype_str, arch=arch)
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_gemm_64x64_4wave_lds_pp_out",
    mutates_args=("out",),
    schema="(Tensor a, Tensor b, Tensor(a0!) out) -> ()",
)
def _gemm_64x64_4wave_lds_pp_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    assert a.is_cuda and b.is_cuda and out.is_cuda
    assert a.dtype in (torch.float16, torch.bfloat16) and a.dtype == b.dtype
    assert out.dtype == torch.float32
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and M % 64 == 0 and N % 64 == 0 and K % 16 == 0
    assert out.shape == (M, N)
    _compile(M, N, K, _DTYPE2STR[a.dtype], get_rocm_arch())(a, b, out)


@_gemm_64x64_4wave_lds_pp_out.register_fake
def _gemm_64x64_4wave_lds_pp_out_fake(a, b, out):
    return None


def gemm_64x64_4wave_lds_pp(A: Tensor, B: Tensor) -> Tensor:
    """4-wave 64×64 MFMA GEMM with cooperative cross-wave LDS sharing +
    2-stage ping-pong prefetch."""
    assert A.is_cuda and B.is_cuda
    assert A.dtype == B.dtype and A.dtype in (torch.float16, torch.bfloat16)
    M, K = A.shape
    _, N = B.shape
    out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    _gemm_64x64_4wave_lds_pp_out(A, B, out)
    return out


__all__ = ["gemm_64x64_4wave_lds_pp"]
