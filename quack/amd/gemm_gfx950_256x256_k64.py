# Copyright (c) 2026, AMD.

"""256×256×K=64 tile MFMA GEMM with inner K-unroll — EXPERIMENTAL.

Intended as the fix for the 256×256-K=16 kernel's hidden bottleneck
(LDS latency serialized before MFMAs). The hope was that inlining 4
K-sub-slices worth of loads + MFMAs per macroiter would let the
compiler reorder load-of-slice-(ks+1) ahead of MFMA-of-slice-ks,
hiding LDS latency behind compute.

**Measured: did NOT work. Got dramatically slower.**

    shape       128×128   256_K16    256_K64    K64/K16    K64/hipBLASLt
    256²        18 μs     34 μs      **210 μs**  0.16×      26.6×
    1024²       59 μs     101 μs     **306 μs**  0.33×      23.6×
    4096²       364 μs    458 μs     **760 μs**  0.60×      6.7×

Compile time alone was 3+ minutes — MLIR code explosion from inlining
4× more operations per macroiter (256 MFMAs + 32 A-frag + 32 B-frag
loads inlined per consume).

**Why it doesn't work**:

1. **No automatic MFMA/LDS interleave**. FlyDSL/MLIR preserves source
   order — the compiler doesn't reorder loads ahead of dependent
   MFMAs on its own. Would need explicit ``rocdl.sched_vmem`` /
   ``sched_mfma`` directives for hipBLASLt-style interleaving.
2. **Register pressure**. 4× the fragment staging (64 A + 64 B =
   128 f16 registers across 4 slices) on top of 256 VGPR accumulators
   stresses register allocation — likely spills or serializes.
3. **ICache pressure**. The fully-unrolled macro body (256 MFMAs + 64
   LDS loads + cooperative stores) inlined 64× (K=4096/64) explodes
   beyond the 32 KiB ICache.

The correct fix for LDS latency hiding is **explicit MFMA scheduling
directives**, not compiler-trusted source-order unroll. Kept in-tree
as evidence of what doesn't work. Not wired into the autotune
dispatcher.

LDS per stage:
    A: (256, 64) f16  = 32 KiB
    B: (64, 256) f16  = 32 KiB
Total (ping-pong): 128 KiB per WG — within MI355X's 160 KiB budget.

Cooperative HBM load per macro-iter:
    A: 16384 f16 / 256 threads = 64 f16/thread = 8 × BufferCopy128b
       thread tid → row=tid, 8 vec8 chunks (k_vec8 ∈ [0, 8))
    B: 16384 f16 / 256 threads = 64 f16/thread = 8 × BufferCopy128b
       thread tid → linear covers 8 of 2048 vec8 positions

Per-lane register footprint: 64 accumulators (256 f32 VGPRs) as
before. Fragment staging adds 16 f16 registers per slice × 4 slices
if the compiler decides to prefetch them — the goal is exactly that
inter-slice reordering.

Scope (MVP): f16 × f16 → f32; M, N multiples of 256, K multiple of 64.
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
_K_INNER = 4           # 4 K-sub-slices per macro-iter (64 K total)

_A_TILE_BYTES = 256 * 64 * 2
_B_TILE_BYTES = 64 * 256 * 2


def _align(x, a):
    return (x + a - 1) & ~(a - 1)


def _build_gemm_256x256_k64(*, M, N, K, dtype_str, arch):
    assert M % 256 == 0 and N % 256 == 0 and K % 64 == 0
    assert dtype_str in ("f16", "bf16")

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_256x256_k64_{dtype_str}_{M}_{N}_{K}_smem",
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
            SmemPtr(base_ptr, a_off0, in_elem_type, shape=(256, 64)),
            SmemPtr(base_ptr, a_off1, in_elem_type, shape=(256, 64)),
        ]
        s_B = [
            SmemPtr(base_ptr, b_off0, in_elem_type, shape=(64, 256)),
            SmemPtr(base_ptr, b_off1, in_elem_type, shape=(64, 256)),
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

        def _stage_load(k_macro_scalar, stage):
            """Cooperative HBM→LDS of a (256 rows × 64 K) A-tile and a
            (64 K × 256 cols) B-tile. k_macro indexes macroiterations."""
            k_base = fx.Int32(k_macro_scalar * 64)
            # --- A: 8 vec8 per thread, all on same row=tid ---
            a_hbm_row = m_base_wg + tid
            a_row_slice = fx.slice(A_buf, (a_hbm_row, None))
            a_div_v = fx.logical_divide(a_row_slice, h8_lay)
            for k_vec in range_constexpr(8):
                a_hbm_vec_idx = fx.Int32(k_macro_scalar * 8) + fx.Int32(k_vec)
                a_chunk = _load_h8(a_div_v, a_hbm_vec_idx)
                for i in range_constexpr(8):
                    s_A[stage].store(
                        a_chunk[i].ir_value(),
                        [_idx(tid),
                         _idx(fx.Int32(k_vec * 8) + fx.Int32(i))],
                    )
            # --- B: 8 vec8 per thread via linear positioning in (64, 32) grid ---
            for pos_off in range_constexpr(8):
                linear_pos = tid * fx.Int32(8) + fx.Int32(pos_off)
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
            """Inner consume: 4 K-sub-slices × 64 MFMAs each = 256
            MFMAs per lane per macroiter. All 4 slices' loads +
            MFMAs are inlined here so the compiler can interleave."""
            for ks in range_constexpr(_K_INNER):
                k_off = fx.Int32(ks * 16)
                # A fragments at K sub-slice `ks`.
                a_frags = []
                for r in range_constexpr(_WAVE_ROWS):
                    a_vals = []
                    a_row_l = _a_row_in_tile(r)
                    for i in range_constexpr(_FRAG_A):
                        k_in_lds = k_off + lane_k_group * fx.Int32(_FRAG_A) + fx.Int32(i)
                        a_vals.append(s_A[stage].load([_idx(a_row_l), _idx(k_in_lds)]))
                    a_frags.append(vector.from_elements(T.vec(_FRAG_A, in_elem_type), a_vals))

                # B fragments at K sub-slice `ks`.
                b_frags = []
                for c in range_constexpr(_WAVE_COLS):
                    b_vals = []
                    b_col_l = _b_col_in_tile(c)
                    for i in range_constexpr(_FRAG_B):
                        k_in_lds = k_off + lane_k_group * fx.Int32(_FRAG_B) + fx.Int32(i)
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

                for r in range_constexpr(_WAVE_ROWS):
                    for c in range_constexpr(_WAVE_COLS):
                        accs[r][c] = _mfma(a_frags[r], b_frags[c], accs[r][c])
            return accs

        k_macros = K // 64

        _stage_load(0, 0)
        _gpu.barrier()

        for k_macro in range_constexpr(k_macros):
            cur = k_macro & 1
            if k_macro < k_macros - 1:
                _stage_load(k_macro + 1, 1 - cur)
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
        got = _build_gemm_256x256_k64(M=M, N=N, K=K, dtype_str=dtype_str, arch=arch)
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_gemm_256x256_k64_out",
    mutates_args=("out",),
    schema="(Tensor a, Tensor b, Tensor(a0!) out) -> ()",
)
def _gemm_256x256_k64_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    assert a.is_cuda and b.is_cuda and out.is_cuda
    assert a.dtype in (torch.float16, torch.bfloat16) and a.dtype == b.dtype
    assert out.dtype == torch.float32
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and M % 256 == 0 and N % 256 == 0 and K % 64 == 0
    assert out.shape == (M, N)
    _compile(M, N, K, _DTYPE2STR[a.dtype], get_rocm_arch())(a, b, out)


@_gemm_256x256_k64_out.register_fake
def _gemm_256x256_k64_out_fake(a, b, out):
    return None


def gemm_256x256_k64(A: Tensor, B: Tensor) -> Tensor:
    """256×256×64 tile MFMA GEMM with inner K-unroll.

    Requires M, N multiples of 256; K multiple of 64.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == B.dtype and A.dtype in (torch.float16, torch.bfloat16)
    M, K = A.shape
    _, N = B.shape
    out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    _gemm_256x256_k64_out(A, B, out)
    return out


__all__ = ["gemm_256x256_k64"]
