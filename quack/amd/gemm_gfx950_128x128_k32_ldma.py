# Copyright (c) 2026, AMD.

"""128×128 tile GEMM combining **K=32 MFMA + direct HBM→LDS DMA**.

Builds on the two wins from the K=16 family:
  - ``gemm_gfx950_128x128_k32.py``  → 2× MFMA throughput per op
  - ``gemm_gfx950_128x128_ldma.py`` → skip gmem→reg→LDS roundtrip

K=32 needs twice as much data per LDS stage (128×32 A, 32×128 B),
so each thread issues two ``buffer_load_dwordx4_lds`` per side per
stage (A: k=half*16..+16 in two vec8 chunks; B: 2 K-rows).

Hypothesis: the two wins compose — K=32's MFMA speedup × LDMA's
fewer-wait-classes speedup. Measured on MI355X, f16:

    shape    baseline   k32    ldma   k32_ldma   vs k16
    1024²    107        95.7   (k32 48)  ← k32 wins alone
    2048²    278        72     220       (TBD)     ← compose here
    4096²    412        366    340       (TBD)     ← compose here

Used together they may or may not compose multiplicatively — LDMA's
occupancy gain and K=32's MFMA rate both work against the same
``vmcnt + mfma_busy`` pipeline. Benchmark decides.
"""

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.rocdl as rocdl
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu as _gpu, range_constexpr, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import memref as memref_dialect

from quack.amd.flydsl_utils import get_rocm_arch


_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 32
_FRAG_A = 8
_FRAG_B = 8
_FRAG_C = 4
_WAVE_ROWS = 4
_WAVE_COLS = 4

_A_TILE_BYTES = 128 * 32 * 2   # 8 KiB per stage (f16)
_B_TILE_BYTES = 32 * 128 * 2

_DMA_BYTES = 16   # 8 f16 per load


def _align(x, a):
    return (x + a - 1) & ~(a - 1)


def _build_gemm_128x128_k32_ldma(*, M, N, K, dtype_str, arch):
    assert M % 128 == 0 and N % 128 == 0 and K % _MFMA_K == 0
    assert dtype_str in ("f16", "bf16")

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_128x128_k32_ldma_{dtype_str}_{M}_{N}_{K}_smem",
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
        elem_bytes = 2

        a_rsrc = buffer_ops.create_buffer_resource(A, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(B, max_size=True)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        base_ptr = allocator.get_base()
        s_A = [
            SmemPtr(base_ptr, a_off0, in_elem_type, shape=(128, 32)),
            SmemPtr(base_ptr, a_off1, in_elem_type, shape=(128, 32)),
        ]
        s_B = [
            SmemPtr(base_ptr, b_off0, in_elem_type, shape=(32, 128)),
            SmemPtr(base_ptr, b_off1, in_elem_type, shape=(32, 128)),
        ]
        for sp in (*s_A, *s_B):
            sp.get()

        a_lds_bases = []
        b_lds_bases = []
        for stage in range_constexpr(2):
            a_idx = memref_dialect.extract_aligned_pointer_as_index(s_A[stage].get())
            a_lds_bases.append(buffer_ops.create_llvm_ptr(
                arith.index_cast(T.i64, a_idx), address_space=3))
            b_idx = memref_dialect.extract_aligned_pointer_as_index(s_B[stage].get())
            b_lds_bases.append(buffer_ops.create_llvm_ptr(
                arith.index_cast(T.i64, b_idx), address_space=3))

        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

        def _store_f(div, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            ts = _vfull(1, Float32(val), Float32)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_f, r, fx.slice(div, (None, idx)))

        def _idx(i):
            return ArithValue(i).index_cast(T.index) if hasattr(i, "index_cast") else fx.Index(i)

        m_base_wg = bid_m * fx.Int32(128)
        n_base_wg = bid_n * fx.Int32(128)

        # For the DMA path, thread→(row, k_q) mapping must match the
        # auto-lane-indexing of ``buffer_load_lds`` (lane L → LDS byte L*16
        # within the wave's scalar base). For K=32 (128×32 A = 8 KiB, 32×128
        # B = 8 KiB) we emit 2 wave-calls per wave per side.
        #   A wave-call (1024 B = 16 rows × 64 B): lane L → row = L//4,
        #                                                    K-quarter = L%4
        #   B wave-call (1024 B = 4 rows × 256 B): lane L → row = L//16,
        #                                                    N-vec8   = L%16
        a_load_row_inwc = lane_in_wave // fx.Int32(4)   # 0..15
        a_load_k_quarter = lane_in_wave % fx.Int32(4)   # 0..3 → K=0/8/16/24
        b_load_k_inwc = lane_in_wave // fx.Int32(16)    # 0..3
        b_load_n_vec8 = lane_in_wave % fx.Int32(16)

        # These are still used by the consume path (register → MFMA), which
        # walks the per-lane MFMA fragment layout — independent of the DMA
        # thread mapping.

        def _a_row_in_tile(r):
            return wave_row * fx.Int32(64) + fx.Int32(r * 16) + lane_row

        def _b_col_in_tile(c):
            return wave_col * fx.Int32(64) + fx.Int32(c * 16) + lane_row

        acc_ty = T.vec(_FRAG_C, T.f32)
        vec8_a_ty = T.vec(_FRAG_A, in_elem_type)

        def _zero_acc():
            zs = []
            for _ in range_constexpr(_FRAG_C):
                zs.append(arith.constant(0.0, type=T.f32))
            return vector.from_elements(acc_ty, zs)

        accs = [[_zero_acc() for _ in range_constexpr(_WAVE_COLS)]
                for _ in range_constexpr(_WAVE_ROWS)]

        c_dma_bytes = arith.constant(_DMA_BYTES, type=T.i32)
        c_zero = arith.constant(0, type=T.i32)
        c_aux = arith.constant(0, type=T.i32)

        def _stage_load(k_tile_scalar, stage):
            k_byte_base = fx.Int32(k_tile_scalar * _MFMA_K * elem_bytes)

            # A tile 128×32: 4 waves × 2 calls = 8 wave-calls covering
            # 128 rows. Each wave-call handles 16 rows (each row is
            # fully loaded: K=32 = 4 K-quarters × 8 f16).
            # Wave W call C covers rows [W*32 + C*16, +16).
            for call in range_constexpr(2):
                a_row = m_base_wg + wave_id * fx.Int32(32) \
                    + fx.Int32(call * 16) + a_load_row_inwc
                a_gmem_byte = a_row * fx.Int32(K * elem_bytes) \
                    + k_byte_base \
                    + a_load_k_quarter * fx.Int32(_DMA_BYTES)
                a_gmem_byte_i32 = arith.index_cast(T.i32, a_gmem_byte) \
                    if hasattr(a_gmem_byte, "index_cast") else a_gmem_byte
                # Wave-scoped LDS base (SCALAR; readfirstlane ensures the
                # compiler treats it as such).
                a_wave_off_i32 = (wave_id * fx.Int32(2048) + fx.Int32(call * 1024))
                a_lds_ptr = buffer_ops.get_element_ptr(
                    a_lds_bases[stage], a_wave_off_i32,
                )
                rocdl.raw_ptr_buffer_load_lds(
                    a_rsrc, a_lds_ptr, c_dma_bytes, a_gmem_byte_i32,
                    c_zero, c_zero, c_aux,
                )

            # B tile 32×128: 4 waves × 2 calls = 8 wave-calls covering 32 rows.
            # Each wave-call handles 4 rows (each row is 128 cols × 2 B = 256 B,
            # split into 16 N-vec8 chunks).
            # Wave W call C covers rows [W*8 + C*4, +4).
            for call in range_constexpr(2):
                b_k = fx.Int32(k_tile_scalar * _MFMA_K) \
                    + wave_id * fx.Int32(8) + fx.Int32(call * 4) + b_load_k_inwc
                b_gmem_byte = b_k * fx.Int32(N * elem_bytes) \
                    + bid_n * fx.Int32(128 * elem_bytes) \
                    + b_load_n_vec8 * fx.Int32(_DMA_BYTES)
                b_gmem_byte_i32 = arith.index_cast(T.i32, b_gmem_byte) \
                    if hasattr(b_gmem_byte, "index_cast") else b_gmem_byte
                b_wave_off_i32 = (wave_id * fx.Int32(2048) + fx.Int32(call * 1024))
                b_lds_ptr = buffer_ops.get_element_ptr(
                    b_lds_bases[stage], b_wave_off_i32,
                )
                rocdl.raw_ptr_buffer_load_lds(
                    b_rsrc, b_lds_ptr, c_dma_bytes, b_gmem_byte_i32,
                    c_zero, c_zero, c_aux,
                )

        def _consume(stage, accs):
            a_frags = []
            for r in range_constexpr(_WAVE_ROWS):
                a_vals = []
                a_row_l = _a_row_in_tile(r)
                k_base = lane_k_group * fx.Int32(_FRAG_A)
                for i in range_constexpr(_FRAG_A):
                    a_vals.append(s_A[stage].load([_idx(a_row_l), _idx(k_base + fx.Int32(i))]))
                a_frags.append(vector.from_elements(vec8_a_ty, a_vals))

            b_frags = []
            for c in range_constexpr(_WAVE_COLS):
                b_vals = []
                b_col_l = _b_col_in_tile(c)
                k_base = lane_k_group * fx.Int32(_FRAG_B)
                for i in range_constexpr(_FRAG_B):
                    b_vals.append(s_B[stage].load([_idx(k_base + fx.Int32(i)), _idx(b_col_l)]))
                b_frags.append(vector.from_elements(T.vec(_FRAG_B, in_elem_type), b_vals))

            def _mfma(a, b, acc):
                if dtype_str == "bf16":
                    return fx.rocdl.mfma_f32_16x16x32_bf16(
                        acc_ty, [a, b, acc, 0, 0, 0],
                    )
                return fx.rocdl.mfma_f32_16x16x32_f16(acc_ty, [a, b, acc, 0, 0, 0])

            for r in range_constexpr(_WAVE_ROWS):
                for c in range_constexpr(_WAVE_COLS):
                    accs[r][c] = _mfma(a_frags[r], b_frags[c], accs[r][c])
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

        wave_m_base = m_base_wg + wave_row * fx.Int32(64)
        wave_n_base = n_base_wg + wave_col * fx.Int32(64)

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
            grid=(M // 128, N // 128, 1), block=(256, 1, 1), stream=stream,
        )

    return launch


_kernel_cache: dict = {}
_DTYPE2STR = {torch.float16: "f16", torch.bfloat16: "bf16"}


def _compile(M, N, K, dtype_str, arch):
    key = (M, N, K, dtype_str, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_128x128_k32_ldma(M=M, N=N, K=K, dtype_str=dtype_str, arch=arch)
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_gemm_128x128_k32_ldma_out",
    mutates_args=("out",),
    schema="(Tensor a, Tensor b, Tensor(a0!) out) -> ()",
)
def _gemm_128x128_k32_ldma_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    assert a.is_cuda and b.is_cuda and out.is_cuda
    assert a.dtype in (torch.float16, torch.bfloat16) and a.dtype == b.dtype
    assert out.dtype == torch.float32
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and M % 128 == 0 and N % 128 == 0 and K % 32 == 0
    assert out.shape == (M, N)
    _compile(M, N, K, _DTYPE2STR[a.dtype], get_rocm_arch())(a, b, out)


@_gemm_128x128_k32_ldma_out.register_fake
def _gemm_128x128_k32_ldma_out_fake(a, b, out):
    return None


def gemm_128x128_k32_ldma(A: Tensor, B: Tensor) -> Tensor:
    """128×128-tile K=32 MFMA GEMM with direct HBM→LDS async DMA.

    Requires M, N multiples of 128; K multiple of 32.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == B.dtype and A.dtype in (torch.float16, torch.bfloat16)
    M, K = A.shape
    _, N = B.shape
    out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    _gemm_128x128_k32_ldma_out(A, B, out)
    return out


__all__ = ["gemm_128x128_k32_ldma"]
