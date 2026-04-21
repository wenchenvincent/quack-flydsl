# Copyright (c) 2026, AMD.

"""Larger-tile MFMA GEMM for gfx950 — 32×32 tile, single-wave, 2×2 MFMA grid.

Each workgroup covers a 32×32 output tile by issuing four 16×16×16 MFMA
instructions. The key perf win over the 16×16 path in
``gemm_gfx950.py`` is **in-register fragment reuse**:

    MFMA(0,0): uses a_top, b_left
    MFMA(0,1): uses a_top, b_right   ← a_top reused
    MFMA(1,0): uses a_bot, b_left    ← b_left reused
    MFMA(1,1): uses a_bot, b_right   ← both reused

Each A fragment is loaded twice per k-tile (top / bottom row halves)
instead of four times if the four MFMAs were independent — same for B.
Concrete on a 4-K-tile loop: 2 × 4 = 8 A-loads (16 f16) instead of
4 × 4 = 16 A-loads (32 f16), a 2× reduction in HBM traffic per flop.

Scope (MVP):
  - f16 × f16 → f32 only (bf16 + epilogue are a follow-up).
  - M, N multiples of 32; K multiple of 16.
  - Single wave (64 threads) per workgroup.
  - No LDS ping-pong yet (future: add smem-staged A/B via double buffer).

Deferred perf work (separate commit tracks):
  - **LDS ping-pong** (2-stage): prefetch K+1 tile while MFMA-ing K.
    Needs ``SmemAllocator`` double-buffer + explicit wait-counts.
    Reference: ``FlyDSL/kernels/mfma_preshuffle_pipeline.py`` lines
    200–380.
  - **B-preshuffle** (XOR-swizzled LDS B layout): eliminates LDS bank
    conflicts when reading B for MFMA. Reference:
    ``FlyDSL/kernels/mfma_preshuffle_pipeline.py:swizzle_xor16`` +
    ``lds_store_16b_xor16``. Combined with LDS ping-pong this is the
    path that closes the gap vs hipBLASLt's MFMA performance on gfx950.
  - **4-wave workgroup** (256 threads, 4×4 MFMA grid covering a 128×128
    tile): biggest single win for large shapes. Requires wave-indexed
    fragment loads. Reference: ``preshuffle_gemm.py`` lines 434–459.

Public API: ``gemm_f16_32x32(A, B) -> C`` — used by ``quack.amd.gemm``
when inputs align to the 32-tile grid and epilogue is empty; otherwise
falls through to the 16×16 path.
"""


import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, range_constexpr, vector
from flydsl.expr.numeric import Float32
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch


_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 16
_FRAG_A = 4
_FRAG_B = 4
_FRAG_C = 4


def _build_gemm_32x32(*, M, N, K, dtype_str, arch):
    """32×32 MFMA GEMM kernel. ``dtype_str`` ∈ {'f16', 'bf16'}."""
    assert M % 32 == 0 and N % 32 == 0 and K % _MFMA_K == 0
    assert dtype_str in ("f16", "bf16")

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_32x32_{dtype_str}_{M}_{N}_{K}_smem",
    )

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y
        tid = fx.thread_idx.x

        lane_row = tid % fx.Int32(16)
        lane_k_group = tid // fx.Int32(16)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        in_elem_type = T.f16 if dtype_str == "f16" else T.bf16
        # Vectorized 64-bit loads for the A/B MFMA fragments: each
        # BufferCopy64b pulls in exactly the 4 f16/bf16 elements a lane
        # needs per A-fragment, replacing four scalar BufferCopy16b
        # issues. B-fragments can't be vectorized the same way because
        # B rows (K dim) aren't contiguous in HBM — the 4 K values live
        # on 4 different rows, so they still go through scalar loads.
        ca_h4 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), in_elem_type)
        ca_h_scalar = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), in_elem_type)
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        h4_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(4, 1), fx.AddressSpace.Register)
        h_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        h4_lay = fx.make_layout(4, 1)
        reg_lay = fx.make_layout(1, 1)

        def _load_h4(div_vec, vec_idx):
            """Load 4 consecutive f16/bf16 elements via buffer_load_b64."""
            r = fx.memref_alloca(h4_reg_ty, h4_lay)
            fx.copy_atom_call(ca_h4, fx.slice(div_vec, (None, vec_idx)), r)
            return fx.memref_load_vec(r)

        def _load_h_scalar(div, idx):
            r = fx.memref_alloca(h_reg_ty, reg_lay)
            fx.copy_atom_call(ca_h_scalar, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _store_f(div, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            ts = _vfull(1, Float32(val), Float32)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_f, r, fx.slice(div, (None, idx)))

        m_base = bid_m * fx.Int32(32)
        n_base = bid_n * fx.Int32(32)

        # Each lane participates in 4 MFMAs; its row-in-16 (lane_row) and
        # col-in-16 (lane_row) stay the same, but the effective A row / B
        # col shifts by 16 for the bottom / right half tiles.
        a_row_top = m_base + lane_row
        a_row_bot = m_base + fx.Int32(16) + lane_row
        b_col_left = n_base + lane_row
        b_col_right = n_base + fx.Int32(16) + lane_row

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

        # Precompute vec4-divided row tensors once — per k-tile we only
        # shift the vec index.
        row_a_top = fx.slice(A_buf, (a_row_top, None))
        row_a_bot = fx.slice(A_buf, (a_row_bot, None))
        a_div_top_v = fx.logical_divide(row_a_top, h4_lay)
        a_div_bot_v = fx.logical_divide(row_a_bot, h4_lay)

        for k_tile in range_constexpr(k_tiles):
            # vec4 K-index for this lane: lane_k_group ∈ [0,4), plus k_tile*4.
            vec_idx_A = lane_k_group + fx.Int32(k_tile * 4)

            # A fragments: one BufferCopy64b per fragment, 4 elements each.
            a_top = _load_h4(a_div_top_v, vec_idx_A)
            a_bot = _load_h4(a_div_bot_v, vec_idx_A)

            # B: per-k-row slice. B[lane_k_base + i, b_col] for i ∈ [0,4).
            # Each of the 4 K-rows is a separate row of B, so we still do
            # 4 scalar loads per B fragment — the Ks aren't contiguous in
            # HBM. (B in (K, N) layout: each K row-stride apart.) Issue
            # them as 4 × BufferCopy16b but load 2 columns from each row
            # per call via... actually B rows are the unit; keep scalar.
            k_tile_base_i32 = fx.Int32(k_tile * _MFMA_K)
            lane_k_base = lane_k_group * fx.Int32(_FRAG_B) + k_tile_base_i32
            b_left_vals = []
            b_right_vals = []
            for i in range_constexpr(_FRAG_B):
                row_b_k = fx.slice(B_buf, (lane_k_base + fx.Int32(i), None))
                b_div = fx.logical_divide(row_b_k, reg_lay)
                b_left_vals.append(_load_h_scalar(b_div, b_col_left))
                b_right_vals.append(_load_h_scalar(b_div, b_col_right))
            b_left = vector.from_elements(T.vec(_FRAG_B, in_elem_type), b_left_vals)
            b_right = vector.from_elements(T.vec(_FRAG_B, in_elem_type), b_right_vals)

            # --- 4 MFMAs reusing the fragments ---
            def _mfma(a, b, acc):
                if dtype_str == "bf16":
                    a_i = vector.bitcast(T.vec(_FRAG_A, T.i16), a)
                    b_i = vector.bitcast(T.vec(_FRAG_B, T.i16), b)
                    return fx.rocdl.mfma_f32_16x16x16bf16_1k(
                        acc_ty, [a_i, b_i, acc, 0, 0, 0],
                    )
                return fx.rocdl.mfma_f32_16x16x16f16(
                    acc_ty, [a, b, acc, 0, 0, 0],
                )

            acc00 = _mfma(a_top, b_left, acc00)
            acc01 = _mfma(a_top, b_right, acc01)
            acc10 = _mfma(a_bot, b_left, acc10)
            acc11 = _mfma(a_bot, b_right, acc11)

        # --- Write back four 16×16 sub-tiles ---
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
    def launch(
        A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, C).launch(
            grid=(M // 32, N // 32, 1), block=(64, 1, 1), stream=stream,
        )

    return launch


_kernel_cache: dict = {}

_DTYPE2STR = {torch.float16: "f16", torch.bfloat16: "bf16"}


def _compile(M, N, K, dtype_str, arch):
    key = (M, N, K, dtype_str, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_32x32(M=M, N=N, K=K, dtype_str=dtype_str, arch=arch)
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_gemm_32x32_out",
    mutates_args=("out",),
    schema="(Tensor a, Tensor b, Tensor(a0!) out) -> ()",
)
def _gemm_32x32_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    assert a.is_cuda and b.is_cuda and out.is_cuda
    assert a.dtype in (torch.float16, torch.bfloat16) and a.dtype == b.dtype
    assert out.dtype == torch.float32
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and M % 32 == 0 and N % 32 == 0 and K % 16 == 0
    assert out.shape == (M, N)
    _compile(M, N, K, _DTYPE2STR[a.dtype], get_rocm_arch())(a, b, out)


@_gemm_32x32_out.register_fake
def _gemm_32x32_out_fake(a, b, out):
    return None


def gemm_32x32(A: Tensor, B: Tensor) -> Tensor:
    """f16/bf16 × f16/bf16 → f32 MFMA GEMM with 32×32 output tiles.

    Requires M, N multiples of 32 and K a multiple of 16. A and B must
    have matching dtypes.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == B.dtype and A.dtype in (torch.float16, torch.bfloat16)
    M, K = A.shape
    _, N = B.shape
    out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    _gemm_32x32_out(A, B, out)
    return out


# Backwards-compatible alias.
def gemm_f16_32x32(A: Tensor, B: Tensor) -> Tensor:
    return gemm_32x32(A, B)


__all__ = ["gemm_32x32", "gemm_f16_32x32"]
