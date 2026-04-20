# Copyright (c) 2026, AMD.

"""Standard-dtype MFMA GEMM for gfx950 — proof-of-life minimal kernel.

**Scope of this file today:** a minimum-viable FlyDSL MFMA kernel that
does ``C = A @ B`` for f16 inputs on a single-tile-per-workgroup grid
with a 16x16x16 MFMA. No LDS, no ping-pong, no preshuffle, no epilogue,
no split-K, no stream-K. Each workgroup is a single wave (64 threads)
that covers a 16x16 output tile by iterating K in 16-element chunks.

Why so minimal: writing a production MFMA GEMM end-to-end is ~1500 LoC
and weeks of tuning. This file proves the FlyDSL MFMA authoring path
works end-to-end against a PyTorch reference — the shared GEMM infra
and this kernel together form the foundation the full port sits on.

Layout reference (CDNA3/CDNA4 wave64 ``v_mfma_f32_16x16x16f16``):
    - A fragment (f16x4 per lane): ``A[lane % 16, (lane // 16)*4 + 0..3]``
    - B fragment (f16x4 per lane): ``B[(lane // 16)*4 + 0..3, lane % 16]``
    - C fragment (f32x4 per lane): ``C[(lane // 16)*4 + 0..3, lane % 16]``

Extensions (subsequent commits):
    - Larger tiles (128×128 via 4 MFMA waves per workgroup).
    - LDS ping-pong prefetch.
    - Epilogue (bias / activation / CShuffle).
    - Stream-K scheduling via ``quack/amd/tile_scheduler.py``.
    - bf16 / f32 dtype paths.
"""

from typing import Optional

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, range_constexpr, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32, Float16, BFloat16, Int32, Numeric
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch, torch2flydsl_dtype_map


# For f16 MFMA: v_mfma_f32_16x16x16f16
_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 16
_FRAG_A = 4  # f16 values per lane
_FRAG_B = 4
_FRAG_C = 4  # f32 values per lane


def _build_gemm_16x16_f16(*, M, N, K, arch):
    """Compile a 16x16-tile MFMA GEMM for ``M, N, K`` as compile-time constants.

    Requires M % 16 == 0, N % 16 == 0, K % 16 == 0. Grid = ``(M/16, N/16, 1)``.
    Block = (64, 1, 1) — one wave per tile.
    """
    assert M % _MFMA_M == 0 and N % _MFMA_N == 0 and K % _MFMA_K == 0
    allocator = SmemAllocator(None, arch=arch, global_sym_name="quack_amd_gemm_f16_smem")
    # No LDS usage in the MVP; SmemAllocator still needs finalize() to emit the
    # (empty) shared-memory symbol.

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y
        tid = fx.thread_idx.x  # 0..63

        lane_row = tid % fx.Int32(16)
        lane_k_group = tid // fx.Int32(16)  # 0..3
        # k_offset in the A/B fragment layout: each group of 16 lanes carries a
        # different K-slice within one MFMA call. For K=16, only group 0 touches
        # the matrix at all? No — all four groups carry complementary K values.

        # Buffer-backed tensors for raw buffer_load access.
        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        # Output tile offset in (M, N).
        m_base = bid_m * fx.Int32(_MFMA_M)
        n_base = bid_n * fx.Int32(_MFMA_N)

        # Per-lane A row / B col inside this tile.
        a_row = m_base + lane_row
        b_col = n_base + lane_row  # same lane_row index serves B's col

        # Scalar-load helper wrappers. We fall back to the copy_atom_call
        # machinery used elsewhere — less optimal than buffer_load but
        # reliably correct.
        ca_h = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), T.f16)
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        h_reg_ty = fx.MemRefType.get(T.f16, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

        def _load_h_scalar(div, idx):
            r = fx.memref_alloca(h_reg_ty, reg_lay)
            fx.copy_atom_call(ca_h, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _store_f_scalar(div, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            ts = _vfull(1, Float32(val), Float32)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_f, r, fx.slice(div, (None, idx)))

        # Accumulator: f32x4 per lane, zero-init.
        acc_ty = T.vec(_FRAG_C, T.f32)
        _zero_list = []
        for _ in range_constexpr(_FRAG_C):
            _zero_list.append(arith.constant(0.0, type=T.f32))
        acc = vector.from_elements(acc_ty, _zero_list)

        # Iterate over K tiles (each tile = 16 K elements → one MFMA call).
        k_tiles = K // _MFMA_K
        for k_tile in range_constexpr(k_tiles):
            k_tile_base = fx.Int32(k_tile * _MFMA_K)
            # This lane's K offset within the tile: group 0 → 0..3, group 1 → 4..7, ...
            lane_k_base = lane_k_group * fx.Int32(_FRAG_A) + k_tile_base

            # Load this lane's A fragment (f16x4): A[a_row, lane_k_base + 0..3]
            # A is row-major (M, K); row_a is the bid_m*16 + lane_row-th row of A.
            row_a = fx.slice(A_buf, (a_row, None))
            a_div = fx.logical_divide(row_a, fx.make_layout(1, 1))
            a_vals = []
            for i in range_constexpr(_FRAG_A):
                a_vals.append(_load_h_scalar(a_div, lane_k_base + fx.Int32(i)))
            a_frag = vector.from_elements(T.vec(_FRAG_A, T.f16), a_vals)

            # Load this lane's B fragment (f16x4): B[lane_k_base + 0..3, b_col]
            # B is row-major (K, N); we pick one row per k and the b_col-th column.
            b_vals = []
            for i in range_constexpr(_FRAG_B):
                row_b_k = fx.slice(B_buf, (lane_k_base + fx.Int32(i), None))
                b_div = fx.logical_divide(row_b_k, fx.make_layout(1, 1))
                b_vals.append(_load_h_scalar(b_div, b_col))
            b_frag = vector.from_elements(T.vec(_FRAG_B, T.f16), b_vals)

            # MFMA: acc_new = a_frag @ b_frag + acc
            acc = fx.rocdl.mfma_f32_16x16x16f16(acc_ty, [a_frag, b_frag, acc, 0, 0, 0])

        # Store C: 4 rows per lane at column `lane_row` in the output tile.
        # C[(bid_m*16) + (lane_k_group*4 + i), (bid_n*16) + lane_row] = acc[i]
        for i in range_constexpr(_FRAG_C):
            out_row = m_base + lane_k_group * fx.Int32(_FRAG_C) + fx.Int32(i)
            out_col = n_base + lane_row
            row_c = fx.slice(C_buf, (out_row, None))
            c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))
            val_i = vector.extract(acc, static_position=[i], dynamic_position=[])
            _store_f_scalar(c_div, out_col, val_i)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, C).launch(
            grid=(M // _MFMA_M, N // _MFMA_N, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    return launch


_kernel_cache: dict = {}


def _compile(M, N, K, arch):
    key = (M, N, K, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_16x16_f16(M=M, N=N, K=K, arch=arch)
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_gemm_f16_16x16_out",
    mutates_args=("out",),
    schema="(Tensor A, Tensor B, Tensor(a0!) out) -> ()",
)
def _gemm_f16_16x16_out(A: Tensor, B: Tensor, out: Tensor) -> None:
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    assert out.dtype == torch.float32
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert out.shape == (M, N)
    assert M % 16 == 0 and N % 16 == 0 and K % 16 == 0
    assert all(t.stride(-1) == 1 for t in (A, B, out))
    _compile(M, N, K, get_rocm_arch())(A, B, out)


@_gemm_f16_16x16_out.register_fake
def _gemm_f16_16x16_out_fake(A, B, out):
    return None


def gemm_f16_mfma(A: Tensor, B: Tensor) -> Tensor:
    """Proof-of-life FlyDSL MFMA GEMM: f16 × f16 → f32.

    Requires M, N, K all multiples of 16. One workgroup per 16x16 output
    tile. This is a minimal demonstration of the MFMA path; for real
    workloads use ``quack.amd.gemm`` (currently the torch fallback).
    """
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    _, N = B.shape
    out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    _gemm_f16_16x16_out(A, B, out)
    return out


__all__ = ["gemm_f16_mfma", "_gemm_f16_16x16_out"]
