# Copyright (c) 2026, AMD.

"""Symmetric MFMA GEMM: C = A @ A.T (inner product of rows).

Computes ``C[i, j] = sum_k A[i, k] * A[j, k]`` where A is ``(M, K)``
row-major. Output is ``(M, M)``.

Instead of letting a generic GEMM compute (M, K) × (K, M), which would
need a materialised A.T, this kernel reads A twice with different row
indices — once as the "A fragment" (row = lane_row in M) and once as
the "B fragment" (col = lane_row in M). Both reads go to the same
row-major A tensor directly, no transpose copy needed.

Scope (MVP):
    - f16 / bf16 input, f32 / f16 / bf16 output
    - M, K multiples of 16
    - No bias / activation / alpha / beta / C (basic symmetric GEMM only)
    - Full (M, M) output — triangular-only output (skip j > i) is a
      separate optimisation and not currently exposed.
"""

from typing import Optional

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, range_constexpr, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32, Numeric
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch


_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 16
_FRAG_C = 4


def _build_symmetric_16x16(*, M, K, dtype_str, out_dtype_str, arch):
    assert dtype_str in {"f16", "bf16"}
    assert out_dtype_str in {"f32", "f16", "bf16"}
    assert M % _MFMA_M == 0 and K % _MFMA_K == 0
    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_sym_{dtype_str}_{out_dtype_str}_{M}_{K}_smem",
    )

    @flyc.kernel
    def kernel(A: fx.Tensor, C: fx.Tensor):
        # Grid = (M/16, M/16, 1) — one workgroup per 16×16 output tile.
        bid_i = fx.block_idx.x  # "A row" tile index
        bid_j = fx.block_idx.y  # "B row" tile index (also a row of A)
        tid = fx.thread_idx.x
        lane_row = tid % fx.Int32(16)
        lane_k_group = tid // fx.Int32(16)

        in_elem_type = T.f16 if dtype_str == "f16" else T.bf16
        out_elem_type = (
            T.f32 if out_dtype_str == "f32"
            else T.f16 if out_dtype_str == "f16"
            else T.bf16
        )
        out_bufcopy = (
            fx.rocdl.BufferCopy32b() if out_dtype_str == "f32"
            else fx.rocdl.BufferCopy16b()
        )
        ca_h = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), in_elem_type)
        ca_out = fx.make_copy_atom(out_bufcopy, out_elem_type)
        h_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        out_reg_ty = fx.MemRefType.get(out_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        def _load_h(div, idx):
            r = fx.memref_alloca(h_reg_ty, reg_lay)
            fx.copy_atom_call(ca_h, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _store_out(div, idx, val_f32):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(out_reg_ty, reg_lay)
            elem_py = Numeric.from_ir_type(out_reg_ty.element_type)
            if fx.const_expr(out_dtype_str == "f32"):
                ts = _vfull(1, Float32(val_f32), Float32)
            else:
                val = ArithValue(val_f32).truncf(out_elem_type)
                ts = _vfull(1, elem_py(val), elem_py)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_out, r, fx.slice(div, (None, idx)))

        acc_ty = T.vec(_FRAG_C, T.f32)
        zeros = []
        for _ in range_constexpr(_FRAG_C):
            zeros.append(arith.constant(0.0, type=T.f32))
        acc = vector.from_elements(acc_ty, zeros)

        i_base = bid_i * fx.Int32(_MFMA_M)
        j_base = bid_j * fx.Int32(_MFMA_N)

        # Per-lane A-row (for A fragment) and A-row (for "B fragment" = A[j]).
        a_row_i = i_base + lane_row   # row index into A for the A fragment
        a_row_j = j_base + lane_row   # row index into A for the "B" fragment

        k_tiles = K // _MFMA_K
        for k_tile in range_constexpr(k_tiles):
            k_tile_base = fx.Int32(k_tile * _MFMA_K)
            lane_k_base = lane_k_group * fx.Int32(_FRAG_C) + k_tile_base

            # Load A fragment — same pattern as gemm_gfx950.
            row_ai = fx.slice(A_buf, (a_row_i, None))
            ai_div = fx.logical_divide(row_ai, fx.make_layout(1, 1))
            a_vals = []
            for i in range_constexpr(_FRAG_C):
                a_vals.append(_load_h(ai_div, lane_k_base + fx.Int32(i)))
            a_frag = vector.from_elements(T.vec(_FRAG_C, in_elem_type), a_vals)

            # Load "B fragment" from A[a_row_j, ...] — same lane_k_base as A.
            # This matches the (N, K) layout that the blockscaled kernel uses:
            # per-lane 4 K-consecutive values at row = lane_row.
            row_aj = fx.slice(A_buf, (a_row_j, None))
            aj_div = fx.logical_divide(row_aj, fx.make_layout(1, 1))
            b_vals = []
            for i in range_constexpr(_FRAG_C):
                b_vals.append(_load_h(aj_div, lane_k_base + fx.Int32(i)))
            b_frag = vector.from_elements(T.vec(_FRAG_C, in_elem_type), b_vals)

            if fx.const_expr(dtype_str == "bf16"):
                a_i16 = vector.bitcast(T.vec(_FRAG_C, T.i16), a_frag)
                b_i16 = vector.bitcast(T.vec(_FRAG_C, T.i16), b_frag)
                acc = fx.rocdl.mfma_f32_16x16x16bf16_1k(
                    acc_ty, [a_i16, b_i16, acc, 0, 0, 0],
                )
            else:
                acc = fx.rocdl.mfma_f32_16x16x16f16(
                    acc_ty, [a_frag, b_frag, acc, 0, 0, 0],
                )

        # Store C[i, j] at output row i_base + lane_k_group*4+i, col j_base + lane_row.
        for i in range_constexpr(_FRAG_C):
            out_row = i_base + lane_k_group * fx.Int32(_FRAG_C) + fx.Int32(i)
            out_col = j_base + lane_row
            row_c = fx.slice(C_buf, (out_row, None))
            c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))
            val_i = vector.extract(acc, static_position=[i], dynamic_position=[])
            _store_out(c_div, out_col, val_i)

    @flyc.jit
    def launch(A: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, C).launch(
            grid=(M // _MFMA_M, M // _MFMA_N, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    return launch


_kernel_cache: dict = {}

_OUT_DTYPE_MAP = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}


def gemm_symmetric(
    A: Tensor,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Symmetric GEMM: ``C = A @ A.T``.

    - ``A``: ``(M, K)`` f16/bf16, K-contiguous.
    - ``out_dtype``: f32 (default), f16, bf16.
    - Output: ``(M, M)``.
    - M, K multiples of 16.
    """
    assert A.is_cuda
    assert A.dtype in (torch.float16, torch.bfloat16)
    assert A.stride(-1) == 1
    if fx.const_expr(out_dtype is None):
        out_dtype = torch.float32
    M, K = A.shape
    assert M % 16 == 0 and K % 16 == 0
    dtype_str = "f16" if A.dtype == torch.float16 else "bf16"
    key = (M, K, dtype_str, _OUT_DTYPE_MAP[out_dtype], get_rocm_arch())
    launcher = _kernel_cache.get(key)
    if fx.const_expr(launcher is None):
        launcher = _build_symmetric_16x16(
            M=M, K=K, dtype_str=dtype_str,
            out_dtype_str=_OUT_DTYPE_MAP[out_dtype], arch=get_rocm_arch(),
        )
        _kernel_cache[key] = launcher
    out = torch.empty(M, M, device=A.device, dtype=out_dtype)
    launcher(A, out)
    return out


__all__ = ["gemm_symmetric"]
