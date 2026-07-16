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
from flydsl.expr import arith, range_constexpr, vector, math as _fm
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


def _build_symmetric_16x16(
    *, M, K, dtype_str, out_dtype_str, arch,
    has_bias: bool = False,
    activation: Optional[str] = None,
    has_alpha: bool = False,
    has_c: bool = False,
):
    """Compile a 16x16-tile symmetric MFMA GEMM ``C = alpha*(A@A.T) + beta*Cin + bias``.

    Optional epilogue flags mirror ``gemm_gfx950._build_gemm_16x16`` (with
    ``N == M``): ``has_bias`` adds a per-column f32 bias, ``activation`` ∈
    {relu, relu_sq, gelu_tanh_approx, silu} applies post-bias, ``has_alpha``
    scales the accumulator by a runtime ``alpha``, and ``has_c`` adds
    ``beta * Cin[row, col]`` (``Cin`` same-shape f32).
    """
    assert dtype_str in {"f16", "bf16"}
    assert out_dtype_str in {"f32", "f16", "bf16"}
    assert activation in {None, "relu", "relu_sq", "gelu_tanh_approx", "silu"}
    assert M % _MFMA_M == 0 and K % _MFMA_K == 0
    ab_suffix = f"_{'a' if has_alpha else 'na'}{'c' if has_c else 'nc'}"
    sym_suffix = f"_{'b' if has_bias else 'nb'}_{activation or 'none'}{ab_suffix}"
    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_sym_{dtype_str}_{out_dtype_str}_{M}_{K}{sym_suffix}_smem",
    )

    @flyc.kernel
    def kernel(
        A: fx.Tensor,
        Bias: fx.Tensor, Cin: fx.Tensor,
        alpha: fx.Float32, beta: fx.Float32,
        C: fx.Tensor,
    ):
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
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        ca_out = fx.make_copy_atom(out_bufcopy, out_elem_type)
        h_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        out_reg_ty = fx.MemRefType.get(out_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        C_buf = fx.rocdl.make_buffer_tensor(C)
        if fx.const_expr(has_bias):
            Bias_buf = fx.rocdl.make_buffer_tensor(Bias)
            bias_div = fx.logical_divide(Bias_buf, fx.make_layout(1, 1))
        if fx.const_expr(has_c):
            Cin_buf = fx.rocdl.make_buffer_tensor(Cin)

        def _load_h(div, idx):
            r = fx.memref_alloca(h_reg_ty, reg_lay)
            fx.copy_atom_call(ca_h, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _load_f(div, idx):
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            fx.copy_atom_call(ca_f, fx.slice(div, (None, idx)), r)
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

        # Epilogue: optional alpha/beta*Cin + bias + activation, then store.
        # Bias is per-column (the output column j_base + lane_row is an M-index,
        # same shape/semantics as gemm_gfx950's (N,) bias with N=M).
        if fx.const_expr(has_bias):
            bias_val = ArithValue(_load_f(bias_div, j_base + lane_row))
        if fx.const_expr(has_alpha):
            alpha_av = ArithValue(alpha)
        if fx.const_expr(has_c):
            beta_av = ArithValue(beta)
        # Store C[i, j] at output row i_base + lane_k_group*4+i, col j_base + lane_row.
        for i in range_constexpr(_FRAG_C):
            out_row = i_base + lane_k_group * fx.Int32(_FRAG_C) + fx.Int32(i)
            out_col = j_base + lane_row
            row_c = fx.slice(C_buf, (out_row, None))
            c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))
            val_i = vector.extract(acc, static_position=[i], dynamic_position=[])
            val = ArithValue(val_i)
            if fx.const_expr(has_alpha):
                val = val * alpha_av
            if fx.const_expr(has_c):
                row_cin = fx.slice(Cin_buf, (out_row, None))
                cin_div = fx.logical_divide(row_cin, fx.make_layout(1, 1))
                cin_val = ArithValue(_load_f(cin_div, out_col))
                val = val + beta_av * cin_val
            if fx.const_expr(has_bias):
                val = val + bias_val
            zero = arith.constant(0.0, type=T.f32)
            if fx.const_expr(activation == "relu"):
                val = val.maximumf(zero)
            elif fx.const_expr(activation == "relu_sq"):
                val = val.maximumf(zero) * val
            elif fx.const_expr(activation == "gelu_tanh_approx"):
                import math as _py_math
                c1 = _py_math.sqrt(2.0 / _py_math.pi)
                c2 = 0.044715 * c1
                x = val
                x_sq = x * x
                tanh_arg = x * (c1 + c2 * x_sq)
                tanh_z = Float32(1.0) - Float32(2.0) / (
                    Float32(1.0) + _fm.exp(Float32(2.0) * tanh_arg, fastmath="fast")
                )
                val = x * (Float32(0.5) + Float32(0.5) * tanh_z)
            elif fx.const_expr(activation == "silu"):
                val = val / (Float32(1.0) + _fm.exp(-val, fastmath="fast"))
            _store_out(c_div, out_col, val)

    @flyc.jit
    def launch(A: fx.Tensor,
               Bias: fx.Tensor, Cin: fx.Tensor,
               alpha: fx.Float32, beta: fx.Float32,
               C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, Bias, Cin, alpha, beta, C).launch(
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
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: Optional[Tensor] = None,
) -> Tensor:
    """Symmetric GEMM: ``out = act(alpha * (A @ A.T) + beta * C + bias)``.

    - ``A``: ``(M, K)`` f16/bf16, K-contiguous.
    - ``out_dtype``: f32 (default), f16, bf16.
    - ``bias``: optional ``(M,)`` f32, added per output column.
    - ``activation``: optional {relu, relu_sq, gelu_tanh_approx, silu}.
    - ``alpha``: scales the ``A @ A.T`` accumulator (default 1.0).
    - ``beta`` / ``C``: optional ``(M, M)`` f32 residual added as ``beta * C``.
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
    assert activation in {None, "relu", "relu_sq", "gelu_tanh_approx", "silu"}
    has_bias = bias is not None
    has_c = C is not None
    # alpha!=1 needs the runtime-scaled path; beta only matters when C is given.
    has_alpha = has_c or (alpha != 1.0)
    dtype_str = "f16" if A.dtype == torch.float16 else "bf16"
    key = (M, K, dtype_str, _OUT_DTYPE_MAP[out_dtype], get_rocm_arch(),
           has_bias, activation, has_alpha, has_c)
    launcher = _kernel_cache.get(key)
    if fx.const_expr(launcher is None):
        launcher = _build_symmetric_16x16(
            M=M, K=K, dtype_str=dtype_str,
            out_dtype_str=_OUT_DTYPE_MAP[out_dtype], arch=get_rocm_arch(),
            has_bias=has_bias, activation=activation,
            has_alpha=has_alpha, has_c=has_c,
        )
        _kernel_cache[key] = launcher
    out = torch.empty(M, M, device=A.device, dtype=out_dtype)
    # Dummy 1-element tensors for disabled epilogue inputs (kernel never reads
    # them under the matching const_expr=False guard).
    dummy = A.new_empty(1, dtype=torch.float32)
    bias_t = bias if has_bias else dummy
    cin_t = C if has_c else dummy
    if has_bias:
        assert bias.shape == (M,) and bias.dtype == torch.float32
    if has_c:
        assert C.shape == (M, M) and C.dtype == torch.float32
    launcher(A, bias_t, cin_t, float(alpha), float(beta), out)
    return out


__all__ = ["gemm_symmetric"]
