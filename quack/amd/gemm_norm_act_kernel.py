# Copyright (c) 2026, AMD.

"""Fused norm-scale + activation MFMA GEMM for gfx950.

Computes ``out = activation((A @ B + bias) * colvec[row] * rowvec[col])``
in one kernel. ``colvec`` is typically ``rstd`` from a prior norm
(per-row scale, shape ``(M,)``); ``rowvec`` is a learned per-column
weight (shape ``(N,)``). Both are optional.

The epilogue sequence per-lane:
    acc_f32 = A @ B            (MFMA accumulator)
    + bias[col]                (f32, per-column, optional)
    * colvec[row]              (f32, per-row, optional)
    * rowvec[col]              (f32, per-column, optional)
    activation(.)              (optional)
    truncate to out_dtype
    store

Scope: 16×16 single-wave tile, f16 × f16 inputs, f32/f16/bf16 output.
M, N, K multiples of 16. Activations:
    None / relu / relu_sq / silu / gelu_tanh_approx.

Saves one HBM round-trip vs the torch-composed
``quack.amd.gemm.gemm_norm_act`` path — the colvec/rowvec multiplies
happen in the MFMA epilogue register before the output store.
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
_FRAG_A = 4
_FRAG_B = 4
_FRAG_C = 4

_DTYPE2STR = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32"}


def _build_gemm_norm_act_f16(
    *, M, N, K, activation, out_dtype_str, has_bias, has_colvec, has_rowvec, arch,
):
    assert M % _MFMA_M == 0 and N % _MFMA_N == 0 and K % _MFMA_K == 0
    assert activation in (None, "relu", "relu_sq", "silu", "gelu_tanh_approx")
    assert out_dtype_str in ("f32", "f16", "bf16")

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=(
            f"quack_amd_gemm_norm_act_{activation or 'none'}"
            f"_out{out_dtype_str}"
            f"_{'b' if has_bias else 'nb'}"
            f"_{'c' if has_colvec else 'nc'}"
            f"_{'r' if has_rowvec else 'nr'}"
            f"_{M}_{N}_{K}_smem"
        ),
    )

    @flyc.kernel
    def kernel(
        A: fx.Tensor, B: fx.Tensor,
        Bias: fx.Tensor, Colvec: fx.Tensor, Rowvec: fx.Tensor,
        Out: fx.Tensor,
    ):
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y
        tid = fx.thread_idx.x

        lane_row = tid % fx.Int32(16)
        lane_k_group = tid // fx.Int32(16)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        Out_buf = fx.rocdl.make_buffer_tensor(Out)
        if has_bias:
            Bias_buf = fx.rocdl.make_buffer_tensor(Bias)
            bias_div = fx.logical_divide(Bias_buf, fx.make_layout(1, 1))
        if has_colvec:
            Colvec_buf = fx.rocdl.make_buffer_tensor(Colvec)
            colvec_div = fx.logical_divide(Colvec_buf, fx.make_layout(1, 1))
        if has_rowvec:
            Rowvec_buf = fx.rocdl.make_buffer_tensor(Rowvec)
            rowvec_div = fx.logical_divide(Rowvec_buf, fx.make_layout(1, 1))

        in_elem_type = T.f16
        out_elem_type = {"f32": T.f32, "f16": T.f16, "bf16": T.bf16}[out_dtype_str]

        ca_h = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), in_elem_type)
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        ca_out = fx.make_copy_atom(
            fx.rocdl.BufferCopy32b() if out_dtype_str == "f32" else fx.rocdl.BufferCopy16b(),
            out_elem_type,
        )
        h_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        out_reg_ty = fx.MemRefType.get(out_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

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
            if out_dtype_str == "f32":
                ts = _vfull(1, Float32(val_f32), Float32)
            else:
                val = ArithValue(val_f32).truncf(out_elem_type)
                ts = _vfull(1, elem_py(val), elem_py)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_out, r, fx.slice(div, (None, idx)))

        m_base = bid_m * fx.Int32(_MFMA_M)
        n_base = bid_n * fx.Int32(_MFMA_N)
        a_row = m_base + lane_row
        b_col = n_base + lane_row

        acc_ty = T.vec(_FRAG_C, T.f32)
        _zero_list = []
        for _ in range_constexpr(_FRAG_C):
            _zero_list.append(arith.constant(0.0, type=T.f32))
        acc = vector.from_elements(acc_ty, _zero_list)

        k_tiles = K // _MFMA_K
        for k_tile in range_constexpr(k_tiles):
            k_tile_base = fx.Int32(k_tile * _MFMA_K)
            lane_k_base = lane_k_group * fx.Int32(_FRAG_A) + k_tile_base

            row_a = fx.slice(A_buf, (a_row, None))
            a_div = fx.logical_divide(row_a, fx.make_layout(1, 1))
            a_vals = []
            for i in range_constexpr(_FRAG_A):
                a_vals.append(_load_h(a_div, lane_k_base + fx.Int32(i)))
            a_frag = vector.from_elements(T.vec(_FRAG_A, in_elem_type), a_vals)

            b_vals = []
            for i in range_constexpr(_FRAG_B):
                row_b_k = fx.slice(B_buf, (lane_k_base + fx.Int32(i), None))
                b_div = fx.logical_divide(row_b_k, fx.make_layout(1, 1))
                b_vals.append(_load_h(b_div, b_col))
            b_frag = vector.from_elements(T.vec(_FRAG_B, in_elem_type), b_vals)

            acc = fx.rocdl.mfma_f32_16x16x16f16(
                acc_ty, [a_frag, b_frag, acc, 0, 0, 0],
            )

        # Epilogue: bias → colvec × rowvec → activation.
        out_col = n_base + lane_row
        if has_bias:
            bias_val = ArithValue(_load_f(bias_div, out_col))
        if has_rowvec:
            rowvec_val = ArithValue(_load_f(rowvec_div, out_col))

        for i in range_constexpr(_FRAG_C):
            out_row = m_base + lane_k_group * fx.Int32(_FRAG_C) + fx.Int32(i)
            row_out = fx.slice(Out_buf, (out_row, None))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            val_i = vector.extract(acc, static_position=[i], dynamic_position=[])
            val = ArithValue(val_i)

            if has_bias:
                val = val + bias_val
            if has_colvec:
                col_val = ArithValue(_load_f(colvec_div, out_row))
                val = val * col_val
            if has_rowvec:
                val = val * rowvec_val

            zero = arith.constant(0.0, type=T.f32)
            if activation == "relu":
                val = val.maximumf(zero)
            elif activation == "relu_sq":
                val = val.maximumf(zero) * val
            elif activation == "silu":
                val = val / (Float32(1.0) + _fm.exp(-val, fastmath="fast"))
            elif activation == "gelu_tanh_approx":
                import math as _py_math
                c1 = _py_math.sqrt(2.0 / _py_math.pi)
                c2 = 0.044715 * c1
                x = val
                x_sq = x * x
                tanh_arg = x * (Float32(c1) + Float32(c2) * x_sq)
                tanh_z = Float32(1.0) - Float32(2.0) / (
                    Float32(1.0) + _fm.exp(Float32(2.0) * tanh_arg, fastmath="fast")
                )
                val = x * (Float32(0.5) + Float32(0.5) * tanh_z)

            _store_out(out_div, out_col, val)

    @flyc.jit
    def launch(
        A: fx.Tensor, B: fx.Tensor,
        Bias: fx.Tensor, Colvec: fx.Tensor, Rowvec: fx.Tensor,
        Out: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, Bias, Colvec, Rowvec, Out).launch(
            grid=(M // _MFMA_M, N // _MFMA_N, 1),
            block=(64, 1, 1), stream=stream,
        )

    return launch


_kernel_cache: dict = {}


def _compile(M, N, K, activation, out_dtype, has_bias, has_colvec, has_rowvec, arch):
    key = (M, N, K, activation, out_dtype, has_bias, has_colvec, has_rowvec, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_norm_act_f16(
            M=M, N=N, K=K,
            activation=activation,
            out_dtype_str=_DTYPE2STR[out_dtype],
            has_bias=has_bias, has_colvec=has_colvec, has_rowvec=has_rowvec,
            arch=arch,
        )
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_gemm_norm_act_fused_out",
    mutates_args=("out",),
    schema=(
        "(Tensor a, Tensor b, Tensor bias, Tensor colvec, Tensor rowvec, "
        "Tensor(a0!) out, str? activation, bool has_bias, bool has_colvec, "
        "bool has_rowvec) -> ()"
    ),
)
def _gemm_norm_act_fused_out(
    a: Tensor, b: Tensor, bias: Tensor, colvec: Tensor, rowvec: Tensor,
    out: Tensor, activation: Optional[str],
    has_bias: bool, has_colvec: bool, has_rowvec: bool,
) -> None:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and M % 16 == 0 and N % 16 == 0 and K % 16 == 0
    _compile(
        M, N, K, activation, out.dtype,
        has_bias, has_colvec, has_rowvec, get_rocm_arch(),
    )(a, b, bias, colvec, rowvec, out)


@_gemm_norm_act_fused_out.register_fake
def _gemm_norm_act_fused_out_fake(
    a, b, bias, colvec, rowvec, out, activation, has_bias, has_colvec, has_rowvec,
):
    return None


def gemm_norm_act_fused(
    A: Tensor, B: Tensor,
    colvec: Optional[Tensor] = None,
    rowvec: Optional[Tensor] = None,
    *,
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Fully-fused ``activation((A@B + bias) * colvec * rowvec)``.

    All epilogue ops happen in the MFMA kernel's per-lane f32 register
    before the output store — saves the HBM round-trip of the
    torch-composed ``quack.amd.gemm.gemm_norm_act`` path.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    M, K = A.shape
    _, N = B.shape
    out_dtype = out_dtype or A.dtype
    out = torch.empty(M, N, device=A.device, dtype=out_dtype)
    # Stand-in tensors for unused optional slots; the kernel only
    # dereferences them when the corresponding flag is set.
    _bias = bias.to(torch.float32) if bias is not None else out
    _colvec = colvec.to(torch.float32) if colvec is not None else out
    _rowvec = rowvec.to(torch.float32) if rowvec is not None else out
    _gemm_norm_act_fused_out(
        A, B, _bias, _colvec, _rowvec, out, activation,
        bias is not None, colvec is not None, rowvec is not None,
    )
    return out


__all__ = ["gemm_norm_act_fused"]
