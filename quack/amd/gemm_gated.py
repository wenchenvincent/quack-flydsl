# Copyright (c) 2026, AMD.

"""Gated MFMA GEMM — fused `gate_fn(A @ B_gate) * (A @ B_up)`.

For an input ``B`` that's the concatenation ``[B_gate, B_up]`` along N
(the canonical weight layout for SwiGLU/ReGLU/GeGLU MLPs), computes::

    out[m, j] = gate_fn(A[m] @ B_gate[:, j]) * (A[m] @ B_up[:, j])

where ``j`` ranges over the first half ``[0, N/2)`` of the concatenated
B's columns. The output shape is ``(M, N/2)``.

Each workgroup owns one 16×N/16 chunk of ``out``, computing two
side-by-side 16×16 accumulators (gate half + up half) and combining
them in the epilogue — saves the memory round-trip that an unfused
``gemm + elementwise`` pipeline would incur.

Scope (MVP):
    - f16/bf16 × f16/bf16 → f32/f16/bf16
    - N must be a multiple of 32 (so N/2 is a multiple of the 16 tile)
    - K multiple of 16
    - Activations: swiglu / reglu / geglu / glu

Grid = ``(M/16, N/32, 1)`` (note N/32, not N/16 — each wg produces 16
output columns but consumes 32 B-columns: gate + up).
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


_SUPPORTED_GATES = {"swiglu", "reglu", "geglu", "glu", "swiglu_oai"}


def _apply_gate(gate_val, up_val, gate_type):
    """Apply the gating function to f32 ArithValues. Same math as
    gemm_gfx950.py's inlined activations — computed via hardware exp."""
    if fx.const_expr(gate_type == "swiglu"):
        # silu(gate) * up. silu(x) = x / (1 + exp(-x)).
        silu = gate_val / (Float32(1.0) + _fm.exp(-gate_val, fastmath="fast"))
        return silu * up_val
    if fx.const_expr(gate_type == "reglu"):
        # relu(gate) * up.
        zero = arith.constant(0.0, type=T.f32)
        return gate_val.maximumf(zero) * up_val
    if fx.const_expr(gate_type == "geglu"):
        # gelu_tanh_approx(gate) * up.
        import math as _py_math
        c1 = _py_math.sqrt(2.0 / _py_math.pi)
        c2 = 0.044715 * c1
        x = gate_val
        x_sq = x * x
        tanh_arg = x * (c1 + c2 * x_sq)
        tanh_z = Float32(1.0) - Float32(2.0) / (Float32(1.0) + _fm.exp(Float32(2.0) * tanh_arg, fastmath="fast"))
        gelu = x * (Float32(0.5) + Float32(0.5) * tanh_z)
        return gelu * up_val
    if fx.const_expr(gate_type == "glu"):
        # sigmoid(gate) * up. sigmoid(x) = 1 / (1 + exp(-x)).
        sig = Float32(1.0) / (Float32(1.0) + _fm.exp(-gate_val, fastmath="fast"))
        return sig * up_val
    if fx.const_expr(gate_type == "swiglu_oai"):
        # gpt-oss variant: silu_oai(gate) * (up + 1),
        # silu_oai(x) = 0.5x * tanh(1.702 * 0.5x) + 0.5x.
        half = Float32(0.5) * gate_val
        z = Float32(1.702) * half
        # tanh(z) = 1 - 2 / (1 + exp(2z)).
        tanh_z = Float32(1.0) - Float32(2.0) / (
            Float32(1.0) + _fm.exp(Float32(2.0) * z, fastmath="fast")
        )
        silu_oai = half * tanh_z + half
        return silu_oai * (up_val + Float32(1.0))
    raise ValueError(f"unknown gate_type: {gate_type}")


def _build_gated_16x16(*, M, N, K, dtype_str, out_dtype_str, gate_type, arch):
    """N is the *full* B columns (gate+up); output has N/2 columns."""
    assert dtype_str in {"f16", "bf16"}
    assert out_dtype_str in {"f32", "f16", "bf16"}
    assert gate_type in _SUPPORTED_GATES
    assert N % 32 == 0, "gated GEMM requires N % 32 == 0 (N/2 multiple of 16)"
    assert M % _MFMA_M == 0 and K % _MFMA_K == 0
    H = N // 2  # hidden / output column count

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_gated_{dtype_str}_{out_dtype_str}_{gate_type}_smem",
    )

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        bid_m = fx.block_idx.x
        bid_h = fx.block_idx.y   # 0..H/16-1 — selects one 16-col output tile
        tid = fx.thread_idx.x
        lane_row = tid % fx.Int32(16)
        lane_k_group = tid // fx.Int32(16)

        in_elem_type = T.f16 if dtype_str == "f16" else T.bf16
        out_elem_type = (
            T.f32 if out_dtype_str == "f32"
            else T.f16 if out_dtype_str == "f16"
            else T.bf16
        )
        out_bufcopy = fx.rocdl.BufferCopy32b() if out_dtype_str == "f32" else fx.rocdl.BufferCopy16b()
        ca_h4 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), in_elem_type)
        ca_h = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), in_elem_type)
        ca_out = fx.make_copy_atom(out_bufcopy, out_elem_type)
        h4_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(4, 1), fx.AddressSpace.Register)
        h_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        out_reg_ty = fx.MemRefType.get(out_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        h4_lay = fx.make_layout(4, 1)
        reg_lay = fx.make_layout(1, 1)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        def _load_h4(div_vec, vec_idx):
            r = fx.memref_alloca(h4_reg_ty, h4_lay)
            fx.copy_atom_call(ca_h4, fx.slice(div_vec, (None, vec_idx)), r)
            return fx.memref_load_vec(r)

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
        acc_gate = vector.from_elements(acc_ty, zeros)
        acc_up = vector.from_elements(acc_ty, zeros)

        m_base = bid_m * fx.Int32(_MFMA_M)
        # bid_h picks which output-column-group we produce. That maps to
        # B-columns [bid_h*16, bid_h*16+16) for gate AND
        # [H + bid_h*16, H + bid_h*16+16) for up.
        h_base = bid_h * fx.Int32(_MFMA_N)
        gate_col = h_base + lane_row            # B col for gate half
        up_col = h_base + lane_row + fx.Int32(H)  # B col for up half
        a_row = m_base + lane_row

        # A row is shared across the whole K loop — precompute vec4 divide.
        row_a = fx.slice(A_buf, (a_row, None))
        a_div_v = fx.logical_divide(row_a, h4_lay)

        # K-tile loop: compute gate and up accumulators in parallel.
        k_tiles = K // _MFMA_K
        for k_tile in range_constexpr(k_tiles):
            k_tile_base = fx.Int32(k_tile * _MFMA_K)
            lane_k_base = lane_k_group * fx.Int32(_FRAG_C) + k_tile_base
            vec_idx_A = lane_k_group + fx.Int32(k_tile * 4)

            # A fragment (shared by gate and up): one BufferCopy64b.
            a_frag = _load_h4(a_div_v, vec_idx_A)

            # B fragments for gate and up halves.
            b_gate_vals = []
            b_up_vals = []
            for i in range_constexpr(_FRAG_C):
                row_b_k = fx.slice(B_buf, (lane_k_base + fx.Int32(i), None))
                b_div = fx.logical_divide(row_b_k, fx.make_layout(1, 1))
                b_gate_vals.append(_load_h(b_div, gate_col))
                b_up_vals.append(_load_h(b_div, up_col))
            b_gate_frag = vector.from_elements(T.vec(_FRAG_C, in_elem_type), b_gate_vals)
            b_up_frag = vector.from_elements(T.vec(_FRAG_C, in_elem_type), b_up_vals)

            if fx.const_expr(dtype_str == "bf16"):
                a_frag_i16 = vector.bitcast(T.vec(_FRAG_C, T.i16), a_frag)
                b_gate_i16 = vector.bitcast(T.vec(_FRAG_C, T.i16), b_gate_frag)
                b_up_i16 = vector.bitcast(T.vec(_FRAG_C, T.i16), b_up_frag)
                acc_gate = fx.rocdl.mfma_f32_16x16x16bf16_1k(
                    acc_ty, [a_frag_i16, b_gate_i16, acc_gate, 0, 0, 0],
                )
                acc_up = fx.rocdl.mfma_f32_16x16x16bf16_1k(
                    acc_ty, [a_frag_i16, b_up_i16, acc_up, 0, 0, 0],
                )
            else:
                acc_gate = fx.rocdl.mfma_f32_16x16x16f16(
                    acc_ty, [a_frag, b_gate_frag, acc_gate, 0, 0, 0],
                )
                acc_up = fx.rocdl.mfma_f32_16x16x16f16(
                    acc_ty, [a_frag, b_up_frag, acc_up, 0, 0, 0],
                )

        # Output: each lane writes 4 rows at column `h_base + lane_row`.
        # out has shape (M, H), so the output column is `h_base + lane_row`.
        for i in range_constexpr(_FRAG_C):
            out_row = m_base + lane_k_group * fx.Int32(_FRAG_C) + fx.Int32(i)
            out_col = h_base + lane_row
            row_c = fx.slice(C_buf, (out_row, None))
            c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))
            g = vector.extract(acc_gate, static_position=[i], dynamic_position=[])
            u = vector.extract(acc_up, static_position=[i], dynamic_position=[])
            combined = _apply_gate(ArithValue(g), ArithValue(u), gate_type)
            _store_out(c_div, out_col, combined)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, C).launch(
            grid=(M // _MFMA_M, H // _MFMA_N, 1),
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


def gemm_gated(
    A: Tensor, B: Tensor, gate_type: str = "swiglu",
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Gated MFMA GEMM: ``gate_fn(A @ B_gate) * (A @ B_up)``.

    ``B`` is ``[B_gate, B_up]`` concatenated along N. Output shape
    ``(M, N/2)``. N must be a multiple of 32, M/K multiples of 16.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype in (torch.float16, torch.bfloat16)
    assert A.dtype == B.dtype
    assert gate_type in _SUPPORTED_GATES
    if fx.const_expr(out_dtype is None):
        out_dtype = A.dtype
    assert out_dtype in _OUT_DTYPE_MAP
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert M % 16 == 0 and N % 32 == 0 and K % 16 == 0
    dtype_str = "f16" if A.dtype == torch.float16 else "bf16"
    out_dtype_str = _OUT_DTYPE_MAP[out_dtype]
    key = (M, N, K, dtype_str, out_dtype_str, gate_type, get_rocm_arch())
    launcher = _kernel_cache.get(key)
    if fx.const_expr(launcher is None):
        launcher = _build_gated_16x16(
            M=M, N=N, K=K, dtype_str=dtype_str, out_dtype_str=out_dtype_str,
            gate_type=gate_type, arch=get_rocm_arch(),
        )
        _kernel_cache[key] = launcher
    out = torch.empty(M, N // 2, device=A.device, dtype=out_dtype)
    launcher(A, B, out)
    return out


__all__ = ["gemm_gated"]
