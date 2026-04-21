# Copyright (c) 2026, AMD.

"""Fused gated-activation-backward MFMA GEMM for gfx950.

Computes, per output column ``j`` (where ``PreAct`` holds interleaved
``(gate, up)`` pairs in the last axis):

    dout = (A @ B)[:, j]
    postact[:, j] = gate_fn(gate) * up
    dx[:, 2j]   = dout * up * gate_fn_prime(gate)
    dx[:, 2j+1] = dout * gate_fn(gate)            # = act_on_gate

The MFMA tile is 16×16 in the ``dout``-column space; each tile writes
the (16, 16) postact slice plus the (16, 32) interleaved dx slice.

Gate types (mirrors NVIDIA QuACK):
    swiglu   — silu(gate)  * up
    reglu    — relu(gate)  * up
    geglu    — gelu_th(gate) * up
    glu      — sigmoid(gate) * up

Scope (MVP): f16 × f16 inputs, f32/f16/bf16 output, M multiples of 16,
N (postact last-dim) multiples of 16, K multiple of 16. ``PreAct`` is
``(M, 2N)`` with interleaved ``gate, up``.

Relationship to ``quack.amd.gemm.gemm_dgated``: that path composes the
MFMA GEMM with torch autograd for the gate-backward. This kernel
saves the HBM round-trip through an f32 accumulator buffer — the
gate-fwd/bwd happens in the epilogue register.
"""

from typing import Optional, Tuple

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


def _build_gemm_dgated_f16(*, M, N, K, gate_type, out_dtype_str, preact_dtype_str, arch):
    """N here is the postact column count; PreAct is (M, 2N)."""
    assert M % _MFMA_M == 0 and N % _MFMA_N == 0 and K % _MFMA_K == 0
    assert gate_type in ("swiglu", "reglu", "geglu", "glu")
    assert out_dtype_str in ("f32", "f16", "bf16")

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=(
            f"quack_amd_gemm_dgated_{gate_type}"
            f"_out{out_dtype_str}_pre{preact_dtype_str}"
            f"_{M}_{N}_{K}_smem"
        ),
    )

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, PreAct: fx.Tensor,
               Dx: fx.Tensor, Postact: fx.Tensor):
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y
        tid = fx.thread_idx.x

        lane_row = tid % fx.Int32(16)
        lane_k_group = tid // fx.Int32(16)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        PreAct_buf = fx.rocdl.make_buffer_tensor(PreAct)
        Dx_buf = fx.rocdl.make_buffer_tensor(Dx)
        Postact_buf = fx.rocdl.make_buffer_tensor(Postact)

        in_elem_type = T.f16
        pre_elem_type = {"f16": T.f16, "bf16": T.bf16, "f32": T.f32}[preact_dtype_str]
        out_elem_type = {"f32": T.f32, "f16": T.f16, "bf16": T.bf16}[out_dtype_str]

        ca_h = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), in_elem_type)
        ca_pre = fx.make_copy_atom(
            fx.rocdl.BufferCopy32b() if preact_dtype_str == "f32" else fx.rocdl.BufferCopy16b(),
            pre_elem_type,
        )
        ca_out = fx.make_copy_atom(
            fx.rocdl.BufferCopy32b() if out_dtype_str == "f32" else fx.rocdl.BufferCopy16b(),
            out_elem_type,
        )
        h_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        pre_reg_ty = fx.MemRefType.get(pre_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        out_reg_ty = fx.MemRefType.get(out_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

        def _load_h(div, idx):
            r = fx.memref_alloca(h_reg_ty, reg_lay)
            fx.copy_atom_call(ca_h, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _load_pre(div, idx):
            r = fx.memref_alloca(pre_reg_ty, reg_lay)
            fx.copy_atom_call(ca_pre, fx.slice(div, (None, idx)), r)
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
        n_base = bid_n * fx.Int32(_MFMA_N)   # postact col base
        a_row = m_base + lane_row
        b_col = n_base + lane_row            # dout col in postact space

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

        # Fused epilogue: per-lane load (gate, up), compute gate bwd, store.
        for i in range_constexpr(_FRAG_C):
            out_row = m_base + lane_k_group * fx.Int32(_FRAG_C) + fx.Int32(i)
            postact_col = n_base + lane_row
            # PreAct is (M, 2N); interleaved gate at 2*col, up at 2*col+1.
            gate_col = postact_col * fx.Int32(2)
            up_col = gate_col + fx.Int32(1)

            pre_row = fx.slice(PreAct_buf, (out_row, None))
            pre_div = fx.logical_divide(pre_row, fx.make_layout(1, 1))
            dx_row = fx.slice(Dx_buf, (out_row, None))
            dx_div = fx.logical_divide(dx_row, fx.make_layout(1, 1))
            post_row = fx.slice(Postact_buf, (out_row, None))
            post_div = fx.logical_divide(post_row, fx.make_layout(1, 1))

            gate_e = _load_pre(pre_div, gate_col)
            up_e = _load_pre(pre_div, up_col)
            gate = ArithValue(
                gate_e if preact_dtype_str == "f32" else ArithValue(gate_e).extf(T.f32)
            )
            up = ArithValue(
                up_e if preact_dtype_str == "f32" else ArithValue(up_e).extf(T.f32)
            )

            val_i = vector.extract(acc, static_position=[i], dynamic_position=[])
            dout = ArithValue(val_i)

            one = arith.constant(1.0, type=T.f32)
            zero = arith.constant(0.0, type=T.f32)

            if gate_type == "swiglu":
                # y = silu(gate) * up; sig = sigmoid(gate)
                sig = ArithValue(one) / (ArithValue(one) + _fm.exp(-gate, fastmath="fast"))
                silu_g = gate * sig
                silu_prime = sig * (ArithValue(one) + gate * (ArithValue(one) - sig))
                postact = silu_g * up
                # d y / d gate = up * silu_prime(gate)
                # d y / d up   = silu(gate)
                dgate = dout * up * silu_prime
                dup = dout * silu_g
            elif gate_type == "reglu":
                is_pos = gate > Float32(0.0)
                relu_g = ArithValue(is_pos.select(gate, ArithValue(zero)))
                postact = relu_g * up
                dgate = dout * up * ArithValue(is_pos.select(one, zero))
                dup = dout * relu_g
            elif gate_type == "geglu":
                import math as _py_math
                c1 = _py_math.sqrt(2.0 / _py_math.pi)
                c2 = 0.044715 * c1
                x = gate
                x_sq = x * x
                tanh_arg = x * (Float32(c1) + Float32(c2) * x_sq)
                tanh_z = ArithValue(one) - Float32(2.0) / (
                    ArithValue(one) + _fm.exp(Float32(2.0) * tanh_arg, fastmath="fast")
                )
                gelu_g = x * (Float32(0.5) + Float32(0.5) * tanh_z)
                sech_sq = ArithValue(one) - tanh_z * tanh_z
                gelu_prime = Float32(0.5) * (ArithValue(one) + tanh_z) + (
                    Float32(0.5) * x * sech_sq * (Float32(c1) + Float32(3.0 * c2) * x_sq)
                )
                postact = gelu_g * up
                dgate = dout * up * gelu_prime
                dup = dout * gelu_g
            elif gate_type == "glu":
                sig = ArithValue(one) / (ArithValue(one) + _fm.exp(-gate, fastmath="fast"))
                postact = sig * up
                dgate = dout * up * sig * (ArithValue(one) - sig)
                dup = dout * sig

            _store_out(dx_div, gate_col, dgate)
            _store_out(dx_div, up_col, dup)
            _store_out(post_div, postact_col, postact)

    @flyc.jit
    def launch(
        A: fx.Tensor, B: fx.Tensor, PreAct: fx.Tensor,
        Dx: fx.Tensor, Postact: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, PreAct, Dx, Postact).launch(
            grid=(M // _MFMA_M, N // _MFMA_N, 1),
            block=(64, 1, 1), stream=stream,
        )

    return launch


_kernel_cache: dict = {}


def _compile(M, N, K, gate_type, out_dtype, preact_dtype, arch):
    key = (M, N, K, gate_type, out_dtype, preact_dtype, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_dgated_f16(
            M=M, N=N, K=K,
            gate_type=gate_type,
            out_dtype_str=_DTYPE2STR[out_dtype],
            preact_dtype_str=_DTYPE2STR[preact_dtype],
            arch=arch,
        )
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_gemm_dgated_fused_out",
    mutates_args=("dx", "postact"),
    schema=(
        "(Tensor a, Tensor b, Tensor preact, Tensor(a0!) dx, "
        "Tensor(a1!) postact, str gate_type) -> ()"
    ),
)
def _gemm_dgated_fused_out(
    a: Tensor, b: Tensor, preact: Tensor,
    dx: Tensor, postact: Tensor, gate_type: str,
) -> None:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert preact.shape == (M, 2 * N)
    assert dx.shape == (M, 2 * N)
    assert postact.shape == (M, N)
    _compile(M, N, K, gate_type, dx.dtype, preact.dtype, get_rocm_arch())(
        a, b, preact, dx, postact,
    )


@_gemm_dgated_fused_out.register_fake
def _gemm_dgated_fused_out_fake(a, b, preact, dx, postact, gate_type):
    return None


def gemm_dgated_fused(
    A: Tensor, B: Tensor, PreAct: Tensor,
    gate_type: str = "swiglu",
    *,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """Fully-fused gated-activation-bwd GEMM.

    Returns ``(dx, postact)`` matching
    ``quack.amd.gemm.gemm_dgated``'s unfused surface.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    M, K = A.shape
    _, N = B.shape
    out_dtype = out_dtype or A.dtype
    postact_dtype = postact_dtype or PreAct.dtype
    dx = torch.empty(M, 2 * N, device=A.device, dtype=out_dtype)
    postact = torch.empty(M, N, device=A.device, dtype=postact_dtype)
    _gemm_dgated_fused_out(A, B, PreAct, dx, postact, gate_type)
    return dx, postact


__all__ = ["gemm_dgated_fused"]
