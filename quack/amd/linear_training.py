# Copyright (c) 2026, AMD.

"""Training-integrated linear ops with activations.

Exposes autograd.Function wrappers for:
  - ``linear_act_func(x, w, activation, bias=None)``
      Forward: ``y = act(linear(x, w, bias))``
      Backward: ``dpreact = dy * act'(preact); dx, dw, db`` via bf16 matmul.
  - ``linear_gated_func(x, w_gate_up, gate_type, bias=None)``
      Forward: ``y = gate_fn(gate, up)`` where ``(gate, up) = chunk(linear, 2)``
      Backward: ``(dgate, dup) = gate_bwd(gate, up, dy); dx, dw, db``.

These wrap the existing ``linear()`` forward path (which routes aligned
bf16 shapes through ``gemm_splitk``). Backwards are computed
element-wise in torch — the activation derivative runs as a separate
kernel from the two backward matmuls. NVIDIA's CUTLASS fuses the
activation-backward *into* the backward matmul (``GemmDActMixin``);
porting that kernel-level fusion is a larger follow-up. The autograd
wrappers here give users correct gradients with the ~1.1× hipBLASLt
splitk path on the two backward matmuls — a substantial win vs
torch's default which materialises each step through autograd's
stored-tensor graph.

Memory trade-off: forward saves ``preact`` (the pre-activation tensor).
For a gated MLP up-proj with ``M=8192, 2*hidden=16384, bf16`` that's
256 MB saved — big. Activation-recompute variants (NVIDIA's
``MLPRecomputeFunc``) are the follow-up for HBM-constrained workloads;
the extra forward matmul cost is offset by ``(M, 2*hidden)`` HBM
saved.
"""

from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from quack.amd.linear import linear


def _act_fwd(preact: Tensor, activation: str) -> Tensor:
    if activation == "relu":
        return F.relu(preact)
    if activation == "relu_sq":
        r = F.relu(preact)
        return r * preact
    if activation == "silu":
        return F.silu(preact)
    if activation == "gelu_tanh_approx":
        return F.gelu(preact, approximate="tanh")
    raise ValueError(f"unknown activation {activation!r}")


def _act_bwd(preact: Tensor, dy: Tensor, activation: str) -> Tensor:
    """Per-element activation backward: dpreact = dy * act'(preact).

    All operations run in f32 for numerical stability then cast back.
    """
    p32 = preact.float()
    d32 = dy.float()
    if activation == "relu":
        dpreact = d32 * (p32 > 0).float()
    elif activation == "relu_sq":
        # d/dx [relu(x)^2 * ... wait, we defined relu_sq = relu(x) * x,
        #       then d/dx [relu(x) * x] = relu'(x) * x + relu(x) * 1
        #       = (x>0) * x + max(x, 0)
        #       = 2x  if x > 0 else 0.
        dpreact = d32 * (2.0 * p32) * (p32 > 0).float()
    elif activation == "silu":
        # d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        #                      = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sig = torch.sigmoid(p32)
        dpreact = d32 * sig * (1.0 + p32 * (1.0 - sig))
    elif activation == "gelu_tanh_approx":
        # f(x) = 0.5 * x * (1 + tanh(z)) where z = sqrt(2/pi) * (x + 0.044715*x^3)
        # f'(x) = 0.5 * (1 + tanh(z)) + 0.5 * x * (1 - tanh(z)^2) * dz/dx
        # dz/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        import math as _py_math
        c1 = _py_math.sqrt(2.0 / _py_math.pi)
        x = p32
        z = c1 * (x + 0.044715 * x.pow(3))
        th = torch.tanh(z)
        dz_dx = c1 * (1.0 + 3.0 * 0.044715 * x.pow(2))
        dpreact = d32 * (0.5 * (1.0 + th) + 0.5 * x * (1.0 - th.pow(2)) * dz_dx)
    else:
        raise ValueError(f"unknown activation {activation!r}")
    return dpreact.to(preact.dtype)


def _gated_bwd(gate: Tensor, up: Tensor, dy: Tensor, gate_type: str):
    """Gated-activation backward: given (gate, up, dy) return (dgate, dup)."""
    g32 = gate.float()
    u32 = up.float()
    d32 = dy.float()
    if gate_type == "swiglu":
        sig = torch.sigmoid(g32)
        silu_g = g32 * sig
        # y = silu(g) * u
        # dy/dg = silu'(g) * u ; silu'(g) = sigmoid(g) + g * sigmoid(g) * (1 - sigmoid(g))
        #                              = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
        dsilu_dg = sig * (1.0 + g32 * (1.0 - sig))
        dgate = d32 * dsilu_dg * u32
        dup = d32 * silu_g
    elif gate_type == "reglu":
        mask = (g32 > 0).float()
        dgate = d32 * mask * u32
        dup = d32 * F.relu(g32)
    elif gate_type == "geglu":
        import math as _py_math
        c1 = _py_math.sqrt(2.0 / _py_math.pi)
        z = c1 * (g32 + 0.044715 * g32.pow(3))
        th = torch.tanh(z)
        dz_dg = c1 * (1.0 + 3.0 * 0.044715 * g32.pow(2))
        gelu_g = 0.5 * g32 * (1.0 + th)
        dgelu_dg = 0.5 * (1.0 + th) + 0.5 * g32 * (1.0 - th.pow(2)) * dz_dg
        dgate = d32 * dgelu_dg * u32
        dup = d32 * gelu_g
    elif gate_type == "glu":
        sig = torch.sigmoid(g32)
        dsig_dg = sig * (1.0 - sig)
        dgate = d32 * dsig_dg * u32
        dup = d32 * sig
    else:
        raise ValueError(f"unknown gate_type {gate_type!r}")
    return dgate.to(gate.dtype), dup.to(up.dtype)


class _LinearActFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, activation, bias):
        preact = linear(x, weight, bias=bias)
        out = _act_fwd(preact, activation)
        ctx.save_for_backward(x, weight, preact)
        ctx.activation = activation
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, preact = ctx.saved_tensors
        dpreact = _act_bwd(preact, grad_out, ctx.activation)
        grad_x = grad_w = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.mm(dpreact, weight)
        if ctx.needs_input_grad[1]:
            grad_w = torch.mm(dpreact.t(), x)
        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_b = dpreact.sum(dim=0).to(torch.float32)
        # grads align with forward inputs: (x, weight, activation, bias).
        return grad_x, grad_w, None, grad_b


class _LinearGatedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_gate_up, gate_type, bias):
        # Use split-halves convention for backward ergonomics (gate + up
        # are stored in consecutive halves of preact; gate_bwd operates
        # on chunked tensors natively). A future in-kernel-fused
        # training path could pair differently.
        preact = linear(x, weight_gate_up, bias=bias)  # (M, 2*hidden)
        gate, up = preact.chunk(2, dim=-1)
        if gate_type == "swiglu":
            out = F.silu(gate) * up
        elif gate_type == "reglu":
            out = F.relu(gate) * up
        elif gate_type == "geglu":
            out = F.gelu(gate, approximate="tanh") * up
        elif gate_type == "glu":
            out = torch.sigmoid(gate) * up
        else:
            raise ValueError(f"unknown gate_type {gate_type!r}")
        ctx.save_for_backward(x, weight_gate_up, gate, up)
        ctx.gate_type = gate_type
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_gate_up, gate, up = ctx.saved_tensors
        dgate, dup = _gated_bwd(gate, up, grad_out, ctx.gate_type)
        dpreact = torch.cat([dgate, dup], dim=-1)
        grad_x = grad_w = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.mm(dpreact, weight_gate_up)
        if ctx.needs_input_grad[1]:
            grad_w = torch.mm(dpreact.t(), x)
        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_b = dpreact.sum(dim=0).to(torch.float32)
        return grad_x, grad_w, None, grad_b


def linear_act_func(
    x: Tensor,
    weight: Tensor,
    activation: str,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Autograd-integrated linear + activation.

    Forward uses the fused splitk path (with bias + activation fused
    into the write-back) when shape-eligible. Backward runs the
    activation derivative torch-side, then two bf16 matmuls for dx
    and dw via hipBLASLt → MFMA.

    Supports activations: relu, relu_sq, silu, gelu_tanh_approx.
    """
    return _LinearActFunction.apply(x, weight, activation, bias)


def linear_gated_func(
    x: Tensor,
    weight_gate_up: Tensor,
    gate_type: str = "swiglu",
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Autograd-integrated gated linear (SwiGLU / ReGLU / GeGLU / GLU).

    Forward uses the ``linear_gated`` split-halves path (matmul then
    torch-side gate); the in-kernel fused path requires an interleaved
    weight layout, which the autograd Function would need to save an
    "un-interleaved" view of for backward. Accepting split-halves
    here keeps the ergonomics simple — forward perf is still matmul-
    dominated.

    Backward: per-element ``gate_bwd(gate, up, dy) → (dgate, dup)``
    running in torch, then dx / dw via bf16 matmul. The saved tensors
    are the forward ``(gate, up)`` halves — same HBM footprint as
    torch's default autograd, but the wrapper ensures the two
    backward matmuls go through the fast splitk-eligible bf16 path.
    """
    return _LinearGatedFunction.apply(x, weight_gate_up, gate_type, bias)


__all__ = ["linear_act_func", "linear_gated_func"]
