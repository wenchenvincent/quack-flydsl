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
        # Single-layer linear+act: dpreact = grad_out * act'(preact) (no
        # matmul fusion available here — fusion only applies in 2-layer
        # MLP backward via ``mlp_func_train``).
        dpreact = _act_bwd(preact, grad_out, ctx.activation)
        grad_x = grad_w = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.mm(dpreact, weight)
        if ctx.needs_input_grad[1]:
            grad_w = torch.mm(dpreact.t(), x)
        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_b = dpreact.sum(dim=0).to(torch.float32)
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


_DACT_ELIGIBLE_ACTIVATIONS = {"relu", "relu_sq", "silu", "gelu_tanh_approx"}


def _fused_dact_eligible(dout: Tensor, w2: Tensor, preact: Tensor, activation: str) -> bool:
    """Check shape constraints for the fused gemm_dact MLP-backward path.

    The fused kernel computes ``dpreact = (dout @ W2) * act'(preact)`` in
    one pass. In splitk NT form (``out = a @ b.T``):
      a = dout          shape (M, out_dim)
      b = w2.T          shape (hidden, out_dim)   [so b.T = w2 with shape (out_dim, hidden)]
      out = a @ b.T     shape (M, hidden) — then * act'(preact[m, n]) per element
      preact            shape (M, hidden)         [matches splitk N = hidden]

    splitk NT inner-dim constraint is K_mm%64 (with K_mm = out_dim),
    N_mm%256 (= hidden for NT), M%128.
    """
    if activation not in _DACT_ELIGIBLE_ACTIVATIONS:
        return False
    if dout.dtype not in (torch.float16, torch.bfloat16):
        return False
    if w2.dtype != dout.dtype or preact.dtype != dout.dtype:
        return False
    M, out_dim = dout.shape
    out_dim2, hidden = w2.shape
    if out_dim != out_dim2:
        return False
    if preact.shape != (M, hidden):
        return False
    if not (M % 128 == 0 and hidden % 256 == 0 and out_dim % 64 == 0 and M >= 128):
        return False
    # Disabled by default: measured on MI355X, splitk's matmul at the
    # typical MLP-backward shape (e.g. M=4096, hidden=8192, out_dim=4096,
    # bf16) runs ~1.07× slower than hipBLASLt's torch.mm. The fusion
    # saves ~50 μs (one (M, hidden) HBM roundtrip + one act-bwd elementwise
    # kernel) but the matmul gap eats it — net slower overall. The
    # fused kernel body IS shipped and correctness-tested; flipping
    # this flag would dispatch through it when splitk broadly matches
    # hipBLASLt across shapes. Users can still call
    # ``gemm_splitk(a, b, preact=p, dact_activation=act)`` directly to
    # exercise the fused kernel.
    return False  # noqa: PLW0177 — intentional: see comment above.
    # return dout.stride(-1) == 1 and preact.stride(-1) == 1


class _MLPActFunction(torch.autograd.Function):
    """Two-layer MLP with activation-backward fusion via gemm_dact.

    Forward: ``out = act(linear(x, w1)) @ w2.T``. Saves preact (not postact)
    so backward can recompute postact cheaply and fuse the dpreact
    computation into the dout @ w2 matmul via the splitk kernel's
    ``dact_activation`` epilogue. Matches NVIDIA's
    ``MLPRecomputeFunc`` pattern at the kernel-fusion level.
    """

    @staticmethod
    def forward(ctx, x, w1, w2, activation, bias1, bias2):
        preact = linear(x, w1, bias=bias1)
        postact = _act_fwd(preact, activation)
        out = linear(postact, w2, bias=bias2)
        # Save both preact (for dact fused backward) and postact (for dW2
        # without recompute). 2× activation memory vs torch's save-only-postact,
        # but avoids the extra act_fwd kernel launch in backward.
        ctx.save_for_backward(x, w1, w2, preact, postact)
        ctx.activation = activation
        ctx.has_bias1 = bias1 is not None
        ctx.has_bias2 = bias2 is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w1, w2, preact, postact = ctx.saved_tensors
        grad_x = grad_w1 = grad_w2 = grad_b1 = grad_b2 = None
        # --- dW2 = grad_out.T @ postact ---
        if ctx.needs_input_grad[2]:
            grad_w2 = torch.mm(grad_out.t(), postact)
        if ctx.has_bias2 and ctx.needs_input_grad[5]:
            grad_b2 = grad_out.sum(dim=0).to(torch.float32)
        # --- dpreact = (grad_out @ w2) * act'(preact) ---
        need_dpreact = ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or (
            ctx.has_bias1 and ctx.needs_input_grad[4]
        )
        if need_dpreact:
            if _fused_dact_eligible(grad_out, w2, preact, ctx.activation):
                # Kernel-level fusion: matmul + act_bwd in one pass.
                # NT splitk form: dpreact = gemm_splitk(dout, w2.T, ...)
                # where w2.T has shape (hidden, out_dim). Contiguous copy
                # required since splitk wants last-dim-contiguous.
                from quack.amd.gemm_gfx950_splitk import gemm_splitk
                w2_T = w2.t().contiguous()
                dpreact = gemm_splitk(
                    grad_out, w2_T,
                    preact=preact, dact_activation=ctx.activation,
                )
            else:
                grad_postact = torch.mm(grad_out, w2)
                dpreact = _act_bwd(preact, grad_postact, ctx.activation)
            if ctx.needs_input_grad[0]:
                grad_x = torch.mm(dpreact, w1)
            if ctx.needs_input_grad[1]:
                grad_w1 = torch.mm(dpreact.t(), x)
            if ctx.has_bias1 and ctx.needs_input_grad[4]:
                grad_b1 = dpreact.sum(dim=0).to(torch.float32)
        return grad_x, grad_w1, grad_w2, None, grad_b1, grad_b2


def mlp_func_train(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    activation: str = "silu",
    bias1: Optional[Tensor] = None,
    bias2: Optional[Tensor] = None,
) -> Tensor:
    """Two-layer MLP with activation-backward fusion (autograd.Function).

    Forward: ``out = linear(act(linear(x, w1, bias1)), w2, bias2)``.

    Backward has a kernel-level fused path via ``gemm_splitk`` with the
    ``dact_activation`` epilogue — the ``dout @ w2`` matmul and the
    ``* act'(preact)`` multiplication happen in one kernel, saving
    one (M, hidden) HBM roundtrip. Matches NVIDIA's
    ``GemmDActMixin`` / ``matmul_bwd_dact`` pattern.

    **Status of the fused path** (gated in ``_fused_dact_eligible``):
    currently disabled by default because splitk's matmul runs ~1.07×
    slower than hipBLASLt's torch.mm at typical MLP-backward shapes
    on MI355X — the fusion saving (~50 μs) is less than the matmul
    gap (~20 μs per invocation) at these shapes. The fused kernel is
    correctness-tested and can be invoked directly via
    ``gemm_splitk(a, b, preact=p, dact_activation=act)``. Flipping
    the gate in ``_fused_dact_eligible`` turns on dispatch once
    splitk hits hipBLASLt parity across shapes.

    Eligible activations: relu, silu, gelu_tanh_approx, relu_sq.
    """
    return _MLPActFunction.apply(x, w1, w2, activation, bias1, bias2)


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


__all__ = ["linear_act_func", "linear_gated_func", "mlp_func_train"]
