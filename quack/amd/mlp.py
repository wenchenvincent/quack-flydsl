# Copyright (c) 2026, AMD.

"""MLP — AMDGPU port of `quack/mlp.py`.

Thin composition of two ``quack.amd.linear`` calls with an activation or
gated activation between them.

Two entry points:
  - ``mlp`` — inference, forward-only, no autograd wiring.
  - ``mlp_train`` — autograd-aware, uses ``LinearActFunc`` + ``LinearFunc``
    so ``.backward()`` reaches our NN/TN kernels.
"""

from typing import Optional

from torch import Tensor

from quack.amd.linear import linear, linear_gated, linear_act_train, linear_train


def mlp(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    activation: str = "silu",
    bias1: Optional[Tensor] = None,
    bias2: Optional[Tensor] = None,
) -> Tensor:
    """Standard two-layer MLP: ``w2 @ activation(w1 @ x + bias1) + bias2``.

    Weights are ``(out, in)`` like ``torch.nn.Linear``.
    """
    h = linear(x, w1, bias=bias1, activation=activation)
    y = linear(h, w2, bias=bias2)
    return y


def gated_mlp(
    x: Tensor,
    w_gate_up: Tensor,
    w_down: Tensor,
    gate_type: str = "swiglu",
    bias_gate_up: Optional[Tensor] = None,
    bias_down: Optional[Tensor] = None,
) -> Tensor:
    """Gated MLP (SwiGLU-style).

    ``w_gate_up`` is a single matrix of shape ``(2 * hidden, in_features)``;
    the output of ``x @ w_gate_up.T`` is split into ``(gate, up)`` halves and
    combined via ``gate_type`` before projection through ``w_down``.

    Routes through ``linear_gated`` (splitk + torch-side gating) when the
    up-proj matmul is aligned, else the existing NN ``gemm_gated`` kernel.
    """
    h = linear_gated(x, w_gate_up, gate_type=gate_type, bias=bias_gate_up)
    return linear(h, w_down, bias=bias_down)


def mlp_train(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    activation: str = "relu",
    bias1: Optional[Tensor] = None,
    bias2: Optional[Tensor] = None,
) -> Tensor:
    """Autograd-aware two-layer MLP: ``w2 @ act(w1 @ x + b1) + b2``.

    Uses ``LinearActFunc`` for the first layer (activation fused into the
    autograd Function's saved state) and ``LinearFunc`` for the second.
    Both layers route backward through ``gemm_nn`` / ``gemm_tn`` so the
    full training step stays on our kernels.

    Weights are ``(out, in)`` like ``torch.nn.Linear``.
    """
    h = linear_act_train(x, w1, activation=activation, bias=bias1)
    if bias2 is None:
        y = linear_train(h, w2)
    else:
        # MVP: bias2 on the output layer falls back to the
        # unfused-act LinearActFunc with activation='none' semantics — but
        # LinearActFunc requires a real activation, so we manually add bias2.
        y = linear_train(h, w2) + bias2
    return y


__all__ = ["mlp", "gated_mlp", "mlp_train"]
