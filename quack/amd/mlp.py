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

from quack.amd.linear import linear, linear_gated


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
    """Autograd-aware two-layer MLP: ``out = act(x @ w1.T + b1) @ w2.T + b2``.

    Delegates to ``mlp_func_train`` — a single fused autograd Function whose
    backward fuses the activation gradient into the ``dout @ w2`` matmul
    (``gemm_splitk`` dact epilogue) for eligible bf16 shapes, and uses robust
    matmuls otherwise. This is the one source of truth for two-layer MLP
    training; it replaces the earlier ``linear_act_train`` + ``linear_train``
    composition, which routed backward through ``gemm_nn`` / ``gemm_tn`` and
    crashed at large shapes via the ``gemm_gfx950_nn_big`` tile-swizzle drift.
    It also handles ``bias2`` natively (the old path added it out-of-kernel).

    Weights are ``(out, in)`` like ``torch.nn.Linear``.
    """
    from quack.amd.linear_training import mlp_func_train

    return mlp_func_train(x, w1, w2, activation=activation, bias1=bias1, bias2=bias2)


__all__ = ["mlp", "gated_mlp", "mlp_train"]
