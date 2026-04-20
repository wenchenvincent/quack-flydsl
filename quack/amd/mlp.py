# Copyright (c) 2026, AMD.

"""MLP — AMDGPU port of `quack/mlp.py`.

Thin composition of two ``quack.amd.linear`` calls with an activation or
gated activation between them.
"""

from typing import Optional

import torch
from torch import Tensor

from quack.amd.linear import linear
from quack.amd.gemm import gemm_gated


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
    """
    # Fuse x @ w_gate_up.T + bias + gate(...)
    h = gemm_gated(x, w_gate_up.transpose(-1, -2), gate_type=gate_type, bias=bias_gate_up)
    return linear(h, w_down, bias=bias_down)


__all__ = ["mlp", "gated_mlp"]
