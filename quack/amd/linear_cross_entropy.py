# Copyright (c) 2026, AMD.

"""Fused linear + cross-entropy — AMDGPU port of `quack/linear_cross_entropy.py`.

Thin composition of ``quack.amd.linear`` and ``quack.amd.cross_entropy``.
Fusing the projection with the loss computation (QuACK's NVIDIA-side
optimisation) is a follow-up.
"""

from typing import Optional, Tuple

from torch import Tensor

from quack.amd.linear import linear
from quack.amd.cross_entropy import cross_entropy_fwd


def linear_cross_entropy(
    x: Tensor,
    weight: Tensor,
    target: Tensor,
    bias: Optional[Tensor] = None,
    return_lse: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Compute ``logits = x @ weight.T + bias``, then per-row cross-entropy.

    Returns ``(loss, lse_or_None)`` matching QuACK's NVIDIA-side public shape.
    """
    logits = linear(x, weight, bias=bias)
    return cross_entropy_fwd(logits, target, return_lse=return_lse)


__all__ = ["linear_cross_entropy"]
