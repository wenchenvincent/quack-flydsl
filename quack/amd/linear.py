# Copyright (c) 2026, AMD.

"""Linear — AMDGPU port of `quack/linear.py`.

Thin wrapper over ``quack.amd.gemm``. When the FlyDSL GEMM kernels land
(Phase 2 follow-up), this module will automatically pick up the perf
improvement since it just dispatches to ``gemm``.
"""

from typing import Optional

from torch import Tensor

from quack.amd.gemm import gemm


def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
) -> Tensor:
    """``y = x @ weight.T + bias`` then optional activation.

    ``weight`` is ``(out_features, in_features)`` matching ``torch.nn.Linear``.
    Materialises ``weight.T`` so the GEMM's last-dim-contiguous requirement
    is met — for repeated calls with the same weight, pass a pre-transposed
    tensor directly to ``quack.amd.gemm.gemm`` to skip the copy.
    """
    w_t = weight.transpose(-1, -2).contiguous()
    return gemm(x, w_t, bias=bias, activation=activation)


__all__ = ["linear"]
