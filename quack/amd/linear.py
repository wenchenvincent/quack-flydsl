# Copyright (c) 2026, AMD.

"""Linear — AMDGPU port of `quack/linear.py`.

Thin wrapper over ``quack.amd.gemm``. When the FlyDSL GEMM kernels land
(Phase 2 follow-up), this module will automatically pick up the perf
improvement since it just dispatches to ``gemm``.
"""

from typing import Optional

import torch
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


def linear_residual(
    x: Tensor,
    weight: Tensor,
    residual: Tensor,
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
    alpha: float = 1.0,
    residual_scale: float = 1.0,
) -> Tensor:
    """Fused linear + residual: ``y = activation(alpha * (x @ W.T) + residual_scale * residual + bias)``.

    Common pattern in transformer residual branches — fuses the matmul
    with the residual add in one kernel, saving the (M, N) round-trip.

    When shapes are MFMA-eligible (f16/bf16 × f16/bf16 → same; M, in_f,
    out_f multiples of 16), routes through the FlyDSL MFMA kernel via
    ``gemm(alpha=alpha, beta=residual_scale, C=residual)``.

    ``residual`` must be f32 (the kernel's accumulator dtype). Callers
    with half-precision residuals should upcast via ``.float()`` before
    calling.
    """
    assert residual.dtype == torch.float32, (
        "linear_residual requires f32 residual (kernel expects f32 C)"
    )
    w_t = weight.transpose(-1, -2).contiguous()
    return gemm(
        x, w_t,
        bias=bias, activation=activation,
        alpha=alpha, beta=residual_scale, C=residual,
    )


__all__ = ["linear", "linear_residual"]
