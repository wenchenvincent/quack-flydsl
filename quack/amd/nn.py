# Copyright (c) 2026, AMD.

"""High-level layer for the AMD reduction kernels.

Wraps the bare ``*_fwd``/``*_bwd`` kernels in ``quack.amd.{rmsnorm,softmax,
cross_entropy}`` with ``torch.autograd.Function``s, functional entry points,
and ``nn.Module``s — mirroring the NVIDIA ``quack.rmsnorm`` surface. The
kernel files stay kernel-only; this is the user-facing layer.
"""

import torch
from torch import Tensor

from quack.amd.rmsnorm import rmsnorm_fwd, rmsnorm_bwd


class RMSNormFunction(torch.autograd.Function):
    """Autograd wrapper over the AMD ``rmsnorm_fwd``/``rmsnorm_bwd`` kernels."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        need_grad = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        out, rstd, _ = rmsnorm_fwd(x, weight, eps=eps, store_rstd=need_grad)
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, rstd = ctx.saved_tensors
        dx, dw = rmsnorm_bwd(x, weight, dout.contiguous(), rstd, eps=ctx.eps)
        return dx, dw, None


def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm over the last dim, autograd-enabled. Flattens leading dims."""
    n = x.shape[-1]
    out = RMSNormFunction.apply(x.reshape(-1, n), weight, eps)
    return out.reshape(x.shape)


class RMSNorm(torch.nn.Module):
    """RMSNorm layer backed by the AMD reduction kernels."""

    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        return rmsnorm(x, self.weight, self.eps)
