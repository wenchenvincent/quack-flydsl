# Copyright (c) 2026, AMD.

"""High-level layer for the AMD reduction kernels.

Wraps the bare ``*_fwd``/``*_bwd`` kernels in ``quack.amd.{rmsnorm,softmax,
cross_entropy}`` with ``torch.autograd.Function``s, functional entry points,
and ``nn.Module``s — mirroring the NVIDIA ``quack.rmsnorm`` surface. The
kernel files stay kernel-only; this is the user-facing layer.
"""

import torch
from torch import Tensor

from quack.amd.rmsnorm import rmsnorm_fwd, rmsnorm_bwd, layernorm_fwd, layernorm_bwd
from quack.amd.softmax import softmax_fwd, softmax_bwd


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


class LayerNormFunction(torch.autograd.Function):
    """Autograd wrapper over the AMD ``layernorm_fwd``/``layernorm_bwd`` kernels."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        need_grad = any(ctx.needs_input_grad[:3])
        out, rstd, mean, _residual_out = layernorm_fwd(
            x, weight, bias=bias, eps=eps, store_stats=need_grad
        )
        ctx.save_for_backward(x, weight, bias, rstd, mean)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, bias, rstd, mean = ctx.saved_tensors
        dx, dw, db = layernorm_bwd(x, weight, dout.contiguous(), rstd, mean, bias=bias, eps=ctx.eps)
        return dx, dw, db, None


def layernorm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-6) -> Tensor:
    """LayerNorm over the last dim, autograd-enabled. Flattens leading dims."""
    n = x.shape[-1]
    out = LayerNormFunction.apply(x.reshape(-1, n), weight, bias, eps)
    return out.reshape(x.shape)


class LayerNorm(torch.nn.Module):
    """LayerNorm layer backed by the AMD reduction kernels."""

    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        return layernorm(x, self.weight, self.bias, self.eps)


class SoftmaxFunction(torch.autograd.Function):
    """Autograd wrapper for the AMD softmax kernels (2D, last-dim)."""

    @staticmethod
    def forward(ctx, x):
        y = softmax_fwd(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensors
        return softmax_bwd(dy.contiguous(), y)


def softmax(x, dim=-1):
    """Softmax over ``dim``. The AMD kernel is last-dim 2D, so move ``dim`` last."""
    if dim != -1 and dim != x.ndim - 1:
        x = x.movedim(dim, -1)
        y = softmax(x, dim=-1)
        return y.movedim(-1, dim)
    n = x.shape[-1]
    y = SoftmaxFunction.apply(x.reshape(-1, n))
    return y.reshape(x.shape)
