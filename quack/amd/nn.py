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
from quack.amd.cross_entropy import cross_entropy_fwd, cross_entropy_bwd
from quack.amd.topk import topk  # noqa: F401
from quack.amd.linear import linear_train, linear_act_train
from quack.amd.linear_training import mlp_func_train
from quack.amd.linear_cross_entropy import linear_cross_entropy


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
    """RMSNorm over the last dim, autograd-enabled. Flattens leading dims.

    Note: the AMD kernel currently supports only the default ``eps=1e-6``; any
    other value raises ``NotImplementedError`` (runtime eps is a later pass).
    """
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
    """LayerNorm over the last dim, autograd-enabled. Flattens leading dims.

    Note: the AMD kernel currently supports only the default ``eps=1e-6``; any
    other value raises ``NotImplementedError`` (runtime eps is a later pass).
    """
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


class CrossEntropyFunction(torch.autograd.Function):
    """Autograd wrapper for the AMD cross-entropy kernels. Returns per-row loss."""

    @staticmethod
    def forward(ctx, x, target, ignore_index):
        loss, lse = cross_entropy_fwd(x, target, return_lse=True, ignore_index=ignore_index)
        ctx.save_for_backward(x, target, lse)
        ctx.ignore_index = ignore_index
        return loss  # per-row, float32

    @staticmethod
    def backward(ctx, dloss):
        x, target, lse = ctx.saved_tensors
        dx = cross_entropy_bwd(x, target, lse, dloss.contiguous(), ignore_index=ctx.ignore_index)
        return dx, None, None


def cross_entropy(x, target, ignore_index=-100, reduction="mean"):
    """Cross-entropy loss over the class dim, autograd-enabled.

    ``x`` is ``(..., V)`` logits and ``target`` is ``(...)`` int over the leading
    dims; leading dims are flattened like the other layers here. Reduction ∈
    {none, mean, sum}; ``none`` returns per-element loss shaped like ``target``,
    ``mean`` normalizes by the count of non-ignored elements.

    Edge case: if every element is ignored, ``mean`` returns 0.0 (via
    ``clamp_min(1)``) rather than nan as ``torch.nn.functional.cross_entropy`` does.
    """
    v = x.shape[-1]
    x2d = x.reshape(-1, v)
    tgt1d = target.reshape(-1)
    loss = CrossEntropyFunction.apply(x2d, tgt1d, ignore_index)  # per-element
    if reduction == "none":
        return loss.reshape(target.shape)
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        valid = (tgt1d != ignore_index).sum().clamp_min(1)
        return loss.sum() / valid
    raise ValueError(f"unknown reduction: {reduction!r}")


__all__ = [
    "RMSNormFunction",
    "rmsnorm",
    "RMSNorm",
    "LayerNormFunction",
    "layernorm",
    "LayerNorm",
    "SoftmaxFunction",
    "softmax",
    "CrossEntropyFunction",
    "cross_entropy",
    "topk",
    "Linear",
    "MLP",
    "LinearCrossEntropy",
]


# ---------------------------------------------------------------------------
# Linear / MLP / LinearCrossEntropy — training nn.Module wrappers
# ---------------------------------------------------------------------------
#
# Thin torch.nn.Module wrappers over the autograd-enabled functional entry
# points in quack.amd.linear / quack.amd.linear_training /
# quack.amd.linear_cross_entropy. Those modules stay kernel-only; this is
# the user-facing layer, mirroring the RMSNorm/LayerNorm/Softmax/CrossEntropy
# pattern above.


class Linear(torch.nn.Module):
    """Autograd-aware linear layer (``y = act(x @ W.T + b)``) on AMD kernels.

    ``activation=None`` is a plain linear; a string activation uses the fused
    ``linear_act_train`` path. Weights are ``(out, in)`` like ``torch.nn.Linear``.

    Shape constraints (inherited from the training kernels, bf16/f16 only):
    ``out_features`` must be a multiple of 256 and the flattened batch a
    multiple of 128 — the ``gemm_splitk`` NT tile is 128×256. Other shapes
    raise an assertion from the kernel rather than silently falling back.
    """

    def __init__(
        self, in_features, out_features, bias=True, activation=None, device=None, dtype=None,
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.bias = (
            torch.nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
            if bias
            else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.activation is not None:
            return linear_act_train(x, self.weight, self.activation, bias=self.bias)
        y = linear_train(x, self.weight)
        return y if self.bias is None else y + self.bias


class MLP(torch.nn.Module):
    """Autograd-aware two-layer MLP backed by ``mlp_func_train`` (fused-dact).

    Shape constraints for the fused backward path (bf16/f16): flattened batch
    a multiple of 128, ``hidden_features`` a multiple of 256, ``out_features``
    a multiple of 64. Non-eligible shapes still run but fall back to the
    unfused ``torch.mm`` + elementwise activation-backward.
    """

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        activation="silu",
        bias=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.activation = activation
        self.w1 = torch.nn.Parameter(
            torch.empty(hidden_features, in_features, device=device, dtype=dtype)
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(out_features, hidden_features, device=device, dtype=dtype)
        )
        self.bias1 = (
            torch.nn.Parameter(torch.zeros(hidden_features, device=device, dtype=dtype))
            if bias
            else None
        )
        self.bias2 = (
            torch.nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
            if bias
            else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.w1, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.w2, a=5**0.5)

    def forward(self, x):
        return mlp_func_train(
            x, self.w1, self.w2, activation=self.activation, bias1=self.bias1, bias2=self.bias2
        )


class LinearCrossEntropy(torch.nn.Module):
    """Fused linear + cross-entropy training layer: ``loss = CE(x @ W.T, target)``.

    Wraps ``quack.amd.linear_cross_entropy.linear_cross_entropy``, which
    routes through a fused chunked fwd+bwd kernel when ``x``/``weight``
    require grad — it never materialises the full ``(B*L, V)`` logits.

    ``bias`` is **not supported**: the grad path of ``linear_cross_entropy``
    has no gradient wrt bias (documented limitation of the fused kernel), so
    this module refuses to construct with ``bias=True`` rather than silently
    producing a layer whose bias never trains.
    """

    def __init__(
        self,
        in_features,
        num_classes,
        bias=False,
        chunk_size=None,
        *,
        ignore_index=-100,
        label_smoothing=0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert not bias, (
            "LinearCrossEntropy does not support bias: the fused grad path in "
            "linear_cross_entropy has no gradient wrt bias."
        )
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.chunk_size = chunk_size
        self.weight = torch.nn.Parameter(
            torch.empty(num_classes, in_features, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: Tensor, target: Tensor, loss_weight=None) -> Tensor:
        kwargs = dict(
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            loss_weight=loss_weight,
        )
        if self.chunk_size is not None:
            kwargs["chunk_size"] = self.chunk_size
        loss, _lse = linear_cross_entropy(x, self.weight, target, **kwargs)
        return loss
