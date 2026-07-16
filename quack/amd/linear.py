# Copyright (c) 2026, AMD.

"""Linear — AMDGPU port of `quack/linear.py`.

Dispatches:

  - **plain bf16/f16 linear** (no bias, no activation, shape fits splitk):
    ``quack.amd.gemm_splitk`` → ~1.05-1.12× hipBLASLt at 4096²+
  - **plain MX-FP8 linear** (fp8 input + scales): ``quack.amd.mxfp8_gemm``
    → 1.5 PFLOPS at 8192³
  - **linear with bias/activation/residual**: falls back to ``quack.amd.gemm``
    which has the epilogue-capable NN MFMA kernel.

``weight`` layout is ``(out_features, in_features)`` matching
``torch.nn.Linear`` — this is NT w.r.t. the matmul, which matches
``gemm_splitk`` / ``mxfp8_gemm`` natively (no transpose copy needed).
"""

from typing import Optional

import torch
from torch import Tensor

from quack.amd.gemm import gemm


_SPLITK_ACTIVATIONS = {"relu", "relu_sq", "gelu_tanh_approx", "silu"}


def _splitk_eligible(
    x: Tensor, weight: Tensor, bias, activation,
) -> bool:
    """True iff (x, weight, bias, activation) can route directly through
    ``gemm_splitk`` with its fused-epilogue path."""
    if activation is not None and activation not in _SPLITK_ACTIVATIONS:
        return False
    if bias is not None and bias.dtype != torch.float32:
        # Fused bias path is f32-only; cast callers go through NN fallback.
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    if weight.dtype != x.dtype:
        return False
    if x.stride(-1) != 1 or weight.stride(-1) != 1:
        return False
    if x.dim() != 2 or weight.dim() != 2:
        return False
    M, K = x.shape
    N, K2 = weight.shape
    if K != K2:
        return False
    # gemm_splitk's default config uses tile_m=128, tile_n=256, tile_k=64.
    if not (M % 128 == 0 and N % 256 == 0 and K % 64 == 0 and M >= 128):
        return False
    if bias is not None and bias.shape != (N,):
        return False
    return True


def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
) -> Tensor:
    """``y = act(x @ weight.T + bias)``.

    ``weight`` is ``(out_features, in_features)`` matching ``torch.nn.Linear``.

    Fast path: when inputs are bf16/f16 with aligned shapes and the
    activation is one of ``{relu, relu_sq, gelu_tanh_approx, silu}``,
    routes through ``gemm_splitk`` with a fused epilogue — bias and
    activation are applied inside the matmul write-back in one kernel.
    Otherwise falls back to the NN ``gemm`` path which also supports
    the full epilogue matrix.
    """
    if _splitk_eligible(x, weight, bias, activation):
        from quack.amd.gemm_gfx950_splitk import gemm_splitk
        return gemm_splitk(
            x, weight, bias=bias,
            activation=activation if activation is not None else "none",
        )
    # Epilogue or non-aligned shape → NN kernel via transpose.
    w_t = weight.transpose(-1, -2).contiguous()
    return gemm(x, w_t, bias=bias, activation=activation)


_GATED_ACTIVATIONS = {"swiglu", "reglu", "geglu", "glu"}


def linear_gated(
    x: Tensor,
    weight_gate_up: Tensor,
    gate_type: str = "swiglu",
    bias: Optional[Tensor] = None,
    *,
    weight_interleaved: bool = False,
) -> Tensor:
    """Gated linear: ``y = gate(linear(x, w_gate_up, bias))``.

    ``weight_gate_up`` is ``(2 * hidden, in_features)`` — the first
    ``hidden`` rows produce the gate values, the remaining ``hidden``
    produce the up values. Output is ``(M, hidden)``.

    Gate types:
      - ``"swiglu"``: ``silu(gate) * up``
      - ``"reglu"``:  ``relu(gate) * up``
      - ``"geglu"``:  ``gelu_tanh_approx(gate) * up``
      - ``"glu"``:    ``sigmoid(gate) * up``

    Fast path — **true in-kernel fusion** via ``gemm_splitk(gate_type=...)``:
      - Caller must pre-interleave ``weight_gate_up`` so that rows
        ``[2i, 2i+1]`` are ``(gate_i, up_i)`` using
        ``quack.amd.gemm_gfx950_splitk.interleave_gated_weight``. When
        ``weight_interleaved=True`` is passed, we use this fast path.
      - The kernel writes only ``(M, hidden)`` to HBM; no intermediate
        ``(M, 2*hidden)`` materialisation. Cuts HBM traffic ~2× on the
        output side of the matmul.

    Fallback path — split-halves + torch-side gating:
      - When ``weight_interleaved=False`` (default) or shape isn't
        splitk-eligible. Runs the full ``(M, 2*hidden)`` matmul then
        applies the gate in a torch elementwise kernel.

    Bias is applied per-column of the matmul output (before gating)
    in both paths. For the fused path, bias shape must be
    ``(2*hidden,)`` in INTERLEAVED order matching the weight — call
    ``interleave_gated_weight(bias.unsqueeze(-1)).squeeze(-1)`` or
    interleave manually.
    """
    assert gate_type in _GATED_ACTIVATIONS, (
        f"gate_type must be one of {_GATED_ACTIVATIONS}, got {gate_type!r}"
    )
    two_hidden, in_features = weight_gate_up.shape
    assert two_hidden % 2 == 0, (
        f"weight_gate_up's first dim must be even (= 2*hidden), got {two_hidden}"
    )

    if weight_interleaved and _splitk_eligible(
        x, weight_gate_up, bias, activation=None,
    ):
        # Fused fast path — writes (M, hidden) directly.
        from quack.amd.gemm_gfx950_splitk import gemm_splitk
        return gemm_splitk(x, weight_gate_up, bias=bias, gate_type=gate_type)

    # Split-halves fallback: full matmul + torch gating.
    if _splitk_eligible(x, weight_gate_up, bias, activation=None):
        from quack.amd.gemm_gfx950_splitk import gemm_splitk
        out = gemm_splitk(x, weight_gate_up, bias=bias)
    else:
        w_t = weight_gate_up.transpose(-1, -2).contiguous()
        out = gemm(x, w_t, bias=bias)

    gate, up = out.chunk(2, dim=-1)
    if gate_type == "swiglu":
        return torch.nn.functional.silu(gate) * up
    if gate_type == "reglu":
        return torch.relu(gate) * up
    if gate_type == "geglu":
        return torch.nn.functional.gelu(gate, approximate="tanh") * up
    if gate_type == "glu":
        return torch.sigmoid(gate) * up
    raise NotImplementedError(f"gate_type={gate_type!r}")


def linear_mxfp8(
    x: Tensor,
    weight: Tensor,
    scale_x: Tensor,
    scale_w: Tensor,
    *,
    weight_shuffled: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
) -> Tensor:
    """MX-FP8 linear: ``y = activation(x @ weight.T + bias)`` with per-block fp8 scales.

    Args:
      x:              (M, K) fp8_e4m3fn. Row-major, last-dim contiguous.
      weight:         (N, K) fp8_e4m3fn. Either plain row-major or pre-shuffled
                      via ``quack.amd.gemm_gfx950_blockscaled.shuffle_b``
                      (set ``weight_shuffled=True`` to skip the in-launcher shuffle).
      scale_x:        (K//128, M) f32. ``scale_x[block_k, m]``.
      scale_w:        (N//128, K//128) f32. ``scale_w[block_n, block_k]``.
      weight_shuffled: if True, ``weight`` is already in the kernel's preshuffled form.
      out_dtype:      bf16 or f16.
      bias:           optional (N,) f32 bias, fused into the kernel's
                      writeback (added in f32 before trunc_f → bf16).
      activation:     optional ``"relu" | "relu_sq" | "silu" | "gelu_tanh_approx"``.
                      Fused post-bias in the writeback.

    Returns: (M, N) tensor in ``out_dtype``.

    Constraints: M % 32 == 0, N % 128 == 0, K % 128 == 0.
    """
    from quack.amd.gemm_gfx950_blockscaled import mxfp8_gemm
    return mxfp8_gemm(
        x, weight, scale_x, scale_w,
        shuffled=weight_shuffled, out_dtype=out_dtype,
        bias=bias,
        activation=activation if activation is not None else "none",
    )


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


__all__ = [
    "linear",
    "linear_gated",
    "linear_mxfp8",
    "linear_residual",
    "LinearFunc",
    "linear_train",
    "LinearActFunc",
    "linear_act_train",
]


# ---------------------------------------------------------------------------
# Training autograd Function
# ---------------------------------------------------------------------------
#
# Wires the three GEMM layouts (NT / NN / TN) to fwd / DX / DW:
#   fwd: y = x @ W.T           → gemm_splitk (NT, existing — K-inner both)
#   DX:  dx = dy @ W           → gemm_nn (Phase 3)
#   DW:  dW = dy.T @ x         → gemm_tn (Phase 4)
#
# Without this autograd Function, calling ``linear(x, W).backward()`` goes
# through torch's default autograd (which doesn't know about our custom ops)
# and gradients either error out or fall back to an unoptimised path.
# ``LinearFunc`` binds bwd to our NN/TN kernels directly.


def _compute_dweight_maybe_fused(ctx, dout2: Tensor, x: Tensor) -> Tensor:
    """Compute ``dW = dout2.T @ x`` (TN), fusing the grad-accumulation into
    ``weight.grad`` in-place when ``ctx.fuse_grad_accum`` is set.

    Mirrors NVIDIA ``quack/linear.py``: when fusing, accumulate straight into
    the existing ``weight_og.grad`` buffer via ``gemm_tn(accumulate=True)``,
    then return that buffer as ``dweight`` and null out ``weight_og.grad`` — so
    PyTorch's autograd engine does not *also* add ``dweight`` on top (which
    would double-count). Falls back to a fresh ``dW`` on the first backward
    (``grad is None``) or under ``torch.compile`` (dynamo can't trace the
    saved-tensor ``.grad`` mutation).
    """
    from quack.amd.gemm_gfx950_tn import gemm_tn

    weight_og = getattr(ctx, "weight_og", None)
    if (
        getattr(ctx, "fuse_grad_accum", False)
        and weight_og is not None
        and weight_og.grad is not None
        and not torch.compiler.is_compiling()
    ):
        gemm_tn(dout2, x, out=weight_og.grad, accumulate=True)
        dweight = weight_og.grad
        weight_og.grad = None
        return dweight
    return gemm_tn(dout2, x)


class LinearFunc(torch.autograd.Function):
    """Autograd Function for plain linear (no bias, no activation).

    Shape conventions match ``torch.nn.Linear``:
      - ``x``: (..., in_features)
      - ``weight``: (out_features, in_features)
      - returns: (..., out_features)

    Bias-fused and activation-fused variants are Phase 6 (fused epilogues).
    """

    @staticmethod
    def forward(ctx, x, weight, fuse_grad_accum=False):
        from quack.amd.gemm_gfx950_splitk import gemm_splitk
        batch_shape = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        # NT fwd: y = x @ W.T where W is (out, in)
        out = gemm_splitk(x2, weight)
        ctx.save_for_backward(x2, weight)
        ctx.batch_shape = batch_shape
        ctx.fuse_grad_accum = fuse_grad_accum
        # Stash the original leaf weight (whose .grad the optimizer reads) so
        # backward can accumulate dW straight into weight.grad.
        ctx.weight_og = weight if fuse_grad_accum else None
        return out.reshape(*batch_shape, out.shape[-1])

    @staticmethod
    def backward(ctx, dout):
        from quack.amd.gemm_gfx950_nn import gemm_nn

        x, weight = ctx.saved_tensors
        dout2 = dout.reshape(-1, dout.shape[-1]).contiguous()

        dx = None
        if ctx.needs_input_grad[0]:
            # NN: dx = dout @ W. dout is (bs, out) row-major (K=out inner);
            # W stored (out, in) row-major (in inner) — matches gemm_nn's
            # (K, N) interpretation of B.
            dx = gemm_nn(dout2, weight)
            dx = dx.reshape(*ctx.batch_shape, dx.shape[-1])

        dweight = None
        if ctx.needs_input_grad[1]:
            # TN: dW = dout.T @ x. gemm_tn treats axis 0 of both operands
            # as contraction (= batch dim here); output is (out, in).
            dweight = _compute_dweight_maybe_fused(ctx, dout2, x)

        return dx, dweight, None


def linear_train(x: Tensor, weight: Tensor, fuse_grad_accum: bool = False) -> Tensor:
    """Autograd-aware plain linear (``y = x @ W.T``) for training.

    Routes forward through ``gemm_splitk`` (NT) and backward through
    ``gemm_nn`` (DX) + ``gemm_tn`` (DW) via ``LinearFunc``. Use this when
    ``torch.is_grad_enabled()`` and no bias/activation; the simpler
    ``linear(...)`` entry above is forward-only and faster for inference.

    ``fuse_grad_accum=True`` accumulates ``dW`` directly into ``weight.grad``
    in-place (via ``gemm_tn(accumulate=True)``) instead of returning a fresh
    tensor for autograd to add — saving a full ``(out, in)`` allocation +
    separate add per micro-batch step. Only active from the *second* backward
    onward (when ``weight.grad`` already exists) and outside ``torch.compile``.

    Constraints (MVP — matches the Phase 3/4 kernel constraints):
      - dtype ∈ {f16, bf16}
      - x last dim (in_features) % 64 == 0
      - weight.shape[0] (out_features) divisible by 256 (for DX's BLOCK_N=256)
      - batch dim (after flatten) % 128 == 0
    """
    return LinearFunc.apply(x, weight, fuse_grad_accum)


# ---------------------------------------------------------------------------
# Activation-fused linear autograd
# ---------------------------------------------------------------------------


def _act_fn(activation: str):
    """Return the torch callable for an activation name."""
    if activation == "relu":
        return torch.relu
    if activation == "relu_sq":
        return lambda p: torch.relu(p) ** 2
    if activation == "gelu_tanh_approx":
        return lambda p: torch.nn.functional.gelu(p, approximate="tanh")
    if activation == "silu":
        return torch.nn.functional.silu
    raise ValueError(f"unsupported activation: {activation!r}")


def _dact_mul(dout: Tensor, preact: Tensor, activation: str) -> Tensor:
    """Compute ``dout * activation'(preact)`` elementwise.

    Uses closed-form derivatives (works inside torch.autograd's no-grad
    backward context). All outputs match ``dout``'s dtype.
    """
    p = preact.to(dout.dtype)
    if activation == "relu":
        return dout * (p > 0).to(dout.dtype)
    if activation == "relu_sq":
        # d/dp (relu(p) * p) = 2*p * (p > 0)
        return dout * (2.0 * p * (p > 0).to(dout.dtype))
    if activation == "silu":
        # silu(p) = p * sigmoid(p); silu'(p) = sigmoid(p) * (1 + p*(1 - sigmoid(p)))
        sig = torch.sigmoid(p)
        return dout * (sig * (1.0 + p * (1.0 - sig)))
    if activation == "gelu_tanh_approx":
        # gelu_tanh(p) = 0.5 * p * (1 + tanh(z))  where z = sqrt(2/pi) * (p + 0.044715*p^3)
        # d/dp = 0.5*(1 + tanh(z)) + 0.5*p*sech^2(z) * dz/dp
        #      = 0.5*(1 + tanh(z)) + 0.5*p*(1 - tanh(z)^2) * sqrt(2/pi)*(1 + 3*0.044715*p^2)
        import math as _m
        c1 = _m.sqrt(2.0 / _m.pi)
        c2 = 0.044715 * c1
        three_c2 = 3.0 * 0.044715 * c1
        p_sq = p * p
        z = p * (c1 + c2 * p_sq)
        tanh_z = torch.tanh(z)
        sech2 = 1.0 - tanh_z * tanh_z
        dz_dp = c1 + three_c2 * p_sq
        deriv = 0.5 * (1.0 + tanh_z) + 0.5 * p * sech2 * dz_dp
        return dout * deriv
    raise ValueError(f"unsupported activation: {activation!r}")


class LinearActFunc(torch.autograd.Function):
    """Autograd Function for linear + activation: ``y = act(x @ W.T + b)``.

    Saves ``preact`` for the backward pass (needed for ``act'(preact)``).
    For inputs requiring grad, backward computes:
      dpreact = dy * act'(preact)     (elementwise, torch op)
      dx      = dpreact @ W           (gemm_nn)
      dW      = dpreact.T @ x         (gemm_tn)
      dbias   = dpreact.sum(0)        (reduction, torch op)
    """

    @staticmethod
    def forward(ctx, x, weight, bias, activation, fuse_grad_accum=False):
        from quack.amd.gemm_gfx950_splitk import gemm_splitk
        batch_shape = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        # Run splitk WITHOUT activation fusion so we get preact. The
        # activation costs one elementwise pass (cheap vs the matmul).
        preact = gemm_splitk(x2, weight, bias=bias)
        postact = _act_fn(activation)(preact)
        ctx.save_for_backward(x2, weight, preact)
        ctx.activation = activation
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.batch_shape = batch_shape
        ctx.fuse_grad_accum = fuse_grad_accum
        ctx.weight_og = weight if fuse_grad_accum else None
        return postact.reshape(*batch_shape, postact.shape[-1])

    @staticmethod
    def backward(ctx, dout):
        from quack.amd.gemm_gfx950_nn import gemm_nn

        x, weight, preact = ctx.saved_tensors
        dout2 = dout.reshape(-1, dout.shape[-1]).contiguous()
        dpreact = _dact_mul(dout2, preact, ctx.activation)

        dx = None
        if ctx.needs_input_grad[0]:
            dx = gemm_nn(dpreact, weight)
            dx = dx.reshape(*ctx.batch_shape, dx.shape[-1])

        dweight = None
        if ctx.needs_input_grad[1]:
            dweight = _compute_dweight_maybe_fused(ctx, dpreact, x)

        dbias = None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            # Accumulate bias grad in f32 — bf16/f16 batch-sum introduces
            # ~2.0 error over bs=256. Then cast to bias's saved dtype.
            dbias = dpreact.sum(0, dtype=torch.float32).to(ctx.bias_dtype)

        # activation (4th) and fuse_grad_accum (5th) args have no grad
        return dx, dweight, dbias, None, None


def linear_act_train(
    x: Tensor,
    weight: Tensor,
    activation: str,
    bias: Optional[Tensor] = None,
    fuse_grad_accum: bool = False,
) -> Tensor:
    """Autograd-aware ``y = act(x @ W.T + b)``.

    Fwd uses gemm_splitk + torch activation (preact saved for bwd).
    Bwd: dpreact via torch elementwise, dx/dW via gemm_nn/gemm_tn.

    ``fuse_grad_accum=True`` accumulates ``dW`` into ``weight.grad`` in-place
    (second backward onward, outside ``torch.compile``); see ``linear_train``.
    """
    return LinearActFunc.apply(x, weight, bias, activation, fuse_grad_accum)
