# Copyright (c) 2026, AMD.

"""MX-FP8 ops — quantisation + autograd-integrated linear.

Provides:
  - ``quantize_mxfp8(x)``: split a bf16/f16 tensor into fp8 values + per
    128-K-block f32 scales matching the MX-FP8 standard.
  - ``linear_mxfp8_func(x, weight)``: autograd-integrated linear that
    runs the forward in fp8 via the native ``mfma_scale`` kernel and
    the backward in bf16 (standard mixed-precision MX-FP8 training
    pattern — gradients need full precision for stability).

The quantisation math:
    scale_block = 128 along the K axis.
    For each block, ``scale = max_abs / FP8_MAX`` (FP8 E4M3 max = 448.0).
    ``x_fp8 = round(x / scale).to(fp8_e4m3fn)``.
    ``x_bf16 = x_fp8.float() * scale`` (dequantisation).

Current quantisation is implemented in torch ops (not a dedicated
kernel) — cheap relative to the matmul so it's not on the critical
path. A dedicated quantiser kernel would eliminate two (M, K) traversals
per call; tracked as a follow-up.

Backward uses ``gemm_splitk`` (bf16 NT) for ``dx`` and a torch ``mm``
for ``dw`` — the weight-grad shape (N, M) @ (M, K) isn't the splitk
NT-layout shape.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from quack.amd.gemm_gfx950_blockscaled import mxfp8_gemm, SCALE_BLOCK_K


# E4M3 FN: max representable value is 448.0 (the FN variant excludes
# NaN at the max exponent). This is the standard MX-FP8 element dtype.
_FP8_E4M3_MAX = 448.0


def quantize_mxfp8(x: Tensor, *, transpose_scale: bool = False) -> Tuple[Tensor, Tensor]:
    """Quantise ``x`` (bf16/f16/f32, shape (R, K)) to fp8_e4m3fn + f32 scales.

    Returns ``(x_fp8, scale)`` where:
      - ``x_fp8``: (R, K) fp8_e4m3fn.
      - ``scale``: (K//128, R) f32 if ``transpose_scale`` (A-path layout),
                   else (R//128, K//128) requires separate handling — not
                   this path. Plain ``(R, K//128)`` f32 otherwise.

    ``K`` must be a multiple of 128.
    """
    R, K = x.shape
    assert K % SCALE_BLOCK_K == 0, f"K={K} must be a multiple of {SCALE_BLOCK_K}"
    # Split K into (num_blocks, block_size). Find max |x| per block.
    x_blocked = x.float().view(R, K // SCALE_BLOCK_K, SCALE_BLOCK_K)
    # amax along the block dim; (R, num_blocks).
    amax = x_blocked.abs().amax(dim=-1).clamp_min(1e-12)
    scale = amax / _FP8_E4M3_MAX           # (R, num_blocks)
    # Divide then cast (torch handles fp8 E4M3 saturation).
    x_scaled = x_blocked / scale.unsqueeze(-1)
    x_fp8 = x_scaled.view(R, K).to(torch.float8_e4m3fn)
    if transpose_scale:
        # A-path: mxfp8_gemm expects scale_a shape (K//128, M) indexed [block_k, m].
        scale_out = scale.transpose(0, 1).contiguous()
    else:
        scale_out = scale.contiguous()
    return x_fp8, scale_out


def _quantize_weight_with_block_scale(w: Tensor) -> Tuple[Tensor, Tensor]:
    """Quantise weight ``(N, K)`` into fp8 + scale ``(N//128, K//128)``.

    MX-FP8 scale_b indexes per-128-N × per-128-K block (not per-N-row
    like scale_a). We take ``max_abs`` over the 128×128 block.
    """
    N, K = w.shape
    assert N % 128 == 0 and K % SCALE_BLOCK_K == 0
    # Reshape to (N//128, 128, K//128, 128) then find max over the 128×128 block.
    w_blocked = w.float().view(
        N // 128, 128, K // SCALE_BLOCK_K, SCALE_BLOCK_K,
    )
    # amax over the inner (128, 128) dims.
    amax = w_blocked.abs().amax(dim=(1, 3)).clamp_min(1e-12)  # (N//128, K//128)
    scale = amax / _FP8_E4M3_MAX
    # Dequantise denominator broadcast: (N//128, 1, K//128, 1).
    w_scaled = w_blocked / scale.view(N // 128, 1, K // SCALE_BLOCK_K, 1)
    w_fp8 = w_scaled.view(N, K).to(torch.float8_e4m3fn)
    return w_fp8, scale.contiguous()


class _LinearMXFP8Function(torch.autograd.Function):
    """Autograd wrapper: MX-FP8 forward, bf16 backward.

    Forward: quantise x (per-K-block) and w (per-128×128-block) to fp8,
    call the native ``mfma_scale`` kernel, return bf16 output.

    Backward: use bf16 matmul against the original (pre-quantisation)
    x and weight tensors. This is the standard MX-FP8 training
    convention — gradient precision matters more than compute density
    for convergence.
    """

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias, activation):
        ctx.x_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.activation = activation
        ctx.has_bias = bias is not None
        is_activated = activation is not None and activation != "none"
        # Quantise on the fly. Scale layouts match mxfp8_gemm's contract.
        x_fp8, scale_x = quantize_mxfp8(x, transpose_scale=True)   # (K//128, M)
        w_fp8, scale_w = _quantize_weight_with_block_scale(weight)  # (N//128, K//128)
        # Unfused forward: get the pre-activation from the kernel (bias
        # stays fused since it has no bwd data dependency), then apply
        # the activation on torch side. Saving preact lets the backward
        # compute the exact per-sample act_prime without recompute.
        # Fused activation in the kernel would 2× either fwd or bwd —
        # the unfused path is the same 3 matmuls as the naive pattern,
        # with the bias add amortised into the kernel. See NV QuACK
        # ``MLPRecomputeFunc`` for the alternative (preact-recompute
        # on bwd) — not measured faster at AMD shapes given the mxfp8
        # kernel's current ~32% peak efficiency.
        preact = mxfp8_gemm(
            x_fp8, w_fp8, scale_x, scale_w, out_dtype=x.dtype,
            bias=bias,
            activation="none",
        )
        if is_activated:
            if activation == "relu":
                out = torch.relu(preact)
            elif activation == "silu":
                out = torch.nn.functional.silu(preact)
            elif activation == "relu_sq":
                out = torch.relu(preact) * preact
            elif activation == "gelu_tanh_approx":
                out = torch.nn.functional.gelu(preact, approximate="tanh")
            else:
                raise ValueError(f"unknown activation {activation!r}")
        else:
            out = preact
        ctx.save_for_backward(
            x, weight,
            bias if bias is not None else torch.empty(0),
            preact if is_activated else torch.empty(0),
        )
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, weight, bias_saved, preact = ctx.saved_tensors
        # dpreact = grad_out * act'(preact)  (elementwise, in f32).
        # When activation is None, dpreact = grad_out directly.
        if ctx.activation is None or ctx.activation == "none":
            dpreact = grad_out
        else:
            p32 = preact.float()
            g32 = grad_out.float()
            if ctx.activation == "relu":
                deriv = (p32 > 0).float()
            elif ctx.activation == "relu_sq":
                # d/dx[relu(x)*x] = 2*x * (x > 0)
                deriv = 2.0 * p32 * (p32 > 0).float()
            elif ctx.activation == "silu":
                sig = torch.sigmoid(p32)
                deriv = sig * (1.0 + p32 * (1.0 - sig))
            elif ctx.activation == "gelu_tanh_approx":
                import math as _m
                c1 = _m.sqrt(2.0 / _m.pi)
                z = c1 * (p32 + 0.044715 * p32.pow(3))
                th = torch.tanh(z)
                dz = c1 * (1.0 + 3.0 * 0.044715 * p32.pow(2))
                deriv = 0.5 * (1.0 + th) + 0.5 * p32 * (1.0 - th * th) * dz
            else:
                raise ValueError(f"unknown activation {ctx.activation!r}")
            dpreact = (g32 * deriv).to(grad_out.dtype)
        # dx = dpreact @ weight  — NN matmul, (M, N) × (N, K) → (M, K).
        # dw = dpreact.T @ x     — NN matmul, (N, M) × (M, K) → (N, K).
        # torch.mm / hipBLASLt handle these at ~2× fp8 compute density
        # (bf16 matrix-core throughput). Training stays numerically stable
        # because dpreact arrived in bf16, not fp8.
        grad_x = torch.mm(dpreact, weight) if ctx.needs_input_grad[0] else None
        grad_w = torch.mm(dpreact.t(), x) if ctx.needs_input_grad[1] else None
        grad_bias = None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = dpreact.sum(dim=0).to(torch.float32)
        return grad_x, grad_w, grad_bias, None


def linear_mxfp8_func(
    x: Tensor, weight: Tensor,
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
) -> Tensor:
    """MX-FP8 linear with autograd: ``y = activation(x @ weight.T + bias)``.

    Drop-in for ``torch.nn.functional.linear(x, weight, bias)`` when both
    inputs are bf16/f16 and shape-aligned (M%32==0, N%128==0, K%128==0).
    Forward runs the native ``mfma_scale_f32_16x16x128_f8f6f4`` kernel
    via on-the-fly MX-FP8 quantisation with fused bias in the writeback;
    backward runs bf16 matmul.

    Accepts tensors with ``requires_grad=True`` and participates in the
    autograd graph — no manual scale plumbing needed.

    Training with ``activation``: the autograd forward uses the unfused
    kernel path (fused bias only; activation torch-side) and saves the
    preact tensor so backward can compute ``act'(preact)`` exactly. This
    matches the total-matmul count of the naive
    ``F.silu(F.linear(x, w, b))`` pattern at the torch level (3 matmuls
    for fwd + bwd). The fused-kernel activation path in
    ``mxfp8_gemm(..., activation=...)`` is still available for
    no-grad / inference use where it saves one kernel launch per call.

    Supported activations: relu, relu_sq, silu, gelu_tanh_approx.

    Limitations:
      - x and weight must be the same dtype (bf16 or f16). Mixed-precision
        across them isn't supported.
    """
    assert x.is_cuda and weight.is_cuda
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert weight.dtype == x.dtype
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    if bias is not None:
        assert bias.dim() == 1 and bias.size(0) == N and bias.dtype == torch.float32
    return _LinearMXFP8Function.apply(x, weight, bias, activation)


__all__ = ["linear_mxfp8_func", "quantize_mxfp8"]
