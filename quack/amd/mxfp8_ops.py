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
        # Save pre-quant bf16 tensors for backward.
        ctx.save_for_backward(x, weight, bias if bias is not None else torch.empty(0))
        ctx.x_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.activation = activation
        ctx.has_bias = bias is not None
        # Quantise on the fly. Scale layouts match mxfp8_gemm's contract.
        x_fp8, scale_x = quantize_mxfp8(x, transpose_scale=True)  # (K//128, M)
        w_fp8, scale_w = _quantize_weight_with_block_scale(weight)  # (N//128, K//128)
        preact = mxfp8_gemm(
            x_fp8, w_fp8, scale_x, scale_w, out_dtype=x.dtype,
            bias=bias, activation=activation or "none",
        )
        if activation is not None and activation != "none":
            # Save pre-activation for activation-backward. When activation
            # is None, preact == output and we skip the save (bwd is just
            # two matmuls).
            #
            # For the fused path mxfp8_gemm returns the POST-activation
            # output — so the "preact" we need for bwd is out_of_linear =
            # preact = linear(x, w, bias). Recompute by running mxfp8_gemm
            # without activation would 2× the forward cost. Alternative:
            # back-apply inverse-activation. Neither is great.
            #
            # Pragmatic choice: save the POST-activation ``y`` and in
            # backward, recompute the pre-activation via a cheap torch
            # pass. swiglu/silu/gelu_tanh aren't easily invertible, so
            # we actually save ``y`` and a small extra kernel launch to
            # derive ``dy_preact = act_bwd(y, grad_out)``. That's a
            # follow-up. For now: the kernel epi is used on forward for
            # fast inference; training with activation dispatches to the
            # unfused path (bias only in the kernel, activation torch-side).
            raise NotImplementedError(
                "training-mode activation not wired through the fused "
                "epi path yet (it needs preact-recompute infra). Call "
                "mxfp8_gemm(..., activation=...) directly for inference."
            )
        return preact

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, weight, bias_saved = ctx.saved_tensors
        # grad_out is (M, N) in x.dtype (bf16 typically).
        # dx = grad_out @ weight  — NN matmul, (M, N) × (N, K) → (M, K).
        # dw = grad_out.T @ x     — NN matmul, (N, M) × (M, K) → (N, K).
        # torch.mm / hipBLASLt handle these at ~2× fp8 compute density
        # (bf16 matrix-core throughput). Training stays numerically stable
        # because grad_out arrived in bf16, not fp8.
        grad_x = torch.mm(grad_out, weight) if ctx.needs_input_grad[0] else None
        grad_w = torch.mm(grad_out.t(), x) if ctx.needs_input_grad[1] else None
        grad_bias = None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_out.sum(dim=0).to(torch.float32)
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

    Limitations:
      - ``activation`` only supported in forward/inference. The backward
        path for activated mxfp8 requires preact recompute (follow-up).
        For activation in training use ``linear(x, w, bias, activation)``
        on bf16, or ``mxfp8_gemm(..., activation=...)`` directly (no
        autograd).
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
