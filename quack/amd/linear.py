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


def _splitk_eligible(
    x: Tensor, weight: Tensor, bias, activation,
) -> bool:
    """True iff (x, weight) can route directly through ``gemm_splitk``."""
    if bias is not None or activation is not None:
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
    return M % 128 == 0 and N % 256 == 0 and K % 64 == 0 and M >= 128


def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
) -> Tensor:
    """``y = x @ weight.T + bias`` then optional activation.

    ``weight`` is ``(out_features, in_features)`` matching ``torch.nn.Linear``.
    """
    if _splitk_eligible(x, weight, bias, activation):
        # Fast path: NT bf16/f16 GEMM via the stream-K-capable kernel.
        from quack.amd.gemm_gfx950_splitk import gemm_splitk
        return gemm_splitk(x, weight)
    # Epilogue or non-aligned shape → NN kernel via transpose.
    w_t = weight.transpose(-1, -2).contiguous()
    return gemm(x, w_t, bias=bias, activation=activation)


def linear_mxfp8(
    x: Tensor,
    weight: Tensor,
    scale_x: Tensor,
    scale_w: Tensor,
    *,
    weight_shuffled: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """MX-FP8 linear: ``y = x @ weight.T`` with per-block fp8 scales.

    Args:
      x:              (M, K) fp8_e4m3fn. Row-major, last-dim contiguous.
      weight:         (N, K) fp8_e4m3fn. Either plain row-major or pre-shuffled
                      via ``quack.amd.gemm_gfx950_blockscaled.shuffle_b``
                      (set ``weight_shuffled=True`` to skip the in-launcher shuffle).
      scale_x:        (K//128, M) f32. ``scale_x[block_k, m]``.
      scale_w:        (N//128, K//128) f32. ``scale_w[block_n, block_k]``.
      weight_shuffled: if True, ``weight`` is already in the kernel's preshuffled form.
      out_dtype:      bf16 or f16.

    Returns: (M, N) tensor in ``out_dtype``.

    Constraints: M % 32 == 0, N % 128 == 0, K % 128 == 0.
    """
    from quack.amd.gemm_gfx950_blockscaled import mxfp8_gemm
    return mxfp8_gemm(
        x, weight, scale_x, scale_w,
        shuffled=weight_shuffled, out_dtype=out_dtype,
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


__all__ = ["linear", "linear_mxfp8", "linear_residual"]
