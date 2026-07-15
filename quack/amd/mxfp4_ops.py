# Copyright (c) 2026, AMD.

"""MXFP4 quantization helpers — OCP MX standard (e2m1 element + e8m0 scale).

MXFP4: 4-bit e2m1 elements (1 sign, 2 exp, 1 mantissa), grouped in 32-element
blocks that share one e8m0 (8-bit, exponent-only, power-of-2) scale. Used by
``gemm_mxfp4`` on gfx950 via ``mfma_scale_f32_16x16x128_f8f6f4``.

torch has no fp4 dtype, so we pack two e2m1 nibbles per uint8 and carry the
e8m0 scales as a separate uint8 tensor (one per 32-element block).
"""

import torch
from torch import Tensor

# e2m1 magnitudes by code (0..7); sign bit is 0x8.
_E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MAX = 6.0
MX_BLOCK = 32


def _nearest_e2m1_code(mag: Tensor) -> Tensor:
    """Round non-negative magnitudes to the nearest e2m1 level code (0..7)."""
    levels = _E2M1_LEVELS.to(mag.device)
    # nearest level index per element
    d = (mag.unsqueeze(-1) - levels).abs()
    return d.argmin(dim=-1).to(torch.uint8)


def quantize_mxfp4(x: Tensor, axis: int = -1):
    """Quantize ``x`` (f32/f16/bf16) along ``axis`` into MXFP4.

    Returns ``(packed, scales, shape)`` where ``packed`` is uint8 with two
    e2m1 nibbles per byte along the quantized axis (low nibble = even index),
    ``scales`` is uint8 e8m0 (one per 32-element block along ``axis``), and
    ``shape`` is the original element shape. ``x``'s size along ``axis`` must
    be a multiple of 32.
    """
    x = x.movedim(axis, -1).float().contiguous()
    *lead, K = x.shape
    assert K % MX_BLOCK == 0, f"axis size {K} must be a multiple of {MX_BLOCK}"
    nblk = K // MX_BLOCK
    xb = x.reshape(*lead, nblk, MX_BLOCK)

    absmax = xb.abs().amax(dim=-1)  # (*lead, nblk)
    # e8m0 scale exponent: target the top e2m1 level (6 = 1.5*2^2, exp 2).
    # scale = 2^(floor(log2(absmax)) - 2); clamp for zero blocks.
    exp = torch.floor(torch.log2(absmax.clamp_min(1e-30))) - 2
    exp = exp.clamp(-127, 127)
    # e8m0 byte = exp + 127 (bias). all-zero block -> exp very negative -> byte 0.
    scale_byte = torch.where(absmax > 0, (exp + 127), torch.zeros_like(exp)).to(torch.uint8)
    scale = torch.pow(2.0, exp)

    q = xb / scale.unsqueeze(-1).clamp_min(1e-30)
    q = q.clamp(-_E2M1_MAX, _E2M1_MAX)
    code = _nearest_e2m1_code(q.abs())
    sign = (q < 0).to(torch.uint8) * 8
    nib = (sign | code).reshape(*lead, K)  # e2m1 nibbles, one per element

    # pack two nibbles per byte along the last axis
    nib = nib.reshape(*lead, K // 2, 2)
    packed = (nib[..., 0] | (nib[..., 1] << 4)).to(torch.uint8)
    return packed, scale_byte.to(torch.uint8), tuple(x.shape)


def dequantize_mxfp4(packed: Tensor, scale_byte: Tensor, shape) -> Tensor:
    """Inverse of :func:`quantize_mxfp4` → f32 tensor of ``shape``."""
    *lead, K = shape
    levels = _E2M1_LEVELS.to(packed.device)
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    nib = torch.stack([lo, hi], dim=-1).reshape(*lead, K).long()
    mag = levels[nib & 0x7]
    sign = torch.where((nib & 0x8) > 0, -1.0, 1.0)
    vals = mag * sign  # (*lead, K)
    scale = torch.pow(2.0, scale_byte.float() - 127.0)  # (*lead, K/32)
    scale = torch.where(
        scale_byte.unsqueeze(-1) == 0, torch.zeros_like(scale).unsqueeze(-1), scale.unsqueeze(-1)
    )
    vals = vals.reshape(*lead, K // MX_BLOCK, MX_BLOCK) * scale
    return vals.reshape(*lead, K)


__all__ = ["quantize_mxfp4", "dequantize_mxfp4", "MX_BLOCK"]
