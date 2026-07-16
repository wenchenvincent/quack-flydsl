# Copyright (c) 2026, AMD.

"""MXFP6 quantization helpers — OCP MX standard (e2m3 element + e8m0 scale).

MXFP6 (e2m3): 6-bit elements (1 sign, 2 exp, 3 mantissa), grouped in 32-element
blocks sharing one e8m0 (power-of-2) scale. Used by ``gemm_mxfp6`` on gfx950 via
``mfma_scale_f32_16x16x128_f8f6f4`` with ``cbsz=blgp=2`` (fp6 selector).

torch has no fp6 dtype; 6-bit codes are bit-packed LSB-first (element ``i``'s 6
bits occupy bit positions ``6i .. 6i+5`` of the little-endian byte stream) — the
exact layout the MFMA reads from register i32[0..5]. Because 6 bits don't align
to bytes, the quantized axis size must be a multiple of 4 (``4*6 = 24`` bits =
3 bytes); MX blocks of 32 give ``32*6/8 = 24`` bytes, always aligned.
"""

import numpy as np
import torch
from torch import Tensor

MX_BLOCK = 32


def _e2m3_levels() -> torch.Tensor:
    """The 64 e2m3 codes → signed magnitude (index = 6-bit code s<<5|e<<3|m)."""
    vals = torch.zeros(64, dtype=torch.float32)
    for e in range(4):
        for m in range(8):
            if e == 0:
                mag = (m / 8.0)  # subnormal: (m/8) * 2^0
            else:
                mag = (1.0 + m / 8.0) * (2.0 ** (e - 1))
            code = (e << 3) | m
            vals[code] = mag
            vals[0x20 | code] = -mag  # sign bit
    return vals


_E2M3_LEVELS = _e2m3_levels()
_E2M3_MAG = _E2M3_LEVELS[:32].clone()  # magnitudes for codes 0..31 (sign=0)
_E2M3_MAX = 7.5


def _nearest_e2m3_code(x: Tensor) -> Tensor:
    """Round ``x`` (any sign) to the nearest e2m3 6-bit code (0..63)."""
    mag = x.abs()
    levels = _E2M3_MAG.to(x.device)
    d = (mag.unsqueeze(-1) - levels).abs()
    code = d.argmin(dim=-1).to(torch.uint8)  # 0..31 magnitude code
    sign = (x < 0).to(torch.uint8) << 5
    return sign | code


def quantize_mxfp6(x: Tensor, axis: int = -1):
    """Quantize ``x`` (f32/f16/bf16) along ``axis`` into MXFP6 (e2m3).

    Returns ``(packed, scales, shape)`` where ``packed`` is uint8 with 6-bit
    codes bit-packed LSB-first along the quantized axis (``K*6/8`` bytes),
    ``scales`` is uint8 e8m0 (one per 32-element block), and ``shape`` is the
    original element shape. ``x``'s size along ``axis`` must be a multiple of 32.
    """
    x = x.movedim(axis, -1).float().contiguous()
    *lead, K = x.shape
    assert K % MX_BLOCK == 0, f"axis size {K} must be a multiple of {MX_BLOCK}"
    nblk = K // MX_BLOCK
    xb = x.reshape(*lead, nblk, MX_BLOCK)

    absmax = xb.abs().amax(dim=-1)
    exp = torch.floor(torch.log2(absmax.clamp_min(1e-30))) - 2
    exp = exp.clamp(-127, 127)
    scale_byte = torch.where(absmax > 0, (exp + 127), torch.zeros_like(exp)).to(torch.uint8)
    scale = torch.pow(2.0, exp)

    q = xb / scale.unsqueeze(-1).clamp_min(1e-30)
    q = q.clamp(-_E2M3_MAX, _E2M3_MAX)
    codes = _nearest_e2m3_code(q).reshape(*lead, K)  # 6-bit codes, one per elem

    # bit-pack LSB-first: element i's 6 bits at positions 6i..6i+5.
    codes_np = codes.cpu().numpy().astype(np.uint8)
    flat = codes_np.reshape(-1, K)
    bits = ((flat[:, :, None] >> np.arange(6, dtype=np.uint8)) & 1).reshape(flat.shape[0], K * 6)
    packed = np.packbits(bits, axis=-1, bitorder="little")  # (rows, K*6/8)
    packed = torch.from_numpy(packed).to(x.device).reshape(*lead, K * 6 // 8)
    return packed, scale_byte.to(torch.uint8), tuple(x.shape)


def dequantize_mxfp6(packed: Tensor, scale_byte: Tensor, shape) -> Tensor:
    """Inverse of :func:`quantize_mxfp6` → f32 tensor of ``shape``."""
    *lead, K = shape
    packed_np = packed.cpu().numpy().astype(np.uint8).reshape(-1, K * 6 // 8)
    bits = np.unpackbits(packed_np, axis=-1, bitorder="little").reshape(-1, K, 6)
    weights = (1 << np.arange(6, dtype=np.uint32))
    codes = (bits.astype(np.uint32) * weights).sum(-1).astype(np.int64)  # 0..63
    codes_t = torch.from_numpy(codes).to(packed.device).reshape(*lead, K)
    vals = _E2M3_LEVELS.to(packed.device)[codes_t]
    scale = torch.pow(2.0, scale_byte.float() - 127.0)
    scale = torch.where(scale_byte == 0, torch.zeros_like(scale), scale)
    vals = vals.reshape(*lead, K // MX_BLOCK, MX_BLOCK) * scale.unsqueeze(-1)
    return vals.reshape(*lead, K)


__all__ = ["quantize_mxfp6", "dequantize_mxfp6", "MX_BLOCK"]
