# Copyright (c) 2026, AMD.

"""Blockscaled MX-FP8 GEMM for gfx950 (CDNA4).

**Current state: correctness-first MVP via dequant + bf16 GEMM.**

MX-FP8 / MX-FP4 is a format where a (M, K) tensor is stored as fp8
values plus per-K-block f32 scales (block size 128). The forward
operation is::

    out[m, n] = sum_k  a_fp8[m, k] * scale_a[k//128, m]
                       * b_fp8[n, k] * scale_b[n//128, k//128]

hipBLASLt doesn't cover this on ROCm 6.x, so the Fly path is the
only one available. The native intrinsic is
``rocdl.mfma_scale_f32_16x16x128_f8f6f4`` — 2× the compute density
of bf16 MFMA at the same tile.

This MVP runs the math in two stages:

1. **Dequantize** fp8 → bf16 using the per-block scales (one scale
   per 128 K elements).
2. **Run standard NT bf16 GEMM** via ``gemm_splitk`` (our MFMA
   K=32 + DMA-to-LDS port, already at ~hipBLASLt parity).

This delivers the API and correctness but not the fp8 compute-density
benefit — that requires a kernel-native ``mfma_scale_f32_16x16x128_f8f6f4``
path. The reference implementation
``FlyDSL/kernels/blockscale_preshuffle_gemm.py`` is ~880 LoC with
~1100 LoC of helper dependencies (mfma_preshuffle_pipeline.py,
mfma_epilogues.py). Porting that is a follow-up commit track;
the MVP here unblocks MX-FP8 callers end-to-end at bf16 throughput.

Layouts (match FlyDSL reference):
  - ``a``:        (M, K) fp8_e4m3fn   — row-major
  - ``b``:        (N, K) fp8_e4m3fn   — NT (c = a @ b.T)
  - ``scale_a``:  (K//128, M) f32     — indexed [block_k, m]
  - ``scale_b``:  (N//128, K//128) f32 — indexed [block_n, block_k]
  - ``out``:      (M, N) bf16
"""

from typing import Optional

import torch
from torch import Tensor

from quack.amd.gemm_gfx950_splitk import gemm_splitk


SCALE_BLOCK_K = 128
SCALE_BLOCK_N = 128


def _dequantize_fp8_to_bf16(
    x_fp8: Tensor, scales: Tensor, scale_transposed: bool, block_size_n: int = 1,
) -> Tensor:
    """Dequantize an fp8 tensor using per-K-block scales.

    ``x_fp8``:  (R, K) fp8_e4m3fn (R is M for A, N for B).
    ``scales``: f32. If ``scale_transposed`` (A path), shape (K//128, R)
                indexed [block_k, r]. Otherwise (B path, block_size_n=128),
                shape (R//128, K//128) indexed [block_n, block_k].
    """
    R, K = x_fp8.shape
    num_k_blocks = K // SCALE_BLOCK_K
    x_f32 = x_fp8.float()
    # Reshape K dim into (num_k_blocks, 128) so we can broadcast scales per block.
    x_blocked = x_f32.view(R, num_k_blocks, SCALE_BLOCK_K)
    if scale_transposed:
        # scales (num_k_blocks, R) → broadcast over last K-block dim.
        sc = scales.transpose(0, 1).unsqueeze(-1)            # (R, num_k_blocks, 1)
    else:
        # scales (R//block_size_n, num_k_blocks) → expand to (R, num_k_blocks, 1).
        sc = (
            scales
            .repeat_interleave(block_size_n, dim=0)
            .unsqueeze(-1)
        )  # (R, num_k_blocks, 1)
    out = (x_blocked * sc).view(R, K).to(torch.bfloat16)
    return out


@torch.library.custom_op(
    "quack_amd::_mxfp8_gemm_out",
    mutates_args=("out",),
    schema="(Tensor a, Tensor b, Tensor scale_a, Tensor scale_b, Tensor(a0!) out) -> ()",
)
def _mxfp8_gemm_out(
    a: Tensor, b: Tensor, scale_a: Tensor, scale_b: Tensor, out: Tensor,
) -> None:
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    assert scale_a.dtype == torch.float32 and scale_b.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2, f"A K={K} != B K={K2}"
    assert out.shape == (M, N)
    assert K % SCALE_BLOCK_K == 0, f"K={K} must be divisible by {SCALE_BLOCK_K}"
    assert N % SCALE_BLOCK_N == 0, f"N={N} must be divisible by {SCALE_BLOCK_N}"
    assert scale_a.shape == (K // SCALE_BLOCK_K, M), \
        f"scale_a shape {scale_a.shape} != ({K // SCALE_BLOCK_K}, {M})"
    assert scale_b.shape == (N // SCALE_BLOCK_N, K // SCALE_BLOCK_K), \
        f"scale_b shape {scale_b.shape} != ({N // SCALE_BLOCK_N}, {K // SCALE_BLOCK_K})"

    # Dequantize to bf16 in two ops.
    a_bf16 = _dequantize_fp8_to_bf16(a, scale_a, scale_transposed=True)
    b_bf16 = _dequantize_fp8_to_bf16(
        b, scale_b, scale_transposed=False, block_size_n=SCALE_BLOCK_N,
    )

    # Route through the fastest NT path we have when shape allows.
    # splitk requires N % 256 == 0 (tile_n=256 default).
    if (
        M % 128 == 0 and N % 256 == 0 and K % 64 == 0
        and M >= 128
    ):
        gemm_splitk(a_bf16, b_bf16, out)
    else:
        out.copy_(torch.nn.functional.linear(a_bf16, b_bf16).to(torch.bfloat16))


@_mxfp8_gemm_out.register_fake
def _mxfp8_gemm_out_fake(a, b, scale_a, scale_b, out):
    return None


def mxfp8_gemm(
    a: Tensor, b: Tensor,
    scale_a: Tensor, scale_b: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
    """MX-FP8 blockscaled GEMM: ``out = a @ b.T`` with per-block scales.

    Args:
      a:        (M, K) fp8_e4m3fn, row-major.
      b:        (N, K) fp8_e4m3fn, row-major (NT layout).
      scale_a:  (K//128, M) f32. ``scale_a[block_k, m]`` scales the
                128-element k-block of row m.
      scale_b:  (N//128, K//128) f32. ``scale_b[block_n, block_k]``
                scales a 128 × 128 tile of b[block_n * 128 : (block_n+1) * 128,
                block_k * 128 : (block_k+1) * 128].
      out:      optional preallocated (M, N) bf16 output.

    Constraints: K multiple of 128; N multiple of 128.

    Note: current implementation dequantizes to bf16 and runs the bf16
    GEMM path. Kernel-native ``mfma_scale`` path is a follow-up; see
    module docstring.
    """
    assert a.is_cuda and a.dtype == torch.float8_e4m3fn
    assert b.is_cuda and b.dtype == torch.float8_e4m3fn
    M, K = a.shape
    N, _ = b.shape
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
    _mxfp8_gemm_out(a, b, scale_a, scale_b, out)
    return out


__all__ = ["mxfp8_gemm", "SCALE_BLOCK_K", "SCALE_BLOCK_N"]
