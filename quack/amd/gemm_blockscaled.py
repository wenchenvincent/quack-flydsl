# Copyright (c) 2026, AMD.

"""Block-scaled fp8/fp4 GEMM — AMDGPU port of `quack/gemm_blockscaled_interface.py`.

**Current state:** API surface + torch reference. The FlyDSL MFMA kernel
body using ``rocdl.mfma_scale_f32_16x16x128_f8f6f4`` is the remaining
piece; scaffolded here so callers can use the public function today
while the real kernel is incrementally built out.

Plan for the kernel body (see `/root/.claude/plans/this-is-a-repo-functional-star.md`
§"Kernel 1: blockscaled fp8 GEMM"):

    - Per-lane fp8 operand packing: 32 fp8 values per lane (32 bytes) →
      4 × i64 → bitcast to i32x8 (see FlyDSL's
      ``blockscale_preshuffle_gemm.py:447 pack_i64x4_to_i32x8``).
    - Single ``mfma_scale_f32_16x16x128_f8f6f4`` call per scale block
      (K=128). Scales in the MFMA operand are set to neutral
      (``0x7F7F7F7F``); actual scaling is applied post-MFMA via
      ``math.fma`` of the scale-block product into the running f32
      accumulator (see reference lines 542–585).
    - Scale layout: ``A_scale`` is ``(K//128, M)`` f32 (transposed!);
      ``B_scale`` is ``(N//128, K//128)`` f32. Each workgroup loads
      its (M-row, scale-block) slice once per K tile (see reference
      lines 469–507 ``load_scales_for_tile``).
    - LDS: 2-stage ping-pong on A. B either preshuffled (reference
      default) or direct buffer_load on CDNA4 (simpler first pass).

References:
    - ``/workspace/FlyDSL/kernels/blockscale_preshuffle_gemm.py`` (881 LoC)
      — primary source, has explicit ``_is_gfx950`` branching.
    - ``/workspace/FlyDSL/kernels/mfma_preshuffle_pipeline.py`` — shared
      helpers (``pack_i64x4_to_i32x8``, ``lds_store_16b_xor16``).
    - CDNA4 scaled MFMA memory: `/root/.claude/projects/-workspace-quack/memory/reference_mfma_layouts.md`
      — per-lane fragment layout for the standard 16x16x16 variants,
      which extends to 16x16x128 by holding 32 fp8 values per lane
      instead of 4.

Supported today via the torch-reference dequantise→matmul path (exact
same contract the FlyDSL kernel will provide):
    - ``A`` fp8_e4m3fn, ``B`` fp8_e4m3fn — most common MX-style quant.
    - ``A_scale``, ``B_scale`` f32 per block of 128 K-elements.
    - ``out_dtype`` bf16 (default) or f32.
"""

from typing import Optional

import torch
from torch import Tensor


_SUPPORTED_IN_DTYPES = {torch.float8_e4m3fn}
_SUPPORTED_OUT_DTYPES = {torch.bfloat16, torch.float32}


def _dequantize_mxfp8(x_fp8: Tensor, scale: Tensor, block_k: int, axis: int) -> Tensor:
    """Dequantise an fp8 tensor scaled by block along ``axis``.

    ``scale`` shape depends on caller convention:
      - For A (shape (M, K)): scale is (K//block_k, M) — transposed.
      - For B (shape (K, N)): scale is (N//block_k, K//block_k).
    ``axis`` is the K-dim of ``x_fp8``: 1 for A (last dim), 0 for B.
    """
    x_f32 = x_fp8.to(torch.float32)
    if axis == 1:
        # A: (M, K). scale is (K//block_k, M). Rearrange to per-element.
        M, K = x_f32.shape
        assert K % block_k == 0
        # scale[kb, m] applies to x[m, kb*block_k:(kb+1)*block_k]
        # → broadcast to (M, K)
        scale_m_kb = scale.transpose(0, 1)  # (M, K//block_k)
        scale_expanded = scale_m_kb.repeat_interleave(block_k, dim=1)
        return x_f32 * scale_expanded
    else:  # axis == 0, B
        K, N = x_f32.shape
        assert K % block_k == 0
        # scale[n_block, kb] applies to x[kb*block_k:(kb+1)*block_k, n_block*128:(n_block+1)*128]
        # Torch: broadcast N per block, K per block.
        _num_n_blocks, num_k_blocks = scale.shape
        # For MVP we assume N % 128 == 0 matches scale's n block granularity.
        scale_kb_nb = scale.transpose(0, 1)  # (K//block_k, N//128)
        scale_k_expanded = scale_kb_nb.repeat_interleave(block_k, dim=0)  # (K, N//128)
        scale_full = scale_k_expanded.repeat_interleave(128, dim=1)  # (K, N)
        return x_f32 * scale_full


def mxfp8_gemm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Optional[Tensor] = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Block-scaled fp8 × fp8 → bf16/f32 GEMM.

    Shapes:
        A: ``(M, K)`` fp8_e4m3fn
        B: ``(K, N)`` fp8_e4m3fn
        A_scale: ``(K // 128, M)`` f32 (transposed)
        B_scale: ``(N // 128, K // 128)`` f32
        out: ``(M, N)`` of ``out_dtype`` (allocated if None)

    ``M, N, K`` must all be multiples of 128 (one scale block per K tile,
    one per 128 N-cols). Matches the layout of
    ``FlyDSL/kernels/blockscale_preshuffle_gemm.py``.

    Currently computes via a torch dequantise → matmul reference; will be
    replaced by a FlyDSL MFMA kernel using
    ``rocdl.mfma_scale_f32_16x16x128_f8f6f4`` in a follow-up commit.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype in _SUPPORTED_IN_DTYPES and B.dtype in _SUPPORTED_IN_DTYPES
    assert out_dtype in _SUPPORTED_OUT_DTYPES
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert K % 128 == 0 and N % 128 == 0, "M, N, K must be multiples of 128"
    assert A_scale.shape == (K // 128, M)
    assert B_scale.shape == (N // 128, K // 128)
    assert A_scale.dtype == torch.float32 and B_scale.dtype == torch.float32

    # Torch reference: dequantise inputs, matmul in f32, cast to out dtype.
    A_f32 = _dequantize_mxfp8(A, A_scale, block_k=128, axis=1)
    B_f32 = _dequantize_mxfp8(B, B_scale, block_k=128, axis=0)
    result = (A_f32 @ B_f32).to(out_dtype)

    if out is None:
        return result
    out.copy_(result)
    return out


__all__ = ["mxfp8_gemm"]
