# Copyright (c) 2026, AMD.

"""MXFP6 (e2m3 + e8m0) scaled MFMA GEMM on gfx950 — ``C = A @ B.T``.

The kernel quantizes A/B to MXFP6 internally, so the reference dequantizes the
*same* quantized tensors and matmuls — the hardware-scaled MFMA (cbsz=blgp=2)
must reproduce the dequant-then-matmul result exactly.
"""

import pytest
import torch

from quack.amd.gemm_gfx950_mxfp6 import gemm_mxfp6
from quack.amd.mxfp6_ops import quantize_mxfp6, dequantize_mxfp6


def _dequant_ref(a, b):
    ap, asc, ash = quantize_mxfp6(a, axis=-1)
    bp, bsc, bsh = quantize_mxfp6(b, axis=-1)
    return dequantize_mxfp6(ap, asc, ash) @ dequantize_mxfp6(bp, bsc, bsh).t()


@pytest.mark.parametrize(
    "M,K,N", [(16, 128, 16), (64, 256, 48), (128, 512, 96), (32, 128, 256), (16, 384, 16)]
)
def test_gemm_mxfp6_matches_dequant(M, K, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(M + K + N)
    a = torch.randn(M, K, device="cuda") * 1.5
    b = torch.randn(N, K, device="cuda") * 1.5
    c = gemm_mxfp6(a, b)
    ref = _dequant_ref(a, b)
    assert c.shape == (M, N) and c.dtype == torch.float32
    rel = (c - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    assert rel < 1e-4, f"MXFP6 GEMM {M}x{K}x{N} rel err {rel}"


def test_gemm_mxfp6_close_to_true_matmul():
    # e2m3 has 3 mantissa bits, so it tracks the true matmul far better than MXFP4.
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(7)
    M, K, N = 64, 256, 64
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(N, K, device="cuda")
    c = gemm_mxfp6(a, b)
    ref = a @ b.t()
    rel = (c - ref).norm() / ref.norm()
    assert rel < 0.1, f"MXFP6 vs true matmul rel err {rel.item()}"


def test_gemm_mxfp6_rejects_bad_shapes():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    a = torch.randn(15, 128, device="cuda")
    b = torch.randn(16, 128, device="cuda")
    with pytest.raises(AssertionError):
        gemm_mxfp6(a, b)
    a = torch.randn(16, 100, device="cuda")
    b = torch.randn(16, 100, device="cuda")
    with pytest.raises(AssertionError):
        gemm_mxfp6(a, b)
