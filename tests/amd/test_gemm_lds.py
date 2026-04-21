# Copyright (c) 2026, AMD.

"""Tests for the 32×32-tile MFMA GEMM with LDS ping-pong prefetch."""

import pytest
import torch

from quack.amd.gemm_gfx950_lds import gemm_f16_32x32_lds


@pytest.mark.parametrize("M", [256, 128, 64, 32])
@pytest.mark.parametrize("N", [32, 64, 128, 256])
@pytest.mark.parametrize("K", [32, 64, 128])
def test_gemm_f16_32x32_lds(M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = gemm_f16_32x32_lds(A, B)
    ref = A.float() @ B.float()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)


def test_gemm_f16_32x32_lds_matches_tiled():
    """Numerical equivalence vs the non-LDS 32×32 path."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_tiled import gemm_f16_32x32
    torch.manual_seed(0)
    A = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    B = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    C_lds = gemm_f16_32x32_lds(A, B)
    C_tiled = gemm_f16_32x32(A, B)
    torch.testing.assert_close(C_lds, C_tiled, atol=1e-5, rtol=1e-5)
