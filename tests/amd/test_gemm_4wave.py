# Copyright (c) 2026, AMD.

"""Tests for the 4-wave 64×64-tile MFMA GEMM."""

import pytest
import torch

from quack.amd.gemm_gfx950_4wave import gemm_f16_64x64_4wave


@pytest.mark.parametrize("M", [256, 128, 64])
@pytest.mark.parametrize("N", [64, 128, 256])
@pytest.mark.parametrize("K", [64, 128])
def test_gemm_f16_64x64_4wave(M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = gemm_f16_64x64_4wave(A, B)
    ref = A.float() @ B.float()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)


def test_gemm_f16_64x64_4wave_matches_tiled():
    """Numerical equivalence vs the single-wave 32×32 path."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_tiled import gemm_f16_32x32
    torch.manual_seed(0)
    A = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    B = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    C_4w = gemm_f16_64x64_4wave(A, B)
    C_1w = gemm_f16_32x32(A, B)
    torch.testing.assert_close(C_4w, C_1w, atol=1e-5, rtol=1e-5)
