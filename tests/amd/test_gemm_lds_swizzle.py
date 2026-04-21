# Copyright (c) 2026, AMD.

"""Tests for the LDS ping-pong + B-preshuffle XOR-swizzle MFMA kernel."""

import pytest
import torch

from quack.amd.gemm_gfx950_lds_swizzle import gemm_f16_32x32_lds_swz


@pytest.mark.parametrize("M", [256, 128, 64, 32])
@pytest.mark.parametrize("N", [32, 64, 128, 256])
@pytest.mark.parametrize("K", [32, 64, 128])
def test_gemm_f16_32x32_lds_swz(M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = gemm_f16_32x32_lds_swz(A, B)
    ref = A.float() @ B.float()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)


def test_gemm_f16_32x32_lds_swz_matches_lds():
    """Numerical equivalence vs the non-swizzle LDS path."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_lds import gemm_f16_32x32_lds
    torch.manual_seed(0)
    A = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    B = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    C_swz = gemm_f16_32x32_lds_swz(A, B)
    C_lds = gemm_f16_32x32_lds(A, B)
    torch.testing.assert_close(C_swz, C_lds, atol=1e-5, rtol=1e-5)
