# Copyright (c) 2026, AMD.

"""Stream-K persistent GEMM tests."""

import pytest
import torch

from quack.amd.gemm_streamk import gemm_f16_streamk


@pytest.mark.parametrize("M", [256, 128, 64])
@pytest.mark.parametrize("N", [64, 128, 256])
@pytest.mark.parametrize("K", [64, 128])
def test_gemm_f16_streamk(M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = gemm_f16_streamk(A, B)
    ref = A.float() @ B.float()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)


def test_gemm_f16_streamk_matches_persistent():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_persistent import gemm_f16_persistent
    torch.manual_seed(0)
    A = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    B = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    C_streamk = gemm_f16_streamk(A, B)
    C_pers = gemm_f16_persistent(A, B)
    torch.testing.assert_close(C_streamk, C_pers, atol=1e-5, rtol=1e-5)
