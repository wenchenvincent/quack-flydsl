# Copyright (c) 2026, AMD.

"""Symmetric MFMA GEMM: C = A @ A.T."""

import pytest
import torch

from quack.amd.gemm_symmetric import gemm_symmetric


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [128, 64, 16])
@pytest.mark.parametrize("K", [16, 64, 128])
def test_gemm_symmetric(dtype, M, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    C = gemm_symmetric(A)
    ref = A.float() @ A.float().t()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)


def test_gemm_symmetric_output_dtype():
    """out_dtype=bf16/f16 downcasts the accumulator."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    C = gemm_symmetric(A, out_dtype=torch.bfloat16)
    assert C.dtype == torch.bfloat16


def test_gemm_symmetric_is_symmetric():
    """Result should be symmetric: C[i,j] == C[j,i]."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    C = gemm_symmetric(A)
    torch.testing.assert_close(C, C.t(), atol=1e-4, rtol=1e-4)
