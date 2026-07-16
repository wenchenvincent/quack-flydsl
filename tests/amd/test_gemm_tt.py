# Copyright (c) 2026, AMD.

"""TT-layout GEMM: C = A.T @ B.T (torch/hipBLASLt dispatch on transposed views)."""

import pytest
import torch

from quack.amd.gemm_gfx950_tt import gemm_tt


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N", [(64, 64, 64), (128, 256, 96), (256, 128, 512), (16, 512, 16)]
)
def test_gemm_tt_matches_torch(dtype, M, K, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(M + K + N)
    A = torch.randn(K, M, device="cuda", dtype=dtype)  # (K, M)
    B = torch.randn(N, K, device="cuda", dtype=dtype)  # (N, K)
    out = gemm_tt(A, B)
    ref = A.float().t() @ B.float().t()  # (M, N)
    assert out.shape == (M, N)
    atol = max(1e-2, K * 3e-5)
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=1e-2)


def test_gemm_tt_epilogue():
    """bias + alpha + beta*C epilogue flows through the gemm() dispatch."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, K, N = 128, 128, 64
    A = torch.randn(K, M, device="cuda", dtype=torch.float16)
    B = torch.randn(N, K, device="cuda", dtype=torch.float16)
    bias = torch.randn(N, device="cuda", dtype=torch.float32)
    Cres = torch.randn(M, N, device="cuda", dtype=torch.float32)
    alpha, beta = 0.5, 1.5
    out = gemm_tt(A, B, bias=bias, alpha=alpha, beta=beta, C=Cres, activation="relu")
    ref = alpha * (A.float().t() @ B.float().t()) + beta * Cres + bias
    ref = torch.relu(ref)
    atol = max(1e-2, K * 3e-5)
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=1e-2)


def test_gemm_tt_rejects_bad_shapes():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    A = torch.randn(32, 16, device="cuda", dtype=torch.float16)  # (K=32, M=16)
    B = torch.randn(16, 64, device="cuda", dtype=torch.float16)  # (N=16, K=64) -> mismatch
    with pytest.raises(AssertionError):
        gemm_tt(A, B)
