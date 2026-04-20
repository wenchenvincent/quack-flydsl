# Copyright (c) 2026, AMD.

"""Numerical correctness of the proof-of-life FlyDSL MFMA GEMM for gfx950.

Covers `quack.amd.gemm_gfx950.gemm_f16_mfma` — a real FlyDSL kernel using
``rocdl.mfma_f32_16x16x16f16``. Constraints: M, N, K all multiples of 16
(single-tile-per-workgroup scope). Output dtype is always f32; callers
who want f16/bf16 output should downcast on the host or wait for the full
MFMA port.
"""

import pytest
import torch

from quack.amd.gemm_gfx950 import gemm_f16_mfma


@pytest.mark.parametrize("M", [128, 64, 16])  # largest first — see conftest.py
@pytest.mark.parametrize("N", [16, 64, 128])
@pytest.mark.parametrize("K", [16, 64, 128])
def test_gemm_f16_mfma(M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = gemm_f16_mfma(A, B)
    ref = A.float() @ B.float()
    # f16 → f32 MFMA accumulates in f32 so tolerance is tight, but K values
    # of random f16 operands accumulate ULP noise ~ sqrt(K) * eps(f16).
    atol = max(2e-3, K * 1e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=1e-3)


def test_gemm_f16_mfma_square():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    C = gemm_f16_mfma(x, x)
    ref = x.float() @ x.float()
    torch.testing.assert_close(C, ref, atol=1e-3, rtol=1e-3)


def test_gemm_f16_mfma_rejects_wrong_dtype():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    A = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    B = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    with pytest.raises(AssertionError):
        gemm_f16_mfma(A, B)
