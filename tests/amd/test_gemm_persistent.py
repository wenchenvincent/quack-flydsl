# Copyright (c) 2026, AMD.

"""Persistent-kernel MFMA GEMM — wires tile_scheduler to a real FlyDSL kernel."""

import pytest
import torch

from quack.amd.gemm_persistent import gemm_f16_persistent


@pytest.mark.parametrize("M", [256, 128, 64, 16])  # largest first — see conftest.py
@pytest.mark.parametrize("N", [16, 64, 128, 256])
@pytest.mark.parametrize("K", [16, 64, 128])
def test_gemm_f16_persistent(M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = gemm_f16_persistent(A, B)
    ref = A.float() @ B.float()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)


def test_gemm_f16_persistent_matches_non_persistent():
    """Both kernels should produce bit-equivalent (or ULP-equivalent) outputs."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950 import gemm_mfma
    torch.manual_seed(0)
    M, N, K = 128, 128, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C_pers = gemm_f16_persistent(A, B)
    C_std = gemm_mfma(A, B, out_dtype=torch.float32)
    torch.testing.assert_close(C_pers, C_std, atol=1e-5, rtol=1e-5)
