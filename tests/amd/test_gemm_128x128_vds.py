# Copyright (c) 2026, AMD.

"""Correctness tests for 128×128 GEMM with vectorised LDS ops (experimental).

The kernel is neutral-to-slightly-worse than the scalar-LDS baseline
at 2048²+ — see module docstring. These tests keep it compiling + correct
so future LDS-layout experiments can branch from it.
"""

import pytest
import torch

from quack.amd.gemm_gfx950_128x128_vds import gemm_128x128_vds


@pytest.mark.parametrize("M", [512, 256])
@pytest.mark.parametrize("N", [256, 512])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_128x128_vds(M, N, K, dtype):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    C = gemm_128x128_vds(A, B)
    ref = A.float() @ B.float()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)


def test_matches_128x128_scalar():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_128x128 import gemm_128x128
    torch.manual_seed(0)
    A = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    B = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    C_vds = gemm_128x128_vds(A, B)
    C_sca = gemm_128x128(A, B)
    torch.testing.assert_close(C_vds, C_sca, atol=1e-4, rtol=1e-4)
