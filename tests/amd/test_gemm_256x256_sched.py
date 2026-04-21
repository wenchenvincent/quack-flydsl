# Copyright (c) 2026, AMD.

"""Correctness tests for the 256×256 GEMM with explicit MFMA scheduling
directives (experimental).

The kernel is documented as a slower-than-baseline variant (see the
module docstring). These tests just confirm it compiles and produces
correct results so the file remains a reference for future scheduling
experiments.
"""

import pytest
import torch

from quack.amd.gemm_gfx950_256x256_sched import gemm_256x256_sched


@pytest.mark.parametrize("M", [512, 256])
@pytest.mark.parametrize("N", [256, 512])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_256x256_sched(M, N, K, dtype):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    C = gemm_256x256_sched(A, B)
    ref = A.float() @ B.float()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)


def test_matches_256x256_k16():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_256x256 import gemm_256x256
    torch.manual_seed(0)
    A = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    B = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    C_sched = gemm_256x256_sched(A, B)
    C_k16 = gemm_256x256(A, B)
    torch.testing.assert_close(C_sched, C_k16, atol=1e-4, rtol=1e-4)
