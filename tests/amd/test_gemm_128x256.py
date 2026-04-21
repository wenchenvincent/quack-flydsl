# Copyright (c) 2026, AMD.

"""Smoke tests for the experimental 128×256 8-wave GEMM kernel.

Not wired into the autotune dispatcher — see the module docstring for
why. Tests here confirm the kernel compiles and produces correct
output so future arch evaluation has a known-working baseline.
"""

import pytest
import torch

from quack.amd.gemm_gfx950_128x256 import gemm_128x256


@pytest.mark.parametrize("M", [256, 128])
@pytest.mark.parametrize("N", [256, 512])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_128x256(M, N, K, dtype):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    C = gemm_128x256(A, B)
    ref = A.float() @ B.float()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)
