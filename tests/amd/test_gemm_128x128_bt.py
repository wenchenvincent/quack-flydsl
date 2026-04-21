# Copyright (c) 2026, AMD.

"""Correctness tests for the experimental B-transposed 128×128 kernel.

Not wired into the autotune dispatcher — see the module docstring.
"""

import pytest
import torch

from quack.amd.gemm_gfx950_128x128_bt import gemm_128x128_bt


@pytest.mark.parametrize("M", [256, 128])
@pytest.mark.parametrize("N", [128, 256])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_128x128_bt(M, N, K, dtype):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    C = gemm_128x128_bt(A, B)
    ref = A.float() @ B.float()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)
