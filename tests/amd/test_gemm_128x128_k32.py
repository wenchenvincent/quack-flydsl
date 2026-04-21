# Copyright (c) 2026, AMD.

"""Correctness tests for the K=32 MFMA 128×128-tile GEMM.

Uses ``mfma_f32_16x16x32_f16`` / ``mfma_f32_16x16x32_bf16`` on gfx950
for 2× per-op throughput vs the K=16 baseline. Autotune dispatches
shapes ≥ 1024² here.
"""

import pytest
import torch

from quack.amd.gemm_gfx950_128x128_k32 import gemm_128x128_k32


@pytest.mark.parametrize("M", [512, 256])
@pytest.mark.parametrize("N", [256, 512])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_128x128_k32(M, N, K, dtype):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    C = gemm_128x128_k32(A, B)
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
    C_k32 = gemm_128x128_k32(A, B)
    C_k16 = gemm_128x128(A, B)
    torch.testing.assert_close(C_k32, C_k16, atol=1e-4, rtol=1e-4)
