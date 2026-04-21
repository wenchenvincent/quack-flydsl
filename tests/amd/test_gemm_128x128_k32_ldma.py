# Copyright (c) 2026, AMD.

"""Correctness tests for 128×128 K=32 MFMA GEMM with direct HBM→LDS DMA.

Combines the K=32 MFMA (``gemm_gfx950_128x128_k32``) and direct-DMA
(``gemm_gfx950_128x128_ldma``) wins. The DMA thread mapping is
rearranged so lane L within each wave writes LDS byte L*16 (matching
``buffer_load_lds`` auto-indexing), independent of the consume-path
MFMA fragment layout.
"""

import pytest
import torch

from quack.amd.gemm_gfx950_128x128_k32_ldma import gemm_128x128_k32_ldma


@pytest.mark.parametrize("M", [512, 256])
@pytest.mark.parametrize("N", [256, 512])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_128x128_k32_ldma(M, N, K, dtype):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    C = gemm_128x128_k32_ldma(A, B)
    ref = A.float() @ B.float()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)


def test_matches_k32_baseline():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_128x128_k32 import gemm_128x128_k32
    torch.manual_seed(0)
    A = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    B = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    C_combo = gemm_128x128_k32_ldma(A, B)
    C_k32 = gemm_128x128_k32(A, B)
    torch.testing.assert_close(C_combo, C_k32, atol=1e-4, rtol=1e-4)
