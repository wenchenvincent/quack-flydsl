# Copyright (c) 2026, AMD.

"""Tests for the 128×128-tile MFMA GEMM (cooperative LDS + ping-pong)."""

import pytest
import torch

from quack.amd.gemm_gfx950_128x128 import gemm_128x128


@pytest.mark.parametrize("M", [512, 256, 128])
@pytest.mark.parametrize("N", [128, 256, 512])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_128x128(M, N, K, dtype):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    C = gemm_128x128(A, B)
    ref = A.float() @ B.float()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)


def test_gemm_128x128_matches_64x64():
    """Numerical equivalence with the 64×64 cooperative-LDS kernel."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_4wave_lds_pp import gemm_64x64_4wave_lds_pp
    torch.manual_seed(0)
    A = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    B = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    C_128 = gemm_128x128(A, B)
    C_64 = gemm_64x64_4wave_lds_pp(A, B)
    torch.testing.assert_close(C_128, C_64, atol=1e-5, rtol=1e-5)


def test_autotune_picks_128x128_for_huge():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_autotune import select_best_kernel
    # ≥ 1024² with K÷32 → K=32 + LDMA combo (strictly dominates others at these sizes).
    assert select_best_kernel(4096, 4096, 4096, torch.float16, plain=True) == "128x128_k32_ldma"
    assert select_best_kernel(8192, 8192, 4096, torch.bfloat16, plain=True) == "128x128_k32_ldma"
    assert select_best_kernel(2048, 2048, 2048, torch.float16, plain=True) == "128x128_k32_ldma"
    assert select_best_kernel(1024, 1024, 1024, torch.bfloat16, plain=True) == "128x128_k32_ldma"
    # ≥ 4096² with K÷16 but not ÷32 → LDMA alone (K=32 unavailable).
    assert select_best_kernel(4096, 4096, 4080, torch.float16, plain=True) == "128x128_ldma"
    # K not divisible by 16 → fall back to 64×64 lds_pp.
    assert select_best_kernel(2048, 2048, 2064, torch.float16, plain=True) == "4wave_64x64_lds_pp"
