# Copyright (c) 2026, AMD.

"""Tests for 4-wave 64×64 MFMA GEMM with cooperative LDS + ping-pong prefetch."""

import pytest
import torch

from quack.amd.gemm_gfx950_4wave_lds_pp import gemm_64x64_4wave_lds_pp


@pytest.mark.parametrize("M", [256, 128, 64])
@pytest.mark.parametrize("N", [64, 128, 256])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_64x64_4wave_lds_pp(M, N, K, dtype):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    C = gemm_64x64_4wave_lds_pp(A, B)
    ref = A.float() @ B.float()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)


def test_pp_matches_lds():
    """Numerical equivalence with single-stage LDS."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_4wave_lds import gemm_64x64_4wave_lds
    torch.manual_seed(0)
    A = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    B = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    C_pp = gemm_64x64_4wave_lds_pp(A, B)
    C_lds = gemm_64x64_4wave_lds(A, B)
    torch.testing.assert_close(C_pp, C_lds, atol=1e-5, rtol=1e-5)


def test_autotune_picks_lds_pp_for_large():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_autotune import select_best_kernel
    assert select_best_kernel(2048, 2048, 1024, torch.float16, plain=True) == "4wave_64x64_lds_pp"
    assert select_best_kernel(4096, 4096, 4096, torch.bfloat16, plain=True) == "4wave_64x64_lds_pp"
    # 1024² still uses non-LDS 4wave.
    assert select_best_kernel(1024, 1024, 1024, torch.float16, plain=True) == "4wave_64x64"
