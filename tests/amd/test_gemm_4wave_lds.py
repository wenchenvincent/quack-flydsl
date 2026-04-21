# Copyright (c) 2026, AMD.

"""Tests for the 4-wave 64×64 MFMA GEMM with cooperative cross-wave LDS sharing."""

import pytest
import torch

from quack.amd.gemm_gfx950_4wave_lds import gemm_64x64_4wave_lds


@pytest.mark.parametrize("M", [256, 128, 64])
@pytest.mark.parametrize("N", [64, 128, 256])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_64x64_4wave_lds(M, N, K, dtype):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    C = gemm_64x64_4wave_lds(A, B)
    ref = A.float() @ B.float()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)


def test_gemm_4wave_lds_matches_4wave_nonlds():
    """Numerical equivalence vs the non-LDS 4-wave path."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_4wave import gemm_64x64_4wave
    torch.manual_seed(0)
    A = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    B = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    C_lds = gemm_64x64_4wave_lds(A, B)
    C_ref = gemm_64x64_4wave(A, B)
    torch.testing.assert_close(C_lds, C_ref, atol=1e-5, rtol=1e-5)


def test_lds_variant_registered_in_autotune():
    """The single-stage LDS kernel is still registered in case callers
    want to target it directly; the dispatcher now routes large shapes
    to the ping-pong variant (``4wave_64x64_lds_pp``) — see
    ``test_gemm_4wave_lds_pp.test_autotune_picks_lds_pp_for_large``."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_autotune import KERNEL_REGISTRY, get_kernel
    assert "4wave_64x64_lds" in KERNEL_REGISTRY
    fn = get_kernel("4wave_64x64_lds")
    torch.manual_seed(0)
    A = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    B = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    out = fn(A, B)
    assert out.shape == (128, 128) and out.dtype == torch.float32
