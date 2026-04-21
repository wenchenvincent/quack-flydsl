# Copyright (c) 2026, AMD.

"""Tests for the 32×32-tile MFMA GEMM path."""

import pytest
import torch

from quack.amd.gemm_gfx950_tiled import gemm_f16_32x32


@pytest.mark.parametrize("M", [256, 128, 64, 32])  # largest M first — grid-bake note.
@pytest.mark.parametrize("N", [32, 64, 128, 256])
@pytest.mark.parametrize("K", [32, 64, 128])
def test_gemm_f16_32x32(M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = gemm_f16_32x32(A, B)
    ref = A.float() @ B.float()
    # f16 inputs, f32 accumulator → roughly 3 ULP per K dot in f32.
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)


def test_gemm_f16_32x32_matches_16x16():
    """Numerical equivalence vs the 16×16 path (within fp32 ULP noise)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950 import gemm_mfma
    torch.manual_seed(0)
    A = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    B = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    C_32 = gemm_f16_32x32(A, B)
    C_16 = gemm_mfma(A, B, out_dtype=torch.float32)
    torch.testing.assert_close(C_32, C_16, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("M,N,K", [(64, 64, 64), (128, 128, 128)])
def test_gemm_32x32_bf16(M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_tiled import gemm_32x32
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    C = gemm_32x32(A, B)
    ref = A.float() @ B.float()
    torch.testing.assert_close(C, ref, atol=max(5e-3, K * 2e-5), rtol=5e-3)


@pytest.mark.parametrize(
    "variant",
    [
        "quack.amd.gemm_gfx950_tiled:gemm_32x32",
        "quack.amd.gemm_gfx950_lds:gemm_32x32_lds",
        "quack.amd.gemm_gfx950_lds_swizzle:gemm_32x32_lds_swz",
        "quack.amd.gemm_gfx950_4wave:gemm_64x64_4wave",
    ],
)
def test_tiled_variants_bf16(variant):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    import importlib
    mod, fn_name = variant.split(":")
    fn = getattr(importlib.import_module(mod), fn_name)
    torch.manual_seed(0)
    M, N, K = 128, 128, 128   # aligned for all variants
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    C = fn(A, B)
    ref = A.float() @ B.float()
    torch.testing.assert_close(C, ref, atol=max(5e-3, K * 2e-5), rtol=5e-3)
