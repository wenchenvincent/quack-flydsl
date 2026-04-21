# Copyright (c) 2026, AMD.

"""Correctness tests for the stream-K NT-layout GEMM port.

The kernel is a port of FlyDSL's ``kernels/hgemm_splitk.py`` reference
implementation — 128×256 output tile, K=32 MFMA, B pre-shuffle, direct
HBM→LDS async DMA, and OnlineScheduler-driven vmem/mfma interleave.

Layout: ``c = a @ b.T`` where ``a`` is (M, K) and ``b`` is (N, K).
"""

import pytest
import torch

from quack.amd.gemm_gfx950_splitk import gemm_splitk, shuffle_b


@pytest.mark.parametrize("M", [512, 256])
@pytest.mark.parametrize("N", [256, 512])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_splitk(M, N, K, dtype):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(N, K, device="cuda", dtype=dtype)
    C = gemm_splitk(A, B)
    # Compare against hipBLASLt's F.linear (same NT semantics, same dtype
    # precision) — expected to match bit-exactly on matching MFMA variants.
    ref = torch.nn.functional.linear(A, B)
    torch.testing.assert_close(C, ref, atol=1e-3, rtol=1e-3)


def test_matches_hipblaslt_exactly():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    B = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    C = gemm_splitk(A, B)
    ref = torch.nn.functional.linear(A, B)
    # Our port + hipBLASLt both use mfma_f32_16x16x32_f16 on gfx950 —
    # the per-lane accumulation order matches, so results are bit-exact.
    err = (C.float() - ref.float()).abs().max().item()
    assert err == 0.0, f"expected bit-exact match, got max err {err}"


def test_shuffled_b_path():
    """Pre-shuffling B outside the kernel should give the same result as
    doing it inline."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    B = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    C_inline = gemm_splitk(A, B)                        # shuffle inside
    C_pre = gemm_splitk(A, shuffle_b(B), shuffled=True)  # shuffle outside
    torch.testing.assert_close(C_inline, C_pre, atol=0, rtol=0)
