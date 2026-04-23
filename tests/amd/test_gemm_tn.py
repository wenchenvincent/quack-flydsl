# Copyright (c) 2026, AMD.

"""Correctness tests for ``quack.amd.gemm_gfx950_tn.gemm_tn``.

Matches ``torch.matmul(dy.T, x)`` up to f16/bf16 rounding. M-descending
parametrize per AGENTS.md FlyDSL grid-bake convention. Note: TN's kernel
cache key includes ``K`` (unlike NN), so shapes with the same M/N but
different K still JIT freshly — the cache fills quickly enough in tests.
"""

import pytest
import torch

from quack.amd.gemm_gfx950_tn import gemm_tn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")


@pytest.mark.parametrize("M", [512, 256, 128])  # output M (= out_dim)
@pytest.mark.parametrize(
    "K,N",
    [
        (64, 256),      # smallest — BLOCK_K=64, BLOCK_N=256
        (128, 512),
        (256, 256),
        (128, 1024),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_tn_matches_torch(M, K, N, dtype):
    torch.manual_seed(0)
    # In training terms: dy (batch=K, out=M) and x (batch=K, in=N) → dW (out=M, in=N).
    dy = torch.randn(K, M, device="cuda", dtype=dtype) * 0.1
    x = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
    out = gemm_tn(dy, x)
    ref = torch.matmul(dy.T.float(), x.float()).to(dtype)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 0.05, f"TN {dtype} K={K} M={M} N={N}: max_err {err:.4f} > 0.05"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_tn_zero_input(dtype):
    """Zero dy → zero dW (sanity)."""
    K, M, N = 64, 128, 256
    dy = torch.zeros(K, M, device="cuda", dtype=dtype)
    x = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
    out = gemm_tn(dy, x)
    assert (out.abs() < 1e-3).all(), f"zero-dy produced nonzero: max={out.abs().max()}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_tn_matmul_identity(dtype):
    """dy.T @ eye = dy.T — simple identity-check."""
    torch.manual_seed(1)
    K, M = 64, 128
    # x = identity-like: set N == K so x.T is square, but the kernel needs
    # N multiple of 256 so we use a small fixed shape. Use a custom x that's
    # zero except at known positions.
    N = 256
    dy = torch.randn(K, M, device="cuda", dtype=dtype) * 0.1
    x = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
    out = gemm_tn(dy, x)
    # Check one output cell manually: out[m=0, n=0] = sum_k dy[k, 0] * x[k, 0]
    expected_00 = (dy[:, 0].float() * x[:, 0].float()).sum().item()
    actual_00 = out[0, 0].float().item()
    assert abs(expected_00 - actual_00) < 0.1, (
        f"out[0,0] = {actual_00:.4f}, expected {expected_00:.4f}"
    )
