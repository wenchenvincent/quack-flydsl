# Copyright (c) 2026, AMD.

import pytest
import torch

from quack.amd.softmax import softmax_fwd, softmax_bwd


def _tol(dtype):
    return {
        torch.float32: (5e-6, 5e-6),
        torch.float16: (1e-3, 1e-3),
        torch.bfloat16: (1e-2, 1e-2),
    }[dtype]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [128, 4, 1])  # largest M first — conftest grid-bake note.
@pytest.mark.parametrize("N", [256, 1024, 4096])
def test_softmax_fwd(dtype, M, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    y = softmax_fwd(x)
    y_ref = torch.softmax(x.float(), dim=-1).to(dtype)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("M", [1, 4])
@pytest.mark.parametrize("N", [256, 1024])
def test_softmax_bwd(dtype, M, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn(M, N, device="cuda", dtype=dtype)
    y_ref = torch.softmax(x, dim=-1)
    dx_ref = torch.autograd.grad(y_ref, x, dy)[0]
    y = softmax_fwd(x.detach())
    dx = softmax_bwd(dy, y)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(dx, dx_ref, atol=atol, rtol=rtol)


# Multi-wave path — N > block_threads so the per-row reduction needs
# cross-wave (LDS-backed) aggregation across the block.
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("N", [8192, 16384, 32768])
def test_softmax_fwd_multiwave(dtype, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(4, N, device="cuda", dtype=dtype)
    y = softmax_fwd(x)
    y_ref = torch.softmax(x.float(), dim=-1).to(dtype)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("N", [131072])
def test_softmax_large_n(N):
    """Lock the existing large-N capability (fwd+bwd correct at 128K)."""
    torch.manual_seed(0)
    M = 4
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    y = softmax_fwd(x)
    ref = torch.softmax(x, dim=-1)
    assert torch.allclose(y, ref, atol=1e-6, rtol=1e-6)
    dy = torch.randn_like(y)
    dx = softmax_bwd(dy, y)
    xg = x.detach().clone().requires_grad_(True)
    dx_ref = torch.autograd.grad(torch.softmax(xg, dim=-1), xg, dy)[0]
    assert torch.allclose(dx, dx_ref, atol=1e-6, rtol=1e-6)
