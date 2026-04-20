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
@pytest.mark.parametrize("M", [1, 4, 128])
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
