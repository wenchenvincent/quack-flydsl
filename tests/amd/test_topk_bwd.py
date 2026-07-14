import pytest
import torch

from quack.amd.topk import topk


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M,N,k", [(64, 256, 8), (32, 1024, 16), (16, 128, 4)])
def test_topk_backward_matches_torch(dtype, M, N, k):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)

    vals, idx = topk(x, k)
    rvals, ridx = torch.topk(xr, k, dim=-1)

    tol = 2e-2 if dtype is torch.bfloat16 else 1e-5
    assert torch.allclose(vals.float(), rvals.float(), atol=tol, rtol=tol)

    g = torch.randn_like(vals)
    vals.backward(g)
    rvals.backward(g)
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=tol, rtol=tol)


def test_topk_indices_non_differentiable():
    x = torch.randn(8, 64, device="cuda", requires_grad=True)
    vals, idx = topk(x, 4)
    assert not idx.requires_grad
    vals.sum().backward()
    assert x.grad is not None


def _ref_topk_softmax(x, k):
    rvals, ridx = torch.topk(x.float(), k, dim=-1)
    return torch.softmax(rvals, dim=-1), ridx


@pytest.mark.parametrize("M,N,k", [(64, 256, 8), (32, 512, 16)])
def test_topk_softmax_backward(M, N, k):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)

    sm, idx = topk(x, k, softmax=True)
    rsm, ridx = _ref_topk_softmax(xr, k)

    assert torch.allclose(sm.float(), rsm, atol=1e-5, rtol=1e-5)
    g = torch.randn_like(sm)
    sm.backward(g)
    rsm.backward(g)
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=1e-4, rtol=1e-4)


def test_topk_backward_single_wave_kernel():
    # N in _SINGLE_WAVE_N exercises the topk_mfma single-wave forward kernel
    torch.manual_seed(0)
    x = torch.randn(16, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    vals, idx = topk(x, 4)
    rvals, ridx = torch.topk(xr, 4, dim=-1)
    assert torch.allclose(vals, rvals, atol=1e-5, rtol=1e-5)
    g = torch.randn_like(vals)
    vals.backward(g)
    rvals.backward(g)
    assert torch.allclose(x.grad, xr.grad, atol=1e-5, rtol=1e-5)
