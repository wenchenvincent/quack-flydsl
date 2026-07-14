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
