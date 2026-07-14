import pytest
import torch

from quack.amd.nn import RMSNorm, rmsnorm


def _ref_rmsnorm(x, w, eps=1e-6):
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * rms) * w.float()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M,N", [(128, 256), (32, 1024)])
def test_rmsnorm_autograd(dtype, M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(N, device="cuda", dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)

    out = rmsnorm(x, w)
    ref = _ref_rmsnorm(xr, wr).to(dtype)

    tol = 2e-2 if dtype is torch.bfloat16 else 1e-4
    assert torch.allclose(out.float(), ref.float(), atol=tol, rtol=tol)

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=tol * 5, rtol=tol * 5)
    assert torch.allclose(w.grad.float(), wr.grad.float(), atol=tol * 5, rtol=tol * 5)


def test_rmsnorm_module():
    torch.manual_seed(0)
    m = RMSNorm(256, device="cuda", dtype=torch.float32)
    x = torch.randn(64, 256, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    y = m(x)
    ref = _ref_rmsnorm(xr, m.weight)
    assert y.shape == (64, 256)
    assert torch.allclose(y.float(), ref.float(), atol=1e-4, rtol=1e-4)
    g = torch.randn_like(y)
    y.backward(g)
    ref.backward(g)
    assert m.weight.grad is not None
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=5e-4, rtol=5e-4)


def test_rmsnorm_3d_flatten():
    torch.manual_seed(0)
    B, S, N = 4, 16, 256
    x = torch.randn(B, S, N, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    out = rmsnorm(x, w)
    ref = _ref_rmsnorm(xr, wr)
    assert out.shape == (B, S, N)
    assert torch.allclose(out.float(), ref.float(), atol=1e-4, rtol=1e-4)
    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=5e-4, rtol=5e-4)
    assert torch.allclose(w.grad.float(), wr.grad.float(), atol=5e-4, rtol=5e-4)
