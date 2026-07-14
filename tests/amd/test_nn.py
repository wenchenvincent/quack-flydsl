import pytest
import torch

from quack.amd.nn import RMSNorm, rmsnorm
from quack.amd.nn import LayerNorm, layernorm
from quack.amd.nn import softmax as amd_softmax
from quack.amd.nn import cross_entropy as amd_ce


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
    wr = m.weight.detach().clone().requires_grad_(True)
    y = m(x)
    ref = _ref_rmsnorm(xr, wr)
    assert y.shape == (64, 256)
    assert torch.allclose(y.float(), ref.float(), atol=1e-4, rtol=1e-4)
    g = torch.randn_like(y)
    y.backward(g)
    ref.backward(g)
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=5e-4, rtol=5e-4)
    assert torch.allclose(m.weight.grad.float(), wr.grad.float(), atol=5e-4, rtol=5e-4)


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


def _ref_layernorm(x, w, b, eps=1e-6):
    xf = x.float()
    mu = xf.mean(-1, keepdim=True)
    var = xf.var(-1, keepdim=True, unbiased=False)
    return ((xf - mu) * torch.rsqrt(var + eps)) * w.float() + b.float()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_layernorm_autograd(dtype):
    torch.manual_seed(0)
    M, N = 128, 256
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(N, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(N, device="cuda", dtype=dtype, requires_grad=True)
    xr, wr, br = (t.detach().clone().requires_grad_(True) for t in (x, w, b))

    out = layernorm(x, w, b)
    ref = _ref_layernorm(xr, wr, br).to(dtype)
    tol = 2e-2 if dtype is torch.bfloat16 else 1e-4
    assert torch.allclose(out.float(), ref.float(), atol=tol, rtol=tol)

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    for a, bb in ((x, xr), (w, wr), (b, br)):
        assert torch.allclose(a.grad.float(), bb.grad.float(), atol=tol * 5, rtol=tol * 5)


def test_layernorm_module():
    torch.manual_seed(0)
    m = LayerNorm(256, device="cuda", dtype=torch.float32)
    torch.nn.init.normal_(m.weight)
    torch.nn.init.normal_(m.bias)
    x = torch.randn(64, 256, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    wr = m.weight.detach().clone().requires_grad_(True)
    br = m.bias.detach().clone().requires_grad_(True)
    y = m(x)
    ref = _ref_layernorm(xr, wr, br)
    assert y.shape == (64, 256)
    assert torch.allclose(y.float(), ref.float(), atol=1e-4, rtol=1e-4)
    g = torch.randn_like(y)
    y.backward(g)
    ref.backward(g)
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=5e-4, rtol=5e-4)
    assert torch.allclose(m.weight.grad.float(), wr.grad.float(), atol=5e-4, rtol=5e-4)
    assert torch.allclose(m.bias.grad.float(), br.grad.float(), atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M,N", [(128, 256), (64, 1024)])
def test_softmax_autograd(dtype, M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    out = amd_softmax(x, dim=-1)
    ref = torch.softmax(xr.float(), dim=-1).to(dtype)
    tol = 2e-2 if dtype is torch.bfloat16 else 1e-4
    assert torch.allclose(out.float(), ref.float(), atol=tol, rtol=tol)
    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=tol * 5, rtol=tol * 5)


def test_softmax_non_last_dim():
    # exercise the dim-movement path: softmax over dim=0
    torch.manual_seed(0)
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    out = amd_softmax(x, dim=0)
    ref = torch.softmax(xr, dim=0)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    assert torch.allclose(x.grad, xr.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_cross_entropy_autograd(reduction):
    torch.manual_seed(0)
    M, V = 256, 512
    x = torch.randn(M, V, device="cuda", dtype=torch.float32, requires_grad=True)
    tgt = torch.randint(0, V, (M,), device="cuda")
    xr = x.detach().clone().requires_grad_(True)

    loss = amd_ce(x, tgt, reduction=reduction)
    ref = torch.nn.functional.cross_entropy(xr, tgt, reduction=reduction)

    assert torch.allclose(loss.float(), ref.float(), atol=1e-3, rtol=1e-3)
    (loss.sum() if reduction == "none" else loss).backward()
    (ref.sum() if reduction == "none" else ref).backward()
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=1e-3, rtol=1e-3)


def test_cross_entropy_ignore_index():
    torch.manual_seed(0)
    M, V = 128, 256
    x = torch.randn(M, V, device="cuda", dtype=torch.float32, requires_grad=True)
    tgt = torch.randint(0, V, (M,), device="cuda")
    tgt[::4] = -100  # ignore every 4th row
    xr = x.detach().clone().requires_grad_(True)
    loss = amd_ce(x, tgt, ignore_index=-100, reduction="mean")
    ref = torch.nn.functional.cross_entropy(xr, tgt, ignore_index=-100, reduction="mean")
    assert torch.allclose(loss.float(), ref.float(), atol=1e-3, rtol=1e-3)
    loss.backward()
    ref.backward()
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=1e-3, rtol=1e-3)


def test_public_exports():
    import quack.amd as qa

    for name in ("RMSNorm", "LayerNorm", "rmsnorm", "layernorm", "softmax", "cross_entropy"):
        assert hasattr(qa, name), f"quack.amd.{name} not exported"
