# Copyright (c) 2026, AMD.

import pytest
import torch
import torch.nn.functional as F

from quack.amd.cross_entropy import (
    cross_entropy_fwd, cross_entropy_bwd, cross_entropy_fwd_bwd,
)


def _tol(dtype):
    return {
        torch.float32: (1e-4, 1e-4),
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (2e-2, 2e-2),
    }[dtype]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tgt_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("M", [128, 4, 1])  # largest M first — see conftest.py
@pytest.mark.parametrize("N", [256, 1024, 4096])
def test_cross_entropy_fwd(dtype, tgt_dtype, M, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    target = torch.randint(0, N, (M,), device="cuda", dtype=tgt_dtype)

    loss, lse = cross_entropy_fwd(x, target, return_lse=True)

    ref_loss = torch.nn.functional.cross_entropy(x.float(), target.long(), reduction="none")
    ref_lse = torch.logsumexp(x.float(), dim=-1)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(loss, ref_loss, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, ref_lse, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [128, 4, 1])  # largest M first — see conftest.py
@pytest.mark.parametrize("N", [256, 1024, 4096])
def test_cross_entropy_bwd(dtype, M, N):
    # Real FlyDSL dx kernel now (replaces the torch-host fallback). The fwd
    # kernel produces `lse`, and the bwd kernel derives
    # `dx[m,j] = (exp(x[m,j] - lse[m]) - (j==target[m])) * dloss[m]`.
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)
    loss_ref = torch.nn.functional.cross_entropy(x.float(), target, reduction="none")
    dloss = torch.randn(M, device="cuda", dtype=torch.float32)
    dx_ref = torch.autograd.grad(loss_ref, x, dloss)[0]

    _, lse = cross_entropy_fwd(x.detach(), target, return_lse=True)
    dx = cross_entropy_bwd(x.detach(), target, lse, dloss)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(dx, dx_ref.to(dtype), atol=atol, rtol=rtol)


# --- ignore_index / label_smoothing / loss_weight -------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("N", [256, 1024])
def test_cross_entropy_ignore_index(dtype, N):
    """Target == ignore_index rows should emit 0 loss and 0 dx."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M = 64
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)
    target[::8] = -100  # every 8th row ignored

    loss, _, dx = cross_entropy_fwd_bwd(x, target, ignore_index=-100)
    ref_loss = F.cross_entropy(x.float(), target, reduction="none", ignore_index=-100)
    # torch masks ignored rows to 0 in loss — matches our kernel.
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(loss, ref_loss, atol=atol, rtol=rtol)
    # Ignored rows' dx must be exactly 0.
    assert (dx[::8] == 0).all()

    # Non-ignored rows match plain CE dx (dloss=1 convention).
    ref_dx = (F.softmax(x.float(), dim=-1) - F.one_hot(target.clamp_min(0), N).float()).to(dtype)
    ref_dx[::8] = 0
    torch.testing.assert_close(dx.float(), ref_dx.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("smoothing", [0.05, 0.1, 0.2])
def test_cross_entropy_label_smoothing(dtype, smoothing):
    """Label smoothing loss: (1-α)*nll + α*(lse - mean(x));
    dx = softmax - smooth_target."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, N = 64, 512
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)

    loss, _, dx = cross_entropy_fwd_bwd(x, target, label_smoothing=smoothing)
    ref_loss = F.cross_entropy(x.float(), target, reduction="none", label_smoothing=smoothing)
    smooth_tgt = (1 - smoothing) * F.one_hot(target, N).float() + smoothing / N
    ref_dx = (F.softmax(x.float(), dim=-1) - smooth_tgt).to(dtype)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(loss, ref_loss, atol=atol, rtol=rtol)
    torch.testing.assert_close(dx.float(), ref_dx.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cross_entropy_loss_weight(dtype):
    """Per-row ``loss_weight`` multiplies loss and dx (== dloss folded in)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, N = 64, 512
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)
    w = torch.rand(M, device="cuda", dtype=torch.float32) + 0.5

    loss, _, dx = cross_entropy_fwd_bwd(x, target, loss_weight=w)
    ref_loss = F.cross_entropy(x.float(), target, reduction="none") * w
    ref_dx = ((F.softmax(x.float(), dim=-1) - F.one_hot(target, N).float()) * w.unsqueeze(-1)).to(dtype)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(loss, ref_loss, atol=atol, rtol=rtol)
    torch.testing.assert_close(dx.float(), ref_dx.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cross_entropy_all_features_combined(dtype):
    """ignore_index + label_smoothing + loss_weight together."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, N = 64, 512
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)
    target[::16] = -100
    w = torch.rand(M, device="cuda", dtype=torch.float32) + 0.5
    alpha = 0.15

    loss, _, dx = cross_entropy_fwd_bwd(
        x, target, ignore_index=-100, label_smoothing=alpha, loss_weight=w,
    )
    ref_loss = F.cross_entropy(
        x.float(), target, reduction="none",
        ignore_index=-100, label_smoothing=alpha,
    ) * w
    ref_loss[::16] = 0
    smooth_tgt = (1 - alpha) * F.one_hot(target.clamp_min(0), N).float() + alpha / N
    ref_dx = ((F.softmax(x.float(), dim=-1) - smooth_tgt) * w.unsqueeze(-1)).to(dtype)
    ref_dx[::16] = 0

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(loss, ref_loss, atol=atol, rtol=rtol)
    torch.testing.assert_close(dx.float(), ref_dx.float(), atol=atol, rtol=rtol)


def test_linear_cross_entropy_nonuniform_grad_loss():
    """Previously raised NotImplementedError. Now recomputes the
    backward with loss_weight=grad_loss, producing correct per-sample
    weighted gradients."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.linear_cross_entropy import linear_cross_entropy
    torch.manual_seed(0)
    B_L, V, d = 256, 1024, 128
    x = torch.randn(B_L, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(V, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(0, V, (B_L,), device="cuda", dtype=torch.int64)
    # Non-uniform per-row weight — simulates per-sample loss scaling.
    per_sample_w = (torch.rand(B_L, device="cuda") + 0.5).to(torch.float32)

    loss, _ = linear_cross_entropy(x, w, target)
    # (loss * per_sample_w).sum() gives non-uniform grad_loss = per_sample_w.
    (loss * per_sample_w).sum().backward()

    # Reference via torch.
    xr = x.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    logits = torch.nn.functional.linear(xr, wr)
    ref_loss = F.cross_entropy(logits.float(), target, reduction="none")
    (ref_loss * per_sample_w).sum().backward()

    torch.testing.assert_close(x.grad, xr.grad, atol=5e-1, rtol=5e-2)
    torch.testing.assert_close(w.grad.float(), wr.grad.float(), atol=5e-1, rtol=5e-2)


def test_linear_cross_entropy_ignore_index():
    """linear_cross_entropy with ignore_index — ignored rows drop out of
    the loss and contribute 0 to dx/dw."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.linear_cross_entropy import linear_cross_entropy
    torch.manual_seed(0)
    B_L, V, d = 256, 1024, 128
    x = torch.randn(B_L, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(V, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(0, V, (B_L,), device="cuda", dtype=torch.int64)
    target[::4] = -100

    loss, _ = linear_cross_entropy(x, w, target, ignore_index=-100)
    loss.sum().backward()

    xr = x.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    logits = torch.nn.functional.linear(xr, wr)
    ref_loss = F.cross_entropy(
        logits.float(), target, reduction="none", ignore_index=-100,
    )
    ref_loss.sum().backward()

    # Ignored rows contribute 0, so sums match.
    torch.testing.assert_close(loss.sum(), ref_loss.sum(), atol=5e-1, rtol=1e-2)
    torch.testing.assert_close(x.grad, xr.grad, atol=5e-1, rtol=5e-2)
    torch.testing.assert_close(w.grad.float(), wr.grad.float(), atol=5e-1, rtol=5e-2)


@pytest.mark.parametrize("N", [131072])
def test_cross_entropy_large_vocab(N):
    """Lock the existing huge-vocab capability (fwd+bwd correct at 128K)."""
    torch.manual_seed(0)
    M = 4
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    tgt = torch.randint(0, N, (M,), device="cuda")
    loss, lse = cross_entropy_fwd(x, tgt, return_lse=True)
    ref = torch.nn.functional.cross_entropy(x, tgt, reduction="none")
    assert torch.allclose(loss.float(), ref.float(), atol=1e-3, rtol=1e-3)
    dloss = torch.randn(M, device="cuda", dtype=torch.float32)
    dx = cross_entropy_bwd(x, tgt, lse, dloss)
    xg = x.detach().clone().requires_grad_(True)
    ref2 = torch.nn.functional.cross_entropy(xg, tgt, reduction="none")
    dx_ref = torch.autograd.grad(ref2, xg, dloss)[0]
    assert torch.allclose(dx, dx_ref, atol=1e-3, rtol=1e-3)
