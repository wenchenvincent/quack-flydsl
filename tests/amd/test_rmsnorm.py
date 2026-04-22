# Copyright (c) 2026, AMD.

"""Numerical-correctness tests for quack.amd.rmsnorm against a PyTorch reference.

Every test compares kernel output against a float32 ground-truth reference
computed in PyTorch, per AGENTS.md.
"""

import pytest
import torch

from quack.amd.rmsnorm import rmsnorm_fwd, layernorm_fwd, rmsnorm_bwd, layernorm_bwd


def _ref_rmsnorm(x, weight, eps=1e-6):
    x_f32 = x.float()
    w_f32 = weight.float()
    rstd = torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
    y_f32 = x_f32 * rstd * w_f32
    return y_f32.to(x.dtype), rstd.squeeze(-1)


def _ref_rmsnorm_bwd(x, weight, dout, rstd, eps=1e-6):
    x_f32 = x.float()
    x_hat = x_f32 * rstd.unsqueeze(1)
    wdy = dout.float() * weight.float()
    c1 = (x_hat * wdy).mean(dim=-1, keepdim=True)
    dx = (wdy - x_hat * c1) * rstd.unsqueeze(1)
    dw = (dout.float() * x_hat).sum(dim=0)
    return dx.to(x.dtype), dw.to(weight.dtype)


def _tol(dtype):
    return {
        torch.float32: (5e-6, 5e-6),
        torch.float16: (1e-3, 1e-3),
        torch.bfloat16: (1e-2, 1e-2),
    }[dtype]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [1, 4, 128])
@pytest.mark.parametrize("N", [64, 256, 1024, 4096])
def test_rmsnorm_fwd(dtype, M, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    out, _, _ = rmsnorm_fwd(x, w)
    ref_out, _ = _ref_rmsnorm(x, w)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)


def test_rmsnorm_fwd_matches_without_weight():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(8, 256, device="cuda", dtype=torch.float32)
    out, _, _ = rmsnorm_fwd(x)
    w_ones = torch.ones(256, device="cuda", dtype=torch.float32)
    ref_out, _ = _ref_rmsnorm(x, w_ones)
    torch.testing.assert_close(out, ref_out, atol=5e-6, rtol=5e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("N", [64, 256, 1024])
def test_rmsnorm_fwd_store_rstd(dtype, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(16, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    out, rstd, _ = rmsnorm_fwd(x, w, store_rstd=True)
    ref_out, ref_rstd = _ref_rmsnorm(x, w)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(rstd, ref_rstd, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [128, 4, 1])  # largest M first — see conftest.py
@pytest.mark.parametrize("N", [64, 256, 1024, 4096])
def test_rmsnorm_bwd(dtype, M, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    dout = torch.randn(M, N, device="cuda", dtype=dtype)

    _, rstd, _ = rmsnorm_fwd(x, w, store_rstd=True)
    dx, dw = rmsnorm_bwd(x, w, dout, rstd)
    ref_dx, ref_dw = _ref_rmsnorm_bwd(x, w, dout, rstd)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(dx, ref_dx, atol=atol, rtol=rtol)
    # dw accumulates M terms in f32 along a single thread (kernelised
    # column-parallel reduce). Torch's reference uses a tree-reduced sum,
    # so the two diverge at the ULP level as M grows — match QuACK's
    # NVIDIA bands (atol=1e-4, rtol=1e-3) rather than the fwd-pointwise
    # band used for dx.
    dw_atol = {torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[dtype]
    torch.testing.assert_close(dw, ref_dw, atol=dw_atol, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M,N", [(8192, 4096), (4096, 8192), (2048, 4096)])
def test_rmsnorm_bwd_dw_2stage(dtype, M, N):
    """Explicit coverage of the M≥1024, M%128=0, N%128=0 path that now
    routes through the 2-stage partial+final kernels. Correctness check
    against torch reference — the 2-stage reduction order differs from
    the single-thread version but both should match torch within the
    usual reduction tolerance band."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(N, device="cuda", dtype=dtype)
    dout = torch.randn(M, N, device="cuda", dtype=dtype)
    _, rstd, _ = rmsnorm_fwd(x, w, store_rstd=True)
    _, dw = rmsnorm_bwd(x, w, dout, rstd)
    ref_dw = (dout.float() * (x.float() * rstd.float().unsqueeze(-1))).sum(dim=0).to(dtype)
    dw_atol = {torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-1}[dtype]
    torch.testing.assert_close(dw, ref_dw, atol=dw_atol, rtol=1e-3)


def _ref_layernorm_bwd(x, weight, dout, rstd, mean, bias=None, eps=1e-6):
    x_f = x.float()
    w_f = weight.float()
    x_hat = (x_f - mean.unsqueeze(1)) * rstd.unsqueeze(1)
    wdy = dout.float() * w_f
    c0 = wdy.mean(dim=-1, keepdim=True)
    c1 = (wdy * x_hat).mean(dim=-1, keepdim=True)
    dx = (wdy - c0 - x_hat * c1) * rstd.unsqueeze(1)
    dw = (dout.float() * x_hat).sum(dim=0)
    db = dout.float().sum(dim=0) if bias is not None else None
    return (
        dx.to(x.dtype),
        dw.to(weight.dtype),
        db.to(bias.dtype) if db is not None else None,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [128, 4, 1])
@pytest.mark.parametrize("N", [64, 256, 1024, 4096])
@pytest.mark.parametrize("with_bias", [False, True])
def test_layernorm_bwd(dtype, M, N, with_bias):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    b = torch.randn(N, device="cuda", dtype=dtype) if with_bias else None
    dout = torch.randn(M, N, device="cuda", dtype=dtype)
    _, rstd, mean, _ = layernorm_fwd(x, w, bias=b, store_stats=True)
    dx, dw, db = layernorm_bwd(x, w, dout, rstd, mean, bias=b)
    ref_dx, ref_dw, ref_db = _ref_layernorm_bwd(x, w, dout, rstd, mean, bias=b)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(dx, ref_dx, atol=atol, rtol=rtol)
    dw_atol = {torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[dtype]
    torch.testing.assert_close(dw, ref_dw, atol=dw_atol, rtol=1e-3)
    if with_bias:
        torch.testing.assert_close(db, ref_db, atol=dw_atol, rtol=1e-3)


# ---------------------------------------------------------------------------
# Per-head (3D) layouts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("B, H, N", [(4, 8, 128), (1, 16, 256), (2, 4, 1024)])
def test_rmsnorm_fwd_per_head(dtype, B, H, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(B, H, N, device="cuda", dtype=dtype)
    w = torch.randn(H, N, device="cuda", dtype=dtype)
    out, _, _ = rmsnorm_fwd(x, w)
    x_f = x.float()
    rstd = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = (x_f * rstd * w.float()).to(dtype)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("B, H, N", [(4, 8, 128), (2, 4, 1024)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_layernorm_fwd_per_head(dtype, B, H, N, with_bias):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(B, H, N, device="cuda", dtype=dtype)
    w = torch.randn(H, N, device="cuda", dtype=dtype)
    b = torch.randn(H, N, device="cuda", dtype=dtype) if with_bias else None
    out, _, _, _ = layernorm_fwd(x, w, bias=b)
    x_f = x.float()
    mean = x_f.mean(-1, keepdim=True)
    var = x_f.var(-1, keepdim=True, unbiased=False)
    rstd = torch.rsqrt(var + 1e-6)
    ref = (x_f - mean) * rstd * w.float()
    if with_bias:
        ref = ref + b.float()
    ref = ref.to(dtype)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Bias + LayerNorm
# ---------------------------------------------------------------------------


def _ref_rmsnorm_bias(x, weight, bias, eps=1e-6):
    x_f32 = x.float()
    rstd = torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
    y = x_f32 * rstd * weight.float() + bias.float()
    return y.to(x.dtype)


def _ref_layernorm(x, weight, bias, eps=1e-6):
    x_f32 = x.float()
    mean = x_f32.mean(dim=-1, keepdim=True)
    var = x_f32.var(dim=-1, keepdim=True, unbiased=False)
    rstd = torch.rsqrt(var + eps)
    y = (x_f32 - mean) * rstd * weight.float()
    if bias is not None:
        y = y + bias.float()
    return y.to(x.dtype), rstd.squeeze(-1), mean.squeeze(-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [128, 4, 1])
@pytest.mark.parametrize("N", [256, 1024])
def test_rmsnorm_fwd_with_bias(dtype, M, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    b = torch.randn(N, device="cuda", dtype=dtype)
    out, _, _ = rmsnorm_fwd(x, w, bias=b)
    ref = _ref_rmsnorm_bias(x, w, b)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [128, 4, 1])
@pytest.mark.parametrize("N", [256, 1024, 4096])
def test_layernorm_fwd(dtype, M, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    out, _, _, _ = layernorm_fwd(x, w)
    ref_out, _, _ = _ref_layernorm(x, w, None)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [128, 4, 1])
@pytest.mark.parametrize("N", [256, 1024])
def test_layernorm_fwd_with_bias(dtype, M, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    b = torch.randn(N, device="cuda", dtype=dtype)
    out, _, _, _ = layernorm_fwd(x, w, bias=b)
    ref_out, _, _ = _ref_layernorm(x, w, b)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("N", [256, 1024])
def test_layernorm_fwd_store_stats(dtype, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M = 16
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    out, rstd, mean, _ = layernorm_fwd(x, w, store_stats=True)
    ref_out, ref_rstd, ref_mean = _ref_layernorm(x, w, None)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(rstd, ref_rstd, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(mean, ref_mean, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Residual (x + residual) → norm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [128, 4, 1])
@pytest.mark.parametrize("N", [256, 1024])
def test_rmsnorm_fwd_with_residual(dtype, M, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    r = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    out, _, res_out = rmsnorm_fwd(x, w, residual=r, store_residual_out=True)
    ref_out, _ = _ref_rmsnorm(x + r, w)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(res_out, (x + r).to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [128, 4])
@pytest.mark.parametrize("N", [256, 1024])
def test_layernorm_fwd_with_residual_and_bias(dtype, M, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    r = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    b = torch.randn(N, device="cuda", dtype=dtype)
    out, _, _, _ = layernorm_fwd(x, w, bias=b, residual=r)
    ref_out, _, _ = _ref_layernorm(x + r, w, b)
    # Combined residual + bias accumulates more ULP noise than a single op —
    # bump tolerance slightly over the base _tol() band for f16/bf16.
    base_atol, base_rtol = _tol(dtype)
    atol = base_atol if dtype is torch.float32 else base_atol * 3
    rtol = base_rtol if dtype is torch.float32 else base_rtol * 3
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)
