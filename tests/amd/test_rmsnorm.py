# Copyright (c) 2026, AMD.

"""Numerical-correctness tests for quack.amd.rmsnorm against a PyTorch reference.

Every test compares kernel output against a float32 ground-truth reference
computed in PyTorch, per AGENTS.md.
"""

import pytest
import torch

from quack.amd.rmsnorm import rmsnorm_fwd, layernorm_fwd, rmsnorm_bwd


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
    out, _ = rmsnorm_fwd(x, w)
    ref_out, _ = _ref_rmsnorm(x, w)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)


def test_rmsnorm_fwd_matches_without_weight():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(8, 256, device="cuda", dtype=torch.float32)
    out, _ = rmsnorm_fwd(x)
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
    out, rstd = rmsnorm_fwd(x, w, store_rstd=True)
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

    _, rstd = rmsnorm_fwd(x, w, store_rstd=True)
    dx, dw = rmsnorm_bwd(x, w, dout, rstd)
    ref_dx, ref_dw = _ref_rmsnorm_bwd(x, w, dout, rstd)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(dx, ref_dx, atol=atol, rtol=rtol)
    torch.testing.assert_close(dw, ref_dw, atol=atol, rtol=rtol)


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
    out, _ = rmsnorm_fwd(x, w, bias=b)
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
    out, _, _ = layernorm_fwd(x, w)
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
    out, _, _ = layernorm_fwd(x, w, bias=b)
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
    out, rstd, mean = layernorm_fwd(x, w, store_stats=True)
    ref_out, ref_rstd, ref_mean = _ref_layernorm(x, w, None)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(rstd, ref_rstd, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(mean, ref_mean, atol=1e-4, rtol=1e-4)
