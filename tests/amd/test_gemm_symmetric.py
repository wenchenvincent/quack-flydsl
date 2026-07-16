# Copyright (c) 2026, AMD.

"""Symmetric MFMA GEMM: C = A @ A.T."""

import pytest
import torch

from quack.amd.gemm_symmetric import gemm_symmetric


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [128, 64, 16])
@pytest.mark.parametrize("K", [16, 64, 128])
def test_gemm_symmetric(dtype, M, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    C = gemm_symmetric(A)
    ref = A.float() @ A.float().t()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)


def test_gemm_symmetric_output_dtype():
    """out_dtype=bf16/f16 downcasts the accumulator."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    C = gemm_symmetric(A, out_dtype=torch.bfloat16)
    assert C.dtype == torch.bfloat16


def test_gemm_symmetric_is_symmetric():
    """Result should be symmetric: C[i,j] == C[j,i]."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    C = gemm_symmetric(A)
    torch.testing.assert_close(C, C.t(), atol=1e-4, rtol=1e-4)


def _act_ref(x, activation):
    if activation is None:
        return x
    if activation == "relu":
        return torch.relu(x)
    if activation == "relu_sq":
        return torch.relu(x) * x
    if activation == "gelu_tanh_approx":
        return torch.nn.functional.gelu(x, approximate="tanh")
    if activation == "silu":
        return torch.nn.functional.silu(x)
    raise ValueError(activation)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", [None, "relu", "relu_sq", "gelu_tanh_approx", "silu"])
@pytest.mark.parametrize("has_bias", [False, True])
def test_gemm_symmetric_epilogue(dtype, activation, has_bias):
    """bias + activation epilogue vs act(A@A.T + bias) f32 reference."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, K = 64, 64
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    bias = torch.randn(M, device="cuda", dtype=torch.float32) if has_bias else None
    C = gemm_symmetric(A, bias=bias, activation=activation)
    ref = A.float() @ A.float().t()
    if has_bias:
        ref = ref + bias  # per-column (last-dim) broadcast
    ref = _act_ref(ref, activation)
    atol = max(1e-2, K * 3e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=1e-2)


@pytest.mark.parametrize("alpha,beta", [(2.0, 0.0), (0.5, 1.5), (1.0, 1.0)])
def test_gemm_symmetric_alpha_beta_c(alpha, beta):
    """alpha*(A@A.T) + beta*C residual."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, K = 64, 48
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    Cres = torch.randn(M, M, device="cuda", dtype=torch.float32)
    out = gemm_symmetric(A, alpha=alpha, beta=beta, C=Cres)
    ref = alpha * (A.float() @ A.float().t()) + beta * Cres
    atol = max(1e-2, K * 3e-5)
    torch.testing.assert_close(out, ref, atol=atol, rtol=1e-2)


def test_gemm_symmetric_dispatch_no_fallback():
    """gemm_symmetric(A, bias=...) routes to the dedicated kernel, not the
    A.T-transpose fallback (which would materialise a (K,M) copy)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm import gemm_symmetric as dispatch_symmetric
    from quack.amd.gemm_symmetric import _kernel_cache

    torch.manual_seed(0)
    A = torch.randn(32, 32, device="cuda", dtype=torch.float16)
    bias = torch.randn(32, device="cuda", dtype=torch.float32)
    before = len(_kernel_cache)
    out = dispatch_symmetric(A, bias=bias)
    # A dedicated-kernel cache entry must have been created (proof it did not
    # fall back to gemm(A, A.T)).
    assert len(_kernel_cache) > before
    ref = A.float() @ A.float().t() + bias
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
