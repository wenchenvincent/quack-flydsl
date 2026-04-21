# Copyright (c) 2026, AMD.

"""Tests for gemm_dact / gemm_dgated (activation-backward fused GEMM)."""

import pytest
import torch

from quack.amd.gemm import gemm_dact, gemm_dgated


@pytest.mark.parametrize("activation", [None, "relu", "relu_sq", "silu", "gelu_tanh_approx"])
@pytest.mark.parametrize("M, K, N", [(64, 64, 128), (128, 64, 64)])
def test_gemm_dact(activation, M, K, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    preact = torch.randn(M, N, device="cuda", dtype=torch.float16)

    dx, postact = gemm_dact(A, B, preact, activation=activation)

    dout = A.float() @ B.float()
    if activation is None:
        ref_dx = dout
        ref_post = preact
    else:
        pa = preact.float().detach().requires_grad_(True)
        fn = {
            "relu": lambda z: torch.relu(z),
            "relu_sq": lambda z: torch.relu(z) ** 2,
            "silu": torch.nn.functional.silu,
            "gelu_tanh_approx": lambda z: torch.nn.functional.gelu(z, approximate="tanh"),
        }[activation]
        y = fn(pa)
        (grad,) = torch.autograd.grad(y.sum(), pa)
        ref_dx = dout * grad
        ref_post = fn(preact.float()).to(preact.dtype)

    torch.testing.assert_close(dx.float(), ref_dx.float(), atol=0.05, rtol=0.02)
    torch.testing.assert_close(postact.float(), ref_post.float(), atol=1e-2, rtol=1e-2)


def test_gemm_dact_routing_eligibility():
    """Verify the eligibility predicate gates correctly on shape/dtype."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm import _gemm_dact_fused_eligible
    A = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    B = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    pre = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    # Eligible
    assert _gemm_dact_fused_eligible(A, B, pre, "silu")
    # Non-aligned M → not eligible
    A_bad = torch.randn(17, 64, device="cuda", dtype=torch.float16)
    pre_bad = torch.randn(17, 128, device="cuda", dtype=torch.float16)
    assert not _gemm_dact_fused_eligible(A_bad, B, pre_bad, "silu")
    # Unsupported activation
    assert not _gemm_dact_fused_eligible(A, B, pre, "tanh")
    # Wrong dtype
    A_f32 = A.float()
    B_f32 = B.float()
    assert not _gemm_dact_fused_eligible(A_f32, B_f32, pre, "silu")


@pytest.mark.parametrize("activation", [None, "relu", "silu"])
def test_gemm_dact_fused_matches_unfused(activation):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_dact_kernel import gemm_dact_fused
    torch.manual_seed(0)
    A = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    B = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    pre = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    dx_f, post_f = gemm_dact_fused(A, B, pre, activation=activation)

    # Unfused reference: MFMA GEMM then torch.
    dout = A.float() @ B.float()
    if activation is None:
        ref_dx = dout.to(torch.float16)
        ref_post = pre
    else:
        pa = pre.float().detach().requires_grad_(True)
        fn = {
            "relu": torch.relu,
            "silu": torch.nn.functional.silu,
        }[activation]
        y = fn(pa)
        (grad,) = torch.autograd.grad(y.sum(), pa)
        ref_dx = (dout * grad).to(torch.float16)
        ref_post = fn(pre.float()).to(pre.dtype)
    torch.testing.assert_close(dx_f.float(), ref_dx.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(post_f.float(), ref_post.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("gate_type", ["swiglu", "reglu", "geglu", "glu"])
@pytest.mark.parametrize("M, K, N", [(64, 64, 128)])
def test_gemm_dgated(gate_type, M, K, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    preact = torch.randn(M, 2 * N, device="cuda", dtype=torch.float16)

    dx, postact = gemm_dgated(A, B, preact, gate_type=gate_type)
    assert dx.shape == preact.shape
    assert postact.shape == (M, N)
    # Sanity: no NaN/Inf.
    assert torch.isfinite(dx).all()
    assert torch.isfinite(postact).all()
