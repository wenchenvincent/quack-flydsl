# Copyright (c) 2026, AMD.

"""End-to-end integration tests across the AMD public API.

Compares ``quack.amd.{linear, mlp, linear_cross_entropy}`` against a
float32 torch reference, exercising the full dispatch chain including
auto-dispatch to the tiled / LDS / 4-wave kernels on aligned shapes.
"""

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M, N, K", [(64, 128, 64), (128, 128, 128)])
def test_linear_matches_torch(dtype, M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.linear import linear
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    w = torch.randn(N, K, device="cuda", dtype=dtype)
    b = torch.randn(N, device="cuda", dtype=dtype)
    out = linear(x, w, b)
    ref = (x.float() @ w.float().t() + b.float()).to(dtype)
    atol = 5e-3 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=atol)


def test_mlp_matches_torch():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.mlp import mlp
    torch.manual_seed(0)
    M, D, H = 64, 64, 128
    x = torch.randn(M, D, device="cuda", dtype=torch.float16)
    w1 = torch.randn(H, D, device="cuda", dtype=torch.float16)
    w2 = torch.randn(D, H, device="cuda", dtype=torch.float16)
    out = mlp(x, w1, w2, activation="relu")
    hidden = torch.relu((x.float() @ w1.float().t()))
    ref = (hidden @ w2.float().t()).to(torch.float16)
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "fn_path, M, N, K",
    [
        ("quack.amd.gemm:gemm", 64, 64, 64),
        ("quack.amd.gemm:gemm", 128, 128, 128),   # triggers 4wave_64x64
        ("quack.amd.gemm:gemm", 96, 96, 96),       # triggers lds_swz_32x32
        ("quack.amd.gemm:gemm", 48, 48, 48),       # triggers mfma_16x16
    ],
)
def test_gemm_dispatch_across_tiers(fn_path, M, N, K):
    """Exercise auto-dispatch across the tile-size tiers (all eligible
    plain-gemm calls) and confirm numerical correctness end-to-end."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    import importlib
    mod, fn = fn_path.split(":")
    gemm_fn = getattr(importlib.import_module(mod), fn)
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    out = gemm_fn(A, B, out_dtype=torch.float32)
    ref = A.float() @ B.float()
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


def test_fused_dact_matches_unfused():
    """Confirm fused dact routing produces the same result as torch composition."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm import gemm_dact
    torch.manual_seed(0)
    A = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    B = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    pre = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    dx, post = gemm_dact(A, B, pre, activation="silu")
    # Reference path
    dout = A.float() @ B.float()
    pa = pre.float().detach().requires_grad_(True)
    y = torch.nn.functional.silu(pa)
    (grad,) = torch.autograd.grad(y.sum(), pa)
    ref_dx = (dout * grad).to(torch.float16)
    ref_post = torch.nn.functional.silu(pre)
    torch.testing.assert_close(dx.float(), ref_dx.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(post.float(), ref_post.float(), atol=5e-2, rtol=5e-2)


def test_rmsnorm_plus_linear_end_to_end():
    """Compose RMSNorm + linear — catches cross-module integration bugs."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.rmsnorm import rmsnorm_fwd
    from quack.amd.linear import linear
    torch.manual_seed(0)
    M, D, H = 64, 64, 128
    x = torch.randn(M, D, device="cuda", dtype=torch.float16)
    norm_w = torch.randn(D, device="cuda", dtype=torch.float16)
    proj_w = torch.randn(H, D, device="cuda", dtype=torch.float16)
    y, _, _ = rmsnorm_fwd(x, norm_w)
    out = linear(y, proj_w)
    # Reference
    rstd = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    y_ref = (x.float() * rstd * norm_w.float()).to(torch.float16)
    out_ref = (y_ref.float() @ proj_w.float().t()).to(torch.float16)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)
