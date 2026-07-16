# Copyright (c) 2026, AMD.

"""fuse_grad_accum: dW accumulated into weight.grad in-place (G2)."""

import pytest
import torch

from quack.amd.linear import linear_train, linear_act_train


def _shapes():
    # batch % 128, in % 256, out % 256 (satisfies gemm_tn K%64/M%128/N%256).
    return dict(batch=128, in_f=256, out_f=256)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fuse_grad_accum_two_passes(dtype):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    s = _shapes()
    W = torch.randn(s["out_f"], s["in_f"], device="cuda", dtype=dtype, requires_grad=True)

    def run(fuse):
        w = W.detach().clone().requires_grad_(True)
        g_after = []
        for step in range(2):
            x = torch.randn(s["batch"], s["in_f"], device="cuda", dtype=dtype, requires_grad=True)
            torch.manual_seed(100 + step)
            x = torch.randn(s["batch"], s["in_f"], device="cuda", dtype=dtype)
            y = linear_train(x, w, fuse_grad_accum=fuse)
            y.sum().backward()
        return w.grad.detach().float().clone()

    g_fused = run(True)
    g_plain = run(False)
    # Both accumulate over the two steps (no zero_grad between); must match.
    err = (g_fused - g_plain).abs().max().item()
    rel = err / (g_plain.abs().max().item() + 1e-6)
    print(f"\n{dtype} fuse_grad_accum 2-pass: max_err={err:.3f} rel={rel:.4f}")
    assert rel < 0.03, f"fused vs plain grad mismatch rel={rel}"


def test_fuse_grad_accum_first_pass_fallback():
    """First backward (weight.grad is None) must match the non-fused result
    (exercises the fallback branch, not the fused kernel)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(1)
    s = _shapes()
    x = torch.randn(s["batch"], s["in_f"], device="cuda", dtype=torch.bfloat16)
    W = torch.randn(s["out_f"], s["in_f"], device="cuda", dtype=torch.bfloat16)

    wf = W.clone().requires_grad_(True)
    linear_train(x, wf, fuse_grad_accum=True).sum().backward()

    wp = W.clone().requires_grad_(True)
    linear_train(x, wp, fuse_grad_accum=False).sum().backward()

    torch.testing.assert_close(wf.grad, wp.grad, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("activation", ["relu", "silu"])
def test_fuse_grad_accum_act(activation):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(2)
    s = _shapes()
    W = torch.randn(s["out_f"], s["in_f"], device="cuda", dtype=torch.bfloat16)

    def run(fuse):
        w = W.clone().requires_grad_(True)
        for step in range(2):
            torch.manual_seed(200 + step)
            x = torch.randn(s["batch"], s["in_f"], device="cuda", dtype=torch.bfloat16)
            linear_act_train(x, w, activation, fuse_grad_accum=fuse).sum().backward()
        return w.grad.detach().float()

    g_fused, g_plain = run(True), run(False)
    rel = (g_fused - g_plain).abs().max().item() / (g_plain.abs().max().item() + 1e-6)
    assert rel < 0.03, f"act fused vs plain mismatch rel={rel}"
