# Copyright (c) 2026, AMD.

"""Training-path correctness tests for ``quack.amd.linear.linear_train``.

Verifies that forward, dx (``dy @ W``), and dW (``dy.T @ x``) all match
``torch.nn.functional.linear``'s autograd to within bf16/f16 rounding
tolerance. This is the end-to-end gate on the autograd wiring of our
three GEMM kernels (NT fwd, NN dx, TN dw).
"""

import pytest
import torch

from quack.amd.linear import linear_train

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA/ROCm"
)


def _leaf(shape, dtype, scale=0.1):
    t = (torch.randn(shape, device="cuda", dtype=dtype) * scale).detach()
    t.requires_grad_(True)
    return t


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    # Constraints from the three kernels combined:
    #   fwd (gemm_splitk NT): bs>=128 & %128, out_f%256, in_f%64
    #   dx  (gemm_nn):        bs%128, out_f%64, in_f%256 (BLOCK_N=256)
    #   dw  (gemm_tn):        bs%64, out_f%128, in_f%256
    # Smallest valid shape: bs=128, in_f=256, out_f=256.
    "bs, in_f, out_f",
    [
        (128, 256, 256),
        (256, 256, 512),
        (512, 256, 1024),
        (1024, 512, 1024),
    ],
)
def test_linear_train_matches_torch(bs, in_f, out_f, dtype):
    """Fwd + dx + dW all within f16/bf16 tolerance."""
    torch.manual_seed(0)
    x = _leaf((bs, in_f), dtype)
    W = _leaf((out_f, in_f), dtype)

    # Reference
    x_ref = x.detach().clone().requires_grad_(True)
    W_ref = W.detach().clone().requires_grad_(True)
    y_ref = torch.nn.functional.linear(x_ref, W_ref)
    y_ref.sum().backward()

    # Our autograd path
    y = linear_train(x, W)
    y.sum().backward()

    fwd_err = (y.float() - y_ref.float()).abs().max().item()
    dx_err = (x.grad.float() - x_ref.grad.float()).abs().max().item()
    dw_err = (W.grad.float() - W_ref.grad.float()).abs().max().item()

    # Loose tolerances — bf16/f16 accumulation of large K grows error.
    tol = 0.5 if max(bs, in_f, out_f) > 512 else 0.05
    assert fwd_err < tol, f"fwd {dtype} {bs}x{in_f}x{out_f}: {fwd_err:.4f} > {tol}"
    assert dx_err < tol, f"dx {dtype} {bs}x{in_f}x{out_f}: {dx_err:.4f} > {tol}"
    assert dw_err < tol, f"dw {dtype} {bs}x{in_f}x{out_f}: {dw_err:.4f} > {tol}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_linear_train_no_dx_grad(dtype):
    """When x doesn't require grad, dx should be skipped (no kernel call)."""
    torch.manual_seed(1)
    x = (torch.randn(256, 256, device="cuda", dtype=dtype) * 0.1).detach()
    W = _leaf((512, 256), dtype)
    y = linear_train(x, W)
    y.sum().backward()
    assert W.grad is not None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_linear_train_mlp_composition(dtype):
    """Two-layer MLP: fwd = W2 @ relu(W1 @ x). Verify all four grads via torch."""
    torch.manual_seed(2)
    bs, hidden = 256, 512
    in_f, out_f = 256, 256
    x = _leaf((bs, in_f), dtype)
    W1 = _leaf((hidden, in_f), dtype)
    W2 = _leaf((out_f, hidden), dtype)

    # Reference
    x_ref = x.detach().clone().requires_grad_(True)
    W1_ref = W1.detach().clone().requires_grad_(True)
    W2_ref = W2.detach().clone().requires_grad_(True)
    h_ref = torch.relu(torch.nn.functional.linear(x_ref, W1_ref))
    y_ref = torch.nn.functional.linear(h_ref, W2_ref)
    y_ref.sum().backward()

    # Our — torch relu between the two linear_train calls
    h = torch.relu(linear_train(x, W1))
    y = linear_train(h, W2)
    y.sum().backward()

    dx_err = (x.grad.float() - x_ref.grad.float()).abs().max().item()
    dW1_err = (W1.grad.float() - W1_ref.grad.float()).abs().max().item()
    dW2_err = (W2.grad.float() - W2_ref.grad.float()).abs().max().item()

    tol = 0.5
    assert dx_err < tol, f"MLP dx {dtype}: {dx_err:.4f}"
    assert dW1_err < tol, f"MLP dW1 {dtype}: {dW1_err:.4f}"
    assert dW2_err < tol, f"MLP dW2 {dtype}: {dW2_err:.4f}"
