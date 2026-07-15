# Copyright (c) 2026, AMD.

"""Correctness tests for the MLP activation-recompute autograd path.

``mlp_recompute_train`` trades saved (M, hidden) activation memory for an
extra forward-shaped matmul in backward — verify it produces numerically
identical results to ``mlp_func_train`` (which saves both preact and
postact), and that it genuinely saves less.
"""

import pytest
import torch

from quack.amd.linear_training import mlp_func_train, mlp_recompute_train

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")


@pytest.mark.parametrize("bias", [False, True])
def test_mlp_recompute_matches_mlp_func_train(bias):
    torch.manual_seed(0)
    M, IN, HID, OUT = 128, 256, 256, 128
    x = torch.randn(M, IN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn(HID, IN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(OUT, HID, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b1 = torch.randn(HID, device="cuda", dtype=torch.bfloat16, requires_grad=True) if bias else None
    b2 = torch.randn(OUT, device="cuda", dtype=torch.bfloat16, requires_grad=True) if bias else None

    def run(fn):
        xs = x.detach().clone().requires_grad_(True)
        w1s = w1.detach().clone().requires_grad_(True)
        w2s = w2.detach().clone().requires_grad_(True)
        b1s = b1.detach().clone().requires_grad_(True) if bias else None
        b2s = b2.detach().clone().requires_grad_(True) if bias else None
        out = fn(xs, w1s, w2s, activation="silu", bias1=b1s, bias2=b2s)
        out.sum().backward()
        return out, xs.grad, w1s.grad, w2s.grad

    o_ref, gx_ref, gw1_ref, gw2_ref = run(mlp_func_train)
    o_rc, gx_rc, gw1_rc, gw2_rc = run(mlp_recompute_train)

    # recompute must be numerically identical math to the non-recompute path
    assert torch.allclose(o_rc.float(), o_ref.float(), atol=6e-2, rtol=6e-2)
    assert torch.allclose(gx_rc.float(), gx_ref.float(), atol=6e-2, rtol=6e-2)
    assert torch.allclose(gw1_rc.float(), gw1_ref.float(), atol=6e-2, rtol=6e-2)
    assert torch.allclose(gw2_rc.float(), gw2_ref.float(), atol=6e-2, rtol=6e-2)


def test_mlp_recompute_saves_less():
    """Recompute saves only (x, w1, w2[, biases]) — NOT preact/postact.

    IN and OUT are deliberately != HID (and != M) here: with IN == HID == 256
    (as in the numerics test above), the saved ``x`` tensor's shape (M, IN)
    is indistinguishable from a (M, HID) activation and would produce a
    false positive on the shape-based check below. Distinct dims make the
    check unambiguous.
    """
    torch.manual_seed(0)
    M, IN, HID, OUT = 128, 192, 256, 192
    x = torch.randn(M, IN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn(HID, IN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(OUT, HID, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = mlp_recompute_train(x, w1, w2, activation="silu")
    # the graph node's saved tensors should not include a (M, HID) preact or postact
    saved = out.grad_fn.saved_tensors
    assert not any(t.shape == (M, HID) for t in saved), (
        f"recompute unexpectedly saved a (M,HID) activation: {[t.shape for t in saved]}"
    )
