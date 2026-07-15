# Copyright (c) 2026, AMD.

"""nn.Module wrapper tests for the AMD Linear / MLP / LinearCrossEntropy layers.

Verifies the ``torch.nn.Module`` wrappers in ``quack.amd.nn`` (Gap G1) against
plain torch references — fwd values and grads, bf16, loose tolerance (these
are the same training kernels exercised in ``test_linear_train.py`` /
``test_linear_cross_entropy.py``, just via the module surface).
"""

import pytest
import torch
import torch.nn.functional as F

from quack.amd.nn import Linear, MLP, LinearCrossEntropy


@pytest.mark.parametrize("bias", [True, False])
def test_linear_module_matches_torch(bias):
    torch.manual_seed(0)
    # linear_train's no-activation fwd path (gemm_splitk, NT) hard-asserts
    # out_features % 256 == 0 (BLOCK_N=256) — see linear_train's docstring
    # and test_linear_train.py's own constraint comment. 256 is the
    # smallest valid out_features for this path.
    M, IN, OUT = 128, 256, 256
    m = Linear(IN, OUT, bias=bias, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.Linear(IN, OUT, bias=bias, device="cuda", dtype=torch.bfloat16)
    # sync params so outputs are comparable
    with torch.no_grad():
        m.weight.copy_(ref.weight)
        if bias:
            m.bias.copy_(ref.bias)
    x = torch.randn(M, IN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    y = m(x)
    yr = ref(xr)
    assert torch.allclose(y.float(), yr.float(), atol=6e-2, rtol=6e-2)
    g = torch.randn_like(y)
    y.backward(g)
    yr.backward(g)
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=6e-2, rtol=6e-2)
    assert torch.allclose(m.weight.grad.float(), ref.weight.grad.float(), atol=6e-2, rtol=6e-2)
    if bias:
        assert torch.allclose(m.bias.grad.float(), ref.bias.grad.float(), atol=1.0, rtol=6e-2)


def test_mlp_module():
    torch.manual_seed(0)
    M, IN, HID, OUT = 128, 256, 256, 128
    m = MLP(IN, HID, OUT, activation="silu", device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, IN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = m(x)
    assert y.shape == (M, OUT)
    y.sum().backward()
    assert x.grad is not None and m.w1.grad is not None and m.w2.grad is not None


def test_linear_cross_entropy_module_rejects_bias():
    with pytest.raises(AssertionError):
        LinearCrossEntropy(128, 512, bias=True, device="cuda", dtype=torch.bfloat16)


def test_linear_cross_entropy_module():
    torch.manual_seed(0)
    B_L, IN, V = 256, 128, 512
    m = LinearCrossEntropy(IN, V, chunk_size=256, device="cuda", dtype=torch.bfloat16)
    ref_w = m.weight.detach().clone().float().requires_grad_(True)

    x = torch.randn(B_L, IN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(0, V, (B_L,), device="cuda", dtype=torch.int64)
    xr = x.detach().clone().float().requires_grad_(True)

    loss = m(x, target)
    ref_loss = F.cross_entropy(F.linear(xr, ref_w), target, reduction="none")

    atol = 5e-1
    assert torch.allclose(loss.float(), ref_loss.float(), atol=atol, rtol=5e-2)

    loss.sum().backward()
    ref_loss.sum().backward()
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=atol, rtol=5e-2)
    assert torch.allclose(m.weight.grad.float(), ref_w.grad.float(), atol=atol, rtol=5e-2)
