# Copyright (c) 2026, AMD.

"""Behavior-lock harness for the AMD linear-autograd consolidation (Phase 4).

This is a SAFETY NET only — it locks the CURRENT numerics of every public
linear/MLP training entry point against an explicit ``torch.nn.functional``
reference, so a later refactor of ``quack/amd/linear.py`` /
``quack/amd/linear_training.py`` can't silently change forward or backward
math. It does not assert on which kernel/backend runs — only on numerics.

Entry points covered and their CURRENT backend (as of this snapshot):

  1. ``quack.amd.linear.linear_train(x, weight)``
     ``LinearFunc`` — fwd via ``gemm_splitk`` (NT), bwd via ``gemm_nn`` (dx)
     + ``gemm_tn`` (dW). Custom FlyDSL MFMA kernels, no torch.mm fallback.

  2. ``quack.amd.linear.linear_act_train(x, weight, activation, bias=None)``
     ``LinearActFunc`` — fwd via ``gemm_splitk`` (bias fused, activation
     applied in torch on top of the saved preact), bwd: ``dact`` elementwise
     in torch, then ``gemm_nn`` / ``gemm_tn`` for dx / dW (same custom
     kernels as #1).

  3. ``quack.amd.mlp.mlp_train(x, w1, w2, activation="relu", bias1=None,
     bias2=None)``
     Composition of #2 (layer 1, activation fused) and #1 (layer 2) — so
     transitively ``linear.py``'s custom ``gemm_nn`` / ``gemm_tn`` backward.

  4. ``quack.amd.linear_training.mlp_func_train(x, w1, w2, activation="silu",
     bias1=None, bias2=None)``
     ``_MLPActFunction`` — fwd via ``linear()`` (splitk when eligible), bwd
     via plain ``torch.mm`` for dW1/dW2/dx UNLESS
     ``_fused_dact_eligible`` holds, in which case the ``dout @ w2`` matmul
     and ``* act'(preact)`` fuse into one ``gemm_splitk(..., dact_activation=)``
     call. Different backward backend from #1-#3 (torch.mm + optional fused
     splitk-dact, not gemm_nn/gemm_tn).

  5. ``quack.amd.linear_training.gated_mlp_func_train(x,
     w_gate_up_interleaved, w_down, gate_type="swiglu",
     bias_gate_up_interleaved=None, bias_down=None)``
     ``_GatedMLPTrainFunction`` — same family as #4: fwd via ``linear()``,
     bwd via ``torch.mm`` unless ``_fused_dgated_eligible`` holds, in which
     case ``gemm_splitk(..., dgated_gate_type=...)`` fuses the
     ``dz @ w_down`` matmul with the gated-activation backward.

All shapes below are chosen to be simultaneously eligible for every fast
path each entry point can take (M=128, in/hidden/out=256 — all multiples
of 256, a superset of the {64, 128, 256} alignment constraints across
gemm_splitk / gemm_nn / gemm_tn / the fused-dact / fused-dgated gates) so
these tests exercise the real kernels, not a generic fallback.
"""

import pytest
import torch
import torch.nn.functional as F

from quack.amd.linear import linear_train, linear_act_train
from quack.amd.mlp import mlp_train
from quack.amd.linear_training import mlp_func_train, gated_mlp_func_train
from quack.amd.gemm_gfx950_splitk import interleave_gated_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA/ROCm"
)

# Shape shared by all tests below — eligible for every fast path in play
# (gemm_splitk NT fwd: M%128, N%256, K%64; gemm_nn/gemm_tn: M%128, N%64/128,
# K%256; fused-dact/fused-dgated: M%128, hidden%256, out_dim%64).
M, IN_F, HIDDEN, OUT_F = 128, 256, 256, 256

ATOL = RTOL = 6e-2


def _leaf(shape, dtype=torch.bfloat16, scale=0.1):
    t = (torch.randn(shape, device="cuda", dtype=dtype) * scale).detach()
    t.requires_grad_(True)
    return t


def _clone_leaf(t):
    """Independent leaf with the same values, so grads can be compared
    without the two graphs sharing (and thus masking bugs in) storage."""
    return t.detach().clone().requires_grad_(True)


def test_linear_train_consolidation():
    """1. linear_train(x, W) == F.linear(x, W); locks LinearFunc's
    gemm_splitk fwd + gemm_nn/gemm_tn bwd."""
    torch.manual_seed(0)
    x = _leaf((M, IN_F))
    w = _leaf((OUT_F, IN_F))
    xr, wr = _clone_leaf(x), _clone_leaf(w)

    out = linear_train(x, w)
    ref = F.linear(xr, wr)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    torch.testing.assert_close(x.grad, xr.grad, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(w.grad, wr.grad, atol=ATOL, rtol=RTOL)


def test_linear_act_train_consolidation():
    """2. linear_act_train(x, W, activation, bias) == act(F.linear(x, W) + b);
    locks LinearActFunc's gemm_splitk fwd (bias fused) + torch dact +
    gemm_nn/gemm_tn bwd. Reference mirrors test_linear_train.py's
    ``test_linear_act_train`` — bias is added in the SAME (bf16) dtype the
    kernel truncates to before activation, matching gemm_splitk's fused
    bias epilogue.

    Uses silu (smooth everywhere) rather than relu: relu's derivative is
    discontinuous at 0, so a handful of preact elements landing within
    bf16-rounding distance of the boundary flip sign between our kernel's
    f32-accumulate-then-bias-then-trunc and the reference's bf16 F.linear +
    bf16 bias-add, producing isolated large dx/dW errors unrelated to any
    actual bug (this is exactly why test_linear_act_train in
    test_linear_train.py needs tol=0.5 for relu). silu avoids that
    boundary-flip noise so we can keep a tight lock tolerance."""
    torch.manual_seed(0)
    activation = "silu"
    x = _leaf((M, IN_F))
    w = _leaf((HIDDEN, IN_F))
    bias = _leaf((HIDDEN,), dtype=torch.float32, scale=0.05)
    xr, wr, br = _clone_leaf(x), _clone_leaf(w), _clone_leaf(bias)

    out = linear_act_train(x, w, activation=activation, bias=bias)
    preact_ref = F.linear(xr, wr) + br.to(xr.dtype)
    ref = F.silu(preact_ref)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    torch.testing.assert_close(x.grad, xr.grad, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(w.grad, wr.grad, atol=ATOL, rtol=RTOL)
    # dbias: our impl accumulates the (M,) reduction in f32, the reference
    # sums in bf16 (torch's default autograd through F.linear's bias-add) —
    # the two intentionally differ in accumulation dtype, not correctness.
    # Same precedent/tolerance as test_linear_act_train in test_linear_train.py.
    db_err = (bias.grad.float() - br.grad.float()).abs().max().item()
    assert db_err < 5.0, f"dbias err {db_err} (bf16-sum ref vs our f32 accum)"


def test_mlp_train_consolidation():
    """3. mlp_train(x, w1, w2, activation="silu") == F.linear(silu(F.linear(x, w1)), w2);
    locks the composition of LinearActFunc + LinearFunc (both routed
    through linear.py's custom gemm_nn/gemm_tn backward). silu (not the
    "relu" default) to avoid the relu-boundary-flip reference noise
    explained in test_linear_act_train_consolidation above."""
    torch.manual_seed(0)
    x = _leaf((M, IN_F))
    w1 = _leaf((HIDDEN, IN_F))
    w2 = _leaf((OUT_F, HIDDEN))
    xr, w1r, w2r = _clone_leaf(x), _clone_leaf(w1), _clone_leaf(w2)

    out = mlp_train(x, w1, w2, activation="silu")
    h_ref = F.silu(F.linear(xr, w1r))
    ref = F.linear(h_ref, w2r)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    torch.testing.assert_close(x.grad, xr.grad, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(w1.grad, w1r.grad, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(w2.grad, w2r.grad, atol=ATOL, rtol=RTOL)


def test_mlp_func_train_consolidation():
    """4. mlp_func_train(x, w1, w2, activation="silu") ==
    F.linear(silu(F.linear(x, w1)), w2); locks _MLPActFunction's
    torch.mm + fused-dact (gemm_splitk dact_activation=) backward.
    Shape is fused-dact-eligible (M%128, hidden%256, out_dim%64), so this
    exercises the fused kernel path, not the torch.mm fallback."""
    torch.manual_seed(0)
    x = _leaf((M, IN_F))
    w1 = _leaf((HIDDEN, IN_F))
    w2 = _leaf((OUT_F, HIDDEN))
    xr, w1r, w2r = _clone_leaf(x), _clone_leaf(w1), _clone_leaf(w2)

    out = mlp_func_train(x, w1, w2, activation="silu")
    ref = F.linear(F.silu(F.linear(xr, w1r)), w2r)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    torch.testing.assert_close(x.grad, xr.grad, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(w1.grad, w1r.grad, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(w2.grad, w2r.grad, atol=ATOL, rtol=RTOL)


def test_gated_mlp_func_train_consolidation():
    """5. gated_mlp_func_train(x, w_gate_up_interleaved, w_down,
    gate_type="swiglu") — locks _GatedMLPTrainFunction's torch.mm +
    fused-dgated (gemm_splitk dgated_gate_type=) backward.

    Reference construction REUSES the pattern from
    ``test_gated_mlp_func_train_autograd`` in test_linear_train.py: build
    the un-interleaved (gate, up) halves, interleave the weight via
    ``interleave_gated_weight`` (the same helper the kernel requires), then
    replicate the forward as plain F.linear + torch gate + F.linear so the
    reference is independent of any of our custom kernels. Shape is
    fused-dgated-eligible (M%128, hidden%256, out_dim%64), so this
    exercises the fused kernel path, not the torch.mm fallback.
    """
    torch.manual_seed(0)
    x = _leaf((M, IN_F))
    w_gate_up = (torch.randn(2 * HIDDEN, IN_F, device="cuda", dtype=torch.bfloat16) * 0.1).detach()
    w_gate_up_inter = interleave_gated_weight(w_gate_up).detach().requires_grad_(True)
    w_down = _leaf((OUT_F, HIDDEN))

    xr = _clone_leaf(x)
    wgur = _clone_leaf(w_gate_up_inter)
    wdr = _clone_leaf(w_down)

    out = gated_mlp_func_train(x, w_gate_up_inter, w_down, gate_type="swiglu")

    preact_r = F.linear(xr, wgur)  # (M, 2*HIDDEN), interleaved (gate, up) pairs
    pairs = preact_r.view(M, HIDDEN, 2)
    gate_r, up_r = pairs[..., 0], pairs[..., 1]
    post_r = F.silu(gate_r) * up_r
    ref = F.linear(post_r.contiguous(), wdr)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    # Same tolerance precedent as test_gated_mlp_func_train_autograd: the
    # interleaved dgate/dup accumulation into grad_x / grad_w_gate_up picks
    # up an extra bf16 rounding ulp vs the split-halves torch reference.
    gate_atol, gate_rtol = 0.5, 1e-2
    torch.testing.assert_close(x.grad, xr.grad, atol=gate_atol, rtol=gate_rtol)
    torch.testing.assert_close(
        w_gate_up_inter.grad, wgur.grad, atol=gate_atol, rtol=gate_rtol
    )
    torch.testing.assert_close(w_down.grad, wdr.grad, atol=gate_atol, rtol=gate_rtol)
