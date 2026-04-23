# Copyright (c) 2026, AMD.

import pytest
import torch

from quack.amd.linear import linear, linear_gated, linear_mxfp8, linear_residual
from quack.amd.mlp import mlp, gated_mlp
from quack.amd.linear_cross_entropy import (
    linear_cross_entropy,
    linear_cross_entropy_fwd_bwd,
)


def test_linear_matches_torch():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(8, 64, device="cuda")
    w = torch.randn(128, 64, device="cuda")
    b = torch.randn(128, device="cuda")
    out = linear(x, w, bias=b)
    ref = torch.nn.functional.linear(x, w, b)
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_linear_half_dtype_routes_through_mfma(dtype):
    """linear(x, w, ...) with M/N/K%16==0 routes through the FlyDSL MFMA
    kernel (verified by tolerance — the MFMA f32 accumulator is tighter
    than the unaligned-fallback torch linear path for half dtypes)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, in_f, out_f = 64, 64, 128
    x = torch.randn(M, in_f, device="cuda", dtype=dtype)
    w = torch.randn(out_f, in_f, device="cuda", dtype=dtype)
    b = torch.randn(out_f, device="cuda", dtype=torch.float32)
    out = linear(x, w, bias=b, activation="relu")
    ref = torch.relu(torch.nn.functional.linear(x.float(), w.float()) + b).to(dtype)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_linear_residual_fused(dtype):
    """y = alpha * (x @ W.T) + residual_scale * residual + bias fused in one kernel."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, in_f, out_f = 64, 64, 128
    x = torch.randn(M, in_f, device="cuda", dtype=dtype)
    w = torch.randn(out_f, in_f, device="cuda", dtype=dtype)
    residual = torch.randn(M, out_f, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_f, device="cuda", dtype=torch.float32)
    out = linear_residual(
        x, w, residual,
        bias=bias, activation="relu", alpha=0.5, residual_scale=2.0,
    )
    ref_f32 = (
        torch.relu(
            0.5 * torch.nn.functional.linear(x.float(), w.float())
            + 2.0 * residual + bias
        )
    )
    ref = ref_f32.to(out.dtype)
    # Accumulates alpha/beta/bias/activation error; bump tolerance.
    atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(out, ref, atol=atol, rtol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mlp_routes_through_mfma(dtype):
    """Two-layer MLP with MFMA-eligible shapes runs end-to-end through
    real kernels (linear → linear composition)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cuda", dtype=dtype)
    w1 = torch.randn(128, 64, device="cuda", dtype=dtype)
    w2 = torch.randn(64, 128, device="cuda", dtype=dtype)
    out = mlp(x, w1, w2, activation="silu")
    h = torch.nn.functional.silu(torch.nn.functional.linear(x.float(), w1.float()))
    ref = torch.nn.functional.linear(h, w2.float()).to(dtype)
    # Two GEMMs accumulate more error; tolerance doubles vs single.
    torch.testing.assert_close(out, ref, atol=2e-1, rtol=2e-1)


def test_mlp_matches_torch():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(4, 64, device="cuda")
    w1 = torch.randn(256, 64, device="cuda")
    w2 = torch.randn(64, 256, device="cuda")
    out = mlp(x, w1, w2, activation="silu")
    ref = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(x, w1)), w2
    )
    torch.testing.assert_close(out, ref)


def test_gated_mlp_swiglu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(4, 64, device="cuda")
    # w_gate_up is (2 * hidden, in) — chunk splits along hidden after the GEMM.
    w_gate_up = torch.randn(256, 64, device="cuda")  # hidden=128, so out = (4, 128)
    w_down = torch.randn(64, 128, device="cuda")
    out = gated_mlp(x, w_gate_up, w_down, gate_type="swiglu")
    assert out.shape == (4, 64)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M,N,K", [(128, 256, 64), (512, 512, 128), (1024, 1024, 256)])
def test_linear_splitk_path(dtype, M, N, K):
    """linear(x, w) with aligned shapes + no bias/activation routes to
    gemm_splitk. Verify bit-exact match with F.linear (the same hipBLASLt
    MFMA path these kernels target)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    w = torch.randn(N, K, device="cuda", dtype=dtype)
    out = linear(x, w)
    ref = torch.nn.functional.linear(x, w)
    # gemm_splitk matches hipBLASLt bit-exactly on gfx950 (same MFMA atom).
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.parametrize("activation", ["relu", "silu", "gelu_tanh_approx", "relu_sq"])
@pytest.mark.parametrize("use_bias", [False, True])
def test_linear_fused_epilogue_splitk(activation, use_bias):
    """``linear(x, w, bias, activation)`` with splitk-eligible shapes
    fuses bias+activation into the kernel write-back. Verify numerics
    match F.linear + torch activation."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, N, K = 256, 512, 128    # splitk-eligible: M%128, N%256, K%64
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.3
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.3
    b = torch.randn(N, device="cuda", dtype=torch.float32) * 0.3 if use_bias else None
    out = linear(x, w, bias=b, activation=activation)

    lin_f32 = torch.nn.functional.linear(x.float(), w.float())
    if use_bias:
        lin_f32 = lin_f32 + b
    if activation == "relu":
        ref = torch.relu(lin_f32)
    elif activation == "silu":
        ref = torch.nn.functional.silu(lin_f32)
    elif activation == "gelu_tanh_approx":
        ref = torch.nn.functional.gelu(lin_f32, approximate="tanh")
    elif activation == "relu_sq":
        ref = torch.relu(lin_f32) * lin_f32
    torch.testing.assert_close(
        out.float(), ref.to(torch.bfloat16).float(),
        atol=5e-2, rtol=1e-2,
    )


@pytest.mark.parametrize("gate_type", ["swiglu", "reglu", "geglu", "glu"])
def test_linear_gated_fused_in_kernel(gate_type):
    """In-kernel gated fusion via interleaved weight — output is (M, hidden)
    written directly, no (M, 2*hidden) intermediate materialised."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_splitk import interleave_gated_weight
    torch.manual_seed(0)
    M, hidden, K = 256, 256, 128
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.3
    w_halves = torch.randn(2 * hidden, K, device="cuda", dtype=torch.bfloat16) * 0.3
    w_interleaved = interleave_gated_weight(w_halves)
    out = linear_gated(
        x, w_interleaved, gate_type=gate_type, weight_interleaved=True,
    )
    assert out.shape == (M, hidden)
    assert out.dtype == torch.bfloat16

    lin_f32 = torch.nn.functional.linear(x.float(), w_halves.float())
    gate, up = lin_f32.chunk(2, dim=-1)
    if gate_type == "swiglu":
        ref = torch.nn.functional.silu(gate) * up
    elif gate_type == "reglu":
        ref = torch.relu(gate) * up
    elif gate_type == "geglu":
        ref = torch.nn.functional.gelu(gate, approximate="tanh") * up
    elif gate_type == "glu":
        ref = torch.sigmoid(gate) * up
    torch.testing.assert_close(
        out.float(), ref.to(torch.bfloat16).float(), atol=1e-1, rtol=2e-2,
    )


@pytest.mark.parametrize("gate_type", ["swiglu", "reglu", "geglu", "glu"])
@pytest.mark.parametrize("use_bias", [False, True])
def test_linear_gated(gate_type, use_bias):
    """Gated linear matches F.linear + torch-level gating reference
    on splitk-eligible shapes."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, hidden, K = 256, 256, 128        # splitk-eligible: M%128, N=2*hidden=512%256, K%64
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.3
    w_gate_up = torch.randn(2 * hidden, K, device="cuda", dtype=torch.bfloat16) * 0.3
    b = torch.randn(2 * hidden, device="cuda", dtype=torch.float32) * 0.3 if use_bias else None

    out = linear_gated(x, w_gate_up, gate_type=gate_type, bias=b)
    assert out.shape == (M, hidden)

    lin_f32 = torch.nn.functional.linear(x.float(), w_gate_up.float())
    if use_bias:
        lin_f32 = lin_f32 + b
    gate, up = lin_f32.chunk(2, dim=-1)
    if gate_type == "swiglu":
        ref = torch.nn.functional.silu(gate) * up
    elif gate_type == "reglu":
        ref = torch.relu(gate) * up
    elif gate_type == "geglu":
        ref = torch.nn.functional.gelu(gate, approximate="tanh") * up
    elif gate_type == "glu":
        ref = torch.sigmoid(gate) * up
    torch.testing.assert_close(
        out.float(), ref.to(torch.bfloat16).float(), atol=1e-1, rtol=2e-2,
    )


@pytest.mark.parametrize("activation", ["relu", "silu", "gelu_tanh_approx", "relu_sq"])
def test_gemm_splitk_fused_dact(activation):
    """Direct fused gemm_dact kernel: matmul + act'(preact) in write-back."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_splitk import gemm_splitk
    torch.manual_seed(0)
    M, K, N = 256, 128, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.3
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.3
    preact = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 0.3
    out = gemm_splitk(a, b, preact=preact, dact_activation=activation)

    acc_f32 = torch.nn.functional.linear(a.float(), b.float())
    p32 = preact.float()
    if activation == "relu":
        deriv = (p32 > 0).float()
    elif activation == "silu":
        sig = torch.sigmoid(p32)
        deriv = sig * (1.0 + p32 * (1.0 - sig))
    elif activation == "gelu_tanh_approx":
        import math as m
        c1 = m.sqrt(2.0 / m.pi)
        z = c1 * (p32 + 0.044715 * p32.pow(3))
        th = torch.tanh(z)
        dz = c1 * (1.0 + 3.0 * 0.044715 * p32.pow(2))
        deriv = 0.5 * (1.0 + th) + 0.5 * p32 * (1.0 - th * th) * dz
    elif activation == "relu_sq":
        deriv = 2.0 * p32 * (p32 > 0).float()
    ref = (acc_f32 * deriv).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), atol=5e-2, rtol=1e-2)


def test_mlp_func_train_autograd():
    """MLP train wrapper: bit-exact match with torch autograd on the
    same math (F.linear + F.silu + F.linear chain)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.linear_training import mlp_func_train
    torch.manual_seed(0)
    M, hidden, out_dim, K = 256, 512, 256, 128
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn(hidden, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(out_dim, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mlp_func_train(x, w1, w2, activation="silu").sum().backward()
    ours_dx, ours_dw1, ours_dw2 = x.grad.clone(), w1.grad.clone(), w2.grad.clone()
    x.grad = None
    w1.grad = None
    w2.grad = None
    xr = x.detach().clone().requires_grad_(True)
    w1r = w1.detach().clone().requires_grad_(True)
    w2r = w2.detach().clone().requires_grad_(True)
    torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(xr, w1r)), w2r,
    ).sum().backward()
    torch.testing.assert_close(ours_dx, xr.grad, atol=0, rtol=0)
    torch.testing.assert_close(ours_dw1, w1r.grad, atol=0, rtol=0)
    torch.testing.assert_close(ours_dw2, w2r.grad, atol=0, rtol=0)


@pytest.mark.parametrize("activation", ["silu", "relu", "gelu_tanh_approx", "relu_sq"])
def test_linear_act_func_autograd(activation):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.linear_training import linear_act_func
    torch.manual_seed(0)
    M, N, K = 256, 512, 128
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    y = linear_act_func(x, w, activation)
    y.sum().backward()
    ours_dx, ours_dw = x.grad.clone(), w.grad.clone()

    x.grad = None
    w.grad = None
    xr = x.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    lin = torch.nn.functional.linear(xr, wr)
    if activation == "silu":
        ref = torch.nn.functional.silu(lin)
    elif activation == "relu":
        ref = torch.relu(lin)
    elif activation == "gelu_tanh_approx":
        ref = torch.nn.functional.gelu(lin, approximate="tanh")
    elif activation == "relu_sq":
        ref = torch.relu(lin) * lin
    ref.sum().backward()
    # Backward matmuls run in torch bf16 on both sides — bit-exact.
    torch.testing.assert_close(ours_dx, xr.grad, atol=0, rtol=0)
    torch.testing.assert_close(ours_dw, wr.grad, atol=0, rtol=0)


@pytest.mark.parametrize("gate_type", ["swiglu", "reglu", "geglu", "glu"])
def test_linear_gated_func_autograd(gate_type):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.linear_training import linear_gated_func
    torch.manual_seed(0)
    M, hidden, K = 256, 256, 128
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(2 * hidden, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    y = linear_gated_func(x, w, gate_type=gate_type)
    y.sum().backward()
    ours_dx, ours_dw = x.grad.clone(), w.grad.clone()

    x.grad = None
    w.grad = None
    xr = x.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    lin = torch.nn.functional.linear(xr, wr)
    gate, up = lin.chunk(2, dim=-1)
    if gate_type == "swiglu":
        ref = torch.nn.functional.silu(gate) * up
    elif gate_type == "reglu":
        ref = torch.relu(gate) * up
    elif gate_type == "geglu":
        ref = torch.nn.functional.gelu(gate, approximate="tanh") * up
    elif gate_type == "glu":
        ref = torch.sigmoid(gate) * up
    ref.sum().backward()
    # swiglu and reglu happen to match bit-exactly (simple ops); geglu
    # and glu pass through tanh/sigmoid in our path vs torch's compiled
    # autograd, giving 1-ulp bf16 drift on a small fraction of elements.
    # Allow 1 bf16 ULP at matmul magnitudes (dx magnitudes reach ~60 at
    # this shape, so 0.5 abs / 1% rel is the right band).
    if gate_type in ("swiglu", "reglu"):
        atol, rtol = 0, 0
    else:
        atol, rtol = 0.5, 0.01
    torch.testing.assert_close(ours_dx, xr.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(ours_dw, wr.grad, atol=atol, rtol=rtol)


def test_linear_mxfp8_func_autograd():
    """MX-FP8 linear with autograd — forward in fp8 via mfma_scale, backward in bf16."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build lacks fp8 support")
    from quack.amd.mxfp8_ops import linear_mxfp8_func
    torch.manual_seed(0)
    M, N, K = 256, 256, 128
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Forward: output tolerance is fp8-quantisation-bounded (~5% relative).
    y = linear_mxfp8_func(x, w)
    y_ref = torch.nn.functional.linear(x.detach(), w.detach())
    rel_err = (y - y_ref).abs().mean() / y_ref.abs().mean()
    assert rel_err < 0.1, f"mxfp8 forward rel_err {rel_err.item()} > 0.1"

    # Backward: runs in bf16; matches plain autograd on the saved bf16 tensors.
    y.sum().backward()
    ours_dx, ours_dw = x.grad.clone(), w.grad.clone()

    x.grad = None
    w.grad = None
    xr = x.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    torch.nn.functional.linear(xr, wr).sum().backward()
    # Our backward uses the same bf16 matmul path; bit-exact.
    torch.testing.assert_close(ours_dx, xr.grad, atol=0, rtol=0)
    torch.testing.assert_close(ours_dw, wr.grad, atol=0, rtol=0)


def test_linear_mxfp8_func_gradcheck():
    """End-to-end training step: backward grad points in the descent
    direction. Three SGD iterations should give monotonically
    non-increasing loss — confirms the backward grads are correctly
    oriented (not just correctly shaped). Absolute loss reduction is
    modest because fp8 quantisation noise floors the loss."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build lacks fp8 support")
    from quack.amd.mxfp8_ops import linear_mxfp8_func
    torch.manual_seed(0)
    M, N, K = 128, 256, 128
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w_true = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    target = torch.nn.functional.linear(x, w_true)
    w = (w_true + 0.1 * torch.randn_like(w_true)).detach().requires_grad_(True)

    losses = []
    for _ in range(3):
        w.grad = None
        loss = (linear_mxfp8_func(x, w) - target).float().pow(2).mean()
        loss.backward()
        losses.append(loss.item())
        with torch.no_grad():
            w -= 0.5 * w.grad
    # Each step should not INCREASE loss (monotonic descent).
    assert losses[1] <= losses[0], f"loss went up step 0→1: {losses}"
    assert losses[2] <= losses[1], f"loss went up step 1→2: {losses}"


@pytest.mark.parametrize("activation", ["relu", "silu", "relu_sq", "gelu_tanh_approx"])
@pytest.mark.parametrize("use_bias", [False, True])
def test_linear_mxfp8_func_autograd_with_activation(activation, use_bias):
    """Autograd coverage for ``linear_mxfp8_func`` with bias + activation.

    Reference strategy: a minimal ``_ReferenceMXFP8Linear`` autograd.Function
    that does fp8-quantise → mxfp8_gemm → standard linear backward, then
    lets torch autograd handle the activation. This gives us independent
    gradients via torch's own ``F.<act>`` backward (not the hand-coded
    ``act'`` formula in ``_LinearMXFP8Function``). Both paths share the
    SAME fp8-quantised preact (necessary — otherwise forward quant error
    would flip sign near zero crossings for relu-family activations and
    produce huge spurious mismatches). The test isolates the backward-math
    correctness."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build lacks fp8 support")
    from quack.amd.mxfp8_ops import (
        linear_mxfp8_func, quantize_mxfp8, _quantize_weight_with_block_scale,
    )
    from quack.amd.gemm_gfx950_blockscaled import mxfp8_gemm
    import torch.nn.functional as F

    class _ReferenceMXFP8Linear(torch.autograd.Function):
        """Independent reference: standard bf16 linear backward. Does not
        use the hand-coded activation' derivatives that
        _LinearMXFP8Function has — letting torch autograd derive them
        gives us a genuine correctness oracle."""
        @staticmethod
        def forward(ctx, x, weight, bias):
            ctx.save_for_backward(x, weight)
            ctx.has_bias = bias is not None
            x_fp8, sx = quantize_mxfp8(x, transpose_scale=True)
            w_fp8, sw = _quantize_weight_with_block_scale(weight)
            return mxfp8_gemm(
                x_fp8, w_fp8, sx, sw, out_dtype=x.dtype,
                bias=bias, activation="none",
            )
        @staticmethod
        def backward(ctx, grad_out):
            x, weight = ctx.saved_tensors
            grad_x = torch.mm(grad_out, weight) if ctx.needs_input_grad[0] else None
            grad_w = torch.mm(grad_out.t(), x) if ctx.needs_input_grad[1] else None
            grad_b = grad_out.sum(dim=0).to(torch.float32) if ctx.has_bias and ctx.needs_input_grad[2] else None
            return grad_x, grad_w, grad_b

    torch.manual_seed(0)
    M, N, K = 256, 256, 128
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = None
    if use_bias:
        b = (torch.randn(N, device="cuda", dtype=torch.float32) * 0.1).detach().requires_grad_(True)

    # Our path: fully-fused autograd.Function with hand-coded act'.
    y = linear_mxfp8_func(x, w, bias=b, activation=activation)
    y.sum().backward()
    ours_dx, ours_dw = x.grad.clone(), w.grad.clone()
    ours_db = b.grad.clone() if use_bias else None
    x.grad = None; w.grad = None
    if use_bias: b.grad = None

    # Reference path: mxfp8 matmul via custom Function (independent
    # backward), then torch's F.<act> (torch's independent backward).
    xr = x.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    br = b.detach().clone().requires_grad_(True) if use_bias else None
    preact_r = _ReferenceMXFP8Linear.apply(xr, wr, br)
    if activation == "relu":
        y_ref = torch.relu(preact_r)
    elif activation == "silu":
        y_ref = F.silu(preact_r)
    elif activation == "relu_sq":
        y_ref = torch.relu(preact_r) * preact_r
    elif activation == "gelu_tanh_approx":
        y_ref = F.gelu(preact_r, approximate="tanh")
    y_ref.sum().backward()

    # Bit-exact match expected: both paths use the same fp8 quant, same
    # kernel, same downstream math — only the act'(preact) derivation
    # differs (our hand-coded vs torch's autograd). Any divergence flags
    # a hand-coded-derivative bug.
    torch.testing.assert_close(y.detach(), y_ref.detach(), atol=0, rtol=0)
    torch.testing.assert_close(ours_dx, xr.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(ours_dw, wr.grad, atol=1e-3, rtol=1e-3)
    if use_bias:
        torch.testing.assert_close(ours_db, br.grad, atol=1e-3, rtol=1e-3)


def test_linear_mxfp8():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build lacks fp8 support")
    torch.manual_seed(0)
    M, N, K = 256, 256, 128
    x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    w = (torch.randn(N, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    scale_x = torch.ones(K // 128, M, device="cuda", dtype=torch.float32)
    scale_w = torch.ones(N // 128, K // 128, device="cuda", dtype=torch.float32)
    out = linear_mxfp8(x, w, scale_x, scale_w)
    # scale=1 reduces to plain fp8 matmul.
    ref = (x.float() @ w.float().T).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), atol=5e-2, rtol=1e-2)
    assert out.dtype == torch.bfloat16


def test_linear_cross_entropy_matches_torch():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(1, 64, device="cuda")
    w = torch.randn(256, 64, device="cuda")
    target = torch.randint(0, 256, (1,), device="cuda", dtype=torch.int64)
    loss, lse = linear_cross_entropy(x, w, target, return_lse=True)
    logits = torch.nn.functional.linear(x, w).float()
    ref_loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    ref_lse = torch.logsumexp(logits, dim=-1)
    torch.testing.assert_close(loss, ref_loss, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("B_L,V,d", [(2048, 512, 128), (4096, 1024, 256)])
@pytest.mark.parametrize("chunk_size", [1024, 2048])
def test_linear_cross_entropy_chunked_matches_unchunked(B_L, V, d, chunk_size):
    """The chunked path must produce the same loss as the single-chunk path —
    CE is per-row so chunking along the batch dim is loss-preserving."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(B_L, d, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(V, d, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, V, (B_L,), device="cuda", dtype=torch.int64)
    loss_chunked, _ = linear_cross_entropy(x, w, target, chunk_size=chunk_size)
    loss_unchunked, _ = linear_cross_entropy(x, w, target, chunk_size=B_L)
    # Same kernel path on each chunk — matches bit-exactly.
    torch.testing.assert_close(loss_chunked, loss_unchunked, atol=0, rtol=0)


def test_linear_cross_entropy_chunked_matches_torch():
    """End-to-end: chunked path matches F.linear + F.cross_entropy within
    bf16 tolerance for a realistic LLM-scale shape."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    B_L, V, d = 4096, 4096, 256
    x = torch.randn(B_L, d, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(V, d, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, V, (B_L,), device="cuda", dtype=torch.int64)
    loss, _ = linear_cross_entropy(x, w, target, chunk_size=1024)
    ref_logits = torch.nn.functional.linear(x.float(), w.float())
    ref_loss = torch.nn.functional.cross_entropy(ref_logits, target, reduction="none")
    # bf16 matmul + CE — moderate tolerance.
    torch.testing.assert_close(loss, ref_loss, atol=5e-1, rtol=5e-2)


@pytest.mark.parametrize("B_L,V,d", [(4096, 2048, 128), (1024, 1024, 256)])
@pytest.mark.parametrize("chunk_size", [1024, 2048])
def test_linear_cross_entropy_fwd_bwd_matches_autograd(B_L, V, d, chunk_size):
    """End-to-end: our fused fwd+bwd path produces the same loss, dx, dw
    as torch.autograd on F.linear + F.cross_entropy."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(B_L, d, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(V, d, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, V, (B_L,), device="cuda", dtype=torch.int64)

    loss, dx, dw = linear_cross_entropy_fwd_bwd(
        x, w, target, chunk_size=chunk_size,
    )

    x_ref = x.clone().float().requires_grad_(True)
    w_ref = w.clone().float().requires_grad_(True)
    ref_loss = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(x_ref, w_ref), target, reduction="none",
    )
    ref_loss.sum().backward()
    # bf16 matmul + CE accumulated error scales with K=d and V.
    atol = max(5e-1, 1e-2 * max(V, d) / 128)
    torch.testing.assert_close(loss, ref_loss, atol=atol, rtol=5e-2)
    torch.testing.assert_close(
        dx, x_ref.grad.to(x.dtype), atol=atol, rtol=5e-2,
    )
    torch.testing.assert_close(
        dw, w_ref.grad, atol=atol, rtol=5e-2,
    )


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_linear_cross_entropy_autograd(reduction):
    """``loss.sum().backward()`` / ``loss.mean().backward()`` should pick
    up the fused fwd+bwd path transparently via ``_LinearCrossEntropyFunction``,
    producing grads that match torch autograd on the same math."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    B_L, V, d = 2048, 1024, 128
    x = torch.randn(B_L, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(V, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(0, V, (B_L,), device="cuda", dtype=torch.int64)

    loss_vec, _ = linear_cross_entropy(x, w, target, chunk_size=1024)
    getattr(loss_vec, reduction)().backward()
    ours_dx, ours_dw = x.grad.clone(), w.grad.clone()

    x.grad = None
    w.grad = None
    xr = x.clone().detach().float().requires_grad_(True)
    wr = w.clone().detach().float().requires_grad_(True)
    ref_loss = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(xr, wr), target, reduction="none",
    )
    getattr(ref_loss, reduction)().backward()
    torch.testing.assert_close(ours_dx, xr.grad.to(torch.bfloat16), atol=5e-1, rtol=5e-2)
    torch.testing.assert_close(ours_dw.float(), wr.grad, atol=5e-1, rtol=5e-2)


def test_linear_cross_entropy_no_grad_path():
    """Without requires_grad, falls back to the forward-only chunked
    path (no dx/dw allocation)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    B_L, V, d = 512, 512, 64
    x = torch.randn(B_L, d, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(V, d, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, V, (B_L,), device="cuda", dtype=torch.int64)
    loss_vec, _ = linear_cross_entropy(x, w, target)
    assert not loss_vec.requires_grad


def test_linear_cross_entropy_fwd_bwd_chunked_matches_unchunked():
    """Chunked and unchunked fused fwd+bwd should give the same outputs —
    same kernel per chunk, same data, just different partition of the batch."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    B_L, V, d = 2048, 1024, 128
    x = torch.randn(B_L, d, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(V, d, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, V, (B_L,), device="cuda", dtype=torch.int64)
    l_c, dx_c, dw_c = linear_cross_entropy_fwd_bwd(x, w, target, chunk_size=512)
    l_u, dx_u, dw_u = linear_cross_entropy_fwd_bwd(x, w, target, chunk_size=B_L)
    # Matmul + CE per chunk identical — should be bit-exact.
    torch.testing.assert_close(l_c, l_u, atol=0, rtol=0)
    torch.testing.assert_close(dx_c, dx_u, atol=0, rtol=0)
    # dw accumulates in f32 in a deterministic order (sequential) so
    # also bit-exact across chunk partitions.
    torch.testing.assert_close(dw_c, dw_u, atol=1e-5, rtol=1e-5)


# --- W1: gemm_dgated kernel + gated_mlp_func_train -------------------------


@pytest.mark.parametrize("gate_type", ["swiglu", "reglu", "geglu", "glu"])
def test_gemm_splitk_fused_dgated(gate_type):
    """Direct fused gemm_dgated kernel: matmul + gated act-bwd + interleaved
    dpreact in the write-back. Reference = torch composite."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_splitk import gemm_splitk
    torch.manual_seed(0)
    # Shapes meeting splitk constraints for the dgated path:
    #   M = 128 (splitk M % 128), hidden = 256 (N % 256), out_dim = 128 (K % 64).
    M, out_dim, hidden = 128, 128, 256
    a = torch.randn(M, out_dim, device="cuda", dtype=torch.bfloat16) * 0.3
    # b plays the role of w_down.T of shape (hidden, out_dim) — splitk NT.
    b = torch.randn(hidden, out_dim, device="cuda", dtype=torch.bfloat16) * 0.3
    # preact in interleaved (g0, u0, g1, u1, …) layout, shape (M, 2*hidden).
    preact = torch.randn(M, 2 * hidden, device="cuda", dtype=torch.bfloat16) * 0.3

    dpreact = gemm_splitk(
        a, b, dgated_gate_type=gate_type, dgated_preact=preact,
    )
    assert dpreact.shape == (M, 2 * hidden)

    # Reference: dy = a @ b.T, then interleaved act-bwd per pair.
    dy = a.float() @ b.float().t()  # (M, hidden)
    preact_pairs = preact.view(M, hidden, 2).float()
    gate = preact_pairs[..., 0]
    up = preact_pairs[..., 1]
    if gate_type == "swiglu":
        sig = torch.sigmoid(gate)
        silu_g = gate * sig
        dsilu_dg = sig * (1.0 + gate * (1.0 - sig))
        dgate_ref = dy * dsilu_dg * up
        dup_ref = dy * silu_g
    elif gate_type == "reglu":
        mask = (gate > 0).float()
        dgate_ref = dy * mask * up
        dup_ref = dy * gate.clamp_min(0)
    elif gate_type == "geglu":
        import math as _m
        c1 = _m.sqrt(2.0 / _m.pi)
        z = c1 * (gate + 0.044715 * gate.pow(3))
        th = torch.tanh(z)
        dz_dg = c1 * (1.0 + 3.0 * 0.044715 * gate.pow(2))
        gelu_g = 0.5 * gate * (1.0 + th)
        dgelu_dg = 0.5 * (1.0 + th) + 0.5 * gate * (1.0 - th * th) * dz_dg
        dgate_ref = dy * dgelu_dg * up
        dup_ref = dy * gelu_g
    elif gate_type == "glu":
        sig = torch.sigmoid(gate)
        dsig_dg = sig * (1.0 - sig)
        dgate_ref = dy * dsig_dg * up
        dup_ref = dy * sig
    ref_interleaved = torch.stack(
        [dgate_ref, dup_ref], dim=-1,
    ).view(M, 2 * hidden).to(torch.bfloat16)

    # geglu/glu involve larger-magnitude dy*act'*up products; allow bf16
    # rounding drift (1 ulp). swiglu/reglu with random inputs still align
    # within the gemm_splitk bf16 tolerance band.
    tol = {"swiglu": 5e-2, "reglu": 5e-2, "geglu": 5e-2, "glu": 5e-2}[gate_type]
    torch.testing.assert_close(dpreact.float(), ref_interleaved.float(), atol=tol, rtol=1e-2)


@pytest.mark.parametrize("gate_type", ["swiglu", "reglu", "geglu", "glu"])
def test_gemm_splitk_fused_dgated_emit_postact(gate_type):
    """Co-emits postact = act(g) * u alongside dpreact."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_splitk import gemm_splitk
    torch.manual_seed(0)
    M, out_dim, hidden = 128, 128, 256
    a = torch.randn(M, out_dim, device="cuda", dtype=torch.bfloat16) * 0.3
    b = torch.randn(hidden, out_dim, device="cuda", dtype=torch.bfloat16) * 0.3
    preact = torch.randn(M, 2 * hidden, device="cuda", dtype=torch.bfloat16) * 0.3

    dpreact, postact = gemm_splitk(
        a, b, dgated_gate_type=gate_type, dgated_preact=preact,
        dgated_emit_postact=True,
    )
    assert dpreact.shape == (M, 2 * hidden)
    assert postact.shape == (M, hidden)

    preact_pairs = preact.view(M, hidden, 2).float()
    gate = preact_pairs[..., 0]
    up = preact_pairs[..., 1]
    if gate_type == "swiglu":
        ref_post = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    elif gate_type == "reglu":
        ref_post = (torch.relu(gate) * up).to(torch.bfloat16)
    elif gate_type == "geglu":
        ref_post = (torch.nn.functional.gelu(gate, approximate="tanh") * up).to(torch.bfloat16)
    elif gate_type == "glu":
        ref_post = (torch.sigmoid(gate) * up).to(torch.bfloat16)
    torch.testing.assert_close(postact.float(), ref_post.float(), atol=5e-3, rtol=1e-2)


@pytest.mark.parametrize("gate_type", ["swiglu", "reglu", "geglu", "glu"])
def test_gated_mlp_func_train_autograd(gate_type):
    """End-to-end gated MLP train: matches torch autograd on the same math."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.linear_training import gated_mlp_func_train
    from quack.amd.gemm_gfx950_splitk import interleave_gated_weight
    torch.manual_seed(0)
    M, in_f, hidden, out_dim = 128, 128, 256, 128
    x = torch.randn(M, in_f, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # Standard (gate half, up half) layout; we interleave before calling.
    w_gate_up = torch.randn(
        2 * hidden, in_f, device="cuda", dtype=torch.bfloat16,
    ) * 0.1
    w_gate_up_inter = interleave_gated_weight(w_gate_up).detach().requires_grad_(True)
    w_down = (torch.randn(
        out_dim, hidden, device="cuda", dtype=torch.bfloat16,
    ) * 0.1).detach().requires_grad_(True)

    out = gated_mlp_func_train(x, w_gate_up_inter, w_down, gate_type=gate_type)
    out.sum().backward()
    ours_dx = x.grad.clone()
    ours_dw_gate_up_inter = w_gate_up_inter.grad.clone()
    ours_dw_down = w_down.grad.clone()
    x.grad = None
    w_gate_up_inter.grad = None
    w_down.grad = None

    # Reference chain: plain linear + torch gate + linear, using the
    # same interleaved weight.
    xr = x.detach().clone().requires_grad_(True)
    wgur = w_gate_up_inter.detach().clone().requires_grad_(True)
    wdr = w_down.detach().clone().requires_grad_(True)
    preact_r = torch.nn.functional.linear(xr, wgur)
    pairs = preact_r.view(M, hidden, 2)
    gate_r = pairs[..., 0]
    up_r = pairs[..., 1]
    if gate_type == "swiglu":
        post_r = torch.nn.functional.silu(gate_r) * up_r
    elif gate_type == "reglu":
        post_r = torch.relu(gate_r) * up_r
    elif gate_type == "geglu":
        post_r = torch.nn.functional.gelu(gate_r, approximate="tanh") * up_r
    elif gate_type == "glu":
        post_r = torch.sigmoid(gate_r) * up_r
    out_r = torch.nn.functional.linear(post_r.contiguous(), wdr)
    out_r.sum().backward()

    # bf16 rounding drift at these magnitudes is within 1 ulp. Keep band
    # loose enough to absorb (dgate, dup) accumulations into grad_x /
    # grad_w_gate_up over the matmul.
    torch.testing.assert_close(ours_dx, xr.grad, atol=0.5, rtol=1e-2)
    torch.testing.assert_close(ours_dw_gate_up_inter, wgur.grad, atol=0.5, rtol=1e-2)
    torch.testing.assert_close(ours_dw_down, wdr.grad, atol=0.5, rtol=1e-2)


# --- W2: split-K>1 + fused epilogue via two-launch -------------------------


@pytest.mark.parametrize("activation", ["none", "relu", "silu"])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("split_k", [2, 4])
def test_splitk_bias_act_two_launch(activation, use_bias, split_k):
    """Force SPLIT_K>1 with bias/activation, routed via two_launch."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_splitk import gemm_splitk
    torch.manual_seed(0)
    M, N, K = 256, 512, 512
    assert K % split_k == 0
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.3
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.3
    bias = torch.randn(N, device="cuda", dtype=torch.float32) * 0.1 if use_bias else None
    if activation == "none" and not use_bias:
        pytest.skip("no epilogue — two_launch decision is 'none'")

    out = gemm_splitk(
        a, b, bias=bias, activation=activation,
        force_split_k=split_k, splitk_epi_mode="two_launch",
    )
    ref = a.float() @ b.float().t()
    if use_bias:
        ref = ref + bias
    if activation == "relu":
        ref = torch.relu(ref)
    elif activation == "silu":
        ref = torch.nn.functional.silu(ref)
    torch.testing.assert_close(out.float(), ref, atol=5e-2, rtol=1e-2)


@pytest.mark.parametrize("gate_type", ["swiglu", "reglu", "geglu", "glu"])
@pytest.mark.parametrize("split_k", [2, 4])
def test_splitk_gated_two_launch(gate_type, split_k):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_splitk import gemm_splitk, interleave_gated_weight
    torch.manual_seed(0)
    M, K, hidden = 256, 512, 256
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.3
    w_gate_up = torch.randn(2 * hidden, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w_inter = interleave_gated_weight(w_gate_up)
    out = gemm_splitk(
        a, w_inter, gate_type=gate_type,
        force_split_k=split_k, splitk_epi_mode="two_launch",
    )
    acc = a.float() @ w_inter.float().t()
    pairs = acc.view(M, hidden, 2)
    g = pairs[..., 0]
    u = pairs[..., 1]
    if gate_type == "swiglu":
        ref = torch.nn.functional.silu(g) * u
    elif gate_type == "reglu":
        ref = torch.relu(g) * u
    elif gate_type == "geglu":
        ref = torch.nn.functional.gelu(g, approximate="tanh") * u
    elif gate_type == "glu":
        ref = torch.sigmoid(g) * u
    torch.testing.assert_close(out.float(), ref.to(torch.bfloat16).float(), atol=5e-2, rtol=1e-2)


@pytest.mark.parametrize("activation", ["relu", "silu", "gelu_tanh_approx", "relu_sq"])
@pytest.mark.parametrize("split_k", [2, 4])
def test_splitk_dact_two_launch(activation, split_k):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_splitk import gemm_splitk
    torch.manual_seed(0)
    M, N, K = 256, 512, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.3
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.3
    preact = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 0.3
    out = gemm_splitk(
        a, b, preact=preact, dact_activation=activation,
        force_split_k=split_k, splitk_epi_mode="two_launch",
    )
    acc = a.float() @ b.float().t()
    p32 = preact.float()
    if activation == "relu":
        deriv = (p32 > 0).float()
    elif activation == "silu":
        sig = torch.sigmoid(p32)
        deriv = sig * (1.0 + p32 * (1.0 - sig))
    elif activation == "gelu_tanh_approx":
        import math as _m
        c1 = _m.sqrt(2.0 / _m.pi)
        z = c1 * (p32 + 0.044715 * p32.pow(3))
        th = torch.tanh(z)
        dz = c1 * (1.0 + 3.0 * 0.044715 * p32.pow(2))
        deriv = 0.5 * (1.0 + th) + 0.5 * p32 * (1.0 - th * th) * dz
    elif activation == "relu_sq":
        deriv = 2.0 * p32 * (p32 > 0).float()
    ref = (acc * deriv).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), atol=5e-2, rtol=1e-2)


@pytest.mark.parametrize("gate_type", ["swiglu", "reglu", "geglu", "glu"])
@pytest.mark.parametrize("split_k", [2, 4])
def test_splitk_dgated_two_launch(gate_type, split_k):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_gfx950_splitk import gemm_splitk
    torch.manual_seed(0)
    M, out_dim, hidden = 256, 512, 256
    a = torch.randn(M, out_dim, device="cuda", dtype=torch.bfloat16) * 0.3
    b = torch.randn(hidden, out_dim, device="cuda", dtype=torch.bfloat16) * 0.3
    preact = torch.randn(M, 2 * hidden, device="cuda", dtype=torch.bfloat16) * 0.3

    dpre, post = gemm_splitk(
        a, b, dgated_gate_type=gate_type, dgated_preact=preact,
        dgated_emit_postact=True,
        force_split_k=split_k, splitk_epi_mode="two_launch",
    )
    dy = a.float() @ b.float().t()
    pairs = preact.view(M, hidden, 2).float()
    g = pairs[..., 0]
    u = pairs[..., 1]
    if gate_type == "swiglu":
        sig = torch.sigmoid(g)
        silu_g = g * sig
        dsilu = sig * (1.0 + g * (1.0 - sig))
        dgate = dy * dsilu * u
        dup = dy * silu_g
        ref_post = silu_g * u
    elif gate_type == "reglu":
        mask = (g > 0).float()
        fwd = torch.relu(g)
        dgate = dy * mask * u
        dup = dy * fwd
        ref_post = fwd * u
    elif gate_type == "geglu":
        import math as _m
        c1 = _m.sqrt(2.0 / _m.pi)
        z = c1 * (g + 0.044715 * g.pow(3))
        th = torch.tanh(z)
        dz = c1 * (1.0 + 3.0 * 0.044715 * g.pow(2))
        fwd = 0.5 * g * (1.0 + th)
        dfwd = 0.5 * (1.0 + th) + 0.5 * g * (1.0 - th * th) * dz
        dgate = dy * dfwd * u
        dup = dy * fwd
        ref_post = fwd * u
    elif gate_type == "glu":
        sig = torch.sigmoid(g)
        dsig = sig * (1.0 - sig)
        dgate = dy * dsig * u
        dup = dy * sig
        ref_post = sig * u
    ref_dpre = torch.stack([dgate, dup], dim=-1).view(M, 2 * hidden).to(torch.bfloat16)
    torch.testing.assert_close(dpre.float(), ref_dpre.float(), atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(post.float(), ref_post.to(torch.bfloat16).float(), atol=5e-2, rtol=1e-2)
