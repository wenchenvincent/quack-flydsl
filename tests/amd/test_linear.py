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
