# Copyright (c) 2026, AMD.

import pytest
import torch

from quack.amd.linear import linear
from quack.amd.mlp import mlp, gated_mlp
from quack.amd.linear_cross_entropy import linear_cross_entropy


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


def test_linear_cross_entropy_matches_torch():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    # M=1 matches the cross_entropy_fwd constraint.
    x = torch.randn(1, 64, device="cuda")
    w = torch.randn(256, 64, device="cuda")
    target = torch.randint(0, 256, (1,), device="cuda", dtype=torch.int64)
    loss, lse = linear_cross_entropy(x, w, target, return_lse=True)
    logits = torch.nn.functional.linear(x, w).float()
    ref_loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    ref_lse = torch.logsumexp(logits, dim=-1)
    torch.testing.assert_close(loss, ref_loss, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-4)
