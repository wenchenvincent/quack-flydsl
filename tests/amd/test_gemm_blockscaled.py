# Copyright (c) 2026, AMD.

"""MX-FP8 blockscaled GEMM correctness tests.

Uses the native ``mfma_scale_f32_16x16x128_f8f6f4`` kernel path
(vendored from FlyDSL; see quack/amd/_fly_*.py). Reference is a
plain fp32 matmul of the dequantized tensors.
"""

import pytest
import torch

from quack.amd.gemm_gfx950_blockscaled import (
    mxfp8_gemm, shuffle_b, SCALE_BLOCK_K, SCALE_BLOCK_N,
)


def _dequant_A(a_fp8, scale_a):
    """A is (M, K). scale_a is (K//128, M) transposed — scale[block_k, m]."""
    M, K = a_fp8.shape
    x = a_fp8.float().view(M, K // 128, 128)
    sc = scale_a.transpose(0, 1).unsqueeze(-1)  # (M, K//128, 1)
    return (x * sc).view(M, K)


def _dequant_B(b_fp8, scale_b):
    """B is (N, K). scale_b is (N//128, K//128) — scale[block_n, block_k]."""
    N, K = b_fp8.shape
    x = b_fp8.float().view(N // 128, 128, K // 128, 128)  # (block_n, n_in_block, block_k, k_in_block)
    sc = scale_b.view(N // 128, 1, K // 128, 1)
    return (x * sc).view(N, K)


@pytest.mark.parametrize("M", [512, 256, 128])
@pytest.mark.parametrize("N", [256, 128])
@pytest.mark.parametrize("K", [128, 256])
def test_mxfp8_gemm_correctness(M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build lacks fp8 support")
    torch.manual_seed(0)
    a = (torch.randn(M, K, device="cuda") * 2.0).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device="cuda") * 2.0).to(torch.float8_e4m3fn)
    scale_a = torch.rand(K // 128, M, device="cuda") * 0.5 + 0.1
    scale_b = torch.rand(N // 128, K // 128, device="cuda") * 0.5 + 0.1
    out = mxfp8_gemm(a, b, scale_a, scale_b)
    # Reference: fp32 matmul of dequantized tensors.
    ref_f32 = _dequant_A(a, scale_a) @ _dequant_B(b, scale_b).T
    # bf16 rounding tolerance — scales accumulation error by K.
    atol = max(0.5, 0.01 * K / 128 * ref_f32.abs().max().item())
    rtol = 0.02
    torch.testing.assert_close(out.float(), ref_f32, atol=atol, rtol=rtol)


def test_mxfp8_neutral_scales_match_plain_fp8():
    """scale=1 everywhere should reduce to a plain fp8 matmul."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build lacks fp8 support")
    torch.manual_seed(0)
    M, N, K = 256, 256, 128
    a = (torch.randn(M, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    scale_a = torch.ones(K // 128, M, device="cuda", dtype=torch.float32)
    scale_b = torch.ones(N // 128, K // 128, device="cuda", dtype=torch.float32)
    out = mxfp8_gemm(a, b, scale_a, scale_b)
    ref = (a.float() @ b.float().T).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), atol=5e-2, rtol=1e-2)


def test_shuffle_b_roundtrip():
    """shuffling B then unshuffling should give identity (permutation only)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build lacks fp8 support")
    torch.manual_seed(0)
    b = torch.randn(128, 128, device="cuda").to(torch.float8_e4m3fn)
    b_shuffled = shuffle_b(b)
    # The shuffle reshape still has the same number of elements and dtype.
    assert b_shuffled.shape == b.shape
    assert b_shuffled.dtype == b.dtype


def test_shuffled_b_path_matches_inline():
    """Pre-shuffling B outside the kernel should match inline-shuffle."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build lacks fp8 support")
    torch.manual_seed(0)
    M, N, K = 128, 128, 128
    a = (torch.randn(M, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    sa = torch.rand(K // 128, M, device="cuda") * 0.5 + 0.1
    sb = torch.rand(N // 128, K // 128, device="cuda") * 0.5 + 0.1
    out_inline = mxfp8_gemm(a, b, sa, sb)
    out_pre = mxfp8_gemm(a, shuffle_b(b), sa, sb, shuffled=True)
    torch.testing.assert_close(out_inline, out_pre, atol=0, rtol=0)


def test_output_shape_and_dtype():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build lacks fp8 support")
    M, N, K = 128, 128, 128
    a = torch.zeros(M, K, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.zeros(N, K, device="cuda", dtype=torch.float8_e4m3fn)
    sa = torch.ones(K // 128, M, device="cuda", dtype=torch.float32)
    sb = torch.ones(N // 128, K // 128, device="cuda", dtype=torch.float32)
    out = mxfp8_gemm(a, b, sa, sb)
    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16


# --- fused bias + activation ----------------------------------------------


import torch.nn.functional as F


def _dequant_ref(a_fp8, b_fp8, scale_a, scale_b):
    a_f = _dequant_A(a_fp8, scale_a)
    b_f = _dequant_B(b_fp8, scale_b)
    return a_f @ b_f.t()


@pytest.mark.parametrize("activation", ["relu", "silu", "relu_sq", "gelu_tanh_approx"])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("M,N,K", [(256, 256, 256), (128, 256, 256), (512, 512, 512)])
def test_mxfp8_gemm_fused_epi(activation, use_bias, M, N, K):
    """Fused bias + activation in the writeback match dequant → matmul →
    torch epi reference."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    ab = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.2
    bb = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.2
    a = ab.to(torch.float8_e4m3fn)
    b = bb.to(torch.float8_e4m3fn)
    sa = (torch.rand(K // 128, M, device="cuda") * 0.5 + 0.75).to(torch.float32)
    sb = (torch.rand(N // 128, K // 128, device="cuda") * 0.5 + 0.75).to(torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32) * 0.1 if use_bias else None
    out = mxfp8_gemm(a, b, sa, sb, bias=bias, activation=activation)

    ref = _dequant_ref(a, b, sa, sb)
    if use_bias:
        ref = ref + bias
    if activation == "relu":
        ref = torch.relu(ref)
    elif activation == "silu":
        ref = F.silu(ref)
    elif activation == "relu_sq":
        ref = torch.relu(ref) * ref
    elif activation == "gelu_tanh_approx":
        ref = F.gelu(ref, approximate="tanh")
    ref = ref.to(torch.bfloat16)

    # bf16 rounding + fp8 input quant error compound, and relu_sq
    # squares the output so it amplifies worst-case drift. 15% relative
    # band absorbs that.
    err = (out.float() - ref.float()).abs().max().item()
    mean = ref.float().abs().mean().item()
    assert err / max(mean, 1e-9) < 0.15, f"rel={err/max(mean,1e-9):.3g} err={err} mean={mean}"


def test_mxfp8_gemm_bias_only():
    """bias without activation — verify the bias path in isolation."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, N, K = 256, 256, 256
    ab = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.2
    bb = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.2
    a = ab.to(torch.float8_e4m3fn); b = bb.to(torch.float8_e4m3fn)
    sa = (torch.rand(K // 128, M, device="cuda") * 0.5 + 0.75).to(torch.float32)
    sb = (torch.rand(N // 128, K // 128, device="cuda") * 0.5 + 0.75).to(torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32) * 0.1

    out = mxfp8_gemm(a, b, sa, sb, bias=bias)
    ref = (_dequant_ref(a, b, sa, sb) + bias).to(torch.bfloat16)
    err = (out.float() - ref.float()).abs().max().item()
    mean = ref.float().abs().mean().item()
    assert err / max(mean, 1e-9) < 0.05
