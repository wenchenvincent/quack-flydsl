# Copyright (c) 2026, AMD.

"""Blockscaled MX-FP8 GEMM correctness tests.

Current implementation is a dequant + bf16 GEMM MVP (see module
docstring in gemm_gfx950_blockscaled.py). Kernel-native mfma_scale
path is a follow-up; these tests lock in the API contract + scale
semantics so a future kernel port doesn't silently regress.
"""

import pytest
import torch

from quack.amd.gemm_gfx950_blockscaled import (
    mxfp8_gemm, _dequantize_fp8_to_bf16,
)


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
    # Reference uses the same dequant (bit-exact match path).
    a_bf16 = _dequantize_fp8_to_bf16(a, scale_a, scale_transposed=True)
    b_bf16 = _dequantize_fp8_to_bf16(
        b, scale_b, scale_transposed=False, block_size_n=128,
    )
    ref = (a_bf16.float() @ b_bf16.float().T).to(torch.bfloat16)
    # bf16 GEMM tolerance with K up to 256.
    tol = max(5e-3, K * 5e-4)
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=1e-2)


def test_mxfp8_neutral_scales_match_plain_matmul():
    """scale=1 everywhere should reduce to plain fp8 matmul."""
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
