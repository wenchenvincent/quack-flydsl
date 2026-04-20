# Copyright (c) 2026, AMD.

"""Surface tests for quack.amd.gemm (torch-fallback implementation).

Replace with numerical-correctness tests against a torch reference at
matching tolerance bands when the FlyDSL kernel ports land.
"""

import pytest
import torch

from quack.amd.gemm import gemm, gemm_act, gemm_gated, gemm_symmetric


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M, N, K", [(64, 64, 64), (128, 256, 128)])
def test_gemm_matches_torch(dtype, M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    out = gemm(A, B)
    ref = A @ B
    torch.testing.assert_close(out, ref)


def test_gemm_with_bias_and_activation():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(32, 64, device="cuda")
    B = torch.randn(64, 128, device="cuda")
    bias = torch.randn(128, device="cuda")
    out = gemm_act(A, B, activation="relu", bias=bias)
    ref = torch.relu(A @ B + bias)
    torch.testing.assert_close(out, ref)


def test_gemm_gated_swiglu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(8, 64, device="cuda")
    B = torch.randn(64, 128, device="cuda")  # must be even last dim for chunk(2)
    out = gemm_gated(A, B, gate_type="swiglu")
    assert out.shape == (8, 64)


def test_gemm_symmetric():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(8, 64, device="cuda")
    out = gemm_symmetric(A)
    ref = A @ A.transpose(-1, -2)
    torch.testing.assert_close(out, ref)
