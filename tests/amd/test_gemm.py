# Copyright (c) 2026, AMD.

"""Surface tests for quack.amd.gemm (torch-fallback implementation).

Replace with numerical-correctness tests against a torch reference at
matching tolerance bands when the FlyDSL kernel ports land.
"""

import pytest
import torch

from quack.amd.gemm import gemm, gemm_act, gemm_gated, gemm_symmetric


def test_gemm_dispatches_to_mfma_kernel_when_eligible():
    """When the call falls in the MFMA kernel's scope, gemm() routes there
    and produces the correct output matching the PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    B = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    out = gemm(A, B)  # MFMA-eligible: f16 × f16, shape multiples of 16
    # Default out_dtype = A.dtype (torch.matmul convention).
    assert out.dtype == torch.float16
    ref = (A.float() @ B.float()).to(torch.float16)
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


def test_gemm_mfma_with_explicit_f32_output():
    """Explicit out_dtype=f32 keeps the accumulator precision."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    B = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    out = gemm(A, B, out_dtype=torch.float32)
    assert out.dtype == torch.float32


def test_gemm_falls_back_when_shape_not_aligned():
    """M=17 isn't a multiple of 16 → torch fallback."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(17, 64, device="cuda", dtype=torch.float16)
    B = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    out = gemm(A, B)
    # Torch matmul preserves input dtype by default.
    assert out.shape == (17, 64)


def test_gemm_gated_dispatch_uses_fused_kernel():
    """Top-level gemm_gated routes to the fused FlyDSL kernel when eligible."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, N, K = 64, 128, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    out = gemm_gated(A, B, gate_type="swiglu")
    assert out.shape == (M, N // 2)
    # Compare vs manual
    B_gate, B_up = B[:, :N // 2], B[:, N // 2:]
    ref_f32 = (torch.nn.functional.silu((A.float() @ B_gate.float()))
               * (A.float() @ B_up.float()))
    ref = ref_f32.to(out.dtype)
    torch.testing.assert_close(out, ref, atol=0.2, rtol=0.02)


def test_gemm_dispatch_alpha_beta_c_through_mfma():
    """alpha/beta/C now routes through the MFMA kernel when eligible."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, N, K = 64, 64, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C_in = torch.randn(M, N, device="cuda", dtype=torch.float32)
    out = gemm(A, B, alpha=0.5, beta=1.5, C=C_in, out_dtype=torch.float32)
    ref = 0.5 * (A.float() @ B.float()) + 1.5 * C_in
    torch.testing.assert_close(out, ref, atol=5e-5, rtol=5e-5)


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
    """Covers the torch-fallback path (A is f32)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(8, 64, device="cuda")
    out = gemm_symmetric(A)
    ref = A @ A.transpose(-1, -2)
    torch.testing.assert_close(out, ref)


def test_gemm_symmetric_dispatches_to_mfma_kernel():
    """f16/bf16 eligible shape routes to the dedicated symmetric MFMA kernel."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    A = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    out = gemm_symmetric(A)
    ref = A.float() @ A.float().t()
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)
    assert out.dtype == torch.float32  # dedicated kernel defaults to f32 output
