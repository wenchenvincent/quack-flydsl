# Copyright (c) 2026, AMD.

"""Surface tests for quack.amd.gemm_blockscaled (torch-reference path).

Exercises the API shape + dequantise helpers. When the FlyDSL MFMA kernel
body lands, these tests double as the correctness harness for it (the
torch reference doesn't change).
"""

import pytest
import torch

from quack.amd.gemm_blockscaled import mxfp8_gemm


def _random_fp8(shape, device="cuda"):
    """Random tensor quantised to fp8_e4m3fn by a roundtrip through f32."""
    return torch.randn(shape, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)


def _random_scales(shape, device="cuda"):
    """Random positive per-block f32 scales in a sane range."""
    return (0.5 + torch.rand(shape, device=device, dtype=torch.float32))


@pytest.mark.parametrize("M", [128, 256])
@pytest.mark.parametrize("N", [128, 256])
@pytest.mark.parametrize("K", [128, 256])
def test_mxfp8_gemm_matches_manual_dequantise(M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch fp8 not available")
    torch.manual_seed(0)
    A = _random_fp8((M, K))
    B = _random_fp8((K, N))
    A_scale = _random_scales((K // 128, M))
    B_scale = _random_scales((N // 128, K // 128))

    C = mxfp8_gemm(A, B, A_scale, B_scale)
    assert C.dtype == torch.bfloat16
    assert C.shape == (M, N)

    # Compute an independent manual dequantise for cross-check.
    # A: (M, K). A_scale[kb, m] -> expand to (M, K).
    a_scale_m_k = A_scale.transpose(0, 1).repeat_interleave(128, dim=1)
    A_f32 = A.to(torch.float32) * a_scale_m_k
    # B: (K, N). B_scale[nb, kb] -> expand to (K, N).
    b_scale_k_n = B_scale.transpose(0, 1).repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    B_f32 = B.to(torch.float32) * b_scale_k_n
    ref = (A_f32 @ B_f32).to(torch.bfloat16)
    torch.testing.assert_close(C, ref, atol=0.05, rtol=0.05)


def test_mxfp8_gemm_f32_output():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch fp8 not available")
    torch.manual_seed(0)
    M, N, K = 128, 128, 128
    A = _random_fp8((M, K))
    B = _random_fp8((K, N))
    A_scale = _random_scales((K // 128, M))
    B_scale = _random_scales((N // 128, K // 128))
    C = mxfp8_gemm(A, B, A_scale, B_scale, out_dtype=torch.float32)
    assert C.dtype == torch.float32


def test_mxfp8_gemm_rejects_bad_shapes():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch fp8 not available")
    A = _random_fp8((129, 128))  # M=129 not multiple of 128 — expected to fail earlier
    B = _random_fp8((128, 128))
    A_scale = _random_scales((1, 129))
    B_scale = _random_scales((1, 1))
    # Shape check should fail: (K=128 is fine, but M=129 combined with
    # 128-multiple requirement on the K-dim indexing isn't explicitly
    # enforced at this level — the M asymmetry still matters for the
    # scale layout). Rely on the K/N % 128 check.
    A2 = _random_fp8((128, 127))  # K=127 breaks K % 128 == 0
    with pytest.raises(AssertionError):
        mxfp8_gemm(A2, B, A_scale, B_scale)


# ---------------------------------------------------------------------------
# Real FlyDSL MFMA blockscaled kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M", [256, 128])   # largest M first — see conftest.py
@pytest.mark.parametrize("N", [256, 128])   # multiple n-blocks supported
@pytest.mark.parametrize("K", [128, 256, 512])
def test_mxfp8_gemm_mfma_matches_torch_reference(M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch fp8 not available")
    from quack.amd.gemm_blockscaled_kernel import mxfp8_gemm_mfma
    torch.manual_seed(0)
    A = _random_fp8((M, K))
    B_std = _random_fp8((K, N))
    B = B_std.T.contiguous()  # kernel expects (N, K) layout
    A_scale = _random_scales((K // 128, M))
    B_scale = _random_scales((N // 128, K // 128))

    C = mxfp8_gemm_mfma(A, B, A_scale, B_scale)
    C_ref = mxfp8_gemm(A, B_std, A_scale, B_scale, out_dtype=torch.float32)
    # Rounding tolerance scales with K (accumulation length) for fp8 inputs.
    atol = max(2e-3, K * 3e-6)
    torch.testing.assert_close(C, C_ref, atol=atol, rtol=2e-3)


@pytest.mark.parametrize("K", [128, 256])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_mxfp8_gemm_uses_mfma_kernel_when_flagged(K, out_dtype):
    """use_mfma_kernel=True routes through the FlyDSL MFMA path via the
    top-level mxfp8_gemm. Output must match the torch reference for each
    supported output dtype."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch fp8 not available")
    torch.manual_seed(0)
    M, N = 128, 128
    A = _random_fp8((M, K))
    B = _random_fp8((K, N))
    A_scale = _random_scales((K // 128, M))
    B_scale = _random_scales((N // 128, K // 128))
    C_kernel = mxfp8_gemm(A, B, A_scale, B_scale, use_mfma_kernel=True, out_dtype=out_dtype)
    C_ref = mxfp8_gemm(A, B, A_scale, B_scale, use_mfma_kernel=False, out_dtype=out_dtype)
    assert C_kernel.dtype == out_dtype
    # Widen tolerance for half-precision outputs (the truncation happens
    # at the scalar store, accumulator is f32 in both paths).
    atol = 0.05 if out_dtype == torch.float32 else 0.1
    torch.testing.assert_close(C_kernel, C_ref, atol=atol, rtol=atol)


def test_mxfp8_gemm_mfma_all_ones():
    """All-1s fp8 × all-1s fp8 × all-1 scale → each output element = K (accumulation)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch fp8 not available")
    from quack.amd.gemm_blockscaled_kernel import mxfp8_gemm_mfma
    M, N, K = 128, 128, 128
    A = torch.ones(M, K, device="cuda").to(torch.float8_e4m3fn)
    B = torch.ones(N, K, device="cuda").to(torch.float8_e4m3fn)  # (N, K) layout
    sa = torch.ones(1, M, device="cuda", dtype=torch.float32)
    sb = torch.ones(1, 1, device="cuda", dtype=torch.float32)
    C = mxfp8_gemm_mfma(A, B, sa, sb)
    assert torch.all(C == float(K)), f"expected all {K}, got unique {torch.unique(C)}"
