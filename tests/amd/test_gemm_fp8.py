"""Standard (unscaled) fp8-e4m3 MFMA GEMM on gfx950."""

import pytest
import torch

from quack.amd.gemm_gfx950_fp8 import gemm_fp8


def _fp8(x):
    return x.to(torch.float8_e4m3fn)


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N",
    [(16, 32, 16), (64, 128, 64), (128, 256, 128), (256, 64, 512)],
)
def test_gemm_fp8_matches_dequant(M, K, N, out_dtype):
    torch.manual_seed(0)
    a_f = torch.randn(M, K, device="cuda") * 0.3
    b_f = torch.randn(K, N, device="cuda") * 0.3
    a, b = _fp8(a_f), _fp8(b_f)
    c = gemm_fp8(a, b, out_dtype=out_dtype)
    # Reference: dequantize the SAME fp8 tensors and matmul in f32.
    ref = a.float() @ b.float()
    assert c.shape == (M, N)
    assert c.dtype == out_dtype
    rel = (c.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    assert rel < 0.05, f"rel err {rel} too high (out_dtype={out_dtype})"


def test_gemm_fp8_rejects_non_fp8():
    a = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        gemm_fp8(a, b)


def test_gemm_fp8_rejects_bad_shape():
    a = torch.randn(16, 48, device="cuda").to(torch.float8_e4m3fn)  # K=48 not %32
    b = torch.randn(48, 16, device="cuda").to(torch.float8_e4m3fn)
    with pytest.raises(AssertionError):
        gemm_fp8(a, b)
