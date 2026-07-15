"""MXFP4 (e2m1 + e8m0) scaled MFMA GEMM on gfx950 — ``C = A @ B.T``.

The kernel quantizes A/B to MXFP4 internally, so the reference dequantizes the
*same* quantized tensors and matmuls — the hardware-scaled MFMA must reproduce
the dequant-then-matmul result exactly (the only rounding is the shared
quantization step, cancelled on both sides).
"""

import pytest
import torch

from quack.amd.gemm_gfx950_mxfp4 import gemm_mxfp4
from quack.amd.mxfp4_ops import quantize_mxfp4, dequantize_mxfp4


def _dequant_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ap, asc, ash = quantize_mxfp4(a, axis=-1)
    bp, bsc, bsh = quantize_mxfp4(b, axis=-1)
    return dequantize_mxfp4(ap, asc, ash) @ dequantize_mxfp4(bp, bsc, bsh).t()


@pytest.mark.parametrize(
    "M,K,N",
    [(16, 128, 16), (64, 256, 48), (128, 512, 96), (32, 128, 256), (16, 384, 16)],
)
def test_gemm_mxfp4_matches_dequant(M, K, N):
    torch.manual_seed(M + K + N)
    a = torch.randn(M, K, device="cuda") * 1.5
    b = torch.randn(N, K, device="cuda") * 1.5
    c = gemm_mxfp4(a, b)
    ref = _dequant_ref(a, b)
    assert c.shape == (M, N) and c.dtype == torch.float32
    err = (c - ref).abs().max().item()
    rel = err / (ref.abs().max().item() + 1e-6)
    assert rel < 1e-4, f"MXFP4 GEMM {M}x{K}x{N} rel err {rel} (max {err})"


def test_gemm_mxfp4_close_to_true_matmul():
    # Against the true f32 matmul, error is bounded by MXFP4 quantization only.
    torch.manual_seed(7)
    M, K, N = 64, 256, 64
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(N, K, device="cuda")
    c = gemm_mxfp4(a, b)
    ref = a @ b.t()
    rel = (c - ref).norm() / ref.norm()
    # MXFP4 carries ~1 mantissa bit, so ~15-20% error vs full-precision is
    # the format floor here — the exactness-vs-dequant tests pin the kernel.
    assert rel < 0.25, f"MXFP4 vs true matmul rel err {rel.item()}"


def test_gemm_mxfp4_rejects_bad_shapes():
    a = torch.randn(15, 128, device="cuda")  # M % 16 != 0
    b = torch.randn(16, 128, device="cuda")
    with pytest.raises(AssertionError):
        gemm_mxfp4(a, b)
    a = torch.randn(16, 100, device="cuda")  # K % 128 != 0
    b = torch.randn(16, 100, device="cuda")
    with pytest.raises(AssertionError):
        gemm_mxfp4(a, b)
