"""Standard int8 MFMA GEMM on gfx950 — exact i32 accumulate."""

import pytest
import torch

from quack.amd.gemm_gfx950_int8 import gemm_int8


@pytest.mark.parametrize(
    "M,K,N",
    [(16, 32, 16), (64, 128, 64), (128, 256, 128), (256, 64, 512)],
)
def test_gemm_int8_exact(M, K, N):
    torch.manual_seed(0)
    a = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (K, N), device="cuda", dtype=torch.int8)
    c = gemm_int8(a, b)
    # Exact integer reference via float (|partial sums| well under 2**24).
    ref = (a.float() @ b.float()).to(torch.int32)
    assert c.dtype == torch.int32 and c.shape == (M, N)
    assert torch.equal(c, ref), f"int8 GEMM not exact, max_diff={(c - ref).abs().max().item()}"


def test_gemm_int8_full_range():
    # full int8 range to exercise sign + magnitude
    torch.manual_seed(1)
    M, K, N = 64, 64, 64
    a = torch.randint(-128, 128, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 128, (K, N), device="cuda", dtype=torch.int8)
    c = gemm_int8(a, b)
    ref = (a.float() @ b.float()).to(torch.int32)
    assert torch.equal(c, ref)


def test_gemm_int8_rejects_non_int8():
    a = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        gemm_int8(a, b)
