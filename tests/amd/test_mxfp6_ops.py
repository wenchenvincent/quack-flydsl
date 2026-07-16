"""MXFP6 quantize/dequantize round-trip (OCP MX: e2m3 element + e8m0 scale)."""

import pytest
import torch

from quack.amd.mxfp6_ops import quantize_mxfp6, dequantize_mxfp6, MX_BLOCK


@pytest.mark.parametrize("shape", [(16, 128), (8, 256), (4, 32), (32, 64)])
def test_roundtrip_shapes(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda") * 2.0
    packed, scale, orig = quantize_mxfp6(x, axis=-1)
    assert packed.dtype == torch.uint8 and scale.dtype == torch.uint8
    K = shape[-1]
    assert packed.shape == (*shape[:-1], K * 6 // 8)
    assert scale.shape == (*shape[:-1], K // MX_BLOCK)
    xq = dequantize_mxfp6(packed, scale, orig)
    assert xq.shape == shape
    # e2m3 has 3 mantissa bits -> mean rel error well under MXFP4's ~0.11.
    rel = (xq - x).abs().mean() / x.abs().mean()
    assert rel < 0.08, f"round-trip mean rel err {rel.item()}"


def test_exact_representable_values():
    # e2m3 grid points * power-of-2 scale round-trip exactly.
    x = torch.tensor([[1.0, 2.0, 4.0, 6.0, -1.0, -3.0, 0.5, 1.5] * 4], device="cuda")
    packed, scale, orig = quantize_mxfp6(x, axis=-1)
    xq = dequantize_mxfp6(packed, scale, orig)
    torch.testing.assert_close(xq, x, rtol=0, atol=0)


def test_more_precise_than_mxfp4():
    from quack.amd.mxfp4_ops import quantize_mxfp4, dequantize_mxfp4
    torch.manual_seed(3)
    x = torch.randn(8, 256, device="cuda")
    p6, s6, sh6 = quantize_mxfp6(x)
    p4, s4, sh4 = quantize_mxfp4(x)
    err6 = (dequantize_mxfp6(p6, s6, sh6) - x).abs().mean()
    err4 = (dequantize_mxfp4(p4, s4, sh4) - x).abs().mean()
    assert err6 < err4, f"MXFP6 ({err6}) should beat MXFP4 ({err4})"


def test_requires_block_multiple():
    x = torch.randn(4, 48, device="cuda")  # 48 not a multiple of 32
    with pytest.raises(AssertionError):
        quantize_mxfp6(x, axis=-1)
