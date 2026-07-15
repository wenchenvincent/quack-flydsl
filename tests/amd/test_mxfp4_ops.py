"""MXFP4 quantize/dequantize round-trip (OCP MX: e2m1 element + e8m0 scale)."""

import pytest
import torch

from quack.amd.mxfp4_ops import quantize_mxfp4, dequantize_mxfp4, MX_BLOCK


@pytest.mark.parametrize("shape", [(16, 128), (8, 256), (4, 32), (32, 64)])
def test_roundtrip_shapes(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda") * 2.0
    packed, scale, orig = quantize_mxfp4(x, axis=-1)
    assert packed.dtype == torch.uint8 and scale.dtype == torch.uint8
    K = shape[-1]
    assert packed.shape == (*shape[:-1], K // 2)
    assert scale.shape == (*shape[:-1], K // MX_BLOCK)
    xq = dequantize_mxfp4(packed, scale, orig)
    assert xq.shape == shape
    # MXFP4 has ~1 mantissa bit; mean relative error should be well under 0.25.
    rel = (xq - x).abs().mean() / x.abs().mean()
    assert rel < 0.25, f"round-trip mean rel err {rel.item()}"


def test_exact_representable_values():
    # Values that land exactly on e2m1 levels * a power-of-2 scale round-trip exactly.
    x = torch.tensor([[1.0, 2.0, 4.0, 6.0, -1.0, -3.0, 0.5, 0.0] * 4], device="cuda")
    packed, scale, orig = quantize_mxfp4(x, axis=-1)
    xq = dequantize_mxfp4(packed, scale, orig)
    torch.testing.assert_close(xq, x, rtol=0, atol=0)


def test_requires_block_multiple():
    x = torch.randn(4, 48, device="cuda")  # 48 not a multiple of 32
    with pytest.raises(AssertionError):
        quantize_mxfp4(x, axis=-1)
