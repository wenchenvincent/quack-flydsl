# Copyright (c) 2026, AMD.

"""Stochastic-rounding fp8 quantization (G8) — hardware v_cvt_sr_fp8_f32."""

import pytest
import torch

from quack.amd.rounding import quantize_fp8_sr


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_sr_unbiased(dtype):
    """Averaging SR-quantized x over many seeds is ~unbiased (E[SR(x)]≈x),
    and closer to x than round-to-nearest for values between grid points."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    # values in a range where fp8 spacing is coarse -> RTN is visibly biased
    x = torch.rand(4096, device="cuda") * 6.0 + 1.0  # (1, 7)
    n_seeds = 256
    acc = torch.zeros_like(x)
    for s in range(n_seeds):
        acc += quantize_fp8_sr(x, seed=s, dtype=dtype).float()
    mean_sr = acc / n_seeds
    bias_sr = (mean_sr - x).abs().mean().item()
    bias_rtn = (x.to(dtype).float() - x).abs().mean().item()
    print(f"\n{dtype}: SR mean-bias={bias_sr:.5f}  RTN bias={bias_rtn:.5f}")
    # SR averaged over seeds must be far less biased than a single RTN pass.
    assert bias_sr < bias_rtn * 0.25, f"SR not unbiased: {bias_sr} vs RTN {bias_rtn}"


def _fp8_grid():
    """All 256 float8_e4m3fn values as a sorted f32 tensor (finite only)."""
    codes = torch.arange(256, dtype=torch.uint8)
    vals = codes.view(torch.float8_e4m3fn).float()
    vals = vals[torch.isfinite(vals)]
    return torch.unique(vals).sort().values


def test_sr_brackets_value():
    """Every SR result is one of the two fp8 grid points bracketing x
    (checked against the exact enumerated fp8 grid, not an approximation)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(1)
    x = torch.rand(2048, device="cuda") * 10.0
    grid = _fp8_grid().to(x.device)
    # exact bracket: largest grid pt <= x (floor) and smallest >= x (ceil)
    idx = torch.searchsorted(grid, x)
    ceil = grid[idx.clamp(max=len(grid) - 1)]
    floor = grid[(idx - 1).clamp(min=0)]
    for s in range(8):
        q = quantize_fp8_sr(x, seed=s).float()
        # SR must land on floor or ceil — never outside the bracket.
        assert torch.all(q >= floor - 1e-4), (s, (q - floor).min().item())
        assert torch.all(q <= ceil + 1e-4), (s, (q - ceil).max().item())


def test_sr_deterministic():
    """Same seed -> identical output; different seed -> generally different."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(2)
    x = torch.rand(1000, device="cuda") * 4.0 + 0.5  # non-multiple of 256
    a = quantize_fp8_sr(x, seed=7)
    b = quantize_fp8_sr(x, seed=7)
    c = quantize_fp8_sr(x, seed=8)
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), "not deterministic"
    # different seeds should differ on at least some between-grid elements
    assert not torch.equal(a.view(torch.uint8), c.view(torch.uint8)), "seed ignored"


def test_sr_exact_grid_values_unchanged():
    """Values exactly on the fp8 grid must round to themselves (no dither)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    x = torch.tensor([0.0, 1.0, 2.0, 4.0, 0.5, 1.5, -2.0, -1.0] * 4, device="cuda")
    for s in range(4):
        q = quantize_fp8_sr(x, seed=s).float()
        torch.testing.assert_close(q, x, atol=0, rtol=0)
