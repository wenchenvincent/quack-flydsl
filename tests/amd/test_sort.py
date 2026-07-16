# Copyright (c) 2026, AMD.

"""Per-row bitonic sort / argsort (G14) vs torch.sort."""

import pytest
import torch

from quack.amd.sort import sort, argsort


@pytest.mark.parametrize("N", [4096, 1024, 256, 128])
@pytest.mark.parametrize("descending", [True, False])
def test_sort_values_match_torch(N, descending):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(N)
    x = torch.randn(64, N, device="cuda", dtype=torch.float32)
    vals, idx = sort(x, descending=descending)
    ref_vals, _ = torch.sort(x, dim=-1, descending=descending)
    # sorted values must match exactly (bitonic is exact; only tie-order differs)
    torch.testing.assert_close(vals, ref_vals, atol=0, rtol=0)


@pytest.mark.parametrize("N", [512, 2048])
def test_argsort_is_valid_permutation(N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(N + 1)
    x = torch.randn(32, N, device="cuda", dtype=torch.float32)
    vals, idx = sort(x, descending=True)
    # gathering x by idx must reproduce the returned sorted values
    gathered = torch.gather(x, -1, idx.long())
    torch.testing.assert_close(gathered, vals, atol=0, rtol=0)
    # each row's indices are a permutation of 0..N-1
    for r in range(x.shape[0]):
        assert torch.equal(idx[r].long().sort().values, torch.arange(N, device=x.device))


def test_argsort_matches_torch_ordering_no_ties():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    # distinct values -> unique argsort, must match torch exactly
    x = torch.randperm(256, device="cuda").float().unsqueeze(0)
    idx = argsort(x, descending=False)
    ref = torch.argsort(x, dim=-1, descending=False).int()
    assert torch.equal(idx, ref)


def test_sort_fallback_unsupported_shape():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    x = torch.randn(4, 300, device="cuda")  # N not power-of-2 -> torch fallback
    vals, idx = sort(x, descending=True)
    ref_vals, ref_idx = torch.sort(x, dim=-1, descending=True)
    torch.testing.assert_close(vals, ref_vals)
