# Copyright (c) 2026, AMD.

"""Single-wave bitonic-sort top-k kernel tests."""

import pytest
import torch

from quack.amd.topk_kernel import topk_mfma


# M-descending to dodge the JIT's grid-arg bake-in (see memory).
@pytest.mark.parametrize("M", [128, 17, 4, 1])
@pytest.mark.parametrize("N", [8, 16, 32, 64])
@pytest.mark.parametrize("k", [1, 2, 4, 8, 16, 32, 64])
def test_topk_matches_torch(M, N, k):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if k > N:
        pytest.skip("k > N")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    v, i = topk_mfma(x, k=k)
    ref_v, ref_i = torch.topk(x, k=k, dim=-1)
    torch.testing.assert_close(v, ref_v)
    # Tie-breaking may differ from torch.topk — verify indices by gathering.
    gathered = torch.gather(x, -1, i.long())
    torch.testing.assert_close(gathered, ref_v)


def test_topk_descending_order():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(4, 32, device="cuda", dtype=torch.float32)
    v, _ = topk_mfma(x, k=8)
    # Values must be non-increasing along k.
    assert (v[:, :-1] >= v[:, 1:]).all()
