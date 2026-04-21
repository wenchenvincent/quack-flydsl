# Copyright (c) 2026, AMD.

"""LDS bitonic-sort top-k kernel tests (N 128..4096)."""

import pytest
import torch

from quack.amd.topk_lds import topk_lds


@pytest.mark.parametrize("M", [128, 17, 4, 1])
@pytest.mark.parametrize("N", [128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("k", [1, 4, 16, 64, 128])
def test_topk_lds_matches_torch(M, N, k):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    if k > N:
        pytest.skip("k > N")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    v, i = topk_lds(x, k=k)
    ref_v, _ = torch.topk(x, k=k, dim=-1)
    torch.testing.assert_close(v, ref_v)
    # Gather by our indices should match the sorted values.
    gathered = torch.gather(x, -1, i.long())
    torch.testing.assert_close(gathered, ref_v)


def test_topk_lds_descending():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(4, 512, device="cuda", dtype=torch.float32)
    v, _ = topk_lds(x, k=64)
    assert (v[:, :-1] >= v[:, 1:]).all()
