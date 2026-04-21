# Copyright (c) 2026, AMD.

"""Surface tests for quack.amd.topk (torch-fallback implementation).

When a kernel port lands, this file should grow numerical-correctness tests
against a torch reference with the same atol/rtol bands as the other kernels.
"""

import pytest
import torch

from quack.amd.topk import topk_fwd


@pytest.mark.parametrize("k", [2, 8, 32])
@pytest.mark.parametrize("M", [1, 4])
@pytest.mark.parametrize("N", [256, 1024])
def test_topk_fwd_matches_torch(k, M, N):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    vals, idx, sm = topk_fwd(x, k)
    vals_ref, _ = torch.topk(x, k, dim=-1)
    torch.testing.assert_close(vals, vals_ref)
    # Indices should pick out the same values (tie-breaking may differ).
    gathered = torch.gather(x, -1, idx.long())
    torch.testing.assert_close(gathered, vals_ref)
    assert sm is None


def test_topk_fwd_softmax():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(4, 128, device="cuda", dtype=torch.float32)
    vals, idx, sm = topk_fwd(x, 8, softmax=True)
    assert sm is not None
    expected_sm = torch.softmax(vals, dim=-1)
    torch.testing.assert_close(sm, expected_sm)
