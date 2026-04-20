# Copyright (c) 2026, AMD.

"""TopK — AMDGPU port of `quack/topk.py`.

Current state: **API surface only**, delegating to ``torch.topk`` on-device.
A proper bitonic-sort kernel port (matching QuACK's constraints of
N, k powers of 2 and k ≤ 128) is a Phase 1 follow-up. This stub exists so
callers can import ``quack.amd.topk`` with a stable signature, and so the
test harness has a pass to gate on.

Why not a kernel now:
    QuACK's topk kernel is ~600 lines of bitonic-sort machinery; porting it
    exercises primitives (subgroup-level compare+swap, `ds_bpermute`) not yet
    shared with other AMD kernels in this branch. Follow up: port after the
    GEMM family lands so we can reuse the shared sort helpers.
"""

from typing import Tuple

import torch
from torch import Tensor


def topk_fwd(
    x: Tensor, k: int, softmax: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return top-k values + indices per row.

    Returns ``(values, indices, softmax_values_or_None)`` matching QuACK's
    NVIDIA-side public shape.

    TODO: replace with a FlyDSL bitonic-sort kernel (see module docstring).
    """
    assert x.is_cuda and x.dim() == 2
    values, indices = torch.topk(x, k, dim=-1)
    if softmax:
        sm = torch.softmax(values.float(), dim=-1).to(values.dtype)
        return values, indices, sm
    return values, indices, None


__all__ = ["topk_fwd"]
