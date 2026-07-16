# Copyright (c) 2026, AMD.

"""Per-row bitonic sort / argsort on gfx950 — reusable primitive.

The LDS top-k kernel (``topk_lds.py``) already runs a *full* O(N log²N)
ascending bitonic sort of every row in shared memory and then writes only the
top ``k``. A full sort is therefore the same kernel with ``k = N`` (write all N
sorted elements descending), exposed here as a first-class primitive so the
bitonic sort is reusable rather than buried inside top-k.

For ``N`` outside the supported power-of-2 set (128..4096) this falls back to
``torch.sort``. Note the bitonic sort is **not stable** — equal keys may take
either order — so returned indices are *a* valid argsort, not necessarily the
stable one (values are exact).

Public API:
    sort(x, dim=-1, descending=True) -> (values, indices)
    argsort(x, dim=-1, descending=True) -> indices
"""

from typing import Tuple

import torch
from torch import Tensor

from quack.amd.topk_lds import _topk_lds_f32_out

_SUPPORTED_N = (128, 256, 512, 1024, 2048, 4096)


def _bitonic_eligible(x: Tensor, dim: int) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.float32
        and x.dim() == 2
        and dim in (-1, 1)
        and x.stride(-1) == 1
        and x.shape[-1] in _SUPPORTED_N
    )


def sort(x: Tensor, dim: int = -1, descending: bool = True) -> Tuple[Tensor, Tensor]:
    """Per-row sort of a 2-D f32 tensor along the last dim.

    Uses the FlyDSL LDS bitonic-sort kernel when ``x`` is 2-D f32, last-dim
    contiguous, and ``N`` is a power of two in ``[128, 4096]``; otherwise falls
    back to ``torch.sort``. Returns ``(values, indices)`` like ``torch.sort``
    (indices int32 on the fast path).
    """
    if not _bitonic_eligible(x, dim):
        return torch.sort(x, dim=dim, descending=descending)
    M, N = x.shape
    vals = torch.empty(M, N, device=x.device, dtype=torch.float32)
    idx = torch.empty(M, N, device=x.device, dtype=torch.int32)
    # k = N → the kernel writes all N elements in DESCENDING order.
    _topk_lds_f32_out(x.contiguous(), vals, idx)
    if not descending:
        vals = vals.flip(-1).contiguous()
        idx = idx.flip(-1).contiguous()
    return vals, idx


def argsort(x: Tensor, dim: int = -1, descending: bool = True) -> Tensor:
    """Per-row argsort — indices that sort ``x`` (see :func:`sort`)."""
    return sort(x, dim=dim, descending=descending)[1]


__all__ = ["sort", "argsort"]
