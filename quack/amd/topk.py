# Copyright (c) 2026, AMD.

"""TopK — AMDGPU port of `quack/topk.py`.

Dispatches to FlyDSL bitonic-sort kernels when shape constraints are met:

  - N ∈ {8, 16, 32, 64}  →  ``topk_mfma``  (single-wave shuffle_xor sort)
  - N ∈ {128, 256, 512, 1024, 2048, 4096}  →  ``topk_lds``  (LDS sort)
  - Otherwise (non-power-of-2 N, N > 4096, etc.)  →  ``torch.topk``

The FlyDSL path requires f32 input, last-dim contiguous, k ≤ 128, both
N and k powers of 2.
"""

from typing import Tuple

import torch
from torch import Tensor

from quack.amd.topk_kernel import topk_mfma as _topk_mfma_single_wave
from quack.amd.topk_lds import topk_lds as _topk_lds


_SINGLE_WAVE_N = (8, 16, 32, 64)
_LDS_N = (128, 256, 512, 1024, 2048, 4096)
_POW2_K = (1, 2, 4, 8, 16, 32, 64, 128)


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _kernel_eligible(x: Tensor, k: int) -> bool:
    if not (x.is_cuda and x.dim() == 2 and x.dtype == torch.float32):
        return False
    if x.stride(-1) != 1:
        return False
    N = x.size(-1)
    if k not in _POW2_K or k > N:
        return False
    return N in _SINGLE_WAVE_N or N in _LDS_N


def topk_fwd(
    x: Tensor, k: int, softmax: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return ``(values, indices, softmax_values_or_None)`` — top-k per row.

    Values are in descending order along the last dim. Indices are
    into the input's last dim. If ``softmax`` is True, the third
    return is ``softmax(values)`` with the input dtype.

    Uses a FlyDSL bitonic kernel when eligible (f32, last-dim contiguous,
    N and k powers of 2 in the supported range). Falls back to
    ``torch.topk`` otherwise — callers that hit the fallback get the
    same numeric answer, just via a different kernel.
    """
    assert x.is_cuda and x.dim() == 2
    if _kernel_eligible(x, k):
        N = x.size(-1)
        if N in _SINGLE_WAVE_N:
            values, indices = _topk_mfma_single_wave(x, k=k)
        else:
            values, indices = _topk_lds(x, k=k)
    else:
        values, indices = torch.topk(x, k, dim=-1)
        indices = indices.to(torch.int32)
    if softmax:
        sm = torch.softmax(values.float(), dim=-1).to(values.dtype)
        return values, indices, sm
    return values, indices, None


__all__ = ["topk_fwd"]
