# Copyright (c) 2026, AMD.

"""Variable-length sequence utilities for packed-batch GEMM on AMDGPU.

QuACK's varlen_m pattern packs batched sequences of varying length into
a single ``(total_M, K)`` tensor, with ``cu_seqlens_m`` holding the
cumulative row offset for each sample (shape ``(B+1,)``, i32). For plain
GEMM (no per-sample bias / activation / masking), varlen reduces to a
regular ``(total_M, K) @ (K, N)`` matmul — the AMD port takes that
trivial path here.

Future work, mirroring the NVIDIA ``varlen_utils`` scope:
  - Per-sample bias lookup (bias is ``(B, N)`` rather than ``(N,)``).
  - Gather-A via ``A_idx`` (row-index remap for MoE / packed flash).
  - concat_layout outputs (QKV concatenation / MoE expert concat).

These are callable by setting ``cu_seqlens_m`` and optional ``A_idx`` on
``quack.amd.gemm.gemm``; the dispatcher picks the varlen path. Each
extension is a discrete commit on top of this surface.
"""

from typing import NamedTuple, Optional

import torch
from torch import Tensor


class VarlenArgs(NamedTuple):
    """Lightweight container for varlen inputs.

    ``cu_seqlens_m``: shape ``(B + 1,)``, int32 on device. ``[0, L1, L1+L2,
    ..., total_M]``.
    ``A_idx``: optional gather-A row index, shape ``(total_M,)``, int32.
    """
    cu_seqlens_m: Optional[Tensor] = None
    A_idx: Optional[Tensor] = None


def validate_varlen(cu_seqlens_m: Tensor, total_M: int) -> int:
    """Assert cu_seqlens_m is a well-formed cumulative-length tensor for
    a ``(total_M, K)`` packed batch. Returns the batch size ``B``."""
    assert cu_seqlens_m.is_cuda
    assert cu_seqlens_m.dtype in (torch.int32, torch.int64)
    assert cu_seqlens_m.dim() == 1
    assert cu_seqlens_m.size(0) >= 2
    B = cu_seqlens_m.size(0) - 1
    # Host-side sanity; invalid cu_seqlens would silently corrupt compute
    # paths that use them, so pay the 1-element sync.
    last = cu_seqlens_m[-1].item()
    assert last == total_M, (
        f"cu_seqlens_m[-1]={last} doesn't match total rows {total_M}"
    )
    first = cu_seqlens_m[0].item()
    assert first == 0, f"cu_seqlens_m[0]={first} must be 0"
    return B


def seqlens_from_cu(cu_seqlens_m: Tensor) -> Tensor:
    """Return per-sample sequence lengths from a cumulative-offset tensor."""
    return cu_seqlens_m[1:] - cu_seqlens_m[:-1]


__all__ = ["VarlenArgs", "validate_varlen", "seqlens_from_cu"]
