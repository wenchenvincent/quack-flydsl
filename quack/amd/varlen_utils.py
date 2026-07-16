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
    cu_seqlens_k: Optional[Tensor] = None


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


def validate_varlen_k(cu_seqlens_k: Tensor, total_K: int) -> int:
    """Assert cu_seqlens_k is a well-formed cumulative-length tensor for a
    varlen-K grouped GEMM over a shared ``(M, total_K) @ (total_K, N)``.

    Same shape contract as :func:`validate_varlen` but checked against the
    contraction axis (``A``'s last dim / ``B``'s first dim) instead of rows.
    Returns the group count ``L``.
    """
    assert cu_seqlens_k.is_cuda
    assert cu_seqlens_k.dtype in (torch.int32, torch.int64)
    assert cu_seqlens_k.dim() == 1
    assert cu_seqlens_k.size(0) >= 2
    L = cu_seqlens_k.size(0) - 1
    last = cu_seqlens_k[-1].item()
    assert last == total_K, (
        f"cu_seqlens_k[-1]={last} doesn't match total contraction {total_K}"
    )
    first = cu_seqlens_k[0].item()
    assert first == 0, f"cu_seqlens_k[0]={first} must be 0"
    return L


def seqlens_from_cu(cu_seqlens_m: Tensor) -> Tensor:
    """Return per-sample sequence lengths from a cumulative-offset tensor."""
    return cu_seqlens_m[1:] - cu_seqlens_m[:-1]


def row_to_sample_idx(cu_seqlens_m: Tensor) -> Tensor:
    """Expand cu_seqlens_m into a ``(total_M,)`` int32 tensor mapping each
    packed row to its sample index.

    Example::
        cu  = [0, 3, 5, 7]         # 3 samples, lengths [3, 2, 2]
        out = [0, 0, 0, 1, 1, 2, 2]

    Used by varlen gather paths that need to look up per-sample metadata
    (e.g., per-sample bias ``bias[sample_idx[m], :]``). Runs on-device via
    ``repeat_interleave``.
    """
    assert cu_seqlens_m.is_cuda
    seqlens = seqlens_from_cu(cu_seqlens_m).to(torch.int64)
    B = seqlens.size(0)
    sample_ids = torch.arange(B, device=cu_seqlens_m.device, dtype=torch.int32)
    return sample_ids.repeat_interleave(seqlens)


__all__ = [
    "VarlenArgs", "validate_varlen", "validate_varlen_k",
    "seqlens_from_cu", "row_to_sample_idx",
]
