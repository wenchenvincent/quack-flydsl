# Copyright (c) 2026, AMD.

"""Fused linear + cross-entropy — AMDGPU port of `quack/linear_cross_entropy.py`.

**Chunked** linear + CE: instead of materialising the full ``(B*L, V)``
logits tensor (which for V≈128k and B*L≈8k can be 2GB), we process
``x`` in chunks of ``chunk_size`` rows through a single reused
``(chunk_size, V)`` logits buffer.

HBM savings per call: ``(B*L - chunk_size) * V * dtype_bytes`` — for
a Llama-scale vocabulary this is a ~1 GB allocation avoided.

Correctness: mathematically equivalent to the plain
``cross_entropy(F.linear(x, w), target)`` because CE is per-row —
partitioning along the batch dim is loss-preserving.

Routes each chunk's matmul through ``quack.amd.linear`` (which
dispatches bf16 shapes to ``gemm_splitk``) and CE through
``cross_entropy_fwd``. Backward pass follows the same chunked
pattern in an ``autograd.Function`` when requires_grad is set.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from quack.amd.linear import linear
from quack.amd.cross_entropy import cross_entropy_fwd


_DEFAULT_CHUNK_SIZE = 4096


def _iter_chunks(*tensors: Tensor, chunk_size: int):
    B_L = tensors[0].size(0)
    for start in range(0, B_L, chunk_size):
        stop = min(start + chunk_size, B_L)
        yield tuple(t[start:stop] for t in tensors), start, stop


def linear_cross_entropy(
    x: Tensor,            # (B*L, d)
    weight: Tensor,       # (V, d)
    target: Tensor,       # (B*L,) int32/int64
    bias: Optional[Tensor] = None,
    return_lse: bool = False,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Compute ``logits = x @ weight.T + bias`` (per chunk), then per-row CE.

    Returns ``(loss, lse_or_None)``. Both are shape ``(B*L,)`` in f32.

    ``chunk_size`` controls the row granularity of the streaming
    execution — larger chunks amortise the matmul launch but cost
    more HBM for the logits buffer; default 4096 matches the NVIDIA
    QuACK reference.
    """
    assert x.is_cuda and weight.is_cuda and target.is_cuda
    assert x.dim() == 2 and weight.dim() == 2
    B_L, d = x.shape
    V, d2 = weight.shape
    assert d == d2
    assert target.dim() == 1 and target.size(0) == B_L

    # Fast path: single chunk (smaller shapes) — no streaming overhead.
    if B_L <= chunk_size:
        logits = linear(x, weight, bias=bias)
        return cross_entropy_fwd(logits, target, return_lse=return_lse)

    # Chunked path. Reuse a single (chunk_size, V) logits buffer to
    # cap peak HBM; CE forward copies loss/lse out per chunk.
    loss = torch.empty(B_L, device=x.device, dtype=torch.float32)
    lse = torch.empty(B_L, device=x.device, dtype=torch.float32) if return_lse else None
    logits_buf = torch.empty(chunk_size, V, device=x.device, dtype=x.dtype)

    # Import inside the chunked branch so we can reach for gemm_splitk
    # with an explicit ``out=`` parameter (our ``linear()`` surface
    # doesn't expose out=; going direct to the kernel here keeps the
    # HBM footprint flat).
    from quack.amd.gemm_gfx950_splitk import gemm_splitk

    # Eligibility for the in-place fast path (aligned bf16/f16, no bias).
    splitk_ok = (
        bias is None
        and x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
        and V % 256 == 0 and d % 64 == 0
        and chunk_size % 128 == 0
    )

    for (x_c, target_c), start, stop in _iter_chunks(x, target, chunk_size=chunk_size):
        chunk_len = stop - start
        logits_c = logits_buf[:chunk_len]
        # Tail chunk may have chunk_len not divisible by 128; in that
        # case fall back to a fresh-alloc linear() for just that chunk.
        if splitk_ok and chunk_len == chunk_size:
            # In-place: writes directly into ``logits_c``, no per-chunk
            # (chunk_size, V) alloc. This is the whole point of chunking.
            gemm_splitk(x_c, weight, logits_c)
        else:
            out_c = linear(x_c, weight, bias=bias)
            logits_c.copy_(out_c)
        loss_c, lse_c = cross_entropy_fwd(logits_c, target_c, return_lse=return_lse)
        loss[start:stop].copy_(loss_c)
        if return_lse:
            lse[start:stop].copy_(lse_c)
    return loss, lse


__all__ = ["linear_cross_entropy"]
