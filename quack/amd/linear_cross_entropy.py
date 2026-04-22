# Copyright (c) 2026, AMD.

"""Fused linear + cross-entropy — AMDGPU port of `quack/linear_cross_entropy.py`.

**Chunked + fused-backward** linear + CE. Two wins:

1. **Chunked logits buffer.** Instead of materialising the full
   ``(B*L, V)`` logits tensor, we process ``x`` in chunks through a
   reused ``(chunk_size, V)`` buffer. Saves ``(B*L - chunk_size) * V *
   dtype_bytes`` of HBM — for a Llama-scale vocab, ~1 GB+.

2. **Fused fwd+bwd CE with inplace dlogits.** Per chunk, the CE kernel
   writes ``softmax(logits) - one_hot(target)`` **into the same
   logits buffer** (aliasing — the buffer transitions from
   ``logits`` to ``dlogits`` in one pass). No separate dlogits alloc;
   one kernel launch for both forward loss and backward dx.

Correctness: mathematically equivalent to the plain
``cross_entropy(F.linear(x, w), target)``. CE is per-row, so chunking
along the batch dim is loss-preserving.

Backward (via ``linear_cross_entropy_bwd``): chunked matmul pattern
matching the NVIDIA QuACK reference:
  - ``dx[chunk] = dlogits[chunk] @ weight``
  - ``dw = dlogits.T @ x`` (accumulated across chunks except the last)
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from quack.amd.linear import linear
from quack.amd.cross_entropy import cross_entropy_fwd, cross_entropy_fwd_bwd


_DEFAULT_CHUNK_SIZE = 4096


def _iter_chunks(*tensors: Tensor, chunk_size: int):
    B_L = tensors[0].size(0)
    for start in range(0, B_L, chunk_size):
        stop = min(start + chunk_size, B_L)
        yield tuple(t[start:stop] for t in tensors), start, stop


def _splitk_eligible_for_chunking(x, weight, bias, chunk_size):
    """Can ``gemm_splitk(x_c, weight, logits_c)`` with ``out=`` be used?"""
    if bias is not None:
        return False
    V = weight.size(0)
    d = weight.size(1)
    return (
        x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
        and V % 256 == 0 and d % 64 == 0
        and chunk_size % 128 == 0
    )


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

    This is the **forward-only** entry point. For a full training
    forward/backward call ``linear_cross_entropy_fwd_bwd``, which
    returns ``(loss, dx, dw)`` in the same chunked pass (no extra
    full-logits allocation).
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

    loss = torch.empty(B_L, device=x.device, dtype=torch.float32)
    lse = torch.empty(B_L, device=x.device, dtype=torch.float32) if return_lse else None
    logits_buf = torch.empty(chunk_size, V, device=x.device, dtype=x.dtype)

    from quack.amd.gemm_gfx950_splitk import gemm_splitk
    splitk_ok = _splitk_eligible_for_chunking(x, weight, bias, chunk_size)

    for (x_c, target_c), start, stop in _iter_chunks(x, target, chunk_size=chunk_size):
        chunk_len = stop - start
        logits_c = logits_buf[:chunk_len]
        if splitk_ok and chunk_len == chunk_size:
            gemm_splitk(x_c, weight, logits_c)
        else:
            out_c = linear(x_c, weight, bias=bias)
            logits_c.copy_(out_c)
        loss_c, lse_c = cross_entropy_fwd(logits_c, target_c, return_lse=return_lse)
        loss[start:stop].copy_(loss_c)
        if return_lse:
            lse[start:stop].copy_(lse_c)
    return loss, lse


def linear_cross_entropy_fwd_bwd(
    x: Tensor,            # (B*L, d)
    weight: Tensor,       # (V, d)
    target: Tensor,       # (B*L,)
    bias: Optional[Tensor] = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Chunked fused forward + backward: loss, dx, dw in one pass.

    Returns ``(loss, dx, dw)``:
      - ``loss``: (B*L,) f32 — per-row CE loss (dloss=1 convention).
      - ``dx``:   (B*L, d) same dtype as ``x`` — gradient wrt input.
      - ``dw``:   (V, d) f32 — gradient wrt weight (accumulated across chunks).

    HBM footprint per call (vs unchunked fwd+bwd):
      unchunked: 2*(B*L, V) logits+dlogits + dw buffer + dx
      chunked:   (chunk_size, V) reused as logits→dlogits + dw + dx

    Savings: ``(2*B*L - chunk_size) * V * dtype_bytes``. The 2× comes
    from the inplace-backward trick — the fused CE kernel writes dx
    into the logits buffer so dlogits never needs its own allocation.

    Algorithm (matches QuACK NVIDIA reference):
      for each chunk:
        1. gemm_splitk(x_c, weight, logits_c)       # out= into reused buffer
        2. cross_entropy_fwd_bwd(logits_c, target_c,
                                 dx=logits_c)       # inplace: logits_c → dlogits_c
        3. torch.mm(dlogits_c, weight, out=dx[c])   # dx contribution
        4. dw += dlogits_c.T @ x_c                  # dw accumulation
    """
    assert x.is_cuda and weight.is_cuda and target.is_cuda
    assert x.dim() == 2 and weight.dim() == 2
    B_L, d = x.shape
    V, d2 = weight.shape
    assert d == d2
    assert target.dim() == 1 and target.size(0) == B_L
    assert bias is None, "bias not yet supported in fwd+bwd chunked path"

    loss = torch.empty(B_L, device=x.device, dtype=torch.float32)
    dx = torch.empty_like(x)
    dw = torch.zeros(V, d, device=x.device, dtype=torch.float32)

    # Single-chunk fast path.
    if B_L <= chunk_size:
        logits = linear(x, weight)                                # (B_L, V)
        loss_out, _, dlogits = cross_entropy_fwd_bwd(
            logits, target, dx=logits, return_lse=False,
        )                                                          # inplace: logits → dlogits
        loss.copy_(loss_out)
        torch.mm(dlogits, weight, out=dx)                          # dx = dlogits @ W
        dw.copy_(dlogits.t().float() @ x.float())                  # dw = dlogits^T @ x
        return loss, dx, dw

    # Chunked path: use ``torch.nn.functional.linear`` (hipBLASLt under
    # the hood on ROCm) for the matmul. Our ``quack.amd.linear`` routes
    # to ``gemm_splitk`` for eligible shapes, but ``gemm_splitk`` has a
    # FlyDSL grid-dim specialisation quirk where varying M across calls
    # in the same process can mis-compile. hipBLASLt has no such quirk
    # and is already at ~parity on this shape class.
    for (x_c, target_c, dx_c), start, stop in _iter_chunks(
        x, target, dx, chunk_size=chunk_size,
    ):
        logits_c = torch.nn.functional.linear(x_c, weight, bias)
        loss_c, _, dlogits_c = cross_entropy_fwd_bwd(
            logits_c, target_c, dx=logits_c, return_lse=False,
        )
        loss[start:stop].copy_(loss_c)
        # dlogits_c IS logits_c (aliased).
        torch.mm(dlogits_c, weight, out=dx_c)
        dw.add_(dlogits_c.t().float() @ x_c.float())
    return loss, dx, dw


__all__ = ["linear_cross_entropy", "linear_cross_entropy_fwd_bwd"]
