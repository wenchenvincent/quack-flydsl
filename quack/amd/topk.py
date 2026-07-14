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
_ALL_KERNEL_N = _SINGLE_WAVE_N + _LDS_N
_MAX_KERNEL_N = _ALL_KERNEL_N[-1]  # 4096
_POW2_K = (1, 2, 4, 8, 16, 32, 64, 128)


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _next_kernel_n(n: int) -> int:
    """Smallest kernel-supported N ≥ ``n`` (or 0 if too large)."""
    for cand in _ALL_KERNEL_N:
        if cand >= n:
            return cand
    return 0


def _kernel_eligible(x: Tensor, k: int) -> bool:
    """True iff the exact-N kernel path accepts ``(x, k)`` with no padding."""
    if not (x.is_cuda and x.dim() == 2 and x.dtype == torch.float32):
        return False
    if x.stride(-1) != 1:
        return False
    N = x.size(-1)
    if k not in _POW2_K or k > N:
        return False
    return N in _ALL_KERNEL_N


def _pad_kernel_eligible(x: Tensor, k: int) -> bool:
    """True iff we can pad ``x`` with -inf up to a kernel-supported N and
    run the existing kernel. Requires the padded N to be ≤ 4096."""
    if not (x.is_cuda and x.dim() == 2 and x.dtype == torch.float32):
        return False
    if x.stride(-1) != 1:
        return False
    N = x.size(-1)
    if k not in _POW2_K or k > N:
        return False
    padded = _next_kernel_n(N)
    return padded != 0 and padded != N  # "padded != N" → there's actually padding to do


def topk_fwd(
    x: Tensor, k: int, softmax: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return ``(values, indices, softmax_values_or_None)`` — top-k per row.

    Values are in descending order along the last dim. Indices are
    into the input's last dim. If ``softmax`` is True, the third
    return is ``softmax(values)`` with the input dtype.

    Dispatch order:
      1. Exact-N kernel (N ∈ {8,16,32,64, 128,256,512,1024,2048,4096},
         k ∈ {1,2,4,8,16,32,64,128}, k ≤ N, f32, last-dim contiguous).
      2. Padded kernel — non-power-of-2 N ≤ 4096: pad to next kernel N
         with -inf, run, slice first k. ``-inf`` guarantees the padded
         slots never beat a real value, so returned indices stay < N.
      3. ``torch.topk`` fallback — N > 4096 or non-f32 or non-contiguous.
    """
    assert x.is_cuda and x.dim() == 2
    N_orig = x.size(-1)
    if _kernel_eligible(x, k):
        if N_orig in _SINGLE_WAVE_N:
            values, indices = _topk_mfma_single_wave(x, k=k)
        else:
            values, indices = _topk_lds(x, k=k)
    elif _pad_kernel_eligible(x, k):
        padded_n = _next_kernel_n(N_orig)
        M = x.size(0)
        pad_width = padded_n - N_orig
        # ``-inf`` sentinel — comparing ≥ always False against any finite
        # value, so the pad columns never win top-k. NaN inputs would
        # break this, but torch.topk has the same caveat.
        padding = torch.full(
            (M, pad_width), float("-inf"),
            device=x.device, dtype=x.dtype,
        )
        x_padded = torch.cat([x, padding], dim=-1).contiguous()
        if padded_n in _SINGLE_WAVE_N:
            values, indices = _topk_mfma_single_wave(x_padded, k=k)
        else:
            values, indices = _topk_lds(x_padded, k=k)
        # Indices are into x_padded; since pad columns carry -inf they
        # cannot appear in top-k results (k ≤ N_orig is asserted above),
        # so indices remain valid in the original-N space.
    else:
        values, indices = torch.topk(x, k, dim=-1)
        indices = indices.to(torch.int32)
    if softmax:
        sm = torch.softmax(values.float(), dim=-1).to(values.dtype)
        return values, indices, sm
    return values, indices, None


def topk_bwd(dvalues, values, indices, N, softmax=False):
    """Scatter ``dvalues`` back into a full ``(M, N)`` gradient at ``indices``.

    When ``softmax`` is True, ``values`` are the softmax outputs and the
    softmax Jacobian ``y*(g - sum(y*g))`` is applied before scattering.
    """
    if softmax:
        y = values
        dvalues = y * (dvalues - (dvalues * y).sum(-1, keepdim=True))
    M = indices.shape[0]
    dx = torch.zeros(M, N, dtype=dvalues.dtype, device=dvalues.device)
    dx.scatter_(1, indices.long(), dvalues)
    return dx


class TopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, softmax):
        values, indices, sm = topk_fwd(x, k, softmax=softmax)
        out_values = sm if softmax else values
        ctx.save_for_backward(out_values if softmax else None, indices)
        ctx.N = x.shape[-1]
        ctx.softmax = softmax
        ctx.mark_non_differentiable(indices)
        ctx.set_materialize_grads(False)
        return out_values, indices

    @staticmethod
    def backward(ctx, dvalues, dindices=None):
        saved_values, indices = ctx.saved_tensors
        if dvalues is None:
            return None, None, None
        dx = topk_bwd(dvalues, saved_values, indices, ctx.N, softmax=ctx.softmax)
        return dx, None, None


def topk(x, k, softmax=False):
    """Autograd-enabled top-k over the last dim. Returns ``(values, indices)``."""
    return TopKFunction.apply(x, k, softmax)


__all__ = ["topk_fwd", "topk_bwd", "topk", "TopKFunction"]
