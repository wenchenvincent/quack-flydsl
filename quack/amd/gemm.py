# Copyright (c) 2026, AMD.

"""GEMM family — AMDGPU port of `quack/gemm.py` and friends.

**Current state: API surface + torch fallback.** The full kernel ports —
preshuffle MFMA pipeline for CDNA3/CDNA4, WMMA for RDNA4, stream-K scheduler,
and the epilogue zoo (bias, act, dact, gated, dgated, norm_act, symmetric,
blockscaled) — are a substantial follow-up matching QuACK's CUTe-DSL scale.

What's shipped here:
    - ``gemm(A, B, bias=None, activation=None)`` — standard GEMM matching
      QuACK's NVIDIA API, delegating to ``torch.mm``/``torch.addmm`` +
      ``quack.amd.activation``.
    - Wrapper scaffolding for the variants (``gemm_act``, ``gemm_gated``, …)
      so downstream callers (linear, mlp) import the same names.

What's NOT yet shipped (tracked for Phase 2 follow-up):
    - FlyDSL MFMA/WMMA kernels (reference: ``FlyDSL/kernels/preshuffle_gemm.py``,
      ``FlyDSL/kernels/hgemm_splitk.py``, ``FlyDSL/kernels/rdna_f16_gemm.py``).
    - Stream-K tile scheduler (plan: ``quack/amd/tile_scheduler.py`` using
      rocdl atomics).
    - blockscaled fp8/fp4 (reference: ``FlyDSL/kernels/gemm_fp8fp4_gfx1250.py``,
      ``FlyDSL/kernels/moe_blockscale_2stage.py``).
    - Fused epilogues.
"""

from typing import Optional

import torch
from torch import Tensor


def gemm(
    A: Tensor,
    B: Tensor,
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """GEMM: ``D = alpha * A @ B + beta * C + bias`` then optional activation.

    Matches QuACK's NVIDIA ``gemm`` API. Current implementation is a torch
    fallback; kernel port is Phase 2 follow-up.
    """
    assert A.is_cuda and B.is_cuda
    out = alpha * (A @ B)
    if C is not None and beta != 0.0:
        out = out + beta * C
    if bias is not None:
        out = out + bias
    if activation is not None:
        if activation == "relu":
            out = torch.relu(out)
        elif activation == "gelu_tanh_approx":
            out = torch.nn.functional.gelu(out, approximate="tanh")
        elif activation == "silu":
            out = torch.nn.functional.silu(out)
        elif activation == "relu_sq":
            out = torch.relu(out) * out
        else:
            raise NotImplementedError(f"activation={activation!r}")
    if out_dtype is not None and out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out


def gemm_act(A, B, activation: str, bias=None, **kw):
    return gemm(A, B, bias=bias, activation=activation, **kw)


def gemm_gated(A, B, gate_type: str = "swiglu", **kw):
    """Gated GEMM: split the output along the last dim into ``(gate, up)``
    and apply the gating function. Matches QuACK's ``gemm_gated`` semantics.
    """
    out = gemm(A, B, **kw)
    gate, up = out.chunk(2, dim=-1)
    if gate_type == "swiglu":
        return torch.nn.functional.silu(gate) * up
    if gate_type == "reglu":
        return torch.relu(gate) * up
    if gate_type == "geglu":
        return torch.nn.functional.gelu(gate, approximate="tanh") * up
    if gate_type == "glu":
        return torch.sigmoid(gate) * up
    raise NotImplementedError(f"gate_type={gate_type!r}")


def gemm_symmetric(A, **kw):
    """Symmetric GEMM: C = A @ A^T."""
    return gemm(A, A.transpose(-1, -2), **kw)


__all__ = ["gemm", "gemm_act", "gemm_gated", "gemm_symmetric"]
