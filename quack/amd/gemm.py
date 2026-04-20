# Copyright (c) 2026, AMD.

"""GEMM family — AMDGPU port of `quack/gemm.py` and friends.

**Current state: API surface routing through torch / ROCm BLAS.** For
f16/bf16/f32 matmuls, ``torch.mm`` on AMD goes through hipBLASLt which
uses MFMA on gfx942/gfx950 — so the fast path is already MFMA-accelerated,
just not authored in FlyDSL. A dedicated FlyDSL MFMA kernel here would
ship tile/pipeline/stream-K tuning on top, which is where custom GEMMs
win over hipBLASLt. That's Phase 2 follow-up work matching QuACK's scale.

What's shipped:
    - ``gemm(A, B, bias=None, activation=None, alpha, beta, C, out_dtype)``
    - ``gemm_act(A, B, activation, bias=None, …)``
    - ``gemm_gated(A, B, gate_type='swiglu', …)``
    - ``gemm_symmetric(A, …)`` — C = A @ A.T
    - ``linear`` / ``mlp`` / ``linear_cross_entropy`` via ``quack/amd/linear.py``

What's NOT yet shipped (substantial follow-up):
    - Hand-tuned FlyDSL MFMA/WMMA kernels (reference:
      ``FlyDSL/kernels/preshuffle_gemm.py`` ~1500 lines, ``hgemm_splitk.py``
      ~850 lines, ``rdna_f16_gemm.py``).
    - Stream-K tile scheduler (plan: ``quack/amd/tile_scheduler.py`` using
      rocdl atomics — valuable specifically on gfx950/CDNA4).
    - Blockscaled fp8/fp4 (reference: ``FlyDSL/kernels/gemm_fp8fp4_gfx1250.py``,
      ``moe_blockscale_2stage.py``). hipBLASLt does not cover these so the
      FlyDSL kernel is the only path and is higher-priority than the
      standard-dtype GEMM port.
    - Fused epilogues beyond the simple activation/bias/gate set above.
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
