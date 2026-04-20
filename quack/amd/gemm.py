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
    - Blockscaled fp8/fp4. **Supported on gfx950/CDNA4** (our hardware) via
      ``rocdl.mfma_scale_f32_16x16x128_f8f6f4`` — reference:
      ``FlyDSL/kernels/blockscale_preshuffle_gemm.py`` and
      ``kernels/moe_blockscale_2stage.py`` (both have explicit ``_is_gfx950``
      branches). ``kernels/gemm_fp8fp4_gfx1250.py`` is a separate WMMA-based
      variant for gfx1250/MI450. hipBLASLt does not cover block-scaled
      quantisation, so the FlyDSL kernel is the only path and is
      higher-priority than the standard-dtype GEMM port.
    - Fused epilogues beyond the simple activation/bias/gate set above.
"""

from typing import Optional

import torch
from torch import Tensor


# Set of (input dtype, activation) combos the real FlyDSL MFMA kernel covers.
_MFMA_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_MFMA_SUPPORTED_ACTIVATIONS = {None, "relu", "relu_sq", "gelu_tanh_approx", "silu"}
_MFMA_SUPPORTED_OUT_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


def _mfma_eligible(A, B, bias, activation, alpha, beta, C, out_dtype):
    """Can the FlyDSL MFMA kernel handle this call?

    The kernel's supported scope:
      - f16/bf16 inputs, M/N/K multiples of 16
      - alpha any float, beta any float (C required when beta != 0)
      - C: optional f32 tensor, same shape as output
      - activations: relu / relu_sq / gelu_tanh_approx / silu
      - out_dtype: f32 / f16 / bf16
    """
    if A.dtype not in _MFMA_SUPPORTED_DTYPES or A.dtype != B.dtype:
        return False
    if A.dim() != 2 or B.dim() != 2:
        return False
    M, K = A.shape
    K2, N = B.shape
    if K != K2 or M % 16 or N % 16 or K % 16:
        return False
    if activation not in _MFMA_SUPPORTED_ACTIVATIONS:
        return False
    if out_dtype not in _MFMA_SUPPORTED_OUT_DTYPES:
        return False
    if C is not None and (C.shape != (M, N) or C.dtype != torch.float32):
        return False
    if beta != 0.0 and C is None:
        return False
    if bias is not None:
        if bias.dim() != 1 or bias.size(0) != N:
            return False
    return True


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

    Matches QuACK's NVIDIA ``gemm`` API. Dispatches to the FlyDSL MFMA
    kernel (``quack.amd.gemm_gfx950.gemm_mfma``) when the call falls
    inside the kernel's supported scope; otherwise falls back to a
    torch-native path (which itself routes through hipBLASLt → MFMA on
    CDNA, so the fallback is still MFMA-accelerated).

    Supported by the FlyDSL kernel today (grows over time):
        - dtypes: f16/bf16 inputs, f32/f16/bf16 output
        - shapes: M, N, K all multiples of 16
        - alpha=1, beta=0, C=None
        - activations: relu / relu_sq / gelu_tanh_approx / silu
        - per-column f32 bias
    """
    assert A.is_cuda and B.is_cuda
    # Default output dtype: match input (torch.matmul convention), not f32.
    effective_out_dtype = out_dtype or A.dtype
    if _mfma_eligible(A, B, bias, activation, alpha, beta, C, effective_out_dtype):
        from quack.amd.gemm_gfx950 import gemm_mfma
        _bias = bias
        if _bias is not None and _bias.dtype != torch.float32:
            _bias = _bias.to(torch.float32)
        _C = C
        if _C is not None and _C.dtype != torch.float32:
            _C = _C.to(torch.float32)
        return gemm_mfma(
            A, B,
            bias=_bias, activation=activation,
            out_dtype=effective_out_dtype,
            alpha=alpha, beta=beta, C=_C,
        )
    # Torch fallback (hipBLASLt on AMD → MFMA under the hood).
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


def _gated_mfma_eligible(A, B, gate_type, out_dtype):
    if A.dtype not in _MFMA_SUPPORTED_DTYPES or A.dtype != B.dtype:
        return False
    if A.dim() != 2 or B.dim() != 2:
        return False
    M, K = A.shape
    K2, N = B.shape
    if K != K2 or M % 16 or N % 32 or K % 16:
        return False
    if gate_type not in {"swiglu", "reglu", "geglu", "glu"}:
        return False
    if out_dtype not in _MFMA_SUPPORTED_OUT_DTYPES:
        return False
    return True


def gemm_gated(A, B, gate_type: str = "swiglu", **kw):
    """Gated GEMM: split the output along the last dim into ``(gate, up)``
    and apply the gating function. Matches QuACK's ``gemm_gated`` semantics.

    Dispatches to the fused ``quack.amd.gemm_gated.gemm_gated`` FlyDSL
    kernel when inputs are eligible (f16/bf16 × f16/bf16, M multiple of
    16, N multiple of 32, K multiple of 16). Otherwise falls back to
    a torch gemm + elementwise pipeline.
    """
    out_dtype = kw.get("out_dtype") or A.dtype
    if _gated_mfma_eligible(A, B, gate_type, out_dtype):
        # The fused kernel saves the round-trip that the torch path takes.
        from quack.amd.gemm_gated import gemm_gated as _gemm_gated_kernel
        return _gemm_gated_kernel(A, B, gate_type=gate_type, out_dtype=out_dtype)

    # Torch fallback.
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
