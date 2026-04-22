# Copyright (c) 2026, AMD.

"""Linear — AMDGPU port of `quack/linear.py`.

Dispatches:

  - **plain bf16/f16 linear** (no bias, no activation, shape fits splitk):
    ``quack.amd.gemm_splitk`` → ~1.05-1.12× hipBLASLt at 4096²+
  - **plain MX-FP8 linear** (fp8 input + scales): ``quack.amd.mxfp8_gemm``
    → 1.5 PFLOPS at 8192³
  - **linear with bias/activation/residual**: falls back to ``quack.amd.gemm``
    which has the epilogue-capable NN MFMA kernel.

``weight`` layout is ``(out_features, in_features)`` matching
``torch.nn.Linear`` — this is NT w.r.t. the matmul, which matches
``gemm_splitk`` / ``mxfp8_gemm`` natively (no transpose copy needed).
"""

from typing import Optional

import torch
from torch import Tensor

from quack.amd.gemm import gemm


_SPLITK_ACTIVATIONS = {"relu", "relu_sq", "gelu_tanh_approx", "silu"}


def _splitk_eligible(
    x: Tensor, weight: Tensor, bias, activation,
) -> bool:
    """True iff (x, weight, bias, activation) can route directly through
    ``gemm_splitk`` with its fused-epilogue path."""
    if activation is not None and activation not in _SPLITK_ACTIVATIONS:
        return False
    if bias is not None and bias.dtype != torch.float32:
        # Fused bias path is f32-only; cast callers go through NN fallback.
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    if weight.dtype != x.dtype:
        return False
    if x.stride(-1) != 1 or weight.stride(-1) != 1:
        return False
    if x.dim() != 2 or weight.dim() != 2:
        return False
    M, K = x.shape
    N, K2 = weight.shape
    if K != K2:
        return False
    # gemm_splitk's default config uses tile_m=128, tile_n=256, tile_k=64.
    if not (M % 128 == 0 and N % 256 == 0 and K % 64 == 0 and M >= 128):
        return False
    if bias is not None and bias.shape != (N,):
        return False
    return True


def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
) -> Tensor:
    """``y = act(x @ weight.T + bias)``.

    ``weight`` is ``(out_features, in_features)`` matching ``torch.nn.Linear``.

    Fast path: when inputs are bf16/f16 with aligned shapes and the
    activation is one of ``{relu, relu_sq, gelu_tanh_approx, silu}``,
    routes through ``gemm_splitk`` with a fused epilogue — bias and
    activation are applied inside the matmul write-back in one kernel.
    Otherwise falls back to the NN ``gemm`` path which also supports
    the full epilogue matrix.
    """
    if _splitk_eligible(x, weight, bias, activation):
        from quack.amd.gemm_gfx950_splitk import gemm_splitk
        return gemm_splitk(
            x, weight, bias=bias,
            activation=activation if activation is not None else "none",
        )
    # Epilogue or non-aligned shape → NN kernel via transpose.
    w_t = weight.transpose(-1, -2).contiguous()
    return gemm(x, w_t, bias=bias, activation=activation)


_GATED_ACTIVATIONS = {"swiglu", "reglu", "geglu", "glu"}


def linear_gated(
    x: Tensor,
    weight_gate_up: Tensor,
    gate_type: str = "swiglu",
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Gated linear: ``y = gate(linear(x, w_gate_up, bias))``.

    ``weight_gate_up`` is ``(2 * hidden, in_features)`` — the first
    ``hidden`` rows produce the gate values, the remaining ``hidden``
    produce the up values. Output is ``(M, hidden)``.

    Gate types:
      - ``"swiglu"``: ``silu(gate) * up``
      - ``"reglu"``:  ``relu(gate) * up``
      - ``"geglu"``:  ``gelu_tanh_approx(gate) * up``
      - ``"glu"``:    ``sigmoid(gate) * up``

    Dispatch: when the matmul is splitk-eligible, the (M, 2*hidden)
    output runs through ``gemm_splitk`` (with fused bias if provided)
    and the gating is applied torch-side. The matmul dominates the
    cost so the elementwise gating adds <5% overhead; true in-kernel
    fusion of the gating into the write-back is a follow-up once the
    split-output shape transformation is wired.
    """
    assert gate_type in _GATED_ACTIVATIONS, (
        f"gate_type must be one of {_GATED_ACTIVATIONS}, got {gate_type!r}"
    )
    two_hidden, in_features = weight_gate_up.shape
    assert two_hidden % 2 == 0, (
        f"weight_gate_up's first dim must be even (= 2*hidden), got {two_hidden}"
    )

    # Route matmul through splitk when eligible (no gate_type baked yet,
    # just forward + optional bias).
    if _splitk_eligible(x, weight_gate_up, bias, activation=None):
        from quack.amd.gemm_gfx950_splitk import gemm_splitk
        out = gemm_splitk(x, weight_gate_up, bias=bias)
    else:
        # NN fallback via existing gemm path.
        w_t = weight_gate_up.transpose(-1, -2).contiguous()
        out = gemm(x, w_t, bias=bias)

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


def linear_mxfp8(
    x: Tensor,
    weight: Tensor,
    scale_x: Tensor,
    scale_w: Tensor,
    *,
    weight_shuffled: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """MX-FP8 linear: ``y = x @ weight.T`` with per-block fp8 scales.

    Args:
      x:              (M, K) fp8_e4m3fn. Row-major, last-dim contiguous.
      weight:         (N, K) fp8_e4m3fn. Either plain row-major or pre-shuffled
                      via ``quack.amd.gemm_gfx950_blockscaled.shuffle_b``
                      (set ``weight_shuffled=True`` to skip the in-launcher shuffle).
      scale_x:        (K//128, M) f32. ``scale_x[block_k, m]``.
      scale_w:        (N//128, K//128) f32. ``scale_w[block_n, block_k]``.
      weight_shuffled: if True, ``weight`` is already in the kernel's preshuffled form.
      out_dtype:      bf16 or f16.

    Returns: (M, N) tensor in ``out_dtype``.

    Constraints: M % 32 == 0, N % 128 == 0, K % 128 == 0.
    """
    from quack.amd.gemm_gfx950_blockscaled import mxfp8_gemm
    return mxfp8_gemm(
        x, weight, scale_x, scale_w,
        shuffled=weight_shuffled, out_dtype=out_dtype,
    )


def linear_residual(
    x: Tensor,
    weight: Tensor,
    residual: Tensor,
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
    alpha: float = 1.0,
    residual_scale: float = 1.0,
) -> Tensor:
    """Fused linear + residual: ``y = activation(alpha * (x @ W.T) + residual_scale * residual + bias)``.

    Common pattern in transformer residual branches — fuses the matmul
    with the residual add in one kernel, saving the (M, N) round-trip.

    When shapes are MFMA-eligible (f16/bf16 × f16/bf16 → same; M, in_f,
    out_f multiples of 16), routes through the FlyDSL MFMA kernel via
    ``gemm(alpha=alpha, beta=residual_scale, C=residual)``.

    ``residual`` must be f32 (the kernel's accumulator dtype). Callers
    with half-precision residuals should upcast via ``.float()`` before
    calling.
    """
    assert residual.dtype == torch.float32, (
        "linear_residual requires f32 residual (kernel expects f32 C)"
    )
    w_t = weight.transpose(-1, -2).contiguous()
    return gemm(
        x, w_t,
        bias=bias, activation=activation,
        alpha=alpha, beta=residual_scale, C=residual,
    )


__all__ = ["linear", "linear_gated", "linear_mxfp8", "linear_residual"]
