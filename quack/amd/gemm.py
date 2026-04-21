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

from typing import Optional, Tuple

import torch
from torch import Tensor


def _arch_dispatch_mfma():
    """Return the arch-specific ``gemm_mfma`` function for the current device.

    gfx942 (CDNA3) and gfx950 (CDNA4) share the same MFMA atoms so both
    route through ``gemm_gfx950.gemm_mfma``. RDNA4 (gfx1201) and
    gfx1250 use WMMA — those stubs raise NotImplementedError until the
    WMMA kernels land, so eligibility checks in _mfma_eligible should
    gate them out before reaching here (they don't yet — see
    NotImplementedError path below).
    """
    from quack.amd.flydsl_utils import get_rocm_arch
    arch = get_rocm_arch()
    if arch in ("gfx942", "gfx950"):
        from quack.amd.gemm_gfx950 import gemm_mfma
        return gemm_mfma
    if arch == "gfx1201":
        from quack.amd.gemm_gfx1201 import gemm_mfma
        return gemm_mfma
    if arch == "gfx1250":
        from quack.amd.gemm_gfx1250 import gemm_mfma
        return gemm_mfma
    raise NotImplementedError(f"No MFMA/WMMA GEMM kernel for arch {arch!r}")


# Set of (input dtype, activation) combos the real FlyDSL MFMA kernel covers.
_MFMA_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_MFMA_SUPPORTED_ACTIVATIONS = {None, "relu", "relu_sq", "gelu_tanh_approx", "silu"}
_MFMA_SUPPORTED_OUT_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


def _arch_supports_mfma():
    """Does the current arch have an MFMA kernel landed (not a stub)?"""
    try:
        from quack.amd.flydsl_utils import get_rocm_arch
        arch = get_rocm_arch()
    except Exception:
        return False
    return arch in ("gfx942", "gfx950")


def _pick_fast_path(A, B, bias, activation, alpha, beta, C, out_dtype):
    """Pick the widest-tile f16 kernel that fits the shape via the
    autotune selector.

    Only triggers for plain ``A @ B`` calls (no bias/act/alpha/beta/C)
    in f16 with f32 output — the epilogue-capable 16×16 path handles
    everything else. Delegates to ``quack.amd.gemm_autotune`` which
    consults any tuned-shape table, falling back to a "widest aligned
    tile wins" heuristic.

    Returns the output tensor or ``None`` if no fast path applies.
    """
    if A.dtype not in (torch.float16, torch.bfloat16) or A.dtype != B.dtype:
        return None
    if out_dtype != torch.float32:
        return None
    if bias is not None or activation is not None:
        return None
    if alpha != 1.0 or beta != 0.0 or C is not None:
        return None
    if not _arch_supports_mfma():
        return None
    M, K = A.shape
    _, N = B.shape
    from quack.amd.gemm_autotune import select_best_kernel, get_kernel
    kernel_name = select_best_kernel(M, N, K, A.dtype, plain=True)
    if kernel_name is None or kernel_name == "mfma_16x16":
        return None
    return get_kernel(kernel_name)(A, B)


def _mfma_eligible(A, B, bias, activation, alpha, beta, C, out_dtype):
    """Can the FlyDSL MFMA kernel handle this call?

    The kernel's supported scope:
      - f16/bf16 inputs, M/N/K multiples of 16
      - alpha any float, beta any float (C required when beta != 0)
      - C: optional f32 tensor, same shape as output
      - activations: relu / relu_sq / gelu_tanh_approx / silu
      - out_dtype: f32 / f16 / bf16
    """
    if not _arch_supports_mfma():
        return False
    if A.dtype not in _MFMA_SUPPORTED_DTYPES or A.dtype != B.dtype:
        return False
    if A.dim() != 2 or B.dim() != 2:
        return False
    # Kernel requires both inputs' last-dim stride to be 1. Callers that
    # pass transposed tensors (e.g. linear(x, w.T)) need to materialise.
    if A.stride(-1) != 1 or B.stride(-1) != 1:
        return False
    M, K = A.shape
    K2, N = B.shape
    if K != K2 or M % 16 or N % 16 or K % 16:
        return False
    if activation not in _MFMA_SUPPORTED_ACTIVATIONS:
        return False
    if out_dtype not in _MFMA_SUPPORTED_OUT_DTYPES:
        return False
    if C is not None and (C.shape != (M, N) or C.dtype != torch.float32 or C.stride(-1) != 1):
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
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
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

    Varlen (``cu_seqlens_m``): packed-batch input where A is
    ``(total_M, K)`` with samples concatenated along M. For plain GEMM
    (no per-sample bias / activation / masking) this is equivalent to a
    regular ``(total_M, K) @ (K, N)`` matmul — the dispatcher just
    routes through the standard path.

    ``A_idx``: optional gather-A row index; when present ``A`` is
    gathered via ``A[A_idx]`` before the matmul (host-side for the MVP).
    """
    assert A.is_cuda and B.is_cuda
    if cu_seqlens_m is not None:
        from quack.amd.varlen_utils import validate_varlen
        validate_varlen(cu_seqlens_m, A.size(0))
    if A_idx is not None:
        A = A.index_select(0, A_idx.long())
    # Per-sample bias (``(B, N)``) needs per-row lookup before it can be
    # added. For the MVP we expand to ``(total_M, N)`` on-device using
    # ``row_to_sample_idx`` and add after the GEMM on the torch path — a
    # fused kernel path that consults cu_seqlens in the epilogue is a
    # follow-up.
    if cu_seqlens_m is not None and bias is not None and bias.dim() == 2:
        from quack.amd.varlen_utils import row_to_sample_idx
        B_count, N_bias = bias.shape
        assert N_bias == B.size(-1), "per-sample bias last-dim must match N"
        sample_ids = row_to_sample_idx(cu_seqlens_m)
        assert sample_ids.size(0) == A.size(0)
        per_row_bias = bias[sample_ids.long()]  # (total_M, N)
        # Route through the torch path; bias_val added below as a full
        # (M, N) tensor rather than the eligible-kernel's per-column bias.
        out_f32 = gemm(
            A, B, alpha=alpha, beta=beta, C=C,
            out_dtype=torch.float32, bias=None, activation=None,
        )
        out_f32 = out_f32 + per_row_bias.to(torch.float32)
        if activation is not None:
            if activation == "relu":
                out_f32 = torch.relu(out_f32)
            elif activation == "relu_sq":
                out_f32 = torch.relu(out_f32) * out_f32
            elif activation == "silu":
                out_f32 = torch.nn.functional.silu(out_f32)
            elif activation == "gelu_tanh_approx":
                out_f32 = torch.nn.functional.gelu(out_f32, approximate="tanh")
            else:
                raise NotImplementedError(f"activation={activation!r}")
        return out_f32.to(out_dtype or A.dtype)
    # Default output dtype: match input (torch.matmul convention), not f32.
    effective_out_dtype = out_dtype or A.dtype
    if _mfma_eligible(A, B, bias, activation, alpha, beta, C, effective_out_dtype):
        # Fast path: if the call has no epilogue (alpha=1, beta=0, bias=None,
        # activation=None, C=None) and inputs are f16, pick the widest
        # tile the shape allows from the larger-tile MFMA variants.
        picked = _pick_fast_path(A, B, bias, activation, alpha, beta, C, effective_out_dtype)
        if picked is not None:
            return picked
        gemm_mfma = _arch_dispatch_mfma()
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


def gemm_symmetric(A, out_dtype=None, **kw):
    """Symmetric GEMM: C = A @ A.T.

    Routes through the dedicated FlyDSL kernel
    ``quack.amd.gemm_symmetric.gemm_symmetric`` when inputs are eligible
    (f16/bf16, M/K multiples of 16, last-dim contig). Falls back to
    ``gemm(A, A.T)`` otherwise.
    """
    if (
        A.is_cuda
        and A.dtype in (torch.float16, torch.bfloat16)
        and A.dim() == 2
        and A.stride(-1) == 1
        and A.size(0) % 16 == 0 and A.size(1) % 16 == 0
        and not kw   # dedicated kernel doesn't yet take bias/activation/etc.
    ):
        from quack.amd.gemm_symmetric import gemm_symmetric as _gemm_symmetric_kernel
        return _gemm_symmetric_kernel(A, out_dtype=out_dtype)
    return gemm(A, A.transpose(-1, -2).contiguous(), out_dtype=out_dtype, **kw)


def _act_bwd(preact: Tensor, activation: Optional[str]) -> Tensor:
    """Pointwise activation backward: ``d(act(x))/dx`` evaluated at ``preact``."""
    if activation is None:
        return torch.ones_like(preact)
    x = preact.float()
    if activation == "relu":
        return (x > 0).to(preact.dtype)
    if activation == "relu_sq":
        return (2.0 * torch.relu(x)).to(preact.dtype)
    if activation == "silu":
        sig = torch.sigmoid(x)
        return (sig * (1 + x * (1 - sig))).to(preact.dtype)
    if activation == "gelu_tanh_approx":
        # d/dx of gelu_tanh(x).  Use torch autograd for reliability.
        preact_f = x.detach().requires_grad_(True)
        y = torch.nn.functional.gelu(preact_f, approximate="tanh")
        (grad,) = torch.autograd.grad(y.sum(), preact_f)
        return grad.to(preact.dtype)
    raise NotImplementedError(f"activation={activation!r}")


def _gemm_dact_fused_eligible(A, B, PreAct, activation):
    if A.dtype != torch.float16 or B.dtype != torch.float16:
        return False
    if A.dim() != 2 or B.dim() != 2 or PreAct.dim() != 2:
        return False
    M, K = A.shape
    K2, N = B.shape
    if K != K2 or M % 16 or N % 16 or K % 16:
        return False
    if PreAct.shape != (M, N) or PreAct.stride(-1) != 1:
        return False
    if A.stride(-1) != 1 or B.stride(-1) != 1:
        return False
    if activation not in (None, "relu", "relu_sq", "silu", "gelu_tanh_approx"):
        return False
    if not _arch_supports_mfma():
        return False
    return True


def gemm_dact(
    A: Tensor,
    B: Tensor,
    PreAct: Tensor,
    activation: Optional[str] = None,
    *,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """Activation-backward fused GEMM: returns ``(dx, postact)`` where
    ``dx = (A @ B) * activation'(PreAct)`` and ``postact = activation(PreAct)``.

    Dispatches to the fused FlyDSL kernel
    (``quack.amd.gemm_dact_kernel.gemm_dact_fused``) when eligible —
    the activation-fwd/bwd is computed in the MFMA epilogue from the
    per-lane accumulator, saving the HBM round-trip through a
    transient f32 buffer. Falls back to a torch-composed path
    otherwise (non-f16 inputs, unsupported shapes).
    """
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = PreAct.dtype if postact_dtype is None else postact_dtype
    if _gemm_dact_fused_eligible(A, B, PreAct, activation):
        from quack.amd.gemm_dact_kernel import gemm_dact_fused
        return gemm_dact_fused(
            A, B, PreAct, activation=activation,
            out_dtype=out_dtype, postact_dtype=postact_dtype,
        )
    # Torch fallback.
    dout = gemm(A, B, out_dtype=torch.float32)
    act_prime = _act_bwd(PreAct, activation)
    dx = (dout * act_prime.float()).to(out_dtype)
    if activation is None:
        postact = PreAct.to(postact_dtype)
    elif activation == "relu":
        postact = torch.relu(PreAct).to(postact_dtype)
    elif activation == "relu_sq":
        postact = (torch.relu(PreAct) ** 2).to(postact_dtype)
    elif activation == "silu":
        postact = torch.nn.functional.silu(PreAct).to(postact_dtype)
    elif activation == "gelu_tanh_approx":
        postact = torch.nn.functional.gelu(PreAct, approximate="tanh").to(postact_dtype)
    else:
        raise NotImplementedError(f"activation={activation!r}")
    return dx, postact


_GATED_BWD_FNS = {
    "swiglu": lambda gate, up, dout: (
        dout * up * torch.sigmoid(gate) * (1.0 + gate * (1.0 - torch.sigmoid(gate))),
        dout * torch.nn.functional.silu(gate),
    ),
    "reglu": lambda gate, up, dout: (
        dout * up * (gate > 0).to(dout.dtype),
        dout * torch.relu(gate),
    ),
    "geglu": lambda gate, up, dout: (
        dout * up * torch.autograd.grad(
            torch.nn.functional.gelu(
                gate.detach().requires_grad_(True), approximate="tanh"
            ).sum(),
            gate.detach().requires_grad_(True) if False else gate,
        )[0],
        dout * torch.nn.functional.gelu(gate, approximate="tanh"),
    ),
    "glu": lambda gate, up, dout: (
        dout * up * torch.sigmoid(gate) * (1.0 - torch.sigmoid(gate)),
        dout * torch.sigmoid(gate),
    ),
}


def _gemm_dgated_fused_eligible(A, B, PreAct, gate_type):
    if A.dtype != torch.float16 or B.dtype != torch.float16:
        return False
    if A.dim() != 2 or B.dim() != 2 or PreAct.dim() != 2:
        return False
    M, K = A.shape
    K2, N = B.shape
    if K != K2 or M % 16 or N % 16 or K % 16:
        return False
    if PreAct.shape != (M, 2 * N) or PreAct.stride(-1) != 1:
        return False
    if A.stride(-1) != 1 or B.stride(-1) != 1:
        return False
    if gate_type not in ("swiglu", "reglu", "geglu", "glu"):
        return False
    if not _arch_supports_mfma():
        return False
    return True


def gemm_dgated(
    A: Tensor,
    B: Tensor,
    PreAct: Tensor,
    gate_type: str = "swiglu",
    *,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """Gated-activation-backward fused GEMM.

    ``PreAct`` is ``(M, 2N)`` with interleaved ``gate, up`` in the last
    axis (matching NVIDIA QuACK's convention). Returns ``(dx, postact)``
    where ``dx`` is ``(M, 2N)`` interleaved ``(dgate, dup)`` and
    ``postact`` is ``(M, N)`` forward output.

    Dispatches to the fused FlyDSL kernel
    (``quack.amd.gemm_dgated_kernel.gemm_dgated_fused``) when eligible
    — the gate fwd/bwd happens in the MFMA epilogue per-lane. Falls
    back to torch autograd composition otherwise.
    """
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = PreAct.dtype if postact_dtype is None else postact_dtype
    if _gemm_dgated_fused_eligible(A, B, PreAct, gate_type):
        from quack.amd.gemm_dgated_kernel import gemm_dgated_fused
        return gemm_dgated_fused(
            A, B, PreAct, gate_type=gate_type,
            out_dtype=out_dtype, postact_dtype=postact_dtype,
        )
    dout = gemm(A, B, out_dtype=torch.float32)
    gate = PreAct[..., ::2].float()
    up = PreAct[..., 1::2].float()
    if gate_type == "swiglu":
        postact = (torch.nn.functional.silu(gate) * up)
    elif gate_type == "reglu":
        postact = torch.relu(gate) * up
    elif gate_type == "geglu":
        postact = torch.nn.functional.gelu(gate, approximate="tanh") * up
    elif gate_type == "glu":
        postact = torch.sigmoid(gate) * up
    else:
        raise NotImplementedError(f"gate_type={gate_type!r}")
    # Autograd path for robust gradient (matches NVIDIA ref impl).
    g = PreAct[..., ::2].detach().requires_grad_(True)
    u = PreAct[..., 1::2].detach().requires_grad_(True)
    if gate_type == "swiglu":
        y = torch.nn.functional.silu(g) * u
    elif gate_type == "reglu":
        y = torch.relu(g) * u
    elif gate_type == "geglu":
        y = torch.nn.functional.gelu(g, approximate="tanh") * u
    elif gate_type == "glu":
        y = torch.sigmoid(g) * u
    dgate, dup = torch.autograd.grad(y, [g, u], dout)
    dx = torch.stack([dgate, dup], dim=-1).reshape(PreAct.shape)
    return dx.to(out_dtype), postact.to(postact_dtype)


def _gemm_norm_act_fused_eligible(A, B, colvec, rowvec, bias, C, alpha, beta, activation):
    if A.dtype != torch.float16 or B.dtype != torch.float16:
        return False
    if A.dim() != 2 or B.dim() != 2:
        return False
    M, K = A.shape
    K2, N = B.shape
    if K != K2 or M % 16 or N % 16 or K % 16:
        return False
    if A.stride(-1) != 1 or B.stride(-1) != 1:
        return False
    # Fused kernel doesn't yet handle alpha != 1 / beta*C — those stay on
    # the torch path. Bias (N,) and colvec (M,), rowvec (N,) are supported.
    if alpha != 1.0 or beta != 0.0 or C is not None:
        return False
    if activation not in (None, "relu", "relu_sq", "silu", "gelu_tanh_approx"):
        return False
    if bias is not None and (bias.dim() != 1 or bias.size(0) != N):
        return False
    if colvec is not None and (colvec.dim() != 1 or colvec.size(0) != M):
        return False
    if rowvec is not None and (rowvec.dim() != 1 or rowvec.size(0) != N):
        return False
    if not _arch_supports_mfma():
        return False
    return True


def gemm_norm_act(
    A: Tensor,
    B: Tensor,
    colvec: Optional[Tensor] = None,
    rowvec: Optional[Tensor] = None,
    *,
    bias: Optional[Tensor] = None,
    C: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    activation: Optional[str] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Fused (A @ B + bias + beta*C) × colvec × rowvec → activation.

    ``colvec`` is a per-row (M,) scale (typically ``rstd`` from a prior
    norm); ``rowvec`` is a per-column (N,) scale (typically a learned
    weight). Matches QuACK's ``gemm_norm_act`` surface.

    Routes through the fused FlyDSL kernel (``gemm_norm_act_kernel``)
    when eligible — bias / colvec / rowvec / activation all fold into
    the MFMA epilogue register before the output store. Falls back to
    torch-composed path for alpha/beta/C calls which the fused kernel
    doesn't yet support.
    """
    out_dtype = out_dtype or A.dtype
    if _gemm_norm_act_fused_eligible(A, B, colvec, rowvec, bias, C, alpha, beta, activation):
        from quack.amd.gemm_norm_act_kernel import gemm_norm_act_fused
        return gemm_norm_act_fused(
            A, B, colvec=colvec, rowvec=rowvec,
            bias=bias, activation=activation, out_dtype=out_dtype,
        )
    # Compute GEMM + bias + alpha/beta/C in f32 for numerical headroom
    # before the row/col scaling.
    d = gemm(A, B, bias=bias, alpha=alpha, beta=beta, C=C, out_dtype=torch.float32)
    if colvec is not None:
        d = d * colvec.float().unsqueeze(-1)
    if rowvec is not None:
        d = d * rowvec.float().unsqueeze(-2)
    if activation is None:
        pass
    elif activation == "relu":
        d = torch.relu(d)
    elif activation == "relu_sq":
        d = torch.relu(d) ** 2
    elif activation == "silu":
        d = torch.nn.functional.silu(d)
    elif activation == "gelu_tanh_approx":
        d = torch.nn.functional.gelu(d, approximate="tanh")
    else:
        raise NotImplementedError(f"activation={activation!r}")
    return d.to(out_dtype)


__all__ = [
    "gemm", "gemm_act", "gemm_gated", "gemm_symmetric",
    "gemm_dact", "gemm_dgated", "gemm_norm_act",
]
