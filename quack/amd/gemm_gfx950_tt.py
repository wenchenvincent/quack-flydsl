# Copyright (c) 2026, AMD.

"""TT-layout GEMM: ``C = A.T @ B.T`` for AMDGPU.

Layout (BLAS-style, first letter = A's transpose, second = B's):
    - ``A``: ``(K, M)`` M-inner (transposed operand)
    - ``B``: ``(N, K)`` K-inner (transposed operand)
    - ``C = A.T @ B.T`` → ``(M, N)``

**This is a torch/hipBLASLt dispatch, not a dedicated FlyDSL MFMA kernel.**
The AMD training GEMM family only needs NN (dx), TN (dw), and NT (fwd) — TT
has no in-repo consumer (verified 2026-07-15: zero `A.t() @ B.t()` /
`gemm_tt` / `trans_a`/`trans_b` call sites in `quack/amd/`). Rather than ship
a ~700-line dedicated kernel with no caller, `gemm_tt` reuses the existing
:func:`quack.amd.gemm.gemm` path on transposed views: `A.t()` / `B.t()` are
stride/leading-dim flags to hipBLASLt (standard BLAS3 `op(A)`/`op(B)`
contract), so there is no materialized transpose copy. The full epilogue
(bias / activation / alpha / beta / C) flows through `gemm()` unchanged.

A dedicated `gemm_gfx950_tt.py` FlyDSL kernel — TN's A-loader recombined with
NT-pingpong's K-inner B-loader — is a roadmap item, to be built only if a
profiled workload shows this dispatch is a real bottleneck.
"""

from typing import Optional

import torch
from torch import Tensor

from quack.amd.gemm import gemm


def gemm_tt(
    A: Tensor,
    B: Tensor,
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """TT GEMM: ``out = act(alpha * (A.T @ B.T) + beta * C + bias)``.

    - ``A``: ``(K, M)`` — contracted on axis 0.
    - ``B``: ``(N, K)`` — contracted on axis 1.
    - Output: ``(M, N)``.

    Dispatches to :func:`quack.amd.gemm.gemm` on the transposed views
    ``A.t()`` (``(M, K)``) and ``B.t()`` (``(K, N)``); hipBLASLt applies the
    transposes as BLAS flags with no HBM copy. See the module docstring.
    """
    assert A.dim() == 2 and B.dim() == 2, "gemm_tt expects 2-D A, B"
    K_a, M = A.shape
    N, K_b = B.shape
    assert K_a == K_b, f"shared K mismatch: A is (K={K_a}, M={M}), B is (N={N}, K={K_b})"
    assert A.dtype == B.dtype, f"dtype mismatch: {A.dtype} vs {B.dtype}"
    return gemm(
        A.t(), B.t(),
        bias=bias, activation=activation,
        alpha=alpha, beta=beta, C=C, out_dtype=out_dtype,
    )


__all__ = ["gemm_tt"]
