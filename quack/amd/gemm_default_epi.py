# Copyright (c) 2026, AMD.

"""Default GEMM epilogue: `out = alpha * acc + beta * C + row_bias + col_bias`.

AMDGPU counterpart to `quack/gemm_default_epi.py`. Lightweight skeleton
for now — the first AMD GEMM kernel emits the math inline rather than
going through a full mixin hierarchy. This module defines the ``_epi_ops``
tuple and a small ``apply_default_epi`` helper callable from the kernel
body.

Extension points:
    - Stochastic rounding (gfx950 has hardware support; wire via rocdl).
    - CShuffle epilogue (route acc → LDS → reshape → global store).
    - Fused activation (that's `gemm_act_epi.py`, layered on top).
"""

from typing import Optional

from flydsl.expr.arith import ArithValue

from quack.amd.epi_ops import Scalar, RowVecLoad, ColVecLoad


# The ordered tuple that the GEMM kernel iterates over at build time.
# Order matters: scalars first (alpha / beta), then loadable tensors.
DEFAULT_EPI_OPS = (
    Scalar("alpha", dtype="f32", default=1.0),
    Scalar("beta", dtype="f32", default=0.0),
    Scalar("sr_seed", dtype="i32", default=0),
    RowVecLoad("row_bias", dtype="f32"),
    ColVecLoad("col_bias", dtype="f32"),
)


def apply_default_epi(
    acc: ArithValue,
    *,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    c_val: Optional[ArithValue] = None,
    row_bias: Optional[ArithValue] = None,
    col_bias: Optional[ArithValue] = None,
) -> ArithValue:
    """Apply the default epilogue math to a single accumulator lane.

    Intended to be called from inside a ``@flyc.kernel`` body per output
    lane. Each argument is optional — pass ``None`` to skip that term and
    avoid emitting the op.

    Returns the post-epilogue f32 value (caller is responsible for dtype
    truncation to the output tensor's element type).
    """
    out = acc
    if alpha is not None and alpha != 1.0:
        out = ArithValue(out) * ArithValue(alpha)
    if c_val is not None and beta is not None and beta != 0.0:
        out = ArithValue(out) + ArithValue(beta) * ArithValue(c_val)
    if row_bias is not None:
        out = ArithValue(out) + ArithValue(row_bias)
    if col_bias is not None:
        out = ArithValue(out) + ArithValue(col_bias)
    return out


__all__ = ["DEFAULT_EPI_OPS", "apply_default_epi"]
