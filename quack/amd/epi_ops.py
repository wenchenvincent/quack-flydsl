# Copyright (c) 2026, AMD.

"""Composable epilogue operations — AMDGPU port of `quack/epi_ops.py`.

Each ``EpiOp`` is a lightweight descriptor with two lifecycle hooks a GEMM
kernel body calls at fixed points:

  - ``begin(ctx)`` — one-time per-lane setup (e.g. load a per-column bias once,
    shared across a lane's several accumulator rows).
  - ``apply(val, ctx)`` — the per-accumulator-value combine step.

The kernel builds the ordered op list from its compile-time flags, calls
``begin`` once, then ``apply`` per accumulator value inside the writeback loop.

**Compile-time-only branching.** Every op here branches only on compile-time
(Python) values — which optional tensors/scalars are present — so the hooks are
plain Python methods callable directly from a ``@flyc.kernel`` body. The AST
rewriter leaves them alone (it only intercepts ``if`` on *runtime* values). A
future op that needs a genuine ``scf.if`` on a runtime value must call
``ReplaceIfWithDispatch.scf_if_dispatch`` explicitly (see ``gemm_streamk.py``) —
do NOT write a bare ``if runtime_cond:`` in a hook.

Address arithmetic is kernel-specific, so ops never hardcode it: the loader
closures live on :class:`EpiContext`, supplied by the calling kernel.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from flydsl.expr import arith, math as _fm
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32
from flydsl.expr.typing import T


class EpiContext:
    """Per-lane epilogue state bundle passed to every :class:`EpiOp` hook.

    Kernel-agnostic: the calling kernel supplies the loader closures (which
    close over its own buffer handles + tiling address math) and updates
    ``out_row`` before each accumulator value's ``apply`` sweep.

    Fields:
      - ``alpha`` / ``beta``: ``ArithValue`` runtime scalars (or None).
      - ``out_col``: this lane's output column (fixed across its rows).
      - ``out_row``: current output row (set per accumulator value).
      - ``load_row_bias(col)`` -> f32 ir value: per-column bias load.
      - ``load_residual(row, col)`` -> f32 ir value: C[row, col] load.
      - ``bias_val``: cached per-column bias, filled by ``RowBias.begin``.
    """

    def __init__(self, *, alpha=None, beta=None, out_col=None,
                 load_row_bias: Optional[Callable] = None,
                 load_residual: Optional[Callable] = None):
        self.alpha = alpha
        self.beta = beta
        self.out_col = out_col
        self.out_row = None
        self.load_row_bias = load_row_bias
        self.load_residual = load_residual
        self.bias_val = None


@dataclass(frozen=True)
class EpiOp:
    """Base epilogue op. Default hooks are no-ops; subclasses override."""

    name: str = ""

    def begin(self, ctx: EpiContext) -> None:
        return None

    def apply(self, val, ctx: EpiContext):
        return val


@dataclass(frozen=True)
class AlphaScale(EpiOp):
    """``val <- alpha * val`` (accumulator scale)."""

    name: str = "alpha"

    def apply(self, val, ctx):
        return ArithValue(val) * ArithValue(ctx.alpha)


@dataclass(frozen=True)
class BetaResidual(EpiOp):
    """``val <- val + beta * C[row, col]`` (residual add)."""

    name: str = "beta_c"

    def apply(self, val, ctx):
        cin = ctx.load_residual(ctx.out_row, ctx.out_col)
        return ArithValue(val) + ArithValue(ctx.beta) * ArithValue(cin)


@dataclass(frozen=True)
class RowBias(EpiOp):
    """Per-column bias broadcast along M. Loaded once per lane in ``begin``
    (the column is fixed across a lane's accumulator rows), added in ``apply``."""

    name: str = "row_bias"

    def begin(self, ctx):
        ctx.bias_val = ArithValue(ctx.load_row_bias(ctx.out_col))

    def apply(self, val, ctx):
        return ArithValue(val) + ctx.bias_val


@dataclass(frozen=True)
class Activation(EpiOp):
    """Pointwise activation, open-coded in f32 (no libcall)."""

    name: str = "activation"
    kind: str = "relu"

    def apply(self, val, ctx):
        v = ArithValue(val)
        zero = arith.constant(0.0, type=T.f32)
        if self.kind == "relu":
            return v.maximumf(zero)
        if self.kind == "relu_sq":
            return v.maximumf(zero) * v
        if self.kind == "gelu_tanh_approx":
            import math as _py_math
            c1 = _py_math.sqrt(2.0 / _py_math.pi)
            c2 = 0.044715 * c1
            x_sq = v * v
            tanh_arg = v * (c1 + c2 * x_sq)
            tanh_z = Float32(1.0) - Float32(2.0) / (
                Float32(1.0) + _fm.exp(Float32(2.0) * tanh_arg, fastmath="fast")
            )
            return v * (Float32(0.5) + Float32(0.5) * tanh_z)
        if self.kind == "silu":
            return v / (Float32(1.0) + _fm.exp(-v, fastmath="fast"))
        raise ValueError(f"unsupported activation {self.kind!r}")


def build_epilogue(
    *, has_alpha: bool = False, has_c: bool = False,
    has_bias: bool = False, activation: Optional[str] = None,
):
    """Assemble the ordered ``EpiOp`` list from a kernel's compile-time flags.

    Order matches the hand-rolled sequence: alpha scale, then ``beta*C``, then
    bias, then activation (activation is always last, applied to the final
    pre-store value).
    """
    ops = []
    if has_alpha:
        ops.append(AlphaScale())
    if has_c:
        ops.append(BetaResidual())
    if has_bias:
        ops.append(RowBias())
    if activation is not None:
        ops.append(Activation(kind=activation))
    return tuple(ops)


def epilogue_begin(ops, ctx: EpiContext) -> None:
    """Run every op's one-time ``begin`` hook (compile-time-unrolled)."""
    for op in ops:
        op.begin(ctx)


def epilogue_apply(ops, val, ctx: EpiContext):
    """Apply every op's per-value combine to ``val`` (compile-time-unrolled)."""
    for op in ops:
        val = op.apply(val, ctx)
    return val


# ---- Legacy descriptor subset (kept for gemm_default_epi's DEFAULT_EPI_OPS) --


@dataclass(frozen=True)
class Scalar(EpiOp):
    """A scalar operand descriptor (alpha / beta / sr_seed)."""

    dtype: str = "f32"
    default: Optional[Any] = None


@dataclass(frozen=True)
class RowVecLoad(EpiOp):
    """A tensor broadcast along M (one value per N) — e.g. per-column bias."""

    dtype: str = "f32"


@dataclass(frozen=True)
class ColVecLoad(EpiOp):
    """A tensor broadcast along N (one value per M) — e.g. per-row bias."""

    dtype: str = "f32"


__all__ = [
    "EpiContext", "EpiOp", "AlphaScale", "BetaResidual", "RowBias", "Activation",
    "build_epilogue", "epilogue_begin", "epilogue_apply",
    "Scalar", "RowVecLoad", "ColVecLoad",
]
