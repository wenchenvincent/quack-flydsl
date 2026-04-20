# Copyright (c) 2026, AMD.

"""Composable epilogue operation descriptors — AMDGPU port of `quack/epi_ops.py`.

Each ``EpiOp`` is a lightweight dataclass describing a single epilogue
operand (a scalar alpha/beta, a row-broadcast bias, a column-broadcast
bias, etc.). The GEMM kernel body reads the tuple of ``EpiOp``s at build
time and emits the appropriate load / broadcast / apply code inline.

This is a strict subset of QuACK's NVIDIA ``EpiOp`` framework — we drop
the per-stage smem allocation and the ``begin / begin_loop / end``
lifecycle hooks (our first AMD GEMM doesn't need a CShuffle epilogue, so
the simpler "register-only" pattern is fine). When an epilogue grows to
need LDS scratch, extend ``EpiOp`` with the lifecycle methods.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class EpiOp:
    """Base class for epilogue operands. Subclasses carry kind-specific metadata."""

    name: str


@dataclass(frozen=True)
class Scalar(EpiOp):
    """A scalar fed into the epilogue (e.g. ``alpha``, ``beta``, ``sr_seed``).

    ``dtype`` is a Python representation of the target MLIR type — kept as
    a string for now to avoid needing an MLIR context at module import.
    """

    dtype: str = "f32"
    default: Optional[Any] = None


@dataclass(frozen=True)
class RowVecLoad(EpiOp):
    """A tensor broadcast along the M-dim (one value per N).

    Typical use: per-output-column bias. The kernel body loads
    ``vec[n]`` once per thread's N-column and adds it to the accumulator.
    """

    dtype: str = "f32"


@dataclass(frozen=True)
class ColVecLoad(EpiOp):
    """A tensor broadcast along the N-dim (one value per M).

    Typical use: per-output-row bias (less common than row-vec). The
    kernel body loads ``vec[m]`` once per thread's M-row.
    """

    dtype: str = "f32"


__all__ = ["EpiOp", "Scalar", "RowVecLoad", "ColVecLoad"]
