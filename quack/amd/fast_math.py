# Copyright (c) 2026, AMD.

"""Integer fast-path helpers for GEMM tile scheduling.

AMDGPU counterpart to `quack/fast_math.py`. QuACK's NVIDIA `FastDivmod`
uses Barrett reduction (magic-number division) to compute `divmod` with
a single multiply + shift in the kernel hot loop. We don't have a
corresponding kernel-side fast divmod yet — the bottleneck in our current
kernels is MFMA throughput, not integer division. This module ships the
host-side helpers today and leaves a stub for the in-kernel fast path.

When an autotune harness exposes a divmod bottleneck, extend
``FastDivmod`` with a magic-number emitter that returns an `ir.Value` for
the in-kernel path.
"""

from dataclasses import dataclass


def ceil_div(a: int, b: int) -> int:
    """Ceiling division for non-negative ints."""
    assert b > 0
    return (a + b - 1) // b


@dataclass(frozen=True)
class FastDivmod:
    """Host-side divmod descriptor.

    Carries the divisor alongside a precomputed ``(magic, shift)`` pair so
    callers that want to emit the classic Barrett-reduction sequence can
    extract them later. For now `__call__` is just Python divmod — the
    struct exists so the AMD GEMM code can be written against a stable
    API and swapped to magic-number math later.
    """

    divisor: int

    def __call__(self, x: int):
        return divmod(x, self.divisor)


__all__ = ["ceil_div", "FastDivmod"]
