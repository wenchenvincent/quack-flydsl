# Copyright (c) 2026, AMD.

"""Shared copy / LDS helpers for QuACK's AMD kernels.

AMDGPU counterpart to `quack/copy_utils.py`. These helpers centralise the
patterns used across `quack.amd.rmsnorm`, `softmax`, and `cross_entropy`
for scalar global<->register buffer access. Kernels currently inline the
helpers (FlyDSL's AST rewriter only walks the `@flyc.kernel` body, so
helpers defined here are intentionally small / AST-agnostic).

Exports:
    - ``bufcopy_for(bits)``: pick the right ``BufferCopy{16,32}b()`` atom for
      a given element width.
    - ``reserve_lds_bytes(allocator, nbytes, align=16)``: bump the cursor on
      a ``SmemAllocator`` and return the aligned offset (used at kernel-build
      time).
"""

import flydsl.expr as fx
from flydsl.utils.smem_allocator import SmemAllocator


def bufcopy_for(bits: int):
    """AMD ``rocdl.BufferCopy{16,32}b`` atom keyed by element width in bits."""
    return fx.rocdl.BufferCopy16b() if bits <= 16 else fx.rocdl.BufferCopy32b()


def reserve_lds_bytes(allocator: SmemAllocator, nbytes: int, align: int = 16) -> int:
    """Align the allocator cursor to ``align`` bytes and reserve ``nbytes``.

    Mirrors the FlyDSL in-kernel idiom used across
    ``FlyDSL/kernels/*.py``. Call this at kernel-build time (outside
    ``@flyc.kernel``); use the returned offset to construct an ``SmemPtr``
    inside the kernel body.
    """
    offset = allocator._align(allocator.ptr, align)
    allocator.ptr = offset + nbytes
    return offset


__all__ = ["bufcopy_for", "reserve_lds_bytes"]
