# Copyright (c) 2026, AMD.

"""Base class for row-reduction kernels (rmsnorm, softmax, cross_entropy).

FlyDSL counterpart to ``quack/reduction_base.py``. Drops CuTe's
cluster/mbarrier machinery (no direct AMD equivalent pre-gfx1250) and leaves
LDS scratch allocation split across the two natural phases in a FlyDSL kernel
builder:

    - **Build time** (outside ``@flyc.kernel``): reserve bytes in the
      ``SmemAllocator`` cursor via ``reserve_reduction_scratch``, which
      returns byte offsets.
    - **Kernel body time** (inside ``@flyc.kernel``): obtain the base LDS
      memref via ``allocator.get_base()`` and wrap each returned offset in a
      ``SmemPtr`` — see ``kernels/rmsnorm_kernel.py`` in the FlyDSL repo for
      the canonical idiom.

Kernels built on this base typically subclass and set ``_threads_per_row``.
"""

from dataclasses import dataclass
from typing import List, Type

from flydsl.expr.numeric import Float32, Numeric
from flydsl.utils.smem_allocator import SmemAllocator

from quack.amd.flydsl_utils import get_wave_size


@dataclass(frozen=True)
class ReductionScratchPlan:
    """Build-time plan for the LDS scratch used by block-level reductions.

    - ``offsets``: byte offsets within the workgroup's LDS slab, one per
      requested scratch buffer.
    - ``num_slots``: number of entries in each scratch buffer (= num waves).
    - ``elem_bytes``: element size in bytes (currently only 4 / f32).
    """

    offsets: List[int]
    num_slots: int
    elem_bytes: int


class ReductionBase:
    """Configuration + LDS-scratch helpers for row-reduction kernels."""

    def __init__(
        self,
        dtype: Type[Numeric],
        N: int,
        stage: int,
        reduction_dtype: Type[Numeric] = Float32,
    ):
        self.dtype = dtype
        self.N = N
        self.stage = stage
        self.reduction_dtype = reduction_dtype

    # --- Launch / block geometry -----------------------------------------

    def _threads_per_row(self) -> int:
        raise NotImplementedError

    def _num_threads(self) -> int:
        """Workgroup size heuristic — matches QuACK's CuTe side verbatim."""
        return 128 if self.N <= 16384 else 256

    def _num_waves(self, wave_size: int = None) -> int:
        """Number of waves per workgroup for ``_num_threads`` threads."""
        if wave_size is None:
            wave_size = get_wave_size()
        nt = self._num_threads()
        assert nt % wave_size == 0, f"_num_threads={nt} must be a multiple of wave_size={wave_size}"
        return nt // wave_size

    # --- LDS scratch ------------------------------------------------------

    def reserve_reduction_scratch(
        self,
        allocator: SmemAllocator,
        num_scratch: int = 1,
        wave_size: int = None,
    ) -> ReductionScratchPlan:
        """Reserve ``num_scratch`` scratch slabs in the LDS allocator cursor.

        Each slab holds ``num_waves`` elements of ``reduction_dtype``. The
        returned plan carries the byte offsets — construct ``SmemPtr(base,
        offset, T.f32, shape=(num_slots,))`` from inside the ``@flyc.kernel``
        body to turn each offset into a usable pointer.
        """
        num_waves = self._num_waves(wave_size)
        elem_bytes = self._reduction_elem_bytes()
        offsets: List[int] = []
        for _ in range(num_scratch):
            offsets.append(_reserve_lds_bytes(allocator, num_waves * elem_bytes))
        return ReductionScratchPlan(
            offsets=offsets, num_slots=num_waves, elem_bytes=elem_bytes,
        )

    def _reduction_elem_bytes(self) -> int:
        rd = self.reduction_dtype
        if rd is Float32:
            return 4
        raise NotImplementedError(f"reduction_dtype={rd} not wired up yet")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _reserve_lds_bytes(allocator: SmemAllocator, nbytes: int, align: int = 16) -> int:
    """Advance the allocator cursor by ``nbytes`` (aligned), return the start offset.

    Mirrors the FlyDSL in-kernel idiom used in ``kernels/rmsnorm_kernel.py``.
    """
    offset = allocator._align(allocator.ptr, align)
    allocator.ptr = offset + nbytes
    return offset


__all__ = ["ReductionBase", "ReductionScratchPlan"]
