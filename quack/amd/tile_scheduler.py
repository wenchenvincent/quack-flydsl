# Copyright (c) 2026, AMD.

"""Tile scheduling helpers for persistent / stream-K GEMM kernels.

AMDGPU counterpart to `quack/tile_scheduler.py`. QuACK's CuTe-DSL scheduler
supports ``PersistenceMode.{NONE, STATIC, DYNAMIC, CLC}`` with raster order
and swizzle; this module provides the AMD equivalent, minus ``CLC`` (NVIDIA
cluster launch control — no analogue pre-gfx1250).

**Current state:** Python-side helpers (CU count query, raster-order
classification, grid sizing) — these are production-ready and tested
(``tests/amd/test_tile_scheduler.py``). The in-kernel atomic work-counter
loop is documented below but not yet emitted; wire it into a client kernel
when the perf gain vs. one-wg-per-tile is motivated by profiling.

Usage sketch for STATIC persistent (no atomics, even work split)::

    from quack.amd.tile_scheduler import (
        PersistenceMode, get_num_cus, grid_for_persistent,
    )

    num_cus = get_num_cus("gfx950")
    total_tiles = num_tiles_m * num_tiles_n
    plan = grid_for_persistent(total_tiles, num_cus, mode=PersistenceMode.STATIC)

    @flyc.kernel
    def gemm_static_persistent(A, B, C, total_tiles_runtime):
        bid = fx.block_idx.x  # 0..plan.grid-1
        # Each workgroup covers a strided set of tiles.
        for step in range(0, ceil_div(total_tiles_compile_time, plan.grid)):
            tile_idx = bid + step * plan.grid
            if tile_idx < total_tiles_runtime:
                # decode tile_idx -> (tile_m, tile_n) per `plan.raster_order`
                # ... MFMA body ...

Usage sketch for DYNAMIC stream-K (atomic work counter)::

    from flydsl._mlir.dialects import llvm
    # In the kernel body:
    tile_idx = bid   # initial tile
    while tile_idx < total_tiles_runtime:
        # ... process tile at tile_idx ...
        # Grab next tile atomically. `work_counter` is an i32 scalar in
        # GMEM, zero-initialised on the host before launch. The result of
        # AtomicRMWOp is the *old* value, so we add `num_cus` to get the
        # tile index beyond the initial grid.
        new_idx = llvm.AtomicRMWOp(
            llvm.AtomicBinOp.add,
            work_counter_ptr, const_one_i32,
            llvm.AtomicOrdering.monotonic,
            syncscope="agent", alignment=4,
        ).result
        tile_idx = new_idx + num_cus

Partial-tile stream-K (splitting K across workgroups, combining partial
accumulators via atomic fadd on f32 output) is a further extension: the
output tensor must be zero-init on the host, each workgroup writes its
K-slice's partial C via ``rocdl.raw_ptr_buffer_atomic_fadd``, and the
reduction is associative-but-not-deterministic across workgroup order.
"""

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Optional


class PersistenceMode(IntEnum):
    NONE = 0
    STATIC = 1       # Grid = min(total_tiles, num_cus). No work counter.
    DYNAMIC = 2      # Grid = num_cus. Atomic work counter for tile pickup.
    # CLC (NVIDIA cluster launch control) has no pre-gfx1250 analogue and
    # transparently degrades to DYNAMIC on AMD. Callers can pass CLC if they
    # want the intent preserved in code; the scheduler treats it as DYNAMIC.
    CLC = 3


class RasterOrder(IntEnum):
    AlongM = 0       # tile index increments along M first
    AlongN = 1       # tile index increments along N first
    Heuristic = 2    # pick AlongM if tiles_n > tiles_m else AlongN


@dataclass(frozen=True)
class TilePlan:
    """Resolved launch plan for a persistent GEMM-style kernel.

    ``grid``: grid-x to pass to ``.launch(grid=(..., 1, 1))``.
    ``use_work_counter``: True for DYNAMIC/CLC modes (kernel must emit an
    atomic-increment loop that reads from a zero-initialised int32 buffer).
    ``raster_order``: effective raster order (Heuristic resolved to AlongM/N).
    """

    mode: PersistenceMode
    grid: int
    use_work_counter: bool
    raster_order: RasterOrder


# Rough CU count by architecture. These are published configurations; if the
# actual device has fewer enabled CUs (e.g. salvage SKUs), the kernel just
# runs fewer workgroups — correct but slightly suboptimal.
_CU_COUNT_BY_ARCH = {
    "gfx942": 304,   # MI300X (CDNA3)
    "gfx950": 256,   # MI350/MI355X (CDNA4)
    "gfx1201": 64,   # Radeon AI PRO R9700 (RDNA4)
    "gfx1250": 304,  # MI450 placeholder
}


@lru_cache(maxsize=None)
def get_num_cus(arch: str) -> int:
    """Return the nominal compute-unit count for ``arch``."""
    for prefix, cus in _CU_COUNT_BY_ARCH.items():
        if arch.startswith(prefix):
            return cus
    return 64  # conservative default


def resolve_raster_order(
    raster_order: RasterOrder, tiles_m: int, tiles_n: int,
) -> RasterOrder:
    if raster_order is not RasterOrder.Heuristic:
        return raster_order
    return RasterOrder.AlongM if tiles_n > tiles_m else RasterOrder.AlongN


def grid_for_persistent(
    total_tiles: int,
    num_cus: int,
    mode: PersistenceMode,
    tiles_m: int = 0,
    tiles_n: int = 0,
    raster_order: RasterOrder = RasterOrder.Heuristic,
) -> TilePlan:
    """Pick the grid dimension and raster policy for a persistent kernel.

    - ``NONE``: grid = ``total_tiles`` (one workgroup per tile, no persistence).
    - ``STATIC``: grid = ``min(total_tiles, num_cus)``. Tiles are partitioned
      evenly at compile time; each workgroup processes a fixed chunk. No
      atomic counter.
    - ``DYNAMIC`` / ``CLC``: grid = ``num_cus``. Workgroups loop pulling the
      next tile index via atomic increment of a shared work counter.
    """
    if mode is PersistenceMode.NONE:
        return TilePlan(
            mode=mode, grid=total_tiles, use_work_counter=False,
            raster_order=resolve_raster_order(raster_order, tiles_m, tiles_n),
        )
    if mode is PersistenceMode.STATIC:
        return TilePlan(
            mode=mode, grid=min(total_tiles, num_cus), use_work_counter=False,
            raster_order=resolve_raster_order(raster_order, tiles_m, tiles_n),
        )
    # DYNAMIC or CLC
    return TilePlan(
        mode=PersistenceMode.DYNAMIC, grid=num_cus, use_work_counter=True,
        raster_order=resolve_raster_order(raster_order, tiles_m, tiles_n),
    )


def tile_to_mn(
    tile_idx: int, tiles_m: int, tiles_n: int, raster_order: RasterOrder,
) -> tuple:
    """Translate a flat tile index into ``(m, n)`` per raster order.

    Host-side helper mirroring what the kernel would emit; callers can use
    it for testing / reference.
    """
    if raster_order is RasterOrder.AlongM:
        return (tile_idx % tiles_m, tile_idx // tiles_m)
    return (tile_idx // tiles_n, tile_idx % tiles_n)


__all__ = [
    "PersistenceMode",
    "RasterOrder",
    "TilePlan",
    "get_num_cus",
    "resolve_raster_order",
    "grid_for_persistent",
    "tile_to_mn",
]
