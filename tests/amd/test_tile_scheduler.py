# Copyright (c) 2026, AMD.

"""Unit tests for quack.amd.tile_scheduler (host-side scheduler logic).

These tests don't launch kernels — they verify the raster-order resolution,
grid sizing, and tile index translation at the Python level.
"""

import pytest

from quack.amd.tile_scheduler import (
    PersistenceMode, RasterOrder, TilePlan,
    get_num_cus, grid_for_persistent, resolve_raster_order, tile_to_mn,
)


def test_cu_count_gfx950():
    assert get_num_cus("gfx950") == 256


def test_cu_count_unknown_arch_has_fallback():
    assert get_num_cus("gfx9999") > 0


def test_grid_for_persistent_none():
    plan = grid_for_persistent(100, num_cus=64, mode=PersistenceMode.NONE)
    assert plan.grid == 100
    assert plan.use_work_counter is False


def test_grid_for_persistent_static():
    plan = grid_for_persistent(500, num_cus=64, mode=PersistenceMode.STATIC)
    assert plan.grid == 64  # min(500, 64)
    assert plan.use_work_counter is False

    plan = grid_for_persistent(10, num_cus=64, mode=PersistenceMode.STATIC)
    assert plan.grid == 10  # min(10, 64)


def test_grid_for_persistent_dynamic():
    plan = grid_for_persistent(500, num_cus=64, mode=PersistenceMode.DYNAMIC)
    assert plan.grid == 64
    assert plan.use_work_counter is True


def test_grid_for_persistent_clc_degrades_to_dynamic():
    plan = grid_for_persistent(500, num_cus=64, mode=PersistenceMode.CLC)
    assert plan.mode is PersistenceMode.DYNAMIC  # degraded
    assert plan.grid == 64
    assert plan.use_work_counter is True


def test_raster_heuristic_picks_along_m_when_wide():
    # tiles_n > tiles_m → raster along M
    assert resolve_raster_order(RasterOrder.Heuristic, 4, 16) is RasterOrder.AlongM


def test_raster_heuristic_picks_along_n_when_tall():
    assert resolve_raster_order(RasterOrder.Heuristic, 32, 8) is RasterOrder.AlongN


@pytest.mark.parametrize("raster", [RasterOrder.AlongM, RasterOrder.AlongN])
def test_tile_to_mn_covers_all_tiles(raster):
    tiles_m, tiles_n = 6, 4
    total = tiles_m * tiles_n
    seen = set()
    for idx in range(total):
        m, n = tile_to_mn(idx, tiles_m, tiles_n, raster)
        assert 0 <= m < tiles_m and 0 <= n < tiles_n
        seen.add((m, n))
    assert len(seen) == total
