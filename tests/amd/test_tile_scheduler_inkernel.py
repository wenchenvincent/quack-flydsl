# Copyright (c) 2026, AMD.

"""In-kernel tile-scheduler helpers (G13): tile_idx_to_mn + DynamicTileScheduler.

DynamicTileScheduler is exercised end-to-end through gemm_f16_streamk (which
now consumes it); this file adds a host-side unit test of the raster decode and
a stream-K correctness test at a shape whose tile count is not a multiple of
num_cus, so the dynamic work-stealing tail path is actually hit.
"""

import pytest
import torch

from quack.amd.tile_scheduler import tile_to_mn, RasterOrder, get_num_cus


def test_tile_to_mn_both_raster_orders():
    # Host-side reference decode used by the in-kernel tile_idx_to_mn mirror.
    tiles_m, tiles_n = 3, 5
    seen_n = set()
    for idx in range(tiles_m * tiles_n):
        m, n = tile_to_mn(idx, tiles_m, tiles_n, RasterOrder.AlongN)
        assert 0 <= m < tiles_m and 0 <= n < tiles_n
        seen_n.add((m, n))
    assert len(seen_n) == tiles_m * tiles_n  # bijection over the tile grid
    seen_m = set()
    for idx in range(tiles_m * tiles_n):
        m, n = tile_to_mn(idx, tiles_m, tiles_n, RasterOrder.AlongM)
        seen_m.add((m, n))
    assert len(seen_m) == tiles_m * tiles_n


def test_streamk_tail_tiles_via_scheduler():
    """A shape with total_tiles not a multiple of num_cus forces the dynamic
    scheduler's work-stealing + tail-tile guard to be exercised."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm_streamk import gemm_f16_streamk

    num_cus = get_num_cus(__import__("quack.amd.flydsl_utils",
                          fromlist=["get_rocm_arch"]).get_rocm_arch())
    torch.manual_seed(1)
    # Pick M,N so tiles = (M/16)*(N/16) is deliberately not a multiple of num_cus.
    for M, N, K in [(256, 176, 64), (144, 272, 128)]:
        tiles = (M // 16) * (N // 16)
        assert tiles % num_cus != 0, (tiles, num_cus)  # confirm tail case
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        C = gemm_f16_streamk(A, B)
        ref = A.float() @ B.float()
        atol = max(5e-3, K * 2e-5)
        torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)
