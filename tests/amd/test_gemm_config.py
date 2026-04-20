# Copyright (c) 2026, AMD.

"""Unit tests for quack.amd.gemm_config and related host-side helpers."""

import pytest

from quack.amd.gemm_config import (
    GemmConfig, get_default_config, get_all_configs,
)
from quack.amd.tile_scheduler import PersistenceMode
from quack.amd.fast_math import ceil_div, FastDivmod


def test_gemm_config_defaults():
    c = GemmConfig()
    assert c.tile_m == 128 and c.tile_n == 128 and c.tile_k == 64
    assert c.lds_stage == 2
    assert c.block_threads == 256
    assert c.persistence_mode is PersistenceMode.NONE


def test_default_config_gfx950_standard():
    c = get_default_config("gfx950", blockscaled=False)
    assert c.arch == "gfx950"
    assert c.tile_k == 64
    assert c.scale_block_k == 128  # unused but present


def test_default_config_gfx950_blockscaled():
    c = get_default_config("gfx950", blockscaled=True)
    assert c.arch == "gfx950"
    assert c.tile_k == 128
    assert c.scale_block_k == 128


def test_all_configs_standard_is_nonempty_and_all_valid():
    configs = get_all_configs("gfx950", blockscaled=False)
    assert len(configs) > 0
    for c in configs:
        assert c.tile_m % 16 == 0 and c.tile_n % 16 == 0
        assert c.tile_k in {64, 128}


def test_all_configs_blockscaled_uses_k128():
    configs = get_all_configs("gfx950", blockscaled=True)
    assert len(configs) > 0
    for c in configs:
        assert c.tile_k == 128
        assert c.scale_block_k == 128


def test_ceil_div():
    assert ceil_div(0, 8) == 0
    assert ceil_div(1, 8) == 1
    assert ceil_div(8, 8) == 1
    assert ceil_div(9, 8) == 2
    assert ceil_div(100, 7) == 15


def test_fast_divmod_matches_python():
    d = FastDivmod(7)
    for x in (0, 1, 6, 7, 8, 49, 100):
        assert d(x) == divmod(x, 7)


def test_epi_ops_dataclasses_frozen():
    """Ensure frozen=True is honoured so configs can be dict keys."""
    from quack.amd.epi_ops import Scalar, RowVecLoad, ColVecLoad
    s = Scalar("alpha")
    r = RowVecLoad("row_bias")
    c = ColVecLoad("col_bias")
    with pytest.raises(Exception):
        s.name = "beta"
    # dict-key usability
    d = {s: 1, r: 2, c: 3}
    assert len(d) == 3


def test_default_epi_ops_shape():
    from quack.amd.gemm_default_epi import DEFAULT_EPI_OPS
    names = [o.name for o in DEFAULT_EPI_OPS]
    assert names == ["alpha", "beta", "sr_seed", "row_bias", "col_bias"]
