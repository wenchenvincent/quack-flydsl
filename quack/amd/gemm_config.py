# Copyright (c) 2026, AMD.

"""GEMM configuration — AMDGPU counterpart to `quack/gemm_config.py`.

Mirrors the NVIDIA `GemmConfig` dataclass but drops fields that don't
translate to AMD MFMA kernels (`pingpong`, `use_tma_gather`, `cluster_m/n`,
`is_dynamic_persistent` — we use `PersistenceMode` from
`quack.amd.tile_scheduler` directly). Adds AMD-specific fields.

The full autotune machinery (ranking / per-shape tuning) is future work.
For now `get_default_config(dtype, arch, blockscaled)` returns a safe
default per-arch; `get_all_configs(...)` returns the small menu of known-
good tile shapes cribbed from FlyDSL's `_TILE_PRELOAD_TABLE`.
"""

from dataclasses import dataclass
from typing import List, Optional

from quack.amd.tile_scheduler import PersistenceMode


@dataclass(frozen=True)
class GemmConfig:
    """Configuration for a FlyDSL MFMA GEMM kernel build.

    Fields match QuACK's NVIDIA shape where possible so downstream callers
    can be written against one config dataclass and dispatch per-device.
    """

    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 64

    # Swap A and B to reduce the `tile_m` direction when it's the short side.
    swap_ab: bool = False

    # Raster swizzle granularity (0 = no swizzle). Used by the tile scheduler
    # to pick a raster order that maximises L2 reuse.
    max_swizzle_size: int = 8

    # LDS pipeline depth. 1 = no ping-pong; 2 = classic ping-pong.
    lds_stage: int = 2

    # Use B preshuffle (transform B into the MFMA-friendly layout off-device
    # or in a prologue pass). Leaves B as a normal (K, N) row-major tensor
    # when False.
    use_preshuffle_b: bool = False

    # Emit a CShuffle epilogue (reshape accumulator through LDS before
    # store) instead of per-thread scalar stores.
    use_cshuffle_epilog: bool = False

    # Workgroup size in threads. 256 = 4 MFMA warps on CDNA (wave64).
    block_threads: int = 256

    # Blockscaled-only: how many K-elements share one scale.
    scale_block_k: int = 128

    # Persistent-kernel / stream-K mode. Defaults to NONE (one workgroup
    # per tile).
    persistence_mode: PersistenceMode = PersistenceMode.NONE

    # Split-K factor for the hgemm_splitk pattern. 1 = no split.
    split_k: int = 1

    # Target GPU arch this config is for. Used at dispatch time to skip
    # configs that don't apply to the active device.
    arch: str = "gfx950"


# ---------------------------------------------------------------------------
# Default / tile-menu tables
# ---------------------------------------------------------------------------


# Safe default per (arch, blockscaled) — chosen to work for any (M, N, K)
# that's a multiple of the tile. Callers that want the perf-tuned config
# should drive through `get_all_configs` + an autotune harness.
_DEFAULTS = {
    ("gfx950", False): GemmConfig(
        tile_m=128, tile_n=128, tile_k=64,
        lds_stage=2, block_threads=256, arch="gfx950",
    ),
    ("gfx950", True): GemmConfig(
        tile_m=128, tile_n=128, tile_k=128, scale_block_k=128,
        lds_stage=2, block_threads=256, arch="gfx950",
    ),
    ("gfx942", False): GemmConfig(
        tile_m=128, tile_n=128, tile_k=64,
        lds_stage=2, block_threads=256, arch="gfx942",
    ),
    ("gfx942", True): GemmConfig(
        tile_m=128, tile_n=128, tile_k=128, scale_block_k=128,
        lds_stage=2, block_threads=256, arch="gfx942",
    ),
}


def get_default_config(arch: str, blockscaled: bool = False) -> GemmConfig:
    """Return a safe default ``GemmConfig`` for the arch + kernel kind."""
    key = (arch if arch in {a for a, _ in _DEFAULTS} else "gfx950", blockscaled)
    return _DEFAULTS[key]


# Tile menu cribbed from FlyDSL's _TILE_PRELOAD_TABLE (preshuffle_gemm.py:40–120).
# Only the shapes that have been validated upstream; expand as the autotune
# harness grows.
_STANDARD_TILE_MENU = [
    (64, 64, 64), (64, 128, 64), (128, 64, 64),
    (128, 128, 64), (128, 128, 128),
    (128, 256, 64), (128, 256, 128),
    (256, 128, 64), (256, 128, 128),
]

_BLOCKSCALED_TILE_MENU = [
    (64, 64, 128), (64, 128, 128), (128, 64, 128),
    (128, 128, 128), (128, 256, 128), (256, 128, 128),
]


def get_all_configs(
    arch: str = "gfx950",
    blockscaled: bool = False,
    *,
    epilogue: Optional[str] = None,
) -> List[GemmConfig]:
    """Enumerate the known-good tile shapes for ``arch``.

    ``epilogue`` is currently unused but kept in the signature to mirror
    QuACK's NVIDIA ``get_all_configs`` so a future autotune harness can
    narrow by epilogue (e.g. gated variants need `tile_n % 32 == 0`).
    """
    menu = _BLOCKSCALED_TILE_MENU if blockscaled else _STANDARD_TILE_MENU
    return [
        GemmConfig(
            tile_m=m, tile_n=n, tile_k=k,
            lds_stage=2, block_threads=256, arch=arch,
            scale_block_k=128 if blockscaled else 128,
        )
        for (m, n, k) in menu
    ]


__all__ = ["GemmConfig", "get_default_config", "get_all_configs"]
