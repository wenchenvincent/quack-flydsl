# Copyright (c) 2026, AMD.

"""gfx942 (CDNA3 / MI300) MFMA GEMM — shares the gfx950 builder.

gfx942 supports ``v_mfma_f32_16x16x16f16`` and
``v_mfma_f32_16x16x16bf16_1k`` identically to gfx950 (the CDNA4
additions are scaled MFMA + larger LDS; plain MFMA atoms match). So
the standard-dtype GEMM path from ``quack/amd/gemm_gfx950.py`` runs
unchanged — this module just re-exports it for arch-explicit callers.

Deferred for gfx942-specific tuning:
  - LDS-size-aware tiling (64 KiB vs 160 KiB on gfx950 — smaller tiles
    on gfx942 for aggressive pipelining).
  - gfx942 has 304 CUs (MI300X) vs 256 on gfx950 (MI355) — stream-K
    should pick up the difference through ``get_num_cus(arch)``.
"""

from quack.amd.gemm_gfx950 import gemm_mfma as _gemm_mfma


def gemm_mfma(*args, **kwargs):
    """gfx942 MFMA GEMM — delegates to the shared gfx950 builder."""
    return _gemm_mfma(*args, **kwargs)


__all__ = ["gemm_mfma"]
