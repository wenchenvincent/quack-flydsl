# Copyright (c) 2026, AMD.

"""gfx1201 (RDNA4) WMMA GEMM — delegates to ``gemm_rdna_wmma``.

RDNA4 uses wave32 + ``v_wmma_f32_16x16x16_f16`` instead of CDNA's MFMA.
The shared WMMA kernel lives in ``gemm_rdna_wmma`` and is reused by
both gfx1201 and gfx1250 (same instruction family).

Reference (FlyDSL): ``FlyDSL/kernels/rdna_f16_gemm.py`` (larger-tile
4-wave LDS-pipelined variant; the MVP here is the single-wave 16×16
equivalent of ``gemm_gfx950.py``).
"""

from quack.amd.gemm_rdna_wmma import gemm_wmma as _gemm_wmma


def gemm_mfma(*args, **kwargs):
    """gfx1201 'MFMA' GEMM — actually WMMA; kept under the MFMA name
    so the arch dispatcher can call a uniform symbol."""
    return _gemm_wmma(*args, **kwargs)


__all__ = ["gemm_mfma"]
