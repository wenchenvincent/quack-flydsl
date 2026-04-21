# Copyright (c) 2026, AMD.

"""gfx1250 (MI450) WMMA GEMM — delegates to ``gemm_rdna_wmma``.

gfx1250 shares the wave32 WMMA instruction family with gfx1201 plus
new scaled-WMMA variants for fp8/fp4 and 320 KiB LDS per CU. The
standard-dtype path is identical to gfx1201's, so both archs route
through ``gemm_rdna_wmma``. The scaled variants require
``wmma_scale_f32_16x16x128_f8f6f4`` and are a separate commit track.

Reference (FlyDSL):
  - ``FlyDSL/kernels/wmma_gemm_gfx1250.py`` — larger-tile standard.
  - ``FlyDSL/kernels/gemm_fp8fp4_gfx1250.py`` — scaled WMMA fp8/fp4.
"""

from quack.amd.gemm_rdna_wmma import gemm_wmma as _gemm_wmma


def gemm_mfma(*args, **kwargs):
    """gfx1250 'MFMA' GEMM — actually WMMA; shared WMMA kernel."""
    return _gemm_wmma(*args, **kwargs)


__all__ = ["gemm_mfma"]
