# Copyright (c) 2026, AMD.

"""gfx1201 (RDNA4) WMMA GEMM — planned, not yet implemented.

RDNA4 uses wave32 + ``v_wmma_f32_16x16x16_f16`` instead of CDNA's MFMA.
The fragment layout and per-lane counts differ, so this needs a
separate builder rather than a thin alias of gfx950.

Reference port (FlyDSL): ``FlyDSL/kernels/rdna_f16_gemm.py`` covers
the f16 WMMA path; ``rdna_fp8_preshuffle_gemm.py`` covers fp8.
"""


def gemm_mfma(*args, **kwargs):
    raise NotImplementedError(
        "gfx1201 WMMA GEMM is not yet ported; "
        "route through the torch fallback for now. "
        "Reference: FlyDSL/kernels/rdna_f16_gemm.py"
    )


__all__ = ["gemm_mfma"]
