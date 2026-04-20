# Copyright (c) 2026, AMD.

"""gfx1250 (MI450) WMMA + scaled-WMMA GEMM — planned, not yet implemented.

gfx1250 uses wave32 + ``v_wmma_f32_16x16x16_f16`` (like gfx1201) plus
new scaled-MFMA / WMMA variants for fp8/fp4, and ships 320 KiB of LDS
per CU for aggressive pipelining.

Reference ports (FlyDSL):
  - ``FlyDSL/kernels/wmma_gemm_gfx1250.py`` — standard-dtype WMMA.
  - ``FlyDSL/kernels/gemm_fp8fp4_gfx1250.py`` — scaled WMMA for fp8/fp4.
  - ``FlyDSL/kernels/moe_gemm_2stage_common_gfx1250.py`` — MoE variant.
"""


def gemm_mfma(*args, **kwargs):
    raise NotImplementedError(
        "gfx1250 WMMA GEMM is not yet ported; "
        "route through the torch fallback for now. "
        "Reference: FlyDSL/kernels/wmma_gemm_gfx1250.py + gemm_fp8fp4_gfx1250.py"
    )


__all__ = ["gemm_mfma"]
