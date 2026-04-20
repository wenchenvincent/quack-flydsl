# Copyright (c) 2026, AMD.

"""FlyDSL-side counterpart to `quack/cute_dsl_utils.py`.

Dtype mapping, arch / wave-size detection, and parameter base classes used by
the AMD kernels under `quack.amd`.
"""

from typing import Optional
from functools import lru_cache
from dataclasses import dataclass, fields

import torch

import flydsl.expr as fx
from flydsl.expr import Float16, Float32, BFloat16, Int32, Int64
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch


# Torch dtype → FlyDSL Numeric class (matches QuACK's torch2cute_dtype_map shape).
torch2flydsl_dtype_map = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


# Compile-time constants: exempt from the partition helpers below so the
# dataclass / NamedTuple splitter treats them as bake-in values.
StaticTypes = (fx.Constexpr, int, bool, str, float, type(None))


@lru_cache
def get_wave_size(arch: Optional[str] = None) -> int:
    """Wave size in threads for the given GPU arch.

    CDNA (gfx9xx except RDNA ranges): 64. RDNA (gfx10/11/12xx): 32.
    Defaults to the auto-detected current arch.
    """
    if arch is None:
        arch = get_rocm_arch()
    return 32 if is_rdna_arch(arch) else 64


@lru_cache
def get_lds_size_bytes(arch: Optional[str] = None) -> int:
    """LDS capacity in bytes per CU / WGP for the given arch.

    Values from FlyDSL's SmemAllocator (see flydsl/utils/smem_allocator.py):
      - gfx942 (MI300X, CDNA3): 64 KiB
      - gfx950 (MI350/MI355X, CDNA4): 160 KiB
      - gfx1201 (RDNA4): 64 KiB
      - gfx1250 (MI450): 320 KiB
    """
    if arch is None:
        arch = get_rocm_arch()
    if arch.startswith("gfx950"):
        return 160 * 1024
    if arch.startswith("gfx1250"):
        return 320 * 1024
    return 64 * 1024


def _partition_fields(obj):
    """Split dataclass fields into (constexpr_dict, non_constexpr_dict) by type."""
    all_fields = {field.name: getattr(obj, field.name) for field in fields(obj)}
    constexpr = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
    non_constexpr = {n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)}
    return constexpr, non_constexpr


@dataclass
class ParamsBase:
    """Base dataclass for kernel parameter bundles.

    Mirrors QuACK's `cute_dsl_utils.ParamsBase` but targets FlyDSL's JIT
    boundary: fields typed as `fx.Constexpr[T]` are baked in at compile time,
    the rest flow through the runtime argument registry.
    """

    pass


__all__ = [
    "torch2flydsl_dtype_map",
    "StaticTypes",
    "get_rocm_arch",
    "is_rdna_arch",
    "get_wave_size",
    "get_lds_size_bytes",
    "ParamsBase",
    "_partition_fields",
]
