# Copyright (c) 2026, AMD.

"""GEMM autotune skeleton — picks the best kernel variant for a shape.

Today ``quack.amd.gemm`` auto-dispatches based on a fixed tier of
"widest tile wins" heuristics:

    M,N % 64 == 0  →  gemm_f16_64x64_4wave
    M,N % 32 == 0  →  gemm_f16_32x32_lds_swz
    else           →  gemm_gfx950.gemm_mfma (16×16)

This module is the configuration / override layer. Callers that have
**profiled** their workload can drop a ``(M, N, K, dtype) → kernel_name``
entry into ``_TUNED_TABLE`` and the dispatcher will use it instead of
the default heuristic. The table is a list (not a dict) so shape
ranges can be expressed via predicates rather than exact keys.

Empty table today — pending benchmarking runs. The infrastructure is
in place so profiling results land as one-line additions rather than
code edits across the dispatch surface.

Future work (mirrors QuACK NVIDIA's autotune flow):
  - Per-shape perf database (JSON / pickle on disk at ``~/.quack_amd_tune``).
  - ``@autotune`` decorator that JIT-benchmarks the top-K configs on
    first call for an unseen shape and caches the winner.
  - ``quack.amd.cli.tune`` CLI that sweeps and populates the DB.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch


# Available kernel variants — maps a name to a lazy import + launcher.
# The tiled/LDS/swizzle/4-wave kernels all accept f16 and bf16 via a
# shared ``_DTYPE2STR`` flag in their launcher (the 16×16 path is the
# full-epilogue one and uses its own dispatch).
KERNEL_REGISTRY = {
    "mfma_16x16": lambda: _lazy_import(
        "quack.amd.gemm_gfx950", "gemm_mfma"
    ),
    "tiled_32x32": lambda: _lazy_import(
        "quack.amd.gemm_gfx950_tiled", "gemm_32x32"
    ),
    "lds_32x32": lambda: _lazy_import(
        "quack.amd.gemm_gfx950_lds", "gemm_32x32_lds"
    ),
    "lds_swz_32x32": lambda: _lazy_import(
        "quack.amd.gemm_gfx950_lds_swizzle", "gemm_32x32_lds_swz"
    ),
    "4wave_64x64": lambda: _lazy_import(
        "quack.amd.gemm_gfx950_4wave", "gemm_64x64_4wave"
    ),
}


def _lazy_import(module_name: str, attr: str):
    import importlib
    return getattr(importlib.import_module(module_name), attr)


@dataclass(frozen=True)
class TuneEntry:
    """One entry in the tuned-shape table.

    ``predicate`` takes ``(M, N, K, dtype)`` and returns True when the
    entry applies. ``kernel`` is a key into ``KERNEL_REGISTRY``.
    """
    predicate: Callable[[int, int, int, torch.dtype], bool]
    kernel: str
    notes: str = ""


# Populate with profiled winners as benchmarks land. Each entry is
# consulted in order; first match wins. Keep predicates precise (hit
# only the shape they were measured on) to avoid regressing untested
# shapes.
#
# Empirical sweep on MI355X / gfx950, 2026-04-21 (f16 inputs):
#   M, N, K        hipblaslt   tiled_32   lds_32    lds_swz   4wave_64
#   256, 256, 256  26 μs       24 μs ⬅   70 μs     74 μs     77 μs     (tiled wins 3×)
#   512, 512, 512  28 μs       76 μs     73 μs     78 μs     64 μs ⬅  (4wave wins)
#   1024²          24 μs       76 μs     135 μs    135 μs    71 μs ⬅  (4wave wins)
#   2048²          26 μs       481 μs    515 μs    515 μs    482 μs    (≈tie)
# The heuristic's "widest-tile wins" rule mis-picks at 256² by ~3×;
# tiled_32x32 is the right call for small shapes. Above 512, 4wave
# takes over.  Below is the override table keyed on that cut.
_TUNED_TABLE: List[TuneEntry] = [
    TuneEntry(
        predicate=lambda M, N, K, dt: (
            dt in (torch.float16, torch.bfloat16)
            and M <= 384 and N <= 384
            and M % 32 == 0 and N % 32 == 0 and K % 16 == 0
        ),
        kernel="tiled_32x32",
        notes=(
            "Small shapes: LDS/wave overhead dominates on 64×64 tile — "
            "tiled_32x32 in-register wins ~3× at 256² on MI355X."
        ),
    ),
]


def select_best_kernel(
    M: int, N: int, K: int, dtype: torch.dtype,
    *, plain: bool = True,
) -> Optional[str]:
    """Return a key into ``KERNEL_REGISTRY`` for the best kernel variant
    for this shape, or ``None`` if the call needs the epilogue-capable
    16×16 path (e.g. has bias / activation / alpha-beta-C).

    ``plain`` flag: set False to signal the call has an epilogue; we
    only have an epilogue-capable kernel for 16×16 today.
    """
    if not plain:
        return "mfma_16x16"
    if dtype not in (torch.float16, torch.bfloat16):
        return "mfma_16x16"
    # Consult the tuned table first.
    for entry in _TUNED_TABLE:
        if entry.predicate(M, N, K, dtype):
            return entry.kernel
    # Default heuristic: widest aligned tile wins.
    if K % 16:
        return "mfma_16x16"
    if M % 64 == 0 and N % 64 == 0:
        return "4wave_64x64"
    if M % 32 == 0 and N % 32 == 0:
        return "lds_swz_32x32"
    if M % 16 == 0 and N % 16 == 0:
        return "mfma_16x16"
    return None


def get_kernel(name: str):
    """Resolve a kernel key to its callable (lazy-imported)."""
    if name not in KERNEL_REGISTRY:
        raise KeyError(
            f"Unknown kernel {name!r}; registered: {sorted(KERNEL_REGISTRY)}"
        )
    return KERNEL_REGISTRY[name]()


def all_kernel_names() -> Tuple[str, ...]:
    return tuple(KERNEL_REGISTRY.keys())


__all__ = [
    "TuneEntry", "KERNEL_REGISTRY",
    "select_best_kernel", "get_kernel", "all_kernel_names",
]
