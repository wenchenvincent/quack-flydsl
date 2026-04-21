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
KERNEL_REGISTRY = {
    "mfma_16x16": lambda: _lazy_import(
        "quack.amd.gemm_gfx950", "gemm_mfma"
    ),
    "tiled_32x32": lambda: _lazy_import(
        "quack.amd.gemm_gfx950_tiled", "gemm_f16_32x32"
    ),
    "lds_32x32": lambda: _lazy_import(
        "quack.amd.gemm_gfx950_lds", "gemm_f16_32x32_lds"
    ),
    "lds_swz_32x32": lambda: _lazy_import(
        "quack.amd.gemm_gfx950_lds_swizzle", "gemm_f16_32x32_lds_swz"
    ),
    "4wave_64x64": lambda: _lazy_import(
        "quack.amd.gemm_gfx950_4wave", "gemm_f16_64x64_4wave"
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
_TUNED_TABLE: List[TuneEntry] = [
    # Example (commented) — uncomment + tune after benchmarking:
    # TuneEntry(
    #     predicate=lambda M, N, K, dt: (M, N, K) == (4096, 4096, 4096) and dt == torch.float16,
    #     kernel="4wave_64x64",
    #     notes="profiled 2026-04-22: 1.4× vs lds_swz on H200-equivalent AMD shape",
    # ),
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
    if not plain or dtype != torch.float16:
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
