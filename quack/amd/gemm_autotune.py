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
    "4wave_64x64_lds": lambda: _lazy_import(
        "quack.amd.gemm_gfx950_4wave_lds", "gemm_64x64_4wave_lds"
    ),
    "4wave_64x64_lds_pp": lambda: _lazy_import(
        "quack.amd.gemm_gfx950_4wave_lds_pp", "gemm_64x64_4wave_lds_pp"
    ),
    "128x128": lambda: _lazy_import(
        "quack.amd.gemm_gfx950_128x128", "gemm_128x128"
    ),
    "128x128_k32": lambda: _lazy_import(
        "quack.amd.gemm_gfx950_128x128_k32", "gemm_128x128_k32"
    ),
    "128x128_ldma": lambda: _lazy_import(
        "quack.amd.gemm_gfx950_128x128_ldma", "gemm_128x128_ldma"
    ),
    "128x128_k32_ldma": lambda: _lazy_import(
        "quack.amd.gemm_gfx950_128x128_k32_ldma", "gemm_128x128_k32_ldma"
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
# Empirical sweeps on MI355X / gfx950, f16 inputs.
#
# 2026-04-21 (scalar BufferCopy16b loads — pre-vectorization):
#   256²: tiled_32 24 μs, 4wave 77 μs  →  tiled wins 3×
#   512²: tiled_32 76 μs, 4wave 64 μs  →  4wave wins
#   1024²: tiled_32 76 μs, 4wave 71 μs →  4wave wins
#
# 2026-04-21 (vectorized BufferCopy64b A loads):
#   256²:  tiled_32 13 μs, 4wave 11 μs →  4wave wins
#   1024²: tiled_32 78 μs, 4wave 31 μs →  4wave wins 2.3×
#
# Once the A-load path is vectorized, the LDS/wave overhead that made
# tiled_32x32 win at small shapes is amortized by the 4× reduction in
# HBM instructions. 4wave wins across the range for small-medium.
#
# 2026-04-21 cooperative-LDS 4-wave (single-stage ``4wave_64x64_lds``):
#   shape    4wave   4wave_lds  ratio
#   256²     11 μs   13 μs      0.84× — non-LDS wins (barrier cost)
#   1024²    31 μs   47 μs      0.68× — non-LDS wins
#   2048²    165 μs  110 μs     1.50× — cooperative LDS wins
#   4096²    1212 μs 678 μs     1.79× — cooperative LDS wins
#
# With ping-pong (2-stage LDS, ``4wave_64x64_lds_pp``):
#   shape      4wave    lds_pp    ratio (pp/4wave)
#   1024²      31 μs    36 μs     0.86×  — still behind non-LDS
#   2048²      165 μs   106 μs    1.55×  — ≈ LDS, slightly better
#   4096²      1212 μs  660 μs    1.84×  — matches LDS
#
# Ping-pong replaces single-stage LDS at large shapes (same wins, no
# regression at mid-range vs single-stage). Below 2048² the barrier
# cost still outweighs the HBM savings even with overlap — non-LDS
# 4wave remains the default.
# 2026-04-21 128×128 tile (``128x128``) measurements (μs, f16):
#   shape    4wave    lds_pp    128x128   128_vs_pp
#   1024²    31       37        57        0.64×  — 128 hurts (few WGs)
#   2048²    165      108       108       0.99×  — tie
#   4096²    1215     656       370       1.77×  — 128 wins big
#
# 128×128 needs enough workgroups in flight to hide its bigger per-WG
# load cost — crossover near 4096² on MI355X (256 CUs).
#
# 2026-04-21 K=32 MFMA tile (``128x128_k32``): uses mfma_f32_16x16x32_f16
# (2× throughput per op on gfx950). Per-lane fragments 8 f16 (vs 4);
# LDS per stage 8 KiB (vs 4 KiB). Measured at 4096²:
#   shape    k16 (128x128)   k32
#   1024²    58              48     1.21× faster
#   2048²    111             72     1.54× faster   ← K=32 sweet spot
#   4096²    368             366     1.01× (flat — HBM-bound dominates)
# K=32 wins at 1024–3999 where MFMA throughput is the bottleneck; at
# 4096² the working set spills L2 and HBM bandwidth dominates over
# MFMA issue rate.
#
# 2026-04-21 Direct HBM→LDS DMA (``128x128_ldma``): uses
# rocdl.raw_ptr_buffer_load_lds so the coop load skips the
# gmem→reg→LDS roundtrip. Same data, fewer wait-counter classes,
# tighter MFMA/load interleave:
#   shape    k16 (128x128)   ldma
#   1024²    107             95.7   1.12× faster
#   2048²    278             220    1.27× faster
#   4096²    412             340    1.21× faster   ← finally closes some of HBM gap
# LDMA wins everywhere 128x128 runs; replaces the K=16 128x128 entry
# for 4096² since it strictly dominates there.
#
# 2026-04-21 K=32 + LDMA (``128x128_k32_ldma``): combines K=32 MFMA's
# 2× per-op throughput with LDMA's skipped register roundtrip. DMA
# thread mapping is rearranged so that lane L in each wave writes LDS
# byte L*16 (matching buffer_load_lds auto-indexing), independent of
# the MFMA fragment layout. Composes strongly on both wins:
#   shape    k16   k32    ldma   k32_ldma   best
#   1024²    37    29     30     28.7       k32_ldma
#   2048²    104   66     83     56         k32_ldma  (1.86× vs k16)
#   4096²    328   371    276    228        k32_ldma  (1.44× vs k16, 1.21× vs ldma)
# Wins at every K÷32 shape; replaces k32 and ldma entries for those.
_TUNED_TABLE: List[TuneEntry] = [
    TuneEntry(
        predicate=lambda M, N, K, dt: (
            dt in (torch.float16, torch.bfloat16)
            and M >= 1024 and N >= 1024
            and M % 128 == 0 and N % 128 == 0 and K % 32 == 0
        ),
        kernel="128x128_k32_ldma",
        notes=(
            "Medium-plus shapes (≥ 1024² with K÷32): K=32 MFMA + HBM→LDS DMA "
            "composes both per-op throughput and wait-counter wins; 1.44× vs "
            "K=16 baseline, 1.21× vs LDMA alone at 4096² on MI355X."
        ),
    ),
    TuneEntry(
        predicate=lambda M, N, K, dt: (
            dt in (torch.float16, torch.bfloat16)
            and M >= 4096 and N >= 4096
            and M % 128 == 0 and N % 128 == 0 and K % 16 == 0
        ),
        kernel="128x128_ldma",
        notes=(
            "Very large shapes (≥ 4096², K÷16 but not ÷32): 128×128 with "
            "direct HBM→LDS DMA; 1.21× vs the scalar-stage 128×128."
        ),
    ),
    TuneEntry(
        predicate=lambda M, N, K, dt: (
            dt in (torch.float16, torch.bfloat16)
            and M >= 1024 and N >= 1024
            and M % 128 == 0 and N % 128 == 0 and K % 32 == 0
        ),
        kernel="128x128_k32",
        notes=(
            "Medium-large shapes (1024-3999): K=32 MFMA doubles per-op "
            "throughput; 1.21× at 1024², 1.54× at 2048² vs K=16 128x128."
        ),
    ),
    TuneEntry(
        predicate=lambda M, N, K, dt: (
            dt in (torch.float16, torch.bfloat16)
            and M >= 2048 and N >= 2048
            and M % 64 == 0 and N % 64 == 0 and K % 16 == 0
        ),
        kernel="4wave_64x64_lds_pp",
        notes=(
            "Large shapes (2048-3999, K not ÷32): cooperative-LDS + ping-pong; "
            "1.55-1.84× vs non-LDS 4wave on MI355X."
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
