# Copyright (c) 2026, AMD.

"""Shared autotune infrastructure for quack.amd.gemm_gfx950_nn / _tn.

Per-shape config selection works in two modes:

  1. **Heuristic** (default): picks a config based on shape-divisibility
     rules — e.g., ``(128, 128, 64, 1, 4)`` if M%128 and N%128 else
     ``(128, 256, 64, 1, 4)``. Fast — no benchmarking. The rules are
     hand-tuned from the per-kernel sweep results.

  2. **Autotune**: enumerate a small candidate space, bench each on the
     actual shape, cache the winner in memory. First call for a new
     shape runs the bench (10-50ms overhead); subsequent calls use the
     cache with zero overhead. Opt-in via ``autotune_nn()`` /
     ``autotune_tn()`` or ``set_autotune(True)`` to tune on first call.

Cache persistence (optional): if ``QUACK_AMD_TUNE_CACHE`` env var points
to a writable path, the cache is loaded at import time and written on
each new entry, so repeated runs reuse previous results.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch


# Candidate config tuple: (TILE_M, TILE_N, TILE_K, BLOCK_M_WARPS, BLOCK_N_WARPS)
Config = Tuple[int, int, int, int, int]
# Cache key: (kernel_name, dtype_str, M, K, N)
CacheKey = Tuple[str, str, int, int, int]


_CACHE: Dict[CacheKey, Config] = {}
_AUTOTUNE_FIRST_CALL = False  # when True, tune on first call for new shape
_CACHE_FILE = os.environ.get("QUACK_AMD_TUNE_CACHE", "").strip() or None


def _load_cache_from_disk():
    """Best-effort load of the persistent cache at import time."""
    if _CACHE_FILE is None:
        return
    try:
        with open(_CACHE_FILE) as f:
            data = json.load(f)
        for entry in data.get("entries", []):
            key = tuple(entry["key"])
            config = tuple(entry["config"])
            if len(key) == 5 and len(config) == 5:
                _CACHE[key] = config
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass


_load_cache_from_disk()


def _save_cache_to_disk():
    """Best-effort save. Swallows errors — cache is a perf aid, not correctness."""
    if _CACHE_FILE is None:
        return
    try:
        entries = [
            {"key": list(key), "config": list(config)}
            for key, config in _CACHE.items()
        ]
        tmp = f"{_CACHE_FILE}.tmp"
        with open(tmp, "w") as f:
            json.dump({"entries": entries}, f, indent=2)
        os.replace(tmp, _CACHE_FILE)
    except (OSError, TypeError):
        pass


def set_autotune(enable: bool):
    """Toggle first-call autotune for all shapes.

    When enabled, the first call to ``gemm_nn`` / ``gemm_tn`` for a new
    (dtype, M, K, N) tuple runs the autotune search (10-50ms) and caches
    the winning config. Subsequent calls use the cache directly with no
    overhead. When disabled (default), a fast heuristic picks a
    known-good config with no benchmarking.
    """
    global _AUTOTUNE_FIRST_CALL
    _AUTOTUNE_FIRST_CALL = bool(enable)


def get_autotune() -> bool:
    return _AUTOTUNE_FIRST_CALL


def get_cached_config(key: CacheKey) -> Optional[Config]:
    return _CACHE.get(key)


def set_cached_config(key: CacheKey, config: Config):
    _CACHE[key] = config
    _save_cache_to_disk()


def clear_cache():
    _CACHE.clear()
    _save_cache_to_disk()


# ---------------------------------------------------------------------------
# Bench helper
# ---------------------------------------------------------------------------


def bench_config(
    launch_fn: Callable,   # callable that does one kernel launch
    warmup: int = 5,
    iters: int = 20,
    timeout_s: float = 2.0,
) -> float:
    """Time ``launch_fn()`` via CUDA events. Returns min-of-iters wall time (s).

    Caller responsible for pre-compiling the kernel (first call after the
    launcher's LRU cache is populated) so warmup captures steady-state.
    """
    t_start = time.time()
    try:
        for _ in range(warmup):
            if time.time() - t_start > timeout_s:
                break
            launch_fn()
        torch.cuda.synchronize()
        evs = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(iters)
        ]
        for s, e in evs:
            s.record()
            launch_fn()
            e.record()
        torch.cuda.synchronize()
        times = sorted(s.elapsed_time(e) * 1e-3 for s, e in evs)
        return times[0]
    except Exception:
        return float("inf")


def search_best_config(
    kernel_name: str,
    dtype_str: str,
    M: int, K: int, N: int,
    candidates: List[Config],
    launch_factory: Callable[[Config], Callable],
    verbose: bool = False,
) -> Tuple[Config, float]:
    """Bench all candidates, return (best_config, best_time_s).

    ``launch_factory(config)`` returns a 0-arg callable that runs one
    kernel launch at the given config. This lets us try all configs
    against the SAME tensors (callable closed over A, B, out).

    If every candidate fails to compile or launch, returns the first
    candidate with time=inf — callers should handle this.
    """
    best_cfg, best_t = candidates[0], float("inf")
    for cfg in candidates:
        try:
            launch = launch_factory(cfg)
            t = bench_config(launch)
            if verbose:
                print(f"  {kernel_name} {dtype_str} {M}×{N}×{K} {cfg}: "
                      f"{(2*M*N*K/t/1e12 if t>0 else 0):.1f} TF/s ({t*1e6:.1f} μs)")
            if t < best_t:
                best_cfg, best_t = cfg, t
        except (AssertionError, RuntimeError) as e:
            if verbose:
                print(f"  {kernel_name} {dtype_str} {M}×{N}×{K} {cfg}: FAIL ({str(e)[:80]})")
            continue
    return best_cfg, best_t
