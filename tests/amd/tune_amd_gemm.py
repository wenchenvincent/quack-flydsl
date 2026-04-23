# Copyright (c) 2026, AMD.

"""Sweep NN/TN gemm autotune across a shape set, persist results.

Runs ``autotune_nn(A, B)`` / ``autotune_tn(A, B)`` for each (dtype, shape)
combination in the default sweep list (MLP-fwd/dx/dw shapes from the
Phase 1 profile). Prints the winner config + speedup vs heuristic for
each, and writes the cache to a JSON file for reuse.

Usage:
    # Cache to default location (~/.quack_amd_gemm_tune.json):
    PYTHONPATH=/workspace/quack python -m tests.amd.tune_amd_gemm

    # Custom cache path + quick sweep (only medium shapes):
    QUACK_AMD_TUNE_CACHE=/tmp/gemm_tune.json \\
        PYTHONPATH=/workspace/quack python -m tests.amd.tune_amd_gemm --quick

    # Verbose (show per-candidate timings):
    PYTHONPATH=/workspace/quack python -m tests.amd.tune_amd_gemm --verbose

The resulting cache is auto-loaded at import time in any subsequent
Python session where ``QUACK_AMD_TUNE_CACHE`` points to the same file.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from quack.amd import _gemm_tune
from quack.amd.gemm_gfx950_nn import (
    _heuristic_config,
    _autotune_nn_impl,
    _NN_CANDIDATES,
)
from quack.amd.gemm_gfx950_tn import (
    _heuristic_config_tn,
    _autotune_tn_impl,
    _TN_CANDIDATES,
)


# Default shape sweep — MLP-fwd / dx / dw at training-typical (bs, hidden).
# (bs, hidden, out_f) tuples; shapes derived from the Phase 1 profile.
_DEFAULT_SWEEP = [
    # (bs, hidden)
    (2048, 1024),
    (4096, 2048),
    (8192, 4096),
    (16384, 4096),
    (8192, 8192),
    (32768, 8192),
]


def _nn_shapes(bs: int, hidden: int):
    """Yield (M, K, N) for the NN bwd-DX shape of an MLP: dx = dy @ W."""
    out_f = 4 * hidden
    yield ("mlp_dx", bs, out_f, hidden)  # M=bs, K=4h, N=h


def _tn_shapes(bs: int, hidden: int):
    """Yield (K, M, N) for the TN bwd-DW shapes of an MLP: dW = dy.T @ x."""
    out_f = 4 * hidden
    # dW1 = dy1.T @ x : (bs, 4h).T @ (bs, h) -> (4h, h)   K=bs, M=4h, N=h
    yield ("mlp_dw1", bs, out_f, hidden)
    # dW2 = dy2.T @ h : (bs, h).T @ (bs, 4h) -> (h, 4h)   K=bs, M=h, N=4h
    yield ("mlp_dw2", bs, hidden, out_f)


def _bench_one_config(k_fn_factory, config):
    """Bench a specific config via its factory, return wall-time seconds."""
    try:
        launch = k_fn_factory(config)
        return _gemm_tune.bench_config(launch)
    except Exception:
        return float("inf")


def _sweep_nn(bs_hidden_list, dtype_str, dtype, verbose: bool):
    print(f"=== NN ({dtype_str}) ===")
    print(f"{'role':<8} {'shape':<24} {'heuristic':>18} {'autotune':>18} "
          f"{'chosen config':<32} {'speedup':>8}")
    total_tuned = 0
    total_kept_heuristic = 0
    for bs, hidden in bs_hidden_list:
        for role, M, K, N in _nn_shapes(bs, hidden):
            a = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
            b = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
            out = torch.empty(M, N, device="cuda", dtype=dtype)
            # Bench heuristic
            from quack.amd.gemm_gfx950_nn import _compile_nn_kernel
            h_cfg = _heuristic_config(M, N)
            htm, htn, htk, hbmw, hbnw = h_cfg
            h_fn = _compile_nn_kernel(
                dtype_str, K, N, TILE_M=htm, TILE_N=htn, TILE_K=htk,
                BLOCK_M_WARPS=hbmw, BLOCK_N_WARPS=hbnw, _m_hint=M,
            )
            t_h = _gemm_tune.bench_config(lambda: h_fn(out, a, b, M))
            # Autotune
            best_cfg, t_best = _autotune_nn_impl(
                dtype_str, M, K, N, a, b, out, verbose=verbose,
            )
            _gemm_tune.set_cached_config(("nn", dtype_str, M, K, N), best_cfg)
            flops = 2 * M * N * K / 1e12
            speedup = t_h / t_best if t_best > 0 else 0.0
            if best_cfg != h_cfg:
                total_tuned += 1
            else:
                total_kept_heuristic += 1
            print(f"{role:<8} {M}×{N}×{K:<10}"
                  f" {flops/t_h:>8.1f} TF/s {t_h*1e6:>5.0f}μs"
                  f" {flops/t_best:>8.1f} TF/s {t_best*1e6:>5.0f}μs"
                  f" {str(best_cfg):<32} {speedup:>6.2f}x")
    print(f"→ tuned {total_tuned} shapes, kept heuristic on {total_kept_heuristic}")


def _sweep_tn(bs_hidden_list, dtype_str, dtype, verbose: bool):
    print(f"\n=== TN ({dtype_str}) ===")
    print(f"{'role':<8} {'shape':<24} {'heuristic':>18} {'autotune':>18} "
          f"{'chosen config':<32} {'speedup':>8}")
    total_tuned = 0
    total_kept_heuristic = 0
    for bs, hidden in bs_hidden_list:
        for role, K, M, N in _tn_shapes(bs, hidden):
            a = torch.randn(K, M, device="cuda", dtype=dtype) * 0.1
            b = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
            out = torch.empty(M, N, device="cuda", dtype=dtype)
            from quack.amd.gemm_gfx950_tn import _compile_tn_kernel
            h_cfg = _heuristic_config_tn(M, N)
            htm, htn, htk, hbmw, hbnw = h_cfg
            h_fn = _compile_tn_kernel(
                dtype_str, K, M, N, TILE_M=htm, TILE_N=htn, TILE_K=htk,
                BLOCK_M_WARPS=hbmw, BLOCK_N_WARPS=hbnw,
            )
            t_h = _gemm_tune.bench_config(lambda: h_fn(out, a, b))
            best_cfg, t_best = _autotune_tn_impl(
                dtype_str, M, K, N, a, b, out, verbose=verbose,
            )
            _gemm_tune.set_cached_config(("tn", dtype_str, M, K, N), best_cfg)
            flops = 2 * M * N * K / 1e12
            speedup = t_h / t_best if t_best > 0 else 0.0
            if best_cfg != h_cfg:
                total_tuned += 1
            else:
                total_kept_heuristic += 1
            print(f"{role:<8} K={K} {M}×{N:<16}"
                  f" {flops/t_h:>8.1f} TF/s {t_h*1e6:>5.0f}μs"
                  f" {flops/t_best:>8.1f} TF/s {t_best*1e6:>5.0f}μs"
                  f" {str(best_cfg):<32} {speedup:>6.2f}x")
    print(f"→ tuned {total_tuned} shapes, kept heuristic on {total_kept_heuristic}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Only sweep medium-size shapes (bs=4096, 8192)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-candidate timings during autotune")
    parser.add_argument("--dtype", default="both",
                        choices=["bf16", "f16", "both"])
    args = parser.parse_args()

    if args.quick:
        bs_hidden = [(4096, 2048), (8192, 4096)]
    else:
        bs_hidden = _DEFAULT_SWEEP

    dtypes = []
    if args.dtype in ("both", "bf16"):
        dtypes.append(("bf16", torch.bfloat16))
    if args.dtype in ("both", "f16"):
        dtypes.append(("f16", torch.float16))

    if _gemm_tune._CACHE_FILE:
        print(f"Cache file: {_gemm_tune._CACHE_FILE}")
    else:
        print("Cache file: (in-memory only — set QUACK_AMD_TUNE_CACHE env var to persist)")

    t_start = time.time()
    for dtype_str, dtype in dtypes:
        _sweep_nn(bs_hidden, dtype_str, dtype, args.verbose)
        _sweep_tn(bs_hidden, dtype_str, dtype, args.verbose)
    elapsed = time.time() - t_start
    print(f"\nTotal sweep time: {elapsed:.1f}s")
    print(f"Cache size: {len(_gemm_tune._CACHE)} entries")


if __name__ == "__main__":
    main()
