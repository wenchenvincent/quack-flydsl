# Copyright (c) 2026, AMD.

"""W3 benchmark harness — track splitk vs. hipBLASLt over rounds of tuning.

Usage:
    PYTHONPATH=/workspace/quack python -m tests.amd.bench_splitk_profile

Reports, per reference shape:
  - splitk time (best of N iters)
  - hipBLASLt torch.mm time (same)
  - ratio (splitk / hipBLASLt) — target ≤ 1.02×

Shapes cover the regimes where fused-dact would net-save the most:
  A = M=4096, K=8192, N=8192  — canonical LLM FFN down-proj
  B = M=8192, K=8192, N=8192  — square, most CUs occupied
  C = M=2048, K=16384, N=2048 — skinny-K, splitk-favored
"""

import argparse
import json
import time
from pathlib import Path

import torch

from quack.amd.gemm_gfx950_splitk import (
    gemm_splitk,
    _compile_hgemm_kernel,
    _DTYPE2STR,
    shuffle_b,
    _default_kwargs,
    _set_default_kwargs_override,
)


REF_SHAPES = [
    ("A_4k8k8k",  4096,  8192,  8192),
    ("B_8k8k8k",  8192,  8192,  8192),
    ("C_2k16k2k", 2048, 16384,  2048),
]


def _bench(fn, warmup=3, iters=20):
    # Warmup.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # Timed runs — measure minimum to reduce noise.
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


def _bench_splitk(M, N, K, dtype=torch.bfloat16, config_override=None):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=dtype) * 0.3
    b = torch.randn(N, K, device="cuda", dtype=dtype) * 0.3
    b_shuffled = shuffle_b(b)
    # Apply the override so _default_kwargs (used internally by the launcher)
    # picks up our tile/split/waves choices.
    _set_default_kwargs_override(config_override)
    try:
        kwargs = dict(_default_kwargs(M, N, K))
        if kwargs["B_PRE_SHUFFLE"]:
            b_arg = b_shuffled
        else:
            b_arg = b
        out = torch.empty(M, N, device="cuda", dtype=dtype)
        gemm_splitk(a, b_arg, out=out, shuffled=kwargs["B_PRE_SHUFFLE"])
        torch.cuda.synchronize()

        def run():
            gemm_splitk(a, b_arg, out=out, shuffled=kwargs["B_PRE_SHUFFLE"])

        t = _bench(run)
    finally:
        _set_default_kwargs_override(None)
    return t


def _bench_hipblaslt(M, N, K, dtype=torch.bfloat16):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=dtype) * 0.3
    b = torch.randn(N, K, device="cuda", dtype=dtype) * 0.3
    # torch.mm(a, b.T) → hipBLASLt on ROCm.
    out = torch.empty(M, N, device="cuda", dtype=dtype)

    def run():
        torch.matmul(a, b.t(), out=out)

    return _bench(run)


def _flops(M, N, K):
    return 2.0 * M * N * K


def run_baseline():
    results = {}
    print(f"{'shape':<14} {'splitk ms':>10} {'hipBLAS ms':>10} {'ratio':>8} {'splitk TFLOP/s':>16}")
    print("-" * 62)
    for name, M, K, N in REF_SHAPES:
        t_splitk = _bench_splitk(M, N, K)
        t_hip = _bench_hipblaslt(M, N, K)
        ratio = t_splitk / t_hip
        tflops = _flops(M, N, K) / t_splitk / 1e12
        results[name] = dict(M=M, N=N, K=K, splitk_ms=t_splitk*1e3, hip_ms=t_hip*1e3,
                             ratio=ratio, splitk_tflops=tflops)
        print(f"{name:<14} {t_splitk*1e3:>10.3f} {t_hip*1e3:>10.3f} {ratio:>8.3f} {tflops:>16.1f}")
    return results


def sweep_tile_configs():
    """Round 1: sweep tile_m × tile_n × tile_k, report best per shape."""
    configs = [
        # (TILE_M, TILE_N, TILE_K)
        (128, 128,  64),
        (128, 128, 128),
        (128, 256,  64),   # current default
        (128, 256, 128),
        (256, 128,  64),
        (256, 128, 128),
        (256, 256,  64),
        (256, 256, 128),
    ]
    print(f"\n{'shape':<14} {'config':<20} {'ms':>8} {'ratio':>8}")
    print("-" * 52)
    per_shape_best = {}
    for name, M, K, N in REF_SHAPES:
        t_hip = _bench_hipblaslt(M, N, K)
        best = None
        for tm, tn, tk in configs:
            if M < tm:
                continue
            if N % tn != 0:
                continue
            if K % tk != 0:
                continue
            try:
                t = _bench_splitk(M, N, K, config_override=dict(TILE_M=tm, TILE_N=tn, TILE_K=tk))
            except (AssertionError, Exception) as e:
                continue
            ratio = t / t_hip
            tag = f"{tm}x{tn}x{tk}"
            print(f"{name:<14} {tag:<20} {t*1e3:>8.3f} {ratio:>8.3f}")
            if best is None or t < best[2]:
                best = ((tm, tn, tk), tag, t, ratio)
        per_shape_best[name] = best
        print(f"{name:<14} BEST {best[1]:<15} {best[2]*1e3:>8.3f} {best[3]:>8.3f}")
    return per_shape_best


def sweep_b_routing():
    """Round 3: try B_PRE_SHUFFLE=True/False × B_TO_LDS=False (B_PRE_SHUFFLE and
    B_TO_LDS are mutually exclusive; when PRE_SHUFFLE is set, TO_LDS is False
    in the builder). Compare preshuffle vs. direct HBM load.

    The preshuffle path reorders B at init time into the exact layout the
    MFMA B fragment wants; direct HBM is a vanilla load. At skinny-M or
    when B isn't reused much, direct can win.
    """
    print(f"\n{'shape':<14} {'config':<20} {'ms':>8} {'ratio':>8}")
    print("-" * 52)
    per_shape_best = {}
    for name, M, K, N in REF_SHAPES:
        t_hip = _bench_hipblaslt(M, N, K)
        best = None
        for preshuf in [True, False]:
            override = {"B_PRE_SHUFFLE": preshuf}
            try:
                t = _bench_splitk(M, N, K, config_override=override)
            except Exception as e:
                print(f"{name:<14} preshuf={preshuf:<6} FAIL {e}")
                continue
            ratio = t / t_hip
            tag = f"preshuf={preshuf}"
            print(f"{name:<14} {tag:<20} {t*1e3:>8.3f} {ratio:>8.3f}")
            if best is None or t < best[1]:
                best = (tag, t, ratio)
        per_shape_best[name] = best
        print(f"{name:<14} BEST {best[0]} {best[1]*1e3:.3f}ms ratio={best[2]:.3f}")
    return per_shape_best


def sweep_waves_per_eu():
    """Round 2: sweep waves_per_eu ∈ {None, 1, 2, 3, 4} on top of the
    shape's default config."""
    wpe_vals = [None, 1, 2, 3, 4]
    print(f"\n{'shape':<14} {'wpe':<6} {'ms':>8} {'ratio':>8}")
    print("-" * 38)
    per_shape_best = {}
    for name, M, K, N in REF_SHAPES:
        t_hip = _bench_hipblaslt(M, N, K)
        best = None
        for wpe in wpe_vals:
            override = None if wpe is None else {"waves_per_eu": wpe}
            try:
                t = _bench_splitk(M, N, K, config_override=override)
            except Exception as e:
                print(f"{name:<14} {str(wpe):<6} FAIL {type(e).__name__}: {e}")
                continue
            ratio = t / t_hip
            print(f"{name:<14} {str(wpe):<6} {t*1e3:>8.3f} {ratio:>8.3f}")
            if best is None or t < best[1]:
                best = (wpe, t, ratio)
        per_shape_best[name] = best
        print(f"{name:<14} BEST wpe={best[0]} {best[1]*1e3:.3f}ms ratio={best[2]:.3f}")
    return per_shape_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "tiles", "waves", "b_routing"], default="baseline")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if args.mode == "baseline":
        results = run_baseline()
    elif args.mode == "tiles":
        results = sweep_tile_configs()
    elif args.mode == "waves":
        results = sweep_waves_per_eu()
    elif args.mode == "b_routing":
        results = sweep_b_routing()
    if args.output:
        args.output.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
