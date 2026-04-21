#!/usr/bin/env python3
"""AMD top-k benchmark — single-wave bitonic-sort kernel vs torch.topk.

``quack.amd.topk_kernel.topk_mfma`` covers N ∈ {8, 16, 32, 64} and k ≤ N
(the MVP scope). For larger N falls back to torch.topk via
``quack.amd.topk.topk_fwd`` wrapper.

Usage:
    python benchmarks/benchmark_topk_amd.py                # defaults
    python benchmarks/benchmark_topk_amd.py --M 8192 --N 64 --k 16
    python benchmarks/benchmark_topk_amd.py --sweep
"""

import argparse
import time

import torch
from triton.testing import do_bench

from quack.amd.topk_kernel import topk_mfma


def _bench(fn, warmup=10, rep=100):
    fn()
    torch.cuda.synchronize()
    time.sleep(0.1)
    return do_bench(fn, warmup=warmup, rep=rep, return_mode="mean")


def _mem_bw(M, N, k, elem_bytes, ms):
    """Read M*N values + write M*k values + M*k indices (i32)."""
    read_bytes = M * N * elem_bytes
    write_bytes = M * k * (elem_bytes + 4)   # values + i32 indices
    return (read_bytes + write_bytes) / (ms * 1e6)


def bench_one(M, N, k, repeats):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)

    quack_fn = lambda: topk_mfma(x, k=k)
    torch_fn = lambda: torch.topk(x, k, dim=-1)

    ms_q = _bench(quack_fn, rep=repeats)
    ms_t = _bench(torch_fn, rep=repeats)

    gbps_q = _mem_bw(M, N, k, 4, ms_q)
    gbps_t = _mem_bw(M, N, k, 4, ms_t)
    print(
        f"  M={M:>6} N={N:>3} k={k:>2}   "
        f"quack={ms_q*1000:6.2f} us  {gbps_q:5.0f} GB/s   "
        f"torch.topk={ms_t*1000:6.2f} us  {gbps_t:5.0f} GB/s   "
        f"speedup={ms_t/ms_q:4.2f}×"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=8192)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--sweep", action="store_true",
                   help="Sweep (M, N, k) in the single-wave scope.")
    p.add_argument("--repeats", type=int, default=100)
    args = p.parse_args()

    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "unknown")
    print(f"device: {torch.cuda.get_device_name(0)}  arch: {arch}")
    print()

    if args.sweep:
        shapes = [
            (32768, 8, 4),
            (32768, 16, 4), (32768, 16, 8),
            (32768, 32, 4), (32768, 32, 8), (32768, 32, 16),
            (32768, 64, 1), (32768, 64, 8), (32768, 64, 16), (32768, 64, 32), (32768, 64, 64),
            (8192, 64, 8),  (1024, 64, 8),
        ]
        for M, N, k in shapes:
            bench_one(M, N, k, args.repeats)
    else:
        bench_one(args.M, args.N, args.k, args.repeats)


if __name__ == "__main__":
    main()
