#!/usr/bin/env python3
"""AMD RMSNorm / LayerNorm benchmark — fwd + bwd across shapes.

Compares ``quack.amd.{rmsnorm_fwd, rmsnorm_bwd, layernorm_fwd, layernorm_bwd}``
against a torch-compiled reference path. Torch's native RMSNorm lowering
on AMD goes through plain eager ops + hipBLASLt / vector intrinsics; the
``torch.compile`` reference is the relevant baseline for fused kernels.

Usage:
    python benchmarks/benchmark_rmsnorm_amd.py
    python benchmarks/benchmark_rmsnorm_amd.py --M 8192 --N 4096 --dtype bfloat16
    python benchmarks/benchmark_rmsnorm_amd.py --layernorm
    python benchmarks/benchmark_rmsnorm_amd.py --bwd-only
"""

import argparse
import time

import torch
import torch.nn as nn
from triton.testing import do_bench

from quack.amd.rmsnorm import (
    rmsnorm_fwd, rmsnorm_bwd, layernorm_fwd, layernorm_bwd,
)


def _gbps(nbytes, ms):
    return nbytes / (ms * 1e6)


def _bench(fn, warmup=5, rep=30):
    fn()
    torch.cuda.synchronize()
    time.sleep(0.1)
    return do_bench(fn, warmup=warmup, rep=rep, return_mode="mean")


def _torch_rmsnorm(x, w, eps):
    x_f = x.float()
    rstd = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f * rstd * w.float()).to(x.dtype)


def _torch_layernorm(x, w, b, eps):
    x_f = x.float()
    mean = x_f.mean(-1, keepdim=True)
    var = x_f.var(-1, keepdim=True, unbiased=False)
    rstd = torch.rsqrt(var + eps)
    y = (x_f - mean) * rstd * w.float()
    if b is not None:
        y = y + b.float()
    return y.to(x.dtype)


def benchmark_fwd(M, N, dtype, is_layernorm, has_bias, repeats, eps=1e-6):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    b = torch.randn(N, device="cuda", dtype=dtype) if has_bias else None
    # Fwd reads x + w (+ b) and writes y. Roughly 2x/dtype + weight + out.
    nbytes = (2 * M * N + N + (N if has_bias else 0)) * dtype.itemsize

    if is_layernorm:
        quack_fn = lambda: layernorm_fwd(x, w, bias=b)
        torch_ref = torch.compile(lambda: _torch_layernorm(x, w, b, eps))
    else:
        quack_fn = lambda: rmsnorm_fwd(x, w)
        torch_ref = torch.compile(lambda: _torch_rmsnorm(x, w, eps))

    torch_ref()   # compile warmup
    torch_ref()

    ms_q = _bench(quack_fn, rep=repeats)
    ms_t = _bench(torch_ref, rep=repeats)
    print(
        f"  fwd M={M:>6} N={N:>5}  "
        f"quack={ms_q*1000:6.2f} us  {_gbps(nbytes, ms_q):5.0f} GB/s   "
        f"torch.compile={ms_t*1000:6.2f} us   speedup={ms_t/ms_q:4.2f}×"
    )


def benchmark_bwd(M, N, dtype, is_layernorm, has_bias, repeats, eps=1e-6):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    b = torch.randn(N, device="cuda", dtype=dtype) if has_bias else None
    dy = torch.randn(M, N, device="cuda", dtype=dtype)
    # Read rstd once up front so bwd timing isolates just the dx/dw work.
    if is_layernorm:
        _, rstd, mean, _ = layernorm_fwd(x, w, bias=b, store_stats=True)
        quack_fn = lambda: layernorm_bwd(x, w, dy, rstd, mean, bias=b)
    else:
        _, rstd, _ = rmsnorm_fwd(x, w, store_rstd=True)
        quack_fn = lambda: rmsnorm_bwd(x, w, dy, rstd)

    # Torch reference via autograd.
    x_t = x.clone().requires_grad_(True)
    w_t = w.clone().requires_grad_(True)
    b_t = b.clone().requires_grad_(True) if b is not None else None
    if is_layernorm:
        def torch_ref():
            y = _torch_layernorm(x_t, w_t, b_t, eps)
            return torch.autograd.grad(y, [x_t, w_t] + ([b_t] if b_t is not None else []),
                                       dy, retain_graph=True)
    else:
        def torch_ref():
            y = _torch_rmsnorm(x_t, w_t, eps)
            return torch.autograd.grad(y, [x_t, w_t], dy, retain_graph=True)

    torch_ref()

    # Bwd reads x, dy, rstd, w, (mean) and writes dx, dw, (db).
    extra = 1 if is_layernorm else 0
    nbytes = (3 * M * N + (M * (1 + extra)) + (2 + extra) * N) * dtype.itemsize

    ms_q = _bench(quack_fn, rep=repeats)
    ms_t = _bench(torch_ref, rep=repeats)
    tag = "LN-bwd" if is_layernorm else "RMS-bwd"
    print(
        f"  {tag} M={M:>6} N={N:>5}  "
        f"quack={ms_q*1000:6.2f} us  {_gbps(nbytes, ms_q):5.0f} GB/s   "
        f"torch={ms_t*1000:6.2f} us   speedup={ms_t/ms_q:4.2f}×"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=None, help="Override M (else sweep).")
    p.add_argument("--N", type=int, default=None, help="Override N (else sweep).")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--layernorm", action="store_true", help="Bench LayerNorm instead of RMSNorm.")
    p.add_argument("--bias", action="store_true", help="LayerNorm with bias (ignored for RMSNorm).")
    p.add_argument("--fwd-only", action="store_true")
    p.add_argument("--bwd-only", action="store_true")
    p.add_argument("--repeats", type=int, default=30)
    args = p.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "unknown")
    print(f"device: {torch.cuda.get_device_name(0)}  arch: {arch}  dtype: {dtype}")
    print(f"norm: {'LayerNorm' if args.layernorm else 'RMSNorm'}"
          + (f"  (+bias)" if args.layernorm and args.bias else ""))
    print()

    if args.M and args.N:
        shapes = [(args.M, args.N)]
    else:
        # Representative transformer token counts × hidden dims.
        shapes = [
            (1024, 1024), (1024, 4096), (1024, 8192),
            (4096, 1024), (4096, 4096), (4096, 8192),
            (16384, 4096), (32768, 4096),
        ]

    if not args.bwd_only:
        print("Forward:")
        for M, N in shapes:
            benchmark_fwd(M, N, dtype, args.layernorm, args.layernorm and args.bias, args.repeats)
        print()

    if not args.fwd_only:
        print("Backward:")
        for M, N in shapes:
            benchmark_bwd(M, N, dtype, args.layernorm, args.layernorm and args.bias, args.repeats)


if __name__ == "__main__":
    main()
