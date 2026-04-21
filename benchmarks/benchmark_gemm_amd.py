#!/usr/bin/env python3
"""AMD counterpart to ``benchmark_gemm_autotuned.py`` — sweeps GEMM
shapes and compares each ``quack.amd`` kernel variant against
hipBLASLt (reached via ``torch.nn.functional.linear``).

Usage:
    # Default 8192x8192x8192 bf16
    python benchmarks/benchmark_gemm_amd.py

    # Custom size
    python benchmarks/benchmark_gemm_amd.py --M 4096 --N 4096 --K 4096

    # Compare every kernel variant side-by-side
    python benchmarks/benchmark_gemm_amd.py --variants all

    # Print the kernel the auto-dispatcher picks for each shape
    python benchmarks/benchmark_gemm_amd.py --show-dispatch

    # Also run gemm_act (fused activation in the MFMA epilogue)
    python benchmarks/benchmark_gemm_amd.py --activation silu

The reference baseline is ``torch.nn.functional.linear`` which routes
through hipBLASLt on ROCm — the same way cuBLAS is the NV baseline.
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from quack.amd.gemm import gemm, gemm_act
from quack.amd.gemm_autotune import (
    KERNEL_REGISTRY,
    select_best_kernel,
    get_kernel,
)


def tflops(flops, ms):
    return flops / (ms * 1e9)


def _bench(fn, repeats=30):
    """Warm-start + do_bench wrapper. Returns ms."""
    fn()
    torch.cuda.synchronize()
    time.sleep(0.2)
    return do_bench(fn, warmup=5, rep=repeats)


def _hipblaslt_reference(a, b, dtype):
    """torch.nn.functional.linear — routes to hipBLASLt on AMD."""
    w = b.T.contiguous()
    return lambda: F.linear(a, w)


def benchmark_gemm_plain(M, N, K, dtype, variants, repeats):
    """Plain ``A @ B`` benchmark across the selected kernel variants."""
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype) / math.sqrt(K)
    nflops = 2 * M * N * K
    nbytes = (a.numel() + b.numel() + M * N) * dtype.itemsize

    print(f"GEMM: ({M},{K}) × ({K},{N})  dtype={dtype}")

    # hipBLASLt baseline
    ref = _hipblaslt_reference(a, b, dtype)
    ms_ref = _bench(ref, repeats)
    tf_ref = tflops(nflops, ms_ref)
    gbps_ref = nbytes / (ms_ref * 1e6)
    print(f"  hipBLASLt (torch):   {ms_ref:7.3f} ms   {tf_ref:6.1f} TFLOPS   {gbps_ref:5.0f} GB/s")

    # quack.amd top-level dispatch (auto-selects kernel)
    ms = _bench(lambda: gemm(a, b, out_dtype=torch.float32), repeats)
    tf = tflops(nflops, ms)
    print(f"  quack.amd (auto):    {ms:7.3f} ms   {tf:6.1f} TFLOPS   {ms_ref/ms:5.2f}× vs hipBLASLt")

    # Per-variant sweep
    for name in variants:
        try:
            fn = get_kernel(name)
        except KeyError:
            continue
        # Only alignment-compatible variants should run.
        if name == "mfma_16x16" and (M % 16 or N % 16 or K % 16):
            continue
        if name.startswith(("tiled_32x32", "lds_32x32", "lds_swz_32x32")) and (
            M % 32 or N % 32 or K % 16
        ):
            continue
        if name == "4wave_64x64" and (M % 64 or N % 64 or K % 16):
            continue

        # MFMA-16x16 wants the full gemm() signature.
        if name == "mfma_16x16":
            call = lambda: fn(a, b, out_dtype=torch.float32)
        else:
            call = lambda: fn(a, b)
        try:
            ms_v = _bench(call, repeats)
        except Exception as e:
            print(f"  {name:<22}  SKIP ({type(e).__name__}: {e})")
            continue
        tf_v = tflops(nflops, ms_v)
        tag = " ⬅ auto" if name == select_best_kernel(M, N, K, dtype, plain=True) else ""
        print(f"  {name:<22} {ms_v:7.3f} ms   {tf_v:6.1f} TFLOPS   {ms_ref/ms_v:5.2f}×{tag}")

    return ms, tf


def benchmark_gemm_act(M, N, K, activation, dtype, repeats):
    """``act(A @ B)`` with optional activation in the kernel epilogue.

    For gated (swiglu/reglu/geglu): B is (K, 2N), output is (M, N).
    """
    is_gated = activation in ("swiglu", "reglu", "geglu", "glu")
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b_n = 2 * N if is_gated else N
    b = torch.randn(K, b_n, device="cuda", dtype=dtype) / math.sqrt(K)
    nflops = 2 * M * b_n * K

    print(f"GEMM + {activation}: ({M},{K}) × ({K},{b_n})")

    # hipBLASLt + torch activation baseline
    w = b.T.contiguous()
    if is_gated:
        act_torch = {
            "swiglu": F.silu,
            "reglu": F.relu,
            "geglu": lambda x: F.gelu(x, approximate="tanh"),
            "glu": torch.sigmoid,
        }[activation]
        def ref_fn():
            preact = F.linear(a, w)
            g = preact[..., :N]
            u = preact[..., N:]
            return act_torch(g) * u
    else:
        act_torch = {
            "relu": F.relu,
            "relu_sq": lambda x: F.relu(x) ** 2,
            "silu": F.silu,
            "gelu_tanh_approx": lambda x: F.gelu(x, approximate="tanh"),
        }[activation]
        ref_fn = lambda: act_torch(F.linear(a, w))

    ms_ref = _bench(ref_fn, repeats)
    tf_ref = tflops(nflops, ms_ref)
    print(f"  hipBLASLt + torch:   {ms_ref:7.3f} ms   {tf_ref:6.1f} TFLOPS")

    # quack.amd fused
    if is_gated:
        from quack.amd.gemm import gemm_gated
        call = lambda: gemm_gated(a, b, gate_type=activation, out_dtype=dtype)
    else:
        call = lambda: gemm_act(a, b, activation=activation, out_dtype=dtype)
    ms = _bench(call, repeats)
    tf = tflops(nflops, ms)
    print(f"  quack.amd fused:     {ms:7.3f} ms   {tf:6.1f} TFLOPS   {ms_ref/ms:5.2f}× vs hipBLASLt+torch")


def main():
    p = argparse.ArgumentParser(description="AMD GEMM benchmark")
    p.add_argument("--M", type=int, default=8192)
    p.add_argument("--N", type=int, default=8192)
    p.add_argument("--K", type=int, default=8192)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument(
        "--variants",
        default="auto",
        help="'auto' (dispatcher only), 'all' (every eligible variant), or comma-separated names.",
    )
    p.add_argument("--activation", default=None, help="Also run gemm_act with this activation.")
    p.add_argument("--show-dispatch", action="store_true", help="Print the dispatcher's pick + exit.")
    p.add_argument(
        "--transformer",
        action="store_true",
        help="Run the canonical transformer shape sweep (QKV proj / Attn out / FFN down / FFN up).",
    )
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--batch", type=int, default=8192)
    args = p.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    if args.variants == "auto":
        variants = []  # dispatcher only
    elif args.variants == "all":
        variants = list(KERNEL_REGISTRY.keys())
    else:
        variants = [v.strip() for v in args.variants.split(",")]

    if args.show_dispatch:
        for shape in [(args.M, args.N, args.K)]:
            m, n, k = shape
            pick = select_best_kernel(m, n, k, dtype, plain=True)
            print(f"  ({m},{n},{k}) {dtype}: {pick}")
        return

    # Print arch + hardware info
    arch = torch.cuda.get_device_properties(0).gcnArchName if hasattr(
        torch.cuda.get_device_properties(0), "gcnArchName"
    ) else "unknown"
    print(f"device: {torch.cuda.get_device_name(0)}  arch: {arch}")
    print()

    # --- 1. Plain GEMM ---
    print("=" * 70)
    benchmark_gemm_plain(args.M, args.N, args.K, dtype, variants, args.repeats)
    print()

    # --- 2. Activation variant ---
    if args.activation:
        print("=" * 70)
        benchmark_gemm_act(args.M, args.N, args.K, args.activation, dtype, args.repeats)
        print()

    # --- 3. Transformer shape sweep ---
    if args.transformer:
        print("=" * 70)
        print(f"Transformer shapes (batch={args.batch}, dim={args.dim})")
        print("=" * 70)
        head_dim = 128
        n_q = args.dim // head_dim
        n_kv = n_q // 4
        qkv_dim = (n_q + 2 * n_kv) * head_dim
        ffn = int(args.dim * 3.5)
        shapes = [
            ("QKV proj", args.batch, qkv_dim, args.dim),
            ("Attn out", args.batch, args.dim, args.dim),
            ("FFN down", args.batch, args.dim, ffn),
        ]
        for label, m, n, k in shapes:
            print(f"\n  {label}:")
            benchmark_gemm_plain(m, n, k, dtype, variants, args.repeats)
        print(f"\n  FFN up + SwiGLU:")
        benchmark_gemm_act(args.batch, ffn, args.dim, "swiglu", dtype, args.repeats)


if __name__ == "__main__":
    main()
