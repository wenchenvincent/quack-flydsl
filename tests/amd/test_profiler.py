# Copyright (c) 2026, AMD.

"""Reusable GPU benchmarking helpers (G14 profiler half)."""

import pytest
import torch

from quack.amd.profiler import benchmark, benchmark_tflops, Timer


def test_benchmark_positive_and_modes():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    fn = lambda: torch.mm(a, b)
    for mode in ("min", "median", "mean"):
        t = benchmark(fn, warmup=5, iters=20, return_mode=mode)
        assert t > 0, mode
    # min <= median <= mean-ish ordering (min is the smallest sample)
    tmin = benchmark(fn, warmup=5, iters=30, return_mode="min")
    tmed = benchmark(fn, warmup=5, iters=30, return_mode="median")
    assert tmin <= tmed * 1.5  # min never exceeds median (allow noise slack)


def test_benchmark_tflops_plausible():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    M = K = N = 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    tf = benchmark_tflops(lambda: torch.mm(a, b), 2 * M * N * K, warmup=10, iters=30)
    # sane window for a CDNA4 bf16 matmul: well above CPU-ish, below sparse peak.
    assert 20 < tf < 3000, f"implausible TFLOP/s {tf}"


def test_benchmark_monotonic_in_work():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    a = torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16)
    b1 = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
    b2 = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)  # 2x N
    t1 = benchmark(lambda: torch.mm(a, b1), warmup=10, iters=30)
    t2 = benchmark(lambda: torch.mm(a, b2), warmup=10, iters=30)
    assert t2 > t1 * 1.3, f"2x-work op should take longer: {t1} vs {t2}"


def test_timer_context():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    with Timer() as t:
        for _ in range(4):
            torch.mm(a, a)
    assert t.seconds > 0
