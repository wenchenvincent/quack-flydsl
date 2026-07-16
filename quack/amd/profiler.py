# Copyright (c) 2026, AMD.

"""Reusable GPU benchmarking helpers for the AMD port.

Consolidates the CUDA-event min-of-iters timing that was copy-pasted across
``tests/amd/bench_vs_triton.py`` and ``quack/amd/_gemm_tune.py`` into one
canonical primitive. On ROCm ``torch.cuda.Event`` maps to HIP events, so this
times actual GPU execution (not host enqueue).

Public API:
    benchmark(fn, warmup, iters, return_mode) -> seconds
    benchmark_tflops(fn, flops, ...) -> TFLOP/s
    Timer() context manager -> .seconds
"""

import time
from typing import Callable

import torch


def benchmark(
    fn: Callable[[], object],
    warmup: int = 15,
    iters: int = 50,
    return_mode: str = "min",
) -> float:
    """Time ``fn`` on the GPU and return seconds per call.

    Records ``iters`` CUDA-event pairs after ``warmup`` untimed calls.
    ``return_mode`` ∈ {"min", "median", "mean"} reduces the per-iter samples —
    "min" (default) is the least noise-contaminated estimate of kernel time on
    a quiet GPU; "median" is more robust on a shared/contended box.
    """
    assert return_mode in ("min", "median", "mean"), return_mode
    assert iters >= 1 and warmup >= 0
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    time.sleep(0.1)  # let the GPU settle after warmup before timing
    evs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for s, e in evs:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    samples = sorted(s.elapsed_time(e) * 1e-3 for s, e in evs)  # ms -> s
    if return_mode == "min":
        return samples[0]
    if return_mode == "mean":
        return sum(samples) / len(samples)
    return samples[len(samples) // 2]  # median


def benchmark_tflops(
    fn: Callable[[], object],
    flops: float,
    warmup: int = 15,
    iters: int = 50,
    return_mode: str = "min",
) -> float:
    """Benchmark ``fn`` and report throughput in TFLOP/s given the op's ``flops``
    (e.g. ``2 * M * N * K`` for a GEMM)."""
    t = benchmark(fn, warmup=warmup, iters=iters, return_mode=return_mode)
    return flops / t / 1e12 if t > 0 else float("inf")


class Timer:
    """Context manager timing GPU work between enter and exit (seconds).

    Synchronizes on both ends, so it measures wall time including the enclosed
    kernels' execution:

        with Timer() as t:
            y = gemm(a, b)
        print(t.seconds)
    """

    def __enter__(self):
        torch.cuda.synchronize()
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self

    def __exit__(self, *exc):
        self._end.record()
        torch.cuda.synchronize()
        self.seconds = self._start.elapsed_time(self._end) * 1e-3
        return False


__all__ = ["benchmark", "benchmark_tflops", "Timer"]
