# Copyright (c) 2026, AMD.

"""Bench ``quack.amd.gemm_streamk_prod.gemm_streamk`` vs hipBLASLt.

Usage:
    PYTHONPATH=/workspace/quack python -m tests.amd.bench_streamk_prod

Reports, per shape:
  - streamk_prod (our new kernel) TFLOP/s + absolute time
  - gemm_persistent (static baseline) time
  - torch.matmul f16->f32 (hipBLASLt) time
  - ratio vs hipBLASLt (>1 = slower)

Skipping the old ``gemm_streamk`` demo — it's 100-600× slower than
hipBLASLt and makes the bench hang for several minutes.

Reference state (commit that landed this file): streamk_prod is
3-12× slower than hipBLASLt across a range of shapes.  The 16×16
MFMA tile used here means hundreds of thousands of atomic-fadd ops
per call at production shapes, which dominates runtime.  Closing
the gap further requires a 128×128 tile rewrite with 4-warp WGs
and coalesced atomic writes — deferred.
"""

import time
import torch

from quack.amd.gemm_streamk_prod import gemm_streamk
from quack.amd.gemm_persistent import gemm_f16_persistent


_SHAPES = [
    # (M, N, K)
    (128, 128, 128),
    (256, 256, 128),
    (512, 512, 256),
    (1024, 1024, 512),
    (128, 2048, 512),
    (2048, 1024, 512),
    (4096, 4096, 1024),
]


def _bench_events(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    se = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in se:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return min(s.elapsed_time(e) * 1e-3 for s, e in se)


def main():
    print(f"{'shape':<22} {'streamk ms':>11} {'pers ms':>9} {'hip ms':>9}  "
          f"{'SK TFLOP/s':>10}  {'SK/hip':>7}  {'SK/pers':>8}")
    print("-" * 90)
    for M, N, K in _SHAPES:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        # warm compiles
        gemm_streamk(A, B)
        gemm_f16_persistent(A, B)
        t_sk = _bench_events(lambda: gemm_streamk(A, B))
        t_pers = _bench_events(lambda: gemm_f16_persistent(A, B))
        A_f = A.float(); B_f = B.float()
        out = torch.empty(M, N, device="cuda", dtype=torch.float32)
        t_hip = _bench_events(lambda: torch.matmul(A_f, B_f, out=out))
        flops = 2.0 * M * N * K
        shape = f"{M}x{N}x{K}"
        print(f"{shape:<22} {t_sk*1e3:>11.3f} {t_pers*1e3:>9.3f} {t_hip*1e3:>9.3f}  "
              f"{flops/t_sk/1e12:>10.2f}  {t_sk/t_hip:>6.2f}x  {t_sk/t_pers:>7.2f}x")


if __name__ == "__main__":
    main()
