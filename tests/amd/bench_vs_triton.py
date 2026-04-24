# Copyright (c) 2026, AMD.

"""Bench FlyDSL gemm_nn / gemm_tn vs a Triton GEMM vs torch (hipBLASLt).

Apples-to-apples comparison on the same shapes / dtypes. The Triton
kernel here is a standard textbook GEMM (from the Triton tutorial,
adapted to ROCm gfx950 / bf16+f16). Autotune across a few BLOCK_M /
BLOCK_N / BLOCK_K configs so the Triton kernel gets its own best-pick
per shape too — otherwise the comparison is unfair.

Usage:
    PYTHONPATH=/workspace/quack python -m tests.amd.bench_vs_triton
"""

from __future__ import annotations

import time

import torch
import triton
import triton.language as tl

from quack.amd.gemm_gfx950_nn import gemm_nn
from quack.amd.gemm_gfx950_tn import gemm_tn


# Triton matmul kernel — A (M, K) @ B (K, N) -> C (M, N), all row-major.
# Standard tutorial shape; this is the "NN" layout in our terminology.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                      num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
                      num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                      num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
                      num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                      num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
                      num_stages=2, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _triton_nn_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Group swizzle for better L2 re-use.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(A_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_gemm_nn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _triton_nn_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def triton_gemm_tn(a_kn: torch.Tensor, b_kn: torch.Tensor) -> torch.Tensor:
    """Wrap the NN triton kernel to compute a_kn.T @ b_kn (= dW for training).

    Uses a.T as a strided view — triton consumes the stride tuple directly.
    """
    K, M = a_kn.shape
    K2, N = b_kn.shape
    assert K == K2
    a_view = a_kn.transpose(0, 1)  # (M, K) view, strides (1, M)
    return triton_gemm_nn(a_view, b_kn)


def _bench(fn, warmup: int = 15, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    time.sleep(0.1)
    evs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for s, e in evs:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return sorted(s.elapsed_time(e) * 1e-3 for s, e in evs)[0]


def main():
    header = (
        f"{'kernel':<10} {'shape':<24} {'dtype':<6}"
        f" {'flydsl TF/s':>12} {'triton TF/s':>13} {'torch TF/s':>12}"
        f" {'fly/triton':>11} {'fly/torch':>11}"
    )
    print(header); print("-" * len(header))

    for dtype, dt_str in [(torch.bfloat16, "bf16"), (torch.float16, "f16")]:
        # NN: y = x @ W  (both row-major, A K-inner, B N-inner)
        for M, K, N in [(2048, 4096, 1024), (4096, 8192, 2048), (8192, 16384, 4096)]:
            torch.manual_seed(0)
            a = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
            b = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
            out = torch.empty(M, N, device="cuda", dtype=dtype)
            flops = 2 * M * N * K / 1e12

            # Warm compiles separately so the first bench doesn't capture JIT time.
            gemm_nn(a, b, out); torch.cuda.synchronize()
            triton_gemm_nn(a, b); torch.cuda.synchronize()

            t_fly = _bench(lambda: gemm_nn(a, b, out))
            t_tri = _bench(lambda: triton_gemm_nn(a, b))
            t_trc = _bench(lambda: torch.matmul(a, b, out=out))
            print(
                f"{'NN':<10} {M}×{N}×{K:<10} {dt_str:<6}"
                f" {flops/t_fly:>12.1f} {flops/t_tri:>13.1f} {flops/t_trc:>12.1f}"
                f" {t_tri/t_fly:>10.3f}x {t_trc/t_fly:>10.3f}x"
            )

        # TN: dW = dy.T @ x
        for K, M, N in [(2048, 4096, 1024), (4096, 8192, 2048), (8192, 16384, 4096)]:
            torch.manual_seed(0)
            a = torch.randn(K, M, device="cuda", dtype=dtype) * 0.1
            b = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
            out = torch.empty(M, N, device="cuda", dtype=dtype)
            flops = 2 * M * N * K / 1e12

            gemm_tn(a, b, out); torch.cuda.synchronize()
            triton_gemm_tn(a, b); torch.cuda.synchronize()

            t_fly = _bench(lambda: gemm_tn(a, b, out))
            t_tri = _bench(lambda: triton_gemm_tn(a, b))
            t_trc = _bench(lambda: torch.matmul(a.T, b, out=out))
            print(
                f"{'TN K=' + str(K):<10} {M}×{N:<18} {dt_str:<6}"
                f" {flops/t_fly:>12.1f} {flops/t_tri:>13.1f} {flops/t_trc:>12.1f}"
                f" {t_tri/t_fly:>10.3f}x {t_trc/t_fly:>10.3f}x"
            )


if __name__ == "__main__":
    main()
