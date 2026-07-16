# Copyright (c) 2026, AMD.

"""Bench FlyDSL gemm_nn / gemm_tn vs THREE Triton references + torch (hipBLASLt).

Triton baselines:
  1. **Primus-Turbo GEMM** — production Triton kernel from
     ``/workspace/Primus-Turbo/primus_turbo/triton/gemm/gemm_kernel.py``
     (AMD-authored, autotuned for HIP / gfx950).
  2. **Triton tutorial matmul** — faithful reproduction of
     ``triton-lang/triton/python/tutorials/03-matrix-multiplication.py``
     (with GROUP_SIZE_M L2 swizzle, standard across nv/amd).
  3. **Upstream triton_kernels matmul_ogs** — the dense 2D path of
     ``python/triton_kernels/triton_kernels/matmul_ogs.py`` at the
     ``v3.4.0`` tag (compatible with the installed Triton 3.4.0;
     HEAD of main pins constexpr_function / TMA APIs we don't have).
     Vendored read-only into ``/tmp/tk_v34`` — call with no
     routing/gather/scatter/epilogue for a plain dense bf16 matmul.

All Triton kernels have their own autotune / opt-flag sweep, so the
comparison is "best Triton config vs best FlyDSL config" —
apples-to-apples.

Usage:
    PYTHONPATH=/workspace/quack python -m tests.amd.bench_vs_triton
"""

from __future__ import annotations

import sys

import torch
import triton
import triton.language as tl

from quack.amd.gemm_gfx950_nn import gemm_nn
from quack.amd.gemm_gfx950_tn import gemm_tn


# ---------------------------------------------------------------------------
# (1) Primus-Turbo GEMM — direct import (AMD upstream production kernel)
# ---------------------------------------------------------------------------

_PRIMUS_PATH = "/workspace/Primus-Turbo"
if _PRIMUS_PATH not in sys.path:
    sys.path.insert(0, _PRIMUS_PATH)

try:
    from primus_turbo.triton.gemm.gemm_kernel import gemm_triton_kernel as _primus_kernel
    _HAVE_PRIMUS = True
except ImportError:
    _HAVE_PRIMUS = False


def primus_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _primus_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# ---------------------------------------------------------------------------
# (2) Triton tutorial matmul — faithful reproduction of
#     https://github.com/triton-lang/triton/blob/main/python/tutorials/
#     03-matrix-multiplication.py
# ---------------------------------------------------------------------------


def _tutorial_autotune_configs():
    """Matches the tutorial's HIP autotune config block."""
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 2},
                      num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
                      num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
                      num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
                      num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
                      num_warps=4, num_stages=2),
    ]


@triton.autotune(configs=_tutorial_autotune_configs(), key=['M', 'N', 'K'])
@triton.jit
def _tutorial_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Faithful Triton tutorial matmul (M, K) @ (K, N) -> (M, N).

    Standard group-swizzle launch + masked loads + f32 accumulator.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)  # will be cast at store below

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def tutorial_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    _tutorial_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# ---------------------------------------------------------------------------
# (3) Upstream triton_kernels matmul_ogs (v3.4.0 tag) — dense 2D path.
# ---------------------------------------------------------------------------
#
# Main-branch matmul.py pins constexpr_function / TMA APIs that don't exist
# in triton 3.4.0 (the version installed on this ROCm box). The v3.4.0 tag
# of the triton-lang/triton monorepo includes a ``triton_kernels/matmul_ogs``
# module that was shipped together with triton 3.4.0 — the ABI matches.
# We extract it to /tmp/tk_v34 and import from there.

_TK_V34_PATH = "/tmp/tk_v34/python/triton_kernels"
if _TK_V34_PATH not in sys.path:
    sys.path.insert(0, _TK_V34_PATH)

try:
    from triton_kernels.matmul_ogs import matmul_ogs as _matmul_ogs  # noqa: E402
    _HAVE_OGS = True
except ImportError:
    _HAVE_OGS = False


def ogs_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Dense 2D path of upstream triton_kernels.matmul_ogs.

    No routing_data / gather_indx / scatter_indx / bias / activation /
    epilogue — just a plain ``a @ b``. ``matmul_ogs`` views ``w`` as
    ``(1, K, N)`` internally and runs the single-expert grid.
    """
    return _matmul_ogs(a, b, None)


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------


def _bench(fn, warmup: int = 15, iters: int = 50) -> float:
    # Canonical min-of-iters CUDA-event timing lives in quack.amd.profiler.
    from quack.amd.profiler import benchmark

    return benchmark(fn, warmup=warmup, iters=iters, return_mode="min")


def main():
    print(f"Primus-Turbo available: {_HAVE_PRIMUS}")
    print(f"triton_kernels.matmul_ogs (v3.4.0) available: {_HAVE_OGS}")
    header_cells = ["kernel", "shape", "dtype", "flydsl TF/s"]
    if _HAVE_PRIMUS:
        header_cells.append("primus TF/s")
    header_cells.append("tutorial TF/s")
    if _HAVE_OGS:
        header_cells.append("ogs TF/s")
    header_cells += ["torch TF/s",
                     "fly/primus" if _HAVE_PRIMUS else "",
                     "fly/tutorial",
                     "fly/ogs" if _HAVE_OGS else "",
                     "fly/torch"]
    print("  ".join(c.rjust(13) for c in header_cells))

    for dtype, dt_str in [(torch.bfloat16, "bf16"), (torch.float16, "f16")]:
        for M, K, N in [(2048, 4096, 1024), (4096, 8192, 2048), (8192, 16384, 4096)]:
            torch.manual_seed(0)
            a = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
            b = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
            out = torch.empty(M, N, device="cuda", dtype=dtype)
            flops = 2 * M * N * K / 1e12

            # Warm compiles.
            gemm_nn(a, b, out); torch.cuda.synchronize()
            if _HAVE_PRIMUS:
                primus_gemm(a, b); torch.cuda.synchronize()
            tutorial_gemm(a, b); torch.cuda.synchronize()
            if _HAVE_OGS:
                ogs_gemm(a, b); torch.cuda.synchronize()

            t_fly = _bench(lambda: gemm_nn(a, b, out))
            t_prm = _bench(lambda: primus_gemm(a, b)) if _HAVE_PRIMUS else None
            t_tut = _bench(lambda: tutorial_gemm(a, b))
            t_ogs = _bench(lambda: ogs_gemm(a, b)) if _HAVE_OGS else None
            t_trc = _bench(lambda: torch.matmul(a, b, out=out))

            row = [
                "NN", f"{M}×{N}×{K}", dt_str,
                f"{flops/t_fly:.1f}",
            ]
            if _HAVE_PRIMUS:
                row.append(f"{flops/t_prm:.1f}")
            row.append(f"{flops/t_tut:.1f}")
            if _HAVE_OGS:
                row.append(f"{flops/t_ogs:.1f}")
            row += [
                f"{flops/t_trc:.1f}",
                f"{t_prm/t_fly:.2f}x" if _HAVE_PRIMUS else "",
                f"{t_tut/t_fly:.2f}x",
                f"{t_ogs/t_fly:.2f}x" if _HAVE_OGS else "",
                f"{t_trc/t_fly:.2f}x",
            ]
            print("  ".join(c.rjust(13) for c in row))


if __name__ == "__main__":
    main()
