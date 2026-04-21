# Copyright (c) 2026, AMD.

"""MX-FP8 blockscaled GEMM for gfx950 (CDNA4).

Uses ``rocdl.mfma_scale_f32_16x16x128_f8f6f4`` through FlyDSL's
reference ``blockscale_preshuffle_gemm`` kernel, vendored under
``quack/amd/_fly_*.py`` (see those modules' headers for upstream
attribution). 2× the compute density of bf16 MFMA at the same tile
— this is the only path for MX-FP8 workloads since hipBLASLt on
ROCm 6.x doesn't cover them.

Layouts:
  - ``a``:        (M, K) fp8_e4m3fn              — row-major
  - ``b``:        (N, K) fp8_e4m3fn pre-shuffled — NT (c = a @ b.T)
  - ``scale_a``:  (K//128, M) f32                — [block_k, m]
  - ``scale_b``:  (N//128, K//128) f32           — [block_n, block_k]
  - ``out``:      (M, N) bf16 or f16

Call ``shuffle_b(b)`` on a plain (N, K) fp8 tensor once and reuse
the shuffled form across many calls (e.g., a persistent weight
matrix on a transformer linear layer).

MVP tile config: tile_m=128, tile_n=128, tile_k=128. Expanding the
tile menu is a follow-up — the upstream kernel supports (128, 256, 128)
and larger with cshuffle epilogue.
"""

from typing import Optional

import torch
from torch import Tensor

from quack.amd._fly_blockscale_preshuffle_gemm import (
    compile_blockscale_preshuffle_gemm,
)


SCALE_BLOCK_K = 128
SCALE_BLOCK_N = 128


def shuffle_b(b: Tensor) -> Tensor:
    """Pre-shuffle a (N, K) fp8 tensor into the layout the kernel expects.

    Shuffle pattern: group N into 16-row chunks and K into 64-col chunks
    (4 klanes × 16 kpack elements), then transpose so the kpack dim is
    innermost and the nlane dim (=16) follows klane. This matches
    ``make_preshuffle_b_layout(kpack_bytes=16, elem_bytes=1)``.
    """
    N, K = b.shape
    assert N % 16 == 0 and K % 64 == 0, f"shape {b.shape} not aligned"
    x = b.view(N // 16, 16, K // 64, 4, 16)
    x = x.permute(0, 2, 3, 1, 4).contiguous()
    return x.view(N, K)


_DTYPE2OUT = {torch.bfloat16: "bf16", torch.float16: "fp16"}


_kernel_cache: dict = {}


def _compile(M, N, K, tile_m, tile_n, tile_k, out_dtype, cshuffle, waves):
    key = (M, N, K, tile_m, tile_n, tile_k, out_dtype, cshuffle, waves)
    got = _kernel_cache.get(key)
    if got is None:
        got = compile_blockscale_preshuffle_gemm(
            M=M, N=N, K=K,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            out_dtype=out_dtype,
            use_cshuffle_epilog=cshuffle,
            use_async_copy=True,
            waves_per_eu=waves,
        )
        _kernel_cache[key] = got
    return got


def _pick_config(M: int, N: int, K: int):
    """Return (tile_m, tile_n, tile_k, cshuffle, waves_per_eu) for a shape.

    Empirically tuned on MI355X / gfx950 (fp8 e4m3fn → bf16).

    Summary of the sweep:
      - 128×128×128 with waves_per_eu=2 wins at every shape measured
        (square and rectangular, K from 128 to 8192). Peaks at 1.44 PFLOPS
        at 8192³.
      - ``waves_per_eu=None`` (compiler default) costs ~30-40% at mid-range;
        ``waves_per_eu=2`` unlocks the occupancy the kernel needs.
      - CShuffle is ~neutral at 2048²+ and helps only at launch-overhead-
        bound 1024³ (marginal). Default off.
      - 128×256, 256×128, 256×256 tiles all trail 128×128 under these
        conditions — the LDS pressure pushes occupancy down.
    """
    # Small-M fallback (< 128): the kernel requires tile_m to divide M.
    tm = 128 if M % 128 == 0 else (64 if M % 64 == 0 else 32)
    tn = 128 if N % 128 == 0 else 64
    tk = 128
    # CShuffle off by default — it's marginal at best on this kernel
    # and triggers a numerics bug at (128, 128, 128) that we haven't
    # isolated yet. Leaving the flag plumbed so a future tuning pass
    # can re-enable it per shape once the small-shape case is fixed.
    cshuffle = False
    waves = 2
    return (tm, tn, tk, cshuffle, waves)


@torch.library.custom_op(
    "quack_amd::_mxfp8_gemm_out",
    mutates_args=("out",),
    schema="(Tensor a, Tensor b, Tensor scale_a, Tensor scale_b, Tensor(a0!) out, bool shuffled) -> ()",
)
def _mxfp8_gemm_out(
    a: Tensor, b: Tensor, scale_a: Tensor, scale_b: Tensor,
    out: Tensor, shuffled: bool,
) -> None:
    assert a.is_cuda and b.is_cuda and scale_a.is_cuda and scale_b.is_cuda and out.is_cuda
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    assert scale_a.dtype == torch.float32 and scale_b.dtype == torch.float32
    assert out.dtype in (torch.bfloat16, torch.float16)
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2 and out.shape == (M, N)
    assert K % SCALE_BLOCK_K == 0 and N % SCALE_BLOCK_N == 0
    assert scale_a.shape == (K // SCALE_BLOCK_K, M)
    assert scale_b.shape == (N // SCALE_BLOCK_N, K // SCALE_BLOCK_K)

    if not shuffled:
        b = shuffle_b(b)

    tile_m, tile_n, tile_k, cshuffle, waves = _pick_config(M, N, K)
    assert M % tile_m == 0 and N % tile_n == 0 and K % tile_k == 0
    out_dtype = _DTYPE2OUT[out.dtype]
    exe = _compile(M, N, K, tile_m, tile_n, tile_k, out_dtype, cshuffle, waves)
    stream = torch.cuda.current_stream()
    exe(out, a, b, scale_a, scale_b, M, N, stream)


@_mxfp8_gemm_out.register_fake
def _mxfp8_gemm_out_fake(a, b, scale_a, scale_b, out, shuffled):
    return None


def mxfp8_gemm(
    a: Tensor, b: Tensor,
    scale_a: Tensor, scale_b: Tensor,
    out: Optional[Tensor] = None,
    *,
    shuffled: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """MX-FP8 blockscaled GEMM: ``out = a @ b.T`` with per-block scales.

    Args:
      a:        (M, K) fp8_e4m3fn, row-major.
      b:        (N, K) fp8_e4m3fn, row-major OR pre-shuffled (see ``shuffled``).
      scale_a:  (K//128, M) f32. ``scale_a[block_k, m]``.
      scale_b:  (N//128, K//128) f32. ``scale_b[block_n, block_k]``.
      out:      optional preallocated (M, N) ``out_dtype`` output.
      shuffled: True if ``b`` is already passed through ``shuffle_b``.
      out_dtype: bf16 or f16.

    Constraints:
      - M multiple of 32
      - N multiple of 128
      - K multiple of 128
    """
    assert a.is_cuda and a.dtype == torch.float8_e4m3fn
    assert b.is_cuda and b.dtype == torch.float8_e4m3fn
    M, K = a.shape
    N, _ = b.shape
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=out_dtype)
    _mxfp8_gemm_out(a, b, scale_a, scale_b, out, shuffled)
    return out


__all__ = ["mxfp8_gemm", "shuffle_b", "SCALE_BLOCK_K", "SCALE_BLOCK_N"]
