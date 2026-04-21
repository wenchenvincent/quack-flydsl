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


def _compile(M, N, K, tile_m, tile_n, tile_k, out_dtype):
    key = (M, N, K, tile_m, tile_n, tile_k, out_dtype)
    got = _kernel_cache.get(key)
    if got is None:
        got = compile_blockscale_preshuffle_gemm(
            M=M, N=N, K=K,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            out_dtype=out_dtype,
            use_cshuffle_epilog=False,
            use_async_copy=True,
        )
        _kernel_cache[key] = got
    return got


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

    # Default tile config. Upstream reference uses 128×256 for large N
    # (the 4-wave grid along N amortises better), 128×128 for smaller.
    tile_m = 128 if M % 128 == 0 else (64 if M % 64 == 0 else 32)
    tile_n = 256 if (N % 256 == 0 and N >= 512) else 128
    tile_k = 128
    assert M % tile_m == 0 and N % tile_n == 0 and K % tile_k == 0
    out_dtype = _DTYPE2OUT[out.dtype]
    exe = _compile(M, N, K, tile_m, tile_n, tile_k, out_dtype)
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
