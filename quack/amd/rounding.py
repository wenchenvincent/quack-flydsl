# Copyright (c) 2026, AMD.

"""Stochastic rounding for fp8 quantization on gfx950 (CDNA4).

Realizes the ``sr_seed`` epilogue placeholder (``gemm_default_epi.py``) with
genuine hardware stochastic rounding via the ``v_cvt_sr_fp8_f32`` /
``v_cvt_sr_bf8_f32`` instructions (FlyDSL ``rocdl.cvt_sr_fp8_f32`` /
``cvt_sr_bf8_f32``): each f32 value is rounded *up or down* to one of its two
bracketing fp8 grid points, chosen probabilistically by a 32-bit random dither
with bias proportional to the fractional distance. Over many draws the result
is **unbiased** (``E[SR(x)] == x``), unlike round-to-nearest which is biased —
this is what lets low-precision training accumulate gradients without drift.

The random dither is produced in-kernel by a counter-based PRNG (murmur3
finalizer over ``global_element_index ^ mix(seed)``), so no host-side random
tensor is needed and different launches/tiles get decorrelated randomness.

Public API:
    quantize_fp8_sr(x, seed, dtype=torch.float8_e4m3fn) -> fp8 tensor
"""

import functools

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, rocdl, vector
from flydsl.expr.typing import T

from quack.amd.flydsl_utils import get_rocm_arch

_BLOCK = 256

# fp8 variant → (rocdl SR op, torch dtype)
_FP8_VARIANTS = {
    torch.float8_e4m3fn: ("fp8", "cvt_sr_fp8_f32"),
    torch.float8_e5m2: ("bf8", "cvt_sr_bf8_f32"),
}


# murmur3 fmix32 constants, encoded as signed i32 (arith.constant range-checks):
#   0x85EBCA6B → -2048144789, 0xC2B2AE35 → -1028477387
_FMIX_C1 = -2048144789
_FMIX_C2 = -1028477387


def _mix32(v):
    """murmur3 fmix32 — avalanche a 32-bit integer (raw i32 ``ir.Value``) into a
    well-distributed 32-bit random word. Pure arith (muli/xori/shrui), no libcall."""
    def sh(x, n):
        return arith.shrui(x, arith.constant(n, type=T.i32))

    v = arith.xori(v, sh(v, 16))
    v = arith.muli(v, arith.constant(_FMIX_C1, type=T.i32))
    v = arith.xori(v, sh(v, 13))
    v = arith.muli(v, arith.constant(_FMIX_C2, type=T.i32))
    v = arith.xori(v, sh(v, 16))
    return v


@functools.lru_cache(maxsize=8)
def _compile_sr_kernel(variant: str, sr_op_name: str):
    arch = get_rocm_arch()
    assert arch in ("gfx950", "gfx942"), f"fp8 SR needs CDNA3/4, got {arch}"
    sr_op = getattr(rocdl, sr_op_name)

    @flyc.kernel(known_block_size=[_BLOCK, 1, 1])
    def kernel(X: fx.Tensor, OUT: fx.Tensor, seed: fx.Int32):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        gid = bid * fx.Int32(_BLOCK) + tid

        X_buf = fx.rocdl.make_buffer_tensor(X)    # (1, numel) f32
        O_buf = fx.rocdl.make_buffer_tensor(OUT)  # (1, numel) i8 (fp8 bytes)
        cf = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        cb = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), T.i8)
        f_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        b_ty = fx.MemRefType.get(T.i8, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        lay = fx.make_layout(1, 1)

        xd = fx.logical_divide(fx.slice(X_buf, (0, None)), lay)
        od = fx.logical_divide(fx.slice(O_buf, (0, None)), lay)

        # per-element random dither: decorrelate by element index and seed.
        rand = _mix32(arith.xori(gid.ir_value(), _mix32(seed.ir_value())))

        rx = fx.memref_alloca(f_ty, lay)
        fx.copy_atom_call(cf, fx.slice(xd, (None, gid)), rx)
        xval = fx.memref_load_vec(rx)[0].ir_value()

        old = arith.constant(0, type=T.i32)
        res = sr_op(T.i32, xval, rand, old, 0)  # SR-convert f32 -> fp8 byte 0
        res_v4 = vector.bitcast(
            T.vec(4, T.i8), vector.from_elements(T.vec(1, T.i32), [res])
        )
        byte0 = vector.extract(res_v4, static_position=[0], dynamic_position=[])
        rb = fx.memref_alloca(b_ty, lay)
        fx.memref_store_vec(vector.from_elements(T.vec(1, T.i8), [byte0]), rb)
        # OOB stores (gid >= numel) are dropped by the buffer descriptor.
        fx.copy_atom_call(cb, rb, fx.slice(od, (None, gid)))

    @flyc.jit
    def launch(X: fx.Tensor, OUT: fx.Tensor, seed: fx.Int32, n_blocks: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
        kernel(X, OUT, seed).launch(grid=(n_blocks, 1, 1), block=(_BLOCK, 1, 1), stream=stream)

    return launch


def quantize_fp8_sr(
    x: Tensor, seed: int = 0, dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tensor:
    """Stochastically round ``x`` (f32/f16/bf16) to fp8 on gfx950.

    Each element rounds up or down to a bracketing fp8 grid point with
    probability set by its fractional distance (hardware ``v_cvt_sr_fp8_f32``),
    dithered by a per-element PRNG seeded from ``seed``. ``dtype`` is
    ``float8_e4m3fn`` (default) or ``float8_e5m2``. Output has ``x``'s shape.
    """
    assert x.is_cuda, "quantize_fp8_sr needs a CUDA/ROCm tensor"
    assert dtype in _FP8_VARIANTS, f"unsupported fp8 dtype {dtype}"
    variant, sr_op_name = _FP8_VARIANTS[dtype]
    xf = x.detach().to(torch.float32).contiguous().reshape(1, -1)
    numel = xf.shape[1]
    out = torch.empty(1, numel, device=x.device, dtype=torch.uint8)
    n_blocks = (numel + _BLOCK - 1) // _BLOCK
    _compile_sr_kernel(variant, sr_op_name)(xf, out, int(seed), n_blocks)
    return out.view(dtype).reshape(x.shape)


__all__ = ["quantize_fp8_sr"]
