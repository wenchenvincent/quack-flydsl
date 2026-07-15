# Copyright (c) 2026, AMD.

"""MXFP4 (e2m1 + e8m0) scaled MFMA GEMM for gfx950 — ``C = A @ B.T``.

OCP MX standard: 4-bit e2m1 elements in 32-element K-blocks, each block
scaled by one e8m0 (power-of-2) factor. Uses the hardware-scaled
``mfma_scale_f32_16x16x128_f8f6f4`` atom (cbsz=blgp=4 selects fp4, the
scaleA/scaleB operands are e8m0). NT layout: both A ``(M, K)`` and B
``(N, K)`` are K-inner fp4-packed (2 nibbles/byte along K); output is
``(M, N)`` f32.

Empirically-verified layout (gfx950, 2026-07-15):
  - The MFMA reads A/B from register i32[0..3] (32 fp4/lane); i32[4..7]
    are unused. 4 lanes per M/N-row, K-group = lane // 16, so each lane
    carries one 32-element K-block.
  - opselA/opselB = 0 => the hardware uses byte 0 of scaleA/scaleB for
    every lane, so each lane passes its own block's e8m0 scale in byte 0.

Constraints: M % 16, N % 16, K % 128. Grid = ``(M/16, N/16, 1)``,
block = (64, 1, 1).
"""

import functools

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, vector, arith
from flydsl.expr.typing import T
from flydsl.expr.numeric import Float32
from flydsl.utils.smem_allocator import SmemAllocator

from quack.amd.flydsl_utils import get_rocm_arch
from quack.amd.mxfp4_ops import quantize_mxfp4

_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 128  # K per scaled-MFMA
_KBLK = 32  # e8m0 block size
_GROUPS = _MFMA_K // _KBLK  # 4 K-blocks per MFMA (one per lane-group)


@functools.lru_cache(maxsize=256)
def _compile_mxfp4_kernel(N: int, K: int, _m_hint: int = 0):
    arch = get_rocm_arch()
    assert K % _MFMA_K == 0, "mxfp4 GEMM requires K % 128 == 0"
    assert N % _MFMA_N == 0, "mxfp4 GEMM requires N % 16 == 0"

    allocator = SmemAllocator(
        None,
        arch=arch,
        global_sym_name=f"quack_amd_gemm_mxfp4_{K}_{N}_smem",
    )

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel(A: fx.Tensor, SA: fx.Tensor, B: fx.Tensor, SB: fx.Tensor, C: fx.Tensor, m: fx.Int32):
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y
        tid = fx.thread_idx.x

        lane_row = tid % fx.Int32(16)  # M/N index within tile
        lane_g = tid // fx.Int32(16)  # 0..3 → K-block group

        A_buf = fx.rocdl.make_buffer_tensor(A)  # (M, K/2) uint8
        SA_buf = fx.rocdl.make_buffer_tensor(SA)  # (M, K/32) uint8
        B_buf = fx.rocdl.make_buffer_tensor(B)  # (N, K/2) uint8
        SB_buf = fx.rocdl.make_buffer_tensor(SB)  # (N, K/32) uint8
        C_buf = fx.rocdl.make_buffer_tensor(C)

        m_base = bid_m * fx.Int32(_MFMA_M)
        n_base = bid_n * fx.Int32(_MFMA_N)
        a_row = m_base + lane_row
        b_row = n_base + lane_row

        # copy atoms: 16 packed bytes (=32 fp4) per block = one 128-bit load.
        ca_b16 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), T.i8)
        ca_s = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), T.i8)
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        b16_reg_ty = fx.MemRefType.get(T.i8, fx.LayoutType.get(16, 1), fx.AddressSpace.Register)
        s_reg_ty = fx.MemRefType.get(T.i8, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        b16_lay = fx.make_layout(16, 1)
        reg_lay = fx.make_layout(1, 1)

        def _load_frag(buf_div, byte_col):
            """Load 16 packed bytes (32 fp4) → the low 4 i32 of a vec<8xi32>."""
            r = fx.memref_alloca(b16_reg_ty, b16_lay)
            fx.copy_atom_call(ca_b16, fx.slice(buf_div, (None, byte_col)), r)
            v16 = fx.memref_load_vec(r)  # vec<16 x i8>
            lo = vector.bitcast(T.vec(4, T.i32), v16)  # 4 i32 = 32 fp4
            zero = arith.constant(0, type=T.i32)
            elems = [
                vector.extract(lo, static_position=[i], dynamic_position=[])
                for i in range_constexpr(4)
            ] + [zero for _ in range_constexpr(4)]
            return vector.from_elements(T.vec(8, T.i32), elems)

        def _load_scale(sbuf_div, blk):
            r = fx.memref_alloca(s_reg_ty, reg_lay)
            fx.copy_atom_call(ca_s, fx.slice(sbuf_div, (None, blk)), r)
            byte = fx.memref_load_vec(r)[0].ir_value()  # i8 e8m0
            return arith.extui(T.i32, byte)  # scale in byte 0

        def _store_f(div, idx, val_f32):
            from flydsl.expr.vector import full as _vf

            r = fx.memref_alloca(f_reg_ty, reg_lay)
            fx.memref_store_vec(_vf(1, Float32(val_f32), Float32), r)
            fx.copy_atom_call(ca_f, r, fx.slice(div, (None, idx)))

        # per-lane byte views of A/B rows (K/2 packed bytes) and scale rows.
        a_row_div = fx.logical_divide(fx.slice(A_buf, (a_row, None)), b16_lay)
        b_row_div = fx.logical_divide(fx.slice(B_buf, (b_row, None)), b16_lay)
        sa_row_div = fx.logical_divide(fx.slice(SA_buf, (a_row, None)), reg_lay)
        sb_row_div = fx.logical_divide(fx.slice(SB_buf, (b_row, None)), reg_lay)

        acc_ty = T.vec(4, T.f32)
        acc = vector.from_elements(
            acc_ty, [arith.constant(0.0, type=T.f32) for _ in range_constexpr(4)]
        )

        k_tiles = K // _MFMA_K
        for kt in range_constexpr(k_tiles):
            blk = fx.Int32(kt * _GROUPS) + lane_g  # scale-block index
            byte_vec_idx = blk  # 16-byte chunk index == block index
            a_frag = _load_frag(a_row_div, byte_vec_idx)
            b_frag = _load_frag(b_row_div, byte_vec_idx)
            sa = _load_scale(sa_row_div, blk)
            sb = _load_scale(sb_row_div, blk)
            acc = fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                acc_ty,
                [a_frag, b_frag, acc, 4, 4, 0, sa, 0, sb],
            )

        # store: lane holds 4 output rows at col = lane_row.
        out_col = n_base + lane_row
        row_base = (tid // fx.Int32(16)) * fx.Int32(4)
        for e in range_constexpr(4):
            out_row = m_base + row_base + fx.Int32(e)
            val = vector.extract(acc, static_position=[e], dynamic_position=[])
            c_div = fx.logical_divide(fx.slice(C_buf, (out_row, None)), reg_lay)
            _store_f(c_div, out_col, val)

    @flyc.jit
    def launch(
        A: fx.Tensor,
        SA: fx.Tensor,
        B: fx.Tensor,
        SB: fx.Tensor,
        C: fx.Tensor,
        m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        bm = (m + _MFMA_M - 1) // _MFMA_M
        bn = N // _MFMA_N
        launcher = kernel(A, SA, B, SB, C, m)
        launcher.launch(grid=(bm, bn, 1), block=(64, 1, 1), stream=stream)

    return launch


def gemm_mxfp4(a: Tensor, b: Tensor) -> Tensor:
    """MXFP4 GEMM ``C = A @ B.T`` on gfx950.

    ``a`` is ``(M, K)`` and ``b`` is ``(N, K)`` (NT layout), both f32/f16/bf16
    — quantized to MXFP4 (e2m1 + e8m0, 32-element K-blocks) internally. Output
    ``(M, N)`` f32. Constraints: M % 16, N % 16, K % 128.
    """
    assert a.is_cuda and b.is_cuda and a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    Nb, K2 = b.shape
    assert K == K2, f"inner dims must match: A={a.shape} B={b.shape}"
    assert M % 16 == 0 and Nb % 16 == 0 and K % _MFMA_K == 0, (
        f"gemm_mxfp4 requires M%16, N%16, K%128; got {M}×{K}×{Nb}"
    )
    a_packed, a_scale, _ = quantize_mxfp4(a, axis=-1)
    b_packed, b_scale, _ = quantize_mxfp4(b, axis=-1)
    a_packed = a_packed.contiguous()
    b_packed = b_packed.contiguous()
    a_scale = a_scale.contiguous()
    b_scale = b_scale.contiguous()
    out = torch.empty(M, Nb, device=a.device, dtype=torch.float32)
    _compile_mxfp4_kernel(Nb, K, _m_hint=M)(a_packed, a_scale, b_packed, b_scale, out, M)
    return out


__all__ = ["gemm_mxfp4"]
