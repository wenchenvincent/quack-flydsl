# Copyright (c) 2026, AMD.

"""Minimal FlyDSL blockscaled MFMA kernel — single 16x16x128 tile.

Hand-written proof-of-life for ``rocdl.mfma_scale_f32_16x16x128_f8f6f4``
on gfx950/CDNA4. Structure mirrors the ``quack.amd.gemm_gfx950`` standard
MFMA kernel but uses the scaled variant with per-block scale factors
applied post-MFMA via f32 FMA into a running accumulator.

**B is transposed** to keep the K-dim contiguous per column (the MFMA
fragment layout wants each lane to carry 32 K-consecutive fp8 values of
B at its assigned column). Callers pass ``B.T.contiguous()`` instead of
the standard (K, N) row-major B. FlyDSL's reference kernel achieves the
same with an explicit preshuffle pass; this MVP takes the B^T shortcut.

Scope (intentionally minimal):
    - A: (M, K) fp8_e4m3fn, K contiguous (torch default)
    - B: (N, K) fp8_e4m3fn, K contiguous (= standard B transposed)
    - A_scale: (K // 128, M) f32 — one scale per (scale-block, row) pair
    - B_scale: (N // 128, K // 128) f32 — one scale per (n-block, scale-block)
    - Output: (M, N) f32
    - Computes ``C[m, n] = Σ_k dequant(A[m,k]) * dequant(B[n,k])``
    - One workgroup per 16×16 output tile, one MFMA per scale block,
      no LDS pipeline.

Thread layout within a 16x16 output tile (wave64):
    lane_row = tid % 16       # M row within tile (= N col for B)
    lane_k_group = tid // 16  # 0..3 selects K-slice (32 K values per lane)

Fragment packing for ``mfma_scale_f32_16x16x128_f8f6f4``:
    Each lane holds 32 fp8 A values and 32 fp8 B values = 32 bytes each,
    packed as 4 × i64 → i32x8 via vector.bitcast (matches FlyDSL's
    ``pack_i64x4_to_i32x8`` in ``blockscale_preshuffle_gemm.py:447``).

Scaling:
    The MFMA ``scaleA`` / ``scaleB`` operands are set to the neutral
    ``0x7F7F7F7F`` mask. Per-element scaling is applied after the MFMA
    via ``math.fma`` of ``scale_a[row] * scale_b[0]`` into the running
    f32 accumulator — same pattern as the reference kernel's
    ``compute_tile_blockscale`` (lines 542–585 of the reference).
"""

from typing import Optional

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, range_constexpr, vector, buffer_ops
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32, Numeric
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl._mlir import ir
from flydsl._mlir.dialects import math as math_dialect

from quack.amd.flydsl_utils import get_rocm_arch


_TILE_M = 16
_TILE_N = 16
_TILE_K = 128
_FRAG_C = 4


def _build_blockscaled_16x16x128(*, M, N, K, arch):
    """One workgroup per 16×16 output tile, loops K-tiles of 128."""
    assert M % _TILE_M == 0 and N % _TILE_N == 0 and K % _TILE_K == 0
    allocator = SmemAllocator(
        None, arch=arch, global_sym_name=f"quack_amd_bs_gemm_{M}_{N}_{K}_smem",
    )

    @flyc.kernel
    def kernel(
        A: fx.Tensor,       # (M, K) fp8
        B: fx.Tensor,       # (K, N) fp8
        A_scale: fx.Tensor, # (K/128, M) f32
        B_scale: fx.Tensor, # (N/128, K/128) f32
        C: fx.Tensor,       # (M, N) f32
    ):
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y
        tid = fx.thread_idx.x                   # 0..63

        lane_row = tid % fx.Int32(16)           # A row / B col / C col
        lane_k_group = tid // fx.Int32(16)      # 0..3

        A_rsrc = buffer_ops.create_buffer_resource(A)
        B_rsrc = buffer_ops.create_buffer_resource(B)
        AS_rsrc = buffer_ops.create_buffer_resource(A_scale)
        BS_rsrc = buffer_ops.create_buffer_resource(B_scale)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        # C store helper (f32).
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

        def _store_f32(div, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            ts = _vfull(1, Float32(val), Float32)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_f, r, fx.slice(div, (None, idx)))

        # Global accumulator: f32x4 per lane, zero-init.
        acc_ty = T.vec(_FRAG_C, T.f32)
        zeros = []
        for _ in range_constexpr(_FRAG_C):
            zeros.append(arith.constant(0.0, type=T.f32))
        global_acc = vector.from_elements(acc_ty, zeros)

        # M / N tile base (for the output tile this workgroup owns).
        m_base = bid_m * fx.Int32(_TILE_M)   # row offset into A/C (M-dim)
        n_base = bid_n * fx.Int32(_TILE_N)   # col offset into B/C (N-dim)

        # Per-lane A row (bytes from A origin = (m_base + lane_row) * K_bytes).
        a_row = m_base + lane_row            # global A row index
        b_col = n_base + lane_row            # global B column index (B is row-major K×N)

        # K-tile loop: one scale block per iteration.
        k_tiles = K // _TILE_K

        for k_tile in range_constexpr(k_tiles):
            k_tile_base_elems = fx.Int32(k_tile * _TILE_K)
            lane_k_start = lane_k_group * fx.Int32(32) + k_tile_base_elems  # K element offset

            # --- Load A fragment: 32 consecutive fp8 bytes from
            # (a_row, lane_k_start). Using i32 loads so offset is in i32
            # units = byte offset / 4. Two vec_width=4 loads (16 bytes each).
            # Byte offset: a_row*K + lane_k_start. In i32 units: (a_row*K + lane_k_start) / 4.
            # Since K and lane_k_start are both multiples of 4, per-lane
            # offsets align.
            a_byte_offset = a_row * fx.Int32(K) + lane_k_start
            a_i32_offset_low = a_byte_offset // fx.Int32(4)
            a_i32_offset_high = a_i32_offset_low + fx.Int32(4)  # +4 i32s = +16 bytes
            a_lo = buffer_ops.buffer_load(A_rsrc, a_i32_offset_low, vec_width=4, dtype=T.i32)
            a_hi = buffer_ops.buffer_load(A_rsrc, a_i32_offset_high, vec_width=4, dtype=T.i32)
            a_lo_parts = []
            a_hi_parts = []
            for i in range_constexpr(4):
                a_lo_parts.append(vector.extract(a_lo, static_position=[i], dynamic_position=[]))
                a_hi_parts.append(vector.extract(a_hi, static_position=[i], dynamic_position=[]))
            a_i32x8 = vector.from_elements(T.vec(8, T.i32), a_lo_parts + a_hi_parts)

            # --- Load B fragment: same pattern, from (b_col, lane_k_start).
            b_byte_offset = b_col * fx.Int32(K) + lane_k_start
            b_i32_offset_low = b_byte_offset // fx.Int32(4)
            b_i32_offset_high = b_i32_offset_low + fx.Int32(4)
            b_lo = buffer_ops.buffer_load(B_rsrc, b_i32_offset_low, vec_width=4, dtype=T.i32)
            b_hi = buffer_ops.buffer_load(B_rsrc, b_i32_offset_high, vec_width=4, dtype=T.i32)
            b_lo_parts = []
            b_hi_parts = []
            for i in range_constexpr(4):
                b_lo_parts.append(vector.extract(b_lo, static_position=[i], dynamic_position=[]))
                b_hi_parts.append(vector.extract(b_hi, static_position=[i], dynamic_position=[]))
            b_i32x8 = vector.from_elements(T.vec(8, T.i32), b_lo_parts + b_hi_parts)

            # --- MFMA (zero-init local block accumulator, scaleA/B = neutral).
            block_acc = vector.from_elements(acc_ty, zeros)
            block_acc = fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                acc_ty,
                [a_i32x8, b_i32x8, block_acc,
                 0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
            )

            # --- Load scales for this scale block.
            # scale_a[k_tile, m_row] for each of the 4 M-rows this lane touches:
            # row = m_base + lane_k_group * 4 + i (for i in 0..3) — matches C layout.
            # scale_b[n_block, k_tile] single value (assuming N tile fits in one n-block).
            n_block = (n_base + lane_row) // fx.Int32(128)
            sb_elem_offset = n_block * fx.Int32(K // 128) + fx.Int32(k_tile)
            s_b = buffer_ops.buffer_load(BS_rsrc, sb_elem_offset, vec_width=1, dtype=T.f32)
            s_b_av = ArithValue(s_b)

            # Multiply-accumulate block_acc * scale into global_acc.
            for i in range_constexpr(_FRAG_C):
                out_row_i = m_base + lane_k_group * fx.Int32(_FRAG_C) + fx.Int32(i)
                sa_elem_offset = fx.Int32(k_tile) * fx.Int32(M) + out_row_i
                s_a_i = buffer_ops.buffer_load(
                    AS_rsrc, sa_elem_offset, vec_width=1, dtype=T.f32,
                )
                s_prod = ArithValue(s_a_i) * s_b_av
                # fma(block_acc[i], s_prod, global_acc[i]) -> global_acc[i]
                block_i = vector.extract(block_acc, static_position=[i], dynamic_position=[])
                global_i = vector.extract(global_acc, static_position=[i], dynamic_position=[])
                new_i = math_dialect.fma(block_i, s_prod, global_i)
                global_acc = vector.insert(
                    new_i, global_acc,
                    static_position=[i], dynamic_position=[],
                )

        # --- Store C. Each lane: 4 rows at column lane_row.
        for i in range_constexpr(_FRAG_C):
            out_row = m_base + lane_k_group * fx.Int32(_FRAG_C) + fx.Int32(i)
            out_col = n_base + lane_row
            row_c = fx.slice(C_buf, (out_row, None))
            c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))
            val_i = vector.extract(global_acc, static_position=[i], dynamic_position=[])
            _store_f32(c_div, out_col, val_i)

    @flyc.jit
    def launch(
        A: fx.Tensor, B: fx.Tensor,
        A_scale: fx.Tensor, B_scale: fx.Tensor, C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, A_scale, B_scale, C).launch(
            grid=(M // _TILE_M, N // _TILE_N, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    return launch


_kernel_cache: dict = {}


def _compile(M, N, K, arch):
    key = (M, N, K, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_blockscaled_16x16x128(M=M, N=N, K=K, arch=arch)
        _kernel_cache[key] = got
    return got


def mxfp8_gemm_mfma(
    A: Tensor, B: Tensor, A_scale: Tensor, B_scale: Tensor,
) -> Tensor:
    """FlyDSL MFMA blockscaled fp8 GEMM. Returns f32 output.

    Shape contract same as `quack.amd.gemm_blockscaled.mxfp8_gemm`;
    this is the hand-written MFMA path (currently slow — no LDS — but
    exercises the full scaled-MFMA codegen).
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float8_e4m3fn
    assert B.dtype == torch.float8_e4m3fn
    M, K = A.shape
    # B is in (N, K) layout (= standard B transposed).
    N, K_b = B.shape
    assert K == K_b, f"A K={K} != B K={K_b}"
    assert M % _TILE_M == 0 and N % _TILE_N == 0 and K % _TILE_K == 0
    out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    _compile(M, N, K, get_rocm_arch())(A, B, A_scale, B_scale, out)
    return out


__all__ = ["mxfp8_gemm_mfma"]
