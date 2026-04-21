# Copyright (c) 2026, AMD.

"""WMMA GEMM for RDNA (gfx1201 / gfx1250) — wave32, 16×16×16 WMMA.

Unlike CDNA's MFMA path, RDNA uses wave32 and the ``v_wmma_f32_16x16x16_f16``
family of instructions:

    f32[16][16] += f16[16][16] @ f16[16][16]

Fragment layout (wave32, 32 lanes):
  - A frag: f16x8 per lane (16 rows × 16 K = 256 elements / 32 lanes = 8).
  - B frag: f16x8 per lane.
  - Acc:    f32x8 per lane.

MVP scope:
  - f16 × f16 → f32.
  - M, N, K all multiples of 16.
  - Single wave (32 threads) per workgroup, one 16×16 tile per WG.
  - No epilogue, no LDS staging.

This file **compiles on any arch** (the intrinsic is lowered by LLVM
later) but **launching** requires an RDNA4 / gfx1250 device. The public
dispatcher keeps the NotImplementedError on CDNA hardware so callers
don't surprise-fault on a stub launch — this is the kernel that will
wake up once we route through it on real RDNA HW.

Per-lane mapping (RDNA4 WMMA, 32 lanes):
  - lane 0..15 and 16..31 each hold a full A-row / B-col via lane
    doubling — the hardware duplicates the operand across the two
    halves of the wave. Per-lane output f32x8 is 8 accumulator entries
    distributed across the 16 output rows (lane i holds rows at
    positions [i%16, i%16] and some permutation; see the RDNA4 ISA
    manual for the exact unpack rule).

The simpler f16 path uses the ODS wrapper directly, no bitcast needed.
"""


import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, range_constexpr, vector
from flydsl.expr.numeric import Float32
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch


_WMMA_M = 16
_WMMA_N = 16
_WMMA_K = 16
_FRAG_A = 8   # f16 per lane, wave32
_FRAG_B = 8
_FRAG_C = 8   # f32 per lane


def _build_gemm_wmma_16x16_f16(*, M, N, K, arch):
    assert M % _WMMA_M == 0 and N % _WMMA_N == 0 and K % _WMMA_K == 0

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_wmma_16x16_f16_{M}_{N}_{K}_smem",
    )

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y
        tid = fx.thread_idx.x  # 0..31

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        ca_h = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), T.f16)
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        h_reg_ty = fx.MemRefType.get(T.f16, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

        def _load_h(div, idx):
            r = fx.memref_alloca(h_reg_ty, reg_lay)
            fx.copy_atom_call(ca_h, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _store_f(div, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            ts = _vfull(1, Float32(val), Float32)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_f, r, fx.slice(div, (None, idx)))

        # RDNA WMMA per-lane layout: lane-dup pairs (i, i+16) share an A
        # row / B col, but compute 8 disjoint output entries each. For
        # this simple MVP we use the "natural" mapping where lane tid
        # holds row (tid % 16) of A and col (tid % 16) of B.
        lane_idx = tid % fx.Int32(16)

        m_base = bid_m * fx.Int32(_WMMA_M)
        n_base = bid_n * fx.Int32(_WMMA_N)
        a_row = m_base + lane_idx
        b_col = n_base + lane_idx

        acc_ty = T.vec(_FRAG_C, T.f32)
        _zero_list = []
        for _ in range_constexpr(_FRAG_C):
            _zero_list.append(arith.constant(0.0, type=T.f32))
        acc = vector.from_elements(acc_ty, _zero_list)

        k_tiles = K // _WMMA_K
        for k_tile in range_constexpr(k_tiles):
            k_tile_base = fx.Int32(k_tile * _WMMA_K)

            # A fragment: lane-tid holds row (tid % 16) × 16 K elements.
            # For wave32, 2 lanes share a row → 8 K each. We read the
            # 8-K slice owned by this lane: lane < 16 → k∈[0,8), lane
            # ≥ 16 → k∈[8,16).
            half = tid // fx.Int32(16)  # 0 or 1
            k_base = k_tile_base + half * fx.Int32(8)
            row_a = fx.slice(A_buf, (a_row, None))
            a_div = fx.logical_divide(row_a, fx.make_layout(1, 1))
            a_vals = []
            for i in range_constexpr(_FRAG_A):
                a_vals.append(_load_h(a_div, k_base + fx.Int32(i)))
            a_frag = vector.from_elements(T.vec(_FRAG_A, T.f16), a_vals)

            b_vals = []
            for i in range_constexpr(_FRAG_B):
                row_b_k = fx.slice(B_buf, (k_base + fx.Int32(i), None))
                b_div = fx.logical_divide(row_b_k, fx.make_layout(1, 1))
                b_vals.append(_load_h(b_div, b_col))
            b_frag = vector.from_elements(T.vec(_FRAG_B, T.f16), b_vals)

            acc = fx.rocdl.wmma_f32_16x16x16_f16(acc_ty, [a_frag, b_frag, acc])

        # Write back: lane tid writes 8 f32 entries. The WMMA output
        # mapping on RDNA4 places lane tid's f32x8 accumulator at rows
        # (tid % 16) at cols (tid // 16) * 8 + {0..7} — one lane owns
        # a contiguous 8-col stripe of its row.
        col_base = (tid // fx.Int32(16)) * fx.Int32(8)
        for i in range_constexpr(_FRAG_C):
            out_row = m_base + lane_idx
            out_col = n_base + col_base + fx.Int32(i)
            row_c = fx.slice(C_buf, (out_row, None))
            c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))
            val_i = vector.extract(acc, static_position=[i], dynamic_position=[])
            _store_f(c_div, out_col, val_i)

    @flyc.jit
    def launch(
        A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, C).launch(
            grid=(M // _WMMA_M, N // _WMMA_N, 1),
            block=(32, 1, 1), stream=stream,
        )

    return launch


_kernel_cache: dict = {}


def _is_rdna(arch: str) -> bool:
    return arch.startswith("gfx10") or arch.startswith("gfx11") or arch.startswith("gfx12")


def _compile(M, N, K, arch):
    if not _is_rdna(arch):
        raise NotImplementedError(
            f"WMMA GEMM targets RDNA (gfx10/11/12xx); current arch is {arch}. "
            f"On CDNA use quack.amd.gemm_gfx950.gemm_mfma instead."
        )
    key = (M, N, K, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_wmma_16x16_f16(M=M, N=N, K=K, arch=arch)
        _kernel_cache[key] = got
    return got


def gemm_wmma(A: Tensor, B: Tensor) -> Tensor:
    """f16 × f16 → f32 WMMA GEMM for RDNA4 / gfx1250.

    Compiles and runs only on RDNA (gfx10xx / 11xx / 12xx). On CDNA
    (gfx9xx) raises NotImplementedError with a pointer to the MFMA path.

    MVP: single wave (32 threads), 16×16 tile per workgroup, no LDS.
    Larger tiles / LDS ping-pong / B-preshuffle follow the same pattern
    as the CDNA kernels in ``gemm_gfx950_*.py`` but with the wave32
    fragment layout.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    M, K = A.shape
    _, N = B.shape
    out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    _compile(M, N, K, get_rocm_arch())(A, B, out)
    return out


__all__ = ["gemm_wmma"]
