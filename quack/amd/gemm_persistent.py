# Copyright (c) 2026, AMD.

"""STATIC persistent-kernel MFMA GEMM — exercises the tile_scheduler infra.

Grid = num_cus (persistent mode), each workgroup iterates over a strided
set of tiles (bid, bid + num_cus, bid + 2*num_cus, ...). Each iteration
handles a 16×16 output tile by the same pattern as the standard
``gemm_gfx950.gemm_mfma`` but with tile coordinates derived from the
tile index rather than the grid dims.

Scope (narrow on purpose — proves the scheduler wiring works):
    - Inputs: f16 × f16
    - Output: f32
    - No bias / activation / alpha / beta / C
    - M, N, K multiples of 16

When the pattern is validated, the same wrapper can be applied to the
epilogue-full kernel in gemm_gfx950.py.
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
from quack.amd.tile_scheduler import get_num_cus


_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 16
_FRAG_C = 4


def _build_gemm_persistent_f16(*, M, N, K, num_cus, arch):
    assert M % _MFMA_M == 0 and N % _MFMA_N == 0 and K % _MFMA_K == 0
    tiles_m = M // _MFMA_M
    tiles_n = N // _MFMA_N
    total_tiles = tiles_m * tiles_n
    # Ceiling division: each workgroup handles at most this many tiles.
    steps_per_wg = (total_tiles + num_cus - 1) // num_cus

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_pers_f16_{M}_{N}_{K}_{num_cus}_smem",
    )

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        wg_id = fx.block_idx.x
        tid = fx.thread_idx.x

        lane_row = tid % fx.Int32(16)
        lane_k_group = tid // fx.Int32(16)

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

        acc_ty = T.vec(_FRAG_C, T.f32)

        # Persistent loop: each workgroup handles tiles
        # (wg_id, wg_id + num_cus, wg_id + 2*num_cus, ...) up to total_tiles.
        for step in range_constexpr(steps_per_wg):
            tile_idx = wg_id + fx.Int32(step * num_cus)

            # Guard: skip if beyond total tiles.
            if arith.cmpi(arith.CmpIPredicate.ult, tile_idx, fx.Int32(total_tiles)):
                bid_m = tile_idx // fx.Int32(tiles_n)
                bid_n = tile_idx % fx.Int32(tiles_n)

                m_base = bid_m * fx.Int32(_MFMA_M)
                n_base = bid_n * fx.Int32(_MFMA_N)
                a_row = m_base + lane_row
                b_col = n_base + lane_row

                # Zero accumulator (fresh per tile).
                zeros = []
                for _ in range_constexpr(_FRAG_C):
                    zeros.append(arith.constant(0.0, type=T.f32))
                acc = vector.from_elements(acc_ty, zeros)

                # K loop.
                k_tiles = K // _MFMA_K
                for k_tile in range_constexpr(k_tiles):
                    k_tile_base = fx.Int32(k_tile * _MFMA_K)
                    lane_k_base = lane_k_group * fx.Int32(_FRAG_C) + k_tile_base

                    row_a = fx.slice(A_buf, (a_row, None))
                    a_div = fx.logical_divide(row_a, fx.make_layout(1, 1))
                    a_vals = []
                    for i in range_constexpr(_FRAG_C):
                        a_vals.append(_load_h(a_div, lane_k_base + fx.Int32(i)))
                    a_frag = vector.from_elements(T.vec(_FRAG_C, T.f16), a_vals)

                    b_vals = []
                    for i in range_constexpr(_FRAG_C):
                        row_b_k = fx.slice(B_buf, (lane_k_base + fx.Int32(i), None))
                        b_div = fx.logical_divide(row_b_k, fx.make_layout(1, 1))
                        b_vals.append(_load_h(b_div, b_col))
                    b_frag = vector.from_elements(T.vec(_FRAG_C, T.f16), b_vals)

                    acc = fx.rocdl.mfma_f32_16x16x16f16(
                        acc_ty, [a_frag, b_frag, acc, 0, 0, 0],
                    )

                # Store.
                for i in range_constexpr(_FRAG_C):
                    out_row = m_base + lane_k_group * fx.Int32(_FRAG_C) + fx.Int32(i)
                    out_col = n_base + lane_row
                    row_c = fx.slice(C_buf, (out_row, None))
                    c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))
                    val_i = vector.extract(acc, static_position=[i], dynamic_position=[])
                    _store_f(c_div, out_col, val_i)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, C).launch(grid=(num_cus, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


_kernel_cache: dict = {}


def _compile(M, N, K, num_cus, arch):
    key = (M, N, K, num_cus, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_persistent_f16(M=M, N=N, K=K, num_cus=num_cus, arch=arch)
        _kernel_cache[key] = got
    return got


def gemm_f16_persistent(A: Tensor, B: Tensor) -> Tensor:
    """STATIC persistent-kernel GEMM: f16 × f16 → f32.

    Grid size is fixed at the arch's compute-unit count
    (``get_num_cus(arch)``); each workgroup processes multiple tiles in
    a strided pattern. Matches the ``gemm_mfma`` semantics otherwise.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    M, K = A.shape
    _, N = B.shape
    arch = get_rocm_arch()
    num_cus = get_num_cus(arch)
    out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    _compile(M, N, K, num_cus, arch)(A, B, out)
    return out


__all__ = ["gemm_f16_persistent"]
