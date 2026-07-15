# Copyright (c) 2026, AMD.

"""Standard int8 MFMA GEMM for gfx950 — ``C = A @ B`` with i32 accumulate.

MVP: single 16×16 output tile per workgroup, one wave, K-loop of 32 (the
``mfma_i32_16x16x32_i8`` atom). Inputs are ``torch.int8`` ``(M, K)`` and
``(K, N)`` row-major; output is the raw ``int32`` accumulator (exact —
int8×int8 MFMA accumulates in i32, no implicit dequantization). Downstream
dequantization (``acc * scale_a * scale_b``) is the caller's job, matching
how ``mxfp8_ops.py`` handles the fp8 quantized case.

Constraints: M % 16, N % 16, K % 32. Grid = ``(M/16, N/16, 1)``,
block = (64, 1, 1).
"""

import functools
from typing import Optional

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, vector, arith
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator

from quack.amd.flydsl_utils import get_rocm_arch

_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 32
_FRAG_A = 8
_FRAG_B = 8
_FRAG_C = 4


@functools.lru_cache(maxsize=256)
def _compile_int8_kernel(N: int, K: int, _m_hint: int = 0):
    arch = get_rocm_arch()
    assert K % _MFMA_K == 0, "int8 GEMM requires K % 32 == 0"
    assert N % _MFMA_N == 0, "int8 GEMM requires N % 16 == 0"

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_int8_{K}_{N}_smem",
    )

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, m: fx.Int32):
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y
        tid = fx.thread_idx.x

        lane_row = tid % fx.Int32(16)
        lane_k_group = tid // fx.Int32(16)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        m_base = bid_m * fx.Int32(_MFMA_M)
        n_base = bid_n * fx.Int32(_MFMA_N)
        a_row = m_base + lane_row
        b_col = n_base + lane_row

        ca_i8x8 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), T.i8)
        ca_i8 = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), T.i8)
        ca_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.i32)
        i8x8_reg_ty = fx.MemRefType.get(T.i8, fx.LayoutType.get(8, 1), fx.AddressSpace.Register)
        i8_reg_ty = fx.MemRefType.get(T.i8, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        i32_reg_ty = fx.MemRefType.get(T.i32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        i8x8_lay = fx.make_layout(8, 1)
        reg_lay = fx.make_layout(1, 1)

        def _load_a_frag(div_vec, vec_idx):
            r = fx.memref_alloca(i8x8_reg_ty, i8x8_lay)
            fx.copy_atom_call(ca_i8x8, fx.slice(div_vec, (None, vec_idx)), r)
            return fx.memref_load_vec(r)

        def _load_b_scalar(div, idx):
            r = fx.memref_alloca(i8_reg_ty, reg_lay)
            fx.copy_atom_call(ca_i8, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _store_i32_scalar(div, idx, val_i32):
            from flydsl.expr.vector import full as _vfull
            from flydsl.expr.numeric import Int32
            r = fx.memref_alloca(i32_reg_ty, reg_lay)
            ts = _vfull(1, Int32(val_i32), Int32)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_i32, r, fx.slice(div, (None, idx)))

        acc_ty = T.vec(_FRAG_C, T.i32)
        zeros = [arith.constant(0, type=T.i32) for _ in range_constexpr(_FRAG_C)]
        acc = vector.from_elements(acc_ty, zeros)

        row_a = fx.slice(A_buf, (a_row, None))
        a_div_v = fx.logical_divide(row_a, i8x8_lay)

        k_tiles = K // _MFMA_K
        for k_tile in range_constexpr(k_tiles):
            a_vec_idx = fx.Int32(k_tile * (_MFMA_K // 8)) + lane_k_group
            a_frag = _load_a_frag(a_div_v, a_vec_idx)

            k_base = fx.Int32(k_tile * _MFMA_K) + lane_k_group * fx.Int32(8)
            b_vals = []
            for i in range_constexpr(_FRAG_B):
                row_b_k = fx.slice(B_buf, (k_base + fx.Int32(i), None))
                b_div = fx.logical_divide(row_b_k, reg_lay)
                b_vals.append(_load_b_scalar(b_div, b_col))
            b_frag = vector.from_elements(T.vec(_FRAG_B, T.i8), b_vals)

            # int8 MFMA takes A/B as i64 (8 int8 packed).
            a_i64 = vector.extract(
                vector.bitcast(T.vec(1, T.i64), a_frag),
                static_position=[0], dynamic_position=[],
            )
            b_i64 = vector.extract(
                vector.bitcast(T.vec(1, T.i64), b_frag),
                static_position=[0], dynamic_position=[],
            )
            acc = fx.rocdl.mfma_i32_16x16x32_i8(
                acc_ty, [a_i64, b_i64, acc, 0, 0, 0],
            )

        lane_col = tid % fx.Int32(16)
        lane_row_base = (tid // fx.Int32(16)) * fx.Int32(4)
        n_global = n_base + lane_col
        for e in range_constexpr(_FRAG_C):
            m_global = m_base + lane_row_base + fx.Int32(e)
            val = vector.extract(acc, static_position=[e], dynamic_position=[])
            row_c = fx.slice(C_buf, (m_global, None))
            c_div = fx.logical_divide(row_c, reg_lay)
            _store_i32_scalar(c_div, n_global, val)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, m: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        bm = (m + _MFMA_M - 1) // _MFMA_M
        bn = N // _MFMA_N
        launcher = kernel(A, B, C, m)
        launcher.launch(grid=(bm, bn, 1), block=(64, 1, 1), stream=stream)

    return launch


def gemm_int8(a: Tensor, b: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """``C = A @ B`` for int8 inputs on gfx950, exact i32 accumulate.

    ``a`` is ``(M, K)`` int8 row-major, ``b`` is ``(K, N)`` row-major, output
    ``(M, N)`` int32 (the raw accumulator — no dequantization). Constraints:
    M % 16, N % 16, K % 32.
    """
    assert a.is_cuda and b.is_cuda and a.dim() == 2 and b.dim() == 2
    assert a.dtype == torch.int8 and b.dtype == torch.int8, "gemm_int8 requires int8 inputs"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"inner dims must match: A={a.shape} B={b.shape}"
    assert M % 16 == 0 and N % 16 == 0 and K % 32 == 0, (
        f"gemm_int8 requires M%16, N%16, K%32; got {M}×{K}×{N}"
    )
    assert a.stride(-1) == 1 and b.stride(-1) == 1
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=torch.int32)
    _compile_int8_kernel(N, K, _m_hint=M)(a, b, out, M)
    return out


__all__ = ["gemm_int8"]
