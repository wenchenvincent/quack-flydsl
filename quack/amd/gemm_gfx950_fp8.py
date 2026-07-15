# Copyright (c) 2026, AMD.

"""Standard (unscaled) fp8 MFMA GEMM for gfx950 — ``C = A @ B``.

MVP: single 16×16 output tile per workgroup, one wave, K-loop of 32 (the
``mfma_f32_16x16x32_fp8_fp8`` atom). Inputs are ``torch.float8_e4m3fn``
``(M, K)`` and ``(K, N)`` row-major; output is f32 / f16 / bf16 downcast
from the f32 accumulator. No bias / activation / alpha-beta in the MVP —
this closes the "no standard fp8 GEMM" gap (G6); epilogue features mirror
``gemm_gfx950.py`` and can be added later.

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
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32, Numeric
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator

from quack.amd.flydsl_utils import get_rocm_arch

_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 32
_FRAG_A = 8   # fp8 elements per lane along K
_FRAG_B = 8
_FRAG_C = 4

_OUT2STR = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}


@functools.lru_cache(maxsize=256)
def _compile_fp8_kernel(N: int, K: int, out_dtype_str: str, _m_hint: int = 0):
    arch = get_rocm_arch()
    assert K % _MFMA_K == 0, "fp8 GEMM requires K % 32 == 0"
    assert N % _MFMA_N == 0, "fp8 GEMM requires N % 16 == 0"

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_fp8_{out_dtype_str}_{K}_{N}_smem",
    )

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, m: fx.Int32):
        out_elem_type = (
            T.f32 if out_dtype_str == "f32"
            else T.f16 if out_dtype_str == "f16"
            else T.bf16
        )
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y
        tid = fx.thread_idx.x  # 0..63

        lane_row = tid % fx.Int32(16)
        lane_k_group = tid // fx.Int32(16)  # 0..3 → K blocks of 8

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        m_base = bid_m * fx.Int32(_MFMA_M)
        n_base = bid_n * fx.Int32(_MFMA_N)
        a_row = m_base + lane_row
        b_col = n_base + lane_row

        # ---- copy atoms / register types ----
        out_bufcopy = (
            fx.rocdl.BufferCopy32b() if out_dtype_str == "f32"
            else fx.rocdl.BufferCopy16b()
        )
        # 8 fp8 = 8 bytes = one 64-bit vector load.
        ca_f8x8 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), T.f8)
        ca_f8 = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), T.f8)
        ca_out = fx.make_copy_atom(out_bufcopy, out_elem_type)
        f8x8_reg_ty = fx.MemRefType.get(T.f8, fx.LayoutType.get(8, 1), fx.AddressSpace.Register)
        f8_reg_ty = fx.MemRefType.get(T.f8, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        out_reg_ty = fx.MemRefType.get(out_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        f8x8_lay = fx.make_layout(8, 1)
        reg_lay = fx.make_layout(1, 1)

        def _load_a_frag(div_vec, vec_idx):
            """8 contiguous fp8 from a row of A in one 64-bit load."""
            r = fx.memref_alloca(f8x8_reg_ty, f8x8_lay)
            fx.copy_atom_call(ca_f8x8, fx.slice(div_vec, (None, vec_idx)), r)
            return fx.memref_load_vec(r)

        def _load_b_scalar(div, idx):
            r = fx.memref_alloca(f8_reg_ty, reg_lay)
            fx.copy_atom_call(ca_f8, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _store_out_scalar(div, idx, val_f32):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(out_reg_ty, reg_lay)
            elem_py = Numeric.from_ir_type(out_reg_ty.element_type)
            if fx.const_expr(out_dtype_str == "f32"):
                ts = _vfull(1, Float32(val_f32), Float32)
            else:
                val = ArithValue(val_f32).truncf(out_elem_type)
                ts = _vfull(1, elem_py(val), elem_py)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_out, r, fx.slice(div, (None, idx)))

        # ---- accumulator ----
        acc_ty = T.vec(_FRAG_C, T.f32)
        zeros = [arith.constant(0.0, type=T.f32) for _ in range_constexpr(_FRAG_C)]
        acc = vector.from_elements(acc_ty, zeros)

        # A row is fixed across K; view it as vec8 chunks (8 fp8 per chunk).
        row_a = fx.slice(A_buf, (a_row, None))
        a_div_v = fx.logical_divide(row_a, f8x8_lay)

        # ---- K loop (each iter = one 16x16x32 MFMA) ----
        k_tiles = K // _MFMA_K
        for k_tile in range_constexpr(k_tiles):
            # This lane's K block: k_tile*32 + lane_k_group*8 .. +8.
            a_vec_idx = fx.Int32(k_tile * (_MFMA_K // 8)) + lane_k_group
            a_frag = _load_a_frag(a_div_v, a_vec_idx)  # vec<8 x f8>

            k_base = fx.Int32(k_tile * _MFMA_K) + lane_k_group * fx.Int32(8)
            b_vals = []
            for i in range_constexpr(_FRAG_B):
                row_b_k = fx.slice(B_buf, (k_base + fx.Int32(i), None))
                b_div = fx.logical_divide(row_b_k, reg_lay)
                b_vals.append(_load_b_scalar(b_div, b_col))
            b_frag = vector.from_elements(T.vec(_FRAG_B, T.f8), b_vals)

            # fp8 MFMA takes A/B as i64 (8 fp8 packed). bitcast to vec<1xi64>
            # then extract the scalar (vector.bitcast can't target a scalar).
            a_i64 = vector.extract(
                vector.bitcast(T.vec(1, T.i64), a_frag),
                static_position=[0], dynamic_position=[],
            )
            b_i64 = vector.extract(
                vector.bitcast(T.vec(1, T.i64), b_frag),
                static_position=[0], dynamic_position=[],
            )
            acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                acc_ty, [a_i64, b_i64, acc, 0, 0, 0],
            )

        # ---- store: MFMA 16x16 output, lane L holds 4 rows at col L%16 ----
        lane_col = tid % fx.Int32(16)
        lane_row_base = (tid // fx.Int32(16)) * fx.Int32(4)  # 0,4,8,12
        n_global = n_base + lane_col
        row_c = None
        for e in range_constexpr(_FRAG_C):
            m_local = lane_row_base + fx.Int32(e)
            m_global = m_base + m_local
            val = vector.extract(acc, static_position=[e], dynamic_position=[])
            row_c = fx.slice(C_buf, (m_global, None))
            c_div = fx.logical_divide(row_c, reg_lay)
            _store_out_scalar(c_div, n_global, val)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, m: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        bm = (m + _MFMA_M - 1) // _MFMA_M
        bn = N // _MFMA_N
        launcher = kernel(A, B, C, m)
        launcher.launch(grid=(bm, bn, 1), block=(64, 1, 1), stream=stream)

    return launch


def gemm_fp8(a: Tensor, b: Tensor, out_dtype: torch.dtype = torch.float32,
             out: Optional[Tensor] = None) -> Tensor:
    """``C = A @ B`` for fp8 (e4m3) inputs on gfx950.

    ``a`` is ``(M, K)`` float8_e4m3fn row-major, ``b`` is ``(K, N)`` row-major,
    output ``(M, N)`` in ``out_dtype`` (f32/f16/bf16). Accumulation is f32.
    Constraints: M % 16, N % 16, K % 32.
    """
    assert a.is_cuda and b.is_cuda and a.dim() == 2 and b.dim() == 2
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn, (
        "gemm_fp8 requires float8_e4m3fn inputs"
    )
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"inner dims must match: A={a.shape} B={b.shape}"
    assert M % 16 == 0 and N % 16 == 0 and K % 32 == 0, (
        f"gemm_fp8 requires M%16, N%16, K%32; got {M}×{K}×{N}"
    )
    assert a.stride(-1) == 1 and b.stride(-1) == 1
    assert out_dtype in _OUT2STR, f"unsupported out_dtype {out_dtype}"
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=out_dtype)
    _compile_fp8_kernel(N, K, _OUT2STR[out_dtype], _m_hint=M)(a, b, out, M)
    return out


__all__ = ["gemm_fp8"]
