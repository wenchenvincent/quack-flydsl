# Copyright (c) 2026, AMD.

"""4-wave 64×64 MFMA GEMM for gfx950.

256-thread workgroup (4 waves × 64 lanes) covering a 64×64 output
tile via a 2×2 wave grid. Each wave handles an independent 32×32
sub-tile with the same 2×2 MFMA fragment-reuse pattern as
``gemm_gfx950_tiled.py``.

Wave mapping:
    wave_id  = tid // 64   ∈ [0, 4)
    wave_row = wave_id // 2 ∈ [0, 2)   — covers rows [wave_row*32, +32)
    wave_col = wave_id % 2  ∈ [0, 2)   — covers cols [wave_col*32, +32)

Per wave: 4 MFMA accumulators (acc00, acc01, acc10, acc11), each f32x4
per lane; 2 A fragments (top, bot row halves) and 2 B fragments
(left, right col halves) loaded fresh per k-tile, with in-register
reuse across the 4 MFMAs.

Key perf difference vs the single-wave 32×32:
  - 4× output work per workgroup launch (amortises scheduling overhead).
  - 4× parallel MFMA dispatch per cycle (modulo wave scheduler).
  - Same per-lane HBM load profile — no cross-wave LDS sharing yet.

Scope (MVP):
  - f16 × f16 → f32.
  - M, N multiples of 64, K multiple of 16.
  - No LDS staging (single-wave version already has it; cross-wave
    sharing is a follow-up).

A cross-wave LDS-shared variant (with 128×128 tile via 4×4 MFMA grid
per wave) is the next step once this infrastructure is in place.
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


_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 16
_FRAG_A = 4
_FRAG_B = 4
_FRAG_C = 4


def _build_gemm_64x64_4wave(*, M, N, K, dtype_str, arch):
    assert M % 64 == 0 and N % 64 == 0 and K % _MFMA_K == 0
    assert dtype_str in ("f16", "bf16")

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_64x64_4wave_{dtype_str}_{M}_{N}_{K}_smem",
    )

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y
        tid = fx.thread_idx.x

        wave_id = tid // fx.Int32(64)
        lane_in_wave = tid % fx.Int32(64)
        lane_row = lane_in_wave % fx.Int32(16)
        lane_k_group = lane_in_wave // fx.Int32(16)

        # Wave grid: 2x2 inside a 64x64 tile; each wave covers a 32x32 sub-tile.
        wave_row = wave_id // fx.Int32(2)
        wave_col = wave_id % fx.Int32(2)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        in_elem_type = T.f16 if dtype_str == "f16" else T.bf16
        ca_h4 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), in_elem_type)
        ca_h = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), in_elem_type)
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        h4_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(4, 1), fx.AddressSpace.Register)
        h_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        h4_lay = fx.make_layout(4, 1)
        reg_lay = fx.make_layout(1, 1)

        def _load_h4(div_vec, vec_idx):
            r = fx.memref_alloca(h4_reg_ty, h4_lay)
            fx.copy_atom_call(ca_h4, fx.slice(div_vec, (None, vec_idx)), r)
            return fx.memref_load_vec(r)

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

        # Output tile base for this wave's 32×32 sub-tile.
        m_base_wg = bid_m * fx.Int32(64)
        n_base_wg = bid_n * fx.Int32(64)
        m_base = m_base_wg + wave_row * fx.Int32(32)
        n_base = n_base_wg + wave_col * fx.Int32(32)

        a_row_top = m_base + lane_row
        a_row_bot = m_base + fx.Int32(16) + lane_row
        b_col_left = n_base + lane_row
        b_col_right = n_base + fx.Int32(16) + lane_row

        acc_ty = T.vec(_FRAG_C, T.f32)

        def _zero_acc():
            zs = []
            for _ in range_constexpr(_FRAG_C):
                zs.append(arith.constant(0.0, type=T.f32))
            return vector.from_elements(acc_ty, zs)

        acc00 = _zero_acc()
        acc01 = _zero_acc()
        acc10 = _zero_acc()
        acc11 = _zero_acc()

        # Vectorized A loads — precompute vec4-divided rows, per k-tile
        # we just shift the vec index.
        row_a_top = fx.slice(A_buf, (a_row_top, None))
        row_a_bot = fx.slice(A_buf, (a_row_bot, None))
        a_div_top_v = fx.logical_divide(row_a_top, h4_lay)
        a_div_bot_v = fx.logical_divide(row_a_bot, h4_lay)

        k_tiles = K // _MFMA_K
        for k_tile in range_constexpr(k_tiles):
            k_tile_base = fx.Int32(k_tile * _MFMA_K)
            lane_k_base = lane_k_group * fx.Int32(_FRAG_B) + k_tile_base
            vec_idx_A = lane_k_group + fx.Int32(k_tile * 4)

            a_top = _load_h4(a_div_top_v, vec_idx_A)
            a_bot = _load_h4(a_div_bot_v, vec_idx_A)

            b_left_vals = []
            b_right_vals = []
            for i in range_constexpr(_FRAG_B):
                row_b_k = fx.slice(B_buf, (lane_k_base + fx.Int32(i), None))
                b_div = fx.logical_divide(row_b_k, fx.make_layout(1, 1))
                b_left_vals.append(_load_h(b_div, b_col_left))
                b_right_vals.append(_load_h(b_div, b_col_right))
            b_left = vector.from_elements(T.vec(_FRAG_B, in_elem_type), b_left_vals)
            b_right = vector.from_elements(T.vec(_FRAG_B, in_elem_type), b_right_vals)

            def _mfma(a, b, acc):
                if dtype_str == "bf16":
                    a_i = vector.bitcast(T.vec(_FRAG_A, T.i16), a)
                    b_i = vector.bitcast(T.vec(_FRAG_B, T.i16), b)
                    return fx.rocdl.mfma_f32_16x16x16bf16_1k(
                        acc_ty, [a_i, b_i, acc, 0, 0, 0],
                    )
                return fx.rocdl.mfma_f32_16x16x16f16(acc_ty, [a, b, acc, 0, 0, 0])

            acc00 = _mfma(a_top, b_left, acc00)
            acc01 = _mfma(a_top, b_right, acc01)
            acc10 = _mfma(a_bot, b_left, acc10)
            acc11 = _mfma(a_bot, b_right, acc11)

        def _store_subtile(acc, row_base, col_base):
            for i in range_constexpr(_FRAG_C):
                out_row = row_base + lane_k_group * fx.Int32(_FRAG_C) + fx.Int32(i)
                out_col = col_base + lane_row
                row_c = fx.slice(C_buf, (out_row, None))
                c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))
                val_i = vector.extract(acc, static_position=[i], dynamic_position=[])
                _store_f(c_div, out_col, val_i)

        _store_subtile(acc00, m_base,                 n_base)
        _store_subtile(acc01, m_base,                 n_base + fx.Int32(16))
        _store_subtile(acc10, m_base + fx.Int32(16),  n_base)
        _store_subtile(acc11, m_base + fx.Int32(16),  n_base + fx.Int32(16))

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
            grid=(M // 64, N // 64, 1), block=(256, 1, 1), stream=stream,
        )

    return launch


_kernel_cache: dict = {}

_DTYPE2STR = {torch.float16: "f16", torch.bfloat16: "bf16"}


def _compile(M, N, K, dtype_str, arch):
    key = (M, N, K, dtype_str, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_64x64_4wave(M=M, N=N, K=K, dtype_str=dtype_str, arch=arch)
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_gemm_64x64_4wave_out",
    mutates_args=("out",),
    schema="(Tensor a, Tensor b, Tensor(a0!) out) -> ()",
)
def _gemm_64x64_4wave_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    assert a.is_cuda and b.is_cuda and out.is_cuda
    assert a.dtype in (torch.float16, torch.bfloat16) and a.dtype == b.dtype
    assert out.dtype == torch.float32
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and M % 64 == 0 and N % 64 == 0 and K % 16 == 0
    assert out.shape == (M, N)
    _compile(M, N, K, _DTYPE2STR[a.dtype], get_rocm_arch())(a, b, out)


@_gemm_64x64_4wave_out.register_fake
def _gemm_64x64_4wave_out_fake(a, b, out):
    return None


def gemm_64x64_4wave(A: Tensor, B: Tensor) -> Tensor:
    """4-wave 64×64-tile f16/bf16 × f16/bf16 → f32 MFMA GEMM.

    256-thread workgroup (4 waves) covering a 64×64 output tile via
    a 2×2 wave grid; each wave runs the 32×32 MFMA pattern on its own
    sub-tile.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == B.dtype and A.dtype in (torch.float16, torch.bfloat16)
    M, K = A.shape
    _, N = B.shape
    out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    _gemm_64x64_4wave_out(A, B, out)
    return out


def gemm_f16_64x64_4wave(A: Tensor, B: Tensor) -> Tensor:
    """Backwards-compatible f16-only alias."""
    return gemm_64x64_4wave(A, B)


__all__ = ["gemm_64x64_4wave", "gemm_f16_64x64_4wave"]
