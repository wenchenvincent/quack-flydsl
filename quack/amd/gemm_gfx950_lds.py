# Copyright (c) 2026, AMD.

"""LDS-staged + ping-pong 32×32 MFMA GEMM for gfx950.

Extends ``gemm_gfx950_tiled.py`` (in-register-reuse 32×32) with 2-stage
LDS prefetch:

  Stage 0 (prologue):
    each lane loads its A/B fragment slice from global into LDS[stage=0].
  gpu.barrier

  For k in 0..K_tiles-2:
    stage_cur = k & 1, stage_nxt = 1 - stage_cur
    (1) issue next-tile A/B global loads into LDS[stage_nxt]
    (2) load current-stage A/B fragments from LDS into registers
    (3) MFMAs (4 per iter, same 2×2 fragment-reuse as the tiled kernel)
    gpu.barrier — stage_nxt becomes stage_cur next iter

  Final iter: MFMAs on last stage without prefetch.

LDS budget per workgroup (single wave, 32×32 tile, f16):
  Stage: 32×16 A-tile + 16×32 B-tile = 1024 + 1024 = 2 KiB
  Two stages: 4 KiB — well under gfx950's 160 KiB LDS per CU.

Scope (MVP):
  - f16 × f16 → f32 only.
  - M, N multiples of 32, K multiple of 16.
  - No B-preshuffle XOR swizzle yet (separate commit).

The perf win over ``gemm_gfx950_tiled.py`` comes from latency hiding:
while the MFMA instructions of iter k are executing, the HBM→LDS copies
for iter k+1 proceed in parallel. On gfx950 the MFMA issues every ~4
cycles and the HBM→LDS buffer copy is ~50+ cycles for this tile size,
so overlap is worth ~1.5–2× on bandwidth-bound shapes.
"""


import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu as _gpu, range_constexpr, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch


_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 16
_FRAG_A = 4
_FRAG_B = 4
_FRAG_C = 4

# 32×16 A-tile per stage, 16×32 B-tile per stage, both f16 (2 bytes).
_A_TILE_BYTES = 32 * 16 * 2
_B_TILE_BYTES = 16 * 32 * 2


def _align(x, a):
    return (x + a - 1) & ~(a - 1)


def _build_gemm_32x32_lds(*, M, N, K, dtype_str, arch):
    assert M % 32 == 0 and N % 32 == 0 and K % _MFMA_K == 0
    assert dtype_str in ("f16", "bf16")

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_32x32_lds_{dtype_str}_{M}_{N}_{K}_smem",
    )
    # Two stages, each carrying one A tile and one B tile. Layout in LDS:
    #   [stage=0 A (1024B)][stage=0 B (1024B)][stage=1 A (1024B)][stage=1 B (1024B)]
    a_off_stage0 = _align(allocator.ptr, 16)
    allocator.ptr = a_off_stage0 + _A_TILE_BYTES
    b_off_stage0 = _align(allocator.ptr, 16)
    allocator.ptr = b_off_stage0 + _B_TILE_BYTES
    a_off_stage1 = _align(allocator.ptr, 16)
    allocator.ptr = a_off_stage1 + _A_TILE_BYTES
    b_off_stage1 = _align(allocator.ptr, 16)
    allocator.ptr = b_off_stage1 + _B_TILE_BYTES

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y
        tid = fx.thread_idx.x

        lane_row = tid % fx.Int32(16)
        lane_k_group = tid // fx.Int32(16)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        in_elem_type = T.f16 if dtype_str == "f16" else T.bf16

        base_ptr = allocator.get_base()
        # Two-stage LDS views. Shape (32, 16) for A, (16, 32) for B.
        s_a = [
            SmemPtr(base_ptr, a_off_stage0, in_elem_type, shape=(32, 16)),
            SmemPtr(base_ptr, a_off_stage1, in_elem_type, shape=(32, 16)),
        ]
        s_b = [
            SmemPtr(base_ptr, b_off_stage0, in_elem_type, shape=(16, 32)),
            SmemPtr(base_ptr, b_off_stage1, in_elem_type, shape=(16, 32)),
        ]
        for sp in (*s_a, *s_b):
            sp.get()

        ca_h = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), in_elem_type)
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        h_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

        def _load_h_global(div, idx):
            r = fx.memref_alloca(h_reg_ty, reg_lay)
            fx.copy_atom_call(ca_h, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _store_f(div, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            ts = _vfull(1, Float32(val), Float32)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_f, r, fx.slice(div, (None, idx)))

        def _idx(i):
            return ArithValue(i).index_cast(T.index) if hasattr(i, "index_cast") else fx.Index(i)

        m_base = bid_m * fx.Int32(32)
        n_base = bid_n * fx.Int32(32)

        a_row_top = m_base + lane_row
        a_row_bot = m_base + fx.Int32(16) + lane_row
        b_col_left = n_base + lane_row
        b_col_right = n_base + fx.Int32(16) + lane_row

        def _stage_tile_into_lds(k_tile_scalar, stage):
            """Cooperative global→LDS copy for one k-tile of A,B.

            Each of 64 lanes carries 4 A values (top row-half) + 4 A
            (bot) + 4 B (left col-half) + 4 B (right). 64×16 = 1024
            f16 per tile (A and B separately).
            """
            k_tile_base = fx.Int32(k_tile_scalar * _MFMA_K)
            lane_k_base = lane_k_group * fx.Int32(_FRAG_A) + k_tile_base

            # A top half: rows 0..15 of 32×16 LDS tile.
            row_a_top = fx.slice(A_buf, (a_row_top, None))
            row_a_bot = fx.slice(A_buf, (a_row_bot, None))
            a_div_top = fx.logical_divide(row_a_top, fx.make_layout(1, 1))
            a_div_bot = fx.logical_divide(row_a_bot, fx.make_layout(1, 1))
            for i in range_constexpr(_FRAG_A):
                v = _load_h_global(a_div_top, lane_k_base + fx.Int32(i))
                s_a[stage].store(
                    v,
                    [_idx(lane_row),
                     _idx(lane_k_group * fx.Int32(_FRAG_A) + fx.Int32(i))],
                )
            for i in range_constexpr(_FRAG_A):
                v = _load_h_global(a_div_bot, lane_k_base + fx.Int32(i))
                s_a[stage].store(
                    v,
                    [_idx(fx.Int32(16) + lane_row),
                     _idx(lane_k_group * fx.Int32(_FRAG_A) + fx.Int32(i))],
                )

            # B left/right: 16×32 tile.
            for i in range_constexpr(_FRAG_B):
                row_b_k = fx.slice(B_buf, (lane_k_base + fx.Int32(i), None))
                b_div = fx.logical_divide(row_b_k, fx.make_layout(1, 1))
                vl = _load_h_global(b_div, b_col_left)
                vr = _load_h_global(b_div, b_col_right)
                s_b[stage].store(
                    vl,
                    [_idx(lane_k_group * fx.Int32(_FRAG_B) + fx.Int32(i)),
                     _idx(lane_row)],
                )
                s_b[stage].store(
                    vr,
                    [_idx(lane_k_group * fx.Int32(_FRAG_B) + fx.Int32(i)),
                     _idx(fx.Int32(16) + lane_row)],
                )

        def _load_frags_from_lds(stage):
            """Read this lane's MFMA fragments from LDS stage."""
            a_top_vals = []
            a_bot_vals = []
            for i in range_constexpr(_FRAG_A):
                k_in_lds = lane_k_group * fx.Int32(_FRAG_A) + fx.Int32(i)
                a_top_vals.append(s_a[stage].load([_idx(lane_row), _idx(k_in_lds)]))
                a_bot_vals.append(s_a[stage].load([_idx(fx.Int32(16) + lane_row), _idx(k_in_lds)]))
            a_top = vector.from_elements(T.vec(_FRAG_A, in_elem_type), a_top_vals)
            a_bot = vector.from_elements(T.vec(_FRAG_A, in_elem_type), a_bot_vals)

            b_left_vals = []
            b_right_vals = []
            for i in range_constexpr(_FRAG_B):
                k_in_lds = lane_k_group * fx.Int32(_FRAG_B) + fx.Int32(i)
                b_left_vals.append(s_b[stage].load([_idx(k_in_lds), _idx(lane_row)]))
                b_right_vals.append(s_b[stage].load([_idx(k_in_lds), _idx(fx.Int32(16) + lane_row)]))
            b_left = vector.from_elements(T.vec(_FRAG_B, in_elem_type), b_left_vals)
            b_right = vector.from_elements(T.vec(_FRAG_B, in_elem_type), b_right_vals)
            return a_top, a_bot, b_left, b_right

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

        k_tiles = K // _MFMA_K

        # Prologue: stage k=0 into LDS stage 0.
        _stage_tile_into_lds(0, 0)
        _gpu.barrier()

        # Main loop: prefetch next tile into the other LDS stage, MFMA on current.
        for k_tile in range_constexpr(k_tiles):
            cur_stage = k_tile & 1
            # Prefetch k_tile+1 into the other stage, except on last iter.
            if k_tile < k_tiles - 1:
                _stage_tile_into_lds(k_tile + 1, 1 - cur_stage)

            # Consume current stage.
            a_top, a_bot, b_left, b_right = _load_frags_from_lds(cur_stage)

            def _mfma(a, b, acc):
                if dtype_str == "bf16":
                    a_i = vector.bitcast(T.vec(_FRAG_A, T.i16), a)
                    b_i = vector.bitcast(T.vec(_FRAG_B, T.i16), b)
                    return fx.rocdl.mfma_f32_16x16x16bf16_1k(
                        acc_ty, [a_i, b_i, acc, 0, 0, 0],
                    )
                return fx.rocdl.mfma_f32_16x16x16f16(
                    acc_ty, [a, b, acc, 0, 0, 0],
                )

            acc00 = _mfma(a_top, b_left, acc00)
            acc01 = _mfma(a_top, b_right, acc01)
            acc10 = _mfma(a_bot, b_left, acc10)
            acc11 = _mfma(a_bot, b_right, acc11)

            # Sync before the next iter's consume reads the just-prefetched stage.
            _gpu.barrier()

        # Store output.
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
            grid=(M // 32, N // 32, 1), block=(64, 1, 1), stream=stream,
        )

    return launch


_kernel_cache: dict = {}

_DTYPE2STR = {torch.float16: "f16", torch.bfloat16: "bf16"}


def _compile(M, N, K, dtype_str, arch):
    key = (M, N, K, dtype_str, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_32x32_lds(M=M, N=N, K=K, dtype_str=dtype_str, arch=arch)
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_gemm_32x32_lds_out",
    mutates_args=("out",),
    schema="(Tensor a, Tensor b, Tensor(a0!) out) -> ()",
)
def _gemm_32x32_lds_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    assert a.is_cuda and b.is_cuda and out.is_cuda
    assert a.dtype in (torch.float16, torch.bfloat16) and a.dtype == b.dtype
    assert out.dtype == torch.float32
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and M % 32 == 0 and N % 32 == 0 and K % 16 == 0
    assert out.shape == (M, N)
    _compile(M, N, K, _DTYPE2STR[a.dtype], get_rocm_arch())(a, b, out)


@_gemm_32x32_lds_out.register_fake
def _gemm_32x32_lds_out_fake(a, b, out):
    return None


def gemm_32x32_lds(A: Tensor, B: Tensor) -> Tensor:
    """f16/bf16 × f16/bf16 → f32 MFMA GEMM with 32×32 tiles and 2-stage
    LDS ping-pong prefetch.

    Same tile/fragment layout as ``gemm_32x32`` but A,B staged through
    LDS; the next k-tile prefetches in parallel with the current one's
    MFMA consumption — latency-hiding on bandwidth-bound shapes.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == B.dtype and A.dtype in (torch.float16, torch.bfloat16)
    M, K = A.shape
    _, N = B.shape
    out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    _gemm_32x32_lds_out(A, B, out)
    return out


def gemm_f16_32x32_lds(A: Tensor, B: Tensor) -> Tensor:
    """Backwards-compatible f16-only alias."""
    return gemm_32x32_lds(A, B)


__all__ = ["gemm_32x32_lds", "gemm_f16_32x32_lds"]
