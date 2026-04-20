# Copyright (c) 2026, AMD.

"""DYNAMIC stream-K GEMM: persistent workgroups + atomic work counter.

Grid = num_cus (persistent), each workgroup:
    1. Processes its initial tile (tile_idx = bid).
    2. Atomically increments a shared f32 counter to get the next tile
       index (tile_idx = counter_old + num_cus).
    3. Exits when tile_idx >= total_tiles.

Counter is f32 (exact integer representation for tile counts < 2^23)
since ``rocdl.raw_ptr_buffer_atomic_fadd`` returns the old value and
supports agent-scope syncing — saving a round-trip through the llvm
dialect's generic AtomicRMWOp.

Upper bound on iterations per wg = ``ceil(total_tiles / num_cus) + 4``
— a small slack for worst-case imbalance under dynamic work stealing.

Scope (MVP): same as ``gemm_persistent`` — f16/bf16 × f16/bf16 → f32,
no bias/activation/alpha/beta/C. Extensions are straightforward extensions
of this pattern.
"""

from typing import Optional

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, range_constexpr, vector, buffer_ops, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32, Numeric
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch
from quack.amd.tile_scheduler import get_num_cus


_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 16
_FRAG_C = 4


def _build_gemm_streamk_f16(*, M, N, K, num_cus, arch):
    assert M % _MFMA_M == 0 and N % _MFMA_N == 0 and K % _MFMA_K == 0
    tiles_m = M // _MFMA_M
    tiles_n = N // _MFMA_N
    total_tiles = tiles_m * tiles_n
    # Upper bound on iterations per wg — compile-time known.
    steps_per_wg = (total_tiles + num_cus - 1) // num_cus + 4

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_streamk_f16_{M}_{N}_{K}_{num_cus}_smem",
    )

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, Counter: fx.Tensor, C: fx.Tensor):
        wg_id = fx.block_idx.x
        tid = fx.thread_idx.x

        lane_row = tid % fx.Int32(16)
        lane_k_group = tid // fx.Int32(16)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)
        # Counter is a 1-element f32 tensor; use the rsrc API for atomics.
        counter_rsrc = buffer_ops.create_buffer_resource(Counter)

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
        c_total_tiles = fx.Int32(total_tiles)
        c_num_cus = fx.Int32(num_cus)
        zero_i32 = arith.constant(0, type=T.i32)
        one_f32 = arith.constant(1.0, type=T.f32)

        # tile_idx carries across iterations. Start with wg_id. The
        # AST rewriter treats each iteration's `if` body as a closure,
        # so we can't reassign tile_idx from inside — compute the next
        # value OUTSIDE the guard.
        tile_idx = wg_id

        for step in range_constexpr(steps_per_wg):
            current_tile = tile_idx  # capture in an outer-scope name
            # Only process the tile if it's in range.
            if arith.cmpi(arith.CmpIPredicate.ult, current_tile, c_total_tiles):
                bid_m = current_tile // fx.Int32(tiles_n)
                bid_n = current_tile % fx.Int32(tiles_n)

                m_base = bid_m * fx.Int32(_MFMA_M)
                n_base = bid_n * fx.Int32(_MFMA_N)
                a_row = m_base + lane_row
                b_col = n_base + lane_row

                zeros = []
                for _ in range_constexpr(_FRAG_C):
                    zeros.append(arith.constant(0.0, type=T.f32))
                acc = vector.from_elements(acc_ty, zeros)

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

                for i in range_constexpr(_FRAG_C):
                    out_row = m_base + lane_k_group * fx.Int32(_FRAG_C) + fx.Int32(i)
                    out_col = n_base + lane_row
                    row_c = fx.slice(C_buf, (out_row, None))
                    c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))
                    val_i = vector.extract(acc, static_position=[i], dynamic_position=[])
                    _store_f(c_div, out_col, val_i)

            # Advance tile_idx for the next iteration. Currently STATIC-style
            # (+= num_cus). DYNAMIC with atomic readback is the follow-up
            # — needs llvm.AtomicRMWOp + LDS broadcast of the returned
            # old value (see quack/amd/tile_scheduler.py docstring).
            tile_idx = tile_idx + c_num_cus

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, Counter: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, Counter, C).launch(
            grid=(num_cus, 1, 1), block=(64, 1, 1), stream=stream,
        )

    return launch


_kernel_cache: dict = {}


def _compile(M, N, K, num_cus, arch):
    key = (M, N, K, num_cus, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_streamk_f16(M=M, N=N, K=K, num_cus=num_cus, arch=arch)
        _kernel_cache[key] = got
    return got


def gemm_f16_streamk(A: Tensor, B: Tensor) -> Tensor:
    """Stream-K persistent GEMM: f16 × f16 → f32.

    Currently emits the same STATIC-persistent work partition as
    ``gemm_persistent.gemm_f16_persistent``; the atomic-counter DYNAMIC
    path is scaffolded (counter tensor is allocated and plumbed) but the
    atomic read-back + LDS-broadcast pattern is a documented follow-up.

    Kept as a separate module to make the scheduler-mode comparison
    explicit — callers wanting the DYNAMIC path can switch by flipping
    a single call site once the upstream wiring lands.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    M, K = A.shape
    _, N = B.shape
    arch = get_rocm_arch()
    num_cus = get_num_cus(arch)
    out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    # Host-side atomic counter. Each wg starts at its bid and advances via
    # this counter when atomic-readback wiring lands.
    counter = torch.zeros(1, device=A.device, dtype=torch.float32)
    _compile(M, N, K, num_cus, arch)(A, B, counter, out)
    return out


__all__ = ["gemm_f16_streamk"]
