# Copyright (c) 2026, AMD.

"""FlyDSL bitonic-sort top-k kernel.

Per-row top-k selection using a single-wave bitonic sort. Each workgroup
handles one row of the (M, N) input. The 64 lanes of the wave cooperate
to sort N ≤ 64 values into ascending order via ``shuffle_xor``
compare-swaps; top-k elements are then the last k lanes, emitted in
descending order.

Scope (MVP, matches a subset of QuACK's topk_fwd contract):
    - N ∈ {8, 16, 32, 64} (power-of-2 ≤ wave size)
    - k ∈ {1, 2, 4, 8, 16, 32, 64}, k ≤ N
    - Input dtype: f32
    - Output values: f32 in descending order, shape (M, k)
    - Output indices: i32, shape (M, k)

Multi-wave extension for N > 64 is a follow-up: each lane would hold
N/64 values, we'd sort those in-register first (per-lane sort), then
bitonic-merge across waves via LDS.
"""

import math as _py_math
from typing import Optional, Tuple

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, range_constexpr, buffer_ops
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32, Int32, Numeric
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch


WAVE_SIZE = 64


def _build_topk_single_wave_f32(*, N, k, arch):
    assert N in (8, 16, 32, 64), f"single-wave topk requires N ∈ {{8,16,32,64}}, got {N}"
    assert k in (1, 2, 4, 8, 16, 32, 64) and k <= N, f"bad k={k} for N={N}"
    log2_N = int(_py_math.log2(N))

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_topk_f32_{N}_{k}_smem",
    )

    @flyc.kernel
    def kernel(X: fx.Tensor, OutVals: fx.Tensor, OutIdx: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x  # 0..63

        X_buf = fx.rocdl.make_buffer_tensor(X)
        OutVals_buf = fx.rocdl.make_buffer_tensor(OutVals)
        OutIdx_buf = fx.rocdl.make_buffer_tensor(OutIdx)

        row_x = fx.slice(X_buf, (bid, None))
        x_div = fx.logical_divide(row_x, fx.make_layout(1, 1))
        row_v = fx.slice(OutVals_buf, (bid, None))
        v_div = fx.logical_divide(row_v, fx.make_layout(1, 1))
        row_i = fx.slice(OutIdx_buf, (bid, None))
        i_div = fx.logical_divide(row_i, fx.make_layout(1, 1))

        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        ca_i = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.i32)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        i_reg_ty = fx.MemRefType.get(T.i32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

        def _load_f32(div, idx):
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            fx.copy_atom_call(ca_f, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _store_f32(div, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            ts = _vfull(1, Float32(val), Float32)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_f, r, fx.slice(div, (None, idx)))

        def _store_i32(div, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(i_reg_ty, reg_lay)
            ts = _vfull(1, Int32(val), Int32)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_i, r, fx.slice(div, (None, idx)))

        # Each lane takes one element of x[bid, :]. Lanes ≥ N don't
        # participate (the sort runs log2(N) stages so they never partner
        # with real lanes) but we still need a valid load for them — use
        # idx_safe=0 and override the value to -inf.
        neg_inf = arith.constant(float("-inf"), type=T.f32)
        in_range = tid < fx.Int32(N)
        idx_safe = in_range.select(tid, fx.Int32(0))
        x_loaded = _load_f32(x_div, idx_safe)
        my_val = ArithValue(in_range.select(x_loaded, neg_inf))
        my_idx = in_range.select(tid, fx.Int32(-1))

        # Bitonic sort in ascending order. After sorting, lanes
        # N-k..N-1 hold the top-k (largest at lane N-1).
        width = fx.Int32(WAVE_SIZE)
        for stage in range_constexpr(log2_N):
            k_outer = 1 << (stage + 1)  # 2, 4, ..., N
            for sub in range_constexpr(stage + 1):
                j = k_outer >> (sub + 1)  # k_outer/2, k_outer/4, ..., 1
                j_i32 = fx.Int32(j)
                peer_val = my_val.shuffle_xor(j_i32, width)
                peer_idx = my_idx.shuffle_xor(j_i32, width)
                # Ascending bitonic: in each "ascending" sub-block (where
                # (tid & k_outer) == 0), lower lane gets smaller; in
                # "descending" sub-blocks, lower lane gets larger. The
                # combined behaviour after all stages is a global sort.
                direction_ascending = (tid & fx.Int32(k_outer)) == fx.Int32(0)
                am_lower = (tid & j_i32) == fx.Int32(0)
                # "keep smaller" when direction_ascending == am_lower:
                #   - asc block + lower lane → keep smaller
                #   - desc block + higher lane (XOR of both false) → keep smaller
                keep_smaller = direction_ascending == am_lower
                take_mine_if_smaller = my_val < peer_val  # mine is smaller
                take_mine_if_larger = my_val > peer_val
                # When keep_smaller is True: take mine if mine < peer.
                # When keep_smaller is False: take mine if mine > peer.
                take_mine = keep_smaller.select(take_mine_if_smaller, take_mine_if_larger)
                new_val = take_mine.select(my_val, peer_val)
                new_idx = take_mine.select(my_idx, peer_idx)
                my_val = ArithValue(new_val)
                my_idx = new_idx

        # After sort, lanes 0..N-1 are in ascending order. Top-k (desc)
        # is lane (N-1), (N-2), ..., (N-k).
        # Lane (N-1-out_pos) writes out[out_pos].
        # Equivalently: for lane `tid`, if tid ∈ [N-k, N-1], it writes
        # out at position out_pos = (N-1) - tid.
        tid_iv = tid.ir_value() if hasattr(tid, "ir_value") else tid
        lo = arith.constant(N - k, type=T.i32)
        hi = arith.constant(N, type=T.i32)
        if arith.cmpi(arith.CmpIPredicate.sge, tid_iv, lo):
            if arith.cmpi(arith.CmpIPredicate.slt, tid_iv, hi):
                out_pos = fx.Int32(N - 1) - tid
                _store_f32(v_div, out_pos, my_val)
                _store_i32(i_div, out_pos, my_idx)

    @flyc.jit
    def launch(X: fx.Tensor, OutVals: fx.Tensor, OutIdx: fx.Tensor,
               M: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(X, OutVals, OutIdx).launch(
            grid=(M, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream,
        )

    return launch


_kernel_cache: dict = {}


def _compile(N, k, arch):
    key = (N, k, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_topk_single_wave_f32(N=N, k=k, arch=arch)
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_topk_f32_out",
    mutates_args=("out_vals", "out_idx"),
    schema="(Tensor x, Tensor(a0!) out_vals, Tensor(a1!) out_idx) -> ()",
)
def _topk_f32_out(x: Tensor, out_vals: Tensor, out_idx: Tensor) -> None:
    assert x.is_cuda and x.dtype == torch.float32
    assert out_vals.dtype == torch.float32 and out_idx.dtype == torch.int32
    assert x.dim() == 2 and out_vals.dim() == 2 and out_idx.dim() == 2
    assert x.stride(-1) == 1 and out_vals.stride(-1) == 1 and out_idx.stride(-1) == 1
    assert out_vals.shape == out_idx.shape
    M, N = x.shape
    M2, k = out_vals.shape
    assert M == M2
    _compile(N, k, get_rocm_arch())(x, out_vals, out_idx, M)


@_topk_f32_out.register_fake
def _topk_f32_out_fake(x, out_vals, out_idx):
    return None


def topk_mfma(x: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """FlyDSL single-wave bitonic topk.

    Returns ``(values, indices)`` where values are in descending order.
    Requires:
        - ``x`` is 2-D f32, last-dim contiguous
        - ``x.size(-1)`` (= N) ∈ {8, 16, 32, 64}
        - ``k`` power of 2, k ≤ N
    """
    assert x.is_cuda and x.dim() == 2 and x.dtype == torch.float32
    assert x.stride(-1) == 1
    M, N = x.shape
    assert N in (8, 16, 32, 64)
    assert k in (1, 2, 4, 8, 16, 32, 64) and k <= N
    out_vals = torch.empty(M, k, device=x.device, dtype=torch.float32)
    out_idx = torch.empty(M, k, device=x.device, dtype=torch.int32)
    _topk_f32_out(x, out_vals, out_idx)
    return out_vals, out_idx


__all__ = ["topk_mfma"]
