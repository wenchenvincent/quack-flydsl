# Copyright (c) 2026, AMD.

"""LDS-based bitonic-sort top-k for N > 64.

Extends ``quack.amd.topk_kernel`` (single-wave, N ≤ 64) to larger rows
(N up to 4096). Strategy:

  - One workgroup per row (256 threads).
  - Load the full row + synthetic indices into LDS — 8N bytes per row,
    well under LDS budget (N=4096 → 32 KiB).
  - Standard ascending bitonic sort in LDS: for each ``k_outer`` in
    {2, 4, ..., N}, and each ``stride`` in {k_outer/2, ..., 1}, every
    worker w in [0, N/2) does one compare-swap of positions (p0, p1 = p0^stride).
  - Workers are spread across the 256 threads: thread ``tid`` runs
    ``N/2 / 256`` compare-swaps per stage.
  - After sort: lanes [N-k, N-1] hold the top k in ascending order;
    write them to the output in descending order.

Bitonic is a full O(N log²N) sort — not the fastest for top-k with
k ≪ N but simple and correct. A truncated partial-sort / per-lane
top-k + cross-lane merge is a follow-up once correctness is locked in.
"""

import math as _py_math
from typing import Tuple

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, range_constexpr, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32, Int32
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch


_BLOCK_THREADS = 256


def _align(x, a):
    return (x + a - 1) & ~(a - 1)


def _build_topk_lds_f32(*, N, k, arch):
    assert N in (128, 256, 512, 1024, 2048, 4096), f"N must be 128..4096 pow2, got {N}"
    assert k <= 128 and k <= N and (k & (k - 1)) == 0
    log2_N = int(_py_math.log2(N))
    workers = N // 2
    workers_per_thread = max(workers // _BLOCK_THREADS, 1)
    threads_used = min(workers, _BLOCK_THREADS)
    vals_bytes = N * 4
    idxs_bytes = N * 4

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_topk_lds_f32_{N}_{k}_smem",
    )
    vals_off = _align(allocator.ptr, 16); allocator.ptr = vals_off + vals_bytes
    idxs_off = _align(allocator.ptr, 16); allocator.ptr = idxs_off + idxs_bytes

    @flyc.kernel
    def kernel(X: fx.Tensor, OutVals: fx.Tensor, OutIdx: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

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

        base_ptr = allocator.get_base()
        vals_lds = SmemPtr(base_ptr, vals_off, T.f32, shape=(N,))
        idxs_lds = SmemPtr(base_ptr, idxs_off, T.i32, shape=(N,))
        vals_lds.get(); idxs_lds.get()

        def _idx(i):
            return ArithValue(i).index_cast(T.index) if hasattr(i, "index_cast") else fx.Index(i)

        def _load_f32(idx):
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            fx.copy_atom_call(ca_f, fx.slice(x_div, (None, idx)), r)
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

        # --- Load row into LDS (cooperative) ---
        # N / BLOCK_THREADS elements per thread (always >= 1 since N >= 128).
        ELEMS_PER_THREAD_LOAD = max(N // _BLOCK_THREADS, 1)
        active_load = tid < fx.Int32(min(_BLOCK_THREADS, N))
        for i in range_constexpr(ELEMS_PER_THREAD_LOAD):
            pos = tid + fx.Int32(i * _BLOCK_THREADS)
            in_range = pos < fx.Int32(N)
            pos_safe = in_range.select(pos, fx.Int32(0))
            v = _load_f32(pos_safe)
            vals_lds.store(v, [_idx(pos_safe)])
            idxs_lds.store(pos_safe, [_idx(pos_safe)])
        gpu.barrier()

        # --- Bitonic sort (ascending) ---
        for stage in range_constexpr(log2_N):
            k_outer = 1 << (stage + 1)
            for sub in range_constexpr(stage + 1):
                j = k_outer >> (sub + 1)
                # Each thread does `workers_per_thread` compare-swaps.
                for w_local in range_constexpr(workers_per_thread):
                    w = tid + fx.Int32(w_local * _BLOCK_THREADS)
                    if w < fx.Int32(workers):
                        # p0 = ((w & ~(j-1)) << 1) | (w & (j-1))
                        low_mask = fx.Int32(j - 1) if j > 0 else fx.Int32(0)
                        low = w & low_mask
                        high = (w & ~low_mask) << fx.Int32(1)
                        p0 = high | low
                        p1 = p0 | fx.Int32(j)
                        v0 = vals_lds.load([_idx(p0)])
                        v1 = vals_lds.load([_idx(p1)])
                        i0 = idxs_lds.load([_idx(p0)])
                        i1 = idxs_lds.load([_idx(p1)])
                        ascending = (p0 & fx.Int32(k_outer)) == fx.Int32(0)
                        v0_f = ArithValue(v0)
                        v1_f = ArithValue(v1)
                        v0_bigger = v0_f > v1_f
                        # swap when: ascending & v0>v1, OR descending & v0<v1
                        do_swap = ascending.select(v0_bigger, ~v0_bigger)
                        new_v0 = do_swap.select(v1, v0)
                        new_v1 = do_swap.select(v0, v1)
                        new_i0 = do_swap.select(i1, i0)
                        new_i1 = do_swap.select(i0, i1)
                        vals_lds.store(new_v0, [_idx(p0)])
                        vals_lds.store(new_v1, [_idx(p1)])
                        idxs_lds.store(new_i0, [_idx(p0)])
                        idxs_lds.store(new_i1, [_idx(p1)])
                gpu.barrier()

        # --- Write top-k (descending) ---
        # Lanes [N-k, N-1] in sorted array are the top-k in ascending
        # order; output position = (N-1) - lane.
        K_PER_THREAD = max((k + _BLOCK_THREADS - 1) // _BLOCK_THREADS, 1)
        for i in range_constexpr(K_PER_THREAD):
            out_pos = tid + fx.Int32(i * _BLOCK_THREADS)
            if out_pos < fx.Int32(k):
                pos_in_sorted = fx.Int32(N - 1) - out_pos
                v = vals_lds.load([_idx(pos_in_sorted)])
                ix = idxs_lds.load([_idx(pos_in_sorted)])
                _store_f32(v_div, out_pos, ArithValue(v))
                _store_i32(i_div, out_pos, ArithValue(ix))

    @flyc.jit
    def launch(X: fx.Tensor, OutVals: fx.Tensor, OutIdx: fx.Tensor,
               M: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(X, OutVals, OutIdx).launch(
            grid=(M, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream,
        )

    return launch


_kernel_cache: dict = {}


def _compile(N, k, arch):
    key = (N, k, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_topk_lds_f32(N=N, k=k, arch=arch)
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_topk_lds_f32_out",
    mutates_args=("out_vals", "out_idx"),
    schema="(Tensor x, Tensor(a0!) out_vals, Tensor(a1!) out_idx) -> ()",
)
def _topk_lds_f32_out(x: Tensor, out_vals: Tensor, out_idx: Tensor) -> None:
    assert x.is_cuda and x.dtype == torch.float32
    assert out_vals.dtype == torch.float32 and out_idx.dtype == torch.int32
    assert x.dim() == 2 and out_vals.dim() == 2 and out_idx.dim() == 2
    assert x.stride(-1) == 1 and out_vals.stride(-1) == 1 and out_idx.stride(-1) == 1
    M, N = x.shape
    M2, k = out_vals.shape
    assert M == M2
    _compile(N, k, get_rocm_arch())(x, out_vals, out_idx, M)


@_topk_lds_f32_out.register_fake
def _topk_lds_f32_out_fake(x, out_vals, out_idx):
    return None


def topk_lds(x: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """LDS-based FlyDSL bitonic topk for N in {128, 256, 512, 1024, 2048, 4096}.

    Requires ``x`` 2-D f32, last-dim contiguous, N power-of-2 in range.
    ``k`` must be power-of-2 and ≤ min(N, 128).
    """
    assert x.is_cuda and x.dim() == 2 and x.dtype == torch.float32
    assert x.stride(-1) == 1
    M, N = x.shape
    assert N in (128, 256, 512, 1024, 2048, 4096)
    assert k <= 128 and k <= N and (k & (k - 1)) == 0
    out_vals = torch.empty(M, k, device=x.device, dtype=torch.float32)
    out_idx = torch.empty(M, k, device=x.device, dtype=torch.int32)
    _topk_lds_f32_out(x, out_vals, out_idx)
    return out_vals, out_idx


__all__ = ["topk_lds"]
