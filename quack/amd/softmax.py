# Copyright (c) 2026, AMD.

"""Softmax forward + backward — AMDGPU port of `quack/softmax.py`.

Forward:
    y[i,j] = exp(x[i,j] - max(x[i])) / sum_j(exp(x[i,j] - max))
Backward:
    dx[i,j] = y[i,j] * (dy[i,j] - sum_j(y[i,j] * dy[i,j]))

Both are per-row kernels: one workgroup per row. Scalar load/store path;
handles arbitrary N.
"""

import math as _py_math

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu as _gpu, range_constexpr, math as _fm
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Numeric, Float32, Float16, BFloat16
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch, get_wave_size, torch2flydsl_dtype_map

# Emit an ``scf.if`` for a dynamic condition from a closure body. Using the
# explicit dispatch (rather than a plain ``if`` statement) keeps the AST
# rewriter from threading the SmemPtr scratch handle through as scf.if state —
# a SmemPtr is not an MLIR value, which the upstream rewriter now rejects.
_scf_if = ReplaceIfWithDispatch.scf_if_dispatch


def _elem_type_for(numeric_cls):
    if numeric_cls is Float32:
        return T.f32
    if numeric_cls is Float16:
        return T.f16
    if numeric_cls is BFloat16:
        return T.bf16
    raise ValueError(f"unsupported numeric class: {numeric_cls}")


def _reserve(allocator, num_slots, elem_bytes=4):
    offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = offset + num_slots * elem_bytes
    return offset


def _bufcopy(bits):
    return fx.rocdl.BufferCopy16b() if bits <= 16 else fx.rocdl.BufferCopy32b()


def _build_softmax_fwd(*, N, dtype, arch):
    wave_size = get_wave_size(arch)
    block_threads = 128 if N <= 16384 else 256
    num_waves = block_threads // wave_size
    elem_bits = dtype.width

    sym = "quack_amd_softmax_fwd_smem"
    allocator = SmemAllocator(None, arch=arch, global_sym_name=sym)
    off_max = _reserve(allocator, num_waves)
    off_sum = _reserve(allocator, num_waves)

    @flyc.kernel
    def kernel(X: fx.Tensor, Y: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = _elem_type_for(dtype)
        compute_type = T.f32
        n_i32 = fx.Int32(N)
        neg_inf = arith.constant(float("-inf"), type=compute_type)
        zero_f = arith.constant(0.0, type=compute_type)

        base_ptr = allocator.get_base()
        s_max = SmemPtr(base_ptr, off_max, T.f32, shape=(num_waves,))
        s_sum = SmemPtr(base_ptr, off_sum, T.f32, shape=(num_waves,))
        s_max.get()
        s_sum.get()

        X_buf = fx.rocdl.make_buffer_tensor(X)
        Y_buf = fx.rocdl.make_buffer_tensor(Y)
        row_x = fx.slice(X_buf, (bid, None))
        row_y = fx.slice(Y_buf, (bid, None))
        x_div = fx.logical_divide(row_x, fx.make_layout(1, 1))
        y_div = fx.logical_divide(row_y, fx.make_layout(1, 1))

        copy_atom_x = fx.make_copy_atom(_bufcopy(elem_bits), elem_type)
        x_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

        def _load(div, ca, idx):
            r = fx.memref_alloca(x_reg_ty, reg_lay)
            fx.copy_atom_call(ca, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _store(div, ca, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(x_reg_ty, reg_lay)
            elem_py = Numeric.from_ir_type(x_reg_ty.element_type)
            ts = _vfull(1, elem_py(val), elem_py)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca, r, fx.slice(div, (None, idx)))

        def _wave_reduce(v, op):
            width = fx.Int32(wave_size)
            w = v
            for sh in range_constexpr(int(_py_math.log2(wave_size))):
                off = fx.Int32(wave_size // (2 << sh))
                peer = w.shuffle_xor(off, width)
                w = op(w, peer)
            return w

        def _block_reduce(val, smem, op, init_val):
            if fx.const_expr(num_waves == 1):
                return _wave_reduce(val, op)
            lane = tid % fx.Int32(wave_size)
            wave = tid // fx.Int32(wave_size)
            w0 = _wave_reduce(val, op)

            def _publish():
                smem.store(w0, [ArithValue(wave).index_cast(T.index)])

            _scf_if(lane == fx.Int32(0), _publish)
            _gpu.barrier()

            def _combine():
                in_range = lane < fx.Int32(num_waves)
                lane_safe = in_range.select(lane, fx.Int32(0))
                v = smem.load([ArithValue(lane_safe).index_cast(T.index)])
                v = in_range.select(v, Float32(init_val))
                v = _wave_reduce(v, op)

                def _store_final():
                    smem.store(v, [fx.Index(0)])

                _scf_if(lane == fx.Int32(0), _store_final)

            _scf_if(wave == fx.Int32(0), _combine)
            _gpu.barrier()
            return smem.load([fx.Index(0)])

        # Pass 1: row max
        thread_max = neg_inf
        for base in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base)
            is_valid = idx < n_i32
            idx_safe = is_valid.select(idx, fx.Int32(0))
            x_e = _load(x_div, copy_atom_x, idx_safe)
            x = x_e if dtype is Float32 else x_e.extf(compute_type)
            v = is_valid.select(x, neg_inf)
            thread_max = ArithValue(thread_max).maximumf(v)

        row_max = _block_reduce(
            thread_max, s_max, lambda a, b: a.maximumf(b), float("-inf"),
        )
        row_max_av = ArithValue(row_max)

        # Pass 2: row sum of exp(x - max)
        thread_sum = zero_f
        for base in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base)
            is_valid = idx < n_i32
            idx_safe = is_valid.select(idx, fx.Int32(0))
            x_e = _load(x_div, copy_atom_x, idx_safe)
            x = x_e if dtype is Float32 else x_e.extf(compute_type)
            shifted = ArithValue(x) - row_max_av
            e = _fm.exp(shifted, fastmath="fast")
            contrib = is_valid.select(e, zero_f)
            thread_sum = ArithValue(thread_sum) + contrib

        row_sum = _block_reduce(
            thread_sum, s_sum, lambda a, b: a.addf(b, fastmath="fast"), 0.0,
        )
        inv_sum = Float32(1.0) / ArithValue(row_sum)

        # Pass 3: write y
        for base in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base)
            if arith.cmpi(arith.CmpIPredicate.ult, idx, n_i32):
                x_e = _load(x_div, copy_atom_x, idx)
                x = x_e if dtype is Float32 else x_e.extf(compute_type)
                e = _fm.exp(ArithValue(x) - row_max_av, fastmath="fast")
                y_f32 = e * inv_sum
                y_e = y_f32 if dtype is Float32 else y_f32.truncf(elem_type)
                _store(y_div, copy_atom_x, idx, y_e)

    @flyc.jit
    def launch(X: fx.Tensor, Y: fx.Tensor, M: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        
        kernel(X, Y).launch(grid=(M, 1, 1), block=(block_threads, 1, 1), stream=stream)

    return launch


def _build_softmax_bwd(*, N, dtype, arch):
    wave_size = get_wave_size(arch)
    block_threads = 128 if N <= 16384 else 256
    num_waves = block_threads // wave_size
    elem_bits = dtype.width

    sym = "quack_amd_softmax_bwd_smem"
    allocator = SmemAllocator(None, arch=arch, global_sym_name=sym)
    off = _reserve(allocator, num_waves)

    @flyc.kernel
    def kernel(DY: fx.Tensor, Y: fx.Tensor, DX: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = _elem_type_for(dtype)
        compute_type = T.f32
        n_i32 = fx.Int32(N)
        zero_f = arith.constant(0.0, type=compute_type)

        base_ptr = allocator.get_base()
        s_red = SmemPtr(base_ptr, off, T.f32, shape=(num_waves,))
        s_red.get()

        DY_buf = fx.rocdl.make_buffer_tensor(DY)
        Y_buf = fx.rocdl.make_buffer_tensor(Y)
        DX_buf = fx.rocdl.make_buffer_tensor(DX)
        row_dy = fx.slice(DY_buf, (bid, None))
        row_y = fx.slice(Y_buf, (bid, None))
        row_dx = fx.slice(DX_buf, (bid, None))
        dy_div = fx.logical_divide(row_dy, fx.make_layout(1, 1))
        y_div = fx.logical_divide(row_y, fx.make_layout(1, 1))
        dx_div = fx.logical_divide(row_dx, fx.make_layout(1, 1))

        copy_atom = fx.make_copy_atom(_bufcopy(elem_bits), elem_type)
        reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

        def _load(div, idx):
            r = fx.memref_alloca(reg_ty, reg_lay)
            fx.copy_atom_call(copy_atom, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _store(div, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(reg_ty, reg_lay)
            elem_py = Numeric.from_ir_type(reg_ty.element_type)
            ts = _vfull(1, elem_py(val), elem_py)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(copy_atom, r, fx.slice(div, (None, idx)))

        # Pass 1: sum(y * dy)
        thread_acc = zero_f
        for base in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base)
            is_valid = idx < n_i32
            idx_safe = is_valid.select(idx, fx.Int32(0))
            dy_e = _load(dy_div, idx_safe)
            y_e = _load(y_div, idx_safe)
            dy_f = dy_e if dtype is Float32 else dy_e.extf(compute_type)
            y_f = y_e if dtype is Float32 else y_e.extf(compute_type)
            prod = ArithValue(dy_f) * ArithValue(y_f)
            thread_acc = ArithValue(thread_acc) + is_valid.select(prod, zero_f)

        # Inlined block reduce (same pattern as softmax fwd / rmsnorm)
        def _wave_reduce_add(v):
            width = fx.Int32(wave_size)
            w = v
            for sh in range_constexpr(int(_py_math.log2(wave_size))):
                off_sh = fx.Int32(wave_size // (2 << sh))
                peer = w.shuffle_xor(off_sh, width)
                w = w.addf(peer, fastmath="fast")
            return w

        if fx.const_expr(num_waves == 1):
            dot = _wave_reduce_add(thread_acc)
        else:
            lane = tid % fx.Int32(wave_size)
            wave = tid // fx.Int32(wave_size)
            w0 = _wave_reduce_add(thread_acc)

            def _publish_dot():
                s_red.store(w0, [ArithValue(wave).index_cast(T.index)])

            _scf_if(lane == fx.Int32(0), _publish_dot)
            _gpu.barrier()

            def _combine_dot():
                in_range = lane < fx.Int32(num_waves)
                lane_safe = in_range.select(lane, fx.Int32(0))
                v = s_red.load([ArithValue(lane_safe).index_cast(T.index)])
                v = in_range.select(v, Float32(0.0))
                v = _wave_reduce_add(v)

                def _store_final_dot():
                    s_red.store(v, [fx.Index(0)])

                _scf_if(lane == fx.Int32(0), _store_final_dot)

            _scf_if(wave == fx.Int32(0), _combine_dot)
            _gpu.barrier()
            dot = s_red.load([fx.Index(0)])
        dot_av = ArithValue(dot)

        # Pass 2: dx = y * (dy - dot)
        for base in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base)
            if arith.cmpi(arith.CmpIPredicate.ult, idx, n_i32):
                dy_e = _load(dy_div, idx)
                y_e = _load(y_div, idx)
                dy_f = dy_e if dtype is Float32 else dy_e.extf(compute_type)
                y_f = y_e if dtype is Float32 else y_e.extf(compute_type)
                dx_f = ArithValue(y_f) * (ArithValue(dy_f) - dot_av)
                dx_e = dx_f if dtype is Float32 else dx_f.truncf(elem_type)
                _store(dx_div, idx, dx_e)

    @flyc.jit
    def launch(DY: fx.Tensor, Y: fx.Tensor, DX: fx.Tensor, M: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        
        kernel(DY, Y, DX).launch(grid=(M, 1, 1), block=(block_threads, 1, 1), stream=stream)

    return launch


# ---------------------------------------------------------------------------
# Caches + torch custom ops + public API
# ---------------------------------------------------------------------------


_fwd_cache: dict = {}
_bwd_cache: dict = {}


def _compile_fwd(N, dt, arch):
    key = (N, dt, arch)
    got = _fwd_cache.get(key)
    if got is None:
        got = _build_softmax_fwd(N=N, dtype=torch2flydsl_dtype_map[dt], arch=arch)
        _fwd_cache[key] = got
    return got


def _compile_bwd(N, dt, arch):
    key = (N, dt, arch)
    got = _bwd_cache.get(key)
    if got is None:
        got = _build_softmax_bwd(N=N, dtype=torch2flydsl_dtype_map[dt], arch=arch)
        _bwd_cache[key] = got
    return got


@torch.library.custom_op("quack_amd::_softmax_fwd", mutates_args=("out",))
def _softmax_fwd(x: Tensor, out: Tensor) -> None:
    assert x.is_cuda and out.is_cuda
    assert x.dim() == 2 and out.shape == x.shape
    assert x.stride(-1) == 1 and out.stride(-1) == 1
    M, N = x.shape
    _compile_fwd(N, x.dtype, get_rocm_arch())(x, out, M)


@_softmax_fwd.register_fake
def _softmax_fwd_fake(x, out):
    return None


@torch.library.custom_op("quack_amd::_softmax_bwd", mutates_args=("dx",))
def _softmax_bwd(dy: Tensor, y: Tensor, dx: Tensor) -> None:
    assert dy.is_cuda and y.is_cuda and dx.is_cuda
    assert dy.shape == y.shape == dx.shape
    assert all(t.stride(-1) == 1 for t in (dy, y, dx))
    M, N = dy.shape
    _compile_bwd(N, y.dtype, get_rocm_arch())(dy, y, dx, M)


@_softmax_bwd.register_fake
def _softmax_bwd_fake(dy, y, dx):
    return None


def softmax_fwd(x: Tensor) -> Tensor:
    assert x.is_cuda and x.dim() == 2
    x = x if x.stride(-1) == 1 else x.contiguous()
    out = torch.empty_like(x)
    _softmax_fwd(x, out)
    return out


def softmax_bwd(dy: Tensor, y: Tensor) -> Tensor:
    assert dy.is_cuda and y.is_cuda and dy.shape == y.shape
    dy = dy if dy.stride(-1) == 1 else dy.contiguous()
    y = y if y.stride(-1) == 1 else y.contiguous()
    dx = torch.empty_like(dy)
    _softmax_bwd(dy, y, dx)
    return dx


__all__ = ["softmax_fwd", "softmax_bwd", "_softmax_fwd", "_softmax_bwd"]
