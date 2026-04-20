# Copyright (c) 2026, AMD.

"""RMSNorm forward + backward — AMDGPU port of `quack/rmsnorm.py`.

Forward:
    - 2D input (M, N), row-normalised along N.
    - Optional ``weight`` (gamma) and ``store_rstd``.
    - Scalar load/store path; handles arbitrary N.

Backward:
    - ``dx`` computed in an on-device per-row kernel (one workgroup per row).
    - ``dw`` computed on the host via ``torch.sum((dout * x_hat), dim=0)``.

Missing vs QuACK NVIDIA side (tracked for follow-up):
    LayerNorm mode, residual, bias, per-head layout, runtime ``eps``, kernel
    ``dw`` reduction.
"""

from typing import Optional, Tuple, Type

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Numeric, Float32, Float16, BFloat16
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch, get_wave_size, torch2flydsl_dtype_map
from quack.amd.reduce import block_reduce_add


_EPS = 1e-6
_scf_if = ReplaceIfWithDispatch.scf_if_dispatch


def _elem_type_for(numeric_cls):
    if numeric_cls is Float32:
        return T.f32
    if numeric_cls is Float16:
        return T.f16
    if numeric_cls is BFloat16:
        return T.bf16
    raise ValueError(f"unsupported numeric class: {numeric_cls}")


def _reserve_scratch(allocator: SmemAllocator, num_slots: int, elem_bytes: int) -> int:
    offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = offset + num_slots * elem_bytes
    return offset


def _bufcopy_for(bits: int):
    return fx.rocdl.BufferCopy16b() if bits <= 16 else fx.rocdl.BufferCopy32b()


# ---------------------------------------------------------------------------
# Forward — one kernel per (store_rstd, dtype, weight_dtype, N, arch)
# ---------------------------------------------------------------------------


def _build_rmsnorm_fwd(*, N, dtype, weight_dtype, store_rstd, arch):
    wave_size = get_wave_size(arch)
    block_threads = 128 if N <= 16384 else 256
    num_waves = block_threads // wave_size
    elem_bits = dtype.width
    w_elem_bits = weight_dtype.width

    sym = f"quack_amd_rmsnorm_fwd{'_rstd' if store_rstd else ''}_smem"
    allocator = SmemAllocator(None, arch=arch, global_sym_name=sym)
    red_offset = _reserve_scratch(allocator, num_waves, 4)
    fm_fast = arith.FastMathFlags.fast

    if store_rstd:
        @flyc.kernel
        def kernel(X: fx.Tensor, W: fx.Tensor, Y: fx.Tensor, Rstd: fx.Tensor):
            bid = fx.block_idx.x
            tid = fx.thread_idx.x

            elem_type = _elem_type_for(dtype)
            w_elem_type = _elem_type_for(weight_dtype)
            compute_type = T.f32
            n_float = arith.constant(float(N), type=compute_type)

            base_ptr = allocator.get_base()
            s_red = SmemPtr(base_ptr, red_offset, T.f32, shape=(num_waves,))
            s_red.get()

            X_buf = fx.rocdl.make_buffer_tensor(X)
            W_buf = fx.rocdl.make_buffer_tensor(W)
            Y_buf = fx.rocdl.make_buffer_tensor(Y)
            Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)

            row_x = fx.slice(X_buf, (bid, None))
            row_y = fx.slice(Y_buf, (bid, None))
            x_div = fx.logical_divide(row_x, fx.make_layout(1, 1))
            y_div = fx.logical_divide(row_y, fx.make_layout(1, 1))
            w_div = fx.logical_divide(W_buf, fx.make_layout(1, 1))
            rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))

            copy_atom_x = fx.make_copy_atom(_bufcopy_for(elem_bits), elem_type)
            copy_atom_w = fx.make_copy_atom(_bufcopy_for(w_elem_bits), w_elem_type)
            copy_atom_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
            x_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
            w_reg_ty = fx.MemRefType.get(w_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
            f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
            reg_lay = fx.make_layout(1, 1)

            def _load(div, reg_ty, ca, idx):
                r = fx.memref_alloca(reg_ty, reg_lay)
                fx.copy_atom_call(ca, fx.slice(div, (None, idx)), r)
                return fx.memref_load_vec(r)[0].ir_value()

            def _store(div, reg_ty, ca, idx, val):
                from flydsl.expr.vector import full as _vfull
                r = fx.memref_alloca(reg_ty, reg_lay)
                elem_py = Numeric.from_ir_type(reg_ty.element_type)
                ts = _vfull(1, elem_py(val), elem_py)
                fx.memref_store_vec(ts, r)
                fx.copy_atom_call(ca, r, fx.slice(div, (None, idx)))

            # Pass 1: sum of squares
            c_zero_f = arith.constant(0.0, type=compute_type)
            thread_sumsq = c_zero_f
            for base_idx in range_constexpr(0, N, block_threads):
                idx = tid + fx.Int32(base_idx)
                is_valid = idx < fx.Int32(N)
                idx_safe = is_valid.select(idx, fx.Int32(0))
                x_e = _load(x_div, x_reg_ty, copy_atom_x, idx_safe)
                x = x_e if dtype is Float32 else x_e.extf(compute_type)
                x_av = ArithValue(x)
                x2 = x_av * x_av
                x2_safe = is_valid.select(x2, c_zero_f)
                thread_sumsq = ArithValue(thread_sumsq) + x2_safe

            sum_sq = block_reduce_add(thread_sumsq, s_red, num_waves,
                                      wave_size=wave_size, tid=tid)
            mean_sq = ArithValue(sum_sq) / n_float
            rrms = (mean_sq + _EPS).rsqrt(fastmath=fm_fast)

            if tid == fx.Int32(0):
                _store(rstd_div, f_reg_ty, copy_atom_f, bid, rrms)

            # Pass 2: y = (x * rrms) * w
            for base_idx in range_constexpr(0, N, block_threads):
                idx = tid + fx.Int32(base_idx)
                if arith.cmpi(arith.CmpIPredicate.ult, idx, fx.Int32(N)):
                    x_e = _load(x_div, x_reg_ty, copy_atom_x, idx)
                    w_e = _load(w_div, w_reg_ty, copy_atom_w, idx)
                    x = x_e if dtype is Float32 else x_e.extf(compute_type)
                    w = w_e if weight_dtype is Float32 else w_e.extf(compute_type)
                    y = (ArithValue(x) * rrms) * w
                    y_e = y if dtype is Float32 else y.truncf(elem_type)
                    _store(y_div, x_reg_ty, copy_atom_x, idx, y_e)

        @flyc.jit
        def launch(X: fx.Tensor, W: fx.Tensor, Y: fx.Tensor, Rstd: fx.Tensor,
                   M: fx.Int32, stream: fx.Stream = fx.Stream(None)):
            allocator.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()
            
            kernel(X, W, Y, Rstd).launch(
                grid=(M, 1, 1), block=(block_threads, 1, 1), stream=stream,
            )
    else:
        @flyc.kernel
        def kernel(X: fx.Tensor, W: fx.Tensor, Y: fx.Tensor):
            bid = fx.block_idx.x
            tid = fx.thread_idx.x

            elem_type = _elem_type_for(dtype)
            w_elem_type = _elem_type_for(weight_dtype)
            compute_type = T.f32
            n_float = arith.constant(float(N), type=compute_type)

            base_ptr = allocator.get_base()
            s_red = SmemPtr(base_ptr, red_offset, T.f32, shape=(num_waves,))
            s_red.get()

            X_buf = fx.rocdl.make_buffer_tensor(X)
            W_buf = fx.rocdl.make_buffer_tensor(W)
            Y_buf = fx.rocdl.make_buffer_tensor(Y)

            row_x = fx.slice(X_buf, (bid, None))
            row_y = fx.slice(Y_buf, (bid, None))
            x_div = fx.logical_divide(row_x, fx.make_layout(1, 1))
            y_div = fx.logical_divide(row_y, fx.make_layout(1, 1))
            w_div = fx.logical_divide(W_buf, fx.make_layout(1, 1))

            copy_atom_x = fx.make_copy_atom(_bufcopy_for(elem_bits), elem_type)
            copy_atom_w = fx.make_copy_atom(_bufcopy_for(w_elem_bits), w_elem_type)
            x_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
            w_reg_ty = fx.MemRefType.get(w_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
            reg_lay = fx.make_layout(1, 1)

            def _load(div, reg_ty, ca, idx):
                r = fx.memref_alloca(reg_ty, reg_lay)
                fx.copy_atom_call(ca, fx.slice(div, (None, idx)), r)
                return fx.memref_load_vec(r)[0].ir_value()

            def _store(div, reg_ty, ca, idx, val):
                from flydsl.expr.vector import full as _vfull
                r = fx.memref_alloca(reg_ty, reg_lay)
                elem_py = Numeric.from_ir_type(reg_ty.element_type)
                ts = _vfull(1, elem_py(val), elem_py)
                fx.memref_store_vec(ts, r)
                fx.copy_atom_call(ca, r, fx.slice(div, (None, idx)))

            c_zero_f = arith.constant(0.0, type=compute_type)
            thread_sumsq = c_zero_f
            for base_idx in range_constexpr(0, N, block_threads):
                idx = tid + fx.Int32(base_idx)
                is_valid = idx < fx.Int32(N)
                idx_safe = is_valid.select(idx, fx.Int32(0))
                x_e = _load(x_div, x_reg_ty, copy_atom_x, idx_safe)
                x = x_e if dtype is Float32 else x_e.extf(compute_type)
                x_av = ArithValue(x)
                x2 = x_av * x_av
                x2_safe = is_valid.select(x2, c_zero_f)
                thread_sumsq = ArithValue(thread_sumsq) + x2_safe

            sum_sq = block_reduce_add(thread_sumsq, s_red, num_waves,
                                      wave_size=wave_size, tid=tid)
            mean_sq = ArithValue(sum_sq) / n_float
            rrms = (mean_sq + _EPS).rsqrt(fastmath=fm_fast)

            for base_idx in range_constexpr(0, N, block_threads):
                idx = tid + fx.Int32(base_idx)
                if arith.cmpi(arith.CmpIPredicate.ult, idx, fx.Int32(N)):
                    x_e = _load(x_div, x_reg_ty, copy_atom_x, idx)
                    w_e = _load(w_div, w_reg_ty, copy_atom_w, idx)
                    x = x_e if dtype is Float32 else x_e.extf(compute_type)
                    w = w_e if weight_dtype is Float32 else w_e.extf(compute_type)
                    y = (ArithValue(x) * rrms) * w
                    y_e = y if dtype is Float32 else y.truncf(elem_type)
                    _store(y_div, x_reg_ty, copy_atom_x, idx, y_e)

        @flyc.jit
        def launch(X: fx.Tensor, W: fx.Tensor, Y: fx.Tensor,
                   M: fx.Int32, stream: fx.Stream = fx.Stream(None)):
            allocator.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()
            
            kernel(X, W, Y).launch(
                grid=(M, 1, 1), block=(block_threads, 1, 1), stream=stream,
            )

    return launch


# ---------------------------------------------------------------------------
# Backward dx kernel
# ---------------------------------------------------------------------------


def _build_rmsnorm_dx(*, N, dtype, weight_dtype, arch):
    wave_size = get_wave_size(arch)
    block_threads = 128 if N <= 16384 else 256
    num_waves = block_threads // wave_size
    elem_bits = dtype.width
    w_elem_bits = weight_dtype.width

    sym = "quack_amd_rmsnorm_bwd_dx_smem"
    allocator = SmemAllocator(None, arch=arch, global_sym_name=sym)
    red_offset = _reserve_scratch(allocator, num_waves, 4)
    fm_fast = arith.FastMathFlags.fast

    @flyc.kernel
    def kernel(
        X: fx.Tensor, W: fx.Tensor, DOut: fx.Tensor, Rstd: fx.Tensor, DX: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = _elem_type_for(dtype)
        w_elem_type = _elem_type_for(weight_dtype)
        compute_type = T.f32
        n_float = arith.constant(float(N), type=compute_type)

        base_ptr = allocator.get_base()
        s_red = SmemPtr(base_ptr, red_offset, T.f32, shape=(num_waves,))
        s_red.get()

        X_buf = fx.rocdl.make_buffer_tensor(X)
        W_buf = fx.rocdl.make_buffer_tensor(W)
        DOut_buf = fx.rocdl.make_buffer_tensor(DOut)
        Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
        DX_buf = fx.rocdl.make_buffer_tensor(DX)

        row_x = fx.slice(X_buf, (bid, None))
        row_dout = fx.slice(DOut_buf, (bid, None))
        row_dx = fx.slice(DX_buf, (bid, None))
        x_div = fx.logical_divide(row_x, fx.make_layout(1, 1))
        dout_div = fx.logical_divide(row_dout, fx.make_layout(1, 1))
        dx_div = fx.logical_divide(row_dx, fx.make_layout(1, 1))
        w_div = fx.logical_divide(W_buf, fx.make_layout(1, 1))
        rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))

        copy_atom_x = fx.make_copy_atom(_bufcopy_for(elem_bits), elem_type)
        copy_atom_w = fx.make_copy_atom(_bufcopy_for(w_elem_bits), w_elem_type)
        copy_atom_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        x_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        w_reg_ty = fx.MemRefType.get(w_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

        def _load(div, reg_ty, ca, idx):
            r = fx.memref_alloca(reg_ty, reg_lay)
            fx.copy_atom_call(ca, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _store(div, reg_ty, ca, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(reg_ty, reg_lay)
            elem_py = Numeric.from_ir_type(reg_ty.element_type)
            ts = _vfull(1, elem_py(val), elem_py)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca, r, fx.slice(div, (None, idx)))

        rstd_val = ArithValue(_load(rstd_div, f_reg_ty, copy_atom_f, bid))

        c_zero_f = arith.constant(0.0, type=compute_type)
        thread_acc = c_zero_f
        for base_idx in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base_idx)
            is_valid = idx < fx.Int32(N)
            idx_safe = is_valid.select(idx, fx.Int32(0))
            x_e = _load(x_div, x_reg_ty, copy_atom_x, idx_safe)
            d_e = _load(dout_div, x_reg_ty, copy_atom_x, idx_safe)
            w_e = _load(w_div, w_reg_ty, copy_atom_w, idx_safe)
            x = x_e if dtype is Float32 else x_e.extf(compute_type)
            d = d_e if dtype is Float32 else d_e.extf(compute_type)
            w = w_e if weight_dtype is Float32 else w_e.extf(compute_type)
            x_hat = ArithValue(x) * rstd_val
            wdy = ArithValue(d) * w
            contrib = x_hat * wdy
            contrib_safe = is_valid.select(contrib, c_zero_f)
            thread_acc = ArithValue(thread_acc) + contrib_safe

        # Block reduce inlined (matches FlyDSL's kernels/rmsnorm_kernel.py
        # `block_reduce_add2` idiom). External block_reduce_add behaves
        # differently in this context for reasons still under investigation.
        from flydsl.expr import gpu as _gpu

        def _wave_reduce_add(v):
            import math as _m
            width_i32 = fx.Int32(wave_size)
            w = v
            for sh_exp in range_constexpr(int(_m.log2(wave_size))):
                off = fx.Int32(wave_size // (2 << sh_exp))
                peer = w.shuffle_xor(off, width_i32)
                w = w.addf(peer, fastmath="fast")
            return w

        if num_waves == 1:
            sum_xhat_wdy = _wave_reduce_add(thread_acc)
        else:
            _lane = tid % fx.Int32(wave_size)
            _wave = tid // fx.Int32(wave_size)
            _w0 = _wave_reduce_add(thread_acc)
            if _lane == fx.Int32(0):
                _wave_idx = ArithValue(_wave).index_cast(T.index)
                s_red.store(_w0, [_wave_idx])
            _gpu.barrier()
            if _wave == fx.Int32(0):
                _in_range = _lane < fx.Int32(num_waves)
                _lane_safe = _in_range.select(_lane, fx.Int32(0))
                _lane_safe_idx = ArithValue(_lane_safe).index_cast(T.index)
                _v = s_red.load([_lane_safe_idx])
                _z = Float32(0.0)
                _ww = _in_range.select(_v, _z)
                _ww = _wave_reduce_add(_ww)
                if _lane == fx.Int32(0):
                    s_red.store(_ww, [fx.Index(0)])
            _gpu.barrier()
            sum_xhat_wdy = s_red.load([fx.Index(0)])
        c1 = ArithValue(sum_xhat_wdy) / n_float

        for base_idx in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base_idx)
            if arith.cmpi(arith.CmpIPredicate.ult, idx, fx.Int32(N)):
                x_e = _load(x_div, x_reg_ty, copy_atom_x, idx)
                d_e = _load(dout_div, x_reg_ty, copy_atom_x, idx)
                w_e = _load(w_div, w_reg_ty, copy_atom_w, idx)
                x = x_e if dtype is Float32 else x_e.extf(compute_type)
                d = d_e if dtype is Float32 else d_e.extf(compute_type)
                w = w_e if weight_dtype is Float32 else w_e.extf(compute_type)
                x_hat = ArithValue(x) * rstd_val
                wdy = ArithValue(d) * w
                dx_f32 = (wdy - x_hat * c1) * rstd_val
                dx_e = dx_f32 if dtype is Float32 else dx_f32.truncf(elem_type)
                _store(dx_div, x_reg_ty, copy_atom_x, idx, dx_e)

    @flyc.jit
    def launch(
        X: fx.Tensor, W: fx.Tensor, DOut: fx.Tensor, Rstd: fx.Tensor, DX: fx.Tensor,
        M: fx.Int32, stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        
        kernel(X, W, DOut, Rstd, DX).launch(
            grid=(M, 1, 1), block=(block_threads, 1, 1), stream=stream,
        )

    return launch


# ---------------------------------------------------------------------------
# Caches + torch custom ops + public API
# ---------------------------------------------------------------------------


_fwd_cache: dict = {}
_bwd_cache: dict = {}


def _compile_fwd(N, x_dt, w_dt, store_rstd, arch):
    key = (N, x_dt, w_dt, store_rstd, arch)
    got = _fwd_cache.get(key)
    if got is None:
        got = _build_rmsnorm_fwd(
            N=N, dtype=torch2flydsl_dtype_map[x_dt], weight_dtype=torch2flydsl_dtype_map[w_dt],
            store_rstd=store_rstd, arch=arch,
        )
        _fwd_cache[key] = got
    return got


def _compile_bwd(N, x_dt, w_dt, arch):
    key = (N, x_dt, w_dt, arch)
    got = _bwd_cache.get(key)
    if got is None:
        got = _build_rmsnorm_dx(
            N=N, dtype=torch2flydsl_dtype_map[x_dt],
            weight_dtype=torch2flydsl_dtype_map[w_dt], arch=arch,
        )
        _bwd_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_rmsnorm_fwd",
    mutates_args=("out", "rstd"),
    schema="(Tensor x, Tensor weight, Tensor(a0!) out, Tensor(a1!)? rstd) -> ()",
)
def _rmsnorm_fwd(x: Tensor, weight: Tensor, out: Tensor, rstd: Optional[Tensor]) -> None:
    assert x.is_cuda and weight.is_cuda and out.is_cuda
    assert x.dim() == 2 and out.shape == x.shape
    assert weight.dim() == 1 and weight.size(0) == x.size(-1)
    assert x.stride(-1) == 1 and out.stride(-1) == 1
    if rstd is not None:
        assert rstd.dim() == 1 and rstd.size(0) == x.size(0) and rstd.dtype == torch.float32
    M, N = x.shape
    launcher = _compile_fwd(N, x.dtype, weight.dtype, rstd is not None, get_rocm_arch())
    if rstd is None:
        launcher(x, weight, out, M)
    else:
        launcher(x, weight, out, rstd, M)


@_rmsnorm_fwd.register_fake
def _rmsnorm_fwd_fake(x, weight, out, rstd):
    return None


@torch.library.custom_op(
    "quack_amd::_rmsnorm_bwd_dx",
    mutates_args=("dx",),
    schema="(Tensor x, Tensor weight, Tensor dout, Tensor rstd, Tensor(a0!) dx) -> ()",
)
def _rmsnorm_bwd_dx(
    x: Tensor, weight: Tensor, dout: Tensor, rstd: Tensor, dx: Tensor,
) -> None:
    assert x.is_cuda and weight.is_cuda and dout.is_cuda and rstd.is_cuda and dx.is_cuda
    assert x.dim() == 2 and dout.shape == x.shape and dx.shape == x.shape
    assert weight.dim() == 1 and weight.size(0) == x.size(-1)
    assert rstd.dim() == 1 and rstd.size(0) == x.size(0) and rstd.dtype == torch.float32
    assert all(t.stride(-1) == 1 for t in (x, dout, dx))
    M, N = x.shape
    launcher = _compile_bwd(N, x.dtype, weight.dtype, get_rocm_arch())
    launcher(x, weight, dout, rstd, dx, M)


@_rmsnorm_bwd_dx.register_fake
def _rmsnorm_bwd_dx_fake(x, weight, dout, rstd, dx):
    return None


def rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """RMSNorm forward. Returns ``(out, rstd_or_None)``."""
    assert x.is_cuda and x.dim() == 2
    if eps != _EPS:
        raise NotImplementedError(
            f"rmsnorm_fwd(eps={eps}) differs from the baked-in default {_EPS}; "
            "runtime eps is a later pass."
        )
    x = x if x.stride(-1) == 1 else x.contiguous()
    if weight is None:
        weight = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
    out = torch.empty_like(x)
    rstd = torch.empty(x.size(0), device=x.device, dtype=torch.float32) if store_rstd else None
    _rmsnorm_fwd(x, weight, out, rstd)
    return out, rstd


def rmsnorm_bwd(
    x: Tensor,
    weight: Tensor,
    dout: Tensor,
    rstd: Tensor,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """RMSNorm backward. Returns ``(dx, dw)``.

    ``dw`` is computed with torch as ``sum_over_M(dout * (x * rstd))``. Kernel
    only handles per-row ``dx``; ``dw`` cross-row reduction is a perf follow-up.
    """
    assert weight is not None, "rmsnorm_bwd requires a weight tensor"
    if eps != _EPS:
        raise NotImplementedError("runtime eps is a later pass")
    x = x if x.stride(-1) == 1 else x.contiguous()
    dout = dout if dout.stride(-1) == 1 else dout.contiguous()
    dx = torch.empty_like(x)
    _rmsnorm_bwd_dx(x, weight, dout, rstd, dx)
    x_hat = x.float() * rstd.unsqueeze(1)
    dw = (dout.float() * x_hat).sum(dim=0).to(weight.dtype)
    return dx, dw


__all__ = ["rmsnorm_fwd", "rmsnorm_bwd", "_rmsnorm_fwd", "_rmsnorm_bwd_dx"]
