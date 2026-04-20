# Copyright (c) 2026, AMD.

"""Cross-entropy forward — AMDGPU port of `quack/cross_entropy.py`.

Computes per-row loss = -log(softmax(x)[target]) = max(x) + log(sum(exp(x-max))) - x[target]
and optionally stores the log-sum-exp (``lse``) and/or fused gradient
``dx = softmax(x) - one_hot(target)`` in the same kernel.

Backward (when called separately from stored lse) is provided as a scalar
kernel: ``dx = (exp(x - lse) - one_hot(target)) * dloss_scale``.

Not yet supported (tracked for follow-up):
    - ignore_index
    - per-sample loss scaling
    - label smoothing
"""

import math as _py_math
from typing import Optional, Tuple, Type

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu as _gpu, range_constexpr, math as _fm
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Numeric, Float32, Float16, BFloat16, Int32, Int64
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch, get_wave_size, torch2flydsl_dtype_map


def _elem_type_for(numeric_cls):
    if numeric_cls is Float32:
        return T.f32
    if numeric_cls is Float16:
        return T.f16
    if numeric_cls is BFloat16:
        return T.bf16
    if numeric_cls is Int32:
        return T.i32
    if numeric_cls is Int64:
        return T.i64
    raise ValueError(f"unsupported numeric class: {numeric_cls}")


def _reserve(allocator, num_slots, elem_bytes=4):
    offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = offset + num_slots * elem_bytes
    return offset


def _bufcopy(bits):
    return fx.rocdl.BufferCopy16b() if bits <= 16 else fx.rocdl.BufferCopy32b()


def _build_ce_fwd(*, N, dtype, target_dtype, arch):
    """Forward: per-row (loss, lse). Scalar tile path. Targets are int32 or int64."""
    wave_size = get_wave_size(arch)
    block_threads = 128 if N <= 16384 else 256
    num_waves = block_threads // wave_size
    elem_bits = dtype.width

    sym = "quack_amd_ce_fwd_smem"
    allocator = SmemAllocator(None, arch=arch, global_sym_name=sym)
    off_max = _reserve(allocator, num_waves)
    off_sum = _reserve(allocator, num_waves)

    @flyc.kernel
    def kernel(X: fx.Tensor, TGT: fx.Tensor, Loss: fx.Tensor, Lse: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = _elem_type_for(dtype)
        tgt_type = _elem_type_for(target_dtype)
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
        TGT_buf = fx.rocdl.make_buffer_tensor(TGT)
        Loss_buf = fx.rocdl.make_buffer_tensor(Loss)
        Lse_buf = fx.rocdl.make_buffer_tensor(Lse)
        row_x = fx.slice(X_buf, (bid, None))
        x_div = fx.logical_divide(row_x, fx.make_layout(1, 1))
        tgt_div = fx.logical_divide(TGT_buf, fx.make_layout(1, 1))
        loss_div = fx.logical_divide(Loss_buf, fx.make_layout(1, 1))
        lse_div = fx.logical_divide(Lse_buf, fx.make_layout(1, 1))

        ca_x = fx.make_copy_atom(_bufcopy(elem_bits), elem_type)
        ca_tgt = fx.make_copy_atom(
            fx.rocdl.BufferCopy32b() if target_dtype is Int32 else fx.rocdl.BufferCopy64b(),
            tgt_type,
        )
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        x_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        tgt_reg_ty = fx.MemRefType.get(tgt_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
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

        def _wave_reduce(v, op):
            width = fx.Int32(wave_size)
            w = v
            for sh in range_constexpr(int(_py_math.log2(wave_size))):
                off = fx.Int32(wave_size // (2 << sh))
                peer = w.shuffle_xor(off, width)
                w = op(w, peer)
            return w

        def _block_reduce(val, smem, op, init_val):
            if num_waves == 1:
                return _wave_reduce(val, op)
            lane = tid % fx.Int32(wave_size)
            wave = tid // fx.Int32(wave_size)
            w0 = _wave_reduce(val, op)
            if lane == fx.Int32(0):
                smem.store(w0, [ArithValue(wave).index_cast(T.index)])
            _gpu.barrier()
            if wave == fx.Int32(0):
                in_range = lane < fx.Int32(num_waves)
                lane_safe = in_range.select(lane, fx.Int32(0))
                v = smem.load([ArithValue(lane_safe).index_cast(T.index)])
                v = in_range.select(v, Float32(init_val))
                v = _wave_reduce(v, op)
                if lane == fx.Int32(0):
                    smem.store(v, [fx.Index(0)])
            _gpu.barrier()
            return smem.load([fx.Index(0)])

        # Pass 1: row max
        thread_max = neg_inf
        for base in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base)
            is_valid = idx < n_i32
            idx_safe = is_valid.select(idx, fx.Int32(0))
            x_e = _load(x_div, x_reg_ty, ca_x, idx_safe)
            x = x_e if dtype is Float32 else x_e.extf(compute_type)
            v = is_valid.select(x, neg_inf)
            thread_max = ArithValue(thread_max).maximumf(v)
        row_max = _block_reduce(thread_max, s_max, lambda a, b: a.maximumf(b), float("-inf"))
        row_max_av = ArithValue(row_max)

        # Pass 2: row sum_exp
        thread_sum = zero_f
        for base in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base)
            is_valid = idx < n_i32
            idx_safe = is_valid.select(idx, fx.Int32(0))
            x_e = _load(x_div, x_reg_ty, ca_x, idx_safe)
            x = x_e if dtype is Float32 else x_e.extf(compute_type)
            e = _fm.exp(ArithValue(x) - row_max_av, fastmath="fast")
            thread_sum = ArithValue(thread_sum) + is_valid.select(e, zero_f)
        row_sum = _block_reduce(thread_sum, s_sum, lambda a, b: a.addf(b, fastmath="fast"), 0.0)
        log_sum = _fm.log(ArithValue(row_sum), fastmath="fast")
        lse = row_max_av + log_sum

        # Do the scalar target load, x[target] load, and the two scalar
        # stores all inside a single `tid == 0` guard (matches the pattern
        # that works in `quack.amd.rmsnorm` store_rstd).
        if tid == fx.Int32(0):
            t_e = _load(tgt_div, tgt_reg_ty, ca_tgt, bid)
            if target_dtype is Int64:
                t_i32 = t_e.trunci(T.i32)
            else:
                t_i32 = t_e
            x_t_e = _load(x_div, x_reg_ty, ca_x, t_i32)
            x_t = x_t_e if dtype is Float32 else x_t_e.extf(compute_type)
            loss = lse - ArithValue(x_t)
            _store(loss_div, f_reg_ty, ca_f, bid, loss)
            _store(lse_div, f_reg_ty, ca_f, bid, lse)

    @flyc.jit
    def launch(X: fx.Tensor, TGT: fx.Tensor, Loss: fx.Tensor, Lse: fx.Tensor,
               M: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        
        kernel(X, TGT, Loss, Lse).launch(
            grid=(M, 1, 1), block=(block_threads, 1, 1), stream=stream,
        )

    return launch


# ---------------------------------------------------------------------------
# torch custom op + public API
# ---------------------------------------------------------------------------


_fwd_cache: dict = {}
_bwd_cache: dict = {}


def _compile_fwd(N, dt, tgt_dt, arch):
    key = (N, dt, tgt_dt, arch)
    got = _fwd_cache.get(key)
    if got is None:
        got = _build_ce_fwd(
            N=N, dtype=torch2flydsl_dtype_map[dt],
            target_dtype=torch2flydsl_dtype_map[tgt_dt], arch=arch,
        )
        _fwd_cache[key] = got
    return got


def _compile_bwd(N, dt, tgt_dt, arch):
    key = (N, dt, tgt_dt, arch)
    got = _bwd_cache.get(key)
    if got is None:
        got = _build_ce_bwd_dx(
            N=N, dtype=torch2flydsl_dtype_map[dt],
            target_dtype=torch2flydsl_dtype_map[tgt_dt], arch=arch,
        )
        _bwd_cache[key] = got
    return got


# ---------------------------------------------------------------------------
# Backward kernel — per-row `dx = (exp(x - lse) - one_hot(target)) * dloss`.
# ---------------------------------------------------------------------------


def _build_ce_bwd_dx(*, N, dtype, target_dtype, arch):
    wave_size = get_wave_size(arch)
    block_threads = 128 if N <= 16384 else 256
    elem_bits = dtype.width

    sym = "quack_amd_ce_bwd_dx_smem"
    allocator = SmemAllocator(None, arch=arch, global_sym_name=sym)
    # No reduction needed — dx is per-element, driven only by lse[bid] + target[bid].

    @flyc.kernel
    def kernel(
        X: fx.Tensor, TGT: fx.Tensor, Lse: fx.Tensor, DLoss: fx.Tensor, DX: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = _elem_type_for(dtype)
        tgt_type = _elem_type_for(target_dtype)
        compute_type = T.f32
        n_i32 = fx.Int32(N)

        base_ptr = allocator.get_base()
        # allocator.finalize() handles the trivial empty-LDS case.

        X_buf = fx.rocdl.make_buffer_tensor(X)
        TGT_buf = fx.rocdl.make_buffer_tensor(TGT)
        Lse_buf = fx.rocdl.make_buffer_tensor(Lse)
        DLoss_buf = fx.rocdl.make_buffer_tensor(DLoss)
        DX_buf = fx.rocdl.make_buffer_tensor(DX)

        row_x = fx.slice(X_buf, (bid, None))
        row_dx = fx.slice(DX_buf, (bid, None))
        x_div = fx.logical_divide(row_x, fx.make_layout(1, 1))
        dx_div = fx.logical_divide(row_dx, fx.make_layout(1, 1))
        tgt_div = fx.logical_divide(TGT_buf, fx.make_layout(1, 1))
        lse_div = fx.logical_divide(Lse_buf, fx.make_layout(1, 1))
        dloss_div = fx.logical_divide(DLoss_buf, fx.make_layout(1, 1))

        ca_x = fx.make_copy_atom(_bufcopy(elem_bits), elem_type)
        ca_tgt = fx.make_copy_atom(
            fx.rocdl.BufferCopy32b() if target_dtype is Int32 else fx.rocdl.BufferCopy64b(),
            tgt_type,
        )
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        x_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        tgt_reg_ty = fx.MemRefType.get(tgt_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
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

        # Each thread loads row-level scalars once: lse, dloss, target.
        lse_val = ArithValue(_load(lse_div, f_reg_ty, ca_f, bid))
        dloss_val = ArithValue(_load(dloss_div, f_reg_ty, ca_f, bid))
        t_e = _load(tgt_div, tgt_reg_ty, ca_tgt, bid)
        t_i32 = t_e.trunci(T.i32) if target_dtype is Int64 else t_e

        # Per-column: dx[m, j] = (softmax(x)[m, j] - (j == target[m])) * dloss[m]
        #           = (exp(x[m, j] - lse[m]) - (j == target[m])) * dloss[m]
        for base_idx in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base_idx)
            if arith.cmpi(arith.CmpIPredicate.ult, idx, n_i32):
                x_e = _load(x_div, x_reg_ty, ca_x, idx)
                x = x_e if dtype is Float32 else x_e.extf(compute_type)
                softmax_val = _fm.exp(ArithValue(x) - lse_val, fastmath="fast")
                is_target = arith.cmpi(arith.CmpIPredicate.eq, idx, t_i32)
                one_or_zero = is_target.select(ArithValue(Float32(1.0)), ArithValue(Float32(0.0)))
                dx_f32 = (softmax_val - one_or_zero) * dloss_val
                dx_e = dx_f32 if dtype is Float32 else dx_f32.truncf(elem_type)
                _store(dx_div, x_reg_ty, ca_x, idx, dx_e)

    @flyc.jit
    def launch(
        X: fx.Tensor, TGT: fx.Tensor, Lse: fx.Tensor, DLoss: fx.Tensor, DX: fx.Tensor,
        M: fx.Int32, stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(X, TGT, Lse, DLoss, DX).launch(
            grid=(M, 1, 1), block=(block_threads, 1, 1), stream=stream,
        )

    return launch


@torch.library.custom_op(
    "quack_amd::_cross_entropy_fwd",
    mutates_args=("loss", "lse"),
    schema="(Tensor x, Tensor target, Tensor(a0!) loss, Tensor(a1!) lse) -> ()",
)
def _cross_entropy_fwd(x: Tensor, target: Tensor, loss: Tensor, lse: Tensor) -> None:
    assert x.is_cuda and target.is_cuda and loss.is_cuda and lse.is_cuda
    assert x.dim() == 2 and target.dim() == 1 and target.size(0) == x.size(0)
    assert loss.dim() == 1 and lse.dim() == 1
    assert loss.size(0) == x.size(0) and lse.size(0) == x.size(0)
    assert x.stride(-1) == 1
    M, N = x.shape
    _compile_fwd(N, x.dtype, target.dtype, get_rocm_arch())(x, target, loss, lse, M)


@_cross_entropy_fwd.register_fake
def _cross_entropy_fwd_fake(x, target, loss, lse):
    return None


def cross_entropy_fwd(
    x: Tensor,
    target: Tensor,
    return_lse: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Cross-entropy loss per row. Returns ``(loss, lse_or_None)``.

    ``target`` must be int32 or int64 and 1D of length ``x.size(0)``.
    Loss is float32 regardless of ``x`` dtype.
    """
    assert x.is_cuda and x.dim() == 2
    x = x if x.stride(-1) == 1 else x.contiguous()
    loss = torch.empty(x.size(0), device=x.device, dtype=torch.float32)
    lse = torch.empty(x.size(0), device=x.device, dtype=torch.float32)
    _cross_entropy_fwd(x, target, loss, lse)
    return loss, (lse if return_lse else None)


@torch.library.custom_op(
    "quack_amd::_cross_entropy_bwd_dx",
    mutates_args=("dx",),
    schema="(Tensor x, Tensor target, Tensor lse, Tensor dloss, Tensor(a0!) dx) -> ()",
)
def _cross_entropy_bwd_dx(
    x: Tensor, target: Tensor, lse: Tensor, dloss: Tensor, dx: Tensor,
) -> None:
    assert x.is_cuda and target.is_cuda and lse.is_cuda and dloss.is_cuda and dx.is_cuda
    assert x.dim() == 2 and dx.shape == x.shape
    assert target.dim() == 1 and target.size(0) == x.size(0)
    assert lse.dim() == 1 and lse.size(0) == x.size(0) and lse.dtype == torch.float32
    assert dloss.dim() == 1 and dloss.size(0) == x.size(0) and dloss.dtype == torch.float32
    assert x.stride(-1) == 1 and dx.stride(-1) == 1
    M, N = x.shape
    _compile_bwd(N, x.dtype, target.dtype, get_rocm_arch())(x, target, lse, dloss, dx, M)


@_cross_entropy_bwd_dx.register_fake
def _cross_entropy_bwd_dx_fake(x, target, lse, dloss, dx):
    return None


def cross_entropy_bwd(
    x: Tensor,
    target: Tensor,
    lse: Tensor,
    dloss: Optional[Tensor] = None,
) -> Tensor:
    """Gradient of cross-entropy wrt x.

    ``dx[m, j] = (softmax(x)[m, j] - (j == target[m])) * dloss[m]``

    If ``dloss`` is None, defaults to 1 (unscaled gradient).
    """
    assert x.is_cuda and x.dim() == 2
    x = x if x.stride(-1) == 1 else x.contiguous()
    if dloss is None:
        dloss = torch.ones(x.size(0), device=x.device, dtype=torch.float32)
    elif dloss.dtype != torch.float32:
        dloss = dloss.to(torch.float32)
    dx = torch.empty_like(x)
    _cross_entropy_bwd_dx(x, target, lse, dloss, dx)
    return dx


__all__ = [
    "cross_entropy_fwd", "cross_entropy_bwd",
    "_cross_entropy_fwd", "_cross_entropy_bwd_dx",
]
