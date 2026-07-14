# Copyright (c) 2026, AMD.

"""Cross-entropy forward — AMDGPU port of `quack/cross_entropy.py`.

Computes per-row loss = -log(softmax(x)[target]) = max(x) + log(sum(exp(x-max))) - x[target]
and optionally stores the log-sum-exp (``lse``) and/or fused gradient
``dx = softmax(x) - one_hot(target)`` in the same kernel.

Backward (when called separately from stored lse) is provided as a scalar
kernel: ``dx = (exp(x - lse) - one_hot(target)) * dloss_scale``.

Training-compat kwargs (match ``torch.nn.functional.cross_entropy``):
    - ``ignore_index``: rows with ``target == ignore_index`` emit 0 loss
      and 0 gradient. Default ``-100``.
    - ``label_smoothing``: mixes ``α * (lse - mean(x))`` into the loss
      and produces ``smooth_target = (1-α) one_hot + α/N`` for dx.
    - ``loss_weight``: optional ``(M,)`` f32 multiplier. Folded into
      the scale so it costs nothing extra in the kernel.

All three features are compile-time / runtime specialized: default
calls with no features set compile the same kernel body as before.
"""

import math as _py_math
from typing import Optional, Tuple

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
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch, get_wave_size, torch2flydsl_dtype_map

# Emit an ``scf.if`` from a closure body for a dynamic condition. Avoids the
# upstream AST rewriter threading the SmemPtr scratch handle through as scf.if
# state (a SmemPtr is not an MLIR value, which the rewriter now rejects).
_scf_if = ReplaceIfWithDispatch.scf_if_dispatch


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


def _build_ce_fwd(
    *, N, dtype, target_dtype, arch,
    has_smoothing: bool = False, has_loss_weight: bool = False,
):
    """Forward: per-row (loss, lse). Scalar tile path. Targets are int32 or int64.

    ``has_smoothing`` / ``has_loss_weight`` compile-time flags specialize
    the kernel to skip the extra reduction / per-row load when the
    feature is unused.
    """
    wave_size = get_wave_size(arch)
    block_threads = 128 if N <= 16384 else 256
    num_waves = block_threads // wave_size
    elem_bits = dtype.width

    sym_tag = ""
    if fx.const_expr(has_smoothing):
        sym_tag += "_sm"
    if fx.const_expr(has_loss_weight):
        sym_tag += "_lw"
    sym = f"quack_amd_ce_fwd_smem{sym_tag}"
    allocator = SmemAllocator(None, arch=arch, global_sym_name=sym)
    off_max = _reserve(allocator, num_waves)
    off_sum = _reserve(allocator, num_waves)
    off_sum_x = _reserve(allocator, num_waves) if has_smoothing else None

    @flyc.kernel
    def kernel(
        X: fx.Tensor, TGT: fx.Tensor, Loss: fx.Tensor, Lse: fx.Tensor,
        LossWeight: fx.Tensor,
        ignore_index: fx.Int32, smoothing: fx.Float32,
    ):
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
        if fx.const_expr(has_smoothing):
            s_sum_x = SmemPtr(base_ptr, off_sum_x, T.f32, shape=(num_waves,))
            s_sum_x.get()

        X_buf = fx.rocdl.make_buffer_tensor(X)
        TGT_buf = fx.rocdl.make_buffer_tensor(TGT)
        Loss_buf = fx.rocdl.make_buffer_tensor(Loss)
        Lse_buf = fx.rocdl.make_buffer_tensor(Lse)
        if fx.const_expr(has_loss_weight):
            LW_buf = fx.rocdl.make_buffer_tensor(LossWeight)
            lw_div = fx.logical_divide(LW_buf, fx.make_layout(1, 1))
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
            x_e = _load(x_div, x_reg_ty, ca_x, idx_safe)
            x = x_e if dtype is Float32 else x_e.extf(compute_type)
            v = is_valid.select(x, neg_inf)
            thread_max = ArithValue(thread_max).maximumf(v)
        row_max = _block_reduce(thread_max, s_max, lambda a, b: a.maximumf(b), float("-inf"))
        row_max_av = ArithValue(row_max)

        # Pass 2: row sum_exp (+ optional sum_x for label smoothing).
        thread_sum = zero_f
        if fx.const_expr(has_smoothing):
            thread_sum_x = zero_f
        for base in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base)
            is_valid = idx < n_i32
            idx_safe = is_valid.select(idx, fx.Int32(0))
            x_e = _load(x_div, x_reg_ty, ca_x, idx_safe)
            x = x_e if dtype is Float32 else x_e.extf(compute_type)
            e = _fm.exp(ArithValue(x) - row_max_av, fastmath="fast")
            thread_sum = ArithValue(thread_sum) + is_valid.select(e, zero_f)
            if fx.const_expr(has_smoothing):
                thread_sum_x = ArithValue(thread_sum_x) + is_valid.select(ArithValue(x), zero_f)
        row_sum = _block_reduce(thread_sum, s_sum, lambda a, b: a.addf(b, fastmath="fast"), 0.0)
        log_sum = _fm.log(ArithValue(row_sum), fastmath="fast")
        lse = row_max_av + log_sum
        if fx.const_expr(has_smoothing):
            row_sum_x = _block_reduce(
                thread_sum_x, s_sum_x,
                lambda a, b: a.addf(b, fastmath="fast"), 0.0,
            )
            mean_x = ArithValue(row_sum_x) * ArithValue(Float32(1.0 / N))

        # Do the scalar target load, x[target] load, and the two scalar
        # stores all inside a single `tid == 0` guard (matches the pattern
        # that works in `quack.amd.rmsnorm` store_rstd).
        if tid == fx.Int32(0):
            t_e = _load(tgt_div, tgt_reg_ty, ca_tgt, bid)
            if fx.const_expr(target_dtype is Int64):
                t_i32 = t_e.trunci(T.i32)
            else:
                t_i32 = t_e
            is_ignore = arith.cmpi(arith.CmpIPredicate.eq, t_i32, ignore_index)
            t_i32_safe = is_ignore.select(fx.Int32(0), ArithValue(t_i32))
            x_t_e = _load(x_div, x_reg_ty, ca_x, t_i32_safe)
            x_t = x_t_e if dtype is Float32 else x_t_e.extf(compute_type)
            lse_av = ArithValue(lse)
            nll = lse_av - ArithValue(x_t)
            if fx.const_expr(has_smoothing):
                alpha = ArithValue(smoothing)
                one_minus_alpha = ArithValue(Float32(1.0)) - alpha
                loss = one_minus_alpha * nll + alpha * (lse_av - mean_x)
            else:
                loss = nll
            if fx.const_expr(has_loss_weight):
                loss = loss * ArithValue(_load(lw_div, f_reg_ty, ca_f, bid))
            zero_av = ArithValue(Float32(0.0))
            loss = is_ignore.select(zero_av, loss)
            _store(loss_div, f_reg_ty, ca_f, bid, loss)
            _store(lse_div, f_reg_ty, ca_f, bid, lse)

    @flyc.jit
    def launch(
        X: fx.Tensor, TGT: fx.Tensor, Loss: fx.Tensor, Lse: fx.Tensor,
        LossWeight: fx.Tensor,
        ignore_index: fx.Int32, smoothing: fx.Float32,
        M: fx.Int32, stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(X, TGT, Loss, Lse, LossWeight, ignore_index, smoothing).launch(
            grid=(M, 1, 1), block=(block_threads, 1, 1), stream=stream,
        )

    return launch


# ---------------------------------------------------------------------------
# torch custom op + public API
# ---------------------------------------------------------------------------


_fwd_cache: dict = {}
_bwd_cache: dict = {}


def _compile_fwd(N, dt, tgt_dt, arch, has_smoothing, has_loss_weight):
    key = (N, dt, tgt_dt, arch, has_smoothing, has_loss_weight)
    got = _fwd_cache.get(key)
    if got is None:
        got = _build_ce_fwd(
            N=N, dtype=torch2flydsl_dtype_map[dt],
            target_dtype=torch2flydsl_dtype_map[tgt_dt], arch=arch,
            has_smoothing=has_smoothing, has_loss_weight=has_loss_weight,
        )
        _fwd_cache[key] = got
    return got


def _compile_bwd(N, dt, tgt_dt, arch, has_smoothing, has_loss_weight):
    key = (N, dt, tgt_dt, arch, has_smoothing, has_loss_weight)
    got = _bwd_cache.get(key)
    if got is None:
        got = _build_ce_bwd_dx(
            N=N, dtype=torch2flydsl_dtype_map[dt],
            target_dtype=torch2flydsl_dtype_map[tgt_dt], arch=arch,
            has_smoothing=has_smoothing, has_loss_weight=has_loss_weight,
        )
        _bwd_cache[key] = got
    return got


# ---------------------------------------------------------------------------
# Backward kernel — per-row `dx = (exp(x - lse) - one_hot(target)) * dloss`.
# ---------------------------------------------------------------------------


def _build_ce_bwd_dx(
    *, N, dtype, target_dtype, arch,
    has_smoothing: bool = False, has_loss_weight: bool = False,
):
    wave_size = get_wave_size(arch)
    block_threads = 128 if N <= 16384 else 256
    elem_bits = dtype.width

    sym_tag = ""
    if fx.const_expr(has_smoothing):
        sym_tag += "_sm"
    if fx.const_expr(has_loss_weight):
        sym_tag += "_lw"
    sym = f"quack_amd_ce_bwd_dx_smem{sym_tag}"
    allocator = SmemAllocator(None, arch=arch, global_sym_name=sym)
    # No reduction needed — dx is per-element, driven only by lse[bid] + target[bid].

    @flyc.kernel
    def kernel(
        X: fx.Tensor, TGT: fx.Tensor, Lse: fx.Tensor, DLoss: fx.Tensor, DX: fx.Tensor,
        LossWeight: fx.Tensor,
        ignore_index: fx.Int32, smoothing: fx.Float32,
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
        if fx.const_expr(has_loss_weight):
            LW_buf = fx.rocdl.make_buffer_tensor(LossWeight)
            lw_div = fx.logical_divide(LW_buf, fx.make_layout(1, 1))

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

        # Each thread loads row-level scalars once: lse, dloss, target,
        # optional loss_weight. Fold ignore_index + loss_weight into a
        # single ``scale`` so dx stays unconditional in the inner loop.
        lse_val = ArithValue(_load(lse_div, f_reg_ty, ca_f, bid))
        dloss_val = ArithValue(_load(dloss_div, f_reg_ty, ca_f, bid))
        t_e = _load(tgt_div, tgt_reg_ty, ca_tgt, bid)
        t_i32 = t_e.trunci(T.i32) if target_dtype is Int64 else t_e
        is_ignore = arith.cmpi(arith.CmpIPredicate.eq, t_i32, ignore_index)
        zero_av = ArithValue(Float32(0.0))
        scale_val = dloss_val
        if fx.const_expr(has_loss_weight):
            lw_val = ArithValue(_load(lw_div, f_reg_ty, ca_f, bid))
            scale_val = scale_val * lw_val
        scale_val = is_ignore.select(zero_av, scale_val)
        if fx.const_expr(has_smoothing):
            alpha = ArithValue(smoothing)
            one_minus_alpha_av = ArithValue(Float32(1.0)) - alpha
            alpha_over_n = alpha * ArithValue(Float32(1.0 / N))

        # Per-column: dx[m, j] = (softmax(x)[m, j] - smooth_target[j]) * scale
        #             smooth_target[j] = α/N + (j == t ? 1-α : 0)
        #             smooth_target[j] reduces to one_hot when α = 0.
        for base_idx in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base_idx)
            if arith.cmpi(arith.CmpIPredicate.ult, idx, n_i32):
                x_e = _load(x_div, x_reg_ty, ca_x, idx)
                x = x_e if dtype is Float32 else x_e.extf(compute_type)
                softmax_val = _fm.exp(ArithValue(x) - lse_val, fastmath="fast")
                is_target = arith.cmpi(arith.CmpIPredicate.eq, idx, t_i32)
                if fx.const_expr(has_smoothing):
                    match_bonus = is_target.select(one_minus_alpha_av, zero_av)
                    tgt_d = alpha_over_n + match_bonus
                else:
                    tgt_d = is_target.select(ArithValue(Float32(1.0)), zero_av)
                dx_f32 = (softmax_val - tgt_d) * scale_val
                dx_e = dx_f32 if dtype is Float32 else dx_f32.truncf(elem_type)
                _store(dx_div, x_reg_ty, ca_x, idx, dx_e)

    @flyc.jit
    def launch(
        X: fx.Tensor, TGT: fx.Tensor, Lse: fx.Tensor, DLoss: fx.Tensor, DX: fx.Tensor,
        LossWeight: fx.Tensor,
        ignore_index: fx.Int32, smoothing: fx.Float32,
        M: fx.Int32, stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(X, TGT, Lse, DLoss, DX, LossWeight, ignore_index, smoothing).launch(
            grid=(M, 1, 1), block=(block_threads, 1, 1), stream=stream,
        )

    return launch


# ---------------------------------------------------------------------------
# Fused forward + backward (inplace) — per-row (loss, lse) AND per-element
# dx = softmax(x) - one_hot(target) in a single pass. dx may alias x (the
# common "inplace backward" case: the logits buffer becomes dlogits).
#
# Math: LSE and softmax are computed once; the same exp(x - lse) that feeds
# the forward pass's sum is reused as the softmax value for the backward
# write-back. Saves one full HBM pass over the (M, N) logits tensor vs
# calling fwd and bwd separately.
#
# dloss is implicit = 1.0 here (caller scales post-hoc if needed — e.g.
# divide by count for a 'mean' reduction). Keeping it constant-1 lets the
# kernel's hot loop drop the extra multiply.
# ---------------------------------------------------------------------------


def _build_ce_fwd_bwd(
    *, N, dtype, target_dtype, arch, M_hint=0,
    has_smoothing: bool = False, has_loss_weight: bool = False,
):
    """Forward + inplace-backward fused. Writes loss, lse, and dx in one pass.

    ``M_hint`` is baked into the launch grid as a Python int (not an
    ``fx.Int32`` arg) — works around the FlyDSL JIT's grid-dim
    specialisation quirk where the first M for ``grid=(M, 1, 1)``
    freezes in the cached binary. With a Python int, each unique
    ``M_hint`` triggers a fresh compile.

    ``has_smoothing`` / ``has_loss_weight`` are compile-time specializers.
    When off, the kernel drops the label-smoothing / per-row weight
    branches entirely — the default `ignore_index` path stays lean.
    """
    wave_size = get_wave_size(arch)
    block_threads = 128 if N <= 16384 else 256
    num_waves = block_threads // wave_size
    elem_bits = dtype.width

    sym_tag = f"m{M_hint}"
    if fx.const_expr(has_smoothing):
        sym_tag += "_sm"
    if fx.const_expr(has_loss_weight):
        sym_tag += "_lw"
    sym = f"quack_amd_ce_fwd_bwd_smem_{sym_tag}"
    allocator = SmemAllocator(None, arch=arch, global_sym_name=sym)
    off_max = _reserve(allocator, num_waves)
    off_sum = _reserve(allocator, num_waves)
    off_sum_x = _reserve(allocator, num_waves) if has_smoothing else None

    @flyc.kernel
    def kernel(
        X: fx.Tensor, TGT: fx.Tensor,
        Loss: fx.Tensor, Lse: fx.Tensor, DX: fx.Tensor,
        LossWeight: fx.Tensor,
        ignore_index: fx.Int32, smoothing: fx.Float32,
    ):
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
        s_max.get(); s_sum.get()
        if fx.const_expr(has_smoothing):
            s_sum_x = SmemPtr(base_ptr, off_sum_x, T.f32, shape=(num_waves,))
            s_sum_x.get()

        X_buf = fx.rocdl.make_buffer_tensor(X)
        TGT_buf = fx.rocdl.make_buffer_tensor(TGT)
        Loss_buf = fx.rocdl.make_buffer_tensor(Loss)
        Lse_buf = fx.rocdl.make_buffer_tensor(Lse)
        DX_buf = fx.rocdl.make_buffer_tensor(DX)
        if fx.const_expr(has_loss_weight):
            LW_buf = fx.rocdl.make_buffer_tensor(LossWeight)
            lw_div = fx.logical_divide(LW_buf, fx.make_layout(1, 1))

        row_x = fx.slice(X_buf, (bid, None))
        row_dx = fx.slice(DX_buf, (bid, None))
        x_div = fx.logical_divide(row_x, fx.make_layout(1, 1))
        dx_div = fx.logical_divide(row_dx, fx.make_layout(1, 1))
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
            x_e = _load(x_div, x_reg_ty, ca_x, idx_safe)
            x = x_e if dtype is Float32 else x_e.extf(compute_type)
            v = is_valid.select(x, neg_inf)
            thread_max = ArithValue(thread_max).maximumf(v)
        row_max = _block_reduce(thread_max, s_max, lambda a, b: a.maximumf(b), float("-inf"))
        row_max_av = ArithValue(row_max)

        # Pass 2: row sum_exp → lse. Also tracks row sum_x for label
        # smoothing (mean(x) factor in the NLL-uniform mix term).
        thread_sum = zero_f
        if fx.const_expr(has_smoothing):
            thread_sum_x = zero_f
        for base in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base)
            is_valid = idx < n_i32
            idx_safe = is_valid.select(idx, fx.Int32(0))
            x_e = _load(x_div, x_reg_ty, ca_x, idx_safe)
            x = x_e if dtype is Float32 else x_e.extf(compute_type)
            e = _fm.exp(ArithValue(x) - row_max_av, fastmath="fast")
            thread_sum = ArithValue(thread_sum) + is_valid.select(e, zero_f)
            if fx.const_expr(has_smoothing):
                thread_sum_x = ArithValue(thread_sum_x) + is_valid.select(ArithValue(x), zero_f)
        row_sum = _block_reduce(thread_sum, s_sum, lambda a, b: a.addf(b, fastmath="fast"), 0.0)
        log_sum = _fm.log(ArithValue(row_sum), fastmath="fast")
        lse = row_max_av + log_sum
        lse_av = ArithValue(lse)
        if fx.const_expr(has_smoothing):
            row_sum_x = _block_reduce(
                thread_sum_x, s_sum_x,
                lambda a, b: a.addf(b, fastmath="fast"), 0.0,
            )
            mean_x = ArithValue(row_sum_x) * ArithValue(Float32(1.0 / N))

        # Every thread loads target[bid] and x[target] once. The x[target]
        # load MUST happen before Pass 3 — when ``dx`` aliases ``x`` (the
        # inplace-backward case), Pass 3 overwrites x[target] with
        # softmax - 1, so reading x[target] afterward would give the
        # post-backward value, not the original logit. Doing the load
        # per-lane (64-256× duplicated) is simpler than barriering
        # inside an scf.if — the line is cache-resident, near-free.
        #
        # Safe index: when target == ignore_index (commonly -100), the
        # raw ``trunci(i32)`` is a large negative number; AMD buffer
        # SRDs treat it as unsigned and fault. Substitute 0 in that
        # case — the ignored row's loss / dx is overwritten to 0
        # downstream regardless of x[0]'s value.
        t_e = _load(tgt_div, tgt_reg_ty, ca_tgt, bid)
        t_i32 = t_e.trunci(T.i32) if target_dtype is Int64 else t_e
        is_ignore = arith.cmpi(arith.CmpIPredicate.eq, t_i32, ignore_index)
        t_i32_safe = is_ignore.select(fx.Int32(0), ArithValue(t_i32))
        x_t_e = _load(x_div, x_reg_ty, ca_x, t_i32_safe)
        x_t = x_t_e if dtype is Float32 else x_t_e.extf(compute_type)

        # NLL loss = lse - x[target]. For label smoothing:
        #   loss = (1-α) * nll + α * (lse - mean(x))
        nll_av = lse_av - ArithValue(x_t)
        if fx.const_expr(has_smoothing):
            alpha = ArithValue(smoothing)
            one_minus_alpha = ArithValue(Float32(1.0)) - alpha
            uniform_term = lse_av - mean_x
            loss_val = one_minus_alpha * nll_av + alpha * uniform_term
        else:
            loss_val = nll_av

        # ignore_index mask is already computed above (is_ignore).
        if fx.const_expr(has_loss_weight):
            lw_val = ArithValue(_load(lw_div, f_reg_ty, ca_f, bid))
            loss_val = loss_val * lw_val
            scale_val = lw_val
        else:
            scale_val = ArithValue(Float32(1.0))
        zero_av = ArithValue(Float32(0.0))
        loss_val = is_ignore.select(zero_av, loss_val)
        scale_val = is_ignore.select(zero_av, scale_val)

        # Scalar loss/lse store — one thread only.
        if tid == fx.Int32(0):
            _store(loss_div, f_reg_ty, ca_f, bid, loss_val)
            _store(lse_div, f_reg_ty, ca_f, bid, lse)

        # Barrier before Pass 3. In the inplace-backward case (dx aliases
        # x), Pass 3 writes corrupt x. Different waves can be in different
        # execution phases — wave 1 might still be loading x[target] while
        # wave 0 has already started writing x[j]. The barrier forces all
        # waves past the initial x[target] read before anyone starts
        # overwriting x. (Safe in the non-aliased case too; it's always
        # correct to synchronise here.)
        _gpu.barrier()

        # Pass 3: write dx = (softmax(x) - smooth_target) * scale.
        # With smoothing α, smooth_target[idx] = (1-α) on match, α/N elsewhere,
        # with the match position receiving the extra α/N as well — i.e.
        # `tgt_d = α/N + (idx == t ? 1-α : 0)`. When α=0 this collapses
        # to the usual one_hot. scale = 0 when ignored so dx = 0
        # regardless of softmax / target distribution.
        if fx.const_expr(has_smoothing):
            alpha_over_n = alpha * ArithValue(Float32(1.0 / N))
            one_minus_alpha_av = one_minus_alpha
        for base in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base)
            if arith.cmpi(arith.CmpIPredicate.ult, idx, n_i32):
                x_e = _load(x_div, x_reg_ty, ca_x, idx)
                x = x_e if dtype is Float32 else x_e.extf(compute_type)
                softmax_val = _fm.exp(ArithValue(x) - lse_av, fastmath="fast")
                is_target = arith.cmpi(arith.CmpIPredicate.eq, idx, t_i32)
                if fx.const_expr(has_smoothing):
                    match_bonus = is_target.select(one_minus_alpha_av, zero_av)
                    tgt_d = alpha_over_n + match_bonus
                else:
                    one_av = ArithValue(Float32(1.0))
                    tgt_d = is_target.select(one_av, zero_av)
                dx_f32 = (softmax_val - tgt_d) * scale_val
                dx_e = dx_f32 if dtype is Float32 else dx_f32.truncf(elem_type)
                _store(dx_div, x_reg_ty, ca_x, idx, dx_e)

    # Bake the grid dim as a Python int closure so flydsl sees a static
    # grid per compile — each distinct M_hint produces its own binary.
    _M_static = M_hint

    @flyc.jit
    def launch(
        X: fx.Tensor, TGT: fx.Tensor,
        Loss: fx.Tensor, Lse: fx.Tensor, DX: fx.Tensor,
        LossWeight: fx.Tensor,
        ignore_index: fx.Int32, smoothing: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(X, TGT, Loss, Lse, DX, LossWeight, ignore_index, smoothing).launch(
            grid=(_M_static, 1, 1), block=(block_threads, 1, 1), stream=stream,
        )

    return launch


_fwd_bwd_cache: dict = {}


def _compile_fwd_bwd(N, dt, tgt_dt, arch, M, has_smoothing, has_loss_weight):
    # Key includes M because FlyDSL's JIT bakes grid=(M, 1, 1) into the
    # compiled binary on first compile — re-using the binary with a
    # larger M than baked silently clips processing to the baked M rows.
    # Same-M repeat calls hit the cache. has_smoothing / has_loss_weight
    # specialize the kernel body so are part of the cache key.
    key = (N, dt, tgt_dt, arch, M, has_smoothing, has_loss_weight)
    got = _fwd_bwd_cache.get(key)
    if got is None:
        got = _build_ce_fwd_bwd(
            N=N, dtype=torch2flydsl_dtype_map[dt],
            target_dtype=torch2flydsl_dtype_map[tgt_dt], arch=arch,
            M_hint=M,
            has_smoothing=has_smoothing, has_loss_weight=has_loss_weight,
        )
        _fwd_bwd_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_cross_entropy_fwd_bwd",
    mutates_args=("loss", "lse", "dx"),
    schema=(
        "(Tensor x, Tensor target, Tensor(a0!) loss, Tensor(a1!) lse, "
        "Tensor(a2!) dx, int ignore_index, float label_smoothing, "
        "Tensor? loss_weight) -> ()"
    ),
)
def _cross_entropy_fwd_bwd(
    x: Tensor, target: Tensor, loss: Tensor, lse: Tensor, dx: Tensor,
    ignore_index: int,
    label_smoothing: float,
    loss_weight: Optional[Tensor],
) -> None:
    assert x.is_cuda and target.is_cuda and loss.is_cuda and lse.is_cuda and dx.is_cuda
    assert x.dim() == 2 and dx.shape == x.shape and target.dim() == 1
    assert target.size(0) == x.size(0)
    assert loss.dim() == 1 and lse.dim() == 1
    assert loss.size(0) == x.size(0) and lse.size(0) == x.size(0)
    assert x.stride(-1) == 1 and dx.stride(-1) == 1
    assert 0.0 <= label_smoothing < 1.0, (
        f"label_smoothing must be in [0, 1), got {label_smoothing}"
    )
    if loss_weight is not None:
        assert (
            loss_weight.dim() == 1 and loss_weight.size(0) == x.size(0)
            and loss_weight.dtype == torch.float32
            and loss_weight.is_cuda
        )
    M, N = x.shape
    has_smoothing = label_smoothing != 0.0
    has_loss_weight = loss_weight is not None
    # Dummy tensor when loss_weight is None — kernel drops the branch
    # at compile time, but FlyDSL still requires a tensor arg slot.
    lw_arg = loss_weight if has_loss_weight else torch.empty(
        1, device=x.device, dtype=torch.float32,
    )
    launcher = _compile_fwd_bwd(
        N, x.dtype, target.dtype, get_rocm_arch(), M,
        has_smoothing, has_loss_weight,
    )
    launcher(
        x, target, loss, lse, dx, lw_arg,
        int(ignore_index), float(label_smoothing),
    )


@_cross_entropy_fwd_bwd.register_fake
def _cross_entropy_fwd_bwd_fake(x, target, loss, lse, dx, ignore_index, label_smoothing, loss_weight):
    return None


def cross_entropy_fwd_bwd(
    x: Tensor,
    target: Tensor,
    dx: Optional[Tensor] = None,
    return_lse: bool = False,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    loss_weight: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    """Fused CE forward + inplace-backward.

    Returns ``(loss, lse_or_None, dx)``. ``dx`` contains
    ``(softmax(x) - smooth_target) * scale`` where ``smooth_target``
    reflects ``label_smoothing`` and ``scale`` folds in any
    ``loss_weight`` (and zeros rows with ``target == ignore_index``).
    Caller still scales for mean/sum reduction externally.

    If ``dx`` is None, allocates one of the same dtype as ``x``. Pass
    ``dx is x`` to write gradients in place on the logits buffer —
    saves an entire (M, N) alloc.
    """
    assert x.is_cuda and x.dim() == 2
    x = x if x.stride(-1) == 1 else x.contiguous()
    loss = torch.empty(x.size(0), device=x.device, dtype=torch.float32)
    lse = torch.empty(x.size(0), device=x.device, dtype=torch.float32)
    if dx is None:
        dx = torch.empty_like(x)
    _cross_entropy_fwd_bwd(
        x, target, loss, lse, dx,
        ignore_index, label_smoothing, loss_weight,
    )
    return loss, (lse if return_lse else None), dx


@torch.library.custom_op(
    "quack_amd::_cross_entropy_fwd",
    mutates_args=("loss", "lse"),
    schema=(
        "(Tensor x, Tensor target, Tensor(a0!) loss, Tensor(a1!) lse, "
        "int ignore_index, float label_smoothing, "
        "Tensor? loss_weight) -> ()"
    ),
)
def _cross_entropy_fwd(
    x: Tensor, target: Tensor, loss: Tensor, lse: Tensor,
    ignore_index: int, label_smoothing: float,
    loss_weight: Optional[Tensor],
) -> None:
    assert x.is_cuda and target.is_cuda and loss.is_cuda and lse.is_cuda
    assert x.dim() == 2 and target.dim() == 1 and target.size(0) == x.size(0)
    assert loss.dim() == 1 and lse.dim() == 1
    assert loss.size(0) == x.size(0) and lse.size(0) == x.size(0)
    assert x.stride(-1) == 1
    assert 0.0 <= label_smoothing < 1.0
    if loss_weight is not None:
        assert (
            loss_weight.dim() == 1 and loss_weight.size(0) == x.size(0)
            and loss_weight.dtype == torch.float32 and loss_weight.is_cuda
        )
    M, N = x.shape
    has_smoothing = label_smoothing != 0.0
    has_loss_weight = loss_weight is not None
    lw_arg = loss_weight if has_loss_weight else torch.empty(
        1, device=x.device, dtype=torch.float32,
    )
    launcher = _compile_fwd(
        N, x.dtype, target.dtype, get_rocm_arch(),
        has_smoothing, has_loss_weight,
    )
    launcher(
        x, target, loss, lse, lw_arg,
        int(ignore_index), float(label_smoothing), M,
    )


@_cross_entropy_fwd.register_fake
def _cross_entropy_fwd_fake(x, target, loss, lse, ignore_index, label_smoothing, loss_weight):
    return None


def cross_entropy_fwd(
    x: Tensor,
    target: Tensor,
    return_lse: bool = False,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    loss_weight: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Cross-entropy loss per row. Returns ``(loss, lse_or_None)``.

    ``target`` must be int32 or int64 and 1D of length ``x.size(0)``.
    Loss is float32 regardless of ``x`` dtype.

    Features:
      - ``ignore_index`` — rows with ``target == ignore_index`` emit 0
        loss (and 0 dx in backward). Default ``-100`` matches torch.
      - ``label_smoothing`` — mixes in ``α * (lse - mean(x))`` per row.
      - ``loss_weight`` — optional f32 ``(M,)`` tensor; multiplies the
        per-row loss / gradient (ignored rows stay 0).
    """
    assert x.is_cuda and x.dim() == 2
    x = x if x.stride(-1) == 1 else x.contiguous()
    loss = torch.empty(x.size(0), device=x.device, dtype=torch.float32)
    lse = torch.empty(x.size(0), device=x.device, dtype=torch.float32)
    _cross_entropy_fwd(
        x, target, loss, lse, ignore_index, label_smoothing, loss_weight,
    )
    return loss, (lse if return_lse else None)


@torch.library.custom_op(
    "quack_amd::_cross_entropy_bwd_dx",
    mutates_args=("dx",),
    schema=(
        "(Tensor x, Tensor target, Tensor lse, Tensor dloss, "
        "Tensor(a0!) dx, int ignore_index, float label_smoothing, "
        "Tensor? loss_weight) -> ()"
    ),
)
def _cross_entropy_bwd_dx(
    x: Tensor, target: Tensor, lse: Tensor, dloss: Tensor, dx: Tensor,
    ignore_index: int, label_smoothing: float,
    loss_weight: Optional[Tensor],
) -> None:
    assert x.is_cuda and target.is_cuda and lse.is_cuda and dloss.is_cuda and dx.is_cuda
    assert x.dim() == 2 and dx.shape == x.shape
    assert target.dim() == 1 and target.size(0) == x.size(0)
    assert lse.dim() == 1 and lse.size(0) == x.size(0) and lse.dtype == torch.float32
    assert dloss.dim() == 1 and dloss.size(0) == x.size(0) and dloss.dtype == torch.float32
    assert x.stride(-1) == 1 and dx.stride(-1) == 1
    assert 0.0 <= label_smoothing < 1.0
    if loss_weight is not None:
        assert (
            loss_weight.dim() == 1 and loss_weight.size(0) == x.size(0)
            and loss_weight.dtype == torch.float32 and loss_weight.is_cuda
        )
    M, N = x.shape
    has_smoothing = label_smoothing != 0.0
    has_loss_weight = loss_weight is not None
    lw_arg = loss_weight if has_loss_weight else torch.empty(
        1, device=x.device, dtype=torch.float32,
    )
    launcher = _compile_bwd(
        N, x.dtype, target.dtype, get_rocm_arch(),
        has_smoothing, has_loss_weight,
    )
    launcher(
        x, target, lse, dloss, dx, lw_arg,
        int(ignore_index), float(label_smoothing), M,
    )


@_cross_entropy_bwd_dx.register_fake
def _cross_entropy_bwd_dx_fake(x, target, lse, dloss, dx, ignore_index, label_smoothing, loss_weight):
    return None


def cross_entropy_bwd(
    x: Tensor,
    target: Tensor,
    lse: Tensor,
    dloss: Optional[Tensor] = None,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    loss_weight: Optional[Tensor] = None,
) -> Tensor:
    """Gradient of cross-entropy wrt x.

    Without smoothing:
      ``dx[m, j] = (softmax(x)[m, j] - (j == target[m])) * scale[m]``
    With smoothing α:
      ``dx[m, j] = (softmax(x)[m, j] - smooth_target[m, j]) * scale[m]``
      where ``smooth_target[m, j] = α/N + (j == t ? 1-α : 0)``.

    ``scale[m] = dloss[m] * (loss_weight[m] or 1)``, and is zero on rows
    where ``target[m] == ignore_index``.

    If ``dloss`` is None, defaults to 1 (unscaled gradient).
    """
    assert x.is_cuda and x.dim() == 2
    x = x if x.stride(-1) == 1 else x.contiguous()
    if dloss is None:
        dloss = torch.ones(x.size(0), device=x.device, dtype=torch.float32)
    elif dloss.dtype != torch.float32:
        dloss = dloss.to(torch.float32)
    dx = torch.empty_like(x)
    _cross_entropy_bwd_dx(
        x, target, lse, dloss, dx,
        ignore_index, label_smoothing, loss_weight,
    )
    return dx


__all__ = [
    "cross_entropy_fwd", "cross_entropy_bwd",
    "_cross_entropy_fwd", "_cross_entropy_bwd_dx",
]
