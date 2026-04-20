# Copyright (c) 2026, AMD.

"""RMSNorm / LayerNorm forward + backward — AMDGPU port of `quack/rmsnorm.py`.

Forward (``_build_norm_fwd``): one unified builder handles RMSNorm and
LayerNorm, both with or without bias, and optional ``store_rstd`` / ``store_mean``.
Compile-time flags (``is_layernorm``, ``has_bias``, …) specialise the emitted
MLIR so each compiled kernel only carries the code paths it needs.

    y = x_hat * w + b
    RMSNorm:   x_hat = x * rsqrt(mean(x²) + eps)
    LayerNorm: x_hat = (x - mean(x)) * rsqrt(var(x) + eps)

Backward (``_build_rmsnorm_dx``): RMSNorm ``dx`` kernel; ``dw`` via torch on
the host. LayerNorm backward + residual / per-head layouts are follow-ups.
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
# Forward — unified RMSNorm / LayerNorm kernel factory
# ---------------------------------------------------------------------------


def _build_norm_fwd(
    *,
    N: int,
    dtype,
    weight_dtype,
    bias_dtype,
    residual_dtype,
    is_layernorm: bool,
    has_bias: bool,
    has_residual: bool,
    store_rstd: bool,
    store_mean: bool,
    store_residual_out: bool,
    arch: str,
):
    wave_size = get_wave_size(arch)
    block_threads = 128 if N <= 16384 else 256
    num_waves = block_threads // wave_size
    elem_bits = dtype.width
    w_elem_bits = weight_dtype.width
    b_elem_bits = bias_dtype.width if bias_dtype is not None else 32
    r_elem_bits = residual_dtype.width if residual_dtype is not None else 32

    sym = (
        f"quack_amd_{'ln' if is_layernorm else 'rms'}_fwd_"
        f"{'b' if has_bias else 'nb'}_{'r' if store_rstd else 'nr'}_"
        f"{'m' if store_mean else 'nm'}_{'res' if has_residual else 'nres'}_"
        f"{'ro' if store_residual_out else 'nro'}_smem"
    )
    allocator = SmemAllocator(None, arch=arch, global_sym_name=sym)
    # LayerNorm needs 2 reduction slabs (sum, sum_sq); RMSNorm needs 1 (sum_sq).
    off_sumsq = _reserve_scratch(allocator, num_waves, 4)
    off_sum = _reserve_scratch(allocator, num_waves, 4) if is_layernorm else None
    fm_fast = arith.FastMathFlags.fast

    # The kernel body closes over this many compile-time flags; we generate
    # one @flyc.kernel per (is_layernorm, has_bias, store_rstd, store_mean).
    # That's up to 16 specialisations but each is lean and only the ones
    # callers actually ask for get compiled.

    @flyc.kernel
    def kernel(
        X: fx.Tensor,
        W: fx.Tensor,
        B: fx.Tensor,
        Res: fx.Tensor,
        ResOut: fx.Tensor,
        Y: fx.Tensor,
        Rstd: fx.Tensor,
        Mean: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = _elem_type_for(dtype)
        w_elem_type = _elem_type_for(weight_dtype)
        b_elem_type = _elem_type_for(bias_dtype) if has_bias else None
        r_elem_type = _elem_type_for(residual_dtype) if has_residual else None
        compute_type = T.f32
        n_float = arith.constant(float(N), type=compute_type)

        base_ptr = allocator.get_base()
        s_sumsq = SmemPtr(base_ptr, off_sumsq, T.f32, shape=(num_waves,))
        s_sumsq.get()
        if is_layernorm:
            s_sum = SmemPtr(base_ptr, off_sum, T.f32, shape=(num_waves,))
            s_sum.get()

        X_buf = fx.rocdl.make_buffer_tensor(X)
        W_buf = fx.rocdl.make_buffer_tensor(W)
        Y_buf = fx.rocdl.make_buffer_tensor(Y)
        if has_bias:
            B_buf = fx.rocdl.make_buffer_tensor(B)
        if has_residual:
            Res_buf = fx.rocdl.make_buffer_tensor(Res)
        if store_residual_out:
            ResOut_buf = fx.rocdl.make_buffer_tensor(ResOut)
        if store_rstd:
            Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
        if store_mean:
            Mean_buf = fx.rocdl.make_buffer_tensor(Mean)

        row_x = fx.slice(X_buf, (bid, None))
        row_y = fx.slice(Y_buf, (bid, None))
        x_div = fx.logical_divide(row_x, fx.make_layout(1, 1))
        y_div = fx.logical_divide(row_y, fx.make_layout(1, 1))
        w_div = fx.logical_divide(W_buf, fx.make_layout(1, 1))
        if has_bias:
            b_div = fx.logical_divide(B_buf, fx.make_layout(1, 1))
        if has_residual:
            row_res = fx.slice(Res_buf, (bid, None))
            res_div = fx.logical_divide(row_res, fx.make_layout(1, 1))
        if store_residual_out:
            row_res_out = fx.slice(ResOut_buf, (bid, None))
            res_out_div = fx.logical_divide(row_res_out, fx.make_layout(1, 1))
        if store_rstd:
            rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))
        if store_mean:
            mean_div = fx.logical_divide(Mean_buf, fx.make_layout(1, 1))

        copy_atom_x = fx.make_copy_atom(_bufcopy_for(elem_bits), elem_type)
        copy_atom_w = fx.make_copy_atom(_bufcopy_for(w_elem_bits), w_elem_type)
        copy_atom_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        if has_bias:
            copy_atom_b = fx.make_copy_atom(_bufcopy_for(b_elem_bits), b_elem_type)
        if has_residual:
            copy_atom_r = fx.make_copy_atom(_bufcopy_for(r_elem_bits), r_elem_type)
        x_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        w_reg_ty = fx.MemRefType.get(w_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        if has_bias:
            b_reg_ty = fx.MemRefType.get(b_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        if has_residual:
            r_reg_ty = fx.MemRefType.get(r_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
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

        # Pass 1: accumulate sum_sq and (for LayerNorm) sum.
        # If has_residual, we fold (x + residual) into the stats AND into the
        # effective x used in Pass 2. Optionally write (x + residual) to ResOut.
        c_zero_f = arith.constant(0.0, type=compute_type)
        thread_sumsq = c_zero_f
        if is_layernorm:
            thread_sum = c_zero_f
        for base_idx in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base_idx)
            is_valid = idx < fx.Int32(N)
            idx_safe = is_valid.select(idx, fx.Int32(0))
            x_e = _load(x_div, x_reg_ty, copy_atom_x, idx_safe)
            x = x_e if dtype is Float32 else x_e.extf(compute_type)
            x_av = ArithValue(x)
            if has_residual:
                r_e = _load(res_div, r_reg_ty, copy_atom_r, idx_safe)
                r_f = r_e if residual_dtype is Float32 else r_e.extf(compute_type)
                x_av = x_av + ArithValue(r_f)
            x2 = x_av * x_av
            thread_sumsq = ArithValue(thread_sumsq) + is_valid.select(x2, c_zero_f)
            if is_layernorm:
                thread_sum = ArithValue(thread_sum) + is_valid.select(x_av, c_zero_f)
            if has_residual and store_residual_out:
                # Only store when idx is in range (branch so out-of-range lanes skip).
                if arith.cmpi(arith.CmpIPredicate.ult, idx, fx.Int32(N)):
                    # ResOut dtype tracks residual_dtype.
                    ro_e = x_av if residual_dtype is Float32 else x_av.truncf(r_elem_type)
                    _store(res_out_div, r_reg_ty, copy_atom_r, idx, ro_e)

        sum_sq = block_reduce_add(thread_sumsq, s_sumsq, num_waves,
                                  wave_size=wave_size, tid=tid)
        if is_layernorm:
            sum_x = block_reduce_add(thread_sum, s_sum, num_waves,
                                     wave_size=wave_size, tid=tid)
            mean = ArithValue(sum_x) / n_float
            # var = E[x²] - E[x]²
            var_ = (ArithValue(sum_sq) / n_float) - (mean * mean)
            rstd = (var_ + _EPS).rsqrt(fastmath=fm_fast)
        else:
            mean_sq = ArithValue(sum_sq) / n_float
            rstd = (mean_sq + _EPS).rsqrt(fastmath=fm_fast)

        if store_rstd:
            if tid == fx.Int32(0):
                _store(rstd_div, f_reg_ty, copy_atom_f, bid, rstd)
        if store_mean:
            if tid == fx.Int32(0):
                if is_layernorm:
                    _store(mean_div, f_reg_ty, copy_atom_f, bid, mean)
                else:
                    _store(mean_div, f_reg_ty, copy_atom_f, bid, c_zero_f)

        # Pass 2: y = (x_eff [- mean]) * rstd * w [+ b],  where x_eff = x + residual
        # if has_residual else x. (Reload and re-add residual rather than caching
        # to keep the register pressure down.)
        for base_idx in range_constexpr(0, N, block_threads):
            idx = tid + fx.Int32(base_idx)
            if arith.cmpi(arith.CmpIPredicate.ult, idx, fx.Int32(N)):
                x_e = _load(x_div, x_reg_ty, copy_atom_x, idx)
                w_e = _load(w_div, w_reg_ty, copy_atom_w, idx)
                x = x_e if dtype is Float32 else x_e.extf(compute_type)
                w = w_e if weight_dtype is Float32 else w_e.extf(compute_type)
                x_eff = ArithValue(x)
                if has_residual:
                    r_e = _load(res_div, r_reg_ty, copy_atom_r, idx)
                    r_f = r_e if residual_dtype is Float32 else r_e.extf(compute_type)
                    x_eff = x_eff + ArithValue(r_f)
                x_centered = (x_eff - mean) if is_layernorm else x_eff
                y_f32 = (x_centered * rstd) * w
                if has_bias:
                    b_e = _load(b_div, b_reg_ty, copy_atom_b, idx)
                    b = b_e if bias_dtype is Float32 else b_e.extf(compute_type)
                    y_f32 = y_f32 + ArithValue(b)
                y_e = y_f32 if dtype is Float32 else y_f32.truncf(elem_type)
                _store(y_div, x_reg_ty, copy_atom_x, idx, y_e)

    @flyc.jit
    def launch(
        X: fx.Tensor, W: fx.Tensor, B: fx.Tensor, Res: fx.Tensor, ResOut: fx.Tensor,
        Y: fx.Tensor, Rstd: fx.Tensor, Mean: fx.Tensor, M: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(X, W, B, Res, ResOut, Y, Rstd, Mean).launch(
            grid=(M, 1, 1), block=(block_threads, 1, 1), stream=stream,
        )

    return launch


# ---------------------------------------------------------------------------
# Backward dx kernel (RMSNorm only for now)
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

        sum_xhat_wdy = block_reduce_add(thread_acc, s_red, num_waves,
                                        wave_size=wave_size, tid=tid)
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


def _compile_fwd(
    N, x_dt, w_dt, b_dt, r_dt,
    is_layernorm, has_bias, has_residual,
    store_rstd, store_mean, store_residual_out, arch,
):
    key = (
        N, x_dt, w_dt, b_dt, r_dt,
        is_layernorm, has_bias, has_residual,
        store_rstd, store_mean, store_residual_out, arch,
    )
    got = _fwd_cache.get(key)
    if got is None:
        got = _build_norm_fwd(
            N=N,
            dtype=torch2flydsl_dtype_map[x_dt],
            weight_dtype=torch2flydsl_dtype_map[w_dt],
            bias_dtype=torch2flydsl_dtype_map[b_dt] if b_dt is not None else None,
            residual_dtype=torch2flydsl_dtype_map[r_dt] if r_dt is not None else None,
            is_layernorm=is_layernorm,
            has_bias=has_bias,
            has_residual=has_residual,
            store_rstd=store_rstd,
            store_mean=store_mean,
            store_residual_out=store_residual_out,
            arch=arch,
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
    "quack_amd::_norm_fwd",
    mutates_args=("out", "rstd", "mean", "residual_out"),
    schema=(
        "(Tensor x, Tensor weight, Tensor? bias, Tensor? residual, "
        "Tensor(a0!) out, Tensor(a1!)? rstd, Tensor(a2!)? mean, "
        "Tensor(a3!)? residual_out, bool is_layernorm) -> ()"
    ),
)
def _norm_fwd(
    x: Tensor, weight: Tensor,
    bias: Optional[Tensor], residual: Optional[Tensor],
    out: Tensor,
    rstd: Optional[Tensor], mean: Optional[Tensor],
    residual_out: Optional[Tensor],
    is_layernorm: bool,
) -> None:
    assert x.is_cuda and weight.is_cuda and out.is_cuda
    assert x.dim() == 2 and out.shape == x.shape
    assert weight.dim() == 1 and weight.size(0) == x.size(-1)
    assert x.stride(-1) == 1 and out.stride(-1) == 1
    if bias is not None:
        assert bias.dim() == 1 and bias.size(0) == x.size(-1)
    if residual is not None:
        assert residual.shape == x.shape and residual.stride(-1) == 1
    if residual_out is not None:
        assert residual_out.shape == x.shape and residual_out.stride(-1) == 1
    if rstd is not None:
        assert rstd.dim() == 1 and rstd.size(0) == x.size(0) and rstd.dtype == torch.float32
    if mean is not None:
        assert mean.dim() == 1 and mean.size(0) == x.size(0) and mean.dtype == torch.float32
    M, N = x.shape
    has_bias = bias is not None
    has_residual = residual is not None
    store_residual_out = residual_out is not None
    launcher = _compile_fwd(
        N, x.dtype, weight.dtype,
        bias.dtype if has_bias else None,
        residual.dtype if has_residual else None,
        is_layernorm, has_bias, has_residual,
        rstd is not None, mean is not None, store_residual_out,
        get_rocm_arch(),
    )
    # Kernel takes 8 tensors; pass stand-ins for unused optional slots — the
    # kernel only dereferences them when the corresponding flag is set.
    B = bias if has_bias else weight
    Res = residual if has_residual else x
    ResOut = residual_out if store_residual_out else out
    R = rstd if rstd is not None else out
    Me = mean if mean is not None else out
    launcher(x, weight, B, Res, ResOut, out, R, Me, M)


@_norm_fwd.register_fake
def _norm_fwd_fake(x, weight, bias, residual, out, rstd, mean, residual_out, is_layernorm):
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


# ---------------------------------------------------------------------------
# Public Python API
# ---------------------------------------------------------------------------


def rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    eps: float = _EPS,
    store_rstd: bool = False,
    store_residual_out: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """RMSNorm forward. Returns ``(out, rstd_or_None, residual_out_or_None)``.

    If ``residual`` is provided, the kernel computes the norm of ``x + residual``
    (folded in pass 1 and pass 2 without a separate pre-add). When
    ``store_residual_out=True`` the pre-norm ``x + residual`` is written to a
    new tensor with the same dtype as ``residual`` (or ``x`` when residual is
    None) — useful for residual-add→norm chains that need the pre-norm value
    for the next layer's residual.
    """
    assert x.is_cuda and x.dim() == 2
    if eps != _EPS:
        raise NotImplementedError("runtime eps is a later pass")
    x = x if x.stride(-1) == 1 else x.contiguous()
    if weight is None:
        weight = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
    if residual is not None and residual.stride(-1) != 1:
        residual = residual.contiguous()
    out = torch.empty_like(x)
    rstd = torch.empty(x.size(0), device=x.device, dtype=torch.float32) if store_rstd else None
    res_out_dtype = residual.dtype if residual is not None else x.dtype
    residual_out = torch.empty_like(x, dtype=res_out_dtype) if store_residual_out else None
    _norm_fwd(x, weight, bias, residual, out, rstd, None, residual_out, False)
    return out, rstd, residual_out


def layernorm_fwd(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    eps: float = _EPS,
    store_stats: bool = False,
    store_residual_out: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """LayerNorm forward. Returns ``(out, rstd, mean, residual_out)``.

    ``residual`` semantics match ``rmsnorm_fwd``.
    """
    assert x.is_cuda and x.dim() == 2
    if eps != _EPS:
        raise NotImplementedError("runtime eps is a later pass")
    x = x if x.stride(-1) == 1 else x.contiguous()
    if weight is None:
        weight = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
    if residual is not None and residual.stride(-1) != 1:
        residual = residual.contiguous()
    out = torch.empty_like(x)
    rstd = torch.empty(x.size(0), device=x.device, dtype=torch.float32) if store_stats else None
    mean = torch.empty(x.size(0), device=x.device, dtype=torch.float32) if store_stats else None
    res_out_dtype = residual.dtype if residual is not None else x.dtype
    residual_out = torch.empty_like(x, dtype=res_out_dtype) if store_residual_out else None
    _norm_fwd(x, weight, bias, residual, out, rstd, mean, residual_out, True)
    return out, rstd, mean, residual_out


def rmsnorm_bwd(
    x: Tensor,
    weight: Tensor,
    dout: Tensor,
    rstd: Tensor,
    eps: float = _EPS,
) -> Tuple[Tensor, Tensor]:
    """RMSNorm backward. Returns ``(dx, dw)``."""
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


__all__ = [
    "rmsnorm_fwd", "layernorm_fwd", "rmsnorm_bwd",
    "_norm_fwd", "_rmsnorm_bwd_dx",
]
