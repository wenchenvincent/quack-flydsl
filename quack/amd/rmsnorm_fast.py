# Copyright (c) 2026, AMD.

"""Vectorized RMSNorm forward — fast path for the common case.

Matches the FlyDSL upstream pattern from
``FlyDSL/kernels/rmsnorm_kernel.py``: each thread loads 128 bits
(``buffer_load_dwordx4``) per issue using ``BufferCopy128b``, caches
the vector between passes, and reuses it for the normalize+write.

Scope (MVP fast path):
  - dtype: f16 or bf16 inputs, same dtype for weight.
  - N a multiple of VEC_WIDTH (= 16 / elem_bytes; 8 for f16/bf16).
  - No residual / bias / layernorm / per-head. These combinations fall
    back to the generic ``_build_norm_fwd`` kernel.

Perf target: bring the bf16 1024×1024 fwd from ~135 μs (30 GB/s, the
scalar-load kernel) toward the torch.compile ~40 μs baseline. 8×
coalesced loads/stores per lane is the primary unlock.
"""

import math as _py_math

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32, Float16, BFloat16
from flydsl.expr.typing import T, ReductionOp
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch, get_wave_size, torch2flydsl_dtype_map
from quack.amd.reduce import block_reduce_add


_EPS = 1e-6


def _elem_type_for(nc):
    return {Float32: T.f32, Float16: T.f16, BFloat16: T.bf16}[nc]


def _reserve_scratch(allocator, n_slots, elem_bytes):
    off = allocator._align(allocator.ptr, 16)
    allocator.ptr = off + n_slots * elem_bytes
    return off


def _build_rmsnorm_fwd_vec(*, N, dtype, arch):
    """Vectorized f16/bf16 RMSNorm fwd: BufferCopy128b, reg-cached x."""
    assert dtype in (Float16, BFloat16)
    elem_bits = dtype.width               # 16
    vec_width = 128 // elem_bits          # 8 for bf16/f16
    assert N % vec_width == 0, (
        f"fast-path rmsnorm requires N % {vec_width} == 0 for 128b loads; got N={N}"
    )

    wave_size = get_wave_size(arch)
    block_threads = 128 if N <= 16384 else 256
    num_waves = block_threads // wave_size
    # How many vector tiles each thread covers across the row.
    n_vec = N // vec_width                  # e.g. 1024 / 8 = 128 f16-vectors
    tiles_per_thread = (n_vec + block_threads - 1) // block_threads
    assert n_vec % block_threads == 0 or tiles_per_thread == 1, (
        f"this fast-path assumes n_vec ({n_vec}) divides evenly by "
        f"block_threads ({block_threads}); got tiles_per_thread={tiles_per_thread}"
    )

    sym = f"quack_amd_rmsnorm_fwd_vec_{dtype.__name__}_{N}_smem"
    allocator = SmemAllocator(None, arch=arch, global_sym_name=sym)
    off_sumsq = _reserve_scratch(allocator, num_waves, 4)
    fm_fast = arith.FastMathFlags.fast

    @flyc.kernel
    def kernel(X: fx.Tensor, W: fx.Tensor, Y: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = _elem_type_for(dtype)
        compute_type = T.f32
        n_float = arith.constant(float(N), type=compute_type)
        eps_c = arith.constant(_EPS, type=compute_type)

        base_ptr = allocator.get_base()
        s_sumsq = SmemPtr(base_ptr, off_sumsq, T.f32, shape=(num_waves,))
        s_sumsq.get()

        X_buf = fx.rocdl.make_buffer_tensor(X)
        W_buf = fx.rocdl.make_buffer_tensor(W)
        Y_buf = fx.rocdl.make_buffer_tensor(Y)

        row_x = fx.slice(X_buf, (bid, None))
        row_y = fx.slice(Y_buf, (bid, None))
        # Logical-divide by vec_width so indexing is in units of 128-bit vectors.
        x_div = fx.logical_divide(row_x, fx.make_layout(vec_width, 1))
        y_div = fx.logical_divide(row_y, fx.make_layout(vec_width, 1))
        w_div = fx.logical_divide(W_buf, fx.make_layout(vec_width, 1))

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_type)
        vec_reg_ty = fx.MemRefType.get(
            elem_type, fx.LayoutType.get(vec_width, 1), fx.AddressSpace.Register
        )
        vec_reg_lay = fx.make_layout(vec_width, 1)

        def _load_vec(div, idx):
            r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
            fx.copy_atom_call(copy_atom, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)

        def _store_vec(val, div, idx):
            r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
            fx.memref_store_vec(val, r)
            fx.copy_atom_call(copy_atom, r, fx.slice(div, (None, idx)))

        c_zero_f = arith.constant(0.0, type=compute_type)
        thread_sumsq = c_zero_f
        x_cache = []

        # Pass 1: vectorized load + cache + sum_sq. Each thread processes
        # ``tiles_per_thread`` vectors across the row. We spread the N/8
        # vectors round-robin across threads (stride = block_threads) so
        # all lanes issue loads in lockstep.
        for tile in range_constexpr(tiles_per_thread):
            idx = tid + fx.Int32(tile * block_threads)
            vec = _load_vec(x_div, idx)
            x_cache.append(vec)
            # Promote to f32, compute x², in-vector reduce.
            x_f32 = vec.to(Float32)
            x2 = x_f32 * x_f32
            red = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
            thread_sumsq = ArithValue(thread_sumsq) + red

        sum_sq = block_reduce_add(thread_sumsq, s_sumsq, num_waves,
                                  wave_size=wave_size, tid=tid)
        mean_sq = ArithValue(sum_sq) / n_float
        rstd = (mean_sq + eps_c).rsqrt(fastmath=fm_fast)

        # Pass 2: vectorized normalize + gamma + store.
        for tile in range_constexpr(tiles_per_thread):
            idx = tid + fx.Int32(tile * block_threads)
            g = _load_vec(w_div, idx).to(Float32)
            x_f32 = x_cache[tile].to(Float32)
            y = (x_f32 * rstd) * g
            y_e = y.to(dtype)
            _store_vec(y_e, y_div, idx)

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


_fast_cache: dict = {}


def _compile_fast(N, dtype, arch):
    key = (N, dtype, arch)
    got = _fast_cache.get(key)
    if got is None:
        got = _build_rmsnorm_fwd_vec(N=N, dtype=torch2flydsl_dtype_map[dtype], arch=arch)
        _fast_cache[key] = got
    return got


def rmsnorm_fwd_fast_eligible(x: Tensor, weight, bias, residual, store_rstd,
                              store_residual_out) -> bool:
    """Check if this call can use the vectorized fast-path kernel."""
    if x.dim() != 2:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    if weight is None or weight.dtype != x.dtype:
        return False
    if bias is not None or residual is not None or store_residual_out:
        return False
    if store_rstd:
        return False   # generic path owns the rstd-store today
    N = x.size(-1)
    elem_bytes = 2
    vec_width = 16 // elem_bytes
    if N % vec_width:
        return False
    # Fast-path builder requires n_vec divisible by block_threads.
    n_vec = N // vec_width
    block_threads = 128 if N <= 16384 else 256
    if n_vec % block_threads:
        return False
    return True


@torch.library.custom_op(
    "quack_amd::_rmsnorm_fwd_fast",
    mutates_args=("out",),
    schema="(Tensor x, Tensor weight, Tensor(a0!) out) -> ()",
)
def _rmsnorm_fwd_fast(x: Tensor, weight: Tensor, out: Tensor) -> None:
    assert x.is_cuda and weight.is_cuda and out.is_cuda
    assert x.dtype == weight.dtype == out.dtype
    M, N = x.shape
    _compile_fast(N, x.dtype, get_rocm_arch())(x, weight, out, M)


@_rmsnorm_fwd_fast.register_fake
def _rmsnorm_fwd_fast_fake(x, weight, out):
    return None


def rmsnorm_fwd_fast(x: Tensor, weight: Tensor) -> Tensor:
    """Call the vectorized fast path directly (no eligibility check)."""
    out = torch.empty_like(x)
    _rmsnorm_fwd_fast(x, weight, out)
    return out


__all__ = ["rmsnorm_fwd_fast", "rmsnorm_fwd_fast_eligible", "_rmsnorm_fwd_fast"]
