# Copyright (c) 2026, AMD.

"""Shared MFMA-kernel helpers for gfx950 (CDNA4) GEMMs.

Pulled out of ``gemm_gfx950_splitk.py`` so the upcoming NN (``gemm_gfx950_nn.py``)
and TN (``gemm_gfx950_tn.py``) kernels can reuse the genuinely layout-agnostic
pieces without copy-pasting: the MFMA instruction wrappers (``_WmmaHalfK16/K32``),
the hot-loop instruction-mix scheduler (``_OnlineScheduler``), the LDS XOR
swizzle (``swizzle_xor16``), the split-K counter sizing constant, and the
vectorised epilogue helpers (bias + activation / gated / dact / dgated).

What is **not** in here: the A/B load patterns (``ldg_a``, ``sts_a``,
``ldg_matrix_b``, ``lds_matrix_a``) and the split-K counter/barrier code.
Those are inherently layout-specific (the stride-1 axis flips between
NT, NN, and TN) or tied to the kernel's closure variables (the split-K
counter is per-tile, derived from block_idx.xyz in the kernel body).
Each kernel hand-rolls its own load path — matches the FlyDSL reference
kernels (``preshuffle_gemm.py``, ``hgemm_splitk.py``), which also don't
use ``fx.make_tiled_copy`` for the hot-loop loads.

This module is import-only; it holds no mutable state.
"""

import functools
import math as _py_math

from flydsl.expr import arith, math as _fm, range_constexpr, rocdl, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32
from flydsl.expr.typing import T


# ---------------------------------------------------------------------------
# Split-K counter sizing
# ---------------------------------------------------------------------------

# Max threads per WG that read/write the per-tile split-K counter. Sized to
# match BLOCK_THREADS upper bound across all our kernel configs; each kernel's
# split_k_barrier uses this constant to index into the shared counter array.
SPLIT_K_COUNTER_MAX_LEN = 128


# ---------------------------------------------------------------------------
# LDS XOR swizzle (bank-conflict avoidance)
# ---------------------------------------------------------------------------

def swizzle_xor16(row, col_in_bytes, k_blocks16):
    """XOR-16 swizzle on the column-byte offset, keyed by row.

    Matches the FlyDSL ``mfma_preshuffle_pipeline.swizzle_xor16`` pattern —
    rotates the byte offset by ``(row % k_blocks16) * 16`` to spread MFMA-
    fragment reads across all 32 LDS banks regardless of the tile's K-inner
    stride. The ``k_blocks16`` parameter is ``BLOCK_K_BYTES // 16``.
    """
    return col_in_bytes ^ ((row % k_blocks16) * 16)


# ---------------------------------------------------------------------------
# MFMA instruction wrappers (gfx942 K=16, gfx950 K=32)
# ---------------------------------------------------------------------------

class _WmmaHalfK16:
    """CDNA3 (gfx942) ``mfma_f32_16x16x16 f16/bf16`` wrapper."""
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    WMMA_A_FRAG_VALUES = 4
    WMMA_B_FRAG_VALUES = 4
    WMMA_C_FRAG_VALUES = 4

    def __init__(self, dtype: str):
        self.dtype = dtype

    def __call__(self, a_frag, b_frag, c_frag):
        if self.dtype == "bf16":
            a_i = vector.bitcast(T.vec(self.WMMA_A_FRAG_VALUES, T.i16), a_frag)
            b_i = vector.bitcast(T.vec(self.WMMA_B_FRAG_VALUES, T.i16), b_frag)
            return rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a_i, b_i, c_frag, 0, 0, 0])
        return rocdl.mfma_f32_16x16x16f16(
            T.vec(self.WMMA_C_FRAG_VALUES, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0],
        )


class _WmmaHalfK32:
    """CDNA4 (gfx950) ``mfma_f32_16x16x32 f16/bf16`` wrapper — double the K per issue."""
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 32
    WMMA_A_FRAG_VALUES = 8
    WMMA_B_FRAG_VALUES = 8
    WMMA_C_FRAG_VALUES = 4

    def __init__(self, dtype: str):
        self.dtype = dtype

    def __call__(self, a_frag, b_frag, c_frag):
        res_ty = T.vec(self.WMMA_C_FRAG_VALUES, T.f32)
        ops = [a_frag, b_frag, c_frag, 0, 0, 0]
        if self.dtype == "bf16":
            return rocdl.mfma_f32_16x16x32_bf16(res_ty, ops)
        return rocdl.mfma_f32_16x16x32_f16(res_ty, ops)


# ---------------------------------------------------------------------------
# Hot-loop online scheduler (VMEM / MFMA / DSWR balancing)
# ---------------------------------------------------------------------------

class _OnlineScheduler:
    """Pairs ``rocdl.sched_*`` instruction-class hints with a running budget.

    The hot loop issues ``sched_vmem(n) / sched_mfma(m) / sched_dswr(d)`` to
    tell the compiler's scheduler how many of each to emit consecutively.
    ``_OnlineScheduler`` tracks a budget of ``total_signals`` and returns
    ``min(request, remaining)`` per ``consume(...)`` call, so the scheduler
    hints never overshoot the true per-iteration count.
    """

    def __init__(self, total_signals: int, init_count: int = 0):
        self.total_signals = total_signals
        self.current_signal_id = init_count
        self.remaining = init_count

    def release(self, count: int):
        count = min(count, self.total_signals - self.current_signal_id)
        self.current_signal_id += count
        self.remaining += count

    def consume(self, count: int):
        count = min(count, self.remaining)
        self.remaining -= count
        return count


# ---------------------------------------------------------------------------
# Write-back epilogue helpers (act on register-vector accumulator chunks)
# ---------------------------------------------------------------------------

def _apply_epilogue(vec, bias_tensor, col_start, activation, out_dtype, vec_size):
    """Fused ``vec = activation(vec + bias[cols])`` in registers.

    Layout-agnostic — operates on a ``vector<vec_size x out_dtype>`` that's
    already been loaded from LDS (or computed as the matmul output chunk).
    ``bias_tensor`` is the (N,) f32 bias; ``col_start`` is the N-offset of
    the first lane in ``vec``. Both bias and activation are optional — when
    both are "off" the caller skips the call entirely.
    """
    in_dtype = out_dtype
    result_scalars = []
    for i in range_constexpr(vec_size):
        v_i = vector.extract(vec, static_position=[i], dynamic_position=[])
        v_av = ArithValue(v_i)
        if in_dtype is T.f32:
            v_f32 = v_av
        else:
            v_f32 = v_av.extf(T.f32)
        if bias_tensor is not None:
            # bias_tensor[col_start + i] — bias is layout-agnostic, indexed by abs col.
            import flydsl.expr as fx  # local import avoids top-level cycle via fx.Index
            b_i = ArithValue(bias_tensor[col_start + fx.Index(i)])
            v_f32 = v_f32 + b_i
        zero = Float32(0.0)
        one = Float32(1.0)
        if activation == "relu":
            v_f32 = v_f32.maximumf(zero)
        elif activation == "relu_sq":
            v_f32 = v_f32.maximumf(zero) * v_f32
        elif activation == "silu":
            v_f32 = v_f32 / (one + _fm.exp(-v_f32, fastmath="fast"))
        elif activation == "gelu_tanh_approx":
            c1 = Float32(_py_math.sqrt(2.0 / _py_math.pi))
            c2 = Float32(0.044715 * _py_math.sqrt(2.0 / _py_math.pi))
            half = Float32(0.5)
            two = Float32(2.0)
            x_sq = v_f32 * v_f32
            z = v_f32 * (c1 + c2 * x_sq)
            tanh_z = one - two / (one + _fm.exp(two * z, fastmath="fast"))
            v_f32 = v_f32 * (half + half * tanh_z)
        if in_dtype is T.f32:
            v_out = v_f32
        else:
            v_out = v_f32.truncf(in_dtype)
        v_out_ir = v_out.ir_value() if hasattr(v_out, "ir_value") else v_out
        result_scalars.append(v_out_ir)
    return vector.from_elements(T.vec(vec_size, in_dtype), result_scalars)


def _apply_dact(acc_vec, preact_vec, activation, out_dtype, vec_size):
    """Fused activation-backward write-back: ``dpreact = acc * act'(preact)``.

    Layout-agnostic — both inputs are register vectors of the same shape.
    Emits ``dpreact`` cast to ``out_dtype``.
    """
    result = []
    for i in range_constexpr(vec_size):
        a_i = vector.extract(acc_vec, static_position=[i], dynamic_position=[])
        p_i = vector.extract(preact_vec, static_position=[i], dynamic_position=[])
        a_av = ArithValue(a_i)
        p_av = ArithValue(p_i)
        if out_dtype is T.f32:
            a, p = a_av, p_av
        else:
            a, p = a_av.extf(T.f32), p_av.extf(T.f32)
        one = ArithValue(Float32(1.0))
        zero = ArithValue(Float32(0.0))
        half = ArithValue(Float32(0.5))
        two = ArithValue(Float32(2.0))
        if activation == "relu":
            is_pos = p > zero
            dpreact = is_pos.select(a, zero)
        elif activation == "relu_sq":
            is_pos = p > zero
            grad = two * p
            dpreact = is_pos.select(a * grad, zero)
        elif activation == "silu":
            sig = one / (one + _fm.exp(-p, fastmath="fast"))
            deriv = sig * (one + p * (one - sig))
            dpreact = a * deriv
        elif activation == "gelu_tanh_approx":
            c1 = ArithValue(Float32(_py_math.sqrt(2.0 / _py_math.pi)))
            c2 = ArithValue(Float32(0.044715 * _py_math.sqrt(2.0 / _py_math.pi)))
            three_c2 = ArithValue(Float32(3.0 * 0.044715 * _py_math.sqrt(2.0 / _py_math.pi)))
            p_sq = p * p
            z = p * (c1 + c2 * p_sq)
            exp2z = _fm.exp(two * z, fastmath="fast")
            tanh_z = one - two / (one + exp2z)
            sech2_z = one - tanh_z * tanh_z
            dz_dp = c1 + three_c2 * p_sq
            deriv = half * (one + tanh_z) + half * p * sech2_z * dz_dp
            dpreact = a * deriv
        else:
            raise ValueError(f"unsupported dact_activation {activation!r}")
        if out_dtype is T.f32:
            out = dpreact
        else:
            out = dpreact.truncf(out_dtype)
        out_ir = out.ir_value() if hasattr(out, "ir_value") else out
        result.append(out_ir)
    return vector.from_elements(T.vec(vec_size, out_dtype), result)


def _apply_dgated(
    acc_vec, preact0, preact1, gate_type, out_dtype, vec_size, emit_postact,
):
    """Fused gated-activation backward write-back — see ``gemm_gfx950_splitk.py``
    for full semantics. Layout-agnostic: inputs are register vectors.

    Returns ``(dpreact0, dpreact1, postact_vec_or_None)``.
    """

    def _gate_bwd_scalar(g_av, u_av, dy_av):
        if out_dtype is T.f32:
            g, u, dy = g_av, u_av, dy_av
        else:
            g = g_av.extf(T.f32)
            u = u_av.extf(T.f32)
            dy = dy_av.extf(T.f32)
        one = ArithValue(Float32(1.0))
        zero = ArithValue(Float32(0.0))
        if gate_type == "swiglu":
            sig = one / (one + _fm.exp(-g, fastmath="fast"))
            fwd = g * sig
            fwd_prime = sig * (one + g * (one - sig))
            dgate = fwd_prime * u * dy
            dup = fwd * dy
            postact = fwd * u
        elif gate_type == "reglu":
            is_pos = g > zero
            fwd = is_pos.select(g, zero)
            dgate = is_pos.select(u * dy, zero)
            dup = fwd * dy
            postact = fwd * u
        elif gate_type == "geglu":
            c1 = ArithValue(Float32(_py_math.sqrt(2.0 / _py_math.pi)))
            c2 = ArithValue(Float32(0.044715 * _py_math.sqrt(2.0 / _py_math.pi)))
            three_c2 = ArithValue(Float32(3.0 * 0.044715 * _py_math.sqrt(2.0 / _py_math.pi)))
            half = ArithValue(Float32(0.5))
            two = ArithValue(Float32(2.0))
            g_sq = g * g
            z = g * (c1 + c2 * g_sq)
            exp2z = _fm.exp(two * z, fastmath="fast")
            tanh_z = one - two / (one + exp2z)
            sech2_z = one - tanh_z * tanh_z
            dz_dg = c1 + three_c2 * g_sq
            fwd = g * (half + half * tanh_z)
            fwd_prime = half * (one + tanh_z) + half * g * sech2_z * dz_dg
            dgate = fwd_prime * u * dy
            dup = fwd * dy
            postact = fwd * u
        elif gate_type == "glu":
            sig = one / (one + _fm.exp(-g, fastmath="fast"))
            fwd = sig
            fwd_prime = sig * (one - sig)
            dgate = fwd_prime * u * dy
            dup = fwd * dy
            postact = fwd * u
        else:
            raise ValueError(f"unknown dgated_gate_type {gate_type!r}")

        def _cast(v):
            if out_dtype is T.f32:
                return v.ir_value() if hasattr(v, "ir_value") else v
            c = v.truncf(out_dtype)
            return c.ir_value() if hasattr(c, "ir_value") else c

        return _cast(dgate), _cast(dup), _cast(postact)

    assert vec_size % 2 == 0, "dgated epilogue needs even vec_size (pair layout)"
    half = vec_size // 2
    dpreact_out = [[], []]
    postact_scalars = []
    for chunk_idx in range_constexpr(2):
        chunk = preact0 if chunk_idx == 0 else preact1
        for pair_idx in range_constexpr(half):
            g_i = vector.extract(chunk, static_position=[2 * pair_idx], dynamic_position=[])
            u_i = vector.extract(chunk, static_position=[2 * pair_idx + 1], dynamic_position=[])
            acc_idx = chunk_idx * half + pair_idx
            dy_i = vector.extract(acc_vec, static_position=[acc_idx], dynamic_position=[])
            dgate_s, dup_s, postact_s = _gate_bwd_scalar(
                ArithValue(g_i), ArithValue(u_i), ArithValue(dy_i),
            )
            dpreact_out[chunk_idx].append(dgate_s)
            dpreact_out[chunk_idx].append(dup_s)
            postact_scalars.append(postact_s)
    dpreact0 = vector.from_elements(T.vec(vec_size, out_dtype), dpreact_out[0])
    dpreact1 = vector.from_elements(T.vec(vec_size, out_dtype), dpreact_out[1])
    postact_vec = None
    if emit_postact:
        postact_vec = vector.from_elements(T.vec(vec_size, out_dtype), postact_scalars)
    return dpreact0, dpreact1, postact_vec


def _apply_gated_interleaved(pair0, pair1, gate_type, out_dtype, vec_size):
    """Interleaved-pair gated activation — see ``gemm_gfx950_splitk.py`` for
    pair-layout convention. Layout-agnostic.

    ``pair0`` / ``pair1`` store ``(g0, u0, g1, u1, ...)`` adjacency; the
    function produces ``vec_size`` output scalars from ``2 * vec_size`` input
    scalars.
    """

    def _gate_scalar(g_av, u_av):
        if out_dtype is T.f32:
            g, u = g_av, u_av
        else:
            g, u = g_av.extf(T.f32), u_av.extf(T.f32)
        one = ArithValue(Float32(1.0))
        zero = ArithValue(Float32(0.0))
        if gate_type == "swiglu":
            sig = one / (one + _fm.exp(-g, fastmath="fast"))
            out = g * sig * u
        elif gate_type == "reglu":
            out = g.maximumf(zero) * u
        elif gate_type == "geglu":
            c1 = ArithValue(Float32(_py_math.sqrt(2.0 / _py_math.pi)))
            c2 = ArithValue(Float32(0.044715 * _py_math.sqrt(2.0 / _py_math.pi)))
            half = ArithValue(Float32(0.5))
            two = ArithValue(Float32(2.0))
            g_sq = g * g
            z = g * (c1 + c2 * g_sq)
            tanh_z = one - two / (one + _fm.exp(two * z, fastmath="fast"))
            gelu_g = g * (half + half * tanh_z)
            out = gelu_g * u
        elif gate_type == "glu":
            sig = one / (one + _fm.exp(-g, fastmath="fast"))
            out = sig * u
        else:
            raise ValueError(f"unknown gate_type {gate_type!r}")
        if out_dtype is T.f32:
            return out.ir_value() if hasattr(out, "ir_value") else out
        out_cast = out.truncf(out_dtype)
        return out_cast.ir_value() if hasattr(out_cast, "ir_value") else out_cast

    half = vec_size // 2
    assert vec_size % 2 == 0, "interleaved pair load needs even vec_size"
    result = []
    for pair_vec in (pair0, pair1):
        for i in range_constexpr(half):
            g_i = vector.extract(pair_vec, static_position=[2 * i], dynamic_position=[])
            u_i = vector.extract(pair_vec, static_position=[2 * i + 1], dynamic_position=[])
            result.append(_gate_scalar(ArithValue(g_i), ArithValue(u_i)))
    return vector.from_elements(T.vec(vec_size, out_dtype), result)


__all__ = [
    "SPLIT_K_COUNTER_MAX_LEN",
    "swizzle_xor16",
    "_WmmaHalfK16",
    "_WmmaHalfK32",
    "_OnlineScheduler",
    "_apply_epilogue",
    "_apply_dact",
    "_apply_dgated",
    "_apply_gated_interleaved",
]
