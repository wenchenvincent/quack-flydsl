# Copyright (c) 2026, AMD.

"""Wave/block/row reductions for QuACK's AMD kernels.

FlyDSL doesn't expose a reusable reduce library — each in-tree kernel
(layernorm, rmsnorm, softmax) inlines its own reduce helpers. This module
centralises that pattern so QuACK's AMD kernels can share one implementation:

    - ``wave_reduce``: butterfly via ``ArithValue.shuffle_xor``.
    - ``block_reduce``: two-stage wave-reduce → LDS → wave-reduce → broadcast.
    - ``row_reduce``: row-granular variant that parameterises threads-per-row.
    - ``online_softmax_reduce``: fused max/sum with online correction.

These are designed to be called from inside a ``@flyc.kernel`` body. The
``smem_scratch`` arguments are ``SmemPtr`` slabs allocated via
``flydsl.utils.smem_allocator.SmemAllocator`` in the host-side kernel builder.
"""

import math as _py_math
import operator
from typing import Callable, Optional

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemPtr

from quack.amd.flydsl_utils import get_wave_size


# ``scf_if_dispatch(cond, then_fn, else_fn)`` emits an ``scf.if`` for dynamic
# conditions and falls back to a Python branch for compile-time ones. The AST
# rewriter uses the same helper internally for ``if`` inside ``@flyc.kernel``
# bodies; reusing it here lets ``block_reduce`` live outside the kernel.
_scf_if = ReplaceIfWithDispatch.scf_if_dispatch


# --- Wave-level ------------------------------------------------------------


def _log2_int(n: int) -> int:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"wave_reduce requires a power-of-2 width, got {n}")
    return int(_py_math.log2(n))


def wave_reduce(val, op: Callable, width: int):
    """Reduce ``val`` across ``width`` lanes of the current wave via XOR butterfly.

    ``op`` is any callable ``(a, b) -> c`` that operates on FlyDSL Numeric values
    (e.g. ``operator.add`` for sum, ``arith.maximumf`` for max).
    ``width`` must be a power of 2 ≤ the wave size.
    """
    log_width = _log2_int(width)
    width_i32 = fx.Int32(width)
    w = val
    for sh_exp in range_constexpr(log_width):
        offset = fx.Int32(width // (2 << sh_exp))
        peer = w.shuffle_xor(offset, width_i32)
        w = op(w, peer)
    return w


# --- Block-level -----------------------------------------------------------


def block_reduce(
    val,
    op: Callable,
    smem_scratch: SmemPtr,
    num_waves: int,
    *,
    init_val: float = 0.0,
    wave_size: Optional[int] = None,
    tid=None,
):
    """Reduce ``val`` across all threads in the current workgroup.

    Two-stage:
      1. Each wave reduces internally via ``wave_reduce``.
      2. First lane of each wave writes its partial into ``smem_scratch[wave]``.
      3. One barrier; wave 0 reloads all partials (masked to ``num_waves``),
         reduces again, writes slot 0; one barrier.
      4. All threads return ``smem_scratch[0]``.

    ``smem_scratch`` must have at least ``num_waves`` slots of the element type
    of ``val`` (typically f32 for Float32 reductions).
    """
    if wave_size is None:
        wave_size = get_wave_size()
    if tid is None:
        tid = gpu.thread_idx.x

    if num_waves == 1:
        return wave_reduce(val, op, wave_size)

    lane = tid % fx.Int32(wave_size)
    wave = tid // fx.Int32(wave_size)

    partial = wave_reduce(val, op, wave_size)

    def _publish_partial():
        wave_idx = ArithValue(wave).index_cast(T.index)
        smem_scratch.store(partial, [wave_idx])

    _scf_if(lane == fx.Int32(0), _publish_partial)
    gpu.barrier()

    # Wave 0 combines the per-wave partials and writes the final value to slot 0.
    def _combine_in_wave0():
        in_range = lane < fx.Int32(num_waves)
        lane_safe = in_range.select(lane, fx.Int32(0))
        lane_safe_idx = ArithValue(lane_safe).index_cast(T.index)
        v = smem_scratch.load([lane_safe_idx])
        neutral = Float32(init_val)
        v = in_range.select(v, neutral)
        v = wave_reduce(v, op, wave_size)

        def _store_final():
            smem_scratch.store(v, [fx.Index(0)])

        _scf_if(lane == fx.Int32(0), _store_final)

    _scf_if(wave == fx.Int32(0), _combine_in_wave0)
    gpu.barrier()
    return smem_scratch.load([fx.Index(0)])


def block_reduce_add(val, smem_scratch: SmemPtr, num_waves: int, **kw):
    return block_reduce(val, lambda a, b: a.addf(b, fastmath="fast"), smem_scratch, num_waves, init_val=0.0, **kw)


def block_reduce_max(val, smem_scratch: SmemPtr, num_waves: int, **kw):
    return block_reduce(
        val,
        lambda a, b: a.maximumf(b),
        smem_scratch,
        num_waves,
        init_val=float("-inf"),
        **kw,
    )


# --- Row-level -------------------------------------------------------------


def row_reduce(
    val,
    op: Callable,
    threads_per_row: int,
    smem_scratch: Optional[SmemPtr] = None,
    num_waves: Optional[int] = None,
    *,
    init_val: float = 0.0,
    wave_size: Optional[int] = None,
    tid=None,
):
    """Reduce ``val`` across a contiguous group of ``threads_per_row`` threads.

    If ``threads_per_row <= wave_size``, this is a pure intra-wave reduction.
    Otherwise ``smem_scratch`` and ``num_waves`` must be provided and the
    reduction covers the full block (as in QuACK's ``row_reduce`` when
    ``reduction_buffer is not None``).
    """
    if wave_size is None:
        wave_size = get_wave_size()
    intra_wave = min(threads_per_row, wave_size)
    val = wave_reduce(val, op, intra_wave)
    if threads_per_row > wave_size:
        assert smem_scratch is not None and num_waves is not None, (
            "block-level row_reduce requires smem_scratch and num_waves"
        )
        val = block_reduce(
            val, op, smem_scratch, num_waves,
            init_val=init_val, wave_size=wave_size, tid=tid,
        )
    return val


# --- Online softmax --------------------------------------------------------


def online_softmax_reduce(
    max_x,
    sum_exp_x,
    smem_max: SmemPtr,
    smem_sum: SmemPtr,
    num_waves: int,
    *,
    wave_size: Optional[int] = None,
    tid=None,
):
    """Combine per-wave ``(max, sum_exp)`` into block-level online softmax stats.

    Each thread arrives with ``max_x = max over its row-slice`` and
    ``sum_exp_x = sum(exp(x - max_x))``. Returns the block-level ``(max, sum)``
    after the standard online correction
    ``sum_final = Σ_w exp(max_w - max_final) * sum_w``.

    Call this AFTER each thread has already done its intra-wave reduction
    (via ``wave_reduce``). ``smem_max`` / ``smem_sum`` must each have
    ``num_waves`` slots of f32.
    """
    if wave_size is None:
        wave_size = get_wave_size()
    if tid is None:
        tid = gpu.thread_idx.x

    if num_waves == 1:
        return max_x, sum_exp_x

    lane = tid % fx.Int32(wave_size)
    wave = tid // fx.Int32(wave_size)

    # Stage 1: each wave's lane 0 publishes (max, sum).
    def _publish_partials():
        wave_idx = ArithValue(wave).index_cast(T.index)
        smem_max.store(max_x, [wave_idx])
        smem_sum.store(sum_exp_x, [wave_idx])

    _scf_if(lane == fx.Int32(0), _publish_partials)
    gpu.barrier()

    # Stage 2: wave 0 loads per-wave partials, applies online correction,
    # reduces to one (max, sum) pair in slot 0.
    def _combine_in_wave0():
        from flydsl.expr import math as _fm

        in_range = lane < fx.Int32(num_waves)
        lane_safe = in_range.select(lane, fx.Int32(0))
        idx = ArithValue(lane_safe).index_cast(T.index)
        m = smem_max.load([idx])
        s = smem_sum.load([idx])
        neg_inf = Float32(float("-inf"))
        zero = Float32(0.0)
        m = in_range.select(m, neg_inf)
        s = in_range.select(s, zero)
        m_block = wave_reduce(m, lambda a, b: a.maximumf(b), wave_size)
        s = s * _fm.exp(m - m_block, fastmath="fast")
        s_block = wave_reduce(s, lambda a, b: a.addf(b, fastmath="fast"), wave_size)

        def _store_final():
            smem_max.store(m_block, [fx.Index(0)])
            smem_sum.store(s_block, [fx.Index(0)])

        _scf_if(lane == fx.Int32(0), _store_final)

    _scf_if(wave == fx.Int32(0), _combine_in_wave0)
    gpu.barrier()
    return smem_max.load([fx.Index(0)]), smem_sum.load([fx.Index(0)])


__all__ = [
    "wave_reduce",
    "block_reduce",
    "block_reduce_add",
    "block_reduce_max",
    "row_reduce",
    "online_softmax_reduce",
]
