# Copyright (c) 2026, AMD.

"""Stream-K GEMM for gfx950 — production path (under construction).

This file holds the incremental build-out of a production stream-K
kernel.  Phases (each commit adds one):

  Session A — Runtime outer loop (DONE).  Identical semantics to
      ``gemm_persistent`` (whole-tile cycling via ``wg_id + step *
      num_cus``) but the outer tile loop is an scf.for with runtime
      bound.
  Session B — K-split (no tile crossing) (DONE).  Each tile's K-range
      is statically split into ``splits_per_tile`` shards; persistent
      grid walks ``total_tiles * splits_per_tile`` shards cycled
      ``wg_id + step * num_cus``.
  Session C — True stream-K (tile crossing) (THIS COMMIT).  Running
      accumulator + flush-on-tile-change.  WG range is a contiguous
      slice of ``total_iters = total_tiles * iters_per_tile``; tile
      changes inside the range trigger an atomic-fadd flush of the
      running accumulator.
  Session D — Last-partial counter sync + fused epilogue.
  Session E — Dispatch wiring + bench.

The dtype / tile scope (f16 × f16 → f32, 16×16 MFMA) matches
``gemm_persistent`` so each phase can be diffed against the static
baseline at the same shapes.  Narrower than ``gemm_gfx950_splitk``
on purpose — stream-K proper is scheduler-level, the matmul body is
deliberately stripped-down until the scheduler is proven.
"""


import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, range_constexpr, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d, fly as _fly_d

from quack.amd.flydsl_utils import get_rocm_arch
from quack.amd.tile_scheduler import get_num_cus


_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 16
_FRAG_C = 4


def _build_gemm_streamk_prod_f16(*, M, N, K, num_cus, arch):
    """Session A build: runtime outer loop, whole-tile cycling.

    Each WG processes tiles (wg_id, wg_id + num_cus, wg_id + 2*num_cus,
    …) up to total_tiles.  Same tile-walk as ``gemm_persistent`` — the
    only difference is the outer loop is an scf.for (runtime bound)
    instead of ``range_constexpr`` (compile-time unrolled).  Per-tile
    MFMA body is unchanged.
    """
    assert M % _MFMA_M == 0 and N % _MFMA_N == 0 and K % _MFMA_K == 0
    tiles_m = M // _MFMA_M
    tiles_n = N // _MFMA_N
    total_tiles = tiles_m * tiles_n
    steps_per_wg = (total_tiles + num_cus - 1) // num_cus

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_streamk_prod_f16_{M}_{N}_{K}_{num_cus}_smem",
    )

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        wg_id = fx.block_idx.x
        tid = fx.thread_idx.x

        lane_row = tid % fx.Int32(16)
        lane_k_group = tid // fx.Int32(16)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        ca_h = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), T.f16)
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        h_reg_ty = fx.MemRefType.get(T.f16, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

        def _load_h(div, idx):
            r = fx.memref_alloca(h_reg_ty, reg_lay)
            fx.copy_atom_call(ca_h, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _store_f(div, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            ts = _vfull(1, Float32(val), Float32)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_f, r, fx.slice(div, (None, idx)))

        acc_ty = T.vec(_FRAG_C, T.f32)

        # Outer loop — runtime scf.for.  scf.for requires at least one
        # iter_arg in FlyDSL; use a dummy i32 counter that's never read
        # (we derive ``tile_idx`` directly from ``step``, not from the
        # iter state).  ``yield`` passes the dummy through unchanged.
        zero_i32 = arith.constant(0, type=T.i32)
        c_total_tiles = fx.Int32(total_tiles)
        c_num_cus = fx.Int32(num_cus)
        c_tiles_n = fx.Int32(tiles_n)

        for step, state in range(0, steps_per_wg, init=[zero_i32]):
            dummy = state[0]
            tile_idx = wg_id + ArithValue(step).index_cast(T.i32) * c_num_cus
            # Guard against over-run when total_tiles isn't divisible by
            # num_cus.  The last few WGs may have an extra step beyond
            # their actual share; the scf.if makes them no-ops.
            in_range = arith.cmpi(
                arith.CmpIPredicate.ult, tile_idx, c_total_tiles,
            )
            if in_range:
                bid_m = ArithValue(tile_idx) // c_tiles_n
                bid_n = ArithValue(tile_idx) % c_tiles_n

                m_base = ArithValue(bid_m) * fx.Int32(_MFMA_M)
                n_base = ArithValue(bid_n) * fx.Int32(_MFMA_N)
                a_row = ArithValue(m_base) + ArithValue(lane_row)
                b_col = ArithValue(n_base) + ArithValue(lane_row)

                # Zero accumulator (fresh per tile).
                zeros = []
                for _ in range_constexpr(_FRAG_C):
                    zeros.append(arith.constant(0.0, type=T.f32))
                acc = vector.from_elements(acc_ty, zeros)

                # K loop — compile-time unrolled as in the baseline.
                k_tiles = K // _MFMA_K
                for k_tile in range_constexpr(k_tiles):
                    k_tile_base = fx.Int32(k_tile * _MFMA_K)
                    lane_k_base = ArithValue(lane_k_group) * fx.Int32(_FRAG_C) + ArithValue(k_tile_base)

                    row_a = fx.slice(A_buf, (a_row, None))
                    a_div = fx.logical_divide(row_a, fx.make_layout(1, 1))
                    a_vals = []
                    for i in range_constexpr(_FRAG_C):
                        a_vals.append(_load_h(a_div, ArithValue(lane_k_base) + fx.Int32(i)))
                    a_frag = vector.from_elements(T.vec(_FRAG_C, T.f16), a_vals)

                    b_vals = []
                    for i in range_constexpr(_FRAG_C):
                        row_b_k = fx.slice(B_buf, (ArithValue(lane_k_base) + fx.Int32(i), None))
                        b_div = fx.logical_divide(row_b_k, fx.make_layout(1, 1))
                        b_vals.append(_load_h(b_div, b_col))
                    b_frag = vector.from_elements(T.vec(_FRAG_C, T.f16), b_vals)

                    acc = fx.rocdl.mfma_f32_16x16x16f16(
                        acc_ty, [a_frag, b_frag, acc, 0, 0, 0],
                    )

                # Store.
                for i in range_constexpr(_FRAG_C):
                    out_row = ArithValue(m_base) + ArithValue(lane_k_group) * fx.Int32(_FRAG_C) + fx.Int32(i)
                    out_col = ArithValue(n_base) + ArithValue(lane_row)
                    row_c = fx.slice(C_buf, (out_row, None))
                    c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))
                    val_i = vector.extract(acc, static_position=[i], dynamic_position=[])
                    _store_f(c_div, out_col, val_i)

            yield [dummy]

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, C).launch(grid=(num_cus, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


_kernel_cache: dict = {}


def _compile(M, N, K, num_cus, arch):
    key = (M, N, K, num_cus, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_streamk_prod_f16(M=M, N=N, K=K, num_cus=num_cus, arch=arch)
        _kernel_cache[key] = got
    return got


def gemm_f16_streamk_prod(A: Tensor, B: Tensor) -> Tensor:
    """Session A scope: persistent grid + runtime tile loop.

    Matches the semantics of ``gemm_f16_persistent`` exactly — this
    function exists purely to validate that the FlyDSL ``range(...,
    init=...)`` scf.for pattern compiles and runs correctly at the
    outer-loop scope.  Later sessions build K-split on top of this
    scaffold.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    M, K = A.shape
    _, N = B.shape
    arch = get_rocm_arch()
    num_cus = get_num_cus(arch)
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    _compile(M, N, K, num_cus, arch)(A, B, C)
    return C


# ---------------------------------------------------------------------------
# Session B — K-split with atomic-fadd, no tile crossing.
# ---------------------------------------------------------------------------


def _pick_splits_per_tile(total_tiles: int, num_cus: int) -> int:
    """Heuristic: split each tile's K-range into enough shards that
    ``total_tiles * splits_per_tile`` saturates the CU count.

    If tiles >= num_cus, 1 shard/tile (no split) — CUs saturate without
    help.  Otherwise, enough shards to fill CUs with some slack.
    """
    if total_tiles >= num_cus:
        return 1
    # Saturate with modest over-subscription (avoids tail imbalance).
    return max(1, (num_cus + total_tiles - 1) // total_tiles)


def _build_gemm_streamk_b_f16(
    *, M, N, K, num_cus, splits_per_tile, arch,
):
    """Session B build: K-split, no tile crossing.

    Grid = num_cus (persistent).  Each WG iterates shards cycled
    ``wg_id + step * num_cus``, where ``total_shards = total_tiles *
    splits_per_tile``.  A shard identifies ``(tile_idx, k_slice_idx)``;
    the WG accumulates exactly the K-iterations assigned to its
    ``k_slice_idx``, then atomic-fadds the partial into ``C[tile_idx]``.

    No tile crossing means each WG processes at most one tile per shard
    — the running accumulator resets on every outer-loop iteration.
    Session C lifts this restriction.

    Output C MUST be zero-initialised (atomic-fadd accumulates).
    """
    assert M % _MFMA_M == 0 and N % _MFMA_N == 0 and K % _MFMA_K == 0
    assert splits_per_tile >= 1
    tiles_m = M // _MFMA_M
    tiles_n = N // _MFMA_N
    total_tiles = tiles_m * tiles_n
    iters_per_tile = K // _MFMA_K
    assert iters_per_tile % splits_per_tile == 0, (
        f"iters_per_tile ({iters_per_tile}) must be divisible by "
        f"splits_per_tile ({splits_per_tile}) — pick a splits value "
        "that cleanly divides K/_MFMA_K"
    )
    iters_per_split = iters_per_tile // splits_per_tile
    total_shards = total_tiles * splits_per_tile
    steps_per_wg = (total_shards + num_cus - 1) // num_cus

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=(
            f"quack_amd_gemm_streamk_prod_B_f16_{M}_{N}_{K}_{num_cus}_sp{splits_per_tile}_smem"
        ),
    )

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        wg_id = fx.block_idx.x
        tid = fx.thread_idx.x

        lane_row = tid % fx.Int32(16)
        lane_k_group = tid // fx.Int32(16)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        # Note: we do NOT use make_buffer_tensor for C writes — the
        # atomic-fadd path wants an llvm.ptr<1> so we extract the raw
        # pointer from the tensor and compute per-element byte offsets.
        _ptr_type = ir.Type.parse("!llvm.ptr<1>")
        _i64_type = T.i64
        c_raw = C.__fly_values__()[0]
        c_base_ptr_idx = _fly_d.extract_aligned_pointer_as_index(_ptr_type, c_raw)
        c_base_i64 = _llvm_d.PtrToIntOp(_i64_type, c_base_ptr_idx).result

        ca_h = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), T.f16)
        h_reg_ty = fx.MemRefType.get(T.f16, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

        def _load_h(div, idx):
            r = fx.memref_alloca(h_reg_ty, reg_lay)
            fx.copy_atom_call(ca_h, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        acc_ty = T.vec(_FRAG_C, T.f32)
        zero_i32 = arith.constant(0, type=T.i32)
        c_total_shards = fx.Int32(total_shards)
        c_num_cus = fx.Int32(num_cus)
        c_tiles_n = fx.Int32(tiles_n)
        c_splits_per_tile = fx.Int32(splits_per_tile)
        c_N = fx.Int32(N)

        for step, state in range(0, steps_per_wg, init=[zero_i32]):
            dummy = state[0]
            shard_idx = wg_id + ArithValue(step).index_cast(T.i32) * c_num_cus
            in_range = arith.cmpi(
                arith.CmpIPredicate.ult, shard_idx, c_total_shards,
            )
            if in_range:
                tile_idx = ArithValue(shard_idx) // c_splits_per_tile
                k_slice_idx = ArithValue(shard_idx) % c_splits_per_tile

                bid_m = ArithValue(tile_idx) // c_tiles_n
                bid_n = ArithValue(tile_idx) % c_tiles_n
                m_base = ArithValue(bid_m) * fx.Int32(_MFMA_M)
                n_base = ArithValue(bid_n) * fx.Int32(_MFMA_N)
                a_row = ArithValue(m_base) + ArithValue(lane_row)
                b_col = ArithValue(n_base) + ArithValue(lane_row)

                zeros = []
                for _ in range_constexpr(_FRAG_C):
                    zeros.append(arith.constant(0.0, type=T.f32))
                acc = vector.from_elements(acc_ty, zeros)

                # This shard's K-iteration range.
                k_iter_base = ArithValue(k_slice_idx) * fx.Int32(iters_per_split)
                # Inner K loop: compile-time unrolled over iters_per_split.
                for k_local in range_constexpr(iters_per_split):
                    k_tile_base = (ArithValue(k_iter_base) + fx.Int32(k_local)) * fx.Int32(_MFMA_K)
                    lane_k_base = ArithValue(lane_k_group) * fx.Int32(_FRAG_C) + ArithValue(k_tile_base)

                    row_a = fx.slice(A_buf, (a_row, None))
                    a_div = fx.logical_divide(row_a, fx.make_layout(1, 1))
                    a_vals = []
                    for i in range_constexpr(_FRAG_C):
                        a_vals.append(_load_h(a_div, ArithValue(lane_k_base) + fx.Int32(i)))
                    a_frag = vector.from_elements(T.vec(_FRAG_C, T.f16), a_vals)

                    b_vals = []
                    for i in range_constexpr(_FRAG_C):
                        row_b_k = fx.slice(B_buf, (ArithValue(lane_k_base) + fx.Int32(i), None))
                        b_div = fx.logical_divide(row_b_k, fx.make_layout(1, 1))
                        b_vals.append(_load_h(b_div, b_col))
                    b_frag = vector.from_elements(T.vec(_FRAG_C, T.f16), b_vals)

                    acc = fx.rocdl.mfma_f32_16x16x16f16(
                        acc_ty, [a_frag, b_frag, acc, 0, 0, 0],
                    )

                # Atomic-fadd each lane's 4 f32 output fragments into C.
                for i in range_constexpr(_FRAG_C):
                    out_row = ArithValue(m_base) + ArithValue(lane_k_group) * fx.Int32(_FRAG_C) + fx.Int32(i)
                    out_col = ArithValue(n_base) + ArithValue(lane_row)
                    val_i = vector.extract(acc, static_position=[i], dynamic_position=[])
                    # C flat element index, then byte offset.
                    c_flat = ArithValue(out_row) * c_N + ArithValue(out_col)
                    c_byte_off_idx = ArithValue(c_flat) * fx.Int32(4)
                    c_off_i64 = arith.index_cast(
                        _i64_type,
                        ArithValue(c_byte_off_idx).index_cast(T.index),
                    )
                    c_addr_i64 = _llvm_d.AddOp(
                        c_base_i64, c_off_i64, _llvm_d.IntegerOverflowFlags(0),
                    ).result
                    c_addr = _llvm_d.IntToPtrOp(_ptr_type, c_addr_i64).result
                    val_iv = val_i.ir_value() if hasattr(val_i, "ir_value") else val_i
                    _llvm_d.AtomicRMWOp(
                        _llvm_d.AtomicBinOp.fadd,
                        c_addr, val_iv,
                        _llvm_d.AtomicOrdering.monotonic,
                        syncscope="agent", alignment=4,
                    )

            yield [dummy]

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, C).launch(grid=(num_cus, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


_kernel_cache_b: dict = {}


def _compile_b(M, N, K, num_cus, splits_per_tile, arch):
    key = (M, N, K, num_cus, splits_per_tile, arch)
    got = _kernel_cache_b.get(key)
    if got is None:
        got = _build_gemm_streamk_b_f16(
            M=M, N=N, K=K, num_cus=num_cus,
            splits_per_tile=splits_per_tile, arch=arch,
        )
        _kernel_cache_b[key] = got
    return got


def gemm_f16_streamk_prod_b(A: Tensor, B: Tensor) -> Tensor:
    """Session B scope: persistent grid + K-split shards, no tile crossing.

    Splits each tile's K-range into ``splits_per_tile`` shards so
    ``total_tiles * splits_per_tile`` saturates the CU count when
    ``total_tiles < num_cus``.  When ``total_tiles >= num_cus`` the
    heuristic picks ``splits_per_tile = 1`` and this degenerates to
    Session A (with atomic-fadd writeback instead of direct store —
    same answer, slightly slower).
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    M, K = A.shape
    _, N = B.shape
    tiles_m = M // _MFMA_M
    tiles_n = N // _MFMA_N
    total_tiles = tiles_m * tiles_n
    arch = get_rocm_arch()
    num_cus = get_num_cus(arch)
    splits_per_tile = _pick_splits_per_tile(total_tiles, num_cus)
    # Must cleanly divide K/16 — step the heuristic down until it fits.
    iters_per_tile = K // _MFMA_K
    while splits_per_tile > 1 and iters_per_tile % splits_per_tile != 0:
        splits_per_tile -= 1
    # Output must be zero-initialised for atomic-fadd combine.
    C = torch.zeros(M, N, device=A.device, dtype=torch.float32)
    _compile_b(M, N, K, num_cus, splits_per_tile, arch)(A, B, C)
    return C


# ---------------------------------------------------------------------------
# Session C — True stream-K: tile crossing with running accumulator.
# ---------------------------------------------------------------------------


def _build_gemm_streamk_c_f16(*, M, N, K, num_cus, arch):
    """Session C build: running accumulator + flush-on-tile-change.

    Grid = ``num_cus`` persistent.  Each WG is assigned a contiguous
    slice ``[start_iter, end_iter)`` of the ``total_iters = total_tiles
    * iters_per_tile`` global K-iteration stream, with
    ``iters_per_wg = ceil(total_iters / num_cus)``.

    Outer loop carries state: ``(cur_iter, acc, tile_idx, has_partial)``.
    On each step, if still in range:

      - decode ``(t, iter_in_tile)`` from cur_iter
      - if ``has_partial and t != tile_idx``: atomic-fadd the acc for
        tile_idx into C (flush old tile), then reset acc to zero
      - do MFMA for this iter into the (possibly-just-reset) acc
      - bump cur_iter, tile_idx = t, has_partial = 1

    After the loop, if ``has_partial`` is still set, a final flush
    commits the tail acc.

    The accumulator/tile-change path uses ``arith.select`` (scalar cond,
    vector or scalar operands) rather than ``scf.if`` with value
    results — keeps the kernel body a single basic block and sidesteps
    FlyDSL's scf.if vector-result wiring.  The only ``scf.if`` in the
    hot loop guards the atomic-fadd side-effect.
    """
    assert M % _MFMA_M == 0 and N % _MFMA_N == 0 and K % _MFMA_K == 0
    tiles_m = M // _MFMA_M
    tiles_n = N // _MFMA_N
    total_tiles = tiles_m * tiles_n
    iters_per_tile = K // _MFMA_K
    total_iters = total_tiles * iters_per_tile
    iters_per_wg = (total_iters + num_cus - 1) // num_cus
    # scf.for bound — need to iterate at most iters_per_wg + 1 times
    # so the final flush gets a chance (we drain the partial inside the
    # loop via a sentinel-tile comparison, simpler to do after).
    max_iters = iters_per_wg

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=(
            f"quack_amd_gemm_streamk_prod_C_f16_{M}_{N}_{K}_{num_cus}_smem"
        ),
    )

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        wg_id = fx.block_idx.x
        tid = fx.thread_idx.x

        lane_row = tid % fx.Int32(16)
        lane_k_group = tid // fx.Int32(16)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        _ptr_type = ir.Type.parse("!llvm.ptr<1>")
        _i64_type = T.i64
        c_raw = C.__fly_values__()[0]
        c_base_ptr_idx = _fly_d.extract_aligned_pointer_as_index(_ptr_type, c_raw)
        c_base_i64 = _llvm_d.PtrToIntOp(_i64_type, c_base_ptr_idx).result

        ca_h = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), T.f16)
        h_reg_ty = fx.MemRefType.get(T.f16, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        reg_lay = fx.make_layout(1, 1)

        def _load_h(div, idx):
            r = fx.memref_alloca(h_reg_ty, reg_lay)
            fx.copy_atom_call(ca_h, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        acc_ty = T.vec(_FRAG_C, T.f32)
        zero_i32 = arith.constant(0, type=T.i32)
        one_i32 = arith.constant(1, type=T.i32)
        zero_f = arith.constant(0.0, type=T.f32)
        init_acc = vector.from_elements(acc_ty, [zero_f] * _FRAG_C)

        c_iters_per_tile = fx.Int32(iters_per_tile)
        c_iters_per_wg = fx.Int32(iters_per_wg)
        c_total_iters = fx.Int32(total_iters)
        c_tiles_n = fx.Int32(tiles_n)
        c_N = fx.Int32(N)

        # WG's K-iteration range.
        start_iter = ArithValue(wg_id) * c_iters_per_wg
        # end = min(start + iters_per_wg, total_iters).
        end_raw = start_iter + c_iters_per_wg
        end_lt_total = arith.cmpi(
            arith.CmpIPredicate.slt,
            ArithValue(end_raw).ir_value() if hasattr(end_raw, "ir_value") else end_raw,
            c_total_iters.ir_value() if hasattr(c_total_iters, "ir_value") else c_total_iters,
        )
        end_iter = arith.select(
            end_lt_total,
            end_raw.ir_value() if hasattr(end_raw, "ir_value") else end_raw,
            c_total_iters.ir_value() if hasattr(c_total_iters, "ir_value") else c_total_iters,
        )

        # Initial tile_idx = start_iter / iters_per_tile (valid even if
        # start_iter >= total_iters: extra-WG case is handled by
        # in_range guard below).
        init_tile_idx = ArithValue(start_iter).ir_value() if hasattr(start_iter, "ir_value") else start_iter
        init_tile_idx = arith.divsi(init_tile_idx, c_iters_per_tile.ir_value() if hasattr(c_iters_per_tile, "ir_value") else c_iters_per_tile)

        # Helper: emit an atomic-fadd of acc_vec into C[tile_idx].
        # Each lane fadds its 4 fragments.
        def _flush_acc(tile_idx_v, acc_vec):
            bid_m = ArithValue(tile_idx_v) // c_tiles_n
            bid_n = ArithValue(tile_idx_v) % c_tiles_n
            m_base_v = ArithValue(bid_m) * fx.Int32(_MFMA_M)
            n_base_v = ArithValue(bid_n) * fx.Int32(_MFMA_N)
            for i in range_constexpr(_FRAG_C):
                out_row = ArithValue(m_base_v) + ArithValue(lane_k_group) * fx.Int32(_FRAG_C) + fx.Int32(i)
                out_col = ArithValue(n_base_v) + ArithValue(lane_row)
                val_i = vector.extract(acc_vec, static_position=[i], dynamic_position=[])
                c_flat = ArithValue(out_row) * c_N + ArithValue(out_col)
                c_byte_off = ArithValue(c_flat) * fx.Int32(4)
                c_off_i64 = arith.index_cast(
                    _i64_type, ArithValue(c_byte_off).index_cast(T.index),
                )
                c_addr_i64 = _llvm_d.AddOp(
                    c_base_i64, c_off_i64, _llvm_d.IntegerOverflowFlags(0),
                ).result
                c_addr = _llvm_d.IntToPtrOp(_ptr_type, c_addr_i64).result
                val_iv = val_i.ir_value() if hasattr(val_i, "ir_value") else val_i
                _llvm_d.AtomicRMWOp(
                    _llvm_d.AtomicBinOp.fadd,
                    c_addr, val_iv,
                    _llvm_d.AtomicOrdering.monotonic,
                    syncscope="agent", alignment=4,
                )

        # iter_args: cur_iter (i32), acc (vec<4xf32>), tile_idx (i32),
        # has_partial (i32 0/1).
        init_state = [
            start_iter.ir_value() if hasattr(start_iter, "ir_value") else start_iter,
            init_acc,
            init_tile_idx,
            zero_i32,
        ]

        for step, state in range(0, max_iters, init=init_state):
            cur_iter = state[0]
            acc_in = state[1]
            tile_idx_in = state[2]
            has_partial_in = state[3]

            in_range = arith.cmpi(
                arith.CmpIPredicate.slt, cur_iter, end_iter,
            )

            t = arith.divsi(cur_iter, c_iters_per_tile.ir_value() if hasattr(c_iters_per_tile, "ir_value") else c_iters_per_tile)
            iter_in_tile = arith.remsi(cur_iter, c_iters_per_tile.ir_value() if hasattr(c_iters_per_tile, "ir_value") else c_iters_per_tile)

            # Tile change decision (only meaningful when in_range).
            tile_ne = arith.cmpi(arith.CmpIPredicate.ne, t, tile_idx_in)
            has_partial_i1 = arith.cmpi(
                arith.CmpIPredicate.ne, has_partial_in, zero_i32,
            )
            # Flush old tile iff in_range AND has_partial AND tile changed.
            do_flush = arith.andi(
                in_range, arith.andi(has_partial_i1, tile_ne),
            )

            # Emit the flush as a conditional side-effect.
            from flydsl._mlir.dialects import scf as _scf_d
            flush_if = _scf_d.IfOp(do_flush, results_=[], has_else=False)
            with ir.InsertionPoint(flush_if.then_block):
                _flush_acc(tile_idx_in, acc_in)
                _scf_d.YieldOp([])

            # MFMA for this iter.  Address math works even OOB — buffer
            # loads return 0 so MFMA produces 0 which we select-discard.
            bid_m = arith.divsi(t, c_tiles_n.ir_value() if hasattr(c_tiles_n, "ir_value") else c_tiles_n)
            bid_n = arith.remsi(t, c_tiles_n.ir_value() if hasattr(c_tiles_n, "ir_value") else c_tiles_n)
            m_base = ArithValue(bid_m) * fx.Int32(_MFMA_M)
            n_base = ArithValue(bid_n) * fx.Int32(_MFMA_N)
            a_row = ArithValue(m_base) + ArithValue(lane_row)
            b_col = ArithValue(n_base) + ArithValue(lane_row)
            k_tile_base = ArithValue(iter_in_tile) * fx.Int32(_MFMA_K)
            lane_k_base = ArithValue(lane_k_group) * fx.Int32(_FRAG_C) + ArithValue(k_tile_base)

            row_a = fx.slice(A_buf, (a_row, None))
            a_div = fx.logical_divide(row_a, fx.make_layout(1, 1))
            a_vals = []
            for i in range_constexpr(_FRAG_C):
                a_vals.append(_load_h(a_div, ArithValue(lane_k_base) + fx.Int32(i)))
            a_frag = vector.from_elements(T.vec(_FRAG_C, T.f16), a_vals)

            b_vals = []
            for i in range_constexpr(_FRAG_C):
                row_b_k = fx.slice(B_buf, (ArithValue(lane_k_base) + fx.Int32(i), None))
                b_div = fx.logical_divide(row_b_k, fx.make_layout(1, 1))
                b_vals.append(_load_h(b_div, b_col))
            b_frag = vector.from_elements(T.vec(_FRAG_C, T.f16), b_vals)

            # Select the base acc for the MFMA: if we just flushed the
            # old tile, start from zero; else continue acc_in.
            acc_base = arith.select(do_flush, init_acc, acc_in)
            acc_mfma = fx.rocdl.mfma_f32_16x16x16f16(
                acc_ty, [a_frag, b_frag, acc_base, 0, 0, 0],
            )

            # Final state updates: only advance when in_range.
            new_acc = arith.select(in_range, acc_mfma, acc_in)
            new_tile_idx = arith.select(in_range, t, tile_idx_in)
            new_has_partial = arith.select(in_range, one_i32, has_partial_in)
            new_cur_iter = arith.select(
                in_range,
                arith.addi(cur_iter, one_i32),
                cur_iter,
            )

            results = yield [new_cur_iter, new_acc, new_tile_idx, new_has_partial]

        # Final flush for tail partial.
        final_acc = results[1]
        final_tile_idx = results[2]
        final_has_partial = results[3]
        from flydsl._mlir.dialects import scf as _scf_d2
        final_has_partial_i1 = arith.cmpi(
            arith.CmpIPredicate.ne, final_has_partial, zero_i32,
        )
        flush_final_if = _scf_d2.IfOp(
            final_has_partial_i1, results_=[], has_else=False,
        )
        with ir.InsertionPoint(flush_final_if.then_block):
            _flush_acc(final_tile_idx, final_acc)
            _scf_d2.YieldOp([])

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, C).launch(grid=(num_cus, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


_kernel_cache_c: dict = {}


def _compile_c(M, N, K, num_cus, arch):
    key = (M, N, K, num_cus, arch)
    got = _kernel_cache_c.get(key)
    if got is None:
        got = _build_gemm_streamk_c_f16(M=M, N=N, K=K, num_cus=num_cus, arch=arch)
        _kernel_cache_c[key] = got
    return got


def gemm_f16_streamk_prod_c(A: Tensor, B: Tensor) -> Tensor:
    """Session C scope: true stream-K with tile crossing.

    Each WG's K-iteration range is a contiguous slice of the
    ``total_iters = total_tiles * iters_per_tile`` global stream.  When
    the range crosses a tile boundary, the running accumulator is
    atomic-fadd'd into the old tile's output and reset; MFMA continues
    into the new tile.

    Output C MUST be zero-initialised.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    M, K = A.shape
    _, N = B.shape
    arch = get_rocm_arch()
    num_cus = get_num_cus(arch)
    C = torch.zeros(M, N, device=A.device, dtype=torch.float32)
    _compile_c(M, N, K, num_cus, arch)(A, B, C)
    return C


__all__ = [
    "gemm_f16_streamk_prod",
    "gemm_f16_streamk_prod_b",
    "gemm_f16_streamk_prod_c",
]
