# Copyright (c) 2026, AMD.
# Based on FlyDSL's kernels/hgemm_splitk.py (Apache 2.0). Vendored here
# because FlyDSL's ``kernels/`` tree isn't wheel-installed. The kernel
# itself is the reference FlyDSL implementation — attribution preserved;
# only the wrappers (custom-op registration, public launcher) are new.

"""Stream-K / split-K-capable f16/bf16 GEMM for gfx950 (CDNA4).

Architecture (128×256 / K=64, 4-wave WG, B pre-shuffle):
  - Output tile: 128 × 256
  - MFMA: 16×16×32 f16 (K=32 per issue, gfx950/CDNA4)
  - Warps per WG: 1 × 4 along M / N → 256 threads/WG, each wave owns
    a 128×64 sub-tile.
  - LDS:  STAGES=2 ping-pong for A (128×64 f16 per stage, XOR-swizzled).
          B goes gmem → register direct (no LDS) using a pre-shuffled
          layout that matches the MFMA B fragment.
  - Async DMA:  raw_ptr_buffer_load_lds for A. No register roundtrip.
  - Scheduler:  explicit sched_vmem / sched_mfma interleaves per the
    ``OnlineScheduler`` balance.
  - Split-K path: per-tile atomic counter + global_store_dword flag
    orchestrate the partials. Output written via per-pair atomic fadd
    on f16/bf16. Non-split case takes the direct-store fast path.

Public API:

    gemm_splitk(a, b, out=None, *, shuffle_b=False, tune=None)
        a:  (M, K) f16/bf16
        b:  (N, K) f16/bf16 (NT layout — ``c = a @ b.T``)
        out: optional preallocated (M, N) f16/bf16 output

The NT layout (B is (N, K)) matches how cuBLAS-style nn.Linear stores
its weight matrix — ergonomic for transformer linear layers.
"""

import functools
from typing import Optional

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, memref, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import fly_values
from flydsl.expr import arith, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from quack.amd.flydsl_tensor_shim import GTensor, STensor, get_dtype_in_kernel
from quack.amd.flydsl_utils import get_rocm_arch


SPLIT_K_COUNTER_MAX_LEN = 128


def swizzle_xor16(row, col_in_bytes, k_blocks16):
    return col_in_bytes ^ ((row % k_blocks16) * 16)


class _WmmaHalfK16:
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


class _OnlineScheduler:
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


@functools.lru_cache(maxsize=1024)
def _compile_hgemm_kernel(
    dtype: str,
    n: int,
    k: int,
    TILE_M: int = 128,
    TILE_N: int = 256,
    TILE_K: int = 64,
    SPLIT_K: int = 1,
    BLOCK_M_WARPS: int = 1,
    BLOCK_N_WARPS: int = 4,
    B_PRE_SHUFFLE: bool = True,
    B_TO_LDS: bool = False,
    # Fused epilogue (NN-kernel-equivalent): optional per-column f32 bias
    # and element-wise activation applied in the write-back. When enabled,
    # the kernel takes a ``Bias`` tensor argument and applies
    # ``act(y + bias[col])`` inside the C store loops. Keeps the vendored
    # matmul body unchanged; only the final store section differs.
    has_bias: bool = False,
    activation: str = "none",
    # Gated activation: when set, halves the output N dim and writes
    # ``gate_fn(LDS[m, c], LDS[m, c + N/2])`` to ``out[m, c]`` in the
    # write-back. Caller passes weight with shape ``(2*hidden, K)`` and
    # pre-allocates out of shape ``(M, hidden)``. Bias (if set) has
    # shape ``(2*hidden,)`` and is applied per-column BEFORE gating
    # (standard gated-MLP pattern). Only valid with SPLIT_K == 1 —
    # enforced in the public launcher.
    gate_type: str = "none",
    # Fused activation-backward (``gemm_dact`` equivalent). When set,
    # the kernel takes an additional ``PreAct`` tensor input of shape
    # ``(M, N)`` and computes ``dpreact = matmul_acc * act'(preact)``
    # element-wise in the write-back. Matches NVIDIA's GemmDActMixin
    # pattern. Mutually exclusive with bias/activation/gate_type.
    dact_activation: str = "none",
    # Fused gated-activation backward (``gemm_dgated`` equivalent). When
    # set, ``acc = A @ B.T`` of shape ``(M, n)`` is interpreted as the
    # upstream gradient ``dy`` per gated-hidden column. The kernel loads
    # the saved interleaved preact ``(M, 2n)`` = ``(g0, u0, g1, u1, …)``
    # from ``PreAct``, computes ``dpreact[2c]   = act'(g_c) * u_c * dy_c``
    # and              ``dpreact[2c+1] = act(g_c)        * dy_c``, and
    # stores the pair back to the output tensor ``C`` of shape
    # ``(M, 2n)``. Optionally co-emits ``postact[c] = act(g_c) * u_c``
    # to a separate ``Postact`` output tensor (``_EMIT_POSTACT``).
    # Mutually exclusive with bias / activation / gate_type / dact.
    dgated_gate_type: str = "none",
    emit_postact: bool = False,
    # Occupancy hint — sets ``rocdl.waves_per_eu`` attr on the gpu.func.
    # 1 = no limit (max occupancy), 2 = default upstream, higher values
    # constrain occupancy to reduce register pressure.
    waves_per_eu: Optional[int] = None,
    # ``_m_hint`` is NOT used inside the kernel body — it's part of the
    # cache key only, forcing a fresh compile per distinct runtime M.
    # Works around the FlyDSL JIT specialising on the first M passed
    # for ``grid=((m + BLOCK_M - 1) // BLOCK_M, ...)``: subsequent calls
    # with a larger M silently run with the baked grid and miss rows.
    _m_hint: int = 0,
):
    IS_SPLIT_K = SPLIT_K > 1
    BLOCK_K = TILE_K
    assert (k % SPLIT_K == 0) and (k // SPLIT_K >= 1)
    ks = k // SPLIT_K
    assert (ks % BLOCK_K == 0) and (ks // BLOCK_K >= 1)
    assert BLOCK_K >= 32
    if B_PRE_SHUFFLE:
        B_TO_LDS = False
    GPU_ARCH = get_rocm_arch()
    if GPU_ARCH == "gfx942":
        WMMA_IMPL = _WmmaHalfK16(dtype)
        DMA_BYTES = 4
        MFMA_PER_WARP_K = 2
        ASYNC_COPY = False
    else:
        WMMA_IMPL = _WmmaHalfK32(dtype)
        DMA_BYTES = 16
        MFMA_PER_WARP_K = 1
        ASYNC_COPY = True

    WARP_SIZE = 64
    DTYPE_BYTES = 2
    LDG_VEC_SIZE = 8
    STAGES = 2

    WMMA_M = WMMA_IMPL.WMMA_M
    WMMA_N = WMMA_IMPL.WMMA_N
    WMMA_K = WMMA_IMPL.WMMA_K
    WMMA_A_FRAG_VALUES = WMMA_IMPL.WMMA_A_FRAG_VALUES
    WMMA_B_FRAG_VALUES = WMMA_IMPL.WMMA_B_FRAG_VALUES
    WMMA_C_FRAG_VALUES = WMMA_IMPL.WMMA_C_FRAG_VALUES
    WARP_ATOM_M = WMMA_M
    WARP_ATOM_N = WMMA_N
    WARP_ATOM_K = WMMA_K * MFMA_PER_WARP_K
    BLOCK_K_LOOPS = ks // BLOCK_K
    WARP_K_STEPS = BLOCK_K // WARP_ATOM_K
    assert (BLOCK_K % WARP_ATOM_K == 0) and (WARP_K_STEPS >= 1)
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE
    WARP_M_STEPS = TILE_M // BLOCK_M_WARPS // WARP_ATOM_M
    WARP_N_STEPS = TILE_N // BLOCK_N_WARPS // WARP_ATOM_N
    assert (WARP_M_STEPS >= 1) and (WARP_N_STEPS >= 1)
    assert TILE_M % (BLOCK_M_WARPS * WARP_ATOM_M) == 0
    assert TILE_N % (BLOCK_N_WARPS * WARP_ATOM_N) == 0
    WARP_M = WARP_M_STEPS * WARP_ATOM_M
    WARP_N = WARP_N_STEPS * WARP_ATOM_N
    BLOCK_M = BLOCK_M_WARPS * WARP_M
    BLOCK_N = BLOCK_N_WARPS * WARP_N
    assert (n >= BLOCK_N) and (n % BLOCK_N == 0)
    BLOCK_MK_SIZE = BLOCK_M * BLOCK_K
    BLOCK_NK_SIZE = BLOCK_N * BLOCK_K
    BLOCK_MN_SIZE = BLOCK_M * BLOCK_N
    LDG_A_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_B_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_C_X_THREADS = BLOCK_N // LDG_VEC_SIZE
    BLOCK_VECS = LDG_VEC_SIZE * BLOCK_THREADS
    LDG_REG_A_COUNT = BLOCK_MK_SIZE // BLOCK_VECS
    LDG_REG_B_COUNT = BLOCK_NK_SIZE // BLOCK_VECS
    LDG_REG_C_COUNT = BLOCK_MN_SIZE // BLOCK_VECS
    assert (LDG_REG_A_COUNT >= 1) and (LDG_REG_B_COUNT >= 1) and (LDG_REG_C_COUNT >= 1)
    assert BLOCK_MK_SIZE % BLOCK_VECS == 0
    assert BLOCK_NK_SIZE % BLOCK_VECS == 0
    assert BLOCK_MN_SIZE % BLOCK_VECS == 0
    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES

    allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name=f"splitk_smem_{dtype}_{n}_{k}")
    smem_a_offset = allocator._align(allocator.ptr, 16)
    AS_BYTES = STAGES * BLOCK_M * BLOCK_K * DTYPE_BYTES
    AS_BYTES = max(AS_BYTES, BLOCK_M * BLOCK_N * DTYPE_BYTES)
    allocator.ptr = smem_a_offset + AS_BYTES
    if B_TO_LDS:
        smem_b_offset = allocator._align(allocator.ptr, 16)
        allocator.ptr = smem_b_offset + STAGES * BLOCK_N * BLOCK_K * DTYPE_BYTES
    LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES
    LDG_A_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    LDG_REG_A_COUNT_AS = BLOCK_MK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
    LDG_B_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    LDG_REG_B_COUNT_AS = BLOCK_NK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS

    KERNEL_NAME = f"hgemm_{dtype}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_S{STAGES}TN"
    KERNEL_NAME += "_NA" if not ASYNC_COPY else "_AS"
    if B_PRE_SHUFFLE:
        KERNEL_NAME += "_BP"
    if IS_SPLIT_K:
        KERNEL_NAME += f"_SPK{SPLIT_K}"
    if B_TO_LDS:
        KERNEL_NAME += "_BS"

    _HAS_BIAS = has_bias
    _ACT = activation
    _GATE = gate_type
    _IS_GATED = gate_type != "none"
    _DACT = dact_activation
    _IS_DACT = dact_activation != "none"
    _DGATED = dgated_gate_type
    _IS_DGATED = dgated_gate_type != "none"
    _EMIT_POSTACT = bool(emit_postact)
    assert not (_IS_GATED and IS_SPLIT_K), (
        "gated epilogue requires SPLIT_K == 1 — the public launcher forces this"
    )
    assert not (_IS_DACT and IS_SPLIT_K), (
        "dact epilogue requires SPLIT_K == 1"
    )
    assert not (_IS_DGATED and IS_SPLIT_K), (
        "dgated epilogue requires SPLIT_K == 1"
    )
    assert not (_IS_DACT and (_HAS_BIAS or _ACT != "none" or _IS_GATED)), (
        "dact epilogue is mutually exclusive with bias/activation/gate_type"
    )
    assert not (_IS_DGATED and (_HAS_BIAS or _ACT != "none" or _IS_GATED or _IS_DACT)), (
        "dgated epilogue is mutually exclusive with bias/activation/gate_type/dact"
    )
    assert not (_EMIT_POSTACT and not _IS_DGATED), (
        "emit_postact only valid with dgated_gate_type set"
    )
    # Output column span per tile.
    #   - Gated:   halved (each output col comes from 2 matmul cols).
    #   - Dgated:  iteration stays over the acc tile (BLOCK_N); output
    #              tensor has 2*n cols (dpreact shape), but we compute
    #              the 2× HBM stride in the epilogue branch directly.
    #   - Plain/dact/bias/activation: unchanged.
    if _IS_GATED:
        OUT_BLOCK_N = BLOCK_N // 2
        OUT_N = n // 2
    elif _IS_DGATED:
        OUT_BLOCK_N = BLOCK_N
        OUT_N = 2 * n
    else:
        OUT_BLOCK_N = BLOCK_N
        OUT_N = n
    LDG_C_X_THREADS_OUT = OUT_BLOCK_N // LDG_VEC_SIZE
    LDG_REG_C_COUNT_OUT = (BLOCK_M * OUT_BLOCK_N) // (LDG_VEC_SIZE * BLOCK_THREADS)

    @flyc.kernel
    def hgemm_kernel(
        C: fx.Tensor, A: fx.Tensor, B: fx.Tensor,
        m: fx.Int32,
        COUNTER: fx.Tensor,
        signal_state: fx.Int32,
        Bias: fx.Tensor,
        PreAct: fx.Tensor,
        Postact: fx.Tensor,
    ):
        dtype_ = get_dtype_in_kernel(dtype)
        _ptr_type = ir.Type.parse("!llvm.ptr<1>")
        _i64_type = T.i64
        c_zero_d = arith.constant(0.0, type=dtype_)
        acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG_VALUES, T.f32))

        A_ = GTensor(A, dtype=dtype_, shape=(-1, k))
        B_ = GTensor(B, dtype=dtype_, shape=(n, k))
        # Output tensor uses OUT_N:
        #   - plain/bias/act/dact: OUT_N = n
        #   - gated:               OUT_N = n / 2
        #   - dgated:              OUT_N = 2 * n  (dpreact has 2× cols)
        C_ = GTensor(C, dtype=dtype_, shape=(-1, OUT_N))
        if _IS_DACT:
            # dact path: preact has the same (M, N) shape as the matmul output.
            PreAct_ = GTensor(PreAct, dtype=dtype_, shape=(-1, n))
        if _IS_DGATED:
            # dgated path: preact is the saved interleaved (g0,u0,g1,u1,…)
            # with shape (M, 2n).  Same shape as the dpreact output.
            PreAct_ = GTensor(PreAct, dtype=dtype_, shape=(-1, 2 * n))
        if _EMIT_POSTACT:
            # postact = act(g_c) * u_c of shape (M, n) — one scalar per
            # matmul acc column.
            Postact_ = GTensor(Postact, dtype=dtype_, shape=(-1, n))
        if _HAS_BIAS:
            Bias_ = GTensor(Bias, dtype=T.f32, shape=(n,))
        base_ptr = allocator.get_base()
        smem_a_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(STAGES * BLOCK_M * BLOCK_K,))
        as_ = STensor(smem_a_ptr, dtype_, shape=(STAGES, BLOCK_M, BLOCK_K))
        if B_TO_LDS:
            smem_b_ptr = SmemPtr(base_ptr, smem_b_offset, dtype_, shape=(STAGES * BLOCK_N * BLOCK_K,))
            bs_ = STensor(smem_b_ptr, dtype_, shape=(STAGES, BLOCK_N, BLOCK_K))
        smem_c_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(BLOCK_M * BLOCK_N,))
        cs_ = STensor(smem_c_ptr, dtype_, shape=(BLOCK_M, BLOCK_N))
        if B_PRE_SHUFFLE:
            SHUFFLED_B_ = GTensor(B, dtype=dtype_, shape=(
                n // WARP_ATOM_N, k // WARP_ATOM_K,
                WARP_ATOM_K // LDG_VEC_SIZE, WARP_ATOM_N, LDG_VEC_SIZE,
            ))
        if IS_SPLIT_K:
            COUNTER_ = GTensor(COUNTER, dtype=T.i32, shape=(-1,))

        tid = fx.Int32(fx.thread_idx.x)
        wid = tid // WARP_SIZE
        w_tid = tid % WARP_SIZE
        block_m_idx = fx.block_idx.x
        block_n_idx = fx.block_idx.y
        ks_idx = fx.Index(fx.block_idx.z)
        ks_begin = arith.index_cast(T.i32, ks_idx * ks)
        counter_idx = (
            fx.Int32(signal_state * SPLIT_K_COUNTER_MAX_LEN)
            + fx.block_idx.x * fx.Int32(n // BLOCK_N)
            + fx.block_idx.y
        )

        m_offset = fx.Index(block_m_idx * BLOCK_M)
        n_offset = fx.Index(block_n_idx * BLOCK_N)
        # Output N-offset — halved in the gated case because output columns
        # correspond to 2 matmul columns each.
        n_offset_out = fx.Index(block_n_idx * OUT_BLOCK_N)
        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)

        warp_m_idx = wid // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid % BLOCK_N_WARPS * WARP_N
        ldmatrix_a_m_idx = w_tid % WMMA_M
        ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K
        ldmatrix_b_n_idx = w_tid % WMMA_N
        ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
        A_FRAGS_LEN = WARP_K_STEPS * WARP_M_STEPS
        B_FRAGS_LEN = WARP_K_STEPS * WARP_N_STEPS
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
        c_frags = [acc_init] * C_FRAGS_LEN

        def zero_c():
            cond_ks0 = arith.cmpi(arith.CmpIPredicate.eq, ks_idx, fx.Index(0))
            cond_ks0_if = scf.IfOp(cond_ks0, results_=[], has_else=False)
            with ir.InsertionPoint(cond_ks0_if.then_block):
                zero_vec = vector.broadcast(T.vec(LDG_VEC_SIZE, dtype_), c_zero_d)
                for i in range_constexpr(LDG_REG_C_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    m_local_idx = global_tid // LDG_C_X_THREADS
                    n_local_idx = global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE
                    row_idx = m_offset + fx.Index(m_local_idx)
                    cond_boundary = arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m))
                    cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                    with ir.InsertionPoint(cond_boundary_if.then_block):
                        C_.vec_store((row_idx, n_offset + n_local_idx), zero_vec, LDG_VEC_SIZE)
                        scf.YieldOp([])
                scf.YieldOp([])
            rocdl.sched_barrier(0)
            gpu.barrier()
            cond_ks0_if = scf.IfOp(cond_ks0, results_=[], has_else=False)
            with ir.InsertionPoint(cond_ks0_if.then_block):
                is_t0_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(tid), fx.Index(0))
                is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
                with ir.InsertionPoint(is_t0_cond_if.then_block):
                    counter_base_ptr = fly.extract_aligned_pointer_as_index(
                        _ptr_type, fly_values(COUNTER)[0],
                    )
                    counter_base_ptr = llvm.PtrToIntOp(_i64_type, counter_base_ptr).result
                    counter_byte_offset = arith.index_cast(T.i64, fx.Index(counter_idx) * fx.Index(4))
                    counter_ptr = llvm.AddOp(
                        counter_base_ptr, counter_byte_offset, llvm.IntegerOverflowFlags(0),
                    ).result
                    counter_ptr = llvm.IntToPtrOp(_ptr_type, counter_ptr).result
                    counter_ptr_v = getattr(counter_ptr, "_value", counter_ptr)
                    llvm.InlineAsmOp(None, [], "buffer_wbl2 sc0 sc1", "", has_side_effects=True)
                    llvm.InlineAsmOp(
                        None, [counter_ptr_v, arith.constant(1, type=T.i32)],
                        "global_store_dword $0, $1, off sc0 sc1", "v,v",
                        has_side_effects=True,
                    )
                    rocdl.s_waitcnt(0)
                    scf.YieldOp([])
                scf.YieldOp([])
            rocdl.sched_barrier(0)
            gpu.barrier()
            cond_ks0_if = scf.IfOp(cond_ks0, results_=[], has_else=False)
            with ir.InsertionPoint(cond_ks0_if.then_block):
                clean_cond = arith.cmpi(arith.CmpIPredicate.ult, fx.Index(tid), fx.Index(SPLIT_K_COUNTER_MAX_LEN))
                clean_cond_if = scf.IfOp(clean_cond, results_=[], has_else=False)
                with ir.InsertionPoint(clean_cond_if.then_block):
                    clean_counter_idx = (
                        fx.Int32(((signal_state + 2) % 3) * SPLIT_K_COUNTER_MAX_LEN)
                        + fx.Index(tid)
                    )
                    COUNTER_[fx.Index(clean_counter_idx)] = arith.constant(0, type=T.i32)
                    scf.YieldOp([])
                scf.YieldOp([])
            rocdl.sched_barrier(0)
            gpu.barrier()

        def split_k_barrier():
            init_cur = arith.constant(0, type=T.i32)
            w = scf.WhileOp([T.i32], [init_cur])
            before = ir.Block.create_at_start(w.before, [T.i32])
            after = ir.Block.create_at_start(w.after, [T.i32])
            with ir.InsertionPoint(before):
                cur = before.arguments[0]
                need_wait = arith.CmpIOp(
                    arith.CmpIPredicate.eq, cur, arith.constant(0, type=T.i32),
                ).result
                scf.ConditionOp(need_wait, [cur])
            with ir.InsertionPoint(after):
                counter_base_ptr = fly.extract_aligned_pointer_as_index(
                    _ptr_type, fly_values(COUNTER)[0],
                )
                counter_base_ptr = llvm.PtrToIntOp(_i64_type, counter_base_ptr).result
                counter_byte_offset = arith.index_cast(T.i64, fx.Index(counter_idx) * fx.Index(4))
                counter_ptr = llvm.AddOp(
                    counter_base_ptr, counter_byte_offset, llvm.IntegerOverflowFlags(0),
                ).result
                counter_ptr = llvm.IntToPtrOp(_ptr_type, counter_ptr).result
                counter_ptr_v = getattr(counter_ptr, "_value", counter_ptr)
                data = llvm.InlineAsmOp(
                    T.i32, [counter_ptr_v],
                    "global_load_dword $0, $1, off sc1", "=v,v",
                    has_side_effects=True,
                ).result
                rocdl.s_waitcnt(0)
                scf.YieldOp([data])
            gpu.barrier()

        def ldg_a(k_offset):
            vecs = []
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                row_idx = m_offset + fx.Index(m_local_idx)
                safe_row_idx = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m)),
                    row_idx,
                    fx.Index(0),
                )
                col_idx = fx.Index(k_offset + k_local_idx)
                vec = A_.vec_load((safe_row_idx, col_idx), LDG_VEC_SIZE)
                vecs.append(vec)
            return vecs

        def sts_a(vecs, lds_stage):
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
                as_.vec_store(
                    (fx.Index(lds_stage), m_local_idx, col_in_bytes // DTYPE_BYTES),
                    vecs[i], LDG_VEC_SIZE,
                )

        def ldg_sts_a_async(k_offset, lds_stage):
            for i in range_constexpr(LDG_REG_A_COUNT_AS):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS_AS
                k_local_idx = global_tid % LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
                row_idx = m_offset + fx.Index(m_local_idx)
                safe_row_idx = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m)),
                    row_idx, fx.Index(0),
                )
                col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
                global_offset = A_.linear_offset((safe_row_idx, col_idx)) * DTYPE_BYTES
                global_offset = arith.index_cast(T.i32, global_offset)
                lds_offset = as_.linear_offset(
                    (fx.Index(lds_stage), m_local_idx, k_local_idx)
                ) * DTYPE_BYTES
                lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                lds_addr = memref.extract_aligned_pointer_as_index(as_.memptr) + lds_offset
                lds_addr_ = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
                lds_ptr = llvm.inttoptr(lds_ptr_type, lds_addr_)
                rocdl.raw_ptr_buffer_load_lds(
                    A_.rsrc, lds_ptr,
                    arith.constant(DMA_BYTES, type=T.i32),
                    global_offset,
                    arith.constant(0, type=T.i32),
                    arith.constant(0, type=T.i32),
                    arith.constant(1, type=T.i32),
                )

        def lds_matrix_a(lds_stage):
            s = fx.Index(lds_stage)
            a_frags = [0] * (WARP_K_STEPS * WARP_M_STEPS)
            for ii in range_constexpr(WARP_M_STEPS):
                warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                for kk in range_constexpr(WARP_K_STEPS):
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    row = warp_atom_m_idx + ldmatrix_a_m_idx
                    col_in_bytes = (warp_atom_k_idx + ldmatrix_a_k_vec_idx) * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    vec = as_.vec_load(
                        (s, row, col_in_bytes // DTYPE_BYTES),
                        WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K,
                    )
                    a_frags[kk * WARP_M_STEPS + ii] = vec
            return a_frags

        def ldg_matrix_b(k_offset):
            vecs = []
            b_n_intra_base = ldmatrix_b_n_idx
            b_k_intra_vec = ldmatrix_b_k_vec_idx // LDG_VEC_SIZE
            b_n0_base = n_offset // WARP_ATOM_N + warp_n_idx // WARP_ATOM_N
            b_k0_base = k_offset // WARP_ATOM_K
            for kk in range_constexpr(WARP_K_STEPS):
                b_k0 = b_k0_base + kk
                for ii in range_constexpr(WARP_N_STEPS):
                    b_n0 = b_n0_base + ii
                    if not B_PRE_SHUFFLE:
                        warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                        warp_atom_k_idx = kk * WARP_ATOM_K
                        n_idx = n_offset + warp_atom_n_idx + ldmatrix_b_n_idx
                        k_idx = k_offset + warp_atom_k_idx + ldmatrix_b_k_vec_idx
                        vec = B_.vec_load(
                            (n_idx, k_idx), WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K,
                        )
                        vecs.append(vec)
                    else:
                        b_n_intra = b_n_intra_base
                        vec = SHUFFLED_B_.vec_load(
                            (b_n0, b_k0, b_k_intra_vec, b_n_intra, 0), LDG_VEC_SIZE,
                        )
                        vecs.append(vec)
            return vecs

        def block_mma_sync(a_frags, b_frags, c_frags):
            for kk in range_constexpr(WARP_K_STEPS):
                for ii in range_constexpr(WARP_M_STEPS):
                    a_frag = a_frags[kk * WARP_M_STEPS + ii]
                    for jj in range_constexpr(WARP_N_STEPS):
                        b_frag = b_frags[kk * WARP_N_STEPS + jj]
                        if MFMA_PER_WARP_K == 2:
                            a_i64x2 = vector.bitcast(T.i64x2, a_frag)
                            a0_i64 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                            a1_i64 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                            a_v0 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [a0_i64]))
                            a_v1 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [a1_i64]))
                            b_i64x2 = vector.bitcast(T.i64x2, b_frag)
                            b0_i64 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                            b1_i64 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
                            b_v0 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [b0_i64]))
                            b_v1 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [b1_i64]))
                            c_idx = ii * WARP_N_STEPS + jj
                            acc_mid = WMMA_IMPL(a_v0, b_v0, c_frags[c_idx])
                            c_frags[c_idx] = WMMA_IMPL(a_v1, b_v1, acc_mid)
                        else:
                            c_idx = ii * WARP_N_STEPS + jj
                            c_frags[c_idx] = WMMA_IMPL(a_frag, b_frag, c_frags[c_idx])

        if IS_SPLIT_K:
            zero_c()

        # Only the non-B_TO_LDS path is ported (it's the one hgemm's
        # default config uses and the one we measured 1.06-1.15× hipBLASLt).
        sts_a(ldg_a(ks_begin), 0)
        gpu.barrier()
        a_frags = lds_matrix_a(0)
        b_frags = ldg_matrix_b(ks_begin)
        rocdl.sched_barrier(0)

        def hot_loop_scheduler():
            MFMA_TOTAL = WARP_K_STEPS * WARP_M_STEPS * WARP_N_STEPS * MFMA_PER_WARP_K
            LDG_REG_A_COUNT_ = LDG_REG_A_COUNT_AS if ASYNC_COPY else LDG_REG_A_COUNT
            LDG_TOTAL = LDG_REG_A_COUNT_ + WARP_K_STEPS * WARP_N_STEPS
            mfma_ = _OnlineScheduler(MFMA_TOTAL, MFMA_TOTAL)
            ldg_ = _OnlineScheduler(LDG_TOTAL, LDG_TOTAL)
            if ASYNC_COPY:
                AVG_MFMA_COUNT = (MFMA_TOTAL + LDG_TOTAL - 1) // LDG_TOTAL
                for _ in range_constexpr(LDG_TOTAL):
                    rocdl.sched_vmem(ldg_.consume(1))
                    rocdl.sched_mfma(mfma_.consume(AVG_MFMA_COUNT))
            else:
                LDG_STS_TOTAL = LDG_TOTAL + LDG_REG_A_COUNT_
                AVG_MFMA_COUNT = (MFMA_TOTAL + LDG_STS_TOTAL - 1) // LDG_STS_TOTAL
                for _ in range_constexpr(LDG_TOTAL):
                    rocdl.sched_vmem(ldg_.consume(1))
                    rocdl.sched_mfma(mfma_.consume(AVG_MFMA_COUNT))
                for _ in range_constexpr(LDG_REG_A_COUNT_):
                    rocdl.sched_dswr(1)
                    rocdl.sched_mfma(mfma_.consume(AVG_MFMA_COUNT))
            rocdl.sched_barrier(0)

        init_state = [ks_begin, arith.constant(0, index=True)] + c_frags + a_frags + b_frags
        for bki, state in range(1, BLOCK_K_LOOPS, init=init_state):
            k_offset = state[0]
            current_stage = fx.Index(state[1])
            next_stage = 1 - current_stage
            c_frags = state[2 : 2 + C_FRAGS_LEN]
            a_frags = state[2 + C_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN]
            b_frags = state[2 + C_FRAGS_LEN + A_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN + B_FRAGS_LEN]
            if ASYNC_COPY:
                ldg_sts_a_async(k_offset + BLOCK_K, next_stage)
            else:
                a_regs_next = ldg_a(k_offset + BLOCK_K)
            b_frags_next = ldg_matrix_b(k_offset + BLOCK_K)
            block_mma_sync(a_frags, b_frags, c_frags)
            if not ASYNC_COPY:
                sts_a(a_regs_next, next_stage)
            hot_loop_scheduler()
            gpu.barrier()
            a_frags_next = lds_matrix_a(next_stage)
            k_offset = k_offset + fx.Int32(BLOCK_K)
            rocdl.sched_barrier(0)
            results = yield [k_offset, next_stage] + c_frags + a_frags_next + b_frags_next
        c_frags = results[2 : 2 + C_FRAGS_LEN]
        a_frags = results[2 + C_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN]
        b_frags = results[2 + C_FRAGS_LEN + A_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN + B_FRAGS_LEN]
        block_mma_sync(a_frags, b_frags, c_frags)

        # Write to LDS.
        stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES
        stmatrix_c_n_idx = w_tid % WMMA_N
        gpu.barrier()
        for ii in range_constexpr(WARP_M_STEPS):
            warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
            for jj in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                for kk in range_constexpr(WMMA_C_FRAG_VALUES):
                    lds_m_idx = fx.Index(warp_atom_m_idx + stmatrix_c_m_vec_idx + kk)
                    lds_n_idx = fx.Index(warp_atom_n_idx + stmatrix_c_n_idx)
                    val = vector.extract(
                        c_frags[ii * WARP_N_STEPS + jj],
                        static_position=[kk], dynamic_position=[],
                    )
                    cs_[lds_m_idx, lds_n_idx] = val.truncf(dtype_)

        if IS_SPLIT_K:
            split_k_barrier()
            out_raw = fly_values(C)[0]
            out_base_ptr = fly.extract_aligned_pointer_as_index(_ptr_type, out_raw)
            out_base_int = llvm.PtrToIntOp(_i64_type, out_base_ptr).result
            for i in range_constexpr(LDG_REG_C_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
                n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
                m_global_idx = m_offset + m_local_idx
                n_global_idx = n_offset + n_local_idx
                cond_boundary = arith.cmpi(arith.CmpIPredicate.ult, m_global_idx, fx.Index(m))
                cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                with ir.InsertionPoint(cond_boundary_if.then_block):
                    pk_val = cs_.vec_load((m_local_idx, n_local_idx), LDG_VEC_SIZE)
                    linear_bytes_offset = C_.linear_offset((m_global_idx, n_global_idx)) * DTYPE_BYTES
                    vec2_ty = T.vec(2, dtype_)
                    for vec_idx in range_constexpr(LDG_VEC_SIZE // 2):
                        e0 = vector.extract(pk_val, static_position=[vec_idx * 2], dynamic_position=[])
                        e1 = vector.extract(pk_val, static_position=[vec_idx * 2 + 1], dynamic_position=[])
                        pair = vector.from_elements(vec2_ty, [e0, e1])
                        pair_byte_offset = arith.index_cast(
                            T.i64,
                            linear_bytes_offset + fx.Index(vec_idx * 2 * DTYPE_BYTES),
                        )
                        pair_addr_i64 = llvm.AddOp(
                            out_base_int, pair_byte_offset, llvm.IntegerOverflowFlags(0),
                        ).result
                        pair_ptr = llvm.IntToPtrOp(_ptr_type, pair_addr_i64).result
                        pair_ptr_v = getattr(pair_ptr, "_value", pair_ptr)
                        pair_v = getattr(pair, "_value", pair)
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            pair_ptr_v, pair_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent", alignment=4,
                        )
                    scf.YieldOp([])
        else:
            gpu.barrier()
            # Two write-back shapes:
            # - Plain: iterates over (BLOCK_M, BLOCK_N), writes (M, N).
            # - Gated: iterates over (BLOCK_M, BLOCK_N/2), writes (M, N/2).
            #   For each output col c, loads LDS[m, c] (gate half) and
            #   LDS[m, c + BLOCK_N/2] (up half), applies gate_fn in registers,
            #   writes result. Bias (if set) is applied per-column to BOTH
            #   halves before gating — matches torch ``chunk(2, dim=-1)``
            #   convention.
            for i in range_constexpr(LDG_REG_C_COUNT_OUT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS_OUT)
                n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS_OUT * LDG_VEC_SIZE)
                m_global_idx = m_offset + m_local_idx
                cond_boundary = arith.cmpi(arith.CmpIPredicate.ult, m_global_idx, fx.Index(m))
                cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                with ir.InsertionPoint(cond_boundary_if.then_block):
                    if _IS_GATED:
                        # Interleaved-pair convention: matmul output cols
                        # [2c, 2c+1] are (gate, up) pairs. Caller must pass
                        # weight with gate and up rows interleaved (use the
                        # ``interleave_gated_weight`` helper). This is the
                        # same convention as NVIDIA's CUTLASS gated epilogue
                        # — it allows gate/up pairing to stay WITHIN a single
                        # matmul N-tile, avoiding cross-tile sync.
                        #
                        # Each 8-col output chunk at n_local_idx corresponds
                        # to input cols [2*n_local_idx, 2*n_local_idx + 16).
                        in_n_start = n_local_idx * fx.Index(2)
                        pair0 = cs_.vec_load(
                            (m_local_idx, in_n_start), LDG_VEC_SIZE,
                        )  # cols [2c..2c+8) = (g0, u0, g1, u1, g2, u2, g3, u3)
                        pair1 = cs_.vec_load(
                            (m_local_idx, in_n_start + fx.Index(LDG_VEC_SIZE)),
                            LDG_VEC_SIZE,
                        )  # cols [2c+8..2c+16) = (g4, u4, g5, u5, g6, u6, g7, u7)
                        if _HAS_BIAS:
                            pair0 = _apply_epilogue(
                                pair0, Bias_,
                                n_offset + in_n_start,
                                "none", dtype_, LDG_VEC_SIZE,
                            )
                            pair1 = _apply_epilogue(
                                pair1, Bias_,
                                n_offset + in_n_start + fx.Index(LDG_VEC_SIZE),
                                "none", dtype_, LDG_VEC_SIZE,
                            )
                        out_vec = _apply_gated_interleaved(
                            pair0, pair1, _GATE, dtype_, LDG_VEC_SIZE,
                        )
                        C_.vec_store(
                            (m_global_idx, n_offset_out + n_local_idx),
                            out_vec, LDG_VEC_SIZE,
                        )
                    elif _IS_DACT:
                        # Load matmul-accumulator chunk from LDS and
                        # preact chunk from HBM, then compute
                        # dpreact_i = acc_i * act'(preact_i).
                        vec = cs_.vec_load((m_local_idx, n_local_idx), LDG_VEC_SIZE)
                        preact_vec = PreAct_.vec_load(
                            (m_global_idx, n_offset + n_local_idx),
                            LDG_VEC_SIZE,
                        )
                        vec = _apply_dact(
                            vec, preact_vec, _DACT, dtype_, LDG_VEC_SIZE,
                        )
                        C_.vec_store(
                            (m_global_idx, n_offset + n_local_idx),
                            vec, LDG_VEC_SIZE,
                        )
                    elif _IS_DGATED:
                        # Load one acc chunk (LDG_VEC_SIZE scalars) from LDS
                        # and two interleaved preact chunks (2 × LDG_VEC_SIZE
                        # scalars covering 4 + 4 = 8 (gate, up) pairs) from
                        # HBM. Emit two dpreact chunks (interleaved dgate/dup
                        # pairs) and optionally one postact chunk.
                        #
                        # HBM stride: preact and dpreact are (M, 2n); postact
                        # is (M, n) so n_offset maps 1:1.
                        acc_vec = cs_.vec_load((m_local_idx, n_local_idx), LDG_VEC_SIZE)
                        preact_n_start = fx.Index(2) * (n_offset + n_local_idx)
                        preact0 = PreAct_.vec_load(
                            (m_global_idx, preact_n_start), LDG_VEC_SIZE,
                        )
                        preact1 = PreAct_.vec_load(
                            (m_global_idx, preact_n_start + fx.Index(LDG_VEC_SIZE)),
                            LDG_VEC_SIZE,
                        )
                        dpreact0, dpreact1, postact_vec = _apply_dgated(
                            acc_vec, preact0, preact1, _DGATED, dtype_,
                            LDG_VEC_SIZE, emit_postact=_EMIT_POSTACT,
                        )
                        C_.vec_store(
                            (m_global_idx, preact_n_start),
                            dpreact0, LDG_VEC_SIZE,
                        )
                        C_.vec_store(
                            (m_global_idx, preact_n_start + fx.Index(LDG_VEC_SIZE)),
                            dpreact1, LDG_VEC_SIZE,
                        )
                        if _EMIT_POSTACT:
                            Postact_.vec_store(
                                (m_global_idx, n_offset + n_local_idx),
                                postact_vec, LDG_VEC_SIZE,
                            )
                    else:
                        vec = cs_.vec_load((m_local_idx, n_local_idx), LDG_VEC_SIZE)
                        if _HAS_BIAS or _ACT != "none":
                            vec = _apply_epilogue(
                                vec, Bias_ if _HAS_BIAS else None,
                                n_offset + n_local_idx, _ACT, dtype_, LDG_VEC_SIZE,
                            )
                        C_.vec_store(
                            (m_global_idx, n_offset + n_local_idx),
                            vec, LDG_VEC_SIZE,
                        )
                    scf.YieldOp([])
        return

    @flyc.jit
    def launch_hgemm_kernel(
        C: fx.Tensor, A: fx.Tensor, B: fx.Tensor,
        m: fx.Int32,
        COUNTER: fx.Tensor,
        signal_state: fx.Int32,
        Bias: fx.Tensor,
        PreAct: fx.Tensor,
        Postact: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        bm = (m + BLOCK_M - 1) // BLOCK_M
        bn = n // BLOCK_N
        hgemm_kernel._func.__name__ = KERNEL_NAME
        launcher = hgemm_kernel(
            C, A, B, m, COUNTER, signal_state, Bias, PreAct, Postact,
        )
        if waves_per_eu is not None and int(waves_per_eu) >= 1:
            _wpe = int(waves_per_eu)
            for op in ctx.gpu_module_body.operations:
                if (
                    hasattr(op, "attributes")
                    and op.OPERATION_NAME == "gpu.func"
                ):
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32, _wpe,
                    )
        launcher.launch(
            grid=(bm, bn, SPLIT_K), block=(BLOCK_THREADS, 1, 1), stream=stream,
        )

    return launch_hgemm_kernel


def interleave_gated_weight(w_gate_up: Tensor) -> Tensor:
    """Convert a gated weight from split-halves to adjacent-pair layout.

    Input convention (matches ``torch.nn.Linear`` / ``chunk(2, dim=-1)``):
        ``w_gate_up`` shape ``(2 * hidden, in_features)``; rows
        ``[0, hidden)`` produce the gate, rows ``[hidden, 2*hidden)``
        produce the up.

    Output convention (required by ``gemm_splitk(gate_type=...)``):
        Row-interleaved — rows ``[2i, 2i+1]`` are ``(gate_i, up_i)``.
        Matmul output col 2i then holds gate_i and col 2i+1 holds up_i,
        which the kernel's write-back pairs adjacently without needing
        cross-tile synchronisation.

    Call once per weight tensor at model-load time; the result can be
    cached and reused across forward calls.
    """
    assert w_gate_up.dim() == 2
    two_hidden, in_features = w_gate_up.shape
    assert two_hidden % 2 == 0
    hidden = two_hidden // 2
    gate = w_gate_up[:hidden]
    up = w_gate_up[hidden:]
    # Stack alternately along dim 0: (hidden, 2, in_features) -> (2*hidden, in_features).
    return torch.stack([gate, up], dim=1).reshape(2 * hidden, in_features).contiguous()


def shuffle_b(x: Tensor, layout=(16, 16), k_steps=2) -> Tensor:
    """Pre-shuffle a (N, K) B matrix into the layout the kernel expects."""
    x_shape = x.shape
    VEC_SIZE = 16 // x.element_size()
    BN = layout[0]
    BK = layout[1] * k_steps
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} != 0"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} != 0"
    x = x.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // VEC_SIZE, VEC_SIZE)
    x = x.permute(0, 1, 3, 4, 2, 5).contiguous()
    x = x.view(*x_shape)
    return x


def _default_kwargs(m: int, n: int, k: int):
    kwargs = dict(
        TILE_M=128, TILE_N=256, TILE_K=64, SPLIT_K=1,
        BLOCK_M_WARPS=1, BLOCK_N_WARPS=4,
        B_PRE_SHUFFLE=True, B_TO_LDS=False,
    )
    # Small-M split-K presets matching the upstream defaults.
    if m <= 32 and n == 7168 and k == 2048:
        kwargs.update(TILE_K=64, TILE_M=32, TILE_N=128, SPLIT_K=4)
    if m <= 32 and n == 384 and k == 7168:
        kwargs.update(TILE_K=128, TILE_M=16, TILE_N=128, SPLIT_K=8)
    # W3 Round 1 — skinny-M / large-K shapes benefit from square-ish tiles
    # with larger TILE_K. At Shape C (M=2048, K=16384, N=2048) this cuts
    # the splitk/hipBLASLt gap from 1.81× → 1.43× (20% improvement). Shapes
    # A (4096³) and B (8192³) stay on the default 128×256×64 which wins
    # there; the gate below targets the regime where tall-skinny tiles
    # aren't useful (small M, small N, huge K).
    if (
        1024 <= m <= 2048 and 1024 <= n <= 2048 and k >= 16384
        and m % 128 == 0 and n % 128 == 0 and k % 128 == 0
    ):
        kwargs.update(TILE_M=128, TILE_N=128, TILE_K=128)
    if _DEFAULT_KWARGS_OVERRIDE is not None:
        kwargs.update(_DEFAULT_KWARGS_OVERRIDE)
    return kwargs


# W3 bench harness override — merged into ``_default_kwargs``'s result. Set
# to None in production so the tuned defaults apply.
_DEFAULT_KWARGS_OVERRIDE: Optional[dict] = None


def _set_default_kwargs_override(override: Optional[dict]):
    """Test / bench hook for forcing specific tile / split-k / block-warp
    settings without plumbing through every call site. Set to ``None`` to
    restore the tuned default config."""
    global _DEFAULT_KWARGS_OVERRIDE
    _DEFAULT_KWARGS_OVERRIDE = override


def _apply_epilogue(vec, bias_tensor, col_start, activation, out_dtype, vec_size):
    """Compile-time emitted post-matmul epilogue: vec = act(vec + bias[cols]).

    ``vec`` is a ``vector<vec_size x out_dtype>`` loaded from LDS. Bias
    is a (N,) f32 tensor. Both bias and activation are optional (controlled
    by the caller); when both are "off" this helper is bypassed entirely.
    """
    import math as _py_math
    from flydsl.expr import math as _fm
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.numeric import Float32
    in_dtype = out_dtype
    result_scalars = []
    for i in range_constexpr(vec_size):
        v_i = vector.extract(vec, static_position=[i], dynamic_position=[])
        # Widen to f32 for the arithmetic. ArithValue supports .extf() / .truncf().
        v_av = ArithValue(v_i)
        if in_dtype is T.f32:
            v_f32 = v_av
        else:
            v_f32 = v_av.extf(T.f32)
        if bias_tensor is not None:
            b_i = ArithValue(bias_tensor[col_start + fx.Index(i)])
            v_f32 = v_f32 + b_i
        zero = Float32(0.0)
        one = Float32(1.0)
        if activation == "relu":
            v_f32 = v_f32.maximumf(zero)
        elif activation == "relu_sq":
            v_f32 = v_f32.maximumf(zero) * v_f32
        elif activation == "silu":
            # silu(x) = x / (1 + exp(-x))
            v_f32 = v_f32 / (one + _fm.exp(-v_f32, fastmath="fast"))
        elif activation == "gelu_tanh_approx":
            c1 = Float32(_py_math.sqrt(2.0 / _py_math.pi))
            c2 = Float32(0.044715 * _py_math.sqrt(2.0 / _py_math.pi))
            half = Float32(0.5)
            two = Float32(2.0)
            x_sq = v_f32 * v_f32
            z = v_f32 * (c1 + c2 * x_sq)
            # tanh(z) = 1 - 2/(1+exp(2z))
            tanh_z = one - two / (one + _fm.exp(two * z, fastmath="fast"))
            v_f32 = v_f32 * (half + half * tanh_z)
        # Cast back to out dtype.
        if in_dtype is T.f32:
            v_out = v_f32
        else:
            v_out = v_f32.truncf(in_dtype)
        v_out_ir = v_out.ir_value() if hasattr(v_out, "ir_value") else v_out
        result_scalars.append(v_out_ir)
    return vector.from_elements(T.vec(vec_size, in_dtype), result_scalars)


def _apply_dact(acc_vec, preact_vec, activation, out_dtype, vec_size):
    """Fused activation-backward write-back: acc_i = acc_i * act'(preact_i).

    ``acc_vec`` is the matmul accumulator (conceptually ``dout @ W``).
    ``preact_vec`` is the saved forward pre-activation. Produces a
    vec of ``dpreact`` values cast to ``out_dtype``.
    """
    import math as _py_math
    from flydsl.expr import math as _fm
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.numeric import Float32

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
            # relu'(p) = (p > 0). Use (p > 0).select(a, 0) to avoid explicit branches.
            is_pos = p > zero
            dpreact = is_pos.select(a, zero)
        elif activation == "relu_sq":
            # d(relu(p)*p)/dp = 2p if p > 0 else 0. dpreact = acc * 2p * 1_{p>0}
            is_pos = p > zero
            grad = two * p
            dpreact = is_pos.select(a * grad, zero)
        elif activation == "silu":
            # silu'(p) = sigmoid(p) * (1 + p * (1 - sigmoid(p)))
            sig = one / (one + _fm.exp(-p, fastmath="fast"))
            deriv = sig * (one + p * (one - sig))
            dpreact = a * deriv
        elif activation == "gelu_tanh_approx":
            c1 = ArithValue(Float32(_py_math.sqrt(2.0 / _py_math.pi)))
            c2 = ArithValue(Float32(0.044715 * _py_math.sqrt(2.0 / _py_math.pi)))
            three_c2 = ArithValue(Float32(3.0 * 0.044715 * _py_math.sqrt(2.0 / _py_math.pi)))
            p_sq = p * p
            z = p * (c1 + c2 * p_sq)
            # tanh(z) = 1 - 2/(1+exp(2z))
            exp2z = _fm.exp(two * z, fastmath="fast")
            tanh_z = one - two / (one + exp2z)
            sech2_z = one - tanh_z * tanh_z
            # dz/dp = c1 + 3*c2*p^2
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
    """Fused gated-activation backward write-back.

    Inputs:
      ``acc_vec``  —  ``vector<vec_size x out_dtype>`` of dy values (one
                      per gated hidden column).
      ``preact0``  —  ``vector<vec_size x out_dtype>`` covering ``vec_size / 2``
                      ``(gate, up)`` pairs — cols ``[2c, 2c + vec_size)``.
      ``preact1``  —  same, covering cols ``[2c + vec_size, 2c + 2*vec_size)``.

    Outputs (``emit_postact`` controls whether ``postact_vec`` is produced):
      ``dpreact0``  —  interleaved ``(dgate, dup)`` pairs for ``preact0``'s
                       columns. ``[dg0, du0, dg1, du1, dg2, du2, dg3, du3]``.
      ``dpreact1``  —  same, for ``preact1``'s columns.
      ``postact_vec``  —  ``[pa0, pa1, …, pa_{vec_size - 1}]`` with
                          ``pa_c = act(g_c) * u_c``, or ``None`` if
                          ``emit_postact`` is False.

    Activation primes (all computed in f32, cast back to ``out_dtype`` at the
    end). ``s`` denotes ``sigmoid(g)`` reused across primes.
      - ``swiglu``: ``act(g) = g * s``, ``act'(g) = s * (1 + g * (1 - s))``.
      - ``reglu``:  ``act(g) = max(g, 0)``, ``act'(g) = (g > 0) ? 1 : 0``.
      - ``geglu``:  tanh-approx gelu + its derivative.
      - ``glu``:    ``act(g) = s``,       ``act'(g) = s * (1 - s)``.
    """
    import math as _py_math
    from flydsl.expr import math as _fm
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.numeric import Float32

    def _gate_bwd_scalar(g_av, u_av, dy_av):
        """Return (dgate, dup, postact) as out_dtype-cast IR scalars."""
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
    half = vec_size // 2  # pairs per chunk
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
    """Interleaved-pair gated activation: input vectors store
    (g, u, g, u, ...) adjacency. Produces ``vec_size`` output values
    from 2*``vec_size`` input values (one per (g, u) pair).

    ``pair0`` covers matmul cols [2c, 2c + vec_size) = 4 pairs.
    ``pair1`` covers cols [2c + vec_size, 2c + 2*vec_size) = 4 pairs.
    Output covers output cols [c, c + vec_size) = 8 values.

    Gate variants (matching torch's ``chunk(2, dim=-1)`` applied to a
    tensor that's been pre-interleaved by ``interleave_gated_weight``):
      - swiglu: silu(g) * u       = g * sigmoid(g) * u
      - reglu:  max(g, 0) * u
      - geglu:  gelu_tanh_approx(g) * u
      - glu:    sigmoid(g) * u
    """
    import math as _py_math
    from flydsl.expr import math as _fm
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.numeric import Float32

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

    # pair0 holds vec_size interleaved (g0, u0, g1, u1, ...) producing
    # vec_size/2 outputs. pair1 holds another vec_size/2 outputs.
    half = vec_size // 2
    assert vec_size % 2 == 0, "interleaved pair load needs even vec_size"
    result = []
    for pair_vec in (pair0, pair1):
        for i in range_constexpr(half):
            g_i = vector.extract(pair_vec, static_position=[2 * i], dynamic_position=[])
            u_i = vector.extract(pair_vec, static_position=[2 * i + 1], dynamic_position=[])
            result.append(_gate_scalar(ArithValue(g_i), ArithValue(u_i)))
    return vector.from_elements(T.vec(vec_size, out_dtype), result)


_SPLIT_K_SEMAPHORE: dict = {}
_SPLIT_K_SEMAPHORE_STATE: dict = {}


def _get_semaphore(stream: torch.cuda.Stream):
    sem = _SPLIT_K_SEMAPHORE.get(stream)
    if sem is None:
        sem = torch.zeros(
            (3 * SPLIT_K_COUNTER_MAX_LEN,),
            dtype=torch.int32, device=stream.device,
        )
        _SPLIT_K_SEMAPHORE[stream] = sem
        _SPLIT_K_SEMAPHORE_STATE[stream] = 0
    return sem, _SPLIT_K_SEMAPHORE_STATE[stream]


def _advance_state(stream: torch.cuda.Stream):
    _SPLIT_K_SEMAPHORE_STATE[stream] = (_SPLIT_K_SEMAPHORE_STATE[stream] + 1) % 3


_DTYPE2STR = {torch.float16: "f16", torch.bfloat16: "bf16"}


_ALLOWED_ACTIVATIONS = {"none", "relu", "relu_sq", "gelu_tanh_approx", "silu"}
_ALLOWED_GATES = {"none", "swiglu", "reglu", "geglu", "glu"}
_ALLOWED_DACT = {"none", "relu", "relu_sq", "gelu_tanh_approx", "silu"}
_ALLOWED_DGATED = {"none", "swiglu", "reglu", "geglu", "glu"}


# --- W2: split-K>1 + fused-epilogue dispatch -------------------------------
#
# Fused bias/activation/gated/dact/dgated epilogues are non-distributive over
# split-K atomic-fadd partials, so applying them inside the matmul kernel
# requires SPLIT_K=1. When a problem genuinely benefits from SPLIT_K>1
# (typically small-M, large-K), we can still express the fused semantics
# via one of:
#
#   "force_splitk1"  — default, fallback. SPLIT_K=1 with the fused-in-kernel
#                      epilogue. Loses the split-K tile-balance win.
#   "two_launch"     — run SPLIT_K>1 unfused matmul into a scratch buffer
#                      (atomic-fadd partials into scratch), then apply the
#                      epilogue via torch ops as a cheap elementwise pass.
#                      Extra (M, N) scratch + one HBM round-trip for the
#                      post-hoc epi.
#   "last_partial"   — (future) in-kernel last-partial counter that gates
#                      the epi path: every CU atomic-fadds, the last one to
#                      increment per-tile runs the epi and stores. Zero
#                      extra HBM but divergent control flow in the winner
#                      path.
#
# The dispatch table maps ``(M, N, K)`` to a forced mode. Entries are
# populated by the W3 bench harness. Unlisted shapes default to
# ``"force_splitk1"`` — matching the pre-W2 behaviour.
_SPLITK_EPI_MODE_TABLE: dict = {}


def _splitk_epi_mode(
    M: int, N: int, K: int, has_epilogue: bool, proposed_split_k: int,
    override: Optional[str] = None,
) -> str:
    """Decide how to combine SPLIT_K>1 and a fused epilogue.

    Returns ``"none"`` when the decision is moot (no epilogue, or SPLIT_K=1
    proposed). Otherwise returns one of
    ``"force_splitk1" / "two_launch" / "last_partial"``.
    """
    if not has_epilogue or proposed_split_k <= 1:
        return "none"
    if override is not None:
        return override
    return _SPLITK_EPI_MODE_TABLE.get((M, N, K), "force_splitk1")


def _apply_epi_torch(
    acc: Tensor,
    out: Tensor,
    bias: Optional[Tensor],
    activation: str,
    gate_type: str,
    preact: Optional[Tensor],
    dact_activation: str,
    dgated_gate_type: str,
    dgated_preact: Optional[Tensor],
    emit_postact: bool,
    postact_out: Tensor,
) -> None:
    """Two-launch epilogue — torch-side implementation of the kernel
    write-back transforms. Writes into ``out`` (and ``postact_out`` when
    ``emit_postact``) in place.

    ``acc`` is the raw ``a @ b.T`` of shape ``(M, N)`` in the output
    dtype — what the kernel produces when called with no epilogue args.
    """
    acc32 = acc.float()
    if dgated_gate_type != "none":
        # dgated: acc shape (M, hidden), output dpreact shape (M, 2*hidden).
        M, hidden = acc.shape
        pairs = dgated_preact.view(M, hidden, 2).float()
        gate = pairs[..., 0]
        up = pairs[..., 1]
        if dgated_gate_type == "swiglu":
            sig = torch.sigmoid(gate)
            silu_g = gate * sig
            dsilu_dg = sig * (1.0 + gate * (1.0 - sig))
            dgate = acc32 * dsilu_dg * up
            dup = acc32 * silu_g
            post = silu_g * up
        elif dgated_gate_type == "reglu":
            mask = (gate > 0).float()
            fwd = torch.relu(gate)
            dgate = acc32 * mask * up
            dup = acc32 * fwd
            post = fwd * up
        elif dgated_gate_type == "geglu":
            import math as _m
            c1 = _m.sqrt(2.0 / _m.pi)
            z = c1 * (gate + 0.044715 * gate.pow(3))
            th = torch.tanh(z)
            dz_dg = c1 * (1.0 + 3.0 * 0.044715 * gate.pow(2))
            gelu_g = 0.5 * gate * (1.0 + th)
            dgelu_dg = 0.5 * (1.0 + th) + 0.5 * gate * (1.0 - th * th) * dz_dg
            dgate = acc32 * dgelu_dg * up
            dup = acc32 * gelu_g
            post = gelu_g * up
        elif dgated_gate_type == "glu":
            sig = torch.sigmoid(gate)
            dsig_dg = sig * (1.0 - sig)
            dgate = acc32 * dsig_dg * up
            dup = acc32 * sig
            post = sig * up
        dpreact = torch.stack([dgate, dup], dim=-1).view(M, 2 * hidden)
        out.copy_(dpreact.to(out.dtype))
        if emit_postact:
            postact_out.copy_(post.to(postact_out.dtype))
        return

    if bias is not None:
        acc32 = acc32 + bias  # bias f32, broadcasts last dim

    if gate_type != "none":
        # Interleaved-pair gated: acc cols [2c, 2c+1] = (gate, up).
        M, N = acc.shape
        pairs = acc32.view(M, N // 2, 2)
        gate = pairs[..., 0]
        up = pairs[..., 1]
        if gate_type == "swiglu":
            gated = torch.nn.functional.silu(gate) * up
        elif gate_type == "reglu":
            gated = torch.relu(gate) * up
        elif gate_type == "geglu":
            gated = torch.nn.functional.gelu(gate, approximate="tanh") * up
        elif gate_type == "glu":
            gated = torch.sigmoid(gate) * up
        out.copy_(gated.to(out.dtype))
        return

    if dact_activation != "none":
        # dpreact = acc * act'(preact)
        p32 = preact.float()
        if dact_activation == "relu":
            deriv = (p32 > 0).float()
        elif dact_activation == "relu_sq":
            deriv = 2.0 * p32 * (p32 > 0).float()
        elif dact_activation == "silu":
            sig = torch.sigmoid(p32)
            deriv = sig * (1.0 + p32 * (1.0 - sig))
        elif dact_activation == "gelu_tanh_approx":
            import math as _m
            c1 = _m.sqrt(2.0 / _m.pi)
            z = c1 * (p32 + 0.044715 * p32.pow(3))
            th = torch.tanh(z)
            dz_dp = c1 * (1.0 + 3.0 * 0.044715 * p32.pow(2))
            deriv = 0.5 * (1.0 + th) + 0.5 * p32 * (1.0 - th * th) * dz_dp
        out.copy_((acc32 * deriv).to(out.dtype))
        return

    # Plain bias/activation.
    if activation == "relu":
        acc32 = torch.relu(acc32)
    elif activation == "relu_sq":
        acc32 = torch.relu(acc32) * acc32
    elif activation == "silu":
        acc32 = torch.nn.functional.silu(acc32)
    elif activation == "gelu_tanh_approx":
        acc32 = torch.nn.functional.gelu(acc32, approximate="tanh")
    out.copy_(acc32.to(out.dtype))


def _gemm_splitk_raw(
    a: Tensor, b: Tensor, out: Tensor, shuffled: bool, kwargs: dict,
) -> None:
    """Unfused matmul path — no epilogue, respects caller-supplied
    ``kwargs["SPLIT_K"]`` > 1. Used by the two-launch epi dispatcher.
    Writes ``out[M, N] = a @ b.T``.
    """
    M, K = a.shape
    N, _ = b.shape
    if kwargs["B_PRE_SHUFFLE"] and not shuffled:
        b = shuffle_b(b)
    stream = torch.cuda.current_stream()
    sem, state = _get_semaphore(stream)
    if kwargs["SPLIT_K"] > 1:
        bm = (M + kwargs["TILE_M"] - 1) // kwargs["TILE_M"]
        bn = N // kwargs["TILE_N"]
        assert bm * bn <= SPLIT_K_COUNTER_MAX_LEN
    exe = _compile_hgemm_kernel(
        _DTYPE2STR[a.dtype], N, K, **kwargs,
        has_bias=False, activation="none", gate_type="none",
        dact_activation="none", dgated_gate_type="none",
        emit_postact=False,
        _m_hint=M,
    )
    bias_arg = torch.zeros(1, device=a.device, dtype=torch.float32)
    preact_arg = torch.zeros(1, device=a.device, dtype=a.dtype)
    postact_arg = torch.zeros(1, device=a.device, dtype=a.dtype)
    exe(out, a, b, M, sem, state, bias_arg, preact_arg, postact_arg, stream)
    if kwargs["SPLIT_K"] > 1:
        _advance_state(stream)


@torch.library.custom_op(
    "quack_amd::_gemm_splitk_out",
    mutates_args=("out", "postact_out"),
    schema=(
        "(Tensor a, Tensor b, Tensor(a0!) out, bool shuffled, "
        "Tensor? bias, str activation, str gate_type, "
        "Tensor? preact, str dact_activation, "
        "str dgated_gate_type, Tensor? dgated_preact, "
        "Tensor(a1!) postact_out, bool emit_postact, "
        "str? splitk_epi_mode, int? force_split_k) -> ()"
    ),
)
def _gemm_splitk_out(
    a: Tensor, b: Tensor, out: Tensor, shuffled: bool,
    bias: Optional[Tensor], activation: str, gate_type: str,
    preact: Optional[Tensor], dact_activation: str,
    dgated_gate_type: str,
    dgated_preact: Optional[Tensor],
    postact_out: Tensor,
    emit_postact: bool,
    splitk_epi_mode: Optional[str] = None,
    force_split_k: Optional[int] = None,
) -> None:
    assert a.is_cuda and b.is_cuda and out.is_cuda
    assert a.dtype in (torch.float16, torch.bfloat16) and a.dtype == b.dtype == out.dtype
    assert activation in _ALLOWED_ACTIVATIONS
    assert gate_type in _ALLOWED_GATES
    assert dact_activation in _ALLOWED_DACT
    assert dgated_gate_type in _ALLOWED_DGATED
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2
    is_gated = gate_type != "none"
    is_dact = dact_activation != "none"
    is_dgated = dgated_gate_type != "none"
    if is_dact:
        assert preact is not None and preact.shape == (M, N)
        assert preact.dtype == a.dtype and preact.stride(-1) == 1
        assert not is_gated and activation == "none" and bias is None, (
            "dact epilogue is mutually exclusive with bias/activation/gate_type"
        )
    if is_dgated:
        assert dgated_preact is not None and dgated_preact.shape == (M, 2 * N)
        assert dgated_preact.dtype == a.dtype and dgated_preact.stride(-1) == 1
        assert (
            not is_gated and not is_dact and activation == "none" and bias is None
        ), "dgated epilogue is mutually exclusive with bias/activation/gate_type/dact"
        expected_out_shape = (M, 2 * N)
    elif is_gated:
        assert N % 2 == 0, "gated requires N even (output is (M, N/2))"
        expected_out_shape = (M, N // 2)
    else:
        expected_out_shape = (M, N)
    assert out.shape == expected_out_shape
    assert not (is_gated and activation != "none"), (
        "gate_type subsumes activation; combining both is redundant/unsupported"
    )
    if emit_postact:
        assert is_dgated, "emit_postact is only valid with dgated_gate_type set"
        assert postact_out.shape == (M, N)
        assert postact_out.dtype == a.dtype and postact_out.stride(-1) == 1
    has_bias = bias is not None
    if has_bias:
        assert bias.is_cuda and bias.dtype == torch.float32 and bias.shape == (N,)
    kwargs = _default_kwargs(M, N, K)
    if force_split_k is not None:
        assert K % force_split_k == 0, (
            f"force_split_k={force_split_k} must divide K={K}"
        )
        kwargs = dict(kwargs, SPLIT_K=force_split_k)
    has_epilogue = (
        has_bias or activation != "none" or is_gated or is_dact or is_dgated
    )
    # When SPLIT_K>1 is proposed AND there's an epilogue, pick a strategy.
    # "force_splitk1" (default)  — drop SPLIT_K to 1, fuse epi in kernel.
    # "two_launch"               — run SPLIT_K>1 unfused, apply epi in torch
    #                               as a post-matmul pass.
    epi_mode = _splitk_epi_mode(
        M, N, K, has_epilogue, kwargs["SPLIT_K"], override=splitk_epi_mode,
    )
    if epi_mode == "force_splitk1":
        kwargs = dict(kwargs, SPLIT_K=1)
    elif epi_mode == "last_partial":
        # Follow-up: in-kernel last-partial counter. For now, fall back to
        # force_splitk1 so correctness is preserved.
        kwargs = dict(kwargs, SPLIT_K=1)
    elif epi_mode == "two_launch":
        # Run the unfused SPLIT_K>1 matmul into scratch, then torch epi.
        scratch = torch.zeros(M, N, device=a.device, dtype=a.dtype)
        _gemm_splitk_raw(
            a, b, scratch, shuffled, kwargs,
        )
        _apply_epi_torch(
            scratch, out, bias, activation, gate_type,
            preact, dact_activation,
            dgated_gate_type, dgated_preact,
            emit_postact, postact_out,
        )
        return
    # else: epi_mode == "none" — no epilogue or SPLIT_K was already 1.
    if kwargs["B_PRE_SHUFFLE"] and not shuffled:
        b = shuffle_b(b)
    stream = torch.cuda.current_stream()
    sem, state = _get_semaphore(stream)
    if kwargs["SPLIT_K"] > 1:
        bm = (M + kwargs["TILE_M"] - 1) // kwargs["TILE_M"]
        bn = N // kwargs["TILE_N"]
        assert bm * bn <= SPLIT_K_COUNTER_MAX_LEN
    exe = _compile_hgemm_kernel(
        _DTYPE2STR[a.dtype], N, K, **kwargs,
        has_bias=has_bias, activation=activation, gate_type=gate_type,
        dact_activation=dact_activation,
        dgated_gate_type=dgated_gate_type,
        emit_postact=emit_postact,
        _m_hint=M,
    )
    # Kernels always take Bias / PreAct / Postact arg slots; pass 1-element
    # dummies when the corresponding flag is off.
    bias_arg = bias if has_bias else torch.zeros(
        1, device=a.device, dtype=torch.float32,
    )
    if is_dact:
        preact_arg = preact
    elif is_dgated:
        preact_arg = dgated_preact
    else:
        preact_arg = torch.zeros(1, device=a.device, dtype=a.dtype)
    exe(out, a, b, M, sem, state, bias_arg, preact_arg, postact_out, stream)
    if kwargs["SPLIT_K"] > 1:
        _advance_state(stream)


@_gemm_splitk_out.register_fake
def _gemm_splitk_out_fake(
    a, b, out, shuffled, bias, activation, gate_type,
    preact, dact_activation,
    dgated_gate_type, dgated_preact, postact_out, emit_postact,
    splitk_epi_mode=None, force_split_k=None,
):
    return None


def gemm_splitk(
    a: Tensor, b: Tensor, out: Optional[Tensor] = None,
    *,
    shuffled: bool = False,
    bias: Optional[Tensor] = None,
    activation: str = "none",
    gate_type: str = "none",
    preact: Optional[Tensor] = None,
    dact_activation: str = "none",
    dgated_gate_type: str = "none",
    dgated_preact: Optional[Tensor] = None,
    dgated_emit_postact: bool = False,
    dgated_postact_out: Optional[Tensor] = None,
    splitk_epi_mode: Optional[str] = None,
    force_split_k: Optional[int] = None,
) -> Tensor:
    """Stream-K-capable NT GEMM: ``c = act(a @ b.T + bias)`` on gfx950.

    ``a``: (M, K) f16/bf16, ``b``: (N, K) same dtype (NT layout).
    Returns a (M, N) tensor in the same dtype.

    Args:
      out: optional preallocated (M, N) output.
      shuffled: if True, ``b`` is already in the kernel's preshuffled form.
      bias: optional (N,) f32 tensor; when set, added per-column in the
            write-back before the activation.
      activation: ``"none"``, ``"relu"``, ``"relu_sq"``,
            ``"gelu_tanh_approx"``, or ``"silu"``; applied post-bias.
      dgated_gate_type: if set, fuses the gated-activation backward —
            ``a`` is ``dz`` ``(M, out_dim)``, ``b`` is ``w_down``
            ``(out_dim, hidden)``, ``dgated_preact`` is the saved
            interleaved ``(g, u)`` preact ``(M, 2*hidden)``. The kernel
            returns ``dpreact`` of shape ``(M, 2*hidden)``. When
            ``dgated_emit_postact=True`` (or ``dgated_postact_out`` is
            supplied), also emits ``postact = act(g) * u`` of shape
            ``(M, hidden)``; in that case returns ``(dpreact, postact)``.

    Fused epilogue compiles a distinct kernel per ``(has_bias, activation)``
    combination — the bias-less, no-activation default matches the original
    upstream kernel exactly.
    """
    assert a.is_cuda and b.is_cuda
    assert a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16)
    assert activation in _ALLOWED_ACTIVATIONS
    assert gate_type in _ALLOWED_GATES
    assert dact_activation in _ALLOWED_DACT
    assert dgated_gate_type in _ALLOWED_DGATED
    M, K = a.shape
    N, _ = b.shape
    is_gated = gate_type != "none"
    is_dgated = dgated_gate_type != "none"
    if is_dgated:
        out_n = 2 * N
    elif is_gated:
        out_n = N // 2
    else:
        out_n = N
    if out is None:
        out = torch.empty(M, out_n, device=a.device, dtype=a.dtype)
    emit_postact = is_dgated and (dgated_emit_postact or dgated_postact_out is not None)
    if emit_postact:
        if dgated_postact_out is None:
            dgated_postact_out = torch.empty(M, N, device=a.device, dtype=a.dtype)
        postact_arg = dgated_postact_out
    else:
        postact_arg = torch.empty(1, device=a.device, dtype=a.dtype)
    _gemm_splitk_out(
        a, b, out, shuffled, bias, activation, gate_type,
        preact, dact_activation,
        dgated_gate_type, dgated_preact, postact_arg, emit_postact,
        splitk_epi_mode, force_split_k,
    )
    if emit_postact:
        return out, dgated_postact_out
    return out


__all__ = ["gemm_splitk", "shuffle_b", "interleave_gated_weight"]
