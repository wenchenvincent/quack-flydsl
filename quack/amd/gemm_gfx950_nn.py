# Copyright (c) 2026, AMD.

"""NN-layout f16/bf16 GEMM for gfx950 (CDNA4) — training DX.

Computes ``C[M, N] = A[M, K] @ B[K, N]`` with both operands row-major:
A has K stride-1 (inner), B has N stride-1 (inner). Matches what
``dx = dy @ W`` produces during training — dy is (bs, out) row-major
and W is (out, in) row-major (``nn.Linear.weight``, used transposed
for fwd: ``x @ W.T``). For bwd DX, ``dy`` plays the A role and ``W``
plays the B role.

Architecture (default 128×256×64 tile, 4-warp WG):
  - Output tile: 128 × 256
  - MFMA: 16×16×32 f16/bf16 (gfx950 K=32 per issue)
  - Warps/WG: 1×4 along M/N → 256 threads, each wave owns a 128×64 sub-tile
  - LDS: STAGES=2 ping-pong for BOTH A and B (A is K-inner, B is N-inner)
    - A in LDS: ``(STAGES, BLOCK_M, BLOCK_K)`` XOR-swizzled on K-bytes.
      Same layout as the NT kernel's A.
    - B in LDS: ``(STAGES, BLOCK_K, BLOCK_N)`` XOR-swizzled on N-bytes.
      Write is vectorised (N-contiguous). Read is STRIDED: each MFMA
      lane needs 4 (gfx942) or 8 (gfx950) K-values at one N-column,
      which in an N-inner LDS layout requires that many separate scalar
      loads with stride ``BLOCK_N``. Matches the pattern hipBLASLt uses
      for its own NN kernel (confirmed by rocprofv3: both our NN and
      hipBLASLt's NN issue ~2× LDS ops per MFMA vs the NT kernels).
  - Async DMA on A (raw_ptr_buffer_load_lds), gfx950 only. B uses sync
    ``buffer_load`` into registers, then ``vec_store`` to LDS.
  - Scheduler: sched_vmem / sched_mfma interleaves via ``_OnlineScheduler``.

MVP scope (this module): plain matmul only. No fused bias, no activation,
no split-K. Phase 6 adds fused epilogues (see
``gemm_gfx950_mfma_core._apply_epilogue`` which is already layout-agnostic
and will be called from the NN write-back).

Public API:

    gemm_nn(a, b, out=None)
        a:  (M, K) f16/bf16, K stride-1
        b:  (K, N) f16/bf16, N stride-1
        out: optional preallocated (M, N) in same dtype as ``a``
"""

import functools
from typing import Optional

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, memref
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from quack.amd.flydsl_tensor_shim import GTensor, STensor, get_dtype_in_kernel
from quack.amd.flydsl_utils import get_rocm_arch
from quack.amd.gemm_gfx950_mfma_core import (
    _OnlineScheduler,
    _WmmaHalfK16,
    _WmmaHalfK32,
    swizzle_xor16,
)


_DTYPE2STR = {torch.float16: "f16", torch.bfloat16: "bf16"}


@functools.lru_cache(maxsize=1024)
def _compile_nn_kernel(
    dtype: str,
    k: int,
    n: int,
    TILE_M: int = 128,
    TILE_N: int = 256,
    TILE_K: int = 64,
    BLOCK_M_WARPS: int = 1,
    BLOCK_N_WARPS: int = 4,
    _m_hint: int = 0,  # cache-key only; see splitk for grid-bake workaround rationale
):
    BLOCK_K = TILE_K
    assert BLOCK_K >= 32
    assert k % BLOCK_K == 0

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

    BLOCK_K_LOOPS = k // BLOCK_K
    WARP_K_STEPS = BLOCK_K // WARP_ATOM_K
    assert BLOCK_K % WARP_ATOM_K == 0 and WARP_K_STEPS >= 1
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE
    WARP_M_STEPS = TILE_M // BLOCK_M_WARPS // WARP_ATOM_M
    WARP_N_STEPS = TILE_N // BLOCK_N_WARPS // WARP_ATOM_N
    assert WARP_M_STEPS >= 1 and WARP_N_STEPS >= 1
    assert TILE_M % (BLOCK_M_WARPS * WARP_ATOM_M) == 0
    assert TILE_N % (BLOCK_N_WARPS * WARP_ATOM_N) == 0
    WARP_M = WARP_M_STEPS * WARP_ATOM_M
    WARP_N = WARP_N_STEPS * WARP_ATOM_N
    BLOCK_M = BLOCK_M_WARPS * WARP_M
    BLOCK_N = BLOCK_N_WARPS * WARP_N
    assert n >= BLOCK_N and n % BLOCK_N == 0

    BLOCK_MK_SIZE = BLOCK_M * BLOCK_K
    BLOCK_NK_SIZE = BLOCK_N * BLOCK_K
    BLOCK_MN_SIZE = BLOCK_M * BLOCK_N

    # Load-per-thread fanout.
    LDG_A_X_THREADS = BLOCK_K // LDG_VEC_SIZE            # A: K-inner (same as NT)
    LDG_B_X_THREADS = BLOCK_N // LDG_VEC_SIZE            # B: N-inner (NN-specific)
    LDG_C_X_THREADS = BLOCK_N // LDG_VEC_SIZE
    BLOCK_VECS = LDG_VEC_SIZE * BLOCK_THREADS
    LDG_REG_A_COUNT = BLOCK_MK_SIZE // BLOCK_VECS
    LDG_REG_B_COUNT = BLOCK_NK_SIZE // BLOCK_VECS
    LDG_REG_C_COUNT = BLOCK_MN_SIZE // BLOCK_VECS
    assert LDG_REG_A_COUNT >= 1 and LDG_REG_B_COUNT >= 1 and LDG_REG_C_COUNT >= 1
    assert BLOCK_MK_SIZE % BLOCK_VECS == 0
    assert BLOCK_NK_SIZE % BLOCK_VECS == 0
    assert BLOCK_MN_SIZE % BLOCK_VECS == 0

    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES
    BLOCK_N_BYTES = BLOCK_N * DTYPE_BYTES

    # gfx950 async-DMA per-thread fanout.
    LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES
    LDG_A_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    LDG_REG_A_COUNT_AS = BLOCK_MK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS

    # LDS B pad: break row-stride-to-bank-stride alignment on the B side.
    # Bank stride = 128 bytes (32 banks × 4 bytes). BLOCK_N_BYTES = 512 (N=256)
    # is a bank multiple, so stride-8 rows in the B LDS layout land on the
    # same banks → 4-way conflict on every MFMA B-fragment read (135M
    # conflict cycles per call at 8192×16384×4096 bf16). Adding 8 f16 of
    # pad (16 bytes) bumps stride to 544 which mismatches bank period —
    # stride-8 rows now land 2 banks apart, cutting the conflict to 2-way
    # (~34M cycles) and yielding 13-17% kernel speedup.
    # A-side is left unpadded: the existing swizzle_xor16(row, col, k_blocks16)
    # encoding depends on BLOCK_K_BYTES; padding A's K stride would require
    # re-deriving the swizzle's key, deferred to future tune-in.
    B_LDS_PAD = 8 if (BLOCK_N * DTYPE_BYTES) % 128 == 0 else 0
    BS_N_STRIDE = BLOCK_N + B_LDS_PAD

    allocator = SmemAllocator(
        None, arch=GPU_ARCH,
        global_sym_name=f"nn_smem_{dtype}_{k}_{n}",
    )
    smem_a_offset = allocator._align(allocator.ptr, 16)
    AS_BYTES = STAGES * BLOCK_M * BLOCK_K * DTYPE_BYTES
    # C staging (writeback) reuses A's LDS region but needs BLOCK_M*BLOCK_N
    # bytes which may exceed BLOCK_M*BLOCK_K — take the max.
    AS_BYTES = max(AS_BYTES, BLOCK_M * BLOCK_N * DTYPE_BYTES)
    allocator.ptr = smem_a_offset + AS_BYTES
    smem_b_offset = allocator._align(allocator.ptr, 16)
    BS_BYTES = STAGES * BLOCK_K * BS_N_STRIDE * DTYPE_BYTES
    allocator.ptr = smem_b_offset + BS_BYTES

    KERNEL_NAME = f"nn_{dtype}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_S{STAGES}"
    KERNEL_NAME += "_AS" if ASYNC_COPY else "_NA"

    @flyc.kernel
    def nn_kernel(C: fx.Tensor, A: fx.Tensor, B: fx.Tensor, m: fx.Int32):
        dtype_ = get_dtype_in_kernel(dtype)
        c_zero_d = arith.constant(0.0, type=dtype_)
        acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG_VALUES, T.f32))

        # Global tensors.
        A_ = GTensor(A, dtype=dtype_, shape=(-1, k))       # (M, K) row-major, K-inner
        B_ = GTensor(B, dtype=dtype_, shape=(k, n))        # (K, N) row-major, N-inner
        C_ = GTensor(C, dtype=dtype_, shape=(-1, n))       # (M, N) row-major

        # LDS tensors.
        base_ptr = allocator.get_base()
        smem_a_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(STAGES * BLOCK_M * BLOCK_K,))
        as_ = STensor(smem_a_ptr, dtype_, shape=(STAGES, BLOCK_M, BLOCK_K))
        smem_b_ptr = SmemPtr(base_ptr, smem_b_offset, dtype_, shape=(STAGES * BLOCK_K * BS_N_STRIDE,))
        # Last dim is BS_N_STRIDE (= BLOCK_N + PAD); the pad slots are never
        # written/read but break the row-stride-to-bank-stride alignment.
        bs_ = STensor(smem_b_ptr, dtype_, shape=(STAGES, BLOCK_K, BS_N_STRIDE))
        # C writeback-time LDS (aliases A's region, BLOCK_M × BLOCK_N)
        smem_c_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(BLOCK_M * BLOCK_N,))
        cs_ = STensor(smem_c_ptr, dtype_, shape=(BLOCK_M, BLOCK_N))

        tid = fx.Int32(fx.thread_idx.x)
        wid = tid // WARP_SIZE
        w_tid = tid % WARP_SIZE
        block_m_idx = fx.block_idx.x
        block_n_idx = fx.block_idx.y
        m_offset = fx.Index(block_m_idx * BLOCK_M)
        n_offset = fx.Index(block_n_idx * BLOCK_N)
        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)
        n_blocks16 = fx.Int32(BLOCK_N_BYTES // 16)

        warp_m_idx = wid // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid % BLOCK_N_WARPS * WARP_N

        # Per-lane intra-fragment indices.
        #  A fragment: lane owns M=lane%16, K-values starting at 4*(lane//16)
        ldmatrix_a_m_idx = w_tid % WMMA_M
        ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K
        #  B fragment: lane owns N=lane%16, K-values starting at 4*(lane//16)
        ldmatrix_b_n_idx = w_tid % WMMA_N
        ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K

        A_FRAGS_LEN = WARP_K_STEPS * WARP_M_STEPS
        B_FRAGS_LEN = WARP_K_STEPS * WARP_N_STEPS
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
        c_frags = [acc_init] * C_FRAGS_LEN

        # ---------- A-side load path (identical to NT splitk kernel) ----------

        def ldg_a(k_offset):
            """HBM → register chunks. Each thread loads LDG_VEC_SIZE K-contig f16s."""
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
            """Register chunks → LDS with K-byte XOR swizzle."""
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
            """Async HBM → LDS on gfx950 (raw_ptr_buffer_load_lds). No reg roundtrip."""
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
            """LDS → register MFMA A fragment. Each lane reads FRAG K-contig values."""
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

        # ---------- B-side load path (NN-specific: N-inner HBM, LDS-staged) ----------

        def ldg_b(k_offset):
            """HBM → register. Each thread loads LDG_VEC_SIZE N-contig f16s for one K row."""
            vecs = []
            for i in range_constexpr(LDG_REG_B_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                k_local_idx = global_tid // LDG_B_X_THREADS
                n_local_idx = global_tid % LDG_B_X_THREADS * LDG_VEC_SIZE
                row_idx = fx.Index(k_offset + k_local_idx)        # K row (global)
                col_idx = n_offset + fx.Index(n_local_idx)         # N col (global)
                vec = B_.vec_load((row_idx, col_idx), LDG_VEC_SIZE)
                vecs.append(vec)
            return vecs

        def sts_b(vecs, lds_stage):
            """Register → LDS with N-byte XOR swizzle. LDS layout is (STAGES, BLOCK_K, BLOCK_N)."""
            for i in range_constexpr(LDG_REG_B_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                k_local_idx = global_tid // LDG_B_X_THREADS
                n_local_idx = global_tid % LDG_B_X_THREADS * LDG_VEC_SIZE
                col_in_bytes = n_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(k_local_idx, col_in_bytes, n_blocks16)
                bs_.vec_store(
                    (fx.Index(lds_stage), k_local_idx, col_in_bytes // DTYPE_BYTES),
                    vecs[i], LDG_VEC_SIZE,
                )

        def lds_matrix_b(lds_stage):
            """LDS → register MFMA B fragment. Strided scalar loads (stride=BLOCK_N).

            Each lane owns one N column and needs FRAG K-values (from different
            K rows). In the N-inner LDS layout, these K-values are at stride
            BLOCK_N — so ``FRAG`` separate scalar loads per fragment. The XOR
            swizzle on N-bytes ensures each of those scalar loads lands in a
            distinct bank per lane.
            """
            s = fx.Index(lds_stage)
            b_frags = [0] * (WARP_K_STEPS * WARP_N_STEPS)
            FRAG = WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
            for kk in range_constexpr(WARP_K_STEPS):
                for jj in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    k_start = warp_atom_k_idx + ldmatrix_b_k_vec_idx  # lane's K start
                    n_col = warp_atom_n_idx + ldmatrix_b_n_idx        # lane's N col
                    values = []
                    for ii in range_constexpr(FRAG):
                        row = k_start + ii
                        col_in_bytes = n_col * DTYPE_BYTES
                        col_in_bytes = swizzle_xor16(row, col_in_bytes, n_blocks16)
                        v = bs_[s, fx.Index(row), fx.Index(col_in_bytes // DTYPE_BYTES)]
                        values.append(v)
                    vec = vector.from_elements(T.vec(FRAG, dtype_), values)
                    b_frags[kk * WARP_N_STEPS + jj] = vec
            return b_frags

        # ---------- MFMA inner loop (identical to NT splitk kernel) ----------

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

        def hot_loop_scheduler():
            MFMA_TOTAL = WARP_K_STEPS * WARP_M_STEPS * WARP_N_STEPS * MFMA_PER_WARP_K
            LDG_REG_A_COUNT_ = LDG_REG_A_COUNT_AS if ASYNC_COPY else LDG_REG_A_COUNT
            # Total vmem ops per iter: A loads + B loads.
            LDG_TOTAL = LDG_REG_A_COUNT_ + LDG_REG_B_COUNT
            mfma_ = _OnlineScheduler(MFMA_TOTAL, MFMA_TOTAL)
            ldg_ = _OnlineScheduler(LDG_TOTAL, LDG_TOTAL)
            if ASYNC_COPY:
                AVG_MFMA_COUNT = (MFMA_TOTAL + LDG_TOTAL - 1) // LDG_TOTAL
                for _ in range_constexpr(LDG_TOTAL):
                    rocdl.sched_vmem(ldg_.consume(1))
                    rocdl.sched_mfma(mfma_.consume(AVG_MFMA_COUNT))
            else:
                LDG_STS_TOTAL = LDG_TOTAL + LDG_REG_A_COUNT_ + LDG_REG_B_COUNT
                AVG_MFMA_COUNT = (MFMA_TOTAL + LDG_STS_TOTAL - 1) // LDG_STS_TOTAL
                for _ in range_constexpr(LDG_TOTAL):
                    rocdl.sched_vmem(ldg_.consume(1))
                    rocdl.sched_mfma(mfma_.consume(AVG_MFMA_COUNT))
                for _ in range_constexpr(LDG_REG_A_COUNT_ + LDG_REG_B_COUNT):
                    rocdl.sched_dswr(1)
                    rocdl.sched_mfma(mfma_.consume(AVG_MFMA_COUNT))
            rocdl.sched_barrier(0)

        # ---------- Hot loop (prelude + scf.for over K) ----------

        # Prelude: stage-0 loads for A and B.
        k_begin = arith.constant(0, type=T.i32)
        if ASYNC_COPY:
            ldg_sts_a_async(k_begin, 0)
        else:
            sts_a(ldg_a(k_begin), 0)
        b_regs0 = ldg_b(k_begin)
        sts_b(b_regs0, 0)
        gpu.barrier()
        a_frags = lds_matrix_a(0)
        b_frags = lds_matrix_b(0)
        rocdl.sched_barrier(0)

        init_state = (
            [k_begin, arith.constant(0, index=True)]
            + c_frags + a_frags + b_frags
        )
        for _bki, state in range(1, BLOCK_K_LOOPS, init=init_state):
            k_offset = state[0]
            current_stage = fx.Index(state[1])
            next_stage = 1 - current_stage
            c_frags = state[2 : 2 + C_FRAGS_LEN]
            a_frags = state[2 + C_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN]
            b_frags = state[2 + C_FRAGS_LEN + A_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN + B_FRAGS_LEN]
            # Next K-tile loads (overlap with current MFMA).
            if ASYNC_COPY:
                ldg_sts_a_async(k_offset + BLOCK_K, next_stage)
            else:
                a_regs_next = ldg_a(k_offset + BLOCK_K)
            b_regs_next = ldg_b(k_offset + BLOCK_K)
            block_mma_sync(a_frags, b_frags, c_frags)
            if not ASYNC_COPY:
                sts_a(a_regs_next, next_stage)
            sts_b(b_regs_next, next_stage)
            hot_loop_scheduler()
            gpu.barrier()
            a_frags_next = lds_matrix_a(next_stage)
            b_frags_next = lds_matrix_b(next_stage)
            k_offset = k_offset + fx.Int32(BLOCK_K)
            rocdl.sched_barrier(0)
            results = yield [k_offset, next_stage] + c_frags + a_frags_next + b_frags_next
        c_frags = results[2 : 2 + C_FRAGS_LEN]
        a_frags = results[2 + C_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN]
        b_frags = results[2 + C_FRAGS_LEN + A_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN + B_FRAGS_LEN]
        block_mma_sync(a_frags, b_frags, c_frags)

        # ---------- Write-back (acc → LDS → HBM) ----------

        # Acc regs → LDS (BLOCK_M × BLOCK_N). Each MFMA C-fragment lane owns 4
        # M-contiguous values at its N column (standard CDNA C-layout).
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

        gpu.barrier()

        # LDS → HBM. Each thread writes LDG_VEC_SIZE N-contig values.
        for i in range_constexpr(LDG_REG_C_COUNT):
            global_tid = BLOCK_THREADS * i + tid
            m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
            n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
            m_global_idx = m_offset + m_local_idx
            cond_boundary = arith.cmpi(arith.CmpIPredicate.ult, m_global_idx, fx.Index(m))
            from flydsl._mlir.dialects import scf
            cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
            with ir.InsertionPoint(cond_boundary_if.then_block):
                vec = cs_.vec_load((m_local_idx, n_local_idx), LDG_VEC_SIZE)
                C_.vec_store(
                    (m_global_idx, n_offset + n_local_idx), vec, LDG_VEC_SIZE,
                )
                scf.YieldOp([])

    @flyc.jit
    def launch_nn_kernel(
        C: fx.Tensor, A: fx.Tensor, B: fx.Tensor,
        m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        bm = (m + BLOCK_M - 1) // BLOCK_M
        bn = n // BLOCK_N
        nn_kernel._func.__name__ = KERNEL_NAME
        launcher = nn_kernel(C, A, B, m)
        launcher.launch(grid=(bm, bn, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_nn_kernel


# ---------- Public API + torch.library registration ----------


@torch.library.custom_op(
    "quack_amd::_gemm_nn_out",
    mutates_args=("out",),
    schema="(Tensor a, Tensor b, Tensor(a0!) out) -> ()",
)
def _gemm_nn_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    assert a.is_cuda and b.is_cuda and out.is_cuda
    assert a.dim() == 2 and b.dim() == 2 and out.dim() == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"A @ B requires inner dims to match: {a.shape} @ {b.shape}"
    assert out.shape == (M, N), f"out shape {out.shape} != (M, N) = ({M}, {N})"
    assert a.dtype == b.dtype == out.dtype
    assert a.stride(-1) == 1 and b.stride(-1) == 1 and out.stride(-1) == 1
    dtype_str = _DTYPE2STR[a.dtype]
    _compile_nn_kernel(dtype_str, K, N, _m_hint=M)(out, a, b, M)


@_gemm_nn_out.register_fake
def _gemm_nn_out_fake(a, b, out):
    return None


def gemm_nn(a: Tensor, b: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """``C[M, N] = A[M, K] @ B[K, N]`` for f16/bf16 row-major inputs.

    Both operands row-major; A has K stride-1 (inner), B has N stride-1.
    This is the DX layout: ``dx = dy @ W`` — dy plays A, W plays B.

    Constraints (MVP):
      - M multiple of 128; N multiple of 256; K multiple of 64.
      - dtype in {f16, bf16}; output dtype matches input.
    """
    M, K = a.shape
    _, N = b.shape
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=a.dtype)
    _gemm_nn_out(a, b, out)
    return out


__all__ = ["gemm_nn"]
