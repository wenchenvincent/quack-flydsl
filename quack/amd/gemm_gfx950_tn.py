# Copyright (c) 2026, AMD.

"""TN-layout f16/bf16 GEMM for gfx950 (CDNA4) — training DW.

Computes ``C[M, N] = A^T_stored @ B`` where:
  ``A`` is stored as ``(K, M)`` row-major (M-inner) — think of it as
    ``dy`` with shape ``(batch, out)``, where axis 0 is the contraction.
  ``B`` is stored as ``(K, N)`` row-major (N-inner) — ``x`` with shape
    ``(batch, in)``.
  ``C`` is ``(M, N)`` row-major (N-inner) — ``dW`` with shape ``(out, in)``.

Computation: ``C[m, n] = sum_k A[k, m] * B[k, n]``. This is the DW
layout in training: ``dW = dy.T @ x``. The caller passes ``dy`` and
``x`` **without** transposing — the kernel interprets axis 0 as the
contraction dimension.

Architecture (default 128×256×64 tile, 4-warp WG):
  - Output tile: 128 × 256
  - MFMA: 16×16×32 f16/bf16 (gfx950 K=32 per issue)
  - Warps/WG: 1×4 along M/N → 256 threads, each wave owns 128×64 sub-tile
  - LDS staging for BOTH A and B (both have contraction OUTER):
    - A in LDS: ``(STAGES, BLOCK_K, BLOCK_M)`` XOR-swizzled on M-bytes
    - B in LDS: ``(STAGES, BLOCK_K, BLOCK_N)`` XOR-swizzled on N-bytes
    - HBM → LDS vec stores (M-contig for A, N-contig for B) — coalesced
    - LDS → MFMA fragment: STRIDED scalar loads (both sides, since lane
      needs FRAG K-values at its M or N column, stride = BLOCK_M or BLOCK_N)
  - Async DMA via ``raw_ptr_buffer_load_lds`` on A (gfx950 only). B uses
    sync buffer_load into regs, then ``vec_store`` to LDS.

MVP scope: plain matmul only. Phase 6 adds fused bias / act / dact /
dgated (via ``_apply_*`` helpers from ``gemm_gfx950_mfma_core``).

Perf note: like the NN kernel MVP, the strided-scalar LDS reads on
BOTH sides bound perf below Tier-1 until a tune-in pass restructures
the LDS read to vectorised (packed-K layout, ds_read_b128_tr, or
cooperative-warp transpose).

Public API:

    gemm_tn(dy, x, out=None)
        dy:  (K, M) f16/bf16 — in training, ``dy`` with shape (batch, out)
        x:   (K, N) f16/bf16 — in training, ``x`` with shape (batch, in)
        out: optional preallocated (M, N) in same dtype
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
from quack.amd import _gemm_tune


_DTYPE2STR = {torch.float16: "f16", torch.bfloat16: "bf16"}


@functools.lru_cache(maxsize=1024)
def _compile_tn_kernel(
    dtype: str,
    k: int,
    m: int,
    n: int,
    TILE_M: int = 128,
    TILE_N: int = 256,
    TILE_K: int = 64,
    BLOCK_M_WARPS: int = 1,
    BLOCK_N_WARPS: int = 4,
    # OGS-style grid swizzle; see gemm_gfx950_nn.py for the rationale.
    # TN's M is compile-time (comes from the dw shape), so both grid dims
    # are constexpr and swizzle2d / xcd_swizzle can fold to near-static.
    XCD_SWIZZLE: int = 8,
    GROUP_M: int = 4,
):
    BLOCK_K = TILE_K
    assert BLOCK_K >= 32
    assert k % BLOCK_K == 0, f"k={k} must be multiple of BLOCK_K={BLOCK_K}"

    GPU_ARCH = get_rocm_arch()
    if fx.const_expr(GPU_ARCH == "gfx942"):
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
    assert m >= BLOCK_M and m % BLOCK_M == 0, f"m={m} must be multiple of BLOCK_M={BLOCK_M}"
    assert n >= BLOCK_N and n % BLOCK_N == 0, f"n={n} must be multiple of BLOCK_N={BLOCK_N}"

    BLOCK_MK_SIZE = BLOCK_M * BLOCK_K
    BLOCK_NK_SIZE = BLOCK_N * BLOCK_K
    BLOCK_MN_SIZE = BLOCK_M * BLOCK_N

    LDG_A_X_THREADS = BLOCK_M // LDG_VEC_SIZE
    LDG_B_X_THREADS = BLOCK_N // LDG_VEC_SIZE
    LDG_C_X_THREADS = BLOCK_N // LDG_VEC_SIZE
    BLOCK_VECS = LDG_VEC_SIZE * BLOCK_THREADS
    LDG_REG_A_COUNT = BLOCK_MK_SIZE // BLOCK_VECS
    LDG_REG_B_COUNT = BLOCK_NK_SIZE // BLOCK_VECS
    LDG_REG_C_COUNT = BLOCK_MN_SIZE // BLOCK_VECS
    assert LDG_REG_A_COUNT >= 1 and LDG_REG_B_COUNT >= 1 and LDG_REG_C_COUNT >= 1
    assert BLOCK_MK_SIZE % BLOCK_VECS == 0
    assert BLOCK_NK_SIZE % BLOCK_VECS == 0
    assert BLOCK_MN_SIZE % BLOCK_VECS == 0

    BLOCK_M_BYTES = BLOCK_M * DTYPE_BYTES
    BLOCK_N_BYTES = BLOCK_N * DTYPE_BYTES

    LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES
    LDG_A_X_THREADS_AS = BLOCK_M // LDG_ASYNC_VEC_SIZE
    LDG_REG_A_COUNT_AS = BLOCK_MK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS

    # LDS B pad: break the 4-way bank conflict on MFMA B-fragment reads.
    # Matches the NN kernel's fix — 8 f16 of pad makes BLOCK_N_BYTES+16=544
    # mismatch the 128-byte bank period, so stride-8 rows across k_groups
    # land on different banks.
    # A-side pad is deferred: the existing swizzle_xor16(row, col, m_blocks16)
    # key depends on BLOCK_M_BYTES; padding A's M stride without re-deriving
    # the swizzle breaks correctness. Future tune-in.
    B_LDS_PAD = 8 if (BLOCK_N * DTYPE_BYTES) % 128 == 0 else 0
    BS_N_STRIDE = BLOCK_N + B_LDS_PAD

    allocator = SmemAllocator(
        None, arch=GPU_ARCH,
        global_sym_name=f"tn_smem_{dtype}_{k}_{m}_{n}",
    )
    smem_a_offset = allocator._align(allocator.ptr, 16)
    AS_BYTES = STAGES * BLOCK_K * BLOCK_M * DTYPE_BYTES
    AS_BYTES = max(AS_BYTES, BLOCK_M * BLOCK_N * DTYPE_BYTES)
    allocator.ptr = smem_a_offset + AS_BYTES
    smem_b_offset = allocator._align(allocator.ptr, 16)
    BS_BYTES = STAGES * BLOCK_K * BS_N_STRIDE * DTYPE_BYTES
    allocator.ptr = smem_b_offset + BS_BYTES

    KERNEL_NAME = f"tn_{dtype}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_S{STAGES}"
    KERNEL_NAME += "_AS" if ASYNC_COPY else "_NA"

    @flyc.kernel
    def tn_kernel(C: fx.Tensor, A: fx.Tensor, B: fx.Tensor):
        dtype_ = get_dtype_in_kernel(dtype)
        acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG_VALUES, T.f32))

        A_ = GTensor(A, dtype=dtype_, shape=(k, m))
        B_ = GTensor(B, dtype=dtype_, shape=(k, n))
        C_ = GTensor(C, dtype=dtype_, shape=(m, n))

        base_ptr = allocator.get_base()
        smem_a_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(STAGES * BLOCK_K * BLOCK_M,))
        as_ = STensor(smem_a_ptr, dtype_, shape=(STAGES, BLOCK_K, BLOCK_M))
        smem_b_ptr = SmemPtr(base_ptr, smem_b_offset, dtype_, shape=(STAGES * BLOCK_K * BS_N_STRIDE,))
        bs_ = STensor(smem_b_ptr, dtype_, shape=(STAGES, BLOCK_K, BS_N_STRIDE))
        smem_c_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(BLOCK_M * BLOCK_N,))
        cs_ = STensor(smem_c_ptr, dtype_, shape=(BLOCK_M, BLOCK_N))

        tid = fx.Int32(fx.thread_idx.x)
        wid = tid // WARP_SIZE
        w_tid = tid % WARP_SIZE

        # 1D grid: decode flat tile id into (pid_m, pid_n) via xcd_swizzle +
        # swizzle2d (see gemm_gfx950_nn.py). Both bm and bn are compile-time
        # constants on TN (m and n come from the dw-shape closure), so most
        # of the arithmetic folds.
        flat_pid = fx.Int32(fx.block_idx.x)
        bm_c_int = m // BLOCK_M
        bn_c_int = n // BLOCK_N
        bn_c = fx.Int32(bn_c_int)
        bm_c = fx.Int32(bm_c_int)
        if fx.const_expr(XCD_SWIZZLE > 1):
            total_int = bm_c_int * bn_c_int
            xcd_c = fx.Int32(XCD_SWIZZLE)
            pids_per_group = fx.Int32(total_int // XCD_SWIZZLE)
            extra_pids = fx.Int32(total_int % XCD_SWIZZLE)
            xcd_group = flat_pid % xcd_c
            xcd_local = flat_pid // xcd_c
            min_ge = arith.select(
                arith.cmpi(arith.CmpIPredicate.slt, xcd_group, extra_pids),
                xcd_group, extra_pids,
            )
            pid = xcd_group * pids_per_group + fx.Int32(min_ge) + xcd_local
        else:
            pid = flat_pid
        if fx.const_expr(GROUP_M > 1):
            gm_c = fx.Int32(GROUP_M)
            width_c = fx.Int32(GROUP_M * bn_c_int)
            group_id = pid // width_c
            remain = bm_c - group_id * gm_c
            group_size = arith.select(
                arith.cmpi(arith.CmpIPredicate.slt, remain, gm_c),
                remain, gm_c,
            )
            group_size_i = fx.Int32(group_size)
            block_m_idx = group_id * gm_c + (pid % group_size_i)
            block_n_idx = (pid % width_c) // group_size_i
        else:
            block_m_idx = pid // bn_c
            block_n_idx = pid % bn_c
        m_offset = fx.Index(block_m_idx * BLOCK_M)
        n_offset = fx.Index(block_n_idx * BLOCK_N)
        m_blocks16 = fx.Int32(BLOCK_M_BYTES // 16)
        n_blocks16 = fx.Int32(BLOCK_N_BYTES // 16)

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

        # ---------- A-side (M-inner, contract=K outer) ----------

        def ldg_a(k_offset):
            vecs = []
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                k_local_idx = global_tid // LDG_A_X_THREADS
                m_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                row_idx = fx.Index(k_offset + k_local_idx)
                col_idx = m_offset + fx.Index(m_local_idx)
                vec = A_.vec_load((row_idx, col_idx), LDG_VEC_SIZE)
                vecs.append(vec)
            return vecs

        def sts_a(vecs, lds_stage):
            """Register → LDS (no swizzle; tr16_b64 read does intra-block transpose)."""
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                k_local_idx = global_tid // LDG_A_X_THREADS
                m_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                as_.vec_store(
                    (fx.Index(lds_stage), k_local_idx, m_local_idx),
                    vecs[i], LDG_VEC_SIZE,
                )

        def ldg_sts_a_async(k_offset, lds_stage):
            """Async HBM → LDS with no swizzle."""
            for i in range_constexpr(LDG_REG_A_COUNT_AS):
                global_tid = BLOCK_THREADS * i + tid
                k_local_idx = global_tid // LDG_A_X_THREADS_AS
                m_local_idx = global_tid % LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
                row_idx = fx.Index(k_offset + k_local_idx)
                col_idx = m_offset + fx.Index(m_local_idx)
                global_offset = A_.linear_offset((row_idx, col_idx)) * DTYPE_BYTES
                global_offset = arith.index_cast(T.i32, global_offset)
                lds_offset = as_.linear_offset(
                    (fx.Index(lds_stage), k_local_idx, m_local_idx)
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
            """LDS → MFMA A fragment via ds_read_tr16_b64 — symmetric to NN's B-side."""
            s = fx.Index(lds_stage)
            a_frags = [0] * (WARP_K_STEPS * WARP_M_STEPS)
            FRAG = WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K
            assert FRAG == 8
            v4_type = T.vec(4, dtype_)
            v8_type = T.vec(FRAG, dtype_)
            lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
            lb = w_tid % WMMA_M
            sg = lb // 4
            pr = lb % 4
            block_offset = w_tid // WMMA_M
            for kk in range_constexpr(WARP_K_STEPS):
                for ii in range_constexpr(WARP_M_STEPS):
                    warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    halves = []
                    for r in range_constexpr(2):
                        row = warp_atom_k_idx + block_offset * 8 + r * 4 + sg
                        col = warp_atom_m_idx + pr * 4
                        lds_byte_offset = as_.linear_offset(
                            (s, fx.Index(row), fx.Index(col))
                        ) * DTYPE_BYTES
                        lds_base = memref.extract_aligned_pointer_as_index(as_.memptr)
                        lds_addr_idx = lds_base + lds_byte_offset
                        lds_addr_i64 = arith.index_cast(T.i64, lds_addr_idx)
                        lds_ptr = llvm.inttoptr(lds_ptr_type, lds_addr_i64)
                        v4 = rocdl.ds_read_tr16_b64(v4_type, lds_ptr).result
                        halves.append(v4)
                    elems = []
                    for h in range_constexpr(2):
                        for e in range_constexpr(4):
                            elems.append(vector.extract(
                                halves[h], static_position=[e], dynamic_position=[],
                            ))
                    vec = vector.from_elements(v8_type, elems)
                    a_frags[kk * WARP_M_STEPS + ii] = vec
            return a_frags

        # ---------- B-side (N-inner, contract=K outer) ----------

        def ldg_b(k_offset):
            vecs = []
            for i in range_constexpr(LDG_REG_B_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                k_local_idx = global_tid // LDG_B_X_THREADS
                n_local_idx = global_tid % LDG_B_X_THREADS * LDG_VEC_SIZE
                row_idx = fx.Index(k_offset + k_local_idx)
                col_idx = n_offset + fx.Index(n_local_idx)
                vec = B_.vec_load((row_idx, col_idx), LDG_VEC_SIZE)
                vecs.append(vec)
            return vecs

        def sts_b(vecs, lds_stage):
            """Register → LDS, no swizzle (ds_read_tr16_b64 does intra-block transpose)."""
            for i in range_constexpr(LDG_REG_B_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                k_local_idx = global_tid // LDG_B_X_THREADS
                n_local_idx = global_tid % LDG_B_X_THREADS * LDG_VEC_SIZE
                bs_.vec_store(
                    (fx.Index(lds_stage), k_local_idx, n_local_idx),
                    vecs[i], LDG_VEC_SIZE,
                )

        def lds_matrix_b(lds_stage):
            """LDS → MFMA B fragment via ds_read_tr16_b64 — see NN kernel for semantics."""
            s = fx.Index(lds_stage)
            b_frags = [0] * (WARP_K_STEPS * WARP_N_STEPS)
            FRAG = WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
            assert FRAG == 8
            v4_type = T.vec(4, dtype_)
            v8_type = T.vec(FRAG, dtype_)
            lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
            lb = w_tid % WMMA_N
            sg = lb // 4
            pr = lb % 4
            block_offset = w_tid // WMMA_N
            for kk in range_constexpr(WARP_K_STEPS):
                for jj in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    halves = []
                    for r in range_constexpr(2):
                        row = warp_atom_k_idx + block_offset * 8 + r * 4 + sg
                        col = warp_atom_n_idx + pr * 4
                        lds_byte_offset = bs_.linear_offset(
                            (s, fx.Index(row), fx.Index(col))
                        ) * DTYPE_BYTES
                        lds_base = memref.extract_aligned_pointer_as_index(bs_.memptr)
                        lds_addr_idx = lds_base + lds_byte_offset
                        lds_addr_i64 = arith.index_cast(T.i64, lds_addr_idx)
                        lds_ptr = llvm.inttoptr(lds_ptr_type, lds_addr_i64)
                        v4 = rocdl.ds_read_tr16_b64(v4_type, lds_ptr).result
                        halves.append(v4)
                    elems = []
                    for h in range_constexpr(2):
                        for e in range_constexpr(4):
                            elems.append(vector.extract(
                                halves[h], static_position=[e], dynamic_position=[],
                            ))
                    vec = vector.from_elements(v8_type, elems)
                    b_frags[kk * WARP_N_STEPS + jj] = vec
            return b_frags

        # ---------- MFMA inner loop ----------

        def block_mma_sync(a_frags, b_frags, c_frags):
            for kk in range_constexpr(WARP_K_STEPS):
                for ii in range_constexpr(WARP_M_STEPS):
                    a_frag = a_frags[kk * WARP_M_STEPS + ii]
                    for jj in range_constexpr(WARP_N_STEPS):
                        b_frag = b_frags[kk * WARP_N_STEPS + jj]
                        if fx.const_expr(MFMA_PER_WARP_K == 2):
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
            LDG_TOTAL = LDG_REG_A_COUNT_ + LDG_REG_B_COUNT
            mfma_ = _OnlineScheduler(MFMA_TOTAL, MFMA_TOTAL)
            ldg_ = _OnlineScheduler(LDG_TOTAL, LDG_TOTAL)
            if ASYNC_COPY:
                # async-copy path: A via buffer_load_lds (no sts_a), but B
                # still does ldg_b → sts_b. Include LDG_REG_B_COUNT dswr hints
                # so the compiler schedules the B LDS writes against MFMA
                # (was missing — MFMA idled during dswr previously).
                LDG_STS_TOTAL = LDG_TOTAL + LDG_REG_B_COUNT
                AVG_MFMA_COUNT = (MFMA_TOTAL + LDG_STS_TOTAL - 1) // LDG_STS_TOTAL
                for _ in range_constexpr(LDG_TOTAL):
                    rocdl.sched_vmem(ldg_.consume(1))
                    rocdl.sched_mfma(mfma_.consume(AVG_MFMA_COUNT))
                for _ in range_constexpr(LDG_REG_B_COUNT):
                    rocdl.sched_dswr(1)
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

        # ---------- Hot loop ----------

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

        # ---------- Write-back ----------

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

        for i in range_constexpr(LDG_REG_C_COUNT):
            global_tid = BLOCK_THREADS * i + tid
            m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
            n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
            m_global_idx = m_offset + m_local_idx
            vec = cs_.vec_load((m_local_idx, n_local_idx), LDG_VEC_SIZE)
            C_.vec_store(
                (m_global_idx, n_offset + n_local_idx), vec, LDG_VEC_SIZE,
            )

    @flyc.jit
    def launch_tn_kernel(
        C: fx.Tensor, A: fx.Tensor, B: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        # Occupancy hint: matches NN kernel (see wave_per_eu comment there).
        # Empirically best at 3 for our (128, 256, 64) tile on gfx950.
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, 3)
        bm = m // BLOCK_M
        bn = n // BLOCK_N
        tn_kernel._func.__name__ = KERNEL_NAME
        launcher = tn_kernel(C, A, B)
        # 1D grid; in-kernel xcd_swizzle + swizzle2d derive (pid_m, pid_n).
        launcher.launch(grid=(bm * bn, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_tn_kernel


# ---------- Per-shape autotune (see gemm_gfx950_nn.py for full docstrings) ----------


_TN_CANDIDATES = [
    (128, 128, 64, 2, 2),   # default for M%128 && N%128 — 2 waves/EU unlock
    (128, 128, 64, 1, 4),
    (256, 128, 64, 2, 2),   # sometimes wins at M>=8192
    (128, 256, 64, 1, 4),   # MVP-era default
    (128, 128, 128, 1, 2),  # deeper BLOCK_K
    (128, 128, 128, 2, 2),
    (128, 64, 64, 2, 2),
    (64, 128, 64, 1, 4),
    (64, 256, 64, 1, 4),
    (256, 64, 64, 2, 2),
]


def _shape_fits_tn(M: int, N: int, cfg: _gemm_tune.Config) -> bool:
    tm, tn, _, _, _ = cfg
    return M % tm == 0 and N % tn == 0 and M >= tm and N >= tn


def _heuristic_config_tn(M: int, N: int) -> _gemm_tune.Config:
    for cfg in _TN_CANDIDATES:
        if _shape_fits_tn(M, N, cfg):
            return cfg
    return (128, 256, 64, 1, 4)


def _pick_config_tn(
    dtype_str: str, M: int, K: int, N: int,
    a: Tensor, b: Tensor, out: Tensor,
) -> _gemm_tune.Config:
    key = ("tn", dtype_str, M, K, N)
    cached = _gemm_tune.get_cached_config(key)
    if cached is not None:
        return cached
    if _gemm_tune.get_autotune():
        cfg, _ = _autotune_tn_impl(dtype_str, M, K, N, a, b, out, verbose=False)
        _gemm_tune.set_cached_config(key, cfg)
        return cfg
    return _heuristic_config_tn(M, N)


def _autotune_tn_impl(
    dtype_str: str, M: int, K: int, N: int,
    a: Tensor, b: Tensor, out: Tensor,
    verbose: bool = False,
) -> tuple:
    candidates = [c for c in _TN_CANDIDATES if _shape_fits_tn(M, N, c)]
    if not candidates:
        return _heuristic_config_tn(M, N), float("inf")

    def launch_factory(cfg):
        tm, tn, tk, bmw, bnw = cfg
        k_fn = _compile_tn_kernel(
            dtype_str, K, M, N, TILE_M=tm, TILE_N=tn, TILE_K=tk,
            BLOCK_M_WARPS=bmw, BLOCK_N_WARPS=bnw,
        )
        return lambda: k_fn(out, a, b)

    return _gemm_tune.search_best_config(
        "tn", dtype_str, M, K, N, candidates, launch_factory, verbose=verbose,
    )


def autotune_tn(
    a: Tensor, b: Tensor, out: Optional[Tensor] = None, verbose: bool = False,
) -> _gemm_tune.Config:
    """Autotune ``gemm_tn(a, b)`` — see ``autotune_nn`` for full docstring."""
    assert a.is_cuda and b.is_cuda
    assert a.dim() == 2 and b.dim() == 2
    K, M = a.shape
    K2, N = b.shape
    assert K == K2
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=a.dtype)
    dtype_str = _DTYPE2STR[a.dtype]
    cfg, t = _autotune_tn_impl(dtype_str, M, K, N, a, b, out, verbose=verbose)
    _gemm_tune.set_cached_config(("tn", dtype_str, M, K, N), cfg)
    if verbose:
        flops = 2 * M * N * K / 1e12
        tf = flops / t if t > 0 else 0
        print(f"autotune_tn {dtype_str} K={K} {M}×{N} → {cfg} at {tf:.1f} TF/s ({t*1e6:.1f} μs)")
    return cfg


# ---------- Public API ----------


@torch.library.custom_op(
    "quack_amd::_gemm_tn_out",
    mutates_args=("out",),
    schema="(Tensor a, Tensor b, Tensor(a0!) out) -> ()",
)
def _gemm_tn_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    assert a.is_cuda and b.is_cuda and out.is_cuda
    assert a.dim() == 2 and b.dim() == 2 and out.dim() == 2
    K, M = a.shape
    K2, N = b.shape
    assert K == K2, f"dy.T @ x requires same K (axis 0): dy {a.shape}, x {b.shape}"
    assert out.shape == (M, N), f"out shape {out.shape} != (M, N) = ({M}, {N})"
    assert a.dtype == b.dtype == out.dtype
    assert a.stride(-1) == 1 and b.stride(-1) == 1 and out.stride(-1) == 1
    dtype_str = _DTYPE2STR[a.dtype]
    config = _pick_config_tn(dtype_str, M, K, N, a, b, out)
    tm, tn, tk, bmw, bnw = config
    _compile_tn_kernel(
        dtype_str, K, M, N, TILE_M=tm, TILE_N=tn, TILE_K=tk,
        BLOCK_M_WARPS=bmw, BLOCK_N_WARPS=bnw,
    )(out, a, b)


@_gemm_tn_out.register_fake
def _gemm_tn_out_fake(a, b, out):
    return None


def gemm_tn(a: Tensor, b: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Compute ``C[M, N] = A.T @ B`` where A shape is (K, M), B shape is (K, N).

    In training:
      - ``a`` is ``dy`` with shape ``(batch, out)``
      - ``b`` is ``x`` with shape ``(batch, in)``
      - returns ``dW`` with shape ``(out, in)``

    The contraction axis is axis 0 of both operands (the batch dim).

    Constraints (MVP):
      - K (= batch) multiple of 64; M (= out) multiple of 128; N (= in) multiple of 256.
      - dtype ∈ {f16, bf16}; output dtype matches input.
    """
    K, M = a.shape
    _, N = b.shape
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=a.dtype)
    _gemm_tn_out(a, b, out)
    return out


__all__ = ["gemm_tn", "autotune_tn"]
