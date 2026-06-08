# Copyright (c) 2026, AMD.

"""NT-layout bf16/f16 GEMM with HipKittens 8-wave ping-pong (gfx950).

Computes ``C[M, N] = A @ B^T`` where:
  ``A`` is stored as ``(M, K)`` row-major (K-inner)
  ``B`` is stored as ``(N, K)`` row-major (K-inner) — NT layout
  ``C`` is ``(M, N)`` row-major (N-inner)

This is the cuBLAS-style ``nn.Linear`` forward layout: weight matrix W
stored as ``(out_features, in_features)``, activation X as
``(batch, in_features)``, output as ``Y = X @ W^T``.

Architecture (mirrors HipKittens ``256_256_64_32_with32x16.cpp``):
  - Output tile: 256 × 256
  - K-step: 64, split into 2 sub-K iters of DOT_SLICE = 32 each
  - 8 warps in 2×4 grid (warp_row, warp_col) — each owns 128 × 64
  - 4 clusters per K-step: (load-kk0, MFMA-kk0, load-kk1, MFMA-kk1)
  - Initial desync via vestigial ``s_sleep`` on warp_row==1; with only
    1 barrier per K-step the compiler-driven ILP within a single wave's
    instruction stream is what does most of the work, so the desync is
    not load-bearing for THIS kernel. (HK's conditional ``s_barrier``
    pattern — ``if warp_row==1: s_barrier()`` — actually DOES work at
    1 WG/CU on gfx950: warp_row==1 waves take the extra barrier and
    pair with warp_row==0's next barrier through the shared WG counter.
    ATT-verified in ``gemm_gfx950_nt_pingpong_16x32.py``. Earlier
    speculation that the conditional pattern required 2 WG/CU was
    wrong; both mechanisms work at 1 WG/CU.)
  - ``s_setprio(1)/(0)`` wrapping each MFMA
  - MFMA atom: ``mfma_f32_16x16x32_bf16`` (K=32 per atom, matches DOT_SLICE)
  - Async HBM → LDS via ``raw_ptr_buffer_load_lds`` for both A and B
    (both K-inner → symmetric load paths, no ``ds_read_tr16_b64`` needed
    for B's transpose unlike the NN kernel)
  - AGPR passthrough (``amdgpu-agpr-alloc=128,128``) for accumulators

Public API:

    gemm_nt_pingpong(a, b, out=None)
        a: (M, K) bf16/f16
        b: (N, K) bf16/f16 — NT layout (``c = a @ b.T``)
        out: optional preallocated (M, N) bf16/f16

Same shape constraints as the NN big kernel: M % 256, N % 256, K % 64.
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
from quack.amd.gemm_gfx950_mfma_core import _WmmaHalfK32, swizzle_xor16


_DTYPE2STR = {torch.float16: "f16", torch.bfloat16: "bf16"}


@functools.lru_cache(maxsize=1024)
def _compile_nt_pingpong_kernel(
    dtype: str, k: int, n: int, _m_hint: int = 0,
    XCD_SWIZZLE: int = 1,
    GROUP_M: int = 1,
):
    """Compile the NT ping-pong kernel for a specific (dtype, K, N).

    The ``_m_hint`` is part of the cache key only — M is runtime-dynamic.

    ``XCD_SWIZZLE`` and ``GROUP_M`` are tile-scheduling knobs (see the
    swizzle block in the kernel body). Both default to ``1`` (disabled)
    because the pingpong's symmetric K-inner A/B loads and
    compute-hides-L2-miss pipeline make tile-locality remapping
    perf-neutral here — sweeps over ``XCD ∈ {1,4,8} × GM ∈ {1,4,8}``
    move perf ±2% with no clear winner, and (1,1) wins at 8192³. The
    hooks are kept for future tuning if the schedule changes.
    """
    # ----- Tile + warp constants -----
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 64
    BLOCK_M_WARPS = 2
    BLOCK_N_WARPS = 4
    WARP_SIZE = 64
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE   # = 512
    WARP_M = BLOCK_M // BLOCK_M_WARPS                            # = 128
    WARP_N = BLOCK_N // BLOCK_N_WARPS                            # =  64

    # ----- MFMA atom (16x16x32 bf16/f16 on gfx950) -----
    WMMA_IMPL = _WmmaHalfK32(dtype)
    WMMA_M = WMMA_IMPL.WMMA_M                                    # = 16
    WMMA_N = WMMA_IMPL.WMMA_N                                    # = 16
    WMMA_K = WMMA_IMPL.WMMA_K                                    # = 32
    WMMA_A_FRAG = WMMA_IMPL.WMMA_A_FRAG_VALUES                   # =  8
    WMMA_B_FRAG = WMMA_IMPL.WMMA_B_FRAG_VALUES                   # =  8
    WMMA_C_FRAG = WMMA_IMPL.WMMA_C_FRAG_VALUES                   # =  4

    WARP_M_STEPS = WARP_M // WMMA_M                              # =  8
    WARP_N_STEPS = WARP_N // WMMA_N                              # =  4
    # DOT_SLICE = WMMA_K = 32 → one K-step has BLOCK_K/WMMA_K = 2 sub-K iters
    K_SUBITERS = BLOCK_K // WMMA_K                               # =  2
    assert K_SUBITERS == 2, "kernel structured around 2 sub-K iters per K-step"

    DTYPE_BYTES = 2
    STAGES = 2

    # ----- HBM→LDS DMA constants -----
    LDG_VEC_SIZE = 8                                              # 8 bf16 / 16 bytes per thread
    DMA_BYTES = LDG_VEC_SIZE * DTYPE_BYTES                        # = 16
    # For both A and B (K-inner), per-thread tile: 8 K-contig bf16.
    # 64 threads per K-row of LDS (= BLOCK_K / LDG_VEC_SIZE = 8 threads per row).
    LDG_X_THREADS = BLOCK_K // LDG_VEC_SIZE                       # =  8
    # 512 threads × 8 elems = 4096 elems per call. Tile is BLOCK_OUTER × BLOCK_K
    # = 256 × 64 = 16384 elems. → 4 calls per stage.
    LDG_A_REG_COUNT = (BLOCK_M * BLOCK_K) // (LDG_VEC_SIZE * BLOCK_THREADS)
    LDG_B_REG_COUNT = (BLOCK_N * BLOCK_K) // (LDG_VEC_SIZE * BLOCK_THREADS)
    assert LDG_A_REG_COUNT == 4 and LDG_B_REG_COUNT == 4

    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES

    # ----- LDS allocation -----
    # A: (STAGES, BLOCK_M=256, BLOCK_K=64)        →  64 KB
    # B: (STAGES, BLOCK_N=256, BLOCK_K=64)        →  64 KB
    # Total: 128 KB, within 160 KB cap.
    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(
        None, arch=GPU_ARCH,
        global_sym_name=f"nt_pp_smem_{dtype}_{k}_{n}",
    )

    smem_a_offset = allocator._align(allocator.ptr, 16)
    AS_BYTES = STAGES * BLOCK_M * BLOCK_K * DTYPE_BYTES
    allocator.ptr = smem_a_offset + AS_BYTES

    smem_b_offset = allocator._align(allocator.ptr, 16)
    BS_BYTES = STAGES * BLOCK_N * BLOCK_K * DTYPE_BYTES
    allocator.ptr = smem_b_offset + BS_BYTES

    BLOCK_K_LOOPS_HINT = max(1, k // BLOCK_K)

    KERNEL_NAME = f"nt_pp_{dtype}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}"

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def nt_kernel(C: fx.Tensor, A: fx.Tensor, B: fx.Tensor, m: fx.Int32):
        dtype_ = get_dtype_in_kernel(dtype)
        acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG, T.f32))

        # ----- Global tensor descriptors -----
        A_ = GTensor(A, dtype=dtype_, shape=(-1, k))    # (M, K) K-inner
        B_ = GTensor(B, dtype=dtype_, shape=(n, k))     # (N, K) K-inner — NT
        C_ = GTensor(C, dtype=dtype_, shape=(-1, n))    # (M, N) N-inner

        # ----- LDS descriptors -----
        base_ptr = allocator.get_base()
        smem_a_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(STAGES * BLOCK_M * BLOCK_K,))
        as_ = STensor(smem_a_ptr, dtype_, shape=(STAGES, BLOCK_M, BLOCK_K))
        smem_b_ptr = SmemPtr(base_ptr, smem_b_offset, dtype_, shape=(STAGES * BLOCK_N * BLOCK_K,))
        bs_ = STensor(smem_b_ptr, dtype_, shape=(STAGES, BLOCK_N, BLOCK_K))

        # ----- Tile coords (1D grid → 2D via 2-stage swizzle) -----
        # (1) XCD swizzle (OGS-style): remap flat pid so each XCD owns one
        #     contiguous chunk → adjacent WGs share L2.
        # (2) GROUP_M swizzle2d: group GROUP_M consecutive M-tiles per
        #     N-block → adjacent WGs reuse A rows. (HK ``WGM=8``.)
        flat_pid = fx.Int32(fx.block_idx.x)
        bn_c = fx.Int32(n // BLOCK_N)
        bm_rt = (m + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M)
        if fx.const_expr(XCD_SWIZZLE > 1):
            xcd_c = fx.Int32(XCD_SWIZZLE)
            total_tiles = bm_rt * bn_c
            pids_per_group = total_tiles // xcd_c
            extra_pids = total_tiles % xcd_c
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
            width = gm_c * bn_c
            group_id = pid // width
            remain = bm_rt - group_id * gm_c
            group_size = arith.select(
                arith.cmpi(arith.CmpIPredicate.slt, remain, gm_c),
                remain, gm_c,
            )
            group_size_i = fx.Int32(group_size)
            block_m_idx = group_id * gm_c + (pid % group_size_i)
            block_n_idx = (pid % width) // group_size_i
        else:
            block_m_idx = pid // bn_c
            block_n_idx = pid % bn_c
        m_offset = fx.Index(block_m_idx * BLOCK_M)
        n_offset = fx.Index(block_n_idx * BLOCK_N)
        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)

        # ----- Thread / warp indices -----
        tid = fx.Int32(fx.thread_idx.x)
        wid = tid // WARP_SIZE
        w_tid = tid % WARP_SIZE
        warp_m_idx = wid // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid % BLOCK_N_WARPS * WARP_N
        # warp_row in {0, 1} drives the HK ping-pong initial desync.
        warp_row = wid // BLOCK_N_WARPS

        # Per-lane MFMA-fragment indices.
        # For 16x16x32 atom in row_l: 4 lanes per M-row, each holds 8 K-contig.
        lane_m_idx = w_tid % WMMA_M                          # 0..15
        lane_k_vec_idx = (w_tid // WMMA_M) * WMMA_A_FRAG     # 0, 8, 16, 24

        # =========================================================
        # Helper: async HBM → LDS for A (raw_ptr_buffer_load_lds).
        # Each thread DMAs 16 bytes (= 8 bf16) of K-contig data.
        # XOR-swizzle on K bytes to mitigate ds_read bank conflicts.
        # =========================================================
        def ldg_sts_a_async(k_offset, lds_stage):
            for i in range_constexpr(LDG_A_REG_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_X_THREADS
                k_local_idx = global_tid % LDG_X_THREADS * LDG_VEC_SIZE
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

        # =========================================================
        # Helper: async HBM → LDS for B (same pattern as A; B is also
        # K-inner so the access pattern is symmetric — no transpose,
        # no ds_read_tr16_b64 indirection like the NN kernel needs).
        # =========================================================
        def ldg_sts_b_async(k_offset, lds_stage):
            for i in range_constexpr(LDG_B_REG_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                n_local_idx = global_tid // LDG_X_THREADS
                k_local_idx = global_tid % LDG_X_THREADS * LDG_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
                row_idx = n_offset + fx.Index(n_local_idx)
                # N alignment is guaranteed by the public-API check (N % 256 == 0).
                col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
                global_offset = B_.linear_offset((row_idx, col_idx)) * DTYPE_BYTES
                global_offset = arith.index_cast(T.i32, global_offset)
                lds_offset = bs_.linear_offset(
                    (fx.Index(lds_stage), n_local_idx, k_local_idx)
                ) * DTYPE_BYTES
                lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                lds_addr = memref.extract_aligned_pointer_as_index(bs_.memptr) + lds_offset
                lds_addr_ = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
                lds_ptr = llvm.inttoptr(lds_ptr_type, lds_addr_)
                rocdl.raw_ptr_buffer_load_lds(
                    B_.rsrc, lds_ptr,
                    arith.constant(DMA_BYTES, type=T.i32),
                    global_offset,
                    arith.constant(0, type=T.i32),
                    arith.constant(0, type=T.i32),
                    arith.constant(1, type=T.i32),
                )

        # =========================================================
        # Helper: LDS → register A fragments for ONE sub-K (kk_target).
        # Returns WARP_M_STEPS frags (one per atom in M direction).
        # =========================================================
        def lds_matrix_a_kk(lds_stage, kk_target):
            s = fx.Index(lds_stage)
            a_frags = [0] * WARP_M_STEPS
            for ii in range_constexpr(WARP_M_STEPS):
                warp_atom_m_idx = warp_m_idx + ii * WMMA_M
                warp_atom_k_idx = kk_target * WMMA_K
                row = warp_atom_m_idx + lane_m_idx
                col_in_bytes = (warp_atom_k_idx + lane_k_vec_idx) * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                vec = as_.vec_load(
                    (s, row, col_in_bytes // DTYPE_BYTES), WMMA_A_FRAG,
                )
                a_frags[ii] = vec
            return a_frags

        # =========================================================
        # Helper: LDS → register B fragments for ONE sub-K (kk_target).
        # For NT, the LDS layout (N, K) is symmetric to A's (M, K) — so
        # lds_matrix_b uses the same per-lane K-contig pattern as A,
        # just indexed by N rather than M.
        # =========================================================
        def lds_matrix_b_kk(lds_stage, kk_target):
            s = fx.Index(lds_stage)
            b_frags = [0] * WARP_N_STEPS
            for jj in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + jj * WMMA_N
                warp_atom_k_idx = kk_target * WMMA_K
                row = warp_atom_n_idx + lane_m_idx     # lane_m_idx mapping reused (N now)
                col_in_bytes = (warp_atom_k_idx + lane_k_vec_idx) * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                vec = bs_.vec_load(
                    (s, row, col_in_bytes // DTYPE_BYTES), WMMA_B_FRAG,
                )
                b_frags[jj] = vec
            return b_frags

        # =========================================================
        # Helper: MFMA for one sub-K. Wraps with s_setprio(1)/(0)
        # for the HK ping-pong (asymmetric wave population biased
        # by the warp_row==1 initial s_sleep).
        # =========================================================
        def mma_kk(a_frags, b_frags, c_frags):
            rocdl.s_setprio(1)
            for ii in range_constexpr(WARP_M_STEPS):
                a_frag = a_frags[ii]
                for jj in range_constexpr(WARP_N_STEPS):
                    b_frag = b_frags[jj]
                    c_idx = ii * WARP_N_STEPS + jj
                    c_frags[c_idx] = WMMA_IMPL(a_frag, b_frag, c_frags[c_idx])
            rocdl.s_setprio(0)

        # ----- Accumulator init -----
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
        c_frags = [acc_init] * C_FRAGS_LEN

        # s_waitcnt bitfield (gfx9): bits[3:0]=vmcnt_lo, bits[6:4]=expcnt,
        # bits[11:8]=lgkmcnt, bits[15:14]=vmcnt_hi. "No wait" on a field
        # means encoding all-ones for that field.
        VMCNT_0 = 0x0F70  # vmcnt=0, expcnt=7 (no wait), lgkmcnt=15 (no wait)

        # ----- Initial s_sleep desync (vestigial in 1-barrier-per-iter) -----
        # NOT HK's conditional s_barrier — that creates a stagger of one
        # *barrier interval*, which here is a full K-step. Two warp groups
        # would end up on different K-iterations and read mismatched LDS
        # stages. The conditional s_barrier pattern only gives correct
        # stagger when there are ≥2 barriers per K-step (1-cluster lag);
        # this kernel has 1 barrier per K-step, so we keep s_sleep instead.
        if arith.cmpi(arith.CmpIPredicate.eq, warp_row, fx.Int32(1)):
            rocdl.s_sleep(16)

        # ----- Prologue: HBM→LDS iter 0 → stage 0 -----
        # Use bare ``rocdl.s_barrier()`` instead of ``gpu.barrier()`` to
        # skip the release/acquire fence pair MLIR auto-inserts. The
        # vmcnt(0) drain we DO need (so cross-wave LDS reads see the
        # prefetched data) is explicit.
        k_begin = arith.constant(0, type=T.i32)
        ldg_sts_a_async(k_begin, 0)
        ldg_sts_b_async(k_begin, 0)
        rocdl.s_waitcnt(VMCNT_0)
        rocdl.s_barrier()

        # ----- Main loop: 4 clusters per K-step -----
        BLOCK_K_LOOPS = k // BLOCK_K
        init_state = [k_begin, arith.constant(0, index=True)] + c_frags
        for _bki, state in range(1, BLOCK_K_LOOPS, init=init_state):
            k_offset = state[0]
            current_stage = fx.Index(state[1])
            next_stage = 1 - current_stage
            c_frags = state[2 : 2 + C_FRAGS_LEN]

            # Cluster 0: load kk=0 operands + prefetch HBM for next iter.
            a_frags_kk0 = lds_matrix_a_kk(current_stage, 0)
            b_frags_kk0 = lds_matrix_b_kk(current_stage, 0)
            ldg_sts_a_async(k_offset + BLOCK_K, next_stage)
            ldg_sts_b_async(k_offset + BLOCK_K, next_stage)

            # Cluster 1: MFMA on kk=0 (setprio inside mma_kk).
            mma_kk(a_frags_kk0, b_frags_kk0, c_frags)

            # Cluster 2: load kk=1 operands from current_stage.
            a_frags_kk1 = lds_matrix_a_kk(current_stage, 1)
            b_frags_kk1 = lds_matrix_b_kk(current_stage, 1)

            # Cluster 3: MFMA on kk=1.
            mma_kk(a_frags_kk1, b_frags_kk1, c_frags)

            # End-of-iter barrier: drains the async prefetches' vmcnt
            # so the next iter's LDS reads see committed data. Bare
            # rocdl.s_barrier (no fence pair); explicit vmcnt(0) preserved.
            rocdl.s_waitcnt(VMCNT_0)
            rocdl.s_barrier()

            k_offset = k_offset + fx.Int32(BLOCK_K)
            rocdl.sched_barrier(0)
            results = yield [k_offset, next_stage] + c_frags

        # ----- Epilogue: process the final iter (no prefetch) -----
        c_frags = results[2 : 2 + C_FRAGS_LEN]
        final_stage = (BLOCK_K_LOOPS - 1) % 2
        a_frags_kk0_last = lds_matrix_a_kk(final_stage, 0)
        b_frags_kk0_last = lds_matrix_b_kk(final_stage, 0)
        mma_kk(a_frags_kk0_last, b_frags_kk0_last, c_frags)
        a_frags_kk1_last = lds_matrix_a_kk(final_stage, 1)
        b_frags_kk1_last = lds_matrix_b_kk(final_stage, 1)
        mma_kk(a_frags_kk1_last, b_frags_kk1_last, c_frags)

        # =========================================================
        # Writeback: each c_frag is vec<4 x f32>. MFMA 16x16x32 output
        # layout: lane L's 4 values live at (row = (L // 16) * 4 + e,
        # col = L % 16) for e=0..3. For each e, lanes 0..15 hit the same
        # row at cols 0..15 — the SIMD hardware coalesces those 16 short
        # stores into one cache-line transaction. Adding ds_bpermute to
        # repack into vec<4 x bf16> per lane regresses ~5% here (the 16
        # bpermutes/atom × 32 atoms = 512 LDS shuffles outweigh the savings,
        # since the simple pattern is already coalesced). HK's reference
        # also uses scalar stores. Keep it simple.
        # =========================================================
        from flydsl._mlir.dialects import scf
        for ii in range_constexpr(WARP_M_STEPS):
            for jj in range_constexpr(WARP_N_STEPS):
                c_idx = ii * WARP_N_STEPS + jj
                c_vec = c_frags[c_idx]
                lane_col = w_tid % WMMA_N                       # 0..15
                lane_row_base = (w_tid // WMMA_N) * 4           # 0, 4, 8, 12
                warp_atom_m_idx = warp_m_idx + ii * WMMA_M
                warp_atom_n_idx = warp_n_idx + jj * WMMA_N
                n_local = warp_atom_n_idx + lane_col
                n_global = n_offset + fx.Index(n_local)
                for e in range_constexpr(WMMA_C_FRAG):
                    m_local = warp_atom_m_idx + lane_row_base + e
                    m_global = m_offset + fx.Index(m_local)
                    cond = arith.cmpi(
                        arith.CmpIPredicate.ult, m_global, fx.Index(m),
                    )
                    val_f32 = vector.extract(c_vec, static_position=[e], dynamic_position=[])
                    val_dtype = arith.truncf(dtype_, val_f32)
                    cond_if = scf.IfOp(cond, results_=[], has_else=False)
                    with ir.InsertionPoint(cond_if.then_block):
                        C_.vec_store((m_global, n_global), val_dtype, 1)
                        scf.YieldOp([])

    @flyc.jit
    def launch_nt_kernel(
        C: fx.Tensor, A: fx.Tensor, B: fx.Tensor,
        m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        bm = (m + BLOCK_M - 1) // BLOCK_M
        bn = n // BLOCK_N
        total_tiles = bm * bn
        nt_kernel._func.__name__ = KERNEL_NAME
        launcher = nt_kernel(C, A, B, m)
        # AGPR passthrough — let MFMA accumulators live in AGPRs, freeing
        # VGPRs for operands. Mirrors the NN big kernel's mechanism.
        passthrough_attr = ir.ArrayAttr.get([
            ir.ArrayAttr.get([
                ir.StringAttr.get("amdgpu-agpr-alloc"),
                ir.StringAttr.get("128,128"),
            ]),
        ])
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, 3)
                op.attributes["passthrough"] = passthrough_attr
        launcher.launch(grid=(total_tiles, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_nt_kernel


# ----- Public API -----

def gemm_nt_pingpong(
    a: Tensor, b: Tensor, out: Optional[Tensor] = None,
) -> Tensor:
    """HK 8-wave ping-pong NT GEMM (gfx950).

    Computes ``C = A @ B^T`` where ``A`` is ``(M, K)`` row-major and
    ``B`` is ``(N, K)`` row-major (NT layout — the nn.Linear-forward
    pattern). Output ``(M, N)`` row-major in the input dtype.

    Constraints: M % 256 == 0, N % 256 == 0, K % 64 == 0.
    """
    assert a.is_cuda and b.is_cuda
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2, f"inner dims must match: A={a.shape} B={b.shape}"
    assert a.dtype == b.dtype and a.dtype in _DTYPE2STR
    assert M % 256 == 0 and N % 256 == 0 and K % 64 == 0, (
        f"gemm_nt_pingpong requires M%256, N%256, K%64; got {M}×{K}×{N}"
    )
    assert a.stride(-1) == 1 and b.stride(-1) == 1
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=a.dtype)
    else:
        assert out.shape == (M, N) and out.dtype == a.dtype
        assert out.stride(-1) == 1
    dtype_str = _DTYPE2STR[a.dtype]
    _compile_nt_pingpong_kernel(dtype_str, K, N, _m_hint=M)(out, a, b, M)
    return out


__all__ = ["gemm_nt_pingpong"]
