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
  - Initial desync via HK's conditional ``s_barrier`` on warp_row==1.
    The 4 barriers per K-step structure gives a per-cluster rhythm; the
    extra warp_row==1 barrier in the prologue pairs with warp_row==0's
    next barrier through the shared WG counter, locking the two groups
    one cluster out-of-phase forever after. Result: when warp_row==0 is
    MFMA-busy, warp_row==1 is LDS-busy, and vice versa — true pingpong.
    Works at 1 WG/CU on gfx950 (ATT-verified, both here and in the
    sibling 16x32 kernel).
  - ``s_setprio(1)/(0)`` wrapping each MFMA
  - MFMA atom: ``mfma_f32_16x16x32_bf16`` (K=32 per atom, matches DOT_SLICE)
  - Async HBM → LDS via ``raw_ptr_buffer_load_lds`` for both A and B
    (both K-inner → symmetric load paths, no ``ds_read_tr16_b64`` needed
    for B's transpose unlike the NN kernel)
  - No AGPR forcing — at 8 waves/WG and waves_per_eu=2, each wave has
    256 VGPRs; the compiler fits everything in ~210-240 VGPRs with 0
    AGPRs (HK's profile). Forcing AGPR=128,128 splits the unified pool
    and costs ~6% perf via extra VGPR↔AGPR copies.

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
from quack.amd.flydsl_tensor_shim import GTensor, get_dtype_in_kernel
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

    # ----- LDS allocation — per-stage globals + alias scopes -----
    # Four LDS globals: A staging × 2 stages, B staging × 2 stages. Each
    # stage gets its own ``llvm.mlir.global`` + its own alias scope so
    # ds_reads of stage N don't drain pending buffer_load_lds writes to
    # stage N^1 (the prefetch destination). See the sibling 16x32 kernel
    # for the equivalent pattern + a longer explanation.
    AS_STAGE_BYTES = BLOCK_M * BLOCK_K * DTYPE_BYTES      # 32 KB
    BS_STAGE_BYTES = BLOCK_N * BLOCK_K * DTYPE_BYTES      # 32 KB
    LDS_SYMS_A = (
        f"nt_pp_smem_as0_{dtype}_{k}_{n}",
        f"nt_pp_smem_as1_{dtype}_{k}_{n}",
    )
    LDS_SYMS_B = (
        f"nt_pp_smem_bs0_{dtype}_{k}_{n}",
        f"nt_pp_smem_bs1_{dtype}_{k}_{n}",
    )
    LDS_ALIAS_DOMAIN = f'#llvm.alias_scope_domain<id = "nt_pp_{dtype}_{k}_{n}.lds">'
    SCOPE_IDS = ("as0", "as1", "bs0", "bs1")

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

        # ----- LDS descriptors (provenance + per-stage alias scopes) -----
        _LDS_PTR_TY = ir.Type.parse("!llvm.ptr<3>")
        _I8_TY = T.i8
        _GEP_DYN = -(2 ** 31)

        def _gep_lds(base_ptr, byte_offset_i32):
            return llvm.getelementptr(
                _LDS_PTR_TY, base_ptr, [byte_offset_i32], [_GEP_DYN], _I8_TY, None,
            )

        def _scope_attr(ids):
            inner = ", ".join(
                f'#llvm.alias_scope<id = "{i}", domain = {LDS_ALIAS_DOMAIN}>'
                for i in ids
            )
            return ir.Attribute.parse(f"[{inner}]")

        _SCOPE = {sid: _scope_attr((sid,)) for sid in SCOPE_IDS}
        _NOALIAS = {sid: _scope_attr(tuple(o for o in SCOPE_IDS if o != sid))
                    for sid in SCOPE_IDS}

        _as_bases = (
            llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYMS_A[0]),
            llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYMS_A[1]),
        )
        _bs_bases = (
            llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYMS_B[0]),
            llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYMS_B[1]),
        )

        _as_scopes = (_SCOPE["as0"], _SCOPE["as1"])
        _as_noalias = (_NOALIAS["as0"], _NOALIAS["as1"])
        _bs_scopes = (_SCOPE["bs0"], _SCOPE["bs1"])
        _bs_noalias = (_NOALIAS["bs0"], _NOALIAS["bs1"])

        def _make_lds_view_2d(base_ptr, shape, my_scope, other_scopes):
            stride = []
            s = 1
            for sh in reversed(shape):
                stride.insert(0, s)
                s *= sh
            stride = tuple(stride)

            def linear_offset(idxs):
                if not isinstance(idxs, tuple):
                    idxs = (idxs,)
                offset = idxs[0] * stride[0]
                for i in range_constexpr(1, len(idxs)):
                    offset = offset + idxs[i] * stride[i]
                return offset

            def vec_load(idxs, vec_size):
                elem_off = linear_offset(idxs)
                byte_off_idx = elem_off * DTYPE_BYTES
                byte_off_i32 = arith.index_cast(T.i32, byte_off_idx)
                gep = _gep_lds(base_ptr, byte_off_i32)
                vec_t = T.vec(vec_size, dtype_)
                return llvm.LoadOp(
                    vec_t, gep, alignment=2,
                    alias_scopes=my_scope, noalias_scopes=other_scopes,
                ).result

            ns = type("LDSView", (), {})()
            ns.base_ptr = base_ptr
            ns.shape = shape
            ns.stride = stride
            ns.linear_offset = linear_offset
            ns.vec_load = vec_load
            return ns

        as_views = tuple(
            _make_lds_view_2d(_as_bases[s], (BLOCK_M, BLOCK_K),
                              _as_scopes[s], _as_noalias[s])
            for s in range(STAGES)
        )
        bs_views = tuple(
            _make_lds_view_2d(_bs_bases[s], (BLOCK_N, BLOCK_K),
                              _bs_scopes[s], _bs_noalias[s])
            for s in range(STAGES)
        )

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
                view = as_views[lds_stage]

                lds_offset = view.linear_offset((m_local_idx, k_local_idx)) * DTYPE_BYTES

                lds_off_i32 = arith.index_cast(T.i32, lds_offset)

                lds_off_uniform = rocdl.readfirstlane(T.i32, lds_off_i32)

                lds_ptr = _gep_lds(_as_bases[lds_stage], lds_off_uniform)

                rocdl.raw_ptr_buffer_load_lds(
                    A_.rsrc, lds_ptr,
                    arith.constant(DMA_BYTES, type=T.i32),
                    global_offset,
                    arith.constant(0, type=T.i32),
                    arith.constant(0, type=T.i32),
                    arith.constant(1, type=T.i32),

                    alias_scopes=_as_scopes[lds_stage],

                    noalias_scopes=_as_noalias[lds_stage],

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
                view = bs_views[lds_stage]

                lds_offset = view.linear_offset((n_local_idx, k_local_idx)) * DTYPE_BYTES

                lds_off_i32 = arith.index_cast(T.i32, lds_offset)

                lds_off_uniform = rocdl.readfirstlane(T.i32, lds_off_i32)

                lds_ptr = _gep_lds(_bs_bases[lds_stage], lds_off_uniform)

                rocdl.raw_ptr_buffer_load_lds(
                    B_.rsrc, lds_ptr,
                    arith.constant(DMA_BYTES, type=T.i32),
                    global_offset,
                    arith.constant(0, type=T.i32),
                    arith.constant(0, type=T.i32),
                    arith.constant(1, type=T.i32),

                    alias_scopes=_bs_scopes[lds_stage],

                    noalias_scopes=_bs_noalias[lds_stage],

                )

        # =========================================================
        # Helper: LDS → register A fragments for ONE sub-K (kk_target).
        # Returns WARP_M_STEPS frags (one per atom in M direction).
        # =========================================================
        def lds_matrix_a_kk(lds_stage, kk_target):
            view = as_views[lds_stage]  # compile-time stage dispatch
            a_frags = [0] * WARP_M_STEPS
            for ii in range_constexpr(WARP_M_STEPS):
                warp_atom_m_idx = warp_m_idx + ii * WMMA_M
                warp_atom_k_idx = kk_target * WMMA_K
                row = warp_atom_m_idx + lane_m_idx
                col_in_bytes = (warp_atom_k_idx + lane_k_vec_idx) * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                vec = view.vec_load(
                    (row, col_in_bytes // DTYPE_BYTES), WMMA_A_FRAG,
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
            view = bs_views[lds_stage]  # compile-time stage dispatch
            b_frags = [0] * WARP_N_STEPS
            for jj in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + jj * WMMA_N
                warp_atom_k_idx = kk_target * WMMA_K
                row = warp_atom_n_idx + lane_m_idx     # lane_m_idx mapping reused (N now)
                col_in_bytes = (warp_atom_k_idx + lane_k_vec_idx) * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                vec = view.vec_load(
                    (row, col_in_bytes // DTYPE_BYTES), WMMA_B_FRAG,
                )
                b_frags[jj] = vec
            return b_frags

        # =========================================================
        # Helper: MFMA for one sub-K. Wraps with s_setprio(1)/(0)
        # for the HK ping-pong (asymmetric wave population biased
        # by the warp_row==1 prologue conditional s_barrier).
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
        VMCNT_0 = 0x0F70   # vmcnt=0, expcnt=7 (no wait), lgkmcnt=15 (no wait)
        LGKMCNT_0 = 0xC07F  # vmcnt=63 (no wait), lgkmcnt=0

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

        # ----- HK-style conditional s_barrier desync -----
        # Mirrors HipKittens 256_256_64_32_with32x16.cpp:91-93. The 4
        # barriers per K-step below (B1..B4) create a cluster-by-cluster
        # rhythm; the extra barrier on warp_row==1 here pairs with
        # warp_row==0's next barrier through the WG arrival counter, so
        # the two groups run exactly one cluster out-of-phase forever
        # after — true pingpong shape. When warp_row==0 is MFMA-busy
        # (cluster 1 or 3), warp_row==1 is LDS-busy (cluster 0 or 2),
        # and vice versa. ATT-verified to work at 1 WG/CU on gfx950
        # (see ``gemm_gfx950_nt_pingpong_16x32.py`` for the prior
        # verification). DEPENDS on the 4-barriers-per-iter structure
        # below — with the previous 1-barrier-per-iter design this
        # mechanism produces a full-K-step stagger and breaks
        # correctness (NaN output from mismatched LDS stages).
        if arith.cmpi(arith.CmpIPredicate.eq, warp_row, fx.Int32(1)):
            rocdl.s_barrier()

        # ----- Main loop: 2 K-steps unrolled per outer iter -----
        # The 1-K-step-per-iter design used runtime ``current_stage`` /
        # ``next_stage`` from the loop state, so each ``lds_matrix_a_kk``
        # / ``ldg_sts_a_async`` call took a runtime ``lds_stage``. With
        # per-stage alias scopes that's a problem — Python can't index
        # into ``as_views[lds_stage]`` with a runtime value.
        #
        # Unrolling 2× makes ``lds_stage`` a Python int at every call
        # site (sub-step 0 → stage 0, sub-step 1 → stage 1). Same 4
        # barriers per K-step (= 8 barriers per outer iter) as before.
        TOTAL_K_STEPS = k // BLOCK_K
        OUTER_ITERS = TOTAL_K_STEPS // 2
        assert TOTAL_K_STEPS % 2 == 0, "32x16 kernel requires K % 128 == 0"

        def kstep_cluster(read_stage, prefetch_k, prefetch_stage):
            """One K-step body: 4 clusters (LD0 / MMA0 / LD1 / MMA1)."""
            # Cluster 0: ds_read kk=0 + HBM→LDS prefetch for a later K-step.
            a_frags_kk0 = lds_matrix_a_kk(read_stage, 0)
            b_frags_kk0 = lds_matrix_b_kk(read_stage, 0)
            ldg_sts_a_async(prefetch_k, prefetch_stage)
            ldg_sts_b_async(prefetch_k, prefetch_stage)
            rocdl.s_barrier()

            # Cluster 1: MFMA on kk=0. Drain lgkmcnt so ds_reads land first.
            rocdl.s_waitcnt(LGKMCNT_0)
            mma_kk(a_frags_kk0, b_frags_kk0, c_frags)
            rocdl.s_barrier()

            # Cluster 2: ds_read kk=1. vmcnt(0) drains cluster 0's DMA
            # writes one cluster early — by the time r1 (one cluster
            # behind r0) starts the NEXT cluster 0 LDS reads, both warp
            # groups' DMA writes have committed.
            a_frags_kk1 = lds_matrix_a_kk(read_stage, 1)
            b_frags_kk1 = lds_matrix_b_kk(read_stage, 1)
            rocdl.s_waitcnt(VMCNT_0)
            rocdl.s_barrier()

            # Cluster 3: MFMA on kk=1.
            rocdl.s_waitcnt(LGKMCNT_0)
            mma_kk(a_frags_kk1, b_frags_kk1, c_frags)
            rocdl.s_barrier()

        k_zero = arith.constant(0, type=T.i32)
        init_state = [k_zero] + c_frags
        # Loop runs OUTER_ITERS - 1 times. Last outer iter (= last 2
        # K-steps) handled by the epilogue so we can omit the trailing
        # prefetch of K-steps that don't exist.
        for _oi, state in range(0, OUTER_ITERS - 1, init=init_state):
            k_now = state[0]
            k_next_stage0 = k_now + fx.Int32(2 * BLOCK_K)
            k_next_stage1 = k_now + fx.Int32(3 * BLOCK_K)
            c_frags = list(state[1 : 1 + C_FRAGS_LEN])

            # === K-step 2k from stage 0 ===
            # Reads K-step 2k (already in stage 0); prefetches K-step
            # 2k+1 (next sub-step's data) into stage 1.
            kstep_cluster(0, k_now + fx.Int32(BLOCK_K), 1)

            # === K-step 2k+1 from stage 1 ===
            # Reads K-step 2k+1 (just prefetched); prefetches K-step
            # 2k+2 (next outer iter's first sub-step) into stage 0.
            kstep_cluster(1, k_next_stage0, 0)

            rocdl.sched_barrier(0)
            results = yield [k_next_stage0] + c_frags

        # ----- Epilogue: last 2 K-steps -----
        # State after the loop: stage 0 has K-step 2*OUTER_ITERS-2
        # (just prefetched by the loop's last sub-step), stage 1 is
        # stale. We still need to prefetch K-step 2*OUTER_ITERS-1 into
        # stage 1 before reading it.
        c_frags = list(results[1 : 1 + C_FRAGS_LEN])
        k_last = results[0]  # = (OUTER_ITERS - 1) * 2 * BLOCK_K
        # K-step 2*OUTER_ITERS-2 from stage 0; prefetch K-step 2*OUTER_ITERS-1 into stage 1.
        kstep_cluster(0, k_last + fx.Int32(BLOCK_K), 1)
        # K-step 2*OUTER_ITERS-1 from stage 1. No prefetch — but the
        # cluster helper still issues one for code reuse; we just
        # prefetch the FIRST K-step into stage 0 (harmlessly overwrites
        # data nobody reads). The downstream barriers are all there are
        # extant K-steps to do.
        # Simpler: inline the last cluster without the prefetch.
        a_frags_kk0_last = lds_matrix_a_kk(1, 0)
        b_frags_kk0_last = lds_matrix_b_kk(1, 0)
        rocdl.s_barrier()
        rocdl.s_waitcnt(LGKMCNT_0)
        mma_kk(a_frags_kk0_last, b_frags_kk0_last, c_frags)
        rocdl.s_barrier()
        a_frags_kk1_last = lds_matrix_a_kk(1, 1)
        b_frags_kk1_last = lds_matrix_b_kk(1, 1)
        rocdl.s_waitcnt(VMCNT_0)
        rocdl.s_barrier()
        rocdl.s_waitcnt(LGKMCNT_0)
        mma_kk(a_frags_kk1_last, b_frags_kk1_last, c_frags)
        rocdl.s_barrier()

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
            linkage = ir.Attribute.parse('#llvm.linkage<external>')
            for sym, size in (
                (LDS_SYMS_A[0], AS_STAGE_BYTES),
                (LDS_SYMS_A[1], AS_STAGE_BYTES),
                (LDS_SYMS_B[0], BS_STAGE_BYTES),
                (LDS_SYMS_B[1], BS_STAGE_BYTES),
            ):
                llvm.GlobalOp(
                    global_type=ir.Type.parse(f"!llvm.array<{size} x i8>"),
                    sym_name=sym,
                    linkage=linkage,
                    addr_space=3,
                    alignment=1024,
                )
        bm = (m + BLOCK_M - 1) // BLOCK_M
        bn = n // BLOCK_N
        total_tiles = bm * bn
        nt_kernel._func.__name__ = KERNEL_NAME
        launcher = nt_kernel(C, A, B, m)
        # With 8 waves/WG and waves_per_eu=2 we have 256 VGPRs/wave —

        # the compiler can fit everything in VGPRs (~210-240). Forcing

        # ``amdgpu-agpr-alloc=128,128`` (older NN-big pattern) splits the

        # unified pool into 128 VGPR + 128 AGPR and adds VGPR↔AGPR copies

        # per MFMA, costing ~6% perf. HK kernels run with VGPRs=210,

        # AGPRs=0 — same profile we get when we let the compiler decide.
        passthrough_attr = ir.ArrayAttr.get([])
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, 2)
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
