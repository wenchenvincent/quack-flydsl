# Copyright (c) 2026, AMD.

"""NT-layout bf16/f16 GEMM, 4-warp 256×192 tile (gfx950).

Adapted from HK's ``FP8_4wave/4_wave.cu`` to bf16 with the largest tile
that still gives us a chance at **2 WGs/CU residency**. The bet: with
4 warps per WG (vs our 8-warp default), per-WG resources halve, and
even though per-warp work doubles, the lower wave count per SIMD
should let two workgroups share a CU. Two WGs/CU is what HK relies on
to make their barrier-heavy ping-pong patterns fast — while one WG is
barriered, the other keeps the SIMDs busy.

Layout:
    - 4 warps in a 2×2 grid (warp_row, warp_col), 256 threads/WG
    - BLOCK_M=256, BLOCK_N=192, BLOCK_K=64
    - Per-warp tile: 128 × 96 = 4 quadrants of 64 × 48
    - Per-quadrant atoms: 4 (M) × 3 (N) = 12 atoms of 16×16
    - Per-warp C accumulator: 4 × 12 × 4 fp32/lane = 192 fp32/lane
      (128 in AGPRs + 64 spilling to VGPR)

LDS at 2 stages, BLOCK_K=64:
    - A: 256×64×2 = 32KB per stage, 2 stages = 64KB
    - B: 192×64×2 = 24KB per stage, 2 stages = 48KB
    - Total: 112KB (well under 160KB cap)

Tile asymmetry (256 M vs 192 N) means -25% per-WG compute density vs
256×256 8-wave, offset by potentially 2× WG/CU residency. Net is
empirical — if residency improves AND the compiler keeps the schedule
clean, this should beat the 8-wave kernel. If neither plays out, this
file is just a 25%-slower variant.

Public API:

    gemm_nt_4wave(a, b, out=None)

Shape constraints: M % 256, N % 192, K % 64.
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
def _compile_nt_4wave_kernel(
    dtype: str, k: int, n: int, _m_hint: int = 0,
    XCD_SWIZZLE: int = 1,
    GROUP_M: int = 1,
):
    """Compile the 4-wave NT GEMM kernel for ``(dtype, K, N)``.

    4 warps per WG, 256×192 tile, 4 quadrants per warp.
    """
    # ----- Tile + warp constants -----
    BLOCK_M = 256
    BLOCK_N = 192
    BLOCK_K = 64
    BLOCK_M_WARPS = 2
    BLOCK_N_WARPS = 2
    WARP_SIZE = 64
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE   # = 256
    WARP_M = BLOCK_M // BLOCK_M_WARPS                            # = 128
    WARP_N = BLOCK_N // BLOCK_N_WARPS                            # =  96

    # ----- MFMA atom (16x16x32 bf16/f16 on gfx950) -----
    WMMA_IMPL = _WmmaHalfK32(dtype)
    WMMA_M = WMMA_IMPL.WMMA_M                                    # = 16
    WMMA_N = WMMA_IMPL.WMMA_N                                    # = 16
    WMMA_K = WMMA_IMPL.WMMA_K                                    # = 32
    WMMA_A_FRAG = WMMA_IMPL.WMMA_A_FRAG_VALUES                   # =  8
    WMMA_B_FRAG = WMMA_IMPL.WMMA_B_FRAG_VALUES                   # =  8
    WMMA_C_FRAG = WMMA_IMPL.WMMA_C_FRAG_VALUES                   # =  4

    WARP_M_STEPS = WARP_M // WMMA_M                              # =  8
    WARP_N_STEPS = WARP_N // WMMA_N                              # =  6
    # DOT_SLICE = WMMA_K = 32 → one K-step has BLOCK_K/WMMA_K = 2 sub-K iters
    K_SUBITERS = BLOCK_K // WMMA_K                               # =  2
    assert K_SUBITERS == 2, "kernel structured around 2 sub-K iters per K-step"

    # ----- 4-quadrant decomposition (HK interleaved warp layout) -----
    # BLOCK_M=256 split into 2 M-halves of HALF_BLOCK_M=128 rows each.
    # BLOCK_N=192 split into 2 N-halves of HALF_BLOCK_N=96 cols each.
    # Each half further split into 2 warp-stripes (one per warp_row/col
    # in the 2×2 grid).
    #
    # Each warp's 128×96 footprint is split into 4 quadrants of 64×48.
    # Within a quadrant: M_ATOMS_PER_QUAD × N_ATOMS_PER_QUAD = 4 × 3 atoms.
    #
    # HK's interleaved ownership: warp_row=0 takes BLOCK_M rows
    # {0..63, 128..191}, warp_row=1 takes {64..127, 192..255}. Similarly
    # for N: warp_col=0 takes cols {0..47, 96..143}, warp_col=1 takes
    # {48..95, 144..191}.
    #
    # Flat c_frags layout: c_frags[q * ATOMS_PER_QUAD + ai * N_ATOMS_PER_QUAD + aj]
    #   q = m_h * 2 + n_h, ai ∈ 0..3, aj ∈ 0..2.
    HALF_BLOCK_M = BLOCK_M // 2                                  # = 128
    HALF_BLOCK_N = BLOCK_N // 2                                  # =  96
    HALF_WARP_M = WARP_M // 2                                    # =  64
    HALF_WARP_N = WARP_N // 2                                    # =  48
    M_ATOMS_PER_QUAD = WARP_M_STEPS // 2                         # =   4
    N_ATOMS_PER_QUAD = WARP_N_STEPS // 2                         # =   3
    ATOMS_PER_QUAD = M_ATOMS_PER_QUAD * N_ATOMS_PER_QUAD         # =  12

    DTYPE_BYTES = 2
    STAGES = 2

    # ----- HBM→LDS DMA constants -----
    LDG_VEC_SIZE = 8                                              # 8 bf16 / 16 bytes per thread
    DMA_BYTES = LDG_VEC_SIZE * DTYPE_BYTES                        # = 16
    LDG_X_THREADS = BLOCK_K // LDG_VEC_SIZE                       # =  8
    # 256 threads × 8 elems = 2048 elems per call.
    # A tile = 256 × 64 = 16384 elems → 8 calls.
    # B tile = 192 × 64 = 12288 elems → 6 calls.
    LDG_A_REG_COUNT = (BLOCK_M * BLOCK_K) // (LDG_VEC_SIZE * BLOCK_THREADS)
    LDG_B_REG_COUNT = (BLOCK_N * BLOCK_K) // (LDG_VEC_SIZE * BLOCK_THREADS)
    assert LDG_A_REG_COUNT == 8 and LDG_B_REG_COUNT == 6

    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES

    # ----- LDS allocation — per-stage globals + alias scopes -----
    # See gemm_gfx950_nt_pingpong_16x32.py + the implementation guide in
    # docs/superpowers/specs/2026-06-04-flydsl-lds-provenance-todo.md
    # for the design. Four LDS globals (AS@0, AS@1, BS@0, BS@1), each
    # tagged with its own alias scope so SIInsertWaitcnts can drop
    # conservative vmcnt(0) drains in favour of partial vmcnt(N).
    AS_STAGE_BYTES = BLOCK_M * BLOCK_K * DTYPE_BYTES
    BS_STAGE_BYTES = BLOCK_N * BLOCK_K * DTYPE_BYTES
    LDS_SYMS_A = (
        f"nt_4wave_smem_as0_{dtype}_{k}_{n}",
        f"nt_4wave_smem_as1_{dtype}_{k}_{n}",
    )
    LDS_SYMS_B = (
        f"nt_4wave_smem_bs0_{dtype}_{k}_{n}",
        f"nt_4wave_smem_bs1_{dtype}_{k}_{n}",
    )
    LDS_ALIAS_DOMAIN = f'#llvm.alias_scope_domain<id = "nt_4wave_{dtype}_{k}_{n}.lds">'
    SCOPE_IDS = ("as0", "as1", "bs0", "bs1")

    BLOCK_K_LOOPS_HINT = max(1, k // BLOCK_K)

    KERNEL_NAME = f"nt_4wave_{dtype}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}"

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def nt_kernel_4wave(C: fx.Tensor, A: fx.Tensor, B: fx.Tensor, m: fx.Int32):
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
        LDG_A_HALF_REG_COUNT = LDG_A_REG_COUNT // 2               # = 2
        LDG_B_HALF_REG_COUNT = LDG_B_REG_COUNT // 2               # = 2

        def ldg_sts_a_half_async(k_offset, lds_stage, m_half):
            for i in range_constexpr(LDG_A_HALF_REG_COUNT):
                eff_i = m_half * LDG_A_HALF_REG_COUNT + i
                global_tid = BLOCK_THREADS * eff_i + tid
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

        def ldg_sts_b_half_async(k_offset, lds_stage, n_half):
            for i in range_constexpr(LDG_B_HALF_REG_COUNT):
                eff_i = n_half * LDG_B_HALF_REG_COUNT + i
                global_tid = BLOCK_THREADS * eff_i + tid
                n_local_idx = global_tid // LDG_X_THREADS
                k_local_idx = global_tid % LDG_X_THREADS * LDG_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
                row_idx = n_offset + fx.Index(n_local_idx)
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

        # Full-tile loaders (used only by the prologue).
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
        # Helper: load M-half of A from LDS into registers.
        # Returns M_ATOMS_PER_QUAD × K_SUBITERS = 8 frags
        # indexed [ai * K_SUBITERS + kk] for ai ∈ 0..3, kk ∈ 0..1.
        # =========================================================
        def lds_matrix_a_half(lds_stage, m_half):
            n_frags = M_ATOMS_PER_QUAD * K_SUBITERS
            a_frags = [0] * n_frags
            for ai in range_constexpr(M_ATOMS_PER_QUAD):
                warp_atom_m_idx = (m_half * HALF_BLOCK_M
                                   + warp_row * HALF_WARP_M
                                   + ai * WMMA_M)
                row = warp_atom_m_idx + lane_m_idx
                for kk in range_constexpr(K_SUBITERS):
                    warp_atom_k_idx = kk * WMMA_K
                    col_in_bytes = (warp_atom_k_idx + lane_k_vec_idx) * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    view = as_views[lds_stage]

                    vec = view.vec_load(

                        (row, col_in_bytes // DTYPE_BYTES), WMMA_A_FRAG,

                    )
                    a_frags[ai * K_SUBITERS + kk] = vec
            return a_frags

        # =========================================================
        # Helper: load N-half of B from LDS into registers.
        # Returns N_ATOMS_PER_QUAD × K_SUBITERS = 4 frags
        # indexed [aj * K_SUBITERS + kk] for aj ∈ 0..1, kk ∈ 0..1.
        # =========================================================
        def lds_matrix_b_half(lds_stage, n_half):
            warp_col = wid % BLOCK_N_WARPS
            n_frags = N_ATOMS_PER_QUAD * K_SUBITERS
            b_frags = [0] * n_frags
            for aj in range_constexpr(N_ATOMS_PER_QUAD):
                warp_atom_n_idx = (n_half * HALF_BLOCK_N
                                   + warp_col * HALF_WARP_N
                                   + aj * WMMA_N)
                row = warp_atom_n_idx + lane_m_idx
                for kk in range_constexpr(K_SUBITERS):
                    warp_atom_k_idx = kk * WMMA_K
                    col_in_bytes = (warp_atom_k_idx + lane_k_vec_idx) * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    view = bs_views[lds_stage]

                    vec = view.vec_load(

                        (row, col_in_bytes // DTYPE_BYTES), WMMA_B_FRAG,

                    )
                    b_frags[aj * K_SUBITERS + kk] = vec
            return b_frags

        # =========================================================
        # Helper: MFMA for one quadrant (24 mfmas total).
        # =========================================================
        def mma_quadrant(a_h_frags, b_h_frags, c_frags, q):
            rocdl.s_setprio(1)
            base = q * ATOMS_PER_QUAD
            for ai in range_constexpr(M_ATOMS_PER_QUAD):
                for aj in range_constexpr(N_ATOMS_PER_QUAD):
                    c_idx = base + ai * N_ATOMS_PER_QUAD + aj
                    acc = c_frags[c_idx]
                    for kk in range_constexpr(K_SUBITERS):
                        a_frag = a_h_frags[ai * K_SUBITERS + kk]
                        b_frag = b_h_frags[aj * K_SUBITERS + kk]
                        acc = WMMA_IMPL(a_frag, b_frag, acc)
                    c_frags[c_idx] = acc
            rocdl.s_setprio(0)

        # =========================================================
        # Helper: MFMA for one quadrant interleaved with `side_tasks`.
        # Each side_task is a 0-arg callable invoked between mfma groups,
        # fenced by sched_barrier(0). 24 mfmas total per quadrant; side
        # tasks distributed evenly. This mirrors HK's do_interleaved_cluster
        # pattern: by weaving the NEXT cluster's LDS reads and HBM prefetches
        # into THIS cluster's mfma stream, the ds_read latency is hidden
        # behind in-flight mfmas.
        # =========================================================
        TOTAL_MFMAS = M_ATOMS_PER_QUAD * N_ATOMS_PER_QUAD * K_SUBITERS  # = 24

        def mma_quadrant_interleaved(a_h_frags, b_h_frags, c_frags, q, side_tasks):
            rocdl.s_setprio(1)
            base = q * ATOMS_PER_QUAD
            n_tasks = len(side_tasks)
            spacing = max(1, TOTAL_MFMAS // (n_tasks + 1)) if n_tasks else TOTAL_MFMAS + 1
            triggers = []
            for _i in range_constexpr(n_tasks):
                triggers.append((_i + 1) * spacing)
            mfma_count = 0
            task_idx = 0
            for ai in range_constexpr(M_ATOMS_PER_QUAD):
                for aj in range_constexpr(N_ATOMS_PER_QUAD):
                    c_idx = base + ai * N_ATOMS_PER_QUAD + aj
                    acc = c_frags[c_idx]
                    for kk in range_constexpr(K_SUBITERS):
                        a_frag = a_h_frags[ai * K_SUBITERS + kk]
                        b_frag = b_h_frags[aj * K_SUBITERS + kk]
                        acc = WMMA_IMPL(a_frag, b_frag, acc)
                        mfma_count += 1
                        if task_idx < n_tasks and mfma_count == triggers[task_idx]:
                            rocdl.sched_barrier(0)
                            side_tasks[task_idx]()
                            rocdl.sched_barrier(0)
                            task_idx += 1
                    c_frags[c_idx] = acc
            # Fire any remaining side tasks (those whose trigger wasn't
            # reached because we ran out of mfmas).
            for _r in range_constexpr(n_tasks):
                if task_idx < n_tasks:
                    rocdl.sched_barrier(0)
                    side_tasks[task_idx]()
                    rocdl.sched_barrier(0)
                    task_idx += 1
            rocdl.s_setprio(0)

        # =========================================================
        # Per-atom LDS-read helpers (single ds_read) — used as side_tasks
        # in the interleaved cluster.
        # =========================================================
        def lds_load_a_atom(out_list, atom_idx, lds_stage, m_half, ai, kk):
            warp_atom_m_idx = (m_half * HALF_BLOCK_M
                               + warp_row * HALF_WARP_M
                               + ai * WMMA_M)
            row = warp_atom_m_idx + lane_m_idx
            warp_atom_k_idx = kk * WMMA_K
            col_in_bytes = (warp_atom_k_idx + lane_k_vec_idx) * DTYPE_BYTES
            col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
            out_list[atom_idx] = as_views[lds_stage].vec_load(

                (row, col_in_bytes // DTYPE_BYTES), WMMA_A_FRAG,

            )

        def lds_load_b_atom(out_list, atom_idx, lds_stage, n_half, aj, kk):
            warp_col = wid % BLOCK_N_WARPS
            warp_atom_n_idx = (n_half * HALF_BLOCK_N
                               + warp_col * HALF_WARP_N
                               + aj * WMMA_N)
            row = warp_atom_n_idx + lane_m_idx
            warp_atom_k_idx = kk * WMMA_K
            col_in_bytes = (warp_atom_k_idx + lane_k_vec_idx) * DTYPE_BYTES
            col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
            out_list[atom_idx] = bs_views[lds_stage].vec_load(

                (row, col_in_bytes // DTYPE_BYTES), WMMA_B_FRAG,

            )

        # =========================================================
        # Per-call HBM→LDS prefetch helpers (one buffer_load_lds call per
        # invocation, so we can interleave them with mfmas).
        # =========================================================
        def ldg_sts_a_call(k_offset, lds_stage, i):
            # Single iteration of ldg_sts_a_async.
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

        def ldg_sts_b_call(k_offset, lds_stage, i):
            global_tid = BLOCK_THREADS * i + tid
            n_local_idx = global_tid // LDG_X_THREADS
            k_local_idx = global_tid % LDG_X_THREADS * LDG_VEC_SIZE
            col_in_bytes = k_local_idx * DTYPE_BYTES
            col_in_bytes = swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
            row_idx = n_offset + fx.Index(n_local_idx)
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

        # ----- Accumulator init (4 quadrants × 8 atoms = 32 frags) -----
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS                # = 32
        c_frags = [acc_init] * C_FRAGS_LEN

        # ----- Initial s_sleep desync (vestigial, gets absorbed by barrier) -----
        if arith.cmpi(arith.CmpIPredicate.eq, warp_row, fx.Int32(1)):
            rocdl.s_sleep(16)

        # ----- Prologue: HBM→LDS iter 0 → stage 0 -----
        k_begin = arith.constant(0, type=T.i32)
        ldg_sts_a_async(k_begin, 0)
        ldg_sts_b_async(k_begin, 0)
        gpu.barrier()

        # ----- Main loop: 2 K-steps unrolled per outer iter -----
        # Same rebalanced 4-cluster body as before, just unrolled 2× so
        # ``lds_stage`` is a Python int at every helper call site (required
        # by the per-stage alias scopes — see the implementation guide).
        BLOCK_K_LOOPS = k // BLOCK_K
        OUTER_ITERS = BLOCK_K_LOOPS // 2
        assert BLOCK_K_LOOPS % 2 == 0, "nt_4wave requires K % 128 == 0"

        def kstep_body(k_offset, current_stage, next_stage, c_frags):
            """One K-step (4 clusters). current_stage / next_stage are Python ints."""
            k_next = k_offset + fx.Int32(BLOCK_K)
            a_h0 = lds_matrix_a_half(current_stage, 0)
            b_h0 = lds_matrix_b_half(current_stage, 0)
            a_h1 = lds_matrix_a_half(current_stage, 1)

            b_h1_frags = [None] * (N_ATOMS_PER_QUAD * K_SUBITERS)
            c0_side_tasks = []
            for aj in range_constexpr(N_ATOMS_PER_QUAD):
                for kk in range_constexpr(K_SUBITERS):
                    aidx = aj * K_SUBITERS + kk
                    c0_side_tasks.append(
                        (lambda lst=b_h1_frags, i=aidx, aj_=aj, kk_=kk:
                         lds_load_b_atom(lst, i, current_stage, 1, aj_, kk_))
                    )
            for i in range_constexpr(LDG_A_REG_COUNT):
                c0_side_tasks.append(
                    (lambda i_=i: ldg_sts_a_call(k_next, next_stage, i_))
                )
            mma_quadrant_interleaved(a_h0, b_h0, c_frags, 0, c0_side_tasks)

            mma_quadrant(a_h0, b_h1_frags, c_frags, 1)

            c2_side_tasks = []
            for i in range_constexpr(LDG_B_REG_COUNT):
                c2_side_tasks.append(
                    (lambda i_=i: ldg_sts_b_call(k_next, next_stage, i_))
                )
            mma_quadrant_interleaved(a_h1, b_h0, c_frags, 2, c2_side_tasks)

            mma_quadrant(a_h1, b_h1_frags, c_frags, 3)

            gpu.barrier()

        init_state = [k_begin] + c_frags
        # Loop runs OUTER_ITERS - 1 times; last outer iter (2 K-steps) in epilogue.
        for _oi, state in range(0, OUTER_ITERS - 1, init=init_state):
            k_now = state[0]
            c_frags = list(state[1 : 1 + C_FRAGS_LEN])

            # Sub-step 0: K_2i from stage 0; prefetch into stage 1 for next sub-step.
            kstep_body(k_now, 0, 1, c_frags)

            # Sub-step 1: K_2i+1 from stage 1; prefetch into stage 0 for next outer iter.
            kstep_body(k_now + fx.Int32(BLOCK_K), 1, 0, c_frags)

            k_next_outer = k_now + fx.Int32(2 * BLOCK_K)
            rocdl.sched_barrier(0)
            results = yield [k_next_outer] + c_frags

        # ----- Epilogue: last 2 K-steps -----
        # State after loop: stage 0 has K_(2*OUTER_ITERS-2) (just prefetched),
        # stage 1 is stale. We process the K-step in stage 0 while also
        # prefetching K_(2*OUTER_ITERS-1) into stage 1, then process that.
        c_frags = list(results[1 : 1 + C_FRAGS_LEN])
        k_last = results[0]   # = (OUTER_ITERS-1) * 2 * BLOCK_K
        k_last_kstep = k_last + fx.Int32(BLOCK_K)  # K offset for the very last K-step

        # K_(2*OUTER_ITERS-2) from stage 0; prefetch K_(2*OUTER_ITERS-1) into stage 1.
        kstep_body(k_last, 0, 1, c_frags)

        # K_(2*OUTER_ITERS-1) from stage 1. No prefetch needed — bare 4-cluster body.
        a_h0_e = lds_matrix_a_half(1, 0)
        b_h0_e = lds_matrix_b_half(1, 0)
        mma_quadrant(a_h0_e, b_h0_e, c_frags, 0)
        b_h1_e = lds_matrix_b_half(1, 1)
        mma_quadrant(a_h0_e, b_h1_e, c_frags, 1)
        a_h1_e = lds_matrix_a_half(1, 1)
        mma_quadrant(a_h1_e, b_h0_e, c_frags, 2)
        mma_quadrant(a_h1_e, b_h1_e, c_frags, 3)

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
        # Writeback in HK interleaved coordinates. For each quadrant
        # q = m_h*2 + n_h, atom (ai, aj):
        #   warp_atom_m = m_h * HALF_BLOCK_M + warp_row * HALF_WARP_M + ai * WMMA_M
        #   warp_atom_n = n_h * HALF_BLOCK_N + warp_col * HALF_WARP_N + aj * WMMA_N
        warp_col = wid % BLOCK_N_WARPS
        for q in range_constexpr(4):
            m_h = q // 2
            n_h = q % 2
            base = q * ATOMS_PER_QUAD
            for ai in range_constexpr(M_ATOMS_PER_QUAD):
                for aj in range_constexpr(N_ATOMS_PER_QUAD):
                    c_idx = base + ai * N_ATOMS_PER_QUAD + aj
                    c_vec = c_frags[c_idx]
                    lane_col = w_tid % WMMA_N                   # 0..15
                    lane_row_base = (w_tid // WMMA_N) * 4       # 0, 4, 8, 12
                    warp_atom_m_idx = (m_h * HALF_BLOCK_M
                                       + warp_row * HALF_WARP_M
                                       + ai * WMMA_M)
                    warp_atom_n_idx = (n_h * HALF_BLOCK_N
                                       + warp_col * HALF_WARP_N
                                       + aj * WMMA_N)
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
        nt_kernel_4wave._func.__name__ = KERNEL_NAME
        launcher = nt_kernel_4wave(C, A, B, m)
        # Force the C accumulator (192 fp32/lane) into AGPRs to free up
        # VGPRs for operands. Without this hint, LLVM put accumulators
        # in VGPRs and exhausted the 256-VGPR budget needed for 2 waves/SIMD.
        passthrough_attr = ir.ArrayAttr.get([
            ir.ArrayAttr.get([
                ir.StringAttr.get("amdgpu-agpr-alloc"),
                ir.StringAttr.get("192,192"),
            ]),
        ])
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, 2)
                op.attributes["passthrough"] = passthrough_attr
        launcher.launch(grid=(total_tiles, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_nt_kernel


# ----- Public API -----

def gemm_nt_4wave(
    a: Tensor, b: Tensor, out: Optional[Tensor] = None,
) -> Tensor:
    """4-wave NT GEMM kernel on gfx950 (256×192 tile, 4 warps/WG).

    Computes ``C = A @ B^T`` for ``A=(M,K)`` row-major and ``B=(N,K)``
    row-major. Output ``(M, N)`` row-major in the input dtype.

    Constraints: M % 256 == 0, N % 192 == 0, K % 64 == 0.
    """
    assert a.is_cuda and b.is_cuda
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2, f"inner dims must match: A={a.shape} B={b.shape}"
    assert a.dtype == b.dtype and a.dtype in _DTYPE2STR
    assert M % 256 == 0 and N % 192 == 0 and K % 64 == 0, (
        f"gemm_nt_4wave requires M%256, N%192, K%64; got {M}×{K}×{N}"
    )
    assert a.stride(-1) == 1 and b.stride(-1) == 1
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=a.dtype)
    else:
        assert out.shape == (M, N) and out.dtype == a.dtype
        assert out.stride(-1) == 1
    dtype_str = _DTYPE2STR[a.dtype]
    _compile_nt_4wave_kernel(dtype_str, K, N, _m_hint=M)(out, a, b, M)
    return out


__all__ = ["gemm_nt_4wave"]
