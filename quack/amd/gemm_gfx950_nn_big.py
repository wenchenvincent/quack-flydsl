# Copyright (c) 2026, AMD.

"""NN-layout GEMM — 256×256 tile with 8 MFMA warps (OGS-style).

Companion to ``gemm_gfx950_nn.py``: same kernel pattern, but with a
2× larger output tile (256×256 vs 128×128) and 2× more warps per WG
(8 vs 4) to match the OGS ``triton_kernels/matmul_ogs`` default for
large-M, large-N shapes on CDNA4. Targets the shape regime where the
4-warp 128×128 path loses to hipBLASLt / OGS (e.g. M=8192, K=16384,
N=4096 where OGS is 1.27× faster — see commits ce30767, 884996d for
the profile data that motivated this variant).

Why a separate file: making 256×256 fit the 160 KB LDS cap required
halving the C-staging allocation and splitting the write-back into
two passes, which the 128×128 path doesn't benefit from. Keeping the
two kernels in separate files keeps each hot-loop body focused.

LDS budget (128 KB < 160 KB cap):
    A: STAGES=2 × 256 × 64 × 2B = 64 KB
    B: STAGES=2 × 64 × 264    × 2B = 66 KB (264 = 256 + 8 pad)
    C staging: 256 × 128 × 2B     = 64 KB   (half of full BM×BN)
    Total = max(A_single, C) + B = 64 + 66 = 130 KB ✓

2-pass write-back: stmatrix_c emits only the N-left-half frags to LDS,
barrier, vec_store to HBM at cols [0, BN/2); then the same for
[BN/2, BN). With BNW=4 warps across N, each warp's 64-col range falls
cleanly inside exactly one half.

Workgroup size: 8 warps × 64 lanes = 512 threads/WG. Declared on the
``@flyc.kernel`` via ``known_block_size=[512, 1, 1]`` to raise the
default 256-thread cap.

Public API:

    gemm_nn_big(a, b, out=None)
        a: (M, K) f16/bf16, K stride-1 (inner)
        b: (K, N) f16/bf16, N stride-1 (inner)
        out: optional preallocated (M, N) in same dtype

    Also dispatched from ``gemm_gfx950_nn.gemm_nn`` when ``_pick_config``
    returns a (256, 256, *) tile for a shape.
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
def _compile_nn_big_kernel(
    dtype: str,
    k: int,
    n: int,
    TILE_M: int = 256,
    TILE_N: int = 256,
    TILE_K: int = 64,
    BLOCK_M_WARPS: int = 2,
    BLOCK_N_WARPS: int = 4,
    # When True (default), replaces the LDS-staged 2-pass writeback
    # with a direct reg→HBM writeback using ds_bpermute to transpose
    # the MFMA C fragment from "4 rows × 1 col per lane" to "4 cols ×
    # 1 row per lane", then a single 4-element vec_store per atom.
    # Mirrors the OGS matmul_ogs writeback pattern. Closes the 19%
    # non-MFMA gap measured in rocprofv3 at 8192×4096×16384 bf16:
    # +10-18% across all bench shapes vs LDS-staged.
    DIRECT_WRITE: bool = True,
    # Pipeline depth (compute pipeline stages). 2 = default flow:
    # HBM→LDS for next iter overlaps with MFMA, but LDS→regs is
    # serialised after the iter's barrier.
    #
    # 3 = adds a third compute stage by ds_read'ing one extra iter
    # ahead concurrently with MFMA; the prologue prefetches 2 iters
    # into LDS so iter-k's ds_read targets a stage written 1 iter ago,
    # not this iter — no internal barrier needed. STAGES (LDS double-
    # buffering) stays at 2; the 3rd stage is register-prefetch.
    #
    # Status: implemented, opt-in only. Mirrors gluon-tutorials a16w16
    # v5 (local_prefetch). On our 256×256 tile it currently regresses:
    # at TILE_K=64 the doubled live-fragment set spills (the AMDGPU
    # compiler doesn't move prefetch frags into AGPRs, so we hit the
    # 256 VGPR cap and the hot loop scratch-loads 57 spills/iter); at
    # TILE_K=32 there's no spill but the loop overhead from doubling
    # the K-iter count costs more than the ds_read overlap saves.
    # The gluon-tutorials path forward (v7-v8) requires N-slicing or
    # MN-slicing the operand loads + a custom assembly post-processor
    # to make the RA actually use all 512 unified VGPR+AGPR — out of
    # scope without that tooling.
    PIPELINE_DEPTH: int = 2,
    # When True, splits the B-fragment LDS read into two N-halves and
    # runs MFMA in two passes (cols 0..BN/2-1, then cols BN/2..BN-1).
    # The first half's b_frags dies before the second half is loaded,
    # so the register allocator only needs to keep half-B live at any
    # given time. Mirrors gluon-tutorials a16w16 v7 (sliceN).
    #
    # Status: opt-in (default False). Saves only ~1 VGPR at the
    # peak measured on our 8-warp 256×256 config and shows mixed
    # bench results (-3% to +2%) — register pressure on this kernel
    # is dominated by A-side fragments (16 frags vs 8 for B), which
    # SLICE_N alone doesn't reduce. Combining with SLICE_M (the
    # tutorial v8 path) plus an assembly post-processor (their
    # ``amdgcnas``) is what brought their kernel to 98% MFMA
    # efficiency — out of scope without that tooling.
    #
    # Implementation: only wired into the PIPELINE_DEPTH=2 path
    # currently. PIPELINE_DEPTH=3 ignores this flag.
    SLICE_N: bool = False,
    # Grid swizzle:
    #   XCD_SWIZZLE — number of XCDs (chiplets) to distribute work across.
    #     MI355X has 8 XCDs. ``1`` disables.
    #   XCD_CHUNK_SIZE — chunk granularity for XCD assignment, per
    #     HipKittens Algorithm 1 (arxiv:2511.08083). Each XCD receives
    #     consecutive chunks of this many tiles before cycling to the
    #     next XCD. ``0`` falls back to the OGS-style scheme (= large
    #     implicit chunk = total_tiles / XCD_SWIZZLE), giving each XCD
    #     one big contiguous tile range. Small values (HK paper uses
    #     4) keep concurrently-executing XCDs working on nearby tiles
    #     so shared LLC (per-chiplet-pair on MI355X) hits more.
    #   GROUP_M — tile grouping along M for L2 B-matrix reuse across
    #     GROUP_M consecutive M-tiles sharing one N-tile load. Equivalent
    #     to HipKittens' ``window_h``. ``1`` disables.
    XCD_SWIZZLE: int = 4,
    XCD_CHUNK_SIZE: int = 0,
    GROUP_M: int = 1,
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

    # Per-pass write-back fanout (LDS → HBM).
    LDG_C_X_THREADS_H = (BLOCK_N // 2) // LDG_VEC_SIZE
    LDG_REG_C_HALF = (BLOCK_M * (BLOCK_N // 2)) // BLOCK_VECS
    assert LDG_REG_C_HALF >= 1
    assert (BLOCK_M * (BLOCK_N // 2)) % BLOCK_VECS == 0

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

    # C staging halved — we write the N-left-half in pass 1, then
    # N-right-half in pass 2, reusing the same LDS region. This keeps
    # the 256×256 tile under the 160 KB LDS cap (see module docstring).
    assert BLOCK_N % 2 == 0, "halved C-staging requires BLOCK_N even"
    C_LDS_N = BLOCK_N // 2
    C_LDS_BYTES = BLOCK_M * C_LDS_N * DTYPE_BYTES

    allocator = SmemAllocator(
        None, arch=GPU_ARCH,
        global_sym_name=f"nn_big_smem_{dtype}_{k}_{n}_{BLOCK_M}_{BLOCK_N}",
    )
    smem_a_offset = allocator._align(allocator.ptr, 16)
    AS_BYTES = STAGES * BLOCK_M * BLOCK_K * DTYPE_BYTES
    # C staging aliases the A region. For the big kernel we only need
    # ``BLOCK_M × BLOCK_N/2`` bytes per pass, which for 256×256 is 64 KB
    # — within single-stage A's 32 KB? No: max(32K, 64K) = 64K. Still
    # wins vs the full 128 KB that would blow the LDS cap.
    AS_BYTES = max(AS_BYTES, C_LDS_BYTES)
    allocator.ptr = smem_a_offset + AS_BYTES
    smem_b_offset = allocator._align(allocator.ptr, 16)
    BS_BYTES = STAGES * BLOCK_K * BS_N_STRIDE * DTYPE_BYTES
    allocator.ptr = smem_b_offset + BS_BYTES

    KERNEL_NAME = f"nn_big_{dtype}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_S{STAGES}"
    KERNEL_NAME += "_AS" if ASYNC_COPY else "_NA"

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
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
        # C writeback-time LDS (aliases A's region). Shape is BM × (BN/2)
        # since we run writeback in two passes over the N dim to halve
        # LDS use and fit the 160 KB cap.
        smem_c_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(BLOCK_M * C_LDS_N,))
        cs_ = STensor(smem_c_ptr, dtype_, shape=(BLOCK_M, C_LDS_N))

        tid = fx.Int32(fx.thread_idx.x)
        wid = tid // WARP_SIZE
        w_tid = tid % WARP_SIZE

        # 1D grid launch: block_idx.x is a flat tile id in [0, bm*bn).
        # Decode (block_m_idx, block_n_idx) via two-stage swizzle:
        #   (1) xcd_swizzle — remap flat pid so adjacent hardware-pid
        #       tiles end up on the right XCDs for cache locality.
        #       Two modes:
        #         XCD_CHUNK_SIZE == 0: OGS-style — each XCD gets one big
        #           contiguous chunk (=total_tiles/XCD_SWIZZLE). Max
        #           per-XCD L2 locality, minimal LLC cross-XCD sharing.
        #         XCD_CHUNK_SIZE > 0: HipKittens Algorithm 1 — each XCD
        #           gets multiple small chunks of CHUNK_SIZE tiles,
        #           interleaved across cycles. Adjacent XCDs process
        #           nearby tiles concurrently → LLC reuse for shared
        #           operand rows.
        #   (2) swizzle2d (GROUP_M) — remap flat pid to (pid_m, pid_n)
        #       grouping GROUP_M consecutive M-tiles under each pid_n
        #       block (= HK ``window_h``).
        flat_pid = fx.Int32(fx.block_idx.x)
        bn_c = fx.Int32(n // BLOCK_N)               # compile-time
        bm_rt = (m + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M)
        if XCD_SWIZZLE > 1:
            xcd_c = fx.Int32(XCD_SWIZZLE)
            total_tiles = bm_rt * bn_c
            if XCD_CHUNK_SIZE > 0:
                # HipKittens Algorithm 1 — chunk-based remap.
                #   target_xcd = pid % XCD
                #   local_index = pid // XCD
                #   chunk_idx = local_index // CHUNK
                #   position = local_index % CHUNK
                #   remapped = chunk_idx*(XCD*CHUNK) + target_xcd*CHUNK + position
                # Tail (linear_xy >= aligned_limit) passes through
                # unchanged so we don't lose tiles.
                chunk_c = fx.Int32(XCD_CHUNK_SIZE)
                blocks_per_cycle = fx.Int32(XCD_SWIZZLE * XCD_CHUNK_SIZE)
                aligned_limit = (total_tiles // blocks_per_cycle) * blocks_per_cycle
                is_aligned = arith.cmpi(
                    arith.CmpIPredicate.slt, flat_pid, aligned_limit,
                )
                target_xcd = flat_pid % xcd_c
                local_index = flat_pid // xcd_c
                chunk_idx = local_index // chunk_c
                position = local_index % chunk_c
                pid_aligned = (
                    chunk_idx * blocks_per_cycle
                    + target_xcd * chunk_c
                    + position
                )
                pid = arith.select(is_aligned, pid_aligned, flat_pid)
                pid = fx.Int32(pid)
            else:
                # OGS-style contiguous-chunk swizzle.
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
        if GROUP_M > 1:
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
            """Register → LDS (no swizzle; tr16_b64 read handles bank alignment).

            The LDS layout is (STAGES, BLOCK_K, BS_N_STRIDE) with K-outer.
            Writes are straightforward N-contiguous vec_stores; the matching
            ``ds_read_tr16_b64`` reads rely on the hardware transpose to
            rearrange 4×4 sub-blocks into MFMA-fragment layout, and the
            row-stride pad handles bank spreading on reads.
            """
            for i in range_constexpr(LDG_REG_B_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                k_local_idx = global_tid // LDG_B_X_THREADS
                n_local_idx = global_tid % LDG_B_X_THREADS * LDG_VEC_SIZE
                bs_.vec_store(
                    (fx.Index(lds_stage), k_local_idx, n_local_idx),
                    vecs[i], LDG_VEC_SIZE,
                )

        def lds_matrix_b(lds_stage):
            """LDS → register MFMA B fragment via ``rocdl.ds_read_tr16_b64``.

            The CDNA4 hardware-transposed LDS read performs a 4×4 transpose
            within each 16-lane block: for lane L in a block,
            ``output[L, e] = Input[e*4 + (L%16)//4, L%4]`` where ``Input[s, p]``
            is the p-th f16 from the b64 (4 f16) loaded by source lane s.

            Each MFMA B fragment needs 8 K-contig f16 per lane at one N. Two
            ``ds_read_tr16_b64`` calls give this: read 0 produces B[k_base..k_base+3, N],
            read 1 produces B[k_base+4..k_base+7, N]. Per-lane address for
            read r in block ``bl = L // 16``:
              row = warp_atom_k + bl*8 + r*4 + (L%16)//4
              col = warp_atom_n + (L%16)%4 * 4
            After hardware transpose, lane L's output is 4 f16 of B at its
            (N = warp_atom_n + L%16) column. Concatenating the two reads
            produces the 8-f16 fragment.

            This replaces the MVP's 8 scalar strided loads per lane per
            fragment with 2 vectorised b64 reads, eliminating the dominant
            LDS bottleneck (was 92M LDS ops + 135M bank-conflict cycles).
            """
            s = fx.Index(lds_stage)
            b_frags = [0] * (WARP_K_STEPS * WARP_N_STEPS)
            FRAG = WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
            assert FRAG == 8, f"ds_read_tr16_b64 path assumes FRAG=8, got {FRAG}"
            v4_type = T.vec(4, dtype_)
            v8_type = T.vec(FRAG, dtype_)
            lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
            lb = w_tid % WMMA_N                # 0..15 lane-in-block (n_col direction)
            sg = lb // 4                        # 0..3 sub-group
            pr = lb % 4                         # 0..3 position-in-sub-group
            block_offset = w_tid // WMMA_N     # 0..3 k_group block
            for kk in range_constexpr(WARP_K_STEPS):
                for jj in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    halves = []
                    for r in range_constexpr(2):
                        # Per-lane (row, col) targeting the source data for
                        # the 4×4 transpose:
                        row = warp_atom_k_idx + block_offset * 8 + r * 4 + sg
                        col = warp_atom_n_idx + pr * 4
                        # LDS byte address = base + (stage*stride_stage + row*stride_row + col)*2
                        lds_byte_offset = bs_.linear_offset(
                            (s, fx.Index(row), fx.Index(col))
                        ) * DTYPE_BYTES
                        lds_base = memref.extract_aligned_pointer_as_index(bs_.memptr)
                        lds_addr_idx = lds_base + lds_byte_offset
                        lds_addr_i64 = arith.index_cast(T.i64, lds_addr_idx)
                        lds_ptr = llvm.inttoptr(lds_ptr_type, lds_addr_i64)
                        v4 = rocdl.ds_read_tr16_b64(v4_type, lds_ptr).result
                        halves.append(v4)
                    # Concat 2 × v4 into v8 by extracting + reassembling.
                    elems = []
                    for h in range_constexpr(2):
                        for e in range_constexpr(4):
                            elems.append(vector.extract(
                                halves[h], static_position=[e], dynamic_position=[],
                            ))
                    vec = vector.from_elements(v8_type, elems)
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

        # ---------- N-slicing variants (SLICE_N=True) ----------
        if SLICE_N:
            assert WARP_N_STEPS % 2 == 0, "SLICE_N requires even WARP_N_STEPS"
            HALF_N_STEPS = WARP_N_STEPS // 2

        def lds_matrix_b_half(lds_stage, n_half):
            """Like ``lds_matrix_b`` but loads only one N-half.

            Returns ``WARP_K_STEPS * HALF_N_STEPS`` fragments. b_frags_half[
            kk * HALF_N_STEPS + jj_local] = full lds_matrix_b's
            [kk * WARP_N_STEPS + (n_half * HALF_N_STEPS + jj_local)].
            """
            s = fx.Index(lds_stage)
            HALF = WARP_N_STEPS // 2
            j_offset = n_half * HALF
            b_frags_h = [0] * (WARP_K_STEPS * HALF)
            FRAG = WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
            v4_type = T.vec(4, dtype_)
            v8_type = T.vec(FRAG, dtype_)
            lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
            lb = w_tid % WMMA_N
            sg = lb // 4
            pr = lb % 4
            block_offset = w_tid // WMMA_N
            for kk in range_constexpr(WARP_K_STEPS):
                for jj_local in range_constexpr(HALF):
                    jj = j_offset + jj_local
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
                    b_frags_h[kk * HALF + jj_local] = vec
            return b_frags_h

        def block_mma_sync_half(a_frags, b_frags_h, c_frags, n_half):
            """Run MFMAs for one N-half. b_frags_h has WARP_K_STEPS * HALF
            fragments; c_frags is the full output (we only update the half
            corresponding to ``n_half``)."""
            HALF = WARP_N_STEPS // 2
            j_offset = n_half * HALF
            for kk in range_constexpr(WARP_K_STEPS):
                for ii in range_constexpr(WARP_M_STEPS):
                    a_frag = a_frags[kk * WARP_M_STEPS + ii]
                    for jj_local in range_constexpr(HALF):
                        jj = j_offset + jj_local
                        b_frag = b_frags_h[kk * HALF + jj_local]
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
                # In the async-copy path, A goes HBM→LDS directly (no register
                # round-trip, no explicit sts_a). But B still does ldg_b →
                # sts_b, so we must include LDG_REG_B_COUNT dswr hints alongside
                # the vmem hints — otherwise the compiler doesn't schedule the
                # B LDS writes against MFMA, leaving MFMA idle during dswr.
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

        if PIPELINE_DEPTH == 3:
            # 3-stage compute pipeline. Each loop iter overlaps
            # HBM→LDS (for iter k+2) + ds_read (for iter k+1) + MFMA
            # (on iter k regs). STAGES=2 LDS suffices because iter k
            # writes slot k%2 and reads slot (k+1)%2 — different
            # slots — and the data being read was written 1 iter ago
            # (the gpu.barrier at end of prev iter is the fence).
            # Prologue must prefetch 2 iters before entering the
            # steady loop; epilogue drains the last 2 MFMAs.

            # Prologue: HBM→LDS iters 0 and 1, then ds_read iter 0.
            if ASYNC_COPY:
                ldg_sts_a_async(k_begin, 0)
            else:
                sts_a(ldg_a(k_begin), 0)
            b_regs0 = ldg_b(k_begin)
            sts_b(b_regs0, 0)

            k_one = k_begin + fx.Int32(BLOCK_K)
            if ASYNC_COPY:
                ldg_sts_a_async(k_one, 1)
            else:
                sts_a(ldg_a(k_one), 1)
            b_regs1 = ldg_b(k_one)
            sts_b(b_regs1, 1)

            gpu.barrier()
            a_frags = lds_matrix_a(0)
            b_frags = lds_matrix_b(0)
            rocdl.sched_barrier(0)

            # Steady loop: c_iter goes from 0 to BLOCK_K_LOOPS-3
            # (BLOCK_K_LOOPS-2 iters total). Each iter:
            #   * HBM→LDS iter (c_iter+2) → slot c_iter%2
            #   * ds_read iter (c_iter+1) from slot (c_iter+1)%2 → next regs
            #   * MFMA on iter c_iter regs
            # state = [k_offset (= c_iter*BLOCK_K), c_iter%2,
            #          c_frags, a_frags, b_frags]
            init_state = (
                [k_begin, arith.constant(0, index=True)]
                + c_frags + a_frags + b_frags
            )
            STEADY_ITERS = BLOCK_K_LOOPS - 2
            for _bki, state in range(0, STEADY_ITERS, init=init_state):
                k_offset = state[0]
                w_slot = fx.Index(state[1])         # c_iter % 2 — overwriting now-stale data
                r_slot = 1 - w_slot                  # (c_iter+1) % 2 — has iter c_iter+1's data
                c_frags = state[2 : 2 + C_FRAGS_LEN]
                a_frags = state[2 + C_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN]
                b_frags = state[2 + C_FRAGS_LEN + A_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN + B_FRAGS_LEN]

                # HBM→LDS iter c_iter+2 → w_slot.
                k_load = k_offset + fx.Int32(2 * BLOCK_K)
                if ASYNC_COPY:
                    ldg_sts_a_async(k_load, w_slot)
                else:
                    a_regs_next = ldg_a(k_load)
                b_regs_next = ldg_b(k_load)
                # ds_read iter c_iter+1 from r_slot — overlaps with MFMA
                # since the slots differ (no internal barrier).
                a_frags_next = lds_matrix_a(r_slot)
                b_frags_next = lds_matrix_b(r_slot)
                # MFMA on iter c_iter regs.
                block_mma_sync(a_frags, b_frags, c_frags)
                if not ASYNC_COPY:
                    sts_a(a_regs_next, w_slot)
                sts_b(b_regs_next, w_slot)
                hot_loop_scheduler()
                gpu.barrier()
                k_offset = k_offset + fx.Int32(BLOCK_K)
                rocdl.sched_barrier(0)
                # Toggle stage parity for next iter.
                next_w_slot = 1 - w_slot
                results = yield (
                    [k_offset, next_w_slot] + c_frags + a_frags_next + b_frags_next
                )
            # Epilogue: drain last 2 MFMAs.
            # After loop, regs hold iter (BLOCK_K_LOOPS - 2)'s data.
            # Slot ((BLOCK_K_LOOPS - 1) % 2) holds iter BLOCK_K_LOOPS-1's data.
            c_frags = results[2 : 2 + C_FRAGS_LEN]
            a_frags = results[2 + C_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN]
            b_frags = results[2 + C_FRAGS_LEN + A_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN + B_FRAGS_LEN]
            # MFMA on iter (BLOCK_K_LOOPS - 2)
            block_mma_sync(a_frags, b_frags, c_frags)
            # ds_read iter (BLOCK_K_LOOPS - 1) — prologue/loop already wrote it.
            last_slot = (BLOCK_K_LOOPS - 1) % 2
            a_frags_last = lds_matrix_a(last_slot)
            b_frags_last = lds_matrix_b(last_slot)
            block_mma_sync(a_frags_last, b_frags_last, c_frags)
        elif SLICE_N:
            # PIPELINE_DEPTH == 2 with N-slicing. Carry only half-B
            # across iter boundary; load the other half mid-iter
            # (after mma_h0 retires h0's regs). Halves peak B-frag
            # register pressure.
            HALF = WARP_N_STEPS // 2
            B_FRAGS_H_LEN = WARP_K_STEPS * HALF

            # Prologue: HBM→LDS iter 0; ds_read full A, only B-h0.
            if ASYNC_COPY:
                ldg_sts_a_async(k_begin, 0)
            else:
                sts_a(ldg_a(k_begin), 0)
            b_regs0 = ldg_b(k_begin)
            sts_b(b_regs0, 0)
            gpu.barrier()
            a_frags = lds_matrix_a(0)
            b_frags_h0 = lds_matrix_b_half(0, 0)
            rocdl.sched_barrier(0)

            init_state = (
                [k_begin, arith.constant(0, index=True)]
                + c_frags + a_frags + b_frags_h0
            )
            for _bki, state in range(1, BLOCK_K_LOOPS, init=init_state):
                k_offset = state[0]
                current_stage = fx.Index(state[1])
                next_stage = 1 - current_stage
                c_frags = state[2 : 2 + C_FRAGS_LEN]
                a_frags = state[2 + C_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN]
                b_frags_h0 = state[
                    2 + C_FRAGS_LEN + A_FRAGS_LEN
                    : 2 + C_FRAGS_LEN + A_FRAGS_LEN + B_FRAGS_H_LEN
                ]
                # HBM→LDS iter k → next_stage (overlaps with mma_h0).
                if ASYNC_COPY:
                    ldg_sts_a_async(k_offset + BLOCK_K, next_stage)
                else:
                    a_regs_next = ldg_a(k_offset + BLOCK_K)
                b_regs_next = ldg_b(k_offset + BLOCK_K)
                # mma_h0 of iter k-1's data.
                block_mma_sync_half(a_frags, b_frags_h0, c_frags, 0)
                # b_frags_h0 dies here.
                if not ASYNC_COPY:
                    sts_a(a_regs_next, next_stage)
                sts_b(b_regs_next, next_stage)
                hot_loop_scheduler()
                gpu.barrier()
                # Load h1 of iter k-1 from current_stage (different from
                # next_stage just written, no extra barrier needed).
                b_frags_h1 = lds_matrix_b_half(current_stage, 1)
                block_mma_sync_half(a_frags, b_frags_h1, c_frags, 1)
                # b_frags_h1 dies.
                # Load iter k's full A and h0-only B for next iter.
                a_frags_next = lds_matrix_a(next_stage)
                b_frags_next_h0 = lds_matrix_b_half(next_stage, 0)
                k_offset = k_offset + fx.Int32(BLOCK_K)
                rocdl.sched_barrier(0)
                results = yield (
                    [k_offset, next_stage] + c_frags + a_frags_next + b_frags_next_h0
                )
            # Drain: mma_h0 + load h1 + mma_h1 for the last iter.
            c_frags = results[2 : 2 + C_FRAGS_LEN]
            a_frags = results[2 + C_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN]
            b_frags_h0 = results[
                2 + C_FRAGS_LEN + A_FRAGS_LEN
                : 2 + C_FRAGS_LEN + A_FRAGS_LEN + B_FRAGS_H_LEN
            ]
            block_mma_sync_half(a_frags, b_frags_h0, c_frags, 0)
            last_stage = (BLOCK_K_LOOPS - 1) % 2
            b_frags_h1_last = lds_matrix_b_half(last_stage, 1)
            block_mma_sync_half(a_frags, b_frags_h1_last, c_frags, 1)
        else:
            # PIPELINE_DEPTH == 2 (default): single-iter lookahead, ds_read
            # serialised after gpu.barrier. Original flow.
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
        from flydsl._mlir.dialects import scf
        stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES
        stmatrix_c_n_idx = w_tid % WMMA_N

        if DIRECT_WRITE:
            # Direct reg→HBM writeback. Per atom:
            #   1. each lane selects c_frag[src_kk] where src_kk = (l//4)%4
            #      via 3 conditional selects.
            #   2. 4 ds_bpermute transposes the fragment so output lane l
            #      ends up holding 4 f32 values at (row = l//4, cols =
            #      (l%4)*4 + 0..3) of the atom's 16×16 output.
            #   3. pack 4 f32 → 4 bf16 (or f16 via cvt_pkrtz) into a
            #      vector<4 x dtype> per lane.
            #   4. one vec_store of size 4 to HBM at the right (row, col).
            #
            # No LDS staging, no barriers. Per warp: 32 atoms × 1 store.
            # Layout knobs for the transpose:
            src_kk_v = (w_tid // fx.Int32(4)) % fx.Int32(4)  # per-lane runtime
            dst_row_in_atom = w_tid // fx.Int32(4)            # 0..15
            dst_col_start_in_atom = (w_tid % fx.Int32(4)) * fx.Int32(4)
            # Source lane mapping (for output kk_out 0..3):
            #   src_lane(l, kk_out) = (l//16)*16 + (l%4)*4 + kk_out
            sl_base = (w_tid // fx.Int32(16)) * fx.Int32(16) + (w_tid % fx.Int32(4)) * fx.Int32(4)

            # Pre-bitcast eq predicates (per-lane, atom-invariant).
            eq0 = arith.cmpi(arith.CmpIPredicate.eq, src_kk_v, fx.Int32(0))
            eq1 = arith.cmpi(arith.CmpIPredicate.eq, src_kk_v, fx.Int32(1))
            eq2 = arith.cmpi(arith.CmpIPredicate.eq, src_kk_v, fx.Int32(2))

            for ii in range_constexpr(WARP_M_STEPS):
                warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                for jj in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                    c_frag = c_frags[ii * WARP_N_STEPS + jj]
                    # Bitcast each c_frag[kf] to i32 (source for bpermute).
                    cf_i32 = []
                    for kf in range_constexpr(4):
                        v = vector.extract(c_frag, static_position=[kf], dynamic_position=[])
                        cf_i32.append(arith.bitcast(T.i32, v))

                    # 16 ds_bpermutes: bp[kk_out][kf] = src_lane(kk_out)'s c_frag[kf].
                    # Per output kk_out, select bp[kk_out][src_kk] where
                    # src_kk = (l//4)%4 is the OUTPUT lane's selector.
                    out_vals_i32 = []
                    for kk_out in range_constexpr(4):
                        idx_lane = sl_base + fx.Int32(kk_out)
                        idx_byte = idx_lane * fx.Int32(4)
                        idx_v = (idx_byte.ir_value()
                                 if hasattr(idx_byte, "ir_value")
                                 else idx_byte.value)
                        bps = []
                        for kf in range_constexpr(4):
                            bps.append(rocdl.DsBpermuteOp(T.i32, idx_v, cf_i32[kf]).result)
                        # Select output lane's src_kk-th value.
                        sel_i32 = arith.select(
                            eq0, bps[0],
                            arith.select(
                                eq1, bps[1],
                                arith.select(eq2, bps[2], bps[3]),
                            ),
                        )
                        out_vals_i32.append(sel_i32)
                    # Bitcast i32 → f32
                    out_vals_f32 = [arith.bitcast(T.f32, v) for v in out_vals_i32]

                    # Step 3: pack 4 f32 → 4 dtype values.
                    if dtype == "f16":
                        pk01 = rocdl.CvtPkRtz(T.vec(2, T.f16),
                                              out_vals_f32[0],
                                              out_vals_f32[1]).result
                        pk23 = rocdl.CvtPkRtz(T.vec(2, T.f16),
                                              out_vals_f32[2],
                                              out_vals_f32[3]).result
                        e0 = vector.extract(pk01, static_position=[0], dynamic_position=[])
                        e1 = vector.extract(pk01, static_position=[1], dynamic_position=[])
                        e2 = vector.extract(pk23, static_position=[0], dynamic_position=[])
                        e3 = vector.extract(pk23, static_position=[1], dynamic_position=[])
                    else:  # bf16 — default truncf gets paired into v_cvt_pk_bf16_f32 by the backend.
                        e0 = arith.truncf(dtype_, out_vals_f32[0])
                        e1 = arith.truncf(dtype_, out_vals_f32[1])
                        e2 = arith.truncf(dtype_, out_vals_f32[2])
                        e3 = arith.truncf(dtype_, out_vals_f32[3])
                    out_vec = vector.from_elements(T.vec(4, dtype_), [e0, e1, e2, e3])

                    # Step 4: store. Lane l writes 4 bf16 at
                    # (row = warp_atom_m + l//4, col_start = warp_atom_n + (l%4)*4).
                    m_local_idx = fx.Index(warp_atom_m_idx + dst_row_in_atom)
                    n_local_idx = fx.Index(warp_atom_n_idx + dst_col_start_in_atom)
                    m_global_idx = m_offset + m_local_idx
                    cond_boundary = arith.cmpi(
                        arith.CmpIPredicate.ult, m_global_idx, fx.Index(m),
                    )
                    cond_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                    with ir.InsertionPoint(cond_if.then_block):
                        C_.vec_store(
                            (m_global_idx, n_offset + n_local_idx),
                            out_vec, 4,
                        )
                        scf.YieldOp([])
            return  # skip the LDS-staged 2-pass writeback below

        # ---------- 2-pass LDS-staged writeback (default) ----------
        #
        # LDS C staging is halved (BM × BN/2). Each pass:
        #   (a) active warps (whose warp_n_idx falls in this pass's N-half)
        #       write their c_frags to LDS;
        #   (b) all threads vec_load from LDS and vec_store to HBM at
        #       the correct N offset.
        #
        # With the (BMW=2, BNW=4) layout and WARP_N=TILE_N/BNW, each warp's
        # N-range (WARP_N wide) falls cleanly inside one pass's BN/2 slab
        # whenever WARP_N divides BN/2 — holds for 256×256 / (2,4). The
        # runtime ``my_warp_pass`` compare just selects 2 of the 4 N-warps.
        my_wn = wid % fx.Int32(BLOCK_N_WARPS)
        # pass id of this warp's N-range: (wn * WARP_N) // (BN/2)
        my_warp_pass = (my_wn * fx.Int32(WARP_N)) // fx.Int32(C_LDS_N)

        for pass_idx in range_constexpr(2):
            PASS_N_START = pass_idx * C_LDS_N

            gpu.barrier()

            # (a) MFMA fragments → LDS, only for warps whose N-range is in
            # this pass. The cond is uniform within a warp so scf.if on
            # arith.cmpi is lane-safe (see AGENTS.md note).
            in_pass = arith.cmpi(
                arith.CmpIPredicate.eq, my_warp_pass, fx.Int32(pass_idx),
            )
            wrote_if = scf.IfOp(in_pass, results_=[], has_else=False)
            with ir.InsertionPoint(wrote_if.then_block):
                # Pair-packed truncf for f16 via rocdl.cvt.pkrtz. Each MFMA
                # C-fragment stores 4 f32 values at 4 consecutive M rows at
                # one N column; pairing adjacent (kk, kk+1) lets the backend
                # emit ``v_cvt_pkrtz_f16_f32`` which doesn't have the
                # MFMA→VALU hazard stall that scalar ``v_cvt_f16_f32_e32``
                # hits — saves ~34 × 5-cycle s_nops per wave. bf16 already
                # gets packed ``v_cvt_pk_bf16_f32`` out of the default
                # ``truncf`` lowering, no intrinsic needed there.
                _use_pkrtz = dtype == "f16"
                _pk_res_ty = T.vec(2, T.f16) if _use_pkrtz else None
                for ii in range_constexpr(WARP_M_STEPS):
                    warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                    for jj in range_constexpr(WARP_N_STEPS):
                        warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                        # Subtract PASS_N_START so LDS cs_ indices live in
                        # [0, C_LDS_N).
                        warp_atom_n_lds = (
                            warp_atom_n_idx - fx.Int32(PASS_N_START)
                        )
                        c_frag = c_frags[ii * WARP_N_STEPS + jj]
                        if _use_pkrtz:
                            assert WMMA_C_FRAG_VALUES % 2 == 0
                            for kk2 in range_constexpr(WMMA_C_FRAG_VALUES // 2):
                                kk = kk2 * 2
                                v0 = vector.extract(
                                    c_frag, static_position=[kk],
                                    dynamic_position=[],
                                )
                                v1 = vector.extract(
                                    c_frag, static_position=[kk + 1],
                                    dynamic_position=[],
                                )
                                pk = rocdl.CvtPkRtz(_pk_res_ty, v0, v1).result
                                e0 = vector.extract(
                                    pk, static_position=[0],
                                    dynamic_position=[],
                                )
                                e1 = vector.extract(
                                    pk, static_position=[1],
                                    dynamic_position=[],
                                )
                                lds_n_idx = fx.Index(
                                    warp_atom_n_lds + stmatrix_c_n_idx,
                                )
                                cs_[
                                    fx.Index(warp_atom_m_idx
                                             + stmatrix_c_m_vec_idx + kk),
                                    lds_n_idx,
                                ] = e0
                                cs_[
                                    fx.Index(warp_atom_m_idx
                                             + stmatrix_c_m_vec_idx + kk + 1),
                                    lds_n_idx,
                                ] = e1
                        else:
                            for kk in range_constexpr(WMMA_C_FRAG_VALUES):
                                lds_m_idx = fx.Index(
                                    warp_atom_m_idx
                                    + stmatrix_c_m_vec_idx + kk,
                                )
                                lds_n_idx = fx.Index(
                                    warp_atom_n_lds + stmatrix_c_n_idx,
                                )
                                val = vector.extract(
                                    c_frag, static_position=[kk],
                                    dynamic_position=[],
                                )
                                cs_[lds_m_idx, lds_n_idx] = val.truncf(dtype_)
                scf.YieldOp([])

            gpu.barrier()

            # (b) LDS → HBM; all threads write this pass's N-half.
            for i in range_constexpr(LDG_REG_C_HALF):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS_H)
                n_local_idx = fx.Index(
                    global_tid % LDG_C_X_THREADS_H * LDG_VEC_SIZE,
                )
                m_global_idx = m_offset + m_local_idx
                cond_boundary = arith.cmpi(
                    arith.CmpIPredicate.ult, m_global_idx, fx.Index(m),
                )
                cond_boundary_if = scf.IfOp(
                    cond_boundary, results_=[], has_else=False,
                )
                with ir.InsertionPoint(cond_boundary_if.then_block):
                    vec = cs_.vec_load(
                        (m_local_idx, n_local_idx), LDG_VEC_SIZE,
                    )
                    C_.vec_store(
                        (
                            m_global_idx,
                            n_offset + fx.Index(PASS_N_START) + n_local_idx,
                        ),
                        vec, LDG_VEC_SIZE,
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
        total_tiles = bm * bn
        nn_kernel._func.__name__ = KERNEL_NAME
        launcher = nn_kernel(C, A, B, m)
        # Occupancy hint: 2 waves per EU lets the hardware scheduler keep
        # more waves in flight to hide LDS / DMA latency. The c_frags alone
        # hold 128 f32 values per wave (= ~32 vregs), so default auto-
        # allocation tends toward 3-4 waves/EU anyway; pinning at 2 limits
        # the per-wave register count ceiling so the compiler doesn't spill.
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, 3)
        # 1D grid launch; in-kernel xcd_swizzle + swizzle2d derive 2D (pid_m,
        # pid_n) from the flat block_idx.x. Gives the scheduler freedom to
        # redistribute adjacent tiles across XCDs while preserving L2 reuse
        # via GROUP_M-style M-tile grouping.
        launcher.launch(grid=(total_tiles, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_nn_kernel


# ---------- Public API ----------


def gemm_nn_big(a: Tensor, b: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """256×256×64 / 8-warp NN-layout GEMM. Same semantics as ``gemm_nn`` —
    for large-M, large-N shapes where the 4-warp 128×128 path loses to
    hipBLASLt / OGS.

    Constraints: M % 256 == 0 and N % 256 == 0; K % 64 == 0.
    """
    assert a.is_cuda and b.is_cuda
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"inner dims must match: {a.shape} @ {b.shape}"
    assert a.dtype == b.dtype and a.dtype in _DTYPE2STR
    assert M % 256 == 0 and N % 256 == 0 and K % 64 == 0, (
        f"gemm_nn_big requires M%256 and N%256 and K%64; got {M}×{K}×{N}"
    )
    assert a.stride(-1) == 1 and b.stride(-1) == 1
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=a.dtype)
    else:
        assert out.shape == (M, N) and out.dtype == a.dtype
        assert out.stride(-1) == 1
    dtype_str = _DTYPE2STR[a.dtype]
    _compile_nn_big_kernel(dtype_str, K, N, _m_hint=M)(out, a, b, M)
    return out


__all__ = ["gemm_nn_big"]
