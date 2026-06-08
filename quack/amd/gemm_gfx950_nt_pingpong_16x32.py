# Copyright (c) 2026, AMD.

"""NT-layout bf16/f16 GEMM, 4-quadrant pingpong variant (gfx950).

Sibling of ``gemm_gfx950_nt_pingpong.py``. Same MFMA atom
(``mfma_f32_16x16x32_bf16``) and same 256×256 output tile, but the
warp's 128×64 tile is split into a 2×2 grid of 64×32 QUADRANTS, with 4
fp32 accumulators ``C[m_half][n_half]`` instead of one big one. Per
K-step we issue 4 MMA clusters (one per quadrant) of 16 mfmas each
instead of 2 clusters of 32 mfmas — same total work, finer granularity
for the SIMD scheduler to overlap LDS reads with compute.

Operand reuse across the 4 clusters:
    LD A_h0, B_h0  → MMA C[0][0]
    LD B_h1        → MMA C[0][1]   (reuses A_h0)
    LD A_h1        → MMA C[1][0]   (reuses B_h0)
                     MMA C[1][1]   (reuses A_h1, B_h1 — no new LDS)

Models HK ``256_256_64_32_with16x32.cpp``. HK reports +10.9% over the
32x16 variant at 8192³ but attributes it to clock (smaller MFMA →
lower power per cycle → higher sustained clock under TDP cap). We
already use the 16x16x32 atom in both files; the structural delta this
file targets is finer-grain LD/MMA interleaving.

Public API:

    gemm_nt_pingpong_16x32(a, b, out=None)

Same shape constraints: M % 256, N % 256, K % 64.
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
def _compile_nt_pingpong_16x32_kernel(
    dtype: str, k: int, n: int, _m_hint: int = 0,
    XCD_SWIZZLE: int = 1,
    GROUP_M: int = 1,
):
    """Compile the 4-quadrant NT pingpong kernel for ``(dtype, K, N)``.

    Same swizzle hooks as the 32x16 variant — see docstring there for
    why they default off.
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

    # ----- 4-quadrant decomposition (HK interleaved warp layout) -----
    # BLOCK_M=256 split into 2 M-halves of HALF_BLOCK_M=128 rows each.
    # Each M-half is further split into 2 warp-stripes of HALF_WARP_M=64
    # rows (warp_row indexes within the half). Similarly for N.
    #
    # Each warp's 128×64 footprint is split into 4 quadrants of 64×32:
    #   m_h ∈ {0,1}  selects M-half subregion (rows m_h*128 within tile)
    #   n_h ∈ {0,1}  selects N-half subregion (cols n_h*128 within tile)
    # Within a quadrant:
    #   M_ATOMS_PER_QUAD × N_ATOMS_PER_QUAD = 4 × 2 atoms
    #
    # HK's interleaved ownership: warp_row=0 takes BLOCK_M rows {0..63, 128..191},
    # warp_row=1 takes {64..127, 192..255}. (Old contiguous layout: warp_row=0
    # took 0..127 entirely; this is what the 32x16 sibling still uses.)
    #
    # Flat c_frags layout: c_frags[q * ATOMS_PER_QUAD + ai * N_ATOMS_PER_QUAD + aj]
    #   q = m_h * 2 + n_h, ai ∈ 0..3, aj ∈ 0..1.
    HALF_BLOCK_M = BLOCK_M // 2                                  # = 128
    HALF_BLOCK_N = BLOCK_N // 2                                  # = 128
    HALF_WARP_M = WARP_M // 2                                    # =  64
    HALF_WARP_N = WARP_N // 2                                    # =  32
    M_ATOMS_PER_QUAD = WARP_M_STEPS // 2                         # =   4
    N_ATOMS_PER_QUAD = WARP_N_STEPS // 2                         # =   2
    ATOMS_PER_QUAD = M_ATOMS_PER_QUAD * N_ATOMS_PER_QUAD         # =   8

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
    # Attempted: 8 separate ``memref.global`` (one per subregion) to get
    # ``AMDGPULowerModuleLDSPass`` to attach distinct alias scopes per
    # subregion. Didn't work — FlyDSL's ``raw_ptr_buffer_load_lds``
    # constructs the LDS pointer via ``inttoptr(extract_aligned_pointer_as_index
    # + offset)`` (see flydsl/expr/rocdl/__init__.py:386-393), which destroys
    # provenance. The buffer_load_lds intrinsic call ends up with no
    # ``!alias.scope`` metadata, so ``SIInsertWaitcnts`` falls back to
    # the conservative all-LDS-aliasing path either way. To unlock this
    # optimization upstream, FlyDSL would need to expose a memref→llvm.ptr<3>
    # path that doesn't go through int.
    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(
        None, arch=GPU_ARCH,
        global_sym_name=f"nt_pp16x32_smem_{dtype}_{k}_{n}",
    )

    smem_a_offset = allocator._align(allocator.ptr, 16)
    AS_BYTES = STAGES * BLOCK_M * BLOCK_K * DTYPE_BYTES
    allocator.ptr = smem_a_offset + AS_BYTES

    smem_b_offset = allocator._align(allocator.ptr, 16)
    BS_BYTES = STAGES * BLOCK_N * BLOCK_K * DTYPE_BYTES
    allocator.ptr = smem_b_offset + BS_BYTES

    BLOCK_K_LOOPS_HINT = max(1, k // BLOCK_K)

    KERNEL_NAME = f"nt_pp16x32_{dtype}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}"

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def nt_kernel_16x32(C: fx.Tensor, A: fx.Tensor, B: fx.Tensor, m: fx.Int32):
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
        # Helper: load M-half of A from LDS into registers.
        # Returns M_ATOMS_PER_QUAD × K_SUBITERS = 8 frags
        # indexed [ai * K_SUBITERS + kk] for ai ∈ 0..3, kk ∈ 0..1.
        # =========================================================
        def lds_matrix_a_half(lds_stage, m_half):
            s = fx.Index(lds_stage)
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
                    vec = as_.vec_load(
                        (s, row, col_in_bytes // DTYPE_BYTES), WMMA_A_FRAG,
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
            s = fx.Index(lds_stage)
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
                    vec = bs_.vec_load(
                        (s, row, col_in_bytes // DTYPE_BYTES), WMMA_B_FRAG,
                    )
                    b_frags[aj * K_SUBITERS + kk] = vec
            return b_frags

        # =========================================================
        # Helper: MFMA for one quadrant. Does 4M × 2N × 2K = 16 mfmas
        # contracted over K. Updates c_frags[q*8 + ai*2 + aj] in place.
        # Wrapped in s_setprio(1)/(0) for SIMD ping-pong bias.
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

        # ----- Accumulator init (4 quadrants × 8 atoms = 32 frags) -----
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS                # = 32
        c_frags = [acc_init] * C_FRAGS_LEN

        # ----- Waitcnt bitfield helpers (gfx9 encoding) -----
        # bits[3:0]=VMCNT_lo, bits[15:14]=VMCNT_hi, bits[6:4]=EXPCNT, bits[11:8]=LGKMCNT
        # All-ones (don't wait): 0xCF7F
        LGKMCNT_0 = 0xC07F        # drain LDS counter to 0
        LGKMCNT_8 = 0xC07F | (8 << 8)  # drain to lgkmcnt ≤ 8
        VMCNT_0 = 0x0F70          # drain memory counter to 0
        VMCNT_2 = 0x0F70 | 2
        VMCNT_4 = 0x0F70 | 4
        VMCNT_6 = 0x0F70 | 6

        # ----- Prologue stage 1: load K-step 0 fully into stage 0 -----
        # (Mirror HK lines 117-127: load Bs[0][0], As[0][0], Bs[0][1], As[0][1]
        # then desync + vmcnt(4) drain + barrier.)
        ldg_sts_b_half_async(0, 0, 0)
        ldg_sts_a_half_async(0, 0, 0)
        ldg_sts_b_half_async(0, 0, 1)
        ldg_sts_a_half_async(0, 0, 1)

        # HK staggering: warp_row==1 calls one EXTRA s_barrier here.
        # gfx9 s_barrier uses a shared workgroup counter — each wave's
        # barrier call increments it, release fires at WG_SIZE_WAVES.
        # With this extra barrier, warp_row==1's barrier N+1 pairs with
        # warp_row==0's barrier N forever after — the two wave-groups
        # execute the SAME instructions one barrier offset apart, so
        # while one half is doing LDS reads the other is doing MFMAs.
        # Closed at the END of the kernel with `if warp_row==0: barrier`
        # to re-sync.
        if arith.cmpi(arith.CmpIPredicate.eq, warp_row, fx.Int32(1)):
            rocdl.s_barrier()

        rocdl.s_waitcnt(VMCNT_4)
        rocdl.s_barrier()

        # ----- Prologue stage 2: load K-step 1 PARTIALLY into stage 1 -----
        # (Mirror HK lines 130-135: Bs[1][0], As[1][0], Bs[1][1]; As[1][1] is
        # DEFERRED — loaded by main loop's first LD0 to balance the pipeline.)
        ldg_sts_b_half_async(BLOCK_K, 1, 0)
        ldg_sts_a_half_async(BLOCK_K, 1, 0)
        ldg_sts_b_half_async(BLOCK_K, 1, 1)

        rocdl.s_waitcnt(VMCNT_6)
        rocdl.s_barrier()

        # ----- Main loop: 2 K-steps unrolled per iter, 8 LD/MMA clusters -----
        # Faithfully mirrors HK's k_16x32.cpp lines 139-251. Each iter processes
        # K-steps 2k (stage 0) and 2k+1 (stage 1), and prefetches K-steps 2k+2
        # (into stage 0) and 2k+3 (into stage 1 minus A_h1, which is loaded by
        # the next iter's first LD0).
        TOTAL_K_STEPS = k // BLOCK_K
        OUTER_ITERS = TOTAL_K_STEPS // 2
        assert TOTAL_K_STEPS % 2 == 0, "16x32 kernel requires K % 128 == 0"

        k_zero = arith.constant(0, type=T.i32)
        init_state = [k_zero] + c_frags
        # Loop runs OUTER_ITERS - 1 times; last 2 K-steps handled by epilogue.
        for _oi, state in range(0, OUTER_ITERS - 1, init=init_state):
            k_now = state[0]
            k_now_b = k_now + fx.Int32(BLOCK_K)
            k_next_a = k_now + fx.Int32(2 * BLOCK_K)
            k_next_b = k_now + fx.Int32(3 * BLOCK_K)
            c_frags = list(state[1 : 1 + C_FRAGS_LEN])

            # === K-step 2k from stage 0 ===
            # LD0: read B_h0, A_h0 (stage 0); prefetch A_h1 of K-step 2k+1 into stage 1
            b_h0_s0 = lds_matrix_b_half(0, 0)
            a_h0_s0 = lds_matrix_a_half(0, 0)
            ldg_sts_a_half_async(k_now_b, 1, 1)
            rocdl.s_waitcnt(LGKMCNT_8)
            rocdl.s_barrier()
            # MMA0: C[0][0]
            rocdl.s_waitcnt(LGKMCNT_0)
            mma_quadrant(a_h0_s0, b_h0_s0, c_frags, 0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # LD1: read B_h1 (stage 0); prefetch B_h0 of K-step 2k+2 into stage 0
            b_h1_s0 = lds_matrix_b_half(0, 1)
            ldg_sts_b_half_async(k_next_a, 0, 0)
            rocdl.s_barrier()
            # MMA1: C[0][1]
            rocdl.s_waitcnt(LGKMCNT_0)
            mma_quadrant(a_h0_s0, b_h1_s0, c_frags, 1)
            rocdl.s_barrier()

            # LD2: read A_h1 (stage 0); prefetch A_h0 of K-step 2k+2 into stage 0
            a_h1_s0 = lds_matrix_a_half(0, 1)
            ldg_sts_a_half_async(k_next_a, 0, 0)
            rocdl.s_barrier()
            # MMA2: C[1][0]
            rocdl.s_waitcnt(LGKMCNT_0)
            mma_quadrant(a_h1_s0, b_h0_s0, c_frags, 2)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # LD3: read B_h0 (stage 1) for K-step 2k+1; prefetch B_h1 of K-step 2k+2 into stage 0
            b_h0_s1 = lds_matrix_b_half(1, 0)
            ldg_sts_b_half_async(k_next_a, 0, 1)
            rocdl.s_waitcnt(VMCNT_6)
            rocdl.s_barrier()
            # MMA3: C[1][1] — no lgkmcnt drain (regs already loaded)
            mma_quadrant(a_h1_s0, b_h1_s0, c_frags, 3)
            rocdl.s_barrier()

            # === K-step 2k+1 from stage 1 ===
            # LD0: read A_h0 (stage 1); prefetch A_h1 of K-step 2k+2 into stage 0
            a_h0_s1 = lds_matrix_a_half(1, 0)
            ldg_sts_a_half_async(k_next_a, 0, 1)
            rocdl.s_waitcnt(LGKMCNT_8)
            rocdl.s_barrier()
            # MMA0
            rocdl.s_waitcnt(LGKMCNT_0)
            mma_quadrant(a_h0_s1, b_h0_s1, c_frags, 0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # LD1: read B_h1 (stage 1); prefetch B_h0 of K-step 2k+3 into stage 1
            b_h1_s1 = lds_matrix_b_half(1, 1)
            ldg_sts_b_half_async(k_next_b, 1, 0)
            rocdl.s_barrier()
            # MMA1
            rocdl.s_waitcnt(LGKMCNT_0)
            mma_quadrant(a_h0_s1, b_h1_s1, c_frags, 1)
            rocdl.s_barrier()

            # LD2: read A_h1 (stage 1); prefetch A_h0 of K-step 2k+3 into stage 1
            a_h1_s1 = lds_matrix_a_half(1, 1)
            ldg_sts_a_half_async(k_next_b, 1, 0)
            rocdl.s_barrier()
            # MMA2
            rocdl.s_waitcnt(LGKMCNT_0)
            mma_quadrant(a_h1_s1, b_h0_s1, c_frags, 2)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # LD3: NO ds_read; prefetch B_h1 of K-step 2k+3 into stage 1
            ldg_sts_b_half_async(k_next_b, 1, 1)
            rocdl.s_waitcnt(VMCNT_6)
            rocdl.s_barrier()
            # MMA3
            mma_quadrant(a_h1_s1, b_h1_s1, c_frags, 3)
            rocdl.s_barrier()

            results = yield [k_next_a] + c_frags

        # ----- Epilogue 1: K-step TOTAL_K_STEPS-2 from stage 0 -----
        # Mirror HK lines 254-292. State at this point: stage 0 has K-step
        # (OUTER_ITERS-1)*2 fully loaded; stage 1 has K-step (OUTER_ITERS-1)*2+1
        # with A_h1 missing. The epilogue's LD0 loads that A_h1 (HK's
        # `As[toc][1]` deferred load).
        c_frags = list(results[1 : 1 + C_FRAGS_LEN])
        k_last_a = fx.Int32((OUTER_ITERS - 1) * 2 * BLOCK_K)
        k_last_b = fx.Int32((OUTER_ITERS - 1) * 2 * BLOCK_K + BLOCK_K)

        # LD0: read B_h0, A_h0 (stage 0); load deferred A_h1 of last K-step into stage 1
        b_h0_s0 = lds_matrix_b_half(0, 0)
        a_h0_s0 = lds_matrix_a_half(0, 0)
        ldg_sts_a_half_async(k_last_b, 1, 1)
        rocdl.s_barrier()
        # MMA0
        rocdl.s_waitcnt(LGKMCNT_0)
        mma_quadrant(a_h0_s0, b_h0_s0, c_frags, 0)
        rocdl.s_barrier()

        # LD1
        b_h1_s0 = lds_matrix_b_half(0, 1)
        rocdl.s_barrier()
        # MMA1
        rocdl.s_waitcnt(LGKMCNT_0)
        mma_quadrant(a_h0_s0, b_h1_s0, c_frags, 1)
        rocdl.s_barrier()

        # LD2: also need vmcnt drain so the A_h1 load from LD0 has committed.
        a_h1_s0 = lds_matrix_a_half(0, 1)
        rocdl.s_waitcnt(VMCNT_0)
        rocdl.s_barrier()
        # MMA2 + MMA3 fused (per HK epilogue 1)
        rocdl.s_waitcnt(LGKMCNT_0)
        mma_quadrant(a_h1_s0, b_h0_s0, c_frags, 2)
        mma_quadrant(a_h1_s0, b_h1_s0, c_frags, 3)
        rocdl.s_barrier()

        # ----- Epilogue 2: K-step TOTAL_K_STEPS-1 from stage 1 -----
        # Mirror HK lines 295-330. Stage 1 fully loaded now (A_h1 was loaded
        # in epilogue 1).
        b_h0_s1 = lds_matrix_b_half(1, 0)
        a_h0_s1 = lds_matrix_a_half(1, 0)
        rocdl.s_barrier()
        rocdl.s_waitcnt(LGKMCNT_0)
        mma_quadrant(a_h0_s1, b_h0_s1, c_frags, 0)
        rocdl.s_barrier()

        b_h1_s1 = lds_matrix_b_half(1, 1)
        rocdl.s_barrier()
        rocdl.s_waitcnt(LGKMCNT_0)
        mma_quadrant(a_h0_s1, b_h1_s1, c_frags, 1)
        rocdl.s_barrier()

        a_h1_s1 = lds_matrix_a_half(1, 1)
        rocdl.s_barrier()
        rocdl.s_waitcnt(LGKMCNT_0)
        mma_quadrant(a_h1_s1, b_h0_s1, c_frags, 2)
        mma_quadrant(a_h1_s1, b_h1_s1, c_frags, 3)

        # HK staggering close: warp_row==0 calls one extra s_barrier
        # to re-sync with warp_row==1 before the writeback. Pairs with
        # the initial `if warp_row==1: barrier` in the prologue.
        if arith.cmpi(arith.CmpIPredicate.eq, warp_row, fx.Int32(0)):
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
            allocator.finalize()
        bm = (m + BLOCK_M - 1) // BLOCK_M
        bn = n // BLOCK_N
        total_tiles = bm * bn
        nt_kernel_16x32._func.__name__ = KERNEL_NAME
        launcher = nt_kernel_16x32(C, A, B, m)
        # With 8 waves/WG and waves_per_eu=2 we have 256 VGPRs/wave —
        # the compiler can fit everything in VGPRs (~210-240). Forcing
        # ``amdgpu-agpr-alloc=128,128`` (older NN-big pattern) splits the
        # unified pool into 128 VGPR + 128 AGPR and adds VGPR↔AGPR copies
        # per MFMA, costing ~6% perf. HK kernels run with VGPRs=210,
        # AGPRs=0 — same profile we get when we let the compiler decide.
        passthrough_attr = ir.ArrayAttr.get([])
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, 3)
                op.attributes["passthrough"] = passthrough_attr
        launcher.launch(grid=(total_tiles, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_nt_kernel


# ----- Public API -----

def gemm_nt_pingpong_16x32(
    a: Tensor, b: Tensor, out: Optional[Tensor] = None,
) -> Tensor:
    """NT GEMM (4-quadrant 16x32 variant) on gfx950.

    Computes ``C = A @ B^T`` for ``A=(M,K)`` row-major and ``B=(N,K)``
    row-major. Output ``(M, N)`` row-major in the input dtype.

    Constraints: M % 256 == 0, N % 256 == 0, K % 128 == 0 (= 2*BLOCK_K).
    """
    assert a.is_cuda and b.is_cuda
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2, f"inner dims must match: A={a.shape} B={b.shape}"
    assert a.dtype == b.dtype and a.dtype in _DTYPE2STR
    assert M % 256 == 0 and N % 256 == 0 and K % 128 == 0, (
        f"gemm_nt_pingpong_16x32 requires M%256, N%256, K%128; got {M}×{K}×{N}"
    )
    assert a.stride(-1) == 1 and b.stride(-1) == 1
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=a.dtype)
    else:
        assert out.shape == (M, N) and out.dtype == a.dtype
        assert out.stride(-1) == 1
    dtype_str = _DTYPE2STR[a.dtype]
    _compile_nt_pingpong_16x32_kernel(dtype_str, K, N, _m_hint=M)(out, a, b, M)
    return out


__all__ = ["gemm_nt_pingpong_16x32"]
