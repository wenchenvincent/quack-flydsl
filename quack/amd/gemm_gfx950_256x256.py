# Copyright (c) 2026, AMD.

"""256×256 tile MFMA GEMM — EXPERIMENTAL.

Profile-motivated: rocprof showed hipBLASLt's ``MT256x256x64_MI16x16x1``
configuration (256×256 macrotile, 4-wave WG, K=64 per iter) beats our
128×128 by 3.1× at 4096² because of 4× bigger per-WG tile leading to
less HBM traffic and fewer LDS ops per flop.

This kernel replicates the 4-wave × 128×128-per-wave structure with
K=16 per iter (no K-unroll yet).

**Measured: did NOT beat the 128×128 kernel.** MI355X f16:

    shape       128×128     256×256    hipBLASLt   256/128   256/hb
    512²        30 μs       54 μs      10 μs       0.56×     5.4×
    1024²       59 μs       103 μs     13 μs       0.57×     7.7×
    2048²       116 μs      217 μs     30 μs       0.53×     7.2×
    4096²       401 μs      489 μs     125 μs      0.82×     3.9×

**Profile comparison** (at 4096²):
    metric                   128×128     256×256     hipBLASLt
    duration                 356 μs      509 μs      114 μs
    VGPR                     56          192         256
    LDS/WG                   16 KiB      32 KiB      65 KiB
    scratch                  0           **0 (no spill!)**  0
    SQ_LDS_BANK_CONFLICT     84 M        **45 M** (better!) 65 k

Register pressure is NOT the problem (no spills at 192 VGPR).
Bank conflicts actually went DOWN. Yet 256×256 is slower.

**Why it doesn't win**: the consume loop issues all 64 LDS loads
upfront, then stalls waiting for results, then does all 64 MFMAs —
LDS latency isn't hidden behind compute. hipBLASLt's equivalent
kernel uses K=64 inner-unroll, which interleaves 4 k-tiles' worth of
LDS loads with MFMAs so compute hides the load wait.

Also, 256×256 creates 4× fewer workgroups (4 waves/CU vs 16 for
128×128). The hardware scheduler has less latency to hide across
waves.

**Path to winning**: this kernel is an infrastructure milestone for
replicating hipBLASLt's tile shape. The remaining gap is
**MFMA/LDS interleaving** in the inner loop — either manual
via ``rocdl.sched_vmem/mfma`` or via K-unroll. That's out of scope
for this commit.

Not wired into the autotune dispatcher.

**Structure:**

Each 256-thread workgroup (4 waves × 64 lanes) covers a 256×256
output tile via a 2×2 wave grid where **each wave computes a 128×128
register tile** via an **8×8 grid of 16×16 MFMAs**. That's **64 MFMAs
per lane per k-tile**, 4× more than the 128×128 kernel's 16.

Wave mapping:
    wave_id  ∈ [0, 4)
    wave_row = wave_id // 2 ∈ [0, 2)  → rows [0,128) or [128,256)
    wave_col = wave_id % 2  ∈ [0, 2)  → cols [0,128) or [128,256)

Per-lane register tile: 8 row-halves × 8 col-halves = **64
accumulators** (f32x4 each) = **256 f32 VGPRs just for acc**. With
fragment regs and spill pressure on top, we're targeting close to
CDNA4's full register file — if this spills we'd need to split into
K-chunks.

LDS per stage:
    A: (256, 16) f16  = 8 KiB
    B: (16, 256) f16  = 8 KiB
Total (ping-pong, 2 stages): 32 KiB per WG.

Cooperative HBM load (256 threads):
    A: 256×16 = 4096 f16 / 256 = 16 f16/thread = 2 × BufferCopy128b
       thread tid → row=tid, 2 vec8 chunks covering the 16 K elements
    B: 16×256 = 4096 f16 / 256 = 16 f16/thread = 2 × BufferCopy128b
       thread tid → fill (k_row, n_vec8_group) positions at 2 vec8s each

Target: 4096²+ shapes. At 4096² the grid is 16×16 = 256 WGs × 4 waves
= 1024 waves — matching hipBLASLt's waves count exactly.
"""

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu as _gpu, range_constexpr, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch


_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 16
_FRAG_A = 4
_FRAG_B = 4
_FRAG_C = 4
_WAVE_ROWS = 8         # row-halves per wave's 128-row tile (each 16 rows)
_WAVE_COLS = 8         # col-halves per wave's 128-col tile

_A_TILE_BYTES = 256 * 16 * 2
_B_TILE_BYTES = 16 * 256 * 2


def _align(x, a):
    return (x + a - 1) & ~(a - 1)


def _build_gemm_256x256(*, M, N, K, dtype_str, arch):
    assert M % 256 == 0 and N % 256 == 0 and K % _MFMA_K == 0
    assert dtype_str in ("f16", "bf16")

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_256x256_{dtype_str}_{M}_{N}_{K}_smem",
    )
    a_off0 = _align(allocator.ptr, 16); allocator.ptr = a_off0 + _A_TILE_BYTES
    b_off0 = _align(allocator.ptr, 16); allocator.ptr = b_off0 + _B_TILE_BYTES
    a_off1 = _align(allocator.ptr, 16); allocator.ptr = a_off1 + _A_TILE_BYTES
    b_off1 = _align(allocator.ptr, 16); allocator.ptr = b_off1 + _B_TILE_BYTES

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y
        tid = fx.thread_idx.x

        wave_id = tid // fx.Int32(64)
        lane_in_wave = tid % fx.Int32(64)
        lane_row = lane_in_wave % fx.Int32(16)
        lane_k_group = lane_in_wave // fx.Int32(16)

        wave_row = wave_id // fx.Int32(2)
        wave_col = wave_id % fx.Int32(2)

        in_elem_type = T.f16 if dtype_str == "f16" else T.bf16

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        base_ptr = allocator.get_base()
        s_A = [
            SmemPtr(base_ptr, a_off0, in_elem_type, shape=(256, 16)),
            SmemPtr(base_ptr, a_off1, in_elem_type, shape=(256, 16)),
        ]
        s_B = [
            SmemPtr(base_ptr, b_off0, in_elem_type, shape=(16, 256)),
            SmemPtr(base_ptr, b_off1, in_elem_type, shape=(16, 256)),
        ]
        for sp in (*s_A, *s_B):
            sp.get()

        ca_h8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), in_elem_type)
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        h8_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(8, 1), fx.AddressSpace.Register)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        h8_lay = fx.make_layout(8, 1)
        reg_lay = fx.make_layout(1, 1)

        def _load_h8(div_vec, vec_idx):
            r = fx.memref_alloca(h8_reg_ty, h8_lay)
            fx.copy_atom_call(ca_h8, fx.slice(div_vec, (None, vec_idx)), r)
            return fx.memref_load_vec(r)

        def _store_f(div, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            ts = _vfull(1, Float32(val), Float32)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_f, r, fx.slice(div, (None, idx)))

        def _idx(i):
            return ArithValue(i).index_cast(T.index) if hasattr(i, "index_cast") else fx.Index(i)

        m_base_wg = bid_m * fx.Int32(256)
        n_base_wg = bid_n * fx.Int32(256)

        # Cooperative load thread mappings. Each thread handles 2 vec8s
        # per tile (16 f16 total).
        #   A: row = tid, k_vec8 ∈ {0, 1}. tid covers all 256 rows.
        #   B: k_row = tid // 32, n_vec8_base = (tid % 32) * 2, then n_vec8 ∈ {base, base+1}.
        #     OR simpler: 2 loads per thread covering the B (16, 32) vec8 grid = 512 positions,
        #     tid → covers positions 2*tid and 2*tid+1 (linear stride).
        a_load_row = tid
        # B linear position (0..511), then row = pos // 32, n_vec8 = pos % 32.

        def _a_row_in_tile(r):
            return wave_row * fx.Int32(128) + fx.Int32(r * 16) + lane_row

        def _b_col_in_tile(c):
            return wave_col * fx.Int32(128) + fx.Int32(c * 16) + lane_row

        acc_ty = T.vec(_FRAG_C, T.f32)

        def _zero_acc():
            zs = []
            for _ in range_constexpr(_FRAG_C):
                zs.append(arith.constant(0.0, type=T.f32))
            return vector.from_elements(acc_ty, zs)

        # 8×8 accumulator grid per wave = 64 accumulators.
        accs = [[_zero_acc() for _ in range_constexpr(_WAVE_COLS)]
                for _ in range_constexpr(_WAVE_ROWS)]

        def _stage_load(k_tile_scalar, stage):
            k_base = fx.Int32(k_tile_scalar * _MFMA_K)
            # --- A: 2 vec8 per thread, both on row=tid ---
            a_hbm_row = m_base_wg + a_load_row
            a_row_slice = fx.slice(A_buf, (a_hbm_row, None))
            a_div_v = fx.logical_divide(a_row_slice, h8_lay)
            for k_vec in range_constexpr(2):
                a_hbm_vec_idx = fx.Int32(k_tile_scalar * 2) + fx.Int32(k_vec)
                a_chunk = _load_h8(a_div_v, a_hbm_vec_idx)
                for i in range_constexpr(8):
                    s_A[stage].store(
                        a_chunk[i].ir_value(),
                        [_idx(a_load_row),
                         _idx(fx.Int32(k_vec * 8) + fx.Int32(i))],
                    )
            # --- B: 2 vec8 per thread, linear positioning ---
            for pos_off in range_constexpr(2):
                linear_pos = tid * fx.Int32(2) + fx.Int32(pos_off)
                b_k = linear_pos // fx.Int32(32)
                b_n_vec8 = linear_pos % fx.Int32(32)
                b_hbm_k = k_base + b_k
                b_k_slice = fx.slice(B_buf, (b_hbm_k, None))
                b_div_v = fx.logical_divide(b_k_slice, h8_lay)
                b_hbm_vec_idx = bid_n * fx.Int32(256 // 8) + b_n_vec8
                b_chunk = _load_h8(b_div_v, b_hbm_vec_idx)
                for i in range_constexpr(8):
                    s_B[stage].store(
                        b_chunk[i].ir_value(),
                        [_idx(b_k),
                         _idx(b_n_vec8 * fx.Int32(8) + fx.Int32(i))],
                    )

        def _consume(stage, accs):
            # 8 A-row fragments (shared across 8 col-positions each).
            a_frags = []
            for r in range_constexpr(_WAVE_ROWS):
                a_vals = []
                a_row_l = _a_row_in_tile(r)
                for i in range_constexpr(_FRAG_A):
                    k_in_lds = lane_k_group * fx.Int32(_FRAG_A) + fx.Int32(i)
                    a_vals.append(s_A[stage].load([_idx(a_row_l), _idx(k_in_lds)]))
                a_frags.append(vector.from_elements(T.vec(_FRAG_A, in_elem_type), a_vals))

            # 8 B-col fragments (shared across 8 row-positions each).
            b_frags = []
            for c in range_constexpr(_WAVE_COLS):
                b_vals = []
                b_col_l = _b_col_in_tile(c)
                for i in range_constexpr(_FRAG_B):
                    k_in_lds = lane_k_group * fx.Int32(_FRAG_B) + fx.Int32(i)
                    b_vals.append(s_B[stage].load([_idx(k_in_lds), _idx(b_col_l)]))
                b_frags.append(vector.from_elements(T.vec(_FRAG_B, in_elem_type), b_vals))

            def _mfma(a, b, acc):
                if dtype_str == "bf16":
                    a_i = vector.bitcast(T.vec(_FRAG_A, T.i16), a)
                    b_i = vector.bitcast(T.vec(_FRAG_B, T.i16), b)
                    return fx.rocdl.mfma_f32_16x16x16bf16_1k(
                        acc_ty, [a_i, b_i, acc, 0, 0, 0],
                    )
                return fx.rocdl.mfma_f32_16x16x16f16(acc_ty, [a, b, acc, 0, 0, 0])

            # 8×8 = 64 MFMAs, each A fragment shared across 8 col ops (and
            # each B fragment across 8 row ops) = massive in-register reuse.
            for r in range_constexpr(_WAVE_ROWS):
                for c in range_constexpr(_WAVE_COLS):
                    accs[r][c] = _mfma(a_frags[r], b_frags[c], accs[r][c])
            return accs

        k_tiles = K // _MFMA_K

        _stage_load(0, 0)
        _gpu.barrier()

        for k_tile in range_constexpr(k_tiles):
            cur = k_tile & 1
            if k_tile < k_tiles - 1:
                _stage_load(k_tile + 1, 1 - cur)
            accs = _consume(cur, accs)
            _gpu.barrier()

        # Store C: each wave's 128×128 = 8×8 sub-tiles of 16×16.
        wave_m_base = m_base_wg + wave_row * fx.Int32(128)
        wave_n_base = n_base_wg + wave_col * fx.Int32(128)

        def _store_subtile(acc, row_base, col_base):
            for i in range_constexpr(_FRAG_C):
                out_row = row_base + lane_k_group * fx.Int32(_FRAG_C) + fx.Int32(i)
                out_col = col_base + lane_row
                row_c = fx.slice(C_buf, (out_row, None))
                c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))
                val_i = vector.extract(acc, static_position=[i], dynamic_position=[])
                _store_f(c_div, out_col, val_i)

        for r in range_constexpr(_WAVE_ROWS):
            for c in range_constexpr(_WAVE_COLS):
                _store_subtile(
                    accs[r][c],
                    wave_m_base + fx.Int32(r * 16),
                    wave_n_base + fx.Int32(c * 16),
                )

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, C).launch(
            grid=(M // 256, N // 256, 1), block=(256, 1, 1), stream=stream,
        )

    return launch


_kernel_cache: dict = {}
_DTYPE2STR = {torch.float16: "f16", torch.bfloat16: "bf16"}


def _compile(M, N, K, dtype_str, arch):
    key = (M, N, K, dtype_str, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_256x256(M=M, N=N, K=K, dtype_str=dtype_str, arch=arch)
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_gemm_256x256_out",
    mutates_args=("out",),
    schema="(Tensor a, Tensor b, Tensor(a0!) out) -> ()",
)
def _gemm_256x256_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    assert a.is_cuda and b.is_cuda and out.is_cuda
    assert a.dtype in (torch.float16, torch.bfloat16) and a.dtype == b.dtype
    assert out.dtype == torch.float32
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and M % 256 == 0 and N % 256 == 0 and K % 16 == 0
    assert out.shape == (M, N)
    _compile(M, N, K, _DTYPE2STR[a.dtype], get_rocm_arch())(a, b, out)


@_gemm_256x256_out.register_fake
def _gemm_256x256_out_fake(a, b, out):
    return None


def gemm_256x256(A: Tensor, B: Tensor) -> Tensor:
    """256×256-tile MFMA GEMM with cooperative LDS + ping-pong.

    Requires M, N multiples of 256; K multiple of 16. Matches
    hipBLASLt's ``MT256x256x64_MI16x16x1`` macrotile shape (we use
    K=16 per iteration; K=64 with 4× inner unroll is a follow-up).
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == B.dtype and A.dtype in (torch.float16, torch.bfloat16)
    M, K = A.shape
    _, N = B.shape
    out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    _gemm_256x256_out(A, B, out)
    return out


__all__ = ["gemm_256x256"]
