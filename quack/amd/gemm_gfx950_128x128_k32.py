# Copyright (c) 2026, AMD.

"""128×128 tile GEMM using the **K=32 MFMA** variant — 2× throughput
per instruction vs the K=16 baseline.

gfx950 / CDNA4 exposes ``mfma_f32_16x16x32_f16`` (and the bf16 twin)
which consumes twice the K per issue. Per-lane fragments become
8 f16 instead of 4; LDS per stage doubles (8 KiB A + 8 KiB B vs
4 KiB + 4 KiB).

Wave grid and accumulator structure match the K=16 128x128 baseline:
  - 4 waves per WG (2×2 grid), 256 threads
  - Each wave owns a 64×64 register tile via 4×4 grid of 16×16 MFMAs
  - 16 independent MFMAs per k-tile per lane, C acc = 16 × vec4_f32

LDS layout (per stage):
    A: (128, 32) f16  = 8 KiB
    B: (32, 128) f16  = 8 KiB
Total per stage: 16 KiB. Ping-pong = 32 KiB.

HBM cooperative load (each k-tile of 32 K):
    A: 4096 f16 ÷ 256 threads = 16 f16/thread → 2× BufferCopy128b
    B: 4096 f16 ÷ 256 threads = 16 f16/thread → 2× BufferCopy128b

Expected win: 2× MFMA throughput per iter + half as many k-iterations
⇒ near-2× speedup where MFMA issue rate is the bottleneck.
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
_MFMA_K = 32
_FRAG_A = 8     # per-lane A f16 count for K=32 MFMA
_FRAG_B = 8
_FRAG_C = 4
_WAVE_ROWS = 4
_WAVE_COLS = 4

_A_TILE_BYTES = 128 * 32 * 2
_B_TILE_BYTES = 32 * 128 * 2


def _align(x, a):
    return (x + a - 1) & ~(a - 1)


def _build_gemm_128x128_k32(*, M, N, K, dtype_str, arch):
    assert M % 128 == 0 and N % 128 == 0 and K % _MFMA_K == 0
    assert dtype_str in ("f16", "bf16")

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_128x128_k32_{dtype_str}_{M}_{N}_{K}_smem",
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
            SmemPtr(base_ptr, a_off0, in_elem_type, shape=(128, 32)),
            SmemPtr(base_ptr, a_off1, in_elem_type, shape=(128, 32)),
        ]
        s_B = [
            SmemPtr(base_ptr, b_off0, in_elem_type, shape=(32, 128)),
            SmemPtr(base_ptr, b_off1, in_elem_type, shape=(32, 128)),
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

        m_base_wg = bid_m * fx.Int32(128)
        n_base_wg = bid_n * fx.Int32(128)

        # K=32 tile → 4 vec8 chunks per row. 256 threads × 2 chunks = 512 slots.
        # Thread tid → (row = tid // 2, k_vec8_half = tid % 2 → covers 2 of 4 vec8s).
        # Each thread does 2 buffer_loads serially (halves 0 and 1 of its row half).
        a_load_row = tid // fx.Int32(2)
        a_load_half = tid % fx.Int32(2)        # 0 → k=0..15, 1 → k=16..31
        b_load_k = tid // fx.Int32(16)          # 32 K rows → tid 0..255 covers 16 rows twice
        b_load_n_vec8 = tid % fx.Int32(16)

        def _a_row_in_tile(r):
            return wave_row * fx.Int32(64) + fx.Int32(r * 16) + lane_row

        def _b_col_in_tile(c):
            return wave_col * fx.Int32(64) + fx.Int32(c * 16) + lane_row

        acc_ty = T.vec(_FRAG_C, T.f32)
        vec8_a_ty = T.vec(_FRAG_A, in_elem_type)

        def _zero_acc():
            zs = []
            for _ in range_constexpr(_FRAG_C):
                zs.append(arith.constant(0.0, type=T.f32))
            return vector.from_elements(acc_ty, zs)

        accs = [[_zero_acc() for _ in range_constexpr(_WAVE_COLS)]
                for _ in range_constexpr(_WAVE_ROWS)]

        def _stage_load(k_tile_scalar, stage):
            # k_tile_scalar is the k-tile index (step of 32). K-byte base:
            k_base = fx.Int32(k_tile_scalar * _MFMA_K)

            # A: each thread loads two 8-element chunks (k=half*16 and half*16+8)
            a_hbm_row = m_base_wg + a_load_row
            a_row_slice = fx.slice(A_buf, (a_hbm_row, None))
            a_div_v = fx.logical_divide(a_row_slice, h8_lay)
            # HBM vec8 index: K-row 'r' of global A, vec8 chunk = k_base/8 + half*2 + local
            a_hbm_vec_base = fx.Int32(k_tile_scalar * 4) + a_load_half * fx.Int32(2)
            for local in range_constexpr(2):
                vidx = a_hbm_vec_base + fx.Int32(local)
                a_chunk = _load_h8(a_div_v, vidx)
                for i in range_constexpr(8):
                    s_A[stage].store(
                        a_chunk[i].ir_value(),
                        [_idx(a_load_row),
                         _idx(a_load_half * fx.Int32(16) + fx.Int32(local * 8) + fx.Int32(i))],
                    )

            # B: 256 threads × 16 rows × 16 cols/row = 4096; but tile is 32 × 128 = 4096.
            # Use 16 threads per K row: each thread handles 8 f16 in N direction.
            # 32 K rows × 16 threads per K row = 512 slots; with 256 threads each handles 2 K rows.
            # Thread tid → (k_row = tid // 16, n_vec8 = tid % 16)  serves k_row in 0..15
            # Also (k_row_2 = tid // 16 + 16) for the second half.
            for k_half in range_constexpr(2):
                b_hbm_k = k_base + b_load_k + fx.Int32(k_half * 16)
                b_k_slice = fx.slice(B_buf, (b_hbm_k, None))
                b_div_v = fx.logical_divide(b_k_slice, h8_lay)
                b_hbm_vec_idx = bid_n * fx.Int32(128 // 8) + b_load_n_vec8
                b_chunk = _load_h8(b_div_v, b_hbm_vec_idx)
                for i in range_constexpr(8):
                    s_B[stage].store(
                        b_chunk[i].ir_value(),
                        [_idx(b_load_k + fx.Int32(k_half * 16)),
                         _idx(b_load_n_vec8 * fx.Int32(8) + fx.Int32(i))],
                    )

        def _consume(stage, accs):
            # Per-lane K=32 MFMA fragment: 8 f16. Layout:
            #   lane_row → M-row within 16-row tile
            #   lane_k_group → K-group (0..3), covers k = lg*8 .. lg*8+7
            # A[r]: load 8 contig f16 at (a_row_l, lane_k_group*8 + 0..7).
            a_frags = []
            for r in range_constexpr(_WAVE_ROWS):
                a_vals = []
                a_row_l = _a_row_in_tile(r)
                k_base = lane_k_group * fx.Int32(_FRAG_A)
                for i in range_constexpr(_FRAG_A):
                    a_vals.append(s_A[stage].load([_idx(a_row_l), _idx(k_base + fx.Int32(i))]))
                a_frags.append(vector.from_elements(vec8_a_ty, a_vals))

            b_frags = []
            for c in range_constexpr(_WAVE_COLS):
                b_vals = []
                b_col_l = _b_col_in_tile(c)
                k_base = lane_k_group * fx.Int32(_FRAG_B)
                for i in range_constexpr(_FRAG_B):
                    b_vals.append(s_B[stage].load([_idx(k_base + fx.Int32(i)), _idx(b_col_l)]))
                b_frags.append(vector.from_elements(T.vec(_FRAG_B, in_elem_type), b_vals))

            def _mfma(a, b, acc):
                if dtype_str == "bf16":
                    # The K=32 bf16 MFMA takes native bf16 vectors (unlike
                    # the K=16 bf16_1k twin which takes i16).
                    return fx.rocdl.mfma_f32_16x16x32_bf16(
                        acc_ty, [a, b, acc, 0, 0, 0],
                    )
                return fx.rocdl.mfma_f32_16x16x32_f16(acc_ty, [a, b, acc, 0, 0, 0])

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

        wave_m_base = m_base_wg + wave_row * fx.Int32(64)
        wave_n_base = n_base_wg + wave_col * fx.Int32(64)

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
            grid=(M // 128, N // 128, 1), block=(256, 1, 1), stream=stream,
        )

    return launch


_kernel_cache: dict = {}
_DTYPE2STR = {torch.float16: "f16", torch.bfloat16: "bf16"}


def _compile(M, N, K, dtype_str, arch):
    key = (M, N, K, dtype_str, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_128x128_k32(M=M, N=N, K=K, dtype_str=dtype_str, arch=arch)
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_gemm_128x128_k32_out",
    mutates_args=("out",),
    schema="(Tensor a, Tensor b, Tensor(a0!) out) -> ()",
)
def _gemm_128x128_k32_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    assert a.is_cuda and b.is_cuda and out.is_cuda
    assert a.dtype in (torch.float16, torch.bfloat16) and a.dtype == b.dtype
    assert out.dtype == torch.float32
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and M % 128 == 0 and N % 128 == 0 and K % 32 == 0
    assert out.shape == (M, N)
    _compile(M, N, K, _DTYPE2STR[a.dtype], get_rocm_arch())(a, b, out)


@_gemm_128x128_k32_out.register_fake
def _gemm_128x128_k32_out_fake(a, b, out):
    return None


def gemm_128x128_k32(A: Tensor, B: Tensor) -> Tensor:
    """128×128-tile K=32 MFMA GEMM.

    Requires M, N multiples of 128; K multiple of 32.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == B.dtype and A.dtype in (torch.float16, torch.bfloat16)
    M, K = A.shape
    _, N = B.shape
    out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    _gemm_128x128_k32_out(A, B, out)
    return out


__all__ = ["gemm_128x128_k32"]
