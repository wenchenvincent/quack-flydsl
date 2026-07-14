# Copyright (c) 2026, AMD.

"""128×128 tile MFMA GEMM with **vectorised LDS ops** — EXPERIMENTAL,
neutral at large shapes.

Same structure as ``gemm_gfx950_128x128.py``. Two changes:

1. **Vectorised LDS store** (HBM→LDS): each thread's 8-element f16
   chunk lands in LDS via one ``vector.store(vec8xf16)`` instead of
   8 scalar ``memref.store`` ops. 8× fewer store instructions.

2. **Vectorised A-fragment LDS load**: each MFMA A-fragment is 4
   contiguous K-elements per lane, loaded via a single
   ``vector.load_op(vec4xf16)`` instead of four scalar loads.

B LDS reads stay scalar — B fragment is 4 non-contiguous elements
(stride = 128 f16 = 256 bytes between K rows), which would need a
transposed LDS layout to vectorise.

**MI355X f16 measured result (time per call, smaller = better):**

    shape       baseline    vds         vds/base
    1024²       59.3        58.1        1.02x (neutral)
    2048²       113.5       111.6       1.02x (neutral)
    4096²       368.5       376.8       0.98x (marginal regression)

Starting profile motivation (MI355X 4096² f16):
    baseline 128×128:  21M SQ_INSTS_LDS, 84M bank conflicts, 366 μs
    hipBLASLt 256×256:  3.2M SQ_INSTS_LDS, 65K bank conflicts, 112 μs

**Finding**: cutting LDS instruction count in half (store side 8×,
load side 4× on A) has **no wall-time effect** at 2048²+. LDS
throughput is not the bottleneck — the 3.3× hipBLASLt gap at 4096²
comes from somewhere else (MFMA issue rate, HBM→LDS overlap via
async DMA, or persistent/stream-K scheduling). Keeping this file as
a reference so future attempts don't re-derive the same null result.

**Not wired into the autotune dispatcher.**
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
_WAVE_ROWS = 4
_WAVE_COLS = 4

_A_TILE_BYTES = 128 * 16 * 2
_B_TILE_BYTES = 16 * 128 * 2


def _align(x, a):
    return (x + a - 1) & ~(a - 1)


def _build_gemm_128x128_vds(*, M, N, K, dtype_str, arch):
    assert M % 128 == 0 and N % 128 == 0 and K % _MFMA_K == 0
    assert dtype_str in ("f16", "bf16")

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_128x128_vds_{dtype_str}_{M}_{N}_{K}_smem",
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
            SmemPtr(base_ptr, a_off0, in_elem_type, shape=(128, 16)),
            SmemPtr(base_ptr, a_off1, in_elem_type, shape=(128, 16)),
        ]
        s_B = [
            SmemPtr(base_ptr, b_off0, in_elem_type, shape=(16, 128)),
            SmemPtr(base_ptr, b_off1, in_elem_type, shape=(16, 128)),
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

        a_load_row = tid // fx.Int32(2)
        a_load_k_vec8 = tid % fx.Int32(2)
        b_load_k = tid // fx.Int32(16)
        b_load_n_vec8 = tid % fx.Int32(16)

        def _a_row_in_tile(r):
            return wave_row * fx.Int32(64) + fx.Int32(r * 16) + lane_row

        def _b_col_in_tile(c):
            return wave_col * fx.Int32(64) + fx.Int32(c * 16) + lane_row

        acc_ty = T.vec(_FRAG_C, T.f32)
        vec4_a_ty = T.vec(_FRAG_A, in_elem_type)

        def _zero_acc():
            zs = []
            for _ in range_constexpr(_FRAG_C):
                zs.append(arith.constant(0.0, type=T.f32))
            return vector.from_elements(acc_ty, zs)

        accs = [[_zero_acc() for _ in range_constexpr(_WAVE_COLS)]
                for _ in range_constexpr(_WAVE_ROWS)]

        vec8_ty = T.vec(8, in_elem_type)
        a_lds_mem = s_A[0].get()
        a_lds_mem1 = s_A[1].get()
        b_lds_mem = s_B[0].get()
        b_lds_mem1 = s_B[1].get()
        a_lds = [a_lds_mem, a_lds_mem1]
        b_lds = [b_lds_mem, b_lds_mem1]

        def _as_ir_vec(v):
            return v.ir_value() if hasattr(v, "ir_value") else v

        def _stage_load(k_tile_scalar, stage):
            k_base = fx.Int32(k_tile_scalar * _MFMA_K)
            a_hbm_row = m_base_wg + a_load_row
            a_row_slice = fx.slice(A_buf, (a_hbm_row, None))
            a_div_v = fx.logical_divide(a_row_slice, h8_lay)
            a_hbm_vec_idx = fx.Int32(k_tile_scalar * 2) + a_load_k_vec8
            a_chunk = _load_h8(a_div_v, a_hbm_vec_idx)
            vector.store(
                _as_ir_vec(a_chunk), a_lds[stage],
                [_idx(a_load_row), _idx(a_load_k_vec8 * fx.Int32(8))],
            )
            b_hbm_k = k_base + b_load_k
            b_k_slice = fx.slice(B_buf, (b_hbm_k, None))
            b_div_v = fx.logical_divide(b_k_slice, h8_lay)
            b_hbm_vec_idx = bid_n * fx.Int32(128 // 8) + b_load_n_vec8
            b_chunk = _load_h8(b_div_v, b_hbm_vec_idx)
            vector.store(
                _as_ir_vec(b_chunk), b_lds[stage],
                [_idx(b_load_k), _idx(b_load_n_vec8 * fx.Int32(8))],
            )

        def _consume(stage, accs):
            # A: 4 row-fragments via single vec4 ds_read each.
            a_frags = []
            a_mem = s_A[stage].get()
            k_start = lane_k_group * fx.Int32(_FRAG_A)
            for r in range_constexpr(_WAVE_ROWS):
                a_row_l = _a_row_in_tile(r)
                v = vector.load_op(vec4_a_ty, a_mem, [_idx(a_row_l), _idx(k_start)])
                a_frags.append(v)

            # B: still 4 scalar loads per col-fragment (non-contiguous K).
            b_frags = []
            for c in range_constexpr(_WAVE_COLS):
                b_vals = []
                b_col_l = _b_col_in_tile(c)
                for i in range_constexpr(_FRAG_B):
                    k_in_lds = lane_k_group * fx.Int32(_FRAG_B) + fx.Int32(i)
                    b_vals.append(s_B[stage].load([_idx(k_in_lds), _idx(b_col_l)]))
                b_frags.append(vector.from_elements(T.vec(_FRAG_B, in_elem_type), b_vals))

            def _mfma(a, b, acc):
                if fx.const_expr(dtype_str == "bf16"):
                    a_i = vector.bitcast(T.vec(_FRAG_A, T.i16), a)
                    b_i = vector.bitcast(T.vec(_FRAG_B, T.i16), b)
                    return fx.rocdl.mfma_f32_16x16x16bf16_1k(
                        acc_ty, [a_i, b_i, acc, 0, 0, 0],
                    )
                return fx.rocdl.mfma_f32_16x16x16f16(acc_ty, [a, b, acc, 0, 0, 0])

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
    if fx.const_expr(got is None):
        got = _build_gemm_128x128_vds(M=M, N=N, K=K, dtype_str=dtype_str, arch=arch)
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_gemm_128x128_vds_out",
    mutates_args=("out",),
    schema="(Tensor a, Tensor b, Tensor(a0!) out) -> ()",
)
def _gemm_128x128_vds_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    assert a.is_cuda and b.is_cuda and out.is_cuda
    assert a.dtype in (torch.float16, torch.bfloat16) and a.dtype == b.dtype
    assert out.dtype == torch.float32
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and M % 128 == 0 and N % 128 == 0 and K % 16 == 0
    assert out.shape == (M, N)
    _compile(M, N, K, _DTYPE2STR[a.dtype], get_rocm_arch())(a, b, out)


@_gemm_128x128_vds_out.register_fake
def _gemm_128x128_vds_out_fake(a, b, out):
    return None


def gemm_128x128_vds(A: Tensor, B: Tensor) -> Tensor:
    """128×128-tile MFMA GEMM with vectorised A-fragment LDS reads.

    Requires M, N multiples of 128; K multiple of 16.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == B.dtype and A.dtype in (torch.float16, torch.bfloat16)
    M, K = A.shape
    _, N = B.shape
    out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    _gemm_128x128_vds_out(A, B, out)
    return out


__all__ = ["gemm_128x128_vds"]
