# Copyright (c) 2026, AMD.

"""In-kernel varlen-K grouped GEMM for gfx950 — one launch, runtime K-loop.

Computes, for each group ``g``, ``out[g] = A[:, s_g:e_g] @ B[s_g:e_g, :]`` where
``[s_g, e_g)`` are consecutive ``cu_seqlens_k`` offsets. Unlike the host-side
chunked path in ``gemm.py`` (``L`` separate launches + ``O(L)`` boundary syncs),
this is a **single** kernel whose grid ``z`` dimension selects the group: each
workgroup reads its own ``(s_g, e_g)`` from ``cu_seqlens_k`` at runtime and
contracts only that group's K-range via a **runtime-bounded** ``scf.for`` over
``ceil((e_g - s_g) / 16)`` MFMA K-tiles — so total work is ``O(total_K)``, not
``O(L * total_K)``, and there are no per-group host syncs.

Ragged group boundaries need no K alignment: the partial final K-tile is masked
per element (``k_global < e_g ? value : 0``), and zero contributions don't
affect the MFMA sum. Only ``M`` and ``N`` must be multiples of 16 (output tiling).

Scope (MVP): f16/bf16 inputs → f32 output, no epilogue. Grid =
``(M/16, N/16, L)``, block = (64, 1, 1).
"""

import functools

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator

from quack.amd.flydsl_utils import get_rocm_arch
from quack.amd.varlen_utils import validate_varlen_k

_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 16
_FRAG = 4


@functools.lru_cache(maxsize=64)
def _compile_varlen_k(dtype_str: str, N: int, M: int):
    arch = get_rocm_arch()
    assert dtype_str in ("f16", "bf16")
    allocator = SmemAllocator(
        None, arch=arch, global_sym_name=f"quack_amd_varlen_k_{dtype_str}_{M}_{N}_smem",
    )

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel(A: fx.Tensor, B: fx.Tensor, CU: fx.Tensor, OUT: fx.Tensor):
        in_ty = T.f16 if dtype_str == "f16" else T.bf16
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y
        group = fx.block_idx.z
        tid = fx.thread_idx.x
        lane_row = tid % fx.Int32(16)
        lane_g = tid // fx.Int32(16)

        A_buf = fx.rocdl.make_buffer_tensor(A)     # (M, total_K)
        B_buf = fx.rocdl.make_buffer_tensor(B)     # (total_K, N)
        CU_buf = fx.rocdl.make_buffer_tensor(CU)   # (1, L+1) i32
        O_buf = fx.rocdl.make_buffer_tensor(OUT)   # (L*M, N) f32

        ca_h = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), in_ty)
        ca_i = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.i32)
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        h_ty = fx.MemRefType.get(in_ty, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        i_ty = fx.MemRefType.get(T.i32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        f_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        lay = fx.make_layout(1, 1)

        def _load_h(div, idx):
            r = fx.memref_alloca(h_ty, lay)
            fx.copy_atom_call(ca_h, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _load_i(div, idx):
            r = fx.memref_alloca(i_ty, lay)
            fx.copy_atom_call(ca_i, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _store_f(div, idx, val):
            from flydsl.expr.vector import full as _vf
            r = fx.memref_alloca(f_ty, lay)
            fx.memref_store_vec(_vf(1, Float32(val), Float32), r)
            fx.copy_atom_call(ca_f, r, fx.slice(div, (None, idx)))

        # Per-group K-range [s, e) from cu_seqlens_k (runtime).
        cu_div = fx.logical_divide(fx.slice(CU_buf, (0, None)), lay)
        s = _load_i(cu_div, group)
        e = _load_i(cu_div, group + fx.Int32(1))
        k_len = arith.subi(e, s)
        # ceil(k_len / 16) K-tiles — the runtime loop bound.
        k_tiles = arith.divui(arith.addi(k_len, arith.constant(15, type=T.i32)),
                              arith.constant(16, type=T.i32))

        m_base = bid_m * fx.Int32(_MFMA_M)
        n_base = bid_n * fx.Int32(_MFMA_N)
        a_row = m_base + lane_row
        b_col = n_base + lane_row
        row_a = fx.slice(A_buf, (a_row, None))
        a_div = fx.logical_divide(row_a, lay)

        acc_ty = T.vec(_FRAG, T.f32)
        acc0 = vector.from_elements(
            acc_ty, [arith.constant(0.0, type=T.f32) for _ in range_constexpr(_FRAG)]
        )
        zero_h = arith.constant(0.0, type=in_ty)

        # Runtime-bounded K-loop, accumulator carried through scf.for.
        for kt, acc_state in range(0, k_tiles, init=[acc0]):
            acc_in = acc_state[0]
            k_tile_base = s + ArithValue(kt).index_cast(T.i32) * fx.Int32(_MFMA_K)
            lane_k_base = k_tile_base + lane_g * fx.Int32(_FRAG)

            a_vals = []
            b_vals = []
            for i in range_constexpr(_FRAG):
                k_global = lane_k_base + fx.Int32(i)
                valid = arith.cmpi(arith.CmpIPredicate.slt, k_global, e)
                av = _load_h(a_div, k_global)
                a_vals.append(arith.select(valid, av, zero_h))
                row_b = fx.slice(B_buf, (k_global, None))
                b_div = fx.logical_divide(row_b, lay)
                bv = _load_h(b_div, b_col)
                b_vals.append(arith.select(valid, bv, zero_h))
            a_frag = vector.from_elements(T.vec(_FRAG, in_ty), a_vals)
            b_frag = vector.from_elements(T.vec(_FRAG, in_ty), b_vals)

            if fx.const_expr(dtype_str == "bf16"):
                a_i16 = vector.bitcast(T.vec(_FRAG, T.i16), a_frag)
                b_i16 = vector.bitcast(T.vec(_FRAG, T.i16), b_frag)
                acc_new = fx.rocdl.mfma_f32_16x16x16bf16_1k(
                    acc_ty, [a_i16, b_i16, acc_in, 0, 0, 0],
                )
            else:
                acc_new = fx.rocdl.mfma_f32_16x16x16f16(
                    acc_ty, [a_frag, b_frag, acc_in, 0, 0, 0],
                )
            results = yield [acc_new]

        # Final accumulator after the runtime loop (bare value for length-1
        # loop-carry; the scf.for returns init acc0 for a zero-length group).
        acc = results

        # Store acc → OUT[group*M + out_row, out_col].
        for i in range_constexpr(_FRAG):
            out_row = group * fx.Int32(M) + m_base + lane_g * fx.Int32(_FRAG) + fx.Int32(i)
            out_col = n_base + lane_row
            row_o = fx.slice(O_buf, (out_row, None))
            o_div = fx.logical_divide(row_o, lay)
            val = vector.extract(acc, static_position=[i], dynamic_position=[])
            _store_f(o_div, out_col, val)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, CU: fx.Tensor, OUT: fx.Tensor,
               L: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        kernel(A, B, CU, OUT).launch(
            grid=(M // _MFMA_M, N // _MFMA_N, L), block=(64, 1, 1), stream=stream,
        )

    return launch


def gemm_varlen_k_inkernel(A: Tensor, B: Tensor, cu_seqlens_k: Tensor) -> Tensor:
    """In-kernel varlen-K grouped GEMM: ``out[g] = A[:, s_g:e_g] @ B[s_g:e_g, :]``.

    ``A`` is ``(M, total_K)``, ``B`` is ``(total_K, N)`` (f16/bf16, both
    last-dim contiguous). Output is ``(L, M, N)`` f32. One launch over
    ``grid.z = L`` groups; each contracts only its K-range (runtime loop).
    Constraints: M % 16, N % 16. Groups may be ragged (any ``K_g``, incl. 0).
    """
    assert A.is_cuda and B.is_cuda and A.dim() == 2 and B.dim() == 2
    assert A.dtype == B.dtype and A.dtype in (torch.float16, torch.bfloat16)
    assert A.stride(-1) == 1 and B.stride(-1) == 1
    M, total_K = A.shape
    total_K_b, N = B.shape
    assert total_K == total_K_b
    assert M % 16 == 0 and N % 16 == 0, "in-kernel varlen-K needs M%16, N%16"
    L = validate_varlen_k(cu_seqlens_k, total_K)
    cu = cu_seqlens_k.to(torch.int32).reshape(1, -1).contiguous()
    out = torch.empty(L * M, N, device=A.device, dtype=torch.float32)
    dtype_str = "f16" if A.dtype == torch.float16 else "bf16"
    _compile_varlen_k(dtype_str, N, M)(A.contiguous(), B.contiguous(), cu, out, L)
    return out.reshape(L, M, N)


__all__ = ["gemm_varlen_k_inkernel"]
