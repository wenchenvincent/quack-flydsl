# Copyright (c) 2026, AMD.

"""Standard-dtype MFMA GEMM for gfx950 — proof-of-life minimal kernel.

**Scope of this file today:** a minimum-viable FlyDSL MFMA kernel that
does ``C = A @ B`` for f16 inputs on a single-tile-per-workgroup grid
with a 16x16x16 MFMA. No LDS, no ping-pong, no preshuffle, no epilogue,
no split-K, no stream-K. Each workgroup is a single wave (64 threads)
that covers a 16x16 output tile by iterating K in 16-element chunks.

Why so minimal: writing a production MFMA GEMM end-to-end is ~1500 LoC
and weeks of tuning. This file proves the FlyDSL MFMA authoring path
works end-to-end against a PyTorch reference — the shared GEMM infra
and this kernel together form the foundation the full port sits on.

Layout reference (CDNA3/CDNA4 wave64 ``v_mfma_f32_16x16x16f16``):
    - A fragment (f16x4 per lane): ``A[lane % 16, (lane // 16)*4 + 0..3]``
    - B fragment (f16x4 per lane): ``B[(lane // 16)*4 + 0..3, lane % 16]``
    - C fragment (f32x4 per lane): ``C[(lane // 16)*4 + 0..3, lane % 16]``

Extensions (subsequent commits):
    - Larger tiles (128×128 via 4 MFMA waves per workgroup).
    - LDS ping-pong prefetch.
    - Epilogue (bias / activation / CShuffle).
    - Stream-K scheduling via ``quack/amd/tile_scheduler.py``.
    - bf16 / f32 dtype paths.
"""

from typing import Optional

import torch
from torch import Tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, range_constexpr, vector, math as _fm
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import Float32, Numeric
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl._mlir import ir

from quack.amd.flydsl_utils import get_rocm_arch


# For f16 MFMA: v_mfma_f32_16x16x16f16
_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 16
_FRAG_A = 4  # f16 values per lane
_FRAG_B = 4
_FRAG_C = 4  # f32 values per lane


def _build_gemm_16x16(
    *, M, N, K, dtype_str, out_dtype_str, arch,
    has_bias: bool = False,
    activation: Optional[str] = None,
    has_alpha: bool = False,
    has_c: bool = False,
):
    """Compile a 16x16-tile MFMA GEMM.

    ``dtype_str`` ∈ {"f16", "bf16"} selects the MFMA variant and input
    element type. ``out_dtype_str`` ∈ {"f32", "f16", "bf16"} picks the
    output store dtype (downcast from the f32 accumulator). Optional
    ``has_bias`` adds a per-column f32 bias. Optional ``activation`` ∈
    {"relu", "relu_sq", "gelu_tanh_approx", "silu"} applies post-bias.

    ``has_alpha=True`` applies ``alpha * acc`` (alpha is an f32 runtime arg).
    ``has_c=True`` adds ``beta * C[row, col]`` (beta runtime; C is same-shape
    f32). Both flags are independent so ``alpha`` without a C or ``beta *
    C`` without alpha scaling are each a single specialisation.

    Requires M, N, K all multiples of 16. Grid = ``(M/16, N/16, 1)``,
    block = (64,1,1).
    """
    assert dtype_str in {"f16", "bf16"}
    assert out_dtype_str in {"f32", "f16", "bf16"}
    assert activation in {None, "relu", "relu_sq", "gelu_tanh_approx", "silu"}
    assert M % _MFMA_M == 0 and N % _MFMA_N == 0 and K % _MFMA_K == 0
    ab_suffix = f"_{'a' if has_alpha else 'na'}{'c' if has_c else 'nc'}"
    sym_suffix = f"_{'b' if has_bias else 'nb'}_{activation or 'none'}_out{out_dtype_str}{ab_suffix}"
    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"quack_amd_gemm_{dtype_str}{sym_suffix}_smem",
    )
    # No LDS usage in the MVP; SmemAllocator still needs finalize() to emit the
    # (empty) shared-memory symbol.

    @flyc.kernel
    def kernel(
        A: fx.Tensor, B: fx.Tensor,
        Bias: fx.Tensor, Cin: fx.Tensor,
        alpha: fx.Float32, beta: fx.Float32,
        C: fx.Tensor,
    ):
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y
        tid = fx.thread_idx.x  # 0..63

        lane_row = tid % fx.Int32(16)
        lane_k_group = tid // fx.Int32(16)  # 0..3
        # k_offset in the A/B fragment layout: each group of 16 lanes carries a
        # different K-slice within one MFMA call. For K=16, only group 0 touches
        # the matrix at all? No — all four groups carry complementary K values.

        # Buffer-backed tensors for raw buffer_load access.
        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)
        if has_bias:
            Bias_buf = fx.rocdl.make_buffer_tensor(Bias)
            bias_div = fx.logical_divide(Bias_buf, fx.make_layout(1, 1))
        if has_c:
            Cin_buf = fx.rocdl.make_buffer_tensor(Cin)

        # Output tile offset in (M, N).
        m_base = bid_m * fx.Int32(_MFMA_M)
        n_base = bid_n * fx.Int32(_MFMA_N)

        # Per-lane A row / B col inside this tile.
        a_row = m_base + lane_row
        b_col = n_base + lane_row  # same lane_row index serves B's col

        # Scalar-load helper wrappers. We fall back to the copy_atom_call
        # machinery used elsewhere — less optimal than buffer_load but
        # reliably correct.
        in_elem_type = T.f16 if dtype_str == "f16" else T.bf16
        out_elem_type = (
            T.f32 if out_dtype_str == "f32"
            else T.f16 if out_dtype_str == "f16"
            else T.bf16
        )
        out_bufcopy = fx.rocdl.BufferCopy32b() if out_dtype_str == "f32" else fx.rocdl.BufferCopy16b()
        ca_h4 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), in_elem_type)
        ca_h = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), in_elem_type)
        ca_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), T.f32)
        ca_out = fx.make_copy_atom(out_bufcopy, out_elem_type)
        h4_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(4, 1), fx.AddressSpace.Register)
        h_reg_ty = fx.MemRefType.get(in_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        f_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        out_reg_ty = fx.MemRefType.get(out_elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        h4_lay = fx.make_layout(4, 1)
        reg_lay = fx.make_layout(1, 1)

        def _load_h4(div_vec, vec_idx):
            """BufferCopy64b: 4 contiguous f16/bf16 in one HBM instruction."""
            r = fx.memref_alloca(h4_reg_ty, h4_lay)
            fx.copy_atom_call(ca_h4, fx.slice(div_vec, (None, vec_idx)), r)
            return fx.memref_load_vec(r)

        def _load_h_scalar(div, idx):
            r = fx.memref_alloca(h_reg_ty, reg_lay)
            fx.copy_atom_call(ca_h, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _store_f_scalar(div, idx, val):
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            ts = _vfull(1, Float32(val), Float32)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_f, r, fx.slice(div, (None, idx)))

        def _load_f_scalar(div, idx):
            r = fx.memref_alloca(f_reg_ty, reg_lay)
            fx.copy_atom_call(ca_f, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0].ir_value()

        def _store_out_scalar(div, idx, val_f32):
            """Store ``val_f32`` (f32) to ``div`` at ``idx``, downcasting to
            the output dtype if needed."""
            from flydsl.expr.vector import full as _vfull
            r = fx.memref_alloca(out_reg_ty, reg_lay)
            elem_py = Numeric.from_ir_type(out_reg_ty.element_type)
            if out_dtype_str == "f32":
                ts = _vfull(1, Float32(val_f32), Float32)
            else:
                val = ArithValue(val_f32).truncf(out_elem_type)
                ts = _vfull(1, elem_py(val), elem_py)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(ca_out, r, fx.slice(div, (None, idx)))

        # Accumulator: f32x4 per lane, zero-init.
        acc_ty = T.vec(_FRAG_C, T.f32)
        _zero_list = []
        for _ in range_constexpr(_FRAG_C):
            _zero_list.append(arith.constant(0.0, type=T.f32))
        acc = vector.from_elements(acc_ty, _zero_list)

        # Iterate over K tiles (each tile = 16 K elements → one MFMA call).
        # A fragment (f16x4 contiguous along K): vectorized BufferCopy64b.
        row_a = fx.slice(A_buf, (a_row, None))
        a_div_v = fx.logical_divide(row_a, h4_lay)

        k_tiles = K // _MFMA_K
        for k_tile in range_constexpr(k_tiles):
            k_tile_base = fx.Int32(k_tile * _MFMA_K)
            # This lane's K offset within the tile: group 0 → 0..3, group 1 → 4..7, ...
            lane_k_base = lane_k_group * fx.Int32(_FRAG_A) + k_tile_base
            # vec4-unit index for the A fragment.
            vec_idx_A = lane_k_group + fx.Int32(k_tile * 4)

            a_frag = _load_h4(a_div_v, vec_idx_A)

            # Load this lane's B fragment (f16x4): B[lane_k_base + 0..3, b_col]
            # B is row-major (K, N); we pick one row per k and the b_col-th column.
            b_vals = []
            for i in range_constexpr(_FRAG_B):
                row_b_k = fx.slice(B_buf, (lane_k_base + fx.Int32(i), None))
                b_div = fx.logical_divide(row_b_k, fx.make_layout(1, 1))
                b_vals.append(_load_h_scalar(b_div, b_col))
            b_frag = vector.from_elements(T.vec(_FRAG_B, in_elem_type), b_vals)

            # MFMA: acc_new = a_frag @ b_frag + acc. bf16 variant uses the
            # `_1k` instruction; inputs stay in native dtype (the instruction
            # takes i16-viewed operands internally, handled by FlyDSL's
            # rocdl wrapper).
            if dtype_str == "bf16":
                a_frag_i16 = vector.bitcast(T.vec(_FRAG_A, T.i16), a_frag)
                b_frag_i16 = vector.bitcast(T.vec(_FRAG_B, T.i16), b_frag)
                acc = fx.rocdl.mfma_f32_16x16x16bf16_1k(
                    acc_ty, [a_frag_i16, b_frag_i16, acc, 0, 0, 0],
                )
            else:
                acc = fx.rocdl.mfma_f32_16x16x16f16(
                    acc_ty, [a_frag, b_frag, acc, 0, 0, 0],
                )

        # Epilogue: optional alpha/beta*C + bias + activation.
        # Bias is per-column (N-dim); each lane loads bias[out_col] once
        # and adds it to all 4 of its accumulators (they share the same column).
        if has_bias:
            bias_val = ArithValue(_load_f_scalar(bias_div, n_base + lane_row))
        if has_alpha:
            alpha_av = ArithValue(alpha)
        if has_c:
            beta_av = ArithValue(beta)
        # Store C: 4 rows per lane at column `lane_row` in the output tile.
        # C[(bid_m*16) + (lane_k_group*4 + i), (bid_n*16) + lane_row] = acc[i]
        for i in range_constexpr(_FRAG_C):
            out_row = m_base + lane_k_group * fx.Int32(_FRAG_C) + fx.Int32(i)
            out_col = n_base + lane_row
            row_c = fx.slice(C_buf, (out_row, None))
            c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))
            val_i = vector.extract(acc, static_position=[i], dynamic_position=[])
            val = ArithValue(val_i)
            if has_alpha:
                val = val * alpha_av
            if has_c:
                row_cin = fx.slice(Cin_buf, (out_row, None))
                cin_div = fx.logical_divide(row_cin, fx.make_layout(1, 1))
                cin_val = ArithValue(_load_f_scalar(cin_div, out_col))
                val = val + beta_av * cin_val
            if has_bias:
                val = val + bias_val
            # Inlined activations — the module-level helpers use an arg shape
            # that doesn't always round-trip through the epilogue context;
            # open-coding keeps the IR clean.
            zero = arith.constant(0.0, type=T.f32)
            if activation == "relu":
                val = val.maximumf(zero)
            elif activation == "relu_sq":
                val = val.maximumf(zero) * val
            elif activation == "gelu_tanh_approx":
                import math as _py_math
                c1 = _py_math.sqrt(2.0 / _py_math.pi)
                c2 = 0.044715 * c1
                x = val
                x_sq = x * x
                tanh_arg = x * (c1 + c2 * x_sq)
                # tanh(z) = 1 - 2 / (1 + exp(2z))  (no libcall; uses hardware exp).
                tanh_z = Float32(1.0) - Float32(2.0) / (Float32(1.0) + _fm.exp(Float32(2.0) * tanh_arg, fastmath="fast"))
                val = x * (Float32(0.5) + Float32(0.5) * tanh_z)
            elif activation == "silu":
                # silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
                val = val / (Float32(1.0) + _fm.exp(-val, fastmath="fast"))
            _store_out_scalar(c_div, out_col, val)

    @flyc.jit
    def launch(
        A: fx.Tensor, B: fx.Tensor,
        Bias: fx.Tensor, Cin: fx.Tensor,
        alpha: fx.Float32, beta: fx.Float32,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(A, B, Bias, Cin, alpha, beta, C).launch(
            grid=(M // _MFMA_M, N // _MFMA_N, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    return launch


_kernel_cache: dict = {}


_OUT_DTYPE_MAP = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}


def _compile(M, N, K, dtype_str, out_dtype_str, has_bias, activation,
             has_alpha, has_c, arch):
    key = (M, N, K, dtype_str, out_dtype_str, has_bias, activation,
           has_alpha, has_c, arch)
    got = _kernel_cache.get(key)
    if got is None:
        got = _build_gemm_16x16(
            M=M, N=N, K=K, dtype_str=dtype_str, out_dtype_str=out_dtype_str, arch=arch,
            has_bias=has_bias, activation=activation,
            has_alpha=has_alpha, has_c=has_c,
        )
        _kernel_cache[key] = got
    return got


@torch.library.custom_op(
    "quack_amd::_gemm_mfma_out",
    mutates_args=("out",),
    schema=(
        "(Tensor A, Tensor B, Tensor? bias, Tensor? cin, "
        "float alpha, float beta, Tensor(a0!) out, str? activation) -> ()"
    ),
)
def _gemm_mfma_out(
    A: Tensor, B: Tensor, bias: Optional[Tensor], cin: Optional[Tensor],
    alpha: float, beta: float, out: Tensor,
    activation: Optional[str],
) -> None:
    assert A.dtype in (torch.float16, torch.bfloat16)
    assert A.dtype == B.dtype
    assert out.dtype in _OUT_DTYPE_MAP
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert out.shape == (M, N)
    assert M % 16 == 0 and N % 16 == 0 and K % 16 == 0
    assert all(t.stride(-1) == 1 for t in (A, B, out))
    has_bias = bias is not None
    if has_bias:
        assert bias.dim() == 1 and bias.size(0) == N and bias.dtype == torch.float32
    has_alpha = alpha != 1.0
    has_c = cin is not None
    if beta != 0.0 and cin is None:
        raise AssertionError("beta != 0 requires a C tensor")
    if cin is not None:
        assert cin.shape == (M, N) and cin.dtype == torch.float32
        assert cin.stride(-1) == 1
    dtype_str = "f16" if A.dtype == torch.float16 else "bf16"
    out_dtype_str = _OUT_DTYPE_MAP[out.dtype]
    launcher = _compile(
        M, N, K, dtype_str, out_dtype_str, has_bias, activation,
        has_alpha, has_c, get_rocm_arch(),
    )
    B_arg = bias if has_bias else out
    Cin_arg = cin if has_c else out  # stand-in when has_c=False
    launcher(A, B, B_arg, Cin_arg, alpha, beta, out)


@_gemm_mfma_out.register_fake
def _gemm_mfma_out_fake(A, B, bias, cin, alpha, beta, out, activation):
    return None


def gemm_mfma(
    A: Tensor, B: Tensor,
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
    out_dtype: Optional[torch.dtype] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: Optional[Tensor] = None,
) -> Tensor:
    """FlyDSL MFMA GEMM: f16/bf16 × f16/bf16 → f32/f16/bf16.

    Computes ``D = activation(alpha * (A @ B) + beta * C + bias)``.

    - M, N, K all multiples of 16.
    - ``bias``: 1-D f32 tensor of length N (per-column bias, added after
      alpha/beta).
    - ``activation``: None, "relu", "relu_sq", "gelu_tanh_approx", "silu".
    - ``out_dtype``: f32 (default), f16, bf16. Downcast from f32 accumulator.
    - ``alpha``, ``beta``, ``C``: full-GEMM terms. ``C`` must be f32 same
      shape as output. Setting any of these triggers a separate kernel
      specialisation (the fast path for alpha=1 beta=0 C=None skips the
      scalar * acc + scalar * C load).
    """
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    _, N = B.shape
    if bias is not None and bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    if C is not None and C.dtype != torch.float32:
        C = C.to(torch.float32)
    if out_dtype is None:
        out_dtype = torch.float32
    out = torch.empty(M, N, device=A.device, dtype=out_dtype)
    _gemm_mfma_out(A, B, bias, C, float(alpha), float(beta), out, activation)
    return out


def gemm_f16_mfma(A: Tensor, B: Tensor) -> Tensor:
    """Kept for backwards compatibility; delegates to ``gemm_mfma``."""
    return gemm_mfma(A, B)


__all__ = ["gemm_mfma", "gemm_f16_mfma", "_gemm_mfma_out"]
