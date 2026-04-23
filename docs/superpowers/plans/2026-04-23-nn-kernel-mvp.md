# Phase 3 — NN GEMM kernel MVP — Implementation Plan

**Goal:** Land a working `gemm_nn(A, B)` kernel for `C[M,N] = A[M,K] @ B[K,N]`
with both operands row-major (K-inner for A, N-inner for B). bf16 + f16 day 1.
Plain matmul only (no fused epilogue in MVP; Phase 6 handles those). Tier-1
perf (0.5-0.8× hipBLASLt throughput) for landing.

**Architecture:** Direct port of `gemm_gfx950_splitk.py`'s NT kernel with
the B-side load rewritten for (K, N) row-major storage. A-side load
stays identical (both NT and NN have K-inner A). B goes HBM → LDS →
register (vs NT's B going HBM → register direct), with XOR swizzle on
the LDS to avoid bank conflicts. Uses `gemm_gfx950_mfma_core` for
shared helpers.

**Tech Stack:** Same as splitk — FlyDSL on gfx950, `@flyc.kernel`,
`@flyc.jit`, `rocdl.mfma_f32_16x16x16{f16,bf16_1k}` MFMA atoms,
`rocdl.raw_ptr_buffer_load_lds` for async A-side DMA.

**Parent spec:** `docs/superpowers/specs/2026-04-23-nn-tn-kernels-design.md` §2.2.
**Profile data:** `docs/superpowers/specs/hipblaslt_layout_profile.md`.

---

## File Structure

- **Create** `quack/amd/gemm_gfx950_nn.py` — NN kernel + public launcher
  + torch.library.custom_op wrapper. ~600-800 LoC target.
- **Create** `tests/amd/test_gemm_nn.py` — correctness + shape-matrix test.

## Target shapes for MVP (from the profile memo)

Must pass correctness on:
- Small: `M=2048, K=4096, N=1024` (MLP-dx, bs=2048, h=1024)
- Medium: `M=8192, K=16384, N=4096` (MLP-dx, bs=8192, h=4096)
- Plus a few smaller correctness-oracle shapes (M=128, N=64, K=128)

Must hit Tier-1 (≥ 0.5× of hipBLASLt throughput from the profile table)
on medium shape.

---

## Task 1 — Kernel scaffold (imports + constants + config)

Start from `gemm_gfx950_splitk.py` as template. Copy the imports,
`_DTYPE2STR` mapping, `SPLIT_K_COUNTER_MAX_LEN` import, and the top
section through `_compile_hgemm_kernel`'s initial arg-handling. Cut
all the split-K / epilogue / dact / dgated / gated parameters (MVP
has none of these).

- [ ] **Step 1: Copy skeleton + signature**

Create `quack/amd/gemm_gfx950_nn.py` with imports matching splitk and:

```python
@functools.lru_cache(maxsize=1024)
def _compile_nn_kernel(
    dtype: str,
    m_hint: int,
    k: int,
    n: int,
    TILE_M: int = 128,
    TILE_N: int = 256,
    TILE_K: int = 64,
    BLOCK_M_WARPS: int = 1,
    BLOCK_N_WARPS: int = 4,
):
    ...
```

Only kwargs: tile sizes, block-warp config. No B_PRE_SHUFFLE (defer), no
SPLIT_K (defer), no epilogue flags.

Pick same defaults as splitk: `TILE_M=128, TILE_N=256, TILE_K=64, BLOCK_M_WARPS=1,
BLOCK_N_WARPS=4, BLOCK_THREADS=256, STAGES=2`. We'll retune during Tier-2.

- [ ] **Step 2: Import the shared helpers**

```python
from quack.amd.gemm_gfx950_mfma_core import (
    _OnlineScheduler,
    _WmmaHalfK16,
    _WmmaHalfK32,
    swizzle_xor16,
)
```

## Task 2 — Compute derived tile / fragment constants

- [ ] **Step 1: Copy from splitk, adapted for NN**

In `_compile_nn_kernel`, compute:

```python
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
BLOCK_K_LOOPS = k // TILE_K
WARP_K_STEPS = TILE_K // WARP_ATOM_K
BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE
WARP_M_STEPS = TILE_M // BLOCK_M_WARPS // WARP_ATOM_M
WARP_N_STEPS = TILE_N // BLOCK_N_WARPS // WARP_ATOM_N
WARP_M = WARP_M_STEPS * WARP_ATOM_M
WARP_N = WARP_N_STEPS * WARP_ATOM_N
BLOCK_M = BLOCK_M_WARPS * WARP_M
BLOCK_N = BLOCK_N_WARPS * WARP_N
BLOCK_MK_SIZE = BLOCK_M * TILE_K
BLOCK_NK_SIZE = BLOCK_N * TILE_K  # same math, for B sizing
BLOCK_MN_SIZE = BLOCK_M * BLOCK_N
LDG_A_X_THREADS = TILE_K // LDG_VEC_SIZE
LDG_B_X_THREADS = BLOCK_N // LDG_VEC_SIZE  # ← differs from NT: N-inner
LDG_C_X_THREADS = BLOCK_N // LDG_VEC_SIZE
BLOCK_VECS = LDG_VEC_SIZE * BLOCK_THREADS
LDG_REG_A_COUNT = BLOCK_MK_SIZE // BLOCK_VECS
LDG_REG_B_COUNT = BLOCK_NK_SIZE // BLOCK_VECS
assert BLOCK_MK_SIZE % BLOCK_VECS == 0
assert BLOCK_NK_SIZE % BLOCK_VECS == 0
```

Note the one NN-specific line (commented `← differs from NT`): `LDG_B_X_THREADS = BLOCK_N // LDG_VEC_SIZE` instead of `BLOCK_K // LDG_VEC_SIZE` in NT. This is the axis flip.

## Task 3 — SMem allocator (for A + B LDS staging)

Unlike NT (B goes direct HBM→register), NN needs B in LDS too. Allocate
two LDS regions:

- [ ] **Step 1: Allocate A + B staging**

```python
allocator = SmemAllocator(None, arch=GPU_ARCH,
    global_sym_name=f"nn_smem_{dtype}_{m_hint}_{k}_{n}")
smem_a_offset = allocator._align(allocator.ptr, 16)
AS_BYTES = STAGES * BLOCK_M * TILE_K * DTYPE_BYTES
allocator.ptr = smem_a_offset + AS_BYTES

smem_b_offset = allocator._align(allocator.ptr, 16)
BS_BYTES = STAGES * TILE_K * BLOCK_N * DTYPE_BYTES
allocator.ptr = smem_b_offset + BS_BYTES

# Writeback-time LDS region (reuses the A/B LDS space by alias — same trick as splitk).
smem_c_offset = smem_a_offset
```

## Task 4 — Kernel body: A-side load (port from splitk)

- [ ] **Step 1: Copy `ldg_a`, `sts_a`, `ldg_sts_a_async`, `lds_matrix_a` from splitk**

These are IDENTICAL in NN (A has K-inner, exactly like NT). Copy lines 481-556 of
the pre-refactor `gemm_gfx950_splitk.py` (or the post-refactor equivalents).

## Task 5 — Kernel body: B-side load (new for NN)

This is the heart of Phase 3. B is stored (K, N) row-major, N-inner. Load pattern:
- HBM → LDS: each thread loads LDG_VEC_SIZE N-contiguous f16s for one K row
- LDS write: swizzled XOR on the N-byte axis
- LDS → MFMA B-fragment: each lane reads 4 K-values at its N-column (across 4 K rows)

- [ ] **Step 1: Implement `ldg_b` (HBM → registers)**

```python
def ldg_b(k_offset):
    """Load BLOCK_N × TILE_K f16s from B[K_contract, N_new], N-inner."""
    vecs = []
    for i in range_constexpr(LDG_REG_B_COUNT):
        global_tid = BLOCK_THREADS * i + tid
        k_local_idx = global_tid // LDG_B_X_THREADS
        n_local_idx = global_tid % LDG_B_X_THREADS * LDG_VEC_SIZE
        row_idx = fx.Index(k_offset + k_local_idx)
        col_idx = n_offset + fx.Index(n_local_idx)
        # B[row_idx, col_idx] — row-major stride (N, 1)
        vec = B_.vec_load((row_idx, col_idx), LDG_VEC_SIZE)
        vecs.append(vec)
    return vecs
```

- [ ] **Step 2: Implement `sts_b` (registers → LDS with XOR swizzle)**

```python
def sts_b(vecs, lds_stage):
    for i in range_constexpr(LDG_REG_B_COUNT):
        global_tid = BLOCK_THREADS * i + tid
        k_local_idx = global_tid // LDG_B_X_THREADS
        n_local_idx = global_tid % LDG_B_X_THREADS * LDG_VEC_SIZE
        col_in_bytes = n_local_idx * DTYPE_BYTES
        col_in_bytes = swizzle_xor16(k_local_idx, col_in_bytes, n_blocks16)
        bs_.vec_store(
            (fx.Index(lds_stage), k_local_idx, col_in_bytes // DTYPE_BYTES),
            vecs[i], LDG_VEC_SIZE,
        )
```

Where `n_blocks16 = fx.Int32(BLOCK_N * DTYPE_BYTES // 16)`.

- [ ] **Step 3: Implement `lds_matrix_b` (LDS → MFMA B-fragment)**

For NN, each lane needs 4 K-values at its N-column. The N-column is
`warp_n_idx + w_tid % WMMA_N`. The 4 K-values start at
`warp_atom_k_idx + (w_tid // WMMA_N) * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K`.

```python
def lds_matrix_b(lds_stage):
    s = fx.Index(lds_stage)
    b_frags = [0] * (WARP_K_STEPS * WARP_N_STEPS)
    for kk in range_constexpr(WARP_K_STEPS):
        for jj in range_constexpr(WARP_N_STEPS):
            warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
            warp_atom_k_idx = kk * WARP_ATOM_K
            k_start = warp_atom_k_idx + ldmatrix_b_k_idx  # lane's 4 K-values start
            n_col = warp_atom_n_idx + ldmatrix_b_n_idx    # lane's N column
            col_in_bytes = n_col * DTYPE_BYTES
            col_in_bytes = swizzle_xor16(k_start, col_in_bytes, n_blocks16)
            # Read 4 or 8 K-values (stride = BLOCK_N in LDS) — this becomes the
            # B fragment for this warp-atom.
            # Need to load 4/8 scalar values at (k_start + i, n_col) for i in 0..FRAG-1
            # — that's a STRIDED read in LDS.
            # Implementation: issue WMMA_B_FRAG_VALUES*MFMA_PER_WARP_K scalar LDS
            # loads and pack them into a vec<frag, dtype>.
            values = []
            for i in range_constexpr(WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K):
                v = bs_.load((s, k_start + i, col_in_bytes // DTYPE_BYTES))
                values.append(v)
            vec = vector.from_elements(
                T.vec(WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K, dtype_), values,
            )
            b_frags[kk * WARP_N_STEPS + jj] = vec
    return b_frags
```

Where:
```python
ldmatrix_b_n_idx = w_tid % WMMA_N
ldmatrix_b_k_idx = w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
```

This is the trickiest part — correctness-verifiable by running gemm_nn on
a small shape and comparing to torch.matmul. Likely perf-bottleneck too
(strided LDS loads have worse throughput than vectorised).

**Note for tune-in**: if strided scalar LDS loads hurt perf, alternatives are:
1. Transpose B in LDS (write `(N, K)` layout to LDS, then read K-contiguous).
2. Use a different swizzle so 4 consecutive K-values land in one vectorised LDS read.
3. MVP ships with the scalar-strided version; we optimise in tune-in.

## Task 6 — Kernel body: hot loop + MFMA

- [ ] **Step 1: Port `block_mma_sync` (identical to splitk)**

Copy verbatim from splitk.

- [ ] **Step 2: Port hot_loop_scheduler**

Copy from splitk. Adjust `LDG_TOTAL = LDG_REG_A_COUNT_AS + LDG_REG_B_COUNT` since B no longer loads directly in the hot loop — it stages through LDS, so B-loads contribute LDG_REG_B_COUNT to the vmem budget.

- [ ] **Step 3: Port the scf.for iter_args hot loop**

Copy from splitk. iter_args: `[k_offset, stage, *c_frags, *a_frags, *b_frags]`.
Key difference: b_frags come from `lds_matrix_b(stage)` now (not `ldg_matrix_b(k_offset)`).

## Task 7 — Write-back (simple — non-splitk path only)

- [ ] **Step 1: Port C-store path**

Copy the non-split-K path from splitk (lines 724+ in post-refactor version).
No epilogue: just truncate the acc to dtype, LDS-store, then HBM-store via
vec_store. No bias, act, dact, dgated, or gate_type handling.

## Task 8 — `@flyc.jit` launcher + `torch.library.custom_op`

- [ ] **Step 1: Write the jit launcher**

```python
@flyc.jit
def launch_nn_kernel(
    C: fx.Tensor, A: fx.Tensor, B: fx.Tensor, m: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    allocator.finalized = False
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        allocator.finalize()
    bm = (m + BLOCK_M - 1) // BLOCK_M
    bn = n // BLOCK_N
    nn_kernel._func.__name__ = KERNEL_NAME
    launcher = nn_kernel(C, A, B, m)
    launcher.launch(grid=(bm, bn, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)
```

- [ ] **Step 2: torch.library.custom_op wrapper**

```python
@torch.library.custom_op("quack_amd::_gemm_nn_out", mutates_args=("out",),
    schema="(Tensor a, Tensor b, Tensor(a0!) out) -> ()")
def _gemm_nn_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"{a.shape} @ {b.shape} incompatible"
    dt = _DTYPE2STR[a.dtype]
    _compile_nn_kernel(dt, M, K, N)(out, a, b, M)

@_gemm_nn_out.register_fake
def _gemm_nn_out_fake(a, b, out):
    return None

def gemm_nn(a: Tensor, b: Tensor, out: Optional[Tensor] = None) -> Tensor:
    M, K = a.shape
    _, N = b.shape
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=a.dtype)
    _gemm_nn_out(a, b, out)
    return out
```

## Task 9 — Correctness tests

- [ ] **Step 1: Write test matrix**

Create `tests/amd/test_gemm_nn.py`:

```python
import pytest
import torch
from quack.amd.gemm_gfx950_nn import gemm_nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")


@pytest.mark.parametrize("M", [1024, 256, 128])  # M-descending per AGENTS.md
@pytest.mark.parametrize("N", [1024, 128])
@pytest.mark.parametrize("K", [256, 512])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_nn_matches_torch(M, N, K, dtype):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    B = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
    out = gemm_nn(A, B)
    ref = torch.matmul(A.float(), B.float()).to(dtype)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 0.05, f"NN err {err:.4f} > 0.05"
```

## Task 10 — Tier-1 perf check

- [ ] **Step 1: Run on the medium MLP-dx shape from the profile**

```
PYTHONPATH=. python -c "
import torch, time
from quack.amd.gemm_gfx950_nn import gemm_nn
M, K, N = 8192, 16384, 4096
A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16) * 0.1
B = torch.randn(K, N, device='cuda', dtype=torch.bfloat16) * 0.1
out = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)

# Warmup
for _ in range(5): gemm_nn(A, B, out)
torch.cuda.synchronize()

# Time
events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(30)]
for s, e in events:
    s.record(); gemm_nn(A, B, out); e.record()
torch.cuda.synchronize()
t = min(s.elapsed_time(e) * 1e-3 for s, e in events)
flops = 2 * M * N * K / 1e12
print(f'gemm_nn: {t*1e6:.1f} μs, {flops/t:.1f} TFLOP/s')

# hipBLASLt reference
for _ in range(5): torch.matmul(A, B, out=out)
torch.cuda.synchronize()
for s, e in events:
    s.record(); torch.matmul(A, B, out=out); e.record()
torch.cuda.synchronize()
t_ref = min(s.elapsed_time(e) * 1e-3 for s, e in events)
print(f'hipBLASLt: {t_ref*1e6:.1f} μs, {flops/t_ref:.1f} TFLOP/s')
print(f'ratio (our/hip throughput): {(t_ref/t):.3f}×')
"
```

Expected: our throughput ≥ 0.5× of hipBLASLt's 1675 TFLOP/s ≈ ≥ 838 TFLOP/s.
If below Tier-1, note the bottleneck and defer tuning to a follow-up commit.

## Task 11 — Commit

- [ ] **Step 1: Stage and commit**

```
git add quack/amd/gemm_gfx950_nn.py tests/amd/test_gemm_nn.py
git commit -m "[AMD] Phase 3 — NN GEMM kernel MVP (bf16+f16)..."
```
