# NN + TN GEMM kernels for training backward — design

**Status**: design (brainstorming output; implementation plan to follow via `writing-plans`).
**Date**: 2026-04-23.
**Author**: Wen Chen (with brainstorm assist).
**Scope**: port-critical. Unblocks training-competitive perf for `quack.amd` — the current single-layout (NT) kernel leaves DX and DW on hipBLASLt fallbacks, and `quack/amd/linear.py` has no `torch.autograd.Function`, so even forward-fast workloads can't train with competitive step-times.

## 1. Problem statement

In ML training, a `nn.Linear` layer with `weight` shape `(out, in)` (`torch.nn.Linear` convention) generates three GEMM calls per step:

| Op  | Computation          | Row-major layout tag | Stride-1 stored axes                      |
|-----|----------------------|----------------------|--------------------------------------------|
| FWD | `y  = x @ W.T`       | **NT**               | `x[:, K]` inner; `W[:, K]` inner           |
| DX  | `dx = dy @ W`        | **NN**               | `dy[:, N]` inner; `W[:, K]` inner          |
| DW  | `dW = dy.T @ x`      | **TN**               | `dy[:, N]` inner (outer of A); `x[:, K]` inner |

"NT/NN/TN" names use row-major GEMM convention — `A[M,K] × B[K,N] → C[M,N]`, with each letter saying whether the *contraction axis* is the stride-1 axis of that operand. "T" ≡ contraction is inner (stride-1); "N" ≡ contraction is outer.

**Today** `quack/amd/gemm_gfx950_splitk.py` handles NT only (both operands have K stride-1, `B` is stored `(N, K)` row-major = `torch.nn.Linear.weight`). `quack/amd/linear.py` routes forward through this kernel but has no autograd plumbing — so backward in AMD falls back to whatever torch.autograd does with a `@torch.library.custom_op` that has no `register_autograd`, which in practice means broken training.

**Target**: land production-tier NN and TN kernels for bf16/f16, wire a `torch.autograd.Function` in `linear.py` that binds all three layouts, and land Tier-2 perf (0.9–1.2× hipBLASLt **throughput**, i.e. up to 10% slower to 20% faster) on training-representative shapes.

**Ratio convention used throughout this spec**: all perf ratios are expressed as **our-throughput / hipBLASLt-throughput** (TFLOP/s basis). Higher = better. 1.0× = tied. 0.5× means half hipBLASLt's throughput (twice its kernel time). The one exception is §8's "within 1.2× of baseline" phrasing for end-to-end step time, which is a **latency** ratio (lower = better) since step time is naturally a latency metric.

## 2. Approach (A.3 — shared hot-loop core + thin layout shells)

Originally considered:
- **A.1** — three standalone kernel files, each fully hand-written.
- **A.2** — parameterise the existing splitk kernel with a `layout: str` compile-time flag.
- **A.3** — refactor splitk's A/B loads to go through `fx.make_tiled_copy` with layouts expressed via `raked_product`/`composition`; extract a shared hot-loop core; add thin per-layout shells.

**Decision: A.3**, after confirming (via `python3 -c "import flydsl.expr.primitive"`) that FlyDSL has the full CuTe-equivalent layout algebra (`composition`, `raked_product`, `block_product`, `zipped_divide`, `logical_divide`, `flat_divide`, `flat_product`, `tiled_divide`, `coalesce`, `complement`, `tile_to_shape`, plus `TiledCopy`/`ThrCopy`/`TiledMma`/`ThrMma`). The splitk kernel currently hand-rolls offset math (`gemm_gfx950_splitk.py:481-508`, e.g. `global_tid // LDG_A_X_THREADS`) instead of using this algebra — converting to tiled-copy is a prereq for clean layout polymorphism and is a perf-neutral refactor on its own.

### 2.1 Module layout

```
quack/amd/
├── gemm_gfx950_splitk.py        (existing — NT; refactored to use shared core)
├── gemm_gfx950_nn.py            (new — DX layout)
├── gemm_gfx950_tn.py            (new — DW layout)
├── gemm_gfx950_mfma_core.py     (new — shared hot-loop + LDS pipeline + MFMA fragment assembly)
└── linear.py                    (extended — autograd Functions)
```

Each layout shell is ~300-500 LoC: defines the B tiled-copy layout (via `raked_product`), the LDS swizzle function, the MFMA fragment-register reshape, and the default tile config. All three instantiate the same `_gemm_mfma_core(...)` helper for the hot loop.

### 2.2 Per-layout access patterns

**NT (existing)** — `A[M, K] @ B[N, K]^T → C[M, N]`; both inner dims are K (contraction). MFMA fragments assemble from natural `buffer_load` chunks; preshuffle-B possible (current `shuffle_b` in `gemm_gfx950_splitk.py:909`).

**NN (DX)** — `A[M, N_contract] @ B[N_contract, K_out] → C[M, K_out]`; A has contraction inner, B has contraction outer. B access pattern: per K-tile, each `buffer_load` fetches `K_out`-contiguous chunk for one `N_contract` row → stride-`K_out` between rows. LDS staging: write `(N_tile, K_tile)` slab row-by-row; swizzle by XOR on the K_out axis so the MFMA's `(16_N × 16_K)` fragment read is bank-conflict-free. Preshuffle-W possible as a V2 since W is reused across fwd+bwd of the same step.

**TN (DW)** — `A^T[N_out, M_batch] @ B[M_batch, K_in] → C[N_out, K_in]`; here A is dy viewed transposed. dy is stored `(M_batch, N_out)` row-major — dy.T has M_batch *outer* (stride-N_out), and B (x) has M_batch outer (stride-K_in). Both operands have contraction outer. Solution: buffer_load multi-row chunks into LDS (each `buffer_load` reads an N_tile-wide or K_tile-wide chunk of one M-row; iterate over M_tile rows into LDS); MFMA reads M-inner fragments from LDS with xor-swizzle. Split-K *over M_batch* is load-bearing: small-batch DW (M_batch ∈ {128, 512}) requires SPLIT_K=4 or 8 to keep enough workgroups in flight.

### 2.3 Shared hot-loop core

`_gemm_mfma_core(...)` takes:
- `load_a: TiledCopy` — thread-layout × value-layout binding for A's HBM → reg or HBM → LDS path
- `load_b: TiledCopy` — same for B
- `lds_swizzle_a, lds_swizzle_b: Callable` — XOR swizzle for bank-conflict avoidance (may be identity for NT preshuffled path)
- `mfma: TiledMma` — 16×16×16 f16/bf16 MFMA instruction wrapper
- `tile_config: GemmConfig` — tile_m, tile_n, tile_k, lds_stages, block_warps
- `epilogue: Callable` — write-back (default: cast to out_dtype + store via `C_.vec_store`; extensible to bias/act/dact/dgated in Phase 5)

The hot loop, split_k_barrier, and LDS pipeline live once in `_mfma_core`. Each layout shell configures the arguments and calls it.

## 3. Validation plan

### 3.1 Profiling commit (first, no kernel code)

`tests/amd/bench_hipblaslt_layouts.py`:
- Shapes spanning three regimes, all f16 and bf16:
  - **MLP-fwd (NT sanity)**: `(M=bs, K=hidden, N=4*hidden)` for `(bs, hidden) ∈ {(2048, 1024), (8192, 4096), (32768, 8192)}`
  - **MLP-dx (NN)**: `(M=bs, K=4*hidden, N=hidden)` same (bs, hidden) tuples
  - **MLP-dw (TN)**: `(M=4*hidden, K=bs, N=hidden)` and `(M=hidden, K=bs, N=4*hidden)` same tuples
- Runs each shape under `rocprofv3 --pmc SQ_INSTS_MFMA,SQ_INSTS_VALU,SQ_INSTS_VMEM,SQ_INSTS_LDS,TCP_UTCL1_REQUEST,LDS_BANK_CONFLICT_READ,LDS_BANK_CONFLICT_WRITE,GRBM_GUI_ACTIVE`
- Deliverable: `docs/superpowers/specs/hipblaslt_layout_profile.md` with per-shape/per-layout (kernel wall-time, TFLOP/s, MFMA/VALU ratio, VMEM per MFMA, LDS bank-conflict rate) and a short analysis memo of what tricks hipBLASLt uses per layout.

The profile sets **Tier-2 per-shape perf targets** (0.9–1.2× of each shape's hipBLASLt time). No arbitrary picking of targets.

### 3.2 Correctness tests per kernel

- `tests/amd/test_gemm_nn.py` — all shapes above + square (1024, 2048, 4096, 8192). Reference: `torch.matmul(A.float(), B.float()).to(out_dtype)`. Tolerance bands: f32 @ 5e-6, f16/bf16 @ 1e-2. M-descending parametrise (FlyDSL JIT grid-bake quirk, per `tests/amd/conftest.py`).
- `tests/amd/test_gemm_tn.py` — same matrix, plus explicit small-batch (M_batch ∈ {128, 512}) to exercise split-K epilogue correctness.
- `tests/amd/test_linear_train.py` — new. Validates `LinearFunc(x, W).sum().backward()` against a torch-baseline `nn.functional.linear` via `torch.autograd.gradcheck` (fp32) and finite-difference (bf16/f16 with slack).

### 3.3 End-to-end training-step bench

`tests/amd/bench_mlp_train.py`:
- `MLP(hidden, 4*hidden)` and `GatedMLP(hidden, 8/3*hidden)` variants, bs×seq ∈ {2048, 8192}
- Measures full `fwd + bwd` step time vs a torch-native baseline MLP (bf16)
- Ships with each kernel landing, regression-tracked

## 4. Deliverable sequence

1. **Profiling commit** — `tests/amd/bench_hipblaslt_layouts.py` + `docs/.../hipblaslt_layout_profile.md`. No kernel code. Sets perf targets.
2. **Tiled-copy refactor commit** — `gemm_gfx950_splitk.py` A/B loads rewritten via `fx.make_tiled_copy` + `raked_product`. Zero perf regression gate. Extracts `_gemm_mfma_core`.
3. **NN kernel MVP** — `gemm_gfx950_nn.py` + `test_gemm_nn.py`. Tier-1 perf (0.5–0.8× hipBLASLt). bf16 + f16.
4. **NN kernel Tier-2 tune-in** — HW-counter-guided iteration: LDS swizzle constants, tile config, lds_stages, B preshuffle experiment. Ship when Tier-2 hit on representative shapes.
5. **TN kernel MVP** — `gemm_gfx950_tn.py` + `test_gemm_tn.py`. Split-K over M_batch **lands day 1** (not deferred). Tier-1 perf.
6. **TN kernel Tier-2 tune-in** — same methodology.
7. **Autograd Function commit** — `quack/amd/linear.py` with `LinearFunc`, `LinearActFunc`, `DActLinearFunc`; `quack/amd/mlp.py` gains `mlp_func`, `MLPRecomputeFunc`, `MLP(nn.Module)`. Binds NT/NN/TN to fwd/DX/DW.
8. **Training-step bench commit** — `tests/amd/bench_mlp_train.py`. End-to-end measurement.
9. **Phase 5 — fused epilogues** (the three gaps from the prior brainstorm, now kernel-specific tasks):
   - NN-bias / NN-act for DX write-back (unlocks `linear(x, W, act="silu")` backward fast path)
   - TN split-K>1 epilogue path with last-partial-signal (the prior gap #3, now land on TN since that's where split-K matters most)
   - NN-dgated for gated-MLP backward (the prior gap #2)
   - NN-dact / wire `DActLinearFunc` to route through fused `gemm_dact_fused` (the prior gap #1)

Each commit is independently testable. Regressions gated by `pytest tests/amd/` green + the bench harness perf delta.

## 5. Design decisions (locked)

| Decision                     | Choice                                                                |
|------------------------------|------------------------------------------------------------------------|
| Module structure             | A.3 — shared hot-loop core, three layout shells                       |
| First layout                 | NN (structurally closer to NT, validates LDS-swizzle approach)        |
| dtype                        | bf16 + f16 day 1 on both kernels                                       |
| MVP scope per kernel         | Plain matmul (no epilogue); epilogues in Phase 5                      |
| Tile-size autotune           | Fixed hand-tuned MVP config; autotune during Tier-2 tune-in           |
| Stream-K on NN/TN            | Static persistent grid MVP; stream-K only if profile warrants         |
| Split-K on TN                | **Day 1** (load-bearing for small-batch DW)                            |
| FP8                          | Deferred (Phase 6+)                                                    |
| Perf ship gates              | Tier-1 (0.5–0.8× hipBLASLt) for correctness landing; Tier-2 (0.9–1.2×) for tune-in completion |

## 6. Risks

- **Tiled-copy refactor introducing perf regression on existing NT kernel.** Mitigation: step 2's commit must ship with bench comparison against the pre-refactor NT kernel showing zero regression at 4096³ and 8192³.
- **TN kernel needing architecturally different tile config** (e.g., tile_m×tile_n=64×64 for output because the output is small, K_contract=batch is huge). Mitigation: day-1 split-K lets us run more workgroups even at tiny output tiles.
- **Autograd Function edge cases** — `fuse_grad_accum`, `recompute`, `concat_layout` (from the NVIDIA `mlp.py` options menu). Mitigation: MVP ports only the minimal surface (LinearFunc without fuse_grad_accum/concat); advanced options deferred to Phase 5.
- **hipBLASLt Tier-2 target may be unreachable on some shapes.** Mitigation: the profile commit's data determines per-shape targets honestly; we don't commit to a uniform target that's impossible for our kernel. Small-batch DW (M_batch < 512) is the most likely unreachable regime — document and accept.

## 7. Out of scope (for this spec)

- RDNA4 (gfx1201) and gfx1250 ports of NN/TN. Would require WMMA-equivalent fragment assembly; deferred to the existing `gemm_rdna_wmma.py` track.
- Variable-length-sequence GEMM (`cu_seqlens_m` path). Deferred — not blocking for dense training.
- Blockscaled fp8 NN/TN. Orthogonal; blockscaled kernel has its own file (`gemm_gfx950_blockscaled.py`). Extending it for NN/TN follows the same playbook but is deferred past Phase 6.

## 8. Definition of done

- All three layouts (NT, NN, TN) ship with bf16 and f16 at Tier-2 perf on the 9 shapes from §3.1 (bs × hidden × MLP-ratio product).
- `from quack.amd import linear; y = linear(x, W); y.sum().backward()` gives the same gradients as `torch.nn.functional.linear(x, W)` within tolerance.
- `from quack.amd import MLP; m = MLP(h, 4*h); y = m(x); y.sum().backward()` trains a synthetic task and matches `torch.nn.MLP`-equivalent loss curve.
- `tests/amd/bench_mlp_train.py` reports bf16 full training step time within 1.2× of a hipBLASLt-backed baseline.
