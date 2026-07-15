# Richer Symmetric GEMM — Implementation Plan (G5)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Design-level; expand to TDD tasks before executing.

**Goal:** Give `quack.amd.gemm_symmetric` the bias/activation/alpha/beta/C epilogue that its sibling `gemm_gfx950.py` already has, and separately design (with a lower-confidence build recommendation) a triangular tile scheduler that skips the mirrored half of the `(M,M)` output.

**Architecture:** `gemm_symmetric.py`'s dedicated kernel (`_build_symmetric_16x16`) and `gemm_gfx950.py`'s dedicated kernel (`_build_gemm_16x16`) share the same grid shape (`M/16 × N/16`, 64-thread single-wave-per-tile), the same per-lane fragment layout, and — per `gemm_symmetric.py:118`'s own comment ("same pattern as gemm_gfx950") — were written as siblings. `gemm_gfx950.py`'s epilogue (`has_bias`/`has_alpha`/`has_c` flags, `:231-260`) is therefore a near-direct port into `gemm_symmetric.py`'s kernel. The triangular scheduler is a separate, materially harder feature: skipping tile `(j,i)` for `i<j` only saves the redundant MFMA pass if the *epilogue* for the mirrored side is derived from the already-computed accumulator (matching NVIDIA's `TriangularTileScheduler` + `PostAct = D.mT` pattern in `quack/gemm_symmetric.py:317` and `quack/tile_scheduler.py:567`), which requires transposing a 16×16 MFMA accumulator fragment across lanes — a real correctness-sensitive piece of new work, scoped here but flagged as the plan's highest-risk item.

**Tech Stack:** `quack/amd/gemm_symmetric.py`, `quack/amd/gemm_gfx950.py` (epilogue pattern to port), `quack/amd/gemm.py:317-335` (`gemm_symmetric` dispatcher, whose current `not kw` eligibility gate this plan removes), NVIDIA reference `quack/gemm_symmetric.py:299-360` (full-featured `gemm_symmetric` call surface) and `quack/tile_scheduler.py:556-724` (`triangular_idx_to_coord` + `TriangularTileScheduler`), `tests/amd/test_gemm_symmetric.py`.

---

## Background the engineer needs

- **Today's gap has a hidden cost, not just a missing feature.** `quack/amd/gemm.py:317-335`'s `gemm_symmetric()` dispatcher only routes to the dedicated no-transpose kernel `when not kw` (i.e. zero epilogue kwargs); the moment a caller passes `bias=` or `alpha=`, it silently falls back to `gemm(A, A.transpose(-1,-2).contiguous(), ...)` — which materializes a full `(K,M)` transpose copy and defeats the entire point of the dedicated kernel (dual-read of `A` with no transpose, per `gemm_symmetric.py:5-12`'s own docstring). This plan's epilogue phase directly fixes that silent-fallback trap, which is the real user-facing bug behind "richer symmetric GEMM."
- **`gemm_gfx950.py`'s epilogue to port, concretely** (`gemm_gfx950.py:56-97` kernel signature, `:113-115` bias-buffer setup, `:231-260` epilogue application):
  - `has_bias: bool` → extra `Bias: fx.Tensor` kernel arg, one `_load_f_scalar` per lane per output column, added once (shared across a lane's 4 accumulator rows since they're the same column).
  - `has_alpha: bool` → `alpha: fx.Float32` runtime scalar arg, multiplied into each of the 4 per-lane accumulator values.
  - `has_c: bool` → extra `Cin: fx.Tensor` kernel arg + `beta: fx.Float32` runtime scalar, `val = val + beta * cin_val` read per output element (not per-column — `C` is full `(M,N)`, unlike bias).
  - Activation (`relu`/`relu_sq`/`silu`/`gelu_tanh_approx`) is inlined post-bias, post-alpha/beta (`gemm_gfx950.py:260+`, truncated in the excerpt above — read the full block before porting).
  - All four flags are independent `fx.const_expr` specializations (separate compiled kernel per flag combination, matching `_compile`'s cache key at `gemm_gfx950.py:312-321`) — `gemm_symmetric.py`'s existing `_kernel_cache` dict (`:172`) already keys on `(M, K, dtype_str, out_dtype_str, arch)`; extend that key tuple with the same flags.
- **`gemm_symmetric.py`'s current kernel structure to extend** (`_build_symmetric_16x16`, `:46-169`): takes `A: fx.Tensor, C: fx.Tensor` only; computes `acc` via one MFMA loop reading `A[a_row_i]` and `A[a_row_j]` (`:118-134`); writes `C[i,j] = acc` at `:147-154` via `_store_out`. The bias/alpha/beta/C epilogue slots in right before the final `_store_out` loop, structurally identical to where `gemm_gfx950.py` does it.
- **NVIDIA's richer surface as the parity target** (`quack/gemm_symmetric.py:299-360`): `gemm_symmetric(A, B, D, C, tile_count_semaphore, tile_M, tile_N, cluster_M, cluster_N, pingpong, persistent, ..., alpha, beta)` — note NVIDIA's version takes a *second* input `B` (used identically to `A` — the kernel is generic enough to compute `A @ B.T` symmetric-shaped output, with `B==A` for the true symmetric case) and a full persistent/pingpong scheduler stack. AMD's MVP scope should stay narrower: match the **epilogue** richness (bias/alpha/beta/C/activation) without adopting NVIDIA's `B`-as-separate-input generalization or its persistent-scheduler machinery, since neither was asked for here and both are large, separate features.
- **Triangular scheduler reference** (`quack/tile_scheduler.py:556-724`): `triangular_idx_to_coord(idx)` (`:556-564`) maps a linear triangular index to `(row, col)` via `row = ceil(sqrt(2*idx+2.25) - 0.5) - 1`; `TriangularTileScheduler._swizzle_cta` (`:687-724`) additionally groups tiles for L2 reuse — that grouping complexity is cluster-oriented (NVIDIA's `cluster_shape_mn`) and doesn't directly apply to AMD's single-CTA-per-tile grid; the **core index-mapping math** (`triangular_idx_to_coord`) is the reusable piece, the swizzle grouping is not.
- **The hard part NVIDIA's scheduler solves that a naive "just launch `M*(M+1)/2` tiles" doesn't:** simply skipping tile `(j,i)` for `i<j` in the grid and leaving `C[j,i]` unwritten is wrong — the output must still be fully populated (symmetric mirror). NVIDIA's kernel handles this by computing tile `(i,j)`'s accumulator once and writing it to **both** `D[i,j]` and `D[j,i]` via `PostAct = D.mT` in the epilogue arguments (`quack/gemm_symmetric.py:317`) — i.e. the same accumulator is written through two different output views, with `D.mT`'s stride pattern doing the index-swap for free at the memory-access level (NVIDIA's CuTe epilogue naturally handles arbitrary output strides). **AMD's kernel has no such free lunch**: `gemm_symmetric.py`'s `_store_out` writes scalar-at-a-time to a specific `(row,col)` computed from `(i_base, j_base, lane_row, lane_k_group)` (`:147-154`); writing the *mirror* position `C[j,i]` from the *same* accumulator requires either (a) a second explicit store loop with `(row,col)` swapped — trivial for the store-address math, since each lane already knows both its `(row,col)` and its transpose `(col,row)`, no cross-lane data movement needed at all (this is the key simplification AMD gets that a naive "transpose the accumulator across lanes" framing would miss) — **because the epilogue here is a scalar per-lane store, not a vectorized fragment write, the mirror store is just a second `_store_out` call with swapped indices, not a register-level transpose.** This significantly de-risks the AMD triangular scheduler relative to the framing in the task description.

---

## Phase 1 — Investigation

**Files touched:** none (read-only).

1. Read `gemm_gfx950.py`'s epilogue block in full (`:225-270`, the region truncated above) to capture the exact activation-inlining code before porting it — do not re-derive the activation formulas from `gemm_gfx950_mfma_core.py`'s `_apply_epilogue` (a different, vectorized-fragment epilogue helper used by the LDS-tiled kernels); `gemm_symmetric.py`'s scalar-per-lane store loop needs the *scalar* inlined form that `gemm_gfx950.py` already has, not the vector-register helper.
2. Confirm `gemm_symmetric.py`'s `_load_f_scalar`/`_store_out`/`ca_f`/`f_reg_ty` helper set (`:83-98` in `gemm_symmetric.py`) is structurally identical enough to `gemm_gfx950.py`'s (`:83-98` there too, matching line-for-line per the "same pattern" comment) that the port is close to copy-paste — flag any divergence found.
3. Decide the triangular scheduler's grid-launch shape: current kernel launches `grid=(M/16, M/16, 1)` (`gemm_symmetric.py:164`, full square). A triangular launch needs either (a) a 1D grid of size `M/16 * (M/16+1)/2` with `triangular_idx_to_coord` decoding `block_idx.x` in-kernel (matches NVIDIA's `TriangularTileScheduler` approach at the index-math level, minus the cluster swizzle), or (b) keep the 2D `(M/16, M/16)` grid and early-return (`if bid_j > bid_i: return`) — wasting launched-but-idle CTAs. Recommend (a) for real compute savings; (b) is simpler but only saves MFMA cycles, not launch/occupancy — note both, pick (a).

**Done when:** the plan's execution log records the confirmed epilogue-port scope (bias/alpha/beta/C/activation, no `B`-as-separate-input, no persistent scheduler) and the chosen triangular grid strategy (1D `triangular_idx_to_coord`-decoded vs 2D early-return).

---

## Phase 2 — Design: epilogue richness

**Signature change:**
```python
def gemm_symmetric(
    A: Tensor, out_dtype: Optional[torch.dtype] = None,
    bias: Optional[Tensor] = None, activation: Optional[str] = None,
    alpha: float = 1.0, beta: float = 0.0, C: Optional[Tensor] = None,
) -> Tensor:
```
- `bias`: `(M,)` f32, added per-column — note the *column* index for a symmetric `(M,M)` output is also an M-indexed vector, so `bias` has the same shape/semantics as `gemm_gfx950.py`'s `(N,)` bias with `N=M`.
- `C`: `(M,M)` f32, matching output shape, added as `beta * C[row,col]`.
- `activation`: same four-way set (`relu`/`relu_sq`/`silu`/`gelu_tanh_approx`), inlined identically to `gemm_gfx950.py`.
- `_build_symmetric_16x16` gains `has_bias`, `has_alpha`, `has_c`, `activation` params (mirroring `_build_gemm_16x16`'s signature at `gemm_gfx950.py:56-61`); kernel gains `Bias: fx.Tensor, Cin: fx.Tensor, alpha: fx.Float32, beta: fx.Float32` args; the epilogue block (`gemm_symmetric.py:147-154`) is extended with the ported bias/alpha/beta/activation application, applied identically for both the `(i,j)` write and (once Phase 3's triangular scheduler lands) the `(j,i)` mirror write — **important asymmetry to get right:** `C[i,j]`'s bias/beta*C term uses column `j`'s bias and `Cin[i,j]`; the mirrored `C[j,i]`'s term uses column `i`'s bias and `Cin[j,i]` — these are generally **different values** (bias and `C` are not assumed symmetric inputs), so the mirror store cannot just copy `C[i,j]`'s post-epilogue value — it must re-run the epilogue with swapped bias/C indices against the *same pre-epilogue accumulator*. This is why the epilogue-richness and triangular-scheduler phases are coupled at the epilogue-application code, not just at the store address — call this out loudly in the implementation task so whoever builds Phase 3 doesn't try to reuse a single post-epilogue scalar for both stores.
- `quack/amd/gemm.py:317-335`'s dispatcher: relax the `not kw` gate to accept the newly-supported kwargs (`bias`, `activation`, `alpha`, `beta`, `C`) as first-class dedicated-kernel params instead of falling back; keep falling back to `gemm(A, A.T.contiguous(), ...)` only for genuinely unsupported combinations (e.g. if a future caller passes something outside this set).

**Done when (design):** a concrete kernel-arg list and epilogue-code diff (not yet applied) is agreed, cross-checked against `gemm_gfx950.py`'s exact epilogue block from Phase 1.1.

## Phase 3 — Design: triangular tile scheduler

- 1D grid of `NUM_TRI_TILES = (M/16) * (M/16 + 1) / 2` (host-computed Python int, since `M` is compile-time here — `gemm_symmetric.py`'s kernel already closes over a static `M`, unlike the NN/TN kernels' dynamic-`M` support, so this is a compile-time grid size, not a runtime one).
- In-kernel: port `triangular_idx_to_coord` (`quack/tile_scheduler.py:556-564`) from CuTe-DSL to FlyDSL — it's pure integer arithmetic (`ceil`, `sqrt`, no CuTe-specific types), should translate directly using `flydsl.expr.math` equivalents for `sqrt`/`ceil`. Decode `block_idx.x` → `(bid_i, bid_j)` with `bid_i >= bid_j` (or the NVIDIA convention's `row >= col`, whichever direction `triangular_idx_to_coord` assumes — verify against `quack/tile_scheduler.py`'s docstring and, ideally, a small standalone Python sanity check before porting into the kernel).
- Kernel body: compute `acc` once for the `(bid_i, bid_j)` tile (existing MFMA loop, unchanged). Epilogue: run the (now bias/alpha/beta/act-aware) store twice — once at `(i_base, j_base)` using column-`j` bias/`C[i,j]`, once at `(j_base, i_base)` using column-`i` bias/`C[j,i]` — **skip the second store when `bid_i == bid_j`** (diagonal tile, self-mirror, would double-write the same address). Both stores read the same `acc` register; the second store's address math is `out_row = j_base + lane_k_group*4+kk, out_col = i_base + lane_row` — i.e. literally `i_base`/`j_base` swapped relative to the first store, no data-level transpose (per Background's key finding).

**Done when (design):** the two-store epilogue logic above is validated on paper against a small hand-worked `M=32` (2×2 tile grid, 3 triangular tiles) example — tile `(0,0)`, `(1,0)`, `(1,1)` — confirming the store-address swap produces the full symmetric `4x4`-tile-equivalent output with no gaps or double-writes.

---

## Phase 4 — Implementation

**Files:**
- Modify: `quack/amd/gemm_symmetric.py` — extend `_build_symmetric_16x16` per Phase 2's epilogue port and, if Phase 3's design review passes, Phase 3's triangular grid + dual-store epilogue. Extend `_kernel_cache` key tuple with the new flags. Extend `gemm_symmetric()`'s public signature per Phase 2.
- Modify: `quack/amd/gemm.py:317-335` — relax the dispatch eligibility gate per Phase 2.
- Test: `tests/amd/test_gemm_symmetric.py` — extend with parametrized bias/alpha/beta/C/activation combinations, reference `alpha*(A@A.T) + beta*C + bias` computed in f32 torch (matching the file's existing tolerance convention, `atol = max(5e-3, K*2e-5)`). If Phase 3 ships, add a dedicated test asserting the triangular-scheduled kernel's output is bit-identical (or within the same float tolerance) to the full-square MVP kernel's output on a shared random `A`/`bias`/`C`, across a couple of `M` values including one with a non-square-fitting triangular tile count.

**Sequencing recommendation:** land Phase 2 (epilogue richness) as its own commit/PR first — it is lower-risk, directly fixes the silent-fallback bug in Background, and is independently valuable. Land Phase 3 (triangular scheduler) as a second, separate commit — its dual-store epilogue coupling (flagged in Phase 2's design) means it should not be attempted until the epilogue code from Phase 2 is merged and tested standalone.

**Done when:** all extended `tests/amd/test_gemm_symmetric.py` cases pass; `gemm_symmetric(A, bias=..., alpha=..., beta=..., C=...)` no longer silently falls back to the `.contiguous()` transpose path (assert this directly in a test, e.g. by monkeypatching or by checking that the dedicated-kernel cache gains a new entry); if Phase 3 ships, its dedicated test passes and a rough MFMA-cycle-count sanity check (or wall-clock at a large `M`) shows the triangular grid does less compute than the square grid.

---

## Self-Review

**Spec coverage:** Both halves of the prompt — epilogue features (bias/act/alpha/beta/C) and the triangular-skip scheduler — are addressed with concrete file/line references and a design, not just a mention. The prompt's "Effort M" sizing is respected by explicitly sequencing epilogue-richness (the M-effort, do-now deliverable) ahead of the triangular scheduler (materially higher-risk, sequenced second with its own design-review gate before implementation). ✓

**Placeholder scan:** The epilogue port (Phase 2/4) is fully specified down to which existing code (`gemm_gfx950.py`'s scalar epilogue, not `gemm_gfx950_mfma_core.py`'s vector one) to port from, and why. The triangular scheduler's trickiest correctness point — that the mirror store needs a *second* epilogue application with swapped bias/C indices, not a copy of the first store's value — is derived explicitly rather than left implicit, specifically to prevent an implementer from taking the tempting-but-wrong shortcut of writing the same post-epilogue scalar to both `(i,j)` and `(j,i)`. ✓

**Correctness cross-check:** The plan explicitly walks through *why* AMD's scalar-per-lane store architecture avoids the register-level accumulator transpose that a cluster/vectorized-epilogue design (like NVIDIA's) would need — each lane already computes both `(row,col)` and its swap `(col,row)` from its own indices, so the "mirror" is a second store with swapped base offsets, not a cross-lane shuffle. This was verified against `gemm_symmetric.py`'s actual store-address computation (`:147-154`), not assumed by analogy to the NT-pingpong kernel's writeback-shuffle discussion. ✓
