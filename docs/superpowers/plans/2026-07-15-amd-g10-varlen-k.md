# Variable-Length-K GEMM — Implementation Plan (G10)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Design-level; expand to TDD tasks before executing.

**Goal:** Give the AMD GEMM surface a `cu_seqlens_k` (varlen-K / grouped-GEMM-over-K) path with correct semantics matching NVIDIA's reference, starting with a host-side chunked MVP rather than an in-kernel ragged reduction.

**Architecture:** Varlen-K is a **grouped GEMM**, not a packed-reduction: for `L` groups sharing a common `M` (and a common `A`), each group `i` contracts `A[:, cu_seqlens_k[i]:cu_seqlens_k[i+1]]` against `B[cu_seqlens_k[i]:cu_seqlens_k[i+1], :]` into its **own** independent output slice `out[i]` — confirmed against NVIDIA's own reference loop (`quack/gemm_interface.py:553-563`). Because AMD's FlyDSL GEMM kernels compile `K` as a Python-level compile-time constant (`gemm_gfx950_nn.py:73` `k: int`, baking `BLOCK_K_LOOPS = k // BLOCK_K` into the unrolled `scf.for`), an in-kernel ragged-K loop would need the K trip count to become a *runtime* bound threaded through the LDS double-buffer pipeline — a kernel rewrite comparable in scope to the M-dynamic support already shipped, but harder (K drives the pipeline directly; M's dynamism today only gates HBM-read validity per lane). This plan therefore scopes a host-side chunked MVP: slice per group and dispatch each chunk through the existing `quack.amd.gemm.gemm()`, which already auto-routes aligned chunks to the FlyDSL fast path and misaligned ones to the torch/hipBLASLt fallback — for free, with zero new kernel code.

**Tech Stack:** `quack/amd/varlen_utils.py`, `quack/amd/gemm.py` (host-side `cu_seqlens_m` pattern to mirror, `:185-223`), NVIDIA reference `quack/gemm_interface.py:506-565` (varlen_k python path) and `quack/varlen_utils.py` (`VarlenManager.len_k`/`offset_batch_A`/`offset_batch_B`, `:85-89`, `:91-113`, `:156-175`), `tests/amd/test_varlen.py` (existing varlen-M tests to mirror structurally), `tests/test_linear_varlen_k.py` (NVIDIA reference tests, incl. zero-length groups).

---

## Background the engineer needs

- **Confirmed output semantics from the NVIDIA reference** (`quack/gemm_interface.py:553-563`):
  ```python
  else:  # cu_seqlens_k is not None
      L = cu_seqlens_k.shape[0] - 1
      for i in range(L):
          A_slice = (
              A[:, A_idx[cu_seqlens_k[i] : cu_seqlens_k[i + 1]]]
              if gather_A else A[:, cu_seqlens_k[i] : cu_seqlens_k[i + 1]]
          )
          torch.mm(A_slice, B[cu_seqlens_k[i] : cu_seqlens_k[i + 1], :], out=out[i])
  ```
  `A` is `(M, total_K)`, `B` is `(total_K, N)`, output is `(L, M, N)` — **not** `(total_M, N)` like varlen-M. Each group's `K_i` may differ, `M` and `N` are shared across all groups. `A_idx` (when present) gathers along `A`'s K axis (columns), not rows.
- **AMD's existing varlen-M host path to mirror the style of** (`quack/amd/gemm.py:185-223`, esp. the per-sample-bias branch at `:198-223`): validate cu_seqlens on host, do the ragged bookkeeping in torch, and recurse into the plain `gemm()` call for the actual matmul. Varlen-K should follow the same "validate → slice → dispatch to existing kernel infra → recompose" shape, just producing a stacked `(L,M,N)` output instead of a concatenated `(total_M,N)` one.
- **`quack/amd/varlen_utils.py` today only has M-side helpers** (`validate_varlen`, `seqlens_from_cu`, `row_to_sample_idx`, all keyed on `cu_seqlens_m`/`total_M`) — G10 needs `validate_varlen_k(cu_seqlens_k, total_K)`, essentially the same shape check against `A.shape[-1]` instead of `A.shape[0]`.
- **Per-chunk K alignment is the crux of the "free fast path" claim.** `quack.amd.gemm.gemm()`'s MFMA eligibility (`gemm.py:113-148`) requires `K % 16 == 0` among other things. Real `cu_seqlens_k` boundaries are data-dependent (e.g. MoE token counts) and will rarely land on 16-multiples, so **most groups will silently take the torch/hipBLASLt fallback inside `gemm()`** — that's correct and automatic, but the MVP should not be sold as delivering FlyDSL-kernel perf on ragged K; it delivers **correctness and API parity**, with speed opportunistic. This must be stated plainly in the plan and the code's docstring, not glossed over.
- **Zero-length groups are a real, tested case on NVIDIA** (`tests/test_linear_varlen_k.py:224`, `test_gemm_varlen_k_with_zero_lengths`). `K_i == 0` must produce `out[i] = 0` (or `beta*C[i]` if a `C` epilogue input is added later) without calling any matmul — `torch.mm` on a `(M,0) @ (0,N)` shape is well-defined (all zeros) in PyTorch, so this may already fall out for free from a plain per-chunk `gemm()` call, but must be tested explicitly since AMD's `gemm()`/kernel eligibility checks (`M % 16`, etc.) were never exercised with a zero K and could assert incorrectly on an empty slice.
- **`A_idx` gather is on the K axis (columns of A), not rows.** `A[:, A_idx[s:e]]` is a non-contiguous gather along `dim=1`; the result must be `.contiguous()` before it can feed `gemm()` (which requires `A.stride(-1) == 1`). This materialization cost is inherent and shared with NVIDIA's own approach (their TMA-gather path is a separate optimization, out of scope here too).
- **Run the relevant existing tests while investigating:** `pytest tests/amd/test_varlen.py -x` (current M-side varlen surface) and read `tests/test_linear_varlen_k.py` in full for the exact NVIDIA numerical contract, including `test_gemm_add_varlen_k` (beta/C accumulation semantics) if the MVP is scoped to include a `C`/`beta` epilogue.

---

## Phase 1 — Investigation

**Files touched:** none (read-only).

1. Read `tests/test_linear_varlen_k.py` end-to-end (all four tests: `test_gemm_varlen_k_tma_gather_matches_cpasync`, `test_gemm_varlen_k`, `test_gemm_varlen_k_with_zero_lengths`, `test_gemm_add_varlen_k`) to extract the exact numerical contract this AMD plan must match, including whether `alpha`/`C`/`beta` are in scope for parity or a later increment.
2. Confirm there is no existing partial varlen-K plumbing on the AMD side beyond the M-side `VarlenArgs` NamedTuple in `quack/amd/varlen_utils.py:28-36` (which only has `cu_seqlens_m`/`A_idx` fields — no `cu_seqlens_k` field at all today).
3. Decide MVP scope precisely: (a) plain matmul only (`alpha`/`bias`/`activation`/`C` all deferred, matching how `gemm_gfx950_nn.py`'s own MVP docstring at `:30-33` scoped out epilogues initially), or (b) include `alpha`/`C`/`beta` from day one since `gemm()` already supports them per-chunk at no extra cost (the loop just forwards kwargs). Recommend (b) — it's free once the loop exists, and `test_gemm_add_varlen_k` shows NVIDIA parity expects it.

**Done when:** the plan's execution log records the exact chosen scope (plain matmul + alpha/beta/C, or plain-only) and confirms the `(L,M,N)` output-per-group semantics against the NVIDIA test file, not just the interface docstring.

---

## Phase 2 — Design

**Option A — MVP (recommended): host-side chunked loop through the existing `gemm()` dispatcher.**

```python
def gemm_varlen_k(A, B, cu_seqlens_k, A_idx=None, alpha=1.0, beta=0.0, C=None,
                   out_dtype=None) -> Tensor:
    """C[i] = alpha * A[:, s_i:e_i] @ B[s_i:e_i, :] (+ beta * C[i]) for each group i."""
```
- Validate via new `validate_varlen_k(cu_seqlens_k, A.shape[-1])` (mirrors `validate_varlen`, checked against `A`'s K axis instead of `A`'s row count).
- `L = cu_seqlens_k.numel() - 1`; allocate `out = torch.empty(L, M, N, ...)`.
- Loop `i in range(L)` (a real Python loop with `.item()` boundary syncs — same cost profile as `validate_varlen`'s existing single sync, now `O(L)` syncs; flag this as a known scaling limit for large `L`, e.g. MoE with hundreds of experts, and note the in-kernel follow-on removes it):
  - `s, e = cu_seqlens_k[i].item(), cu_seqlens_k[i+1].item()`; `k_i = e - s`.
  - if `k_i == 0`: `out[i] = beta * C[i] if C is not None else 0` (zero-length shortcut, no kernel call).
  - else: `A_i = A[:, A_idx[s:e]].contiguous() if A_idx is not None else A[:, s:e]`; `B_i = B[s:e, :]`; `out[i] = gemm(A_i, B_i, alpha=alpha, beta=beta, C=C[i] if C is not None else None, out_dtype=out_dtype)`.
- Because this dispatches through `gemm()` unchanged, aligned chunks (`k_i % 16 == 0`, `M % 16 == 0`, `N % 16 == 0`) automatically get the FlyDSL MFMA fast path; everything else automatically gets the torch/hipBLASLt fallback. No new kernel code, no new eligibility logic to maintain.

**Option B — in-kernel ragged-K reduction (NOT built by this plan, roadmap only).** See below.

**Wiring into the unified `gemm()` API vs a standalone function:** NVIDIA exposes both `cu_seqlens_m` and `cu_seqlens_k` as mutually-exclusive kwargs on one `gemm()` (`quack/gemm_interface.py:150,176-177`, asserting `not (varlen_m and varlen_k)`). For API parity, prefer wiring `cu_seqlens_k` into `quack/amd/gemm.py`'s existing `gemm()` signature (mirroring the `cu_seqlens_m` branch already there at `gemm.py:188-190`) rather than a separate top-level function — but the *output shape* differs fundamentally between the two modes (`(total_M,N)` concatenated vs `(L,M,N)` stacked), so document that divergence prominently in `gemm()`'s docstring right next to the existing `cu_seqlens_m` documentation (`gemm.py:178-186`), and assert `cu_seqlens_m is None or cu_seqlens_k is None`.

---

## Phase 3 — Implementation

**Files:**
- Modify: `quack/amd/varlen_utils.py` — add `validate_varlen_k(cu_seqlens_k: Tensor, total_K: int) -> int` (returns `L`), reusing `seqlens_from_cu` (already generic over any cu_seqlens tensor, no K-specific change needed there). Add `cu_seqlens_k: Optional[Tensor]` field to `VarlenArgs`.
- Modify: `quack/amd/gemm.py` — add `cu_seqlens_k: Optional[Tensor] = None` param to `gemm()`; branch near the top (alongside the existing `cu_seqlens_m` branch at `:188-190`) into the Phase-2 Option-A loop when set; assert mutual exclusivity with `cu_seqlens_m`. Keep the existing `A_idx` param's semantics for the non-varlen-K case unchanged; document that under `cu_seqlens_k`, `A_idx` gathers along K (columns), not M (rows) — a real semantic split worth a loud docstring note since the same-named param means different things depending on which varlen mode is active (this mirrors NVIDIA's own `gemm_interface.py` docstring comments at `:141` / `:396`, which spell out all four `(varlen_m, varlen_k) × (gather_A, no-gather)` shape combinations inline — copy that comment style).
- Test: `tests/amd/test_varlen_k.py` (new) — mirror `tests/test_linear_varlen_k.py`'s four cases: basic varlen-K matmul, zero-length groups, `A_idx` gather-K, and `alpha`/`beta`/`C` accumulation (if in scope per Phase 1.3). Reference is the literal per-group `torch.mm` loop from `quack/gemm_interface.py:553-563`, executed directly in the test (not re-imported, to keep the test independent of the NVIDIA module).

**Done when:** `gemm(A, B, cu_seqlens_k=...)` matches the torch per-group reference loop to f16/bf16 tolerance across: aligned-K groups (exercises the FlyDSL fast path), misaligned-K groups (exercises the torch fallback), a zero-length group, and (if in scope) `A_idx` gather-K plus `alpha`/`beta`/`C`.

---

## Roadmap — in-kernel ragged-K reduction (Option B, separate follow-on)

Out of scope for this plan. **Trigger:** profiling a real grouped-GEMM workload (e.g. MoE-style routing) shows the host-side loop's per-group kernel-launch overhead or the near-universal torch fallback (since ragged K is rarely 16-aligned) is the bottleneck.

**Why this is a real kernel rewrite, not a small extension:** every existing AMD GEMM kernel (`gemm_gfx950_nn.py`, `_tn.py`, `_nt_pingpong.py`, `_splitk.py`) bakes `K` into `BLOCK_K_LOOPS = k // BLOCK_K` as a **Python compile-time constant** that drives the unrolled `scf.for` loop bound and the LDS double-buffer stage count. Supporting per-group ragged K in-kernel means either (a) recompiling a distinct kernel per unique `K_i` (impractical — `cu_seqlens_k` is data-dependent and only host-visible via a `.item()` sync, defeating caching), or (b) making the K-loop trip count a genuine *runtime* `scf.for` bound with boundary-masked tail-K handling inside the LDS pipeline stages — unlike M's existing dynamism (`m: fx.Int32` in `gemm_gfx950_nn.py:194`), which only gates per-lane HBM-read validity via `arith.select` and never touches the loop structure or LDS staging itself. Masking a *partial* final K-tile inside a double-buffered async-DMA pipeline (`ldg_sts_a_async`/`ldg_sts_b_async`) is the hard, novel part.

**Done criteria for that future plan:** an in-kernel grouped-K GEMM (likely `grid.z = L` or one persistent kernel iterating groups) matches this plan's Option-A torch reference at the same tolerance, and beats Option A's wall-clock on a documented representative grouped-GEMM shape (e.g. MoE expert count × hidden dim).

---

## Self-Review

**Spec coverage:** The prompt's two options ("in-kernel ragged-K handling" vs "host-side chunked fallback as the MVP") are both addressed — Option A is fully specified and buildable now; Option B is honestly scoped as a follow-on with the specific technical obstacle (runtime K-loop + LDS tail masking) named, not hand-waved. The "risk high" framing from the prompt is reflected in Option B's roadmap section rather than smuggled into the MVP's done-criteria. ✓

**Placeholder scan:** Phase 3's `gemm_varlen_k` behavior (including the zero-length shortcut and the `A_idx`-gathers-K-not-M semantic split) is fully specified with a concrete signature and branch logic — no `TODO`/hand-wave in the MVP path. The `.item()`-per-group sync cost is called out explicitly as a known scaling limit rather than hidden. ✓

**Semantic verification:** The `(L,M,N)` stacked-output semantics (as opposed to varlen-M's `(total_M,N)` concatenated semantics) is cross-checked against the literal NVIDIA reference loop at `quack/gemm_interface.py:553-563`, not inferred from the docstring alone — this is the detail most likely to be gotten wrong (conflating varlen-K with varlen-M's packing model), so it's stated three times in this plan (Architecture, Background, Phase 2) to make the divergence impossible to miss during implementation. ✓
