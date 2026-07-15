# TT-Layout GEMM — Implementation Plan (G9)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Design-level; expand to TDD tasks before executing.

**Goal:** Give the AMD GEMM family a correct `C = A.T @ B.T` ("TT") path, and decide honestly whether that should be a dedicated FlyDSL kernel or a thin dispatch onto existing/torch infrastructure.

**Architecture:** QuACK-AMD's three training GEMM layouts (NN = dx, TN = dw, NT = fwd) cover every shape the current AMD port's `linear`/`mlp` training loop actually needs. TT has no caller anywhere in `quack/amd/`. This plan (a) confirms that via a codebase check, (b) ships a correct, cheap `gemm_tt()` MVP that routes through hipBLASLt (which natively supports all four BLAS transpose combinations, no data movement needed) or, if that's proven not free on ROCm, through a `gemm_nn` composition + one transpose kernel, and (c) designs — but does not build — the dedicated-kernel option so a future plan can pick it up if a real TT consumer shows up.

**Tech Stack:** `quack/amd/gemm.py` (dispatch), `quack/amd/gemm_gfx950_nn.py`, `quack/amd/gemm_gfx950_tn.py`, `quack/amd/gemm_gfx950_nt_pingpong.py`, `quack/amd/gemm_gfx950_mfma_core.py` (fragment/loader reference), `tests/amd/test_gemm_tt.py` (new).

---

## Background the engineer needs

- **Layout naming convention in this codebase** (BLAS-style, first letter = A's transpose-ness, second = B's):
  - NN (`gemm_gfx950_nn.py`): `A(M,K)` K-inner, `B(K,N)` N-inner → `C = A @ B`. Training dx.
  - TN (`gemm_gfx950_tn.py`): `A(K,M)` M-inner, `B(K,N)` N-inner, contraction on axis 0 of both → `C = A.T @ B`. Training dw.
  - NT (`gemm_gfx950_nt_pingpong.py` / `gemm_gfx950_splitk.py`): `A(M,K)` K-inner, `B(N,K)` K-inner → `C = A @ B.T`. Training fwd (`y = x @ W.T`).
  - TT (this plan): `A(K,M)` M-inner, `B(N,K)` K-inner → `C = A.T @ B.T`.
- **No TT caller exists today.** `quack/amd/linear.py`, `quack/amd/linear_training.py`, and `quack/amd/mlp.py` (via `linear_training.py`) only ever construct NT/NN/TN calls — confirmed by the docstrings in `gemm_gfx950_nn.py:1-11` and `gemm_gfx950_tn.py:1-11`, which each explicitly tie their layout to one of the three training GEMMs. This is the load-bearing fact behind the YAGNI recommendation below; **Phase 1 re-verifies it** rather than assuming it.
- **TT's fragment-load pieces already exist, split across two kernels.** TT's A-side (`(K,M)`, M-inner, contraction outer) is exactly TN's A-load pattern (`gemm_gfx950_tn.py:264-353`, `ds_read_tr16_b64`-based LDS→register transpose). TT's B-side (`(N,K)`, K-inner) is exactly NT-pingpong's B-load pattern (`gemm_gfx950_nt_pingpong.py:346-420`, straightforward `swizzle_xor16` + vectorised `ds_read`, no transpose). A dedicated TT kernel would be a *recombination* of two already-verified loaders, not new fragment-layout math — this lowers the risk of Option C below relative to a from-scratch kernel, but it is still a full kernel (LDS budgeting, hot-loop scheduling, tail-M/N handling, autotune table) at roughly TN's ~700-line scope.
- **hipBLASLt / torch already handle transpose flags without materialization** for the *torch fallback* path — this is the standard BLAS3 contract (`op(A)`, `op(B)` are stride/leading-dimension flags, not copies). The existing AMD `gemm()` fallback (`quack/amd/gemm.py:246-247`, `out = alpha * (A @ B)`) already relies on this for every layout that misses the MFMA-eligible fast path. This needs a one-line profiling check (Phase 1) rather than blind trust, because `quack/amd/gemm.py:335`'s `gemm_symmetric` fallback deliberately calls `.contiguous()` on a transposed view before handing it to `gemm()` — suggesting some transposed-view path was found unsafe or slow at some point and is worth re-checking specifically for the `.t(), .t()` (both-transposed) combination.
- **Run the relevant existing tests while investigating:** `pytest tests/amd/test_gemm_tn.py -x -k "128"` and `pytest tests/amd/test_gemm_nn.py -x -k "128"` to refresh context on the TN/NN loader shapes before touching anything.

---

## Phase 1 — Investigation

**Files touched:** none (read-only).

1. Grep `quack/amd/` for any existing TT-shaped call (`A.T @ B.T`, `A.t() @ B.t()`, `trans_a`/`trans_b` kwargs, or a `gemm_tt` reference) to confirm there is truly zero caller today. Extend the search to `tests/amd/` and `docs/superpowers/` for any prior design discussion.
2. Benchmark (on a gfx950 box) `torch.mm(A.t(), B.t())` vs `torch.mm(A.t().contiguous(), B.t().contiguous())` for 2-3 representative training-scale shapes (e.g. `(4096,4096)x(4096,4096)`, `(8192,2048)x(2048,4096)`) in bf16, to settle whether PyTorch's ROCm matmul dispatch actually avoids materialization for the doubly-transposed case, or silently inserts a contiguous copy (as `torch.__internal__` sometimes does for BLAS argument normalization). This determines which of Option A/B below is the real MVP.
3. Check whether `torch.library.custom_op`-wrapped `gemm()` (used for the MFMA-eligible fast paths) interacts safely with non-contiguous transposed inputs, or whether `torch.compile`/fake-tensor tracing needs the fallback to always materialize. (Relevant because `gemm_symmetric`'s existing fallback at `quack/amd/gemm.py:335` already made the conservative choice — confirm whether that was a real bug or just caution.)

**Done when:** a short note (in the plan's execution log, not a new doc) states (a) confirmed zero in-repo TT caller, (b) whether `torch.mm(A.t(), B.t())` measurably avoids materialization on ROCm for at least one representative shape, (c) whether `gemm()`'s custom-op wrapper is safe to feed transposed views.

---

## Phase 2 — Design

Three options, in ascending cost/risk:

**Option A (torch-native, zero new kernel code).** `gemm_tt(A, B) = torch.mm(A.t(), B.t())`, or the `alpha`/`beta`/`bias`/`activation`-supporting `A.t() @ B.t()` composed through the existing `quack.amd.gemm.gemm()` fallback arithmetic (reusing its bias/activation/alpha/beta code at `gemm.py:246-265`, just feeding it transposed views instead of materialised tensors). Zero HBM copy if Phase 1.2 confirms hipBLASLt handles the transpose flags natively; otherwise this degrades to Option B's cost automatically (torch inserts whatever copy it needs) — either way it is *correct* and *simple*, which is what matters since there's no perf-critical consumer.

**Option B (compose through `gemm_nn`, output-side transpose).** If Phase 1.2 shows torch does materialize an input copy, compose `D = gemm_nn(B, A)` instead — `B` is `(N,K)` K-inner, matching `gemm_nn`'s A-operand shape directly; `A` is `(K,M)` M-inner, matching `gemm_nn`'s B-operand shape directly (`N=M` in `gemm_nn`'s terms). No input transpose needed at all. Then `C = D.T.contiguous()`, an `O(M*N)` transpose (output-sized, not input-sized — cheaper than transposing either `A`(`K*M`) or `B`(`N*K`) when `K` is large, which is the common case for wide-and-deep training shapes). This reuses a fully-tuned existing kernel and adds one small transpose kernel.

**Option C (dedicated TT FlyDSL kernel — NOT built by this plan).** Combine TN's A-loader (`gemm_gfx950_tn.py:264-353`) with NT-pingpong's B-loader (`gemm_gfx950_nt_pingpong.py:346-420`) into a new `gemm_gfx950_tt.py`, mirroring the ~700-line structure of `gemm_gfx950_tn.py` (LDS double-buffer, `_OnlineScheduler` hot loop, autotune candidate table, `torch.library.custom_op` registration). This is scoped here only as a **roadmap entry** — see below — because Phase 1.1 is expected to show no consumer justifies the engineering cost.

**Recommendation:** ship Option A (or its automatic Option-B fallback depending on Phase 1.2's finding) as the actual deliverable of this plan. Do not build Option C unless a later profiling task identifies a real TT-shaped hot path (e.g. a future attention or MoE-permutation kernel).

---

## Phase 3 — Implementation (Option A/B MVP)

**Files:**
- New: `quack/amd/gemm_gfx950_tt.py` — thin module, no FlyDSL kernel. Houses `gemm_tt(A, B, out=None, alpha=1.0, beta=0.0, C=None, bias=None, activation=None, out_dtype=None)`.
  - Asserts: `A.dim()==2 and B.dim()==2`, `A.shape[0] == B.shape[1]` (shared K), `A.dtype == B.dtype`.
  - Body: per Phase 1.2's finding, either `return gemm(A.t(), B.t(), alpha=alpha, beta=beta, C=C, bias=bias, activation=activation, out_dtype=out_dtype)` (Option A, reusing the existing `quack.amd.gemm.gemm()` epilogue arithmetic on transposed views) or the `gemm_nn(B, A)` + `.T.contiguous()` composition (Option B) with the epilogue (bias/act/alpha/beta/C) applied post-transpose via the same `gemm()` elementwise fallback code.
  - Docstring explicitly states this is a torch/composition dispatch, not a FlyDSL MFMA kernel, and why (cites Phase 1's finding).
- Modify: `quack/amd/gemm.py` — no change to the main `gemm()` dispatcher (TT is not wired as a `trans_a`/`trans_b` flag on the unified API; keep it a separate named function, matching how `gemm_symmetric` and `gemm_gated` are separate named functions rather than flags). Add `gemm_tt` to `quack/amd/gemm.py`'s re-export or leave it importable directly from `quack.amd.gemm_gfx950_tt` — pick whichever matches the package's current convention for "family" functions (check `quack/amd/__init__.py` for precedent before deciding).
- Test: `tests/amd/test_gemm_tt.py` — parametrize dtype (f16/bf16) and 3-4 shapes; reference is `torch.mm(A.float().t(), B.float().t())`; also one test exercising `bias`/`alpha`/`beta`/`C` together against a manual torch composition, to lock the epilogue plumbing.

**Done when:** `gemm_tt` matches the torch reference to f16/bf16 rounding tolerance across parametrized shapes/dtypes, the bias/alpha/beta/C epilogue test passes, and (if Option A) a one-line comment cites the Phase-1 benchmark number showing no extra copy, or (if Option B) the composition is documented as intentionally trading one `O(M*N)` transpose for avoiding a dedicated kernel.

---

## Roadmap — dedicated TT FlyDSL kernel (Option C, separate follow-on)

Out of scope for this plan. **Trigger:** a profiled workload shows `gemm_tt`'s torch/composition path is a measurable bottleneck (not just "TT exists in the API surface").

**Design sketch for that future plan:** `gemm_gfx950_tt.py`, structured like `gemm_gfx950_tn.py` (128×256×64 default tile, 1×4 warp WG) with:
- A-loader = TN's `ldg_a`/`sts_a`/`ldg_sts_a_async`/`lds_matrix_a` (`gemm_gfx950_tn.py:266-353`), unchanged — TT's A has the identical `(K,M)` M-inner shape.
- B-loader = NT-pingpong's `ldg_sts_b_async`/`lds_matrix_b_kk` (`gemm_gfx950_nt_pingpong.py:346-420`) or the splitk kernel's symmetric K-inner B path — TT's B has the identical `(N,K)` K-inner shape, no `ds_read_tr16_b64` transpose needed on that side (unlike NN's B, which is N-inner and does need it — don't copy NN's B-loader by mistake).
- Everything else (MFMA inner loop, `_OnlineScheduler` hot-loop scheduling, XCD/GROUP_M swizzle, autotune candidate table, `torch.library.custom_op` registration) follows the TN kernel's structure directly.
- **Done criteria for that plan:** `gemm_tt` (FlyDSL-backed) matches the Phase-3 torch/composition reference to the same tolerance, and beats it in a documented benchmark at the shape that motivated building it.

---

## Self-Review

**Spec coverage:** The prompt asked to weigh dedicated-kernel vs transpose-into-existing-kernel and explicitly invited a YAGNI call. This plan makes that call explicit and evidence-gated (Phase 1's benchmark decides Option A vs B, not a guess), ships a real correct `gemm_tt()` either way, and scopes the dedicated kernel as an honest, buildable-later roadmap item with a concrete design (loader recombination from TN + NT-pingpong) rather than a vague TODO. ✓

**Placeholder scan:** Phase 3's deliverable is fully specified (function signature, assertions, two concrete body options, test file and its reference). The only deferred work is Option C, which is explicitly out-of-scope with its own trigger condition and design sketch, not a hidden gap in the MVP. ✓

**Type/shape consistency:** TT's `A(K,M)`/`B(N,K)` shapes are cross-checked against NN's `A(M,K)`/`B(K,N)` and TN's `A(K,M)`/`B(K,N)` conventions throughout (Background section) so Option B's `gemm_nn(B,A)` composition is shape-verified on paper: `gemm_nn` expects `A(M',K')` K'-inner and `B(K',N')` N'-inner; substituting `B(N,K)` (K-inner ✓) for `A'` and `A(K,M)` (M-inner ✓) for `B'` gives `M'=N, K'=K, N'=M`, output `(N,M) = D`, and `C = D.T` has shape `(M,N)` — matches the TT contract. ✓
