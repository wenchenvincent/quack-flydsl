# AMD Phase 4 — Consolidate the Two Linear Autograd Stacks

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]`. This is a **refactor** — the safety net (Task 1) must land before any consolidation.

**Goal:** Collapse the two overlapping linear+activation autograd implementations into one source of truth, without changing any observable numerics or breaking either test suite.

**Architecture:** Two stacks exist today:
- **`quack/amd/linear.py`** — `LinearFunc` (`:266`), `LinearActFunc`, gated Funcs. Backward uses the **custom FlyDSL grad kernels**: dx via `gemm_nn` (NN), dw via `gemm_tn` (TN) (`linear.py:290-291`). Entry points: `linear_train`, `linear_act_train`; consumed by `mlp_train` in `quack/amd/mlp.py` (`mlp.py:18`).
- **`quack/amd/linear_training.py`** — `_LinearActFunction` (`:132`), `_LinearGatedFunction` (`:159`), `_MLPActFunction` (`:257`), `_GatedMLPTrainFunction` (`:429`). Backward uses **`torch.mm`** (`:151,153,190,192,...`) EXCEPT `_MLPActFunction`, which additionally has the **fused `gemm_splitk` dact** backward path (`:295-306`). Entry points: `mlp_func_train`, `gated_mlp_func_train`.

They are NOT pure duplicates: `linear.py` has custom dx/dw kernels; `linear_training.py` has the fused-dact backward and the `torch.mm` grads. Consolidation = one implementation that has both capabilities. **Target: keep `linear.py`'s custom-kernel stack as the base** (it's the real port direction — `torch.mm` is a placeholder), graft the fused-dact path onto it, redirect `linear_training.py`'s entry points to it, and delete the duplicate Functions.

**Tech Stack:** `quack/amd/linear.py`, `quack/amd/linear_training.py`, `quack/amd/mlp.py`; tests `tests/amd/test_linear.py`, `tests/amd/test_linear_train.py`.

**Depends on:** Phase 3 should land first (it settles whether the fused-dact path is enabled, which this phase must preserve).

---

## Task 1: Build the safety net — characterize both stacks under test

**Files:**
- Test: `tests/amd/test_linear_consolidation.py` (new — a behavior-lock harness, deleted or folded at the end)

- [ ] **Step 1: Enumerate the current public entry points and which Function each uses**

Read and write down (in the test file's module docstring) the mapping: `linear_train`/`linear_act_train` → `linear.py` Funcs (gemm_nn/gemm_tn backward); `mlp_train` → `linear.py` via `mlp.py`; `mlp_func_train`/`gated_mlp_func_train` → `linear_training.py` Funcs (torch.mm / fused-dact backward). This is the contract that must not change.

- [ ] **Step 2: Write a behavior-lock test comparing EVERY entry point to a torch reference**

Create `tests/amd/test_linear_consolidation.py` with one test per public entry point (`linear_train`, `linear_act_train`, `mlp_train`, `mlp_func_train`, `gated_mlp_func_train`) that runs fwd+bwd and compares out + all input grads against an explicit `torch.nn.functional` reference at a small eligible shape (bf16, M%128, dims %256/%64). Use loose bf16 tolerances (atol/rtol 5e-2). This captures current correct behavior so the refactor can't silently regress it. (Do NOT assert on which backend runs — only on numerics.)

- [ ] **Step 3: Run — establish the green baseline**

Run: `cd /workspace/quack && python -m pytest tests/amd/test_linear_consolidation.py tests/amd/test_linear.py tests/amd/test_linear_train.py -x`
Expected: all pass. If any entry point's reference doesn't match, fix the REFERENCE (match the real forward math) — this step is about capturing truth, not changing code.

- [ ] **Step 4: Commit the safety net**

```bash
git add tests/amd/test_linear_consolidation.py
git commit -m "[AMD] Behavior-lock harness for linear autograd consolidation"
```

## Task 2: Give `linear.py`'s stack the fused-dact capability

**Files:**
- Modify: `quack/amd/linear.py` (add optional fused-dact backward to its MLP/act Function, mirroring `_MLPActFunction`)
- Reference: `quack/amd/linear_training.py:257-316` (`_MLPActFunction`), `:219` (`_fused_dact_eligible`)

- [ ] **Step 1: Move `_fused_dact_eligible` + `_act_bwd` to a shared location**

These helpers (`linear_training.py:58,219`) are needed by both. Move them to `quack/amd/linear.py` (or a small shared module `quack/amd/_linear_common.py` imported by both) so there's one copy. Keep `linear_training.py` importing them from the new home during the transition.

- [ ] **Step 2: Ensure `linear.py`'s MLP/linear-act backward supports the fused path**

`linear.py` already does dx via `gemm_nn` and dw via `gemm_tn`. Add the fused-dact branch (`if _fused_dact_eligible(...): dpreact = gemm_splitk(dout, w2.t().contiguous(), preact=..., dact_activation=...)`) so it matches `_MLPActFunction`'s capability. Where `linear.py` currently lacks a two-layer MLP Function, add one (or extend `LinearActFunc`) so `mlp_func_train`'s behavior can be expressed by it.

- [ ] **Step 3: Run the safety net**

Run: `cd /workspace/quack && python -m pytest tests/amd/test_linear_consolidation.py tests/amd/test_linear.py tests/amd/test_linear_train.py -x`
Expected: still all green (this task only ADDS capability to `linear.py`; nothing is redirected yet).

- [ ] **Step 4: Commit**

```bash
git add quack/amd/linear.py quack/amd/linear_training.py
git commit -m "[AMD] linear.py stack gains fused-dact backward (shared helpers)"
```

## Task 3: Redirect `linear_training.py` entry points to the unified stack

**Files:**
- Modify: `quack/amd/linear_training.py` (`mlp_func_train`, `gated_mlp_func_train` now delegate to `linear.py`'s Functions)

- [ ] **Step 1: Point one entry point at the unified stack**

Rewrite `mlp_func_train` (`linear_training.py:318`) to call `linear.py`'s unified MLP Function instead of `_MLPActFunction.apply`. Keep the signature identical.

- [ ] **Step 2: Run the safety net — this is the critical gate**

Run: `cd /workspace/quack && python -m pytest tests/amd/test_linear_consolidation.py::test_mlp_func_train tests/amd/test_linear_train.py tests/amd/test_linear.py -x`
Expected: green. If numerics shifted beyond bf16 tolerance, the two stacks weren't equivalent — investigate the difference (likely `torch.mm` vs `gemm_nn`/`gemm_tn` accumulation) before proceeding. Do NOT loosen tolerance to hide a real divergence.

- [ ] **Step 3: Repeat for `gated_mlp_func_train`**

Same redirect for `gated_mlp_func_train` (`linear_training.py:503`) → `linear.py`'s gated Function. Run the safety net again.

- [ ] **Step 4: Commit**

```bash
git add quack/amd/linear_training.py
git commit -m "[AMD] Redirect mlp_func_train/gated_mlp_func_train to the unified linear stack"
```

## Task 4: Delete the duplicate Functions

**Files:**
- Modify: `quack/amd/linear_training.py` (remove now-unused `_LinearActFunction`, `_LinearGatedFunction`, `_MLPActFunction`, `_GatedMLPTrainFunction` if nothing else references them)

- [ ] **Step 1: Confirm the duplicates are unreferenced**

Run: `cd /workspace/quack && grep -rn "_LinearActFunction\|_LinearGatedFunction\|_MLPActFunction\|_GatedMLPTrainFunction" quack/ tests/`
Expected: only their own definitions remain (all call sites now route through `linear.py`). If a test references them directly, update the test to the public entry point first.

- [ ] **Step 2: Delete the dead Functions**

Remove the four now-dead `torch.autograd.Function` classes and any helpers left orphaned. Keep the public entry-point functions (`mlp_func_train`, `gated_mlp_func_train`) — they're the API, they just delegate now.

- [ ] **Step 3: Full regression**

Run: `cd /workspace/quack && python -m pytest tests/amd/test_linear.py tests/amd/test_linear_train.py tests/amd/test_linear_consolidation.py -x`
Expected: all green. Also `python -c "import quack.amd"` to confirm no dangling imports.

- [ ] **Step 4: Fold or delete the harness, then commit**

Either keep `test_linear_consolidation.py` as permanent regression coverage (recommended — it's a real cross-entry-point test) or fold its unique cases into the existing suites. Then:
```bash
git add quack/amd/linear_training.py tests/amd/
git commit -m "[AMD] Delete duplicate linear autograd Functions — single source of truth"
```

---

## Self-Review

**Spec coverage:** Priority #4 ("consolidate the two linear autograd stacks"). Task 1 builds the numeric safety net; Task 2 gives the surviving stack the capability the other one had (fused-dact); Task 3 redirects entry points one at a time behind the net; Task 4 deletes the duplicates. The net stays green throughout — the defining discipline of this refactor. ✓

**Placeholder scan:** The one genuinely open design choice — how `linear.py` expresses a two-layer MLP Function (extend `LinearActFunc` vs add a dedicated MLP Function) — is called out in Task 2 Step 2 as an implementation decision to make against the real file, not a hidden TODO. Everything else is concrete (exact files, entry points, grep, test commands). This is a refactor, so per-step "write the code" blocks are less literal than a greenfield plan; the invariant (safety net green after every step) is the spec. ✓

**Type consistency:** Entry points `linear_train`/`linear_act_train`/`mlp_train` (`linear.py`, `mlp.py`) and `mlp_func_train`/`gated_mlp_func_train` (`linear_training.py:318,503`) keep identical signatures; helpers `_fused_dact_eligible`/`_act_bwd` (`linear_training.py:219,58`) move to a shared home referenced by both. ✓
