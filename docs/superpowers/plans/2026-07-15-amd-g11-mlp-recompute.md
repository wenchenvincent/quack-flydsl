# MLP Activation Recompute — Implementation Plan (G11)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Design-level; expand to TDD tasks before executing.

**Goal:** Add an AMD `MLPRecomputeFunc`-equivalent autograd Function that saves only `x` (and the weights), recomputing `preact = x @ W1.T` in backward instead of saving `preact`+`postact`, trading one extra forward-shaped matmul for `(M, hidden)` of activation memory.

**Architecture:** AMD's existing `_MLPActFunction` (`quack/amd/linear_training.py:257-315`) deliberately saves **both** `preact` and `postact` — its own docstring (`:24-29`) already names `MLPRecomputeFunc` as the intended follow-up. The recompute variant's forward is nearly identical (same two `linear()` calls) but drops `ctx.save_for_backward(..., preact, postact)` down to `ctx.save_for_backward(x, w1, w2)`; backward recomputes `preact = linear(x, w1, bias1)`, then must still produce `postact` (needed for `dW2`) and `dpreact` (needed for `dx`/`dW1`) from that recomputed `preact`. The one real subtlety this plan resolves: AMD has **two different fused activation-backward kernels** with different capabilities — `gemm_splitk(dact_activation=...)` (used today by `_MLPActFunction`, bf16/f16, but returns **only** `dpreact`, no `postact` — confirmed at `gemm_gfx950_splitk.py:1246-1247`, `emit_postact` is asserted `dgated`-only) vs `quack.amd.gemm.gemm_dact()` (returns `(dx, postact)` together, but is gated **f16-only** at `gemm.py:360`). Picking the wrong one either silently loses bf16 fusion or requires an extra elementwise kernel — this plan's MVP uses the proven bf16-capable `gemm_splitk` path plus one extra elementwise `_act_fwd` call for `postact`, and scopes the fully-fused f16-only alternative as an explicit stretch option.

**Tech Stack:** `quack/amd/linear_training.py` (`_MLPActFunction` at `:257-315` as the structural template, `_act_fwd`/`_act_bwd`/`_fused_dact_eligible` helpers already defined at `:45-91`/`:219-254`), `quack/amd/gemm_gfx950_splitk.py` (`gemm_splitk`, `dact_activation`/`emit_postact` contract at `:1201-1328`), `quack/amd/gemm.py` (`gemm_dact`/`_gemm_dact_fused_eligible` at `:359-422`, the alternative fused kernel), NVIDIA reference `quack/mlp.py` (`MLPRecomputeFunc` at `:94-202`, `mlp_func`'s `recompute=` flag at `:205-249`), `tests/amd/test_linear_train.py` (shape/tolerance conventions to mirror).

---

## Background the engineer needs

- **NVIDIA's `MLPRecomputeFunc` forward** (`quack/mlp.py:106-133`): calls `ops.matmul_fwd_act(x_flat, weight1.T, activation=activation)` → `(_preact, postact)` — **discards** the forward-computed `preact` immediately (`_preact` is never saved) and only saves `x`/`weight1`/`weight2`, conditionally on `needs_input_grad` (`:120-132`, e.g. `saved_w2` only saved `if need_dact`). AMD's MVP can start simpler — unconditionally save `x, w1, w2` (a small amount more memory than NVIDIA's precise gating, but still eliminates the `(M, hidden)`-sized `preact`+`postact` that is the actual memory target) — and note the tighter conditional-save as a possible follow-up tightening, not a blocker.
- **NVIDIA's backward recompute + fused dact, together** (`quack/mlp.py:148-153`):
  ```python
  if need_dact:
      preact = recompute_fwd(x_flat, weight1.T)
      dpreact, postact = ops.matmul_bwd_dact(dout, weight2, preact, activation=ctx.activation)
  ```
  This is a **single** fused kernel producing `(dpreact, postact)` together — NVIDIA's `gemm_dact` (CUTLASS `GemmDActMixin`) computes both as one epilogue. **AMD does not have an equivalent single bf16-capable kernel today.** This is the plan's central finding, established by reading both AMD dact kernels' contracts directly:
  - `gemm_splitk(..., dact_activation=activation)` (`gemm_gfx950_splitk.py:1201-1328`): dtype-eligible for bf16/f16 (`_ALLOWED_DACT`, used today by `_MLPActFunction.backward` at `linear_training.py:295-305`), but its `emit_postact` flag is hard-asserted to `dgated_gate_type`-only (`gemm_gfx950_splitk.py:1246-1247`: `"emit_postact is only valid with dgated_gate_type set"`) — calling it with `dact_activation` set and expecting a `postact` output will assert. **It returns a single `Tensor` (`dpreact` only) in the dact case**, confirmed by its call site's unpacking (`linear_training.py:302-305`, `dpreact = gemm_splitk(...)`, no tuple).
  - `quack.amd.gemm.gemm_dact()` (`gemm.py:379-422`, backed by `gemm_dact_kernel.gemm_dact_fused`): **does** return `(dx, postact)` as a tuple (`gemm.py:387`, `:402-405`), matching NVIDIA's contract exactly — but `_gemm_dact_fused_eligible` (`gemm.py:359-376`) hard-requires `A.dtype == torch.float16` (`:360`, no bf16 branch at all), a narrower gate than `_fused_dact_eligible`'s bf16-inclusive one (`linear_training.py:234`).
- **Resolution for the MVP:** recompute `preact` via a plain `linear()` call, get `postact` via a **separate** elementwise `_act_fwd(preact, activation)` call (already exists, `linear_training.py:45-55`), and get `dpreact` via the **existing, already-tested** `gemm_splitk(dact_activation=...)` path (reuses `_fused_dact_eligible`'s exact eligibility check, zero new kernel-eligibility logic). This costs one extra elementwise kernel launch relative to NVIDIA's single-fused-kernel mechanism, but is bf16-capable today and needs no new FlyDSL code. The fully-fused `gemm_dact()` alternative (matching NVIDIA's mechanism exactly, one kernel instead of two) is scoped as a stretch/follow-on gated on `A.dtype == float16`, with a note that extending `_gemm_dact_fused_eligible` to bf16 would be its own separate, real kernel-eligibility change (out of scope here — verify with a fresh read of `gemm_dact_kernel.py` before attempting, since the f16-only gate may reflect a genuine kernel-level constraint, not just an unswept eligibility check).
- **The non-dact-eligible fallback branch already exists and should be reused verbatim, fed by the recomputed (not saved) `preact`:** `_MLPActFunction.backward`'s `else` branch (`linear_training.py:306-308`, `grad_postact = torch.mm(grad_out, w2); dpreact = _act_bwd(preact, grad_postact, ctx.activation)`) is dtype/shape-agnostic and works identically whether `preact` came from a saved tensor or a recomputed one.
- **Bias scope decision, and why NVIDIA's own scope narrows it away:** AMD's non-recompute `_MLPActFunction`/`mlp_func_train` **does** support `bias1`/`bias2` (`linear_training.py:268`, `:318-321`). NVIDIA's `MLPRecomputeFunc.forward` (`quack/mlp.py:107`) takes **no bias params at all** — and NVIDIA's own `MLP.forward()` (`quack/mlp.py:299-301`) disallows bias whenever `torch.is_grad_enabled()` is true, for *any* training path, recompute or not. Recommend the AMD MVP match this narrower NVIDIA scope (no bias support in v1, asserted via `assert bias1 is None and bias2 is None` or simply omitted from the signature) rather than inventing a wider surface NVIDIA itself didn't build — reduces scope/risk, and a bias-supporting version is a small, isolable follow-up (needs `bias1` saved as an extra small `(hidden,)` tensor for the recompute step, negligible memory cost, but untested territory until built).
- **Gated MLP recompute (`_GatedMLPTrainFunction`'s analogue) is explicitly out of scope** — matches NVIDIA's own `_MLPGatedOps` being a separate ops-class variant of the same `MLPRecomputeFunc`, not a separate Function; note it as a near-mechanical follow-on once the non-gated version is validated (would use `_gated_fwd_interleaved`/`_gated_bwd_interleaved_fallback`, already defined at `linear_training.py:389-428`, in place of `_act_fwd`/`_act_bwd`).
- **Run the relevant existing tests while investigating:** `pytest tests/amd/test_linear_train.py -x -k "128"` to confirm the current non-recompute path's tolerance/shape conventions before writing the recompute variant's test.

---

## Phase 1 — Investigation

**Files touched:** none (read-only).

1. Read `gemm_dact_kernel.py` in full to confirm whether its `A.dtype == torch.float16`-only gate (`gemm.py:360`) is a genuine MFMA/kernel-structural constraint (e.g. the 16×16 scalar-store kernel family it belongs to, per `gemm_gfx950.py`'s "proof-of-life minimal kernel" framing, may simply never have been extended to bf16) or an accidental omission — this determines whether the "fully-fused stretch option" in Background is a quick follow-up or a real kernel change.
2. Confirm `gemm_splitk`'s `dact_activation` path's `_ALLOWED_DACT` set (referenced but not fully quoted above) matches `_fused_dact_eligible`'s activation set (`linear_training.py:216`, `{"relu","relu_sq","silu","gelu_tanh_approx"}`) exactly — a mismatch here would mean some activations silently fall to the non-fused branch during recompute even when they're fused in the non-recompute path, which is fine correctness-wise but worth knowing for the done-criteria's performance framing.
3. Decide the `ctx.save_for_backward` gating precision: unconditional `(x, w1, w2)` (MVP, simpler) vs NVIDIA's `needs_input_grad`-conditional saves (`quack/mlp.py:119-132`, tighter but more branching). Recommend MVP unconditional for v1; note the tightening as a low-risk follow-up.

**Done when:** the plan's execution log records (a) whether `gemm_dact_fused`'s f16-only gate is structural or just unswept, (b) confirmed activation-set parity between the two dact paths, (c) the chosen save-gating precision.

---

## Phase 2 — Design

**New autograd Function**, structurally parallel to `_MLPActFunction`:

```python
class _MLPRecomputeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, activation):
        preact = linear(x, w1)                  # bias1 out of scope for v1, see Background
        postact = _act_fwd(preact, activation)
        out = linear(postact, w2)
        ctx.save_for_backward(x, w1, w2)         # NOT preact, NOT postact
        ctx.activation = activation
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w1, w2 = ctx.saved_tensors
        grad_x = grad_w1 = grad_w2 = None
        # Recompute preact — the extra matmul we trade for memory.
        preact = linear(x, w1)
        need_dact = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        if need_dact:
            if _fused_dact_eligible(grad_out, w2, preact, ctx.activation):
                postact = _act_fwd(preact, ctx.activation)          # extra elementwise;
                from quack.amd.gemm_gfx950_splitk import gemm_splitk # see Background for why
                dpreact = gemm_splitk(grad_out, w2.t().contiguous(),
                                       preact=preact, dact_activation=ctx.activation)
            else:
                postact = _act_fwd(preact, ctx.activation)
                grad_postact = torch.mm(grad_out, w2)
                dpreact = _act_bwd(preact, grad_postact, ctx.activation)
        else:
            postact = _act_fwd(preact, ctx.activation)  # still needed for dW2 below
            dpreact = None
        if ctx.needs_input_grad[2]:
            grad_w2 = torch.mm(grad_out.t(), postact)
        if need_dact:
            if ctx.needs_input_grad[0]:
                grad_x = torch.mm(dpreact, w1)
            if ctx.needs_input_grad[1]:
                grad_w1 = torch.mm(dpreact.t(), x)
        return grad_x, grad_w1, grad_w2, None
```
(Sketch only — the actual implementation task should re-derive the `needs_input_grad` branching precisely against `_MLPActFunction.backward`'s existing structure rather than copy this sketch verbatim, since the `postact`-always-needed-for-`dW2` vs `dpreact`-only-needed-for-`dx`/`dW1` dependency graph has a few branches worth double-checking against the reference.)

- **Public entry point:** add `recompute: bool = False` param to `mlp_func_train()` (`linear_training.py:318-350`), dispatching to `_MLPRecomputeFunction.apply(x, w1, w2, activation)` when `True` — matches NVIDIA's `mlp_func(..., recompute=False)` flag shape (`quack/mlp.py:205-227`) rather than exposing a separately-named function, for API parity.
- **Memory-assertion strategy for the test:** rather than parsing `torch.cuda.memory_allocated()` deltas (noisy, allocator-dependent), assert directly on `ctx.saved_tensors`' total byte count immediately after a forward call — `sum(t.numel()*t.element_size() for t in ctx.saved_tensors if t is not None)` — comparing `_MLPRecomputeFunction`'s saved bytes against `_MLPActFunction`'s for the same shape; the expected gap is `2 * M * hidden * dtype_size` (both `preact` and `postact` dropped), a precise, deterministic number to assert against rather than a fuzzy "less memory" check.

**Done when (design):** the exact `needs_input_grad` branch structure is re-verified against `_MLPActFunction.backward` line-by-line (not just the sketch above), and the memory-assertion formula is confirmed against the two Functions' actual save-lists.

---

## Phase 3 — Implementation

**Files:**
- Modify: `quack/amd/linear_training.py` — add `_MLPRecomputeFunction` (per Phase 2, non-gated only) and thread `recompute: bool = False` through `mlp_func_train()`. Reuse `_act_fwd`, `_act_bwd`, `_fused_dact_eligible` unchanged (no new eligibility logic — this is the plan's main risk-reduction move, confirmed in Background).
- Test: `tests/amd/test_linear_train.py` (extend) or new `tests/amd/test_mlp_recompute.py`:
  - Numerics: `mlp_func_train(..., recompute=True)` matches `mlp_func_train(..., recompute=False)` (and both match a plain-torch autograd reference) for `dx`, `dW1`, `dW2`, across the activation set and a couple of shapes from `test_linear_train.py`'s existing parametrize table, bf16 and f16.
  - Memory: assert `_MLPRecomputeFunction`'s saved-tensor byte total is smaller than `_MLPActFunction`'s by the expected `2*M*hidden*dtype_size` margin (per Phase 2's formula) for a representative shape.
  - Fused-path coverage: at least one parametrized shape/dtype combo that satisfies `_fused_dact_eligible` (exercises the `gemm_splitk` fused-dact recompute branch) and one that doesn't (exercises the `_act_bwd` fallback branch), so both branches of the new backward are actually tested, not just the fast path.

**Done when:** all new tests pass; the memory assertion's measured gap matches the Phase-2 formula (not just "is smaller"); both branches of the new backward (fused-eligible and fallback) are exercised by at least one test case each.

---

## Roadmap — fully-fused single-kernel recompute + gated-MLP recompute (separate follow-ons)

Out of scope for this plan.

- **Fully-fused recompute** (matching NVIDIA's exact single-kernel `(dpreact, postact)` mechanism): requires either extending `gemm_dact_kernel.py`'s eligibility to bf16 (if Phase 1.1 finds the f16-only gate is unswept rather than structural) or adding an `emit_postact`-style option to `gemm_splitk`'s plain (non-dgated) `dact_activation` path. Trigger: profiling shows the extra elementwise `_act_fwd` call in this plan's MVP backward is a measurable cost at the target training shapes.
- **Gated MLP recompute** (`_GatedMLPTrainFunction` analogue): near-mechanical once the non-gated version above is validated — swap `_act_fwd`/`_act_bwd` for `_gated_fwd_interleaved`/`_gated_bwd_interleaved_fallback` (`linear_training.py:389-428`) and `gemm_splitk(dact_activation=...)` for `gemm_splitk(dgated_gate_type=...)` (which, notably, **is** the one path where `gemm_splitk` already supports `emit_postact` — `gemm_gfx950_splitk.py:1409-1423` — so the gated recompute variant may actually reach full single-kernel fusion *before* the non-gated one does; worth flagging to whoever picks this up, since it inverts the difficulty ordering from Background's analysis).

---

## Self-Review

**Spec coverage:** The prompt asked for a new autograd Function saving only `x` (+weights), recomputing `preact` in backward before the fused-dact step, keeping the fused-dact backward, with a numerics + memory test. All four are addressed with concrete code (Phase 2), file/line references, and a precise (not fuzzy) memory-assertion formula (Phase 2/3). ✓

**Placeholder scan:** The one place this plan could have hand-waved — "just call the fused dact kernel like the non-recompute path does" — is instead where it does its most concrete work: it traces both AMD dact kernels' actual return-value contracts (`gemm_splitk` returns `dpreact`-only for plain dact per the `emit_postact` assertion at `gemm_gfx950_splitk.py:1246-1247`; `gemm.gemm_dact` returns `(dx, postact)` but is f16-gated) before choosing the MVP's two-kernel composition, rather than assuming NVIDIA's single-kernel mechanism transfers unchanged. ✓

**Consistency check:** The `needs_input_grad` branch sketch in Phase 2 is explicitly flagged as a sketch requiring re-derivation against `_MLPActFunction.backward`'s actual structure during implementation, rather than presented as final — this matches the "design-level, not line-by-line TDD" brief while still being honest about where the sketch is unverified. The bias-scope decision (excluding bias1/bias2 from v1) is justified by NVIDIA's own equivalent scope narrowing (`quack/mlp.py:299-301`), not an arbitrary cut. ✓
