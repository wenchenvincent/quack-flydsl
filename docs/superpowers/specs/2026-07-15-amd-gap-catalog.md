# AMD FlyDSL Port — Gap Catalog (2026-07-15)

Single source of truth for what the AMD port (`quack/amd/`) is missing vs the
NVIDIA CuTe-DSL original (`quack/`). Supersedes the parity-review artifact's
gap list, corrected for everything verified/closed during the 2026-07-14/15
sessions. Each gap notes: **status** (verified-absent / stale-doc / partial),
**category**, **effort** (S/M/L), **risk**, and **plan** (file, once written).

Legend — effort: S ≤ half-day plumbing · M = 1–3 day feature · L = multi-day
kernel/infra project. Risk: how likely an autonomous attempt leaves things
broken.

---

## Closed / stale (NOT gaps — recorded so they aren't re-investigated)

| Item | Reality | Evidence |
|---|---|---|
| High-level autograd/module layer (reductions) | **Built** (Phase 1) | `quack/amd/nn.py`, 24 tests |
| TopK backward | **Built** (Phase 2) | `topk_bwd`/`TopKFunction`, 11 tests |
| Fused act-backward "disabled" | **Enabled + 2.17× faster** (Phase 3); docstring was stale | `linear_training.py`, bench |
| Two linear autograd stacks | **Consolidated** (Phase 4) | `mlp_train`→`mlp_func_train` |
| nn_big crashes at large shapes | **Fixed** (const_expr drift) | `3d56123`, large-shape test |
| Large-N / huge-vocab reductions | **Already capable** — CE correct at 512K, one launch | probe, `test_*_large_*` |
| Autotuner "empty table" | **4 real TuneEntry** now; docstring stale | `gemm_autotune.py:171` |

Recurring theme: several "gaps" were stale docstrings or missing tests, not
absent functionality. **Verify before building.**

---

## Genuine gaps — ergonomics (S, low risk)

### G1. No `nn.Module` for Linear / MLP / LinearCrossEntropy
- **Status:** verified-absent. Phase 1 added `RMSNorm`/`LayerNorm` modules only.
- **Category:** ergonomics · **Effort:** S · **Risk:** low (plumbing over working functions).
- **Want:** `Linear(nn.Module)`, `MLP(nn.Module)`, `LinearCrossEntropy(nn.Module)` in `quack/amd/nn.py`, wrapping `linear_train`/`mlp_func_train`/`linear_cross_entropy_fwd_bwd`, exported.
- **Plan:** `2026-07-15-amd-g1-linear-mlp-modules.md` · **IMPLEMENT tonight.**

### G2. No `fuse_grad_accum`
- **Status:** verified-absent (only a bias-grad-f32 comment).
- **Category:** training ergonomics · **Effort:** M · **Risk:** medium (autograd + in-place `.grad` semantics).
- **Want:** optional `fuse_grad_accum=True` that accumulates dW straight into `weight.grad` (via `gemm_tn` out= or torch add-in-place) instead of returning it.
- **Plan:** `2026-07-15-amd-g2-fuse-grad-accum.md` · defer (needs care; not overnight-safe without more review budget).

## Genuine gaps — activations / epilogues

### G3. `swiglu_oai` not wired into the fused gated kernel
- **Status:** partial — activation defined in `activation.py:120` (`swiglu_oai`/`dswiglu_oai`), but `gemm_gated._SUPPORTED_GATES = {swiglu,reglu,geglu,glu}` excludes it (`:52`), and the FlyDSL `_apply_gate` doesn't implement it.
- **Category:** activation coverage · **Effort:** S–M · **Risk:** medium (adds gate math to the FlyDSL kernel body).
- **Plan:** `2026-07-15-amd-g3-swiglu-oai.md` · IMPLEMENT if the kernel change is contained; else plan-only.

### G4. No composable epilogue framework
- **Status:** verified — `epi_ops.py`/`gemm_default_epi.py` are unused scaffold (no kernel imports them); each epilogue is hand-rolled inline.
- **Category:** architecture · **Effort:** L · **Risk:** high (broad refactor).
- **Plan:** `2026-07-15-amd-g4-composable-epilogue.md` · plan-only.

### G5. Symmetric GEMM is A@Aᵀ only
- **Status:** verified — no bias/act/alpha/beta/C, no triangular-skip scheduler.
- **Category:** feature · **Effort:** M · **Risk:** medium (kernel epilogue work).
- **Plan:** `2026-07-15-amd-g5-symmetric-richness.md` · plan-only.

## Genuine gaps — dtypes / formats (L, high risk)

### G6. No standard fp8 / int8 / tf32 GEMM
- **Status:** verified — `_MFMA_SUPPORTED_DTYPES = {float16, bfloat16}`; fp8 only via blockscaled.
- **Category:** dtype · **Effort:** L · **Risk:** high (new MFMA dtype paths).
- **Plan:** `2026-07-15-amd-g6-fp8-int8-gemm.md` · plan-only.

### G7. Blockscaled is MXFP8-e4m3 only
- **Status:** verified — no fp4/nvfp4/e5m2, no standard MX scaling (128-elem f32 vs 16/32-elem e8m0/e4m3), no stochastic rounding.
- **Category:** dtype/format · **Effort:** L · **Risk:** high.
- **Plan:** `2026-07-15-amd-g7-blockscaled-formats.md` · plan-only.

### G8. No stochastic rounding
- **Status:** verified — only a `sr_seed` placeholder + "wire via rocdl" TODO (`gemm_default_epi.py:12`).
- **Category:** numerics · **Effort:** M · **Risk:** medium-high (rocdl SR intrinsic + PRNG in-kernel).
- **Plan:** `2026-07-15-amd-g8-stochastic-rounding.md` · plan-only.

## Genuine gaps — layouts / features

### G9. No TT layout
- **Status:** verified — no TT kernel; falls back to torch/hipBLASLt.
- **Category:** layout · **Effort:** M–L · **Risk:** high (new kernel).
- **Plan:** `2026-07-15-amd-g9-tt-layout.md` · plan-only.

### G10. No varlen-K
- **Status:** verified-absent (varlen-M is host-side only).
- **Category:** feature · **Effort:** L · **Risk:** high.
- **Plan:** `2026-07-15-amd-g10-varlen-k.md` · plan-only.

### G11. No MLP activation-recompute (`MLPRecomputeFunc`)
- **Status:** verified-absent (AMD saves both preact+postact).
- **Category:** memory · **Effort:** M · **Risk:** medium (autograd restructure).
- **Plan:** `2026-07-15-amd-g11-mlp-recompute.md` · plan-only (doable but changes memory behavior).

## Genuine gaps — architecture

### G12. RDNA gfx1201 / gfx1250 are stubs
- **Status:** verified — `raise NotImplementedError` on CDNA; real WMMA GEMM unwritten.
- **Category:** arch · **Effort:** L · **Risk:** high (can't even test on this CDNA box).
- **Plan:** `2026-07-15-amd-g12-rdna-wmma.md` · plan-only.

## Genuine gaps — infrastructure

### G13. In-kernel dynamic tile scheduler
- **Status:** partial — host-side planning only; the atomic work-counter loop is documented but not emitted by the shared scheduler (streamk_prod hand-rolls its own). CLC degrades to DYNAMIC.
- **Category:** infra · **Effort:** L · **Risk:** high.
- **Plan:** `2026-07-15-amd-g13-inkernel-scheduler.md` · plan-only.

### G14. No reusable profiler / no sort subpackage
- **Status:** verified — no `trace.py` equivalent; bitonic sort inlined into topk.
- **Category:** infra/tooling · **Effort:** L · **Risk:** medium.
- **Plan:** `2026-07-15-amd-g14-profiler-sort.md` · plan-only.

### G15. Autotuner docstring stale + table sparse
- **Status:** stale-doc — table has 4 entries; docstring says "empty."
- **Category:** infra · **Effort:** S · **Risk:** low.
- **Plan:** folded into G1's session (trivial docstring fix). **IMPLEMENT tonight.**

---

## Overnight execution decision (autonomous, no feedback available)

**Implement tonight** (low risk, high value, fully testable, no giant-kernel authoring):
- **G1** — Linear/MLP/LinearCrossEntropy nn.Module wrappers.
- **G15** — fix the stale autotuner docstring.
- **G3** — swiglu_oai, *only if* the FlyDSL kernel change stays contained and testable; otherwise plan-only.

**Plan-only tonight** (real kernel/infra projects — building blind overnight would risk leaving broken kernels; each gets a scoped plan for later execution):
- G2, G4, G5, G6, G7, G8, G9, G10, G11, G12, G13, G14.

Rationale: the user is asleep and asked for my judgement. The responsible move
is to land the safe, verifiable plumbing (G1/G15, maybe G3) with full test
coverage and reviews, and to hand over crisp plans for the large kernel work
rather than half-finish a dtype/arch/scheduler kernel that can't be validated
or rolled back cleanly before morning.
