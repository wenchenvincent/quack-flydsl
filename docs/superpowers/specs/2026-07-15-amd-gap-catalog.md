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
- **Status:** fp8 + int8 **DONE** (`2006d1c`, `e3321e1`); tf32 deferred (YAGNI).
- **Category:** dtype · **Effort:** L · **Risk:** high (new MFMA dtype paths).
- **Plan:** `2026-07-15-amd-g6-fp8-int8-gemm.md`.
- **Shipped:** `gemm_fp8(a,b,out_dtype)` (`gemm_gfx950_fp8.py`, `mfma_f32_16x16x32_fp8_fp8`, f32 accumulate, validated vs dequant, rel<0.003) and `gemm_int8(a,b)->int32` (`gemm_gfx950_int8.py`, `mfma_i32_16x16x32_i8`, bit-EXACT — and torch has no CUDA int matmul, so it fills a real hole). Both single-16×16-tile MVPs, K=32, 8 elems/lane packed to i64. Exported from `quack.amd`. 20 tests.

### G7. Blockscaled is MXFP8-e4m3 only
- **Status:** MXFP4 **DONE** (2026-07-15). Was MXFP8-e4m3 only; now a full
  MXFP4 (e2m1 elem + e8m0 scale, 32-elem blocks) GEMM ships via the
  hardware-scaled `mfma_scale_f32_16x16x128_f8f6f4` atom. NVFP4 (e4m3 scales)
  is NOT supported by this instruction, so the hardware-scaled path is
  MXFP4-only; e5m2 fp8 still deferred (YAGNI).
- **Category:** dtype/format · **Effort:** L · **Risk:** high.
- **Plan:** `2026-07-15-amd-g7-blockscaled-formats.md`.
- **Shipped:** `gemm_mxfp4(a, b) -> f32` (`gemm_gfx950_mxfp4.py`, NT `C = A@B.T`,
  M%16/N%16/K%128) + `quantize_mxfp4`/`dequantize_mxfp4` (`mxfp4_ops.py`).
  Empirically-resolved layout: 32 fp4/lane in register i32[0..3], K-group =
  lane//16, per-lane e8m0 scale in scaleA/scaleB byte 0 with opsel=0. Bit-exact
  vs dequant-then-matmul across 5 grid/K shapes; 13 tests. Exported from
  `quack.amd`.

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

## Round-1 gap sweep — status (autonomous run, 2026-07-16)

Post-upstream-merge (v0.6.1) gap sweep. **Landed + tested + committed:**
- **G5** — bias/activation/alpha/beta/C epilogue for `gemm_symmetric` (fixes the
  silent A.T-transpose fallback); +13 tests. `fc196b5`.
- **G9** — `gemm_tt` (TT layout `C=A.T@B.T`) via Option-A dispatch on transposed
  views (no in-repo consumer → YAGNI, dedicated kernel roadmapped); 10 tests. `eddd712`.
- **G2** — `fuse_grad_accum`: `gemm_tn` gains an `accumulate` (read-add-store)
  flag; `LinearFunc`/`LinearActFunc`/`linear_train`/`nn.Linear` accumulate dW
  into `weight.grad` in-place; 5 tests. `c18291b`. (linear_training's torch.mm/
  gemm_dact dweight path deferred — not gemm_tn.)
- **G8** — fp8 stochastic-rounding quantizer `quantize_fp8_sr` via hardware
  `v_cvt_sr_fp8_f32`/`bf8` + in-kernel murmur3 PRNG; SR 12× less biased than RTN,
  exact bracketing; 5 tests. `d5dd50e`. (bf16-GEMM-epilogue SR via unwrapped
  packed intrinsic deferred.)
- **G7 (MXFP6)** — `gemm_mxfp6` (e2m3) via cbsz=2 on the f8f6f4 atom + `mxfp6_ops`;
  bit-exact vs dequant; 14 tests. `feb8057`. **G7 now covers MXFP8+MXFP4+MXFP6.**

**Round 2 (remaining, larger):** G4 (composable epilogue), G10 (varlen-K),
G13 (in-kernel scheduler), G14 (profiler/sort) — all validatable; G12 (RDNA
WMMA) blocked (no RDNA hardware on this CDNA box).

## Overnight execution — final status (autonomous run, 2026-07-15)

**Implemented + tested + committed:**
- **G15** — stale autotuner docstring fixed. `1416a4c`.
- **G3** — `swiglu_oai` wired into the fused gated kernel (forward), +8 tests
  (41 gated pass). `03bfb24`. dgated backward for swiglu_oai = noted follow-up.
- **G1** — `Linear` / `MLP` / `LinearCrossEntropy` `nn.Module` wrappers, 29
  nn-layer tests pass, exported from `quack.amd`. `ddac26d` + docs `9dea368`.
  The drop-in high-level layer is now complete (reductions + linear/MLP).
- **G11** — `mlp_recompute_train` activation-recompute autograd. *(in progress
  as of this write; verify commit before relying on it.)*

**Plan-only (design plans committed `91370d5`; NOT implemented — real kernel/
arch/infra projects, deliberately not built unattended):**
- G2 (fuse_grad_accum), G4 (composable epilogue), G5 (symmetric richness),
  G6 (fp8/int8 GEMM), G7 (blockscaled formats), G8 (stochastic rounding),
  G9 (TT layout), G10 (varlen-K), G12 (RDNA WMMA), G13 (in-kernel scheduler),
  G14 (profiler/sort). Each plan is in `docs/superpowers/plans/2026-07-15-amd-g*`.

Rationale: the user was asleep and asked for my judgement. I landed the safe,
verifiable plumbing (G1/G3/G11/G15) with full test coverage, and handed over
crisp source-grounded plans for the large kernel work — rather than
half-finishing a dtype/arch/scheduler kernel that couldn't be validated or
cleanly rolled back before morning. Recurring lesson banked: verify before
building — several "gaps" (autotuner, fused-dact, large-N, MLPActFunction
docstring) were stale docs or missing tests, not absent functionality.
