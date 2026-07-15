# RDNA gfx1201/gfx1250 WMMA GEMM — Implementation Plan (G12)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Design-level; expand to TDD tasks before executing.

**Goal:** Replace the single-wave 16×16 WMMA MVP stub with a validated, larger-tile RDNA WMMA GEMM kernel that `gemm_gfx1201.py` and `gemm_gfx1250.py` actually dispatch to, correcting a fragment-mapping bug discovered while reading the current code along the way.

**Architecture:** RDNA's WMMA path is structurally different from the CDNA MFMA path this port has already validated (`gemm_gfx950*.py`): wave32 instead of wave64, a full A-row/B-col duplicated across a lane pair (`lane`, `lane+16`) rather than MFMA's per-quarter split, and a different accumulator→output unpack rule. The plan builds a wave32 "WMMA atom" wrapper analogous to `gemm_gfx950_mfma_core.py`'s MFMA atom abstraction, ports the LDS-staged multi-warp tile structure of FlyDSL's already-authored reference kernels (`rdna_f16_gemm.py`, `wmma_gemm_gfx1250.py`) into `quack/amd/gemm_rdna_wmma.py`, and leaves `gemm_gfx1201.py` / `gemm_gfx1250.py`'s existing delegation (`gemm_mfma` → `gemm_wmma`) to pick up the new kernel automatically.

**Tech Stack:** `quack/amd/gemm_rdna_wmma.py`, `quack/amd/gemm_gfx1201.py`, `quack/amd/gemm_gfx1250.py`, `quack/amd/flydsl_utils.py` (`get_wave_size`, `get_lds_size_bytes`), `quack/amd/gemm_gfx950_mfma_core.py` (pattern reference only). FlyDSL references: `/workspace/FlyDSL/kernels/rdna_f16_gemm.py`, `/workspace/FlyDSL/kernels/rdna3_f16_gemm.py`, `/workspace/FlyDSL/kernels/wmma_gemm_gfx1250.py`, `/workspace/FlyDSL/kernels/gemm_common_gfx1250.py`, `/workspace/FlyDSL/kernels/gemm_fp8fp4_gfx1250.py`.

---

## Background

`quack/amd/gemm_rdna_wmma.py` currently ships one kernel builder, `_build_gemm_wmma_16x16_f16` (`gemm_rdna_wmma.py:62-168`): single wave (32 threads), one 16×16×16 tile per workgroup, no LDS, no epilogue, f16×f16→f32 only. `_compile` (`gemm_rdna_wmma.py:178-189`) gates on `_is_rdna(arch)` and raises `NotImplementedError` for anything else, including this box's gfx950 (CDNA4). `gemm_gfx1201.py` and `gemm_gfx1250.py` are both thin delegators to `gemm_wmma` (`gemm_gfx1201.py:14-21`, `gemm_gfx1250.py:16-22`) — no per-arch kernel selection logic needs to change once the underlying kernel is upgraded.

**CRITICAL CONSTRAINT — this work is implement-blind.** This box is gfx950 (MI355X, CDNA4). RDNA WMMA instructions (`v_wmma_f32_16x16x16_f16` family) don't exist on CDNA silicon, and `_compile`'s arch gate raises before any launch is attempted even if they did. There is no way to execute, benchmark, or numerically validate anything produced by this plan on this machine. The only local validation available is:
  1. Direct MLIR/IR inspection — call the kernel-builder function with an explicit `arch="gfx1201"` / `arch="gfx1250"` argument (bypassing `_compile`'s arch-detection, not its correctness), with `FLYDSL_DUMP_IR=1`, and inspect the emitted MLIR / lowered ISA text for verifier errors and for the presence of the expected `v_wmma_*` instructions (grep the disassembly — a silent fallback to scalar codegen would otherwise pass IR verification while being wrong).
  2. Static comparison against FlyDSL's own reference kernels under `/workspace/FlyDSL/kernels/`, which are presumed to have been exercised on real RDNA hardware within the FlyDSL project (this plan cannot independently confirm that presumption — flag it to the human reviewer).
  3. A hardware-independent Python model of the fragment mapping (plain lists/dicts, no FlyDSL) that checks the claimed lane→(row,col) mapping is a bijection over the tile.
  Actual numerical correctness on real RDNA hardware (gfx1201 box or MI450) is out of scope for this plan and must be a follow-up session run on that hardware.

**Scope order:** gfx1201 first (plain WMMA, 64 KiB LDS per `get_lds_size_bytes` in `flydsl_utils.py:48-63`), then gfx1250 for free via the existing delegation (gfx1250's standard-dtype path is identical per its own docstring, `gemm_gfx1250.py:5-9`). gfx1250's scaled-WMMA fp8/fp4 path (`rocdl.wmma_scale_f32_16x16x128_f8f6f4`, confirmed in `/workspace/FlyDSL/kernels/gemm_fp8fp4_gfx1250.py:916,934`) is explicitly out of scope — flagged as a roadmap item at the end.

---

## Phase 1: Fix the C-writeback fragment mapping (bug found during planning)

**Files:** `quack/amd/gemm_rdna_wmma.py`

**Approach:** While reading the current MVP for this plan, its accumulator writeback (`gemm_rdna_wmma.py:141-152`) was found to likely be **transposed**. It computes `out_row = m_base + lane_idx` (`lane_idx = tid % 16`) and `out_col = n_base + col_base + i` (`col_base = (tid // 16) * 8`). FlyDSL's own `rdna_f16_gemm.py`, a larger, presumably-validated WMMA kernel, uses the opposite assignment for its accumulator store (`kernels/rdna_f16_gemm.py:337-338`): `g_row = tile_m0 + wmma_m_off + base8 + si` and `g_col = tile_n0 + wmma_n_off + lane16`, where `lane16 = lane % 16` and `base8 = (lane // 16) * 8`. That is: the reference strides **rows** by `lane // 16` (the 8-wide accumulator stripe) and picks a single **column** by `lane % 16`; the current stub does the reverse. Notably, the stub's A/B fragment *load* addressing already matches the reference convention (`a_row`/`b_col` keyed on `lane % 16`, k-slice keyed on `lane // 16` — see `gemm_rdna_wmma.py:106-107,123-124` vs. `kernels/rdna_f16_gemm.py:219-233`), so only the accumulator unpack looks backwards. Correct `_store_f`'s call sites (or the equivalent in the Phase 3 tiled rewrite) to match the reference's row-stripe/single-column convention, and add a comment citing `rdna_f16_gemm.py:337-338` as the source of truth.

**Risks:** Neither version can be executed here to confirm which is right. Mitigate with two independent, hardware-free checks: (a) match FlyDSL's own reference kernel rather than re-deriving from scratch (agreement with a second implementation is stronger evidence than internal self-consistency alone), and (b) write a throwaway host-side Python model of the lane→(row,col) mapping and assert it is a bijection onto the 16×16 = 256-element tile (32 lanes × 8 elements/lane) — this alone would have caught a transposition only if combined with a shape check, so also assert the *set* of (row,col) pairs matches `{(r,c) : r<16, c<16}` under both the row-major and the (hypothetically wrong) column-major reading, to make sure the fix picks the one that matches the reference exactly rather than merely "a" bijection.

**Done criteria:** the bijection self-check script (throwaway, not committed) passes for the corrected mapping and specifically confirms it differs from the pre-fix mapping only by row/col swap; the diff to `gemm_rdna_wmma.py` is a small, isolated change to the writeback loop; a code comment cites `rdna_f16_gemm.py:337-338`.

## Phase 2: Wave32 WMMA atom wrapper

**Files:** new `quack/amd/gemm_gfx1250_wmma_core.py` (or `gemm_rdna_wmma_core.py` — name to match the existing `gemm_gfx950_mfma_core.py` convention); modify `quack/amd/gemm_rdna_wmma.py` to consume it.

**Approach:** `gemm_gfx950_mfma_core.py`'s `_WmmaHalfK32`/`swizzle_xor16` (referenced from `gemm_gfx950_nt_4wave.py` and friends) exists so the 15+ `gemm_gfx950_*.py` variants share one lane-mapping source of truth instead of re-deriving MFMA fragment addressing per file. Do the analogous thing for WMMA/wave32 before Phase 3's larger port, so the tiled kernel isn't yet another place that re-derives lane arithmetic from scratch: one small object/module owning (a) A/B fragment LDS-load addressing (row/col keyed on `lane % 16`, k-slice keyed on `(lane // 16) * 8 + local`, per Phase 1's confirmed convention), (b) the accumulator→global-output unpack (Phase 1's fix), (c) LDS tile layout with K-padding (`a_k_pad=8`, `b_k_pad=8` from `rdna_f16_gemm.py:52-53`, copied verbatim rather than re-derived — they're already tuned against RDNA4's LDS bank width and this plan has no way to re-tune them). Do **not** attempt to unify this with the existing MFMA atom wrapper into one shared base class — wave32 vs. wave64 changes enough (lane-pair duplication, no need for MFMA's half-K interleave trick) that forcing a common abstraction for a single not-yet-validated kernel is premature.

**Risks:** `get_wave_size(arch)` (`flydsl_utils.py:36-44`) already returns 32 for RDNA archs, so nothing structural blocks reuse of the existing dtype/param-base helpers in `flydsl_utils.py`. The main risk remains the LDS bank-conflict swizzle pattern being un-verifiable without hardware; mitigated by copying the reference's constants rather than inventing new ones.

**Done criteria:** the atom wrapper's index formulas carry inline comments pointing at the corresponding `rdna_f16_gemm.py` line ranges they were transcribed from (that file has no equivalent abstraction — it's all inline — so "correspondence" means the constants and formulas match, not that a function is literally shared).

## Phase 3: Multi-warp tiled kernel body (128×128×32, 4 warps)

**Files:** `quack/amd/gemm_rdna_wmma.py` (add a new tiled build function alongside the existing single-wave one; the single-wave `_build_gemm_wmma_16x16_f16` stays as a minimal IR-diffing / debug target, not deleted)

**Approach:** Port the structure of `rdna_f16_gemm.py`'s `create_wmma_gemm_module` (128×128 output tile, BLOCK_K=32, 2×2 warp grid = 4 warps = 128 threads, double-buffered LDS ping-pong, `group_m`-based L2 swizzle — `kernels/rdna_f16_gemm.py:39-77` for the shape math) into `quack/amd/gemm_rdna_wmma.py`, translated into this repo's existing `@flyc.kernel` / `@flyc.jit` idioms (`fx.Tensor` args, `SmemAllocator`, `fx.rocdl.make_buffer_tensor`) for consistency with the rest of `quack/amd/`, rather than the reference's slightly different `buffer_ops.create_buffer_resource` idiom. f16×f16→f32 only for the MVP, matching this port's established pattern of narrowing dtype first and adding epilogue/bf16/fp8 support in follow-ups (see how `gemm_gfx950.py` itself grew from a bare MFMA loop to the epilogue-carrying kernel it is now). `gemm_gfx1250.py`'s TDM-async-copy variant (`wmma_gemm_gfx1250.py`) is a further optimization layer on top of this shape and is explicitly deferred — gfx1250 gets correctness via the shared plain-WMMA kernel first, TDM is a separate perf plan.

**Risks:** highest-uncertainty phase in this plan — a fully pipelined kernel has far more failure surfaces (barrier placement, double-buffer index parity, tail-K handling, swizzle correctness) than the 16×16 MVP, none of them testable here. Mitigate by porting as close to line-for-line as practical rather than "improving" anything mid-port, and by explicitly flagging to the human reviewer that `rdna_f16_gemm.py`'s claimed hardware validation (implied by its docstring, "inspired by Triton's 93 TFLOPS approach") is an assumption this plan inherits, not something it re-verifies.

**Done criteria:** `FLYDSL_DUMP_IR=1` produces MLIR that lowers cleanly through to `gpu.binary` / amdgcn ISA text for both `arch="gfx1201"` and `arch="gfx1250"` (invoked by calling the builder directly with an explicit `arch=` kwarg — never through `_compile`'s arch gate, and never `.launch()`, which would attempt an actual device call); zero MLIR verifier errors; the disassembled ISA text contains `v_wmma_f32_16x16x16_f16` (or `_bf16`) instructions, confirmed by grep, ruling out a silent scalar-codegen fallback.

## Phase 4: Dispatch cleanup + gfx1250 scaled-WMMA roadmap note

**Files:** `quack/amd/gemm_gfx1201.py`, `quack/amd/gemm_gfx1250.py`

**Approach:** No dispatch-logic changes are needed — both files already delegate `gemm_mfma` straight to `gemm_rdna_wmma.gemm_wmma` (`gemm_gfx1201.py:14-21`, `gemm_gfx1250.py:16-22`), so Phase 3's kernel swap propagates automatically. Update both docstrings to drop the "MVP"/"single-wave" framing (`gemm_gfx1201.py:9-11`, `gemm_gfx1250.py:9-14`) and describe the new tile shape. Add a `# ROADMAP` comment in `gemm_gfx1250.py` pointing at the scaled-WMMA fp8/fp4 path (`rocdl.wmma_scale_f32_16x16x128_f8f6f4`, `rocdl.wmma_scale_f32_32x16x128_f4`; reference `FlyDSL/kernels/gemm_fp8fp4_gfx1250.py:916,934`) as a separate follow-on plan of comparable scope to this one (new block-scale loading path, new fragment layout for the scaled MFMA-equivalent) — explicitly not attempted here.

**Risks:** none beyond what Phase 3 already carries.

**Done criteria:** `gemm_gfx1201.gemm_mfma` and `gemm_gfx1250.gemm_mfma` both resolve, unchanged in their own source, to the Phase 3 tiled kernel; docstrings reflect the new implementation; the roadmap comment is in place and cites the exact intrinsic names.

---

## Roadmap (explicitly out of scope here)

- **gfx1250 scaled-WMMA fp8/fp4** (`gemm_fp8fp4_gfx1250.py` port) — separate plan.
- **gfx1250 TDM async-copy pipelining** (`wmma_gemm_gfx1250.py`) — separate perf plan layered on top of Phase 3's plain-WMMA correctness baseline.
- **On-hardware numerical validation** — must run on real gfx1201/gfx1250 hardware; this plan produces IR-level-validated code only.
- **Epilogue support (bias/activation/alpha/beta)** — mirror `gemm_gfx950.py`'s epilogue growth once the bare tiled WMMA kernel is confirmed correct on real hardware.

## Self-Review

**Spec coverage:** Both archs are addressed with an explicit ordering (gfx1201 first, gfx1250 free via existing delegation, scaled-WMMA deferred) as requested. The wave32-vs-wave64 fragment layout, the `rocdl.wmma_*` intrinsics, an MFMA-atom-equivalent wrapper, and dispatch are each covered by Phases 2–4. The implement-blind constraint is stated prominently in its own Background subsection with three concrete, hardware-free validation mechanisms, not just a caveat buried in a risk line. ✓

**Placeholder scan:** No step defers "figure this out later" without naming what "this" is or where to look — Phase 1's fix is fully specified (including which two line ranges disagree and why one is trusted more); Phase 3's biggest risk (unverifiable pipelining correctness) is named explicitly rather than hidden behind "port carefully." ✓

**Concrete finding, not just a plan:** Phase 1 isn't a generic "review the fragment mapping" task — it reports an actual suspected bug (row/col transposition) found by diffing `gemm_rdna_wmma.py:141-152` against `rdna_f16_gemm.py:337-338` during research for this plan, with the exact conflicting formulas quoted, so the executing agent starts from a concrete hypothesis instead of re-deriving the ISA layout from zero. ✓

**Honesty about validation limits:** Every "Done criteria" in this plan is IR-level (MLIR verification, ISA-text grep, host-side bijection check) rather than a fabricated claim of test-passing or benchmarked performance, consistent with the fact that this box cannot execute RDNA code. ✓
