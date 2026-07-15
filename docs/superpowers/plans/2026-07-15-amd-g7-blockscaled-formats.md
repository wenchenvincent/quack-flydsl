# Blockscaled GEMM Format Expansion (MXFP4 / NVFP4 / MXFP8-e5m2) — Implementation Plan (Gap G7)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Design-level; expand each phase to TDD tasks before executing.

**Goal:** Extend the AMD blockscaled GEMM beyond its current MXFP8-e4m3-only, non-standard 128-block/f32-scale scheme to cover MXFP4 (e2m1 + e8m0, vec 32), and design (not necessarily implement in this pass) NVFP4 (e2m1 + e4m3, vec 16) and MXFP8-e5m2.

**Architecture:** The existing kernel (`_fly_blockscale_preshuffle_gemm.py`, vendored from FlyDSL) already drives `rocdl.mfma_scale_f32_16x16x128_f8f6f4` — an instruction family that natively supports fp8/fp6/fp4 operands via `cbsz`/`blgp` format-select fields, and a *hardware* e8m0-style packed scale operand that the current AMD port deliberately does not use (it passes a neutral scale and applies the real per-block scale via a post-MFMA software FMA instead). MXFP4 support is the highest-value, most tractable extension: same instruction, same pipeline shape, different `cbsz`/`blgp` value and a 2×-denser A/B packing (2 elements/byte instead of 1). NVFP4 and MXFP8-e5m2 get full design sections but land as Phase-2/3 follow-ons given the additional packing/scale-format work each requires.

**Tech Stack:** `quack/amd/gemm_gfx950_blockscaled.py`, `quack/amd/_fly_blockscale_preshuffle_gemm.py`, `quack/amd/_fly_mfma_preshuffle_pipeline.py`, `quack/amd/mxfp8_ops.py`, NVIDIA reference `quack/blockscaled_gemm_utils.py` + `quack/mx_utils.py`, `rocdl.mfma_scale_f32_16x16x128_f8f6f4`, gfx950 (CDNA4).

---

## Background the engineer needs

**What "MXFP8" currently means in this codebase is not the OCP MX standard.** `_fly_blockscale_preshuffle_gemm.py:9` documents "Per-block scaling (ScaleBlockM=1, ScaleBlockN=128, ScaleBlockK=128)" — a 128-element block with an **f32** scale (`quack/amd/gemm_gfx950_blockscaled.py:16`: `scale_a: (K//128, M) f32`, `scale_b: (N//128, K//128) f32`). The real OCP MX standard (and NVIDIA's `quack/mx_utils.py:39` `to_mx`) uses **block_size=32** with an **e8m0** (`torch.float8_e8m0fnu`) scale. So today's "MXFP8" kernel is really a custom coarse-grained fp8-with-f32-scale GEMM that happens to use e4m3 data — it is *not* drop-in compatible with `quack.mx_utils.to_mx`'s output. This distinction matters for every format added in this plan: each new format needs a decision of "match the AMD kernel's existing 128-block/f32-scale convention" vs "match the real MX/NVFP4 spec (32-block/e8m0, or 16-block/e4m3 for NVFP4)". This plan recommends the latter (real spec) for new formats, since (a) it's what `quack/blockscaled_gemm_utils.py` already implements (reusable, not reinvented) and (b) the whole point of "expanding formats" is interop with the standard MX ecosystem, not perpetuating a bespoke scheme. This is a **larger scope decision than a first read of the gap suggests** — flagged explicitly rather than silently assumed.

**The hardware scale operand exists and is currently unused.** `rocdl.mfma_scale_f32_16x16x128_f8f6f4`'s Python wrapper (`flydsl/expr/rocdl/__init__.py:169-183`) takes `[a, b, c, cbsz, blgp, opselA, scaleA, opselB, scaleB]`. The current kernel call site (`_fly_blockscale_preshuffle_gemm.py:569-573`):
```python
block_accs[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
    mfma_res_ty,
    [a128, b128, block_accs[acc_idx],
     0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
)
```
passes `cbsz=0, blgp=0` (both operands FP8/E4M3 format) and `scaleA=scaleB=0x7F7F7F7F` — four packed e8m0 bytes each equal to `0x7F` (bias 127 → unbiased exponent 0 → scale factor 2⁰=1, i.e. **neutral**). The real per-block scale is instead applied in software via `math_dialect.fma(block_accs[acc_idx], combined_scales[mi][ni], current_global[acc_idx])` (`_fly_blockscale_preshuffle_gemm.py:604-612`), where `combined_scales = scale_a[block] * scale_b[block]` is loaded as f32 tensors (`load_scales_for_tile`, lines 496-534) and FMA'd in after every K-sub-block's raw (unscaled) MFMA accumulation. This is a deliberate, working design — not a bug — and this plan's default recommendation is to **keep it** for the new formats too (see Design Decision below), rather than switch to true hardware e8m0 scale application.

**`cbsz`/`blgp` format encoding.** Confirmed via two independent sources in the repo: (1) the WMMA sibling instruction's documented `fmtA`/`fmtB` parameter (`flydsl/expr/rocdl/__init__.py:203-207`, `wmma_scale_f32_16x16x128_f8f6f4`): `0=FP8/E4M3, 1=FP8/E5M2, 2=FP6/E2M3, 3=FP6/E3M2, 4=FP4/E2M1`; (2) this matches AMD's public CDNA3/CDNA4 ISA documentation for the `V_MFMA_SCALE_F32_16X16X128_F8F6F4` CBSZ/BLGP field, which the MFMA and WMMA variants share by design (same "f8f6f4" instruction family name, same OCP-MX-driven format menu). **This needs a one-time hardware confirmation** (Phase 0) before committing MXFP4's `cbsz=4` design, since the WMMA docstring is technically for a different (gfx1250/RDNA) instruction, not the gfx950 MFMA one this plan targets — the encodings are very likely identical (same instruction family, same ISA generation of format menu) but "very likely" is not "verified."

**Existing packing precedent for 4-bit data.** `_fly_mfma_preshuffle_pipeline.py` already has a 4-bit weight-quantization packing/unpacking path: `load_b_raw_w4a16` (line 252), `_int4_to_bf16x4_i64_gfx950` (line 307), `unpack_b_w4a16` (line 363). These handle **int4** codes (W4A16 weight-only quantization), not **e2m1** float4 codes — the semantics differ (int4 unpacks to an integer value scaled by a per-group float, not a direct float4→float lookup) — but the *packing mechanics* (2 values per byte, low/high nibble, `kpack_bytes=8` variant of `make_preshuffle_b_layout`) are directly reusable scaffolding for MXFP4/NVFP4's 2-elements-per-byte A/B layout. Read these three functions before designing MXFP4's preshuffle.

**Quantizers: reuse, don't reinvent.** `quack/blockscaled_gemm_utils.py` (NVIDIA-side, already ported from torchao, **pure PyTorch, no CUDA dependency**) has exactly the quantizers needed:
- `to_mx(x, block_size=32)` → `quack/blockscaled_gemm_utils.py:39` — MXFP8-e4m3, real spec (block 32, e8m0 scale). Not what the AMD kernel needs for its *current* e4m3 path (which is block 128/f32), but directly reusable for a real-MX-spec MXFP8 variant if this plan's format decision (below) is adopted for the existing format too.
- `to_mxfp4(x, block_size=32)` → `:153` — MXFP4, block 32, e8m0 scale, returns packed uint8 (2 nibbles/byte, low-then-high) + `float8_e8m0fnu` scale.
- `to_nvfp4(x, block_size=16, per_tensor_scale=None)` → `:193` — NVFP4, block 16, `float8_e4m3fn` scale, optional f32 per-tensor scale on top.
- `_f32_to_floatx_unpacked` (`:87`) and `_pack_uint4` (`:132`) are the shared low-level primitives all three float4 formats need — reuse directly, do not reimplement.

These are **host-side torch quantizers** (like `quack/amd/mxfp8_ops.py:quantize_mxfp8` today) — no FlyDSL kernel work needed to produce the quantized operands; only the *GEMM kernel's* consumption of e2m1-packed A/B and its scale layout needs new kernel-side work.

**Preshuffle layout (`make_preshuffle_b_layout`, `_fly_mfma_preshuffle_pipeline.py:78-120`).** Currently hardcoded for `elem_bytes ∈ {1, 2}` (fp8 or fp16-ish). e2m1 packed-2-per-byte data is `elem_bytes=1` at the *storage* level (each byte holds 2 logical elements) but the K-dimension packing math (`kpack_elems = kpack_bytes if elem_bytes==1 else kpack_bytes//elem_bytes`) needs a third case: for fp4, 16 storage bytes = 32 logical K-elements, not 16. This is a real, scoped change to `make_preshuffle_b_layout` and `load_b_pack_k32`, not a parameter tweak — budget real engineering time here, not "just pass elem_bytes=0.5."

---

## Design Decision (make this explicitly, before writing kernel code)

**Question:** should MXFP4 (and future NVFP4/MXFP8-e5m2) use the AMD kernel's existing bespoke 128-block/f32-scale convention, or the real OCP MX / NVFP4 spec (32-block/e8m0 for MXFP4, 16-block/e4m3 for NVFP4)?

**Recommendation: real spec**, for these reasons:
1. `quack/blockscaled_gemm_utils.py`'s quantizers already implement the real spec and are directly reusable — matching them means zero new quantizer code.
2. The whole motivation for "expand formats" is presumably eventual interop with MX-standard checkpoints/tooling (this is the point of MXFP4/NVFP4 existing as *named, standardized* formats rather than another bespoke scheme).
3. The *scale application* mechanism (software FMA post-MFMA, neutral hardware scale) is orthogonal to the block size / scale dtype — nothing about switching to 32-block/e8m0 forces switching to hardware scale application. Keep the proven working "neutral hardware scale + software FMA" mechanism; only change the scale *granularity and storage dtype* the software path consumes (`load_scales_for_tile` needs an `sb_per_tile` recompute for a 32-element block instead of 128, and an e8m0→f32 conversion — one bit-shift, matching `mx_utils.py`'s `e8m0_to_f32`-equivalent — before the FMA).

**Consequence:** the existing e4m3 MXFP8 path's *external contract* (`scale_a`/`scale_b` as f32 tensors, 128-element blocks) should **not** be broken by this plan — it's a shipped, tested API (`tests/amd/test_gemm_blockscaled.py`). Ship MXFP4/NVFP4/MXFP8-e5m2 as new format options in the same kernel family with their own (real-spec) scale contracts, rather than retrofitting the existing e4m3 path's granularity. If someone later wants a real-spec MXFP8-e4m3 too, that's an explicit separate follow-up, not silently bundled here.

---

## Phase 0 — Investigation: confirm `cbsz=4` on the gfx950 MFMA (not just the gfx1250 WMMA)

**Goal:** de-risk the single load-bearing assumption everything else depends on.

- [ ] Write a scratch kernel: single 16×16×128 MFMA-scale tile, `cbsz=4, blgp=4` (both operands FP4/E2M1), feeding a hand-picked small e2m1-packed A/B pattern (e.g. all-zeros except one known nonzero pair) and neutral scale bytes. Compare the f32 accumulator against a hand-computed expected value using the FP4 E2M1 code table already present in the repo (`FlyDSL/tests/kernels/utils/fp4_utils.py:mxfp4_to_f32`, values `[0, 0.5, 1, 1.5, 2, 3, 4, 6]` + sign).
- [ ] If `cbsz=4` does *not* produce the expected FP4 semantics on gfx950 (e.g. the MFMA-scale instruction on CDNA4 only implements a subset of the f8f6f4 format menu, unlike the WMMA on gfx1250), this changes the whole plan's premise — stop and re-scope (e.g. fp4 might need a different, non-`f8f6f4` gfx950 instruction, or might not be supported on CDNA4 MFMA at all despite being documented for the WMMA sibling on RDNA). Document the finding either way.
- [ ] While in there, also confirm the **hardware scale operand path** works as expected for a simple case (non-neutral `scaleA`/`scaleB` bytes producing the documented 2^(exp-127) multiplier) — not required for this plan's chosen software-FMA approach, but useful ground-truth in case a future plan wants to switch to hardware scaling for performance.

**Done when:** a passing scratch-kernel test demonstrates FP4×FP4 → f32 accumulation matching the E2M1 code table, on real gfx950 hardware.

## Phase 1 — Design: MXFP4 kernel plan

**Files to touch:** `quack/amd/_fly_mfma_preshuffle_pipeline.py` (preshuffle layout + load helpers), `quack/amd/_fly_blockscale_preshuffle_gemm.py` (kernel body — `cbsz`/`blgp`, scale-block-size parameterization, A/B packing width), `quack/amd/gemm_gfx950_blockscaled.py` (new public entry point), `quack/amd/mxfp8_ops.py` or a new `mxfp4_ops.py` (quantizer wiring — likely just re-exporting `quack.blockscaled_gemm_utils.to_mxfp4` plus the AMD-specific scale-layout transpose/reshape).

- [ ] **Packing width.** `make_preshuffle_b_layout` (`_fly_mfma_preshuffle_pipeline.py:78-120`) needs an `elem_bits=4` case (new parameter, alongside today's implicit 8/16-bit handling via `elem_bytes`) so `kpack_elems` for a 16-byte kpack becomes 32 (not 16) logical FP4 elements. `load_b_pack_k32` (`:386`) needs the matching change. Do not attempt to force this through the existing `elem_bytes ∈ {1,2}` parameter — add an explicit `elem_bits` or `pack_ratio` parameter and update both call sites in `_fly_blockscale_preshuffle_gemm.py` together (`load_b_pack_k32(...)` call at line 287-295, and `make_preshuffle_b_layout` usage — note the *current* kernel builds `layout_b` inline at lines 199-209 rather than calling `make_preshuffle_b_layout`; check both).
- [ ] **A/B storage dtype.** `torch.float4_e2m1fn_x2` is the packed torch dtype (2 codes/byte) used by NVIDIA's `blockscaled_gemm_utils.py` (`_pack_fp4_e2m1fn_codes`). Decide whether the AMD kernel accepts this dtype directly (bitcast to raw `uint8`/`T.f8` storage inside the kernel, same as today's e4m3 `T.f8` buffer treatment) or requires a plain `uint8` tensor — recommend accepting `torch.float4_e2m1fn_x2` at the public API boundary (`mxfp4_gemm(...)`) and `.view(torch.uint8)` internally before the buffer-tensor call, mirroring how e4m3 today is `torch.float8_e4m3fn` at the API and raw bytes inside the kernel.
- [ ] **Scale block size.** `scale_block_k` today defaults to 128 (`compile_blockscale_preshuffle_gemm(..., scale_block_k: int = 128)`, `_fly_blockscale_preshuffle_gemm.py:53`) — for MXFP4, this needs to become 32 (per the Design Decision above). `sb_per_tile = tile_k // scale_block_k` (line 91) and `ku_per_sb = scale_block_k // 64` (line 92) both recompute automatically once `scale_block_k=32` is passed — but `ku_per_sb = 32/64 = 0` under integer division! This is a **real bug the current code would hit** if scale_block_k < 64 — the K-sub-block granularity (`ku_per_sb`) assumes each scale block spans at least one 64-element MFMA K-chunk. For a 32-element scale block, either (a) two scale blocks share one 64-wide MFMA K-chunk (need a `sb_per_ku` ≥ 1 case instead of `ku_per_sb`), or (b) the MFMA K-chunk width itself needs to shrink to 32 for FP4 tiles (plausible — FP4 packs 2× denser, so a K=128-bytes-per-instruction MFMA might naturally span 256 FP4 elements, not 128... verify in Phase 0). **This is exactly the kind of assumption Phase 0's scratch kernel should pin down** before this phase's arithmetic is trusted.
- [ ] **e8m0 scale ingestion.** `load_scales_for_tile` (`_fly_blockscale_preshuffle_gemm.py:496-534`) loads `scale_a`/`scale_b` as `T.f32` via `buffer_ops.buffer_load(..., dtype=T.f32)`. For e8m0 scales, either (a) convert to f32 host-side before the kernel call (simplest — one `.to(torch.float32)` after `to_mxfp4`'s `float8_e8m0fnu` output, exactly the "restore fp32 scale from biased exponent" arithmetic already implemented in `quack/mx_utils.py`'s... actually check `quack/blockscaled_gemm_utils.py` — it returns raw e8m0 codes, not pre-expanded f32; NVIDIA's own kernel presumably ingests e8m0 natively) or (b) load raw uint8 e8m0 bytes in-kernel and expand via a bitshift (`scale_f32 = (e8m0_byte << 23) as f32`, matching `FlyDSL/tests/kernels/utils/fp4_utils.py:e8m0_to_f32`). **Recommend (a) for the MVP** — host-side f32 expansion keeps the kernel-side scale-loading code byte-for-byte identical to today's e4m3 path (just change the *block size* constant, not the load logic), deferring in-kernel e8m0 decode to a perf follow-up. This trades a small host-side preprocessing cost for a much smaller kernel diff.

## Phase 2 — Implement: MXFP4

**Files:** modify `_fly_mfma_preshuffle_pipeline.py`, `_fly_blockscale_preshuffle_gemm.py` per Phase 1; new `quack/amd/mxfp4_ops.py` (quantizer re-export + AMD scale-layout adapter, mirroring `mxfp8_ops.py`'s `quantize_mxfp8`); extend `gemm_gfx950_blockscaled.py` with `mxfp4_gemm(...)`; new `tests/amd/test_gemm_mxfp4.py`.

- [ ] Wire `cbsz=4, blgp=4` into the kernel's MFMA-scale call site, gated on a new `ab_format: str = "e4m3"` (or similar) parameter to `compile_blockscale_preshuffle_gemm`, defaulting to today's behavior so the existing e4m3 path and its tests are untouched.
- [ ] Reference test: dequantize via `quack.blockscaled_gemm_utils`'s own primitives (`e8m0` expansion + the E2M1 code table) rather than hand-rolling a second dequant — check whether `blockscaled_gemm_utils.py` or `mx_utils.py` already exposes a `from_mx`/dequant helper; if not, the dequant-for-test-reference logic can reuse `FlyDSL/tests/kernels/utils/fp4_utils.py:mxfp4_to_f32` (already in the repo, already correct, used by FlyDSL's own gfx1250 fp4 tests).
- [ ] Tolerance: FP4 has only 3 bits of dynamic range within a sign+2m1 mantissa — expect much larger relative error than fp8; derive empirically (per-block quantization error dominates, not accumulation error), do not reuse the fp8 kernel's `atol`/`rtol` constants.

**Done when:** `pytest tests/amd/test_gemm_mxfp4.py -x` passes on real gfx950 hardware, existing `tests/amd/test_gemm_blockscaled.py` (e4m3 path) still passes unmodified (regression guard that the shared kernel-builder changes didn't break the existing format).

## Phase 3 — Design only: NVFP4 and MXFP8-e5m2 (do not implement in this pass)

- [ ] **NVFP4** (`cbsz`/`blgp` presumably still `4` — same E2M1 data format as MXFP4; the difference is entirely in the *scale*: block=16 instead of 32, `float8_e4m3fn` scale instead of e8m0, plus an optional f32 per-tensor scale layered on top (`to_nvfp4`'s `per_tensor_scale` param, `blockscaled_gemm_utils.py:193-232`)). Kernel-side, this means: (a) `sb_per_tile`/`ku_per_sb` recomputed for block=16 (even smaller than MXFP4's 32 — revisit the Phase-1 `ku_per_sb` granularity concern, now more acute), (b) scale load changes from e8m0-bitshift to e4m3-native decode (`arith.extf` from f8e4m3 to f32 — a standard float-extend, no bitshift trickery needed, actually *simpler* than e8m0), (c) an extra scalar multiply for the per-tensor scale, applied either in the software FMA step or as a post-kernel host-side scale (simplest: post-kernel, one `out *= per_tensor_scale` — avoids touching the kernel's inner loop at all for this term).
- [ ] **MXFP8-e5m2**: no packing-width change needed (still 1 byte/element, `elem_bytes=1`, same as e4m3) — purely a `cbsz=1` (per the format table) instead of `cbsz=0`. This is the *cheapest* of the three new formats to add, mechanically — flag as a candidate for pulling forward into Phase 2 if MXFP4's packing-width work turns out to be higher-risk/slower than expected, since e5m2 validates the `cbsz` format-select mechanism in isolation without also debugging new packing math.
- [ ] For both: write the same shape of test plan as Phase 2 (dequant-based reference using each format's exact-representable-in-f32 property for the data type, statistical/empirical tolerance for the scale quantization), but do not write the kernel code — this plan explicitly scopes implementation to MXFP4 only, per the task's stated priority ("Scope MXFP4 first").

---

## Self-Review

**Spec coverage:** Addressed what changes for fp4 packing (2 elements/byte — new `elem_bits`/preshuffle-layout work, not a parameter tweak, with the concrete `ku_per_sb` integer-division bug flagged as a real risk rather than glossed over), scale vec size (32 for MXFP4 vs 128 today — explicit Design Decision made and justified, not left implicit), e8m0-vs-f32 scale (recommended host-side f32 expansion for the MVP with a documented in-kernel-decode follow-up), the quantizer additions (recommended reusing `quack.blockscaled_gemm_utils.to_mxfp4`/`to_nvfp4` rather than reinventing — cites exact functions and line numbers), preshuffle layout changes (scoped to `make_preshuffle_b_layout`/`load_b_pack_k32`), and test strategy (dequant-based, reusing FlyDSL's own existing FP4 test utilities rather than writing new ones). The "keep software scale application or switch to hardware" question posed in the task is answered explicitly (keep software FMA; only the block-size/dtype of what feeds it changes) with reasoning, not left as an open question for the implementer to rediscover. MXFP4 is scoped first per the task's instruction; NVFP4/MXFP8-e5m2 get real design sections (not stubs) but are explicitly not implemented in this pass. ✓

**Placeholder scan:** Every phase step names the exact file and, where the current code lives, the exact function/line range to modify. The one item flagged as a genuine open unknown (whether `cbsz=4` behaves identically on the gfx950 MFMA instruction vs. the gfx1250 WMMA instruction whose docstring is the actual evidence source) is explicitly scoped as Phase 0's deliverable with a concrete verification method (scratch kernel + hand-computed E2M1 table), not silently assumed to transfer. The `ku_per_sb` integer-division-to-zero issue for a 32-element scale block is called out as a concrete bug the naive parameterization would hit, with two candidate fixes named. ✓

**Type consistency:** MXFP4 data path: `torch.float4_e2m1fn_x2` (packed 2/byte) at the public API → raw uint8/`T.f8` buffer storage in-kernel (mirrors today's e4m3fn → `T.f8` treatment) → f32 accumulator (unchanged — `mfma_res_ty = T.f32x4` throughout, MFMA-scale always accumulates in f32 regardless of A/B format) → bf16/f16 output (unchanged store path). Scale path: e8m0 (`float8_e8m0fnu`) at the quantizer boundary → f32 host-side expansion → f32 buffer load in-kernel (byte-for-byte reuse of today's `load_scales_for_tile` load logic, only the block-size constant changes) → f32 software FMA into the f32 accumulator (unchanged mechanism from the e4m3 path). No dtype is left ambiguous across a phase boundary. ✓

---

## Phase 0 RESULT (2026-07-15, verified on gfx950/MI355X)

Ran an in-register fp4 MFMA probe (no HBM load): built A/B fragments as
`vec<8 x i32>` all `0x22222222` (e2m1 nibble `0x2` = value 1.0), neutral
hardware scale (`0x7F7F7F7F`), `cbsz=blgp=4` (fp4 format select), zero
accumulator, then read lane-0's 4 f32 outputs.

**Result: C = 128.0** (all 4 outputs). This is exactly the K=128 dot of
ones, which confirms:
- The e2m1 encoding: nibble `0x2` decodes to 1.0. (0b0010 = sign0 exp01
  mant0 = 1.0 × 2^0.)
- `mfma_scale_f32_16x16x128_f8f6f4([a, b, c, cbsz=4, blgp=4, opselA=0,
  scaleA=0x7F7F7F7F, opselB=0, scaleB=0x7F7F7F7F])` computes a correct
  fp4×fp4 → f32 K=128 accumulate.
- The operand type is `vec<8 x i32>` (256 bits/lane) and the neutral
  scale `0x7F7F7F7F` (e8m0 bias, = 2^0) leaves values unscaled.

This de-risks the hardest part of G7 (the MFMA semantics + fp4 encoding
the earlier draft flagged as the key unknown). **Remaining fp4 work:**
(1) f32→e2m1 quantizer + 2-per-byte packing in torch (no torch fp4
dtype), (2) the HBM→fragment lane mapping for K=128 fp4 (each lane's
32 fp4 of A / B — likely the fp8-K=32 pattern scaled ×4, but needs the
same single-tile empirical check against a dequant reference), (3) real
per-block e8m0 scales via scaleA/scaleB instead of neutral. The probe
kernel is in scratchpad (`test_fp4_probe.py` pattern) for reuse.
