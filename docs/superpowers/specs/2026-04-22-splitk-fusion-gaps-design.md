# splitk fusion gaps — design

**Date**: 2026-04-22
**Branch**: `wip/amd-flydsl-port`
**Scope**: three remaining gaps blocking full training-path fusion in `quack/amd/gemm_gfx950_splitk.py`.

## Context

`quack/amd/gemm_gfx950_splitk.py` is the CDNA4 splitk MFMA kernel we vendored from
FlyDSL `hgemm_splitk.py`, with a fused epilogue layer added for bias /
activation / gated / dact. It ships the forward-path fusions
(`linear`, `linear_gated`, `linear_mxfp8`, `linear_cross_entropy`) and their
autograd wrappers. Three gaps remain before the backward path is fully
fused on AMD:

1. **W1 — `gemm_dgated` kernel**. No AMD-side fused gated-activation backward
   exists. `mlp_func_train` backprop for swiglu/reglu/geglu/glu currently
   falls through torch autograd for the gated-activation step, costing an
   extra `(M, 2*hidden)` HBM round-trip per layer.
2. **W2 — split-K>1 fused epilogue**. Today `_compile_hgemm_kernel` forces
   `SPLIT_K=1` whenever any epilogue is set, because atomic-fadd partials
   are non-distributive over bias/activation/gated/dact. Large-K skinny
   shapes (common in LLM residual branches) lose the split-K tile-balance
   win.
3. **W3 — close the splitk / hipBLASLt gap**. Plain splitk matmul runs
   ~1.07× hipBLASLt at M=4096, hidden=8192, so fused-dact net-saves less
   than the gap loses. `_fused_dact_eligible` returns `False` by default.
   Closing the gap unblocks the fused-dact dispatch.

## Architecture

All three workstreams land inside `quack/amd/gemm_gfx950_splitk.py` and its
autograd consumers (`quack/amd/linear_training.py`). No new files; the
kernel builder grows two compile-time branches (W1, W2) and the autotune
table grows a few entries (W3). Tests go into
`tests/amd/test_linear.py` (already ~1478 tests) and a new
`tests/amd/bench_splitk_profile.py` for W3 microbench.

```
quack/amd/gemm_gfx950_splitk.py
  ├── _compile_hgemm_kernel(..., gate_type, dact_activation, split_k_mode)
  │     ├── forward:   W1 adds dact on gate+up pair path
  │     ├── epilogue:  W2 adds last-partial signal / two-launch branches
  │     └── tile menu: W3 adds tuned configs
  ├── gemm_splitk(...)                  ← public entry
  └── interleave_gated_weight(w)        ← host helper (shared by W1)

quack/amd/linear_training.py
  ├── _fused_dact_eligible(...)         ← W3 flips to True after gap closes
  ├── linear_act_func                   ← W1 plugs gemm_dgated call here
  └── mlp_func_train                    ← W1 wires swiglu/reglu/geglu/glu bwd
```

---

## W1 — `gemm_dgated` kernel (~200 LoC)

### Problem

For `y = act(gate) * up` with weight `w_gate_up` of shape
`(2*hidden, in_features)` interleaved as `(gate_0, up_0, gate_1, up_1, ...)`,
the backward needs:
- `dgate = act'(gate) * up * dy`          — the "dpreact-gate" half
- `dup   = act(gate) * dy`                — the "dpreact-up" half
- `dx    = d(pre_interleaved) @ w_gate_up`

NVIDIA reference: `GemmDGatedMixin` in `quack/gemm_dgated.py` co-emits
`dpreact_interleaved = (dgate_0, dup_0, dgate_1, dup_1, ...)` from the same
epilogue kernel that produces `postact = act(gate) * up` (the forward
output, recomputed for activation memory).

### Design

Compile-time flag `_IS_DGATED` on `_compile_hgemm_kernel`. The GEMM
body is unchanged — the kernel computes the upstream gradient
`acc = dz @ w_down` of shape `(M, hidden)`, where `dz` is the
grad-of-MLP-output and `w_down` is the down-projection. The new epilogue
expands each scalar in `acc` into an interleaved pair in `dpreact`
`(M, 2*hidden)` using the saved `preact` (gate/up pair).

Epilogue (replaces the plain write-back when `_IS_DGATED`). `tile_n`
below is the acc tile (hidden dim); each acc scalar emits 2 dpreact
values:

```python
# acc holds dz @ w_down; preact (gate, up pair) loaded from extra input.
# Emit dpreact_interleaved and (optionally) postact in one pass.
for c in range(tile_n):          # hidden dim
    gate = preact[2*c]
    up   = preact[2*c + 1]
    dy   = acc[c]                # shared scalar for this pair
    # ---- dpreact (interleaved: dgate at 2c, dup at 2c+1) ----
    dpreact[2*c]     = act_fwd_prime(gate) * up * dy    # = dgate
    dpreact[2*c + 1] = act_fwd(gate)            * dy    # = dup
    # ---- postact (optional, for gradient checkpointing recompute) ----
    postact[c] = act_fwd(gate) * up
```

Writes: `dpreact` full `(M, 2*hidden)`; `postact` `(M, hidden)` when
`_EMIT_POSTACT` is set.

Activation primes:
- `swiglu`: `sigmoid(g) + g * sigmoid(g) * (1 - sigmoid(g))` and `sigmoid(g)`.
  Compute `s = sigmoid(g)` once, reuse for both `act_fwd` (= `s * g`) and
  `act_fwd_prime` (= `s * (1 + g - s*g)`).
- `reglu`: `act_fwd = max(g, 0)`, `act_fwd_prime = (g > 0) ? 1 : 0`.
- `geglu`: tanh-approx gelu + its derivative — reuse
  `_gelu_tanh_approx` helpers already in `quack/amd/activation.py`.
- `glu`:   `act_fwd = sigmoid(g)`, `act_fwd_prime = s * (1 - s)`.

The `postact` emission is gated by a second compile-time flag
`_EMIT_POSTACT` because `mlp_func_train` memoises `postact = act(gate)*up`
on the forward pass and only needs `dpreact` on backward. The flag lets
us skip the extra store and halve HBM write traffic when `postact` is
available.

### Public API

```python
def gemm_splitk(
    ...,
    dgated_gate_type: Optional[str] = None,      # NEW: enables dgated path
    dgated_preact: Optional[Tensor] = None,      # (M, 2*hidden) saved from fwd
    dgated_emit_postact: bool = False,           # NEW: also write postact
    dgated_postact_out: Optional[Tensor] = None, # (M, hidden) if emit_postact
):
    # When dgated_gate_type is set:
    #   A = dz,          shape (M, out_features)
    #   B = w_down,      shape (out_features, hidden)  [NT layout]
    #   returns: dpreact (M, 2*hidden), and optionally postact (M, hidden).
```

Consumer in `mlp_func_train.backward`:

```python
# Backward of y = act(gate) * up, then z = y @ w_down.T
#   dy    = dz @ w_down              ← matmul we fuse
#   dpreact = d(act fusion)(dy, preact)
dpreact = gemm_splitk(
    dz, w_down, dgated_gate_type="swiglu",
    dgated_preact=saved_preact,
    dgated_emit_postact=False,   # postact saved on fwd
)
# Followed by regular matmuls:
#   dx         = dpreact @ w_gate_up
#   dw_down    = dy.T @ y       (y = postact)
#   dw_gate_up = dpreact.T @ x
```

### Tests

Add to `tests/amd/test_linear.py`:
- `test_gemm_dgated_{swiglu,reglu,geglu,glu}` — reference = torch autograd
  chain, tolerance matches the existing gated-forward tests (bit-exact for
  swiglu/reglu, `atol=0.5,rtol=0.01` for geglu/glu bf16 rounding).
- `test_mlp_func_train_fused_dgated` — end-to-end MLP backward checks
  that `mlp_func_train` produces same grads as torch autograd reference
  and uses the fused dgated path (check by patching or a dispatch counter).

### Estimated LoC

~200 LoC:
- 40 LoC — epilogue branch inside `_compile_hgemm_kernel`
- 30 LoC — activation prime helpers (most reusable from existing activation.py)
- 30 LoC — public API surface in `gemm_splitk`
- 50 LoC — consumer wiring in `linear_training.py`
- 50 LoC — tests

---

## W2 — split-K>1 fused epilogue path

### Problem

Today in `_compile_hgemm_kernel`:
```python
if has_bias or activation or _IS_GATED or _IS_DACT:
    SPLIT_K = 1   # forced
```
Because multi-CU partials are combined via `atomic_fadd` on the output
tensor, and `bias(a + b) ≠ bias(a) + bias(b)` etc., we can't apply the
epilogue per-partial.

Skinny-tall shapes (`M=16384, K=8192, N=1024`) leave CUs idle with
SPLIT_K=1 and lose 20–40% on the matmul itself before we even talk
about fusion.

### Design — try both, benchmark, dispatch per-shape

**Variant (a) — last-partial counter (in-kernel)**:
- Allocate `ctr: int32(GRID_M * GRID_N)` once, zeroed.
- Each CU computing a tile-partial increments `ctr[tile_idx]` via
  `atomicAdd` and compares to `SPLIT_K - 1`.
- The winning CU (the last partial) applies the fused epilogue as it
  stores; all other CUs do plain atomic-fadd store.
- Pro: one kernel launch, zero extra HBM traffic.
- Con: divergent control flow (last-partial CU does more work), needs a
  `gpu.barrier` + `__threadfence` to publish the partials before the
  winner reads. Complicates register pressure on the winning path.

**Variant (b) — two kernel launches**:
- Launch 1: plain splitk matmul with `SPLIT_K>1`, atomic-fadd into an
  fp32 scratch of shape `(M, N)`.
- Launch 2: tiny standalone epilogue kernel that reads the scratch,
  applies bias/act/gated/dact, writes final dtype. One-pass elementwise;
  cheap.
- Pro: simple, each kernel stays small; epilogue kernel is reusable
  across different splitk configs.
- Con: extra scratch allocation (M*N*4 bytes) and an extra HBM round-trip
  for the scratch.

### Implementation

Single compile-time flag `_SPLIT_K_EPI_MODE ∈ {"none", "last_partial", "two_launch"}`
on `_compile_hgemm_kernel`. Default `"none"` preserves current
SPLIT_K=1 force. At public-API entry:

```python
def _pick_splitk_epi_mode(M, N, K, has_epi) -> str:
    if not has_epi or SPLIT_K == 1:
        return "none"
    # Populated by benchmark table below.
    key = (M, N, K)
    return _SPLITK_EPI_MODE_TABLE.get(key, "two_launch")  # safe default
```

### Benchmark plan

Bench shapes covering the regimes where SPLIT_K>1 helps:

| M    | K     | N     | Expected SPLIT_K | Scenario                     |
|------|-------|-------|------------------|------------------------------|
| 1024 | 16384 | 1024  | 4                | Skinny-K router GEMM         |
| 2048 | 8192  | 2048  | 2                | Balanced mid                 |
| 4096 | 16384 | 4096  | 2                | LLM residual (typical)       |
| 16384| 4096  | 1024  | 1                | Control (no split-K)         |
| 512  | 32768 | 512   | 8                | Extreme-K                    |

Each shape × each epilogue ∈ {bias, silu, swiglu-gated, silu-dact}.
Bench script writes `_SPLITK_EPI_MODE_TABLE` entries automatically.

### Tests

- Extend existing `test_gemm_splitk_*` to include `SPLIT_K>1` with each
  epilogue variant, verified against torch reference at same tolerance
  as SPLIT_K=1.
- `test_splitk_epi_mode_dispatch` — assert the table-based dispatch
  picks the benchmarked-fastest mode for each reference shape.

### Estimated LoC

~300 LoC:
- 80 LoC — variant (a) `last_partial` branch (counter + in-kernel
  epilogue-last path)
- 60 LoC — variant (b) `two_launch` (scratch alloc + standalone
  epilogue kernel)
- 40 LoC — host-side dispatch + table
- 40 LoC — bench script (`tests/amd/bench_splitk_epi.py`)
- 80 LoC — test coverage

---

## W3 — close splitk / hipBLASLt gap

### Problem

`gemm_splitk(bf16, bf16 → bf16)` at `(M=4096, K=8192, N=8192)` runs
~1.07× hipBLASLt (7% slower). Fused-dact saves ~50us on the elementwise
act_bwd but the matmul is already 75us slower than hipBLASLt — net-loss
for the fused path, so `_fused_dact_eligible` is False.

Closing the gap means: at reference training shapes, plain
`gemm_splitk` matmul is within ±2% of hipBLASLt. Then fused-dact net-wins.

### Approach — 5 rounds of profile → fix → remeasure

Timebox: **5 rounds**. Each round ≤ 1 day of wall-clock work
(measure, hypothesis, patch, rerun, commit). If still >2% gap after round
5, document findings and keep `_fused_dact_eligible = False` with a
conservative shape gate (only enable at specific shapes where net-win is
proven).

### Round tooling

Reference shapes:
- Shape A: `M=4096, K=8192, N=8192` — canonical LLM FFN down-proj.
- Shape B: `M=8192, K=8192, N=8192` — square, most CUs occupied.
- Shape C: `M=2048, K=16384, N=2048` — skinny-K, splitk-favored.

Per-round protocol:
```
1. rocprofv3 --pmc SQ_VALU_MFMA_BUSY_CYCLES,SQ_WAVES,\
     SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS,FETCH_SIZE,WRITE_SIZE,\
     GRBM_GUI_ACTIVE -- python -m tests.amd.bench_splitk_profile
2. Compare per-counter ratio vs. hipBLASLt baseline.
3. Identify top-ranked hypothesis from the list below.
4. Land a patch, rerun, record delta.
5. Commit with "[AMD] splitk perf round N — <hypothesis>: Δ=+X%".
```

### Hypothesis priority list

Ordered by expected payoff × feasibility:

1. **Tile config** (round 1 most likely). Current default
   `tile_m=128, tile_n=256, tile_k=64` may be wrong for these shapes.
   Sweep (tile_m, tile_n, tile_k) ∈ product({128,256},{128,256},{32,64,128})
   — 12 configs per shape. FlyDSL disk cache keeps rerun cost to ~1 min
   after first compile.
   Expected gain: 2–5% if we're on a poor tile.
2. **waves_per_eu**. Currently defaults to `2`. For K-heavy GEMMs, `1` can
   hurt occupancy; `4` can hurt register pressure. Sweep {1,2,3,4} per tile.
   Expected gain: 1–3%.
3. **B-preshuffle coverage**. The preshuffle cost amortises at M≥1024
   but can be a net-loss at extreme-skinny M. Bench `preshuffle=False`
   path and dispatch by M threshold.
   Expected gain: 1–2% at small M.
4. **Stream-K tile-balance**. If GRID_M * GRID_N not evenly divisible
   by active CU count, some CUs finish early and stall. Try persistent-k
   scheduling using the existing `quack/amd/tile_scheduler.py` helpers.
   Expected gain: 2–4% at awkward GRID sizes.
5. **LDS bank conflicts**. The vendored XOR swizzle covers common cases
   but may conflict at specific tile_k. Inspect `SQ_LDS_BANK_CONFLICT`
   counter; if high, try alternate swizzle (xor32 vs xor16) or bump LDS
   double-buffer.
   Expected gain: 1–2%.

### Success criteria

After 5 rounds:
- **Primary**: `gemm_splitk(bf16) ≤ 1.02× hipBLASLt` at Shapes A and B.
  Flip `_fused_dact_eligible` to route shapes meeting this bar.
- **Fallback**: if only some shapes clear the bar, `_fused_dact_eligible`
  gates by `(M, K, N)` lookup table.
- **Defeat**: if no shape closes to 2%, document in the kernel module
  docstring and leave dispatch off.

### Tests / bench

- `tests/amd/bench_splitk_profile.py` — records per-round timings to
  `docs/superpowers/specs/splitk_perf_journal.md`.
- Existing tests cover correctness; no new correctness tests needed for W3.

### Estimated LoC

~50 LoC (tuning dict entries) + ~100 LoC bench script. No kernel
structural changes expected unless a round identifies one.

---

## Sequencing

W1 and W2 are independent; W3 unblocks the final dispatch flip in
`_fused_dact_eligible`. Recommended order:

1. **W1** (gemm_dgated) — self-contained, unblocks fused gated-activation
   backward regardless of W3 outcome.
2. **W2** (split-K>1 epilogue) — independent of W1. Doing W2 second lets
   us bench W2's two-launch variant on the dgated epilogue from W1 for
   extra coverage.
3. **W3** (perf close) — last because the autotune table from W2 + the
   dgated kernel from W1 give W3 more signal surface to profile against.

Each workstream lands as its own commit(s). After all three:
`_fused_dact_eligible` becomes a `(M, K, N)` gate populated by W3's
benchmark table.

## Risks / unknowns

- **W1 postact recompute**: if saving `postact` on fwd turns out to cost
  more HBM than re-emitting on bwd, drop `_EMIT_POSTACT` flag and always
  emit. Decision deferred to first benchmark.
- **W2 variant (a) register pressure**: the last-partial winner runs
  both matmul + epilogue; may spill on epilogues with many operands
  (dgated). If so, variant (b) becomes the default and (a) ships only
  for bias/activation.
- **W3 hitting a hardware ceiling**: if rocprofv3 shows MFMA_BUSY>90%
  already on round 1, matmul is MFMA-bound and the gap is a hipBLASLt
  scheduling advantage we can't easily match. In that case pivot W3 to
  "measure fused-dact net-save with W1/W2 changes, dispatch where
  net-positive regardless of absolute gap."
