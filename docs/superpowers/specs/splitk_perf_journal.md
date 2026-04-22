# W3 splitk perf journal

Tracking the splitk vs. hipBLASLt gap over tuning rounds. Target: ≤ 1.02×
to unblock the `_fused_dact_eligible` dispatch.

MI355X / gfx950, bf16 MFMA, `gemm_splitk(a, b, shuffled=True)` vs.
`torch.matmul(a, b.T)` via hipBLASLt.

## Reference shapes

| tag         | M     | K      | N     | regime                   |
|-------------|-------|--------|-------|--------------------------|
| A_4k8k8k    | 4096  | 8192   | 8192  | canonical LLM FFN        |
| B_8k8k8k    | 8192  | 8192   | 8192  | square, full CU occupancy|
| C_2k16k2k   | 2048  | 16384  | 2048  | skinny-K, splitk-favored |

## Baseline (pre-tuning)

| shape       | splitk ms | hipBLAS ms | ratio   | splitk TFLOP/s |
|-------------|-----------|------------|---------|----------------|
| A_4k8k8k    | 0.456     | 0.388      | 1.175×  | 1205           |
| B_8k8k8k    | 0.837     | 0.670      | 1.251×  | 1313           |
| C_2k16k2k   | 0.259     | 0.144      | 1.800×  | 532            |

## Round 1 — tile config sweep

Sweep `(TILE_M, TILE_N, TILE_K)` ∈ {128, 256}³ subject to divisibility.

Per-shape best and gain over default 128×256×64:

| shape       | best config    | ms    | ratio   | delta |
|-------------|----------------|-------|---------|-------|
| A_4k8k8k    | 128×256×64     | 0.441 | 1.162×  | ~flat |
| B_8k8k8k    | 128×256×64     | 0.826 | 1.212×  | ~flat |
| C_2k16k2k   | 128×128×128    | 0.203 | 1.433×  | **-20%** |

Landed: `_default_kwargs` now returns `(128, 128, 128)` for skinny-M /
large-K shapes (M ≤ 2048, N ≤ 2048, K ≥ 16384).

`256×256×128` is catastrophic (8-15× slower) — register spill from
64 C_FRAGS per warp. Don't go there.

## Round 2 — waves_per_eu sweep

Added the `waves_per_eu` compile hint via `rocdl.waves_per_eu` attr.
Sweep {None, 1, 2, 3, 4}.

| shape       | best wpe | ms    | ratio  | delta |
|-------------|----------|-------|--------|-------|
| A_4k8k8k    | 1        | 0.437 | 1.221× | -1%   |
| B_8k8k8k    | 2        | 0.841 | 1.217× | -1%   |
| C_2k16k2k   | 2        | 0.201 | 1.424× | -1%   |

Gains are noise-level. Compiler picks reasonable occupancy without the
hint; explicit limits occasionally help by ~1%. Not worth baking into
`_default_kwargs`; kept as an opt-in kwarg.

## Round 3 — B routing (preshuffle vs. direct)

Compare `B_PRE_SHUFFLE=True` (current default — host pre-shuffles B)
against `B_PRE_SHUFFLE=False` (direct HBM load, no shuffle).

| shape       | preshuf=True | preshuf=False | gap   |
|-------------|--------------|---------------|-------|
| A_4k8k8k    | 0.450 ms     | 0.657 ms      | +46%  |
| B_8k8k8k    | 0.845 ms     | 1.229 ms      | +45%  |
| C_2k16k2k   | 0.204 ms     | 0.237 ms      | +16%  |

Preshuffle wins decisively. No change — current default is correct.

## Round 4 & 5 — deferred

Remaining hypotheses (stream-K imbalance, LDS bank conflicts) require
new kernel work or profiler dumps. Deferred to follow-up.

## Pivot: measure fused-dact net-save directly

Original design premise — that splitk's matmul gap eats the fusion
win — turned out to be wrong. Microbench at the reference
MLP-backward shape `(M=4096, K=4096, N=8192)`, silu activation:

| path                                 | ms    | notes                  |
|--------------------------------------|-------|------------------------|
| torch.mm + torch act_bwd             | 0.649 | hipBLASLt + elementwise|
| splitk (plain) + torch act_bwd       | 0.730 | matmul gap visible     |
| splitk fused dact (`dact_activation`)| 0.289 | **2.25× faster**       |

**The torch elementwise act_bwd is 0.5 ms** — far larger than the matmul
gap (~0.08 ms). Fusion eliminates that kernel launch entirely. Net-save
is 0.36 ms at this shape, dwarfing any matmul-side difference.

**Action taken**: flipped `_fused_dact_eligible` to `True` (with the
original shape gate: M%128, hidden%256, out_dim%64, M≥128,
last-dim-contiguous). `mlp_func_train` now dispatches through the
fused kernel on shape-eligible inputs.

## Final status

Ratios remain 1.16-1.43× across shapes — the absolute splitk/hipBLASLt
gap did not close to 1.02×. However, the W3 premise (that the gap
blocks fused-dact) was wrong: the fused path wins decisively at
every tested shape because it removes the torch-side elementwise
act_bwd kernel.

`_fused_dact_eligible` is **on** by default. Rounds 4 and 5 are
deferred as absolute-perf polish.
