# mxfp8 perf journal

Tracking the mxfp8 GEMM throughput on MI355X / gfx950. Starting state from
commit `0c6a7dd`. Target: close gap vs bf16 hipBLASLt (2× theoretical
advantage from fp8 density).

## Reference shapes

| tag              | M     | N     | K     | scenario              |
|------------------|-------|-------|-------|-----------------------|
| llama7b-ffn-up   | 4096  | 11008 | 4096  | Llama-7B FFN up       |
| llama7b-ffn-dn   | 4096  | 4096  | 11008 | Llama-7B FFN down     |
| square-8k        | 8192  | 8192  | 8192  | canonical big         |
| square-4k        | 4096  | 4096  | 4096  | mid size              |
| llama70b-ffn-up  | 8192  | 14336 | 4096  | Llama-70B FFN up      |
| batch-16k        | 16384 | 8192  | 8192  | larger M              |

## Baseline (CUDA-event timed, commit 0c6a7dd, cshuffle always off)

| shape              | mxfp8 TFLOP/s | bf16 hipBLASLt | ratio | % fp8 peak |
|--------------------|---------------|----------------|-------|-----------|
| 4096³              | 1354          | 1473           | 0.92× | 26%       |
| 8192³              | 1620          | 1694           | 0.96× | 32%       |
| llama7b-ffn-up     | 1516          | 1702           | 0.89× | 29%       |
| llama7b-ffn-dn     | 1396          | 1492           | 0.94× | 27%       |
| batch-16k          | 1496          | 1676           | 0.89× | 29%       |
| llama70b-ffn-up    | 1649          | 1797           | 0.92× | 32%       |

MI355X fp8 MFMA peak ≈ 5140 TFLOP/s; bf16 peak ≈ 2570 TFLOP/s.

**hipBLASLt bf16 reaches 66% of bf16 peak. Our mxfp8 reaches 32% of fp8 peak.**
So we're ~2× short on efficiency despite fp8 being the higher-density format.

## Round 1 — config sweep

Tried combinations of `(tile_m, tile_n, tile_k, cshuffle, waves_per_eu)` at
8192³ under CUDA-event timing. Results (best shown):

| config                          | ms    | TFLOP/s | % peak |
|---------------------------------|-------|---------|--------|
| 128×128×128 cs=False w=2 (old default) | 0.694 | 1583 | 30.8% |
| **128×128×128 cs=True w=2**            | 0.688 | 1599 | 31.1% |
| 128×128×128 cs=False w=None     | 1.182 | 931  | 18.1% |
| 256×128×128 cs=False w=2        | 5.482 | 201  | 3.9% (spill) |
| 128×256×128 cs=False w=2        | 4.255 | 258  | 5.0% (spill) |
| 256×256×128 cs=True  w=2        | 2.264 | 486  | 9.5% (occupancy=1) |
| 128×128×256 cs=False w=2        | 3.112 | 353  | 6.9% (LDS) |

Findings:
- cshuffle=True is +1% on large shapes — not a big win but free.
- Any larger tile register-spills or loses occupancy. 128×128×128 is the
  correct tile for this kernel.
- waves_per_eu=2 is a huge win (+42% vs compiler default). Already set.

## Round 1 correctness gating

`cshuffle=True` produces garbage at K=128 (single K-tile iteration —
prologue/epilogue doesn't fire correctly). Gated to `K >= 256` in
`_pick_config`. Correctness verified across the existing test matrix
(127 tests pass).

## rocprofv3 counter trace at 8192³ (cshuffle=False, w=2)

Per-dispatch averages across 30 reps:

| counter                     | value    | per-wave |
|-----------------------------|----------|----------|
| SQ_WAVES                    | 16384    | —        |
| SQ_INSTS_MFMA               | 1.68e7   | 1024     |
| SQ_INSTS_LDS                | 1.68e7   | 1024     |
| SQ_INSTS_VMEM               | 1.90e7   | 1160     |
| **SQ_INSTS_VALU**           | 1.09e8   | **6660** |
| SQ_LDS_BANK_CONFLICT        | 0        | 0        |
| SQ_VALU_MFMA_BUSY_CYCLES    | 5.37e8   | 32764    |
| VGPR / Accum_VGPR / SGPR    | 128 / 0 / 112 | — |
| LDS                         | 32 KB    | — |
| Scratch                     | 28 B     | minimal spill |

Key signal: **VALU is 6.5× MFMA count**. The per-MFMA overhead is
dominated by (a) the fp8 operand packs (`pack_i64x4_to_i32x8`) which
emit vector-from-elements + bitcast MLIR that lower to several VGPR
moves, and (b) the post-MFMA `math.fma` to multiply the block
accumulator by the combined scale and fold it into the global
accumulator.

Zero LDS bank conflicts, no scratch spill, VGPR fits — the kernel
body is the bottleneck, not the memory hierarchy.

## Where the 2× gap lives

Empirically:
- MFMA issues per dispatch are correct (match theoretical count).
- Compute cycles busy ≈ Python-timed duration minus launch overhead.
- So the dispatch is ~50% of its active time doing non-MFMA work.
  (1024 MFMAs × ~16 cycles each = 16384 busy cycles; measured total
  busy = 32764 cycles per wave. Half is MFMA, half is VALU/other.)

Closing the 2× gap would require rewriting the vendored
`blockscale_preshuffle_gemm` kernel body to overlap MFMA with the
scale FMAs and reduce per-MFMA instruction density. That's real
kernel-authoring work — not a config sweep.

## Decision

Round 1 landed a +1% win from cshuffle=True at K≥256.

Further rounds were **not run** — the remaining gap is in the kernel
body, and rewriting a 884-line vendored FlyDSL kernel is out of scope
for a perf-tuning pass. If AMD / FlyDSL ships an improved
`blockscale_preshuffle_gemm` upstream, re-vendor it.

Proceeding to the epilogue-fusion workstream, which saves ~15-125 μs
per call (launch overhead + elementwise kernel) — net bigger than
the 1% kernel win at shapes up to ~4096³.

## Epilogue fusion landing (2026-04-22)

Wired bias + activation into the vendored kernel's direct-store
writeback path. Activations supported: relu, relu_sq, silu,
gelu_tanh_approx. Bias is an (N,) f32 tensor, fused in f32 before
trunc_f → bf16 store. cshuffle is forced off when has_epi is set
(the LDS-staged path would need duplicate fusion sites).

Measured savings vs `mxfp8_gemm(...); F.silu(out + bias)`:

| shape                  | unfused ms | fused ms | saved           |
|------------------------|-----------:|---------:|-----------------|
| 4096³                  |      0.142 |    0.126 |  15 μs (10.9%)  |
| 8192³                  |      0.936 |    0.777 | 160 μs (17.1%)  |
| 4096×11008×4096 (FFN)  |      0.409 |    0.321 |  88 μs (21.5%)  |

At typical MoE / FFN shapes the fusion is net +11–22%. Much bigger
than the 3% estimate I had before benching — the elementwise-kernel
launch overhead dominates at smaller shapes, and the HBM
round-trip dominates at 8k². Both go away with in-kernel fusion.
