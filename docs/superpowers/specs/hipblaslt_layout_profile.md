# hipBLASLt per-layout profile — MI355X (gfx950)

**Date**: 2026-04-23.
**Hardware**: MI355X (gfx950 / CDNA4), wave64, 8 CU-dies.
**Software**: ROCk module 6.19.3, rocprofv3 1.0.0, PyTorch ROCm build.
**Purpose**: Set per-shape Tier-2 perf targets for the NN and TN kernels
(parent spec `2026-04-23-nn-tn-kernels-design.md` §3.1). Identifies
shapes where hipBLASLt leaves significant headroom (Tier-3 opportunities)
vs. shapes where it's essentially at hardware peak.
**Raw data**:
- `/tmp/layout_bench_{bf16,f16}.txt` — clean CUDA-event timing (authoritative for kernel μs / TFLOP/s)
- `/tmp/layout_profile_{bf16,f16}.txt` — rocprofv3 HW counters (authoritative for MFMA/VALU/VMEM/LDS)
- `rocprof-layout-profile/` (bf16) and `rocprof-layout-profile-f16/` — per-shape per-PMC-pass counter CSVs
**Bench harness**: `tests/amd/bench_hipblaslt_layouts.py`; PMC file: `tests/amd/pmc_counters.txt`.

## Methodology note

Kernel wall-time was measured **both** ways because rocprofv3's PMC multiplexing
(4 passes per dispatch) caused spurious per-dispatch time inflation on some
small-M shapes — most notably `4096x1024x2048 TN bf16` reports 2114 μs under
rocprofv3 vs 30 μs under clean CUDA-event timing (a 70× artefact). **We use
clean-timing μs/TFLOP/s throughout, and rocprofv3 only for counters.**

Dispatches-per-shape: 35 per bench run (5 warmup + 30 timed), min-of-timed
used for the wall-time. Counter values are averaged across all Cijk_*
dispatches matched in the rocprofv3 output.

## bf16 results

| shape                | role        | layout | kernel μs | TFLOP/s | MFMA     | VALU     | VMEM    | LDS      |
|----------------------|-------------|:------:|----------:|--------:|---------:|---------:|--------:|---------:|
| 2048 × 4096 × 1024   | MLP-fwd     | NT     |    22     |  773.9  |  1.05M   |  1.89M   |  215k   |  600k    |
| 2048 × 1024 × 4096   | MLP-dx      | NN     |    44     |  394.0  |  1.05M   |  3.73M   |  350k   |  989k    |
| 4096 × 1024 × 2048   | MLP-dw-down | TN     |    30     |  571.9  |  1.05M   |  3.77M   |  281k   |  1.31M   |
| 1024 × 4096 × 2048   | MLP-dw-up   | TN     |    30     |  579.6  |  1.05M   |  3.77M   |  281k   |  1.31M   |
| 8192 × 16384 × 4096  | MLP-fwd     | NT     |   679     |  1618.5 |  67.23M  |  77.86M  |  8.67M  |  16.92M  |
| 8192 × 4096 × 16384  | MLP-dx      | NN     |   656     |  1674.9 |  67.14M  | 104.32M  |  8.46M  |  16.83M  |
| 16384 × 4096 × 8192  | MLP-dw-down | TN     |   676     |  1625.5 |  67.17M  | 140.63M  |  8.53M  |  16.85M  |
| 4096 × 16384 × 8192  | MLP-dw-up   | TN     |   682     |  1612.7 |  67.17M  | 140.63M  |  8.53M  |  16.85M  |
| 32768 × 32768 × 8192 | MLP-fwd     | NT     | 10907     |  1612.9 |  1.07G   |  1.17G   | 136.5M  | 269.6M   |
| 32768 × 8192 × 32768 | MLP-dx      | NN     | 10996     |  1599.9 |  1.07G   |  1.65G   | 134.8M  | 268.9M   |
| 32768 × 8192 × 32768 | MLP-dw-down | TN     | 11078     |  1588.0 |  1.07G   |  2.19G   | 134.8M  | 268.7M   |
| 8192 × 32768 × 32768 | MLP-dw-up   | TN     | 11092     |  1586.0 |  1.07G   |  2.19G   | 134.8M  | 268.7M   |

## f16 results

| shape                | role        | layout | kernel μs | TFLOP/s | MFMA     | VALU     | VMEM    | LDS      |
|----------------------|-------------|:------:|----------:|--------:|---------:|---------:|--------:|---------:|
| 2048 × 4096 × 1024   | MLP-fwd     | NT     |    23     |  756.2  |  1.05M   |  2.01M   |  215k   |  600k    |
| 2048 × 1024 × 4096   | MLP-dx      | NN     |    43     |  401.0  |  1.05M   |  3.75M   |  350k   |  989k    |
| 4096 × 1024 × 2048   | MLP-dw-down | TN     |    31     |  562.2  |  1.05M   |  3.83M   |  281k   |  1.31M   |
| 1024 × 4096 × 2048   | MLP-dw-up   | TN     |    31     |  557.1  |  1.05M   |  3.83M   |  281k   |  1.31M   |
| 8192 × 16384 × 4096  | MLP-fwd     | NT     |   735     |  1495.0 |  67.23M  |  79.86M  |  8.67M  |  16.92M  |
| 8192 × 4096 × 16384  | MLP-dx      | NN     |   806     |  1363.7 |  67.11M  | 104.11M  |  8.46M  |  25.20M  |
| 16384 × 4096 × 8192  | MLP-dw-down | TN     |   934     |  1177.8 |  67.11M  | 145.24M  |  8.53M  |  25.24M  |
| 4096 × 16384 × 8192  | MLP-dw-up   | TN     |   938     |  1172.0 |  67.11M  | 145.24M  |  8.53M  |  25.24M  |
| 32768 × 32768 × 8192 | MLP-fwd     | NT     | 11481     |  1532.3 |  1.07G   |  1.18G   | 136.5M  | 269.6M   |
| 32768 × 8192 × 32768 | MLP-dx      | NN     | 13174     |  1335.4 |  1.07G   |  1.64G   | 134.8M  | 403.0M   |
| 32768 × 8192 × 32768 | MLP-dw-down | TN     | 15539     |  1132.1 |  1.07G   |  2.24G   | 134.8M  | 403.0M   |
| 8192 × 32768 × 32768 | MLP-dw-up   | TN     | 15643     |  1124.6 |  1.07G   |  2.24G   | 134.8M  | 403.0M   |

## Analysis

### Per-layout kernel-time comparison (bf16)

Three size regimes show distinct patterns:

**Small (bs=2048, hidden=1024)** — large spread, biggest headroom:
- NT: 774 TFLOP/s → 1.00× (baseline)
- NN: 394 TFLOP/s → 0.51× (hipBLASLt NN is **half** of NT throughput)
- TN: 572-580 TFLOP/s → 0.74×
- Interpretation: hipBLASLt's NN kernel at small shapes is especially weak — likely picks a 256x256x64 tile with low CU occupancy (output is only 2048×1024 = 16 tiles). **Tier-3 (winning) opportunity**: if our NN kernel uses a smaller tile (e.g. 128x128x64) to boost tile count, we could double hipBLASLt's NN throughput at these shapes.

**Medium (bs=8192, hidden=4096)** — tight cluster around hipBLASLt peak:
- NT: 1619 TFLOP/s → 1.00×
- NN: 1675 TFLOP/s → 1.03× (slightly faster than NT — hipBLASLt favours NN here)
- TN: 1613, 1625 TFLOP/s → 1.00×
- Interpretation: hipBLASLt well-tuned. Tier-2 (0.9×) requires ~1460 TFLOP/s on our kernel — achievable but not trivial.

**Large (bs=32768, hidden=8192)** — hardware-limited, small spread:
- All layouts 1586-1613 TFLOP/s (within 2%).
- Interpretation: MFMA compute bandwidth ceiling reached. Our kernel matches = Tier-2; beating is possible only via fused epilogue (Tier-3 Phase 5).

### Counter-ratio signals (bf16)

Medium shapes (bs=8192, h=4096), all same ~67M MFMA instructions (= M·N·K/16³ × 4 per tile):

| layout | MFMA   | VALU   | VALU/MFMA | VMEM  | LDS    | VMEM/MFMA | LDS/MFMA |
|:------:|-------:|-------:|----------:|------:|-------:|----------:|---------:|
| NT     | 67.2M  | 77.9M  | 1.16      | 8.67M | 16.9M  | 0.129     | 0.252    |
| NN     | 67.1M  | 104.3M | 1.55      | 8.46M | 16.8M  | 0.126     | 0.251    |
| TN     | 67.2M  | 140.6M | 2.09      | 8.53M | 16.9M  | 0.127     | 0.251    |

**Key finding — VALU/MFMA ratio spread**:
- NT: 1.16 (baseline, minimal non-MFMA work)
- NN: 1.55 (+35% VALU vs NT — more address-calc and element-shuffle work)
- TN: 2.09 (+80% VALU vs NT — significantly more scalar overhead)

This is hipBLASLt's own VALU tax for handling the non-natural-stride layouts: NN needs more address arithmetic because B's K is outer, and TN needs more for both operands having contract-dim outer. **The VALU tax is the theoretical limit on how much we can close the gap** — NT has 1.16 VALU/MFMA so is already 87% MFMA-bound; NN at 1.55 and TN at 2.09 leave room.

VMEM and LDS counts are remarkably flat across layouts (differ by <3%), suggesting hipBLASLt uses a similar LDS-staging pipeline for all three — the layout-difference shows up only in VALU-heavy address/permute code.

Small shapes (bs=2048, h=1024), all ~1.05M MFMA:

| layout | VALU   | VALU/MFMA | VMEM  | LDS    |
|:------:|-------:|----------:|------:|-------:|
| NT     | 1.89M  | 1.80      | 215k  | 600k   |
| NN     | 3.73M  | 3.55      | 350k  | 989k   |
| TN     | 3.77M  | 3.59      | 281k  | 1.31M  |

At small shapes the VALU tax is much heavier across the board (1.80 for NT, 3.55 for NN/TN) — small tiles mean the fixed-cost preamble and epilogue dominate proportionally. Also LDS/MFMA at 1.24 (TN) vs 0.57 (NT) suggests TN's swizzled LDS read/write pressure is 2× what NT pays.

### hipBLASLt kernel selection

Examining `Kernel_Name` in the rocprofv3 CSVs (bs=8192, h=4096, bf16):

| layout | Kernel_Name prefix                                    |
|:------:|-------------------------------------------------------|
| NT     | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0` |
| NN     | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0` |
| TN     | `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0` |

The suffix differs on A and B axes: `Alik`/`Ailk` encodes A's axis order, `Bljk`/`Bjlk` encodes B's. Same tile `MT256x256x64` and MFMA atom `MI16x16x1` across all three — hipBLASLt has **three distinct kernels** (per-layout specialised) that share the same block/tile configuration but differ in the address and load code. Confirms that our plan of three per-layout kernel shells (A.3) matches hipBLASLt's own shape — they didn't unify at the kernel level either.

### f16 vs bf16 — different story

Clean-timing bench shows **much wider f16 layout spread than bf16**:

| regime | bf16 NT→TN spread | f16 NT→TN spread |
|--------|------------------:|-----------------:|
| small  | 26% (774→572)     | 26% (756→562)    |
| medium | 0.4% (1619→1613)  | 21% (1495→1178)  |
| large  | 2% (1613→1586)    | 26% (1532→1132)  |

bf16 hipBLASLt is layout-robust. **f16 hipBLASLt is not** — its f16 NN kernel runs 9-13% slower than f16 NT, and f16 TN runs 21-26% slower.

The counter data tells us **why**. In f16 at medium/large shapes, NN and TN kernels issue **52% more LDS ops per MFMA** than NT:

| shape | layout | LDS/MFMA (bf16) | LDS/MFMA (f16) |
|-------|:------:|----------------:|---------------:|
| 8192×16384×4096   | NT | 0.25 | 0.25 |
| 8192×4096×16384   | NN | 0.25 | **0.38** |
| 16384×4096×8192   | TN | 0.25 | **0.38** |
| 32768×32768×8192  | NT | 0.25 | 0.25 |
| 32768×8192×32768  | NN | 0.25 | **0.38** |
| 32768×8192×32768  | TN | 0.25 | **0.38** |

In bf16 hipBLASLt's three kernels all issue ~0.25 LDS ops per MFMA (a ratio indicating a tight K=64-staged 2-pipeline). In f16, the NN and TN kernels jump to 0.38 — **hipBLASLt's f16 NN/TN kernels have a less-optimised LDS swizzle/read pattern than its f16 NT kernel.** VALU/MFMA ratios are similar across dtypes (so the extra work isn't scalar-unit pressure), but the LDS-round-trip delta explains most of the f16 layout-spread.

**Tier-3 opportunity concrete picture**: if our f16 TN kernel lands a 0.25 LDS/MFMA ratio (matching our own NT), we eliminate the 52% LDS overhead that hipBLASLt's f16 TN pays, and that alone should close the 26% wall-time gap — getting us from Tier-2 parity to Tier-3 winning.

### Tier-2 per-shape budgets

Tier-2 target = our-throughput ÷ hipBLASLt-throughput ≥ 0.9
⇒ our kernel μs ≤ hipBLASLt μs × 1.11

**bf16**:

| shape                | layout | hipBLASLt μs | Tier-2 budget (μs) | Tier-3 threshold (our < hipBLASLt) |
|----------------------|:------:|-------------:|-------------------:|-----------------------------------:|
| 2048 × 4096 × 1024   | NT     |     22       |      24            |     < 22                           |
| 2048 × 1024 × 4096   | NN     |     44       |      49            |     < 44                           |
| 4096 × 1024 × 2048   | TN     |     30       |      33            |     < 30                           |
| 1024 × 4096 × 2048   | TN     |     30       |      33            |     < 30                           |
| 8192 × 16384 × 4096  | NT     |    679       |     754            |    < 679                           |
| 8192 × 4096 × 16384  | NN     |    656       |     728            |    < 656                           |
| 16384 × 4096 × 8192  | TN     |    676       |     751            |    < 676                           |
| 4096 × 16384 × 8192  | TN     |    682       |     757            |    < 682                           |
| 32768 × 32768 × 8192 | NT     |  10907       |   12107            |  < 10907                           |
| 32768 × 8192 × 32768 | NN     |  10996       |   12206            |  < 10996                           |
| 32768 × 8192 × 32768 | TN     |  11078       |   12297            |  < 11078                           |
| 8192 × 32768 × 32768 | TN     |  11092       |   12312            |  < 11092                           |

**f16 Tier-2 budgets** (all entries = 1.11× the f16 hipBLASLt μs from the f16 clean-timing bench):

| shape                | layout | hipBLASLt μs | Tier-2 budget (μs) |
|----------------------|:------:|-------------:|-------------------:|
| 2048 × 4096 × 1024   | NT     |     23       |      26            |
| 2048 × 1024 × 4096   | NN     |     43       |      48            |
| 4096 × 1024 × 2048   | TN     |     31       |      34            |
| 1024 × 4096 × 2048   | TN     |     31       |      34            |
| 8192 × 16384 × 4096  | NT     |    735       |     816            |
| 8192 × 4096 × 16384  | NN     |    806       |     895            |
| 16384 × 4096 × 8192  | TN     |    934       |    1037            |
| 4096 × 16384 × 8192  | TN     |    938       |    1041            |
| 32768 × 32768 × 8192 | NT     |  11481       |   12744            |
| 32768 × 8192 × 32768 | NN     |  13174       |   14623            |
| 32768 × 8192 × 32768 | TN     |  15539       |   17249            |
| 8192 × 32768 × 32768 | TN     |  15643       |   17364            |

## Recommendations

**For the NN kernel (Phase 3)**:
1. **Smallest-shape tile tuning matters most.** At bs=2048, h=1024, hipBLASLt's NN is only 394 TFLOP/s vs 774 for NT. A smaller tile (e.g. 128×128×64 vs hipBLASLt's 256×256×64) would give us more CUs active on the small 2048×1024 output. This is directly our shot at Tier-3.
2. **At medium/large bf16 shapes, target Tier-2 (0.9×) not Tier-3.** hipBLASLt is within 3% of NT perf on NN here — we won't beat a near-peak kernel without fused epilogues (Phase 5).
3. **f16 is a cleaner win than bf16.** hipBLASLt's f16 NN at large shapes is 13% off its f16 NT. Matching our own f16 NT perf on NN gets us Tier-3.

**For the TN kernel (Phase 4)**:
1. **VALU budget is 2× what NT gets** (hipBLASLt's own overhead), so we have more headroom before we hit MFMA-bound. If we can reduce VALU work below hipBLASLt's (tighter LDS swizzle, fewer address-calc ops), we beat them.
2. **Split-K over M_batch is critical at small batches** (as planned day-1) — small-M TN shapes are where tile-count starves the GPU.
3. **f16 TN is the biggest prize.** At all shape sizes, f16 TN runs 21-26% slower than f16 NT on hipBLASLt. Matching our own f16 NT perf on TN is a 1.26× win over hipBLASLt.

**Don't revise the spec**:
1. The A.3 shared-core / three-shell structure matches hipBLASLt's own approach (three per-layout kernels sharing tile config).
2. bf16 + f16 day-1 is correct — f16 has strictly more headroom than bf16 for NN/TN.
3. Fixed-config MVP is safe for medium/large shapes (hipBLASLt itself uses one tile config across layouts at these sizes); autotune matters for small shapes where tile-vs-occupancy tradeoff is delicate.

---

## Appendix — raw command log

```
# PMC counter file validation
rocprofv3 -i tests/amd/pmc_counters.txt -- /bin/true

# Clean timing
PYTHONPATH=. python -m tests.amd.bench_hipblaslt_layouts --dtype bf16 | tee /tmp/layout_bench_bf16.txt
PYTHONPATH=. python -m tests.amd.bench_hipblaslt_layouts --dtype f16  | tee /tmp/layout_bench_f16.txt

# Full HW-counter profile
PYTHONPATH=. python -m tests.amd.bench_hipblaslt_layouts --profile --dtype bf16 | tee /tmp/layout_profile_bf16.txt
PYTHONPATH=. python -m tests.amd.bench_hipblaslt_layouts --profile --dtype f16  --out-dir ./rocprof-layout-profile-f16 | tee /tmp/layout_profile_f16.txt
```
