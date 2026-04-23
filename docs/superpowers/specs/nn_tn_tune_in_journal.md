# NN / TN kernel tune-in journal — Phase 3.5 / 4.5

Tracks the LDS-perf tune-in for ``gemm_gfx950_nn`` and ``gemm_gfx950_tn``
after their MVP landings (commits `af77205` and `f5986fd`).

## MVP baseline (pre-tune-in)

Measured at 8192×4096×16384 bf16 (MLP-dx canonical shape), single-dispatch:

| kernel | throughput | vs hipBLASLt | kernel μs |
|--------|-----------:|-------------:|----------:|
| NN MVP | 453 TF/s | 0.28× | 2446 |
| TN MVP | 225 TF/s | 0.14× | 4895 |

rocprofv3 counters on NN MVP:

| counter | value | per-iter |
|---------|------:|---------:|
| SQ_INSTS_MFMA | 67.11M | ~1.5M |
| SQ_INSTS_VALU | 250M | ~5.6M |
| SQ_INSTS_LDS | 92.86M | ~2.1M |
| SQ_LDS_BANK_CONFLICT | **135.27M** | ~3M stall cycles |
| FETCH_SIZE | 1.05M | — |

## Attempt 1 — transposed LDS layout (scalar scatter writes)

Idea: change B LDS from `(STAGES, BLOCK_K, BLOCK_N)` to `(STAGES, BLOCK_N, BLOCK_K)`
so MFMA fragment reads become stride-1 vec_load per lane. Requires scatter
writes on the LDS store path (each thread's 8 HBM values go to 8 different
N rows at same K column).

**Result: 3× SLOWER.** 144 TF/s → 35 TF/s at the same shape. The 8 scalar
LDS stores per thread (vs 1 b128 vec_store) dominated the hot loop, even
though MFMA-critical-path LDS reads became a single vec_load.

Lesson: scatter writes are catastrophic for throughput. Any LDS transpose
must preserve vectorised writes.

## Attempt 2 — in-place swizzle change

Idea: change `swizzle_xor16` key from `(row % n_blocks16) * 16` to
`(row % 32) * 4` — XOR by 4 bytes (stride 1 f16) to target bank bits 2..6.

**Result: correctness broken.** Max error ~4 on bf16 outputs. The 4-byte
XOR (= stride 2 f16) is smaller than `LDG_VEC_SIZE=8`, so `vec_store(..., 8)`
no longer corresponds 1:1 with `vec_load` — the 8 f16s get placed in a
"rotated" subspan that's incompatible with the load side.

Lesson: any XOR swizzle key must be an integer multiple of
`LDG_VEC_SIZE * DTYPE_BYTES = 16 bytes` to preserve vec_store correctness.

## Attempt 3 — B-LDS row padding (landed in commit `2ac0cb5`)

Idea: add 8 f16 of pad per row (stride 256→264 f16 = 512→528 bytes) so
row stride isn't a multiple of bank period (128 bytes).

**Result: 14-30% speedup — but the mechanism was not what I thought.**

| shape | before | after | speedup |
|-------|-------:|------:|--------:|
| NN 2048×1024×4096 | 144 | 167 TF/s | +16% |
| NN 8192×4096×16384 | 453 | 514 TF/s | +14% |
| TN K=2048 | 124 | 151 TF/s | +22% |
| TN K=8192 | 225 | 293 TF/s | +30% |

However re-profile after the fix shows `SQ_LDS_BANK_CONFLICT` is **still
135M** (same as MVP — NOT reduced). The math confirms: with stride 528
bytes, `8 * 528 = 4224` is still 33 × 128 (bank period) so the 4-way
conflict between k_groups persists.

The speedup came from elsewhere — likely the reduced `SQ_INSTS_VALU`
(250M → 218M). Possible causes: different address-computation paths
after the STensor shape change, register-allocation changes, or
compiler heuristics affecting the VALU/MFMA interleave. The commit
message for `2ac0cb5` is incorrect on this point; the speedup is real
but the attributed root cause is wrong.

Lesson: rocprofv3's bank-conflict counter must be verified after
claimed fixes. Pad within the 16-byte-aligned constraint cannot break
the stride-8-row bank alignment on its own.

## Attempt 4 — A-side pad (reverted)

Same idea applied to A. Complicated by A's swizzle key depending on
`BLOCK_K_BYTES / 16`; padding A's stride required re-deriving the key
to preserve correctness. Broke correctness on first try; reverted.

## What Tier-1 actually requires

The 4-way k_group bank conflict on B reads can NOT be eliminated by
16-byte-aligned pad (`8 × stride mod 128 = 0` for all 16-aligned strides).
To actually vectorise the MFMA B-frag read, one of these is needed:

- **(a) `ds_read_tr16_b64` (CDNA4 hardware-transposed LDS read).** FlyDSL
  exposes this as `rocdl.ds_read_tr16_b64` — used by `flash_attn_func.py`.
  The hardware does a 4×4 transpose across 4 groups of 4 lanes, yielding
  one b64 (4 f16) per lane from the wave-collective LDS load. Two such
  reads would replace the current FRAG=8 scalar loads per lane per
  fragment. Integration requires matching the fragment lane-layout to
  the transpose's source-lane semantics: `result[lane, e] = input[e*4 +
  (lane%16)//4, lane%4]`. ~200 LoC change in `lds_matrix_b`.

- **(b) Cooperative-warp HBM → LDS transpose.** Each thread reads 8 N-contig
  from HBM, then 8 threads collectively do a 3-round butterfly shuffle
  (`ds_bpermute_b32`) to transpose the 8×8 block. Each thread then
  vec_stores 8 K-contig to LDS at a (N, K) layout. ~150 LoC + shuffle
  overhead.

- **(c) B preshuffle** (like the NT kernel does). W is reused between
  fwd (NT) and bwd-DX (NN), so preshuffling for both requires 2× memory.
  Unless we accept the memory hit or preshuffle-on-write at every DW step.

Option (a) is the most promising for NN/TN; it's also how hipBLASLt's
own NN/TN kernels hit their perf. Option (b) is simpler but adds VALU
work. Option (c) is cleanest but memory-expensive.

## Status

- **NN**: 0.32× (medium) / 0.43× (small) hipBLASLt — below Tier-1 (0.5×).
- **TN**: 0.19× (medium) / 0.26× (small) hipBLASLt — below Tier-1.
- Next tune-in step: option (a) — integrate `ds_read_tr16_b64`. This is
  a real kernel rewrite, not a small fix. Estimate 1-2 sessions.

Alternatively — **Phase 5 first**: wire autograd Function, get training
path end-to-end, measure real step time, then return to kernel tune-in
with a meaningful benchmark target. This is the pragmatic order.
