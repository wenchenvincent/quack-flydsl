# AMD Large-N Reductions — Investigation Finding (Phase 5)

**Date:** 2026-07-15 · gfx950 / MI355X

## TL;DR

The parity review listed "large-N / huge-vocab reductions" as a **missing
capability** — no cluster/multi-block reduction, softmax capped ~32K, CE
untested past 4K. Direct measurement shows this is a **test-coverage gap,
not a capability gap.** The existing AMD softmax and cross-entropy kernels
already handle N up to at least **262144 (256K)**, forward *and* backward,
with float32-reference-grade accuracy. No new grid-stride or two-pass
mechanism is needed.

## Measured (probe, 2026-07-15)

Forward, M=4, float32, `max_err` vs torch reference:

| N | softmax_fwd | cross_entropy_fwd |
|---|---|---|
| 16384 | 2.3e-10 | 9.5e-07 |
| 65536 | 5.8e-11 | 9.5e-07 |
| 131072 | 5.8e-11 | 9.5e-07 |
| 262144 | 2.9e-11 | 9.5e-07 |

Backward, `dx max_err` vs `torch.autograd.grad`:

| N | softmax_bwd | cross_entropy_bwd |
|---|---|---|
| 65536 | 1.2e-10 | 1.5e-08 |
| 131072 | 8.7e-11 | 6.0e-08 |
| 262144 | 8.7e-11 | 3.8e-10 |

Every case ran without error and matched the reference. The kernels already
process each row with an internal strided loop across the block's threads,
so a single workgroup handles arbitrarily large N — exactly the "clusterless
grid-stride" mechanism the Phase 5 plan proposed to *build* is effectively
already present.

## What actually shipped for Phase 5

Rather than build a redundant mechanism, lock the existing capability:
- `tests/amd/test_softmax.py::test_softmax_large_n` — fwd+bwd at N=131072.
- `tests/amd/test_cross_entropy.py::test_cross_entropy_large_vocab` — fwd+bwd at N=131072.

Both pass. These are the regression guard that the (already-working) large-N
path stays correct.

## Not done (correctly out of scope)

- The two-pass HBM-workspace escalation in the Phase 5 plan — unnecessary; a
  single block already saturates the reduction at these N with correct output.
- The `reduction_base._reduction_elem_bytes` Float32-only limit — never hit by
  these paths; leave it (YAGNI).

## Correction to the parity review

The parity-review artifact and its reductions section should be read with
this caveat: the large-N reduction gap it lists is **coverage, not
capability**. The kernels work; only the tests were missing. This mirrors the
Phase 3 finding, where the fused-dact path was already enabled despite a
docstring saying "disabled" — several "gaps" in the review were stale-doc /
missing-test artifacts rather than absent functionality.
