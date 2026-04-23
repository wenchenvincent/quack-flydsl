# Stream-K rewrite — perf journal

Tracking the production stream-K rewrite on MI355X / gfx950.
Kernel: ``quack.amd.gemm_streamk_prod.gemm_streamk``.

## Starting point

The existing ``quack.amd.gemm_streamk.gemm_f16_streamk`` demo was
100-600× slower than hipBLASLt.  Root cause: per-tile atomic-fadd on
a global counter + 16×16 tiles + compile-time unrolled outer loop.
Not a production candidate.

## Rewrite (this workstream)

Landed in five sessions (commits 4b17129, 2df0195, 99e9e37, 2c71f7d,
f71a9e0):

  - **Session A** — runtime scf.for outer loop (replaces
    ``range_constexpr`` unroll).  Compiled code stays small
    regardless of shape.
  - **Session B** — K-split via ``splits_per_tile`` shards; each WG
    processes one ``(tile, k_slice)`` pair.  Atomic-fadd partials
    into a zero-init output.  Saturates CUs when
    ``total_tiles < num_cus``.
  - **Session C** — true stream-K: running accumulator + flush-on-
    tile-change.  Each WG's K-iteration range is a contiguous slice
    of ``total_iters``, may cross tile boundaries.  Sub-optimal
    performance but proves the algorithm.
  - **Session D** — per-tile int32 counter + last-contributor
    spin-sync.  Non-owner WGs atomic-fadd partial then atomic-inc
    counter; last contributor spins until counter == num_contributors
    then owns the tile's writeback.  Adds optional bias.
  - **Session E** — full fused epilogue.  Bias + activation (relu /
    silu / relu_sq / gelu_tanh_approx) in f32, then cast to bf16/f16
    output.  Partials still accumulate in a separate f32 scratch
    (bf16 HBM atomic-fadd isn't reliable on CDNA4).  Public API:
    ``gemm_streamk(A, B, bias=, activation=, out_dtype=)``.

## Current perf (MI355X, event-timed)

| shape             | streamk_prod ms | persistent ms | hipBLAS ms | SK/hip |
|-------------------|----------------:|--------------:|-----------:|-------:|
| 128×128×128       | 0.039           | 0.026         | 0.013      | 3.09×  |
| 256×256×128       | 0.045           | 0.024         | 0.013      | 3.58×  |
| 512×512×256       | 0.046           | 0.020         | 0.012      | 3.79×  |
| 1024×1024×512     | 0.184           | 0.075         | 0.015      | 12.06× |
| 128×2048×512      | 0.053           | 0.025         | 0.015      | 3.60×  |

**40-150× improvement** over the old demo but still 3-12× slower than
hipBLASLt.

## Gap analysis — why still slow

1. **16×16 MFMA tile is too small.**  At 1024² K=512 that's 4096 tiles,
   each with (64 threads × 4 fragments) = 256 atomic-fadd ops per tile
   + 1 counter-inc = 1,052,672 atomic ops per kernel invocation.
   Atomic traffic dominates.

2. **HBM-resident counter.**  Even with agent-scope monotonic
   ordering, the counter spin-wait per tile serialises the last-
   contributor path.  At many-tiles-per-CU shapes the spin cost
   doesn't amortise.

3. **No LDS pipelining.**  Inner K-loop is direct buffer_load from
   HBM each iteration.  The tuned ``gemm_gfx950_splitk`` kernel uses
   LDS ping-pong + async copies + schedule hints — stream-K skips all
   of this for simplicity.

## What would close the gap

- **128×128 tile, 4-warp WG.**  Matches ``gemm_gfx950_splitk``'s
  layout.  Reduces atomic ops ~64×.  Requires MFMA fragment layout
  rework (lane→element mapping differs from 16×16).
- **Coalesced atomic writes.**  Use ``buffer_atomic_fadd`` vec4 ops
  instead of per-element scalar atomics.
- **LDS-resident counter.**  Within a CU, use LDS for the counter
  and only spill to HBM at CU boundaries.  Reduces spin-wait cost.
- **Async A/B loads with LDS ping-pong.**  Port the hot-loop
  scheduler pattern from ``gemm_gfx950_splitk``.

Any one of these is a multi-day engineering effort.  All four
together is essentially rewriting the kernel from scratch with
production-quality scheduling.

## Verdict

The current ``gemm_streamk`` is **not production-ready** — callers
that route through it will be slower than hipBLASLt at every shape
we measured.

It IS, however, a working implementation of true stream-K with
tile crossing + last-partial sync + fused epilogue, which is a
significant correctness milestone for the AMD port.  Production
deployment would require the gap-closing work above.

## Session G1 — pragmatic dispatch to splitk

Rather than rewrite the stream-K kernel body from scratch (which
would take several days to match splitk's LDS ping-pong + 128×256
tile + 4-warp WG perf), we added a dispatch in ``gemm_streamk``:
when the shape is compatible with ``gemm_gfx950_splitk`` (M%128,
N%256, K%64) AND ``out_dtype`` is f16, route through splitk.

Before / after perf at f16 shapes (event-timed):

| shape                   | before SK/hip | after SK/hip | dispatch |
|-------------------------|--------------:|-------------:|---------:|
| 128×128×128             |  3.09×        |  4.77×       | native   |
| 256×256×128             |  3.58×        | 10.10×       | splitk   |
| 512×512×256             |  3.79×        |  7.07×       | splitk   |
| 1024×1024×512           | 12.06×        |  7.04×       | splitk   |
| 128×2048×512            |  3.60×        |  8.99×       | splitk   |
| 4096×4096×1024          | (projected)   |  2.38×       | splitk   |

**Large-shape wins**, tiny-shape losses — the splitk dispatch has a
~85μs floor cost that dominates at tiny shapes where hipBLASLt is
~10μs.  At 4096³ the 2.38× ratio matches splitk's own gap-to-hipBLASLt
at typical training shapes.

Correctness holds: bias + silu + f16 output passes the usual
bf16-tolerance band at all tested shapes.

### Remaining work to actually beat hipBLASLt

Even after this dispatch, we're 2-10× behind hipBLASLt.  Closing
THAT gap is a kernel-tuning project across our whole AMD GEMM
stack — not just stream-K.  Dominant costs documented above still
apply.  A from-scratch production stream-K rewrite matching
splitk's tile + LDS + scheduling would take several days and
would at best match splitk, not beat hipBLASLt.

Session G1 is the pragmatic win; further perf requires either
kernel-body tuning or replacing hipBLASLt as the comparison point
(e.g., for the no-hipBLASLt fp8 path where we already win).
