# AMD Phase 5 — Large-N / Huge-Vocab Reductions

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. This phase is **design-heavy**: Task 1 is investigation (no TDD), Tasks 2+ are TDD once the mechanism is chosen. Do not skip Task 1 — the design decision drives everything after.

**Goal:** Let softmax and cross-entropy handle rows too large for a single workgroup — the long-sequence / large-vocabulary shapes (N ≥ 128K) that NVIDIA covers and AMD currently cannot.

**Architecture:** AMD's reductions are **block-only** today. `quack/amd/reduce.py` provides `wave_reduce`/`block_reduce`/`row_reduce` (`:49,69,149`) — one workgroup reduces one row, capped by how much of N fits in registers+LDS across its waves. Softmax's multiwave path already reaches N=32768 (`tests/amd/test_softmax.py:51`); cross-entropy is tested only to N=4096. There is **no cluster / multi-block-per-row mechanism** (unlike NVIDIA's `cluster_reduce`), and `reduction_base._reduction_elem_bytes` (`:105`) hard-limits the reduction dtype to Float32. NVIDIA's cross-entropy handles huge vocab via a split-vocab two-stage path (partial LSE per chunk → combine), keyed off `target_logit` (`quack/cross_entropy.py:171`). AMD needs an equivalent that does NOT rely on a hardware cluster concept — a **two-pass grid-stride reduction through an HBM workspace**: pass 1, each of G blocks reduces its N/G slice of a row to a partial (max, sumexp); pass 2, combine the G partials and finalize. This is the standard clusterless large-reduction pattern.

**Tech Stack:** `quack/amd/reduce.py`, `quack/amd/reduction_base.py`, `quack/amd/softmax.py`, `quack/amd/cross_entropy.py`; FlyDSL kernels; tests `tests/amd/test_softmax.py`, `tests/amd/test_cross_entropy.py`. Design references: `quack/reduce.py` (`cluster_reduce`, `:32`), `quack/cross_entropy.py` (split-vocab, `:171,655`).

**Risk (high):** this is the largest single gap in the port — a genuinely new mechanism, not a wrapper. Scope conservatively: land the two-pass reduction for cross-entropy first (the clearest huge-vocab use case), then reuse it for softmax.

---

## Task 1: Investigation — find the real ceiling and pick the mechanism (NO code changes)

**Deliverable:** a short design note committed to `docs/superpowers/specs/2026-07-14-amd-large-n-reductions.md`.

- [ ] **Step 1: Establish where the current kernels fall over**

Empirically find the largest N each current path handles before it errors or degrades. Run softmax and cross-entropy forward at increasing N (32K, 64K, 128K, 256K) on gfx950 and record: does it raise, silently truncate, or run? Note the failure mode and the LDS/register limit that causes it (cross-reference `get_shared_memory_capacity` in `quack/amd/flydsl_utils.py` and the per-wave register budget).

- [ ] **Step 2: Confirm the mechanism choice against constraints**

Decide between (a) **two-pass grid-stride через HBM workspace** (G blocks per row, partials combined in pass 2) vs (b) **single-block grid-stride loop** (one block strides over the whole row, no second kernel — simpler but serializes N/blockDim iterations). Decision criteria: (b) is far simpler and may suffice up to very large N if the per-block stride loop is bandwidth-bound not latency-bound; (a) is needed only if one block can't saturate memory bandwidth for a single huge row. **Recommend starting with (b)** — a grid-stride loop inside the existing single-block kernel raises the N ceiling with no new kernel or workspace, and is the YAGNI choice. Escalate to (a) only if (b) leaves bandwidth on the table at the target N.

- [ ] **Step 3: Write the design note + pick the first target**

Write `docs/superpowers/specs/2026-07-14-amd-large-n-reductions.md` capturing: the measured ceilings, the chosen mechanism (likely single-block grid-stride first), the target N (e.g. 131072), and which kernel to do first (recommend **cross-entropy forward**, since huge-vocab CE is the canonical use case and its reduction is a single max+sumexp per row). Commit it.

```bash
git add docs/superpowers/specs/2026-07-14-amd-large-n-reductions.md
git commit -m "[docs] Design note — AMD large-N reduction mechanism + measured ceilings"
```

## Task 2: Grid-stride the cross-entropy forward reduction

**Files:**
- Modify: `quack/amd/cross_entropy.py` (the fwd kernel's row-reduction loop)
- Test: `tests/amd/test_cross_entropy.py`

- [ ] **Step 1: Write the failing test — huge-vocab CE forward**

Append to `tests/amd/test_cross_entropy.py` a test at N ≥ 131072 comparing `cross_entropy_fwd` loss to a float32 `torch.nn.functional.cross_entropy` reference:
```python
@pytest.mark.parametrize("N", [131072])
def test_cross_entropy_huge_vocab(N):
    torch.manual_seed(0)
    M = 8
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    tgt = torch.randint(0, N, (M,), device="cuda")
    from quack.amd.cross_entropy import cross_entropy_fwd
    loss, _ = cross_entropy_fwd(x, tgt, return_lse=True)
    ref = torch.nn.functional.cross_entropy(x, tgt, reduction="none")
    assert torch.allclose(loss.float(), ref.float(), atol=1e-3, rtol=1e-3)
```

- [ ] **Step 2: Run to confirm it fails (or errors) at the current ceiling**

Run: `cd /workspace/quack && python -m pytest tests/amd/test_cross_entropy.py::test_cross_entropy_huge_vocab -x`
Expected: FAIL (raises at compile/launch, or produces wrong loss) — this is the ceiling from Task 1.

- [ ] **Step 3: Implement the grid-stride reduction in the CE fwd kernel**

Modify the CE forward kernel's per-row max+sumexp reduction so each thread strides over `ceil(N / block_threads)` elements of its row (accumulating a running max and, in the standard online-softmax way, rescaling the running sumexp when the max updates) before the existing block reduction combines threads. This keeps the single-block structure; only the per-thread inner loop changes from "one element" to "a strided range". Reuse `quack/amd/reduce.py`'s `block_reduce_max`/`block_reduce_add` for the cross-thread step. Follow `port-hipkittens-to-flydsl` conventions for the loop (`range_constexpr` where the count is static; runtime `range` only if N is dynamic).

- [ ] **Step 4: Run to confirm pass**

Run: `cd /workspace/quack && python -m pytest tests/amd/test_cross_entropy.py::test_cross_entropy_huge_vocab -x`
Expected: PASS. Then run the full CE suite to confirm no regression at existing N: `python -m pytest tests/amd/test_cross_entropy.py -x`.

- [ ] **Step 5: Commit**

```bash
git add quack/amd/cross_entropy.py tests/amd/test_cross_entropy.py
git commit -m "[AMD] Cross-entropy forward — grid-stride reduction for huge vocab (N>=128K)"
```

## Task 3: Grid-stride the cross-entropy backward + softmax

**Files:**
- Modify: `quack/amd/cross_entropy.py` (bwd), `quack/amd/softmax.py` (fwd+bwd)
- Test: `tests/amd/test_cross_entropy.py`, `tests/amd/test_softmax.py`

- [ ] **Step 1: Write failing tests** — huge-N cross-entropy backward (gradient vs `F.cross_entropy` autograd at N=131072) and huge-N softmax fwd+bwd (vs `torch.softmax` at N=131072). Both numerical vs float32 reference.

- [ ] **Step 2: Run to confirm they fail at the ceiling.**

Run: `cd /workspace/quack && python -m pytest tests/amd/test_cross_entropy.py -x -k huge tests/amd/test_softmax.py -x -k huge`
Expected: FAIL.

- [ ] **Step 3: Apply the same grid-stride transform** to the CE backward dx loop and the softmax fwd (max+sumexp) and bwd (dot) reductions, reusing the pattern proven in Task 2.

- [ ] **Step 4: Run to confirm pass + full regression** of both suites.

Run: `cd /workspace/quack && python -m pytest tests/amd/test_cross_entropy.py tests/amd/test_softmax.py -x`
Expected: all green, including the new huge-N cases and all existing N.

- [ ] **Step 5: Commit**

```bash
git add quack/amd/cross_entropy.py quack/amd/softmax.py tests/amd/
git commit -m "[AMD] Grid-stride CE backward + softmax fwd/bwd for huge N"
```

## Task 4 (optional, gated on Task 1's finding): escalate to two-pass if grid-stride is bandwidth-limited

Only do this if Task 1 Step 2 concluded a single block can't saturate memory bandwidth for one huge row. It's a separate design + implementation effort (a second reduction kernel + an HBM partials workspace + a combine pass) — spin it into its own plan rather than executing from this outline. Marker left here so the escalation path is explicit, not forgotten.

---

## Notes deliberately OUT of scope

- The `reduction_base._reduction_elem_bytes` Float32-only limit (`:105`) — only lift it if a specific path needs the packed max+sum Int64 reduction dtype; grid-stride max+sumexp in Float32 does not, so leave it (YAGNI). Note if a task actually hits it.
- NVIDIA's cluster hardware path has no AMD analogue and is not the target — the grid-stride/two-pass approach is the clusterless equivalent.

---

## Self-Review

**Spec coverage:** Priority #5 ("large-N / huge-vocab reductions"). Task 1 finds the real ceiling and picks the (YAGNI) mechanism; Tasks 2–3 raise the ceiling for CE then softmax via grid-stride; Task 4 is an explicit, gated escalation. The design-first structure matches the reality that this is the port's largest open gap, not a mechanical wrap. ✓

**Placeholder scan:** Task 1 is explicitly investigation (produces a committed design note, not code) — appropriate for a design-heavy phase and not a hidden TODO. Tasks 2–3 are concrete TDD (real test code, exact files, exact commands). The one deferred item (Task 4 two-pass) is a gated escalation with a stated trigger, spun into its own future plan rather than under-specified here. ✓

**Type consistency:** `cross_entropy_fwd(x, target, return_lse=…)` and the `reduce.py` helpers `block_reduce_max`/`block_reduce_add` match `quack/amd/cross_entropy.py:905` and `quack/amd/reduce.py:131,135`. The grid-stride change is internal to the kernel bodies; public signatures are unchanged. ✓
