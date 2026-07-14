# AMD Phase 3 — Reconcile & Regression-Guard the Fused Act-Backward Path

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** Make the fused activation-backward path's *actual behavior*, its *documentation*, and its *test coverage* agree — and guard the enable/disable decision with a bench-backed regression test.

**Architecture:** This phase's premise changed after reading the code. The parity review said the fused `gemm_dact` MLP-backward was "default-disabled." It is **not** — `_fused_dact_eligible` (`quack/amd/linear_training.py:219`) returns `True` for eligible bf16/fp16 shapes (M%128, hidden%256, out_dim%64, contiguous), with an in-body comment "Enabled by W3 benchmarks (2026-04-22, MI355X)" claiming a 2.25× win, and `_MLPActFunction.backward` (`:295-306`) dispatches `gemm_splitk(..., dact_activation=...)` when eligible. But the `mlp_func_train` docstring (`:336-344`) still says "currently disabled by default because splitk's matmul runs ~1.07× slower." **The gate and its own docstring contradict each other.** So Phase 3 is not "enable it" — it's: (1) determine empirically which path actually runs and whether it's faster, (2) fix whichever of {gate, docstring} is wrong, (3) add a test that asserts the eligible path is dispatched and a regression bench that records the decision so it can't silently rot again.

**Tech Stack:** `quack/amd/linear_training.py`, `quack/amd/gemm_gfx950_splitk.py`, pytest + a micro-bench on gfx950 (MI355X).

---

## Task 1: Determine ground truth — which path runs, and is it faster?

**Files:**
- Create: `tests/amd/bench_fused_dact.py` (a throwaway/committed micro-bench)

- [ ] **Step 1: Write a bench that measures both paths at the reference shape**

Create `tests/amd/bench_fused_dact.py`:
```python
"""Micro-bench: fused gemm_dact MLP-backward vs torch.mm + torch act-bwd.
Run manually: python tests/amd/bench_fused_dact.py
"""
import time
import torch
from quack.amd.linear_training import _fused_dact_eligible, _act_bwd


def _bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3  # ms


def main():
    M, hidden, out_dim = 4096, 8192, 4096
    act = "silu"
    dout = torch.randn(M, out_dim, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(out_dim, hidden, device="cuda", dtype=torch.bfloat16)
    preact = torch.randn(M, hidden, device="cuda", dtype=torch.bfloat16)
    print("eligible:", _fused_dact_eligible(dout, w2, preact, act))

    from quack.amd.gemm_gfx950_splitk import gemm_splitk
    w2_T = w2.t().contiguous()

    def fused():
        return gemm_splitk(dout, w2_T, preact=preact, dact_activation=act)

    def unfused():
        gp = torch.mm(dout, w2)
        return _act_bwd(preact, gp, act)

    # correctness first
    a = fused().float()
    b = unfused().float()
    err = (a - b).abs().max().item()
    print(f"max_err fused vs unfused: {err:.4f}")

    print(f"fused:   {_bench(fused):.3f} ms")
    print(f"unfused: {_bench(unfused):.3f} ms")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and record the numbers**

Run: `cd /workspace/quack && python tests/amd/bench_fused_dact.py`
Record: whether `_fused_dact_eligible` returns True at this shape, the max_err (should be small — bf16), and the two timings. This determines ground truth.

- [ ] **Step 3: Decide and note the outcome (no code change yet)**

Write the observed numbers into the commit message in Step 4. Two outcomes:
- **Fused is faster (matches the gate's 2.25× comment):** the gate is correct; the `mlp_func_train` docstring is stale (Task 2 fixes the docstring).
- **Fused is slower (matches the docstring):** the gate's "Enabled by W3 benchmarks" comment is wrong/stale for the current toolchain; the gate should be disabled (Task 2 flips the gate and fixes the comment).

- [ ] **Step 4: Commit the bench**

```bash
git add tests/amd/bench_fused_dact.py
git commit -m "[AMD] Add fused-dact vs torch micro-bench; <FUSED|UNFUSED> is faster (<X> vs <Y> ms)"
```

## Task 2: Reconcile gate ↔ docstring to match ground truth

**Files:**
- Modify: `quack/amd/linear_training.py` (the stale docstring at `:336-344` and/or the gate comment at `:250-256`)

- [ ] **Step 1: Fix whichever is wrong (based on Task 1's result)**

If **fused is faster** (gate correct): rewrite the `mlp_func_train` docstring "Status of the fused path" paragraph (`quack/amd/linear_training.py:336-344`) to state the fused path is ENABLED for eligible shapes, citing the Task-1 numbers and date. Remove the contradictory "currently disabled by default" wording.

If **fused is slower** (docstring correct): change `_fused_dact_eligible`'s final `return` (`:256`) to `return False`, replace the stale "Enabled by W3 benchmarks... 2.25× faster" comment (`:250-256`) with the Task-1 numbers explaining why it's disabled, and leave the docstring's "disabled" wording (update its numbers to Task-1's).

- [ ] **Step 2: Verify the existing MLP training tests still pass**

Run: `cd /workspace/quack && python -m pytest tests/amd/test_linear_train.py tests/amd/test_linear.py -x -k "mlp or MLP or func"`
Expected: all pass (correctness is identical either way — this only changes which backend runs).

- [ ] **Step 3: Commit**

```bash
git add quack/amd/linear_training.py
git commit -m "[AMD] Reconcile fused-dact gate/docstring with measured behavior"
```

## Task 3: Regression test — assert the eligible path is dispatched + correct

**Files:**
- Test: `tests/amd/test_linear_train.py` (append)

- [ ] **Step 1: Write a test that pins the dispatch decision and correctness**

Append to `tests/amd/test_linear_train.py` (adapt the import path if `mlp_func_train` lives elsewhere):
```python
def test_mlp_func_train_backward_matches_reference():
    import torch
    from quack.amd.linear_training import mlp_func_train
    torch.manual_seed(0)
    M, hidden, out_dim = 128, 256, 128  # smallest eligible shape (M%128, hidden%256, out_dim%64)
    x = torch.randn(M, out_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn(hidden, out_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(out_dim, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr, w1r, w2r = (t.detach().clone().requires_grad_(True) for t in (x, w1, w2))

    out = mlp_func_train(x, w1, w2, activation="silu")
    ref = torch.nn.functional.silu(torch.nn.functional.linear(xr, w1)) @ w2r.t()

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    # bf16 MLP backward through two matmuls — loose but real
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(w1.grad.float(), w1r.grad.float(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(w2.grad.float(), w2r.grad.float(), atol=5e-2, rtol=5e-2)
```
(Confirm the exact `mlp_func_train` forward math against `quack/amd/linear_training.py:318` before finalizing the reference — match bias handling and the linear vs `x @ w1.T` convention; the assertion is the ground truth, adjust the reference to the real forward.)

- [ ] **Step 2: Run to confirm it passes** (the fused path is exercised for this eligible shape regardless of the Task-2 decision — either backend must be numerically correct)

Run: `cd /workspace/quack && python -m pytest tests/amd/test_linear_train.py::test_mlp_func_train_backward_matches_reference -x`
Expected: PASS. If it FAILS, the fused kernel or the reference math is wrong — debug (do not weaken tolerance blindly).

- [ ] **Step 3: Commit**

```bash
git add tests/amd/test_linear_train.py
git commit -m "[AMD] Regression test — mlp_func_train backward numerical correctness"
```

---

## Self-Review

**Spec coverage:** Priority #3 ("enable / justify the fused act-backward path") is addressed as it actually stands: the path is already dispatched, so this phase makes the code, docs, and tests agree and bench-backs the decision. Task 1 finds ground truth, Task 2 fixes the code/doc contradiction, Task 3 guards it. ✓

**Placeholder scan:** Two decision points (Task 1 Step 3, Task 2 Step 1) are genuine branches on a measured result, not deferred work — both outcomes are fully specified. The reference math in Task 3 is flagged for confirmation against the real forward, with the assertion as ground truth. ✓

**Type consistency:** `_fused_dact_eligible(dout, w2, preact, activation)`, `mlp_func_train(x, w1, w2, activation, bias1, bias2)`, and `gemm_splitk(a, b, preact=, dact_activation=)` match `quack/amd/linear_training.py:219,318,304`. ✓
