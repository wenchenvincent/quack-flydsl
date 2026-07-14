# AMD TopK Backward + Autograd — Implementation Plan (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Give the AMD TopK a backward pass and an autograd-enabled `topk()` — the only reduction family in the port with no gradient at all.

**Architecture:** TopK's gradient is a **scatter**: `dx[m, indices[m,j]] = dvalues[m,j]`, zeros elsewhere. That needs no new FlyDSL kernel — a `torch.zeros(...).scatter_` is exact and dtype-agnostic. The softmax-fused case pre-multiplies `dvalues` by the softmax Jacobian over the k values before scattering. We add `TopKBackward` (scatter), `TopKFunction` (autograd), and `topk()` to `quack/amd/topk.py`, mirroring NVIDIA's `quack/topk.py` surface, then export `topk` alongside the Phase-1 `nn.py` layer. Non-f32 *forward* already works through the existing `torch.topk` fallback, so this plan delivers a full autograd story for f32/bf16/fp16; a *native* non-f32 FlyDSL kernel is a separate follow-on (roadmap at the end).

**Tech Stack:** PyTorch autograd + scatter, the existing `quack.amd.topk.topk_fwd` kernel dispatcher, pytest on gfx950 (MI355X).

**Depends on:** nothing hard — but if the Phase-1 `quack/amd/nn.py` exists, Task 3 re-exports `topk` through it. If Phase 1 hasn't landed, Task 3 exports from `quack/amd/__init__.py` directly (both are specified).

---

## Background the engineer needs

- **Verified AMD forward** (`quack/amd/topk.py:69`): `topk_fwd(x, k, softmax=False) -> (values, indices, softmax_values_or_None)`. Values are descending along the last dim; `indices` index into the input's last dim; when `softmax=True` the third element is `softmax(values)` in the input dtype. Non-f32 / N>4096 / non-contiguous inputs route to a `torch.topk` fallback inside `topk_fwd`, so **bf16/fp16 forward already works** (just not via a FlyDSL kernel).
- **NVIDIA shape to mirror** (`quack/topk.py:556`): `TopKFunction.forward` saves `(values_if_softmax, indices)`, calls `ctx.mark_non_differentiable(indices)` and `ctx.set_materialize_grads(False)`; `backward` calls `topk_bwd(dvalues, values, indices, N, softmax) -> dx`. We copy this shape; our `topk_bwd` is a scatter rather than a kernel.
- **The softmax Jacobian** for a row of softmax outputs `y` and upstream grad `g` is `y * (g - sum(y*g))`. Apply it to `dvalues` before scattering when `softmax=True`.
- **Indices dtype:** `scatter_` requires int64 index — always call `indices.long()`.
- **Run one test:** `pytest tests/amd/test_topk_bwd.py -x -k "f32"` on a gfx950 box.

---

## Task 1: TopK backward (scatter) + TopKFunction + topk()

**Files:**
- Modify: `quack/amd/topk.py` (add `topk_bwd`, `TopKFunction`, `topk`)
- Test: `tests/amd/test_topk_bwd.py`

- [ ] **Step 1: Write the failing test** — create `tests/amd/test_topk_bwd.py`:
```python
import pytest
import torch

from quack.amd.topk import topk


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M,N,k", [(64, 256, 8), (32, 1024, 16), (16, 128, 4)])
def test_topk_backward_matches_torch(dtype, M, N, k):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)

    vals, idx = topk(x, k)
    rvals, ridx = torch.topk(xr, k, dim=-1)

    tol = 2e-2 if dtype is torch.bfloat16 else 1e-5
    assert torch.allclose(vals.float(), rvals.float(), atol=tol, rtol=tol)

    g = torch.randn_like(vals)
    vals.backward(g)
    rvals.backward(g)
    # dx is a scatter of g into the topk positions; compare against torch's own grad
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=tol, rtol=tol)


def test_topk_indices_non_differentiable():
    x = torch.randn(8, 64, device="cuda", requires_grad=True)
    vals, idx = topk(x, 4)
    assert not idx.requires_grad
    vals.sum().backward()
    assert x.grad is not None
```

- [ ] **Step 2: Run to confirm it fails**

Run: `pytest tests/amd/test_topk_bwd.py::test_topk_indices_non_differentiable -x`
Expected: FAIL — `ImportError: cannot import name 'topk'` from `quack.amd.topk`.

- [ ] **Step 3: Implement** — add to `quack/amd/topk.py` (after `topk_fwd`, keeping imports at top of file):
```python
def topk_bwd(dvalues, values, indices, N, softmax=False):
    """Scatter ``dvalues`` back into a full ``(M, N)`` gradient at ``indices``.

    When ``softmax`` is True, ``values`` are the softmax outputs and the
    softmax Jacobian ``y*(g - sum(y*g))`` is applied before scattering.
    """
    if softmax:
        y = values
        dvalues = y * (dvalues - (dvalues * y).sum(-1, keepdim=True))
    M = indices.shape[0]
    dx = torch.zeros(M, N, dtype=dvalues.dtype, device=dvalues.device)
    dx.scatter_(1, indices.long(), dvalues)
    return dx


class TopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, softmax):
        values, indices, sm = topk_fwd(x, k, softmax=softmax)
        out_values = sm if softmax else values
        ctx.save_for_backward(out_values if softmax else None, indices)
        ctx.N = x.shape[-1]
        ctx.softmax = softmax
        ctx.mark_non_differentiable(indices)
        ctx.set_materialize_grads(False)
        return out_values, indices

    @staticmethod
    def backward(ctx, dvalues, dindices=None):
        saved_values, indices = ctx.saved_tensors
        if dvalues is None:
            return None, None, None
        dx = topk_bwd(dvalues, saved_values, indices, ctx.N, softmax=ctx.softmax)
        return dx, None, None


def topk(x, k, softmax=False):
    """Autograd-enabled top-k over the last dim. Returns ``(values, indices)``."""
    return TopKFunction.apply(x, k, softmax)
```
Ensure `import torch` and `from torch import Tensor` are present at the top of `quack/amd/topk.py` (they are — `topk_fwd` already uses `Tensor`).

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/amd/test_topk_bwd.py -x -k "backward or non_differentiable"`
Expected: PASS (3 shapes × 2 dtypes + the non-differentiable test).

- [ ] **Step 5: Commit**
```bash
git add quack/amd/topk.py tests/amd/test_topk_bwd.py
git commit -m "[AMD] TopK backward (scatter) + TopKFunction + topk()"
```

---

## Task 2: Softmax-fused backward path

**Files:**
- Modify: `tests/amd/test_topk_bwd.py`

The softmax path is already implemented in Task 1's `topk_bwd`/`TopKFunction`; this task adds the test that exercises and locks it.

- [ ] **Step 1: Write the failing test** — append to `tests/amd/test_topk_bwd.py`:
```python
def _ref_topk_softmax(x, k):
    rvals, ridx = torch.topk(x.float(), k, dim=-1)
    return torch.softmax(rvals, dim=-1), ridx


@pytest.mark.parametrize("M,N,k", [(64, 256, 8), (32, 512, 16)])
def test_topk_softmax_backward(M, N, k):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)

    sm, idx = topk(x, k, softmax=True)
    rsm, ridx = _ref_topk_softmax(xr, k)

    assert torch.allclose(sm.float(), rsm, atol=1e-5, rtol=1e-5)
    g = torch.randn_like(sm)
    sm.backward(g)
    rsm.backward(g)
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run to confirm it passes immediately (implementation already present)**

Run: `pytest tests/amd/test_topk_bwd.py::test_topk_softmax_backward -x`
Expected: PASS. If it FAILS on the gradient, the bug is in the Jacobian sign/reduction in `topk_bwd` — fix `topk_bwd`, not the test. (TDD note: this task's test guards behavior added in Task 1; a green result on first run is expected and acceptable here.)

- [ ] **Step 3: Commit**
```bash
git add tests/amd/test_topk_bwd.py
git commit -m "[AMD] TopK softmax-fused backward — lock gradient correctness"
```

---

## Task 3: Export `topk` from the package

**Files:**
- Modify: `quack/amd/topk.py` (`__all__`)
- Modify: `quack/amd/__init__.py`
- Modify: `quack/amd/nn.py` (re-export, only if it exists from Phase 1)
- Test: `tests/amd/test_topk_bwd.py`

- [ ] **Step 1: Write the failing test** — append to `tests/amd/test_topk_bwd.py`:
```python
def test_topk_exported():
    import quack.amd as qa
    assert hasattr(qa, "topk"), "quack.amd.topk (autograd) not exported"
```

- [ ] **Step 2: Run to confirm it fails**

Run: `pytest tests/amd/test_topk_bwd.py::test_topk_exported -x`
Expected: FAIL — `topk` not on `quack.amd`.

- [ ] **Step 3: Add `topk`/`TopKFunction`/`topk_bwd` to `quack/amd/topk.py`'s `__all__`**

Find the existing `__all__` in `quack/amd/topk.py` (currently `["topk_fwd"]` near line 121) and replace with:
```python
__all__ = ["topk_fwd", "topk", "topk_bwd", "TopKFunction"]
```

- [ ] **Step 4: Re-export from `quack/amd/__init__.py`**

Add:
```python
from quack.amd.topk import topk
```

- [ ] **Step 5: If `quack/amd/nn.py` exists (Phase 1 landed), surface it there too**

Add to `quack/amd/nn.py` imports and `__all__`:
```python
from quack.amd.topk import topk
```
and add `"topk"` to `nn.py`'s `__all__`. If `quack/amd/nn.py` does not exist yet, skip this step — the `__init__.py` export in Step 4 is sufficient.

- [ ] **Step 6: Run to confirm pass**

Run: `pytest tests/amd/test_topk_bwd.py -x`
Expected: all Task-1/2/3 tests PASS.

- [ ] **Step 7: Commit**
```bash
git add quack/amd/topk.py quack/amd/__init__.py tests/amd/test_topk_bwd.py
git commit -m "[AMD] Export autograd topk from package"
```

---

## Roadmap — native non-f32 TopK kernel (separate follow-on)

Out of scope for this plan; open a dedicated plan when perf on bf16/fp16 top-k matters.

**Why separate:** the bitonic compare-swap in `quack/amd/topk_kernel.py` and `quack/amd/topk_lds.py` asserts f32 (`topk_kernel.py:183`, `topk_lds.py:221`) and sorts on raw f32 bit-patterns. Supporting bf16/fp16 natively means doing the compare in the compute type (or an order-preserving int key) inside the FlyDSL kernel — real kernel work with its own correctness surface (NaN handling, tie-breaking, the `-inf` pad sentinel in a narrower type). Today those dtypes route to `torch.topk`, so the **autograd story is already complete** for them; only the FlyDSL-kernel perf path is missing. **Done criteria for that future plan:** bf16/fp16 top-k runs the FlyDSL kernel (not the fallback) for N ≤ 4096 with values/indices matching `torch.topk`, and the backward (already dtype-agnostic here) continues to match.

---

## Self-Review

**Spec coverage:** Priority #2 from the parity review was "TopK backward + non-f32." Backward (the headline gap — the only family with *no* gradient) is fully delivered for f32/bf16/fp16 via Tasks 1–3. Non-f32 is addressed honestly: forward already works via fallback (so autograd is complete), and the harder native-kernel path is scoped as an explicit follow-on rather than under-specified here. ✓

**Placeholder scan:** All steps contain complete, runnable code and exact commands. The only deferred item is the native non-f32 kernel, which is explicitly a separate plan with its own done-criteria, not a hidden TODO. ✓

**Type consistency:** `topk_fwd` returns the 3-tuple `(values, indices, softmax_values_or_None)` per `quack/amd/topk.py:69`; `TopKFunction.forward` consumes all three and returns `(out_values, indices)`; `topk_bwd(dvalues, values, indices, N, softmax)` signature is consistent across its definition, the `TopKFunction.backward` call site, and the mirror of NVIDIA's `topk_bwd`. `indices.long()` handles the int32/int64 scatter requirement. ✓
