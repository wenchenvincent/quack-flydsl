# AMD FlyDSL Port — High-Level Layer + Parity Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement **Phase 1** task-by-task. Steps use checkbox (`- [ ]`) syntax. Phases 2–5 are a scoped roadmap: each must be expanded into its own full plan (via superpowers:writing-plans) before execution — do NOT execute them from the outlines here.

**Goal:** Close the highest-leverage gaps between the AMD FlyDSL port (`quack/amd/`) and the NVIDIA CuTe-DSL original (`quack/`), starting with a drop-in high-level layer (autograd + `nn.Module`) over the reduction kernels that already pass.

**Architecture:** The AMD reduction kernels expose only bare `*_fwd` / `*_bwd` functions. Phase 1 adds a thin, well-tested high-level layer — `torch.autograd.Function` wrappers, functional entry points, and `nn.Module`s — in a single new file `quack/amd/nn.py`, wrapping the existing kernels without touching them. This mirrors NVIDIA's `RMSNormFunction`/`rmsnorm()`/`QuackRMSNorm` pattern but keeps the (already-large) kernel files focused on kernels. Phases 2–5 are independent follow-on subsystems.

**Tech Stack:** PyTorch autograd, FlyDSL kernels (already built), pytest on ROCm/gfx950 (MI355X). Reference implementations in float32 for correctness.

**Scope note (per writing-plans skill):** The five priorities are independent subsystems. Only **Phase 1** is specified to executable (TDD, bite-sized, real code) detail here. Phases 2–5 are sequenced roadmap items — each needs its own plan before it is executed.

---

## File Structure

| File | Responsibility | Phase |
|---|---|---|
| Create `quack/amd/nn.py` | High-level layer: autograd Functions, functional wrappers, `nn.Module`s for rmsnorm/layernorm/softmax/cross-entropy | 1 |
| Modify `quack/amd/__init__.py` | Export the new high-level names | 1 |
| Create `tests/amd/test_nn.py` | Autograd correctness (out + grads vs torch reference) for the new layer | 1 |

Phases 2–5 touch other files (`topk.py`, `topk_kernel.py`, `linear_training.py`, `reduce.py`, `reduction_base.py`) and are scoped in the roadmap section.

---

## Background the engineer needs

- **Run one AMD test:** `pytest tests/amd/test_nn.py -x -k "rmsnorm"` on a gfx950 box.
- **The reduction kernels are 2D, last-dim-contiguous.** `quack/amd/rmsnorm.py` and `quack/amd/softmax.py` operate on `(M, N)` where N is the normalized dim. Any batch dims must be flattened to a single leading M before calling, then restored after.
- **AMD `rmsnorm_fwd` signature** (`quack/amd/rmsnorm.py:1291`):
  ```python
  rmsnorm_fwd(x, weight=None, bias=None, residual=None, eps=1e-6,
              store_rstd=False, store_residual_out=False)
      -> (out, rstd_or_None, residual_out_or_None)
  ```
- **AMD `rmsnorm_bwd` signature** (`quack/amd/rmsnorm.py:1382`):
  ```python
  rmsnorm_bwd(x, weight, dout, rstd, eps=1e-6) -> (dx, dw)
  ```
- **AMD `softmax_fwd(x) -> y` and `softmax_bwd(dy, y) -> dx`** (`quack/amd/softmax.py:390,398`), both 2D last-dim.
- **AMD `cross_entropy_fwd_bwd(...)`** (`quack/amd/cross_entropy.py:824`) computes loss and `dx` together (in-place dx), and supports `ignore_index`, `label_smoothing`, `loss_weight`.
- **Known constraint — runtime eps:** the AMD rmsnorm kernel bakes `eps` at compile time; any value other than the default `1e-6` raises `NotImplementedError("runtime eps is a later pass")` (`quack/amd/rmsnorm.py:1316` etc.). Phase 1 threads `eps` through and lets that error surface; the wrapper does NOT try to work around it. This is documented behavior, not a Phase-1 bug.
- **NVIDIA reference to mirror:** `RMSNormFunction` in `quack/rmsnorm.py` — forward flattens, saves `(x_or_residual_out, weight, rstd)`, backward returns `(dx, dw, ...)`. We copy the *shape* of this, not its residual/prenorm richness (those are follow-ups).

---

## Phase 1 — High-level autograd + module layer

### Task 1: RMSNorm autograd Function + functional + module

**Files:**
- Create: `quack/amd/nn.py`
- Test: `tests/amd/test_nn.py`

- [ ] **Step 1: Write the failing test**

Create `tests/amd/test_nn.py`:
```python
import pytest
import torch

from quack.amd.nn import RMSNorm, rmsnorm


def _ref_rmsnorm(x, w, eps=1e-6):
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * rms) * w.float()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M,N", [(128, 256), (32, 1024)])
def test_rmsnorm_autograd(dtype, M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(N, device="cuda", dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)

    out = rmsnorm(x, w)
    ref = _ref_rmsnorm(xr, wr).to(dtype)

    tol = 2e-2 if dtype is torch.bfloat16 else 1e-4
    assert torch.allclose(out.float(), ref.float(), atol=tol, rtol=tol)

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=tol * 5, rtol=tol * 5)
    assert torch.allclose(w.grad.float(), wr.grad.float(), atol=tol * 5, rtol=tol * 5)


def test_rmsnorm_module():
    torch.manual_seed(0)
    m = RMSNorm(256, device="cuda", dtype=torch.float32)
    x = torch.randn(64, 256, device="cuda", requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert y.shape == (64, 256)
    assert m.weight.grad is not None
    assert x.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/amd/test_nn.py::test_rmsnorm_module -x`
Expected: FAIL with `ModuleNotFoundError: No module named 'quack.amd.nn'`.

- [ ] **Step 3: Write minimal implementation**

Create `quack/amd/nn.py`:
```python
# Copyright (c) 2026, AMD.

"""High-level layer for the AMD reduction kernels.

Wraps the bare ``*_fwd``/``*_bwd`` kernels in ``quack.amd.{rmsnorm,softmax,
cross_entropy}`` with ``torch.autograd.Function``s, functional entry points,
and ``nn.Module``s — mirroring the NVIDIA ``quack.rmsnorm`` surface. The
kernel files stay kernel-only; this is the user-facing layer.
"""

from typing import Optional

import torch
from torch import Tensor

from quack.amd.rmsnorm import rmsnorm_fwd, rmsnorm_bwd


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        need_grad = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        out, rstd, _ = rmsnorm_fwd(x, weight, eps=eps, store_rstd=need_grad)
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, rstd = ctx.saved_tensors
        dx, dw = rmsnorm_bwd(x, weight, dout.contiguous(), rstd, eps=ctx.eps)
        return dx, dw, None


def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm over the last dim, autograd-enabled. Flattens leading dims."""
    n = x.shape[-1]
    out = RMSNormFunction.apply(x.reshape(-1, n), weight, eps)
    return out.reshape(x.shape)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        return rmsnorm(x, self.weight, self.eps)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/amd/test_nn.py -x -k "rmsnorm"`
Expected: PASS (2 params × 2 dtypes + module test).

- [ ] **Step 5: Commit**

```bash
git add quack/amd/nn.py tests/amd/test_nn.py
git commit -m "[AMD] High-level layer: RMSNorm autograd Function + module"
```

### Task 2: LayerNorm autograd Function + functional + module

**Files:**
- Modify: `quack/amd/nn.py`
- Test: `tests/amd/test_nn.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/amd/test_nn.py`:
```python
from quack.amd.nn import LayerNorm, layernorm


def _ref_layernorm(x, w, b, eps=1e-6):
    xf = x.float()
    mu = xf.mean(-1, keepdim=True)
    var = xf.var(-1, keepdim=True, unbiased=False)
    return ((xf - mu) * torch.rsqrt(var + eps)) * w.float() + b.float()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_layernorm_autograd(dtype):
    torch.manual_seed(0)
    M, N = 128, 256
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(N, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(N, device="cuda", dtype=dtype, requires_grad=True)
    xr, wr, br = (t.detach().clone().requires_grad_(True) for t in (x, w, b))

    out = layernorm(x, w, b)
    ref = _ref_layernorm(xr, wr, br).to(dtype)
    tol = 2e-2 if dtype is torch.bfloat16 else 1e-4
    assert torch.allclose(out.float(), ref.float(), atol=tol, rtol=tol)

    g = torch.randn_like(out)
    out.backward(g); ref.backward(g)
    for a, bb in ((x, xr), (w, wr), (b, br)):
        assert torch.allclose(a.grad.float(), bb.grad.float(), atol=tol * 5, rtol=tol * 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/amd/test_nn.py::test_layernorm_autograd -x`
Expected: FAIL with `ImportError: cannot import name 'layernorm'`.

- [ ] **Step 3: Write minimal implementation**

Add to `quack/amd/nn.py` (imports and body):
```python
from quack.amd.rmsnorm import layernorm_fwd, layernorm_bwd
```
Verified signatures (`quack/amd/rmsnorm.py:1409`): `layernorm_bwd(x, weight, dout, rstd, mean, bias=None, eps) -> (dx, dw, db_or_None)` — `db` is produced only when `bias` is passed, so the Function must save and forward `bias`.
```python
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        need_grad = any(ctx.needs_input_grad[:3])
        out, rstd, mean = layernorm_fwd(
            x, weight, bias=bias, eps=eps, store_rstd=need_grad, store_mean=need_grad
        )
        ctx.save_for_backward(x, weight, bias, rstd, mean)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, bias, rstd, mean = ctx.saved_tensors
        dx, dw, db = layernorm_bwd(
            x, weight, dout.contiguous(), rstd, mean, bias=bias, eps=ctx.eps
        )
        return dx, dw, db, None


def layernorm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-6) -> Tensor:
    n = x.shape[-1]
    out = LayerNormFunction.apply(x.reshape(-1, n), weight, bias, eps)
    return out.reshape(x.shape)


class LayerNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        return layernorm(x, self.weight, self.bias, self.eps)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/amd/test_nn.py -x -k "layernorm"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quack/amd/nn.py tests/amd/test_nn.py
git commit -m "[AMD] High-level layer: LayerNorm autograd Function + module"
```

### Task 3: Softmax autograd Function + functional

**Files:**
- Modify: `quack/amd/nn.py`
- Test: `tests/amd/test_nn.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/amd/test_nn.py`:
```python
from quack.amd.nn import softmax as amd_softmax


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M,N", [(128, 256), (64, 1024)])
def test_softmax_autograd(dtype, M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    out = amd_softmax(x, dim=-1)
    ref = torch.softmax(xr.float(), dim=-1).to(dtype)
    tol = 2e-2 if dtype is torch.bfloat16 else 1e-4
    assert torch.allclose(out.float(), ref.float(), atol=tol, rtol=tol)
    g = torch.randn_like(out)
    out.backward(g); ref.backward(g)
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=tol * 5, rtol=tol * 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/amd/test_nn.py::test_softmax_autograd -x`
Expected: FAIL with `ImportError: cannot import name 'softmax'`.

- [ ] **Step 3: Write minimal implementation**

Add to `quack/amd/nn.py`:
```python
from quack.amd.softmax import softmax_fwd, softmax_bwd


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = softmax_fwd(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensors
        return softmax_bwd(dy.contiguous(), y)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Softmax over ``dim``. The AMD kernel is last-dim 2D, so move ``dim`` last."""
    if dim != -1 and dim != x.ndim - 1:
        x = x.movedim(dim, -1)
        y = softmax(x, dim=-1)
        return y.movedim(-1, dim)
    n = x.shape[-1]
    y = SoftmaxFunction.apply(x.reshape(-1, n))
    return y.reshape(x.shape)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/amd/test_nn.py -x -k "softmax"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quack/amd/nn.py tests/amd/test_nn.py
git commit -m "[AMD] High-level layer: Softmax autograd Function"
```

### Task 4: CrossEntropy autograd Function + functional (with reduction modes)

**Files:**
- Modify: `quack/amd/nn.py`
- Test: `tests/amd/test_nn.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/amd/test_nn.py`:
```python
from quack.amd.nn import cross_entropy as amd_ce


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_cross_entropy_autograd(reduction):
    torch.manual_seed(0)
    M, V = 256, 512
    x = torch.randn(M, V, device="cuda", dtype=torch.float32, requires_grad=True)
    tgt = torch.randint(0, V, (M,), device="cuda")
    xr = x.detach().clone().requires_grad_(True)

    loss = amd_ce(x, tgt, reduction=reduction)
    ref = torch.nn.functional.cross_entropy(xr, tgt, reduction=reduction)

    assert torch.allclose(loss.float(), ref.float(), atol=1e-3, rtol=1e-3)
    (loss.sum() if reduction == "none" else loss).backward()
    (ref.sum() if reduction == "none" else ref).backward()
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=1e-3, rtol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/amd/test_nn.py::test_cross_entropy_autograd -x`
Expected: FAIL with `ImportError: cannot import name 'cross_entropy'`.

- [ ] **Step 3: Write minimal implementation**

Verified signatures (`quack/amd/cross_entropy.py:905,983`): `cross_entropy_fwd(x, target, return_lse=False, *, ignore_index=-100, ...) -> (loss, lse_or_None)` (pass `return_lse=True` to get `lse`), and `cross_entropy_bwd(x, target, lse, dloss=None, *, ignore_index=-100, ...) -> dx` (note arg order is `lse` then `dloss`). `target` must be int32/int64, 1D; loss is float32.
```python
from quack.amd.cross_entropy import cross_entropy_fwd, cross_entropy_bwd


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, ignore_index):
        loss, lse = cross_entropy_fwd(x, target, return_lse=True, ignore_index=ignore_index)
        ctx.save_for_backward(x, target, lse)
        ctx.ignore_index = ignore_index
        return loss  # per-row, float32

    @staticmethod
    def backward(ctx, dloss):
        x, target, lse = ctx.saved_tensors
        dx = cross_entropy_bwd(x, target, lse, dloss.contiguous(),
                               ignore_index=ctx.ignore_index)
        return dx, None, None


def cross_entropy(x: Tensor, target: Tensor, ignore_index: int = -100,
                  reduction: str = "mean") -> Tensor:
    loss = CrossEntropyFunction.apply(x, target, ignore_index)  # per-row
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        valid = (target != ignore_index).sum().clamp_min(1)
        return loss.sum() / valid
    raise ValueError(f"unknown reduction: {reduction!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/amd/test_nn.py -x -k "cross_entropy"`
Expected: PASS for all three reductions.

- [ ] **Step 5: Commit**

```bash
git add quack/amd/nn.py tests/amd/test_nn.py
git commit -m "[AMD] High-level layer: CrossEntropy autograd + reduction modes"
```

### Task 5: Export the high-level layer from the package

**Files:**
- Modify: `quack/amd/__init__.py`
- Modify: `quack/amd/nn.py` (add `__all__`)
- Test: `tests/amd/test_nn.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/amd/test_nn.py`:
```python
def test_public_exports():
    import quack.amd as qa
    for name in ("RMSNorm", "LayerNorm", "rmsnorm", "layernorm",
                 "softmax", "cross_entropy"):
        assert hasattr(qa, name), f"quack.amd.{name} not exported"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/amd/test_nn.py::test_public_exports -x`
Expected: FAIL — names not on `quack.amd`.

- [ ] **Step 3: Add `__all__` to `quack/amd/nn.py`**

At the end of `quack/amd/nn.py`:
```python
__all__ = [
    "RMSNormFunction", "rmsnorm", "RMSNorm",
    "LayerNormFunction", "layernorm", "LayerNorm",
    "SoftmaxFunction", "softmax",
    "CrossEntropyFunction", "cross_entropy",
]
```

- [ ] **Step 4: Re-export from `quack/amd/__init__.py`**

Add to `quack/amd/__init__.py`:
```python
from quack.amd.nn import (
    RMSNorm, LayerNorm, rmsnorm, layernorm, softmax, cross_entropy,
)
```

- [ ] **Step 5: Run test + full new-suite to verify**

Run: `pytest tests/amd/test_nn.py -x`
Expected: all Phase-1 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add quack/amd/nn.py quack/amd/__init__.py tests/amd/test_nn.py
git commit -m "[AMD] Export high-level nn layer from package root"
```

---

## Phases 2–5 — Roadmap (each becomes its own plan)

Each item below is a **separate subsystem**. Expand into a full TDD plan (superpowers:writing-plans) before executing. Ordered by leverage.

### Phase 2 — TopK backward + non-f32

**Why:** the only reduction family with no backward at all; also f32-only.
**Key files:** `quack/amd/topk_kernel.py`, `quack/amd/topk_lds.py`, `quack/amd/topk.py`, new autograd in `quack/amd/nn.py`; tests `tests/amd/test_topk.py`.
**Approach:** (a) add a `topk_bwd` kernel — scatter `dvalues` back to the gathered source indices (mirror NVIDIA `_topk_bwd` in `quack/topk.py:460`); (b) add bf16/fp16 support by templating the kernel's element dtype (currently asserts f32 at `topk_kernel.py:183`, `topk_lds.py:221`); (c) wrap in `TopKFunction` + `topk()` with the `softmax` fuse flag. **Risk:** the bitonic sort is written for f32 compare-swap; non-f32 needs a compare in the compute type with correct ordering. **Done:** gradient matches `torch.topk` autograd for f32/bf16/fp16; N>4096 no longer silently falls back where a real kernel exists.

### Phase 3 — Enable / justify the fused activation-backward path

**Why:** the kernel-fused dact/dgated backward exists but is default-disabled (`quack/amd/linear_training.py:219` `_fused_dact_eligible`), so training silently runs the torch path.
**Key files:** `quack/amd/linear_training.py`, `quack/amd/gemm_gfx950_splitk.py`; benches in `tests/amd/`.
**Approach:** bench `gemm_splitk(dact_activation=…)` vs the torch elementwise path across the real MLP shapes; either (a) land the splitk perf work so the fused path wins and flip `_fused_dact_eligible` to on, or (b) if it can't win, document the decision at the gate and add a regression bench so the choice is evidence-backed, not incidental. **Risk:** splitk trails hipBLASLt at MLP shapes today — this may be a perf project, not a wiring change. **Done:** the default backend for act-backward is chosen by a benchmark, recorded in a perf journal, and the eligibility gate reflects it.

### Phase 4 — Consolidate the two linear autograd stacks

**Why:** `quack/amd/linear.py` (custom NN/TN-kernel backward) and `quack/amd/linear_training.py` (`torch.mm` backward) both implement linear+activation autograd, wired to different entry points (`mlp_train` vs `mlp_func_train`) — a drift hazard.
**Key files:** `quack/amd/linear.py`, `quack/amd/linear_training.py`, `quack/amd/mlp.py`; tests `tests/amd/test_linear.py`, `test_linear_train.py`.
**Approach:** pick the custom-kernel stack as the single source of truth, port any `linear_training.py`-only behavior (fused-dact eligibility from Phase 3, gated concat) onto it, redirect `mlp_func_train`/`gated_mlp_func_train` to it, and delete the duplicate `_LinearActFunction`/`_MLPActFunction`. **Risk:** the two stacks have subtly different numerics (custom kernels vs `torch.mm`); the merged path must pass both existing test suites. **Done:** one autograd implementation for linear+act; both test files green; no `torch.mm`-only backward remains except as a documented fallback.

### Phase 5 — Large-N / huge-vocab reductions

**Why:** no cluster/online-softmax reduction and no cross-entropy `lse_partial` → long-sequence and large-vocabulary shapes NVIDIA covers are out of reach.
**Key files:** `quack/amd/reduce.py`, `quack/amd/reduction_base.py`, `quack/amd/softmax.py`, `quack/amd/cross_entropy.py`; tests `tests/amd/test_softmax.py`, `test_cross_entropy.py`.
**Approach:** (a) add a multi-block/grid-stride reduction path in `reduce.py` (AMD has no cluster concept; use a two-pass grid-stride + workspace instead of NVIDIA's cluster reduce); (b) add a split-vocab two-stage cross-entropy (`lse_partial`) mirroring `quack/cross_entropy.py:655`; (c) lift the Float32-only reduction-dtype limit at `reduction_base.py:105` if the packed max+sum path is needed. **Risk:** the largest design gap — AMD's block-only `reduction_base` needs a genuinely new multi-block mechanism, not a wrapper. **Done:** softmax and cross-entropy pass at N≥128K with numerical parity to a float32 reference.

---

## Self-Review

**Spec coverage:** All five priorities from the review are represented — Phase 1 (autograd/module layer, priority #1) is fully specified; Phases 2–5 map 1:1 to priorities #2–#5. ✓

**Placeholder scan:** Phase 1 steps contain complete, runnable code and exact commands. All kernel signatures used in the code are verified against source and inlined (rmsnorm_fwd/bwd, layernorm_fwd/bwd with the `bias`-for-`db` requirement, softmax_fwd/bwd, cross_entropy_fwd with `return_lse=True`, cross_entropy_bwd with `(x, target, lse, dloss)` order) — no deferred "verify later" work remains. Phases 2–5 are intentionally roadmap-level and labeled as needing their own plans (per the skill's scope check for independent subsystems). ✓

**Type consistency:** All kernel calls match verified source signatures — `quack/amd/rmsnorm.py:1291,1382,1409`, `quack/amd/softmax.py:390,398`, `quack/amd/cross_entropy.py:905,983`. Function/module names (`RMSNormFunction`/`rmsnorm`/`RMSNorm`, `LayerNormFunction`/`layernorm`/`LayerNorm`, `SoftmaxFunction`/`softmax`, `CrossEntropyFunction`/`cross_entropy`) are consistent across tasks and the `__all__`/`__init__` exports in Task 5. ✓
