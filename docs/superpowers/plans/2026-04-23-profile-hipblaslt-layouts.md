# Phase 1 — Profile hipBLASLt NT/NN/TN layouts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure hipBLASLt per-shape kernel time + HW counters for the NT (fwd), NN (DX), and TN (DW) layouts that training needs. Produce a per-shape Tier-2 target (our-throughput ÷ hipBLASLt-throughput ≥ 0.9) so NN and TN kernel work can be judged honestly.

**Architecture:** A single Python bench module (`tests/amd/bench_hipblaslt_layouts.py`) that dispatches a shape × layout matrix through `torch.matmul`, times each kernel with CUDA events, and can be re-run under rocprofv3 to collect HW counters. A second commit-artifact is the analysis memo at `docs/superpowers/specs/hipblaslt_layout_profile.md`.

**Tech Stack:** Python 3, PyTorch (ROCm build), torch.cuda events, rocprofv3 v1.0.0 (`/opt/rocm/bin/rocprofv3`), pytest for the unit tests on the bench harness.

**Parent spec:** `docs/superpowers/specs/2026-04-23-nn-tn-kernels-design.md` (§3.1).

---

## File Structure

- **Create** `tests/amd/bench_hipblaslt_layouts.py` — main bench harness. ~250 LoC target. Responsibilities: shape matrix, per-layout torch dispatch, timing, stdout report, rocprofv3-launch mode.
- **Create** `tests/amd/pmc_counters.txt` — rocprofv3 PMC counter spec file (one counter per line).
- **Create** `tests/amd/test_bench_hipblaslt_layouts.py` — unit tests for the bench module's helpers (shape FLOPS calc, layout tensor builder, CSV parser). ~80 LoC.
- **Create** `docs/superpowers/specs/hipblaslt_layout_profile.md` — analysis memo (deliverable).

No modifications to existing code in this commit.

---

## Layout semantics (reference, for the implementer)

For `y = x @ W.T` (FWD): M=batch, K=hidden, N=out. Torch call: `torch.matmul(x[M,K], W[N,K].T)` where `W.T` is a stride-view. This goes to hipBLASLt's NT-natural kernel.

For `dx = dy @ W` (DX, "NN"): M=batch, N_contract=out, K_new=hidden. Torch call: `torch.matmul(dy[M,N], W[N,K])` — both row-major, no view.

For `dW = dy.T @ x` (DW, "TN"): M_out=out, K_contract=batch, N_new=hidden. Torch call: `torch.matmul(dy.t()[N,M], x[M,K])` where `dy.t()` is a stride-view.

Shape conventions match the spec §3.1:

```
MLP-fwd (NT):  (M=bs,           K=hidden,        N=4*hidden)
MLP-dx  (NN):  (M=bs,           K=4*hidden,      N=hidden)
MLP-dw  (TN):  (M=4*hidden,     K=bs,            N=hidden)
               (M=hidden,       K=bs,            N=4*hidden)
```

`(bs, hidden)` tuples: `(2048, 1024), (8192, 4096), (32768, 8192)`.

---

## Task 1 — PMC counter file + validation

**Files:**
- Create: `tests/amd/pmc_counters.txt`

- [ ] **Step 1: Check which counters rocprofv3 knows about**

Run: `rocprofv3 --list-counters 2>&1 | grep -E "^(SQ_|TCP_|GRBM_|LDS_|TCC_|SPI_)" | head -60`
Expected: a list of valid counter names printed. Look specifically for the counters named in our design spec §3.1 (`SQ_INSTS_MFMA`, `SQ_INSTS_VALU`, `SQ_INSTS_VMEM`, `SQ_INSTS_LDS`, `GRBM_GUI_ACTIVE`) — confirm each appears. For the ones named speculatively in the spec (`TCP_UTCL1_REQUEST`, `LDS_BANK_CONFLICT_READ`, `LDS_BANK_CONFLICT_WRITE`), the listed name on this rocprofv3 version may differ. **If a name doesn't appear verbatim**, grep for a near-match (e.g., `grep -i "bank_conflict"`, `grep -i "utcl1"`) and substitute the real name.

- [ ] **Step 2: Create the counter file**

Create `tests/amd/pmc_counters.txt`:

```
pmc: SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INSTS_LDS
pmc: GRBM_GUI_ACTIVE SQ_WAVES
pmc: TCP_UTCL1_REQUEST_sum TCP_UTCL1_HIT_sum
pmc: SQ_LDS_BANK_CONFLICT
```

Replace the third and fourth `pmc:` lines' counters with whatever names Step 1 confirmed on this machine. Each `pmc:` line is one "collection pass" — rocprofv3 runs the target binary once per pass, so fewer passes = faster but fewer counters per run. With 4 passes our target is collected 4 times (~4× bench runtime — acceptable).

- [ ] **Step 3: Validate the counter file**

Run: `rocprofv3 -i tests/amd/pmc_counters.txt -- /bin/true 2>&1 | tail -20`
Expected: rocprofv3 reports it parsed the PMC file successfully (either "profiling complete" or similar; no "unknown counter" errors). If any counter name rejects, edit the file to remove or rename the offending counter.

- [ ] **Step 4: Stage the file (no commit yet — we'll commit at the end of the plan)**

Run: `git add tests/amd/pmc_counters.txt`

---

## Task 2 — Bench harness scaffolding

**Files:**
- Create: `tests/amd/bench_hipblaslt_layouts.py`
- Create: `tests/amd/test_bench_hipblaslt_layouts.py`

- [ ] **Step 1: Write the unit test first**

Create `tests/amd/test_bench_hipblaslt_layouts.py`:

```python
# Copyright (c) 2026, AMD.

"""Unit tests for tests.amd.bench_hipblaslt_layouts helpers.

Covers the pure-Python helpers only (shape-matrix / FLOPS calc / layout
tensor builders / CSV parser) — not the actual hipBLASLt run.
"""

import pytest
import torch

from tests.amd import bench_hipblaslt_layouts as B


def test_shapes_list_non_empty():
    assert len(B.SHAPES) >= 6
    for entry in B.SHAPES:
        assert set(entry.keys()) >= {"name", "M", "N", "K", "layout", "role"}


def test_flops_calc():
    assert B.matmul_flops(1024, 2048, 512) == 2 * 1024 * 2048 * 512


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_build_layout_nt_produces_matching_output_shape():
    M, N, K = 128, 256, 64
    A, B_t, matmul_fn = B.build_layout("NT", M, N, K, torch.float16)
    y = matmul_fn(A, B_t)
    assert y.shape == (M, N)
    assert A.shape == (M, K)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_build_layout_nn_produces_matching_output_shape():
    M, N, K = 128, 64, 256
    A, B_t, matmul_fn = B.build_layout("NN", M, N, K, torch.float16)
    y = matmul_fn(A, B_t)
    assert y.shape == (M, N)
    # NN: both row-major, no transpose view involved
    assert A.stride(-1) == 1 and B_t.stride(-1) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_build_layout_tn_produces_matching_output_shape():
    M, N, K = 64, 128, 256
    A, B_t, matmul_fn = B.build_layout("TN", M, N, K, torch.float16)
    y = matmul_fn(A, B_t)
    assert y.shape == (M, N)
    # TN: A is a transpose view of some (K, M) tensor
    assert A.stride(-2) == 1
```

- [ ] **Step 2: Run the test — expect ImportError / NameError**

Run: `cd /workspace/quack && PYTHONPATH=. pytest tests/amd/test_bench_hipblaslt_layouts.py -x 2>&1 | tail -15`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.amd.bench_hipblaslt_layouts'` or `AttributeError: module has no attribute 'SHAPES'`.

- [ ] **Step 3: Create the bench module skeleton**

Create `tests/amd/bench_hipblaslt_layouts.py`:

```python
# Copyright (c) 2026, AMD.

"""Bench hipBLASLt across the three GEMM layouts training needs.

Usage (direct run):
    PYTHONPATH=/workspace/quack python -m tests.amd.bench_hipblaslt_layouts

Usage (under rocprofv3):
    PYTHONPATH=/workspace/quack rocprofv3 -i tests/amd/pmc_counters.txt \\
        -- python -m tests.amd.bench_hipblaslt_layouts --shape 4096x4096x4096 --layout NT

The per-shape/per-layout table lands in stdout; the rocprofv3 counter
CSVs land in ``./rocprofv3-out/``. See the parent spec at
docs/superpowers/specs/2026-04-23-nn-tn-kernels-design.md §3.1.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable, Tuple

import torch


# (bs, hidden) covering three training regimes.
_BS_HIDDEN = [(2048, 1024), (8192, 4096), (32768, 8192)]


def _mlp_shapes() -> list[dict]:
    out = []
    for bs, h in _BS_HIDDEN:
        out.append({
            "name": f"fwd_bs{bs}_h{h}",
            "role": "MLP-fwd",
            "layout": "NT",
            "M": bs, "K": h, "N": 4 * h,
        })
        out.append({
            "name": f"dx_bs{bs}_h{h}",
            "role": "MLP-dx",
            "layout": "NN",
            "M": bs, "K": 4 * h, "N": h,
        })
        # Two DW variants: down-proj grad (wide K, narrow MN) and up-proj grad.
        out.append({
            "name": f"dw_down_bs{bs}_h{h}",
            "role": "MLP-dw-down",
            "layout": "TN",
            "M": 4 * h, "K": bs, "N": h,
        })
        out.append({
            "name": f"dw_up_bs{bs}_h{h}",
            "role": "MLP-dw-up",
            "layout": "TN",
            "M": h, "K": bs, "N": 4 * h,
        })
    return out


SHAPES: list[dict] = _mlp_shapes()


def matmul_flops(M: int, N: int, K: int) -> int:
    return 2 * M * N * K


def build_layout(
    layout: str, M: int, N: int, K: int, dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Callable]:
    """Return (A, B, matmul_fn) such that matmul_fn(A, B) has shape (M, N).

    A and B are allocated on cuda with the given dtype. matmul_fn is
    torch.matmul with the stride-view (if any) already applied so the call
    site is always ``matmul_fn(A, B)`` regardless of layout.
    """
    assert layout in ("NT", "NN", "TN"), layout
    if layout == "NT":
        # y = A @ B.T; A row-major (M,K); B row-major (N,K) stored — pass as view B.T.
        A = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
        B_raw = torch.randn(N, K, device="cuda", dtype=dtype) * 0.1
        B = B_raw.transpose(-1, -2)  # stride view (K, N), stride (1, K)
        assert A.shape == (M, K) and B.shape == (K, N)
        return A, B, torch.matmul
    if layout == "NN":
        # y = A @ B; both row-major.
        A = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
        B = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
        return A, B, torch.matmul
    # TN: A = (K, M) row-major used as A.T → shape (M, K) with stride (1, K).
    A_raw = torch.randn(K, M, device="cuda", dtype=dtype) * 0.1
    A = A_raw.transpose(-1, -2)
    B = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
    assert A.shape == (M, K) and B.shape == (K, N)
    return A, B, torch.matmul


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default=None, help="M x N x K (e.g. 4096x4096x4096)")
    parser.add_argument("--layout", default=None, choices=["NT", "NN", "TN", None])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "f16"])
    args = parser.parse_args()
    # Placeholder — filled in by Task 3.
    print("bench harness scaffold OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the unit tests — expect PASS**

Run: `cd /workspace/quack && PYTHONPATH=. pytest tests/amd/test_bench_hipblaslt_layouts.py -x 2>&1 | tail -10`
Expected: `5 passed` (or `2 passed, 3 skipped` if CUDA isn't available in this shell).

- [ ] **Step 5: Stage the bench module and test**

Run: `git add tests/amd/bench_hipblaslt_layouts.py tests/amd/test_bench_hipblaslt_layouts.py`

---

## Task 3 — Timing + stdout report (no counters yet)

**Files:**
- Modify: `tests/amd/bench_hipblaslt_layouts.py`

- [ ] **Step 1: Replace `main()` with the real run + add timing helpers**

Open `tests/amd/bench_hipblaslt_layouts.py` and **replace** the `main()` function (and nothing else) with:

```python
def _bench_cuda_events(fn, warmup: int = 5, iters: int = 30) -> float:
    """Return the minimum of `iters` CUDA-event-timed runs, in seconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for s, e in events:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return min(s.elapsed_time(e) * 1e-3 for s, e in events)


@dataclass
class BenchResult:
    name: str
    role: str
    layout: str
    M: int
    N: int
    K: int
    dtype: str
    seconds: float
    tflops: float


def _bench_one(shape: dict, dtype: torch.dtype, dtype_str: str) -> BenchResult:
    A, B, matmul_fn = build_layout(shape["layout"], shape["M"], shape["N"], shape["K"], dtype)
    # Pre-allocated output to avoid per-iter allocation.
    out = torch.empty(shape["M"], shape["N"], device="cuda", dtype=dtype)

    def run():
        torch.matmul(A, B, out=out)

    t = _bench_cuda_events(run)
    return BenchResult(
        name=shape["name"], role=shape["role"], layout=shape["layout"],
        M=shape["M"], N=shape["N"], K=shape["K"], dtype=dtype_str,
        seconds=t, tflops=matmul_flops(shape["M"], shape["N"], shape["K"]) / t / 1e12,
    )


def _print_header():
    print(
        f"{'shape':<30}{'role':<14}{'lay':<5}{'dtype':<6}"
        f"{'ms':>10}{'TFLOP/s':>10}"
    )
    print("-" * 80)


def _print_row(r: BenchResult):
    shape_s = f"{r.M}x{r.N}x{r.K}"
    print(
        f"{shape_s:<30}{r.role:<14}{r.layout:<5}{r.dtype:<6}"
        f"{r.seconds * 1e3:>10.3f}{r.tflops:>10.2f}"
    )


def _filter_shapes(shape_filter: str | None, layout_filter: str | None) -> list[dict]:
    out = []
    for s in SHAPES:
        if shape_filter is not None:
            M, N, K = (int(x) for x in shape_filter.split("x"))
            if (s["M"], s["N"], s["K"]) != (M, N, K):
                continue
        if layout_filter is not None and s["layout"] != layout_filter:
            continue
        out.append(s)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default=None, help="MxNxK (e.g. 4096x4096x4096)")
    parser.add_argument("--layout", default=None, choices=["NT", "NN", "TN"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "f16"])
    args = parser.parse_args()

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shapes = _filter_shapes(args.shape, args.layout)
    if not shapes:
        print(f"No shapes match filter shape={args.shape} layout={args.layout}")
        return 1

    _print_header()
    for s in shapes:
        r = _bench_one(s, torch_dtype, args.dtype)
        _print_row(r)
    return 0
```

- [ ] **Step 2: Add a targeted test for `_bench_cuda_events` and `_bench_one`**

Append to `tests/amd/test_bench_hipblaslt_layouts.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_bench_cuda_events_returns_positive():
    t = B._bench_cuda_events(lambda: torch.zeros(1, device="cuda"), warmup=1, iters=3)
    assert t > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_bench_one_small_shape_runs():
    shape = {"name": "tiny", "role": "sanity", "layout": "NT", "M": 128, "N": 128, "K": 64}
    r = B._bench_one(shape, torch.float16, "f16")
    assert r.seconds > 0 and r.tflops > 0
```

- [ ] **Step 3: Run the unit tests**

Run: `cd /workspace/quack && PYTHONPATH=. pytest tests/amd/test_bench_hipblaslt_layouts.py -x 2>&1 | tail -10`
Expected: `7 passed`.

- [ ] **Step 4: Smoke-run the bench on one shape × 3 layouts**

Run:
```
cd /workspace/quack && PYTHONPATH=. python -m tests.amd.bench_hipblaslt_layouts \
    --shape 4096x4096x4096 --dtype bf16 2>&1 | tail -10
```
Expected: three rows printed (one per layout that happens to match M=N=K=4096 — probably 0 or 1 since that shape doesn't match our canonical shape matrix). **If no rows print**, re-run without `--shape` but limit to `bs=2048, h=1024`: run `PYTHONPATH=. python -m tests.amd.bench_hipblaslt_layouts --shape 2048x4096x1024 --layout NT --dtype bf16` — this should match the `fwd_bs2048_h1024` row.

- [ ] **Step 5: Stage the updated bench**

Run: `git add tests/amd/bench_hipblaslt_layouts.py tests/amd/test_bench_hipblaslt_layouts.py`

---

## Task 4 — rocprofv3 integration

**Files:**
- Modify: `tests/amd/bench_hipblaslt_layouts.py`

- [ ] **Step 1: Confirm rocprofv3 CSV output format**

Run a one-off profiling to capture the output format:

```
cd /workspace/quack && PYTHONPATH=. rocprofv3 \
    -i tests/amd/pmc_counters.txt \
    -d /tmp/rocprof_sanity/ \
    -- python -m tests.amd.bench_hipblaslt_layouts \
       --shape 2048x4096x1024 --layout NT --dtype bf16 2>&1 | tail -20
```
Expected: rocprofv3 creates `/tmp/rocprof_sanity/<pid>/counter_collection.csv` (or similar name — check `ls /tmp/rocprof_sanity/`). The CSV has per-dispatch rows with columns `Dispatch_Id, Kernel_Name, Start_Timestamp, End_Timestamp, Correlation_Id, <counter_1>, <counter_2>, ...`.

- [ ] **Step 2: Capture the exact CSV column names for the parser**

Run: `find /tmp/rocprof_sanity/ -name "*.csv" -exec head -1 {} \;`
Note the exact column header string — the parser in Step 4 uses it.

- [ ] **Step 3: Add the bench module's rocprofv3-launch helper**

Open `tests/amd/bench_hipblaslt_layouts.py` and append (after `_bench_one`):

```python
import csv
import os
import shutil
import subprocess
from pathlib import Path


def _run_under_rocprofv3(
    shape: dict, dtype_str: str, out_dir: Path, pmc_file: Path,
) -> Path:
    """Launch *ourselves* under rocprofv3 for one shape × layout.

    Returns the path to the counter CSV. ``out_dir`` is removed and recreated
    per call so PIDs don't collide across shapes.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shape_tag = f"{shape['M']}x{shape['N']}x{shape['K']}"
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":/workspace/quack"
    cmd = [
        "rocprofv3", "-i", str(pmc_file),
        "-d", str(out_dir), "--output-format", "csv",
        "--", "python", "-m", "tests.amd.bench_hipblaslt_layouts",
        "--shape", shape_tag, "--layout", shape["layout"],
        "--dtype", dtype_str, "--no-header",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"rocprofv3 failed for {shape['name']}:\n"
            f"stdout: {proc.stdout[-2000:]}\n"
            f"stderr: {proc.stderr[-2000:]}"
        )
    # rocprofv3 creates ``<out_dir>/<pid>/...csv`` — pick the most-recent CSV.
    csvs = sorted(out_dir.rglob("*counter*.csv"))
    if not csvs:
        raise RuntimeError(
            f"no counter CSV under {out_dir}; files: "
            f"{[p.name for p in out_dir.rglob('*')]}"
        )
    return csvs[-1]
```

- [ ] **Step 4: Add the `--no-header` flag to the argument parser (so the sub-invocation doesn't print the header twice)**

In the `main()` function, after the other `add_argument` lines and before `args = parser.parse_args()`:

```python
    parser.add_argument("--no-header", action="store_true", help="suppress the table header (used for sub-runs under rocprofv3)")
```

And in `main()`, change `_print_header()` to be guarded:

```python
    if not args.no_header:
        _print_header()
```

- [ ] **Step 5: Stage**

Run: `git add tests/amd/bench_hipblaslt_layouts.py`

---

## Task 5 — CSV parser + counter aggregation

**Files:**
- Modify: `tests/amd/bench_hipblaslt_layouts.py`
- Modify: `tests/amd/test_bench_hipblaslt_layouts.py`

- [ ] **Step 1: Write the parser test (with a fixture CSV string)**

Append to `tests/amd/test_bench_hipblaslt_layouts.py`:

```python
def test_parse_counter_csv_filters_to_hipblaslt_kernels(tmp_path):
    csv_path = tmp_path / "counter.csv"
    csv_path.write_text(
        "Dispatch_Id,Kernel_Name,Start_Timestamp,End_Timestamp,SQ_INSTS_MFMA,SQ_INSTS_VALU\n"
        "1,Cijk_Ailk_Bljk_HHS_BH_MT128x128,100,200,1000,5000\n"
        "2,Cijk_Ailk_Bljk_HHS_BH_MT128x128,300,450,1100,5100\n"
        "3,fill_kernel,500,600,0,0\n"
    )
    parsed = B.parse_counter_csv(csv_path, kernel_name_hint="Cijk")
    # Expect: 2 rows kept, averaged
    assert parsed["n_dispatches"] == 2
    assert parsed["counters"]["SQ_INSTS_MFMA"] == pytest.approx(1050.0)
    assert parsed["counters"]["SQ_INSTS_VALU"] == pytest.approx(5050.0)
    assert parsed["dispatch_time_us"] == pytest.approx(((200 - 100) + (450 - 300)) / 2 / 1000)
```

- [ ] **Step 2: Run the test — expect FAIL**

Run: `cd /workspace/quack && PYTHONPATH=. pytest tests/amd/test_bench_hipblaslt_layouts.py::test_parse_counter_csv_filters_to_hipblaslt_kernels -x 2>&1 | tail -10`
Expected: `AttributeError: module ... has no attribute 'parse_counter_csv'`.

- [ ] **Step 3: Implement the parser**

Append to `tests/amd/bench_hipblaslt_layouts.py`:

```python
def parse_counter_csv(csv_path: Path, kernel_name_hint: str = "Cijk") -> dict:
    """Aggregate rocprofv3 counter CSV rows matching `kernel_name_hint`.

    hipBLASLt fp16/bf16 kernel names typically start with ``Cijk_`` (Tensile).
    Kernels with other names (zero-fill, util) are filtered out.

    Returns {"n_dispatches": int, "dispatch_time_us": float (avg),
             "counters": {name: float (avg)}}.
    """
    rows = []
    counters_hdr: list[str] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        counters_hdr = [
            h for h in reader.fieldnames or []
            if h not in ("Dispatch_Id", "Kernel_Name", "Start_Timestamp", "End_Timestamp", "Correlation_Id", "Queue_Id", "GPU_Id")
        ]
        for row in reader:
            if kernel_name_hint not in row.get("Kernel_Name", ""):
                continue
            rows.append(row)

    if not rows:
        return {"n_dispatches": 0, "dispatch_time_us": 0.0, "counters": {}}

    def _float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    times_us = [
        (_float(r["End_Timestamp"]) - _float(r["Start_Timestamp"])) / 1000.0
        for r in rows
    ]
    counters = {
        c: sum(_float(r.get(c, 0)) for r in rows) / len(rows)
        for c in counters_hdr
    }
    return {
        "n_dispatches": len(rows),
        "dispatch_time_us": sum(times_us) / len(times_us),
        "counters": counters,
    }
```

- [ ] **Step 4: Run the test — expect PASS**

Run: `cd /workspace/quack && PYTHONPATH=. pytest tests/amd/test_bench_hipblaslt_layouts.py -x 2>&1 | tail -10`
Expected: all tests pass.

- [ ] **Step 5: Stage**

Run: `git add tests/amd/bench_hipblaslt_layouts.py tests/amd/test_bench_hipblaslt_layouts.py`

---

## Task 6 — Profile-mode entrypoint

**Files:**
- Modify: `tests/amd/bench_hipblaslt_layouts.py`

- [ ] **Step 1: Add a `--profile` flag that runs every shape under rocprofv3**

In `main()`, add the `--profile` and `--out-dir` flags:

```python
    parser.add_argument("--profile", action="store_true", help="re-launch self under rocprofv3 per shape")
    parser.add_argument("--out-dir", default="./rocprof-layout-profile", help="dir for rocprofv3 CSVs")
    parser.add_argument("--pmc-file", default="tests/amd/pmc_counters.txt")
```

- [ ] **Step 2: Branch on `--profile` in `main()`**

Replace the end of `main()` (the `for s in shapes: _bench_one...` block) with:

```python
    if args.profile:
        # We are the *driver*, launching each shape as a child under rocprofv3.
        out_root = Path(args.out_dir)
        pmc_file = Path(args.pmc_file)
        if not args.no_header:
            _print_profile_header()
        for s in shapes:
            csv_path = _run_under_rocprofv3(
                s, args.dtype, out_root / s["name"], pmc_file,
            )
            counters = parse_counter_csv(csv_path)
            r = BenchResult(
                name=s["name"], role=s["role"], layout=s["layout"],
                M=s["M"], N=s["N"], K=s["K"], dtype=args.dtype,
                seconds=counters["dispatch_time_us"] * 1e-6,
                tflops=matmul_flops(s["M"], s["N"], s["K"]) / (counters["dispatch_time_us"] * 1e-6) / 1e12,
            )
            _print_profile_row(r, counters)
    else:
        if not args.no_header:
            _print_header()
        for s in shapes:
            r = _bench_one(s, torch_dtype, args.dtype)
            _print_row(r)
    return 0
```

- [ ] **Step 3: Add the profile-mode printer functions**

Append to `tests/amd/bench_hipblaslt_layouts.py` (before `main()`):

```python
def _print_profile_header():
    # Short counter column names to keep the table readable.
    print(
        f"{'shape':<22}{'role':<14}{'lay':<4}{'dt':<4}"
        f"{'us':>8}{'TF/s':>7}"
        f"{'MFMA':>11}{'VALU':>11}{'VMEM':>11}{'LDS':>11}"
    )
    print("-" * 110)


def _print_profile_row(r: BenchResult, counters: dict):
    shape_s = f"{r.M}x{r.N}x{r.K}"
    c = counters.get("counters", {})
    def _fmt(name, default=0.0):
        v = c.get(name, default)
        if v >= 1e9:
            return f"{v/1e9:.2f}G"
        if v >= 1e6:
            return f"{v/1e6:.2f}M"
        if v >= 1e3:
            return f"{v/1e3:.1f}k"
        return f"{v:.1f}"
    print(
        f"{shape_s:<22}{r.role:<14}{r.layout:<4}{r.dtype:<4}"
        f"{r.seconds*1e6:>8.1f}{r.tflops:>7.2f}"
        f"{_fmt('SQ_INSTS_MFMA'):>11}{_fmt('SQ_INSTS_VALU'):>11}"
        f"{_fmt('SQ_INSTS_VMEM'):>11}{_fmt('SQ_INSTS_LDS'):>11}"
    )
```

- [ ] **Step 4: Smoke-test one shape under `--profile`**

Run:
```
cd /workspace/quack && PYTHONPATH=. python -m tests.amd.bench_hipblaslt_layouts \
    --profile --shape 2048x4096x1024 --layout NT --dtype bf16 2>&1 | tail -10
```
Expected: one row printed with real counter values in the Ms/Gs. If the CSV parse returns `n_dispatches == 0`, the kernel-name-hint is wrong — `ls /workspace/quack/rocprof-layout-profile/*/` and `head -1 <csv>` to find the real prefix (it's "Cijk_" on MI300/MI350 hipBLASLt but could be "ampere_" on NV or "hgemm_" for non-tensile kernels).

- [ ] **Step 5: Stage**

Run: `git add tests/amd/bench_hipblaslt_layouts.py`

---

## Task 7 — Full-matrix run + data capture

**Files:**
- Create: `/tmp/layout_profile_bf16.txt` (intermediate — not committed)
- Create: `/tmp/layout_profile_f16.txt` (intermediate — not committed)

- [ ] **Step 1: Run the full matrix under rocprofv3 for bf16**

Run:
```
cd /workspace/quack && PYTHONPATH=. python -m tests.amd.bench_hipblaslt_layouts \
    --profile --dtype bf16 2>&1 | tee /tmp/layout_profile_bf16.txt
```
Expected: ~12 rows (3 bs×hidden × 4 layouts). Runtime ~2-5 min (4 PMC passes × 12 shapes × ~5s/shape). If any row fails with "rocprofv3 failed", the failure is usually an OOM on the largest shape — skip it temporarily with `--shape` filters and note in the memo.

- [ ] **Step 2: Run the full matrix for f16**

Run:
```
cd /workspace/quack && PYTHONPATH=. python -m tests.amd.bench_hipblaslt_layouts \
    --profile --dtype f16 2>&1 | tee /tmp/layout_profile_f16.txt
```
Expected: same 12 rows in f16.

- [ ] **Step 3: Capture a non-profile bench run (quick, no subprocess overhead)**

Run:
```
cd /workspace/quack && PYTHONPATH=. python -m tests.amd.bench_hipblaslt_layouts \
    --dtype bf16 2>&1 | tee /tmp/layout_bench_bf16.txt
cd /workspace/quack && PYTHONPATH=. python -m tests.amd.bench_hipblaslt_layouts \
    --dtype f16 2>&1 | tee /tmp/layout_bench_f16.txt
```
Expected: cleaner per-shape TFLOP/s numbers without rocprofv3's PMC-multiplexing overhead. These are the numbers we use for the **Tier-2 perf target** in the memo.

- [ ] **Step 4: Sanity check the data**

Eyeball the captured files: TFLOP/s should be 50-600 range per shape (MI355X peak is ~2500 TFLOPs f16, hipBLASLt typically achieves 60-80% of peak on large shapes, less on small/skinny). MFMA counter should scale roughly with M×N×K/16³. If any shape shows TFLOP/s < 10 or MFMA counter = 0, something is wrong — investigate before writing the memo.

---

## Task 8 — Write the analysis memo

**Files:**
- Create: `docs/superpowers/specs/hipblaslt_layout_profile.md`

- [ ] **Step 1: Draft the memo structure**

Create `docs/superpowers/specs/hipblaslt_layout_profile.md`:

```markdown
# hipBLASLt per-layout profile — MI355X (gfx950)

**Date**: 2026-04-23.
**Hardware**: MI355X (gfx950 / CDNA4), wave64.
**Software**: ROCm <check via `/opt/rocm/bin/rocminfo | head -3`>, rocprofv3 1.0.0.
**Purpose**: Set per-shape Tier-2 perf targets for the NN and TN kernels (spec §3.1).
**Raw data**: `/tmp/layout_bench_{bf16,f16}.txt` (non-profile) and `/tmp/layout_profile_{bf16,f16}.txt` (counter).
**Bench harness**: `tests/amd/bench_hipblaslt_layouts.py`.

## bf16 results

| shape            | role          | layout | kernel μs | TFLOP/s | MFMA    | VALU    | VMEM    | LDS     |
|------------------|---------------|:------:|----------:|--------:|--------:|--------:|--------:|--------:|
| <fill from /tmp/layout_profile_bf16.txt>

## f16 results

<same table from /tmp/layout_profile_f16.txt>

## Analysis

### Per-layout kernel-time comparison

For each (bs, hidden) triple, the three layouts have roughly the same total FLOPs (all are 2·bs·hidden·4·hidden up to permutation of M/N/K). A wall-time spread across layouts tells us how well hipBLASLt handles each. Record the NT/NN/TN ratio per (bs, hidden).

Expected observation: NN and NT are close (hipBLASLt has polished both); TN-small-M is worst-off because it's the layout where hipBLASLt picks a less-specialised kernel variant. If the spread is > 1.5× on any shape, that shape is where our NN/TN kernels have the most headroom.

### Counter-ratio signals (per layout)

- **MFMA / VALU ratio**: high (>100) = compute-bound MFMA; low (<30) = epilogue-heavy or scalar-unit-blocked.
- **VMEM / MFMA ratio**: proxy for memory pressure. NT with preshuffle should be lowest; NN and TN should be higher (more buffer_loads per MFMA because of LDS staging).
- **LDS / MFMA ratio**: LDS read-write pressure. Double-pipelined kernels score around 1.0-2.0; if it spikes above that on one layout, that layout spends more time in LDS than ideal.

Record the ratios per layout in a short subsection with 2-3 sentences of interpretation.

### Perf targets for the NN/TN kernels

Tier-2 target (our-throughput ÷ hipBLASLt-throughput ≥ 0.9) translates to per-shape kernel-time budgets:

| shape | layout | hipBLASLt kernel μs | Tier-2 budget (μs) |
|-------|:------:|--------------------:|-------------------:|
| <fill: target = hipBLASLt_μs × (1 / 0.9) = 1.11× hipBLASLt_μs>

These are the numbers Phase 3 (NN) and Phase 4 (TN) must beat on the tune-in step.

### Kernel-name observations

The `Kernel_Name` column in rocprofv3 CSVs exposes hipBLASLt's internal kernel selection. Record the top-1 kernel name per layout (all should start with `Cijk_` for Tensile kernels):

- NT bs=8192 h=4096: <fill>
- NN bs=8192 h=4096: <fill>
- TN bs=8192 h=4096: <fill>

Different name per layout ⇒ hipBLASLt has specialised kernels per layout; same name ⇒ one kernel handles multiple layouts via runtime dispatch (less likely on Tensile). This informs whether we can expect similar perf characteristics across our three kernels or need independent tuning per layout.

### Recommendations

1. <fill — 2-3 bullet points on what the counters suggest about NN kernel design, e.g. "NN layout shows 2× VMEM / MFMA vs NT — expect our NN kernel to benefit from LDS-staging more than preshuffle">
2. <fill — similar for TN>
3. <fill — any shapes where hipBLASLt is < 30% of peak, which are our best Tier-3 (winning) opportunities>
```

- [ ] **Step 2: Fill the table rows from `/tmp/layout_*` captures**

For the bf16 results section: copy the rows from `/tmp/layout_profile_bf16.txt`, reformatting into markdown table syntax. Each row becomes:

```
| 2048x4096x1024   | MLP-fwd       |  NT    |      <us>|   <TF/s>|  <MFMA>|  <VALU>|  <VMEM>|   <LDS>|
```

Same for f16. Keep the `us` and `TFLOP/s` from the `/tmp/layout_profile_*.txt` (profile-mode counters) — those are the per-dispatch averages.

- [ ] **Step 3: Fill the analysis subsections with concrete numbers**

For each subsection:
- Per-layout comparison: compute NT/NN ratio and NT/TN ratio per (bs, hidden), pick the shape with largest spread, highlight it.
- Counter ratios: compute MFMA/VALU and VMEM/MFMA per row, pick extremes, write 2-3 sentences.
- Tier-2 budgets: multiply each hipBLASLt μs by 1.11 (= 1/0.9) to get our target.
- Kernel names: read the `Kernel_Name` column from the rocprofv3 CSVs (under `rocprof-layout-profile/<shape_name>/`).
- Recommendations: judgement call based on the numbers.

- [ ] **Step 4: Stage**

Run: `git add docs/superpowers/specs/hipblaslt_layout_profile.md`

---

## Task 9 — Final commit

- [ ] **Step 1: Sanity-check what's staged**

Run: `git status --short && git diff --cached --stat`
Expected:
```
A  docs/superpowers/specs/hipblaslt_layout_profile.md
A  tests/amd/bench_hipblaslt_layouts.py
A  tests/amd/pmc_counters.txt
A  tests/amd/test_bench_hipblaslt_layouts.py
```

- [ ] **Step 2: Run the unit tests one last time**

Run: `cd /workspace/quack && PYTHONPATH=. pytest tests/amd/test_bench_hipblaslt_layouts.py -x -v 2>&1 | tail -15`
Expected: all tests pass.

- [ ] **Step 3: Commit**

Run:
```
cd /workspace/quack && git commit -m "$(cat <<'EOF'
[AMD] Phase 1 — hipBLASLt NT/NN/TN layout profile on MI355X

Bench harness + rocprofv3 HW counter collection + per-shape Tier-2
perf targets for the upcoming NN and TN GEMM kernels. Covers 12
shapes (3 bs×hidden × 4 MLP roles) in both bf16 and f16. Sets
the measured baseline that Phase 3 (NN) and Phase 4 (TN) tune-in
steps must beat.

See docs/superpowers/specs/2026-04-23-nn-tn-kernels-design.md §3.1
for scope and docs/superpowers/specs/hipblaslt_layout_profile.md
for the per-shape data and analysis.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Verify the commit**

Run: `git log --oneline -5`
Expected: the new commit is on top of `aeb8f1a` (the spec commit).

---

## Self-review checklist (executor does this before marking Phase 1 done)

- [ ] All 9 tasks completed, every checkbox ticked.
- [ ] `pytest tests/amd/test_bench_hipblaslt_layouts.py` all green.
- [ ] `python -m tests.amd.bench_hipblaslt_layouts` runs cleanly (no `--profile`).
- [ ] `python -m tests.amd.bench_hipblaslt_layouts --profile --dtype bf16` completes without error.
- [ ] `docs/superpowers/specs/hipblaslt_layout_profile.md` has no `<fill>` placeholders.
- [ ] Every row in the memo's two data tables has real numbers (no `N/A` or `—`).
- [ ] The Tier-2 budget column is filled for every row.

---

## What this commit does NOT do

- No kernel code. The tiled-copy refactor (Phase 2) lands next.
- No NN or TN kernel. Those are Phases 3 and 4.
- No autograd wiring. That's Phase 5.
- The memo's "Recommendations" section is informational only — design decisions for NN/TN still live in the parent spec and will be updated via a follow-up commit if the data contradicts them.
