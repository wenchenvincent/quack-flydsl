# Copyright (c) 2026, AMD.

"""Bench hipBLASLt across the three GEMM layouts training needs.

Usage (direct run):
    PYTHONPATH=/workspace/quack python -m tests.amd.bench_hipblaslt_layouts

Usage (under rocprofv3):
    PYTHONPATH=/workspace/quack rocprofv3 -i tests/amd/pmc_counters.txt \\
        -- python -m tests.amd.bench_hipblaslt_layouts --shape 4096x4096x4096 --layout NT

The per-shape/per-layout table lands in stdout; the rocprofv3 counter
CSVs land in ``./rocprof-layout-profile/``. See the parent spec at
docs/superpowers/specs/2026-04-23-nn-tn-kernels-design.md §3.1.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Tuple

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


def _filter_shapes(shape_filter, layout_filter) -> list[dict]:
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


def parse_counter_csvs(csv_paths: Iterable[Path], kernel_name_hint: str = "Cijk") -> dict:
    """Aggregate rocprofv3 v1 counter CSVs (long-format) across all PMC passes.

    rocprofv3 emits one CSV per ``pmc:`` line in the input file; each CSV has
    one row per (dispatch, counter) — not one row per dispatch. This parser:

    - filters rows by ``kernel_name_hint`` substring (Cijk_* for hipBLASLt).
    - within each CSV, averages each counter's value across matching dispatches.
    - merges the per-pass averages into a single ``{counter: avg_value}`` map.
    - computes the average per-dispatch wall-time (End - Start) across all
      matching rows in any pass.

    Returns ``{"n_dispatches": int, "dispatch_time_us": float, "counters": dict}``
    where ``n_dispatches`` is the max count seen in any single pass (sanity
    check that all passes saw the same binary's workload).
    """
    per_pass_counters: dict[str, list[float]] = defaultdict(list)
    dispatch_time_ns: list[float] = []
    dispatches_seen_per_pass: list[set] = []

    for csv_path in csv_paths:
        seen_dispatches = set()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            # Pass-local per-counter accumulation (one entry per matching dispatch).
            pass_counters: dict[str, list[float]] = defaultdict(list)
            for row in reader:
                if kernel_name_hint not in row.get("Kernel_Name", ""):
                    continue
                try:
                    value = float(row["Counter_Value"])
                    start = float(row["Start_Timestamp"])
                    end = float(row["End_Timestamp"])
                except (TypeError, ValueError, KeyError):
                    continue
                counter = row["Counter_Name"]
                pass_counters[counter].append(value)
                dispatch_id = row.get("Dispatch_Id", "")
                if dispatch_id not in seen_dispatches:
                    dispatch_time_ns.append(end - start)
                    seen_dispatches.add(dispatch_id)
        dispatches_seen_per_pass.append(seen_dispatches)
        # Average this pass's counter samples (one sample per dispatch).
        for counter, values in pass_counters.items():
            if values:
                per_pass_counters[counter].append(sum(values) / len(values))

    if not dispatch_time_ns:
        return {"n_dispatches": 0, "dispatch_time_us": 0.0, "counters": {}}

    counters = {c: sum(vals) / len(vals) for c, vals in per_pass_counters.items()}
    # Sanity: n_dispatches is the max-matching-dispatch count across passes.
    n_dispatches = max((len(s) for s in dispatches_seen_per_pass), default=0)
    return {
        "n_dispatches": n_dispatches,
        "dispatch_time_us": (sum(dispatch_time_ns) / len(dispatch_time_ns)) / 1000.0,
        "counters": counters,
    }


def _run_under_rocprofv3(
    shape: dict, dtype_str: str, out_dir: Path, pmc_file: Path,
) -> list[Path]:
    """Launch *ourselves* under rocprofv3 for one shape × layout.

    Returns paths to the counter-collection CSVs (one per PMC pass). ``out_dir``
    is removed and recreated per call so PIDs don't collide across shapes.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shape_tag = f"{shape['M']}x{shape['N']}x{shape['K']}"
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":/workspace/quack"
    cmd = [
        "rocprofv3", "-i", str(pmc_file),
        "-d", str(out_dir), "-f", "csv",
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
    csvs = sorted(out_dir.rglob("*counter_collection.csv"))
    if not csvs:
        raise RuntimeError(
            f"no counter CSV under {out_dir}; files: "
            f"{[p.name for p in out_dir.rglob('*')]}"
        )
    return csvs


def _print_profile_header():
    # Short counter column names to keep the table readable.
    print(
        f"{'shape':<22}{'role':<14}{'lay':<4}{'dt':<4}"
        f"{'us':>8}{'TF/s':>7}"
        f"{'MFMA':>11}{'VALU':>11}{'VMEM':>11}{'LDS':>11}{'waves':>10}"
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
        f"{_fmt('SQ_WAVES'):>10}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default=None, help="MxNxK (e.g. 4096x4096x4096)")
    parser.add_argument("--layout", default=None, choices=["NT", "NN", "TN"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "f16"])
    parser.add_argument("--no-header", action="store_true", help="suppress the table header (used for sub-runs under rocprofv3)")
    parser.add_argument("--profile", action="store_true", help="re-launch self under rocprofv3 per shape")
    parser.add_argument("--out-dir", default="./rocprof-layout-profile", help="dir for rocprofv3 CSVs")
    parser.add_argument("--pmc-file", default="tests/amd/pmc_counters.txt")
    args = parser.parse_args()

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shapes = _filter_shapes(args.shape, args.layout)
    if not shapes:
        print(f"No shapes match filter shape={args.shape} layout={args.layout}")
        return 1

    if args.profile:
        out_root = Path(args.out_dir)
        pmc_file = Path(args.pmc_file)
        if not args.no_header:
            _print_profile_header()
        for s in shapes:
            try:
                csv_paths = _run_under_rocprofv3(
                    s, args.dtype, out_root / s["name"], pmc_file,
                )
                counters = parse_counter_csvs(csv_paths)
                if counters["n_dispatches"] == 0:
                    print(f"{s['name']}: no Cijk dispatches found in profile output")
                    continue
                t = counters["dispatch_time_us"] * 1e-6
                r = BenchResult(
                    name=s["name"], role=s["role"], layout=s["layout"],
                    M=s["M"], N=s["N"], K=s["K"], dtype=args.dtype,
                    seconds=t,
                    tflops=matmul_flops(s["M"], s["N"], s["K"]) / t / 1e12,
                )
                _print_profile_row(r, counters)
            except Exception as e:
                print(f"{s['name']}: profile FAILED: {e}")
    else:
        if not args.no_header:
            _print_header()
        for s in shapes:
            r = _bench_one(s, torch_dtype, args.dtype)
            _print_row(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
