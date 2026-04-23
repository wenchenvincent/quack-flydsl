# Copyright (c) 2026, AMD.

"""End-to-end MLP training-step bench.

Measures ``linear_train`` on a 2-layer MLP (linear → relu → linear) with
forward + backward, compared to torch baselines.

The point: with our own kernels wired through autograd (fwd → gemm_splitk,
DX → gemm_nn, DW → gemm_tn), a full training step now completes entirely
inside the quack.amd stack. Three layout families × two directions
contribute to per-step time; the ratios below tell us whether kernel
perf is dominating real training or being swamped by Python / autograd
overhead.

Usage:
    PYTHONPATH=/workspace/quack python -m tests.amd.bench_mlp_train
"""

import time

import torch

from quack.amd.linear import linear_train
from quack.amd.mlp import mlp_train


_SHAPES = [
    # (batch_seq, hidden, out) — roughly MLP-FFN sizes
    ("small",  2048, 1024, 4096),
    ("medium", 8192, 4096, 16384),
]


def _bench(step_fn, warmup=10, iters=30):
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()
    time.sleep(0.1)
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for s, e in events:
        s.record()
        step_fn()
        e.record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1e-3 for s, e in events)
    return times[0]  # min — best-of-N


def _mlp_step_ours_plain(x, W1, W2):
    """Phase 5: two LinearFunc + explicit torch.relu."""
    h = torch.relu(linear_train(x, W1))
    y = linear_train(h, W2)
    y.sum().backward()
    x.grad = None
    W1.grad = None
    W2.grad = None


def _mlp_step_ours_fused(x, W1, W2):
    """Phase 6: mlp_train (LinearActFunc + LinearFunc, saves one kernel boundary)."""
    y = mlp_train(x, W1, W2, activation="relu")
    y.sum().backward()
    x.grad = None
    W1.grad = None
    W2.grad = None


def _mlp_step_torch(x, W1, W2):
    h = torch.relu(torch.nn.functional.linear(x, W1))
    y = torch.nn.functional.linear(h, W2)
    y.sum().backward()
    x.grad = None
    W1.grad = None
    W2.grad = None


def main():
    header = (
        f"{'shape':<14}{'dtype':<6}{'plain ms':>10}{'fused ms':>10}"
        f"{'torch ms':>10}{'fused/torch':>13}"
    )
    print(header)
    print("-" * len(header))
    for dtype, dtype_str in [(torch.bfloat16, "bf16"), (torch.float16, "f16")]:
        for name, bs, hidden, out_f in _SHAPES:
            torch.manual_seed(0)
            x = (torch.randn(bs, hidden, device="cuda", dtype=dtype) * 0.1).detach()
            x.requires_grad_(True)
            W1 = (torch.randn(out_f, hidden, device="cuda", dtype=dtype) * 0.1).detach()
            W1.requires_grad_(True)
            W2 = (torch.randn(hidden, out_f, device="cuda", dtype=dtype) * 0.1).detach()
            W2.requires_grad_(True)

            xt = x.detach().clone().requires_grad_(True)
            W1t = W1.detach().clone().requires_grad_(True)
            W2t = W2.detach().clone().requires_grad_(True)

            t_plain = _bench(lambda: _mlp_step_ours_plain(x, W1, W2))
            t_fused = _bench(lambda: _mlp_step_ours_fused(x, W1, W2))
            t_torch = _bench(lambda: _mlp_step_torch(xt, W1t, W2t))
            ratio = t_torch / t_fused
            print(f"{name:<14}{dtype_str:<6}"
                  f"{t_plain*1e3:>10.3f}{t_fused*1e3:>10.3f}"
                  f"{t_torch*1e3:>10.3f}{ratio:>12.3f}x")


if __name__ == "__main__":
    main()
