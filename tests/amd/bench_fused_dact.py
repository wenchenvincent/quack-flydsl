"""Micro-bench: fused gemm_dact MLP-backward vs torch.mm + torch act-bwd.

Run manually: python tests/amd/bench_fused_dact.py
Determines whether the fused splitk-dact backward path is actually faster
than the torch.mm + elementwise act-bwd path at the reference MLP shape,
to reconcile the gate (`_fused_dact_eligible`) with the mlp_func_train docstring.
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
    print("shape M,hidden,out_dim =", (M, hidden, out_dim), "act =", act)
    print("eligible:", _fused_dact_eligible(dout, w2, preact, act))

    from quack.amd.gemm_gfx950_splitk import gemm_splitk

    w2_T = w2.t().contiguous()

    def fused():
        return gemm_splitk(dout, w2_T, preact=preact, dact_activation=act)

    def unfused():
        gp = torch.mm(dout, w2)
        return _act_bwd(preact, gp, act)

    a = fused().float()
    b = unfused().float()
    err = (a - b).abs().max().item()
    print(f"max_err fused vs unfused: {err:.4f}")

    tf = _bench(fused)
    tu = _bench(unfused)
    print(f"fused:   {tf:.3f} ms")
    print(f"unfused: {tu:.3f} ms")
    print(f"winner: {'FUSED' if tf < tu else 'UNFUSED'} ({min(tf, tu) / max(tf, tu):.2f}x)")


if __name__ == "__main__":
    main()
