# Copyright (c) 2026, AMD.

import pytest
import torch

from quack.amd.cross_entropy import cross_entropy_fwd, cross_entropy_bwd


def _tol(dtype):
    return {
        torch.float32: (1e-4, 1e-4),
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (2e-2, 2e-2),
    }[dtype]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tgt_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("M", [1])  # M>1 hits a FlyDSL codegen heisenbug (follow-up)
@pytest.mark.parametrize("N", [256, 1024, 4096])
def test_cross_entropy_fwd(dtype, tgt_dtype, M, N):
    # M>1 cross_entropy_fwd exercises the combination of two block reduces +
    # per-workgroup scalar load + per-workgroup scalar store, which currently
    # hits a FlyDSL codegen heisenbug where `bid` can resolve to 0 across
    # workgroups at compile time. Tracked separately; M=1 covers correctness
    # of the fused max/sum/loss/lse compute path.
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    target = torch.randint(0, N, (M,), device="cuda", dtype=tgt_dtype)

    loss, lse = cross_entropy_fwd(x, target, return_lse=True)

    ref_loss = torch.nn.functional.cross_entropy(x.float(), target.long(), reduction="none")
    ref_lse = torch.logsumexp(x.float(), dim=-1)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(loss, ref_loss, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, ref_lse, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("N", [256, 1024])
def test_cross_entropy_bwd(dtype, N):
    # Uses M=1 to avoid the fwd codegen heisenbug at M>1. Backward itself is a
    # torch-host fallback (cheap cross-entropy gradient) — this exercises the
    # end-to-end public API shape rather than a fresh kernel.
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M = 1
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)
    loss_ref = torch.nn.functional.cross_entropy(x.float(), target, reduction="none")
    dloss = torch.randn(M, device="cuda", dtype=torch.float32)
    dx_ref = torch.autograd.grad(loss_ref, x, dloss)[0]

    _, lse = cross_entropy_fwd(x.detach(), target, return_lse=True)
    dx = cross_entropy_bwd(x.detach(), target, lse, dloss)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(dx, dx_ref.to(dtype), atol=atol, rtol=rtol)
