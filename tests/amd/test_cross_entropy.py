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
@pytest.mark.parametrize("M", [128, 4, 1])  # largest M first — see conftest.py
@pytest.mark.parametrize("N", [256, 1024, 4096])
def test_cross_entropy_fwd(dtype, tgt_dtype, M, N):
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
@pytest.mark.parametrize("M", [128, 4, 1])  # largest M first — see conftest.py
@pytest.mark.parametrize("N", [256, 1024, 4096])
def test_cross_entropy_bwd(dtype, M, N):
    # Real FlyDSL dx kernel now (replaces the torch-host fallback). The fwd
    # kernel produces `lse`, and the bwd kernel derives
    # `dx[m,j] = (exp(x[m,j] - lse[m]) - (j==target[m])) * dloss[m]`.
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)
    loss_ref = torch.nn.functional.cross_entropy(x.float(), target, reduction="none")
    dloss = torch.randn(M, device="cuda", dtype=torch.float32)
    dx_ref = torch.autograd.grad(loss_ref, x, dloss)[0]

    _, lse = cross_entropy_fwd(x.detach(), target, return_lse=True)
    dx = cross_entropy_bwd(x.detach(), target, lse, dloss)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(dx, dx_ref.to(dtype), atol=atol, rtol=rtol)
