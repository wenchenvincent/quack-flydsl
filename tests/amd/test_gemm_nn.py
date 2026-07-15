# Copyright (c) 2026, AMD.

"""Correctness tests for ``quack.amd.gemm_gfx950_nn.gemm_nn``.

Matches torch.matmul (the hipBLASLt reference) up to f16/bf16 rounding.
Shape parametrize is M-descending to work around the FlyDSL JIT grid-bake
quirk documented in ``tests/amd/conftest.py``.
"""

import pytest
import torch

from quack.amd.gemm_gfx950_nn import gemm_nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")


@pytest.mark.parametrize("M", [1024, 256, 128])  # M-descending (FlyDSL cache bake-in)
@pytest.mark.parametrize(
    "N,K",
    [
        (256, 64),      # smallest — BLOCK_M=128, BLOCK_N=256, BLOCK_K=64
        (512, 128),
        (512, 256),
        (1024, 256),
        (1024, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_nn_matches_torch(M, N, K, dtype):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    B = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
    out = gemm_nn(A, B)
    ref = torch.matmul(A.float(), B.float()).to(dtype)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 0.05, f"NN {dtype} {M}x{N}x{K}: max_err {err:.4f} > 0.05"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_nn_zero_inputs(dtype):
    """Zero A or zero B → zero output (sanity: no NaN from LDS uninit)."""
    M, N, K = 128, 256, 64
    A = torch.zeros(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
    out = gemm_nn(A, B)
    assert (out.abs() < 1e-3).all(), f"zero-A produced nonzero: max={out.abs().max()}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_nn_matches_torch_mid_shape(dtype):
    """Mid-range shape from the MLP-dx profile matrix (bs=2048, h=1024)."""
    torch.manual_seed(0)
    M, K, N = 2048, 4096, 1024
    A = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    B = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
    out = gemm_nn(A, B)
    ref = torch.matmul(A.float(), B.float()).to(dtype)
    err = (out.float() - ref.float()).abs().max().item()
    # Looser tolerance for larger K accumulation.
    tol = 0.2 if dtype is torch.float16 else 0.2
    assert err < tol, f"{dtype} 2048x1024x4096: max_err {err:.4f} > {tol}"


@pytest.mark.parametrize("MKN", [4096])
def test_gemm_nn_large_shape_nn_big(MKN):
    """Large shape routes gemm_nn -> gemm_gfx950_nn_big (tile-swizzle path).

    Regression guard for the class-1 fx.const_expr drift that crashed
    nn_big at large shapes (NameError: 'pid'/'e0' not defined) — nn_big
    had no test, so the drift slipped the 2026-07-14 sweep.
    """
    torch.manual_seed(0)
    a = torch.randn(MKN, MKN, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(MKN, MKN, device="cuda", dtype=torch.bfloat16)
    c = gemm_nn(a, b)
    ref = (a.float() @ b.float())
    # bf16 accumulation over K=4096 — loose but real.
    assert torch.allclose(c.float(), ref, atol=2.0, rtol=2e-2)
