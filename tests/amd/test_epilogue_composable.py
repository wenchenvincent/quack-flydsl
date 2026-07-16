# Copyright (c) 2026, AMD.

"""Composable epilogue framework (G4): gemm_gfx950's epilogue migrated onto
quack.amd.epi_ops. Regression bar = every (bias, activation, alpha, beta*C)
combination matches the torch reference through the MFMA kernel path."""

import itertools
import pytest
import torch

from quack.amd.gemm import gemm


def _act(x, a):
    if a is None:
        return x
    if a == "relu":
        return torch.relu(x)
    if a == "relu_sq":
        return torch.relu(x) * x
    if a == "gelu_tanh_approx":
        return torch.nn.functional.gelu(x, approximate="tanh")
    if a == "silu":
        return torch.nn.functional.silu(x)
    raise ValueError(a)


@pytest.mark.parametrize(
    "has_bias,activation,has_alpha,has_c",
    list(itertools.product(
        [False, True],
        [None, "relu", "gelu_tanh_approx", "silu"],
        [False, True],
        [False, True],
    )),
)
def test_epilogue_combo(has_bias, activation, has_alpha, has_c):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, K, N = 16, 32, 16  # small aligned shape -> MFMA epilogue kernel
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    alpha = 0.7 if has_alpha else 1.0
    beta = 1.3 if has_c else 0.0
    bias = torch.randn(N, device="cuda", dtype=torch.float32) if has_bias else None
    Cres = torch.randn(M, N, device="cuda", dtype=torch.float32) if has_c else None
    out = gemm(A, B, bias=bias, activation=activation, alpha=alpha,
               beta=beta, C=Cres, out_dtype=torch.float32)
    ref = alpha * (A.float() @ B.float())
    if has_c:
        ref = ref + beta * Cres
    if has_bias:
        ref = ref + bias
    ref = _act(ref, activation)
    atol = max(1e-2, K * 3e-5)
    torch.testing.assert_close(out, ref, atol=atol, rtol=1e-2)


def test_epiops_unit():
    """EpiOp descriptors build the expected op sequence from flags."""
    from quack.amd.epi_ops import build_epilogue, AlphaScale, BetaResidual, RowBias, Activation
    ops = build_epilogue(has_alpha=True, has_c=True, has_bias=True, activation="relu")
    assert [type(o) for o in ops] == [AlphaScale, BetaResidual, RowBias, Activation]
    assert build_epilogue() == ()
    assert [type(o) for o in build_epilogue(has_bias=True)] == [RowBias]
