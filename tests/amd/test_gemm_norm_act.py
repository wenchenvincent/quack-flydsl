# Copyright (c) 2026, AMD.

"""Tests for the fused gemm_norm_act surface."""

import pytest
import torch

from quack.amd.gemm import gemm_norm_act


@pytest.mark.parametrize("activation", [None, "relu", "silu"])
@pytest.mark.parametrize("use_colvec", [False, True])
@pytest.mark.parametrize("use_rowvec", [False, True])
def test_gemm_norm_act(activation, use_colvec, use_rowvec):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, K, N = 64, 64, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    colvec = torch.randn(M, device="cuda", dtype=torch.float32) if use_colvec else None
    rowvec = torch.randn(N, device="cuda", dtype=torch.float32) if use_rowvec else None

    out = gemm_norm_act(A, B, colvec=colvec, rowvec=rowvec, activation=activation)

    ref = (A.float() @ B.float())
    if colvec is not None:
        ref = ref * colvec.unsqueeze(-1)
    if rowvec is not None:
        ref = ref * rowvec.unsqueeze(-2)
    if activation == "relu":
        ref = torch.relu(ref)
    elif activation == "silu":
        ref = torch.nn.functional.silu(ref)
    ref = ref.to(torch.float16)

    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


def test_gemm_norm_act_with_bias_and_C():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, K, N = 64, 64, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    bias = torch.randn(N, device="cuda", dtype=torch.float16)
    C = torch.randn(M, N, device="cuda", dtype=torch.float32)
    colvec = torch.randn(M, device="cuda", dtype=torch.float32)
    rowvec = torch.randn(N, device="cuda", dtype=torch.float32)

    out = gemm_norm_act(
        A, B, colvec=colvec, rowvec=rowvec, bias=bias,
        C=C, alpha=0.5, beta=1.5, activation="relu",
    )
    ref = 0.5 * (A.float() @ B.float()) + 1.5 * C + bias.float()
    ref = ref * colvec.unsqueeze(-1) * rowvec.unsqueeze(-2)
    ref = torch.relu(ref).to(torch.float16)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
