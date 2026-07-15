# Copyright (c) 2026, AMD.

"""Gated MFMA GEMM: fused gate_fn(A @ B_gate) * (A @ B_up) for swiglu/reglu/geglu/glu."""

import pytest
import torch

from quack.amd.gemm_gated import gemm_gated


def _ref_gated(A, B_gate, B_up, gate_type):
    ga = A.float() @ B_gate.float()
    up = A.float() @ B_up.float()
    if gate_type == "swiglu":
        return torch.nn.functional.silu(ga) * up
    if gate_type == "reglu":
        return torch.relu(ga) * up
    if gate_type == "geglu":
        return torch.nn.functional.gelu(ga, approximate="tanh") * up
    if gate_type == "glu":
        return torch.sigmoid(ga) * up
    if gate_type == "swiglu_oai":
        half = 0.5 * ga
        silu_oai = half * torch.tanh(1.702 * half) + half
        return silu_oai * (up + 1.0)
    raise ValueError(gate_type)


@pytest.mark.parametrize("gate", ["swiglu", "reglu", "geglu", "glu", "swiglu_oai"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [128, 16])  # largest M first
@pytest.mark.parametrize("H", [32, 64])   # hidden size = N/2
@pytest.mark.parametrize("K", [64])
def test_gemm_gated(gate, dtype, M, H, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    N = 2 * H
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B_gate = torch.randn(K, H, device="cuda", dtype=dtype)
    B_up = torch.randn(K, H, device="cuda", dtype=dtype)
    B = torch.cat([B_gate, B_up], dim=1)
    C = gemm_gated(A, B, gate_type=gate, out_dtype=torch.float32)
    ref = _ref_gated(A, B_gate, B_up, gate)
    atol = max(1e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=1e-3)


def test_gemm_gated_rejects_odd_N():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    A = torch.randn(16, 16, device="cuda", dtype=torch.float16)
    B = torch.randn(16, 48, device="cuda", dtype=torch.float16)  # N=48, N%32 != 0
    with pytest.raises(AssertionError):
        gemm_gated(A, B)
