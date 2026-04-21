# Copyright (c) 2026, AMD.

"""Varlen cu_seqlens_m + gather-A A_idx tests for the AMD GEMM surface."""

import pytest
import torch

from quack.amd.gemm import gemm
from quack.amd.varlen_utils import validate_varlen, seqlens_from_cu


def test_varlen_args_validate():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    cu = torch.tensor([0, 16, 48, 64], device="cuda", dtype=torch.int32)
    B = validate_varlen(cu, 64)
    assert B == 3
    lens = seqlens_from_cu(cu).tolist()
    assert lens == [16, 32, 16]


def test_gemm_with_cu_seqlens_m():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    total_M, K, N = 64, 64, 128
    A = torch.randn(total_M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    cu = torch.tensor([0, 16, 48, 64], device="cuda", dtype=torch.int32)
    out = gemm(A, B, cu_seqlens_m=cu)
    ref = (A.float() @ B.float()).to(torch.float16)
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


def test_gemm_with_per_sample_bias():
    """Varlen + (B, N) per-sample bias — each sample's rows add a
    different bias vector."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    total_M, K, N = 64, 64, 128
    A = torch.randn(total_M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    cu = torch.tensor([0, 16, 48, 64], device="cuda", dtype=torch.int32)
    # 3 samples, 3 different bias vectors
    bias = torch.randn(3, N, device="cuda", dtype=torch.float16)
    out = gemm(A, B, bias=bias, cu_seqlens_m=cu, activation="silu")
    # Reference: expand bias per row, apply
    sample_idx = torch.tensor([0]*16 + [1]*32 + [2]*16, device="cuda")
    bias_expanded = bias[sample_idx]
    ref = torch.nn.functional.silu((A.float() @ B.float()) + bias_expanded.float())
    ref = ref.to(torch.float16)
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


def test_gemm_with_A_idx():
    """Gather-A: A[A_idx] @ B."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, K, N = 128, 64, 64
    gather_M = 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    A_idx = torch.randint(0, M, (gather_M,), device="cuda", dtype=torch.int32)
    out = gemm(A, B, A_idx=A_idx)
    ref = (A[A_idx.long()].float() @ B.float()).to(torch.float16)
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)
