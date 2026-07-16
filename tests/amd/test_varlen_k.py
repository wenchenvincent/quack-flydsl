# Copyright (c) 2026, AMD.

"""Varlen-K grouped GEMM (G10): C[i] = A[:, s_i:e_i] @ B[s_i:e_i, :] per group.

Reference is the literal per-group torch.mm loop (matching NVIDIA's
gemm_interface varlen_k path), computed inline so the test is self-contained.
"""

import pytest
import torch

from quack.amd.gemm import gemm
from quack.amd.varlen_utils import validate_varlen_k


def _cu(lens, device):
    return torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=device, dtype=torch.int32)


def _ref(A, B, cu, alpha=1.0, beta=0.0, C=None):
    L = cu.numel() - 1
    M, N = A.shape[0], B.shape[1]
    out = torch.empty(L, M, N, dtype=torch.float32, device=A.device)
    cul = cu.tolist()
    for i in range(L):
        s, e = cul[i], cul[i + 1]
        r = alpha * (A[:, s:e].float() @ B[s:e, :].float())
        if C is not None:
            r = r + beta * C[i].float()
        out[i] = r
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_varlen_k_basic(dtype):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(0)
    M, N = 64, 48
    lens = [64, 128, 32]  # mixed: aligned (fast path) and small groups
    total_K = sum(lens)
    A = torch.randn(M, total_K, device="cuda", dtype=dtype)
    B = torch.randn(total_K, N, device="cuda", dtype=dtype)
    cu = _cu(lens, A.device)
    out = gemm(A, B, cu_seqlens_k=cu)
    ref = _ref(A, B, cu)
    assert out.shape == (len(lens), M, N)
    atol = max(1e-2, max(lens) * 3e-5)
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=1e-2)


def test_varlen_k_ragged_alignment():
    """Mix 16-aligned (FlyDSL fast path) and non-aligned (torch fallback) K_i."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(1)
    M, N = 32, 32
    lens = [48, 17, 30, 16]  # 48/16 aligned; 17,30 ragged
    total_K = sum(lens)
    A = torch.randn(M, total_K, device="cuda", dtype=torch.float16)
    B = torch.randn(total_K, N, device="cuda", dtype=torch.float16)
    cu = _cu(lens, A.device)
    out = gemm(A, B, cu_seqlens_k=cu)
    ref = _ref(A, B, cu)
    torch.testing.assert_close(out.float(), ref, atol=1e-1, rtol=1e-2)


def test_varlen_k_zero_length_group():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(2)
    M, N = 32, 32
    lens = [32, 0, 48]  # middle group is empty -> zero output
    total_K = sum(lens)
    A = torch.randn(M, total_K, device="cuda", dtype=torch.float16)
    B = torch.randn(total_K, N, device="cuda", dtype=torch.float16)
    cu = _cu(lens, A.device)
    out = gemm(A, B, cu_seqlens_k=cu)
    assert torch.count_nonzero(out[1]) == 0, "zero-length group must be all-zero"
    ref = _ref(A, B, cu)
    torch.testing.assert_close(out.float(), ref, atol=1e-1, rtol=1e-2)


def test_varlen_k_alpha_beta_c():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(3)
    M, N = 32, 48
    lens = [64, 32]
    total_K = sum(lens)
    A = torch.randn(M, total_K, device="cuda", dtype=torch.float16)
    B = torch.randn(total_K, N, device="cuda", dtype=torch.float16)
    Cres = torch.randn(len(lens), M, N, device="cuda", dtype=torch.float32)
    cu = _cu(lens, A.device)
    out = gemm(A, B, cu_seqlens_k=cu, alpha=0.5, beta=1.5, C=Cres, out_dtype=torch.float32)
    ref = _ref(A, B, cu, alpha=0.5, beta=1.5, C=Cres)
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-2)


def test_varlen_k_gather_a_columns():
    """A_idx gathers A's COLUMNS (K axis) under cu_seqlens_k."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(4)
    M, N = 32, 32
    lens = [32, 32]
    total_K = sum(lens)
    A = torch.randn(M, total_K, device="cuda", dtype=torch.float16)
    B = torch.randn(total_K, N, device="cuda", dtype=torch.float16)
    A_idx = torch.randperm(total_K, device=A.device, dtype=torch.int32)
    cu = _cu(lens, A.device)
    out = gemm(A, B, cu_seqlens_k=cu, A_idx=A_idx)
    # reference: gather A columns per group, matmul with B's K-slice
    L = len(lens)
    cul = cu.tolist()
    ref = torch.empty(L, M, N, dtype=torch.float32, device=A.device)
    for i in range(L):
        s, e = cul[i], cul[i + 1]
        Ai = A[:, A_idx[s:e].long()].float()
        ref[i] = Ai @ B[s:e, :].float()
    torch.testing.assert_close(out.float(), ref, atol=1e-1, rtol=1e-2)


def test_varlen_k_mutually_exclusive():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    A = torch.randn(16, 32, device="cuda", dtype=torch.float16)
    B = torch.randn(32, 16, device="cuda", dtype=torch.float16)
    cu = _cu([32], A.device)
    with pytest.raises(AssertionError):
        gemm(A, B, cu_seqlens_m=cu, cu_seqlens_k=cu)


def test_validate_varlen_k():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    cu = _cu([10, 20, 30], "cuda")
    assert validate_varlen_k(cu, 60) == 3
    with pytest.raises(AssertionError):
        validate_varlen_k(cu, 61)  # wrong total
