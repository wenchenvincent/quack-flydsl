# Copyright (c) 2026, AMD.

"""In-kernel varlen-K grouped GEMM (G10, single-launch runtime-K-loop kernel)."""

import pytest
import torch

from quack.amd.gemm_gfx950_varlen_k import gemm_varlen_k_inkernel


def _cu(lens, dev):
    return torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=dev, dtype=torch.int32)


def _ref(A, B, cu):
    L = cu.numel() - 1
    cul = cu.tolist()
    M, N = A.shape[0], B.shape[1]
    out = torch.empty(L, M, N, dtype=torch.float32, device=A.device)
    for i in range(L):
        s, e = cul[i], cul[i + 1]
        out[i] = A[:, s:e].float() @ B[s:e, :].float()
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lens", [[64, 128, 32], [48, 17, 30, 16], [128, 96]])
def test_inkernel_varlen_k(dtype, lens):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(sum(lens))
    M, N = 32, 48
    total_K = sum(lens)
    A = torch.randn(M, total_K, device="cuda", dtype=dtype)
    B = torch.randn(total_K, N, device="cuda", dtype=dtype)
    cu = _cu(lens, A.device)
    out = gemm_varlen_k_inkernel(A, B, cu)
    ref = _ref(A, B, cu)
    assert out.shape == (len(lens), M, N)
    rel = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    assert rel < 3e-3, f"in-kernel varlen-K {lens} rel {rel}"


def test_inkernel_varlen_k_zero_length():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    torch.manual_seed(9)
    M, N = 32, 32
    lens = [48, 0, 64]
    total_K = sum(lens)
    A = torch.randn(M, total_K, device="cuda", dtype=torch.float16)
    B = torch.randn(total_K, N, device="cuda", dtype=torch.float16)
    cu = _cu(lens, A.device)
    out = gemm_varlen_k_inkernel(A, B, cu)
    assert torch.count_nonzero(out[1]) == 0  # empty group -> zero output
    torch.testing.assert_close(out.float(), _ref(A, B, cu), atol=1e-1, rtol=1e-2)


def test_inkernel_matches_host_path():
    """The in-kernel kernel and the host chunked gemm(cu_seqlens_k=) agree."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    from quack.amd.gemm import gemm
    torch.manual_seed(5)
    M, N = 48, 64
    lens = [64, 48, 80]
    total_K = sum(lens)
    A = torch.randn(M, total_K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(total_K, N, device="cuda", dtype=torch.bfloat16)
    cu = _cu(lens, A.device)
    ink = gemm_varlen_k_inkernel(A, B, cu)
    host = gemm(A, B, cu_seqlens_k=cu, out_dtype=torch.float32)
    torch.testing.assert_close(ink, host, atol=1e-1, rtol=1e-2)
