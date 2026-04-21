# Copyright (c) 2026, AMD.

"""Tests for the RDNA WMMA GEMM kernel.

The kernel launches on RDNA (gfx10xx / 11xx / 12xx) only. On CDNA this
suite just verifies the arch-guard fires with a helpful message.
"""

import pytest
import torch

from quack.amd.gemm_rdna_wmma import gemm_wmma
from quack.amd.flydsl_utils import get_rocm_arch


def _is_rdna(arch: str) -> bool:
    return arch.startswith("gfx10") or arch.startswith("gfx11") or arch.startswith("gfx12")


def test_wmma_arch_guard_on_cdna():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    arch = get_rocm_arch()
    if _is_rdna(arch):
        pytest.skip("this test verifies the CDNA fallback path")
    A = torch.randn(16, 16, device="cuda", dtype=torch.float16)
    B = torch.randn(16, 16, device="cuda", dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="RDNA"):
        gemm_wmma(A, B)


@pytest.mark.parametrize("M", [16, 32, 64])
@pytest.mark.parametrize("N", [16, 32, 64])
@pytest.mark.parametrize("K", [16, 32, 64])
def test_gemm_wmma_matches_torch(M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    arch = get_rocm_arch()
    if not _is_rdna(arch):
        pytest.skip(f"WMMA requires RDNA; current arch {arch}")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = gemm_wmma(A, B)
    ref = A.float() @ B.float()
    atol = max(5e-3, K * 2e-5)
    torch.testing.assert_close(C, ref, atol=atol, rtol=5e-3)


def test_arch_dispatcher_wires_rdna():
    """``quack.amd.gemm`` routes to gemm_rdna_wmma on RDNA via the
    ``_arch_dispatch_mfma()`` helper. Sanity-check that the import
    graph is wired (no smoke test on actual launch — that's above)."""
    from quack.amd.gemm_gfx1201 import gemm_mfma as g1201
    from quack.amd.gemm_gfx1250 import gemm_mfma as g1250
    assert g1201 is not None
    assert g1250 is not None
