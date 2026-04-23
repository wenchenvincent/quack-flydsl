# Copyright (c) 2026, AMD.

"""Unit tests for tests.amd.bench_hipblaslt_layouts helpers.

Covers the pure-Python helpers only (shape-matrix / FLOPS calc / layout
tensor builders / CSV parser) — not the actual hipBLASLt run.
"""

import pytest
import torch

from tests.amd import bench_hipblaslt_layouts as B


def test_shapes_list_non_empty():
    assert len(B.SHAPES) >= 6
    for entry in B.SHAPES:
        assert set(entry.keys()) >= {"name", "M", "N", "K", "layout", "role"}


def test_flops_calc():
    assert B.matmul_flops(1024, 2048, 512) == 2 * 1024 * 2048 * 512


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_build_layout_nt_produces_matching_output_shape():
    M, N, K = 128, 256, 64
    A, B_t, matmul_fn = B.build_layout("NT", M, N, K, torch.float16)
    y = matmul_fn(A, B_t)
    assert y.shape == (M, N)
    assert A.shape == (M, K)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_build_layout_nn_produces_matching_output_shape():
    M, N, K = 128, 64, 256
    A, B_t, matmul_fn = B.build_layout("NN", M, N, K, torch.float16)
    y = matmul_fn(A, B_t)
    assert y.shape == (M, N)
    # NN: both row-major, no transpose view involved
    assert A.stride(-1) == 1 and B_t.stride(-1) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_build_layout_tn_produces_matching_output_shape():
    M, N, K = 64, 128, 256
    A, B_t, matmul_fn = B.build_layout("TN", M, N, K, torch.float16)
    y = matmul_fn(A, B_t)
    assert y.shape == (M, N)
    # TN: A is a transpose view of some (K, M) tensor
    assert A.stride(-2) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_bench_cuda_events_returns_positive():
    t = B._bench_cuda_events(lambda: torch.zeros(1, device="cuda"), warmup=1, iters=3)
    assert t > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_bench_one_small_shape_runs():
    shape = {"name": "tiny", "role": "sanity", "layout": "NT", "M": 128, "N": 128, "K": 64}
    r = B._bench_one(shape, torch.float16, "f16")
    assert r.seconds > 0 and r.tflops > 0


def test_parse_counter_csv_long_format(tmp_path):
    """rocprofv3 v1 emits one row per (dispatch, counter). The parser filters
    by kernel name prefix and averages counter values across dispatches."""
    # Pass-1 CSV: two Cijk dispatches + one unrelated kernel, two counters (MFMA, VALU).
    csv1 = tmp_path / "pmc_1.csv"
    csv1.write_text(
        '"Correlation_Id","Dispatch_Id","Agent_Id","Queue_Id","Process_Id","Thread_Id",'
        '"Grid_Size","Kernel_Id","Kernel_Name","Workgroup_Size","LDS_Block_Size",'
        '"Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count","Counter_Name",'
        '"Counter_Value","Start_Timestamp","End_Timestamp"\n'
        '1,1,"Agent 2",1,100,100,1024,1,"Cijk_tile",256,0,0,128,0,96,"SQ_INSTS_MFMA",1000,100,200\n'
        '2,1,"Agent 2",1,100,100,1024,1,"Cijk_tile",256,0,0,128,0,96,"SQ_INSTS_VALU",5000,100,200\n'
        '3,2,"Agent 2",1,100,100,1024,1,"Cijk_tile",256,0,0,128,0,96,"SQ_INSTS_MFMA",1100,300,410\n'
        '4,2,"Agent 2",1,100,100,1024,1,"Cijk_tile",256,0,0,128,0,96,"SQ_INSTS_VALU",5100,300,410\n'
        '5,3,"Agent 2",1,100,100,1024,1,"fill_kernel",256,0,0,64,0,32,"SQ_INSTS_MFMA",0,500,600\n'
    )
    # Pass-2 CSV: same 2 Cijk dispatches, one counter (GRBM_GUI_ACTIVE).
    csv2 = tmp_path / "pmc_2.csv"
    csv2.write_text(
        '"Correlation_Id","Dispatch_Id","Agent_Id","Queue_Id","Process_Id","Thread_Id",'
        '"Grid_Size","Kernel_Id","Kernel_Name","Workgroup_Size","LDS_Block_Size",'
        '"Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count","Counter_Name",'
        '"Counter_Value","Start_Timestamp","End_Timestamp"\n'
        '1,1,"Agent 2",1,200,200,1024,1,"Cijk_tile",256,0,0,128,0,96,"GRBM_GUI_ACTIVE",50000,1000,1120\n'
        '2,2,"Agent 2",1,200,200,1024,1,"Cijk_tile",256,0,0,128,0,96,"GRBM_GUI_ACTIVE",51000,1300,1410\n'
    )
    parsed = B.parse_counter_csvs([csv1, csv2], kernel_name_hint="Cijk")
    assert parsed["n_dispatches"] == 2  # 2 Cijk dispatches per pass
    # Counters averaged over 2 dispatches
    assert parsed["counters"]["SQ_INSTS_MFMA"] == pytest.approx(1050.0)
    assert parsed["counters"]["SQ_INSTS_VALU"] == pytest.approx(5050.0)
    assert parsed["counters"]["GRBM_GUI_ACTIVE"] == pytest.approx(50500.0)
    # Time: (200-100) + (410-300) + (1120-1000) + (1410-1300), divided by 4 dispatch-rows,
    # converted from ns to us (div 1000).
    # Actually, we average per-pass-per-dispatch duration to get μs per dispatch.
    assert parsed["dispatch_time_us"] > 0


def test_parse_counter_csv_empty_when_no_match(tmp_path):
    csv1 = tmp_path / "pmc_1.csv"
    csv1.write_text(
        '"Correlation_Id","Dispatch_Id","Agent_Id","Queue_Id","Process_Id","Thread_Id",'
        '"Grid_Size","Kernel_Id","Kernel_Name","Workgroup_Size","LDS_Block_Size",'
        '"Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count","Counter_Name",'
        '"Counter_Value","Start_Timestamp","End_Timestamp"\n'
        '1,1,"Agent 2",1,100,100,1024,1,"fill_kernel",256,0,0,64,0,32,"SQ_INSTS_MFMA",0,500,600\n'
    )
    parsed = B.parse_counter_csvs([csv1], kernel_name_hint="Cijk")
    assert parsed["n_dispatches"] == 0
    assert parsed["counters"] == {}
