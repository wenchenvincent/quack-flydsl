This file provides guidance to coding agents (Claude Code, Codex) when working with code in this repository.

## Project Overview

QuACK (Quirky Assortment of CuTe Kernels) — high-performance CUDA kernels written in [CuTe-DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html), targeting H100 (SM90) and B200/B300 (SM100) GPUs. Package name: `quack-kernels`.

## Build & Development

```bash
# Install (dev)
pip install -e '.[dev]'
pre-commit install

# For CUDA 13.1
pip install -e '.[dev,cu13]' --extra-index-url https://download.pytorch.org/whl/cu130

# Lint & format
ruff check --fix quack/ tests/
ruff format quack/ tests/

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_rmsnorm.py -x
pytest tests/test_rmsnorm.py::test_rmsnorm_fwd -x -k "bfloat16"
```

## CuTe DSL Conventions

For code inside `@cute.jit` or `@cute.kernel` decorated functions, only a subset of Python syntax is supported. Follow [Control Flow](docs/dsl_control_flow.rst) and [Limitations](docs/limitations.rst).

Key rules:
- `cutlass.const_expr()` marks compile-time constants; `cutlass.range_constexpr()` unrolls loops at compile time
- `cutlass.range()` is for dynamic runtime loops
- No early `break`/`continue` in loops; no `return` values from jit functions
- Python lists/dicts inside DSL are static (compile-time only)
- Types must be determinable at compile time (no dependent types)
- Variables defined inside control flow bodies are not accessible outside

## Architecture

### Kernel patterns

**Reduction kernels** (`rmsnorm.py`, `softmax.py`, `cross_entropy.py`) inherit from `ReductionBase` in `reduction_base.py`. They share a pattern: configure cluster size, get tiled copies, allocate reduction buffers with mbarriers, then launch a `@cute.kernel`.

**GEMM** has a multi-layer design:
- `gemm.py` — public API, validates inputs, selects SM version, caches compiled kernels
- `gemm_interface.py` — unified interface across SM versions
- `gemm_sm90.py` / `gemm_sm100.py` — SM-specific implementations
- `gemm_default_epi.py` + `gemm_*_epi.py` — epilogue variants (bias, activation, etc.)
- `gemm_config.py` — `GemmConfig` dataclass with tile sizes, cluster dims, swizzle settings

### Core utilities

- `copy_utils.py` — memory copy operations (shared↔register, async copies, tiled copies)
- `layout_utils.py` — layout algebra (transpose, select, expand, permute)
- `cute_dsl_utils.py` — dtype mapping, device capability queries, parameter base classes
- `tile_scheduler.py` — tile scheduling for persistent kernels
- `varlen_utils.py` — variable-length sequence support

### Testing

Tests use pytest with parametrize across dtypes (`float32`, `float16`, `bfloat16`), dimensions, and batch sizes. Each test includes a reference implementation for numerical validation.

Every test must verify **numerical correctness** against a reference, not just shapes or smoke. A test that only checks `.shape` or "doesn't crash" is not a test — it hides bugs. Always compare kernel output values against a PyTorch reference (float32 for ground truth, same dtype for tolerance baseline).

## Iteration Speed

When iterating on kernel code, run a small subset of tests (1-3 parametrizations) rather than the full test suite. Use `-k` or pass specific test IDs to pytest. Only run the full suite when finalizing changes.

## Debugging Failures

When debugging any failure — kernel correctness, torch.compile interaction, test infrastructure — get to the bottom of it. The goal is not to make the test pass; it is to understand **why** it fails and fix the actual cause.

Do not route around the bug just to make a test pass, for example by pruning a config, skipping a path, resetting state, increasing limits, or switching to a different implementation. A workaround that makes CI green but leaves the bug for users is worse than useless — it hides the problem and the deeper issue will eventually surface at greater cost. Only use workarounds after proving the root cause is external (e.g., upstream PyTorch bug) and documenting why.

Start by reproducing the reported failure, then simplify it to the smallest setting that still fails: reduce batch, M/N/K, tile shape, scheduler options, `swap_ab`, beta/C, dtype, and epilogue features where possible. Keep the failing behavior intact while removing unrelated complexity.

Use `cute.printf` inside `@cute.jit` / `@cute.kernel` code to print the relevant locations, tile coordinates, tensor coordinates, predicates, and values. Print at the boundaries between stages, for example TMA load, MMA accumulator, epilogue register values, register-to-smem, smem contents, and TMA store coordinates, until the first bad stage is identified.

After finding a fix, verify that the minimized repro passes, the original repro passes, and that temporarily disabling the fix makes the regression test fail. Regression tests should encode the failure mode, not only the high-level symptom.

## Code Style

- Favor concise, self-explanatory code
- Line length: 100 (ruff)
- Ruff allows: lambda assignment (E731), single-char vars I/O/l (E741), unused locals (F841)

## AMD port (`quack/amd/`)

`quack/amd/` is the AMDGPU port of the NVIDIA kernels, authored in [FlyDSL](https://github.com/ROCm/FlyDSL) instead of CuTe-DSL. The NVIDIA `quack/*.py` kernels remain untouched; the two backends live side-by-side and can be imported independently (e.g., `from quack.amd import rmsnorm_fwd`).

**Install**: `pip install -e '.[amd]'` (pulls FlyDSL from the `[project.optional-dependencies].amd` group).

**Run the AMD test suite**: `pytest tests/amd/` — gated by the presence of an AMD device via `torch.cuda.is_available()` (ROCm exposes AMDGPUs through the `torch.cuda` surface).

**Supported archs** (see `quack/amd/flydsl_utils.py:get_wave_size`):
- **gfx942** (CDNA3 / MI300X, wave64) — shares the gfx950 MFMA builder.
- **gfx950** (CDNA4 / MI350 / MI355X, wave64) — primary validation target; standard MFMA + scaled MFMA for blockscaled fp8/fp4.
- **gfx1201** (RDNA4, wave32) — reduction kernels work; WMMA GEMM kernel is a stub pending the port.
- **gfx1250** (MI450, wave32) — same WMMA-stub status.

**Key gotchas**:
- `@flyc.kernel` bodies are AST-rewritten: plain `if dynamic_cond:` becomes `scf.if`, `for i in range(dyn)` becomes `scf.for`. External helpers must call `ReplaceIfWithDispatch.scf_if_dispatch` explicitly.
- The AST rewriter only converts `if <expr>:` to `scf.if` when `<expr>` is an `arith.cmpi` result or a flydsl `Boolean`. `if arith.andi(i1, i1):` **silently** lets all lanes through — nest `if arith.cmpi(...)` blocks instead.
- `for i, state in range(start, stop, init=[...])` + `yield [new_state]` is the loop-carried scf.for pattern. When the yield list has length 1, `results` comes back as a bare `ArithValue` (not a list).
- FlyDSL's JIT caches launchers by argument **type**; the first call's `M` for `grid=(M, 1, 1)` freezes in the compiled binary. `tests/amd/conftest.py` wipes `~/.flydsl/cache` at session start and test files parametrize `M` with the largest value first (`[128, 4, 1]`).

**Debugging**: `fx.printf("tid={} val={}", tid, x)` inside `@flyc.kernel`; set `FLYDSL_DUMP_IR=1` for MLIR dumps and `FLYDSL_RUNTIME_ENABLE_CACHE=0` during active iteration.

**FlyDSL reference kernels**: `/workspace/FlyDSL/kernels/` — `preshuffle_gemm.py` (CDNA MFMA pipeline), `blockscale_preshuffle_gemm.py` (gfx950 scaled MFMA), `wmma_gemm_gfx1250.py` (RDNA WMMA). Vendor adapted copies under `quack/amd/` rather than runtime-importing since FlyDSL's `kernels/` isn't wheel-installed.
