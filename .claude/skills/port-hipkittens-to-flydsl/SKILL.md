---
name: port-hipkittens-to-flydsl
description: Use when porting a HipKittens C++ kernel to FlyDSL Python (typically for QuACK's quack/amd/ directory). Covers the structural translation, the load-bearing pipeline patterns, and the caveats discovered while porting four NT GEMM variants on gfx950.
---

# Porting HipKittens kernels to FlyDSL

## Overview

HipKittens (HK) kernels are hand-tuned C++ GPU kernels for AMDGPU. FlyDSL is a Python/MLIR DSL that targets the same hardware. Porting an HK kernel to FlyDSL means recreating its scheduling, LDS layout, and barrier structure in Python while navigating a set of FlyDSL-specific constraints that HipCC's C++ frontend doesn't have.

This skill is the **field guide** for that work. It documents the patterns that work, the patterns that look like they should work but don't, and the specific caveats — including ones that contradict written project lore.

**Use this skill when:**
- Starting a port of an HK kernel to `quack/amd/`
- Debugging a port that compiles but doesn't perform like HK
- Adapting a port across kernel variants (e.g., 4-wave to 8-wave)

**Do NOT use this skill for:**
- Authoring a new kernel from scratch (use FlyDSL's `flydsl-kernel-authoring` skill in `/workspace/FlyDSL/.claude/skills/`)
- Debugging a NaN/incorrect output (use `debug-flydsl-kernel` in FlyDSL)

## The recipe

### 1. Locate the HK reference

HK kernels for bf16/fp16 GEMM live at `/tmp/HipKittens/kernels/gemm/bf16fp32/`. The files I worked with:

- `256_256_64_32_with32x16.cpp` — 8-wave 256×256 tile, 32×16 quadrant
- `256_256_64_32_with16x32.cpp` — 8-wave 256×256 tile, 16×32 quadrant
- `FP8_4wave/4_wave.cu` — 4-wave 256×192 tile (FP8 in HK; we adapt to bf16)

Build HK as a pybind11 .so for bench comparison:

```bash
cd /tmp/HipKittens/kernels/gemm/bf16fp32
export THUNDERKITTENS_ROOT=/tmp/HipKittens
export ROCM_PATH=/opt/rocm
# Edit Makefile SRC=... to pick the variant
make clean && make
# Bench: import tk_kernel; tk_kernel.dispatch_micro(A, Bt, C)
```

### 2. Translate the structural skeleton

Map HK constructs to FlyDSL equivalents:

| HK C++ | FlyDSL Python |
|---|---|
| `__shared__` arrays | `llvm.mlir.global` at gpu.module scope + `llvm.mlir.addressof + GEP` inside kernel (see "LDS allocation" below) |
| `__builtin_amdgcn_s_barrier()` | `rocdl.s_barrier()` — **NOT** `gpu.barrier()`, see Caveat 1 |
| `__builtin_amdgcn_s_waitcnt(0)` | `rocdl.s_waitcnt(VMCNT_0)` where `VMCNT_0 = 0x0F70` |
| `__builtin_amdgcn_s_setprio(N)` | `rocdl.s_setprio(N)` |
| `__builtin_amdgcn_sched_barrier(0)` | `rocdl.sched_barrier(0)` |
| `__builtin_amdgcn_s_sleep(N)` | `rocdl.s_sleep(N)` |
| `if (warp_row == 1) s_barrier();` (HK conditional desync) | `if arith.cmpi(arith.CmpIPredicate.eq, warp_row, fx.Int32(1)): rocdl.s_barrier()` — see Caveat 2 |
| `mma_ABt(...)` | A custom helper (see `quack/amd/gemm_gfx950_mfma_core.py`) or inline `rocdl.mfma_*` |
| `G::load(LDS, HBM, ...)` (HBM→LDS DMA) | `rocdl.raw_ptr_buffer_load_lds(rsrc, lds_ptr, ...)` |
| `load(reg, LDS, ...)` (ds_read into registers) | `llvm.LoadOp(vec_ty, gep, alignment=2, alias_scopes=..., noalias_scopes=...)` |

### 3. Set up the LDS allocation (per-stage globals + alias scopes)

This is the **load-bearing** part of getting HK-comparable perf. See the full implementation guide at `docs/superpowers/specs/2026-06-04-flydsl-lds-provenance-todo.md` (the "2026-06-09 implementation LANDED — how it works" section).

The minimum recipe:

```python
# At kernel-compile-time, outside the kernel function:
AS_STAGE_BYTES = BLOCK_M * BLOCK_K * DTYPE_BYTES
BS_STAGE_BYTES = BLOCK_N * BLOCK_K * DTYPE_BYTES
LDS_SYMS_A = (f"<kernel>_smem_as0_{dtype}_{k}_{n}", f"<kernel>_smem_as1_{dtype}_{k}_{n}")
LDS_SYMS_B = (f"<kernel>_smem_bs0_{dtype}_{k}_{n}", f"<kernel>_smem_bs1_{dtype}_{k}_{n}")
LDS_ALIAS_DOMAIN = f'#llvm.alias_scope_domain<id = "<kernel>_{dtype}_{k}_{n}.lds">'
SCOPE_IDS = ("as0", "as1", "bs0", "bs1")

# Inside @flyc.kernel body:
_LDS_PTR_TY = ir.Type.parse("!llvm.ptr<3>")
_GEP_DYN = -(2 ** 31)
def _gep_lds(base, off_i32):
    return llvm.getelementptr(_LDS_PTR_TY, base, [off_i32], [_GEP_DYN], T.i8, None)
def _scope_attr(ids):
    inner = ", ".join(f'#llvm.alias_scope<id = "{i}", domain = {LDS_ALIAS_DOMAIN}>' for i in ids)
    return ir.Attribute.parse(f"[{inner}]")

_SCOPE = {s: _scope_attr((s,)) for s in SCOPE_IDS}
_NOALIAS = {s: _scope_attr(tuple(o for o in SCOPE_IDS if o != s)) for s in SCOPE_IDS}
_as_bases = (llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYMS_A[0]),
             llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYMS_A[1]))
# ... and similar for bs

# Inside launch_*_kernel (the @flyc.jit), emit the 4 globals at module scope:
with ir.InsertionPoint(ctx.gpu_module_body):
    linkage = ir.Attribute.parse('#llvm.linkage<external>')
    for sym, size in ((LDS_SYMS_A[0], AS_STAGE_BYTES), (LDS_SYMS_A[1], AS_STAGE_BYTES),
                      (LDS_SYMS_B[0], BS_STAGE_BYTES), (LDS_SYMS_B[1], BS_STAGE_BYTES)):
        llvm.GlobalOp(global_type=ir.Type.parse(f"!llvm.array<{size} x i8>"),
                      sym_name=sym, linkage=linkage, addr_space=3, alignment=1024)
```

For helper functions that take a `lds_stage` parameter:

```python
def lds_matrix_a_half(lds_stage, m_half):
    view = as_views[lds_stage]  # Python int → compile-time tuple lookup
    # ... use view.linear_offset((row, col)) and view.vec_load((row, col), vec_size)

def ldg_sts_a_call(k_offset, lds_stage, i):
    # ... compute lds_offset ...
    lds_off_i32 = arith.index_cast(T.i32, lds_offset)
    lds_off_uniform = rocdl.readfirstlane(T.i32, lds_off_i32)
    lds_ptr = _gep_lds(_as_bases[lds_stage], lds_off_uniform)
    rocdl.raw_ptr_buffer_load_lds(
        A_.rsrc, lds_ptr, arith.constant(DMA_BYTES, type=T.i32),
        global_offset, arith.constant(0, type=T.i32),
        arith.constant(0, type=T.i32), arith.constant(1, type=T.i32),
        alias_scopes=_as_scopes[lds_stage],
        noalias_scopes=_as_noalias[lds_stage],
    )
```

### 4. 2× unroll the main loop

If the HK reference uses runtime `current_stage = state[1]` from an scf.for, **you must 2× unroll** to make `lds_stage` a Python int at every helper call site. Python tuples can't index by runtime values.

Template:

```python
def kstep_body(k_now, current_stage, next_stage, c_frags, ...):
    """One K-step body — current_stage and next_stage are Python ints (0 or 1)."""
    a_frags = lds_matrix_a_half(current_stage, ...)
    # ... prefetches use next_stage ...
    ldg_sts_a_call(k_prefetch, next_stage, ...)

TOTAL_K_STEPS = k // BLOCK_K
OUTER_ITERS = TOTAL_K_STEPS // 2
assert TOTAL_K_STEPS % 2 == 0, "<kernel> requires K % 128 == 0"

for _oi, state in range(0, OUTER_ITERS - 1, init=init_state):
    k_now = state[0]
    c_frags = list(state[1 : 1 + C_FRAGS_LEN])
    kstep_body(k_now, 0, 1, c_frags, ...)             # sub-step 0
    kstep_body(k_now + fx.Int32(BLOCK_K), 1, 0, c_frags, ...)  # sub-step 1
    results = yield [k_now + fx.Int32(2 * BLOCK_K)] + c_frags

# Epilogue handles the last 2 K-steps WITHOUT prefetching past them.
```

**Watch the epilogue prefetch carefully** — the last main-loop iter prefetches the SECOND-to-last K-step, not the last. The epilogue must process the second-to-last K-step (already in LDS) AND prefetch the last K-step before reading it.

### 5. Verify

Three checks, in order:

1. **Correctness**: `max_err` close to bf16 noise floor (2.0 for 8K-K accumulation).
2. **LLVM IR**: `~/.flydsl/debug/<kernel>/20_llvm_ir.ll` should show:
   - 4 `external addrspace(3) global` declarations
   - 0 `inttoptr` to `ptr addrspace(3)`
   - ~100+ `!alias.scope` sites on `load <vec>, ptr addrspace(3)` and `call @llvm.amdgcn.raw.ptr.buffer.load.lds`
3. **ISA**: `21_final_isa.s` `vmcnt(0)` drains drop dramatically (HK reference is typically 1 per kernel; QuACK pre-fix is 10-30).

If LLVM IR has 0 alias.scope sites despite emitting `alias_scopes=` kwargs, you likely have only 1 LDS global — `AMDGPULowerModuleLDSPass.cpp:1268` only creates scopes when `NumberVars > 1`. Emit 4 globals.

## Caveats (read every one)

### Caveat 1: `gpu.barrier()` is NOT `s_barrier`

`gpu.barrier()` lowers to `fence release(workgroup) + s_barrier + fence acquire(workgroup)`. The release fence forces `SIInsertWaitcnts` to drain `vmcnt(0)` before EVERY barrier on gfx950, even though `s_barrier` itself does NOT auto-drain on gfx950 (see Caveat 4). On a 16-barrier-per-K-step kernel this is **the dominant perf cost** — fixing this alone took our 16x32 from 520 → 961 TF/s (+85%).

**Fix**: Use `rocdl.s_barrier()` directly. Add explicit `rocdl.s_waitcnt(VMCNT_N)` drains only where the data flow requires them.

### Caveat 2: HK's conditional `s_barrier` works at 1 WG/CU on gfx950

The HK desync pattern is:

```cpp
__builtin_amdgcn_s_waitcnt(0);
__builtin_amdgcn_s_barrier();      // all 8 waves sync
if (warp_row == 1) {
    __builtin_amdgcn_s_barrier();  // only 4 waves execute
}
```

Earlier kernel docs (now corrected) claimed this requires **2 WGs/CU residency** or it deadlocks. That's WRONG. On gfx950 the workgroup barrier arrival counter accumulates arrivals across all `s_barrier` ops; `warp_row==1`'s extra barrier pairs with `warp_row==0`'s next barrier through the shared counter. ATT-verified in `gemm_gfx950_nt_pingpong_16x32.py` at 1 WG/CU.

**Fix**: Use HK's pattern verbatim. Do NOT substitute `s_sleep` "because we only have 1 WG/CU."

### Caveat 3: Conditional `s_barrier` desync REQUIRES ≥2 barriers per K-step

The conditional `s_barrier` puts the two warp groups exactly **one barrier interval** out of phase forever after. If you have 4 barriers per K-step, that's 1 cluster — a clean pingpong. If you have **1 barrier per K-step**, that's a full K-step of lag → warp_row==1 reads stage 0 while warp_row==0 has already prefetched K+1 data into stage 0 → NaN output.

**Fix**: Either ensure ≥2 barriers per K-step OR use `s_sleep` for the desync. The 1-barrier-per-K-step design relies on compiler-driven ILP within a single wave's instruction stream — the per-warp desync isn't doing real work there.

### Caveat 4: gfx950 has `BackOffBarrier`; `s_barrier` does NOT auto-drain

`SIInsertWaitcnts.cpp:2633` only auto-inserts `vmcnt(0)` before `S_BARRIER` when **neither** `AutoWaitcntBeforeBarrier` nor `BackOffBarrier` is set. gfx950 inherits from `FeatureISAVersion9_4_Common` which **has `FeatureBackOffBarrier`** (see `llvm/lib/Target/AMDGPU/AMDGPU.td:1686`). So `s_barrier` on gfx950 does NOT auto-drain. HK's pattern of bare `__builtin_amdgcn_s_barrier()` + explicit `__builtin_amdgcn_s_waitcnt(0)` at specific points confirms this.

**Implication**: Don't add explicit `vmcnt(0)` drains "to be safe" — they cost cycles. Add them only where the data flow requires (typically before a barrier that needs to make a pending `buffer_load_lds` write visible to another warp's post-barrier `ds_read`).

### Caveat 5: AST rewriter requires `fx.const_expr(...)` on Python-bool conditions

Upstream FlyDSL's AST rewriter (`flydsl/compiler/ast_rewriter.py:716`) only treats `if expr:` as compile-time when `expr` is a `const_expr(...)` call. Plain `if PY_CONST:` patterns get dispatched through `scf_if_dispatch`, and variables FIRST-defined inside a branch don't escape (the dispatcher only returns vars that were already in the active symbol scope before the if).

**Symptom**: `NameError: name 'pid' is not defined` or similar, at a use site downstream of an `if PY_CONST:` where the var was assigned only inside the branch.

**Fix**: Wrap the condition: `if fx.const_expr(XCD_SWIZZLE > 1):` or pre-initialize the var in the outer scope before the if.

### Caveat 6: Classes defined inside `@flyc.kernel` body break the AST rewriter

If you write a class (e.g., `class _LDSPtr: ...`) inside the kernel body, Python's name-mangling kicks in on the AST rewriter's injected `__check_local_var(...)` calls — they get mangled to `_LDSPtr__check_local_var`, which is undefined. Result: `NameError: name '_LDSPtr__check_local_var' is not defined`.

**Fix**: Use closure namespaces instead of classes:

```python
def _make_lds_view(...):
    def linear_offset(idxs): ...
    def vec_load(idxs, vec_size): ...
    ns = type("LDSView", (), {})()  # anonymous namespace, no class scope
    ns.linear_offset = linear_offset
    ns.vec_load = vec_load
    return ns
```

### Caveat 7: Per-region scopes (2 scopes) do NOT reduce vmcnt drains for pingpong

Per-REGION scopes (AS vs BS, 2 scopes total) don't help. Pingpong writes `AS@next_stage` while reading `AS@current_stage` — both in the same `lds_as` scope, so `ScopedNoAliasAA` says "potentially aliasing" → drain. **Use per-STAGE scopes**: AS@0, AS@1, BS@0, BS@1 (4 scopes).

For this to work, `lds_stage` MUST be a Python int at every helper call site (use 2× unrolling).

### Caveat 8: HipCC emits TBAA + `__restrict__`-derived noalias "for free"

When you compare an HK kernel's IR to your FlyDSL kernel's IR, HK's loads/stores will have `!alias.scope` / `!noalias` metadata even though the C++ source doesn't explicitly declare them — HipCC infers them from the C++ types and `__restrict__` qualifiers. FlyDSL does **not** infer this metadata. You must attach scopes explicitly via the `alias_scopes=` / `noalias_scopes=` kwargs on `llvm.LoadOp` and `rocdl.raw_ptr_buffer_load_lds`. Without explicit attachment, `SIInsertWaitcnts` falls back to "all LDS aliases everything" → full `vmcnt(0)` drains.

### Caveat 9: `inttoptr` to `ptr addrspace(3)` destroys provenance

The legacy pattern in `flydsl/expr/rocdl/__init__.py:386-393` is:

```python
lds_base = memref.extract_aligned_pointer_as_index(raw_memref)  # → index
total_byte_idx = lds_base + byte_off                              # → index
addr_i32 = arith.index_cast(T.i32, total_byte_idx)                # ← provenance dies here
ptr_val = llvm.inttoptr(lds_ptr_ty, addr_i32)                     # → ptr<3>, phantom origin
```

The LLVM IR has `inttoptr i64 %x to ptr addrspace(3)`. `AMDGPULowerModuleLDSPass` can't attach alias scope through inttoptr → no metadata → `SIInsertWaitcnts` falls back to "all aliases everything" → full drains.

**Fix**: Construct the LDS pointer via `llvm.mlir.addressof @<global> + llvm.getelementptr` directly. No intermediate integer.

### Caveat 10: AGPR forcing (`amdgpu-agpr-alloc=N,N`) is load-bearing only for tight register budgets

QuACK's older NN-big kernel forces `amdgpu-agpr-alloc=192,192` (= reserve 192 AGPR slots for C accumulators) because at higher occupancy targets the C tile didn't fit in the architectural VGPRs. For the 8-wave NT pingpong kernels (32x16 and 16x32), this forcing actively HURTS perf by ~6% — the unified 256-VGPR/SIMD pool fits everything as VGPRs (HK's profile: 210 VGPR / 0 AGPR). The 4-wave HK kernel still NEEDS the forcing because its C tile is bigger (192 fp32/lane vs 128) and per-wave VGPR budget is tighter.

**Rule of thumb**: If `~/.flydsl/debug/<kernel>/21_final_isa.s` shows `num_vgpr=256` with non-zero `vgpr_spill_count`, you need AGPR offload. Otherwise let the compiler decide.

### Caveat 11: NN-family kernels (NN, NN_big, TN, splitk) have a structural blocker for per-stage scopes

They reuse the A-region LDS for C write-back coalescing (`cs_` aliased at `smem_a_offset = 0`, size 128 KB). The original single-global allocation lets C and AB share the same LDS bytes temporally. Splitting AB into 4 per-stage globals + a separate 128 KB C global pushes total LDS to 256 KB, which overflows the 160 KB cap.

**If you need to migrate one of these**: either eliminate the LDS C write-back (direct register→HBM stores) or split C across the 4 per-stage globals. See `~/.claude/projects/-workspace-quack/memory/reference_per_stage_scopes_blocker.md` for the full design.

### Caveat 12: Watch for stale "this needs 2 WGs/CU" comments

The QuACK codebase has comments asserting various things require 2 WGs/CU residency that turned out to be wrong on gfx950. If you read a comment like "HK needs 2 WGs/CU for this barrier pattern" — verify against the gfx950 ISA reference before trusting it. The conditional `s_barrier` is the canonical example; the 2-WG/CU claim was repeated for ~2 years before being disproven by ATT analysis.

### Caveat 13: First-time JIT cache and grid-dimension bake-in

FlyDSL's JIT caches launchers by argument **type**. The first call's `M` for `grid=(M, 1, 1)` freezes in the compiled binary. The QuACK test infrastructure handles this — `tests/amd/conftest.py` wipes `~/.flydsl/cache` at session start, and test files parametrize `M` with the largest value first (`[128, 4, 1]`). If you're benching outside the test harness, clear `~/.flydsl/cache` between shape changes.

### Caveat 14: `FLYDSL_DUMP_IR=1` is your friend

Set `FLYDSL_DUMP_IR=1` when benching. The dumps appear in `~/.flydsl/debug/<kernel_name>/`:
- `20_llvm_ir.ll` — final LLVM IR before MC. Check globals, alias.scope metadata, `inttoptr` count.
- `21_final_isa.s` — the AMDGPU assembly. Check `vmcnt(0)` count, VGPR/AGPR usage, `vgpr_spill_count`.

For verification after applying the per-stage scope fix, the metric to watch is `vmcnt(0)` drains. HK kernels typically have **1** per kernel; the QuACK baseline (pre-fix) is **10-30**.

## Bench harness pattern

```python
import torch, time
from quack.amd.<your_kernel> import <fn>

torch.manual_seed(0)
M, N, K = 8192, 8192, 8192   # adjust per shape constraint
A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').contiguous()
B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda').contiguous()  # NT
C = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')

<fn>(A, B, C); torch.cuda.synchronize()
C_ref = (A.float() @ B.float().T).bfloat16()  # NT: B is (N, K)
print(f"max_err={(C - C_ref).abs().max().item():.3f}")  # ~2.0 expected for bf16 + 8K-K

for _ in range(5): <fn>(A, B, C)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50): <fn>(A, B, C)
torch.cuda.synchronize()
print(f"{2*M*N*K*50/(time.time()-t0)/1e12:.1f} TF/s")
```

Per-bench noise is typically <1%. Always do 3-5 runs and report median.

## Apply-this-pattern checklist

For each kernel:

1. ☐ Identify the HK reference structure (clusters per K-step, barrier count, prefetch lookahead).
2. ☐ Remove `gpu.barrier()` → `rocdl.s_barrier()` + explicit `rocdl.s_waitcnt(VMCNT_N)` (Caveat 1).
3. ☐ For pingpong designs: use HK's conditional `s_barrier` desync (Caveat 2), NOT `s_sleep` — but only if ≥2 barriers per K-step (Caveat 3).
4. ☐ Wrap all `if PY_CONST:` in `fx.const_expr(...)` (Caveat 5).
5. ☐ Replace `SmemAllocator` + `STensor` + `inttoptr` with 4 `llvm.mlir.global`s + per-stage views (Caveat 9). 2× unroll if the original used runtime stage state.
6. ☐ Attach `alias_scopes=` and `noalias_scopes=` on every `llvm.LoadOp` and `rocdl.raw_ptr_buffer_load_lds` (Caveat 8). Per-STAGE scopes, not per-region (Caveat 7).
7. ☐ Check AGPR usage — only force `amdgpu-agpr-alloc=N,N` if you have VGPR spills (Caveat 10).
8. ☐ Verify in `20_llvm_ir.ll`: 4 globals, 0 `inttoptr` to ptr<3>, >100 `!alias.scope` sites.
9. ☐ Verify in `21_final_isa.s`: `vmcnt(0)` count ≈ 1 (HK match), `vgpr_spill_count: 0`.
10. ☐ Bench against the HK reference. Acceptable: within ±5% of HK on most shapes. Investigate gaps >10%.

## References

- Implementation guide: `docs/superpowers/specs/2026-06-04-flydsl-lds-provenance-todo.md` (in this repo)
- FlyDSL reference: `/workspace/FlyDSL/.claude/skills/flydsl-kernel-authoring/SKILL.md`
- HK source: `/tmp/HipKittens/kernels/gemm/bf16fp32/`
- The four migrated NT kernels as worked examples:
  - `quack/amd/gemm_gfx950_nt_pingpong.py` (8-wave 32x16)
  - `quack/amd/gemm_gfx950_nt_pingpong_16x32.py` (8-wave 16x32)
  - `quack/amd/gemm_gfx950_nt_4wave_hk.py` (4-wave HK 2-K-step pipeline)
  - `quack/amd/gemm_gfx950_nt_4wave.py` (4-wave 1-K-step rebalanced)
- HK ATT analysis skill: `/tmp/HipKittens/.claude/skills/gemm-att-analysis/SKILL.md` (if present)
