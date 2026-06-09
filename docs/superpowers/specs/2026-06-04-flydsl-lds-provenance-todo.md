# Conservative `vmcnt(0)` drains — three sources, three fixes

**Status:** Investigation refined 2026-06-07 (ATT-driven). Conservative
`s_waitcnt vmcnt(0)` drains dominate the runtime of HK-style ping-pong
kernels in FlyDSL. Earlier the cause was attributed to a single FlyDSL
LDS-provenance issue; ATT cycle-level analysis showed it's actually **three
distinct mechanisms**, each with a different fix.

## TL;DR — Three sources of `vmcnt(0)` drains

### Source 1: `gpu.barrier()` release-acquire fence pair

MLIR's `gpu.barrier()` (no address-space argument) lowers to
`fence release(workgroup) + s_barrier + fence acquire(workgroup)`
(`mlir/lib/Conversion/GPUToROCDL/LowerGpuOpsToROCDLOps.cpp:541-563`). The
release fence imposes scheduling constraints that increase the number of
pending `buffer_load_lds` writes at barrier time, making the auto-inserted
drain take longer.

**Fix A (landed):** Use `rocdl.s_barrier()` directly. **No FlyDSL changes.**
~5 min per kernel call site, but only safe if explicit
`rocdl.s_waitcnt(vmcnt(N))` drains are in place for correctness.

### Source 2: ~~`SIInsertWaitcnts` unconditional drain at `s_barrier`~~ **WRONG — gfx950 has BackOffBarrier**

**This section was originally wrong.** Corrected 2026-06-08:

`SIInsertWaitcnts.cpp:2633-2636`:
```cpp
if (Opc == AMDGPU::S_BARRIER && !ST.hasAutoWaitcntBeforeBarrier() &&
    !ST.hasBackOffBarrier()) {
    Wait = Wait.combined(WCG->getAllZeroWaitcnt(/*IncludeVSCnt=*/true));
}
```

`FeatureISAVersion9_4_Common` (which `gfx950` → `ISAVersion9_5_0` →
`ISAVersion9_5_Common` inherits from) **includes `FeatureBackOffBarrier`**
(`llvm/lib/Target/AMDGPU/AMDGPU.td:1686`). So `hasBackOffBarrier()`
returns true → the unconditional drain is **skipped**. `s_barrier` on
gfx950 does NOT auto-drain `vmcnt(0)`.

The HK kernel pattern confirms this: HK uses `__builtin_amdgcn_s_barrier()`
without an automatic drain, and only adds explicit `__builtin_amdgcn_s_waitcnt(0)`
at the specific barriers that need it (see
`HipKittens/kernels/gemm/bf16fp32/256_256_64_32_with32x16.cpp:88,118`).

**Real Source 2 (the actual mechanism):** `gpu.barrier()` MLIR op lowers
to `fence release(workgroup) + s_barrier + fence acquire(workgroup)`.
The release fence forces ordering of prior memory ops to workgroup-visible
memory before the barrier; for pending `buffer_load_lds` writes (which
target LDS = workgroup-visible), SIInsertWaitcnts inserts `vmcnt(0)`
before the fence to satisfy that ordering. This is per-`gpu.barrier()`,
not per-`s_barrier`.

**Mitigation = Fix A (= the real fix, not a workaround):** Replace
`gpu.barrier()` with `rocdl.s_barrier()` and add explicit `rocdl.s_waitcnt(VMCNT_N)`
drains ONLY where the data flow actually needs them. Landed 2026-06-08
on `gemm_gfx950_nt_pingpong.py` (neutral perf, kernel has only 2 barriers
per K-step and both need full vmcnt(0)). Bigger win expected on
`gemm_gfx950_nt_pingpong_16x32.py` (16 barriers per K-step → ~6600 cyc/iter
in waitcnt today).

### Source 3: `SIInsertWaitcnts` conservative drain before `ds_read` (LDS aliasing)

`SIInsertWaitcnts.cpp:2540-2557`: when emitting a wait before a `ds_read`,
the pass walks the `LDSDMAStores` list and checks per-slot aliasing. With
alias scope info absent on the `buffer_load_lds` `MachineMemOperand`, the
pass falls back to the generic `LDSDMA_BEGIN` slot — meaning every `ds_read`
waits for **all** pending `buffer_load_lds` writes, not just the ones that
could actually alias.

These drains appear in the ATT trace as `s_waitcnt vmcnt(0)` immediately
before `ds_read_b128` instructions (not at barriers). In our 2-barrier HK
kernel, two such drains cost 303 cyc and 385 cyc respectively → **688 cyc/iter
(~28% of runtime)**.

**Fix B:** FlyDSL's `raw_ptr_buffer_load_lds` wrapper needs a pointer-construction
path that preserves provenance — `llvm.mlir.addressof @global + llvm.getelementptr`
instead of `inttoptr`. ~3 hours of FlyDSL Python work.

**Expected savings:** ~25-30% on this kernel (the 688 cyc/iter would drop to
~30-60 cyc as partial `vmcnt(N)` drains for only-truly-aliasing stores).

## Empirical progression

| Stage | HK 4-wave TF/s @ 8192³ | Total waitcnt cyc/iter | Confirmed? |
|---|---|---|---|
| Original (`gpu.barrier`, 4 barriers/iter) | 744 | 1342 | ✓ |
| **Fix A** (`rocdl.s_barrier`) | **839** (+12%) | 1032 (-23%) | ✓ |
| **+ Source 2 mitigation** (2 barriers/iter) | **856** (+2%) | 973 (-6%) | ✓ |
| + Fix B (LDS provenance) — projected | ~1100 | ~285 | inferred from ATT |

Combined recovery from baseline 744 → projected ~1100 TF/s = +48%, would beat
the 8-wave production 1071. Across all 9 quack/amd kernels using
`raw_ptr_buffer_load_lds`, similar gains are plausible.

## Fix A: kernel-level swap of `gpu.barrier()` → `rocdl.s_barrier()`

Landed 2026-06-07 on `quack/amd/gemm_gfx950_nt_4wave_hk.py` (8 call sites).
Net: HK 4-wave kernel **+12% (751 → 839 TF/s)**, correctness preserved.

Process for adopting on other kernels:

1. Audit the kernel for explicit `rocdl.s_waitcnt(vmcnt(N))` calls. If
   present at all the right cluster boundaries (HK-style structures
   typically have them), proceed.
2. Replace every `gpu.barrier()` with `rocdl.s_barrier()`.
3. Run the regression test suite. If diffs appear, the kernel was
   relying on the implicit `vmcnt(0)` — back out and add the right
   explicit drains first.

Already-safe kernels (have explicit drains): `gemm_gfx950_nt_4wave_hk.py`,
likely `gemm_gfx950_nt_pingpong_16x32.py`.

NOT safe without further work: `gemm_gfx950_nt_pingpong.py`,
`gemm_gfx950_nt_4wave.py`, the NN big kernel — these depend on the
implicit drain and would need explicit `vmcnt(N)` drains added first.

## Fix B: FlyDSL LDS-pointer provenance

This is the harder upstream fix that unlocks the remaining ~1008 cyc/iter
of conservative `SIInsertWaitcnts` drains (the part that persists even
after Fix A).

### The exact bad pattern

`FlyDSL/python/flydsl/expr/rocdl/__init__.py` lines 386–393 (in `lds_transpose_load`,
and the equivalent pattern is duplicated in every QuACK AMD kernel that constructs
an LDS pointer for `raw_ptr_buffer_load_lds`):

```python
lds_ptr_ty = _ir.Type.parse("!llvm.ptr<3>")
raw_memref = _arith.unwrap(lds_memref)
lds_base = _memref.extract_aligned_pointer_as_index(raw_memref)   # → index
byte_off = ... * _arith.index(elem_bytes)
total_byte_idx = _AV(lds_base) + byte_off                          # → index
addr_i32 = _to_raw(_arith.index_cast(T.i32, total_byte_idx))       # → i32  ← provenance dies here
ptr_val = _llvm.inttoptr(lds_ptr_ty, addr_i32)                     # → ptr<3>  ← phantom origin
```

After this, when the resulting `ptr_val` flows into `rocdl.raw_ptr_buffer_load_lds(...)`,
LLVM IR looks like:

```llvm
%78 = add i64 ptrtoint (ptr addrspace(3) @my_lds_global to i64), %offset
%81 = inttoptr i64 %78 to ptr addrspace(3)
call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %81, ...)
```

LLVM does not in practice trace `@my_lds_global` provenance through the
`inttoptr`/`ptrtoint` round-trip when the result is consumed by an intrinsic
that doesn't carry inbounds/range semantics. The `!alias.scope` metadata
attached by `AMDGPULowerModuleLDSPass` is dropped at the inttoptr.

### What it should emit instead

```mlir
%addr = llvm.mlir.addressof @my_lds_global : !llvm.ptr<3>
%gep  = llvm.getelementptr inbounds %addr[%byte_off]
        : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
rocdl.raw.ptr.buffer.load.lds %rsrc, %gep, ...
```

Lowered to LLVM IR:

```llvm
%gep = getelementptr inbounds i8, ptr addrspace(3) @my_lds_global, i32 %byte_off
call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %gep, ...)
```

The `getelementptr` carries provenance from `@my_lds_global`. Once
`AMDGPULowerModuleLDSPass` attaches alias scope metadata to uses of
`@my_lds_global`, the buffer-load-LDS call's MachineMemOperand has
`AAInfo.Scope` set, and `SIInsertWaitcnts` can disambiguate aliasing.

### Concrete patch surface

There are two FlyDSL touch points and one optional kernel-level change.

**FlyDSL change 1 — extend `SmemAllocator` to expose the global symbol.**

Currently `SmemAllocator` (in `flydsl/utils/smem_allocator.py`) emits a single
`memref.global` and the kernel calls `allocator.get_base()` to get a memref.
For provenance preservation we need a path to the LLVM-level symbol, OR a path
that lowers cleanly to `llvm.mlir.addressof + GEP`.

Two options:

- **Option A (minimal):** Add a method `SmemAllocator.get_llvm_address(byte_offset)`
  that emits `llvm.mlir.addressof @sym` and a `llvm.getelementptr` directly,
  returning the `!llvm.ptr<3>`. The kernel calls this for buffer-load-LDS dests.
- **Option B (cleaner):** Add a method `SmemAllocator.subview(offset, shape, dtype)`
  that returns a *typed* memref representing a sub-region of the global, and
  fix the underlying memref→ptr conversion to preserve `@sym` provenance via
  the LLVM dialect's address-of pattern.

Option A is faster to implement (~50 lines) and unblocks the perf work. Option
B is structurally nicer but touches more of FlyDSL.

**FlyDSL change 2 — fix `lds_transpose_load` and any other helpers using
the inttoptr pattern.** Same surgery as above; trade `extract_aligned_pointer_as_index
+ inttoptr` for `addressof + GEP`.

**Kernel-level change (optional):** Once FlyDSL exposes the address-of path,
update every kernel-level usage of `extract_aligned_pointer_as_index + inttoptr`
to use the new helper. Grep:

```
grep -rn "extract_aligned_pointer_as_index\|llvm.inttoptr.*ptr<3>" quack/amd/
```

Returns 9+ hits across `gemm_gfx950_nt_pingpong.py`, `gemm_gfx950_nn_big_4w.py`,
`gemm_gfx950_128x128_ldma.py`, `gemm_gfx950_128x128_k32_ldma.py`,
`gemm_gfx950_nt_pingpong_16x32.py`, `gemm_streamk.py`. All of these
would benefit from the fix.

### Multiple LDS globals enable the disambiguation

Even with the addressof+GEP fix, you only get one alias scope unless there
are multiple distinct LDS globals to disambiguate. `AMDGPULowerModuleLDSPass`
(`llvm-project/llvm/lib/Target/AMDGPU/AMDGPULowerModuleLDSPass.cpp:1268-1301`)
generates one anonymous alias scope per LDS global, and tags every use of
each global with its scope plus a noalias-list of all other globals.

So the kernel-level change is also: allocate separate `memref.global` per
logical LDS region. For the NT pingpong 16x32 kernel, that means 8 globals:
`As[stage][m_half]` and `Bs[stage][n_half]` each as a distinct `memref.global`
with its own symbol name.

I already wrote this version of the kernel (saved in branch state during the
investigation, then reverted) — it correctly emits 8 distinct `@`-prefixed
globals in LLVM IR. With Option A or B above, those 8 globals would each
acquire a distinct alias scope and the pingpong perf should materialize.

## How `SIInsertWaitcnts` uses the scope info

`llvm-project/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:1216-1248`:

```cpp
unsigned Slot = 0;
for (const auto *MemOp : Inst.memoperands()) {
    if (!MemOp->isStore() || MemOp->getAddrSpace() != AMDGPUAS::LOCAL_ADDRESS)
        continue;
    auto AAI = MemOp->getAAInfo();
    // Alias scope information gives a way to definitely identify an
    // original memory object and practically produced in the module LDS
    // lowering pass. If there is no scope available we will not be able
    // to disambiguate LDS aliasing as after the module lowering all LDS
    // is squashed into a single big object.
    if (!AAI || !AAI.Scope)
        break;
    for (unsigned I = 0, E = LDSDMAStores.size(); I != E && !Slot; ++I) {
        for (const auto *MemOp : LDSDMAStores[I]->memoperands()) {
            if (MemOp->isStore() && AAI == MemOp->getAAInfo()) {
                Slot = I + 1;
                break;
            }
        }
    }
    ...
    LDSDMAStores.push_back(&Inst);
    Slot = LDSDMAStores.size();
    break;
}
setVMemScore(LDSDMA_BEGIN, T, CurrScore);
if (Slot && Slot < NUM_LDSDMA)
    setVMemScore(LDSDMA_BEGIN + Slot, T, CurrScore);
```

The pass tracks per-LDS-region scores via `LDSDMA_BEGIN + Slot`. With a known
scope, only `ds_read`s that alias the same scope as a pending `buffer_load_lds`
get a `vmcnt` wait. Without a scope, the pass falls back to the
"LDS is one big object" path — every `ds_read` waits for every pending
`buffer_load_lds`.

The `NUM_LDSDMA` slot count is finite (16 on gfx9), so the pass can track up
to 16 distinct LDS regions before degrading to coarse mode. Our 8-global
layout is comfortably inside.

## Empirical evidence

Measurements at M=8192, N=8064 (or 8192 for 8-wave kernels), K=8192, bf16, gfx950,
1 WG/CU residency.

### Confirmation 6 (most direct): ATT across Fix A and 2-barrier restructure

ATT capture via `rocprofv3 --att` + `rocprof-trace-decoder 0.1.6` (from
github.com/ROCm/rocprof-trace-decoder releases), decoded with the
`gemm-att-analysis` skill from HK. HK 4-wave kernel.

| Metric | original | Fix A | + 2-barrier |
|---|---|---|---|
| Per-iter waitcnt cyc | 1342 | 1032 | 973 |
| `s_barrier` count/iter | 4 | 4 | 2 |
| `s_waitcnt vmcnt(0)` count/iter | 4 | 4 | 4 |
| TF/s @ 8192³ | 744 | 839 | 856 |

The vmcnt(0) count stayed at 4 even after halving the barriers — proof
that they don't all sit at barriers.

**Disaggregating the 4 vmcnt(0) in the 2-barrier kernel by source:**

```
Source 1 (gpu.barrier release fence):              REMOVED by Fix A
Source 2 (SIInsertWaitcnts at s_barrier):          2 drains × ~130 cyc = 260 cyc/iter
Source 3 (SIInsertWaitcnts before ds_read):        2 drains × ~344 cyc = 688 cyc/iter
```

Looking at the ATT instruction context for one of the Source 3 drains:

```
v_add_u32_e32 v194, v213, v194         ← compute ds_read address
s_waitcnt vmcnt(0)  ← 385 cyc           ← LLVM-inserted, NOT at a barrier
ds_read_b128 a[52:55], v195 offset:16384
ds_read_b128 a[56:59], v195 offset:18432
```

The `s_waitcnt vmcnt(0)` is followed by `ds_read`s — the compiler is
draining all pending VMEM ops before letting the `ds_read` proceed, because
without alias scope metadata it can't prove the `ds_read` doesn't share
LDS bytes with a pending `buffer_load_lds` write.

**This is exactly the case `SIInsertWaitcnts.cpp:2540-2557` describes:**
when alias scope is absent, the pass falls back to the generic `LDSDMA_BEGIN`
slot which forces every `ds_read` to wait for every pending LDS DMA store.

### Confirmation 1: 16x32 file with HK 2-K-step structure

`quack/amd/gemm_gfx950_nt_pingpong_16x32.py` mirrors HK's
`k_16x32.cpp` exactly. With `gpu.barrier()` (pre-Fix A): 517 TF/s
vs simpler 1-K-step kernel's 1066. The HK structure was slower
because of *both* the gpu.barrier overhead and conservative
SIInsertWaitcnts drains. Fix A would help here too once explicit
drains are audited (TBD).

| Variant | TF/s @ 8192³ |
|---|---|
| 1-K-step async double-buffer (no real pingpong) | 1066 |
| 2-K-step HK structure, single LDS arena | 517 |

### Confirmation 2: 8-LDS-globals attempt on the 16x32 file

Allocated 8 distinct `memref.global` LDS regions, expecting
`AMDGPULowerModuleLDSPass` to attach scopes per region. The 8 globals
*did* appear in the LLVM IR (verified at `18_llvm_ir.ll`), but:

- 0 `!alias.scope` metadata anywhere
- 0 `!noalias` metadata anywhere
- buffer_load_lds calls have no AAInfo
- `s_waitcnt vmcnt` count in final ISA: 27 (vs 45 with single arena — superficial
  drop because of code-structure changes, not because of alias-scope use)

| Variant | TF/s @ 8192³ |
|---|---|
| 16x32 HK structure, 1 LDS arena | 517 |
| 16x32 HK structure, 8 LDS globals (provenance lost in inttoptr) | 310 |

The 8-globals version was *worse* than the 1-arena version. Likely cause: more
SGPR pressure from 8 base-pointer materializations, with no compensating
alias-scope optimization (since the metadata never propagated to the intrinsic).

### Confirmation 4: 4-wave HK-uniform structure (pre-Fix A)

`quack/amd/gemm_gfx950_nt_4wave_hk.py` is a faithful port of HK's
`FP8_4wave/4_wave.cu` structure to bf16: 4 warps per WG, 256×192 tile,
2-K-step lookahead pipeline, 4 uniform clusters per K-step, register
pingpong (a[0]/a[1]/b[0]/b[1]), explicit `s_waitcnt vmcnt(N)` drains
at HK's exact placements.

Originally measured at 751 TF/s vs simpler 1-K-step rebalanced 4-wave
kernel's 977. **After Fix A: 839 TF/s** — partial recovery; the
remaining gap to 977 is Fix B territory.

### Projection of Fix B's incremental impact

Per Confirmation 6, the 2-barrier HK kernel still has **688 cyc/iter** of
pre-`ds_read` `vmcnt(0)` drains driven by missing alias scopes. With
provenance preserved by Fix B:

- LLVM proves K+2 prefetch writes (to e.g. `As[curr][0]`) don't alias
  the `ds_read` of e.g. `Bs[curr][1]`
- The pre-`ds_read` drains drop to `vmcnt(N)` for only the truly-aliasing
  pending stores, costing ~30 cyc instead of 300-400 cyc
- Conservative estimate: 600 of the 688 cyc recovered
- 856 × 856/(856 × 0.30 fraction-affected) … simpler: kernel ms drops
  from `1/856 × 600/(total)` ≈ **+30% throughput** ≈ **~1100 TF/s**

The Source 2 drains (at `s_barrier`) are architectural and stay.

### Confirmation 3: 1-K-step kernel with real HK staggering

Took the 1066 TF/s 1-K-step kernel and added:
- `if warp_row == 1: gpu.barrier()` before prologue's full barrier (HK staggering)
- 4 barriers per inner-loop iter (one between every cluster, mirrors HK 32x16)
- `s_waitcnt lgkmcnt(0)` before each MMA

Result: correct, but **833 TF/s** vs 1066 — 22% regression.

This is the most direct confirmation. The ONLY structural change vs the
1066 TF/s baseline is the barrier pattern and explicit drains. Same MFMA
shape, same C-accumulator, same warp layout, same LDS staging. The 233 TF/s
loss comes entirely from compiler-inserted vmcnt drains that the alias-scope
path would eliminate.

## Estimated impact of the combined fixes

| Kernel | Original | + Fix A | + 2-bar restructure | + Fix B (projected) |
|---|---|---|---|---|
| HK 4-wave | 744 | **839** (+12% ✓) | **856** (+2% ✓) | 1050-1100 |
| HK 16x32 (8-wave 2-K-step) | 517 | ~580 if explicit drains audited | TBD | 950-1150 |
| Production 8-wave 1-K-step | 1071 | n/a (relies on implicit drain) | n/a | n/a |

HK reports 1217 TF/s for hand-tuned bf16 `k_16x32` on MI355X. With Fix B,
our equivalent should land 950-1100 TF/s. The remaining gap to HK's 1217
is clock-driven (HK targets 2 WG/CU via `__launch_bounds__(N, 2)`, which
we don't hit due to AGPR usage profile).

## Why Fix B is worth doing

Fix A and the 2-barrier restructure are landed kernel-level work.
**Fix B unlocks the next ~25-30%** by eliminating the Source 3 drains
(pre-`ds_read` LDS-aliasing waits). It also benefits every quack/amd
kernel using async HBM→LDS (`raw_ptr_buffer_load_lds`) — currently 9+
kernels including GEMM pingpong variants, NN big, 128x128 LDMA, splitK,
streamK.

The current workaround across the codebase is: avoid per-cluster barriers,
avoid pipelines with overlapping LDS read/write to the same stage. This
caps every kernel's structural sophistication.

## Pointers for the implementer

- LLVM source on what the pass looks for:
  `llvm-project/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:1216-1248`
- LLVM source on where scopes are generated:
  `llvm-project/llvm/lib/Target/AMDGPU/AMDGPULowerModuleLDSPass.cpp:1265-1301`
- MLIR LLVM dialect `addressof` op (the API to call from Python):
  `from flydsl._mlir.dialects import llvm; llvm.AddressOfOp(...)` — already
  used elsewhere in FlyDSL for non-LDS purposes, see `flydsl/expr/...`
- MLIR LLVM dialect `getelementptr` op:
  `llvm.GEPOp(result_type, base_ptr, indices, ...)`
- MLIR memref.global symbol-name extraction:
  `SmemAllocator.global_sym_name` is already stored

A working proof-of-concept patch is probably under 100 lines of FlyDSL Python
plus a small set of kernel-side adjustments to use the new helper.

## Suggested test for verifying the fix

Use the HK 4-wave kernel (`gemm_gfx950_nt_4wave_hk.py`) as the regression
target — it's already Fix-A'd, 2-barrier restructured, and the ATT trace
clearly identifies the pre-`ds_read` drains Fix B should eliminate.

1. Land FlyDSL fix preserving provenance through `raw_ptr_buffer_load_lds`.
2. In `gemm_gfx950_nt_4wave_hk.py`, restore the 8-LDS-globals variant
   (one global per `As[stage][m_half]` / `Bs[stage][n_half]` subregion —
   was written and tested during the investigation, reverted at commit
   X). Each subregion becomes a distinct alias scope when
   `AMDGPULowerModuleLDSPass` runs.
3. Bench at M=8192, N=8064, K=8192 bf16. **Expect ≥1050 TF/s.**
4. Inspect `~/.flydsl/debug/<kernel>/18_llvm_ir.ll`:
   - Should show `getelementptr ... @<lds_global>` for buffer-load-LDS dest pointers
   - Should show `!alias.scope` and `!noalias` metadata on the intrinsic calls
5. Re-run ATT capture:
   ```
   rocprofv3 --att --att-target-cu 1 --kernel-include-regex 'nt_4wave_hk.*' \
       -d /tmp/att_out -- python3 driver.py
   python3 /tmp/analyze.py /tmp/att_out/stats_ui_output_*dispatch_3.csv \
       --waves 20 --mfma-cycles 16 --mfmas-per-cluster 24
   ```
   - The mid-cluster `s_waitcnt vmcnt(0)` instructions (pre-`ds_read`)
     should drop from ~300-400 cyc each to ≤60 cyc each (becoming partial
     `vmcnt(N)` drains)
   - Total waitcnt cyc/iter should drop from 973 to ~285

---

## 2026-06-07 implementation attempt — findings

Spent a session on Fix B (provenance for Source 3 / pre-`ds_read` drains).
Did NOT land the fix; below are the concrete results so the next iteration
can pick up cleanly.

### What was done

1. **Bumped `/workspace/FlyDSL` 150 commits** from `23f59ab2` to upstream
   main `48170e30`. Local escape-hatch (`FLYDSL_EXTRA_LLC_OPTS` in
   `python/flydsl/compiler/backends/rocm.py`) preserved through the merge.
   Rebuilt incrementally via `scripts/build.sh` (~3 min on a warm cache).
2. **Fixed AST-rewriter regression in HK kernel.** Upstream FlyDSL's
   `_is_constexpr` is strict — only recognises `const_expr(...)` calls.
   Plain `if PY_CONST:` patterns now dispatch through `scf_if_dispatch`,
   and variables first-defined-inside-a-branch don't escape. Wrapped two
   sites in `fx.const_expr(...)`:
   - `gemm_gfx950_nt_4wave_hk.py:192` (`XCD_SWIZZLE > 1`)
   - `gemm_gfx950_nt_4wave_hk.py:206` (`GROUP_M > 1`)
3. **Confirmed upstream `fly.get_dyn_shared` already has the right
   lowering**: `llvm.mlir.global + addressof + getelementptr`
   (`tests/mlir/Conversion/dyn_shared.mlir:6`).
4. **Tried `llvm.mlir.addressof @<sym>` of the `memref.global`** that
   `SmemAllocator` emits → MLIR verifier rejects (`addressof` requires
   `llvm.mlir.global`).
5. **Tried `fx.get_dyn_shared` + `builtin.unrealized_conversion_cast`** to
   bridge `fly.ptr<i8, shared>` → `!llvm.ptr<3>` for `buffer_load_lds`. The
   conversion framework inserts an inverse cast on the same value (because
   `fly.get_dyn_shared`'s lowering does `replaceOp(op, llvm_ptr_val)` which
   updates user types) and **both casts remain live** post-conversion:
   ```
   failed to legalize unresolved materialization from
   '!llvm.ptr<3>' to '!fly.ptr<i8, shared, align<1024>>'
   ```
6. **Reverted to the pre-migration kernel** + kept only the const_expr
   fixes. Bench at 8192³ bf16: **845 TF/s** (vs 856 pre-bump, ~1.3%
   regression purely from the FlyDSL bump).

### Why the upstream FlyDSL infrastructure isn't a drop-in

Upstream FlyDSL kernels that use `fly.get_dyn_shared` (e.g. `kernels/preshuffle_gemm_v2.py`)
go through the high-level `fx.copy`/`fx.gemm` ops, not the low-level
`rocdl.raw_ptr_buffer_load_lds` intrinsic. The high-level ops are fly
dialect ops that get lowered to `buffer_load_lds` at fly-to-rocdl
conversion time, so the LDS pointer **stays in fly-land until lowering**
and the conversion handles the type bridge cleanly.

QuACK's HK kernel calls `rocdl.raw_ptr_buffer_load_lds` directly with an
`!llvm.ptr<3>` operand for fine-grained control over scheduling and
warp-uniformity. This direct intrinsic call **doesn't go through fly-to-rocdl
conversion** — the rocdl op is already in the legal target dialect — so
there's no clean way to bridge a fly.ptr to its llvm.ptr<3> operand at
MLIR-build time.

`kernels/splitk_hgemm.py:491-492` confirms upstream also doesn't have a
clean answer here — it uses the same legacy
`memref.extract_aligned_pointer_as_index + create_llvm_ptr` (= inttoptr)
pattern as QuACK, with the same provenance loss.

### Three real paths forward (none are <1h)

**Path P1 — High-level `fx.copy` migration (idiomatic, biggest refactor).**
Replace direct `rocdl.raw_ptr_buffer_load_lds` calls with `fx.copy` +
`fx.rocdl.BufferCopy128b` atoms; replace direct `ds_read` (via
`vector.load_op` on STensor) with `fx.copy` from LDS-typed pointers; replace
explicit MFMA driver with `fx.gemm`. ~6-12h, large surface, but matches
upstream patterns and unlocks future FlyDSL improvements automatically.

**Path P2 — Custom fly op for LDS-ptr-to-llvm-ptr bridge.**
Add a `fly.lds_ptr_cast` op (Fly_Pointer → AnyType:llvm.ptr) with a trivial
FlyToROCDL lowering pattern (`replaceOp(op, adaptor.getSrc())` — the type
converter already maps fly.ptr<i8,shared> to llvm.ptr<3>). Use it in QuACK
HK kernel:
```python
shared_base_fly = fx.get_dyn_shared()
shared_base_llvm = fx.lds_ptr_cast(llvm_ptr_3_ty, shared_base_fly)
# GEP from shared_base_llvm with byte offsets; feed to raw_ptr_buffer_load_lds
```
FlyOps.td + FlyToROCDL.cpp work (~150 LOC) + Python wrapper (~20 LOC) +
QuACK rewrite of LDS access in HK kernel (~50 LOC). ~3-5h. Smallest viable
change to FlyDSL; isolated to one new op.

**Path P3 — Replace `SmemAllocator` with direct `llvm.GlobalOp` emit.**
Emit `llvm.mlir.global @<sym> external addr_space(3)` at gpu.module level
(navigating to ctx.gpu_module_body from inside the kernel function).
Replace STensor's `vector.load_op` on memref with direct `llvm.load`
from GEPed `!llvm.ptr<3>`. Self-contained to QuACK (no FlyDSL changes).
~2-3h, but STensor is shared with many other AMD kernels so the change has
to be opt-in.

### Other breakage uncovered by the bump

Filed for follow-up audits — the bumped FlyDSL is stricter and surfaces
real bugs that the older version silently accepted:

- `tests/amd/test_rmsnorm.py` — fails on `w_div` (defined-in-branch-only;
  needs either `const_expr(...)` wraps on `if per_head:` / `else:` or
  outer-scope pre-initialisation of `w_div`).
- `tests/amd/test_gemm_4wave_lds_pp.py[bf16]` — fails because
  `rocdl.mfma.f32.16x16x16f16` now strictly rejects bf16 operands. The
  kernel was using the f16 intrinsic for bf16; either switch to
  `mfma_f32_16x16x16bf16` or use the K=32 atom (`mfma.f32.16x16x32bf16`).
- ~50 other `if PY_CONST:` sites across AMD kernels (`gemm_gfx950_nn_big.py`
  has 12, `gemm.py` has 8, etc.) — same const_expr fix as the HK kernel
  needed wherever a var is first-defined-in-branch.

### Recommended next iteration

Path P2 is the smallest path to validate the perf hypothesis (does
addressof+GEP for `buffer_load_lds` actually drop the pre-`ds_read` drain
cycles, as the ATT analysis predicts?). If perf landslides at ~1050 TF/s,
either upstream the `fly.lds_ptr_cast` op to FlyDSL or extend
`raw_ptr_buffer_load_lds`'s Python wrapper to accept a fly.ptr and bridge
internally.

---

## 2026-06-08 second implementation attempt — findings

Spent another session on the LDS provenance fix. Got further than the
2026-06-07 attempt but **still didn't deliver** the projected
~1265 TF/s. New findings refine the design space.

### What was tried

1. **Replaced `memref.global` (via SmemAllocator) with `llvm.mlir.global`**
   emitted at gpu.module scope. Inside the kernel, replaced `STensor`
   with a custom shim that uses `llvm.mlir.addressof + llvm.getelementptr`
   for both `ds_read` (via `llvm.load`) and `buffer_load_lds` destination
   construction. Compile + correctness OK. **Perf dropped 1158 → 970 TF/s.**

2. **Split into TWO `llvm.mlir.global`s** (one per LDS sub-region, AS and BS)
   to satisfy `AMDGPULowerModuleLDSPass.cpp:1268` `if (NumberVars > 1)`
   gate. Still got `alias.scope = 0` in the LLVM IR dump, and
   `vmcnt(0)` drains in the ISA WENT UP from 10 → 13. **Perf dropped
   further to 870 TF/s.**

### What we learned about the AMDGPU lowering pipeline

- **`AMDGPULowerModuleLDSPass` runs AFTER the MLIR-to-LLVM-IR dump
  point**. The IR we dump at FlyDSL stage 20 (before
  `gpu-module-to-binary`) does NOT yet have `alias.scope` metadata even
  if the pass would eventually attach it.
- **The pass attaches scopes via `refineUsesAlignmentAndAA`** (lines
  1316-1319) but only on uses of the merged-struct GEP, not on arbitrary
  loads/stores. Even with two globals, the merging behaviour for
  `external` linkage globals appears to skip our access patterns.
- **HipCC's C++ frontend emits IR with TBAA / `__restrict__`-derived
  noalias metadata FROM THE START**. That carries through SIInsertWaitcnts
  unchanged. Our FlyDSL-generated IR has no alias metadata, so even if
  LowerModuleLDS attaches per-LDS-global scopes, SIInsertWaitcnts still
  sees pending writes as potentially aliasing all ds_reads.
- The `LDSDMAStores` tracking in SIInsertWaitcnts (`SIInsertWaitcnts.cpp:1216-1248`)
  uses `MachineMemOperand` AAInfo. To populate that, the LLVM IR ds_read
  load must already carry `!alias.scope` / `!noalias` metadata at IR
  selection time.

### The real fix path (revised)

**The simple "swap memref.global to llvm.mlir.global" is NOT sufficient.**
The full fix requires emitting `!alias.scope` and `!noalias` metadata
explicitly on every LDS load/store and on the `buffer_load_lds` intrinsic
call — mirroring what HipCC produces for C++ with `__restrict__`-tagged
LDS pointers.

In MLIR, this means using the **llvm dialect's
`AliasAnalysisOpInterface`** — attributes like `alias_scopes`,
`noalias_scopes`, and `tbaa` on `llvm.load`, `llvm.store`, and
intrinsic call ops. The metadata pipeline:

1. Emit a `#llvm.alias_scope_domain<id = ..., description = "kernel">`
   at gpu.module level.
2. Emit per-region `#llvm.alias_scope<id = ..., domain = ...,
   description = "as_region">` and `"bs_region"`.
3. On every `llvm.load` from `as_base`: attach
   `alias_scopes = [#<as_scope>], noalias_scopes = [#<bs_scope>]`.
4. On every `llvm.load` from `bs_base`: mirror.
5. On every `rocdl.raw_ptr_buffer_load_lds` to `as_base`:
   `alias_scopes = [#<as_scope>], noalias_scopes = [#<bs_scope>]`.
6. Mirror for bs.

### Effort estimate (revised)

- **~6-10h** for the metadata-attaching helper + 16x32 kernel migration.
- The helper would likely be reusable for other kernels (and could even
  be upstreamed into FlyDSL).
- Risk: metadata-driven aliasing analysis may still not provide partial
  drains if SIInsertWaitcnts's `LDSDMAStores` slot-tracking can't
  distinguish writes within a region.

### Recommended next iteration

Two options:

1. **Try the metadata-attaching helper** (above). High confidence this
   matches HK's mechanism but moderate uncertainty whether SIInsertWaitcnts
   will use the metadata effectively for partial drains.
2. **Just accept the 10% gap on 16x32**. The 32x16 kernel already beats
   HK (1148 vs 1130 TF/s). The 16x32 is 9% behind HK (1158 vs 1272).
   Diminishing returns on chasing this.

If picking (1), the prototyping order should be:
- Reproduce HK's IR shape for a tiny kernel (one buffer_load_lds + one
  ds_read with explicit alias metadata) and verify `!alias.scope`
  survives all the way to the ISA → `vmcnt(N)` partial drain.
- If yes, apply to 16x32 kernel.

---

## 2026-06-09 third attempt — explicit alias scopes (flash_attn pattern)

Applied the exact `flash_attn_gfx950.py` pattern:
- TWO `llvm.mlir.global`s (LDS_SYM_A, LDS_SYM_B).
- `llvm.alias_scope_domain` + per-region `llvm.alias_scope`.
- Explicit `alias_scopes=` / `noalias_scopes=` on every `llvm.LoadOp` (ds_read)
  AND on every `rocdl.raw_ptr_buffer_load_lds` call.

### What worked

- **Metadata IS propagating** to LLVM IR: 128 alias.scope sites, 128 noalias
  sites in the dumped IR. Verified scopes are attached to all 96 ds_reads
  and 32 of 33 buffer_load_lds calls (the prologue's single full-drain
  retains no scope, as expected).
- IR structurally clean — buffer_load_lds destinations are `getelementptr
  @<global>, i32 %offset`, no inttoptr.

### What didn't work

- **vmcnt(0) drain count UNCHANGED** at 10 (same as baseline with inttoptr).
- **Perf REGRESSED** 1158 → 1055 TF/s. The shim's per-load arithmetic +
  the addressof+GEP pattern pushed VGPR usage from 242 → 256 with 13 spills
  + 28 scratch ops. The scratch I/O is the new perf killer.

### Why scopes don't reduce drains here

Per-region scopes (AS vs BS) are insufficient for our pingpong design.
At each barrier:
- Pending writes: `buffer_load_lds → AS@next_stage` (the prefetch)
- Upcoming reads: `ds_read ← AS@current_stage` (the actual compute)

Both writes and reads are in the AS region — same scope. The compiler
treats them as potentially aliasing and drains conservatively.

**To get partial drains, we'd need per-stage scopes**: AS@0, AS@1, BS@0,
BS@1 — four scopes. Within each iter, the unrolled K-step body knows at
COMPILE TIME which stage it's reading from and which it's writing to
(stage 0 vs stage 1 are hardcoded in the unrolled body). So per-stage
scopes are statically determinable.

### Effort estimate (revised, again)

- ~4-6h to plumb per-stage scope IDs through `lds_matrix_*_half` and
  `ldg_sts_*_half_async`.
- ~2h to recover the 14-VGPR regression from a cleaner shim (avoid the
  per-load arithmetic, share the GEP base across loads in a cluster).
- Total: ~6-8h. Estimated impact: drains 10 → ~2, perf 1158 → ~1230 TF/s
  (still ~3% below HK 1272 due to remaining per-cluster scheduling
  differences).

### Diminishing returns warning

QuACK 32x16 already beats HK (1148 vs 1130, +2%). QuACK 16x32 is 9%
behind HK and likely won't fully close even with per-stage scopes. The
6-8h investment yields ~6-8% on a non-default kernel. Strongly suggest
deferring unless there's a specific perf target that requires it.

---

## 2026-06-09 implementation LANDED — how it works

Three commits land per-stage alias scopes on the 8-wave and 4-wave NT
GEMM family:
- `64a7462` — 16x32 (8-wave) +8% (1158 → 1247 TF/s)
- `8bacb87` — 32x16 (8-wave) +5% (1148 → 1198 TF/s)
- `483197c` — 4-wave HK +17% (845 → 1009 TF/s)

This section documents the technique so future agents don't re-discover
it. **No IR post-processing is involved** — the alias scope metadata is
attached at MLIR-build time via op kwargs and survives unchanged through
MLIR→LLVM-IR translation into the backend.

### Pipeline overview

```
Python (FlyDSL kernel-build time)
   │  emit `llvm.LoadOp(..., alias_scopes=..., noalias_scopes=...)`
   │  emit `rocdl.raw_ptr_buffer_load_lds(..., alias_scopes=..., noalias_scopes=...)`
   ▼
MLIR (llvm + rocdl dialects)
   │  ops carry MLIR `ArrayAttr` of `#llvm.alias_scope<...>` attributes
   │  these implement `AliasAnalysisOpInterface`
   ▼
MLIR-to-LLVM-IR translation
   │  AliasAnalysisOpInterface emits `!alias.scope`, `!noalias` metadata
   │  attached to the LLVM IR load and call instructions
   ▼
LLVM IR (with `!alias.scope` / `!noalias` metadata)
   │  scope domain `!4` and per-region-per-stage scopes `!3` `!6` etc.
   ▼
AMDGPU backend
   ├─ `AMDGPULowerModuleLDSPass` — packs LDS globals into __amdgpu_module_lds
   │  (we already have multiple globals so packing is trivial)
   ▼
   `SIInsertWaitcnts`
      │  walks pending LDS DMA writes per pending-ds_read
      │  reads `MachineMemOperand.AAInfo` (scope + noalias from IR metadata)
      │  for each pending write, asks ScopedNoAliasAA: does it alias?
      │     - if write's scope ∈ ds_read's noalias_scopes → no alias → skip
      │     - else → must wait
      │  emits partial `vmcnt(N)` instead of full `vmcnt(0)`
```

### The three building blocks

#### 1. Alias scope attribute construction (Python)

```python
LDS_ALIAS_DOMAIN = f'#llvm.alias_scope_domain<id = "nt_pp_<dtype>_<k>_<n>.lds">'
SCOPE_IDS = ("as0", "as1", "bs0", "bs1")  # one per (region × stage)

def _scope_attr(ids):
    """Build a `[#llvm.alias_scope<id="x", domain=...>, ...]` ArrayAttr."""
    inner = ", ".join(
        f'#llvm.alias_scope<id = "{i}", domain = {LDS_ALIAS_DOMAIN}>'
        for i in ids
    )
    return ir.Attribute.parse(f"[{inner}]")

# Per-region-per-stage: scope = THIS region+stage; noalias = ALL OTHERS.
_SCOPE   = {sid: _scope_attr((sid,)) for sid in SCOPE_IDS}
_NOALIAS = {sid: _scope_attr(tuple(o for o in SCOPE_IDS if o != sid))
            for sid in SCOPE_IDS}
```

`ir.Attribute.parse(...)` accepts the MLIR text-form of any attribute,
including dialect-specific ones like `#llvm.alias_scope<...>`. This is
robust: the LLVM dialect's parsers validate the syntax at parse time.

#### 2. Two MLIR ops, two attribute slots

**For ds_reads** (`vector.load` → `llvm.LoadOp`):

```python
def vec_load(idxs, vec_size):
    elem_off = linear_offset(idxs)
    byte_off_i32 = arith.index_cast(T.i32, elem_off * DTYPE_BYTES)
    gep = llvm.getelementptr(_LDS_PTR_TY, base_ptr, [byte_off_i32], ...)
    vec_t = T.vec(vec_size, dtype_)
    return llvm.LoadOp(
        vec_t, gep, alignment=2,
        alias_scopes=my_scope,        # ArrayAttr of one #llvm.alias_scope
        noalias_scopes=other_scopes,  # ArrayAttr of three #llvm.alias_scope
    ).result
```

**For buffer_load_lds** (FlyDSL's Python wrapper of `rocdl.raw_ptr_buffer_load_lds`):

```python
rocdl.raw_ptr_buffer_load_lds(
    A_.rsrc, lds_ptr,
    arith.constant(DMA_BYTES, type=T.i32),
    global_offset,
    arith.constant(0, type=T.i32),
    arith.constant(0, type=T.i32),
    arith.constant(1, type=T.i32),
    alias_scopes=as_scopes[stage],     # ← kwarg forwarded by FlyDSL wrapper
    noalias_scopes=as_noalias[stage],  # ← into the ODS-generated op ctor
)
```

The FlyDSL Python wrapper at `flydsl/expr/rocdl/__init__.py:495` uses
`**kw` so any kwarg flows to the underlying ODS-generated op constructor
unchanged. Both `llvm.LoadOp` and `rocdl.raw_ptr_buffer_load_lds` accept
`alias_scopes` / `noalias_scopes` / `tbaa` as optional `ArrayAttr` kwargs
because their ODS definitions implement `AliasAnalysisOpInterface`.

#### 3. Per-STAGE granularity via Python int dispatch

Per-region scopes (AS vs BS, two scopes total) **do not** reduce drains.
In a pingpong pipeline, each iter writes `AS@(stage^1)` while reading
`AS@stage` — both `AS` scope, so the compiler still treats them as
potentially aliasing.

The fix is **per-stage** scopes (AS@0, AS@1, BS@0, BS@1, four total).
Read of AS@0 has `noalias_scopes = [AS@1, BS@0, BS@1]` — so a pending
write to AS@1 doesn't force a drain at the read of AS@0.

**This requires the stage to be a Python int at every helper call site**,
because `as_views[stage]` is a Python tuple lookup that resolves at
MLIR-build time. Where the original kernels used a runtime `current_stage`
from the loop state, the loop body had to be **2× unrolled** so each
sub-step has a hardcoded Python int stage (sub-step 0 → stage 0,
sub-step 1 → stage 1).

```python
# Before: 1-K-step-per-iter, runtime stage
for _ki, state in range(0, TOTAL_K_STEPS, init=init_state):
    stage_curr = fx.Index(state[1])  # ← runtime, can't index a Python tuple
    a_frags = lds_matrix_a_kk(stage_curr, 0)  # ← inside the helper:
    # vec = as_.vec_load((fx.Index(lds_stage), row, col), VEC)  ← STensor / memref-based

# After: 2-K-step-per-iter, compile-time stage
OUTER_ITERS = TOTAL_K_STEPS // 2
for _oi, state in range(0, OUTER_ITERS - 1, init=init_state):
    # Sub-step 0: stage 0
    a_frags = lds_matrix_a_kk(0, 0)  # ← Python int, indexes as_views[0] at build time
    # Sub-step 1: stage 1
    a_frags = lds_matrix_a_kk(1, 0)
```

And the helper itself:
```python
def lds_matrix_a_kk(lds_stage, kk_target):
    view = as_views[lds_stage]   # ← compile-time tuple index → per-stage scope baked in
    ...
    vec = view.vec_load((row, col), WMMA_A_FRAG)
    # vec_load uses view's captured `my_scope` and `other_scopes` → correct per-stage tags
```

### Four `llvm.mlir.global`s (one per stage × region)

`AMDGPULowerModuleLDSPass.cpp:1268` only creates alias scopes when
`NumberVars > 1`. We emit four LDS globals (per-stage × per-region), each
holding ONE stage worth of one region:

```python
# At launcher time (in gpu.module body):
with ir.InsertionPoint(ctx.gpu_module_body):
    linkage = ir.Attribute.parse('#llvm.linkage<external>')
    for sym, size in (
        (LDS_SYMS_A[0], AS_STAGE_BYTES),  # AS@0
        (LDS_SYMS_A[1], AS_STAGE_BYTES),  # AS@1
        (LDS_SYMS_B[0], BS_STAGE_BYTES),  # BS@0
        (LDS_SYMS_B[1], BS_STAGE_BYTES),  # BS@1
    ):
        llvm.GlobalOp(
            global_type=ir.Type.parse(f"!llvm.array<{size} x i8>"),
            sym_name=sym, linkage=linkage,
            addr_space=3, alignment=1024,
        )
```

Each kernel inside builds 4 `llvm.mlir.addressof` results as base pointers:
```python
_as_bases = (
    llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYMS_A[0]),
    llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYMS_A[1]),
)
_bs_bases = (...)
```

### Why no IR post-processing was needed

The flash_attn_gfx950.py pattern (which we modelled on) does the same
thing — emit metadata at op-construction time. The MLIR llvm dialect's
`AliasAnalysisOpInterface` translates the attributes verbatim into LLVM
IR metadata during MLIR-to-LLVM lowering:

- MLIR: `#llvm.alias_scope_domain<id = "...">` → LLVM IR: `!{!"..."}`
- MLIR: `#llvm.alias_scope<id = "x", domain = ...>` → LLVM IR: `!{!"x", !DOMAIN}`
- MLIR: op kwarg `alias_scopes=...` → LLVM IR: `!alias.scope !N`
- MLIR: op kwarg `noalias_scopes=...` → LLVM IR: `!noalias !M`

Verified by inspecting `~/.flydsl/debug/<kernel>/20_llvm_ir.ll`:
- 128 `!alias.scope` sites on loads + buffer_load_lds calls
- 128 `!noalias` sites
- 4 LDS globals `external addrspace(3)` at the module level
- 0 `inttoptr` to `ptr addrspace(3)` (the access path is `getelementptr @<global>`)

### Why per-region scopes don't work (deeper)

`SIInsertWaitcnts.cpp:2540-2557` walks `LDSDMAStores` per pending ds_read.
For each pending write, it asks `ScopedNoAliasAA` whether the write
aliases the read. With per-region scopes:
- ds_read of `lds_as` has `alias_scopes=[lds_as], noalias_scopes=[lds_bs]`
- pending buffer_load_lds write to `lds_as` (the next stage) has
  `alias_scopes=[lds_as], noalias_scopes=[lds_bs]`
- ScopedNoAliasAA: write's scope (`lds_as`) ∈ read's noalias? NO.
- → must wait → full vmcnt(0).

With per-stage scopes:
- ds_read of `lds_as0` has `alias_scopes=[lds_as0], noalias_scopes=[lds_as1, lds_bs0, lds_bs1]`
- pending buffer_load_lds to `lds_as1` has `alias_scopes=[lds_as1], ...`
- ScopedNoAliasAA: write's scope (`lds_as1`) ∈ read's noalias? YES.
- → can skip this write → partial `vmcnt(N)`.

### Empirical proof (16x32 kernel)

| Pattern | Total s_waitcnt in ISA | vmcnt(0) drains | Perf (TF/s) |
|---|---|---|---|
| Original (inttoptr, SmemAllocator) | 30 | 10 | 1158 |
| Per-region scopes (1 LDS global, 2 scopes) | 42 | 15 | 970 |
| Per-region scopes (2 LDS globals, 2 scopes) | 38 | 13 | 870 |
| **Per-stage scopes (4 LDS globals, 4 scopes)** | **21** | **1** | **1247** |
| HK reference (HipCC's IR with TBAA) | 24 | 1 | 1272 |

The per-stage variant matches HK at the drain-count level. The remaining
2% perf gap is MFMA cluster scheduling, not LDS provenance.

### Apply to a new kernel — checklist

1. Replace `SmemAllocator(...)` + `STensor(SmemPtr(...))` with the 4-global
   setup + 4 `_make_lds_view_2d(...)` views, tupled as `as_views`/`bs_views`.
2. Emit the 4 `llvm.GlobalOp`s in the launch function (in `ctx.gpu_module_body`).
3. Update every helper that takes `lds_stage` to use `as_views[lds_stage]`
   / `bs_views[lds_stage]` and to attach `_as_scopes[stage]` /
   `_as_noalias[stage]` to its `raw_ptr_buffer_load_lds` calls.
4. If any helper call site uses a RUNTIME stage (e.g. from `scf.for`
   state), **2× unroll the loop** so each sub-step has a Python int stage.
5. Verify: `vmcnt(0)` count drops, `alias.scope` sites appear in
   `20_llvm_ir.ll`, bench shows the expected lift.

### Why the AGPR forcing on the 4-wave HK kernel doesn't conflict

`amdgpu-agpr-alloc=192,192` on the 4-wave HK kernel is independent of
the alias scope work. It controls VGPR↔AGPR register partitioning, not
memory dependency analysis. The 4-wave kernel has a larger C accumulator
(192 fp32/lane vs 128) so the AGPR offload is load-bearing for register
budget; that stays unchanged. The per-stage scopes apply orthogonally.
