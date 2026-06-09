# NT pingpong (8-wave) kernel walkthrough — FlyDSL features in depth

This is a top-to-bottom read of `quack/amd/gemm_gfx950_nt_pingpong.py` aimed at a reader who knows GPU kernels and MLIR-adjacent ideas but is new to FlyDSL. The kernel itself is described in its own docstring and in the [`port-hipkittens-to-flydsl` skill](../../.claude/skills/port-hipkittens-to-flydsl/SKILL.md); here I focus on **how FlyDSL works**, especially the compiler-side features the kernel leans on.

I cite source by symbol or comment anchor rather than line number where I can — the file evolves but the symbols don't.

---

## 0. Scope and posture

What this doc covers:
- The Python → MLIR mental model FlyDSL uses
- Compile-time vs runtime values (the single most useful concept to internalize)
- The two decorators (`@flyc.kernel`, `@flyc.jit`)
- Three flavors of loops, three flavors of `if`
- Direct MLIR / LLVM-dialect emission (LDS globals, alias scopes, GEPs)
- ROCDL intrinsics — what they are and how to read them
- The MLIR type system surface (`T.*` factories vs `fx.*` runtime values)
- Occupancy / register-allocation attributes

What this doc does NOT cover:
- How FlyDSL's AST rewriter is implemented (I haven't read the rewriter sources end-to-end)
- The full lowering pipeline from `@flyc.kernel` body to AMDGPU ISA
- Why specific tile sizes / barrier patterns were chosen — see the kernel docstring and the HK reference

Where I'm uncertain I say so explicitly in §17.

---

## 1. Sixty-second mental model

When you write `@flyc.kernel` over a Python function, FlyDSL **does not run that function as Python at runtime**. It walks the AST, rewrites it into MLIR operations (in the `arith`, `scf`, `vector`, `llvm`, `rocdl`, `gpu`, and `memref` dialects), then lowers the resulting MLIR module through to AMDGPU LLVM IR and finally to a GPU object file. The Python body executes **once, at compile time, in a special context** where:
- Python variables that hold MLIR `Value`s build the IR by side effect — calling `arith.constant(...)` doesn't return an `int`, it returns an SSA value that will exist in the compiled kernel.
- Python control flow (`if`, `for`) over **compile-time** values stays in Python and unrolls / specializes the IR.
- Python control flow over **runtime** values (MLIR `Value`s) is rewritten into `scf.if` / `scf.for` by the AST pass.

Hold this model in your head — everything below is a refinement.

---

## 2. The two decorators

Lines `nt_kernel = @flyc.kernel(...)` and `launch_nt_kernel = @flyc.jit` near the bottom of `_compile_nt_pingpong_kernel`.

### `@flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])`

Defines a single **device-side function** that will become a GPU kernel. The `known_block_size` is a compile-time hint that says "this function will always be launched with this block shape." That fact is bolted into the IR and exposed to later passes — useful when block-size-dependent intrinsics (warp reductions, ds_permute lane targets) need to know there's no runtime variation.

The decorated body executes in the special "tracing" context described above.

### `@flyc.jit`

Defines a **host-side launcher**. Inside `launch_nt_kernel` you can:
- Compute grid dimensions from runtime arguments (`bm = (m + BLOCK_M - 1) // BLOCK_M`).
- Mutate the GPU module's IR directly via `CompilationContext.get_current()` — this is the escape hatch the kernel uses to emit `llvm.mlir.global` ops at module scope (§7).
- Call the `@flyc.kernel` with the host-side arguments to produce a `launcher` object.
- Set LLVM function attributes on the kernel before launching (§14).
- `launcher.launch(grid=..., block=..., stream=...)`.

You always need both: the kernel for "what runs on the device," the launcher for "everything the host side has to set up." In QuACK the `@functools.lru_cache(maxsize=1024)` on `_compile_nt_pingpong_kernel` keeps the launcher cached per `(dtype, K, N)` — recompiling is expensive.

---

## 3. Compile-time vs runtime — the dividing line

This is the concept that unlocks everything else. A value in the kernel body is either:

| Compile-time (Python) | Runtime (MLIR `Value`) |
|---|---|
| Plain `int`, `float`, `tuple`, `list`, `dict` | `fx.Int32(...)`, `fx.Index(...)`, results of `arith.*`, `vector.*`, `rocdl.*`, `llvm.*` |
| `BLOCK_M = 256`, `STAGES = 2`, `DTYPE_BYTES = 2` | `tid = fx.Int32(fx.thread_idx.x)`, `m_offset`, `pid`, `c_frags[i]` |
| Visible to `range_constexpr`, Python `if`/`for`, indexing | Visible to `arith.cmpi`, `scf.for`/`scf.if`, MFMA, LDS load/store |

The kernel uses this distinction load-bearing in three places:

1. **`as_views[lds_stage]`** in `lds_matrix_a_kk` and friends. `lds_stage` is a **Python int** at every call site — that's why the loop is 2× unrolled (§4.2 for the scf.for form; §15 for the call-site rationale). `as_views` is a Python tuple; you can't index it with an MLIR `Value`.
2. **`fx.const_expr(XCD_SWIZZLE > 1)`** in the tile-swizzle block. The condition is a Python bool. Without the `fx.const_expr(...)` wrap the upstream FlyDSL AST rewriter will try to lower the `if` into `scf.if`, fail because the predicate isn't `arith.cmpi`-shaped, and (the dangerous failure mode) variables first-defined inside the branch don't escape.
3. **`range_constexpr(N)`** in the helpers — fully unrolls the loop at compile time so the IR has `WARP_M_STEPS * WARP_N_STEPS = 32` MFMA ops inlined.

Internalize this: **a Python `int` and an `fx.Int32` are not the same kind of thing**, and FlyDSL features each consume one or the other.

---

## 4. Loops — three flavors

### 4.1 `for _ in range_constexpr(N): ...`

Compile-time unroll. `N` must be a Python int known at trace time. The body is reissued `N` times into the IR. Used everywhere in the kernel — `ldg_sts_a_async`, `lds_matrix_a_kk`, `mma_kk`, the writeback loops. After tracing there's no loop in the IR.

### 4.2 `for _oi, state in range(start, stop, init=[...]): ... yield [...]`

The scf.for pattern with **loop-carried state**. See the main loop body. The shape:

```python
init_state = [k_zero] + c_frags
for _oi, state in range(0, OUTER_ITERS - 1, init=init_state):
    k_now = state[0]
    c_frags = list(state[1 : 1 + C_FRAGS_LEN])
    # ... mutate c_frags ...
    results = yield [k_next] + c_frags
```

What's actually happening: the FlyDSL AST rewriter sees this `for` pattern (with `init=`) and emits `scf.for` with iter_args. Each loop iteration carries the state forward as SSA values; `yield` provides the next iteration's state. After the loop, the **same generator-style variable** (`results`) holds the final iteration's output state — that's how the epilogue picks up `c_frags`.

Two practical points:

- `state` is a tuple of MLIR `Value`s. You unpack it positionally — there's no field naming. Order matters.
- If `init=` has length 1, FlyDSL hands back a bare `ArithValue` instead of a list — caught me out the first time, mentioned in `AGENTS.md` under "AMD port → Key gotchas." Not an issue here since `init_state` has C_FRAGS_LEN + 1 entries.

### 4.3 Plain `for i in range(start, stop):` over runtime bounds

Not used in this kernel. The AST rewriter would lower it to `scf.for` without iter_args. The `init=` variant is the form you almost always want for kernels — kernels are basically "an accumulator over a loop."

---

## 5. Conditionals — three flavors

### 5.1 Compile-time: `if fx.const_expr(PY_BOOL):`

See the XCD swizzle block:

```python
if fx.const_expr(XCD_SWIZZLE > 1):
    # ... XCD swizzle math ...
else:
    pid = flat_pid
```

`XCD_SWIZZLE` is a Python int (default 1, passed in as a constant). The `fx.const_expr(...)` wrap tells the FlyDSL AST rewriter "this branch is decided at compile time; pick a side and discard the other." The discarded branch's IR is never emitted.

**Why the wrap is needed**: upstream FlyDSL's AST rewriter checks for `fx.const_expr(...)` calls in `if` conditions; bare `if PY_BOOL:` falls into the runtime-`if` path, gets dispatched through `ReplaceIfWithDispatch.scf_if_dispatch`, and variables first-defined inside the branch don't escape into the outer scope. So `pid = flat_pid` would be undefined after a bare `if XCD_SWIZZLE > 1:` block. Caveat 5 in the porting skill.

### 5.2 Runtime: `if arith.cmpi(...): ...`

```python
if arith.cmpi(arith.CmpIPredicate.eq, warp_row, fx.Int32(1)):
    rocdl.s_barrier()
```

`arith.cmpi` returns an `i1` SSA value (an MLIR `Value`). The AST rewriter recognizes `if <cmpi-result>:` and emits `scf.if` with no results. Inside the then-block, IR is emitted normally — but again, anything assigned only inside the branch won't be visible after.

Critical caveat: the rewriter dispatches every non-`const_expr` `if` through `scf_if_dispatch` (see L713 / L588 of `ast_rewriter.py`). That function constructs `scf.IfOp(cond_i1, ...)` from the unwrapped value, and **`scf.IfOp` requires `i1`**. `arith.cmpi(...)` always returns `i1` and is the safe shape. `arith.andi(x, y)` returns whatever width its operands have — if you happen to feed it two `i1`s it works, but if either operand drifts to a wider integer the construction fails or (in older toolchains) silently miscompiles. **Nest `arith.cmpi`s** instead of reaching for `andi`/`ori` as compound predicates. See §17 for the full source-derived story.

### 5.3 Manual `scf.IfOp` + InsertionPoint

```python
cond_if = scf.IfOp(cond, results_=[], has_else=False)
with ir.InsertionPoint(cond_if.then_block):
    C_.vec_store((m_global, n_global), val_dtype, 1)
    scf.YieldOp([])
```

The writeback uses this form directly. Why? Because the masked store is **inside a `range_constexpr` loop** that fully unrolls — and the masked region only contains a single store. Building `scf.IfOp` by hand is a small win in code clarity (no extra closure indirection) and gives the writeback a known shape for the verifier to chew through.

You can always drop down to this level when the AST rewriter is in the way. The pattern is: construct the op, open an InsertionPoint over its block, emit the body, terminate with `scf.YieldOp([])`. For ops with results you'd pass `results_=[type, ...]` and yield matching values.

---

## 6. The type system surface

There are two distinct namespaces; mixing them is a frequent source of confusion.

### `T.*` — MLIR type factories

`T.i8`, `T.i32`, `T.f32`, `T.vec(N, dtype)` produce **`ir.Type` objects**, not values. You use them when you need to tell MLIR "this op produces a vector of 4 f32s":

```python
acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG, T.f32))
vec_t = T.vec(vec_size, dtype_)
return llvm.LoadOp(vec_t, gep, alignment=2, ...).result
```

### `fx.*` — runtime-value wrappers

`fx.Int32(x)`, `fx.Index(x)`, `fx.thread_idx.x`, `fx.block_idx.x` produce **MLIR `Value`s**. You combine them with `+`, `*`, `//`, `%` and they emit `arith.addi` / `arith.muli` / etc. under the hood. They participate in scf.if/scf.for as runtime values.

```python
tid = fx.Int32(fx.thread_idx.x)
wid = tid // WARP_SIZE         # arith.divsi, since tid is Int32 and WARP_SIZE is i32
m_offset = fx.Index(block_m_idx * BLOCK_M)
```

`fx.Index` is for `index` type (MLIR's native pointer-arithmetic type, distinct from `i32`/`i64`). Most ops on `memref` and `vector` expect `index`; ROCDL intrinsics typically want `i32`. The kernel converts between them with `arith.index_cast(T.i32, ...)`.

---

## 7. Emitting LLVM globals at module scope

This is one of the more "compiler-feeling" features in the kernel. The LDS allocation isn't done through FlyDSL's `SmemAllocator` (the higher-level surface most kernels use); instead, the launcher reaches into the GPU module's IR and emits raw `llvm.mlir.global` ops:

```python
ctx = CompilationContext.get_current()
with ir.InsertionPoint(ctx.gpu_module_body):
    linkage = ir.Attribute.parse('#llvm.linkage<external>')
    for sym, size in (
        (LDS_SYMS_A[0], AS_STAGE_BYTES),
        # ...
    ):
        llvm.GlobalOp(
            global_type=ir.Type.parse(f"!llvm.array<{size} x i8>"),
            sym_name=sym,
            linkage=linkage,
            addr_space=3,
            alignment=1024,
        )
```

What's going on:

1. **`CompilationContext.get_current()`** returns the active compilation context. `gpu_module_body` is the MLIR block that wraps everything inside the `gpu.module` op — the device-side translation unit.
2. **`ir.InsertionPoint(block)`** is a context manager: while it's active, any new MLIR op gets inserted into that block. Outside the `with`, the insertion point returns to its previous location.
3. **`llvm.GlobalOp(...)`** emits an LLVM-dialect global variable declaration. `addr_space=3` is AMDGPU's LDS address space (1 = global / HBM, 2 = constant, 3 = LDS, 4 = constant via SMEM, 5 = scratch). `external` linkage + zero initializer is how LDS allocations are typically declared in AMDGPU LLVM IR.
4. **`alignment=1024`** asks for 1024-byte alignment. The actual allocation is owned by `AMDGPULowerModuleLDSPass`, which packs the globals into the kernel's LDS budget at codegen time.

Why this matters: each `llvm.mlir.global` becomes a separate LDS region with its own provenance — and crucially with its own alias scope when `AMDGPULowerModuleLDSPass` runs. That pass only creates alias-scope metadata when there is more than one LDS variable: `AMDGPULowerModuleLDSPass.cpp:1267-1276` gates the whole scope-creation block on `NumberVars > 1`. With a single global, no scope metadata is attached, and the alias analysis pass conservatively treats every LDS access as potentially aliasing every other. Four globals is well above the threshold and lets the pass tag the loads/stores with both anonymous per-variable scopes and (because we explicitly emitted our own) the named per-stage scopes from §9.

---

## 8. Provenance-preserving LDS pointers

Inside the kernel body, the kernel constructs LDS pointers like this:

```python
_LDS_PTR_TY = ir.Type.parse("!llvm.ptr<3>")
_GEP_DYN = -(2 ** 31)

def _gep_lds(base_ptr, byte_offset_i32):
    return llvm.getelementptr(
        _LDS_PTR_TY, base_ptr, [byte_offset_i32], [_GEP_DYN], _I8_TY, None,
    )

_as_bases = (
    llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYMS_A[0]),
    llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYMS_A[1]),
)
```

Three pieces:

- **`!llvm.ptr<3>`** is MLIR's parseable spelling for "LLVM opaque pointer in address space 3 (LDS)." The opaque-pointer transition means the pointee type isn't part of the pointer type — we say what kind of bytes it points at when we use it (the `_I8_TY` arg to `getelementptr`).
- **`llvm.mlir_addressof(ptr_ty, "global_sym")`** takes the symbol name we just declared (e.g. `nt_pp_smem_as0_bf16_8192_8192`) and gives us an SSA value: a pointer to the start of that global. This is what carries provenance.
- **`llvm.getelementptr(...)`** does pointer arithmetic without breaking provenance. The fourth argument, `[_GEP_DYN]`, is the MLIR convention: each entry in `rawConstantIndices` is either a concrete `i32` constant or the sentinel `kDynamicIndex = std::numeric_limits<int32_t>::min() = -(2**31)`, defined at `mlir/include/mlir/Dialect/LLVMIR/LLVMOps.td:367`. A `_GEP_DYN` sentinel says "this index slot is taken from the `indices` SSA-value list" — i.e., the byte offset is a runtime value.

Why this matters: the alternative — the "legacy" FlyDSL path — was `arith.index_cast` → `llvm.inttoptr`. `inttoptr` discards provenance: the resulting `ptr<3>` looks to LLVM like it might alias *anything* in LDS, so `AMDGPULowerModuleLDSPass` can't attach `!alias.scope` metadata to loads/stores through it, and `SIInsertWaitcnts` falls back to a full `vmcnt(0)` drain at every barrier. By going `addressof + GEP` we keep the pointer attached to its symbol, the pass attaches scope metadata for free, and partial drains are possible. This is Caveat 9 in the porting skill, and the cost it saves is measurable — see `reference_lds_provenance_perf_cost.md` in memory.

---

## 9. Alias scopes (the `!alias.scope` / `!noalias` metadata story)

The kernel constructs **per-stage** alias scopes for LDS regions:

```python
LDS_ALIAS_DOMAIN = f'#llvm.alias_scope_domain<id = "nt_pp_{dtype}_{k}_{n}.lds">'
SCOPE_IDS = ("as0", "as1", "bs0", "bs1")

def _scope_attr(ids):
    inner = ", ".join(
        f'#llvm.alias_scope<id = "{i}", domain = {LDS_ALIAS_DOMAIN}>'
        for i in ids
    )
    return ir.Attribute.parse(f"[{inner}]")

_SCOPE = {sid: _scope_attr((sid,)) for sid in SCOPE_IDS}
_NOALIAS = {sid: _scope_attr(tuple(o for o in SCOPE_IDS if o != sid))
            for sid in SCOPE_IDS}
```

This is direct MLIR attribute construction — `ir.Attribute.parse(...)` takes a string in MLIR's textual attribute syntax and gives back an attribute object. The structures here are LLVM-dialect aliases for the LLVM IR `!alias.scope` / `!noalias` metadata:

- **Domain** (`alias_scope_domain`): a top-level "category." All scopes in the same domain can be compared by the alias analysis pass.
- **Scope** (`alias_scope`): a leaf within a domain. Two memory accesses tagged with the same scope are assumed to potentially alias; accesses tagged with disjoint scopes are guaranteed not to alias (modulo the noalias list).
- **noalias list**: when a load is tagged with `alias_scopes=[X]` and `noalias_scopes=[Y, Z]`, it explicitly cannot alias accesses tagged with `Y` or `Z`.

Per-stage means there are **four** scopes — one per `(side, stage)` combination — not two. The reason is in Caveat 7 of the porting skill: per-region scopes (`lds_a` vs `lds_b`, 2 scopes) don't help pingpong because a prefetch writes `AS@stage1` while a read reads `AS@stage0`, both in the same `lds_a` scope, so the alias analysis pass conservatively says "potentially aliasing" and forces a full `vmcnt(0)` drain. With per-stage scopes the writes and reads are in disjoint scopes and partial drains are possible.

The scopes are attached at op construction time:

```python
return llvm.LoadOp(
    vec_t, gep, alignment=2,
    alias_scopes=my_scope, noalias_scopes=other_scopes,
).result
```

and

```python
rocdl.raw_ptr_buffer_load_lds(
    ..., alias_scopes=_as_scopes[lds_stage], noalias_scopes=_as_noalias[lds_stage],
)
```

Once attached, the metadata flows through `AMDGPULowerModuleLDSPass` and reaches `SIInsertWaitcnts`. Crucially, the pass merges our scopes with its own anonymous ones via `MDNode::getMostGenericAliasScope` (see `AMDGPULowerModuleLDSPass.cpp:1316-1320`) — so we get both layers of granularity, not just one. `SIInsertWaitcnts` then uses the combined metadata to decide which `vmcnt`/`lgkmcnt` drains can be replaced with partial drains or skipped entirely.

---

## 10. Counterfactual: what if we used coarser scopes?

The porting skill ([Caveat 7](../../.claude/skills/port-hipkittens-to-flydsl/SKILL.md)) says "per-region scopes don't help pingpong." That claim deserves a side-by-side, because the *code* differences between three configurations are tightly localized — most of the kernel doesn't change.

The three configurations:

- **A.** Single global, no scopes (the legacy `SmemAllocator` + `inttoptr` path).
- **B.** Per-region scopes — 2 globals (AS, BS), 2 scopes ("a", "b").
- **C.** Per-stage scopes — 4 globals (AS@0, AS@1, BS@0, BS@1), 4 scopes ("as0", "as1", "bs0", "bs1"). What the kernel uses.

### A. Single global, no scopes

```python
# At kernel-compile-time:
LDS_SYM = f"nt_pp_smem_{dtype}_{k}_{n}"  # one global

# Inside @flyc.kernel body:
allocator = SmemAllocator()
as_ptr = allocator.alloc(STAGES * BLOCK_M * BLOCK_K, dtype)
bs_ptr = allocator.alloc(STAGES * BLOCK_N * BLOCK_K, dtype)
as_ = STensor(as_ptr, dtype, shape=(STAGES, BLOCK_M, BLOCK_K))
bs_ = STensor(bs_ptr, dtype, shape=(STAGES, BLOCK_K, BLOCK_N))

# DMA helper — runtime stage is fine, no scope plumbing:
def ldg_sts_a_async(k_offset, lds_stage):  # lds_stage is RUNTIME ir.Value
    for i in range_constexpr(LDG_A_REG_COUNT):
        ...
        # SmemAllocator → extract index → i32 → inttoptr:
        lds_base = memref.extract_aligned_pointer_as_index(as_.raw_ptr)
        lds_off  = lds_base + lds_stage * AS_STAGE_BYTES + computed_offset
        addr_i32 = arith.index_cast(T.i32, lds_off)
        lds_ptr  = llvm.inttoptr(_LDS_PTR_TY, addr_i32)   # ← provenance dies
        rocdl.raw_ptr_buffer_load_lds(
            A_.rsrc, lds_ptr, ...,
            # no alias_scopes / noalias_scopes — pass has nothing to attach
        )
```

LLVM IR for the LDS pointer + DMA:

```llvm
%addr   = ...                                          ; i64 byte address
%lds_p  = inttoptr i64 %addr to ptr addrspace(3)       ; provenance lost
call void @llvm.amdgcn.raw.ptr.buffer.load.lds(..., ptr addrspace(3) %lds_p, ...)
```

`AMDGPULowerModuleLDSPass` sees `NumberVars = 1` — even without `inttoptr`, the `NumberVars > 1` gate (§9) would short-circuit scope generation. With `inttoptr` on top, the pass also can't trace provenance back to the global. `SIInsertWaitcnts` falls back to "all LDS aliases everything," and every barrier in the K-loop ends up with an explicit or implicit `s_waitcnt vmcnt(0)`.

### B. Per-region scopes (2 globals)

```python
# At kernel-compile-time:
AS_TOTAL_BYTES = STAGES * BLOCK_M * BLOCK_K * DTYPE_BYTES
BS_TOTAL_BYTES = STAGES * BLOCK_N * BLOCK_K * DTYPE_BYTES
LDS_SYMS = (f"nt_pp_smem_a_{dtype}_{k}_{n}",
            f"nt_pp_smem_b_{dtype}_{k}_{n}")
LDS_ALIAS_DOMAIN = f'#llvm.alias_scope_domain<id = "nt_pp_{dtype}_{k}_{n}.lds">'
SCOPE_IDS = ("a", "b")

# Inside @flyc.kernel body:
_a_base = llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYMS[0])
_b_base = llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYMS[1])
_SCOPE_A = _scope_attr(("a",)); _SCOPE_B = _scope_attr(("b",))
_NOALIAS_A = _scope_attr(("b",)); _NOALIAS_B = _scope_attr(("a",))

# DMA helper — runtime stage is STILL fine; offset just gets a stage multiplier:
def ldg_sts_a_async(k_offset, lds_stage):  # lds_stage is RUNTIME ir.Value
    for i in range_constexpr(LDG_A_REG_COUNT):
        ...
        stage_off  = lds_stage * AS_STAGE_BYTES         # arith.muli at runtime
        total_off  = stage_off + computed_offset
        off_i32    = arith.index_cast(T.i32, total_off)
        lds_ptr    = _gep_lds(_a_base, off_i32)
        rocdl.raw_ptr_buffer_load_lds(
            A_.rsrc, lds_ptr, ...,
            alias_scopes=_SCOPE_A,
            noalias_scopes=_NOALIAS_A,   # only "B-side accesses don't alias A-side"
        )
```

LLVM IR for a write to stage 1:

```llvm
%off    = ...                                          ; runtime: muli + add
%lds_p  = getelementptr i8, ptr addrspace(3) @nt_pp_smem_a_..., i32 %off
call void @llvm.amdgcn.raw.ptr.buffer.load.lds(...,
            ptr addrspace(3) %lds_p, ...,
            !alias.scope !{!"a"},
            !noalias     !{!"b"})
```

What this buys: a B-side `ds_read` no longer waits on pending A-side `buffer_load_lds`. Cross-region drains can be partial.

What it doesn't buy — the pingpong steady state:

```python
# K-loop body, schematically:
ldg_sts_a_async(k_next, lds_stage=1)   # write to A region, stage 1 — scope "a"
...
lds_matrix_a_kk(lds_stage=0, kk=0)     # read from A region, stage 0 — scope "a"
```

**Both ops are in scope `"a"`**. The alias analysis pass uses the rule "same scope ⇒ may alias" (it has no information distinguishing stage 0 from stage 1 within the region). The reader's `noalias = {"b"}` doesn't exclude the writer. Result: at the barrier between them, `SIInsertWaitcnts` still emits `s_waitcnt vmcnt(0)` for the A writes — same outcome as Configuration A for the pingpong pattern.

### C. Per-stage scopes (4 globals) — what the kernel uses

```python
# At kernel-compile-time:
LDS_SYMS_A = (f"nt_pp_smem_as0_{dtype}_{k}_{n}",
              f"nt_pp_smem_as1_{dtype}_{k}_{n}")
LDS_SYMS_B = (f"nt_pp_smem_bs0_{dtype}_{k}_{n}",
              f"nt_pp_smem_bs1_{dtype}_{k}_{n}")
SCOPE_IDS = ("as0", "as1", "bs0", "bs1")

# Inside @flyc.kernel body:
_as_bases = (llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYMS_A[0]),
             llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYMS_A[1]))
# ... bs_bases similar ...
_as_scopes  = (_SCOPE["as0"],  _SCOPE["as1"])
_as_noalias = (_NOALIAS["as0"], _NOALIAS["as1"])

# DMA helper — lds_stage MUST be a Python int (no runtime indexing on tuples):
def ldg_sts_a_async(k_offset, lds_stage):  # lds_stage is Python int (0 or 1)
    for i in range_constexpr(LDG_A_REG_COUNT):
        ...
        # No stage multiplication — each stage has its OWN base:
        off_i32 = arith.index_cast(T.i32, computed_offset)
        lds_ptr = _gep_lds(_as_bases[lds_stage], off_i32)   # COMPILE-TIME pick
        rocdl.raw_ptr_buffer_load_lds(
            A_.rsrc, lds_ptr, ...,
            alias_scopes  = _as_scopes[lds_stage],          # COMPILE-TIME pick
            noalias_scopes= _as_noalias[lds_stage],         # the OTHER three scopes
        )
```

LLVM IR for a prefetch into stage 1, with a stage-0 read still in flight:

```llvm
%off    = ...                                          ; runtime
%lds_p  = getelementptr i8, ptr addrspace(3) @nt_pp_smem_as1_..., i32 %off
call void @llvm.amdgcn.raw.ptr.buffer.load.lds(...,
            ptr addrspace(3) %lds_p, ...,
            !alias.scope !{!"as1"},
            !noalias     !{!"as0", !"bs0", !"bs1"})

; ... a parallel ds_read of stage 0 ...
%rd_p   = getelementptr i8, ptr addrspace(3) @nt_pp_smem_as0_..., i32 %off2
%val    = load <8 x bf16>, ptr addrspace(3) %rd_p,
            !alias.scope !{!"as0"},
            !noalias     !{!"as1", !"bs0", !"bs1"}
```

Now `as0` ∈ reader's noalias-set of the writer (and vice versa). The pass concludes "guaranteed not to alias." `SIInsertWaitcnts` skips the `vmcnt(0)` drain at the intervening barrier.

### Side-by-side

| | A. single global | B. per-region (2) | C. per-stage (4) |
|---|---|---|---|
| LDS globals | 1 | 2 | 4 |
| LDS pointer construction | `inttoptr` | `addressof + GEP` | `addressof + GEP` |
| `lds_stage` is | Runtime `ir.Value` | Runtime `ir.Value` | **Python int** |
| Helper accepts runtime stage | ✓ | ✓ | ✗ — needs 2× unroll |
| `AMDGPULowerModuleLDSPass` adds scopes | ✗ (`NumberVars = 1`) | ✓ | ✓ |
| Cross-region wait elision | ✗ | ✓ | ✓ |
| **Pingpong (same region, different stage) elision** | ✗ | ✗ | ✓ |
| `s_waitcnt vmcnt(0)` per K-step (gfx950) | ~4 | ~2-3 | ~0 (only explicit) |

The 2× unroll is a real cost — the IR for the K-loop body doubles, instruction-cache pressure goes up, and the scheduler has fewer freedoms than a tight loop. On the NT pingpong shapes it's a net win because the `vmcnt`-elision savings dominate. On NN-family / TN / splitk it's not even available — those kernels have a structural blocker (LDS-aliased C write-back; [Caveat 11](../../.claude/skills/port-hipkittens-to-flydsl/SKILL.md)) and stay on Configuration A.

---

## 11. The closure-namespace pattern

`_make_lds_view_2d` returns an object built like this:

```python
ns = type("LDSView", (), {})()
ns.base_ptr = base_ptr
ns.shape = shape
ns.stride = stride
ns.linear_offset = linear_offset
ns.vec_load = vec_load
return ns
```

That `type("LDSView", (), {})()` creates an instance of an **anonymous class with no body**, then attaches closures as attributes. Why not a regular class? Because the FlyDSL AST rewriter's injected `__check_local_var(...)` calls get **Python name-mangled** if they appear inside a class scope — they become `_LDSView__check_local_var`, which is undefined. Bug, fix is to use a namespace object that has no class scope around the closures. Caveat 6 in the porting skill.

You'll see this pattern wherever helper objects need closures and the closure references runtime-value state.

---

## 12. ROCDL intrinsics

`rocdl.*` ops are direct MLIR wrappers around AMDGPU LLVM IR intrinsics. They're the lowest-level primitives you can hit from FlyDSL without writing inline assembly. The kernel uses:

| Op | Intrinsic | What it does |
|---|---|---|
| `rocdl.s_barrier()` | `llvm.amdgcn.s.barrier` | The bare `s_barrier` instruction. No fence pairs. |
| `rocdl.s_waitcnt(imm)` | `llvm.amdgcn.s.waitcnt` | Emits `s_waitcnt` with a literal 16-bit encoding. `VMCNT_0 = 0x0F70` and `LGKMCNT_0 = 0xC07F` are the encodings; see the comment above their definition for the bit layout. |
| `rocdl.s_setprio(N)` | `llvm.amdgcn.s.setprio` | Sets the wave's instruction-issue priority. Used to bias the scheduler toward MFMA-busy waves during their cluster. |
| `rocdl.sched_barrier(0)` | `llvm.amdgcn.sched.barrier` | Compile-time scheduling fence. The mask says which instruction types are *allowed* to cross; `0 = NONE` allowed = nothing crosses. (See §17 for the inversion in `invertSchedBarrierMask`.) Used between unrolled iterations to keep the cluster schedule honest. |
| `rocdl.raw_ptr_buffer_load_lds(...)` | `llvm.amdgcn.raw.ptr.buffer.load.lds` | Asynchronous HBM → LDS DMA. Uses a buffer resource descriptor (the SRD; §13) and a destination LDS pointer. |
| `rocdl.readfirstlane(T.i32, x)` | `llvm.amdgcn.readfirstlane` | Reads lane 0's value into a scalar register. Promotes a value the compiler doesn't realize is uniform into the SGPR file. |
| `rocdl.mfma_f32_16x16x32_bf16(...)` | `llvm.amdgcn.mfma.f32.16x16x32.bf16` | The MFMA atom. Returns `vec<4 x f32>` C-fragment per lane. |

A subtle one: **`rocdl.readfirstlane`** at the call site is wrapping `lds_off_i32`, which is the per-thread LDS byte offset. For a `buffer_load_lds` issued with the same destination from every lane in a wave, the LDS pointer is uniform across the wave — but the compiler can't always tell from the data-flow shape that it is. Calling `readfirstlane` makes the uniformity explicit, which lets the destination pointer live in an SGPR pair rather than a VGPR pair. Smaller code, no per-lane redundancy.

The MLIR-level "intrinsic op" pattern is uniform: you import from `flydsl.expr.rocdl`, call the op with positional arguments matching the LLVM IR intrinsic signature, and (for some) pass extra MLIR-level attributes as keyword arguments (`alias_scopes=`, `noalias_scopes=` here). The op IDs all map 1:1 to LLVM intrinsics defined in `llvm/IR/IntrinsicsAMDGPU.td`.

---

## 13. Buffer resource descriptors

`A_.rsrc` and `B_.rsrc` are the AMDGPU **buffer resource descriptors** — 128-bit SGPR quads that encode `{base_addr, num_records, stride, flags}`. `raw_ptr_buffer_load_lds` needs an SRD as its first operand. `GTensor` builds one up-front from the input tensor's data pointer and size; it's reused for every DMA call.

If you're new to AMDGPU and used to CUDA: this replaces the global-memory pointer that CUDA implicitly threads through every `__global__` load. AMDGPU's `buffer_load_*` family is preferred over `flat_load_*` because the SRD-based addressing supports range-bound checks (the "num_records" field) for free, which the compiler exploits for bounds-checked loads. The `raw_ptr_*` family takes a pointer for the LDS destination, the `raw_*` family takes everything in SGPRs/VGPRs.

The kernel uses the SRD form everywhere it touches HBM: `ldg_sts_a_async`, `ldg_sts_b_async`.

---

## 14. Occupancy + register allocation knobs

Right before `launcher.launch(...)`, the kernel sets:

```python
op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, 2)
op.attributes["passthrough"] = passthrough_attr  # empty array
```

`waves_per_eu` is an LLVM function attribute the AMDGPU backend reads to bound the register budget the allocator may use. Setting it to N tells the allocator "size the kernel so at least N waves fit per SIMD/EU." Larger values → smaller VGPR budget per wave → more waves can be resident.

Why 2 specifically for this kernel: the kernel runs at **8 waves / WG, 1 WG / CU**. With 4 SIMDs / CU on CDNA4, that's 8 / 4 = **2 waves / SIMD** at the intended occupancy. So `waves_per_eu = 2` is the *largest* budget the allocator can use without missing the target — exactly what we want for a register-hungry MFMA pipeline. Setting it higher (3, 4, ...) would unnecessarily restrict the per-wave VGPR pool and may force spills or AGPR offload.

`passthrough` is an MLIR LLVM-dialect attribute that gets carried through to LLVM IR function attributes verbatim. Empty here, but the slot is reserved for adding things like `"amdgpu-agpr-alloc"="0,0"` to force the AGPR budget (Caveat 10 in the porting skill).

---

## 15. Walkthrough: how the pieces fit, in source order

A short tour now that the features are explained. I'll quote section headers from the source comments where I can.

### Tile + warp constants block

Pure Python integers. `BLOCK_M`, `BLOCK_N`, etc. These are compile-time constants used to size loops and select MFMA atoms. None of these survive as IR — they get folded into op operands.

### `WMMA_IMPL = _WmmaHalfK32(dtype)` and friends

`_WmmaHalfK32` is a small Python class (in `gemm_gfx950_mfma_core.py`) that wraps the right `rocdl.mfma_*` op for the dtype. Its `__call__` returns a new `vec<4 x f32>` accumulator. Used inside `mma_kk`.

### LDS allocation constants

Numeric constants only — they get baked into the global names (`f"nt_pp_smem_as0_{dtype}_{k}_{n}"`) so distinct `(dtype, K, N)` compilations get distinct globals. This is important: globals are module-scope, so reusing the same symbol across compilation units would be an error.

### `@flyc.kernel` body opens

The decorated function is `nt_kernel(C, A, B, m)`. `m` is a runtime `Int32`; everything else is a tensor descriptor. The first MLIR op is the constant zero accumulator vector:

```python
acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG, T.f32))
```

### `GTensor(...)` descriptors

These wrap the raw `fx.Tensor` (an MLIR `memref<?xT>`) into a higher-level helper with `.linear_offset`, `.vec_load`, `.vec_store`, `.rsrc`. Shape `(-1, k)` means "M is dynamic, K is constexpr." The dynamic dim is what makes M runtime-flexible without recompiling.

### LDS descriptors (provenance + per-stage alias scopes)

This is where the work from §7, §8, §9 comes together. The `as_bases` and `bs_bases` tuples are pairs of `llvm.mlir_addressof` results — one base pointer per stage. The `_SCOPE` / `_NOALIAS` dicts are the alias metadata attributes built from string-parsed MLIR attribute syntax. `_make_lds_view_2d` packages it all into a closure-namespace object so the helpers downstream can call `.vec_load((row, col), 8)` without redoing the GEP machinery.

### Tile coords (1D grid → 2D via 2-stage swizzle)

The `fx.const_expr` conditional gates the swizzle math out at compile time when the swizzle is disabled. Note `arith.select(...)` for the runtime "min" — there's no `min()` op in `arith`, so you build it from `cmpi + select`.

### Thread / warp indices

```python
tid = fx.Int32(fx.thread_idx.x)
wid = tid // WARP_SIZE
w_tid = tid % WARP_SIZE
warp_row = wid // BLOCK_N_WARPS
```

`fx.thread_idx.x` is the FlyDSL surface for `gpu.thread_id x` — the workitem coordinate. `fx.block_idx.x` is `gpu.block_id x`. These are runtime values; arithmetic on them emits `arith.divsi`, `arith.remsi`, etc.

### Helpers: `ldg_sts_a_async`, `ldg_sts_b_async`

Inner Python functions that close over `m_offset`, `n_offset`, `k_blocks16`, `_as_bases`, etc. The `for i in range_constexpr(LDG_A_REG_COUNT)` loop fully unrolls — `LDG_A_REG_COUNT = 4` means 4 separate `raw_ptr_buffer_load_lds` ops per call. Within each iteration:

1. Compute the global byte offset (HBM source).
2. Compute the LDS byte offset (LDS dest).
3. `readfirstlane` the LDS offset to put it in an SGPR.
4. Build the LDS dest pointer via `_gep_lds(_as_bases[lds_stage], lds_off_uniform)`.
5. Emit the DMA op with `alias_scopes=` and `noalias_scopes=` for the stage.

The bounds-check on rows (`arith.select(...)` against `m`) is what makes M runtime-dynamic safe even when it's not a multiple of `BLOCK_M`.

### Helpers: `lds_matrix_a_kk`, `lds_matrix_b_kk`

LDS → register fragment loads. Returns a Python list of MLIR vector `Value`s. The Python list is fine — `range_constexpr` unrolls, so by the time IR is materialized each entry is a distinct SSA value. The MFMA wrapper consumes them by indexing the Python list with compile-time `ii` / `jj`.

### Helper: `mma_kk`

Wraps the priority-toggle + MFMA loop:

```python
def mma_kk(a_frags, b_frags, c_frags):
    rocdl.s_setprio(1)
    for ii in range_constexpr(WARP_M_STEPS):
        ...
        for jj in range_constexpr(WARP_N_STEPS):
            ...
            c_frags[c_idx] = WMMA_IMPL(a_frag, b_frag, c_frags[c_idx])
    rocdl.s_setprio(0)
```

`c_frags` is a **Python list of MLIR `Value`s**. Mutating `c_frags[c_idx] = WMMA_IMPL(...)` replaces the entry in the Python list with the new MFMA result — and because the next MFMA in the unroll reads `c_frags[c_idx]` again, the dependency chain is built naturally. After the loop, `c_frags` holds the final accumulator values.

### Prologue

The first prefetch, then a single `vmcnt(0)` drain + `s_barrier()` to publish the data across the workgroup. The HK-style conditional `s_barrier` desync follows — a single `arith.cmpi`-guarded extra barrier that splits the 8 waves into two pingpong groups by phase. Caveats 2, 3 in the porting skill.

### Main loop (2× unrolled)

The scf.for `range(0, OUTER_ITERS - 1, init=init_state)` pattern. Each outer iter handles 2 K-steps via `kstep_cluster` calls. The key point: `kstep_cluster(0, ..., 1)` and `kstep_cluster(1, ..., 0)` are **separate Python calls** with `read_stage` and `prefetch_stage` as Python ints (0 or 1) — so `as_views[read_stage]` and `as_views[prefetch_stage]` resolve at compile time. This is the whole reason for the 2× unroll.

The `rocdl.sched_barrier(0)` at the bottom prevents instruction scheduling across the loop's back-edge — keeps the scheduler from hoisting things in ways that would break the per-iteration cluster structure.

### Epilogue

The same K-step body inlined twice for the final two K-steps, with manual control over when to NOT prefetch (the last K-step's prefetch would be off the end of `K`). The structure is repeated rather than wrapped in a helper because the prefetch absence is the only meaningful difference and inlining keeps the dependency chain transparent.

### Writeback

The `range_constexpr(WARP_M_STEPS)` over the C-fragment grid. For each fragment, for each of its 4 elements, build the masked store with `scf.IfOp` + InsertionPoint. The mask is `m_global < m` — N is guaranteed `% 256 == 0` by the public-API check so no N mask is needed. `vec_store((m_global, n_global), val_dtype, 1)` is `GTensor`'s helper to emit `memref.store` with the right type cast.

The simple scalar-store pattern is intentional — see the comment block before the writeback: the SIMD's natural coalescing handles this case fine, and the `ds_bpermute`-based repacking that you might reach for first is a 5% regression here.

### Launcher `@flyc.jit`

Three jobs:

1. Emit the four `llvm.GlobalOp`s at module scope (§7).
2. Compute grid dims from the runtime `m`.
3. Set the `waves_per_eu` attribute on the kernel function before launch (§14).

The pattern `for op in ctx.gpu_module_body.operations: if ... op.OPERATION_NAME == "gpu.func":` is how you find the GPU function inside the module after the kernel has been traced. You can attach LLVM IR attributes to it directly.

---

## 16. Compile-time vs runtime cheat sheet

| You want to … | Use |
|---|---|
| Branch on a Python `bool` | `if fx.const_expr(PY_BOOL): ... else: ...` |
| Branch on a runtime `i1` | `if arith.cmpi(pred, a, b): ...` (only `cmpi`/`Boolean`) |
| Branch with manual control | `scf.IfOp(...)` + `ir.InsertionPoint(...)` + `scf.YieldOp([...])` |
| Unroll a fixed-count loop | `for i in range_constexpr(N): ...` |
| Runtime loop with accumulator | `for _, state in range(s, e, init=[...]): ... yield [...]` |
| Index a Python tuple by stage | The stage must be a Python int. Unroll to make it one. |
| Combine compile-time conditions | Plain Python: `if X and Y and not Z: ...` (folds away) |
| Combine runtime conditions | Nest `arith.cmpi`s — **do not** use `arith.andi` as a branch predicate |
| Get an MLIR Value into a SGPR | `rocdl.readfirstlane(T.i32, x)` |
| Build a vector type | `T.vec(N, T.f32)` |
| Build a typed constant | `arith.constant(7, type=T.i32)` |
| Build a vector constant | `arith.constant_vector(0.0, T.vec(N, T.f32))` |
| Add metadata to an LLVM-dialect op | Pass as kwarg: `alias_scopes=`, `noalias_scopes=`, `alignment=` |
| Cast index ↔ i32 | `arith.index_cast(T.i32, x_index)` and reverse |
| Cast f32 → bf16/f16 | `arith.truncf(dtype_, x_f32)` |
| Pull an element out of a vec | `vector.extract(vec, static_position=[e], dynamic_position=[])` |
| Reinterpret-cast a vec's element type | `vector.bitcast(T.vec(N, T.i16), vec_f16)` |

---

## 17. Verified findings (and what's still unverified)

I dove into the LLVM / MLIR / FlyDSL sources to pin down the items I'd flagged uncertain in earlier drafts. Verified against current source:

### Verified — `_GEP_DYN = -(2 ** 31)`

`/workspace/llvm-project/mlir/include/mlir/Dialect/LLVMIR/LLVMOps.td:367`:
```cpp
constexpr static int32_t kDynamicIndex = std::numeric_limits<int32_t>::min();
```
`std::numeric_limits<int32_t>::min() == INT32_MIN == -2147483648 == -(2 ** 31)`. The kernel's `_GEP_DYN` constant is exactly `LLVM::GEPOp::kDynamicIndex`. ✓

### Verified — `AMDGPULowerModuleLDSPass` requires `NumberVars > 1`

`/workspace/llvm-project/llvm/lib/Target/AMDGPU/AMDGPULowerModuleLDSPass.cpp:1267-1276`:
```cpp
const size_t NumberVars = LDSVarsToTransform.size();
if (NumberVars > 1) {
  AliasScopes.reserve(NumberVars);
  MDNode *Domain = MDB.createAnonymousAliasScopeDomain();
  for (size_t I = 0; I < NumberVars; I++) {
    MDNode *Scope = MDB.createAnonymousAliasScope(Domain);
    AliasScopes.push_back(Scope);
  }
  NoAliasList.append(&AliasScopes[1], AliasScopes.end());
}
```

With exactly one LDS global, no scopes get attached and the alias analysis pass treats all LDS accesses as potentially aliasing. The kernel emits **four** globals (`as0`, `as1`, `bs0`, `bs1`), well above the threshold.

A bonus discovery: lines 1314-1320 of the same file show how the pass merges metadata when an instruction **already** carries `!alias.scope` (from our explicit emission):
```cpp
if (AliasScope && I->mayReadOrWriteMemory()) {
  MDNode *AS = I->getMetadata(LLVMContext::MD_alias_scope);
  AS = (AS ? MDNode::getMostGenericAliasScope(AS, AliasScope) : AliasScope);
  I->setMetadata(LLVMContext::MD_alias_scope, AS);
```
The pass calls `MDNode::getMostGenericAliasScope` to combine our **named** per-stage scopes with its **anonymous** per-variable scopes. So loads end up tagged with both granularities; `SIInsertWaitcnts` sees the union.

### Verified — `rocdl.sched_barrier(0)` is a hard scheduling fence

`/workspace/llvm-project/llvm/lib/Target/AMDGPU/AMDGPUIGroupLP.cpp:67-83` defines the mask enum:
```cpp
enum class SchedGroupMask {
  NONE = 0u,
  ALU = 1u << 0,
  VALU = 1u << 1,
  ...
  ALL = ALU | VALU | SALU | MFMA | VMEM | VMEM_READ | VMEM_WRITE | DS | DS_READ | DS_WRITE | TRANS,
};
```
And lines 2634-2645 show the mask is **inverted** when building the SchedGroup:
```cpp
void IGroupLPDAGMutation::addSchedBarrierEdges(SUnit &SchedBarrier) {
  ...
  auto InvertedMask =
      invertSchedBarrierMask((SchedGroupMask)MI.getOperand(0).getImm());
  SchedGroup SG(InvertedMask, std::nullopt, DAG, TII);
  for (SUnit &SU : DAG->SUnits)
    if (SG.canAddSU(SU))
      SG.add(SU);
```

The mask the user passes is "which instructions are *allowed* to cross"; the scheduler builds a SchedGroup with the inverted mask (instructions that **may not** cross). `sched_barrier(0)`:
- Mask = `NONE = 0`
- InvertedMask = `~0` (after the implication fixups in `invertSchedBarrierMask`) = effectively all instruction classes
- The SchedGroup includes **every** SUnit in the DAG — nothing may be scheduled past the barrier.

So `sched_barrier(0)` is "no instructions cross." ✓

### Verified — AST rewriter behavior

`/workspace/FlyDSL/python/flydsl/compiler/ast_rewriter.py`:

- L58-63: `_is_constexpr(node)` returns `True` iff `node` is a `Call` whose target name is `"const_expr"`.
- L713-715: `ReplaceIfWithDispatch.visit_If` short-circuits when `_is_constexpr(node.test)` — that branch stays Python-level (compile-time).
- L490-495: `_is_dynamic(cond)` returns `True` if `cond` is an `ir.Value` or has a `.value` attribute that is one. Everything else falls through to the static-cond path.
- L588-648: `scf_if_dispatch` constructs `scf.IfOp(cond_i1, ...)` with the unwrapped `ir.Value`. The op is then verified by MLIR — which means the value **must be `i1`** to construct successfully.

So the operational story is:

| You wrote | Path taken |
|---|---|
| `if fx.const_expr(PY_BOOL):` | AST visitor stays in Python; condition resolved at trace time |
| `if arith.cmpi(...):` | Runtime dispatch; `cmpi` returns `i1` → `scf.IfOp` valid |
| `if arith.andi(i1, i1):` | Runtime dispatch; `andi(i1, i1)` returns `i1` → `scf.IfOp` valid (mechanically) |
| `if arith.andi(i32, i32):` | Runtime dispatch; `andi(i32, i32)` returns `i32` → `scf.IfOp` would fail verification, or worse: be silently miscompiled in older builds |
| `if py_a and py_b:` (Python `and` between MLIR values) | Python's `and` calls `__bool__` on the first value — undefined for `ir.Value`; result depends on FlyDSL wrapper |

The porting skill's caveat — *"`if arith.andi(...):` silently lets all lanes through"* — was a real observation, but the source I read here shows that **if both operands really are `i1`, the dispatch works correctly**. The failure mode I called out almost certainly involved one of the bottom two rows (wider operands, or accidental Python `and`/`or`). Either way the recommendation stands: **nest `arith.cmpi` calls** rather than reach for `andi` — it's clearer, immune to operand-width drift, and matches the rewriter's blessed shape.

### Still unverified

- **Per-cluster timing claims** ("warp_row==0 is MFMA-busy while warp_row==1 is LDS-busy" — kernel docstring). These come from ATT analysis quoted in the docstring; I have not re-run ATT in this session.
- **The exact root cause of "silently lets all lanes through"** for an `andi` use I personally observed. As of this re-read, with current FlyDSL source, `andi(i1, i1)` should be safe. If you hit the symptom, file an issue with the IR dump.
- **`waves_per_eu` perf sensitivity**: I confirmed `= 2` is correct given the kernel's intended 1 WG/CU × 4 SIMDs occupancy, and that correctness is intact. I did not bench `2` vs `3` vs `4` in this session to quantify the perf delta — but the change is in the direction the docstring already recommended.

---

## 18. Further reading

- `quack/amd/gemm_gfx950_nt_pingpong_16x32.py` — the sibling kernel; same patterns, 16x32 quadrant.
- `quack/amd/gemm_gfx950_nt_4wave_hk.py` — 4-wave pingpong, an instructive contrast (different barrier density).
- `.claude/skills/port-hipkittens-to-flydsl/SKILL.md` — recipe + caveats for porting HK kernels.
- `docs/superpowers/specs/2026-06-04-flydsl-lds-provenance-todo.md` — the implementation log for the per-stage scope work (the "2026-06-09 implementation LANDED" section is the end-state writeup).
- `docs/dsl_control_flow.rst` and `docs/limitations.rst` — FlyDSL's own surface-level reference.
- HK reference: `/tmp/HipKittens/kernels/gemm/bf16fp32/256_256_64_32_with32x16.cpp` (if installed).
