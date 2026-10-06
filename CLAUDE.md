# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## The goal

**Port llama.cpp to Zig — pure Zig apart from the GPU kernels and the Metal Objective-C host layer.** The deliverable is a `llama-cli` reimplemented in Zig: not a recreation of the llama.cpp repo, and not a Zig wrapper around the C++ library. `SPEC.md` states what the deliverable is and does — artifacts, flag surface, generation semantics, and the conformance gates. `README.md` has the stages; `PLAN.md` has the forward plan — decisions, scope measurements, and what comes next; `NOTES.md` is the record of how each completed step was actually done and what each gate caught. Read `SPEC.md` first when the question is "what should this do", `PLAN.md` when it is "what is decided and what is next", and `NOTES.md` when it is "why is it done this way".

Three decisions worth knowing before touching anything:

- **`ggml-metal-device.m` and `ggml-metal-context.m` stay Objective-C** (3,091 lines), compiled by `zig cc`. Zig has no Obj-C frontend, and reaching Metal from Zig would mean hand-writing every call against `objc_msgSend`. Everything else is ported.
- **CUDA is in scope, and its kernels stay CUDA** — `PLAN.md` "Stage 6 — CUDA" has the measurement. Three things about it differ from Metal and are easy to get wrong:
  - **It is big.** 278 files, ~43,900 lines (24,720 `.cu`, 18,747 `.cuh`), with `ggml-cuda.cu` alone at 5,583 — roughly **3.3× the whole Metal backend**.
  - **The Metal split does not transfer.** Metal works because MSL and host code are in *different files*; CUDA interleaves them, and all **161** `<<<…>>>` launches sit inside host functions in syntax only `nvcc` can compile. The route that transfers is `cudaLaunchKernel`, which is plain C and callable from Zig — the same position Zig already occupies when it calls into `ggml-metal-device.m`.
  - **No gate here can see it.** This machine is an Apple M5 Max with no `nvcc` and no NVIDIA GPU, and macOS has had no CUDA support since 10.13. Every correctness gate below is a local bit-exact diff, so for CUDA there are currently **none**. Resolving that is a prerequisite to writing CUDA code, not a follow-up — see the three options in `PLAN.md`.
- **Our `llama-cli` mimics upstream's argument set** for the features we support, so an existing command line runs unchanged against it. It is our own Zig binary, not upstream's linked against our library.

Where we are: **Stages 1, 2 and 3 complete; Stage 4 under way — all of `ggml/src/ggml-cpu/` is Zig.**

**No C compiles anywhere under `llama.cpp/ggml/src/` any more.** All six translation units are Zig, 598 exported symbols, every one swapped in and byte-verified. `ggml_base_sources` and `ggml_cpu_c_sources` in `build/llamacpp.zig` are both empty.

- `ggml-alloc.c` → `src/ggml/alloc.zig`.
- `ggml.c` → 381/381 symbols, split across `src/ggml/{impl,types,context,runtime,ops,graph,quantize}.zig`.
- `ggml-quants.c` → 79/79 symbols, split across `src/ggml/quants/`.
- `ggml-cpu/ggml-cpu.c` → 65/65 symbols, split across `src/ggml/cpu/`.
- `ggml-cpu/quants.c` → 45/45 symbols, split across `src/ggml/cpu/quants/`.
- `ggml-cpu/arch/arm/quants.c` → 28/28 symbols, split across `src/ggml/cpu/quants/arm/`.

**For a C++ translation unit the contract is its *unmangled* exports.** `scripts/port-coverage` takes `LANG_=cxx`, compiles with `zig c++`, and filters out `_Z…` names and `__clang_call_terminate`. That filter is measured, not assumed — see above. A file among the four exceptions must be ported together with whatever needs its vtables.

**Stage 4 has begun, and the measurement that shapes it is done.** Every exported symbol of every ggml C++ object was intersected with the undefined symbols of every other object in `libggml.a` and `libllama.a`: **the external contract of 20 of the 24 C++ translation units in ggml is a pure C ABI.** A C++ file exports thousands of mangled symbols — `gguf.cpp` alone exports 2,002 — but they are template instantiations and inline functions, emitted weakly into every object that needs them, and almost none are reached from outside. So the Stage 3 swap mechanism carries over unchanged for the large majority. `PLAN.md` names the four exceptions and what each forces.

Seventeen C++ translation units are ported and swapped:

- `ggml-threading.cpp` → `src/ggml/threading.zig` (3 symbols).
- `ggml-backend-reg.cpp` → `src/ggml/backend_reg.zig` (16 symbols). It took `ggml-backend-dl.cpp` out of the build with it: those three `dl_*` functions have C++ linkage Zig cannot provide, and the registry was their only caller.
- `gguf.cpp` → `src/ggml/gguf.zig` (61 symbols). The first STL-heavy unit.
- `ggml-backend.cpp` → `src/ggml/backend.zig` + `src/ggml/backend_sched.zig` (102 symbols), split along the C's own `// scheduler` seam.
- `ggml-cpu/binary-ops.cpp` → `src/ggml/cpu/binary_ops.zig` (4 symbols).
- `ggml-cpu/unary-ops.cpp` → `src/ggml/cpu/unary_ops.zig` (23 symbols).
- `ggml-cpu/vec.cpp` → `src/ggml/cpu/vec.zig` (10 symbols).
- `ggml-cpu/ops.cpp` → `src/ggml/cpu/ops/` (88 symbols, 19 files by op family). The largest translation unit in ggml.
- `ggml-cpu/llamafile/sgemm.cpp` → `src/ggml/cpu/ops/sgemm.zig` (1 symbol).
- The **vtable cluster**, which had to move as one unit because `repack.cpp` derives from the two abstract bases `traits.cpp` declares: `ggml-cpu/traits.cpp` → `src/ggml/cpu/extra.zig` (2 symbols), `ggml-cpu/ggml-cpu.cpp` → `src/ggml/cpu/cpu_backend.zig` (7), `ggml-cpu/repack.cpp` + `ggml-cpu/arch/arm/repack.cpp` → `src/ggml/cpu/repack/` (36 + 28).
- `ggml-metal/ggml-metal-common.cpp` → `src/ggml/metal/common.zig` (6 symbols). The Metal group's first unit, and the only one with no Metal API in it.
- `ggml-metal/ggml-metal.cpp` → `src/ggml/metal/backend.zig` (6 symbols). The backend, device and registry. Its three buffer types and two buffer ifaces collapse to one comptime-parameterised body each — `diff` says the ifaces differ only in an assertion's polarity, and the buffer types only in a name suffix and whether `alloc_buffer` asks for shared memory.
- `ggml-metal/ggml-metal-device.cpp` → `src/ggml/metal/library.zig` (73 symbols). 68 `get_pipeline_*` functions that each build an MSL function name and a cache key, plus the pipeline cache behind them. Its 112 `FC_*`/`OP_*`/`N_*`/`SZ_*` constants are **extracted from `ggml-metal-impl.h` by script** into `src/ggml/metal/impl_c.zig` and checked against the header by `make struct-layout` — a wrong one is the quietest fault in the Metal port, since `FC_UNARY + 1` names the slot a function constant is written to and an off-by-one silently configures a different kernel.
- `ggml-metal/ggml-metal-tuning.cpp` → `src/ggml/metal/{tuning,tuning_table}.zig` (7 symbols). **Its contract is entirely C++-linkage** — seven functions in `namespace ggml_metal_tuning`, zero unmangled exports — so `port-coverage` would read it as 0/0 = 100% without the `MANGLED_` list it now takes. Zig exports the mangled names with `@export`, which is what lets the three C++ files that call it stay C++ while each is ported separately.

**Both counts `PLAN.md` carried for this group were low** — gguf was recorded as 44 and backend as 82. Measure the contract before estimating a C++ file; do not trust the survey figure.

`ggml-backend.cpp` is also the counterexample to "C++ is the hard part": it has **zero `throw`, zero `catch`**, no `std::string`, no `std::map`, no templates and no virtual functions. Its own first line is `// Note: porting this file to C++ is a work in progress`. What made it hard was the five-pass assignment algorithm in `ggml_backend_sched_split_graph`.

Only `ggml-backend-meta.cpp` (4 symbols), `ggml-opt.cpp` (9) and `ggml.cpp` (0, to be dropped) remain in `ggml_base_cxx_sources`. In `ggml_metal_cxx_sources` only `ggml-metal-ops.cpp` (66 symbols) is left. **Nothing under `ggml-cpu/` compiles from C or C++ any more** except three translation units that are empty on this target: `hbm.cpp` needs `GGML_USE_CPU_HBM`, `amx/amx.cpp` and `amx/mmq.cpp` need `__AMX_INT8__`. What remains is the Metal host layer and all of libllama.

**For `repack.cpp` the unmangled-export contract is not the whole contract, and that cost a day.** Its 36 unmangled symbols are the gemv/gemm and `quantize_mat` kernels; the `CPU_REPACK` buffer type, `extra_buffer_type` and the sixteen `tensor_traits` instantiations that decide when to call them are all C++-linkage and invisible to `nm -gU | grep -v _Z`. Measured: with the dispatch stubbed, `port-coverage` read `36 / 36 symbols (100%)` and `node-diff` failed on the first `MUL_MAT`. `scripts/cluster-check` now asserts `repack.dispatch_implemented` at comptime so the count cannot stand alone again.

`make validate` reports the exact counts.

**Porting a translation unit is treated as all-or-nothing.** A file is ported completely before it is swapped; the linker's missing-symbol errors are the completeness check. Develop a large port without wiring it in, so the build stays green until it is finished.

An incremental alternative was trialled and parked: compiling the C with `-Dggml_foo=ggml_foo_c` renames a symbol *and its internal callers* within that translation unit, leaving the external name free for a Zig version. It looked like it worked and did not — see `NOTES.md`, "The incremental-swap trial". The wholesale swap of `ggml.c` made it moot.

**Every ported file states its source as a full path.** The doc comment at the
top of a ported file names the reference file it replaces as a path from the
repository root, with extension — `llama.cpp/ggml/src/ggml-alloc.c`, not
`ggml-alloc.c` — plus the pinned tag and commit. A file drawing on more than one
source lists all of them. Within the file, each ported declaration names the C
function and the line it began at using the bare filename, since the header has
already established the path. Files that are not ports (`module.zig`,
`ported.zig`) say so explicitly rather than staying silent.

**Those citations are the diff map, and `make port-links` enforces them.** 850
of them, across 42 files. When the pin moves, they are what says which of our
functions a given upstream change lands in — so a stale line number is worse
than no line number, because it sends you to the wrong function and says
nothing. The rules:

- **A citation points at the first line of the definition, and names the commit
  it was read at** — ``Ports `ggml_hash_set_new` (ggml.c:6516 @c1d0e7a00)``.
  Where the name only appears on the closing line, as in `ggml-common.h`'s
  anonymous `typedef struct { … } block_q4_K;`, the opening line is cited and
  the checker follows the brace.
- **The commit goes on every citation, not once per file.** An upstream sync
  does not move a whole file at once: one function gets re-read against a newer
  commit while its forty neighbours stay put, which is exactly what a diff
  produces. A single per-file commit cannot express that and would be lying
  about the other forty the moment anyone did it. `port-links` reads each cited
  file **as it was at that citation's own commit**, via `git show
  <sha>:<path>`, so a file may cite as many commits as it needs to. The working
  tree is a fast path only when the citation is at the checkout's `HEAD`.
- **A citation may name several symbols**, `(ggml-impl.h:173, 179, 185, 191
  @c1d0e7a00)`; names and numbers pair up in order.
- **Where a symbol has several `#if` arms**, cite the arm this target compiles
  and say which one in the prose.
- **A citation may point at our own `harness/`** — `testing.zig` cites the C
  that captured its goldens. Those carry no upstream commit and are not asked
  for one.

The gate was negative-tested on all seven failure classes: a line number
drifted by two, a line number deleted, a renamed symbol, a missing commit, an
unknown commit, and — the two that prove the per-citation commit is real — a
citation at an older commit whose line is right *there* passing while the line
that is right at `HEAD` fails. Before it
existed the convention had rotted badly — **216 of the first 790 citations were
stale**, two pointing at the wrong file entirely, and every `Mirrors …` citation
in `blocks.zig` had never been checked at all because the first version of the
checker only looked for the word `Ports`.

**A citation may wrap across comment lines, and for a long time that meant
it was not checked at all.** `findCitation` reads one line, and an
unclosed `(` fell through its `orelse continue` — no citation, no error,
no check. **36 citations across nine files** were wrapped that way, 16 of
them long-standing. The checker now joins continuation comment lines
before parsing, which took the checked count from 1,703 to 1,779 and
surfaced 52 problems that had always been there. Negative-tested: a
wrapped citation drifted by two now fails. Same class as the `Mirrors …`
hole, and the same lesson — **a checker that silently skips input is
indistinguishable from one that passes.**

**The library root is `src/ggml/module.zig`, and it is the only import list.** `ported.zig` re-exports it rather than repeating it, because repeating it let the two diverge: `port-coverage` compiles `ported.zig` while the library is built from `module.zig`, so a file added to one and not the other read `6 / 6 symbols (100%), complete and swapped into the build` for a library containing none of them. The linker caught that one only because something referenced the symbols — a kernel reached through a dispatch table would have passed. Add a newly ported file to `module.zig`.

**`ggml-impl.h` is not importable** — it includes `<arm_neon.h>`, whose `__mfp8` type translate-c cannot parse. `src/ggml/impl.zig` hand-writes what ported code needs from it: assertions, logging, `GGML_PAD`, the hash set, the float conversions, `struct ggml_cgraph`, and C-pointer narrowing helpers. Add to it rather than trying to widen the import.

**`make validate` is the fast check; parity is the real one.** `validate` runs formatting, the scaffold, the unit tests, the ported tests in both debug and release, `scripts/port-coverage`, `scripts/port-links`, and the graph diff. It proves the port is *consistent*, not *correct*.

Correctness has fourteen gates, and none of them subsumes the others:

- **`make graph-diff`** builds 131 nodes through every constructor family and diffs op, shape, strides, `op_params`, and `src[]` against the C. Needs no model. **The sharpest gate for the port's actual failure mode** — a constructor writing the wrong thing — and the only one that catches a halved softmax scale. It found the real `ggml_permute` bug.
- **`make backend-ops`** runs upstream's `test-backend-ops` against our library: 21,093 op configurations, Metal against CPU, all passing. The broadest exercise, and the only one that *executes* the constructors at scale. **It cannot catch a consistently-wrong constructor** — both backends read the same `op_params` and agree. It catches crashes, assertion failures, and shapes a backend has no kernel for. `--diff` runs the same binary against the C reference and diffs the two
  **logs** — 25,176 lines identical. Read that for what it is: on a clean run
  `test-backend-ops` prints an op descriptor and `OK`, and prints the error
  value *only when it exceeds the threshold*, so the diff compares which
  configurations ran and which passed, **not** any computed value. It is a
  structural check, and it does not make `backend-ops` a value-level oracle.
- **`make probe`** proves ported code is *on the execution path*. Green does not mean running: two definitions of a symbol in one static archive is not a link error, the linker just picks one silently. It takes **two runs**, `-Dprobe-ported` and `-Dprobe-cpu`, because the allocator fires while the graph is being planned and would mask the CPU dispatch every time.
- **`scripts/vecdot-prefix` localises a dot-product mismatch.** Not a gate — a diagnostic. It asks the shipped C kernel for every prefix of a row, so the first prefix that diverges isolates the loop iteration, and `nblk = 1` versus `nblk = 2` separates the scalar tail from the vector body. It is what turned an unguessable 1-ULP difference in `iq4_nl` into a solved equation.
- **The `_generic` dot products have no gate but their goldens.** `arch-fallback.h` renames nothing from `quants.c` on ARM and `arch/arm/quants.c` supplies every real entry point, so all 25 `ggml_vec_dot_*_generic` are exported and unreachable. `test-backend-ops` cannot see them and neither can token parity. `src/ggml/cpu/quants/golden.zig` holds the exact `f32` bits the C produces for **six** input patterns, regenerated by `scripts/vecdot-golden`. Compare on **bits**, never a tolerance.

  Two holes in that golden set were found by injection and closed. **Every `nvfp4` ARM golden was `0x00000000`** because the harness never called `ggml_cpu_init` and `GGML_CPU_UE4M3_TO_FP32` is a *table lookup* on NEON — a kernel returning zero would have passed all six. And **no pattern hit an exact tie**, so swapping round-half-to-even for round-half-away passed everything; the `ties` pattern exists to fix that. `testing.zig` now refuses a golden row that is entirely zero.
- **The float dot products have no gate but their goldens either.**
  `ggml_vec_dot_f32`, `_f16` and `_bf16` are accumulating reductions that
  nothing else can see: `make port` never reaches the CPU kernels on a Metal
  machine, and `backend-ops` compares with a tolerance.
  `src/ggml/cpu/vec_golden.zig` holds the exact bits for **seven** patterns at
  two lengths, regenerated by `scripts/vec-golden`, contraction **on**.
  Compare on **bits**.

  The seventh pattern exists because of an escape. With six, swapping the
  **pairwise** `vaddvq_f32` for an ordered `@reduce(.Add, ...)` passed
  everything — the trap named below in "Porting notes", invisible to the gate
  meant to catch it. The two reductions differ on 23.5% of random 4-lane
  vectors, so six patterns agreeing was luck. Lane `k` holds every element
  with `i % 4 == k`, so `skewed` keys magnitude on `i % 4`; no pattern that
  varies magnitude per *element* can reach it.

- **`make ops-diff`** is the **value-level oracle for `ggml-cpu`**. It
  executes ~300 op cases on the CPU backend with fixed inputs — every op
  family, every quantized `mul_mat` type — and compares the output **on
  bits** against the stock C, **once at one thread and once at three**.
  Nothing but `node-diff` can do this: `graph-diff` runs with `no_alloc` and
  never computes, `backend-ops` compares with a tolerance, and `make port`
  never reaches the CPU kernels on a Metal machine. `OPS_DIFF_OUT=<dir>` keeps
  both full outputs, and `OPS_FULL=<label substring>` on the harness prints
  every element of the matching op.

  Four things about it are not interchangeable with the other diffs:

  - **Its reference is `llama.cpp.zmake`, not the Apple-clang CMake build,**
    and **both sides must be `--release=fast`.** `graph-diff` and `sched-diff`
    compare structure, which no compiler changes; this compares bits, and both
    a different compiler *and* a different optimization level move them.
    Measured: against Apple clang the baseline failed on `xielu` and Q6_K
    `mul_mat`, and at a mismatched optimization level on `rope` and
    `timestep_embedding` — all with `ops.cpp` unported and identical on both
    sides.
  - **The reference run writes its quantized blocks and the port replays
    them.** `ggml_quantize_chunk` produces different Q6_K bytes in the two
    builds (Q4_K and Q8_0 match), which is the accepted consequence of leaving
    the quantizers' fusions unnamed. Without sharing, the two `mul_mat`
    kernels get different weights and the gate stops being about the kernel.
  - **Activations run twice, at amplitude 1 and 30.** With `[-1, 1)` inputs
    only, a `softplus` cutoff moved from 20 to 2 escapes, because no input
    reaches either threshold. **A gate's input distribution is as much a part
    of it as its comparison.**

  - **Three threads, not just one.** At one thread every kernel's row split
    is the trivial one and flash attention's split-KV path never splits. Of
    eight faults injected into the `ops.cpp` port, two showed **only** at
    three threads.

  Negative-tested 12/12 across two rounds, including both faults that escaped
  `backend-ops`, after closing one escape by widening an input range.

- **`make repack-diff`** is the **only oracle the 36 interleaved `repack`
  kernels have.** They are unreachable from `test-backend-ops` and
  `ops-diff`, which never allocate a `CPU_REPACK` buffer, and from `make
  port`, which runs on Metal. `node-diff` reaches them and stops at the
  *first* divergent node — one kernel per run, with a model load and two
  decodes in between. This calls every gemv, gemm and `quantize_mat` next
  to the reference C++ **in one process** and compares on **bits**.

  It works by compiling the three reference translation units with all 66
  of their unmangled exports renamed to `ref_*` on the command line. That
  is the same `-D` rename `NOTES.md` records as unusable for *swapping* a
  translation unit, because it renames internal callers too — which is
  exactly what is wanted here.

  Measured, on the swap: `node-diff` named one node; this named twelve
  kernels and four distinct faults, three of them in shapes
  `ggml_repack_get_optimal_repack_type` cannot select on this target and
  so dead to every other gate. **Half of these kernels are exported and
  unreachable here** — the `8x8` and `4x8` shapes need AVX2, AVX-512, SVE
  or `__ARM_FEATURE_MATMUL_INT8` — the same position as the `_generic` dot
  products above.

  It also covers the eleven block converters, which are `static` in the C
  and have no exported name: those run **through the real buffer type**, so
  `ggml_repack_get_optimal_repack_type`, `init_tensor` and `set_tensor` are
  diffed too, and *which instantiation each side chose* is part of the
  comparison. 44 checks. Negative-tested 4/4 — one live kernel, one
  generic no other gate can reach, one converter, one wrong dispatch
  choice.

- **`make node-diff`** hashes the output of **every node** of a real model's
  decode — a prompt and one single-token step — and diffs ported against
  `llama.cpp.zmake`, at 1 and 4 threads. CPU device by default, `ARGS=--gpu`
  for Metal. It is `ops-diff`'s exactness on the graph the model actually
  runs: it found the `q5_K`, `tq1_0` and `tq2_0` dot-product epilogues, which
  had passed their goldens, as the first divergent `MUL_MAT` of a CPU-only
  Qwen3.5 decode, and the Zig `[*c]` miscompile below as scheduler copies
  with the wrong `op`. The first differing line names the op.

- **`make node-diff ARGS=--gpu` is the only gate that sees the Metal host layer**, and the CPU default cannot. Measured with an unconditional abort in `ggml_graph_optimize`: the `--gpu` run dies, the CPU run passes 2750 nodes. Metal node hashes are reproducible run to run and across builds, so this is an exact oracle and not merely a smoke test — verified before porting anything in `ggml-metal/`, not after.
- **`make tuning-diff`** sweeps **3.5M** flash-attention tuning lookups against the reference — every device id, KV type, head-size pair, both sides of every bucket edge, the family fallback. `tuning_table.zig` is 936 transcribed rows and no other gate can tell a dropped row from a kept one: `node-diff --gpu` reaches only the few keys one decode uses on one SKU. **It passed while testing nothing, twice over** — the archive held both definitions of each mangled symbol, and the harness had been compiled with the reference's `-D` renames so both sides called the reference. 5/5 after fixing both.
- **`make metal-diff`** asks the ported Metal buffer-type and device vtables the same questions as the reference's, with both registries in one process. **`node-diff ARGS=--gpu` sees almost none of `ggml-metal.cpp`**: measured, dropping the four `FLASH_ATTN_EXT` scratch terms from `get_alloc_size` gives 2048 bytes where the reference gives 331776 — a 162× under-allocation on a graph with 12 FA nodes — and node-diff passed it. Four of the five arms of that switch are never evaluated by a Qwen3.5 decode at all, which has no `CUMSUM`, `ARGSORT`, `TOP_K` or `MUL_MAT_ID`. 65 answers, negative-tested 6/6.

  Its `MUL_MAT_ID` probes **straddle `op_offload_min_batch_size`**, which defaults to 32. The first version used one tensor with `ne[1] = 2, ne[2] = 16` — both below the threshold — so `offload_op` reading the wrong dimension gave the same answer and the injection escaped. Fixing the shapes took it from 44 checks to 65.
- **`make struct-layout`** asks the C compiler whether the Metal structs and constants the port declares are right: `sizeof`, `alignof`, every `offsetof`, every field's width and name. **3,069 + 289 facts.** It covers `device_c.zig`'s hand-declared `ggml_metal_device_props` and `ggml_metal_pipeline_with_params`, the 112 `FC_*`/`OP_*`/`N_*`/`SZ_*` constants in `impl_c.zig`, and the 65 `ggml_metal_kargs_*` structs in `kargs.zig` — 936 fields the Metal kernels read **by offset**, where one wrong offset silently feeds a kernel a different number. `impl_c.zig` and `kargs.zig` are **generated** (`scripts/gen-kargs`), not transcribed; regenerate them when the pin moves. The struct is hand-written because importing `ggml-metal-device.h` would mean adding it to `impl.zig`'s single `cImport` and widening the `c` namespace every ported file sees — and a second `cImport` would produce two incompatible `*ggml_tensor`. Negative-tested 3/3, and it caught a real error on first use: three places said "22 fields" where there are 19.
- **`make abi-check`** calls the port's by-value-struct entry points **as a C caller does**. It is the gate for a toolchain bug rather than a translation one, and it exists because nine gates missed `ggml_bf16_to_fp32` returning 0.0 for every input — the row variant takes a pointer and works, Zig-internal callers never cross the C ABI, and nothing in a decode calls the scalar form. Negative-tested 4 of 11; the two zero inputs pass vacuously, which is why non-zero ones are in it.
- **`make sched-diff`** diffs the scheduler's backend assignments against the reference C, through two stub devices in `harness/sched_dump.c` that differ only in which ops they claim. **Nothing else can see those decisions.** Measured: turning off pass 4's `view_src` propagation leaves `parity-cli` at 6/6 and `graph-diff` at 131/131. Parity has to miss it — at `--temp 0` the sampler takes an argmax and `backend-ops` has shown Metal and CPU agree on all 21,093 op configurations, so moving an op between backends shifts the last bits and not the token. **Token parity measures *what* was computed, never *where*.** One fault still escapes even this: removing pass 2's CPU skip, because pass 3 re-derives the same answer.
- **`make parity-cli`** runs the actual `llamazig` binary against a reference C driver, both greedy. Covers what `parity-port` structurally cannot, because it lives in the binary rather than the library: argument parsing, tokenizer flags, the sampler chain, the decode loop. Negative-tested — a `--temp` that parses but never reaches the sampler fails all six prompts.

  **It is also the only gate that runs the library with Zig's safety checks on.** Every bit-exact gate above builds `--release=fast`, deliberately — they compare bits and the optimization level moves them — so none of them can see an illegal narrowing, an integer overflow or an out-of-bounds index. Measured: `ggml_backend_cpu_device_supports_op` narrowed `op->src[0]` with `impl.one` before the `GGML_OP_NONE` early return, and the scheduler asks about **leaf** tensors, which have no sources. Seven ReleaseFast gates passed; this one aborted before the first token. **A gate's optimization level is part of what it can see.**
- **`scripts/parity-port`** diffs generated tokens against `llama.cpp.zmake`. `--cpu` (`make parity-port-cpu`) loads the model on the CPU device alone, so the ported CPU kernels carry the whole forward pass — on a Metal machine they otherwise never do; it was 4/6 prompts divergent until the three epilogues above were fixed. End-to-end and **coarse** — measured, not guessed: doubling RoPE's `freq_base` passes, and so does halving the softmax scale. It catches structural faults (`ADD` as `SUB` fails all six prompts) and gross numeric ones (`freq_base = 1.0` fails all six). Never read a parity pass as "the numerics are right".

`scripts/port-links` is not on that list because it checks nothing about what the code *computes*. It keeps the port's map back to upstream honest, which is a maintenance gate rather than a correctness one.

`NOTES.md` has the full fault-injection table showing which gate catches what.

## Float contraction

**`zig cc` defaults to `-ffp-contract=on`, and so does Apple clang, but Zig
*language* code is strict.** The C fuses `a*b + c` into one FMA — one rounding
instead of two — and a literal Zig translation does not.

**The project handles this two different ways, and the difference is the
availability of an oracle.**

- **In the quantizers (`src/ggml/quants/`), the fusion is not named.** The
  sites are implicit in 5,000 lines of plain arithmetic, and guessing broke:
  fusing `makeQxQuants` fixed `q3_K` and simultaneously broke `q4_0`, the same
  function. `scripts/quants-golden` builds its reference with
  `-ffp-contract=off` and the ported code stays plain. Consequence: bytes our
  quantizers produce can differ in the last bit from a stock build — invisible
  to the deliverable, which reads model files and never writes one.
- **In the NEON kernels (`src/ggml/cpu/quants/arm/`), every fusion *is*
  named.** There the sites are explicit intrinsic calls — `vmlaq_n_f32`,
  `vfmaq_f32` — or a single `acc += a * b` statement per kernel, and
  `scripts/vecdot-prefix` pins the shape by asking the shipped kernel for every
  row prefix. Sixteen equations per kernel, not a guess. These are the live
  kernels, so their goldens are captured with contraction **on** and the port
  matches the shipped binary bit for bit.

- **In `cpu/unary_ops.zig`, `xielu` names its two.** `make ops-diff` is the
  oracle, and it found the site: the C fuses, strict Zig did not, and the
  result was 1 ULP out while `backend-ops` passed it. **Which** multiply
  clang fuses was measured, not reasoned about — two feed each add, and over
  200k inputs `fma(alpha_p*x, x, beta*x)` matches all 99,596 positive samples
  where fusing the `beta` term matches 91%. It is the **left** operand of the
  `+`.

- **In `cpu/repack/epilogue.zig`, one helper names the fusion for twenty
  sites.** Every generic `repack` gemv and gemm ends on `sumf[j] += sumi *
  GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d`, and strict Zig was 1
  ULP out in ten kernels. `make repack-diff` is the oracle. **Which**
  multiply was read straight off the reference's disassembly:
  `ref_ggml_gemv_iq4_nl_8x8_q8_0` at `-O2` emits `fmul s2, s2, s3` then
  `fmadd s2, s2, s1, s4` — the inner product rounds, the outer one fuses
  with the add. The **left** operand of the `+` again.

The rule that separates these: **name a fusion only when something can tell
you you got it wrong.** `scripts/vecdot-prefix`, `make ops-diff`, `make
repack-diff` and `make node-diff` are that something; for the quantizers
there is nothing, so they stay plain.

**One expression is fused; one *reduction loop* is not, entirely.** The
loop vectorizer turns `acc += a[i] * b[i]` over contiguous memory into a
strict in-order reduction: groups of four products are **rounded**
(`fmul.4s`) and added one lane at a time, and only the `t % 4` left over run
scalar and **fused** (`fmadd`). It does so only behind two runtime guards —
more than three iterations, and no loaded array starting within 64 bytes
below the store. `vec.zig`'s `vectorizedTail` and `ops/common.zig`'s
`strictLoopVectorized` encode it; measured from the reference's disassembly
and IR, not inferred. An elementwise `y[i] = a*b + c` stays fused when
vectorized (`fmla`). Which loops get vectorized is a cost-model decision, so
every such site is one the oracle confirmed.

**Clang folds a `sinf`/`cosf` pair on one argument into Apple's
`__sincosf_stret`**, which rounds differently from the separate calls (4.0%
of arguments for `sin`). The reference `ops.o` imports neither `sinf` nor
`cosf`. `ops/common.zig`'s `sinCos` calls the same routine.

**In the clang-contracted epilogue `sumf += d*x - m*y`, only the right-hand
side fuses**: `fmadd(d, x, -(m*y))`, then a plain add into `sumf`. And
`sumf += d * x` returned from a helper as `d * x` is *not* the same as the C's
one expression. Three dot products had exactly these bugs and passed their
goldens.

## Threads in ported code

`ggml-cpu.c` is the first ported file that runs threads, and two decisions
there are worth knowing before touching `src/ggml/cpu/threading.zig`:

- **The mutex and condition variable are pthreads', not Zig's.** Zig 0.16's
  `std.Io.Mutex` and `std.Io.Condition` take an `Io` on every lock and wait,
  and a backend entered through a C ABI has none to thread through. The C's
  `ggml_mutex_lock` expands to `pthread_mutex_lock` on this platform anyway, so
  calling it directly is both the workable choice and the closer translation.
- **Zig 0.16 has no `@fence`.** `ggml_barrier` and the polling threads both
  want a standalone `atomic_thread_fence(seq_cst)`. The port uses the C's own
  thread-sanitizer fallback — a `fetchAdd` of zero with `seq_cst` — on the same
  field the C names. The memory orderings around it are the C's, verbatim;
  they are not incidental and should not be "tidied".

**Zig 0.16 miscompiles `p.*.arr[i]` when `p` is a C pointer (`[*c]T`).** It
types the expression as the whole array and indexes in steps of the array's
size, in Debug and ReleaseFast alike. `dupTensorLayout`'s
`dup.*.nb[i] = tensor.*.nb[i]` copied 128 bytes and gave every scheduler input
copy its source's `op`, `flags` and `src` pointers. `p[0].arr[i]`, `*T`,
`[*]T` and `?*T` are all fine: narrow a `[*c]` with `impl.one` before indexing
an array field through it. A test written in the same style passes vacuously —
it reads both sides equally wrong.

**Zig can supply a C++-mangled symbol, where the signature is POD.**
`@export(&f, .{ .name = "_ZN17ggml_metal_tuning11fa_vec_pick…" })` resolves
and the ABI matches, which is what let `ggml-metal-tuning.cpp` be ported
alone instead of dragging the other three Metal C++ files with it. Take the
name from `nm` on the reference object, never construct it. `CLAUDE.md`'s
older note that `ggml-backend-dl.cpp` has "C++ linkage Zig cannot provide"
remains true of *that* file: its signatures name `std::filesystem::path`.

**Zig 0.16 miscompiles a small `extern struct` *received* by value across
the C ABI.** The callee sees **zeros**. Measured over ten shapes on
aarch64-macos: broken unless the size is a multiple of 4 with no padding —
1, 2, 4-with-padding, 6 bytes and a 2-byte struct forced to `align(4)` all
fail; one `u32`, one `f32`, 8 bytes, 12 bytes and 16 bytes all work.
Alignment is not the rule: 8 bytes at align 1 works and 2 bytes at align 4
fails. It is **one-directional** — Zig as the *caller* passes them
correctly.

This had broken `ggml_bf16_to_fp32` since `ggml.c` was ported: `ggml_bf16_t`
is `struct { uint16_t bits; }`, and the function returned 0.0 for every
input. **Nine gates missed it** because the row variant takes a pointer,
Zig-internal callers never cross the C ABI, and no decode calls the scalar
form. **Declare such a parameter as an integer of the same size and
`@bitCast`**, and add the entry point to `harness/abi_structs.c`.

**Zig reserves every `iN` and `uN` as an integer type name, and the C uses
half of them as loop indices.** Hit so far: `i0`, `i1`, `i2`, `i3`, `i11`,
`i12`, `i13` and `u12`. `src/ggml/cpu/mulmat.zig`, `cpu/repack/dispatch.zig`
and `metal/common.zig` rename them `j11`, `j12`, `j13` and so on, digit for
digit, so the index arithmetic can still be read against the C line by line.
Do not renumber them.

## Where the C's own answer is unspecified

Two places, both measured rather than assumed, both excluded from the goldens
rather than matched. When the reference is unspecified, reproducing it is not
the goal:

- `quantize_row_iq4_nl_ref` **reads uninitialized memory** for an all-zero
  block. Two consecutive calls in one process return different bytes. Ours
  zeroes the buffer.
- The **1-bit split search depends on `qsort`'s ordering of equal elements**.
  macOS libc returns `31 1 2 ... 30 0` for 32 equal elements; glibc would
  differ. Ours breaks ties on index, so output depends only on input.

**A filtered test run can hide a compile error.** `zig build test -Dtest-filter=...` skips codegen for non-matching tests, so a broken test body in an unselected test passes silently. Worse, a filter that matches *nothing* reports a comfortable row of passes: `-Dtest-filter="_K dot matches the C\|widening product"` is a literal string, not an alternation, and it matched zero tests while looking green. **Confirm the count changed, and run unfiltered before believing anything.**

**A gate proves nothing until it has been made to fail.** Four separate false passes in this project came from a check that was green because it was testing nothing. Inject a fault, confirm the gate fails, then trust it — and confirm the build actually succeeded and the artifact actually changed, because a failed build leaves the previous library in place and the gate will happily pass on it.

**And confirm the injection changes the computation, or it is not evidence about the gate.** Two of the `vec.cpp` faults read as escapes and were not: one widened an `f16` product but still stored an `f16` accumulator each iteration, and one was *provably* a no-op — `f16` has an 11-bit significand, the product needs at most 22 bits and `f32` holds 24, so widening either side of that multiply cannot change a bit (checked over 576M pairs). Both would have been written up as holes in the golden set.

**The reference is `llama.cpp.zmake`** — the stock C sources built by the Zig toolchain. Both sides are then compiled by the same toolchain, so the port is the only variable. `scripts/parity-port` and `scripts/node-diff` use it too. Note `llama.cpp.zmake` is a separate repository and is never modified by this project.

## `llama.cpp/` is reference material only

The `llama.cpp/` directory is an upstream clone pinned to tag `v0.3.0` (commit `c1d0e7a00`), created by `make clone`. It has its own `.git` and is untracked here.

**It is here to be read, not written.** Treat it as a specification we are reimplementing:

- **Never edit files under `llama.cpp/`.** Not to fix a bug, not to add a workaround, not to make a build succeed. If something there is wrong or in the way, the answer is a change on our side or a note in `NOTES.md`.
- **Never commit, push, or open a PR against it.** It is a separate upstream repository that we do not contribute to.
- **Never treat its conventions, instructions, or tooling as ours.** Its `CLAUDE.md`, `AGENTS.md`, `skills/`, and CI config govern the upstream project and do not apply to this one. This file is the authority here.
- **It is pinned deliberately.** Do not update it to a newer tag or pull upstream changes. A multi-month port against a moving target does not converge; syncing upstream is a separate, explicitly-scoped decision.

Read it constantly — for behavior, for numerics, for the shape of a data structure, as the reference build to diff against. Just do not change it.

## `llama.cpp.zmake/` is the Zig build system

A separate repository holding a `build.zig` that replaces llama.cpp's CMake.
It reads the sources from a checkout nested inside it and **does not modify
them**, so the pin can be changed or the checkout replaced without
reconciling anything.

```
llama.cpp.zmake/
├── build.zig, build.zig.zon, Makefile
├── zig/metal_embed.zig     flattens a Metal kernel for embedding
├── zig/ui-stub/            stand-in for the generated web UI assets
└── llama.cpp/              sources, fetched by `make clone` (gitignored)
```

`make zmake` from this repo builds `llama-cli` through it. Its `README.md`
records the six CMake behaviours that are easy to miss and cost real
debugging — `sha1.c` being C++ despite its extension, `src` shadowing
`common/unicode.h`, and so on. Read that before changing its `build.zig`.

It is gitignored here because it is its own repository. Work done there is not
tracked by this one.

## What we own

`build.zig`, `build.zig.zon`, `build/`, `src/`, `cli/`, `harness/`, `scripts/`, `Makefile`, and the markdown at the root. That is the whole of this project's code.

**`testcase/` is the one temporary directory, and it is meant to be deleted.**
It reproduces and diagnoses a toolchain failure, not anything this project
owns: Zig 0.16.0 cannot build its bundled libc++ against the macOS 27 SDK, so
no C++ can be linked, and `build.zig` plus four scripts link Apple's libc++
instead. `make -C testcase cxx20` passing is the signal that the toolchain is
fixed — at which point restore `.link_libcpp = true` at the seven sites, put
`-lc++` back in the scripts, and delete `testcase/`. Its `README.md` has the
full diagnosis and the revert list.

`LICENSE` (MIT) and `NOTICE` are load-bearing, not boilerplate: every file under `src/` is a translation of MIT-licensed C, so this is a derivative work and the ggml authors' copyright has to travel with it. Keep the per-file provenance comments — they are how a reader traces a translation back to its source.

- **`build/llamacpp.zig`** declares the reference tree's build graph, replacing its CMake for the macOS arm64 configuration. It compiles upstream sources unmodified.
- **`build/metal_embed.zig`** is a build-time tool that flattens one Metal kernel and its headers into a single MSL file plus an assembly stub. It replaces the `cat`/`sed` pipeline CMake uses for `GGML_METAL_EMBED_LIBRARY`.
- **`harness/smoke.zig`** loads a model through libllama's C ABI and generates. It is the only check that proves inference actually works, and it rehearses the C-ABI boundary that Stages 3 and 4 depend on.

Note `build/` holds build *sources*, not build output. CMake's output directory was moved to `cmake-build/` to free the name.

- **`cli/`** builds `llama-cli`, our replacement for upstream's binary of that name — same name so an existing command line runs unchanged against it, but it is our Zig binary, not upstream's linked against our library. `args.zig` parses upstream's flag surface, `session.zig` loads a model and generates, `chat.zig` renders chat templates, `main.zig` stays thin. `c.zig` holds the single `@cImport` of `llama.h` — two of them produce two incompatible `*llama_model`, and the error says only that `*cimport.struct_llama_model` will not coerce to `*cimport.struct_llama_model`. `upstream_flags.zig` is a generated inventory of every flag upstream accepts, used only to tell "you typed a real flag we have not got to yet" apart from "you made a typo".

**Chat templates are Jinja, and trust runs the other way from upstream's.** `zigjinja` is the one permitted dependency, wired into `cli/` only. Three things about it are worth knowing before touching `cli/chat.zig`:

- **Template literals are trusted; everything an expression emits is not.** Upstream marks strings that came *from* input (`common/jinja/README.md`); we mark the complement, so trust is the closed set the template author wrote rather than the open set we remembered to taint. A filter chain, a `set`, a loop variable — all reach output through an expression and all come out untrusted with nothing tracking them. `Session.tokenizeSegments` then sets `parse_special` per run. Measured: the attack text tokenizes to 7 tokens *including the real `<|im_start|>`* with it on, and 17 harmless ones with it off.
- **The marking rides on `Environment.finalize`, and the bytecode VM ignores it.** `applyFinalize` has exactly one call site, `compiler.zig:642`, on the AST path. `jinja.compiler.compile` picks the bytecode VM whenever the template allows it, which silently drops the marking and returns a prompt that looks correct and is entirely trusted. `chat.zig` calls `Compiler.compile(template, false)` for that reason. Found by tests failing, not by reading.
- **Marking is idempotent on purpose.** Qwen3.5 does `{% set c = render_content(...) %}{{ c }}`, so a macro's already-marked output passes through `mark` again; wrapping twice nested the sentinels and swallowed the macro body's own literal text into an untrusted run.

**The engine's strict parsing is partial.** `{{ unclosed`, a `for` with no `endfor` and `{{ 1 + }}` are reported as parse errors, but `{% bogusstatement %}` still renders as **empty output and reports success**. `chat.zig` rejects a render that produced nothing from a non-empty conversation; that guard is still the only thing standing between that class of broken template and a silently empty prompt. (Under `vibe-jinja` all five rendered empty and reported success; `zigjinja` v2.0.1 fixed the swallowed syntax errors.)

**`-cnv` holds a conversation, and re-renders the whole history every turn.** A template decides for itself where the system prompt goes and how a turn is framed, so appending to the previous render is not safe in general. `Session.generateTurn` diffs the new prompt against the decoded tokens and keeps their common prefix, so the re-render costs tokenization rather than decoding.

**`llama_memory_seq_rm` returns a bool, and on Qwen3.5 it returns false.** Recurrent and hybrid models cannot drop a range from the middle of the cache; the refusal is the only signal, and discarding it decodes the next turn on top of a stale cache at wrong positions. The symptom was the model emitting `<|im_end|>` as literal text mid-reply, which reads as a template bug and is not one. `generateTurn` checks the result and clears the cache instead. Found by a three-turn conversation, not by any unit test.

**Chat templating is opt-in, where upstream templates by default.** Raw completion is what `make port` and `make ref` diff against each other, so defaulting it on would change the one gate that covers the whole binary. Any chat flag turns it on; `--no-jinja` vetoes.

The greeting placeholders this repo started from (`src/base.zig`, `cli/client.zig`) are gone.

## Build state

All steps pass.

```sh
zig build            # the llamazig scaffold -> zig-out/bin/llama-cli (fast, ~0.2s)
zig build lib        # static library        -> zig-out/lib/libllamazig.a
zig build cli        # CLI                   -> zig-out/bin/llama-cli (links libllama)
zig build run        # run the CLI; args after `--`
zig build test       # unit tests; -Dtest-filter="..." narrows the run
zig build test-cli   # the CLI's tests only (~6s, vs ~1m for the library's)
zig build docs       # autodoc               -> zig-out/docs/

zig build reference  # llama.cpp -> zig-out/lib/{libggml.a, libllama.a}
zig build smoke      # load a model and generate; args after `--`
```

The stage-1 builds go through CMake and live in a separate tree:

```sh
make cmake           # fetch CMake into .tools/ -- not installed on this machine
make buildmacos      # control build, Apple clang -> cmake-build/apple/
make buildmacos-zig  # same sources via zig cc    -> cmake-build/zig/
make parity          # diff their token streams; the correctness gate
```

The reference build is deliberately **not** part of the default step — it is ~130k lines of C++ and takes ~16s in release, versus 0.2s for the scaffold.

```sh
zig build smoke --release=fast -- Qwen3.5-2B-Q4_K_M.gguf "The capital of France is" 24
```

**`llama.cpp/` is a submodule**, pinned at `c1d0e7a00` (v0.3.0) in
`.gitmodules`. `*.gguf` is ignored.

**`llama.cpp.zmake/` tracks the build system only** — `build.zig`,
`build.zig.zon`, `Makefile`, `zig/`. Its own `.gitignore` says so: *"the
llama.cpp sources, fetched by `make clone`. Not vendored here: this
repository is the build system, not a fork."* So the sources the gates
actually compile live in an **untracked** checkout at
`llama.cpp.zmake/llama.cpp`.

**Submoduling `llama.cpp.zmake` would therefore not pin the reference** — it
would pin the `build.zig` and leave the sources exactly as unpinned.
`scripts/check-reference-pin` does the job instead: every gate that compares
on bits (`ops-diff`, `node-diff`, `parity-port`) sources it and aborts unless
that checkout is at the same commit as our `llama.cpp` submodule. Without it
a drifted reference would compile a *different upstream* and the gate would
report the version gap as a porting bug. Negative-tested 3/3 by moving the
checkout back five commits.

## Toolchain

Zig **0.16.0**, declared as `minimum_zig_version` in `build.zig.zon` and what is installed. The code uses the 0.16 std APIs — `std.process.Init` as the `main` parameter, `std.Io` threaded explicitly through constructors, `std.Io.Writer` rather than the old writer interfaces. Do not fall back to pre-0.16 idioms.

`zigjinja` is a sibling repository checked out beside this one at `../zigjinja` — its own repository, already on Zig 0.16. `build.zig.zon` takes it as a path dependency. It is a fork of `gremlin-labs/vibe-jinja`, which this project depended on previously as a nested `vibe-jinja/` checkout.

The library must never gain external dependencies. This is a hard constraint, not a preference: the point of the port is that `zig build` alone produces the binary, and anything linking `libllamazig` inherits nothing.

**One exception has been granted, and only one:** `inferise/zigjinja` (pure Zig, MIT) for chat templates, confined to `cli/`. See `PLAN.md` Decisions 25 and 27. Pin an exact commit, never a branch. Do not add a second dependency without the same explicit grant.

## Commands

```sh
make build        # zig build
make dist         # zig build --release=fast
make clone        # clone llama.cpp and check out v0.3.0
make buildmacos   # reference CMake build of llama.cpp for arm64 macOS
make buildios     # reference CMake build for iOS
make qwen35       # download a Qwen3.5-2B GGUF (also qwen35xs, qwen35xl)
make port         # our ported CLI
make ref          # the same completion loop against the stock C libraries
make cli          # our CLI in interactive conversation mode
make ref-chat     # upstream's llama-cli REPL (different thing -- see below)
```

**`make port` and `make ref` must produce identical output.** They run the same
tokenize/decode/sample loop over the same prompt; `ref` links the stock C
libraries, `port` links the port. If they ever diverge, that is a real porting
bug and not a configuration difference.

Both take `MODEL`, `PROMPT`, `NPRED`, `TEMP`, `SEED`, and `ARGS`:

```sh
make port TEMP=0 PROMPT="def fibonacci(n):"   # greedy
make ref  TEMP=0 PROMPT="def fibonacci(n):"
make port SEED=$RANDOM                         # sample freely
```

**Both targets print the same timings line**, `[ Prompt: N t/s | Generation: N
t/s ]`, on stdout. `harness/raw_completion.c` runs the same warmup our CLI does
— throwaway decode, memory clear, `llama_perf_context_reset` — because without
it the first `llama_decode` carries Metal pipeline compilation and that lands in
the prompt counter.

**Do not read the two throughput figures as a comparison of the port against
the C.** Two confounds: the prompt figure swings by over 100 t/s run to run on
this machine, and `make ref` links the *Apple-clang* CMake libraries, so a
compiler difference is folded in. For a like-for-like build comparison use
`make parity`, which diffs two builds of the same C.

**`TEMP` and `SEED` are shared and `SEED` is fixed at 42**, where upstream's
default seed is random. A pair of targets meant to be diffed has to be
reproducible — and at temp 0.80 an unlucky seed sends a 2B model into a
repetition loop, which reads as a port bug and is not one.

**`make ref-chat` is a different thing and will not match.** It runs upstream's
`llama-cli`, which has **no raw-completion mode**: its `-st` is a single *turn
of a conversation*, so it applies the model's chat template, prints a banner and
a slash-command list, and answers as an assistant. The model is being asked a
different question. That is why `scripts/parity-cli` links its own C driver
(`harness/raw_completion.c`, shared with `make ref`) rather than calling
`llama-cli`.

`make ref-chat` needs `make zmake` first. The older `make cli` target runs
upstream's `llama-cli` from the *CMake* reference build against a model path
this repository does not contain; `make ref-chat` supersedes it.

`make buildmacos` builds the **reference implementation**, not our code. Its flag set is the Stage 1 target configuration: static, Metal on with the library embedded, Accelerate on, BLAS and OpenMP off, no curl. It is what a ported binary gets diffed against for correctness. When `build.zig` replaces CMake in Stage 2, these become the hardcoded macOS defaults rather than flags a caller passes.

Note that `make buildmacos` uses CMake's `-G Xcode` generator, which drives `xcodebuild` and therefore selects Apple clang regardless of `CMAKE_C_COMPILER`. Stage 1 — compiling llama.cpp with `zig cc` — will need `-G Ninja` instead. See `PLAN.md`.

## Code conventions

Followed consistently across `src/` and `cli/`; match them. They are the house style for ported code too, which matters: transliterated C and C++ will not naturally come out looking like this, and making it do so is part of the work.

- **Barrel per directory.** Each source directory has a `module.zig` re-exporting every public type. Siblings import through the barrel (`@import("module.zig")`), never each other directly. Consumers import the library as `@import("llamazig")`.
- **`src/root.zig` is docs-only.** Zig's autodoc cannot root a module at `module.zig`, so `root.zig` exists purely as the `docs` step's root and re-exports the barrel. Don't put code in it.
- **Allocator first, and the owner frees.** `init(allocator, ...)` stores the allocator as a field; helpers reach for `self.allocator` rather than taking one again. Whatever the struct's allocator produced, the struct's `deinit` frees. Slices handed to callers are borrowed and documented as invalid after the next call or `deinit`.
- **`init` returns an error union even when it cannot fail** — noted in the doc comment as "reserved for the lifecycle convention".
- **Doc comments carry a `Parameters:` list and a `Return:` line** on every public function, including what the return borrows and how long it stays valid. Struct-level `///` explains why the type exists, not just what it is.
- **Tests live at the bottom of the file they cover**, under a `// ---` / `// Unit Tests` banner, opening with `test { std.testing.refAllDecls(...) }`. Shared fixtures (`test_subject`, `test_greeting`) live in `src/module.zig` so every file asserts against the same values.
- **Executables stay thin.** `cli/main.zig` only unpacks `std.process.Init` and delegates to `cli.Client`, so all behavior stays reachable from unit tests.
- **`std.testing.allocator` is the leak check.** Tests that call an allocating method more than once exist specifically to prove the previous allocation was released.

## Porting notes

For when Stages 3 and 4 begin:

- **`zig translate-c` output is a draft to read, never a deliverable to commit.** It produces `c_int`-everywhere, pointer-arithmetic-heavy Zig that violates every convention above.
- **Correctness is measured against the reference build,** not against reading the code. Diff token output from the ported binary and the `make buildmacos` binary on the same model, prompt, and seed.
- **ggml's macro-heavy headers hold real logic in the preprocessor.** `ggml-impl.h`, `ggml-common.h`, and `simd-mappings.h` become `comptime` functions and generics — a rewrite, not a translation.
- **The Metal kernels under `ggml/src/ggml-metal/kernels/` are never ported.** They are Metal Shading Language compiled by the GPU driver; they travel as embedded data.
- **Zig has no ACLE intrinsics, and mostly does not need them.** `src/ggml/cpu/quants/arm/neon.zig` reproduces the 82 the NEON kernels use; all but one are a line of `@Vector` arithmetic. Three traps, each measured rather than reasoned about: **NEON integer arithmetic wraps** (so every op there uses `+%`, `*%`); **`vaddvq_f32` is a pairwise reduction**, not the ordered one `@reduce(.Add, ...)` gives; and **`vshlq_u8` shifts *right* on a negative amount**, which `q2_0` depends on.
- **`vdotq_s32` has no portable equivalent and does not need one.** It is integer, so the widening-multiply form is bit-identical rather than merely close. Measured cost: none detectable. Generation is 222-234 t/s either side of the port across three runs each, and the prompt figure varies by more than 100 t/s run to run — so the honest claim is "no measurable change", not a number. Metal carries the model; the quantized CPU path is barely on the critical path here.
- **Only a third of `arch/arm/quants.c` compiles here.** 1,556 of 4,319 lines; the rest is behind `__ARM_FEATURE_SVE` and `__ARM_FEATURE_MATMUL_INT8`. Measure before estimating a file in that directory — `zig cc -E` plus the linemarkers gives the live-line count.

## Building the reference tree

Facts established by getting `zig build reference` working. They are easy to
rediscover the hard way, so they are recorded here.

- **`zig cc`, `zig c++`, and Obj-C all work** against the macOS SDK under Zig 0.16.0, including `-framework Foundation` and libc++.
- **Do not pass `-fobjc-arc`.** Upstream's `.m` files use manual reference counting and bridge freely between `void *` and Obj-C object pointers; ARC rejects them outright.
- **`sanitize_c` must be `.off`.** Zig enables the C sanitizers in Debug, and upstream is not UBSan-clean — it aborts in `llama-graph.cpp` on `applying non-zero offset to null pointer`. Reference code compiles as shipped; we do not fix it.
- **The Metal shaders need no `xcrun metal`.** With `GGML_METAL_EMBED_LIBRARY`, the `.metal` *source* is flattened and `.incbin`-ed into a `__DATA,__ggml_metallib` section, and the driver compiles it at load time. `build/metal_embed.zig` does the flattening. Never enable the non-embed path: it requires `xcrun -sdk macosx metal`, which Zig can never replace.
- **CMake's `-G Xcode` cannot drive `zig cc`.** It shells out to `xcodebuild`, which picks Apple clang and ignores `CMAKE_C_COMPILER`. Moot now that `build.zig` owns the build.
- **Metal is reproducible run-to-run.** Three greedy runs on the same model and prompt produced token-identical output, so the token-parity gate is usable as written.
- **CMake is not on PATH and `/usr/local` is root-owned.** `make cmake` fetches 4.4.3 into `.tools/`. The Makefile's `CMAKE` variable prefers that over PATH. `ninja` is still absent, which is why the generator is `-G "Unix Makefiles"` — `make` is present.
- **The parity gate works and passes.** `scripts/parity` diffs the Apple-clang and zig-cc `llama-cli` binaries at `--temp 0` with a fixed seed. 4/4 prompts token-identical. Use it after any change that could affect arithmetic.
- **`zig cc` and Apple clang do not select the same ARM features.** CMake's probe through `zig cc` reports `HAVE_MATMUL_INT8 - Failed` and `HAVE_SVE - Failed` where Apple clang may not. Output is identical regardless, so this costs throughput rather than correctness — but the two builds are not running the same quant kernels. Revisit when Stage 3 ports `arch/arm/`.

## macOS SDK resolution

`build.zig` has an `Xcode` helper that resolves the Apple Silicon macOS SDK via `std.zig.system.darwin.getSdk` and adds it as a framework path to the library, and separately to the docs library. It runs only when `builtin.os.tag == .macos`. If linking against system frameworks fails, that resolution is where to look.
