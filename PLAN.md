# PLAN.md

Implementation plan for the llama.cpp -> Zig port described in `README.md`.

**Forward-looking only.** How each completed step was actually done — what it
cost, what each gate caught, and the false passes found by injection — is in
`NOTES.md`. This file says what is decided and what comes next.

Written against `llama.cpp` at tag **v0.3.0** (commit `c1d0e7a00`), Zig **0.16.0**, macOS arm64.

---

---

## Decisions

Settled in review. The plan below assumes all of these.

| # | Question | Decision |
|---|---|---|
| 1 | Target binary | **Port ggml + libllama only; write our own Zig CLI on top.** Decisions 16 and 20 refine this: the CLI is ours, but flag-compatible with upstream's. |
| 2 | `nlohmann::json` | **Moot** — Decision 1 removes it from the target set entirely. |
| 3 | Upstream tracking | **Freeze at v0.3.0.** Any future sync is a separate, explicitly-scoped project. |
| 4 | Correctness bar | **Token-identical output** vs. the reference build, fixed seed, `--temp 0`, canonical model `Qwen3.5-2B-Q4_K_M.gguf`. Logit-distance tolerance is the documented fallback if bit-exactness proves unreachable. Reference is `llama.cpp.zmake` — see Decision 17. |
| 5 | Model architectures | **Qwen3.5 first**, backfill the rest later. |
| 6 | Metal kernels | **Stay as MSL**, carried verbatim as embedded data. |
| 7 | Ported code location | **Under `src/`.** See "Layout" below. |
| 8 | Timeline | **Not a constraint.** Estimates removed from this plan; ordering and dependencies are what matter. |
| 9 | Layout | **Mirror upstream's component boundaries under `src/`**, each with its own barrel. See below. |
| 10 | ~~Chat templates~~ | ~~Built-in templates only (libllama's 174).~~ **Superseded by Decision 25**, which adds full Jinja via a dependency. |
| 11 | ~~CLI flag surface~~ | ~~Minimal and idiomatic. No flag-compatibility.~~ **Superseded by Decision 20**, which requires flag compatibility. |
| 12 | Metal | **Required throughout.** No CPU-only interval; the Obj-C host layer cannot be deferred. |
| 13 | Obj-C endgame | **Keep the two `.m` files as Obj-C**, compiled by `zig cc`. Decision 19 makes this the stated goal rather than a shortfall. |
| 14 | C ABI endgame | **Idiomatic Zig core with a thin `extern "C"` shim on top.** Confirmed by Decision 21. |
| 15 | Quantization kernels | **All 27**, not just the five the canonical model uses. |
| 16 | `llama-cli` | **Required, linking our libllama.** Not upstream's binary against upstream's library. |
| 17 | Parity reference | **`llama.cpp.zmake`.** Its stock-C build is the reference; CMake is no longer needed for the gate. |
| 18 | `llama.cpp.zmake` | **Do not touch it.** It is a reference repository, separate from the port. |
| 19 | The stated goal | **"Pure Zig apart from the Metal Objective-C host layer."** `README.md` updated to say so. |
| 20 | `llama-cli` | **Ours, but flag-compatible.** A Zig CLI linking our libllama, mimicking upstream's argument set for the features we have ported. A drop-in replacement for what we support. |
| 21 | C ABI endgame | **Confirmed:** idiomatic Zig core with a thin `extern "C"` shim on top. |
| 22 | Unsupported flags | **Reject** with a clear error and a non-zero exit. Never silently ignore. |
| 23 | Parity mode | **Raw completion.** Greedy, no chat template, prompt passed through verbatim. The gate does not have to hold for chat invocations. |
| 24 | Refactoring to idiomatic Zig | **Per subsystem**, as each stops being called from C++ — not all at once at the end. |
| 25 | Chat templates | **Full Jinja, via `inferise/zigjinja`** (a fork of `gremlin-labs/vibe-jinja`) as a single dependency. Supersedes Decision 10; permitted by the exception in Decision 27. |
| 26 | CLI timing | **Start building it now**, against the current libllama, rather than after the port completes. |
| 27 | The dependency rule | **Stands, with one named exception: the Jinja engine, now `zigjinja`.** Not narrowed, not dropped — explicitly excepted. Applied to `cli/`, so `libllamazig` stays dependency-free. |
| 28 | Order of work | **Finish `ggml.c` first, then the CLI.** The graph machinery is the hard part of what remains; the CLI is built while the first swap is being verified. |
| 29 | Op-level diagnostics | **Build `test-backend-ops` against our libraries**, since it is what localises a failed swap. Not core to the port; justified by validation alone. |
| 30 | Interactive mode | **One-shot first, interactive before the port is called complete.** Both, sequenced. |
| 31 | Graph diff | **Build it before the swap**, ahead of `test-backend-ops`, because it targets the constructor code that actually exists today. |
| 32 | Quantization coverage | **Round-trip unit tests are enough** for kernels no local model exercises. Treat any escape as a bug when it appears. |
| 33 | Graph diff scope | **Synthetic** — call each ported constructor directly, so coverage is breadth across all ported symbols rather than depth on one model's path. |
| 34 | Incremental swap | ~~Trial it.~~ **Trialled and inconclusive — parked.** The renames link, but the ported code does not execute. See `NOTES.md`. |
| 35 | Graph diff coverage | **The ~150 constructors that compute something**, extending only if a bug escapes. Pure getters stay covered by their golden-checksum unit tests. |
| 36 | Definition of complete | **ggml and libllama both ported, `llama-cli` running Qwen3.5 with token parity.** All 151 architectures and cross-platform are named follow-on milestones, not part of this push. |
| 37 | Licence | **MIT**, matching llama.cpp and `zigjinja`. |
| 38 | Copyright holder | **`Marc Lavergne`, provisionally.** Taken from the repository's git author config; revisit before any public release. |
| 40 | Order within Stage 4 | **Gated work first.** `ggml-backend-meta.cpp` and `ggml-opt.cpp` move to the end of Stage 4's ggml half, behind the `ggml-cpu/` C++. Neither is on the inference path, so neither has an end-to-end gate — and meta is **~96% unexecutable on this machine**: `ggml_backend_meta_device` is only built under `LLAMA_SPLIT_MODE_TENSOR`, our CLI never sets `split_mode`, and there is one GPU. The `ggml-cpu/` C++ is the opposite: covered by **`backend-ops`**, which runs CPU against Metal over 21,093 op configurations. Note that is the *only* gate that reaches it — measured, by putting an abort in `ggml_compute_forward_mul` and watching `make port` finish without hitting it. On a Metal machine the CPU op kernels never execute during inference, so `parity-cli` and `make port` cannot see them at all. One real gate still beats meta and opt's zero. Supersedes the ordering in "Order of work" step 1. |
| 39 | Threading primitives in ported code | **pthreads directly**, not `std.Io.Mutex` / `std.Io.Condition`. Zig 0.16 requires an `Io` on every call and a backend entered through a C ABI has none. `std.Thread.spawn` is still used for the workers. |

---

---

## 0. Scope, as settled

Decision 1 cuts the project roughly in half and removes every hard C++ dependency. Here is what is in and what is out.

### In scope

| Area | Lines | Language | Notes |
|---|---:|---|---|
| `ggml/src/*.c` | 14,983 | C11 | `ggml.c` (8,067), `ggml-quants.c` (5,667), `ggml-alloc.c` (1,249) |
| `ggml/src/*.cpp` | 8,417 | C++17 | backend, backend-reg, backend-meta, opt, gguf |
| `ggml/src/ggml-cpu/` C | 9,558 | C11 | `ggml-cpu.c`, `quants.c`, `arch/arm/quants.c` |
| `ggml/src/ggml-cpu/` C++ | 28,125 | C++17 | `ops.cpp` (12,021), `arch/arm/repack.cpp` (5,156), `repack.cpp` (4,836), `llamafile/sgemm.cpp` (4,164) |
| `ggml/src/ggml-metal/` host | 13,269 | C++17 + **Obj-C** | 10,178 C++ (`ggml-metal-ops.cpp` is 5,369) plus **3,091 Obj-C** in two `.m` files |
| `src/` top level (libllama) | 55,135 | C++17 | mmap, model-loader, arch, vocab, model, context, graph, batch, kv-cache, sampler, chat |
| `src/models/qwen35.cpp` | 645 | C++17 | plus whatever shared scaffolding it needs |
| `cli/` | new | **Zig** | written fresh, not ported |
| `ggml-metal/kernels/` | 12,319 | MSL | **not ported** — embedded verbatim |

Roughly **130,000 lines** to port: **24,541 C** in Stage 3, **105,591 C++ and Obj-C** in Stage 4.

Counts exclude the backends and CPU architectures the target flag set never compiles: `arch/x86`, `riscv`, `powerpc`, `s390`, `wasm`, `loongarch`, plus `kleidiai`, `amx`, and `spacemit`.

### Out of scope

| Area | Lines | Why |
|---|---:|---|
| `tools/server/` | ~21,400 | Decision 1 |
| `tools/cli/` | ~1,200 | Replaced by our own Zig CLI |
| `common/` | ~34,000 | Arg parsing, sampling glue, console — we write Zig equivalents |
| `common/jinja/` | ~6,300 | Only reachable from `common/` |
| `vendor/nlohmann`, `vendor/cpp-httplib` | — | Only reachable from `common/` and the server |
| `src/models/` minus qwen35 | ~37,900 | Decision 5; backfill later |

That is ~101,000 lines and all three of the hardest C++ dependencies, gone.

**Decision 20 keeps it out.** The `llama-cli` we ship is ours: Zig, linking our
libllama, with no `common/`, no server, and no vendored libraries. The ~57,000
lines stay out of the build as well as out of the port.

What Decision 20 *does* add is a compatibility obligation. The CLI must accept
upstream's argument set for the features we have ported, so that anyone with a
working `llama-cli` command line can run it against our binary unchanged. That
is a behavioural requirement on a few hundred lines of Zig, not a build
requirement on 57,000 lines of C++ — but it is not free either, since matching
flags means matching what those flags *do*. Decisions 22 and 23 settle the rest.

Note the balance this leaves: **ggml is the larger half of the port** (74,352 lines) and libllama the smaller (55,780). That inverts the intuition that the model code is where the work is.

### The verification that makes this work

**`ggml/` and `src/` contain no references to `nlohmann`, `cpp-httplib`, or `jinja`.** Checked directly against the v0.3.0 tree. The library we are porting is self-contained C and C++ with no third-party template metaprogramming anywhere in it. This is the single fact that makes Decision 1 a genuine simplification rather than a deferral.

Two consequences worth knowing up front:

- **Chat templates come for free.** `llama_chat_apply_template` is part of libllama's public API and `src/llama-chat.cpp` implements 174 built-in templates by heuristic string matching — explicitly *not* a Jinja parser. Our Zig CLI gets chat support without porting a template engine. Decision 25 then adds full Jinja on top.
- **Sampling comes for free.** `src/llama-sampler.cpp` (~4,400 lines) is inside libllama, so the sampler chain API is ours once libllama is ported. `common/sampling.cpp` is only a convenience wrapper, and we write our own.

The `.metal` kernels are excluded permanently. Metal Shading Language has no Zig equivalent and the GPU driver consumes MSL source directly.

### Layout (Decision 9)

Ported code mirrors upstream's component boundaries so provenance stays traceable and diffing against the reference stays cheap, with each component carrying its own barrel per the `CLAUDE.md` convention:

```
src/module.zig       barrel — re-exports the components below
src/ggml/            module.zig + ported ggml (the larger half)
src/llama/           module.zig + ported libllama
src/models/          module.zig + qwen35.zig
cli/                 the Zig CLI — stays outside src/, as it is now
```

### Deferred, not cancelled

Tracked here so they are not silently lost:

- ~~**`common/jinja/`** (6,349 lines).~~ No longer deferred, and no longer ported: Decision 25 takes `inferise/zigjinja` instead, a pure-Zig Jinja2 implementation. That removes 6,349 lines of C++ from the port entirely, at the cost of one dependency, explicitly excepted from the no-dependency rule by Decision 27.
- **`src/models/`** beyond `qwen35.cpp` (37,921 lines across 150 files). Decision 5.
- **Non-arm64 CPU backends** (`arch/x86` and friends). Needed for Stage 5's Linux targets.

---

---

## The test harness, and why it shapes everything

This deserves stating before the stages, because it determines their order.

**The reference is `llama.cpp.zmake`** (Decision 17): the stock C sources built
by the Zig toolchain, with no CMake anywhere. That is a better reference than
the Apple-clang CMake build it replaces, because both sides are then compiled by
the same toolchain and **the port becomes the only variable**. The old reference
conflated the two — CMake's ARM probe through `zig cc` reported
`HAVE_MATMUL_INT8` and `HAVE_SVE` as unavailable, so the two builds were not
even running the same quant kernels.

`llama.cpp.zmake` is a separate repository and **is not modified by this
project** (Decision 18). It is read, built, and diffed against. Nothing
port-specific goes into it.

**Two levels of gate:**

1. **Library level.** One driver linked against the ported libraries and against
   the reference libraries; diff the tokens. `scripts/parity-port` does this,
   and still points at the old CMake reference — repointing it at
   `llama.cpp.zmake` is outstanding work.
2. **Binary level.** Our `llama-cli` against the reference `llama-cli`, same
   model, prompt and seed. Decision 20 makes this possible: the same command
   line drives both, because our CLI mimics upstream's argument set.

   Decision 23 fixes the shape of that comparison: **raw completion only** —
   greedy sampling, no chat template, prompt passed through verbatim. It
   exercises everything the port can get wrong — tokenizer, graph construction,
   every kernel, the sampler — while removing prompt assembly, which is
   upstream's `common/` rather than ours.

   Decision 25 weakens the original reason for this restriction. With full
   Jinja available our CLI *can* in principle assemble the same prompt as
   upstream, so a chat-mode diff is no longer impossible — only unproven, since
   two independent Jinja implementations agreeing on every template in the wild
   is an assumption, not a fact. Raw completion stays the gate; a chat-mode diff
   becomes a possible extra check rather than an abandoned one.

**Why the C ABI is preserved throughout.** Every ported file keeps the symbols
and signatures of the translation unit it replaces, so the still-C++ code above
links against it unchanged. `build/llamacpp.zig` swaps a `.c` source for a
`.zig` one and nothing else moves. That is what makes each file an
independently verifiable, independently revertible step.

**The catch, and it is a real one.** A translation unit is all-or-nothing. A
single function cannot move to Zig without editing the C file, which is
forbidden — so a large file must be ported *completely* before anything about it
can be tested against the reference. `ggml.c` is 381 symbols; until the last one
lands, the ported code is exercised only by `zig build test-port`, which links
it against `ggml-quants.c` and `ggml-threading.cpp` alone. Unit tests are real,
but they are not parity.

---

---

## Stages 0-3 — done

The record of how each was done, what it cost, and what each gate caught is in
`NOTES.md`. What matters forward is the outcome and the gate:

| Stage | Outcome | Gate that proves it |
|---|---|---|
| **0** Repo hygiene | `build.zig` fixed, package renamed to `llamazig`, `-Dtest-filter` wired. The greeting scaffold (`src/base.zig`, `cli/client.zig`) is gone. | `zig build && zig build test` |
| **1** Build llama.cpp with `zig cc` | The whole tree compiles under the Zig toolchain. | `scripts/parity` — Apple-clang vs. zig-cc `llama-cli`, 4/4 prompts token-identical |
| **2** Replace CMake with `build.zig` | `llama.cpp.zmake` builds `llama-cli` and everything it links, no CMake anywhere. `build/llamacpp.zig` builds the same tree with ported files substituted. | `llama-cli` token-identical to the CMake build, 4/4 prompts |
| **3** Port the C | **No C compiles anywhere under `ggml/src/`.** Six translation units, 598 exported symbols, all swapped in and byte-verified. | `make validate`, `make graph-diff`, `make backend-ops`, `make probe`, `make parity-cli` |

Stage 3's six units: `ggml-alloc.c`, `ggml.c` (381 symbols), `ggml-quants.c`
(79), `ggml-cpu.c` (65), `ggml-cpu/quants.c` (45), `arch/arm/quants.c` (28).
`ggml_base_sources` and `ggml_cpu_c_sources` in `build/llamacpp.zig` are both
empty. `scripts/port-coverage` reports the counts.

**Two Stage 0 items remain**, both cheap and both still open:

- **Pin `llama.cpp/` as a submodule at v0.3.0** (Decision 3). It is an
  untracked nested clone. `main` has no commits yet, so pinning is free now and
  expensive later.
- **`scripts/parity-port` still points at the retired Apple-clang CMake
  reference** and needs repointing at `llama.cpp.zmake` (Decision 17).

---

## Stage 4 — Port the C++ to Zig

**105,591 lines** of C++17 and Obj-C after Decision 1 and Decision 5:
`ggml/src/*.cpp` (8,417), `ggml-cpu/` C++ (28,125), the Metal host layer
(13,269), `src/llama-*.cpp` (55,135), and `src/models/qwen35.cpp` (645).

This is the bulk of the project. Four files carry a quarter of it:
`ggml-cpu/ops.cpp` (12,021), `ggml-metal-ops.cpp` (5,369),
`arch/arm/repack.cpp` (5,156), `ggml-cpu/repack.cpp` (4,836).

No Boost, no `nlohmann::json`, no HTTP library, no Jinja engine — Decision 1
removed all of them. What remains is STL containers, `std::function`, RAII,
exceptions, templates, and virtual dispatch.

| C++ | Zig |
|---|---|
| `std::vector<T>` | `std.ArrayList(T)` — with explicit allocator plumbing |
| `std::string` | `[]const u8` + `std.ArrayList(u8)`; ownership decided per site |
| `std::unordered_map` | `std.HashMap` / `std.StringHashMap` |
| RAII destructors | explicit `deinit` + `defer` — **every call site changes** |
| exceptions | error unions — **every signature changes** |
| virtual dispatch | tagged unions, or vtable structs (`std.Io`-style) |
| templates | `comptime` generics — usually cleaner |

### The measurement that settles the method

The open question entering this stage was how much C++ linkage actually crosses
a translation-unit boundary — because anything that does cannot be replaced by
Zig, which emits no vtables, no RTTI, and no mangled names.

**Measured, not assumed.** Every exported symbol of every ggml C++ object was
intersected with the undefined symbols of every other object in `libggml.a` and
`libllama.a`. The result:

| Group | TUs | TUs whose external contract is **pure C ABI** |
|---|---:|---|
| `ggml/src/*.cpp` | 8 | 7 of 8 |
| `ggml-cpu/*.cpp` | 11 | 9 of 11 |
| `ggml-metal/*.cpp` | 5 | 4 of 5 |

**So the swap mechanism from Stage 3 carries over unchanged for the large
majority.** A C++ file exports thousands of mangled symbols — `gguf.cpp` alone
exports 2,002 — but they are template instantiations and inline functions,
emitted weakly into every TU that needs them. Almost none are *reached from
outside*. `gguf.cpp`'s real contract is **61 C-ABI functions** — measured at the pin
when it was ported; an earlier count of 44 was wrong.
`ggml-backend-meta.cpp`'s 2,495 lines present **4**.

**Four TUs are the exception, and each names its own constraint:**

| TU | What crosses the boundary | Consequence |
|---|---|---|
| `ggml-backend-dl.cpp` | 3 functions, one taking `std::filesystem::path &` | Only `ggml-backend-reg.cpp` calls them. **Port the two together**; the Zig registry calls `dlopen` directly and the file disappears. |
| `ggml-cpu/traits.cpp` | **vtables and RTTI** for `ggml::cpu::extra_buffer_type` and `tensor_traits` | Real virtual dispatch across TUs. Cannot move until every class deriving from them moves — `repack.cpp`, `arch/arm/repack.cpp`, `amx.cpp`. **One cluster, ported together.** |
| `ggml-cpu/ggml-cpu.cpp` | `ggml_backend_cpu_get_extra_buffer_types()`, returning `std::vector` | Same cluster as above. |
| `ggml-metal/ggml-metal-tuning.cpp` | 7 functions in a C++ namespace | Called only by `ggml-metal-ops.cpp`. Small; port the pair together. |

That is the whole of the C++-linkage problem in ggml. It is three clusters, not
a pervasive condition, and it was the thing worth knowing before starting.

### The contract for a C++ translation unit

`scripts/port-coverage` derives a C file's contract by compiling it and reading
its exported symbols. For a C++ file the same trick works with one filter:
**the contract is the unmangled exports**, plus any mangled symbol another TU
actually references. The mangled remainder is internal and is not reproduced.

### Order of work

Bottom-up, so each layer lands on ported foundations.

**1. `ggml/src/*.cpp`** (8,417) — closest to C, minimal STL, and now measured:

| File | Lines | External C-ABI symbols | Note |
|---|---:|---:|---|
| `ggml-threading.cpp` | 12 | 2 + 1 global | **Ported.** A `std::mutex` and two wrappers. The warm-up. |
| `ggml.cpp` | 26 | **0** | A static initializer installing a `std::terminate` handler. Nothing links to it. |
| `ggml-backend-dl.cpp` | 48 | 3, **mangled** | **Gone.** Left the build with the registry, its only caller. |
| `ggml-backend-reg.cpp` | 593 | 8 | **Ported.** Backend registration and discovery. |
| `ggml-opt.cpp` | 1,094 | 9 | Training/optimizer API. Not on the inference path. |
| `gguf.cpp` | 1,706 | 61 | **Ported.** The GGUF reader. `std::vector`/`std::string` throughout. |
| `ggml-backend.cpp` | 2,443 | 102 | **Ported.** Buffers and `ggml_backend_sched`. Split across `backend.zig` and `backend_sched.zig`. |
| `ggml-backend-meta.cpp` | 2,495 | 4 | |

`ggml.cpp` is the one file with no Zig answer: `std::get_terminate` and
`std::set_terminate` are C++ runtime facilities, and its only effect is to print
a ggml backtrace when a C++ exception escapes. It stays until the C++ that can
throw is gone, then it is dropped rather than ported. Recorded here so that is a
decision and not an oversight.

**2. `ggml-cpu/` C++** (28,125 raw, **15,488 live** on this target) —

`binary-ops.cpp`, `unary-ops.cpp`, `vec.cpp` and `ops.cpp` are **ported** —
`src/ggml/cpu/binary_ops.zig` (4 symbols), `src/ggml/cpu/unary_ops.zig` (23),
`src/ggml/cpu/vec.zig` (10) and `src/ggml/cpu/ops/` (88). Templates map
cleanly onto `comptime`, as predicted.

**On a Metal machine the CPU op kernels never run during inference.**
Measured: an `impl.abort` at the top of `ggml_compute_forward_mul` lets
`make port` finish without hitting it, so `parity-cli`, `make port` and
`graph-diff` cannot see this group. Its gates are `make ops-diff` (bits, fixed
cases), `make node-diff` (bits, a real model's graph on the CPU device) and
`make parity-port-cpu` (tokens, CPU alone), plus `backend-ops` for crashes and
gross errors. `make validate` runs **none** of them; run all four after every
file here.

**`vec.cpp` got its gate built first, and that was the right call.** Its three
dot products are *accumulating* reductions where summation order and FP
contraction decide the last bits, and nothing in the project could see them.
`scripts/vec-golden` + `harness/vec_golden.c` + `src/ggml/cpu/vec_testing.zig`
now capture the exact bits the C produces, compared on **bits**, contraction
**on**.

It paid for itself immediately: with the first six input patterns, replacing
the **pairwise** `vaddvq_f32` with an ordered `@reduce(.Add, ...)` passed
everything — the exact trap `CLAUDE.md` warns about, invisible to the gate
built to catch it. A seventh pattern, `skewed`, keys magnitude on `i % 4`,
which is the lane index, and closes it. See `NOTES.md`.

**The general lesson for the rest of this group: build the oracle before the
port, not after.** A port written first will be checked against whatever gate
happens to exist, and a gate written second gets tuned, however unconsciously,
to pass the code already written.

### `ops.cpp`: done

**88 of 88 symbols, swapped in**, split by op family across 19 files under
`src/ggml/cpu/ops/` — the split follows the C's own `// ggml_compute_forward_xxx`
banners. Ported in one sitting by seven parallel workers, one op family group
each, against an oracle extended *before* the swap: `make ops-diff` went from
109 cases to ~300, at one thread **and three**, and covers every family. See
`NOTES.md`, "Porting `ggml-cpu/ops.cpp`".

**`ggml_compute_forward_mul_mat` is not among the 88.** It came with
`ggml-cpu.c` and is `src/ggml/cpu/mulmat.zig`. Check the contract before
assuming a `ggml_compute_forward_*` belongs to a file.

What the swap found, none of which any gate before it could see:

- **Two compiler behaviours the port has to reproduce**, both in
  `src/ggml/cpu/ops/common.zig`: clang folds a `sinf`/`cosf` pair into
  Apple's `__sincosf_stret`, which rounds differently; and the loop
  vectorizer turns a plain `acc += a[i]*b[i]` reduction into rounded groups
  of four and a fused remainder, behind runtime alias guards.
- **Four bugs in code ported in earlier sessions**, all found end to end:
  `ggml_vec_dot_f32`'s scalar tail (the same vectorizer behaviour), and the
  `q5_K`, `tq1_0` and `tq2_0` dot-product epilogues. Their goldens passed all
  four.
- **A Zig 0.16 miscompile**: `p.*.arr[i]` through a `[*c]` indexes in steps
  of the whole array. It corrupted every scheduler input copy.

Two new gates came out of it: `make node-diff`, every node of a real model's
decode on bits, and `make parity-port-cpu`, which loads the model on the CPU
alone. On a Metal machine they and `ops-diff` are the only things that ever
run the CPU kernels.

**Measure live lines, not raw ones, before estimating anything in this
group.** `zig cc -E` plus the linemarkers gives the count that compiles on this
target, and the gap is large: `llamafile/sgemm.cpp` is 4,164 raw but **391
live** — the templated x86 bulk is behind `__AVX__`/`__AVX512F__`.
`amx/mmq.cpp` (2,511), `amx/amx.cpp` (249) and `hbm.cpp` (55) are **entirely
dead** here, 2,815 lines that need no port at all. `arch/arm/repack.cpp` is
5,156 raw and 2,528 live; `repack.cpp` 4,836 and 3,164.

That shrinks the vtable cluster: `amx.cpp` has 0 live lines, so `traits.cpp`
+ `repack.cpp` + `arch/arm/repack.cpp` is **three** files and 5,720 live
lines, plus `ggml-cpu.cpp`, whose `ggml_backend_cpu_get_extra_buffer_types()`
returns a `std::vector` and so moves with them.

**3. `ggml-metal` host layer** (13,269). Decision 13 settles the hard part: the
**3,091 lines of Obj-C** in `ggml-metal-device.m` (2,352) and
`ggml-metal-context.m` (739) **stay as Obj-C**, compiled by `zig cc`. Only the
10,178 lines of C++ around them are ported. `ggml-metal-tuning.cpp` pairs with
`ggml-metal-ops.cpp`.

**4. `src/llama-*.cpp` core** (55,135) — mmap, model-loader, arch, vocab, model,
context, graph, batch, kv-cache, sampler, chat. This is libllama and the actual
prize. Sub-order matters: loader and vocab first (testable against known GGUF
metadata without running inference), then graph and context, then kv-cache.

**5. `src/models/qwen35.cpp`** (645) plus its shared scaffolding.

**6. The Zig CLI.** Decision 26 pulled this forward and it is built — see
`NOTES.md`, "The CLI, built now". By the end of Stage 4 it links a fully ported
libllama rather than a partly ported one, but it is the same binary.

### What the CLI can and cannot support

The dividing line is not what upstream's CLI does; it is **what lives in
libllama versus what lives in `common/`**. libllama is being ported in full, so
anything implemented there is available to our CLI for the cost of the argument
handling and a few API calls.

**In scope, because libllama implements it** — sampling in all its forms (41
public entry points), GBNF grammars, **LoRA and aLoRA adapters**, session state
save and restore, and the 174 built-in chat templates.

**In scope via one dependency** — full Jinja chat templates, using
`inferise/zigjinja` (Decision 25). Built; see `NOTES.md`, "Chat templates".

**Out of scope, because the implementation is in a component we do not build:**

| Feature | Lives in |
|---|---|
| `--json-schema` | `common/json-schema-to-grammar.cpp` |
| `-hf`, model download | `common/download.cpp` |
| Speculative decoding | `common/speculative.cpp` |
| Multimodal, `--mmproj` | `tools/mtmd/` |
| Server flags | `tools/server/` |
| `--rpc` | `ggml/src/ggml-rpc/` |

LoRA is *not* in that list, though an earlier draft had it there:
`src/llama-adapter.cpp` is part of libllama and `llama.h` exposes the full
adapter API. Only the `--lora` argument plumbing lives in `common/arg.cpp`.

Keep the upstream harness as a test-only build target after cutover. It is the
only independent check that libllama still behaves, and it costs nothing but
build time.

**Gate:** `zig build` produces `llama-cli`, which generates token-identical
output to the reference on the canonical model, prompt, and seed.

---

## Refactoring to idiomatic Zig

The porting method forces C-shaped code. Every ported file exports the symbols
and signatures of the translation unit it replaces, because that is what lets a
half-ported library link at all. By the end of Stage 4 the "Zig core" would
otherwise be several thousand lines written to a C ABI: out-params, opaque
pointers, error codes.

**Decision 24: refactor per subsystem, as each stops being called from C++.**
Once nothing outside ggml calls into it, ggml's internals can be reworked to
error unions, slices and real types while libllama is still being ported.
Decision 21's `extern "C"` shim then sits at the boundary, keeping the C ABI for
outside callers.

The alternative — one large refactor at the end — concentrates the risk against
a codebase whose only gate is end-to-end token parity. Per-subsystem spreads it,
at the cost of some code being written twice. That cost is accepted.

Practical consequence: **do not over-invest in making the C-shaped intermediate
pleasant.** It is scaffolding for one subsystem's lifetime.

---

---

## Stage 5 — Cross-platform

Targets: macOS, iOS, Linux arm64, Linux x86_64.

Zig makes the *build* side nearly free once `build.zig` owns everything — `-Dtarget=aarch64-linux-gnu` and so on. The work is backend coverage. Decision 12 keeps Metal on the critical path for macOS and iOS, but Linux has no Metal, so this is where a CPU-only path first has to stand on its own, and where `ggml-cpu/arch/x86/` (AVX2/AVX-512) becomes a second SIMD port that Stage 3 deliberately skipped.

iOS additionally needs the `buildios` flag set and the increased-memory-limit entitlement already noted in the `Makefile`.

Backfilling `src/models/` beyond Qwen3.5 (Decision 5) belongs here or later — it is repetitive graph-building code and parallelizes well once the first architecture proves the pattern.

---

---

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| ARM NEON quant kernels resist translation | Silent numerical drift, or a perf cliff | Per-block round-trip tests; keep the C version behind a build flag until the Zig one matches |
| ~~Obj-C Metal host layer~~ | Retired by Decision 13: the two `.m` files stay Obj-C | The cost is that the binary is not pure Zig — a goal question, not a risk |
| A 381-symbol translation unit has no parity check until it is complete | A systematic error in `ggml.c` stays invisible for the whole port | Unit tests per file, plus `scripts/port-coverage` to keep the endpoint visible; accept that the real gate lands late |
| The diagnostic tool does not match the risk | `test-backend-ops` compares two backends over one graph, so it cannot see a wrong graph | Retired: `make graph-diff` covers constructors, and `make backend-ops` the kernels. See `NOTES.md` |
| Two build descriptions drift apart | The port builds something subtly different from the reference | Only one is ours to change; `llama.cpp.zmake` is read-only (Decision 18) |
| `zigjinja` is an external dependency | A supply-chain and maintenance exposure the rule otherwise forbids | Pin an exact commit, never a branch; confine it to `cli/` so `libllamazig` stays dependency-free; vendor it under its MIT licence if it goes unmaintained |
| Two Jinja implementations disagree on a real template | Chat output differs from upstream for reasons unrelated to the port | The parity gate is raw-completion (Decision 23), so this cannot mask a port bug — it is a correctness risk for users, not for the gate |
| Bit-exactness unreachable across compilers | Decision 4's gate becomes unusable | Detect early — Stage 1 tests exactly this before any port work. Fall back to logit-distance tolerance and document it |
| Parity harness rots | The whole verification story collapses | It is a build target, not a script someone remembers to run. Keep it in `zig build` |
| Metal output not reproducible run-to-run | Decision 4's token-parity gate becomes meaningless | Test it in Stage 1 step 4, before any port work depends on it |
| Zig 0.16 is pre-1.0; std churns | Breakage on toolchain bumps | Pin the exact version in `build.zig.zon`; bump deliberately, never incidentally |
| Performance regression vs. hand-tuned C++ | Port is "done" but unusable | Tokens/sec in the harness from Stage 1; treat a regression as a bug |
| Qwen-only hides architecture-general bugs | Ported libllama silently overfits to one model | Backfill a second, structurally different architecture before declaring libllama done |

---

---

## What "complete" means

Decision 36 fixes the endpoint, because three earlier decisions referred to it
without defining it:

**Complete = ggml and libllama both ported to Zig, and `llama-cli` generating
token-identical output to the reference on Qwen3.5.**

That is the first point at which the README's claim is true of a real model
end to end. What it includes, and one part of it is easy to undercount:

- All of Stage 3 — `ggml.c`, `ggml-quants.c`, `ggml-cpu.c`, `arch/arm/quants.c`.
- All of Stage 4's ggml half, **including the ~10,178 lines of C++ in the Metal
  backend**. "ggml ported" means the backend too; only the 3,091 lines of
  Objective-C are excepted (Decision 13).
- libllama, and `src/models/qwen35.cpp`.
- The CLI, one-shot *and* interactive (Decision 30).

**Explicitly not included**, and tracked as follow-on milestones:

| Milestone | What it adds |
|---|---|
| All architectures | The other 150 files under `src/models/` (~37,900 lines) |
| Cross-platform | Stage 5: Linux and the x86 SIMD kernels Stage 3 skipped |

---

---

## What happens next

1. ~~`ggml-threading.cpp`~~ — **done.** The first C++ translation unit, and the
   proof that the Stage 3 swap mechanism carries over: `src/ggml/threading.zig`,
   3 symbols, `ggml_base_cxx_sources` one file shorter.
2. ~~`ggml-backend-reg.cpp` + `ggml-backend-dl.cpp`~~ — **done.**
   `src/ggml/backend_reg.zig`, 16 symbols, both files out of the build. The
   registry was the only caller of the three `dl_*` functions, so the C++-linkage
   problem dissolved exactly as the measurement predicted.
3. ~~`gguf.cpp`~~ — **done.** `src/ggml/gguf.zig`, 61 symbols (not the 44
   recorded here before: measured at the pin). The two the C++ exports beyond
   those, `gguf_type_size` and `gguf_write_to_buf`, are declared in
   `ggml-impl.h` inside `#ifdef __cplusplus` for `tests/test-gguf.cpp`, which
   this project does not build, so they are mangled and outside the contract.
   Twelve unit tests, eight fault injections all caught — three only after the
   tests were strengthened. See `NOTES.md`.

   **Its end-to-end gate is still owed**: `make port` cannot run while this
   machine cannot link C++ (see `NOTES.md`, "Blocked: every link-time gate").
4. ~~`ggml-backend.cpp`~~ — **done.** `src/ggml/backend.zig` (the vtable
   dispatch) and `src/ggml/backend_sched.zig` (the scheduler), 102 symbols —
   not the 82 recorded here before. Every gate green with it in the path:
   `make port`, `parity-cli` 6/6, `graph-diff` 131 nodes, `backend-ops`
   21,093 configurations, `probe`. See `NOTES.md`.

   Worth recording for the files still to come: it contains **no `throw`, no
   `catch`, no `std::string`, no templates and no virtual functions** — the
   difficulty was the five-pass assignment algorithm, not the C++.
5. ~~`ggml-backend-meta.cpp` and `ggml-opt.cpp`~~ — **deferred to the end of
   the ggml half by Decision 40.** Measured counts: meta is **8** symbols over
   2,495 lines (not 4), opt is **37** over 1,094 (not 9).

   Neither is on the inference path, so neither has an end-to-end gate. Worse,
   meta is **~96% unexecutable here**: `ggml_backend_meta_device` is only
   constructed under `LLAMA_SPLIT_MODE_TENSOR` (`llama.cpp:176, 217`), our CLI
   never sets `split_mode`, and this machine has one GPU. `alloc.zig:1318`
   guards its allocator path behind `buft_is_meta`, always false. Only the
   three `is_meta` predicates run, and they always return false. Porting 2,400
   lines of multi-GPU tensor-parallel machinery that cannot be executed or
   gated is the worst value in the project; it waits until it is the only
   thing left.

   `ggml-opt.cpp` additionally holds a `std::mt19937` whose `std::shuffle`
   order is **implementation-defined** — `std::uniform_int_distribution` is
   unspecified, so libc++ and libstdc++ differ. That is the third instance of
   the category in `CLAUDE.md`, "Where the C's own answer is unspecified", and
   it gets the same treatment: implement MT19937 faithfully, document the
   shuffle divergence, do not chase it.

**Every symbol count this plan carried for Stage 4's first group was low** —
gguf 44 vs 61, backend 82 vs 102, meta 4 vs 8, opt 9 vs 37. Measure the
contract with `port-coverage` before estimating; the original survey figures
are not usable as estimates.
6. **`ggml.cpp`** (26 lines, 0 symbols) — dropped rather than ported, once no
   C++ in the build can throw. See the note under "Order of work" above.
7. ~~`ggml-cpu/binary-ops.cpp`, `unary-ops.cpp`, `vec.cpp`, `ops.cpp`~~ —
   **done.** 125 symbols between them. `ops.cpp` is the largest translation
   unit in ggml; see "`ops.cpp`: done" above.
8. **`llamafile/sgemm.cpp`** (391 live lines). Templated NEON matmul kernels,
   called from `mulmat.zig`. Self-contained, and its gate already exists:
   `ops-diff` runs f32 and quantized `mul_mat` at shapes that take it.
9. **The vtable cluster** — `traits.cpp`, `repack.cpp`, `arch/arm/repack.cpp`
   and `ggml-cpu.cpp`, together. Virtual dispatch across translation units;
   the C++ class hierarchy becomes a Zig vtable struct, and every class
   deriving from `extra_buffer_type` moves at once. `repack` reorders
   `Q4_0`/`Q4_K`/`IQ4_NL` weights into interleaved blocks, so `node-diff` on
   a model with those types is its gate, alongside `ops-diff`.

That empties `ggml-cpu/`. Then the Metal host layer, then libllama.

**One open item carried from Stage 0**: pinning `llama.cpp/` as a submodule.
Repointing `scripts/parity-port` at `llama.cpp.zmake` is done.

**Tidy-up the `ops.cpp` port left behind**, deliberately not done in the
sitting that ported it (Decision 24: do not over-invest in the C-shaped
intermediate). Seven workers each wrote their own small helpers — a byte
offset `at`, `std::min`/`std::max` stand-ins that keep the C++'s NaN
behaviour, the NEON reduce — so there are several copies of each. Fold them
into `ops/common.zig` when the files are next touched.

**Two gaps in the chat-template work**, carried from `NOTES.md`:

- **A `chat-parity` gate.** `common/jinja` is seven self-contained `.cpp` files,
  so upstream's engine can render the same template and the two outputs be
  diffed. Nothing yet checks that two independent Jinja implementations agree
  on the templates in the wild, which is an assumption rather than a fact.
- **No gate covers the conversation loop.** `parity-cli` runs one-shot prompts;
  multi-turn behaviour, KV prefix reuse and the hybrid-cache fallback are
  checked by hand only.

---

## Questions

*None open.* Thirty-nine decisions cover scope, method, verification, the
endpoint, and licensing. The question Stage 4 opened with — how much C++
linkage actually crosses a translation-unit boundary — has been measured rather
than guessed, and the answer is in "The measurement that settles the method"
above.

New questions get appended here as they arise. Answered ones move into the
decisions table rather than accumulating, so this section stays a list of what
is genuinely undecided.
