# PLAN.md

Implementation plan for the llama.cpp -> Zig port described in `README.md`.

Written against `llama.cpp` at tag **v0.3.0** (commit `c1d0e7a00`), Zig **0.16.0**, macOS arm64.

---

## Decisions

Settled in review. The plan below assumes all of these.

| # | Question | Decision |
|---|---|---|
| 1 | Target binary | **Port ggml + libllama only; write a new thin Zig CLI on top.** Not a drop-in `llama-cli` clone. |
| 2 | `nlohmann::json` | **Moot** — Q1 removes it from the target set entirely. |
| 3 | Upstream tracking | **Freeze at v0.3.0.** Any future sync is a separate, explicitly-scoped project. |
| 4 | Correctness bar | **Token-identical output** vs. the reference build, fixed seed, `--temp 0`, canonical model `Qwen3.5-2B-Q4_K_M.gguf`. Logit-distance tolerance is the documented fallback if bit-exactness proves unreachable. |
| 5 | Model architectures | **Qwen3.5 first**, backfill the rest later. |
| 6 | Metal kernels | **Stay as MSL**, carried verbatim as embedded data. |
| 7 | Ported code location | **Under `src/`.** See Q9 below for the layout detail. |
| 8 | Timeline | **Not a constraint.** Estimates removed from this plan; ordering and dependencies are what matter. |
| 9 | Layout | **Mirror upstream's component boundaries under `src/`**, each with its own barrel. See below. |
| 10 | Chat templates | **Built-in templates only** (libllama's 174). Porting `common/jinja/` is deferred, not cancelled. |
| 11 | CLI flag surface | **Minimal and idiomatic.** No flag-compatibility with `llama-cli`. |
| 12 | Metal | **Required throughout.** No CPU-only interval; the Obj-C host layer cannot be deferred. |

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

Note the balance this leaves: **ggml is the larger half of the port** (74,352 lines) and libllama the smaller (55,780). That inverts the intuition that the model code is where the work is.

### The verification that makes this work

**`ggml/` and `src/` contain no references to `nlohmann`, `cpp-httplib`, or `jinja`.** Checked directly against the v0.3.0 tree. The library we are porting is self-contained C and C++ with no third-party template metaprogramming anywhere in it. This is the single fact that makes Decision 1 a genuine simplification rather than a deferral.

Two consequences worth knowing up front:

- **Chat templates come for free.** `llama_chat_apply_template` is part of libllama's public API and `src/llama-chat.cpp` implements 174 built-in templates by heuristic string matching — explicitly *not* a Jinja parser. Our Zig CLI gets chat support without porting a template engine. See Q10.
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

- **`common/jinja/`** (6,349 lines). Decision 10 takes libllama's 174 built-in templates, which covers the canonical model. Models whose GGUF ships a custom Jinja template will render with the wrong template or fail; revisit when one matters.
- **`src/models/`** beyond `qwen35.cpp` (37,921 lines across 150 files). Decision 5.
- **Non-arm64 CPU backends** (`arch/x86` and friends). Needed for Stage 5's Linux targets.

---

## The test harness, and why it shapes everything

This deserves stating before the stages, because it determines their order.

Upstream `llama-cli` is **not** the deliverable, but it **is** the test harness for nearly the entire port. The mechanism: every ported file keeps its C ABI, exporting the same symbols with the same signatures. `build.zig` swaps a `.c` or `.cpp` source for a `.zig` one, and the still-unported C++ above it links against the result unchanged.

So through Stage 3 and most of Stage 4, we build the full upstream `llama-cli` — server, `common/`, JSON and all — as a **test fixture**, running it against progressively-more-Zig internals and diffing its token output. It is compiled but never shipped. Only when libllama itself is fully ported does our own Zig CLI take over as the driver.

This gives every single ported file its own independently verifiable, independently revertable step, against a reference that exercises the real inference path. Nothing in the plan is more important than keeping this harness green.

---

## Stage 0 — Repo hygiene

Prerequisite for everything else.

1. ~~**Fix `build.zig`.**~~ **Done.** The web-console executable and test module pointed at a non-existent `web/` directory and broke `zig build`, `test`, and `run`. Removed; all six steps now pass, 9/9 tests.
2. ~~**Rename the template identity.**~~ **Done.** `.name = .llamazig`; artifacts are `zig-out/bin/llamazig` and `libllamazig.a`. The `.zon` fingerprint was updated for the new name checksum while preserving the package id.
3. ~~**Wire `-Dtest-filter`.**~~ **Done.** Now passed to `b.addTest(.filters)`. Verified: 9 tests unfiltered, 5 with `-Dtest-filter="falls back"`.
4. **Delete the placeholder scaffold.** `src/base.zig` and `cli/client.zig` are template greeting code. They document the house conventions (see `CLAUDE.md`) but must not survive into the port. Still present — deleting them now would leave the module empty, so this waits for Stage 3's first ported file.
5. **Pin `llama.cpp/` as a submodule at v0.3.0** (Decision 3). Still outstanding: it is an untracked nested clone. `*.gguf` is now ignored, and CMake's output directory moved to `cmake-build/` so `build/` could hold build sources.

**Gate:** `zig build && zig build test` green. Met, except the submodule.

---

## Stage 1 — Build llama.cpp with the Zig compiler, driven by CMake — **done**

> **Built and verified.** `make buildmacos-zig` configures and builds the entire llama.cpp tree with `zig cc` / `zig c++` under CMake. 100% of targets build, including `llama-cli`, `llama-server`, the test suite, and the SvelteKit UI. The resulting `llama-cli` runs at ~222 t/s on Metal.
>
> **The gate is met.** `scripts/parity` diffs the Apple-clang and zig-cc binaries on the same model, prompt, and seed at `--temp 0`. **4/4 prompts token-identical.** Decision 4's correctness bar is therefore real and achievable, not aspirational.
>
> **What it took:**
> - `scripts/zigcc` and `scripts/zigcxx` wrappers, because `CMAKE_C_COMPILER` takes a single executable path and `zig cc` is two words.
> - `-G "Unix Makefiles"` instead of `-G Xcode`. Xcode's generator shells out to `xcodebuild`, which selects Apple clang and ignores `CMAKE_C_COMPILER`. Unix Makefiles also avoids needing ninja, since `make` is already present.
> - CMake itself, which was not installed. `make cmake` fetches 4.4.3 into `.tools/` rather than installing system-wide, since `/usr/local` is root-owned.
> - Separate `cmake-build/apple` and `cmake-build/zig` trees so the control and the subject do not clobber each other.
>
> **It just worked otherwise.** No source changes, no flag fights, no ABI conflicts between Zig's libc++ and the SDK. The predicted fallout (nullability warnings, ARC, NEON detection) did not materialise under CMake, which passes different flags than the hand-written `build.zig` does.
>
> **One difference from Apple clang worth remembering.** CMake's ARM feature probe through `zig cc` reports `HAVE_MATMUL_INT8 - Failed` and `HAVE_SVE - Failed`, while `HAVE_DOTPROD`, `HAVE_FMA`, and `HAVE_FP16_VECTOR_ARITHMETIC` succeed. Output is identical anyway, so this costs throughput rather than correctness, but it means the two builds are not taking the same code path through the quant kernels. Worth revisiting when Stage 3 ports `arch/arm/`.
>
> The original plan follows, for the record.

### Original plan

Goal: the same `llama-cli`, with every C/C++/Obj-C translation unit compiled by `zig cc` / `zig c++` instead of Apple clang. No source changes to llama.cpp.

**Already verified on this machine:** `zig cc` compiles and links C; `zig c++` compiles C++17 against Zig's bundled libc++; `zig cc -x objective-c ... -framework Foundation` compiles and links Obj-C against the macOS SDK. All three prerequisites hold under Zig 0.16.0.

**The blocker to plan around:** the `-G Xcode` generator in the current `Makefile` cannot be used. It drives `xcodebuild`, which selects Apple clang and ignores `CMAKE_C_COMPILER`. Stage 1 switches to `-G Ninja`.

1. Add `scripts/zigcc` and `scripts/zigcxx` wrappers — CMake needs a single executable path, and `zig cc` is two words.
2. Add `make buildmacos-zig`: the same `-D` flags as `buildmacos`, plus `-G Ninja -DCMAKE_C_COMPILER=.../zigcc -DCMAKE_CXX_COMPILER=.../zigcxx -DCMAKE_ASM_COMPILER=.../zigcc`. Leave `buildmacos` untouched as the reference build.
3. Expected fallout, roughly in order:
   - `-Wnullability-completeness` noise from Zig's libc++ headers against the Apple SDK (already observed in a probe). Suppress; do not chase.
   - Obj-C ARC and framework flags that CMake passes and `zig cc` handles differently.
   - `arch/arm/` NEON intrinsics and `-mcpu` feature detection. `GGML_NATIVE=OFF` is already in the flag set and helps here.
   - Zig's libc++ vs. the SDK's. If ABI conflicts appear, `-nostdinc++` plus explicit SDK include paths is the escape hatch.
4. **Establish the reproducibility baseline before trusting the gate.** Decision 12 puts Metal on the critical path, and GPU reduction order is not guaranteed stable. Run the *reference* binary against itself twice with `-ngl 99` and confirm token-identical output. If Metal is not run-to-run reproducible, Decision 4's gate is unusable as written and we fall back to the logit-distance tolerance — better to learn that here than mid-port.
5. **Build the parity harness** (Decision 4). Runs two `llama-cli` binaries against `Qwen3.5-2B-Q4_K_M.gguf` with a fixed prompt, fixed seed, `--temp 0`, **and `-ngl 99`** — Decision 12 means the harness exercises Metal at every stage, never a CPU-only path. Record tokens/sec too: a correct port that is three times slower is still a bug. This is the gate for every stage after this one.

**Gate:** the `zig`-compiled `llama-cli` produces token-identical output to the Apple-clang `llama-cli` on the canonical model and prompt, with Metal enabled.

---

## Stage 2 — Replace CMake with `build.zig` — **done for ggml and libllama**

> **Built and verified.** `zig build reference` produces `libggml.a` and `libllama.a` from the vendored sources with no CMake, no Ninja, and no `xcrun metal`. `zig build smoke` loads `Qwen3.5-2B-Q4_K_M.gguf`, offloads 25/25 layers to Metal, and generates. Release build: ~16s wall on 18 cores.
>
> **Delivered:**
> - `build/llamacpp.zig` — the build graph. ggml (base + CPU + Metal backends) as one static archive, libllama with all 151 architectures globbed from `src/models/`.
> - `build/metal_embed.zig` — a build-time Zig tool replacing CMake's `cat`/`sed` shader pipeline. All 20 kernel libraries embed and the driver compiles them at load.
> - `harness/smoke.zig` — inference through libllama's C ABI, which also rehearses the boundary Stages 3-4 rely on.
>
> **Three things that bit, recorded so they don't bite twice:**
> - `-fobjc-arc` must **not** be set. Upstream's `.m` files use manual reference counting and bridge `void *` to object pointers freely.
> - `sanitize_c` must be `.off`. Zig enables C sanitizers in Debug and upstream is not UBSan-clean — it aborts in `llama-graph.cpp` on a null-pointer offset.
> - `-G Xcode` genuinely cannot drive `zig cc`, as predicted. Moot now.
>
> **Independently validated against CMake.** The same C driver linked against `build.zig`'s libraries and against CMake+Apple-clang's libraries produces byte-identical output on 3/3 prompts. So `build.zig` is not merely *a* working build — it computes what the reference computes.
>
> **Not built:** `common/`, `tools/server/`, and the upstream `llama-cli`. Decision 1 puts them outside the port, and `zig build smoke` covers verification. Note Stage 1 *does* build all of these via CMake, so upstream's exact driver is available whenever it is wanted — see Q16.
>
> The original plan follows, for the record.

### Original plan

Still compiling upstream sources unmodified; only the build system changes. Best effort-to-value ratio in the plan, and it de-risks everything after it.

Port the ~1,700 lines of CMake that matter (`CMakeLists.txt`, `ggml/CMakeLists.txt`, `ggml/src/CMakeLists.txt`, `src/`, `common/`, `tools/`) into `build.zig`. Per `README.md`, the macOS `-D` flag set becomes the hardcoded default rather than options.

Note that Stage 2 still builds `common/` and the server, because Stage 1's harness needs them (see above). They are build targets, not port targets.

1. **`addLibrary` per CMake target**, preserving the graph: `ggml-base` -> `ggml-cpu` -> `ggml-metal` -> `ggml` -> `llama` -> `llama-common` -> `llama-server-impl` -> `llama-cli-impl` -> `llama-cli`.
2. **Metal shader embedding — easier than expected.** With `GGML_METAL_EMBED_LIBRARY=ON` the `.metal` files are *not* compiled by `xcrun metal`. CMake concatenates headers, strips `#include`/`#pragma once` with `sed`, inlines `ggml-common.h` and `ggml-metal-impl.h`, and emits a `.s` file that `.incbin`s the resulting MSL **source** into a `__DATA,__ggml_metallib` section; the Metal driver compiles it at runtime. That is text munging plus an assemble step — a `std.Build.Step.Run` and `zig cc` cover both. **Never enable the non-embed path**; it requires `xcrun -sdk macosx metal`, which Zig can never replace.
3. **Generated headers:** replicate `cmake/build-info.cmake` and `git-vars.cmake` as a small `build.zig` step emitting `build-info.cpp`.
4. **Reuse the existing `Xcode` SDK helper** in `build.zig` for framework paths (Metal, Foundation, Accelerate).
5. **Vendored headers** (`nlohmann`, `cpp-httplib`, `sheredom`, `stb`) are header-only or single-file: include paths, not build targets. They exist only to keep the harness compiling.

**Gate:** `zig build` alone — no CMake, no Ninja anywhere in the loop — produces a `llama-cli` that passes the parity harness.

---

## Stage 3 — Port the C to Zig

**24,541 lines**: `ggml.c`, `ggml-alloc.c`, `ggml-quants.c`, `ggml-cpu/ggml-cpu.c`, `ggml-cpu/quants.c`, `ggml-cpu/arch/arm/quants.c`.

One translation unit at a time, C ABI preserved, harness green after each. Order, most-testable first:

1. **`ggml-alloc.c`** (1,249) — self-contained allocator, easy to unit-test in isolation. Good first file to shake out the workflow.
2. **`ggml.c`** (8,067) — tensor and graph core. Large but mechanical: structs, enums, shape math.
3. **`ggml-quants.c`** (5,667) — reference quantize/dequantize. Ideal test target: round-trip every block type against the C version.
4. **`ggml-cpu/ggml-cpu.c`** — op dispatch and threading.
5. **`ggml-cpu/quants.c` and `arch/arm/quants.c`** (4,319) — **the hard one.** Dense ARM NEON intrinsics. Zig has `@Vector` and `std.simd`, with inline asm as a fallback, but this will not transliterate and is the likeliest place for silent numerical drift.

Per-file gates: a Zig unit test comparing against the C implementation on fixed inputs, plus the parity harness still green.

Mechanics:
- **`zig translate-c` output is a draft to read, not a deliverable to commit.** It produces `c_int`-everywhere, pointer-arithmetic-heavy Zig that violates every convention in `CLAUDE.md`. Consult it, then write real Zig.
- **ggml's macro-heavy headers hold real logic in the preprocessor.** `ggml-impl.h`, `ggml-common.h`, `simd-mappings.h` become `comptime` functions and generics — a rewrite, not a translation.
- **Map threading to `std.Thread` deliberately** rather than mirroring `ggml-cpu.c`'s pthread calls.

---

## Stage 4 — Port the C++ to Zig

**105,591 lines** of C++17 and Obj-C after Decision 1 and Decision 5: `ggml/src/*.cpp` (8,417), `ggml-cpu/` C++ (28,125), the Metal host layer (13,269), `src/llama-*.cpp` (55,135), and `src/models/qwen35.cpp` (645).

This is the bulk of the project. Four files carry a quarter of it: `ggml-cpu/ops.cpp` (12,021), `ggml-metal-ops.cpp` (5,369), `arch/arm/repack.cpp` (5,156), `ggml-cpu/repack.cpp` (4,836).

No Boost, no `nlohmann::json`, no HTTP library, no Jinja engine — Decision 1 removed all of them. What remains is STL containers, `std::function`, RAII, exceptions, templates, and virtual dispatch.

| C++ | Zig |
|---|---|
| `std::vector<T>` | `std.ArrayList(T)` — with explicit allocator plumbing |
| `std::string` | `[]const u8` + `std.ArrayList(u8)`; ownership decided per site |
| `std::unordered_map` | `std.HashMap` / `std.StringHashMap` |
| RAII destructors | explicit `deinit` + `defer` — **every call site changes** |
| exceptions | error unions — **every signature changes** |
| virtual dispatch | tagged unions, or vtable structs (`std.Io`-style) |
| templates | `comptime` generics — usually cleaner |

Bottom-up, so each layer lands on ported foundations:

1. **`ggml/src/*.cpp`** (8,417) — backend, backend-reg, gguf, opt. Closest to C, minimal STL.
2. **`ggml-cpu/` C++** (28,125) — `ops.cpp` holds the CPU op implementations and `repack.cpp` the quantized-weight repacking. Loops over tensors rather than STL-heavy abstraction, so it ports more like Stage 3 than like libllama. `llamafile/sgemm.cpp` (4,164) is templated matmul kernels and is the awkward one.
3. **`ggml-metal` host layer** (13,269), of which **3,091 is Obj-C** across `ggml-metal-device.m` (2,352) and `ggml-metal-context.m` (739). **Zig has no Obj-C frontend**, so this means direct `objc_msgSend` runtime calls. At 3,091 lines it is not a shim-sized problem, and Decision 12 means it cannot be deferred or routed around. It is the least predictable item in the plan — prototype `objc_msgSend` interop against one small call path *before* committing to the approach. See Q13.
4. **`src/llama-*.cpp` core** (55,135) — mmap, model-loader, arch, vocab, model, context, graph, batch, kv-cache, sampler, chat. This is libllama and the actual prize. Sub-order matters: loader and vocab first (testable against known GGUF metadata without running inference), then graph and context, then kv-cache.
5. **`src/models/qwen35.cpp`** (645) plus its shared scaffolding.
6. **The Zig CLI.** Once libllama is ported, the upstream harness is no longer the driver and `cli/` becomes the real entry point: model load, tokenize, sampler chain, generation loop, and libllama's built-in `llama_chat_apply_template` for chat (Decision 10). Per Decision 11 the flag surface is minimal and idiomatic — roughly `-m`, `-p`, `-n`, `-ngl`, `--temp`, `--seed` — not a reimplementation of `common/arg.cpp`'s 4,691 lines. Written to the conventions in `CLAUDE.md`, not transliterated.

Keep the upstream harness as a test-only build target after cutover. It is the only independent check that libllama still behaves, and it costs nothing but build time.

**Do not enter Stage 4 without the parity harness green and Stage 3's per-op numerical tests in place.** Without them this stage is unverifiable.

**Gate:** `zig build` produces `llamazig`, which generates token-identical output to the reference `llama-cli` on the canonical model, prompt, and seed.

---

## Stage 5 — Cross-platform

Targets: macOS, iOS, Linux arm64, Linux x86_64.

Zig makes the *build* side nearly free once `build.zig` owns everything — `-Dtarget=aarch64-linux-gnu` and so on. The work is backend coverage. Decision 12 keeps Metal on the critical path for macOS and iOS, but Linux has no Metal, so this is where a CPU-only path first has to stand on its own, and where `ggml-cpu/arch/x86/` (AVX2/AVX-512) becomes a second SIMD port that Stage 3 deliberately skipped.

iOS additionally needs the `buildios` flag set and the increased-memory-limit entitlement already noted in the `Makefile`.

Backfilling `src/models/` beyond Qwen3.5 (Decision 5) belongs here or later — it is repetitive graph-building code and parallelizes well once the first architecture proves the pattern.

---

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| ARM NEON quant kernels resist translation | Silent numerical drift, or a perf cliff | Per-block round-trip tests; keep the C version behind a build flag until the Zig one matches |
| Obj-C Metal host layer | Zig has no Obj-C frontend, and this is 3,091 lines, not a shim | Prototype `objc_msgSend` interop at Stage 4 step 3 *before* committing; keeping the two `.m` files as permanent Obj-C is an acceptable answer |
| Bit-exactness unreachable across compilers | Decision 4's gate becomes unusable | Detect early — Stage 1 tests exactly this before any port work. Fall back to logit-distance tolerance and document it |
| Parity harness rots | The whole verification story collapses | It is a build target, not a script someone remembers to run. Keep it in `zig build` |
| Metal output not reproducible run-to-run | Decision 4's token-parity gate becomes meaningless | Test it in Stage 1 step 4, before any port work depends on it |
| Zig 0.16 is pre-1.0; std churns | Breakage on toolchain bumps | Pin the exact version in `build.zig.zon`; bump deliberately, never incidentally |
| Performance regression vs. hand-tuned C++ | Port is "done" but unusable | Tokens/sec in the harness from Stage 1; treat a regression as a bug |
| Qwen-only hides architecture-general bugs | Ported libllama silently overfits to one model | Backfill a second, structurally different architecture before declaring libllama done |

---

## Sequencing

Stages 0-2 deliver standalone value regardless of what happens after: a pure-`zig build` llama.cpp with no CMake dependency, cross-compilable to every target. **Commit to 0-2, then re-plan 3-5 with real numbers** from having done it.

---

## Questions

**Q13. What is the endgame for the 3,091 lines of Obj-C?** Decision 12 puts the Metal host layer on the critical path, and Zig has no Obj-C frontend. Two ways this ends: (a) call the Objective-C runtime directly from Zig via `objc_msgSend` — genuinely pure Zig, but 3,091 lines of message-sending written against an untyped C API, and the least predictable work in the plan; (b) keep `ggml-metal-device.m` and `ggml-metal-context.m` as Obj-C compiled by `zig cc`, and port everything around them — ships sooner and is far lower risk, but the result is not a pure-Zig binary, which is the stated goal in `README.md`.

I do not want to guess at this one, because it decides whether "port llama.cpp to pure Zig" is literal. Is (b) an acceptable end state, or is (a) required?

> b is fine here for now
>

**Q14. Does the ported library keep its C ABI permanently, or only during the port?** Preserving `extern "C"` exports is the technique that lets the upstream harness work (see "The test harness" above), so it is not optional *during* the port. The question is what happens after. Keeping it means llamazig can be a drop-in `libggml`/`libllama` replacement for any C or C++ project — a substantial and durable win, at the cost of C-shaped signatures (out-params, error codes, opaque pointers) permanently constraining the public API. Dropping it once nothing C++ calls in gives idiomatic Zig with error unions and slices, but makes llamazig usable only from Zig.

Note this is largely reversible: an idiomatic Zig core with a thin `extern "C"` shim on top gets both, and is what I would build unless you want the C ABI to *be* the API.

> Yes
>

**Q15. Do we port all 27 quantization kernels, or only what the canonical model needs?** ggml defines 44 tensor types with 27 dequantization kernels. I inspected `Qwen3.5-2B-Q4_K_M.gguf` directly: its 320 tensors use exactly **five** types — F32, Q4_K, Q5_K, Q6_K, Q8_0. Porting only those first would cut a large fraction of Stage 3's three quant files (11,325 lines combined, most of it per-type kernels) and the same again in `arch/arm/repack.cpp`. The cost is that the other two models already downloaded — `IQ4_XS` and `UD-Q4_K_XL` — would not load until the rest are backfilled.

This is the same shape of call as Decision 5. Five types first and backfill, or all 27?

> All
>

**Q16. Do we still want the upstream `llama-cli` built?** Stage 2 built ggml and libllama, which is everything Decision 1 puts in scope, and `zig build smoke` covers verification. Building the full `llama-cli` would add `common/` (34,094), `tools/server/` (21,435), and the vendored `nlohmann` and `cpp-httplib` to the *build* — not to the port. The upside is being able to run standard llama.cpp CLI workflows against our build, and having upstream's exact driver available during Stages 3-4. The downside is carrying ~57k lines of build surface we have already decided not to port.

Note it would not give us an independent reference binary: same sources, same compiler, so it cannot catch what our own build gets wrong. Only an Apple-clang build via CMake does that.

>
>

**Q17. Should CMake be installed to enable the parity harness?** Decision 4's gate is token-identical output versus a reference build, and there is no reference build without CMake, which is not installed here. This is not urgent for Stage 2 but is a hard prerequisite for Stage 3. Do you want to install CMake (Homebrew is also absent, so this means an official installer or building it), or should the port proceed against a different correctness bar until then?

>
>
