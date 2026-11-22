# CLAUDE.md

Guidance for agents working in this repository.

## Purpose

- **A Zig port of llama.cpp**: ggml, libllama, libmtmd, `llama-cli`, `llama-server` and upstream's smaller tools, pure Zig apart from GPU kernels and the two Metal Objective-C files.
- **Kept in sync with upstream one release at a time** (`SYNC.md`) until a working `../zigggml` can be swapped in.
- **No idiomatic rewrite here.** That is `../zigggml` (began as `src/ggml/`, must stay a drop-in: same C symbols). The pre-move rewrite is on branch `ggml-rework` (`d4d1922`).
- **zigggml's A/B gates run this repository's harnesses** (`harness/`, `llama.cpp.zmake`, `cuda-ref`, `vk-ref`, `scripts/win-run`, `harness/port_links.zig`). After changing any of them, run zigggml's matching `make ab-*` group.

## Root files

Exactly four markdown files. Never add another.

| File | For |
|---|---|
| `README.md` | Users: what it is, build, run, `llama-cli`'s behaviour, known gaps |
| `CLAUDE.md` | Agents: rules, decisions, traps, gates |
| `SYNC.md` | The sync procedure and its live state (pin, open pickups, deferred, scope) |
| `HISTORY.md` | One short entry per release |

- A plan is temporary: once executed, delete it; lasting decisions go below, a line goes in `HISTORY.md`.
- Code comments state their reason inline; never point at a markdown section.

## Layout

| Path | What |
|---|---|
| `src/ggml/` | ggml. Root `module.zig` is the **only** import list (`ported.zig` re-exports it) |
| `src/ggml/{cpu,metal,cuda,vulkan}/` | CPU backend and GPU host layers |
| `src/llama/` | libllama. `api/` holds every `llama.h` function (unexported); `exports.zig` exports them; `harness_root.zig` re-exports without exports |
| `src/mtmd/` | libmtmd; same wiring (`exports.zig`, `harness_root.zig`) |
| `stdlib/` | The platform layer every module shares: locks (`sync/`), standard streams, files, the file system and paths, clocks, the environment, `errno`, C number parsing and `GGML_PAD`, strings, libm (`libm.zig`), every C library function still called (`libc.zig`, the ledger), the libc++ reproductions (`libcxx.zig`), CPU count, tty, subprocesses. Its own module (`build/llamacpp.zig`'s `stdlibModule`) |
| `cli/` | `llama-cli`: `args.zig`, `fit.zig`, `session.zig`, `conversation.zig`, `common_chat/`, `json/`, `peg/` |
| `tools/` | `llama-server` (`tools/server/`), the ten tools, `llama-tts`, `llama-mtmd-cli`, shared `tools/common/` |
| `harness/` | Gate drivers and reference-side code |
| `scripts/` | Gate scripts, generators, sync tooling |
| `build/` | Build sources (not output): `llamacpp.zig` (reference build graph), `upstream.zig` (the pin), `metal_embed.zig` |
| `xcframework/` | `llama.xcframework`, drop-in for upstream's |
| `llama.cpp/` | Upstream at the pin. **Read-only** |
| `llama.cpp.zmake/` | Separate repo: Zig build of stock upstream, the reference. Gitignored here |

## Hard rules

- **Never edit `llama.cpp/`** (or `llama.cpp.zmake/llama.cpp/`). No fixes, no workarounds, no commits, no PRs. Its `CLAUDE.md`, `AGENTS.md`, skills and CI do not apply here. Moved only by `SYNC.md`.
- **`llama.cpp.zmake`** is modified only under a grant: Zig 0.17, `arch/x86`, Vulkan, Windows, and the standing sync grant (its `build.zig`, `build.zig.zon`, `Makefile`, for that release only).
- **No dependencies.** One exception: `mlavergn/zigjinja` (`../zigjinja`, path dependency), confined to `cli/` and the server's `cli/common_chat`. Pin an exact commit.
- **`LICENSE` carries both copyrights** (this project, the ggml authors). Every file under `src/` is a translation of MIT C; keep the provenance headers.
- **Out of scope, by decision:** `llama-server`'s executable and HTTP front on Windows; any BLAS but Accelerate (zDNN, ZenDNN, OpenBLAS... are vendor libraries); x86 below `x86_64_v3`; Apple OS below 26.0 on any platform.
- **`llama-server-core` is in scope on Windows** (zigllm drives it in-process): `ServerContext`, its queues and the data layer build there. MCP, the server-side tools, `subproc`'s waiter and `oaicompat`'s media loading (`download.zig`, httplib's client) are POSIX only until their `_WIN32` arms are ported; `harness/server_core_test.zig` lists both sets.
- **`llama-cli` refuses what it does not support** (non-zero exit, clear error), never ignores it; flags keep upstream's names and defaults.
- **Library logs go through upstream's callbacks** (`llama_log_set`, `ggml_log_set`): never `std.debug.print` or `std.log` in library code, which write to stderr past them. Never `@panic` either; abort with `impl.abort`. A `@panic` pulls in `compiler_rt`, whose `expf`/`logf` shadow libm's.
- **Library code that needs an `Io` takes `std.Io.Threaded.global_single_threaded`**: the C API hands it none. No worker threads or signal handlers; `concurrent` fails and `async` runs inline; it sees no environment on POSIX (`getenv` stays). Never `Threaded.init` in a library: it installs `SIGIO` and `SIGPIPE` handlers in the host. It builds and links in every Apple slice.
- **Keep upstream's `GGML_ASSERT` in every build mode**; port a C `assert` as nothing (`NDEBUG` in the reference) and its following unsigned arithmetic as `-%`/`+%`.

### Porting

- **A translation unit is ported whole, then swapped.** Missing-symbol link errors are the completeness check. Develop beside the C, unwired.
- **Contract of a C++ TU = its unmangled exports** (`scripts/port-coverage LANG_=cxx`) -- but not always the whole contract: `repack.cpp`'s buffer type and traits are C++-linkage. Measure before estimating; survey counts were low twice.
- **ggml swaps file by file; libllama swapped once at `llama.h`** (one strongly connected component of C++); CUDA once at `ggml-cuda.h`.
- **Never import `exports.zig` from a `harness_root.zig`**: two definitions in one link, one picked silently.
- **libmtmd calls libllama through the port's `api`, never `c.llama_*`** (in a harness that symbol is the reference's).
- **A C++ throw is `error.Thrown`** with an `impl.Exception`; a `llama.h` entry terminates on it. Callers whose upstream catches it use `api/vocab.zig`'s unexported `tokenize`/`tokenToPiece`/`detokenize`.
- **One `@cImport` per header** (`cli/c.zig`, `impl.zig`): two give incompatible `*llama_model`.
- **`ggml-impl.h` is not importable** (`__mfp8`); add what you need to `src/ggml/impl.zig`.
- **`zig translate-c` output is a draft to read, never to commit.**
- **Metal kernels, CUDA kernels and Vulkan shaders are never ported**; they travel as embedded data.
- **A newly ported file goes into `module.zig`.**
- **Ported code calls ported code through `@import`, never an `extern` of its C name**: the definition's type checks the call. Per-architecture definitions resolve by `builtin.cpu.arch`; an `arch-fallback.h` rename is an `@export` plus a `pub const` of the same name in that architecture's `fallback.zig`. `extern` stays only where something else may define the symbol: `impl.zig`'s `ggml_log_internal` and `backend_meta.zig` (compiled alone against upstream's C by `meta-diff`, `opt-diff`, `blas-node-diff`), the Objective-C, CUB shims, Accelerate, and the CUDA/Vulkan registrations (`-D<backend>-port`).
- **A harness reaches both sides by name at link time**, so zigggml can link its library in place of ours; write it `@extern(*const T, .{ .name = ... })` with one shared `T` (`repack_diff.zig`'s `ours`/`reference`), never a hand-typed signature per symbol.
- **Locks are `stdlib.sync.Mutex` and `stdlib.sync.Condition`** (`stdlib/sync/`): futexes, no C library, no `std.Io`. Never `pthread_mutex_t`, `pthread_cond_t`, SRW locks, `std.Io.Mutex` or a per-file wrapper. A module that imports `stdlib` and compiles beside libllama takes libllama's instance (`llamacpp.stdlibOf`); a gate script that declares modules by hand passes `--dep stdlib` and `-Mstdlib=stdlib/module.zig`.
- **Platform helpers live only in `stdlib/`**: file reads and writes, the file system and `std::filesystem` paths, clocks and sleeps, the environment, `errno` text, C's number conversions, every C library declaration (`libc.zig`; a file never declares one itself; a function std declares is an alias of `std.c`'s, an `extern` only where std lacks it or types it stricter than C, as `mlock` and `setlocale`), libm declarations and clang's rewrites of them (`libm.sinCos`, `math_errno`), libc++'s algorithms, RNG and `random_device`, the CPU count, the C streams, `exit`. Never a local copy, never a function that only forwards. Where upstream does two different things, `stdlib` has one function per idiom (`file.readStream` keeps what came before a read error, `Dir.readFileAlloc` fails whole); a call site uses the one upstream's code there uses. A port of an upstream function that upstream defines twice keeps its citation where upstream put it and delegates. Where std already does the job (`std.ascii`, `std.mem.sliceTo`, `Dir.readFileAlloc`, `Dir.writeFile`), call std.

### Citations (`make port-links`)

- File header: the reference path from the repo root with extension (`llama.cpp/ggml/src/ggml-alloc.c`), tag and commit. Non-ports say so.
- Each declaration: ``Ports `ggml_hash_set_new` (ggml.c:6516 @c1d0e7a00)`` -- first line of the definition, **commit on every citation**. Several symbols: `(ggml-impl.h:173, 179 @sha)`. A folded definition: `…, inlining `g` (x.c:N @sha)`.
- Multiple `#if` arms: cite the one this target compiles and say which.
- `port-links` reads each citation at its own commit; it joins wrapped comment lines. A stale number sends a sync to the wrong function.

## Machines

| Machine | Runs | Cannot run |
|---|---|---|
| Mac (M5 Max) | Metal, Apple CPU, macOS-only gates, the xcframework | CUDA, Vulkan |
| GB10 (`spark-dbff`, Linux aarch64, compute 12.1, CUDA 13.0) | CUDA, Vulkan (SDK 1.4.363 in `.tools/vulkan/`), Linux arm64 CPU | Metal |
| DESKTOP-BR7J5S9 (Windows, Haswell) | x86 AVX2 gates via `scripts/win-run` | everything else |
| JWHLPTOP (Windows, Tiger Lake) | AVX-512; **excluded** (no access) | |

- Reached through `../zigupdater` (`SYNC.md`, Step 8). Work a machine cannot finish goes on `SYNC.md`'s open pickups, never skipped silently.
- A Linux arm64 podman container on the Mac (glibc 2.41, 2 GB, no model) runs the driver-twice gates natively before the GB10.

## Backends in brief

- **Metal:** all host C++ is Zig (`src/ggml/metal/`). Kernels embedded as flattened MSL (`build/metal_embed.zig`), compiled by the driver; never `xcrun metal`. `impl_c.zig`'s constants and `kargs.zig` (generated, `scripts/gen-kargs`) are checked by `make struct-layout`. `tuning.zig` exports C++-mangled names via `@export` (take them from `nm`).
- **CUDA**: 278 files; unmodified `.cu` compiled to embedded fatbins loaded with `cuLibraryLoadData`, each kernel named by its demangled template spelling from a generated table, every launch's argument sizes checked against `cuKernelGetParamInfo` in every build mode, CUB behind `cub_shim.cu`; launches go through `cuLaunchKernelEx` from `src/ggml/cuda/`. Env flags `GGML_CUDA_ENABLE_UNIFIED_MEMORY`, `_NO_PINNED`, `_REGISTER_HOST` are `atoi != 0`. A CUDA build links `libstdc++.so`, whose personality aborts every C++ throw in that process: gates that catch reference throws build `-Dcuda=false`.
- **Vulkan**: host in `src/ggml/vulkan/`; embeds the reference build's SPIR-V (`build/vulkan_shaders.zig`), never regenerates (`glslc -O` is not reproducible on `tri_f16`). Four GB10 arms: coopmat2, KHR coopmat, none, none without integer dot. Vulkan gates build `-Dcuda=false`.
- **`supports_op`** on CUDA and Vulkan is the C's predicate alone. `-D<backend>-port=false` builds upstream's host layer for the gates.

## Gates

**A gate proves nothing until it has failed.** Inject a fault, watch it fail, undo by reverse replacement. Confirm the build succeeded and the artifact changed (a failed build leaves the old library; a script run directly links whatever `zig-out/lib` holds, so rebuild first -- the `make` targets do). Confirm the injection changes the computation. A checker that silently skips input looks like one that passes.

- Compare on **bits**, never a tolerance. Bit-exact gates build `--release=fast` on both sides; optimization level and compiler move bits.
- **A gate's input distribution is part of the gate**: amplitudes, tie cases, thread counts, shapes straddling thresholds.
- **A gate's optimization level is part of what it sees**: only safe builds (`parity-cli`, `FIT_DIFF_O=ReleaseSafe`, `chat-parse-diff`) catch UB.
- `scripts/check-reference-pin` aborts every bit-exact gate unless zmake's checkout and tarball match the pin.
- `make validate` = format, scaffold, unit tests, ported tests (debug and release), `port-coverage`, `port-links`, `graph-diff`. Consistency, not correctness.
- `zig build test -Dtest-filter=...` skips non-matching codegen and a filter matching nothing looks green: check the count, run unfiltered.

| Gate | What only it sees |
|---|---|
| `graph-diff` | Constructor output (op, shape, strides, `op_params`, `src`); a halved softmax scale |
| `backend-ops` (`--diff`) | 26,751 op configs, Metal vs CPU at a tolerance; crashes and missing kernels, not values |
| `probe` | Ported code on the execution path (two runs: `-Dprobe-ported`, `-Dprobe-cpu`) |
| goldens (`vecdot-`, `vec-`, `quants-golden`) | Unreachable `_generic` dot products and float reductions, exact bits; `generic` siblings |
| `ops-diff` | Every CPU op on bits, 1 and 3 threads, amplitudes 1 and 30; reference writes quantized blocks, port replays |
| `repack-diff` | The 36 repack kernels and 12 converters, 4 and 20 rows |
| `node-diff` (`ARGS=--gpu`) | Every node of a real decode on bits, per node and fused; `--gpu` is the only Metal host-layer gate |
| `metal-diff` | Metal buffer-type/device vtables, fusion table, `get_alloc_size` |
| `metal-ops-diff` | Metal encoders on bits for ops a Qwen decode lacks; sentinel-filled `dst` |
| `tuning-diff` | 10.3M flash-attention tuning lookups |
| `struct-layout` | Metal structs and constants by offset; `cuda-struct-layout` likewise |
| `abi-check` | By-value struct entry points called as C does |
| `sched-diff` | Scheduler backend assignments (parity cannot see *where*) |
| `parity-cli`, `parity-port` (`-cpu`) | Whole binary vs a C driver; coarse (doubling `freq_base` passes) |
| `context-diff` | Context, batch, KV/recurrent memory, graph, state API: every node and log line, 9 configs |
| `arch-model-diff`, `mtp-diff`, `gemma4-assistant-diff` | Every architecture, MTP heads, the assistant model, on fixtures |
| `moe-slru-diff` | The MoE cache policy (an expert elsewhere computes the same bits) |
| `sampler-diff`, `libcxx-diff`, `regex-diff` | Sampler and grammar; libc++ sort/heap/RNG; `std::regex`/`wregex` over all code points |
| `unicode-diff`, `vocab-diff`, `arch-diff`, `loader-diff`, `api-diff` | Unicode tables, vocabularies, arch tables, metadata loading, every other `llama.h` entry |
| `jinja-diff` | Template text, input marks and part boundaries, under two time zones |
| `json-`, `schema-`, `peg-`, `chat-parse-`, `chat-msg-diff` | The chat stack, against upstream's tests recorded and replayed |
| `cnv-diff` | `llama-cli -cnv` vs a reference conversation over upstream's chat stack; prefix reuse |
| `fit-diff` | `--fit` on fake GPUs; every arm of `fit.cpp` reached |
| `tools-diff` | The tools, `llama-tts`, `llama-mtmd-cli`: stdout, stderr, exit, written files |
| `mtmd-diff` (+ `-image-`, `-audio-`, `-decode-diff`, `mtmd-exports`) | libmtmd on mmproj fixtures, preprocessors, decoders |
| `spec-diff` | Speculative drafters, n-gram stores |
| `server-task-diff`, `http-diff`, `server-diff` | Server data layer; HTTP on the wire; whole server on transcripts (`--self` first) |
| `server-core-consumer` (`test-server-core`) | `llama-server-core` as a consumer builds it: exported names, one `llama.h`, the API compiled, a completion run; Windows compiled |
| `blas-diff`, `blas-node-diff` | BLAS backend |
| `adapter-`, `quant-`, `opt-`, `opt-llama-`, `meta-`, `batch-`, `chat-diff` | LoRA/control vectors, quantizer, optimizer, meta backend, batch, chat API |
| `host-arms-diff` | Thread priority, affinity, NUMA, default seed |
| `cuda-ops-`, `cuda-node-`, `cuda-backend-diff`, `cuda-parity-cli` | CUDA (GB10) |
| `vk-repro`, `vk-ops-`, `vk-node-`, `vk-backend-diff`, `vk-push-layout`, `vk-parity-cli` | Vulkan (GB10); `test-backend-ops` never calls `graph_optimize`, `vk-node-diff` does |
| `x86-gates`, `win-host-diffs` | x86 kernels and host arms on Windows silicon |
| `port-links`, `port-coverage` | Citations; symbol contracts (maintenance, not correctness) |
| `xcframework-check` | Drop-in compatibility with upstream's xcframework |

Diagnostic, not a gate: `scripts/vecdot-prefix` finds the first diverging row prefix of a dot product.

## Toolchain traps

Zig **0.17.0**, 0.16-style std (`std.process.Init`, explicit `std.Io`, `std.Io.Writer`).

- **`p.*.arr[i]` through `[*c]T` miscompiles** (indexes by the whole array). Narrow with `impl.one` first.
- **A small `extern struct` received by value across the C ABI reads zeros** unless its size is a multiple of 4 with no padding. Take an integer of the same size and `@bitCast`; add it to `harness/abi_structs.c`.
- **`compiler_rt` shadows libm's `expf`/`logf`**: no `@panic` in libraries; native Linux executables link gcc's `libgcc.a` first (`llamacpp.addLibgcc`); `libm_check` guards it.
- **`std.c.LC` is glibc's numbering on every OS** (Darwin and Windows: `LC_ALL` 0, `LC_CTYPE` 2, `LC_NUMERIC` 4): use `libc.LC`. **`std.c.mlock` demands a page-aligned address**; C's takes any.
- **`std.c.Stat` is `void` on Linux and Windows** (`stdlib/fs.zig`); `std.c.clock_gettime` on Windows (`stdlib/clock.zig`); `std.c.write` takes a `HANDLE` on Windows (use `_write`).
- **`char` is `c_char`**, unsigned on aarch64 Linux.
- **`iN`/`uN` are type names**: the C's `i0`, `i11`... become `j0`, `j11`, digit for digit.
- **A Zig range `a..b` with `a > b` is illegal** where the C loop runs zero times.
- **A struct returned by value must not point into itself** (`Model.ownTensorSplit`).
- **No `@fence`**: `fetchAdd(0, .seq_cst)` on the field the C names. Keep the C's orderings verbatim.
- **No C variadic can be defined** on aarch64 Linux: the logger formats with `snprintf` over a tuple.
- **x86:** declare `extern fn @"llvm.x86..."` with `callconv(.{ .x86_64_sysv = .{} })`; `va_list` is a pointer parameter (`impl.cVaList`); the ggml library sets `use_llvm`.
- **`zig translate-c` on a cache hit hangs when stdout is a pipe**: redirect to a file.
- 0.17 renames: `dupeZ` -> `dupeSentinel(u8, x, 0)`; `.decls` -> `decl_names`; no `**`; `@intFromEnum` -> `@backingInt`. Comptime loops over declarations need `@setEvalBranchQuota`.

## Float contraction

`zig cc` contracts `a*b + c` to FMA (`-ffp-contract=on`); Zig code is strict. **Name a fusion (`@mulAdd`) only where an oracle can say you got it wrong.**

- Quantizers (`src/ggml/quants/`): never fused; goldens built `-ffp-contract=off`.
- NEON kernels, `xielu`, repack epilogues (`cpu/repack/epilogue.zig`): fused where measured. Clang fuses the **left** operand of the `+`. In `sumf += d*x - m*y` only `fmadd(d, x, -(m*y))` fuses.
- Vectorized reductions `acc += a[i]*b[i]`: groups rounded, only the tail fused (`simd.strictDot`, `strictLoopVectorized`, `strictSplit`). Interleave is per CPU model (`simd.strict_loop`; A14+ = 4, `generic` and A7-A13 = 8); an unmeasured model is a compile error.
- Darwin folds `sinf`/`cosf` into `__sincosf_stret` (`ops/common.zig`'s `sinCos`); Linux does not (`-fmath-errno`, `simd.math_errno`).
- Read the whole loop's disassembly, not one `fmadd`.

## NEON and CPU targets

- NEON integer arithmetic wraps (`+%`, `*%`); `vaddvq_f32` is pairwise, not `@reduce(.Add)`; `vshlq_u8` shifts right on negative counts; `vmaxq_f32` is `FMAX` (NaN-propagating), not `@max`.
- `neon.dotq_s32`/`mmlaq_s32` call `sdot`/`smmla` by LLVM intrinsic name when available; portable forms otherwise. **`neon.ggml_vdotq_s32` is not `neon.dotq_s32`** without dotprod: call the one the C spells.
- Write kernels against `simd.F32`/`simd.F16` (`simd-mappings.h`), never hard-coded lane counts.
- **Gate `src/ggml/cpu/` changes at `GATE_CPU` default, `apple_m3` (i8mm: two rows per call, different repack shapes), `generic` (Linux arm64 baseline), `apple_a7`, `apple_a12`.** `scripts/gate-cpu` checks both libraries were really built for that CPU. `generic` gives Linux's CPU, not its OS defaults.
- x86: load vectors through a vector pointer, never an array `@bitCast`; run-time shift counts via `psrli`/`pslli`; v3 has F16C and BMI2 (port that arm); a harness without `ggml_cpu_init` reads an all-zero fp16 table. AVX-512 is the AVX2 code at twice the width (`repack/x86/wide.zig`); confirm 512-bit code by disassembly. AVX512-BF16/FP16 are a compile error until gated.
- Windows: `std::random_device`, `wchar_t` and the C ctype table differ (the clock seed, 16-bit `wregex`, `[[:blank:]]` misses `\t`).

## libc++ reproduction

`stdlib/libcxx.zig` and `src/llama/regex.zig` reproduce the libc++ the reference links (Zig's bundled 22.1.8; headers identical to Apple's 22.1.6), not any correct algorithm.

- `make_heap` takes a different path for arithmetic element types.
- `<regex>`'s class/collating tables and `__match_any_but_newline` live in the dylib; measured, not guessed. `wregex` differs from `regex` (U+2028/9, names >= 127, `\u`, narrowing).
- `unordered_map` order matters (`llama-vocab.cpp` picks the first EOT match): `libcxx.StringMap` reproduces CityHash64 (Apple's unfixed 32-bit shift), prime growth, rehash.
- `std::sort` is insertion sort below 30 elements: ties only reorder in long ranges.
- `regex_iterator` is two flags (`__no_update_pos`, `match_prev_avail`), not a loop.

## Where the C is unspecified

Excluded from goldens, not matched:

- `quantize_row_iq4_nl_ref` reads uninitialized memory on an all-zero block; ours zeroes.
- 1-bit split search depends on `qsort`'s tie order; ours breaks ties on index.

## CLI and chat

- **The CLI compiles libllama's module into itself** (`@import("llama")`); `fit.zig` calls `llama-ext.h` functions as Zig. Output limit `n_outputs_max = 1`.
- **Chat renders through `cli/common_chat`**; `cli/chat.zig` remains for `jinja-diff`. Templating is opt-in (raw completion is what `make port`/`make ref` diff); any chat flag turns it on, `--no-jinja` vetoes.
- **Input is marked as upstream marks it**; zigjinja's provenance carries marks through what upstream carries them through. `bos_token`/`eos_token` are template text. Empty parts are parts.
- **Marks survive only on the AST path**: `Compiler.compile(template, false)`, never the bytecode VM.
- A render that produced nothing from a non-empty conversation is rejected (`{% bogus %}` renders empty).
- `-cnv` re-renders the whole history each turn and keeps the common token prefix. **`llama_memory_seq_rm` returns false on hybrid models** (Qwen3.5): clear the cache then.
- `upstream_flags.zig` (generated) separates "unsupported" from "unknown" flags.

## Building

```sh
zig build                 # scaffold
zig build cli             # llama-cli
zig build reference       # libggml.a, libllama.a, libllama-ref.a (upstream, oracle)
zig build test            # unit tests; test-cli for the CLI's (~6s)
make validate             # fast consistency check
make port / make ref      # same loop, port vs stock C; must match (MODEL PROMPT NPRED TEMP SEED=42 ARGS)
make zmake                # build the reference through llama.cpp.zmake
```

- **The checkout path has a space.** Quote every path in recipes; never pass an absolute path to a make function or use one as a target. macOS make is 3.81 (no `.SHELLFLAGS`).
- `sanitize_c = .off` for upstream (not UBSan-clean); never `-fobjc-arc`.
- zmake compiles the release tarball named in its `build.zig.zon` (fetched into `zig-pkg/`); the nested `llama.cpp.zmake/llama.cpp` checkout remains for `types-golden` and `check-reference-pin`.
- `make ref-chat` runs upstream's REPL `llama-cli`, which has no raw-completion mode; it will not match `make port`.
- Timings from `make port`/`make ref` are not a port-vs-C comparison (`make ref` links Apple-clang libraries; prompt figures swing).
- macOS SDK resolution: `build.zig`'s `Xcode` helper; `appleSdk` gives it to Apple targets only, never to a Windows or Linux one.

## Code conventions

- **Barrels.** Each directory's `module.zig` re-exports its files (`pub const x = @import("x.zig")`) and forces nothing. A file outside a directory reaches it only through that directory's barrel (`const quants = @import("../../quants/module.zig"); quants.blocks.X`); a parent's file through the parent's barrel (`const mtmd = @import("../module.zig"); mtmd.clip.x()`). Same-directory siblings import each other by name; a file importing its own barrel names it `mod`. `src/root.zig` is docs-only.
- **No file-scope member aliases** (`const libc = @import("stdlib").libc;`, `const Vocab = mod.Vocab;`, `const c = impl.c;`): write the full path at the use (`stdlib.libc.write(...)`). Values in a function may be named.
- **The C API is the library's, not the consumers'.** `src/llama/lib.zig` and `src/mtmd/lib.zig` force every exporting file, only when that library is the compilation root or under test. `cli/`, `tools/` and the server call the Zig API (`llama.api.llama_decode(...)`); a comptime `_ = @import("llama").lib` in a consumer is a stopgap until they do. Harnesses call upstream by C name.
- Allocator first, owner frees; `init(allocator, ...)` stores it.
- `init` returns an error union even when it cannot fail.
- Public functions: doc comment with `Parameters:` and `Return:` (what it borrows, how long). Struct `///` says why it exists.
- Tests at the bottom under `// ---` / `// Unit Tests`, starting `test { std.testing.refAllDecls(...) }`. `std.testing.allocator` is the leak check.
- Executables stay thin: `main.zig` unpacks `std.process.Init` and delegates.
