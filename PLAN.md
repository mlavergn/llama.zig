# PLAN.md

Implementation plan for the llama.cpp -> Zig port described in `README.md`.

**Forward-looking only.** How each completed step was actually done — what it
cost, what each gate caught, and the false passes found by injection — is in
`NOTES.md`. This file says what is decided and what comes next.

Written against `llama.cpp` at tag **v0.3.0** (commit `c1d0e7a00`), Zig **0.17.0**. Two machines: macOS arm64 (Apple M5 Max, Metal) and Linux aarch64 (NVIDIA GB10, CUDA) — see Decision 43.

---

---

## Decisions

Settled in review. The plan below assumes all of these.

| # | Question | Decision |
|---|---|---|
| 1 | Target binary | **Port ggml + libllama only; write our own Zig CLI on top.** Decisions 16 and 20 refine this: the CLI is ours, but flag-compatible with upstream's. |
| 2 | `nlohmann::json` | **Moot** — Decision 1 removes it from the target set entirely. |
| 3 | Upstream tracking | **Follow upstream's releases, one at a time** (revised 2026-10-27; `MAINTENANCE.md`, Decisions 1–20). The port froze at v0.3.0 through Stages 1–8. Each sync now pulls the next ggml-org release tag, ports what changed, and commits as that version, following `SYNC.md`. `UPSTREAM.md` records where the port stands. |
| 4 | Correctness bar | **Token-identical output** vs. the reference build, fixed seed, `--temp 0`, canonical model `Qwen3.5-2B-Q4_K_M.gguf`. Logit-distance tolerance is the documented fallback if bit-exactness proves unreachable. Reference is `llama.cpp.zmake` — see Decision 17. |
| 5 | Model architectures | **Qwen3.5 first**, backfill the rest later. |
| 6 | Metal kernels | **Stay as MSL**, carried verbatim as embedded data. |
| 6a | CUDA backend | **In scope.** The CUDA *kernels* stay as CUDA, like Decision 6's MSL; Zig takes the host layer. See "Stage 6 — CUDA" for the measurement, the split that is not the Metal split, and how the gate question was settled (Decision 43). |
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
| 24 | Refactoring to idiomatic Zig | **Per subsystem**, as each stops being called from C++ — not all at once at the end. **Not started**, and since the swap nothing is called from C++ any more, so it is open for every subsystem (see "What is left"). |
| 25 | Chat templates | **Full Jinja, via `mlavergn/zigjinja`** (a fork of `gremlin-labs/vibe-jinja`) as a single dependency. Supersedes Decision 10; permitted by the exception in Decision 27. |
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
| 40 | Order within Stage 4 | **Gated work first.** `ggml-backend-meta.cpp` and `ggml-opt.cpp` move to the end of Stage 4's ggml half, behind the `ggml-cpu/` C++. Neither is on the inference path, so neither has an end-to-end gate — and meta is **~96% unexecutable on this machine**: `ggml_backend_meta_device` is only built under `LLAMA_SPLIT_MODE_TENSOR`, our CLI never sets `split_mode`, and there is one GPU. The `ggml-cpu/` C++ is the opposite: covered by **`backend-ops`**, which runs CPU against Metal over 21,093 op configurations. Note that is the *only* gate that reaches it — measured, by putting an abort in `ggml_compute_forward_mul` and watching `make port` finish without hitting it. On a Metal machine the CPU op kernels never execute during inference, so `parity-cli` and `make port` cannot see them at all. One real gate still beats meta and opt's zero. Supersedes the ordering in "Order of work" step 1. **Since done**, both ported and swapped last as decided. The "~96% unexecutable" estimate for meta was wrong: it accepts the same device twice, so `scripts/meta-diff` runs a real all-reduce on {Metal, Metal} and reaches 94.8% of its lines. |
| 39 | Threading primitives in ported code | **pthreads directly**, not `std.Io.Mutex` / `std.Io.Condition`. Zig 0.16 requires an `Io` on every call and a backend entered through a C ABI has none. `std.Thread.spawn` is still used for the workers. |
| 43 | CUDA gate machine | **Option (1), settled by hardware.** A Linux aarch64 machine with an NVIDIA GB10 (compute 12.1, `sm_121a`), CUDA 13.0, driver 580, 20 cores, 121 GB unified memory. CUDA work is done and gated there; Metal work stays on the Mac. Each machine records what it could not run in "Mac pickup" / "CUDA pickup" below rather than skipping it silently. |
| 44 | How CUDA kernels travel | **As per-translation-unit fatbins, loaded through the driver API.** Every upstream `.cu` is compiled unmodified by `nvcc -fatbin` with the reference's exact flags; the fatbin is embedded in `libggml.a` the way the metallib is, loaded with `cuLibraryLoadData` and each kernel found by its mangled name with `cuLibraryGetKernel`. The upstream host code in those files is parsed by `nvcc` only to instantiate templates; none of it is linked. The Zig host layer launches with `cuLaunchKernelEx`. Measured on the GB10 before deciding: a fatbin built this way loads, launches from Zig and computes correctly. |
| 45 | Naming a kernel | **By its demangled template spelling, resolved at comptime from a generated table.** A build step reads every fatbin's entry symbols (`cuobjdump -symbols`), demangles them with `cu++filt`, and writes a table from `mul_mat_q<(ggml_type)12, 64, false>` to the mangled name. The Zig dispatch builds the key with `comptimePrint`; a key that upstream's own host code does not instantiate is a **compile error**, not a runtime miss. The table is regenerated from the build output on every build, so it cannot drift from the pin. |
| 46 | CUDA kernel argument ABI | **Checked on every launch in safe builds.** `cuKernelGetParamInfo` reports each parameter's offset and size; the launch helper asserts each argument's Zig size against it. A wrong integer width is otherwise a silent wrong answer. By-value structs (`uint3` fastdiv triples, `float2`, the `*_args` structs) additionally go through `make cuda-struct-layout`, the CUDA analogue of `struct-layout`, since a size check cannot see field order. |
| 47 | CUB device-wide algorithms | **One C-ABI `.cu` shim of our own, treated like cuBLAS.** `cumsum.cu` calls `cub::DeviceScan::InclusiveSum` and `argsort.cu` CUB's segmented sort; their host-side dispatch lives in CUB's header templates, version-specific to the toolkit, and is library code rather than ggml's. `src/ggml/cuda/cub_shim.cu` wraps exactly those instantiations behind `extern "C"`. It is the only CUDA host code we write in C++, and it is not a port of anything. |
| 48 | CUDA swap unit | **Once, at `ggml-cuda.h`** — Decision 42's method, for Decision 42's reason. The host layer is one C++ object graph (`ggml_backend_cuda_context`, its virtual pools, RAII allocations) reached across 60-odd files through mangled names, while its external contract is the eleven C functions of `ggml-cuda.h`. The Zig backend is developed beside upstream's, both registered in one process for the gates, its `supports_op` claiming only ported ops; the swap removes upstream's host objects from `libggml.a`. |
| 49 | How the other architectures are gated | **On upstream's own fixtures, through `context-diff`.** No real model of 140 architectures exists here, and none is needed: `tests/test-llama-archs.cpp` builds a small random model of every architecture from one recipe, and `-o` writes them. Compiled unmodified against `libllama-ref.a`, its output is upstream's definition of a model of each architecture, not ours. `make arch-model-diff` runs each through the context harness on both devices -- every log line, node, output bit and the state API. Where the recipe cannot build an architecture, `harness/arch_fixtures_extra.cpp` and `harness/arch_fixtures_draft.cpp` add only the keys or tensors the C++ demands, from the same `get_gguf_ctx`, and the C++ must accept what they write. A case the C++ itself aborts on matches only when the port aborts on the same assertion. |
| 50 | Architecture dispatch | **A table, one file per upstream class.** `llama_model_mapping`'s class is a file exporting the three overrides, and `models/module.zig`'s `forArch` is the mapping; a class reusing another's methods is an `Entry` naming them. `llama_layer` and `llama_model`'s tensors are generated (`scripts/gen-layer`), so every architecture finds its members, in upstream's order. |
| 42 | libllama's swap unit | **Measured: libllama is one strongly connected component, so it swaps once, at `llama.h` — with one exception that goes first.** 22 of its core translation units plus every `models/*.cpp` reach each other through 367 mangled C++ symbols and 27 C ones; ggml's "the external contract is a pure C ABI" does not carry over. The exception is **`llama-sampler.cpp` + `llama-grammar.cpp`**: inbound, `llama-context.o` reaches them through the C `llama_sampler_i` vtable, three C functions and two mangled ones with plain-pointer signatures, which Zig can `@export`; outbound, every call into `llama_vocab` has a public C equivalent. So that pair is ported and swapped first, and everything else in libllama is developed beside the C++ and gated in-process subsystem by subsystem until the single swap. See "Order of work" step 4. **Done: swapped**, with the C++ kept as `libllama-ref.a` for the in-process gates. |
| 51 | x86 gate machine | **A laptop**, the x86 counterpart of Decision 43. x86 work is gated there and recorded in an "x86 pickup" list like the Mac and CUDA ones; Metal and CUDA gates cannot run on it, and x86 gates cannot run anywhere else. Its CPU model, OS and whether it has AVX-512 are recorded when it is set up: without AVX-512, the v4 arms (Decision 53) have no native gate and wait for a machine that has it. |
| 52 | x86 reference | **`llama.cpp.zmake` gains `arch/x86`**, an explicit grant against Decision 18 as the Zig 0.17 migration was. Its `build.zig` refuses any non-aarch64 target today (build.zig:361); the change adds the x86 sources and their defines and nothing else, so the reference stays upstream compiled unmodified. |
| 53 | x86 levels | **AVX2 first (x86-64-v3: AVX2, FMA, F16C), then AVX-512 (v4).** A static binary compiles one `GGML_SIMD` arm, and upstream's `GGML_CPU_ALL_VARIANTS` needs `GGML_BACKEND_DL`, which is out of scope. v1 (scalar) and v2 (SSE3) are not targets. **Both done**: AVX2 exact on the Haswell laptop, AVX-512 exact at `x86_64_v4` and `tigerlake` (VNNI) on `JWHLPTOP`, a Tiger Lake (`NOTES.md`, "Stage 5 step 6: AVX-512"). |
| 54 | Vulkan gate machine | **The GB10**, as for CUDA (Decision 43). NVIDIA's driver exposes every extension the shaders probe for -- cooperative matrix 1 and 2, integer dot product, bfloat16 -- so the reference compiles its widest set of arms there. Vulkan gates cannot run on this Mac: it has no Vulkan driver, and MoltenVK would translate to Metal with a reduced extension set. An Arduino UNO Q (Adreno, a mobile Vulkan driver) is available as a later second device; not now. |
| 55 | Vulkan reference | **`llama.cpp.zmake` gains `-Dvulkan`**, an explicit grant against Decision 18, as Decision 52 was. It builds upstream's own `vulkan-shaders-gen` for the host, runs it once for the header and once per shader file with `glslc`, and asks `glslc` which shader extensions it accepts exactly as `ggml-vulkan/CMakeLists.txt` does. **The toolchain is part of the reference**: which C++ compiles depends on the Vulkan header version and on those `glslc` answers (64 `#if` sites on the seven extension macros, more on `VK_HEADER_VERSION`), and the SPIR-V depends on `glslc`'s optimizer. Both sides of every Vulkan gate are built with the same `glslc` and headers, and the shaders the port embeds are the reference generator's output, identical by construction. |
| 56 | When Vulkan starts | **Now, as Stage 7, beside Stage 5** -- whose x86 step waits on the laptop. Measured first ("Stage 7 — Vulkan" below), reproducibility before any code, as Stage 6 began. |
| 57 | Windows | **A target, and x86 is gated on it natively** (asked for on 2026-10-18: "that was always the plan ... get it building on Windows"). `x86_64-windows-gnu` -- MinGW, the only Windows ABI Zig ships a libc for -- at the same `-Dcpu` levels as Linux. **`llama.cpp.zmake` gains a `windows` platform** under that grant, a third against Decision 18: `_CRT_SECURE_NO_WARNINGS` and `ws2_32` from CMake's `WIN32` arms, and CMake's own `gnu11`/`gnu++17` dialects there only (below). The gate machine is Decision 51's laptop: a MacBookPro11,5 (Core i7-4980HQ, Haswell, x86-64-v3, **no AVX-512**, 16 GB) running Windows 11 Pro 26100 under Boot Camp, reached through `zigupdater`'s developer link (`push`, `tty`). Its firmware leaves VT-x off, so WSL2 cannot start there; the gates are Windows binaries. |
| 58 | The chat-template engine | **Fix `zigjinja` until `make jinja-diff` passes** (asked for on 2026-10-22, `MAINTENANCE.md` Decisions 13-18), not port `common/jinja`. "Passes" means upstream's 68 templates over seven conversations agree **byte for byte in text, in which side raises, and in every byte's input mark**, both ways; `bos_token`/`eos_token` are the one exception, bound as provenance tokens and re-trusted. `strftime_now` is local time on both sides and checked under two zones. zigjinja stays its own engine, so `jinja-diff` runs at every upstream sync. **Done, at zigjinja v3.0.0 (`de36d7b`)**: 469 renders identical and 7 where both raise, under `TZ=UTC` and `TZ=Australia/Lord_Howe`; `NOTES.md`, "zigjinja against `jinja-diff`". `build.zig.zon` still takes zigjinja by path, so pinning `de36d7b` is the step left. |
| 59 | The Vulkan toolchain, and who generates the shaders | **Pinned at SDK tag `vulkan-sdk-1.4.363.0` and built from source, never the distro's** -- `scripts/vk-toolchain` into `.tools/vulkan/`: Khronos's `Vulkan-Headers` and `SPIRV-Headers` at that tag, and `glslc` from shaderc v2026.4, whose `DEPS` pin glslang and SPIRV-Tools at exactly that tag's commits. Ubuntu 24.04's `glslc` is shaderc 2023.8, too old for coopmat2 and bfloat16, and there is no root on the GB10. **And one generator run feeds both sides**: measured, `glslc -O` reads uninitialized memory on `tri_f16`, whose bytes follow heap state from run to run (`NOTES.md`, "Stage 7: the GB10 pickup"), so Decision 55's "identical by construction" holds only when the port embeds the reference build's generated files rather than generating its own. |
| 60 | zDNN, ZenDNN and BLAS | **BLAS on Apple, over Accelerate; zDNN and ZenDNN not ported** (asked on 2026-10-24: "not external libs"). Each of the three is a thin ggml backend whose whole job is to call a vendor library -- measured, BLAS 530 lines around `cblas_sgemm`, zDNN 924 around IBM's `libzdnn` on s390x NNPA, ZenDNN 838 around AMD's C++ `zendnnl.hpp`, which upstream downloads and builds with oneDNN, libxsmm and AOCL. Without the library nothing is left to port, and the library-gains-no-dependencies rule stands, so zDNN and ZenDNN stay out. BLAS is the exception on Apple only: the port **already links Accelerate** on every Darwin target and calls its vDSP from the CPU backend (`GGML_USE_ACCELERATE`), so a BLAS backend over it adds nothing. Off Apple it would need OpenBLAS or another vendor library, so it is not built there. **Opt-in, `-Dblas`**: the reference configuration has BLAS off, and on by default it would change which device computes every large CPU `MUL_MAT` in every gate. Gated as each backend before it: upstream's `ggml-blas.cpp` with its exports renamed beside the port, in one process, on bits. **Done** (`NOTES.md`, "BLAS over Accelerate"): `src/ggml/blas.zig`, swapped in under `-Dblas`, `make blas-diff` and `make blas-node-diff` identical. |
| 61 | Upstream's other tools | **Ported, each as its own Zig binary under `tools/`** (asked on 2026-10-25: "do them all please"): `gguf-split`, `quantize`, `tokenize`, `fit-params`, `export-lora`, `batched-bench`, `cvector-generator`, `imatrix`, `perplexity`, `llama-bench`. This reverses SPEC §2's "never writes one" for these binaries -- `quantize`, `gguf-split`, `export-lora`, `imatrix` and `cvector-generator` write GGUF -- while the library and `llama-cli` stay as they were. They share `tools/common/`, the ported parts of upstream's `common/`: `common_params_parse`'s option table for these examples (161 options, help text exact), `common_log`, `common_init`, the imatrix loader. Each is gated by `scripts/tools-diff <tool>` against upstream's own binary, built unmodified over `libllama-ref` (`zig build tools-ref`): stdout, stderr, exit code and every file written. `tts` was held: it links `mtmd`, the 26,700-line multimodal library -- Decision 62. The downloader (`-hf`, `-mu`, `-dr`) and the INI config files are refused: the first would be an external dependency. `NOTES.md`, "Upstream's other tools". |
| 62 | libmtmd and `llama-tts` | **Ported, with `llama-mtmd-cli`** (asked on 2026-10-26: "port mtmd and tts, don't stop until they are completed"). Upstream's multimodal library, `tools/mtmd/` -- 17,399 lines of C++ plus 8,292 in 43 graph files, 60 projector types -- is `src/mtmd/`, every translation unit, and its C contract is the 82 `mtmd_*`/`mtmd_helper_*` functions of `mtmd.h` and `mtmd-helper.h`, exported by `src/mtmd/exports.zig` and equal to the reference's by `nm` (`make mtmd-exports`). It is pure Zig like the rest: the stb_image and miniaudio decoders upstream compiles into `mtmd-helper.cpp` are ported to Zig as far as it reaches them (`src/mtmd/vendor/`), and the SHA-256 bitmap id is `std.crypto`'s. No `MTMD_VIDEO`, as the reference builds none: the video helpers are their `#else` arms. The debugging entry points (`mtmd_debug_*`, C++ linkage, `std::vector` arguments) are not ported; `mtmd_get_memory_usage`, C++ linkage and a `std::map`, is Zig-only. No mmproj file exists here, so `make mmproj-fixtures` writes one small random mmproj per projector type, checked through upstream's libmtmd. Gated by `make mtmd-diff` (libmtmd through its C API, port against upstream, every fixture, CPU and Metal, per node and fused), `make mtmd-image-diff`, `make mtmd-audio-diff`, `make mtmd-decode-diff` and `make mtmd-exports`; the two binaries by `make tools-diff TOOL=tts|mtmd-cli` against upstream's own (28 and 39 cases). `llama-mtmd-cli` formats prompts through upstream's legacy (non-Jinja) chat templates, ported in `tools/common/chat.zig`; `--jinja` is refused, as is `MTMD_DEBUG_GRAPH` (upstream's graph dumper is not ported), and the console's Windows arm falls back to plain line reading. `NOTES.md`, "Stage 8: libmtmd and `llama-tts`". |
| 63 | The xcframework | **A drop-in for upstream's `build-xcframework.sh` output** (asked on 2026-10-26: "Our make xcframework target needs to produce an xcframework that is drop in compatible with it"), replacing the static `llamazig.xcframework`. `xcframework/` (renamed from `xcfwk/`) builds `llama.xcframework`: upstream's seven slices (macOS, iOS, tvOS and visionOS, devices and simulators, arm64 and x86_64 as upstream has them), its minimums (iOS 16.4, macOS 13.3, tvOS 16.4, visionOS 1.0), dynamic frameworks with its install names and dSYMs, and its headers, module map and `Info.plist` byte for byte -- `mtmd.h`, `mtmd-helper.h` and `ggml-blas.h` included, since libmtmd (Decision 62) and the BLAS backend (Decision 60) are ported. Each slice architecture is `zig build xcframework`: `libggml.a` with BLAS and `libmtmd.a`, libllama compiled into it. **The arm64 CPUs are Apple clang's defaults for each target** (`apple_m1` for macOS and the simulators, `apple_a7` for iOS and tvOS devices, `apple_a12` for visionOS), so it runs wherever upstream's does; x86_64 is `x86_64_v3`, the port's floor. Built with `../zig17`, Zig 0.17.0 patched for Apple's mobile SDKs. Gated by `make -C xcframework compat` against upstream's own, built by its script from a `git archive` of the pin; a binary built against upstream's framework runs on ours unrelinked (`dropin-macos`); upstream's `examples/llama.swiftui` builds against it untouched and runs on an iPhone 15 (`swiftui`). Gating `apple_a7` found the port's vectorizer model wrong for every Apple core before the A14. `NOTES.md`, "llama.xcframework". |

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

- ~~**`common/jinja/`** (6,349 lines).~~ No longer deferred, and no longer ported: Decision 25 takes `mlavergn/zigjinja` instead, a pure-Zig Jinja2 implementation. That removes 6,349 lines of C++ from the port entirely, at the cost of one dependency, explicitly excepted from the no-dependency rule by Decision 27.
- ~~**`src/models/`** beyond `qwen35.cpp` (37,921 lines across 150 files).~~ No longer deferred: every architecture is ported (Decision 36's first follow-on, "What happens next" item 12).
- ~~**Non-arm64 CPU backends** (`arch/x86` and friends).~~ No longer deferred: x86_64 is Stage 5 steps 5–6 (Decisions 51–53).

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

**Stage 0's two follow-ups are done**: `llama.cpp/` is a submodule pinned at
`c1d0e7a00` (v0.3.0, Decision 3), and `scripts/parity-port` diffs against
`llama.cpp.zmake` (Decision 17), with `scripts/check-reference-pin` aborting
any bit-level gate whose reference checkout has drifted from that pin.

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
| `ggml-metal/ggml-metal-tuning.cpp` | 7 functions in a C++ namespace | **This row was wrong.** Measured: `ggml-metal.o` references 5 of them, `ggml-metal-ops.o` 2, `ggml-metal-device.o` 1 — **three** translation units, so every remaining Metal C++ file needs it. It goes first, and the other three stay C++ meanwhile, because **Zig can export the mangled names** with `@export`. See `NOTES.md`. |

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

**2. `ggml-cpu/` C++** (28,125 raw, **15,488 live** on this target) — **done.**

Every live translation unit is ported: `binary-ops.cpp` (4 symbols),
`unary-ops.cpp` (23), `vec.cpp` (10), `ops.cpp` (88),
`llamafile/sgemm.cpp` (1), and the vtable cluster — `traits.cpp` (2),
`ggml-cpu.cpp` (7), `repack.cpp` (36) and `arch/arm/repack.cpp` (28). What
still compiles from C++ here is `hbm.cpp`, `amx/amx.cpp` and `amx/mmq.cpp`,
all three **empty on this target**. Templates map cleanly onto `comptime`,
as predicted.

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

### The vtable cluster: done, and what it cost

All four swapped in one step, which is the only way they go: `repack.cpp`
derives from the two abstract bases `traits.cpp` declares, and two
definitions of a symbol in one static archive is not a link error.

Two things are worth carrying forward from it.

**The unmangled-export contract is not the contract for `repack.cpp`, and
believing it cost a day.** All 36 unmangled symbols were ported and
`cluster-check` read 100% while the ~940 lines that decide when to call
them — the buffer type, `extra_buffer_type`, sixteen `tensor_traits`
instantiations, `ggml_repack_get_optimal_repack_type`, eleven block
converters — were still C++ and still stubbed. `node-diff` failed on the
first `MUL_MAT`. The script now asserts a flag at comptime that the port
has to flip. **Before porting any of the remaining four exceptions, work
out what its contract actually is.**

**`make repack-diff` is a new gate and it was overdue.** The 36 kernels
had no oracle: `test-backend-ops` and `ops-diff` never allocate a
`CPU_REPACK` buffer, `make port` runs on Metal, and `node-diff` stops at
the first divergent node. Half of them cannot even be selected on this
target. It found four distinct faults in one run where `node-diff` found
one, three of them in dead shapes. See `NOTES.md`.

**3. `ggml-metal` host layer** (13,269). **Started.**
`ggml-metal-common.cpp` and `ggml-metal-tuning.cpp` are ported and swapped
— `src/ggml/metal/{common,tuning,tuning_table}.zig`, 6/6 and 7/7 — the
first because it holds no Metal API at all, the second because **every
remaining Metal C++ file calls it**. Measured
before starting: four of the five C++ units have a pure C ABI contract, the
Objective-C boundary is pure C in both directions (9 symbols out, 57 in),
and `node-diff --gpu` is an exact oracle that the CPU default cannot
substitute for. **`ggml-metal-tuning.cpp` had zero unmangled exports and
seven mangled** — the last of the four exceptions, and the sharpest form of
the `repack.cpp` trap, since `port-coverage` read it as 0/0 = 100%. It is
done: Zig exports the seven under their mangled names, so the three C++
files that call it — `ggml-metal.cpp` (5), `ggml-metal-ops.cpp` (2),
`ggml-metal-device.cpp` (1), **not one as this plan said** — keep linking
and can each be ported on their own.

**The Metal group is done.** `ggml-metal.cpp` →
`src/ggml/metal/backend.zig` (6/6), `ggml-metal-device.cpp` →
`library.zig` (73/73), `ggml-metal-ops.cpp` → `ops.zig` (66/66), plus
`common.zig` (6/6) and `tuning.zig` (7/7). **`ggml_metal_cxx_sources` is
empty**; only the two `.m` files remain, by Decision 13.

`ggml-metal-ops.cpp` was the last and largest: 4,068 live lines, 53 per-op
encoders behind one dispatch switch plus 13 further exports. Two things
it established:

- **A stubbed export is indistinguishable from a working one to a symbol
  count.** Its 66 `ggml_metal_op_*` were deliberately *private* Zig
  functions while the bodies were stubs — exporting them would have made
  `port-coverage` read `66 / 66 (100%)` for a file whose every dispatch
  aborted, exactly what `repack.cpp` did before `cluster-check` existed.
  `ops.encoders_implemented` is the comptime guard, and the switch to
  `pub export fn` was the *last* step, so the linker's missing-symbol
  errors were what proved completeness.
- **The `kargs` structs are generated, and that is load-bearing.** 66
  structs, 946 fields, which the kernels read by offset. `make
  struct-layout` checks all of them (3,103 facts). Zig also refuses a
  struct literal with a field missing, where the C's designated
  initialiser silently zero-fills — so a field added upstream breaks the
  build instead of becoming 0.

**`ggml-metal-ops.cpp` is next and its foundation is in place.** 4,068
live lines, 66 unmangled exports, **zero mangled** — a pure C ABI contract,
so the ordinary swap applies. One STL use in the whole file. It is the op
encoder: per op, pick the pipeline (now ours), fill an argument struct,
dispatch.

Its cost is those argument structs. **65 `ggml_metal_kargs_*` structs
carrying 936 fields**, which the kernels read **by offset** — a field at
the wrong offset feeds a kernel a different number, and nothing in the
ported code could notice. So `src/ggml/metal/kargs.zig` is **generated by
`scripts/gen-kargs`** from the header, and `make struct-layout` compares
every struct's size and alignment and every field's offset, width and
**name**: **3,069 facts, all agreeing.**

The name comparison is not redundant. Negative-tested 4/4, and the case
that justifies it is two adjacent same-size fields swapped: `sizeof`,
`alignof` and every offset are identical, and only the name check sees it.
The generator also **exits** on a field it cannot parse rather than
warning — it silently dropped seven array fields on the first run, which
would have emitted structs of the wrong size.

`ggml-metal-device.cpp` is 68 `get_pipeline_*` functions that build a
kernel name and a cache key, the 4 `ggml_metal_pipelines_*` that back the
cache, and `ggml_metal_device_get`. Two things it established:

- **The 112 `FC_*`/`OP_*`/`N_*`/`SZ_*` constants are extracted from
  `ggml-metal-impl.h` by script and checked by `make struct-layout`**, not
  typed. A wrong one is the quietest fault available here: `FC_UNARY + 1`
  names the slot a Metal function constant is written to, so an off-by-one
  configures a different constant and the kernel does something else.
  Negative-tested 3/3 — wrong `FC_` index, wrong `OP_` number, dropped
  constant (caught by name, value and count).
- **`node-diff --gpu` on a Qwen3.5 Q4_K decode reaches only a slice of
  this file, and the slice is narrower than it looks.** Measured with
  unconditional aborts: of seven sites probed, **six never execute** — the
  float `mul_mv` arm, the `!has_tensor` `mul_mm` arm (this machine has the
  tensor instructions), `flash_attn_ext`'s `bc_mask`, `rms_norm` at
  `n_fuse == 2`, `soft_max`, and `count_equal`. Only the quantized
  `mul_mv` arm aborted. Six faults injected into those sites all read as
  `PASS`, and **none of them is evidence about the gate** — they were
  inert, not escapes. This is the `vec.cpp` lesson again: confirm the
  injection changes the computation before believing the result.

**The Metal headers are hand-declared, not imported** (`metal/device_c.zig`).
They import cleanly, but only `impl.zig`'s single `cImport` can hold them
without creating two incompatible `*ggml_tensor`, and putting them there
widens the `c` namespace every ported file sees. `make struct-layout` is
what keeps the hand-written `ggml_metal_device_props` honest.

Decision 13 settles the hard part: the
**3,091 lines of Obj-C** in `ggml-metal-device.m` (2,352) and
`ggml-metal-context.m` (739) **stay as Obj-C**, compiled by `zig cc`. Only the
10,178 lines of C++ around them are ported. `ggml-metal-tuning.cpp` pairs with
`ggml-metal-ops.cpp`.

**4. `src/llama-*.cpp` core** (55,135) — mmap, model-loader, arch, vocab, model,
context, graph, batch, kv-cache, sampler, chat. This is libllama and the actual
prize. Sub-order matters: loader and vocab first (testable against known GGUF
metadata without running inference), then graph and context, then kv-cache.

**The ggml method does not carry over, measured.** The same intersection that
settled ggml — every object's defined symbols against every other object's
undefined ones, in `zig-out/lib/libllama.a` — gives the reverse answer here:

| | ggml | libllama |
|---|---|---|
| C-ABI symbols crossing a translation-unit boundary | most of the contract | **27** |
| C++-linkage symbols crossing one | almost none | **367** |
| objects referencing another's vtable or typeinfo | 3, one cluster | **25** |

Computing strongly connected components over that reference graph gives **one
component of 22 core translation units plus all 146 model objects**, and
leaves hanging off it: `llama-impl`, `-mmap`, `-arch`, `-hparams`, `-chat`,
`-io`, `-memory`, `-cparams`, `unicode`, `unicode-data` (and `llama-quant`
above it). The cycle is structural, not incidental: `llama_model` is a base
class with one subclass per architecture, `llama_model_create` names every
subclass's constructor, the models call back into `llm_graph_context`, and the
kv-cache variants implement `llama_memory_i`. Most of the leaves are no easier
— their inbound calls are mangled and take `std::string` or `std::vector`,
which rules out the `@export` route `ggml-metal-tuning.cpp` used.

So **the component swaps once, at its real external contract, `llama.h`**
(Decision 42). Until then the Zig is developed beside the C++, not wired in,
and each subsystem is gated **in-process against the C++ through `llama.h`** —
the `repack-diff` shape — rather than by swapping it. Loader and vocab first
still holds; it is now "first to be gated", not "first to be swapped".

**One cluster peels off: `llama-sampler.cpp` + `llama-grammar.cpp`** (5,910
lines). Its contract is **38 C functions plus two mangled ones** —
`llama_sampler_backend_begin(llama_sampler *)` and
`llama_sampler_backend_n_nodes(const llama_sampler *)`, plain pointers, so Zig
can `@export` them. `llama-context.o` and `llama-graph.o` otherwise reach
samplers only through the C `llama_sampler_i` vtable and
`llama_sampler_chain_get`/`_n`/`llama_sampler_name`; `struct
llama_sampler_chain`'s layout never crosses. Grammar's 349 mangled exports are
called by the sampler alone. Outbound, the pair calls `llama_vocab` members
whose public C wrappers are exact equivalents — `token_to_piece(id)` is the
cache `llama_token_to_piece(…, lstrip = 0, special = true)` reads. The two
exceptions are `llama_sampler_init_dry_testing`, a C++-linkage entry point for
`tests/test-sampling.cpp` that is outside the contract, and the DRY clone's
`llama_vocab dummy_vocab`, which only exists to call `init_dry` with no
breakers.

**Done.** `src/llama/sampler/`, `src/llama/grammar/`, with `std::regex`
ported in `src/llama/regex.zig` and the libc++ algorithms the sampler depends
on in `src/llama/libcxx.zig`; `llama_sources` is two files shorter. Its oracle
was built first: `make sampler-diff`, 40,289 records on bits, plus
`make libcxx-diff` and `make regex-diff` for the reproduced standard library.
The sampler leans on three things the C++ standard leaves to the library —
`std::discrete_distribution`, and the tie order of `std::sort` and
`std::partial_sort` — so they are reproduced from the libc++ headers the
reference is compiled against, and the oracle includes ties deliberately. It
found two real bugs and four holes in itself; see `NOTES.md`, "libllama is one
component".

**Next: loader and vocab, gated in-process, not swapped.** The rest of
libllama is the one component, so the work from here is written beside the
C++ and checked against it through `llama.h` until the single swap. The
first gate to build is a tokenizer diff -- tokenize and detokenize a corpus
through the C++ vocab and through the Zig one in one process -- before
`llama-vocab.cpp` (4,440 lines) is ported; then a GGUF-metadata and hparams
diff for `llama-model-loader.cpp`.

**Done within it: `llama-vocab.cpp`**, all six tokenizer types, gated by
`make vocab-diff` (55.0M checks over 41 files, 22 of them synthesized for
the types and error paths no real file reaches). Its metadata reader,
`gguf_meta.zig`, is the first piece of `llama-model-loader.cpp`. See
`NOTES.md`, "`llama-vocab.cpp` -- the vocabulary".

**Done within it: everything `llama_model_load` does before
`load_tensors`.** `llama-arch.cpp` (`src/llama/arch.zig`, tables generated
by `scripts/gen-arch`, gated by `make arch-diff`, 124,290 checks),
`llama-hparams.cpp` (`hparams.zig`, fields generated by
`scripts/gen-hparams` in upstream's layout), the metadata half of
`llama-model-loader.cpp` (`loader.zig`: the constructor, splits, the
weights' order, the file-type guess, the metadata dump, `print_info`; every
typed getter in `gguf_meta.zig`), and of `llama-model.cpp` (`model.zig`:
base `load_hparams`, Qwen3.5's `load_arch_hparams`, `print_info`, the RoPE
rule; tables from `scripts/gen-model-tables`). Gated by `make loader-diff`,
13,719 checks, every `llama_hparams` member compared through a
layout-checked mirror. See `NOTES.md`, "The loader and the
hyperparameters".

**Done within it: `load_tensors`.** `llama-mmap.cpp` (`mmap.zig`), the
tensor half of `llama-model-loader.cpp`, the device and buffer-type
selection, Qwen3.5's `load_arch_tensors`. `loader-diff` grew to every
created tensor's name, shape, type, buffer and data hash under 21 parameter
variants: 150,557 checks.

**Done: the rest, and the swap.** Batch (`batch-diff`), chat templates
(`chat-diff`), the KV cache, the recurrent and hybrid memories, the graph
builder, Qwen3.5's graph and the delta-net blocks, `llama_context` and its
state API (`context-diff`, ~2.4M checks over 9 model configurations
including tensor parallelism and a device without the fused Gated Delta
Net), `LLAMA_SPLIT_MODE_TENSOR`'s split state (`meta_split.zig`, compared
tensor by tensor), adapters (`adapter-diff`), the quantizer and saver
(`quant-diff`), training (`opt-llama-diff`), and every other `llama.h`
entry point (`api-diff`). Then the single swap: `libllama.a` is the Zig
alone, `exports.zig` exports 240 symbols equal to the reference's, and the
C++ survives only as `libllama-ref.a` for the in-process harnesses.
`parity-cli`, `parity-port`, `parity-port-cpu`, `node-diff` (CPU and GPU)
and `sampler-diff` pass on it. See `NOTES.md`, "libllama swapped".

**Also done within it: `unicode.cpp` and `unicode-data.cpp`**, with their own
in-process oracle, `make unicode-diff` (7.07M checks, every pattern set in
`llama-vocab.cpp`). They needed `std::wregex` and `regex_iterator`, so
`regex.zig` is now generic over the character type and `regex-diff` covers
both engines. See `NOTES.md`, "`unicode.cpp` -- the vocabulary's first
piece".

**5. `src/models/qwen35.cpp`** (645) plus its shared scaffolding. **Done**:
`src/llama/models/qwen35.zig` and `delta_net.zig`, gated by `context-diff`.

**6. The Zig CLI.** Decision 26 pulled this forward and it is built — see
`NOTES.md`, "The CLI, built now". It now links the fully ported libllama, and
no C++ runtime.

### What the CLI can and cannot support

The dividing line is not what upstream's CLI does; it is **what lives in
libllama versus what lives in `common/`**. libllama is being ported in full, so
anything implemented there is available to our CLI for the cost of the argument
handling and a few API calls.

**In scope, because libllama implements it** — sampling in all its forms (41
public entry points), GBNF grammars, **LoRA and aLoRA adapters**, session state
save and restore, and the 174 built-in chat templates.

**In scope via one dependency** — full Jinja chat templates, using
`mlavergn/zigjinja` (Decision 25). Built; see `NOTES.md`, "Chat templates".

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

## Stage 6 — CUDA

**Decided: CUDA is in scope.** This section is the measurement and the
shape of the work, not an estimate — `ggml-cuda` was measured the same way
every other unit in this plan was.

### Scope, measured

| | |
|---|---|
| Files | **278** under `ggml/src/ggml-cuda/` |
| `.cu` | **24,720** lines, 187 files |
| `.cuh` | **18,747** lines, 86 files |
| `.h` | 480 lines, 3 files |
| **Total** | **~43,900 lines** |
| `ggml-cuda.cu` | **5,583** lines — the backend/device/buffer layer, the analogue of `ggml-metal.cpp` |
| `__global__` kernels | **145**, in 59 of the 187 `.cu` files |
| `__device__` functions | 444 |
| `<<<…>>>` launch sites | **161** |

For comparison, all of `ggml/src/ggml-metal/` is 13,269 lines. **CUDA is
roughly 3.3× the entire Metal backend**, and about 80% the size of
`src/llama-*.cpp` (55,135), which Stage 4 treats as the main prize.

### The Metal split does not transfer, and that is the central fact

Decision 13 works for Metal because the two languages live in **different
files**: MSL kernels in `ggml-metal/kernels/*.metal`, host code in
`.cpp`/`.m`. Zig took the `.cpp`, the `.m` stayed Obj-C, and the boundary
between them is a pure C ABI — measured, 9 symbols out and 57 in.

CUDA has no such seam. **Host and device code are interleaved in the same
translation unit**, and the 161 `<<<grid, block, shmem, stream>>>` launches
sit *inside* the host functions. That syntax is device-aware and only
`nvcc` or clang's CUDA frontend can compile it; Zig has no CUDA frontend
any more than it has an Obj-C one.

So "kernels stay, host layer ports" cannot be applied file-by-file the way
it was for Metal. The route that *does* transfer:

- **Kernels compile to PTX/cubin by `nvcc` and travel as device code** —
  exactly Decision 6's treatment of MSL, and exactly what `CLAUDE.md`
  already says about the Metal kernels: "they travel as embedded data".
- **Each `<<<…>>>` becomes a `cudaLaunchKernel` call.** The CUDA *runtime*
  API is plain C, so Zig can call it directly. That puts the Zig side in
  the same position it already occupies for Metal, where it calls
  `ggml_metal_encoder_dispatch_threadgroups` across a C ABI into the `.m`.
- **`.cuh` headers are the `simd-mappings.h` problem again, 18,747 lines
  of it.** They are macro- and template-heavy device code; the ones that
  are purely `__device__` are not ported at all, they compile with the
  kernels. Only the host-reachable declarations need Zig equivalents.

Zig does list `nvptx` and `nvptx64` as targets, so emitting PTX from Zig
is *conceivable* — but that backend is experimental and this plan does not
depend on it. Kernels stay CUDA.

### The gate question, settled by hardware (Decision 43)

This section used to say that **no gate in this project could run for
CUDA**, because the only machine was an Apple M5 Max, and that porting
~43,900 lines with no oracle would produce exactly the artifact the project
refuses — green, consistent, unverified. It recommended option (1), a CUDA
machine. That machine now exists: Linux aarch64, an NVIDIA **GB10**
(compute 12.1, built for `sm_121a`), CUDA **13.0**, driver 580.

Measured on it before any CUDA code was written:

- **The reference CUDA backend builds and runs.** v0.0.23 compiles all of
  upstream's `ggml-cuda` unmodified with `nvcc` into `libggml.a`. Qwen3.5-2B
  generates at ~137 t/s on the GPU against ~18 t/s on the CPU device.
- **CUDA decodes are bit-reproducible run to run.** `harness/node_dump.c`
  linked against the CUDA build gives the same hash over all **2,750** node
  outputs in three consecutive runs. So `node-diff` can be an **exact**
  CUDA oracle, as it is for Metal — not a tolerance check.
- **The driver API route works from Zig.** A fatbin built from
  `scale.cu` with the reference's flags loads through `cuLibraryLoadData`, its
  kernel is found by mangled name, launched by `cuLaunchKernel` from Zig, and
  computes the right answer; `cuKernelGetParamInfo` reports every parameter's
  offset and size. That is Decisions 44–46.
- **`test-backend-ops` compares through one function**,
  `ggml_backend_compare_graph_backend` (test-backend-ops.cpp:1493), which our
  Zig ggml exports. Compiling the test with that name `-D`-renamed to a
  harness function that runs backend 1 against **the reference CUDA
  backend instead of the CPU** and compares on **bits** turns upstream's
  ~21,000 op configurations into an exact port-vs-reference oracle. That is
  `make cuda-ops-diff`, below.

### The design

- **Device code** (Decision 44): every `.cu` the reference compiles is also
  compiled `nvcc -fatbin` with the same flags. The fatbins are embedded in
  `libggml.a` and loaded lazily, one `CUlibrary` per translation unit.
- **Kernel names** (Decision 45): a generated `kernels.zig` maps the
  demangled `name<args>` to the mangled symbol and its fatbin.
- **Launches** go through one helper, `launch(kernel, grid, block, shmem,
  stream, args)`, which takes a tuple, checks each element's size against
  `cuKernelGetParamInfo` in safe builds (Decision 46), and sets the PDL
  attribute exactly where upstream's `ggml_cuda_kernel_launch` does.
- **Runtime calls** — streams, events, memory, graphs, cuBLAS, the VMM
  pool's `cuMem*` — are the plain C APIs, through one `@cImport` of
  `cuda.h`, `cuda_runtime_api.h` and `cublas_v2.h` in
  `src/ggml/cuda/c.zig`. None of those headers includes ggml's, so this
  second `cImport` cannot produce incompatible ggml types.
- **CUB** (Decision 47): `src/ggml/cuda/cub_shim.cu`.
- **Layout**: `src/ggml/cuda/`, one Zig file per upstream `.cu` host layer,
  with `backend.zig` for `ggml-cuda.cu`'s device/buffer/backend/registry
  and `graph.zig` for its compute loop, fusion and CUDA graphs.

### The gates

| Gate | What it compares | Status |
|---|---|---|
| `scripts/cuda-ops-diff` | `test-backend-ops` with every case run on the **ported** CUDA backend and the **reference** CUDA backend in one process, compared on bits; also both backends' `supports_op` answers; plus harness-owned extra cases for paths upstream's list never reaches | **built**; 14,581/14,581 at the swap |
| `scripts/cuda-node-diff` | `node_dump` linked against `libggml` built with the reference host layer and with the port; every node of a Qwen3.5 decode, exact; then a **fused** pass (no per-node callback, so fusion and CUDA graphs run) over 33 decodes' logits | **built**; identical |
| `scripts/cuda-backend-diff` | the device, buffer-type and registry vtables' answers (`get_alloc_size` incl. flash attention at several addresses, alignment, memory, props, `offload_op`, `supports_buft`, features), the `metal-diff` of CUDA | **built**; identical |
| `scripts/cuda-parity-cli` | tokens from the shipped `llama-cli` against the reference driver on upstream's libllama and CUDA host layer | **built**; 6/6 |
| `cuda-struct-layout` | every by-value kernel argument struct: size, alignment, field offsets, from `nvcc` | **not built** — sizes are checked per launch by Decision 46; offsets are not |

As everywhere else in this repository, **each gate is negative-tested
before it is trusted.**

### How an op is ported

The pattern `scale.zig` and `binbcast.zig` set; follow it exactly.

- **One Zig file per upstream `.cu`**, `src/ggml/cuda/<name>.zig`, with
  the provenance header every ported file carries (full path, pin) and a
  citation on every ported function, `(binbcast.cu:160 @c1d0e7a00)`,
  pointing at the definition's first line.
- **Only the host code is ported.** `__global__` and `__device__` code is
  never translated: the kernels come from the fatbin of the same `.cu`.
- **A kernel is named `kernels.get("<tu>", "<key>")`.** `<tu>` is the
  translation unit whose host code launches it (the `.cu` stem, or the
  `template-instances/` stem for `mmq`, `mmf`, `fattn-*`). `<key>` is
  `cu++filt`'s spelling minus return type and parameters; build it with
  `kernels.tmpl`, `typeName`, `constPtr`, `int`, `long`, `ulong`, `boolean`,
  `ggmlType` rather than by hand. A wrong key is a compile error that
  names the nearest keys. `kernels.half` and `kernels.bf16` stand for
  `half` and `nv_bfloat16`.
- **Launch the way the C launches.** `kernel<<<g, b, smem, stream>>>(…)`
  is `kernels.launch`; `ggml_cuda_kernel_launch(kernel, params, …)` is
  `kernels.launchPdl` (programmatic dependent launch). Mixing them up
  passes every bit-exact gate and changes scheduling, so it is checked by
  reading, not by the gates. `CUDA_SET_SHARED_MEMORY_LIMIT` is
  `kernels.setSharedMemoryLimit`.
- **Pass every argument at the kernel parameter's exact type.** C++
  converts implicitly at the launch (`size_t` into a `uint32_t`
  parameter, `int64_t` into `int`); Zig must `@intCast`/`@truncate` to
  the parameter's width. The first launch of every kernel checks each
  argument's size against `cuKernelGetParamInfo` and aborts naming the
  argument, so a width mistake fails loudly. A C++ parameter pack
  (`src1_ptrs...`) is appended with `kernels.append`.
- **Mirror the host arithmetic's types.** `dim3` members are `unsigned
  int`; `std::min<unsigned int>(int64, …)` truncates first; integer
  division of signed values truncates toward zero (`@divTrunc`).
  `init_fastdiv_values` is `common.initFastdivValues`. Temporary device
  memory is `common.PoolAlloc(T)` with `defer x.deinit()` — **release
  order matters for the VMM pool**, which asserts frees come in reverse
  order of allocation, exactly as the C++ destructors run.
- **Wire it in `graph.zig`** — the case in `computeForward` and the same
  case in `isPorted` — and nothing else: `supports.zig` already holds the
  C's whole predicate.
- **Gate it**: rebuild (`zig build reference -Dcuda-port --release=fast
  --prefix zig-out/cuda-port`), then `scripts/cuda-ops-diff -o <OP>` for
  every op the file serves. Every case the reference supports must run and
  be bit-identical. **Then inject a fault** into the new host code — a
  wrong stride, a wrong launch dimension that changes the result, a
  swapped kernel variant — and confirm the gate fails, before reverting.
  An op whose cases are all skipped by `test-backend-ops` is not gated;
  say so.

### Order of work

Steps 1–7 are **done**; `NOTES.md`, "Stage 6", has what each gate
measured and caught.

1. **Docs and the gate machine** — this section. Done.
2. **Infrastructure.** Fatbin build, `kernels.zig` generation, `c.zig`, the
   launch helper with the ABI check, the reference CUDA objects compiled a
   second time with `ggml-cuda.h`'s C exports `-D`-renamed to `ref_*` for
   the in-process gates.
3. **The gates above**, negative-tested on the reference itself (a
   `-D`-injected fault in a copied reference object must fail them).
4. **`ggml-cuda.cu`'s skeleton**: device discovery (`ggml_cuda_init`), the
   legacy and VMM pools, the device and pinned-host buffer types, the
   backend and device vtables, the registry, events, and a compute loop
   with **no fusion and no CUDA graphs**. `supports_op` is the reference's
   logic AND-ed with "this op's host code is ported".
5. **Ops, simplest first, by family**, each gated by `cuda-ops-diff` as it
   lands: elementwise (`scale`, `unary`, `binbcast`, `clamp`, …), copies and
   indexing (`cpy`, `getrows`, `set-rows`, `concat`), norms, `softmax`,
   `rope`, then what Qwen3.5 needs (`ssm-conv`, `gated_delta_net`), then
   matrix multiplication (`mmvq`, `mmvf`, `mmq`, `mmf`, cuBLAS) and flash
   attention (`fattn-vec`, `-tile`, `-mma`), then the long tail.
6. **The compute loop's fusion (`ggml_cuda_try_fuse`) and CUDA graphs.**
   `cuda-ops-diff` has fusion cases; `cuda-node-diff` sees the real ones.
7. **The swap at `ggml-cuda.h`** when `cuda-ops-diff` shows `supports_op`
   equal on every case: upstream's host objects leave `libggml.a`, the
   reference stays only in the gate harnesses. **Done**: `-Dcuda-port`
   defaults to true and `isPorted` is gone.
8. **Next on this machine**: `cuda-struct-layout`; Makefile targets for the
   four CUDA scripts (left to the Mac pickup so `make` is changed where its
   macOS targets can be re-run); the multi-GPU collectives below if a
   second GPU appears.

Out of reach on one GB10 and left unported until a machine can run it:
the multi-GPU paths — peer copies, `ggml_backend_cuda_comm_*` (NCCL and the
internal all-reduce), `allreduce.cu`. The meta backend can still put the
same device in twice (`NOTES.md`, "meta-diff"), which reaches more of it
than it sounds.

### Ordering

CUDA came after Stage 4 finished libllama, as planned: `ggml-cuda.cu` is a
`ggml-backend` implementation, and the backend and scheduler interfaces it
plugs into are ported and gated.

### Mac pickup

**Done** (`NOTES.md`, "Mac pickup: the Mac gates after Stage 6"). Every
Mac gate passes over the files the CUDA work touched -- `make validate`,
`node-diff ARGS=--gpu`, `metal-diff`, `sched-diff`, `graph-diff`,
`parity-cli`, `backend-ops` -- after two fixes the CUDA machine could not
have found, because both are in gates it did not run:

- **`scripts/port-coverage` failed to compile anything.** Its stand-in
  `config` module lacked the three options the CUDA build added; the
  script's own comment says every option has to appear there.
- **`scripts/port-links` had never checked a CUDA citation.** It indexed
  `.c`, `.h`, `.cpp` and `.hpp` only, so all 589 citations of a `.cu` or
  `.cuh` read "no such file". With the extensions added, 34 real problems
  were left and are fixed: C++ member names the checker cannot parse
  (`A::b`, `T<U>`, `~A`) rewritten in the house ``A`'s `b`` form, seven
  line numbers that pointed at the wrong line, and three citations with no
  commit. 4,381 citations pass; negative-tested.

`node-diff` now runs the fused pass on both devices, and on Metal it
separates the way it did on CUDA: a fused `RMS_NORM + MUL` reading the
wrong buffer passes all 2,750 per-node comparisons and fails the fused one
from the prompt on. `zig build test` passes here (459/461, 2 skipped); the
GB10 failure moves to the CUDA pickup below.

### CUDA pickup

**Done** (`NOTES.md`, "CUDA pickup: the Mac's list, run on the GB10"). The
four `make cuda-*` targets pass, `zig build test` is green (the banner
only, as on the Mac), `port-links` passes, and `arch-model-diff` gives 157 /
159 on CUDA and the CPU, the Mac's result, with `gemma4-assistant-diff`
identical. Three things needed fixing first: the arch scripts linked Metal
(now `scripts/platform-libs`); linking `libstdc++.so` beside Zig's libc++
made every C++ exception in the reference abort; and T5's `q8_0` case
is a deterministic upstream abort on CUDA rather than Metal's
nondeterminism, so its skip is Darwin-only. `scripts/port-coverage` is
Mac-only by construction and was not run.

**Back to the Mac** -- **done** (`NOTES.md`, "Stage 5: the measurement").
`gen-arch-fixtures`, `arch-model-diff` and `gemma4-assistant-diff` take
their link line from `scripts/platform-libs`, and `arch-model-diff`'s
`abort_of` also reads a formatted `ggml_abort`. Run here: 159 fixtures
written, `arch-model-diff` 157 / 159 with the same two `REF-ABORT` and T5 at
`gpu:17+3abort+1nondet`, `gemma4-assistant-diff` identical on Metal and the
CPU -- unchanged, as expected.

### mtmd pickup

Decision 62 is gated on this Mac only. **For the GB10**: `make
mmproj-fixtures` and `make mtmd-diff` there, on the CPU, CUDA and Vulkan.
Unmeasured off Darwin: the audio preprocessor's `__sincosf_stret` and
`__exp10` folds, keyed on Darwin as ggml's `simd.math_errno` is; the
encoder graphs on CUDA and Vulkan, which no gate has run; and
`helper.zig`'s glibc `strerror`/`errno` path; `make tools-diff TOOL=tts`
and `TOOL=mtmd-cli` too. dr_mp3's x86 synthesis rounding (`_mm_cvtps_epi32`)
is written and ungated. **For Windows**: libmtmd compiles for
`x86_64-windows-gnu` and has not run there -- `helper.zig`'s `_wfopen` arm and
`clip.zig`'s `rand_s` seed are unrun -- and the two tools inherit the tools'
`clock_gettime` break below; `console.zig`'s Windows arm is not ported, so
`llama-mtmd-cli` reads plain lines there.

### Tools pickup

Decision 61's ten tools are gated on this Mac only. **For the GB10**: run
`make tools-diff` there. The Linux paths are untested: `io.zig`'s glibc
`stdout`, `harness/tools/fixed_time.c` through `LD_PRELOAD`, and the
reference built with `-fmath-errno` (the perplexity's `logf`/`expf` sites).
On a CUDA build `cvector-generator`'s PCA runs on CUDA on both sides
(`GGML_USE_CUDA` reaches the reference's tool sources, `backends.cuda` the
port's), a path no case has run. **For Windows**: the tools are not yet
built for `x86_64-windows-gnu`; `io.zig`'s `__acrt_iob_func` arm and
`init.zig`'s `SetPriorityClass` arm are written and unrun, and the clock pin
has no Windows interposer, so the PCA cases need another way to fix `time`
there.

---

## Stage 5 — Cross-platform

Targets: macOS, iOS, Linux arm64, Linux x86_64, Windows x86_64.

This section is the measurement, the same way Stage 4 and Stage 6 began:
cross-builds from this Mac, `zig cc -E` live-line sets per target and CPU
level (`NOTES.md`, "Stage 5: the measurement"), and the bit-exact gates run
at the CPU configuration Linux arm64 actually builds for.

### Where each target stands, measured

| Target | Builds | Gated | What is left |
|---|---|---|---|
| macOS arm64 | yes | every gate in `CLAUDE.md` | nothing |
| iOS, tvOS, visionOS, and their simulators | yes — `llama.xcframework`'s seven slices, upstream's (Decision 63): devices at `apple_a7` (iOS, tvOS) and `apple_a12` (visionOS), simulators at `apple_m1` and `x86_64_v3` | `make -C xcframework check`: structural gates per slice, `compat` against upstream's own xcframework, `smoke-sim` on the iOS, tvOS and visionOS simulators; `smoke-device` and `swiftui` on an iPhone 15; the device CPUs through `GATE_CPU=apple_a7` and `apple_a12` | the x86_64 slices are built and linked, never run: no Intel Mac, no Rosetta here; tvOS and visionOS devices unrun; the increased-memory-limit entitlement is the consuming app's |
| Linux arm64 + CUDA | yes, natively on the GB10 | the four `cuda-*` gates, `arch-model-diff`, `gemma4-assistant-diff` | the CPU path's numerics — below |
| Linux arm64, CPU only | **yes**, cross-built from this Mac against glibc 2.39 | `GATE_CPU=generic` here. Measured failing — `ops-diff` 86 / 632, `repack-diff` 26 / 60, `node-diff` from node 27, `parity-port --cpu` 2 / 6 — and **exact since step 2**: 650 / 650, 60 / 60, 2,750 nodes, 6 / 6. **On the GB10 itself** (the pickup, done): `ops-diff` 650 / 650, `repack-diff` 68 / 68, `node-diff` 2,750 nodes, `parity-port --cpu` 6 / 6, `host-arms-diff` 18 / 18, `vocab-diff` -- after two Linux-only fixes (`NOTES.md`, "The GB10 pickup, run"). **On an Arduino UNO Q** (four Kryo cores, exactly the `generic` baseline, 3.6 GB): our `llama-cli` against `raw_completion.c` over the zmake reference, 7 / 7 on two models at default flags (`NOTES.md`, "The UNO Q, and `--fit`") | `set_numa_thread_affinity` needs a multi-socket machine. Vulkan on the UNO Q's Adreno 702 -- **done**: parity 7 / 7 and `vk-node-diff` identical on the board (`NOTES.md`, "Vulkan on the UNO Q's Adreno 702") |
| Linux x86_64 | **yes, at x86-64-v3**, cross-built; the same kernels as Windows | the x86 gates run on Windows; Linux x86 shares every kernel and differs only in its OS arms | its OS arms on Linux x86 hardware |
| Windows x86_64 | **yes, cross-built from this Mac** (`x86_64-windows-gnu`, x86-64-v3): libraries, `libllama-ref`, and our `llama-cli.exe`; the reference's `llama-cli.exe` through `make -C llama.cpp.zmake windows-x86_64` | **exact on the laptop**: `make x86-gates` -- 28 quant kernels, 68 repack checks, 712 `ops-diff` results, 2,750 decode nodes -- and 6 / 6 parity prompts; the host gates `make win-host-diffs` -- `host-arms-diff`, `libcxx-diff`, `regex-diff`, `unicode-diff`; at AVX-512 (`x86_64_v4`, `tigerlake`), the four `x86-*` gates exact on `JWHLPTOP` | the AVX512-BF16 and -FP16 arms, on hardware with them |

### Linux arm64: a different CPU from the one every gate ran on

**Zig's baseline for `aarch64-linux` is `generic`, not `apple_m1`.** It
has NEON and FMA, but no `dotprod` and no `fullfp16`; `zig cc -dM` on it and
on `aarch64-macos -mcpu=generic` give the same `__ARM_FEATURE_*` set. The
GB10 builds at that baseline, so its CPU device compiles C arms no Mac gate
had ever compiled:

- **`simd-mappings.h`'s F16 arm without
  `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`** (simd-mappings.h:303): `f16`
  vectors are widened to `f32x4` on load, computed in `f32` with
  `GGML_F16_STEP` 16 / `GGML_F16_EPR` 4, and narrowed on store. The port
  has only the other arm — `f16x8` arithmetic, step 32 — in `vec.zig`,
  `ops/vecinline.zig` and `ops/sgemm.zig`'s `tinyBLAS<8, f16x8>`. That is a
  different computation, not a different schedule of the same one.
- **The `#else` arms of `__ARM_FEATURE_DOTPROD`** in `arch/arm/quants.c`:
  `nvfp4`, `tq1_0` and `tq2_0` (80 lines; the `tq` ones accumulate in
  `int16x8_t` through `vmlal_s8`), and `ggml_vdotq_s32`'s fallback in
  `ggml-cpu-impl.h:310`. Integer, so plausibly exact — which is what a gate
  is for.
- llamafile's `tinyBLAS_Q0_ARM` (sgemm.cpp:1216) is behind
  `__ARM_FEATURE_DOTPROD` and compiled out, and its f16 `tinyBLAS<8,
  float16x8_t>` arm is behind `FP16_VECTOR_ARITHMETIC`; the C falls to
  `tinyBLAS<4, float32x4_t>` over f16 A and f32 B. The port's `sgemm.zig`
  has neither guard.
- In `arch/arm/repack.cpp` every `dotprod` kernel falls to its `_generic`
  body (`ggml_gemv_q4_K_8x4_q8_K` → repack.cpp:706). The port always runs
  the NEON body.

**None of the GB10's gates compared the CPU kernels.** `context-diff` and
`arch-model-diff` put `libllama-ref.a` beside the port's libllama over **one**
`libggml.a`, so ggml is not a side of the comparison; `ops-diff`,
`repack-diff` and CPU `node-diff` never ran there. So `GATE_CPU=generic`
on this Mac — whose feature macros are the GB10 baseline's — is the first
bit-exact gate the Linux arm64 CPU path has had, and **it fails**:

| gate at `GATE_CPU=generic` | result | what differs |
|---|---|---|
| `ops-diff` | **86 of 632** | f16 `mul_mat` and every f16-K/V `flash_attn_ext` (the F16 arm); `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q1_0`, `q2_0` `mul_mat` (sgemm's guard, and the `#else` dot-product arms); `ssm_conv`, `ssm_scan`, `conv_2d_direct`, `conv_3d_direct` (below) |
| `repack-diff` | **26 of 60** | last-bit in the `q4_0`/`q8_0`/`iq4_nl`/`mxfp4` 4x4 and 4x8 kernels; **gross** in every `q4_K`/`q5_K`/`q6_K` 8x4 and 8x8 one, where the C runs `_generic` and the port the NEON body. The dispatcher selects no repack type at `generic` on either side, so these are exported and unreachable, as the `_generic` dot products are on the Mac |
| CPU `node-diff` | **differs from node 27**, `SSM_CONV` in Qwen3.5's first layer, at 1 and 4 threads; fused decodes differ from step 0 |
| `parity-port --cpu` | **2 of 6 prompts** generate different tokens |

At `baseline` the same tree is green (`ops-diff` 650 / 650), so the
failures belong to the CPU, not to the tree.

**`ssm_conv` is the finding that matters beyond this target.** Its inner
loop is plain scalar C, `sumf += s[…] * c[…]`, and the port reproduces
what clang made of it — `ops/common.zig`'s `strictLoopVectorized`: a
strict in-order reduction over rounded vector products, with the leftover
elements fused. Disassembled at both CPUs, clang interleaves that loop
**four `fmul.4s` wide for `apple_m1` and two for `generic`**, so the
vector body covers 16 elements an iteration on one and 8 on the other, and
a different set of elements falls to the fused tail. **The target CPU's
cost model is part of the reference's arithmetic.** Every
`strictLoopVectorized` and `vectorizedTail` site was measured for one CPU,
and is right for that CPU only. x86 will have its own at every level.

**Three host-OS arms took the Apple branch unconditionally:**

- `ggml_thread_apply_affinity` and `ggml_thread_apply_priority`, whose
  Linux arms are ggml-cpu.c:2614 and 2643. `threading.zig` has the
  Apple arm with macOS's constants — `SCHED_OTHER = 1`, `SCHED_FIFO = 4`,
  an 8-byte `sched_param`. On Linux those are 0 and 1, `PRIO_LOW` is
  `SCHED_BATCH`, and affinity is a real `pthread_setaffinity_np`. Our CLI
  exposes no `--prio` or `--cpu-mask`, so only a library caller setting
  `ggml_threadpool_params` reaches it — where it is wrong today.
- NUMA: `ggml_numa_init`, `set_numa_thread_affinity` and the `cpuset` in
  `g_state` (ggml-cpu.c:636, 2148-2212). The port has the non-Linux `#else`
  only, so `ggml_is_numa` is always false on Linux.
- `get_rng_seed` (llama-sampler.cpp:340) calls `arc4random`, which is
  libc++'s Apple arm. On Linux, libc++'s `random_device` is
  `_LIBCPP_USING_DEV_RANDOM` — `/dev/urandom`, with `entropy()` from the
  `RNDGETENTCNT` ioctl, so a container without that ioctl takes the
  `system_clock` branch. `arc4random` is also what gives the Linux binary a
  **glibc 2.36 floor**: the default `aarch64-linux-gnu` cross-build fails to
  link on it and nothing else.

Everything else in libllama already has its Linux arms: measured over every
`src/llama*.cpp`, the only Linux-only live lines are `llama-mmap.cpp`'s
direct I/O, which `mmap.zig` ports, and one line of `llama-quant.cpp`. In
ggml, `ggml.c`'s Linux backtrace and `ggml-cpu.cpp`'s `/proc/cpuinfo`
description are ported.

### Linux x86_64: a second CPU backend

Live lines of upstream that x86_64 compiles and the ported ARM build does
not, per level (`zig cc -E`, both libraries' CPU translation units):

| file | v1 | v2 (SSE4.2) | v3 (AVX2, FMA, F16C) | v4 (AVX-512) |
|---|---:|---:|---:|---:|
| `arch/x86/quants.c` | 461 | 542 | 1,525 | 1,520 |
| `arch/x86/repack.cpp` | 129 | 129 | 2,365 | 4,319 |
| `llamafile/sgemm.cpp` | 32 | 32 | 410 | 431 |
| `ggml-cpu.c` | 155 | 157 | 191 | 211 |
| `vec.h` + `vec.cpp` | 86 | 62 | 95 | 66 |
| `ops.cpp` (scalar `#else` arms) | 94 | 58 | 0 | 0 |
| other (`ggml.c`, `ggml-cpu.cpp`, `ggml-quants.c`, `simd-*.h`) | 73 | 75 | 87 | 85 |
| **total** | **1,030** | **1,055** | **4,673** | **6,632** |

About 200 of those lines are Linux rather than x86: ~150 of `ggml-cpu.c`'s
are the host arms above, shared with Linux arm64, and `ggml.c`'s 31 and
`ggml-cpu.cpp`'s 21 are Linux arms the port already has.

**That table undercounts, and the reason is the central fact of this
stage.** `simd-mappings.h` is macros, so a `GGML_F32_VEC_FMA` loop in
`ops.cpp` or `vec.h` is the *same live line* on both targets and expands to
different code. x86 has four `GGML_SIMD` configurations, not one:

| level | `simd-mappings.h` arm | `GGML_F32_EPR` | `GGML_F32_STEP` | reduction |
|---|---|---:|---:|---|
| v1 (Zig's `x86_64` baseline: SSE2, no SSE3) | none — `GGML_SIMD` undefined | — | — | the scalar loops |
| v2 | `__SSE3__` | 4 | 32 | `GGML_F32x4_REDUCE`, SSE |
| v3 | `__AVX__` | 8 | 32 | `GGML_F32x8_REDUCE` |
| v4 | `__AVX512F__` | 16 | 64 | `GGML_F32x16_REDUCE` |
| *ported (NEON)* | `__ARM_NEON` | 4 | 16 | pairwise `vaddvq_f32` |

The port hard-wires the NEON row: `ops/common.zig`'s `f32x4` and
`addvq_f32`, imported straight from `quants/arm/neon.zig` by five files
outside `arm/` — `vec.zig` and `ops/{common,recurrent,ssm,sgemm}.zig` —
while `quants/module.zig` and `repack/module.zig` import their `arm/`
barrels unconditionally. The macros are used at **335
sites** in `ops.cpp`, `vec.h` and `vec.cpp`. Step and lane count change
which elements share an accumulator, and the reduction changes the order
they meet; either moves the last bit. And `vec.h` has x86-only bodies too —
`ggml_v_expf`, `ggml_v_silu` and friends in AVX2 and AVX-512.

The intrinsics: **244 distinct `_mm*`** across the live x86 files, against
the 82 NEON ones `neon.zig` reproduces. `vmmlaq_s32` showed the route —
Zig calls LLVM intrinsics by name (`llvm.x86.avx2.pmadd.ub.sw` and so on) —
and most are a line of `@Vector` arithmetic.

### Three questions x86 needed answered, now Decisions 51–53

Settled: **a laptop** is the x86 gate machine (Decision 51), **`llama.cpp.zmake` gains `arch/x86`** (Decision 52), and **AVX2 ships first, then AVX-512** (Decision 53). What was measured to put the questions:

- **The x86 gate machine.** Neither machine here runs x86:
  this Mac (macOS 27) has no Rosetta installed, and the GB10 is aarch64.
  Every x86 kernel is bit-exact-or-wrong exactly as the ARM ones were, and
  the project's rule is no oracle, no port. The options, measured as far as
  they can be from here:
  1. **A real x86_64 Linux host**, ideally Zen 4 or later so AVX-512 runs
     natively too. The CUDA precedent (Decision 43).
  2. **Rosetta 2** on this Mac — AVX2 but no AVX-512, not installed, and
     Apple is retiring it.
  3. **QEMU user-mode** in the Podman VM this Mac already has — AVX2 since
     QEMU 7.2, no AVX-512. Both sides of every gate run under the same
     emulator, so its exactness matters less than its determinism, but it
     is an emulator vouching for kernels by emulating them.
- **The x86 reference.** `llama.cpp.zmake`'s `build.zig`
  refuses any target but aarch64 (build.zig:361, "the ggml source list is
  arm64 only"), and Decision 18 keeps that repository read-only. Either it
  gains `arch/x86` under an explicit grant, as the 0.17 migration was, or
  the x86 gates compare against our own `build/llamacpp.zig` reference,
  which compiles upstream unmodified and is ours to change.
- **Which x86 levels ship.** A static binary compiles one
  `GGML_SIMD` arm; upstream's answer, `GGML_CPU_ALL_VARIANTS`, needs
  `GGML_BACKEND_DL`, which is out of scope. Recommended: **v3** first —
  AVX2/FMA/F16C, every x86_64 CPU since Haswell — then v4. v1 and v2 are
  the scalar and SSE arms and cheap once the SIMD layer is parameterised,
  but nothing needs them.

### Order of work

Linux arm64 first: it has a gate machine here (`GATE_CPU=generic`), it is
small, and it is wrong today -- in its numerics and in its host arms.

1. ~~**Measure Linux arm64's CPU path**~~ — **done**, above: it fails at
   `GATE_CPU=generic` in four classes.
2. ~~**Make the CPU path exact at `generic`**~~ — **done** (`NOTES.md`,
   "Stage 5 step 2"). `ops-diff` 650 / 650, `repack-diff` 60 / 60, CPU
   `node-diff` 2,750 nodes and 33 fused decodes, `parity-port --cpu` 6 / 6,
   all at `generic`, with `baseline` and `apple_m3` unchanged, and `zig
   build test` green at all three. Two things beyond the plan below: the
   strict-loop width is `features.strict_loop`, keyed on the CPU **model**
   and a compile error for any model not measured; and the ARM and float dot
   goldens have a `generic` set (`GOLDEN_CPU=generic` on both scripts),
   chosen by `features.golden_generic`. The plan as written was: each class chosen on
   `builtin.cpu` exactly as the C's `#if` or the C compiler chooses:
   - the F16 arm without `fullfp16` in `vec.zig` and `ops/vecinline.zig`;
   - sgemm's two guards, so `generic` gets `tinyBLAS<4, f32x4>` over f16
     and no `tinyBLAS_Q0_ARM`;
   - the `#else` arms of the `dotprod` dot products, localised first with
     `scripts/vecdot-prefix` — integer accumulation should be exact, so a
     difference there is in an epilogue;
   - every `strictLoopVectorized` / `vectorizedTail` site re-measured from
     the reference's disassembly at `generic` and made a function of the
     CPU, not a constant;
   - the repack `_generic` fallbacks, reached the way the C reaches them.

   Gate: `ops-diff`, `repack-diff`, CPU `node-diff` and `parity-port
   --cpu` at `generic`, then at `baseline` and `apple_m3` to show nothing
   moved. **`GATE_CPU=generic` joins the gate set**, since it is the Linux
   arm64 baseline. The vector goldens (`scripts/vec-golden`,
   `vecdot-golden`) are captured at one CPU and need a `generic` set.
3. ~~**The Linux host arms**~~ — **written; runtime gate on the GB10**
   (`NOTES.md`, "Stage 5 step 3"). Priority, affinity and NUMA in
   `threading.zig`, each beside the Apple arm and chosen as the C's
   `__linux__` / `__gnu_linux__` chooses; `get_rng_seed`'s
   `_LIBCPP_USING_DEV_RANDOM` arm in `sampler/core.zig`. The default
   `aarch64-linux-gnu` cross-build now links, and needs glibc 2.29, not
   2.36. On the Mac every gate is unchanged. Nothing here can run them, so
   their gate is the GB10 item below.
4. ~~**A CPU-only `make validate` on Linux.**~~ — **done on the Mac's side;
   to be run on the GB10** (`NOTES.md`, "Stage 5 step 4"). The eight
   in-process diffs take their link line from `scripts/platform-libs`, whose
   Linux arm now adds CUDA only where the toolkit is installed.
   `scripts/port-coverage --target <triple>` measures another platform's
   contract by cross-compiling, and reads the host's by default: at
   `aarch64-linux-gnu` every unit is 100%, the Metal units are reported as
   not part of the target, and `src/ggml/varargs.c` counts where the build
   compiles it. `sched-diff` and `graph-diff` keep their Apple-clang CMake
   reference and are skipped, with a message, where it is absent; what they
   compare is structure, which the Mac covers. `validate`'s `sched-diff`
   line also no longer swallows a failure.
5. ~~**The SIMD layer.**~~ — **done** (`NOTES.md`, "Stage 5 step 5").
   `cpu/simd.zig` is `simd-mappings.h`'s `GGML_SIMD` block: `F32` and
   `F16` (both NEON `F16` arms, `F16.half_precision` telling them apart),
   chosen at comptime on `builtin.cpu`, plus the vectorizer's strict-loop
   shape, moved there from `features.zig`, `vec.zig` and `ops/common.zig`.
   Six files imported `neon.zig` directly, not five, and two more
   hard-coded `GGML_F32_EPR` as 4; all of them now go through it. The two
   float dot products, `rwkv_wkv7` and `ssm_scan` are written against
   `F32`/`F16` as the C writes them, which merged `ggml_vec_dot_f16`'s two
   arms into one body. Gate: **the machine code did not change** — the
   port's `libggml.a` at `baseline`, `apple_m3` and `generic` disassembles
   identically before and after, apart from a register rename in the two
   float dot products' vector loops — plus `zig build test` at all three
   and `make validate`. Any other architecture is a compile error naming
   step 6.
   The x86 reference is ready (Decision 52, `NOTES.md`, "Decision 52:
   `llama.cpp.zmake` builds x86_64"): `make -C llama.cpp.zmake
   linux-x86_64`, AVX2 by default, `X86_CPU=x86_64_v4` for AVX-512, both
   cross-built and checked for their instructions.
6. **x86_64** (Decisions 51–53) -- **done at AVX2, and exact on the
   Windows laptop's Haswell** (`NOTES.md`, "Stage 5 step 6: x86 begins" and
   "Stage 5 step 6: the AVX2 kernels"). **AVX-512 is done too**, at
   `x86_64_v4` and `tigerlake`, exact on `JWHLPTOP` (`NOTES.md`, "Stage 5
   step 6: AVX-512").
   - **Done first, exact against the x86-64-v3 reference under QEMU**: every
     shared function with an x86 arm, found by preprocessing each CPU
     translation unit for both targets and diffing them function by
     function -- `simd.zig`'s AVX arm, the strict-loop shape (never
     vectorized at v3), the F16C conversions, the feature predicates,
     `ggml_thread_cpu_relax`, `vec.cpp`'s AVX2 SiLU/softmax/`cvar` and
     `dot_bf16`, `vec.h`'s `mad_f16`/`scale_f16`, and `sgemm`'s
     `tinyBLAS<8, __m256>` for f32, f16 and bf16. `ops-diff` there: **492
     of 650 identical, and every one of the 158 that differ is a quantized
     path a stand-in feeds.**
   - **Then the 39 kernels and one class**: `arch/x86/quants.c`'s 27
     exports, `arch/x86/repack.cpp`'s 12 and `tinyBLAS_Q0_AVX`, over
     `quants/x86/avx2.zig`, the intrinsic layer as `neon.zig` is for ARM --
     plus `q2_0`, whose generic body `arch-fallback.h` makes the live x86
     kernel and which clang ships fused. `-Dx86-incomplete` and its stand-ins
     are gone; an x86 build is the real thing.
   - **Gated on the laptop**, four scripts behind `make x86-gates`, 2m17s:
     `x86-quants-diff` (28 kernels, 2,400 trials each), `x86-repack-diff`
     (68 checks), `x86-ops-diff` (712 results at 1 and 3 threads) and
     `x86-node-diff` (2,750 nodes of the Qwen3.5 2B `Q4_K_M` decode at 1
     and 4 threads, and the fused pass), all identical; the six
     `parity-port` prompts, 6 / 6 at 64 tokens. Our `llama-cli.exe`
     generates at 14.3 t/s against the reference's 14.5 on the same
     machine. Each of nine injected faults failed its gate.
   - **Left at AVX2**: three rarely used kernels are slower than the C's,
     though exact -- `tq1_0` 1.6x, `iq1_m` 1.4x and the generic `q2_0`
     1.6x -- and prompt processing runs at ~0.9x. The Windows builds of
     `regex-diff`, `unicode-diff`, `libcxx-diff` and `host-arms-diff` have
     since run there and pass (`NOTES.md`, "The Windows pickup's host
     gates, run").

### Windows x86_64

Measured the way Linux was: every ggml and libllama translation unit
preprocessed for `x86_64-linux-gnu` and `x86_64-windows-gnu` at x86-64-v3,
the live lines per upstream file subtracted. **400 lines are Windows-only,
in six files**, and every one is now ported:

| file | Windows-only | what |
|---|---:|---|
| `src/llama-mmap.cpp` | 189 | `llama_file` over the Win32 handle behind its `FILE *` (`ReadFile` in 64 MiB chunks, `has_direct_io` true at alignment 1), `CreateFileMappingA` / `MapViewOfFile` / `PrefetchVirtualMemory`, `VirtualLock` growing the working set once |
| `ggml-cpu/ggml-cpu.c` | 87 | SRW locks and condition variables -- the `_shared` lock and wait are a reader's hold there, not the plain lock -- affinity through `SetThreadAffinityMask`, priority and power-throttling opt-out |
| `ggml.c` | 56 | `_aligned_malloc`, the `QueryPerformanceCounter` timer, `ggml_fopen` through `_wfopen`, no backtrace |
| `ggml-cpu/ggml-cpu.cpp` | 32 | the registry's `ProcessorNameString`, `GlobalMemoryStatusEx` |
| `ggml-backend-dl.cpp` + `.h` | 23 | `LoadLibraryW` with the error dialog suppressed, `GetProcAddress`, `FreeLibrary` |
| `ggml-backend-reg.cpp` | 13 | `GetModuleFileNameW`, `ggml-*.dll`; with libc++'s `std::filesystem`, a `FindFirstFileW` scan and `\` joins |

**What a line diff cannot see**, found by reading what the same lines mean
on Windows, and ported:

- `std::mutex` is an `SRWLOCK` in libc++ there; `impl.CxxMutex` (ggml and
  libllama each) drives whichever the target's libc++ uses.
- `std::random_device` is `_LIBCPP_USING_WIN32_RANDOM`, whose `entropy()`
  is 0, so `get_rng_seed` always takes `system_clock` -- which libc++ reads
  from `GetSystemTimePreciseAsFileTime`.
- `wchar_t` is 16 bits, so `unicode_regex_split`'s `std::wregex` fallback
  narrows every code point above U+FFFF, in the pattern and the text alike.
- `stdout` and `stderr` are `__acrt_iob_func(1)` and `(2)`, not symbols.

**Three toolchain facts it took**, each a build failure first:

- Zig 0.17 refuses `std.lang.VaList` on x86_64 Windows under LLVM
  ("disabled due to miscompilations"), as on aarch64 Linux, so
  `src/ggml/varargs.c` defines ggml's three variadics there too.
- The Win64 C convention passes a vector by reference, so an `extern fn
  @"llvm.x86..."` declared with the default `.c` reaches LLVM with a pointer
  operand and aborts the build ("Intrinsic has incorrect argument type").
  The x86 intrinsics are declared `.x86_64_sysv`, which is what `.c` means on
  Linux anyway. **Every intrinsic the AVX2 kernels add must be declared the
  same way.**
- MinGW's `<math.h>` hides `M_PI` under `__STRICT_ANSI__`, and
  `tools/mtmd/mtmd-audio.cpp` defines `_USE_MATH_DEFINES` too late. CMake
  never sees it because it compiles `-std=gnu11`/`gnu++17` everywhere
  (`C_EXTENSIONS`/`CXX_EXTENSIONS` default on); zmake's other platforms use
  the strict dialects they were verified and calibrated with, so only its
  Windows platform takes CMake's.

**Settled by a run on Windows** (`NOTES.md`, "The Windows pickup's host
gates, run"): the UCRT's "C" locale is ASCII-only, as Apple's is. The one
difference is `\t`, which its table does not mark `_BLANK`, so
`[[:blank:]]` misses a tab in `std::regex` and `std::wregex` alike;
`regex.zig` has the arm.

### Windows pickup

Run on the Windows laptop (Decision 57), through `zupdater push` and
`tty`:

- **Done.** `make x86-gates` -- `x86-quants-diff`, `x86-repack-diff`,
  `x86-ops-diff`, `x86-node-diff` -- all identical, and the six
  `parity-port` prompts 6 / 6 (`NOTES.md`, "Stage 5 step 6: the AVX2
  kernels"). Each script cross-builds here and runs through
  `scripts/win-run`, which pushes the program, runs it, and cleans what
  ConPTY sends back; the gates compare on the laptop and only the verdict
  returns. Before the kernels, at `-Dx86-incomplete`: 4 / 6 prompts, and
  `node-diff` diverging at node 13, a `q5_K` stand-in.
- ~~`regex-diff`, `unicode-diff`, `libcxx-diff` and `host-arms-diff` built
  for Windows~~ -- **done**: `make win-host-diffs`, all four identical --
  168 host-arms lines, 452,904, 43,537,552 and 7,070,454 checks -- after
  the `\t`-is-not-blank arm in `regex.zig` and a narrowing fix in the
  regex harness (`NOTES.md`, "The Windows pickup's host gates, run").

### GB10 pickup

**Done** (`NOTES.md`, "The GB10 pickup, run"). It found two bugs the
container could not: Zig's `compiler_rt` shadowing glibc's `expf` and
`logf` in every CUDA build -- through the CUDA layer's `@panic("OOM")` --
and `char` being unsigned on aarch64 Linux in the RWKV unescape. Both
fixed; `libm_check` now guards the first in every bit-exact gate. The
checkout there needs a `llama.cpp.zmake -> ../llama.cpp.zmake` symlink. The
list as it was written:

Run on the GB10 after steps 2–4. Everything here is written, and the gate
scripts now take their link line from `scripts/platform-libs`, so they run
there unchanged (`NOTES.md`, "Preparing the GB10 pickup"). A Linux arm64
container on this Mac -- the podman VM, `debian:stable-slim`, glibc 2.41,
binaries cross-built here -- already ran three of them natively and
passed: `ops-diff` 650 / 650, `repack-diff` 60 / 60, and every
host-arms scenario. What is left is the hardware and a model:

- `ops-diff`, `repack-diff`, CPU `node-diff` and `parity-port --cpu` at its
  baseline -- `node-diff` and `parity-port` load a model, which the
  container's 2 GB VM cannot hold. `ops-diff` there confirms the two things
  settled from the Linux reference's objects rather than from a Mac:
  `sinCos` calling `sinf` and `cosf` separately, and `ssm_scan`'s Mamba-1
  loop staying scalar and fused (`simd.math_errno`).
- `make host-arms-diff` on real hardware: priority at all five levels, CPU
  masks loose, strict, paused and invalid, every NUMA strategy and a second
  init, and the default seed's source. The GB10 has one NUMA node, as the
  container does, so `set_numa_thread_affinity`'s body stays unreached; it
  needs a multi-socket machine.
- `make validate`: everything but `sched-diff` and `graph-diff`, which
  report themselves skipped.

---

---

## Stage 7 — Vulkan

Decisions 54–56: gated on the GB10, against `llama.cpp.zmake -Dvulkan`,
started now beside Stage 5. Nothing is ported yet; this is the measurement,
taken on this Mac from the pinned source without a Vulkan driver.

### Scope, measured

| | measured | for comparison |
|---|---|---|
| host code | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, **one C++ translation unit, 19,787 lines**: 23 `throw`, 17 `catch`, 19 maps, 27 thread/mutex uses, 107 `#if`; written against `vulkan.hpp`, the C++ binding (497 `vk::` uses) | Metal's ported C++ was ~10,200 lines; CUDA's whole backend ~43,900 |
| shaders | 147 `.comp` and 26 `.glsl`, ~21,400 lines of GLSL; **1,440 SPIR-V variants with no optional extension, 1,980 with all seven**, plus 86 dispatch tables and the `arr_dmmv_*` arrays | Metal: 20 kernel files, embedded as source |
| generator | `vulkan-shaders/vulkan-shaders-gen.cpp`, 1,355 lines: a host build tool that drives `glslc`; nothing of it ships | `build/metal_embed.zig` |
| contract | **9 C exports**: the 8 in `ggml-vulkan.h` and `vk_conv_block_sizes`, an internal table that is missing its `static`. Imports: 32 `ggml_*` functions, all C; 4 loader entry points (`vkGetInstanceProcAddr`, `vkGetDeviceProcAddr`, `vkGetPhysicalDeviceFeatures2`, `vkCmdCopyBuffer`), everything else through `vkGetInstanceProcAddr`; 1,185 shader symbols by name. **No C++-linkage symbol crosses it.** | `ggml-cuda.h`: the same shape |
| host → shader ABI | 46 push-constant structs, matched by blocks in 50 shader files; 398 pipeline-creation sites | Metal's 65 `kargs` structs, `make struct-layout` |

Measured by compiling `ggml-vulkan.cpp` for `aarch64-linux-gnu` against
Khronos's headers at SDK 1.4.363 and the generator's own header, and running
the generator's header pass at both extension extremes. The first pass at
both numbers was wrong -- zsh passed the seven `-D` flags as one argument --
and was caught because the zmake build, which passes them properly,
declared 1,980 variants where the hand measurement said 1,656.

### What decides what compiles

Unlike every backend so far, **the source does not fix the reference**:

- **`glslc`'s answers.** Seven extension probes decide seven macros, which
  gate 64 `#if` sites in `ggml-vulkan.cpp` and which variants the generator
  emits (1,440 to 1,980).
- **The Vulkan header version.** `VK_HEADER_VERSION >= 287` and `>= 301`
  arms, and `#if defined(VK_NV_cooperative_matrix2)` and friends: an older
  header compiles cooperative-matrix-2 out entirely.
- **`glslc`'s optimizer.** Most variants are compiled `-O`, so the SPIR-V
  bytes depend on the `glslc` build.

So the toolchain is part of the reference, like the upstream pin, and gets
the same treatment: one `glslc` and one set of headers build both sides of
every Vulkan gate, recorded with every result (`scripts/vk-repro` prints
them), and the port embeds the *reference generator's output* rather than
compiling its own -- identical SPIR-V by construction, as CUDA's fatbins are
(Decision 44).

### The method, proposed

To be confirmed by the reproducibility measurement, but the shape follows
from the contract:

- **Shaders travel as data.** Upstream's generator, built unmodified, writes
  the SPIR-V **once**, in the reference build, and the port embeds those
  files. Running the same tool again with the same `glslc` is not the same
  reference: `tri_f16` is not reproducible (Decision 59).
- **The host is Zig against Vulkan's C API.** `vulkan.hpp` is a C++
  wrapper over a plain C API Zig can call directly, through
  `vkGetInstanceProcAddr` as the C++'s dynamic dispatcher does -- easier
  than Metal (Objective-C) or CUDA (`<<<>>>`). `vk::SystemError` throws
  become `VkResult` checks.
- **Variants by generated table.** 1,440–1,980 names and the dispatch tables,
  read from the generator's header at build time, as CUDA's kernel table is
  read from the fatbins (Decision 45).
- **Push constants checked by the compiler.** The 46 structs are the Metal
  `kargs` problem again -- one wrong offset silently feeds a shader another
  number -- so they are generated or checked against the C++'s `offsetof`,
  `make struct-layout`'s way.
- **Swap once, at `ggml-vulkan.h`**: 9 symbols, all-or-nothing, as Decisions
  42 and 48 did for libllama and CUDA.

### Gates, planned

- `make vk-repro` -- **written**: the reference's node hashes, per node and
  fused, run against themselves. Everything below assumes it passes.
- `vk-ops-diff`: `harness/ops_dump.c` on the Vulkan device, port against
  reference, on bits.
- `vk-node-diff` -- **written**: `node-diff --gpu` over a Vulkan-only build,
  per node then fused, on all four arms.
- `vk-backend-diff` -- **written**: the vtables' answers, one side per
  process under 13 environments; grows with each layer (buffer types,
  `get_alloc_size`, `supports_op`).
- `vk-push-layout` -- **written** (the planned `vk-struct-layout`): the 46
  generated push-constant structs, 1,266 facts, against the C++.
- `vk-ops-diff` -- **written**: test-backend-ops with the reference where
  the CPU goes, on bits, plus extra cases upstream lacks.
- `vk-parity-cli` -- **written**: tokens through our `llama-cli`.

### Order of work

1. ~~**Measure the source**~~ -- done, above.
2. ~~**The reference build**~~ -- done: `make -C llama.cpp.zmake lib
   VULKAN=1` (Decision 55). Checked here with a stand-in `glslc` that writes
   stub SPIR-V: every probe, both generator passes, all 1,980 variants
   defined in the archive, `ggml_backend_vk_reg` registered, native and
   cross-built for `aarch64-linux-gnu`; without `-Dvulkan` the archive is
   byte-identical to before. Real shaders need the GB10.
3. ~~**On the GB10**~~ -- done: the toolchain pinned (Decision 59), all
   seven probes pass, 1,980 variants, and `make vk-repro` passes on all
   four of the GB10's arms -- coopmat2, KHR coopmat, none, none without
   integer dot. `NOTES.md`, "Stage 7: the GB10 pickup".
4. **The gates, then the host port in `src/ggml/vulkan/`, outside in** --
   under way (`NOTES.md`, "Stage 7 step 4"):
   - ~~the build~~: `-Dvulkan` here, reference and port modes, `vk-ref`,
     the shader table from the reference's own generated files.
   - ~~`vk-node-diff`~~ written, negative-tested; ~~`vk-backend-diff`~~
     written, negative-tested 3/3.
   - ~~the registry and the device's descriptive answers~~ -- identical in
     13 environments.
   - ~~the logical device, the pipeline walk, the buffers, the backend~~
     (`NOTES.md`, "Stage 7 step 5").
   - ~~`vk-ops-diff`~~ (`harness/cuda_ops_diff.c`'s method), ~~every op
     family~~, ~~the compute loop's fusion~~, ~~`graph_optimize`~~ --
     every supported case bit-identical, and `vk-node-diff` identical on
     all four arms (`NOTES.md`, "Stage 7 steps 6 and 7").
   - ~~`vk-parity-cli`~~ -- our `llama-cli` against `raw_completion.c` over
     upstream's libllama and Vulkan host layer, 6/6 prompts identical,
     negative-tested 3/3 (`NOTES.md`, "Stage 7 step 8").
5. ~~The swap~~ -- done: `-Dvulkan-port` defaults to true, so `-Dvulkan`
   builds the Zig host layer; `-Dvulkan-port=false` builds upstream's, for
   the gates. `supports_op` is the C++'s predicate alone, and `vk-ops-diff`
   fails on a disagreement either way.

### The UNO Q's Adreno 702

**Done** (`NOTES.md`, "Vulkan on the UNO Q's Adreno 702"). A fifth arm --
fp16, no integer dot, no matrix cores, 16 KiB shared memory, unified
memory -- under Mesa's Turnip, cross-built from this Mac with the pinned
toolchain (`scripts/vk-toolchain` builds on macOS now) against the board's
loader, `aarch64-linux-gnu.2.41`. Parity 7 / 7 on plain-d6; `vk-node-diff`
identical per node on plain-d6 and Qwen3-0.6B and fused on plain-d6, the
reference reproducible against itself; negative-tested 2/2. Qwen3's fused
graph at the default submit size trips the kernel's GPU hang check on both
sides; at `GGML_VK_MAX_NODES_PER_SUBMIT` 16 or 4 it is identical too.

### Vulkan pickup (GB10)

~~Done~~ (`NOTES.md`, "Stage 7: the GB10 pickup"). The toolchain is
`scripts/vk-toolchain` at SDK 1.4.363 (Decision 59); the GB10 reports
`VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2`,
`VK_KHR_shader_integer_dot_product` and `VK_KHR_shader_bfloat16`; all seven
probes pass; `make vk-repro` passes per node and fused on every arm. One
finding changes the method: `glslc -O` is not reproducible on `tri_f16`, so
the port embeds the reference's generated SPIR-V rather than regenerating
it.

## Upstream syncs

`SYNC.md` is the procedure, and `UPSTREAM.md` is the record. A sync's Mac
gates block it. The other machines' runs go on the lists below, and they
must clear before the **next** sync starts (MAINTENANCE.md, Decision 7).

### Pickup for the v0.4.0 sync

**GB10 (`spark-dbff`, `~/src/llama.zig`)**

Items 1-4 **done** (`NOTES.md`, "The v0.4.0 GB10 pickup: CUDA and
Vulkan"): the library port unchanged, every gate passing after harness
fixes -- `ggml_cast`'s self-reference in both ops harnesses, which had
crashed `cuda-ops-diff` and kept the fused TOPK_QSA from ever being
compared, and libgcc ahead of `compiler_rt` for `node-diff`'s reference.
Upstream's tools built and ran on Linux for the first time, after two
fixes in the port's tools and CUDA link paths; `tools-diff` passes all
twelve, the `$PLAIN` cases included. What stays open is below the list.

1. **CUDA:** a real `zig build -Dcuda`, which rebuilds the fatbins and the
   kernel table. It must resolve every new key:
   - `moe_weighted_reduction_f32` and `swiglu_clamp_kernel<float|__half>`;
   - `flash_attn_mask_to_sparse_indices`, and `flash_attn_ext_f16<…>` with
     `use_sparse` as the 7th bool;
   - `mul_mat_vec_q_moe<T, 2, has_fusion>` and `mm_ids_helper<10>`.

   Each new kernel's first launch also checks its argument sizes, including
   the 48-byte fusion struct passed by value.
2. **CUDA gates.**
   - `cuda-ops-diff`: covers `test_swiglu_clamp`, `test_moe_weighted_reduction`,
     and sparse flash attention, which needs KV ≥ 4096 at head size 512 or 576.
   - `cuda-node-diff`, per node and fused: the only gate that drives
     `graph_optimize`'s `add_alloc_dep`.
   - `cuda-backend-diff` and `cuda-parity-cli`.
   - Re-run `gen-mmq-config` and `gen-fattn-config` there; the output must
     not change.
3. **Vulkan:** rebuild both sides at SDK 1.4.363 (the toolchain didn't move),
   then on all four arms:
   - `vk-push-layout` and `vk-repro`.
   - `vk-ops-diff`: watch cross-entropy loss and its backward pass (including
     the wg512 variant above 1024 classes), `LIGHTNING_INDEXER` for every K
     type, `SWIGLU_CLAMP`, radix `TOP_K` at large K, and `MUL_MAT_ID` row-id
     hoisting and coopmat2 K padding.
   - `vk-node-diff`, per node and fused. The new `is_src_of` changes
     `graph_optimize`'s node order.
   - `vk-backend-diff`, whose harness now passes `graph_optimize` its
     params, and `vk-parity-cli`.
   - TOPK_QSA fusion: ~~covered only by a fused `vk-node-diff` on the
     `qwen4exp` fixture~~. The fixture cannot reach it: its
     `indexer_top_k` of 8 keeps the width under the radix regime's 1,024.
     Only `vk-ops-diff`'s wide `TOPK_QSA` cases do -- and until the
     harness fix they were never compared, the reference never fusing.
     `vk-node-diff` on the fixture, through `node_dump`'s new `tokens:`
     prompt, is identical on all four arms.
4. **Linux arm64 CPU set:** `ops-diff`, `repack-diff`, `node-diff`,
   `parity-port --cpu`, `host-arms-diff` and `quant-diff`. Quantize now maps
   the file on Linux.

Still open on the GB10:

- **`lint`**: no `zlint` here that takes the styleguide's `-c`.

**Mac, after the GB10 pickup** -- **done** (`NOTES.md`, "The v0.4.0
Mac re-checks"): `make validate` clean, and `tools-diff` identical for all
twelve tools, `export-lora` and `imatrix` included.

- `tools-diff export-lora` and `tools-diff imatrix`: their setups link
  through `platform-libs` now, which carries no `-lobjc` where the old
  Mac line had one.
- `tools-diff` for every tool: the free-memory masks and
  `GGML_NO_BACKTRACE` are now set for every case.

**DESKTOP-BR7J5S9 (x86_64_v3)**

Items 5 and 6 **done**, and `win-host-diffs` from 7, run from the GB10
(`NOTES.md`, "DESKTOP-BR7J5S9: the x86_64_v3 items, from the GB10").
`x86-node-diff` passes on Qwen3.5-2B `Q4_K_M` and `IQ4_XS` alike. **Open: `quant-diff` on Windows**, which has no Windows mode --
its harness and its model and vocab inputs are host-only.

5. `make x86-gates`. Only `x86-ops-diff` reaches the IQ panel GEMM
   (`iqp.cpp`, live only under AVX2); look at its `iqp …` lines. Also run
   `x86-node-diff` on `Qwen3.5-2B-IQ4_XS.gguf` if the model is on the
   laptop.
6. **Reachability:** one port-side run with `GGML_NO_IQ_PANEL` set must make
   those lines differ, proving the panel ran rather than `vec_dot` on both
   sides. This needs a `_putenv_s` in the wrapper.
7. `quant-diff`, which now maps the file on Windows, and `win-host-diffs`.

**JWHLPTOP (AVX-512)** -- **open**: the agent did not connect on
2026-10-28 (nor did the GB10's or DESKTOP's, after their items ran). Marc
directed v0.4.1 to start over this one open pickup; it carries into the
v0.4.1 pickup list.

8. `X86_CPU=x86_64_v4 X86_AGENT=JWHLPTOP make x86-gates`.
9. `X86_CPU=tigerlake X86_AGENT=JWHLPTOP make x86-gates`: the GFNI,
   AVX-512 VNNI and `GGML_IQP_USE_BIAS` arms.

No machine has the AVX-VNNI-only or AVX-VNNI-INT8 arms of `iqp`. Both are
compile-checked only.

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
| All architectures | **Done.** Every entry of `llama_model_mapping`, 146, under `src/llama/models/` (~44,000 lines with the six new memory types); see "What happens next" |
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

   Its end-to-end gate, owed while this machine could not link C++
   (`NOTES.md`, "Blocked: every link-time gate"), was paid when the link
   came back: every `make port`, `parity-cli` and `node-diff` run loads its
   model through `gguf.zig`.
4. ~~`ggml-backend.cpp`~~ — **done.** `src/ggml/backend.zig` (the vtable
   dispatch) and `src/ggml/backend_sched.zig` (the scheduler), 102 symbols —
   not the 82 recorded here before. Every gate green with it in the path:
   `make port`, `parity-cli` 6/6, `graph-diff` 131 nodes, `backend-ops`
   21,093 configurations, `probe`. See `NOTES.md`.

   Worth recording for the files still to come: it contains **no `throw`, no
   `catch`, no `std::string`, no templates and no virtual functions** — the
   difficulty was the five-pass assignment algorithm, not the C++.
5. ~~`ggml-backend-meta.cpp` and `ggml-opt.cpp`~~ — **done** (`backend_meta.zig`,
   `opt.zig`; `meta-diff`, `opt-diff`). They were deferred to the end of
   the ggml half by Decision 40, and the record below is what was expected then. Measured counts: meta is **8** symbols over
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
   shuffle divergence, do not chase it. **Superseded in the doing**: the
   reference is libc++, so `opt.zig` reproduces libc++'s `shuffle` and
   `uniform_int_distribution` exactly, measured against the real templates
   by `opt-diff`; the divergence is libstdc++'s, and only off this platform.

**Every symbol count this plan carried for Stage 4's first group was low** —
gguf 44 vs 61, backend 82 vs 102, meta 4 vs 8, opt 9 vs 37. Measure the
contract with `port-coverage` before estimating; the original survey figures
are not usable as estimates.
6. ~~**`ggml.cpp`**~~ (26 lines, 0 symbols) — **dropped**, now that no C++ in
   the build can throw. See the note under "Order of work" above.
7. ~~`ggml-cpu/binary-ops.cpp`, `unary-ops.cpp`, `vec.cpp`, `ops.cpp`~~ —
   **done.** 125 symbols between them. `ops.cpp` is the largest translation
   unit in ggml; see "`ops.cpp`: done" above.
8. ~~`llamafile/sgemm.cpp`~~ — **done.** `src/ggml/cpu/ops/sgemm.zig`, one
   symbol, 391 live lines of 4,164. The claim that its gate already existed
   was **three-quarters true**: the f16 kernel needs `n >= 8` and the harness
   topped out at 7, so one of the four live instantiations was uncovered.
   Two cases closed it and a probe proved all four are now reached. See
   `NOTES.md`.
**`GGML_USE_CPU_REPACK` is now defined on both sides.** It was not, and
upstream's CMake defaults it on, so `repack.cpp` and `arch/arm/repack.cpp`
were compiling into the library unreachable — 5,692 live lines of dead code.
Measured, and confirmed by execution. That is what makes the cluster below
real work rather than deletable; see `NOTES.md`.

9. ~~**The vtable cluster**~~ — **done.** `traits.cpp`, `repack.cpp`,
   `arch/arm/repack.cpp` and `ggml-cpu.cpp`, swapped in one step, 6,180 live
   lines, 73 unmangled symbols. **The unit is forced by five symbols, not by
   the 261 mangled exports the four objects carry** — the cluster's
   *external* contract is a pure C ABI like the other twenty. What the
   external contract did *not* cover was `repack.cpp`'s own C++-linkage
   dispatch; see "The vtable cluster: done, and what it cost" above, and
   `make repack-diff`, the gate it needed.

   The five are `ggml::cpu::tensor_traits::~tensor_traits()`,
   `ggml::cpu::extra_buffer_type::~extra_buffer_type()`, the typeinfo for
   both, and `ggml_backend_cpu_get_extra_buffer_types()`, which returns a
   `std::vector<ggml_backend_buffer_type_t> &` (ggml-cpu.cpp:42). So:
   `traits.cpp` declares two abstract bases, `repack.cpp` derives from them,
   `ggml-cpu.cpp` hands the instances out. Bases become a Zig vtable struct,
   derived types become structs holding a pointer to one, the vector becomes
   a static array — the move `backend_reg.zig` already made.

   **`repack.cpp` and `arch/arm/repack.cpp` are mutually dependent across 35
   plain C symbols** (`ggml_gemm_*`/`ggml_gemv_*` one way, the `_generic`
   fallbacks the other) and would move together with or without the C++.

   `repack` reorders `Q4_0`/`Q4_K`/`IQ4_NL` weights into interleaved blocks,
   so `node-diff` on a model with those types was expected to be its gate,
   alongside `ops-diff`. **"Check that gate arm by arm before porting" was
   the right instruction and it was not followed.** `node-diff` reaches the
   kernels but reports one node and stops; `ops-diff` never allocates a
   `CPU_REPACK` buffer and cannot reach them at all; and half the kernels
   cannot be selected on this CPU. `make repack-diff` is what the group
   actually needed, and it found four faults where `node-diff` found one.

**`ggml-cpu/` is empty.** What still compiles from C++ there — `hbm.cpp`,
`amx/amx.cpp`, `amx/mmq.cpp` — is zero live lines on this target. Next the
Metal host layer, then libllama.

**Stage 0 is closed.** `llama.cpp/` is a submodule pinned at `c1d0e7a00`;
this plan and `CLAUDE.md` both claimed otherwise long after it was done, which
is worth remembering about status lines in these files — check the tree, not
the prose.

**The reference-pin question is settled, and not the way it was posed.**
`llama.cpp.zmake` vendors no sources — it is the build system, and the
llama.cpp it compiles is an untracked checkout inside it. Submoduling it
would pin the `build.zig` and leave the sources unpinned, which is the half
that matters. `scripts/check-reference-pin` asserts instead that the checkout
is at our submodule's commit, and the three bit-exact gates source it.
Decision 41.
Repointing `scripts/parity-port` at `llama.cpp.zmake` is done.

**The `ops.cpp` tidy-up is done.** Seven workers writing in parallel each
grew their own small helpers; 21 duplicate bodies across 18 files are now one
definition each in `ops/common.zig`, with a one-line alias where a file used
a different local spelling:

| helper | copies | |
|---|---|---|
| byte offset `i*nb` | 6 | named `at` in three files and `off` in three |
| typed pointer at a byte offset | 5 | three signatures, differing only in the offset's integer type and `[*]T` vs `*T` |
| stride widened to `i64` | 4 | |
| `std::min` / `std::max` | 4 | |
| `ggml_barrier(params->threadpool)` | 2 | |
| the NEON `GGML_F32x4_REDUCE` | 2 | named in `recurrent.zig`, open-coded in `ssm.zig` |

**The call sites were left alone.** Folding the bodies removes the thing that
can drift; rewriting ~240 call sites to a new spelling would have risked a
transcription error for no gain, and each file still reads against the C.

Two things the survey turned up that the note above had wrong:

- **`zeroDst` is not a duplicate.** `conv.zig`'s memsets the whole tensor;
  `linalg.zig`'s fills `f32` zeros on thread 0 and then barriers. Merging
  them on the strength of the shared name would have been a real bug. The
  latter is now `zeroDstAndSync`, so the difference is visible.
- **Three files imported `std` without using it.** Pre-existing, unrelated to
  the parallel writing; dropped.

Gate: `make ops-diff`, 592 op results identical either side of the fold.

10. ~~`llama-sampler.cpp` + `llama-grammar.cpp`~~ — **done.** The first of
    libllama, and the one piece that swaps ahead of the rest (Decision 42).
    40 symbols, 38 C and two mangled. `make sampler-diff` 40,289 records,
    `make libcxx-diff` 189,684 checks, `make regex-diff` 538,416. The
    vocabulary, the architecture tables and the model's metadata have since
    followed -- see "Order of work" step 4 for where libllama stands.

11. ~~The rest of libllama, and the swap at `llama.h`~~ — **done**. Stage 4
    is complete: no C or C++ compiles into `libggml.a` or `libllama.a`
    apart from the two Obj-C files. See `NOTES.md`, "libllama swapped".

**Stage 6, CUDA, is done and swapped** on the GB10 machine (Decision 43;
"Stage 6 — CUDA"; `NOTES.md`, "Stage 6"). Mac-only follow-ups from the CUDA
work are collected under "Mac pickup", done. The xcframework no longer
links Zig's libc++ into each slice -- nothing needed it since the swap, and
the prelinked object is byte-identical without it -- and rebuilding it
found the iOS slices broken by a `std.debug.print` in library code, now
fixed and guarded by `make validate` (`NOTES.md`, "The xcframework after
the swap").

12. ~~**The other architectures**~~ (Decision 36's first follow-on) --
    **done.** All 146 entries of `llama_model_mapping`, one file per
    upstream class, with the memory types they needed: the iSWA KV cache,
    the hybrid iSWA memory, the MSA, DSA, DSA-iSWA and DSv4 caches, and the
    graph inputs for no-cache attention, MLA's K-only cache, T5's
    cross-attention and the hybrid K-only memory. Gated by Decision 49:
    `make arch-model-diff` over 159 fixtures, and `make
    gemma4-assistant-diff`. What the fixtures cannot reach -- MTP heads,
    paths behind keys the recipe never sets -- is listed per architecture in
    `NOTES.md`, "The other architectures".

**Next:** Stage 5, cross-platform -- measured; see "Stage 5 — Cross-platform"
for where each target stands and the order of work. Step 1 found that
the port was not bit-exact on Linux arm64's baseline CPU; step 2 made it so;
steps 3 and 4 ran on the GB10 in its pickup, which found and fixed two
Linux-only bugs; step 5 put the SIMD layer in. Step 6, x86_64, is done at AVX2 and exact on the Windows laptop,
and at AVX-512 (`x86_64_v4`, `tigerlake`) on a Tiger Lake.

**Beside it, Stage 7, Vulkan** (Decisions 54–56, 59; "Stage 7 — Vulkan"):
measured, the reference built on the GB10 with a pinned toolchain, and
exact -- `make vk-repro` passes on all four arms. Step 4 is under way: this
repository builds Vulkan, `vk-node-diff` and `vk-backend-diff` exist, and
the whole host layer is ported -- every op, the fusions, `graph_optimize`
-- with `vk-ops-diff` bit-identical, `vk-node-diff` identical per node and
fused on all four arms, and `vk-parity-cli` 6/6. **The swap is done**:
`-Dvulkan` builds the Zig host layer. **`vk-ops-diff` is bit-identical on
all four arms** wherever the reference can compute: the no-coopmat arms in
full, and the KHR arm's FLASH_ATTN_EXT cases past the point where the
reference stalls against itself, which the gate now reaches by running
that family per head size and finishing a stalled part one case per
process. Two KHR cases hang in the reference alone and are reported, not
compared (`NOTES.md`, "The Vulkan arms, completed").

**Two gaps in the chat-template work**, carried from `NOTES.md`:

- ~~**A `chat-parity` gate.**~~ **Done, and passing** against zigjinja
  v3.0.0 (`de36d7b`): `make jinja-diff`, 68 upstream templates × 7
  conversations against upstream's engine, text, raises and input marks
  identical, plus the clock under two zones (Decision 58). It started at 26 of 68 (`NOTES.md`, "`jinja-diff`"); the
  zigjinja fixes and the provenance scheme that closed it are in `NOTES.md`,
  "zigjinja against `jinja-diff`".
- ~~**No gate covers the conversation loop.**~~ **Done**: `make cnv-diff`,
  our `llama-cli -cnv` against a reference conversation over upstream's
  Jinja engine and libllama, on a hybrid and a plain-KV model, with the
  cache kept each turn compared through `LLAMAZIG_CNV_TRACE`. 13/13,
  negative-tested 7/7 (`NOTES.md`, "`cnv-diff`").

**One gap the UNO Q found**:

- ~~**The CLI did not fit to memory.**~~ **Done**: upstream's `llama-cli`
  shrinks an unset context, and on GPUs moves layers and tensors to system
  memory, before it loads (`common_fit_params`); ours asked a 3.6 GB board
  for a 4.6 GB KV cache and was OOM-killed. `cli/fit.zig` ports
  `common/fit.cpp` whole, calling libllama's `llama-ext.h` functions as Zig
  (the CLI now compiles libllama's module into itself); `-fit`, `-fitt`,
  `-fitc`. Gated by `make fit-diff` -- 68 cases on fake GPUs with chosen
  memory, the CPU alone and Metal, every arm reached, 12/12 injections --
  and end to end by three `cnv-diff` scenarios, 16/16. On the board it
  picks upstream's 18,176 tokens, once the CLI took upstream's output
  limits (`NOTES.md`, "The UNO Q, and `--fit`").

## What is left

Everything above is done on the machines this project has. The last item
done was **Vulkan on the UNO Q's Adreno 702**: the pinned toolchain now
builds on this Mac, the Vulkan CLI and reference cross-build against the
board's own loader, and on the board parity is 7 / 7 and `vk-node-diff`
identical per node and fused (`NOTES.md`, "Vulkan on the UNO Q's Adreno
702"). And **iOS on a device**: `make -C xcfwk smoke-device` (now `xcframework/`) runs the
framework on an iPhone 15 and is byte-identical to the port on this Mac,
six prompts on the CPU and six on Metal (`NOTES.md`, "llamazig on an
iPhone") -- which also found the port's footprint growing ~7 MB with every
model loaded and freed, now flat: `token_to_id`'s storage and `text_buf`
are sized before they are filled (`NOTES.md`, "The reload footprint"). And
**ggml's BLAS backend on Apple, over Accelerate** (Decision 60), with zDNN,
ZenDNN and every other BLAS left out as vendor libraries. And **AVX-512**
(Decision 53): `x86_64_v4` and `tigerlake` exact on `JWHLPTOP`, a second
Windows machine (`NOTES.md`, "Stage 5 step 6: AVX-512"). And **libmtmd with
`llama-tts` and `llama-mtmd-cli`** (Decision 62): every projector type ported and
gated on fixtures (`NOTES.md`, "Stage 8: libmtmd and `llama-tts`"). And
**`llama.xcframework`** (Decision 63), a drop-in for upstream's
`build-xcframework.sh` output (`NOTES.md`, "llama.xcframework"). What remains,
and what each one is waiting for:

| Item | Waiting for | State |
|---|---|---|
| **AVX512-BF16 and AVX512-FP16 arms** | a Sapphire Rapids or newer CPU | a compile error until then; Tiger Lake has neither |
| **Linux x86_64's OS arms** | Linux x86 hardware | its kernels are Windows', gated there; only the `__linux__` threading, affinity and NUMA arms are unrun on x86 |
| **`set_numa_thread_affinity`** | a multi-socket machine | every machine here has one NUMA node |
| **Decision 24, idiomatic Zig** | a decision to schedule it | not started; open for every subsystem since the swap |
| **The xcframework's x86_64 slices** (Decision 63) | an Intel Mac, or Rosetta installed here | built, linked and held to upstream's by `compat`; never run. The x86 kernels are gated on Windows; the Darwin x86 OS arms are not |
| **The xcframework on tvOS and visionOS devices** | running `smoke-device`'s app there | both slices built and gated structurally, and their CPUs (`apple_a7`, `apple_a12`) by the root gates; their simulators pass `smoke-sim` |

---

## Questions

Stage 5's three x86 questions are Decisions 51–53.

Earlier questions are settled in the decisions table. The question Stage 4
opened with — how much C++ linkage actually crosses a translation-unit
boundary — has been measured rather than guessed, and the answer is in "The
measurement that settles the method" above.

New questions get appended here as they arise. Answered ones move into the
decisions table rather than accumulating, so this section stays a list of what
is genuinely undecided.
