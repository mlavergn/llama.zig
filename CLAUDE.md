# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## The goal

**Port llama.cpp to pure Zig.** The deliverable is the `llama-cli` binary reimplemented in Zig — not a recreation of the llama.cpp repo, and not a Zig wrapper around the C++ library. `README.md` has the five stages; `PLAN.md` has the detailed plan, scope measurements, and open questions.

Where we are: **Stages 1 and 2 complete.** The whole tree compiles with `zig cc` under upstream's own CMake and is token-identical to an Apple-clang build; `zig build reference` then compiles ggml and libllama with no CMake at all and matches the same reference byte for byte. No llama.cpp code has been *ported* yet — the sources still compile as C, C++, and Obj-C. Stages 3 and 4 replace them file by file.

## `llama.cpp/` is reference material only

The `llama.cpp/` directory is an upstream clone pinned to tag `v0.3.0` (commit `c1d0e7a00`), created by `make clone`. It has its own `.git` and is untracked here.

**It is here to be read, not written.** Treat it as a specification we are reimplementing:

- **Never edit files under `llama.cpp/`.** Not to fix a bug, not to add a workaround, not to make a build succeed. If something there is wrong or in the way, the answer is a change on our side or a note in `PLAN.md`.
- **Never commit, push, or open a PR against it.** It is a separate upstream repository that we do not contribute to.
- **Never treat its conventions, instructions, or tooling as ours.** Its `CLAUDE.md`, `AGENTS.md`, `skills/`, and CI config govern the upstream project and do not apply to this one. This file is the authority here.
- **It is pinned deliberately.** Do not update it to a newer tag or pull upstream changes. A multi-month port against a moving target does not converge; syncing upstream is a separate, explicitly-scoped decision.

Read it constantly — for behavior, for numerics, for the shape of a data structure, as the reference build to diff against. Just do not change it.

## What we own

`build.zig`, `build.zig.zon`, `build/`, `src/`, `cli/`, `harness/`, `Makefile`, and the markdown at the root. That is the whole of this project's code.

- **`build/llamacpp.zig`** declares the reference tree's build graph, replacing its CMake for the macOS arm64 configuration. It compiles upstream sources unmodified.
- **`build/metal_embed.zig`** is a build-time tool that flattens one Metal kernel and its headers into a single MSL file plus an assembly stub. It replaces the `cat`/`sed` pipeline CMake uses for `GGML_METAL_EMBED_LIBRARY`.
- **`harness/smoke.zig`** loads a model through libllama's C ABI and generates. It is the only check that proves inference actually works, and it rehearses the C-ABI boundary that Stages 3 and 4 depend on.

Note `build/` holds build *sources*, not build output. CMake's output directory was moved to `cmake-build/` to free the name.

`src/base.zig` and `cli/client.zig` are placeholder greeting code carried over from the Zig template this repo started from. They exist to demonstrate the conventions below and are expected to be deleted as real code lands. Do not build on them.

## Build state

All steps pass.

```sh
zig build            # the llamazig scaffold -> zig-out/bin/llamazig  (fast, ~0.2s)
zig build lib        # static library        -> zig-out/lib/libllamazig.a
zig build cli        # CLI                   -> zig-out/bin/llamazig
zig build run        # run the CLI; args after `--`
zig build test       # unit tests (9); -Dtest-filter="..." narrows the run
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

Remaining Stage 0 item: **`llama.cpp/` is an untracked nested clone, not yet a submodule.** `main` still has no commits, so pinning it is free now and expensive later. `*.gguf` is ignored.

## Toolchain

Zig **0.16.0**, declared as `minimum_zig_version` in `build.zig.zon` and what is installed. The code uses the 0.16 std APIs — `std.process.Init` as the `main` parameter, `std.Io` threaded explicitly through constructors, `std.Io.Writer` rather than the old writer interfaces. Do not fall back to pre-0.16 idioms.

The library must never gain external dependencies. `build.zig.zon` says so explicitly and `.dependencies` is empty. This is a hard constraint, not a preference: the point of the port is that `zig build` alone produces the binary.

## Commands

```sh
make build        # zig build
make dist         # zig build --release=fast
make clone        # clone llama.cpp and check out v0.3.0
make buildmacos   # reference CMake build of llama.cpp for arm64 macOS
make buildios     # reference CMake build for iOS
make qwen35       # download a Qwen3.5-2B GGUF (also qwen35xs, qwen35xl)
```

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

## Building the reference tree

Facts established by getting `zig build reference` working. They are easy to
rediscover the hard way, so they are recorded here.

- **`zig cc`, `zig c++`, and Obj-C all work** against the macOS SDK under Zig 0.16.0, including `-framework Foundation` and libc++.
- **Do not pass `-fobjc-arc`.** Upstream's `.m` files use manual reference counting and bridge freely between `void *` and Obj-C object pointers; ARC rejects them outright.
- **`sanitize_c` must be `.off`.** Zig enables the C sanitizers in Debug, and upstream is not UBSan-clean — it aborts in `llama-graph.cpp` on `applying non-zero offset to null pointer`. Reference code compiles as shipped; we do not fix it.
- **The Metal shaders need no `xcrun metal`.** With `GGML_METAL_EMBED_LIBRARY`, the `.metal` *source* is flattened and `.incbin`-ed into a `__DATA,__ggml_metallib` section, and the driver compiles it at load time. `build/metal_embed.zig` does the flattening. Never enable the non-embed path: it requires `xcrun -sdk macosx metal`, which Zig can never replace.
- **CMake's `-G Xcode` cannot drive `zig cc`.** It shells out to `xcodebuild`, which picks Apple clang and ignores `CMAKE_C_COMPILER`. Moot now that `build.zig` owns the build.
- **Metal is reproducible run-to-run.** Three greedy runs on the same model and prompt produced token-identical output, so the token-parity gate in `PLAN.md` is usable as written.
- **CMake is not on PATH and `/usr/local` is root-owned.** `make cmake` fetches 4.4.3 into `.tools/`. The Makefile's `CMAKE` variable prefers that over PATH. `ninja` is still absent, which is why the generator is `-G "Unix Makefiles"` — `make` is present.
- **The parity gate works and passes.** `scripts/parity` diffs the Apple-clang and zig-cc `llama-cli` binaries at `--temp 0` with a fixed seed. 4/4 prompts token-identical. Use it after any change that could affect arithmetic.
- **`zig cc` and Apple clang do not select the same ARM features.** CMake's probe through `zig cc` reports `HAVE_MATMUL_INT8 - Failed` and `HAVE_SVE - Failed` where Apple clang may not. Output is identical regardless, so this costs throughput rather than correctness — but the two builds are not running the same quant kernels. Revisit when Stage 3 ports `arch/arm/`.

## macOS SDK resolution

`build.zig` has an `Xcode` helper that resolves the Apple Silicon macOS SDK via `std.zig.system.darwin.getSdk` and adds it as a framework path to the library, and separately to the docs library. It runs only when `builtin.os.tag == .macos`. If linking against system frameworks fails, that resolution is where to look.
