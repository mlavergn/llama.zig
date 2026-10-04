# llamazig

A port of [llama.cpp](https://github.com/ggerganov/llama.cpp) to Zig, for cross-platform portability and to leverage the Zig compiler.

| | |
|---|---|
| [`SPEC.md`](SPEC.md) | **What** it is and does — artifacts, flags, behaviour, conformance |
| `README.md` | **Why**, and where the port has got to |
| [`PLAN.md`](PLAN.md) | **How** — decisions, scope measurements, what comes next |
| [`NOTES.md`](NOTES.md) | **Why** — the record of each completed step and what each gate caught |

## Goal

Port llama.cpp to Zig — pure Zig apart from the Metal Objective-C host layer.
This is a big project.

The exception is deliberate. `ggml-metal-device.m` and `ggml-metal-context.m`
are 3,091 lines of Objective-C driving the Metal API, and Zig has no
Objective-C frontend; reaching that API from Zig would mean hand-writing every
call against the untyped `objc_msgSend` runtime. Those two files stay
Objective-C, compiled by `zig cc`. Everything else is ported.

We are not recreating the llama.cpp repo — only the binary. We are porting the `.c` and `.cpp` files.

The primary goal is for llama.cpp, which becomes llamazig, to build for **macOS**. Other platforms come later.

## Stages

The port proceeds in five stages. Each one has to land before the next begins.

### 1. Build with the Zig compiler, driven by the existing Make and CMake — *done*

llama.cpp's own build system, with `zig cc` / `zig c++` in place of the platform toolchain. The whole tree builds, `llama-cli` included, and its output is token-identical to an Apple-clang build of the same sources.

```sh
make cmake           # fetch CMake into .tools/ (not installed on this machine)
make buildmacos      # control build, Apple clang  -> cmake-build/apple/
make buildmacos-zig  # same sources via zig cc     -> cmake-build/zig/
make parity          # diff the two, greedy, fixed seed
```

The generator is `-G "Unix Makefiles"`, not `-G Xcode`: Xcode's generator shells out to `xcodebuild`, which selects Apple clang and ignores `CMAKE_C_COMPILER`. It also avoids needing ninja. Target configuration:

```sh
cmake ../llama.cpp \
  -G Xcode \
  -DCMAKE_SYSTEM_NAME=Darwin \
  -DCMAKE_OSX_SYSROOT=macosx \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_ACCELERATE=ON \
  -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_NATIVE=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_CURL=OFF \
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO
```

### 2. Migrate the build system to Zig — *done for ggml and libllama*

`build.zig` compiles the vendored tree with the Zig toolchain and no CMake at all. The `-D` options above are hardcoded as the macOS defaults. The sources are compiled exactly as upstream ships them; nothing under `llama.cpp/` is modified.

```sh
zig build reference   # -> zig-out/lib/{libggml.a, libllama.a}
zig build smoke -- Qwen3.5-2B-Q4_K_M.gguf "The capital of France is"
```

Working: ggml with its CPU and Metal backends, all 20 Metal shader libraries embedded, and libllama with all 151 model architectures. Inference runs with 25/25 layers offloaded to the GPU. Verified against CMake: the same driver linked against `build.zig`'s libraries and against the Apple-clang libraries gives byte-identical output.

Not yet built here: `common/`, `tools/server/`, and the upstream `llama-cli`. Those are outside the port's scope and are available from the Stage 1 build when needed.

### 3. Convert the C files to Zig — *done*

C maps cleanly to Zig, and the shape-and-struct work has been straightforward.
The friction has been elsewhere: `ggml-impl.h` is unparseable by `translate-c`
(it includes `<arm_neon.h>`), `translate-c` names anonymous union members
positionally so block layouts are restated by hand and asserted, and the C
contracts `a*b + c` into a single FMA by default where Zig fuses only where
`@mulAdd` says so — which changes bytes. See `CLAUDE.md`.

`ggml-cpu.c` added a new kind: it is the only ported file that runs *threads*.
Its mutexes and condition variables are pthreads' rather than Zig's, because
Zig 0.16's `std.Io.Mutex` takes an `Io` on every call and a backend entered
through a C ABI has none to pass. The barrier's memory orderings are the C's,
verbatim.

The NEON kernels were expected to be the hard part and were less bad than
feared: only 1,556 of `arch/arm/quants.c`'s 4,319 lines compile on this target
(the rest is SVE and i8mm), and of 82 distinct intrinsics all but one are a
line of Zig `@Vector` arithmetic. The exception, `vdotq_s32`, is integer — so
the portable form is bit-identical, and no throughput change is measurable
above the run-to-run noise.

### 4. Convert the C++ files to Zig

This will be challenging. The reliance on the STL and/or Boost is unknown. C++ does not map cleanly to Zig.

### 5. Validate across platforms

Target macOS, iOS, Linux (arm64), and Linux (x86_64). This is the end goal.

## Getting started

Requires **Zig 0.16.0**.

```sh
make build       # zig build         -> zig-out/bin/llama-cli
make dist        # zig build --release=fast
make reference   # build ggml + libllama from the vendored sources
make smoke       # load a model and generate, to prove it works
zig build test   # run the unit tests (-Dtest-filter="..." narrows)
zig build docs   # generate docs into zig-out/docs/
```

To work with the reference implementation:

```sh
make clone       # clone llama.cpp and check out v0.3.0
make buildmacos  # reference CMake build for arm64 macOS (the Stage 1 flag set above)
make buildios    # reference CMake build for iOS
```

Models for testing:

```sh
make qwen35      # Qwen3.5-2B Q4_K_M    (1.28 GB, standard)
make qwen35xs    # Qwen3.5-2B IQ4_XS    (1.17 GB, smaller)
make qwen35xl    # Qwen3.5-2B UD-Q4_K_XL (1.34 GB, dynamic quant, better quality)
```

## Status

**Stages 1, 2 and 3 complete. Stage 4 is under way: all of
`ggml/src/ggml-cpu/` is Zig now, along with five of the `ggml/src/*.cpp`.**

**No C compiles anywhere under `llama.cpp/ggml/src/` any more.** All six
translation units are Zig — 598 exported symbols, each byte-verified against
the C it replaces:

| File | Lines | State |
|---|---:|---|
| `ggml-alloc.c` | 1,249 | ported, swapped in |
| `ggml.c` | 8,067 | ported, swapped in — 381/381 symbols |
| `ggml-quants.c` | 5,667 | ported, swapped in — 79/79 symbols |
| `ggml-cpu/ggml-cpu.c` | 3,900 | ported, swapped in — 65/65 symbols |
| `ggml-cpu/quants.c` | 1,339 | ported, swapped in — 45/45 symbols |
| `ggml-cpu/arch/arm/quants.c` | 4,319 | ported, swapped in — 28/28 symbols |

Thirteen C++ translation units have followed them, so **nothing under
`ggml/src/ggml-cpu/` compiles from C or C++ any more** — the op kernels
(`ops.cpp`, `vec.cpp`, `binary-ops.cpp`, `unary-ops.cpp`), the llamafile
fast path (`llamafile/sgemm.cpp`), and the vtable cluster that had to move
as one unit (`traits.cpp`, `ggml-cpu.cpp`, `repack.cpp`,
`arch/arm/repack.cpp`). Above ggml-cpu, `ggml-threading.cpp`,
`ggml-backend-reg.cpp`, `gguf.cpp` and `ggml-backend.cpp` are ported too.
What is left is `ggml-backend-meta.cpp`, `ggml-opt.cpp`, the Metal host
layer and all of libllama.

Throughput is unchanged by the port, as far as this machine can tell.
Generation sits at 222-234 t/s on Qwen3.5-2B Q4_K_M for both `make port` and
`make ref` across three runs each; the prompt figure ranges 507-640 and 549-638
respectively, a spread far wider than any difference between them. Read those
as "no measurable change", not as a comparison — and note `make ref` links the
Apple-clang libraries, so a compiler difference is folded in too.

There is also a working CLI. `llama-cli` is our own binary, not upstream's
relinked: it mimics `llama-cli`'s flag surface for the features we support and
refuses the rest rather than ignoring them. One-shot completion only —
interactive conversation is still owed.

**How correctness is checked.** Several gates, none of which subsumes the
others, and each negative-tested by injecting a fault and confirming it fails:

| Gate | What it covers |
|---|---|
| `make graph-diff` | 131 constructor nodes vs the C — shapes, strides, `op_params`, wiring |
| `make ops-diff` | ~300 CPU op cases computed and compared **on bits** against the stock C, at 1 and 3 threads |
| `make node-diff` | every node of a real Qwen3.5 decode, on bits, CPU (default) or Metal |
| `make repack-diff` | all 36 interleaved repack kernels against the reference C++, on bits, in one process |
| `make backend-ops` | 21,093 op configurations, Metal against CPU |
| `make probe` | proves ported code is on the execution path at all — allocator and CPU dispatch, one run each |
| `make parity-port` / `parity-port-cpu` | tokens, ported libraries vs stock, one driver — on Metal, or on the CPU alone |
| `make port` / `make ref` | the CLI binary against the same loop on stock libraries |

Token parity is coarser than it looks — doubling RoPE's `freq_base` passes it —
which is why the bit-level diffs exist alongside. On a Metal machine inference
never reaches the CPU kernels at all, so only `ops-diff`, `node-diff`,
`repack-diff` and the `--cpu` parity runs can see them. Some kernels no gate
but `repack-diff` can reach, because the dispatch cannot select them on this
CPU at all. `NOTES.md` carries the measurements.

The reference implementation is pinned at llama.cpp **v0.3.0**.

## Repo layout

| Path | What it is |
|---|---|
| `src/` | The llamazig library. `module.zig` is the barrel; `root.zig` exists only so autodoc has a root. |
| `cli/` | The command-line executable. Thin — the behavior lives in `client.zig` so tests can reach it. |
| `build.zig` | The build graph: `lib`, `cli`, `run`, `test`, `docs`, `reference`, `smoke`. |
| `build/` | Build sources, not build output. `llamacpp.zig` replaces llama.cpp's CMake; `metal_embed.zig` flattens Metal shaders for embedding. |
| `harness/` | `smoke.zig` — loads a model through libllama's C ABI and generates, proving the reference build actually infers. The other files here are gate drivers. |
| `scripts/` | `zigcc` / `zigcxx` wrappers so CMake can drive the Zig toolchain, and `parity` to diff two builds' token streams. |
| `.tools/` | Locally installed CMake. Not tracked; `make cmake` fetches it. |
| `llama.cpp/` | The upstream reference clone at v0.3.0. Source material for the port, not our code. |
| `SPEC.md` | The specification: what llamazig is and does. |

## Licence

MIT; see [`LICENSE`](LICENSE).

This is a derivative work — the code under `src/` is a translation of
[llama.cpp](https://github.com/ggerganov/llama.cpp), which is MIT licensed by
the ggml authors. [`NOTICE`](NOTICE) carries their copyright and names the
pinned commit each file was translated from.

## See also

- [`PLAN.md`](PLAN.md) — the forward plan: decisions, scope measurements, risks, and what comes next.
- [`NOTES.md`](NOTES.md) — the record: how each completed step was done, and the false passes found by injection.
- [`CLAUDE.md`](CLAUDE.md) — build state, toolchain notes, and code conventions.
