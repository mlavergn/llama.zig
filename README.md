# llamazig

A port of [llama.cpp](https://github.com/ggerganov/llama.cpp) to pure Zig, for cross-platform portability and to leverage the Zig compiler.

## Goal

Port llama.cpp to pure Zig. This is a big project.

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

### 3. Convert the C files to Zig

This should be fairly straightforward, since C maps cleanly to Zig.

### 4. Convert the C++ files to Zig

This will be challenging. The reliance on the STL and/or Boost is unknown. C++ does not map cleanly to Zig.

### 5. Validate across platforms

Target macOS, iOS, Linux (arm64), and Linux (x86_64). This is the end goal.

## Getting started

Requires **Zig 0.16.0**.

```sh
make build       # zig build         -> zig-out/bin/llamazig
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

**Stages 1 and 2 complete.**

- The whole llama.cpp tree compiles with `zig cc` under its own CMake, and the resulting `llama-cli` is token-identical to an Apple-clang build (`make parity`, 4/4 prompts).
- `zig build reference` builds ggml and libllama with no CMake at all, and matches the Apple-clang libraries byte for byte through the same driver.

No llama.cpp code has been *ported* yet — the sources still compile as C, C++, and Obj-C. Stages 3 and 4 replace them file by file.

The reference implementation is pinned at llama.cpp **v0.3.0**.

## Repo layout

| Path | What it is |
|---|---|
| `src/` | The llamazig library. `module.zig` is the barrel; `root.zig` exists only so autodoc has a root. |
| `cli/` | The command-line executable. Thin — the behavior lives in `client.zig` so tests can reach it. |
| `build.zig` | The build graph: `lib`, `cli`, `run`, `test`, `docs`, `reference`, `smoke`. |
| `build/` | Build sources, not build output. `llamacpp.zig` replaces llama.cpp's CMake; `metal_embed.zig` flattens Metal shaders for embedding. |
| `harness/` | `smoke.zig` — loads a model through libllama's C ABI and generates, proving the reference build actually infers. |
| `scripts/` | `zigcc` / `zigcxx` wrappers so CMake can drive the Zig toolchain, and `parity` to diff two builds' token streams. |
| `.tools/` | Locally installed CMake. Not tracked; `make cmake` fetches it. |
| `llama.cpp/` | The upstream reference clone at v0.3.0. Source material for the port, not our code. |

## See also

- [`PLAN.md`](PLAN.md) — detailed implementation plan, scope measurements, risks, and open questions.
- [`CLAUDE.md`](CLAUDE.md) — build state, toolchain notes, and code conventions.
