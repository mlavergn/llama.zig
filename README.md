# llamazig

A port of [llama.cpp](https://github.com/ggerganov/llama.cpp) to pure Zig, for cross-platform portability and to leverage the Zig compiler.

## Goal

Port llama.cpp to pure Zig. This is a big project.

We are not recreating the llama.cpp repo — only the binary. We are porting the `.c` and `.cpp` files.

The primary goal is for llama.cpp, which becomes llamazig, to build for **macOS**. Other platforms come later.

## Stages

The port proceeds in five stages. Each one has to land before the next begins.

### 1. Build with the Zig compiler, driven by the existing Make and CMake

Keep llama.cpp's build system, but have it invoke `zig cc` / `zig c++` instead of the platform toolchain. Target this configuration:

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

### 2. Migrate the build system to Zig

Once we can build using only the Zig compiler, replace CMake with `build.zig`.

llama.cpp uses CMake files that need porting to the Zig build system. `build.zig` should replace CMake **completely**. The `-D` options above become the hardcoded macOS defaults rather than flags a caller has to pass.

### 3. Convert the C files to Zig

This should be fairly straightforward, since C maps cleanly to Zig.

### 4. Convert the C++ files to Zig

This will be challenging. The reliance on the STL and/or Boost is unknown. C++ does not map cleanly to Zig.

### 5. Validate across platforms

Target macOS, iOS, Linux (arm64), and Linux (x86_64). This is the end goal.

## Getting started

Requires **Zig 0.16.0**.

```sh
make build      # zig build         -> zig-out/bin/llamazig, zig-out/lib/libllamazig.a
make dist       # zig build --release=fast
zig build test  # run the unit tests
zig build docs  # generate docs into zig-out/docs/
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

Stage 0 — the Zig scaffold builds and its tests pass. No llama.cpp code has been ported yet.

The reference implementation is pinned at llama.cpp **v0.3.0**.

## Repo layout

| Path | What it is |
|---|---|
| `src/` | The llamazig library. `module.zig` is the barrel; `root.zig` exists only so autodoc has a root. |
| `cli/` | The command-line executable. Thin — the behavior lives in `client.zig` so tests can reach it. |
| `build.zig` | The build graph: `lib`, `cli`, `run`, `test`, `docs`. Will grow to replace llama.cpp's CMake in Stage 2. |
| `llama.cpp/` | The upstream reference clone at v0.3.0. Source material for the port, not our code. |

## See also

- [`PLAN.md`](PLAN.md) — detailed implementation plan, scope measurements, risks, and open questions.
- [`CLAUDE.md`](CLAUDE.md) — build state, toolchain notes, and code conventions.
