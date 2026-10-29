//! The upstream release the reference tree is pinned to, and what CMake
//! configures its version files with there.
//!
//! **Not a port.** Build description: the one place `build/` reads the pin
//! from. A sync moves it with the `llama.cpp/` submodule, and
//! `scripts/check-reference-pin` holds it, `llama.cpp.zmake/build.zig` and
//! the two literals the port returns (`src/ggml/runtime.zig`,
//! `src/llama/api/llama.zig`) to the submodule's `HEAD`.

/// The release tag the submodule is checked out at.
pub const tag = "v0.4.1";

/// `LLAMA_VERSION` with `LLAMA_BUILD_IS_DEV` off, as for a release tag
/// (llama.cpp/CMakeLists.txt:6-21).
pub const version = "0.4.1";

/// `GGML_VERSION`: ggml's own version, not llama.cpp's
/// (llama.cpp/ggml/CMakeLists.txt:6-31).
pub const ggml_version = "0.24.0";

/// `git rev-parse --short HEAD`, as cmake/build-info.cmake takes it.
pub const commit = "b29c606e2";

/// The build tag's number (`git describe --tags --match 'b*'`).
pub const build_number = 10964;
