//! x86's `arch-fallback.h` renames for `repack.cpp`: the 28 kernels
//! `arch/x86/repack.cpp` has no version of, so the generic body is the real
//! one.
//!
//! # Provenance
//!
//! **Not a port of a function.** It reproduces
//! `llama.cpp/ggml/src/ggml-cpu/arch-fallback.h` at v1.0.0 (`b1b770d90`),
//! lines 88-119, the `__x86_64__` block, which `#define`s each generic to
//! the bare name before `repack.cpp` compiles. The bodies are the generic
//! ports in `cpu/repack/`; this exports each under the name the C gives it.
//! `arm/fallback.zig` does the same for ARM's eight.

const repack = @import("../module.zig");
