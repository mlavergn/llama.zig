//! Barrel for the ported ggml.
//!
//! # Provenance
//!
//! Everything under `src/ggml/` is a port of C sources in the pinned reference
//! checkout, v1.0.0 (`b1b770d90`).
//!
//! **Every ported file states its source as a full path from the repository
//! root, with extension** — `llama.cpp/ggml/src/ggml-alloc.c`, not
//! `ggml-alloc.c` — so the original can be opened without guessing where it
//! lives. A file drawing on more than one source lists all of them.
//!
//! Within a file, each ported declaration names the C function and the line it
//! started at, using the bare filename since the header has already established
//! the path. Line numbers are as of each citation's commit and move when a
//! sync re-reads the definition.
//!
//! This file is the exception: it is a barrel, not a port of anything.
//!
//! # How the swap works
//!
//! A ported file keeps the C ABI of the translation unit it replaces: same
//! symbols, same signatures. `build/llamacpp.zig` drops the `.c` file from the
//! source list, and the still-C++ code above links against these exports
//! without noticing. That makes every file an independent, revertible step
//! that the parity harness can check on its own.

const std = @import("std");
const config = @import("config");

pub const alloc = @import("alloc.zig"); // ggml-alloc.c
pub const impl = @import("impl.zig");
pub const types = @import("types.zig");
pub const context = @import("context.zig");
pub const runtime = @import("runtime.zig");
pub const ops = @import("ops.zig");
pub const graph = @import("graph.zig");
pub const quantize = @import("quantize.zig");
pub const quants = @import("quants/module.zig");
pub const cpu = @import("cpu/module.zig");
pub const threading = @import("threading.zig"); // ggml-threading.cpp
pub const backend_reg = @import("backend_reg.zig"); // ggml-backend-reg.cpp + ggml-backend-dl.cpp
pub const gguf = @import("gguf.zig"); // gguf.cpp
pub const backend = @import("backend.zig");
pub const backend_sched = @import("backend_sched.zig");
pub const backend_meta = @import("backend_meta.zig"); // ggml-backend-meta.cpp
pub const opt = @import("opt.zig"); // ggml-opt.cpp
pub const metal = if (config.use_metal) @import("metal/module.zig") else struct {};
pub const cuda = if (config.cuda_port) @import("cuda/module.zig") else struct {};
pub const vulkan = if (config.vulkan_port) @import("vulkan/module.zig") else struct {};
pub const blas = if (config.blas_port) @import("blas.zig") else struct {};

pub const win32 = @import("win32.zig");

/// The C API: every exported symbol. Forced below only when libggml is the
/// compilation root or under test; see `lib.zig`.
pub const lib = @import("lib.zig");

comptime {
    if (@import("root") == @This() or @import("builtin").is_test) _ = lib;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
