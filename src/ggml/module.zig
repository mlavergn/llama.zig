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
//! This file is the exception: it is a barrel, not a port of anything, and
//! `ported.zig` is likewise scaffolding.
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

comptime {
    // libggml built on its own (or under test): every ported file, so the
    // archive exports the C API. A file that reaches this barrel from inside
    // the port, or a gate that compiles one file against upstream's C, forces
    // nothing.
    if (@import("root") == @This() or @import("builtin").is_test) {
        // Ported translation units export C symbols that nothing in Zig references,
        // so they need forcing into the compilation or the archive ships without
        // them and the link fails with symbols the C++ side still expects.
        _ = alloc;
        // ggml.c, split across seven files:
        _ = impl;
        _ = types;
        _ = context;
        _ = runtime;
        _ = ops;
        _ = graph;
        _ = quantize;
        // ggml-quants.c, split across src/ggml/quants/:
        _ = quants;
        // ggml-cpu/ggml-cpu.c, split across src/ggml/cpu/:
        _ = cpu;
        _ = threading;
        _ = backend_reg;
        _ = gguf;
        // ggml-backend.cpp, split in two: the vtable dispatch and the scheduler.
        _ = backend;
        _ = backend_sched;
        _ = backend_meta;
        _ = opt;
        // ggml-metal/, the host layer. The two .m files stay Objective-C.
        //
        // **Behind `config.use_metal`, like the registry's reference to it.**
        // `metal/backend.zig` calls 39 functions that live in
        // `ggml-metal-device.m`, `-context.m`, `-device.cpp` and `-ops.cpp`,
        // and the `test-port` root deliberately links none of them — its own
        // comment says "the ported registry must not reference the Metal
        // backend". Importing this unconditionally made `make validate` fail
        // with 39 undefined symbols at that step.
        if (config.use_metal) _ = metal;
        // ggml-cuda/, the host layer; the kernels travel as fatbins.
        // Behind `config.cuda_port` until the swap:
        // until then the default build links upstream's host objects, which
        // export the same ten `ggml-cuda.h` symbols.
        if (config.cuda_port) _ = cuda;
        // ggml-vulkan/, the host layer; the shaders travel as the reference
        // generator's SPIR-V. Behind
        // `config.vulkan_port` until the swap: until then the default build
        // compiles upstream's `ggml-vulkan.cpp`, which exports the same symbols.
        if (config.vulkan_port) _ = vulkan;
        // ggml-blas/, over Accelerate only. Behind
        // `config.blas_port`: `-Dblas-port=false` compiles upstream's
        // `ggml-blas.cpp` instead, for the gates, which exports the same four
        // symbols.
        if (config.blas_port) _ = blas;
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
