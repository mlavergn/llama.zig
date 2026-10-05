//! Barrel for the ported ggml.
//!
//! # Provenance
//!
//! Everything under `src/ggml/` is a port of C sources in the pinned reference
//! checkout, v0.3.0 (`c1d0e7a00`).
//!
//! **Every ported file states its source as a full path from the repository
//! root, with extension** — `llama.cpp/ggml/src/ggml-alloc.c`, not
//! `ggml-alloc.c` — so the original can be opened without guessing where it
//! lives. A file drawing on more than one source lists all of them.
//!
//! Within a file, each ported declaration names the C function and the line it
//! started at, using the bare filename since the header has already established
//! the path. Line numbers are as of the pinned commit and only move if the pin
//! moves, which per Decision 3 it does not.
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

comptime {
    // Ported translation units export C symbols that nothing in Zig references,
    // so they need forcing into the compilation or the archive ships without
    // them and the link fails with symbols the C++ side still expects.
    _ = @import("alloc.zig"); // ggml-alloc.c
    // ggml.c, split across seven files:
    _ = @import("impl.zig");
    _ = @import("types.zig");
    _ = @import("context.zig");
    _ = @import("runtime.zig");
    _ = @import("ops.zig");
    _ = @import("graph.zig");
    _ = @import("quantize.zig");
    // ggml-quants.c, split across src/ggml/quants/:
    _ = @import("quants/module.zig");
    // ggml-cpu/ggml-cpu.c, split across src/ggml/cpu/:
    _ = @import("cpu/module.zig");
    _ = @import("threading.zig"); // ggml-threading.cpp
    _ = @import("backend_reg.zig"); // ggml-backend-reg.cpp + ggml-backend-dl.cpp
    _ = @import("gguf.zig"); // gguf.cpp
    // ggml-backend.cpp, split in two: the vtable dispatch and the scheduler.
    _ = @import("backend.zig");
    _ = @import("backend_sched.zig");
    // ggml-metal/, the host layer. The two .m files stay Objective-C.
    //
    // **Behind `config.use_metal`, like the registry's reference to it.**
    // `metal/backend.zig` calls 39 functions that live in
    // `ggml-metal-device.m`, `-context.m`, `-device.cpp` and `-ops.cpp`,
    // and the `test-port` root deliberately links none of them — its own
    // comment says "the ported registry must not reference the Metal
    // backend". Importing this unconditionally made `make validate` fail
    // with 39 undefined symbols at that step.
    if (config.use_metal) _ = @import("metal/module.zig");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
