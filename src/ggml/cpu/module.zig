//! Barrel for the ported CPU backend.
//!
//! # Provenance
//!
//! **Not a port.** Nothing here corresponds to a file in the reference
//! checkout; it is scaffolding this project owns, matching `module.zig` one
//! directory up.
//!
//! Everything it gathers is a port of
//! `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` at v0.3.0 (`c1d0e7a00`), split
//! seven ways:
//!
//! - `defs.zig`      — the structs and constants the rest share
//! - `traits.zig`    — the per-type kernel table
//! - `convert.zig`   — float conversions and the lookup tables
//! - `features.zig`  — what the target CPU can do, and one-time init
//! - `tensor.zig`    — host reads and writes of single elements
//! - `mulmat.zig`    — matrix multiplication, plain and expert-routed
//! - `forward.zig`   — the op dispatch
//! - `plan.zig`      — thread counts and work-buffer sizing
//! - `threading.zig` — the threadpool and the graph loop
//!
//! `quants/` is a second translation unit, `ggml-cpu/quants.c`, gathered here
//! because it is part of the same backend.
//!
//! # What is still C
//!
//! The rest of `ggml/src/ggml-cpu/` is untouched: the kernels in `ops.cpp`,
//! `vec.cpp` and `repack.cpp`, the NEON dot products in `arch/arm/quants.c`,
//! and the backend registration in `ggml-cpu.cpp`. The NEON kernels are the
//! remainder of Stage 3 step 5; the C++ is Stage 4.

const std = @import("std");

comptime {
    // Ported translation units export C symbols that nothing in Zig
    // references, so they need forcing into the compilation or the archive
    // ships without them and the link fails with symbols the C++ still wants.
    _ = @import("defs.zig");
    _ = @import("traits.zig");
    _ = @import("convert.zig");
    _ = @import("features.zig");
    _ = @import("tensor.zig");
    _ = @import("mulmat.zig");
    _ = @import("forward.zig");
    _ = @import("plan.zig");
    _ = @import("threading.zig");
    // ggml-cpu/quants.c, split across src/ggml/cpu/quants/:
    _ = @import("quants/module.zig");
    _ = @import("binary_ops.zig"); // ggml-cpu/binary-ops.cpp
    _ = @import("unary_ops.zig"); // ggml-cpu/unary-ops.cpp
    _ = @import("vec.zig"); // ggml-cpu/vec.cpp
    // Not a port: the golden check for vec.cpp's float dot products.
    _ = @import("vec.zig");
    _ = @import("vec_testing.zig");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = @import("defs.zig");
    _ = @import("traits.zig");
    _ = @import("convert.zig");
    _ = @import("features.zig");
    _ = @import("tensor.zig");
    _ = @import("mulmat.zig");
    _ = @import("forward.zig");
    _ = @import("plan.zig");
    _ = @import("threading.zig");
    _ = @import("binary_ops.zig");
    _ = @import("unary_ops.zig");
    _ = @import("vec.zig");
    _ = @import("vec_testing.zig");
}
