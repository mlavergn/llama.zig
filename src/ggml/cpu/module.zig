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
//! Nothing under `ggml/src/ggml-cpu/` compiles from C or C++ on this
//! target any more, apart from three translation units that are empty
//! here: `hbm.cpp` (needs `GGML_USE_CPU_HBM`), `amx/amx.cpp` and
//! `amx/mmq.cpp` (both `__AMX_INT8__`).

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
    // ggml-cpu/ops.cpp, split by op family across src/ggml/cpu/ops/:
    _ = @import("ops/module.zig");
    // The vtable cluster, which moves as one unit -- ggml-cpu/traits.cpp,
    // ggml-cpu.cpp, repack.cpp and arch/arm/repack.cpp. See
    // `repack/module.zig` for why a partial swap reads as success.
    _ = @import("extra.zig"); // ggml-cpu/traits.cpp
    _ = @import("cpu_backend.zig"); // ggml-cpu/ggml-cpu.cpp
    _ = @import("repack/module.zig"); // ggml-cpu/repack.cpp, arch/arm/repack.cpp
    // Not a port: the golden check for vec.cpp's float dot products.
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
    _ = @import("ops/module.zig");
    _ = @import("vec_testing.zig");
}
