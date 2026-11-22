//! Barrel for the ported CPU backend.
//!
//! # Provenance
//!
//! **Not a port.** Nothing here corresponds to a file in the reference
//! checkout; it is scaffolding this project owns, matching `module.zig` one
//! directory up.
//!
//! Everything it gathers is a port of
//! `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` at v1.0.0 (`b1b770d90`), split
//! seven ways:
//!
//! - `defs.zig`      — the structs and constants the rest share
//! - `traits.zig`    — the per-type kernel table
//! - `convert.zig`   — float conversions and the lookup tables
//! - `features.zig`  — what the target CPU can do, and one-time init
//! - `simd.zig`      — `simd-mappings.h`'s vector layer, chosen per target
//! - `tensor.zig`    — host reads and writes of single elements
//! - `mulmat.zig`    — matrix multiplication, plain and expert-routed
//! - `forward.zig`   — the op dispatch
//! - `plan.zig`      — thread counts and work-buffer sizing
//! - `threading.zig` — the threadpool and the graph loop
//!
//! `tiled.zig` and `tiled_kernel.zig` are `ggml-cpu/tiled/tiled.cpp` and
//! `ggml-cpu/tiled/tiled-kernel.cpp`, the tiled K-quant `mul_mat` that
//! `mulmat.zig` dispatches to (live on x86 only).
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

pub const defs = @import("defs.zig");
pub const traits = @import("traits.zig");
pub const convert = @import("convert.zig");
pub const features = @import("features.zig");
pub const tensor = @import("tensor.zig");
pub const mulmat = @import("mulmat.zig");
pub const forward = @import("forward.zig");
pub const plan = @import("plan.zig");
pub const threading = @import("threading.zig");
pub const quants = @import("quants/module.zig");
pub const binary_ops = @import("binary_ops.zig"); // ggml-cpu/binary-ops.cpp
pub const unary_ops = @import("unary_ops.zig"); // ggml-cpu/unary-ops.cpp
pub const vec = @import("vec.zig"); // ggml-cpu/vec.cpp
pub const tiled = @import("tiled.zig"); // ggml-cpu/tiled/tiled.cpp
pub const tiled_kernel = @import("tiled_kernel.zig"); // ggml-cpu/tiled/tiled-kernel.cpp
pub const ops = @import("ops/module.zig");
pub const extra = @import("extra.zig"); // ggml-cpu/traits.cpp
pub const cpu_backend = @import("cpu_backend.zig"); // ggml-cpu/ggml-cpu.cpp
pub const repack = @import("repack/module.zig"); // ggml-cpu/repack.cpp, arch/arm/repack.cpp
pub const vec_testing = @import("vec_testing.zig");
pub const simd = @import("simd.zig");

pub const vec_golden_generic = @import("vec_golden_generic.zig");

pub const vec_golden = @import("vec_golden.zig");

comptime {
    // Ported translation units export C symbols that nothing in Zig
    // references, so they need forcing into the compilation or the archive
    // ships without them and the link fails with symbols the C++ still wants.
    _ = defs;
    _ = traits;
    _ = convert;
    _ = features;
    _ = tensor;
    _ = mulmat;
    _ = forward;
    _ = plan;
    _ = threading;
    // ggml-cpu/quants.c, split across src/ggml/cpu/quants/:
    _ = quants;
    _ = binary_ops;
    _ = unary_ops;
    _ = vec;
    _ = tiled;
    _ = tiled_kernel;
    // ggml-cpu/ops.cpp, split by op family across src/ggml/cpu/ops/:
    _ = ops;
    // The vtable cluster, which moves as one unit -- ggml-cpu/traits.cpp,
    // ggml-cpu.cpp, repack.cpp and arch/arm/repack.cpp. See
    // `repack/module.zig` for why a partial swap reads as success.
    _ = extra;
    _ = cpu_backend;
    _ = repack;
    // Not a port: the golden check for vec.cpp's float dot products.
    _ = vec_testing;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = simd;
}
