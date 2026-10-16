//! Barrel for the CUDA host layer.
//!
//! **Not a port.** Everything it imports is; see each file. Built only with
//! `-Dcuda-port` until the swap (PLAN.md, Decision 48), and then the files
//! here replace upstream's `ggml-cuda` host code, whose kernels travel as
//! fatbins (Decision 44).

pub const common = @import("common.zig");
pub const kernels = @import("kernels.zig");
pub const backend = @import("backend.zig");
pub const graph = @import("graph.zig");
pub const supports = @import("supports.zig");

comptime {
    // `backend.zig` exports `ggml-cuda.h`.
    _ = backend;
}
