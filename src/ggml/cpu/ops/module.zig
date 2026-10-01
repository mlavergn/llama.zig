//! Barrel for the `ops.cpp` port.
//!
//! **Not a port.** Nothing here corresponds to a file in the reference
//! checkout; it re-exports the kernels so siblings can reach each other
//! without importing across the directory, per the project's barrel
//! convention.
//!
//! `llama.cpp/ggml/src/ggml-cpu/ops.cpp` is 8,273 live lines and 88 exported
//! symbols — the largest translation unit in ggml — so it is split by op
//! family rather than kept as one file. The split is ours; the C has no
//! internal structure beyond `// ggml_compute_forward_xxx` banner comments,
//! and those are what the files follow.

pub const common = @import("common.zig");
pub const arith = @import("arith.zig");
pub const dup = @import("dup.zig");
pub const reduce = @import("reduce.zig");
pub const vecinline = @import("vecinline.zig");

comptime {
    _ = common;
    _ = arith;
    _ = dup;
    _ = reduce;
    _ = vecinline;
}

test {
    @import("std").testing.refAllDecls(@This());
}
