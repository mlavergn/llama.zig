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

pub const activation = @import("activation.zig");
pub const arith = @import("arith.zig");
pub const attention = @import("attention.zig");
pub const common = @import("common.zig");
pub const conv = @import("conv.zig");
pub const custom = @import("custom.zig");
pub const dup = @import("dup.zig");
pub const gather = @import("gather.zig");
pub const linalg = @import("linalg.zig");
pub const norm = @import("norm.zig");
pub const pad = @import("pad.zig");
pub const pool = @import("pool.zig");
pub const recurrent = @import("recurrent.zig");
pub const reduce = @import("reduce.zig");
pub const repeat = @import("repeat.zig");
pub const rope = @import("rope.zig");
pub const softmax = @import("softmax.zig");
pub const sgemm = @import("sgemm.zig");
pub const sort = @import("sort.zig");
pub const ssm = @import("ssm.zig");
pub const vecinline = @import("vecinline.zig");
pub const window = @import("window.zig");

comptime {
    _ = activation;
    _ = arith;
    _ = attention;
    _ = common;
    _ = conv;
    _ = custom;
    _ = dup;
    _ = gather;
    _ = linalg;
    _ = norm;
    _ = pad;
    _ = pool;
    _ = recurrent;
    _ = reduce;
    _ = repeat;
    _ = rope;
    _ = softmax;
    _ = sgemm;
    _ = sort;
    _ = ssm;
    _ = vecinline;
    _ = window;
}

test {
    @import("std").testing.refAllDecls(@This());
}
