//! ggml's goldens: the bits the C reference computes for kernels no other
//! gate reaches (the `_generic` dot products, the float reductions, the
//! quantizers), and the tests that hold the port to them.
//!
//! **Not a port.** The `test-port` step compiles this as its own test root,
//! importing the ported ggml as the module `ggml`. The ARM kernel checks
//! exist only where those kernels compile.

const std = @import("std");
const builtin = @import("builtin");

pub const cpu_quants_arm_golden = @import("cpu_quants_arm_golden.zig");
pub const cpu_quants_arm_golden_generic = @import("cpu_quants_arm_golden_generic.zig");
pub const cpu_quants_arm_iq = if (builtin.cpu.arch == .aarch64) @import("cpu_quants_arm_iq.zig") else struct {};
pub const cpu_quants_arm_k = if (builtin.cpu.arch == .aarch64) @import("cpu_quants_arm_k.zig") else struct {};
pub const cpu_quants_arm_legacy = if (builtin.cpu.arch == .aarch64) @import("cpu_quants_arm_legacy.zig") else struct {};
pub const cpu_quants_arm_rows = if (builtin.cpu.arch == .aarch64) @import("cpu_quants_arm_rows.zig") else struct {};
pub const cpu_quants_arm_ternary = if (builtin.cpu.arch == .aarch64) @import("cpu_quants_arm_ternary.zig") else struct {};
pub const cpu_quants_fixture = @import("cpu_quants_fixture.zig");
pub const cpu_quants_golden = @import("cpu_quants_golden.zig");
pub const cpu_quants_iq = @import("cpu_quants_iq.zig");
pub const cpu_quants_k = @import("cpu_quants_k.zig");
pub const cpu_quants_legacy = @import("cpu_quants_legacy.zig");
pub const cpu_quants_rows = @import("cpu_quants_rows.zig");
pub const cpu_quants_ternary = @import("cpu_quants_ternary.zig");
pub const cpu_vec = @import("cpu_vec.zig");
pub const cpu_vec_golden = @import("cpu_vec_golden.zig");
pub const cpu_vec_golden_generic = @import("cpu_vec_golden_generic.zig");
pub const features = @import("features.zig");
pub const impl = @import("impl.zig");
pub const types = @import("types.zig");
pub const quants_legacy = @import("quants_legacy.zig");
pub const quants_k = @import("quants_k.zig");
pub const quants_iq4 = @import("quants_iq4.zig");
pub const quants_ternary = @import("quants_ternary.zig");
pub const quants_chunks = @import("quants_chunks.zig");
pub const quants_fixture = @import("quants_fixture.zig");
pub const quants_golden = @import("quants_golden.zig");
pub const quants_iq = @import("quants_iq.zig");

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
