//! Barrel for the `ggml-metal/` host-layer port.
//!
//! **Not a port.** It re-exports the pieces so siblings reach each other
//! through one import, per the project's barrel convention.
//!
//! # What stays Objective-C
//!
//! `ggml-metal-device.m` and `ggml-metal-context.m` (3,091 lines) are
//! never ported — Decision 13. The boundary between them and this
//! directory is a pure C ABI, measured: the `.m` files need 9 symbols
//! from the C++ group and provide 57 to it, none of them C++-linkage.

const std = @import("std");

pub const backend = @import("backend.zig");
pub const common = @import("common.zig");
pub const device_c = @import("device_c.zig");
pub const impl_c = @import("impl_c.zig");
pub const kargs = @import("kargs.zig");
pub const library = @import("library.zig");
pub const ops = @import("ops.zig");

pub const tuning = @import("tuning.zig");
pub const tuning_table = @import("tuning_table.zig");

comptime {
    _ = backend;
    _ = common;
    _ = device_c;
    _ = impl_c;
    _ = kargs;
    _ = library;
    _ = ops;
    _ = tuning;
    _ = tuning_table;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
