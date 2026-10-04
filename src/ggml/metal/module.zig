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

pub const common = @import("common.zig");

comptime {
    _ = common;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
