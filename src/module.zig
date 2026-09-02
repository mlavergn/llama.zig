//! Library barrel for llamazig.
//!
//! Re-exports every public type in `src/` so sibling files and consumers
//! import through this module rather than reaching for each other directly.

const std = @import("std");
const builtin = @import("builtin");

pub const is_debug = builtin.mode == .Debug;

pub const Base = @import("base.zig").Base;

// -----------------------------------------------------------------------------
// Test Fixtures
//
// Shared by the unit tests across this module so every file greets the same
// subject and asserts against the same rendering of it.

/// The subject the tests greet.
pub const test_subject: []const u8 = "Zig";

/// The greeting `test_subject` is expected to produce.
pub const test_greeting: []const u8 = "Hello Zig";

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
