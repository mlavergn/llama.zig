//! CLI barrel for the llamazig demo executable.
//!
//! Re-exports every public type in `cli/` so the entry point imports through
//! this module rather than reaching for sibling files directly.

const std = @import("std");

pub const Client = @import("client.zig").Client;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
