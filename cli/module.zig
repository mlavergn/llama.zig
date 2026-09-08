//! Barrel for the `llamazig` CLI.
//!
//! # Provenance
//!
//! **Not a port.** Nothing under `cli/` corresponds to a file in the reference
//! checkout. The flag surface is modelled on upstream's `llama-cli` (see
//! `args.zig`), but the code is ours.
//!
//! Siblings import through this file rather than reaching for each other, and
//! the executable is thin enough that everything it does stays reachable from
//! the unit tests.

const std = @import("std");

pub const c = @import("c.zig");
pub const args = @import("args.zig");
pub const chat = @import("chat.zig");
pub const session = @import("session.zig");
pub const upstream_flags = @import("upstream_flags.zig");

pub const Args = args.Args;
pub const Template = chat.Template;
pub const Message = chat.Message;
pub const Session = session.Session;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = @import("c.zig");
    _ = @import("args.zig");
    _ = @import("chat.zig");
    _ = @import("session.zig");
}
