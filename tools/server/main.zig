//! `llama-server`'s entry point: unpacks the process and hands it to
//! `server.zig`.
//!
//! # Provenance
//!
//! Ports `main` (tools/server/main.cpp:3 @d81235049) from
//! `llama.cpp/tools/server/main.cpp` (v0.6.0, `d81235049`).

const std = @import("std");
const server = @import("server.zig");

comptime {
    // libllama, compiled in (see `build/tools.zig`).
    _ = @import("llama");
}

/// Ports `main` (tools/server/main.cpp:3 @d81235049).
pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return server.run(arena, argv);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
