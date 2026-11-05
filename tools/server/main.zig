//! `llama-server`'s entry point: unpacks the process and hands it to
//! `server.zig`.
//!
//! # Provenance
//!
//! Ports `main` (tools/server/main.cpp:3 @620d2225c) from
//! `llama.cpp/tools/server/main.cpp` (v0.7.0, `620d2225c`).

const std = @import("std");
const server = @import("server.zig");

comptime {
    // libllama, compiled in (see `build/tools.zig`).
    _ = @import("llama");
}

/// Ports `main` (tools/server/main.cpp:3 @620d2225c).
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
