//! `llama-server`'s entry point: unpacks the process and hands it to
//! `server.zig`.
//!
//! # Provenance
//!
//! Ports `main` (tools/server/main.cpp:3 @b1b770d90) from
//! `llama.cpp/tools/server/main.cpp` (v1.0.0, `b1b770d90`).

const std = @import("std");
const stdlib = @import("stdlib");
const server = @import("server.zig");

/// Ports `main` (tools/server/main.cpp:3 @b1b770d90).
pub fn main(init: std.process.Init) !u8 {
    stdlib.stdio.bufferAsC();
    stdlib.env.set(init.environ_map);
    defer stdlib.stdio.flushAll();
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return server.run(arena, argv);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = @import("models.zig");
    _ = @import("models_sched.zig");
    _ = @import("download.zig");
    _ = @import("hf_cache.zig");
    _ = @import("file_lock.zig");
    _ = @import("connect.zig");
}
