//! `llama-server`'s entry point: unpacks the process and hands it to
//! `server.zig`.
//!
//! # Provenance
//!
//! Ports `main` (tools/server/main.cpp:3 @cd959bae9) from
//! `llama.cpp/tools/server/main.cpp` (v0.8.0, `8642541b2`).

const std = @import("std");
const server = @import("server.zig");

comptime {
    // libllama, compiled in (see `build/tools.zig`).
    _ = @import("llama");
}

/// Ports `main` (tools/server/main.cpp:3 @cd959bae9).
pub fn main(init: std.process.Init) !u8 {
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
