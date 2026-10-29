//! `build-info.h`'s four values and its printer.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/build-info.cpp.in` (v0.4.1, `b29c606e2`) as
//! `llama.cpp.zmake` fills it in -- build number, commit and compiler from
//! `build/tools.zig`, the same values the reference tools are built with, so
//! `--version` reads the same from both sides of a gate.

const std = @import("std");
const options = @import("build_info");
const io = @import("io.zig");

/// Ports `llama_build_number` (build-info.cpp.in:11 @b29c606e2).
pub const build_number: c_int = options.build_number;
/// Ports `llama_commit` (build-info.cpp.in:15 @b29c606e2).
pub const commit: [:0]const u8 = options.commit ++ "";
/// Ports `llama_compiler` (build-info.cpp.in:19 @b29c606e2).
pub const compiler: [:0]const u8 = options.compiler ++ "";
/// Ports `llama_build_target` (build-info.cpp.in:23 @b29c606e2).
pub const build_target: [:0]const u8 = options.build_target ++ "";

/// Ports `llama_print_build_info` (build-info.cpp.in:32 @b29c606e2): two lines
/// on `stream`, which `build-info.h` defaults to `stderr` -- every caller
/// ported passes `io.cStderr()`.
pub fn print(version: [*:0]const u8, stream: *anyopaque) void {
    io.fprintf(stream, "version: %s (build %d, commit %s)\n", .{ version, build_number, commit.ptr });
    io.fprintf(stream, "built with %s for %s\n", .{ compiler.ptr, build_target.ptr });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
