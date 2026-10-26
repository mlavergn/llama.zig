//! `build-info.h`'s four values and its printer.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/build-info.cpp.in` (v0.3.0, `c1d0e7a00`) as
//! `llama.cpp.zmake` fills it in -- build number, commit and compiler from
//! `build/tools.zig`, the same values the reference tools are built with, so
//! `--version` reads the same from both sides of a gate.

const std = @import("std");
const options = @import("build_info");
const io = @import("io.zig");

/// Ports `llama_build_number` (build-info.cpp.in:11 @c1d0e7a00).
pub const build_number: c_int = options.build_number;
/// Ports `llama_commit` (build-info.cpp.in:15 @c1d0e7a00).
pub const commit: [:0]const u8 = options.commit ++ "";
/// Ports `llama_compiler` (build-info.cpp.in:19 @c1d0e7a00).
pub const compiler: [:0]const u8 = options.compiler ++ "";
/// Ports `llama_build_target` (build-info.cpp.in:23 @c1d0e7a00).
pub const build_target: [:0]const u8 = options.build_target ++ "";

/// Ports `llama_print_build_info` (build-info.cpp.in:32 @c1d0e7a00): two lines
/// on stderr.
pub fn print(version: [*:0]const u8) void {
    io.eprintf("version: %s (build %d, commit %s)\n", .{ version, build_number, commit.ptr });
    io.eprintf("built with %s for %s\n", .{ compiler.ptr, build_target.ptr });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
