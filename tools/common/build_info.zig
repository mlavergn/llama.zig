//! `build-info.h`'s four values and its printer.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/build-info.cpp.in` (v1.0.0, `b1b770d90`) as
//! `llama.cpp.zmake` fills it in -- build number, commit and compiler from
//! `build/tools.zig`, the same values the reference tools are built with, so
//! `--version` reads the same from both sides of a gate.

const std = @import("std");
const mod = @import("module.zig");
const stdlib = @import("stdlib");
const options = @import("build_info");

/// Ports `llama_build_number` (build-info.cpp.in:11 @b1b770d90).
pub const build_number: c_int = options.build_number;
/// Ports `llama_commit` (build-info.cpp.in:15 @b1b770d90).
pub const commit: [:0]const u8 = options.commit ++ "";
/// Ports `llama_compiler` (build-info.cpp.in:19 @b1b770d90).
pub const compiler: [:0]const u8 = options.compiler ++ "";
/// Ports `llama_build_target` (build-info.cpp.in:23 @b1b770d90).
pub const build_target: [:0]const u8 = options.build_target ++ "";

/// Ports `llama_print_build_info` (build-info.cpp.in:32 @b1b770d90): two lines
/// on `stream`, which `build-info.h` defaults to `stderr` -- every caller
/// ported passes `stdlib.stdio.stderr()`.
pub fn print(version: [*:0]const u8, stream: *stdlib.stdio.Stream) void {
    mod.io.fprintf(stream, "version: %s (build %d, commit %s)\n", .{ version, build_number, commit.ptr });
    mod.io.fprintf(stream, "built with %s for %s\n", .{ compiler.ptr, build_target.ptr });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
