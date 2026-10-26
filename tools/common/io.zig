//! The C library's `stdout` and `stderr`, which the tools write through with
//! upstream's own `printf` formats.
//!
//! # Provenance
//!
//! **Not a port.** `cStderr` is `src/ggml/impl.zig`'s, for the same reason:
//! Darwin names the streams `__stdoutp` and `__stderrp` behind macros, glibc
//! exports them as themselves, and MinGW's UCRT hands them out through
//! `__acrt_iob_func`. Writing through C stdio rather than `std.Io` keeps the
//! C's buffering too -- `stdout` buffered, `stderr` not -- so a tool's two
//! streams interleave on a terminal as upstream's do.

const std = @import("std");
const builtin = @import("builtin");

extern fn fprintf(stream: *anyopaque, format: [*:0]const u8, ...) c_int;
extern fn fflush(stream: ?*anyopaque) c_int;
extern fn __acrt_iob_func(index: c_uint) callconv(.c) *anyopaque;

fn stream(comptime index: c_uint, comptime name: []const u8) *anyopaque {
    if (comptime builtin.os.tag == .windows) return __acrt_iob_func(index);
    return @extern(*const *anyopaque, .{ .name = if (builtin.os.tag.isDarwin()) "__" ++ name ++ "p" else name }).*;
}

/// libc's `stdin`.
pub fn cStdin() *anyopaque {
    return stream(0, "stdin");
}

/// libc's `stdout`.
pub fn cStdout() *anyopaque {
    return stream(1, "stdout");
}

/// libc's `stderr`.
pub fn cStderr() *anyopaque {
    return stream(2, "stderr");
}

/// `printf(fmt, ...)`.
pub fn printf(comptime fmt: [*:0]const u8, args: anytype) void {
    _ = @call(.auto, fprintf, .{ cStdout(), fmt } ++ args);
}

/// `fprintf(stderr, fmt, ...)`.
pub fn eprintf(comptime fmt: [*:0]const u8, args: anytype) void {
    _ = @call(.auto, fprintf, .{ cStderr(), fmt } ++ args);
}

/// `fflush(stdout)`.
pub fn flushStdout() void {
    _ = fflush(cStdout());
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
