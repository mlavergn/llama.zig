//! What the tools write to standard output and error, with upstream's own
//! `printf` formats, through `stdlib.stdio`: the streams on `std.Io`, buffered
//! as C's stdio buffers them, which libggml's `ggml_abort` flushes too.
//!
//! # Provenance
//!
//! **Not a port.** A tool's `main` calls `stdlib.stdio.bufferAsC` before it
//! writes and `stdlib.stdio.flushAll` on its way out; `stdlib.stdio.exit`
//! flushes before it exits. Each message is formatted with `snprintf` and
//! the C's format string, as `log.zig`'s are:
//! Zig's formatting differs from `printf` in the last digit of a rounded
//! float, and the C-format formatter that replaces `snprintf` is its own
//! step.

const std = @import("std");
const stdlib = @import("stdlib");

/// `fprintf(stream, fmt, ...)`: the C's format, written to `stream`.
pub fn fprintf(s: *stdlib.stdio.Stream, comptime fmt: [*:0]const u8, args: anytype) void {
    var stack: [1024]u8 = undefined;
    const n = @call(.auto, stdlib.libc.snprintf, .{ &stack, stack.len, fmt } ++ args);
    if (n < 0) return;
    const len: usize = @intCast(n);
    if (len < stack.len) return s.write(stack[0..len]);
    const heap = std.heap.page_allocator.alloc(u8, len + 1) catch return;
    defer std.heap.page_allocator.free(heap);
    _ = @call(.auto, stdlib.libc.snprintf, .{ heap.ptr, heap.len, fmt } ++ args);
    s.write(heap[0..len]);
}

/// `printf(fmt, ...)`.
pub fn printf(comptime fmt: [*:0]const u8, args: anytype) void {
    fprintf(stdlib.stdio.stdout(), fmt, args);
}

/// `fprintf(stderr, fmt, ...)`.
pub fn eprintf(comptime fmt: [*:0]const u8, args: anytype) void {
    fprintf(stdlib.stdio.stderr(), fmt, args);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
