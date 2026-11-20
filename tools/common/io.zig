//! What the tools write to standard output and error, with upstream's own
//! `printf` formats, through `sync.stdio`: the streams on `std.Io`, buffered
//! as C's stdio buffers them, which libggml's `ggml_abort` flushes too.
//!
//! # Provenance
//!
//! **Not a port.** A tool's `main` calls `begin` before it writes and `end`
//! on its way out; `cExit` flushes before it exits. Each message is
//! formatted with `snprintf` and the C's format string, as `log.zig`'s are:
//! Zig's formatting differs from `printf` in the last digit of a rounded
//! float, and the C-format formatter that replaces `snprintf` is its own
//! step.

const std = @import("std");
const builtin = @import("builtin");
const sync = @import("sync");

/// One of the two output streams.
pub const Stream = sync.stdio.Stream;

extern fn snprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, ...) c_int;
extern fn __acrt_iob_func(index: c_uint) callconv(.c) *anyopaque;

/// The process's standard output.
pub fn stdout() *Stream {
    return sync.stdio.stdout();
}

/// The process's standard error.
pub fn stderr() *Stream {
    return sync.stdio.stderr();
}

/// For a tool's `main`, before it writes: the streams buffered as C starts
/// a process.
pub fn begin() void {
    sync.stdio.bufferAsC();
}

/// For a tool's `main`, on its way out: what C's `exit` flushes.
pub fn end() void {
    sync.stdio.flushAll();
}

extern fn exit(code: c_int) noreturn;

/// The C's `exit`: what is buffered for stdout and stderr is written first,
/// as C's `exit` flushes stdio. Every tool exits through this. It still
/// ends in libc's `exit` while the console's input and the libraries hold C
/// streams that `exit` must flush.
pub fn exitFlushed(code: c_int) noreturn {
    end();
    exit(code);
}

/// Before a read from standard input: what C writes out first, a
/// line-buffered stdout.
pub fn beforeRead() void {
    sync.stdio.flushLineBuffered();
}

/// libc's `stdin`, which `console.zig` still reads with `getline` and
/// `fgetwc` until its input moves to `std.Io`. Darwin names it `__stdinp`
/// behind a macro, glibc exports it as itself, and MinGW's UCRT hands it
/// out through `__acrt_iob_func`.
pub fn cStdin() *anyopaque {
    if (comptime builtin.os.tag == .windows) return __acrt_iob_func(0);
    return @extern(*const *anyopaque, .{ .name = if (builtin.os.tag.isDarwin()) "__stdinp" else "stdin" }).*;
}

/// `fprintf(stream, fmt, ...)`: the C's format, written to `stream`.
pub fn fprintf(s: *Stream, comptime fmt: [*:0]const u8, args: anytype) void {
    var stack: [1024]u8 = undefined;
    const n = @call(.auto, snprintf, .{ &stack, stack.len, fmt } ++ args);
    if (n < 0) return;
    const len: usize = @intCast(n);
    if (len < stack.len) return s.write(stack[0..len]);
    const heap = std.heap.page_allocator.alloc(u8, len + 1) catch return;
    defer std.heap.page_allocator.free(heap);
    _ = @call(.auto, snprintf, .{ heap.ptr, heap.len, fmt } ++ args);
    s.write(heap[0..len]);
}

/// `printf(fmt, ...)`.
pub fn printf(comptime fmt: [*:0]const u8, args: anytype) void {
    fprintf(stdout(), fmt, args);
}

/// `fprintf(stderr, fmt, ...)`.
pub fn eprintf(comptime fmt: [*:0]const u8, args: anytype) void {
    fprintf(stderr(), fmt, args);
}

/// `fflush(stdout)`.
pub fn flushStdout() void {
    stdout().flush();
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
