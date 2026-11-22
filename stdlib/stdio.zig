//! The process's standard output and error as C's stdio buffers them, on
//! `std.Io`, shared by every module that writes to them.
//!
//! **Not a port.** It lives in `stdlib` because every module shares that
//! one instance: libggml's `ggml_abort` flushes the same stdout buffer a
//! tool writes to, as upstream's `fflush(stdout)` flushes the tool's.
//!
//! # Buffering
//!
//! Unbuffered by default, which is right when the libraries are linked
//! into an application that writes to the same descriptors through its own
//! C library. An executable that owns its streams calls `bufferAsC`: stdout
//! line-buffered on a terminal and fully buffered otherwise, stderr
//! unbuffered, as C starts a process; it calls `flushAll` before it exits,
//! as C's `exit` flushes. A line-buffered write goes out through its last
//! newline and keeps the rest, as glibc's and Darwin's `fwrite` do.
//!
//! # Text mode
//!
//! On Windows C's streams start in text mode, so each `\n` reaches the
//! descriptor as `\r\n`; this writes them so.

const std = @import("std");
const builtin = @import("builtin");
const Mutex = @import("sync/mutex.zig").Mutex;

/// How a stream holds what is written to it: `_IONBF`, `_IOLBF`, `_IOFBF`.
pub const Buffering = enum { none, line, full };

/// Where a stream's bytes go.
pub const Target = union(enum) {
    out,
    err,
    /// A file the owner opened and closes: a C stream from `fopen`.
    file: std.Io.File,
};

/// An output stream: standard output or error, or a file written as C
/// writes one it opened with `fopen(path, "w")`.
pub const Stream = struct {
    target: Target,
    buffering: Buffering = .none,
    /// Windows text mode: each `\n` is written as `\r\n`.
    text: bool = builtin.os.tag == .windows,
    mutex: Mutex = .{},
    len: usize = 0,
    buf: [4096]u8 = undefined,

    fn file(self: *const Stream) std.Io.File {
        return switch (self.target) {
            .out => std.Io.File.stdout(),
            .err => std.Io.File.stderr(),
            .file => |f| f,
        };
    }

    /// Writes `bytes`, buffered as the stream is.
    ///
    /// Parameters:
    /// - `bytes`: what to write; borrowed for the call.
    pub fn write(self: *Stream, bytes: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        switch (self.buffering) {
            .none => {
                self.put(bytes);
                self.drain();
            },
            .full => self.put(bytes),
            .line => {
                const last_nl = std.mem.lastIndexOfScalar(u8, bytes, '\n') orelse {
                    self.put(bytes);
                    return;
                };
                self.put(bytes[0 .. last_nl + 1]);
                self.drain();
                self.put(bytes[last_nl + 1 ..]);
            },
        }
    }

    /// Writes out what is buffered: `fflush`.
    pub fn flush(self: *Stream) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.drain();
    }

    /// Appends `bytes` to the buffer, translated in text mode, writing the
    /// buffer out each time it fills.
    fn put(self: *Stream, bytes: []const u8) void {
        for (bytes) |ch| {
            if (self.text and ch == '\n') self.putByte('\r');
            self.putByte(ch);
        }
    }

    fn putByte(self: *Stream, ch: u8) void {
        if (self.len == self.buf.len) self.drain();
        self.buf[self.len] = ch;
        self.len += 1;
    }

    /// Writes the buffer to the descriptor. A failed write drops what was
    /// buffered, as C's stdio does on a write error.
    fn drain(self: *Stream) void {
        if (self.len == 0) return;
        const io = std.Io.Threaded.global_single_threaded.io();
        self.file().writeStreamingAll(io, self.buf[0..self.len]) catch {};
        self.len = 0;
    }
};

var out_stream: Stream = .{ .target = .out };
var err_stream: Stream = .{ .target = .err };

/// The process's standard output.
pub fn stdout() *Stream {
    return &out_stream;
}

/// The process's standard error.
pub fn stderr() *Stream {
    return &err_stream;
}

/// Buffers both streams as C starts a process: stdout line-buffered on a
/// terminal and fully buffered otherwise, stderr unbuffered. For an
/// executable that owns its streams, before it writes.
pub fn bufferAsC() void {
    const io = std.Io.Threaded.global_single_threaded.io();
    const tty = std.Io.File.stdout().isTty(io) catch false;
    out_stream.mutex.lock();
    out_stream.buffering = if (tty) .line else .full;
    out_stream.mutex.unlock();
}

/// What C does before a read from standard input: a line-buffered stdout
/// is written out, so a prompt without a newline is on screen first.
pub fn flushLineBuffered() void {
    out_stream.mutex.lock();
    defer out_stream.mutex.unlock();
    if (out_stream.buffering == .line) out_stream.drain();
}

/// Writes out both streams' buffers: what C's `exit` and `fflush(NULL)` do.
pub fn flushAll() void {
    out_stream.flush();
    err_stream.flush();
}

/// C's `exit(code)`: what is buffered for stdout and stderr is written
/// first, as C's `exit` flushes stdio, then the C library's `exit`, which
/// still flushes the C streams the libraries and the console write through.
pub fn exit(code: c_int) noreturn {
    flushAll();
    @import("libc.zig").exit(code);
}

/// One of the C library's three standard streams.
pub const CStream = enum(c_uint) { stdin = 0, stdout = 1, stderr = 2 };

/// The C library's `FILE *` for `which`, for code that still writes
/// through C's stdio: Darwin names them `__stdinp`, `__stdoutp` and
/// `__stderrp` behind a macro, glibc exports them as themselves, and
/// MinGW's UCRT hands them out through `__acrt_iob_func`.
///
/// Return: the stream, the C library's for the process's life.
pub fn cStream(which: CStream) *anyopaque {
    if (comptime builtin.os.tag == .windows) return @import("libc.zig").__acrt_iob_func(@backingInt(which));
    return switch (which) {
        inline else => |w| @extern(*const *anyopaque, .{
            .name = if (builtin.os.tag.isDarwin()) "__" ++ @tagName(w) ++ "p" else @tagName(w),
        }).*,
    };
}

// ---------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the buffer holds bytes as written, with CR LF for LF in text mode" {
    var s: Stream = .{ .target = .out, .buffering = .full, .text = false };
    s.put("ab\ncd");
    try std.testing.expectEqualStrings("ab\ncd", s.buf[0..s.len]);
    var t: Stream = .{ .target = .out, .buffering = .full, .text = true };
    t.put("a\nb\r\n");
    try std.testing.expectEqualStrings("a\r\nb\r\r\n", t.buf[0..t.len]);
}

test "cStream names the C library's three streams" {
    try std.testing.expect(cStream(.stdin) != cStream(.stdout));
    try std.testing.expect(cStream(.stdout) != cStream(.stderr));
}
