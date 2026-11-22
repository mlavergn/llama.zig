//! Files the tools read and write as upstream's do through `std::ifstream`,
//! `std::ofstream` and C `FILE *`s, on `std.Io.File`.
//!
//! # Provenance
//!
//! **Not a port.** An output is written straight through, unbuffered:
//! upstream's streams write through a C `FILE *`, which `exit` flushes, so
//! when a tool exits on a failure everything it wrote is on disk, and
//! writing straight through leaves the same file. An input is read at the
//! offset a `seekg` or `fseek` would set.

const std = @import("std");
const builtin = @import("builtin");
const errno = @import("errno.zig");

/// The `Io` for shared code no `main` hands one to: `tools_common` is also
/// compiled into `llama-server-core`, whose callers pass none. The global
/// instance starts no threads and installs no signal handlers.
pub fn defaultIo() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

/// What `Out.zeros` writes from.
const zero_block: [4096]u8 = @splat(0);

/// An output file, as upstream's `std::ofstream` or `FILE *` opened `"wb"`.
pub const Out = struct {
    w: std.Io.File.Writer,

    /// Creates or truncates `path`.
    ///
    /// Parameters:
    /// - `io`: the `Io` the file is written through; must outlive `Out`.
    /// - `path`: the file, relative to the working directory or absolute.
    ///
    /// Return: the file; `std.Io.File.OpenError` when it cannot be created.
    pub fn create(io: std.Io, path: []const u8) std.Io.File.OpenError!Out {
        const f = try std.Io.Dir.cwd().createFile(io, path, .{});
        return .{ .w = f.writer(io, &.{}) };
    }

    /// Opens `path` to write at its end, creating it when missing:
    /// `std::ofstream` opened `std::ios::app`, `fopen(path, "ab")`. One
    /// writer, so writing from the size at open lands where `O_APPEND`
    /// would.
    ///
    /// Parameters:
    /// - `io`: the `Io` the file is written through; must outlive `Out`.
    /// - `path`: the file, relative to the working directory or absolute.
    ///
    /// Return: the file; the open or stat error.
    pub fn append(io: std.Io, path: []const u8) !Out {
        const f = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = false });
        errdefer f.close(io);
        var out: Out = .{ .w = f.writer(io, &.{}) };
        try out.w.seekTo(try f.length(io));
        return out;
    }

    /// Writes `bytes` at the current offset.
    ///
    /// Return: nothing; `error.WriteFailed` when not all were written.
    pub fn write(self: *Out, bytes: []const u8) std.Io.Writer.Error!void {
        try self.w.interface.writeAll(bytes);
    }

    /// Writes `text` as a C stream opened in text mode writes it
    /// (`fopen(path, "w")`, a default `std::ofstream`): on Windows the C
    /// library turns each LF into CR LF; elsewhere text mode is binary.
    ///
    /// Return: nothing; `error.WriteFailed` when not all was written.
    pub fn writeText(self: *Out, text: []const u8) std.Io.Writer.Error!void {
        if (comptime builtin.os.tag != .windows) return self.write(text);
        var rest = text;
        while (std.mem.indexOfScalar(u8, rest, '\n')) |nl| {
            try self.write(rest[0..nl]);
            try self.write("\r\n");
            rest = rest[nl + 1 ..];
        }
        try self.write(rest);
    }

    /// Writes `n` zero bytes at the current offset: upstream's `zeros`,
    /// which writes them one at a time, as one write per block.
    ///
    /// Return: nothing; `error.WriteFailed` when not all were written.
    pub fn zeros(self: *Out, n: usize) std.Io.Writer.Error!void {
        var left = n;
        while (left > 0) {
            const k = @min(left, zero_block.len);
            try self.write(zero_block[0..k]);
            left -= k;
        }
    }

    /// Moves the offset the next write lands at: `seekp`, `fseek`.
    ///
    /// Return: nothing; an error when the offset cannot be set.
    pub fn seek(self: *Out, off: u64) !void {
        try self.w.seekTo(off);
    }

    /// Closes the file.
    ///
    /// Parameters:
    /// - `io`: the `Io` it was created with.
    pub fn close(self: *Out, io: std.Io) void {
        self.w.file.close(io);
    }

    /// Closes the file and says whether that failed, as `fclose` and a
    /// stream's `close` do: nothing is buffered, so only the system's
    /// `close` can fail (a delayed write error on a network file system),
    /// which `std.Io.File.close` does not report. Windows reports nothing.
    ///
    /// Return: false when `close` failed.
    pub fn closeChecked(self: *Out, io: std.Io) bool {
        if (comptime builtin.os.tag == .windows) {
            self.close(io);
            return true;
        }
        return std.posix.errno(std.posix.system.close(self.w.file.handle)) == .SUCCESS;
    }
};

/// Opens `path` to read: `std::ifstream` or `fopen(path, "rb")`.
///
/// Return: the file; `std.Io.File.OpenError` when it cannot be opened.
pub fn openIn(io: std.Io, path: []const u8) std.Io.File.OpenError!std.Io.File {
    return std.Io.Dir.cwd().openFile(io, path, .{});
}

/// Reads all of `path` as a C stream opened in text mode reads it
/// (`fopen(path, "r")`, a default `std::ifstream`): on Windows the C library
/// turns CR LF into LF and stops at Ctrl-Z; elsewhere text mode is binary.
///
/// Return: the text, owned by `a`; the open or read error.
pub fn readText(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .unlimited);
    if (comptime builtin.os.tag != .windows) return bytes;
    defer a.free(bytes);
    return a.dupe(u8, textModeWindows(bytes));
}

/// The Windows C library's text-mode read, in place: CR LF becomes LF, and
/// Ctrl-Z ends the file.
fn textModeWindows(bytes: []u8) []u8 {
    const end = std.mem.indexOfScalar(u8, bytes, 0x1a) orelse bytes.len;
    var w: usize = 0;
    for (0..end) |i| {
        if (bytes[i] == '\r' and i + 1 < end and bytes[i + 1] == '\n') continue;
        bytes[w] = bytes[i];
        w += 1;
    }
    return bytes[0..w];
}

/// Reads `buf.len` bytes of `f` at `off`: a `seekg` then a `read`, or an
/// `fseek` then an `fread`.
///
/// Return: false on a short read or a read error.
pub fn readAt(io: std.Io, f: std.Io.File, buf: []u8, off: u64) bool {
    const n = f.readPositionalAll(io, buf, off) catch return false;
    return n == buf.len;
}

/// Reads all of `path` as `std::ifstream(path, std::ios::binary)` and
/// `ss << f.rdbuf()` read it, or a loop of `read`s: a read error ends the
/// copy, keeping what came before it, so a directory, which opens, reads
/// as nothing.
///
/// Return: the bytes, owned by `a`; null when the file cannot be opened;
/// `error.OutOfMemory`.
pub fn readStream(io: std.Io, a: std.mem.Allocator, path: []const u8) std.mem.Allocator.Error!?[]u8 {
    const f = openIn(io, path) catch return null;
    defer f.close(io);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var buf: [65536]u8 = undefined;
    while (true) {
        const n = f.readStreaming(io, &.{&buf}) catch break;
        if (n == 0) break;
        try out.appendSlice(a, buf[0..n]);
    }
    return try out.toOwnedSlice(a);
}

/// The first line of `path`, as `std::getline` reads it from a fresh
/// `std::ifstream`: up to the first `'\n'`, which is dropped; a read error
/// ends it.
///
/// Return: the line, owned by `a`; null when the file cannot be opened;
/// empty when the file is.
pub fn readFirstLine(io: std.Io, a: std.mem.Allocator, path: []const u8) std.mem.Allocator.Error!?[]u8 {
    const f = openIn(io, path) catch return null;
    defer f.close(io);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var buf: [4096]u8 = undefined;
    while (true) {
        const n = f.readStreaming(io, &.{&buf}) catch break;
        if (n == 0) break;
        if (std.mem.indexOfScalar(u8, buf[0..n], '\n')) |nl| {
            try out.appendSlice(a, buf[0..nl]);
            break;
        }
        try out.appendSlice(a, buf[0..n]);
    }
    return try out.toOwnedSlice(a);
}

/// Deletes `path`: `remove`, `std::filesystem::remove`.
///
/// Return: false when it could not be deleted.
pub fn remove(io: std.Io, path: []const u8) bool {
    std.Io.Dir.cwd().deleteFile(io, path) catch return false;
    return true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "an output reads back as written: zeros, a seek back, an overwrite" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}/out.bin", .{&tmp.sub_path});
    defer std.testing.allocator.free(path);

    var out = try Out.create(io, path);
    try out.zeros(5000);
    try out.write("tail");
    try out.seek(0);
    try out.write("head");
    out.close(io);

    const f = try openIn(io, path);
    defer f.close(io);
    var buf: [5004]u8 = undefined;
    try std.testing.expect(readAt(io, f, &buf, 0));
    try std.testing.expectEqualStrings("head", buf[0..4]);
    try std.testing.expect(std.mem.allEqual(u8, buf[4..5000], 0));
    try std.testing.expectEqualStrings("tail", buf[5000..]);
    try std.testing.expect(!readAt(io, f, &buf, 1));

    var app = try Out.append(io, path);
    try app.write("!");
    try std.testing.expect(app.closeChecked(io));
    const all = (try readStream(io, std.testing.allocator, path)) orelse return error.TestUnexpectedResult;
    defer std.testing.allocator.free(all);
    try std.testing.expectEqual(@as(usize, 5005), all.len);
    try std.testing.expectEqualStrings("tail!", all[5000..]);

    try std.testing.expect(remove(io, path));
    try std.testing.expect(!remove(io, path));
    try std.testing.expect((try readStream(io, std.testing.allocator, path)) == null);
}

test "a directory opens and reads as nothing" {
    const all = (try readStream(std.testing.io, std.testing.allocator, ".")) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 0), all.len);
}

test "Windows text mode: CR LF becomes LF, a lone CR stays, Ctrl-Z ends the file" {
    var buf = "a\r\nb\rc\r\n\x1ad\r\n".*;
    try std.testing.expectEqualStrings("a\nb\rc\n", textModeWindows(&buf));
    var none = "plain\n".*;
    try std.testing.expectEqualStrings("plain\n", textModeWindows(&none));
}

test "a file error reads as the C library's text for its errno" {
    const io = std.testing.io;
    const missing = openIn(io, "/nonexistent/file.gguf");
    try std.testing.expectError(error.FileNotFound, missing);
    try std.testing.expectEqualStrings(std.mem.span(errno.strerror(@backingInt(std.c.E.NOENT))), std.mem.span(errno.text(error.FileNotFound)));
    try std.testing.expectEqualStrings("Unexpected", std.mem.span(errno.text(error.Unexpected)));
}

test "readFirstLine stops at the first newline and drops it" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/etag", .{&tmp.sub_path});
    defer a.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = "\"abc\"\nrest\n" });
    const line = (try readFirstLine(io, a, path)) orelse return error.TestUnexpectedResult;
    defer a.free(line);
    try std.testing.expectEqualStrings("\"abc\"", line);
    try std.testing.expect((try readFirstLine(io, a, "/nonexistent/etag")) == null);
}
