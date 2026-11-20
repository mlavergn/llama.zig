//! `stat` and `lstat`, as the ported tools ask them, on `std.Io`.
//!
//! # Provenance
//!
//! **Not a port of a file.** Upstream calls `stat` through libc and
//! `std::filesystem`; this asks `std.Io.Dir.statFile`, on every target, and
//! answers in `stat`'s terms: the file type bits of `st_mode`, the size, and
//! on failure the `errno` (`file.errnoOf`). On Windows the C runtime has no
//! `lstat`, so a symlink is always followed, and MinGW's `stat` strips a
//! trailing separator, short of a root; both are kept.

const std = @import("std");
const builtin = @import("builtin");
const file = @import("file.zig");

/// The file type bits of `st_mode`: the same on Linux, Darwin and the CRT.
pub const IFMT: u32 = 0o170000;
pub const IFDIR: u32 = 0o040000;
pub const IFREG: u32 = 0o100000;
pub const IFLNK: u32 = 0o120000;

/// What the tools read of a `struct stat`: the file type bits of `st_mode`
/// (no permission bits) and the size.
pub const Info = struct {
    mode: u32 = 0,
    size: u64 = 0,
};

/// `stat(path)`, or `lstat(path)` when `follow` is false.
///
/// Parameters:
///   path   - the path, NUL-terminated
///   follow - whether a final symlink is followed
///   out    - receives the mode and size on success
/// Return: 0 on success, else the `errno` value `stat` would have set.
pub fn statErrno(path: [*:0]const u8, follow: bool, out: *Info) c_int {
    var name: []const u8 = std.mem.span(path);
    if (comptime builtin.os.tag == .windows) {
        while (name.len > 1 and (name[name.len - 1] == '/' or name[name.len - 1] == '\\') and name[name.len - 2] != ':') {
            name = name[0 .. name.len - 1];
        }
    }
    const follow_symlinks = follow or builtin.os.tag == .windows;
    const st = std.Io.Dir.cwd().statFile(file.defaultIo(), name, .{ .follow_symlinks = follow_symlinks }) catch |e| {
        return file.errnoOf(e) orelse @backingInt(std.c.E.IO);
    };
    out.* = .{
        .mode = switch (st.kind) {
            .directory => IFDIR,
            .file => IFREG,
            .sym_link => IFLNK,
            else => 0,
        },
        .size = st.size,
    };
    return 0;
}

/// `stat(path)` (or `lstat`), ignoring why it failed.
///
/// Parameters:
///   path   - the path, NUL-terminated
///   follow - whether a final symlink is followed
/// Return: the mode and size, or null when the call failed.
pub fn stat(path: [*:0]const u8, follow: bool) ?Info {
    var info: Info = .{};
    return if (statErrno(path, follow, &info) == 0) info else null;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "stat classifies a directory and a missing path" {
    const dir = stat(".", true) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(IFDIR, dir.mode & IFMT);
    var info: Info = .{};
    try std.testing.expectEqual(@as(c_int, @backingInt(std.c.E.NOENT)), statErrno("/nonexistent/llamazig", true, &info));
}

test "a file is regular, with its size" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, ".zig-cache/tmp/{s}/f", .{&tmp.sub_path}, 0);
    var out = try file.Out.create(io, path);
    try out.write("12345");
    out.close(io);
    const info = stat(path, true) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(IFREG, info.mode & IFMT);
    try std.testing.expectEqual(@as(u64, 5), info.size);
}
