//! `stat` and `lstat`, as the ported tools ask them, on every POSIX target.
//!
//! # Provenance
//!
//! **Not a port of a file.** Upstream calls `stat` through libc and
//! `std::filesystem`; the port did too, through `std.c.Stat`, which is
//! `void` on Linux (Zig 0.17 declares no glibc `struct stat`), so nothing
//! that named it compiled there. On Linux this asks the kernel with
//! `statx`, as `tools/quantize.zig`'s `sameFile` already did, which is what
//! glibc's own `stat` does underneath; elsewhere it is libc's `stat`.

const std = @import("std");
const builtin = @import("builtin");

/// The file type bits of `st_mode`: the same on Linux and Darwin.
pub const IFMT: u32 = 0o170000;
pub const IFDIR: u32 = 0o040000;
pub const IFREG: u32 = 0o100000;
pub const IFLNK: u32 = 0o120000;

/// What the tools read of a `struct stat`.
pub const Info = struct {
    mode: u32 = 0,
    size: u64 = 0,
};

extern "c" fn lstat(path: [*:0]const u8, buf: *std.c.Stat) c_int;

/// `stat(path)`, or `lstat(path)` when `follow` is false.
///
/// Parameters:
///   path   - the path, NUL-terminated
///   follow - whether a final symlink is followed
///   out    - receives the mode and size on success
/// Return: 0 on success, else the `errno` value `stat` would have set.
pub fn statErrno(path: [*:0]const u8, follow: bool, out: *Info) c_int {
    if (comptime builtin.os.tag == .linux) {
        const linux = std.os.linux;
        var sx: linux.Statx = undefined;
        const flags: u32 = if (follow) 0 else linux.AT.SYMLINK_NOFOLLOW;
        const rc = linux.statx(linux.AT.FDCWD, path, flags, .{ .TYPE = true, .MODE = true, .SIZE = true }, &sx);
        const e = linux.errno(rc);
        if (e != .SUCCESS) return @intCast(@backingInt(e));
        out.* = .{ .mode = sx.mode, .size = sx.size };
        return 0;
    }
    var st: std.c.Stat = undefined;
    const rc = if (follow) std.c.stat(path, &st) else lstat(path, &st);
    if (rc != 0) return std.c._errno().*;
    out.* = .{ .mode = @intCast(st.mode), .size = @intCast(st.size) };
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
    try std.testing.expect(statErrno("/nonexistent/llamazig", true, &info) != 0);
}
