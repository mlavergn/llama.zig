//! The file system as the port asks it, on `std.Io`, answering in `errno`
//! as the C did: `stat` and `lstat`, the POSIX path calls behind
//! `std::filesystem` -- `mkdir`, `rmdir`, `unlink`, `remove`, `rename`,
//! `symlink`, `readlink`, `realpath`, `getcwd` and `opendir`/`readdir`.
//!
//! # Provenance
//!
//! **Not a port of a file.** Each call is the `std.Io.Dir` operation over
//! the same system call, its error mapped back to the `errno` it came from
//! (`errno.from`), so `strerror` prints what upstream prints; an error no
//! `errno` stands behind reads as `EIO`. A listing is in the order the
//! system returns entries (`getdents64`, `getdirentries`), which is
//! `readdir`'s and so libc++'s `directory_iterator`'s.
//!
//! `stat` answers in `stat`'s terms: the file type bits of `st_mode`, the
//! size, and on failure the `errno`. On Windows the C runtime has no
//! `lstat`, so a symlink is always followed, and MinGW's `stat` strips a
//! trailing separator, short of a root; both are kept.

const std = @import("std");
const mod = @import("module.zig");
const builtin = @import("builtin");

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
///   path   - the path
///   follow - whether a final symlink is followed
///   out    - receives the mode and size on success
/// Return: 0 on success, else the `errno` value `stat` would have set.
pub fn statErrno(path: []const u8, follow: bool, out: *Info) c_int {
    var name: []const u8 = path;
    if (comptime builtin.os.tag == .windows) {
        while (name.len > 1 and (name[name.len - 1] == '/' or name[name.len - 1] == '\\') and name[name.len - 2] != ':') {
            name = name[0 .. name.len - 1];
        }
    }
    const follow_symlinks = follow or builtin.os.tag == .windows;
    const st = std.Io.Dir.cwd().statFile(mod.file.defaultIo(), name, .{ .follow_symlinks = follow_symlinks }) catch |e| {
        return mod.errno.from(e);
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
///   path   - the path
///   follow - whether a final symlink is followed
/// Return: the mode and size, or null when the call failed.
pub fn stat(path: []const u8, follow: bool) ?Info {
    var info: Info = .{};
    return if (statErrno(path, follow, &info) == 0) info else null;
}

/// `mkdir(path, 0777)`.
///
/// Return: 0, or the `errno`.
pub fn mkdir(path: []const u8) c_int {
    std.Io.Dir.cwd().createDir(mod.file.defaultIo(), path, .default_dir) catch |e| return mod.errno.from(e);
    return 0;
}

/// `rmdir(path)`.
///
/// Return: 0, or the `errno`.
pub fn rmdir(path: []const u8) c_int {
    std.Io.Dir.cwd().deleteDir(mod.file.defaultIo(), path) catch |e| return mod.errno.from(e);
    return 0;
}

/// `unlink(path)`. On Darwin, where `unlink` of a directory fails with
/// `EPERM`, std reports it as `IsDir`; that is turned back into `EPERM`.
///
/// Return: 0, or the `errno`.
pub fn unlink(path: []const u8) c_int {
    std.Io.Dir.cwd().deleteFile(mod.file.defaultIo(), path) catch |e| {
        if (e == error.IsDir and comptime builtin.os.tag.isDarwin()) return @backingInt(std.c.E.PERM);
        return mod.errno.from(e);
    };
    return 0;
}

/// C's `remove(path)`: `unlink`, and `rmdir` when it names a directory.
///
/// Return: 0, or the `errno`.
pub fn remove(path: []const u8) c_int {
    std.Io.Dir.cwd().deleteFile(mod.file.defaultIo(), path) catch |e| {
        if (e == error.IsDir) return rmdir(path);
        return mod.errno.from(e);
    };
    return 0;
}

/// `rename(old, new)`.
///
/// Return: 0, or the `errno`.
pub fn rename(old: []const u8, new: []const u8) c_int {
    const cwd = std.Io.Dir.cwd();
    cwd.rename(old, cwd, new, mod.file.defaultIo()) catch |e| return mod.errno.from(e);
    return 0;
}

/// `symlink(target, link)`.
///
/// Return: 0, or the `errno`.
pub fn symlink(target: []const u8, link: []const u8) c_int {
    std.Io.Dir.cwd().symLink(mod.file.defaultIo(), target, link, .{}) catch |e| return mod.errno.from(e);
    return 0;
}

/// `readlink(path)`.
///
/// Parameters:
/// - `buf`: where the target goes.
///
/// Return: the target, in `buf`; null when `path` is not a link or cannot
/// be read.
pub fn readLink(path: []const u8, buf: []u8) ?[]u8 {
    const n = std.Io.Dir.cwd().readLink(mod.file.defaultIo(), path, buf) catch return null;
    return buf[0..n];
}

/// `realpath(path, NULL)`.
///
/// Return: the canonical path, owned by `a`; null when an element does not
/// exist or memory runs out.
pub fn realPathAlloc(a: std.mem.Allocator, path: []const u8) ?[]u8 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = std.Io.Dir.cwd().realPathFile(mod.file.defaultIo(), path, &buf) catch return null;
    return a.dupe(u8, buf[0..n]) catch null;
}

/// `getcwd`.
///
/// Return: the working directory, owned by `a`; null when it cannot be
/// read or memory runs out.
pub fn currentPathAlloc(a: std.mem.Allocator) ?[]u8 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = std.process.currentPath(mod.file.defaultIo(), &buf) catch return null;
    return a.dupe(u8, buf[0..n]) catch null;
}

/// What `std::filesystem::status(p).type()` and `symlink_status` answer,
/// with an error folded into `none`.
pub const FileType = enum { none, regular, directory, symlink, other };

/// `std::filesystem::status(p, ec).type()`, or `symlink_status` when
/// `follow` is false.
pub fn fileType(p: []const u8, follow: bool) FileType {
    const st = stat(p, follow) orelse return .none;
    return switch (st.mode & IFMT) {
        IFREG => .regular,
        IFDIR => .directory,
        IFLNK => .symlink,
        else => .other,
    };
}

/// `std::filesystem::exists(p)`: `stat` succeeds.
pub fn exists(p: []const u8) bool {
    return stat(p, true) != null;
}

/// `std::filesystem::is_directory(p)`, following links.
pub fn isDirectory(p: []const u8) bool {
    const st = stat(p, true) orelse return false;
    return st.mode & IFMT == IFDIR;
}

/// `std::filesystem::is_regular_file(p)`, following links.
pub fn isRegularFile(p: []const u8) bool {
    const st = stat(p, true) orelse return false;
    return st.mode & IFMT == IFREG;
}

/// `std::filesystem::is_symlink(p)`: `lstat` says so.
pub fn isSymlink(p: []const u8) bool {
    const st = stat(p, false) orelse return false;
    return st.mode & IFMT == IFLNK;
}

/// The size `stat` reports, following links.
///
/// Return: null when `stat` fails.
pub fn fileSize(p: []const u8) ?u64 {
    const st = stat(p, true) orelse return null;
    return st.size;
}

/// `stat`'s answer as libc++'s `posix_stat` classifies it.
const Status = enum { unknown, not_found, directory, other };

fn statusOf(p: []const u8, err: *c_int) Status {
    var st: Info = .{};
    const e = statErrno(p, true, &st);
    if (e == 0) return if (st.mode & IFMT == IFDIR) .directory else .other;
    if (e == @backingInt(std.c.E.NOENT) or e == @backingInt(std.c.E.NOTDIR)) return .not_found;
    err.* = e;
    return .unknown;
}

/// `std::filesystem::create_directories(p, ec)`: libc++'s
/// `__create_directories` (`src/filesystem/operations.cpp` in Zig's bundled
/// libc++ 22), `mkdir -p` with mode 0777.
///
/// Return: 0, or the `errno`: `EEXIST` when `p` is not a directory,
/// `ENOTDIR` when a parent is not one.
pub fn createDirectories(p: []const u8) c_int {
    var e: c_int = 0;
    switch (statusOf(p, &e)) {
        .unknown => return e,
        .directory => return 0,
        .other => return @backingInt(std.c.E.EXIST),
        .not_found => {},
    }
    const up = mod.path.parent(p);
    if (up.len != 0) {
        switch (statusOf(up, &e)) {
            .unknown => return e,
            .not_found => {
                if (std.mem.eql(u8, up, p)) return @backingInt(std.c.E.INVAL);
                const r = createDirectories(up);
                if (r != 0) return r;
            },
            .other => return @backingInt(std.c.E.NOTDIR),
            .directory => {},
        }
    }
    // `__create_directory`: `mkdir(p, 0777)`, an existing directory is fine
    const mec = mkdir(p);
    if (mec == 0) return 0;
    if (mec != @backingInt(std.c.E.EXIST)) return mec;
    var ignored: c_int = 0;
    if (statusOf(p, &ignored) != .directory) return mec;
    return 0;
}

/// `std::filesystem::remove(p, ec)`: C's `remove`, a missing path not an
/// error.
///
/// Return: 0, or the `errno` of a failure other than `ENOENT`.
pub fn removeIfExists(p: []const u8) c_int {
    const e = remove(p);
    if (e == @backingInt(std.c.E.NOENT)) return 0;
    return e;
}

/// `std::filesystem::remove_all(p, ec)`: depth first, in `readdir`'s order,
/// never following a link.
///
/// Return: how many entries were removed, and the `errno` that stopped it
/// (0 when nothing did).
pub fn removeAll(a: std.mem.Allocator, p: []const u8) struct { n: u64, err: c_int } {
    const st = stat(p, false) orelse return .{ .n = 0, .err = 0 };
    var n: u64 = 0;
    if (st.mode & IFMT == IFDIR) {
        const names = listNames(a, p) catch return .{ .n = n, .err = @backingInt(std.c.E.NOMEM) };
        defer {
            for (names) |name| a.free(name);
            a.free(names);
        }
        for (names) |name| {
            const child = mod.path.join(a, p, name) catch return .{ .n = n, .err = @backingInt(std.c.E.NOMEM) };
            defer a.free(child);
            const r = removeAll(a, child);
            n += r.n;
            if (r.err != 0) return .{ .n = n, .err = r.err };
        }
        const e = rmdir(p);
        if (e != 0) return .{ .n = n, .err = e };
    } else {
        const e = unlink(p);
        if (e != 0) return .{ .n = n, .err = e };
    }
    return .{ .n = n + 1, .err = 0 };
}

/// A directory's entry names in `readdir`'s order, `.` and `..` aside.
///
/// Return: the names and the list, owned by `a`; empty when the directory
/// cannot be opened or a read fails partway, keeping what came before.
pub fn listNames(a: std.mem.Allocator, p: []const u8) std.mem.Allocator.Error![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (out.items) |name| a.free(name);
        out.deinit(a);
    }
    var dir = switch (Dir.open(p)) {
        .ok => |d| d,
        .err => return out.toOwnedSlice(a),
    };
    defer dir.close();
    while (true) switch (dir.next()) {
        .name => |name| try out.append(a, try a.dupe(u8, name)),
        .end, .err => break,
    };
    return out.toOwnedSlice(a);
}

/// `std::filesystem::copy(from, to, ec)` of one regular file, over `to`.
///
/// Return: 0, or the `errno` of the read or write that failed.
pub fn copyFile(a: std.mem.Allocator, from: []const u8, to: []const u8) c_int {
    const io = mod.file.defaultIo();
    const data = std.Io.Dir.cwd().readFileAlloc(io, from, a, .unlimited) catch |e| return mod.errno.from(e);
    defer a.free(data);
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = to, .data = data }) catch |e| return mod.errno.from(e);
    return 0;
}

/// `std::filesystem::absolute(p)`: `p` under the working directory unless
/// it is absolute; `p` itself when the working directory cannot be read.
///
/// Return: owned by `a`.
pub fn absolute(a: std.mem.Allocator, p: []const u8) std.mem.Allocator.Error![]u8 {
    if (mod.path.isAbsolute(p)) return a.dupe(u8, p);
    const cwd = currentPathAlloc(a) orelse return a.dupe(u8, p);
    defer a.free(cwd);
    return mod.path.join(a, cwd, p);
}

/// `readlink(path)`, its target copied.
///
/// Return: owned by `a`; null when `p` is not a link or memory runs out.
pub fn readLinkAlloc(a: std.mem.Allocator, p: []const u8) ?[]u8 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const target = readLink(p, &buf) orelse return null;
    return a.dupe(u8, target) catch null;
}

/// `std::filesystem::temp_directory_path(ec)`, libc++'s: the first of
/// `TMPDIR`, `TMP`, `TEMP`, `TEMPDIR` that is set, else `/tmp`; it must be
/// a directory.
///
/// Return: the path, valid for the process's life; empty on an error.
pub fn tempDirectoryPath() []const u8 {
    var p: []const u8 = "/tmp";
    for ([_][]const u8{ "TMPDIR", "TMP", "TEMP", "TEMPDIR" }) |name| {
        if (mod.env.get(name)) |v| {
            p = v;
            break;
        }
    }
    if (!isDirectory(p)) return "";
    return p;
}

/// `std::filesystem::equivalent(a, b, ec)`: both exist and are one file --
/// the same device and inode. std's `Stat` carries no device, so this asks
/// `stat` (`statx` on Linux, where `std.c.Stat` is void) itself.
pub fn equivalent(a: [:0]const u8, b: [:0]const u8) bool {
    const std_io = mod.file.defaultIo();
    if (comptime builtin.os.tag == .windows) {
        // Windows compares by volume serial and file index; resolving both
        // paths is the closest the C library offers.
        var ra: [260:0]u8 = undefined;
        var rb: [260:0]u8 = undefined;
        if (mod.libc._fullpath(&ra, a.ptr, ra.len) == null or mod.libc._fullpath(&rb, b.ptr, rb.len) == null) return false;
        const fb = mod.file.openIn(std_io, b) catch return false;
        fb.close(std_io);
        return std.ascii.eqlIgnoreCase(std.mem.sliceTo(&ra, 0), std.mem.sliceTo(&rb, 0));
    }
    if (comptime builtin.os.tag == .linux) {
        // `std.c.Stat` is void on Linux; `statx` gives the same device and
        // inode, following symlinks as `stat` does.
        const linux = std.os.linux;
        var sa: linux.Statx = undefined;
        var sb: linux.Statx = undefined;
        const mask: linux.STATX = .{ .INO = true };
        if (linux.errno(linux.statx(linux.AT.FDCWD, a.ptr, 0, mask, &sa)) != .SUCCESS or
            linux.errno(linux.statx(linux.AT.FDCWD, b.ptr, 0, mask, &sb)) != .SUCCESS) return false;
        return sa.dev_major == sb.dev_major and sa.dev_minor == sb.dev_minor and sa.ino == sb.ino;
    }
    var sa: std.c.Stat = undefined;
    var sb: std.c.Stat = undefined;
    if (std.c.stat(a.ptr, &sa) != 0 or std.c.stat(b.ptr, &sb) != 0) return false;
    return sa.dev == sb.dev and sa.ino == sb.ino;
}

/// An open directory, read as `opendir` and `readdir` read it: `.` and
/// `..` never appear.
pub const Dir = struct {
    dir: std.Io.Dir,
    it: std.Io.Dir.Iterator,

    /// `opendir(path)`, following a final link.
    ///
    /// Return: the directory, or the `errno` it could not be opened with.
    pub fn open(path: []const u8) union(enum) { ok: Dir, err: c_int } {
        const d = std.Io.Dir.cwd().openDir(mod.file.defaultIo(), path, .{ .iterate = true }) catch |e| return .{ .err = mod.errno.from(e) };
        return .{ .ok = .{ .dir = d, .it = d.iterate() } };
    }

    /// `readdir`.
    ///
    /// Return: the next entry's name, valid until the next call; null at
    /// the end; the `errno` of a failed read.
    pub fn next(self: *Dir) union(enum) { name: []const u8, end, err: c_int } {
        const ent = self.it.next(mod.file.defaultIo()) catch |e| return .{ .err = mod.errno.from(e) };
        return if (ent) |en| .{ .name = en.name } else .end;
    }

    /// `closedir`.
    pub fn close(self: *Dir) void {
        self.dir.close(mod.file.defaultIo());
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a tree is made, listed, linked, renamed and removed, with the C's errno" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{&tmp.sub_path});
    defer a.free(root);
    const d = try std.fmt.allocPrint(a, "{s}/d", .{root});
    defer a.free(d);
    const f = try std.fmt.allocPrint(a, "{s}/d/f", .{root});
    defer a.free(f);
    const g = try std.fmt.allocPrint(a, "{s}/d/g", .{root});
    defer a.free(g);
    const l = try std.fmt.allocPrint(a, "{s}/d/l", .{root});
    defer a.free(l);

    try std.testing.expectEqual(@as(c_int, 0), mkdir(d));
    try std.testing.expectEqual(@as(c_int, @backingInt(std.c.E.EXIST)), mkdir(d));
    var out = try mod.file.Out.create(std.testing.io, f);
    out.close(std.testing.io);

    try std.testing.expectEqual(@as(c_int, 0), symlink("f", l));
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("f", readLink(l, &buf) orelse return error.TestUnexpectedResult);
    try std.testing.expect(readLink(f, &buf) == null);
    const real = realPathAlloc(a, l) orelse return error.TestUnexpectedResult;
    defer a.free(real);
    try std.testing.expect(std.mem.endsWith(u8, real, "/d/f"));

    try std.testing.expectEqual(@as(c_int, 0), rename(f, g));
    try std.testing.expectEqual(@as(c_int, @backingInt(std.c.E.NOENT)), rename(f, g));

    var names: [3][]const u8 = undefined;
    var n: usize = 0;
    var dir = switch (Dir.open(d)) {
        .ok => |x| x,
        .err => return error.TestUnexpectedResult,
    };
    while (true) switch (dir.next()) {
        .name => |name| {
            names[n] = try a.dupe(u8, name);
            n += 1;
        },
        .end => break,
        .err => return error.TestUnexpectedResult,
    };
    dir.close();
    defer for (names[0..n]) |name| a.free(name);
    try std.testing.expectEqual(@as(usize, 2), n);

    try std.testing.expectEqual(@as(c_int, @backingInt(std.c.E.NOTEMPTY)), rmdir(d));
    try std.testing.expectEqual(@as(c_int, @backingInt(std.c.E.NOTDIR)), switch (Dir.open(g)) {
        .ok => |x| blk: {
            var y = x;
            y.close();
            break :blk 0;
        },
        .err => |e| e,
    });
    const expected_unlink_dir: std.c.E = if (builtin.os.tag.isDarwin()) .PERM else .ISDIR;
    try std.testing.expectEqual(@as(c_int, @backingInt(expected_unlink_dir)), unlink(d));
    try std.testing.expectEqual(@as(c_int, 0), unlink(l));
    try std.testing.expectEqual(@as(c_int, 0), remove(g));
    try std.testing.expectEqual(@as(c_int, @backingInt(std.c.E.NOENT)), remove(g));
    try std.testing.expectEqual(@as(c_int, 0), remove(d));

    const cwd = currentPathAlloc(a) orelse return error.TestUnexpectedResult;
    defer a.free(cwd);
    try std.testing.expect(cwd.len > 0 and cwd[0] == '/');
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
    const path = try std.fmt.bufPrint(&buf, ".zig-cache/tmp/{s}/f", .{&tmp.sub_path});
    var out = try mod.file.Out.create(io, path);
    try out.write("12345");
    out.close(io);
    const info = stat(path, true) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(IFREG, info.mode & IFMT);
    try std.testing.expectEqual(@as(u64, 5), info.size);
}

test "createDirectories, listNames, copyFile and removeAll answer as libc++ does" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{&tmp.sub_path});
    defer a.free(root);
    const deep = try std.fmt.allocPrint(a, "{s}/x/y/z", .{root});
    defer a.free(deep);
    const f = try std.fmt.allocPrint(a, "{s}/x/f", .{root});
    defer a.free(f);
    const g = try std.fmt.allocPrint(a, "{s}/x/g", .{root});
    defer a.free(g);
    const under_f = try std.fmt.allocPrint(a, "{s}/x/f/sub", .{root});
    defer a.free(under_f);
    const x = try std.fmt.allocPrint(a, "{s}/x", .{root});
    defer a.free(x);

    try std.testing.expectEqual(@as(c_int, 0), createDirectories(deep));
    try std.testing.expectEqual(@as(c_int, 0), createDirectories(deep)); // already there
    try std.testing.expect(isDirectory(deep));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = f, .data = "abc" });
    try std.testing.expectEqual(@as(c_int, @backingInt(std.c.E.EXIST)), createDirectories(f));
    try std.testing.expectEqual(@as(c_int, @backingInt(std.c.E.NOTDIR)), createDirectories(under_f));

    try std.testing.expectEqual(@as(c_int, 0), copyFile(a, f, g));
    try std.testing.expectEqual(@as(?u64, 3), fileSize(g));
    try std.testing.expectEqual(FileType.regular, fileType(g, true));
    try std.testing.expectEqual(FileType.none, fileType(under_f, true));

    const names = try listNames(a, x);
    defer {
        for (names) |n| a.free(n);
        a.free(names);
    }
    try std.testing.expectEqual(@as(usize, 3), names.len); // y, f, g

    // x, y, z, f, g
    const r = removeAll(a, x);
    try std.testing.expectEqual(@as(c_int, 0), r.err);
    try std.testing.expectEqual(@as(u64, 5), r.n);
    try std.testing.expect(!exists(x));
    try std.testing.expectEqual(@as(c_int, 0), removeIfExists(x));
}

test "equivalent: a file is itself, two files are not, a missing one is not" {
    try std.testing.expect(equivalent("build.zig", "./build.zig"));
    try std.testing.expect(!equivalent("build.zig", "build.zig.zon"));
    try std.testing.expect(!equivalent("build.zig", "/nonexistent/x"));
}
