//! `std::filesystem::path`'s lexical operations on POSIX paths, as libc++
//! implements them: `operator/`, `parent_path`, the elements `iterator`
//! gives, `lexically_normal` and `lexically_relative`.
//!
//! # Provenance
//!
//! **Not a port of a file.** Upstream calls libc++; these follow
//! `src/filesystem/path.cpp` in Zig's bundled libc++ 22, and the tests below
//! are libc++'s own answers, recorded by a C++ program built with
//! `zig c++`. Separators are `/` only: the Windows arms are not reproduced.

const std = @import("std");

const Allocator = std.mem.Allocator;

/// `path::is_absolute()` for a POSIX path.
pub fn isAbsolute(p: []const u8) bool {
    return p.len != 0 and p[0] == '/';
}

/// `p / q`: `q` when it is absolute, else `p` and `q` with one separator
/// between them unless `p` is empty or already ends in one.
///
/// Return: owned by `a`.
pub fn join(a: Allocator, p: []const u8, q: []const u8) Allocator.Error![]u8 {
    if (isAbsolute(q)) return a.dupe(u8, q);
    if (p.len == 0 or p[p.len - 1] == '/') return std.mem.concat(a, u8, &.{ p, q });
    return std.mem.concat(a, u8, &.{ p, "/", q });
}

/// `path::parent_path()`: everything before the last element, its trailing
/// separators dropped; every leading separator is the root, which is kept.
///
/// Return: a slice of `p`.
pub fn parent(p: []const u8) []const u8 {
    // no relative path: the root, or nothing
    if (std.mem.indexOfNone(u8, p, "/") == null) return p;
    if (p[p.len - 1] == '/') {
        // the last element is the empty filename after the separator
        return std.mem.trimEnd(u8, p, "/");
    }
    const cut = std.mem.lastIndexOfScalar(u8, p, '/') orelse return "";
    const head = std.mem.trimEnd(u8, p[0 .. cut + 1], "/");
    if (head.len != 0) return head;
    // the root is every leading separator
    return p[0 .. std.mem.indexOfNone(u8, p, "/") orelse p.len];
}

/// A path's elements as `path::iterator` gives them: `/` for the root
/// directory, then each filename, and an empty element for a trailing
/// separator.
///
/// Return: slices of `p` (and the literal `/`), in a list owned by `a`.
pub fn elements(a: Allocator, p: []const u8) Allocator.Error![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(a);
    if (p.len == 0) return out.toOwnedSlice(a);
    var i: usize = 0;
    if (p[0] == '/') {
        try out.append(a, "/");
        while (i < p.len and p[i] == '/') i += 1;
    }
    while (i < p.len) {
        const start = i;
        while (i < p.len and p[i] != '/') i += 1;
        try out.append(a, p[start..i]);
        if (i < p.len) {
            while (i < p.len and p[i] == '/') i += 1;
            if (i == p.len) try out.append(a, "");
        }
    }
    return out.toOwnedSlice(a);
}

/// `path::lexically_normal()`, libc++'s algorithm: separators collapsed,
/// `.` dropped, `name/..` folded, `..` at the root dropped, a trailing
/// separator kept only after a name, `.` for nothing.
///
/// Return: owned by `a`.
pub fn normal(a: Allocator, p: []const u8) Allocator.Error![]u8 {
    if (p.len == 0) return a.dupe(u8, "");
    const PartKind = enum { root, name, dotdot };
    const Part = struct { s: []const u8, k: PartKind };
    var parts: std.ArrayList(Part) = .empty;
    defer parts.deinit(a);
    var maybe_need_trailing_sep = false;
    var i: usize = 0;
    if (p[0] == '/') {
        try parts.append(a, .{ .s = "/", .k = .root });
        while (i < p.len and p[i] == '/') i += 1;
    }
    while (i < p.len) {
        const start = i;
        while (i < p.len and p[i] != '/') i += 1;
        const name = p[start..i];
        const trailing_sep = blk: {
            var j = i;
            while (j < p.len and p[j] == '/') j += 1;
            const had_sep = j > i;
            i = j;
            break :blk had_sep and j == p.len;
        };
        const last: ?PartKind = if (parts.items.len == 0) null else parts.items[parts.items.len - 1].k;
        if (std.mem.eql(u8, name, ".")) {
            maybe_need_trailing_sep = true;
        } else if (std.mem.eql(u8, name, "..")) {
            if (last == .name) {
                _ = parts.pop();
            } else if (last != .root) {
                try parts.append(a, .{ .s = "..", .k = .dotdot });
            }
            maybe_need_trailing_sep = last == .name;
        } else {
            try parts.append(a, .{ .s = name, .k = .name });
            maybe_need_trailing_sep = false;
        }
        if (trailing_sep) maybe_need_trailing_sep = true;
    }
    if (parts.items.len == 0) return a.dupe(u8, ".");
    const need_trailing_sep = maybe_need_trailing_sep and parts.items[parts.items.len - 1].k == .name;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    for (parts.items) |part| {
        if (out.items.len != 0 and out.items[out.items.len - 1] != '/') try out.append(a, '/');
        try out.appendSlice(a, part.s);
    }
    if (need_trailing_sep) try out.append(a, '/');
    return out.toOwnedSlice(a);
}

/// `path::lexically_relative(base)`: the path that, appended to `base`,
/// names `p`; empty when there is none.
///
/// Return: owned by `a`.
pub fn relative(a: Allocator, p: []const u8, base: []const u8) Allocator.Error![]u8 {
    const pe = try elements(a, p);
    defer a.free(pe);
    const be = try elements(a, base);
    defer a.free(be);
    var i: usize = 0;
    while (i < pe.len and i < be.len and std.mem.eql(u8, pe[i], be[i])) i += 1;
    if (i == pe.len and i == be.len) return a.dupe(u8, ".");
    var n: i64 = 0;
    for (be[i..]) |e| {
        if (std.mem.eql(u8, e, "..")) {
            n -= 1;
        } else if (e.len != 0 and !std.mem.eql(u8, e, ".")) {
            n += 1;
        }
    }
    if (n < 0) return a.dupe(u8, "");
    if (n == 0 and (i == pe.len or pe[i].len == 0)) return a.dupe(u8, ".");
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var k: i64 = 0;
    while (k < n) : (k += 1) {
        if (out.items.len != 0) try out.append(a, '/');
        try out.appendSlice(a, "..");
    }
    for (pe[i..]) |e| {
        if (out.items.len != 0 and out.items[out.items.len - 1] != '/') try out.append(a, '/');
        try out.appendSlice(a, e);
    }
    return out.toOwnedSlice(a);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

const Case = struct { in: []const u8, out: []const u8 };
const Pair = struct { a: []const u8, b: []const u8, out: []const u8 };

// libc++ 22's answers, from `zig c++` over `<filesystem>`.
const normal_cases = [_]Case{
    .{ .in = "", .out = "" },
    .{ .in = ".", .out = "." },
    .{ .in = "..", .out = ".." },
    .{ .in = "/", .out = "/" },
    .{ .in = "//", .out = "/" },
    .{ .in = "///", .out = "/" },
    .{ .in = "a", .out = "a" },
    .{ .in = "a/", .out = "a/" },
    .{ .in = "a//", .out = "a/" },
    .{ .in = "/a", .out = "/a" },
    .{ .in = "//a", .out = "/a" },
    .{ .in = "/a/", .out = "/a/" },
    .{ .in = "a/b", .out = "a/b" },
    .{ .in = "a/b/", .out = "a/b/" },
    .{ .in = "a//b", .out = "a/b" },
    .{ .in = "a/./b", .out = "a/b" },
    .{ .in = "a/../b", .out = "b" },
    .{ .in = "a/b/..", .out = "a/" },
    .{ .in = "a/b/../", .out = "a/" },
    .{ .in = "a/b/.", .out = "a/b/" },
    .{ .in = "a/b/./", .out = "a/b/" },
    .{ .in = "./a", .out = "a" },
    .{ .in = "../a", .out = "../a" },
    .{ .in = "../../a", .out = "../../a" },
    .{ .in = "a/../..", .out = ".." },
    .{ .in = "a/../../b", .out = "../b" },
    .{ .in = "/..", .out = "/" },
    .{ .in = "/../a", .out = "/a" },
    .{ .in = "/a/..", .out = "/" },
    .{ .in = "/a/../..", .out = "/" },
    .{ .in = "/a/./", .out = "/a/" },
    .{ .in = "a/..", .out = "." },
    .{ .in = "a/../", .out = "." },
    .{ .in = "./", .out = "." },
    .{ .in = "../", .out = ".." },
    .{ .in = ".//", .out = "." },
    .{ .in = "..//", .out = ".." },
    .{ .in = "a/.//b/", .out = "a/b/" },
    .{ .in = "/./", .out = "/" },
    .{ .in = "/.", .out = "/" },
    .{ .in = "a/b/c/../../d", .out = "a/d" },
    .{ .in = "foo/./bar/..", .out = "foo/" },
    .{ .in = "foo/.///bar/../", .out = "foo/" },
    .{ .in = "/x/y/../../../z", .out = "/z" },
    .{ .in = "x/../../y/./", .out = "../y/" },
    .{ .in = ".hidden/..", .out = "." },
    .{ .in = "a/...", .out = "a/..." },
    .{ .in = ".../b", .out = ".../b" },
    .{ .in = "a/b/../c/./d/", .out = "a/c/d/" },
};
const parent_cases = [_]Case{
    .{ .in = "", .out = "" },
    .{ .in = ".", .out = "" },
    .{ .in = "..", .out = "" },
    .{ .in = "/", .out = "/" },
    .{ .in = "//", .out = "//" },
    .{ .in = "///", .out = "///" },
    .{ .in = "a", .out = "" },
    .{ .in = "a/", .out = "a" },
    .{ .in = "a//", .out = "a" },
    .{ .in = "/a", .out = "/" },
    .{ .in = "//a", .out = "//" },
    .{ .in = "/a/", .out = "/a" },
    .{ .in = "a/b", .out = "a" },
    .{ .in = "a/b/", .out = "a/b" },
    .{ .in = "a//b", .out = "a" },
    .{ .in = "a/./b", .out = "a/." },
    .{ .in = "a/../b", .out = "a/.." },
    .{ .in = "a/b/..", .out = "a/b" },
    .{ .in = "a/b/../", .out = "a/b/.." },
    .{ .in = "a/b/.", .out = "a/b" },
    .{ .in = "a/b/./", .out = "a/b/." },
    .{ .in = "./a", .out = "." },
    .{ .in = "../a", .out = ".." },
    .{ .in = "../../a", .out = "../.." },
    .{ .in = "a/../..", .out = "a/.." },
    .{ .in = "a/../../b", .out = "a/../.." },
    .{ .in = "/..", .out = "/" },
    .{ .in = "/../a", .out = "/.." },
    .{ .in = "/a/..", .out = "/a" },
    .{ .in = "/a/../..", .out = "/a/.." },
    .{ .in = "/a/./", .out = "/a/." },
    .{ .in = "a/..", .out = "a" },
    .{ .in = "a/../", .out = "a/.." },
    .{ .in = "./", .out = "." },
    .{ .in = "../", .out = ".." },
    .{ .in = ".//", .out = "." },
    .{ .in = "..//", .out = ".." },
    .{ .in = "a/.//b/", .out = "a/.//b" },
    .{ .in = "/./", .out = "/." },
    .{ .in = "/.", .out = "/" },
    .{ .in = "a/b/c/../../d", .out = "a/b/c/../.." },
    .{ .in = "foo/./bar/..", .out = "foo/./bar" },
    .{ .in = "foo/.///bar/../", .out = "foo/.///bar/.." },
    .{ .in = "/x/y/../../../z", .out = "/x/y/../../.." },
    .{ .in = "x/../../y/./", .out = "x/../../y/." },
    .{ .in = ".hidden/..", .out = ".hidden" },
    .{ .in = "a/...", .out = "a" },
    .{ .in = ".../b", .out = "..." },
    .{ .in = "a/b/../c/./d/", .out = "a/b/../c/./d" },
};
const join_cases = [_]Pair{
    .{ .a = "", .b = "", .out = "" },
    .{ .a = "", .b = "a", .out = "a" },
    .{ .a = "a", .b = "", .out = "a/" },
    .{ .a = "a", .b = "b", .out = "a/b" },
    .{ .a = "a/", .b = "b", .out = "a/b" },
    .{ .a = "a", .b = "/b", .out = "/b" },
    .{ .a = "/", .b = "b", .out = "/b" },
    .{ .a = "a//", .b = "b", .out = "a//b" },
    .{ .a = "a", .b = "b/", .out = "a/b/" },
    .{ .a = "/a", .b = "../b", .out = "/a/../b" },
};
const relative_cases = [_]Pair{
    .{ .a = "/a/b", .b = "/a", .out = "b" },
    .{ .a = "/a", .b = "/a/b", .out = ".." },
    .{ .a = "/a/b/c", .b = "/a/d", .out = "../b/c" },
    .{ .a = "/a", .b = "/a", .out = "." },
    .{ .a = "/a/b/", .b = "/a", .out = "b/" },
    .{ .a = "/x/y", .b = "/a/b", .out = "../../x/y" },
    .{ .a = "/a/b", .b = "/a/b/", .out = "." },
    .{ .a = "/", .b = "/a", .out = ".." },
    .{ .a = "/a", .b = "/", .out = "a" },
};

test "the lexical operations answer as libc++ does" {
    const a = std.testing.allocator;
    for (normal_cases) |c| {
        const got = try normal(a, c.in);
        defer a.free(got);
        try std.testing.expectEqualStrings(c.out, got);
    }
    for (parent_cases) |c| try std.testing.expectEqualStrings(c.out, parent(c.in));
    for (join_cases) |c| {
        const got = try join(a, c.a, c.b);
        defer a.free(got);
        try std.testing.expectEqualStrings(c.out, got);
    }
    for (relative_cases) |c| {
        const got = try relative(a, c.a, c.b);
        defer a.free(got);
        try std.testing.expectEqualStrings(c.out, got);
    }
}
