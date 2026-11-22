//! The string operations std lacks: `std::string`'s ordering as a sort
//! comparator, and upstream's `string_replace_all`.
//!
//! # Provenance
//!
//! Ports `string_replace_all`, `string_strip`, `string_remove_suffix` and
//! both `string_split` overloads from `llama.cpp/common/common.cpp` and
//! `common/common.h` (v1.0.0, `b1b770d90`), which the CLI, the tools, the
//! server and libmtmd all call. Each declaration names what it ports and the
//! line it began at. `lessThan` is **not a port**:
//! libc++'s `operator<` on `std::string`, a byte-wise compare then the
//! shorter first. Character classes are `std.ascii` (`isWhitespace` is C's
//! `isspace` in the "C" locale) and a C string's end is
//! `std.mem.sliceTo(s, 0)`.

const std = @import("std");

/// `a < b` for two `std::string`s, as `std.sort`'s comparator.
pub fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// Ports `string_replace_all` (common/common.cpp:528 @b1b770d90): every
/// occurrence of `search`, left to right, never rescanning a replacement;
/// an empty `search` changes nothing.
///
/// Return: a copy, owned by `a`.
pub fn replaceAll(a: std.mem.Allocator, s: []const u8, search: []const u8, replace: []const u8) std.mem.Allocator.Error![]u8 {
    if (search.len == 0) return a.dupe(u8, s);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var last: usize = 0;
    while (std.mem.indexOfPos(u8, s, last, search)) |pos| {
        try out.appendSlice(a, s[last..pos]);
        try out.appendSlice(a, replace);
        last = pos + search.len;
    }
    try out.appendSlice(a, s[last..]);
    return out.toOwnedSlice(a);
}

/// Ports `string_remove_suffix` (common/common.h:887 @b1b770d90), returning the
/// shortened string where the C shortens in place and returns true.
pub fn removeSuffix(str: []const u8, suffix: []const u8) ?[]const u8 {
    if (!std.mem.endsWith(u8, str, suffix)) return null;
    return str[0 .. str.len - suffix.len];
}

/// Ports `string_strip` (common/common.cpp:479 @b1b770d90): `str` without
/// the leading and trailing bytes `std::isspace` accepts in the "C" locale.
///
/// Return: a slice of `str`.
pub fn strip(str: []const u8) []const u8 {
    var start: usize = 0;
    var end: usize = str.len;
    while (start < end and std.ascii.isWhitespace(str[start])) start += 1;
    while (end > start and std.ascii.isWhitespace(str[end - 1])) end -= 1;
    return str[start..end];
}

/// Ports `string_split` (common/common.h:855 @b1b770d90): the parts
/// between each `delim`, including empty ones and the last. The parts
/// borrow `str`; the caller frees the list.
pub fn split(allocator: std.mem.Allocator, str: []const u8, delim: u8) !std.ArrayList([]const u8) {
    var parts: std.ArrayList([]const u8) = .empty;
    errdefer parts.deinit(allocator);
    var it = std.mem.splitScalar(u8, str, delim);
    while (it.next()) |part| try parts.append(allocator, part);
    return parts;
}

/// Ports `string_split` (common/common.cpp:561 @b1b770d90), the overload
/// taking a string delimiter: every piece, empty ones included. The pieces
/// borrow `s`.
///
/// Return: the list, owned by `a`.
pub fn splitString(a: std.mem.Allocator, s: []const u8, delim: []const u8) std.mem.Allocator.Error![][]const u8 {
    var parts: std.ArrayList([]const u8) = .empty;
    errdefer parts.deinit(a);
    var it = std.mem.splitSequence(u8, s, delim);
    while (it.next()) |p| try parts.append(a, p);
    return parts.toOwnedSlice(a);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "replaceAll never rescans a replacement, and an empty search changes nothing" {
    const a = std.testing.allocator;
    const cases = [_][4][]const u8{
        .{ "aaa", "a", "aa", "aaaaaa" }, .{ "abcabc", "bc", "", "aa" },
        .{ "abc", "", "x", "abc" },      .{ "abc", "z", "x", "abc" },
        .{ "", "a", "b", "" },           .{ "{d}{d}", "{d}", "{d}x", "{d}x{d}x" },
    };
    for (cases) |c| {
        const got = try replaceAll(a, c[0], c[1], c[2]);
        defer a.free(got);
        try std.testing.expectEqualStrings(c[3], got);
    }
}

test "lessThan orders as std::string does: bytes, then the shorter first" {
    try std.testing.expect(lessThan({}, "ab", "abc"));
    try std.testing.expect(!lessThan({}, "abc", "ab"));
    try std.testing.expect(lessThan({}, "Z", "a"));
    try std.testing.expect(lessThan({}, "a\x7f", "a\x80"));
    try std.testing.expect(!lessThan({}, "x", "x"));
}

test "strip drops C's white space from both ends, split keeps empty pieces" {
    try std.testing.expectEqualStrings("a b", strip(" \t\na b\x0b\x0c\r"));
    try std.testing.expectEqualStrings("", strip(" \n "));
    try std.testing.expectEqualStrings("x", removeSuffix("x.gguf", ".gguf").?);
    try std.testing.expect(removeSuffix("x", ".gguf") == null);
    var parts = try split(std.testing.allocator, ",a,,b,", ',');
    defer parts.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 5), parts.items.len);
    const pieces = try splitString(std.testing.allocator, "a::b::", "::");
    defer std.testing.allocator.free(pieces);
    try std.testing.expectEqual(@as(usize, 3), pieces.len);
}
