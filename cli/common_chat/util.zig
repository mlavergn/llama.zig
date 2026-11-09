//! Error conversions between `cli/common_chat/`'s layers, and the
//! `std`'s `string` operations the chat code leans on.
//!
//! # Provenance
//!
//! **Not a port of a file.** The conversions stand where the C++ simply
//! lets one exception type propagate through another layer's code; the
//! string helpers port `string_starts_with`, `string_ends_with` and
//! `string_replace_all` from `llama.cpp/common/common.h` and
//! `llama.cpp/common/common.cpp` (llama.cpp v0.8.0, `8642541b2`), each
//! cited where it is defined.

const std = @import("std");
const json = @import("json");
const peg = @import("peg");
const exception = @import("exception.zig");

const Allocator = std.mem.Allocator;
const Error = exception.Error;

/// A `cli/peg` error as a chat one, its `what()` carried over.
/// `std`'s `out_of_range` is a `std`'s `logic_error` that no caller here tells
/// apart from a `std`'s `runtime_error` -- the server's `ex_wrapper` answers
/// 500 for both -- so it becomes one, its text kept.
pub fn fromPeg(e: peg.Error) Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.JsonError => error.JsonError,
        error.InvalidArgument => exception.throw(error.InvalidArgument, "{s}", .{peg.what(e)}),
        error.RuntimeError, error.OutOfRange => exception.throw(error.RuntimeError, "{s}", .{peg.what(e)}),
    };
}

/// A chat error as a `cli/peg` one, for a builder callback that called
/// back into chat code, its `what()` carried over.
pub fn toPeg(e: Error) peg.Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.JsonError => error.JsonError,
        error.InvalidArgument => peg.errors.raise(error.InvalidArgument, &.{exception.what(e)}),
        error.RuntimeError => peg.errors.raise(error.RuntimeError, &.{exception.what(e)}),
    };
}

/// A `cli/json` error as a chat one.
pub fn fromJson(e: json.Error) Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.JsonError => error.JsonError,
    };
}

/// A `cli/json/schema.zig` error as a chat one, its `what()` carried over.
pub fn fromSchema(e: json.schema.Error) Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.JsonError => error.JsonError,
        error.RuntimeError, error.InvalidArgument => |x| exception.throw(x, "{s}", .{json.schema.what()}),
    };
}

/// `std::string::substr(pos, n)`: `std`'s `out_of_range` ("basic_string")
/// past the end, the count clamped.
pub fn substr(s: []const u8, pos: usize, n: usize) Error![]const u8 {
    if (pos > s.len) return exception.throw(error.RuntimeError, "basic_string", .{});
    return s[pos..][0..@min(n, s.len - pos)];
}

/// `std::string::substr(pos)`.
pub fn substrFrom(s: []const u8, pos: usize) Error![]const u8 {
    return substr(s, pos, std.math.maxInt(usize));
}

/// `std::string::npos`.
pub const npos: usize = std.math.maxInt(usize);

/// `std::string::find(needle)`, `npos` when absent; "" is found at 0.
pub fn find(s: []const u8, needle: []const u8) usize {
    return std.mem.indexOf(u8, s, needle) orelse npos;
}

/// `std::string::find(needle, pos)`.
pub fn findFrom(s: []const u8, needle: []const u8, pos: usize) usize {
    if (pos > s.len) return npos;
    return if (std.mem.indexOfPos(u8, s, pos, needle)) |i| i else npos;
}

/// `std::string::rfind(needle)`, `npos` when absent; "" is found at the end.
pub fn rfind(s: []const u8, needle: []const u8) usize {
    return std.mem.lastIndexOf(u8, s, needle) orelse npos;
}

/// `s.find(needle) != std::string::npos`.
pub fn contains(s: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, s, needle) != null;
}

/// Ports `string_replace_all` (common/common.cpp:528 @cd959bae9): every
/// occurrence of `search`, left to right, never rescanning a replacement;
/// an empty `search` changes nothing.
///
/// Return: owned by `a`.
pub fn replaceAll(a: Allocator, s: []const u8, search: []const u8, replace: []const u8) Error![]u8 {
    if (search.len == 0) return a.dupe(u8, s);
    var out: std.ArrayList(u8) = .empty;
    var last: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, s, pos, search)) |at| {
        try out.appendSlice(a, s[last..at]);
        try out.appendSlice(a, replace);
        pos = at + search.len;
        last = pos;
    }
    try out.appendSlice(a, s[last..]);
    return out.toOwnedSlice(a);
}

/// `GGML_ASSERT(cond)`: kept in every build mode, as upstream's is; a
/// failure writes the condition to stderr and aborts, never `@panic` --
/// CLAUDE.md: a panic handler pulls Zig's `compiler_rt` libm in.
pub fn ggmlAssert(ok: bool, comptime cond: []const u8) void {
    if (ok) return;
    const text = "GGML_ASSERT(" ++ cond ++ ") failed\n";
    _ = writeFd2(text);
    std.c.abort();
}

/// `write(2, bytes, n)`. On Windows `std.c.write` takes a `HANDLE`, so this
/// is the CRT's `_write` on descriptor 2 there, as MinGW's `write` is.
fn writeFd2(bytes: []const u8) isize {
    if (comptime @import("builtin").os.tag == .windows) {
        const crt_write = @extern(*const fn (c_int, [*]const u8, c_uint) callconv(.c) c_int, .{ .name = "_write" });
        return crt_write(2, bytes.ptr, @intCast(@min(bytes.len, std.math.maxInt(c_int))));
    }
    return std.c.write(2, bytes.ptr, bytes.len);
}

/// `std`'s `isspace` in the C locale.
pub fn isSpace(c: u8) bool {
    return c == ' ' or (c >= '\t' and c <= '\r');
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "replaceAll does not rescan what it wrote" {
    const a = std.testing.allocator;
    const s = try replaceAll(a, "aaa", "a", "aa");
    defer a.free(s);
    try std.testing.expectEqualStrings("aaaaaa", s);
}
