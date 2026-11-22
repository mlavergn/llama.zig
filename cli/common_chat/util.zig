//! Error conversions between `cli/common_chat/`'s layers, and the
//! `std`'s `string` operations the chat code leans on.
//!
//! # Provenance
//!
//! **Not a port of a file.** The conversions stand where the C++ simply
//! lets one exception type propagate through another layer's code; the
//! string helpers port `string_starts_with`, `string_ends_with` and
//! `string_replace_all` from `llama.cpp/common/common.h` and
//! `llama.cpp/common/common.cpp` (llama.cpp v1.0.0, `b1b770d90`), each
//! cited where it is defined.

const std = @import("std");
const mod = @import("module.zig");
const json = @import("json");
const peg = @import("peg");

/// A `cli/peg` error as a chat one, its `what()` carried over.
/// `std`'s `out_of_range` is a `std`'s `logic_error` that no caller here tells
/// apart from a `std`'s `runtime_error` -- the server's `ex_wrapper` answers
/// 500 for both -- so it becomes one, its text kept.
pub fn fromPeg(e: peg.Error) mod.exception.Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.JsonError => error.JsonError,
        error.InvalidArgument => mod.exception.throw(error.InvalidArgument, "{s}", .{peg.what(e)}),
        error.RuntimeError, error.OutOfRange => mod.exception.throw(error.RuntimeError, "{s}", .{peg.what(e)}),
    };
}

/// A chat error as a `cli/peg` one, for a builder callback that called
/// back into chat code, its `what()` carried over.
pub fn toPeg(e: mod.exception.Error) peg.Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.JsonError => error.JsonError,
        error.InvalidArgument => peg.errors.raise(error.InvalidArgument, &.{mod.exception.what(e)}),
        error.RuntimeError => peg.errors.raise(error.RuntimeError, &.{mod.exception.what(e)}),
    };
}

/// A `cli/json` error as a chat one.
pub fn fromJson(e: json.Error) mod.exception.Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.JsonError => error.JsonError,
    };
}

/// A `cli/json/schema.zig` error as a chat one, its `what()` carried over.
pub fn fromSchema(e: json.schema.Error) mod.exception.Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.JsonError => error.JsonError,
        error.RuntimeError, error.InvalidArgument => |x| mod.exception.throw(x, "{s}", .{json.schema.what()}),
    };
}

/// `std::string::substr(pos, n)`: `std`'s `out_of_range` ("basic_string")
/// past the end, the count clamped.
pub fn substr(s: []const u8, pos: usize, n: usize) mod.exception.Error![]const u8 {
    if (pos > s.len) return mod.exception.throw(error.RuntimeError, "basic_string", .{});
    return s[pos..][0..@min(n, s.len - pos)];
}

/// `std::string::substr(pos)`.
pub fn substrFrom(s: []const u8, pos: usize) mod.exception.Error![]const u8 {
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

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
