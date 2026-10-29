//! `common_json_error`: the error every failing JSON call returns, and the
//! `what()` text it carries.
//!
//! # Provenance
//!
//! Ports `nlohmann::detail::exception::name`, `parse_error::create` and its
//! `position_string` from `llama.cpp/vendor/nlohmann/json.hpp`, and the
//! wrapping of `common_json`'s `guard` from `llama.cpp/common/json.cpp`
//! (llama.cpp v0.4.1, `b29c606e2`). Each declaration names what it ports
//! and the line it began at.
//!
//! # Why the message lives beside the error
//!
//! Upstream throws `common_json_error(e.what())` and its callers print
//! `e.what()` -- `json-schema-to-grammar.cpp` and `chat.cpp` both put it in
//! front of the user. A Zig error carries no payload, so every failing call
//! returns `error.JsonError` and leaves the exact `what()` text in a
//! per-thread buffer that `what()` returns. That buffer is the only global
//! state in `cli/json/`; it lives on `std.heap.smp_allocator` because it
//! outlives any caller's allocator.
//!
//! With `JSON_DIAGNOSTICS` and `JSON_DIAGNOSTIC_POSITIONS` both 0, as
//! upstream builds it, `exception::diagnostics` is always empty and is not
//! ported.

const std = @import("std");

/// Every fallible call in `cli/json/` returns this. `JsonError` is a
/// `common_json_error`; its text is `what()`.
pub const Error = error{ JsonError, OutOfMemory };

/// The exception classes `nlohmann::detail` names in a message.
pub const Kind = enum {
    parse_error,
    invalid_iterator,
    type_error,
    out_of_range,
    other_error,
};

threadlocal var what_buf: std.ArrayList(u8) = .empty;

/// The `what()` of the last `error.JsonError` returned on this thread.
///
/// Return: borrowed; valid until the next failing call on this thread.
pub fn what() []const u8 {
    return what_buf.items;
}

/// Ports `exception`'s `name` (json.hpp:4526 @b29c606e2) and the
/// `type_error::create` family: sets `what()` to
/// `[json.exception.<kind>.<id>] <message>`, cut at its first NUL byte as
/// `std::runtime_error(const char *)` cuts it.
///
/// Parameters:
/// - `kind`, `id`: the exception class and number.
/// - `fmt`, `args`: the message after the prefix.
///
/// Return: `error.JsonError`, or `error.OutOfMemory` if the text could not
/// be kept.
pub fn fail(kind: Kind, id: u16, comptime fmt: []const u8, args: anytype) Error {
    const a = std.heap.smp_allocator;
    what_buf.clearRetainingCapacity();
    what_buf.print(a, "[json.exception.{s}.{d}] ", .{ @tagName(kind), id }) catch return error.OutOfMemory;
    what_buf.print(a, fmt, args) catch return error.OutOfMemory;
    // the text reaches std::runtime_error through `w.c_str()`, so it ends
    // at the first NUL: a key with a NUL in it is cut there
    if (std.mem.indexOfScalar(u8, what_buf.items, 0)) |nul| what_buf.shrinkRetainingCapacity(nul);
    return error.JsonError;
}

/// `position_t`: where the lexer stands, as it counts.
pub const Position = struct {
    /// the total number of characters read
    chars_read_total: usize = 0,
    /// the number of characters read in the current line
    chars_read_current_line: usize = 0,
    /// the number of lines read
    lines_read: usize = 0,
};

/// Ports `parse_error`'s `create` (json.hpp:4639 @b29c606e2) and
/// `position_string` (json.hpp:4670 @b29c606e2): a `parse_error` with its
/// line and column.
///
/// Parameters:
/// - `id`: the exception number (101 for every syntax error).
/// - `pos`: the lexer's position.
/// - `msg`: the message after the position.
///
/// Return: as `fail`.
pub fn parseError(id: u16, pos: Position, msg: []const u8) Error {
    return fail(.parse_error, id, "parse error at line {d}, column {d}: {s}", .{
        pos.lines_read + 1, pos.chars_read_current_line, msg,
    });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "fail writes nlohmann's prefix" {
    try std.testing.expectEqual(error.JsonError, fail(.type_error, 302, "type must be {s}, but is {s}", .{ "string", "number" }));
    try std.testing.expectEqualStrings("[json.exception.type_error.302] type must be string, but is number", what());
    try std.testing.expectEqual(error.JsonError, parseError(101, .{ .chars_read_total = 3, .chars_read_current_line = 2, .lines_read = 1 }, "x"));
    try std.testing.expectEqualStrings("[json.exception.parse_error.101] parse error at line 2, column 2: x", what());
}
