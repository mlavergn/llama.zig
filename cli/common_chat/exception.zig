//! The C++ exceptions `common/chat.cpp` and the server throw, as Zig errors
//! with their `what()` text beside them.
//!
//! # Provenance
//!
//! **Not a port of a file.** Upstream throws `std::invalid_argument`,
//! `std::runtime_error` and `common_json_error` (a `std::runtime_error`),
//! and its callers tell them apart: the server's `ex_wrapper`
//! (server.cpp:55 @620d2225c) answers 400 for the first and, since v0.6.0,
//! for a `common_json_error`, 500 for the rest, and every caller prints
//! `e.what()`. A Zig error carries
//! no payload, so each class is an error and the text lives in a per-thread
//! buffer, as `cli/json/errors.zig` keeps nlohmann's.

const std = @import("std");
const json = @import("json");

/// Every throwing call here returns one of these. `JsonError` is a
/// `common_json_error`, whose text is `json.what()`.
pub const Error = error{ InvalidArgument, RuntimeError, JsonError, OutOfMemory };

threadlocal var what_buf: std.ArrayList(u8) = .empty;

/// `e.what()` for an error this module's callers returned on this thread.
///
/// Parameters:
/// - `e`: the error caught.
///
/// Return: borrowed until the next throw on this thread.
pub fn what(e: anyerror) []const u8 {
    return switch (e) {
        error.JsonError => json.what(),
        error.OutOfMemory => "std::bad_alloc",
        else => what_buf.items,
    };
}

/// `throw std::invalid_argument(...)` or `throw std::runtime_error(...)`:
/// sets the text and returns the error. The text ends at its first NUL, as
/// a `std::exception` built from `std::string` reports it through `c_str()`.
///
/// Parameters:
/// - `e`: `error.InvalidArgument` or `error.RuntimeError`.
/// - `fmt`, `args`: the message, formatted by Zig's `std.fmt`.
///
/// Return: `e`, or `error.OutOfMemory`.
pub fn throw(e: Error, comptime fmt: []const u8, args: anytype) Error {
    const a = std.heap.smp_allocator;
    var next: std.ArrayList(u8) = .empty;
    next.print(a, fmt, args) catch return error.OutOfMemory;
    if (std.mem.indexOfScalar(u8, next.items, 0)) |nul| next.shrinkRetainingCapacity(nul);
    what_buf.deinit(a);
    what_buf = next;
    return e;
}

/// `throw X(prefix + e.what() + suffix)`: rethrows `inner` as `outer` with
/// its text wrapped, as a `catch (const std::exception & e)` that throws a
/// new exception does.
///
/// Return: `outer`, or `error.OutOfMemory` when `inner` was.
pub fn rethrow(outer: Error, inner: anyerror, prefix: []const u8, suffix: []const u8) Error {
    if (inner == error.OutOfMemory) return error.OutOfMemory;
    const a = std.heap.smp_allocator;
    const w = a.dupe(u8, what(inner)) catch return error.OutOfMemory;
    defer a.free(w);
    return throw(outer, "{s}{s}{s}", .{ prefix, w, suffix });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "rethrow wraps the inner text" {
    try std.testing.expectEqual(error.InvalidArgument, throw(error.InvalidArgument, "bad {d}", .{7}));
    try std.testing.expectEqualStrings("bad 7", what(error.InvalidArgument));
    try std.testing.expectEqual(error.RuntimeError, rethrow(error.RuntimeError, error.InvalidArgument, "Failed: ", "!"));
    try std.testing.expectEqualStrings("Failed: bad 7!", what(error.RuntimeError));
}
