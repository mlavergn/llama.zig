//! The C++ exceptions the PEG parser throws, as Zig errors with their
//! `what()` text beside them, and the two places it writes diagnostics.
//!
//! # Provenance
//!
//! **Not a port of a file.** `llama.cpp/common/peg-parser.cpp` throws
//! `std::runtime_error` (a rule not found, no root, a malformed serialized
//! arena), lets `std::out_of_range` out of `std::vector::at` (a parser or
//! AST id past the end), and passes on `common_json_error` and the schema
//! converter's `std::invalid_argument`. A Zig error carries no payload, so
//! each class is an error and the text lives in a per-thread buffer, as
//! `cli/json/errors.zig` keeps nlohmann's and `cli/json/schema.zig` the
//! converter's.
//!
//! Its debug trace goes to `stderr` with `fprintf`, and one line through
//! `LOG_DBG`; here they go through `stderr_sink` and `log_dbg_sink`, so a
//! caller -- `make peg-diff` -- can capture them.

const std = @import("std");
const json = @import("json");

/// Every fallible call in `cli/peg/`. `OutOfRange` is `std::out_of_range`,
/// `JsonError` a `common_json_error` (its text is `json.what()`).
pub const Error = error{ RuntimeError, InvalidArgument, OutOfRange, JsonError, OutOfMemory };

threadlocal var what_buf: std.ArrayList(u8) = .empty;

/// `e.what()` for an error returned on this thread.
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

/// `throw X(text)`: sets `what()` and returns `e`. The text ends at its
/// first NUL, as a `std::exception` built from a `std::string` reports it.
///
/// Parameters:
/// - `e`: the error to return.
/// - `parts`: concatenated into the text.
///
/// Return: `e`, or `error.OutOfMemory`.
pub fn raise(e: Error, parts: []const []const u8) Error {
    // built apart from `what_buf`, since a part may be `what()` itself
    const a = std.heap.smp_allocator;
    var text: std.ArrayList(u8) = .empty;
    for (parts) |p| text.appendSlice(a, p) catch {
        text.deinit(a);
        return error.OutOfMemory;
    };
    if (std.mem.indexOfScalar(u8, text.items, 0)) |nul| text.shrinkRetainingCapacity(nul);
    what_buf.deinit(a);
    what_buf = text;
    return e;
}

/// `std::vector::at` past the end: libc++ throws `std::out_of_range`
/// reading "vector".
pub fn outOfRange() Error {
    return raise(error.OutOfRange, &.{"vector"});
}

/// A `cli/json/` error as one of ours.
pub fn fromJson(e: json.Error) Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.JsonError => error.JsonError,
    };
}

/// An error from `cli/json/schema.zig` or `schema_to_grammar.zig`, its
/// text -- `json.schema.what()` -- carried over.
pub fn fromSchema(e: json.schema.Error) Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        // the schema module copied `json.what()`, which still holds it
        error.JsonError => error.JsonError,
        error.RuntimeError => raise(error.RuntimeError, &.{json.schema.what()}),
        error.InvalidArgument => raise(error.InvalidArgument, &.{json.schema.what()}),
    };
}

/// Where the debug trace's `fprintf(stderr, ...)` lines go. Defaults to
/// file descriptor 2.
pub var stderr_sink: *const fn (text: []const u8) void = writeStderr;

/// Where `LOG_DBG` lines go; null drops them, as `common_log` does below
/// its default verbosity.
pub var log_dbg_sink: ?*const fn (text: []const u8) void = null;

fn writeStderr(text: []const u8) void {
    var rest = text;
    while (rest.len > 0) {
        const n = writeFd2(rest);
        if (n <= 0) return;
        rest = rest[@intCast(n)..];
    }
}

/// `write(2, bytes, n)`. On Windows `std.c.write` takes a `HANDLE`, so this
/// is the CRT's `_write` on descriptor 2 there, as MinGW's `write` is.
pub fn writeFd2(bytes: []const u8) isize {
    if (comptime @import("builtin").os.tag == .windows) {
        const crt_write = @extern(*const fn (c_int, [*]const u8, c_uint) callconv(.c) c_int, .{ .name = "_write" });
        return crt_write(2, bytes.ptr, @intCast(@min(bytes.len, std.math.maxInt(c_int))));
    }
    return std.c.write(2, bytes.ptr, bytes.len);
}

/// `fprintf(stderr, fmt, args...)`, through `stderr_sink`.
pub fn eprint(comptime fmt: []const u8, args: anytype) Error!void {
    const a = std.heap.smp_allocator;
    const text = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(text);
    stderr_sink(text);
}

/// `LOG_DBG(fmt, args...)`, through `log_dbg_sink`.
pub fn logDbg(comptime fmt: []const u8, args: anytype) Error!void {
    const sink = log_dbg_sink orelse return;
    const a = std.heap.smp_allocator;
    const text = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(text);
    sink(text);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "raise cuts the text at a NUL" {
    try std.testing.expectEqual(error.RuntimeError, raise(error.RuntimeError, &.{ "Rule not found: a", "\x00b" }));
    try std.testing.expectEqualStrings("Rule not found: a", what(error.RuntimeError));
    try std.testing.expectEqual(error.OutOfRange, outOfRange());
    try std.testing.expectEqualStrings("vector", what(error.OutOfRange));
}
