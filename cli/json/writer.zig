//! The JSON text `common_json::dump_safe()` writes, built key by key, for
//! `--log-jsonl`.
//!
//! # Provenance
//!
//! Ports `common_json::dump_safe` from `llama.cpp/common/json.cpp` and the
//! string escaping of `nlohmann::detail::serializer::dump_escaped` from
//! `llama.cpp/vendor/nlohmann/json.hpp` (v0.4.1, `b29c606e2`), as
//! `common/log.cpp` uses them. Each declaration names what it ports and the
//! line it began at.
//!
//! `dump_safe` is `dump(-1, ' ', false, error_handler_t::replace)` over an
//! `ordered_json`: compact, keys in insertion order, non-ASCII written as is,
//! and an invalid UTF-8 sequence replaced by U+FFFD rather than thrown. The
//! callers build their objects key by key, in the C's order, with `Object`,
//! and never hold a `Value`; the escaping is `serializer.zig`'s.

const std = @import("std");
const serializer = @import("serializer.zig");

/// Ports `dump_escaped` (json.hpp:19112 @b29c606e2), a member of
/// `serializer`, as `dump_safe` (json.cpp:343 @b29c606e2) calls it:
/// `ensure_ascii` false and `error_handler_t::replace`. Writes `s` without
/// its quotes.
///
/// Parameters:
/// - `out`: appended to.
/// - `a`: `out`'s allocator.
/// - `s`: the bytes to escape; need not be valid UTF-8.
///
/// Return: `error.OutOfMemory` only.
pub fn writeEscaped(out: *std.ArrayList(u8), a: std.mem.Allocator, s: []const u8) error{OutOfMemory}!void {
    serializer.dumpEscaped(out, a, s, false, .replace) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.JsonError => unreachable,
    };
}

/// A JSON object written as `dump_safe` writes an `ordered_json` one:
/// `{"k":v,...}`, keys in the order they are put.
pub const Object = struct {
    out: *std.ArrayList(u8),
    a: std.mem.Allocator,
    n: usize = 0,

    /// Opens an object at the end of `out`.
    ///
    /// Parameters:
    /// - `out`: appended to; must outlive the `Object`.
    /// - `a`: `out`'s allocator.
    ///
    /// Return: the open object; `error.OutOfMemory` only.
    pub fn begin(out: *std.ArrayList(u8), a: std.mem.Allocator) !Object {
        try out.append(a, '{');
        return .{ .out = out, .a = a };
    }

    fn key(self: *Object, k: []const u8) !void {
        if (self.n != 0) try self.out.append(self.a, ',');
        self.n += 1;
        try self.out.append(self.a, '"');
        try writeEscaped(self.out, self.a, k);
        try self.out.appendSlice(self.a, "\":");
    }

    /// Puts a string member.
    pub fn string(self: *Object, k: []const u8, v: []const u8) !void {
        try self.key(k);
        try self.out.append(self.a, '"');
        try writeEscaped(self.out, self.a, v);
        try self.out.append(self.a, '"');
    }

    /// Puts an integer member, signed or unsigned, as `dump_integer` writes
    /// it: plain decimal.
    pub fn int(self: *Object, k: []const u8, v: anytype) !void {
        try self.key(k);
        try self.out.print(self.a, "{d}", .{v});
    }

    /// Puts a member whose value is already JSON text.
    pub fn raw(self: *Object, k: []const u8, v: []const u8) !void {
        try self.key(k);
        try self.out.appendSlice(self.a, v);
    }

    /// Closes the object.
    pub fn end(self: *Object) !void {
        try self.out.append(self.a, '}');
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

fn escaped(s: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try writeEscaped(&out, std.testing.allocator, s);
    return out.toOwnedSlice(std.testing.allocator);
}

test "dump_escaped escapes controls and keeps non-ASCII" {
    const cases = [_][2][]const u8{
        .{ "a\"b\\c", "a\\\"b\\\\c" },
        .{ "\x08\t\n\x0c\r", "\\b\\t\\n\\f\\r" },
        .{ "\x01\x1f\x7f", "\\u0001\\u001f\x7f" },
        .{ "caf\xc3\xa9 \xe2\x82\xac", "caf\xc3\xa9 \xe2\x82\xac" },
    };
    for (cases) |cs| {
        const got = try escaped(cs[0]);
        defer std.testing.allocator.free(got);
        try std.testing.expectEqualStrings(cs[1], got);
    }
}

test "dump_escaped replaces invalid UTF-8 as nlohmann's replace handler does" {
    const cases = [_][2][]const u8{
        // a lone continuation byte
        .{ "a\x80b", "a\xef\xbf\xbdb" },
        // a truncated sequence before an ASCII byte: the byte is read again
        .{ "a\xc3b", "a\xef\xbf\xbdb" },
        // a truncated sequence at the end
        .{ "a\xe2\x82", "a\xef\xbf\xbd" },
        // an overlong encoding is rejected at its second byte
        .{ "\xc0\xaf", "\xef\xbf\xbd\xef\xbf\xbd" },
        // a surrogate
        .{ "\xed\xa0\x80", "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd" },
        // past U+10FFFF
        .{ "\xf4\x90\x80\x80", "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd" },
        // a complete code point, then a stray continuation byte
        .{ "\xe2\x82\x80\x80x", "\xe2\x82\x80\xef\xbf\xbdx" },
    };
    for (cases) |cs| {
        const got = try escaped(cs[0]);
        defer std.testing.allocator.free(got);
        try std.testing.expectEqualStrings(cs[1], got);
    }
}

test "Object writes compact members in order" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    var o = try Object.begin(&out, std.testing.allocator);
    try o.string("type", "log");
    try o.int("time", @as(i64, 0));
    try o.int("free", @as(usize, 7));
    try o.raw("rows", "[]");
    try o.end();
    try std.testing.expectEqualStrings("{\"type\":\"log\",\"time\":0,\"free\":7,\"rows\":[]}", out.items);
}
