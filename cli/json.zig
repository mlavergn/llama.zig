//! The JSON text `common_json::dump_safe()` writes, for `--log-jsonl`.
//!
//! # Provenance
//!
//! Ports `nlohmann::detail::serializer`'s `dump_escaped` and `decode` from
//! `llama.cpp/vendor/nlohmann/json.hpp`, and `common_json::dump_safe` from
//! `llama.cpp/common/json.cpp` (v0.4.1, `b29c606e2`). Each declaration names
//! what it ports and the line it began at.
//!
//! `dump_safe` is `dump(-1, ' ', false, error_handler_t::replace)` over an
//! `ordered_json`: compact, keys in insertion order, non-ASCII written as is,
//! and an invalid UTF-8 sequence replaced by U+FFFD rather than thrown. The
//! callers build their objects key by key, in the C's order, with
//! `Object`; only the string escaping has rules worth porting.

const std = @import("std");

/// `UTF8_ACCEPT` and `UTF8_REJECT` (json.hpp:18779, 18780 @b29c606e2).
const utf8_accept: u8 = 0;
const utf8_reject: u8 = 1;

/// The table of `decode` (json.hpp:19621 @b29c606e2): a byte's class, then
/// the transitions, Bjoern Hoehrmann's UTF-8 automaton.
const utf8d = [400]u8{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 00..1F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 20..3F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 40..5F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 60..7F
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 80..9F
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, // A0..BF
    8, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // C0..DF
    0xA, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x4, 0x3, 0x3, // E0..EF
    0xB, 0x6, 0x6, 0x6, 0x5, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, // F0..FF
    0x0, 0x1, 0x2, 0x3, 0x5, 0x8, 0x7, 0x1, 0x1, 0x1, 0x4, 0x6, 0x1, 0x1, 0x1, 0x1, // s0..s0
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, // s1..s2
    1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, // s3..s4
    1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, // s5..s6
    1, 3, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // s7..s8
};

/// Ports `decode` (json.hpp:19621 @b29c606e2), a member of `serializer`:
/// one step of the automaton.
fn decode(state: *u8, codep: *u32, byte: u8) u8 {
    const t = utf8d[byte];
    codep.* = if (state.* != utf8_accept)
        (byte & 0x3f) | (codep.* << 6)
    else
        (@as(u32, 0xFF) >> @intCast(t)) & byte;
    state.* = utf8d[256 + @as(usize, state.*) * 16 + t];
    return state.*;
}

/// Ports `dump_escaped` (json.hpp:19112 @b29c606e2), a member of
/// `serializer`, as `dump_safe` calls it: `ensure_ascii` false and
/// `error_handler_t::replace`. Writes `s` without its quotes.
///
/// The C writes through a 512-byte buffer it flushes only at a code-point
/// boundary, so its "reset to the last accepted byte" is a truncation of
/// what was written since; `out.items.len` is that position here.
///
/// Parameters:
/// - `out`: appended to.
/// - `a`: `out`'s allocator.
/// - `s`: the bytes to escape; need not be valid UTF-8.
///
/// Return: `error.OutOfMemory` only.
pub fn writeEscaped(out: *std.ArrayList(u8), a: std.mem.Allocator, s: []const u8) !void {
    var codepoint: u32 = 0;
    var state: u8 = utf8_accept;

    // number of bytes written at the point of the last valid byte
    var bytes_after_last_accept: usize = out.items.len;
    var undumped_chars: usize = 0;

    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        const byte = s[i];

        switch (decode(&state, &codepoint, byte)) {
            utf8_accept => { // decode found a new code point
                switch (codepoint) {
                    0x08 => try out.appendSlice(a, "\\b"),
                    0x09 => try out.appendSlice(a, "\\t"),
                    0x0A => try out.appendSlice(a, "\\n"),
                    0x0C => try out.appendSlice(a, "\\f"),
                    0x0D => try out.appendSlice(a, "\\r"),
                    0x22 => try out.appendSlice(a, "\\\""),
                    0x5C => try out.appendSlice(a, "\\\\"),
                    else => {
                        // escape control characters (0x00..0x1F); with
                        // ensure_ascii off, nothing else
                        if (codepoint <= 0x1F) {
                            // nlohmann's "\\u%04x": two zero digits, then the
                            // control character's two lowercase hex digits
                            const hex = "0123456789abcdef";
                            try out.appendSlice(a, &.{ '\\', 'u', '0', '0', hex[codepoint >> 4], hex[codepoint & 0xF] });
                        } else {
                            // copy byte to buffer (all previous bytes
                            // been copied have in default case above)
                            try out.append(a, byte);
                        }
                    },
                }
                // remember the byte position of this accept
                bytes_after_last_accept = out.items.len;
                undumped_chars = 0;
            },
            utf8_reject => { // decode found invalid UTF-8 byte
                // in case we saw this character the first time, we would
                // like to read it again, because the byte may be OK for
                // itself, but just not OK for the previous sequence
                if (undumped_chars > 0) i -= 1;

                // reset length buffer to the last accepted index; thus
                // removing/ignoring the invalid characters
                out.shrinkRetainingCapacity(bytes_after_last_accept);

                // add a replacement character
                try out.appendSlice(a, "\xEF\xBF\xBD");
                bytes_after_last_accept = out.items.len;

                undumped_chars = 0;

                // continue processing the string
                state = utf8_accept;
            },
            else => { // decode found yet incomplete multi-byte code point
                // code point will not be escaped - copy byte to buffer
                try out.append(a, byte);
                undumped_chars += 1;
            },
        }
    }

    // we finish reading, but do not accept: string was incomplete
    if (state != utf8_accept) {
        // write all accepted bytes, then a replacement character
        out.shrinkRetainingCapacity(bytes_after_last_accept);
        try out.appendSlice(a, "\xEF\xBF\xBD");
    }
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
