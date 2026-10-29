//! The text `ordered_json::dump` writes.
//!
//! # Provenance
//!
//! Ports `nlohmann::detail::serializer` -- `dump`, `dump_escaped`,
//! `dump_integer`, `dump_float`, `hex_bytes` and `decode` -- and
//! `basic_json::dump` from `llama.cpp/vendor/nlohmann/json.hpp`, and
//! `common_json::dump`/`dump_safe` from `llama.cpp/common/json.cpp`
//! (llama.cpp v0.4.1, `b29c606e2`). Each declaration names what it ports and
//! the line it began at. The number text is `dtoa.zig`'s.
//!
//! The C writes strings through a 512-byte buffer it flushes only at a code
//! point boundary, so its "reset to the last accepted byte" is a truncation
//! of what was written since that boundary; `out.items.len` is that
//! position here.

const std = @import("std");
const errors = @import("errors.zig");
const dtoa = @import("dtoa.zig");
const Value = @import("value.zig").Value;

const Error = errors.Error;

/// `error_handler_t`: what `dump` does with a string that is not UTF-8.
pub const ErrorHandler = enum {
    /// throw `type_error.316`
    strict,
    /// replace each invalid sequence with U+FFFD
    replace,
    /// drop each invalid sequence
    ignore,
};

/// `basic_json::dump`'s parameters.
pub const Options = struct {
    /// below 0: compact; 0 and up: one member per line, this many
    /// `indent_char` per level
    indent: i32 = -1,
    indent_char: u8 = ' ',
    /// escape everything from U+007F up as `\uxxxx`
    ensure_ascii: bool = false,
    error_handler: ErrorHandler = .strict,
};

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

/// Ports `hex_bytes` (json.hpp:19398 @b29c606e2): two uppercase digits.
fn hexByte(byte: u8) [2]u8 {
    const nibble_to_hex = "0123456789ABCDEF";
    return .{ nibble_to_hex[byte / 16], nibble_to_hex[byte % 16] };
}

/// `snprintf("\\u%04x")`: a backslash-u and four lowercase hex digits.
fn appendU16(out: *std.ArrayList(u8), a: std.mem.Allocator, cp: u16) !void {
    const hex = "0123456789abcdef";
    try out.appendSlice(a, &.{ '\\', 'u', hex[cp >> 12], hex[(cp >> 8) & 0xF], hex[(cp >> 4) & 0xF], hex[cp & 0xF] });
}

/// Ports `dump_escaped` (json.hpp:19112 @b29c606e2): `s` without its
/// quotes, escaped as JSON, with `\b \t \n \f \r \" \\` short, other
/// controls as `\u00xx`, and -- with `ensure_ascii` -- everything from
/// U+007F up as `\uxxxx`, astral code points as a surrogate pair.
///
/// Parameters:
/// - `out`: appended to; on `error.JsonError` it holds a partial string.
/// - `a`: `out`'s allocator.
/// - `s`: any bytes.
/// - `ensure_ascii`, `handler`: as `Options`.
///
/// Return: `error.JsonError` (`type_error.316`) for invalid UTF-8 under
/// `.strict`; `error.OutOfMemory`.
pub fn dumpEscaped(out: *std.ArrayList(u8), a: std.mem.Allocator, s: []const u8, ensure_ascii: bool, handler: ErrorHandler) Error!void {
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
                        // escape control characters (0x00..0x1F) or, if
                        // ensure_ascii parameter is used, non-ASCII characters
                        if (codepoint <= 0x1F or (ensure_ascii and codepoint >= 0x7F)) {
                            if (codepoint <= 0xFFFF) {
                                try appendU16(out, a, @intCast(codepoint));
                            } else {
                                try appendU16(out, a, @truncate(0xD7C0 + (codepoint >> 10)));
                                try appendU16(out, a, @truncate(0xDC00 + (codepoint & 0x3FF)));
                            }
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
                switch (handler) {
                    .strict => return errors.fail(.type_error, 316, "invalid UTF-8 byte at index {d}: 0x{s}", .{ i, &hexByte(byte) }),
                    .ignore, .replace => {
                        // in case we saw this character the first time, we
                        // would like to read it again, because the byte may
                        // be OK for itself, but just not OK for the previous
                        // sequence
                        if (undumped_chars > 0) i -= 1;

                        // reset length buffer to the last accepted index;
                        // thus removing/ignoring the invalid characters
                        out.shrinkRetainingCapacity(bytes_after_last_accept);

                        if (handler == .replace) {
                            // add a replacement character
                            try out.appendSlice(a, if (ensure_ascii) "\\ufffd" else "\xEF\xBF\xBD");
                            bytes_after_last_accept = out.items.len;
                        }

                        undumped_chars = 0;

                        // continue processing the string
                        state = utf8_accept;
                    },
                }
            },
            else => { // decode found yet incomplete multi-byte code point
                if (!ensure_ascii) {
                    // code point will not be escaped - copy byte to buffer
                    try out.append(a, byte);
                }
                undumped_chars += 1;
            },
        }
    }

    // we finished processing the string
    if (state != utf8_accept) {
        // we finish reading, but do not accept: string was incomplete
        switch (handler) {
            .strict => return errors.fail(.type_error, 316, "incomplete UTF-8 string; last byte: 0x{s}", .{&hexByte(s[s.len - 1])}),
            .ignore => out.shrinkRetainingCapacity(bytes_after_last_accept),
            .replace => {
                out.shrinkRetainingCapacity(bytes_after_last_accept);
                try out.appendSlice(a, if (ensure_ascii) "\\ufffd" else "\xEF\xBF\xBD");
            },
        }
    }
}

/// Ports `dump_float` (json.hpp:19520 @b29c606e2): `null` for NaN and the
/// infinities, `to_chars` otherwise.
fn dumpFloat(out: *std.ArrayList(u8), a: std.mem.Allocator, x: f64) !void {
    if (!std.math.isFinite(x)) return out.appendSlice(a, "null");
    var buf: [dtoa.max_len]u8 = undefined;
    try out.appendSlice(a, dtoa.toChars(&buf, x));
}

/// `indent_string`: the serializer's 512 copies of `indent_char`, which
/// `dump` doubles with `resize(size() * 2, ' ')` when a level needs more --
/// so past column 512 the indent is spaces, whatever `indent_char` is.
const indent_string_len = 512;

/// `o->write_characters(indent_string.c_str(), n)`.
fn indentBy(out: *std.ArrayList(u8), a: std.mem.Allocator, c: u8, n: usize) !void {
    const k = @min(n, indent_string_len);
    try out.appendNTimes(a, c, k);
    try out.appendNTimes(a, ' ', n - k);
}

/// Ports `serializer`'s `dump` (json.hpp:18828 @b29c606e2): one value,
/// recursively. Integers are `dump_integer` (json.hpp:19435 @b29c606e2),
/// which is plain decimal.
fn dumpValue(out: *std.ArrayList(u8), a: std.mem.Allocator, val: *const Value, pretty_print: bool, opts: Options, indent_step: usize, current_indent: usize) Error!void {
    switch (val.*) {
        .object => |m| {
            const ms = m.members.items;
            if (ms.len == 0) return out.appendSlice(a, "{}");
            if (pretty_print) {
                try out.appendSlice(a, "{\n");
                const new_indent = current_indent + indent_step;
                for (ms, 0..) |*mem, i| {
                    try indentBy(out, a, opts.indent_char, new_indent);
                    try out.append(a, '"');
                    try dumpEscaped(out, a, mem.key, opts.ensure_ascii, opts.error_handler);
                    try out.appendSlice(a, "\": ");
                    try dumpValue(out, a, &mem.value, true, opts, indent_step, new_indent);
                    if (i + 1 < ms.len) try out.appendSlice(a, ",\n");
                }
                try out.append(a, '\n');
                try indentBy(out, a, opts.indent_char, current_indent);
                try out.append(a, '}');
            } else {
                try out.append(a, '{');
                for (ms, 0..) |*mem, i| {
                    try out.append(a, '"');
                    try dumpEscaped(out, a, mem.key, opts.ensure_ascii, opts.error_handler);
                    try out.appendSlice(a, "\":");
                    try dumpValue(out, a, &mem.value, false, opts, indent_step, current_indent);
                    if (i + 1 < ms.len) try out.append(a, ',');
                }
                try out.append(a, '}');
            }
        },
        .array => |arr| {
            const xs = arr.items;
            if (xs.len == 0) return out.appendSlice(a, "[]");
            if (pretty_print) {
                try out.appendSlice(a, "[\n");
                const new_indent = current_indent + indent_step;
                for (xs, 0..) |*x, i| {
                    try indentBy(out, a, opts.indent_char, new_indent);
                    try dumpValue(out, a, x, true, opts, indent_step, new_indent);
                    if (i + 1 < xs.len) try out.appendSlice(a, ",\n");
                }
                try out.append(a, '\n');
                try indentBy(out, a, opts.indent_char, current_indent);
                try out.append(a, ']');
            } else {
                try out.append(a, '[');
                for (xs, 0..) |*x, i| {
                    try dumpValue(out, a, x, false, opts, indent_step, current_indent);
                    if (i + 1 < xs.len) try out.append(a, ',');
                }
                try out.append(a, ']');
            }
        },
        .string => |s| {
            try out.append(a, '"');
            try dumpEscaped(out, a, s, opts.ensure_ascii, opts.error_handler);
            try out.append(a, '"');
        },
        .boolean => |b| try out.appendSlice(a, if (b) "true" else "false"),
        .number_integer => |x| try out.print(a, "{d}", .{x}),
        .number_unsigned => |x| try out.print(a, "{d}", .{x}),
        .number_float => |x| try dumpFloat(out, a, x),
        .discarded => try out.appendSlice(a, "<discarded>"),
        .null => try out.appendSlice(a, "null"),
    }
}

/// Ports `basic_json`'s `dump` (json.hpp:21313 @b29c606e2): `val` as text,
/// appended to `out`.
///
/// Parameters:
/// - `out`: appended to; on `error.JsonError` it holds partial text.
/// - `a`: `out`'s allocator.
/// - `val`: the value.
/// - `opts`: indent, indent character, `ensure_ascii`, UTF-8 handling.
///
/// Return: `error.JsonError` (`type_error.316`) for a string that is not
/// UTF-8 under `.strict`; `error.OutOfMemory`.
pub fn dumpTo(out: *std.ArrayList(u8), a: std.mem.Allocator, val: *const Value, opts: Options) Error!void {
    if (opts.indent >= 0) {
        try dumpValue(out, a, val, true, opts, @intCast(opts.indent), 0);
    } else {
        try dumpValue(out, a, val, false, opts, 0, 0);
    }
}

/// Ports `common_json`'s `dump` (json.cpp:339 @b29c606e2): strict about
/// UTF-8, non-ASCII written as is.
///
/// Parameters:
/// - `a`: allocates the text.
/// - `val`: the value.
/// - `indent`: below 0 compact, otherwise spaces per level.
///
/// Return: the text, owned by the caller.
pub fn dump(a: std.mem.Allocator, val: *const Value, indent: i32) Error![]u8 {
    return dumpOptions(a, val, .{ .indent = indent });
}

/// Ports `common_json`'s `dump_safe` (json.cpp:343 @b29c606e2): as `dump`,
/// with invalid UTF-8 replaced by U+FFFD, so it cannot fail but for memory.
///
/// Return: the text, owned by the caller.
pub fn dumpSafe(a: std.mem.Allocator, val: *const Value, indent: i32) error{OutOfMemory}![]u8 {
    return dumpOptions(a, val, .{ .indent = indent, .error_handler = .replace }) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.JsonError => unreachable,
    };
}

/// `basic_json::dump` with every parameter.
///
/// Return: the text, owned by the caller.
pub fn dumpOptions(a: std.mem.Allocator, val: *const Value, opts: Options) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try dumpTo(&out, a, val, opts);
    return out.toOwnedSlice(a);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

fn escaped(s: []const u8, ensure_ascii: bool, handler: ErrorHandler) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(std.testing.allocator);
    try dumpEscaped(&out, std.testing.allocator, s, ensure_ascii, handler);
    return out.toOwnedSlice(std.testing.allocator);
}

test "dump_escaped with ensure_ascii writes surrogate pairs and \\u007f" {
    const got = try escaped("\x7f\xc3\xa9\xf0\x9f\x98\x80", true, .strict);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings("\\u007f\\u00e9\\ud83d\\ude00", got);
}

test "dump_escaped strict reports the byte and its index" {
    try std.testing.expectError(error.JsonError, escaped("ab\xc3(", false, .strict));
    try std.testing.expectEqualStrings("[json.exception.type_error.316] invalid UTF-8 byte at index 3: 0x28", errors.what());
    try std.testing.expectError(error.JsonError, escaped("ab\xe2\x82", false, .strict));
    try std.testing.expectEqualStrings("[json.exception.type_error.316] incomplete UTF-8 string; last byte: 0x82", errors.what());
    const got = try escaped("a\x80b\xe2\x82", true, .ignore);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings("ab", got);
}

test "dump indents nested containers" {
    const a = std.testing.allocator;
    var v = Value.emptyObject();
    defer v.deinit(a);
    try v.set(a, "a", try Value.from(a, 1.5));
    var arr = Value.emptyArray();
    try arr.pushBack(a, .null);
    try arr.pushBack(a, Value.emptyObject());
    try v.set(a, "b", arr);
    const s = try dump(a, &v, 2);
    defer a.free(s);
    try std.testing.expectEqualStrings("{\n  \"a\": 1.5,\n  \"b\": [\n    null,\n    {}\n  ]\n}", s);
    const c = try dump(a, &v, -1);
    defer a.free(c);
    const compact = "{\"a\":1.5,\"b\":[null,{}]}";
    try std.testing.expectEqualStrings(compact, c);
}
