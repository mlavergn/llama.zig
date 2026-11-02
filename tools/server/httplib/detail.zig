//! cpp-httplib's `detail` namespace as the server half reaches it: ASCII
//! classification, the insertion-ordered multimap behind `Headers` and
//! `Params`, field validation, the list splitters, URL coding, the header
//! and query parsers, `Range` and `Accept` parsing, and `status_message`.
//!
//! # Provenance
//!
//! Ports parts of `llama.cpp/vendor/cpp-httplib/httplib.cpp` and
//! `llama.cpp/vendor/cpp-httplib/httplib.h` (cpp-httplib 0.57.1, vendored
//! at v0.5.0, `7fe450e19`), built as `llama.cpp.zmake` builds it: no SSL,
//! no zlib, brotli or zstd, and `build/tools.zig`'s `httplib_defines`. Each
//! declaration names what it ports and the line it began at.
//!
//! # No content codings
//!
//! The reference compiles none of `CPPHTTPLIB_ZLIB_SUPPORT`,
//! `CPPHTTPLIB_BROTLI_SUPPORT` and `CPPHTTPLIB_ZSTD_SUPPORT`, so
//! `encoding_type` can only answer `None` and `create_decompressor` only
//! null. Those are ported as their answers; what the compiled-out arms
//! would do is not.

const std = @import("std");

/// `CPPHTTPLIB_REQUEST_URI_MAX_LENGTH`, as `httplib_defines` sets it.
pub const request_uri_max_length: usize = 32768;
/// `CPPHTTPLIB_HEADER_MAX_LENGTH` (httplib.h:114 @7fe450e19).
pub const header_max_length: usize = 8192;
/// `CPPHTTPLIB_HEADER_MAX_COUNT` (httplib.h:118 @7fe450e19).
pub const header_max_count: usize = 100;
/// `CPPHTTPLIB_MULTIPART_FORM_DATA_FILE_MAX_COUNT` (httplib.h:126 @7fe450e19).
pub const multipart_form_data_file_max_count: usize = 1024;
/// `CPPHTTPLIB_PAYLOAD_MAX_LENGTH` (httplib.h:130 @7fe450e19).
pub const payload_max_length: usize = 100 * 1024 * 1024;
/// `CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH`, as `httplib_defines` sets it.
pub const form_url_encoded_payload_max_length: usize = 1048576;
/// `CPPHTTPLIB_RANGE_MAX_COUNT` (httplib.h:148 @7fe450e19).
pub const range_max_count: usize = 1024;
/// `CPPHTTPLIB_MAX_LINE_LENGTH` (httplib.h:211 @7fe450e19).
pub const max_line_length: usize = 32768;
/// `CPPHTTPLIB_RECV_BUFSIZ` (httplib.h:172 @7fe450e19).
pub const recv_bufsiz: usize = 16384;
/// `CPPHTTPLIB_KEEPALIVE_TIMEOUT_SECOND` (httplib.h:26 @7fe450e19).
pub const keepalive_timeout_second: i64 = 5;
/// `CPPHTTPLIB_KEEPALIVE_TIMEOUT_CHECK_INTERVAL_USECOND` (httplib.h:30 @7fe450e19).
pub const keepalive_timeout_check_interval_usecond: i64 = 10000;
/// `CPPHTTPLIB_KEEPALIVE_MAX_COUNT` (httplib.h:34 @7fe450e19).
pub const keepalive_max_count: usize = 100;
/// `CPPHTTPLIB_SERVER_READ_TIMEOUT_SECOND` (httplib.h:46 @7fe450e19).
pub const server_read_timeout_second: i64 = 5;
/// `CPPHTTPLIB_SERVER_WRITE_TIMEOUT_SECOND` (httplib.h:54 @7fe450e19).
pub const server_write_timeout_second: i64 = 5;
/// `CPPHTTPLIB_THREAD_POOL_IDLE_TIMEOUT` (httplib.h:195 @7fe450e19).
pub const thread_pool_idle_timeout: i64 = 3;
/// `CPPHTTPLIB_LISTEN_BACKLOG`, as `httplib_defines` sets it.
pub const listen_backlog: c_uint = 512;
/// `CPPHTTPLIB_TCP_NODELAY`, as `httplib_defines` sets it.
pub const tcp_nodelay = true;

/// `StatusCode` (httplib.h:958 @7fe450e19): the members the server writes.
pub const StatusCode = struct {
    pub const continue_100: c_int = 100;
    pub const ok_200: c_int = 200;
    pub const partial_content_206: c_int = 206;
    pub const bad_request_400: c_int = 400;
    pub const not_found_404: c_int = 404;
    pub const payload_too_large_413: c_int = 413;
    pub const uri_too_long_414: c_int = 414;
    pub const unsupported_media_type_415: c_int = 415;
    pub const range_not_satisfiable_416: c_int = 416;
    pub const expectation_failed_417: c_int = 417;
    pub const internal_server_error_500: c_int = 500;
};

// -----------------------------------------------------------------------------
// ASCII classification

/// Ports `is_ascii_digit` (httplib.h:612 @7fe450e19).
pub fn isAsciiDigit(c: u8) bool {
    return '0' <= c and c <= '9';
}

/// Ports `is_ascii_alpha` (httplib.h:614 @7fe450e19).
pub fn isAsciiAlpha(c: u8) bool {
    return ('a' <= c and c <= 'z') or ('A' <= c and c <= 'Z');
}

/// Ports `is_ascii_alnum` (httplib.h:618 @7fe450e19).
pub fn isAsciiAlnum(c: u8) bool {
    return isAsciiDigit(c) or isAsciiAlpha(c);
}

/// Ports `to_lower` (httplib.h:624 @7fe450e19), in `case_ignore`: ASCII
/// and the Latin-1 capitals, through the C's table -- 0xC0 to 0xDE but for
/// 0xD7, the multiplication sign.
pub fn toLower(c: u8) u8 {
    return switch (c) {
        'A'...'Z' => c + 32,
        0xC0...0xD6, 0xD8...0xDE => c + 32,
        else => c,
    };
}

/// Ports `equal` (httplib.h:656 @7fe450e19), in `case_ignore`.
pub fn caseIgnoreEqual(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |ca, cb| if (toLower(ca) != toLower(cb)) return false;
    return true;
}

/// Ports `is_space_or_tab` (httplib.cpp:1149 @7fe450e19).
pub fn isSpaceOrTab(c: u8) bool {
    return c == ' ' or c == '\t';
}

/// Ports `is_numeric` (httplib.h:3423 @7fe450e19).
pub fn isNumeric(s: []const u8) bool {
    if (s.len == 0) return false;
    for (s) |c| if (!isAsciiDigit(c)) return false;
    return true;
}

// -----------------------------------------------------------------------------
// The multimap

/// Ports `insertion_ordered_multimap` (httplib.h:1052 @7fe450e19): entries
/// in a flat list, in the order they were inserted, looked up by a linear
/// scan. `case_insensitive` is the C's `KeyEqual`: `Headers` compare names
/// as `case_ignore::equal_to`, `Params` and the form maps as
/// `std::equal_to`.
///
/// Keys and values are borrowed: the owner of the map owns what they point
/// into (a request's arena, in practice).
pub fn OrderedMultimap(comptime V: type, comptime case_insensitive: bool) type {
    return struct {
        const Self = @This();

        /// One `value_type`.
        pub const Entry = struct { key: []const u8, value: V };

        entries: std.ArrayList(Entry) = .empty,

        fn keysEqual(x: []const u8, y: []const u8) bool {
            return if (case_insensitive) caseIgnoreEqual(x, y) else std.mem.eql(u8, x, y);
        }

        /// `emplace`: appends, keeping duplicates.
        ///
        /// Parameters:
        /// - `a`: the list's allocator.
        /// - `key`, `value`: borrowed.
        ///
        /// Return: the new entry's index, or `error.OutOfMemory`.
        pub fn emplace(self: *Self, a: std.mem.Allocator, key: []const u8, value: V) error{OutOfMemory}!usize {
            try self.entries.append(a, .{ .key = key, .value = value });
            return self.entries.items.len - 1;
        }

        /// `index_of`: the first entry whose key equals `key`.
        ///
        /// Return: its index, or null.
        pub fn find(self: *const Self, key: []const u8) ?usize {
            for (self.entries.items, 0..) |e, i| if (keysEqual(e.key, key)) return i;
            return null;
        }

        /// `count`.
        pub fn count(self: *const Self, key: []const u8) usize {
            var n: usize = 0;
            for (self.entries.items) |e| {
                if (keysEqual(e.key, key)) n += 1;
            }
            return n;
        }

        /// The index after `i` among the entries `equal_range(key)` walks,
        /// or null at its end.
        pub fn nextOf(self: *const Self, key: []const u8, i: usize) ?usize {
            var j = i + 1;
            while (j < self.entries.items.len) : (j += 1) {
                if (keysEqual(self.entries.items[j].key, key)) return j;
            }
            return null;
        }

        /// `erase(key)`: removes every entry with that key.
        ///
        /// Return: how many were removed.
        pub fn eraseKey(self: *Self, key: []const u8) usize {
            const before = self.entries.items.len;
            var w: usize = 0;
            for (self.entries.items) |e| {
                if (keysEqual(e.key, key)) continue;
                self.entries.items[w] = e;
                w += 1;
            }
            self.entries.shrinkRetainingCapacity(w);
            return before - w;
        }

        /// `get_multimap_value`: the `id`th value under `key`.
        ///
        /// Return: borrowed, or null where the C returns a default value.
        pub fn get(self: *const Self, key: []const u8, id: usize) ?V {
            var i = self.find(key) orelse return null;
            var k = id;
            while (k > 0) : (k -= 1) i = self.nextOf(key, i) orelse return null;
            return self.entries.items[i].value;
        }

        /// A copy with its own list; keys and values still borrowed.
        ///
        /// Return: the copy, or `error.OutOfMemory`.
        pub fn clone(self: *const Self, a: std.mem.Allocator) error{OutOfMemory}!Self {
            var out: Self = .{};
            try out.entries.appendSlice(a, self.entries.items);
            return out;
        }
    };
}

/// `Headers` (httplib.h:1311 @7fe450e19): names compared case-insensitively.
pub const Headers = OrderedMultimap([]const u8, true);

/// `Params` (httplib.h:1316 @7fe450e19): names compared exactly.
pub const Params = OrderedMultimap([]const u8, false);

// -----------------------------------------------------------------------------
// Field validation

/// Ports `is_token_char` (httplib.cpp:5464 @7fe450e19), in `fields`.
pub fn isTokenChar(c: u8) bool {
    return isAsciiAlnum(c) or switch (c) {
        '!', '#', '$', '%', '&', '\'', '*', '+', '-', '.', '^', '_', '`', '|', '~' => true,
        else => false,
    };
}

/// Ports `is_token` (httplib.cpp:5470 @7fe450e19), in `fields`.
pub fn isToken(s: []const u8) bool {
    if (s.len == 0) return false;
    for (s) |c| if (!isTokenChar(c)) return false;
    return true;
}

/// Ports `is_field_name` (httplib.cpp:5478 @7fe450e19), in `fields`.
pub fn isFieldName(s: []const u8) bool {
    return isToken(s);
}

/// Ports `is_vchar` (httplib.cpp:5480 @7fe450e19), in `fields`: on a signed
/// `char`, so a byte at or above 0x80 is not one.
fn isVchar(c: u8) bool {
    return c >= 33 and c <= 126;
}

/// Ports `is_obs_text` (httplib.cpp:5482 @7fe450e19), in `fields`.
fn isObsText(c: u8) bool {
    return 128 <= c;
}

/// Ports `is_field_vchar` (httplib.cpp:5484 @7fe450e19), in `fields`.
pub fn isFieldVchar(c: u8) bool {
    return isVchar(c) or isObsText(c);
}

/// Ports `is_field_content` (httplib.cpp:5486 @7fe450e19), in `fields`.
fn isFieldContent(s: []const u8) bool {
    if (s.len == 0) return true;
    if (s.len == 1) return isFieldVchar(s[0]);
    if (s.len == 2) return isFieldVchar(s[0]) and isFieldVchar(s[1]);
    var i: usize = 0;
    if (!isFieldVchar(s[i])) return false;
    i += 1;
    while (i < s.len - 1) {
        const c = s[i];
        i += 1;
        if (!(c == ' ' or c == '\t' or isFieldVchar(c))) return false;
    }
    return isFieldVchar(s[i]);
}

/// Ports `is_field_value` (httplib.cpp:5511 @7fe450e19), in `fields`.
pub fn isFieldValue(s: []const u8) bool {
    return isFieldContent(s);
}

/// Ports `is_field_valid` (httplib.cpp:5513 @7fe450e19), in `fields`.
pub fn isFieldValid(name: []const u8, value: []const u8) bool {
    return isFieldName(name) and isFieldValue(value);
}

// -----------------------------------------------------------------------------
// Splitting

/// A half-open span `[first, second)` of a slice.
pub const Span = struct { first: usize, second: usize };

/// Ports `trim` (httplib.cpp:1295 @7fe450e19): narrows `[left, right)` of
/// `b` past spaces and tabs.
pub fn trim(b: []const u8, left_in: usize, right_in: usize) Span {
    var left = left_in;
    var right = right_in;
    while (left < b.len and isSpaceOrTab(b[left])) left += 1;
    while (right > 0 and isSpaceOrTab(b[right - 1])) right -= 1;
    return .{ .first = left, .second = right };
}

/// Ports `trim_copy` (httplib.cpp:1306 @7fe450e19), as a borrowed slice;
/// the C's `substr` of a crossed span is empty here as there.
pub fn trimCopy(s: []const u8) []const u8 {
    const r = trim(s, 0, s.len);
    if (r.first >= r.second) return s[0..0];
    return s[r.first..r.second];
}

/// Ports `trim_double_quotes_copy` (httplib.cpp:1311 @7fe450e19), borrowed.
pub fn trimDoubleQuotesCopy(s: []const u8) []const u8 {
    if (s.len >= 2 and s[0] == '"' and s[s.len - 1] == '"') return s[1 .. s.len - 1];
    return s;
}

/// Ports `divide` (httplib.cpp:1319 @7fe450e19): `data` at its first `d`.
///
/// Return: the halves either side, the second empty when `d` is absent.
pub fn divide(data: []const u8, d: u8) struct { lhs: []const u8, rhs: []const u8 } {
    const it = std.mem.findScalar(u8, data, d) orelse data.len;
    const found: usize = @intFromBool(it != data.len);
    return .{ .lhs = data[0..it], .rhs = data[it + found ..] };
}

/// Ports `split` (httplib.cpp:1344 @7fe450e19): calls `fn_.call(segment)`
/// for each trimmed, non-empty segment of `s` between `d`s, splitting at
/// most `m - 1` times.
///
/// `fn_` is any value with a `call([]const u8) E!void` method, so a caller
/// keeps its state in it as the C's lambda captures.
pub fn split(s: []const u8, d: u8, m: usize, fn_: anytype) !void {
    var i: usize = 0;
    var beg: usize = 0;
    var count: usize = 1;
    while (i < s.len) : (i += 1) {
        if (s[i] == d and count < m) {
            const r = trim(s[0..i], beg, i);
            if (r.first < r.second) try fn_.call(s[r.first..r.second]);
            beg = i + 1;
            count += 1;
        }
    }
    if (i != 0) {
        const r = trim(s, beg, i);
        if (r.first < r.second) try fn_.call(s[r.first..r.second]);
    }
}

/// Ports `split_unquoted` (httplib.cpp:1369 @7fe450e19): `split`, where a
/// delimiter inside a quoted-string is not one.
pub fn splitUnquoted(s: []const u8, d: u8, m: usize, fn_: anytype) !void {
    var i: usize = 0;
    var beg: usize = 0;
    var count: usize = 1;
    var in_quotes = false;
    while (i < s.len) : (i += 1) {
        if (s[i] == '"') {
            in_quotes = !in_quotes;
        } else if (s[i] == d and !in_quotes and count < m) {
            const r = trim(s[0..i], beg, i);
            if (r.first < r.second) try fn_.call(s[r.first..r.second]);
            beg = i + 1;
            count += 1;
        }
    }
    if (i != 0) {
        const r = trim(s, beg, i);
        if (r.first < r.second) try fn_.call(s[r.first..r.second]);
    }
}

/// Ports `split_find` (httplib.cpp:1415 @7fe450e19): `split` that stops at
/// the first segment `pred` accepts.
///
/// Return: whether one did.
pub fn splitFind(s: []const u8, d: u8, m: usize, ctx: anytype, comptime pred: fn (@TypeOf(ctx), []const u8) bool) bool {
    var i: usize = 0;
    var beg: usize = 0;
    var count: usize = 1;
    while (i < s.len) : (i += 1) {
        if (s[i] == d and count < m) {
            const r = trim(s[0..i], beg, i);
            if (r.first < r.second and pred(ctx, s[r.first..r.second])) return true;
            beg = i + 1;
            count += 1;
        }
    }
    if (i != 0) {
        const r = trim(s, beg, i);
        if (r.first < r.second and pred(ctx, s[r.first..r.second])) return true;
    }
    return false;
}

/// Ports `divide_param_pair` (httplib.cpp:1403 @7fe450e19): a header
/// parameter at its first `=`, both sides trimmed.
pub fn divideParamPair(s: []const u8) struct { key: []const u8, val: []const u8 } {
    const d = divide(s, '=');
    const kr = trim(d.lhs, 0, d.lhs.len);
    const vr = trim(d.rhs, 0, d.rhs.len);
    return .{
        .key = if (kr.first < kr.second) d.lhs[kr.first..kr.second] else d.lhs[0..0],
        .val = if (vr.first < vr.second) d.rhs[vr.first..vr.second] else d.rhs[0..0],
    };
}

/// Ports `divide_query_pair` (httplib.cpp:4263 @7fe450e19).
pub fn divideQueryPair(s: []const u8) struct { key: []const u8, val: []const u8 } {
    const d = divide(s, '=');
    return .{ .key = d.lhs, .val = d.rhs };
}

// -----------------------------------------------------------------------------
// Numbers

/// Ports `is_hex` (httplib.cpp:479 @7fe450e19).
///
/// Return: the digit's value, or null.
pub fn isHex(c: u8) ?u8 {
    if (isAsciiDigit(c)) return c - '0';
    if ('A' <= c and c <= 'F') return c - 'A' + 10;
    if ('a' <= c and c <= 'f') return c - 'a' + 10;
    return null;
}

/// Ports `from_hex_to_i` (httplib.cpp:493 @7fe450e19): `cnt` hex digits of
/// `s` from `i`; the C's `s[size()]` is its NUL and fails the same way.
pub fn fromHexToI(s: []const u8, i_in: usize, cnt_in: usize) ?c_int {
    if (i_in >= s.len) return null;
    var val: c_int = 0;
    var i = i_in;
    var cnt = cnt_in;
    while (cnt > 0) : ({
        i += 1;
        cnt -= 1;
    }) {
        if (i >= s.len or s[i] == 0) return null;
        const v = isHex(s[i]) orelse return null;
        val = val * 16 + v;
    }
    return val;
}

/// Ports `from_i_to_hex` (httplib.cpp:510 @7fe450e19), into `buf`.
///
/// Return: the digits, a slice of `buf`.
pub fn fromIToHex(buf: *[16]u8, n_in: usize) []const u8 {
    const charset = "0123456789abcdef";
    var n = n_in;
    var i: usize = buf.len;
    while (true) {
        i -= 1;
        buf[i] = charset[n & 15];
        n >>= 4;
        if (n == 0) break;
    }
    return buf[i..];
}

/// Ports `to_utf8` (httplib.cpp:607 @7fe450e19).
///
/// Return: the bytes written to `buff`, 0 for a surrogate.
pub fn toUtf8(code: c_int, buff: *[4]u8) usize {
    if (code < 0x0080) {
        buff[0] = @intCast(code & 0x7F);
        return 1;
    } else if (code < 0x0800) {
        buff[0] = @intCast(0xC0 | ((code >> 6) & 0x1F));
        buff[1] = @intCast(0x80 | (code & 0x3F));
        return 2;
    } else if (code < 0xD800) {
        buff[0] = @intCast(0xE0 | ((code >> 12) & 0xF));
        buff[1] = @intCast(0x80 | ((code >> 6) & 0x3F));
        buff[2] = @intCast(0x80 | (code & 0x3F));
        return 3;
    } else if (code < 0xE000) {
        return 0;
    } else if (code < 0x10000) {
        buff[0] = @intCast(0xE0 | ((code >> 12) & 0xF));
        buff[1] = @intCast(0x80 | ((code >> 6) & 0x3F));
        buff[2] = @intCast(0x80 | (code & 0x3F));
        return 3;
    } else if (code < 0x110000) {
        buff[0] = @intCast(0xF0 | ((code >> 18) & 0x7));
        buff[1] = @intCast(0x80 | ((code >> 12) & 0x3F));
        buff[2] = @intCast(0x80 | ((code >> 6) & 0x3F));
        buff[3] = @intCast(0x80 | (code & 0x3F));
        return 4;
    }
    return 0;
}

/// What `from_chars` answers.
pub const FromChars = enum { ok, invalid_argument, result_out_of_range };

/// Ports `from_chars` (httplib.h:727 @7fe450e19) for an integer type, base
/// 10: an optional `-`, then digits to the first non-digit.
pub fn fromChars(comptime T: type, s: []const u8, value: *T) FromChars {
    value.* = 0;
    var p: usize = 0;
    var negative = false;
    if (p != s.len and s[p] == '-') {
        negative = true;
        p += 1;
    }
    if (p == s.len) return .invalid_argument;
    const max = std.math.maxInt(T);
    var result: T = 0;
    while (p != s.len) : (p += 1) {
        const c = s[p];
        var digit: i32 = -1;
        if (isAsciiDigit(c)) {
            digit = c - '0';
        } else if ('a' <= c and c <= 'z') {
            digit = c - 'a' + 10;
        } else if ('A' <= c and c <= 'Z') {
            digit = c - 'A' + 10;
        } else break;
        if (digit < 0 or digit >= 10) break;
        const dg: T = @intCast(digit);
        if (result > @divTrunc(max - dg, 10)) return .result_out_of_range;
        result = result * 10 + dg;
    }
    if (p == 0 or (negative and p == 1)) return .invalid_argument;
    value.* = if (negative) 0 -% result else result;
    return .ok;
}

/// Ports `from_chars` (httplib.h:781 @7fe450e19) for `double`:
/// `1*DIGIT [ "." *DIGIT ]`.
///
/// Return: the value and how many bytes it took, or null.
pub fn fromCharsF64(s: []const u8) ?struct { value: f64, end: usize } {
    const powers_of_ten = [_]f64{ 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18 };
    const max_frac_digits: usize = powers_of_ten.len - 1;
    var mantissa: u64 = 0;
    var frac_digits: usize = 0;
    var seen_digit = false;
    const limit: u64 = (std.math.maxInt(u64) - 9) / 10;
    var p: usize = 0;
    while (p != s.len and isAsciiDigit(s[p])) : (p += 1) {
        seen_digit = true;
        if (mantissa <= limit) mantissa = mantissa * 10 + (s[p] - '0');
    }
    if (p != s.len and s[p] == '.') {
        p += 1;
        while (p != s.len and isAsciiDigit(s[p])) : (p += 1) {
            seen_digit = true;
            if (frac_digits < max_frac_digits and mantissa <= limit) {
                mantissa = mantissa * 10 + (s[p] - '0');
                frac_digits += 1;
            }
        }
    }
    if (!seen_digit) return null;
    return .{ .value = @as(f64, @floatFromInt(mantissa)) / powers_of_ten[frac_digits], .end = p };
}

// -----------------------------------------------------------------------------
// URL coding

/// Ports `decode_path_component` (httplib.cpp:6581 @7fe450e19): `%XX`, and
/// `%uXXXX` as UTF-8.
///
/// Return: allocated by `a`.
pub fn decodePathComponent(a: std.mem.Allocator, component: []const u8) error{OutOfMemory}![]u8 {
    var result: std.ArrayList(u8) = .empty;
    try result.ensureTotalCapacity(a, component.len);
    var i: usize = 0;
    while (i < component.len) : (i += 1) {
        if (component[i] == '%' and i + 1 < component.len) {
            if (component[i + 1] == 'u') {
                if (fromHexToI(component, i + 2, 4)) |val| {
                    var buff: [4]u8 = undefined;
                    const len = toUtf8(val, &buff);
                    if (len > 0) try result.appendSlice(a, buff[0..len]);
                    i += 5;
                } else try result.append(a, component[i]);
            } else {
                if (fromHexToI(component, i + 1, 2)) |val| {
                    try result.append(a, @intCast(val));
                    i += 2;
                } else try result.append(a, component[i]);
            }
        } else try result.append(a, component[i]);
    }
    return result.toOwnedSlice(a);
}

/// Ports `encode_query_component` (httplib.cpp:6618 @7fe450e19), with
/// `space_as_plus` at its default, `true`.
///
/// Return: allocated by `a`.
pub fn encodeQueryComponent(a: std.mem.Allocator, component: []const u8) error{OutOfMemory}![]u8 {
    var result: std.ArrayList(u8) = .empty;
    try result.ensureTotalCapacity(a, component.len * 3);
    for (component) |c| {
        if (isAsciiAlnum(c) or c == '-' or c == '.' or c == '_' or c == '~') {
            try result.append(a, c);
        } else if (c == ' ') {
            try result.append(a, '+');
        } else if (c == '+') {
            try result.appendSlice(a, "%2B");
        } else if (c == '!' or c == '$' or c == '\'' or c == '(' or c == ')' or c == '*' or c == ',' or c == ';') {
            try result.append(a, c);
        } else if (c == ':' or c == '@' or c == '/' or c == '?') {
            try result.append(a, c);
        } else {
            const hex = "0123456789ABCDEF";
            try result.appendSlice(a, &.{ '%', hex[c >> 4], hex[c & 15] });
        }
    }
    return result.toOwnedSlice(a);
}

/// Ports `decode_query_component` (httplib.cpp:6673 @7fe450e19), with
/// `plus_as_space` at its default, `true`.
///
/// Return: allocated by `a`.
pub fn decodeQueryComponent(a: std.mem.Allocator, component: []const u8) error{OutOfMemory}![]u8 {
    var result: std.ArrayList(u8) = .empty;
    try result.ensureTotalCapacity(a, component.len);
    var i: usize = 0;
    while (i < component.len) : (i += 1) {
        if (component[i] == '%' and i + 2 < component.len) {
            if (fromHexToI(component, i + 1, 2)) |val| {
                try result.append(a, @intCast(val));
                i += 2;
            } else try result.append(a, component[i]);
        } else if (component[i] == '+') {
            try result.append(a, ' ');
        } else try result.append(a, component[i]);
    }
    return result.toOwnedSlice(a);
}

/// Ports `parse_query_text` (httplib.cpp:4273 @7fe450e19): `key=value`
/// pairs between `&`s into `params`, decoded, an exact repeat of an earlier
/// pair skipped and an empty key dropped. The second overload
/// (httplib.cpp:4291 @7fe450e19) is this one.
pub fn parseQueryText(a: std.mem.Allocator, data: []const u8, params: *Params) error{OutOfMemory}!void {
    const Ctx = struct {
        a: std.mem.Allocator,
        params: *Params,
        cache: std.ArrayList([]const u8) = .empty,
        pub fn call(self: *@This(), kv: []const u8) error{OutOfMemory}!void {
            for (self.cache.items) |seen| if (std.mem.eql(u8, seen, kv)) return;
            try self.cache.append(self.a, kv);
            const p = divideQueryPair(kv);
            if (p.key.len != 0) {
                _ = try self.params.emplace(self.a, try decodeQueryComponent(self.a, p.key), try decodeQueryComponent(self.a, p.val));
            }
        }
    };
    var ctx: Ctx = .{ .a = a, .params = params };
    defer ctx.cache.deinit(a);
    try split(data, '&', std.math.maxInt(usize), &ctx);
}

// -----------------------------------------------------------------------------
// Media types and parameters

/// One `key=value` of a media type's parameters, in `std::map` order.
pub const MediaParams = std.ArrayList(Params.Entry);

/// Ports `extract_media_type` (httplib.cpp:2824 @7fe450e19): the type and
/// subtype before the first `;`, trimmed, and -- when `params` is given --
/// the parameters after it, `std::map::emplace`d: the first of a repeated
/// key wins.
///
/// Return: borrowed from `content_type`.
pub fn extractMediaType(a: std.mem.Allocator, content_type: []const u8, params: ?*MediaParams) error{OutOfMemory}![]const u8 {
    var media_type = content_type;
    if (std.mem.findScalar(u8, content_type, ';')) |semi| {
        const param_str = content_type[semi + 1 ..];
        media_type = content_type[0..semi];
        if (params) |ps| {
            const Ctx = struct {
                a: std.mem.Allocator,
                ps: *MediaParams,
                pub fn call(self: *@This(), seg: []const u8) error{OutOfMemory}!void {
                    const kv = divideParamPair(seg);
                    if (kv.key.len == 0) return;
                    const key = trimCopy(kv.key);
                    for (self.ps.items) |e| if (std.mem.eql(u8, e.key, key)) return;
                    try self.ps.append(self.a, .{ .key = key, .value = trimDoubleQuotesCopy(kv.val) });
                }
            };
            var ctx: Ctx = .{ .a = a, .ps = ps };
            try splitUnquoted(param_str, ';', std.math.maxInt(usize), &ctx);
        }
    }
    return trimCopy(media_type);
}

/// Ports `parse_multipart_boundary` (httplib.cpp:4362 @7fe450e19).
///
/// Return: the boundary, borrowed from `content_type`, or null.
pub fn parseMultipartBoundary(a: std.mem.Allocator, content_type: []const u8) error{OutOfMemory}!?[]const u8 {
    var params: MediaParams = .empty;
    defer params.deinit(a);
    _ = try extractMediaType(a, content_type, &params);
    for (params.items) |e| {
        if (std.mem.eql(u8, e.key, "boundary")) {
            if (e.value.len != 0 and e.value.len <= 70) return e.value;
            return null;
        }
    }
    return null;
}

/// Ports `parse_disposition_params` (httplib.cpp:4376 @7fe450e19): a
/// `Content-Disposition`'s parameters into `params`, an exact repeat of an
/// earlier one skipped.
pub fn parseDispositionParams(a: std.mem.Allocator, s: []const u8, params: *Params) error{OutOfMemory}!void {
    const Ctx = struct {
        a: std.mem.Allocator,
        params: *Params,
        cache: std.ArrayList([]const u8) = .empty,
        pub fn call(self: *@This(), kv: []const u8) error{OutOfMemory}!void {
            for (self.cache.items) |seen| if (std.mem.eql(u8, seen, kv)) return;
            try self.cache.append(self.a, kv);
            const p = divideParamPair(kv);
            if (p.key.len != 0) _ = try self.params.emplace(self.a, trimDoubleQuotesCopy(p.key), trimDoubleQuotesCopy(p.val));
        }
    };
    var ctx: Ctx = .{ .a = a, .params = params };
    defer ctx.cache.deinit(a);
    try splitUnquoted(s, ';', std.math.maxInt(usize), &ctx);
}

/// Ports `parse_quality` (httplib.cpp:2878 @7fe450e19): a list element's
/// token and its `q=` weight.
///
/// Return: null where the C returns false.
pub fn parseQuality(s: []const u8) ?struct { token: []const u8, quality: f64 } {
    var quality: f64 = 1.0;
    const d = divide(s, ';');
    const r = trim(d.lhs, 0, d.lhs.len);
    const token = if (r.first < r.second) d.lhs[r.first..r.second] else d.lhs[0..0];
    if (token.len == 0) return null;
    if (d.rhs.len == 0) return .{ .token = token, .quality = quality };

    const Scan = struct {
        invalid: bool = false,
        quality: *f64,
        fn pred(self: *@This(), pb: []const u8) bool {
            if (pb.len < 2) return false;
            if ((pb[0] != 'q' and pb[0] != 'Q') or pb[1] != '=') return false;
            const vr = trim(pb, 2, pb.len);
            if (vr.first >= vr.second) {
                self.invalid = true;
                return true;
            }
            // `from_chars` succeeds on a leading number and leaves the rest
            const v = fromCharsF64(pb[vr.first..vr.second]) orelse {
                self.invalid = true;
                return true;
            };
            if (v.value < 0.0 or v.value > 1.0) {
                self.invalid = true;
                return true;
            }
            self.quality.* = v.value;
            return true;
        }
    };
    var scan: Scan = .{ .quality = &quality };
    _ = splitFind(d.rhs, ';', std.math.maxInt(usize), &scan, Scan.pred);
    if (scan.invalid) return null;
    return .{ .token = token, .quality = quality };
}

/// Ports `parse_accept_header` (httplib.cpp:4462 @7fe450e19), as far as its
/// answer: whether every element is a valid media range. The server never
/// reads `accept_content_types`, the order the C sorts it into, so the list
/// is not kept.
pub fn parseAcceptHeader(a: std.mem.Allocator, s: []const u8) error{OutOfMemory}!bool {
    if (s.len == 0) return true;
    const Ctx = struct {
        a: std.mem.Allocator,
        has_invalid_entry: bool = false,
        pub fn call(self: *@This(), seg: []const u8) error{OutOfMemory}!void {
            const entry = trimCopy(seg);
            const q = parseQuality(entry) orelse {
                self.has_invalid_entry = true;
                return;
            };
            const media_type = try extractMediaType(self.a, q.token, null);
            if (media_type.len == 0) {
                self.has_invalid_entry = true;
                return;
            }
            if (!std.mem.eql(u8, media_type, "*") and std.mem.findScalar(u8, media_type, '/') == null) {
                self.has_invalid_entry = true;
                return;
            }
        }
    };
    var ctx: Ctx = .{ .a = a };
    try split(s, ',', std.math.maxInt(usize), &ctx);
    return !ctx.has_invalid_entry;
}

/// `Range`: `first` and `last`, -1 where absent.
pub const Range = struct { first: isize, second: isize };

/// Ports `parse_range_header` (httplib.cpp:4398 @7fe450e19): `bytes=` and
/// one or more `first-last` specs.
///
/// Return: whether it parsed; `ranges` holds what it read either way.
pub fn parseRangeHeader(a: std.mem.Allocator, s: []const u8, ranges: *std.ArrayList(Range)) error{OutOfMemory}!bool {
    if (!(s.len > 7 and std.mem.eql(u8, s[0..6], "bytes="))) return false;
    const Ctx = struct {
        a: std.mem.Allocator,
        ranges: *std.ArrayList(Range),
        all_valid_ranges: bool = true,
        fn allDigits(x: []const u8) bool {
            for (x) |ch| if (!isAsciiDigit(ch)) return false;
            return true;
        }
        pub fn call(self: *@This(), seg: []const u8) error{OutOfMemory}!void {
            if (!self.all_valid_ranges) return;
            const it = std.mem.findScalar(u8, seg, '-') orelse {
                self.all_valid_ranges = false;
                return;
            };
            const lhs = seg[0..it];
            const rhs = seg[it + 1 ..];
            if (!allDigits(lhs) or !allDigits(rhs)) {
                self.all_valid_ranges = false;
                return;
            }
            var first: isize = -1;
            if (lhs.len != 0) {
                if (fromChars(isize, lhs, &first) != .ok) {
                    self.all_valid_ranges = false;
                    return;
                }
            }
            var last: isize = -1;
            if (rhs.len != 0) {
                var v: isize = 0;
                if (fromChars(isize, rhs, &v) == .ok) last = v;
            }
            if ((first == -1 and last == -1) or (first != -1 and last != -1 and first > last)) {
                self.all_valid_ranges = false;
                return;
            }
            try self.ranges.append(self.a, .{ .first = first, .second = last });
        }
    };
    var ctx: Ctx = .{ .a = a, .ranges = ranges };
    try split(s[6..], ',', std.math.maxInt(usize), &ctx);
    return ctx.all_valid_ranges and ranges.items.len != 0;
}

// -----------------------------------------------------------------------------
// Headers

/// Ports `str2tag` (httplib.cpp:2736 @7fe450e19).
fn str2tag(s: []const u8) u32 {
    var h: u32 = 0;
    for (s) |c| h = ((std.math.maxInt(u32) >> 6) & (h *% 33)) ^ c;
    return h;
}

/// Ports `is_prohibited_header_name` (httplib.cpp:3352 @7fe450e19): by its
/// hash, as the C's `switch` compares.
pub fn isProhibitedHeaderName(name: []const u8) bool {
    const tag = str2tag(name);
    inline for (.{ "REMOTE_ADDR", "REMOTE_PORT", "LOCAL_ADDR", "LOCAL_PORT" }) |n| {
        if (tag == comptime str2tag(n)) return true;
    }
    return false;
}

/// Ports `has_header` (httplib.cpp:3364 @7fe450e19), what `Request`'s
/// `has_header` calls.
pub fn hasHeader(headers: *const Headers, key: []const u8) bool {
    if (isProhibitedHeaderName(key)) return false;
    return headers.find(key) != null;
}

/// Ports `get_header_value` (httplib.cpp:3369 @7fe450e19) with `def` "" and
/// `id` 0. A prohibited name throws in the C; the server asks for none.
pub fn getHeaderValue(headers: *const Headers, key: []const u8) []const u8 {
    if (isProhibitedHeaderName(key)) return "";
    return headers.get(key, 0) orelse "";
}

/// Ports `get_header_value_u64` (httplib.h:3427 @7fe450e19) with `id` 0.
pub fn getHeaderValueU64(headers: *const Headers, key: []const u8, def: usize, is_invalid_value: *bool) usize {
    is_invalid_value.* = false;
    const v = headers.get(key, 0) orelse return def;
    if (isNumeric(v)) {
        var val: usize = 0;
        if (fromChars(usize, v, &val) == .result_out_of_range) {
            is_invalid_value.* = true;
            return std.math.maxInt(usize);
        }
        return val;
    }
    is_invalid_value.* = true;
    return def;
}

/// Ports `get_combined_header_value` (httplib.cpp:3398 @7fe450e19): every
/// non-empty field line under `key`, joined by `", "`.
///
/// Return: allocated by `a`.
pub fn getCombinedHeaderValue(a: std.mem.Allocator, headers: *const Headers, key: []const u8) error{OutOfMemory}![]u8 {
    var combined: std.ArrayList(u8) = .empty;
    var it = headers.find(key);
    while (it) |i| : (it = headers.nextOf(key, i)) {
        const v = headers.entries.items[i].value;
        if (v.len == 0) continue;
        if (combined.items.len != 0) try combined.appendSlice(a, ", ");
        try combined.appendSlice(a, v);
    }
    return combined.toOwnedSlice(a);
}

/// Ports `has_header_token` (httplib.cpp:3413 @7fe450e19): whether any line
/// under `key` lists `token`, case-insensitively.
pub fn hasHeaderToken(headers: *const Headers, key: []const u8, token: []const u8) bool {
    const S = struct {
        fn pred(t: []const u8, seg: []const u8) bool {
            return caseIgnoreEqual(seg, t);
        }
    };
    var it = headers.find(key);
    while (it) |i| : (it = headers.nextOf(key, i)) {
        if (splitFind(headers.entries.items[i].value, ',', std.math.maxInt(usize), token, S.pred)) return true;
    }
    return false;
}

/// Ports `set_header` (httplib.cpp:3442 @7fe450e19): adds the field when
/// its name and value are valid, and silently drops it otherwise.
pub fn setHeader(a: std.mem.Allocator, headers: *Headers, key: []const u8, val: []const u8) error{OutOfMemory}!void {
    if (isFieldValid(key, val)) _ = try headers.emplace(a, key, val);
}

/// Ports `is_chunked_transfer_encoding` (httplib.cpp:3697 @7fe450e19): whether
/// the final coding of the combined `Transfer-Encoding` is `chunked`.
pub fn isChunkedTransferEncoding(headers: *const Headers) bool {
    var it = headers.find("Transfer-Encoding") orelse return false;
    var last_coding: []const u8 = "";
    const Ctx = struct {
        last: *[]const u8,
        pub fn call(self: *@This(), seg: []const u8) error{}!void {
            self.last.* = seg;
        }
    };
    while (true) {
        last_coding = "";
        var ctx: Ctx = .{ .last = &last_coding };
        split(headers.entries.items[it].value, ',', std.math.maxInt(usize), &ctx) catch |err| switch (err) {};
        it = headers.nextOf("Transfer-Encoding", it) orelse break;
    }
    return caseIgnoreEqual(last_coding, "chunked");
}

/// Ports `has_conflicting_content_length` (httplib.cpp:3729 @7fe450e19).
pub fn hasConflictingContentLength(headers: *const Headers) bool {
    var invalid = false;
    return hasHeader(headers, "Transfer-Encoding") and getHeaderValueU64(headers, "Content-Length", 0, &invalid) > 0;
}

/// Ports `is_known_content_encoding` (httplib.cpp:3311 @7fe450e19), and
/// with it `is_zlib_encoding`, `is_brotli_encoding` and `is_zstd_encoding`
/// (httplib.cpp:3296, 3301, 3305 @7fe450e19).
pub fn isKnownContentEncoding(encoding: []const u8) bool {
    return caseIgnoreEqual(encoding, "gzip") or caseIgnoreEqual(encoding, "deflate") or
        caseIgnoreEqual(encoding, "br") or caseIgnoreEqual(encoding, "zstd");
}

/// Ports `is_websocket_upgrade` (httplib.cpp:818 @7fe450e19).
pub fn isWebsocketUpgrade(method: []const u8, headers: *const Headers) bool {
    if (!std.mem.eql(u8, method, "GET")) return false;
    if (!hasHeaderToken(headers, "Upgrade", "websocket")) return false;
    if (!hasHeaderToken(headers, "Connection", "upgrade")) return false;
    const ws_key = getHeaderValue(headers, "Sec-WebSocket-Key");
    if (ws_key.len != 24 or ws_key[22] != '=' or ws_key[23] != '=') return false;
    const b64chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (ws_key[0..22]) |ch| if (std.mem.findScalar(u8, b64chars, ch) == null) return false;
    return std.mem.eql(u8, getHeaderValue(headers, "Sec-WebSocket-Version"), "13");
}

/// Ports `status_message` (httplib.cpp:6283 @7fe450e19).
pub fn statusMessage(status: c_int) []const u8 {
    return switch (status) {
        100 => "Continue",
        101 => "Switching Protocol",
        102 => "Processing",
        103 => "Early Hints",
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        203 => "Non-Authoritative Information",
        204 => "No Content",
        205 => "Reset Content",
        206 => "Partial Content",
        207 => "Multi-Status",
        208 => "Already Reported",
        226 => "IM Used",
        300 => "Multiple Choices",
        301 => "Moved Permanently",
        302 => "Found",
        303 => "See Other",
        304 => "Not Modified",
        305 => "Use Proxy",
        306 => "unused",
        307 => "Temporary Redirect",
        308 => "Permanent Redirect",
        400 => "Bad Request",
        401 => "Unauthorized",
        402 => "Payment Required",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        406 => "Not Acceptable",
        407 => "Proxy Authentication Required",
        408 => "Request Timeout",
        409 => "Conflict",
        410 => "Gone",
        411 => "Length Required",
        412 => "Precondition Failed",
        413 => "Payload Too Large",
        414 => "URI Too Long",
        415 => "Unsupported Media Type",
        416 => "Range Not Satisfiable",
        417 => "Expectation Failed",
        418 => "I'm a teapot",
        421 => "Misdirected Request",
        422 => "Unprocessable Content",
        423 => "Locked",
        424 => "Failed Dependency",
        425 => "Too Early",
        426 => "Upgrade Required",
        428 => "Precondition Required",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        451 => "Unavailable For Legal Reasons",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        505 => "HTTP Version Not Supported",
        506 => "Variant Also Negotiates",
        507 => "Insufficient Storage",
        508 => "Loop Detected",
        510 => "Not Extended",
        511 => "Network Authentication Required",
        else => "Internal Server Error",
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the query decoder takes + as a space and leaves a short escape" {
    const a = std.testing.allocator;
    const s = try decodeQueryComponent(a, "a+b%41%4");
    defer a.free(s);
    try std.testing.expectEqualStrings("a bA%4", s);
}

test "the path decoder reads %u as UTF-8 and drops a surrogate" {
    const a = std.testing.allocator;
    const s = try decodePathComponent(a, "%u00e9%uD800x%2");
    defer a.free(s);
    try std.testing.expectEqualStrings("\xc3\xa9x%2", s);
}

test "a field value may not start or end with a space" {
    try std.testing.expect(isFieldValue("a b"));
    try std.testing.expect(!isFieldValue(" ab"));
    try std.testing.expect(!isFieldValue("a\r\nb"));
    try std.testing.expect(isFieldValue(""));
}

test "the prohibited names are found by their hash" {
    try std.testing.expect(isProhibitedHeaderName("REMOTE_ADDR"));
    try std.testing.expect(!isProhibitedHeaderName("remote_addr"));
}
