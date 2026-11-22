//! C's numeric conversions -- `strtol` and its siblings, `strtod`,
//! `strtof`, `atoi`, `atol`, `atof` -- in Zig, as the C library answers
//! them in the C locale: the value, where the conversion stopped, and
//! whether it overflowed (`errno == ERANGE`); and libc++'s `std::stoi` over
//! them.
//!
//! # Provenance
//!
//! **Not a port of a file.** Upstream calls the C library; std's `parseInt`
//! and `parseFloat` reject what is not a number from end to end, so this
//! scans the prefix the C library accepts and hands the arithmetic to std.
//! `parseFloat` rounds correctly, as `strtod` and `strtof` do, so a scanned
//! prefix reads to the same bits. `long` is 32 bits on Windows, which moves
//! where `strtol` and `strtoul` saturate.

const std = @import("std");
const builtin = @import("builtin");

/// What a conversion answers: the value, the offset where it stopped (`end
/// - s`; 0 when nothing converted), and whether the value was out of range.
pub fn Parsed(comptime T: type) type {
    return struct {
        value: T,
        end: usize,
        range: bool = false,
    };
}

fn skipSpace(s: []const u8) usize {
    var i: usize = 0;
    while (i < s.len and std.ascii.isWhitespace(s[i])) i += 1;
    return i;
}

fn digitValue(ch: u8) ?u8 {
    return switch (ch) {
        '0'...'9' => ch - '0',
        'a'...'z' => ch - 'a' + 10,
        'A'...'Z' => ch - 'A' + 10,
        else => null,
    };
}

/// The integer conversions: `strtol`, `strtoll`, `strtoul`, `strtoull` for
/// `T` = `c_long`, `c_longlong`, `c_ulong`, `c_ulonglong`.
///
/// Parameters:
/// - `T`: the C type the conversion returns.
/// - `s`: the text, read up to its end or a byte that ends the number.
/// - `base_in`: 0 (from the prefix: `0x` hex, `0` octal, else decimal) or 2
///   to 36.
///
/// Return: the value, saturated and flagged out of range on overflow; an
/// unsigned conversion of a negative number wraps, as C's does.
pub fn strtoInt(comptime T: type, s: []const u8, base_in: u8) Parsed(T) {
    const unsigned = @typeInfo(T).int.signedness == .unsigned;
    var i = skipSpace(s);
    var neg = false;
    if (i < s.len and (s[i] == '+' or s[i] == '-')) {
        neg = s[i] == '-';
        i += 1;
    }
    var base: u8 = base_in;
    // the offset a bare "0x" leaves the conversion at: after the 0
    var zero_x_end: ?usize = null;
    if ((base == 0 or base == 16) and i + 1 < s.len and s[i] == '0' and (s[i + 1] == 'x' or s[i + 1] == 'X')) {
        zero_x_end = i + 1;
        i += 2;
        base = 16;
    } else if (base == 0) {
        base = if (i < s.len and s[i] == '0') 8 else 10;
    }

    const U = @Int(.unsigned, @bitSizeOf(T));
    var acc: U = 0;
    var overflow = false;
    const start = i;
    while (i < s.len) : (i += 1) {
        const d = digitValue(s[i]) orelse break;
        if (d >= base) break;
        const mul = @mulWithOverflow(acc, @as(U, base));
        const add = @addWithOverflow(mul[0], @as(U, d));
        if (mul[1] != 0 or add[1] != 0) overflow = true;
        acc = add[0];
    }
    if (i == start) {
        return .{ .value = 0, .end = zero_x_end orelse 0 };
    }

    if (unsigned) {
        if (overflow) return .{ .value = std.math.maxInt(T), .end = i, .range = true };
        return .{ .value = if (neg) 0 -% acc else acc, .end = i };
    }
    const max_pos: U = std.math.maxInt(T);
    if (neg) {
        if (overflow or acc > max_pos + 1) return .{ .value = std.math.minInt(T), .end = i, .range = true };
        return .{ .value = @bitCast(0 -% acc), .end = i };
    }
    if (overflow or acc > max_pos) return .{ .value = std.math.maxInt(T), .end = i, .range = true };
    return .{ .value = @intCast(acc), .end = i };
}

/// `strtol`.
pub fn strtol(s: []const u8, base: u8) Parsed(c_long) {
    return strtoInt(c_long, s, base);
}

/// `strtoll`.
pub fn strtoll(s: []const u8, base: u8) Parsed(c_longlong) {
    return strtoInt(c_longlong, s, base);
}

/// `strtoul`.
pub fn strtoul(s: []const u8, base: u8) Parsed(c_ulong) {
    return strtoInt(c_ulong, s, base);
}

/// `strtoull`.
pub fn strtoull(s: []const u8, base: u8) Parsed(c_ulonglong) {
    return strtoInt(c_ulonglong, s, base);
}

/// `atoi`: glibc and Darwin answer `(int)strtol(s, NULL, 10)`; Windows'
/// C runtime saturates at `int`'s range instead.
///
/// Return: the value; 0 when nothing converted.
pub fn atoi(s: []const u8) c_int {
    const v = strtol(s, 10).value;
    if (comptime builtin.os.tag == .windows) return @intCast(std.math.clamp(v, std.math.minInt(c_int), std.math.maxInt(c_int)));
    return @truncate(v);
}

/// `atoi(s)` over a C string, as the libraries call it on `getenv`'s
/// answer.
pub fn atoiC(s: [*:0]const u8) c_int {
    return atoi(std.mem.span(s));
}

/// `atol`: `strtol(s, NULL, 10)`.
pub fn atol(s: []const u8) c_long {
    return strtol(s, 10).value;
}

/// `atof`: `strtod(s, NULL)`.
pub fn atof(s: []const u8) f64 {
    return strtod(s).value;
}

/// What `std::stoi` or `std::stoull` gives: the value, or which of its two
/// exceptions it throws.
///
/// Parameters:
/// - `T`: the result type.
/// - `name`: the function, as its `what()` names it.
pub fn Converted(comptime T: type, comptime name: []const u8) type {
    return union(enum) {
        ok: T,
        /// `std::invalid_argument`, `what()` "<name>: no conversion".
        no_conversion,
        /// `std::out_of_range`, `what()` "<name>: out of range".
        out_of_range,

        /// `e.what()` for the two throws.
        pub fn what(self: @This()) [*:0]const u8 {
            return switch (self) {
                .ok => "",
                .no_conversion => name ++ ": no conversion",
                .out_of_range => name ++ ": out of range",
            };
        }
    };
}

/// `std::stoi`'s result.
pub const Stoi = Converted(c_int, "stoi");
/// `std::stoull`'s result.
pub const Stoull = Converted(u64, "stoull");

/// `std::stoi(s)`: libc++'s `string.cpp`, `strtol` in base 10 over the C
/// string, then `invalid_argument` if nothing converted and `out_of_range`
/// if `errno` says so or the value leaves `int`.
///
/// Return: the value, or which exception upstream throws.
pub fn stoi(s: []const u8) Stoi {
    const p = strtol(std.mem.sliceTo(s, 0), 10);
    const v = p.value;
    if (p.end == 0) return .no_conversion;
    if (p.range or v < std.math.minInt(c_int) or v > std.math.maxInt(c_int)) return .out_of_range;
    return .{ .ok = @intCast(v) };
}

/// `std::stoull(s)` in base 10: `strtoull` over the C string, its range
/// error and a string with no digits as the two throws.
///
/// Return: the value, or which exception upstream throws.
pub fn stoull(s: []const u8) Stoull {
    const p = strtoull(std.mem.sliceTo(s, 0), 10);
    if (p.end == 0) return .no_conversion;
    if (p.range) return .out_of_range;
    return .{ .ok = p.value };
}

/// `strtod`.
pub fn strtod(s: []const u8) Parsed(f64) {
    return strtoFloat(f64, s);
}

/// `strtof`.
pub fn strtof(s: []const u8) Parsed(f32) {
    return strtoFloat(f32, s);
}

fn startsWithIgnoreCase(s: []const u8, prefix: []const u8) bool {
    return s.len >= prefix.len and std.ascii.eqlIgnoreCase(s[0..prefix.len], prefix);
}

fn isHex(ch: u8) bool {
    return std.ascii.isHex(ch);
}

/// The floating conversions in the C locale: optional space and sign, then
/// `inf` or `infinity`, `nan` with an optional `(n-char-sequence)`, a hex
/// float (`0x`, hex digits with an optional point, an optional `p`
/// exponent), or a decimal one (digits with an optional point, an optional
/// `e` exponent); an exponent counts only when a digit follows it.
///
/// Parameters:
/// - `T`: `f32` (`strtof`) or `f64` (`strtod`).
/// - `s`: the text.
///
/// Return: the value, the offset where the number ended (0 when there was
/// none), and whether it overflowed to infinity or underflowed to a
/// subnormal or zero from nonzero digits.
pub fn strtoFloat(comptime T: type, s: []const u8) Parsed(T) {
    var i = skipSpace(s);
    const num_start = i;
    var neg = false;
    if (i < s.len and (s[i] == '+' or s[i] == '-')) {
        neg = s[i] == '-';
        i += 1;
    }
    const rest = s[i..];
    if (startsWithIgnoreCase(rest, "infinity")) {
        return .{ .value = if (neg) -std.math.inf(T) else std.math.inf(T), .end = i + 8 };
    }
    if (startsWithIgnoreCase(rest, "inf")) {
        return .{ .value = if (neg) -std.math.inf(T) else std.math.inf(T), .end = i + 3 };
    }
    if (startsWithIgnoreCase(rest, "nan")) {
        var end = i + 3;
        if (end < s.len and s[end] == '(') {
            // the n-char-sequence: letters, digits and `_`; Darwin's C
            // library takes anything up to the closing parenthesis
            var j = end + 1;
            while (j < s.len and s[j] != ')' and (builtin.os.tag.isDarwin() or std.ascii.isAlphanumeric(s[j]) or s[j] == '_')) j += 1;
            if (j < s.len and s[j] == ')') end = j + 1;
        }
        const nan = std.math.nan(T);
        return .{ .value = if (neg) -nan else nan, .end = end };
    }

    var hex = false;
    var j = i;
    if (j + 1 < s.len and s[j] == '0' and (s[j + 1] == 'x' or s[j + 1] == 'X')) {
        const after = j + 2;
        if ((after < s.len and isHex(s[after])) or (after + 1 < s.len and s[after] == '.' and isHex(s[after + 1]))) {
            hex = true;
            j = after;
        }
    }
    const isDigit = if (hex) &isHex else &std.ascii.isDigit;

    var n_digits: usize = 0;
    var nonzero = false;
    while (j < s.len and isDigit(s[j])) : (j += 1) {
        n_digits += 1;
        if (s[j] != '0') nonzero = true;
    }
    if (j < s.len and s[j] == '.') {
        var k = j + 1;
        var frac: usize = 0;
        while (k < s.len and isDigit(s[k])) : (k += 1) {
            frac += 1;
            if (s[k] != '0') nonzero = true;
        }
        if (n_digits + frac > 0) {
            n_digits += frac;
            j = k;
        }
    }
    if (n_digits == 0) return .{ .value = 0, .end = 0 };

    const exp_char: u8 = if (hex) 'p' else 'e';
    if (j < s.len and std.ascii.toLower(s[j]) == exp_char) {
        var k = j + 1;
        if (k < s.len and (s[k] == '+' or s[k] == '-')) k += 1;
        if (k < s.len and std.ascii.isDigit(s[k])) {
            while (k < s.len and std.ascii.isDigit(s[k])) k += 1;
            j = k;
        }
    }

    const text = s[num_start..j];
    const value = parseScanned(T, text, hex);
    const inf = std.math.isInf(value);
    // underflow, as each C library reports it (measured): glibc, the
    // exact value below the smallest normal, before rounding, and inexact;
    // Darwin, a result below normal, from decimal whatever its exactness,
    // from hex when inexact. Windows' C runtime is given glibc's rule.
    const below_normal = !std.math.isNormal(value);
    const tiny = nonzero and !inf and if (comptime builtin.os.tag.isDarwin())
        below_normal and (!hex or !exactBelowNormal(T, text, hex))
    else
        (below_normal and !exactBelowNormal(T, text, hex)) or
            (@abs(value) == std.math.floatMin(T) and lessThanMinNormal(T, text, hex));
    return .{ .value = value, .end = j, .range = inf or tiny };
}

/// The exponent of the smallest subnormal `T` holds: 2^-149, 2^-1074.
fn minSubnormalExp(comptime T: type) i64 {
    return std.math.floatExponentMin(T) - std.math.floatFractionalBits(T);
}

/// Whether the number `text` spells, already known to be below `T`'s
/// smallest normal, is exactly a value of `T`: a subnormal's bits run down
/// to 2^`minSubnormalExp` and no further. Integer arithmetic only: a wider
/// float here would link `compiler_rt`, whose `expf` and `logf` shadow the
/// C library's.
fn exactBelowNormal(comptime T: type, text: []const u8, hex: bool) bool {
    const min_exp = minSubnormalExp(T);
    var t = text;
    if (t.len > 0 and (t[0] == '+' or t[0] == '-')) t = t[1..];
    if (hex) t = t[2..];
    // mantissa digits, and the exponent part
    const exp_at = std.mem.indexOfAny(u8, t, if (hex) "pP" else "eE");
    const mant = if (exp_at) |e| t[0..e] else t;
    var exp10_or_2: i64 = 0;
    if (exp_at) |e| exp10_or_2 = std.math.clamp(strtoll(t[e + 1 ..], 10).value, -1_000_000, 1_000_000);
    const point = std.mem.indexOfScalar(u8, mant, '.');
    const int_part = if (point) |q| mant[0..q] else mant;
    const frac_part = if (point) |q| mant[q + 1 ..] else "";

    if (hex) {
        // the lowest set bit: in the last nonzero hex digit
        const all_frac = std.mem.trimEnd(u8, frac_part, "0");
        if (all_frac.len > 0) {
            const v = digitValue(all_frac[all_frac.len - 1]).?;
            const low = exp10_or_2 - 4 * @as(i64, @intCast(all_frac.len)) + @ctz(v);
            return low >= min_exp;
        }
        const ip = std.mem.trimEnd(u8, int_part, "0");
        const v = digitValue(ip[ip.len - 1]).?;
        const low = exp10_or_2 + 4 * @as(i64, @intCast(int_part.len - ip.len)) + @ctz(v);
        return low >= min_exp;
    }

    // decimal N * 10^e, N without leading or trailing zeros: exact only when
    // 5^-e divides N, which needs more digits than a float or double
    // subnormal can be spelled in short
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.heap.page_allocator);
    for (int_part) |ch| buf.append(std.heap.page_allocator, ch - '0') catch return false;
    for (frac_part) |ch| buf.append(std.heap.page_allocator, ch - '0') catch return false;
    var e = exp10_or_2 - @as(i64, @intCast(frac_part.len));
    var lead: usize = 0;
    while (lead < buf.items.len and buf.items[lead] == 0) lead += 1;
    var n = buf.items[lead..];
    while (n.len > 0 and n[n.len - 1] == 0) : (e += 1) n = n[0 .. n.len - 1];
    if (n.len == 0 or e >= 0) return false;
    const min_digits: usize = if (T == f32) 89 else 715;
    if (n.len < min_digits) return false;
    // N / 5^-e, digit by digit
    var k: i64 = 0;
    while (k < -e) : (k += 1) {
        if (divSmall(n, 5) != 0) return false;
        n = trimLeadingZeros(n);
    }
    // the bits of N / 5^-e run down to 2^(e + its trailing zero bits)
    var low = e;
    while (low < min_exp) : (low += 1) {
        if (divSmall(n, 2) != 0) return false;
        n = trimLeadingZeros(n);
    }
    return true;
}

/// Whether the number `text` spells is, exactly, below `T`'s smallest
/// normal 2^-k: for a result that rounded up to that normal, whether glibc
/// counts it tiny. Integer arithmetic only, as `exactBelowNormal`.
fn lessThanMinNormal(comptime T: type, text: []const u8, hex: bool) bool {
    const k: i64 = -std.math.floatExponentMin(T);
    var t = text;
    if (t.len > 0 and (t[0] == '+' or t[0] == '-')) t = t[1..];
    if (hex) t = t[2..];
    const exp_at = std.mem.indexOfAny(u8, t, if (hex) "pP" else "eE");
    const mant = if (exp_at) |e| t[0..e] else t;
    var exp: i64 = 0;
    if (exp_at) |e| exp = std.math.clamp(strtoll(t[e + 1 ..], 10).value, -1_000_000, 1_000_000);
    const point = std.mem.indexOfScalar(u8, mant, '.') orelse mant.len;
    if (hex) {
        // the top set bit, from the first nonzero hex digit
        for (mant, 0..) |ch, i| {
            if (ch == '.') continue;
            const v = digitValue(ch).?;
            if (v == 0) continue;
            const pos: i64 = if (i < point) @as(i64, @intCast(point - 1 - i)) * 4 else -@as(i64, @intCast(i - point)) * 4;
            const top = exp + pos + (3 - @as(i64, @clz(@as(u4, @intCast(v)))));
            return top < -k;
        }
        return true;
    }
    // decimal: 0.d1 d2 ... * 10^(n + e) against 2^-k = 0.c1 c2 ... * 10^(L - k),
    // the c being the digits of 5^k
    var digits: std.ArrayList(u8) = .empty;
    defer digits.deinit(std.heap.page_allocator);
    var first: ?usize = null;
    for (mant, 0..) |ch, i| {
        if (ch == '.') continue;
        if (first == null and ch == '0') continue;
        if (first == null) first = i;
        digits.append(std.heap.page_allocator, ch - '0') catch return false;
    }
    const f = first orelse return true;
    const lead_exp: i64 = if (f < point) exp + @as(i64, @intCast(point - f)) else exp - @as(i64, @intCast(f - point - 1));
    var five: std.ArrayList(u8) = .empty;
    defer five.deinit(std.heap.page_allocator);
    five.append(std.heap.page_allocator, 1) catch return false;
    var n: i64 = 0;
    while (n < k) : (n += 1) {
        var carry: u16 = 0;
        var i = five.items.len;
        while (i > 0) : (i -= 1) {
            const v = @as(u16, five.items[i - 1]) * 5 + carry;
            five.items[i - 1] = @intCast(v % 10);
            carry = v / 10;
        }
        while (carry > 0) : (carry /= 10) five.insert(std.heap.page_allocator, 0, @intCast(carry % 10)) catch return false;
    }
    const pow_exp: i64 = @as(i64, @intCast(five.items.len)) - k;
    if (lead_exp != pow_exp) return lead_exp < pow_exp;
    const len = @max(digits.items.len, five.items.len);
    for (0..len) |i| {
        const a = if (i < digits.items.len) digits.items[i] else 0;
        const b = if (i < five.items.len) five.items[i] else 0;
        if (a != b) return a < b;
    }
    return false;
}

/// Divides the decimal digits `n` (most significant first) by `d` in place.
///
/// Return: the remainder.
fn divSmall(n: []u8, d: u8) u8 {
    var rem: u16 = 0;
    for (n) |*digit| {
        const cur = rem * 10 + digit.*;
        digit.* = @intCast(cur / d);
        rem = cur % d;
    }
    return @intCast(rem);
}

fn trimLeadingZeros(n: []u8) []u8 {
    var i: usize = 0;
    while (i + 1 < n.len and n[i] == 0) i += 1;
    return n[i..];
}

/// std's reading of a number `strtoFloat` scanned, which for a hex float
/// needs an exponent: `p0` is added where the text has none.
fn parseScanned(comptime T: type, text: []const u8, hex: bool) T {
    if (!hex or std.mem.indexOfAny(u8, text, "pP") != null) return std.fmt.parseFloat(T, text) catch 0;
    var buf: [512]u8 = undefined;
    if (text.len + 2 <= buf.len) {
        @memcpy(buf[0..text.len], text);
        @memcpy(buf[text.len..][0..2], "p0");
        return std.fmt.parseFloat(T, buf[0 .. text.len + 2]) catch 0;
    }
    const a = std.heap.page_allocator;
    const z = std.mem.concat(a, u8, &.{ text, "p0" }) catch return 0;
    defer a.free(z);
    return std.fmt.parseFloat(T, z) catch 0;
}

/// Ports `GGML_PAD` (ggml.h:94 @b1b770d90): `(x + n - 1) & ~(n - 1)` in
/// `T`'s own arithmetic, which wraps as the C's `size_t`, `uint32_t` and
/// two's-complement `int64_t` do. A mask, so for an `n` that is not a power
/// of two it is not a round-up, as in the C.
pub inline fn pad(comptime T: type, x: T, n: T) T {
    return (x +% n -% 1) & ~(n -% 1);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "integers stop, saturate and wrap as C's do" {
    try std.testing.expectEqual(Parsed(c_long){ .value = 42, .end = 4 }, strtol("  42x", 10));
    try std.testing.expectEqual(Parsed(c_long){ .value = 0, .end = 1 }, strtol("0x", 0));
    try std.testing.expectEqual(Parsed(c_long){ .value = 255, .end = 4 }, strtol("0xff", 0));
    try std.testing.expectEqual(Parsed(c_long){ .value = 8, .end = 3 }, strtol("010", 0));
    try std.testing.expectEqual(Parsed(c_long){ .value = 0, .end = 0 }, strtol("  -x", 10));
    try std.testing.expectEqual(Parsed(c_ulonglong){ .value = std.math.maxInt(c_ulonglong), .end = 2 }, strtoull("-1", 10));
    try std.testing.expect(strtoll("99999999999999999999", 10).range);
}

test "stoi converts a prefix and throws as libc++ does" {
    try std.testing.expectEqual(Stoi{ .ok = 12 }, stoi(" 12abc"));
    try std.testing.expectEqual(Stoi.no_conversion, stoi("abc"));
    try std.testing.expectEqual(Stoi.out_of_range, stoi("99999999999"));
}

test "stoull converts a prefix, wraps a sign and throws as libc++ does" {
    try std.testing.expectEqual(Stoull{ .ok = 42 }, stoull(" 42x"));
    try std.testing.expectEqual(Stoull{ .ok = std.math.maxInt(u64) }, stoull("-1"));
    try std.testing.expectEqual(Stoull{ .ok = 7 }, stoull("7\x00 9"));
    try std.testing.expectEqual(Stoull.no_conversion, stoull("abc"));
    try std.testing.expectEqual(Stoull.out_of_range, stoull("18446744073709551616"));
    try std.testing.expectEqualStrings("stoull: out of range", std.mem.span((Stoull{ .out_of_range = {} }).what()));
}

test "pad is GGML_PAD: a round-up to a power of two, wrapping as size_t does" {
    try std.testing.expectEqual(@as(usize, 32), pad(usize, 1, 32));
    try std.testing.expectEqual(@as(usize, 32), pad(usize, 32, 32));
    try std.testing.expectEqual(@as(u32, 0), pad(u32, 0, 256));
    try std.testing.expectEqual(@as(usize, 0), pad(usize, std.math.maxInt(usize), 32));
    try std.testing.expectEqual(@as(i64, 96), pad(i64, 65, 32));
}
