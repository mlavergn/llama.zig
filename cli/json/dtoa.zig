//! The shortest-ish decimal text of a double, as `ordered_json::dump` writes it.
//!
//! # Provenance
//!
//! Ports `nlohmann::detail::dtoa_impl` and `nlohmann::detail::to_chars` from
//! `llama.cpp/vendor/nlohmann/json.hpp` (nlohmann/json 3.12.0, vendored at
//! llama.cpp v0.4.1, `b29c606e2`). Each declaration names what it ports and
//! the line it began at.
//!
//! This is Florian Loitsch's **Grisu2**, not a shortest-digits algorithm:
//! the digits always read back to the same double, but for a small fraction
//! of inputs they are one digit longer, or one unit off in the last place,
//! compared with Ryu or `std.fmt`. `dump` writes exactly these digits, so
//! anything that compares JSON text with upstream's has to produce them too;
//! `make json-diff` holds the two to the same bytes.

const std = @import("std");

/// Ports `diyfp` (json.hpp:17677 @b29c606e2): `f * 2^e`.
const DiyFp = struct {
    f: u64,
    e: i32,

    const precision = 64;

    /// Ports `diyfp`'s `sub` (json.hpp:17690 @b29c606e2): `x - y`, same exponent.
    fn sub(x: DiyFp, y: DiyFp) DiyFp {
        std.debug.assert(x.e == y.e);
        std.debug.assert(x.f >= y.f);
        return .{ .f = x.f - y.f, .e = x.e };
    }

    /// Ports `diyfp`'s `mul` (json.hpp:17702 @b29c606e2): the upper 64 bits of
    /// the 128-bit product, rounded half up.
    fn mul(x: DiyFp, y: DiyFp) DiyFp {
        const u_lo = x.f & 0xFFFFFFFF;
        const u_hi = x.f >> 32;
        const v_lo = y.f & 0xFFFFFFFF;
        const v_hi = y.f >> 32;

        const p0 = u_lo * v_lo;
        const p1 = u_lo * v_hi;
        const p2 = u_hi * v_lo;
        const p3 = u_hi * v_hi;

        const p0_hi = p0 >> 32;
        const p1_lo = p1 & 0xFFFFFFFF;
        const p1_hi = p1 >> 32;
        const p2_lo = p2 & 0xFFFFFFFF;
        const p2_hi = p2 >> 32;

        var q = p0_hi + p1_lo + p2_lo;
        q += @as(u64, 1) << (64 - 32 - 1); // round, ties up

        const h = p3 + p2_hi + p1_hi + (q >> 32);
        return .{ .f = h, .e = x.e + y.e + 64 };
    }

    /// Ports `diyfp`'s `normalize` (json.hpp:17767 @b29c606e2).
    fn normalize(x: DiyFp) DiyFp {
        std.debug.assert(x.f != 0);
        var r = x;
        while ((r.f >> 63) == 0) {
            r.f <<= 1;
            r.e -= 1;
        }
        return r;
    }

    /// Ports `diyfp`'s `normalize_to` (json.hpp:17784 @b29c606e2).
    fn normalizeTo(x: DiyFp, target_exponent: i32) DiyFp {
        const delta: u6 = @intCast(x.e - target_exponent);
        return .{ .f = x.f << delta, .e = target_exponent };
    }
};

/// `boundaries`: `v` and the midpoints to its neighbours, normalized.
const Boundaries = struct {
    w: DiyFp,
    minus: DiyFp,
    plus: DiyFp,
};

/// Ports `compute_boundaries` (json.hpp:17809 @b29c606e2) for `double`.
fn computeBoundaries(value: f64) Boundaries {
    std.debug.assert(std.math.isFinite(value) and value > 0);

    const k_precision = 53; // = p (includes the hidden bit)
    const k_bias = 1024 - 1 + (k_precision - 1);
    const k_min_exp = 1 - k_bias;
    const k_hidden_bit: u64 = @as(u64, 1) << (k_precision - 1);

    const bits: u64 = @bitCast(value);
    const big_e = bits >> (k_precision - 1);
    const big_f = bits & (k_hidden_bit - 1);

    const is_denormal = big_e == 0;
    const v: DiyFp = if (is_denormal)
        .{ .f = big_f, .e = k_min_exp }
    else
        .{ .f = big_f + k_hidden_bit, .e = @as(i32, @intCast(big_e)) - k_bias };

    const lower_boundary_is_closer = big_f == 0 and big_e > 1;
    const m_plus: DiyFp = .{ .f = (2 * v.f) + 1, .e = v.e - 1 };
    const m_minus: DiyFp = if (lower_boundary_is_closer)
        .{ .f = (4 * v.f) - 1, .e = v.e - 2 } // (B)
    else
        .{ .f = (2 * v.f) - 1, .e = v.e - 1 }; // (A)

    const w_plus = m_plus.normalize();
    const w_minus = m_minus.normalizeTo(w_plus.e);

    return .{ .w = v.normalize(), .minus = w_minus, .plus = w_plus };
}

const k_alpha = -60;
const k_gamma = -32;

/// `cached_power`: `f * 2^e ~= 10^k`.
const CachedPower = struct {
    f: u64,
    e: i32,
    k: i32,
};

/// The table of `get_cached_power_for_binary_exponent`.
const cached_powers = [79]CachedPower{
    .{ .f = 0xAB70FE17C79AC6CA, .e = -1060, .k = -300 },
    .{ .f = 0xFF77B1FCBEBCDC4F, .e = -1034, .k = -292 },
    .{ .f = 0xBE5691EF416BD60C, .e = -1007, .k = -284 },
    .{ .f = 0x8DD01FAD907FFC3C, .e = -980, .k = -276 },
    .{ .f = 0xD3515C2831559A83, .e = -954, .k = -268 },
    .{ .f = 0x9D71AC8FADA6C9B5, .e = -927, .k = -260 },
    .{ .f = 0xEA9C227723EE8BCB, .e = -901, .k = -252 },
    .{ .f = 0xAECC49914078536D, .e = -874, .k = -244 },
    .{ .f = 0x823C12795DB6CE57, .e = -847, .k = -236 },
    .{ .f = 0xC21094364DFB5637, .e = -821, .k = -228 },
    .{ .f = 0x9096EA6F3848984F, .e = -794, .k = -220 },
    .{ .f = 0xD77485CB25823AC7, .e = -768, .k = -212 },
    .{ .f = 0xA086CFCD97BF97F4, .e = -741, .k = -204 },
    .{ .f = 0xEF340A98172AACE5, .e = -715, .k = -196 },
    .{ .f = 0xB23867FB2A35B28E, .e = -688, .k = -188 },
    .{ .f = 0x84C8D4DFD2C63F3B, .e = -661, .k = -180 },
    .{ .f = 0xC5DD44271AD3CDBA, .e = -635, .k = -172 },
    .{ .f = 0x936B9FCEBB25C996, .e = -608, .k = -164 },
    .{ .f = 0xDBAC6C247D62A584, .e = -582, .k = -156 },
    .{ .f = 0xA3AB66580D5FDAF6, .e = -555, .k = -148 },
    .{ .f = 0xF3E2F893DEC3F126, .e = -529, .k = -140 },
    .{ .f = 0xB5B5ADA8AAFF80B8, .e = -502, .k = -132 },
    .{ .f = 0x87625F056C7C4A8B, .e = -475, .k = -124 },
    .{ .f = 0xC9BCFF6034C13053, .e = -449, .k = -116 },
    .{ .f = 0x964E858C91BA2655, .e = -422, .k = -108 },
    .{ .f = 0xDFF9772470297EBD, .e = -396, .k = -100 },
    .{ .f = 0xA6DFBD9FB8E5B88F, .e = -369, .k = -92 },
    .{ .f = 0xF8A95FCF88747D94, .e = -343, .k = -84 },
    .{ .f = 0xB94470938FA89BCF, .e = -316, .k = -76 },
    .{ .f = 0x8A08F0F8BF0F156B, .e = -289, .k = -68 },
    .{ .f = 0xCDB02555653131B6, .e = -263, .k = -60 },
    .{ .f = 0x993FE2C6D07B7FAC, .e = -236, .k = -52 },
    .{ .f = 0xE45C10C42A2B3B06, .e = -210, .k = -44 },
    .{ .f = 0xAA242499697392D3, .e = -183, .k = -36 },
    .{ .f = 0xFD87B5F28300CA0E, .e = -157, .k = -28 },
    .{ .f = 0xBCE5086492111AEB, .e = -130, .k = -20 },
    .{ .f = 0x8CBCCC096F5088CC, .e = -103, .k = -12 },
    .{ .f = 0xD1B71758E219652C, .e = -77, .k = -4 },
    .{ .f = 0x9C40000000000000, .e = -50, .k = 4 },
    .{ .f = 0xE8D4A51000000000, .e = -24, .k = 12 },
    .{ .f = 0xAD78EBC5AC620000, .e = 3, .k = 20 },
    .{ .f = 0x813F3978F8940984, .e = 30, .k = 28 },
    .{ .f = 0xC097CE7BC90715B3, .e = 56, .k = 36 },
    .{ .f = 0x8F7E32CE7BEA5C70, .e = 83, .k = 44 },
    .{ .f = 0xD5D238A4ABE98068, .e = 109, .k = 52 },
    .{ .f = 0x9F4F2726179A2245, .e = 136, .k = 60 },
    .{ .f = 0xED63A231D4C4FB27, .e = 162, .k = 68 },
    .{ .f = 0xB0DE65388CC8ADA8, .e = 189, .k = 76 },
    .{ .f = 0x83C7088E1AAB65DB, .e = 216, .k = 84 },
    .{ .f = 0xC45D1DF942711D9A, .e = 242, .k = 92 },
    .{ .f = 0x924D692CA61BE758, .e = 269, .k = 100 },
    .{ .f = 0xDA01EE641A708DEA, .e = 295, .k = 108 },
    .{ .f = 0xA26DA3999AEF774A, .e = 322, .k = 116 },
    .{ .f = 0xF209787BB47D6B85, .e = 348, .k = 124 },
    .{ .f = 0xB454E4A179DD1877, .e = 375, .k = 132 },
    .{ .f = 0x865B86925B9BC5C2, .e = 402, .k = 140 },
    .{ .f = 0xC83553C5C8965D3D, .e = 428, .k = 148 },
    .{ .f = 0x952AB45CFA97A0B3, .e = 455, .k = 156 },
    .{ .f = 0xDE469FBD99A05FE3, .e = 481, .k = 164 },
    .{ .f = 0xA59BC234DB398C25, .e = 508, .k = 172 },
    .{ .f = 0xF6C69A72A3989F5C, .e = 534, .k = 180 },
    .{ .f = 0xB7DCBF5354E9BECE, .e = 561, .k = 188 },
    .{ .f = 0x88FCF317F22241E2, .e = 588, .k = 196 },
    .{ .f = 0xCC20CE9BD35C78A5, .e = 614, .k = 204 },
    .{ .f = 0x98165AF37B2153DF, .e = 641, .k = 212 },
    .{ .f = 0xE2A0B5DC971F303A, .e = 667, .k = 220 },
    .{ .f = 0xA8D9D1535CE3B396, .e = 694, .k = 228 },
    .{ .f = 0xFB9B7CD9A4A7443C, .e = 720, .k = 236 },
    .{ .f = 0xBB764C4CA7A44410, .e = 747, .k = 244 },
    .{ .f = 0x8BAB8EEFB6409C1A, .e = 774, .k = 252 },
    .{ .f = 0xD01FEF10A657842C, .e = 800, .k = 260 },
    .{ .f = 0x9B10A4E5E9913129, .e = 827, .k = 268 },
    .{ .f = 0xE7109BFBA19C0C9D, .e = 853, .k = 276 },
    .{ .f = 0xAC2820D9623BF429, .e = 880, .k = 284 },
    .{ .f = 0x80444B5E7AA7CF85, .e = 907, .k = 292 },
    .{ .f = 0xBF21E44003ACDD2D, .e = 933, .k = 300 },
    .{ .f = 0x8E679C2F5E44FF8F, .e = 960, .k = 308 },
    .{ .f = 0xD433179D9C8CB841, .e = 986, .k = 316 },
    .{ .f = 0x9E19DB92B4E31BA9, .e = 1013, .k = 324 },
};

/// Ports `get_cached_power_for_binary_exponent` (json.hpp:17948 @b29c606e2).
fn cachedPowerForBinaryExponent(e: i32) CachedPower {
    const k_cached_powers_min_dec_exp = -300;
    const k_cached_powers_dec_step = 8;

    std.debug.assert(e >= -1500 and e <= 1500);
    const f = k_alpha - e - 1;
    // C integer division truncates toward zero
    const k = @divTrunc(f * 78913, 1 << 18) + @intFromBool(f > 0);

    const index = @divTrunc(-k_cached_powers_min_dec_exp + k + (k_cached_powers_dec_step - 1), k_cached_powers_dec_step);
    return cached_powers[@intCast(index)];
}

/// Ports `find_largest_pow10` (json.hpp:18112 @b29c606e2): `k` such that
/// `10^(k-1) <= n < 10^k`, and `10^(k-1)` in `pow10`.
fn findLargestPow10(n: u32, pow10: *u32) i32 {
    const steps = [_]u32{ 1000000000, 100000000, 10000000, 1000000, 100000, 10000, 1000, 100, 10 };
    for (steps, 0..) |p, i| {
        if (n >= p) {
            pow10.* = p;
            return @intCast(10 - i);
        }
    }
    pow10.* = 1;
    return 1;
}

/// Ports `grisu2_round` (json.hpp:18166 @b29c606e2).
fn grisu2Round(buf: []u8, len: usize, dist: u64, delta: u64, rest_in: u64, ten_k: u64) void {
    var rest = rest_in;
    while (rest < dist and delta - rest >= ten_k and
        (rest +% ten_k < dist or dist - rest > rest +% ten_k -% dist))
    {
        std.debug.assert(buf[len - 1] != '0');
        buf[len - 1] -= 1;
        rest +%= ten_k;
    }
}

/// Ports `grisu2_digit_gen` (json.hpp:18207 @b29c606e2): the digits of
/// `M+`, cut where they fall inside `[M-, M+]`.
fn grisu2DigitGen(buffer: []u8, length: *usize, decimal_exponent: *i32, m_minus: DiyFp, w: DiyFp, m_plus: DiyFp) void {
    std.debug.assert(m_plus.e >= k_alpha and m_plus.e <= k_gamma);

    var delta = DiyFp.sub(m_plus, m_minus).f;
    var dist = DiyFp.sub(m_plus, w).f;

    const shift: u6 = @intCast(-m_plus.e);
    const one: DiyFp = .{ .f = @as(u64, 1) << shift, .e = m_plus.e };

    var p1: u32 = @intCast(m_plus.f >> shift);
    var p2: u64 = m_plus.f & (one.f - 1);

    std.debug.assert(p1 > 0);

    var pow10: u32 = 0;
    const k = findLargestPow10(p1, &pow10);

    var n = k;
    while (n > 0) {
        const d = p1 / pow10;
        const r = p1 % pow10;
        buffer[length.*] = @intCast('0' + d);
        length.* += 1;
        p1 = r;
        n -= 1;

        const rest = (@as(u64, p1) << shift) + p2;
        if (rest <= delta) {
            decimal_exponent.* += n;
            const ten_n = @as(u64, pow10) << shift;
            grisu2Round(buffer, length.*, dist, delta, rest, ten_n);
            return;
        }

        pow10 /= 10;
    }

    std.debug.assert(p2 > delta);

    var m: i32 = 0;
    while (true) {
        p2 *= 10;
        const d = p2 >> shift;
        const r = p2 & (one.f - 1);
        buffer[length.*] = @intCast('0' + d);
        length.* += 1;
        p2 = r;
        m += 1;

        delta *%= 10;
        dist *%= 10;

        if (p2 <= delta) break;
    }

    decimal_exponent.* -= m;
    const ten_m = one.f;
    grisu2Round(buffer, length.*, dist, delta, p2, ten_m);
}

/// Ports `grisu2` (json.hpp:18448 @b29c606e2) and its `FloatType` overload
/// (json.hpp:18508 @b29c606e2) for `double`.
fn grisu2(buf: []u8, len: *usize, decimal_exponent: *i32, value: f64) void {
    const b = computeBoundaries(value);

    const cached = cachedPowerForBinaryExponent(b.plus.e);
    const c_minus_k: DiyFp = .{ .f = cached.f, .e = cached.e };

    const w = DiyFp.mul(b.w, c_minus_k);
    const w_minus = DiyFp.mul(b.minus, c_minus_k);
    const w_plus = DiyFp.mul(b.plus, c_minus_k);

    const m_minus: DiyFp = .{ .f = w_minus.f + 1, .e = w_minus.e };
    const m_plus: DiyFp = .{ .f = w_plus.f - 1, .e = w_plus.e };

    decimal_exponent.* = -cached.k;

    grisu2DigitGen(buf, len, decimal_exponent, m_minus, w, m_plus);
}

/// Ports `append_exponent` (json.hpp:18548 @b29c606e2): `e+NN`, at least two
/// digits, always signed.
fn appendExponent(buf: []u8, pos: usize, e_in: i32) usize {
    std.debug.assert(e_in > -1000 and e_in < 1000);
    var p = pos;
    var e = e_in;
    if (e < 0) {
        e = -e;
        buf[p] = '-';
    } else {
        buf[p] = '+';
    }
    p += 1;

    var k: u32 = @intCast(e);
    if (k < 10) {
        buf[p] = '0';
        buf[p + 1] = @intCast('0' + k);
        p += 2;
    } else if (k < 100) {
        buf[p] = @intCast('0' + k / 10);
        k %= 10;
        buf[p + 1] = @intCast('0' + k);
        p += 2;
    } else {
        buf[p] = @intCast('0' + k / 100);
        k %= 100;
        buf[p + 1] = @intCast('0' + k / 10);
        k %= 10;
        buf[p + 2] = @intCast('0' + k);
        p += 3;
    }
    return p;
}

/// Ports `format_buffer` (json.hpp:18600 @b29c606e2): fixed notation for
/// `10^min_exp <= v < 10^max_exp`, exponential otherwise.
fn formatBuffer(buf: []u8, len: usize, decimal_exponent: i32, min_exp: i32, max_exp: i32) usize {
    const k: i32 = @intCast(len);
    const n: i32 = k + decimal_exponent;
    const ku: usize = len;

    if (k <= n and n <= max_exp) {
        // digits[000].0
        const nu: usize = @intCast(n);
        @memset(buf[ku..nu], '0');
        buf[nu] = '.';
        buf[nu + 1] = '0';
        return nu + 2;
    }

    if (0 < n and n <= max_exp) {
        // dig.its
        const nu: usize = @intCast(n);
        std.mem.copyBackwards(u8, buf[nu + 1 .. ku + 1], buf[nu..ku]);
        buf[nu] = '.';
        return ku + 1;
    }

    if (min_exp < n and n <= 0) {
        // 0.[000]digits
        const neg: usize = @intCast(-n);
        std.mem.copyBackwards(u8, buf[2 + neg .. 2 + neg + ku], buf[0..ku]);
        buf[0] = '0';
        buf[1] = '.';
        @memset(buf[2 .. 2 + neg], '0');
        return 2 + neg + ku;
    }

    var p: usize = 0;
    if (k == 1) {
        // dE+123
        p = 1;
    } else {
        // d.igitsE+123
        std.mem.copyBackwards(u8, buf[2 .. ku + 1], buf[1..ku]);
        buf[1] = '.';
        p = 1 + ku;
    }

    buf[p] = 'e';
    return appendExponent(buf, p + 1, n - 1);
}

/// The longest text `toChars` writes: a sign, 17 digits, a point, `e-324`,
/// with room for `format_buffer`'s shuffles.
pub const max_len = 32;

/// Ports `to_chars` (json.hpp:18685 @b29c606e2) for `double`: `value` as
/// `dump` writes a finite number.
///
/// Parameters:
/// - `buf`: at least `max_len` bytes; written from the start.
/// - `value`: finite; NaN and infinity are the caller's (`dump` writes `null`).
///
/// Return: the text, a prefix of `buf`.
pub fn toChars(buf: []u8, value_in: f64) []u8 {
    std.debug.assert(std.math.isFinite(value_in));
    std.debug.assert(buf.len >= max_len);

    var value = value_in;
    var first: usize = 0;
    if (std.math.signbit(value)) {
        value = -value;
        buf[0] = '-';
        first = 1;
    }

    if (value == 0) { // +-0
        buf[first] = '0';
        buf[first + 1] = '.';
        buf[first + 2] = '0';
        return buf[0 .. first + 3];
    }

    const digits = buf[first..];
    var len: usize = 0;
    var decimal_exponent: i32 = 0;
    grisu2(digits, &len, &decimal_exponent, value);

    // std::numeric_limits<double>::digits10
    const k_min_exp = -4;
    const k_max_exp = 15;

    const end = formatBuffer(digits, len, decimal_exponent, k_min_exp, k_max_exp);
    return buf[0 .. first + end];
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

fn expectChars(expected: []const u8, v: f64) !void {
    var buf: [max_len]u8 = undefined;
    try std.testing.expectEqualStrings(expected, toChars(&buf, v));
}

test "to_chars writes nlohmann's text" {
    try expectChars("0.0", 0.0);
    try expectChars("-0.0", -0.0);
    try expectChars("1.0", 1.0);
    try expectChars("-1.5", -1.5);
    try expectChars("0.1", 0.1);
    try expectChars("0.0001", 0.0001);
    try expectChars("1e-05", 0.00001);
    try expectChars("100000000000000.0", 1e14);
    try expectChars("1e+15", 1e15);
    try expectChars("1.7976931348623157e+308", std.math.floatMax(f64));
    try expectChars("5e-324", std.math.floatTrueMin(f64));
    try expectChars("0.30000000000000004", @as(f64, 0.1) + @as(f64, 0.2));
    try expectChars("1.2345678901234567e+19", 12345678901234567890.0);
}
