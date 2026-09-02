//! Reference dot products for the K-quant super-block formats.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/quants.c` (v0.3.0, `c1d0e7a00`),
//! the `_generic` dot products at lines 565-905. Each function names the C it
//! replaces and the line it began at.
//!
//! # Unreachable on this target
//!
//! As with `legacy.zig`: `arch/arm/quants.c` supplies the real entry points and
//! these `_generic` names are exported and never called. `golden.zig` is their
//! only gate.
//!
//! # The accumulation order is the specification
//!
//! Four of the five build the result in an eight-wide `sums` array that is
//! **not** reset between super-blocks, add the per-block `dmin` correction into
//! `sumf` as they go, and only fold `sums` into `sumf` after the last block.
//! Float addition is not associative, so that order is part of the answer, not
//! an implementation detail. The C's comment on `q3_K` says it is written the
//! way it is so the compiler can vectorise it; the shape is kept regardless.

const std = @import("std");
const impl = @import("../../impl.zig");
const convert = @import("../convert.zig");
const blocks = @import("../../quants/blocks.zig");
const c = impl.c;

inline fn f(h: u16) f32 {
    return convert.cpuFp16ToFp32(h);
}

inline fn as(comptime Block: type, p: ?*const anyopaque) [*]const Block {
    return @ptrCast(@alignCast(p.?));
}

/// Ports the 12-byte scale/min unpacking `q4_K` and `q5_K` share
/// (quants.c:735 and quants.c:817).
///
/// The six-bit scales and mins are packed across twelve bytes; this shuffles
/// them into four words, the low two holding eight scales and the high two
/// eight mins, both byte-addressable.
///
/// Parameters:
/// - `packed_scales`: the block's twelve scale bytes.
///
/// Return: the four words, to be read as bytes.
fn unpackScalesMins(packed_scales: *const [12]u8) [4]u32 {
    const kmask1: u32 = 0x3f3f3f3f;
    const kmask2: u32 = 0x0f0f0f0f;
    const kmask3: u32 = 0x03030303;

    var utmp: [4]u32 = undefined;
    @memcpy(std.mem.asBytes(&utmp)[0..12], packed_scales);

    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    return utmp;
}

/// Ports `ggml_vec_dot_q2_K_q8_K_generic` (ggml-cpu/quants.c:565 @c1d0e7a00).
///
/// Two-bit weights with a four-bit scale and a four-bit minimum per group of
/// sixteen. The minima are handled in one pass up front against `q8_K`'s
/// per-group sums, which is what `bsums` exists for.
pub export fn ggml_vec_dot_q2_K_q8_K_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q2_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |i| {
        const sc = &x[i].scales;

        var summs: i32 = 0;
        for (0..16) |j| {
            summs += @as(i32, y[i].bsums[j]) * (sc[j] >> 4);
        }

        const dall = y[i].d * f(x[i].d);
        const dmin = y[i].d * f(x[i].dmin);

        var isum: i32 = 0;
        var is: usize = 0;
        var q2: usize = 0; // offset into x[i].qs
        var q8: usize = 0; // offset into y[i].qs

        for (0..c.QK_K / 128) |_| {
            // `u8`, not `u3`: the C's `shift += 2` runs one more time than it
            // is used, ending at 8, which a `u3` cannot hold. Narrowed at each
            // use instead, where the value is always 0, 2, 4 or 6.
            var shift: u8 = 0;
            for (0..4) |_| {
                var d: i32 = sc[is] & 0xF;
                is += 1;
                var isuml: i32 = 0;
                for (0..16) |l| {
                    isuml += @as(i32, y[i].qs[q8 + l]) * ((x[i].qs[q2 + l] >> @intCast(shift)) & 3);
                }
                isum += d * isuml;

                d = sc[is] & 0xF;
                is += 1;
                isuml = 0;
                for (16..32) |l| {
                    isuml += @as(i32, y[i].qs[q8 + l]) * ((x[i].qs[q2 + l] >> @intCast(shift)) & 3);
                }
                isum += d * isuml;

                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }

        sumf += dall * @as(f32, @floatFromInt(isum)) - dmin * @as(f32, @floatFromInt(summs));
    }

    s[0] = sumf;
}

/// The scratch four of these kernels share.
///
/// `sums` survives the whole row; the rest are per-super-block. The C declares
/// them as locals at the top of the function and relies on `memset` to reset
/// the right ones, which is easy to misread -- hence the struct.
const Scratch = struct {
    /// The unpacked weights of one super-block, in signed form.
    aux8: [c.QK_K]i8 = undefined,
    /// Eight products at a time, so the scale multiply stays 32-bit.
    aux16: [8]i16 = undefined,
    /// Eight scaled accumulators, reset per super-block.
    aux32: [8]i32 = @splat(0),
    /// Eight float accumulators, **not** reset between super-blocks.
    sums: [8]f32 = @splat(0),

    /// Adds `q8 * a` scaled by `scale` into `aux32`, eight elements at a time,
    /// the inner step every one of these kernels repeats.
    inline fn accumulate(self: *Scratch, scale: i32, q8: []const i8, a: []const i8) void {
        for (0..8) |l| self.aux16[l] = @as(i16, q8[l]) *% a[l];
        for (0..8) |l| self.aux32[l] += scale * self.aux16[l];
    }

    /// Folds one super-block's accumulators into the float sums.
    ///
    /// Named `foldInto` rather than `scale` because `accumulate` already takes
    /// a parameter by that name.
    inline fn foldInto(self: *Scratch, d: f32) void {
        for (0..8) |l| self.sums[l] += d * @as(f32, @floatFromInt(self.aux32[l]));
    }

    /// The row total: the C sums the eight lanes after the block loop.
    inline fn total(self: *const Scratch, sumf_in: f32) f32 {
        var sumf = sumf_in;
        for (0..8) |l| sumf += self.sums[l];
        return sumf;
    }
};

/// Ports `ggml_vec_dot_q3_K_q8_K_generic` (ggml-cpu/quants.c:617 @c1d0e7a00).
///
/// Three-bit weights: two bits in `qs` and an inverted high bit in `hmask`,
/// so a clear mask bit subtracts 4. Scales are six-bit, biased by 32, packed
/// across twelve bytes in a different layout from `q4_K`'s.
pub export fn ggml_vec_dot_q3_K_q8_K_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;

    const x = as(blocks.Q3_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    var sc: Scratch = .{};
    var auxs: [4]u32 = undefined;

    // Never added to inside the loop: unlike q4_K and q5_K, q3_K has no
    // minimum to subtract, so the whole result comes out of `sums`.
    const sumf: f32 = 0;
    for (0..@intCast(nb)) |i| {
        const hm = &x[i].hmask;
        sc.aux32 = @splat(0);

        var a: usize = 0;
        var q3: usize = 0;
        var m: u8 = 1;

        var j: usize = 0;
        while (j < c.QK_K) : (j += 128) {
            inline for (0..4) |half| {
                const shift: u3 = half * 2;
                for (0..32) |l| sc.aux8[a + l] = @intCast((x[i].qs[q3 + l] >> shift) & 3);
                for (0..32) |l| sc.aux8[a + l] -= if (hm[l] & m != 0) 0 else 4;
                a += 32;
                // Wrapping: the eighth shift takes 128 to 0, which the C does
                // by truncation and never reads.
                m *%= 2;
            }
            q3 += 32;
        }
        a = 0;

        @memcpy(std.mem.asBytes(&auxs)[0..12], &x[i].scales);
        const tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        // Read back as signed bytes, which is what the C's `scales` alias does.
        const scales: *const [16]i8 = @ptrCast(&auxs);

        var q8: usize = 0;
        for (0..c.QK_K / 16) |sj| {
            const scale: i32 = @as(i32, scales[sj]) - 32;
            sc.accumulate(scale, y[i].qs[q8..], sc.aux8[a..]);
            q8 += 8;
            a += 8;
            sc.accumulate(scale, y[i].qs[q8..], sc.aux8[a..]);
            q8 += 8;
            a += 8;
        }

        sc.foldInto(f(x[i].d) * y[i].d);
    }

    s[0] = sc.total(sumf);
}

/// The body `q4_K` and `q5_K` share (quants.c:696 and quants.c:771).
///
/// They differ only in how `aux8` is filled: `q4_K` takes the nibble as-is,
/// `q5_K` adds 16 where the corresponding `qh` bit is set. Everything after
/// that -- the scale unpacking, the min correction against `bsums`, the
/// accumulation order -- is identical, and duplicating it once per format is
/// how a divergence gets introduced.
inline fn dotQ4Q5K(
    comptime Block: type,
    n: c_int,
    s: [*c]f32,
    vx: ?*const anyopaque,
    vy: ?*const anyopaque,
) void {
    const x = as(Block, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    var sc: Scratch = .{};

    var sumf: f32 = 0;
    for (0..@intCast(nb)) |i| {
        sc.aux32 = @splat(0);

        var a: usize = 0;
        var q4: usize = 0;
        var m: u8 = 1;

        for (0..c.QK_K / 64) |_| {
            for (0..32) |l| sc.aux8[a + l] = @bitCast(x[i].qs[q4 + l] & 0xF);
            if (Block == blocks.Q5_K) {
                for (0..32) |l| sc.aux8[a + l] += if (x[i].qh[l] & m != 0) 16 else 0;
                m *%= 2;
            }
            a += 32;

            for (0..32) |l| sc.aux8[a + l] = @bitCast(x[i].qs[q4 + l] >> 4);
            if (Block == blocks.Q5_K) {
                for (0..32) |l| sc.aux8[a + l] += if (x[i].qh[l] & m != 0) 16 else 0;
                // Wrapping, as in q3_K: the last shift is never read.
                m *%= 2;
            }
            a += 32;
            q4 += 32;
        }

        const utmp = unpackScalesMins(&x[i].scales);
        const scales: *const [16]u8 = @ptrCast(&utmp);
        const mins = scales[8..];

        var sumi: i32 = 0;
        for (0..c.QK_K / 16) |j| sumi += @as(i32, y[i].bsums[j]) * mins[j / 2];

        a = 0;
        var is: usize = 0;
        var q8: usize = 0;
        for (0..c.QK_K / 32) |_| {
            const scale: i32 = scales[is];
            is += 1;
            for (0..4) |_| {
                sc.accumulate(scale, y[i].qs[q8..], sc.aux8[a..]);
                q8 += 8;
                a += 8;
            }
        }

        sc.foldInto(f(x[i].d) * y[i].d);
        const dmin = f(x[i].dmin) * y[i].d;
        sumf -= dmin * @as(f32, @floatFromInt(sumi));
    }

    s[0] = sc.total(sumf);
}

/// Ports `ggml_vec_dot_q4_K_q8_K_generic` (ggml-cpu/quants.c:696 @c1d0e7a00).
pub export fn ggml_vec_dot_q4_K_q8_K_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    dotQ4Q5K(blocks.Q4_K, n, s, vx, vy);
}

/// Ports `ggml_vec_dot_q5_K_q8_K_generic` (ggml-cpu/quants.c:771 @c1d0e7a00).
pub export fn ggml_vec_dot_q5_K_q8_K_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    dotQ4Q5K(blocks.Q5_K, n, s, vx, vy);
}

/// Ports `ggml_vec_dot_q6_K_q8_K_generic` (ggml-cpu/quants.c:851 @c1d0e7a00).
///
/// Six-bit weights biased by 32: four bits in `ql` and two in `qh`. The scales
/// are already signed bytes, so there is no unpacking and no minimum.
pub export fn ggml_vec_dot_q6_K_q8_K_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q6_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    var sc: Scratch = .{};

    // As in q3_K: no minimum, so nothing is added here inside the loop.
    const sumf: f32 = 0;
    for (0..@intCast(nb)) |i| {
        sc.aux32 = @splat(0);

        var a: usize = 0;
        var q4: usize = 0;
        var qh: usize = 0;

        var j: usize = 0;
        while (j < c.QK_K) : (j += 128) {
            for (0..32) |l| {
                const h = x[i].qh[qh + l];
                sc.aux8[a + l + 0] = @as(i8, @bitCast((x[i].ql[q4 + l + 0] & 0xF) | (((h >> 0) & 3) << 4))) -% 32;
                sc.aux8[a + l + 32] = @as(i8, @bitCast((x[i].ql[q4 + l + 32] & 0xF) | (((h >> 2) & 3) << 4))) -% 32;
                sc.aux8[a + l + 64] = @as(i8, @bitCast((x[i].ql[q4 + l + 0] >> 4) | (((h >> 4) & 3) << 4))) -% 32;
                sc.aux8[a + l + 96] = @as(i8, @bitCast((x[i].ql[q4 + l + 32] >> 4) | (((h >> 6) & 3) << 4))) -% 32;
            }
            a += 128;
            q4 += 64;
            qh += 32;
        }

        a = 0;
        var is: usize = 0;
        var q8: usize = 0;
        for (0..c.QK_K / 16) |_| {
            const scale: i32 = x[i].scales[is];
            is += 1;
            sc.accumulate(scale, y[i].qs[q8..], sc.aux8[a..]);
            q8 += 8;
            a += 8;
            sc.accumulate(scale, y[i].qs[q8..], sc.aux8[a..]);
            q8 += 8;
            a += 8;
        }

        sc.foldInto(f(x[i].d) * y[i].d);
    }

    s[0] = sc.total(sumf);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

const testing = @import("testing.zig");
const golden = @import("golden.zig");

test "q2_K dot matches the C" {
    try testing.check(ggml_vec_dot_q2_K_q8_K_generic, c.GGML_TYPE_Q2_K, c.GGML_TYPE_Q8_K, golden.q2_K);
}

test "q3_K dot matches the C" {
    try testing.check(ggml_vec_dot_q3_K_q8_K_generic, c.GGML_TYPE_Q3_K, c.GGML_TYPE_Q8_K, golden.q3_K);
}

test "q4_K dot matches the C" {
    try testing.check(ggml_vec_dot_q4_K_q8_K_generic, c.GGML_TYPE_Q4_K, c.GGML_TYPE_Q8_K, golden.q4_K);
}

test "q5_K dot matches the C" {
    try testing.check(ggml_vec_dot_q5_K_q8_K_generic, c.GGML_TYPE_Q5_K, c.GGML_TYPE_Q8_K, golden.q5_K);
}

test "q6_K dot matches the C" {
    try testing.check(ggml_vec_dot_q6_K_q8_K_generic, c.GGML_TYPE_Q6_K, c.GGML_TYPE_Q8_K, golden.q6_K);
}

test "the shared scale unpacking splits eight scales and eight mins" {
    // The twelve bytes hold sixteen six-bit fields. Getting the shuffle wrong
    // reads a min as a scale, which is a plausible-looking wrong answer rather
    // than a crash.
    var packed_scales: [12]u8 = undefined;
    for (&packed_scales, 0..) |*v, i| v.* = @intCast(i * 17 % 256);

    const utmp = unpackScalesMins(&packed_scales);
    const bytes: *const [16]u8 = @ptrCast(&utmp);

    // Every field is six bits, so none may exceed 63.
    for (bytes) |b| try std.testing.expect(b <= 63);
}
