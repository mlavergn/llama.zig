//! The fixture the generic dot products are tested through.
//!
//! # Provenance
//!
//! **Not a port.** This is our own code. It reproduces the input generation in
//! `harness/vecdot_golden.c` so the values in `golden.zig` can be compared
//! against, and it has to match that harness exactly -- the LCG, the patterns,
//! the element count, and which quantizer each operand goes through.
//!
//! # Shared by both quant translation units
//!
//! `ggml-cpu/quants.c` and `ggml-cpu/arch/arm/quants.c` compute the same
//! products by different routes, so they are checked against the same inputs
//! and differ only in which golden module they compare to. `check` and
//! `checkRow` take the golden row as `anytype` for that reason.
//!
//! # Why this is the whole gate for the generic kernels
//!
//! Nothing on this target calls `ggml_vec_dot_*_generic`. `arch-fallback.h`
//! renames nothing from `quants.c` on ARM, and `arch/arm/quants.c` supplies
//! every real entry point, so the generic names are exported and unreachable.
//! `test-backend-ops` cannot see them; neither can token parity. If these
//! comparisons are wrong, nothing else will notice.
//!
//! So the comparison is on **raw bits**, not a tolerance. A dot product right
//! to six decimals and wrong in the last bit is a porting bug, and the point of
//! a golden captured from the C is to catch exactly that.

const std = @import("std");
const impl = @import("../../impl.zig");
const golden = @import("golden.zig");
const c = impl.c;

/// Elements per dot product, from the generator. Divisible by every block size
/// in play: 256 (`QK_K`), 128 (`QK1_0`), 64 (`QK2_0`, `QK_NVFP4`) and 32.
pub const nelem = golden.nelem;

/// The input patterns, in the generator's order.
///
/// Only `zeros` should produce a zero result. An earlier set had two patterns
/// that collapsed to zero for almost every kernel, which a stub that did
/// nothing but write 0 would have passed.
pub const Pattern = enum {
    /// The common path.
    random,
    /// Both operands all-zero: every symmetric quantizer's divide-by-zero
    /// guard, and a dot product of nothing.
    zeros,
    /// Fixed signs, asymmetric magnitudes, coprime periods -- sign handling in
    /// the nibble unpacking, without the sum cancelling.
    signs,
    /// `y = -x`, so the sum cancels to near zero while the terms stay large.
    opposed,
    /// Operands four orders of magnitude apart, with x's fp16 delta down in the
    /// subnormals. A swapped scale shows up here and nowhere else.
    lopsided,
    /// Exact halves at the quantization step. Element 0 is 127 so `amax` is
    /// 127, `d` is 1 and `id` is 1, making every other element an exact tie.
    ///
    /// No pseudorandom input hits a tie, so without this pattern
    /// round-half-to-even and round-half-away-from-zero are indistinguishable
    /// -- measured: replacing `fcvtns` with `@round` passed every other one.
    ties,

    /// The golden for this pattern, from either module's `Dot` or `Row`.
    fn pick(self: Pattern, g: anytype) @TypeOf(g.random) {
        return switch (self) {
            .random => g.random,
            .zeros => g.zeros,
            .signs => g.signs,
            .opposed => g.opposed,
            .lopsided => g.lopsided,
            .ties => g.ties,
        };
    }
};

/// Ports the generator's `lcg_next` (vecdot_golden.c:133).
///
/// The exact constants matter: they decide the inputs, and the goldens were
/// captured from them.
const Lcg = struct {
    state: u32 = 1,

    fn next(self: *Lcg) f32 {
        self.state = 1103515245 *% self.state +% 12345;
        return (@as(f32, @floatFromInt(self.state >> 16)) / 32768.0) - 1.0;
    }
};

/// Ports `fill`, which sets both operands for one pattern
/// (vecdot_golden.c:141).
///
/// Parameters:
/// - `p`: which pattern.
/// - `x`, `y`: destination rows, `nelem` floats each.
fn fill(p: Pattern, x: []f32, y: []f32) void {
    var lcg = Lcg{};
    for (0..x.len) |i| {
        switch (p) {
            .random => {
                x[i] = lcg.next();
                y[i] = lcg.next();
            },
            .zeros => {
                x[i] = 0.0;
                y[i] = 0.0;
            },
            .signs => {
                x[i] = if (i % 2 != 0) 1.0 else -0.5;
                y[i] = if (i % 3 != 0) 0.75 else -1.0;
            },
            .opposed => {
                x[i] = lcg.next();
                y[i] = -x[i];
            },
            .lopsided => {
                x[i] = lcg.next() * 1e-4;
                y[i] = lcg.next() * 4.0;
            },
            .ties => {
                if (i == 0) {
                    x[i] = 127.0;
                    y[i] = 127.0;
                } else {
                    const half = @as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) + 0.5;
                    x[i] = half;
                    y[i] = -half;
                }
            },
        }
    }
}

/// The signature every `ggml_vec_dot_*` has.
pub const DotFn = *const fn (
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void;

extern fn quantize_row_q8_0_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
extern fn quantize_row_q8_1_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
extern fn quantize_row_q8_K_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;

/// Ports `quant_y`, which quantizes the right operand
/// (vecdot_golden.c:225).
///
/// The reference quantizer, not `ggml_quantize_chunk`: that has no case for
/// `q8_1` or `q8_K` at all, and one path for all three is less to get wrong.
fn quantY(t: c.enum_ggml_type, src: [*c]const f32, dst: *anyopaque, n: i64) !void {
    switch (t) {
        c.GGML_TYPE_Q8_0 => quantize_row_q8_0_ref(src, dst, n),
        c.GGML_TYPE_Q8_1 => quantize_row_q8_1_ref(src, dst, n),
        c.GGML_TYPE_Q8_K => quantize_row_q8_K_ref(src, dst, n),
        else => return error.UnexpectedVecDotType,
    }
}

extern fn ggml_quantize_chunk(
    t: c.enum_ggml_type,
    src: [*c]const f32,
    dst: ?*anyopaque,
    start: i64,
    nrows: i64,
    n_per_row: i64,
    imatrix: [*c]const f32,
) usize;

/// Runs one kernel against its goldens, on every pattern.
///
/// Parameters:
/// - `dot`: the kernel under test.
/// - `xt`: the left operand's type.
/// - `yt`: the right operand's type, which is `xt`'s `vec_dot_type`.
/// - `g`: the golden row for this kernel.
///
/// Return: nothing; fails the test on the first pattern that differs, naming
/// the pattern and printing both values in decimal and in hex.
pub fn check(dot: DotFn, xt: c.enum_ggml_type, yt: c.enum_ggml_type, g: anytype) !void {
    // The NEON `nvfp4` kernel reads `ggml_table_f32_ue4m3`, which only
    // `ggml_cpu_init` fills. A test binary never runs a graph, so nothing else
    // would call it, and every scale would be zero. Idempotent.
    ggml_cpu_init();

    // No kernel's goldens may be entirely zero. That is not a style rule: the
    // `nvfp4` ARM goldens *were* all zero once, captured before the harness
    // called `ggml_cpu_init`, and a kernel returning nothing would have passed
    // all six patterns.
    try expectNotAllZero(g);

    // Sized for the widest quantized row plus slack; a block layout that grew
    // would overrun otherwise, and this is a test, so the memory is free.
    var xf: [nelem]f32 = undefined;
    var yf: [nelem]f32 = undefined;
    var xq: [nelem * 4]u8 align(8) = undefined;
    var yq: [nelem * 4]u8 align(8) = undefined;

    // All-ones, and passed for every type rather than only the ones
    // `ggml_quantize_requires_imatrix` names -- matching the generator. What
    // the test needs from x is deterministic bytes; how they were arrived at
    // does not matter.
    const imatrix: [nelem]f32 = @splat(1.0);

    inline for (comptime std.enums.values(Pattern)) |p| {
        fill(p, &xf, &yf);

        @memset(&xq, 0);
        @memset(&yq, 0);

        _ = ggml_quantize_chunk(xt, &xf, &xq, 0, 1, nelem, &imatrix);
        try quantY(yt, &yf, &yq, nelem);

        var s: f32 = 0;
        dot(nelem, &s, 0, &xq, 0, &yq, 0, 1);

        const want = p.pick(g);
        const got: u32 = @bitCast(s);
        if (got != want) {
            std.debug.print(
                "{s}: want 0x{X:0>8} ({d}), got 0x{X:0>8} ({d})\n",
                .{ @tagName(p), want, @as(f32, @bitCast(want)), got, s },
            );
            return error.GoldenMismatch;
        }
    }
}

/// The signature every `quantize_row_*` has.
pub const RowFn = *const fn (x: [*c]const f32, y: ?*anyopaque, k: i64) callconv(.c) void;

extern fn ggml_row_size(t: c.enum_ggml_type, ne: i64) usize;
extern fn ggml_cpu_init() void;

/// Runs one row quantizer against its goldens, on every pattern.
///
/// Parameters:
/// - `quant`: the quantizer under test.
/// - `t`: the type it writes, for sizing the output.
/// - `g`: the golden row for this quantizer.
///
/// Return: nothing; fails on the first pattern whose output checksum differs.
pub fn checkRow(quant: RowFn, t: c.enum_ggml_type, g: anytype) !void {
    ggml_cpu_init();
    try expectNotAllZero(g);

    var xf: [nelem]f32 = undefined;
    var yf: [nelem]f32 = undefined;
    var out: [nelem * 4]u8 align(8) = undefined;

    inline for (comptime std.enums.values(Pattern)) |p| {
        fill(p, &xf, &yf);
        @memset(&out, 0);

        quant(&xf, &out, nelem);

        const bytes = ggml_row_size(t, nelem);
        const got = fnv1a(out[0..bytes]);
        const want = p.pick(g);
        if (got != want) {
            std.debug.print(
                "{s}: want 0x{X:0>16}, got 0x{X:0>16}\n",
                .{ @tagName(p), want, got },
            );
            return error.GoldenMismatch;
        }
    }
}

/// Fails when every pattern's golden is zero.
///
/// A golden row of all zeros means the capture was broken, not that the kernel
/// computes zero: `zeros` is the only pattern that should ever produce it.
fn expectNotAllZero(g: anytype) !void {
    var any = false;
    inline for (comptime std.enums.values(Pattern)) |p| {
        if (p.pick(g) != 0) any = true;
    }
    if (!any) {
        std.debug.print("every golden for this kernel is zero -- the capture is broken\n", .{});
        return error.GoldensAllZero;
    }
}

/// FNV-1a, matching the generator's.
fn fnv1a(bytes: []const u8) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (bytes) |b| {
        h ^= b;
        h *%= 0x100000001b3;
    }
    return h;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the LCG matches the generator's first values" {
    // If this drifts, every golden comparison silently tests different inputs
    // and passes or fails for the wrong reason.
    var lcg = Lcg{};
    // 1103515245*1 + 12345 = 1103527590; >>16 = 16838; /32768 - 1
    try std.testing.expectEqual(
        @as(f32, 16838.0 / 32768.0 - 1.0),
        lcg.next(),
    );
}

test "only the zeros pattern is all-zero on both sides" {
    var x: [nelem]f32 = undefined;
    var y: [nelem]f32 = undefined;

    fill(.zeros, &x, &y);
    for (x, y) |xi, yi| {
        try std.testing.expectEqual(@as(f32, 0.0), xi);
        try std.testing.expectEqual(@as(f32, 0.0), yi);
    }

    for ([_]Pattern{ .random, .signs, .opposed, .lopsided, .ties }) |p| {
        fill(p, &x, &y);
        var any_x = false;
        var any_y = false;
        for (x, y) |xi, yi| {
            if (xi != 0.0) any_x = true;
            if (yi != 0.0) any_y = true;
        }
        try std.testing.expect(any_x);
        try std.testing.expect(any_y);
    }
}

test "the goldens are not mostly zero" {
    // The gate this file is: 25 kernels times 5 patterns. If most of the
    // expected values were zero, a kernel that wrote nothing but 0 would pass,
    // which is how the first version of the generator behaved.
    const rows = [_]golden.Dot{
        golden.q1_0,  golden.q2_0,    golden.q4_0,   golden.q4_1,   golden.q5_0,
        golden.q5_1,  golden.q8_0,    golden.mxfp4,  golden.nvfp4,  golden.q2_K,
        golden.q3_K,  golden.q4_K,    golden.q5_K,   golden.q6_K,   golden.tq1_0,
        golden.tq2_0, golden.iq2_xxs, golden.iq2_xs, golden.iq2_s,  golden.iq3_xxs,
        golden.iq3_s, golden.iq1_s,   golden.iq1_m,  golden.iq4_nl, golden.iq4_xs,
    };

    var zeros: usize = 0;
    var total: usize = 0;
    for (rows) |g| {
        for (comptime std.enums.values(Pattern)) |p| {
            total += 1;
            if (p.pick(g) == 0) zeros += 1;
        }
    }

    try std.testing.expectEqual(@as(usize, 25 * 6), total);
    // 25 of the zeros are the `zeros` pattern, which is correct. Allow a
    // handful more for kernels whose scale genuinely underflows on `lopsided`,
    // and for `ties`, where y = -x makes cancellation the expected answer.
    try std.testing.expect(zeros <= 60);
}
