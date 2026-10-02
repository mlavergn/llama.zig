//! Checks the three float dot products against `vec_golden.zig`, on bits.
//!
//! # Provenance
//!
//! **Not a port.** Nothing here corresponds to a file in the reference
//! checkout; it is the test side of `scripts/vec-golden`, and it mirrors the
//! input patterns in `harness/vec_golden.c` exactly. `src/ggml/cpu/quants/`
//! has the same split between a generated `golden.zig` and a `testing.zig`
//! that drives it.
//!
//! # Why this exists
//!
//! `ggml_vec_dot_f32`, `_f16` and `_bf16` are accumulating reductions, and no
//! other gate in this project can see them:
//!
//! - **`make parity-cli` and `make port` never reach them.** On a Metal
//!   machine the CPU kernels do not run during inference — measured, by
//!   putting an abort in one and watching generation finish.
//! - **`make backend-ops` compares with NMSE at `1e-7`** over uniform
//!   `[-150, 150]` inputs. That catches a kernel wrong across the domain and
//!   misses one wrong in a band of it — measured on `unary-ops.cpp`, where a
//!   `softplus` cutoff moved from 20 to 2 produced an NMSE around `8e-9` and
//!   passed.
//! - **`scripts/vecdot-prefix` and `scripts/vecdot-golden` cover only the
//!   quantized kernels.**
//!
//! So these values are the whole gate, and they are compared on **bits**. A
//! dot product right to six decimals and wrong in the last bit is a porting
//! bug, and reassociating the sum is the easiest way to cause one.
//!
//! **Which pattern catches what was measured by injection, not assumed.**
//!
//! - **`random` and `lopsided` discriminate summation order.** A plain
//!   left-to-right sum differs from the C in the last bit on exactly those
//!   two and agrees on the other five. `opposed` does not discriminate,
//!   despite looking like it should: with `y == -x` every term is `-x*x`,
//!   all the same sign, and any order gives the same answer. Mixed signs
//!   *and* mixed magnitudes are what make reassociation visible.
//! - **`skewed` is the only one that separates the *pairwise* final reduce
//!   from an ordered one**, and it exists because of an escape. With the
//!   original six, replacing `vaddvq_f32` with `@reduce(.Add, ...)` passed
//!   every pattern at both lengths — the exact trap `CLAUDE.md` warns
//!   about, invisible to the gate meant to catch it. A brute-force search
//!   over random lane vectors puts the two reductions at odds 23.5% of the
//!   time, so six patterns agreeing was luck rather than a property of the
//!   reduction. `skewed` keys the magnitude on `i % 4`, which is the lane
//!   index, and makes the disagreement deterministic.
//!
//! Keep those three. The other four are there for zero, for exact ties, for
//! cancellation, and for sign structure.
//!
//! # The patterns must stay in step with the harness
//!
//! `fillPattern` below reproduces `harness/vec_golden.c`'s `fill` and its LCG.
//! If one changes and the other does not, the goldens silently describe
//! different inputs from the ones being checked. Re-run `scripts/vec-golden`
//! after touching either.

const std = @import("std");
const impl = @import("../impl.zig");
const golden = @import("vec_golden.zig");
const c = impl.c;

/// The three kernels, by their C names. Resolved at link time from whoever
/// provides them — `vec.cpp` until the swap, `src/ggml/cpu/vec.zig` now.
/// That is the point: the same test covered both sides of it.
extern fn ggml_vec_dot_f32(n: c_int, s: *f32, bs: usize, x: [*]const f32, bx: usize, y: [*]const f32, by: usize, nrc: c_int) void;
extern fn ggml_vec_dot_f16(n: c_int, s: *f32, bs: usize, x: [*]c.ggml_fp16_t, bx: usize, y: [*]c.ggml_fp16_t, by: usize, nrc: c_int) void;
extern fn ggml_vec_dot_bf16(n: c_int, s: *f32, bs: usize, x: [*]c.ggml_bf16_t, bx: usize, y: [*]c.ggml_bf16_t, by: usize, nrc: c_int) void;

extern fn ggml_cpu_init() void;

/// Mirrors the `pattern` enum of `harness/vec_golden.c`.
const Pattern = enum { random, zeros, signs, opposed, lopsided, ties, skewed };

/// Mirrors the LCG of `harness/vec_golden.c`. The constants are the C's, and
/// the `>> 16` then `/ 32768.0` is its exact expression — a different but
/// equivalent-looking scaling would change every golden.
const Lcg = struct {
    state: u32 = 1,

    fn next(self: *Lcg) f32 {
        self.state = 1103515245 *% self.state +% 12345;
        return @as(f32, @floatFromInt(self.state >> 16)) / 32768.0 - 1.0;
    }
};

/// Mirrors `fill` in `harness/vec_golden.c`.
fn fillPattern(p: Pattern, x: []f32, y: []f32) void {
    var lcg: Lcg = .{};
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
            .skewed => {
                // Lane k of the f32 kernel's surviving accumulator holds
                // every element with `i % 4 == k`, so this is the only way to
                // shape the lanes from the input. See `harness/vec_golden.c`.
                x[i] = lcg.next() * (if (i % 4 == 3) @as(f32, 1.0) else @as(f32, 0x1p-8));
                y[i] = lcg.next();
            },
            .ties => {
                if (i == 0) {
                    x[i] = 127.0;
                    y[i] = 127.0;
                } else {
                    const half: f32 = @as(f32, @floatFromInt(@as(i64, @intCast(i % 9)) - 4)) + 0.5;
                    x[i] = half;
                    y[i] = -half;
                }
            },
        }
    }
}

/// Pulls one pattern's expected bits out of a `Dot`.
fn expectedBits(d: golden.Dot, p: Pattern) u32 {
    return switch (p) {
        .random => d.random,
        .zeros => d.zeros,
        .signs => d.signs,
        .opposed => d.opposed,
        .lopsided => d.lopsided,
        .ties => d.ties,
        .skewed => d.skewed,
    };
}

const all_patterns = [_]Pattern{ .random, .zeros, .signs, .opposed, .lopsided, .ties, .skewed };

/// Runs one kernel over every pattern at one length and compares on bits.
fn checkKernel(
    comptime which: enum { f32_, f16_, bf16_ },
    n: usize,
    want: golden.Dot,
) !void {
    const alloc = std.testing.allocator;
    const xf = try alloc.alloc(f32, n);
    defer alloc.free(xf);
    const yf = try alloc.alloc(f32, n);
    defer alloc.free(yf);

    // Only `zeros` may produce a zero result. A kernel that returned 0 for
    // everything would otherwise pass a golden set that happened to be zero,
    // which is the hole `quants/testing.zig` found by injection.
    var nonzero_seen: usize = 0;

    for (all_patterns) |p| {
        fillPattern(p, xf, yf);

        var s: f32 = 0;
        switch (which) {
            .f32_ => ggml_vec_dot_f32(@intCast(n), &s, 0, xf.ptr, 0, yf.ptr, 0, 1),
            .f16_ => {
                const xh = try alloc.alloc(c.ggml_fp16_t, n);
                defer alloc.free(xh);
                const yh = try alloc.alloc(c.ggml_fp16_t, n);
                defer alloc.free(yh);
                for (0..n) |i| {
                    xh[i] = impl.fp32ToFp16(xf[i]);
                    yh[i] = impl.fp32ToFp16(yf[i]);
                }
                ggml_vec_dot_f16(@intCast(n), &s, 0, xh.ptr, 0, yh.ptr, 0, 1);
            },
            .bf16_ => {
                const xb = try alloc.alloc(c.ggml_bf16_t, n);
                defer alloc.free(xb);
                const yb = try alloc.alloc(c.ggml_bf16_t, n);
                defer alloc.free(yb);
                for (0..n) |i| {
                    xb[i] = .{ .bits = impl.fp32ToBf16(xf[i]) };
                    yb[i] = .{ .bits = impl.fp32ToBf16(yf[i]) };
                }
                ggml_vec_dot_bf16(@intCast(n), &s, 0, xb.ptr, 0, yb.ptr, 0, 1);
            },
        }

        const got: u32 = @bitCast(s);
        const exp = expectedBits(want, p);
        if (got != exp) {
            std.debug.print(
                "{s} n={d} pattern={s}: got 0x{X:0>8} ({d}), want 0x{X:0>8}\n",
                .{ @tagName(which), n, @tagName(p), got, s, exp },
            );
            return error.GoldenMismatch;
        }
        if (exp != 0) nonzero_seen += 1;
    }

    // Five of the six patterns must be non-zero, by construction.
    try std.testing.expectEqual(@as(usize, all_patterns.len - 1), nonzero_seen);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the float dot products match the C, bit for bit" {
    // `ggml_vec_dot_f16` reads the f16 conversion tables, which are empty
    // until this runs. The quant goldens were silently all-zero for exactly
    // this reason before it was added there.
    ggml_cpu_init();

    try checkKernel(.f32_, golden.nelem, golden.dot_f32);
    try checkKernel(.f16_, golden.nelem, golden.dot_f16);
    try checkKernel(.bf16_, golden.nelem, golden.dot_bf16);
}

test "the float dot products match the C on a length with a scalar tail" {
    // `nelem_tail` is not a whole number of vector steps, so the remainder
    // loop runs — and it accumulates in a different type from the body: f32
    // for the f32 kernel, double for the other two. A port that gets the
    // vector body right and the tail wrong passes the test above and fails
    // this one.
    ggml_cpu_init();

    try checkKernel(.f32_, golden.nelem_tail, golden.dot_f32_tail);
    try checkKernel(.f16_, golden.nelem_tail, golden.dot_f16_tail);
    try checkKernel(.bf16_, golden.nelem_tail, golden.dot_bf16_tail);
}

test "the float dot products match the C on short lengths" {
    // The f32 kernel's leftover loop is not compiled as it reads: the
    // reference vectorizes it into rounded groups of four products and fuses
    // only the last `t % 4`. `nelem_tail`'s seven-element tail cannot tell
    // fused, unfused and split apart -- measured -- so these lengths exist
    // to. See `vectorizedTail` in `vec.zig`.
    ggml_cpu_init();

    inline for (golden.short_lens) |n| {
        const suf = std.fmt.comptimePrint("_n{d}", .{n});
        try checkKernel(.f32_, n, @field(golden, "dot_f32" ++ suf));
        try checkKernel(.f16_, n, @field(golden, "dot_f16" ++ suf));
        try checkKernel(.bf16_, n, @field(golden, "dot_bf16" ++ suf));
    }
}
