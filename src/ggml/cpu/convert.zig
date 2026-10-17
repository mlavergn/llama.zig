//! Row-at-a-time conversions between the float formats, and the lookup tables
//! `ggml_cpu_init` fills.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (v0.3.0, `c1d0e7a00`),
//! the tables at line 78 and the converters at line 3440. Each declaration
//! names the C it replaces and the line it began at.
//!
//! # SIMD only on x86
//!
//! The C bodies are mostly `#if defined(__F16C__)` and `#elif
//! defined(__riscv_zvfh)` blocks over a scalar tail. On ARM none of them
//! compile and only the tail survives. On x86-64-v3 the F16C blocks do, and
//! they are **not** the tail's arithmetic for every input: `vcvtps2ph` keeps a
//! NaN's payload where the software narrowing does not, and `vcvtph2ps`
//! quiets a signalling NaN where the table the tail reads does not. So the
//! blocks call the instructions -- LLVM's `vcvtps2ph` intrinsic, and the
//! vector `fpext` that lowers to `vcvtph2ps` -- and the tails stay software.
//! See the note on `cpuFp32ToFp16` below.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("../impl.zig");
const c = impl.c;

/// Whether this build compiles the C's `#if defined(__F16C__)` blocks.
const has_f16c = builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .f16c);

// -----------------------------------------------------------------------------
// The precomputed tables
//
// All three are filled by `ggml_cpu_init`. They are `export` because the
// still-C++ half of the backend reaches them through `simd-mappings.h`:
// `GGML_CPU_UE4M3_TO_FP32` indexes `ggml_table_f32_ue4m3` directly on NEON.

/// Ports `ggml_table_f32_f16` (ggml-cpu.c:80 @c1d0e7a00). 256 KB.
pub export var ggml_table_f32_f16: [1 << 16]f32 = @splat(0);

/// Ports `ggml_table_f32_e8m0_half` (ggml-cpu.c:83 @c1d0e7a00). 1 KB.
pub export var ggml_table_f32_e8m0_half: [1 << 8]f32 = @splat(0);

/// Ports `ggml_table_f32_ue4m3` (ggml-cpu.c:86 @c1d0e7a00). 1 KB.
pub export var ggml_table_f32_ue4m3: [1 << 8]f32 = @splat(0);

/// The GELU tables, still C++ in `ggml-cpu/vec.cpp` but filled here because
/// `ggml_cpu_init` is what fills them (vec.h:62, vec.h:65).
// Defined by `cpu/vec.zig`, which ports `vec.cpp` where the C declares them.
// They were `extern var` against `vec.o` until that file was ported.
const vec = @import("vec.zig");
const ggml_table_gelu_f16 = &vec.ggml_table_gelu_f16;
const ggml_table_gelu_quick_f16 = &vec.ggml_table_gelu_quick_f16;

// -----------------------------------------------------------------------------
// Scalar conversions
//
// `simd-mappings.h` resolves `GGML_CPU_FP16_TO_FP32` and
// `GGML_CPU_FP32_TO_FP16` differently on this target, and the asymmetry is
// easy to miss, so each direction is named separately here.

/// Ports `GGML_CPU_FP16_TO_FP32` (simd-mappings.h:44 @c1d0e7a00), the NEON
/// arm, and `ggml_lookup_fp16_to_fp32` (simd-mappings.h:147 @c1d0e7a00),
/// every other.
///
/// On ARM the C widens through `__fp16`, a hardware conversion; elsewhere it
/// reads `ggml_table_f32_f16`, which `initTables` fills from the software
/// routine. Widening is exact both ways for every number. They differ only
/// on a signalling NaN, which the hardware quiets and the table does not --
/// so the arm matters, and it is the one the C takes.
pub inline fn cpuFp16ToFp32(h: c.ggml_fp16_t) f32 {
    if (comptime builtin.cpu.arch == .aarch64) return @floatCast(@as(f16, @bitCast(h)));
    return ggml_table_f32_f16[h];
}

/// Ports `GGML_CPU_FP32_TO_FP16` (simd-mappings.h:157 @c1d0e7a00).
///
/// **Not** the hardware narrowing, despite the direction above. The NEON block
/// in `simd-mappings.h` defines `GGML_CPU_COMPUTE_FP32_TO_FP16` but never
/// promotes it to `GGML_CPU_FP32_TO_FP16`, so the fallback at line 157 wins
/// and the C ends up in `ggml_compute_fp32_to_fp16` -- the software routine
/// `src/ggml/impl.zig` already ports. Substituting `@floatCast` here would
/// change the bytes for subnormals and for NaN payloads.
pub inline fn cpuFp32ToFp16(f: f32) c.ggml_fp16_t {
    return impl.fp32ToFp16(f);
}

// -----------------------------------------------------------------------------
// Row converters
//
// These are what the traits table's `from_float` slots point at for the
// non-quantised types.

/// Ports `ggml_cpu_fp32_to_fp32` (ggml-cpu.c:3440 @c1d0e7a00).
///
/// Parameters:
/// - `x`: source row.
/// - `y`: destination row, `n` floats.
/// - `n`: element count.
pub export fn ggml_cpu_fp32_to_fp32(x: [*c]const f32, y: [*c]f32, n: i64) void {
    if (n <= 0) return;
    @memcpy(y[0..@intCast(n)], x[0..@intCast(n)]);
}

extern fn @"llvm.x86.vcvtps2ph.256"(@Vector(8, f32), i32) @Vector(8, i16);
extern fn @"llvm.x86.vcvtps2ph.128"(@Vector(4, f32), i32) @Vector(8, i16);

/// `_MM_FROUND_TO_NEAREST_INT`, the rounding immediate the C passes.
const round_nearest: i32 = 0;

/// Ports `ggml_cpu_fp32_to_fp16` (ggml-cpu.c:3444 @c1d0e7a00): with F16C,
/// blocks of eight and then four through `vcvtps2ph`, then the software tail.
pub export fn ggml_cpu_fp32_to_fp16(x: [*c]const f32, y: [*c]c.ggml_fp16_t, n: i64) void {
    const nn: usize = @intCast(n);
    var i: usize = 0;
    if (comptime has_f16c) {
        while (i + 7 < nn) : (i += 8) {
            const v: @Vector(8, f32) = x[i..][0..8].*;
            y[i..][0..8].* = @bitCast(@"llvm.x86.vcvtps2ph.256"(v, round_nearest));
        }
        while (i + 3 < nn) : (i += 4) {
            const v: @Vector(4, f32) = x[i..][0..4].*;
            const h: [8]u16 = @bitCast(@"llvm.x86.vcvtps2ph.128"(v, round_nearest));
            y[i..][0..4].* = h[0..4].*;
        }
    }
    while (i < nn) : (i += 1) y[i] = cpuFp32ToFp16(x[i]);
}

/// Ports `ggml_cpu_fp16_to_fp32` (ggml-cpu.c:3477 @c1d0e7a00): with F16C,
/// blocks of eight and then four through `vcvtph2ps`, then the tail through
/// `GGML_CPU_FP16_TO_FP32`.
pub export fn ggml_cpu_fp16_to_fp32(x: [*c]const c.ggml_fp16_t, y: [*c]f32, n: i64) void {
    const nn: usize = @intCast(n);
    var i: usize = 0;
    if (comptime has_f16c) {
        while (i + 7 < nn) : (i += 8) {
            const h: @Vector(8, f16) = @bitCast(@as(@Vector(8, u16), x[i..][0..8].*));
            y[i..][0..8].* = @as(@Vector(8, f32), @floatCast(h));
        }
        while (i + 3 < nn) : (i += 4) {
            const h: @Vector(4, f16) = @bitCast(@as(@Vector(4, u16), x[i..][0..4].*));
            y[i..][0..4].* = @as(@Vector(4, f32), @floatCast(h));
        }
    }
    while (i < nn) : (i += 1) y[i] = cpuFp16ToFp32(x[i]);
}

/// Ports `ggml_cpu_fp32_to_bf16` (ggml-cpu.c:3531 @c1d0e7a00).
pub export fn ggml_cpu_fp32_to_bf16(x: [*c]const f32, y: [*c]c.ggml_bf16_t, n: i64) void {
    for (0..@intCast(n)) |i| y[i] = .{ .bits = impl.fp32ToBf16(x[i]) };
}

/// Ports `ggml_cpu_bf16_to_fp32` (ggml-cpu.c:3545 @c1d0e7a00).
///
/// The `__AVX2__` arm widens eight at a time by shifting each `u16` left 16
/// bits, which is exactly what the scalar `GGML_BF16_TO_FP32` does to one, so
/// one loop covers both arms.
pub export fn ggml_cpu_bf16_to_fp32(x: [*c]const c.ggml_bf16_t, y: [*c]f32, n: i64) void {
    for (0..@intCast(n)) |i| y[i] = impl.bf16ToFp32(x[i].bits);
}

/// Ports `ggml_cpu_fp32_to_i32` (ggml-cpu.c:3538 @c1d0e7a00).
///
/// The C writes `y[i] = x[i]`, a float-to-int conversion: truncation toward
/// zero, and undefined once the value leaves `int32_t`'s range. `impl.truncTo`
/// saturates instead of trapping, which is what Zig needs and what the
/// hardware does anyway.
pub export fn ggml_cpu_fp32_to_i32(x: [*c]const f32, y: [*c]i32, n: i64) void {
    for (0..@intCast(n)) |i| y[i] = impl.truncTo(i32, x[i]);
}

// -----------------------------------------------------------------------------
// Table initialisation

/// Ports the table-filling block of `ggml_cpu_init` (ggml-cpu.c:3823 @c1d0e7a00).
///
/// Called once, under the critical section `ggml_cpu_init` holds. Fills the
/// f16 fan-out table, both GELU tables, and the two 256-entry
/// micro-exponent tables.
pub fn initTables() void {
    for (0..1 << 16) |i| {
        const f = impl.fp16ToFp32(@intCast(i));
        ggml_table_f32_f16[i] = f;
        ggml_table_gelu_f16[i] = cpuFp32ToFp16(gelu(f));
        ggml_table_gelu_quick_f16[i] = cpuFp32ToFp16(geluQuick(f));
    }

    for (0..1 << 8) |i| {
        ggml_table_f32_e8m0_half[i] = impl.e8m0ToFp32Half(@intCast(i));
    }

    for (0..1 << 8) |i| {
        ggml_table_f32_ue4m3[i] = impl.ue4m3ToFp32(@intCast(i));
    }
}

/// Ports `GELU_COEF_A` (vec.h:963 @c1d0e7a00).
const gelu_coef_a: f32 = 0.044715;

/// Ports `GELU_QUICK_COEF` (vec.h:964 @c1d0e7a00).
const gelu_quick_coef: f32 = -1.702;

/// Ports `SQRT_2_OVER_PI` (vec.h:965 @c1d0e7a00).
const sqrt_2_over_pi: f32 = 0.79788456080286535587989211986876;

/// Ports `ggml_gelu_f32` (vec.h:968 @c1d0e7a00).
///
/// `static inline` in a header the port cannot import, and used only to fill
/// the table above, so it lives here rather than being reached across.
fn gelu(x: f32) f32 {
    return 0.5 * x * (1.0 + std.math.tanh(sqrt_2_over_pi * x * (1.0 + gelu_coef_a * x * x)));
}

/// Ports `ggml_gelu_quick_f32` (vec.h:1017 @c1d0e7a00).
fn geluQuick(x: f32) f32 {
    return x * (1.0 / (1.0 + @exp(gelu_quick_coef * x)));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "fp16 round trip through the two directions the C uses" {
    // Widening is exact, so every half must survive the round trip. This also
    // pins the asymmetry: the two directions come from different routines.
    var h: u16 = 0;
    while (true) {
        const exponent = (h >> 10) & 0x1F;
        const mantissa = h & 0x3FF;
        const is_nan = exponent == 0x1F and mantissa != 0;
        if (!is_nan) {
            try std.testing.expectEqual(h, cpuFp32ToFp16(cpuFp16ToFp32(h)));
        }
        if (h == 0xFFFF) break;
        h += 1;
    }
}

test "hardware and software widening agree" {
    // `cpuFp16ToFp32` is the hardware path and `impl.fp16ToFp32` the software
    // one. The C mixes both freely, so they had better not differ.
    var h: u16 = 0;
    while (true) {
        const exponent = (h >> 10) & 0x1F;
        const mantissa = h & 0x3FF;
        if (!(exponent == 0x1F and mantissa != 0)) {
            try std.testing.expectEqual(impl.fp16ToFp32(h), cpuFp16ToFp32(h));
        }
        if (h == 0xFFFF) break;
        h += 1;
    }
}

test "fp32 to i32 truncates toward zero and saturates" {
    const x = [_]f32{ 1.9, -1.9, 0.0, -0.5, 3.0e10, -3.0e10 };
    var y: [x.len]i32 = undefined;
    ggml_cpu_fp32_to_i32(&x, &y, x.len);

    try std.testing.expectEqual(@as(i32, 1), y[0]);
    try std.testing.expectEqual(@as(i32, -1), y[1]);
    try std.testing.expectEqual(@as(i32, 0), y[2]);
    try std.testing.expectEqual(@as(i32, 0), y[3]);
    try std.testing.expectEqual(std.math.maxInt(i32), y[4]);
    try std.testing.expectEqual(std.math.minInt(i32), y[5]);
}

test "fp32 to fp32 copies and tolerates an empty row" {
    const x = [_]f32{ 1.0, 2.0, 3.0 };
    var y: [3]f32 = @splat(0);
    ggml_cpu_fp32_to_fp32(&x, &y, 3);
    try std.testing.expectEqualSlices(f32, &x, &y);

    // `@memcpy` on a zero-length slice of a null pointer would be a problem;
    // the C's `memcpy(y, x, 0)` is not. Guarding is cheaper than proving no
    // caller does it.
    ggml_cpu_fp32_to_fp32(null, null, 0);
}

test "bf16 keeps the top half of the float" {
    const x = [_]f32{ 1.0, -2.5, 0.0 };
    var y: [3]c.ggml_bf16_t = undefined;
    var back: [3]f32 = undefined;

    ggml_cpu_fp32_to_bf16(&x, &y, 3);
    ggml_cpu_bf16_to_fp32(&y, &back, 3);

    // These three are exactly representable, so the round trip is lossless.
    try std.testing.expectEqualSlices(f32, &x, &back);
}
