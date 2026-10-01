//! Row-at-a-time conversions between the float formats, and the lookup tables
//! `ggml_cpu_init` fills.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (v0.3.0, `c1d0e7a00`),
//! the tables at line 78 and the converters at line 3440. Each declaration
//! names the C it replaces and the line it began at.
//!
//! # No SIMD here
//!
//! The C bodies are mostly `#if defined(__F16C__)` and `#elif
//! defined(__riscv_zvfh)` blocks over a scalar tail. On this target none of
//! them are compiled: only the tail survives, so that is all this file
//! contains. The narrowing is a bit-manipulation routine either way, not a
//! hardware instruction -- see the note on `cpuFp32ToFp16` below.

const std = @import("std");
const impl = @import("../impl.zig");
const c = impl.c;

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

/// Ports `GGML_CPU_FP16_TO_FP32` (simd-mappings.h:44 @c1d0e7a00), the NEON arm.
///
/// On ARM the C widens through `__fp16`, a hardware conversion, rather than
/// the bit-manipulation routine in `ggml-impl.h`. Widening is exact in both,
/// so this is a speed choice rather than a numeric one -- but it is the one
/// the C makes.
pub inline fn cpuFp16ToFp32(h: c.ggml_fp16_t) f32 {
    return @floatCast(@as(f16, @bitCast(h)));
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

/// Ports `ggml_cpu_fp32_to_fp16` (ggml-cpu.c:3444 @c1d0e7a00).
pub export fn ggml_cpu_fp32_to_fp16(x: [*c]const f32, y: [*c]c.ggml_fp16_t, n: i64) void {
    for (0..@intCast(n)) |i| y[i] = cpuFp32ToFp16(x[i]);
}

/// Ports `ggml_cpu_fp16_to_fp32` (ggml-cpu.c:3477 @c1d0e7a00).
pub export fn ggml_cpu_fp16_to_fp32(x: [*c]const c.ggml_fp16_t, y: [*c]f32, n: i64) void {
    for (0..@intCast(n)) |i| y[i] = cpuFp16ToFp32(x[i]);
}

/// Ports `ggml_cpu_fp32_to_bf16` (ggml-cpu.c:3531 @c1d0e7a00).
pub export fn ggml_cpu_fp32_to_bf16(x: [*c]const f32, y: [*c]c.ggml_bf16_t, n: i64) void {
    for (0..@intCast(n)) |i| y[i] = .{ .bits = impl.fp32ToBf16(x[i]) };
}

/// Ports `ggml_cpu_bf16_to_fp32` (ggml-cpu.c:3545 @c1d0e7a00).
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
