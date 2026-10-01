//! The `vec.h` inlines the `ops.cpp` kernels call.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/vec.h` at v0.3.0 (`c1d0e7a00`).
//!
//! **These are `inline static` in the C, so they have no symbols** and
//! `cpu/vec.zig` — which ports `vec.cpp`'s *exported* contract — does not
//! provide them. They are compiled into each caller, and here that caller is
//! the `ops.cpp` port.
//!
//! Every one of them has an `#if defined(__AVX2__)` or similar vector arm that
//! is **dead on this target**, so what is ported is the scalar loop the C
//! falls through to. Where that is the case the doc comment says so, per the
//! `#if`-arm rule in `CLAUDE.md`.

const std = @import("std");
const impl = @import("../../impl.zig");

const c = impl.c;

/// Ports `ggml_vec_add_f32` (vec.h:89 @c1d0e7a00), the scalar fallthrough —
/// its only other arm is `__AVX2__`, which this target does not compile.
pub inline fn add_f32(n: i64, z: [*]f32, x: [*]const f32, y: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) z[i] = x[i] + y[i];
}

/// Ports `ggml_vec_add_f16` (vec.h:104 @c1d0e7a00).
pub inline fn add_f16(n: i64, z: [*]c.ggml_fp16_t, x: [*]const c.ggml_fp16_t, y: [*]const c.ggml_fp16_t) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        z[i] = impl.fp32ToFp16(impl.fp16ToFp32(x[i]) + impl.fp16ToFp32(y[i]));
    }
}

/// Ports `ggml_vec_add1_f32` (vec.h:109 @c1d0e7a00).
pub inline fn add1_f32(n: i64, z: [*]f32, x: [*]const f32, v: f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) z[i] = x[i] + v;
}

/// Ports `ggml_vec_acc_f32` (vec.h:110 @c1d0e7a00).
pub inline fn acc_f32(n: i64, y: [*]f32, x: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) y[i] += x[i];
}

/// Ports `ggml_vec_acc1_f32` (vec.h:111 @c1d0e7a00).
pub inline fn acc1_f32(n: i64, y: [*]f32, v: f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) y[i] += v;
}

/// Ports `ggml_vec_set_f32` (vec.h:118 @c1d0e7a00).
pub inline fn set_f32(n: i64, x: [*]f32, v: f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) x[i] = v;
}

/// Ports `ggml_vec_cpy_f32` (vec.h:119 @c1d0e7a00).
pub inline fn cpy_f32(n: i64, y: [*]f32, x: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) y[i] = x[i];
}

// -----------------------------------------------------------------------------
// Reductions
//
// `ggml_vec_sum_f32` and `ggml_vec_max_f32` have an Accelerate arm, and
// `GGML_USE_ACCELERATE` is defined for this target, so the vDSP call is what
// compiles and the `#ifndef` scalar loop is dead. The `_ggf` variants have no
// Accelerate arm at all -- that is the whole reason they exist alongside
// `ggml_vec_sum_f32`, which on this target cannot accumulate in `double`.

/// Mirrors `vDSP_Length`, the `unsigned long` element count.
const Length = c_ulong;
const Stride = isize;

extern fn vDSP_sve(a: [*]const f32, ia: Stride, s: *f32, n: Length) void;
extern fn vDSP_maxv(a: [*]const f32, ia: Stride, s: *f32, n: Length) void;

/// Ports `ggml_vec_sum_f32` (vec.h:1495 @c1d0e7a00), the `GGML_USE_ACCELERATE`
/// arm.
///
/// **Not interchangeable with `sum_f32_ggf`.** This one sums in `f32` through
/// vDSP; that one sums in `double`. `mean` and `sum_rows` call this, `sum`
/// calls the other, and swapping them changes the result.
pub inline fn sum_f32(n: i64, s: *f32, x: [*]const f32) void {
    vDSP_sve(x, 1, s, @intCast(n));
}

/// Ports `ggml_vec_sum_f32_ggf` (vec.h:1517 @c1d0e7a00). Accumulates in
/// `ggml_float`, a `double`.
pub inline fn sum_f32_ggf(n: i64, s: *f64, x: [*]const f32) void {
    var sum: f64 = 0.0;
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) sum += @as(f64, x[i]);
    s.* = sum;
}

/// Ports `ggml_vec_sum_f16_ggf` (vec.h:1525 @c1d0e7a00). Despite the `_ggf`,
/// this one accumulates in `f32`; only the `f32` variant above widens.
pub inline fn sum_f16_ggf(n: i64, s: *f32, x: [*]const c.ggml_fp16_t) void {
    var sum: f32 = 0.0;
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) sum += impl.fp16ToFp32(x[i]);
    s.* = sum;
}

/// Ports `ggml_vec_sum_bf16_ggf` (vec.h:1533 @c1d0e7a00).
pub inline fn sum_bf16_ggf(n: i64, s: *f32, x: [*]const c.ggml_bf16_t) void {
    var sum: f32 = 0.0;
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) sum += impl.bf16ToFp32(x[i].bits);
    s.* = sum;
}

/// Ports `ggml_vec_cumsum_f32` (vec.h:1507 @c1d0e7a00).
pub inline fn cumsum_f32(n: i64, y: [*]f32, x: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        y[i] = if (i == 0) x[i] else y[i - 1] + x[i];
    }
}

/// Ports `ggml_vec_max_f32` (vec.h:1541 @c1d0e7a00), the
/// `GGML_USE_ACCELERATE` arm.
pub inline fn max_f32(n: i64, s: *f32, x: [*]const f32) void {
    vDSP_maxv(x, 1, s, @intCast(n));
}

/// Ports `ggml_vec_argmax_f32` (vec.h:1558 @c1d0e7a00).
///
/// Note the C's tie-breaking: it reassigns `max` first and then compares
/// `max == x[i]`, so the **last** index attaining the maximum wins, not the
/// first. Written the same way here rather than as the `>` test it looks like.
pub inline fn argmax_f32(n: i64, s: *i32, x: [*]const f32) void {
    var max: f32 = -std.math.inf(f32);
    var idx: i32 = 0;
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        max = @max(max, x[i]);
        if (max == x[i]) idx = @intCast(i);
    }
    s.* = idx;
}
