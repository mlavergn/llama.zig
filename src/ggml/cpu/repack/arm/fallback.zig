//! The seven `arch/arm/repack.cpp` entry points that delegate to the
//! generic on the baseline (M1) target -- five of them only there.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` at v0.4.1
//! (`b29c606e2`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # Why these are one line each
//!
//! Each has a NEON body behind `__ARM_FEATURE_SVE`,
//! `__ARM_FEATURE_MATMUL_INT8` or `__ARM_FEATURE_DOTPROD` in a combination
//! the baseline does not select. What survives the preprocessor there is
//! the trailing call to the `_generic` implementation, and that is what is
//! ported.
//!
//! **Five of them gain a body with i8mm.** `-Dcpu=apple_m3`, which the
//! xcframework's macOS slice targets, defines `__ARM_FEATURE_MATMUL_INT8`,
//! and the `q4_0` 4x8, `q4_K`/`q5_K`/`q6_K` 8x8 and `q8_0` 4x8 gemms then
//! compile their NEON arm instead of the call. Those arms are in
//! `mmla.zig`, chosen here at comptime as the C's `#if` chooses. The `q4_0`
//! 8x8 gemv and gemm need SVE as well, which no Apple core has, and stay
//! generic everywhere.
//!
//! **These seven are the only reason seven `_generic` symbols in
//! `repack.cpp` are reachable at all** -- at the baseline; with i8mm, two.
//! The other generics there are exported and dead; see `repack/q4_0.zig`.
//!
//! Measured from the preprocessed source, not read off the header: each
//! body below is the whole of the live function.

const std = @import("std");
const impl = @import("../../../impl.zig");
const mmla = @import("mmla.zig");

// The `_generic` implementations, in `cpu/repack/`. Declared rather than
// imported so the call reads as the C's does, across what is a translation
// unit boundary there.
extern fn ggml_gemv_q4_0_8x8_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void;
extern fn ggml_gemm_q4_0_4x8_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void;
extern fn ggml_gemm_q4_0_8x8_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void;
extern fn ggml_gemm_q4_K_8x8_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void;
extern fn ggml_gemm_q5_K_8x8_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void;
extern fn ggml_gemm_q6_K_8x8_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void;
extern fn ggml_gemm_q8_0_4x8_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void;

/// Ports `ggml_gemv_q4_0_8x8_q8_0` (arch/arm/repack.cpp:339 @b29c606e2).
pub export fn ggml_gemv_q4_0_8x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    ggml_gemv_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

/// Ports `ggml_gemm_q4_0_4x8_q8_0` (arch/arm/repack.cpp:2307 @b29c606e2).
pub export fn ggml_gemm_q4_0_4x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    if (comptime mmla.enabled) return mmla.gemmQ4_0_4x8(n, s, bs, vx, vy, nr, nc);
    ggml_gemm_q4_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

/// Ports `ggml_gemm_q4_0_8x8_q8_0` (arch/arm/repack.cpp:2728 @b29c606e2).
pub export fn ggml_gemm_q4_0_8x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    ggml_gemm_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

/// Ports `ggml_gemm_q4_K_8x8_q8_K` (arch/arm/repack.cpp:3752 @b29c606e2).
pub export fn ggml_gemm_q4_K_8x8_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    if (comptime mmla.enabled) return mmla.gemmQ4_K_8x8(n, s, bs, vx, vy, nr, nc);
    ggml_gemm_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

/// Ports `ggml_gemm_q5_K_8x8_q8_K` (arch/arm/repack.cpp:4272 @b29c606e2).
pub export fn ggml_gemm_q5_K_8x8_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    if (comptime mmla.enabled) return mmla.gemmQ5_K_8x8(n, s, bs, vx, vy, nr, nc);
    ggml_gemm_q5_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

/// Ports `ggml_gemm_q6_K_8x8_q8_K` (arch/arm/repack.cpp:4721 @b29c606e2).
pub export fn ggml_gemm_q6_K_8x8_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    if (comptime mmla.enabled) return mmla.gemmQ6_K_8x8(n, s, bs, vx, vy, nr, nc);
    ggml_gemm_q6_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

/// Ports `ggml_gemm_q8_0_4x8_q8_0` (arch/arm/repack.cpp:5006 @b29c606e2).
pub export fn ggml_gemm_q8_0_4x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    if (comptime mmla.enabled) return mmla.gemmQ8_0_4x8(n, s, bs, vx, vy, nr, nc);
    ggml_gemm_q8_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

// -----------------------------------------------------------------------------
// ARM's `arch-fallback.h` renames
//
// `arch-fallback.h:75-83` renames eight `repack.cpp` generics to the bare
// names on ARM, because `arch/arm/repack.cpp` has no version of them at all.
// They used to be exported bare from the generic files; x86 defines some of
// the same names itself, so the renames are made per architecture, here and
// in `x86/fallback.zig`.

const nibble4 = @import("../nibble4.zig");
const q2_k = @import("../q2_k.zig");
const quantize = @import("../quantize.zig");

comptime {
    _ = impl;
    _ = std;
    @export(&quantize.ggml_quantize_mat_q8_K_4x4_generic, .{ .name = "ggml_quantize_mat_q8_K_4x4" });
    @export(&quantize.ggml_quantize_mat_q8_K_4x8_generic, .{ .name = "ggml_quantize_mat_q8_K_4x8" });
    @export(&nibble4.ggml_gemv_iq4_nl_8x8_q8_0_generic, .{ .name = "ggml_gemv_iq4_nl_8x8_q8_0" });
    @export(&nibble4.ggml_gemv_mxfp4_8x8_q8_0_generic, .{ .name = "ggml_gemv_mxfp4_8x8_q8_0" });
    @export(&q2_k.ggml_gemv_q2_K_8x8_q8_K_generic, .{ .name = "ggml_gemv_q2_K_8x8_q8_K" });
    @export(&nibble4.ggml_gemm_iq4_nl_8x8_q8_0_generic, .{ .name = "ggml_gemm_iq4_nl_8x8_q8_0" });
    @export(&nibble4.ggml_gemm_mxfp4_8x8_q8_0_generic, .{ .name = "ggml_gemm_mxfp4_8x8_q8_0" });
    @export(&q2_k.ggml_gemm_q2_K_8x8_q8_K_generic, .{ .name = "ggml_gemm_q2_K_8x8_q8_K" });
}
