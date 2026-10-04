//! The seven `arch/arm/repack.cpp` entry points that delegate to the
//! generic on this target.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # Why these are one line each
//!
//! Each has a NEON body behind `__ARM_FEATURE_SVE`,
//! `__ARM_FEATURE_MATMUL_INT8` or `__ARM_FEATURE_DOTPROD` in a combination
//! this build does not select — `zig cc` reports `HAVE_MATMUL_INT8 -
//! Failed` and `HAVE_SVE - Failed`, which `CLAUDE.md` records. What
//! survives the preprocessor is the trailing call to the `_generic`
//! implementation, and that is what is ported.
//!
//! **These seven are the only reason seven `_generic` symbols in
//! `repack.cpp` are reachable at all.** The other generics there are
//! exported and dead; see `repack/q4_0.zig`.
//!
//! Measured from the preprocessed source, not read off the header: each
//! body below is the whole of the live function.

const std = @import("std");
const impl = @import("../../../impl.zig");

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

/// Ports `ggml_gemv_q4_0_8x8_q8_0` (arch/arm/repack.cpp:339 @c1d0e7a00).
pub export fn ggml_gemv_q4_0_8x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    ggml_gemv_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

/// Ports `ggml_gemm_q4_0_4x8_q8_0` (arch/arm/repack.cpp:2307 @c1d0e7a00).
pub export fn ggml_gemm_q4_0_4x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    ggml_gemm_q4_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

/// Ports `ggml_gemm_q4_0_8x8_q8_0` (arch/arm/repack.cpp:2728 @c1d0e7a00).
pub export fn ggml_gemm_q4_0_8x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    ggml_gemm_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

/// Ports `ggml_gemm_q4_K_8x8_q8_K` (arch/arm/repack.cpp:3752 @c1d0e7a00).
pub export fn ggml_gemm_q4_K_8x8_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    ggml_gemm_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

/// Ports `ggml_gemm_q5_K_8x8_q8_K` (arch/arm/repack.cpp:4272 @c1d0e7a00).
pub export fn ggml_gemm_q5_K_8x8_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    ggml_gemm_q5_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

/// Ports `ggml_gemm_q6_K_8x8_q8_K` (arch/arm/repack.cpp:4721 @c1d0e7a00).
pub export fn ggml_gemm_q6_K_8x8_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    ggml_gemm_q6_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

/// Ports `ggml_gemm_q8_0_4x8_q8_0` (arch/arm/repack.cpp:5006 @c1d0e7a00).
pub export fn ggml_gemm_q8_0_4x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    ggml_gemm_q8_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

comptime {
    _ = impl;
    _ = std;
}
