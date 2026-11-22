//! The C library's math functions, declared once, and the two ways clang
//! rewrites them that the reference's bits depend on.
//!
//! # Provenance
//!
//! **Not a port.** Upstream calls libm; the port calls the same functions,
//! by the same symbols, so the bits are the C library's. Zig's `@sqrt`,
//! `@exp` and friends are std's and round differently in places, so a
//! ported call that the C makes through libm comes here.
//!
//! # What clang does to them
//!
//! `math_errno` is clang's `-fmath-errno`, on by default off Darwin. With it
//! off (Darwin), clang folds a `sinf`/`cosf` pair on one argument into
//! Apple's `__sincosf_stret` (`__sincos_stret` for `double`), which rounds
//! differently from the separate calls, and turns `pow(10, x)` into
//! `__exp10(x)`. With it on (Linux), the pair stays two calls. Measured from
//! the reference objects' imports (`ops.o`, `mtmd-audio.o`, `mistral3.o`).

const std = @import("std");
const builtin = @import("builtin");

pub extern "c" fn sqrtf(x: f32) f32;
pub extern "c" fn sqrt(x: f64) f64;
pub extern "c" fn expf(x: f32) f32;
pub extern "c" fn exp(x: f64) f64;
pub extern "c" fn exp2f(x: f32) f32;
pub extern "c" fn expm1f(x: f32) f32;
pub extern "c" fn logf(x: f32) f32;
pub extern "c" fn log(x: f64) f64;
pub extern "c" fn log2f(x: f32) f32;
pub extern "c" fn log2(x: f64) f64;
pub extern "c" fn log10(x: f64) f64;
pub extern "c" fn log1pf(x: f32) f32;
pub extern "c" fn powf(x: f32, y: f32) f32;
pub extern "c" fn pow(x: f64, y: f64) f64;
pub extern "c" fn sinf(x: f32) f32;
pub extern "c" fn sin(x: f64) f64;
pub extern "c" fn cosf(x: f32) f32;
pub extern "c" fn cos(x: f64) f64;
pub extern "c" fn tanhf(x: f32) f32;
pub extern "c" fn erff(x: f32) f32;
pub extern "c" fn floorf(x: f32) f32;
pub extern "c" fn floor(x: f64) f64;
pub extern "c" fn ceilf(x: f32) f32;
pub extern "c" fn roundf(x: f32) f32;
pub extern "c" fn truncf(x: f32) f32;
pub extern "c" fn nearbyintf(x: f32) f32;
pub extern "c" fn nearbyint(x: f64) f64;
pub extern "c" fn lround(x: f64) c_long;
pub extern "c" fn fabsf(x: f32) f32;
pub extern "c" fn fmaxf(x: f32, y: f32) f32;
pub extern "c" fn fmax(x: f64, y: f64) f64;
pub extern "c" fn fminf(x: f32, y: f32) f32;
pub extern "c" fn fmin(x: f64, y: f64) f64;
pub extern "c" fn ldexpf(x: f32, e: c_int) f32;
/// `sqrtf(n)` on an integer, which C converts to `float` first: rounded to
/// nearest, as `@floatFromInt` rounds.
pub inline fn sqrtfInt(n: anytype) f32 {
    return sqrtf(@floatFromInt(n));
}

/// `log2f(n)` on an integer, converted as `sqrtfInt` converts it.
pub inline fn log2fInt(n: anytype) f32 {
    return log2f(@floatFromInt(n));
}

/// Apple's `exp10`, which clang calls for `pow(10, x)` on Darwin.
pub extern "c" fn __exp10(x: f64) f64;
pub extern "c" fn __exp10f(x: f32) f32;

/// The `__float2` and `__double2` Apple's `__sincos*_stret` return.
pub const SinCos = extern struct { sin: f32, cos: f32 };
pub const SinCos64 = extern struct { sin: f64, cos: f64 };
pub extern "c" fn __sincosf_stret(x: f32) SinCos;
pub extern "c" fn __sincos_stret(x: f64) SinCos64;

/// Whether the reference compiler treats libm calls as writing `errno`:
/// clang's `-fmath-errno`, its default on Linux and off on Darwin. A
/// property of the reference build that changes its bits in two ways, both
/// measured on the `aarch64-linux-gnu` reference against the `aarch64-macos`
/// one at the same CPU:
///
/// - a `sinf`/`cosf` pair on one argument is **not** folded into one
///   combined call (`sinCos`); and
/// - a strict loop that calls into libm is **not** vectorized, so it runs
///   scalar and all fused. `ssm_scan`'s Mamba-1 loop calls `expf`: on Darwin
///   its `ssm_scan` has twelve `expf` call sites, eight of them one per
///   lane, and on Linux four.
///
/// The other Linux-only libm imports were checked and do not move a bit:
/// `sqrtf` beside an `fsqrt` for the `errno` path, `lroundf` where Darwin
/// inlines `fcvtas`.
pub const math_errno = !builtin.os.tag.isDarwin();

/// `sinf(x)` and `cosf(x)` on one argument, **as clang emits the pair**.
///
/// The C calls `sinf` and `cosf`, but clang folds the pair into one libcall
/// on Darwin -- `__sincosf_stret` -- and Apple's combined routine does not
/// round as the separate ones do. Measured over 10.9M arguments: `sin`
/// differs in 4.0% of them, `cos` in 0.5%. The reference `ops.o` imports
/// `___sincosf_stret` and neither `_sinf` nor `_cosf`; `make ops-diff`
/// caught it on every f32 rope and on `timestep_embedding`. Off Darwin
/// there is no fold: the `aarch64-linux-gnu` reference's `ops.o` imports
/// `sinf` and `cosf` and no `sincosf`.
pub inline fn sinCos(x: f32) SinCos {
    if (comptime !math_errno) return __sincosf_stret(x);
    return .{ .sin = sinf(x), .cos = cosf(x) };
}

/// `sin(x)` and `cos(x)` on one argument, as clang emits the pair.
pub inline fn sinCos64(x: f64) SinCos64 {
    if (comptime !math_errno) return __sincos_stret(x);
    return .{ .sin = sin(x), .cos = cos(x) };
}

/// `pow(10.0, x)`, as clang emits it.
pub inline fn pow10(x: f64) f64 {
    if (comptime !math_errno) return __exp10(x);
    return pow(10.0, x);
}

/// `powf(10.0f, x)`, as clang emits it.
pub inline fn pow10f(x: f32) f32 {
    if (comptime !math_errno) return __exp10f(x);
    return powf(10.0, x);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the pair and the powers of ten answer as the separate calls do, near enough" {
    const sc = sinCos(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.47942554), sc.sin, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.87758255), sc.cos, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 1000.0), pow10(3.0), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), pow10f(2.0), 1e-4);
}
