//! The SSE, AVX and AVX2 intrinsics `ggml-cpu/arch/x86/` uses, as Zig.
//!
//! # Provenance
//!
//! **Not a port of a file.** Each function here reproduces one Intel
//! intrinsic that `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` or
//! `arch/x86/repack.cpp` (v0.3.0, `c1d0e7a00`) calls, and is named after it
//! with the leading underscore dropped: `_mm256_maddubs_epi16` is
//! `mm256_maddubs_epi16`. The semantics are Intel's.
//!
//! # Why this layer exists
//!
//! It is `quants/arm/neon.zig` for x86: gathering the intrinsics keeps each
//! kernel readable line by line against the C, and puts every semantic trap
//! in one place.
//!
//! # The types are the C's
//!
//! An `__m256i` has no lane type; the C passes one register to an `epi8`
//! operation and the next to an `epi16` one. So `m256i` and `m128i` are
//! clang's own definitions -- `long long` lanes -- and each function
//! reinterprets its operands as the lanes its instruction works on. A kernel
//! then reads exactly as the C does, with no casts between the calls.
//!
//! # Plain lanes in Zig, the rest through LLVM
//!
//! Where an instruction's lanes are independent and ordinary -- `and`,
//! wrapping `add`, a shift, a compare -- it is `@Vector` arithmetic, and
//! LLVM selects the instruction. Where it is not, the function calls LLVM's
//! x86 intrinsic by name, so the instruction is the one the C compiles to and
//! its semantics come with it:
//!
//! - **`pmaddubsw` saturates.** Unsigned bytes times signed bytes, adjacent
//!   pairs summed into an `i16` with *signed saturation*. 255 × 127 × 2
//!   overflows.
//! - **`pshufb` works within each 128-bit half** and zeroes a byte whose
//!   index has its top bit set. It is not a 32-byte permute.
//! - **`psign` zeroes as well as negates**: a zero in the second operand
//!   gives zero, and `-128` stays `-128`.
//! - **`phadd` pairs neighbours of each operand in turn**, and per 128-bit
//!   half in the 256-bit forms.
//! - **The packs saturate**, `packus` to unsigned.
//!
//! Those intrinsics are declared `callconv(.{ .x86_64_sysv = .{} })`, as
//! `cpu/convert.zig` explains: the Win64 C convention passes a vector by
//! reference, and LLVM rejects an intrinsic called with a pointer.
//!
//! # Integers wrap
//!
//! `paddb`, `psubw` and the rest truncate; Zig's `+` panics in a safe build.
//! Every integer operation here uses the wrapping form.
//!
//! # Only compiled for x86
//!
//! Nothing here is analysed on another target: `quants/module.zig` and
//! `repack/module.zig` import `x86/` only for `x86_64`. Its oracle is
//! `scripts/x86-quants-diff`, on the Windows laptop.

const std = @import("std");

// -----------------------------------------------------------------------------
// Types

/// `__m256i`, as clang defines it: four `long long` lanes, reinterpreted by
/// each operation.
pub const m256i = @Vector(4, i64);
/// `__m128i`.
pub const m128i = @Vector(2, i64);
/// `__m256`.
pub const m256 = @Vector(8, f32);
/// `__m128`.
pub const m128 = @Vector(4, f32);

pub const i8x32 = @Vector(32, i8);
pub const u8x32 = @Vector(32, u8);
pub const i16x16 = @Vector(16, i16);
pub const u16x16 = @Vector(16, u16);
pub const i32x8 = @Vector(8, i32);
pub const u32x8 = @Vector(8, u32);
pub const i8x16 = @Vector(16, i8);
pub const u8x16 = @Vector(16, u8);
pub const i16x8 = @Vector(8, i16);
pub const u16x8 = @Vector(8, u16);
pub const i32x4 = @Vector(4, i32);
pub const u32x4 = @Vector(4, u32);

/// Reinterprets a register as other lanes, as every intrinsic does.
pub inline fn as(comptime V: type, x: anytype) V {
    return @bitCast(x);
}

// -----------------------------------------------------------------------------
// LLVM's x86 intrinsics
//
// Named exactly as LLVM names them. Each is called only from the wrapper
// below that carries the Intel name.

const sysv: std.builtin.CallingConvention = .{ .x86_64_sysv = .{} };

extern fn @"llvm.x86.avx2.pmadd.ub.sw"(u8x32, i8x32) callconv(sysv) i16x16;
extern fn @"llvm.x86.ssse3.pmadd.ub.sw.128"(u8x16, i8x16) callconv(sysv) i16x8;
extern fn @"llvm.x86.avx2.pmadd.wd"(i16x16, i16x16) callconv(sysv) i32x8;
extern fn @"llvm.x86.sse2.pmadd.wd"(i16x8, i16x8) callconv(sysv) i32x4;
extern fn @"llvm.x86.avx2.pshuf.b"(i8x32, i8x32) callconv(sysv) i8x32;
extern fn @"llvm.x86.ssse3.pshuf.b.128"(i8x16, i8x16) callconv(sysv) i8x16;
extern fn @"llvm.x86.avx2.psign.b"(i8x32, i8x32) callconv(sysv) i8x32;
extern fn @"llvm.x86.ssse3.psign.b.128"(i8x16, i8x16) callconv(sysv) i8x16;
extern fn @"llvm.x86.ssse3.phadd.w.128"(i16x8, i16x8) callconv(sysv) i16x8;
extern fn @"llvm.x86.ssse3.phadd.d.128"(i32x4, i32x4) callconv(sysv) i32x4;
extern fn @"llvm.x86.avx2.phadd.w"(i16x16, i16x16) callconv(sysv) i16x16;
extern fn @"llvm.x86.avx2.phadd.d"(i32x8, i32x8) callconv(sysv) i32x8;
extern fn @"llvm.x86.sse2.packuswb.128"(i16x8, i16x8) callconv(sysv) i8x16;
extern fn @"llvm.x86.avx2.packuswb"(i16x16, i16x16) callconv(sysv) i8x32;
extern fn @"llvm.x86.sse2.packsswb.128"(i16x8, i16x8) callconv(sysv) i8x16;
extern fn @"llvm.x86.sse2.packssdw.128"(i32x4, i32x4) callconv(sysv) i16x8;
extern fn @"llvm.x86.avx2.packsswb"(i16x16, i16x16) callconv(sysv) i8x32;
extern fn @"llvm.x86.avx2.packssdw"(i32x8, i32x8) callconv(sysv) i16x16;
extern fn @"llvm.x86.avx2.permd"(i32x8, i32x8) callconv(sysv) i32x8;
extern fn @"llvm.x86.avx2.pmovmskb"(i8x32) callconv(sysv) i32;
extern fn @"llvm.x86.sse2.pmovmskb.128"(i8x16) callconv(sysv) i32;
extern fn @"llvm.x86.avx.cvt.ps2dq.256"(m256) callconv(sysv) i32x8;
extern fn @"llvm.x86.avx.round.ps.256"(m256, i32) callconv(sysv) m256;
extern fn @"llvm.x86.bmi.pdep.64"(u64, u64) callconv(sysv) u64;
extern fn @"llvm.x86.avx.movmsk.ps.256"(m256) callconv(sysv) i32;
extern fn @"llvm.x86.avx2.psrli.w"(i16x16, i32) callconv(sysv) i16x16;
extern fn @"llvm.x86.avx2.pslli.w"(i16x16, i32) callconv(sysv) i16x16;
extern fn @"llvm.x86.avx2.psrli.d"(i32x8, i32) callconv(sysv) i32x8;
extern fn @"llvm.x86.avx2.pslli.d"(i32x8, i32) callconv(sysv) i32x8;
extern fn @"llvm.x86.avx2.psrli.q"(@Vector(4, i64), i32) callconv(sysv) @Vector(4, i64);
extern fn @"llvm.x86.sse2.psrli.w"(i16x8, i32) callconv(sysv) i16x8;
extern fn @"llvm.x86.sse2.pslli.w"(i16x8, i32) callconv(sysv) i16x8;
extern fn @"llvm.x86.avx2.permps"(m256, i32x8) callconv(sysv) m256;
extern fn @"llvm.x86.sse3.hadd.ps"(m128, m128) callconv(sysv) m128;
extern fn @"llvm.x86.avx.hadd.ps.256"(m256, m256) callconv(sysv) m256;

// -----------------------------------------------------------------------------
// Construction, loads and stores

pub inline fn mm256_setzero_si256() m256i {
    return @splat(0);
}

pub inline fn mm_setzero_si128() m128i {
    return @splat(0);
}

pub inline fn mm256_setzero_ps() m256 {
    return @splat(0);
}

pub inline fn mm256_set1_epi8(v: i8) m256i {
    return as(m256i, @as(i8x32, @splat(v)));
}

pub inline fn mm256_set1_epi16(v: i16) m256i {
    return as(m256i, @as(i16x16, @splat(v)));
}

pub inline fn mm256_set1_epi32(v: i32) m256i {
    return as(m256i, @as(i32x8, @splat(v)));
}

pub inline fn mm256_set1_epi64x(v: i64) m256i {
    return @splat(v);
}

/// `_mm256_set_epi64x(e3, e2, e1, e0)`: highest lane first, as Intel's `set`
/// forms all are.
pub inline fn mm256_set_epi64x(e3: i64, e2: i64, e1: i64, e0: i64) m256i {
    return .{ e0, e1, e2, e3 };
}

/// `_mm256_setr_epi8`: lowest lane first.
pub inline fn mm256_setr_epi8(v: [32]i8) m256i {
    return as(m256i, @as(i8x32, v));
}

/// `_mm256_setr_epi32`: lowest lane first.
pub inline fn mm256_setr_epi32(v: [8]i32) m256i {
    return as(m256i, @as(i32x8, v));
}

pub inline fn mm_set1_epi8(v: i8) m128i {
    return as(m128i, @as(i8x16, @splat(v)));
}

pub inline fn mm_set1_epi16(v: i16) m128i {
    return as(m128i, @as(i16x8, @splat(v)));
}

pub inline fn mm_set1_epi32(v: i32) m128i {
    return as(m128i, @as(i32x4, @splat(v)));
}

/// `_mm256_set_epi16(e15, ..., e0)`, highest lane first: `v[0]` is `e15`.
pub inline fn mm256_set_epi16(v: [16]i16) m256i {
    var r: [16]i16 = undefined;
    for (0..16) |k| r[k] = v[15 - k];
    return as(m256i, @as(i16x16, r));
}

/// `_mm256_set_epi32(e7, ..., e0)`, highest lane first: `v[0]` is `e7`.
pub inline fn mm256_set_epi32(v: [8]i32) m256i {
    return as(m256i, i32x8{ v[7], v[6], v[5], v[4], v[3], v[2], v[1], v[0] });
}

/// `_mm_set_epi64x(e1, e0)`, highest lane first.
pub inline fn mm_set_epi64x(e1: i64, e0: i64) m128i {
    return .{ e0, e1 };
}

pub inline fn mm_set1_epi64x(v: i64) m128i {
    return @splat(v);
}

/// `_mm_set_epi8(e15, ..., e0)`, highest lane first: `v[0]` is `e15`.
pub inline fn mm_set_epi8(v: [16]i8) m128i {
    var r: [16]i8 = undefined;
    for (0..16) |k| r[k] = v[15 - k];
    return as(m128i, @as(i8x16, r));
}

/// `_mm256_set_ps(e7, ..., e0)`, highest lane first: `v[0]` is `e7`.
pub inline fn mm256_set_ps(v: [8]f32) m256 {
    return .{ v[7], v[6], v[5], v[4], v[3], v[2], v[1], v[0] };
}

/// `_mm_setr_epi8`: lowest lane first.
pub inline fn mm_setr_epi8(v: [16]i8) m128i {
    return as(m128i, @as(i8x16, v));
}

/// `_mm_set_epi32(e3, e2, e1, e0)`, highest lane first.
pub inline fn mm_set_epi32(e3: i32, e2: i32, e1: i32, e0: i32) m128i {
    return as(m128i, i32x4{ e0, e1, e2, e3 });
}

pub inline fn mm256_set1_ps(v: f32) m256 {
    return @splat(v);
}

/// `_mm256_set_m128(hi, lo)`.
pub inline fn mm256_set_m128(hi: m128, lo: m128) m256 {
    return .{ lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3] };
}

/// `_mm_load_ps`: four floats.
pub inline fn mm_load_ps(p: *const [4]f32) m128 {
    var rp = p;
    _ = &rp;
    const v: *align(1) const m128 = @ptrCast(rp);
    return v.*;
}

/// `_mm256_loadu_ps`: eight floats, no alignment.
pub inline fn mm256_loadu_ps(p: [*]const f32) m256 {
    var rp = p;
    _ = &rp;
    const v: *align(1) const m256 = @ptrCast(rp);
    return v.*;
}

/// `_mm256_broadcast_ss(&v)`: the same register as `_mm256_set1_ps(v)`.
pub inline fn mm256_broadcast_ss(p: *const f32) m256 {
    return @splat(p.*);
}

pub inline fn mm_setzero_ps() m128 {
    return @splat(0);
}

pub inline fn mm_set1_ps(v: f32) m128 {
    return @splat(v);
}

/// `_mm256_loadu_si256`: 32 bytes, no alignment.
///
/// Through a pointer to the vector type, never as an array bit-cast to one:
/// measured, the array form compiles to 32 `movzbl` and `vpinsrb` per load,
/// which made the K-quant kernels 10-15 times slower than the C's with
/// every bit still right.
///
/// The pointer is copied to a runtime variable first: a vector has no
/// defined memory layout at comptime, so a load from a comptime-known table
/// -- the scale shuffles, `kvalues_iq4nl` -- would otherwise be a compile
/// error. LLVM still folds a load from constant data.
pub inline fn mm256_loadu_si256(p: *const anyopaque) m256i {
    var rp = p;
    _ = &rp;
    const v: *align(1) const m256i = @ptrCast(rp);
    return v.*;
}

/// `_mm_loadu_si128`: 16 bytes, no alignment, as `mm256_loadu_si256`.
pub inline fn mm_loadu_si128(p: *const anyopaque) m128i {
    var rp = p;
    _ = &rp;
    const v: *align(1) const m128i = @ptrCast(rp);
    return v.*;
}

pub inline fn mm256_storeu_si256(p: *anyopaque, a: m256i) void {
    const v: *align(1) m256i = @ptrCast(p);
    v.* = a;
}

/// `_mm256_storeu_ps`: eight floats, no alignment.
pub inline fn mm256_storeu_ps(p: [*]f32, a: m256) void {
    const v: *align(1) m256 = @ptrCast(p);
    v.* = a;
}

pub inline fn mm_storeu_si128(p: *anyopaque, a: m128i) void {
    const v: *align(1) m128i = @ptrCast(p);
    v.* = a;
}

// -----------------------------------------------------------------------------
// Moving between halves

/// `_mm256_extracti128_si256(a, imm)` and `_mm256_extractf128_si256`, which
/// select the same bits.
pub inline fn mm256_extracti128_si256(a: m256i, comptime imm: u1) m128i {
    return if (imm == 0) .{ a[0], a[1] } else .{ a[2], a[3] };
}

pub const mm256_extractf128_si256 = mm256_extracti128_si256;

/// `_mm256_castsi128_si256`: the high half is undefined in the C; zero here.
/// Every use in the ported code overwrites or discards it.
pub inline fn mm256_castsi128_si256(a: m128i) m256i {
    return .{ a[0], a[1], 0, 0 };
}

/// `_mm256_permute2f128_si256(a, b, imm)`: each 128-bit half of the result
/// chosen from `a`'s two halves and `b`'s by a nibble of `imm`, or zeroed by
/// its bit 3.
pub inline fn mm256_permute2f128_si256(a: m256i, b: m256i, comptime imm: u8) m256i {
    const halves = [4]m128i{ .{ a[0], a[1] }, .{ a[2], a[3] }, .{ b[0], b[1] }, .{ b[2], b[3] } };
    const lo: m128i = if (imm & 0x08 != 0) @splat(0) else halves[imm & 3];
    const hi: m128i = if (imm & 0x80 != 0) @splat(0) else halves[(imm >> 4) & 3];
    return .{ lo[0], lo[1], hi[0], hi[1] };
}

/// `_mm256_blend_epi32(a, b, imm)`: 32-bit lane `k` from `b` where bit `k`
/// of `imm` is set, else from `a`.
pub inline fn mm256_blend_epi32(a: m256i, b: m256i, comptime imm: u8) m256i {
    const mask: @Vector(8, bool) = comptime blk: {
        var m: [8]bool = undefined;
        for (0..8) |k| m[k] = (imm >> k) & 1 != 0;
        break :blk m;
    };
    return as(m256i, @select(i32, mask, as(i32x8, b), as(i32x8, a)));
}

/// `_mm256_shuffle_epi32(a, imm)`: within each 128-bit half, lane `k` is
/// lane `(imm >> 2k) & 3` of that half.
pub inline fn mm256_shuffle_epi32(a: m256i, comptime imm: u8) m256i {
    const sel = comptime blk: {
        var m: [8]i32 = undefined;
        for (0..8) |k| m[k] = @intCast((k / 4) * 4 + ((imm >> (2 * (k % 4))) & 3));
        break :blk m;
    };
    return as(m256i, @shuffle(i32, as(i32x8, a), undefined, sel));
}

/// `_mm256_shuffle_ps(a, b, imm)`: within each 128-bit half, lanes 0 and 1
/// from `a` and lanes 2 and 3 from `b`, each chosen by two bits of `imm`.
pub inline fn mm256_shuffle_ps(a: m256, b: m256, comptime imm: u8) m256 {
    const sel = comptime blk: {
        var m: [8]i32 = undefined;
        for (0..8) |k| {
            const half = (k / 4) * 4;
            const lane: i32 = @intCast(half + ((imm >> (2 * (k % 4))) & 3));
            // Negative indices select from `b` in `@shuffle`.
            m[k] = if (k % 4 < 2) lane else ~lane;
        }
        break :blk m;
    };
    return @shuffle(f32, a, b, sel);
}

/// `_mm256_permutevar8x32_ps(a, idx)`: a full permute of the eight floats.
pub inline fn mm256_permutevar8x32_ps(a: m256, idx: m256i) m256 {
    return @"llvm.x86.avx2.permps"(a, as(i32x8, idx));
}

/// `_mm256_blend_epi16(a, b, imm)`: sixteen-bit lane `k` of each 128-bit
/// half from `b` where bit `k` of `imm` is set, else from `a`.
pub inline fn mm256_blend_epi16(a: m256i, b: m256i, comptime imm: u8) m256i {
    const mask: @Vector(16, bool) = comptime blk: {
        var m: [16]bool = undefined;
        for (0..16) |k| m[k] = (imm >> (k % 8)) & 1 != 0;
        break :blk m;
    };
    return as(m256i, @select(i16, mask, as(i16x16, b), as(i16x16, a)));
}

/// `_mm256_castsi256_si128`: the low half.
pub inline fn mm256_castsi256_si128(a: m256i) m128i {
    return .{ a[0], a[1] };
}

/// `MM256_SET_M128I` (arch/x86/quants.c:26 @c1d0e7a00) and
/// `_mm256_set_m128i`: the first operand high, the second low.
pub inline fn mm256_set_m128i(hi: m128i, lo: m128i) m256i {
    return .{ lo[0], lo[1], hi[0], hi[1] };
}

pub inline fn mm256_extractf128_ps(a: m256, comptime imm: u1) m128 {
    return if (imm == 0) .{ a[0], a[1], a[2], a[3] } else .{ a[4], a[5], a[6], a[7] };
}

pub inline fn mm256_castps256_ps128(a: m256) m128 {
    return .{ a[0], a[1], a[2], a[3] };
}

/// `_mm_unpacklo_epi64(a, b)`: `a`'s low 64 bits, then `b`'s.
pub inline fn mm_unpacklo_epi64(a: m128i, b: m128i) m128i {
    return .{ a[0], b[0] };
}

/// `_mm_unpackhi_epi64(a, b)`: `a`'s high 64 bits, then `b`'s.
pub inline fn mm_unpackhi_epi64(a: m128i, b: m128i) m128i {
    return .{ a[1], b[1] };
}

/// `_mm_shuffle_epi32(a, imm)`: lane `k` of the result is lane
/// `(imm >> 2k) & 3` of `a`.
pub inline fn mm_shuffle_epi32(a: m128i, comptime imm: u8) m128i {
    const v = as(i32x4, a);
    return as(m128i, i32x4{ v[imm & 3], v[(imm >> 2) & 3], v[(imm >> 4) & 3], v[(imm >> 6) & 3] });
}

/// `_MM_SHUFFLE(z, y, x, w)`.
pub inline fn MM_SHUFFLE(comptime z: u8, comptime y: u8, comptime x: u8, comptime w: u8) u8 {
    return (z << 6) | (y << 4) | (x << 2) | w;
}

/// `_mm_unpacklo_epi8(a, b)`: the low eight bytes of each, interleaved,
/// `a` first.
pub inline fn mm_unpacklo_epi8(a: m128i, b: m128i) m128i {
    const x = as(u8x16, a);
    const y = as(u8x16, b);
    var r: u8x16 = undefined;
    inline for (0..8) |k| {
        r[2 * k] = x[k];
        r[2 * k + 1] = y[k];
    }
    return as(m128i, r);
}

/// `_mm_extract_epi32(a, imm)`.
pub inline fn mm_extract_epi32(a: m128i, comptime imm: u2) i32 {
    return as(i32x4, a)[imm];
}

/// `_mm_cvtsi128_si32`: lane 0.
pub inline fn mm_cvtsi128_si32(a: m128i) i32 {
    return as(i32x4, a)[0];
}

/// `_mm_cvtss_f32`: lane 0.
pub inline fn mm_cvtss_f32(a: m128) f32 {
    return a[0];
}

// -----------------------------------------------------------------------------
// Bitwise

pub inline fn mm256_and_si256(a: m256i, b: m256i) m256i {
    return a & b;
}

pub inline fn mm256_or_si256(a: m256i, b: m256i) m256i {
    return a | b;
}

pub inline fn mm256_xor_si256(a: m256i, b: m256i) m256i {
    return a ^ b;
}

/// `_mm256_andnot_si256(a, b)`: `~a & b` -- the *first* operand is inverted.
pub inline fn mm256_andnot_si256(a: m256i, b: m256i) m256i {
    return ~a & b;
}

pub inline fn mm_and_si128(a: m128i, b: m128i) m128i {
    return a & b;
}

pub inline fn mm_or_si128(a: m128i, b: m128i) m128i {
    return a | b;
}

pub inline fn mm_xor_si128(a: m128i, b: m128i) m128i {
    return a ^ b;
}

pub inline fn mm_andnot_si128(a: m128i, b: m128i) m128i {
    return ~a & b;
}

// -----------------------------------------------------------------------------
// Shifts
//
// Through LLVM's `psrli`/`pslli`, which take the count as a scalar and give
// Intel's rule for a count past the lane width -- zero -- in the one
// instruction. The count may be a run-time value: `q3_K` and `q5_K` shift by
// `bit++`, and a portable form with a branch and a splatted 4-bit count made
// both of them two to three times slower than the C.

pub inline fn mm256_srli_epi16(a: m256i, count: u32) m256i {
    return as(m256i, @"llvm.x86.avx2.psrli.w"(as(i16x16, a), @bitCast(count)));
}

pub inline fn mm256_slli_epi16(a: m256i, count: u32) m256i {
    return as(m256i, @"llvm.x86.avx2.pslli.w"(as(i16x16, a), @bitCast(count)));
}

pub inline fn mm256_slli_epi32(a: m256i, count: u32) m256i {
    return as(m256i, @"llvm.x86.avx2.pslli.d"(as(i32x8, a), @bitCast(count)));
}

pub inline fn mm256_srli_epi32(a: m256i, count: u32) m256i {
    return as(m256i, @"llvm.x86.avx2.psrli.d"(as(i32x8, a), @bitCast(count)));
}

pub inline fn mm256_srli_epi64(a: m256i, count: u32) m256i {
    return @"llvm.x86.avx2.psrli.q"(a, @bitCast(count));
}

/// `_mm256_sllv_epi32(a, count)`: each lane by its own count; a count past
/// 31 gives zero.
pub inline fn mm256_sllv_epi32(a: m256i, count: m256i) m256i {
    const n = as(u32x8, count);
    const shifted = as(u32x8, a) << @intCast(n & @as(u32x8, @splat(31)));
    return as(m256i, @select(u32, n < @as(u32x8, @splat(32)), shifted, @as(u32x8, @splat(0))));
}

/// `_mm256_srlv_epi64(a, count)`: each lane by its own count; a count past
/// 63 gives zero.
pub inline fn mm256_srlv_epi64(a: m256i, count: m256i) m256i {
    const V = @Vector(4, u64);
    const n = as(V, count);
    const shifted = as(V, a) >> @intCast(n & @as(V, @splat(63)));
    return as(m256i, @select(u64, n < @as(V, @splat(64)), shifted, @as(V, @splat(0))));
}

/// `_mm256_bsrli_epi128(a, imm)`: each 128-bit half shifted right by `imm`
/// bytes, zeros shifted in.
pub inline fn mm256_bsrli_epi128(a: m256i, comptime imm: u8) m256i {
    const sel = comptime blk: {
        var m: [32]i32 = undefined;
        for (0..32) |k| {
            const src = (k % 16) + imm;
            m[k] = if (src < 16) @intCast((k / 16) * 16 + src) else -1;
        }
        break :blk m;
    };
    const zero: u8x32 = @splat(0);
    return as(m256i, @shuffle(u8, as(u8x32, a), zero, sel));
}

pub inline fn mm_srli_epi16(a: m128i, count: u32) m128i {
    return as(m128i, @"llvm.x86.sse2.psrli.w"(as(i16x8, a), @bitCast(count)));
}

pub inline fn mm_slli_epi16(a: m128i, count: u32) m128i {
    return as(m128i, @"llvm.x86.sse2.pslli.w"(as(i16x8, a), @bitCast(count)));
}

// -----------------------------------------------------------------------------
// Arithmetic, wrapping

pub inline fn mm256_add_epi8(a: m256i, b: m256i) m256i {
    return as(m256i, as(i8x32, a) +% as(i8x32, b));
}

pub inline fn mm256_add_epi16(a: m256i, b: m256i) m256i {
    return as(m256i, as(i16x16, a) +% as(i16x16, b));
}

pub inline fn mm256_add_epi32(a: m256i, b: m256i) m256i {
    return as(m256i, as(i32x8, a) +% as(i32x8, b));
}

pub inline fn mm256_sub_epi8(a: m256i, b: m256i) m256i {
    return as(m256i, as(i8x32, a) -% as(i8x32, b));
}

pub inline fn mm256_sub_epi16(a: m256i, b: m256i) m256i {
    return as(m256i, as(i16x16, a) -% as(i16x16, b));
}

pub inline fn mm256_sub_epi32(a: m256i, b: m256i) m256i {
    return as(m256i, as(i32x8, a) -% as(i32x8, b));
}

/// `_mm256_subs_epu8`: unsigned bytes, saturating at zero.
pub inline fn mm256_subs_epu8(a: m256i, b: m256i) m256i {
    return as(m256i, as(u8x32, a) -| as(u8x32, b));
}

/// `_mm256_avg_epu8`: `(a + b + 1) >> 1` on unsigned bytes, without
/// overflow -- the rounding is upward, not to even.
pub inline fn mm256_avg_epu8(a: m256i, b: m256i) m256i {
    const wa: @Vector(32, u16) = as(u8x32, a);
    const wb: @Vector(32, u16) = as(u8x32, b);
    return as(m256i, @as(u8x32, @intCast((wa + wb + @as(@Vector(32, u16), @splat(1))) >> @splat(1))));
}

/// `_mm256_mullo_epi16`: the low sixteen bits of each product.
pub inline fn mm256_mullo_epi16(a: m256i, b: m256i) m256i {
    return as(m256i, as(i16x16, a) *% as(i16x16, b));
}

pub inline fn mm_add_epi8(a: m128i, b: m128i) m128i {
    return as(m128i, as(i8x16, a) +% as(i8x16, b));
}

pub inline fn mm_add_epi32(a: m128i, b: m128i) m128i {
    return as(m128i, as(i32x4, a) +% as(i32x4, b));
}

pub inline fn mm_add_epi16(a: m128i, b: m128i) m128i {
    return as(m128i, as(i16x8, a) +% as(i16x8, b));
}

pub inline fn mm_sub_epi8(a: m128i, b: m128i) m128i {
    return as(m128i, as(i8x16, a) -% as(i8x16, b));
}

pub inline fn mm_sub_epi16(a: m128i, b: m128i) m128i {
    return as(m128i, as(i16x8, a) -% as(i16x8, b));
}

/// `_mm256_cmpeq_epi8`: all ones where equal.
pub inline fn mm256_cmpeq_epi8(a: m256i, b: m256i) m256i {
    const eq = as(i8x32, a) == as(i8x32, b);
    return as(m256i, @select(i8, eq, @as(i8x32, @splat(-1)), @as(i8x32, @splat(0))));
}

// -----------------------------------------------------------------------------
// Widening

/// `_mm256_cvtepu8_epi16`: sixteen bytes zero-extended.
pub inline fn mm256_cvtepu8_epi16(a: m128i) m256i {
    return as(m256i, @as(@Vector(16, u16), as(u8x16, a)));
}

/// `_mm256_cvtepi16_epi32`: eight `i16` sign-extended.
pub inline fn mm256_cvtepi16_epi32(a: m128i) m256i {
    return as(m256i, @as(i32x8, as(i16x8, a)));
}

/// `_mm256_cvtepi8_epi64`: the low four bytes sign-extended.
pub inline fn mm256_cvtepi8_epi64(a: m128i) m256i {
    const b = as(i8x16, a);
    return .{ b[0], b[1], b[2], b[3] };
}

/// `_mm256_cvtepi8_epi16`: sixteen bytes sign-extended.
pub inline fn mm256_cvtepi8_epi16(a: m128i) m256i {
    return as(m256i, @as(i16x16, as(i8x16, a)));
}

// -----------------------------------------------------------------------------
// The instructions with their own semantics, through LLVM

/// `_mm256_maddubs_epi16(a, b)`: `a` **unsigned**, `b` signed, adjacent
/// products summed into `i16` with signed saturation.
pub inline fn mm256_maddubs_epi16(a: m256i, b: m256i) m256i {
    return as(m256i, @"llvm.x86.avx2.pmadd.ub.sw"(as(u8x32, a), as(i8x32, b)));
}

pub inline fn mm_maddubs_epi16(a: m128i, b: m128i) m128i {
    return as(m128i, @"llvm.x86.ssse3.pmadd.ub.sw.128"(as(u8x16, a), as(i8x16, b)));
}

/// `_mm256_madd_epi16`: adjacent `i16` products summed into `i32`.
pub inline fn mm256_madd_epi16(a: m256i, b: m256i) m256i {
    return as(m256i, @"llvm.x86.avx2.pmadd.wd"(as(i16x16, a), as(i16x16, b)));
}

pub inline fn mm_madd_epi16(a: m128i, b: m128i) m128i {
    return as(m128i, @"llvm.x86.sse2.pmadd.wd"(as(i16x8, a), as(i16x8, b)));
}

/// `_mm256_shuffle_epi8`: a byte table lookup within each 128-bit half; an
/// index with its top bit set gives zero.
pub inline fn mm256_shuffle_epi8(a: m256i, b: m256i) m256i {
    return as(m256i, @"llvm.x86.avx2.pshuf.b"(as(i8x32, a), as(i8x32, b)));
}

pub inline fn mm_shuffle_epi8(a: m128i, b: m128i) m128i {
    return as(m128i, @"llvm.x86.ssse3.pshuf.b.128"(as(i8x16, a), as(i8x16, b)));
}

/// `_mm256_sign_epi8(a, b)`: `a` negated where `b` is negative, zeroed where
/// `b` is zero.
pub inline fn mm256_sign_epi8(a: m256i, b: m256i) m256i {
    return as(m256i, @"llvm.x86.avx2.psign.b"(as(i8x32, a), as(i8x32, b)));
}

pub inline fn mm_sign_epi8(a: m128i, b: m128i) m128i {
    return as(m128i, @"llvm.x86.ssse3.psign.b.128"(as(i8x16, a), as(i8x16, b)));
}

/// `_mm_hadd_epi16(a, b)`: `a`'s neighbouring pairs, then `b`'s, wrapping.
pub inline fn mm_hadd_epi16(a: m128i, b: m128i) m128i {
    return as(m128i, @"llvm.x86.ssse3.phadd.w.128"(as(i16x8, a), as(i16x8, b)));
}

pub inline fn mm_hadd_epi32(a: m128i, b: m128i) m128i {
    return as(m128i, @"llvm.x86.ssse3.phadd.d.128"(as(i32x4, a), as(i32x4, b)));
}

/// `_mm256_hadd_epi16`: per 128-bit half, `a`'s pairs then `b`'s.
pub inline fn mm256_hadd_epi16(a: m256i, b: m256i) m256i {
    return as(m256i, @"llvm.x86.avx2.phadd.w"(as(i16x16, a), as(i16x16, b)));
}

pub inline fn mm256_hadd_epi32(a: m256i, b: m256i) m256i {
    return as(m256i, @"llvm.x86.avx2.phadd.d"(as(i32x8, a), as(i32x8, b)));
}

/// `_mm_packus_epi16`: sixteen `i16` to bytes, saturated to `[0, 255]`.
pub inline fn mm_packus_epi16(a: m128i, b: m128i) m128i {
    return as(m128i, @"llvm.x86.sse2.packuswb.128"(as(i16x8, a), as(i16x8, b)));
}

/// `_mm256_packus_epi16`, per 128-bit half.
pub inline fn mm256_packus_epi16(a: m256i, b: m256i) m256i {
    return as(m256i, @"llvm.x86.avx2.packuswb"(as(i16x16, a), as(i16x16, b)));
}

pub inline fn mm_packs_epi16(a: m128i, b: m128i) m128i {
    return as(m128i, @"llvm.x86.sse2.packsswb.128"(as(i16x8, a), as(i16x8, b)));
}

pub inline fn mm_packs_epi32(a: m128i, b: m128i) m128i {
    return as(m128i, @"llvm.x86.sse2.packssdw.128"(as(i32x4, a), as(i32x4, b)));
}

/// `_mm256_packs_epi16`, per 128-bit half.
pub inline fn mm256_packs_epi16(a: m256i, b: m256i) m256i {
    return as(m256i, @"llvm.x86.avx2.packsswb"(as(i16x16, a), as(i16x16, b)));
}

/// `_mm256_packs_epi32`, per 128-bit half.
pub inline fn mm256_packs_epi32(a: m256i, b: m256i) m256i {
    return as(m256i, @"llvm.x86.avx2.packssdw"(as(i32x8, a), as(i32x8, b)));
}

/// `_mm256_permutevar8x32_epi32(a, idx)`: a full 32-byte permute of `i32`
/// lanes, by the low three bits of each index.
pub inline fn mm256_permutevar8x32_epi32(a: m256i, idx: m256i) m256i {
    return as(m256i, @"llvm.x86.avx2.permd"(as(i32x8, a), as(i32x8, idx)));
}

pub inline fn mm256_movemask_epi8(a: m256i) i32 {
    return @"llvm.x86.avx2.pmovmskb"(as(i8x32, a));
}

pub inline fn mm_movemask_epi8(a: m128i) i32 {
    return @"llvm.x86.sse2.pmovmskb.128"(as(i8x16, a));
}

/// `_mm256_cvtps_epi32`: to integer under the current rounding mode, which
/// is round-to-nearest-even; `0x80000000` for anything out of range.
pub inline fn mm256_cvtps_epi32(a: m256) m256i {
    return as(m256i, @"llvm.x86.avx.cvt.ps2dq.256"(a));
}

/// `_MM_ROUND_NEAREST`: as a `round` immediate, round to nearest even.
pub const MM_ROUND_NEAREST: i32 = 0x0000;

/// `_mm256_round_ps(a, imm)`.
pub inline fn mm256_round_ps(a: m256, comptime imm: i32) m256 {
    return @"llvm.x86.avx.round.ps.256"(a, imm);
}

pub inline fn mm_hadd_ps(a: m128, b: m128) m128 {
    return @"llvm.x86.sse3.hadd.ps"(a, b);
}

pub inline fn mm256_hadd_ps(a: m256, b: m256) m256 {
    return @"llvm.x86.avx.hadd.ps.256"(a, b);
}

// -----------------------------------------------------------------------------
// BMI2

/// `_pdep_u64(src, mask)`: the low bits of `src` deposited, in order, at the
/// set bits of `mask`.
pub inline fn pdep_u64(src: u64, mask: u64) u64 {
    return @"llvm.x86.bmi.pdep.64"(src, mask);
}

// -----------------------------------------------------------------------------
// Float

/// `_mm256_cvtepi32_ps`: `cvtdq2ps`, rounding to nearest even, as
/// `@floatFromInt` does.
pub inline fn mm256_cvtepi32_ps(a: m256i) m256 {
    return @floatFromInt(as(i32x8, a));
}

pub inline fn mm_cvtepi32_ps(a: m128i) m128 {
    return @floatFromInt(as(i32x4, a));
}

pub inline fn mm256_add_ps(a: m256, b: m256) m256 {
    return a + b;
}

pub inline fn mm256_sub_ps(a: m256, b: m256) m256 {
    return a - b;
}

pub inline fn mm256_mul_ps(a: m256, b: m256) m256 {
    return a * b;
}

/// `_mm256_fmadd_ps(a, b, c)`: `a * b + c`, one rounding.
pub inline fn mm256_fmadd_ps(a: m256, b: m256, c: m256) m256 {
    return @mulAdd(m256, a, b, c);
}

/// `_mm_fmadd_ps(a, b, c)`: `a * b + c`, one rounding.
pub inline fn mm_fmadd_ps(a: m128, b: m128, c: m128) m128 {
    return @mulAdd(m128, a, b, c);
}

/// `_mm256_andnot_ps(a, b)`: `~a & b` on the bits.
pub inline fn mm256_andnot_ps(a: m256, b: m256) m256 {
    return as(m256, ~as(u32x8, a) & as(u32x8, b));
}

/// `_mm256_or_ps`, `_mm256_and_ps`: on the bits.
pub inline fn mm256_or_ps(a: m256, b: m256) m256 {
    return as(m256, as(u32x8, a) | as(u32x8, b));
}

pub inline fn mm256_and_ps(a: m256, b: m256) m256 {
    return as(m256, as(u32x8, a) & as(u32x8, b));
}

/// `_mm256_cmp_ps(a, b, _CMP_EQ_OQ)`: all ones where equal and neither is
/// NaN.
pub inline fn mm256_cmpeq_oq_ps(a: m256, b: m256) m256 {
    return as(m256, @select(u32, a == b, @as(u32x8, @splat(0xffffffff)), @as(u32x8, @splat(0))));
}

/// `_mm256_movemask_ps`: the eight sign bits.
pub inline fn mm256_movemask_ps(a: m256) i32 {
    return @"llvm.x86.avx.movmsk.ps.256"(a);
}

/// `_mm256_max_ps(a, b)`: `a > b ? a : b` per lane -- `b` when either is NaN
/// or both are zero, which is not `@max`.
pub inline fn mm256_max_ps(a: m256, b: m256) m256 {
    return @select(f32, a > b, a, b);
}

pub inline fn mm_max_ps(a: m128, b: m128) m128 {
    return @select(f32, a > b, a, b);
}

/// `_mm_max_ss`: lane 0 as `_mm_max_ps`, the rest from `a`.
pub inline fn mm_max_ss(a: m128, b: m128) m128 {
    var r = a;
    r[0] = if (a[0] > b[0]) a[0] else b[0];
    return r;
}

pub inline fn mm_add_ps(a: m128, b: m128) m128 {
    return a + b;
}

/// `_mm_add_ss`: lane 0 summed, the rest from `a`.
pub inline fn mm_add_ss(a: m128, b: m128) m128 {
    var r = a;
    r[0] = a[0] + b[0];
    return r;
}

/// `_mm_movehl_ps(a, b)`: `b`'s high pair, then `a`'s.
pub inline fn mm_movehl_ps(a: m128, b: m128) m128 {
    return .{ b[2], b[3], a[2], a[3] };
}

/// `_mm_movehdup_ps`: the odd lanes, each twice.
pub inline fn mm_movehdup_ps(a: m128) m128 {
    return .{ a[1], a[1], a[3], a[3] };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
