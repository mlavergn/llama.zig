//! The NEON `q4_0` × `q8_0` interleaved gemv.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # The nibbles are shifted into place, not masked down
//!
//! `b << 4` puts the low nibble in a byte's high half; `b & 0xf0` leaves
//! the high nibble there. Both operands are therefore 16× too large, and
//! the kernel corrects at the end with `vcvtq_n_f32_s32(ret, 4)` — convert
//! and divide by 16 in one instruction.
//!
//! **That is not the scalar path's correction.** `repack/q4_0.zig` shifts
//! the integer sum right by four, which truncates; this scales an exact
//! integer. The two agree only because the sum is a multiple of 16 before
//! either is applied — and the NEON one is the live kernel.
//!
//! **NEON integer arithmetic wraps**, so `<<` here is `<<%` in Zig terms;
//! `@shlExact` would be wrong, and the shift is on `i8` lanes where the
//! sign bit moves out.
//!
//! # The two shapes differ in how the activation is fed
//!
//! `4x4` broadcasts a 32-bit lane with `vdotq_laneq_s32`, one accumulator.
//! `4x8` broadcasts an 8-byte group into both halves of a vector and uses
//! two accumulators merged by the **pairwise** `vpaddq_s32`.

const std = @import("std");
const impl = @import("../../../impl.zig");
const blocks = @import("../blocks.zig");
const neon = @import("../../quants/arm/neon.zig");

const c = impl.c;
const f32x4 = neon.f32x4;
const i32x4 = neon.i32x4;
const i8x16 = neon.i8x16;

inline fn loadq(p: [*]const i8) i8x16 {
    return @bitCast(@as(@Vector(16, i8), p[0..16].*));
}

/// `b << 4` on `i8` lanes. Done on the unsigned reinterpretation because
/// the bits shifted out of a signed lane are discarded, which `u8 << 4`
/// says unambiguously and Zig has no `<<%` to say otherwise.
inline fn shl4(v: i8x16) i8x16 {
    const u: @Vector(16, u8) = @bitCast(v);
    return @bitCast(u << @as(@Vector(16, u3), @splat(4)));
}

/// `b & 0xf0` on `i8` lanes.
inline fn hi4(v: i8x16) i8x16 {
    const u: @Vector(16, u8) = @bitCast(v);
    return @bitCast(u & @as(@Vector(16, u8), @splat(0xf0)));
}

/// Ports `ggml_gemv_q4_0_4x4_q8_0` (arch/arm/repack.cpp:212 @c1d0e7a00).
pub export fn ggml_gemv_q4_0_4x4_q8_0(n: c_int, s_out: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    _ = bs;
    _ = nr;
    const qk: c_int = 32;
    const nb = @divTrunc(n, qk);
    const ncols_interleaved: c_int = 4;

    var b_ptr: [*]const blocks.block_q4_0x4 = @ptrCast(@alignCast(vx));
    var s = s_out;

    var col: c_int = 0;
    while (col < nc) : (col += ncols_interleaved) {
        var a_ptr: [*]const c.block_q8_0 = @ptrCast(@alignCast(vy));
        var acc: f32x4 = @splat(0);
        var b: c_int = 0;
        while (b < nb) : (b += 1) {
            const qs: [*]const i8 = @ptrCast(&b_ptr[0].qs);
            const b0 = loadq(qs);
            const b1 = loadq(qs + 16);
            const b2 = loadq(qs + 32);
            const b3 = loadq(qs + 48);
            const bd = neon.load_f16x4(@ptrCast(&b_ptr[0].d));

            const aqs: [*]const i8 = @ptrCast(&a_ptr[0].qs);
            const a0 = loadq(aqs);
            const a1 = loadq(aqs + @as(usize, @intCast(@divTrunc(qk, 2))));
            const ad = neon.dup_f16x4(a_ptr[0].d);

            var ret: i32x4 = @splat(0);
            ret = neon.dotq_laneq_s32(ret, shl4(b0), a0, 0);
            ret = neon.dotq_laneq_s32(ret, shl4(b1), a0, 1);
            ret = neon.dotq_laneq_s32(ret, shl4(b2), a0, 2);
            ret = neon.dotq_laneq_s32(ret, shl4(b3), a0, 3);
            ret = neon.dotq_laneq_s32(ret, hi4(b0), a1, 0);
            ret = neon.dotq_laneq_s32(ret, hi4(b1), a1, 1);
            ret = neon.dotq_laneq_s32(ret, hi4(b2), a1, 2);
            ret = neon.dotq_laneq_s32(ret, hi4(b3), a1, 3);

            acc = neon.fma_f32(acc, neon.cvtq_n_f32_s32(ret, 4), neon.cvt_f32_f16(ad) * neon.cvt_f32_f16(bd));
            a_ptr += 1;
            b_ptr += 1;
        }
        s[0..4].* = acc;
        s += @intCast(ncols_interleaved);
    }
}

/// Ports `ggml_gemv_q4_0_4x8_q8_0` (arch/arm/repack.cpp:273 @c1d0e7a00).
pub export fn ggml_gemv_q4_0_4x8_q8_0(n: c_int, s_out: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    _ = bs;
    _ = nr;
    const nb = @divTrunc(n, 32);
    const ncols_interleaved: c_int = 4;

    var b_ptr: [*]const blocks.block_q4_0x4 = @ptrCast(@alignCast(vx));
    var s = s_out;

    var col: c_int = 0;
    while (col < nc) : (col += ncols_interleaved) {
        var a_ptr: [*]const c.block_q8_0 = @ptrCast(@alignCast(vy));
        var acc: f32x4 = @splat(0);
        var b: c_int = 0;
        while (b < nb) : (b += 1) {
            const qs: [*]const i8 = @ptrCast(&b_ptr[0].qs);
            const b0 = loadq(qs);
            const b1 = loadq(qs + 16);
            const b2 = loadq(qs + 32);
            const b3 = loadq(qs + 48);
            const bd = neon.load_f16x4(@ptrCast(&b_ptr[0].d));

            const aqs: [*]const i8 = @ptrCast(&a_ptr[0].qs);
            const a0 = neon.dupq_i8x16_from8(aqs);
            const a1 = neon.dupq_i8x16_from8(aqs + 8);
            const a2 = neon.dupq_i8x16_from8(aqs + 16);
            const a3 = neon.dupq_i8x16_from8(aqs + 24);
            const ad = neon.dup_f16x4(a_ptr[0].d);

            var ret0: i32x4 = @splat(0);
            var ret1: i32x4 = @splat(0);
            ret0 = neon.dotq_s32(ret0, shl4(b0), a0);
            ret1 = neon.dotq_s32(ret1, shl4(b1), a0);
            ret0 = neon.dotq_s32(ret0, shl4(b2), a1);
            ret1 = neon.dotq_s32(ret1, shl4(b3), a1);
            ret0 = neon.dotq_s32(ret0, hi4(b0), a2);
            ret1 = neon.dotq_s32(ret1, hi4(b1), a2);
            ret0 = neon.dotq_s32(ret0, hi4(b2), a3);
            ret1 = neon.dotq_s32(ret1, hi4(b3), a3);

            const ret = neon.paddq_s32(ret0, ret1);

            acc = neon.fma_f32(acc, neon.cvtq_n_f32_s32(ret, 4), neon.cvt_f32_f16(ad) * neon.cvt_f32_f16(bd));
            a_ptr += 1;
            b_ptr += 1;
        }
        s[0..4].* = acc;
        s += @intCast(ncols_interleaved);
    }
}

comptime {
    _ = std;
}
