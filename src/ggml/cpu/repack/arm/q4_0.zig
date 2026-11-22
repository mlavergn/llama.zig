//! The NEON `q4_0` × `q8_0` interleaved gemv.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` at v1.0.0
//! (`b1b770d90`). Each declaration below names the C++ it replaces and the
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
const ggml = @import("../../../module.zig");

// The `_generic` bodies, in `cpu/repack/`, that each export below falls to
// without `__ARM_FEATURE_DOTPROD` -- the C's trailing call after its `#if`.
// Zig's `aarch64-linux` baseline, `generic`, takes them; no Apple core does.

inline fn loadq(p: [*]const i8) ggml.cpu.quants.arch.neon.i8x16 {
    return @bitCast(@as(@Vector(16, i8), p[0..16].*));
}

/// `b << 4` on `i8` lanes. Done on the unsigned reinterpretation because
/// the bits shifted out of a signed lane are discarded, which `u8 << 4`
/// says unambiguously and Zig has no `<<%` to say otherwise.
inline fn shl4(v: ggml.cpu.quants.arch.neon.i8x16) ggml.cpu.quants.arch.neon.i8x16 {
    const u: @Vector(16, u8) = @bitCast(v);
    return @bitCast(u << @as(@Vector(16, u3), @splat(4)));
}

/// `b & 0xf0` on `i8` lanes.
inline fn hi4(v: ggml.cpu.quants.arch.neon.i8x16) ggml.cpu.quants.arch.neon.i8x16 {
    const u: @Vector(16, u8) = @bitCast(v);
    return @bitCast(u & @as(@Vector(16, u8), @splat(0xf0)));
}

/// Ports `ggml_gemv_q4_0_4x4_q8_0` (arch/arm/repack.cpp:230 @b1b770d90).
pub export fn ggml_gemv_q4_0_4x4_q8_0(n: c_int, s_out: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    if (comptime !ggml.cpu.quants.arch.neon.has_dotprod) return ggml.cpu.repack.q4_0.ggml_gemv_q4_0_4x4_q8_0_generic(n, s_out, bs, vx, vy, nr, nc);
    const qk: c_int = 32;
    const nb = @divTrunc(n, qk);
    const ncols_interleaved: c_int = 4;

    var b_ptr: [*]const ggml.cpu.repack.blocks.block_q4_0x4 = @ptrCast(@alignCast(vx));
    var s = s_out;

    var col: c_int = 0;
    while (col < nc) : (col += ncols_interleaved) {
        var a_ptr: [*]const ggml.impl.c.block_q8_0 = @ptrCast(@alignCast(vy));
        var acc: ggml.cpu.quants.arch.neon.f32x4 = @splat(0);
        var b: c_int = 0;
        while (b < nb) : (b += 1) {
            const qs: [*]const i8 = @ptrCast(&b_ptr[0].qs);
            const b0 = loadq(qs);
            const b1 = loadq(qs + 16);
            const b2 = loadq(qs + 32);
            const b3 = loadq(qs + 48);
            const bd = ggml.cpu.quants.arch.neon.load_f16x4(@ptrCast(&b_ptr[0].d));

            const aqs: [*]const i8 = @ptrCast(&a_ptr[0].qs);
            const a0 = loadq(aqs);
            const a1 = loadq(aqs + @as(usize, @intCast(@divTrunc(qk, 2))));
            const ad = ggml.cpu.quants.arch.neon.dup_f16x4(a_ptr[0].d);

            var ret: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);
            ret = ggml.cpu.quants.arch.neon.dotq_laneq_s32(ret, shl4(b0), a0, 0);
            ret = ggml.cpu.quants.arch.neon.dotq_laneq_s32(ret, shl4(b1), a0, 1);
            ret = ggml.cpu.quants.arch.neon.dotq_laneq_s32(ret, shl4(b2), a0, 2);
            ret = ggml.cpu.quants.arch.neon.dotq_laneq_s32(ret, shl4(b3), a0, 3);
            ret = ggml.cpu.quants.arch.neon.dotq_laneq_s32(ret, hi4(b0), a1, 0);
            ret = ggml.cpu.quants.arch.neon.dotq_laneq_s32(ret, hi4(b1), a1, 1);
            ret = ggml.cpu.quants.arch.neon.dotq_laneq_s32(ret, hi4(b2), a1, 2);
            ret = ggml.cpu.quants.arch.neon.dotq_laneq_s32(ret, hi4(b3), a1, 3);

            acc = ggml.cpu.quants.arch.neon.fma_f32(acc, ggml.cpu.quants.arch.neon.cvtq_n_f32_s32(ret, 4), ggml.cpu.quants.arch.neon.cvt_f32_f16(ad) * ggml.cpu.quants.arch.neon.cvt_f32_f16(bd));
            a_ptr += 1;
            b_ptr += 1;
        }
        s[0..4].* = acc;
        s += @intCast(ncols_interleaved);
    }
}

/// Ports `ggml_gemv_q4_0_4x8_q8_0` (arch/arm/repack.cpp:291 @b1b770d90).
pub export fn ggml_gemv_q4_0_4x8_q8_0(n: c_int, s_out: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    if (comptime !ggml.cpu.quants.arch.neon.has_dotprod) return ggml.cpu.repack.q4_0.ggml_gemv_q4_0_4x8_q8_0_generic(n, s_out, bs, vx, vy, nr, nc);
    const nb = @divTrunc(n, 32);
    const ncols_interleaved: c_int = 4;

    var b_ptr: [*]const ggml.cpu.repack.blocks.block_q4_0x4 = @ptrCast(@alignCast(vx));
    var s = s_out;

    var col: c_int = 0;
    while (col < nc) : (col += ncols_interleaved) {
        var a_ptr: [*]const ggml.impl.c.block_q8_0 = @ptrCast(@alignCast(vy));
        var acc: ggml.cpu.quants.arch.neon.f32x4 = @splat(0);
        var b: c_int = 0;
        while (b < nb) : (b += 1) {
            const qs: [*]const i8 = @ptrCast(&b_ptr[0].qs);
            const b0 = loadq(qs);
            const b1 = loadq(qs + 16);
            const b2 = loadq(qs + 32);
            const b3 = loadq(qs + 48);
            const bd = ggml.cpu.quants.arch.neon.load_f16x4(@ptrCast(&b_ptr[0].d));

            const aqs: [*]const i8 = @ptrCast(&a_ptr[0].qs);
            const a0 = ggml.cpu.quants.arch.neon.dupq_i8x16_from8(aqs);
            const a1 = ggml.cpu.quants.arch.neon.dupq_i8x16_from8(aqs + 8);
            const a2 = ggml.cpu.quants.arch.neon.dupq_i8x16_from8(aqs + 16);
            const a3 = ggml.cpu.quants.arch.neon.dupq_i8x16_from8(aqs + 24);
            const ad = ggml.cpu.quants.arch.neon.dup_f16x4(a_ptr[0].d);

            var ret0: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);
            var ret1: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);
            ret0 = ggml.cpu.quants.arch.neon.dotq_s32(ret0, shl4(b0), a0);
            ret1 = ggml.cpu.quants.arch.neon.dotq_s32(ret1, shl4(b1), a0);
            ret0 = ggml.cpu.quants.arch.neon.dotq_s32(ret0, shl4(b2), a1);
            ret1 = ggml.cpu.quants.arch.neon.dotq_s32(ret1, shl4(b3), a1);
            ret0 = ggml.cpu.quants.arch.neon.dotq_s32(ret0, hi4(b0), a2);
            ret1 = ggml.cpu.quants.arch.neon.dotq_s32(ret1, hi4(b1), a2);
            ret0 = ggml.cpu.quants.arch.neon.dotq_s32(ret0, hi4(b2), a3);
            ret1 = ggml.cpu.quants.arch.neon.dotq_s32(ret1, hi4(b3), a3);

            const ret = ggml.cpu.quants.arch.neon.paddq_s32(ret0, ret1);

            acc = ggml.cpu.quants.arch.neon.fma_f32(acc, ggml.cpu.quants.arch.neon.cvtq_n_f32_s32(ret, 4), ggml.cpu.quants.arch.neon.cvt_f32_f16(ad) * ggml.cpu.quants.arch.neon.cvt_f32_f16(bd));
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
