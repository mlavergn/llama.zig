//! The tiled K-quant matmul's tiles and microkernels.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/tiled/tiled-kernel.h` and
//! `llama.cpp/ggml/src/ggml-cpu/tiled/tiled-kernel.cpp` (v0.8.0,
//! `8642541b2`), new at v0.6.0 and replacing `iqp.cpp`. Each declaration
//! names the C it replaces and the line it began at; the driver that calls
//! these is `tiled.zig`.
//!
//! # Three arms, chosen at comptime as the C's `#if` chooses
//!
//! `tiled_run_microtile`, `tiled_repack_src1` and `tiled_repack_src0` each
//! have an arm per ISA:
//!
//! - `__AVX512VNNI__ && __AVX512VL__ && __AVX512DQ__` (`tigerlake`):
//!   `vpdpbusd` over 512 bits, the activations transposed into a
//!   `[k/4][row][4]` interleave, two 8x16 band passes per microtile.
//! - `__AVX2__` (x86-64-v3, `x86_64_v4`): `maddubs` then `madd`, the
//!   activations pre-biased by 128 for the formats whose debiased weights
//!   fit, the `sign` trick for `iq4_xs`.
//! - neither: the scalar reference, and repacks that do nothing.
//!
//! The `__AVX__ && !__AVX2__` arm is not ported, as nothing below
//! x86-64-v3 is (`simd.zig` refuses the target).
//!
//! **Only the x86 arms are ever live**: `tiled.zig`'s `ggml_tiled_supported`
//! is a stub returning false off AVX, so on ARM nothing here runs. Every arm
//! is nevertheless written in portable Zig -- each intrinsic as its exact
//! integer semantics, `maddubs`'s saturation and `sign`'s wrap included,
//! with the real instruction substituted where the build has it -- so all
//! three compile on every target and the unit tests hold them against each
//! other and against `vec_dot` on this Mac.
//!
//! # The float arithmetic
//!
//! Everything up to the per-row result is integer, so the arms agree there
//! except where an instruction's own semantics bite (`maddubs` saturating,
//! `sign` of -128). The epilogues are written in the C with intrinsics and
//! named here exactly: the AVX2 arm's `_mm256_fnmadd_ps` and
//! `_mm256_fmadd_ps` are fused, the VNNI arm's final `_mm512_add_ps` of a
//! `_mm512_mul_ps` is not. Clang contracts operators within one expression,
//! not across intrinsic calls, so neither depends on `-ffp-contract`. The
//! scalar arm is unreachable in every build and stays plain.
//!
//! # Loop index names
//!
//! `i0` and `i1` are Zig integer type names; the C's `i0` is `ri0` here.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("../impl.zig");
const qblocks = @import("../quants/blocks.zig");
const c = impl.c;

/// Ports `TILED_TILE_K` (tiled-kernel.h:22 @8642541b2): one `QK_K` block.
pub const tile_k = 256;

/// Ports `TILED_TILE_ROWS` (tiled-kernel.h:23 @8642541b2): the most rows a
/// window holds; ragged at the edges.
pub const tile_rows = 256;

/// Ports `TILED_MICRO` (tiled-kernel.h:24 @8642541b2): the microtile edge,
/// and the granularity of `q8_K`'s `bsums`.
pub const micro = 16;

/// Ports `TILED_WS_SLOT` (tiled-kernel.h:25 @8642541b2): one thread's
/// workspace, a clean 512 KB.
pub const ws_slot = 512 * 1024;

/// Ports `NB_MAX` (tiled-kernel.h:31 @8642541b2): the most subblocks a
/// 256-element block has (16-wide ones).
pub const nb_max = tile_k / 16;

/// Ports `struct tiled_tile_src0` (tiled-kernel.h:30 @8642541b2): the weight
/// side of a window, shared by every format.
///
/// `scales` and `mins` are sized for 16-wide subblocks; the 32-wide formats
/// index them at stride 8 and leave the rest unused.
pub const TileSrc0 = extern struct {
    /// Unsigned quants, widened to `u8`.
    q: [tile_rows * tile_k]u8 align(64),
    /// One `d` per input block, widened to `f32`.
    d: [tile_rows]f32,
    /// `dmin` per input block, where the format has one.
    dmin: [tile_rows]f32,
    /// Per-subblock scale.
    scales: [tile_rows * nb_max]i32,
    /// Per-subblock min where `HAS_MIN`; the act-bias correction in slot 0
    /// of each row's block otherwise (see `repackSrc0`).
    mins: [tile_rows * nb_max]i32,
};

/// Ports `struct tiled_tile_src1` (tiled-kernel.h:41 @8642541b2): the
/// activation side, built from `q8_K` rows.
pub const TileSrc1 = extern struct {
    /// `q8` codes, one byte per element; transposed for `vpdpbusd` on VNNI.
    q: [tile_rows * tile_k]i8 align(64),
    /// The per-16 code sums from `q8_K`, widened to `i32`.
    bsums: [(tile_k / 16) * tile_rows]i32 align(64),
    /// `q8_K`'s `d`.
    d: [tile_rows]f32,
};

/// Ports `struct tiled_ws` (tiled-kernel.h:51 @8642541b2): one thread's
/// workspace, carved out of `wdata`.
pub const Ws = extern struct {
    src0: TileSrc0,
    src1: TileSrc1,
    acc: [tile_rows * tile_rows]f32 align(64),
};

comptime {
    // tiled-kernel.h's three static_asserts.
    std.debug.assert(@sizeOf(Ws) <= ws_slot);
    std.debug.assert(@offsetOf(Ws, "src0") % 64 == 0);
    std.debug.assert(@offsetOf(Ws, "src1") % 64 == 0);
    std.debug.assert(@offsetOf(Ws, "acc") % 64 == 0);
    // tiled-kernel.cpp's: the repacks read `q8_K` in 4-byte words.
    std.debug.assert(@sizeOf(qblocks.Q8_K) == 292 and @offsetOf(qblocks.Q8_K, "qs") == 4);
}

// -----------------------------------------------------------------------------
// Which arm this build compiles

fn x86Has(comptime f: std.Target.x86.Feature) bool {
    return builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, f);
}

/// One of the C's `#if` arms.
pub const Arm = enum { scalar, avx2, vnni };

/// The arm `tiled_run_microtile` takes on this build.
pub const arm: Arm = blk: {
    if (x86Has(.avx512vnni) and x86Has(.avx512vl) and x86Has(.avx512dq)) break :blk .vnni;
    if (x86Has(.avx2)) {
        // The C's `__FMA__` arms: x86-64-v3 always has FMA, and the
        // non-FMA epilogue is not ported.
        if (!x86Has(.fma)) @compileError("the non-FMA epilogue of tiled-kernel.cpp is not ported");
        break :blk .avx2;
    }
    if (x86Has(.avx)) @compileError("the __AVX__ && !__AVX2__ arm of tiled-kernel.cpp is not ported");
    break :blk .scalar;
};

/// `__AVX2__`: also what `tiled_repack_src0` keys on, so the VNNI arm takes
/// it too.
pub const has_avx2 = arm != .scalar;

/// Whether the real AVX2 and AVX512-VNNI instructions can be emitted.
const native_avx2 = x86Has(.avx2);
const native_vnni512 = x86Has(.avx512vnni) and x86Has(.avx512bw);

// -----------------------------------------------------------------------------
// The intrinsics, as their exact integer semantics

const u8x32 = @Vector(32, u8);
const i8x32 = @Vector(32, i8);
const i16x16 = @Vector(16, i16);
const i32x8 = @Vector(8, i32);
const f32x8 = @Vector(8, f32);
const i8x64 = @Vector(64, i8);
const u8x64 = @Vector(64, u8);
const i32x16 = @Vector(16, i32);
const f32x16 = @Vector(16, f32);

const sysv: std.builtin.CallingConvention = .{ .x86_64_sysv = .{} };
extern fn @"llvm.x86.avx2.pmadd.ub.sw"(u8x32, i8x32) callconv(sysv) i16x16;
extern fn @"llvm.x86.avx2.pmadd.wd"(i16x16, i16x16) callconv(sysv) i32x8;
extern fn @"llvm.x86.avx2.psign.b"(i8x32, i8x32) callconv(sysv) i8x32;
extern fn @"llvm.x86.avx2.phadd.d"(i32x8, i32x8) callconv(sysv) i32x8;
extern fn @"llvm.x86.avx512.vpdpbusd.512"(i32x16, i8x64, i8x64) callconv(sysv) i32x16;

/// Loads `n` lanes through a vector pointer (an array bitcast to a vector
/// compiles to a lane-by-lane gather on x86).
inline fn load(comptime V: type, p: anytype) V {
    return @as(*align(1) const V, @ptrCast(p)).*;
}

inline fn store(comptime V: type, p: anytype, v: V) void {
    @as(*align(1) V, @ptrCast(p)).* = v;
}

/// `_mm256_maddubs_epi16(a, b)`: `a` unsigned, `b` signed, neighbouring
/// products summed into `i16` with signed saturation.
inline fn maddubs(a: u8x32, b: i8x32) i16x16 {
    if (comptime native_avx2) return @"llvm.x86.avx2.pmadd.ub.sw"(a, b);
    var out: i16x16 = undefined;
    inline for (0..16) |k| {
        const s = @as(i32, a[2 * k]) * @as(i32, b[2 * k]) + @as(i32, a[2 * k + 1]) * @as(i32, b[2 * k + 1]);
        out[k] = @intCast(std.math.clamp(s, std.math.minInt(i16), std.math.maxInt(i16)));
    }
    return out;
}

/// `_mm256_madd_epi16(a, b)`: neighbouring `i16` products summed into `i32`,
/// wrapping (only `-32768 * -32768` twice can).
inline fn madd(a: i16x16, b: i16x16) i32x8 {
    if (comptime native_avx2) return @"llvm.x86.avx2.pmadd.wd"(a, b);
    var out: i32x8 = undefined;
    inline for (0..8) |k| {
        const s = @as(i64, a[2 * k]) * b[2 * k] + @as(i64, a[2 * k + 1]) * b[2 * k + 1];
        out[k] = @truncate(s);
    }
    return out;
}

/// `_mm256_sign_epi8(a, b)`: `a` negated (wrapping, so -128 stays -128)
/// where `b` is negative, zero where `b` is zero.
inline fn sign(a: i8x32, b: i8x32) i8x32 {
    if (comptime native_avx2) return @"llvm.x86.avx2.psign.b"(a, b);
    const zero: i8x32 = @splat(0);
    return @select(i8, b < zero, zero -% a, @select(i8, b == zero, zero, a));
}

/// `_mm512_dpbusd_epi32(src, a, b)`: each lane plus the four products of an
/// unsigned byte of `a` and the signed byte of `b` beside it, no saturation.
inline fn dpbusd512(src: i32x16, a: u8x64, b: i8x64) i32x16 {
    if (comptime native_vnni512) return @"llvm.x86.avx512.vpdpbusd.512"(src, @bitCast(a), b);
    var out = src;
    inline for (0..16) |l| {
        var s: i32 = 0;
        inline for (0..4) |k| s +%= @as(i32, a[4 * l + k]) * @as(i32, b[4 * l + k]);
        out[l] +%= s;
    }
    return out;
}

/// `_mm256_hadd_epi32(a, b)`: per 128-bit half, `a`'s neighbouring pairs, then
/// `b`'s, wrapping.
inline fn hadd(a: i32x8, b: i32x8) i32x8 {
    if (comptime native_avx2) return @"llvm.x86.avx2.phadd.d"(a, b);
    const even = @shuffle(i32, a, b, [8]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2), 4, 6, ~@as(i32, 4), ~@as(i32, 6) });
    const odd = @shuffle(i32, a, b, [8]i32{ 1, 3, ~@as(i32, 1), ~@as(i32, 3), 5, 7, ~@as(i32, 5), ~@as(i32, 7) });
    return even +% odd;
}

/// The reduction `tiled_run_microtile_avx2` ends each group on
/// (tiled-kernel.cpp:227 @8642541b2): lane `t` is the wrapping sum of
/// `acc[t]`'s eight lanes. Two rounds of `hadd` leave four column sums in
/// each 128-bit half, over input lanes 0..3 and 4..7, and the halves are
/// `_mm256_permute2x128_si256`'d apart and added.
inline fn columnSums(acc: *const [8]i32x8) i32x8 {
    const ab = hadd(acc[0], acc[1]);
    const cd = hadd(acc[2], acc[3]);
    const ef = hadd(acc[4], acc[5]);
    const gh = hadd(acc[6], acc[7]);

    const abcd = hadd(ab, cd);
    const efgh = hadd(ef, gh);
    const lo = @shuffle(i32, abcd, efgh, [8]i32{ 0, 1, 2, 3, ~@as(i32, 0), ~@as(i32, 1), ~@as(i32, 2), ~@as(i32, 3) });
    const hi = @shuffle(i32, abcd, efgh, [8]i32{ 4, 5, 6, 7, ~@as(i32, 4), ~@as(i32, 5), ~@as(i32, 6), ~@as(i32, 7) });
    return lo +% hi;
}

// -----------------------------------------------------------------------------
// Unpack primitives
//
// Each has an `__AVX2__` arm and a scalar `#else`. The AVX2 arms are byte-wise
// identical to the scalar ones on every input: shifts are masked before they
// can cross a byte, `_mm_shuffle_epi8` sees only indices below 16,
// `_mm_packus_epi16` only values in [119, 137], and `sign32`'s `v ^ m - m`
// is the negation the scalar spells out. One portable body serves both.

/// Ports `tiled_unpk_nib4` (tiled-kernel.h:129 @8642541b2), the scalar arm,
/// and so `tiled_unpk_nib4` (tiled-kernel.h:68 @8642541b2), the `__AVX2__`
/// one: packed 4-bit codes to low and high nibbles.
pub inline fn unpkNib4(src: [*]const u8, lo: [*]u8, hi: [*]u8) void {
    const v = load(u8x32, src);
    store(u8x32, lo, v & @as(u8x32, @splat(0x0F)));
    store(u8x32, hi, v >> @splat(4));
}

/// Ports `tiled_unpk_2bit` (tiled-kernel.h:133 @8642541b2), and its
/// `__AVX2__` arm `tiled_unpk_2bit` (tiled-kernel.h:75 @8642541b2): the
/// 2-bit values at bit offset `S`.
pub inline fn unpk2bit(comptime S: u3, src: [*]const u8, dst: [*]u8) void {
    store(u8x32, dst, (load(u8x32, src) >> @splat(S)) & @as(u8x32, @splat(3)));
}

/// Ports `tiled_unpk_or` (tiled-kernel.h:137 @8642541b2), and its `__AVX2__`
/// arm `tiled_unpk_or` (tiled-kernel.h:81 @8642541b2): ORs the `M`-bit
/// value at bit offset `S` of `src` into bit offset `D` of `dst`.
pub inline fn unpkOr(comptime S: u3, comptime D: u3, comptime M: u8, dst: [*]u8, src: [*]const u8) void {
    const v = ((load(u8x32, src) >> @splat(S)) & @as(u8x32, @splat(M))) << @splat(D);
    store(u8x32, dst, load(u8x32, dst) | v);
}

/// Ports `tiled_lut8` (tiled-kernel.h:140 @8642541b2), and its `__AVX2__`
/// arm `tiled_lut8` (tiled-kernel.h:93 @8642541b2): `dst[j] = lut[src[j]]`
/// for 16 bytes.
pub inline fn lut8(lut: *const [16]u8, src: [*]const u8, dst: [*]u8) void {
    for (0..16) |j| dst[j] = lut[src[j]];
}

/// Ports `tiled_unpk_sign32` (tiled-kernel.h:143 @8642541b2), and its
/// `__AVX2__` arm `tiled_unpk_sign32` (tiled-kernel.h:101 @8642541b2): 32
/// grid magnitudes and four sign bytes to codes biased by 128.
pub inline fn unpkSign32(g: [4]u64, signs: [*]const u8, dst32: [*]u8) void {
    for (0..4) |l| {
        const v: [8]u8 = @bitCast(std.mem.nativeToLittle(u64, g[l]));
        const s = signs[l];
        for (0..8) |j| {
            dst32[8 * l + j] = if (s & (@as(u8, 1) << @intCast(j)) != 0) 128 -% v[j] else 128 +% v[j];
        }
    }
}

/// Ports `tiled_unpk_tern8` (tiled-kernel.h:155 @8642541b2), and its
/// `__AVX2__` arm `tiled_unpk_tern8` (tiled-kernel.h:119 @8642541b2): eight
/// ternary grid bytes to `128 + delta + 8 * (int8_t) src[j]`.
pub inline fn unpkTern8(src: *const [8]u8, delta: i8, dst: [*]u8) void {
    for (0..8) |j| {
        const sv: i8 = @bitCast(src[j]);
        dst[j] = @truncate(@as(u32, @bitCast(128 + @as(i32, delta) + 8 * @as(i32, sv))));
    }
}

// -----------------------------------------------------------------------------
// The microtile

/// The strides and offsets every microtile arm opens with. `NK`: 1 for the
/// standard single-slab path, whose strides are compile-time; 0 for the
/// narrow path's runtime `num_k` and `slab`.
const Geometry = struct {
    qk_stride: usize,
    qk_off: usize,
    nb_stride: usize,
    nb_off: usize,
    bs_stride: usize,
    bs_off: usize,
    seff: usize,

    inline fn of(comptime NB: usize, comptime NK: c_int, num_k: c_int, slab: c_int) Geometry {
        const nkr: usize = if (NK > 0) NK else @intCast(num_k);
        const seff: usize = if (NK > 0) 0 else @intCast(slab);
        return .{
            .qk_stride = nkr * tile_k,
            .qk_off = seff * tile_k,
            .nb_stride = NB * nkr,
            .nb_off = seff * NB,
            .bs_stride = if (nkr == 1) tile_rows else micro,
            .bs_off = seff * micro,
            .seff = seff,
        };
    }
};

/// Ports `tiled_run_microtile_scalar` (tiled-kernel.cpp:16 @8642541b2): the
/// reference, exact integer subdots per subblock, folded into `f32` one
/// subblock at a time. The `#else` arm of `tiled_run_microtile`, never
/// reached (see the file header), so plain.
fn microScalar(
    comptime SUBBLK: usize,
    comptime HAS_MIN: bool,
    comptime BIAS: i32,
    comptime NK: c_int,
    src0: *const TileSrc0,
    src1: *const TileSrc1,
    ri0: c_int,
    j0: c_int,
    num_k: c_int,
    slab: c_int,
    buf: [*]f32,
    buf_stride: c_int,
) void {
    const NB = tile_k / SUBBLK;
    const NS = SUBBLK / 16;
    const gm = Geometry.of(NB, NK, num_k, slab);
    const bstride: usize = @intCast(buf_stride);

    var acc: [micro][micro]f32 = @splat(@splat(0));

    for (0..NB) |s| {
        for (0..micro) |j| {
            const br: usize = @as(usize, @intCast(j0)) + j;
            const q1 = src1.q[br * gm.qk_stride + gm.qk_off + s * SUBBLK ..];
            var bsum: i32 = 0;
            for (0..NS) |u| bsum +%= src1.bsums[(gm.bs_off + s * NS + u) * gm.bs_stride + br];

            for (0..micro) |i| {
                const ar: usize = @as(usize, @intCast(ri0)) + i;
                const d_off = gm.seff * micro + ar;
                const q0 = src0.q[ar * gm.qk_stride + gm.qk_off + s * SUBBLK ..];

                var raw: i32 = 0;
                for (0..SUBBLK) |e| raw +%= @as(i32, q0[e]) * @as(i32, q1[e]);

                var corr = raw;
                if (BIAS != 0) corr -%= BIAS *% bsum;
                const scales_raw = src0.scales[ar * gm.nb_stride + gm.nb_off + s] *% corr;
                if (HAS_MIN) {
                    const mins_bsum = src0.mins[ar * gm.nb_stride + gm.nb_off + s] *% bsum;
                    acc[i][j] += src0.d[d_off] * @as(f32, @floatFromInt(scales_raw)) -
                        src0.dmin[d_off] * @as(f32, @floatFromInt(mins_bsum));
                } else {
                    acc[i][j] += src0.d[d_off] * @as(f32, @floatFromInt(scales_raw));
                }
            }
        }
    }

    // Apply d and write out to buf
    for (0..micro) |i| {
        for (0..micro) |j| {
            const row = @as(usize, @intCast(ri0)) + i;
            const col = @as(usize, @intCast(j0)) + j;
            buf[row * bstride + col] += src1.d[gm.seff * micro + col] * acc[i][j];
        }
    }
}

/// Ports `tiled_run_micro_vnni_8x16` (tiled-kernel.cpp:95 @8642541b2): an
/// 8x16 band, `s1 += scales * (raw - BIAS * bsums)` and `s2 += mins * bsums`
/// per subblock in `i32`, then `d*s1 - dmin*s2` per row.
fn microVnni8x16(
    comptime SUBBLK: usize,
    comptime HAS_MIN: bool,
    comptime BIAS: i32,
    comptime NK: c_int,
    src0: *const TileSrc0,
    src1: *const TileSrc1,
    ri0: usize,
    j0: usize,
    num_k: c_int,
    slab: c_int,
    buf: [*]f32,
    buf_stride: usize,
) void {
    const NB = tile_k / SUBBLK;
    const NS = SUBBLK / 16;
    const NG = SUBBLK / 4;
    const NUM_ROWS = 8;
    const gm = Geometry.of(NB, NK, num_k, slab);

    const d1_vec = load(f32x16, &src1.d[gm.seff * micro + j0]);

    var s1_acc: [NUM_ROWS]i32x16 = @splat(@splat(0));
    var s2_acc: [NUM_ROWS]i32x16 = @splat(@splat(0));

    const bsums = &src1.bsums;
    for (0..NB) |s| {
        var bsums32 = load(i32x16, &bsums[(gm.bs_off + s * NS) * gm.bs_stride + j0]);
        for (1..NS) |u| bsums32 +%= load(i32x16, &bsums[(gm.bs_off + s * NS + u) * gm.bs_stride + j0]);

        var bias32: i32x16 = @splat(0);
        if (BIAS != 0) bias32 = bsums32 *% @as(i32x16, @splat(BIAS));

        var acc16: [NUM_ROWS]i32x16 = @splat(@splat(0));

        for (0..NG) |g| {
            const kg = s * NG + g;
            // in-place interleave layout: [kg%16 @ TILED_TILE_K][kg/16 @ 64][row @ 4]
            const codes = load(i8x64, &src1.q[(j0 / micro) * (micro * gm.qk_stride) + gm.qk_off + (kg % micro) * gm.qk_stride + (kg / micro) * (micro * 4)]);
            for (0..NUM_ROWS) |t| {
                const w4 = load(@Vector(4, u8), &src0.q[(ri0 + t) * gm.qk_stride + gm.qk_off + kg * 4]);
                const u4b: u8x64 = @bitCast(@as(@Vector(16, u32), @splat(@bitCast(w4))));
                acc16[t] = dpbusd512(acc16[t], u4b, codes);
            }
        }

        // int correction: s1_acc += scales*(raw-BIAS*bsums)
        for (0..NUM_ROWS) |t| {
            const ar = ri0 + t;
            var rawi = acc16[t];
            if (BIAS != 0) rawi -%= bias32;
            s1_acc[t] +%= rawi *% @as(i32x16, @splat(src0.scales[ar * gm.nb_stride + gm.nb_off + s]));
        }
    }

    // s2_acc += mins*bsums, after the band pass as in the C
    if (HAS_MIN) {
        for (0..NB) |s| {
            var bsums32 = load(i32x16, &bsums[(gm.bs_off + s * NS) * gm.bs_stride + j0]);
            for (1..NS) |u| bsums32 +%= load(i32x16, &bsums[(gm.bs_off + s * NS + u) * gm.bs_stride + j0]);
            for (0..NUM_ROWS) |t| {
                s2_acc[t] +%= bsums32 *% @as(i32x16, @splat(src0.mins[(ri0 + t) * gm.nb_stride + gm.nb_off + s]));
            }
        }
    }

    // epilogue: int->float, apply per-row scales, store to buf
    for (0..NUM_ROWS) |t| {
        const ar = ri0 + t;
        const d_off = gm.seff * micro + ar;
        const f1: f32x16 = @floatFromInt(s1_acc[t]);
        var result = f1 * @as(f32x16, @splat(src0.d[d_off]));
        if (HAS_MIN) {
            // `_mm512_fnmadd_ps`: one rounding
            const f2: f32x16 = @floatFromInt(s2_acc[t]);
            result = @mulAdd(f32x16, @splat(-src0.dmin[d_off]), f2, result);
        }
        const p = buf + (ri0 + t) * buf_stride + j0;
        // `_mm512_add_ps` of a `_mm512_mul_ps`: two roundings
        store(f32x16, p, load(f32x16, p) + result * d1_vec);
    }
}

/// Ports `tiled_run_microtile_vnni` (tiled-kernel.cpp:201 @8642541b2): a
/// 16x16 microtile as two 8x16 band passes.
fn microVnni(
    comptime SUBBLK: usize,
    comptime HAS_MIN: bool,
    comptime BIAS: i32,
    comptime NK: c_int,
    src0: *const TileSrc0,
    src1: *const TileSrc1,
    ri0: c_int,
    j0: c_int,
    num_k: c_int,
    slab: c_int,
    buf: [*]f32,
    buf_stride: c_int,
) void {
    const r: usize = @intCast(ri0);
    const j: usize = @intCast(j0);
    const bs: usize = @intCast(buf_stride);
    microVnni8x16(SUBBLK, HAS_MIN, BIAS, NK, src0, src1, r, j, num_k, slab, buf, bs);
    microVnni8x16(SUBBLK, HAS_MIN, BIAS, NK, src0, src1, r + 8, j, num_k, slab, buf, bs);
}

/// Ports `tiled_run_microtile_avx2` (tiled-kernel.cpp:227 @8642541b2): the
/// existing `vec_dot` algorithms over an 8-column group, one row at a time.
///
/// `SUBBLK` 32: one `maddubs` per subblock and column under a broadcast
/// scale. `SUBBLK` 16: two subblocks per 32-byte load, the scale applied per
/// 128-bit half. With `BIAS` the weight is debiased to `w = q0 - BIAS` and
/// either multiplied by activations pre-biased by 128 (`ACTBIAS`, corrected
/// in the epilogue by `repackSrc0`'s precomputed term) or, for `iq4_xs`,
/// through the `sign` trick. Both `maddubs` and `sign` keep their x86
/// semantics, which is what makes this arm's integers this arm's.
fn microAvx2(
    comptime SUBBLK: usize,
    comptime HAS_MIN: bool,
    comptime BIAS: i32,
    comptime NK: c_int,
    comptime ACTBIAS: bool,
    src0: *const TileSrc0,
    src1: *const TileSrc1,
    ri0: c_int,
    j0_: c_int,
    num_k: c_int,
    slab: c_int,
    buf: [*]f32,
    buf_stride: c_int,
) void {
    comptime std.debug.assert(SUBBLK == 16 or SUBBLK == 32);
    const NB = tile_k / SUBBLK;
    const GROUP = 8; // Process 8 src1 columns concurrently (1 YMM vector)
    const gm = Geometry.of(NB, NK, num_k, slab);
    const j0: usize = @intCast(j0_);
    const bstride: usize = @intCast(buf_stride);

    const bias8: i8 = if (BIAS != 0) @bitCast(@as(u8, @truncate(@as(u32, @bitCast(BIAS))))) else 0;

    for (0..micro) |i| {
        const ar = @as(usize, @intCast(ri0)) + i;
        const d_off = gm.seff * micro + ar;
        const d0 = src0.d[d_off];
        const dmin0 = src0.dmin[d_off];
        const q0 = src0.q[ar * gm.qk_stride + gm.qk_off ..];
        const scales_row = src0.scales[ar * gm.nb_stride + gm.nb_off ..];
        const mins_row = src0.mins[ar * gm.nb_stride + gm.nb_off ..];

        var g: usize = 0;
        while (g < micro) : (g += GROUP) {
            // Pre-calculate weight pointers for this column group
            var q1_ptr: [GROUP][*]const i8 = undefined;
            for (0..GROUP) |t| q1_ptr[t] = src1.q[(j0 + g + t) * gm.qk_stride + gm.qk_off ..].ptr;

            var s2: i32x8 = @splat(0);
            var acc: [GROUP]i32x8 = @splat(@splat(0));

            if (SUBBLK == 32) {
                for (0..NB) |s| {
                    const q0_32 = load(u8x32, q0[s * SUBBLK ..].ptr);
                    const scales16: i16x16 = @splat(@truncate(scales_row[s]));

                    if (BIAS != 0) {
                        const w: i8x32 = @as(i8x32, @bitCast(q0_32)) -% @as(i8x32, @splat(bias8)); // debiased weight
                        if (ACTBIAS) {
                            // activation pre-biased (+128) by repack; q1b unsigned, w signed
                            for (0..GROUP) |t| {
                                const q1b_32 = load(u8x32, q1_ptr[t] + s * SUBBLK);
                                acc[t] +%= madd(scales16, maddubs(q1b_32, w));
                            }
                        } else {
                            // iq4_xs (Wmax > 64 overflows act-bias): sign trick, both operands <= 127
                            const ax: u8x32 = @bitCast(sign(w, w));
                            for (0..GROUP) |t| {
                                const q1_32 = load(i8x32, q1_ptr[t] + s * SUBBLK);
                                acc[t] +%= madd(scales16, maddubs(ax, sign(q1_32, w)));
                            }
                        }
                    } else {
                        for (0..GROUP) |t| {
                            const q1_32 = load(i8x32, q1_ptr[t] + s * SUBBLK);
                            acc[t] +%= madd(scales16, maddubs(q0_32, q1_32));
                        }
                    }

                    if (HAS_MIN) {
                        const bsums_v = load(i32x8, &src1.bsums[(gm.bs_off + s * 2) * gm.bs_stride + j0 + g]) +%
                            load(i32x8, &src1.bsums[(gm.bs_off + s * 2 + 1) * gm.bs_stride + j0 + g]);
                        s2 +%= bsums_v *% @as(i32x8, @splat(mins_row[s]));
                    }
                }
            } else { // SUBBLK == 16
                var sp: usize = 0;
                while (sp < NB) : (sp += 2) {
                    const q0_32 = load(u8x32, q0[sp * SUBBLK ..].ptr);
                    // `_mm256_set_m128i(set1(scales[sp + 1]), set1(scales[sp]))`
                    const lo: @Vector(8, i16) = @splat(@truncate(scales_row[sp]));
                    const hi: @Vector(8, i16) = @splat(@truncate(scales_row[sp + 1]));
                    const scalesv: i16x16 = std.simd.join(lo, hi);
                    if (BIAS != 0) {
                        const w: i8x32 = @as(i8x32, @bitCast(q0_32)) -% @as(i8x32, @splat(bias8)); // debiased weight
                        if (ACTBIAS) {
                            for (0..GROUP) |t| {
                                const q1b_32 = load(u8x32, q1_ptr[t] + sp * SUBBLK);
                                acc[t] +%= madd(scalesv, maddubs(q1b_32, w));
                            }
                        } else {
                            const ax: u8x32 = @bitCast(sign(w, w));
                            for (0..GROUP) |t| {
                                const q1_32 = load(i8x32, q1_ptr[t] + sp * SUBBLK);
                                acc[t] +%= madd(scalesv, maddubs(ax, sign(q1_32, w)));
                            }
                        }
                    } else {
                        for (0..GROUP) |t| {
                            const q1_32 = load(i8x32, q1_ptr[t] + sp * SUBBLK);
                            acc[t] +%= madd(scalesv, maddubs(q0_32, q1_32));
                        }
                    }

                    if (HAS_MIN) {
                        const bsums0_v = load(i32x8, &src1.bsums[(gm.bs_off + sp) * gm.bs_stride + j0 + g]);
                        const bsums1_v = load(i32x8, &src1.bsums[(gm.bs_off + sp + 1) * gm.bs_stride + j0 + g]);
                        s2 +%= (bsums0_v *% @as(i32x8, @splat(mins_row[sp]))) +%
                            (bsums1_v *% @as(i32x8, @splat(mins_row[sp + 1])));
                    }
                }
            }

            // Reduce the eight accumulators to column sums in acc[0]..acc[7] order, with additions modulo 2^32.
            var s1_vec = columnSums(&acc);

            if (BIAS != 0 and ACTBIAS) {
                // act-bias correction: raw = desired + 128*sum_s scales[s]*w_bsum[s]; corr precomputed in repack_src0
                s1_vec -%= @as(i32x8, @splat(mins_row[0]));
            }

            const src1_d_vec = load(f32x8, &src1.d[gm.seff * micro + j0 + g]);
            var res_vec = @as(f32x8, @splat(d0)) * @as(f32x8, @floatFromInt(s1_vec));

            if (HAS_MIN) {
                // `_mm256_fnmadd_ps`: res_vec - dmin0 * s2, one rounding
                res_vec = @mulAdd(f32x8, @splat(-dmin0), @as(f32x8, @floatFromInt(s2)), res_vec);
            }

            const buf_ptr = buf + ar * bstride + j0 + g;
            // `_mm256_fmadd_ps`: buf + res_vec * src1_d_vec, one rounding
            store(f32x8, buf_ptr, @mulAdd(f32x8, res_vec, src1_d_vec, load(f32x8, buf_ptr)));
        }
    }
}

/// Ports `tiled_run_microtile` (tiled-kernel.cpp:506 @8642541b2): one 16x16
/// microtile over one 256-K slab, accumulated into `buf` (`j`-major, row
/// width `buf_stride`). `num_k == 1` takes the `NK = 1` instantiation, the
/// narrow path the runtime-strided `NK = 0` one.
///
/// The C instantiates seven `<SUBBLK, HAS_MIN, BIAS, ACTBIAS>` tuples
/// explicitly (tiled-kernel.cpp:547 to :565); here the comptime parameters
/// of `tiled.zig`'s `formats` do the same.
pub fn runMicrotile(
    comptime a: Arm,
    comptime SUBBLK: usize,
    comptime HAS_MIN: bool,
    comptime BIAS: i32,
    comptime ACTBIAS: bool,
    src0: *const TileSrc0,
    src1: *const TileSrc1,
    ri0: c_int,
    j0: c_int,
    num_k: c_int,
    slab: c_int,
    buf: [*]f32,
    buf_stride: c_int,
) void {
    const standard = num_k == 1;
    switch (a) {
        .vnni => if (standard)
            microVnni(SUBBLK, HAS_MIN, BIAS, 1, src0, src1, ri0, j0, num_k, slab, buf, buf_stride)
        else
            microVnni(SUBBLK, HAS_MIN, BIAS, 0, src0, src1, ri0, j0, num_k, slab, buf, buf_stride),
        .avx2 => if (standard)
            microAvx2(SUBBLK, HAS_MIN, BIAS, 1, ACTBIAS, src0, src1, ri0, j0, num_k, slab, buf, buf_stride)
        else
            microAvx2(SUBBLK, HAS_MIN, BIAS, 0, ACTBIAS, src0, src1, ri0, j0, num_k, slab, buf, buf_stride),
        .scalar => if (standard)
            microScalar(SUBBLK, HAS_MIN, BIAS, 1, src0, src1, ri0, j0, num_k, slab, buf, buf_stride)
        else
            microScalar(SUBBLK, HAS_MIN, BIAS, 0, src0, src1, ri0, j0, num_k, slab, buf, buf_stride),
    }
}

// -----------------------------------------------------------------------------
// The repacks

/// Ports `tiled_repack_src1` (tiled-kernel.cpp:572 @8642541b2), the
/// `__AVX512VNNI__` arm, `tiled_repack_src1` (tiled-kernel.cpp:641
/// @8642541b2), the `__AVX2__` one, and `tiled_repack_src1`
/// (tiled-kernel.cpp:658 @8642541b2), the empty `#else`.
///
/// VNNI transposes each 16x16 block of `i32` words of a 16-row band in
/// place, giving the `[kg%16][kg/16][row][4]` layout `vpdpbusd` reads (the C
/// does it with four `permutex2var` butterfly phases and a column order; the
/// layout its microtile reads is the plain transpose). AVX2, when `bias`,
/// flips each code's sign bit: `q + 128` as an unsigned byte.
pub fn repackSrc1(comptime a: Arm, src1: *TileSrc1, row0: c_int, num_k: c_int, bias: bool) void {
    switch (a) {
        .vnni => {
            const n_tiles: usize = @as(usize, @intCast(num_k)) * (tile_k / (micro * 4));
            const row_stride: usize = @as(usize, @intCast(num_k)) * tile_k;
            const base: [*]u8 = @as([*]u8, @ptrCast(&src1.q)) + @as(usize, @intCast(row0)) * row_stride;
            for (0..n_tiles) |t| {
                const cbase = base + t * (micro * 4);
                var v: [16][16]u32 = undefined;
                for (0..16) |r| v[r] = @bitCast(load(@Vector(16, u32), cbase + r * row_stride));
                for (0..16) |col| {
                    var out: [16]u32 = undefined;
                    for (0..16) |r| out[r] = v[r][col];
                    store(@Vector(16, u32), cbase + col * row_stride, out);
                }
            }
        },
        .avx2 => {
            if (!bias) return;
            const row_stride: usize = @as(usize, @intCast(num_k)) * tile_k;
            const base: [*]u8 = @as([*]u8, @ptrCast(&src1.q)) + @as(usize, @intCast(row0)) * row_stride;
            const nbytes = row_stride; // full row width
            for (0..micro) |r| {
                const row = base + r * row_stride;
                var off: usize = 0;
                while (off < nbytes) : (off += 32) {
                    store(u8x32, row + off, load(u8x32, row + off) ^ @as(u8x32, @splat(0x80)));
                }
            }
        },
        .scalar => {},
    }
}

/// Ports `tiled_byte_sum_32` and `tiled_byte_sum_16`
/// (tiled-kernel.cpp:665, 674 @8642541b2): `n` unsigned bytes summed, which
/// `maddubs` by one then `madd` by one computes exactly.
inline fn byteSum(comptime n: usize, p: [*]const u8) i32 {
    var s: i32 = 0;
    for (0..n) |k| s += p[k];
    return s;
}

/// Ports `tiled_repack_src0` (tiled-kernel.cpp:687 @8642541b2): for the
/// act-bias formats, each row's correction `128 * sum_s scales[s] *
/// (sum of the debiased weights of subblock s)`, stored in `mins[r][0]`. Only
/// under `__AVX2__`, which the VNNI arm's targets also define; a no-op
/// otherwise and when `corr` is false.
pub fn repackSrc0(comptime a: Arm, comptime SUBBLK: usize, tile: *TileSrc0, n_rows: c_int, num_k: c_int, BIAS: i32, corr: bool) void {
    if (a == .scalar) return;
    if (!corr) return;
    const NB = tile_k / SUBBLK;
    const nk: usize = @intCast(num_k);
    const nb_stride = NB * nk;
    for (0..nk) |slab| {
        for (0..@intCast(n_rows)) |r| {
            const q = tile.q[r * (nk * tile_k) + slab * tile_k ..].ptr;
            const scales = tile.scales[r * nb_stride + slab * NB ..];
            var corr_val: i32 = 0;
            for (0..NB) |s| {
                const qsum = byteSum(SUBBLK, q + s * SUBBLK);
                corr_val +%= 128 *% scales[s] *% (qsum -% @as(i32, SUBBLK) *% BIAS);
            }
            tile.mins[r * nb_stride + slab * NB] = corr_val;
        }
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the portable intrinsics keep the x86 edge cases" {
    // maddubs saturates a pair to i16; sign leaves -128 at -128.
    var a: u8x32 = @splat(255);
    var b: i8x32 = @splat(127);
    try std.testing.expectEqual(@as(i16, 32767), maddubs(a, b)[0]);
    a = @splat(255);
    b = @splat(-128);
    try std.testing.expectEqual(@as(i16, -32768), maddubs(a, b)[3]);
    const m128: i8x32 = @splat(-128);
    const neg: i8x32 = @splat(-1);
    try std.testing.expectEqual(@as(i8, -128), sign(m128, neg)[5]);
    try std.testing.expectEqual(@as(i8, 0), sign(m128, @splat(0))[5]);
    const x: i16x16 = @splat(std.math.minInt(i16));
    try std.testing.expectEqual(std.math.minInt(i32), madd(x, x)[2]);
}

test "columnSums puts each accumulator's wrapping sum in its lane" {
    var rows: [8][8]i32 = undefined;
    var acc: [8]i32x8 = undefined;
    var state: u32 = 1;
    for (&rows, &acc) |*row, *a| {
        for (row) |*v| {
            state = state *% 1103515245 +% 12345;
            v.* = @bitCast(state);
        }
        a.* = row.*;
    }
    const got: [8]i32 = columnSums(&acc);
    for (rows, got) |row, sum| {
        var want: i32 = 0;
        for (row) |v| want +%= v;
        try std.testing.expectEqual(want, sum);
    }
}

test "unpack primitives match the C's scalar arm" {
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    var src: [32]u8 = undefined;
    rand.bytes(&src);

    var lo: [32]u8 = undefined;
    var hi: [32]u8 = undefined;
    unpkNib4(&src, &lo, &hi);
    for (0..32) |l| {
        try std.testing.expectEqual(src[l] & 0xF, lo[l]);
        try std.testing.expectEqual(src[l] >> 4, hi[l]);
    }

    var dst: [32]u8 = undefined;
    unpk2bit(6, &src, &dst);
    for (0..32) |l| try std.testing.expectEqual((src[l] >> 6) & 3, dst[l]);

    var acc: [32]u8 = lo;
    unpkOr(4, 4, 1, &acc, &src);
    for (0..32) |l| try std.testing.expectEqual(lo[l] | (((src[l] >> 4) & 1) << 4), acc[l]);
}

test "the VNNI transpose puts word (r, c) at (c, r)" {
    const a = std.testing.allocator;
    const t = try a.create(TileSrc1);
    defer a.destroy(t);
    const q: [*]u8 = @ptrCast(&t.q);
    for (0..16) |r| {
        for (0..tile_k) |k| q[r * tile_k + k] = @truncate(r * 7 + k);
    }
    var before: [16 * tile_k]u8 = undefined;
    @memcpy(&before, q[0 .. 16 * tile_k]);
    repackSrc1(.vnni, t, 0, 1, false);
    // kg (a 4-byte group of a row) lands at [kg%16][kg/16][row][4].
    for (0..16) |r| {
        for (0..tile_k / 4) |kg| {
            for (0..4) |b| {
                try std.testing.expectEqual(before[r * tile_k + kg * 4 + b], q[(kg % 16) * tile_k + (kg / 16) * 64 + r * 4 + b]);
            }
        }
    }
}
