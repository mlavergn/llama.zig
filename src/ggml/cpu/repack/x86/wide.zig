//! The register width the x86 repack gemms run at: `__m256` everywhere, and
//! `__m512` in the `__AVX512BW__ && __AVX512DQ__` arms of the gemms.
//!
//! # Provenance
//!
//! **Not a port of a file.** Each function reproduces one Intel intrinsic
//! `llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` (v0.4.1, `b29c606e2`)
//! calls, at both widths -- `_mm256_maddubs_epi16` and
//! `_mm512_maddubs_epi16` are `Wide(256).maddubs_epi16` and
//! `Wide(512).maddubs_epi16` -- plus `join`, which names how the AVX-512
//! arms build their registers.
//!
//! # Why the gemms are generic over width
//!
//! Each AVX-512 gemm arm is its AVX2 arm on two interleaved blocks at once:
//! the same intrinsics in the same order (328 `shuffle_epi32`, 256
//! `maddubs_epi16` and 192 `add_epi16` in each of `q2_K`'s arms, and so on),
//! every one of them confined to a 128-bit lane or to its own element. What
//! differs is how a register is filled, and the C fills them two ways:
//!
//! - the columns, their scales and their mins, as
//!   `_mm512_inserti32x8(_mm512_castsi256_si512(lo), hi, 1)` -- the 256-bit
//!   value for block `x` in the low half and for block `x + 1` in the high --
//!   or `GGML_F32Cx8x2_LOAD(d0, d1)`, the same for scales;
//! - the activation rows and their scales, the 256-bit value repeated.
//!
//! `join` is both. So one kernel body over `Wide(bits)` is both arms, and
//! each lane of the AVX-512 result is computed exactly as the AVX2 arm
//! computes it for that column.
//!
//! # The two sign tricks agree
//!
//! `mul_sum_i8_pairs_acc_int32x8` takes `_mm256_sign_epi8(x, x)` and
//! `_mm256_sign_epi8(y, x)`; the 512-bit helper takes `_mm512_abs_epi8(x)`
//! and a masked `0 - y` where `x < 0`. They give the same bytes except where
//! `x == 0`, which `sign` zeroes in `y` and the masked form does not -- but
//! the product with `|x| = 0` is zero either way. Each width uses its own.

const std = @import("std");
const builtin = @import("builtin");
const avx = @import("../../quants/x86/avx2.zig");
const common = @import("common.zig");

const m256i = avx.m256i;
const m128i = avx.m128i;
const m256 = avx.m256;

/// The C's `__AVX512BW__ && __AVX512DQ__`, which compiles the gemms'
/// 512-bit arms.
pub const has_avx512bw_dq = builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx512bw) and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx512dq);

/// `__AVX512VNNI__`, which turns the 512-bit `mul_sum_us8_pairs` into one
/// `vpdpbusd`; `tigerlake` has it, x86-64-v4 does not.
const has_avx512vnni = builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vnni);

const sysv: std.builtin.CallingConvention = .{ .x86_64_sysv = .{} };

extern fn @"llvm.x86.avx512.pmaddubs.w.512"(@Vector(64, u8), @Vector(64, i8)) callconv(sysv) @Vector(32, i16);
extern fn @"llvm.x86.avx512.pmaddw.d.512"(@Vector(32, i16), @Vector(32, i16)) callconv(sysv) @Vector(16, i32);
extern fn @"llvm.x86.avx512.pshuf.b.512"(@Vector(64, i8), @Vector(64, i8)) callconv(sysv) @Vector(64, i8);
extern fn @"llvm.x86.avx512.psrli.w.512"(@Vector(32, i16), i32) callconv(sysv) @Vector(32, i16);
extern fn @"llvm.x86.avx512.vpdpbusd.512"(@Vector(16, i32), @Vector(64, i8), @Vector(64, i8)) callconv(sysv) @Vector(16, i32);

/// The intrinsics at `bits` wide, 256 or 512.
pub fn Wide(comptime bits: u16) type {
    if (bits != 256 and bits != 512) @compileError("x86 repack runs at 256 or 512 bits");
    return struct {
        pub const width = bits;

        /// `__m256i` or `__m512i`: `long long` lanes, as clang defines them.
        pub const I = @Vector(bits / 64, i64);
        /// `__m256` or `__m512`.
        pub const F = @Vector(bits / 32, f32);
        /// Floats per register: eight columns, or sixteen.
        pub const cols = bits / 32;

        const n8 = bits / 8;
        const n16 = bits / 16;
        const n32 = bits / 32;

        inline fn as(comptime V: type, x: anytype) V {
            return @bitCast(x);
        }

        /// The register the C builds from 256-bit halves: `lo` alone at 256
        /// bits; at 512, `_mm512_inserti32x8(_mm512_castsi256_si512(lo), hi,
        /// 1)`.
        pub inline fn join(lo: m256i, hi: m256i) I {
            if (bits == 256) return lo;
            return @shuffle(i64, lo, hi, [8]i32{ 0, 1, 2, 3, -1, -2, -3, -4 });
        }

        /// `join` for floats: `_mm512_insertf32x8(_mm512_castps256_ps512(lo),
        /// hi, 1)`, or `GGML_F32Cx8x2_LOAD`'s widening of two loads.
        pub inline fn joinF(lo: m256, hi: m256) F {
            if (bits == 256) return lo;
            return @shuffle(f32, lo, hi, [16]i32{ 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8 });
        }

        /// `join(v, v)`: an activation register, the same in both halves.
        pub inline fn rep(v: m256i) I {
            return join(v, v);
        }

        /// `joinF(v, v)`.
        pub inline fn repF(v: m256) F {
            return joinF(v, v);
        }

        pub inline fn setzero() I {
            return @splat(0);
        }

        pub inline fn setzero_ps() F {
            return @splat(0);
        }

        pub inline fn set1_epi8(x: i8) I {
            return as(I, @as(@Vector(n8, i8), @splat(x)));
        }

        pub inline fn and_si(a: I, b: I) I {
            return a & b;
        }

        pub inline fn add_epi16(a: I, b: I) I {
            if (bits == 256) return avx.mm256_add_epi16(a, b);
            return as(I, as(@Vector(n16, i16), a) +% as(@Vector(n16, i16), b));
        }

        pub inline fn add_epi32(a: I, b: I) I {
            if (bits == 256) return avx.mm256_add_epi32(a, b);
            return as(I, as(@Vector(n32, i32), a) +% as(@Vector(n32, i32), b));
        }

        /// `srli_epi16` by a constant, as every repack call site shifts.
        pub inline fn srli_epi16(a: I, comptime count: u32) I {
            if (bits == 256) return avx.mm256_srli_epi16(a, count);
            return as(I, @"llvm.x86.avx512.psrli.w.512"(as(@Vector(32, i16), a), count));
        }

        /// `pmaddubsw`: unsigned bytes of `a` times signed bytes of `b`,
        /// pairs summed with signed saturation.
        pub inline fn maddubs_epi16(a: I, b: I) I {
            if (bits == 256) return avx.mm256_maddubs_epi16(a, b);
            return as(I, @"llvm.x86.avx512.pmaddubs.w.512"(as(@Vector(64, u8), a), as(@Vector(64, i8), b)));
        }

        pub inline fn madd_epi16(a: I, b: I) I {
            if (bits == 256) return avx.mm256_madd_epi16(a, b);
            return as(I, @"llvm.x86.avx512.pmaddw.d.512"(as(@Vector(32, i16), a), as(@Vector(32, i16), b)));
        }

        /// `pshufb`, within each 128-bit lane.
        pub inline fn shuffle_epi8(a: I, b: I) I {
            if (bits == 256) return avx.mm256_shuffle_epi8(a, b);
            return as(I, @"llvm.x86.avx512.pshuf.b.512"(as(@Vector(64, i8), a), as(@Vector(64, i8), b)));
        }

        /// `_mm256_shuffle_epi32` / `_mm512_shuffle_epi32`: the same
        /// four-lane permute in every 128-bit lane.
        pub inline fn shuffle_epi32(a: I, comptime imm: u8) I {
            if (bits == 256) return avx.mm256_shuffle_epi32(a, imm);
            const sel = comptime blk: {
                @setEvalBranchQuota(10000);
                var m: [n32]i32 = undefined;
                for (0..n32) |k| m[k] = @intCast((k / 4) * 4 + ((imm >> (2 * (k % 4))) & 3));
                break :blk m;
            };
            return as(I, @shuffle(i32, as(@Vector(n32, i32), a), undefined, sel));
        }

        /// `_mm256_blend_epi32(a, b, imm)` and the `_mm512_mask_blend_epi32`
        /// the AVX-512 arms write for it, whose 16-bit mask repeats `imm`
        /// (`204` and `0xCCCC`).
        pub inline fn blend_epi32(a: I, b: I, comptime imm: u8) I {
            if (bits == 256) return avx.mm256_blend_epi32(a, b, imm);
            const mask: @Vector(n32, bool) = comptime blk: {
                @setEvalBranchQuota(10000);
                var m: [n32]bool = undefined;
                for (0..n32) |k| m[k] = (imm >> (k % 8)) & 1 != 0;
                break :blk m;
            };
            return as(I, @select(i32, mask, as(@Vector(n32, i32), b), as(@Vector(n32, i32), a)));
        }

        /// `_mm256_shuffle_ps(a, a, imm)` / `_mm512_shuffle_ps(a, a, imm)`:
        /// every call site shuffles a register with itself.
        pub inline fn shuffle_ps(a: F, comptime imm: u8) F {
            if (bits == 256) return avx.mm256_shuffle_ps(a, a, imm);
            const sel = comptime blk: {
                @setEvalBranchQuota(10000);
                var m: [n32]i32 = undefined;
                for (0..n32) |k| m[k] = @intCast((k / 4) * 4 + ((imm >> (2 * (k % 4))) & 3));
                break :blk m;
            };
            return @shuffle(f32, a, undefined, sel);
        }

        pub inline fn cvtepi32_ps(a: I) F {
            return @floatFromInt(as(@Vector(n32, i32), a));
        }

        /// `fmadd_ps(a, b, c)`: `a * b + c`, fused.
        pub inline fn fmadd_ps(a: F, b: F, cc: F) F {
            return @mulAdd(F, a, b, cc);
        }

        pub inline fn mul_ps(a: F, b: F) F {
            return a * b;
        }

        pub inline fn sub_ps(a: F, b: F) F {
            return a - b;
        }

        pub inline fn storeu_ps(p: [*]f32, v: F) void {
            p[0..cols].* = v;
        }

        /// `mul_sum_i8_pairs_acc_int32x8` (arch/x86/repack.cpp:165
        /// @b29c606e2) and `mul_sum_i8_pairs_acc_int32x16`
        /// (arch/x86/repack.cpp:134 @b29c606e2), the latter with
        /// `mul_sum_us8_pairs_acc_int32x16` and `sum_i16_pairs_acc_int32x16`
        /// (arch/x86/repack.cpp:123, 118 @b29c606e2) inlined: `vpdpbusd` with
        /// `__AVX512VNNI__`, `maddubs` then `madd` without.
        pub inline fn mul_sum_i8_pairs_acc(acc: I, x: I, y: I) I {
            if (bits == 256) return common.mul_sum_i8_pairs_acc_int32x8(acc, x, y);
            const xb = as(@Vector(64, i8), x);
            const yb = as(@Vector(64, i8), y);
            // `_mm512_abs_epi8`, wrapping: |-128| is 0x80.
            const ax = @select(i8, xb < @as(@Vector(64, i8), @splat(0)), @as(@Vector(64, i8), @splat(0)) -% xb, xb);
            // `_mm512_mask_sub_epi8(y, movepi8_mask(x), 0, y)`.
            const sy = @select(i8, xb < @as(@Vector(64, i8), @splat(0)), @as(@Vector(64, i8), @splat(0)) -% yb, yb);
            if (has_avx512vnni) return as(I, @"llvm.x86.avx512.vpdpbusd.512"(as(@Vector(16, i32), acc), ax, sy));
            const dot = maddubs_epi16(as(I, ax), as(I, sy));
            return add_epi32(acc, madd_epi16(as(I, @as(@Vector(32, i16), @splat(1))), dot));
        }
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
