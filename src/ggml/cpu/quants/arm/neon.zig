//! The NEON intrinsics `ggml-cpu/arch/arm/quants.c` uses, as Zig vector code.
//!
//! # Provenance
//!
//! **Not a port of a file.** Each function here reproduces one ACLE intrinsic
//! that `llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` (v0.3.0, `c1d0e7a00`)
//! calls, and is named after it. The semantics come from the ARM C Language
//! Extensions; where Zig's obvious equivalent differs, the difference is
//! documented and the ARM behaviour is what is implemented.
//!
//! # Why this layer exists
//!
//! Zig has no ACLE intrinsics. It has `@Vector`, which lowers to NEON for
//! ordinary arithmetic, so most of the 82 intrinsics that file uses are one
//! line each. Writing those inline at 1,015 call sites would bury the
//! algorithm; gathering them here keeps each kernel readable against the C and
//! puts every semantic trap in one place.
//!
//! # The three traps
//!
//! - **NEON integer arithmetic wraps.** `addv`, `add`, `mul` and friends
//!   truncate on overflow; Zig's `+`, `*` panic in a safe build. Every integer
//!   operation here uses the wrapping form, which is both correct and
//!   panic-free.
//! - **`vaddvq_f32` is a pairwise reduction, not a sequential one.** It lowers
//!   to two `faddp`, giving `(a0+a1) + (a2+a3)`. `@reduce(.Add, ...)` on floats
//!   is *ordered* — `((a0+a1)+a2)+a3` — and the two differ in the last bit.
//!   Measured, not assumed: for `{1e8, 1, -1e8, 1}` the first gives 0 and the
//!   second gives 1.
//! - **`vmlaq_n_f32` is fused.** It reads as multiply-then-add, and clang
//!   compiles it to `fmla` — one rounding, not two. So it is `@mulAdd` here.
//!   Unlike the plain `a*b + c` in the quantizers, this is an explicit
//!   intrinsic call, so the fusion site is known rather than guessed.
//!
//! # `vdotq_s32` has no Zig equivalent, and does not need one
//!
//! The dot-product instruction is not expressible in portable Zig. But it is
//! *integer*: four `i8`x`i8` products summed into an `i32` lane, exact, with no
//! rounding to preserve. So the widening-multiply-and-shuffle form below gives
//! bit-identical results and only costs throughput. See `dotq_s32`.

const std = @import("std");

// -----------------------------------------------------------------------------
// Types
//
// Named as the ACLE names them, so a call site reads against the C.

pub const i8x8 = @Vector(8, i8);
pub const i8x16 = @Vector(16, i8);
pub const u8x8 = @Vector(8, u8);
pub const u8x16 = @Vector(16, u8);
pub const i16x4 = @Vector(4, i16);
pub const i16x8 = @Vector(8, i16);
pub const u16x4 = @Vector(4, u16);
pub const u16x8 = @Vector(8, u16);
pub const i32x2 = @Vector(2, i32);
pub const i32x4 = @Vector(4, i32);
pub const u32x2 = @Vector(2, u32);
pub const u32x4 = @Vector(4, u32);
pub const f32x4 = @Vector(4, f32);

/// The element type of a vector.
fn Elem(comptime V: type) type {
    return @typeInfo(V).vector.child;
}

/// The lane count of a vector.
fn lanes(comptime V: type) comptime_int {
    return @typeInfo(V).vector.len;
}

// -----------------------------------------------------------------------------
// Loads and stores
//
// `vld1q_*` and `vld1_*`. NEON's loads have no alignment requirement, so
// neither do these: the pointer is taken as `align(1)`, which is what the C's
// `vld1q_s8((const int8_t *) p)` amounts to.

/// Ports `vld1q_*` and `vld1_*`: a vector read from unaligned memory.
///
/// Parameters:
/// - `V`: the vector type to read.
/// - `p`: source bytes; need not be aligned.
///
/// Return: the vector.
pub inline fn load(comptime V: type, p: [*]const u8) V {
    return @as(*align(1) const V, @ptrCast(p)).*;
}

/// The same, from a typed pointer.
pub inline fn loadFrom(comptime V: type, p: [*]const Elem(V)) V {
    return @as(*align(1) const V, @ptrCast(p)).*;
}

/// Ports `vst1q_u8`: a vector written to unaligned memory.
pub inline fn store(comptime V: type, p: [*]u8, v: V) void {
    @as(*align(1) V, @ptrCast(p)).* = v;
}

/// Ports `vdupq_n_*` and `vdup_n_*`: every lane the same.
pub inline fn dup(comptime V: type, x: Elem(V)) V {
    return @splat(x);
}

/// Ports `vcreate_u8`: the eight bytes of a `u64`, little-endian.
pub inline fn create_u8(x: u64) u8x8 {
    return @bitCast(x);
}

/// Ports `vgetq_lane_*` and `vget_lane_*`.
pub inline fn lane(v: anytype, comptime i: usize) Elem(@TypeOf(v)) {
    return v[i];
}

/// Ports `vset_lane_u32`.
pub inline fn setLane_u32(x: u32, v: u32x2, comptime i: usize) u32x2 {
    var out = v;
    out[i] = x;
    return out;
}

// -----------------------------------------------------------------------------
// Halves and reinterpretation

/// Ports `vget_low_*`: the first half of a 128-bit vector.
pub inline fn low(v: anytype) @Vector(lanes(@TypeOf(v)) / 2, Elem(@TypeOf(v))) {
    const n = lanes(@TypeOf(v)) / 2;
    const idx = comptime blk: {
        var m: [n]i32 = undefined;
        for (0..n) |i| m[i] = @intCast(i);
        break :blk m;
    };
    return @shuffle(Elem(@TypeOf(v)), v, undefined, idx);
}

/// Ports `vget_high_*`: the second half of a 128-bit vector.
pub inline fn high(v: anytype) @Vector(lanes(@TypeOf(v)) / 2, Elem(@TypeOf(v))) {
    const n = lanes(@TypeOf(v)) / 2;
    const idx = comptime blk: {
        var m: [n]i32 = undefined;
        for (0..n) |i| m[i] = @intCast(i + n);
        break :blk m;
    };
    return @shuffle(Elem(@TypeOf(v)), v, undefined, idx);
}

/// Ports `vcombine_*`: two 64-bit halves into one 128-bit vector, `a` low.
pub inline fn combine(a: anytype, b: @TypeOf(a)) @Vector(lanes(@TypeOf(a)) * 2, Elem(@TypeOf(a))) {
    const n = lanes(@TypeOf(a));
    const idx = comptime blk: {
        var m: [n * 2]i32 = undefined;
        for (0..n) |i| m[i] = @intCast(i);
        // A negative index selects from the second operand, `~i` being `-i - 1`.
        for (0..n) |i| m[n + i] = ~@as(i32, @intCast(i));
        break :blk m;
    };
    return @shuffle(Elem(@TypeOf(a)), a, b, idx);
}

/// Ports the `vreinterpret*` family: a bit-for-bit retype, no lane movement.
pub inline fn cast(comptime V: type, v: anytype) V {
    return @bitCast(v);
}

// -----------------------------------------------------------------------------
// Integer arithmetic
//
// Wrapping throughout, because NEON wraps. A plain `+` here would panic in a
// Debug build on inputs the hardware handles silently.

/// Ports `vaddq_*` and `vadd_*`.
pub inline fn add(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return a +% b;
}

/// Ports `vsubq_*` and `vsub_*`.
pub inline fn sub(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return a -% b;
}

/// Ports `vmulq_*` and `vmul_*`.
pub inline fn mul(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return a *% b;
}

/// Ports `vandq_*` and `vand_*`.
pub inline fn @"and"(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return a & b;
}

/// Ports `vorrq_*`.
pub inline fn orr(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return a | b;
}

/// Ports `vbicq_*`: `a AND NOT b`. The operand order is the reverse of what
/// the name suggests -- it is *a* that survives.
pub inline fn bic(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return a & ~b;
}

/// Ports `vshrq_n_*` and `vshr_n_*`: shift right by a compile-time amount.
///
/// On unsigned lanes this is logical, on signed lanes arithmetic, which is what
/// Zig's `>>` already does for each.
pub inline fn shrN(v: anytype, comptime n: comptime_int) @TypeOf(v) {
    const V = @TypeOf(v);
    // The shift amount vector is lane-width-log2 wide, not lane wide -- a
    // `@bitCast` of a same-width vector is a size mismatch.
    const amount: @Vector(lanes(V), std.math.Log2Int(Elem(V))) = @splat(n);
    return v >> amount;
}

/// Ports `vshlq_n_*`: shift left by a compile-time amount, discarding the bits
/// that leave the lane.
pub inline fn shlN(v: anytype, comptime n: comptime_int) @TypeOf(v) {
    const V = @TypeOf(v);
    const amount: @Vector(lanes(V), std.math.Log2Int(Elem(V))) = @splat(n);
    return v << amount;
}

/// Ports `vshlq_u8` and `vshlq_u16` with a **vector** shift amount.
///
/// ACLE's variable shift takes *signed* amounts and **shifts right when they
/// are negative**. `ggml_vec_dot_q2_0_q8_0` relies on that: it passes
/// `{0,-2,-4,-6,...}` to extract four 2-bit fields from a replicated byte in
/// one instruction. Reading the name as "shift left" and asserting the amount
/// non-negative -- which an earlier version of this function did -- turns that
/// kernel into nonsense.
///
/// A shift of at least the lane width gives zero rather than being undefined,
/// which is also ACLE's answer and not Zig's.
pub inline fn shlq(v: anytype, amount: anytype) @TypeOf(v) {
    const V = @TypeOf(v);
    const E = Elem(V);
    const bits = @bitSizeOf(E);
    const n = lanes(V);

    const va: [n]E = v;
    const aa: [n]Elem(@TypeOf(amount)) = amount;
    var out: [n]E = undefined;

    inline for (0..n) |i| {
        const k: i32 = aa[i];
        if (k >= bits or k <= -bits) {
            out[i] = 0;
        } else if (k >= 0) {
            out[i] = va[i] << @intCast(k);
        } else {
            out[i] = va[i] >> @intCast(-k);
        }
    }
    return out;
}

/// Ports `vceqq_*`: an all-ones lane where equal, all-zero where not.
///
/// Zig's `==` on vectors yields a `bool` vector, not the mask NEON produces, so
/// the widening back to a lane-sized mask is explicit.
pub inline fn ceq(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    const V = @TypeOf(a);
    const ones: V = @splat(~@as(Elem(V), 0));
    const zeros: V = @splat(0);
    return @select(Elem(V), a == b, ones, zeros);
}

/// Ports `vhaddq_u8`: `(a + b) >> 1`, computed without overflowing the lane.
pub inline fn hadd_u8(a: u8x16, b: u8x16) u8x16 {
    const wa: u16x16 = a;
    const wb: u16x16 = b;
    const one: u16x16 = @splat(1);
    return @truncate((wa + wb) >> one);
}

const u16x16 = @Vector(16, u16);

/// Ports `vmovl_u8`: zero-extend eight `u8` to eight `u16`.
pub inline fn movl_u8(v: u8x8) u16x8 {
    return v;
}

/// Ports `vmovl_s8`: sign-extend eight `i8` to eight `i16`.
pub inline fn movl_s8(v: i8x8) i16x8 {
    return v;
}

/// Ports `vmovl_u16`: zero-extend four `u16` to four `u32`.
pub inline fn movl_u16(v: u16x4) u32x4 {
    return v;
}

/// Ports `vmull_s16`: widening multiply, four `i16` pairs to four `i32`.
///
/// The widening is the point -- an `i16 * i16` product does not fit an `i16`,
/// and doing this as `mul` on `i16x4` would silently truncate.
pub inline fn mull_s16(a: i16x4, b: i16x4) i32x4 {
    const wa: i32x4 = a;
    const wb: i32x4 = b;
    return wa *% wb;
}

/// Ports `vmlaq_s32`: `a + b * c`, elementwise, on integers.
pub inline fn mla_s32(a: i32x4, b: i32x4, cc: i32x4) i32x4 {
    return a +% b *% cc;
}

/// Ports `vpaddq_s32`: pairwise add across two vectors.
///
/// Result is `{a0+a1, a2+a3, b0+b1, b2+b3}` -- the halves do **not**
/// interleave.
pub inline fn paddq_s32(a: i32x4, b: i32x4) i32x4 {
    const evens = @shuffle(i32, a, b, [4]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2) });
    const odds = @shuffle(i32, a, b, [4]i32{ 1, 3, ~@as(i32, 1), ~@as(i32, 3) });
    return evens +% odds;
}

/// Ports `vpaddq_s16`: the same, on eight-lane vectors.
pub inline fn paddq_s16(a: i16x8, b: i16x8) i16x8 {
    const evens = @shuffle(i16, a, b, [8]i32{ 0, 2, 4, 6, ~@as(i32, 0), ~@as(i32, 2), ~@as(i32, 4), ~@as(i32, 6) });
    const odds = @shuffle(i16, a, b, [8]i32{ 1, 3, 5, 7, ~@as(i32, 1), ~@as(i32, 3), ~@as(i32, 5), ~@as(i32, 7) });
    return evens +% odds;
}

/// Ports `vpaddlq_s16`: pairwise **widening** add, eight `i16` to four `i32`.
pub inline fn paddlq_s16(v: i16x8) i32x4 {
    const w: @Vector(8, i32) = v;
    const evens = @shuffle(i32, w, undefined, [4]i32{ 0, 2, 4, 6 });
    const odds = @shuffle(i32, w, undefined, [4]i32{ 1, 3, 5, 7 });
    return evens +% odds;
}

/// Ports `vzip1_u8` and `vzip2_u8`: interleave the low or high halves.
pub inline fn zip1_u8(a: u8x8, b: u8x8) u8x8 {
    return @shuffle(u8, a, b, [8]i32{ 0, ~@as(i32, 0), 1, ~@as(i32, 1), 2, ~@as(i32, 2), 3, ~@as(i32, 3) });
}

/// Ports `vzip2_u8`.
pub inline fn zip2_u8(a: u8x8, b: u8x8) u8x8 {
    return @shuffle(u8, a, b, [8]i32{ 4, ~@as(i32, 4), 5, ~@as(i32, 5), 6, ~@as(i32, 6), 7, ~@as(i32, 7) });
}

/// Ports `vqtbl1q_u8`, which `ggml-cpu-impl.h` wraps as `ggml_vqtbl1q_u8`: a
/// byte permute of `t` by `idx`.
///
/// An index of 16 or more yields **zero**, not an out-of-range read -- that is
/// the `tbl` instruction's defining behaviour and the reason it is `tbl` and
/// not a gather.
///
/// Where `idx` is a constant, as it is at every call site in `quants.c`,
/// `@shuffle` expresses the same thing and lowers to the same instruction.
/// This form takes a runtime vector so a call site can read like the C.
pub inline fn qtbl1q_u8(t: u8x16, idx: u8x16) u8x16 {
    const ta: [16]u8 = t;
    const ia: [16]u8 = idx;
    var out: [16]u8 = undefined;
    inline for (0..16) |i| {
        out[i] = if (ia[i] < 16) ta[ia[i]] else 0;
    }
    return out;
}

/// Ports `vqtbl1q_s8`: the same permute over signed lanes.
pub inline fn qtbl1q_s8(t: i8x16, idx: u8x16) i8x16 {
    return @bitCast(qtbl1q_u8(@bitCast(t), idx));
}

// -----------------------------------------------------------------------------
// Reductions

/// Ports `vaddvq_s32`: the sum of all four lanes.
///
/// Order is irrelevant here -- integer addition is associative, and both NEON
/// and `@reduce` wrap the same way on overflow.
pub inline fn addvq_s32(v: i32x4) i32 {
    return @reduce(.Add, v);
}

/// Ports `vaddvq_f32`: the sum of all four lanes, **pairwise**.
///
/// `@reduce(.Add, ...)` would be wrong. On floats it is an ordered reduction,
/// `((a0+a1)+a2)+a3`, where `vaddvq_f32` lowers to two `faddp` and gives
/// `(a0+a1) + (a2+a3)`. Float addition is not associative, so the two differ in
/// the last bit -- measured on `{1e8, 1, -1e8, 1}`, where they give 0 and 1.
pub inline fn addvq_f32(v: f32x4) f32 {
    return (v[0] + v[1]) + (v[2] + v[3]);
}

/// Ports `vmaxvq_f32`: the largest lane.
///
/// `@reduce(.Max, ...)` matches for every input this file sees, which is always
/// the output of `vabsq_f32` over model weights. The two can differ on NaN,
/// where ARM's `fmaxv` and LLVM's reduction disagree about propagation; a NaN
/// weight is outside what either the C or this port defines.
pub inline fn maxvq_f32(v: f32x4) f32 {
    return @reduce(.Max, v);
}

// -----------------------------------------------------------------------------
// Float arithmetic

/// Ports `vabsq_f32`.
pub inline fn abs_f32(v: f32x4) f32x4 {
    return @abs(v);
}

/// Ports `vmaxq_f32`: elementwise maximum.
pub inline fn max_f32(a: f32x4, b: f32x4) f32x4 {
    return @max(a, b);
}

/// Ports `vmulq_f32`.
pub inline fn mul_f32(a: f32x4, b: f32x4) f32x4 {
    return a * b;
}

/// Ports `vmulq_n_f32`: every lane times one scalar.
pub inline fn mul_n_f32(v: f32x4, x: f32) f32x4 {
    const s: f32x4 = @splat(x);
    return v * s;
}

// -----------------------------------------------------------------------------
// Half-precision vectors
//
// `cpu/vec.zig` kept these local when it was the only user; `ops/sgemm.zig`
// needs the same three, so they live here now rather than in a second copy.

/// `float16x8_t`.
pub const f16x8 = @Vector(8, f16);

/// `float16x4_t`.
pub const f16x4 = @Vector(4, f16);

/// Ports `vfmaq_f16(c, b, a)`: `a*b + c` in half precision, **fused** --
/// one rounding, not two.
pub inline fn fma_f16(acc: f16x8, a: f16x8, b: f16x8) f16x8 {
    return @mulAdd(f16x8, a, b, acc);
}

/// Ports `vaddq_f16`.
pub inline fn add_f16(a: f16x8, b: f16x8) f16x8 {
    return a + b;
}

/// Ports `vcvt_f32_f16(vget_low_f16(v))` and its `vget_high_f16` twin:
/// widen one half of an `f16x8` to `f32x4`.
pub inline fn cvt_f32_f16_half(v: f16x8, comptime upper: bool) f32x4 {
    const base: usize = if (upper) 4 else 0;
    const half: f16x4 = .{ v[base], v[base + 1], v[base + 2], v[base + 3] };
    return @floatCast(half);
}

/// Ports `vmlaq_n_f32`: `a + v * x`, **fused**.
///
/// The intrinsic reads as a multiply and an add, and clang emits `fmla` for it
/// -- one rounding rather than two. Written as `a + v * s` this would round
/// twice and differ in the last bit, so it is `@mulAdd`.
pub inline fn mla_n_f32(a: f32x4, v: f32x4, x: f32) f32x4 {
    const s: f32x4 = @splat(x);
    return @mulAdd(f32x4, v, s, a);
}

/// Ports `vfmaq_f32`: `a + b * cc`, fused by definition.
pub inline fn fma_f32(a: f32x4, b: f32x4, cc: f32x4) f32x4 {
    return @mulAdd(f32x4, b, cc, a);
}

/// Ports `vcvtq_f32_s32`: four `i32` widened to four `f32`.
pub inline fn cvt_f32_s32(v: i32x4) f32x4 {
    return @floatFromInt(v);
}

/// Ports `vcvtnq_s32_f32`: convert to `i32`, rounding to nearest with **ties
/// to even**.
///
/// That is `fcvtns`, and it is not `@round`, which is ties-away-from-zero. The
/// add-and-mask trick below is the same one `nearest_int` uses in
/// `ggml-quants.c`, and the two must agree: ggml's scalar and NEON quantizers
/// produce the same model bytes.
///
/// The trick is exact only within +/-2^22, which every caller here satisfies --
/// the values are weights scaled into +/-127.
pub inline fn cvtnq_s32_f32(v: f32x4) i32x4 {
    const magic: f32x4 = @splat(12582912.0); // 1.5 * 2^23
    const shifted = v + magic;
    const bits: i32x4 = @bitCast(shifted);
    const mask: i32x4 = @splat(0x007fffff);
    const bias: i32x4 = @splat(0x00400000);
    return (bits & mask) -% bias;
}

// -----------------------------------------------------------------------------
// The dot product

/// Ports `vdotq_s32`: four groups of four `i8` products, accumulated.
///
/// `result[i] = acc[i] + sum(a[4i+k] * b[4i+k], k = 0..3)`.
///
/// There is no portable Zig for the `sdot` instruction, and none is needed:
/// this is integer arithmetic, so the widening-multiply form below is
/// **bit-identical**, not merely close. Only throughput is lost, and the
/// quantized CPU path is not what carries a model when Metal is present.
///
/// Parameters:
/// - `acc`: the accumulator, added to lane-wise.
/// - `a`, `b`: sixteen `i8` lanes each.
///
/// Return: the four accumulated sums.
pub inline fn dotq_s32(acc: i32x4, a: i8x16, b: i8x16) i32x4 {
    const wa: @Vector(16, i32) = a;
    const wb: @Vector(16, i32) = b;
    const p = wa *% wb;

    const g0 = @shuffle(i32, p, undefined, [4]i32{ 0, 4, 8, 12 });
    const g1 = @shuffle(i32, p, undefined, [4]i32{ 1, 5, 9, 13 });
    const g2 = @shuffle(i32, p, undefined, [4]i32{ 2, 6, 10, 14 });
    const g3 = @shuffle(i32, p, undefined, [4]i32{ 3, 7, 11, 15 });

    return acc +% g0 +% g1 +% g2 +% g3;
}

/// Ports `vdotq_laneq_s32`: `dotq_s32` with one 32-bit lane of `b`
/// broadcast across all four.
///
/// The ACLE spells this as a lane index; clang expands it to
/// `vdotq_s32(acc, a, splatq_laneq(b, lane))`, which is what this is. The
/// splat is on the **32-bit** reinterpretation, so it replicates four
/// consecutive `i8` values, not one.
///
/// `idx` is `comptime` because the instruction encodes it. Named `idx`
/// rather than `lane` because this file already has a `lane` accessor.
pub inline fn dotq_laneq_s32(acc: i32x4, a: i8x16, b: i8x16, comptime idx: u2) i32x4 {
    const words: i32x4 = @bitCast(b);
    const splat: i32x4 = @splat(words[idx]);
    return dotq_s32(acc, a, @bitCast(splat));
}

/// Ports `vpadd_s32`: the **two-lane** pairwise add,
/// `[a0+a1, b0+b1]`. Not `a + b`.
pub inline fn padd_s32(a: i32x2, b: i32x2) i32x2 {
    return .{ a[0] +% a[1], b[0] +% b[1] };
}

/// Ports `vmla_s32`: the two-lane integer `acc + a * b`, wrapping.
pub inline fn mla_s32x2(acc: i32x2, a: i32x2, b: i32x2) i32x2 {
    return acc +% (a *% b);
}

/// Ports `vcvt_f32_s32` for two lanes.
pub inline fn cvt_f32_s32x2(v: i32x2) @Vector(2, f32) {
    return @floatFromInt(v);
}

/// Ports `vmlal_lane_s16`: `acc + widen(a) * widen(b[idx])`, the lane
/// variant of `mlal_s16`. `idx` is `comptime` because the instruction
/// encodes it.
pub inline fn mlal_lane_s16(acc: i32x4, a: i16x4, b: i16x4, comptime idx: u2) i32x4 {
    const splat: i16x4 = @splat(b[idx]);
    return acc +% mull_s16(a, splat);
}

/// Ports `vmlaq_s32`: integer `acc + a * b`, wrapping.
pub inline fn mlaq_s32(acc: i32x4, a: i32x4, b: i32x4) i32x4 {
    return acc +% (a *% b);
}

/// Ports `vsliq_n_u8`: shift-left-and-insert. The low `n` bits of `a` are
/// kept and `b` is shifted up by `n` into the rest.
///
/// `q5_K` uses it to fold a single high bit onto a masked nibble, where
/// `a` is already `& 0x0f` and `b` is 0 or 1 — so in that use it is
/// `a | (b << 4)`. The general form is written here because that is what
/// the instruction does, and a caller that has not pre-masked `a` would
/// otherwise get a silently different answer.
pub inline fn sli_n_u8(a: u8x16, b: u8x16, comptime n: u3) u8x16 {
    const keep: u8x16 = @splat((@as(u8, 1) << n) - 1);
    return (a & keep) | (b << @as(@Vector(16, u3), @splat(n)));
}

/// Ports `vmovl_s16`: widen `i16x4` to `i32x4`, sign-extending.
pub inline fn movl_s16(v: i16x4) i32x4 {
    return v;
}

/// Ports `vmlsq_f32`: `acc - a * b`, **fused** — one rounding, like its
/// `vmlaq_f32` counterpart. The subtraction is folded into the FMA by
/// negating the multiplier, not applied afterwards.
pub inline fn mlsq_f32(acc: f32x4, a: f32x4, b: f32x4) f32x4 {
    return @mulAdd(f32x4, -a, b, acc);
}

/// Ports `vmlal_s16`: `acc + widen(a) * widen(b)`, a widening
/// multiply-accumulate from `i16x4` pairs into `i32x4`.
pub inline fn mlal_s16(acc: i32x4, a: i16x4, b: i16x4) i32x4 {
    return acc +% mull_s16(a, b);
}

/// Ports `vcvtq_n_f32_s32`: convert to `f32` **and** divide by `2^n`, in
/// one instruction.
///
/// The repack kernels use it to undo a `<< 4` they applied to the weights
/// before the dot product, where the scalar `_generic` form instead shifts
/// the integer sum right by four. The two are not the same rounding: this
/// scales an exact integer, the shift truncates.
pub inline fn cvtq_n_f32_s32(v: i32x4, comptime n: comptime_int) f32x4 {
    const scale: f32x4 = @splat(1.0 / @as(f32, 1 << n));
    return cvt_f32_s32(v) * scale;
}

/// Ports `vld1q_dup_s64` reinterpreted as `i8x16`: one 8-byte group
/// broadcast into both halves of a vector.
pub inline fn dupq_i8x16_from8(p: [*]const i8) i8x16 {
    const half: @Vector(8, i8) = p[0..8].*;
    return combine(half, half);
}

/// Ports `vcvt_f32_f16`: widen four `f16` to `f32x4`.
pub inline fn cvt_f32_f16(v: f16x4) f32x4 {
    return @floatCast(v);
}

/// Ports `vld1_f16`: four `f16` from memory, as the `u16` the import gives.
pub inline fn load_f16x4(p: [*]const u16) f16x4 {
    return @bitCast(@as(@Vector(4, u16), p[0..4].*));
}

/// Ports `vld1_dup_f16`: one `f16` broadcast to all four lanes.
pub inline fn dup_f16x4(v: u16) f16x4 {
    const bits: @Vector(4, u16) = @splat(v);
    return @bitCast(bits);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "halves and combine round trip" {
    const v: i8x16 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    const lo = low(v);
    const hi = high(v);

    try std.testing.expectEqual(@as(i8, 0), lo[0]);
    try std.testing.expectEqual(@as(i8, 7), lo[7]);
    try std.testing.expectEqual(@as(i8, 8), hi[0]);
    try std.testing.expectEqual(@as(i8, 15), hi[7]);

    const back = combine(lo, hi);
    try std.testing.expectEqual(v, back);
}

test "vdotq_s32 matches the definition on every lane" {
    var aa: [16]i8 = undefined;
    var ba: [16]i8 = undefined;
    for (0..16) |i| {
        aa[i] = @intCast(@as(i32, @intCast(i)) - 8);
        ba[i] = @intCast(15 - @as(i32, @intCast(i)));
    }
    const a: i8x16 = aa;
    const b: i8x16 = ba;
    const acc: i32x4 = .{ 100, 200, 300, 400 };
    const got: [4]i32 = dotq_s32(acc, a, b);

    // The definition, spelled out. Arrays, not vectors: a vector index has to
    // be comptime-known.
    const acca: [4]i32 = acc;
    for (0..4) |g| {
        var want: i32 = acca[g];
        for (0..4) |k| want += @as(i32, aa[4 * g + k]) * ba[4 * g + k];
        try std.testing.expectEqual(want, got[g]);
    }
}

test "vdotq_s32 wraps rather than trapping, as sdot does" {
    const a: i8x16 = @splat(127);
    const b: i8x16 = @splat(127);
    const acc: i32x4 = @splat(std.math.maxInt(i32));
    // Would panic with a checked add; the hardware truncates.
    _ = dotq_s32(acc, a, b);
}

test "the float reduction is pairwise, not sequential" {
    // The case that separates them. If `addvq_f32` were `@reduce(.Add, ...)`
    // this would be 1, and every kernel using it would be off by a bit.
    const v: f32x4 = .{ 1e8, 1.0, -1e8, 1.0 };
    try std.testing.expectEqual(@as(f32, 0.0), addvq_f32(v));
    try std.testing.expectEqual(@as(f32, 1.0), @reduce(.Add, v));
}

test "vmlaq_n_f32 is fused" {
    // A case where one rounding differs from two: b*c is exactly representable
    // only before the add.
    const a: f32x4 = @splat(1.0);
    const b: f32x4 = @splat(std.math.floatEps(f32) / 2.0);
    const fused = mla_n_f32(a, b, 1.0);
    const unfused = a + b * @as(f32x4, @splat(1.0));

    // Fused keeps the tiny addend's influence through a single rounding.
    try std.testing.expectEqual(@as(f32, 1.0), unfused[0]);
    try std.testing.expectEqual(@as(f32, 1.0), fused[0]);

    // The shape that actually separates them.
    const big: f32x4 = @splat(1.0);
    const x: f32x4 = @splat(1.0 + std.math.floatEps(f32));
    try std.testing.expectEqual(
        @mulAdd(f32, x[0], x[0], -big[0]),
        mla_n_f32(-big, x, x[0])[0],
    );
}

test "cvtnq_s32_f32 rounds halves to even, not away from zero" {
    // The distinction `@round` gets wrong. 0.5 -> 0, 1.5 -> 2, 2.5 -> 2.
    const v: f32x4 = .{ 0.5, 1.5, 2.5, 3.5 };
    const got = cvtnq_s32_f32(v);

    try std.testing.expectEqual(@as(i32, 0), got[0]);
    try std.testing.expectEqual(@as(i32, 2), got[1]);
    try std.testing.expectEqual(@as(i32, 2), got[2]);
    try std.testing.expectEqual(@as(i32, 4), got[3]);

    // Negative halves too, and ordinary values.
    const w: f32x4 = .{ -0.5, -1.5, -2.5, 7.4 };
    const gw = cvtnq_s32_f32(w);
    try std.testing.expectEqual(@as(i32, 0), gw[0]);
    try std.testing.expectEqual(@as(i32, -2), gw[1]);
    try std.testing.expectEqual(@as(i32, -2), gw[2]);
    try std.testing.expectEqual(@as(i32, 7), gw[3]);
}

test "cvtnq_s32_f32 agrees with ggml's scalar nearest_int" {
    // They must: ggml's scalar and NEON quantizers produce the same bytes.
    const helpers = @import("../../../quants/helpers.zig");

    var seed: u32 = 1;
    for (0..4096) |_| {
        seed = 1103515245 *% seed +% 12345;
        const x = (@as(f32, @floatFromInt(seed >> 8)) / 65536.0) - 128.0;
        const v: f32x4 = @splat(x);
        try std.testing.expectEqual(helpers.nearestInt(x), cvtnq_s32_f32(v)[0]);
    }
}

test "pairwise adds keep the halves separate" {
    const a: i32x4 = .{ 1, 2, 3, 4 };
    const b: i32x4 = .{ 10, 20, 30, 40 };
    const got = paddq_s32(a, b);
    try std.testing.expectEqual(i32x4{ 3, 7, 30, 70 }, got);

    const w: i16x8 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try std.testing.expectEqual(i32x4{ 3, 7, 11, 15 }, paddlq_s16(w));
}

test "vceqq_u8 yields a lane mask, not a bool" {
    const a: u8x16 = @splat(3);
    var b: u8x16 = @splat(3);
    b[5] = 4;

    const m = ceq(a, b);
    try std.testing.expectEqual(@as(u8, 0xFF), m[0]);
    try std.testing.expectEqual(@as(u8, 0x00), m[5]);
}

test "vbicq keeps the first operand" {
    const a: u8x16 = @splat(0xF0);
    const b: u8x16 = @splat(0x30);
    try std.testing.expectEqual(@as(u8, 0xC0), bic(a, b)[0]);
}

test "halving add does not overflow the lane" {
    const a: u8x16 = @splat(255);
    const b: u8x16 = @splat(255);
    try std.testing.expectEqual(@as(u8, 255), hadd_u8(a, b)[0]);

    const c1: u8x16 = @splat(200);
    const c2: u8x16 = @splat(100);
    try std.testing.expectEqual(@as(u8, 150), hadd_u8(c1, c2)[0]);
}

test "the variable shift goes right on a negative amount" {
    // What `ggml_vec_dot_q2_0_q8_0` depends on: four 2-bit fields pulled out of
    // one replicated byte by `{0,-2,-4,-6}`.
    const v: u8x16 = @splat(0b11100100);
    const amount: i8x16 = .{ 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6 };
    const got: [16]u8 = shlq(v, amount);

    try std.testing.expectEqual(@as(u8, 0b11100100), got[0]);
    try std.testing.expectEqual(@as(u8, 0b00111001), got[1]);
    try std.testing.expectEqual(@as(u8, 0b00001110), got[2]);
    try std.testing.expectEqual(@as(u8, 0b00000011), got[3]);

    // Masked to two bits, that is the sequence 0,1,2,3.
    const mask: u8x16 = @splat(3);
    const fields: [16]u8 = shlq(v, amount) & mask;
    try std.testing.expectEqual(@as(u8, 0), fields[0]);
    try std.testing.expectEqual(@as(u8, 1), fields[1]);
    try std.testing.expectEqual(@as(u8, 2), fields[2]);
    try std.testing.expectEqual(@as(u8, 3), fields[3]);
}

test "a shift of at least the lane width gives zero" {
    const v: u8x16 = @splat(0xFF);
    const wide: i8x16 = @splat(8);
    try std.testing.expectEqual(@as(u8, 0), shlq(v, wide)[0]);

    const wide_neg: i8x16 = @splat(-8);
    try std.testing.expectEqual(@as(u8, 0), shlq(v, wide_neg)[0]);
}

test "the table permute zeroes out-of-range indices" {
    var ta: [16]u8 = undefined;
    for (&ta, 0..) |*x, i| x.* = @intCast(i * 3);
    const t: u8x16 = ta;

    var ia: [16]u8 = @splat(0);
    ia[0] = 5;
    ia[1] = 16; // out of range
    ia[2] = 255;
    const idx: u8x16 = ia;

    const got: [16]u8 = qtbl1q_u8(t, idx);
    try std.testing.expectEqual(@as(u8, 15), got[0]);
    try std.testing.expectEqual(@as(u8, 0), got[1]);
    try std.testing.expectEqual(@as(u8, 0), got[2]);
}

test "integer ops wrap as NEON does" {
    const m: i32x4 = @splat(std.math.maxInt(i32));
    const one: i32x4 = @splat(1);
    try std.testing.expectEqual(std.math.minInt(i32), add(m, one)[0]);

    const big: i8x16 = @splat(127);
    try std.testing.expectEqual(@as(i8, 1), mul(big, big)[0]);
}
