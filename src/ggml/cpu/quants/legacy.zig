//! Reference dot products for the block-per-32 formats and the two float4
//! codebooks.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/quants.c` (v0.9.0, `cd959bae9`),
//! the `_generic` dot products at lines 127-480 and 1254. Each function names
//! the C it replaces and the line it began at.
//!
//! # Every function here but one is unreachable on this target
//!
//! The exception is `ggml_vec_dot_f8_e4m3_f32`, new at v0.9.0, which has no
//! `_generic` suffix and no architecture replacement.
//!
//! `arch-fallback.h` renames nothing from `quants.c` on ARM, and
//! `arch/arm/quants.c` supplies the real `ggml_vec_dot_q4_0_q8_0` and its
//! siblings. These `_generic` names are exported and never called.
//!
//! They are still part of the translation unit's symbol contract, and they are
//! what a `GGML_CPU_GENERIC` build runs. `golden.zig` is the only thing that
//! checks them -- see `testing.zig`.
//!
//! # Shape of the C
//!
//! Each is: for every block, unpack the quantized weights, accumulate an
//! integer dot product against the `q8` right-hand operand, then scale by the
//! product of the two blocks' deltas. The integer accumulation is exact; the
//! only floating-point is the per-block scale and the running sum, so the
//! accumulation *order* is what has to match, not just the arithmetic.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("../../impl.zig");
const convert = @import("../convert.zig");
const simd = @import("../simd.zig");
const blocks = @import("../../quants/blocks.zig");
const c = impl.c;

/// The unsuffixed `d` of a block, widened. `GGML_CPU_FP16_TO_FP32` on ARM is
/// the hardware conversion -- see `convert.cpuFp16ToFp32`.
inline fn f(h: u16) f32 {
    return convert.cpuFp16ToFp32(h);
}

/// The block array a `vx` or `vy` pointer really is.
inline fn as(comptime Block: type, p: ?*const anyopaque) [*]const Block {
    return @ptrCast(@alignCast(p.?));
}

/// Ports `ggml_vec_dot_q1_0_q8_0_generic` (ggml-cpu/quants.c:132 @cd959bae9).
///
/// One-bit weights: each set bit selects `+y`, each clear bit `-y`. A q1_0
/// block covers 128 elements, so it spans four q8_0 blocks and each of those
/// carries its own delta.
///
/// Parameters:
/// - `n`: elements in the row; must be a multiple of `QK1_0`.
/// - `s`: destination for the scalar result.
/// - `vx`: the q1_0 row.
/// - `vy`: the q8_0 row.
/// - `nrc`: rows per call; must be 1 for the generic kernels.
///
/// Return: nothing; writes `s`.
pub export fn ggml_vec_dot_q1_0_q8_0_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK1_0;
    const nb = @divTrunc(n, qk);

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q1_0, vx);
    const y = as(blocks.Q8_0, vy);

    var sumf: f32 = 0.0;

    for (0..@intCast(nb)) |i| {
        const d0 = f(x[i].d);

        var sumi: f32 = 0.0;

        for (0..4) |k| {
            const yb = &y[i * 4 + k];
            const d1 = f(yb.d);
            var sumi_block: i32 = 0;

            const bits = x[i].qs[k * 4 ..];
            const qy = &yb.qs;

            for (0..4) |b| {
                const mask = bits[b];
                const q = qy[b * 8 ..][0..8];
                for (0..8) |bit| {
                    const v: i32 = q[bit];
                    sumi_block += if ((mask >> @intCast(bit)) & 1 != 0) v else -v;
                }
            }

            sumi += d1 * @as(f32, @floatFromInt(sumi_block));
        }

        sumf += d0 * sumi;
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_q2_0_q8_0_generic` (ggml-cpu/quants.c:182 @cd959bae9).
///
/// Two-bit weights mapped `{0,1,2,3} -> {-1,0,1,2}`. A q2_0 block covers 64
/// elements and so spans two q8_0 blocks.
pub export fn ggml_vec_dot_q2_0_q8_0_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK2_0;
    const nb = @divTrunc(n, qk);

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q2_0, vx);
    const y = as(blocks.Q8_0, vy);

    var sumf: f32 = 0.0;

    for (0..@intCast(nb)) |i| {
        const d0 = f(x[i].d);

        var sumi: f32 = 0.0;

        // One q2_0 block (64 weights) maps to two q8_0 blocks.
        for (0..2) |k| {
            const yb = &y[i * 2 + k];
            const d1 = f(yb.d);
            var sumi_block: i32 = 0;

            const qs = x[i].qs[k * 8 ..];
            const qy = &yb.qs;

            for (0..8) |b| {
                const byte = qs[b];
                inline for (0..4) |t| {
                    const w: i32 = @as(i32, (byte >> (2 * t)) & 3) - 1;
                    sumi_block += w * qy[b * 4 + t];
                }
            }

            sumi += d1 * @as(f32, @floatFromInt(sumi_block));
        }

        sumf += d0 * sumi;
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_q4_0_q8_0_generic` (ggml-cpu/quants.c:230 @cd959bae9).
///
/// Four-bit weights, biased by 8. The low nibbles cover the first half of the
/// block and the high nibbles the second, which is why `sumi0` and `sumi1`
/// index `y` differently.
pub export fn ggml_vec_dot_q4_0_q8_0_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK8_0;
    const nb = @divTrunc(n, qk);

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q4_0, vx);
    const y = as(blocks.Q8_0, vy);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |ib| {
        var sumi0: i32 = 0;
        var sumi1: i32 = 0;

        for (0..qk / 2) |j| {
            const v0: i32 = @as(i32, x[ib].qs[j] & 0x0F) - 8;
            const v1: i32 = @as(i32, x[ib].qs[j] >> 4) - 8;

            sumi0 += v0 * x_q8(y, ib, j);
            sumi1 += v1 * x_q8(y, ib, j + qk / 2);
        }

        const sumi = sumi0 + sumi1;
        sumf += @as(f32, @floatFromInt(sumi)) * f(x[ib].d) * f(y[ib].d);
    }

    s[0] = sumf;
}

/// One element of a q8_0 row, as `i32`.
inline fn x_q8(y: [*]const blocks.Q8_0, ib: usize, j: usize) i32 {
    return y[ib].qs[j];
}

/// Ports `ggml_vec_dot_q4_1_q8_1_generic` (ggml-cpu/quants.c:267 @cd959bae9).
///
/// Unbiased four-bit weights with a per-block minimum. The `m*s` term is the
/// minimum times the right operand's *sum*, which `q8_1` carries in place of
/// `q8_0`'s second delta.
pub export fn ggml_vec_dot_q4_1_q8_1_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK8_1;
    const nb = @divTrunc(n, qk);

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q4_1, vx);
    const y = as(blocks.Q8_1, vy);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |ib| {
        var sumi0: i32 = 0;
        var sumi1: i32 = 0;

        for (0..qk / 2) |j| {
            const v0: i32 = x[ib].qs[j] & 0x0F;
            const v1: i32 = x[ib].qs[j] >> 4;

            sumi0 += v0 * y[ib].qs[j];
            sumi1 += v1 * y[ib].qs[j + qk / 2];
        }

        const sumi = sumi0 + sumi1;
        sumf += (f(x[ib].d) * f(y[ib].d)) * @as(f32, @floatFromInt(sumi)) +
            f(x[ib].m) * f(y[ib].s);
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_mxfp4_q8_0_generic` (ggml-cpu/quants.c:303 @cd959bae9).
///
/// Four-bit indices into `kvalues_mxfp4`, with a shared 8-bit exponent instead
/// of an fp16 delta. Note the C reaches for `GGML_E8M0_TO_FP32_HALF`, the
/// arithmetic form, rather than the table `simd-mappings.h` offers.
pub export fn ggml_vec_dot_mxfp4_q8_0_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(nrc == 1);
    std.debug.assert(@rem(n, c.QK_MXFP4) == 0);
    comptime std.debug.assert(c.QK_MXFP4 == c.QK8_0);
    _ = .{ bs, bx, by };

    const x = as(blocks.MXFP4, vx);
    const y = as(blocks.Q8_0, vy);

    const nb = @divTrunc(n, c.QK_MXFP4);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |ib| {
        const d = f(y[ib].d) * impl.e8m0ToFp32Half(x[ib].e);

        var sumi1: i32 = 0;
        var sumi2: i32 = 0;
        for (0..c.QK_MXFP4 / 2) |j| {
            sumi1 += @as(i32, y[ib].qs[j]) * c.kvalues_mxfp4[x[ib].qs[j] & 0xf];
            sumi2 += @as(i32, y[ib].qs[j + c.QK_MXFP4 / 2]) * c.kvalues_mxfp4[x[ib].qs[j] >> 4];
        }
        sumf += d * @as(f32, @floatFromInt(sumi1 + sumi2));
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_nvfp4_q8_0_generic` (ggml-cpu/quants.c:335 @cd959bae9).
///
/// A 64-element super-block of four 16-element sub-blocks, each with its own
/// ue4m3 scale, laid over two q8_0 blocks. The C uses `ggml_ue4m3_to_fp32`
/// here, not the NEON table lookup `GGML_CPU_UE4M3_TO_FP32` would give.
pub export fn ggml_vec_dot_nvfp4_q8_0_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(nrc == 1);
    std.debug.assert(@rem(n, c.QK_NVFP4) == 0);
    _ = .{ bs, bx, by };

    const x = as(blocks.NVFP4, vx);
    const y = as(blocks.Q8_0, vy);

    const nb = @divTrunc(n, c.QK_NVFP4);
    const sub = c.QK_NVFP4_SUB;

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |ib| {
        for (0..4) |s_idx| {
            const d = impl.ue4m3ToFp32(x[ib].d[s_idx]);
            const q8_block = s_idx / 2;
            const q8_off = (s_idx % 2) * sub;
            const dy = f(y[2 * ib + q8_block].d);

            var sumi_lo: i32 = 0;
            var sumi_hi: i32 = 0;
            for (0..sub / 2) |j| {
                const qv = x[ib].qs[s_idx * (sub / 2) + j];
                const q = &y[2 * ib + q8_block].qs;
                sumi_lo += @as(i32, q[q8_off + j]) * c.kvalues_mxfp4[qv & 0xf];
                sumi_hi += @as(i32, q[q8_off + j + sub / 2]) * c.kvalues_mxfp4[qv >> 4];
            }

            sumf += dy * d * @as(f32, @floatFromInt(sumi_lo + sumi_hi));
        }
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_f8_e4m3_f32` (ggml-cpu/quants.c:370 @cd959bae9).
///
/// **Live, unlike every other dot product in this file**: it has no
/// `_generic` suffix and no `arch/` replacement, so it is the entry point
/// `type_traits_cpu` calls for `GGML_TYPE_F8_E4M3` on every target. The
/// right-hand operand is plain `f32` (`vec_dot_type` is `F32`).
///
/// The C's `GGML_SIMD` arm, written against `GGML_F32_VEC`: each register's
/// lanes are looked up in `ggml_table_f32_f8_e4m3` and fused into one of
/// `GGML_F32_ARR` accumulators, which `GGML_F32_VEC_REDUCE` folds; the
/// `n % GGML_F32_STEP` tail runs scalar. Its `__AVX2__` and
/// `__AVX512F__ && __AVX512BW__` arms fetch the same lanes with
/// `_mm256_i32gather_ps` and `_mm512_i32gather_ps` from the same table, so
/// they load bit-identical vectors and share this body (ported without an x86
/// oracle; the Windows pickup gates it). The tail is `f8_tail_group`'s
/// shape, not plain fused arithmetic -- see there. The C's `assert(nrc == 1)`
/// is compiled out in the reference and ports as nothing.
pub export fn ggml_vec_dot_f8_e4m3_f32(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    _ = .{ bs, bx, by, nrc };
    const F = simd.F32;
    const table = &convert.ggml_table_f32_f8_e4m3;

    const x: [*]const u8 = @ptrCast(vx.?);
    const y: [*]const f32 = @ptrCast(@alignCast(vy.?));
    const nn: usize = @intCast(n);

    var sumf: f32 = 0.0;
    var pos: usize = 0;
    var sum: [F.arr]F.Vec = @splat(F.zero);
    const np = nn - nn % F.step;
    while (pos < np) : (pos += F.step) {
        inline for (0..F.arr) |j| {
            const offset = pos + j * F.epr;
            var values: [F.epr]f32 = undefined;
            inline for (0..F.epr) |k| values[k] = table[x[offset + k]];
            sum[j] = F.fma(sum[j], F.load(&values), F.load(y + offset));
        }
    }
    sumf = F.reduce(&sum);
    const t = nn - pos;
    const group = f8_tail_group orelse t + 1;
    const split = pos + (t - t % group);
    while (pos < split) : (pos += 1) {
        sumf += table[x[pos]] * y[pos];
    }
    while (pos < nn) : (pos += 1) {
        sumf = @mulAdd(f32, table[x[pos]], y[pos], sumf);
    }
    s[0] = sumf;
}

/// How the reference compiler emits the scalar tail of
/// `ggml_vec_dot_f8_e4m3_f32`, `for (; pos < n; ++pos) sumf += table[x[pos]]
/// * y[pos];`. Not a port of anything in the C: a property of the compiler,
/// which the C's arithmetic inherits.
///
/// The table lookup is a gather NEON has no instruction for, so the loop
/// vectorizer does not vectorize this loop the way `simd.strict_loop`
/// describes; it **interleaves it as scalar code**, which is still a strict
/// in-order reduction: groups of products are rounded (`fmul s`) and added in
/// order (`fadd s`), and only the `t % group` left over run fused (`fmadd`).
/// The group is the interleave factor, entered at that many iterations.
/// Measured from `quants.c` compiled by the same toolchain at each CPU:
///
/// - Apple A14 and later (`max_interleave_factor_4`): groups of 4.
/// - `generic` and Apple A7 to A13: groups of 2.
/// - `x86_64_v3`, `x86_64_v4`, `tigerlake`: unrolled by four but every
///   product `vfmadd...ss`, so null -- all fused.
///
/// `make ops-diff` saw it: all-fused failed at `GATE_CPU=generic` on a
/// 37-element row (a tail of 5), where the products round. With the f8 fill
/// alone the products are exact in `f32` and no shape could be seen.
const f8_tail_group: ?usize = blk: {
    const model = builtin.cpu.model.name;
    if (builtin.cpu.arch == .aarch64 and std.mem.startsWith(u8, model, "apple_"))
        break :blk if (std.Target.aarch64.featureSetHas(builtin.cpu.features, .max_interleave_factor_4)) 4 else 2;
    if (builtin.cpu.arch == .aarch64 and std.mem.eql(u8, model, "generic"))
        break :blk 2;
    if (builtin.cpu.arch == .x86_64 and (std.mem.eql(u8, model, "x86_64_v3") or std.mem.eql(u8, model, "x86_64_v4") or std.mem.eql(u8, model, "tigerlake")))
        break :blk null;
    @compileError("the f8 dot product's tail shape is not measured for CPU model '" ++ model ++ "'");
};

/// Ports `ggml_vec_dot_q5_0_q8_0_generic` (ggml-cpu/quants.c:413 @cd959bae9).
///
/// Five-bit weights biased by 16: four bits in `qs` and the fifth spread across
/// `qh` as one bit per element. The two shift expressions are not symmetric --
/// the high half's shift is `j + 12`, not `j + 16`, because the bit is being
/// moved into position 4 rather than extracted to position 0.
pub export fn ggml_vec_dot_q5_0_q8_0_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK8_0;
    const nb = @divTrunc(n, qk);

    std.debug.assert(@rem(n, qk) == 0);
    comptime std.debug.assert(c.QK8_0 == c.QK5_0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q5_0, vx);
    const y = as(blocks.Q8_0, vy);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |ib| {
        const qh = std.mem.readInt(u32, &x[ib].qh, .little);

        var sumi0: i32 = 0;
        var sumi1: i32 = 0;

        for (0..qk / 2) |j| {
            const sh: u5 = @intCast(j);
            const xh_0: u8 = @truncate(((qh & (@as(u32, 1) << sh)) >> sh) << 4);
            const xh_1: u8 = @truncate((qh & (@as(u32, 1) << @intCast(j + 16))) >> @intCast(j + 12));

            const x0: i32 = @as(i8, @bitCast(((x[ib].qs[j] & 0x0F) | xh_0) -% 16));
            const x1: i32 = @as(i8, @bitCast(((x[ib].qs[j] >> 4) | xh_1) -% 16));

            sumi0 += x0 * y[ib].qs[j];
            sumi1 += x1 * y[ib].qs[j + qk / 2];
        }

        const sumi = sumi0 + sumi1;
        sumf += (f(x[ib].d) * f(y[ib].d)) * @as(f32, @floatFromInt(sumi));
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_q5_1_q8_1_generic` (ggml-cpu/quants.c:456 @cd959bae9).
///
/// Unbiased five-bit weights with a per-block minimum. The fifth bit is masked
/// to 0x10 rather than shifted into place, so unlike `q5_0` both halves use the
/// same form.
pub export fn ggml_vec_dot_q5_1_q8_1_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK8_1;
    const nb = @divTrunc(n, qk);

    std.debug.assert(@rem(n, qk) == 0);
    comptime std.debug.assert(c.QK8_1 == c.QK5_1);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q5_1, vx);
    const y = as(blocks.Q8_1, vy);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |ib| {
        const qh = std.mem.readInt(u32, &x[ib].qh, .little);

        var sumi0: i32 = 0;
        var sumi1: i32 = 0;

        for (0..qk / 2) |j| {
            const xh_0: u8 = @truncate(((qh >> @intCast(j)) << 4) & 0x10);
            const xh_1: u8 = @truncate((qh >> @intCast(j + 12)) & 0x10);

            const x0: i32 = (x[ib].qs[j] & 0xF) | xh_0;
            const x1: i32 = (x[ib].qs[j] >> 4) | xh_1;

            sumi0 += x0 * y[ib].qs[j];
            sumi1 += x1 * y[ib].qs[j + qk / 2];
        }

        const sumi = sumi0 + sumi1;
        sumf += (f(x[ib].d) * f(y[ib].d)) * @as(f32, @floatFromInt(sumi)) +
            f(x[ib].m) * f(y[ib].s);
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_q8_0_q8_0_generic` (ggml-cpu/quants.c:499 @cd959bae9).
///
/// The simplest of the family: both operands are already eight-bit, so it is
/// an integer dot product and one scale per block.
pub export fn ggml_vec_dot_q8_0_q8_0_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK8_0;
    const nb = @divTrunc(n, qk);

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q8_0, vx);
    const y = as(blocks.Q8_0, vy);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |ib| {
        var sumi: i32 = 0;

        for (0..qk) |j| {
            sumi += @as(i32, x[ib].qs[j]) * y[ib].qs[j];
        }

        sumf += @as(f32, @floatFromInt(sumi)) * (f(x[ib].d) * f(y[ib].d));
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_iq4_nl_q8_0_generic` (ggml-cpu/quants.c:1302 @cd959bae9).
///
/// Four-bit indices into `kvalues_iq4nl`, the non-linear codebook, otherwise
/// shaped like `q4_0`.
pub export fn ggml_vec_dot_iq4_nl_q8_0_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(nrc == 1);
    std.debug.assert(@rem(n, c.QK4_NL) == 0);
    comptime std.debug.assert(c.QK4_NL == c.QK8_0);
    _ = .{ bs, bx, by };

    const x = as(blocks.IQ4_NL, vx);
    const y = as(blocks.Q8_0, vy);

    const nb = @divTrunc(n, c.QK4_NL);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |ib| {
        const d = f(y[ib].d) * f(x[ib].d);
        var sumi1: i32 = 0;
        var sumi2: i32 = 0;
        for (0..c.QK4_NL / 2) |j| {
            sumi1 += @as(i32, y[ib].qs[j]) * c.kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += @as(i32, y[ib].qs[j + c.QK4_NL / 2]) * c.kvalues_iq4nl[x[ib].qs[j] >> 4];
        }
        sumf += d * @as(f32, @floatFromInt(sumi1 + sumi2));
    }

    s[0] = sumf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

const testing = @import("testing.zig");
const golden = @import("golden.zig");

test "q1_0 dot matches the C" {
    try testing.check(ggml_vec_dot_q1_0_q8_0_generic, c.GGML_TYPE_Q1_0, c.GGML_TYPE_Q8_0, golden.q1_0);
}

test "q2_0 dot matches the C" {
    try testing.check(ggml_vec_dot_q2_0_q8_0_generic, c.GGML_TYPE_Q2_0, c.GGML_TYPE_Q8_0, golden.q2_0);
}

test "q4_0 dot matches the C" {
    try testing.check(ggml_vec_dot_q4_0_q8_0_generic, c.GGML_TYPE_Q4_0, c.GGML_TYPE_Q8_0, golden.q4_0);
}

test "q4_1 dot matches the C" {
    try testing.check(ggml_vec_dot_q4_1_q8_1_generic, c.GGML_TYPE_Q4_1, c.GGML_TYPE_Q8_1, golden.q4_1);
}

test "q5_0 dot matches the C" {
    try testing.check(ggml_vec_dot_q5_0_q8_0_generic, c.GGML_TYPE_Q5_0, c.GGML_TYPE_Q8_0, golden.q5_0);
}

test "q5_1 dot matches the C" {
    try testing.check(ggml_vec_dot_q5_1_q8_1_generic, c.GGML_TYPE_Q5_1, c.GGML_TYPE_Q8_1, golden.q5_1);
}

test "q8_0 dot matches the C" {
    try testing.check(ggml_vec_dot_q8_0_q8_0_generic, c.GGML_TYPE_Q8_0, c.GGML_TYPE_Q8_0, golden.q8_0);
}

test "mxfp4 dot matches the C" {
    try testing.check(ggml_vec_dot_mxfp4_q8_0_generic, c.GGML_TYPE_MXFP4, c.GGML_TYPE_Q8_0, golden.mxfp4);
}

test "nvfp4 dot matches the C" {
    try testing.check(ggml_vec_dot_nvfp4_q8_0_generic, c.GGML_TYPE_NVFP4, c.GGML_TYPE_Q8_0, golden.nvfp4);
}

test "iq4_nl dot matches the C" {
    try testing.check(ggml_vec_dot_iq4_nl_q8_0_generic, c.GGML_TYPE_IQ4_NL, c.GGML_TYPE_Q8_0, golden.iq4_nl);
}
