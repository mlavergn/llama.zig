//! The IQ panel path: a batched `mul_mat` for the grid-based IQ types.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/iqp.cpp` and
//! `llama.cpp/ggml/src/ggml-cpu/iqp.h` (v0.4.1, `b29c606e2`): the portable
//! half of the file -- the panel layout, the eight decoders, the `#else`
//! arms, the eligibility tests and the two entry points. The `__AVX2__` arms
//! are `iqp_x86.zig`, chosen at comptime as the C's `#if` chooses. Each
//! declaration names the C it replaces and the line it began at.
//!
//! # What it does
//!
//! Instead of one `vec_dot` per output element, it decodes eight `src0` rows
//! at a time into per-thread scratch -- a `BlockX8` per super-block, the
//! rows' values as `int8` interleaved four bytes at a time, with an integer
//! scale per 16-value sub-block and one float per row -- and runs an integer
//! GEMM of that panel against `src1`'s `q8_K` rows, four at a time. The
//! panel reproduces `dequantize_row_iq*` exactly: `dfac * iscale * q` is
//! bit-identical to it, which `GGML_IQP_VERIFY` asserts.
//!
//! # Only live where AVX2 is
//!
//! `iqp_supported_common` requires `ggml_cpu_has_avx2()`, which is a
//! compile-time answer. So on ARM, and on x86 below x86-64-v3, nothing here
//! runs: the predicates return false, `ggml_graph_plan` reserves nothing, and
//! `mulmat.zig` takes the `vec_dot` loop. The generic arms below are therefore
//! unreachable in every build -- the C compiles them and never calls them --
//! and they stay **plain**. Clang does fuse their
//! `sumf += (float) sumi * (dfac * d)` (at `-O2` on aarch64 the scale product
//! is an `fmul` and the accumulation an `fmla`, read from the disassembly),
//! but no gate can run that code, so naming the fusion would be a guess
//! nothing could check. The live epilogue is the AVX2 arm's explicit
//! `_mm256_fmadd_ps`.
//!
//! What can run here is the wiring: the unit tests below force both entry
//! points through these arms and hold them to `vec_dot` row for row, which
//! is where an index, tile-tail or scatter mistake would show.
//!
//! # The oracle
//!
//! `scripts/x86-ops-diff` on x86-64-v3 (Haswell, `DESKTOP-BR7J5S9`) and at
//! `x86_64_v4` and `tigerlake` (`JWHLPTOP`): `harness/ops_dump.c` has an IQ
//! `mul_mat` and `mul_mat_id` section shaped to reach this path for every type
//! in `IQP_TYPE_LIST`.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("../impl.zig");
const types = @import("../types.zig");
const qblocks = @import("../quants/blocks.zig");
const defs = @import("defs.zig");
const traits = @import("traits.zig");
const convert = @import("convert.zig");
const features = @import("features.zig");
const threading = @import("threading.zig");
const c = impl.c;

const Tensor = defs.Tensor;
const ComputeParams = defs.ComputeParams;
const Q8K = qblocks.Q8_K;
const QK_K = c.QK_K;

/// `__AVX2__`: whether the `iqp_x86.zig` arms are the ones compiled.
pub const avx2 = x86Has(.avx2);

/// The `__AVX2__` arms, analysed only where they are compiled.
const x86 = if (avx2) @import("iqp_x86.zig") else struct {};

fn x86Has(comptime f: std.Target.x86.Feature) bool {
    return builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, f);
}

/// Ports `GGML_IQP_MIN_BATCH` (iqp.cpp:20 @b29c606e2): the smallest `src1`
/// batch for which the decode pays for itself.
pub const min_batch: i64 = 8;

/// Ports `GGML_IQP_MIN_BATCH_ID` (iqp.cpp:23 @b29c606e2): the same, per
/// expert, for `MUL_MAT_ID`.
pub const min_batch_id: i64 = 8;

/// Ports `ggml_cpu_iqp_mul_mat_id_min_batch` (iqp.cpp:25 @b29c606e2).
///
/// Parameters:
/// - `cne1`: how many `src1` rows were routed to one expert.
///
/// Return: whether that is enough rows for the panel path.
pub export fn ggml_cpu_iqp_mul_mat_id_min_batch(cne1: i64) bool {
    return cne1 >= min_batch_id;
}

/// Ports `IQP_NB_ROWS` (iqp.cpp:30 @b29c606e2): `src0` rows per panel.
pub const nb_rows = 8;

/// Ports `IQP_SB_SIZE` (iqp.cpp:32 @b29c606e2): weights per sub-block.
pub const sb_size = 16;

/// Ports `IQP_NSB` (iqp.cpp:33 @b29c606e2): sub-blocks per super-block.
pub const nsb = QK_K / sb_size;

/// Ports `block_iqp_x8` (iqp.cpp:37 @b29c606e2).
///
/// One super-block of a grid-based IQ type decoded to `int8`, eight rows
/// interleaved: `dfac[row] * iscales[sb*8 + row] * qs` is bit-identical to
/// `dequantize_row_iq*`.
pub const BlockX8 = extern struct {
    /// The `f32` super-block scale, `d * 2^-k`.
    dfac: [8]f32,
    /// `128 * sum(qs * iscale)`, filled only under `use_bias`.
    bias: [8]i32,
    /// Integer sub-block scales, in `[-32, 31]`.
    iscales: [nsb * 8]i8,
    /// `qs[sb*128 + g*32 + row*4 + k]` is column `sb*16 + g*4 + k`.
    qs: [QK_K * 8]i8,
};

comptime {
    std.debug.assert(@sizeOf(BlockX8) == 8 * @sizeOf(f32) + 8 * @sizeOf(i32) + nsb * 8 + QK_K * 8);
}

/// Ports `GGML_IQP_USE_BIAS` (iqp.cpp:49 @b29c606e2).
///
/// With VNNI the activations are fed to `vpdpbusd` as unsigned bytes,
/// `y + 128`, and `bias[]` corrects for it; without, the kernels use the
/// `maddubs` sign trick and `bias[]` is never filled. `tigerlake` has
/// AVX512-VNNI and VL; x86-64-v3 and v4 have neither.
pub const use_bias = avx2 and ((x86Has(.avx512vnni) and x86Has(.avx512vl)) or x86Has(.avxvnni));

/// `GGML_IQP_VERIFY`, a debugging switch the reference does not define: when
/// set, every decoded value is checked against `to_float`. Kept, off.
const verify = false;

/// Ports `ggml_cpu_iqp_row_size` (iqp.cpp:54 @b29c606e2): the bytes of one
/// `src1` row converted to `q8_K`.
inline fn rowSize(dst: *const Tensor) usize {
    return types.ggml_row_size(c.GGML_TYPE_Q8_K, impl.one(Tensor, dst.src[1]).ne[0]);
}

/// Ports `iqp_unpack_ksigns` (iqp.cpp:59 @b29c606e2).
///
/// The low seven bits of `v` are the first seven signs and the eighth is
/// their parity.
pub inline fn unpackKsigns(v: u32) u8 {
    var p = v ^ (v >> 4);

    p ^= p >> 2;
    p ^= p >> 1;

    return @truncate(v ^ ((p & 1) << 7));
}

// -----------------------------------------------------------------------------
// The value stores, each with its `__AVX2__` arm in `iqp_x86.zig`

/// Ports `iqp_store_signed_x8` (iqp.cpp:98 @b29c606e2): 32 values from four
/// 8-byte grid entries, sign byte `l` of `signs` applied to group `l`.
pub inline fn storeSignedX8(dst: [*]i8, g0: u64, g1: u64, g2: u64, g3: u64, signs: u32) void {
    if (comptime avx2) return x86.storeSignedX8(dst, g0, g1, g2, g3, signs);

    const g = [4]u64{ g0, g1, g2, g3 };

    for (0..4) |l| {
        const grid: [8]u8 = @bitCast(g[l]);
        const s: u8 = @truncate(signs >> @intCast(8 * l));

        for (0..8) |j| {
            dst[8 * l + j] = @bitCast(if (s & c.kmask_iq2xs[j] != 0) 0 -% grid[j] else grid[j]);
        }
    }
}

/// Ports `iqp_store_signed_x4` (iqp.cpp:124 @b29c606e2): the same, but the
/// eight values of group `l` come from two 4-byte grid entries.
pub inline fn storeSignedX4(
    dst: [*]i8,
    g0a: u32,
    g0b: u32,
    g1a: u32,
    g1b: u32,
    g2a: u32,
    g2b: u32,
    g3a: u32,
    g3b: u32,
    signs: u32,
) void {
    if (comptime avx2) return x86.storeSignedX4(dst, g0a, g0b, g1a, g1b, g2a, g2b, g3a, g3b, signs);

    const ga = [4]u32{ g0a, g1a, g2a, g3a };
    const gb = [4]u32{ g0b, g1b, g2b, g3b };

    for (0..4) |l| {
        const grid1: [4]u8 = @bitCast(ga[l]);
        const grid2: [4]u8 = @bitCast(gb[l]);
        const s: u8 = @truncate(signs >> @intCast(8 * l));

        for (0..4) |j| {
            dst[8 * l + j + 0] = @bitCast(if (s & c.kmask_iq2xs[j + 0] != 0) 0 -% grid1[j] else grid1[j]);
            dst[8 * l + j + 4] = @bitCast(if (s & c.kmask_iq2xs[j + 4] != 0) 0 -% grid2[j] else grid2[j]);
        }
    }
}

/// Ports `iqp_store_iq1_x8` (iqp.cpp:158 @b29c606e2): 32 values of
/// `8 * grid + delta` from four 8-byte grid entries whose bytes are in
/// `{-1, 0, 1}`, byte `l` of `deltas` applying to group `l`.
pub inline fn storeIq1X8(dst: [*]i8, g0: u64, g1: u64, g2: u64, g3: u64, deltas: u32) void {
    if (comptime avx2) return x86.storeIq1X8(dst, g0, g1, g2, g3, deltas);

    const g = [4]u64{ g0, g1, g2, g3 };

    for (0..4) |l| {
        const grid: [8]i8 = @bitCast(g[l]);
        const delta: i8 = @bitCast(@as(u8, @truncate(deltas >> @intCast(8 * l))));

        for (0..8) |j| {
            dst[8 * l + j] = 8 *% grid[j] +% delta;
        }
    }
}

/// Ports `iqp_store_iq4_x32` (iqp.cpp:188 @b29c606e2): 32 values from 16
/// packed nibbles through `kvalues_iq4nl`, low nibbles first, then high.
pub inline fn storeIq4X32(dst: [*]i8, qs: [*]const u8) void {
    if (comptime avx2) return x86.storeIq4X32(dst, qs);

    for (0..16) |j| {
        dst[j + 0] = c.kvalues_iq4nl[qs[j] & 0xf];
        dst[j + 16] = c.kvalues_iq4nl[qs[j] >> 4];
    }
}

/// Ports `iqp_weighted_sum` (iqp.cpp:207 @b29c606e2): `sum(qs * iscale)`
/// over one super-block, at most `256 * 127 * 32`.
///
/// Compiled only under `use_bias`, which implies AVX2, so the `#else` arm
/// below is unreachable in the C too; it is kept for the record.
pub inline fn weightedSum(vals: *const [QK_K]i8, iscales: *const [nsb]i8) i32 {
    if (comptime avx2) return x86.weightedSum(vals, iscales);

    var wsum: i32 = 0;

    for (0..nsb) |sb| {
        var vsum: i32 = 0;

        for (0..sb_size) |k| {
            vsum += vals[sb * sb_size + k];
        }

        wsum += @as(i32, iscales[sb]) * vsum;
    }

    return wsum;
}

// -----------------------------------------------------------------------------
// The decoders: one super-block of one row to `int8` values, integer
// sub-block scales and the row's float scale

const DecodeFn = fn (vx: *const anyopaque, vals: *[QK_K]i8, iscales: *[nsb]i8, dfac: *f32) void;

inline fn fp16(h: u16) f32 {
    return convert.cpuFp16ToFp32(h);
}

inline fn blockOf(comptime B: type, vx: *const anyopaque) *const B {
    return @ptrCast(@alignCast(vx));
}

/// The four signs of one 32-value group, from the 7-bit packed form the
/// `iq2_xxs` and `iq3_xxs` scale words carry.
inline fn ksigns4(aux: u32) u32 {
    return @as(u32, unpackKsigns((aux >> 0) & 127)) |
        @as(u32, unpackKsigns((aux >> 7) & 127)) << 8 |
        @as(u32, unpackKsigns((aux >> 14) & 127)) << 16 |
        @as(u32, unpackKsigns((aux >> 21) & 127)) << 24;
}

/// Ports `iqp_decode_iq2_xxs` (iqp.cpp:251 @b29c606e2).
///
/// `db = d * (0.5 + ls) * 0.25 = (d / 8) * (2 * ls + 1)`, `ls` four bits.
fn decodeIq2Xxs(vx: *const anyopaque, vals: *[QK_K]i8, iscales: *[nsb]i8, dfac: *f32) void {
    const x = blockOf(qblocks.IQ2_XXS, vx);

    dfac.* = fp16(x.d) * 0.125;

    for (0..QK_K / 32) |ib32| {
        const aux32: [2]u32 = @bitCast(x.qs[4 * ib32 ..][0..4].*);
        const aux8: [4]u8 = @bitCast(aux32[0]);
        const ls: i8 = @intCast(2 * (aux32[1] >> 28) + 1);

        iscales[2 * ib32 + 0] = ls;
        iscales[2 * ib32 + 1] = ls;

        storeSignedX8(
            vals[32 * ib32 ..].ptr,
            c.iq2xxs_grid[aux8[0]],
            c.iq2xxs_grid[aux8[1]],
            c.iq2xxs_grid[aux8[2]],
            c.iq2xxs_grid[aux8[3]],
            ksigns4(aux32[1]),
        );
    }
}

/// Ports `iqp_decode_iq2_xs` (iqp.cpp:280 @b29c606e2).
fn decodeIq2Xs(vx: *const anyopaque, vals: *[QK_K]i8, iscales: *[nsb]i8, dfac: *f32) void {
    const x = blockOf(qblocks.IQ2_XS, vx);

    dfac.* = fp16(x.d) * 0.125;

    for (0..QK_K / 32) |ib32| {
        iscales[2 * ib32 + 0] = @intCast(2 * (x.scales[ib32] & 0xf) + 1);
        iscales[2 * ib32 + 1] = @intCast(2 * (x.scales[ib32] >> 4) + 1);

        const q = x.qs[4 * ib32 ..][0..4];

        const signs = @as(u32, unpackKsigns(q[0] >> 9)) | @as(u32, unpackKsigns(q[1] >> 9)) << 8 |
            @as(u32, unpackKsigns(q[2] >> 9)) << 16 |
            @as(u32, unpackKsigns(q[3] >> 9)) << 24;

        storeSignedX8(
            vals[32 * ib32 ..].ptr,
            c.iq2xs_grid[q[0] & 511],
            c.iq2xs_grid[q[1] & 511],
            c.iq2xs_grid[q[2] & 511],
            c.iq2xs_grid[q[3] & 511],
            signs,
        );
    }
}

/// Ports `iqp_decode_iq2_s` (iqp.cpp:303 @b29c606e2). The sign bytes are the
/// tail of `qs`, at `qs + QK_K/8`.
fn decodeIq2S(vx: *const anyopaque, vals: *[QK_K]i8, iscales: *[nsb]i8, dfac: *f32) void {
    const x = blockOf(qblocks.IQ2_S, vx);

    var qs: [*]const u8 = &x.qs;
    const qh: [*]const u8 = &x.qh;
    var signs: [*]const u8 = qs + QK_K / 8;

    dfac.* = fp16(x.d) * 0.125;

    for (0..QK_K / 32) |ib32| {
        iscales[2 * ib32 + 0] = @intCast(2 * (x.scales[ib32] & 0xf) + 1);
        iscales[2 * ib32 + 1] = @intCast(2 * (x.scales[ib32] >> 4) + 1);

        const sbits = @as(u32, signs[0]) | @as(u32, signs[1]) << 8 | @as(u32, signs[2]) << 16 |
            @as(u32, signs[3]) << 24;
        const h: u32 = qh[ib32];

        storeSignedX8(
            vals[32 * ib32 ..].ptr,
            c.iq2s_grid[qs[0] | (h << 8 & 0x300)],
            c.iq2s_grid[qs[1] | (h << 6 & 0x300)],
            c.iq2s_grid[qs[2] | (h << 4 & 0x300)],
            c.iq2s_grid[qs[3] | (h << 2 & 0x300)],
            sbits,
        );
        qs += 4;
        signs += 4;
    }
}

/// Ports `iqp_decode_iq3_xxs` (iqp.cpp:330 @b29c606e2). The scales and signs
/// are the tail of `qs`, at `qs + QK_K/4`.
///
/// `db = d * (0.5 + ls) * 0.5 = (d / 4) * (2 * ls + 1)`, `ls` four bits.
fn decodeIq3Xxs(vx: *const anyopaque, vals: *[QK_K]i8, iscales: *[nsb]i8, dfac: *f32) void {
    const x = blockOf(qblocks.IQ3_XXS, vx);

    var qs: [*]const u8 = &x.qs;
    const scales_and_signs: [*]const u8 = qs + QK_K / 4;

    dfac.* = fp16(x.d) * 0.25;

    for (0..QK_K / 32) |ib32| {
        const aux32 = std.mem.readInt(u32, scales_and_signs[4 * ib32 ..][0..4], .little);
        const ls: i8 = @intCast(2 * (aux32 >> 28) + 1);

        iscales[2 * ib32 + 0] = ls;
        iscales[2 * ib32 + 1] = ls;

        storeSignedX4(
            vals[32 * ib32 ..].ptr,
            c.iq3xxs_grid[qs[0]],
            c.iq3xxs_grid[qs[1]],
            c.iq3xxs_grid[qs[2]],
            c.iq3xxs_grid[qs[3]],
            c.iq3xxs_grid[qs[4]],
            c.iq3xxs_grid[qs[5]],
            c.iq3xxs_grid[qs[6]],
            c.iq3xxs_grid[qs[7]],
            ksigns4(aux32),
        );
        qs += 8;
    }
}

/// Ports `iqp_decode_iq3_s` (iqp.cpp:363 @b29c606e2).
///
/// `db = d * (1 + 2 * ls)`, `ls` four bits; one scale nibble per 32 values,
/// two 32-value groups per pass.
fn decodeIq3S(vx: *const anyopaque, vals: *[QK_K]i8, iscales: *[nsb]i8, dfac: *f32) void {
    const x = blockOf(qblocks.IQ3_S, vx);

    var qs: [*]const u8 = &x.qs;
    var qh: [*]const u8 = &x.qh;
    var signs: [*]const u8 = &x.signs;

    dfac.* = fp16(x.d);

    var k: usize = 0;

    var ib32: usize = 0;
    while (ib32 < QK_K / 32) : (ib32 += 2) {
        const db1: i8 = @intCast(1 + 2 * (x.scales[ib32 / 2] & 0xf));
        const db2: i8 = @intCast(1 + 2 * (x.scales[ib32 / 2] >> 4));

        iscales[2 * ib32 + 0] = db1;
        iscales[2 * ib32 + 1] = db1;
        iscales[2 * ib32 + 2] = db2;
        iscales[2 * ib32 + 3] = db2;

        for (0..2) |h| {
            const sbits = @as(u32, signs[0]) | @as(u32, signs[1]) << 8 | @as(u32, signs[2]) << 16 |
                @as(u32, signs[3]) << 24;
            const hb: u32 = qh[h];

            storeSignedX4(
                vals[k..].ptr,
                c.iq3s_grid[qs[0] | ((hb << 8) & 256)],
                c.iq3s_grid[qs[1] | ((hb << 7) & 256)],
                c.iq3s_grid[qs[2] | ((hb << 6) & 256)],
                c.iq3s_grid[qs[3] | ((hb << 5) & 256)],
                c.iq3s_grid[qs[4] | ((hb << 4) & 256)],
                c.iq3s_grid[qs[5] | ((hb << 3) & 256)],
                c.iq3s_grid[qs[6] | ((hb << 2) & 256)],
                c.iq3s_grid[qs[7] | ((hb << 1) & 256)],
                sbits,
            );

            k += 32;
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
}

/// Ports `iqp_decode_iq1_s` (iqp.cpp:406 @b29c606e2).
///
/// `dequantize_row_iq1_*` computes `dl * (grid[j] + delta)` with
/// `delta = ±1/8`, so the panel stores `8 * grid[j] ± 1` and folds the `/8`
/// into `dfac`. `dl = d * (2 * ls + 1) * 0.125`, `ls` three bits.
fn decodeIq1S(vx: *const anyopaque, vals: *[QK_K]i8, iscales: *[nsb]i8, dfac: *f32) void {
    const x = blockOf(qblocks.IQ1_S, vx);

    var qs: [*]const u8 = &x.qs;
    const qh = &x.qh;

    dfac.* = fp16(x.d) * 0.125;

    for (0..QK_K / 32) |ib| {
        const h: u32 = qh[ib];
        const dl: i8 = @intCast(2 * ((h >> 12) & 7) + 1);
        const delta: i8 = if (h & 0x8000 != 0) -1 else 1;

        iscales[2 * ib + 0] = dl;
        iscales[2 * ib + 1] = dl;

        storeIq1X8(
            vals[32 * ib ..].ptr,
            c.iq1s_grid[qs[0] | (((h >> 0) & 7) << 8)],
            c.iq1s_grid[qs[1] | (((h >> 3) & 7) << 8)],
            c.iq1s_grid[qs[2] | (((h >> 6) & 7) << 8)],
            c.iq1s_grid[qs[3] | (((h >> 9) & 7) << 8)],
            @as(u32, @as(u8, @bitCast(delta))) *% 0x01010101,
        );
        qs += 4;
    }
}

/// Ports `iqp_decode_iq1_m` (iqp.cpp:432 @b29c606e2).
///
/// `block_iq1_m` has no `d`: the fp16 super-block scale is spread over the
/// top nibbles of the four scale words (`blocks.iq1mScale`).
fn decodeIq1M(vx: *const anyopaque, vals: *[QK_K]i8, iscales: *[nsb]i8, dfac: *f32) void {
    const x = blockOf(qblocks.IQ1_M, vx);

    const sc: [*]const u16 = @ptrCast(@alignCast(&x.scales));

    dfac.* = fp16(qblocks.iq1mScale(sc)) * 0.125;

    var qs: [*]const u8 = &x.qs;
    var qh: [*]const u8 = &x.qh;

    for (0..QK_K / 32) |ib| {
        const s: u32 = sc[ib / 2];
        const shift: u5 = @intCast(6 * (ib % 2));
        iscales[2 * ib + 0] = @intCast(2 * ((s >> shift) & 0x7) + 1);
        iscales[2 * ib + 1] = @intCast(2 * ((s >> (shift + 3)) & 0x7) + 1);

        const h0: u32 = qh[0];
        const h1: u32 = qh[1];
        const idx = [4]u32{
            qs[0] | ((h0 << 8) & 0x700),
            qs[1] | ((h0 << 4) & 0x700),
            qs[2] | ((h1 << 8) & 0x700),
            qs[3] | ((h1 << 4) & 0x700),
        };
        const deltas = @as(u32, if (h0 & 0x08 != 0) 0xff else 0x01) |
            @as(u32, if (h0 & 0x80 != 0) 0xff else 0x01) << 8 |
            @as(u32, if (h1 & 0x08 != 0) 0xff else 0x01) << 16 |
            @as(u32, if (h1 & 0x80 != 0) 0xff else 0x01) << 24;

        storeIq1X8(
            vals[32 * ib ..].ptr,
            c.iq1s_grid[idx[0]],
            c.iq1s_grid[idx[1]],
            c.iq1s_grid[idx[2]],
            c.iq1s_grid[idx[3]],
            deltas,
        );
        qs += 4;
        qh += 2;
    }
}

/// Ports `iqp_decode_iq4_xs` (iqp.cpp:470 @b29c606e2).
///
/// `dl = d * (ls - 32)`, `ls` six bits, so the integer scale is in
/// `[-32, 31]`.
fn decodeIq4Xs(vx: *const anyopaque, vals: *[QK_K]i8, iscales: *[nsb]i8, dfac: *f32) void {
    const x = blockOf(qblocks.IQ4_XS, vx);

    var qs: [*]const u8 = &x.qs;

    dfac.* = fp16(x.d);

    for (0..QK_K / 32) |ib| {
        const lo: u32 = (x.scales_l[ib / 2] >> @intCast(4 * (ib % 2))) & 0xf;
        const hi: u32 = (@as(u32, x.scales_h) >> @intCast(2 * ib)) & 3;
        const ls: i32 = @intCast(lo | (hi << 4));
        const dl: i8 = @intCast(ls - 32);

        iscales[2 * ib + 0] = dl;
        iscales[2 * ib + 1] = dl;

        storeIq4X32(vals[32 * ib ..].ptr, qs);
        qs += 16;
    }
}

/// Ports `IQP_TYPE_LIST` (iqp.cpp:494 @b29c606e2): the types the panel path
/// takes, each with its decoder. Expanded by the eligibility test and the
/// decode dispatch, as the C's X-macro is.
pub const type_list = [_]struct { t: c.enum_ggml_type, decode: *const DecodeFn }{
    .{ .t = c.GGML_TYPE_IQ2_XXS, .decode = &decodeIq2Xxs },
    .{ .t = c.GGML_TYPE_IQ2_XS, .decode = &decodeIq2Xs },
    .{ .t = c.GGML_TYPE_IQ2_S, .decode = &decodeIq2S },
    .{ .t = c.GGML_TYPE_IQ3_XXS, .decode = &decodeIq3Xxs },
    .{ .t = c.GGML_TYPE_IQ3_S, .decode = &decodeIq3S },
    .{ .t = c.GGML_TYPE_IQ1_S, .decode = &decodeIq1S },
    .{ .t = c.GGML_TYPE_IQ1_M, .decode = &decodeIq1M },
    .{ .t = c.GGML_TYPE_IQ4_XS, .decode = &decodeIq4Xs },
};

/// Ports `iqp_decode_superblock` (iqp.cpp:504 @b29c606e2).
///
/// Return: false for a type outside `type_list`.
fn decodeSuperblock(@"type": c.enum_ggml_type, vx: *const anyopaque, vals: *[QK_K]i8, iscales: *[nsb]i8, dfac: *f32) bool {
    inline for (type_list) |e| {
        if (@"type" == e.t) {
            e.decode(vx, vals, iscales, dfac);
            return true;
        }
    }
    return false;
}

/// Ports `iqp_decode_panel_8` (iqp.cpp:565 @b29c606e2).
///
/// Decodes `nb_rows` consecutive source rows, starting at `src` and
/// `nb01` bytes apart, into a panel of `nblocks` blocks at `dst`.
fn decodePanel8(@"type": c.enum_ggml_type, src: [*]const u8, nb01: usize, nblocks: i64, dst_in: [*]BlockX8) void {
    const bsize = types.ggml_type_size(@"type");

    var vals: [nb_rows][QK_K]i8 = undefined;
    var iscales: [nb_rows][nsb]i8 = undefined;
    var dfac: [nb_rows]f32 = undefined;

    var dst = dst_in;

    for (0..@intCast(nblocks)) |x| {
        for (0..nb_rows) |r| {
            const blk = src + r * nb01 + x * bsize;

            const ok = decodeSuperblock(@"type", blk, &vals[r], &iscales[r], &dfac[r]);
            impl.assert(ok, "ok");

            if (comptime verify) {
                // The panel must reproduce the reference dequantization bit
                // for bit.
                var ref: [QK_K]f32 = undefined;
                types.ggml_get_type_traits(@"type").to_float.?(@ptrCast(blk), &ref, QK_K);
                for (0..QK_K) |j| {
                    const scale = dfac[r] * @as(f32, @floatFromInt(iscales[r][j / sb_size]));
                    impl.assert(scale * @as(f32, @floatFromInt(vals[r][j])) == ref[j], "scale * vals[r][j] == ref[j]");
                }
            }
        }

        for (0..nb_rows) |r| {
            dst[0].dfac[r] = dfac[r];

            for (0..nsb) |sb| {
                dst[0].iscales[sb * nb_rows + r] = iscales[r][sb];
            }

            if (comptime use_bias) {
                dst[0].bias[r] = 128 * weightedSum(&vals[r], &iscales[r]);
            }
        }

        if (comptime avx2) {
            for (0..QK_K / 32) |grp| {
                x86.interleaveX8(dst[0].qs[grp * 256 ..].ptr, &vals, grp * 32);
            }
        } else {
            for (0..nb_rows) |r| {
                for (0..nsb) |sb| {
                    for (0..sb_size / 4) |g| {
                        @memcpy(
                            dst[0].qs[sb * 128 + g * 32 + r * 4 ..][0..4],
                            vals[r][sb * sb_size + g * 4 ..][0..4],
                        );
                    }
                }
            }
        }

        dst += 1;
    }
}

// -----------------------------------------------------------------------------
// The generic kernels: `vx` points at `BlockX8`, `vy` at plain, not
// interleaved, `block_q8_K` rows. Unreachable, as the file header says, and
// so unfused.

/// Ports `iqp_gemv_8x8_q8_K_generic` (iqp.cpp:626 @b29c606e2).
fn gemvGeneric(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void {
    const nb: usize = @intCast(@divTrunc(n, QK_K));
    const ncols_interleaved = 8;

    std.debug.assert(@rem(n, QK_K) == 0);
    std.debug.assert(@rem(nc, ncols_interleaved) == 0);

    _ = bs;
    _ = nr;

    const b_ptr_start: [*]const BlockX8 = @ptrCast(@alignCast(vx));
    const a_ptr: [*]const Q8K = @ptrCast(@alignCast(vy));

    for (0..@intCast(@divTrunc(nc, ncols_interleaved))) |x| {
        const b_ptr = b_ptr_start + x * nb;

        var sumf: [8]f32 = @splat(0);

        for (0..nb) |l| {
            var sumi: [8]i32 = @splat(0);

            for (0..nsb) |sb| {
                var isum: [8]i32 = @splat(0);

                for (0..4) |g| {
                    for (0..ncols_interleaved) |j| {
                        for (0..4) |k| {
                            isum[j] += @as(i32, b_ptr[l].qs[sb * 128 + g * 32 + j * 4 + k]) *
                                a_ptr[l].qs[sb * 16 + g * 4 + k];
                        }
                    }
                }

                for (0..ncols_interleaved) |j| {
                    sumi[j] += isum[j] * b_ptr[l].iscales[sb * 8 + j];
                }
            }

            for (0..ncols_interleaved) |j| {
                sumf[j] += @as(f32, @floatFromInt(sumi[j])) * (b_ptr[l].dfac[j] * a_ptr[l].d);
            }
        }

        for (0..ncols_interleaved) |j| {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

/// Ports `iqp_gemm_tile_4_generic` (iqp.cpp:681 @b29c606e2): one 4-row by
/// `nc`-column tile; `s` points at the first of the four output rows, `bs`
/// floats apart.
fn gemmTile4Generic(nb: usize, s: [*]f32, bs: usize, b_ptr_start: [*]const BlockX8, a_ptr: [4][*]const Q8K, nc: c_int) void {
    const ncols_interleaved = 8;

    for (0..@intCast(@divTrunc(nc, ncols_interleaved))) |x| {
        const b_ptr = b_ptr_start + x * nb;

        var sumf: [4][8]f32 = @splat(@splat(0));

        for (0..nb) |l| {
            for (0..4) |m| {
                var sumi: [8]i32 = @splat(0);

                for (0..nsb) |sb| {
                    var isum: [8]i32 = @splat(0);

                    for (0..4) |g| {
                        for (0..ncols_interleaved) |j| {
                            for (0..4) |k| {
                                isum[j] += @as(i32, b_ptr[l].qs[sb * 128 + g * 32 + j * 4 + k]) *
                                    a_ptr[m][l].qs[sb * 16 + g * 4 + k];
                            }
                        }
                    }

                    for (0..ncols_interleaved) |j| {
                        sumi[j] += isum[j] * b_ptr[l].iscales[sb * 8 + j];
                    }
                }

                for (0..ncols_interleaved) |j| {
                    sumf[m][j] += @as(f32, @floatFromInt(sumi[j])) * (b_ptr[l].dfac[j] * a_ptr[m][l].d);
                }
            }
        }

        for (0..4) |m| {
            for (0..ncols_interleaved) |j| {
                s[m * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

/// Ports `iqp_gemm_8x8_q8_K_generic` (iqp.cpp:734 @b29c606e2).
fn gemmGeneric(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void {
    const nb: usize = @intCast(@divTrunc(n, QK_K));

    std.debug.assert(@rem(n, QK_K) == 0);
    std.debug.assert(@rem(nr, 4) == 0);
    std.debug.assert(@rem(nc, 8) == 0);

    const b_ptr_start: [*]const BlockX8 = @ptrCast(@alignCast(vx));
    const a_ptr_start: [*]const Q8K = @ptrCast(@alignCast(vy));

    for (0..@intCast(@divTrunc(nr, 4))) |y| {
        var a_ptr: [4][*]const Q8K = undefined;
        for (0..4) |m| {
            a_ptr[m] = a_ptr_start + (y * 4 + m) * nb;
        }

        gemmTile4Generic(nb, s + y * 4 * bs, bs, b_ptr_start, a_ptr, nc);
    }
}

/// Ports `iqp_gemm_8x8_q8_K_p4_generic` (iqp.cpp:760 @b29c606e2).
fn gemmP4Generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const [4]*const anyopaque, nc: c_int) void {
    const nb: usize = @intCast(@divTrunc(n, QK_K));

    std.debug.assert(@rem(n, QK_K) == 0);
    std.debug.assert(@rem(nc, 8) == 0);

    var a_ptr: [4][*]const Q8K = undefined;
    for (0..4) |m| {
        a_ptr[m] = @ptrCast(@alignCast(vy[m]));
    }

    gemmTile4Generic(nb, s, bs, @ptrCast(@alignCast(vx)), a_ptr, nc);
}

// -----------------------------------------------------------------------------
// The kernels the entry points call, each taking its `__AVX2__` arm where
// that is compiled

/// Ports `iqp_gemv_8x8_q8_K` (iqp.cpp:934 @b29c606e2): one `src1` row
/// against a panel.
fn gemv(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void {
    std.debug.assert(@rem(n, QK_K) == 0);
    std.debug.assert(@rem(nc, 8) == 0);

    if (comptime avx2) return x86.gemv(n, s, @ptrCast(@alignCast(vx)), @ptrCast(@alignCast(vy)), nc);

    gemvGeneric(n, s, bs, vx, vy, nr, nc);
}

/// Ports `iqp_gemm_8x8_q8_K` (iqp.cpp:976 @b29c606e2): `nr` consecutive
/// `src1` rows, a multiple of four, against a panel.
fn gemm(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void {
    std.debug.assert(@rem(n, QK_K) == 0);
    std.debug.assert(@rem(nr, 4) == 0);
    std.debug.assert(@rem(nc, 8) == 0);

    if (comptime avx2) return x86.gemm(n, s, bs, @ptrCast(@alignCast(vx)), @ptrCast(@alignCast(vy)), nr, nc);

    gemmGeneric(n, s, bs, vx, vy, nr, nc);
}

/// Ports `iqp_gemm_8x8_q8_K_p4` (iqp.cpp:1013 @b29c606e2): `gemm` at
/// `nr = 4`, the four activation rows passed separately, for the scattered
/// rows of `MUL_MAT_ID`.
fn gemmP4(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const [4]*const anyopaque, nc: c_int) void {
    std.debug.assert(@rem(n, QK_K) == 0);
    std.debug.assert(@rem(nc, 8) == 0);

    if (comptime avx2) {
        const nb: usize = @intCast(@divTrunc(n, QK_K));
        var a_ptr: [4][*]const Q8K = undefined;
        for (0..4) |m| {
            a_ptr[m] = @ptrCast(@alignCast(vy[m]));
        }
        return x86.gemmTile4(nb, s, bs, @ptrCast(@alignCast(vx)), a_ptr, nc);
    }

    gemmP4Generic(n, s, bs, vx, vy, nc);
}

// -----------------------------------------------------------------------------
// Eligibility

/// Ports `iqp_type_supported` (iqp.cpp:1042 @b29c606e2).
fn typeSupported(@"type": c.enum_ggml_type) bool {
    inline for (type_list) |e| {
        if (@"type" == e.t) return true;
    }
    return false;
}

/// The C's function-local `static const bool disabled`: `GGML_NO_IQ_PANEL`
/// read once, the first time control reaches it. 0 is not read yet, 1 unset,
/// 2 set. Two threads racing to the first read store the same answer.
var no_iq_panel: std.atomic.Value(u8) = .init(0);

/// The escape hatch to A/B the panel against the plain `vec_dot` path
/// without rebuilding; `--no-repack` does not cover this path.
fn panelDisabled() bool {
    var v = no_iq_panel.load(.acquire);
    if (v == 0) {
        v = if (std.c.getenv("GGML_NO_IQ_PANEL") != null) 2 else 1;
        no_iq_panel.store(v, .release);
    }
    return v == 2;
}

/// Ports `iqp_supported_common` (iqp.cpp:1053 @b29c606e2): what `MUL_MAT`
/// and `MUL_MAT_ID` both require.
fn supportedCommon(dst: *const Tensor) bool {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    if (!typeSupported(src0.type)) return false;

    // The path assumes the src1 conversion type is q8_K.
    if (traits.table[@intCast(src0.type)].vec_dot_type != c.GGML_TYPE_Q8_K) return false;

    if (panelDisabled()) return false;

    if (features.ggml_cpu_has_avx2() == 0) return false;

    if (src1.type != c.GGML_TYPE_F32) return false;

    if (@rem(src0.ne[0], QK_K) != 0 or @rem(src0.ne[1], nb_rows) != 0) return false;

    if (src0.ne[3] != 1 or src1.ne[3] != 1 or !types.ggml_is_contiguous(src0)) return false;

    if (dst.type != c.GGML_TYPE_F32 or dst.nb[0] != @sizeOf(f32)) return false;

    return true;
}

/// Ports `ggml_cpu_iqp_supports_mul_mat` (iqp.cpp:1095 @b29c606e2).
///
/// Parameters:
/// - `dst`: a `MUL_MAT` node.
///
/// Return: whether `mulmat.zig` hands it to `ggml_compute_forward_mul_mat_iqp`.
pub export fn ggml_cpu_iqp_supports_mul_mat(dst: *const Tensor) bool {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    if (!supportedCommon(dst)) return false;

    if (src1.ne[1] < min_batch) return false;

    // Plain 2D weight matmuls only; src1 may still be batched over ne12.
    if (src0.ne[2] != 1) return false;

    return true;
}

/// Ports `ggml_cpu_iqp_supports_mul_mat_id` (iqp.cpp:1115 @b29c606e2).
///
/// A node-level test only: each expert is decided by
/// `ggml_cpu_iqp_mul_mat_id_min_batch`.
///
/// Parameters:
/// - `dst`: a `MUL_MAT_ID` node.
///
/// Return: whether any expert could take the panel path, and so whether the
/// work buffer reserves panels for it.
pub export fn ggml_cpu_iqp_supports_mul_mat_id(dst: *const Tensor) bool {
    const ids = impl.one(Tensor, dst.src[2]);

    if (!supportedCommon(dst)) return false;

    // Skip the node entirely, work buffer included, if no expert can reach
    // the per-expert threshold.
    if (!ggml_cpu_iqp_mul_mat_id_min_batch(ids.ne[0] * ids.ne[1])) return false;

    return true;
}

// -----------------------------------------------------------------------------
// Entry points

/// Ports `ggml_compute_forward_mul_mat_id_iqp` (iqp.cpp:1130 @b29c606e2).
///
/// One expert. Must be called after `src1` has been converted to `q8_K` in
/// `params.wdata`, and the threads have synchronised on it.
///
/// Parameters:
/// - `params`: this thread's index, count, and work buffer.
/// - `dst`: the `MUL_MAT_ID` node.
/// - `cur_a`: the expert.
/// - `cne1`: how many rows were routed to it.
/// - `expert_rows`: its row of the `(i1, i2)` routing table.
/// - `panels`: the base of the per-thread panel scratches.
pub export fn ggml_compute_forward_mul_mat_id_iqp(
    params: *const ComputeParams,
    dst: *Tensor,
    cur_a: i64,
    cne1: i64,
    expert_rows: [*]const i32,
    panels: ?*anyopaque,
) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const l = defs.BinaryLocals.of(src0, src1, dst);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nblocks = @divTrunc(l.ne00, QK_K);

    const nbw1 = rowSize(dst);

    const panel: [*]BlockX8 = @ptrCast(@alignCast(@as([*]u8, @ptrCast(panels.?)) +
        @as(usize, @intCast(ith)) * ggml_cpu_iqp_scratch_size(dst)));

    const src0_cur: [*]const u8 = @as([*]const u8, @ptrCast(src0.data.?)) + @as(usize, @intCast(cur_a)) * l.nb02;

    const ngroups = @divTrunc(l.ne01, nb_rows);

    const g0 = @divTrunc(ngroups * ith, nth);
    const g1 = @divTrunc(ngroups * (ith + 1), nth);

    const wdata: [*]const u8 = @ptrCast(params.wdata.?);
    const data: [*]u8 = @ptrCast(dst.data.?);

    var g = g0;
    while (g < g1) : (g += 1) {
        const r: usize = @intCast(g * nb_rows);

        decodePanel8(src0.type, src0_cur + r * l.nb01, l.nb01, nblocks, panel);

        // The dst rows are scattered, so the gemm writes into tmp and it is
        // copied out row by row.
        var tmp: [4 * nb_rows]f32 = undefined;

        var k: i64 = 0;
        while (k < cne1) : (k += 4) {
            const nrows = @min(4, cne1 - k);

            // A short tail tile duplicates its last row into the unused
            // slots; the padding is never copied out.
            var rows: [4]*const anyopaque = undefined;

            for (0..4) |m| {
                const kk: usize = @intCast(k + @min(@as(i64, @intCast(m)), nrows - 1));

                rows[m] = wdata + @as(usize, @intCast(@rem(@as(i64, expert_rows[2 * kk + 0]), l.ne11) +
                    @as(i64, expert_rows[2 * kk + 1]) * l.ne11)) * nbw1;
            }

            gemmP4(@intCast(l.ne00), &tmp, nb_rows, panel, &rows, nb_rows);

            for (0..@intCast(nrows)) |m| {
                const km: usize = @intCast(k + @as(i64, @intCast(m)));
                const dst_col: [*]f32 = @ptrCast(@alignCast(data +
                    @as(usize, @intCast(expert_rows[2 * km + 0])) * l.nb1 +
                    @as(usize, @intCast(expert_rows[2 * km + 1])) * l.nb2));
                @memcpy(dst_col[r..][0..nb_rows], tmp[m * nb_rows ..][0..nb_rows]);
            }
        }
    }
}

/// Ports `ggml_cpu_iqp_scratch_size` (iqp.cpp:1189 @b29c606e2).
///
/// Parameters:
/// - `dst`: a node the panel path takes.
///
/// Return: one thread's panel scratch in bytes, padded to 64.
pub export fn ggml_cpu_iqp_scratch_size(dst: *const Tensor) usize {
    const ne00 = impl.one(Tensor, dst.src[0]).ne[0];
    return impl.pad(@as(usize, @intCast(@divTrunc(ne00, QK_K))) * @sizeOf(BlockX8), 64);
}

/// Ports `ggml_compute_forward_mul_mat_iqp` (iqp.cpp:1193 @b29c606e2).
///
/// Must be called after `src1` has been converted to `q8_K` in
/// `params.wdata` and the threads have synchronised on it, with the chunk
/// counter already reset; the panels sit past the converted rows.
///
/// Parameters:
/// - `params`: this thread's index, count, and work buffer.
/// - `dst`: the `MUL_MAT` node.
pub export fn ggml_compute_forward_mul_mat_iqp(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const l = defs.BinaryLocals.of(src0, src1, dst);

    const ith = params.ith;
    const nth: i64 = params.nth;

    const nblocks = @divTrunc(l.ne00, QK_K);

    const nbw1 = types.ggml_row_size(c.GGML_TYPE_Q8_K, l.ne10);
    const nbw2 = nbw1 * @as(usize, @intCast(l.ne11));

    const scratch_size = ggml_cpu_iqp_scratch_size(dst);

    const scratch_offset = impl.pad(nbw2 * @as(usize, @intCast(l.ne12)), 64);

    impl.assert(
        scratch_offset + @as(usize, @intCast(nth)) * scratch_size <= params.wsize,
        "scratch_offset + (size_t) nth * scratch_size <= params->wsize",
    );

    const wdata: [*]u8 = @ptrCast(params.wdata.?);
    const panel: [*]BlockX8 = @ptrCast(@alignCast(wdata + scratch_offset + @as(usize, @intCast(ith)) * scratch_size));

    const nrows = l.ne11;

    const ngroups = @divTrunc(l.ne01, nb_rows);

    // Aim for four chunks per thread; the caller has already reset the chunk
    // counter. On NUMA systems, one chunk per thread.
    const chunks_per_thread: i64 = if (threading.ggml_is_numa()) 1 else 4;
    const groups_per_chunk = @max(@as(i64, 1), @divTrunc(ngroups + nth * chunks_per_thread - 1, nth * chunks_per_thread));
    const nchunk = @divTrunc(ngroups + groups_per_chunk - 1, groups_per_chunk);

    const data: [*]u8 = @ptrCast(dst.data.?);

    var current_chunk: c_int = ith;

    while (current_chunk < nchunk) {
        const g0 = @as(i64, current_chunk) * groups_per_chunk;
        const g1 = @min(g0 + groups_per_chunk, ngroups);

        var g = g0;
        while (g < g1) : (g += 1) {
            const r: usize = @intCast(g * nb_rows);

            decodePanel8(src0.type, @as([*]const u8, @ptrCast(src0.data.?)) + r * l.nb01, l.nb01, nblocks, panel);

            for (0..@intCast(l.ne12)) |j12| {
                const src1_ptr: [*]const u8 = wdata + j12 * nbw2;
                const dst_ptr: [*]u8 = data + j12 * l.nb2;

                const full = nrows - @rem(nrows, 4);

                if (nrows > 3) {
                    gemm(
                        @intCast(l.ne00),
                        @as([*]f32, @ptrCast(@alignCast(dst_ptr))) + r,
                        l.nb1 / l.nb0,
                        panel,
                        src1_ptr,
                        @intCast(full),
                        nb_rows,
                    );
                }
                var iter = full;
                while (iter < nrows) : (iter += 1) {
                    gemv(
                        @intCast(l.ne00),
                        @as([*]f32, @ptrCast(@alignCast(dst_ptr + @as(usize, @intCast(iter)) * l.nb1))) + r,
                        @intCast(l.ne01),
                        panel,
                        src1_ptr + nbw1 * @as(usize, @intCast(iter)),
                        1,
                        nb_rows,
                    );
                }
            }
        }

        current_chunk = threading.ggml_threadpool_chunk_add(params.threadpool.?, 1);
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "unpackKsigns restores even parity" {
    // Seven sign bits and a parity bit that makes the count of set bits even,
    // which is what the 7-bit packing relies on.
    for (0..128) |v| {
        const b = unpackKsigns(@intCast(v));
        try std.testing.expectEqual(@as(u8, @intCast(v)), b & 0x7f);
        try std.testing.expectEqual(@as(u8, 0), @popCount(b) % 2);
    }
}

/// Random bytes as one block of `t`, with a modest finite fp16 scale so
/// every decoded value is a normal float.
fn randomBlock(t: c.enum_ggml_type, block: []u8, rand: std.Random) void {
    rand.bytes(block);
    const scale: u16 = 0x3000 | (rand.int(u16) & 0x03ff);
    if (t == c.GGML_TYPE_IQ1_M) {
        // No `d`: the scale is the top nibbles of the four scale words.
        const sc: *align(1) [4]u16 = @ptrCast(block[block.len - 8 ..]);
        for (0..4) |w| sc[w] = (sc[w] & 0x0fff) | (@as(u16, @truncate(scale >> @intCast(4 * w))) << 12);
    } else {
        std.mem.writeInt(u16, block[0..2], scale, .little);
    }
}

test "every type in the list decodes to what dequantize_row gives" {
    // `dfac * iscale * q` must be bit-identical to the reference
    // dequantization: the `GGML_IQP_VERIFY` check, run on random blocks of
    // every type, through whichever store arms this target compiles.
    var prng = std.Random.DefaultPrng.init(0x1a2b);
    const rand = prng.random();

    inline for (type_list) |e| {
        const bsize = types.ggml_type_size(e.t);
        var buf: [512]u8 align(16) = undefined;
        std.debug.assert(bsize <= buf.len);

        for (0..64) |_| {
            randomBlock(e.t, buf[0..bsize], rand);

            var vals: [QK_K]i8 = undefined;
            var iscales: [nsb]i8 = undefined;
            var dfac: f32 = undefined;
            try std.testing.expect(decodeSuperblock(e.t, &buf, &vals, &iscales, &dfac));

            var ref: [QK_K]f32 = undefined;
            types.ggml_get_type_traits(e.t).to_float.?(@ptrCast(&buf), &ref, QK_K);

            for (0..QK_K) |j| {
                const s = dfac * @as(f32, @floatFromInt(iscales[j / sb_size]));
                try std.testing.expectEqual(ref[j], s * @as(f32, @floatFromInt(vals[j])));
            }
        }
    }
}

/// A tensor header over caller-owned data, contiguous.
fn testTensor(t: c.enum_ggml_type, ne: [4]i64, data: *anyopaque) Tensor {
    var x = std.mem.zeroes(Tensor);
    x.type = t;
    x.ne = ne;
    x.nb[0] = types.ggml_type_size(t);
    x.nb[1] = types.ggml_row_size(t, ne[0]);
    x.nb[2] = x.nb[1] * @as(usize, @intCast(ne[1]));
    x.nb[3] = x.nb[2] * @as(usize, @intCast(ne[2]));
    x.data = data;
    return x;
}

/// What `vec_dot` gives for one `src0` row against one `q8_K` row.
fn vecDotRow(t: c.enum_ggml_type, n: i64, x: *const anyopaque, y: *const anyopaque) f32 {
    var out: f32 = undefined;
    traits.table[@intCast(t)].vec_dot.?(@intCast(n), &out, 0, x, 0, y, 0, 1);
    return out;
}

/// The test's work buffer: random `q8_K` rows followed by room for panels.
const TestWork = struct {
    buf: []align(64) u8,
    nbw1: usize,
};

fn testWork(n_rows: usize, ne10: i64, extra: usize, rand: std.Random) !TestWork {
    const nbw1 = types.ggml_row_size(c.GGML_TYPE_Q8_K, ne10);
    const buf = try std.testing.allocator.alignedAlloc(u8, .@"64", impl.pad(n_rows * nbw1, 64) + extra);
    var row: [512]f32 = undefined;
    for (0..n_rows) |i| {
        for (row[0..@intCast(ne10)]) |*v| v.* = rand.float(f32) * 2 - 1;
        traits.table[c.GGML_TYPE_Q8_K].from_float.?(&row, buf.ptr + i * nbw1, ne10);
    }
    return .{ .buf = buf, .nbw1 = nbw1 };
}

test "the panel entry points agree with vec_dot, row for row" {
    // The panel path is dead off AVX2, so this forces it: both entry points
    // through this target's arms against `vec_dot` on the same q8_K rows.
    // Different summation orders, so a tolerance -- but a wrong row, column,
    // tile tail or scatter is off by the size of the answer, not its last bit.
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rand = prng.random();
    const a = std.testing.allocator;

    const ne00: i64 = 512;
    const nblk: usize = @intCast(@divExact(ne00, QK_K));

    inline for (type_list) |e| {
        const bsize = types.ggml_type_size(e.t);
        const row_bytes = bsize * nblk;

        // MUL_MAT: 16 rows (two panels) x 11 columns (a gemm of 8 and a gemv
        // tail of 3) x 2 batches.
        {
            const ne01: i64 = 16;
            const ne11: i64 = 11;
            const ne12: i64 = 2;

            const w0 = try a.alignedAlloc(u8, .@"16", row_bytes * @as(usize, @intCast(ne01)));
            defer a.free(w0);
            for (0..@intCast(ne01 * @as(i64, @intCast(nblk)))) |b| randomBlock(e.t, w0[b * bsize ..][0..bsize], rand);

            var src1_dummy: f32 = 0;
            var src0 = testTensor(e.t, .{ ne00, ne01, 1, 1 }, w0.ptr);
            var src1 = testTensor(c.GGML_TYPE_F32, .{ ne00, ne11, ne12, 1 }, &src1_dummy);
            const out = try a.alloc(f32, @intCast(ne01 * ne11 * ne12));
            defer a.free(out);
            var dst = testTensor(c.GGML_TYPE_F32, .{ ne01, ne11, ne12, 1 }, out.ptr);
            dst.src[0] = &src0;
            dst.src[1] = &src1;

            const work = try testWork(@intCast(ne11 * ne12), ne00, ggml_cpu_iqp_scratch_size(&dst), rand);
            defer a.free(work.buf);

            var tp: defs.Threadpool = undefined;
            tp.current_chunk = .init(1);
            const params: ComputeParams = .{
                .ith = 0,
                .nth = 1,
                .wsize = work.buf.len,
                .wdata = work.buf.ptr,
                .threadpool = &tp,
                .use_ref = false,
            };
            ggml_compute_forward_mul_mat_iqp(&params, &dst);

            var scale: f32 = 0;
            for (out) |v| scale = @max(scale, @abs(v));
            try std.testing.expect(scale > 0);

            for (0..@intCast(ne12)) |j12| {
                for (0..@intCast(ne11)) |j11| {
                    for (0..@intCast(ne01)) |j01| {
                        const want = vecDotRow(e.t, ne00, w0.ptr + j01 * row_bytes, work.buf.ptr + (j12 * @as(usize, @intCast(ne11)) + j11) * work.nbw1);
                        const got = out[(j12 * @as(usize, @intCast(ne11)) + j11) * @as(usize, @intCast(ne01)) + j01];
                        try std.testing.expectApproxEqAbs(want, got, 1e-4 * scale);
                    }
                }
            }
        }

        // MUL_MAT_ID: expert 1 of 2, ten routed rows scattered over two
        // slots and seven tokens -- two whole tiles and a tail of two.
        {
            const ne01: i64 = 8;
            const n_as: i64 = 2;
            const ne11: i64 = 2;
            const ne12: i64 = 7;

            const w0 = try a.alignedAlloc(u8, .@"16", row_bytes * @as(usize, @intCast(ne01 * n_as)));
            defer a.free(w0);
            for (0..@intCast(ne01 * n_as * @as(i64, @intCast(nblk)))) |b| randomBlock(e.t, w0[b * bsize ..][0..bsize], rand);

            var src1_dummy: f32 = 0;
            var ids_dummy: i32 = 0;
            var src0 = testTensor(e.t, .{ ne00, ne01, n_as, 1 }, w0.ptr);
            var src1 = testTensor(c.GGML_TYPE_F32, .{ ne00, ne11, ne12, 1 }, &src1_dummy);
            var ids = testTensor(c.GGML_TYPE_I32, .{ 2, ne12, 1, 1 }, &ids_dummy);
            const out = try a.alloc(f32, @intCast(ne01 * 2 * ne12));
            defer a.free(out);
            @memset(out, std.math.nan(f32));
            var dst = testTensor(c.GGML_TYPE_F32, .{ ne01, 2, ne12, 1 }, out.ptr);
            dst.src[0] = &src0;
            dst.src[1] = &src1;
            dst.src[2] = &ids;

            const work = try testWork(@intCast(ne11 * ne12), ne00, ggml_cpu_iqp_scratch_size(&dst), rand);
            defer a.free(work.buf);

            const routed = [_][2]i32{ .{ 0, 0 }, .{ 1, 1 }, .{ 0, 2 }, .{ 1, 3 }, .{ 1, 4 }, .{ 0, 5 }, .{ 1, 6 }, .{ 0, 1 }, .{ 0, 3 }, .{ 1, 5 } };
            const expert_rows: [*]const i32 = @ptrCast(&routed);

            const params: ComputeParams = .{
                .ith = 0,
                .nth = 1,
                .wsize = work.buf.len,
                .wdata = work.buf.ptr,
                .threadpool = null,
                .use_ref = false,
            };
            const panels = work.buf.ptr + impl.pad(@as(usize, @intCast(ne11 * ne12)) * work.nbw1, 64);
            ggml_compute_forward_mul_mat_id_iqp(&params, &dst, 1, routed.len, expert_rows, panels);

            var scale: f32 = 0;
            for (out) |v| {
                if (!std.math.isNan(v)) scale = @max(scale, @abs(v));
            }
            try std.testing.expect(scale > 0);

            var written: usize = 0;
            for (out) |v| written += @intFromBool(!std.math.isNan(v));
            try std.testing.expectEqual(routed.len * @as(usize, @intCast(ne01)), written);

            for (routed) |rm| {
                const j1: usize = @intCast(rm[0]);
                const j2: usize = @intCast(rm[1]);
                const q8 = work.buf.ptr + (j1 % @as(usize, @intCast(ne11)) + j2 * @as(usize, @intCast(ne11))) * work.nbw1;
                for (0..@intCast(ne01)) |j01| {
                    const want = vecDotRow(e.t, ne00, w0.ptr + (@as(usize, @intCast(ne01)) + j01) * row_bytes, q8);
                    const got = out[(j2 * 2 + j1) * @as(usize, @intCast(ne01)) + j01];
                    try std.testing.expectApproxEqAbs(want, got, 1e-4 * scale);
                }
            }
        }
    }
}

test "the panel interleaves four bytes of each row per group" {
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();

    var rows: [nb_rows][@sizeOf(qblocks.IQ4_XS)]u8 align(16) = undefined;
    for (&rows) |*row| {
        rand.bytes(row);
        std.mem.writeInt(u16, row[0..2], 0x3c00, .little);
    }

    var panel: [1]BlockX8 = undefined;
    decodePanel8(c.GGML_TYPE_IQ4_XS, @ptrCast(&rows), @sizeOf(qblocks.IQ4_XS), 1, &panel);

    for (0..nb_rows) |r| {
        var vals: [QK_K]i8 = undefined;
        var iscales: [nsb]i8 = undefined;
        var dfac: f32 = undefined;
        _ = decodeSuperblock(c.GGML_TYPE_IQ4_XS, &rows[r], &vals, &iscales, &dfac);

        try std.testing.expectEqual(dfac, panel[0].dfac[r]);
        for (0..nsb) |sb| {
            try std.testing.expectEqual(iscales[sb], panel[0].iscales[sb * nb_rows + r]);
            for (0..sb_size) |col| {
                const g = col / 4;
                const k = col % 4;
                try std.testing.expectEqual(vals[sb * sb_size + col], panel[0].qs[sb * 128 + g * 32 + r * 4 + k]);
            }
        }
        if (use_bias) {
            try std.testing.expectEqual(128 * weightedSum(&vals, &iscales), panel[0].bias[r]);
        }
    }
}
