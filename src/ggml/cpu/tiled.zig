//! The tiled K-quant matmul: `mul_mat` and `mul_mat_id` for the K- and IQ
//! quants as an integer GEMM over 256x256 windows.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/tiled/tiled.cpp` and
//! `llama.cpp/ggml/src/ggml-cpu/tiled/tiled.h` (v1.0.0, `b1b770d90`): the per-format unpackers, the
//! eligibility test, the workspace sizing, the window drivers and the two
//! entry points `ggml-cpu.c` calls. The microkernels and the tile structs are
//! `tiled_kernel.zig`. Each declaration names the C it replaces and the line
//! it began at.
//!
//! # Only live on x86
//!
//! `ggml_tiled_supported` has two arms. Off AVX it is a stub returning
//! false (tiled.cpp:598 @b1b770d90), which is the arm this Mac compiles: so here
//! `ggml_tiled_wdata_size` is 0, `ggml_graph_plan` only pads the `mul_mat`
//! work size to 64, and both entry points return false at once. On AVX2 and
//! up it takes every K-quant and grid IQ type whose `src0` is not in a
//! repack buffer, gated at run time by `GGML_CPU_TILED_MM` (default on);
//! `GGML_CPU_TILED_MM_FORCE=1` takes it below the 8-row batch floor too.
//!
//! The drivers below are compiled on every target, and the unit tests force
//! them through each of `tiled_kernel.zig`'s arms on this Mac against
//! `vec_dot`. The bit-exact gate is the Windows pickup: `x86-ops-diff` on
//! x86-64-v3 (`DESKTOP-BR7J5S9`) and at `x86_64_v4` and `tigerlake`
//! (`JWHLPTOP`), whose `harness/ops_dump.c` has a tiled section shaped to
//! reach the standard and narrow paths of both ops for every format.
//!
//! # The C's quirk kept
//!
//! `ggml_tiled_supported`'s `switch` returns from every arm, so the `src1`
//! type and contiguity checks after it are dead in the C. They are kept,
//! unreachable, rather than tidied into live code.

const std = @import("std");
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");

inline fn fp16(h: u16) f32 {
    return ggml.impl.fp16ToFp32(h);
}

inline fn readU32(p: [*]const u8) u32 {
    return std.mem.readInt(u32, p[0..4], .little);
}

// -----------------------------------------------------------------------------
// Unpacking src0, one routine per format

/// The geometry every unpacker opens with.
const Unpack = struct {
    qk_stride: usize,
    nb_stride: usize,

    inline fn of(comptime NB: usize, n_rows: c_int, num_k: c_int) Unpack {
        ggml.impl.assert(n_rows <= ggml.cpu.tiled_kernel.tile_rows, "n_rows <= TILED_TILE_ROWS");
        const nk: usize = @intCast(num_k);
        return .{ .qk_stride = nk * ggml.cpu.tiled_kernel.tile_k, .nb_stride = NB * nk };
    }
};

/// The K-quants' 12-byte packed 6-bit scale and min decode, as the
/// reference kernels do it: eight scales then eight mins.
fn kScaleMins(scales12: *const [12]u8) [16]u8 {
    const kmask1: u32 = 0x3f3f3f3f;
    const kmask2: u32 = 0x0f0f0f0f;
    const kmask3: u32 = 0x03030303;
    var utmp: [4]u32 = .{ readU32(scales12[0..]), readU32(scales12[4..]), readU32(scales12[8..]), 0 };
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    var out: [16]u8 = undefined;
    for (0..4) |k| std.mem.writeInt(u32, out[4 * k ..][0..4], utmp[k], .little);
    return out;
}

/// Ports `tiled_unpack_src0` for `block_q4_K` (tiled.cpp:21 @b1b770d90).
fn unpackQ4K(rows: [*]const ggml.quants.blocks.Q4_K, row_stride: i64, n_rows: c_int, tile: *ggml.cpu.tiled_kernel.TileSrc0, num_k: c_int) void {
    const NB = 8; // 32-wide subblocks
    const u = Unpack.of(NB, n_rows, num_k);
    for (0..@intCast(num_k)) |slab| {
        for (0..@intCast(n_rows)) |r| {
            const d_off = slab * ggml.cpu.tiled_kernel.micro + r;
            const q_off = r * u.qk_stride + slab * ggml.cpu.tiled_kernel.tile_k;
            const s_off = r * u.nb_stride + slab * NB;
            const x = &rows[r * @as(usize, @intCast(row_stride)) + slab];

            tile.d[d_off] = fp16(x.d);
            tile.dmin[d_off] = fp16(x.dmin);

            const sm = kScaleMins(&x.scales);
            for (0..NB) |s| {
                tile.scales[s_off + s] = sm[s];
                tile.mins[s_off + s] = sm[8 + s];
            }

            // extract the 4-bit codes (low 4 + high 4), same extraction as the reference kernels
            const q = tile.q[q_off..].ptr;
            ggml.cpu.tiled_kernel.unpkNib4(x.qs[0..].ptr, q + 0, q + 32);
            ggml.cpu.tiled_kernel.unpkNib4(x.qs[32..].ptr, q + 64, q + 96);
            ggml.cpu.tiled_kernel.unpkNib4(x.qs[64..].ptr, q + 128, q + 160);
            ggml.cpu.tiled_kernel.unpkNib4(x.qs[96..].ptr, q + 192, q + 224);
        }
    }
}

/// Ports `tiled_unpack_src0` for `block_q5_K` (tiled.cpp:66 @b1b770d90): the
/// `q4_K` codes with the fifth bit ORed in from `qh`.
fn unpackQ5K(rows: [*]const ggml.quants.blocks.Q5_K, row_stride: i64, n_rows: c_int, tile: *ggml.cpu.tiled_kernel.TileSrc0, num_k: c_int) void {
    const NB = 8; // 32-wide subblocks
    const u = Unpack.of(NB, n_rows, num_k);
    for (0..@intCast(num_k)) |slab| {
        for (0..@intCast(n_rows)) |r| {
            const d_off = slab * ggml.cpu.tiled_kernel.micro + r;
            const q_off = r * u.qk_stride + slab * ggml.cpu.tiled_kernel.tile_k;
            const s_off = r * u.nb_stride + slab * NB;
            const x = &rows[r * @as(usize, @intCast(row_stride)) + slab];

            tile.d[d_off] = fp16(x.d);
            tile.dmin[d_off] = fp16(x.dmin);

            const sm = kScaleMins(&x.scales);
            for (0..NB) |s| {
                tile.scales[s_off + s] = sm[s];
                tile.mins[s_off + s] = sm[8 + s];
            }

            // 64-element chunk j uses qh bits 2j (low 32) and 2j+1 (high 32)
            const q = tile.q[q_off..].ptr;
            ggml.cpu.tiled_kernel.unpkNib4(x.qs[0..].ptr, q + 0, q + 32);
            ggml.cpu.tiled_kernel.unpkNib4(x.qs[32..].ptr, q + 64, q + 96);
            ggml.cpu.tiled_kernel.unpkNib4(x.qs[64..].ptr, q + 128, q + 160);
            ggml.cpu.tiled_kernel.unpkNib4(x.qs[96..].ptr, q + 192, q + 224);
            const qh = x.qh[0..].ptr;
            ggml.cpu.tiled_kernel.unpkOr(0, 4, 1, q + 0, qh);
            ggml.cpu.tiled_kernel.unpkOr(1, 4, 1, q + 32, qh);
            ggml.cpu.tiled_kernel.unpkOr(2, 4, 1, q + 64, qh);
            ggml.cpu.tiled_kernel.unpkOr(3, 4, 1, q + 96, qh);
            ggml.cpu.tiled_kernel.unpkOr(4, 4, 1, q + 128, qh);
            ggml.cpu.tiled_kernel.unpkOr(5, 4, 1, q + 160, qh);
            ggml.cpu.tiled_kernel.unpkOr(6, 4, 1, q + 192, qh);
            ggml.cpu.tiled_kernel.unpkOr(7, 4, 1, q + 224, qh);
        }
    }
}

/// Ports `tiled_unpack_src0` for `block_q6_K` (tiled.cpp:121 @b1b770d90).
fn unpackQ6K(rows: [*]const ggml.quants.blocks.Q6_K, row_stride: i64, n_rows: c_int, tile: *ggml.cpu.tiled_kernel.TileSrc0, num_k: c_int) void {
    const NB = 16; // 16-wide subblocks
    const u = Unpack.of(NB, n_rows, num_k);
    for (0..@intCast(num_k)) |slab| {
        for (0..@intCast(n_rows)) |r| {
            const d_off = slab * ggml.cpu.tiled_kernel.micro + r;
            const q_off = r * u.qk_stride + slab * ggml.cpu.tiled_kernel.tile_k;
            const s_off = r * u.nb_stride + slab * NB;
            const x = &rows[r * @as(usize, @intCast(row_stride)) + slab];
            tile.d[d_off] = fp16(x.d);

            // 6-bit code = 4 low bits (ql) | 2 high bits (qh)
            const q = tile.q[q_off..].ptr;
            for (0..2) |half| {
                const out = q + 128 * half;
                const ql = x.ql[64 * half ..].ptr;
                const qh = x.qh[32 * half ..].ptr;
                ggml.cpu.tiled_kernel.unpkNib4(ql + 0, out + 0, out + 64);
                ggml.cpu.tiled_kernel.unpkNib4(ql + 32, out + 32, out + 96);
                ggml.cpu.tiled_kernel.unpkOr(0, 4, 3, out + 0, qh);
                ggml.cpu.tiled_kernel.unpkOr(2, 4, 3, out + 32, qh);
                ggml.cpu.tiled_kernel.unpkOr(4, 4, 3, out + 64, qh);
                ggml.cpu.tiled_kernel.unpkOr(6, 4, 3, out + 96, qh);
            }
            // scale is a plain int8 per 16-element subblock (16 per 256-K)
            for (0..NB) |s| tile.scales[s_off + s] = x.scales[s];
        }
    }
}

/// Ports `tiled_unpack_src0` for `block_q3_K` (tiled.cpp:152 @b1b770d90):
/// 2-bit codes with the third bit from `hmask`, 6-bit scales stored as
/// `scales - 32`.
fn unpackQ3K(rows: [*]const ggml.quants.blocks.Q3_K, row_stride: i64, n_rows: c_int, tile: *ggml.cpu.tiled_kernel.TileSrc0, num_k: c_int) void {
    const NB = 16; // 16-wide subblocks
    const u = Unpack.of(NB, n_rows, num_k);
    for (0..@intCast(num_k)) |slab| {
        for (0..@intCast(n_rows)) |r| {
            const d_off = slab * ggml.cpu.tiled_kernel.micro + r;
            const q_off = r * u.qk_stride + slab * ggml.cpu.tiled_kernel.tile_k;
            const s_off = r * u.nb_stride + slab * NB;
            const x = &rows[r * @as(usize, @intCast(row_stride)) + slab];
            tile.d[d_off] = fp16(x.d);

            const q = tile.q[q_off..].ptr;
            const s0 = x.qs[0..].ptr;
            const s1 = x.qs[32..].ptr;
            const o0 = q;
            const o1 = q + 128;
            const hm = x.hmask[0..].ptr;
            ggml.cpu.tiled_kernel.unpk2bit(0, s0, o0);
            ggml.cpu.tiled_kernel.unpkOr(0, 2, 1, o0, hm);
            ggml.cpu.tiled_kernel.unpk2bit(2, s0, o0 + 32);
            ggml.cpu.tiled_kernel.unpkOr(1, 2, 1, o0 + 32, hm);
            ggml.cpu.tiled_kernel.unpk2bit(4, s0, o0 + 64);
            ggml.cpu.tiled_kernel.unpkOr(2, 2, 1, o0 + 64, hm);
            ggml.cpu.tiled_kernel.unpk2bit(6, s0, o0 + 96);
            ggml.cpu.tiled_kernel.unpkOr(3, 2, 1, o0 + 96, hm);
            ggml.cpu.tiled_kernel.unpk2bit(0, s1, o1);
            ggml.cpu.tiled_kernel.unpkOr(4, 2, 1, o1, hm);
            ggml.cpu.tiled_kernel.unpk2bit(2, s1, o1 + 32);
            ggml.cpu.tiled_kernel.unpkOr(5, 2, 1, o1 + 32, hm);
            ggml.cpu.tiled_kernel.unpk2bit(4, s1, o1 + 64);
            ggml.cpu.tiled_kernel.unpkOr(6, 2, 1, o1 + 64, hm);
            ggml.cpu.tiled_kernel.unpk2bit(6, s1, o1 + 96);
            ggml.cpu.tiled_kernel.unpkOr(7, 2, 1, o1 + 96, hm);

            // 6-bit scale decode (same kmask trick as the reference), stored as (scales - 32)
            const kmask1: u32 = 0x03030303;
            const kmask2: u32 = 0x0f0f0f0f;
            var auxs: [4]u32 = .{ readU32(x.scales[0..].ptr), readU32(x.scales[4..].ptr), readU32(x.scales[8..].ptr), 0 };
            const tmp = auxs[2];
            auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            var bytes: [16]u8 = undefined;
            for (0..4) |k| std.mem.writeInt(u32, bytes[4 * k ..][0..4], auxs[k], .little);
            for (0..NB) |s| tile.scales[s_off + s] = @as(i32, @as(i8, @bitCast(bytes[s]))) - 32;
        }
    }
}

/// Ports `tiled_unpack_src0` for `block_q2_K` (tiled.cpp:199 @b1b770d90).
fn unpackQ2K(rows: [*]const ggml.quants.blocks.Q2_K, row_stride: i64, n_rows: c_int, tile: *ggml.cpu.tiled_kernel.TileSrc0, num_k: c_int) void {
    const NB = 16; // 16-wide subblocks
    const u = Unpack.of(NB, n_rows, num_k);
    for (0..@intCast(num_k)) |slab| {
        for (0..@intCast(n_rows)) |r| {
            const d_off = slab * ggml.cpu.tiled_kernel.micro + r;
            const q_off = r * u.qk_stride + slab * ggml.cpu.tiled_kernel.tile_k;
            const s_off = r * u.nb_stride + slab * NB;
            const x = &rows[r * @as(usize, @intCast(row_stride)) + slab];
            tile.d[d_off] = fp16(x.d);
            tile.dmin[d_off] = fp16(x.dmin);

            const q = tile.q[q_off..].ptr;
            for (0..2) |half| {
                const s = x.qs[32 * half ..].ptr;
                const out = q + 128 * half;
                ggml.cpu.tiled_kernel.unpk2bit(0, s, out + 0);
                ggml.cpu.tiled_kernel.unpk2bit(2, s, out + 32);
                ggml.cpu.tiled_kernel.unpk2bit(4, s, out + 64);
                ggml.cpu.tiled_kernel.unpk2bit(6, s, out + 96);
            }
            // scale/min packed in one byte per 16-element subblock: low 4 bits = scale, high 4 = min
            for (0..NB) |s| {
                tile.scales[s_off + s] = x.scales[s] & 0xF;
                tile.mins[s_off + s] = x.scales[s] >> 4;
            }
        }
    }
}

/// `lut` in `tiled_unpack_src0` for `block_iq4_xs`: `kvalues_iq4nl + 128`.
const iq4_lut: [16]u8 = blk: {
    var t: [16]u8 = undefined;
    for (0..16) |k| t[k] = @intCast(@as(i32, ggml.impl.c.kvalues_iq4nl[k]) + 128);
    break :blk t;
};

/// Ports `tiled_unpack_src0` for `block_iq4_xs` (tiled.cpp:238 @b1b770d90):
/// 4-bit codes through `kvalues_iq4nl + 128`, 6-bit scales stored as
/// `ls - 32`.
fn unpackIq4Xs(rows: [*]const ggml.quants.blocks.IQ4_XS, row_stride: i64, n_rows: c_int, tile: *ggml.cpu.tiled_kernel.TileSrc0, num_k: c_int) void {
    const NB = 8; // 32-wide subblocks
    const u = Unpack.of(NB, n_rows, num_k);
    for (0..@intCast(num_k)) |slab| {
        for (0..@intCast(n_rows)) |r| {
            const d_off = slab * ggml.cpu.tiled_kernel.micro + r;
            const q_off = r * u.qk_stride + slab * ggml.cpu.tiled_kernel.tile_k;
            const s_off = r * u.nb_stride + slab * NB;
            const x = &rows[r * @as(usize, @intCast(row_stride)) + slab];
            tile.d[d_off] = fp16(x.d);

            for (0..NB) |s| {
                const lo: u32 = (x.scales_l[s / 2] >> @intCast(4 * (s % 2))) & 0xf;
                const hi: u32 = (@as(u32, x.scales_h) >> @intCast(2 * s)) & 3;
                const ls: i32 = @intCast(lo | (hi << 4));
                tile.scales[s_off + s] = ls - 32;
            }

            // 4-bit codes through the LUT: low nibbles of a byte pair come first
            const q = tile.q[q_off..].ptr;
            var lo: [32]u8 = undefined;
            var hi: [32]u8 = undefined;
            for (0..4) |w| {
                ggml.cpu.tiled_kernel.unpkNib4(x.qs[32 * w ..].ptr, &lo, &hi);
                ggml.cpu.tiled_kernel.lut8(&iq4_lut, lo[0..].ptr, q + 64 * w + 0);
                ggml.cpu.tiled_kernel.lut8(&iq4_lut, hi[0..].ptr, q + 64 * w + 16);
                ggml.cpu.tiled_kernel.lut8(&iq4_lut, lo[16..].ptr, q + 64 * w + 32);
                ggml.cpu.tiled_kernel.lut8(&iq4_lut, hi[16..].ptr, q + 64 * w + 48);
            }
        }
    }
}

/// Ports `tiled_unpack_src0` for `block_iq2_xxs` (tiled.cpp:285 @b1b770d90).
fn unpackIq2Xxs(rows: [*]const ggml.quants.blocks.IQ2_XXS, row_stride: i64, n_rows: c_int, tile: *ggml.cpu.tiled_kernel.TileSrc0, num_k: c_int) void {
    const NB = 8; // 32-wide subblocks
    const u = Unpack.of(NB, n_rows, num_k);
    for (0..@intCast(num_k)) |slab| {
        for (0..@intCast(n_rows)) |r| {
            const d_off = slab * ggml.cpu.tiled_kernel.micro + r;
            const q_off = r * u.qk_stride + slab * ggml.cpu.tiled_kernel.tile_k;
            const s_off = r * u.nb_stride + slab * NB;
            const x = &rows[r * @as(usize, @intCast(row_stride)) + slab];
            tile.d[d_off] = fp16(x.d) * 0.125;

            const qs: [*]const u8 = @ptrCast(&x.qs);
            var g: [4]u64 = undefined;
            var signs4: [4]u8 = undefined;
            for (0..NB) |ib32| {
                const aux0 = readU32(qs + 8 * ib32);
                const aux1 = readU32(qs + 8 * ib32 + 4);
                tile.scales[s_off + ib32] = @intCast(2 * (aux1 >> 28) + 1);
                for (0..4) |l| {
                    g[l] = ggml.impl.c.iq2xxs_grid[@as(u8, @truncate(aux0 >> @intCast(8 * l)))];
                    signs4[l] = ggml.impl.c.ksigns_iq2xs[(aux1 >> @intCast(7 * l)) & 127];
                }
                ggml.cpu.tiled_kernel.unpkSign32(g, &signs4, tile.q[q_off + 32 * ib32 ..].ptr);
            }
        }
    }
}

/// Ports `tiled_unpack_src0` for `block_iq2_xs` (tiled.cpp:318 @b1b770d90).
fn unpackIq2Xs(rows: [*]const ggml.quants.blocks.IQ2_XS, row_stride: i64, n_rows: c_int, tile: *ggml.cpu.tiled_kernel.TileSrc0, num_k: c_int) void {
    const NB = 16; // 16-wide subblocks
    const u = Unpack.of(NB, n_rows, num_k);
    for (0..@intCast(num_k)) |slab| {
        for (0..@intCast(n_rows)) |r| {
            const d_off = slab * ggml.cpu.tiled_kernel.micro + r;
            const q_off = r * u.qk_stride + slab * ggml.cpu.tiled_kernel.tile_k;
            const s_off = r * u.nb_stride + slab * NB;
            const x = &rows[r * @as(usize, @intCast(row_stride)) + slab];
            tile.d[d_off] = fp16(x.d) * 0.125;

            var g: [4]u64 = undefined;
            var signs4: [4]u8 = undefined;
            for (0..ggml.impl.c.QK_K / 32) |ib32| {
                tile.scales[s_off + 2 * ib32 + 0] = 2 * @as(i32, x.scales[ib32] & 0xf) + 1;
                tile.scales[s_off + 2 * ib32 + 1] = 2 * @as(i32, x.scales[ib32] >> 4) + 1;
                const q = x.qs[4 * ib32 ..];
                for (0..4) |l| {
                    g[l] = ggml.impl.c.iq2xs_grid[q[l] & 511];
                    signs4[l] = ggml.impl.c.ksigns_iq2xs[q[l] >> 9];
                }
                ggml.cpu.tiled_kernel.unpkSign32(g, &signs4, tile.q[q_off + 32 * ib32 ..].ptr);
            }
        }
    }
}

/// Ports `tiled_unpack_src0` for `block_iq2_s` (tiled.cpp:350 @b1b770d90):
/// the sign bytes share `qs`, as in the reference.
fn unpackIq2S(rows: [*]const ggml.quants.blocks.IQ2_S, row_stride: i64, n_rows: c_int, tile: *ggml.cpu.tiled_kernel.TileSrc0, num_k: c_int) void {
    const NB = 16; // 16-wide subblocks
    const u = Unpack.of(NB, n_rows, num_k);
    for (0..@intCast(num_k)) |slab| {
        for (0..@intCast(n_rows)) |r| {
            const d_off = slab * ggml.cpu.tiled_kernel.micro + r;
            const q_off = r * u.qk_stride + slab * ggml.cpu.tiled_kernel.tile_k;
            const s_off = r * u.nb_stride + slab * NB;
            const x = &rows[r * @as(usize, @intCast(row_stride)) + slab];
            tile.d[d_off] = fp16(x.d) * 0.125;

            var qs: [*]const u8 = &x.qs;
            var signs: [*]const u8 = x.qs[ggml.impl.c.QK_K / 8 ..].ptr;
            var g: [4]u64 = undefined;
            for (0..ggml.impl.c.QK_K / 32) |ib32| {
                tile.scales[s_off + 2 * ib32 + 0] = 2 * @as(i32, x.scales[ib32] & 0xf) + 1;
                tile.scales[s_off + 2 * ib32 + 1] = 2 * @as(i32, x.scales[ib32] >> 4) + 1;
                for (0..4) |l| {
                    const hb: u32 = (@as(u32, x.qh[ib32]) << @intCast(8 - 2 * l)) & 0x300;
                    g[l] = ggml.impl.c.iq2s_grid[qs[l] | hb];
                }
                ggml.cpu.tiled_kernel.unpkSign32(g, signs, tile.q[q_off + 32 * ib32 ..].ptr);
                qs += 4;
                signs += 4;
            }
        }
    }
}

/// Ports `tiled_unpack_src0` for `block_iq3_xxs` (tiled.cpp:383 @b1b770d90).
fn unpackIq3Xxs(rows: [*]const ggml.quants.blocks.IQ3_XXS, row_stride: i64, n_rows: c_int, tile: *ggml.cpu.tiled_kernel.TileSrc0, num_k: c_int) void {
    const NB = 8; // 32-wide subblocks
    const u = Unpack.of(NB, n_rows, num_k);
    for (0..@intCast(num_k)) |slab| {
        for (0..@intCast(n_rows)) |r| {
            const d_off = slab * ggml.cpu.tiled_kernel.micro + r;
            const q_off = r * u.qk_stride + slab * ggml.cpu.tiled_kernel.tile_k;
            const s_off = r * u.nb_stride + slab * NB;
            const x = &rows[r * @as(usize, @intCast(row_stride)) + slab];
            tile.d[d_off] = fp16(x.d) * 0.25;

            var qs: [*]const u8 = &x.qs;
            const scales_and_signs: [*]const u8 = x.qs[ggml.impl.c.QK_K / 4 ..].ptr;
            var g: [4]u64 = undefined;
            var signs4: [4]u8 = undefined;
            for (0..ggml.impl.c.QK_K / 32) |ib32| {
                const aux32 = readU32(scales_and_signs + 4 * ib32);
                tile.scales[s_off + ib32] = @intCast(2 * (aux32 >> 28) + 1);
                for (0..4) |l| {
                    // bytes 8*l .. 8*l+7: low entry e1 (4 values) then high entry e2 (4 values)
                    g[l] = @as(u64, ggml.impl.c.iq3xxs_grid[qs[2 * l + 1]]) << 32 | ggml.impl.c.iq3xxs_grid[qs[2 * l + 0]];
                    signs4[l] = ggml.impl.c.ksigns_iq2xs[(aux32 >> @intCast(7 * l)) & 127];
                }
                ggml.cpu.tiled_kernel.unpkSign32(g, &signs4, tile.q[q_off + 32 * ib32 ..].ptr);
                qs += 8;
            }
        }
    }
}

/// Ports `tiled_unpack_src0` for `block_iq3_s` (tiled.cpp:418 @b1b770d90).
fn unpackIq3S(rows: [*]const ggml.quants.blocks.IQ3_S, row_stride: i64, n_rows: c_int, tile: *ggml.cpu.tiled_kernel.TileSrc0, num_k: c_int) void {
    const NB = 8; // 32-wide subblocks
    const u = Unpack.of(NB, n_rows, num_k);
    for (0..@intCast(num_k)) |slab| {
        for (0..@intCast(n_rows)) |r| {
            const d_off = slab * ggml.cpu.tiled_kernel.micro + r;
            const q_off = r * u.qk_stride + slab * ggml.cpu.tiled_kernel.tile_k;
            const s_off = r * u.nb_stride + slab * NB;
            const x = &rows[r * @as(usize, @intCast(row_stride)) + slab];
            tile.d[d_off] = fp16(x.d);

            var qs: [*]const u8 = &x.qs;
            var qh: [*]const u8 = &x.qh;
            var signs: [*]const u8 = &x.signs;
            var ib32: usize = 0;
            while (ib32 < ggml.impl.c.QK_K / 32) : (ib32 += 2) {
                tile.scales[s_off + ib32 + 0] = 1 + 2 * @as(i32, x.scales[ib32 / 2] & 0xf);
                tile.scales[s_off + ib32 + 1] = 1 + 2 * @as(i32, x.scales[ib32 / 2] >> 4);
                for (0..2) |h| {
                    var g: [4]u64 = undefined;
                    for (0..4) |l| {
                        const qhv: u32 = qh[h];
                        // bytes 8*l .. 8*l+7: low entry e1 (4 values) then high entry e2 (4 values)
                        g[l] = @as(u64, ggml.impl.c.iq3s_grid[qs[2 * l + 1] | ((qhv << @intCast(7 - 2 * l)) & 256)]) << 32 |
                            ggml.impl.c.iq3s_grid[qs[2 * l + 0] | ((qhv << @intCast(8 - 2 * l)) & 256)];
                    }
                    ggml.cpu.tiled_kernel.unpkSign32(g, signs, tile.q[q_off + (ib32 + h) * 32 ..].ptr);
                    qs += 8;
                    signs += 4;
                }
                qh += 2;
            }
        }
    }
}

/// Ports `tiled_unpack_src0` for `block_iq1_s` (tiled.cpp:458 @b1b770d90):
/// the ternary grid times 8, offset by the delta and 128.
fn unpackIq1S(rows: [*]const ggml.quants.blocks.IQ1_S, row_stride: i64, n_rows: c_int, tile: *ggml.cpu.tiled_kernel.TileSrc0, num_k: c_int) void {
    const NB = 8; // 32-wide subblocks
    const u = Unpack.of(NB, n_rows, num_k);
    for (0..@intCast(num_k)) |slab| {
        for (0..@intCast(n_rows)) |r| {
            const d_off = slab * ggml.cpu.tiled_kernel.micro + r;
            const q_off = r * u.qk_stride + slab * ggml.cpu.tiled_kernel.tile_k;
            const s_off = r * u.nb_stride + slab * NB;
            const x = &rows[r * @as(usize, @intCast(row_stride)) + slab];
            tile.d[d_off] = fp16(x.d) * 0.125;

            var qs: [*]const u8 = &x.qs;
            for (0..ggml.impl.c.QK_K / 32) |ib| {
                const hw: u16 = x.qh[ib];
                tile.scales[s_off + ib] = 2 * @as(i32, (hw >> 12) & 7) + 1;
                const delta: i8 = if (hw & 0x8000 != 0) -1 else 1;
                for (0..4) |l| {
                    const entry: u64 = ggml.impl.c.iq1s_grid[qs[l] | (@as(u32, (hw >> @intCast(3 * l)) & 7) << 8)];
                    const bytes: [8]u8 = @bitCast(std.mem.nativeToLittle(u64, entry));
                    ggml.cpu.tiled_kernel.unpkTern8(&bytes, delta, tile.q[q_off + 32 * ib + 8 * l ..].ptr);
                }
                qs += 4;
            }
        }
    }
}

/// Ports `tiled_unpack_src0` for `block_iq1_m` (tiled.cpp:490 @b1b770d90):
/// `iq1_s` with the f16 scale spread across the scale words and a delta per
/// eight.
fn unpackIq1M(rows: [*]const ggml.quants.blocks.IQ1_M, row_stride: i64, n_rows: c_int, tile: *ggml.cpu.tiled_kernel.TileSrc0, num_k: c_int) void {
    const NB = 16; // 16-wide subblocks
    const u = Unpack.of(NB, n_rows, num_k);
    for (0..@intCast(num_k)) |slab| {
        for (0..@intCast(n_rows)) |r| {
            const d_off = slab * ggml.cpu.tiled_kernel.micro + r;
            const q_off = r * u.qk_stride + slab * ggml.cpu.tiled_kernel.tile_k;
            const s_off = r * u.nb_stride + slab * NB;
            const x = &rows[r * @as(usize, @intCast(row_stride)) + slab];

            var sc: [4]u16 = undefined;
            for (0..4) |k| sc[k] = std.mem.readInt(u16, x.scales[2 * k ..][0..2], .little);
            tile.d[d_off] = fp16(ggml.quants.blocks.iq1mScale(&sc)) * 0.125;

            var qs: [*]const u8 = &x.qs;
            var qh: [*]const u8 = &x.qh;
            for (0..ggml.impl.c.QK_K / 32) |ib| {
                const hw = sc[ib / 2];
                const sh: u4 = @intCast(6 * (ib % 2));
                tile.scales[s_off + 2 * ib + 0] = 2 * @as(i32, (hw >> sh) & 7) + 1;
                tile.scales[s_off + 2 * ib + 1] = 2 * @as(i32, (hw >> (sh + 3)) & 7) + 1;

                const idx = [4]u32{
                    qs[0] | ((@as(u32, qh[0]) << 8) & 0x700),
                    qs[1] | ((@as(u32, qh[0]) << 4) & 0x700),
                    qs[2] | ((@as(u32, qh[1]) << 8) & 0x700),
                    qs[3] | ((@as(u32, qh[1]) << 4) & 0x700),
                };
                const delta = [4]i8{
                    if (qh[0] & 0x08 != 0) -1 else 1, if (qh[0] & 0x80 != 0) -1 else 1,
                    if (qh[1] & 0x08 != 0) -1 else 1, if (qh[1] & 0x80 != 0) -1 else 1,
                };
                for (0..4) |l| {
                    const entry: u64 = ggml.impl.c.iq1s_grid[idx[l]];
                    const bytes: [8]u8 = @bitCast(std.mem.nativeToLittle(u64, entry));
                    ggml.cpu.tiled_kernel.unpkTern8(&bytes, delta[l], tile.q[q_off + 32 * ib + 8 * l ..].ptr);
                }
                qs += 4;
                qh += 2;
            }
        }
    }
}

/// Ports `tiled_unpack_src1_q8_K` (tiled.cpp:540 @b1b770d90): `num_k`
/// consecutive `q8_K` blocks of each row into the tile, the ragged tail of
/// the 16-row band zeroed. Codes stay natural; `repackSrc1` rearranges them
/// just in time.
fn unpackSrc1Q8K(rows: [*]const [*]const ggml.quants.blocks.Q8_K, n_rows: c_int, tile: *ggml.cpu.tiled_kernel.TileSrc1, kblk: c_int, num_k: c_int) void {
    ggml.impl.assert(n_rows <= ggml.cpu.tiled_kernel.tile_rows, "n_rows <= TILED_TILE_ROWS");
    const nr: usize = @intCast(n_rows);
    const nk: usize = @intCast(num_k);
    const kb: usize = @intCast(kblk);
    const n_padded = (nr + ggml.cpu.tiled_kernel.micro - 1) & ~@as(usize, ggml.cpu.tiled_kernel.micro - 1);
    const bs_stride: usize = if (nk == 1) ggml.cpu.tiled_kernel.tile_rows else ggml.cpu.tiled_kernel.micro;
    const n16 = ggml.cpu.tiled_kernel.tile_k / ggml.cpu.tiled_kernel.micro;
    for (0..nk) |slab| {
        const q_off = slab * ggml.cpu.tiled_kernel.tile_k;
        for (0..n_padded) |r| {
            const dst = tile.q[r * (nk * ggml.cpu.tiled_kernel.tile_k) + q_off ..][0..ggml.cpu.tiled_kernel.tile_k];
            if (r < nr) {
                @memcpy(dst, &rows[r][kb + slab].qs);
            } else {
                @memset(dst, 0);
            }
        }
        // d and bsums (ISA-independent)
        for (0..n_padded) |r| {
            if (r < nr) {
                const x = &rows[r][kb + slab];
                for (0..n16) |s| tile.bsums[(slab * n16 + s) * bs_stride + r] = x.bsums[s];
                tile.d[slab * ggml.cpu.tiled_kernel.micro + r] = x.d;
            } else {
                for (0..n16) |s| tile.bsums[(slab * n16 + s) * bs_stride + r] = 0;
                tile.d[slab * ggml.cpu.tiled_kernel.micro + r] = 0.0;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Eligibility and workspace

/// The C's `std::call_once` statics, behind one mutex.
const EnvFlags = struct {
    ready: std.atomic.Value(bool) = .init(false),
    mutex: stdlib.sync.Mutex = .{},
    enabled: bool = true,
    forced: bool = false,

    fn get(self: *EnvFlags) *EnvFlags {
        if (self.ready.load(.acquire)) return self;
        self.mutex.lock();
        defer self.mutex.unlock();
        if (!self.ready.load(.acquire)) {
            const env = stdlib.env.getenv("GGML_CPU_TILED_MM");
            self.enabled = env == null or stdlib.num.atoiC(env.?) != 0;
            const fenv = stdlib.env.getenv("GGML_CPU_TILED_MM_FORCE");
            self.forced = fenv != null and stdlib.num.atoiC(fenv.?) == 1;
            self.ready.store(true, .release);
        }
        return self;
    }
};

var env_flags: EnvFlags = .{};

/// Ports `ggml_tiled_matmul_enabled` (tiled.cpp:573 @b1b770d90):
/// `GGML_CPU_TILED_MM`, the master switch, on unless set to 0.
fn matmulEnabled() bool {
    return env_flags.get().enabled;
}

/// Ports `ggml_tiled_matmul_forced` (tiled.cpp:584 @b1b770d90):
/// `GGML_CPU_TILED_MM_FORCE=1` takes the tiled path even when unprofitable.
fn matmulForced() bool {
    return env_flags.get().forced;
}

/// Ports `ggml_tiled_supported` (tiled.cpp:605 @b1b770d90), the arm under
/// `__AVX512VNNI__ || __AVX2__ || __AVX__`, and `ggml_tiled_supported`
/// (tiled.cpp:598 @b1b770d90), the stub every other build compiles.
fn tiledSupported(src0: *const ggml.impl.c.ggml_tensor, src1: *const ggml.impl.c.ggml_tensor) bool {
    // The C goes on to refuse a `src1` that is neither F32 nor a contiguous
    // Q8_K, after the `switch` below -- dead code there, since every arm of
    // the `switch` returns, and not compiled here (Zig rejects unreachable
    // statements). So `src1` is never consulted.
    _ = src1;

    if (comptime ggml.cpu.tiled_kernel.arm == .scalar) return false;

    if (!matmulEnabled()) return false;

    // repack-buffer weights hold a repacked layout, let that kernel handle
    if (src0.extra != null) return false;

    // Supported quant types for src0
    switch (src0.type) {
        ggml.impl.c.GGML_TYPE_Q6_K,
        ggml.impl.c.GGML_TYPE_Q5_K,
        ggml.impl.c.GGML_TYPE_Q4_K,
        ggml.impl.c.GGML_TYPE_Q3_K,
        ggml.impl.c.GGML_TYPE_Q2_K,
        ggml.impl.c.GGML_TYPE_IQ4_XS,
        ggml.impl.c.GGML_TYPE_IQ2_XXS,
        ggml.impl.c.GGML_TYPE_IQ2_XS,
        ggml.impl.c.GGML_TYPE_IQ2_S,
        ggml.impl.c.GGML_TYPE_IQ3_XXS,
        ggml.impl.c.GGML_TYPE_IQ3_S,
        ggml.impl.c.GGML_TYPE_IQ1_S,
        ggml.impl.c.GGML_TYPE_IQ1_M,
        => return true,
        else => return false,
    }
}

/// Ports `ggml_tiled_ws_size` (tiled.cpp:647 @b1b770d90): one thread's slot,
/// 0 when disabled.
fn wsSize() usize {
    if (!matmulEnabled()) return 0;
    return ggml.cpu.tiled_kernel.ws_slot;
}

/// Ports `ggml_tiled_wdata_size` (tiled.cpp:654 @b1b770d90).
///
/// Parameters:
/// - `n_tasks`: the threads the node runs on.
/// - `dst`: the `MUL_MAT` or `MUL_MAT_ID` node.
///
/// Return: the work-buffer bytes to reserve: 64 for alignment plus a slot
/// per thread, or 0 when the tiled path cannot take the node.
pub export fn ggml_tiled_wdata_size(n_tasks: c_int, dst: *const ggml.impl.c.ggml_tensor) callconv(.c) usize {
    if (!tiledSupported(ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]), ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[1]))) {
        return 0; // unsupported, don't allocate
    }
    return 64 + @as(usize, @intCast(n_tasks)) * wsSize();
}

/// Ports `ggml_tiled_narrow_k_extent` (tiled.cpp:664 @b1b770d90): how many
/// K a narrow window reads contiguously, within a 31 KB L1 budget.
fn narrowKExtent(n_rows: i64, ne00: i64) c_int {
    const l1: c_int = 31 * 1024; // fit weights in 31k to leave room for scales

    var rows: c_int = @intCast(n_rows);
    if (rows < ggml.cpu.tiled_kernel.micro) rows = ggml.cpu.tiled_kernel.micro;
    var ke = @divTrunc(l1, ggml.cpu.tiled_kernel.micro + rows);
    ke &= ~@as(c_int, ggml.cpu.tiled_kernel.tile_k - 1); // floor to a multiple of 256
    const ke_max: c_int = ggml.cpu.tiled_kernel.tile_rows * ggml.cpu.tiled_kernel.tile_k / ggml.cpu.tiled_kernel.micro; // buffer ceiling
    if (ke > ke_max) ke = ke_max;
    while (ke >= ggml.cpu.tiled_kernel.tile_k and @rem(ne00, ke) != 0) {
        ke -= ggml.cpu.tiled_kernel.tile_k;
    }
    return if (ke < ggml.cpu.tiled_kernel.tile_k) 0 else ke;
}

/// Ports `tiled_store_window` (tiled.cpp:680 @b1b770d90): the `j`-major
/// window written back `i`-major, 16x8 blocks then the ragged tails.
fn storeWindow(buf: [*]const f32, n_src0: c_int, n_src1: c_int, buf_stride: c_int, dst: [*]f32, dst_stride: usize) void {
    const bs: usize = @intCast(buf_stride);
    const n0: usize = @intCast(n_src0);
    const n1: usize = @intCast(n_src1);
    var ri: usize = 0;
    while (ri + 16 <= n0) : (ri += 16) {
        var rj: usize = 0;
        while (rj + 8 <= n1) : (rj += 8) {
            for (0..8) |uu| {
                for (0..16) |t| dst[(ri + t) + (rj + uu) * dst_stride] = buf[(ri + t) * bs + rj + uu];
            }
        }
        // ragged j tail
        while (rj < n1) : (rj += 1) {
            for (0..16) |t| dst[(ri + t) + rj * dst_stride] = buf[(ri + t) * bs + rj];
        }
    }
    // ragged i tail
    while (ri < n0) : (ri += 1) {
        for (0..n1) |j| dst[ri + j * dst_stride] = buf[ri * bs + j];
    }
}

/// Ports `TILED_MMID_GROUP` (tiled.cpp:719 @b1b770d90): `src0` rows per
/// `MUL_MAT_ID` group, finer than a window so small experts still split.
const mmid_group = 64;

/// Ports `tiled_store_window_scatter` (tiled.cpp:723 @b1b770d90):
/// `storeWindow` into columns that are not contiguous with each other.
fn storeWindowScatter(buf: [*]const f32, n_src0: c_int, n_src1: c_int, buf_stride: c_int, col_ptrs: [*]const [*]f32) void {
    const bs: usize = @intCast(buf_stride);
    const n0: usize = @intCast(n_src0);
    const n1: usize = @intCast(n_src1);
    var ri: usize = 0;
    while (ri + 16 <= n0) : (ri += 16) {
        var rj: usize = 0;
        while (rj + 8 <= n1) : (rj += 8) {
            for (0..8) |uu| {
                for (0..16) |t| col_ptrs[rj + uu][ri + t] = buf[(ri + t) * bs + rj + uu];
            }
        }
        // ragged j tail
        while (rj < n1) : (rj += 1) {
            for (0..16) |t| col_ptrs[rj][ri + t] = buf[(ri + t) * bs + rj];
        }
    }
    // ragged i tail
    while (ri < n0) : (ri += 1) {
        for (0..n1) |j| col_ptrs[j][ri] = buf[ri * bs + j];
    }
}

// -----------------------------------------------------------------------------
// The formats

/// One `tiled_matmul_dispatch` instantiation: the `src0` type, its block,
/// and the kernel constants.
const Format = struct {
    t: ggml.impl.c.enum_ggml_type,
    B: type,
    SUBBLK: usize,
    HAS_MIN: bool,
    BIAS: i32,
    ACTBIAS: bool,
};

fn fmt(comptime t: ggml.impl.c.enum_ggml_type, comptime B: type, comptime SUBBLK: usize, comptime HAS_MIN: bool, comptime BIAS: i32, comptime ACTBIAS: bool) Format {
    return .{ .t = t, .B = B, .SUBBLK = SUBBLK, .HAS_MIN = HAS_MIN, .BIAS = BIAS, .ACTBIAS = ACTBIAS };
}

/// The constants of `ggml_tiled_matmul_type_dispatch` (tiled.cpp:1164
/// @b1b770d90), by `src0` type.
fn formatOf(comptime t: ggml.impl.c.enum_ggml_type) Format {
    return switch (t) {
        ggml.impl.c.GGML_TYPE_Q6_K => fmt(t, ggml.quants.blocks.Q6_K, 16, false, 32, true),
        ggml.impl.c.GGML_TYPE_Q5_K => fmt(t, ggml.quants.blocks.Q5_K, 32, true, 0, false),
        ggml.impl.c.GGML_TYPE_Q4_K => fmt(t, ggml.quants.blocks.Q4_K, 32, true, 0, false),
        ggml.impl.c.GGML_TYPE_Q3_K => fmt(t, ggml.quants.blocks.Q3_K, 16, false, 4, true),
        ggml.impl.c.GGML_TYPE_Q2_K => fmt(t, ggml.quants.blocks.Q2_K, 16, true, 0, false),
        ggml.impl.c.GGML_TYPE_IQ4_XS => fmt(t, ggml.quants.blocks.IQ4_XS, 32, false, 128, false),
        ggml.impl.c.GGML_TYPE_IQ2_XXS => fmt(t, ggml.quants.blocks.IQ2_XXS, 32, false, 128, true),
        ggml.impl.c.GGML_TYPE_IQ2_XS => fmt(t, ggml.quants.blocks.IQ2_XS, 16, false, 128, true),
        ggml.impl.c.GGML_TYPE_IQ2_S => fmt(t, ggml.quants.blocks.IQ2_S, 16, false, 128, true),
        ggml.impl.c.GGML_TYPE_IQ3_XXS => fmt(t, ggml.quants.blocks.IQ3_XXS, 32, false, 128, true),
        ggml.impl.c.GGML_TYPE_IQ3_S => fmt(t, ggml.quants.blocks.IQ3_S, 32, false, 128, true),
        ggml.impl.c.GGML_TYPE_IQ1_S => fmt(t, ggml.quants.blocks.IQ1_S, 32, false, 128, true),
        ggml.impl.c.GGML_TYPE_IQ1_M => fmt(t, ggml.quants.blocks.IQ1_M, 16, false, 128, true),
        else => @compileError("not a tiled format"),
    };
}

/// The thirteen `src0` types, in `ggml_tiled_matmul_type_dispatch`'s order.
pub const tiled_types = [_]ggml.impl.c.enum_ggml_type{
    ggml.impl.c.GGML_TYPE_Q6_K,  ggml.impl.c.GGML_TYPE_Q5_K,    ggml.impl.c.GGML_TYPE_Q4_K,    ggml.impl.c.GGML_TYPE_Q3_K,
    ggml.impl.c.GGML_TYPE_Q2_K,  ggml.impl.c.GGML_TYPE_IQ4_XS,  ggml.impl.c.GGML_TYPE_IQ2_XXS, ggml.impl.c.GGML_TYPE_IQ2_XS,
    ggml.impl.c.GGML_TYPE_IQ2_S, ggml.impl.c.GGML_TYPE_IQ3_XXS, ggml.impl.c.GGML_TYPE_IQ3_S,   ggml.impl.c.GGML_TYPE_IQ1_S,
    ggml.impl.c.GGML_TYPE_IQ1_M,
};

/// The C's `tiled_unpack_src0` overload for `f`'s block type.
inline fn unpackFn(comptime f: Format) *const fn ([*]const f.B, i64, c_int, *ggml.cpu.tiled_kernel.TileSrc0, c_int) void {
    return switch (f.t) {
        ggml.impl.c.GGML_TYPE_Q6_K => unpackQ6K,
        ggml.impl.c.GGML_TYPE_Q5_K => unpackQ5K,
        ggml.impl.c.GGML_TYPE_Q4_K => unpackQ4K,
        ggml.impl.c.GGML_TYPE_Q3_K => unpackQ3K,
        ggml.impl.c.GGML_TYPE_Q2_K => unpackQ2K,
        ggml.impl.c.GGML_TYPE_IQ4_XS => unpackIq4Xs,
        ggml.impl.c.GGML_TYPE_IQ2_XXS => unpackIq2Xxs,
        ggml.impl.c.GGML_TYPE_IQ2_XS => unpackIq2Xs,
        ggml.impl.c.GGML_TYPE_IQ2_S => unpackIq2S,
        ggml.impl.c.GGML_TYPE_IQ3_XXS => unpackIq3Xxs,
        ggml.impl.c.GGML_TYPE_IQ3_S => unpackIq3S,
        ggml.impl.c.GGML_TYPE_IQ1_S => unpackIq1S,
        ggml.impl.c.GGML_TYPE_IQ1_M => unpackIq1M,
        else => @compileError("not a tiled format"),
    };
}

// -----------------------------------------------------------------------------
// The drivers

/// Ports `tiled_mmid_gemm_window` (tiled.cpp:760 @b1b770d90): one `(g, k)`
/// macrotile of `MUL_MAT_ID` -- zero the window, sweep K in 256-element
/// slabs (or `narrowKExtent` chunks for 16 rows or fewer), scatter the rows.
fn mmidGemmWindow(
    comptime a: ggml.cpu.tiled_kernel.Arm,
    comptime f: Format,
    dst: *ggml.impl.c.ggml_tensor,
    src0: *const ggml.impl.c.ggml_tensor,
    src0_cur: [*]const u8,
    r: i64,
    k: i64,
    nrows: i64,
    expert_rows: [*]const i32,
    rows: [*]const [*]const ggml.quants.blocks.Q8_K,
    ws: *ggml.cpu.tiled_kernel.Ws,
) void {
    const unpack = unpackFn(f);
    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];

    const r_end = @min(r + mmid_group, ne01);
    const n_src0: c_int = @intCast(r_end - r);

    const src0_bs = ggml.types.ggml_type_size(src0.type);
    const src0_stride: i64 = @intCast(src0.nb[1] / src0_bs);

    // the window is at most TILED_MMID_GROUP x TILED_TILE_K, so zero only that region of acc
    for (0..@intCast(n_src0)) |i| @memset(ws.acc[i * ggml.cpu.tiled_kernel.tile_rows ..][0..@intCast(nrows)], 0);

    // scattered writeback: column m goes to its routed dst row; r * nb[0] is the window row offset
    var col_ptrs: [ggml.cpu.tiled_kernel.tile_rows][*]f32 = undefined;
    for (0..@intCast(nrows)) |m| {
        const km: usize = @intCast(k + @as(i64, @intCast(m)));
        const e0: usize = @intCast(expert_rows[2 * km + 0]);
        const e1: usize = @intCast(expert_rows[2 * km + 1]);
        col_ptrs[m] = @ptrCast(@alignCast(@as([*]u8, @ptrCast(dst.data.?)) + @as(usize, @intCast(r)) * dst.nb[0] + e0 * dst.nb[1] + e1 * dst.nb[2]));
    }

    if (nrows <= ggml.cpu.tiled_kernel.micro) {
        // narrow path (small nrows): memory bound, so longer skinny tiles
        const k_extent = narrowKExtent(nrows, ne00);
        const num_k = @divTrunc(k_extent, ggml.cpu.tiled_kernel.tile_k);
        var k0: i64 = 0;
        while (k0 < ne00) : (k0 += k_extent) {
            const kstart: c_int = @intCast(@divTrunc(k0, ggml.cpu.tiled_kernel.tile_k));
            unpackSrc1Q8K(rows, @intCast(nrows), &ws.src1, kstart, num_k);
            ggml.cpu.tiled_kernel.repackSrc1(a, &ws.src1, 0, num_k, f.ACTBIAS);
            var ir0 = r;
            while (ir0 < r_end) : (ir0 += ggml.cpu.tiled_kernel.micro) {
                const n0: c_int = @intCast(@min(ggml.cpu.tiled_kernel.micro, r_end - ir0));
                const wbase: [*]const f.B = @ptrCast(@alignCast(src0_cur + @as(usize, @intCast(ir0)) * src0.nb[1] + @as(usize, @intCast(kstart)) * src0_bs));
                unpack(wbase, src0_stride, n0, &ws.src0, num_k);
                ggml.cpu.tiled_kernel.repackSrc0(a, f.SUBBLK, &ws.src0, n0, num_k, f.BIAS, f.ACTBIAS);
                const buf = ws.acc[@as(usize, @intCast(ir0 - r)) * ggml.cpu.tiled_kernel.tile_rows ..].ptr;
                var slab: c_int = 0;
                while (slab < num_k) : (slab += 1) {
                    ggml.cpu.tiled_kernel.runMicrotile(a, f.SUBBLK, f.HAS_MIN, f.BIAS, f.ACTBIAS, &ws.src0, &ws.src1, 0, 0, num_k, slab, buf, ggml.cpu.tiled_kernel.tile_rows);
                }
            }
        }
    } else {
        // standard per-slab path: K is stepped in 256-element slabs
        var ib: i64 = 0;
        while (ib < ne00) : (ib += ggml.cpu.tiled_kernel.tile_k) {
            const kblk: c_int = @intCast(@divTrunc(ib, ggml.cpu.tiled_kernel.tile_k));
            unpack(@ptrCast(@alignCast(src0_cur + @as(usize, @intCast(r)) * src0.nb[1] + @as(usize, @intCast(kblk)) * src0_bs)), src0_stride, n_src0, &ws.src0, 1);
            ggml.cpu.tiled_kernel.repackSrc0(a, f.SUBBLK, &ws.src0, n_src0, 1, f.BIAS, f.ACTBIAS);
            unpackSrc1Q8K(rows, @intCast(nrows), &ws.src1, kblk, 1);
            // 16x16 microtiles sweeping the window; repack each src1 band just-in-time
            var ir1: i64 = 0;
            while (ir1 < nrows) : (ir1 += ggml.cpu.tiled_kernel.micro) {
                ggml.cpu.tiled_kernel.repackSrc1(a, &ws.src1, @intCast(ir1), 1, f.ACTBIAS);
                var ir0 = r;
                while (ir0 < r_end) : (ir0 += ggml.cpu.tiled_kernel.micro) {
                    ggml.cpu.tiled_kernel.runMicrotile(a, f.SUBBLK, f.HAS_MIN, f.BIAS, f.ACTBIAS, &ws.src0, &ws.src1, @intCast(ir0 - r), @intCast(ir1), 1, 0, &ws.acc, ggml.cpu.tiled_kernel.tile_rows);
                }
            }
        }
    }

    storeWindowScatter(&ws.acc, n_src0, @intCast(nrows), ggml.cpu.tiled_kernel.tile_rows, &col_ptrs);
}

/// Ports `ggml_compute_forward_mul_mat_id_tiled_one_expert`
/// (tiled.cpp:838 @b1b770d90): one expert, its routed rows swept in
/// windows of up to 256 over this thread's share of `src0`'s row groups.
fn mulMatIdOneExpert(
    comptime a: ggml.cpu.tiled_kernel.Arm,
    comptime f: Format,
    params: *const ggml.cpu.defs.ComputeParams,
    dst: *ggml.impl.c.ggml_tensor,
    cur_a: i64,
    cne1: i64,
    expert_rows: [*]const i32,
    scratch: [*]u8,
) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[1]);

    const l = ggml.cpu.defs.BinaryLocals.of(src0, src1, dst);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const vec_dot_type = ggml.cpu.traits.table[@intCast(src0.type)].vec_dot_type;

    const nbw1: usize = if (src1.type == vec_dot_type) l.nb11 else ggml.types.ggml_row_size(vec_dot_type, l.ne10);
    const wdata: [*]const u8 = @ptrCast(if (src1.type == vec_dot_type) src1.data.? else params.wdata.?);

    const src0_cur: [*]const u8 = @as([*]const u8, @ptrCast(src0.data.?)) + @as(usize, @intCast(cur_a)) * l.nb02;

    // this thread's workspace
    const ws: *ggml.cpu.tiled_kernel.Ws = @ptrCast(@alignCast(scratch + @as(usize, @intCast(ith)) * wsSize()));

    // groups of TILED_MMID_GROUP rows; rounded up, the window tail is clamped in the gemm
    const ngroups = @divTrunc(l.ne01 + mmid_group - 1, mmid_group);

    const g0 = @divTrunc(ngroups * ith, nth);
    const g1 = @divTrunc(ngroups * (ith + 1), nth);

    // no rows for this thread; nothing to do
    if (g0 >= g1) return;

    var k: i64 = 0;
    while (k < cne1) : (k += ggml.cpu.tiled_kernel.tile_k) {
        const nrows = @min(ggml.cpu.tiled_kernel.tile_k, cne1 - k);

        var rows: [ggml.cpu.tiled_kernel.tile_rows][*]const ggml.quants.blocks.Q8_K = undefined;
        for (0..@intCast(nrows)) |i| {
            const ki: usize = @intCast(k + @as(i64, @intCast(i)));
            const j11 = @rem(@as(i64, expert_rows[2 * ki + 0]), l.ne11);
            const j12: i64 = expert_rows[2 * ki + 1];
            rows[i] = @ptrCast(@alignCast(wdata + @as(usize, @intCast(j11 + j12 * l.ne11)) * nbw1));
        }

        var g = g0;
        while (g < g1) : (g += 1) {
            const r = g * mmid_group;
            mmidGemmWindow(a, f, dst, src0, src0_cur, r, k, nrows, expert_rows, &rows, ws);
        }
    }
}

/// Ports `ggml_compute_forward_mul_mat_tiled_one_chunk`
/// (tiled.cpp:896 @b1b770d90): one chunk of `MUL_MAT`, in 256x256 windows
/// clamped at each `src1` batch boundary.
fn mulMatOneChunk(
    comptime a: ggml.cpu.tiled_kernel.Arm,
    comptime f: Format,
    params: *const ggml.cpu.defs.ComputeParams,
    dst: *ggml.impl.c.ggml_tensor,
    ir0_start: i64,
    ir0_end: i64,
    ir1_start: i64,
    ir1_end: i64,
    ws: *ggml.cpu.tiled_kernel.Ws,
) void {
    const unpack = unpackFn(f);
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[1]);

    const l = ggml.cpu.defs.BinaryLocals.of(src0, src1, dst);

    const vec_dot_type = ggml.cpu.traits.table[@intCast(src0.type)].vec_dot_type;

    // broadcast factors
    const r2 = @divTrunc(l.ne12, l.ne02);
    const r3 = @divTrunc(l.ne13, l.ne03);

    if (ir0_start >= ir0_end or ir1_start >= ir1_end) return;

    const wdata: [*]const u8 = @ptrCast(if (src1.type == vec_dot_type) src1.data.? else params.wdata.?);
    const row_size = ggml.types.ggml_row_size(vec_dot_type, l.ne10);
    const src0_bs = ggml.types.ggml_type_size(src0.type);
    const src1_bs = ggml.types.ggml_type_size(vec_dot_type);

    ggml.impl.assert(@rem(l.ne00, 256) == 0, "ne00 % 256 == 0");

    const src0_stride: i64 = @intCast(l.nb01 / src0_bs); // blocks between src0 rows
    const src1_stride: usize = (if (src1.type == vec_dot_type) src1.nb[1] else row_size) / src1_bs;

    const TILE: i64 = 256;
    const MICRO: i64 = 16;

    var iir1 = ir1_start;
    while (iir1 < ir1_end) {
        var iir1_end = @min(iir1 + TILE, ir1_end);
        const bnd = (@divTrunc(iir1, l.ne11) + 1) * l.ne11;
        if (bnd < iir1_end) iir1_end = bnd;

        const n_src1: c_int = @intCast(iir1_end - iir1);

        // batch coords, constant within the clamped window
        const j13 = @divTrunc(iir1, l.ne12 * l.ne11);
        const j12 = @divTrunc(iir1 - j13 * l.ne12 * l.ne11, l.ne11);
        const j11 = iir1 - j13 * l.ne12 * l.ne11 - j12 * l.ne11;

        const j02 = @divTrunc(j12, r2);
        const j03 = @divTrunc(j13, r3);

        const src0_row: [*]const u8 = @as([*]const u8, @ptrCast(src0.data.?)) + @as(usize, @intCast(j02)) * src0.nb[2] + @as(usize, @intCast(j03)) * src0.nb[3];
        const dst_col: [*]u8 = @as([*]u8, @ptrCast(dst.data.?)) + @as(usize, @intCast(j12)) * l.nb2 + @as(usize, @intCast(j13)) * l.nb3;

        // rows[r] is the window's row r base block; the k-slab offset is applied in the unpack
        var rows: [ggml.cpu.tiled_kernel.tile_rows][*]const ggml.quants.blocks.Q8_K = undefined;
        for (0..@intCast(n_src1)) |r| {
            rows[r] = @ptrCast(@alignCast(wdata + (@as(usize, @intCast(iir1)) + r) * src1_stride * src1_bs));
        }

        var iir0 = ir0_start;
        while (iir0 < ir0_end) : (iir0 += TILE) {
            const iir0_end = @min(iir0 + TILE, ir0_end);
            const n_src0: c_int = @intCast(iir0_end - iir0);

            // result buffer zeroed once per macrotile
            @memset(&ws.acc, 0);

            if (n_src1 <= ggml.cpu.tiled_kernel.micro) {
                // narrow path: few rows, memory bound; longer, skinnier tiles
                const k_extent = narrowKExtent(n_src1, l.ne00);
                const num_k = @divTrunc(k_extent, ggml.cpu.tiled_kernel.tile_k);
                var k0: i64 = 0;
                while (k0 < l.ne00) : (k0 += k_extent) {
                    const kstart: c_int = @intCast(@divTrunc(k0, ggml.cpu.tiled_kernel.tile_k));
                    unpackSrc1Q8K(&rows, n_src1, &ws.src1, kstart, num_k);
                    ggml.cpu.tiled_kernel.repackSrc1(a, &ws.src1, 0, num_k, f.ACTBIAS);
                    var ir0 = iir0;
                    while (ir0 < iir0_end) : (ir0 += MICRO) {
                        const n0: c_int = @intCast(@min(MICRO, iir0_end - ir0));
                        const wbase: [*]const f.B = @ptrCast(@alignCast(src0_row + @as(usize, @intCast(ir0)) * l.nb01 + @as(usize, @intCast(kstart)) * src0_bs));
                        unpack(wbase, src0_stride, n0, &ws.src0, num_k);
                        ggml.cpu.tiled_kernel.repackSrc0(a, f.SUBBLK, &ws.src0, n0, num_k, f.BIAS, f.ACTBIAS);
                        const buf = ws.acc[@as(usize, @intCast(ir0 - iir0)) * ggml.cpu.tiled_kernel.tile_rows ..].ptr;
                        var slab: c_int = 0;
                        while (slab < num_k) : (slab += 1) {
                            ggml.cpu.tiled_kernel.runMicrotile(a, f.SUBBLK, f.HAS_MIN, f.BIAS, f.ACTBIAS, &ws.src0, &ws.src1, 0, 0, num_k, slab, buf, ggml.cpu.tiled_kernel.tile_rows);
                        }
                    }
                }
            } else {
                // standard per-slab path: K is stepped in 256-element chunks
                var ib: i64 = 0;
                while (ib < l.ne00) : (ib += TILE) {
                    const kblk: c_int = @intCast(@divTrunc(ib, TILE));
                    unpack(@ptrCast(@alignCast(src0_row + @as(usize, @intCast(iir0)) * l.nb01 + @as(usize, @intCast(kblk)) * src0_bs)), src0_stride, n_src0, &ws.src0, 1);
                    ggml.cpu.tiled_kernel.repackSrc0(a, f.SUBBLK, &ws.src0, n_src0, 1, f.BIAS, f.ACTBIAS);
                    unpackSrc1Q8K(&rows, n_src1, &ws.src1, kblk, 1);

                    // 16x16 microtiles sweeping the window; repack each src1 band before first use
                    var ir1 = iir1;
                    while (ir1 < iir1_end) : (ir1 += MICRO) {
                        ggml.cpu.tiled_kernel.repackSrc1(a, &ws.src1, @intCast(ir1 - iir1), 1, f.ACTBIAS);
                        var ir0 = iir0;
                        while (ir0 < iir0_end) : (ir0 += MICRO) {
                            ggml.cpu.tiled_kernel.runMicrotile(a, f.SUBBLK, f.HAS_MIN, f.BIAS, f.ACTBIAS, &ws.src0, &ws.src1, @intCast(ir0 - iir0), @intCast(ir1 - iir1), 1, 0, &ws.acc, ggml.cpu.tiled_kernel.tile_rows);
                        }
                    }
                }
            }
            // write acc back out from L2 to main memory
            storeWindow(
                &ws.acc,
                n_src0,
                n_src1,
                ggml.cpu.tiled_kernel.tile_rows,
                @ptrCast(@alignCast(dst_col + @as(usize, @intCast(iir0)) * l.nb0 + @as(usize, @intCast(j11)) * l.nb1)),
                l.nb1 / l.nb0,
            );
        }
        iir1 = iir1_end;
    }
}

/// Ports `ggml_compute_forward_mul_mat_tiled_driver` (tiled.cpp:1028
/// @b1b770d90): the whole `MUL_MAT`, entered by every thread -- `src1` to
/// `q8_K`, a barrier, then chunks of up to 256x256 handed out by the
/// threadpool's counter, the chunk size halved until there are four per
/// thread.
fn mulMatDriver(comptime a: ggml.cpu.tiled_kernel.Arm, comptime f: Format, params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[1]);

    const l = ggml.cpu.defs.BinaryLocals.of(src0, src1, dst);

    const ith: c_int = params.ith;
    const nth: c_int = params.nth;

    const vec_dot_type = ggml.cpu.traits.table[@intCast(src0.type)].vec_dot_type;
    const from_float = ggml.cpu.traits.table[@intCast(vec_dot_type)].from_float;

    ggml.impl.assert(l.ne0 == l.ne01, "ne0 == ne01");
    ggml.impl.assert(l.ne1 == l.ne11, "ne1 == ne11");
    ggml.impl.assert(l.ne2 == l.ne12, "ne2 == ne12");
    ggml.impl.assert(l.ne3 == l.ne13, "ne3 == ne13");

    // we don't support permuted src0 or src1
    ggml.impl.assert(l.nb10 == ggml.types.ggml_type_size(src1.type), "nb10 == ggml_type_size(src1->type)");

    // dst cannot be transposed or permuted
    ggml.impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");
    ggml.impl.assert(l.nb0 <= l.nb1, "nb0 <= nb1");
    ggml.impl.assert(l.nb1 <= l.nb2, "nb1 <= nb2");
    ggml.impl.assert(l.nb2 <= l.nb3, "nb2 <= nb3");

    if (src1.type != vec_dot_type) {
        const wdata: [*]u8 = @ptrCast(params.wdata.?);

        const nbw0 = ggml.types.ggml_type_size(vec_dot_type);
        const nbw1 = ggml.types.ggml_row_size(vec_dot_type, l.ne10);
        const nbw2 = nbw1 * @as(usize, @intCast(l.ne11));
        const nbw3 = nbw2 * @as(usize, @intCast(l.ne12));

        ggml.impl.assert(src1.type == ggml.impl.c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");

        for (0..@intCast(l.ne13)) |j13| {
            for (0..@intCast(l.ne12)) |j12| {
                for (0..@intCast(l.ne11)) |j11| {
                    const bs = ggml.types.ggml_blck_size(vec_dot_type);
                    const block_start = @divTrunc(@as(i64, ith) * @divTrunc(l.ne10, bs), nth);
                    const block_end = @divTrunc((@as(i64, ith) + 1) * @divTrunc(l.ne10, bs), nth);
                    from_float.?(
                        @ptrCast(@alignCast(@as([*]const u8, @ptrCast(src1.data.?)) + j13 * src1.nb[3] + j12 * src1.nb[2] + j11 * src1.nb[1] + @as(usize, @intCast(block_start * bs)) * src1.nb[0])),
                        wdata + j13 * nbw3 + j12 * nbw2 + j11 * nbw1 + @as(usize, @intCast(block_start)) * nbw0,
                        (block_end - block_start) * bs,
                    );
                }
            }
        }
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.
        ggml.cpu.threading.ggml_threadpool_chunk_set(params.threadpool.?, nth);
    }

    ggml.cpu.threading.ggml_barrier(params.threadpool.?);

    const nr0 = l.ne0;
    const nr1 = l.ne1 * l.ne2 * l.ne3;

    var chunk_size: i64 = 256; //TILED_TILE_ROWS;

    var nchunk0 = @divTrunc(nr0 + chunk_size - 1, chunk_size);
    var nchunk1 = @divTrunc(nr1 + chunk_size - 1, chunk_size);

    // Step down chunk size if too few chunks to saturate cores, minimum is microtile size
    while (nchunk0 * nchunk1 < @as(i64, nth) * 4 and chunk_size > 16) {
        chunk_size = @divTrunc(chunk_size, 2);
        nchunk0 = @divTrunc(nr0 + chunk_size - 1, chunk_size);
        nchunk1 = @divTrunc(nr1 + chunk_size - 1, chunk_size);
    }

    const dr0 = @divTrunc(nr0 + nchunk0 - 1, nchunk0);
    const dr1 = @divTrunc(nr1 + nchunk1 - 1, nchunk1);

    var current_chunk: i64 = ith;

    // per-thread workspace: after any converted src1 data in wdata, aligned to 64
    var ws_base: usize = @intFromPtr(params.wdata.?);
    if (src1.type != vec_dot_type) {
        ws_base += stdlib.num.pad(usize, ggml.types.ggml_row_size(vec_dot_type, ggml.types.ggml_nelements(src1)), 64);
    }
    ws_base = (ws_base + 63) & ~@as(usize, 63);
    const ws: *ggml.cpu.tiled_kernel.Ws = @ptrFromInt(ws_base + @as(usize, @intCast(ith)) * wsSize());

    while (current_chunk < nchunk0 * nchunk1) {
        const ith0 = @rem(current_chunk, nchunk0);
        const ith1 = @divTrunc(current_chunk, nchunk0);

        const ir0_start = dr0 * ith0;
        const ir0_end = @min(ir0_start + dr0, nr0);

        const ir1_start = dr1 * ith1;
        const ir1_end = @min(ir1_start + dr1, nr1);

        mulMatOneChunk(a, f, params, dst, ir0_start, ir0_end, ir1_start, ir1_end, ws);

        if (nth >= nchunk0 * nchunk1) break;

        current_chunk = ggml.cpu.threading.ggml_threadpool_chunk_add(params.threadpool.?, 1);
    }
}

/// Ports `tiled_matmul_dispatch` (tiled.cpp:1149 @b1b770d90) and
/// `ggml_tiled_matmul_type_dispatch` (tiled.cpp:1164 @b1b770d90): the
/// `src0` type switch both ops share -- one expert of `MUL_MAT_ID` when
/// `expert_rows` is set, the whole `MUL_MAT` otherwise.
///
/// Return: false for a type the tiled path does not take.
fn typeDispatch(
    comptime a: ggml.cpu.tiled_kernel.Arm,
    params: *const ggml.cpu.defs.ComputeParams,
    dst: *ggml.impl.c.ggml_tensor,
    expert_rows: ?[*]const i32,
    cur_a: i64,
    cne1: i64,
    scratch: ?[*]u8,
) bool {
    inline for (tiled_types) |t| {
        if (ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]).type == t) {
            const f = comptime formatOf(t);
            if (expert_rows) |er| {
                mulMatIdOneExpert(a, f, params, dst, cur_a, cne1, er, scratch.?);
            } else {
                mulMatDriver(a, f, params, dst);
            }
            return true;
        }
    }
    return false;
}

/// Ports `ggml_tiled_min_batch` (tiled.cpp:1202 @b1b770d90): profitable from
/// 8 rows, or always when forced.
fn minBatch(rows: i64) bool {
    return rows >= 8 or matmulForced();
}

/// Ports `ggml_compute_forward_mul_mat_tiled` (tiled.cpp:1209 @b1b770d90).
///
/// Parameters:
/// - `params`: this thread's index, count and work buffer.
/// - `dst`: the `MUL_MAT` node.
///
/// Return: true when the whole op was computed here; false to fall through
/// to the `vec_dot` path, always so off x86.
pub export fn ggml_compute_forward_mul_mat_tiled(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) callconv(.c) bool {
    // --use-ref means bail out and go back to vec_dot reference impl
    if (params.use_ref) return false;
    if (!tiledSupported(ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]), ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[1]))) return false;
    if (!minBatch(ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[1]).ne[1])) return false;
    return typeDispatch(ggml.cpu.tiled_kernel.arm, params, dst, null, 0, 0, null);
}

/// Ports `ggml_compute_forward_mul_mat_id_tiled` (tiled.cpp:1227
/// @b1b770d90): one expert of `MUL_MAT_ID`, its eligibility decided here.
///
/// Parameters:
/// - `params`: this thread's index, count and work buffer.
/// - `dst`: the `MUL_MAT_ID` node.
/// - `cur_a`: the expert.
/// - `cne1`: how many rows were routed to it.
/// - `expert_rows`: its `(slot, token)` pairs in the routing table.
/// - `scratch`: the per-thread workspaces `ggml_tiled_wdata_size` reserved.
///
/// Return: true when the expert was computed here.
pub export fn ggml_compute_forward_mul_mat_id_tiled(
    params: *const ggml.cpu.defs.ComputeParams,
    dst: *ggml.impl.c.ggml_tensor,
    cur_a: i64,
    cne1: i64,
    expert_rows: [*]const i32,
    scratch: [*]u8,
) callconv(.c) bool {
    if (params.use_ref) return false;
    if (!tiledSupported(ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]), ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[1]))) return false;
    // profitability is per expert: the rows routed to this expert
    if (!minBatch(cne1)) return false;
    return typeDispatch(ggml.cpu.tiled_kernel.arm, params, dst, expert_rows, cur_a, cne1, scratch);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "off x86 the tiled path declines and reserves nothing" {
    if (comptime ggml.cpu.tiled_kernel.arm != .scalar) return error.SkipZigTest;
    var src0 = std.mem.zeroes(ggml.impl.c.ggml_tensor);
    var src1 = std.mem.zeroes(ggml.impl.c.ggml_tensor);
    src0.type = ggml.impl.c.GGML_TYPE_Q4_K;
    src1.type = ggml.impl.c.GGML_TYPE_F32;
    var dst = std.mem.zeroes(ggml.impl.c.ggml_tensor);
    dst.src[0] = &src0;
    dst.src[1] = &src1;
    try std.testing.expectEqual(@as(usize, 0), ggml_tiled_wdata_size(4, &dst));
}

test "narrowKExtent divides ne00 and fits the budget" {
    try std.testing.expectEqual(@as(c_int, 768), narrowKExtent(1, 768));
    try std.testing.expectEqual(@as(c_int, 512), narrowKExtent(16, 1024));
    try std.testing.expectEqual(@as(c_int, 256), narrowKExtent(16, 256 * 7));
}

/// Random bytes as one block of `t`, with a modest finite fp16 scale.
fn randomBlock(t: ggml.impl.c.enum_ggml_type, block: []u8, rand: std.Random) void {
    rand.bytes(block);
    const scale: u16 = 0x3000 | (rand.int(u16) & 0x03ff);
    switch (t) {
        ggml.impl.c.GGML_TYPE_IQ1_M => {
            const sc: *align(1) [4]u16 = @ptrCast(block[block.len - 8 ..]);
            for (0..4) |w| sc[w] = (sc[w] & 0x0fff) | (@as(u16, @truncate(scale >> @intCast(4 * w))) << 12);
        },
        ggml.impl.c.GGML_TYPE_Q2_K, ggml.impl.c.GGML_TYPE_Q3_K, ggml.impl.c.GGML_TYPE_Q6_K => {
            // `d` (and Q2_K's `dmin`) sit at the end of the block.
            const tail = if (t == ggml.impl.c.GGML_TYPE_Q2_K) block.len - 4 else block.len - 2;
            std.mem.writeInt(u16, block[tail..][0..2], scale, .little);
            if (t == ggml.impl.c.GGML_TYPE_Q2_K) std.mem.writeInt(u16, block[tail + 2 ..][0..2], scale ^ 0x0100, .little);
        },
        ggml.impl.c.GGML_TYPE_Q4_K, ggml.impl.c.GGML_TYPE_Q5_K => {
            std.mem.writeInt(u16, block[0..2], scale, .little);
            std.mem.writeInt(u16, block[2..4], scale ^ 0x0100, .little);
        },
        else => std.mem.writeInt(u16, block[0..2], scale, .little),
    }
}

fn testTensor(t: ggml.impl.c.enum_ggml_type, ne: [4]i64, data: *anyopaque) ggml.impl.c.ggml_tensor {
    var x = std.mem.zeroes(ggml.impl.c.ggml_tensor);
    x.type = t;
    x.ne = ne;
    x.nb[0] = ggml.types.ggml_type_size(t);
    x.nb[1] = ggml.types.ggml_row_size(t, ne[0]);
    x.nb[2] = x.nb[1] * @as(usize, @intCast(ne[1]));
    x.nb[3] = x.nb[2] * @as(usize, @intCast(ne[2]));
    x.data = data;
    return x;
}

fn vecDotRow(t: ggml.impl.c.enum_ggml_type, n: i64, x: *const anyopaque, y: *const anyopaque) f32 {
    var out: f32 = undefined;
    ggml.cpu.traits.table[@intCast(t)].vec_dot.?(@intCast(n), &out, 0, x, 0, y, 0, 1);
    return out;
}

/// Runs `MUL_MAT` through the tiled driver on arm `a` and holds every
/// element to `vec_dot` on the same `q8_K` rows. Different summation
/// orders, so a tolerance; a wrong row, column, slab or tail is off by the
/// size of the answer.
fn checkMulMat(comptime a: ggml.cpu.tiled_kernel.Arm, comptime t: ggml.impl.c.enum_ggml_type, ne00: i64, ne01: i64, ne11: i64, rand: std.Random) !void {
    const alloc = std.testing.allocator;
    const nblk: usize = @intCast(@divExact(ne00, ggml.impl.c.QK_K));
    const bsize = ggml.types.ggml_type_size(t);
    const row_bytes = bsize * nblk;

    const w0 = try alloc.alignedAlloc(u8, .@"64", row_bytes * @as(usize, @intCast(ne01)));
    defer alloc.free(w0);
    for (0..@as(usize, @intCast(ne01)) * nblk) |b| randomBlock(t, w0[b * bsize ..][0..bsize], rand);

    const x1 = try alloc.alloc(f32, @intCast(ne00 * ne11));
    defer alloc.free(x1);
    for (x1) |*v| v.* = rand.float(f32) * 2 - 1;

    var src0 = testTensor(t, .{ ne00, ne01, 1, 1 }, w0.ptr);
    var src1 = testTensor(ggml.impl.c.GGML_TYPE_F32, .{ ne00, ne11, 1, 1 }, x1.ptr);
    const out = try alloc.alloc(f32, @intCast(ne01 * ne11));
    defer alloc.free(out);
    @memset(out, std.math.nan(f32));
    var dst = testTensor(ggml.impl.c.GGML_TYPE_F32, .{ ne01, ne11, 1, 1 }, out.ptr);
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    const nbw1 = ggml.types.ggml_row_size(ggml.impl.c.GGML_TYPE_Q8_K, ne00);
    const wsize = stdlib.num.pad(usize, nbw1 * @as(usize, @intCast(ne11)), 64) + 64 + ggml.cpu.tiled_kernel.ws_slot;
    const work = try alloc.alignedAlloc(u8, .@"64", wsize);
    defer alloc.free(work);

    var tp: ggml.cpu.defs.Threadpool = undefined;
    tp.n_graph = .init(1);
    tp.current_chunk = .init(1);
    const params: ggml.cpu.defs.ComputeParams = .{
        .ith = 0,
        .nth = 1,
        .wsize = work.len,
        .wdata = work.ptr,
        .threadpool = &tp,
        .use_ref = false,
    };
    try std.testing.expect(typeDispatch(a, &params, &dst, null, 0, 0, null));

    var scale: f32 = 0;
    for (out) |v| scale = @max(scale, @abs(v));
    try std.testing.expect(scale > 0);
    for (0..@intCast(ne11)) |j| {
        for (0..@intCast(ne01)) |i| {
            const want = vecDotRow(t, ne00, w0.ptr + i * row_bytes, work.ptr + j * nbw1);
            const got = out[j * @as(usize, @intCast(ne01)) + i];
            try std.testing.expectApproxEqAbs(want, got, 2e-4 * scale);
        }
    }
}

test "the MUL_MAT driver agrees with vec_dot on every arm and format" {
    var prng = std.Random.DefaultPrng.init(0x711ed);
    const rand = prng.random();
    inline for (.{ ggml.cpu.tiled_kernel.Arm.scalar, ggml.cpu.tiled_kernel.Arm.avx2, ggml.cpu.tiled_kernel.Arm.vnni }) |a| {
        inline for (tiled_types) |t| {
            // standard path (32 > 16 columns, a ragged 16-row tail) and narrow
            try checkMulMat(a, t, 512, 40, 33, rand);
            try checkMulMat(a, t, 768, 24, 5, rand);
        }
    }
}

test "the MUL_MAT_ID expert path agrees with vec_dot" {
    var prng = std.Random.DefaultPrng.init(0x1d1d);
    const rand = prng.random();
    const alloc = std.testing.allocator;
    inline for (.{ ggml.cpu.tiled_kernel.Arm.scalar, ggml.cpu.tiled_kernel.Arm.avx2, ggml.cpu.tiled_kernel.Arm.vnni }) |a| {
        inline for (.{ ggml.impl.c.GGML_TYPE_Q4_K, ggml.impl.c.GGML_TYPE_Q6_K, ggml.impl.c.GGML_TYPE_IQ4_XS, ggml.impl.c.GGML_TYPE_IQ2_XS }) |t| {
            const ne00: i64 = 512;
            const ne01: i64 = 80;
            const n_as: i64 = 2;
            const ne11: i64 = 2;
            const ne12: i64 = 13;
            const nblk: usize = 2;
            const bsize = ggml.types.ggml_type_size(t);
            const row_bytes = bsize * nblk;

            const w0 = try alloc.alignedAlloc(u8, .@"64", row_bytes * @as(usize, @intCast(ne01 * n_as)));
            defer alloc.free(w0);
            for (0..@as(usize, @intCast(ne01 * n_as)) * nblk) |b| randomBlock(t, w0[b * bsize ..][0..bsize], rand);

            var dummy: f32 = 0;
            var src0 = testTensor(t, .{ ne00, ne01, n_as, 1 }, w0.ptr);
            var src1 = testTensor(ggml.impl.c.GGML_TYPE_F32, .{ ne00, ne11, ne12, 1 }, &dummy);
            const out = try alloc.alloc(f32, @intCast(ne01 * 2 * ne12));
            defer alloc.free(out);
            @memset(out, std.math.nan(f32));
            var dst = testTensor(ggml.impl.c.GGML_TYPE_F32, .{ ne01, 2, ne12, 1 }, out.ptr);
            dst.src[0] = &src0;
            dst.src[1] = &src1;

            const nbw1 = ggml.types.ggml_row_size(ggml.impl.c.GGML_TYPE_Q8_K, ne00);
            const n_q8: usize = @intCast(ne11 * ne12);
            const scratch_off = stdlib.num.pad(usize, n_q8 * nbw1, 64);
            const work = try alloc.alignedAlloc(u8, .@"64", scratch_off + ggml.cpu.tiled_kernel.ws_slot);
            defer alloc.free(work);
            var row: [512]f32 = undefined;
            for (0..n_q8) |i| {
                for (&row) |*v| v.* = rand.float(f32) * 2 - 1;
                ggml.cpu.traits.table[ggml.impl.c.GGML_TYPE_Q8_K].from_float.?(&row, work.ptr + i * nbw1, ne00);
            }

            // ten routed rows (narrow), then all 26 (standard)
            const routed = [_][2]i32{ .{ 0, 0 }, .{ 1, 1 }, .{ 0, 2 }, .{ 1, 3 }, .{ 1, 4 }, .{ 0, 5 }, .{ 1, 6 }, .{ 0, 1 }, .{ 0, 3 }, .{ 1, 5 } };
            var all: [26][2]i32 = undefined;
            for (0..26) |k| all[k] = .{ @intCast(k % 2), @intCast(k / 2) };

            inline for (.{ routed[0..], all[0..] }) |rs| {
                @memset(out, std.math.nan(f32));
                const params: ggml.cpu.defs.ComputeParams = .{
                    .ith = 0,
                    .nth = 1,
                    .wsize = work.len,
                    .wdata = work.ptr,
                    .threadpool = null,
                    .use_ref = false,
                };
                try std.testing.expect(typeDispatch(a, &params, &dst, @ptrCast(rs.ptr), 1, rs.len, work.ptr + scratch_off));

                var scale: f32 = 0;
                for (out) |v| if (!std.math.isNan(v)) {
                    scale = @max(scale, @abs(v));
                };
                try std.testing.expect(scale > 0);
                for (rs) |rm| {
                    const j1: usize = @intCast(rm[0]);
                    const j2: usize = @intCast(rm[1]);
                    const q8 = work.ptr + (j1 % 2 + j2 * 2) * nbw1;
                    for (0..@intCast(ne01)) |j01| {
                        const want = vecDotRow(t, ne00, w0.ptr + (@as(usize, @intCast(ne01)) + j01) * row_bytes, q8);
                        const got = out[(j2 * 2 + j1) * @as(usize, @intCast(ne01)) + j01];
                        try std.testing.expectApproxEqAbs(want, got, 2e-4 * scale);
                    }
                }
            }
        }
    }
}
