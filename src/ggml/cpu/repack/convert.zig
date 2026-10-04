//! The `make_block_*` interleavers and the `repack_*_to_*_bl` drivers that
//! walk a tensor through them.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration names the C function and the line it
//! began at.
//!
//! # What is here and what is not
//!
//! Eleven converters, one per live `tensor_traits` block type. The five
//! `*_16_bl` drivers and the four `make_block_*x16` interleavers next to
//! them in the C are reachable only from the `<…, 1, 16>` specialisations
//! of `repack` (ggml-cpu/repack.cpp:3938 @c1d0e7a00), and every one of
//! those is inside `#if defined __riscv_zvfh`. They are `static` and
//! unreferenced on this target, so they export nothing and there is no
//! contract to meet. `blocks.zig` still carries their layouts.
//!
//! # The interleave is a byte permutation, not arithmetic
//!
//! Every one of these moves `blck_size_interleave` bytes at a time from one
//! of `N` consecutive source blocks into a stride-`N` slot of the output,
//! so `node-diff` sees any error as a wrong `MUL_MAT` rather than a drifting
//! last bit. The one exception is `q4_0`, which also flips the sign bias of
//! every nibble — the `0x88…` mask — because the interleaved kernels read
//! unsigned nibbles where `block_q4_0` stores them biased by 8.

const std = @import("std");
const impl = @import("../../impl.zig");
const src_blocks = @import("../../quants/blocks.zig");
const blocks = @import("blocks.zig");

const c = impl.c;

const QK4_0 = 32;
const QK8_0 = 32;
const QK_K = 256;
const QK4_NL = 32;
const QK_MXFP4 = 32;

/// `memcpy` of a compile-time-unknown length between fixed arrays, which is
/// what every `memcpy(&out.qs[dst], &in[id].qs[src], blck_size_interleave)`
/// in the C is.
inline fn copy(dst: []u8, src: []const u8, n: usize) void {
    @memcpy(dst[0..n], src[0..n]);
}

/// The `uint64_t elems; memcpy(&elems, …); elems ^= xor_mask; memcpy(…,
/// &elems, …)` idiom the `q4_0` interleavers use.
///
/// Reading and writing little-endian makes this byte-for-byte what the C
/// does on this target; the mask repeats one byte, so the result would be
/// the same either way.
inline fn copyXor(comptime W: type, dst: []u8, src: []const u8, mask: W) void {
    const n = @sizeOf(W);
    const elems = std.mem.readInt(W, src[0..n], .little) ^ mask;
    std.mem.writeInt(W, dst[0..n], elems, .little);
}

// -----------------------------------------------------------------------------
// The interleavers

/// Ports `make_block_q8_0x4` (ggml-cpu/repack.cpp:2725 @c1d0e7a00).
fn makeBlockQ8_0x4(in: *const [4]src_blocks.Q8_0, blck_size_interleave: u32) blocks.block_q8_0x4 {
    var out: blocks.block_q8_0x4 = undefined;

    for (0..4) |i| out.d[i] = in[i].d;

    const end = QK8_0 * 4 / blck_size_interleave;
    for (0..end) |i| {
        const src_id = i % 4;
        const src_offset = (i / 4) * blck_size_interleave;
        const dst_offset = i * blck_size_interleave;
        copy(
            @ptrCast(out.qs[dst_offset..]),
            @ptrCast(in[src_id].qs[src_offset..]),
            blck_size_interleave,
        );
    }
    return out;
}

/// Ports `make_block_q4_0x4` (ggml-cpu/repack.cpp:2742 @c1d0e7a00).
fn makeBlockQ4_0x4(in: *const [4]src_blocks.Q4_0, blck_size_interleave: u32) blocks.block_q4_0x4 {
    var out: blocks.block_q4_0x4 = undefined;

    for (0..4) |i| out.d[i] = in[i].d;

    const end = QK4_0 * 2 / blck_size_interleave;

    if (blck_size_interleave == 8) {
        const xor_mask: u64 = 0x8888888888888888;
        for (0..end) |i| {
            const src_id = i % 4;
            const src_offset = (i / 4) * blck_size_interleave;
            const dst_offset = i * blck_size_interleave;
            copyXor(u64, @ptrCast(out.qs[dst_offset..]), in[src_id].qs[src_offset..], xor_mask);
        }
    } else if (blck_size_interleave == 4) {
        const xor_mask: u32 = 0x88888888;
        for (0..end) |i| {
            const src_id = i % 4;
            const src_offset = (i / 4) * blck_size_interleave;
            const dst_offset = i * blck_size_interleave;
            copyXor(u32, @ptrCast(out.qs[dst_offset..]), in[src_id].qs[src_offset..], xor_mask);
        }
    } else {
        impl.assert(false, "blck_size_interleave is 4 or 8");
    }

    return out;
}

/// Ports `make_block_q4_0x8` (ggml-cpu/repack.cpp:2787 @c1d0e7a00).
fn makeBlockQ4_0x8(in: *const [8]src_blocks.Q4_0, blck_size_interleave: u32) blocks.block_q4_0x8 {
    var out: blocks.block_q4_0x8 = undefined;

    for (0..8) |i| out.d[i] = in[i].d;

    const end = QK4_0 * 4 / blck_size_interleave;
    const xor_mask: u64 = 0x8888888888888888;

    for (0..end) |i| {
        const src_id = i % 8;
        const src_offset = (i / 8) * blck_size_interleave;
        const dst_offset = i * blck_size_interleave;
        copyXor(u64, @ptrCast(out.qs[dst_offset..]), in[src_id].qs[src_offset..], xor_mask);
    }

    return out;
}

/// The 12-byte scale/min repacking carried verbatim by both
/// `make_block_q4_Kx8` and `make_block_q5_Kx8`
/// (ggml-cpu/repack.cpp:2836, 3007 @c1d0e7a00).
///
/// Q4_K and Q5_K both hold 8 six-bit scales and 8 six-bit mins in 12 bytes.
/// The interleaved form groups them the other way round: each 12 bytes of
/// `out.scales` carries one sub-block's scale and min from all eight input
/// blocks.
fn packKScales(out_scales: *[96]u8, in_scales: *const [8][12]u8) void {
    var s: [8]u8 = undefined;
    var m: [8]u8 = undefined;

    for (0..4) |i| {
        for (0..8) |j| {
            s[j] = in_scales[j][i] & 63;
            m[j] = in_scales[j][i + 4] & 63;
        }
        packTwelve(out_scales[i * 12 ..][0..12], &s, &m);
    }

    for (0..4) |i| {
        for (0..8) |j| {
            s[j] = ((in_scales[j][i] & 192) >> 2) | (in_scales[j][i + 8] & 15);
            m[j] = ((in_scales[j][i + 4] & 192) >> 2) | ((in_scales[j][i + 8] & 240) >> 4);
        }
        packTwelve(out_scales[48 + i * 12 ..][0..12], &s, &m);
    }
}

/// One 12-byte group of `packKScales`: eight 6-bit scales and eight 6-bit
/// mins, low four bits of each in the first eight bytes and the high two in
/// the last four.
///
/// The C writes the twelve stores out longhand; the widest value either
/// expression can produce is `63 + (48 << 2) == 255`, so the `int`
/// arithmetic it does them in never loses a bit to the narrowing store.
inline fn packTwelve(dst: *[12]u8, s: *const [8]u8, m: *const [8]u8) void {
    for (0..4) |j| {
        dst[j] = (s[j] & 63) + ((s[j + 4] & 48) << 2);
        dst[j + 4] = (m[j] & 63) + ((m[j + 4] & 48) << 2);
        dst[j + 8] = (s[j + 4] & 15) + ((m[j + 4] & 15) << 4);
    }
}

/// Ports `make_block_q4_Kx8` (ggml-cpu/repack.cpp:2836 @c1d0e7a00).
fn makeBlockQ4_Kx8(in: *const [8]src_blocks.Q4_K, blck_size_interleave: u32) blocks.block_q4_Kx8 {
    var out: blocks.block_q4_Kx8 = undefined;

    for (0..8) |i| out.d[i] = in[i].d;
    for (0..8) |i| out.dmin[i] = in[i].dmin;

    const end = QK_K * 4 / blck_size_interleave;

    for (0..end) |i| {
        const src_id = i % 8;
        const src_offset = (i / 8) * blck_size_interleave;
        const dst_offset = i * blck_size_interleave;
        copy(out.qs[dst_offset..], in[src_id].qs[src_offset..], blck_size_interleave);
    }

    var in_scales: [8][12]u8 = undefined;
    for (0..8) |j| in_scales[j] = in[j].scales;
    packKScales(&out.scales, &in_scales);

    return out;
}

/// Ports `make_block_q2_Kx8` (ggml-cpu/repack.cpp:2965 @c1d0e7a00).
fn makeBlockQ2_Kx8(in: *const [8]src_blocks.Q2_K, blck_size_interleave: u32) blocks.block_q2_Kx8 {
    var out: blocks.block_q2_Kx8 = undefined;

    for (0..8) |i| out.d[i] = in[i].d;
    for (0..8) |i| out.dmin[i] = in[i].dmin;

    const end = QK_K * 2 / blck_size_interleave;

    for (0..end) |i| {
        const src_id = i % 8;
        const src_offset = (i / 8) * blck_size_interleave;
        const dst_offset = i * blck_size_interleave;
        copy(out.qs[dst_offset..], in[src_id].qs[src_offset..], @sizeOf(u64));
    }

    // Q2_K packs 16 four-bit scales and 16 four-bit mins into 16 bytes, so
    // unlike Q4_K and Q5_K no unpacking is needed -- only a shuffle.
    for (0..128) |i| {
        const src1 = (i % 16) / 2;
        const src2 = ((i / 16) * 2) + (i % 2);
        out.scales[i] = in[src1].scales[src2];
    }
    return out;
}

/// Ports `make_block_q5_Kx8` (ggml-cpu/repack.cpp:3007 @c1d0e7a00).
fn makeBlockQ5_Kx8(in: *const [8]src_blocks.Q5_K, blck_size_interleave: u32) blocks.block_q5_Kx8 {
    var out: blocks.block_q5_Kx8 = undefined;

    for (0..8) |i| out.d[i] = in[i].d;
    for (0..8) |i| out.dmin[i] = in[i].dmin;

    const end = QK_K * 4 / blck_size_interleave;

    for (0..end) |i| {
        const src_id = i % 8;
        const src_offset = (i / 8) * blck_size_interleave;
        const dst_offset = i * blck_size_interleave;
        copy(out.qs[dst_offset..], in[src_id].qs[src_offset..], blck_size_interleave);
    }

    // The high bits take the same chunk size, a quarter as many chunks:
    // Q5_K indexes them as `qh[qs_idx % 32] >> (qs_idx / 32)`.
    for (0..end / 4) |i| {
        const src_id = i % 8;
        const src_offset = (i / 8) * blck_size_interleave;
        const dst_offset = i * blck_size_interleave;
        copy(out.qh[dst_offset..], in[src_id].qh[src_offset..], blck_size_interleave);
    }

    var in_scales: [8][12]u8 = undefined;
    for (0..8) |j| in_scales[j] = in[j].scales;
    packKScales(&out.scales, &in_scales);

    return out;
}

/// Ports `make_block_q6_Kx8` (ggml-cpu/repack.cpp:3092 @c1d0e7a00).
fn makeBlockQ6_Kx8(in: *const [8]src_blocks.Q6_K, blck_size_interleave: u32) blocks.block_q6_Kx8 {
    var out: blocks.block_q6_Kx8 = undefined;
    const n_blocks = 8;

    for (0..n_blocks) |i| out.d[i] = in[i].d;

    const end_ls = QK_K * 4 / blck_size_interleave;
    for (0..end_ls) |i| {
        const src_id = i % n_blocks;
        const src_offset = (i / n_blocks) * blck_size_interleave;
        const dst_offset = i * blck_size_interleave;
        copy(out.ql[dst_offset..], in[src_id].ql[src_offset..], blck_size_interleave);
    }

    const end_hs = end_ls / 2;
    for (0..end_hs) |i| {
        const src_id = i % n_blocks;
        const src_offset = (i / n_blocks) * blck_size_interleave;
        const dst_offset = i * blck_size_interleave;
        copy(out.qh[dst_offset..], in[src_id].qh[src_offset..], blck_size_interleave);
    }

    // Q6_K's scales are already full `int8`, one per 16 elements, so they
    // interleave the same way the quants do.
    const n_scales = QK_K / 16;
    for (0..n_blocks) |i| {
        for (0..n_scales) |j| {
            out.scales[j * n_blocks + i] = in[i].scales[j];
        }
    }

    return out;
}

/// Ports `make_block_iq4_nlx4` (ggml-cpu/repack.cpp:3566 @c1d0e7a00).
fn makeBlockIq4NlX4(in: *const [4]src_blocks.IQ4_NL, blck_size_interleave: u32) blocks.block_iq4_nlx4 {
    var out: blocks.block_iq4_nlx4 = undefined;

    for (0..4) |i| out.d[i] = in[i].d;

    const end = QK4_NL * 2 / blck_size_interleave;

    // The C's `blck_size_interleave == 8` arm is commented out above this
    // one, with a `// TODO: this branch seems wrong`.
    if (blck_size_interleave == 4) {
        for (0..end) |i| {
            const src_id = i % 4;
            const src_offset = (i / 4) * blck_size_interleave;
            const dst_offset = i * blck_size_interleave;
            copy(out.qs[dst_offset..], in[src_id].qs[src_offset..], @sizeOf(u32));
        }
    } else {
        impl.assert(false, "blck_size_interleave is 4");
    }

    return out;
}

/// Ports `make_block_iq4_nlx8` (ggml-cpu/repack.cpp:3634 @c1d0e7a00).
fn makeBlockIq4NlX8(in: *const [8]src_blocks.IQ4_NL, blck_size_interleave: u32) blocks.block_iq4_nlx8 {
    var out: blocks.block_iq4_nlx8 = undefined;

    for (0..8) |i| out.d[i] = in[i].d;

    const end = QK4_NL * 4 / blck_size_interleave;

    if (blck_size_interleave == 8) {
        for (0..end) |i| {
            const src_id = i % 8;
            const src_offset = (i / 8) * blck_size_interleave;
            const dst_offset = i * blck_size_interleave;
            copy(out.qs[dst_offset..], in[src_id].qs[src_offset..], @sizeOf(u64));
        }
    } else {
        impl.assert(false, "blck_size_interleave is 8");
    }

    return out;
}

/// Ports `make_block_mxfp4x4` (ggml-cpu/repack.cpp:3748 @c1d0e7a00).
fn makeBlockMxfp4x4(in: *const [4]src_blocks.MXFP4, blck_size_interleave: u32) blocks.block_mxfp4x4 {
    var out: blocks.block_mxfp4x4 = undefined;

    for (0..4) |i| out.e[i] = in[i].e;

    const end = QK_MXFP4 * 2 / blck_size_interleave;

    if (blck_size_interleave == 4) {
        for (0..end) |i| {
            const src_id = i % 4;
            const src_offset = (i / 4) * blck_size_interleave;
            const dst_offset = i * blck_size_interleave;
            copy(out.qs[dst_offset..], in[src_id].qs[src_offset..], @sizeOf(u32));
        }
    } else {
        impl.assert(false, "blck_size_interleave is 4");
    }

    return out;
}

/// Ports `make_block_mxfp4x8` (ggml-cpu/repack.cpp:3805 @c1d0e7a00).
fn makeBlockMxfp4x8(in: *const [8]src_blocks.MXFP4, blck_size_interleave: u32) blocks.block_mxfp4x8 {
    var out: blocks.block_mxfp4x8 = undefined;

    for (0..8) |i| out.e[i] = in[i].e;

    const end = QK_MXFP4 * 4 / blck_size_interleave;

    if (blck_size_interleave == 8) {
        for (0..end) |i| {
            const src_id = i % 8;
            const src_offset = (i / 8) * blck_size_interleave;
            const dst_offset = i * blck_size_interleave;
            copy(out.qs[dst_offset..], in[src_id].qs[src_offset..], @sizeOf(u64));
        }
    } else {
        impl.assert(false, "blck_size_interleave is 8");
    }

    return out;
}

// -----------------------------------------------------------------------------
// The drivers

/// The shape all eleven `repack_*_to_*_bl` functions share.
///
/// Parameters:
/// - `Src`: the stored block type.
/// - `Dst`: the interleaved block type.
/// - `rows`: `nrows_interleaved` — how many rows one `Dst` holds.
/// - `qk`: elements per `Src` block.
/// - `require_ne0_mul8`: whether the guard includes `t->ne[0] % 8 != 0`.
///   `repack_iq4_nl_to_iq4_nl_8_bl` (ggml-cpu/repack.cpp:3658 @c1d0e7a00)
///   and `repack_mxfp4_to_mxfp4_8_bl` (ggml-cpu/repack.cpp:3829
///   @c1d0e7a00) are the two that omit it; the other nine have it.
/// - `make`: the interleaver.
///
/// Return: 0, or -1 when the tensor's shape cannot be interleaved — which
/// is how `ggml_repack_get_optimal_repack_type` learns a weight is not a
/// candidate, so it is a value and not an error.
fn Repacker(
    comptime Src: type,
    comptime Dst: type,
    comptime rows: usize,
    comptime qk: i64,
    comptime require_ne0_mul8: bool,
    comptime make: fn (*const [rows]Src, u32) Dst,
) type {
    return struct {
        fn run(t: *c.ggml_tensor, interleave_block: c_int, data: *const anyopaque, data_size: usize) callconv(.c) c_int {
            const dst_base: [*]Dst = @ptrCast(@alignCast(t.data.?));
            const src_base: [*]const Src = @ptrCast(@alignCast(data));
            var dst_tmp: [rows]Src = undefined;
            const nrow: usize = @intCast(c.ggml_nrows(t));
            const nblocks: usize = @intCast(@divTrunc(t.ne[0], qk));

            impl.assert(data_size == nrow * nblocks * @sizeOf(Src), "data_size matches the tensor");

            if (@rem(t.ne[1], @as(i64, rows)) != 0) return -1;
            if (require_ne0_mul8 and @rem(t.ne[0], 8) != 0) return -1;

            var dst = dst_base;
            var src = src_base;
            var b: usize = 0;
            while (b < nrow) : (b += rows) {
                for (0..nblocks) |x| {
                    for (0..rows) |i| {
                        dst_tmp[i] = src[x + i * nblocks];
                    }
                    dst[0] = make(&dst_tmp, @intCast(interleave_block));
                    dst += 1;
                }
                src += rows * nblocks;
            }
            return 0;
        }
    };
}

/// Ports `repack_q4_0_to_q4_0_4_bl` (ggml-cpu/repack.cpp:3200 @c1d0e7a00).
pub const repackQ4_0toQ4_0_4 = Repacker(src_blocks.Q4_0, blocks.block_q4_0x4, 4, QK4_0, true, makeBlockQ4_0x4).run;

/// Ports `repack_q4_0_to_q4_0_8_bl` (ggml-cpu/repack.cpp:3449 @c1d0e7a00).
pub const repackQ4_0toQ4_0_8 = Repacker(src_blocks.Q4_0, blocks.block_q4_0x8, 8, QK4_0, true, makeBlockQ4_0x8).run;

/// Ports `repack_q4_K_to_q4_K_8_bl` (ggml-cpu/repack.cpp:3231 @c1d0e7a00).
pub const repackQ4_KtoQ4_K_8 = Repacker(src_blocks.Q4_K, blocks.block_q4_Kx8, 8, QK_K, true, makeBlockQ4_Kx8).run;

/// Ports `repack_q2_K_to_q2_K_8_bl` (ggml-cpu/repack.cpp:3292 @c1d0e7a00).
pub const repackQ2_KtoQ2_K_8 = Repacker(src_blocks.Q2_K, blocks.block_q2_Kx8, 8, QK_K, true, makeBlockQ2_Kx8).run;

/// Ports `repack_q5_K_to_q5_K_8_bl` (ggml-cpu/repack.cpp:3388 @c1d0e7a00).
pub const repackQ5_KtoQ5_K_8 = Repacker(src_blocks.Q5_K, blocks.block_q5_Kx8, 8, QK_K, true, makeBlockQ5_Kx8).run;

/// Ports `repack_q6_K_to_q6_K_8_bl` (ggml-cpu/repack.cpp:3420 @c1d0e7a00).
pub const repackQ6_KtoQ6_K_8 = Repacker(src_blocks.Q6_K, blocks.block_q6_Kx8, 8, QK_K, true, makeBlockQ6_Kx8).run;

/// Ports `repack_q8_0_to_q8_0_4_bl` (ggml-cpu/repack.cpp:3480 @c1d0e7a00).
pub const repackQ8_0toQ8_0_4 = Repacker(src_blocks.Q8_0, blocks.block_q8_0x4, 4, QK8_0, true, makeBlockQ8_0x4).run;

/// Ports `repack_iq4_nl_to_iq4_nl_4_bl` (ggml-cpu/repack.cpp:3601 @c1d0e7a00).
pub const repackIq4NltoIq4Nl_4 = Repacker(src_blocks.IQ4_NL, blocks.block_iq4_nlx4, 4, QK4_NL, true, makeBlockIq4NlX4).run;

/// Ports `repack_iq4_nl_to_iq4_nl_8_bl` (ggml-cpu/repack.cpp:3658 @c1d0e7a00).
pub const repackIq4NltoIq4Nl_8 = Repacker(src_blocks.IQ4_NL, blocks.block_iq4_nlx8, 8, QK4_NL, false, makeBlockIq4NlX8).run;

/// Ports `repack_mxfp4_to_mxfp4_4_bl` (ggml-cpu/repack.cpp:3772 @c1d0e7a00).
pub const repackMxfp4toMxfp4_4 = Repacker(src_blocks.MXFP4, blocks.block_mxfp4x4, 4, QK_MXFP4, true, makeBlockMxfp4x4).run;

/// Ports `repack_mxfp4_to_mxfp4_8_bl` (ggml-cpu/repack.cpp:3829 @c1d0e7a00).
pub const repackMxfp4toMxfp4_8 = Repacker(src_blocks.MXFP4, blocks.block_mxfp4x8, 8, QK_MXFP4, false, makeBlockMxfp4x8).run;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "packTwelve never overflows a byte" {
    // `63 + (48 << 2) == 255` is the widest either expression reaches, so
    // the C's `int` arithmetic loses nothing to its narrowing store and the
    // `u8` arithmetic here cannot trap.
    var s: [8]u8 = .{255} ** 8;
    var m: [8]u8 = .{255} ** 8;
    var dst: [12]u8 = undefined;
    packTwelve(&dst, &s, &m);
    for (dst[0..8]) |v| try std.testing.expectEqual(@as(u8, 255), v);
    for (dst[8..12]) |v| try std.testing.expectEqual(@as(u8, 255), v);

    s = .{0} ** 8;
    m = .{0} ** 8;
    packTwelve(&dst, &s, &m);
    for (dst) |v| try std.testing.expectEqual(@as(u8, 0), v);
}

test "the q4_0 interleave flips every nibble's sign bias" {
    var in: [4]src_blocks.Q4_0 = undefined;
    for (&in, 0..) |*b, i| {
        b.d = @intCast(i);
        @memset(&b.qs, 0x00);
    }
    const out = makeBlockQ4_0x4(&in, 8);
    // 0x00 ^ 0x88 == 0x88: both nibbles move from -8 to 0.
    for (out.qs) |v| try std.testing.expectEqual(@as(i8, @bitCast(@as(u8, 0x88))), v);
}
