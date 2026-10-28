//! The MP3 decoder miniaudio's `ma_decoder` tries third: dr_mp3, as
//! miniaudio embeds it (minimp3 underneath).
//!
//! # Provenance
//!
//! Ports `ma_dr_mp3` and its `ma_dr_mp3dec` core from
//! `llama.cpp/vendor/miniaudio/miniaudio.h` (miniaudio v0.11.25, dr_mp3
//! embedded) at v0.4.0 (`5266f24da`). Each declaration names the function it
//! ports and the line it began at.
//!
//! # As `mtmd-helper.cpp` compiles it
//!
//! - **SIMD on aarch64 and x86_64 alike** (`MA_DR_MP3_ONLY_SIMD`): the
//!   antialias butterflies, the 36-point IMDCT's first eight lanes, the
//!   DCT-II and the synthesis window run four lanes at a time with separate
//!   multiplies and adds -- `vaddq_f32(vmulq_f32(...))` does not contract --
//!   and the scalar tails contract as clang contracts any `a*b + c`. The
//!   synthesis rounds to `int16` the NEON way (`vcvtq_s32_f32` of `x + 0.5`,
//!   minus one below zero, saturated) on aarch64 and the SSE way
//!   (`_mm_cvtps_epi32`, round to nearest even, saturated) on x86.
//! - **`int16` output** (`MA_DR_MP3_FLOAT_OUTPUT` is not defined): the
//!   decoder's `f32` read is its `s16` samples times 2^-15.
//! - **Layers I and II included** (`MA_DR_MP3_ONLY_MP3` is not defined).
//!
//! The decoder state is laid out as the C's (`extern struct`), because
//! corrupt data makes the Huffman reader run past the main-data buffer
//! into the fields after it. Reading past the buffer is recorded in
//! `read_indeterminate` rather than reproduced: the bytes there include
//! a pointer.

const std = @import("std");
const builtin = @import("builtin");
const Io = @import("audio_io.zig").Io;

/// Set when a decode read memory whose contents the C leaves to the
/// implementation: past the main-data buffer, past the input in the
/// Xing/LAME tag parse, or a callback buffer's unfilled tail.
pub threadlocal var read_indeterminate: bool = false;

/// `MA_DR_MP3_HAVE_SSE`: x86 rounds the synthesis through SSE, everything
/// else through NEON.
const have_sse = builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .x86;

const max_bitreservoir_bytes = 511;
const max_free_format_frame_size = 2304;
const max_l3_frame_payload_bytes = max_free_format_frame_size;
const max_frame_sync_matches = 10;
const hdr_size = 4;
const short_block_type = 2;
const stop_block_type = 3;
const mode_mono = 3;
const mode_joint_stereo = 1;
const bits_dequantizer_out = -1;
const max_scf = 255 + bits_dequantizer_out * 4 - 210;
const max_scfi = (max_scf + 3) & ~@as(c_int, 3);
const max_pcm_frames_per_mp3_frame = 1152;
const max_samples_per_frame = max_pcm_frames_per_mp3_frame * 2;
const min_data_chunk_size = 16384;
const data_chunk_size = min_data_chunk_size * 4;

// header accessors, from `MA_DR_MP3_HDR_IS_MONO` (miniaudio.h:92733 @5266f24da)
fn hdrIsMono(h: [*]const u8) bool {
    return (h[3] & 0xC0) == 0xC0;
}
fn hdrIsMsStereo(h: [*]const u8) bool {
    return (h[3] & 0xE0) == 0x60;
}
fn hdrIsFreeFormat(h: [*]const u8) bool {
    return (h[2] & 0xF0) == 0;
}
fn hdrIsCrc(h: [*]const u8) bool {
    return (h[1] & 1) == 0;
}
fn hdrTestPadding(h: [*]const u8) bool {
    return (h[2] & 0x2) != 0;
}
fn hdrTestMpeg1(h: [*]const u8) bool {
    return (h[1] & 0x8) != 0;
}
fn hdrTestNotMpeg25(h: [*]const u8) bool {
    return (h[1] & 0x10) != 0;
}
fn hdrTestIStereo(h: [*]const u8) bool {
    return (h[3] & 0x10) != 0;
}
fn hdrTestMsStereo(h: [*]const u8) bool {
    return (h[3] & 0x20) != 0;
}
fn hdrGetStereoMode(h: [*]const u8) c_int {
    return (h[3] >> 6) & 3;
}
fn hdrGetStereoModeExt(h: [*]const u8) c_int {
    return (h[3] >> 4) & 3;
}
fn hdrGetLayer(h: [*]const u8) c_int {
    return (h[1] >> 1) & 3;
}
fn hdrGetBitrate(h: [*]const u8) c_int {
    return h[2] >> 4;
}
fn hdrGetSampleRate(h: [*]const u8) c_int {
    return (h[2] >> 2) & 3;
}
fn hdrGetMySampleRate(h: [*]const u8) c_int {
    return hdrGetSampleRate(h) + (((h[1] >> 3) & 1) + ((h[1] >> 4) & 1)) * 3;
}
fn hdrIsFrame576(h: [*]const u8) bool {
    return (h[1] & 14) == 2;
}
fn hdrIsLayer1(h: [*]const u8) bool {
    return (h[1] & 6) == 6;
}

/// `ma_dr_mp3_bs`.
const Bs = extern struct {
    buf: [*]const u8,
    pos: c_int,
    limit: c_int,
};

/// `ma_dr_mp3_L3_gr_info`.
const GrInfo = extern struct {
    sfbtab: [*]const u8,
    part_23_length: u16,
    big_values: u16,
    scalefac_compress: u16,
    global_gain: u8,
    block_type: u8,
    mixed_block_flag: u8,
    n_long_sfb: u8,
    n_short_sfb: u8,
    table_select: [3]u8,
    region_count: [3]u8,
    subblock_gain: [3]u8,
    preflag: u8,
    scalefac_scale: u8,
    count1_table: u8,
    scfsi: u8,
};

/// `ma_dr_mp3dec_scratch`.
const Scratch = extern struct {
    bs: Bs,
    maindata: [max_bitreservoir_bytes + max_l3_frame_payload_bytes]u8,
    gr_info: [4]GrInfo,
    grbuf: [2][576]f32,
    scf: [40]f32,
    syn: [18 + 15][2 * 32]f32,
    ist_pos: [2][39]u8,
};

/// `ma_dr_mp3dec`.
const Dec = extern struct {
    mdct_overlap: [2][9 * 32]f32,
    qmf_state: [15 * 2 * 32]f32,
    reserv: c_int,
    free_format_bytes: c_int,
    header: [4]u8,
    reserv_buf: [511]u8,
    scratch: Scratch,
};

/// `ma_dr_mp3dec_frame_info`.
const FrameInfo = struct {
    frame_bytes: c_int = 0,
    channels: c_int = 0,
    sample_rate: c_int = 0,
    layer: c_int = 0,
    bitrate_kbps: c_int = 0,
};

/// `ma_dr_mp3_L12_scale_info`.
const L12ScaleInfo = struct {
    scf: [3 * 64]f32 = @splat(0),
    total_bands: u8 = 0,
    stereo_bands: u8 = 0,
    bitalloc: [64]u8 = @splat(0),
    scfcod: [64]u8 = @splat(0),
};

/// `ma_dr_mp3_L12_subband_alloc`.
const L12SubbandAlloc = struct { tab_offset: u8, code_tab_width: u8, band_count: u8 };

/// Ports `ma_dr_mp3_bs_init` (miniaudio.h:92901 @5266f24da).
fn bsInit(bs: *Bs, data: [*]const u8, bytes: c_int) void {
    bs.buf = data;
    bs.pos = 0;
    bs.limit = bytes *% 8;
}

/// Ports `ma_dr_mp3_bs_get_bits` (miniaudio.h:92907 @5266f24da).
fn bsGetBits(bs: *Bs, n: c_int) u32 {
    var cache: u32 = 0;
    const s: u32 = @as(u32, @bitCast(bs.pos)) & 7;
    var sh: c_int = n + @as(c_int, @intCast(s));
    var p = bs.buf + @as(usize, @intCast(bs.pos >> 3));
    bs.pos += n;
    if (bs.pos > bs.limit) return 0;
    var next: u32 = p[0] & (@as(u32, 255) >> @intCast(s));
    p += 1;
    while (true) {
        sh -= 8;
        if (sh <= 0) break;
        cache |= next << @intCast(sh);
        next = p[0];
        p += 1;
    }
    return cache | (next >> @intCast(-sh));
}

/// Ports `ma_dr_mp3_hdr_valid` (miniaudio.h:92922 @5266f24da).
fn hdrValid(h: [*]const u8) bool {
    return h[0] == 0xff and
        ((h[1] & 0xF0) == 0xf0 or (h[1] & 0xFE) == 0xe2) and
        hdrGetLayer(h) != 0 and
        hdrGetBitrate(h) != 15 and
        hdrGetSampleRate(h) != 3;
}

/// Ports `ma_dr_mp3_hdr_compare` (miniaudio.h:92930 @5266f24da).
fn hdrCompare(h1: [*]const u8, h2: [*]const u8) bool {
    return hdrValid(h2) and
        ((h1[1] ^ h2[1]) & 0xFE) == 0 and
        ((h1[2] ^ h2[2]) & 0x0C) == 0 and
        hdrIsFreeFormat(h1) == hdrIsFreeFormat(h2);
}

/// Ports `ma_dr_mp3_hdr_bitrate_kbps` (miniaudio.h:92937 @5266f24da).
fn hdrBitrateKbps(h: [*]const u8) u32 {
    const mpeg1: usize = @intFromBool(hdrTestMpeg1(h));
    const layer: usize = @intCast(hdrGetLayer(h) - 1);
    const br: usize = @intCast(hdrGetBitrate(h));
    return 2 * @as(u32, halfrate[mpeg1 * 45 + layer * 15 + br]);
}

/// Ports `ma_dr_mp3_hdr_sample_rate_hz` (miniaudio.h:92945 @5266f24da).
fn hdrSampleRateHz(h: [*]const u8) u32 {
    return g_hz[@intCast(hdrGetSampleRate(h))] >> @intFromBool(!hdrTestMpeg1(h)) >> @intFromBool(!hdrTestNotMpeg25(h));
}

/// Ports `ma_dr_mp3_hdr_frame_samples` (miniaudio.h:92950 @5266f24da).
fn hdrFrameSamples(h: [*]const u8) u32 {
    return if (hdrIsLayer1(h)) 384 else (@as(u32, 1152) >> @intFromBool(hdrIsFrame576(h)));
}

/// Ports `ma_dr_mp3_hdr_frame_bytes` (miniaudio.h:92954 @5266f24da).
fn hdrFrameBytes(h: [*]const u8, free_format_size: c_int) c_int {
    var frame_bytes: c_int = @bitCast(hdrFrameSamples(h) *% hdrBitrateKbps(h) *% 125 / hdrSampleRateHz(h));
    if (hdrIsLayer1(h)) frame_bytes &= ~@as(c_int, 3);
    return if (frame_bytes != 0) frame_bytes else free_format_size;
}

/// Ports `ma_dr_mp3_hdr_padding` (miniaudio.h:92963 @5266f24da).
fn hdrPadding(h: [*]const u8) c_int {
    return if (hdrTestPadding(h)) (if (hdrIsLayer1(h)) 4 else 1) else 0;
}

/// Ports `ma_dr_mp3_L12_subband_alloc_table` (miniaudio.h:92968 @5266f24da).
fn l12SubbandAllocTable(hdr: [*]const u8, sci: *L12ScaleInfo) []const L12SubbandAlloc {
    const g_alloc_L1 = [_]L12SubbandAlloc{.{ .tab_offset = 76, .code_tab_width = 4, .band_count = 32 }};
    const g_alloc_L2M2 = [_]L12SubbandAlloc{ .{ .tab_offset = 60, .code_tab_width = 4, .band_count = 4 }, .{ .tab_offset = 44, .code_tab_width = 3, .band_count = 7 }, .{ .tab_offset = 44, .code_tab_width = 2, .band_count = 19 } };
    const g_alloc_L2M1 = [_]L12SubbandAlloc{ .{ .tab_offset = 0, .code_tab_width = 4, .band_count = 3 }, .{ .tab_offset = 16, .code_tab_width = 4, .band_count = 8 }, .{ .tab_offset = 32, .code_tab_width = 3, .band_count = 12 }, .{ .tab_offset = 40, .code_tab_width = 2, .band_count = 7 } };
    const g_alloc_L2M1_lowrate = [_]L12SubbandAlloc{ .{ .tab_offset = 44, .code_tab_width = 4, .band_count = 2 }, .{ .tab_offset = 44, .code_tab_width = 3, .band_count = 10 } };
    const mode = hdrGetStereoMode(hdr);
    var nbands: c_int = undefined;
    const stereo_bands: c_int = if (mode == mode_mono) 0 else if (mode == mode_joint_stereo) (hdrGetStereoModeExt(hdr) << 2) + 4 else 32;
    var alloc: []const L12SubbandAlloc = undefined;
    if (hdrIsLayer1(hdr)) {
        alloc = &g_alloc_L1;
        nbands = 32;
    } else if (!hdrTestMpeg1(hdr)) {
        alloc = &g_alloc_L2M2;
        nbands = 30;
    } else {
        const sample_rate_idx = hdrGetSampleRate(hdr);
        var kbps = hdrBitrateKbps(hdr) >> @intFromBool(mode != mode_mono);
        if (kbps == 0) kbps = 192;
        alloc = &g_alloc_L2M1;
        nbands = 27;
        if (kbps < 56) {
            alloc = &g_alloc_L2M1_lowrate;
            nbands = if (sample_rate_idx == 2) 12 else 8;
        } else if (kbps >= 96 and sample_rate_idx != 1) {
            nbands = 30;
        }
    }
    sci.total_bands = @intCast(nbands);
    sci.stereo_bands = @intCast(@min(stereo_bands, nbands));
    return alloc;
}

/// `g_deq_L12` (miniaudio.h:93010 @5266f24da): computed in `float` at
/// compile time, as the C's initializer is.
const g_deq_L12: [54]f32 = blk: {
    const divs = [18]f32{ 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 3, 5, 9 };
    var t: [54]f32 = undefined;
    for (divs, 0..) |d, i| {
        t[i * 3 + 0] = @as(f32, 9.53674316e-07) / d;
        t[i * 3 + 1] = @as(f32, 7.56931807e-07) / d;
        t[i * 3 + 2] = @as(f32, 6.00777173e-07) / d;
    }
    break :blk t;
};

/// Ports `ma_dr_mp3_L12_read_scalefactors` (miniaudio.h:93008 @5266f24da).
fn l12ReadScalefactors(bs: *Bs, pba: []const u8, scfcod: []const u8, bands: usize, scf: []f32) void {
    var o: usize = 0;
    for (0..bands) |i| {
        var s: f32 = 0;
        const ba: c_int = pba[i];
        const mask: c_int = if (ba != 0) 4 + ((@as(c_int, 19) >> @intCast(scfcod[i])) & 3) else 0;
        var m: c_int = 4;
        while (m != 0) : (m >>= 1) {
            if (mask & m != 0) {
                const b: c_int = @intCast(bsGetBits(bs, 6));
                s = g_deq_L12[@intCast(ba * 3 - 6 + @rem(b, 3))] * @as(f32, @floatFromInt(@as(c_int, 1 << 21) >> @intCast(@divTrunc(b, 3))));
            }
            scf[o] = s;
            o += 1;
        }
    }
}

/// Ports `ma_dr_mp3_L12_read_scale_info` (miniaudio.h:93031 @5266f24da).
fn l12ReadScaleInfo(hdr: [*]const u8, bs: *Bs, sci: *L12ScaleInfo) void {
    const subband_alloc = l12SubbandAllocTable(hdr, sci);
    var sa: usize = 0;
    var k: usize = 0;
    var ba_bits: c_int = 0;
    var ba_code_tab: []const u8 = &g_bitalloc_code_tab;
    for (0..sci.total_bands) |i| {
        if (i == k) {
            k += subband_alloc[sa].band_count;
            ba_bits = subband_alloc[sa].code_tab_width;
            ba_code_tab = g_bitalloc_code_tab[subband_alloc[sa].tab_offset..];
            sa += 1;
        }
        var ba = ba_code_tab[bsGetBits(bs, ba_bits)];
        sci.bitalloc[2 * i] = ba;
        if (i < sci.stereo_bands) ba = ba_code_tab[bsGetBits(bs, ba_bits)];
        sci.bitalloc[2 * i + 1] = if (sci.stereo_bands != 0) ba else 0;
    }
    for (0..2 * @as(usize, sci.total_bands)) |i| {
        sci.scfcod[i] = if (sci.bitalloc[i] != 0) (if (hdrIsLayer1(hdr)) 2 else @intCast(bsGetBits(bs, 2))) else 6;
    }
    l12ReadScalefactors(bs, &sci.bitalloc, &sci.scfcod, @as(usize, sci.total_bands) * 2, &sci.scf);
    var i: usize = sci.stereo_bands;
    while (i < sci.total_bands) : (i += 1) sci.bitalloc[2 * i + 1] = 0;
}

/// Ports `ma_dr_mp3_L12_dequantize_granule` (miniaudio.h:93073 @5266f24da).
fn l12DequantizeGranule(grbuf: [*]f32, bs: *Bs, sci: *L12ScaleInfo, group_size: c_int) c_int {
    var choff: c_int = 576;
    const gs: usize = @intCast(group_size);
    for (0..4) |j| {
        var dst: [*]f32 = grbuf + gs * j;
        for (0..2 * @as(usize, sci.total_bands)) |i| {
            const ba: c_int = sci.bitalloc[i];
            if (ba != 0) {
                if (ba < 17) {
                    const half: c_int = (@as(c_int, 1) << @intCast(ba - 1)) - 1;
                    for (0..gs) |kk| dst[kk] = @floatFromInt(@as(c_int, @bitCast(bsGetBits(bs, ba))) - half);
                } else {
                    const mod: u32 = (@as(u32, 2) << @intCast(ba - 17)) + 1;
                    var code = bsGetBits(bs, @intCast(mod + 2 - (mod >> 3)));
                    for (0..gs) |kk| {
                        dst[kk] = @floatFromInt(@as(c_int, @bitCast(code % mod -% mod / 2)));
                        code /= mod;
                    }
                }
            }
            dst = if (choff >= 0) dst + @as(usize, @intCast(choff)) else dst - @as(usize, @intCast(-choff));
            choff = 18 - choff;
        }
    }
    return group_size * 4;
}

/// Ports `ma_dr_mp3_L12_apply_scf_384` (miniaudio.h:93107 @5266f24da).
fn l12ApplyScf384(sci: *L12ScaleInfo, scf_in: [*]const f32, dst_in: [*]f32) void {
    const sb: usize = sci.stereo_bands;
    const n = (@as(usize, sci.total_bands) - sb) * 18;
    @memcpy((dst_in + 576 + sb * 18)[0..n], (dst_in + sb * 18)[0..n]);
    var dst = dst_in;
    var scf = scf_in;
    for (0..sci.total_bands) |_| {
        for (0..12) |k| {
            dst[k + 0] *= scf[0];
            dst[k + 576] *= scf[3];
        }
        dst += 18;
        scf += 6;
    }
}

/// Ports `ma_dr_mp3_L3_read_side_info` (miniaudio.h:93121 @5266f24da).
fn l3ReadSideInfo(bs: *Bs, gr_in: [*]GrInfo, hdr: [*]const u8) c_int {
    var gr = gr_in;
    var tables: u32 = undefined;
    var scfsi: u32 = 0;
    var main_data_begin: c_int = undefined;
    var part_23_sum: c_int = 0;
    var gr_count: c_int = if (hdrIsMono(hdr)) 1 else 2;
    var sr_idx = hdrGetMySampleRate(hdr);
    sr_idx -= @intFromBool(sr_idx != 0);
    const sri: usize = @intCast(sr_idx);
    if (hdrTestMpeg1(hdr)) {
        gr_count *= 2;
        main_data_begin = @intCast(bsGetBits(bs, 9));
        scfsi = bsGetBits(bs, 7 + gr_count);
    } else {
        main_data_begin = @intCast(bsGetBits(bs, 8 + gr_count) >> @intCast(gr_count));
    }
    while (true) {
        if (hdrIsMono(hdr)) scfsi <<= 4;
        gr[0].part_23_length = @intCast(bsGetBits(bs, 12));
        part_23_sum += gr[0].part_23_length;
        gr[0].big_values = @intCast(bsGetBits(bs, 9));
        if (gr[0].big_values > 288) return -1;
        gr[0].global_gain = @intCast(bsGetBits(bs, 8));
        gr[0].scalefac_compress = @intCast(bsGetBits(bs, if (hdrTestMpeg1(hdr)) 4 else 9));
        gr[0].sfbtab = g_scf_long[sri * 23 ..].ptr;
        gr[0].n_long_sfb = 22;
        gr[0].n_short_sfb = 0;
        if (bsGetBits(bs, 1) != 0) {
            gr[0].block_type = @intCast(bsGetBits(bs, 2));
            if (gr[0].block_type == 0) return -1;
            gr[0].mixed_block_flag = @intCast(bsGetBits(bs, 1));
            gr[0].region_count[0] = 7;
            gr[0].region_count[1] = 255;
            if (gr[0].block_type == short_block_type) {
                scfsi &= 0x0F0F;
                if (gr[0].mixed_block_flag == 0) {
                    gr[0].region_count[0] = 8;
                    gr[0].sfbtab = g_scf_short[sri * 40 ..].ptr;
                    gr[0].n_long_sfb = 0;
                    gr[0].n_short_sfb = 39;
                } else {
                    gr[0].sfbtab = g_scf_mixed[sri * 40 ..].ptr;
                    gr[0].n_long_sfb = if (hdrTestMpeg1(hdr)) 8 else 6;
                    gr[0].n_short_sfb = 30;
                }
            }
            tables = bsGetBits(bs, 10);
            tables <<= 5;
            gr[0].subblock_gain[0] = @intCast(bsGetBits(bs, 3));
            gr[0].subblock_gain[1] = @intCast(bsGetBits(bs, 3));
            gr[0].subblock_gain[2] = @intCast(bsGetBits(bs, 3));
        } else {
            gr[0].block_type = 0;
            gr[0].mixed_block_flag = 0;
            tables = bsGetBits(bs, 15);
            gr[0].region_count[0] = @intCast(bsGetBits(bs, 4));
            gr[0].region_count[1] = @intCast(bsGetBits(bs, 3));
            gr[0].region_count[2] = 255;
        }
        gr[0].table_select[0] = @truncate(tables >> 10);
        gr[0].table_select[1] = @truncate((tables >> 5) & 31);
        gr[0].table_select[2] = @truncate(tables & 31);
        gr[0].preflag = if (hdrTestMpeg1(hdr)) @intCast(bsGetBits(bs, 1)) else @intFromBool(gr[0].scalefac_compress >= 500);
        gr[0].scalefac_scale = @intCast(bsGetBits(bs, 1));
        gr[0].count1_table = @intCast(bsGetBits(bs, 1));
        gr[0].scfsi = @truncate((scfsi >> 12) & 15);
        scfsi <<= 4;
        gr += 1;
        gr_count -= 1;
        if (gr_count == 0) break;
    }
    if (part_23_sum + bs.pos > bs.limit + main_data_begin * 8) return -1;
    return main_data_begin;
}

/// Ports `ma_dr_mp3_L3_read_scalefactors` (miniaudio.h:93240 @5266f24da).
fn l3ReadScalefactors(scf_in: [*]u8, ist_pos_in: [*]u8, scf_size: *const [4]u8, scf_count: [*]const u8, bitbuf: *Bs, scfsi_in: c_int) void {
    var scf = scf_in;
    var ist_pos = ist_pos_in;
    var scfsi = scfsi_in;
    var i: usize = 0;
    while (i < 4 and scf_count[i] != 0) : ({
        i += 1;
        scfsi *%= 2;
    }) {
        const cnt: usize = scf_count[i];
        if (scfsi & 8 != 0) {
            @memcpy(scf[0..cnt], ist_pos[0..cnt]);
        } else {
            const bits: c_int = scf_size[i];
            if (bits == 0) {
                @memset(scf[0..cnt], 0);
                @memset(ist_pos[0..cnt], 0);
            } else {
                const max_scf_v: c_int = if (scfsi < 0) (@as(c_int, 1) << @intCast(bits)) - 1 else -1;
                for (0..cnt) |k| {
                    const s: c_int = @intCast(bsGetBits(bitbuf, bits));
                    ist_pos[k] = if (s == max_scf_v) 0xff else @truncate(@as(c_uint, @bitCast(s)));
                    scf[k] = @truncate(@as(c_uint, @bitCast(s)));
                }
            }
        }
        ist_pos += cnt;
        scf += cnt;
    }
    scf[0] = 0;
    scf[1] = 0;
    scf[2] = 0;
}

/// Ports `ma_dr_mp3_L3_ldexp_q2` (miniaudio.h:93272 @5266f24da).
fn l3LdexpQ2(y_in: f32, exp_q2_in: c_int) f32 {
    var y = y_in;
    var exp_q2 = exp_q2_in;
    while (true) {
        const e = @min(30 * 4, exp_q2);
        y *= g_expfrac[@intCast(e & 3)] * @as(f32, @floatFromInt(@as(c_int, 1 << 30) >> @intCast(e >> 2)));
        exp_q2 -= e;
        if (exp_q2 <= 0) break;
    }
    return y;
}

/// Ports `ma_dr_mp3_L3_decode_scalefactors` (miniaudio.h:93287 @5266f24da).
fn l3DecodeScalefactors(hdr: [*]const u8, ist_pos: [*]u8, bs: *Bs, gr: *const GrInfo, scf: [*]f32, ch: c_int) void {
    var scf_partition: [*]const u8 = g_scf_partitions[(@as(usize, @intFromBool(gr.n_short_sfb != 0)) + @intFromBool(gr.n_long_sfb == 0)) * 28 ..].ptr;
    var scf_size: [4]u8 = undefined;
    var iscf: [40]u8 = undefined;
    const scf_shift: c_int = @as(c_int, gr.scalefac_scale) + 1;
    var scfsi: c_int = gr.scfsi;
    if (hdrTestMpeg1(hdr)) {
        const part: c_int = g_scfc_decode[gr.scalefac_compress];
        scf_size[0] = @intCast(part >> 2);
        scf_size[1] = scf_size[0];
        scf_size[2] = @intCast(part & 3);
        scf_size[3] = scf_size[2];
    } else {
        const ist: c_int = @intFromBool(hdrTestIStereo(hdr) and ch != 0);
        var sfc: c_int = @as(c_int, gr.scalefac_compress) >> @intCast(ist);
        var k: c_int = ist * 3 * 4;
        var modprod: c_int = undefined;
        while (sfc >= 0) : ({
            sfc -= modprod;
            k += 4;
        }) {
            modprod = 1;
            var i: c_int = 3;
            while (i >= 0) : (i -= 1) {
                scf_size[@intCast(i)] = @intCast(@rem(@divTrunc(sfc, modprod), g_mod[@intCast(k + i)]));
                modprod *= g_mod[@intCast(k + i)];
            }
        }
        scf_partition += @as(usize, @intCast(k));
        scfsi = -16;
    }
    l3ReadScalefactors(&iscf, ist_pos, &scf_size, scf_partition, bs, scfsi);
    if (gr.n_short_sfb != 0) {
        const sh: u3 = @intCast(3 - scf_shift);
        var i: usize = 0;
        while (i < gr.n_short_sfb) : (i += 3) {
            const nl: usize = gr.n_long_sfb;
            iscf[nl + i + 0] +%= gr.subblock_gain[0] << sh;
            iscf[nl + i + 1] +%= gr.subblock_gain[1] << sh;
            iscf[nl + i + 2] +%= gr.subblock_gain[2] << sh;
        }
    } else if (gr.preflag != 0) {
        const g_preamp_v = g_preamp;
        for (0..10) |i| iscf[11 + i] +%= g_preamp_v[i];
    }
    const gain_exp: c_int = @as(c_int, gr.global_gain) + bits_dequantizer_out * 4 - 210 - @as(c_int, if (hdrIsMsStereo(hdr)) 2 else 0);
    const gain = l3LdexpQ2(@floatFromInt(@as(c_int, 1) << @intCast(@divTrunc(max_scfi, 4))), max_scfi - gain_exp);
    for (0..@as(usize, gr.n_long_sfb) + gr.n_short_sfb) |i| {
        scf[i] = l3LdexpQ2(gain, @as(c_int, iscf[i]) << @intCast(scf_shift));
    }
}

/// Ports `ma_dr_mp3_L3_pow_43` (miniaudio.h:93352 @5266f24da): the
/// polynomial's two `a + b*c` contract.
fn l3Pow43(x_in: c_int) f32 {
    var x = x_in;
    var mult: c_int = 256;
    if (x < 129) return g_pow43[@intCast(16 + x)];
    if (x < 1024) {
        mult = 16;
        x <<= 3;
    }
    const sign: c_int = 2 * x & 64;
    const frac = @as(f32, @floatFromInt((x & 63) - sign)) / @as(f32, @floatFromInt((x & ~@as(c_int, 63)) + sign));
    const inner = @mulAdd(f32, frac, 2.0 / 9.0, 4.0 / 3.0);
    const poly = @mulAdd(f32, frac, inner, 1.0);
    return g_pow43[@intCast(16 + ((x + sign) >> 6))] * poly * @as(f32, @floatFromInt(mult));
}

/// Reads byte `i` of the main-data stream: the buffer, or past it, which
/// the C reads as the scratch fields that follow.
fn mainByte(d: *Dec, p: [*]const u8) u8 {
    const base = @intFromPtr(&d.scratch.maindata);
    const a = @intFromPtr(p);
    if (a >= base and a < base + d.scratch.maindata.len) return p[0];
    read_indeterminate = true;
    return 0;
}

/// Ports `ma_dr_mp3_L3_huffman` (miniaudio.h:93369 @5266f24da).
fn l3Huffman(d: *Dec, dst_in: [*]f32, bs: *Bs, gr_info: *const GrInfo, scf_in: [*]const f32, layer3gr_limit: c_int) void {
    var dst = dst_in;
    var scf = scf_in;
    var one: f32 = 0.0;
    var ireg: usize = 0;
    var big_val_cnt: c_int = gr_info.big_values;
    var sfb = gr_info.sfbtab;
    var bs_next_ptr = bs.buf + @as(usize, @intCast(@divTrunc(bs.pos, 8)));
    var bs_cache: u32 = ((((@as(u32, mainByte(d, bs_next_ptr)) * 256 + mainByte(d, bs_next_ptr + 1)) * 256 + mainByte(d, bs_next_ptr + 2)) * 256 + mainByte(d, bs_next_ptr + 3))) << @intCast(bs.pos & 7);
    var bs_sh: c_int = (bs.pos & 7) - 8;
    bs_next_ptr += 4;

    const H = struct {
        fn peek(cache: u32, n: c_int) u32 {
            return if (n == 0) 0 else cache >> @intCast(32 - n);
        }
    };

    while (big_val_cnt > 0) {
        const tab_num: usize = gr_info.table_select[ireg];
        var sfb_cnt: c_int = gr_info.region_count[ireg];
        ireg += 1;
        const codebook = tabs[@intCast(tabindex[tab_num])..];
        const linbits: c_int = g_linbits[tab_num];
        while (true) {
            const np: c_int = sfb[0] / 2;
            sfb += 1;
            var pairs_to_decode: c_int = @min(big_val_cnt, np);
            one = scf[0];
            scf += 1;
            while (true) {
                var w: c_int = 5;
                var leaf: c_int = codebook[H.peek(bs_cache, w)];
                while (leaf < 0) {
                    bs_cache = shl(bs_cache, w);
                    bs_sh += w;
                    w = leaf & 7;
                    leaf = codebook[@intCast(@as(c_int, @intCast(H.peek(bs_cache, w))) - (leaf >> 3))];
                }
                bs_cache = shl(bs_cache, leaf >> 8);
                bs_sh += leaf >> 8;
                for (0..2) |_| {
                    var lsb: c_int = leaf & 0x0F;
                    if (linbits != 0 and lsb == 15) {
                        lsb += @intCast(H.peek(bs_cache, linbits));
                        bs_cache = shl(bs_cache, linbits);
                        bs_sh += linbits;
                        while (bs_sh >= 0) {
                            bs_cache |= @as(u32, mainByte(d, bs_next_ptr)) << @intCast(bs_sh);
                            bs_next_ptr += 1;
                            bs_sh -= 8;
                        }
                        dst[0] = one * l3Pow43(lsb) * @as(f32, if (@as(i32, @bitCast(bs_cache)) < 0) -1 else 1);
                    } else {
                        dst[0] = g_pow43[@intCast(16 + lsb - 16 * @as(c_int, @intCast(bs_cache >> 31)))] * one;
                    }
                    const f: c_int = if (lsb != 0) 1 else 0;
                    bs_cache = shl(bs_cache, f);
                    bs_sh += f;
                    dst += 1;
                    leaf >>= 4;
                }
                while (bs_sh >= 0) {
                    bs_cache |= @as(u32, mainByte(d, bs_next_ptr)) << @intCast(bs_sh);
                    bs_next_ptr += 1;
                    bs_sh -= 8;
                }
                pairs_to_decode -%= 1;
                if (pairs_to_decode == 0) break;
            }
            big_val_cnt -= np;
            if (big_val_cnt <= 0) break;
            sfb_cnt -= 1;
            if (sfb_cnt < 0) break;
        }
    }

    var np: c_int = 1 - big_val_cnt;
    const bspos = struct {
        fn f(next: [*]const u8, buf: [*]const u8, sh: c_int) c_int {
            return @as(c_int, @intCast((@intFromPtr(next) - @intFromPtr(buf)) * 8)) - 24 + sh;
        }
    }.f;
    outer: while (true) : (dst += 4) {
        const codebook_count1: []const u8 = if (gr_info.count1_table != 0) &tab33 else &tab32;
        var leaf: c_int = codebook_count1[H.peek(bs_cache, 4)];
        if (leaf & 8 == 0) {
            leaf = codebook_count1[@intCast((leaf >> 3) + @as(c_int, @intCast(H.peek(bs_cache << 4, leaf & 3))))];
        }
        bs_cache = shl(bs_cache, leaf & 7);
        bs_sh += leaf & 7;
        if (bspos(bs_next_ptr, bs.buf, bs_sh) > layer3gr_limit) break;
        inline for (.{ 0, 1, 2, 3 }) |s| {
            if (s == 0 or s == 2) {
                np -= 1;
                if (np == 0) {
                    np = sfb[0] / 2;
                    sfb += 1;
                    if (np == 0) break :outer;
                    one = scf[0];
                    scf += 1;
                }
            }
            if (leaf & (128 >> s) != 0) {
                dst[s] = if (@as(i32, @bitCast(bs_cache)) < 0) -one else one;
                bs_cache = shl(bs_cache, 1);
                bs_sh += 1;
            }
        }
        while (bs_sh >= 0) {
            bs_cache |= @as(u32, mainByte(d, bs_next_ptr)) << @intCast(bs_sh);
            bs_next_ptr += 1;
            bs_sh -= 8;
        }
    }
    bs.pos = layer3gr_limit;
}

fn shl(x: u32, n: c_int) u32 {
    return if (n >= 32) 0 else x << @intCast(n);
}

/// Ports `ma_dr_mp3_L3_midside_stereo` (miniaudio.h:93498 @5266f24da):
/// adds and subtractions only, so the vector body and the tail agree.
fn l3MidsideStereo(left: [*]f32, n: usize) void {
    const right = left + 576;
    for (0..n) |i| {
        const a = left[i];
        const b = right[i];
        left[i] = a + b;
        right[i] = a - b;
    }
}

/// Ports `ma_dr_mp3_L3_intensity_stereo_band` (miniaudio.h:93526 @5266f24da).
fn l3IntensityStereoBand(left: [*]f32, n: usize, kl: f32, kr: f32) void {
    for (0..n) |i| {
        left[i + 576] = left[i] * kr;
        left[i] = left[i] * kl;
    }
}

/// Ports `ma_dr_mp3_L3_stereo_top_band` (miniaudio.h:93535 @5266f24da).
fn l3StereoTopBand(right_in: [*]const f32, sfb: [*]const u8, nbands: usize, max_band: *[3]c_int) void {
    var right = right_in;
    max_band.* = .{ -1, -1, -1 };
    for (0..nbands) |i| {
        var k: usize = 0;
        while (k < sfb[i]) : (k += 2) {
            if (right[k] != 0 or right[k + 1] != 0) {
                max_band[i % 3] = @intCast(i);
                break;
            }
        }
        right += sfb[i];
    }
}

/// Ports `ma_dr_mp3_L3_stereo_process` (miniaudio.h:93552 @5266f24da).
fn l3StereoProcess(left_in: [*]f32, ist_pos: [*]const u8, sfb: [*]const u8, hdr: [*]const u8, max_band: *const [3]c_int, mpeg2_sh: c_int) void {
    var left = left_in;
    const max_pos: u32 = if (hdrTestMpeg1(hdr)) 7 else 64;
    var i: usize = 0;
    while (sfb[i] != 0) : (i += 1) {
        const ipos: u32 = ist_pos[i];
        if (@as(c_int, @intCast(i)) > max_band[i % 3] and ipos < max_pos) {
            const s: f32 = if (hdrTestMsStereo(hdr)) 1.41421356 else 1;
            var kl: f32 = undefined;
            var kr: f32 = undefined;
            if (hdrTestMpeg1(hdr)) {
                kl = g_pan[2 * ipos];
                kr = g_pan[2 * ipos + 1];
            } else {
                kl = 1;
                kr = l3LdexpQ2(1, @intCast(((ipos + 1) >> 1) << @intCast(mpeg2_sh)));
                if (ipos & 1 != 0) {
                    kl = kr;
                    kr = 1;
                }
            }
            l3IntensityStereoBand(left, sfb[i], kl * s, kr * s);
        } else if (hdrTestMsStereo(hdr)) {
            l3MidsideStereo(left, sfb[i]);
        }
        left += sfb[i];
    }
}

/// Ports `ma_dr_mp3_L3_intensity_stereo` (miniaudio.h:93584 @5266f24da).
fn l3IntensityStereo(left: [*]f32, ist_pos: [*]u8, gr: [*]const GrInfo, hdr: [*]const u8) void {
    var max_band: [3]c_int = undefined;
    const n_sfb: c_int = @as(c_int, gr[0].n_long_sfb) + gr[0].n_short_sfb;
    const max_blocks: c_int = if (gr[0].n_short_sfb != 0) 3 else 1;
    l3StereoTopBand(left + 576, gr[0].sfbtab, @intCast(n_sfb), &max_band);
    if (gr[0].n_long_sfb != 0) {
        const m = @max(@max(max_band[0], max_band[1]), max_band[2]);
        max_band = .{ m, m, m };
    }
    var i: c_int = 0;
    while (i < max_blocks) : (i += 1) {
        const default_pos: c_int = if (hdrTestMpeg1(hdr)) 3 else 0;
        const itop = n_sfb - max_blocks + i;
        const prev = itop - max_blocks;
        ist_pos[@intCast(itop)] = @intCast(if (max_band[@intCast(i)] >= prev) default_pos else ist_pos[@intCast(prev)]);
    }
    l3StereoProcess(left, ist_pos, gr[0].sfbtab, hdr, &max_band, gr[1].scalefac_compress & 1);
}

/// Ports `ma_dr_mp3_L3_reorder` (miniaudio.h:93602 @5266f24da).
fn l3Reorder(grbuf: [*]f32, scratch: [*]f32, sfb_in: [*]const u8) void {
    var sfb = sfb_in;
    var src = grbuf;
    var dst: usize = 0;
    while (true) : (sfb += 3) {
        const len: usize = sfb[0];
        if (len == 0) break;
        for (0..len) |_| {
            scratch[dst] = src[0];
            scratch[dst + 1] = src[len];
            scratch[dst + 2] = src[2 * len];
            dst += 3;
            src += 1;
        }
        src += 2 * len;
    }
    @memcpy(grbuf[0..dst], scratch[0..dst]);
}

/// Ports `ma_dr_mp3_L3_antialias` (miniaudio.h:93617 @5266f24da): the
/// vector body covers all eight, with no contraction.
fn l3Antialias(grbuf_in: [*]f32, nbands_in: c_int) void {
    var grbuf = grbuf_in;
    var nbands = nbands_in;
    while (nbands > 0) : ({
        nbands -= 1;
        grbuf += 18;
    }) {
        for (0..8) |i| {
            const u = grbuf[18 + i];
            const d = grbuf[17 - i];
            const uc0 = u * g_aa[i];
            const dc1 = d * g_aa[8 + i];
            const uc1 = u * g_aa[8 + i];
            const dc0 = d * g_aa[i];
            grbuf[18 + i] = uc0 - dc1;
            grbuf[17 - i] = uc1 + dc0;
        }
    }
}

/// Ports `ma_dr_mp3_L3_dct3_9` (miniaudio.h:93650 @5266f24da), scalar and
/// contracted where the C has `a + b*c`.
fn l3Dct39(y: *[9]f32) void {
    var s0 = y[0];
    var s2 = y[2];
    var s4 = y[4];
    var s6 = y[6];
    var s8 = y[8];
    var t0 = @mulAdd(f32, s6, 0.5, s0);
    s0 -= s6;
    var t4 = (s4 + s2) * 0.93969262;
    var t2 = (s8 + s2) * 0.76604444;
    s6 = (s4 - s8) * 0.17364818;
    s4 += s8 - s2;
    s2 = @mulAdd(f32, -s4, 0.5, s0);
    y[4] = s4 + s0;
    s8 = t0 - t2 + s6;
    s0 = t0 - t4 + t2;
    s4 = t0 + t4 - s6;
    var s1 = y[1];
    var s3 = y[3];
    var s5 = y[5];
    var s7 = y[7];
    s3 *= 0.86602540;
    t0 = (s5 + s1) * 0.98480775;
    t4 = (s5 - s7) * 0.34202014;
    t2 = (s1 + s7) * 0.64278761;
    s1 = (s1 - s5 - s7) * 0.86602540;
    s5 = t0 - s3 - t2;
    s7 = t4 - s3 - t0;
    s3 = t4 + s3 - t2;
    y[0] = s4 - s7;
    y[1] = s2 + s1;
    y[2] = s0 - s3;
    y[3] = s8 + s5;
    y[5] = s8 - s5;
    y[6] = s0 + s3;
    y[7] = s2 - s1;
    y[8] = s4 + s7;
}

/// Ports `ma_dr_mp3_L3_imdct36` (miniaudio.h:93683 @5266f24da): lanes 0-7
/// through the vector body, unfused; lane 8 through the scalar tail,
/// contracted.
fn l3Imdct36(grbuf_in: [*]f32, overlap_in: [*]f32, window: [*]const f32, nbands: c_int) void {
    var grbuf = grbuf_in;
    var overlap = overlap_in;
    var j: c_int = 0;
    while (j < nbands) : ({
        j += 1;
        grbuf += 18;
        overlap += 9;
    }) {
        var co: [9]f32 = undefined;
        var si: [9]f32 = undefined;
        co[0] = -grbuf[0];
        si[0] = grbuf[17];
        for (0..4) |i| {
            si[8 - 2 * i] = grbuf[4 * i + 1] - grbuf[4 * i + 2];
            co[1 + 2 * i] = grbuf[4 * i + 1] + grbuf[4 * i + 2];
            si[7 - 2 * i] = grbuf[4 * i + 4] - grbuf[4 * i + 3];
            co[2 + 2 * i] = -(grbuf[4 * i + 3] + grbuf[4 * i + 4]);
        }
        l3Dct39(&co);
        l3Dct39(&si);
        si[1] = -si[1];
        si[3] = -si[3];
        si[5] = -si[5];
        si[7] = -si[7];
        for (0..8) |i| {
            const ovl = overlap[i];
            const sum = co[i] * g_twid9[9 + i] + si[i] * g_twid9[i];
            overlap[i] = co[i] * g_twid9[i] - si[i] * g_twid9[9 + i];
            grbuf[i] = ovl * window[i] - sum * window[9 + i];
            grbuf[17 - i] = ovl * window[9 + i] + sum * window[i];
        }
        const i: usize = 8;
        const ovl = overlap[i];
        const sum = @mulAdd(f32, co[i], g_twid9[9 + i], si[i] * g_twid9[i]);
        overlap[i] = @mulAdd(f32, co[i], g_twid9[i], -(si[i] * g_twid9[9 + i]));
        grbuf[i] = @mulAdd(f32, ovl, window[i], -(sum * window[9 + i]));
        grbuf[17 - i] = @mulAdd(f32, ovl, window[9 + i], sum * window[i]);
    }
}

/// Ports `ma_dr_mp3_L3_idct3` (miniaudio.h:93735 @5266f24da).
fn l3Idct3(x0: f32, x1: f32, x2: f32, dst: *[3]f32) void {
    const m1 = x1 * 0.86602540;
    const a1 = @mulAdd(f32, -x2, 0.5, x0);
    dst[1] = x0 + x2;
    dst[0] = a1 + m1;
    dst[2] = a1 - m1;
}

/// Ports `ma_dr_mp3_L3_imdct12` (miniaudio.h:93743 @5266f24da).
fn l3Imdct12(x: [*]const f32, dst: [*]f32, overlap: [*]f32) void {
    var co: [3]f32 = undefined;
    var si: [3]f32 = undefined;
    l3Idct3(-x[0], x[6] + x[3], x[12] + x[9], &co);
    l3Idct3(x[15], x[12] - x[9], x[6] - x[3], &si);
    si[1] = -si[1];
    for (0..3) |i| {
        const ovl = overlap[i];
        const sum = @mulAdd(f32, co[i], g_twid3[3 + i], si[i] * g_twid3[i]);
        overlap[i] = @mulAdd(f32, co[i], g_twid3[i], -(si[i] * g_twid3[3 + i]));
        dst[i] = @mulAdd(f32, ovl, g_twid3[2 - i], -(sum * g_twid3[5 - i]));
        dst[5 - i] = @mulAdd(f32, ovl, g_twid3[5 - i], sum * g_twid3[2 - i]);
    }
}

/// Ports `ma_dr_mp3_L3_imdct_short` (miniaudio.h:93760 @5266f24da).
fn l3ImdctShort(grbuf_in: [*]f32, overlap_in: [*]f32, nbands_in: c_int) void {
    var grbuf = grbuf_in;
    var overlap = overlap_in;
    var nbands = nbands_in;
    while (nbands > 0) : ({
        nbands -= 1;
        overlap += 9;
        grbuf += 18;
    }) {
        var tmp: [18]f32 = undefined;
        @memcpy(&tmp, grbuf[0..18]);
        @memcpy(grbuf[0..6], overlap[0..6]);
        l3Imdct12(&tmp, grbuf + 6, overlap + 6);
        l3Imdct12(tmp[1..].ptr, grbuf + 12, overlap + 6);
        l3Imdct12(tmp[2..].ptr, overlap, overlap + 6);
    }
}

/// Ports `ma_dr_mp3_L3_change_sign` (miniaudio.h:93772 @5266f24da).
fn l3ChangeSign(grbuf_in: [*]f32) void {
    var grbuf = grbuf_in + 18;
    var b: usize = 0;
    while (b < 32) : ({
        b += 2;
        grbuf += 36;
    }) {
        var i: usize = 1;
        while (i < 18) : (i += 2) grbuf[i] = -grbuf[i];
    }
}

/// Ports `ma_dr_mp3_L3_imdct_gr` (miniaudio.h:93779 @5266f24da).
fn l3ImdctGr(grbuf_in: [*]f32, overlap_in: [*]f32, block_type: u32, n_long_bands: u32) void {
    var grbuf = grbuf_in;
    var overlap = overlap_in;
    if (n_long_bands != 0) {
        l3Imdct36(grbuf, overlap, &g_mdct_window, @intCast(n_long_bands));
        grbuf += 18 * n_long_bands;
        overlap += 9 * n_long_bands;
    }
    if (block_type == short_block_type) {
        l3ImdctShort(grbuf, overlap, @intCast(32 - n_long_bands));
    } else {
        l3Imdct36(grbuf, overlap, g_mdct_window[if (block_type == stop_block_type) 18 else 0..].ptr, @intCast(32 - n_long_bands));
    }
}

/// Ports `ma_dr_mp3_L3_save_reservoir` (miniaudio.h:93796 @5266f24da).
fn l3SaveReservoir(h: *Dec, s: *Scratch) void {
    var pos: c_int = @intCast(@divTrunc(@as(u32, @bitCast(s.bs.pos)) +% 7, 8));
    var remains: c_int = @bitCast(@as(u32, @bitCast(s.bs.limit)) / 8 -% @as(u32, @bitCast(pos)));
    if (remains > max_bitreservoir_bytes) {
        pos += remains - max_bitreservoir_bytes;
        remains = max_bitreservoir_bytes;
    }
    if (remains > 0) {
        const r: usize = @intCast(remains);
        std.mem.copyForwards(u8, h.reserv_buf[0..r], s.maindata[@intCast(pos)..][0..r]);
    }
    h.reserv = remains;
}

/// Ports `ma_dr_mp3_L3_restore_reservoir` (miniaudio.h:93811 @5266f24da).
fn l3RestoreReservoir(h: *Dec, bs: *Bs, s: *Scratch, main_data_begin: c_int) bool {
    const frame_bytes: usize = @intCast(@divTrunc(bs.limit - bs.pos, 8));
    const bytes_have: usize = @intCast(@min(h.reserv, main_data_begin));
    @memcpy(s.maindata[0..bytes_have], h.reserv_buf[@intCast(@max(0, h.reserv - main_data_begin))..][0..bytes_have]);
    @memcpy(s.maindata[bytes_have..][0..frame_bytes], (bs.buf + @as(usize, @intCast(@divTrunc(bs.pos, 8))))[0..frame_bytes]);
    bsInit(&s.bs, &s.maindata, @intCast(bytes_have + frame_bytes));
    return h.reserv >= main_data_begin;
}

/// Ports `ma_dr_mp3_L3_decode` (miniaudio.h:93820 @5266f24da).
fn l3Decode(h: *Dec, s: *Scratch, gr_info_in: [*]GrInfo, nch: usize) void {
    for (0..nch) |ch| {
        const layer3gr_limit = s.bs.pos + gr_info_in[ch].part_23_length;
        l3DecodeScalefactors(&h.header, &s.ist_pos[ch], &s.bs, &gr_info_in[ch], &s.scf, @intCast(ch));
        l3Huffman(h, &s.grbuf[ch], &s.bs, &gr_info_in[ch], &s.scf, layer3gr_limit);
    }
    if (hdrTestIStereo(&h.header)) {
        l3IntensityStereo(&s.grbuf[0], &s.ist_pos[1], gr_info_in, &h.header);
    } else if (hdrIsMsStereo(&h.header)) {
        l3MidsideStereo(&s.grbuf[0], 576);
    }
    var gr_info = gr_info_in;
    for (0..nch) |ch| {
        var aa_bands: c_int = 31;
        const n_long_bands: c_int = @as(c_int, if (gr_info[0].mixed_block_flag != 0) 2 else 0) << @intFromBool(hdrGetMySampleRate(&h.header) == 2);
        if (gr_info[0].n_short_sfb != 0) {
            aa_bands = n_long_bands - 1;
            l3Reorder(s.grbuf[ch][@intCast(n_long_bands * 18)..].ptr, @ptrCast(&s.syn[0]), gr_info[0].sfbtab + gr_info[0].n_long_sfb);
        }
        l3Antialias(&s.grbuf[ch], aa_bands);
        l3ImdctGr(&s.grbuf[ch], &h.mdct_overlap[ch], gr_info[0].block_type, @intCast(n_long_bands));
        l3ChangeSign(&s.grbuf[ch]);
        gr_info += 1;
    }
}

/// Ports `ma_dr_mp3d_DCT_II` (miniaudio.h:93850 @5266f24da), the vector
/// body: every lane is one `k`, with no contraction, and the stores sum
/// `t[2][i] + (t[3][i] + t[3][i + 1])` where the scalar fallback would
/// not. A 2-lane store at the last block writes only `k` and `k + 1`.
fn dct2(grbuf: [*]f32, n: usize) void {
    var k: usize = 0;
    while (k < n) : (k += 4) {
        const lanes: usize = if (@as(isize, @intCast(k)) > @as(isize, @intCast(n)) - 3) 2 else 4;
        for (0..lanes) |lane| {
            const y = grbuf + k + lane;
            var t: [4][8]f32 = undefined;
            for (0..8) |i| {
                const x0 = y[i * 18];
                const x1 = y[(15 - i) * 18];
                const x2 = y[(16 + i) * 18];
                const x3 = y[(31 - i) * 18];
                const t0 = x0 + x3;
                const t1 = x1 + x2;
                const t2 = (x1 - x2) * g_sec[3 * i + 0];
                const t3 = (x0 - x3) * g_sec[3 * i + 1];
                t[0][i] = t0 + t1;
                t[1][i] = (t0 - t1) * g_sec[3 * i + 2];
                t[2][i] = t3 + t2;
                t[3][i] = (t3 - t2) * g_sec[3 * i + 2];
            }
            for (0..4) |r| {
                const x = &t[r];
                var x0 = x[0];
                var x1 = x[1];
                var x2 = x[2];
                var x3 = x[3];
                var x4 = x[4];
                var x5 = x[5];
                var x6 = x[6];
                var x7 = x[7];
                var xt = x0 - x7;
                x0 = x0 + x7;
                x7 = x1 - x6;
                x1 = x1 + x6;
                x6 = x2 - x5;
                x2 = x2 + x5;
                x5 = x3 - x4;
                x3 = x3 + x4;
                x4 = x0 - x3;
                x0 = x0 + x3;
                x3 = x1 - x2;
                x1 = x1 + x2;
                x[0] = x0 + x1;
                x[4] = (x0 - x1) * 0.70710677;
                x5 = x5 + x6;
                x6 = (x6 + x7) * 0.70710677;
                x7 = x7 + xt;
                x3 = (x3 + x4) * 0.70710677;
                x5 = x5 - x7 * 0.198912367;
                x7 = x7 + x5 * 0.382683432;
                x5 = x5 - x7 * 0.198912367;
                x0 = xt - x6;
                xt = xt + x6;
                x[1] = (xt + x7) * 0.50979561;
                x[2] = (x4 + x3) * 0.54119611;
                x[3] = (x0 - x5) * 0.60134488;
                x[5] = (x0 + x5) * 0.89997619;
                x[6] = (x4 - x3) * 1.30656302;
                x[7] = (xt - x7) * 2.56291556;
            }
            var yy = y;
            for (0..7) |i| {
                const s = t[3][i] + t[3][i + 1];
                yy[0 * 18] = t[0][i];
                yy[1 * 18] = t[2][i] + s;
                yy[2 * 18] = t[1][i] + t[1][i + 1];
                yy[3 * 18] = t[2][1 + i] + s;
                yy += 4 * 18;
            }
            yy[0 * 18] = t[0][7];
            yy[1 * 18] = t[2][7] + t[3][7];
            yy[2 * 18] = t[1][7];
            yy[3 * 18] = t[3][7];
        }
    }
}

/// Ports `ma_dr_mp3d_scale_pcm` (miniaudio.h:94002 @5266f24da), the
/// non-ARMv6 arm.
fn scalePcm(sample: f32) i16 {
    if (sample >= 32766.5) return 32767;
    if (sample <= -32767.5) return -32768;
    var s: i16 = @intFromFloat(sample + 0.5);
    s -= @intFromBool(s < 0);
    return s;
}

/// The vector synthesis's rounding of one lane to `int16`.
fn simdToS16(a: f32) i16 {
    if (have_sse) {
        // `_mm_cvtps_epi32` of the clamped value: round to nearest even
        const c = @max(@min(a, 32767.0), -32768.0);
        const r = if (@abs(c - @trunc(c)) == 0.5) 2 * @round(c / 2) else @round(c);
        return @intFromFloat(r);
    }
    // `vaddq_f32(a, 0.5)`, `vcvtq_s32_f32` (saturating, toward zero),
    // `vqaddq_s32` with -1 below zero, `vqmovn_s32`
    const b = a + 0.5;
    const t: i32 = if (std.math.isNan(b)) 0 else if (b >= 2147483647.0) std.math.maxInt(i32) else if (b <= -2147483648.0) std.math.minInt(i32) else @intFromFloat(b);
    const m: i32 = if (b < 0) -1 else 0;
    const q = std.math.clamp(@as(i64, t) + m, std.math.minInt(i32), std.math.maxInt(i32));
    return @intCast(std.math.clamp(q, -32768, 32767));
}

/// Ports `ma_dr_mp3d_synth_pair` (miniaudio.h:94024 @5266f24da): every
/// `a += x * c` contracts.
fn synthPair(pcm: [*]i16, nch: usize, z_in: [*]const f32) void {
    var z = z_in;
    var a: f32 = (z[14 * 64] - z[0]) * 29;
    a = @mulAdd(f32, z[1 * 64] + z[13 * 64], 213, a);
    a = @mulAdd(f32, z[12 * 64] - z[2 * 64], 459, a);
    a = @mulAdd(f32, z[3 * 64] + z[11 * 64], 2037, a);
    a = @mulAdd(f32, z[10 * 64] - z[4 * 64], 5153, a);
    a = @mulAdd(f32, z[5 * 64] + z[9 * 64], 6574, a);
    a = @mulAdd(f32, z[8 * 64] - z[6 * 64], 37489, a);
    a = @mulAdd(f32, z[7 * 64], 75038, a);
    pcm[0] = scalePcm(a);
    z += 2;
    a = z[14 * 64] * 104;
    a = @mulAdd(f32, z[12 * 64], 1567, a);
    a = @mulAdd(f32, z[10 * 64], 9727, a);
    a = @mulAdd(f32, z[8 * 64], 64019, a);
    a = @mulAdd(f32, z[6 * 64], -9975, a);
    a = @mulAdd(f32, z[4 * 64], -45, a);
    a = @mulAdd(f32, z[2 * 64], 146, a);
    a = @mulAdd(f32, z[0 * 64], -5, a);
    pcm[16 * nch] = scalePcm(a);
}

/// Ports `ma_dr_mp3d_synth` (miniaudio.h:94047 @5266f24da), the vector
/// body: four lanes, unfused.
fn synth(xl: [*]const f32, dstl: [*]i16, nch: usize, lins: [*]f32) void {
    const xr = xl + 576 * (nch - 1);
    const dstr = dstl + (nch - 1);
    const zlin = lins + 15 * 64;
    var w: usize = 0;
    zlin[4 * 15] = xl[18 * 16];
    zlin[4 * 15 + 1] = xr[18 * 16];
    zlin[4 * 15 + 2] = xl[0];
    zlin[4 * 15 + 3] = xr[0];
    zlin[4 * 31] = xl[1 + 18 * 16];
    zlin[4 * 31 + 1] = xr[1 + 18 * 16];
    zlin[4 * 31 + 2] = xl[1];
    zlin[4 * 31 + 3] = xr[1];
    synthPair(dstr, nch, lins + 4 * 15 + 1);
    synthPair(dstr + 32 * nch, nch, lins + 4 * 15 + 64 + 1);
    synthPair(dstl, nch, lins + 4 * 15);
    synthPair(dstl + 32 * nch, nch, lins + 4 * 15 + 64);
    var i: isize = 14;
    while (i >= 0) : (i -= 1) {
        const iu: usize = @intCast(i);
        zlin[4 * iu] = xl[18 * (31 - iu)];
        zlin[4 * iu + 1] = xr[18 * (31 - iu)];
        zlin[4 * iu + 2] = xl[1 + 18 * (31 - iu)];
        zlin[4 * iu + 3] = xr[1 + 18 * (31 - iu)];
        zlin[4 * iu + 64] = xl[1 + 18 * (1 + iu)];
        zlin[4 * iu + 64 + 1] = xr[1 + 18 * (1 + iu)];
        (zlin + 4 * iu - 64)[2] = xl[18 * (1 + iu)];
        (zlin + 4 * iu - 64)[3] = xr[18 * (1 + iu)];
        var a: [4]f32 = undefined;
        var b: [4]f32 = undefined;
        // V0(0) V2(1) V1(2) V2(3) V1(4) V2(5) V1(6) V2(7)
        inline for (.{ 0, 2, 1, 2, 1, 2, 1, 2 }, 0..) |kind, k| {
            const w0 = g_win[w];
            const w1 = g_win[w + 1];
            w += 2;
            const vz = zlin + 4 * iu - 64 * k;
            const vy = zlin + 4 * iu - 64 * (15 - k);
            for (0..4) |j| {
                const bt = vz[j] * w1 + vy[j] * w0;
                switch (kind) {
                    0 => {
                        b[j] = bt;
                        a[j] = vz[j] * w0 - vy[j] * w1;
                    },
                    1 => {
                        b[j] = b[j] + bt;
                        a[j] = a[j] + (vz[j] * w0 - vy[j] * w1);
                    },
                    else => {
                        b[j] = b[j] + bt;
                        a[j] = a[j] + (vy[j] * w1 - vz[j] * w0);
                    },
                }
            }
        }
        dstr[(15 - iu) * nch] = simdToS16(a[1]);
        dstr[(17 + iu) * nch] = simdToS16(b[1]);
        dstl[(15 - iu) * nch] = simdToS16(a[0]);
        dstl[(17 + iu) * nch] = simdToS16(b[0]);
        dstr[(47 - iu) * nch] = simdToS16(a[3]);
        dstr[(49 + iu) * nch] = simdToS16(b[3]);
        dstl[(47 - iu) * nch] = simdToS16(a[2]);
        dstl[(49 + iu) * nch] = simdToS16(b[2]);
    }
}

/// Ports `ma_dr_mp3d_synth_granule` (miniaudio.h:94191 @5266f24da).
fn synthGranule(qmf_state: [*]f32, grbuf: [*]f32, nbands: usize, nch: usize, pcm: [*]i16, lins: [*]f32) void {
    for (0..nch) |i| dct2(grbuf + 576 * i, nbands);
    @memcpy(lins[0 .. 15 * 64], qmf_state[0 .. 15 * 64]);
    var i: usize = 0;
    while (i < nbands) : (i += 2) synth(grbuf + i, pcm + 32 * nch * i, nch, lins + i * 64);
    if (nch == 1) {
        var j: usize = 0;
        while (j < 15 * 64) : (j += 2) qmf_state[j] = lins[nbands * 64 + j];
    } else {
        @memcpy(qmf_state[0 .. 15 * 64], (lins + nbands * 64)[0 .. 15 * 64]);
    }
}

/// Ports `ma_dr_mp3d_match_frame` (miniaudio.h:94216 @5266f24da).
fn matchFrame(hdr: [*]const u8, mp3_bytes: c_int, frame_bytes: c_int) bool {
    var i: c_int = 0;
    var nmatch: c_int = 0;
    while (nmatch < max_frame_sync_matches) : (nmatch += 1) {
        i += hdrFrameBytes(hdr + @as(usize, @intCast(i)), frame_bytes) + hdrPadding(hdr + @as(usize, @intCast(i)));
        if (i + hdr_size > mp3_bytes) return nmatch > 0;
        if (!hdrCompare(hdr, hdr + @as(usize, @intCast(i)))) return false;
    }
    return true;
}

/// Ports `ma_dr_mp3d_find_frame` (miniaudio.h:94229 @5266f24da).
fn findFrame(mp3_in: [*]const u8, mp3_bytes: c_int, free_format_bytes: *c_int, ptr_frame_bytes: *c_int) c_int {
    var mp3 = mp3_in;
    var i: c_int = 0;
    while (i < mp3_bytes - hdr_size) : ({
        i += 1;
        mp3 += 1;
    }) {
        if (hdrValid(mp3)) {
            var frame_bytes = hdrFrameBytes(mp3, free_format_bytes.*);
            var frame_and_padding = frame_bytes + hdrPadding(mp3);
            var k: c_int = hdr_size;
            while (frame_bytes == 0 and k < max_free_format_frame_size and i + 2 * k < mp3_bytes - hdr_size) : (k += 1) {
                if (hdrCompare(mp3, mp3 + @as(usize, @intCast(k)))) {
                    const fb = k - hdrPadding(mp3);
                    const nextfb = fb + hdrPadding(mp3 + @as(usize, @intCast(k)));
                    if (i + k + nextfb + hdr_size > mp3_bytes or !hdrCompare(mp3, mp3 + @as(usize, @intCast(k + nextfb)))) continue;
                    frame_and_padding = k;
                    frame_bytes = fb;
                    free_format_bytes.* = fb;
                }
            }
            if ((frame_bytes != 0 and i + frame_and_padding <= mp3_bytes and
                matchFrame(mp3, mp3_bytes - i, frame_bytes)) or
                (i == 0 and frame_and_padding == mp3_bytes))
            {
                ptr_frame_bytes.* = frame_and_padding;
                return i;
            }
            free_format_bytes.* = 0;
        }
    }
    ptr_frame_bytes.* = 0;
    return mp3_bytes;
}

/// Ports `ma_dr_mp3dec_decode_frame` (miniaudio.h:94268 @5266f24da).
fn decodeFrame(dec: *Dec, mp3: [*]const u8, mp3_bytes: c_int, pcm_in: ?[*]i16, info: *FrameInfo) c_int {
    var i: c_int = 0;
    var frame_size: c_int = 0;
    var success: c_int = 1;
    if (mp3_bytes > 4 and dec.header[0] == 0xff and hdrCompare(&dec.header, mp3)) {
        frame_size = hdrFrameBytes(mp3, dec.free_format_bytes) + hdrPadding(mp3);
        if (frame_size != mp3_bytes and (frame_size + hdr_size > mp3_bytes or !hdrCompare(mp3, mp3 + @as(usize, @intCast(frame_size))))) {
            frame_size = 0;
        }
    }
    if (frame_size == 0) {
        @memset(std.mem.asBytes(dec), 0);
        i = findFrame(mp3, mp3_bytes, &dec.free_format_bytes, &frame_size);
        if (frame_size == 0 or i + frame_size > mp3_bytes) {
            info.frame_bytes = i;
            return 0;
        }
    }
    const hdr = mp3 + @as(usize, @intCast(i));
    @memcpy(&dec.header, hdr[0..hdr_size]);
    info.frame_bytes = i + frame_size;
    info.channels = if (hdrIsMono(hdr)) 1 else 2;
    info.sample_rate = @intCast(hdrSampleRateHz(hdr));
    info.layer = 4 - hdrGetLayer(hdr);
    info.bitrate_kbps = @intCast(hdrBitrateKbps(hdr));
    const nch: usize = @intCast(info.channels);

    var bs_frame: Bs = undefined;
    bsInit(&bs_frame, hdr + hdr_size, frame_size - hdr_size);
    if (hdrIsCrc(hdr)) _ = bsGetBits(&bs_frame, 16);

    if (info.layer == 3) {
        const main_data_begin = l3ReadSideInfo(&bs_frame, &dec.scratch.gr_info, hdr);
        if (main_data_begin < 0 or bs_frame.pos > bs_frame.limit) {
            dec.header[0] = 0; // `ma_dr_mp3dec_init`
            return 0;
        }
        success = @intFromBool(l3RestoreReservoir(dec, &bs_frame, &dec.scratch, main_data_begin));
        if (success != 0) if (pcm_in) |pcm0| {
            var pcm = pcm0;
            const ngr: usize = if (hdrTestMpeg1(hdr)) 2 else 1;
            for (0..ngr) |igr| {
                @memset(std.mem.sliceAsBytes(&dec.scratch.grbuf), 0);
                l3Decode(dec, &dec.scratch, dec.scratch.gr_info[igr * nch ..].ptr, nch);
                synthGranule(&dec.qmf_state, @ptrCast(&dec.scratch.grbuf), 18, nch, pcm, @ptrCast(&dec.scratch.syn));
                pcm += 576 * nch;
            }
        };
        l3SaveReservoir(dec, &dec.scratch);
    } else {
        var sci: L12ScaleInfo = .{};
        var pcm = pcm_in orelse return @intCast(hdrFrameSamples(hdr));
        l12ReadScaleInfo(hdr, &bs_frame, &sci);
        @memset(std.mem.sliceAsBytes(&dec.scratch.grbuf), 0);
        const grbuf: [*]f32 = @ptrCast(&dec.scratch.grbuf);
        i = 0;
        for (0..3) |igr| {
            i += l12DequantizeGranule(grbuf + @as(usize, @intCast(i)), &bs_frame, &sci, info.layer | 1);
            if (i == 12) {
                i = 0;
                l12ApplyScf384(&sci, sci.scf[igr..].ptr, grbuf);
                synthGranule(&dec.qmf_state, grbuf, 12, nch, pcm, @ptrCast(&dec.scratch.syn));
                @memset(std.mem.sliceAsBytes(&dec.scratch.grbuf), 0);
                pcm += 384 * nch;
            }
            if (bs_frame.pos > bs_frame.limit) {
                dec.header[0] = 0;
                return 0;
            }
        }
    }
    return success * @as(c_int, @intCast(hdrFrameSamples(&dec.header)));
}

/// The bytes a decoded frame came from, for the Xing/LAME parse, which
/// reads past the frame without a bound.
const FrameRef = struct {
    /// From the frame's start to the end of what the C may read defined.
    bytes: []const u8 = &.{},

    fn at(r: FrameRef, i: usize) u8 {
        if (i < r.bytes.len) return r.bytes[i];
        read_indeterminate = true;
        return 0;
    }
};

/// `char` as the C reads it into an `int`: signed where the target's is.
fn charToU32(b: u8) u32 {
    if (@typeInfo(c_char).int.signedness == .signed) return @bitCast(@as(i32, @as(i8, @bitCast(b))));
    return b;
}

/// Ports `ma_dr_mp3`'s reading state.
pub const Mp3 = struct {
    gpa: std.mem.Allocator,
    /// The memory stream (`memory.pData`, `memory.dataSize`,
    /// `memory.currentReadPos`) or the decoder's callbacks.
    io: Io,
    /// The whole buffer: a memory stream's `dataSize` can shrink below it.
    full: []const u8,
    memory_mode: bool,
    dec: Dec,
    channels: u32 = 0,
    sample_rate: u32 = 0,
    mp3_frame_channels: u32 = 0,
    mp3_frame_sample_rate: u32 = 0,
    pcm_frames_consumed_in_mp3_frame: u32 = 0,
    pcm_frames_remaining_in_mp3_frame: u32 = 0,
    pcm_frames: [max_samples_per_frame * 2]i16 = @splat(0),
    current_pcm_frame: u64 = 0,
    stream_cursor: u64 = 0,
    stream_length: u64 = 0,
    stream_start_offset: u64 = 0,
    delay_in_pcm_frames: u32 = 0,
    padding_in_pcm_frames: u32 = 0,
    total_pcm_frame_count: u64 = 0,
    data: []u8 = &.{},
    data_size: usize = 0,
    data_consumed: usize = 0,
    at_end: bool = false,

    /// Ports `ma_dr_mp3_init_memory_with_metadata` (miniaudio.h:95037 @5266f24da)
    /// for a `dr_mp3` stream, and `ma_dr_mp3_init` (miniaudio.h:94980 @5266f24da)
    /// over the decoder's callbacks for a `decoder` one.
    ///
    /// Return: the decoder, owned by the caller (`deinit`), or null where
    /// the C returns `MA_FALSE`.
    pub fn init(gpa: std.mem.Allocator, io: Io) ?*Mp3 {
        if (io.data.len == 0) return null;
        const m = gpa.create(Mp3) catch return null;
        m.* = .{ .gpa = gpa, .io = io, .full = io.data, .memory_mode = io.mode != .decoder, .dec = undefined };
        @memset(std.mem.asBytes(&m.dec), 0);
        if (!m.initInternal()) {
            m.deinit();
            return null;
        }
        return m;
    }

    /// Releases the decoder.
    pub fn deinit(m: *Mp3) void {
        if (m.data.len != 0) m.gpa.free(m.data);
        m.gpa.destroy(m);
    }

    /// Ports `ma_dr_mp3_init_internal` (miniaudio.h:94741 @5266f24da), with
    /// no metadata callback.
    fn initInternal(m: *Mp3) bool {
        m.dec.header[0] = 0;
        m.stream_cursor = 0;
        m.stream_length = std.math.maxInt(u64);
        m.stream_start_offset = 0;
        m.delay_in_pcm_frames = 0;
        m.padding_in_pcm_frames = 0;
        m.total_pcm_frame_count = std.math.maxInt(u64);
        var detected_mp3_frame_count: u32 = 0xFFFFFFFF;

        if (m.io.seek(0, .end)) {
            var stream_len: i64 = m.io.tell();
            var stream_end_offset: c_int = 0;
            if (stream_len > 128) {
                if (m.io.seek(stream_end_offset - 128, .end)) {
                    var id3: [3]u8 = undefined;
                    if (m.io.read(&id3) == 3 and std.mem.eql(u8, &id3, "TAG")) {
                        stream_end_offset -= 128;
                        stream_len -= 128;
                    }
                }
            }
            if (stream_len > 32) {
                if (m.io.seek(stream_end_offset - 32, .end)) {
                    var ape: [32]u8 = undefined;
                    if (m.io.read(&ape) == 32 and std.mem.eql(u8, ape[0..8], "APETAGEX")) {
                        const tag_size = charToU32(ape[24]) | (charToU32(ape[25]) << 8) | (charToU32(ape[26]) << 16) | (charToU32(ape[27]) << 24);
                        const total = @as(u32, 32) +% tag_size;
                        if (@as(i64, total) < stream_len) {
                            stream_end_offset = @bitCast(@as(u32, @bitCast(stream_end_offset)) -% total);
                            stream_len -= total;
                        }
                    }
                }
            }
            if (!m.io.seek(0, .set)) return false;
            m.stream_length = @bitCast(stream_len);
            if (m.memory_mode) m.io.data = m.full[0..@intCast(m.stream_length)];
        }

        var header: [10]u8 = undefined;
        if (m.io.read(&header) != 10) return false;
        if (std.mem.eql(u8, header[0..3], "ID3")) {
            var tag_size: u32 = ((@as(u32, header[6]) & 0x7F) << 21) | ((@as(u32, header[7]) & 0x7F) << 14) | ((@as(u32, header[8]) & 0x7F) << 7) | (@as(u32, header[9]) & 0x7F);
            if (header[5] & 0x10 != 0) tag_size += 10;
            if (!m.io.seek(@as(i32, @bitCast(tag_size)), .cur)) return false;
            m.stream_start_offset += 10 + @as(u64, tag_size);
            m.stream_cursor = m.stream_start_offset;
        } else {
            if (!m.io.seek(0, .set)) return false;
        }

        var info: FrameInfo = .{};
        var ref: FrameRef = .{};
        const first_count = m.decodeNextFrameEx(&m.pcm_frames, &info, &ref);
        if (first_count == 0) return false;

        var bs: Bs = undefined;
        bsInit(&bs, ref.bytes.ptr + hdr_size, info.frame_bytes - hdr_size);
        if (hdrIsCrc(ref.bytes.ptr)) _ = bsGetBits(&bs, 16);
        var gr_info: [4]GrInfo = undefined;
        if (l3ReadSideInfo(&bs, &gr_info, ref.bytes.ptr) >= 0) {
            const beg: usize = hdr_size + @as(usize, @intCast(@divTrunc(bs.pos, 8)));
            var p = beg;
            const is_xing = ref.at(p) == 'X' and ref.at(p + 1) == 'i' and ref.at(p + 2) == 'n' and ref.at(p + 3) == 'g';
            const is_info = ref.at(p) == 'I' and ref.at(p + 1) == 'n' and ref.at(p + 2) == 'f' and ref.at(p + 3) == 'o';
            if (is_xing or is_info) {
                const flags = ref.at(p + 7);
                p += 8;
                if (flags & 0x01 != 0) {
                    detected_mp3_frame_count = @as(u32, ref.at(p)) << 24 | @as(u32, ref.at(p + 1)) << 16 | @as(u32, ref.at(p + 2)) << 8 | ref.at(p + 3);
                    p += 4;
                }
                if (flags & 0x02 != 0) p += 4;
                if (flags & 0x04 != 0) p += 100;
                if (flags & 0x08 != 0) p += 4;
                if (ref.at(p) != 0) {
                    p += 21;
                    if (@as(c_int, @intCast(p)) + 14 < info.frame_bytes) {
                        const delay: c_int = @intCast(((@as(u32, ref.at(p)) << 4) | (@as(u32, ref.at(p + 1)) >> 4)) + (528 + 1));
                        var padding: c_int = @as(c_int, @intCast(((@as(u32, ref.at(p + 1)) & 0xF) << 8) | ref.at(p + 2))) - (528 + 1);
                        if (padding < 0) padding = 0;
                        m.delay_in_pcm_frames = @intCast(delay);
                        m.padding_in_pcm_frames = @intCast(padding);
                    }
                }
                m.pcm_frames_remaining_in_mp3_frame = 0;
                m.stream_start_offset += @as(u32, @bitCast(info.frame_bytes));
                m.stream_cursor = m.stream_start_offset;
                m.dec.header[0] = 0;
            }
        }

        if (detected_mp3_frame_count != 0xFFFFFFFF) m.total_pcm_frame_count = detected_mp3_frame_count *% first_count;
        m.channels = m.mp3_frame_channels;
        m.sample_rate = m.mp3_frame_sample_rate;
        return true;
    }

    /// Ports `ma_dr_mp3_decode_next_frame_ex` and its two arms,
    /// `ma_dr_mp3_decode_next_frame_ex__memory` and
    /// `ma_dr_mp3_decode_next_frame_ex__callbacks` (miniaudio.h:94713, 94678, 94592 @5266f24da).
    fn decodeNextFrameEx(m: *Mp3, pcm: ?[*]i16, frame_info: ?*FrameInfo, frame_ref: ?*FrameRef) u32 {
        if (m.memory_mode) return m.decodeNextFrameMemory(pcm, frame_info, frame_ref);
        return m.decodeNextFrameCallbacks(pcm, frame_info, frame_ref);
    }

    fn frameDecoded(m: *Mp3, info: FrameInfo, frame_info: ?*FrameInfo) u32 {
        const n = hdrFrameSamples(&m.dec.header);
        m.pcm_frames_consumed_in_mp3_frame = 0;
        m.pcm_frames_remaining_in_mp3_frame = n;
        m.mp3_frame_channels = @intCast(info.channels);
        m.mp3_frame_sample_rate = @intCast(info.sample_rate);
        if (frame_info) |fi| fi.* = info;
        return n;
    }

    /// Ports `ma_dr_mp3_decode_next_frame_ex__memory` (miniaudio.h:94678 @5266f24da).
    fn decodeNextFrameMemory(m: *Mp3, pcm: ?[*]i16, frame_info: ?*FrameInfo, frame_ref: ?*FrameRef) u32 {
        if (m.at_end) return 0;
        var info: FrameInfo = .{};
        var n: u32 = 0;
        while (true) {
            const avail = m.io.data.len -% m.io.pos;
            const got = decodeFrame(&m.dec, m.full.ptr + m.io.pos, @truncate(@as(isize, @bitCast(avail))), pcm, &info);
            if (got > 0) {
                n = m.frameDecoded(info, frame_info);
                if (frame_ref) |r| r.* = .{ .bytes = m.full[m.io.pos..] };
                break;
            } else if (info.frame_bytes > 0) {
                m.io.pos += @intCast(info.frame_bytes);
                m.stream_cursor += @intCast(info.frame_bytes);
            } else break;
        }
        m.io.pos +%= @as(usize, @bitCast(@as(isize, info.frame_bytes)));
        m.stream_cursor +%= @as(u64, @bitCast(@as(i64, info.frame_bytes)));
        return n;
    }

    /// Ports `ma_dr_mp3__on_read` and `ma_dr_mp3__on_read_clamped`
    /// (miniaudio.h:94516, 94528 @5266f24da).
    fn readClamped(m: *Mp3, buf: []u8) usize {
        var want = buf.len;
        if (m.stream_length != std.math.maxInt(u64)) {
            const remaining = m.stream_length -% m.stream_cursor;
            if (want > remaining) want = @intCast(remaining);
        }
        if (want == 0) return 0;
        const n = m.io.read(buf[0..want]);
        m.stream_cursor += n;
        return n;
    }

    fn grow(m: *Mp3, cap: usize) bool {
        if (m.data.len == 0) {
            m.data = m.gpa.alloc(u8, cap) catch return false;
        } else {
            m.data = m.gpa.realloc(m.data, cap) catch return false;
        }
        return true;
    }

    /// Ports `ma_dr_mp3_decode_next_frame_ex__callbacks` (miniaudio.h:94592 @5266f24da).
    fn decodeNextFrameCallbacks(m: *Mp3, pcm: ?[*]i16, frame_info: ?*FrameInfo, frame_ref: ?*FrameRef) u32 {
        if (m.at_end) return 0;
        while (true) {
            var info: FrameInfo = .{};
            if (m.data_size < min_data_chunk_size) {
                if (m.data.len != 0) std.mem.copyForwards(u8, m.data[0..m.data_size], m.data[m.data_consumed..][0..m.data_size]);
                m.data_consumed = 0;
                if (m.data.len < data_chunk_size) {
                    if (!m.grow(data_chunk_size)) return 0;
                }
                const n = m.readClamped(m.data[m.data_size..]);
                if (n == 0 and m.data_size == 0) {
                    m.at_end = true;
                    return 0;
                }
                m.data_size += n;
            }
            if (m.data_size > std.math.maxInt(c_int)) {
                m.at_end = true;
                return 0;
            }
            const got = decodeFrame(&m.dec, m.data.ptr + m.data_consumed, @intCast(m.data_size), pcm, &info);
            m.data_consumed += @intCast(info.frame_bytes);
            m.data_size -= @intCast(info.frame_bytes);
            if (got > 0) {
                const n = m.frameDecoded(info, frame_info);
                if (frame_ref) |r| {
                    const start = m.data_consumed - @as(usize, @intCast(info.frame_bytes));
                    r.* = .{ .bytes = m.data[start .. m.data_consumed + m.data_size] };
                }
                return n;
            } else if (info.frame_bytes == 0) {
                std.mem.copyForwards(u8, m.data[0..m.data_size], m.data[m.data_consumed..][0..m.data_size]);
                m.data_consumed = 0;
                if (m.data.len == m.data_size) {
                    if (!m.grow(m.data.len + data_chunk_size)) return 0;
                }
                const n = m.readClamped(m.data[m.data_size..]);
                if (n == 0) {
                    m.at_end = true;
                    return 0;
                }
                m.data_size += n;
            }
        }
    }

    /// Ports `ma_dr_mp3_read_pcm_frames_raw` (miniaudio.h:95208 @5266f24da)
    /// at `int16`.
    fn readPcmFramesRaw(m: *Mp3, frames_in: u64, out: ?[]i16) u64 {
        var frames = frames_in;
        var total: u64 = 0;
        while (frames > 0) {
            if (m.current_pcm_frame < m.delay_in_pcm_frames) {
                const skip: u32 = @intCast(@min(m.pcm_frames_remaining_in_mp3_frame, m.delay_in_pcm_frames - m.current_pcm_frame));
                m.current_pcm_frame += skip;
                m.pcm_frames_consumed_in_mp3_frame += skip;
                m.pcm_frames_remaining_in_mp3_frame -= skip;
            }
            var consume: u32 = @intCast(@min(m.pcm_frames_remaining_in_mp3_frame, frames));
            if (m.total_pcm_frame_count != std.math.maxInt(u64) and m.total_pcm_frame_count > m.padding_in_pcm_frames) {
                if (m.current_pcm_frame < m.total_pcm_frame_count - m.padding_in_pcm_frames) {
                    const to_padding = (m.total_pcm_frame_count - m.padding_in_pcm_frames) - m.current_pcm_frame;
                    if (consume > to_padding) consume = @intCast(to_padding);
                } else break;
            }
            if (out) |o| {
                const dst = o[@intCast(total * m.channels)..][0 .. consume * m.channels];
                const src = m.pcm_frames[m.pcm_frames_consumed_in_mp3_frame * m.mp3_frame_channels ..][0 .. consume * m.channels];
                @memcpy(dst, src);
            }
            m.current_pcm_frame += consume;
            m.pcm_frames_consumed_in_mp3_frame += consume;
            m.pcm_frames_remaining_in_mp3_frame -= consume;
            total += consume;
            frames -= consume;
            if (frames == 0) break;
            if (m.total_pcm_frame_count != std.math.maxInt(u64) and m.total_pcm_frame_count > m.padding_in_pcm_frames and
                m.current_pcm_frame >= m.total_pcm_frame_count - m.padding_in_pcm_frames) break;
            if (m.decodeNextFrameEx(&m.pcm_frames, null, null) == 0) break;
        }
        return total;
    }

    /// Ports `ma_dr_mp3_read_pcm_frames_f32` (miniaudio.h:95265 @5266f24da)
    /// without `MA_DR_MP3_FLOAT_OUTPUT`: `int16` samples times 2^-15.
    pub fn readPcmFramesF32(m: *Mp3, frames: u64, out: []f32) u64 {
        var temp: [8192]i16 = undefined;
        var total: u64 = 0;
        while (total < frames) {
            const now = @min(temp.len / m.channels, frames - total);
            const got = m.readPcmFramesRaw(now, &temp);
            if (got == 0) break;
            const n: usize = @intCast(got * m.channels);
            for (out[@intCast(total * m.channels)..][0..n], temp[0..n]) |*d, s| {
                var x: f32 = @floatFromInt(s);
                x = x * 0.000030517578125;
                d.* = x;
            }
            total += got;
        }
        return total;
    }

    /// Ports `ma_dr_mp3_reset` and `ma_dr_mp3_seek_to_start_of_stream`
    /// (miniaudio.h:95323, 95333 @5266f24da).
    fn seekToStartOfStream(m: *Mp3) bool {
        // `ma_dr_mp3__on_seek_64`: the cursor follows each step
        if (!m.io.seekFromStart(m.stream_start_offset)) return false;
        m.stream_cursor = m.stream_start_offset;
        m.pcm_frames_consumed_in_mp3_frame = 0;
        m.pcm_frames_remaining_in_mp3_frame = 0;
        m.current_pcm_frame = 0;
        m.data_size = 0;
        m.at_end = false;
        m.dec.header[0] = 0;
        return true;
    }

    /// Ports `ma_dr_mp3_seek_to_pcm_frame` (miniaudio.h:95423 @5266f24da)
    /// with no seek table, which `ma_mp3` never binds: brute force.
    fn seekToPcmFrame(m: *Mp3, frame_index: u64) bool {
        if (frame_index == 0) return m.seekToStartOfStream();
        if (frame_index == m.current_pcm_frame) return true;
        if (frame_index < m.current_pcm_frame) {
            if (!m.seekToStartOfStream()) return false;
        }
        const offset = frame_index - m.current_pcm_frame;
        return m.readPcmFramesRaw(offset, null) == offset;
    }

    /// Ports `ma_dr_mp3_get_pcm_frame_count` and
    /// `ma_dr_mp3_get_mp3_and_pcm_frame_count` (miniaudio.h:95477, 95437 @5266f24da).
    pub fn getPcmFrameCount(m: *Mp3) u64 {
        if (m.total_pcm_frame_count != std.math.maxInt(u64)) {
            var total = m.total_pcm_frame_count;
            if (total >= m.delay_in_pcm_frames) total -= m.delay_in_pcm_frames;
            if (total >= m.padding_in_pcm_frames) total -= m.padding_in_pcm_frames;
            return total;
        }
        const current = m.current_pcm_frame;
        if (!m.seekToStartOfStream()) return 0;
        var total: u64 = 0;
        while (true) {
            const n = m.decodeNextFrameEx(null, null, null);
            if (n == 0) break;
            total += n;
        }
        if (!m.seekToStartOfStream()) return 0;
        if (!m.seekToPcmFrame(current)) return 0;
        return total;
    }
};

/// The decoder's kernels, for `make mtmd-decode-diff`'s kernel diff.
pub const kernels = struct {
    pub fn dct39(y: *[9]f32) void {
        l3Dct39(y);
    }
    pub fn imdct36(grbuf: [*]f32, overlap: [*]f32, long_window: bool, stop: bool, nbands: c_int) void {
        _ = long_window;
        l3Imdct36(grbuf, overlap, g_mdct_window[if (stop) 18 else 0..].ptr, nbands);
    }
    pub fn imdctShort(grbuf: [*]f32, overlap: [*]f32, nbands: c_int) void {
        l3ImdctShort(grbuf, overlap, nbands);
    }
    pub fn antialias(grbuf: [*]f32, nbands: c_int) void {
        l3Antialias(grbuf, nbands);
    }
    pub fn dctII(grbuf: [*]f32, n: usize) void {
        dct2(grbuf, n);
    }
    pub fn synthGranule_(qmf_state: [*]f32, grbuf: [*]f32, nbands: usize, nch: usize, pcm: [*]i16, lins: [*]f32) void {
        synthGranule(qmf_state, grbuf, nbands, nch, pcm, lins);
    }
    pub fn synthPair_(pcm: [*]i16, nch: usize, z: [*]const f32) void {
        synthPair(pcm, nch, z);
    }
    pub fn pow43(x: c_int) f32 {
        return l3Pow43(x);
    }
    pub fn ldexpQ2(y: f32, e: c_int) f32 {
        return l3LdexpQ2(y, e);
    }
    /// The decoder's state, laid out as the C's.
    pub const Decoder = Dec;
    /// One `ma_dr_mp3dec_decode_frame`.
    pub fn decodeOne(dec: *Dec, mp3: []const u8, pcm: [*]i16, frame_bytes: *c_int) c_int {
        var info: FrameInfo = .{};
        const n = decodeFrame(dec, mp3.ptr, @intCast(mp3.len), pcm, &info);
        frame_bytes.* = info.frame_bytes;
        return n;
    }
    pub const Bitstream = Bs;
    pub const Granule = GrInfo;
    pub fn decodeScalefactors(hdr: [*]const u8, ist_pos: [*]u8, bs: *Bs, gr: *const GrInfo, scf: [*]f32, ch: c_int) void {
        l3DecodeScalefactors(hdr, ist_pos, bs, gr, scf, ch);
    }
    pub fn getBits(bs: *Bs, n: c_int) u32 {
        return bsGetBits(bs, n);
    }
    pub fn readSideInfo(bs: *Bs, gr: [*]GrInfo, hdr: [*]const u8) c_int {
        return l3ReadSideInfo(bs, gr, hdr);
    }
    /// `g_mdct_window`, for the reference's calls.
    pub const mdct_window = &g_mdct_window;
};

// -----------------------------------------------------------------------------
// tables

/// `halfrate` (miniaudio/miniaudio.h:92939 @5266f24da).
const halfrate = [90]u8{
    0,  4,  8,   12,  16,  20,  24,  28,  32,  40,  48,  56,  64,  72, 80, 0,
    4,  8,  12,  16,  20,  24,  28,  32,  40,  48,  56,  64,  72,  80, 0,  16,
    24, 28, 32,  40,  48,  56,  64,  72,  80,  88,  96,  112, 128, 0,  16, 20,
    24, 28, 32,  40,  48,  56,  64,  80,  96,  112, 128, 160, 0,   16, 24, 28,
    32, 40, 48,  56,  64,  80,  96,  112, 128, 160, 192, 0,   16,  32, 48, 64,
    80, 96, 112, 128, 144, 160, 176, 192, 208, 224,
};

/// `g_bitalloc_code_tab` (miniaudio/miniaudio.h:93033 @5266f24da).
const g_bitalloc_code_tab = [92]u8{
    0,  17, 3,  4, 5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
    0,  17, 18, 3, 19, 4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 16,
    0,  17, 18, 3, 19, 4,  5,  16, 0,  17, 18, 16, 0,  17, 18, 19,
    4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14, 15, 0,  17, 18, 3,
    19, 4,  5,  6, 7,  8,  9,  10, 11, 12, 13, 14, 0,  2,  3,  4,
    5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15, 16,
};

/// `g_scf_long` (miniaudio/miniaudio.h:93123 @5266f24da).
const g_scf_long = [184]u8{
    6,  6,  6,  6,  6,  6,   8,  10, 12,  14, 16, 20, 24, 28, 32, 38,
    46, 52, 60, 68, 58, 54,  0,  12, 12,  12, 12, 12, 12, 16, 20, 24,
    28, 32, 40, 48, 56, 64,  76, 90, 2,   2,  2,  2,  2,  0,  6,  6,
    6,  6,  6,  6,  8,  10,  12, 14, 16,  20, 24, 28, 32, 38, 46, 52,
    60, 68, 58, 54, 0,  6,   6,  6,  6,   6,  6,  8,  10, 12, 14, 16,
    18, 22, 26, 32, 38, 46,  54, 62, 70,  76, 36, 0,  6,  6,  6,  6,
    6,  6,  8,  10, 12, 14,  16, 20, 24,  28, 32, 38, 46, 52, 60, 68,
    58, 54, 0,  4,  4,  4,   4,  4,  4,   6,  6,  8,  8,  10, 12, 16,
    20, 24, 28, 34, 42, 50,  54, 76, 158, 0,  4,  4,  4,  4,  4,  4,
    6,  6,  6,  8,  10, 12,  16, 18, 22,  28, 34, 40, 46, 54, 54, 192,
    0,  4,  4,  4,  4,  4,   4,  6,  6,   8,  10, 12, 16, 20, 24, 30,
    38, 46, 56, 68, 84, 102, 26, 0,
};

/// `g_scf_short` (miniaudio/miniaudio.h:93133 @5266f24da).
const g_scf_short = [320]u8{
    4,  4,  4,  4,  4,  4,  4,  4,  4,  6,  6,  6,  8,  8,  8,  10,
    10, 10, 12, 12, 12, 14, 14, 14, 18, 18, 18, 24, 24, 24, 30, 30,
    30, 40, 40, 40, 18, 18, 18, 0,  8,  8,  8,  8,  8,  8,  8,  8,
    8,  12, 12, 12, 16, 16, 16, 20, 20, 20, 24, 24, 24, 28, 28, 28,
    36, 36, 36, 2,  2,  2,  2,  2,  2,  2,  2,  2,  26, 26, 26, 0,
    4,  4,  4,  4,  4,  4,  4,  4,  4,  6,  6,  6,  6,  6,  6,  8,
    8,  8,  10, 10, 10, 14, 14, 14, 18, 18, 18, 26, 26, 26, 32, 32,
    32, 42, 42, 42, 18, 18, 18, 0,  4,  4,  4,  4,  4,  4,  4,  4,
    4,  6,  6,  6,  8,  8,  8,  10, 10, 10, 12, 12, 12, 14, 14, 14,
    18, 18, 18, 24, 24, 24, 32, 32, 32, 44, 44, 44, 12, 12, 12, 0,
    4,  4,  4,  4,  4,  4,  4,  4,  4,  6,  6,  6,  8,  8,  8,  10,
    10, 10, 12, 12, 12, 14, 14, 14, 18, 18, 18, 24, 24, 24, 30, 30,
    30, 40, 40, 40, 18, 18, 18, 0,  4,  4,  4,  4,  4,  4,  4,  4,
    4,  4,  4,  4,  6,  6,  6,  8,  8,  8,  10, 10, 10, 12, 12, 12,
    14, 14, 14, 18, 18, 18, 22, 22, 22, 30, 30, 30, 56, 56, 56, 0,
    4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  6,  6,  6,  6,
    6,  6,  10, 10, 10, 12, 12, 12, 14, 14, 14, 16, 16, 16, 20, 20,
    20, 26, 26, 26, 66, 66, 66, 0,  4,  4,  4,  4,  4,  4,  4,  4,
    4,  4,  4,  4,  6,  6,  6,  8,  8,  8,  12, 12, 12, 16, 16, 16,
    20, 20, 20, 26, 26, 26, 34, 34, 34, 42, 42, 42, 12, 12, 12, 0,
};

/// `g_scf_mixed` (miniaudio/miniaudio.h:93143 @5266f24da).
const g_scf_mixed = [320]u8{
    6,  6,  6,  6,  6,  6,  6,  6,  6,  8,  8,  8,  10, 10, 10, 12,
    12, 12, 14, 14, 14, 18, 18, 18, 24, 24, 24, 30, 30, 30, 40, 40,
    40, 18, 18, 18, 0,  0,  0,  0,  12, 12, 12, 4,  4,  4,  8,  8,
    8,  12, 12, 12, 16, 16, 16, 20, 20, 20, 24, 24, 24, 28, 28, 28,
    36, 36, 36, 2,  2,  2,  2,  2,  2,  2,  2,  2,  26, 26, 26, 0,
    6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  8,  8,  8,  10,
    10, 10, 14, 14, 14, 18, 18, 18, 26, 26, 26, 32, 32, 32, 42, 42,
    42, 18, 18, 18, 0,  0,  0,  0,  6,  6,  6,  6,  6,  6,  6,  6,
    6,  8,  8,  8,  10, 10, 10, 12, 12, 12, 14, 14, 14, 18, 18, 18,
    24, 24, 24, 32, 32, 32, 44, 44, 44, 12, 12, 12, 0,  0,  0,  0,
    6,  6,  6,  6,  6,  6,  6,  6,  6,  8,  8,  8,  10, 10, 10, 12,
    12, 12, 14, 14, 14, 18, 18, 18, 24, 24, 24, 30, 30, 30, 40, 40,
    40, 18, 18, 18, 0,  0,  0,  0,  4,  4,  4,  4,  4,  4,  6,  6,
    4,  4,  4,  6,  6,  6,  8,  8,  8,  10, 10, 10, 12, 12, 12, 14,
    14, 14, 18, 18, 18, 22, 22, 22, 30, 30, 30, 56, 56, 56, 0,  0,
    4,  4,  4,  4,  4,  4,  6,  6,  4,  4,  4,  6,  6,  6,  6,  6,
    6,  10, 10, 10, 12, 12, 12, 14, 14, 14, 16, 16, 16, 20, 20, 20,
    26, 26, 26, 66, 66, 66, 0,  0,  4,  4,  4,  4,  4,  4,  6,  6,
    4,  4,  4,  6,  6,  6,  8,  8,  8,  12, 12, 12, 16, 16, 16, 20,
    20, 20, 26, 26, 26, 34, 34, 34, 42, 42, 42, 12, 12, 12, 0,  0,
};

/// `g_scf_partitions` (miniaudio/miniaudio.h:93289 @5266f24da).
const g_scf_partitions = [84]u8{
    6,  5,  5,  5, 6,  5,  5,  5, 6,  5,  7,  3,  11, 10, 0,  0,
    7,  7,  7,  0, 6,  6,  6,  3, 8,  8,  5,  0,  8,  9,  6,  12,
    6,  9,  9,  9, 6,  9,  12, 6, 15, 18, 0,  0,  6,  15, 12, 0,
    6,  12, 9,  6, 6,  18, 9,  0, 9,  9,  6,  12, 9,  9,  9,  9,
    9,  9,  12, 6, 18, 18, 0,  0, 12, 12, 12, 0,  12, 9,  9,  6,
    15, 12, 9,  0,
};

/// `g_scfc_decode` (miniaudio/miniaudio.h:93300 @5266f24da).
const g_scfc_decode = [16]u8{
    0, 1, 2, 3, 12, 5, 6, 7, 9, 10, 11, 13, 14, 15, 18, 19,
};

/// `g_mod` (miniaudio/miniaudio.h:93306 @5266f24da).
const g_mod = [24]u8{
    5, 5, 4, 4, 5, 5, 4, 1, 4, 3, 1, 1, 5, 6, 6, 1,
    4, 4, 4, 1, 4, 3, 1, 1,
};

/// `g_preamp` (miniaudio/miniaudio.h:93332 @5266f24da).
const g_preamp = [10]u8{
    1, 1, 1, 1, 2, 2, 3, 3, 3, 2,
};

/// `g_expfrac` (miniaudio/miniaudio.h:93274 @5266f24da).
const g_expfrac = [4]f32{
    9.31322575e-10, 7.83145814e-10, 6.58544508e-10, 5.53767716e-10,
};

/// `ma_dr_mp3_g_pow43` (miniaudio/miniaudio.h:93348 @5266f24da).
const g_pow43 = [145]f32{
    0.0,        -1.0,       -2.519842,  -4.326749,  -6.349604,  -8.549880,  -10.902724, -13.390518, -16.000000, -18.720754, -21.544347, -24.463781, -27.473142, -30.567351, -33.741992, -36.993181,
    0.0,        1.0,        2.519842,   4.326749,   6.349604,   8.549880,   10.902724,  13.390518,  16.000000,  18.720754,  21.544347,  24.463781,  27.473142,  30.567351,  33.741992,  36.993181,
    40.317474,  43.711787,  47.173345,  50.699631,  54.288352,  57.937408,  61.644865,  65.408941,  69.227979,  73.100443,  77.024898,  81.000000,  85.024491,  89.097188,  93.216975,  97.382800,
    101.593667, 105.848633, 110.146801, 114.487321, 118.869381, 123.292209, 127.755065, 132.257246, 136.798076, 141.376907, 145.993119, 150.646117, 155.335327, 160.060199, 164.820202, 169.614826,
    174.443577, 179.305980, 184.201575, 189.129918, 194.090580, 199.083145, 204.107210, 209.162385, 214.248292, 219.364564, 224.510845, 229.686789, 234.892058, 240.126328, 245.389280, 250.680604,
    256.000000, 261.347174, 266.721841, 272.123723, 277.552547, 283.008049, 288.489971, 293.998060, 299.532071, 305.091761, 310.676898, 316.287249, 321.922592, 327.582707, 333.267377, 338.976394,
    344.709550, 350.466646, 356.247482, 362.051866, 367.879608, 373.730522, 379.604427, 385.501143, 391.420496, 397.362314, 403.326427, 409.312672, 415.320884, 421.350905, 427.402579, 433.475750,
    439.570269, 445.685987, 451.822757, 457.980436, 464.158883, 470.357960, 476.577530, 482.817459, 489.077615, 495.357868, 501.658090, 507.978156, 514.317941, 520.677324, 527.056184, 533.454404,
    539.871867, 546.308458, 552.764065, 559.238575, 565.731879, 572.243870, 578.774440, 585.323483, 591.890898, 598.476581, 605.080431, 611.702349, 618.342238, 625.000000, 631.675540, 638.368763,
    645.079578,
};

/// `tabs` (miniaudio/miniaudio.h:93371 @5266f24da).
const tabs = [2164]i16{
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    785,   785,   785,   785,   784,   784,   784,   784,   513,   513,   513,   513,   513,   513,   513,   513,
    256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
    -255,  1313,  1298,  1282,  785,   785,   785,   785,   784,   784,   784,   784,   769,   769,   769,   769,
    256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
    290,   288,   -255,  1313,  1298,  1282,  769,   769,   769,   769,   529,   529,   529,   529,   529,   529,
    529,   529,   528,   528,   528,   528,   528,   528,   528,   528,   512,   512,   512,   512,   512,   512,
    512,   512,   290,   288,   -253,  -318,  -351,  -367,  785,   785,   785,   785,   784,   784,   784,   784,
    769,   769,   769,   769,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
    256,   256,   256,   256,   819,   818,   547,   547,   275,   275,   275,   275,   561,   560,   515,   546,
    289,   274,   288,   258,   -254,  -287,  1329,  1299,  1314,  1312,  1057,  1057,  1042,  1042,  1026,  1026,
    784,   784,   784,   784,   529,   529,   529,   529,   529,   529,   529,   529,   769,   769,   769,   769,
    768,   768,   768,   768,   563,   560,   306,   306,   291,   259,   -252,  -413,  -477,  -542,  1298,  -575,
    1041,  1041,  784,   784,   784,   784,   769,   769,   769,   769,   256,   256,   256,   256,   256,   256,
    256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   -383,  -399,  1107,  1092,  1106,  1061,
    849,   849,   789,   789,   1104,  1091,  773,   773,   1076,  1075,  341,   340,   325,   309,   834,   804,
    577,   577,   532,   532,   516,   516,   832,   818,   803,   816,   561,   561,   531,   531,   515,   546,
    289,   289,   288,   258,   -252,  -429,  -493,  -559,  1057,  1057,  1042,  1042,  529,   529,   529,   529,
    529,   529,   529,   529,   784,   784,   784,   784,   769,   769,   769,   769,   512,   512,   512,   512,
    512,   512,   512,   512,   -382,  1077,  -415,  1106,  1061,  1104,  849,   849,   789,   789,   1091,  1076,
    1029,  1075,  834,   834,   597,   581,   340,   340,   339,   324,   804,   833,   532,   532,   832,   772,
    818,   803,   817,   787,   816,   771,   290,   290,   290,   290,   288,   258,   -253,  -349,  -414,  -447,
    -463,  1329,  1299,  -479,  1314,  1312,  1057,  1057,  1042,  1042,  1026,  1026,  785,   785,   785,   785,
    784,   784,   784,   784,   769,   769,   769,   769,   768,   768,   768,   768,   -319,  851,   821,   -335,
    836,   850,   805,   849,   341,   340,   325,   336,   533,   533,   579,   579,   564,   564,   773,   832,
    578,   548,   563,   516,   321,   276,   306,   291,   304,   259,   -251,  -572,  -733,  -830,  -863,  -879,
    1041,  1041,  784,   784,   784,   784,   769,   769,   769,   769,   256,   256,   256,   256,   256,   256,
    256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   -511,  -527,  -543,  1396,  1351,  1381,
    1366,  1395,  1335,  1380,  -559,  1334,  1138,  1138,  1063,  1063,  1350,  1392,  1031,  1031,  1062,  1062,
    1364,  1363,  1120,  1120,  1333,  1348,  881,   881,   881,   881,   375,   374,   359,   373,   343,   358,
    341,   325,   791,   791,   1123,  1122,  -703,  1105,  1045,  -719,  865,   865,   790,   790,   774,   774,
    1104,  1029,  338,   293,   323,   308,   -799,  -815,  833,   788,   772,   818,   803,   816,   322,   292,
    307,   320,   561,   531,   515,   546,   289,   274,   288,   258,   -251,  -525,  -605,  -685,  -765,  -831,
    -846,  1298,  1057,  1057,  1312,  1282,  785,   785,   785,   785,   784,   784,   784,   784,   769,   769,
    769,   769,   512,   512,   512,   512,   512,   512,   512,   512,   1399,  1398,  1383,  1367,  1382,  1396,
    1351,  -511,  1381,  1366,  1139,  1139,  1079,  1079,  1124,  1124,  1364,  1349,  1363,  1333,  882,   882,
    882,   882,   807,   807,   807,   807,   1094,  1094,  1136,  1136,  373,   341,   535,   535,   881,   775,
    867,   822,   774,   -591,  324,   338,   -671,  849,   550,   550,   866,   864,   609,   609,   293,   336,
    534,   534,   789,   835,   773,   -751,  834,   804,   308,   307,   833,   788,   832,   772,   562,   562,
    547,   547,   305,   275,   560,   515,   290,   290,   -252,  -397,  -477,  -557,  -622,  -653,  -719,  -735,
    -750,  1329,  1299,  1314,  1057,  1057,  1042,  1042,  1312,  1282,  1024,  1024,  785,   785,   785,   785,
    784,   784,   784,   784,   769,   769,   769,   769,   -383,  1127,  1141,  1111,  1126,  1140,  1095,  1110,
    869,   869,   883,   883,   1079,  1109,  882,   882,   375,   374,   807,   868,   838,   881,   791,   -463,
    867,   822,   368,   263,   852,   837,   836,   -543,  610,   610,   550,   550,   352,   336,   534,   534,
    865,   774,   851,   821,   850,   805,   593,   533,   579,   564,   773,   832,   578,   578,   548,   548,
    577,   577,   307,   276,   306,   291,   516,   560,   259,   259,   -250,  -2107, -2507, -2764, -2909, -2974,
    -3007, -3023, 1041,  1041,  1040,  1040,  769,   769,   769,   769,   256,   256,   256,   256,   256,   256,
    256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   -767,  -1052, -1213, -1277, -1358, -1405,
    -1469, -1535, -1550, -1582, -1614, -1647, -1662, -1694, -1726, -1759, -1774, -1807, -1822, -1854, -1886, 1565,
    -1919, -1935, -1951, -1967, 1731,  1730,  1580,  1717,  -1983, 1729,  1564,  -1999, 1548,  -2015, -2031, 1715,
    1595,  -2047, 1714,  -2063, 1610,  -2079, 1609,  -2095, 1323,  1323,  1457,  1457,  1307,  1307,  1712,  1547,
    1641,  1700,  1699,  1594,  1685,  1625,  1442,  1442,  1322,  1322,  -780,  -973,  -910,  1279,  1278,  1277,
    1262,  1276,  1261,  1275,  1215,  1260,  1229,  -959,  974,   974,   989,   989,   -943,  735,   478,   478,
    495,   463,   506,   414,   -1039, 1003,  958,   1017,  927,   942,   987,   957,   431,   476,   1272,  1167,
    1228,  -1183, 1256,  -1199, 895,   895,   941,   941,   1242,  1227,  1212,  1135,  1014,  1014,  490,   489,
    503,   487,   910,   1013,  985,   925,   863,   894,   970,   955,   1012,  847,   -1343, 831,   755,   755,
    984,   909,   428,   366,   754,   559,   -1391, 752,   486,   457,   924,   997,   698,   698,   983,   893,
    740,   740,   908,   877,   739,   739,   667,   667,   953,   938,   497,   287,   271,   271,   683,   606,
    590,   712,   726,   574,   302,   302,   738,   736,   481,   286,   526,   725,   605,   711,   636,   724,
    696,   651,   589,   681,   666,   710,   364,   467,   573,   695,   466,   466,   301,   465,   379,   379,
    709,   604,   665,   679,   316,   316,   634,   633,   436,   436,   464,   269,   424,   394,   452,   332,
    438,   363,   347,   408,   393,   448,   331,   422,   362,   407,   392,   421,   346,   406,   391,   376,
    375,   359,   1441,  1306,  -2367, 1290,  -2383, 1337,  -2399, -2415, 1426,  1321,  -2431, 1411,  1336,  -2447,
    -2463, -2479, 1169,  1169,  1049,  1049,  1424,  1289,  1412,  1352,  1319,  -2495, 1154,  1154,  1064,  1064,
    1153,  1153,  416,   390,   360,   404,   403,   389,   344,   374,   373,   343,   358,   372,   327,   357,
    342,   311,   356,   326,   1395,  1394,  1137,  1137,  1047,  1047,  1365,  1392,  1287,  1379,  1334,  1364,
    1349,  1378,  1318,  1363,  792,   792,   792,   792,   1152,  1152,  1032,  1032,  1121,  1121,  1046,  1046,
    1120,  1120,  1030,  1030,  -2895, 1106,  1061,  1104,  849,   849,   789,   789,   1091,  1076,  1029,  1090,
    1060,  1075,  833,   833,   309,   324,   532,   532,   832,   772,   818,   803,   561,   561,   531,   560,
    515,   546,   289,   274,   288,   258,   -250,  -1179, -1579, -1836, -1996, -2124, -2253, -2333, -2413, -2477,
    -2542, -2574, -2607, -2622, -2655, 1314,  1313,  1298,  1312,  1282,  785,   785,   785,   785,   1040,  1040,
    1025,  1025,  768,   768,   768,   768,   -766,  -798,  -830,  -862,  -895,  -911,  -927,  -943,  -959,  -975,
    -991,  -1007, -1023, -1039, -1055, -1070, 1724,  1647,  -1103, -1119, 1631,  1767,  1662,  1738,  1708,  1723,
    -1135, 1780,  1615,  1779,  1599,  1677,  1646,  1778,  1583,  -1151, 1777,  1567,  1737,  1692,  1765,  1722,
    1707,  1630,  1751,  1661,  1764,  1614,  1736,  1676,  1763,  1750,  1645,  1598,  1721,  1691,  1762,  1706,
    1582,  1761,  1566,  -1167, 1749,  1629,  767,   766,   751,   765,   494,   494,   735,   764,   719,   749,
    734,   763,   447,   447,   748,   718,   477,   506,   431,   491,   446,   476,   461,   505,   415,   430,
    475,   445,   504,   399,   460,   489,   414,   503,   383,   474,   429,   459,   502,   502,   746,   752,
    488,   398,   501,   473,   413,   472,   486,   271,   480,   270,   -1439, -1455, 1357,  -1471, -1487, -1503,
    1341,  1325,  -1519, 1489,  1463,  1403,  1309,  -1535, 1372,  1448,  1418,  1476,  1356,  1462,  1387,  -1551,
    1475,  1340,  1447,  1402,  1386,  -1567, 1068,  1068,  1474,  1461,  455,   380,   468,   440,   395,   425,
    410,   454,   364,   467,   466,   464,   453,   269,   409,   448,   268,   432,   1371,  1473,  1432,  1417,
    1308,  1460,  1355,  1446,  1459,  1431,  1083,  1083,  1401,  1416,  1458,  1445,  1067,  1067,  1370,  1457,
    1051,  1051,  1291,  1430,  1385,  1444,  1354,  1415,  1400,  1443,  1082,  1082,  1173,  1113,  1186,  1066,
    1185,  1050,  -1967, 1158,  1128,  1172,  1097,  1171,  1081,  -1983, 1157,  1112,  416,   266,   375,   400,
    1170,  1142,  1127,  1065,  793,   793,   1169,  1033,  1156,  1096,  1141,  1111,  1155,  1080,  1126,  1140,
    898,   898,   808,   808,   897,   897,   792,   792,   1095,  1152,  1032,  1125,  1110,  1139,  1079,  1124,
    882,   807,   838,   881,   853,   791,   -2319, 867,   368,   263,   822,   852,   837,   866,   806,   865,
    -2399, 851,   352,   262,   534,   534,   821,   836,   594,   594,   549,   549,   593,   593,   533,   533,
    848,   773,   579,   579,   564,   578,   548,   563,   276,   276,   577,   576,   306,   291,   516,   560,
    305,   305,   275,   259,   -251,  -892,  -2058, -2620, -2828, -2957, -3023, -3039, 1041,  1041,  1040,  1040,
    769,   769,   769,   769,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
    256,   256,   256,   256,   -511,  -527,  -543,  -559,  1530,  -575,  -591,  1528,  1527,  1407,  1526,  1391,
    1023,  1023,  1023,  1023,  1525,  1375,  1268,  1268,  1103,  1103,  1087,  1087,  1039,  1039,  1523,  -604,
    815,   815,   815,   815,   510,   495,   509,   479,   508,   463,   507,   447,   431,   505,   415,   399,
    -734,  -782,  1262,  -815,  1259,  1244,  -831,  1258,  1228,  -847,  -863,  1196,  -879,  1253,  987,   987,
    748,   -767,  493,   493,   462,   477,   414,   414,   686,   669,   478,   446,   461,   445,   474,   429,
    487,   458,   412,   471,   1266,  1264,  1009,  1009,  799,   799,   -1019, -1276, -1452, -1581, -1677, -1757,
    -1821, -1886, -1933, -1997, 1257,  1257,  1483,  1468,  1512,  1422,  1497,  1406,  1467,  1496,  1421,  1510,
    1134,  1134,  1225,  1225,  1466,  1451,  1374,  1405,  1252,  1252,  1358,  1480,  1164,  1164,  1251,  1251,
    1238,  1238,  1389,  1465,  -1407, 1054,  1101,  -1423, 1207,  -1439, 830,   830,   1248,  1038,  1237,  1117,
    1223,  1148,  1236,  1208,  411,   426,   395,   410,   379,   269,   1193,  1222,  1132,  1235,  1221,  1116,
    976,   976,   1192,  1162,  1177,  1220,  1131,  1191,  963,   963,   -1647, 961,   780,   -1663, 558,   558,
    994,   993,   437,   408,   393,   407,   829,   978,   813,   797,   947,   -1743, 721,   721,   377,   392,
    844,   950,   828,   890,   706,   706,   812,   859,   796,   960,   948,   843,   934,   874,   571,   571,
    -1919, 690,   555,   689,   421,   346,   539,   539,   944,   779,   918,   873,   932,   842,   903,   888,
    570,   570,   931,   917,   674,   674,   -2575, 1562,  -2591, 1609,  -2607, 1654,  1322,  1322,  1441,  1441,
    1696,  1546,  1683,  1593,  1669,  1624,  1426,  1426,  1321,  1321,  1639,  1680,  1425,  1425,  1305,  1305,
    1545,  1668,  1608,  1623,  1667,  1592,  1638,  1666,  1320,  1320,  1652,  1607,  1409,  1409,  1304,  1304,
    1288,  1288,  1664,  1637,  1395,  1395,  1335,  1335,  1622,  1636,  1394,  1394,  1319,  1319,  1606,  1621,
    1392,  1392,  1137,  1137,  1137,  1137,  345,   390,   360,   375,   404,   373,   1047,  -2751, -2767, -2783,
    1062,  1121,  1046,  -2799, 1077,  -2815, 1106,  1061,  789,   789,   1105,  1104,  263,   355,   310,   340,
    325,   354,   352,   262,   339,   324,   1091,  1076,  1029,  1090,  1060,  1075,  833,   833,   788,   788,
    1088,  1028,  818,   818,   803,   803,   561,   561,   531,   531,   816,   771,   546,   546,   289,   274,
    288,   258,   -253,  -317,  -381,  -446,  -478,  -509,  1279,  1279,  -811,  -1179, -1451, -1756, -1900, -2028,
    -2189, -2253, -2333, -2414, -2445, -2511, -2526, 1313,  1298,  -2559, 1041,  1041,  1040,  1040,  1025,  1025,
    1024,  1024,  1022,  1007,  1021,  991,   1020,  975,   1019,  959,   687,   687,   1018,  1017,  671,   671,
    655,   655,   1016,  1015,  639,   639,   758,   758,   623,   623,   757,   607,   756,   591,   755,   575,
    754,   559,   543,   543,   1009,  783,   -575,  -621,  -685,  -749,  496,   -590,  750,   749,   734,   748,
    974,   989,   1003,  958,   988,   973,   1002,  942,   987,   957,   972,   1001,  926,   986,   941,   971,
    956,   1000,  910,   985,   925,   999,   894,   970,   -1071, -1087, -1102, 1390,  -1135, 1436,  1509,  1451,
    1374,  -1151, 1405,  1358,  1480,  1420,  -1167, 1507,  1494,  1389,  1342,  1465,  1435,  1450,  1326,  1505,
    1310,  1493,  1373,  1479,  1404,  1492,  1464,  1419,  428,   443,   472,   397,   736,   526,   464,   464,
    486,   457,   442,   471,   484,   482,   1357,  1449,  1434,  1478,  1388,  1491,  1341,  1490,  1325,  1489,
    1463,  1403,  1309,  1477,  1372,  1448,  1418,  1433,  1476,  1356,  1462,  1387,  -1439, 1475,  1340,  1447,
    1402,  1474,  1324,  1461,  1371,  1473,  269,   448,   1432,  1417,  1308,  1460,  -1711, 1459,  -1727, 1441,
    1099,  1099,  1446,  1386,  1431,  1401,  -1743, 1289,  1083,  1083,  1160,  1160,  1458,  1445,  1067,  1067,
    1370,  1457,  1307,  1430,  1129,  1129,  1098,  1098,  268,   432,   267,   416,   266,   400,   -1887, 1144,
    1187,  1082,  1173,  1113,  1186,  1066,  1050,  1158,  1128,  1143,  1172,  1097,  1171,  1081,  420,   391,
    1157,  1112,  1170,  1142,  1127,  1065,  1169,  1049,  1156,  1096,  1141,  1111,  1155,  1080,  1126,  1154,
    1064,  1153,  1140,  1095,  1048,  -2159, 1125,  1110,  1137,  -2175, 823,   823,   1139,  1138,  807,   807,
    384,   264,   368,   263,   868,   838,   853,   791,   867,   822,   852,   837,   866,   806,   865,   790,
    -2319, 851,   821,   836,   352,   262,   850,   805,   849,   -2399, 533,   533,   835,   820,   336,   261,
    578,   548,   563,   577,   532,   532,   832,   772,   562,   562,   547,   547,   305,   275,   560,   515,
    290,   290,   288,   258,
};

/// `tab32` (miniaudio/miniaudio.h:93387 @5266f24da).
const tab32 = [28]u8{
    130, 162, 193, 209, 44,  28, 76,  140, 9,   9,  9,   9,   9, 9, 9, 9,
    190, 254, 222, 238, 126, 94, 157, 157, 109, 61, 173, 205,
};

/// `tab33` (miniaudio/miniaudio.h:93388 @5266f24da).
const tab33 = [16]u8{
    252, 236, 220, 204, 188, 172, 156, 140, 124, 108, 92, 76, 60, 44, 28, 12,
};

/// `tabindex` (miniaudio/miniaudio.h:93389 @5266f24da).
const tabindex = [32]i16{
    0,    32,   64,   98,   0,    132,  180,  218,  292,  364,  426,  538,  648,  746,  0,    1126,
    1460, 1460, 1460, 1460, 1460, 1460, 1460, 1460, 1842, 1842, 1842, 1842, 1842, 1842, 1842, 1842,
};

/// `g_linbits` (miniaudio/miniaudio.h:93390 @5266f24da).
const g_linbits = [32]u8{
    0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0,  0,
    1, 2, 3, 4, 6, 8, 10, 13, 4, 5, 6, 7, 8, 9, 11, 13,
};

/// `g_pan` (miniaudio/miniaudio.h:93554 @5266f24da).
const g_pan = [14]f32{
    0.0, 1.0, 0.21132487, 0.78867513, 0.36602540, 0.63397460, 0.5, 0.5, 0.63397460, 0.36602540, 0.78867513, 0.21132487, 1.0, 0.0,
};

/// `g_aa` (miniaudio/miniaudio.h:93619 @5266f24da).
const g_aa = [16]f32{
    0.85749293, 0.88174200, 0.94962865, 0.98331459, 0.99551782, 0.99916056, 0.99989920, 0.99999316, 0.51449576, 0.47173197, 0.31337745, 0.18191320, 0.09457419, 0.04096558, 0.01419856, 0.00369997,
};

/// `g_twid9` (miniaudio/miniaudio.h:93686 @5266f24da).
const g_twid9 = [18]f32{
    0.73727734, 0.79335334, 0.84339145, 0.88701083, 0.92387953, 0.95371695, 0.97629601, 0.99144486, 0.99904822, 0.67559021, 0.60876143, 0.53729961, 0.46174861, 0.38268343, 0.30070580, 0.21643961,
    0.13052619, 0.04361938,
};

/// `g_twid3` (miniaudio/miniaudio.h:93745 @5266f24da).
const g_twid3 = [6]f32{
    0.79335334, 0.92387953, 0.99144486, 0.60876143, 0.38268343, 0.13052619,
};

/// `g_mdct_window` (miniaudio/miniaudio.h:93781 @5266f24da).
const g_mdct_window = [36]f32{
    0.99904822, 0.99144486, 0.97629601, 0.95371695, 0.92387953, 0.88701083, 0.84339145, 0.79335334, 0.73727734, 0.04361938, 0.13052619, 0.21643961, 0.30070580, 0.38268343, 0.46174861, 0.53729961,
    0.60876143, 0.67559021, 1.0,        1.0,        1.0,        1.0,        1.0,        1.0,        0.99144486, 0.92387953, 0.79335334, 0.0,        0.0,        0.0,        0.0,        0.0,
    0.0,        0.13052619, 0.38268343, 0.60876143,
};

/// `g_sec` (miniaudio/miniaudio.h:93852 @5266f24da).
const g_sec = [24]f32{
    10.19000816, 0.50060302, 0.50241929, 3.40760851, 0.50547093, 0.52249861, 2.05778098, 0.51544732, 0.56694406, 1.48416460, 0.53104258, 0.64682180, 1.16943991, 0.55310392, 0.78815460, 0.97256821,
    0.58293498,  1.06067765, 0.83934963, 0.62250412, 1.72244716, 0.74453628, 0.67480832, 5.10114861,
};

/// `g_win` (miniaudio/miniaudio.h:94052 @5266f24da).
const g_win = [240]f32{
    -1.0, 26.0, -31.0, 208.0, 218.0, 401.0,  -519.0,  2063.0, 2000.0, 4788.0, -5517.0, 7134.0, 5959.0,  35640.0, -39336.0, 74992.0,
    -1.0, 24.0, -35.0, 202.0, 222.0, 347.0,  -581.0,  2080.0, 1952.0, 4425.0, -5879.0, 7640.0, 5288.0,  33791.0, -41176.0, 74856.0,
    -1.0, 21.0, -38.0, 196.0, 225.0, 294.0,  -645.0,  2087.0, 1893.0, 4063.0, -6237.0, 8092.0, 4561.0,  31947.0, -43006.0, 74630.0,
    -1.0, 19.0, -41.0, 190.0, 227.0, 244.0,  -711.0,  2085.0, 1822.0, 3705.0, -6589.0, 8492.0, 3776.0,  30112.0, -44821.0, 74313.0,
    -1.0, 17.0, -45.0, 183.0, 228.0, 197.0,  -779.0,  2075.0, 1739.0, 3351.0, -6935.0, 8840.0, 2935.0,  28289.0, -46617.0, 73908.0,
    -1.0, 16.0, -49.0, 176.0, 228.0, 153.0,  -848.0,  2057.0, 1644.0, 3004.0, -7271.0, 9139.0, 2037.0,  26482.0, -48390.0, 73415.0,
    -2.0, 14.0, -53.0, 169.0, 227.0, 111.0,  -919.0,  2032.0, 1535.0, 2663.0, -7597.0, 9389.0, 1082.0,  24694.0, -50137.0, 72835.0,
    -2.0, 13.0, -58.0, 161.0, 224.0, 72.0,   -991.0,  2001.0, 1414.0, 2330.0, -7910.0, 9592.0, 70.0,    22929.0, -51853.0, 72169.0,
    -2.0, 11.0, -63.0, 154.0, 221.0, 36.0,   -1064.0, 1962.0, 1280.0, 2006.0, -8209.0, 9750.0, -998.0,  21189.0, -53534.0, 71420.0,
    -2.0, 10.0, -68.0, 147.0, 215.0, 2.0,    -1137.0, 1919.0, 1131.0, 1692.0, -8491.0, 9863.0, -2122.0, 19478.0, -55178.0, 70590.0,
    -3.0, 9.0,  -73.0, 139.0, 208.0, -29.0,  -1210.0, 1870.0, 970.0,  1388.0, -8755.0, 9935.0, -3300.0, 17799.0, -56778.0, 69679.0,
    -3.0, 8.0,  -79.0, 132.0, 200.0, -57.0,  -1283.0, 1817.0, 794.0,  1095.0, -8998.0, 9966.0, -4533.0, 16155.0, -58333.0, 68692.0,
    -4.0, 7.0,  -85.0, 125.0, 189.0, -83.0,  -1356.0, 1759.0, 605.0,  814.0,  -9219.0, 9959.0, -5818.0, 14548.0, -59838.0, 67629.0,
    -4.0, 7.0,  -91.0, 117.0, 177.0, -106.0, -1428.0, 1698.0, 402.0,  545.0,  -9416.0, 9916.0, -7154.0, 12980.0, -61289.0, 66494.0,
    -5.0, 6.0,  -97.0, 111.0, 163.0, -127.0, -1498.0, 1634.0, 185.0,  288.0,  -9585.0, 9838.0, -8540.0, 11455.0, -62684.0, 65290.0,
};

/// `g_hz` (miniaudio/miniaudio.h:92947 @5266f24da).
const g_hz = [3]u32{
    44100, 48000, 32000,
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a buffer with no frame opens nothing" {
    try std.testing.expect(Mp3.init(std.testing.allocator, .{ .data = "ID3\x04\x00\x00\x00\x00\x00\x00 not an mp3 at all", .mode = .dr_mp3 }) == null);
}
