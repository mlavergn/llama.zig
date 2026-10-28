//! The FLAC decoder miniaudio's `ma_decoder` tries second: dr_flac, as
//! miniaudio embeds it, native FLAC and Ogg-FLAC.
//!
//! # Provenance
//!
//! Ports the reading half of `ma_dr_flac` in
//! `llama.cpp/vendor/miniaudio/miniaudio.h` (miniaudio v0.11.25, dr_flac
//! embedded) at v0.4.0 (`5266f24da`). Each declaration names the function it
//! ports and the line it began at.
//!
//! # What is reachable
//!
//! `ma_decoder` opens dr_flac with no metadata callback and no container
//! hint, so `ma_dr_flac__init_private` takes a stream that, after any
//! ID3v2 tags, starts `fLaC` or `OggS`. **Ogg-FLAC is reachable from the
//! helper**: its sniff accepts an `ID3` prefix, dr_wav refuses the file, and
//! dr_flac skips the tag. No seek is ever asked for, so the seek-table,
//! binary-search and brute-force seekers are not here; the seek table is
//! still read at open, because whether that read succeeds decides where
//! decoding starts.
//!
//! # As `mtmd-helper.cpp` compiles it
//!
//! On aarch64 the residual decoder and the stereo `f32` converters take
//! their NEON arms (`MA_DR_FLAC_SUPPORT_NEON`), and on x86_64 the converters
//! take SSE2's. The residual arms are integer arithmetic equal to the scalar
//! one -- 32-bit sums wrap the same way in any order, and the 64-bit arm's
//! products are exact -- so the scalar arm is ported. The converters are
//! not equal: for 24 bits or fewer the SIMD arms shift to a 24-bit scale
//! and multiply by 2^-23, where the scalar arm (taken above 24 bits) shifts
//! to 32 bits; at 24 bits the two disagree on mid-side's low bit. Both are
//! ported, chosen by `bitsPerSample` as the C chooses.
//!
//! The decoded-samples buffer persists across frames: a subframe whose
//! samples fail to decode is still reported decoded (`decode_subframe`
//! ignores the result), and a frame whose CRC then matches by chance plays
//! what the buffer held. It starts zeroed here; the reference harness
//! allocates through `calloc` to match.

const std = @import("std");
const Io = @import("audio_io.zig").Io;
const Origin = @import("audio_io.zig").Origin;

/// Set when a conversion shifted a sample by 32 or more, which the C
/// leaves undefined (`u32 << 32`); the samples are then not comparable.
pub threadlocal var read_indeterminate: bool = false;

const buffer_size = 4096; // MA_DR_FLAC_BUFFER_SIZE
const cache_bits = 64; // MA_64BIT: ma_dr_flac_cache_t is 64 bits
const cache_bytes = 8;
const l2_line_count = buffer_size / cache_bytes;
const max_simd_vector_size = 64; // MA_DR_FLAC_MAX_SIMD_VECTOR_SIZE
const ogg_max_page_size = 65307;
const ogg_capture_pattern_crc32: u32 = 1605413199;
const seekpoint_size_in_bytes = 18;

const subframe_constant = 0;
const subframe_verbatim = 1;
const subframe_fixed = 8;
const subframe_lpc = 32;
const subframe_reserved = 255;
const assignment_independent = 0;
const assignment_left_side = 8;
const assignment_right_side = 9;
const assignment_mid_side = 10;

// -----------------------------------------------------------------------------
// The byte source: the memory stream, or the Ogg layer over it

/// `ma_dr_flac_oggbs` (miniaudio.h:89310 @5266f24da).
const OggBs = struct {
    current_byte_pos: u64 = 0,
    first_byte_pos: u64 = 0,
    serial_number: u32 = 0,
    bytes_remaining_in_page: u32 = 0,
    page_data_size: u32 = 0,
    page_data: [ogg_max_page_size]u8 = undefined,
};

/// `ma_dr_flac_ogg_page_header`.
const PageHeader = struct {
    header_type: u8 = 0,
    serial_number: u32 = 0,
    checksum: u32 = 0,
    segment_count: u8 = 0,
    segment_table: [255]u8 = undefined,

    /// Ports `ma_dr_flac_ogg__get_page_body_size` (miniaudio.h:89218 @5266f24da).
    fn bodySize(h: *const PageHeader) u32 {
        var n: u32 = 0;
        for (h.segment_table[0..h.segment_count]) |s| n += s;
        return n;
    }
};

/// What the bit stream reads from: `onRead`/`onSeek` over the memory stream,
/// or `ma_dr_flac__on_read_ogg`/`ma_dr_flac__on_seek_ogg` over the Ogg layer.
const Stream = struct {
    io: Io,
    ogg: ?*OggBs = null,

    fn read(s: *Stream, buf: []u8) usize {
        if (s.ogg) |o| return onReadOgg(s, o, buf);
        return s.io.read(buf);
    }

    fn seek(s: *Stream, offset: i32, origin: Origin) bool {
        if (s.ogg) |o| return onSeekOgg(s, o, offset, origin);
        return s.io.seek(offset, origin);
    }
};

/// Ports `ma_dr_flac_crc32_byte` (miniaudio.h:89177 @5266f24da).
fn crc32Byte(crc: u32, data: u8) u32 {
    return (crc << 8) ^ crc32_table[@as(u8, @truncate(crc >> 24)) ^ data];
}

/// Ports `ma_dr_flac_ogg__read_page_header_after_capture_pattern` (miniaudio.h:89227 @5266f24da).
fn readPageHeaderAfterCapture(io: *Io, h: *PageHeader, bytes_read: *u32, crc: *u32) bool {
    var data: [23]u8 = undefined;
    if (io.read(&data) != 23) return false;
    bytes_read.* += 23;
    h.header_type = data[1];
    h.serial_number = std.mem.readInt(u32, data[10..14], .little);
    h.checksum = std.mem.readInt(u32, data[18..22], .little);
    h.segment_count = data[22];
    @memset(data[18..22], 0);
    for (data) |b| crc.* = crc32Byte(crc.*, b);
    if (io.read(h.segment_table[0..h.segment_count]) != h.segment_count) return false;
    bytes_read.* += h.segment_count;
    for (h.segment_table[0..h.segment_count]) |b| crc.* = crc32Byte(crc.*, b);
    return true;
}

/// Ports `ma_dr_flac_ogg__read_page_header` (miniaudio.h:89263 @5266f24da):
/// scan byte by byte for the capture pattern.
fn readPageHeader(io: *Io, h: *PageHeader, bytes_read: *u32, crc: *u32) bool {
    var id: [4]u8 = undefined;
    bytes_read.* = 0;
    if (io.read(&id) != 4) return false;
    bytes_read.* += 4;
    while (true) {
        if (std.mem.eql(u8, &id, "OggS")) {
            crc.* = ogg_capture_pattern_crc32;
            return readPageHeaderAfterCapture(io, h, bytes_read, crc);
        }
        id[0] = id[1];
        id[1] = id[2];
        id[2] = id[3];
        if (io.read(id[3..4]) != 1) return false;
        bytes_read.* += 1;
    }
}

/// Ports `ma_dr_flac_oggbs__seek_physical` (miniaudio.h:89317 @5266f24da).
fn seekPhysical(s: *Stream, o: *OggBs, offset_in: u64, origin: Origin) bool {
    var offset = offset_in;
    if (origin == .set) {
        if (offset <= 0x7FFFFFFF) {
            if (!s.io.seek(@intCast(offset), .set)) return false;
            o.current_byte_pos = offset;
            return true;
        }
        if (!s.io.seek(0x7FFFFFFF, .set)) return false;
        o.current_byte_pos = offset;
        return seekPhysical(s, o, offset - 0x7FFFFFFF, .cur);
    }
    while (offset > 0x7FFFFFFF) {
        if (!s.io.seek(0x7FFFFFFF, .cur)) return false;
        o.current_byte_pos += 0x7FFFFFFF;
        offset -= 0x7FFFFFFF;
    }
    if (!s.io.seek(@intCast(offset), .cur)) return false;
    o.current_byte_pos += offset;
    return true;
}

/// Ports `ma_dr_flac_oggbs__goto_next_page` (miniaudio.h:89348 @5266f24da).
fn gotoNextPage(s: *Stream, o: *OggBs, recover: bool) bool {
    var h: PageHeader = .{};
    while (true) {
        var crc: u32 = 0;
        var bytes_read: u32 = 0;
        if (!readPageHeader(&s.io, &h, &bytes_read, &crc)) return false;
        o.current_byte_pos += bytes_read;
        const body = h.bodySize();
        if (body > ogg_max_page_size) continue;
        if (h.serial_number != o.serial_number) {
            if (body > 0 and !seekPhysical(s, o, body, .cur)) return false;
            continue;
        }
        const n = s.io.read(o.page_data[0..body]);
        o.current_byte_pos += n;
        if (n != body) return false;
        o.page_data_size = body;
        var actual = crc;
        for (o.page_data[0..body]) |b| actual = crc32Byte(actual, b);
        if (actual != h.checksum) {
            if (recover) continue;
            _ = gotoNextPage(s, o, true);
            return false;
        }
        o.bytes_remaining_in_page = body;
        return true;
    }
}

/// Ports `ma_dr_flac__on_read_ogg` (miniaudio.h:89448 @5266f24da).
fn onReadOgg(s: *Stream, o: *OggBs, buf: []u8) usize {
    var read: usize = 0;
    while (read < buf.len) {
        const want = buf.len - read;
        const at = o.page_data_size - o.bytes_remaining_in_page;
        if (o.bytes_remaining_in_page >= want) {
            @memcpy(buf[read..][0..want], o.page_data[at..][0..want]);
            read += want;
            o.bytes_remaining_in_page -= @intCast(want);
            break;
        }
        if (o.bytes_remaining_in_page > 0) {
            @memcpy(buf[read..][0..o.bytes_remaining_in_page], o.page_data[at..][0..o.bytes_remaining_in_page]);
            read += o.bytes_remaining_in_page;
            o.bytes_remaining_in_page = 0;
        }
        if (!gotoNextPage(s, o, true)) break;
    }
    return read;
}

/// Ports `ma_dr_flac__on_seek_ogg` (miniaudio.h:89476 @5266f24da).
fn onSeekOgg(s: *Stream, o: *OggBs, offset: i32, origin: Origin) bool {
    switch (origin) {
        .set => {
            // `(int)oggbs->firstBytePos`, widened back to `ma_uint64`
            const first: i32 = @truncate(@as(i64, @bitCast(o.first_byte_pos)));
            if (!seekPhysical(s, o, @bitCast(@as(i64, first)), .set)) return false;
            if (!gotoNextPage(s, o, false)) return false;
            return onSeekOgg(s, o, offset, .cur);
        },
        .cur => {
            var seeked: i32 = 0;
            while (seeked < offset) {
                const remaining = offset - seeked;
                if (o.bytes_remaining_in_page >= @as(u32, @bitCast(remaining))) {
                    o.bytes_remaining_in_page -= @bitCast(remaining);
                    break;
                }
                if (o.bytes_remaining_in_page > 0) {
                    seeked += @intCast(o.bytes_remaining_in_page);
                    o.bytes_remaining_in_page = 0;
                }
                if (!gotoNextPage(s, o, false)) return false;
            }
            return true;
        },
        .end => return false,
    }
}

// -----------------------------------------------------------------------------
// The bit stream

/// `ma_dr_flac_bs` (miniaudio.h:62548 @5266f24da).
const Bs = struct {
    stream: *Stream,
    unaligned_byte_count: usize = 0,
    /// The raw bytes of `unalignedCache`.
    unaligned_cache: [cache_bytes]u8 = @splat(0),
    next_l2_line: u32 = l2_line_count,
    consumed_bits: u32 = cache_bits,
    /// `cacheL2` as the bytes the read callback filled.
    cache_l2: [buffer_size]u8 = @splat(0),
    cache: u64 = 0,
    crc16: u16 = 0,
    crc16_cache: u64 = 0,
    crc16_cache_ignored_bytes: u32 = 0,

    fn remaining(bs: *const Bs) u32 {
        return cache_bits - bs.consumed_bits;
    }

    /// `ma_dr_flac__be2host__cache_line(bs->cacheL2[i])`.
    fn line(bs: *const Bs, i: usize) u64 {
        return std.mem.readInt(u64, bs.cache_l2[i * 8 ..][0..8], .big);
    }
};

/// `MA_DR_FLAC_CACHE_L1_SELECTION_MASK`.
fn selMask(n: u32) u64 {
    if (n == 0) return 0;
    return ~(~@as(u64, 0) >> @intCast(n));
}

/// `MA_DR_FLAC_CACHE_L1_SELECT_AND_SHIFT`: for a zero count the C shifts
/// by 64, of a value already masked to zero.
fn selectAndShift(cache: u64, n: u32) u64 {
    if (n == 0) return 0;
    return (cache & selMask(n)) >> @intCast(cache_bits - n);
}

fn shl64(x: u64, n: u32) u64 {
    return if (n >= 64) 0 else x << @intCast(n);
}

/// Ports `ma_dr_flac_crc8` (miniaudio.h:85336 @5266f24da).
fn crc8(crc_in: u8, data: u32, count: u32) u8 {
    var crc = crc_in;
    const whole = count >> 3;
    const left: u5 = @intCast(count - whole * 8);
    const mask: u32 = (@as(u32, 1) << left) - 1;
    var k: u32 = whole;
    while (k > 0) : (k -= 1) {
        const shift: u5 = @intCast((k - 1) * 8 + left);
        crc = crc8_table[crc ^ @as(u8, @truncate((data & (@as(u32, 0xFF) << shift)) >> shift))];
    }
    if (left > 0) crc = @as(u8, @truncate(@as(u32, crc) << left)) ^ crc8_table[@as(u8, @truncate(@as(u32, crc) >> @intCast(8 - @as(u32, left)))) ^ @as(u8, @truncate(data & mask))];
    return crc;
}

/// Ports `ma_dr_flac_crc16_byte` and `ma_dr_flac_crc16_bytes` (miniaudio.h:85377, 85395 @5266f24da).
fn crc16Bytes(crc_in: u16, data: u64, count: u32) u16 {
    var crc = crc_in;
    if (count == 0 or count > 8) return crc;
    var k: u32 = count;
    while (k > 0) : (k -= 1) {
        const b: u8 = @truncate(data >> @intCast((k - 1) * 8));
        crc = (crc << 8) ^ crc16_table[@as(u8, @truncate(crc >> 8)) ^ b];
    }
    return crc;
}

/// Ports `ma_dr_flac__reset_crc16` (miniaudio.h:85514 @5266f24da).
fn resetCrc16(bs: *Bs) void {
    bs.crc16 = 0;
    bs.crc16_cache_ignored_bytes = bs.consumed_bits >> 3;
}

/// Ports `ma_dr_flac__update_crc16` (miniaudio.h:85519 @5266f24da).
fn updateCrc16(bs: *Bs) void {
    if (bs.crc16_cache_ignored_bytes == 0) {
        bs.crc16 = crc16Bytes(bs.crc16, bs.crc16_cache, 8);
    } else {
        bs.crc16 = crc16Bytes(bs.crc16, bs.crc16_cache, cache_bytes -% bs.crc16_cache_ignored_bytes);
        bs.crc16_cache_ignored_bytes = 0;
    }
}

/// Ports `ma_dr_flac__flush_crc16` (miniaudio.h:85528 @5266f24da).
fn flushCrc16(bs: *Bs) u16 {
    if (bs.remaining() == 0) {
        updateCrc16(bs);
    } else {
        bs.crc16 = crc16Bytes(bs.crc16, bs.crc16_cache >> @intCast(bs.remaining()), (bs.consumed_bits >> 3) -% bs.crc16_cache_ignored_bytes);
        bs.crc16_cache_ignored_bytes = bs.consumed_bits >> 3;
    }
    return bs.crc16;
}

/// Ports `ma_dr_flac__reload_l1_cache_from_l2` (miniaudio.h:85540 @5266f24da):
/// `bs.cache` is left raw, as the C leaves it.
fn reloadL1FromL2(bs: *Bs) bool {
    if (bs.next_l2_line < l2_line_count) {
        bs.cache = std.mem.readInt(u64, bs.cache_l2[bs.next_l2_line * 8 ..][0..8], .native);
        bs.next_l2_line += 1;
        return true;
    }
    if (bs.unaligned_byte_count > 0) return false;
    const n = bs.stream.read(&bs.cache_l2);
    bs.next_l2_line = 0;
    if (n == buffer_size) {
        bs.cache = std.mem.readInt(u64, bs.cache_l2[0..8], .native);
        bs.next_l2_line = 1;
        return true;
    }
    const aligned = n / cache_bytes;
    bs.unaligned_byte_count = n - aligned * cache_bytes;
    if (bs.unaligned_byte_count > 0) @memcpy(&bs.unaligned_cache, bs.cache_l2[aligned * 8 ..][0..8]);
    if (aligned > 0) {
        const offset = l2_line_count - aligned;
        var i = aligned;
        while (i > 0) : (i -= 1) @memcpy(bs.cache_l2[(i - 1 + offset) * 8 ..][0..8], bs.cache_l2[(i - 1) * 8 ..][0..8]);
        bs.next_l2_line = @intCast(offset);
        bs.cache = std.mem.readInt(u64, bs.cache_l2[bs.next_l2_line * 8 ..][0..8], .native);
        bs.next_l2_line += 1;
        return true;
    }
    bs.next_l2_line = l2_line_count;
    return false;
}

/// `ma_dr_flac__be2host__cache_line` of a raw cache value.
fn be2host(raw: u64) u64 {
    return std.mem.bigToNative(u64, raw);
}

/// Ports `ma_dr_flac__reload_cache` (miniaudio.h:85576 @5266f24da).
fn reloadCache(bs: *Bs) bool {
    updateCrc16(bs);
    if (reloadL1FromL2(bs)) {
        bs.cache = be2host(bs.cache);
        bs.consumed_bits = 0;
        bs.crc16_cache = bs.cache;
        return true;
    }
    const n = bs.unaligned_byte_count;
    if (n == 0) {
        bs.consumed_bits = cache_bits;
        return false;
    }
    bs.consumed_bits = @intCast((cache_bytes - n) * 8);
    bs.cache = std.mem.readInt(u64, &bs.unaligned_cache, .big);
    bs.cache &= selMask(bs.remaining());
    bs.unaligned_byte_count = 0;
    bs.crc16_cache = bs.cache >> @intCast(bs.consumed_bits);
    bs.crc16_cache_ignored_bytes = bs.consumed_bits >> 3;
    return true;
}

/// Ports `ma_dr_flac__reset_cache` (miniaudio.h:85606 @5266f24da).
fn resetCache(bs: *Bs) void {
    bs.next_l2_line = l2_line_count;
    bs.consumed_bits = cache_bits;
    bs.cache = 0;
    bs.unaligned_byte_count = 0;
    bs.unaligned_cache = @splat(0);
    bs.crc16_cache = 0;
    bs.crc16_cache_ignored_bytes = 0;
}

/// Ports `ma_dr_flac__read_uint32` (miniaudio.h:85618 @5266f24da), the
/// 64-bit arm.
fn readU32(bs: *Bs, bit_count: u32) ?u32 {
    if (bs.consumed_bits == cache_bits) {
        if (!reloadCache(bs)) return null;
    }
    if (bit_count <= bs.remaining()) {
        const r: u32 = @truncate(selectAndShift(bs.cache, bit_count));
        bs.consumed_bits += bit_count;
        bs.cache = shl64(bs.cache, bit_count);
        return r;
    }
    const hi_count = bs.remaining();
    const lo_count = bit_count - hi_count;
    const hi: u32 = @truncate(selectAndShift(bs.cache, hi_count));
    if (!reloadCache(bs)) return null;
    if (lo_count > bs.remaining()) return null;
    const r = (hi << @intCast(lo_count)) | @as(u32, @truncate(selectAndShift(bs.cache, lo_count)));
    bs.consumed_bits += lo_count;
    bs.cache = shl64(bs.cache, lo_count);
    return r;
}

/// Ports `ma_dr_flac__read_int32` (miniaudio.h:85665 @5266f24da).
fn readI32(bs: *Bs, bit_count: u32) ?i32 {
    var r = readU32(bs, bit_count) orelse return null;
    if (bit_count < 32) {
        // for a zero count the C shifts by -1, of a zero result
        const signbit: u32 = if (bit_count == 0) 0 else (r >> @intCast(bit_count - 1)) & 1;
        r |= (~signbit +% 1) << @intCast(bit_count);
    }
    return @bitCast(r);
}

/// Ports `ma_dr_flac__read_uint64` (miniaudio.h:85684 @5266f24da).
fn readU64(bs: *Bs, bit_count: u32) ?u64 {
    const hi = readU32(bs, bit_count - 32) orelse return null;
    const lo = readU32(bs, 32) orelse return null;
    return (@as(u64, hi) << 32) | lo;
}

/// Ports `ma_dr_flac__read_uint16` and `ma_dr_flac__read_uint8`
/// (miniaudio.h:85715, 85743 @5266f24da).
fn readU8(bs: *Bs, bit_count: u32) ?u8 {
    return @truncate(readU32(bs, bit_count) orelse return null);
}

/// Ports `ma_dr_flac__seek_bits` (miniaudio.h:85769 @5266f24da).
fn seekBits(bs: *Bs, bits_in: usize) bool {
    var bits = bits_in;
    if (bits <= bs.remaining()) {
        bs.consumed_bits += @intCast(bits);
        bs.cache = shl64(bs.cache, @intCast(bits));
        return true;
    }
    bits -= bs.remaining();
    bs.consumed_bits += bs.remaining();
    bs.cache = 0;
    while (bits >= cache_bits) {
        _ = readU64(bs, cache_bits) orelse return false;
        bits -= cache_bits;
    }
    while (bits >= 8) {
        _ = readU8(bs, 8) orelse return false;
        bits -= 8;
    }
    if (bits > 0) _ = readU8(bs, @intCast(bits)) orelse return false;
    return true;
}

/// Ports `ma_dr_flac__find_and_seek_to_next_sync_code` (miniaudio.h:85814 @5266f24da).
fn findAndSeekToNextSyncCode(bs: *Bs) bool {
    if (!seekBits(bs, bs.remaining() & 7)) return false;
    while (true) {
        resetCrc16(bs);
        const hi = readU8(bs, 8) orelse return false;
        if (hi == 0xFF) {
            const lo = readU8(bs, 6) orelse return false;
            if (lo == 0x3E) return true;
            if (!seekBits(bs, bs.remaining() & 7)) return false;
        }
    }
}

/// Ports `ma_dr_flac__seek_past_next_set_bit` (miniaudio.h:86008 @5266f24da).
fn seekPastNextSetBit(bs: *Bs) ?u32 {
    var zero_counter: u32 = 0;
    while (bs.cache == 0) {
        zero_counter +%= bs.remaining();
        if (!reloadCache(bs)) return null;
    }
    if (bs.cache == 1) {
        const off = zero_counter +% bs.remaining() -% 1;
        if (!reloadCache(bs)) return null;
        return off;
    }
    const plus1: u32 = @as(u32, @clz(bs.cache)) + 1;
    if (plus1 > bs.remaining()) return null;
    bs.consumed_bits += plus1;
    bs.cache = shl64(bs.cache, plus1);
    return zero_counter +% plus1 -% 1;
}

const Utf8 = union(enum) { ok: u64, at_end, bad };

/// Ports `ma_dr_flac__read_utf8_coded_number` (miniaudio.h:86064 @5266f24da).
fn readUtf8CodedNumber(bs: *Bs, crc_out: *u8) Utf8 {
    var crc = crc_out.*;
    const b0 = readU8(bs, 8) orelse return .at_end;
    crc = crc8(crc, b0, 8);
    if (b0 & 0x80 == 0) {
        crc_out.* = crc;
        return .{ .ok = b0 };
    }
    const byte_count: u32 = if (b0 & 0xE0 == 0xC0) 2 else if (b0 & 0xF0 == 0xE0) 3 else if (b0 & 0xF8 == 0xF0) 4 else if (b0 & 0xFC == 0xF8) 5 else if (b0 & 0xFE == 0xFC) 6 else if (b0 == 0xFE) 7 else return .bad;
    var result: u64 = b0 & (@as(u8, 0xFF) >> @intCast(byte_count + 1));
    for (1..byte_count) |_| {
        const b = readU8(bs, 8) orelse return .at_end;
        crc = crc8(crc, b, 8);
        result = (result << 6) | (b & 0x3F);
    }
    crc_out.* = crc;
    return .{ .ok = result };
}

// -----------------------------------------------------------------------------
// Residuals and subframes

/// Ports `ma_dr_flac__ilog2_u32` and `ma_dr_flac__use_64_bit_prediction`
/// (miniaudio.h:86115, 86126 @5266f24da).
fn use64BitPrediction(bps: u32, order: u32, precision: u32) bool {
    const ilog2: u32 = 32 - @as(u32, @clz(order));
    return bps +% precision +% ilog2 > 32;
}

/// Ports `ma_dr_flac__calculate_prediction_32` (miniaudio.h:86133 @5266f24da):
/// the C's `int` sum, which a corrupt stream overflows, wraps.
fn prediction32(order: u32, shift: i32, coefficients: []const i32, s: []const i32, n: usize) i32 {
    var p: i32 = 0;
    var j: usize = order;
    while (j > 0) : (j -= 1) p +%= coefficients[j - 1] *% s[n - j];
    return p >> @intCast(shift);
}

/// Ports `ma_dr_flac__calculate_prediction_64` (miniaudio.h:86174 @5266f24da), the `MA_64BIT` arm.
fn prediction64(order: u32, shift: i32, coefficients: []const i32, s: []const i32, n: usize) i32 {
    var p: i64 = 0;
    var j: usize = order;
    while (j > 0) : (j -= 1) p +%= @as(i64, coefficients[j - 1]) * s[n - j];
    return @truncate(p >> @intCast(shift));
}

/// Ports `ma_dr_flac__read_rice_parts_x1` (miniaudio.h:86470 @5266f24da).
fn readRiceParts(bs: *Bs, rice_param: u8, zero_out: *u32, part_out: *u32) bool {
    const plus1: u32 = @as(u32, rice_param) + 1;
    const plus1_shift: u32 = cache_bits - plus1;
    const plus1_max_consumed: u32 = cache_bits - plus1;
    var cache = bs.cache;
    var consumed = bs.consumed_bits;
    var lz: u32 = @clz(cache);
    if (lz < 64) {
        zero_out.* = lz;
    } else {
        var zero_counter: u32 = cache_bits -% consumed;
        while (true) {
            if (bs.next_l2_line < l2_line_count) {
                updateCrc16(bs);
                cache = bs.line(bs.next_l2_line);
                bs.next_l2_line += 1;
                consumed = 0;
                bs.crc16_cache = cache;
            } else {
                if (!reloadCache(bs)) return false;
                cache = bs.cache;
                consumed = bs.consumed_bits;
            }
            lz = @clz(cache);
            zero_counter +%= lz;
            if (lz < 64) break;
        }
        zero_out.* = zero_counter;
    }
    // extract_rice_param_part:
    cache <<= @intCast(lz);
    consumed += lz;
    if (consumed <= plus1_max_consumed) {
        part_out.* = @truncate(cache >> @intCast(plus1_shift));
        cache = shl64(cache, plus1);
        consumed += plus1;
    } else {
        const part_hi: u32 = @truncate(cache >> @intCast(plus1_shift));
        const lo_count = consumed - plus1_max_consumed;
        if (bs.next_l2_line < l2_line_count) {
            updateCrc16(bs);
            cache = bs.line(bs.next_l2_line);
            bs.next_l2_line += 1;
            consumed = lo_count;
            bs.crc16_cache = cache;
        } else {
            if (!reloadCache(bs)) return false;
            if (lo_count > bs.remaining()) return false;
            cache = bs.cache;
            consumed = bs.consumed_bits + lo_count;
        }
        const part_lo: u32 = @truncate(cache >> @intCast(cache_bits - lo_count));
        part_out.* = part_hi | part_lo;
        cache = shl64(cache, lo_count);
    }
    bs.cache = cache;
    bs.consumed_bits = consumed;
    return true;
}

/// One residual: zero count and parameter part joined, then zig-zag decoded.
fn riceValue(zero: u32, part: u32, rice_param: u8, mask: u32) u32 {
    var v = part & mask;
    v |= zero << @intCast(rice_param);
    return (v >> 1) ^ (if (v & 1 != 0) @as(u32, 0xFFFFFFFF) else 0);
}

/// Ports `ma_dr_flac__decode_samples_with_residual__rice__scalar` and
/// `ma_dr_flac__decode_samples_with_residual__rice__scalar_zeroorder` (miniaudio.h:86643, 86616 @5266f24da): residuals are read
/// four at a time and written only when all four were read.
fn decodeRice(bs: *Bs, bps: u32, count: u32, rice_param: u8, order: u32, shift: i32, precision: u32, coefficients: []const i32, s: []i32, start: usize) bool {
    const mask: u32 = @truncate(~(~@as(u64, 0) << @intCast(rice_param)));
    var zero: [4]u32 = @splat(0);
    var part: [4]u32 = @splat(0);
    if (order == 0) {
        for (0..count) |i| {
            if (!readRiceParts(bs, rice_param, &zero[0], &part[0])) return false;
            s[start + i] = @bitCast(riceValue(zero[0], part[0], rice_param, mask));
        }
        return true;
    }
    const wide = use64BitPrediction(bps, order, precision);
    const end4 = count & ~@as(u32, 3);
    var n: usize = start;
    while (n < start + end4) : (n += 4) {
        for (0..4) |k| if (!readRiceParts(bs, rice_param, &zero[k], &part[k])) return false;
        for (0..4) |k| {
            const r = riceValue(zero[k], part[k], rice_param, mask);
            const p = if (wide) prediction64(order, shift, coefficients, s, n + k) else prediction32(order, shift, coefficients, s, n + k);
            s[n + k] = @bitCast(r +% @as(u32, @bitCast(p)));
        }
    }
    var i = end4;
    while (i < count) : (i += 1) {
        if (!readRiceParts(bs, rice_param, &zero[0], &part[0])) return false;
        const r = riceValue(zero[0], part[0], rice_param, mask);
        const p = if (wide) prediction64(order, shift, coefficients, s, n) else prediction32(order, shift, coefficients, s, n);
        s[n] = @bitCast(r +% @as(u32, @bitCast(p)));
        n += 1;
    }
    return true;
}

/// Ports `ma_dr_flac__decode_samples_with_residual__unencoded` (miniaudio.h:87471 @5266f24da).
fn decodeUnencoded(bs: *Bs, bps: u32, count: u32, unencoded_bps: u8, order: u32, shift: i32, precision: u32, coefficients: []const i32, s: []i32, start: usize) bool {
    for (0..count) |i| {
        const n = start + i;
        if (unencoded_bps > 0) {
            s[n] = readI32(bs, unencoded_bps) orelse return false;
        } else s[n] = 0;
        const p = if (use64BitPrediction(bps, order, precision)) prediction64(order, shift, coefficients, s, n) else prediction32(order, shift, coefficients, s, n);
        s[n] +%= p;
    }
    return true;
}

/// Ports `ma_dr_flac__decode_samples_with_residual` (miniaudio.h:87493 @5266f24da).
fn decodeResidual(bs: *Bs, bps: u32, block_size: u32, order: u32, shift: i32, precision: u32, coefficients: []const i32, s: []i32) bool {
    const method = readU8(bs, 2) orelse return false;
    if (method != 0 and method != 1) return false;
    var pos: usize = order;
    const partition_order = readU8(bs, 4) orelse return false;
    if (partition_order > 8) return false;
    const per = block_size / (@as(u32, 1) << @intCast(partition_order));
    if (per < order) return false;
    var in_partition = per - order;
    var partitions: u32 = @as(u32, 1) << @intCast(partition_order);
    while (true) {
        var rice: u8 = 0;
        if (method == 0) {
            rice = readU8(bs, 4) orelse return false;
            if (rice == 15) rice = 0xFF;
        } else {
            rice = readU8(bs, 5) orelse return false;
            if (rice == 31) rice = 0xFF;
        }
        if (rice != 0xFF) {
            if (!decodeRice(bs, bps, in_partition, rice, order, shift, precision, coefficients, s, pos)) return false;
        } else {
            const unencoded = readU8(bs, 5) orelse return false;
            if (!decodeUnencoded(bs, bps, in_partition, unencoded, order, shift, precision, coefficients, s, pos)) return false;
        }
        pos += in_partition;
        if (partitions == 1) break;
        partitions -= 1;
        if (partition_order != 0) in_partition = per;
    }
    return true;
}

/// `ma_dr_flac_subframe`.
const Subframe = struct {
    subframe_type: u8 = 0,
    wasted_bits_per_sample: u8 = 0,
    lpc_order: u8 = 0,
    /// Where the subframe's samples start in the decoded buffer; null
    /// until decoded, as the C's `pSamplesS32`.
    samples: ?usize = null,
};

/// `ma_dr_flac_frame_header`.
const FrameHeader = struct {
    pcm_frame_number: u64 = 0,
    flac_frame_number: u32 = 0,
    sample_rate: u32 = 0,
    block_size: u16 = 0,
    channel_assignment: u8 = 0,
    bits_per_sample: u8 = 0,
    crc8: u8 = 0,
};

/// `ma_dr_flac_frame`.
const Frame = struct {
    header: FrameHeader = .{},
    pcm_frames_remaining: u32 = 0,
    subframes: [8]Subframe = @splat(.{}),
};

/// Ports `ma_dr_flac__read_next_flac_frame_header` (miniaudio.h:87708 @5266f24da).
fn readNextFrameHeader(bs: *Bs, streaminfo_bps: u8, header: *FrameHeader) bool {
    const sample_rate_table = [12]u32{ 0, 88200, 176400, 192000, 8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000 };
    const bps_table = [8]u8{ 0, 8, 12, 0xFF, 16, 20, 24, 0xFF };
    while (true) {
        var crc: u8 = 0xCE;
        if (!findAndSeekToNextSyncCode(bs)) return false;
        var reserved = readU8(bs, 1) orelse return false;
        if (reserved == 1) continue;
        crc = crc8(crc, reserved, 1);
        const blocking = readU8(bs, 1) orelse return false;
        crc = crc8(crc, blocking, 1);
        const block_size = readU8(bs, 4) orelse return false;
        if (block_size == 0) continue;
        crc = crc8(crc, block_size, 4);
        const sample_rate = readU8(bs, 4) orelse return false;
        crc = crc8(crc, sample_rate, 4);
        const assignment = readU8(bs, 4) orelse return false;
        if (assignment > 10) continue;
        crc = crc8(crc, assignment, 4);
        const bps = readU8(bs, 3) orelse return false;
        if (bps == 3 or bps == 7) continue;
        crc = crc8(crc, bps, 3);
        reserved = readU8(bs, 1) orelse return false;
        if (reserved == 1) continue;
        crc = crc8(crc, reserved, 1);
        switch (readUtf8CodedNumber(bs, &crc)) {
            .at_end => return false,
            .bad => continue,
            .ok => |v| if (blocking == 1) {
                header.flac_frame_number = 0;
                header.pcm_frame_number = v;
            } else {
                header.flac_frame_number = @truncate(v);
                header.pcm_frame_number = 0;
            },
        }
        if (block_size == 1) {
            header.block_size = 192;
        } else if (block_size <= 5) {
            header.block_size = @as(u16, 576) << @intCast(block_size - 2);
        } else if (block_size == 6) {
            header.block_size = readU8(bs, 8) orelse return false;
            crc = crc8(crc, header.block_size, 8);
            header.block_size +%= 1;
        } else if (block_size == 7) {
            header.block_size = @truncate(readU32(bs, 16) orelse return false);
            crc = crc8(crc, header.block_size, 16);
            if (header.block_size == 0xFFFF) return false;
            header.block_size += 1;
        } else {
            header.block_size = @as(u16, 256) << @intCast(block_size - 8);
        }
        if (sample_rate <= 11) {
            header.sample_rate = sample_rate_table[sample_rate];
        } else if (sample_rate == 12) {
            header.sample_rate = readU32(bs, 8) orelse return false;
            crc = crc8(crc, header.sample_rate, 8);
            header.sample_rate *%= 1000;
        } else if (sample_rate == 13) {
            header.sample_rate = readU32(bs, 16) orelse return false;
            crc = crc8(crc, header.sample_rate, 16);
        } else if (sample_rate == 14) {
            header.sample_rate = readU32(bs, 16) orelse return false;
            crc = crc8(crc, header.sample_rate, 16);
            header.sample_rate *%= 10;
        } else continue;
        header.channel_assignment = assignment;
        header.bits_per_sample = bps_table[bps];
        if (header.bits_per_sample == 0) header.bits_per_sample = streaminfo_bps;
        if (header.bits_per_sample != streaminfo_bps) return false;
        header.crc8 = readU8(bs, 8) orelse return false;
        if (header.crc8 != crc) continue;
        return true;
    }
}

/// Ports `ma_dr_flac__read_subframe_header` (miniaudio.h:87861 @5266f24da).
fn readSubframeHeader(bs: *Bs, sf: *Subframe) bool {
    const header = readU8(bs, 8) orelse return false;
    if (header & 0x80 != 0) return false;
    sf.lpc_order = 0;
    const t: u8 = (header & 0x7E) >> 1;
    if (t == 0) {
        sf.subframe_type = subframe_constant;
    } else if (t == 1) {
        sf.subframe_type = subframe_verbatim;
    } else if (t & 0x20 != 0) {
        sf.subframe_type = subframe_lpc;
        sf.lpc_order = (t & 0x1F) + 1;
    } else if (t & 0x08 != 0) {
        sf.subframe_type = subframe_fixed;
        sf.lpc_order = t & 0x07;
        if (sf.lpc_order > 4) {
            sf.subframe_type = subframe_reserved;
            sf.lpc_order = 0;
        }
    } else sf.subframe_type = subframe_reserved;
    if (sf.subframe_type == subframe_reserved) return false;
    sf.wasted_bits_per_sample = 0;
    if (header & 0x01 == 1) {
        const w = seekPastNextSetBit(bs) orelse return false;
        sf.wasted_bits_per_sample = @truncate(w +% 1);
    }
    return true;
}

/// Ports `ma_dr_flac__decode_samples__constant`, `ma_dr_flac__decode_samples__verbatim`,
/// `ma_dr_flac__decode_samples__fixed` and `ma_dr_flac__decode_samples__lpc`
/// (miniaudio.h:87625, 87637, 87649, 87671 @5266f24da). The
/// result is discarded by the caller, as the C discards it.
fn decodeSamples(bs: *Bs, sf: *const Subframe, block_size: u32, bps: u32, s: []i32) void {
    switch (sf.subframe_type) {
        subframe_constant => {
            const v = readI32(bs, bps) orelse return;
            @memset(s[0..block_size], v);
        },
        subframe_verbatim => for (0..block_size) |i| {
            s[i] = readI32(bs, bps) orelse return;
        },
        subframe_fixed => {
            const table = [5][4]i32{ .{ 0, 0, 0, 0 }, .{ 1, 0, 0, 0 }, .{ 2, -1, 0, 0 }, .{ 3, -3, 1, 0 }, .{ 4, -6, 4, -1 } };
            for (0..sf.lpc_order) |i| s[i] = readI32(bs, bps) orelse return;
            _ = decodeResidual(bs, bps, block_size, sf.lpc_order, 0, 4, &table[sf.lpc_order], s);
        },
        subframe_lpc => {
            for (0..sf.lpc_order) |i| s[i] = readI32(bs, bps) orelse return;
            var precision = readU8(bs, 4) orelse return;
            if (precision == 15) return;
            precision += 1;
            const shift: i8 = @truncate(readI32(bs, 5) orelse return);
            if (shift < 0) return;
            var coefficients: [32]i32 = @splat(0);
            for (0..sf.lpc_order) |i| coefficients[i] = readI32(bs, precision) orelse return;
            _ = decodeResidual(bs, bps, block_size, sf.lpc_order, shift, precision, &coefficients, s);
        },
        else => {},
    }
}

/// Ports `ma_dr_flac__decode_subframe` (miniaudio.h:87905 @5266f24da).
fn decodeSubframe(bs: *Bs, frame: *Frame, index: usize, decoded: []i32, at: usize) bool {
    const sf = &frame.subframes[index];
    if (!readSubframeHeader(bs, sf)) return false;
    var bps: u32 = frame.header.bits_per_sample;
    const a = frame.header.channel_assignment;
    if ((a == assignment_left_side or a == assignment_mid_side) and index == 1) {
        bps += 1;
    } else if (a == assignment_right_side and index == 0) bps += 1;
    if (bps > 32) return false;
    if (sf.wasted_bits_per_sample >= bps) return false;
    bps -= sf.wasted_bits_per_sample;
    sf.samples = at;
    if (frame.header.block_size < sf.lpc_order) return false;
    decodeSamples(bs, sf, frame.header.block_size, bps, decoded[at..]);
    return true;
}

// -----------------------------------------------------------------------------
// The decoder

/// Ports `ma_dr_flac` as `ma_decoder` drives it.
pub const Flac = struct {
    gpa: std.mem.Allocator,
    stream: Stream,
    bs: Bs,
    ogg: ?*OggBs = null,
    sample_rate: u32 = 0,
    channels: u32 = 0,
    bits_per_sample: u8 = 0,
    max_block_size: u16 = 0,
    total_pcm_frame_count: u64 = 0,
    frame: Frame = .{},
    current_pcm_frame: u64 = 0,
    /// `pDecodedSamples`: one block per channel, zeroed at open.
    decoded: []i32 = &.{},

    /// Ports `ma_dr_flac_open_memory` (miniaudio.h:90113 @5266f24da) for a
    /// `dr_flac` stream, and `ma_dr_flac_open` (miniaudio.h:90163 @5266f24da)
    /// over the decoder's callbacks for a `decoder` one: both reach
    /// `ma_dr_flac_open_with_metadata_private` (miniaudio.h:89802 @5266f24da)
    /// with no metadata callback and an unknown container.
    ///
    /// Return: the decoder, owned by the caller (`deinit`), or null where
    /// the C returns NULL.
    pub fn open(gpa: std.mem.Allocator, io: Io) ?*Flac {
        if (io.data.len == 0) return null;
        const f = gpa.create(Flac) catch return null;
        f.* = .{ .gpa = gpa, .stream = .{ .io = io }, .bs = undefined };
        f.bs = .{ .stream = &f.stream };
        if (!f.openPrivate()) {
            f.deinit();
            return null;
        }
        return f;
    }

    /// Releases the decoder.
    pub fn deinit(f: *Flac) void {
        if (f.decoded.len != 0) f.gpa.free(f.decoded);
        if (f.ogg) |o| f.gpa.destroy(o);
        f.gpa.destroy(f);
    }

    fn openPrivate(f: *Flac) bool {
        const io = &f.stream.io;
        // `ma_dr_flac__init_private` (miniaudio.h:89719 @5266f24da)
        resetCache(&f.bs);
        var running_file_pos: u64 = 0;
        var id: [4]u8 = undefined;
        while (true) {
            if (io.read(&id) != 4) return false;
            running_file_pos += 4;
            if (!std.mem.eql(u8, id[0..3], "ID3")) break;
            var header: [6]u8 = undefined;
            if (io.read(&header) != 6) return false;
            running_file_pos += 6;
            const x = std.mem.readInt(u32, header[2..6], .big);
            // `ma_dr_flac__unsynchsafe_32` (miniaudio.h:85271 @5266f24da)
            var size: u32 = ((x & 0x7F000000) >> 3) | ((x & 0x007F0000) >> 2) | ((x & 0x00007F00) >> 1) | (x & 0x7F);
            if (header[1] & 0x10 != 0) size += 10;
            if (!io.seek(@as(i32, @bitCast(size)), .cur)) return false;
            running_file_pos += size;
        }
        var has_metadata_blocks = false;
        var is_ogg = false;
        var ogg_first_byte_pos: u64 = 0;
        var ogg_serial: u32 = 0;
        if (std.mem.eql(u8, &id, "fLaC")) {
            // `ma_dr_flac__init_private__native` (miniaudio.h:89050 @5266f24da), not relaxed
            const blk = readBlockHeader(io) orelse return false;
            if (blk.block_type != 0 or blk.size != 34) return false;
            if (!f.readStreaminfo(io)) return false;
            has_metadata_blocks = !blk.is_last;
        } else if (std.mem.eql(u8, &id, "OggS")) {
            if (!f.initOgg(&running_file_pos, &ogg_first_byte_pos, &ogg_serial)) return false;
            has_metadata_blocks = true;
            is_ogg = true;
        } else return false;

        if (is_ogg) {
            const o = f.gpa.create(OggBs) catch return false;
            o.* = .{ .current_byte_pos = ogg_first_byte_pos, .first_byte_pos = ogg_first_byte_pos, .serial_number = ogg_serial };
            f.ogg = o;
            f.stream.ogg = o;
        }

        var first_frame_pos: u64 = 42;
        var seektable_pos: u64 = 0;
        var seekpoint_count: u32 = 0;
        if (has_metadata_blocks) {
            if (!f.readMetadata(&first_frame_pos, &seektable_pos, &seekpoint_count)) return false;
        }

        // the decoded samples, in whole 64-byte vectors per channel
        const per_vector = max_simd_vector_size / 4;
        const vectors: usize = (@as(usize, f.max_block_size) + per_vector - 1) / per_vector;
        f.decoded = f.gpa.alloc(i32, vectors * per_vector * f.channels) catch return false;
        @memset(f.decoded, 0);

        if (f.ogg == null and seektable_pos != 0) {
            const s = &f.stream;
            if (s.seek(@bitCast(@as(u32, @truncate(seektable_pos))), .set)) {
                var point: [seekpoint_size_in_bytes]u8 = undefined;
                for (0..seekpoint_count) |_| {
                    if (s.read(&point) != point.len) break;
                }
                if (!s.seek(@bitCast(@as(u32, @truncate(first_frame_pos))), .set)) return false;
            }
        }
        return true;
    }

    const BlockHeader = struct { is_last: bool, block_type: u8, size: u32 };

    /// Ports `ma_dr_flac__read_and_decode_block_header` (miniaudio.h:88563 @5266f24da).
    fn readBlockHeader(src: anytype) ?BlockHeader {
        var b: [4]u8 = undefined;
        if (src.read(&b) != 4) return null;
        const v = std.mem.readInt(u32, &b, .big);
        return .{ .is_last = v & 0x80000000 != 0, .block_type = @truncate((v & 0x7F000000) >> 24), .size = v & 0x00FFFFFF };
    }

    /// Ports `ma_dr_flac__read_streaminfo` (miniaudio.h:88573 @5266f24da):
    /// the four reads, each of which must be whole.
    fn readStreaminfo(f: *Flac, io: *Io) bool {
        var block_sizes: [4]u8 = undefined;
        var frame_sizes: [6]u8 = undefined;
        var props: [8]u8 = undefined;
        var md5: [16]u8 = undefined;
        if (io.read(&block_sizes) != 4) return false;
        if (io.read(&frame_sizes) != 6) return false;
        if (io.read(&props) != 8) return false;
        if (io.read(&md5) != 16) return false;
        const bsz = std.mem.readInt(u32, &block_sizes, .big);
        const p = std.mem.readInt(u64, &props, .big);
        f.max_block_size = @truncate(bsz & 0xFFFF);
        f.sample_rate = @truncate(p >> 44);
        f.channels = @as(u32, @truncate((p >> 41) & 7)) + 1;
        f.bits_per_sample = @as(u8, @truncate((p >> 36) & 0x1F)) + 1;
        f.total_pcm_frame_count = p & 0xFFFFFFFFF;
        return true;
    }

    /// Ports `ma_dr_flac__init_private__ogg` (miniaudio.h:89611 @5266f24da).
    fn initOgg(f: *Flac, running_file_pos: *u64, first_byte_pos: *u64, serial: *u32) bool {
        const io = &f.stream.io;
        var h: PageHeader = .{};
        var crc: u32 = ogg_capture_pattern_crc32;
        var bytes_read: u32 = 0;
        if (!readPageHeaderAfterCapture(io, &h, &bytes_read, &crc)) return false;
        running_file_pos.* += bytes_read;
        while (true) {
            if (h.header_type & 0x02 == 0) return false;
            const body = h.bodySize();
            if (body == 51) {
                var left: u32 = body;
                var packet_type: [1]u8 = undefined;
                if (io.read(&packet_type) != 1) return false;
                left -= 1;
                if (packet_type[0] == 0x7F) {
                    var sig: [4]u8 = undefined;
                    if (io.read(&sig) != 4) return false;
                    left -= 4;
                    if (std.mem.eql(u8, &sig, "FLAC")) {
                        var mapping: [2]u8 = undefined;
                        if (io.read(&mapping) != 2) return false;
                        if (mapping[0] != 1) return false;
                        if (!io.seek(2, .cur)) return false;
                        if (io.read(&sig) != 4) return false;
                        if (!std.mem.eql(u8, &sig, "fLaC")) return false;
                        const blk = readBlockHeader(io) orelse return false;
                        if (blk.block_type != 0 or blk.size != 34) return false;
                        if (!f.readStreaminfo(io)) return false;
                        running_file_pos.* += body;
                        first_byte_pos.* = running_file_pos.* -% 79;
                        serial.* = h.serial_number;
                        return true;
                    }
                    if (!io.seek(@as(i32, @bitCast(left)), .cur)) return false;
                } else {
                    if (!io.seek(@as(i32, @bitCast(left)), .cur)) return false;
                }
            } else {
                if (!io.seek(@as(i32, @bitCast(body)), .cur)) return false;
            }
            running_file_pos.* += body;
            if (!readPageHeader(io, &h, &bytes_read, &crc)) return false;
            running_file_pos.* += bytes_read;
        }
    }

    /// Ports `ma_dr_flac__read_and_decode_metadata` (miniaudio.h:88664 @5266f24da)
    /// with no metadata callback: each block's size is checked and skipped.
    fn readMetadata(f: *Flac, first_frame_pos: *u64, seektable_pos: *u64, seekpoint_count: *u32) bool {
        const s = &f.stream;
        var running: u64 = 42;
        var st_pos: u64 = 0;
        var st_size: u32 = 0;
        while (true) {
            const blk = readBlockHeader(s) orelse return false;
            var is_last = blk.is_last;
            running += 4;
            switch (blk.block_type) {
                2 => if (blk.size < 4) return false, // APPLICATION
                3 => { // SEEKTABLE
                    st_pos = running;
                    st_size = blk.size;
                },
                4 => if (blk.size < 8) return false, // VORBIS_COMMENT
                5 => if (blk.size < 396) return false, // CUESHEET
                6 => if (blk.size < 32) return false, // PICTURE
                else => {},
            }
            if (blk.size > 0) {
                if (!s.seek(@intCast(blk.size), .cur)) is_last = true;
            }
            running += blk.size;
            if (is_last) break;
        }
        seektable_pos.* = st_pos;
        seekpoint_count.* = st_size / seekpoint_size_in_bytes;
        first_frame_pos.* = running;
        return true;
    }

    /// Ports `ma_dr_flac__decode_flac_frame` (miniaudio.h:88032 @5266f24da).
    fn decodeFrame(f: *Flac) enum { ok, err, at_end, crc_mismatch } {
        f.frame.subframes = @splat(.{});
        if (f.frame.header.block_size > f.max_block_size) return .err;
        const channel_count = channelCount(f.frame.header.channel_assignment);
        if (channel_count != f.channels) return .err;
        for (0..channel_count) |i| {
            if (!decodeSubframe(&f.bs, &f.frame, i, f.decoded, @as(usize, f.frame.header.block_size) * i)) return .err;
        }
        const padding: u32 = f.bs.remaining() & 7;
        if (padding > 0) _ = readU8(&f.bs, padding) orelse return .at_end;
        const actual = flushCrc16(&f.bs);
        const desired: u16 = @truncate(readU32(&f.bs, 16) orelse return .at_end);
        if (actual != desired) return .crc_mismatch;
        f.frame.pcm_frames_remaining = f.frame.header.block_size;
        return .ok;
    }

    /// Ports `ma_dr_flac__read_and_decode_next_flac_frame` (miniaudio.h:88105 @5266f24da).
    fn readAndDecodeNextFrame(f: *Flac) bool {
        while (true) {
            if (!readNextFrameHeader(&f.bs, f.bits_per_sample, &f.frame.header)) return false;
            switch (f.decodeFrame()) {
                .ok => return true,
                .crc_mismatch => continue,
                else => return false,
            }
        }
    }

    /// Ports `ma_dr_flac_read_pcm_frames_f32` (miniaudio.h:92266 @5266f24da).
    ///
    /// Parameters:
    /// - `frames`: the frames wanted.
    /// - `out`: at least `frames * channels` samples.
    ///
    /// Return: the frames written.
    pub fn readPcmFramesF32(f: *Flac, frames_in: u64, out: []f32) u64 {
        if (frames_in == 0) return 0;
        var frames = frames_in;
        const unused: u32 = 32 - @as(u32, f.bits_per_sample);
        var read: u64 = 0;
        var o: usize = 0;
        while (frames > 0) {
            if (f.frame.pcm_frames_remaining == 0) {
                if (!f.readAndDecodeNextFrame()) break;
                continue;
            }
            const channel_count = channelCount(f.frame.header.channel_assignment);
            const first: usize = f.frame.header.block_size - f.frame.pcm_frames_remaining;
            const n: usize = @intCast(@min(frames, f.frame.pcm_frames_remaining));
            const dst = out[o..][0 .. n * channel_count];
            if (channel_count == 2) {
                const s0 = f.decoded[f.frame.subframes[0].samples.? + first ..][0..n];
                const s1 = f.decoded[f.frame.subframes[1].samples.? + first ..][0..n];
                const w0: u32 = f.frame.subframes[0].wasted_bits_per_sample;
                const w1: u32 = f.frame.subframes[1].wasted_bits_per_sample;
                const simd = f.bits_per_sample <= 24;
                switch (f.frame.header.channel_assignment) {
                    assignment_left_side => sideToF32(.left, simd, unused, w0, w1, s0, s1, dst),
                    assignment_right_side => sideToF32(.right, simd, unused, w0, w1, s0, s1, dst),
                    assignment_mid_side => midSideToF32(simd, unused, w0, w1, s0, s1, dst),
                    else => sideToF32(.independent, simd, unused, w0, w1, s0, s1, dst),
                }
            } else {
                for (0..n) |i| {
                    for (0..channel_count) |j| {
                        const sf = &f.frame.subframes[j];
                        const shift = unused + sf.wasted_bits_per_sample;
                        const v: i32 = @bitCast(shl32(@bitCast(f.decoded[sf.samples.? + first + i]), shift));
                        dst[i * channel_count + j] = @floatCast(@as(f64, @floatFromInt(v)) / 2147483648.0);
                    }
                }
            }
            read += n;
            o += n * channel_count;
            frames -= n;
            f.current_pcm_frame += n;
            f.frame.pcm_frames_remaining -= @intCast(n);
        }
        return read;
    }
};

/// Ports `ma_dr_flac__get_channel_count_from_channel_assignment` (miniaudio.h:88026 @5266f24da).
fn channelCount(assignment: u8) u32 {
    const lookup = [11]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 2, 2, 2 };
    return lookup[assignment];
}

/// `u32 << n`, which the C leaves undefined at 32 and more.
fn shl32(x: u32, n: u32) u32 {
    if (n >= 32) {
        read_indeterminate = true;
        return 0;
    }
    return x << @intCast(n);
}

const Side = enum { left, right, independent };

/// Ports the left-side, right-side and independent `f32` converters,
/// `ma_dr_flac_read_pcm_frames_f32__decode_left_side__scalar`,
/// `ma_dr_flac_read_pcm_frames_f32__decode_left_side__neon`,
/// `ma_dr_flac_read_pcm_frames_f32__decode_right_side__scalar`,
/// `ma_dr_flac_read_pcm_frames_f32__decode_right_side__neon`,
/// `ma_dr_flac_read_pcm_frames_f32__decode_independent_stereo__scalar` and
/// `ma_dr_flac_read_pcm_frames_f32__decode_independent_stereo__neon`
/// (miniaudio.h:91619, 91689, 91758, 91828, 92156, 92218 @5266f24da): the
/// SIMD arms at 24 bits and fewer, the scalar arm above, which agree
/// between their four-frame bodies and their tails.
fn sideToF32(comptime kind: Side, simd: bool, unused: u32, w0: u32, w1: u32, s0: []const i32, s1: []const i32, out: []f32) void {
    const bias: u32 = if (simd) 8 else 0;
    const shift0 = unused + w0 - bias;
    const shift1 = unused + w1 - bias;
    const scale: f32 = if (simd) 1.0 / 8388608.0 else 1.0 / 2147483648.0;
    for (s0, s1, 0..) |a, b, i| {
        const x0 = shl32(@bitCast(a), shift0);
        const x1 = shl32(@bitCast(b), shift1);
        const l, const r = switch (kind) {
            .left => .{ x0, x0 -% x1 },
            .right => .{ x1 +% x0, x1 },
            .independent => .{ x0, x1 },
        };
        out[i * 2 + 0] = @as(f32, @floatFromInt(@as(i32, @bitCast(l)))) * scale;
        out[i * 2 + 1] = @as(f32, @floatFromInt(@as(i32, @bitCast(r)))) * scale;
    }
}

/// Ports the mid-side `f32` converters, `ma_dr_flac_read_pcm_frames_f32__decode_mid_side__scalar`
/// and `ma_dr_flac_read_pcm_frames_f32__decode_mid_side__neon` (miniaudio.h:91896, 92060 @5266f24da).
/// The scalar arm's tail writes `(mid + side) >> 1` shifted left where its
/// four-frame body writes `mid + side` shifted one place less; they agree,
/// because mid's low bit is side's and `mid + side` is always even. Both
/// forms are kept as the C has them.
fn midSideToF32(simd: bool, unused: u32, w0: u32, w1: u32, s0: []const i32, s1: []const i32, out: []f32) void {
    const n = s0.len;
    const n4 = n & ~@as(usize, 3);
    for (0..n) |i| {
        const side = shl32(@bitCast(s1[i]), w1);
        const mid = (shl32(@bitCast(s0[i]), w0) << 1) | (side & 1);
        const sum = mid +% side;
        const diff = mid -% side;
        var l: i32 = undefined;
        var r: i32 = undefined;
        var scale: f32 = undefined;
        if (simd) {
            scale = 1.0 / 8388608.0;
            const shift = unused - 8;
            if (shift == 0) {
                l = @as(i32, @bitCast(sum)) >> 1;
                r = @as(i32, @bitCast(diff)) >> 1;
            } else {
                l = @bitCast(sum << @intCast(shift - 1));
                r = @bitCast(diff << @intCast(shift - 1));
            }
        } else {
            scale = 1.0 / 2147483648.0;
            if (i < n4 and unused > 0) {
                l = @bitCast(sum << @intCast(unused - 1));
                r = @bitCast(diff << @intCast(unused - 1));
            } else if (i < n4) {
                l = @as(i32, @bitCast(sum)) >> 1;
                r = @as(i32, @bitCast(diff)) >> 1;
            } else {
                l = @bitCast(shl32(@bitCast(@as(i32, @bitCast(sum)) >> 1), unused));
                r = @bitCast(shl32(@bitCast(@as(i32, @bitCast(diff)) >> 1), unused));
            }
        }
        out[i * 2 + 0] = @as(f32, @floatFromInt(l)) * scale;
        out[i * 2 + 1] = @as(f32, @floatFromInt(r)) * scale;
    }
}

/// Ports the container sniff at the head of `ma_dr_flac__init_private`
/// (miniaudio.h:89719 @5266f24da): whether dr_flac reaches its native or Ogg
/// initialisation over either of the streams `ma_decoder_init_memory` gives
/// it. The gate uses it to count FLAC inputs apart.
pub fn claims(buf: []const u8) bool {
    return claimsOver(.{ .data = buf, .mode = .dr_flac }) or claimsOver(.{ .data = buf, .mode = .decoder });
}

fn claimsOver(io_in: Io) bool {
    var io = io_in;
    var id: [4]u8 = undefined;
    while (true) {
        if (io.read(&id) != 4) return false;
        if (!std.mem.eql(u8, id[0..3], "ID3")) break;
        var header: [6]u8 = undefined;
        if (io.read(&header) != 6) return false;
        const x = std.mem.readInt(u32, header[2..6], .big);
        var size: u32 = ((x & 0x7F000000) >> 3) | ((x & 0x007F0000) >> 2) | ((x & 0x00007F00) >> 1) | (x & 0x7F);
        if (header[1] & 0x10 != 0) size += 10;
        if (!io.seek(@as(i32, @bitCast(size)), .cur)) return false;
    }
    return std.mem.eql(u8, &id, "fLaC") or std.mem.eql(u8, &id, "OggS");
}

// -----------------------------------------------------------------------------
// tables

/// `ma_dr_flac__crc8_table` (miniaudio/miniaudio.h:85280 @5266f24da).
const crc8_table = [256]u8{
    0x00, 0x07, 0x0E, 0x09, 0x1C, 0x1B, 0x12, 0x15,
    0x38, 0x3F, 0x36, 0x31, 0x24, 0x23, 0x2A, 0x2D,
    0x70, 0x77, 0x7E, 0x79, 0x6C, 0x6B, 0x62, 0x65,
    0x48, 0x4F, 0x46, 0x41, 0x54, 0x53, 0x5A, 0x5D,
    0xE0, 0xE7, 0xEE, 0xE9, 0xFC, 0xFB, 0xF2, 0xF5,
    0xD8, 0xDF, 0xD6, 0xD1, 0xC4, 0xC3, 0xCA, 0xCD,
    0x90, 0x97, 0x9E, 0x99, 0x8C, 0x8B, 0x82, 0x85,
    0xA8, 0xAF, 0xA6, 0xA1, 0xB4, 0xB3, 0xBA, 0xBD,
    0xC7, 0xC0, 0xC9, 0xCE, 0xDB, 0xDC, 0xD5, 0xD2,
    0xFF, 0xF8, 0xF1, 0xF6, 0xE3, 0xE4, 0xED, 0xEA,
    0xB7, 0xB0, 0xB9, 0xBE, 0xAB, 0xAC, 0xA5, 0xA2,
    0x8F, 0x88, 0x81, 0x86, 0x93, 0x94, 0x9D, 0x9A,
    0x27, 0x20, 0x29, 0x2E, 0x3B, 0x3C, 0x35, 0x32,
    0x1F, 0x18, 0x11, 0x16, 0x03, 0x04, 0x0D, 0x0A,
    0x57, 0x50, 0x59, 0x5E, 0x4B, 0x4C, 0x45, 0x42,
    0x6F, 0x68, 0x61, 0x66, 0x73, 0x74, 0x7D, 0x7A,
    0x89, 0x8E, 0x87, 0x80, 0x95, 0x92, 0x9B, 0x9C,
    0xB1, 0xB6, 0xBF, 0xB8, 0xAD, 0xAA, 0xA3, 0xA4,
    0xF9, 0xFE, 0xF7, 0xF0, 0xE5, 0xE2, 0xEB, 0xEC,
    0xC1, 0xC6, 0xCF, 0xC8, 0xDD, 0xDA, 0xD3, 0xD4,
    0x69, 0x6E, 0x67, 0x60, 0x75, 0x72, 0x7B, 0x7C,
    0x51, 0x56, 0x5F, 0x58, 0x4D, 0x4A, 0x43, 0x44,
    0x19, 0x1E, 0x17, 0x10, 0x05, 0x02, 0x0B, 0x0C,
    0x21, 0x26, 0x2F, 0x28, 0x3D, 0x3A, 0x33, 0x34,
    0x4E, 0x49, 0x40, 0x47, 0x52, 0x55, 0x5C, 0x5B,
    0x76, 0x71, 0x78, 0x7F, 0x6A, 0x6D, 0x64, 0x63,
    0x3E, 0x39, 0x30, 0x37, 0x22, 0x25, 0x2C, 0x2B,
    0x06, 0x01, 0x08, 0x0F, 0x1A, 0x1D, 0x14, 0x13,
    0xAE, 0xA9, 0xA0, 0xA7, 0xB2, 0xB5, 0xBC, 0xBB,
    0x96, 0x91, 0x98, 0x9F, 0x8A, 0x8D, 0x84, 0x83,
    0xDE, 0xD9, 0xD0, 0xD7, 0xC2, 0xC5, 0xCC, 0xCB,
    0xE6, 0xE1, 0xE8, 0xEF, 0xFA, 0xFD, 0xF4, 0xF3,
};

/// `ma_dr_flac__crc16_table` (miniaudio/miniaudio.h:85298 @5266f24da).
const crc16_table = [256]u16{
    0x0000, 0x8005, 0x800F, 0x000A, 0x801B, 0x001E, 0x0014, 0x8011,
    0x8033, 0x0036, 0x003C, 0x8039, 0x0028, 0x802D, 0x8027, 0x0022,
    0x8063, 0x0066, 0x006C, 0x8069, 0x0078, 0x807D, 0x8077, 0x0072,
    0x0050, 0x8055, 0x805F, 0x005A, 0x804B, 0x004E, 0x0044, 0x8041,
    0x80C3, 0x00C6, 0x00CC, 0x80C9, 0x00D8, 0x80DD, 0x80D7, 0x00D2,
    0x00F0, 0x80F5, 0x80FF, 0x00FA, 0x80EB, 0x00EE, 0x00E4, 0x80E1,
    0x00A0, 0x80A5, 0x80AF, 0x00AA, 0x80BB, 0x00BE, 0x00B4, 0x80B1,
    0x8093, 0x0096, 0x009C, 0x8099, 0x0088, 0x808D, 0x8087, 0x0082,
    0x8183, 0x0186, 0x018C, 0x8189, 0x0198, 0x819D, 0x8197, 0x0192,
    0x01B0, 0x81B5, 0x81BF, 0x01BA, 0x81AB, 0x01AE, 0x01A4, 0x81A1,
    0x01E0, 0x81E5, 0x81EF, 0x01EA, 0x81FB, 0x01FE, 0x01F4, 0x81F1,
    0x81D3, 0x01D6, 0x01DC, 0x81D9, 0x01C8, 0x81CD, 0x81C7, 0x01C2,
    0x0140, 0x8145, 0x814F, 0x014A, 0x815B, 0x015E, 0x0154, 0x8151,
    0x8173, 0x0176, 0x017C, 0x8179, 0x0168, 0x816D, 0x8167, 0x0162,
    0x8123, 0x0126, 0x012C, 0x8129, 0x0138, 0x813D, 0x8137, 0x0132,
    0x0110, 0x8115, 0x811F, 0x011A, 0x810B, 0x010E, 0x0104, 0x8101,
    0x8303, 0x0306, 0x030C, 0x8309, 0x0318, 0x831D, 0x8317, 0x0312,
    0x0330, 0x8335, 0x833F, 0x033A, 0x832B, 0x032E, 0x0324, 0x8321,
    0x0360, 0x8365, 0x836F, 0x036A, 0x837B, 0x037E, 0x0374, 0x8371,
    0x8353, 0x0356, 0x035C, 0x8359, 0x0348, 0x834D, 0x8347, 0x0342,
    0x03C0, 0x83C5, 0x83CF, 0x03CA, 0x83DB, 0x03DE, 0x03D4, 0x83D1,
    0x83F3, 0x03F6, 0x03FC, 0x83F9, 0x03E8, 0x83ED, 0x83E7, 0x03E2,
    0x83A3, 0x03A6, 0x03AC, 0x83A9, 0x03B8, 0x83BD, 0x83B7, 0x03B2,
    0x0390, 0x8395, 0x839F, 0x039A, 0x838B, 0x038E, 0x0384, 0x8381,
    0x0280, 0x8285, 0x828F, 0x028A, 0x829B, 0x029E, 0x0294, 0x8291,
    0x82B3, 0x02B6, 0x02BC, 0x82B9, 0x02A8, 0x82AD, 0x82A7, 0x02A2,
    0x82E3, 0x02E6, 0x02EC, 0x82E9, 0x02F8, 0x82FD, 0x82F7, 0x02F2,
    0x02D0, 0x82D5, 0x82DF, 0x02DA, 0x82CB, 0x02CE, 0x02C4, 0x82C1,
    0x8243, 0x0246, 0x024C, 0x8249, 0x0258, 0x825D, 0x8257, 0x0252,
    0x0270, 0x8275, 0x827F, 0x027A, 0x826B, 0x026E, 0x0264, 0x8261,
    0x0220, 0x8225, 0x822F, 0x022A, 0x823B, 0x023E, 0x0234, 0x8231,
    0x8213, 0x0216, 0x021C, 0x8219, 0x0208, 0x820D, 0x8207, 0x0202,
};

/// `ma_dr_flac__crc32_table` (miniaudio/miniaudio.h:89110 @5266f24da).
const crc32_table = [256]u32{
    0x00000000, 0x04C11DB7, 0x09823B6E, 0x0D4326D9, 0x130476DC, 0x17C56B6B, 0x1A864DB2, 0x1E475005,
    0x2608EDB8, 0x22C9F00F, 0x2F8AD6D6, 0x2B4BCB61, 0x350C9B64, 0x31CD86D3, 0x3C8EA00A, 0x384FBDBD,
    0x4C11DB70, 0x48D0C6C7, 0x4593E01E, 0x4152FDA9, 0x5F15ADAC, 0x5BD4B01B, 0x569796C2, 0x52568B75,
    0x6A1936C8, 0x6ED82B7F, 0x639B0DA6, 0x675A1011, 0x791D4014, 0x7DDC5DA3, 0x709F7B7A, 0x745E66CD,
    0x9823B6E0, 0x9CE2AB57, 0x91A18D8E, 0x95609039, 0x8B27C03C, 0x8FE6DD8B, 0x82A5FB52, 0x8664E6E5,
    0xBE2B5B58, 0xBAEA46EF, 0xB7A96036, 0xB3687D81, 0xAD2F2D84, 0xA9EE3033, 0xA4AD16EA, 0xA06C0B5D,
    0xD4326D90, 0xD0F37027, 0xDDB056FE, 0xD9714B49, 0xC7361B4C, 0xC3F706FB, 0xCEB42022, 0xCA753D95,
    0xF23A8028, 0xF6FB9D9F, 0xFBB8BB46, 0xFF79A6F1, 0xE13EF6F4, 0xE5FFEB43, 0xE8BCCD9A, 0xEC7DD02D,
    0x34867077, 0x30476DC0, 0x3D044B19, 0x39C556AE, 0x278206AB, 0x23431B1C, 0x2E003DC5, 0x2AC12072,
    0x128E9DCF, 0x164F8078, 0x1B0CA6A1, 0x1FCDBB16, 0x018AEB13, 0x054BF6A4, 0x0808D07D, 0x0CC9CDCA,
    0x7897AB07, 0x7C56B6B0, 0x71159069, 0x75D48DDE, 0x6B93DDDB, 0x6F52C06C, 0x6211E6B5, 0x66D0FB02,
    0x5E9F46BF, 0x5A5E5B08, 0x571D7DD1, 0x53DC6066, 0x4D9B3063, 0x495A2DD4, 0x44190B0D, 0x40D816BA,
    0xACA5C697, 0xA864DB20, 0xA527FDF9, 0xA1E6E04E, 0xBFA1B04B, 0xBB60ADFC, 0xB6238B25, 0xB2E29692,
    0x8AAD2B2F, 0x8E6C3698, 0x832F1041, 0x87EE0DF6, 0x99A95DF3, 0x9D684044, 0x902B669D, 0x94EA7B2A,
    0xE0B41DE7, 0xE4750050, 0xE9362689, 0xEDF73B3E, 0xF3B06B3B, 0xF771768C, 0xFA325055, 0xFEF34DE2,
    0xC6BCF05F, 0xC27DEDE8, 0xCF3ECB31, 0xCBFFD686, 0xD5B88683, 0xD1799B34, 0xDC3ABDED, 0xD8FBA05A,
    0x690CE0EE, 0x6DCDFD59, 0x608EDB80, 0x644FC637, 0x7A089632, 0x7EC98B85, 0x738AAD5C, 0x774BB0EB,
    0x4F040D56, 0x4BC510E1, 0x46863638, 0x42472B8F, 0x5C007B8A, 0x58C1663D, 0x558240E4, 0x51435D53,
    0x251D3B9E, 0x21DC2629, 0x2C9F00F0, 0x285E1D47, 0x36194D42, 0x32D850F5, 0x3F9B762C, 0x3B5A6B9B,
    0x0315D626, 0x07D4CB91, 0x0A97ED48, 0x0E56F0FF, 0x1011A0FA, 0x14D0BD4D, 0x19939B94, 0x1D528623,
    0xF12F560E, 0xF5EE4BB9, 0xF8AD6D60, 0xFC6C70D7, 0xE22B20D2, 0xE6EA3D65, 0xEBA91BBC, 0xEF68060B,
    0xD727BBB6, 0xD3E6A601, 0xDEA580D8, 0xDA649D6F, 0xC423CD6A, 0xC0E2D0DD, 0xCDA1F604, 0xC960EBB3,
    0xBD3E8D7E, 0xB9FF90C9, 0xB4BCB610, 0xB07DABA7, 0xAE3AFBA2, 0xAAFBE615, 0xA7B8C0CC, 0xA379DD7B,
    0x9B3660C6, 0x9FF77D71, 0x92B45BA8, 0x9675461F, 0x8832161A, 0x8CF30BAD, 0x81B02D74, 0x857130C3,
    0x5D8A9099, 0x594B8D2E, 0x5408ABF7, 0x50C9B640, 0x4E8EE645, 0x4A4FFBF2, 0x470CDD2B, 0x43CDC09C,
    0x7B827D21, 0x7F436096, 0x7200464F, 0x76C15BF8, 0x68860BFD, 0x6C47164A, 0x61043093, 0x65C52D24,
    0x119B4BE9, 0x155A565E, 0x18197087, 0x1CD86D30, 0x029F3D35, 0x065E2082, 0x0B1D065B, 0x0FDC1BEC,
    0x3793A651, 0x3352BBE6, 0x3E119D3F, 0x3AD08088, 0x2497D08D, 0x2056CD3A, 0x2D15EBE3, 0x29D4F654,
    0xC5A92679, 0xC1683BCE, 0xCC2B1D17, 0xC8EA00A0, 0xD6AD50A5, 0xD26C4D12, 0xDF2F6BCB, 0xDBEE767C,
    0xE3A1CBC1, 0xE760D676, 0xEA23F0AF, 0xEEE2ED18, 0xF0A5BD1D, 0xF464A0AA, 0xF9278673, 0xFDE69BC4,
    0x89B8FD09, 0x8D79E0BE, 0x803AC667, 0x84FBDBD0, 0x9ABC8BD5, 0x9E7D9662, 0x933EB0BB, 0x97FFAD0C,
    0xAFB010B1, 0xAB710D06, 0xA6322BDF, 0xA2F33668, 0xBCB4666D, 0xB8757BDA, 0xB5365D03, 0xB1F740B4,
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "dr_flac claims fLaC, and fLaC behind an ID3 tag" {
    try std.testing.expect(claims("fLaC\x00\x00\x00\x22"));
    try std.testing.expect(claims("ID3\x04\x00\x00\x00\x00\x00\x02xxfLaC"));
    try std.testing.expect(!claims("ID3\x04\x00\x00\x00\x00\x00\x02xx\xff\xfb"));
}

test "the CRC-8 of a FLAC frame header byte matches a bitwise reference" {
    var ref: u8 = 0;
    const data: u32 = 0xA5;
    var bit: u5 = 8;
    while (bit > 0) {
        bit -= 1;
        const b: u8 = @truncate((data >> bit) & 1);
        ref = if (ref & 0x80 != 0) ((ref << 1) | b) ^ 0x07 else (ref << 1) | b;
    }
    try std.testing.expectEqual(crc8_table[0xA5], crc8(0, data, 8));
}
