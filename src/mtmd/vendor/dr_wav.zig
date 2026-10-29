//! The WAV decoder miniaudio's `ma_decoder` tries first: dr_wav, as
//! miniaudio embeds it.
//!
//! # Provenance
//!
//! Ports the reading half of `ma_dr_wav` in
//! `llama.cpp/vendor/miniaudio/miniaudio.h` (miniaudio v0.11.25, dr_wav
//! embedded) at v0.4.1 (`b29c606e2`). Each declaration names the function it
//! ports and the line it began at.
//!
//! # What is reachable
//!
//! `mtmd-helper` hands miniaudio only buffers its sniff accepted -- `RIFF` +
//! `WAVE`, `ID3`, an MPEG sync word, or `fLaC` -- so of dr_wav's five
//! containers only RIFF can ever open: `RIFX`, `riff` (W64), `RF64` and
//! `FORM` (AIFF) start with bytes no sniffed buffer has. The decoder asks
//! for `f32`, so only `ma_dr_wav_read_pcm_frames_f32` and what it calls are
//! here. Metadata parsing is off (`ma_dr_wav_init_memory` passes no flags).

const std = @import("std");
const audio_io = @import("audio_io.zig");
const Io = audio_io.Io;

const format_pcm = 0x1;
const format_adpcm = 0x2;
const format_ieee_float = 0x3;
const format_alaw = 0x6;
const format_mulaw = 0x7;
const format_dvi_adpcm = 0x11;
const format_extensible = 0xFFFE;

/// `MA_DR_WAV_MAX_SAMPLE_RATE`, `MA_DR_WAV_MAX_CHANNELS` and `MA_DR_WAV_MAX_BITS_PER_SAMPLE` (miniaudio.h:79998, 80001, 80004 @b29c606e2).
const max_sample_rate = 384000;
const max_channels = 256;
const max_bits_per_sample = 64;

/// `ma_dr_wav_fmt`.
const Fmt = struct {
    format_tag: u16 = 0,
    channels: u16 = 0,
    sample_rate: u32 = 0,
    avg_bytes_per_sec: u32 = 0,
    block_align: u16 = 0,
    bits_per_sample: u16 = 0,
    extended_size: u16 = 0,
    valid_bits_per_sample: u16 = 0,
    channel_mask: u32 = 0,
    sub_format: [16]u8 = @splat(0),
};

fn u16le(b: []const u8) u16 {
    return std.mem.readInt(u16, b[0..2], .little);
}

fn s16le(b: []const u8) i16 {
    return std.mem.readInt(i16, b[0..2], .little);
}

fn u32le(b: []const u8) u32 {
    return std.mem.readInt(u32, b[0..4], .little);
}

/// Ports `ma_dr_wav`'s reading state (miniaudio.h:62262 @b29c606e2).
pub const Wav = struct {
    io: Io,
    fmt: Fmt = .{},
    sample_rate: u32 = 0,
    channels: u16 = 0,
    bits_per_sample: u16 = 0,
    translated_format_tag: u16 = 0,
    total_pcm_frame_count: u64 = 0,
    data_chunk_data_size: u64 = 0,
    data_chunk_data_pos: u64 = 0,
    bytes_remaining: u64 = 0,
    read_cursor_in_pcm_frames: u64 = 0,
    msadpcm: struct {
        bytes_remaining_in_block: u32 = 0,
        predictor: [2]u16 = .{ 0, 0 },
        delta: [2]i32 = .{ 0, 0 },
        cached_frames: [4]i32 = @splat(0),
        cached_frame_count: u32 = 0,
        prev_frames: [2][2]i32 = @splat(@splat(0)),
    } = .{},
    ima: struct {
        bytes_remaining_in_block: u32 = 0,
        predictor: [2]i32 = .{ 0, 0 },
        step_index: [2]i32 = .{ 0, 0 },
        cached_frames: [16]i32 = @splat(0),
        cached_frame_count: u32 = 0,
    } = .{},

    /// Ports `ma_dr_wav_init__internal` (miniaudio.h:81197 @b29c606e2), the
    /// RIFF container with no metadata and no chunk callback.
    ///
    /// Parameters:
    /// - `io`: the stream, positioned at the start of the buffer.
    ///
    /// Return: the decoder, or null where the C returns `MA_FALSE`.
    pub fn init(io: Io) ?Wav {
        var w: Wav = .{ .io = io };
        if (!w.initInternal()) return null;
        return w;
    }

    fn initInternal(w: *Wav) bool {
        var cursor: u64 = 0;
        var fmt: Fmt = .{};
        var data_chunk_size: u64 = 0;
        var found_fmt = false;
        var found_data = false;

        var riff: [4]u8 = undefined;
        if (w.readCounted(&riff, &cursor) != 4) return false;
        // every other container starts with bytes no sniffed buffer has
        if (!std.mem.eql(u8, &riff, "RIFF")) return false;

        var chunk_size_bytes: [4]u8 = undefined;
        if (w.readCounted(&chunk_size_bytes, &cursor) != 4) return false;
        var wave: [4]u8 = undefined;
        if (w.readCounted(&wave, &cursor) != 4) return false;
        if (!std.mem.eql(u8, &wave, "WAVE")) return false;

        while (true) {
            // `ma_dr_wav__read_chunk_header` (miniaudio.h:80343 @b29c606e2)
            var id: [4]u8 = undefined;
            if (w.io.read(&id) != 4) break;
            var size_bytes: [4]u8 = undefined;
            if (w.io.read(&size_bytes) != 4) break;
            const size_in_bytes: u64 = u32le(&size_bytes);
            const padding_size: u64 = size_in_bytes % 2;
            cursor +%= 8;

            var chunk_size = size_in_bytes;

            if (std.mem.eql(u8, &id, "fmt ")) {
                var fmt_data: [16]u8 = undefined;
                found_fmt = true;
                if (w.io.read(&fmt_data) != 16) return false;
                cursor +%= 16;
                fmt.format_tag = u16le(fmt_data[0..]);
                fmt.channels = u16le(fmt_data[2..]);
                fmt.sample_rate = u32le(fmt_data[4..]);
                fmt.avg_bytes_per_sec = u32le(fmt_data[8..]);
                fmt.block_align = u16le(fmt_data[12..]);
                fmt.bits_per_sample = u16le(fmt_data[14..]);
                fmt.extended_size = 0;
                fmt.valid_bits_per_sample = 0;
                fmt.channel_mask = 0;
                fmt.sub_format = @splat(0);
                if (size_in_bytes > 16) {
                    var cb_size: [2]u8 = undefined;
                    if (w.io.read(&cb_size) != 2) return false;
                    cursor +%= 2;
                    var bytes_read_so_far: u64 = 18;
                    fmt.extended_size = u16le(&cb_size);
                    if (fmt.extended_size > 0) {
                        if (fmt.format_tag == format_extensible) {
                            if (fmt.extended_size != 22) return false;
                            var ext: [22]u8 = undefined;
                            if (w.io.read(&ext) != 22) return false;
                            fmt.valid_bits_per_sample = u16le(ext[0..]);
                            fmt.channel_mask = u32le(ext[2..]);
                            @memcpy(&fmt.sub_format, ext[6..22]);
                        } else {
                            if (!w.io.seek(fmt.extended_size, .cur)) return false;
                        }
                        cursor +%= fmt.extended_size;
                        bytes_read_so_far += fmt.extended_size;
                    }
                    // `(int)(header.sizeInBytes - bytesReadSoFar)`
                    const back: i32 = @bitCast(@as(u32, @truncate(size_in_bytes -% bytes_read_so_far)));
                    if (!w.io.seek(back, .cur)) return false;
                    cursor +%= size_in_bytes -% bytes_read_so_far;
                }
                if (padding_size > 0) {
                    if (!w.io.seekForward(padding_size)) break;
                    cursor +%= padding_size;
                }
                continue;
            }

            if (std.mem.eql(u8, &id, "data")) {
                found_data = true;
                w.data_chunk_data_pos = cursor;
                data_chunk_size = chunk_size;
                break; // not sequential, and no metadata to collect past it
            }

            if (std.mem.eql(u8, &id, "fact")) {
                var sample_count: [4]u8 = undefined;
                if (w.readCounted(&sample_count, &cursor) != 4) return false;
                chunk_size -%= 4;
                // `sampleCountFromFactChunk` is set only when the format tag
                // read so far is ADPCM, and `pWav->translatedFormatTag` is
                // still zero here: a RIFF fact chunk never counts
                chunk_size +%= padding_size;
                if (!w.io.seekForward(chunk_size)) break;
                cursor +%= chunk_size;
                continue;
            }

            chunk_size +%= padding_size;
            if (!w.io.seekForward(chunk_size)) break;
            cursor +%= chunk_size;
        }

        if (!found_fmt or !found_data) return false;
        if ((fmt.sample_rate == 0 or fmt.sample_rate > max_sample_rate) or
            (fmt.channels == 0 or fmt.channels > max_channels) or
            (fmt.bits_per_sample == 0 or fmt.bits_per_sample > max_bits_per_sample) or
            fmt.block_align == 0) return false;

        var translated = fmt.format_tag;
        if (translated == format_extensible) translated = u16le(&fmt.sub_format);

        if (!w.io.seekFromStart(w.data_chunk_data_pos)) return false;

        if (w.io.seek(0, .end)) {
            const file_size: u64 = @intCast(w.io.tell());
            if (data_chunk_size +% w.data_chunk_data_pos > file_size) data_chunk_size = file_size -% w.data_chunk_data_pos;
        }

        if (data_chunk_size == 0xFFFFFFFF) {
            data_chunk_size = 0;
            while (true) {
                var temp: [4096]u8 = undefined;
                const n = w.io.read(&temp);
                data_chunk_size += n;
                if (n < temp.len) break;
            }
        }

        if (!w.io.seekFromStart(w.data_chunk_data_pos)) return false;

        w.fmt = fmt;
        w.sample_rate = fmt.sample_rate;
        w.channels = fmt.channels;
        w.bits_per_sample = fmt.bits_per_sample;
        w.translated_format_tag = translated;

        if (!isCompressedFormatTag(translated)) {
            const bpf = w.bytesPerPcmFrame();
            if (bpf > 0) data_chunk_size -= data_chunk_size % bpf;
        }

        w.bytes_remaining = data_chunk_size;
        w.data_chunk_data_size = data_chunk_size;

        const bpf = w.bytesPerPcmFrame();
        if (bpf == 0) return false;
        w.total_pcm_frame_count = data_chunk_size / bpf;
        if (translated == format_adpcm) {
            var block_count = data_chunk_size / fmt.block_align;
            if (block_count * fmt.block_align < data_chunk_size) block_count += 1;
            const header_bytes = block_count *% (6 * @as(u64, fmt.channels));
            w.total_pcm_frame_count = ((data_chunk_size -% header_bytes) *% 2) / fmt.channels;
        }
        if (translated == format_dvi_adpcm) {
            var block_count = data_chunk_size / fmt.block_align;
            if (block_count * fmt.block_align < data_chunk_size) block_count += 1;
            const header_bytes = block_count *% (4 * @as(u64, fmt.channels));
            w.total_pcm_frame_count = ((data_chunk_size -% header_bytes) *% 2) / fmt.channels;
            w.total_pcm_frame_count +%= block_count;
        }

        if (translated == format_adpcm or translated == format_dvi_adpcm) {
            if (w.channels > 2) return false;
        }
        if (w.bytesPerPcmFrame() == 0) return false;
        return true;
    }

    /// Ports `ma_dr_wav__on_read` (miniaudio.h:80409 @b29c606e2).
    fn readCounted(w: *Wav, buf: []u8, cursor: *u64) usize {
        const n = w.io.read(buf);
        cursor.* +%= n;
        return n;
    }

    /// Ports `ma_dr_wav_get_bytes_per_pcm_frame` (miniaudio.h:81155 @b29c606e2).
    fn bytesPerPcmFrame(w: *const Wav) u32 {
        var bpf: u32 = undefined;
        if (w.bits_per_sample & 0x7 == 0) {
            bpf = (@as(u32, w.bits_per_sample) * w.fmt.channels) >> 3;
        } else {
            bpf = w.fmt.block_align;
        }
        if (w.translated_format_tag == format_alaw or w.translated_format_tag == format_mulaw) {
            if (bpf != w.fmt.channels) return 0;
        }
        return bpf;
    }

    /// Ports `ma_dr_wav_read_raw` (miniaudio.h:82762 @b29c606e2), into a
    /// buffer.
    fn readRaw(w: *Wav, buf: []u8) usize {
        var bytes_to_read = buf.len;
        if (bytes_to_read == 0) return 0;
        if (bytes_to_read > w.bytes_remaining) bytes_to_read = @intCast(w.bytes_remaining);
        if (bytes_to_read == 0) return 0;
        const bpf = w.bytesPerPcmFrame();
        if (bpf == 0) return 0;
        const n = w.io.read(buf[0..bytes_to_read]);
        w.read_cursor_in_pcm_frames += n / bpf;
        w.bytes_remaining -= n;
        return n;
    }

    /// Ports `ma_dr_wav_read_pcm_frames_le` (miniaudio.h:82811 @b29c606e2),
    /// the only arm `ma_dr_wav_read_pcm_frames` takes for RIFF.
    fn readPcmFramesLe(w: *Wav, frames_in: u64, buf: []u8) u64 {
        var frames = frames_in;
        if (frames == 0) return 0;
        if (isCompressedFormatTag(w.translated_format_tag)) return 0;
        const remaining = w.total_pcm_frame_count -% w.read_cursor_in_pcm_frames;
        if (frames > remaining) frames = remaining;
        const bpf = w.bytesPerPcmFrame();
        if (bpf == 0) return 0;
        const bytes = frames * bpf;
        if (bytes == 0) return 0;
        return w.readRaw(buf[0..@intCast(bytes)]) / bpf;
    }

    /// Ports `ma_dr_wav_read_pcm_frames_f32` (miniaudio.h:83968 @b29c606e2).
    ///
    /// Parameters:
    /// - `frames`: the frames wanted.
    /// - `out`: at least `frames * channels` samples.
    ///
    /// Return: the frames written.
    pub fn readPcmFramesF32(w: *Wav, frames: u64, out: []f32) u64 {
        if (frames == 0) return 0;
        return switch (w.translated_format_tag) {
            format_pcm => w.readF32Chunked(frames, out, .pcm),
            format_adpcm, format_dvi_adpcm => w.readF32Adpcm(frames, out),
            format_ieee_float => w.readF32Chunked(frames, out, .ieee),
            format_alaw => w.readF32Chunked(frames, out, .alaw),
            format_mulaw => w.readF32Chunked(frames, out, .mulaw),
            else => 0,
        };
    }

    const Conv = enum { pcm, ieee, alaw, mulaw };

    /// Ports `ma_dr_wav_read_pcm_frames_f32__pcm`, `ma_dr_wav_read_pcm_frames_f32__ieee`,
    /// `ma_dr_wav_read_pcm_frames_f32__alaw` and `ma_dr_wav_read_pcm_frames_f32__mulaw`
    /// (miniaudio.h:83786, 83840, 83878, 83923 @b29c606e2), which
    /// share a body.
    fn readF32Chunked(w: *Wav, frames_in: u64, out: []f32, comptime conv: Conv) u64 {
        if (conv == .ieee and w.translated_format_tag == format_ieee_float and w.bits_per_sample == 32) {
            // read the floats as they are
            const bytes = std.mem.sliceAsBytes(out);
            return w.readPcmFramesLe(frames_in, bytes[0..@min(bytes.len, @as(usize, @intCast(@min(frames_in, std.math.maxInt(u32)))) * w.bytesPerPcmFrame())]);
        }
        var sample_data: [4096]u8 = @splat(0);
        const bpf = w.bytesPerPcmFrame();
        if (bpf == 0) return 0;
        const bps = bpf / w.channels;
        if (bps == 0 or bpf % w.channels != 0) return 0;

        var frames = frames_in;
        var total: u64 = 0;
        var o: usize = 0;
        while (frames > 0) {
            const this_iter = @min(frames, sample_data.len / bpf);
            const read = w.readPcmFramesLe(this_iter, &sample_data);
            if (read == 0) break;
            const samples: usize = @intCast(read * w.channels);
            if (samples * bps > sample_data.len) break;
            const dst = out[o..][0..samples];
            switch (conv) {
                .pcm => pcmToF32(dst, &sample_data, bps),
                .ieee => ieeeToF32(dst, &sample_data, bps),
                .alaw => for (dst, 0..) |*d, i| {
                    d.* = @as(f32, @floatFromInt(@as(i16, @bitCast(alaw_table[sample_data[i]])))) / 32768.0;
                },
                .mulaw => for (dst, 0..) |*d, i| {
                    d.* = @as(f32, @floatFromInt(@as(i16, @bitCast(mulaw_table[sample_data[i]])))) / 32768.0;
                },
            }
            o += samples;
            frames -= read;
            total += read;
        }
        return total;
    }

    /// Ports `ma_dr_wav_read_pcm_frames_f32__msadpcm_ima` (miniaudio.h:83821 @b29c606e2).
    fn readF32Adpcm(w: *Wav, frames_in: u64, out: []f32) u64 {
        var samples16: [2048]i16 = undefined;
        var frames = frames_in;
        var total: u64 = 0;
        var o: usize = 0;
        while (frames > 0) {
            const this_iter = @min(frames, samples16.len / w.channels);
            const read = w.readPcmFramesS16(this_iter, &samples16);
            if (read == 0) break;
            const n: usize = @intCast(read * w.channels);
            for (out[o..][0..n], samples16[0..n]) |*d, s| d.* = @as(f32, @floatFromInt(s)) * 0.000030517578125;
            o += n;
            frames -= read;
            total += read;
        }
        return total;
    }

    /// Ports `ma_dr_wav_read_pcm_frames_s16` (miniaudio.h:83614 @b29c606e2),
    /// the ADPCM arms `__msadpcm_ima` reaches.
    fn readPcmFramesS16(w: *Wav, frames: u64, out: []i16) u64 {
        if (frames == 0) return 0;
        return switch (w.translated_format_tag) {
            format_adpcm => w.readS16MsAdpcm(frames, out),
            format_dvi_adpcm => w.readS16Ima(frames, out),
            else => 0,
        };
    }

    /// Ports `ma_dr_wav_read_pcm_frames_s16__msadpcm` (miniaudio.h:83091 @b29c606e2).
    fn readS16MsAdpcm(w: *Wav, frames_in: u64, out: []i16) u64 {
        const adaptation_table = [16]i32{ 230, 230, 230, 230, 307, 409, 512, 614, 768, 614, 512, 409, 307, 230, 230, 230 };
        const coeff1_table = [7]i32{ 256, 512, 0, 192, 240, 460, 392 };
        const coeff2_table = [7]i32{ 0, -256, 0, 64, 0, -208, -232 };
        const m = &w.msadpcm;
        const ch: u32 = w.channels;
        var frames = frames_in;
        var total: u64 = 0;
        var o: usize = 0;
        while (w.read_cursor_in_pcm_frames < w.total_pcm_frame_count) {
            if (m.cached_frame_count == 0 and m.bytes_remaining_in_block == 0) {
                if (ch == 1) {
                    var header: [7]u8 = undefined;
                    if (w.io.read(&header) != 7) return total;
                    m.bytes_remaining_in_block = @as(u32, w.fmt.block_align) -% 7;
                    m.predictor[0] = header[0];
                    m.delta[0] = s16le(header[1..]);
                    m.prev_frames[0][1] = s16le(header[3..]);
                    m.prev_frames[0][0] = s16le(header[5..]);
                    m.cached_frames[2] = m.prev_frames[0][0];
                    m.cached_frames[3] = m.prev_frames[0][1];
                    m.cached_frame_count = 2;
                    if (m.predictor[0] >= 7) return total;
                } else {
                    var header: [14]u8 = undefined;
                    if (w.io.read(&header) != 14) return total;
                    m.bytes_remaining_in_block = @as(u32, w.fmt.block_align) -% 14;
                    m.predictor[0] = header[0];
                    m.predictor[1] = header[1];
                    m.delta[0] = s16le(header[2..]);
                    m.delta[1] = s16le(header[4..]);
                    m.prev_frames[0][1] = s16le(header[6..]);
                    m.prev_frames[1][1] = s16le(header[8..]);
                    m.prev_frames[0][0] = s16le(header[10..]);
                    m.prev_frames[1][0] = s16le(header[12..]);
                    m.cached_frames[0] = m.prev_frames[0][0];
                    m.cached_frames[1] = m.prev_frames[1][0];
                    m.cached_frames[2] = m.prev_frames[0][1];
                    m.cached_frames[3] = m.prev_frames[1][1];
                    m.cached_frame_count = 2;
                    if (m.predictor[0] >= 7 or m.predictor[1] >= 7) return total;
                }
            }

            while (frames > 0 and m.cached_frame_count > 0 and w.read_cursor_in_pcm_frames < w.total_pcm_frame_count) {
                for (0..ch) |s| out[o + s] = @truncate(m.cached_frames[(4 - m.cached_frame_count * ch) + s]);
                o += ch;
                frames -= 1;
                total += 1;
                w.read_cursor_in_pcm_frames += 1;
                m.cached_frame_count -= 1;
            }
            if (frames == 0) break;

            if (m.cached_frame_count == 0) {
                if (m.bytes_remaining_in_block == 0) continue;
                var nibbles: [1]u8 = undefined;
                if (w.io.read(&nibbles) != 1) return total;
                m.bytes_remaining_in_block -= 1;
                const nb = nibbles[0];
                var nibble0: i32 = (nb & 0xF0) >> 4;
                if (nb & 0x80 != 0) nibble0 = @bitCast(@as(u32, @bitCast(nibble0)) | 0xFFFFFFF0);
                var nibble1: i32 = nb & 0x0F;
                if (nb & 0x08 != 0) nibble1 = @bitCast(@as(u32, @bitCast(nibble1)) | 0xFFFFFFF0);
                const p0 = m.predictor[0];
                if (p0 >= 7) return total;
                const s0 = msadpcmSample(m.prev_frames[0], coeff1_table[p0], coeff2_table[p0], nibble0, m.delta[0]);
                m.delta[0] = adaptDelta(adaptation_table[(nb & 0xF0) >> 4], m.delta[0]);
                m.prev_frames[0][0] = m.prev_frames[0][1];
                m.prev_frames[0][1] = s0;
                if (ch == 1) {
                    const s1 = msadpcmSample(m.prev_frames[0], coeff1_table[p0], coeff2_table[p0], nibble1, m.delta[0]);
                    m.delta[0] = adaptDelta(adaptation_table[nb & 0x0F], m.delta[0]);
                    m.prev_frames[0][0] = m.prev_frames[0][1];
                    m.prev_frames[0][1] = s1;
                    m.cached_frames[2] = s0;
                    m.cached_frames[3] = s1;
                    m.cached_frame_count = 2;
                } else {
                    const p1 = m.predictor[1];
                    if (p1 >= 7) return total;
                    const s1 = msadpcmSample(m.prev_frames[1], coeff1_table[p1], coeff2_table[p1], nibble1, m.delta[1]);
                    m.delta[1] = adaptDelta(adaptation_table[nb & 0x0F], m.delta[1]);
                    m.prev_frames[1][0] = m.prev_frames[1][1];
                    m.prev_frames[1][1] = s1;
                    m.cached_frames[2] = s0;
                    m.cached_frames[3] = s1;
                    m.cached_frame_count = 1;
                }
            }
        }
        return total;
    }

    /// Ports `ma_dr_wav_read_pcm_frames_s16__ima` (miniaudio.h:83226 @b29c606e2).
    fn readS16Ima(w: *Wav, frames_in: u64, out: []i16) u64 {
        const index_table = [16]i32{ -1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8 };
        const im = &w.ima;
        const ch: u32 = w.channels;
        var frames = frames_in;
        var total: u64 = 0;
        var o: usize = 0;
        while (w.read_cursor_in_pcm_frames < w.total_pcm_frame_count) {
            if (im.cached_frame_count == 0 and im.bytes_remaining_in_block == 0) {
                if (ch == 1) {
                    var header: [4]u8 = undefined;
                    if (w.io.read(&header) != 4) return total;
                    im.bytes_remaining_in_block = @as(u32, w.fmt.block_align) -% 4;
                    if (header[2] >= ima_step_table.len) {
                        _ = w.io.seek(@as(i32, @bitCast(im.bytes_remaining_in_block)), .cur);
                        im.bytes_remaining_in_block = 0;
                        return total;
                    }
                    im.predictor[0] = @as(i16, @bitCast(u16le(&header)));
                    im.step_index[0] = header[2];
                    im.cached_frames[15] = im.predictor[0];
                    im.cached_frame_count = 1;
                } else {
                    var header: [8]u8 = undefined;
                    if (w.io.read(&header) != 8) return total;
                    im.bytes_remaining_in_block = @as(u32, w.fmt.block_align) -% 8;
                    if (header[2] >= ima_step_table.len or header[6] >= ima_step_table.len) {
                        _ = w.io.seek(@as(i32, @bitCast(im.bytes_remaining_in_block)), .cur);
                        im.bytes_remaining_in_block = 0;
                        return total;
                    }
                    im.predictor[0] = s16le(header[0..]);
                    im.step_index[0] = header[2];
                    im.predictor[1] = s16le(header[4..]);
                    im.step_index[1] = header[6];
                    im.cached_frames[14] = im.predictor[0];
                    im.cached_frames[15] = im.predictor[1];
                    im.cached_frame_count = 1;
                }
            }

            while (frames > 0 and im.cached_frame_count > 0 and w.read_cursor_in_pcm_frames < w.total_pcm_frame_count) {
                for (0..ch) |s| out[o + s] = @truncate(im.cached_frames[(16 - im.cached_frame_count * ch) + s]);
                o += ch;
                frames -= 1;
                total += 1;
                w.read_cursor_in_pcm_frames += 1;
                im.cached_frame_count -= 1;
            }
            if (frames == 0) break;

            if (im.cached_frame_count == 0) {
                if (im.bytes_remaining_in_block == 0) continue;
                im.cached_frame_count = 8;
                for (0..ch) |c| {
                    var nibbles: [4]u8 = undefined;
                    if (w.io.read(&nibbles) != 4) {
                        im.cached_frame_count = 0;
                        return total;
                    }
                    im.bytes_remaining_in_block -%= 4;
                    for (0..4) |b| {
                        inline for (.{ 0, 1 }) |half| {
                            const nib: u8 = if (half == 0) nibbles[b] & 0x0F else (nibbles[b] & 0xF0) >> 4;
                            const step = ima_step_table[@intCast(im.step_index[c])];
                            var diff: i32 = step >> 3;
                            if (nib & 1 != 0) diff += step >> 2;
                            if (nib & 2 != 0) diff += step >> 1;
                            if (nib & 4 != 0) diff += step;
                            if (nib & 8 != 0) diff = -diff;
                            const predictor = std.math.clamp(im.predictor[c] + diff, -32768, 32767);
                            im.predictor[c] = predictor;
                            im.step_index[c] = std.math.clamp(im.step_index[c] + index_table[nib], 0, @as(i32, ima_step_table.len - 1));
                            im.cached_frames[(16 - im.cached_frame_count * ch) + (b * 2 + half) * ch + c] = predictor;
                        }
                    }
                }
            }
        }
        return total;
    }
};

/// One MS-ADPCM prediction: the C's `int` arithmetic, which a corrupt
/// delta overflows, wraps here as clang compiles it.
fn msadpcmSample(prev: [2]i32, coeff1: i32, coeff2: i32, nibble: i32, delta: i32) i32 {
    var s = ((prev[1] *% coeff1) +% (prev[0] *% coeff2)) >> 8;
    s +%= nibble *% delta;
    return std.math.clamp(s, -32768, 32767);
}

fn adaptDelta(adaptation: i32, delta: i32) i32 {
    return @intCast(std.math.clamp((@as(i64, adaptation) * delta) >> 8, 16, 0x7FFFFFFF));
}

/// Ports `ma_dr_wav__is_compressed_format_tag` (miniaudio.h:80326 @b29c606e2).
fn isCompressedFormatTag(tag: u16) bool {
    return tag == format_adpcm or tag == format_dvi_adpcm;
}

/// Ports `ma_dr_wav__pcm_to_f32` (miniaudio.h:83734 @b29c606e2) and the
/// converters it calls, `ma_dr_wav_u8_to_f32`, `ma_dr_wav_s16_to_f32`,
/// `ma_dr_wav_s24_to_f32` and `ma_dr_wav_s32_to_f32` (miniaudio.h:84012, 84031, 84041, 84056 @b29c606e2).
fn pcmToF32(out: []f32, in: []const u8, bps: u32) void {
    switch (bps) {
        1 => for (out, 0..) |*d, i| {
            // two statements in the C, so two roundings
            var x: f32 = @floatFromInt(in[i]);
            x = x * 0.00784313725490196078;
            x = x - 1;
            d.* = x;
        },
        2 => for (out, 0..) |*d, i| {
            d.* = @as(f32, @floatFromInt(s16le(in[i * 2 ..]))) * 0.000030517578125;
        },
        3 => for (out, 0..) |*d, i| {
            const a = @as(u32, in[i * 3 + 0]) << 8;
            const b = @as(u32, in[i * 3 + 1]) << 16;
            const c = @as(u32, in[i * 3 + 2]) << 24;
            const x: f64 = @floatFromInt(@as(i32, @bitCast(a | b | c)) >> 8);
            d.* = @floatCast(x * 0.00000011920928955078125);
        },
        4 => for (out, 0..) |*d, i| {
            const v = std.mem.readInt(i32, in[i * 4 ..][0..4], .little);
            d.* = @floatCast(@as(f64, @floatFromInt(v)) / 2147483648.0);
        },
        else => {
            if (bps > 8) {
                @memset(out, 0);
                return;
            }
            var p: usize = 0;
            for (out) |*d| {
                var sample: u64 = 0;
                var shift: u32 = (8 - bps) * 8;
                for (0..bps) |j| {
                    sample |= @as(u64, in[p + j]) << @intCast(shift);
                    shift += 8;
                }
                p += bps;
                d.* = @floatCast(@as(f64, @floatFromInt(@as(i64, @bitCast(sample)))) / 9223372036854775807.0);
            }
        },
    }
}

/// Ports `ma_dr_wav__ieee_to_f32` (miniaudio.h:83770 @b29c606e2).
fn ieeeToF32(out: []f32, in: []const u8, bps: u32) void {
    switch (bps) {
        4 => for (out, 0..) |*d, i| {
            d.* = @bitCast(std.mem.readInt(u32, in[i * 4 ..][0..4], .little));
        },
        8 => for (out, 0..) |*d, i| {
            d.* = @floatCast(@as(f64, @bitCast(std.mem.readInt(u64, in[i * 8 ..][0..8], .little))));
        },
        else => @memset(out, 0),
    }
}

/// The IMA `stepTable` (miniaudio.h:83234 @b29c606e2).
const ima_step_table = [89]i32{
    7,     8,     9,     10,    11,    12,    13,    14,    16,    17,
    19,    21,    23,    25,    28,    31,    34,    37,    41,    45,
    50,    55,    60,    66,    73,    80,    88,    97,    107,   118,
    130,   143,   157,   173,   190,   209,   230,   253,   279,   307,
    337,   371,   408,   449,   494,   544,   598,   658,   724,   796,
    876,   963,   1060,  1166,  1282,  1411,  1552,  1707,  1878,  2066,
    2272,  2499,  2749,  3024,  3327,  3660,  4026,  4428,  4871,  5358,
    5894,  6484,  7132,  7845,  8630,  9493,  10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
};

/// `ma_dr_wav_gAlawTable` (miniaudio.h:83347 @b29c606e2).
const alaw_table = [256]u16{
    0xEA80, 0xEB80, 0xE880, 0xE980, 0xEE80, 0xEF80, 0xEC80, 0xED80, 0xE280, 0xE380, 0xE080, 0xE180, 0xE680, 0xE780, 0xE480, 0xE580,
    0xF540, 0xF5C0, 0xF440, 0xF4C0, 0xF740, 0xF7C0, 0xF640, 0xF6C0, 0xF140, 0xF1C0, 0xF040, 0xF0C0, 0xF340, 0xF3C0, 0xF240, 0xF2C0,
    0xAA00, 0xAE00, 0xA200, 0xA600, 0xBA00, 0xBE00, 0xB200, 0xB600, 0x8A00, 0x8E00, 0x8200, 0x8600, 0x9A00, 0x9E00, 0x9200, 0x9600,
    0xD500, 0xD700, 0xD100, 0xD300, 0xDD00, 0xDF00, 0xD900, 0xDB00, 0xC500, 0xC700, 0xC100, 0xC300, 0xCD00, 0xCF00, 0xC900, 0xCB00,
    0xFEA8, 0xFEB8, 0xFE88, 0xFE98, 0xFEE8, 0xFEF8, 0xFEC8, 0xFED8, 0xFE28, 0xFE38, 0xFE08, 0xFE18, 0xFE68, 0xFE78, 0xFE48, 0xFE58,
    0xFFA8, 0xFFB8, 0xFF88, 0xFF98, 0xFFE8, 0xFFF8, 0xFFC8, 0xFFD8, 0xFF28, 0xFF38, 0xFF08, 0xFF18, 0xFF68, 0xFF78, 0xFF48, 0xFF58,
    0xFAA0, 0xFAE0, 0xFA20, 0xFA60, 0xFBA0, 0xFBE0, 0xFB20, 0xFB60, 0xF8A0, 0xF8E0, 0xF820, 0xF860, 0xF9A0, 0xF9E0, 0xF920, 0xF960,
    0xFD50, 0xFD70, 0xFD10, 0xFD30, 0xFDD0, 0xFDF0, 0xFD90, 0xFDB0, 0xFC50, 0xFC70, 0xFC10, 0xFC30, 0xFCD0, 0xFCF0, 0xFC90, 0xFCB0,
    0x1580, 0x1480, 0x1780, 0x1680, 0x1180, 0x1080, 0x1380, 0x1280, 0x1D80, 0x1C80, 0x1F80, 0x1E80, 0x1980, 0x1880, 0x1B80, 0x1A80,
    0x0AC0, 0x0A40, 0x0BC0, 0x0B40, 0x08C0, 0x0840, 0x09C0, 0x0940, 0x0EC0, 0x0E40, 0x0FC0, 0x0F40, 0x0CC0, 0x0C40, 0x0DC0, 0x0D40,
    0x5600, 0x5200, 0x5E00, 0x5A00, 0x4600, 0x4200, 0x4E00, 0x4A00, 0x7600, 0x7200, 0x7E00, 0x7A00, 0x6600, 0x6200, 0x6E00, 0x6A00,
    0x2B00, 0x2900, 0x2F00, 0x2D00, 0x2300, 0x2100, 0x2700, 0x2500, 0x3B00, 0x3900, 0x3F00, 0x3D00, 0x3300, 0x3100, 0x3700, 0x3500,
    0x0158, 0x0148, 0x0178, 0x0168, 0x0118, 0x0108, 0x0138, 0x0128, 0x01D8, 0x01C8, 0x01F8, 0x01E8, 0x0198, 0x0188, 0x01B8, 0x01A8,
    0x0058, 0x0048, 0x0078, 0x0068, 0x0018, 0x0008, 0x0038, 0x0028, 0x00D8, 0x00C8, 0x00F8, 0x00E8, 0x0098, 0x0088, 0x00B8, 0x00A8,
    0x0560, 0x0520, 0x05E0, 0x05A0, 0x0460, 0x0420, 0x04E0, 0x04A0, 0x0760, 0x0720, 0x07E0, 0x07A0, 0x0660, 0x0620, 0x06E0, 0x06A0,
    0x02B0, 0x0290, 0x02F0, 0x02D0, 0x0230, 0x0210, 0x0270, 0x0250, 0x03B0, 0x0390, 0x03F0, 0x03D0, 0x0330, 0x0310, 0x0370, 0x0350,
};

/// `ma_dr_wav_gMulawTable` (miniaudio.h:83365 @b29c606e2).
const mulaw_table = [256]u16{
    0x8284, 0x8684, 0x8A84, 0x8E84, 0x9284, 0x9684, 0x9A84, 0x9E84, 0xA284, 0xA684, 0xAA84, 0xAE84, 0xB284, 0xB684, 0xBA84, 0xBE84,
    0xC184, 0xC384, 0xC584, 0xC784, 0xC984, 0xCB84, 0xCD84, 0xCF84, 0xD184, 0xD384, 0xD584, 0xD784, 0xD984, 0xDB84, 0xDD84, 0xDF84,
    0xE104, 0xE204, 0xE304, 0xE404, 0xE504, 0xE604, 0xE704, 0xE804, 0xE904, 0xEA04, 0xEB04, 0xEC04, 0xED04, 0xEE04, 0xEF04, 0xF004,
    0xF0C4, 0xF144, 0xF1C4, 0xF244, 0xF2C4, 0xF344, 0xF3C4, 0xF444, 0xF4C4, 0xF544, 0xF5C4, 0xF644, 0xF6C4, 0xF744, 0xF7C4, 0xF844,
    0xF8A4, 0xF8E4, 0xF924, 0xF964, 0xF9A4, 0xF9E4, 0xFA24, 0xFA64, 0xFAA4, 0xFAE4, 0xFB24, 0xFB64, 0xFBA4, 0xFBE4, 0xFC24, 0xFC64,
    0xFC94, 0xFCB4, 0xFCD4, 0xFCF4, 0xFD14, 0xFD34, 0xFD54, 0xFD74, 0xFD94, 0xFDB4, 0xFDD4, 0xFDF4, 0xFE14, 0xFE34, 0xFE54, 0xFE74,
    0xFE8C, 0xFE9C, 0xFEAC, 0xFEBC, 0xFECC, 0xFEDC, 0xFEEC, 0xFEFC, 0xFF0C, 0xFF1C, 0xFF2C, 0xFF3C, 0xFF4C, 0xFF5C, 0xFF6C, 0xFF7C,
    0xFF88, 0xFF90, 0xFF98, 0xFFA0, 0xFFA8, 0xFFB0, 0xFFB8, 0xFFC0, 0xFFC8, 0xFFD0, 0xFFD8, 0xFFE0, 0xFFE8, 0xFFF0, 0xFFF8, 0x0000,
    0x7D7C, 0x797C, 0x757C, 0x717C, 0x6D7C, 0x697C, 0x657C, 0x617C, 0x5D7C, 0x597C, 0x557C, 0x517C, 0x4D7C, 0x497C, 0x457C, 0x417C,
    0x3E7C, 0x3C7C, 0x3A7C, 0x387C, 0x367C, 0x347C, 0x327C, 0x307C, 0x2E7C, 0x2C7C, 0x2A7C, 0x287C, 0x267C, 0x247C, 0x227C, 0x207C,
    0x1EFC, 0x1DFC, 0x1CFC, 0x1BFC, 0x1AFC, 0x19FC, 0x18FC, 0x17FC, 0x16FC, 0x15FC, 0x14FC, 0x13FC, 0x12FC, 0x11FC, 0x10FC, 0x0FFC,
    0x0F3C, 0x0EBC, 0x0E3C, 0x0DBC, 0x0D3C, 0x0CBC, 0x0C3C, 0x0BBC, 0x0B3C, 0x0ABC, 0x0A3C, 0x09BC, 0x093C, 0x08BC, 0x083C, 0x07BC,
    0x075C, 0x071C, 0x06DC, 0x069C, 0x065C, 0x061C, 0x05DC, 0x059C, 0x055C, 0x051C, 0x04DC, 0x049C, 0x045C, 0x041C, 0x03DC, 0x039C,
    0x036C, 0x034C, 0x032C, 0x030C, 0x02EC, 0x02CC, 0x02AC, 0x028C, 0x026C, 0x024C, 0x022C, 0x020C, 0x01EC, 0x01CC, 0x01AC, 0x018C,
    0x0174, 0x0164, 0x0154, 0x0144, 0x0134, 0x0124, 0x0114, 0x0104, 0x00F4, 0x00E4, 0x00D4, 0x00C4, 0x00B4, 0x00A4, 0x0094, 0x0084,
    0x0078, 0x0070, 0x0068, 0x0060, 0x0058, 0x0050, 0x0048, 0x0040, 0x0038, 0x0030, 0x0028, 0x0020, 0x0018, 0x0010, 0x0008, 0x0000,
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a 16-bit mono WAV reads back its samples" {
    const wav = "RIFF\x2c\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x04\x00\x00\x00\x00\x40\x00\xc0";
    var w = Wav.init(.{ .data = wav, .mode = .dr_wav }).?;
    try std.testing.expectEqual(@as(u64, 2), w.total_pcm_frame_count);
    var out: [2]f32 = undefined;
    try std.testing.expectEqual(@as(u64, 2), w.readPcmFramesF32(2, &out));
    try std.testing.expectEqual(@as(f32, 0.5), out[0]);
    try std.testing.expectEqual(@as(f32, -0.5), out[1]);
}
