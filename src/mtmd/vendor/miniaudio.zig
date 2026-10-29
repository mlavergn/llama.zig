//! The audio decoder `mtmd-helper` reaches: miniaudio's `ma_decoder` over a
//! memory buffer, asked for mono `f32` at the model's sample rate.
//!
//! # Provenance
//!
//! Ports, from `llama.cpp/vendor/miniaudio/miniaudio.h` (miniaudio v0.11.25)
//! at v0.4.1 (`b29c606e2`), the decoder path `mtmd-helper.cpp` compiles
//! with `MA_NO_ENCODING`, `MA_NO_DEVICE_IO`, `MA_NO_RESOURCE_MANAGER`,
//! `MA_NO_NODE_GRAPH`, `MA_NO_ENGINE` and `MA_NO_GENERATION`: backend
//! selection, the data-source read loop, the data converter (mono mix-down
//! and the linear resampler with its low-pass filter) and the decoder's
//! read loop; and `audio_helpers` from `llama.cpp/tools/mtmd/mtmd-helper.cpp`.
//! The backends are `dr_wav.zig`, `dr_flac.zig` and `dr_mp3.zig`. Each
//! declaration names the function it ports and the line it began at.
//!
//! # The decoder configuration, fixed
//!
//! `ma_decoder_config_init(ma_format_f32, 1, rate)` leaves everything else
//! at its default, which fixes the converter: every backend decodes to
//! `f32`, so there is never a format conversion; one input channel is a
//! passthrough and more are summed and divided (`mono_out`); a different
//! rate resamples linearly through a 4th-order low-pass filter. The four
//! execution paths that leaves -- passthrough, `channels_only`,
//! `resample_only`, `channels_first` -- are the only ones here.
//!
//! The read loops are ported as loops, chunk sizes included, rather than as
//! "decode it all, then convert": a backend called again after a short read
//! can produce more frames (dr_wav's MS-ADPCM does), so how often each is
//! called is part of the output.

const std = @import("std");
const audio_io = @import("audio_io.zig");
const dr_wav = @import("dr_wav.zig");
/// The FLAC backend, public for the gate.
pub const dr_flac = @import("dr_flac.zig");
/// The MP3 backend, public for the gate's kernel diff.
pub const dr_mp3 = @import("dr_mp3.zig");
const Io = audio_io.Io;

/// Whether the last decode read memory the C leaves unspecified (see
/// `dr_mp3.read_indeterminate`); its samples are then not comparable.
pub fn readIndeterminate() bool {
    return dr_mp3.read_indeterminate or dr_flac.read_indeterminate;
}

/// Clears `readIndeterminate`.
pub fn resetIndeterminate() void {
    dr_mp3.read_indeterminate = false;
    dr_flac.read_indeterminate = false;
}

/// Ports `audio_helpers`' `is_audio_file` (mtmd-helper.cpp:306 @b29c606e2).
///
/// Parameters:
/// - `buf`: the file's bytes.
///
/// Return: whether the helper treats `buf` as audio rather than an image.
pub fn isAudioFile(buf: []const u8) bool {
    if (buf.len < 12) return false;
    const is_wav = std.mem.eql(u8, buf[0..4], "RIFF") and std.mem.eql(u8, buf[8..12], "WAVE");
    const is_mp3 = std.mem.eql(u8, buf[0..3], "ID3") or (buf[0] == 0xFF and (buf[1] & 0xE0) == 0xE0);
    const is_flac = std.mem.eql(u8, buf[0..4], "fLaC");
    return is_wav or is_mp3 or is_flac;
}

/// Ports `audio_helpers`' `decode_audio_from_buf` (mtmd-helper.cpp:325 @b29c606e2).
///
/// Parameters:
/// - `gpa`: allocates the decoder's state and `out`'s storage.
/// - `buf`: the file's bytes.
/// - `target_sample_rate`: the model's rate.
/// - `out`: resized to the decoder's length, zero-filled, then read into.
///   Left resized when the read fails, as upstream's vector is.
///
/// Return: false wherever the C++ returns false. Where upstream's
/// `std::vector::resize` would throw -- a corrupt header claiming more
/// frames than memory holds -- this returns false instead.
pub fn decodeAudioFromBuf(gpa: std.mem.Allocator, buf: []const u8, target_sample_rate: c_int, out: *std.ArrayList(f32)) bool {
    var decoder = Decoder.initMemory(gpa, buf, @bitCast(target_sample_rate)) orelse return false;
    defer decoder.deinit();
    const frame_count = decoder.lengthInPcmFrames();
    if (frame_count > std.math.maxInt(usize) / @sizeOf(f32)) return false;
    const n: usize = @intCast(frame_count);
    const old = out.items.len;
    out.resize(gpa, n) catch return false;
    if (n > old) @memset(out.items[old..], 0);
    const read = decoder.readPcmFrames(out.items);
    return read.ok;
}

/// `MA_MAX_CHANNELS` (miniaudio.h:4280 @b29c606e2).
const max_channels = 254;
/// `MA_DATA_CONVERTER_STACK_BUFFER_SIZE` (miniaudio.h:17489 @b29c606e2).
const stack_buffer_size = 4096;
/// `MA_DEFAULT_RESAMPLER_LPF_ORDER` (miniaudio.h:12233 @b29c606e2).
const lpf_order = 4;
/// `MA_PI_D` (miniaudio.h:12190 @b29c606e2).
const pi_d: f64 = 3.14159265358979323846264;

/// The decoding backend `ma_decoder` settled on.
pub const Backend = union(enum) {
    wav: dr_wav.Wav,
    flac: *dr_flac.Flac,
    mp3: *dr_mp3.Mp3,

    fn channels(b: *const Backend) u32 {
        return switch (b.*) {
            .wav => |*w| w.channels,
            .flac => |f| f.channels,
            .mp3 => |m| m.channels,
        };
    }

    fn sampleRate(b: *const Backend) u32 {
        return switch (b.*) {
            .wav => |*w| w.sample_rate,
            .flac => |f| f.sample_rate,
            .mp3 => |m| m.sample_rate,
        };
    }

    /// Ports `ma_wav_get_length_in_pcm_frames`,
    /// `ma_flac_get_length_in_pcm_frames` and
    /// `ma_mp3_get_length_in_pcm_frames` (miniaudio.h:63822, 64468, 65165 @b29c606e2).
    fn length(b: *Backend) u64 {
        return switch (b.*) {
            .wav => |*w| w.total_pcm_frame_count,
            .flac => |f| f.total_pcm_frame_count,
            .mp3 => |m| m.getPcmFrameCount(),
        };
    }

    /// Ports `ma_wav_read_pcm_frames`, `ma_flac_read_pcm_frames` and
    /// `ma_mp3_read_pcm_frames` (miniaudio.h:63637, 64285, 64990 @b29c606e2)
    /// at `f32`: the frames read, `MA_AT_END` when there are none.
    fn read(b: *Backend, out: []f32, frames: u64) u64 {
        return switch (b.*) {
            .wav => |*w| w.readPcmFramesF32(frames, out),
            .flac => |f| f.readPcmFramesF32(frames, out),
            .mp3 => |m| m.readPcmFramesF32(frames, out),
        };
    }

    fn deinit(b: *Backend) void {
        switch (b.*) {
            .wav => {},
            .flac => |f| f.deinit(),
            .mp3 => |m| m.deinit(),
        }
    }
};

/// A read's outcome: the frames delivered, and whether miniaudio returned
/// `MA_SUCCESS`.
pub const ReadResult = struct { frames: u64, ok: bool };

/// Ports `ma_decoder` as `mtmd-helper` configures it.
pub const Decoder = struct {
    backend: Backend,
    output_sample_rate: u32,
    converter: Converter,

    /// Ports `ma_decoder_init_memory` (miniaudio.h:66484 @b29c606e2) with
    /// `ma_decoder_config_init(ma_format_f32, 1, sample_rate)`.
    ///
    /// Parameters:
    /// - `gpa`: allocates the FLAC and MP3 decoders' state.
    /// - `buf`: the file; borrowed for the decoder's lifetime.
    /// - `sample_rate`: the output rate, 0 for the file's own.
    ///
    /// Return: the decoder, or null where the C fails.
    pub fn initMemory(gpa: std.mem.Allocator, buf: []const u8, sample_rate: u32) ?Decoder {
        if (buf.len == 0) return null;

        // each backend over its own memory stream, whose bad seeks fail
        // (`ma_decoder_init_{wav,flac,mp3}_from_memory__internal`)
        if (initBackend(gpa, buf, .dr_wav, .dr_flac, .dr_mp3)) |backend| return postinit(backend, sample_rate);

        // then through the decoder's own callbacks, whose bad seeks clamp
        // (`ma_decoder__preinit_memory_wrapper`, `ma_decoder_init__internal`)
        if (initBackend(gpa, buf, .decoder, .decoder, .decoder)) |backend| return postinit(backend, sample_rate);
        return null;
    }

    fn initBackend(gpa: std.mem.Allocator, buf: []const u8, wav_mode: Io.Mode, flac_mode: Io.Mode, mp3_mode: Io.Mode) ?Backend {
        if (dr_wav.Wav.init(.{ .data = buf, .mode = wav_mode })) |w| return .{ .wav = w };
        if (dr_flac.Flac.open(gpa, .{ .data = buf, .mode = flac_mode })) |f| return .{ .flac = f };
        if (dr_mp3.Mp3.init(gpa, .{ .data = buf, .mode = mp3_mode })) |m| return .{ .mp3 = m };
        return null;
    }

    /// Ports `ma_decoder__postinit` and `ma_decoder__init_data_converter`
    /// (miniaudio.h:66246, 62915 @b29c606e2).
    fn postinit(backend_in: Backend, sample_rate: u32) ?Decoder {
        var backend = backend_in;
        const channels_in = backend.channels();
        const rate_in = backend.sampleRate();
        if (channels_in > max_channels) {
            backend.deinit();
            return null;
        }
        const rate_out = if (sample_rate == 0) rate_in else sample_rate;
        const converter = Converter.init(channels_in, rate_in, rate_out) orelse {
            backend.deinit();
            return null;
        };
        return .{ .backend = backend, .output_sample_rate = rate_out, .converter = converter };
    }

    /// Releases the backend.
    pub fn deinit(d: *Decoder) void {
        d.backend.deinit();
    }

    /// Ports `ma_decoder_get_length_in_pcm_frames` (miniaudio.h:67626 @b29c606e2).
    ///
    /// Return: the length in output frames.
    pub fn lengthInPcmFrames(d: *Decoder) u64 {
        const internal = d.backend.length();
        const rate_in = d.backend.sampleRate();
        if (rate_in == d.output_sample_rate) return internal;
        return frameCountAfterResampling(d.output_sample_rate, rate_in, internal);
    }

    /// Ports `ma_decoder_read_pcm_frames` (miniaudio.h:67374 @b29c606e2).
    ///
    /// Parameters:
    /// - `out`: receives up to `out.len` mono frames.
    ///
    /// Return: the frames written and whether the C returns `MA_SUCCESS`.
    pub fn readPcmFrames(d: *Decoder, out: []f32) ReadResult {
        const frame_count: u64 = out.len;
        if (frame_count == 0) return .{ .frames = 0, .ok = false }; // MA_INVALID_ARGS

        var total: u64 = 0;
        var ok = true;
        if (d.converter.path == .passthrough) {
            const r = dataSourceRead(&d.backend, out, frame_count);
            total = r.frames;
            ok = r.ok;
        } else {
            const ch_in = d.backend.channels();
            var intermediary: [stack_buffer_size / 4]f32 = undefined;
            const cap: u64 = stack_buffer_size / (4 * ch_in);
            while (total < frame_count) {
                const to_out = frame_count - total;
                var to_in: u64 = @min(to_out, cap);
                const required = d.converter.requiredInputFrameCount(to_out);
                if (to_in > required) to_in = required;

                var read_in: u64 = 0;
                if (required > 0) {
                    const r = dataSourceRead(&d.backend, intermediary[0..], to_in);
                    read_in = r.frames;
                    // MA_AT_END is not a failure here: the resampler may
                    // still hold frames
                    ok = r.ok or r.at_end;
                    if (!ok) break;
                }

                var read_out = to_out;
                d.converter.process(intermediary[0..@intCast(read_in * ch_in)], &read_in, out[@intCast(total)..], &read_out);
                ok = true;
                total += read_out;
                if (read_in == 0 and read_out == 0) break;
            }
        }
        if (ok and total == 0) ok = false; // MA_AT_END
        return .{ .frames = total, .ok = ok };
    }
};

const DsRead = struct { frames: u64, ok: bool, at_end: bool };

/// Ports `ma_data_source_read_pcm_frames` (miniaudio.h:59337 @b29c606e2)
/// for an unchained, unlooped source with no range: the backend is called
/// until it has delivered `frame_count` frames or delivers none.
fn dataSourceRead(backend: *Backend, out: []f32, frame_count: u64) DsRead {
    const ch = backend.channels();
    var total: u64 = 0;
    while (total < frame_count) {
        // `ma_data_source_read_pcm_frames_within_range`: MA_AT_END on none
        const n = backend.read(out[@intCast(total * ch)..], frame_count - total);
        total += n;
        if (n == 0) break; // MA_AT_END, end of the chain
    }
    if (total == 0) return .{ .frames = 0, .ok = false, .at_end = true };
    return .{ .frames = total, .ok = true, .at_end = false };
}

/// Ports `ma_calculate_frame_count_after_resampling` (miniaudio.h:17461 @b29c606e2),
/// including its use of `sampleRateIn / sampleRateOut` where the
/// resampler's own estimate divides the other way.
pub fn frameCountAfterResampling(rate_out: u32, rate_in: u32, frames_in: u64) u64 {
    if (rate_in == 0 or rate_out == 0 or frames_in == 0) return 0;
    if (rate_out == rate_in) return frames_in;
    var out = (frames_in *% rate_out) / rate_in;
    const from_frac = (out *% (rate_in / rate_out)) / rate_out;
    const prelim = (out *% (rate_in % rate_out)) +% from_frac;
    if (prelim <= frames_in) out += 1;
    return out;
}

/// The data converter's execution path (`ma_data_converter_execution_path`).
const Path = enum { passthrough, channels_only, resample_only, channels_first };

/// Ports `ma_data_converter` for `f32` in and out, `channels_in` to one.
const Converter = struct {
    path: Path,
    channels_in: u32,
    resampler: LinearResampler,

    /// Ports `ma_data_converter_init_preallocated` (miniaudio.h:56082 @b29c606e2).
    fn init(channels_in: u32, rate_in: u32, rate_out: u32) ?Converter {
        // one channel to one is a passthrough; more go through `mono_out`
        const has_channel_converter = channels_in != 1;
        const has_resampler = rate_in != rate_out;
        var c: Converter = .{ .path = .passthrough, .channels_in = channels_in, .resampler = undefined };
        if (has_resampler) c.resampler = LinearResampler.init(rate_in, rate_out) orelse return null;
        c.path = if (has_channel_converter)
            (if (has_resampler) .channels_first else .channels_only)
        else if (has_resampler) .resample_only else .passthrough;
        return c;
    }

    /// Ports `ma_data_converter_get_required_input_frame_count` (miniaudio.h:56987 @b29c606e2).
    fn requiredInputFrameCount(c: *const Converter, out_frames: u64) u64 {
        if (c.path == .resample_only or c.path == .channels_first) return c.resampler.requiredInputFrameCount(out_frames);
        return out_frames;
    }

    /// Ports `ma_data_converter_process_pcm_frames` (miniaudio.h:56917 @b29c606e2).
    fn process(c: *Converter, in: []const f32, count_in: *u64, out: []f32, count_out: *u64) void {
        switch (c.path) {
            .passthrough => unreachable, // the decoder reads straight through
            .channels_only => {
                // `ma_data_converter_process_pcm_frames__channels_only` (miniaudio.h:56478 @b29c606e2)
                const n = @min(count_in.*, count_out.*);
                monoOut(out, in, n, c.channels_in);
                count_in.* = n;
                count_out.* = n;
            },
            .resample_only => c.resampler.process(in, count_in, out, count_out),
            .channels_first => c.processChannelsFirst(in, count_in, out, count_out),
        }
    }

    /// Ports `ma_data_converter_process_pcm_frames__channels_first` (miniaudio.h:56753 @b29c606e2).
    fn processChannelsFirst(c: *Converter, in: []const f32, count_in: *u64, out: []f32, count_out: *u64) void {
        const frame_count_in = count_in.*;
        const frame_count_out = count_out.*;
        var processed_in: u64 = 0;
        var processed_out: u64 = 0;
        var mid: [stack_buffer_size / 4]f32 = undefined;
        const mid_cap: u64 = mid.len; // one channel out
        while (processed_out < frame_count_out) {
            var out_iter: u64 = @min(frame_count_out - processed_out, mid_cap);
            var in_iter: u64 = @min(frame_count_in - processed_in, mid_cap);
            const required = c.resampler.requiredInputFrameCount(out_iter);
            if (in_iter > required) in_iter = required;

            monoOut(&mid, in[@intCast(processed_in * c.channels_in)..], in_iter, c.channels_in);
            c.resampler.process(mid[0..@intCast(in_iter)], &in_iter, out[@intCast(processed_out)..], &out_iter);

            processed_in += in_iter;
            processed_out += out_iter;
            if (out_iter == 0) break; // consumed all of our input data
        }
        count_in.* = processed_in;
        count_out.* = processed_out;
    }
};

/// Ports `ma_channel_converter_process_pcm_frames__mono_out` (miniaudio.h:55649 @b29c606e2)
/// at `f32`: a sum in channel order, then one division.
fn monoOut(out: []f32, in: []const f32, frames: u64, channels_in: u32) void {
    const n: usize = @intCast(frames);
    const ch: usize = channels_in;
    for (0..n) |f| {
        var t: f32 = 0;
        for (0..ch) |c| t += in[f * ch + c];
        out[f] = t / @as(f32, @floatFromInt(channels_in));
    }
}

extern "c" fn sin(x: f64) f64;

/// `ma_biquad` (miniaudio.h:4631 @b29c606e2), at f32 with one channel.
const Biquad = struct {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    r1: f32 = 0,
    r2: f32 = 0,

    /// Ports `ma_lpf2__get_biquad_config` and `ma_biquad_reinit`
    /// (miniaudio.h:47774, 47308 @b29c606e2).
    fn lpf2(sample_rate: u32, cutoff: f64, q: f64) Biquad {
        const w = 2 * pi_d * cutoff / @as(f64, @floatFromInt(sample_rate));
        const s = sin(w);
        const c = cosd(w);
        const a = s / (2 * q);
        const b0 = (1 - c) / 2;
        const b1 = 1 - c;
        const b2 = (1 - c) / 2;
        const a0 = 1 + a;
        const a1 = -2 * c;
        const a2 = 1 - a;
        return .{
            .b0 = @floatCast(b0 / a0),
            .b1 = @floatCast(b1 / a0),
            .b2 = @floatCast(b2 / a0),
            .a1 = @floatCast(a1 / a0),
            .a2 = @floatCast(a2 / a0),
        };
    }

    /// Ports `ma_biquad_process_pcm_frame_f32__direct_form_2_transposed`
    /// (miniaudio.h:47372 @b29c606e2). Clang contracts each statement: the
    /// first product fuses with the add, and of `b1*x - a1*y + r2` the left
    /// product fuses with the subtraction before `r2` is added.
    fn process(bq: *Biquad, x: f32) f32 {
        const y = @mulAdd(f32, bq.b0, x, bq.r1);
        const r1 = @mulAdd(f32, bq.b1, x, -(bq.a1 * y)) + bq.r2;
        const r2 = @mulAdd(f32, bq.b2, x, -(bq.a2 * y));
        bq.r1 = r1;
        bq.r2 = r2;
        return y;
    }
};

/// Ports `ma_cosd` (miniaudio.h:12444 @b29c606e2): `sin(pi/2 - x)`.
fn cosd(x: f64) f64 {
    return sin((pi_d * 0.5) - x);
}

/// Ports `ma_linear_resampler` at `f32` with one channel, the only count
/// the converter gives it.
const LinearResampler = struct {
    rate_in: u32, // simplified
    rate_out: u32,
    in_advance_int: u32,
    in_advance_frac: u32,
    in_time_int: u32 = 1, // force an input sample to be loaded for the first output frame
    in_time_frac: u32 = 0,
    x0: f32 = 0,
    x1: f32 = 0,
    lpf: [lpf_order / 2]Biquad,

    /// Ports `ma_linear_resampler_init_preallocated` and
    /// `ma_linear_resampler_set_rate_internal` (miniaudio.h:53213, 53074 @b29c606e2).
    fn init(rate_in_full: u32, rate_out_full: u32) ?LinearResampler {
        if (rate_in_full == 0 or rate_out_full == 0) return null;
        const g = gcf(rate_in_full, rate_out_full);
        const rate_in = rate_in_full / g;
        const rate_out = rate_out_full / g;
        const lpf_rate = @max(rate_in, rate_out);
        const cutoff: f64 = @as(f64, @floatFromInt(@min(rate_in, rate_out))) * 0.5 * 1.0;
        var lpf: [lpf_order / 2]Biquad = undefined;
        // `ma_lpf_reinit__internal` (miniaudio.h:48026 @b29c606e2): an even
        // order is all second-order sections
        for (&lpf, 0..) |*bq, i| {
            const a = @as(f64, @floatFromInt(1 + i * 2)) * (pi_d / @as(f64, @floatFromInt(lpf_order * 2)));
            const q = 1 / (2 * cosd(a));
            bq.* = Biquad.lpf2(lpf_rate, cutoff, q);
        }
        return .{
            .rate_in = rate_in,
            .rate_out = rate_out,
            .in_advance_int = rate_in / rate_out,
            .in_advance_frac = rate_in % rate_out,
            .lpf = lpf,
        };
    }

    /// Ports `ma_gcf_u32`.
    fn gcf(a_in: u32, b_in: u32) u32 {
        var a = a_in;
        var b = b_in;
        while (b != 0) {
            const t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    /// Ports `ma_lpf_process_pcm_frame_f32` (miniaudio.h:48267 @b29c606e2).
    fn filter(r: *LinearResampler, x: f32) f32 {
        var y = x;
        for (&r.lpf) |*bq| y = bq.process(y);
        return y;
    }

    /// Ports `ma_linear_resampler_interpolate_frame_f32` and
    /// `ma_mix_f32_fast` (miniaudio.h:53332, 13978 @b29c606e2): three
    /// statements, so three roundings.
    fn interpolate(r: *const LinearResampler) f32 {
        const a = @as(f32, @floatFromInt(r.in_time_frac)) / @as(f32, @floatFromInt(r.rate_out));
        const r0 = r.x1 - r.x0;
        const r1 = r0 * a;
        return r.x0 + r1;
    }

    fn advance(r: *LinearResampler) void {
        r.in_time_int += r.in_advance_int;
        r.in_time_frac += r.in_advance_frac;
        if (r.in_time_frac >= r.rate_out) {
            r.in_time_frac -= r.rate_out;
            r.in_time_int += 1;
        }
    }

    /// Ports `ma_linear_resampler_process_pcm_frames_f32_downsample` and
    /// `ma_linear_resampler_process_pcm_frames_f32_upsample` (miniaudio.h:53514, 53590 @b29c606e2).
    fn process(r: *LinearResampler, in: []const f32, count_in: *u64, out: []f32, count_out: *u64) void {
        const downsample = r.rate_in > r.rate_out;
        const frame_count_in = count_in.*;
        const frame_count_out = count_out.*;
        var processed_in: u64 = 0;
        var processed_out: u64 = 0;
        while (processed_out < frame_count_out) {
            // load the buffers, filtering every input sample when downsampling
            while (r.in_time_int > 0 and frame_count_in > processed_in) {
                r.x0 = r.x1;
                r.x1 = in[@intCast(processed_in)];
                if (downsample and r.rate_in != r.rate_out) r.x1 = r.filter(r.x1);
                processed_in += 1;
                r.in_time_int -= 1;
            }
            if (r.in_time_int > 0) break; // ran out of input data

            var y = r.interpolate();
            if (!downsample and r.rate_in != r.rate_out) y = r.filter(y);
            out[@intCast(processed_out)] = y;
            processed_out += 1;
            r.advance();
        }
        count_in.* = processed_in;
        count_out.* = processed_out;
    }

    /// Ports `ma_linear_resampler_get_required_input_frame_count` (miniaudio.h:53744 @b29c606e2).
    fn requiredInputFrameCount(r: *const LinearResampler, out_frames_in: u64) u64 {
        if (out_frames_in == 0) return 0;
        var input: u64 = r.in_time_int;
        const out_frames = out_frames_in - 1;
        input +%= out_frames *% r.in_advance_int;
        input +%= (r.in_time_frac +% (out_frames *% r.in_advance_frac)) / r.rate_out;
        return input;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the sniff takes RIFF/WAVE, ID3, an MPEG sync word and fLaC" {
    try std.testing.expect(isAudioFile("RIFF\x00\x00\x00\x00WAVEfmt "));
    try std.testing.expect(isAudioFile("ID3\x04\x00\x00\x00\x00\x00\x00\x00\x00"));
    try std.testing.expect(isAudioFile("\xff\xfb\x90\x00\x00\x00\x00\x00\x00\x00\x00\x00"));
    try std.testing.expect(isAudioFile("fLaC\x00\x00\x00\x22\x00\x00\x00\x00"));
    try std.testing.expect(!isAudioFile("RIFF\x00\x00\x00\x00WEBPVP8 "));
    try std.testing.expect(!isAudioFile("fLaC"));
}

test "the length after resampling keeps upstream's arithmetic" {
    try std.testing.expectEqual(@as(u64, 16000), frameCountAfterResampling(16000, 44100, 44100));
    try std.testing.expectEqual(@as(u64, 0), frameCountAfterResampling(16000, 44100, 0));
}
