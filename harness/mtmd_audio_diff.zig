//! `make mtmd-audio-diff`: `src/mtmd/audio.zig` against upstream's
//! `tools/mtmd/mtmd-audio.cpp`, compiled unmodified beside it
//! (`harness/mtmd_audio_ref.cpp`), in one process.
//!
//! Every preprocessor runs on both sides over the same PCM and hparams, and
//! everything it returns must agree **on bits**: the ok flag, the number of
//! mels, each mel's `n_len`, `n_len_org`, `n_mel` and every float -- and
//! every log line either side wrote. The streaming inverse STFT runs a
//! sequence of frames, a flush, a reset and more frames. Signals: silence,
//! sine sweeps, white noise, impulses, a hard-clipped square, large
//! amplitudes; lengths from a handful of samples to past 60 s, at every
//! chunk boundary of the class plus or minus one.
//!
//! **Not a port.** Usage: `zig-out/bin/mtmd-audio-diff [--quick] [filter]`.

const std = @import("std");
const audio = @import("mtmd_audio");
const mtmd = audio.deps;
const Hparams = mtmd.model.Hparams;

extern fn ref_audio_hparams_new() ?*anyopaque;
extern fn ref_audio_hparams_free(hp: ?*anyopaque) void;
extern fn ref_audio_hparams_set(hp: ?*anyopaque, n_mel_bins: c_int, n_fft: c_int, window_len: c_int, hop_len: c_int, sample_rate: c_int, chunk_len: c_int, mimi_downsample: c_int) void;
extern fn ref_audio_hparams_set_parakeet(hp: ?*anyopaque, mel_filters: [*]const f32, n_filters: usize, window: [*]const f32, n_window: usize) void;
extern fn ref_audio_set_log(cb: ?*const fn (c_int, [*c]const u8, ?*anyopaque) callconv(.c) void, ud: ?*anyopaque) void;
extern fn ref_audio_proc_new(kind: c_int, hp: ?*anyopaque) ?*anyopaque;
extern fn ref_audio_proc_free(r: ?*anyopaque) void;
extern fn ref_audio_proc_run(r: ?*anyopaque, samples: [*]const f32, n: usize, out: *?*anyopaque, out_len: *usize) c_int;
extern fn ref_istft_new(n_fft: c_int, hop: c_int) ?*anyopaque;
extern fn ref_istft_free(s: ?*anyopaque) void;
extern fn ref_istft_reset(s: ?*anyopaque) void;
extern fn ref_istft_frame(s: ?*anyopaque, spec: [*]const f32, out: [*]f32) usize;
extern fn ref_istft_flush(s: ?*anyopaque, out: [*]f32) usize;
extern fn free(p: ?*anyopaque) void;

const gpa = std.heap.c_allocator;

// -----------------------------------------------------------------------------
// logs

var log_ref: std.ArrayList(u8) = .empty;
var log_port: std.ArrayList(u8) = .empty;

fn logRef(level: c_int, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    log_ref.print(gpa, "[{d}] {s}", .{ level, std.mem.span(text) }) catch {};
}

fn logPort(level: mtmd.impl.c.enum_ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    log_port.print(gpa, "[{d}] {s}", .{ @as(c_int, @intCast(level)), std.mem.span(text) }) catch {};
}

// -----------------------------------------------------------------------------
// configurations

/// One preprocessor configuration: the hparams both sides get.
const Config = struct {
    name: []const u8,
    kind: audio.Kind,
    n_mel: c_int,
    n_fft: c_int,
    window: c_int,
    hop: c_int,
    sr: c_int,
    chunk_len: c_int = 0,
    mimi_downsample: c_int = 0,
    /// The chunk length in samples each class splits at, for the boundary
    /// lengths; 0 for none.
    chunk_samples: usize = 0,
};

const configs = [_]Config{
    .{ .name = "whisper-80", .kind = .whisper, .n_mel = 80, .n_fft = 400, .window = 400, .hop = 160, .sr = 16000, .chunk_samples = 480000 },
    .{ .name = "whisper-128-win300", .kind = .whisper, .n_mel = 128, .n_fft = 400, .window = 300, .hop = 160, .sr = 16000, .chunk_samples = 480000 },
    .{ .name = "whisper-nfft512", .kind = .whisper, .n_mel = 64, .n_fft = 512, .window = 400, .hop = 160, .sr = 16000, .chunk_samples = 480000 },
    .{ .name = "conformer", .kind = .conformer, .n_mel = 128, .n_fft = 512, .window = 400, .hop = 160, .sr = 16000 },
    .{ .name = "granite", .kind = .granite_speech, .n_mel = 160, .n_fft = 512, .window = 400, .hop = 160, .sr = 16000 },
    .{ .name = "gemma4a", .kind = .gemma4a, .n_mel = 128, .n_fft = 512, .window = 320, .hop = 160, .sr = 16000, .chunk_samples = 480000 },
    .{ .name = "gemma4ua", .kind = .gemma4ua, .n_mel = 640, .n_fft = 0, .window = 0, .hop = 0, .sr = 16000, .chunk_samples = 640 },
    .{ .name = "qwen3a", .kind = .qwen3a, .n_mel = 128, .n_fft = 400, .window = 400, .hop = 160, .sr = 16000, .chunk_samples = 800 * 160 },
    .{ .name = "dots3note-60s", .kind = .dots3note, .n_mel = 128, .n_fft = 400, .window = 400, .hop = 160, .sr = 16000, .chunk_len = 60, .chunk_samples = 960000 },
    .{ .name = "dots3note-1s", .kind = .dots3note, .n_mel = 80, .n_fft = 400, .window = 400, .hop = 160, .sr = 16000, .chunk_len = 1, .chunk_samples = 16000 },
    .{ .name = "mimo-audio", .kind = .mimo_audio, .n_mel = 128, .n_fft = 960, .window = 960, .hop = 240, .sr = 24000 },
    .{ .name = "qwen3tts-spk", .kind = .qwen3tts_spk, .n_mel = 128, .n_fft = 1024, .window = 1024, .hop = 256, .sr = 24000 },
    .{ .name = "qwen3tts-spk-odd", .kind = .qwen3tts_spk, .n_mel = 40, .n_fft = 600, .window = 480, .hop = 150, .sr = 12000 },
    .{ .name = "pockettts", .kind = .pockettts, .n_mel = 0, .n_fft = 0, .window = 0, .hop = 0, .sr = 24000, .mimi_downsample = 16, .chunk_samples = 1920 },
    .{ .name = "parakeet-80", .kind = .parakeet, .n_mel = 80, .n_fft = 512, .window = 400, .hop = 160, .sr = 16000 },
    .{ .name = "parakeet-128-win512", .kind = .parakeet, .n_mel = 128, .n_fft = 512, .window = 512, .hop = 160, .sr = 16000 },
};

// -----------------------------------------------------------------------------
// signals

const Signal = enum { silence, sweep, noise, impulses, square_clip, loud };

fn makeSignal(sig: Signal, n: usize, sr: c_int, seed: u64) ![]f32 {
    const s = try gpa.alloc(f32, n);
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    const fsr: f64 = @floatFromInt(@max(sr, 1));
    for (s, 0..) |*v, i| {
        const t: f64 = @as(f64, @floatFromInt(i)) / fsr;
        v.* = switch (sig) {
            .silence => 0,
            .sweep => @floatCast(0.5 * @sin(2.0 * std.math.pi * (100.0 * t + 1500.0 * t * t))),
            .noise => r.float(f32) * 2 - 1,
            .impulses => if (i % 997 == 0) 1.0 else if (i % 1301 == 7) -0.75 else 0.0,
            .square_clip => if (@sin(2.0 * std.math.pi * 440.0 * t) >= 0) 1.0 else -1.0,
            .loud => (r.float(f32) * 2 - 1) * 3000.0,
        };
    }
    return s;
}

// -----------------------------------------------------------------------------
// the port's result, in the shim's blob format

fn portBlob(mels: []const audio.Mel) ![]u8 {
    var blob: std.ArrayList(u8) = .empty;
    try blob.appendSlice(gpa, std.mem.asBytes(&@as(i64, @intCast(mels.len))));
    for (mels) |m| {
        try blob.appendSlice(gpa, std.mem.asBytes(&m.n_len));
        try blob.appendSlice(gpa, std.mem.asBytes(&m.n_len_org));
        try blob.appendSlice(gpa, std.mem.asBytes(&m.n_mel));
        try blob.appendSlice(gpa, std.mem.asBytes(&@as(i64, @intCast(m.data.items.len))));
        try blob.appendSlice(gpa, std.mem.sliceAsBytes(m.data.items));
    }
    return blob.toOwnedSlice(gpa);
}

var n_cases: usize = 0;
var n_checks: usize = 0;
var n_fail: usize = 0;

fn fail(comptime fmt: []const u8, args: anytype) void {
    n_fail += 1;
    if (n_fail <= 40) std.debug.print("FAIL " ++ fmt ++ "\n", args);
}

/// Where two blobs first differ, as a float index past the headers.
fn compareBlobs(name: []const u8, len: usize, sig: Signal, ref: []const u8, port: []const u8) void {
    n_checks += ref.len / 4;
    if (std.mem.eql(u8, ref, port)) return;
    var i: usize = 0;
    while (i < @min(ref.len, port.len) and ref[i] == port[i]) i += 1;
    fail("{s} n={d} {s}: blobs differ at byte {d} (ref {d} bytes, port {d})", .{ name, len, @tagName(sig), i, ref.len, port.len });
    if (i + 4 <= @min(ref.len, port.len)) {
        const a = i & ~@as(usize, 3);
        const rv: f32 = @bitCast(std.mem.readInt(u32, ref[a..][0..4], .little));
        const pv: f32 = @bitCast(std.mem.readInt(u32, port[a..][0..4], .little));
        std.debug.print("     ref {e} ({x:0>8})  port {e} ({x:0>8})\n", .{ rv, @as(u32, @bitCast(rv)), pv, @as(u32, @bitCast(pv)) });
    }
}

fn runCase(cfg: *const Config, hp_port: *const Hparams, hp_ref: ?*anyopaque, len: usize, sig: Signal) !void {
    n_cases += 1;
    const samples = try makeSignal(sig, len, cfg.sr, @as(u64, len) *% 2654435761 +% @backingInt(sig));
    defer gpa.free(samples);

    log_ref.clearRetainingCapacity();
    log_port.clearRetainingCapacity();

    // reference
    const r = ref_audio_proc_new(@backingInt(cfg.kind), hp_ref);
    defer ref_audio_proc_free(r);
    var blob: ?*anyopaque = null;
    var blob_len: usize = 0;
    const ok_ref = ref_audio_proc_run(r, samples.ptr, samples.len, &blob, &blob_len) != 0;
    defer free(blob);
    const ref_bytes = @as([*]const u8, @ptrCast(blob.?))[0..blob_len];

    // port
    var p = audio.Preprocessor.init(cfg.kind, hp_port);
    defer p.deinit();
    try p.initialize();
    var mels: std.ArrayList(audio.Mel) = .empty;
    defer {
        for (mels.items) |*m| m.deinit();
        mels.deinit(gpa);
    }
    const ok_port = try p.preprocess(samples, &mels);
    const port_bytes = try portBlob(mels.items);
    defer gpa.free(port_bytes);

    n_checks += 1;
    if (ok_ref != ok_port) fail("{s} n={d} {s}: ok ref={} port={}", .{ cfg.name, len, @tagName(sig), ok_ref, ok_port });
    compareBlobs(cfg.name, len, sig, ref_bytes, port_bytes);
    n_checks += 1;
    if (!std.mem.eql(u8, log_ref.items, log_port.items)) {
        fail("{s} n={d} {s}: logs differ", .{ cfg.name, len, @tagName(sig) });
        std.debug.print("     ref:  {s}\n     port: {s}\n", .{ log_ref.items, log_port.items });
    }
}

fn lengths(cfg: *const Config, quick: bool, out: *std.ArrayList(usize)) !void {
    const base = [_]usize{ 0, 1, 2, 5, 37, 159, 160, 161, 199, 200, 201, 255, 256, 257, 383, 384, 385, 400, 401, 511, 512, 513, 1000, 4801, 16000, 23999, 24001 };
    for (base) |l| try out.append(gpa, l);
    if (cfg.n_fft > 0) {
        const nf: usize = @intCast(cfg.n_fft);
        for ([_]usize{ nf / 2 - 1, nf / 2, nf / 2 + 1, nf - 1, nf, nf + 1, 2 * nf + 3 }) |l| try out.append(gpa, l);
    }
    const sr: usize = @intCast(cfg.sr);
    try out.append(gpa, 3 * sr + 7);
    if (!quick) {
        try out.append(gpa, 17 * sr + 123);
        if (cfg.chunk_samples > 0) {
            const cs = cfg.chunk_samples;
            for ([_]usize{ cs - 1, cs, cs + 1, 2 * cs - 1, 2 * cs, 2 * cs + 1 }) |l| try out.append(gpa, l);
        }
        // past 30 s (pockettts truncates there) and past 60 s
        try out.append(gpa, 31 * sr + 11);
        if (cfg.kind != .whisper and cfg.kind != .dots3note) try out.append(gpa, 61 * sr + 5);
    }
}

fn parakeetTables(n_mel: c_int, n_fft: c_int, window: c_int) !struct { []f32, []f32 } {
    const bins: usize = @intCast(@divTrunc(n_fft, 2) + 1);
    const filt = try gpa.alloc(f32, @as(usize, @intCast(n_mel)) * bins);
    var rng = std.Random.DefaultPrng.init(0x5eed);
    for (filt) |*v| v.* = if (rng.random().float(f32) < 0.7) 0.0 else rng.random().float(f32) * 0.01;
    const win = try gpa.alloc(f32, @intCast(window));
    for (win, 0..) |*v, i| {
        const x = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(window));
        v.* = @floatCast(0.5 - 0.5 * @cos(2.0 * std.math.pi * x) + 0.001 * @as(f64, @floatFromInt(i % 3)));
    }
    return .{ filt, win };
}

fn runConfig(cfg: *const Config, quick: bool) !void {
    var hp: Hparams = .{
        .n_mel_bins = cfg.n_mel,
        .audio_n_fft = cfg.n_fft,
        .audio_window_len = cfg.window,
        .audio_hop_len = cfg.hop,
        .audio_sample_rate = cfg.sr,
        .audio_chunk_len = cfg.chunk_len,
        .mimi_downsample = cfg.mimi_downsample,
    };
    const hp_ref = ref_audio_hparams_new();
    defer ref_audio_hparams_free(hp_ref);
    ref_audio_hparams_set(hp_ref, cfg.n_mel, cfg.n_fft, cfg.window, cfg.hop, cfg.sr, cfg.chunk_len, cfg.mimi_downsample);
    var tables: ?struct { []f32, []f32 } = null;
    defer if (tables) |t| {
        gpa.free(t[0]);
        gpa.free(t[1]);
    };
    if (cfg.kind == .parakeet) {
        tables = try parakeetTables(cfg.n_mel, cfg.n_fft, cfg.window);
        try hp.mel_filters.appendSlice(gpa, tables.?[0]);
        try hp.window.appendSlice(gpa, tables.?[1]);
        ref_audio_hparams_set_parakeet(hp_ref, tables.?[0].ptr, tables.?[0].len, tables.?[1].ptr, tables.?[1].len);
    }
    defer {
        hp.mel_filters.deinit(gpa);
        hp.window.deinit(gpa);
    }

    var lens: std.ArrayList(usize) = .empty;
    defer lens.deinit(gpa);
    try lengths(cfg, quick, &lens);
    const before = n_fail;
    for (lens.items) |len| {
        // the conformer and parakeet normalize per feature and need two
        // valid frames; upstream asserts (conformer) or divides by zero
        if (cfg.kind == .conformer and len < 2 * @as(usize, @intCast(cfg.hop))) continue;
        // qwen3tts reflects past its start below the pad (upstream reads out of bounds)
        if (cfg.kind == .qwen3tts_spk and len > 0 and len < @as(usize, @intCast(@divTrunc(cfg.n_fft - cfg.hop, 2))) + 1) continue;
        const sigs: []const Signal = if (len > 200000) &.{ .noise, .sweep } else &.{ .silence, .sweep, .noise, .impulses, .square_clip, .loud };
        for (sigs) |sig| try runCase(cfg, &hp, hp_ref, len, sig);
    }
    std.debug.print("{s:<22} {s}\n", .{ cfg.name, if (n_fail == before) "ok" else "FAIL" });
}

fn runIstft(n_fft: c_int, hop: c_int, n_frames: usize) !void {
    n_cases += 1;
    const s_ref = ref_istft_new(n_fft, hop);
    defer ref_istft_free(s_ref);
    var s_port = try audio.StreamingIstft.init(n_fft, hop);
    defer s_port.deinit();

    const bins: usize = @intCast(@divTrunc(n_fft, 2) + 1);
    const spec = try gpa.alloc(f32, bins * 2);
    defer gpa.free(spec);
    const out_ref = try gpa.alloc(f32, @intCast(n_fft * 4));
    defer gpa.free(out_ref);
    var rng = std.Random.DefaultPrng.init(@intCast(n_fft * 31 + hop));
    const before = n_fail;

    for (0..2) |round| {
        for (0..n_frames) |f| {
            for (spec) |*v| v.* = (rng.random().float(f32) * 2 - 1) * @as(f32, @floatFromInt(f % 5 + 1));
            const n_ref = ref_istft_frame(s_ref, spec.ptr, out_ref.ptr);
            var o = try s_port.processFrame(spec.ptr);
            defer o.deinit(gpa);
            n_checks += n_ref + 1;
            if (n_ref != o.items.len or !std.mem.eql(u8, std.mem.sliceAsBytes(out_ref[0..n_ref]), std.mem.sliceAsBytes(o.items))) {
                fail("istft n_fft={d} hop={d} round {d} frame {d}: differs (ref {d} samples, port {d})", .{ n_fft, hop, round, f, n_ref, o.items.len });
            }
        }
        const n_ref = ref_istft_flush(s_ref, out_ref.ptr);
        var o = try s_port.flush();
        defer o.deinit(gpa);
        n_checks += n_ref + 1;
        if (n_ref != o.items.len or !std.mem.eql(u8, std.mem.sliceAsBytes(out_ref[0..n_ref]), std.mem.sliceAsBytes(o.items))) {
            fail("istft n_fft={d} hop={d} round {d}: flush differs", .{ n_fft, hop, round });
        }
        // mid-stream reset after the first round: frames then flush again
        ref_istft_reset(s_ref);
        s_port.reset();
    }
    std.debug.print("istft {d:>5}/{d:<4}         {s}\n", .{ @as(u32, @intCast(n_fft)), @as(u32, @intCast(hop)), if (n_fail == before) "ok" else "FAIL" });
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var quick = false;
    var filter: ?[]const u8 = null;
    for (args[1..]) |a| {
        if (std.mem.eql(u8, a, "--quick")) quick = true else filter = a;
    }

    ref_audio_set_log(logRef, null);
    mtmd.impl.g_logger_state.log_callback = logPort;

    for (&configs) |*cfg| {
        if (filter) |f| if (std.mem.indexOf(u8, cfg.name, f) == null) continue;
        try runConfig(cfg, quick);
    }
    if (filter == null or std.mem.indexOf(u8, "istft", filter.?) != null) {
        for ([_][2]c_int{ .{ 16, 4 }, .{ 20, 5 }, .{ 100, 25 }, .{ 1024, 256 }, .{ 1920, 480 }, .{ 30, 30 }, .{ 25, 7 } }) |c| try runIstft(c[0], c[1], 40);
    }

    if (n_fail != 0) {
        std.debug.print("FAIL: {d} of {d} checks over {d} cases differ\n", .{ n_fail, n_checks, n_cases });
        return 1;
    }
    std.debug.print("PASS: {d} cases, {d} checks identical on bits\n", .{ n_cases, n_checks });
    return 0;
}
