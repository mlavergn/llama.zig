//! `make mtmd-decode-diff`: the vendored decoders `mtmd-helper` reaches,
//! `src/mtmd/vendor/`, against upstream's `stb_image.h` and `miniaudio.h`
//! compiled unmodified beside them (`harness/mtmd_decode_ref.cpp`), in one
//! process, compared on bits.
//!
//! **Not a port.** For every input -- the files named on the command line
//! (directories are walked) and the ones synthesized here -- and for every
//! truncation and corruption of each, it asks both sides:
//!
//! - `stbi_load_from_memory` at `req_comp` 3, as the helper does, and at 0,
//!   1, 2 and 4 for the base files: null-ness, the failure reason, `x`, `y`,
//!   `comp` and every byte;
//! - `decode_audio_from_buf` at several rates: the step that failed, the
//!   length, and every sample's bits.
//!
//! Usage: mtmd-decode-diff [--quick] [--verbose] [--list] [--only=<substring>] [path...]
//!
//! `--list` prints each input's audio outcome on the reference side, to
//! show a corpus decodes rather than failing on both sides.

const std = @import("std");
const vendor = @import("vendor");
const stb = vendor.stb_image;
const ma = vendor.miniaudio;

extern fn ref_stbi_load(buf: [*]const u8, len: c_int, req_comp: c_int, x: *c_int, y: *c_int, comp: *c_int, reason: *?[*:0]const u8) ?[*]u8;
extern fn ref_stbi_free(data: ?[*]u8) void;
extern fn ref_audio_decode(buf: [*]const u8, len: usize, rate: c_int, cap: u64, out: *?[*]f32, n: *u64) c_int;
extern fn ref_audio_free(p: ?[*]f32) void;
extern fn ref_mp3_dct3_9(y: [*]f32) void;
extern fn ref_mp3_imdct36(grbuf: [*]f32, overlap: [*]f32, window: [*]const f32, nbands: c_int) void;
extern fn ref_mp3_imdct_short(grbuf: [*]f32, overlap: [*]f32, nbands: c_int) void;
extern fn ref_mp3_antialias(grbuf: [*]f32, nbands: c_int) void;
extern fn ref_mp3_dct2(grbuf: [*]f32, n: c_int) void;
extern fn ref_mp3_synth_granule(qmf: [*]f32, grbuf: [*]f32, nbands: c_int, nch: c_int, pcm: [*]i16, lins: [*]f32) void;
extern fn ref_mp3_pow43(x: c_int) f32;
extern fn ref_mp3_dec_size() usize;
extern fn ref_mp3_synth_pair(pcm: [*]i16, nch: c_int, z: [*]const f32) void;
extern fn ref_frame_count_after_resampling(out: c_uint, in: c_uint, n: u64) u64;
extern fn ref_mp3_decode_scalefactors(hdr: [*]const u8, ist_pos: [*]u8, bs: *anyopaque, gr: *const anyopaque, scf: [*]f32, ch: c_int) void;
extern fn ref_mp3_bs_get_bits(bs: *anyopaque, n: c_int) u32;
extern fn ref_mp3_decode_frame(dec: *anyopaque, mp3: [*]const u8, mp3_bytes: c_int, pcm: [*]i16, frame_bytes: *c_int) c_int;
extern fn ref_mp3_ldexp_q2(y: f32, e: c_int) f32;

/// A length past which neither side allocates: a corrupt header can claim
/// billions of frames, and upstream would then zero-fill gigabytes.
const frame_cap: u64 = 1 << 26;

const Stats = struct {
    image_checks: u64 = 0,
    audio_checks: u64 = 0,
    image_fail: u64 = 0,
    audio_fail: u64 = 0,
    image_ok_both: u64 = 0,
    audio_ok_both: u64 = 0,
    reported: u64 = 0,
    ref_crashes: u64 = 0,
    indeterminate: u64 = 0,
    flac_checks: u64 = 0,
    flac_ok_both: u64 = 0,
};

var stats: Stats = .{};
var quick = false;
var verbose = false;
var list_outcomes = false;

fn report(comptime fmt: []const u8, args: anytype) void {
    stats.reported += 1;
    if (verbose or stats.reported <= 40) std.debug.print("DIFF " ++ fmt ++ "\n", args);
}

extern "c" fn fork() c_int;
extern "c" fn waitpid(pid: c_int, status: *c_int, options: c_int) c_int;
extern "c" fn _exit(code: c_int) noreturn;
extern "c" fn signal(sig: c_int, handler: ?*const fn (c_int) callconv(.c) void) ?*const anyopaque;

/// Whether the reference crashes on `buf`, asked in a child process. Only
/// a PIC can: a failed core load hands a null buffer to
/// `stbi__convert_format` (see `stb_image.zig`'s `picLoadImpl`).
fn refCrashes(buf: []const u8, req: c_int) bool {
    if (buf.len < 92 or !std.mem.eql(u8, buf[0..4], "\x53\x80\xF6\x34") or !std.mem.eql(u8, buf[88..92], "PICT")) return false;
    const pid = fork();
    if (pid == 0) {
        // the expected crash needs no stack trace from a safe build's handler
        _ = signal(11, null); // SIGSEGV, SIG_DFL
        _ = signal(10, null); // SIGBUS on Darwin
        _ = signal(7, null); // SIGBUS on Linux
        var x: c_int = 0;
        var y: c_int = 0;
        var c: c_int = 0;
        var reason: ?[*:0]const u8 = null;
        ref_stbi_free(ref_stbi_load(buf.ptr, @intCast(@min(buf.len, std.math.maxInt(c_int))), req, &x, &y, &c, &reason));
        _exit(0);
    }
    var status: c_int = 0;
    _ = waitpid(pid, &status, 0);
    return (status & 0x7f) != 0; // terminated by a signal
}

fn checkImage(gpa: std.mem.Allocator, name: []const u8, buf: []const u8, req: c_int) void {
    stats.image_checks += 1;
    if (refCrashes(buf, req)) {
        stats.ref_crashes += 1;
        switch (stb.loadFromMemory(gpa, buf, req)) {
            .ok => |img| {
                gpa.free(img.pixels);
                stats.image_fail += 1;
                report("{s} req={d}: the reference crashes, the port decoded", .{ name, req });
            },
            .failed => {},
        }
        return;
    }
    var rx: c_int = 0;
    var ry: c_int = 0;
    var rc: c_int = 0;
    var reason: ?[*:0]const u8 = null;
    const len: c_int = @intCast(@min(buf.len, std.math.maxInt(c_int)));
    const rdata = ref_stbi_load(buf.ptr, len, req, &rx, &ry, &rc, &reason);
    defer ref_stbi_free(rdata);
    const port = stb.loadFromMemory(gpa, buf, req);
    switch (port) {
        .ok => |img| {
            defer gpa.free(img.pixels);
            if (rdata == null) {
                stats.image_fail += 1;
                return report("{s} req={d}: port decoded {d}x{d}, ref failed ({s})", .{ name, req, img.x, img.y, if (reason) |r| std.mem.span(r) else "(null)" });
            }
            if (img.x != rx or img.y != ry or img.comp != rc) {
                stats.image_fail += 1;
                return report("{s} req={d}: shape port {d}x{d}x{d} ref {d}x{d}x{d}", .{ name, req, img.x, img.y, img.comp, rx, ry, rc });
            }
            const out_n: usize = @intCast(if (req != 0) req else rc);
            const n = @as(usize, @intCast(rx)) * @as(usize, @intCast(ry)) * out_n;
            if (img.pixels.len != n) {
                stats.image_fail += 1;
                return report("{s} req={d}: port returned {d} bytes, expected {d}", .{ name, req, img.pixels.len, n });
            }
            if (stb.read_indeterminate) {
                // the C read memory it never wrote: the pixels are unspecified
                stats.indeterminate += 1;
                return;
            }
            if (!std.mem.eql(u8, img.pixels, rdata.?[0..n])) {
                stats.image_fail += 1;
                const i = std.mem.indexOfDiff(u8, img.pixels, rdata.?[0..n]).?;
                return report("{s} req={d}: byte {d} port {d} ref {d}", .{ name, req, i, img.pixels[i], rdata.?[i] });
            }
            stats.image_ok_both += 1;
        },
        .failed => |why| {
            if (rdata != null) {
                stats.image_fail += 1;
                return report("{s} req={d}: ref decoded {d}x{d}, port failed ({s})", .{ name, req, rx, ry, why orelse "(null)" });
            }
            const rs: ?[]const u8 = if (reason) |r| std.mem.span(r) else null;
            if (!optEq(rs, why)) {
                stats.image_fail += 1;
                return report("{s} req={d}: reason port '{s}' ref '{s}'", .{ name, req, why orelse "(null)", rs orelse "(null)" });
            }
        },
    }
}

fn optEq(a: ?[]const u8, b: ?[]const u8) bool {
    if (a == null and b == null) return true;
    if (a == null or b == null) return false;
    return std.mem.eql(u8, a.?, b.?);
}

fn checkAudio(gpa: std.mem.Allocator, name: []const u8, buf: []const u8, rate: c_int) void {
    stats.audio_checks += 1;
    if (ma.dr_flac.claims(buf)) stats.flac_checks += 1;
    var rout: ?[*]f32 = null;
    var rn: u64 = 0;
    const rstage = ref_audio_decode(buf.ptr, buf.len, rate, frame_cap, &rout, &rn);
    defer ref_audio_free(rout);

    ma.resetIndeterminate();
    var pstage: c_int = 0;
    var pn: u64 = 0;
    var pout: []f32 = &.{};
    defer gpa.free(pout);
    if (ma.Decoder.initMemory(gpa, buf, @intCast(rate))) |dec_c| {
        var dec = dec_c;
        defer dec.deinit();
        pn = dec.lengthInPcmFrames();
        if (pn > frame_cap) {
            pstage = 3;
        } else {
            pout = gpa.alloc(f32, @intCast(pn)) catch @panic("oom");
            @memset(pout, 0);
            const r = dec.readPcmFrames(pout);
            pstage = if (r.ok) 0 else 4;
        }
    } else pstage = 1;

    if (list_outcomes and std.mem.indexOfScalar(u8, name, '#') == null) std.debug.print("LIST {s} rate={d}: stage {d}, {d} frames\n", .{ name, rate, rstage, rn });
    if (pstage != rstage or (pstage != 1 and pn != rn)) {
        stats.audio_fail += 1;
        return report("{s} rate={d}: port stage {d} len {d}, ref stage {d} len {d}", .{ name, rate, pstage, pn, rstage, rn });
    }
    if (ma.readIndeterminate()) {
        stats.indeterminate += 1;
        return;
    }
    if (pstage == 0 or pstage == 4) {
        const r = rout.?[0..@intCast(rn)];
        for (pout, r, 0..) |a, b, i| {
            if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) {
                stats.audio_fail += 1;
                return report("{s} rate={d}: sample {d}/{d} port {e} ref {e}", .{ name, rate, i, rn, a, b });
            }
        }
        if (pstage == 0) stats.audio_ok_both += 1;
        if (pstage == 0 and ma.dr_flac.claims(buf)) stats.flac_ok_both += 1;
    }
}

/// 0 keeps the file's own rate: the backend's samples, unconverted.
const rates = [_]c_int{ 16000, 24000, 0, 44100, 48000, 8000 };

fn checkAll(gpa: std.mem.Allocator, name: []const u8, buf: []const u8, base: bool) void {
    if (only) |o| if (std.mem.indexOf(u8, name, o) == null) return;
    if (base and ma.isAudioFile(buf) and (std.mem.endsWith(u8, name, ".mp3") or std.mem.startsWith(u8, name, "synth/mp3"))) mp3Frames(gpa, name, buf);
    if (ma.isAudioFile(buf)) {
        const nrates: usize = if (base and !quick) rates.len else 2;
        for (rates[0..nrates]) |r| checkAudio(gpa, name, buf, r);
    }
    checkImage(gpa, name, buf, 3);
    if (base) for ([_]c_int{ 0, 1, 2, 4 }) |r| checkImage(gpa, name, buf, r);
}

var only: ?[]const u8 = null;

/// Every truncation and corruption a base file gets.
fn mutate(gpa: std.mem.Allocator, name: []const u8, buf: []const u8) !void {
    var label: std.ArrayList(u8) = .empty;
    defer label.deinit(gpa);
    if (buf.len == 0) return;
    // truncations
    const cuts = [_]usize{ 1, 2, 3, 4, 8, 11, 12, 16, 24, 32, 44, 64, 100, 128, 256, 512 };
    for (cuts) |c| if (c < buf.len) try runMut(gpa, &label, name, "cut", c, buf[0..c]);
    var k: usize = 1;
    while (k < 16) : (k += 1) try runMut(gpa, &label, name, "cut", buf.len * k / 16, buf[0 .. buf.len * k / 16]);
    try runMut(gpa, &label, name, "cut", buf.len - 1, buf[0 .. buf.len - 1]);

    // corruptions, seeded by the name so a failure reproduces
    var prng = std.Random.DefaultPrng.init(std.hash.Wyhash.hash(0, name));
    const rnd = prng.random();
    const copy = try gpa.dupe(u8, buf);
    defer gpa.free(copy);
    const n_flips: usize = if (quick) 24 else 96;
    for (0..n_flips) |i| {
        @memcpy(copy, buf);
        const nbytes = 1 + rnd.uintLessThan(usize, 4);
        for (0..nbytes) |_| {
            // half the corruptions land in the first 256 bytes, where the headers are
            const pos = if (i % 2 == 0) rnd.uintLessThan(usize, @min(buf.len, 256)) else rnd.uintLessThan(usize, buf.len);
            copy[pos] = switch (rnd.uintLessThan(u8, 4)) {
                0 => rnd.int(u8),
                1 => copy[pos] ^ (@as(u8, 1) << rnd.int(u3)),
                2 => 0xff,
                else => 0,
            };
        }
        try runMut(gpa, &label, name, "flip", i, copy);
    }
}

fn runMut(gpa: std.mem.Allocator, label: *std.ArrayList(u8), name: []const u8, kind: []const u8, i: usize, bytes: []const u8) !void {
    label.clearRetainingCapacity();
    try label.print(gpa, "{s}#{s}{d}", .{ name, kind, i });
    checkAll(gpa, label.items, bytes, false);
}

fn runFile(gpa: std.mem.Allocator, name: []const u8, bytes: []const u8) !void {
    checkAll(gpa, name, bytes, true);
    try mutate(gpa, name, bytes);
}

/// Decodes `buf` frame by frame through both decoders and compares each
/// frame's PCM and the decoder state after it: where a whole-file diff
/// names a sample, this names the frame and the field.
fn mp3Frames(gpa: std.mem.Allocator, name: []const u8, buf: []const u8) void {
    const Dec = ma.dr_mp3.kernels.Decoder;
    if (ref_mp3_dec_size() != @sizeOf(Dec)) {
        stats.audio_fail += 1;
        return report("mp3 decoder layout: C {d} bytes, port {d}", .{ ref_mp3_dec_size(), @sizeOf(Dec) });
    }
    const dr = gpa.create(Dec) catch @panic("oom");
    defer gpa.destroy(dr);
    const dp = gpa.create(Dec) catch @panic("oom");
    defer gpa.destroy(dp);
    @memset(std.mem.asBytes(dr), 0);
    @memset(std.mem.asBytes(dp), 0);
    var pos: usize = 0;
    var frame: usize = 0;
    while (pos < buf.len and frame < 100000) : (frame += 1) {
        var pr: [1152 * 2]i16 = @splat(0);
        var pp: [1152 * 2]i16 = @splat(0);
        var fr: c_int = 0;
        var fp: c_int = 0;
        const nr = ref_mp3_decode_frame(dr, buf[pos..].ptr, @intCast(buf.len - pos), &pr, &fr);
        const np = ma.dr_mp3.kernels.decodeOne(dp, buf[pos..], &pp, &fp);
        stats.audio_checks += 1;
        if (nr != np or fr != fp) {
            stats.audio_fail += 1;
            return report("{s} frame {d}: samples {d}/{d}, bytes {d}/{d} (ref/port)", .{ name, frame, nr, np, fr, fp });
        }
        var state_differs = false;
        inline for (.{ "mdct_overlap", "qmf_state", "reserv", "reserv_buf", "header" }) |f| {
            if (!std.mem.eql(u8, std.mem.asBytes(&@field(dr, f)), std.mem.asBytes(&@field(dp, f)))) state_differs = true;
        }
        inline for (.{ "grbuf", "scf", "ist_pos", "maindata" }) |f| {
            if (!std.mem.eql(u8, std.mem.asBytes(&@field(dr.scratch, f)), std.mem.asBytes(&@field(dp.scratch, f)))) state_differs = true;
        }
        if (state_differs or !std.mem.eql(i16, &pr, &pp)) {
            stats.audio_fail += 1;
            const i = std.mem.indexOfDiff(i16, &pr, &pp) orelse 0;
            report("{s} frame {d}: pcm[{d}] ref {d} port {d}", .{ name, frame, i, pr[i], pp[i] });
            inline for (.{ "reserv", "reserv_buf", "header" }) |f| {
                if (!std.mem.eql(u8, std.mem.asBytes(&@field(dr, f)), std.mem.asBytes(&@field(dp, f)))) report("  {s} differs", .{f});
            }
            {
                const a = std.mem.bytesAsSlice(u8, std.mem.asBytes(&dr.scratch.ist_pos));
                const b = std.mem.bytesAsSlice(u8, std.mem.asBytes(&dp.scratch.ist_pos));
                if (std.mem.indexOfDiff(u8, a, b)) |j| report("  ist_pos[{d}] ref {d} port {d}", .{ j, a[j], b[j] });
                const ga: []const f32 = std.mem.bytesAsSlice(f32, std.mem.asBytes(&dr.scratch.grbuf));
                const gb: []const f32 = std.mem.bytesAsSlice(f32, std.mem.asBytes(&dp.scratch.grbuf));
                if (bitsEq(ga, gb)) |j| report("  grbuf[{d}] ref {e} port {e}", .{ j, ga[j], gb[j] });
                const sa: []const f32 = &dr.scratch.scf;
                const sb: []const f32 = &dp.scratch.scf;
                if (bitsEq(sa, sb)) |j| report("  scf[{d}] ref {e} port {e}", .{ j, sa[j], sb[j] });
                for (0..4) |g| {
                    const ra = std.mem.asBytes(&dr.scratch.gr_info[g])[8..];
                    const rb = std.mem.asBytes(&dp.scratch.gr_info[g])[8..];
                    report("  gr_info[{d}] ref {any}", .{ g, ra });
                    if (!std.mem.eql(u8, ra, rb)) report("  gr_info[{d}] prt {any}", .{ g, rb });
                }
            }
            inline for (.{ "mdct_overlap", "qmf_state" }) |f| {
                if (!std.mem.eql(u8, std.mem.asBytes(&@field(dr, f)), std.mem.asBytes(&@field(dp, f)))) report("  {s} differs", .{f});
            }
            inline for (.{ "grbuf", "scf", "syn", "ist_pos", "maindata" }) |f| {
                if (!std.mem.eql(u8, std.mem.asBytes(&@field(dr.scratch, f)), std.mem.asBytes(&@field(dp.scratch, f)))) report("  scratch.{s} differs", .{f});
            }
            return;
        }
        if (fr <= 0) break;
        pos += @intCast(fr);
    }
}

fn bitsEq(a: []const f32, b: []const f32) ?usize {
    for (a, b, 0..) |x, y, i| if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) return i;
    return null;
}

fn kernelFail(comptime what: []const u8, trial: usize, i: usize) void {
    stats.audio_fail += 1;
    report("mp3 kernel " ++ what ++ " trial {d}: element {d} differs", .{ trial, i });
}

/// `ma_calculate_frame_count_after_resampling` over every pair of common
/// rates and the lengths around each rate's multiples, where its `<=`
/// decides an extra frame.
fn lengthSweep() void {
    const rate_list = [_]u32{ 8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 7, 1 };
    for (rate_list) |out| for (rate_list) |in| {
        var n: u64 = 0;
        while (n < 2000) : (n += 1) {
            const a = ref_frame_count_after_resampling(out, in, n);
            const b = ma.frameCountAfterResampling(out, in, n);
            stats.audio_checks += 1;
            if (a != b) {
                stats.audio_fail += 1;
                return report("frame count after resampling {d} -> {d}, {d} frames: ref {d} port {d}", .{ in, out, n, a, b });
            }
        }
    };
}

/// `ma_dr_mp3d_synth_pair` on many windows at amplitudes where `int16`
/// rounding is decided by the last bit: its contractions show nowhere else.
fn synthPairSweep() void {
    var prng = std.Random.DefaultPrng.init(0x5151);
    const rnd = prng.random();
    const trials: usize = if (quick) 20000 else 200000;
    var z: [16 * 64]f32 = undefined;
    for (0..trials) |t| {
        const amp: f32 = 0.01 + 0.4 * rnd.float(f32);
        for (&z) |*x| x.* = (rnd.float(f32) * 2 - 1) * amp;
        var pa: [64]i16 = @splat(0);
        var pb: [64]i16 = @splat(0);
        ref_mp3_synth_pair(&pa, 1, &z);
        ma.dr_mp3.kernels.synthPair_(&pb, 1, &z);
        if (pa[0] != pb[0] or pa[16] != pb[16]) {
            kernelFail("synth_pair", t, 0);
            return;
        }
    }
    stats.audio_checks += trials;
}

/// dr_mp3's kernels on random inputs, each beside the reference's own.
fn mp3Kernels() void {
    const k = ma.dr_mp3.kernels;
    var prng = std.Random.DefaultPrng.init(0x3333);
    const rnd = prng.random();
    const trials: usize = if (quick) 200 else 2000;
    for (0..trials) |t| {
        const amp: f32 = switch (t % 4) {
            0 => 1,
            1 => 1000,
            2 => 1e-3,
            else => 30000,
        };
        var a: [2400]f32 = undefined;
        for (&a) |*x| x.* = (rnd.float(f32) * 2 - 1) * amp;
        var b = a;
        var ov_a: [2 * 288]f32 = undefined;
        for (&ov_a) |*x| x.* = (rnd.float(f32) * 2 - 1) * amp;
        var ov_b = ov_a;

        ref_mp3_dct3_9(&a);
        k.dct39(b[0..9]);
        if (bitsEq(a[0..9], b[0..9])) |i| kernelFail("dct3_9", t, i);

        const stop = t % 2 == 1;
        ref_mp3_imdct36(&a, &ov_a, k.mdct_window[if (stop) 18 else 0..].ptr, 32);
        k.imdct36(&b, &ov_b, true, stop, 32);
        if (bitsEq(a[0..576], b[0..576]) orelse bitsEq(&ov_a, &ov_b)) |i| kernelFail("imdct36", t, i);

        ref_mp3_imdct_short(&a, &ov_a, 30);
        k.imdctShort(&b, &ov_b, 30);
        if (bitsEq(a[0..576], b[0..576]) orelse bitsEq(&ov_a, &ov_b)) |i| kernelFail("imdct_short", t, i);

        ref_mp3_antialias(&a, 31);
        k.antialias(&b, 31);
        if (bitsEq(a[0..576], b[0..576])) |i| kernelFail("antialias", t, i);

        const n: usize = if (t % 3 == 0) 12 else 18;
        ref_mp3_dct2(&a, @intCast(n));
        k.dctII(&b, n);
        ref_mp3_dct2(a[576..].ptr, @intCast(n));
        k.dctII(b[576..].ptr, n);
        if (bitsEq(a[0..1152], b[0..1152])) |i| kernelFail("DCT_II", t, i);

        var qa: [960]f32 = undefined;
        for (&qa) |*x| x.* = (rnd.float(f32) * 2 - 1) * amp;
        var qb = qa;
        var la: [33 * 64]f32 = @splat(0);
        var lb: [33 * 64]f32 = @splat(0);
        var pa: [1152 * 2]i16 = @splat(0);
        var pb: [1152 * 2]i16 = @splat(0);
        const nch: usize = 1 + t % 2;
        for (&a, &b) |*x, *y| {
            x.* = (rnd.float(f32) * 2 - 1) * amp;
            y.* = x.*;
        }
        ref_mp3_synth_granule(&qa, &a, @intCast(n), @intCast(nch), &pa, &la);
        k.synthGranule_(&qb, &b, n, nch, &pb, &lb);
        if (!std.mem.eql(i16, &pa, &pb)) kernelFail("synth_granule pcm", t, std.mem.indexOfDiff(i16, &pa, &pb).?);
        if (bitsEq(&qa, &qb)) |i| kernelFail("synth_granule qmf", t, i);

        {
            var bytes: [64]u8 = undefined;
            rnd.bytes(&bytes);
            const Bs = k.Bitstream;
            var ba: Bs = .{ .buf = &bytes, .pos = 0, .limit = 64 * 8 - rnd.intRangeLessThan(c_int, 0, 64) };
            var bb = ba;
            for (0..40) |_| {
                const nb = rnd.intRangeAtMost(c_int, 0, 24);
                const va = ref_mp3_bs_get_bits(&ba, nb);
                const vb = k.getBits(&bb, nb);
                if (va != vb or ba.pos != bb.pos) kernelFail("bs_get_bits", t, @intCast(nb));
            }
            // scalefactors over random side info
            var gra: [4]k.Granule = std.mem.zeroes([4]k.Granule);
            var bsi: Bs = .{ .buf = &bytes, .pos = 0, .limit = 64 * 8 };
            const hdrs = [_][4]u8{ .{ 0xFF, 0xFB, 0x90, 0xC4 }, .{ 0xFF, 0xFB, 0x90, 0x44 }, .{ 0xFF, 0xF3, 0x90, 0x54 }, .{ 0xFF, 0xF3, 0x90, 0xC4 } };
            const hdr = &hdrs[t % hdrs.len];
            if (k.readSideInfo(&bsi, &gra, hdr) >= 0) {
                var data2: [64]u8 = undefined;
                rnd.bytes(&data2);
                // a mono header fills one granule per channel's worth
                const sf_nch: usize = if ((hdr[3] & 0xC0) == 0xC0) 1 else 2;
                for (0..sf_nch) |ch| {
                    var bsa: Bs = .{ .buf = &data2, .pos = 0, .limit = 64 * 8 };
                    var bsb = bsa;
                    var ista: [39]u8 = undefined;
                    rnd.bytes(&ista);
                    var istb = ista;
                    var sca: [40]f32 = @splat(0);
                    var scb: [40]f32 = @splat(0);
                    ref_mp3_decode_scalefactors(hdr, &ista, &bsa, &gra[ch], &sca, @intCast(ch));
                    k.decodeScalefactors(hdr, &istb, &bsb, &gra[ch], &scb, @intCast(ch));
                    if (!std.mem.eql(u8, &ista, &istb) or bsa.pos != bsb.pos) {
                        kernelFail("decode_scalefactors ist_pos", t, std.mem.indexOfDiff(u8, &ista, &istb) orelse 99);
                        report("  hdr {d} ch {d} gr {any} pos {d}/{d}\n  ref {any}\n  prt {any}", .{ t % hdrs.len, ch, std.mem.asBytes(&gra[ch])[8..], bsa.pos, bsb.pos, ista, istb });
                    }
                    if (bitsEq(&sca, &scb)) |i| kernelFail("decode_scalefactors scf", t, i);
                }
            }
        }
        const x: c_int = rnd.intRangeAtMost(c_int, 0, 8206);
        if (@as(u32, @bitCast(ref_mp3_pow43(x))) != @as(u32, @bitCast(k.pow43(x)))) kernelFail("pow_43", t, @intCast(x));
        const e: c_int = rnd.intRangeAtMost(c_int, 0, 400);
        if (@as(u32, @bitCast(ref_mp3_ldexp_q2(amp, e))) != @as(u32, @bitCast(k.ldexpQ2(amp, e)))) kernelFail("ldexp_q2", t, @intCast(e));
        stats.audio_checks += 1;
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var paths: std.ArrayList([]const u8) = .empty;
    defer paths.deinit(gpa);
    for (args[1..]) |a| {
        if (std.mem.eql(u8, a, "--quick")) {
            quick = true;
        } else if (std.mem.eql(u8, a, "--list")) {
            list_outcomes = true;
        } else if (std.mem.eql(u8, a, "--verbose")) {
            verbose = true;
        } else if (std.mem.startsWith(u8, a, "--only=")) {
            only = a["--only=".len..];
        } else try paths.append(gpa, a);
    }

    mp3Kernels();
    synthPairSweep();
    lengthSweep();
    var corpus = try synth(gpa);
    defer {
        for (corpus.items) |f| {
            gpa.free(f.name);
            gpa.free(f.bytes);
        }
        corpus.deinit(gpa);
    }
    var files: usize = 0;
    for (corpus.items) |f| {
        try runFile(gpa, f.name, f.bytes);
        files += 1;
    }
    for (paths.items) |p| files += try walk(gpa, io, p);

    std.debug.print("mtmd-decode-diff: {d} files; {d} image checks ({d} decoded on both), {d} audio checks ({d} decoded on both); {d} reference crashes, {d} unspecified; FLAC {d} checks ({d} decoded on both); {d} differ\n", .{ files, stats.image_checks, stats.image_ok_both, stats.audio_checks, stats.audio_ok_both, stats.ref_crashes, stats.indeterminate, stats.flac_checks, stats.flac_ok_both, stats.image_fail + stats.audio_fail });
    if (stats.image_fail + stats.audio_fail != 0) std.process.exit(1);
    if (stats.image_ok_both == 0 or stats.audio_ok_both == 0 or (stats.flac_checks > 0 and stats.flac_ok_both == 0)) {
        std.debug.print("FAIL: nothing decoded on both sides; the gate compared only failures\n", .{});
        std.process.exit(1);
    }
}

fn walk(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !usize {
    const cwd = std.Io.Dir.cwd();
    const stat = cwd.statFile(io, path, .{}) catch |e| {
        std.debug.print("cannot stat {s}: {s}\n", .{ path, @errorName(e) });
        return error.BadPath;
    };
    if (stat.kind == .directory) {
        var dir = try cwd.openDir(io, path, .{ .iterate = true });
        defer dir.close(io);
        var names: std.ArrayList([]u8) = .empty;
        defer {
            for (names.items) |n| gpa.free(n);
            names.deinit(gpa);
        }
        var it = dir.iterate();
        while (try it.next(io)) |e| try names.append(gpa, try std.fs.path.join(gpa, &.{ path, e.name }));
        std.mem.sort([]u8, names.items, {}, struct {
            fn lt(_: void, a: []u8, b: []u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);
        var n: usize = 0;
        for (names.items) |p| n += try walk(gpa, io, p);
        return n;
    }
    const bytes = try cwd.readFileAlloc(io, path, gpa, .limited(64 << 20));
    defer gpa.free(bytes);
    try runFile(gpa, path, bytes);
    return 1;
}

// -----------------------------------------------------------------------------
// synthesized inputs

const File = struct { name: []u8, bytes: []u8 };

const Builder = struct {
    gpa: std.mem.Allocator,
    b: std.ArrayList(u8) = .empty,

    fn u8_(s: *Builder, v: u8) void {
        s.b.append(s.gpa, v) catch @panic("oom");
    }
    fn bytes(s: *Builder, v: []const u8) void {
        s.b.appendSlice(s.gpa, v) catch @panic("oom");
    }
    fn le16(s: *Builder, v: u16) void {
        s.bytes(&std.mem.toBytes(std.mem.nativeToLittle(u16, v)));
    }
    fn le32(s: *Builder, v: u32) void {
        s.bytes(&std.mem.toBytes(std.mem.nativeToLittle(u32, v)));
    }
    fn be16(s: *Builder, v: u16) void {
        s.bytes(&std.mem.toBytes(std.mem.nativeToBig(u16, v)));
    }
    fn be32(s: *Builder, v: u32) void {
        s.bytes(&std.mem.toBytes(std.mem.nativeToBig(u32, v)));
    }
    fn patch32le(s: *Builder, at: usize, v: u32) void {
        std.mem.writeInt(u32, s.b.items[at..][0..4], v, .little);
    }
};

fn add(gpa: std.mem.Allocator, list: *std.ArrayList(File), name: []const u8, b: *Builder) void {
    list.append(gpa, .{ .name = gpa.dupe(u8, name) catch @panic("oom"), .bytes = b.b.toOwnedSlice(gpa) catch @panic("oom") }) catch @panic("oom");
}

fn synth(gpa: std.mem.Allocator) !std.ArrayList(File) {
    var list: std.ArrayList(File) = .empty;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rnd = prng.random();
    var name_buf: [128]u8 = undefined;

    // WAV: every format tag dr_wav decodes
    const WavSpec = struct { tag: u16, bits: u16, ch: u16, rate: u32, ext: bool = false, block: u16 = 0, extra: bool = false };
    const wav_specs = [_]WavSpec{
        .{ .tag = 1, .bits = 8, .ch = 1, .rate = 16000 },
        .{ .tag = 1, .bits = 16, .ch = 1, .rate = 16000 },
        .{ .tag = 1, .bits = 16, .ch = 2, .rate = 44100 },
        .{ .tag = 1, .bits = 16, .ch = 1, .rate = 22050, .extra = true },
        .{ .tag = 1, .bits = 24, .ch = 2, .rate = 48000 },
        .{ .tag = 1, .bits = 32, .ch = 1, .rate = 8000 },
        .{ .tag = 1, .bits = 16, .ch = 6, .rate = 48000 },
        .{ .tag = 1, .bits = 16, .ch = 3, .rate = 11025, .ext = true },
        .{ .tag = 1, .bits = 12, .ch = 2, .rate = 16000, .block = 4 },
        .{ .tag = 1, .bits = 40, .ch = 1, .rate = 16000 },
        .{ .tag = 1, .bits = 64, .ch = 1, .rate = 16000 },
        .{ .tag = 3, .bits = 32, .ch = 1, .rate = 16000 },
        .{ .tag = 3, .bits = 32, .ch = 2, .rate = 96000, .ext = true },
        .{ .tag = 3, .bits = 64, .ch = 2, .rate = 44100 },
        .{ .tag = 6, .bits = 8, .ch = 1, .rate = 8000 },
        .{ .tag = 7, .bits = 8, .ch = 2, .rate = 16000 },
        .{ .tag = 2, .bits = 4, .ch = 1, .rate = 16000, .block = 256 },
        .{ .tag = 2, .bits = 4, .ch = 2, .rate = 22050, .block = 512 },
        .{ .tag = 0x11, .bits = 4, .ch = 1, .rate = 16000, .block = 256 },
        .{ .tag = 0x11, .bits = 4, .ch = 2, .rate = 44100, .block = 512 },
    };
    for (wav_specs, 0..) |sp, idx| {
        var b: Builder = .{ .gpa = gpa };
        const bpf: u16 = if (sp.block != 0) sp.block else sp.ch * (sp.bits / 8);
        const frames: usize = 3000 + idx * 37;
        b.bytes("RIFF");
        b.le32(0);
        b.bytes("WAVE");
        if (sp.extra) {
            b.bytes("LIST");
            b.le32(5);
            b.bytes("INFOx\x00"); // odd size, padded
        }
        b.bytes("fmt ");
        b.le32(if (sp.ext) 40 else if (sp.tag == 2 or sp.tag == 0x11) 20 else 16);
        b.le16(if (sp.ext) 0xFFFE else sp.tag);
        b.le16(sp.ch);
        b.le32(sp.rate);
        b.le32(sp.rate * bpf);
        b.le16(bpf);
        b.le16(sp.bits);
        if (sp.ext) {
            b.le16(22);
            b.le16(sp.bits);
            b.le32(0);
            b.le16(sp.tag);
            b.bytes("\x00\x00\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71");
        } else if (sp.tag == 2 or sp.tag == 0x11) {
            b.le16(2);
            b.le16(if (sp.tag == 2) (sp.block - 7 * sp.ch) * 2 / sp.ch + 2 else (sp.block - 4 * sp.ch) * 2 / sp.ch + 1);
        }
        if (sp.extra) {
            b.bytes("fact");
            b.le32(4);
            b.le32(@intCast(frames));
        }
        b.bytes("data");
        const data_size: usize = if (sp.block != 0 and (sp.tag == 2 or sp.tag == 0x11)) 6 * @as(usize, sp.block) + sp.block / 3 else frames * bpf;
        b.le32(@intCast(data_size));
        const start = b.b.items.len;
        for (0..data_size) |i| {
            // a waveform for PCM, noise for the compressed streams
            const t: f32 = @floatFromInt(i);
            b.u8_(if (sp.tag == 2 or sp.tag == 0x11) rnd.int(u8) else @truncate(@as(u32, @intFromFloat(@abs(@sin(t * 0.013) * 255.0))) ^ rnd.uintLessThan(u8, 4)));
        }
        if (sp.tag == 2) {
            // block headers with valid predictors
            var off: usize = 0;
            while (off + sp.block <= data_size) : (off += sp.block) {
                for (0..sp.ch) |c| b.b.items[start + off + c] = rnd.uintLessThan(u8, 7);
            }
        } else if (sp.tag == 0x11) {
            var off: usize = 0;
            while (off + sp.block <= data_size) : (off += sp.block) {
                for (0..sp.ch) |c| b.b.items[start + off + c * 4 + 2] = rnd.uintLessThan(u8, 89);
            }
        } else if (sp.tag == 3 and sp.bits == 32) {
            var i: usize = 0;
            while (i + 4 <= data_size) : (i += 4) std.mem.writeInt(u32, b.b.items[start + i ..][0..4], @bitCast(@sin(@as(f32, @floatFromInt(i)) * 0.01) * 0.9), .little);
        } else if (sp.tag == 3 and sp.bits == 64) {
            var i: usize = 0;
            while (i + 8 <= data_size) : (i += 8) std.mem.writeInt(u64, b.b.items[start + i ..][0..8], @bitCast(@cos(@as(f64, @floatFromInt(i)) * 0.003) * 0.7), .little);
        }
        b.patch32le(4, @intCast(b.b.items.len - 8));
        add(gpa, &list, try std.fmt.bufPrint(&name_buf, "synth/wav-{d}-tag{x}-{d}bit-{d}ch-{d}", .{ idx, sp.tag, sp.bits, sp.ch, sp.rate }), &b);
    }

    try synthFlac(gpa, &list, rnd);
    try synthImages(gpa, &list, rnd);
    return list;
}

// -----------------------------------------------------------------------------
// a FLAC and Ogg-FLAC encoder, for the corpus: every subframe type, LPC
// order, residual coding and channel assignment dr_flac decodes

const BitWriter = struct {
    gpa: std.mem.Allocator,
    bytes: std.ArrayList(u8) = .empty,
    acc: u64 = 0,
    nbits: u6 = 0,

    fn put(w: *BitWriter, value: u64, count: u32) void {
        var c = count;
        while (c > 0) {
            c -= 1;
            const bit: u64 = (value >> @intCast(c)) & 1;
            w.acc = (w.acc << 1) | bit;
            w.nbits += 1;
            if (w.nbits == 8) {
                w.bytes.append(w.gpa, @truncate(w.acc)) catch @panic("oom");
                w.acc = 0;
                w.nbits = 0;
            }
        }
    }
    fn putSigned(w: *BitWriter, value: i64, count: u32) void {
        w.put(@bitCast(value), count);
    }
    fn align8(w: *BitWriter) void {
        while (w.nbits != 0) w.put(0, 1);
    }
};

fn crc8Bits(data: []const u8) u8 {
    var crc: u8 = 0;
    for (data) |b| {
        crc ^= b;
        for (0..8) |_| crc = if (crc & 0x80 != 0) (crc << 1) ^ 0x07 else crc << 1;
    }
    return crc;
}

fn crc16Bits(data: []const u8) u16 {
    var crc: u16 = 0;
    for (data) |b| {
        crc ^= @as(u16, b) << 8;
        for (0..8) |_| crc = if (crc & 0x8000 != 0) (crc << 1) ^ 0x8005 else crc << 1;
    }
    return crc;
}

fn oggCrc(data: []const u8) u32 {
    var crc: u32 = 0;
    for (data) |b| {
        crc ^= @as(u32, b) << 24;
        for (0..8) |_| crc = if (crc & 0x80000000 != 0) (crc << 1) ^ 0x04C11DB7 else crc << 1;
    }
    return crc;
}

const FlacSpec = struct {
    channels: u32,
    bps: u32,
    rate: u32,
    block: u32, // 0: vary per frame
    frames: u32,
    variable: bool = false,
    assignment: u8 = 0, // for stereo: 0 independent (code 1), 8 left/side, 9 right/side, 10 mid/side, 255 rotate
    kinds: []const u8 = "cvfl", // subframe kinds to rotate through
    max_order: u32 = 12,
    rice2: bool = false,
    escape: bool = false,
    wasted: bool = false,
    rate_code: u8 = 255, // 255: pick from the table or 12/13/14
    metadata: bool = true,
    seektable: bool = false,
    total_zero: bool = false,
    /// Every LPC coefficient at the precision's maximum, shifted 1-4: sums
    /// that pass 2^31 where the width rule still says 33 or 34 bits. The
    /// shift matters: unshifted, the 64-bit sum truncated to 32 bits is the
    /// wrapped 32-bit sum, and the two predictions agree.
    big_coefs: u32 = 0,
};

/// The decoder's prediction, so residuals reproduce samples exactly.
fn flacPredict(bps: u32, order: u32, precision: u32, shift: u5, coefs: []const i32, s: []const i32, n: usize) i32 {
    const ilog2: u32 = 32 - @as(u32, @clz(order));
    if (bps + precision + ilog2 > 32) {
        var p: i64 = 0;
        for (0..order) |j| p += @as(i64, coefs[j]) * s[n - 1 - j];
        return @truncate(p >> shift);
    }
    var p: i32 = 0;
    for (0..order) |j| p +%= coefs[j] *% s[n - 1 - j];
    return p >> shift;
}

fn writeResidual(w: *BitWriter, rnd: std.Random, spec: FlacSpec, res: []const i32, block: usize, order: usize) void {
    const rice2 = spec.rice2 or rnd.uintLessThan(u8, 4) == 0;
    w.put(@intFromBool(rice2), 2);
    var porder: u32 = rnd.uintLessThan(u32, 5);
    while (porder > 0 and ((block >> @intCast(porder)) < order or (block & ((@as(usize, 1) << @intCast(porder)) - 1)) != 0)) porder -= 1;
    w.put(porder, 4);
    const parts = @as(usize, 1) << @intCast(porder);
    var idx: usize = 0;
    for (0..parts) |pi| {
        const len = (block >> @intCast(porder)) - (if (pi == 0) order else 0);
        const part = res[idx..][0..len];
        idx += len;
        var max_u: u64 = 0;
        var sum_u: u64 = 0;
        for (part) |r| {
            const u: u32 = @bitCast((r << 1) ^ (r >> 31));
            max_u = @max(max_u, u);
            sum_u += u;
        }
        var param: u32 = 0;
        const mean = if (len > 0) sum_u / len else 0;
        while ((@as(u64, 1) << @intCast(param)) < mean + 1 and param < (if (rice2) @as(u32, 30) else 14)) param += 1;
        const too_long = len > 0 and (max_u >> @intCast(param)) > 4000;
        if ((spec.escape and rnd.boolean()) or too_long) {
            var nbits: u32 = 0;
            for (part) |r| {
                var need: u32 = 1;
                while (need < 32 and (r < -(@as(i64, 1) << @intCast(need - 1)) or r >= (@as(i64, 1) << @intCast(need - 1)))) need += 1;
                nbits = @max(nbits, need);
            }
            if (nbits <= 31) {
                w.put(if (rice2) 31 else 15, if (rice2) 5 else 4);
                w.put(nbits, 5);
                if (nbits > 0) for (part) |r| w.putSigned(r, nbits);
                continue;
            }
        }
        w.put(param, if (rice2) 5 else 4);
        for (part) |r| {
            const u: u32 = @bitCast((r << 1) ^ (r >> 31));
            const q = u >> @intCast(param);
            var k: u32 = 0;
            while (k < q) : (k += 1) w.put(0, 1);
            w.put(1, 1);
            if (param > 0) w.put(u & ((@as(u32, 1) << @intCast(param)) - 1), param);
        }
    }
}

fn writeSubframe(w: *BitWriter, gpa: std.mem.Allocator, rnd: std.Random, spec: FlacSpec, kind: u8, samples: []const i32, sbps: u32, wasted: u32) void {
    const block = samples.len;
    const bits = sbps - wasted;
    const s = gpa.alloc(i32, block) catch @panic("oom");
    defer gpa.free(s);
    for (samples, s) |x, *y| y.* = x >> @intCast(wasted);
    const type_code: u32, const order: u32 = switch (kind) {
        'c' => .{ 0, 0 },
        'v' => .{ 1, 0 },
        'f' => blk: {
            const o = @min(rnd.uintLessThan(u32, 5), @as(u32, @intCast(block)));
            break :blk .{ 8 | o, o };
        },
        else => blk: {
            const o = @min(1 + rnd.uintLessThan(u32, spec.max_order), @as(u32, @intCast(block)));
            break :blk .{ 32 | (o - 1), o };
        },
    };
    w.put(0, 1);
    w.put(type_code, 6);
    if (wasted > 0) {
        w.put(1, 1);
        w.put(0, wasted - 1);
        w.put(1, 1);
    } else w.put(0, 1);
    switch (kind) {
        'c' => w.putSigned(s[0], bits),
        'v' => for (s) |x| w.putSigned(x, bits),
        else => {
            for (s[0..order]) |x| w.putSigned(x, bits);
            var coefs: [32]i32 = @splat(0);
            var precision: u32 = 4;
            var shift: u5 = 0;
            if (kind == 'f') {
                const table = [5][4]i32{ .{ 0, 0, 0, 0 }, .{ 1, 0, 0, 0 }, .{ 2, -1, 0, 0 }, .{ 3, -3, 1, 0 }, .{ 4, -6, 4, -1 } };
                for (0..order) |j| coefs[j] = table[order][j];
            } else {
                precision = if (spec.big_coefs != 0) spec.big_coefs else 1 + rnd.uintLessThan(u32, 15);
                shift = if (spec.big_coefs != 0) @as(u5, 1) + rnd.int(u2) else rnd.int(u4);
                w.put(precision - 1, 4);
                w.put(shift, 5);
                const lim: i32 = @intCast((@as(u32, 1) << @intCast(precision - 1)) - 1);
                for (0..order) |j| {
                    coefs[j] = if (spec.big_coefs != 0) lim else if (j == 0) @min(lim, @as(i32, 1) << shift) else rnd.intRangeAtMost(i32, -@min(lim, 3), @min(lim, 3));
                    w.putSigned(coefs[j], precision);
                }
            }
            const res = gpa.alloc(i32, block) catch @panic("oom");
            defer gpa.free(res);
            for (order..block) |n| res[n - order] = s[n] -% flacPredict(bits, order, precision, shift, &coefs, s, n);
            writeResidual(w, rnd, spec, res[0 .. block - order], block, order);
        },
    }
}

fn utf8Number(out: *std.ArrayList(u8), gpa: std.mem.Allocator, v: u64) void {
    if (v < 0x80) return out.append(gpa, @intCast(v)) catch @panic("oom");
    var n: u32 = 2;
    while (n < 7 and v >= (@as(u64, 1) << @intCast(5 * n + 1))) n += 1;
    const lead_bits: u6 = @intCast(7 - n);
    out.append(gpa, @truncate((@as(u64, 0xFF) << @intCast(8 - n)) | (v >> @intCast(6 * (n - 1))) & ((@as(u64, 1) << lead_bits) - 1))) catch @panic("oom");
    var k: u32 = n - 1;
    while (k > 0) : (k -= 1) out.append(gpa, @truncate(0x80 | ((v >> @intCast(6 * (k - 1))) & 0x3F))) catch @panic("oom");
}

/// One FLAC stream: `fLaC`, metadata, frames.
fn flacStream(gpa: std.mem.Allocator, rnd: std.Random, spec: FlacSpec) std.ArrayList(u8) {
    var out: std.ArrayList(u8) = .empty;
    var blocks: std.ArrayList(u32) = .empty;
    defer blocks.deinit(gpa);
    const block_choices = [_]u32{ 192, 576, 1152, 2304, 4608, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 1, 17, 100, 255, 256, 257, 4000, 65535 };
    var total: u64 = 0;
    var max_block: u32 = 0;
    for (0..spec.frames) |_| {
        const b = if (spec.block != 0) spec.block else block_choices[rnd.uintLessThan(usize, block_choices.len - 2)];
        blocks.append(gpa, b) catch @panic("oom");
        total += b;
        max_block = @max(max_block, b);
    }
    out.appendSlice(gpa, "fLaC") catch @panic("oom");
    // STREAMINFO
    var si: [38]u8 = undefined;
    si[0] = if (spec.metadata or spec.seektable) 0x00 else 0x80;
    std.mem.writeInt(u24, si[1..4], 34, .big);
    std.mem.writeInt(u16, si[4..6], @intCast(if (spec.block != 0) spec.block else 16), .big);
    std.mem.writeInt(u16, si[6..8], @intCast(@min(max_block, 65535)), .big);
    @memset(si[8..14], 0);
    const props: u64 = (@as(u64, spec.rate) << 44) | (@as(u64, spec.channels - 1) << 41) | (@as(u64, spec.bps - 1) << 36) | (if (spec.total_zero) 0 else total & 0xFFFFFFFFF);
    std.mem.writeInt(u64, si[14..22], props, .big);
    @memset(si[22..38], 0);
    out.appendSlice(gpa, &si) catch @panic("oom");
    if (spec.seektable) {
        out.append(gpa, if (spec.metadata) 3 else 0x83) catch @panic("oom");
        out.appendSlice(gpa, &.{ 0, 0, 36 }) catch @panic("oom");
        for (0..36) |_| out.append(gpa, rnd.int(u8)) catch @panic("oom");
    }
    if (spec.metadata) {
        // VORBIS_COMMENT, APPLICATION, PADDING (last)
        out.appendSlice(gpa, &.{ 4, 0, 0, 16 }) catch @panic("oom");
        out.appendSlice(gpa, "\x04\x00\x00\x00zigs\x01\x00\x00\x00\x00\x00\x00\x00") catch @panic("oom");
        out.appendSlice(gpa, &.{ 2, 0, 0, 8 }) catch @panic("oom");
        out.appendSlice(gpa, "ZIG!data") catch @panic("oom");
        out.appendSlice(gpa, &.{ 0x81, 0, 0, 13 }) catch @panic("oom");
        for (0..13) |_| out.append(gpa, 0) catch @panic("oom");
    }
    // frames
    var sample_no: u64 = 0;
    const rate_table = [_]u32{ 0, 88200, 176400, 192000, 8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000 };
    for (blocks.items, 0..) |block, fi| {
        var hdr: std.ArrayList(u8) = .empty;
        defer hdr.deinit(gpa);
        hdr.appendSlice(gpa, &.{ 0xFF, if (spec.variable) 0xF9 else 0xF8 }) catch @panic("oom");
        var bcode: u8 = 0;
        var bextra: u32 = 0; // 0 none, 8 or 16 bits
        if (block == 192) bcode = 1 else if (block == 576 or block == 1152 or block == 2304 or block == 4608) bcode = @intCast(2 + std.math.log2_int(u32, block / 576)) else if (block >= 256 and block <= 32768 and std.math.isPowerOfTwo(block)) bcode = @intCast(8 + std.math.log2_int(u32, block / 256)) else if (block <= 256) {
            bcode = 6;
            bextra = 8;
        } else {
            bcode = 7;
            bextra = 16;
        }
        var rcode: u8 = 0;
        var rextra: u32 = 0;
        var rval: u32 = 0;
        if (spec.rate_code != 255) {
            rcode = spec.rate_code;
        } else if (std.mem.indexOfScalar(u32, rate_table[1..], spec.rate)) |ri| {
            rcode = @intCast(ri + 1);
        } else if (spec.rate % 1000 == 0 and spec.rate / 1000 < 256) {
            rcode = 12;
            rextra = 8;
            rval = spec.rate / 1000;
        } else if (spec.rate < 65536) {
            rcode = 13;
            rextra = 16;
            rval = spec.rate;
        } else if (spec.rate % 10 == 0) {
            rcode = 14;
            rextra = 16;
            rval = spec.rate / 10;
        }
        hdr.append(gpa, (bcode << 4) | rcode) catch @panic("oom");
        const rotation = [_]u8{ 1, 8, 9, 10 };
        const assignment: u8 = if (spec.channels == 2) (if (spec.assignment == 255) rotation[fi % 4] else if (spec.assignment == 0) 1 else spec.assignment) else @intCast(spec.channels - 1);
        const scode: u8 = switch (spec.bps) {
            8 => 1,
            12 => 2,
            16 => 4,
            20 => 5,
            24 => 6,
            else => 0,
        };
        hdr.append(gpa, (assignment << 4) | (scode << 1)) catch @panic("oom");
        utf8Number(&hdr, gpa, if (spec.variable) sample_no else fi);
        if (bextra == 8) hdr.append(gpa, @intCast(block - 1)) catch @panic("oom");
        if (bextra == 16) hdr.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToBig(u16, @intCast(block - 1)))) catch @panic("oom");
        if (rextra == 8) hdr.append(gpa, @intCast(rval)) catch @panic("oom");
        if (rextra == 16) hdr.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToBig(u16, @intCast(rval)))) catch @panic("oom");
        hdr.append(gpa, crc8Bits(hdr.items)) catch @panic("oom");

        var w: BitWriter = .{ .gpa = gpa };
        defer w.bytes.deinit(gpa);
        w.bytes.appendSlice(gpa, hdr.items) catch @panic("oom");
        const chan = gpa.alloc(i32, block) catch @panic("oom");
        defer gpa.free(chan);
        for (0..spec.channels) |c| {
            var sbps = spec.bps;
            if ((assignment == 8 or assignment == 10) and c == 1) sbps += 1;
            if (assignment == 9 and c == 0) sbps += 1;
            const wasted: u32 = if (spec.wasted and rnd.boolean()) 1 + rnd.uintLessThan(u32, @min(sbps - 1, 6)) else 0;
            const kind = spec.kinds[(fi + c) % spec.kinds.len];
            const amp: f64 = std.math.ldexp(@as(f64, 0.9), @as(i32, @intCast(sbps - 1 - wasted)));
            const noise: i64 = @max(1, @as(i64, @intFromFloat(amp / 64)));
            for (chan, 0..) |*x, i| {
                const t: f64 = @floatFromInt(sample_no + i);
                var v: i64 = @intFromFloat(@sin(t * (0.01 + 0.003 * @as(f64, @floatFromInt(c)))) * amp);
                v += rnd.intRangeAtMost(i64, -noise, noise);
                const lim = (@as(i64, 1) << @intCast(sbps - 1 - wasted)) - 1;
                v = std.math.clamp(v, -lim - 1, lim);
                x.* = @intCast(v << @intCast(wasted));
            }
            writeSubframe(&w, gpa, rnd, spec, kind, chan, sbps, wasted);
        }
        w.align8();
        const crc = crc16Bits(w.bytes.items);
        w.put(crc, 16);
        out.appendSlice(gpa, w.bytes.items) catch @panic("oom");
        sample_no += block;
    }
    return out;
}

/// Wraps a native FLAC stream as Ogg-FLAC: the 51-byte mapping packet,
/// then the metadata and frames split across pages of random sizes, with
/// an optional foreign stream's page interleaved and a page whose CRC is
/// wrong.
fn oggWrap(gpa: std.mem.Allocator, rnd: std.Random, flac: []const u8, foreign: bool, bad_crc: bool) std.ArrayList(u8) {
    var out: std.ArrayList(u8) = .empty;
    const serial: u32 = 0x1234abcd;
    var seq: u32 = 0;
    const Page = struct {
        fn emit(o: *std.ArrayList(u8), g: std.mem.Allocator, header_type: u8, ser: u32, sq: u32, body: []const u8, corrupt: bool) void {
            const start = o.items.len;
            o.appendSlice(g, "OggS") catch @panic("oom");
            o.append(g, 0) catch @panic("oom");
            o.append(g, header_type) catch @panic("oom");
            o.appendSlice(g, &std.mem.toBytes(std.mem.nativeToLittle(u64, 0))) catch @panic("oom");
            o.appendSlice(g, &std.mem.toBytes(std.mem.nativeToLittle(u32, ser))) catch @panic("oom");
            o.appendSlice(g, &std.mem.toBytes(std.mem.nativeToLittle(u32, sq))) catch @panic("oom");
            o.appendSlice(g, &.{ 0, 0, 0, 0 }) catch @panic("oom");
            const nseg = body.len / 255 + 1;
            o.append(g, @intCast(nseg)) catch @panic("oom");
            for (0..nseg) |k| o.append(g, if (k + 1 < nseg) 255 else @intCast(body.len % 255)) catch @panic("oom");
            o.appendSlice(g, body) catch @panic("oom");
            var crc = oggCrc(o.items[start..]);
            if (corrupt) crc ^= 1;
            std.mem.writeInt(u32, o.items[start + 22 ..][0..4], crc, .little);
        }
    };
    if (foreign) {
        Page.emit(&out, gpa, 0x02, 0x99, 0, "\x01vorbis-not-flac", false);
    }
    var first: [51]u8 = undefined;
    first[0] = 0x7F;
    @memcpy(first[1..5], "FLAC");
    first[5] = 1;
    first[6] = 0;
    first[7] = 0;
    first[8] = 1;
    @memcpy(first[9..51], flac[0..42]);
    Page.emit(&out, gpa, 0x02, serial, seq, &first, false);
    seq += 1;
    var i: usize = 42;
    var page_no: usize = 0;
    while (i < flac.len) {
        const n = @min(flac.len - i, 1 + rnd.uintLessThan(usize, 9000));
        if (foreign and page_no == 1) Page.emit(&out, gpa, 0, 0x99, 1, "foreign page body", false);
        if (bad_crc and page_no == 2) Page.emit(&out, gpa, 0, serial, seq, flac[i..][0..@min(n, 50)], true);
        Page.emit(&out, gpa, if (page_no == 0) 0 else 0x01, serial, seq, flac[i..][0..n], false);
        seq += 1;
        i += n;
        page_no += 1;
    }
    return out;
}

/// An ID3v2 tag in front, as the helper's sniff needs for a stream that
/// does not start `fLaC`.
fn id3Prefix(gpa: std.mem.Allocator, body: []const u8, tag_len: u32, footer: bool) std.ArrayList(u8) {
    var out: std.ArrayList(u8) = .empty;
    out.appendSlice(gpa, "ID3\x04\x00") catch @panic("oom");
    out.append(gpa, if (footer) 0x10 else 0) catch @panic("oom");
    out.appendSlice(gpa, &.{ @intCast((tag_len >> 21) & 0x7F), @intCast((tag_len >> 14) & 0x7F), @intCast((tag_len >> 7) & 0x7F), @intCast(tag_len & 0x7F) }) catch @panic("oom");
    for (0..tag_len + @as(u32, if (footer) 10 else 0)) |k| out.append(gpa, @truncate(k)) catch @panic("oom");
    out.appendSlice(gpa, body) catch @panic("oom");
    return out;
}

fn synthFlac(gpa: std.mem.Allocator, list: *std.ArrayList(File), rnd: std.Random) !void {
    var name_buf: [160]u8 = undefined;
    const specs = [_]FlacSpec{
        .{ .channels = 1, .bps = 16, .rate = 16000, .block = 4096, .frames = 4 },
        .{ .channels = 2, .bps = 16, .rate = 44100, .block = 4608, .frames = 3, .assignment = 255 },
        .{ .channels = 2, .bps = 24, .rate = 48000, .block = 0, .frames = 12, .assignment = 255, .max_order = 32 },
        .{ .channels = 2, .bps = 8, .rate = 8000, .block = 192, .frames = 20, .assignment = 10, .wasted = true },
        .{ .channels = 2, .bps = 12, .rate = 22050, .block = 0, .frames = 10, .assignment = 8, .rice2 = true },
        .{ .channels = 2, .bps = 20, .rate = 96000, .block = 1152, .frames = 6, .assignment = 9, .escape = true },
        .{ .channels = 2, .bps = 28, .rate = 32000, .block = 0, .frames = 8, .assignment = 255, .max_order = 32 },
        .{ .channels = 2, .bps = 31, .rate = 44100, .block = 1000, .frames = 5, .assignment = 10 },
        .{ .channels = 2, .bps = 32, .rate = 48000, .block = 513, .frames = 4, .assignment = 0, .kinds = "vfl" },
        .{ .channels = 1, .bps = 32, .rate = 12345, .block = 0, .frames = 6, .max_order = 32 },
        .{ .channels = 1, .bps = 4, .rate = 11025, .block = 0, .frames = 9, .wasted = true },
        .{ .channels = 3, .bps = 16, .rate = 24000, .block = 2048, .frames = 3 },
        .{ .channels = 6, .bps = 24, .rate = 48000, .block = 1024, .frames = 3, .escape = true },
        .{ .channels = 8, .bps = 16, .rate = 44100, .block = 576, .frames = 4, .wasted = true },
        .{ .channels = 1, .bps = 16, .rate = 7999, .block = 0, .frames = 8, .variable = true },
        .{ .channels = 2, .bps = 16, .rate = 640000, .block = 4096, .frames = 2, .assignment = 255 },
        .{ .channels = 1, .bps = 16, .rate = 44100, .block = 300, .frames = 6, .rate_code = 0, .metadata = false },
        .{ .channels = 1, .bps = 24, .rate = 22050, .block = 4096, .frames = 3, .seektable = true },
        .{ .channels = 2, .bps = 16, .rate = 16000, .block = 1024, .frames = 3, .total_zero = true },
        .{ .channels = 1, .bps = 16, .rate = 16000, .block = 64, .frames = 30, .kinds = "l", .max_order = 32, .rice2 = true },
        .{ .channels = 1, .bps = 20, .rate = 16000, .block = 256, .frames = 12, .kinds = "l", .max_order = 3, .rice2 = true, .big_coefs = 12 },
        .{ .channels = 2, .bps = 19, .rate = 16000, .block = 256, .frames = 12, .kinds = "l", .max_order = 3, .rice2 = true, .big_coefs = 13, .assignment = 255 },
    };
    for (specs, 0..) |sp, idx| {
        var native = flacStream(gpa, rnd, sp);
        defer native.deinit(gpa);
        var b: Builder = .{ .gpa = gpa };
        b.bytes(native.items);
        add(gpa, list, try std.fmt.bufPrint(&name_buf, "synth/flac-{d}-{d}ch-{d}bit-{d}", .{ idx, sp.channels, sp.bps, sp.rate }), &b);
        // behind an ID3 tag, and Ogg-wrapped behind one (the helper's sniff
        // takes neither a bare OggS nor anything else)
        var tagged = id3Prefix(gpa, native.items, 37 + @as(u32, @intCast(idx)), idx % 3 == 0);
        defer tagged.deinit(gpa);
        b.bytes(tagged.items);
        add(gpa, list, try std.fmt.bufPrint(&name_buf, "synth/flac-{d}-id3", .{idx}), &b);
        var ogg = oggWrap(gpa, rnd, native.items, idx % 4 == 1, idx % 5 == 2);
        defer ogg.deinit(gpa);
        var ogg_tagged = id3Prefix(gpa, ogg.items, 10, false);
        defer ogg_tagged.deinit(gpa);
        b.bytes(ogg_tagged.items);
        add(gpa, list, try std.fmt.bufPrint(&name_buf, "synth/flac-{d}-ogg", .{idx}), &b);
    }
}

fn synthImages(gpa: std.mem.Allocator, list: *std.ArrayList(File), rnd: std.Random) !void {
    var name_buf: [128]u8 = undefined;
    const sizes = [_][2]u32{ .{ 1, 1 }, .{ 3, 5 }, .{ 17, 9 }, .{ 8, 8 }, .{ 33, 2 } };

    // PNG over stored deflate blocks: every colour type and depth, both interlace modes, tRNS, every filter
    const PngSpec = struct { color: u8, depths: []const u8 };
    const png_specs = [_]PngSpec{
        .{ .color = 0, .depths = &.{ 1, 2, 4, 8, 16 } },
        .{ .color = 2, .depths = &.{ 8, 16 } },
        .{ .color = 3, .depths = &.{ 1, 2, 4, 8 } },
        .{ .color = 4, .depths = &.{ 8, 16 } },
        .{ .color = 6, .depths = &.{ 8, 16 } },
    };
    for (png_specs) |sp| for (sp.depths) |depth| for ([_]u8{ 0, 1 }) |interlace| for ([_]bool{ false, true }) |trns| for (sizes) |sz| {
        if (trns and (sp.color == 4 or sp.color == 6)) continue;
        var b: Builder = .{ .gpa = gpa };
        try pngFile(&b, rnd, sz[0], sz[1], sp.color, depth, interlace, trns);
        add(gpa, list, try std.fmt.bufPrint(&name_buf, "synth/png-c{d}-d{d}-i{d}-t{d}-{d}x{d}", .{ sp.color, depth, interlace, @intFromBool(trns), sz[0], sz[1] }), &b);
    };

    // BMP: palettes, 16/24/32 bits, bitfields, V4/V5, OS/2, top-down
    const BmpSpec = struct { bpp: u16, hsz: u32, compress: u32 = 0, topdown: bool = false, masks: [4]u32 = .{ 0, 0, 0, 0 } };
    const bmp_specs = [_]BmpSpec{
        .{ .bpp = 1, .hsz = 40 },
        .{ .bpp = 4, .hsz = 40 },
        .{ .bpp = 8, .hsz = 40, .topdown = true },
        .{ .bpp = 8, .hsz = 12 },
        .{ .bpp = 24, .hsz = 12 },
        .{ .bpp = 16, .hsz = 40 },
        .{ .bpp = 16, .hsz = 40, .compress = 3, .masks = .{ 0xF800, 0x07E0, 0x001F, 0 } },
        .{ .bpp = 24, .hsz = 40 },
        .{ .bpp = 24, .hsz = 56, .topdown = true },
        .{ .bpp = 32, .hsz = 40 },
        .{ .bpp = 32, .hsz = 40, .compress = 3, .masks = .{ 0x00FF0000, 0x0000FF00, 0x000000FF, 0 } },
        .{ .bpp = 32, .hsz = 108, .compress = 3, .masks = .{ 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000 } },
        .{ .bpp = 32, .hsz = 124, .compress = 3, .masks = .{ 0x000003FF, 0x000FFC00, 0x3FF00000, 0xC0000000 } },
        .{ .bpp = 16, .hsz = 124, .compress = 3, .masks = .{ 0x7C00, 0x03E0, 0x001F, 0x8000 } },
    };
    for (bmp_specs, 0..) |sp, idx| for (sizes) |sz| {
        var b: Builder = .{ .gpa = gpa };
        bmpFile(&b, rnd, sz[0], sz[1], sp.bpp, sp.hsz, sp.compress, sp.topdown, sp.masks);
        add(gpa, list, try std.fmt.bufPrint(&name_buf, "synth/bmp-{d}-{d}bpp-h{d}-{d}x{d}", .{ idx, sp.bpp, sp.hsz, sz[0], sz[1] }), &b);
    };

    // TGA: colour-mapped, true colour and grey, raw and RLE, either origin
    const TgaSpec = struct { kind: u8, bpp: u8, pal_bits: u8 = 0, desc: u8 = 0 };
    const tga_specs = [_]TgaSpec{
        .{ .kind = 1, .bpp = 8, .pal_bits = 24 },
        .{ .kind = 1, .bpp = 8, .pal_bits = 16 },
        .{ .kind = 9, .bpp = 8, .pal_bits = 32, .desc = 0x20 },
        .{ .kind = 2, .bpp = 24 },
        .{ .kind = 2, .bpp = 32, .desc = 0x28 },
        .{ .kind = 2, .bpp = 16 },
        .{ .kind = 2, .bpp = 15, .desc = 0x20 },
        .{ .kind = 3, .bpp = 8 },
        .{ .kind = 3, .bpp = 16 },
        .{ .kind = 10, .bpp = 24 },
        .{ .kind = 10, .bpp = 16 },
        .{ .kind = 11, .bpp = 8, .desc = 0x20 },
    };
    for (tga_specs, 0..) |sp, idx| for (sizes) |sz| {
        var b: Builder = .{ .gpa = gpa };
        tgaFile(&b, rnd, sz[0], sz[1], sp.kind, sp.bpp, sp.pal_bits, sp.desc);
        add(gpa, list, try std.fmt.bufPrint(&name_buf, "synth/tga-{d}-k{d}-{d}bpp-{d}x{d}", .{ idx, sp.kind, sp.bpp, sz[0], sz[1] }), &b);
    };

    // PNM
    for ([_]u8{ '5', '6' }) |t| for ([_]u32{ 255, 100, 65535, 1000 }) |maxv| for (sizes) |sz| {
        var b: Builder = .{ .gpa = gpa };
        var hdr: [64]u8 = undefined;
        b.bytes(try std.fmt.bufPrint(&hdr, "P{c}\n# a comment\n{d} {d}\n{d}\n", .{ t, sz[0], sz[1], maxv }));
        const n = sz[0] * sz[1] * @as(u32, if (t == '6') 3 else 1) * @as(u32, if (maxv > 255) 2 else 1);
        for (0..n) |_| b.u8_(rnd.int(u8));
        add(gpa, list, try std.fmt.bufPrint(&name_buf, "synth/pnm-P{c}-{d}-{d}x{d}", .{ t, maxv, sz[0], sz[1] }), &b);
    };

    // HDR: flat below 8 wide, run-length encoded from 8
    for ([_][2]u32{ .{ 3, 4 }, .{ 7, 2 }, .{ 8, 3 }, .{ 20, 5 }, .{ 64, 2 } }) |sz| for ([_]bool{ false, true }) |rgbe_hdr| {
        var b: Builder = .{ .gpa = gpa };
        var hdr: [96]u8 = undefined;
        b.bytes(try std.fmt.bufPrint(&hdr, "{s}\nFORMAT=32-bit_rle_rgbe\nEXPOSURE=1.0\n\n-Y {d} +X {d}\n", .{ if (rgbe_hdr) "#?RGBE" else "#?RADIANCE", sz[1], sz[0] }));
        if (sz[0] < 8) {
            for (0..sz[0] * sz[1]) |_| {
                b.u8_(rnd.int(u8));
                b.u8_(rnd.int(u8));
                b.u8_(rnd.int(u8));
                b.u8_(if (rnd.boolean()) 0 else 110 + rnd.uintLessThan(u8, 40));
            }
        } else {
            for (0..sz[1]) |_| {
                b.bytes("\x02\x02");
                b.be16(@intCast(sz[0]));
                for (0..4) |k| {
                    var left = sz[0];
                    while (left > 0) {
                        if (rnd.boolean() and left >= 3) {
                            const run = @min(left, 2 + rnd.uintLessThan(u32, 20));
                            b.u8_(@intCast(128 + run));
                            b.u8_(if (k == 3) 120 + rnd.uintLessThan(u8, 20) else rnd.int(u8));
                            left -= run;
                        } else {
                            const dump = @min(left, 1 + rnd.uintLessThan(u32, 10));
                            b.u8_(@intCast(dump));
                            for (0..dump) |_| b.u8_(if (k == 3) 120 + rnd.uintLessThan(u8, 20) else rnd.int(u8));
                            left -= dump;
                        }
                    }
                }
            }
        }
        add(gpa, list, try std.fmt.bufPrint(&name_buf, "synth/hdr-{d}x{d}-{d}", .{ sz[0], sz[1], @intFromBool(rgbe_hdr) }), &b);
    };

    // PSD: raw and RLE, 8 and 16 bits, 3 and 4 channels
    for ([_]u16{ 3, 4, 1 }) |channels| for ([_]u16{ 8, 16 }) |depth| for ([_]u16{ 0, 1 }) |comp| for (sizes) |sz| {
        if (comp == 1 and depth == 16) continue;
        var b: Builder = .{ .gpa = gpa };
        b.bytes("8BPS");
        b.be16(1);
        b.bytes("\x00\x00\x00\x00\x00\x00");
        b.be16(channels);
        b.be32(sz[1]);
        b.be32(sz[0]);
        b.be16(depth);
        b.be16(3);
        b.be32(0);
        b.be32(4);
        b.bytes("abcd");
        b.be32(0);
        b.be16(comp);
        const px = sz[0] * sz[1];
        if (comp == 0) {
            for (0..px * channels * (depth / 8)) |_| b.u8_(rnd.int(u8));
        } else {
            for (0..sz[1] * channels) |_| b.be16(0);
            for (0..channels) |_| {
                var left = px;
                while (left > 0) {
                    if (rnd.boolean() and left >= 2) {
                        const run = @min(left, 2 + rnd.uintLessThan(u32, 30));
                        b.u8_(@intCast(257 - run));
                        b.u8_(rnd.int(u8));
                        left -= run;
                    } else {
                        const lit = @min(left, 1 + rnd.uintLessThan(u32, 20));
                        b.u8_(@intCast(lit - 1));
                        for (0..lit) |_| b.u8_(rnd.int(u8));
                        left -= lit;
                    }
                    if (rnd.uintLessThan(u8, 16) == 0) b.u8_(128);
                }
            }
        }
        add(gpa, list, try std.fmt.bufPrint(&name_buf, "synth/psd-c{d}-d{d}-r{d}-{d}x{d}", .{ channels, depth, comp, sz[0], sz[1] }), &b);
    };

    // PIC: uncompressed, pure RLE and mixed RLE packets, with and without alpha
    for ([_]u8{ 0, 1, 2 }) |kind| for ([_]bool{ false, true }) |alpha| for (sizes) |sz| {
        var b: Builder = .{ .gpa = gpa };
        b.bytes("\x53\x80\xF6\x34");
        for (0..84) |_| b.u8_(0);
        b.bytes("PICT");
        b.be16(@intCast(sz[0]));
        b.be16(@intCast(sz[1]));
        b.be32(0x3F800000);
        b.be16(3);
        b.be16(0);
        // packets: chained, size 8, type, channel mask
        b.bytes(&.{ @intFromBool(alpha), 8, kind, 0xE0 });
        if (alpha) b.bytes(&.{ 0, 8, kind, 0x10 });
        const masks: []const u8 = if (alpha) &.{ 3, 1 } else &.{3};
        for (0..sz[1]) |_| for (masks) |nch| {
            var left = sz[0];
            switch (kind) {
                0 => for (0..sz[0] * nch) |_| b.u8_(rnd.int(u8)),
                1 => while (left > 0) {
                    const c = @min(left, 1 + rnd.uintLessThan(u32, 5));
                    b.u8_(@intCast(c));
                    for (0..nch) |_| b.u8_(rnd.int(u8));
                    left -= c;
                },
                else => while (left > 0) {
                    if (rnd.boolean() and left >= 2) {
                        const c = @min(left, 2 + rnd.uintLessThan(u32, 6));
                        if (rnd.uintLessThan(u8, 4) == 0) {
                            b.u8_(128);
                            b.be16(@intCast(c));
                        } else b.u8_(@intCast(c + 127));
                        for (0..nch) |_| b.u8_(rnd.int(u8));
                        left -= c;
                    } else {
                        const c = @min(left, 1 + rnd.uintLessThan(u32, 4));
                        b.u8_(@intCast(c - 1));
                        for (0..c * nch) |_| b.u8_(rnd.int(u8));
                        left -= c;
                    }
                },
            }
        };
        add(gpa, list, try std.fmt.bufPrint(&name_buf, "synth/pic-k{d}-a{d}-{d}x{d}", .{ kind, @intFromBool(alpha), sz[0], sz[1] }), &b);
    };

    // GIF: global and local tables, interlace, transparency, background
    for ([_]bool{ false, true }) |interlace| for ([_]bool{ false, true }) |local| for ([_]bool{ false, true }) |transparent| for (sizes) |sz| {
        var b: Builder = .{ .gpa = gpa };
        b.bytes("GIF89a");
        b.le16(@intCast(sz[0] + 2));
        b.le16(@intCast(sz[1] + 1));
        b.u8_(if (local) 0x00 else 0xF7);
        b.u8_(3); // background index
        b.u8_(0);
        if (!local) for (0..256 * 3) |_| b.u8_(rnd.int(u8));
        if (transparent) b.bytes(&.{ 0x21, 0xF9, 4, 0x05, 10, 0, 7, 0 });
        b.bytes(&.{ 0x21, 0xFE, 3, 'h', 'i', '!', 0 });
        b.u8_(0x2C);
        b.le16(1);
        b.le16(1);
        b.le16(@intCast(sz[0]));
        b.le16(@intCast(sz[1]));
        b.u8_(@as(u8, if (interlace) 0x40 else 0) | @as(u8, if (local) 0x87 else 0));
        if (local) for (0..256 * 3) |_| b.u8_(rnd.int(u8));
        gifRaster(&b, rnd, sz[0] * sz[1]);
        b.u8_(0x3B);
        add(gpa, list, try std.fmt.bufPrint(&name_buf, "synth/gif-i{d}-l{d}-t{d}-{d}x{d}", .{ @intFromBool(interlace), @intFromBool(local), @intFromBool(transparent), sz[0], sz[1] }), &b);
    };
}

/// An LZW stream of literal codes at 8 bits, cleared often enough that the
/// code size never grows past 9.
fn gifRaster(b: *Builder, rnd: std.Random, npix: u32) void {
    b.u8_(8);
    var bits: u32 = 0;
    var nbits: u5 = 0;
    var block: std.ArrayList(u8) = .empty;
    defer block.deinit(b.gpa);
    const emit = struct {
        fn f(bb: *Builder, blk: *std.ArrayList(u8), acc: *u32, n: *u5, code: u32) void {
            acc.* |= code << n.*;
            n.* += 9;
            while (n.* >= 8) {
                blk.append(bb.gpa, @truncate(acc.*)) catch @panic("oom");
                acc.* >>= 8;
                n.* -= 8;
            }
        }
    }.f;
    var since_clear: u32 = 0;
    emit(b, &block, &bits, &nbits, 256);
    for (0..npix) |_| {
        if (since_clear == 200) {
            emit(b, &block, &bits, &nbits, 256);
            since_clear = 0;
        }
        emit(b, &block, &bits, &nbits, rnd.int(u8));
        since_clear += 1;
    }
    emit(b, &block, &bits, &nbits, 257);
    if (nbits > 0) block.append(b.gpa, @truncate(bits)) catch @panic("oom");
    var i: usize = 0;
    while (i < block.items.len) {
        const n = @min(255, block.items.len - i);
        b.u8_(@intCast(n));
        b.bytes(block.items[i..][0..n]);
        i += n;
    }
    b.u8_(0);
}

fn crc32(data: []const u8) u32 {
    return std.hash.Crc32.hash(data);
}

fn pngChunk(b: *Builder, kind: *const [4]u8, data: []const u8) void {
    b.be32(@intCast(data.len));
    const start = b.b.items.len;
    b.bytes(kind);
    b.bytes(data);
    b.be32(crc32(b.b.items[start..]));
}

fn pngFile(b: *Builder, rnd: std.Random, w: u32, h: u32, color: u8, depth: u8, interlace: u8, trns: bool) !void {
    const gpa = b.gpa;
    b.bytes("\x89PNG\r\n\x1a\n");
    var ihdr: [13]u8 = undefined;
    std.mem.writeInt(u32, ihdr[0..4], w, .big);
    std.mem.writeInt(u32, ihdr[4..8], h, .big);
    ihdr[8] = depth;
    ihdr[9] = color;
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = interlace;
    pngChunk(b, "IHDR", &ihdr);
    pngChunk(b, "tEXt", "Comment\x00synth");
    const channels: u32 = switch (color) {
        0, 3 => 1,
        2 => 3,
        4 => 2,
        else => 4,
    };
    if (color == 3) {
        var pal: [256 * 3]u8 = undefined;
        rnd.bytes(&pal);
        const n: usize = @as(usize, 1) << @intCast(depth);
        pngChunk(b, "PLTE", pal[0 .. n * 3]);
        if (trns) {
            var t: [16]u8 = undefined;
            rnd.bytes(&t);
            pngChunk(b, "tRNS", t[0..@min(n, 16)]);
        }
    } else if (trns) {
        var t: [6]u8 = undefined;
        rnd.bytes(&t);
        if (depth < 16) for (0..3) |k| {
            t[k * 2] = 0;
            t[k * 2 + 1] &= @intCast((@as(u16, 1) << @intCast(depth)) - 1);
        };
        pngChunk(b, "tRNS", t[0 .. channels * 2]);
    }
    // the raw scanlines, per pass
    var raw: std.ArrayList(u8) = .empty;
    defer raw.deinit(gpa);
    const passes: []const [4]u32 = if (interlace == 1) &.{ .{ 0, 0, 8, 8 }, .{ 4, 0, 8, 8 }, .{ 0, 4, 4, 8 }, .{ 2, 0, 4, 4 }, .{ 0, 2, 2, 4 }, .{ 1, 0, 2, 2 }, .{ 0, 1, 1, 2 } } else &.{.{ 0, 0, 1, 1 }};
    for (passes) |p| {
        if (w <= p[0] or h <= p[1]) continue;
        const pw = (w - p[0] + p[2] - 1) / p[2];
        const ph = (h - p[1] + p[3] - 1) / p[3];
        const row_bytes = (pw * channels * depth + 7) / 8;
        for (0..ph) |_| {
            try raw.append(gpa, rnd.uintLessThan(u8, 5));
            for (0..row_bytes) |_| try raw.append(gpa, rnd.int(u8));
        }
    }
    // zlib with stored blocks
    var z: std.ArrayList(u8) = .empty;
    defer z.deinit(gpa);
    try z.appendSlice(gpa, "\x78\x01");
    var i: usize = 0;
    while (true) {
        const n = @min(raw.items.len - i, 1000);
        const final = i + n == raw.items.len;
        try z.append(gpa, @intFromBool(final));
        try z.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u16, @intCast(n))));
        try z.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u16, ~@as(u16, @intCast(n)))));
        try z.appendSlice(gpa, raw.items[i..][0..n]);
        i += n;
        if (final) break;
    }
    try z.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToBig(u32, std.hash.Adler32.hash(raw.items))));
    // two IDAT chunks, so their concatenation is exercised
    const half = z.items.len / 2;
    pngChunk(b, "IDAT", z.items[0..half]);
    pngChunk(b, "IDAT", z.items[half..]);
    pngChunk(b, "IEND", "");
}

fn bmpFile(b: *Builder, rnd: std.Random, w: u32, h: u32, bpp: u16, hsz: u32, compress: u32, topdown: bool, masks: [4]u32) void {
    const pal_n: u32 = if (bpp <= 8) @as(u32, 1) << @intCast(bpp) else 0;
    const pal_entry: u32 = if (hsz == 12) 3 else 4;
    const extra_masks: u32 = if (hsz == 40 and compress == 3) 12 else 0;
    const offset = 14 + hsz + extra_masks + pal_n * pal_entry;
    const row = ((w * bpp + 31) / 32) * 4;
    b.bytes("BM");
    b.le32(offset + row * h);
    b.le32(0);
    b.le32(offset);
    b.le32(hsz);
    if (hsz == 12) {
        b.le16(@intCast(w));
        b.le16(@intCast(h));
        b.le16(1);
        b.le16(bpp);
    } else {
        b.le32(w);
        b.le32(if (topdown) @bitCast(-@as(i32, @intCast(h))) else h);
        b.le16(1);
        b.le16(bpp);
        b.le32(compress);
        b.le32(row * h);
        b.le32(2835);
        b.le32(2835);
        b.le32(0);
        b.le32(0);
        if (hsz == 56) for (0..4) |_| b.le32(0);
        if (extra_masks != 0) {
            b.le32(masks[0]);
            b.le32(masks[1]);
            b.le32(masks[2]);
        }
        if (hsz >= 108) {
            for (masks) |m| b.le32(m);
            b.le32(0x73524742);
            for (0..12) |_| b.le32(0);
            if (hsz == 124) for (0..4) |_| b.le32(0);
        }
    }
    for (0..pal_n * pal_entry) |_| b.u8_(rnd.int(u8));
    for (0..row * h) |_| b.u8_(rnd.int(u8));
}

fn tgaFile(b: *Builder, rnd: std.Random, w: u32, h: u32, kind: u8, bpp: u8, pal_bits: u8, desc: u8) void {
    const indexed = kind == 1 or kind == 9;
    const pal_len: u16 = if (indexed) 200 else 0;
    b.u8_(3); // id length
    b.u8_(@intFromBool(indexed));
    b.u8_(kind);
    b.le16(0);
    b.le16(pal_len);
    b.u8_(pal_bits);
    b.le16(0);
    b.le16(0);
    b.le16(@intCast(w));
    b.le16(@intCast(h));
    b.u8_(bpp);
    b.u8_(desc);
    b.bytes("id!");
    for (0..@as(usize, pal_len) * ((pal_bits + 7) / 8)) |_| b.u8_(rnd.int(u8));
    const px_bytes: u32 = (bpp + 7) / 8;
    const n = w * h;
    if (kind >= 9) {
        var left = n;
        while (left > 0) {
            const c = @min(left, 1 + rnd.uintLessThan(u32, 128));
            if (rnd.boolean()) {
                b.u8_(@intCast(0x80 | (c - 1)));
                for (0..px_bytes) |_| b.u8_(if (indexed) rnd.uintLessThan(u8, 210) else rnd.int(u8));
            } else {
                b.u8_(@intCast(c - 1));
                for (0..c * px_bytes) |_| b.u8_(if (indexed) rnd.uintLessThan(u8, 210) else rnd.int(u8));
            }
            left -= c;
        }
    } else {
        for (0..n * px_bytes) |_| b.u8_(if (indexed) rnd.uintLessThan(u8, 210) else rnd.int(u8));
    }
}
