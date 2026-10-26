//! `make mtmd-image-diff`: every image preprocessor of `src/mtmd/image.zig`
//! against upstream's `mtmd-image.cpp`, compiled unmodified beside it
//! (`harness/mtmd_image_ref.cpp`), in one process, on bits.
//!
//! **Not a port.** A harness. Each case is a preprocessor kind, hparams and
//! an image; both sides' outputs -- the grid, the overview and every entry,
//! each with its size, flags, anyres info and every float's bits -- are
//! serialized the same way and must be byte-equal, and so must every line
//! the two log.
//!
//! Usage: mtmd-image-diff [kind-name-substring] [-v]

const std = @import("std");
const mtmd = @import("mtmd");
const image = mtmd.image;
const impl = mtmd.impl;
const model = mtmd.model;
const names = mtmd.names;

const RefHparams = extern struct {
    image_size: i32 = 0,
    patch_size: i32 = 0,
    n_merge: i32 = 1,
    image_longest_edge: i32 = 0,
    image_min_pixels: i32 = -1,
    image_max_pixels: i32 = -1,
    image_resize_algo: i32 = 1,
    image_resize_pad: i32 = 1,
    image_pad_color: [4]u8 = .{ 0, 0, 0, 0 },
    n_candidates: i32 = 0,
    candidates: [64]i32 = @splat(0),
    preproc_min_tiles: i32 = 0,
    preproc_max_tiles: i32 = 0,
    preproc_tile_size: i32 = 0,
    image_resize_algo_rf: i32 = 1,
    image_resize_algo_ov: i32 = 1,
    image_pad_rf: i32 = 1,
    image_pad_ov: i32 = 0,
    image_pad_color_rf: [4]u8 = .{ 0, 0, 0, 0 },
    image_pad_color_ov: [4]u8 = .{ 0, 0, 0, 0 },
    image_mean: [3]f32 = .{ 0.5, 0.5, 0.5 },
    image_std: [3]f32 = .{ 0.5, 0.5, 0.5 },
};

extern fn ref_preprocess(kind: c_int, proj_type: c_int, hp: *const RefHparams, rgb: ?[*]const u8, nx: c_int, ny: c_int, out: *?[*]u8, out_len: *usize) c_int;
extern fn ref_free(p: ?[*]u8) void;
extern fn ref_set_log(f: *const fn (c_int, [*c]const u8, ?*anyopaque) callconv(.c) void, ud: ?*anyopaque) void;

const gpa = std.heap.c_allocator;

var ref_log: std.ArrayList(u8) = .empty;
var port_log: std.ArrayList(u8) = .empty;

fn capture(level: c_int, text: [*c]const u8, ud: ?*anyopaque) callconv(.c) void {
    const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(ud.?));
    buf.print(gpa, "[{d}] {s}", .{ level, std.mem.span(text) }) catch {};
}

fn portCapture(level: impl.c.enum_ggml_log_level, text: [*c]const u8, ud: ?*anyopaque) callconv(.c) void {
    capture(@intCast(level), text, ud);
}

/// The port's hparams for a case, built from the same flat struct the
/// reference reads.
fn toHparams(rh: *const RefHparams) !model.Hparams {
    var h: model.Hparams = .{};
    h.image_size = rh.image_size;
    h.patch_size = rh.patch_size;
    h.n_merge = rh.n_merge;
    h.image_longest_edge = rh.image_longest_edge;
    h.image_min_pixels = rh.image_min_pixels;
    h.image_max_pixels = rh.image_max_pixels;
    h.image_resize_algo = @fromBackingInt(@intCast(rh.image_resize_algo));
    h.image_resize_pad = @fromBackingInt(@intCast(rh.image_resize_pad));
    h.image_pad_color = rh.image_pad_color[0..3].*;
    h.image_pad_color_rf = rh.image_pad_color_rf[0..3].*;
    h.image_pad_color_ov = rh.image_pad_color_ov[0..3].*;
    h.image_mean = rh.image_mean;
    h.image_std = rh.image_std;
    var i: usize = 0;
    while (i < @as(usize, @intCast(rh.n_candidates))) : (i += 1) {
        try h.image_res_candidates.append(gpa, .{ .width = rh.candidates[2 * i], .height = rh.candidates[2 * i + 1] });
    }
    h.preproc_min_tiles = rh.preproc_min_tiles;
    h.preproc_max_tiles = rh.preproc_max_tiles;
    h.preproc_tile_size = rh.preproc_tile_size;
    h.image_resize_algo_rf = @fromBackingInt(@intCast(rh.image_resize_algo_rf));
    h.image_resize_algo_ov = @fromBackingInt(@intCast(rh.image_resize_algo_ov));
    h.image_pad_rf = @fromBackingInt(@intCast(rh.image_pad_rf));
    h.image_pad_ov = @fromBackingInt(@intCast(rh.image_pad_ov));
    return h;
}

fn putI32(out: *std.ArrayList(u8), v: i32) !void {
    try out.appendSlice(gpa, std.mem.asBytes(&v));
}

/// `put_image` of the reference shim, over the port's image.
fn putImage(out: *std.ArrayList(u8), img: *const impl.ImageF32) !void {
    try putI32(out, img.nx());
    try putI32(out, img.ny());
    try putI32(out, @intFromBool(img.add_viewsep));
    try putI32(out, @intFromBool(img.add_newline));
    try putI32(out, img.anyres.grid_x);
    try putI32(out, img.anyres.grid_y);
    try putI32(out, img.anyres.orig_nx);
    try putI32(out, img.anyres.orig_ny);
    const n: u64 = img.roBuf().len;
    try out.appendSlice(gpa, std.mem.asBytes(&n));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(img.roBuf()));
}

fn serialize(res: *const image.PreprocOut) !std.ArrayList(u8) {
    var out: std.ArrayList(u8) = .empty;
    try putI32(&out, res.grid_x);
    try putI32(&out, res.grid_y);
    try putI32(&out, @intFromBool(res.hasOverview()));
    try putImage(&out, &res.overview);
    const n: u64 = res.entries.items.len;
    try out.appendSlice(gpa, std.mem.asBytes(&n));
    for (res.entries.items) |*e| try putImage(&out, e);
    return out;
}

const Pattern = enum { random, gradient, checker, white, placeholder };

fn makeImage(nx: c_int, ny: c_int, pattern: Pattern, seed: u64) ![]u8 {
    const n = @as(usize, @intCast(nx)) * @as(usize, @intCast(ny)) * 3;
    const buf = try gpa.alloc(u8, n);
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const px = i / 3;
        const x = px % @as(usize, @intCast(nx));
        const y = px / @as(usize, @intCast(nx));
        buf[i] = switch (pattern) {
            .random, .placeholder => r.int(u8),
            .gradient => @intCast((x * 255 / @max(1, @as(usize, @intCast(nx)) - 1) + (i % 3) * 40 + y) % 256),
            .checker => if (((x / 3) + (y / 3)) % 2 == 0) 0 else 255,
            .white => 255,
        };
    }
    return buf;
}

const Case = struct {
    kind: image.Kind,
    proj: names.ProjectorType = .mlp,
    hp: RefHparams,
    nx: c_int,
    ny: c_int,
    pattern: Pattern,
    label: []const u8,
};

var n_cases: usize = 0;
var n_fail: usize = 0;
var n_floats: usize = 0;
var verbose = false;
var n_log_bytes: usize = 0;
var n_tiled: usize = 0;
var n_overview: usize = 0;
var n_placeholder: usize = 0;
var kinds_tiled: std.EnumSet(image.Kind) = .empty;

fn runCase(cs: *const Case, seed: u64) !void {
    const rgb = try makeImage(cs.nx, cs.ny, cs.pattern, seed);
    defer gpa.free(rgb);
    const ph = cs.pattern == .placeholder;

    ref_log.clearRetainingCapacity();
    port_log.clearRetainingCapacity();

    var out_ptr: ?[*]u8 = null;
    var out_len: usize = 0;
    if (ref_preprocess(@backingInt(cs.kind), @backingInt(cs.proj), &cs.hp, if (ph) null else rgb.ptr, cs.nx, cs.ny, &out_ptr, &out_len) != 0) return error.RefFailed;
    defer ref_free(out_ptr);
    const ref_bytes = out_ptr.?[0..out_len];

    var hp = try toHparams(&cs.hp);
    defer hp.deinit();
    var img: impl.ImageU8 = .{};
    defer img.deinit();
    try img.setSize(.{ .width = cs.nx, .height = cs.ny }, ph);
    if (!ph) try img.cpyBuf(rgb);
    const pp = image.Preprocessor.init(cs.kind, &hp, cs.proj);
    var res = try pp.preprocess(&img);
    defer res.deinit();
    var port_bytes = try serialize(&res);
    defer port_bytes.deinit(gpa);

    n_cases += 1;
    n_log_bytes += ref_log.items.len;
    if (res.grid_x > 0 and res.grid_y > 0) {
        n_tiled += 1;
        kinds_tiled.insert(cs.kind);
    }
    if (res.hasOverview()) n_overview += 1;
    if (ph) n_placeholder += 1;
    for (res.entries.items) |*e| n_floats += e.roBuf().len;
    n_floats += res.overview.roBuf().len;

    const same = std.mem.eql(u8, ref_bytes, port_bytes.items);
    const same_log = std.mem.eql(u8, ref_log.items, port_log.items);
    if (!same or !same_log) {
        n_fail += 1;
        if (n_fail <= 20) {
            std.debug.print("FAIL {s} kind={s} {d}x{d} {s}: out {s} ({d} vs {d} bytes), log {s}\n", .{
                cs.label,                       @tagName(cs.kind), cs.nx,                cs.ny,                              @tagName(cs.pattern),
                if (same) "same" else "DIFFER", ref_bytes.len,     port_bytes.items.len, if (same_log) "same" else "DIFFER",
            });
            if (!same) {
                const m = @min(ref_bytes.len, port_bytes.items.len);
                var k: usize = 0;
                while (k < m and ref_bytes[k] == port_bytes.items[k]) k += 1;
                std.debug.print("    first differing byte {d}\n", .{k});
            }
            if (!same_log) std.debug.print("    ref log:\n{s}    port log:\n{s}", .{ ref_log.items, port_log.items });
        }
    } else if (verbose) {
        std.debug.print("ok   {s} kind={s} {d}x{d} {s} ({d} entries)\n", .{ cs.label, @tagName(cs.kind), cs.nx, cs.ny, @tagName(cs.pattern), res.entries.items.len });
    }
}

const sizes = [_][2]c_int{
    .{ 1, 1 },     .{ 2, 2 },      .{ 3, 7 },      .{ 7, 3 },       .{ 16, 16 },   .{ 31, 17 },
    .{ 64, 48 },   .{ 100, 37 },   .{ 37, 100 },   .{ 224, 224 },   .{ 299, 301 }, .{ 336, 336 },
    .{ 400, 300 }, .{ 449, 449 },  .{ 513, 129 },  .{ 129, 513 },   .{ 640, 480 }, .{ 1000, 40 },
    .{ 40, 1000 }, .{ 1280, 720 }, .{ 720, 1280 }, .{ 1500, 1100 },
    // an edge at an odd multiple of half the alignment, where lfm2's
    // round-half-to-even decides whether the image is tiled
    .{ 512, 272 }, .{ 1024, 528 },
};

const clip_mean = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
const clip_std = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };

fn setCandidates(hp: *RefHparams, cands: []const [2]i32) void {
    hp.n_candidates = @intCast(cands.len);
    for (cands, 0..) |cnd, i| {
        hp.candidates[2 * i] = cnd[0];
        hp.candidates[2 * i + 1] = cnd[1];
    }
}

/// The hparams variants for one kind.
fn variants(kind: image.Kind, out: *std.ArrayList(struct { RefHparams, names.ProjectorType, []const u8 })) !void {
    var base: RefHparams = .{ .image_mean = clip_mean, .image_std = clip_std };
    switch (kind) {
        .fixed_size => {
            for ([_]i32{ 0, 1, 2 }) |algo| for ([_]i32{ 0, 1, 2 }) |pad| for ([_][4]u8{ .{ 0, 0, 0, 0 }, .{ 127, 128, 129, 0 }, .{ 255, 0, 128, 0 } }) |col| {
                var h = base;
                h.image_size = if (algo == 1) 224 else 336;
                h.patch_size = 14;
                h.image_resize_algo = algo;
                h.image_resize_pad = pad;
                h.image_pad_color = col;
                try out.append(gpa, .{ h, .mlp, "fixed" });
            };
        },
        .llava_uhd, .minicpmv, .internvl => {
            for ([_]bool{ false, true }) |pins| for ([_]i32{ 1, 2 }) |merge| for ([_][2]i32{ .{ 1, 0 }, .{ 0, 1 }, .{ 2, 2 } }) |algos| for ([_][2]i32{ .{ 1, 0 }, .{ 2, 1 }, .{ 0, 2 } }) |pads| {
                if (kind == .internvl and !pins) continue;
                var h = base;
                h.image_size = if (kind == .llava_uhd) 336 else 448;
                h.patch_size = 14;
                h.n_merge = merge;
                h.image_resize_algo_rf = algos[0];
                h.image_resize_algo_ov = algos[1];
                h.image_pad_rf = pads[0];
                h.image_pad_ov = pads[1];
                h.image_pad_color_rf = .{ 10, 20, 30, 0 };
                h.image_pad_color_ov = .{ 200, 100, 50, 0 };
                if (pins) {
                    const s = h.image_size;
                    setCandidates(&h, &.{ .{ s, 2 * s }, .{ 2 * s, s }, .{ 2 * s, 2 * s }, .{ 3 * s, s }, .{ s, 3 * s } });
                }
                try out.append(gpa, .{ h, .mlp, if (pins) "pinpoints" else "dynamic" });
            };
        },
        .dyn_size => {
            for ([_]i32{ 14, 16 }) |ps| for ([_]i32{ 1, 2 }) |merge| for ([_][2]i32{ .{ 8, 1024 }, .{ 4, 64 }, .{ 2048, 4096 } }) |mm| for ([_]i32{ 0, 1, 2 }) |algo| for ([_]i32{ 0, 1 }) |pad| {
                var h = base;
                h.patch_size = ps;
                h.n_merge = merge;
                const area = ps * ps * merge * merge;
                h.image_min_pixels = mm[0] * area;
                h.image_max_pixels = mm[1] * area;
                h.image_resize_algo = algo;
                h.image_resize_pad = pad;
                h.image_pad_color = .{ 1, 2, 3, 0 };
                try out.append(gpa, .{ h, .qwen2vl, "dyn" });
            };
        },
        .longest_edge => {
            for ([_]i32{ 512, 1024 }) |le| for ([_]i32{ 0, 1, 2 }) |merge| for ([_][2]i32{ .{ -1, -1 }, .{ 4096, 0 }, .{ 0, 65536 } }) |mm| {
                var h = base;
                h.patch_size = 14;
                h.n_merge = merge;
                h.image_longest_edge = le;
                h.image_min_pixels = mm[0];
                h.image_max_pixels = mm[1];
                h.image_resize_algo = 1;
                h.image_resize_pad = 0;
                try out.append(gpa, .{ h, .lightonocr, "longest" });
            };
        },
        .lfm2 => {
            for ([_][2]i32{ .{ 64, 256 }, .{ 16, 64 }, .{ 256, 1024 } }) |mm| for ([_]i32{ 0, 1, 2 }) |algo| {
                var h = base;
                h.patch_size = 16;
                h.n_merge = 2;
                h.image_size = 512;
                h.image_min_pixels = mm[0] * 1024;
                h.image_max_pixels = mm[1] * 1024;
                h.image_resize_algo_rf = algo;
                h.image_resize_algo_ov = algo;
                h.image_pad_rf = 0;
                h.image_pad_ov = 0;
                try out.append(gpa, .{ h, .lfm2, "lfm2" });
            };
        },
        .idefics3 => {
            for ([_]i32{ 384, 512 }) |is| for ([_]i32{ 0, 1536, 700 }) |le| for ([_][2]i32{ .{ -1, -1 }, .{ 100000, 0 }, .{ 0, 500000 } }) |mm| {
                var h = base;
                h.image_size = is;
                h.patch_size = 16;
                h.image_longest_edge = le;
                h.image_min_pixels = mm[0];
                h.image_max_pixels = mm[1];
                h.image_resize_algo_rf = 2;
                h.image_resize_algo_ov = 2;
                h.image_pad_rf = 0;
                h.image_pad_ov = 0;
                try out.append(gpa, .{ h, .idefics3, "idefics3" });
            };
        },
        .deepseekocr => {
            for ([_]names.ProjectorType{ .deepseekocr, .deepseekocr2 }) |pt| for ([_][2]i32{ .{ 256, 160 }, .{ 1024, 640 } }) |bt| for ([_][2]i32{ .{ 2, 9 }, .{ 1, 6 }, .{ 1, 40 } }) |tiles| {
                var h = base;
                h.image_size = bt[0];
                h.preproc_tile_size = bt[1];
                h.preproc_min_tiles = tiles[0];
                h.preproc_max_tiles = tiles[1];
                h.image_pad_color = .{ 127, 127, 127, 0 };
                try out.append(gpa, .{ h, pt, "deepseek" });
            };
        },
        .step3vl => {
            for ([_]i32{ 728, 300 }) |is| for ([_]i32{ 0, 1000 }) |le| {
                var h = base;
                h.image_size = is;
                h.patch_size = 14;
                h.image_longest_edge = le;
                try out.append(gpa, .{ h, .step3vl, "step3vl" });
            };
        },
        .youtuvl => {
            for ([_]i32{ 0, 256 * 256, 1024 * 256 }) |mx| for ([_]i32{ 0, 1, 2 }) |algo| for ([_]i32{ 0, 1, 2 }) |pad| {
                var h = base;
                h.patch_size = 16;
                h.n_merge = 2;
                h.image_max_pixels = if (mx == 0) -1 else mx;
                h.image_resize_algo = algo;
                h.image_resize_pad = pad;
                h.image_pad_color = .{ 9, 9, 9, 0 };
                try out.append(gpa, .{ h, .youtuvl, "youtuvl" });
            };
        },
        .granite => {
            for ([_]i32{ 0, 1, 2 }) |algo| for ([_][2]i32{ .{ 1, 0 }, .{ 2, 2 } }) |pads| {
                var h = base;
                h.image_size = 384;
                h.patch_size = 16;
                h.image_resize_algo_rf = algo;
                h.image_resize_algo_ov = algo;
                h.image_pad_rf = pads[0];
                h.image_pad_ov = pads[1];
                h.image_pad_color_rf = .{ 0, 128, 255, 0 };
                setCandidates(&h, &.{ .{ 384, 768 }, .{ 768, 384 }, .{ 768, 768 }, .{ 1152, 384 }, .{ 384, 1152 } });
                try out.append(gpa, .{ h, .granite4_vision, "granite" });
            };
        },
        .muse_glimmer => {
            for ([_]i32{ 14, 16 }) |ps| for ([_]i32{ 1, 2 }) |merge| for ([_]i32{ 64, 1024 }) |mt| {
                var h = base;
                h.patch_size = ps;
                h.n_merge = merge;
                h.image_max_pixels = mt * ps * ps * merge * merge;
                h.image_resize_algo = 1;
                try out.append(gpa, .{ h, .muse_glimmer, "muse" });
            };
        },
    }
    base = .{};
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    var filter: ?[]const u8 = null;
    for (args[1..]) |a| {
        if (std.mem.eql(u8, a, "-v")) verbose = true else filter = a;
    }

    ref_set_log(capture, &ref_log);
    impl.g_logger_state = .{ .log_callback = portCapture, .log_callback_user_data = &port_log };

    var seed: u64 = 1;
    inline for (@typeInfo(image.Kind).@"enum".field_names, @typeInfo(image.Kind).@"enum".field_values) |fname, fvalue| {
        const kind: image.Kind = @fromBackingInt(@intCast(fvalue));
        if (filter == null or std.mem.indexOf(u8, fname, filter.?) != null) {
            var vs: std.ArrayList(struct { RefHparams, names.ProjectorType, []const u8 }) = .empty;
            defer vs.deinit(gpa);
            try variants(kind, &vs);
            for (vs.items, 0..) |v, vi| {
                for (sizes, 0..) |sz, si| {
                    // every variant runs a spread of sizes; each size runs
                    // under every variant at least once, with a rotating
                    // pattern
                    if ((si + vi) % 3 != 0 and !(vi == 0)) continue;
                    const pattern: Pattern = switch ((si + vi) % 5) {
                        0 => .random,
                        1 => .gradient,
                        2 => .checker,
                        3 => .white,
                        else => if (vi % 2 == 0) .placeholder else .random,
                    };
                    seed +%= 0x9e3779b97f4a7c15;
                    const cs: Case = .{ .kind = kind, .proj = v[1], .hp = v[0], .nx = sz[0], .ny = sz[1], .pattern = pattern, .label = v[2] };
                    try runCase(&cs, seed);
                }
            }
        }
    }

    if (n_fail != 0) {
        std.debug.print("FAIL: {d} of {d} mtmd-image cases differ from upstream's\n", .{ n_fail, n_cases });
        return 1;
    }
    std.debug.print("coverage: {d} tiled ({d} kinds), {d} with an overview, {d} placeholders, {d} log bytes\n", .{ n_tiled, kinds_tiled.count(), n_overview, n_placeholder, n_log_bytes });
    std.debug.print("PASS: {d} mtmd-image cases identical to upstream's ({d} floats, and every log line)\n", .{ n_cases, n_floats });
    return 0;
}
