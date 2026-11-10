//! `make mtmd-image-diff`: every image preprocessor of `src/mtmd/image.zig`
//! against upstream's `mtmd-image.cpp`, compiled unmodified beside it
//! (`harness/mtmd_image_ref.cpp`), in one process, on bits.
//!
//! **Not a port.** A harness. Each case is a preprocessor kind, hparams and
//! an image; both sides' outputs -- the grid, the overview and every entry,
//! each with its size, flags, anyres info and every float's bits -- are
//! serialized the same way and must be byte-equal, and so must every line
//! the two log. A case upstream throws in must throw the same `what()` here,
//! and a preprocessor's `preprocess_video_frame` and `get_video_frame_size`
//! run beside `preprocess`.
//!
//! Three more sections cover what a run over images cannot reach: `sweep`
//! compares the static helpers upstream exposes for its tests over a grid of
//! sizes and hparams; `vectors` holds both sides to the expected values in
//! `tests/test-mtmd-impl.cpp` (HF's and torchvision's own answers, not
//! upstream's); `death` forks each side through the aborts and compares the
//! message.
//!
//! Usage: mtmd-image-diff [kind-name-substring | sweep | vectors | death] [-v]

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
    dsv4_max_n_token: i32 = 0,
    dsv4_max_wh_ratio: i32 = 0,
    max_slice_nums: i32 = 9, // v0.7.0
    preproc_max_pixels_tolerance: f32 = 1.0, // v0.8.0
    image_round_half_even: i32 = 0, // v0.8.0
    image_floor_before_align: i32 = 0, // v0.8.0
};

const ImageSize = impl.ImageSize;

extern fn ref_preprocess(kind: c_int, proj_type: c_int, hp: *const RefHparams, rgb: ?[*]const u8, nx: c_int, ny: c_int, video: c_int, out: *?[*]u8, out_len: *usize) c_int;
extern fn ref_video_frame_size(kind: c_int, hp: *const RefHparams, n_frames: c_int, width: c_int, height: c_int, out: *ImageSize) c_int;
extern fn ref_dyn_target_size(hp: *const RefHparams, w: c_int, h: c_int, out: *ImageSize) void;
extern fn ref_longest_target_size(hp: *const RefHparams, w: c_int, h: c_int, out: *ImageSize, what: *?[*]u8, what_len: *usize) c_int;
extern fn ref_qwen3vl_smart_resize(w: c_int, h: c_int, factor: c_int, min_pixels: i64, max_pixels: i64, n_frames: c_int, temporal_factor: c_int, out: *ImageSize) void;
extern fn ref_lfm2_should_tile(hp: *const RefHparams, w: c_int, h: c_int) c_int;
extern fn ref_lfm2_smart_resize(hp: *const RefHparams, w: c_int, h: c_int, out: *ImageSize) void;
extern fn ref_lfm2_grid_layout(hp: *const RefHparams, height: c_int, width: c_int, out: *ImageSize) void;
extern fn ref_deepseek41v_plan_grid(w: c_int, h: c_int, p: c_int, r: c_int, max_n_token: c_int, min_pixels: c_int, best_w: *c_int, best_h: *c_int) void;
extern fn ref_lfm2_slice_hash(hp: *const RefHparams, rgb: [*]const u8, nx: c_int, ny: c_int) u32;
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
    h.dsv4_max_n_token = rh.dsv4_max_n_token;
    h.dsv4_max_wh_ratio = rh.dsv4_max_wh_ratio;
    h.max_slice_nums = rh.max_slice_nums;
    h.preproc_max_pixels_tolerance = rh.preproc_max_pixels_tolerance;
    h.image_round_half_even = rh.image_round_half_even != 0;
    h.image_floor_before_align = rh.image_floor_before_align != 0;
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
    /// through `preprocess_video_frame`
    video: bool = false,
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
var n_threw: usize = 0;
var kinds_threw: std.EnumSet(image.Kind) = .empty;
var n_video: usize = 0;
var n_checks: usize = 0;
var n_death: usize = 0;

/// One failed comparison outside `runCase`.
fn failCheck(comptime fmt: []const u8, args: anytype) void {
    n_fail += 1;
    if (n_fail <= 20) std.debug.print("FAIL " ++ fmt ++ "\n", args);
}

fn runCase(cs: *const Case, seed: u64) !void {
    const rgb = try makeImage(cs.nx, cs.ny, cs.pattern, seed);
    defer gpa.free(rgb);
    const ph = cs.pattern == .placeholder;

    ref_log.clearRetainingCapacity();
    port_log.clearRetainingCapacity();

    var out_ptr: ?[*]u8 = null;
    var out_len: usize = 0;
    const ref_rc = ref_preprocess(@backingInt(cs.kind), @backingInt(cs.proj), &cs.hp, if (ph) null else rgb.ptr, cs.nx, cs.ny, @intFromBool(cs.video), &out_ptr, &out_len);
    if (ref_rc != 0 and ref_rc != 2) return error.RefFailed;
    defer ref_free(out_ptr);
    const ref_threw = ref_rc == 2;
    const ref_bytes = out_ptr.?[0..out_len];

    var hp = try toHparams(&cs.hp);
    defer hp.deinit();
    var img: impl.ImageU8 = .{};
    defer img.deinit();
    try img.setSize(.{ .width = cs.nx, .height = cs.ny }, ph);
    if (!ph) try img.cpyBuf(rgb);
    const pp = image.Preprocessor.init(cs.kind, &hp, cs.proj);
    const port_res = if (cs.video) pp.preprocessVideoFrame(&img) else pp.preprocess(&img);
    const port_threw = if (port_res) |_| false else |err| switch (err) {
        error.Runtime => true,
        else => return err,
    };
    var res: image.PreprocOut = port_res catch .{};
    defer res.deinit();
    var port_bytes: std.ArrayList(u8) = .empty;
    defer port_bytes.deinit(gpa);
    if (port_threw) {
        try port_bytes.appendSlice(gpa, mtmd.clip.what());
    } else {
        port_bytes.deinit(gpa);
        port_bytes = try serialize(&res);
    }

    n_cases += 1;
    n_log_bytes += ref_log.items.len;
    if (cs.video) n_video += 1;
    if (ref_threw) {
        n_threw += 1;
        kinds_threw.insert(cs.kind);
    }
    if (res.grid_x > 0 and res.grid_y > 0) {
        n_tiled += 1;
        kinds_tiled.insert(cs.kind);
    }
    if (res.hasOverview()) n_overview += 1;
    if (ph) n_placeholder += 1;
    for (res.entries.items) |*e| n_floats += e.roBuf().len;
    n_floats += res.overview.roBuf().len;

    // the video size, which every kind answers
    var ref_vsz: ImageSize = .{ .width = -1, .height = -1 };
    var port_vsz = ref_vsz;
    const n_frames = @as(c_int, @intCast(seed % 9)) + 1;
    const ref_has = ref_video_frame_size(@backingInt(cs.kind), &cs.hp, n_frames, cs.nx, cs.ny, &ref_vsz) != 0;
    const port_has = pp.getVideoFrameSize(n_frames, cs.nx, cs.ny, &port_vsz);
    if (ref_has != port_has or ref_vsz.width != port_vsz.width or ref_vsz.height != port_vsz.height) {
        failCheck("{s} kind={s} {d}x{d}: get_video_frame_size({d}) {any} {d}x{d} vs {any} {d}x{d}", .{
            cs.label,        @tagName(cs.kind), cs.nx,          cs.ny,    n_frames,
            ref_has,         ref_vsz.width,     ref_vsz.height, port_has, port_vsz.width,
            port_vsz.height,
        });
    }

    const same = ref_threw == port_threw and std.mem.eql(u8, ref_bytes, port_bytes.items);
    const same_log = std.mem.eql(u8, ref_log.items, port_log.items);
    if (!same or !same_log) {
        n_fail += 1;
        if (n_fail <= 20) {
            std.debug.print("FAIL {s} kind={s} {d}x{d} {s}{s}: out {s} ({d} vs {d} bytes), log {s}\n", .{
                cs.label,             @tagName(cs.kind),                  cs.nx,                          cs.ny,
                @tagName(cs.pattern), if (cs.video) " video" else "",     if (same) "same" else "DIFFER", ref_bytes.len,
                port_bytes.items.len, if (same_log) "same" else "DIFFER",
            });
            if (ref_threw != port_threw) {
                std.debug.print("    threw: ref {any}, port {any}\n", .{ ref_threw, port_threw });
            } else if (ref_threw and !same) {
                std.debug.print("    ref: {s}\n    port: {s}\n", .{ ref_bytes, port_bytes.items });
            }
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
    // exactly 8 tokens of 14-pixel patches merged 2x2 (6272 pixels) with
    // neither edge aligned, where glm5v's `>= min_px` keeps the scale at 1
    .{ 98, 64 },   .{ 64, 98 },
    // v0.7.0: at minicpmv's 448 / 56, the overview's resize lands an edge
    // on an odd multiple of half the alignment (476 = 8.5 * 56, 364 =
    // 6.5 * 56), where its round-half-to-even and std::round part
       .{ 100, 113 },  .{ 100, 151 },
};

/// Sizes only one kind runs: each sits on a decision that kind makes, and
/// several would abort another kind.
fn extraSizes(kind: image.Kind) []const [2]c_int {
    return switch (kind) {
        // torchvision's precision search on a knife edge; three of each, so
        // the rotation over variants reaches each one
        .fixed_size => &.{ .{ 669, 669 }, .{ 669, 669 }, .{ 669, 669 }, .{ 1021, 1021 }, .{ 1021, 1021 }, .{ 1021, 1021 } },
        // tests/test-mtmd-impl.cpp's, where half an alignment decides; the
        // first are `image_round_half_even`'s
        .dyn_size => &.{ .{ 768, 400 }, .{ 800, 336 }, .{ 400, 400 }, .{ 640, 240 }, .{ 966, 560 } },
        // below, at and above the longest edge, an edge floored to zero, and
        // upstream's test cases
        .longest_edge => &.{ .{ 463, 463 }, .{ 463, 300 }, .{ 448, 280 }, .{ 2000, 1000 }, .{ 3000, 1 }, .{ 1, 3000 }, .{ 512, 511 }, .{ 700, 512 }, .{ 1025, 77 } },
        // upstream's test cases: a tie between grids at 720x1600 in double
        // and not in float, half an alignment at 720, 400, 169
        .lfm2 => &.{ .{ 704, 704 }, .{ 720, 720 }, .{ 736, 736 }, .{ 1024, 977 }, .{ 1056, 384 }, .{ 400, 600 }, .{ 480, 336 }, .{ 64, 169 }, .{ 1024, 768 }, .{ 720, 1600 }, .{ 420, 1440 }, .{ 255, 2160 }, .{ 620, 410 }, .{ 2000, 1500 } },
        // an edge rounding to 0 (the max then keeps it at the factor), half
        // an alignment, and sizes either side of the pixel budgets
        .lfm3vl, .qwen3vl => &.{ .{ 3, 1000 }, .{ 1000, 3 }, .{ 15, 16 }, .{ 16, 16 }, .{ 17, 48 }, .{ 48, 80 }, .{ 80, 48 }, .{ 400, 400 }, .{ 720, 720 }, .{ 3000, 2000 } },
        // aspect ratios past both solver limits, an image scaled up to the
        // minimum, and one exactly on the destination's ratio
        .deepseek41v => &.{ .{ 3000, 2 }, .{ 2, 3000 }, .{ 1, 400 }, .{ 400, 1 }, .{ 2000, 3 }, .{ 3, 2000 }, .{ 96, 96 }, .{ 192, 96 }, .{ 2000, 2000 }, .{ 1920, 1080 } },
        // an overview edge at 14.5 * 28, where half to even and std::round
        // part
        .minicpm_ocr_1_0 => &.{ .{ 170, 140 }, .{ 140, 170 } },
        else => &.{},
    };
}

/// `RESIZE_ALGO_FLAG_TORCH`.
const torch = model.ResizeAlgo.flag_torch;

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
            // the torchvision resize beside Pillow's, at image sizes up and
            // down from the test sizes; at 3, where the weights are so small
            // the precision search never stops early; and at 77 and 7, where
            // from `extraSizes`' 669 and 1021 the largest weight is just
            // under a power of two and only the search's 0.5 rounds it up to
            // 2^15. Below 1600 only six pairs of sizes do that.
            for ([_]i32{ 0, 1, 2, torch | 0, torch | 1 }) |algo| for ([_]i32{ 0, 1, 2 }) |pad| for ([_][4]u8{ .{ 0, 0, 0, 0 }, .{ 127, 128, 129, 0 }, .{ 255, 0, 128, 0 } }) |col| {
                var h = base;
                h.image_size = if (algo == 1) 224 else 336;
                if (algo & torch != 0 and col[0] == 255) h.image_size = 3;
                if (algo & torch != 0 and col[0] == 127) h.image_size = if (algo == torch | 1) 77 else 7;
                h.patch_size = 14;
                h.image_resize_algo = algo;
                h.image_resize_pad = pad;
                h.image_pad_color = col;
                try out.append(gpa, .{ h, .mlp, "fixed" });
            };
        },
        .llava_uhd, .minicpmv, .internvl, .minicpm_ocr_1_0 => {
            // v0.7.0: minicpmv at the 4x merge too (its slice align is
            // patch * 4 whatever the merge), and max_slice_nums rotating
            // through the default, smaller caps and 0 (which means 9).
            // v0.8.0: minicpm_ocr_1_0's slice align follows the merge, and
            // a merge of 0 means 2 there
            var n_var: usize = 0;
            for ([_]bool{ false, true }) |pins| for ([_]i32{ 0, 1, 2, 4 }) |merge| for ([_][2]i32{ .{ 1, 0 }, .{ 0, 1 }, .{ 2, 2 } }) |algos| for ([_][2]i32{ .{ 1, 0 }, .{ 2, 1 }, .{ 0, 2 } }) |pads| {
                if (kind == .internvl and !pins) continue;
                if (merge == 4 and kind != .minicpmv and kind != .minicpm_ocr_1_0) continue;
                if (merge == 0 and kind != .minicpm_ocr_1_0) continue;
                var h = base;
                h.max_slice_nums = ([_]i32{ 9, 4, 0, 2 })[n_var % 4];
                n_var += 1;
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
            for ([_]i32{ 14, 16 }) |ps| for ([_]i32{ 1, 2 }) |merge| for ([_][2]i32{ .{ 8, 1024 }, .{ 4, 64 }, .{ 2048, 4096 } }) |mm| for ([_]i32{ 0, 1, 2, torch | 0, torch | 1 }) |algo| for ([_]i32{ 0, 1 }) |pad| for ([_]i32{ 0, 1 }) |even| {
                var h = base;
                h.patch_size = ps;
                h.n_merge = merge;
                const area = ps * ps * merge * merge;
                h.image_min_pixels = mm[0] * area;
                h.image_max_pixels = mm[1] * area;
                h.image_resize_algo = algo;
                h.image_resize_pad = pad;
                h.image_pad_color = .{ 1, 2, 3, 0 };
                h.image_round_half_even = even;
                try out.append(gpa, .{ h, .qwen2vl, "dyn" });
            };
        },
        .lfm3vl, .qwen3vl => {
            // as load_hparams sets it, PAD_NONE, and with padding; budgets
            // around the test sizes, an unset minimum; torchvision's resize
            // (v0.9.0: qwen3vl is lfm3vl with dyn_size's image size)
            for ([_]i32{ 14, 16 }) |ps| for ([_][2]i32{ .{ 64, 256 }, .{ 16, 64 }, .{ 256, 1024 }, .{ 0, 4096 }, .{ -1, 8 } }) |mm| for ([_]i32{ 0, 1, 2, torch | 0, torch | 1 }) |algo| for ([_]i32{ 0, 1 }) |pad| {
                // dyn_size's image size asserts both budgets
                if (kind == .qwen3vl and mm[0] <= 0) continue;
                var h = base;
                h.patch_size = ps;
                h.n_merge = 2;
                const area = ps * ps * 4;
                h.image_min_pixels = if (mm[0] < 0) -1 else mm[0] * area;
                h.image_max_pixels = mm[1] * area;
                h.image_resize_algo = algo;
                h.image_resize_pad = pad;
                h.image_pad_color = .{ 4, 5, 6, 0 };
                if (kind == .lfm3vl) try out.append(gpa, .{ h, .lfm3vl, "lfm3vl" }) else try out.append(gpa, .{ h, .qwen3vl, "qwen3vl" });
            };
        },
        .longest_edge => {
            for ([_]i32{ 512, 1024 }) |le| for ([_]i32{ 0, 1, 2 }) |merge| for ([_][2]i32{ .{ -1, -1 }, .{ 4096, 0 }, .{ 0, 65536 } }) |mm| for ([_]i32{ 0, 1 }) |floored| {
                var h = base;
                h.patch_size = 14;
                h.n_merge = merge;
                h.image_longest_edge = le;
                h.image_min_pixels = mm[0];
                h.image_max_pixels = mm[1];
                h.image_resize_algo = 1;
                h.image_resize_pad = 0;
                h.image_floor_before_align = floored;
                try out.append(gpa, .{ h, .lightonocr, "longest" });
            };
        },
        .lfm2 => {
            // the tiling as load_hparams defaults it, and in hparams a
            // gguf can set: no splitting at all, a single-tile minimum, a
            // minimum above two, other tile sizes and tolerances
            for ([_][2]i32{ .{ 64, 256 }, .{ 16, 64 }, .{ 256, 1024 } }) |mm| for ([_]i32{ 0, 1, 2, torch | 0, torch | 1 }) |algo| for ([_][4]i32{ .{ 2, 10, 512, 20 }, .{ 1, 1, 512, 20 }, .{ 1, 4, 256, 10 }, .{ 3, 12, 384, 15 } }) |tiling| {
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
                h.preproc_min_tiles = tiling[0];
                h.preproc_max_tiles = tiling[1];
                h.preproc_tile_size = tiling[2];
                h.preproc_max_pixels_tolerance = @as(f32, @floatFromInt(tiling[3])) / 10.0;
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
        .deepseek4v => {
            // the token caps load_hparams sets: the default, a custom one at
            // its floor of 16, and one between; min pixels as the gguf sets
            // them, lowered to the max, and absent; the wh-ratio cap on and off
            for ([_]i32{ 384, 16, 100 }) |mt| for ([_]i32{ 4, 64, 0 }) |mn| for ([_]i32{ 8, 0 }) |wh| for ([_]i32{ 1, 2 }) |algo| {
                var h = base;
                h.patch_size = 16;
                h.n_merge = 3;
                h.dsv4_max_n_token = mt;
                h.dsv4_max_wh_ratio = wh;
                const area = h.patch_size * h.patch_size * h.n_merge * h.n_merge;
                h.image_max_pixels = mt * area;
                h.image_min_pixels = if (mn == 0) 0 else @min(mn * area, h.image_max_pixels);
                h.image_resize_algo = algo;
                h.image_pad_color = .{ 127, 127, 127, 0 };
                try out.append(gpa, .{ h, .deepseek4v, "deepseek4v" });
            };
        },
        .deepseek41v => {
            // as load_hparams sets it, a custom token cap down to where the
            // solver's narrow branches run, min pixels on and off, Pillow's
            // algorithms and the grey pad and another
            for ([_]i32{ 2, 3 }) |merge| for ([_]i32{ 1024, 384, 16, 100 }) |mt| for ([_]i32{ 4, 64, 0 }) |mn| for ([_]i32{ 1, 2 }) |algo| {
                var h = base;
                h.patch_size = 16;
                h.n_merge = merge;
                h.dsv4_max_n_token = mt;
                const area = h.patch_size * h.patch_size * h.n_merge * h.n_merge;
                h.image_max_pixels = mt * area;
                h.image_min_pixels = if (mn == 0) 0 else @min(mn * area, h.image_max_pixels);
                h.image_resize_algo = algo;
                h.image_pad_color = if (merge == 2) .{ 127, 127, 127, 0 } else .{ 3, 200, 40, 0 };
                try out.append(gpa, .{ h, .deepseek41v, "deepseek41v" });
            };
        },
        .glm5v => {
            // budgets below, around and above every size's aligned canvas,
            // so the min-budget upscale, the height search and the plain
            // pad each run; min == max; a max below one aligned cell; and
            // 226 tokens at patch 14, where 224x224's upscaled edge (420.9)
            // floors onto a cell boundary, so only the C's ceil reaches 448
            for ([_]i32{ 14, 16 }) |ps| for ([_][2]i32{ .{ 8, 1024 }, .{ 4, 64 }, .{ 2048, 4096 }, .{ 64, 64 }, .{ 1, 1 }, .{ 226, 4096 } }) |mm| for ([_]i32{ 0, 1, 2 }) |algo| {
                var h = base;
                h.patch_size = ps;
                h.n_merge = 2;
                const area = ps * ps * 4;
                h.image_min_pixels = mm[0] * area;
                h.image_max_pixels = mm[1] * area;
                h.image_resize_algo = algo;
                h.image_resize_pad = 1;
                h.image_pad_color = .{ 1, 2, 3, 0 };
                try out.append(gpa, .{ h, .glm5v, "glm5v" });
            };
        },
        .cohere2v => {
            // as load_hparams sets it: the refined image stretched
            // (PAD_NONE); tile counts from one (never tiled) to the cap's
            // edge, and tile sizes either side of the test sizes
            for ([_]i32{ 32, 64, 364 }) |is| for ([_]i32{ 1, 2, 6, 12, 256 }) |mt| for ([_][2]i32{ .{ 1, 1 }, .{ 2, 0 } }) |algos| {
                var h = base;
                h.image_size = is;
                h.patch_size = 14;
                h.n_merge = 2;
                h.preproc_max_tiles = mt;
                h.image_resize_algo_rf = algos[0];
                h.image_resize_algo_ov = algos[1];
                h.image_pad_rf = 0;
                h.image_pad_ov = 0;
                try out.append(gpa, .{ h, .cohere2v, "cohere2v" });
            };
        },
    }
    base = .{};
}

// -----------------------------------------------------------------------------
// sweep

fn sameSize(a: ImageSize, b: ImageSize) bool {
    return a.width == b.width and a.height == b.height;
}

/// Every width and height in a spread: each small value, then steps that
/// land on the multiples of the alignments, and the extremes.
const sweep_edges = blk: {
    var edges: [147]c_int = undefined;
    var n = 0;
    for (1..65) |v| {
        edges[n] = v;
        n += 1;
    }
    for (1..51) |v| {
        edges[n] = 64 + v * 23;
        n += 1;
    }
    for ([_]c_int{ 400, 720, 736, 977, 1024, 1440, 1540, 1600, 2000, 2160, 3000, 4000, 5000, 7999, 8192, 12000, 16384, 20000, 32768, 40000, 3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47 }) |v| {
        edges[n] = v;
        n += 1;
    }
    break :blk edges;
};

fn sweepDyn() !void {
    for ([_][2]i32{ .{ 14, 1 }, .{ 14, 2 }, .{ 16, 1 }, .{ 16, 2 } }) |pm| for ([_][2]i32{ .{ 8, 4096 }, .{ 4, 64 }, .{ 1, 1 } }) |mm| for ([_]i32{ 0, 1 }) |even| {
        var rh: RefHparams = .{ .patch_size = pm[0], .n_merge = pm[1], .image_round_half_even = even };
        const area = pm[0] * pm[0] * pm[1] * pm[1];
        rh.image_min_pixels = mm[0] * area;
        rh.image_max_pixels = mm[1] * area;
        var hp = try toHparams(&rh);
        defer hp.deinit();
        for (sweep_edges) |w| for (sweep_edges) |h| {
            var ref: ImageSize = .{};
            ref_dyn_target_size(&rh, w, h, &ref);
            const port = image.Preprocessor.dynSizeGetTargetSize(&hp, .{ .width = w, .height = h });
            n_checks += 1;
            if (!sameSize(ref, port)) failCheck("dyn get_target_size {d}x{d} p{d} m{d} even={d}: {d}x{d} vs {d}x{d}", .{ w, h, pm[0], pm[1], even, ref.width, ref.height, port.width, port.height });
        };
    };
}

fn sweepLongest() !void {
    for ([_]i32{ 7, 512, 1024, 1540 }) |le| for ([_]i32{ 0, 1, 2 }) |merge| for ([_][2]i32{ .{ -1, -1 }, .{ 4096, 0 }, .{ 0, 65536 }, .{ 0, 1048576 } }) |mm| for ([_]i32{ 0, 1 }) |floored| {
        const rh: RefHparams = .{ .patch_size = 14, .n_merge = merge, .image_longest_edge = le, .image_min_pixels = mm[0], .image_max_pixels = mm[1], .image_floor_before_align = floored };
        var hp = try toHparams(&rh);
        defer hp.deinit();
        for (sweep_edges) |w| for (sweep_edges) |h| {
            var ref: ImageSize = .{};
            var what_ptr: ?[*]u8 = null;
            var what_len: usize = 0;
            const rc = ref_longest_target_size(&rh, w, h, &ref, &what_ptr, &what_len);
            defer if (rc == 2) ref_free(what_ptr);
            n_checks += 1;
            if (image.Preprocessor.longestEdgeGetTargetSize(&hp, .{ .width = w, .height = h })) |port| {
                if (rc != 0 or !sameSize(ref, port)) failCheck("longest get_target_size {d}x{d} le{d} m{d} floor={d}: rc {d} {d}x{d} vs {d}x{d}", .{ w, h, le, merge, floored, rc, ref.width, ref.height, port.width, port.height });
            } else |err| {
                if (err != error.Runtime) return err;
                n_threw += 1;
                if (rc != 2 or !std.mem.eql(u8, what_ptr.?[0..what_len], mtmd.clip.what())) failCheck("longest get_target_size {d}x{d} le{d} m{d} floor={d}: rc {d} vs threw {s}", .{ w, h, le, merge, floored, rc, mtmd.clip.what() });
            }
        };
    };
}

fn sweepLfm3vl() void {
    const budgets = [_][2]i64{ .{ 65536, 262144 }, .{ 4096, 25165824 }, .{ 0, 4194304 }, .{ 1048576, 1048576 }, .{ 1024, 2048 } };
    for ([_]c_int{ 32, 28, 16 }) |factor| for (budgets) |mm| for ([_][2]c_int{ .{ 1, 1 }, .{ 1, 2 }, .{ 2, 2 }, .{ 3, 2 }, .{ 4, 2 }, .{ 5, 2 }, .{ 7, 2 }, .{ 8, 2 }, .{ 16, 2 }, .{ 101, 2 }, .{ 5, 3 } }) |ft| {
        for (sweep_edges) |w| for (sweep_edges) |h| {
            var ref: ImageSize = .{};
            ref_qwen3vl_smart_resize(w, h, factor, mm[0], mm[1], ft[0], ft[1], &ref);
            const port = image.Preprocessor.qwen3vlSmartResize(w, h, factor, mm[0], mm[1], ft[0], ft[1]);
            n_checks += 1;
            if (!sameSize(ref, port)) failCheck("lfm3vl smart_resize {d}x{d} f{d} {d}..{d} frames {d}/{d}: {d}x{d} vs {d}x{d}", .{ w, h, factor, mm[0], mm[1], ft[0], ft[1], ref.width, ref.height, port.width, port.height });
        };
    };
}

/// `get_video_frame_size`: its budgets are constants, so frame counts up to
/// and past 24 at 1024x1024 land a product exactly on `longest_edge`, and
/// three to five at 32x32 exactly on `shortest_edge`.
fn sweepVideoFrame() !void {
    for ([_]i32{ 14, 16 }) |ps| {
        const rh: RefHparams = .{ .patch_size = ps, .n_merge = 2, .image_min_pixels = 64 * ps * ps * 4, .image_max_pixels = 256 * ps * ps * 4 };
        var hp = try toHparams(&rh);
        defer hp.deinit();
        // v0.9.0: qwen3vl owns the video size, lfm3vl inherits it
        for ([_]struct { image.Kind, names.ProjectorType }{ .{ .lfm3vl, .lfm3vl }, .{ .qwen3vl, .qwen3vl } }) |kp| {
            const pp = image.Preprocessor.init(kp[0], &hp, kp[1]);
            var n_frames: c_int = 1;
            while (n_frames <= 32) : (n_frames += 1) {
                for (sweep_edges) |w| for (sweep_edges) |h| {
                    var ref: ImageSize = .{};
                    var port: ImageSize = .{};
                    const ref_has = ref_video_frame_size(@backingInt(kp[0]), &rh, n_frames, w, h, &ref) != 0;
                    const port_has = pp.getVideoFrameSize(n_frames, w, h, &port);
                    n_checks += 1;
                    if (!ref_has or !port_has or !sameSize(ref, port)) failCheck("get_video_frame_size {s} {d} frames {d}x{d} p{d}: {d}x{d} vs {d}x{d}", .{ @tagName(kp[0]), n_frames, w, h, ps, ref.width, ref.height, port.width, port.height });
                };
            }
        }
    }
}

fn sweepLfm2() !void {
    for ([_][2]i32{ .{ 64, 256 }, .{ 16, 64 }, .{ 256, 1024 } }) |mm| for ([_][4]i32{ .{ 2, 10, 512, 20 }, .{ 1, 1, 512, 20 }, .{ 1, 4, 256, 10 }, .{ 3, 12, 384, 15 }, .{ 2, 64, 512, 11 } }) |tiling| {
        const rh: RefHparams = .{
            .patch_size = 16,
            .n_merge = 2,
            .image_min_pixels = mm[0] * 1024,
            .image_max_pixels = mm[1] * 1024,
            .preproc_min_tiles = tiling[0],
            .preproc_max_tiles = tiling[1],
            .preproc_tile_size = tiling[2],
            .preproc_max_pixels_tolerance = @as(f32, @floatFromInt(tiling[3])) / 10.0,
        };
        var hp = try toHparams(&rh);
        defer hp.deinit();
        for (sweep_edges) |w| for (sweep_edges) |h| {
            const sz: ImageSize = .{ .width = w, .height = h };
            const ref_tile = ref_lfm2_should_tile(&rh, w, h) != 0;
            const port_tile = image.Preprocessor.lfm2ShouldTile(&hp, sz);
            var ref: ImageSize = .{};
            ref_lfm2_smart_resize(&rh, w, h, &ref);
            const port = image.Preprocessor.lfm2SmartResize(&hp, sz);
            var ref_grid: ImageSize = .{};
            ref_lfm2_grid_layout(&rh, h, w, &ref_grid);
            const port_grid = try image.Preprocessor.lfm2GetGridLayout(&hp, h, w);
            n_checks += 3;
            if (ref_tile != port_tile) failCheck("lfm2 should_tile {d}x{d} tiling {any}: {any} vs {any}", .{ w, h, tiling, ref_tile, port_tile });
            if (!sameSize(ref, port)) failCheck("lfm2 smart_resize {d}x{d} {any}: {d}x{d} vs {d}x{d}", .{ w, h, mm, ref.width, ref.height, port.width, port.height });
            if (!sameSize(ref_grid, port_grid)) failCheck("lfm2 get_grid_layout {d}x{d} tiling {any}: {d}x{d} vs {d}x{d}", .{ w, h, tiling, ref_grid.width, ref_grid.height, port_grid.width, port_grid.height });
        };
    };
}

fn sweepPlanGrid() void {
    for ([_][2]c_int{ .{ 16, 2 }, .{ 16, 3 }, .{ 14, 2 } }) |pr| for ([_]c_int{ 1024, 384, 16, 100, 9 }) |mt| for ([_]c_int{ 0, 4096, 65536, 1000000 }) |mn| {
        for (sweep_edges) |w| for (sweep_edges) |h| {
            var rw: c_int = -1;
            var rh: c_int = -1;
            ref_deepseek41v_plan_grid(w, h, pr[0], pr[1], mt, mn, &rw, &rh);
            var pw: c_int = -1;
            var ph: c_int = -1;
            image.Preprocessor.deepseek41vPlanGrid(w, h, pr[0], pr[1], mt, mn, &pw, &ph);
            n_checks += 1;
            if (rw != pw or rh != ph) failCheck("plan_grid {d}x{d} p{d} r{d} max {d} min {d}: {d}x{d} vs {d}x{d}", .{ w, h, pr[0], pr[1], mt, mn, rw, rh, pw, ph });
        };
    };
}

// -----------------------------------------------------------------------------
// vectors

fn expectSize(name: []const u8, got: ImageSize, want: [2]c_int) void {
    n_checks += 1;
    if (got.width != want[0] or got.height != want[1]) failCheck("vector {s}: {d}x{d}, HF says {d}x{d}", .{ name, got.width, got.height, want[0], want[1] });
}

/// `make_test_image` (tests/test-mtmd-impl.cpp:118): the LCG's pixels.
fn lcgImage(nx: c_int, ny: c_int) ![]u8 {
    const buf = try gpa.alloc(u8, @as(usize, @intCast(nx)) * @as(usize, @intCast(ny)) * 3);
    var s: u32 = 1;
    for (buf) |*b| {
        s = s *% 1103515245 +% 12345;
        b.* = @truncate(s >> 16);
    }
    return buf;
}

fn hashPixels(img: *const impl.ImageU8, seed: u32) u32 {
    var hash = seed;
    for (img.roBuf()) |v| hash = (hash ^ v) *% 16777619;
    return hash;
}

/// The expected values in `tests/test-mtmd-impl.cpp`, which come from
/// transformers and torchvision: both sides must give them, so a wrong
/// reading of upstream shared by this harness and the port fails here.
fn runVectors() !void {
    {
        // test_image_preprocessor_longest_edge
        const rh: RefHparams = .{ .patch_size = 14, .n_merge = 2, .image_longest_edge = 1540 };
        var hp = try toHparams(&rh);
        defer hp.deinit();
        for ([_][4]c_int{ .{ 463, 463, 476, 476 }, .{ 463, 300, 476, 308 }, .{ 448, 280, 448, 280 }, .{ 2000, 1000, 1540, 784 } }) |v| {
            var ref: ImageSize = .{};
            var what_ptr: ?[*]u8 = null;
            var what_len: usize = 0;
            if (ref_longest_target_size(&rh, v[0], v[1], &ref, &what_ptr, &what_len) != 0) return error.RefFailed;
            expectSize("longest_edge ref", ref, .{ v[2], v[3] });
            expectSize("longest_edge port", try image.Preprocessor.longestEdgeGetTargetSize(&hp, .{ .width = v[0], .height = v[1] }), .{ v[2], v[3] });
        }
    }
    // test_image_preprocessor_dyn_size
    for ([_][5]c_int{ .{ 16, 768, 400, 768, 384 }, .{ 16, 800, 336, 800, 320 }, .{ 16, 400, 400, 384, 384 }, .{ 16, 640, 240, 640, 256 }, .{ 14, 966, 560, 952, 560 } }) |v| {
        const area = v[0] * v[0] * 4;
        const rh: RefHparams = .{ .patch_size = v[0], .n_merge = 2, .image_round_half_even = 1, .image_min_pixels = 8 * area, .image_max_pixels = 4096 * area };
        var hp = try toHparams(&rh);
        defer hp.deinit();
        var ref: ImageSize = .{};
        ref_dyn_target_size(&rh, v[1], v[2], &ref);
        expectSize("dyn_size ref", ref, .{ v[3], v[4] });
        expectSize("dyn_size port", image.Preprocessor.dynSizeGetTargetSize(&hp, .{ .width = v[1], .height = v[2] }), .{ v[3], v[4] });
    }

    // test_image_preprocessor_lfm2
    var rh: RefHparams = .{
        .patch_size = 16,
        .n_merge = 2,
        .image_min_pixels = 64 * 1024,
        .image_max_pixels = 256 * 1024,
        .preproc_min_tiles = 2,
        .preproc_max_tiles = 10,
        .preproc_tile_size = 512,
        .preproc_max_pixels_tolerance = 2.0,
        .image_pad_rf = 0,
    };
    var hp = try toHparams(&rh);
    defer hp.deinit();
    for ([_]struct { c_int, c_int, bool }{ .{ 704, 704, false }, .{ 720, 720, false }, .{ 736, 736, true }, .{ 1024, 977, true }, .{ 1056, 384, false } }) |v| {
        n_checks += 2;
        if ((ref_lfm2_should_tile(&rh, v[0], v[1]) != 0) != v[2]) failCheck("vector should_tile ref {d}x{d}", .{ v[0], v[1] });
        if (image.Preprocessor.lfm2ShouldTile(&hp, .{ .width = v[0], .height = v[1] }) != v[2]) failCheck("vector should_tile port {d}x{d}", .{ v[0], v[1] });
    }
    for ([_][4]c_int{ .{ 400, 600, 384, 608 }, .{ 480, 336, 480, 320 }, .{ 64, 169, 160, 416 }, .{ 1024, 768, 576, 416 } }) |v| {
        var ref: ImageSize = .{};
        ref_lfm2_smart_resize(&rh, v[0], v[1], &ref);
        expectSize("lfm2 smart_resize ref", ref, .{ v[2], v[3] });
        expectSize("lfm2 smart_resize port", image.Preprocessor.lfm2SmartResize(&hp, .{ .width = v[0], .height = v[1] }), .{ v[2], v[3] });
    }
    for ([_][4]c_int{ .{ 720, 1600, 2, 4 }, .{ 420, 1440, 1, 3 }, .{ 255, 2160, 1, 8 }, .{ 1024, 768, 3, 2 } }) |v| {
        var ref: ImageSize = .{};
        ref_lfm2_grid_layout(&rh, v[1], v[0], &ref);
        expectSize("lfm2 grid ref", ref, .{ v[2], v[3] });
        expectSize("lfm2 grid port", try image.Preprocessor.lfm2GetGridLayout(&hp, v[1], v[0]), .{ v[2], v[3] });
    }
    {
        // do_image_splitting=false
        var one = rh;
        one.preproc_min_tiles = 1;
        one.preproc_max_tiles = 1;
        var hp_one = try toHparams(&one);
        defer hp_one.deinit();
        n_checks += 2;
        if (ref_lfm2_should_tile(&one, 1024, 977) != 0) failCheck("vector should_tile ref max_tiles=1", .{});
        if (image.Preprocessor.lfm2ShouldTile(&hp_one, .{ .width = 1024, .height = 977 })) failCheck("vector should_tile port max_tiles=1", .{});
    }

    // test_image_preprocessor_lfm2_torch_resize: Lfm2VlImageProcessor's pixels
    for ([_]struct { i32, c_int, c_int, u32 }{
        .{ 0, 620, 410, 0xf5e9c941 },
        .{ 0, 2000, 1500, 0x3331f56c },
        .{ 1, 620, 410, 0x2b34eac9 },
        .{ 1, 2000, 1500, 0xb035d942 },
    }) |v| {
        const rgb = try lcgImage(v[1], v[2]);
        defer gpa.free(rgb);
        rh.image_resize_algo_rf = torch | v[0];
        rh.image_resize_algo_ov = torch | v[0];
        var hpt = try toHparams(&rh);
        defer hpt.deinit();
        var img: impl.ImageU8 = .{};
        defer img.deinit();
        try img.setSize(.{ .width = v[1], .height = v[2] }, false);
        try img.cpyBuf(rgb);
        const pp = image.Preprocessor.init(.lfm2, &hpt, .lfm2);
        var inst = try pp.getSliceInstructions(img.getSize());
        defer inst.deinit();
        var sliced = try pp.sliceImage(&img, &inst);
        defer sliced.deinit();
        var hash: u32 = 2166136261;
        for (sliced.slices.items) |*sl| hash = hashPixels(sl, hash);
        hash = hashPixels(&sliced.overview, hash);
        const ref_hash = ref_lfm2_slice_hash(&rh, rgb.ptr, v[1], v[2]);
        n_checks += 2;
        if (ref_hash != v[3]) failCheck("vector torch resize ref {d}x{d} algo {d}: {x} vs torchvision {x}", .{ v[1], v[2], v[0], ref_hash, v[3] });
        if (hash != v[3]) failCheck("vector torch resize port {d}x{d} algo {d}: {x} vs torchvision {x}", .{ v[1], v[2], v[0], hash, v[3] });
    }
}

// -----------------------------------------------------------------------------
// death

extern "c" fn fork() c_int;
extern "c" fn pipe(fds: *[2]c_int) c_int;
extern "c" fn dup2(old: c_int, new: c_int) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn waitpid(pid: c_int, status: *c_int, options: c_int) c_int;
extern "c" fn _exit(code: c_int) noreturn;
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

const DeathResult = struct { status: c_int, msg: []u8 };

/// Runs `body` in a forked child with its stderr captured, the backtrace off.
fn inChild(comptime body: anytype, args: anytype) !DeathResult {
    var fds: [2]c_int = undefined;
    if (pipe(&fds) != 0) return error.PipeFailed;
    const pid = fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        _ = setenv("GGML_NO_BACKTRACE", "1", 1);
        _ = dup2(fds[1], 2);
        _ = close(fds[0]);
        @call(.auto, body, args) catch {};
        _exit(0);
    }
    _ = close(fds[1]);
    var msg: std.ArrayList(u8) = .empty;
    var chunk: [4096]u8 = undefined;
    while (true) {
        const n = read(fds[0], &chunk, chunk.len);
        if (n <= 0) break;
        try msg.appendSlice(gpa, chunk[0..@intCast(n)]);
    }
    _ = close(fds[0]);
    var status: c_int = 0;
    _ = waitpid(pid, &status, 0);
    return .{ .status = status, .msg = try msg.toOwnedSlice(gpa) };
}

/// `__FILE__` is a full path on the reference's side and a bare name on the
/// port's: the message from the file's name on.
fn fromBasename(msg: []const u8) []const u8 {
    const line_end = std.mem.indexOfScalar(u8, msg, '\n') orelse msg.len;
    const name = std.mem.lastIndexOfScalar(u8, msg[0..line_end], '/') orelse return msg;
    return msg[name + 1 ..];
}

fn refDie(kind: image.Kind, rh: *const RefHparams, rgb: []const u8, nx: c_int, ny: c_int) !void {
    var out_ptr: ?[*]u8 = null;
    var out_len: usize = 0;
    _ = ref_preprocess(@backingInt(kind), 0, rh, rgb.ptr, nx, ny, 0, &out_ptr, &out_len);
}

fn portDie(kind: image.Kind, rh: *const RefHparams, rgb: []const u8, nx: c_int, ny: c_int) !void {
    var hp = try toHparams(rh);
    var img: impl.ImageU8 = .{};
    try img.setSize(.{ .width = nx, .height = ny }, false);
    try img.cpyBuf(rgb);
    const pp = image.Preprocessor.init(kind, &hp, .mlp);
    _ = try pp.preprocess(&img);
}

fn compareDeath(label: []const u8, phrase: []const u8, ref: DeathResult, port: DeathResult) void {
    defer gpa.free(ref.msg);
    defer gpa.free(port.msg);
    n_death += 1;
    const ref_msg = fromBasename(ref.msg);
    const port_msg = fromBasename(port.msg);
    if (std.mem.indexOf(u8, ref_msg, phrase) == null) failCheck("death {s}: the reference said [{s}] (status {d}), not [{s}]", .{ label, ref.msg, ref.status, phrase });
    if (ref.status != port.status or !std.mem.eql(u8, ref_msg, port_msg)) {
        failCheck("death {s}: status {d} vs {d}, message [{s}] vs [{s}]", .{ label, ref.status, port.status, ref_msg, port_msg });
    }
}

fn runDeaths() !void {
    const rgb = try makeImage(64, 48, .random, 7);
    defer gpa.free(rgb);
    const Die = struct { []const u8, []const u8, image.Kind, RefHparams };
    for ([_]Die{
        .{ "torch lanczos", "torchvision has no lanczos", .fixed_size, .{ .image_size = 224, .patch_size = 14, .image_resize_algo = torch | 2 } },
        .{ "unknown algo", "unsupported resize algo: 3", .fixed_size, .{ .image_size = 224, .patch_size = 14, .image_resize_algo = 3 } },
        .{ "unknown torch algo", "unsupported resize algo: 3", .fixed_size, .{ .image_size = 224, .patch_size = 14, .image_resize_algo = torch | 3 } },
        .{ "dyn_size pixels", "hparams.image_min_pixels > 0", .dyn_size, .{ .patch_size = 14, .image_min_pixels = 0, .image_max_pixels = 1024 } },
        .{ "longest_edge unset", "hparams.image_longest_edge > 0", .longest_edge, .{ .patch_size = 14 } },
    }) |d| {
        const ref = try inChild(refDie, .{ d[2], &d[3], rgb, 64, 48 });
        const port = try inChild(portDie, .{ d[2], &d[3], rgb, 64, 48 });
        compareDeath(d[0], d[1], ref, port);
    }
}

fn wanted(filter: ?[]const u8, name: []const u8) bool {
    return filter == null or std.mem.indexOf(u8, name, filter.?) != null;
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
            const extra = extraSizes(kind);
            for (vs.items, 0..) |v, vi| {
                for (0..sizes.len + extra.len) |si| {
                    const sz = if (si < sizes.len) sizes[si] else extra[si - sizes.len];
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
                    var cs: Case = .{ .kind = kind, .proj = v[1], .hp = v[0], .nx = sz[0], .ny = sz[1], .pattern = pattern, .label = v[2] };
                    try runCase(&cs, seed);
                    // lfm3vl snaps a video frame to the grid; for every
                    // other kind the default must be `preprocess`
                    if (kind == .lfm3vl or si % 11 == 0) {
                        cs.video = true;
                        try runCase(&cs, seed);
                    }
                }
            }
        }
    }

    if (wanted(filter, "sweep")) {
        try sweepDyn();
        try sweepLongest();
        sweepLfm3vl();
        try sweepVideoFrame();
        try sweepLfm2();
        sweepPlanGrid();
    }
    if (wanted(filter, "vectors")) try runVectors();
    if (wanted(filter, "death")) try runDeaths();

    if (n_fail != 0) {
        std.debug.print("FAIL: {d} of {d} mtmd-image cases and {d} checks differ from upstream's\n", .{ n_fail, n_cases, n_checks });
        return 1;
    }
    // a run with no throw reached no arm that throws
    if (filter == null and (n_threw == 0 or !kinds_threw.contains(.longest_edge) or !kinds_threw.contains(.deepseek41v))) {
        std.debug.print("FAIL: {d} cases threw, and longest_edge and deepseek41v must\n", .{n_threw});
        return 1;
    }
    std.debug.print("coverage: {d} tiled ({d} kinds), {d} with an overview, {d} placeholders, {d} threw ({d} kinds), {d} video frames, {d} log bytes\n", .{ n_tiled, kinds_tiled.count(), n_overview, n_placeholder, n_threw, kinds_threw.count(), n_video, n_log_bytes });
    std.debug.print("PASS: {d} mtmd-image cases identical to upstream's ({d} floats, and every log line), {d} helper checks, {d} aborts\n", .{ n_cases, n_floats, n_checks, n_death });
    return 0;
}
