//! `clip_encode`'s per-projector input switch, and the per-projector body
//! of `clip_n_output_tokens`.
//!
//! # Provenance
//!
//! Ports two switches of `llama.cpp/tools/mtmd/clip.cpp` at v0.3.0
//! (`c1d0e7a00`): the one that sets each projector's graph inputs in
//! `clip_encode`, and `clip_n_output_tokens`'s token counts. Each
//! declaration names what it ports and the line it began at.
//!
//! # Numerics, as the reference compiler emits them
//!
//! - libm goes through the C library (`extern fn`), never a Zig builtin.
//! - On Darwin clang folds a `sinf`/`cosf` pair on one argument into
//!   `__sincosf_stret` (the release reference `clip.o` imports it and neither
//!   `sinf` nor `cosf`); elsewhere `-fmath-errno` keeps two calls. `sinCos`.
//! - A same-type `a*b + c` is contracted, the left product fused first
//!   (`@mulAdd`); a sum of products folds left to right into nested fmas.
//! - The two libc++ distributions drawn here, `uniform_real_distribution`
//!   and `normal_distribution<float>`, are reproduced from the bundled
//!   headers (`uniformRealF32`, `NormalF32`).

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("impl.zig");
const clip = @import("clip.zig");
const graph = @import("graph.zig");
const libcxx = @import("llama").libcxx;
const c = impl.c;
const gpa = impl.gpa;

extern fn expf(x: f32) f32;
extern fn logf(x: f32) f32;
extern fn powf(x: f32, y: f32) f32;
extern fn sinf(x: f32) f32;
extern fn cosf(x: f32) f32;
extern fn sqrtf(x: f32) f32;
extern fn exp(x: f64) f64;
extern fn log(x: f64) f64;
extern fn sqrt(x: f64) f64;
extern fn lround(x: f64) c_long;
extern fn snprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, ...) c_int;

const Float2 = extern struct { sin: f32, cos: f32 };
extern fn __sincosf_stret(x: f32) Float2;

/// `sinf(x)` and `cosf(x)` as clang emits the pair on one argument: one
/// `__sincosf_stret` on Darwin, two calls elsewhere (see the file header).
fn sinCos(x: f32) Float2 {
    if (comptime builtin.os.tag.isDarwin()) return __sincosf_stret(x);
    return .{ .sin = sinf(x), .cos = cosf(x) };
}

/// libc++'s `uniform_real_distribution<float>(a, b)(g)`
/// (`<__random/uniform_real_distribution.h>`): `(b - a) * canonical + a`,
/// contracted.
fn uniformRealF32(g: *libcxx.Mt19937, a: f32, b: f32) f32 {
    return @mulAdd(f32, b - a, libcxx.canonicalF32(g), a);
}

/// libc++'s `normal_distribution<float>` (`<__random/normal_distribution.h>`):
/// Marsaglia's polar method, keeping the second value for the next draw.
const NormalF32 = struct {
    mean: f32,
    stddev: f32,
    v: f32 = 0,
    v_hot: bool = false,

    fn next(self: *NormalF32, g: *libcxx.Mt19937) f32 {
        var up: f32 = undefined;
        if (self.v_hot) {
            self.v_hot = false;
            up = self.v;
        } else {
            var u: f32 = undefined;
            var v: f32 = undefined;
            var s: f32 = undefined;
            while (true) {
                u = uniformRealF32(g, -1, 1);
                v = uniformRealF32(g, -1, 1);
                s = @mulAdd(f32, u, u, v * v);
                if (!(s > 1 or s == 0)) break;
            }
            const fp = sqrtf(-2 * logf(s) / s);
            self.v = v * fp;
            self.v_hot = true;
            up = u * fp;
        }
        return @mulAdd(f32, up, self.stddev, self.mean);
    }
};

/// `CLIP_ALIGN` (clip-impl.h:435 @c1d0e7a00), over `int`.
fn clipAlign(x: c_int, n: c_int) c_int {
    return @divTrunc(x + n - 1, n) * n;
}

fn allocI32(n: usize, fill: i32) clip.Error![]i32 {
    const v = try gpa.alloc(i32, n);
    @memset(v, fill);
    return v;
}

fn allocF32(n: usize, fill: f32) clip.Error![]f32 {
    const v = try gpa.alloc(f32, n);
    @memset(v, fill);
    return v;
}

fn usz(x: anytype) usize {
    return @intCast(x);
}

const lowest = -std.math.floatMax(f32);

/// Ports the per-projector input switch of `clip_encode` (mtmd/clip.cpp:4369
/// @c1d0e7a00), entered at mtmd/clip.cpp:4534.
///
/// Return: false where the C++ `return false`s out of `clip_encode` (a
/// qwen3tts code out of range), having logged why.
pub fn setInputsChecked(st: *clip.EncodeState) clip.Error!bool {
    const func = "clip_encode";
    const ctx = st.ctx;
    const model = &ctx.model;
    const hparams = &model.hparams;
    const params = st.params;
    const imgs = st.imgs;
    const gf = st.gf;

    const image_size_width = st.image_size_width;
    const image_size_height = st.image_size_height;
    const patch_size = st.patch_size;
    const num_patches = st.num_patches;
    const n_pos = st.n_pos;
    const pos_w = st.pos_w;
    const pos_h = st.pos_h;

    switch (model.proj_type) {
        .muse_glimmer => {
            const grid_w = pos_w; // image_size_width  / patch_size
            const grid_h = pos_h; // image_size_height / patch_size
            const n_tok = grid_w * grid_h;
            const pgrid: c_int = @intFromFloat(sqrt(@floatFromInt(graph.ne(model.position_embeddings, 1)))); // 32
            const f = hparams.n_merge; // downsample 2

            // --- sparse window grouping (pgrid x pgrid windows) ---
            const win = pgrid;
            const nwin_h = @divTrunc(grid_h + win - 1, win);
            const nwin_w = @divTrunc(grid_w + win - 1, win);
            var sp_perm: std.ArrayList(i32) = .empty;
            defer sp_perm.deinit(gpa);
            try sp_perm.ensureTotalCapacity(gpa, usz(n_tok));
            var sp_slens: std.ArrayList(c_int) = .empty;
            defer sp_slens.deinit(gpa);
            var wy: c_int = 0;
            while (wy < nwin_h) : (wy += 1) {
                var wx: c_int = 0;
                while (wx < nwin_w) : (wx += 1) {
                    var cnt: c_int = 0;
                    var hh: c_int = 0;
                    while (hh < win) : (hh += 1) {
                        var ww: c_int = 0;
                        while (ww < win) : (ww += 1) {
                            const gy = wy * win + hh;
                            const gx = wx * win + ww;
                            if (gy < grid_h and gx < grid_w) {
                                try sp_perm.append(gpa, gy * grid_w + gx);
                                cnt += 1;
                            }
                        }
                    }
                    if (cnt > 0) try sp_slens.append(gpa, cnt);
                }
            }
            const rpos_w = try allocI32(usz(n_tok), 0);
            defer gpa.free(rpos_w);
            const rpos_h = try allocI32(usz(n_tok), 0);
            defer gpa.free(rpos_h);
            const inv_perm = try allocI32(usz(n_tok), 0);
            defer gpa.free(inv_perm);
            for (0..usz(n_tok)) |i| {
                const orig = sp_perm.items[i];
                rpos_w[i] = @rem(orig, grid_w) + 1; // 1-indexed
                rpos_h[i] = @divTrunc(orig, grid_w) + 1;
                inv_perm[usz(orig)] = @intCast(i);
            }
            st.setInputI32("muse_glimmer_sp_perm", sp_perm.items);
            st.setInputI32("muse_glimmer_inv_perm", inv_perm);
            st.setInputI32("muse_glimmer_pos_w", rpos_w);
            st.setInputI32("muse_glimmer_pos_h", rpos_h);

            // block-diagonal window mask (permuted order)
            const nt = usz(n_tok);
            const sp_mask = try allocF32(nt * nt, -std.math.inf(f32));
            defer gpa.free(sp_mask);
            {
                var off: usize = 0;
                for (sp_slens.items) |s| {
                    for (0..usz(s)) |a| {
                        for (0..usz(s)) |b| sp_mask[(off + a) * nt + (off + b)] = 0.0;
                    }
                    off += usz(s);
                }
            }
            st.setInputF32("muse_glimmer_sp_mask", sp_mask);

            // pixel-shuffle gather (original order): f*f spatial neighbours grouped
            var dsp: std.ArrayList(i32) = .empty;
            defer dsp.deinit(gpa);
            try dsp.ensureTotalCapacity(gpa, nt);
            var oy: c_int = 0;
            while (oy < @divTrunc(grid_h, f)) : (oy += 1) {
                var ox: c_int = 0;
                while (ox < @divTrunc(grid_w, f)) : (ox += 1) {
                    var ry: c_int = 0;
                    while (ry < f) : (ry += 1) {
                        var rx: c_int = 0;
                        while (rx < f) : (rx += 1) try dsp.append(gpa, (oy * f + ry) * grid_w + (ox * f + rx));
                    }
                }
            }
            st.setInputI32("muse_glimmer_ds_perm", dsp.items);
        },
        .minicpmv => {
            // inspired from siglip
            const positions = try minicpmvPositions(pos_h, pos_w);
            defer gpa.free(positions);
            st.setInputI32("positions", positions);

            // inputs for resampler projector
            // set the 2D positions (using float for sinusoidal embedding)
            const n_patches_per_col = @divTrunc(image_size_width, patch_size);
            const pos_data = try allocF32(usz(n_pos), 0);
            defer gpa.free(pos_data);
            // dimension H
            for (pos_data, 0..) |*p, i| p.* = @floatFromInt(@divTrunc(@as(c_int, @intCast(i)), n_patches_per_col));
            st.setInputF32("pos_h", pos_data);
            // dimension W
            for (pos_data, 0..) |*p, i| p.* = @floatFromInt(@rem(@as(c_int, @intCast(i)), n_patches_per_col));
            st.setInputF32("pos_w", pos_data);
            // base frequency omega
            const base_freq: f32 = 10000.0;
            const n_embd_proj = clip.nMmprojEmbd(ctx);
            const q = @divTrunc(n_embd_proj, 4);
            const omega = try allocF32(usz(q), 0);
            defer gpa.free(omega);
            for (omega, 0..) |*o, i| o.* = 1.0 / powf(base_freq, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(q)));
            st.setInputF32("omega", omega);
        },
        .minicpmv4_6 => {
            const is_4x = hparams.n_merge == 2;

            // SigLIP position buckets (same as resampler path)
            const positions = try minicpmvPositions(pos_h, pos_w);
            defer gpa.free(positions);
            st.setInputI32("positions", positions);

            const half_h = @divTrunc(pos_h, 2);
            const half_w = @divTrunc(pos_w, 2);

            if (!is_4x) {
                // window reorder indices for 2x2 windows
                const window_idx = try allocI32(usz(n_pos), 0);
                defer gpa.free(window_idx);
                const inv_window_idx = try allocI32(usz(n_pos), 0);
                defer gpa.free(inv_window_idx);
                {
                    var k: usize = 0;
                    var wi: c_int = 0;
                    while (wi < half_h) : (wi += 1) {
                        var wj: c_int = 0;
                        while (wj < half_w) : (wj += 1) {
                            window_idx[k] = (2 * wi) * pos_w + (2 * wj);
                            window_idx[k + 1] = (2 * wi) * pos_w + (2 * wj + 1);
                            window_idx[k + 2] = (2 * wi + 1) * pos_w + (2 * wj);
                            window_idx[k + 3] = (2 * wi + 1) * pos_w + (2 * wj + 1);
                            k += 4;
                        }
                    }
                    for (0..usz(n_pos)) |i| inv_window_idx[usz(window_idx[i])] = @intCast(i);
                }
                st.setInputI32("vit_merger_window_idx", window_idx);
                st.setInputI32("vit_merger_inv_window_idx", inv_window_idx);

                // block-diagonal attention mask over the 4-token windows
                const np = usz(n_pos);
                const window_mask_data = try allocF32(np * np, lowest);
                defer gpa.free(window_mask_data);
                for (0..np / 4) |wi| {
                    for (0..4) |i| {
                        for (0..4) |j| window_mask_data[(wi * 4 + i) * np + (wi * 4 + j)] = 0.0;
                    }
                }
                st.setInputF32("vit_merger_window_mask", window_mask_data);

                // ViT merger 2x2 downsample indices
                const names_v = [_][*:0]const u8{ "vit_merger_ds_idx_0", "vit_merger_ds_idx_1", "vit_merger_ds_idx_2", "vit_merger_ds_idx_3" };
                const offs = [_][2]c_int{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
                for (names_v, offs) |nm, o| {
                    const idx = try makeDsIdx(o[0], o[1], half_h, half_w, pos_w);
                    defer gpa.free(idx);
                    st.setInputI32(nm, idx);
                }
            }

            const merger_h = if (is_4x) pos_h else half_h;
            const merger_w = if (is_4x) pos_w else half_w;
            const names_m = [_][*:0]const u8{ "merger_ds_idx_0", "merger_ds_idx_1", "merger_ds_idx_2", "merger_ds_idx_3" };
            const offs = [_][2]c_int{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
            for (names_m, offs) |nm, o| {
                const idx = try makeDsIdx(o[0], o[1], @divTrunc(merger_h, 2), @divTrunc(merger_w, 2), merger_w);
                defer gpa.free(idx);
                st.setInputI32(nm, idx);
            }
        },
        .qwen2vl, .qwen3vl, .glm4v => {
            const merge_ratio = hparams.n_merge;
            const pw = @divTrunc(image_size_width, patch_size);
            const ph = @divTrunc(image_size_height, patch_size);
            const positions = try allocI32(usz(n_pos * 4), 0);
            defer gpa.free(positions);
            const np = usz(num_patches);
            var ptr: usize = 0;
            var y: c_int = 0;
            while (y < ph) : (y += merge_ratio) {
                var x: c_int = 0;
                while (x < pw) : (x += merge_ratio) {
                    for (0..2) |dy| {
                        for (0..2) |dx| {
                            const yy = y + @as(c_int, @intCast(dy));
                            const xx = x + @as(c_int, @intCast(dx));
                            positions[ptr] = yy;
                            positions[np + ptr] = xx;
                            positions[2 * np + ptr] = yy;
                            positions[3 * np + ptr] = xx;
                            ptr += 1;
                        }
                    }
                }
            }
            st.setInputI32("positions", positions);
        },
        .step3vl => {
            const pos_data = try allocI32(usz(n_pos), 0);
            defer gpa.free(pos_data);
            for (pos_data, 0..) |*p, i| p.* = @divTrunc(@as(c_int, @intCast(i)), pos_w);
            st.setInputI32("pos_h", pos_data);
            for (pos_data, 0..) |*p, i| p.* = @rem(@as(c_int, @intCast(i)), pos_w);
            st.setInputI32("pos_w", pos_data);
        },
        .paddleocr => {
            const merge_ratio = hparams.n_merge;
            const pw = @divTrunc(image_size_width, patch_size);
            const ph = @divTrunc(image_size_height, patch_size);
            const positions = try allocI32(usz(n_pos * 4), 0);
            defer gpa.free(positions);
            const np = usz(num_patches);
            var ptr: usize = 0;
            // NOTE: same as Qwen-VL, but x and y are swapped
            var y: c_int = 0;
            while (y < ph) : (y += merge_ratio) {
                for (0..2) |dy| {
                    var x: c_int = 0;
                    while (x < pw) : (x += merge_ratio) {
                        for (0..2) |dx| {
                            const yy = y + @as(c_int, @intCast(dy));
                            const xx = x + @as(c_int, @intCast(dx));
                            positions[ptr] = yy;
                            positions[np + ptr] = xx;
                            positions[2 * np + ptr] = yy;
                            positions[3 * np + ptr] = xx;
                            ptr += 1;
                        }
                    }
                }
            }
            st.setInputI32("positions", positions);
        },
        .minimax_m3 => {
            const n_merge = hparams.n_merge;
            const gh = @divTrunc(image_size_height, patch_size);
            const gw = @divTrunc(image_size_width, patch_size);
            var ph_list: std.ArrayList(i32) = .empty;
            defer ph_list.deinit(gpa);
            var pw_list: std.ArrayList(i32) = .empty;
            defer pw_list.deinit(gpa);
            try ph_list.ensureTotalCapacity(gpa, usz(@max(gh * gw, 0)));
            try pw_list.ensureTotalCapacity(gpa, usz(@max(gh * gw, 0)));
            var bh: c_int = 0;
            while (bh < @divTrunc(gh, n_merge)) : (bh += 1) {
                var bw: c_int = 0;
                while (bw < @divTrunc(gw, n_merge)) : (bw += 1) {
                    var mh: c_int = 0;
                    while (mh < n_merge) : (mh += 1) {
                        var mw: c_int = 0;
                        while (mw < n_merge) : (mw += 1) {
                            try ph_list.append(gpa, bh * n_merge + mh);
                            try pw_list.append(gpa, bw * n_merge + mw);
                        }
                    }
                }
            }
            st.setInputI32("minimax_pos_h", ph_list.items);
            st.setInputI32("minimax_pos_w", pw_list.items);
        },
        .dots_ocr, .dots3note_v => {
            const pw = @divTrunc(image_size_width, patch_size);
            const ph = @divTrunc(image_size_height, patch_size);
            const np = usz(ph * pw);
            const positions = try allocI32(np * 4, 0);
            defer gpa.free(positions);
            var ptr: usize = 0;
            // flat layout: [h, w, h, w] for each patch, raster order (matching conv2d output)
            var y: c_int = 0;
            while (y < ph) : (y += 1) {
                var x: c_int = 0;
                while (x < pw) : (x += 1) {
                    positions[ptr] = y;
                    positions[np + ptr] = x;
                    positions[2 * np + ptr] = y;
                    positions[3 * np + ptr] = x;
                    ptr += 1;
                }
            }
            st.setInputI32("positions", positions);
        },
        .qwen25vl, .exaone4_5, .youtuvl => {
            // pw * ph = number of tokens output by ViT after apply patch merger
            // ipw * ipw = number of vision token been processed inside ViT
            const use_window_attn = if (model.proj_type == .qwen25vl or model.proj_type == .exaone4_5)
                hparams.n_wa_pattern > 0
            else
                hparams.wa_layer_indexes.count() != 0;
            const merge_ratio: c_int = 2;
            const pw = @divTrunc(@divTrunc(image_size_width, patch_size), merge_ratio);
            const ph = @divTrunc(@divTrunc(image_size_height, patch_size), merge_ratio);
            const ipw = @divTrunc(image_size_width, patch_size);
            const iph = @divTrunc(image_size_height, patch_size);

            const idx = try allocI32(usz(ph * pw), 0);
            defer gpa.free(idx);
            const inv_idx = try allocI32(usz(ph * pw), 0);
            defer gpa.free(inv_idx);

            if (use_window_attn) {
                const attn_window_size = if (hparams.attn_window_size > 0) hparams.attn_window_size else 112;
                const grid_window = @divTrunc(@divTrunc(attn_window_size, patch_size), merge_ratio);
                var dst: c_int = 0;
                // [num_vision_tokens, num_vision_tokens] attention mask tensor
                const nv = usz(ipw * iph);
                const mask = try allocF32(nv * nv, lowest);
                defer gpa.free(mask);
                var mask_row: usize = 0;

                var y: c_int = 0;
                while (y < ph) : (y += grid_window) {
                    var x: c_int = 0;
                    while (x < pw) : (x += grid_window) {
                        const win_h = @min(grid_window, ph - y);
                        const win_w = @min(grid_window, pw - x);
                        const dst_0 = dst;
                        // group all tokens belong to the same window togather (to a continue range)
                        var dy: c_int = 0;
                        while (dy < win_h) : (dy += 1) {
                            var dx: c_int = 0;
                            while (dx < win_w) : (dx += 1) {
                                const src = (y + dy) * pw + (x + dx);
                                impl.ggmlAssert(src < @as(c_int, @intCast(idx.len)), "clip.cpp", 4871, "src < (int)idx.size()");
                                impl.ggmlAssert(dst < @as(c_int, @intCast(inv_idx.len)), "clip.cpp", 4872, "dst < (int)inv_idx.size()");
                                idx[usz(src)] = dst;
                                inv_idx[usz(dst)] = src;
                                dst += 1;
                            }
                        }

                        var r: c_int = 0;
                        while (r < win_h * win_w * merge_ratio * merge_ratio) : (r += 1) {
                            const row_offset = mask_row * nv;
                            const lo = row_offset + usz(dst_0 * merge_ratio * merge_ratio);
                            const hi = row_offset + usz(dst * merge_ratio * merge_ratio);
                            @memset(mask[lo..hi], 0.0);
                            mask_row += 1;
                        }
                    }
                }

                st.setInputI32("window_idx", idx);
                st.setInputI32("inv_window_idx", inv_idx);
                st.setInputF32("window_mask", mask);
            } else {
                for (idx, 0..) |*v, i| v.* = @intCast(i);
            }

            const mpow: usize = usz(merge_ratio * merge_ratio);
            const positions = try allocI32(usz(n_pos * 4), 0);
            defer gpa.free(positions);
            const np = usz(num_patches);

            var ptr: usize = 0;
            var y: c_int = 0;
            while (y < iph) : (y += merge_ratio) {
                var x: c_int = 0;
                while (x < ipw) : (x += merge_ratio) {
                    for (0..2) |dy| {
                        for (0..2) |dx| {
                            var remap: usize = usz(idx[ptr / mpow]);
                            remap = (remap * mpow) + (ptr % mpow);
                            const yy = y + @as(c_int, @intCast(dy));
                            const xx = x + @as(c_int, @intCast(dx));
                            positions[remap] = yy;
                            positions[np + remap] = xx;
                            positions[2 * np + remap] = yy;
                            positions[3 * np + remap] = xx;
                            ptr += 1;
                        }
                    }
                }
            }
            st.setInputI32("positions", positions);
        },
        .mimovl => {
            const merge = hparams.n_merge; // 2
            const merge_unit = merge * merge; // 4
            const patch = hparams.patch_size; // 16
            const H = @divTrunc(image_size_height, patch);
            const W = @divTrunc(image_size_width, patch);
            const n_pos_full = H * W;
            const llm_h = @divTrunc(H, merge);
            const llm_w = @divTrunc(W, merge);
            const n_units = llm_h * llm_w; // n_pos / merge_unit
            const npf = usz(n_pos_full);
            const mu = usz(merge_unit);

            // Row-major merge-tile-ordered (h, w) positions
            const pos_h_row = try allocI32(npf, 0);
            defer gpa.free(pos_h_row);
            const pos_w_row = try allocI32(npf, 0);
            defer gpa.free(pos_w_row);
            {
                var idx: usize = 0;
                var ty: c_int = 0;
                while (ty < llm_h) : (ty += 1) {
                    var tx: c_int = 0;
                    while (tx < llm_w) : (tx += 1) {
                        var dy: c_int = 0;
                        while (dy < merge) : (dy += 1) {
                            var dx: c_int = 0;
                            while (dx < merge) : (dx += 1) {
                                pos_h_row[idx] = ty * merge + dy;
                                pos_w_row[idx] = tx * merge + dx;
                                idx += 1;
                            }
                        }
                    }
                }
            }

            // Col-major merge-unit permutation
            const idx_col = try allocF32(usz(n_units), 0);
            defer gpa.free(idx_col);
            var r: c_int = 0;
            while (r < llm_h) : (r += 1) {
                var cc: c_int = 0;
                while (cc < llm_w) : (cc += 1) {
                    const u_row = r * llm_w + cc;
                    const u_col = cc * llm_h + r;
                    idx_col[usz(u_col)] = @floatFromInt(u_row);
                }
            }

            // Col-mode positions: permute pos_*_row by idx_col
            const pos_h_col = try allocI32(npf, 0);
            defer gpa.free(pos_h_col);
            const pos_w_col = try allocI32(npf, 0);
            defer gpa.free(pos_w_col);
            for (0..usz(n_units)) |u| {
                const src: usize = usz(@as(c_int, @intFromFloat(idx_col[u])));
                for (0..mu) |k| {
                    pos_h_col[u * mu + k] = pos_h_row[src * mu + k];
                    pos_w_col[u * mu + k] = pos_w_row[src * mu + k];
                }
            }

            // Pack into ggml_rope_multi VISION-mode layout: h in slot 0, w in slot 1
            const positions_row = try allocI32(npf * 4, 0);
            defer gpa.free(positions_row);
            const positions_col = try allocI32(npf * 4, 0);
            defer gpa.free(positions_col);
            for (0..npf) |i| {
                positions_row[0 * npf + i] = pos_h_row[i];
                positions_row[1 * npf + i] = pos_w_row[i];
                positions_col[0 * npf + i] = pos_h_col[i];
                positions_col[1 * npf + i] = pos_w_col[i];
            }

            // Banded 1D sliding-window mask
            const window = hparams.attn_window_size;
            impl.ggmlAssert(window > 0, "clip.cpp", 4990, "window > 0");
            const mask = try allocF32(npf * npf, lowest);
            defer gpa.free(mask);
            var q: c_int = 0;
            while (q < n_pos_full) : (q += 1) {
                const lo = @max(0, q - window);
                const hi = @min(n_pos_full - 1, q + window);
                var k = lo;
                while (k <= hi) : (k += 1) mask[usz(q) * npf + usz(k)] = 0.0;
            }

            st.setInputI32("mimovl_positions_row", positions_row);
            st.setInputI32("mimovl_positions_col", positions_col);
            st.setInputF32("mimovl_idx_col", idx_col);
            st.setInputF32("mimovl_window_mask", mask);
        },
        .pixtral, .kimivl, .kimik25, .lightonocr => {
            // set the 2D positions
            const n_patches_per_col = @divTrunc(image_size_width, patch_size);
            const pos_data = try allocI32(usz(n_pos), 0);
            defer gpa.free(pos_data);
            // dimension H
            for (pos_data, 0..) |*p, i| p.* = @divTrunc(@as(c_int, @intCast(i)), n_patches_per_col);
            st.setInputI32("pos_h", pos_data);
            // dimension W
            for (pos_data, 0..) |*p, i| p.* = @rem(@as(c_int, @intCast(i)), n_patches_per_col);
            st.setInputI32("pos_w", pos_data);
        },
        .glm_edge => {
            // llava and other models
            const positions = try allocI32(usz(n_pos), 0);
            defer gpa.free(positions);
            for (positions, 0..) |*p, i| p.* = @intCast(i);
            st.setInputI32("positions", positions);
        },
        .mlp, .mlp_norm, .ldp, .ldpv2 => {
            // llava and other models
            const positions = try allocI32(usz(n_pos), 0);
            defer gpa.free(positions);
            for (positions, 0..) |*p, i| p.* = @intCast(i);
            st.setInputI32("positions", positions);

            // skip dim 0 only if we have CLS to avoid going out of bounds
            const patch_offset: i32 = if (model.class_embedding != null) 1 else 0;
            const patches = try allocI32(usz(num_patches), 0);
            defer gpa.free(patches);
            for (patches, 0..) |*p, i| p.* = @as(i32, @intCast(i)) + patch_offset;
            st.setInputI32("patches", patches);
        },
        .pockettts_spkenc => {
            try st.setPocketttsTfmInputs();
        },
        .pockettts_gen => {
            if (params.gen_process == .gen_wav) {
                impl.ggmlAssert(params.feats != null, "clip.cpp", 5062, "params->feats != nullptr");
                st.setInputF32("inp_feats", params.feats.?.items);
                // positions and mask are derived in-graph from the persisted counter
                try st.setGenStateIn();
            } else {
                // flow matching starts from gaussian noise, std = sqrt(temp)
                const t = st.getInpTensor("inp_noise");
                // Config.default_temperature, for a caller that does not set one
                const temp: f32 = if (params.temp > 0.0) params.temp else 0.7;
                var dist: NormalF32 = .{ .mean = 0.0, .stddev = sqrtf(temp) };
                const noise = try allocF32(usz(c.ggml_nelements(t)), 0);
                defer gpa.free(noise);
                for (noise) |*v| v.* = dist.next(&ctx.rng);
                st.setInputF32("inp_noise", noise);
            }
        },
        .gemma4v, .gemma4uv => {
            // set (col, row) patch positions for learned positional embedding
            const n_cols = @divTrunc(image_size_width, patch_size);
            const pos_x = try allocI32(usz(num_patches), 0);
            defer gpa.free(pos_x);
            const pos_y = try allocI32(usz(num_patches), 0);
            defer gpa.free(pos_y);
            for (0..usz(num_patches)) |i| {
                const ii: c_int = @intCast(i);
                pos_x[i] = @rem(ii, n_cols);
                pos_y[i] = @divTrunc(ii, n_cols);
            }
            st.setInputI32("pos_x", pos_x);
            st.setInputI32("pos_y", pos_y);
        },
        .deepseekocr, .deepseekocr2 => {
            impl.ggmlAssert((pos_w == pos_h) or (pos_h >= pos_w and @rem(pos_h, pos_w) == 0), "clip.cpp", 5096, "(pos_w == pos_h) || (pos_h >= pos_w && pos_h % pos_w == 0)");

            const window = hparams.attn_window_size;
            const pos = pos_w;
            const rel_pos_indices_local = try allocI32(usz(window * window), 0);
            defer gpa.free(rel_pos_indices_local);
            const rel_pos_indices_global = try allocI32(usz(pos * pos), 0);
            defer gpa.free(rel_pos_indices_global);

            var q: c_int = 0;
            while (q < window) : (q += 1) {
                var k: c_int = 0;
                while (k < window) : (k += 1) rel_pos_indices_local[usz(q * window + k)] = q - k + window - 1;
            }
            q = 0;
            while (q < pos) : (q += 1) {
                var k: c_int = 0;
                while (k < pos) : (k += 1) rel_pos_indices_global[usz(q * pos + k)] = q - k + pos - 1;
            }

            st.setInputI32("rel_pos_indices_local", rel_pos_indices_local);
            st.setInputI32("rel_pos_indices_global", rel_pos_indices_global);

            if (ctx.projType() == .deepseekocr2) {
                // qwen2 encoder attention mask
                const num_image_tokens = @divTrunc(num_patches, 16);
                const seq_len = num_image_tokens * 2;
                const sl = usz(seq_len);
                const qwen2_mask = try allocF32(sl * sl, 0.0);
                defer gpa.free(qwen2_mask);
                var i: c_int = 0;
                while (i < seq_len) : (i += 1) {
                    var j: c_int = 0;
                    while (j < seq_len) : (j += 1) {
                        const zero = if (i < num_image_tokens) j < num_image_tokens else (j < num_image_tokens or j <= i);
                        qwen2_mask[usz(i) * sl + usz(j)] = if (zero) 0.0 else -1e9;
                    }
                }
                st.setInputF32("qwen2_attn_mask", qwen2_mask);
            }
        },
        .gemma3, .gemma3nv, .idefics3, .internvl, .nemotron_v2_vl, .qwen2a, .qwen3a, .glma, .ultravox, .lfm2, .voxtral, .meralion, .music_flamingo, .janus_pro, .phi4, .cogvlm, .yasa2, .gemma4ua, .qwen3tts_spkenc => {
            // do nothing
        },
        .qwen3tts_gen => {
            if (params.gen_process == .gen_wav) {
                impl.ggmlAssert(params.codes != null, "clip.cpp", 5173, "params->codes != nullptr");
                const codes_in = params.codes.?.items;

                // frame-major input to group-major, rear-padded with code 0 up to one window
                const n_codes: i64 = graph.ne(model.gen_code_head_w, 2) + 1;
                const n_frames_w: i64 = hparams.wav_tfm_swa;
                const n_frames: i64 = @divTrunc(@as(i64, @intCast(codes_in.len)), n_codes);
                impl.ggmlAssert(n_frames > 0 and n_frames <= n_frames_w, "clip.cpp", 5179, "n_frames > 0 && n_frames <= n_frames_w");

                // codes are used as ggml_get_rows indices, so check them against the codebook vocab
                const vocab_first = graph.ne(model.c2w.quant_first_cb_w, 1);
                const vocab_rest = graph.ne(model.c2w.quant_rest_cb_w, 1);
                var f: i64 = 0;
                while (f < n_frames) : (f += 1) {
                    var g: i64 = 0;
                    while (g < n_codes) : (g += 1) {
                        const cd = codes_in[usz(f * n_codes + g)];
                        const vocab = if (g == 0) vocab_first else vocab_rest;
                        if (cd < 0 or @as(i64, cd) >= vocab) {
                            impl.logErr("%s: code out of range (frame %lld, group %lld, code %d, vocab %lld)\n", .{ func, @as(c_longlong, f), @as(c_longlong, g), cd, @as(c_longlong, vocab) });
                            return false;
                        }
                    }
                }

                const codes = try allocI32(usz(n_frames_w * n_codes), 0);
                defer gpa.free(codes);
                f = 0;
                while (f < n_frames) : (f += 1) {
                    var g: i64 = 0;
                    while (g < n_codes) : (g += 1) codes[usz(g * n_frames_w + f)] = codes_in[usz(f * n_codes + g)];
                }
                st.setInputI32("inp_codes", codes);
                try st.setGenStateIn();
            } else {
                // code0 indexes gen_code_out_embd_w via ggml_get_rows; bound it
                const vocab0 = graph.ne(model.gen_code_out_embd_w, 1);
                if (params.code0 < 0 or @as(i64, params.code0) >= vocab0) {
                    impl.logErr("%s: code0 out of range (%d, vocab %lld)\n", .{ func, params.code0, @as(c_longlong, vocab0) });
                    return false;
                }
                const code0 = [_]i32{params.code0};
                st.setInputI32("inp_code0", &code0);

                // one uniform(0,1) draw per codebook, used by do_sampling()
                const n_acoustic = graph.ne(model.gen_code_head_w, 2);
                var g: i64 = 0;
                while (g < n_acoustic) : (g += 1) {
                    const r = [_]f32{uniformRealF32(&ctx.rng, 0.0, 1.0)};
                    var nb: [64]u8 = undefined;
                    _ = snprintf(&nb, nb.len, "inp_rand_%lld", @as(c_longlong, g));
                    st.setInputF32(@ptrCast(&nb), &r);
                }
            }
        },
        .hunyuanvl => {
            // the 2D position embedding with the reference's sf=(target+0.1)/n_grid
            // bilinear sampling, computed on the CPU
            impl.ggmlAssert(model.position_embeddings != null, "clip.cpp", 5229, "model.position_embeddings != nullptr");
            const src_t = model.position_embeddings;
            const n_embd: i64 = graph.ne(src_t, 0);
            const n_pos_e: i64 = graph.ne(src_t, 1); // = n_grid * n_grid
            const n_grid: c_int = @intCast(lround(sqrt(@floatFromInt(n_pos_e))));
            impl.ggmlAssert(@as(i64, n_grid) * n_grid == n_pos_e, "clip.cpp", 5234, "(int64_t)n_grid * n_grid == n_pos");
            const out_w = pos_w; // pw
            const out_h = pos_h; // ph
            const ne0 = usz(n_embd);

            // Pull weight to host.
            const src = try allocF32(usz(n_embd * n_pos_e), 0);
            defer gpa.free(src);
            c.ggml_backend_tensor_get(src_t, src.ptr, 0, c.ggml_nbytes(src_t));

            // dst[(y*out_w + x) * n_embd + c]
            const dst = try allocF32(ne0 * usz(out_h) * usz(out_w), 0);
            defer gpa.free(dst);

            const sx: f32 = (@as(f32, @floatFromInt(out_w)) + 0.1) / @as(f32, @floatFromInt(n_grid));
            const sy: f32 = (@as(f32, @floatFromInt(out_h)) + 0.1) / @as(f32, @floatFromInt(n_grid));

            var y: c_int = 0;
            while (y < out_h) : (y += 1) {
                // pixel-center convention (align_corners=False)
                const fy = (@as(f32, @floatFromInt(y)) + 0.5) / sy - 0.5;
                var y0: c_int = @intFromFloat(@floor(fy));
                var y1 = y0 + 1;
                y0 = std.math.clamp(y0, 0, n_grid - 1);
                y1 = std.math.clamp(y1, 0, n_grid - 1);
                const wy1 = std.math.clamp(fy - @as(f32, @floatFromInt(y0)), 0.0, 1.0);
                const wy0 = 1.0 - wy1;
                var x: c_int = 0;
                while (x < out_w) : (x += 1) {
                    const fx = (@as(f32, @floatFromInt(x)) + 0.5) / sx - 0.5;
                    var x0: c_int = @intFromFloat(@floor(fx));
                    var x1 = x0 + 1;
                    x0 = std.math.clamp(x0, 0, n_grid - 1);
                    x1 = std.math.clamp(x1, 0, n_grid - 1);
                    const wx1 = std.math.clamp(fx - @as(f32, @floatFromInt(x0)), 0.0, 1.0);
                    const wx0 = 1.0 - wx1;

                    const w00 = wy0 * wx0;
                    const w01 = wy0 * wx1;
                    const w10 = wy1 * wx0;
                    const w11 = wy1 * wx1;

                    const ng = usz(n_grid);
                    const s00 = src[(usz(y0) * ng + usz(x0)) * ne0 ..];
                    const s01 = src[(usz(y0) * ng + usz(x1)) * ne0 ..];
                    const s10 = src[(usz(y1) * ng + usz(x0)) * ne0 ..];
                    const s11 = src[(usz(y1) * ng + usz(x1)) * ne0 ..];
                    const d = dst[(usz(y) * usz(out_w) + usz(x)) * ne0 ..];
                    for (0..ne0) |ch| {
                        // ((w00*s00 + w01*s01) + w10*s10) + w11*s11, contracted left to right
                        const t1 = @mulAdd(f32, w00, s00[ch], w01 * s01[ch]);
                        const t2 = @mulAdd(f32, w10, s10[ch], t1);
                        d[ch] = @mulAdd(f32, w11, s11[ch], t2);
                    }
                }
            }

            st.setInputF32("hunyuanvl_pos_embd", dst);
        },
        .llama4 => {
            // set the 2D positions
            const n_patches_per_col = @divTrunc(image_size_width, patch_size);
            const pos_data = try allocI32(usz(num_patches + 1), 0); // +1 for the [CLS] token
            defer gpa.free(pos_data);
            // last pos is always kept 0, it's for CLS
            // dimension H
            for (0..usz(num_patches)) |i| pos_data[i] = @divTrunc(@as(c_int, @intCast(i)), n_patches_per_col) + 1;
            st.setInputI32("pos_h", pos_data);
            // dimension W
            for (0..usz(num_patches)) |i| pos_data[i] = @rem(@as(c_int, @intCast(i)), n_patches_per_col) + 1;
            st.setInputI32("pos_w", pos_data);
        },
        .dots3note_a => {
            impl.ggmlAssert(imgs.entries.items.len == 1, "clip.cpp", 5306, "imgs.entries.size() == 1");
            const np = @divTrunc(imgs.entries.items[0].nx() + 7, 8); // 3x stride-2 conv2d
            const positions = try allocI32(usz(np), 0);
            defer gpa.free(positions);
            for (positions, 0..) |*p, i| p.* = @intCast(i);
            st.setInputI32("positions", positions);
        },
        .gemma4a => {
            impl.ggmlAssert(imgs.entries.items.len == 1, "clip.cpp", 5316, "imgs.entries.size() == 1");
            const img0 = &imgs.entries.items[0];
            // Compute n_pos matching SSCP output: two stride-2 convs
            var np = img0.nx();
            for (0..2) |_| np = @divTrunc(np - 1, 2) + 1;

            // Chunked local attention: blocked causal mask and RPE
            const chunk_size: c_int = 12;
            const max_past: c_int = 12;
            const context_size = chunk_size + max_past;
            const num_blocks = @divTrunc(np + chunk_size - 1, chunk_size);

            // Blocked causal attention mask: [context_size, chunk_size, num_blocks]
            {
                const mask = try allocF32(usz(context_size * chunk_size * num_blocks), -1e9);
                defer gpa.free(mask);
                var b: c_int = 0;
                while (b < num_blocks) : (b += 1) {
                    var q: c_int = 0;
                    while (q < chunk_size) : (q += 1) {
                        const gq = b * chunk_size + q;
                        var k: c_int = 0;
                        while (k < context_size) : (k += 1) {
                            const gk = b * chunk_size - max_past + k;
                            if (gq < np and gk >= 0 and gk < np and gk <= gq and (gq - gk) < max_past) {
                                mask[usz(k + q * context_size + b * context_size * chunk_size)] = 0.0;
                            }
                        }
                    }
                }
                st.setInputF32("kq_mask", mask);
            }

            // Sinusoidal RPE: 13 positions [12, 11, ..., 0]
            {
                const n_embd = ctx.model.hparams.n_embd;
                const num_timescales = @divTrunc(n_embd, 2);
                const log_timescale_increment = logf(10000.0) / @as(f32, @floatFromInt(@max(num_timescales - 1, 1)));
                const rpe_len = max_past + 1;
                const pos_emb = try allocF32(usz(n_embd * rpe_len), 0.0);
                defer gpa.free(pos_emb);
                var p: c_int = 0;
                while (p < rpe_len) : (p += 1) {
                    const position: f32 = @floatFromInt(max_past - p);
                    var i: c_int = 0;
                    while (i < num_timescales) : (i += 1) {
                        const inv_ts = expf(-@as(f32, @floatFromInt(i)) * log_timescale_increment);
                        const scaled = position * inv_ts;
                        const sc = sinCos(scaled);
                        pos_emb[usz(p * n_embd + i)] = sc.sin;
                        pos_emb[usz(p * n_embd + i + num_timescales)] = sc.cos;
                    }
                }
                st.setInputF32("pos_emb", pos_emb);
            }
        },
        .mimo_audio => {
            impl.ggmlAssert(imgs.entries.items.len == 1, "clip.cpp", 5366, "imgs.entries.size() == 1");
            const n_frames = imgs.entries.items[0].nx();
            const np = @divTrunc(n_frames - 1, 2) + 1; // matches conv1(s=1)+conv2(s=2) output length
            const npu = usz(np);

            const positions = try allocI32(npu, 0);
            defer gpa.free(positions);
            for (positions, 0..) |*p, i| p.* = @intCast(i);
            st.setInputI32("mimo_audio_positions", positions);

            const window = hparams.attn_window_size;
            impl.ggmlAssert(window > 0, "clip.cpp", 5377, "window > 0");

            const neg_inf = lowest;
            const full_mask = try allocF32(npu * npu, 0);
            defer gpa.free(full_mask);
            const window_mask = try allocF32(npu * npu, 0);
            defer gpa.free(window_mask);
            var q: c_int = 0;
            while (q < np) : (q += 1) {
                var k: c_int = 0;
                while (k < np) : (k += 1) {
                    const causal_ok = k <= q;
                    full_mask[usz(q) * npu + usz(k)] = if (causal_ok) 0.0 else neg_inf;
                    window_mask[usz(q) * npu + usz(k)] = if (causal_ok and (q - k) <= window) 0.0 else neg_inf;
                }
            }
            st.setInputF32("mimo_audio_full_mask", full_mask);
            st.setInputF32("mimo_audio_window_mask", window_mask);

            // input_local_transformer: block-diagonal mask + in-group positions
            {
                const n_pos_ds = @divTrunc(np - 2, 2) + 1; // matches downsample conv (k=2,s=2,p=0)
                const group_size = hparams.audio_local_group_size;
                impl.ggmlAssert(group_size > 0, "clip.cpp", 5396, "group_size > 0");
                const n_groups = @divTrunc(n_pos_ds + group_size - 1, group_size);
                const n_padded = n_groups * group_size;
                const npd = usz(n_padded);

                const local_positions = try allocI32(npd, 0);
                defer gpa.free(local_positions);
                for (local_positions, 0..) |*p, i| p.* = @rem(@as(c_int, @intCast(i)), group_size);
                st.setInputI32("mimo_audio_local_positions", local_positions);

                const local_mask = try allocF32(npd * npd, 0);
                defer gpa.free(local_mask);
                var qq: c_int = 0;
                while (qq < n_padded) : (qq += 1) {
                    var k: c_int = 0;
                    while (k < n_padded) : (k += 1) {
                        const same_group = @divTrunc(qq, group_size) == @divTrunc(k, group_size);
                        local_mask[usz(qq) * npd + usz(k)] = if (same_group) 0.0 else neg_inf;
                    }
                }
                st.setInputF32("mimo_audio_local_mask", local_mask);
            }
        },
        .lfm2a => {
            impl.ggmlAssert(imgs.entries.items.len == 1, "clip.cpp", 5418, "imgs.entries.size() == 1");
            const n_frames = nOutputTokens(ctx, &imgs.entries.items[0]);

            const d_model: c_int = 512;
            const seq_len = n_frames * 2 - 1;
            const pos_emb = try allocF32(usz(d_model * seq_len), 0);
            defer gpa.free(pos_emb);
            const inv_freq = try gpa.alloc(f64, usz(@divTrunc(d_model, 2)));
            defer gpa.free(inv_freq);
            for (inv_freq, 0..) |*v, i| {
                // `(float)d_model` and `2.0f * (float)i` widen to double against `std::log(10000.0)`
                const two_i: f32 = 2.0 * @as(f32, @floatFromInt(i));
                v.* = exp(-(log(10000.0) / @as(f64, @floatFromInt(d_model))) * @as(f64, two_i));
            }
            var pos: i64 = 0;
            while (pos < seq_len) : (pos += 1) {
                for (inv_freq, 0..) |fq, i| {
                    const ang: f32 = @floatCast(@as(f64, @floatFromInt(@as(i64, n_frames) - pos - 1)) * fq);
                    const sc = sinCos(ang);
                    pos_emb[usz(pos * d_model) + 2 * i + 0] = sc.sin; // even
                    pos_emb[usz(pos * d_model) + 2 * i + 1] = sc.cos; // odd
                }
            }
            st.setInputF32("pos_emb", pos_emb);
        },
        .parakeet => {
            impl.ggmlAssert(imgs.entries.items.len == 1, "clip.cpp", 5439, "imgs.entries.size() == 1");
            const attn_mask = c.ggml_graph_get_tensor(gf, "attn_mask");
            const n_q: c_int = @intCast(attn_mask.*.ne[1]);
            const n_k: c_int = @intCast(attn_mask.*.ne[0]);
            const n_frames = imgs.entries.items[0].nx();
            const n_tokens_real = @divTrunc(n_frames + (hparams.subsampling_factor - 1), hparams.subsampling_factor);
            const mask_value: f32 = -1e30;

            const mask_data = try allocF32(usz(n_q * n_k), 0);
            defer gpa.free(mask_data);
            if (n_k == n_q) {
                // full attention: mask keys that are padding
                var q: c_int = 0;
                while (q < n_q) : (q += 1) {
                    var k: c_int = 0;
                    while (k < n_k) : (k += 1) mask_data[usz(q * n_k + k)] = if (k >= n_tokens_real) mask_value else 0.0;
                }
            } else {
                // local attention: mask keys outside the valid window
                const att_left = @divTrunc(n_k, 2);
                var q: c_int = 0;
                while (q < n_q) : (q += 1) {
                    var k: c_int = 0;
                    while (k < n_k) : (k += 1) {
                        const key = q - att_left + k;
                        mask_data[usz(q * n_k + k)] = if (key >= 0 and key < n_tokens_real) 0.0 else mask_value;
                    }
                }
            }
            st.setInputF32(@ptrCast(&attn_mask.*.name), mask_data);

            // local attention skew mask: zeroes out the probs that were
            // computed for keys outside the valid sliding window.
            const local_mask = c.ggml_graph_get_tensor(gf, "local_mask");
            if (local_mask != null) {
                const lm_k: c_int = @intCast(local_mask.*.ne[0]);
                const lm_q: c_int = @intCast(local_mask.*.ne[1]);
                const window_size = lm_k - lm_q + 1;
                const lm_data = try allocF32(usz(lm_q * lm_k), 0);
                defer gpa.free(lm_data);
                var q: c_int = 0;
                while (q < lm_q) : (q += 1) {
                    var k: c_int = 0;
                    while (k < lm_k) : (k += 1) {
                        const rel = k - q;
                        lm_data[usz(q * lm_k + k)] = if (rel >= 0 and rel < window_size) 1.0 else 0.0;
                    }
                }
                st.setInputF32(@ptrCast(&local_mask.*.name), lm_data);
            }

            // Generate rotation frequencies for relative positional encoding.
            {
                const n_state = hparams.n_embd;
                const d_half = @divTrunc(n_state, 2);
                const log_10000 = logf(10000.0);
                const freqs = try allocF32(usz(d_half), 0);
                defer gpa.free(freqs);
                for (freqs, 0..) |*fr, k| {
                    fr.* = expf(-(@as(f32, @floatFromInt(@as(c_int, @intCast(k)) * 2)) * log_10000 / @as(f32, @floatFromInt(n_state))));
                }
                st.setInputF32("pos_freqs", freqs);
            }

            // relative positional distances, scaled by the frequency into angles
            {
                // window_size is only known after graph construction: read it back
                const rel_pos = c.ggml_graph_get_tensor(gf, "rel_positions");
                const window_size: c_int = @intCast(rel_pos.*.ne[1]);
                const pos = try allocF32(usz(window_size), 0);
                defer gpa.free(pos);
                if (c.ggml_graph_get_tensor(gf, "local_mask") != null) {
                    const att_left = @divTrunc(window_size, 2);
                    for (pos, 0..) |*p, t| p.* = @floatFromInt(att_left - @as(c_int, @intCast(t)));
                } else {
                    const n_time = @divTrunc(window_size + 1, 2);
                    for (pos, 0..) |*p, t| p.* = @floatFromInt(n_time - 1 - @as(c_int, @intCast(t)));
                }
                st.setInputF32(@ptrCast(&rel_pos.*.name), pos);
            }
        },
        .granite_speech => {
            const context_size = ctx.model.hparams.audio_chunk_size;
            const max_pos_emb = ctx.model.hparams.audio_max_pos_emb;
            const cs = usz(context_size);

            const dists = try allocI32(cs * cs, 0);
            defer gpa.free(dists);
            var i: c_int = 0;
            while (i < context_size) : (i += 1) {
                var j: c_int = 0;
                while (j < context_size) : (j += 1) {
                    var d = i - j;
                    if (d < -context_size) d = -context_size;
                    if (d > context_size) d = context_size;
                    dists[usz(i) * cs + usz(j)] = d + max_pos_emb;
                }
            }
            st.setInputI32("attn_dists", dists);

            const n_frames = image_size_width;
            const remainder = @rem(n_frames, context_size);
            if (remainder > 0) {
                const num_blocks = @divTrunc(n_frames + context_size - 1, context_size);
                const mask = try allocF32(cs * cs * usz(num_blocks), 0.0);
                defer gpa.free(mask);
                const neg_inf = -std.math.inf(f32);
                const last_block_offset = usz(num_blocks - 1) * cs * cs;
                var q: c_int = 0;
                while (q < context_size) : (q += 1) {
                    var k: c_int = 0;
                    while (k < context_size) : (k += 1) {
                        if (q >= remainder or k >= remainder) mask[last_block_offset + usz(q) * cs + usz(k)] = neg_inf;
                    }
                }
                st.setInputF32("attn_mask", mask);
            }
        },
        .granite4_vision => {
            // precomputed permutation index tensors for the _win / _unwin /
            // spatial sampling reshapes, named by g4v_gather()
            const ps = model.hparams.patch_size;
            const image_side = @divTrunc(imgs.entries.items[0].nx(), ps);
            const window_side = hparams.downsample_window_side;
            const query_side = hparams.downsample_query_side;
            const n = @divTrunc(image_side, window_side);
            const new_side = n * query_side;

            // Stage 1b only uses block 0's permutations; future stages
            // will upload all blocks.
            for (0..hparams.feature_layers.items.len) |bid| {
                var nb: [96]u8 = undefined;
                {
                    const idx = try makeWinIdx(image_side, window_side);
                    defer gpa.free(idx);
                    upload(gf, g4vName(&nb, bid, "win_idx"), idx);
                }
                {
                    const idx = try makeWinIdx(new_side, query_side);
                    defer gpa.free(idx);
                    upload(gf, g4vName(&nb, bid, "qwin_idx"), idx);
                }
                {
                    const idx = try makeUnwinIdx(new_side, query_side);
                    defer gpa.free(idx);
                    upload(gf, g4vName(&nb, bid, "unwin_idx"), idx);
                }
                const spatial_offset = hparams.proj_spatial_offsets.items[bid];
                if (spatial_offset >= 0) {
                    const idx = try makeSpatialIdx(image_side, spatial_offset);
                    defer gpa.free(idx);
                    upload(gf, g4vName(&nb, bid, "spatial_idx"), idx);
                }
            }
        },
        else => impl.abort("clip.cpp", 5637, "Unknown projector type"),
    }
    return true;
}

/// `clip_encode`'s input switch for a caller that cannot see its
/// `return false`: a rejected input surfaces as `error.Runtime` with an
/// empty message, the reason already logged. `clip.encode` should call
/// `setInputsChecked` and return false instead.
pub fn setInputs(st: *clip.EncodeState) clip.Error!void {
    if (!try setInputsChecked(st)) return clip.throwStr("");
}

/// The SigLIP position buckets the two MiniCPM-V arms of `clip_encode` share:
/// `floor(70.0*i/n)` per axis, over the
/// C++'s two `int[1024]`. The caller frees the slice.
fn minicpmvPositions(pos_h: c_int, pos_w: c_int) clip.Error![]i32 {
    const positions = try allocI32(usz(pos_h * pos_w), 0);
    var bucket_coords_h: [1024]c_int = undefined;
    var bucket_coords_w: [1024]c_int = undefined;
    var i: c_int = 0;
    while (i < pos_h) : (i += 1) bucket_coords_h[usz(i)] = @intFromFloat(@floor(70.0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(pos_h))));
    i = 0;
    while (i < pos_w) : (i += 1) bucket_coords_w[usz(i)] = @intFromFloat(@floor(70.0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(pos_w))));
    var id: usize = 0;
    i = 0;
    while (i < pos_h) : (i += 1) {
        var j: c_int = 0;
        while (j < pos_w) : (j += 1) {
            positions[id] = bucket_coords_h[usz(i)] * 70 + bucket_coords_w[usz(j)];
            id += 1;
        }
    }
    return positions;
}

/// The `make_ds_idx` lambda of the MiniCPM-V 4.6 arm (mtmd/clip.cpp:4667
/// @c1d0e7a00). The caller frees the slice.
fn makeDsIdx(off_r: c_int, off_c: c_int, ds_h: c_int, ds_w: c_int, stride_w: c_int) clip.Error![]i32 {
    const idx = try allocI32(usz(@max(ds_h * ds_w, 0)), 0);
    var i: c_int = 0;
    while (i < ds_h) : (i += 1) {
        var j: c_int = 0;
        while (j < ds_w) : (j += 1) idx[usz(i * ds_w + j)] = (2 * i + off_r) * stride_w + (2 * j + off_c);
    }
    return idx;
}

/// The `make_win_idx` lambda of the Granite4 Vision arm (mtmd/clip.cpp:5568
/// @c1d0e7a00): raster to window order, `dst[w*win*win + p]` the source
/// index. The caller frees the slice.
fn makeWinIdx(side: c_int, win: c_int) clip.Error![]i32 {
    const nn = @divTrunc(side, win);
    const idx = try allocI32(usz(side) * usz(side), 0);
    var wy: c_int = 0;
    while (wy < nn) : (wy += 1) {
        var wx: c_int = 0;
        while (wx < nn) : (wx += 1) {
            var iy: c_int = 0;
            while (iy < win) : (iy += 1) {
                var ix: c_int = 0;
                while (ix < win) : (ix += 1) {
                    const w = wy * nn + wx;
                    const p = iy * win + ix;
                    const y = wy * win + iy;
                    const x = wx * win + ix;
                    idx[usz(w) * usz(win * win) + usz(p)] = y * side + x;
                }
            }
        }
    }
    return idx;
}

/// The `make_unwin_idx` lambda (mtmd/clip.cpp:5587 @c1d0e7a00).
fn makeUnwinIdx(side: c_int, win: c_int) clip.Error![]i32 {
    const fwd = try makeWinIdx(side, win);
    defer gpa.free(fwd);
    const inv = try allocI32(fwd.len, 0);
    for (fwd, 0..) |v, i| inv[usz(v)] = @intCast(i);
    return inv;
}

/// The `make_spatial_idx` lambda (mtmd/clip.cpp:5596 @c1d0e7a00).
fn makeSpatialIdx(side: c_int, offset: i32) clip.Error![]i32 {
    const off_y = (offset >> 1) & 1;
    const off_x = offset & 1;
    const new_s = @divTrunc(side, 2);
    const idx = try allocI32(usz(new_s) * usz(new_s), 0);
    var y: c_int = 0;
    while (y < new_s) : (y += 1) {
        var x: c_int = 0;
        while (x < new_s) : (x += 1) idx[usz(y * new_s + x)] = (y * 2 + off_y) * side + (x * 2 + off_x);
    }
    return idx;
}

/// `"g4v_blk" + std::to_string(bid) + "_" + suffix`.
fn g4vName(buf: *[96]u8, bid: usize, comptime suffix: [*:0]const u8) [*:0]const u8 {
    _ = snprintf(buf, buf.len, "g4v_blk%zu_%s", bid, suffix);
    return @ptrCast(buf);
}

/// The `upload` lambda (mtmd/clip.cpp:5610 @c1d0e7a00): the same permutation
/// for every tile of the stacked image.
fn upload(gf: ?*c.ggml_cgraph, name: [*:0]const u8, idx: []const i32) void {
    const t = c.ggml_graph_get_tensor(gf, name);
    impl.ggmlAssert(t != null, "clip.cpp", 5613, "t");
    impl.ggmlAssert(@rem(c.ggml_nelements(t), @as(i64, @intCast(idx.len))) == 0, "clip.cpp", 5613, "ggml_nelements(t) % (int64_t) idx.size() == 0");
    const n_rep: usize = @intCast(@as(c_int, @truncate(@as(i64, @intCast(@as(usize, @intCast(c.ggml_nelements(t))) / idx.len)))));
    const buf = gpa.alloc(i32, idx.len * n_rep) catch impl.abort("clip.cpp", 5621, "std::bad_alloc");
    defer gpa.free(buf);
    for (0..n_rep) |i| @memcpy(buf[i * idx.len ..][0..idx.len], idx);
    c.ggml_backend_tensor_set(t, buf.ptr, 0, c.ggml_nbytes(t));
}

/// Ports `clip_n_output_tokens` (mtmd/clip.cpp:4043 @c1d0e7a00): the number of
/// embeddings the projector produces for `img`.
pub fn nOutputTokens(ctx: *const clip.Ctx, img: *const impl.ImageF32) c_int {
    const params = &ctx.model.hparams;

    // for models with fixed size image, the input image is already pre-processed and resized to square
    const patch_size = params.patch_size;
    var n_patches = @divTrunc(img.nx(), patch_size) * @divTrunc(img.ny(), patch_size);

    switch (ctx.projType()) {
        .mlp, .mlp_norm, .janus_pro, .phi4 => {
            // do nothing
        },
        .yasa2 => {
            n_patches = 64; // adaptive average pooling to 8x8 tokens
        },
        .ldp, .ldpv2, .glm_edge => {
            n_patches = @divTrunc(n_patches, 4);
            if (ctx.model.mm_boi != null) n_patches += 2; // for BOI and EOI token embeddings
        },
        .minicpmv => {
            // Use actual config value if available, otherwise fall back to hardcoded values
            if (params.minicpmv_query_num > 0) {
                n_patches = params.minicpmv_query_num;
            } else {
                // Fallback to hardcoded values for legacy models
                n_patches = switch (params.minicpmv_version) {
                    2 => 96,
                    3, 4, 5, 6, 100045 => 64,
                    else => impl.abort("clip.cpp", 4096, "Unknown minicpmv version"),
                };
            }
        },
        .minicpmv4_6 => {
            n_patches = @divTrunc(n_patches, params.n_merge * params.n_merge);
        },
        .qwen2vl, .qwen25vl, .qwen3vl, .exaone4_5, .mimovl, .minimax_m3, .glm4v, .youtuvl, .muse_glimmer => {
            // dynamic size (2 conv, so double patch size)
            const x_patch = @divTrunc(img.nx(), params.patch_size * 2);
            const y_patch = @divTrunc(img.ny(), params.patch_size * 2);
            n_patches = x_patch * y_patch;
        },
        .step3vl => {
            const x_patch = @divTrunc(img.nx(), params.patch_size * params.n_merge);
            const y_patch = @divTrunc(img.ny(), params.patch_size * params.n_merge);
            n_patches = x_patch * y_patch;
        },
        .gemma3, .gemma4v, .gemma4uv, .idefics3, .internvl, .nemotron_v2_vl, .llama4 => {
            // both X and Y are downscaled by the scale factor
            const scale_factor = ctx.model.hparams.n_merge;
            n_patches = @divTrunc(n_patches, scale_factor * scale_factor);
        },
        .gemma3nv => {
            // MobileNetV5 MSFA adapter always outputs fixed 16x16 resolution
            n_patches = @divTrunc(ctx.model.hparams.image_size, ctx.model.hparams.patch_size);
        },
        .lfm2, .kimivl, .kimik25 => {
            // dynamic size
            const out_patch_size = params.patch_size * ctx.model.hparams.n_merge;
            const x_patch = @divTrunc(clipAlign(img.nx(), out_patch_size), out_patch_size);
            const y_patch = @divTrunc(clipAlign(img.ny(), out_patch_size), out_patch_size);
            n_patches = x_patch * y_patch;
        },
        .paddleocr, .dots_ocr, .dots3note_v => {
            // dynamic size
            const n_merge = ctx.model.hparams.n_merge;
            const stride = n_merge * n_merge;
            n_patches = @divTrunc(clipAlign(n_patches, stride), stride);
        },
        .dots3note_a => {
            // 3x stride-2 conv2d over mel frames
            n_patches = @divTrunc(img.nx() + 7, 8);
        },
        .pixtral, .lightonocr => {
            // dynamic size
            const n_merge = ctx.model.hparams.n_merge;
            const n_patches_x = @divTrunc(@divTrunc(img.nx(), patch_size), n_merge);
            const n_patches_y = @divTrunc(@divTrunc(img.ny(), patch_size), n_merge);
            if (ctx.model.token_embd_img_break != null) {
                n_patches = n_patches_y * n_patches_x + n_patches_y - 1; // + one [IMG_BREAK] per row, except the last row
            } else {
                n_patches = n_patches_y * n_patches_x;
            }
        },
        .voxtral, .ultravox, .qwen2a, .meralion, .music_flamingo => {
            n_patches = img.nx();

            const proj_stack_factor = ctx.model.hparams.proj_stack_factor;
            if (ctx.model.audioHasStackFrames()) {
                impl.ggmlAssert(proj_stack_factor > 0, "clip.cpp", 4190, "proj_stack_factor > 0");
                const n_len = clipAlign(n_patches, proj_stack_factor);
                n_patches = @divTrunc(n_len, proj_stack_factor);
            }

            // whisper downscales input token by half after conv1d
            n_patches = @divTrunc(n_patches, 2);

            if (ctx.model.audioHasAvgpool()) {
                // divide by 2 because of nn.AvgPool1d(2, stride=2)
                n_patches = @divTrunc(n_patches, 2);
            }
        },
        .qwen3a => {
            // chunk_size=100 frames --> 3x stride-2 conv2d --> 13 tokens per chunk
            const chunk_size: c_int = 100;
            const tokens_per_chunk: c_int = 13;
            n_patches = @divTrunc(img.nx(), chunk_size) * tokens_per_chunk;
        },
        .glma => {
            n_patches = img.nx();
            // whisper downscales input token by half after conv1d
            n_patches = @divTrunc(n_patches, 2);
            // reshape by merge_factor
            n_patches = @divTrunc(n_patches, ctx.model.hparams.proj_stack_factor);
            // for BOI and EOI token embeddings
            n_patches += 2;
        },
        .cogvlm => {
            n_patches += 2; // for BOI and EOI token embeddings
        },
        .deepseekocr => {
            // SAM encoder applies two stride-2 convolutions (net_2 and net_3)
            // that reduce spatial dimensions by 4x in each direction (16x total)
            n_patches = @divTrunc(n_patches, 16);

            if (img.add_viewsep) {
                // global view: one image-newline per token-row + trailing view separator
                const h: c_int = @intFromFloat(sqrtf(@floatFromInt(n_patches)));
                n_patches = h * (h + 1) + 1;
            } else if (img.ny() >= img.nx() and @rem(img.ny(), img.nx()) == 0) {
                // tile row: one image-newline per token-row
                const grid_w = @divTrunc(img.ny(), img.nx());
                const tile_patches = @divTrunc(img.nx(), patch_size * 4); // patches per tile side (SAM divides by 4)
                const h = tile_patches;
                n_patches = (tile_patches * grid_w + 1) * h;
            }
        },
        .hunyuanvl => {
            const merge = ctx.model.hparams.n_merge;
            const ow = @divTrunc(@divTrunc(img.nx(), patch_size), merge);
            const oh = @divTrunc(@divTrunc(img.ny(), patch_size), merge);
            n_patches = (ow + 1) * oh + 2;
        },
        .deepseekocr2 => {
            // 1024 global view -> 256 query tokens + 1 view separator = 257;
            // 768 local tile   -> 144 query tokens, no separator.
            n_patches = @divTrunc(n_patches, 16);
            if (img.add_viewsep) n_patches += 1; // view separator, appended only after the global view
        },
        .lfm2a => {
            n_patches = @divTrunc(@divTrunc(@divTrunc(img.nx() + 1, 2) + 1, 2) + 1, 2);
        },
        .gemma4a => {
            // Two Conv2D stride-2: O = floor((I - 1) / 2) + 1
            var n = img.nx();
            for (0..2) |_| n = @divTrunc(n - 1, 2) + 1;
            n_patches = n;
        },
        .parakeet => {
            n_patches = @divTrunc(img.nx() + (params.subsampling_factor - 1), params.subsampling_factor);
        },
        .gemma4ua => {
            n_patches = img.nx(); // no downsampling: one token per raw waveform frame
        },
        .mimo_audio => {
            // conv1(s=1) + conv2(s=2) -> RVQ-encoder downsample conv(k=2,s=2)
            var n = img.nx();
            n = @divTrunc(n - 1, 2) + 1; // conv1 + conv2
            n = @divTrunc(n - 2, 2) + 1; // downsample conv
            const group_size = params.audio_local_group_size;
            n_patches = @divTrunc(n + group_size - 1, group_size);
        },
        .granite_speech => {
            const ws = ctx.model.hparams.audio_proj_window_size;
            const ds = ctx.model.hparams.audio_proj_downsample_rate;
            n_patches = @divTrunc(img.nx() + ws - 1, ws) * @divTrunc(ws, ds);
        },
        .qwen3tts_spkenc => {
            // pooling gives one speaker embedding, whatever the clip length is
            n_patches = 1;
        },
        .qwen3tts_gen => {
            // one hidden-state vector fed back to the talker per call
            n_patches = 1;
        },
        .pockettts_spkenc => {
            // one conditioning row per 12.5Hz frame
            const hop = ctx.model.hparams.mimi_downsample * 120;
            n_patches = @divTrunc(img.nx(), hop);
        },
        .pockettts_gen => {
            // one latent per call for GEN_CODE, GEN_WAV sizes its input from the caller
            n_patches = 1;
        },
        .granite4_vision => {
            // Per-tile output token count: each projector block outputs
            // query_side^2 tokens per window x n^2 windows.
            const window_side = ctx.model.hparams.downsample_window_side;
            const query_side = ctx.model.hparams.downsample_query_side;
            const side = @divTrunc(img.nx(), params.patch_size);
            const n = @divTrunc(side, window_side);
            const out_side = query_side * n;
            n_patches = out_side * out_side;
            if (img.anyres.isTiled()) {
                // overview tile, then the unpadded tile grid with one newline per row
                const u = impl.anyresUnpad(img.anyres.grid_x * out_side, img.anyres.grid_y * out_side, img.anyres.orig_nx, img.anyres.orig_ny);
                n_patches += u.out_h * (u.out_w + 1);
            }
        },
        else => impl.abort("clip.cpp", 4337, "unsupported projector type"),
    }

    return n_patches;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "normal_distribution<float> draws in pairs and keeps the second" {
    var g = libcxx.Mt19937.init(5489);
    var d: NormalF32 = .{ .mean = 0, .stddev = 1 };
    const a = d.next(&g);
    try std.testing.expect(d.v_hot);
    const b = d.next(&g);
    try std.testing.expect(!d.v_hot);
    try std.testing.expect(a != b);
}

test "make_win_idx and make_unwin_idx are inverse permutations" {
    const fwd = try makeWinIdx(6, 3);
    defer gpa.free(fwd);
    const inv = try makeUnwinIdx(6, 3);
    defer gpa.free(inv);
    for (fwd, 0..) |v, i| try std.testing.expectEqual(@as(i32, @intCast(i)), inv[@intCast(v)]);
}
