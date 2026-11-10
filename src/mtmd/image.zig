//! The image preprocessors: an RGB image in, the normalized `f32` crops
//! and overview a vision encoder reads out.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/mtmd-image.cpp` and
//! `llama.cpp/tools/mtmd/mtmd-image.h` at v0.9.0 (`cd959bae9`). Each
//! declaration names what it ports and the line it began at.
//!
//! # One preprocessor type, not a class tree
//!
//! Upstream has a base class with nineteen subclasses, three of which
//! (`minicpmv`, `lfm2`, `cohere2v`) override the virtual
//! `get_slice_instructions` that `llava_uhd`'s `preprocess` calls, one
//! (`qwen3vl`) the video virtuals, and one (`lfm3vl`, derived from
//! `qwen3vl`) `dyn_size`'s `target_size` as well. Here
//! `Preprocessor` carries a `Kind` and dispatches on it at the same points
//! -- `preprocess`, `preprocessVideoFrame`, `getVideoFrameSize`, `targetSize`
//! and `getSliceInstructions` -- so the call graph is upstream's. A
//! preprocessor reads `clip_hparams` through `clip_get_hparams(ctx)`, or
//! since v0.8.0 is handed them, upstream; it takes a `*const Hparams` here,
//! with the projector type `deepseekocr` asks its context for.
//!
//! # Arithmetic
//!
//! Every `float` and `double` is the C's, and every `a*b + c` clang
//! contracts at `-ffp-contract=on` is an `@mulAdd` here -- the bicubic
//! kernel, the fixed-point weight rounding, step3vl's bilinear lerp.
//! Transcendentals (`sin`, `log`, `logf`, `nearbyint`) call the C library,
//! as the reference does. `make mtmd-image-diff` holds every output to the
//! reference on bits.

const std = @import("std");
const impl = @import("impl.zig");
const model = @import("model.zig");
const names = @import("names.zig");
const clip = @import("clip.zig");
const libcxx = @import("llama").libcxx;

const c = impl.c;
const gpa = impl.gpa;
const ImageU8 = impl.ImageU8;
const ImageF32 = impl.ImageF32;
const ImageSize = impl.ImageSize;
const Hparams = model.Hparams;
const ResizeAlgo = model.ResizeAlgo;
const PadStyle = model.PadStyle;

const libm = struct {
    extern "c" fn sin(x: f64) f64;
    extern "c" fn log(x: f64) f64;
    extern "c" fn logf(x: f32) f32;
    extern "c" fn nearbyint(x: f64) f64;
    extern "c" fn nearbyintf(x: f32) f32;
};

fn assert_(ok: bool, comptime line: c_int, comptime expr: [*:0]const u8) void {
    impl.ggmlAssert(ok, "mtmd-image.cpp", line, expr);
}

fn toF(x: anytype) f32 {
    return @floatFromInt(x);
}

fn toD(x: anytype) f64 {
    return @floatFromInt(x);
}

/// `static_cast<int>(f)`: truncation toward zero.
/// `CLIP_ALIGN` (clip-impl.h:485 @cd959bae9).
fn clipAlign(x: c_int, n: c_int) c_int {
    return @divTrunc(x + n - 1, n) * n;
}

/// `std::min` over two doubles: `(b < a) ? b : a`.
fn cMin(a: f64, b: f64) f64 {
    return if (b < a) b else a;
}

/// `std::max` over two doubles: `(a < b) ? b : a`.
fn cMax(a: f64, b: f64) f64 {
    return if (a < b) b else a;
}

fn toI(x: anytype) c_int {
    return @intFromFloat(x);
}

/// Ports `mtmd_image_preproc_out` (mtmd-image.h:11 @cd959bae9): what a
/// preprocessor produces. The caller owns it.
pub const PreprocOut = struct {
    entries: std.ArrayList(ImageF32) = .empty,
    /// overview image (downscaled image)
    overview: ImageF32 = .{},
    /// grid size is required for llava-uhd style models
    grid_x: c_int = 0,
    grid_y: c_int = 0,

    pub fn deinit(self: *PreprocOut) void {
        for (self.entries.items) |*e| e.deinit();
        self.entries.deinit(gpa);
        self.overview.deinit();
    }

    /// Ports `append` (mtmd-image.cpp:7 @cd959bae9),
    /// a member of `mtmd_image_preproc_out`: one `u8` image.
    pub fn appendU8(self: *PreprocOut, hparams: *const Hparams, img: *const ImageU8, normalized: bool) !void {
        var dst: ImageF32 = .{};
        errdefer dst.deinit();
        try dst.fromU8(img);
        if (normalized) dst.normalize(&hparams.image_mean, &hparams.image_std);
        try self.entries.append(gpa, dst);
    }

    /// Ports `append` (mtmd-image.cpp:16 @cd959bae9),
    /// a member of `mtmd_image_preproc_out`: several `u8` images.
    pub fn appendU8s(self: *PreprocOut, hparams: *const Hparams, imgs: []const ImageU8, normalized: bool) !void {
        for (imgs) |*img| try self.appendU8(hparams, img, normalized);
    }

    /// Ports `append` (mtmd-image.cpp:22 @cd959bae9),
    /// a member of `mtmd_image_preproc_out`: an `f32` image, moved in; `img`
    /// is left empty.
    pub fn appendF32(self: *PreprocOut, hparams: *const Hparams, img: *ImageF32, normalized: bool) !void {
        if (normalized) img.normalize(&hparams.image_mean, &hparams.image_std);
        try self.entries.append(gpa, img.*);
        img.* = .{};
    }

    /// Ports `append_overview` (mtmd-image.cpp:29
    /// @cd959bae9), a member of `mtmd_image_preproc_out`.
    pub fn appendOverview(self: *PreprocOut, hparams: *const Hparams, img: *const ImageU8, normalized: bool) !void {
        try self.overview.fromU8(img);
        if (normalized) self.overview.normalize(&hparams.image_mean, &hparams.image_std);
    }

    /// `has_overview` (mtmd-image.h:24 @cd959bae9).
    pub fn hasOverview(self: *const PreprocOut) bool {
        return self.overview.nx() > 0 or self.overview.ny() > 0;
    }
};

// -----------------------------------------------------------------------------
// img_tool

/// Ports `img_tool` (mtmd-image.cpp:38 @cd959bae9): the image
/// manipulations every preprocessor shares.
pub const ImgTool = struct {
    /// Ports `resize` (mtmd-image.cpp:39 @cd959bae9), a member of
    /// `img_tool`.
    pub fn resize(src: *const ImageU8, dst: *ImageU8, target_resolution: ImageSize, algo: ResizeAlgo, padding: PadStyle, pad_color: [3]u8) !void {
        try dst.setSize(target_resolution, src.isPlaceholder());

        if (src.isPlaceholder()) {
            // no-op for placeholder image, just set the size and return
            return;
        }

        if (dst.getSize().eql(src.getSize())) {
            // no resize needed, simple copy
            try dst.cpyBuf(src.roBuf());
            return;
        }

        if (padding == .none) {
            // direct resize
            try resizePillow(src, dst, target_resolution.width, target_resolution.height, algo);
        } else {
            // resize with padding
            var resized_image: ImageU8 = .{};
            defer resized_image.deinit();
            const scale_w = toF(target_resolution.width) / toF(src.getSize().width);
            const scale_h = toF(target_resolution.height) / toF(src.getSize().height);
            const scale = @min(scale_w, scale_h);

            var new_width: c_int = undefined;
            var new_height: c_int = undefined;
            if (padding == .nearest) {
                new_width = @min(toI(@round(toF(src.getSize().width) * scale)), target_resolution.width);
                new_height = @min(toI(@round(toF(src.getSize().height) * scale)), target_resolution.height);
            } else {
                new_width = @min(toI(@ceil(toF(src.getSize().width) * scale)), target_resolution.width);
                new_height = @min(toI(@ceil(toF(src.getSize().height) * scale)), target_resolution.height);
            }

            try resizePillow(src, &resized_image, new_width, new_height, algo);

            // fill dst with pad_color
            fill(dst, pad_color);

            var offset_x: c_int = undefined;
            var offset_y: c_int = undefined;
            if (padding == .nearest) {
                offset_x = toI(@round(toF(target_resolution.width - new_width) / 2.0));
                offset_y = toI(@round(toF(target_resolution.height - new_height) / 2.0));
            } else {
                offset_x = @divTrunc(target_resolution.width - new_width, 2);
                offset_y = @divTrunc(target_resolution.height - new_height, 2);
            }
            composite(dst, &resized_image, offset_x, offset_y);
        }
    }

    /// Ports `crop` (mtmd-image.cpp:95 @cd959bae9), a member of
    /// `img_tool`.
    pub fn crop(image: *const ImageU8, dst: *ImageU8, x: c_int, y: c_int, w: c_int, h: c_int) !void {
        assert_(x >= 0 and y >= 0 and w > 0 and h > 0, 96, "x >= 0 && y >= 0 && w > 0 && h > 0");
        assert_(x + w <= image.getSize().width and y + h <= image.getSize().height, 97, "x + w <= image.get_size().width && y + h <= image.get_size().height");
        try dst.setSize(.{ .width = w, .height = h }, image.isPlaceholder());

        if (image.isPlaceholder()) {
            // no-op for placeholder image, just set the size and return
            return;
        }

        var i: c_int = 0;
        while (i < h) : (i += 1) {
            var j: c_int = 0;
            while (j < w) : (j += 1) dst.setPixel(j, i, image.getPixel(x + j, y + i));
        }
    }

    /// Ports `calc_size_opt` (mtmd-image.cpp:112 @cd959bae9), a
    /// member of `img_tool`.
    pub const CalcSizeOpt = struct {
        align_size: c_int = 1,
        min_pixels: c_int = 0, // 0 = disabled
        max_pixels: c_int = 0, // 0 = disabled
        // applied before min/max_pixels, so min_pixels can push an edge back above longest_edge
        longest_edge: c_int = 0, // 0 = disabled
        round_half_even: bool = false, // Python round() instead of std::round
    };

    /// Ports `calc_size_preserved_ratio` (mtmd-image.cpp:123
    /// @cd959bae9), a member of `img_tool`: the size of the **resized**
    /// image, preserving the aspect ratio and aligned to `align_size`
    /// ("smart_resize" in transformers).
    pub fn calcSizePreservedRatio(inp_size: ImageSize, opts: CalcSizeOpt) ImageSize {
        assert_(opts.align_size > 0, 124, "opts.align_size > 0");
        const width = inp_size.width;
        const height = inp_size.height;
        if (width <= 0 or height <= 0) return .{ .width = 0, .height = 0 };

        const f = opts.align_size;
        const F = struct {
            fn roundBy(fa: c_int, even: bool, x: f32) c_int {
                const q = x / toF(fa);
                return toI(if (even) libm.nearbyintf(q) else @round(q)) * fa;
            }
            fn ceilBy(fa: c_int, x: f32) c_int {
                return toI(@ceil(x / toF(fa))) * fa;
            }
            fn floorBy(fa: c_int, x: f32) c_int {
                return toI(@floor(x / toF(fa))) * fa;
            }
        };

        var w_bar: c_int = undefined;
        var h_bar: c_int = undefined;
        if (opts.longest_edge > 0) {
            const scale = @min(toF(opts.longest_edge) / toF(width), toF(opts.longest_edge) / toF(height));
            w_bar = F.ceilBy(f, toF(width) * scale);
            h_bar = F.ceilBy(f, toF(height) * scale);
        } else {
            // always align up first
            w_bar = @max(opts.align_size, F.roundBy(f, opts.round_half_even, toF(width)));
            h_bar = @max(opts.align_size, F.roundBy(f, opts.round_half_even, toF(height)));
        }

        if (opts.max_pixels > 0 and h_bar *% w_bar > opts.max_pixels) {
            const beta = @sqrt(toF(height) * toF(width) / toF(opts.max_pixels));
            h_bar = @max(opts.align_size, F.floorBy(f, toF(height) / beta));
            w_bar = @max(opts.align_size, F.floorBy(f, toF(width) / beta));
        } else if (opts.min_pixels > 0 and h_bar *% w_bar < opts.min_pixels) {
            const beta = @sqrt(toF(opts.min_pixels) / (toF(height) * toF(width)));
            h_bar = F.ceilBy(f, toF(height) * beta);
            w_bar = F.ceilBy(f, toF(width) * beta);
        }

        return .{ .width = w_bar, .height = h_bar };
    }

    /// Ports `composite` (mtmd-image.cpp:164 @cd959bae9), a member
    /// of `img_tool`: draw `src` into `dst` at an offset.
    pub fn composite(dst: *ImageU8, src: *const ImageU8, offset_x: c_int, offset_y: c_int) void {
        if (src.isPlaceholder()) {
            // no-op for placeholder image
            return;
        }

        const src_size = src.getSize();
        const dst_size = dst.getSize();
        var y: c_int = 0;
        while (y < src_size.height) : (y += 1) {
            var x: c_int = 0;
            while (x < src_size.width) : (x += 1) {
                const dx = x + offset_x;
                const dy = y + offset_y;
                // skip pixels that would be out of bounds in the destination
                if (dx < 0 or dy < 0 or dx >= dst_size.width or dy >= dst_size.height) continue;
                dst.setPixel(dx, dy, src.getPixel(x, y));
            }
        }
    }

    /// Ports `fill` (mtmd-image.cpp:186 @cd959bae9), a member of
    /// `img_tool`: a solid colour.
    pub fn fill(img: *ImageU8, color: [3]u8) void {
        if (img.isPlaceholder()) {
            // no-op for placeholder image
            return;
        }

        const size = img.getSize();
        var y: c_int = 0;
        while (y < size.height) : (y += 1) {
            var x: c_int = 0;
            while (x < size.width) : (x += 1) img.setPixel(x, y, color);
        }
    }

    // Fixed-point precision: 22 bits = 32 (int32_t) - 8 (uint8_t pixels) - 2 (headroom for accumulation)
    const precision_bits = 32 - 8 - 2;

    /// The `resample_filter` lambda
    /// (mtmd-image.cpp:239 @cd959bae9) of `img_tool::resize_pillow`: the weight at distance `x`. For
    /// bicubic, Pillow (and PyTorch with antialias) uses a = -0.5, where GGML
    /// (and PyTorch without antialias) uses -0.75. `algo` is the base
    /// algorithm, without the torchvision flag.
    fn resampleFilter(algo: ResizeAlgo, x_in: f64) f64 {
        var x = x_in;
        if (algo == .lanczos) {
            if (-3.0 <= x and x < 3.0) return sinc(x) * sinc(x / 3.0);
            return 0.0;
        }

        if (x < 0.0) x = -x;

        if (algo == .bilinear) return if (x < 1.0) 1.0 - x else 0.0;

        const a: f64 = -0.5;
        if (x < 1.0) return @mulAdd(f64, @mulAdd(f64, a + 2.0, x, -(a + 3.0)) * x, x, 1);
        if (x < 2.0) return @mulAdd(f64, @mulAdd(f64, x - 5, x, 8), x, -4) * a;
        return 0.0; // Zero outside [-2, 2]
    }

    fn sinc(v: f64) f64 {
        if (v == 0.0) return 1.0;
        const pi_v = v * 3.141592653589793238462643383279502884;
        return libm.sin(pi_v) / pi_v;
    }

    /// The `clip8` lambda of `img_tool::resize_pillow`.
    fn clip8(val: i32) u8 {
        if (val < 0) return 0;
        if (val > 255) return 255;
        return @intCast(val);
    }

    /// The `precompute_weights` lambda
    /// (mtmd-image.cpp:291 @cd959bae9) of `img_tool::resize_pillow`: one dimension's input ranges and
    /// fixed-point weights.
    ///
    /// Return: the kernel size.
    fn precomputeWeights(algo: ResizeAlgo, int16_weights: bool, filter_support: f64, in_size: c_int, out_size: c_int, bounds: *std.ArrayList(c_int), weights: *std.ArrayList(i32)) !c_int {
        assert_(in_size > 0 and out_size > 0, 293, "inSize > 0 && outSize > 0");

        // ratio of input range to output size; keep filterscale = 1 when
        // upsampling, widen the filter when downsampling
        const scale = toD(in_size) / toD(out_size);
        var filterscale = scale;
        if (filterscale < 1.0) filterscale = 1.0;

        // Determine filter support radius and kernel size
        const support = filter_support * filterscale;
        const ksize: c_int = toI(@ceil(support)) * 2 + 1;
        const ks: usize = @intCast(ksize);
        const outs: usize = @intCast(out_size);

        const pre_weights = try gpa.alloc(f64, outs * ks);
        defer gpa.free(pre_weights);
        @memset(pre_weights, 0);
        try bounds.resize(gpa, outs * 2);

        var xx: usize = 0;
        while (xx < outs) : (xx += 1) {
            // center in input space (pixel-center convention: +0.5)
            const center = (toD(xx) + 0.5) * scale;
            var ww: f64 = 0.0; // Sum of weights for normalization
            const ss = 1.0 / filterscale; // Scale factor for filter function

            var xmin: c_int = toI(center - support + 0.5);
            if (xmin < 0) xmin = 0;

            var xmax: c_int = toI(center + support + 0.5);
            if (xmax > in_size) xmax = in_size;

            xmax -= xmin;

            var x: c_int = 0;
            while (x < xmax) : (x += 1) {
                // distance from input pixel center to output pixel center in input space
                const w = resampleFilter(algo, (toD(x + xmin) - center + 0.5) * ss);
                pre_weights[xx * ks + @as(usize, @intCast(x))] = w;
                ww += w;
            }

            // Normalize weights to sum to 1.0 (preserves brightness)
            x = 0;
            while (x < xmax) : (x += 1) {
                if (ww != 0.0) pre_weights[xx * ks + @as(usize, @intCast(x))] /= ww;
            }

            // Zero-pad remaining kernel positions
            while (x < ksize) : (x += 1) pre_weights[xx * ks + @as(usize, @intCast(x))] = 0;

            bounds.items[xx * 2 + 0] = xmin;
            bounds.items[xx * 2 + 1] = xmax;
        }

        // torchvision rounds the weights to int16, with as many fractional
        // bits as the largest weight allows; scaled back to precision_bits,
        // they give the same output in the passes below. The product is by a
        // power of two, so it is exact and clang's fused form is this one.
        var precision: c_int = precision_bits;
        if (int16_weights) {
            const weight_max = std.mem.max(f64, pre_weights);
            precision = 0;
            while (precision < precision_bits) : (precision += 1) {
                const scaled = 0.5 + weight_max * toD(@as(c_int, 1) << @intCast(precision + 1));
                if (toI(scaled) >= (1 << 15)) break;
            }
        }

        // round(float * 2^precision): Pillow adds +/- 0.5 then truncates
        // toward zero
        try weights.resize(gpa, outs * ks);
        const fxp_scale = std.math.ldexp(@as(f64, 1.0), precision);
        const to_precision_bits = @as(i32, 1) << @intCast(precision_bits - precision);
        for (pre_weights, 0..) |pw, i| {
            const rounded = @mulAdd(f64, pw, fxp_scale, if (pw < 0) -0.5 else 0.5);
            weights.items[i] = @as(i32, @intFromFloat(rounded)) * to_precision_bits;
        }

        return ksize;
    }

    /// The `resample_horizontal` lambda
    /// (mtmd-image.cpp:388 @cd959bae9) of `img_tool::resize_pillow`: width from `in_nx` to `out_nx`.
    /// The caller owns the result.
    fn resampleHorizontal(src: []const u8, in_nx: c_int, in_ny: c_int, out_nx: c_int, ksize: c_int, bounds: []const c_int, weights: []const i32) ![]u8 {
        const onx: usize = @intCast(out_nx);
        const inx: usize = @intCast(in_nx);
        const iny: usize = @intCast(in_ny);
        const ks: usize = @intCast(ksize);
        const out = try gpa.alloc(u8, onx * iny * 3);
        @memset(out, 0);

        for (0..iny) |yy| {
            const src_row = src[yy * inx * 3 ..];
            const dst_row = out[yy * onx * 3 ..];

            for (0..onx) |xx| {
                const xmin: usize = @intCast(bounds[xx * 2 + 0]);
                const xcnt: usize = @intCast(bounds[xx * 2 + 1]);
                const k = weights[xx * ks ..];
                var p = src_row[xmin * 3 ..];

                // rounding bias (0.5 in fixed-point)
                var ss0: i32 = 1 << (precision_bits - 1);
                var ss1: i32 = 1 << (precision_bits - 1);
                var ss2: i32 = 1 << (precision_bits - 1);

                for (0..xcnt) |x| {
                    ss0 +%= @as(i32, p[0]) *% k[x];
                    ss1 +%= @as(i32, p[1]) *% k[x];
                    ss2 +%= @as(i32, p[2]) *% k[x];
                    p = p[3..];
                }

                dst_row[xx * 3 + 0] = clip8(ss0 >> precision_bits);
                dst_row[xx * 3 + 1] = clip8(ss1 >> precision_bits);
                dst_row[xx * 3 + 2] = clip8(ss2 >> precision_bits);
            }
        }

        return out;
    }

    /// The `resample_vertical` lambda
    /// (mtmd-image.cpp:431 @cd959bae9) of `img_tool::resize_pillow`: height to `out_ny`, a whole row at
    /// a time. The caller owns the result.
    fn resampleVertical(src: []const u8, in_nx: c_int, out_ny: c_int, ksize: c_int, bounds: []const c_int, weight: []const i32) ![]u8 {
        const row_elems = @as(usize, @intCast(in_nx)) * 3;
        const ony: usize = @intCast(out_ny);
        const ks: usize = @intCast(ksize);
        const out = try gpa.alloc(u8, row_elems * ony);
        @memset(out, 0);
        const acc = try gpa.alloc(i32, row_elems);
        defer gpa.free(acc);

        for (0..ony) |yy| {
            const ymin: usize = @intCast(bounds[yy * 2 + 0]);
            const ycnt: usize = @intCast(bounds[yy * 2 + 1]);
            const k = weight[yy * ks ..];

            // Rounding bias (0.5 in fixed-point)
            @memset(acc, 1 << (precision_bits - 1));

            for (0..ycnt) |y| {
                const src_row = src[(ymin + y) * row_elems ..];
                const w = k[y];
                for (0..row_elems) |i| acc[i] +%= @as(i32, src_row[i]) *% w;
            }

            const dst_row = out[yy * row_elems ..];
            for (0..row_elems) |i| dst_row[i] = clip8(acc[i] >> precision_bits);
        }

        return out;
    }

    /// Ports `resize_pillow` (mtmd-image.cpp:210 @cd959bae9), a
    /// member of `img_tool`: Pillow-compatible separable resampling
    /// (bilinear, bicubic, Lanczos) in 22-bit fixed point, adapted upstream
    /// from Pillow's `Resample.c`. `flag_torch` is the same, except for the
    /// int16 weights of torchvision.
    fn resizePillow(img: *const ImageU8, dst: *ImageU8, target_width: c_int, target_height: c_int, algo_flagged: ResizeAlgo) !void {
        const int16_weights = algo_flagged.isTorch();
        const algo = algo_flagged.base();

        if (int16_weights and algo == .lanczos) {
            impl.abort("mtmd-image.cpp", 224, "unsupported resize algo: torchvision has no lanczos");
        }

        // Filter support radius
        const filter_support: f64 = switch (algo) {
            .bilinear => 1.0,
            .bicubic => 2.0,
            .lanczos => 3.0,
            else => {
                _ = c.ggml_abort("mtmd-image.cpp", 234, "unsupported resize algo: %d", @as(c_int, @backingInt(algo)));
                unreachable;
            },
        };

        const src_width = img.getSize().width;
        const src_height = img.getSize().height;

        // A sanity check on the target size.
        if (target_width <= 0 or target_width > 65536 or target_height <= 0 or target_height > 65536) {
            return clip.throwErr("resize target %dx%d is out of range (max 65536)", .{ target_width, target_height });
        }

        const need_horizontal = target_width != src_width;
        const need_vertical = target_height != src_height;

        var bounds_horiz: std.ArrayList(c_int) = .empty;
        defer bounds_horiz.deinit(gpa);
        var bounds_vert: std.ArrayList(c_int) = .empty;
        defer bounds_vert.deinit(gpa);
        var weights_horiz: std.ArrayList(i32) = .empty;
        defer weights_horiz.deinit(gpa);
        var weights_vert: std.ArrayList(i32) = .empty;
        defer weights_vert.deinit(gpa);
        var ksize_horiz: c_int = 0;
        var ksize_vert: c_int = 0;

        if (need_horizontal) ksize_horiz = try precomputeWeights(algo, int16_weights, filter_support, src_width, target_width, &bounds_horiz, &weights_horiz);
        if (need_vertical) ksize_vert = try precomputeWeights(algo, int16_weights, filter_support, src_height, target_height, &bounds_vert, &weights_vert);

        const src = img.roBuf();
        if (need_horizontal and need_vertical) {
            const temp = try resampleHorizontal(src, src_width, src_height, target_width, ksize_horiz, bounds_horiz.items, weights_horiz.items);
            defer gpa.free(temp);
            try dst.setSize(.{ .width = target_width, .height = target_height }, false);
            const out = try resampleVertical(temp, target_width, target_height, ksize_vert, bounds_vert.items, weights_vert.items);
            defer gpa.free(out);
            try dst.cpyBuf(out);
        } else if (need_horizontal) {
            try dst.setSize(.{ .width = target_width, .height = src_height }, false);
            const out = try resampleHorizontal(src, src_width, src_height, target_width, ksize_horiz, bounds_horiz.items, weights_horiz.items);
            defer gpa.free(out);
            try dst.cpyBuf(out);
        } else if (need_vertical) {
            try dst.setSize(.{ .width = src_width, .height = target_height }, false);
            const out = try resampleVertical(src, src_width, target_height, ksize_vert, bounds_vert.items, weights_vert.items);
            defer gpa.free(out);
            try dst.cpyBuf(out);
        } else {
            // No resizing needed - direct copy
            try dst.setSize(img.getSize(), false);
            try dst.cpyBuf(img.roBuf());
        }
    }
};

// -----------------------------------------------------------------------------
// preprocessors

/// Which of upstream's preprocessor classes a `Preprocessor` is.
pub const Kind = enum {
    /// `mtmd_image_preprocessor_llava_uhd` (mtmd-image.h:65 @cd959bae9)
    llava_uhd,
    /// `mtmd_image_preprocessor_fixed_size` (mtmd-image.h:130 @cd959bae9)
    fixed_size,
    /// `mtmd_image_preprocessor_dyn_size` (mtmd-image.h:138 @cd959bae9)
    dyn_size,
    /// `mtmd_image_preprocessor_longest_edge` (mtmd-image.h:172 @cd959bae9)
    longest_edge,
    /// `mtmd_image_preprocessor_minicpmv` (mtmd-image.h:204 @cd959bae9)
    minicpmv,
    /// `mtmd_image_preprocessor_lfm2` (mtmd-image.h:243 @cd959bae9)
    lfm2,
    /// `mtmd_image_preprocessor_idefics3` (mtmd-image.h:261 @cd959bae9)
    idefics3,
    /// `mtmd_image_preprocessor_internvl` (mtmd-image.h:266 @cd959bae9)
    internvl,
    /// `mtmd_image_preprocessor_deepseekocr` (mtmd-image.h:280 @cd959bae9)
    deepseekocr,
    /// `mtmd_image_preprocessor_step3vl` (mtmd-image.h:306 @cd959bae9)
    step3vl,
    /// `mtmd_image_preprocessor_youtuvl` (mtmd-image.h:333 @cd959bae9)
    youtuvl,
    /// `mtmd_image_preprocessor_granite` (mtmd-image.h:339 @cd959bae9)
    granite,
    /// `mtmd_image_preprocessor_muse_glimmer` (mtmd-image.h:345 @cd959bae9)
    muse_glimmer,
    /// `mtmd_image_preprocessor_deepseek4v` (mtmd-image.h:179 @cd959bae9),
    /// last so the earlier kinds keep their ordinals, which
    /// `harness/mtmd_image_ref.cpp` switches on
    deepseek4v,
    /// `mtmd_image_preprocessor_glm5v` (mtmd-image.h:166 @cd959bae9), new
    /// at v0.6.0, after `deepseek4v` for the same reason
    glm5v,
    /// `mtmd_image_preprocessor_cohere2v` (mtmd-image.h:273 @cd959bae9), new
    /// at v0.7.0, last for the same reason
    cohere2v,
    /// `mtmd_image_preprocessor_lfm3vl` (mtmd-image.h:160 @cd959bae9), new
    /// at v0.8.0, after `cohere2v` for the same reason; derived from
    /// `qwen3vl` since v0.9.0
    lfm3vl,
    /// `mtmd_image_preprocessor_deepseek41v` (mtmd-image.h:195 @cd959bae9),
    /// new at v0.8.0, after `lfm3vl` for the same reason
    deepseek41v,
    /// `mtmd_image_preprocessor_minicpm_ocr_1_0` (mtmd-image.h:230
    /// @cd959bae9), new at v0.8.0, after `deepseek41v` for the same reason
    minicpm_ocr_1_0,
    /// `mtmd_image_preprocessor_qwen3vl` (mtmd-image.h:147 @cd959bae9), new
    /// at v0.9.0, last for the same reason: `dyn_size` for images, video
    /// frames sized as `Qwen3VLVideoProcessor`
    qwen3vl,
};

/// Ports `slice_coordinates`
/// (mtmd-image.h:70 @cd959bae9).
pub const SliceCoordinates = struct {
    x: c_int = 0,
    y: c_int = 0,
    size: ImageSize = .{},
};

/// Ports `slice_instructions`
/// (mtmd-image.h:76 @cd959bae9).
pub const SliceInstructions = struct {
    overview_size: ImageSize = .{}, // size of downscaled image
    refined_size: ImageSize = .{}, // size of image right before slicing (must be multiple of slice size)
    grid_size: ImageSize = .{}, // grid_size.width * grid_size.height = number of slices
    slices: std.ArrayList(SliceCoordinates) = .empty,

    pub fn deinit(self: *SliceInstructions) void {
        self.slices.deinit(gpa);
    }
};

/// Ports `slice_output` (mtmd-image.h:85
/// @cd959bae9).
pub const SliceOutput = struct {
    overview: ImageU8 = .{},
    slices: std.ArrayList(ImageU8) = .empty,

    pub fn deinit(self: *SliceOutput) void {
        self.overview.deinit();
        for (self.slices.items) |*s| s.deinit();
        self.slices.deinit(gpa);
    }
};

/// `mtmd_image_preprocessor_step3vl`'s `default_image_longest_edge`,
/// `default_image_crop_size`, `small_aspect_ratio_limit`,
/// `wide_aspect_ratio_limit` and `crop_rounding_threshold`
/// (mtmd-image.h:312, 313, 314, 315, 316 @cd959bae9).
const step3vl_default_image_longest_edge: c_int = 3024;
const step3vl_default_image_crop_size: c_int = 504;
const step3vl_small_aspect_ratio_limit: f32 = 1.5;
const step3vl_wide_aspect_ratio_limit: f32 = 4.0;
const step3vl_crop_rounding_threshold: f32 = 0.2;

/// Ports `mtmd_image_preprocessor` (mtmd-image.h:30 @cd959bae9) and its
/// subclasses: one preprocessor of kind `kind`.
pub const Preprocessor = struct {
    kind: Kind,
    hparams: *const Hparams,

    // `fuse_row` (mtmd-image.h:291 @cd959bae9) and the four members after
    // it, of `mtmd_image_preprocessor_deepseekocr`, set at construction as
    // upstream sets them
    fuse_row: bool = false, // v1 fuses a tile-row into one image; v2 keeps tiles separate
    base_size: c_int = 0, // global view
    tile_size: c_int = 0, // each tile
    min_tiles: c_int = 0,
    max_tiles: c_int = 0,

    /// The constructors, from a context and since v0.8.0 from `hparams`
    /// themselves: `hparams` as `clip_get_hparams(ctx)` returns it,
    /// `proj_type` as `clip_get_projector_type(ctx)` does (only
    /// `deepseekocr` reads it). `hparams` must outlive the preprocessor.
    pub fn init(kind: Kind, hparams: *const Hparams, proj_type: names.ProjectorType) Preprocessor {
        var self: Preprocessor = .{ .kind = kind, .hparams = hparams };
        if (kind == .deepseekocr) {
            self.fuse_row = proj_type == .deepseekocr;
            self.base_size = hparams.image_size;
            self.tile_size = hparams.preproc_tile_size;
            self.min_tiles = hparams.preproc_min_tiles;
            self.max_tiles = hparams.preproc_max_tiles;
        }
        return self;
    }

    /// The virtual `preprocess` (mtmd-image.h:37 @cd959bae9), dispatched.
    /// The caller owns the result.
    pub fn preprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        return switch (self.kind) {
            .llava_uhd, .minicpmv, .minicpm_ocr_1_0 => self.llavaUhdPreprocess(img),
            .fixed_size => self.fixedSizePreprocess(img),
            .dyn_size, .qwen3vl, .lfm3vl => self.dynSizePreprocess(img),
            .glm5v => self.glm5vPreprocess(img),
            .longest_edge => self.longestEdgePreprocess(img),
            .deepseek4v => self.deepseek4vPreprocess(img),
            .deepseek41v => self.deepseek41vPreprocess(img),
            .lfm2 => self.lfm2Preprocess(img),
            .idefics3 => self.idefics3Preprocess(img),
            .internvl => self.internvlPreprocess(img),
            .deepseekocr => self.deepseekocrPreprocess(img),
            .step3vl => self.step3vlPreprocess(img),
            .youtuvl => self.youtuvlPreprocess(img),
            .granite => self.granitePreprocess(img),
            .muse_glimmer => self.museGlimmerPreprocess(img),
            .cohere2v => self.cohere2vPreprocess(img),
        };
    }

    /// The virtual `preprocess_video_frame` (mtmd-image.h:39 @cd959bae9),
    /// dispatched: the frames of a video, which the video helper may have
    /// already sized. `preprocess` unless the kind overrides it. The caller
    /// owns the result.
    pub fn preprocessVideoFrame(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        return switch (self.kind) {
            .qwen3vl, .lfm3vl => self.qwen3vlPreprocessVideoFrame(img),
            else => self.preprocess(img),
        };
    }

    /// The virtual `get_video_frame_size` (mtmd-image.h:41 @cd959bae9),
    /// dispatched: the frame size of a video with `n_frames` sampled frames,
    /// as the reference video processor gives it.
    ///
    /// Parameters:
    /// - `n_frames`: sampled frames.
    /// - `width`, `height`: the source frame's size.
    /// - `out`: written when the kind has a size; left alone otherwise.
    ///
    /// Return: false if the kind has none.
    pub fn getVideoFrameSize(self: *const Preprocessor, n_frames: c_int, width: c_int, height: c_int, out: *ImageSize) bool {
        return switch (self.kind) {
            .qwen3vl, .lfm3vl => self.qwen3vlGetVideoFrameSize(n_frames, width, height, out),
            else => false,
        };
    }

    /// The virtual `get_slice_instructions` (mtmd-image.h:83 @cd959bae9),
    /// dispatched. The caller owns the result.
    pub fn getSliceInstructions(self: *const Preprocessor, original_size: ImageSize) !SliceInstructions {
        return switch (self.kind) {
            .minicpmv, .minicpm_ocr_1_0 => self.minicpmvGetSliceInstructions(original_size),
            .lfm2 => self.lfm2GetSliceInstructions(original_size),
            .cohere2v => self.cohere2vGetSliceInstructions(original_size),
            else => self.llavaUhdGetSliceInstructions(original_size),
        };
    }

    /// Ports `get_slice_align` (mtmd-image.h:93 @cd959bae9), a virtual
    /// member of `mtmd_image_preprocessor_llava_uhd`: align slices to a
    /// multiple of the merger factor (integer merger tokens per slice); and
    /// its override `get_slice_align` (mtmd-image.h:210 @cd959bae9) in
    /// `mtmd_image_preprocessor_minicpmv`: always `patch_size * 4`, even in
    /// 4x mode (the 2x2 vit_merger slot stays); and its override
    /// `get_slice_align` (mtmd-image.h:234 @cd959bae9) in
    /// `mtmd_image_preprocessor_minicpm_ocr_1_0`: `find_best_resize` aligns
    /// to `patch_size * n_merge` (28 in 4x), with a merge of 2 when unset.
    fn getSliceAlign(self: *const Preprocessor) c_int {
        const hparams = self.hparams;
        if (self.kind == .minicpmv) return hparams.patch_size * 4;
        const unset: c_int = if (self.kind == .minicpm_ocr_1_0) 2 else 1;
        const merge: c_int = if (hparams.n_merge > 0) hparams.n_merge else unset;
        return hparams.patch_size * merge;
    }

    /// Ports `align_round` (mtmd-image.h:99 @cd959bae9), a virtual member of
    /// `mtmd_image_preprocessor_llava_uhd`: the rounding for snapping a
    /// length to a multiple of the align size, `std::round`; and its
    /// override `align_round` (mtmd-image.h:215 @cd959bae9) in
    /// `mtmd_image_preprocessor_minicpmv`, which breaks ties to even, as
    /// Python's `round()` does.
    fn alignRound(self: *const Preprocessor, v: f64) c_int {
        if (self.kind == .minicpmv or self.kind == .minicpm_ocr_1_0) {
            const fl = @floor(v);
            const diff = v - fl;
            if (diff > 0.5) return @as(c_int, @intFromFloat(fl)) + 1;
            if (diff < 0.5) return @intFromFloat(fl);
            const lo: c_int = @intFromFloat(fl);
            return if (@rem(lo, 2) == 0) lo else lo + 1;
        }
        return @intFromFloat(@round(v));
    }

    // -------------------------------------------------------------------------
    // llava_uhd

    /// Ports `preprocess` (mtmd-image.cpp:519
    /// @cd959bae9), a member of `mtmd_image_preprocessor_llava_uhd`.
    fn llavaUhdPreprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        const original_size = img.getSize();
        var inst = try self.getSliceInstructions(original_size);
        defer inst.deinit();
        var sliced = try self.sliceImage(img, &inst);
        defer sliced.deinit();

        var output: PreprocOut = .{};
        errdefer output.deinit();
        try output.appendOverview(self.hparams, &sliced.overview, true);
        try output.appendU8s(self.hparams, sliced.slices.items, true);
        output.grid_x = inst.grid_size.width;
        output.grid_y = inst.grid_size.height;

        return output;
    }

    /// Ports `get_slice_instructions`
    /// (mtmd-image.cpp:533 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_llava_uhd`.
    fn llavaUhdGetSliceInstructions(self: *const Preprocessor, original_size: ImageSize) !SliceInstructions {
        const func = "get_slice_instructions";
        const hparams = self.hparams;
        var res: SliceInstructions = .{};
        errdefer res.deinit();
        const patch_size = self.getSliceAlign();
        const slice_size = hparams.image_size;
        const original_width = original_size.width;
        const original_height = original_size.height;

        const has_slices = original_size.width > slice_size or original_size.height > slice_size;
        const has_pinpoints = hparams.image_res_candidates.items.len != 0;

        if (!has_slices) {
            // skip slicing logic
            res.overview_size = .{ .width = slice_size, .height = slice_size };
            res.refined_size = .{ .width = 0, .height = 0 };
            res.grid_size = .{ .width = 0, .height = 0 };
            return res;
        }

        if (has_pinpoints) {
            // has pinpoints, use them to calculate the grid size (e.g. llava-1.6)
            const refine_size = selectBestResolution(original_size, hparams.image_res_candidates.items);
            res.overview_size = .{ .width = slice_size, .height = slice_size };
            res.refined_size = refine_size;
            res.grid_size = .{ .width = 0, .height = 0 };

            impl.logDbg("%s: using pinpoints for slicing\n", .{func});
            impl.logDbg("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d\n", .{
                func,                    original_width,           original_height,
                res.overview_size.width, res.overview_size.height, res.refined_size.width,
                res.refined_size.height,
            });

            var y: c_int = 0;
            while (y < refine_size.height) : (y += slice_size) {
                var x: c_int = 0;
                while (x < refine_size.width) : (x += slice_size) {
                    var slice: SliceCoordinates = .{ .x = x, .y = y };
                    slice.size.width = @min(slice_size, refine_size.width - x);
                    slice.size.height = @min(slice_size, refine_size.height - y);
                    try res.slices.append(gpa, slice);
                    impl.logDbg("%s: slice %d: x=%d, y=%d, size=%dx%d\n", .{
                        func,    @as(c_int, @intCast(res.slices.items.len)) - 1, slice.x,
                        slice.y, slice.size.width,                               slice.size.height,
                    });
                }
            }

            res.grid_size.height = @divTrunc(refine_size.height, slice_size);
            res.grid_size.width = @divTrunc(refine_size.width, slice_size);
            impl.logDbg("%s: grid size: %d x %d\n", .{ func, res.grid_size.width, res.grid_size.height });

            return res;
        }

        // no pinpoints, dynamically calculate the grid size (e.g. minicpmv)

        const best_size = self.getBestResize(original_size, slice_size, patch_size, !has_slices);
        res.overview_size = best_size;

        {
            const max_slice_nums: c_int = if (hparams.max_slice_nums > 0) hparams.max_slice_nums else 9;
            const log_ratio = libm.logf(toF(original_width) / toF(original_height));
            const ratio = toF(original_width) * toF(original_height) / toF(slice_size * slice_size);
            const multiple: c_int = @intFromFloat(@min(@as(f64, @ceil(ratio)), toD(max_slice_nums)));

            const best_grid = try getBestGrid(max_slice_nums, multiple, log_ratio);
            const refine_size = self.getRefineSize(original_size, best_grid, slice_size, patch_size, true);
            res.grid_size = best_grid;
            res.refined_size = refine_size;

            impl.logDbg("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d, grid size: %d x %d\n", .{
                func,                    original_width,           original_height,
                res.overview_size.width, res.overview_size.height, res.refined_size.width,
                res.refined_size.height, res.grid_size.width,      res.grid_size.height,
            });

            const width = refine_size.width;
            const height = refine_size.height;
            const grid_x = @divTrunc(width, best_grid.width);
            const grid_y = @divTrunc(height, best_grid.height);
            var patches_y: c_int = 0;
            var ic: c_int = 0;
            while (patches_y < refine_size.height and ic < best_grid.height) : ({
                patches_y += grid_y;
                ic += 1;
            }) {
                var patches_x: c_int = 0;
                var jc: c_int = 0;
                while (patches_x < refine_size.width and jc < best_grid.width) : ({
                    patches_x += grid_x;
                    jc += 1;
                }) {
                    const slice: SliceCoordinates = .{ .x = patches_x, .y = patches_y, .size = .{ .width = grid_x, .height = grid_y } };
                    try res.slices.append(gpa, slice);
                    impl.logDbg("%s: slice %d: x=%d, y=%d, size=%dx%d\n", .{
                        func,    @as(c_int, @intCast(res.slices.items.len)) - 1, slice.x,
                        slice.y, slice.size.width,                               slice.size.height,
                    });
                }
            }
        }

        return res;
    }

    /// Ports `slice_image` (mtmd-image.cpp:636
    /// @cd959bae9), a member of `mtmd_image_preprocessor_llava_uhd`. The
    /// caller owns the result.
    pub fn sliceImage(self: *const Preprocessor, img: *const ImageU8, inst: *const SliceInstructions) !SliceOutput {
        const hparams = self.hparams;
        var output: SliceOutput = .{};
        errdefer output.deinit();

        // resize to overview size
        try ImgTool.resize(img, &output.overview, inst.overview_size, hparams.image_resize_algo_ov, hparams.image_pad_ov, hparams.image_pad_color_ov);

        if (inst.slices.items.len == 0) {
            // no slices, just return the overview image
            return output;
        }

        // resize to refined size
        var refined_img: ImageU8 = .{};
        defer refined_img.deinit();
        try ImgTool.resize(img, &refined_img, inst.refined_size, hparams.image_resize_algo_rf, hparams.image_pad_rf, hparams.image_pad_color_rf);

        // create slices
        for (inst.slices.items) |slice| {
            var img_slice: ImageU8 = .{};
            errdefer img_slice.deinit();
            try ImgTool.crop(&refined_img, &img_slice, slice.x, slice.y, slice.size.width, slice.size.height);
            try output.slices.append(gpa, img_slice);
        }

        return output;
    }

    /// Ports `get_best_resize`
    /// (mtmd-image.cpp:668 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_llava_uhd`.
    fn getBestResize(self: *const Preprocessor, original_size: ImageSize, scale_resolution: c_int, patch_size: c_int, allow_upscale: bool) ImageSize {
        var width = original_size.width;
        var height = original_size.height;
        if ((width *% height > scale_resolution *% scale_resolution) or allow_upscale) {
            const r = toF(width) / toF(height);
            height = toI(toF(scale_resolution) / @sqrt(r));
            width = toI(toF(height) * r);
        }
        return .{ .width = self.ensureDivide(width, patch_size), .height = self.ensureDivide(height, patch_size) };
    }

    /// Ports `resize_maintain_aspect_ratio`
    /// (mtmd-image.cpp:682 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_llava_uhd`.
    fn resizeMaintainAspectRatio(orig: ImageSize, target_max: ImageSize) ImageSize {
        const scale_width = toF(target_max.width) / toF(orig.width);
        const scale_height = toF(target_max.height) / toF(orig.height);
        const scale = @min(scale_width, scale_height);
        return .{ .width = toI(toF(orig.width) * scale), .height = toI(toF(orig.height) * scale) };
    }

    /// Ports `select_best_resolution`
    /// (mtmd-image.cpp:692 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_llava_uhd`: the candidate that wastes the
    /// least resolution.
    pub fn selectBestResolution(original_size: ImageSize, possible_resolutions: []const ImageSize) ImageSize {
        const func = "select_best_resolution";
        // uninitialized in the C++ when no candidate is taken
        var best_fit: ImageSize = .{};
        var min_wasted_area: c_int = std.math.maxInt(c_int);
        var max_effective_resolution: c_int = 0;

        for (possible_resolutions) |candidate| {
            const target_size = resizeMaintainAspectRatio(original_size, candidate);
            const effective_resolution = @min(target_size.width *% target_size.height, original_size.width *% original_size.height);
            const wasted_area = (candidate.width *% candidate.height) -% effective_resolution;

            if (effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_area < min_wasted_area)) {
                max_effective_resolution = effective_resolution;
                min_wasted_area = wasted_area;
                best_fit = candidate;
            }

            impl.logDbg("%s: candidate: %d x %d, target: %d x %d, wasted: %d, effective: %d\n", .{
                func, candidate.width, candidate.height, target_size.width, target_size.height, wasted_area, effective_resolution,
            });
        }

        return best_fit;
    }

    /// Ports `ensure_divide`
    /// (mtmd-image.cpp:716 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_llava_uhd`.
    fn ensureDivide(self: *const Preprocessor, length: c_int, patch_size: c_int) c_int {
        return @max(self.alignRound(toD(length) / toD(patch_size)) *% patch_size, patch_size);
    }

    /// Ports `get_refine_size`
    /// (mtmd-image.cpp:720 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_llava_uhd`.
    fn getRefineSize(self: *const Preprocessor, original_size: ImageSize, grid: ImageSize, scale_resolution: c_int, patch_size: c_int, allow_upscale: bool) ImageSize {
        const width = original_size.width;
        const height = original_size.height;
        const grid_x = grid.width;
        const grid_y = grid.height;

        const refine_width = self.ensureDivide(width, grid_x);
        const refine_height = self.ensureDivide(height, grid_y);

        const grid_size: ImageSize = .{ .width = @divTrunc(refine_width, grid_x), .height = @divTrunc(refine_height, grid_y) };

        const best_grid_size = self.getBestResize(grid_size, scale_resolution, patch_size, allow_upscale);
        return .{ .width = best_grid_size.width * grid_x, .height = best_grid_size.height * grid_y };
    }

    /// Ports `get_best_grid`
    /// (mtmd-image.cpp:743 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_llava_uhd`.
    fn getBestGrid(max_slice_nums: c_int, multiple: c_int, log_ratio: f32) !ImageSize {
        var candidate_split_grids_nums: std.ArrayList(c_int) = .empty;
        defer candidate_split_grids_nums.deinit(gpa);
        for ([_]c_int{ multiple - 1, multiple, multiple + 1 }) |i| {
            if (i == 1 or i > max_slice_nums) continue;
            try candidate_split_grids_nums.append(gpa, i);
        }

        var candidate_grids: std.ArrayList(ImageSize) = .empty;
        defer candidate_grids.deinit(gpa);
        for (candidate_split_grids_nums.items) |split_grids_nums| {
            var m: c_int = 1;
            while (m <= split_grids_nums) : (m += 1) {
                if (@rem(split_grids_nums, m) == 0) try candidate_grids.append(gpa, .{ .width = m, .height = @divTrunc(split_grids_nums, m) });
            }
        }

        var best_grid: ImageSize = .{ .width = 1, .height = 1 };
        var min_error = std.math.inf(f32);
        for (candidate_grids.items) |grid| {
            const err: f32 = @floatCast(@abs(@as(f64, log_ratio) - libm.log(1.0 * toD(grid.width) / toD(grid.height))));
            if (err < min_error) {
                best_grid = grid;
                min_error = err;
            }
        }
        return best_grid;
    }

    // -------------------------------------------------------------------------
    // fixed_size, dyn_size, longest_edge

    /// Ports `preprocess`
    /// (mtmd-image.cpp:779 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_fixed_size`: downscale or upscale to a fixed
    /// size.
    fn fixedSizePreprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        const hparams = self.hparams;
        var resized_image: ImageU8 = .{};
        defer resized_image.deinit();
        const sz = hparams.image_size;
        try ImgTool.resize(img, &resized_image, .{ .width = sz, .height = sz }, hparams.image_resize_algo, hparams.image_resize_pad, hparams.image_pad_color);
        var output: PreprocOut = .{};
        errdefer output.deinit();
        try output.appendU8(hparams, &resized_image, true);
        return output;
    }

    /// Ports `get_target_size` (mtmd-image.cpp:795 @cd959bae9), a static
    /// member of `mtmd_image_preprocessor_dyn_size`: a multiple of
    /// `patch_size * n_merge`, preserving the aspect ratio.
    pub fn dynSizeGetTargetSize(hparams: *const Hparams, original_size: ImageSize) ImageSize {
        assert_(hparams.image_min_pixels > 0 and hparams.image_max_pixels > 0, 796, "hparams.image_min_pixels > 0 && hparams.image_max_pixels > 0");
        // the original pixtral model doesn't have n_merge
        const cur_merge = hparams.n_merge;
        return ImgTool.calcSizePreservedRatio(original_size, .{
            .align_size = hparams.patch_size * cur_merge,
            .min_pixels = hparams.image_min_pixels,
            .max_pixels = hparams.image_max_pixels,
            .longest_edge = 0,
            .round_half_even = hparams.image_round_half_even,
        });
    }

    /// The virtual `target_size` (mtmd-image.h:143 @cd959bae9) of
    /// `mtmd_image_preprocessor_dyn_size`, dispatched; and its override
    /// `target_size` (mtmd-image.cpp:883 @cd959bae9) in
    /// `mtmd_image_preprocessor_lfm3vl` (`qwen3vl` keeps `dyn_size`'s).
    fn targetSize(self: *const Preprocessor, original_size: ImageSize) ImageSize {
        const hparams = self.hparams;
        if (self.kind == .lfm3vl) {
            return qwen3vlSmartResize(original_size.width, original_size.height, hparams.patch_size * hparams.n_merge, hparams.image_min_pixels, hparams.image_max_pixels, 1, 1);
        }
        return dynSizeGetTargetSize(hparams, original_size);
    }

    /// Ports `preprocess` (mtmd-image.cpp:810 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_dyn_size`: resize to `target_size`.
    fn dynSizePreprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        return self.qwen3vlResizeTo(img, self.targetSize(img.getSize()));
    }

    /// Ports `smart_resize` (mtmd-image.cpp:826 @cd959bae9), a static member
    /// of `mtmd_image_preprocessor_qwen3vl` (`lfm3vl`'s until v0.9.0): as transformers'
    /// Qwen2-VL `smart_resize`, rounding before the pixel checks (an edge may
    /// round to 0) and computing in `double`. With `temporal_factor > 1` it
    /// is the Qwen3-VL video variant over `n_frames`; upstream's defaults for
    /// the two are 1.
    pub fn qwen3vlSmartResize(width: c_int, height: c_int, factor: c_int, min_pixels: i64, max_pixels: i64, n_frames: c_int, temporal_factor: c_int) ImageSize {
        var h = toD(height);
        var w = toD(width);
        const fac = toD(factor);
        // the video variant first upscales edges below factor
        if (temporal_factor > 1 and (h < fac or w < fac)) {
            const scale = cMax(fac / h, fac / w);
            h = toD(toI(h * scale));
            w = toD(toI(w * scale));
        }

        const t_bar = @as(i64, @intFromFloat(libm.nearbyint(toD(n_frames) / toD(temporal_factor)))) * temporal_factor;
        var h_bar = toI(libm.nearbyint(h / fac)) * factor;
        var w_bar = toI(libm.nearbyint(w / fac)) * factor;
        if (t_bar * h_bar * w_bar > max_pixels) {
            const beta = @sqrt(toD(n_frames) * h * w / toD(max_pixels));
            h_bar = toI(@floor(h / beta / fac)) * factor;
            w_bar = toI(@floor(w / beta / fac)) * factor;
        } else if (t_bar * h_bar * w_bar < min_pixels) {
            const beta = @sqrt(toD(min_pixels) / (toD(n_frames) * h * w));
            h_bar = toI(@ceil(h * beta / fac)) * factor;
            w_bar = toI(@ceil(w * beta / fac)) * factor;
        }

        return .{ .width = @max(factor, w_bar), .height = @max(factor, h_bar) };
    }

    /// Ports `get_video_frame_size` (mtmd-image.cpp:853 @cd959bae9), a member
    /// of `mtmd_image_preprocessor_qwen3vl`: `Qwen3VLVideoProcessor`'s (and
    /// `Lfm3VlVideoProcessor`'s) size, `shortest_edge` 4096, `longest_edge`
    /// 25165824, `temporal_patch_size` 2.
    fn qwen3vlGetVideoFrameSize(self: *const Preprocessor, n_frames: c_int, width: c_int, height: c_int, out: *ImageSize) bool {
        const hparams = self.hparams;
        out.* = qwen3vlSmartResize(width, height, hparams.patch_size * hparams.n_merge, 4096, 25165824, n_frames, 2);
        return true;
    }

    /// Ports `preprocess_video_frame` (mtmd-image.cpp:859 @cd959bae9), a
    /// member of `mtmd_image_preprocessor_qwen3vl`: the video helper sizes the
    /// frames, so they are only snapped to the grid.
    fn qwen3vlPreprocessVideoFrame(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        const f = self.hparams.patch_size * self.hparams.n_merge;
        const size = img.getSize();
        return self.qwen3vlResizeTo(img, .{
            .width = @max(f, toI(libm.nearbyint(toD(size.width) / toD(f))) * f),
            .height = @max(f, toI(libm.nearbyint(toD(size.height) / toD(f))) * f),
        });
    }

    /// Ports `resize_to` (mtmd-image.cpp:868 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_qwen3vl`. `dyn_size`'s `preprocess`
    /// (mtmd-image.cpp:810 @cd959bae9) has the same body over `target_size`,
    /// and calls this. The caller owns the result.
    pub fn qwen3vlResizeTo(self: *const Preprocessor, img: *const ImageU8, size: ImageSize) !PreprocOut {
        const hparams = self.hparams;
        var resized_image: ImageU8 = .{};
        defer resized_image.deinit();
        try ImgTool.resize(img, &resized_image, size, hparams.image_resize_algo, hparams.image_resize_pad, hparams.image_pad_color);
        var output: PreprocOut = .{};
        errdefer output.deinit();
        try output.appendU8(hparams, &resized_image, true);
        return output;
    }

    /// Ports `preprocess` (mtmd-image.cpp:894 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_glm5v`, new at v0.6.0: the canvas is
    /// ceil-aligned to `patch_size * n_merge` and fitted to the token budget;
    /// the image is rescaled only to meet the budget and sits top-left, with
    /// black padding on the right and bottom.
    fn glm5vPreprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        const hparams = self.hparams;
        assert_(hparams.image_min_pixels > 0 and hparams.image_max_pixels > 0, 895, "hparams.image_min_pixels > 0 && hparams.image_max_pixels > 0");

        const factor: i64 = hparams.patch_size * hparams.n_merge;
        const min_px: i64 = hparams.image_min_pixels; // single-frame pixel counts
        const max_px: i64 = hparams.image_max_pixels;
        const height: i64 = img.getSize().height;
        const width: i64 = img.getSize().width;

        const Align = struct {
            f: i64,
            fn of(a: @This(), v: i64) i64 {
                return @divTrunc(v + a.f - 1, a.f) * a.f;
            }
        };
        const al: Align = .{ .f = factor };

        // aligned canvas within the budget
        var canvas_h = al.of(height);
        var canvas_w = al.of(width);

        if (canvas_h * canvas_w < min_px) {
            const scale = @sqrt(toD(min_px) / toD(height * width));
            canvas_h = al.of(@max(1, @as(i64, @intFromFloat(@ceil(toD(height) * scale)))));
            canvas_w = al.of(@max(1, @as(i64, @intFromFloat(@ceil(toD(width) * scale)))));
        }

        if (canvas_h * canvas_w > max_px) {
            // largest content height whose aligned canvas fits the budget
            var lo: i64 = 1;
            var hi: i64 = height;
            var best_h: i64 = factor;
            var best_w: i64 = factor;
            while (lo <= hi) {
                const ch = @divTrunc(lo + hi, 2);
                const cw = @max(1, @divTrunc(width * ch, height));
                const ah = al.of(ch);
                const aw = al.of(cw);
                if (ah * aw <= max_px) {
                    best_h = ah;
                    best_w = aw;
                    lo = ch + 1;
                } else {
                    hi = ch - 1;
                }
            }
            canvas_h = best_h;
            canvas_w = best_w;
        }

        // Scaled to fit the canvas, and never upscaled, unless below the min budget
        var scale = cMin(toD(canvas_h) / toD(height), toD(canvas_w) / toD(width));
        if (height * width >= min_px) {
            scale = cMin(1.0, scale);
        }
        const content_h: c_int = @intCast(@max(1, @min(canvas_h, @as(i64, @intFromFloat(@floor(toD(height) * scale))))));
        const content_w: c_int = @intCast(@max(1, @min(canvas_w, @as(i64, @intFromFloat(@floor(toD(width) * scale))))));

        var content: ImageU8 = .{};
        defer content.deinit();
        try ImgTool.resize(img, &content, .{ .width = content_w, .height = content_h }, hparams.image_resize_algo, .none, .{ 0, 0, 0 });

        var canvas: ImageU8 = .{};
        defer canvas.deinit();
        try canvas.setSize(.{ .width = @intCast(canvas_w), .height = @intCast(canvas_h) }, img.isPlaceholder());
        ImgTool.fill(&canvas, .{ 0, 0, 0 });
        ImgTool.composite(&canvas, &content, 0, 0);

        var output: PreprocOut = .{};
        errdefer output.deinit();
        try output.appendU8(hparams, &canvas, true);
        return output;
    }

    /// Ports `get_target_size` (mtmd-image.cpp:961 @cd959bae9), a static
    /// member of `mtmd_image_preprocessor_longest_edge`: as `dyn_size`'s,
    /// with the longest edge capped at `image_longest_edge`.
    ///
    /// Return: `error.Runtime`, as `clip.throwErr` gives it, when flooring
    /// before the alignment leaves an edge at zero.
    pub fn longestEdgeGetTargetSize(hparams: *const Hparams, original_size: ImageSize) !ImageSize {
        assert_(hparams.image_longest_edge > 0, 962, "hparams.image_longest_edge > 0");
        var size = original_size;
        if (hparams.image_floor_before_align) {
            const ratio = toD(@max(size.width, size.height)) / toD(hparams.image_longest_edge);
            if (ratio > 1.0) {
                size.width = toI(@floor(toD(size.width) / ratio));
                size.height = toI(@floor(toD(size.height) / ratio));
                if (size.width <= 0 or size.height <= 0) {
                    return clip.throwErr("image dimensions must be positive after resizing", .{});
                }
            }
        }

        // the original pixtral model doesn't have n_merge
        const cur_merge: c_int = if (hparams.n_merge == 0) 1 else hparams.n_merge;
        return ImgTool.calcSizePreservedRatio(size, .{
            .align_size = hparams.patch_size * cur_merge,
            .min_pixels = @max(0, hparams.image_min_pixels),
            .max_pixels = @max(0, hparams.image_max_pixels),
            .longest_edge = @min(hparams.image_longest_edge, @max(size.width, size.height)),
        });
    }

    /// Ports `preprocess` (mtmd-image.cpp:988 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_longest_edge`: as `dyn_size`, with the longest
    /// edge capped at `image_longest_edge`.
    fn longestEdgePreprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        const hparams = self.hparams;
        var resized_image: ImageU8 = .{};
        defer resized_image.deinit();
        const target_size = try longestEdgeGetTargetSize(hparams, img.getSize());
        try ImgTool.resize(img, &resized_image, target_size, hparams.image_resize_algo, hparams.image_resize_pad, hparams.image_pad_color);
        var output: PreprocOut = .{};
        errdefer output.deinit();
        try output.appendU8(hparams, &resized_image, true);
        return output;
    }

    // -------------------------------------------------------------------------
    // minicpmv, lfm2

    /// Ports `get_slice_instructions`
    /// (mtmd-image.cpp:1004 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_minicpmv`.
    fn minicpmvGetSliceInstructions(self: *const Preprocessor, original_size: ImageSize) !SliceInstructions {
        const hparams = self.hparams;
        // overview only for small images, unlike generic llava-uhd which slices once one side exceeds scale resolution
        const slice_size = hparams.image_size;
        const ratio = toF(original_size.width) * toF(original_size.height) / toF(slice_size * slice_size);
        if (ratio <= 1.0) {
            var inst: SliceInstructions = .{};
            inst.overview_size = self.getBestResize(original_size, slice_size, self.getSliceAlign(), true);
            inst.refined_size = .{ .width = 0, .height = 0 };
            inst.grid_size = .{ .width = 0, .height = 0 };
            return inst;
        }
        return self.llavaUhdGetSliceInstructions(original_size);
    }

    /// Ports `preprocess` (mtmd-image.cpp:1022
    /// @cd959bae9), a member of `mtmd_image_preprocessor_lfm2`.
    fn lfm2Preprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        var inst = try self.getSliceInstructions(img.getSize());
        defer inst.deinit();
        if (inst.slices.items.len != 0) return self.llavaUhdPreprocess(img);

        // single tile: no thumbnail (output.overview would emit
        // <|img_thumbnail|>, which is not wanted here)
        var sliced = try self.sliceImage(img, &inst);
        defer sliced.deinit();
        var output: PreprocOut = .{};
        errdefer output.deinit();
        try output.appendU8(self.hparams, &sliced.overview, true);
        return output;
    }

    /// Ports `should_tile` (mtmd-image.cpp:1036
    /// @cd959bae9), a member of `mtmd_image_preprocessor_lfm2`.
    pub fn lfm2ShouldTile(hparams: *const Hparams, original_size: ImageSize) bool {
        const align_size = hparams.patch_size * hparams.n_merge;

        const R = struct {
            fn roundBy(a: c_int, x: f32) c_int {
                return @as(c_int, @intFromFloat(libm.nearbyint(@as(f64, x) / toD(a)))) * a;
            }
        };

        const h_bar = @max(hparams.patch_size, R.roundBy(align_size, toF(original_size.height)));
        const w_bar = @max(hparams.patch_size, R.roundBy(align_size, toF(original_size.width)));

        return hparams.preproc_max_tiles > 1 and
            toD(h_bar) * toD(w_bar) > toD(hparams.image_max_pixels) * @as(f64, hparams.preproc_max_pixels_tolerance);
    }

    /// Ports `smart_resize` (mtmd-image.cpp:1055 @cd959bae9), a static member
    /// of `mtmd_image_preprocessor_lfm2`: `Lfm2VlImageProcessor.smart_resize`,
    /// Python's round half to even in `double`.
    pub fn lfm2SmartResize(hparams: *const Hparams, original_size: ImageSize) ImageSize {
        const align_size = hparams.patch_size * hparams.n_merge;
        const al = toD(align_size);
        const height = toD(original_size.height);
        const width = toD(original_size.width);

        var h_bar = @max(align_size, toI(libm.nearbyint(height / al)) * align_size);
        var w_bar = @max(align_size, toI(libm.nearbyint(width / al)) * align_size);

        if (toD(h_bar) * toD(w_bar) > toD(hparams.image_max_pixels)) {
            const beta = @sqrt(height * width / toD(hparams.image_max_pixels));
            h_bar = @max(align_size, toI(@floor(height / beta / al)) * align_size);
            w_bar = @max(align_size, toI(@floor(width / beta / al)) * align_size);
        } else if (toD(h_bar) * toD(w_bar) < toD(hparams.image_min_pixels)) {
            const beta = @sqrt(toD(hparams.image_min_pixels) / (height * width));
            h_bar = toI(@ceil(height * beta / al)) * align_size;
            w_bar = toI(@ceil(width * beta / al)) * align_size;
        }

        return .{ .width = w_bar, .height = h_bar };
    }

    /// Ports `get_slice_instructions`
    /// (mtmd-image.cpp:1078 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_lfm2`.
    fn lfm2GetSliceInstructions(self: *const Preprocessor, original_size: ImageSize) !SliceInstructions {
        const func = "get_slice_instructions";
        const hparams = self.hparams;
        var inst: SliceInstructions = .{};
        errdefer inst.deinit();
        inst.overview_size = lfm2SmartResize(hparams, original_size);

        const needs_tiling = lfm2ShouldTile(hparams, original_size);
        const tile_size = hparams.preproc_tile_size;

        if (!needs_tiling) {
            inst.refined_size = .{ .width = 0, .height = 0 };
            inst.grid_size = .{ .width = 0, .height = 0 };
            return inst;
        }

        const grid = try lfm2GetGridLayout(hparams, original_size.height, original_size.width);

        inst.grid_size = grid;
        inst.refined_size = .{ .width = tile_size * grid.width, .height = tile_size * grid.height };

        impl.logDbg("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d, grid size: %d x %d\n", .{
            func,                     original_size.width,       original_size.height,
            inst.overview_size.width, inst.overview_size.height, inst.refined_size.width,
            inst.refined_size.height, grid.width,                grid.height,
        });

        var row: c_int = 0;
        while (row < grid.height) : (row += 1) {
            var col: c_int = 0;
            while (col < grid.width) : (col += 1) {
                const slice: SliceCoordinates = .{
                    .x = col * tile_size,
                    .y = row * tile_size,
                    .size = .{ .width = tile_size, .height = tile_size },
                };
                try inst.slices.append(gpa, slice);
                impl.logDbg("%s: slice %d: x=%d, y=%d, size=%d x %d\n", .{
                    func,    @as(c_int, @intCast(inst.slices.items.len)) - 1, slice.x,
                    slice.y, slice.size.width,                                slice.size.height,
                });
            }
        }

        return inst;
    }

    /// Ports `find_closest_aspect_ratio`
    /// (mtmd-image.cpp:1120 @cd959bae9), a static member of
    /// `mtmd_image_preprocessor_lfm2`: in `double` as in Python, since `float`
    /// breaks exact ties such as 720/1600 between 2/4 and 2/5.
    fn lfm2FindClosestAspectRatio(hparams: *const Hparams, aspect_ratio: f64, target_ratios: []const ImageSize, width: c_int, height: c_int) ImageSize {
        var best_ratio_diff = std.math.floatMax(f64);
        var best_ratio: ImageSize = .{ .width = 1, .height = 1 };
        const area = toD(width) * toD(height);

        for (target_ratios) |ratio| {
            const target_aspect_ratio = toD(ratio.width) / toD(ratio.height);
            const ratio_diff = @abs(aspect_ratio - target_aspect_ratio);
            if (ratio_diff < best_ratio_diff) {
                best_ratio_diff = ratio_diff;
                best_ratio = ratio;
            } else if (ratio_diff == best_ratio_diff) {
                const tile_size = hparams.preproc_tile_size;
                const target_area = toD(tile_size *% tile_size *% ratio.width *% ratio.height);
                if (area > 0.5 * target_area) best_ratio = ratio;
            }
        }
        return best_ratio;
    }

    /// Ports `find_closest_aspect_ratio`
    /// (mtmd-image.cpp:1396 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_deepseekocr`, which stays in `float`.
    fn findClosestAspectRatio(tile_size: c_int, aspect_ratio: f32, target_ratios: []const ImageSize, width: c_int, height: c_int) ImageSize {
        var best_ratio_diff = std.math.floatMax(f32);
        var best_ratio: ImageSize = .{ .width = 1, .height = 1 };
        const area = toF(width *% height);

        for (target_ratios) |ratio| {
            const target_aspect_ratio = toF(ratio.width) / toF(ratio.height);
            const ratio_diff = @abs(aspect_ratio - target_aspect_ratio);
            if (ratio_diff < best_ratio_diff) {
                best_ratio_diff = ratio_diff;
                best_ratio = ratio;
            } else if (ratio_diff == best_ratio_diff) {
                const target_area = toF(tile_size *% tile_size *% ratio.width *% ratio.height);
                if (area > 0.5 * target_area) best_ratio = ratio;
            }
        }
        return best_ratio;
    }

    fn areaLess(_: void, a: ImageSize, b: ImageSize) bool {
        return a.width * a.height < b.width * b.height;
    }

    /// Ports `get_target_ratios`
    /// (mtmd-image.cpp:1146 @cd959bae9) and
    /// `get_target_ratios`
    /// (mtmd-image.cpp:1369 @cd959bae9), each a member of its class: every
    /// (w, h) tiling of `min_tiles` to `max_tiles` tiles, by area, with
    /// libc++'s unstable `std::sort` deciding the order among equal areas.
    /// The caller owns the result.
    fn getTargetRatios(min_tiles: c_int, max_tiles: c_int) !std.ArrayList(ImageSize) {
        var ratios: std.ArrayList(ImageSize) = .empty;
        errdefer ratios.deinit(gpa);
        var n = min_tiles;
        while (n <= max_tiles) : (n += 1) {
            var w: c_int = 1;
            while (w <= n) : (w += 1) {
                var h: c_int = 1;
                while (h <= n) : (h += 1) {
                    if (w * h < min_tiles or w * h > max_tiles) continue;
                    var found = false;
                    for (ratios.items) |r| {
                        if (r.width == w and r.height == h) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) try ratios.append(gpa, .{ .width = w, .height = h });
                }
            }
        }
        libcxx.sort(ImageSize, ratios.items, {}, areaLess);
        return ratios;
    }

    /// Ports `get_grid_layout`
    /// (mtmd-image.cpp:1174 @cd959bae9), a static member of
    /// `mtmd_image_preprocessor_lfm2`.
    pub fn lfm2GetGridLayout(hparams: *const Hparams, height: c_int, width: c_int) !ImageSize {
        const aspect_ratio = toD(width) / toD(height);
        var ratios = try getTargetRatios(hparams.preproc_min_tiles, hparams.preproc_max_tiles);
        defer ratios.deinit(gpa);
        return lfm2FindClosestAspectRatio(hparams, aspect_ratio, ratios.items, width, height);
    }

    // -------------------------------------------------------------------------
    // idefics3, internvl

    /// Ports `preprocess` (mtmd-image.cpp:1184
    /// @cd959bae9), a member of `mtmd_image_preprocessor_idefics3`: an
    /// aspect-preserving resize to the longest edge, then a stretch to
    /// multiples of `image_size`, rounding up.
    fn idefics3Preprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        const func = "preprocess";
        const hparams = self.hparams;
        const original_size = img.getSize();

        // old gguf files have no preprocessor longest size, custom token limits also need the generic size below
        if (hparams.image_longest_edge > 0 and hparams.image_min_pixels <= 0 and hparams.image_max_pixels <= 0) {
            const tile_size = hparams.image_size;
            const longest_edge = hparams.image_longest_edge;
            const aspect_ratio = toD(original_size.width) / toD(original_size.height);

            var resized_size: ImageSize = .{};
            if (original_size.width >= original_size.height) {
                resized_size.width = longest_edge;
                resized_size.height = @intFromFloat(toD(longest_edge) / aspect_ratio);
                resized_size.height += @rem(resized_size.height, 2);
            } else {
                resized_size.height = longest_edge;
                resized_size.width = @intFromFloat(toD(longest_edge) * aspect_ratio);
                resized_size.width += @rem(resized_size.width, 2);
            }

            const grid_x = @divTrunc(resized_size.width + tile_size - 1, tile_size);
            const grid_y = @divTrunc(resized_size.height + tile_size - 1, tile_size);
            const refined_size: ImageSize = .{ .width = grid_x * tile_size, .height = grid_y * tile_size };

            var resized_img: ImageU8 = .{};
            defer resized_img.deinit();
            try ImgTool.resize(img, &resized_img, resized_size, hparams.image_resize_algo, .none, .{ 0, 0, 0 });

            var refined_img: ImageU8 = .{};
            defer refined_img.deinit();
            try ImgTool.resize(&resized_img, &refined_img, refined_size, hparams.image_resize_algo, .none, .{ 0, 0, 0 });

            var overview: ImageU8 = .{};
            defer overview.deinit();
            try ImgTool.resize(&refined_img, &overview, .{ .width = tile_size, .height = tile_size }, hparams.image_resize_algo, .none, .{ 0, 0, 0 });

            var slices: std.ArrayList(ImageU8) = .empty;
            defer {
                for (slices.items) |*sl| sl.deinit();
                slices.deinit(gpa);
            }
            var y: c_int = 0;
            while (y < grid_y) : (y += 1) {
                var x: c_int = 0;
                while (x < grid_x) : (x += 1) {
                    var slice: ImageU8 = .{};
                    errdefer slice.deinit();
                    try ImgTool.crop(&refined_img, &slice, x * tile_size, y * tile_size, tile_size, tile_size);
                    try slices.append(gpa, slice);
                }
            }

            impl.logDbg("%s: grid size: %d x %d (%d tiles) + overview\n", .{ func, grid_x, grid_y, grid_x * grid_y });

            var output: PreprocOut = .{};
            errdefer output.deinit();
            try output.appendOverview(hparams, &overview, true);
            try output.appendU8s(hparams, slices.items, true);
            output.grid_x = grid_x;
            output.grid_y = grid_y;
            return output;
        }

        const refined_size = ImgTool.calcSizePreservedRatio(original_size, .{
            .align_size = hparams.image_size,
            .min_pixels = @max(0, hparams.image_min_pixels),
            .max_pixels = @max(0, hparams.image_max_pixels),
            .longest_edge = hparams.image_longest_edge,
        });

        var instructions: SliceInstructions = .{};
        defer instructions.deinit();
        instructions.overview_size = .{ .width = hparams.image_size, .height = hparams.image_size };
        instructions.refined_size = refined_size;
        instructions.grid_size = .{
            .width = toI(@ceil(toF(refined_size.width) / toF(hparams.image_size))),
            .height = toI(@ceil(toF(refined_size.height) / toF(hparams.image_size))),
        };
        var y: c_int = 0;
        while (y < refined_size.height) : (y += hparams.image_size) {
            var x: c_int = 0;
            while (x < refined_size.width) : (x += hparams.image_size) {
                try instructions.slices.append(gpa, .{
                    .x = x,
                    .y = y,
                    .size = .{
                        .width = @min(hparams.image_size, refined_size.width - x),
                        .height = @min(hparams.image_size, refined_size.height - y),
                    },
                });
            }
        }
        var sliced = try self.sliceImage(img, &instructions);
        defer sliced.deinit();

        var output: PreprocOut = .{};
        errdefer output.deinit();
        try output.appendOverview(hparams, &sliced.overview, true);
        try output.appendU8s(hparams, sliced.slices.items, true);
        output.grid_x = instructions.grid_size.width;
        output.grid_y = instructions.grid_size.height;
        return output;
    }

    // -------------------------------------------------------------------------
    // cohere2v

    /// Ports `preprocess` (mtmd-image.cpp:1284 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_cohere2v`: the image stretched to a grid of
    /// tiles, slices first, then a thumbnail if there is more than one tile.
    fn cohere2vPreprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        const hparams = self.hparams;
        var inst = try self.getSliceInstructions(img.getSize());
        defer inst.deinit();
        var sliced = try self.sliceImage(img, &inst);
        defer sliced.deinit();

        var output: PreprocOut = .{};
        errdefer output.deinit();
        if (sliced.slices.items.len == 0) {
            try output.appendOverview(hparams, &sliced.overview, true);
            return output;
        }
        // slices first, then thumbnail
        try output.appendU8s(hparams, sliced.slices.items, true);
        try output.appendOverview(hparams, &sliced.overview, true);
        output.grid_x = inst.grid_size.width;
        output.grid_y = inst.grid_size.height;
        return output;
    }

    /// Ports `get_slice_instructions` (mtmd-image.cpp:1301 @cd959bae9), a
    /// member of `mtmd_image_preprocessor_cohere2v`: the grid with the least
    /// upscale, or if every grid needs a downscale, the one with the least
    /// downscale; grids are visited by tile count, then by width, the same
    /// order as HF for ties.
    fn cohere2vGetSliceInstructions(self: *const Preprocessor, original_size: ImageSize) !SliceInstructions {
        const hparams = self.hparams;
        const tile = hparams.image_size;

        var best_down: f64 = -1.0;
        var best_up: f64 = std.math.floatMax(f64);
        var grid_down: ImageSize = .{ .width = 1, .height = 1 };
        var grid_up: ImageSize = .{ .width = 0, .height = 0 };
        var n: c_int = 1;
        while (n <= hparams.preproc_max_tiles) : (n += 1) {
            var w: c_int = 1;
            while (w <= n) : (w += 1) {
                if (@rem(n, w) != 0) continue;
                const g: ImageSize = .{ .width = w, .height = @divTrunc(n, w) };
                // std::min: the first unless the second is less
                const sx = toD(g.width *% tile) / toD(original_size.width);
                const sy = toD(g.height *% tile) / toD(original_size.height);
                const scale = if (sy < sx) sy else sx;
                if (scale < 1.0) {
                    if (scale > best_down) {
                        best_down = scale;
                        grid_down = g;
                    }
                } else if (scale < best_up) {
                    best_up = scale;
                    grid_up = g;
                }
            }
        }
        const grid = if (grid_up.width > 0) grid_up else grid_down;

        var inst: SliceInstructions = .{};
        errdefer inst.deinit();
        inst.overview_size = .{ .width = tile, .height = tile };
        inst.refined_size = .{ .width = tile * grid.width, .height = tile * grid.height };
        inst.grid_size = grid;
        if (grid.width * grid.height > 1) {
            var y: c_int = 0;
            while (y < grid.height) : (y += 1) {
                var x: c_int = 0;
                while (x < grid.width) : (x += 1) {
                    try inst.slices.append(gpa, .{ .x = x * tile, .y = y * tile, .size = .{ .width = tile, .height = tile } });
                }
            }
        }
        return inst;
    }

    /// Ports `preprocess` (mtmd-image.cpp:1350
    /// @cd959bae9), a member of `mtmd_image_preprocessor_internvl`: the slices
    /// first, then the overview.
    fn internvlPreprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        const hparams = self.hparams;
        assert_(hparams.image_res_candidates.items.len != 0, 1351, "!hparams.image_res_candidates.empty()");
        const original_size = img.getSize();
        var inst = try self.getSliceInstructions(original_size);
        defer inst.deinit();
        var sliced = try self.sliceImage(img, &inst);
        defer sliced.deinit();

        var output: PreprocOut = .{};
        errdefer output.deinit();
        // InternVL: slices first, then overview
        try output.appendU8s(hparams, sliced.slices.items, true);
        try output.appendOverview(hparams, &sliced.overview, true);
        output.grid_x = inst.grid_size.width;
        output.grid_y = inst.grid_size.height;
        return output;
    }

    // -------------------------------------------------------------------------
    // deepseek4v
    //
    // DeepSeek-V4-Flash-Vision: port of load_image / safe_resize /
    // solve_resize_ratio / grid_tokens from inference/image_processor.py.
    // The resize solver picks the largest target size (multiple of
    // patch_size) whose LLM token block fits max_n_token.

    /// Ports `grid_info` (mtmd-image.h:184 @cd959bae9), private to
    /// `mtmd_image_preprocessor_deepseek4v`.
    const Dsv4GridInfo = struct {
        n_llm_h: c_int,
        n_llm_w: c_int,
        n_tokens: c_int, // token count of the block (incl. newline/pad rows and start/end, excl. lead pads)
    };

    /// Ports `grid_tokens` (mtmd-image.cpp:1429 @cd959bae9), a static
    /// member of `mtmd_image_preprocessor_deepseek4v`.
    fn dsv4GridTokens(best_height: c_int, best_width: c_int, patch_size: c_int, r: c_int) Dsv4GridInfo {
        var g: Dsv4GridInfo = undefined;
        g.n_llm_h = @divTrunc(@divTrunc(best_height, patch_size) + r - 1, r);
        g.n_llm_w = @divTrunc(@divTrunc(best_width, patch_size) + r - 1, r);
        g.n_tokens = impl.dsv4GetBlockLayout(g.n_llm_w, g.n_llm_h, 0).n_out;
        return g;
    }

    /// Ports `solve_resize_ratio` (mtmd-image.cpp:1438 @cd959bae9), a
    /// static member of `mtmd_image_preprocessor_deepseek4v`.
    fn dsv4SolveResizeRatio(height: c_int, width: c_int, p: c_int, r: c_int, max_n_token: c_int, best_height: *c_int, best_width: *c_int) void {
        const ratio = toD(height) / toD(width);
        const max_w_f = @sqrt(toD(max_n_token - 2) / ratio + 0.25) - 0.5;
        const max_h_f = max_w_f * ratio;
        if (max_w_f < 1.0) {
            const max_w: c_int = 1;
            var max_h = @divTrunc(max_n_token - 2, max_w + 1);
            if (@rem(max_h, 2) == 1) {
                max_h -= 1;
            }
            best_width.* = max_w * p * r;
            best_height.* = max_h * p * r;
        } else if (max_h_f < 2.0) {
            const max_h: c_int = 2;
            // guard tiny budgets; cannot be hit with the current lower bound on max_n_token
            const max_w = @max(@divTrunc(max_n_token - 2, max_h) - 1, 2);
            best_width.* = max_w * p * r;
            best_height.* = max_h * p * r;
        } else {
            const max_w_i: c_int = @intFromFloat(@floor(max_w_f));
            var max_h_i: c_int = @intFromFloat(@floor(max_h_f));
            if (@rem(max_h_i, 2) == 1) {
                max_h_i -= 1;
            }
            const bw = toD(max_w_i) * toD(p) * toD(r) / toD(width);
            const bh = toD(max_h_i) * toD(p) * toD(r) / toD(height);
            const beta = if (bh < bw) bh else bw; // std::min
            best_width.* = @as(c_int, @intFromFloat(@floor(toD(width) * beta / toD(p)))) * p;
            best_height.* = @as(c_int, @intFromFloat(@floor(toD(height) * beta / toD(p)))) * p;
        }
    }

    /// Ports `safe_resize` (mtmd-image.cpp:1472 @cd959bae9), a static
    /// member of `mtmd_image_preprocessor_deepseek4v`.
    fn dsv4SafeResize(height: c_int, width: c_int, best_height: *c_int, best_width: *c_int, p: c_int, r: c_int, max_n_token_in: c_int) void {
        var max_n_token = max_n_token_in;
        max_n_token -= 4 - 1; // reserve room for the position-dependent lead pads (COMPRESS_PAD_TO - 1)
        var g = dsv4GridTokens(best_height.*, best_width.*, p, r);
        var budget = max_n_token;
        while (g.n_tokens > max_n_token) {
            dsv4SolveResizeRatio(height, width, p, r, budget, best_height, best_width);
            g = dsv4GridTokens(best_height.*, best_width.*, p, r);
            budget -= 1;
        }
    }

    /// Ports `preprocess` (mtmd-image.cpp:1485 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_deepseek4v` (`load_image` in the
    /// reference): an aspect-preserving, padded resize to the solver's
    /// size, or a stretch for an extreme aspect ratio.
    fn deepseek4vPreprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        const hparams = self.hparams;
        var out: PreprocOut = .{};
        errdefer out.deinit();

        const p = hparams.patch_size;
        const r = hparams.n_merge;
        const max_n_token = hparams.dsv4_max_n_token;
        const max_wh = hparams.dsv4_max_wh_ratio;

        const orig = img.getSize();
        var width = orig.width;
        var height = orig.height;
        if (max_wh > 0 and width > height * max_wh) {
            width = height * max_wh;
        }
        if (hparams.image_min_pixels > 0 and width * height > 0 and width * height < hparams.image_min_pixels) {
            const up = @sqrt(toD(hparams.image_min_pixels) / (toD(width) * toD(height)));
            width = @intFromFloat(toD(width) * up);
            height = @intFromFloat(toD(height) * up);
        }
        var best_width = clipAlign(width, p);
        var best_height = clipAlign(height, p);
        dsv4SafeResize(height, width, &best_height, &best_width, p, r, max_n_token);

        var resized: ImageU8 = .{};
        defer resized.deinit();
        if (max_wh > 0 and orig.width >= max_wh * orig.height) {
            // extreme aspect ratio: plain stretch resize, no padding
            try ImgTool.resize(img, &resized, .{ .width = best_width, .height = best_height }, hparams.image_resize_algo, .none, .{ 0, 0, 0 });
        } else {
            // aspect-preserving resize + centered padding (PIL ImageOps.pad)
            try ImgTool.resize(img, &resized, .{ .width = best_width, .height = best_height }, hparams.image_resize_algo, .nearest, hparams.image_pad_color);
        }

        try out.appendU8(hparams, &resized, true);
        return out;
    }

    // -------------------------------------------------------------------------
    // deepseek41v
    //
    // DeepSeek-V4.1-Flash: port of plan_image_grid / load_image from
    // inference/image_processor.py (V4.1). It differs from deepseek4v: block =
    // h * (w + 1) + 2 tokens, no even-row rounding, a single solve, no stretch
    // path.

    /// The `n_tokens` lambda of `plan_grid` below.
    fn dsv41NTokens(bh: c_int, bw: c_int, p: c_int, r: c_int) c_int {
        const n_llm_h = @divTrunc(@divTrunc(bh, p) + r - 1, r);
        const n_llm_w = @divTrunc(@divTrunc(bw, p) + r - 1, r);
        return n_llm_h * (n_llm_w + 1) + 2;
    }

    /// Ports `plan_grid` (mtmd-image.cpp:1530 @cd959bae9), a static member of
    /// `mtmd_image_preprocessor_deepseek41v`: the target pixel size (multiples
    /// of `patch_size`) for an image of the given size.
    pub fn deepseek41vPlanGrid(width_in: c_int, height_in: c_int, p: c_int, r: c_int, max_n_token: c_int, min_pixels: c_int, best_width: *c_int, best_height: *c_int) void {
        var width = width_in;
        var height = height_in;
        const area = @as(i64, width) * height;
        if (min_pixels > 0 and area > 0 and area < min_pixels) {
            // python: ratio = (min_pixels / (w * h)) ** 0.5; int(w * ratio)
            const ratio = @sqrt(toD(min_pixels) / (toD(width) * toD(height)));
            width = toI(toD(width) * ratio);
            height = toI(toD(height) * ratio);
        }
        best_width.* = clipAlign(width, p);
        best_height.* = clipAlign(height, p);
        if (dsv41NTokens(best_height.*, best_width.*, p, r) > max_n_token) {
            // solve_resize_ratio: largest aspect-preserving size whose token block fits
            const ratio = toD(height) / toD(width);
            const max_w_f = @sqrt(toD(max_n_token - 2) / ratio + 0.25) - 0.5;
            const max_h_f = max_w_f * ratio;
            const cell = p * r;
            if (max_w_f < 1.0) {
                best_height.* = @divTrunc(max_n_token - 2, 2) * cell;
                best_width.* = cell;
            } else if (max_h_f < 1.0) {
                best_height.* = cell;
                best_width.* = (max_n_token - 3) * cell;
            } else {
                const beta = cMin(@floor(max_w_f) * toD(cell) / toD(width), @floor(max_h_f) * toD(cell) / toD(height));
                best_height.* = toI(@floor(toD(height) * beta / toD(p))) * p;
                best_width.* = toI(@floor(toD(width) * beta / toD(p))) * p;
            }
            assert_(dsv41NTokens(best_height.*, best_width.*, p, r) <= max_n_token, 1562, "n_tokens(best_height, best_width) <= max_n_token");
        }
    }

    /// Ports `preprocess` (mtmd-image.cpp:1566 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_deepseek41v` (`load_image` in the reference):
    /// PIL's `ImageOps.pad(color=127)`, which is `ImageOps.contain` then a
    /// centered paste, with Python's round half to even, and the normalized
    /// input rounded to bf16.
    fn deepseek41vPreprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        const hparams = self.hparams;
        var out: PreprocOut = .{};
        errdefer out.deinit();

        const p = hparams.patch_size;
        const orig = img.getSize();
        var best_width: c_int = 0;
        var best_height: c_int = 0;
        deepseek41vPlanGrid(orig.width, orig.height, p, hparams.n_merge, hparams.dsv4_max_n_token, hparams.image_min_pixels, &best_width, &best_height);

        const R = struct {
            fn halfEven(x: f64) c_int {
                return toI(libm.nearbyint(x));
            }
        };
        var new_w = best_width;
        var new_h = best_height;
        const im_ratio = toD(orig.width) / toD(orig.height);
        const dest_ratio = toD(best_width) / toD(best_height);
        if (im_ratio != dest_ratio) {
            if (im_ratio > dest_ratio) {
                new_h = R.halfEven(toD(orig.height) / toD(orig.width) * toD(best_width));
            } else {
                new_w = R.halfEven(toD(orig.width) / toD(orig.height) * toD(best_height));
            }
        }
        var resized: ImageU8 = .{};
        defer resized.deinit();
        try ImgTool.resize(img, &resized, .{ .width = new_w, .height = new_h }, hparams.image_resize_algo, .none, .{ 0, 0, 0 });

        if (new_w == best_width and new_h == best_height) {
            try out.appendU8(hparams, &resized, true);
        } else {
            var padded: ImageU8 = .{};
            defer padded.deinit();
            try padded.setSize(.{ .width = best_width, .height = best_height }, img.isPlaceholder());
            if (!img.isPlaceholder()) {
                ImgTool.fill(&padded, hparams.image_pad_color);
                const off_x = R.halfEven(toD(best_width - new_w) * 0.5);
                const off_y = R.halfEven(toD(best_height - new_h) * 0.5);
                ImgTool.composite(&padded, &resized, off_x, off_y);
            }
            try out.appendU8(hparams, &padded, true);
        }

        // load_image rounds the normalized input to bf16 (round to nearest even)
        for (out.entries.items) |*e| {
            // `get_ro_buf` throws on a placeholder
            if (e.isPlaceholder()) return clip.throwErr("this clip_image_f32 is a placeholder", .{});
            for (e.buf.items) |*v| v.* = c.ggml_bf16_to_fp32(c.ggml_fp32_to_bf16(v.*));
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // deepseekocr

    /// Ports `preprocess`
    /// (mtmd-image.cpp:1617 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_deepseekocr`: a padded global view, and a
    /// local tile grid for an image larger than one tile.
    fn deepseekocrPreprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        const func = "preprocess";
        const hparams = self.hparams;
        const tile_size = self.tile_size;
        var output: PreprocOut = .{};
        errdefer output.deinit();
        var grid_w: c_int = 0;
        var grid_h: c_int = 0;
        const img_size = img.getSize();

        // global view: aspect-preserving fit-and-pad to base_size
        var padded: ImageU8 = .{};
        defer padded.deinit();
        try ImgTool.resize(img, &padded, .{ .width = self.base_size, .height = self.base_size }, .bicubic, .nearest, hparams.image_pad_color);
        try output.appendOverview(hparams, &padded, true);
        output.overview.add_viewsep = true;

        // if this condition doesn't hold, the output is overview only, no tiles
        if (img_size.width > tile_size or img_size.height > tile_size) {
            const aspect_ratio = toF(img_size.width) / toF(img_size.height);
            var target_ratios = try getTargetRatios(self.min_tiles, self.max_tiles);
            defer target_ratios.deinit(gpa);
            const grid = findClosestAspectRatio(tile_size, aspect_ratio, target_ratios.items, img_size.width, img_size.height);
            grid_w = grid.width;
            grid_h = grid.height;

            var refined: ImageU8 = .{};
            defer refined.deinit();
            try ImgTool.resize(img, &refined, .{ .width = tile_size * grid_w, .height = tile_size * grid_h }, .bicubic, .none, .{ 0, 0, 0 });

            var row: c_int = 0;
            while (row < grid_h) : (row += 1) {
                if (self.fuse_row) {
                    // concat all tiles in this row into one image along the
                    // H axis, w = tile_size, h = tile_size * grid_w, so the
                    // whole row is always processed together
                    var row_img: ImageU8 = .{};
                    defer row_img.deinit();
                    try row_img.setSize(.{ .width = tile_size, .height = tile_size * grid_w }, false);
                    var col: c_int = 0;
                    while (col < grid_w) : (col += 1) {
                        var py: c_int = 0;
                        while (py < tile_size) : (py += 1) {
                            var px: c_int = 0;
                            while (px < tile_size) : (px += 1) {
                                row_img.setPixel(px, col * tile_size + py, refined.getPixel(col * tile_size + px, row * tile_size + py));
                            }
                        }
                    }
                    try output.appendU8(hparams, &row_img, true);
                } else {
                    var col: c_int = 0;
                    while (col < grid_w) : (col += 1) {
                        var tile: ImageU8 = .{};
                        defer tile.deinit();
                        try ImgTool.crop(&refined, &tile, col * tile_size, row * tile_size, tile_size, tile_size);
                        try output.appendU8(hparams, &tile, true);
                    }
                }
            }
            if (self.fuse_row) {
                grid_w = 1; // each fused row is one image; a single output column
            }
        }

        impl.logDbg("%s: grid size: %d x %d (%d tiles) + global view\n", .{ func, grid_w, grid_h, grid_w * grid_h });
        impl.logDbg("%s: overview size: %d x %d\n", .{ func, padded.getSize().width, padded.getSize().height });

        output.grid_x = grid_w;
        output.grid_y = grid_h;
        return output;
    }

    // -------------------------------------------------------------------------
    // step3vl

    /// Ports `img_u8_resize_bilinear_to_f32`
    /// (mtmd-image.cpp:1687 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_step3vl`: normalize, then interpolate.
    fn step3vlResizeBilinearToF32(src: *const ImageU8, dst: *ImageF32, target_width: c_int, target_height: c_int, mean: *const [3]f32, std_: *const [3]f32) !void {
        const src_size = src.getSize();
        if (src_size.width == target_width and src_size.height == target_height) {
            try dst.fromU8(src);
            dst.normalize(mean, std_);
            return;
        }

        try dst.setSize(.{ .width = target_width, .height = target_height }, false, false);

        if (src.isPlaceholder()) {
            // no-op for placeholder image, just set the size and return
            return;
        }

        const scale_x = toF(src_size.width) / toF(target_width);
        const scale_y = toF(src_size.height) / toF(target_height);

        const tw: usize = @intCast(target_width);
        const th: usize = @intCast(target_height);
        const local_buf = try gpa.alloc(f32, 3 * tw * th);
        defer gpa.free(local_buf);
        @memset(local_buf, 0);

        var y: c_int = 0;
        while (y < target_height) : (y += 1) {
            const src_y = @mulAdd(f32, toF(y) + 0.5, scale_y, -0.5);
            const y0_floor = toI(@floor(src_y));
            const y0 = @max(0, @min(y0_floor, src_size.height - 1));
            const y1 = @max(0, @min(y0_floor + 1, src_size.height - 1));
            const ly = src_y - toF(y0_floor);

            var x: c_int = 0;
            while (x < target_width) : (x += 1) {
                const src_x = @mulAdd(f32, toF(x) + 0.5, scale_x, -0.5);
                const x0_floor = toI(@floor(src_x));
                const x0 = @max(0, @min(x0_floor, src_size.width - 1));
                const x1 = @max(0, @min(x0_floor + 1, src_size.width - 1));
                const lx = src_x - toF(x0_floor);

                const p00 = src.getPixel(x0, y0);
                const p01 = src.getPixel(x1, y0);
                const p10 = src.getPixel(x0, y1);
                const p11 = src.getPixel(x1, y1);

                const idx_dst = 3 * (@as(usize, @intCast(y)) * tw + @as(usize, @intCast(x)));
                for (0..3) |ch| {
                    const v00 = (toF(p00[ch]) / 255.0 - mean[ch]) / std_[ch];
                    const v01 = (toF(p01[ch]) / 255.0 - mean[ch]) / std_[ch];
                    const v10 = (toF(p10[ch]) / 255.0 - mean[ch]) / std_[ch];
                    const v11 = (toF(p11[ch]) / 255.0 - mean[ch]) / std_[ch];

                    const top = @mulAdd(f32, v01 - v00, lx, v00);
                    const bot = @mulAdd(f32, v11 - v10, lx, v10);
                    local_buf[idx_dst + ch] = @mulAdd(f32, bot - top, ly, top);
                }
            }
        }
        try dst.cpyBuf(local_buf);
    }

    /// Ports `get_image_longest_edge`
    /// (mtmd-image.cpp:1748 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_step3vl`.
    fn step3vlGetImageLongestEdge(params: *const Hparams) c_int {
        return if (params.image_longest_edge > 0) params.image_longest_edge else step3vl_default_image_longest_edge;
    }

    /// Ports `determine_window_size`
    /// (mtmd-image.cpp:1752 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_step3vl`.
    fn step3vlDetermineWindowSize(params: *const Hparams, longer: c_int, shorter: c_int) c_int {
        const image_size = params.image_size;
        const crop_size = step3vl_default_image_crop_size;
        const aspect_ratio = toF(longer) / toF(shorter);

        if (longer <= image_size) return if (aspect_ratio > step3vl_small_aspect_ratio_limit) shorter else 0;

        return if (aspect_ratio > step3vl_wide_aspect_ratio_limit) @min(shorter, crop_size) else crop_size;
    }

    /// Ports `calc_crop_extent`
    /// (mtmd-image.cpp:1764 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_step3vl`.
    fn step3vlCalcCropExtent(length: c_int, window_size: c_int) c_int {
        const ratio = toF(length) / toF(window_size);
        if (ratio < 1.0) return length;

        const decimal = ratio - @floor(ratio);
        const rounded: c_int = if (decimal > step3vl_crop_rounding_threshold) toI(@floor(ratio)) + 1 else toI(@floor(ratio));
        return window_size * rounded;
    }

    /// Ports `calc_grid` (mtmd-image.cpp:1777
    /// @cd959bae9), a member of `mtmd_image_preprocessor_step3vl`. The caller
    /// owns the result.
    fn step3vlCalcGrid(length: c_int, window_size: c_int) !std.ArrayList(c_int) {
        const n: c_int = if (length <= window_size) 1 else toI(@ceil(toF(length - window_size) / toF(window_size) + 1.0));
        var starts: std.ArrayList(c_int) = .empty;
        errdefer starts.deinit(gpa);
        try starts.resize(gpa, @intCast(n));

        for (starts.items, 0..) |*s, i| s.* = window_size * @as(c_int, @intCast(i));

        if (n > 1 and starts.items[starts.items.len - 1] + window_size > length) {
            starts.items[starts.items.len - 1] = length - window_size;
        }

        return starts;
    }

    /// Ports `prepare_image` (mtmd-image.cpp:1794
    /// @cd959bae9), a member of `mtmd_image_preprocessor_step3vl`. The caller
    /// owns the result.
    fn step3vlPrepareImage(img: *const ImageU8, params: *const Hparams) !ImageU8 {
        var resized = try img.clone();
        errdefer resized.deinit();
        const img_size = img.getSize();
        const aspect_ratio: f32 = if (img_size.height > 0) toF(img_size.width) / toF(img_size.height) else 1.0;
        if (@min(img_size.width, img_size.height) < 32 and
            (aspect_ratio > step3vl_wide_aspect_ratio_limit or aspect_ratio < 1.0 / step3vl_wide_aspect_ratio_limit))
        {
            const square_size = @max(img_size.width, img_size.height);
            var padded: ImageU8 = .{};
            errdefer padded.deinit();
            try padded.setSize(.{ .width = square_size, .height = square_size }, false);
            ImgTool.fill(&padded, .{ 0, 0, 0 });
            ImgTool.composite(&padded, img, 0, 0);
            resized.deinit();
            resized = padded;
        }

        const max_image_size = step3vlGetImageLongestEdge(params);
        const resized_size = resized.getSize();
        if (@max(resized_size.width, resized_size.height) > max_image_size) {
            const scale = toF(max_image_size) / toF(@max(resized_size.width, resized_size.height));
            const new_size: ImageSize = .{
                .width = @max(1, toI(@floor(toF(resized_size.width) * scale))),
                .height = @max(1, toI(@floor(toF(resized_size.height) * scale))),
            };
            var scaled: ImageU8 = .{};
            errdefer scaled.deinit();
            try ImgTool.resize(&resized, &scaled, new_size, .bilinear, .none, .{ 0, 0, 0 });
            resized.deinit();
            resized = scaled;
        }

        return resized;
    }

    /// Ports `crop_with_black_padding`
    /// (mtmd-image.cpp:1825 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_step3vl`. The caller owns the result.
    fn step3vlCropWithBlackPadding(image: *const ImageU8, x: c_int, y: c_int, w: c_int, h: c_int) !ImageU8 {
        var dst: ImageU8 = .{};
        errdefer dst.deinit();
        try dst.setSize(.{ .width = w, .height = h }, false);
        ImgTool.fill(&dst, .{ 0, 0, 0 });

        const img_size = image.getSize();
        const src_x0 = @max(0, x);
        const src_y0 = @max(0, y);
        const src_x1 = @min(img_size.width, x + w);
        const src_y1 = @min(img_size.height, y + h);

        if (src_x0 >= src_x1 or src_y0 >= src_y1) return dst;

        const dst_x0 = src_x0 - x;
        const dst_y0 = src_y0 - y;

        var yy: c_int = 0;
        while (yy < src_y1 - src_y0) : (yy += 1) {
            var xx: c_int = 0;
            while (xx < src_x1 - src_x0) : (xx += 1) {
                dst.setPixel(dst_x0 + xx, dst_y0 + yy, image.getPixel(src_x0 + xx, src_y0 + yy));
            }
        }

        return dst;
    }

    /// Ports `build_slice_instructions`
    /// (mtmd-image.cpp:1852 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_step3vl`. The caller owns the result.
    pub fn step3vlBuildSliceInstructions(params: *const Hparams, prepared_size: ImageSize) !SliceInstructions {
        var instructions: SliceInstructions = .{};
        errdefer instructions.deinit();
        instructions.overview_size = prepared_size;

        const window_size = step3vlDetermineWindowSize(params, @max(prepared_size.width, prepared_size.height), @min(prepared_size.width, prepared_size.height));
        if (window_size <= 0) {
            instructions.refined_size = .{ .width = 0, .height = 0 };
            instructions.grid_size = .{ .width = 0, .height = 0 };
            return instructions;
        }

        const crop_width = step3vlCalcCropExtent(prepared_size.width, window_size);
        const crop_height = step3vlCalcCropExtent(prepared_size.height, window_size);
        instructions.refined_size = .{ .width = crop_width, .height = crop_height };

        var xs = try step3vlCalcGrid(crop_width, window_size);
        defer xs.deinit(gpa);
        var ys = try step3vlCalcGrid(crop_height, window_size);
        defer ys.deinit(gpa);
        instructions.grid_size = .{ .width = @intCast(xs.items.len), .height = @intCast(ys.items.len) };

        for (ys.items) |y| {
            for (xs.items) |x| {
                try instructions.slices.append(gpa, .{ .x = x, .y = y, .size = .{ .width = window_size, .height = window_size } });
            }
        }

        return instructions;
    }

    /// Ports `preprocess` (mtmd-image.cpp:1892
    /// @cd959bae9), a member of `mtmd_image_preprocessor_step3vl`.
    fn step3vlPreprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        const hparams = self.hparams;
        var prepared = try step3vlPrepareImage(img, hparams);
        defer prepared.deinit();
        var instructions = try step3vlBuildSliceInstructions(hparams, prepared.getSize());
        defer instructions.deinit();

        var output: PreprocOut = .{};
        errdefer output.deinit();
        // overview (normalized f32, already includes mean/std)
        try step3vlResizeBilinearToF32(&prepared, &output.overview, hparams.image_size, hparams.image_size, &hparams.image_mean, &hparams.image_std);

        if (instructions.slices.items.len == 0) {
            output.grid_x = 0;
            output.grid_y = 0;
            return output;
        }

        var img_for_crop = try prepared.clone();
        defer img_for_crop.deinit();
        const prepared_size = prepared.getSize();
        if (instructions.refined_size.width != prepared_size.width or instructions.refined_size.height != prepared_size.height) {
            var refined: ImageU8 = .{};
            errdefer refined.deinit();
            try ImgTool.resize(&prepared, &refined, instructions.refined_size, .bilinear, .none, .{ 0, 0, 0 });
            img_for_crop.deinit();
            img_for_crop = refined;
        }

        const crop_size = step3vl_default_image_crop_size;
        for (instructions.slices.items) |slice| {
            // pad the part of a patch past the source image with black
            var patch = try step3vlCropWithBlackPadding(&img_for_crop, slice.x, slice.y, slice.size.width, slice.size.height);
            defer patch.deinit();

            var patch_f32: ImageF32 = .{};
            defer patch_f32.deinit();
            try step3vlResizeBilinearToF32(&patch, &patch_f32, crop_size, crop_size, &hparams.image_mean, &hparams.image_std);
            try output.appendF32(hparams, &patch_f32, false);
        }

        output.grid_x = instructions.grid_size.width;
        output.grid_y = instructions.grid_size.height;

        return output;
    }

    // -------------------------------------------------------------------------
    // youtuvl, granite, muse_glimmer

    /// Ports `preprocess` (mtmd-image.cpp:1946
    /// @cd959bae9), a member of `mtmd_image_preprocessor_youtuvl`: a linear
    /// search, in steps of 0.02, for the scale that fits `max_num_patches`.
    fn youtuvlPreprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        const hparams = self.hparams;
        const patch_size = hparams.patch_size; // typically 16
        const merge_size = hparams.n_merge; // typically 2
        const align_size = patch_size * merge_size; // 32

        const max_num_patches: c_int = if (hparams.image_max_pixels > 0) @divTrunc(hparams.image_max_pixels, patch_size * patch_size) else 256;

        const img_size = img.getSize();
        var scale: f32 = 1.0;
        var target_height = img_size.height;
        var target_width = img_size.width;

        const S = struct {
            fn scaled(al: c_int, sc: f32, size: c_int) c_int {
                const scaled_size = toF(size) * sc;
                // Round up to nearest multiple of align_size
                const aligned = toI(@ceil(scaled_size / toF(al))) * al;
                // Ensure at least one patch
                return @max(al, aligned);
            }
        };

        while (scale > 0.0) {
            target_height = S.scaled(align_size, scale, img_size.height);
            target_width = S.scaled(align_size, scale, img_size.width);

            const num_patches_h = @divTrunc(target_height, patch_size);
            const num_patches_w = @divTrunc(target_width, patch_size);
            const num_patches = num_patches_h *% num_patches_w;

            if (num_patches > max_num_patches) {
                scale -= 0.02;
            } else {
                break;
            }
        }

        const new_size: ImageSize = .{ .width = target_width, .height = target_height };

        var resized: ImageU8 = .{};
        defer resized.deinit();
        // the pad colour is the default, black, not `image_pad_color`
        try ImgTool.resize(img, &resized, new_size, hparams.image_resize_algo, hparams.image_resize_pad, .{ 0, 0, 0 });

        var output: PreprocOut = .{};
        errdefer output.deinit();
        try output.appendU8(hparams, &resized, true);
        return output;
    }

    /// Ports `preprocess` (mtmd-image.cpp:1995
    /// @cd959bae9), a member of `mtmd_image_preprocessor_granite`: llava-next
    /// "anyres", the overview and every tile stacked on the Y axis so the
    /// whole grid goes through one graph.
    fn granitePreprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        const func = "preprocess";
        const hparams = self.hparams;
        assert_(hparams.image_res_candidates.items.len != 0, 1996, "!hparams.image_res_candidates.empty()");

        const orig_size = img.getSize();
        const tile_size = hparams.image_size;
        assert_(tile_size > 0, 2000, "tile_size > 0");

        // llava-next always encodes an overview plus a grid of tiles, even for small images
        const refined_size = selectBestResolution(orig_size, hparams.image_res_candidates.items);
        const grid_x = @divTrunc(refined_size.width, tile_size);
        const grid_y = @divTrunc(refined_size.height, tile_size);

        // the tiles are stacked on the Y axis, a big grid overflows the stacked image height
        assert_(grid_x >= 0 and grid_x <= 1024 and grid_y >= 0 and grid_y <= 1024, 2008, "grid_x >= 0 && grid_x <= 1024 && grid_y >= 0 && grid_y <= 1024");

        var overview: ImageU8 = .{};
        defer overview.deinit();
        try ImgTool.resize(img, &overview, .{ .width = tile_size, .height = tile_size }, hparams.image_resize_algo_ov, hparams.image_pad_ov, hparams.image_pad_color_ov);

        var refined: ImageU8 = .{};
        defer refined.deinit();
        try ImgTool.resize(img, &refined, refined_size, hparams.image_resize_algo_rf, hparams.image_pad_rf, hparams.image_pad_color_rf);

        var stacked: ImageU8 = .{};
        defer stacked.deinit();
        try stacked.setSize(.{ .width = tile_size, .height = tile_size * (1 + grid_x * grid_y) }, false);
        const T = struct {
            fn copyTile(dst: *ImageU8, ts: c_int, src: *const ImageU8, src_x: c_int, src_y: c_int, dst_idx: c_int) void {
                var py: c_int = 0;
                while (py < ts) : (py += 1) {
                    var px: c_int = 0;
                    while (px < ts) : (px += 1) dst.setPixel(px, dst_idx * ts + py, src.getPixel(src_x + px, src_y + py));
                }
            }
        };
        T.copyTile(&stacked, tile_size, &overview, 0, 0, 0);
        var ty: c_int = 0;
        while (ty < grid_y) : (ty += 1) {
            var tx: c_int = 0;
            while (tx < grid_x) : (tx += 1) T.copyTile(&stacked, tile_size, &refined, tx * tile_size, ty * tile_size, 1 + ty * grid_x + tx);
        }

        impl.logDbg("%s: grid size: %d x %d (%d tiles) + overview\n", .{ func, grid_x, grid_y, grid_x * grid_y });

        var output: PreprocOut = .{};
        errdefer output.deinit();
        try output.appendU8(hparams, &stacked, true);
        const entry = &output.entries.items[output.entries.items.len - 1];
        entry.anyres.grid_x = grid_x;
        entry.anyres.grid_y = grid_y;
        entry.anyres.orig_nx = orig_size.width;
        entry.anyres.orig_ny = orig_size.height;
        return output;
    }

    /// Ports `preprocess`
    /// (mtmd-image.cpp:2090 @cd959bae9), a member of
    /// `mtmd_image_preprocessor_muse_glimmer`: the patch grid closest to the
    /// input aspect ratio under the token cap, stretched to.
    fn museGlimmerPreprocess(self: *const Preprocessor, img: *const ImageU8) !PreprocOut {
        const hparams = self.hparams;
        const patch_hw = hparams.patch_size * hparams.n_merge;
        const patch_area = hparams.patch_size * hparams.patch_size * hparams.n_merge * hparams.n_merge;
        assert_(patch_area > 0 and hparams.image_max_pixels > 0, 2093, "patch_area > 0 && hparams.image_max_pixels > 0");
        const max_tokens = @divTrunc(hparams.image_max_pixels, patch_area);

        const original_size = img.getSize();
        const target_size = museGlimmerGridSize(original_size.width, original_size.height, patch_hw, max_tokens);

        // PIL resizes directly to (target_w, target_h) -- a stretch, no padding.
        var resized_image: ImageU8 = .{};
        defer resized_image.deinit();
        try ImgTool.resize(img, &resized_image, target_size, hparams.image_resize_algo, .none, .{ 0, 0, 0 });

        var output: PreprocOut = .{};
        errdefer output.deinit();
        try output.appendU8(hparams, &resized_image, true);
        return output;
    }
};

/// Ports `muse_glimmer_grid_size` (mtmd-image.cpp:2052 @cd959bae9):
/// transformers' `get_aspect_ratio_preserving_size`.
fn museGlimmerGridSize(img_w: c_int, img_h: c_int, patch_hw: c_int, max_tokens: c_int) ImageSize {
    var i_nph = toD(img_h) / toD(patch_hw);
    var i_npw = toD(img_w) / toD(patch_hw);
    const ratio: f64 = if (i_nph > 0.0) i_npw / i_nph else 1.0;
    if (i_nph * i_npw > toD(max_tokens)) {
        i_nph = @sqrt(toD(max_tokens) / ratio);
        i_npw = i_nph * ratio;
    }
    const hs = [2]c_int{ toI(@floor(i_nph)), toI(@ceil(i_nph)) };
    const ws = [2]c_int{ toI(@floor(i_npw)), toI(@ceil(i_npw)) };
    const target_ar = toD(img_h) / toD(img_w);
    var best_nph: c_int = -1;
    var best_npw: c_int = -1;
    var best_d: f64 = 0.0;
    for (0..2) |a| {
        for (0..2) |b| {
            const nph = hs[a];
            const npw = ws[b];
            if (nph < 1 or npw < 1 or nph *% npw > max_tokens) continue;
            const d = @abs(toD(nph) / toD(npw) - target_ar);
            const n_tokens = nph *% npw;
            const best_n_tokens = best_nph *% best_npw;
            if (best_nph < 0 or d < best_d or (d == best_d and n_tokens > best_n_tokens)) {
                best_nph = nph;
                best_npw = npw;
                best_d = d;
            }
        }
    }
    if (best_nph < 0) { // no candidate fit under the cap: round and clamp
        best_nph = @max(1, toI(@round(i_nph)));
        best_npw = @max(1, toI(@round(i_npw)));
    }
    return .{ .width = best_npw * patch_hw, .height = best_nph * patch_hw };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "calc_size_preserved_ratio aligns and caps" {
    const s = ImgTool.calcSizePreservedRatio(.{ .width = 1000, .height = 500 }, .{ .align_size = 28, .min_pixels = 0, .max_pixels = 0 });
    try std.testing.expectEqual(@as(c_int, 1008), s.width);
    try std.testing.expectEqual(@as(c_int, 504), s.height);
    const z = ImgTool.calcSizePreservedRatio(.{ .width = 0, .height = 5 }, .{});
    try std.testing.expectEqual(@as(c_int, 0), z.width);
}

test "a resize to the same size copies, and one to a new size has its bytes" {
    var src: ImageU8 = .{};
    defer src.deinit();
    try src.setSize(.{ .width = 4, .height = 3 }, false);
    for (src.buf.items, 0..) |*b, i| b.* = @intCast(i * 7 % 256);
    var dst: ImageU8 = .{};
    defer dst.deinit();
    try ImgTool.resize(&src, &dst, .{ .width = 4, .height = 3 }, .bicubic, .ceil, .{ 0, 0, 0 });
    try std.testing.expectEqualSlices(u8, src.buf.items, dst.buf.items);
    try ImgTool.resize(&src, &dst, .{ .width = 7, .height = 2 }, .lanczos, .none, .{ 0, 0, 0 });
    try std.testing.expectEqual(@as(usize, 7 * 2 * 3), dst.buf.items.len);
}

test "round_half_even parts from std::round on an exact half" {
    const opts: ImgTool.CalcSizeOpt = .{ .align_size = 32, .min_pixels = 8 * 1024, .max_pixels = 4096 * 1024 };
    var even = opts;
    even.round_half_even = true;
    const size: ImageSize = .{ .width = 400, .height = 400 }; // 12.5 alignments
    try std.testing.expectEqual(@as(c_int, 416), ImgTool.calcSizePreservedRatio(size, opts).width);
    try std.testing.expectEqual(@as(c_int, 384), ImgTool.calcSizePreservedRatio(size, even).width);
}

test "qwen3vl's smart_resize keeps an edge that rounds to zero at the factor" {
    const s = Preprocessor.qwen3vlSmartResize(3, 1000, 32, 0, 1 << 20, 1, 1);
    try std.testing.expectEqual(@as(c_int, 32), s.width);
    try std.testing.expectEqual(@as(c_int, 992), s.height);
}

test "only qwen3vl and lfm3vl have a video frame size, and a frame defaults to preprocess" {
    var hparams: Hparams = .{};
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    var out: ImageSize = .{ .width = -1, .height = -1 };
    const dyn = Preprocessor.init(.dyn_size, &hparams, .mlp);
    try std.testing.expect(!dyn.getVideoFrameSize(4, 640, 480, &out));
    try std.testing.expectEqual(@as(c_int, -1), out.width);
    const lfm3vl = Preprocessor.init(.lfm3vl, &hparams, .lfm3vl);
    try std.testing.expect(lfm3vl.getVideoFrameSize(4, 640, 480, &out));
    try std.testing.expectEqual(@as(c_int, 640), out.width);
    try std.testing.expectEqual(@as(c_int, 480), out.height);
    out = .{ .width = -1, .height = -1 };
    const qwen3vl = Preprocessor.init(.qwen3vl, &hparams, .qwen3vl);
    try std.testing.expect(qwen3vl.getVideoFrameSize(4, 640, 480, &out));
    try std.testing.expectEqual(@as(c_int, 640), out.width);
    try std.testing.expectEqual(@as(c_int, 480), out.height);
}

test "deepseek41v's plan_grid stays within the token budget" {
    var w: c_int = 0;
    var h: c_int = 0;
    Preprocessor.deepseek41vPlanGrid(4000, 3000, 16, 2, 1024, 0, &w, &h);
    try std.testing.expect(dsv41Tokens(w, h) <= 1024);
    // past the flat limit: one row of cells
    Preprocessor.deepseek41vPlanGrid(5000, 1, 16, 2, 16, 0, &w, &h);
    try std.testing.expectEqual(@as(c_int, 32), h);
    try std.testing.expectEqual(@as(c_int, 13 * 32), w);
}

fn dsv41Tokens(w: c_int, h: c_int) c_int {
    return Preprocessor.dsv41NTokens(h, w, 16, 2);
}
