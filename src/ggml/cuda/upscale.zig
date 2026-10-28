//! `GGML_OP_UPSCALE`: nearest, bilinear (optionally antialiased) and
//! bicubic resampling.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/upscale.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/upscale.cuh` at v0.4.0 (`5266f24da`). The
//! kernels -- `upscale_f32`, `upscale_f32_bilinear`,
//! `upscale_f32_bilinear_antialias`, `upscale_f32_bicubic` -- stay CUDA.
//!
//! # `int` arithmetic
//!
//! The launchers take every stride and extent as `int` and compute
//! `dst_size = ne10 * ne11 * ne12 * ne13` in `int` before widening, so the
//! product wraps the way the C's does on this target (`*%`).

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_UPSCALE_BLOCK_SIZE` (upscale.cuh:3 @5266f24da).
const block_size = 256;

fn params(dst_size: i64, stream: cu.cudaStream_t) kernels.LaunchParams {
    const num_blocks: i64 = @divTrunc(dst_size + block_size - 1, block_size);
    return .{
        .grid = .{ .x = @truncate(@as(u64, @bitCast(num_blocks))) },
        .block = .{ .x = block_size },
        .stream = stream,
    };
}

/// Ports `upscale_f32_cuda` (upscale.cu:218 @5266f24da), which launches with
/// `<<<>>>`.
fn upscaleF32Cuda(x: ?*const anyopaque, dst: ?*anyopaque, nb00: c_int, nb01: c_int, nb02: c_int, nb03: c_int, ne10: c_int, ne11: c_int, ne12: c_int, ne13: c_int, sf0: f32, sf1: f32, sf2: f32, sf3: f32, stream: cu.cudaStream_t) void {
    const dst_size: i64 = ne10 *% ne11 *% ne12 *% ne13;
    kernels.launch(kernels.get("upscale", "upscale_f32"), params(dst_size, stream), .{ x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3 });
}

/// Ports `upscale_f32_bilinear_cuda` (upscale.cu:229 @5266f24da), which
/// launches with `<<<>>>`.
fn upscaleF32BilinearCuda(x: ?*const anyopaque, dst: ?*anyopaque, nb00: c_int, nb01: c_int, nb02: c_int, nb03: c_int, ne00_src: c_int, ne01_src: c_int, ne10_dst: c_int, ne11_dst: c_int, ne12_dst: c_int, ne13_dst: c_int, sf0: f32, sf1: f32, sf2: f32, sf3: f32, pixel_offset: f32, antialias: bool, stream: cu.cudaStream_t) void {
    const dst_size: i64 = ne10_dst *% ne11_dst *% ne12_dst *% ne13_dst;
    const args = .{ x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset };
    if (antialias) {
        kernels.launch(kernels.get("upscale", "upscale_f32_bilinear_antialias"), params(dst_size, stream), args);
    } else {
        kernels.launch(kernels.get("upscale", "upscale_f32_bilinear"), params(dst_size, stream), args);
    }
}

/// Ports `upscale_f32_bicubic_cuda` (upscale.cu:245 @5266f24da), which
/// launches with `<<<>>>`.
fn upscaleF32BicubicCuda(x: ?*const anyopaque, dst: ?*anyopaque, nb00: c_int, nb01: c_int, nb02: c_int, nb03: c_int, ne00_src: c_int, ne01_src: c_int, ne10_dst: c_int, ne11_dst: c_int, ne12_dst: c_int, ne13_dst: c_int, sf0: f32, sf1: f32, sf2: f32, sf3: f32, pixel_offset: f32, stream: cu.cudaStream_t) void {
    const dst_size: i64 = ne10_dst *% ne11_dst *% ne12_dst *% ne13_dst;
    kernels.launch(kernels.get("upscale", "upscale_f32_bicubic"), params(dst_size, stream), .{ x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset });
}

fn int(x: anytype) c_int {
    return @truncate(@as(i64, @bitCast(@as(u64, @intCast(x)))));
}

fn int64(x: i64) c_int {
    return @truncate(x);
}

/// `(float) a / b` with `b` an `int64_t`: both sides become `float`.
fn ratio(a: i64, b: i64) f32 {
    return @as(f32, @floatFromInt(a)) / @as(f32, @floatFromInt(b));
}

/// Ports `ggml_cuda_op_upscale` (upscale.cu:257 @5266f24da).
pub fn opUpscale(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");

    const mode_flags: c_int = dst.op_params[0];
    const mode = mode_flags & 0xFF;

    var sf0 = ratio(dst.ne[0], src0.ne[0]);
    var sf1 = ratio(dst.ne[1], src0.ne[1]);
    const sf2 = ratio(dst.ne[2], src0.ne[2]);
    const sf3 = ratio(dst.ne[3], src0.ne[3]);

    var pixel_offset: f32 = 0.5;
    if ((mode_flags & c.GGML_SCALE_FLAG_ALIGN_CORNERS) != 0) {
        sf0 = if (dst.ne[0] > 1 and src0.ne[0] > 1) ratio(dst.ne[0] - 1, src0.ne[0] - 1) else sf0;
        sf1 = if (dst.ne[1] > 1 and src0.ne[1] > 1) ratio(dst.ne[1] - 1, src0.ne[1] - 1) else sf1;
        pixel_offset = 0.0;
    }

    const nb = [4]c_int{ int(src0.nb[0]), int(src0.nb[1]), int(src0.nb[2]), int(src0.nb[3]) };
    const ne = [4]c_int{ int64(dst.ne[0]), int64(dst.ne[1]), int64(dst.ne[2]), int64(dst.ne[3]) };

    if (mode == c.GGML_SCALE_MODE_NEAREST) {
        upscaleF32Cuda(src0_d, dst_d, nb[0], nb[1], nb[2], nb[3], ne[0], ne[1], ne[2], ne[3], sf0, sf1, sf2, sf3, stream);
    } else if (mode == c.GGML_SCALE_MODE_BILINEAR) {
        const antialias = (mode_flags & c.GGML_SCALE_FLAG_ANTIALIAS) != 0;
        upscaleF32BilinearCuda(src0_d, dst_d, nb[0], nb[1], nb[2], nb[3], int64(src0.ne[0]), int64(src0.ne[1]), ne[0], ne[1], ne[2], ne[3], sf0, sf1, sf2, sf3, pixel_offset, antialias, stream);
    } else if (mode == c.GGML_SCALE_MODE_BICUBIC) {
        upscaleF32BicubicCuda(src0_d, dst_d, nb[0], nb[1], nb[2], nb[3], int64(src0.ne[0]), int64(src0.ne[1]), ne[0], ne[1], ne[2], ne[3], sf0, sf1, sf2, sf3, pixel_offset, stream);
    }
}
