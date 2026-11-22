//! Dequantization and float conversion: the `ggml_get_to_{fp32,fp16,bf16}`
//! getters and their non-contiguous `_nc` variants.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/convert.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/convert.cuh` at v1.0.0 (`b1b770d90`). The
//! `dequantize_block*`, `convert_unary` and `convert_unary_cont_vec4`
//! kernels stay CUDA.
//!
//! # Function pointers
//!
//! The C returns a pointer to a template instantiation, which the caller
//! later invokes. Here each getter returns a pointer to a Zig function
//! generated per instantiation; the destination element type only selects
//! the kernel, so the pointer type erases it (`?*anyopaque`).

const ggml = @import("../module.zig");
/// Ports `CUDA_DEQUANTIZE_BLOCK_SIZE` (convert.cuh:4 @b1b770d90).
const dequantize_block_size = 256;

/// Ports `CUDA_Q8_0_NE_ALIGN` (convert.cu:6 @b1b770d90).
const q8_0_ne_align = 2048;

/// Ports `to_t_cuda_t` (convert.cuh:7 @b1b770d90), with the element type erased.
pub const ToFn = *const fn (x: ?*const anyopaque, y: ?*anyopaque, k: i64, stream: ggml.cuda.common.cu.cudaStream_t) void;

/// Ports `to_t_nc_cuda_t` (convert.cuh:22 @b1b770d90), with the element type erased.
pub const ToNcFn = *const fn (x: ?*const anyopaque, y: ?*anyopaque, ne00: i64, ne01: i64, ne02: i64, ne03: i64, s01: i64, s02: i64, s03: i64, stream: ggml.cuda.common.cu.cudaStream_t) void;

/// The `dequantize_kernel_t` a `dequantize_block` is instantiated with:
/// its name and its `qk`/`qr`.
const Deq = struct { name: []const u8, qk: comptime_int, qr: comptime_int };

const q1_0: Deq = .{ .name = "dequantize_q1_0", .qk = ggml.impl.ggml_c.QK1_0, .qr = 1 };
const q2_0: Deq = .{ .name = "dequantize_q2_0", .qk = ggml.impl.ggml_c.QK2_0, .qr = 1 };
const q4_0: Deq = .{ .name = "dequantize_q4_0", .qk = ggml.impl.ggml_c.QK4_0, .qr = 2 };
const q4_1: Deq = .{ .name = "dequantize_q4_1", .qk = ggml.impl.ggml_c.QK4_1, .qr = 2 };
const q5_0: Deq = .{ .name = "dequantize_q5_0", .qk = ggml.impl.ggml_c.QK5_0, .qr = 2 };
const q5_1: Deq = .{ .name = "dequantize_q5_1", .qk = ggml.impl.ggml_c.QK5_1, .qr = 2 };
const q8_0: Deq = .{ .name = "dequantize_q8_0", .qk = ggml.impl.ggml_c.QK8_0, .qr = 1 };

fn i32Of(x: i64) c_int {
    return @truncate(x);
}

/// Ports `dequantize_block_cuda` (convert.cu:252 @b1b770d90).
fn DequantizeBlock(comptime d: Deq, comptime T: type) type {
    return struct {
        const k = ggml.cuda.kernels.get("convert", ggml.cuda.kernels.tmpl("dequantize_block", &.{ ggml.cuda.kernels.int(d.qk), ggml.cuda.kernels.int(d.qr), "&" ++ d.name, ggml.cuda.kernels.typeName(T) }));

        fn nc(vx: ?*const anyopaque, y: ?*anyopaque, ne00: i64, ne01: i64, ne02: i64, ne03: i64, s01: i64, s02: i64, s03: i64, stream: ggml.cuda.common.cu.cudaStream_t) void {
            const ne0203 = ne02 * ne03;
            const ne02_fdv = ggml.cuda.common.initFastdivValues(@bitCast(ne02));
            ggml.cuda.kernels.launch(k, .{
                .grid = .{
                    .x = @truncate(@as(u64, @bitCast(@divTrunc(ne00 + 2 * dequantize_block_size - 1, 2 * dequantize_block_size)))),
                    .y = @bitCast(i32Of(@min(ne01, 65535))),
                    .z = @bitCast(i32Of(@min(ne0203, 65535))),
                },
                .block = .{ .x = dequantize_block_size },
                .stream = stream,
            }, .{ vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03 });
        }

        /// Ports `dequantize_block_cont_cuda` (convert.cu:263 @b1b770d90).
        fn cont(vx: ?*const anyopaque, y: ?*anyopaque, kk: i64, stream: ggml.cuda.common.cu.cudaStream_t) void {
            nc(vx, y, kk, 1, 1, 1, @divTrunc(kk, d.qk), @divTrunc(kk, d.qk), @divTrunc(kk, d.qk), stream);
        }
    };
}

/// Ports `dequantize_block_q8_0_f16_cuda` (convert.cu:267 @b1b770d90).
fn dequantizeBlockQ8_0F16(vx: ?*const anyopaque, y: ?*anyopaque, k: i64, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const num_blocks: c_int = @truncate(@divTrunc(k + q8_0_ne_align - 1, q8_0_ne_align));
    const p: ggml.cuda.kernels.LaunchParams = .{ .grid = .{ .x = @bitCast(num_blocks) }, .block = .{ .x = ggml.cuda.common.warp_size }, .stream = stream };
    if (@rem(k, q8_0_ne_align) == 0) {
        ggml.cuda.kernels.launch(ggml.cuda.kernels.get("convert", "dequantize_block_q8_0_f16<(bool)0>"), p, .{ vx, y, k });
    } else {
        ggml.cuda.kernels.launch(ggml.cuda.kernels.get("convert", "dequantize_block_q8_0_f16<(bool)1>"), p, .{ vx, y, k });
    }
}

/// How a `dequantize_row_*_cuda` sizes its grid.
const RowGrid = enum {
    /// `nb = k / QK_K`.
    per_block,
    /// `nb = (k + QK_K - 1) / QK_K`.
    per_block_ceil,
    /// `nb32 = k / 32`, `nb = (k + 255) / 256`, and `nb32` is passed.
    q4,
    /// `nb32 = k / QK4_NL`, `nb = (k + QK_K - 1) / QK_K`, and `nb32` is
    /// passed: `iq4_nl`.
    per_block_ceil_nb32,
};

/// Ports the `dequantize_row_*_cuda` launchers from `dequantize_row_q2_K_cuda`
/// (convert.cu:279 @b1b770d90) to `dequantize_row_mxfp4_cuda` (convert.cu:378
/// @b1b770d90): one `<<<nb, threads>>>` launch of `dequantize_block_<name>`.
/// `dequantize_row_iq4_nl_cuda` (convert.cu:359 @b1b770d90) passes its
/// block count, so the kernel can stop at a row's end.
fn DequantizeRow(comptime name: []const u8, comptime threads: comptime_int, comptime grid: RowGrid, comptime T: type) type {
    return struct {
        const k = ggml.cuda.kernels.get("convert", ggml.cuda.kernels.tmpl("dequantize_block_" ++ name, &.{ggml.cuda.kernels.typeName(T)}));

        fn run(vx: ?*const anyopaque, y: ?*anyopaque, kk: i64, stream: ggml.cuda.common.cu.cudaStream_t) void {
            switch (grid) {
                .per_block => {
                    const nb: c_int = @truncate(@divTrunc(kk, ggml.impl.ggml_c.QK_K));
                    ggml.cuda.kernels.launch(k, .{ .grid = .{ .x = @bitCast(nb) }, .block = .{ .x = threads }, .stream = stream }, .{ vx, y });
                },
                .per_block_ceil => {
                    const nb: c_int = @truncate(@divTrunc(kk + ggml.impl.ggml_c.QK_K - 1, ggml.impl.ggml_c.QK_K));
                    ggml.cuda.kernels.launch(k, .{ .grid = .{ .x = @bitCast(nb) }, .block = .{ .x = threads }, .stream = stream }, .{ vx, y });
                },
                .q4 => {
                    const nb32: c_int = @truncate(@divTrunc(kk, 32));
                    const nb: c_int = @truncate(@divTrunc(kk + 255, 256));
                    ggml.cuda.kernels.launch(k, .{ .grid = .{ .x = @bitCast(nb) }, .block = .{ .x = threads }, .stream = stream }, .{ vx, y, nb32 });
                },
                .per_block_ceil_nb32 => {
                    const nb32: c_int = @truncate(@divTrunc(kk, ggml.impl.ggml_c.QK4_NL));
                    const nb: c_int = @truncate(@divTrunc(kk + ggml.impl.ggml_c.QK_K - 1, ggml.impl.ggml_c.QK_K));
                    ggml.cuda.kernels.launch(k, .{ .grid = .{ .x = @bitCast(nb) }, .block = .{ .x = threads }, .stream = stream }, .{ vx, y, nb32 });
                },
            }
        }
    };
}

/// Ports `dequantize_row_nvfp4_cuda` (convert.cu:413 @b1b770d90).
fn DequantizeRowNvfp4(comptime T: type) type {
    return struct {
        const k = ggml.cuda.kernels.get("convert", ggml.cuda.kernels.tmpl("dequantize_block_nvfp4", &.{ggml.cuda.kernels.typeName(T)}));

        fn run(vx: ?*const anyopaque, y: ?*anyopaque, kk: i64, stream: ggml.cuda.common.cu.cudaStream_t) void {
            ggml.impl.assert(@rem(kk, ggml.impl.ggml_c.QK_NVFP4) == 0, "k % QK_NVFP4 == 0");
            const nb: c_int = @truncate(@divTrunc(kk, ggml.impl.ggml_c.QK_NVFP4));
            ggml.cuda.kernels.launch(k, .{ .grid = .{ .x = @bitCast(nb) }, .block = .{ .x = 32 }, .stream = stream }, .{ vx, y, kk });
        }
    };
}

/// Ports `convert_unary_cuda` (convert.cu:472 @b1b770d90) and
/// `convert_unary_cont_cuda` (convert.cu:483 @b1b770d90). A contiguous
/// conversion whose length is a multiple of four and whose two buffers are
/// aligned to four elements (`alignof(cvt_vec4<T>)`, convert.cu:448
/// @b1b770d90) moves four elements per thread.
fn ConvertUnary(comptime S: type, comptime T: type) type {
    return struct {
        const k = ggml.cuda.kernels.get("convert", ggml.cuda.kernels.tmpl("convert_unary", &.{ ggml.cuda.kernels.typeName(S), ggml.cuda.kernels.typeName(T) }));

        fn nc(vx: ?*const anyopaque, y: ?*anyopaque, ne00: i64, ne01: i64, ne02: i64, ne03: i64, s01: i64, s02: i64, s03: i64, stream: ggml.cuda.common.cu.cudaStream_t) void {
            const ne0203 = ne02 * ne03;
            const ne02_fdv = ggml.cuda.common.initFastdivValues(@bitCast(ne02));
            ggml.cuda.kernels.launch(k, .{
                .grid = .{
                    .x = @truncate(@as(u64, @bitCast(@divTrunc(ne00 + dequantize_block_size - 1, dequantize_block_size)))),
                    .y = @bitCast(i32Of(@min(ne01, 65535))),
                    .z = @bitCast(i32Of(@min(ne0203, 65535))),
                },
                .block = .{ .x = dequantize_block_size },
                .stream = stream,
            }, .{ vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03 });
        }

        fn cont(vx: ?*const anyopaque, y: ?*anyopaque, kk: i64, stream: ggml.cuda.common.cu.cudaStream_t) void {
            if (@rem(kk, 4) == 0 and
                @intFromPtr(vx) % (@sizeOf(S) * 4) == 0 and
                @intFromPtr(y) % (@sizeOf(T) * 4) == 0)
            {
                const k_vec4 = comptime ggml.cuda.kernels.get("convert", ggml.cuda.kernels.tmpl("convert_unary_cont_vec4", &.{ ggml.cuda.kernels.typeName(S), ggml.cuda.kernels.typeName(T) }));
                const k4 = @divTrunc(kk, 4);
                const num_blocks: i64 = @divTrunc(k4 + dequantize_block_size - 1, dequantize_block_size);
                ggml.cuda.kernels.launch(k_vec4, .{
                    .grid = .{ .x = @truncate(@as(u64, @bitCast(num_blocks))) },
                    .block = .{ .x = dequantize_block_size },
                    .stream = stream,
                }, .{ vx, y, k4 });
                return;
            }

            nc(vx, y, kk, 1, 1, 1, kk, kk, kk, stream);
        }
    };
}

/// The cases the three contiguous getters share: every quantized type,
/// each to `T`.
fn quantizedTo(comptime T: type, t: ggml.impl.ggml_c.enum_ggml_type) ?ToFn {
    return switch (t) {
        ggml.impl.ggml_c.GGML_TYPE_Q1_0 => DequantizeBlock(q1_0, T).cont,
        ggml.impl.ggml_c.GGML_TYPE_Q2_0 => DequantizeBlock(q2_0, T).cont,
        ggml.impl.ggml_c.GGML_TYPE_Q4_0 => DequantizeRow("q4_0", 32, .q4, T).run,
        ggml.impl.ggml_c.GGML_TYPE_Q4_1 => DequantizeRow("q4_1", 32, .q4, T).run,
        ggml.impl.ggml_c.GGML_TYPE_Q5_0 => DequantizeBlock(q5_0, T).cont,
        ggml.impl.ggml_c.GGML_TYPE_Q5_1 => DequantizeBlock(q5_1, T).cont,
        ggml.impl.ggml_c.GGML_TYPE_Q8_0 => DequantizeBlock(q8_0, T).cont,
        ggml.impl.ggml_c.GGML_TYPE_Q2_K => DequantizeRow("q2_K", 64, .per_block, T).run,
        ggml.impl.ggml_c.GGML_TYPE_Q3_K => DequantizeRow("q3_K", 64, .per_block, T).run,
        ggml.impl.ggml_c.GGML_TYPE_Q4_K => DequantizeRow("q4_K", 32, .per_block, T).run,
        ggml.impl.ggml_c.GGML_TYPE_Q5_K => DequantizeRow("q5_K", 64, .per_block, T).run,
        ggml.impl.ggml_c.GGML_TYPE_Q6_K => DequantizeRow("q6_K", 64, .per_block, T).run,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS => DequantizeRow("iq2_xxs", 32, .per_block, T).run,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XS => DequantizeRow("iq2_xs", 32, .per_block, T).run,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_S => DequantizeRow("iq2_s", 32, .per_block, T).run,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS => DequantizeRow("iq3_xxs", 32, .per_block, T).run,
        ggml.impl.ggml_c.GGML_TYPE_IQ1_S => DequantizeRow("iq1_s", 32, .per_block, T).run,
        ggml.impl.ggml_c.GGML_TYPE_IQ1_M => DequantizeRow("iq1_m", 32, .per_block, T).run,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_NL => DequantizeRow("iq4_nl", 32, .per_block_ceil_nb32, T).run,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_XS => DequantizeRow("iq4_xs", 32, .per_block_ceil, T).run,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_S => DequantizeRow("iq3_s", 32, .per_block, T).run,
        ggml.impl.ggml_c.GGML_TYPE_MXFP4 => DequantizeRow("mxfp4", 32, .per_block_ceil, T).run,
        ggml.impl.ggml_c.GGML_TYPE_NVFP4 => DequantizeRowNvfp4(T).run,
        else => null,
    };
}

/// Ports `ggml_get_to_bf16_cuda` (convert.cu:496 @b1b770d90). Every getter
/// below takes `F8_E4M3`.
pub fn getToBf16(t: ggml.impl.ggml_c.enum_ggml_type) ?ToFn {
    return switch (t) {
        ggml.impl.ggml_c.GGML_TYPE_F8_E4M3 => ConvertUnary(ggml.cuda.kernels.fp8_e4m3, ggml.cuda.kernels.bf16).cont,
        ggml.impl.ggml_c.GGML_TYPE_F32 => ConvertUnary(f32, ggml.cuda.kernels.bf16).cont,
        ggml.impl.ggml_c.GGML_TYPE_F16 => ConvertUnary(ggml.cuda.kernels.half, ggml.cuda.kernels.bf16).cont,
        else => quantizedTo(ggml.cuda.kernels.bf16, t),
    };
}

/// Ports `ggml_get_to_fp16_cuda` (convert.cu:555 @b1b770d90).
pub fn getToFp16(t: ggml.impl.ggml_c.enum_ggml_type) ?ToFn {
    return switch (t) {
        ggml.impl.ggml_c.GGML_TYPE_Q8_0 => if (ggml.cuda.common.fp16Available(ggml.cuda.common.info().devices[@intCast(ggml.cuda.common.getDevice())].cc))
            dequantizeBlockQ8_0F16
        else
            DequantizeBlock(q8_0, ggml.cuda.kernels.half).cont,
        ggml.impl.ggml_c.GGML_TYPE_F8_E4M3 => ConvertUnary(ggml.cuda.kernels.fp8_e4m3, ggml.cuda.kernels.half).cont,
        ggml.impl.ggml_c.GGML_TYPE_F32 => ConvertUnary(f32, ggml.cuda.kernels.half).cont,
        ggml.impl.ggml_c.GGML_TYPE_BF16 => ConvertUnary(ggml.cuda.kernels.bf16, ggml.cuda.kernels.half).cont,
        else => quantizedTo(ggml.cuda.kernels.half, t),
    };
}

/// Ports `ggml_get_to_fp32_cuda` (convert.cu:617 @b1b770d90).
pub fn getToFp32(t: ggml.impl.ggml_c.enum_ggml_type) ?ToFn {
    return switch (t) {
        ggml.impl.ggml_c.GGML_TYPE_F8_E4M3 => ConvertUnary(ggml.cuda.kernels.fp8_e4m3, f32).cont,
        ggml.impl.ggml_c.GGML_TYPE_F16 => ConvertUnary(ggml.cuda.kernels.half, f32).cont,
        ggml.impl.ggml_c.GGML_TYPE_BF16 => ConvertUnary(ggml.cuda.kernels.bf16, f32).cont,
        else => quantizedTo(f32, t),
    };
}

/// The cases the three `_nc` getters share.
fn quantizedNcTo(comptime T: type, t: ggml.impl.ggml_c.enum_ggml_type) ?ToNcFn {
    return switch (t) {
        ggml.impl.ggml_c.GGML_TYPE_Q1_0 => DequantizeBlock(q1_0, T).nc,
        ggml.impl.ggml_c.GGML_TYPE_Q2_0 => DequantizeBlock(q2_0, T).nc,
        ggml.impl.ggml_c.GGML_TYPE_Q4_0 => DequantizeBlock(q4_0, T).nc,
        ggml.impl.ggml_c.GGML_TYPE_Q4_1 => DequantizeBlock(q4_1, T).nc,
        ggml.impl.ggml_c.GGML_TYPE_Q5_0 => DequantizeBlock(q5_0, T).nc,
        ggml.impl.ggml_c.GGML_TYPE_Q5_1 => DequantizeBlock(q5_1, T).nc,
        ggml.impl.ggml_c.GGML_TYPE_Q8_0 => DequantizeBlock(q8_0, T).nc,
        else => null,
    };
}

/// Ports `ggml_get_to_fp16_nc_cuda` (convert.cu:676 @b1b770d90).
pub fn getToFp16Nc(t: ggml.impl.ggml_c.enum_ggml_type) ?ToNcFn {
    return switch (t) {
        ggml.impl.ggml_c.GGML_TYPE_F8_E4M3 => ConvertUnary(ggml.cuda.kernels.fp8_e4m3, ggml.cuda.kernels.half).nc,
        ggml.impl.ggml_c.GGML_TYPE_F32 => ConvertUnary(f32, ggml.cuda.kernels.half).nc,
        ggml.impl.ggml_c.GGML_TYPE_BF16 => ConvertUnary(ggml.cuda.kernels.bf16, ggml.cuda.kernels.half).nc,
        else => quantizedNcTo(ggml.cuda.kernels.half, t),
    };
}

/// Ports `ggml_get_to_bf16_nc_cuda` (convert.cu:703 @b1b770d90).
pub fn getToBf16Nc(t: ggml.impl.ggml_c.enum_ggml_type) ?ToNcFn {
    return switch (t) {
        ggml.impl.ggml_c.GGML_TYPE_F8_E4M3 => ConvertUnary(ggml.cuda.kernels.fp8_e4m3, ggml.cuda.kernels.bf16).nc,
        ggml.impl.ggml_c.GGML_TYPE_F32 => ConvertUnary(f32, ggml.cuda.kernels.bf16).nc,
        ggml.impl.ggml_c.GGML_TYPE_F16 => ConvertUnary(ggml.cuda.kernels.half, ggml.cuda.kernels.bf16).nc,
        else => quantizedNcTo(ggml.cuda.kernels.bf16, t),
    };
}

/// Ports `ggml_get_to_fp32_nc_cuda` (convert.cu:730 @b1b770d90).
pub fn getToFp32Nc(t: ggml.impl.ggml_c.enum_ggml_type) ?ToNcFn {
    return switch (t) {
        ggml.impl.ggml_c.GGML_TYPE_F8_E4M3 => ConvertUnary(ggml.cuda.kernels.fp8_e4m3, f32).nc,
        ggml.impl.ggml_c.GGML_TYPE_F16 => ConvertUnary(ggml.cuda.kernels.half, f32).nc,
        ggml.impl.ggml_c.GGML_TYPE_BF16 => ConvertUnary(ggml.cuda.kernels.bf16, f32).nc,
        else => quantizedNcTo(f32, t),
    };
}
