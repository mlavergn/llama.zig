//! `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID`: choosing among MMVF, MMF,
//! MMVQ, MMQ and cuBLAS, the cuBLAS path itself, and the predicates the
//! fusion pass asks before fusing a matrix multiplication.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` at v1.0.0
//! (`b1b770d90`): `batched_mul_mat_traits`, `ggml_cuda_mul_mat_cublas_impl`,
//! `ggml_cuda_mul_mat_cublas`, `ggml_cuda_should_fuse_mul_mat`,
//! `ggml_cuda_should_fuse_mul_mat_vec_{f,q}`,
//! `ggml_cuda_get_mul_mat_scale`, `ggml_cuda_get_mul_mat_id_scale`,
//! `ggml_cuda_mul_mat_q_fusion_matches`, `ggml_cuda_should_fuse_mul_mat_q`,
//! `ggml_cuda_mul_mat`,
//! `ggml_cuda_mul_mat_id_needs_sync` and `ggml_cuda_mul_mat_id`. The kernel
//! `k_compute_batched_ptrs` stays CUDA.

const std = @import("std");
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");

fn i32Of(x: i64) c_int {
    return @truncate(x);
}

/// Ports `batched_mul_mat_traits` (ggml-cuda.cu:1786 @b1b770d90), for each `T`.
const Traits = struct {
    cuda_type_size: usize,
    compute_type: ggml.cuda.common.cu.cublasComputeType_t,
    data_type: ggml.cuda.common.cu.cudaDataType,
    ggml_type_val: ggml.impl.ggml_c.enum_ggml_type,
    alpha: *const anyopaque,
    beta: *const anyopaque,
    convert: *const fn (ggml.impl.ggml_c.enum_ggml_type) ?ggml.cuda.convert.ToFn,
    convert_nc: *const fn (ggml.impl.ggml_c.enum_ggml_type) ?ggml.cuda.convert.ToNcFn,
};

const one_f32: f32 = 1.0;
const zero_f32: f32 = 0.0;
/// `half(1.0)` and `half(0.0)`.
const one_f16: u16 = 0x3C00;
const zero_f16: u16 = 0x0000;

fn traitsOf(comptime t: ggml.impl.ggml_c.enum_ggml_type) Traits {
    return switch (t) {
        ggml.impl.ggml_c.GGML_TYPE_F32 => .{ .cuda_type_size = 4, .compute_type = ggml.cuda.common.cu.CUBLAS_COMPUTE_32F, .data_type = ggml.cuda.common.cu.CUDA_R_32F, .ggml_type_val = ggml.impl.ggml_c.GGML_TYPE_F32, .alpha = &one_f32, .beta = &zero_f32, .convert = ggml.cuda.convert.getToFp32, .convert_nc = ggml.cuda.convert.getToFp32Nc },
        ggml.impl.ggml_c.GGML_TYPE_BF16 => .{ .cuda_type_size = 2, .compute_type = ggml.cuda.common.cu.CUBLAS_COMPUTE_32F, .data_type = ggml.cuda.common.cu.CUDA_R_16BF, .ggml_type_val = ggml.impl.ggml_c.GGML_TYPE_BF16, .alpha = &one_f32, .beta = &zero_f32, .convert = ggml.cuda.convert.getToBf16, .convert_nc = ggml.cuda.convert.getToBf16Nc },
        ggml.impl.ggml_c.GGML_TYPE_F16 => .{ .cuda_type_size = 2, .compute_type = ggml.cuda.common.cu.CUBLAS_COMPUTE_16F, .data_type = ggml.cuda.common.cu.CUDA_R_16F, .ggml_type_val = ggml.impl.ggml_c.GGML_TYPE_F16, .alpha = &one_f16, .beta = &zero_f16, .convert = ggml.cuda.convert.getToFp16, .convert_nc = ggml.cuda.convert.getToFp16Nc },
        else => @compileError("no batched_mul_mat_traits"),
    };
}

/// Ports `ggml_cuda_mul_mat_cublas_impl` (ggml-cuda.cu:1831 @b1b770d90).
/// `dst`'s leading dimension is `nb1 / sizeof(float)`, not
/// `ne0`: an F32 chunk from `mulMatCublas` keeps the full destination's row
/// stride.
fn mulMatCublasImpl(comptime compute_type: ggml.impl.ggml_c.enum_ggml_type, ctx: *ggml.cuda.common.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const traits = comptime traitsOf(compute_type);
    const ts_cuda = traits.cuda_type_size;

    // F32 chunks can have padding between rows to preserve the original destination stride.
    ggml.impl.assert(ggml.types.ggml_is_contiguous(dst) or
        (compute_type == ggml.impl.ggml_c.GGML_TYPE_F32 and ggml.types.ggml_is_contiguous_1(dst)), "ggml_is_contiguous(dst) || (compute_type == GGML_TYPE_F32 && ggml_is_contiguous_1(dst))");

    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];
    const ne10 = src1.ne[0];
    const ne11 = src1.ne[1];
    const ne12 = src1.ne[2];
    const ne13 = src1.ne[3];
    const ne1 = dst.ne[1];

    const ldc: i64 = @intCast(dst.nb[1] / @sizeOf(f32));
    const ne_dst = ggml.types.ggml_nelements(dst);
    const main_stream = ctx.stream();
    const cublas_h = ctx.cublasHandle();

    const src0_ts = ggml.types.ggml_type_size(src0.type);
    ggml.impl.assert(src0.nb[0] == src0_ts, "nb00 == src0_ts");
    var s01: i64 = @intCast(src0.nb[1] / src0_ts);
    var s02: i64 = @intCast(src0.nb[2] / src0_ts);
    var s03: i64 = @intCast(src0.nb[3] / src0_ts);

    const src1_ts = ggml.types.ggml_type_size(src1.type);
    ggml.impl.assert(src1.nb[0] == src1_ts, "nb10 == src1_ts");
    var s11: i64 = @intCast(src1.nb[1] / src1_ts);
    var s12: i64 = @intCast(src1.nb[2] / src1_ts);
    var s13: i64 = @intCast(src1.nb[3] / src1_ts);

    const dst_ddf: ?*f32 = @ptrCast(@alignCast(dst.data));

    var src0_ptr: ?*const anyopaque = null;
    var src1_ptr: ?*const anyopaque = null;

    // `ggml_cuda_pool_alloc<cuda_t>`, counted in bytes here.
    var src0_alloc = ggml.cuda.common.PoolAlloc(u8).init(ctx.pool());
    defer src0_alloc.deinit();
    var src1_alloc = ggml.cuda.common.PoolAlloc(u8).init(ctx.pool());
    defer src1_alloc.deinit();

    var is_src0_cont_2 = ggml.types.ggml_is_contiguous_2(src0);
    var is_src1_cont_2 = ggml.types.ggml_is_contiguous_2(src1);

    if (src0.type == compute_type) {
        src0_ptr = src0.data;
    } else {
        _ = src0_alloc.alloc(@as(usize, @intCast(ggml.types.ggml_nelements(src0))) * ts_cuda);

        if (ggml.types.ggml_is_contiguously_allocated(src0)) {
            const convert_func = traits.convert(src0.type);
            ggml.impl.assert(convert_func != null, "convert_func != nullptr");
            convert_func.?(src0.data, src0_alloc.get(), ggml.types.ggml_nelements(src0), main_stream);
            const src0_bs: i64 = ggml.types.ggml_blck_size(src0.type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const convert_func = traits.convert_nc(src0.type);
            ggml.impl.assert(convert_func != null, "convert_func != nullptr");
            convert_func.?(src0.data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01 * s01;
            s03 = ne02 * s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    if (src1.type == compute_type) {
        src1_ptr = src1.data;
    } else {
        _ = src1_alloc.alloc(@as(usize, @intCast(ggml.types.ggml_nelements(src1))) * ts_cuda);

        if (ggml.types.ggml_is_contiguously_allocated(src1)) {
            const convert_func = traits.convert(src1.type);
            ggml.impl.assert(convert_func != null, "convert_func != nullptr");
            convert_func.?(src1.data, src1_alloc.get(), ggml.types.ggml_nelements(src1), main_stream);
            const src1_bs: i64 = ggml.types.ggml_blck_size(src1.type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const convert_func = traits.convert_nc(src1.type);
            ggml.impl.assert(convert_func != null, "convert_func != nullptr");
            convert_func.?(src1.data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11 * s11;
            s13 = ne12 * s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    var dst_temp = ggml.cuda.common.PoolAlloc(u8).init(ctx.pool());
    defer dst_temp.deinit();
    var dst_ptr: ?*anyopaque = null;
    var nbd2 = dst.nb[2];
    var nbd3 = dst.nb[3];

    var cu_compute_type = traits.compute_type;
    var cu_data_type = traits.data_type;
    const cu_data_type_a = traits.data_type;
    const cu_data_type_b = traits.data_type;
    var alpha = traits.alpha;
    var beta = traits.beta;

    const x = ggml.cuda.common.info().devices[@intCast(ctx.device)].cc;
    var prefer_f32_output = false;
    if (compute_type == ggml.impl.ggml_c.GGML_TYPE_F16) {
        // `cc == VOLTA || RDNA4 || CDNA`; the AMD tests are false here.
        prefer_f32_output = x == ggml.cuda.common.cc.volta;
    } else if (compute_type == ggml.impl.ggml_c.GGML_TYPE_BF16) {
        // `!RDNA3 && !RDNA4_M && !CDNA`; the AMD tests are false here.
        prefer_f32_output = true;
    }

    if (prefer_f32_output) {
        dst_ptr = dst_ddf;
        const f32_traits = comptime traitsOf(ggml.impl.ggml_c.GGML_TYPE_F32);
        cu_compute_type = f32_traits.compute_type;
        cu_data_type = f32_traits.data_type;
        alpha = f32_traits.alpha;
        beta = f32_traits.beta;
    } else {
        if (compute_type == ggml.impl.ggml_c.GGML_TYPE_F32) {
            dst_ptr = dst_ddf; // Direct F32 output
        } else {
            dst_ptr = dst_temp.alloc(@as(usize, @intCast(ne_dst)) * ts_cuda);
            nbd2 /= @sizeOf(f32) / ts_cuda;
            nbd3 /= @sizeOf(f32) / ts_cuda;
        }
    }

    ggml.impl.assert(@rem(ne12, ne02) == 0, "ne12 % ne02 == 0");
    ggml.impl.assert(@rem(ne13, ne03) == 0, "ne13 % ne03 == 0");

    // broadcast factors
    const r2 = @divTrunc(ne12, ne02);
    const r3 = @divTrunc(ne13, ne03);

    if (compute_type == ggml.impl.ggml_c.GGML_TYPE_F32 and ne12 == 1 and ne13 == 1) {
        ggml.cuda.common.cublasCheck(ggml.cuda.common.cu.cublasSgemm_v2(cublas_h, ggml.cuda.common.cu.CUBLAS_OP_T, ggml.cuda.common.cu.CUBLAS_OP_N, i32Of(ne01), i32Of(ne11), i32Of(ne10), @ptrCast(@alignCast(alpha)), @ptrCast(@alignCast(src0_ptr)), i32Of(s01), @ptrCast(@alignCast(src1_ptr)), i32Of(s11), @ptrCast(@alignCast(beta)), @ptrCast(@alignCast(dst_ptr)), i32Of(ldc)), "cublasSgemm(...)", @src());
    } else if (ne12 == 1 and ne13 == 1) {
        ggml.cuda.common.cublasCheck(ggml.cuda.common.cu.cublasGemmEx(cublas_h, ggml.cuda.common.cu.CUBLAS_OP_T, ggml.cuda.common.cu.CUBLAS_OP_N, i32Of(ne01), i32Of(ne11), i32Of(ne10), alpha, src0_ptr, cu_data_type_a, i32Of(s01), src1_ptr, cu_data_type_b, i32Of(s11), beta, dst_ptr, cu_data_type, i32Of(ldc), cu_compute_type, ggml.cuda.common.cu.CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx(...)", @src());
    } else if (r2 == 1 and r3 == 1 and is_src0_cont_2 and is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be
        // determined from dim 3:
        const sma = if (ne02 == 1) s03 else s02;
        const smb = if (ne12 == 1) s13 else s12;

        // there is no broadcast and src0, src1 are contiguous across dims
        // 2, 3: use cublasGemmStridedBatchedEx
        ggml.cuda.common.cublasCheck(ggml.cuda.common.cu.cublasGemmStridedBatchedEx(cublas_h, ggml.cuda.common.cu.CUBLAS_OP_T, ggml.cuda.common.cu.CUBLAS_OP_N, i32Of(ne01), i32Of(ne11), i32Of(ne10), alpha, src0_ptr, cu_data_type_a, i32Of(s01), sma, src1_ptr, cu_data_type_b, i32Of(s11), smb, beta, dst_ptr, cu_data_type, i32Of(ldc), ne1 * ldc, i32Of(ne12 * ne13), cu_compute_type, ggml.cuda.common.cu.CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmStridedBatchedEx(...)", @src());
    } else {
        // use cublasGemmBatchedEx
        const ne23 = ne12 * ne13;

        var ptrs_src = ggml.cuda.common.PoolAlloc(?*const anyopaque).initAlloc(ctx.pool(), @intCast(2 * ne23));
        defer ptrs_src.deinit();
        var ptrs_dst = ggml.cuda.common.PoolAlloc(?*anyopaque).initAlloc(ctx.pool(), @intCast(1 * ne23));
        defer ptrs_dst.deinit();

        const src_type_size = ts_cuda;

        const threads_x = 16;
        const threads_y = 16;
        ggml.cuda.kernels.launch(comptime ggml.cuda.kernels.get("ggml-cuda", "k_compute_batched_ptrs"), .{
            .grid = .{ .x = @intCast(@divTrunc(ne13 + threads_x - 1, threads_x)), .y = @intCast(@divTrunc(ne12 + threads_y - 1, threads_y)) },
            .block = .{ .x = threads_x, .y = threads_y },
            .stream = main_stream,
        }, .{
            src0_ptr,                                   src1_ptr,                                   dst_ptr,
            @as(?*anyopaque, @ptrCast(ptrs_src.get())), @as(?*anyopaque, @ptrCast(ptrs_dst.get())), ne12,
            ne13,                                       ne23,                                       @as(usize, @intCast(s02)) * src_type_size,
            @as(usize, @intCast(s03)) * src_type_size,  @as(usize, @intCast(s12)) * src_type_size,  @as(usize, @intCast(s13)) * src_type_size,
            nbd2,                                       nbd3,                                       r2,
            r3,
        });

        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGetLastError(), "cudaGetLastError()", @src());

        const ps = ptrs_src.get().?;
        ggml.cuda.common.cublasCheck(ggml.cuda.common.cu.cublasGemmBatchedEx(cublas_h, ggml.cuda.common.cu.CUBLAS_OP_T, ggml.cuda.common.cu.CUBLAS_OP_N, i32Of(ne01), i32Of(ne11), i32Of(ne10), alpha, ps + 0 * @as(usize, @intCast(ne23)), cu_data_type_a, i32Of(s01), ps + 1 * @as(usize, @intCast(ne23)), cu_data_type_b, i32Of(s11), beta, @ptrCast(ptrs_dst.get().?), cu_data_type, i32Of(ldc), i32Of(ne23), cu_compute_type, ggml.cuda.common.cu.CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmBatchedEx(...)", @src());
    }

    // Convert output back to F32 if needed
    if (cu_data_type != ggml.cuda.common.cu.CUDA_R_32F) {
        const to_fp32_cuda = ggml.cuda.convert.getToFp32(traits.ggml_type_val).?;
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

/// Ports `ggml_cuda_mul_mat_cublas` (ggml-cuda.cu:2044 @b1b770d90). An F32 product over an F16 or BF16 `src0` whose F32 copy would
/// pass 512 MiB is done in chunks of `src0` rows, each converted on its own.
/// The precision is `ggml_prec_get_acc`'s and an `F8_E4M3`
/// weight converts as a quantized one does.
pub fn mulMatCublas(ctx: *ggml.cuda.common.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const x = ggml.cuda.common.info().devices[@intCast(ctx.device)].cc;
    const prec: ggml.impl.ggml_c.enum_ggml_prec = ggml.ops.ggml_prec_get_acc(dst);
    var compute_type = src0.type;
    if (ggml.types.ggml_is_quantized(compute_type) or compute_type == ggml.impl.ggml_c.GGML_TYPE_F8_E4M3) {
        compute_type = if (ggml.cuda.common.fastFp16HardwareAvailable(x)) ggml.impl.ggml_c.GGML_TYPE_F16 else ggml.impl.ggml_c.GGML_TYPE_F32;
    } else if (compute_type == ggml.impl.ggml_c.GGML_TYPE_F16 and !ggml.cuda.common.fastFp16HardwareAvailable(x)) {
        compute_type = ggml.impl.ggml_c.GGML_TYPE_F32;
    } else if (compute_type == ggml.impl.ggml_c.GGML_TYPE_BF16 and !ggml.cuda.common.fastBf16HardwareAvailable(x)) {
        if (ggml.cuda.common.cc.isAmd(x) and src1.ne[1] > 32) {
            compute_type = ggml.impl.ggml_c.GGML_TYPE_F32;
        }
        if (ggml.cuda.common.cc.isNvidia(x) and src1.ne[1] > @as(i64, if (x >= ggml.cuda.common.cc.volta) 8 else 128)) {
            compute_type = ggml.impl.ggml_c.GGML_TYPE_F32;
        }
    }
    // F16 is the only compute type that can not satisfy a request for BF16
    if (prec == ggml.impl.ggml_c.GGML_PREC_BF16 and compute_type == ggml.impl.ggml_c.GGML_TYPE_F16) {
        compute_type = if (ggml.cuda.common.fastBf16HardwareAvailable(x)) ggml.impl.ggml_c.GGML_TYPE_BF16 else ggml.impl.ggml_c.GGML_TYPE_F32;
    } else if (prec == ggml.impl.ggml_c.GGML_PREC_F32) {
        compute_type = ggml.impl.ggml_c.GGML_TYPE_F32;
    }

    if (stdlib.env.getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE")) |env_c| {
        var buf: [64]u8 = undefined;
        const env = std.mem.span(env_c);
        const n = @min(env.len, buf.len);
        for (env[0..n], 0..) |ch, i| buf[i] = std.ascii.toLower(ch);
        const env_cpp = buf[0..n];
        if (std.mem.eql(u8, env_cpp, "f32") or std.mem.eql(u8, env_cpp, "fp32")) {
            compute_type = ggml.impl.ggml_c.GGML_TYPE_F32;
        } else if (std.mem.eql(u8, env_cpp, "f16") or std.mem.eql(u8, env_cpp, "fp16")) {
            compute_type = ggml.impl.ggml_c.GGML_TYPE_F16;
        } else if (std.mem.eql(u8, env_cpp, "bf16")) {
            compute_type = ggml.impl.ggml_c.GGML_TYPE_BF16;
        } else if (!std.mem.eql(u8, env_cpp, "auto")) {
            var z: [65]u8 = undefined;
            @memcpy(z[0..n], env_cpp);
            z[n] = 0;
            ggml.impl.logWarn("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", .{ "ggml_cuda_mul_mat_cublas", @as([*:0]const u8, @ptrCast(&z)) });
        }
    }

    const max_src0_convert_size: usize = 512 * 1024 * 1024;
    const src0_f32_size = @as(usize, @bitCast(ggml.types.ggml_nelements(src0))) * @sizeOf(f32);

    if (compute_type == ggml.impl.ggml_c.GGML_TYPE_F32 and
        (src0.type == ggml.impl.ggml_c.GGML_TYPE_F16 or src0.type == ggml.impl.ggml_c.GGML_TYPE_BF16) and
        src0_f32_size > max_src0_convert_size)
    {
        const f32_row_size = src0_f32_size / @as(usize, @bitCast(src0.ne[1]));
        const rows_per_chunk: i64 = @max(1, @as(i64, @bitCast(max_src0_convert_size / f32_row_size)));

        if (rows_per_chunk < src0.ne[1]) {
            // `i01` is a Zig type name; `j01` is the C's `i01`.
            var src0_chunk = src0.*;
            var dst_chunk = dst.*;

            var j01: i64 = 0;
            while (j01 < src0.ne[1]) : (j01 += rows_per_chunk) {
                src0_chunk.ne[1] = @min(rows_per_chunk, src0.ne[1] - j01);
                src0_chunk.data = @as([*]u8, @ptrCast(src0.data.?)) + @as(usize, @intCast(j01)) * src0.nb[1];

                dst_chunk.ne[0] = src0_chunk.ne[1];
                dst_chunk.data = @as([*]u8, @ptrCast(dst.data.?)) + @as(usize, @intCast(j01)) * dst.nb[0];

                mulMatCublasImpl(ggml.impl.ggml_c.GGML_TYPE_F32, ctx, &src0_chunk, src1, &dst_chunk);
            }
            return;
        }
    }

    switch (compute_type) {
        ggml.impl.ggml_c.GGML_TYPE_F32 => mulMatCublasImpl(ggml.impl.ggml_c.GGML_TYPE_F32, ctx, src0, src1, dst),
        ggml.impl.ggml_c.GGML_TYPE_BF16 => mulMatCublasImpl(ggml.impl.ggml_c.GGML_TYPE_BF16, ctx, src0, src1, dst),
        ggml.impl.ggml_c.GGML_TYPE_F16 => mulMatCublasImpl(ggml.impl.ggml_c.GGML_TYPE_F16, ctx, src0, src1, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

/// Ports `ggml_cuda_should_fuse_mul_mat` (ggml-cuda.cu:2125 @b1b770d90).
pub fn shouldFuseMulMat(ffn_up: *const ggml.impl.ggml_c.ggml_tensor, ffn_gate: *const ggml.impl.ggml_c.ggml_tensor, glu: *const ggml.impl.ggml_c.ggml_tensor, ffn_up_bias: ?*const ggml.impl.ggml_c.ggml_tensor, ffn_gate_bias: ?*const ggml.impl.ggml_c.ggml_tensor, ffn_up_scale: ?*const ggml.impl.ggml_c.ggml_tensor, ffn_gate_scale: ?*const ggml.impl.ggml_c.ggml_tensor) bool {
    const has_bias = ffn_up_bias != null or ffn_gate_bias != null;
    const has_scale = ffn_up_scale != null or ffn_gate_scale != null;

    if (has_bias and (ffn_up_bias == null or ffn_gate_bias == null)) return false;
    if (has_scale and (ffn_up_scale == null or ffn_gate_scale == null)) return false;

    const is_mul_mat = ffn_up.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT and ffn_gate.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT and glu.op == ggml.impl.ggml_c.GGML_OP_GLU;
    const is_mul_mat_id = ffn_up.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID and ffn_gate.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID and glu.op == ggml.impl.ggml_c.GGML_OP_GLU;

    if (!is_mul_mat and !is_mul_mat_id) return false;

    const expected_bias_op: ggml.impl.ggml_c.enum_ggml_op = if (is_mul_mat) ggml.impl.ggml_c.GGML_OP_ADD else ggml.impl.ggml_c.GGML_OP_ADD_ID;
    const ffn_up_bias_src: *const ggml.impl.ggml_c.ggml_tensor = if (has_scale) ffn_up_scale.? else ffn_up;
    const ffn_gate_bias_src: *const ggml.impl.ggml_c.ggml_tensor = if (has_scale) ffn_gate_scale.? else ffn_gate;
    const ffn_up_out: *const ggml.impl.ggml_c.ggml_tensor = if (has_bias) ffn_up_bias.? else ffn_up_bias_src;
    const ffn_gate_out: *const ggml.impl.ggml_c.ggml_tensor = if (has_bias) ffn_gate_bias.? else ffn_gate_bias_src;

    if (glu.src[0] != ffn_gate_out or glu.src[1] != ffn_up_out) return false;

    if (has_scale) {
        const us = ffn_up_scale.?;
        const gs = ffn_gate_scale.?;
        if (us.op != ggml.impl.ggml_c.GGML_OP_MUL or gs.op != ggml.impl.ggml_c.GGML_OP_MUL) return false;
        const up_has_mm = us.src[0] == ffn_up or us.src[1] == ffn_up;
        const gate_has_mm = gs.src[0] == ffn_gate or gs.src[1] == ffn_gate;
        if (!up_has_mm or !gate_has_mm) return false;
    }

    if (has_bias) {
        const ub = ffn_up_bias.?;
        const gb = ffn_gate_bias.?;
        if (ub.op != expected_bias_op or gb.op != expected_bias_op) return false;

        if (expected_bias_op == ggml.impl.ggml_c.GGML_OP_ADD) {
            const up_has_mul = ub.src[0] == ffn_up_bias_src or ub.src[1] == ffn_up_bias_src;
            const gate_has_mul = gb.src[0] == ffn_gate_bias_src or gb.src[1] == ffn_gate_bias_src;
            if (!up_has_mul or !gate_has_mul) return false;
        } else { // GGML_OP_ADD_ID
            if (ub.src[0] != ffn_up_bias_src or gb.src[0] != ffn_gate_bias_src) return false;
            if (ub.src[2] != ffn_up.src[2] or gb.src[2] != ffn_gate.src[2]) return false;
        }
    }

    const up0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, ffn_up.src[0]);
    const gate0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, ffn_gate.src[0]);
    if (up0.type != gate0.type or !ggml.types.ggml_are_same_shape(up0, gate0) or !ggml.types.ggml_are_same_stride(up0, gate0)) return false;

    if (ffn_up.src[1] != ffn_gate.src[1]) return false;

    if (is_mul_mat_id and ffn_up.src[2] != ffn_gate.src[2]) return false;

    const glu_op = ggml.types.ggml_get_glu_op(glu);
    if (glu_op != ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU and glu_op != ggml.impl.ggml_c.GGML_GLU_OP_GEGLU and glu_op != ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU_OAI and
        glu_op != ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU_CLAMP) return false;

    // swapped
    if (ggml.impl.getOpParamsI32(glu, 1) != 0) return false;

    return true;
}

/// Ports `ggml_cuda_should_fuse_mul_mat_vec_f` (ggml-cuda.cu:2219 @b1b770d90).
/// An `F8_E4M3` weight qualifies, and the gate's weight, when
/// given, must take MMVF too.
pub fn shouldFuseMulMatVecF(tensor: *const ggml.impl.ggml_c.ggml_tensor, gate: ?*const ggml.impl.ggml_c.ggml_tensor) bool {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, tensor.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, tensor.src[1]);
    const dst = tensor;

    const is_mul_mat_id = tensor.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID;

    var use_mul_mat_vec_f = (src0.type == ggml.impl.ggml_c.GGML_TYPE_F32 or src0.type == ggml.impl.ggml_c.GGML_TYPE_F16 or src0.type == ggml.impl.ggml_c.GGML_TYPE_BF16 or
        src0.type == ggml.impl.ggml_c.GGML_TYPE_F8_E4M3) and
        src1.type == ggml.impl.ggml_c.GGML_TYPE_F32 and dst.type == ggml.impl.ggml_c.GGML_TYPE_F32;

    const x = ggml.cuda.common.info().devices[@intCast(ggml.cuda.common.getDevice())].cc;
    const warp_size = ggml.cuda.common.info().devices[@intCast(ggml.cuda.common.getDevice())].warp_size;
    const ncols: i64 = if (is_mul_mat_id) src1.ne[2] else src1.ne[1];
    use_mul_mat_vec_f = use_mul_mat_vec_f and ggml.cuda.mmvf.shouldUseMmvf(src0, x, warp_size, ncols);

    if (gate) |g| {
        if (!ggml.cuda.mmvf.shouldUseMmvf(ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, g.src[0]), x, warp_size, ncols)) return false;
    }

    //we only support fusion for ncols_dst = 1
    if (tensor.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT and dst.ne[1] != 1) return false;
    if (tensor.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID and dst.ne[2] != 1) return false;

    return use_mul_mat_vec_f;
}

/// Ports `ggml_cuda_should_fuse_mul_mat_vec_q` (ggml-cuda.cu:2252 @b1b770d90).
/// It asks `ggml_cuda_should_use_mmvq` for the batch limit.
pub fn shouldFuseMulMatVecQ(tensor: *const ggml.impl.ggml_c.ggml_tensor) bool {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, tensor.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, tensor.src[1]);
    const dst = tensor;

    const bad_padding_clear = ggml.backend.ggml_backend_buffer_get_usage(src0.buffer) == ggml.impl.ggml_c.GGML_BACKEND_BUFFER_USAGE_COMPUTE and
        ggml.types.ggml_nbytes(src0) != ggml.backend.ggml_backend_buffer_get_alloc_size(src0.buffer, src0) and src0.view_src != null;

    const x = ggml.cuda.common.info().devices[@intCast(ggml.cuda.common.getDevice())].cc;
    const use_mul_mat_vec_q = ggml.cuda.mmvq.shouldUseMmvq(src0.type, x, src1.ne[1]) and !bad_padding_clear and
        src1.type == ggml.impl.ggml_c.GGML_TYPE_F32 and dst.type == ggml.impl.ggml_c.GGML_TYPE_F32;

    // fusion is not universally faster on Pascal
    if (x <= ggml.cuda.common.cc.pascal) return false;
    //we only support fusion for ncols_dst = 1
    if (tensor.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT and dst.ne[1] != 1) return false;
    if (tensor.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID and dst.ne[2] > ggml.cuda.mmvq.getMmidMaxBatch(src0.type, x)) return false;

    return use_mul_mat_vec_q;
}

/// Ports `ggml_cuda_get_mul_mat_scale` (ggml-cuda.cu:2335 @b1b770d90): the
/// weight scale that `scale_node` (`MUL`) applies to the output of `mm_node`
/// (`MUL_MAT`), or null. It is shared with `graph_optimize`,
/// and accepts `F8_E4M3` weights too.
pub fn getMulMatScale(scale_node: *const ggml.impl.ggml_c.ggml_tensor, mm_node: *const ggml.impl.ggml_c.ggml_tensor) ?*const ggml.impl.ggml_c.ggml_tensor {
    const scale_lhs_mm = scale_node.src[0] == mm_node;
    const scale_rhs_mm = scale_node.src[1] == mm_node;
    if (!scale_lhs_mm and !scale_rhs_mm) return null;

    const scale = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, if (scale_lhs_mm) scale_node.src[1] else scale_node.src[0]);
    const w = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, mm_node.src[0]);
    if ((w.type != ggml.impl.ggml_c.GGML_TYPE_NVFP4 and w.type != ggml.impl.ggml_c.GGML_TYPE_F8_E4M3) or scale_node.type != ggml.impl.ggml_c.GGML_TYPE_F32 or
        scale.type != ggml.impl.ggml_c.GGML_TYPE_F32 or !ggml.types.ggml_is_contiguous(scale) or ggml.types.ggml_nelements(scale) != 1 or
        !ggml.types.ggml_are_same_shape(scale_node, mm_node)) return null;

    return scale;
}

/// Ports `ggml_cuda_get_mul_mat_id_scale` (ggml-cuda.cu:2353 @b1b770d90): the
/// same for `MUL_MAT_ID` with one scale per expert -- `RESHAPE`, `REPEAT`
/// and `GET_ROWS` (by the ids) of the scales, then the `MUL`.
pub fn getMulMatIdScale(reshape: *const ggml.impl.ggml_c.ggml_tensor, repeat: *const ggml.impl.ggml_c.ggml_tensor, getrows_n: *const ggml.impl.ggml_c.ggml_tensor, scale_node: *const ggml.impl.ggml_c.ggml_tensor, mm_node: *const ggml.impl.ggml_c.ggml_tensor) ?*const ggml.impl.ggml_c.ggml_tensor {
    if (repeat.src[0] != reshape or getrows_n.src[0] != repeat or getrows_n.src[1] != mm_node.src[2]) return null;
    if (!((scale_node.src[0] == mm_node and scale_node.src[1] == getrows_n) or
        (scale_node.src[0] == getrows_n and scale_node.src[1] == mm_node))) return null;

    const scale = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, reshape.src[0]);
    const w = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, mm_node.src[0]);
    if ((w.type != ggml.impl.ggml_c.GGML_TYPE_NVFP4 and w.type != ggml.impl.ggml_c.GGML_TYPE_F8_E4M3) or scale_node.type != ggml.impl.ggml_c.GGML_TYPE_F32 or
        scale.type != ggml.impl.ggml_c.GGML_TYPE_F32 or !ggml.types.ggml_is_contiguous(scale) or ggml.types.ggml_nelements(scale) != w.ne[2] or
        !ggml.types.ggml_are_same_shape(scale_node, mm_node)) return null;

    return scale;
}

/// Ports `ggml_cuda_mul_mat_q_fusion_matches` (ggml-cuda.cu:2375 @b1b770d90):
/// whether gate/up/GLU has a fused MMQ kernel at all. Buffer-
/// and batch-size independent, so `graph_optimize` produces the same graph
/// topology for every ubatch size. `up` may be any node here, so its sources
/// are not read until its op says they exist. `up` may be a
/// `MUL_MAT_ID` too (the MoE gate/up/GLU). `up_scale` and
/// `gate_scale` are the `MUL` nodes applying the NVFP4 weight scales between
/// the matmuls and the GLU, and both matmuls must ask one `src1` precision.
pub fn mulMatQFusionMatches(up: *const ggml.impl.ggml_c.ggml_tensor, gate: *const ggml.impl.ggml_c.ggml_tensor, glu: *const ggml.impl.ggml_c.ggml_tensor, device: c_int, up_scale: ?*const ggml.impl.ggml_c.ggml_tensor, gate_scale: ?*const ggml.impl.ggml_c.ggml_tensor) bool {
    if (up.op != ggml.impl.ggml_c.GGML_OP_MUL_MAT and up.op != ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID) return false;

    const x_up = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, up.src[0]);
    const y = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, up.src[1]);

    if (!ggml.cuda.mmq.fusionSupported(x_up.type) or y.type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;

    // The fused kernel is only compiled for weights whose rows fill whole tiles (no fallback).
    if (ggml.cuda.mmq.needsFallback(x_up.ne[1])) return false;

    if (!shouldFuseMulMat(up, gate, glu, null, null, up_scale, gate_scale)) return false;

    // The fused kernel runs both MUL_MATs at one src1 precision, see ggml_cuda_mul_mat_q.
    if (ggml.ops.ggml_prec_get_src(up, 1) != ggml.ops.ggml_prec_get_src(gate, 1)) return false;

    // The fused write-back applies SWIGLU_OAI with the default alpha and limit of ggml_cuda_op_swiglu_oai_single.
    if (ggml.types.ggml_get_glu_op(glu) == ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU_OAI and (ggml.impl.getOpParamsF32(glu, 2) != 1.702 or ggml.impl.getOpParamsF32(glu, 3) != 7.0)) return false;

    // NVIDIA only: the fused write-back relies on the MMA accumulator layout.
    return ggml.cuda.common.turingMmaAvailable(ggml.cuda.common.info().devices[@intCast(device)].cc);
}

/// The C's `bad_padding_clear` lambda in `ggml_cuda_should_fuse_mul_mat_q`:
/// the padding of a compute-buffer weight that is a view cannot be cleared,
/// see `ggml_cuda_mul_mat`.
fn badPaddingClear(x: *const ggml.impl.ggml_c.ggml_tensor) bool {
    return ggml.backend.ggml_backend_buffer_get_usage(x.buffer) == ggml.impl.ggml_c.GGML_BACKEND_BUFFER_USAGE_COMPUTE and
        ggml.types.ggml_nbytes(x) != ggml.backend.ggml_backend_buffer_get_alloc_size(x.buffer, x) and x.view_src != null;
}

/// Ports `ggml_cuda_should_fuse_mul_mat_q` (ggml-cuda.cu:2407 @b1b770d90):
/// `mulMatQFusionMatches` and the checks that depend on the
/// buffers and the batch. For `MUL_MAT_ID` the kernel choice is
/// `ggml_cuda_mul_mat_id`'s, with the nominal tiles of all experts. A `MUL_MAT` that few-row MMQ takes is not fused.
pub fn shouldFuseMulMatQ(up: *const ggml.impl.ggml_c.ggml_tensor, gate: *const ggml.impl.ggml_c.ggml_tensor, glu: *const ggml.impl.ggml_c.ggml_tensor, up_scale: ?*const ggml.impl.ggml_c.ggml_tensor, gate_scale: ?*const ggml.impl.ggml_c.ggml_tensor) bool {
    // A hinted matmul can be fused, but its unfused path may be faster (e.g. FWHT for GGML_HINT_SRC0_IS_HADAMARD).
    if (ggml.impl.getOpParamsI32(up, 1) != ggml.impl.ggml_c.GGML_HINT_NONE or ggml.impl.getOpParamsI32(gate, 1) != ggml.impl.ggml_c.GGML_HINT_NONE) return false;

    const device = ggml.cuda.common.getDevice();
    if (!mulMatQFusionMatches(up, gate, glu, device, up_scale, gate_scale) or badPaddingClear(ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, up.src[0])) or badPaddingClear(ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, gate.src[0]))) return false;

    // The batch size checks are done here, not in ggml_cuda_mul_mat_q_fusion_matches.
    const x_up = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, up.src[0]);
    const y = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, up.src[1]);
    const x = ggml.cuda.common.info().devices[@intCast(device)].cc;

    // For MUL_MAT_ID: same kernel choice as ggml_cuda_mul_mat_id, with the nominal tiles of all experts (the per-expert token counts are only known on the GPU).
    const is_id = up.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID;
    const ncols: i64 = if (is_id) y.ne[2] else y.ne[1];
    const nchannels: i64 = if (is_id) x_up.ne[2] else y.ne[2] * y.ne[3];
    const use_mmvq = if (is_id) ncols <= ggml.cuda.mmvq.getMmidMaxBatch(x_up.type, x) else ggml.cuda.mmvq.shouldUseMmvq(x_up.type, x, ncols);
    // do not fuse the matmuls that few-row MMQ takes in ggml_cuda_mul_mat
    if (!is_id and ggml.cuda.mmq_few.shouldUseMmqFew(x_up, y, x)) return false;
    if (use_mmvq or !ggml.cuda.mmq.shouldUseMmq(x_up.type, x, ncols, if (is_id) x_up.ne[2] else 0)) return false;

    // The fused kernel needs one block per tile (no stream-k split), so only fuse if the tiling is efficient.
    return ggml.cuda.mmq.fusionIsEfficient(x_up.type, ggml.ops.ggml_prec_get_src(up, 1), x_up.ne[1], ncols, nchannels, device);
}

/// Ports `ggml_cuda_mul_mat` (ggml-cuda.cu:2446 @b1b770d90). Few-row MMQ takes a per-type number of `src1` columns up to 24 ahead of
/// MMVQ and MMQ, and an `F8_E4M3` weight tries cuBLASLt's FP8 GEMM before
/// cuBLAS.
pub fn mulMat(ctx: *ggml.cuda.common.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const ne01 = src0.ne[1];
    const ne11 = src1.ne[1];
    const ne2 = dst.ne[2];
    const ne3 = dst.ne[3];

    if (ggml.cuda.fwht.opMulMatUseFwht(dst) and ggml.cuda.fwht.opFwht(ctx, src1, dst)) return;

    // If src0 is a temporary compute buffer it may have some padding that
    // needs to be cleared for mul_mat_vec_q or mul_mat_q. But if src0 is
    // also a view of another tensor then this cannot be done safely, so
    // cuBLAS is used.
    const bad_padding_clear = ggml.backend.ggml_backend_buffer_get_usage(src0.buffer) == ggml.impl.ggml_c.GGML_BACKEND_BUFFER_USAGE_COMPUTE and
        ggml.types.ggml_nbytes(src0) != ggml.backend.ggml_backend_buffer_get_alloc_size(src0.buffer, src0) and src0.view_src != null;
    if (bad_padding_clear or src1.type != ggml.impl.ggml_c.GGML_TYPE_F32 or dst.type != ggml.impl.ggml_c.GGML_TYPE_F32) {
        mulMatCublas(ctx, src0, src1, dst);
        return;
    }

    const x = ggml.cuda.common.info().devices[@intCast(ctx.device)].cc;
    const warp_size = ggml.cuda.common.info().devices[@intCast(ctx.device)].warp_size;

    if (ggml.cuda.mmvf.shouldUseMmvf(src0, x, warp_size, ne11)) {
        // The custom vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml.cuda.mmvf.mulMatVecF(ctx, src0, src1, null, dst, null);
        return;
    }
    // A transposed vector can still use MMVF (i.e. ne01 == 1)
    if (ne01 == 1 and ne11 > ggml.cuda.mmvf.max_batch_size and ne2 == 1 and ne3 == 1 and
        src0.type == ggml.impl.ggml_c.GGML_TYPE_F32 and
        ggml.types.ggml_is_contiguous(src0) and ggml.types.ggml_is_contiguous(src1) and ggml.types.ggml_is_contiguous(dst) and
        ggml.cuda.mmvf.shouldUseMmvf(src1, x, warp_size, 1))
    {
        var dst_vec = dst.*;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0] * @as(usize, @intCast(ne11));
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml.cuda.mmvf.mulMatVecF(ctx, src1, src0, null, &dst_vec, null);
        return;
    }
    if (ggml.cuda.mmf.shouldUseMmf(src0.type, x, warp_size, &src0.ne, &src0.nb, i32Of(ne11), false)) {
        ggml.cuda.mmf.mulMatF(ctx, src0, src1, null, dst);
        return;
    }
    // from a per-type number of src1 columns up to 24, few-row MMQ is faster than MMVQ and MMQ
    const use_mmq_few = ggml.cuda.mmq_few.shouldUseMmqFew(src0, src1, x);
    if (!use_mmq_few and ggml.cuda.mmvq.shouldUseMmvq(src0.type, x, ne11)) {
        ggml.cuda.mmvq.mulMatVecQ(ctx, src0, src1, null, dst, null);
        return;
    }
    if (use_mmq_few) {
        ggml.cuda.mmq_few.mulMatQFew(ctx, src0, src1, dst);
        return;
    }
    if (ggml.cuda.mmq.shouldUseMmq(src0.type, x, ne11, 0)) {
        ggml.cuda.mmq.mulMatQ(ctx, src0, src1, null, dst, null);
        return;
    }
    if (src0.type == ggml.impl.ggml_c.GGML_TYPE_F8_E4M3 and ggml.cuda.fp8.mulMatFp8(ctx, src0, src1, dst, null)) return;
    mulMatCublas(ctx, src0, src1, dst);
}

/// Ports `ggml_cuda_mul_mat_id_needs_sync` (ggml-cuda.cu:2512 @b1b770d90).
pub fn mulMatIdNeedsSync(dst: *const ggml.impl.ggml_c.ggml_tensor, x: c_int) bool {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    const warp_size = ggml.cuda.common.info().devices[@intCast(ggml.cuda.common.getDevice())].warp_size;

    if (src1.type != ggml.impl.ggml_c.GGML_TYPE_F32 or dst.type != ggml.impl.ggml_c.GGML_TYPE_F32) return true;

    if (dst.ne[2] <= ggml.cuda.mmvq.max_batch_size) {
        if (ggml.types.ggml_is_quantized(src0.type)) {
            if (dst.ne[2] <= ggml.cuda.mmvq.getMmidMaxBatch(src0.type, x)) return false;
        } else if (src0.type == ggml.impl.ggml_c.GGML_TYPE_F8_E4M3 and ggml.cuda.mmvf.shouldUseMmvf(src0, x, warp_size, dst.ne[2])) {
            return false;
        } else if (src0.type != ggml.impl.ggml_c.GGML_TYPE_F8_E4M3 and ggml.cuda.common.cc.isAmd(x)) {
            return false;
        }
    }

    if (ggml.cuda.mmq.shouldUseMmq(src0.type, x, src1.ne[2], src0.ne[2])) return false;

    if (ggml.cuda.mmf.shouldUseMmf(src0.type, x, warp_size, &src0.ne, &src0.nb, i32Of(src1.ne[2]), true)) return false;

    return true;
}

/// Ports `ggml_cuda_mul_mat_id` (ggml-cuda.cu:2545 @b1b770d90). An
/// `F8_E4M3` weight takes MMVF at a small batch on every vendor.
pub fn mulMatId(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);
    const ids = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[2]);

    ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type  == GGML_TYPE_F32");

    const ne02 = src0.ne[2];
    const nb02 = src0.nb[2];
    const ne10 = src1.ne[0];
    const ne11 = src1.ne[1];
    const ne12 = src1.ne[2];
    const nb11 = src1.nb[1];
    const nb12 = src1.nb[2];
    const nb13 = src1.nb[3];
    const ne0 = dst.ne[0];
    const ne2 = dst.ne[2];
    const nb1 = dst.nb[1];
    const nb2 = dst.nb[2];
    const nb3 = dst.nb[3];

    const x = ggml.cuda.common.info().devices[@intCast(ggml.cuda.common.getDevice())].cc;
    const warp_size = ggml.cuda.common.info().devices[@intCast(ggml.cuda.common.getDevice())].warp_size;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1.type == ggml.impl.ggml_c.GGML_TYPE_F32 and dst.type == ggml.impl.ggml_c.GGML_TYPE_F32) {
        comptime std.debug.assert(ggml.cuda.mmvq.max_batch_size == ggml.cuda.mmvf.max_batch_size);
        if (ne2 <= ggml.cuda.mmvq.max_batch_size) {
            if (ggml.types.ggml_is_quantized(src0.type)) {
                const mmvq_mmid_max = ggml.cuda.mmvq.getMmidMaxBatch(src0.type, x);
                if (ne2 <= mmvq_mmid_max) {
                    ggml.cuda.mmvq.mulMatVecQ(ctx, src0, src1, ids, dst, null);
                    return;
                }
            } else if (src0.type == ggml.impl.ggml_c.GGML_TYPE_F8_E4M3 and ggml.cuda.mmvf.shouldUseMmvf(src0, x, warp_size, ne2)) {
                ggml.cuda.mmvf.mulMatVecF(ctx, src0, src1, ids, dst, null);
                return;
            } else {
                if (src0.type != ggml.impl.ggml_c.GGML_TYPE_F8_E4M3 and ggml.cuda.common.cc.isAmd(x)) {
                    ggml.cuda.mmvf.mulMatVecF(ctx, src0, src1, ids, dst, null);
                    return;
                }
            }
        }

        if (ggml.cuda.mmq.shouldUseMmq(src0.type, x, ne12, ne02)) {
            ggml.cuda.mmq.mulMatQ(ctx, src0, src1, ids, dst, null);
            return;
        }

        if (ggml.cuda.mmf.shouldUseMmf(src0.type, x, warp_size, &src0.ne, &src0.nb, i32Of(src1.ne[2]), true)) {
            ggml.cuda.mmf.mulMatF(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs,
    // because it requires stream synchronization
    ggml.impl.assert(mulMatIdNeedsSync(dst, x), "ggml_cuda_mul_mat_id_needs_sync(dst, cc)");
    const stream = ctx.stream();

    ggml.impl.assert(nb12 % nb11 == 0, "nb12 % nb11 == 0");
    ggml.impl.assert(nb2 % nb1 == 0, "nb2  % nb1  == 0");

    const type_src1_sorted: ggml.impl.ggml_c.enum_ggml_type = if ((src0.type == ggml.impl.ggml_c.GGML_TYPE_F16 and !ggml.cuda.common.fastFp16HardwareAvailable(x)) or
        ggml.types.ggml_is_quantized(src0.type) or src0.type == ggml.impl.ggml_c.GGML_TYPE_F8_E4M3) ggml.impl.ggml_c.GGML_TYPE_F32 else src0.type;
    const type_dst_sorted: ggml.impl.ggml_c.enum_ggml_type = ggml.impl.ggml_c.GGML_TYPE_F32;
    const ts_src1_sorted = ggml.types.ggml_type_size(type_src1_sorted);
    const ts_dst_sorted = ggml.types.ggml_type_size(type_dst_sorted);

    const n_expert_used = ids.ne[0];
    const ne_get_rows = ne12 * n_expert_used;
    const n_rows: usize = @intCast(ne_get_rows);

    // `ids_to_sorted_host`, with `ids_from_sorted_host` appended after it
    // as the C does once the loop is done.
    const ids_sorted_host = std.heap.c_allocator.alloc(i32, 2 * n_rows) catch ggml.impl.abort("out of memory");
    defer std.heap.c_allocator.free(ids_sorted_host);
    var n_to_sorted: usize = 0;
    const ids_from_sorted_host = ids_sorted_host[n_rows..];
    @memset(ids_from_sorted_host, 0);

    var ids_buf_dev = ggml.cuda.common.PoolAlloc(i32).initAlloc(ctx.pool(), 2 * n_rows);
    defer ids_buf_dev.deinit();

    const tokens_per_expert = std.heap.c_allocator.alloc(i32, @intCast(ne02)) catch ggml.impl.abort("out of memory");
    defer std.heap.c_allocator.free(tokens_per_expert);
    @memset(tokens_per_expert, 0);

    const n_src1_sorted: usize = @intCast(ne12 * n_expert_used * ne10);
    var src1_sorted = ggml.cuda.common.PoolAlloc(u8).initAlloc(ctx.pool(), n_src1_sorted * ts_src1_sorted);
    defer src1_sorted.deinit();
    const n_dst_sorted: usize = @intCast(ne2 * n_expert_used * ne0);
    var dst_sorted = ggml.cuda.common.PoolAlloc(u8).initAlloc(ctx.pool(), n_dst_sorted * ts_dst_sorted);
    defer dst_sorted.deinit();

    const ids_host = std.heap.c_allocator.alloc(u8, ggml.types.ggml_nbytes(ids)) catch ggml.impl.abort("out of memory");
    defer std.heap.c_allocator.free(ids_host);
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpyAsync(ids_host.ptr, ids.data, ggml.types.ggml_nbytes(ids), ggml.cuda.common.cu.cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream)", @src());
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamSynchronize(stream), "cudaStreamSynchronize(stream)", @src());

    var j02: i64 = 0;
    while (j02 < ne02) : (j02 += 1) { // expert matrices
        var j12: i64 = 0;
        while (j12 < ne12) : (j12 += 1) { // tokens
            var iex: i64 = 0;
            while (iex < n_expert_used) : (iex += 1) {
                const off: usize = @as(usize, @intCast(j12)) * ids.nb[1] + @as(usize, @intCast(iex)) * ids.nb[0];
                const expert_to_use = std.mem.readInt(i32, ids_host[off..][0..4], .little);
                // The C's `assert` on `expert_to_use`'s range compiles out.
                if (expert_to_use == j02) {
                    ids_from_sorted_host[@intCast(j12 * n_expert_used + iex)] = @intCast(n_to_sorted);
                    ggml.impl.assert(n_to_sorted < n_rows, "ids_to_sorted_host.size() == size_t(ne_get_rows)");
                    ids_sorted_host[n_to_sorted] = @intCast(j12 * ne11 + @rem(iex, ne11));
                    n_to_sorted += 1;
                    tokens_per_expert[@intCast(j02)] += 1;
                    break;
                }
            }
        }
    }
    ggml.impl.assert(n_to_sorted == n_rows, "ids_to_sorted_host.size() == size_t(ne_get_rows)");

    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpyAsync(@ptrCast(ids_buf_dev.ptr), ids_sorted_host.ptr, 2 * n_rows * @sizeOf(i32), ggml.cuda.common.cu.cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream)", @src());
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamSynchronize(stream), "cudaStreamSynchronize(stream)", @src());

    const ids_dev: [*]i32 = @ptrCast(ids_buf_dev.ptr.?);
    const ids_to_sorted: *const i32 = &ids_dev[0];
    const ids_from_sorted: *const i32 = &ids_dev[n_rows];

    const rows_i32: usize = n_rows * @sizeOf(i32);
    ggml.cuda.getrows.getRowsCuda(src1.data, src1.type, ids_to_sorted, src1_sorted.get(), type_src1_sorted, ne10, nb11, nb12, nb13, ne_get_rows, 1, 1, @sizeOf(i32), rows_i32, rows_i32, @as(usize, @intCast(ne10)) * ts_src1_sorted, n_src1_sorted * ts_src1_sorted, n_src1_sorted * ts_src1_sorted, stream);
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGetLastError(), "cudaGetLastError()", @src());

    var src1_data_cur: [*]u8 = src1_sorted.get().?;
    var dst_data_cur: [*]u8 = dst_sorted.get().?;
    j02 = 0;
    while (j02 < ne02) : (j02 += 1) {
        const tokens = tokens_per_expert[@intCast(j02)];
        if (tokens == 0) continue;

        var src0_slice: ggml.impl.ggml_c.ggml_tensor = src0.*;
        src0_slice.ne[2] = 1;
        src0_slice.nb[3] = src0_slice.nb[2];
        src0_slice.op = ggml.impl.ggml_c.GGML_OP_VIEW;
        src0_slice.view_src = dst.src[0]; // non-const pointer to src0
        src0_slice.data = @as([*]u8, @ptrCast(src0.data.?)) + @as(usize, @intCast(j02)) * nb02;

        var src1_slice: ggml.impl.ggml_c.ggml_tensor = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
        src1_slice.buffer = src1.buffer;
        src1_slice.type = type_src1_sorted;
        src1_slice.ne = .{ ne10, tokens, 1, 1 };
        src1_slice.nb[0] = ts_src1_sorted;
        src1_slice.nb[1] = @as(usize, @intCast(src1_slice.ne[0])) * src1_slice.nb[0];
        src1_slice.nb[2] = @as(usize, @intCast(src1_slice.ne[1])) * src1_slice.nb[1];
        src1_slice.nb[3] = @as(usize, @intCast(src1_slice.ne[2])) * src1_slice.nb[2];
        src1_slice.data = src1_data_cur;

        var dst_slice: ggml.impl.ggml_c.ggml_tensor = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
        dst_slice.op_params = dst.op_params;
        dst_slice.buffer = dst.buffer;
        dst_slice.type = type_dst_sorted;
        dst_slice.ne = .{ ne0, tokens, 1, 1 };
        dst_slice.nb[0] = ts_dst_sorted;
        dst_slice.nb[1] = @as(usize, @intCast(dst_slice.ne[0])) * dst_slice.nb[0];
        dst_slice.nb[2] = @as(usize, @intCast(dst_slice.ne[1])) * dst_slice.nb[1];
        dst_slice.nb[3] = @as(usize, @intCast(dst_slice.ne[2])) * dst_slice.nb[2];
        dst_slice.data = dst_data_cur;

        mulMat(ctx, &src0_slice, &src1_slice, &dst_slice);
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGetLastError(), "cudaGetLastError()", @src());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur += dst_slice.nb[2];
    }

    ggml.cuda.getrows.getRowsCuda(dst_sorted.get(), type_dst_sorted, ids_from_sorted, dst.data, dst.type, ne0, @as(usize, @intCast(ne0)) * ts_dst_sorted, n_rows * @as(usize, @intCast(ne0)) * ts_dst_sorted, n_rows * @as(usize, @intCast(ne0)) * ts_dst_sorted, ne_get_rows, 1, 1, @sizeOf(i32), rows_i32, rows_i32, nb1, nb2, nb3, stream);
}
