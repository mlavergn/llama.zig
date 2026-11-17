//! The OCP FP8 (`F8_E4M3`) matrix multiplication through cuBLASLt: `src1`
//! scaled by its absolute maximum and quantized to FP8, then one FP8 GEMM
//! with the weight's and the activation's scales applied by cuBLASLt.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/fp8.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/fp8.cuh` at v1.0.0 (`b1b770d90`), new in
//! that release (#28413), the host half. The kernels, `fp8_amax_partials`,
//! `fp8_amax_final` and `quantize_fp8_e4m3`, stay CUDA. The arm compiled is
//! the CUDA one (`CUDART_VERSION >= 11080`); the HIP/MUSA stub that returns
//! false is not.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;
const cublasCheck = common.cublasCheck;

/// Ports `fp8_abs_src` (fp8.cu:5 @b1b770d90), passed to `fp8_amax_partials`
/// by value: the source and its shape and strides, in elements.
const AbsSrc = extern struct {
    x: ?*const f32,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    s1: i64,
    s2: i64,
    s3: i64,
};

/// Ports `fp8_set_batch` (fp8.cu:74 @b1b770d90).
fn setBatch(desc: cu.cublasLtMatrixLayout_t, count: i32, stride: i64) void {
    var n = count;
    var s = stride;
    cublasCheck(cu.cublasLtMatrixLayoutSetAttribute(desc, cu.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &n, @sizeOf(i32)), "cublasLtMatrixLayoutSetAttribute(desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count))", @src());
    cublasCheck(cu.cublasLtMatrixLayoutSetAttribute(desc, cu.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &s, @sizeOf(i64)), "cublasLtMatrixLayoutSetAttribute(desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride))", @src());
}

/// Ports `fp8_destroy_matmul` (fp8.cu:79 @b1b770d90).
fn destroyMatmul(op_desc: cu.cublasLtMatmulDesc_t, a_desc: cu.cublasLtMatrixLayout_t, b_desc: cu.cublasLtMatrixLayout_t, d_desc: cu.cublasLtMatrixLayout_t) void {
    cublasCheck(cu.cublasLtMatrixLayoutDestroy(d_desc), "cublasLtMatrixLayoutDestroy(d_desc)", @src());
    cublasCheck(cu.cublasLtMatrixLayoutDestroy(b_desc), "cublasLtMatrixLayoutDestroy(b_desc)", @src());
    cublasCheck(cu.cublasLtMatrixLayoutDestroy(a_desc), "cublasLtMatrixLayoutDestroy(a_desc)", @src());
    cublasCheck(cu.cublasLtMatmulDescDestroy(op_desc), "cublasLtMatmulDescDestroy(op_desc)", @src());
}

fn u32Of(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// Ports `ggml_cuda_mul_mat_fp8` (fp8.cu:88 @b1b770d90): false, having done
/// nothing, where the hardware, the types or the layout do not fit, or
/// cuBLASLt offers no algorithm. `weight_scale` (one `F32`) is the
/// per-tensor weight scale a fused `MUL` applies.
pub fn mulMatFp8(ctx: *common.Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor, weight_scale: ?*const Tensor) bool {
    const x = common.info().devices[@intCast(ctx.device)].cc;
    if (!common.fp8MmaHardwareAvailable(x) or src0.type != c.GGML_TYPE_F8_E4M3 or src1.type != c.GGML_TYPE_F32 or
        dst.type != c.GGML_TYPE_F32 or !c.ggml_is_contiguous(dst) or @rem(src0.ne[0], 16) != 0 or @rem(src0.ne[1], 16) != 0 or
        src0.nb[0] != @sizeOf(u8) or src0.nb[1] != @as(usize, @bitCast(src0.ne[0])) or src1.nb[0] != @sizeOf(f32) or
        !common.isAligned(src0, 16) or !common.isAligned(dst, 16))
    {
        return false;
    }

    if (weight_scale) |w| {
        impl.assert(w.type == c.GGML_TYPE_F32 and c.ggml_is_contiguous(w) and c.ggml_nelements(w) == 1, "!weight_scale || (weight_scale->type == GGML_TYPE_F32 && ggml_is_contiguous(weight_scale) && ggml_nelements(weight_scale) == 1)");
    }

    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];
    const nb02 = src0.nb[2];
    const nb03 = src0.nb[3];
    const ne10 = src1.ne[0];
    const ne11 = src1.ne[1];
    const ne12 = src1.ne[2];
    const ne13 = src1.ne[3];
    const nb11 = src1.nb[1];
    const nb12 = src1.nb[2];
    const nb13 = src1.nb[3];
    const ne0 = dst.ne[0];
    impl.assert(ne10 == ne00, "ne10 == ne00");
    impl.assert(ne0 == ne01, "ne0 == ne01");
    impl.assert(@rem(ne12, ne02) == 0, "ne12 % ne02 == 0");
    impl.assert(@rem(ne13, ne03) == 0, "ne13 % ne03 == 0");

    const stream = ctx.stream();
    const ne_src1 = c.ggml_nelements(src1);
    var src1_fp8 = common.PoolAlloc(u8).initAlloc(ctx.pool(), @intCast(ne_src1));
    defer src1_fp8.deinit();
    var src1_scale = common.PoolAlloc(f32).initAlloc(ctx.pool(), 1);
    defer src1_scale.deinit();
    var src1_amax = common.PoolAlloc(f32).initAlloc(ctx.pool(), 1);
    defer src1_amax.deinit();

    const s11: i64 = @intCast(nb11 / @sizeOf(f32));
    const s12: i64 = @intCast(nb12 / @sizeOf(f32));
    const s13: i64 = @intCast(nb13 / @sizeOf(f32));
    const abs_src: AbsSrc = .{ .x = @ptrCast(@alignCast(src1.data)), .ne0 = ne10, .ne1 = ne11, .ne2 = ne12, .s1 = s11, .s2 = s12, .s3 = s13 };
    const reduce_blocks: c_int = @intCast(@min(@divTrunc(ne_src1 + 255, 256), 1024));
    var reduce_tmp = common.PoolAlloc(f32).initAlloc(ctx.pool(), @intCast(reduce_blocks));
    defer reduce_tmp.deinit();
    const reduce_ptr: ?*f32 = &reduce_tmp.ptr.?[0];
    const amax_ptr: ?*f32 = &src1_amax.ptr.?[0];
    const scale_ptr: ?*f32 = &src1_scale.ptr.?[0];
    kernels.launch(kernels.get("fp8", "fp8_amax_partials"), .{ .grid = .{ .x = @bitCast(reduce_blocks) }, .block = .{ .x = 256 }, .stream = stream }, .{ abs_src, ne_src1, reduce_ptr });
    kernels.launch(kernels.get("fp8", "fp8_amax_final"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 }, .stream = stream }, .{ @as(?*const f32, reduce_ptr), reduce_blocks, amax_ptr });

    kernels.launch(kernels.get("fp8", "quantize_fp8_e4m3"), .{ .grid = .{ .x = u32Of(@divTrunc(ne_src1 + 255, 256)) }, .block = .{ .x = 256 }, .stream = stream }, .{
        @as(?*const f32, @ptrCast(@alignCast(src1.data))), @as(?*u8, &src1_fp8.ptr.?[0]), @as(?*const f32, amax_ptr), scale_ptr,
        ne10,                                              ne11,                          ne12,                       ne_src1,
        s11,                                               s12,                           s13,
    });
    common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());

    var op_desc: cu.cublasLtMatmulDesc_t = null;
    var a_desc: cu.cublasLtMatrixLayout_t = null;
    var b_desc: cu.cublasLtMatrixLayout_t = null;
    var d_desc: cu.cublasLtMatrixLayout_t = null;
    cublasCheck(cu.cublasLtMatmulDescCreate(&op_desc, cu.CUBLAS_COMPUTE_32F, cu.CUDA_R_32F), "cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F)", @src());
    const trans_a: cu.cublasOperation_t = cu.CUBLAS_OP_T;
    cublasCheck(cu.cublasLtMatmulDescSetAttribute(op_desc, cu.CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, @sizeOf(cu.cublasOperation_t)), "cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a))", @src());
    cublasCheck(cu.cublasLtMatmulDescSetAttribute(op_desc, cu.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, @ptrCast(&src1_scale.ptr), @sizeOf(?[*]f32)), "cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &src1_scale.ptr, sizeof(src1_scale.ptr))", @src());
    if (weight_scale) |w| {
        const w_ptr: ?*const f32 = @ptrCast(@alignCast(w.data));
        cublasCheck(cu.cublasLtMatmulDescSetAttribute(op_desc, cu.CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, @ptrCast(&w_ptr), @sizeOf(?*const f32)), "cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_ptr, sizeof(scale_ptr))", @src());
    }
    const r2 = @divTrunc(ne12, ne02);
    const r3 = @divTrunc(ne13, ne03);

    // src1 batches that share a src0 matrix are adjacent in the compact src1 copy and in dst, fold them into N
    const n: i64 = ne11 * (if (ne02 == 1) r2 * r3 else r2);

    // batch the remaining matmuls with a fixed src0 stride, or issue one batched call per i3 if there is none
    var batch_count: i64 = ne02;
    var stride_a: i64 = @intCast(nb02);
    var n_calls: i64 = ne13;
    if (ne02 == 1) {
        batch_count = ne03;
        stride_a = @intCast(nb03);
        n_calls = 1;
    } else if (r3 == 1 and (ne03 == 1 or nb03 == @as(usize, @bitCast(ne02)) * nb02)) {
        batch_count = ne02 * ne03;
        n_calls = 1;
    }

    const u = struct {
        fn of(v: i64) u64 {
            return @bitCast(v);
        }
    }.of;
    cublasCheck(cu.cublasLtMatrixLayoutCreate(&a_desc, cu.CUDA_R_8F_E4M3, u(ne00), u(ne01), ne00), "cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, ne00, ne01, ne00)", @src());
    cublasCheck(cu.cublasLtMatrixLayoutCreate(&b_desc, cu.CUDA_R_8F_E4M3, u(ne10), u(n), ne10), "cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, ne10, n, ne10)", @src());
    cublasCheck(cu.cublasLtMatrixLayoutCreate(&d_desc, cu.CUDA_R_32F, u(ne0), u(n), ne0), "cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_32F, ne0, n, ne0)", @src());
    if (batch_count > 1) {
        setBatch(a_desc, @truncate(batch_count), stride_a);
        setBatch(b_desc, @truncate(batch_count), ne10 * n);
        setBatch(d_desc, @truncate(batch_count), ne0 * n);
    }

    // TODO: Allow workspace in the heuristic search for potential performance gains.
    // cuBLASLt takes caller-owned workspace per matmul call.
    var preference: cu.cublasLtMatmulPreference_t = null;
    cublasCheck(cu.cublasLtMatmulPreferenceCreate(&preference), "cublasLtMatmulPreferenceCreate(&preference)", @src());
    var heuristic: cu.cublasLtMatmulHeuristicResult_t = undefined;
    var returned: c_int = 0;
    const status = cu.cublasLtMatmulAlgoGetHeuristic(ctx.cublasltHandle(), op_desc, a_desc, b_desc, d_desc, d_desc, preference, 1, &heuristic, &returned);
    cublasCheck(cu.cublasLtMatmulPreferenceDestroy(preference), "cublasLtMatmulPreferenceDestroy(preference)", @src());
    if (status != cu.CUBLAS_STATUS_SUCCESS or returned == 0) {
        destroyMatmul(op_desc, a_desc, b_desc, d_desc);
        return false;
    }

    const alpha: f32 = 1.0;
    const beta: f32 = 0.0;
    // `i3` is a Zig type name; `j3` is the C's `i3`.
    var j3: i64 = 0;
    while (j3 < n_calls) : (j3 += 1) {
        const a: [*]const u8 = @as([*]const u8, @ptrCast(src0.data.?)) + @as(usize, @intCast(@divTrunc(j3, r3))) * nb03;
        const b: [*]const u8 = src1_fp8.ptr.? + @as(usize, @intCast(j3 * ne12 * ne11 * ne10));
        const d: [*]u8 = @as([*]u8, @ptrCast(dst.data.?)) + @as(usize, @intCast(j3)) * dst.nb[3];
        cublasCheck(cu.cublasLtMatmul(ctx.cublasltHandle(), op_desc, &alpha, a, a_desc, b, b_desc, &beta, d, d_desc, d, d_desc, &heuristic.algo, null, 0, stream), "cublasLtMatmul(ctx.cublaslt_handle(), op_desc, &alpha, a, a_desc, b, b_desc, &beta, d, d_desc, d, d_desc, &heuristic.algo, nullptr, 0, stream)", @src());
    }

    destroyMatmul(op_desc, a_desc, b_desc, d_desc);
    return true;
}
