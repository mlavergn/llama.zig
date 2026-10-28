//! `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID`: choosing among MMVF, MMF,
//! MMVQ, MMQ and cuBLAS, the cuBLAS path itself, and the predicates the
//! fusion pass asks before fusing a matrix multiplication.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` at v0.4.0
//! (`5266f24da`): `batched_mul_mat_traits`, `ggml_cuda_mul_mat_cublas_impl`,
//! `ggml_cuda_mul_mat_cublas`, `ggml_cuda_should_fuse_mul_mat`,
//! `ggml_cuda_should_fuse_mul_mat_vec_{f,q}`, `ggml_cuda_mul_mat`,
//! `ggml_cuda_mul_mat_id_needs_sync` and `ggml_cuda_mul_mat_id`. The kernel
//! `k_compute_batched_ptrs` stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");
const convert = @import("convert.zig");
const mmvq = @import("mmvq.zig");
const mmvf = @import("mmvf.zig");
const mmq = @import("mmq.zig");
const mmf = @import("mmf.zig");
const fwht = @import("fwht.zig");
const getrows = @import("getrows.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;
const half = kernels.half;
const bf16 = kernels.bf16;

fn i32Of(x: i64) c_int {
    return @truncate(x);
}

/// Ports `batched_mul_mat_traits` (ggml-cuda.cu:1367 @5266f24da), for each `T`.
const Traits = struct {
    cuda_type_size: usize,
    compute_type: cu.cublasComputeType_t,
    data_type: cu.cudaDataType,
    ggml_type_val: c.enum_ggml_type,
    alpha: *const anyopaque,
    beta: *const anyopaque,
    convert: *const fn (c.enum_ggml_type) ?convert.ToFn,
    convert_nc: *const fn (c.enum_ggml_type) ?convert.ToNcFn,
};

const one_f32: f32 = 1.0;
const zero_f32: f32 = 0.0;
/// `half(1.0)` and `half(0.0)`.
const one_f16: u16 = 0x3C00;
const zero_f16: u16 = 0x0000;

fn traitsOf(comptime t: c.enum_ggml_type) Traits {
    return switch (t) {
        c.GGML_TYPE_F32 => .{ .cuda_type_size = 4, .compute_type = cu.CUBLAS_COMPUTE_32F, .data_type = cu.CUDA_R_32F, .ggml_type_val = c.GGML_TYPE_F32, .alpha = &one_f32, .beta = &zero_f32, .convert = convert.getToFp32, .convert_nc = convert.getToFp32Nc },
        c.GGML_TYPE_BF16 => .{ .cuda_type_size = 2, .compute_type = cu.CUBLAS_COMPUTE_32F, .data_type = cu.CUDA_R_16BF, .ggml_type_val = c.GGML_TYPE_BF16, .alpha = &one_f32, .beta = &zero_f32, .convert = convert.getToBf16, .convert_nc = convert.getToBf16Nc },
        c.GGML_TYPE_F16 => .{ .cuda_type_size = 2, .compute_type = cu.CUBLAS_COMPUTE_16F, .data_type = cu.CUDA_R_16F, .ggml_type_val = c.GGML_TYPE_F16, .alpha = &one_f16, .beta = &zero_f16, .convert = convert.getToFp16, .convert_nc = convert.getToFp16Nc },
        else => @compileError("no batched_mul_mat_traits"),
    };
}

/// Ports `ggml_cuda_mul_mat_cublas_impl` (ggml-cuda.cu:1412 @5266f24da).
fn mulMatCublasImpl(comptime compute_type: c.enum_ggml_type, ctx: *common.Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    const traits = comptime traitsOf(compute_type);
    const ts_cuda = traits.cuda_type_size;

    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];
    const ne10 = src1.ne[0];
    const ne11 = src1.ne[1];
    const ne12 = src1.ne[2];
    const ne13 = src1.ne[3];
    const ne0 = dst.ne[0];
    const ne1 = dst.ne[1];

    const ne_dst = c.ggml_nelements(dst);
    const main_stream = ctx.stream();
    const cublas_h = ctx.cublasHandle();

    const src0_ts = c.ggml_type_size(src0.type);
    impl.assert(src0.nb[0] == src0_ts, "nb00 == src0_ts");
    var s01: i64 = @intCast(src0.nb[1] / src0_ts);
    var s02: i64 = @intCast(src0.nb[2] / src0_ts);
    var s03: i64 = @intCast(src0.nb[3] / src0_ts);

    const src1_ts = c.ggml_type_size(src1.type);
    impl.assert(src1.nb[0] == src1_ts, "nb10 == src1_ts");
    var s11: i64 = @intCast(src1.nb[1] / src1_ts);
    var s12: i64 = @intCast(src1.nb[2] / src1_ts);
    var s13: i64 = @intCast(src1.nb[3] / src1_ts);

    const dst_ddf: ?*f32 = @ptrCast(@alignCast(dst.data));

    var src0_ptr: ?*const anyopaque = null;
    var src1_ptr: ?*const anyopaque = null;

    // `ggml_cuda_pool_alloc<cuda_t>`, counted in bytes here.
    var src0_alloc = common.PoolAlloc(u8).init(ctx.pool());
    defer src0_alloc.deinit();
    var src1_alloc = common.PoolAlloc(u8).init(ctx.pool());
    defer src1_alloc.deinit();

    var is_src0_cont_2 = c.ggml_is_contiguous_2(src0);
    var is_src1_cont_2 = c.ggml_is_contiguous_2(src1);

    if (src0.type == compute_type) {
        src0_ptr = src0.data;
    } else {
        _ = src0_alloc.alloc(@as(usize, @intCast(c.ggml_nelements(src0))) * ts_cuda);

        if (c.ggml_is_contiguously_allocated(src0)) {
            const convert_func = traits.convert(src0.type);
            impl.assert(convert_func != null, "convert_func != nullptr");
            convert_func.?(src0.data, src0_alloc.get(), c.ggml_nelements(src0), main_stream);
            const src0_bs: i64 = c.ggml_blck_size(src0.type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const convert_func = traits.convert_nc(src0.type);
            impl.assert(convert_func != null, "convert_func != nullptr");
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
        _ = src1_alloc.alloc(@as(usize, @intCast(c.ggml_nelements(src1))) * ts_cuda);

        if (c.ggml_is_contiguously_allocated(src1)) {
            const convert_func = traits.convert(src1.type);
            impl.assert(convert_func != null, "convert_func != nullptr");
            convert_func.?(src1.data, src1_alloc.get(), c.ggml_nelements(src1), main_stream);
            const src1_bs: i64 = c.ggml_blck_size(src1.type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const convert_func = traits.convert_nc(src1.type);
            impl.assert(convert_func != null, "convert_func != nullptr");
            convert_func.?(src1.data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11 * s11;
            s13 = ne12 * s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    var dst_temp = common.PoolAlloc(u8).init(ctx.pool());
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

    const x = common.info().devices[@intCast(ctx.device)].cc;
    var prefer_f32_output = false;
    if (compute_type == c.GGML_TYPE_F16) {
        // `cc == VOLTA || RDNA4 || CDNA`; the AMD tests are false here.
        prefer_f32_output = x == common.cc.volta;
    } else if (compute_type == c.GGML_TYPE_BF16) {
        prefer_f32_output = true;
    }

    if (prefer_f32_output) {
        dst_ptr = dst_ddf;
        const f32_traits = comptime traitsOf(c.GGML_TYPE_F32);
        cu_compute_type = f32_traits.compute_type;
        cu_data_type = f32_traits.data_type;
        alpha = f32_traits.alpha;
        beta = f32_traits.beta;
    } else {
        if (compute_type == c.GGML_TYPE_F32) {
            dst_ptr = dst_ddf; // Direct F32 output
        } else {
            dst_ptr = dst_temp.alloc(@as(usize, @intCast(ne_dst)) * ts_cuda);
            nbd2 /= @sizeOf(f32) / ts_cuda;
            nbd3 /= @sizeOf(f32) / ts_cuda;
        }
    }

    impl.assert(@rem(ne12, ne02) == 0, "ne12 % ne02 == 0");
    impl.assert(@rem(ne13, ne03) == 0, "ne13 % ne03 == 0");

    // broadcast factors
    const r2 = @divTrunc(ne12, ne02);
    const r3 = @divTrunc(ne13, ne03);

    if (compute_type == c.GGML_TYPE_F32 and ne12 == 1 and ne13 == 1) {
        common.cublasCheck(cu.cublasSgemm_v2(cublas_h, cu.CUBLAS_OP_T, cu.CUBLAS_OP_N, i32Of(ne01), i32Of(ne11), i32Of(ne10), @ptrCast(@alignCast(alpha)), @ptrCast(@alignCast(src0_ptr)), i32Of(s01), @ptrCast(@alignCast(src1_ptr)), i32Of(s11), @ptrCast(@alignCast(beta)), @ptrCast(@alignCast(dst_ptr)), i32Of(ne0)), "cublasSgemm(...)", @src());
    } else if (ne12 == 1 and ne13 == 1) {
        common.cublasCheck(cu.cublasGemmEx(cublas_h, cu.CUBLAS_OP_T, cu.CUBLAS_OP_N, i32Of(ne01), i32Of(ne11), i32Of(ne10), alpha, src0_ptr, cu_data_type_a, i32Of(s01), src1_ptr, cu_data_type_b, i32Of(s11), beta, dst_ptr, cu_data_type, i32Of(ne0), cu_compute_type, cu.CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx(...)", @src());
    } else if (r2 == 1 and r3 == 1 and is_src0_cont_2 and is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be
        // determined from dim 3:
        const sma = if (ne02 == 1) s03 else s02;
        const smb = if (ne12 == 1) s13 else s12;

        // there is no broadcast and src0, src1 are contiguous across dims
        // 2, 3: use cublasGemmStridedBatchedEx
        common.cublasCheck(cu.cublasGemmStridedBatchedEx(cublas_h, cu.CUBLAS_OP_T, cu.CUBLAS_OP_N, i32Of(ne01), i32Of(ne11), i32Of(ne10), alpha, src0_ptr, cu_data_type_a, i32Of(s01), sma, src1_ptr, cu_data_type_b, i32Of(s11), smb, beta, dst_ptr, cu_data_type, i32Of(ne0), ne1 * ne0, i32Of(ne12 * ne13), cu_compute_type, cu.CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmStridedBatchedEx(...)", @src());
    } else {
        // use cublasGemmBatchedEx
        const ne23 = ne12 * ne13;

        var ptrs_src = common.PoolAlloc(?*const anyopaque).initAlloc(ctx.pool(), @intCast(2 * ne23));
        defer ptrs_src.deinit();
        var ptrs_dst = common.PoolAlloc(?*anyopaque).initAlloc(ctx.pool(), @intCast(1 * ne23));
        defer ptrs_dst.deinit();

        const src_type_size = ts_cuda;

        const threads_x = 16;
        const threads_y = 16;
        kernels.launch(comptime kernels.get("ggml-cuda", "k_compute_batched_ptrs"), .{
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

        common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());

        const ps = ptrs_src.get().?;
        common.cublasCheck(cu.cublasGemmBatchedEx(cublas_h, cu.CUBLAS_OP_T, cu.CUBLAS_OP_N, i32Of(ne01), i32Of(ne11), i32Of(ne10), alpha, ps + 0 * @as(usize, @intCast(ne23)), cu_data_type_a, i32Of(s01), ps + 1 * @as(usize, @intCast(ne23)), cu_data_type_b, i32Of(s11), beta, @ptrCast(ptrs_dst.get().?), cu_data_type, i32Of(ne0), i32Of(ne23), cu_compute_type, cu.CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmBatchedEx(...)", @src());
    }

    // Convert output back to F32 if needed
    if (cu_data_type != cu.CUDA_R_32F) {
        const to_fp32_cuda = convert.getToFp32(traits.ggml_type_val).?;
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

/// Ports `ggml_cuda_mul_mat_cublas` (ggml-cuda.cu:1625 @5266f24da).
pub fn mulMatCublas(ctx: *common.Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    const x = common.info().devices[@intCast(ctx.device)].cc;
    var compute_type = src0.type;
    if (c.ggml_is_quantized(compute_type)) {
        compute_type = if (common.fastFp16HardwareAvailable(x)) c.GGML_TYPE_F16 else c.GGML_TYPE_F32;
    } else if (compute_type == c.GGML_TYPE_F16 and !common.fastFp16HardwareAvailable(x)) {
        compute_type = c.GGML_TYPE_F32;
    }
    if (dst.op_params[0] == c.GGML_PREC_F32) {
        compute_type = c.GGML_TYPE_F32;
    }

    if (std.c.getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE")) |env_c| {
        var buf: [64]u8 = undefined;
        const env = std.mem.span(env_c);
        const n = @min(env.len, buf.len);
        for (env[0..n], 0..) |ch, i| buf[i] = std.ascii.toLower(ch);
        const env_cpp = buf[0..n];
        if (std.mem.eql(u8, env_cpp, "f32") or std.mem.eql(u8, env_cpp, "fp32")) {
            compute_type = c.GGML_TYPE_F32;
        } else if (std.mem.eql(u8, env_cpp, "f16") or std.mem.eql(u8, env_cpp, "fp16")) {
            compute_type = c.GGML_TYPE_F16;
        } else if (std.mem.eql(u8, env_cpp, "bf16")) {
            compute_type = c.GGML_TYPE_BF16;
        } else if (!std.mem.eql(u8, env_cpp, "auto")) {
            var z: [65]u8 = undefined;
            @memcpy(z[0..n], env_cpp);
            z[n] = 0;
            impl.logWarn("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", .{ "ggml_cuda_mul_mat_cublas", @as([*:0]const u8, @ptrCast(&z)) });
        }
    }

    switch (compute_type) {
        c.GGML_TYPE_F32 => mulMatCublasImpl(c.GGML_TYPE_F32, ctx, src0, src1, dst),
        c.GGML_TYPE_BF16 => mulMatCublasImpl(c.GGML_TYPE_BF16, ctx, src0, src1, dst),
        c.GGML_TYPE_F16 => mulMatCublasImpl(c.GGML_TYPE_F16, ctx, src0, src1, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_cuda_should_fuse_mul_mat` (ggml-cuda.cu:1668 @5266f24da).
pub fn shouldFuseMulMat(ffn_up: *const Tensor, ffn_gate: *const Tensor, glu: *const Tensor, ffn_up_bias: ?*const Tensor, ffn_gate_bias: ?*const Tensor, ffn_up_scale: ?*const Tensor, ffn_gate_scale: ?*const Tensor) bool {
    const has_bias = ffn_up_bias != null or ffn_gate_bias != null;
    const has_scale = ffn_up_scale != null or ffn_gate_scale != null;

    if (has_bias and (ffn_up_bias == null or ffn_gate_bias == null)) return false;
    if (has_scale and (ffn_up_scale == null or ffn_gate_scale == null)) return false;

    const is_mul_mat = ffn_up.op == c.GGML_OP_MUL_MAT and ffn_gate.op == c.GGML_OP_MUL_MAT and glu.op == c.GGML_OP_GLU;
    const is_mul_mat_id = ffn_up.op == c.GGML_OP_MUL_MAT_ID and ffn_gate.op == c.GGML_OP_MUL_MAT_ID and glu.op == c.GGML_OP_GLU;

    if (!is_mul_mat and !is_mul_mat_id) return false;

    const expected_bias_op: c.enum_ggml_op = if (is_mul_mat) c.GGML_OP_ADD else c.GGML_OP_ADD_ID;
    const ffn_up_bias_src: *const Tensor = if (has_scale) ffn_up_scale.? else ffn_up;
    const ffn_gate_bias_src: *const Tensor = if (has_scale) ffn_gate_scale.? else ffn_gate;
    const ffn_up_out: *const Tensor = if (has_bias) ffn_up_bias.? else ffn_up_bias_src;
    const ffn_gate_out: *const Tensor = if (has_bias) ffn_gate_bias.? else ffn_gate_bias_src;

    if (glu.src[0] != ffn_gate_out or glu.src[1] != ffn_up_out) return false;

    if (has_scale) {
        const us = ffn_up_scale.?;
        const gs = ffn_gate_scale.?;
        if (us.op != c.GGML_OP_MUL or gs.op != c.GGML_OP_MUL) return false;
        const up_has_mm = us.src[0] == ffn_up or us.src[1] == ffn_up;
        const gate_has_mm = gs.src[0] == ffn_gate or gs.src[1] == ffn_gate;
        if (!up_has_mm or !gate_has_mm) return false;
    }

    if (has_bias) {
        const ub = ffn_up_bias.?;
        const gb = ffn_gate_bias.?;
        if (ub.op != expected_bias_op or gb.op != expected_bias_op) return false;

        if (expected_bias_op == c.GGML_OP_ADD) {
            const up_has_mul = ub.src[0] == ffn_up_bias_src or ub.src[1] == ffn_up_bias_src;
            const gate_has_mul = gb.src[0] == ffn_gate_bias_src or gb.src[1] == ffn_gate_bias_src;
            if (!up_has_mul or !gate_has_mul) return false;
        } else { // GGML_OP_ADD_ID
            if (ub.src[0] != ffn_up_bias_src or gb.src[0] != ffn_gate_bias_src) return false;
            if (ub.src[2] != ffn_up.src[2] or gb.src[2] != ffn_gate.src[2]) return false;
        }
    }

    const up0 = impl.one(Tensor, ffn_up.src[0]);
    const gate0 = impl.one(Tensor, ffn_gate.src[0]);
    if (up0.type != gate0.type or !c.ggml_are_same_shape(up0, gate0) or !c.ggml_are_same_stride(up0, gate0)) return false;

    if (ffn_up.src[1] != ffn_gate.src[1]) return false;

    if (is_mul_mat_id and ffn_up.src[2] != ffn_gate.src[2]) return false;

    const glu_op = c.ggml_get_glu_op(glu);
    if (glu_op != c.GGML_GLU_OP_SWIGLU and glu_op != c.GGML_GLU_OP_GEGLU and glu_op != c.GGML_GLU_OP_SWIGLU_OAI and
        glu_op != c.GGML_GLU_OP_SWIGLU_CLAMP) return false;

    // swapped
    if (impl.getOpParamsI32(glu, 1) != 0) return false;

    return true;
}

/// Ports `ggml_cuda_should_fuse_mul_mat_vec_f` (ggml-cuda.cu:1762 @5266f24da).
pub fn shouldFuseMulMatVecF(tensor: *const Tensor) bool {
    const src0 = impl.one(Tensor, tensor.src[0]);
    const src1 = impl.one(Tensor, tensor.src[1]);
    const dst = tensor;

    const is_mul_mat_id = tensor.op == c.GGML_OP_MUL_MAT_ID;

    var use_mul_mat_vec_f = (src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16 or src0.type == c.GGML_TYPE_BF16) and
        src1.type == c.GGML_TYPE_F32 and dst.type == c.GGML_TYPE_F32;

    const x = common.info().devices[@intCast(common.getDevice())].cc;
    use_mul_mat_vec_f = use_mul_mat_vec_f and mmvf.shouldUseMmvf(src0.type, x, &src0.ne, &src0.nb, if (is_mul_mat_id) src1.ne[2] else src1.ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor.op == c.GGML_OP_MUL_MAT and dst.ne[1] != 1) return false;
    if (tensor.op == c.GGML_OP_MUL_MAT_ID and dst.ne[2] != 1) return false;

    return use_mul_mat_vec_f;
}

/// Ports `ggml_cuda_should_fuse_mul_mat_vec_q` (ggml-cuda.cu:1789 @5266f24da).
pub fn shouldFuseMulMatVecQ(tensor: *const Tensor) bool {
    const src0 = impl.one(Tensor, tensor.src[0]);
    const src1 = impl.one(Tensor, tensor.src[1]);
    const dst = tensor;

    const bad_padding_clear = c.ggml_backend_buffer_get_usage(src0.buffer) == c.GGML_BACKEND_BUFFER_USAGE_COMPUTE and
        c.ggml_nbytes(src0) != c.ggml_backend_buffer_get_alloc_size(src0.buffer, src0) and src0.view_src != null;

    const use_mul_mat_vec_q = c.ggml_is_quantized(src0.type) and !bad_padding_clear and src1.type == c.GGML_TYPE_F32 and
        dst.type == c.GGML_TYPE_F32 and src1.ne[1] <= mmvq.max_batch_size;

    // fusion is not universally faster on Pascal
    const x = common.info().devices[@intCast(common.getDevice())].cc;
    if (x <= common.cc.pascal) return false;
    //we only support fusion for ncols_dst = 1
    if (tensor.op == c.GGML_OP_MUL_MAT and dst.ne[1] != 1) return false;
    if (tensor.op == c.GGML_OP_MUL_MAT_ID and dst.ne[2] > mmvq.getMmidMaxBatch(src0.type, x)) return false;

    return use_mul_mat_vec_q;
}

/// Ports `ggml_cuda_mul_mat` (ggml-cuda.cu:1818 @5266f24da).
pub fn mulMat(ctx: *common.Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    const ne01 = src0.ne[1];
    const ne11 = src1.ne[1];
    const ne2 = dst.ne[2];
    const ne3 = dst.ne[3];

    const hint = impl.getOpParamsI32(dst, 1);
    if (hint == c.GGML_HINT_SRC0_IS_HADAMARD and fwht.opFwht(ctx, src1, dst)) return;

    // If src0 is a temporary compute buffer it may have some padding that
    // needs to be cleared for mul_mat_vec_q or mul_mat_q. But if src0 is
    // also a view of another tensor then this cannot be done safely, so
    // cuBLAS is used.
    const bad_padding_clear = c.ggml_backend_buffer_get_usage(src0.buffer) == c.GGML_BACKEND_BUFFER_USAGE_COMPUTE and
        c.ggml_nbytes(src0) != c.ggml_backend_buffer_get_alloc_size(src0.buffer, src0) and src0.view_src != null;
    if (bad_padding_clear or src1.type != c.GGML_TYPE_F32 or dst.type != c.GGML_TYPE_F32) {
        mulMatCublas(ctx, src0, src1, dst);
        return;
    }

    const x = common.info().devices[@intCast(ctx.device)].cc;
    const warp_size = common.info().devices[@intCast(ctx.device)].warp_size;

    if (mmvf.shouldUseMmvf(src0.type, x, &src0.ne, &src0.nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        mmvf.mulMatVecF(ctx, src0, src1, null, dst, null);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 and ne11 > mmvf.max_batch_size and ne2 == 1 and ne3 == 1 and
        src0.type == c.GGML_TYPE_F32 and
        c.ggml_is_contiguous(src0) and c.ggml_is_contiguous(src1) and c.ggml_is_contiguous(dst) and
        mmvf.shouldUseMmvf(src1.type, x, &src1.ne, &src1.nb, 1))
    {
        var dst_vec = dst.*;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0] * @as(usize, @intCast(ne11));
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        mmvf.mulMatVecF(ctx, src1, src0, null, &dst_vec, null);
        return;
    }
    if (mmf.shouldUseMmf(src0.type, x, warp_size, &src0.ne, &src0.nb, i32Of(ne11), false)) {
        mmf.mulMatF(ctx, src0, src1, null, dst);
        return;
    }
    if (mmvq.shouldUseMmvq(src0.type, x, ne11)) {
        mmvq.mulMatVecQ(ctx, src0, src1, null, dst, null);
        return;
    }
    if (mmq.shouldUseMmq(src0.type, x, ne11, 0)) {
        mmq.mulMatQ(ctx, src0, src1, null, dst);
        return;
    }
    mulMatCublas(ctx, src0, src1, dst);
}

/// Ports `ggml_cuda_mul_mat_id_needs_sync` (ggml-cuda.cu:1876 @5266f24da).
pub fn mulMatIdNeedsSync(dst: *const Tensor, x: c_int) bool {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    if (src1.type != c.GGML_TYPE_F32 or dst.type != c.GGML_TYPE_F32) return true;

    if (dst.ne[2] <= mmvq.max_batch_size) {
        if (c.ggml_is_quantized(src0.type)) {
            if (dst.ne[2] <= mmvq.getMmidMaxBatch(src0.type, x)) return false;
        } else if (common.cc.isAmd(x)) {
            return false;
        }
    }

    if (mmq.shouldUseMmq(src0.type, x, src1.ne[2], src0.ne[2])) return false;

    if (mmf.shouldUseMmf(src0.type, x, common.warp_size, &src0.ne, &src0.nb, i32Of(src1.ne[2]), true)) return false;

    return true;
}

/// Ports `ggml_cuda_mul_mat_id` (ggml-cuda.cu:1905 @5266f24da).
pub fn mulMatId(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);
    const ids = impl.one(Tensor, dst.src[2]);

    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type  == GGML_TYPE_F32");

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

    const x = common.info().devices[@intCast(common.getDevice())].cc;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1.type == c.GGML_TYPE_F32 and dst.type == c.GGML_TYPE_F32) {
        comptime std.debug.assert(mmvq.max_batch_size == mmvf.max_batch_size);
        if (ne2 <= mmvq.max_batch_size) {
            if (c.ggml_is_quantized(src0.type)) {
                const mmvq_mmid_max = mmvq.getMmidMaxBatch(src0.type, x);
                if (ne2 <= mmvq_mmid_max) {
                    mmvq.mulMatVecQ(ctx, src0, src1, ids, dst, null);
                    return;
                }
            } else {
                if (common.cc.isAmd(x)) {
                    mmvf.mulMatVecF(ctx, src0, src1, ids, dst, null);
                    return;
                }
            }
        }

        if (mmq.shouldUseMmq(src0.type, x, ne12, ne02)) {
            mmq.mulMatQ(ctx, src0, src1, ids, dst);
            return;
        }

        if (mmf.shouldUseMmf(src0.type, x, common.warp_size, &src0.ne, &src0.nb, i32Of(src1.ne[2]), true)) {
            mmf.mulMatF(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs,
    // because it requires stream synchronization
    impl.assert(mulMatIdNeedsSync(dst, x), "ggml_cuda_mul_mat_id_needs_sync(dst, cc)");
    const stream = ctx.stream();

    impl.assert(nb12 % nb11 == 0, "nb12 % nb11 == 0");
    impl.assert(nb2 % nb1 == 0, "nb2  % nb1  == 0");

    const type_src1_sorted: c.enum_ggml_type = if ((src0.type == c.GGML_TYPE_F16 and !common.fastFp16HardwareAvailable(x)) or
        c.ggml_is_quantized(src0.type)) c.GGML_TYPE_F32 else src0.type;
    const type_dst_sorted: c.enum_ggml_type = c.GGML_TYPE_F32;
    const ts_src1_sorted = c.ggml_type_size(type_src1_sorted);
    const ts_dst_sorted = c.ggml_type_size(type_dst_sorted);

    const n_expert_used = ids.ne[0];
    const ne_get_rows = ne12 * n_expert_used;
    const n_rows: usize = @intCast(ne_get_rows);

    // `ids_to_sorted_host`, with `ids_from_sorted_host` appended after it
    // as the C does once the loop is done.
    const ids_sorted_host = common.allocator.alloc(i32, 2 * n_rows) catch impl.abort("out of memory");
    defer common.allocator.free(ids_sorted_host);
    var n_to_sorted: usize = 0;
    const ids_from_sorted_host = ids_sorted_host[n_rows..];
    @memset(ids_from_sorted_host, 0);

    var ids_buf_dev = common.PoolAlloc(i32).initAlloc(ctx.pool(), 2 * n_rows);
    defer ids_buf_dev.deinit();

    const tokens_per_expert = common.allocator.alloc(i32, @intCast(ne02)) catch impl.abort("out of memory");
    defer common.allocator.free(tokens_per_expert);
    @memset(tokens_per_expert, 0);

    const n_src1_sorted: usize = @intCast(ne12 * n_expert_used * ne10);
    var src1_sorted = common.PoolAlloc(u8).initAlloc(ctx.pool(), n_src1_sorted * ts_src1_sorted);
    defer src1_sorted.deinit();
    const n_dst_sorted: usize = @intCast(ne2 * n_expert_used * ne0);
    var dst_sorted = common.PoolAlloc(u8).initAlloc(ctx.pool(), n_dst_sorted * ts_dst_sorted);
    defer dst_sorted.deinit();

    const ids_host = common.allocator.alloc(u8, c.ggml_nbytes(ids)) catch impl.abort("out of memory");
    defer common.allocator.free(ids_host);
    common.cudaCheck(cu.cudaMemcpyAsync(ids_host.ptr, ids.data, c.ggml_nbytes(ids), cu.cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream)", @src());
    common.cudaCheck(cu.cudaStreamSynchronize(stream), "cudaStreamSynchronize(stream)", @src());

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
                    impl.assert(n_to_sorted < n_rows, "ids_to_sorted_host.size() == size_t(ne_get_rows)");
                    ids_sorted_host[n_to_sorted] = @intCast(j12 * ne11 + @rem(iex, ne11));
                    n_to_sorted += 1;
                    tokens_per_expert[@intCast(j02)] += 1;
                    break;
                }
            }
        }
    }
    impl.assert(n_to_sorted == n_rows, "ids_to_sorted_host.size() == size_t(ne_get_rows)");

    common.cudaCheck(cu.cudaMemcpyAsync(@ptrCast(ids_buf_dev.ptr), ids_sorted_host.ptr, 2 * n_rows * @sizeOf(i32), cu.cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream)", @src());
    common.cudaCheck(cu.cudaStreamSynchronize(stream), "cudaStreamSynchronize(stream)", @src());

    const ids_dev: [*]i32 = @ptrCast(ids_buf_dev.ptr.?);
    const ids_to_sorted: *const i32 = &ids_dev[0];
    const ids_from_sorted: *const i32 = &ids_dev[n_rows];

    const rows_i32: usize = n_rows * @sizeOf(i32);
    getrows.getRowsCuda(src1.data, src1.type, ids_to_sorted, src1_sorted.get(), type_src1_sorted, ne10, nb11, nb12, nb13, ne_get_rows, 1, 1, @sizeOf(i32), rows_i32, rows_i32, @as(usize, @intCast(ne10)) * ts_src1_sorted, n_src1_sorted * ts_src1_sorted, n_src1_sorted * ts_src1_sorted, stream);
    common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());

    var src1_data_cur: [*]u8 = src1_sorted.get().?;
    var dst_data_cur: [*]u8 = dst_sorted.get().?;
    j02 = 0;
    while (j02 < ne02) : (j02 += 1) {
        const tokens = tokens_per_expert[@intCast(j02)];
        if (tokens == 0) continue;

        var src0_slice: Tensor = src0.*;
        src0_slice.ne[2] = 1;
        src0_slice.nb[3] = src0_slice.nb[2];
        src0_slice.op = c.GGML_OP_VIEW;
        src0_slice.view_src = dst.src[0]; // non-const pointer to src0
        src0_slice.data = @as([*]u8, @ptrCast(src0.data.?)) + @as(usize, @intCast(j02)) * nb02;

        var src1_slice: Tensor = std.mem.zeroes(Tensor);
        src1_slice.buffer = src1.buffer;
        src1_slice.type = type_src1_sorted;
        src1_slice.ne = .{ ne10, tokens, 1, 1 };
        src1_slice.nb[0] = ts_src1_sorted;
        src1_slice.nb[1] = @as(usize, @intCast(src1_slice.ne[0])) * src1_slice.nb[0];
        src1_slice.nb[2] = @as(usize, @intCast(src1_slice.ne[1])) * src1_slice.nb[1];
        src1_slice.nb[3] = @as(usize, @intCast(src1_slice.ne[2])) * src1_slice.nb[2];
        src1_slice.data = src1_data_cur;

        var dst_slice: Tensor = std.mem.zeroes(Tensor);
        dst_slice.buffer = dst.buffer;
        dst_slice.type = type_dst_sorted;
        dst_slice.ne = .{ ne0, tokens, 1, 1 };
        dst_slice.nb[0] = ts_dst_sorted;
        dst_slice.nb[1] = @as(usize, @intCast(dst_slice.ne[0])) * dst_slice.nb[0];
        dst_slice.nb[2] = @as(usize, @intCast(dst_slice.ne[1])) * dst_slice.nb[1];
        dst_slice.nb[3] = @as(usize, @intCast(dst_slice.ne[2])) * dst_slice.nb[2];
        dst_slice.data = dst_data_cur;

        mulMat(ctx, &src0_slice, &src1_slice, &dst_slice);
        common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur += dst_slice.nb[2];
    }

    getrows.getRowsCuda(dst_sorted.get(), type_dst_sorted, ids_from_sorted, dst.data, dst.type, ne0, @as(usize, @intCast(ne0)) * ts_dst_sorted, n_rows * @as(usize, @intCast(ne0)) * ts_dst_sorted, n_rows * @as(usize, @intCast(ne0)) * ts_dst_sorted, ne_get_rows, 1, 1, @sizeOf(i32), rows_i32, rows_i32, nb1, nb2, nb3, stream);
}
