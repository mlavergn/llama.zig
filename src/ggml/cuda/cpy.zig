//! `GGML_OP_CPY`, `GGML_OP_DUP` and `GGML_OP_CONT`: copy one tensor into
//! another, converting or (de)quantizing on the way. `cpy` is also what
//! `set.zig` and the rest of the backend call for a tensor copy.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/cpy.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/cpy.cuh` at v0.4.0 (`5266f24da`). The
//! kernels -- `cpy_scalar`, `cpy_scalar_transpose`, `cpy_scalar_contiguous`,
//! `cpy_f32_q`, `cpy_q_f32` -- stay CUDA. The MUSA `mudnnMemcpyAsync` arm
//! is not ported: this build is CUDA only.
//!
//! # Launches
//!
//! The scalar kernels go through `ggml_cuda_kernel_launch` (PDL), the
//! quantizing and dequantizing ones through `<<<>>>`. That split is
//! upstream's and is kept.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;
const half = kernels.half;
const bf16 = kernels.bf16;

/// Ports `CUDA_CPY_BLOCK_SIZE` (cpy.cuh:3 @5266f24da).
const block_size = 64;

/// Ports `CUDA_CPY_TILE_DIM_2D` (cpy.cu:10 @5266f24da).
const tile_dim_2d = 32;

/// Ports `CUDA_CPY_BLOCK_NM` (cpy.cu:11 @5266f24da).
const block_nm = 8;

/// Ports `CUDA_CPY_BLOCK_ROWS` (cpy.cu:12 @5266f24da).
const block_rows = 8;

const int_max = std.math.maxInt(c_int);
const max_dims: usize = c.GGML_MAX_DIMS;
const ushrt_max = std.math.maxInt(c_ushort);

/// A `dim3` member from an `int64_t`: C++ converts modulo 2^32.
fn dimOf(v: i64) u32 {
    return @truncate(@as(u64, @bitCast(v)));
}

/// The fifteen `int64_t` sizes and byte strides every copy kernel takes,
/// in the C's order.
const Args = struct {
    ne: i64,
    ne00: i64,
    ne01: i64,
    ne02: i64,
    nb00: i64,
    nb01: i64,
    nb02: i64,
    nb03: i64,
    ne10: i64,
    ne11: i64,
    ne12: i64,
    nb10: i64,
    nb11: i64,
    nb12: i64,
    nb13: i64,
};

/// Ports `ggml_cpy_scalar_contiguous_cuda` (cpy.cu:195 @5266f24da).
fn cpyScalarContiguousCuda(comptime S: type, comptime D: type, cx: ?*const anyopaque, cdst: ?*anyopaque, ne: i64, stream: cu.cudaStream_t) void {
    const num_blocks: i64 = @divTrunc(ne + block_size - 1, block_size);
    impl.assert(num_blocks <= int_max, "num_blocks <= INT_MAX");
    const key = comptime kernels.tmpl("cpy_scalar_contiguous", &.{ kernels.typeName(S), kernels.typeName(D) });
    kernels.launchPdl(kernels.get("cpy", key), .{
        .grid = .{ .x = dimOf(num_blocks) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{ cx, cdst, ne });
}

/// The fifteen size and stride arguments as a tuple, after `cx, cdst`.
fn kernelArgs(cx: ?*const anyopaque, cdst: ?*anyopaque, a: Args) struct { ?*const anyopaque, ?*anyopaque, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } {
    return .{ cx, cdst, a.ne, a.ne00, a.ne01, a.ne02, a.nb00, a.nb01, a.nb02, a.nb03, a.ne10, a.ne11, a.ne12, a.nb10, a.nb11, a.nb12, a.nb13 };
}

/// Ports `ggml_cpy_scalar_cuda` (cpy.cu:206 @5266f24da) with its
/// `launch_scalar_generic` lambda.
fn cpyScalarCuda(comptime S: type, comptime D: type, comptime transposed: bool, cx: ?*const anyopaque, cdst: ?*anyopaque, a: Args, stream: cu.cudaStream_t) void {
    const launchScalarGeneric = struct {
        fn f(cx_: ?*const anyopaque, cdst_: ?*anyopaque, a_: Args, stream_: cu.cudaStream_t) void {
            const num_blocks: i64 = @divTrunc(a_.ne + block_size - 1, block_size);
            impl.assert(num_blocks <= int_max, "num_blocks <= INT_MAX");
            const key = comptime kernels.tmpl("cpy_scalar", &.{"&" ++ kernels.tmpl("cpy_1_scalar", &.{ kernels.typeName(S), kernels.typeName(D) })});
            kernels.launchPdl(kernels.get("cpy", key), .{
                .grid = .{ .x = dimOf(num_blocks) },
                .block = .{ .x = block_size },
                .stream = stream_,
            }, kernelArgs(cx_, cdst_, a_));
        }
    }.f;

    if (transposed) {
        impl.assert(a.ne == a.ne00 * a.ne01 * a.ne02, "ne == ne00*ne01*ne02");
        var ne00n: i64 = undefined;
        var ne01n: i64 = undefined;
        var ne02n: i64 = undefined;
        if (a.nb00 <= a.nb02) { // most likely safe to handle nb00 = nb02 case here
            ne00n = a.ne00;
            ne01n = a.ne01;
            ne02n = a.ne02;
        } else {
            ne00n = a.ne00;
            ne01n = a.ne01 * a.ne02;
            ne02n = 1;
        }

        const grid_x: i64 = @divTrunc(ne01n + tile_dim_2d - 1, tile_dim_2d);
        const grid_y: i64 = @divTrunc(ne00n + tile_dim_2d - 1, tile_dim_2d);
        const grid_z: i64 = @divTrunc(@divTrunc(a.ne, ne01n * ne00n) + block_nm - 1, block_nm);
        impl.assert(grid_x <= int_max, "grid_x <= INT_MAX");
        if (grid_y > ushrt_max or grid_z > ushrt_max) {
            launchScalarGeneric(cx, cdst, a, stream);
        } else {
            var t = a;
            t.ne00 = ne00n;
            t.ne01 = ne01n;
            t.ne02 = ne02n;
            const key = comptime kernels.tmpl("cpy_scalar_transpose", &.{kernels.typeName(D)});
            kernels.launchPdl(kernels.get("cpy", key), .{
                .grid = .{ .x = dimOf(grid_x), .y = dimOf(grid_y), .z = dimOf(grid_z) },
                .block = .{ .x = tile_dim_2d, .y = block_rows, .z = 1 },
                .stream = stream,
            }, kernelArgs(cx, cdst, t));
        }
    } else {
        launchScalarGeneric(cx, cdst, a, stream);
    }
}

/// Ports the six quantizing launchers, which differ only in the block
/// function and its size: `ggml_cpy_f32_q8_0_cuda` (cpy.cu:250 @5266f24da),
/// `ggml_cpy_f32_q4_0_cuda` (cpy.cu:273 @5266f24da),
/// `ggml_cpy_f32_q4_1_cuda` (cpy.cu:299 @5266f24da),
/// `ggml_cpy_f32_q5_0_cuda` (cpy.cu:325 @5266f24da),
/// `ggml_cpy_f32_q5_1_cuda` (cpy.cu:351 @5266f24da) and
/// `ggml_cpy_f32_iq4_nl_cuda` (cpy.cu:377 @5266f24da).
fn cpyF32QCuda(comptime blck: []const u8, comptime qk: comptime_int, comptime qk_name: []const u8, cx: ?*const anyopaque, cdst: ?*anyopaque, a: Args, stream: cu.cudaStream_t) void {
    impl.assert(@rem(a.ne, qk) == 0, "ne % " ++ qk_name ++ " == 0");
    const num_blocks: i64 = @divTrunc(@divTrunc(a.ne, qk) + block_size - 1, block_size);
    impl.assert(num_blocks <= int_max, "num_blocks <= INT_MAX");
    const key = comptime kernels.tmpl("cpy_f32_q", &.{ "&" ++ blck, kernels.int(qk) });
    kernels.launch(kernels.get("cpy", key), .{
        .grid = .{ .x = dimOf(num_blocks) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, kernelArgs(cx, cdst, a));
}

/// Ports the five dequantizing launchers: `ggml_cpy_q8_0_f32_cuda`
/// (cpy.cu:262 @5266f24da), `ggml_cpy_q4_0_f32_cuda` (cpy.cu:285
/// @5266f24da), `ggml_cpy_q4_1_f32_cuda` (cpy.cu:311 @5266f24da),
/// `ggml_cpy_q5_0_f32_cuda` (cpy.cu:337 @5266f24da) and
/// `ggml_cpy_q5_1_f32_cuda` (cpy.cu:363 @5266f24da). None asserts
/// divisibility, as in the C.
fn cpyQF32Cuda(comptime blck: []const u8, comptime qk: comptime_int, cx: ?*const anyopaque, cdst: ?*anyopaque, a: Args, stream: cu.cudaStream_t) void {
    const num_blocks: i64 = @divTrunc(@divTrunc(a.ne, qk) + block_size - 1, block_size);
    impl.assert(num_blocks <= int_max, "num_blocks <= INT_MAX");
    const key = comptime kernels.tmpl("cpy_q_f32", &.{ "&" ++ blck, kernels.int(qk) });
    kernels.launch(kernels.get("cpy", key), .{
        .grid = .{ .x = dimOf(num_blocks) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, kernelArgs(cx, cdst, a));
}

/// `cpy_blck_q_f32<&dequantize_qX, qk>`, the block function the generic
/// dequantizers are instantiated with.
fn blckQF32(comptime dq: []const u8) []const u8 {
    return kernels.tmpl("cpy_blck_q_f32", &.{ "&" ++ dq, kernels.int(32) });
}

/// A 2D strided copy's shape, from `cpyAsMemcpy2d`.
const Memcpy2d = struct {
    width: usize = 0,
    height: usize = 0,
    spitch: usize = 0,
    dpitch: usize = 0,
};

/// Ports `ggml_cuda_cpy_as_memcpy_2d` (cpy.cu:391 @5266f24da): whether a
/// same-type copy reduces to `height` rows of `width` contiguous bytes, so
/// it can use `cudaMemcpy2DAsync` instead of the scalar kernel.
fn cpyAsMemcpy2d(src0: *const Tensor, src1: *const Tensor, m: *Memcpy2d) bool {
    // require matching shape: a reshaped copy maps elements by flat order,
    // which the prefix walk below does not handle
    if (src0.type != src1.type or !c.ggml_are_same_shape(src0, src1)) {
        return false;
    }

    // grow the contiguous prefix block shared by both tensors
    var block_nb: usize = c.ggml_element_size(src0);
    var d: usize = 0;
    while (d < max_dims) : (d += 1) {
        if (src0.nb[d] != block_nb or src1.nb[d] != block_nb) {
            break;
        }
        block_nb *%= @bitCast(src0.ne[d]);
    }

    // d == 0: nothing contiguous; d == GGML_MAX_DIMS: fully contiguous (handled by memcpy)
    if (d == 0 or d == max_dims) {
        return false;
    }

    // dim d carries the rows; everything above it must be a single element
    for (d + 1..max_dims) |i| {
        if (src0.ne[i] != 1) {
            return false;
        }
    }

    m.width = block_nb;
    m.height = @bitCast(src0.ne[d]);
    m.spitch = src0.nb[d];
    m.dpitch = src1.nb[d];

    return m.spitch >= m.width and m.dpitch >= m.width;
}

/// Ports `ggml_cuda_cpy` (cpy.cu:429 @5266f24da).
///
/// Parameters:
/// - `ctx`: the backend context whose stream is used.
/// - `src0`: the tensor copied from.
/// - `src1`: the tensor copied into; same element count as `src0`.
pub fn cpy(ctx: *common.Context, src0: *const Tensor, src1: *Tensor) void {
    const ne = c.ggml_nelements(src0);
    impl.assert(ne == c.ggml_nelements(src1), "ne == ggml_nelements(src1)");

    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];

    const nb00: i64 = @bitCast(@as(u64, src0.nb[0]));
    const nb01: i64 = @bitCast(@as(u64, src0.nb[1]));
    const nb02: i64 = @bitCast(@as(u64, src0.nb[2]));
    const nb03: i64 = @bitCast(@as(u64, src0.nb[3]));

    const ne10 = src1.ne[0];
    const ne11 = src1.ne[1];
    const ne12 = src1.ne[2];

    const nb10: i64 = @bitCast(@as(u64, src1.nb[0]));
    const nb11: i64 = @bitCast(@as(u64, src1.nb[1]));
    const nb12: i64 = @bitCast(@as(u64, src1.nb[2]));
    const nb13: i64 = @bitCast(@as(u64, src1.nb[3]));

    const main_stream = ctx.stream();

    const src0_ddc: ?*const anyopaque = src0.data;
    const src1_ddc: ?*anyopaque = src1.data;

    const contiguous_srcs = c.ggml_is_contiguous(src0) and c.ggml_is_contiguous(src1);
    const esize: i64 = @intCast(c.ggml_element_size(src0));
    const can_be_transposed = nb01 == esize and src0.ne[3] == 1 and nb02 == ne00 * ne01 * esize;

    var mc: Memcpy2d = .{};

    const a: Args = .{
        .ne = ne,
        .ne00 = ne00,
        .ne01 = ne01,
        .ne02 = ne02,
        .nb00 = nb00,
        .nb01 = nb01,
        .nb02 = nb02,
        .nb03 = nb03,
        .ne10 = ne10,
        .ne11 = ne11,
        .ne12 = ne12,
        .nb10 = nb10,
        .nb11 = nb11,
        .nb12 = nb12,
        .nb13 = nb13,
    };

    const t0 = src0.type;
    const t1 = src1.type;
    const F32 = c.GGML_TYPE_F32;
    const F16 = c.GGML_TYPE_F16;
    const BF16 = c.GGML_TYPE_BF16;
    const I32 = c.GGML_TYPE_I32;

    if (t0 == t1 and contiguous_srcs) {
        impl.assert(c.ggml_nbytes(src0) == c.ggml_nbytes(src1), "ggml_nbytes(src0) == ggml_nbytes(src1)");
        common.cudaCheck(cu.cudaMemcpyAsync(src1_ddc, src0_ddc, c.ggml_nbytes(src0), cu.cudaMemcpyDeviceToDevice, main_stream), "cudaMemcpyAsync(src1_ddc, src0_ddc, ggml_nbytes(src0), cudaMemcpyDeviceToDevice, main_stream)", @src());
    } else if (cpyAsMemcpy2d(src0, src1, &mc)) {
        common.cudaCheck(cu.cudaMemcpy2DAsync(src1_ddc, mc.dpitch, src0_ddc, mc.spitch, mc.width, mc.height, cu.cudaMemcpyDeviceToDevice, main_stream), "cudaMemcpy2DAsync(src1_ddc, mc_dpitch, src0_ddc, mc_spitch, mc_width, mc_height, cudaMemcpyDeviceToDevice, main_stream)", @src());
    } else if (t0 == F32 and t1 == F32) {
        if (can_be_transposed) {
            cpyScalarCuda(f32, f32, true, src0_ddc, src1_ddc, a, main_stream);
        } else {
            cpyScalarCuda(f32, f32, false, src0_ddc, src1_ddc, a, main_stream);
        }
    } else if (t0 == F32 and t1 == BF16) {
        if (contiguous_srcs) {
            cpyScalarContiguousCuda(f32, bf16, src0_ddc, src1_ddc, ne, main_stream);
        } else {
            cpyScalarCuda(f32, bf16, false, src0_ddc, src1_ddc, a, main_stream);
        }
    } else if (t0 == F32 and t1 == F16) {
        if (contiguous_srcs) {
            cpyScalarContiguousCuda(f32, half, src0_ddc, src1_ddc, ne, main_stream);
        } else {
            cpyScalarCuda(f32, half, false, src0_ddc, src1_ddc, a, main_stream);
        }
    } else if (t0 == F32 and t1 == c.GGML_TYPE_Q8_0) {
        cpyF32QCuda("cpy_blck_f32_q8_0", 32, "QK8_0", src0_ddc, src1_ddc, a, main_stream);
    } else if (t0 == c.GGML_TYPE_Q8_0 and t1 == F32) {
        cpyQF32Cuda("cpy_blck_q8_0_f32", 32, src0_ddc, src1_ddc, a, main_stream);
    } else if (t0 == F32 and t1 == c.GGML_TYPE_Q4_0) {
        cpyF32QCuda("cpy_blck_f32_q4_0", 32, "QK4_0", src0_ddc, src1_ddc, a, main_stream);
    } else if (t0 == c.GGML_TYPE_Q4_0 and t1 == F32) {
        cpyQF32Cuda(comptime blckQF32("dequantize_q4_0"), 32, src0_ddc, src1_ddc, a, main_stream);
    } else if (t0 == F32 and t1 == c.GGML_TYPE_Q4_1) {
        cpyF32QCuda("cpy_blck_f32_q4_1", 32, "QK4_1", src0_ddc, src1_ddc, a, main_stream);
    } else if (t0 == c.GGML_TYPE_Q4_1 and t1 == F32) {
        cpyQF32Cuda(comptime blckQF32("dequantize_q4_1"), 32, src0_ddc, src1_ddc, a, main_stream);
    } else if (t0 == F32 and t1 == c.GGML_TYPE_Q5_0) {
        cpyF32QCuda("cpy_blck_f32_q5_0", 32, "QK5_0", src0_ddc, src1_ddc, a, main_stream);
    } else if (t0 == c.GGML_TYPE_Q5_0 and t1 == F32) {
        cpyQF32Cuda(comptime blckQF32("dequantize_q5_0"), 32, src0_ddc, src1_ddc, a, main_stream);
    } else if (t0 == F32 and t1 == c.GGML_TYPE_IQ4_NL) {
        cpyF32QCuda("cpy_blck_f32_iq4_nl", 32, "QK4_NL", src0_ddc, src1_ddc, a, main_stream);
    } else if (t0 == F32 and t1 == c.GGML_TYPE_Q5_1) {
        cpyF32QCuda("cpy_blck_f32_q5_1", 32, "QK5_1", src0_ddc, src1_ddc, a, main_stream);
    } else if (t0 == c.GGML_TYPE_Q5_1 and t1 == F32) {
        cpyQF32Cuda(comptime blckQF32("dequantize_q5_1"), 32, src0_ddc, src1_ddc, a, main_stream);
    } else if (t0 == F16 and t1 == F16) {
        if (can_be_transposed) {
            cpyScalarCuda(half, half, true, src0_ddc, src1_ddc, a, main_stream);
        } else {
            cpyScalarCuda(half, half, false, src0_ddc, src1_ddc, a, main_stream);
        }
    } else if (t0 == F16 and t1 == BF16) {
        if (contiguous_srcs) {
            cpyScalarContiguousCuda(half, bf16, src0_ddc, src1_ddc, ne, main_stream);
        } else {
            cpyScalarCuda(half, bf16, false, src0_ddc, src1_ddc, a, main_stream);
        }
    } else if (t0 == F16 and t1 == F32) {
        if (contiguous_srcs) {
            cpyScalarContiguousCuda(half, f32, src0_ddc, src1_ddc, ne, main_stream);
        } else {
            cpyScalarCuda(half, f32, false, src0_ddc, src1_ddc, a, main_stream);
        }
    } else if (t0 == BF16 and t1 == BF16) {
        if (can_be_transposed) {
            cpyScalarCuda(bf16, bf16, true, src0_ddc, src1_ddc, a, main_stream);
        } else {
            cpyScalarCuda(bf16, bf16, false, src0_ddc, src1_ddc, a, main_stream);
        }
    } else if (t0 == BF16 and t1 == F16) {
        if (contiguous_srcs) {
            cpyScalarContiguousCuda(bf16, half, src0_ddc, src1_ddc, ne, main_stream);
        } else {
            cpyScalarCuda(bf16, half, false, src0_ddc, src1_ddc, a, main_stream);
        }
    } else if (t0 == BF16 and t1 == F32) {
        if (contiguous_srcs) {
            cpyScalarContiguousCuda(bf16, f32, src0_ddc, src1_ddc, ne, main_stream);
        } else {
            cpyScalarCuda(bf16, f32, false, src0_ddc, src1_ddc, a, main_stream);
        }
    } else if (t0 == I32 and t1 == I32) {
        if (can_be_transposed) {
            cpyScalarCuda(i32, i32, true, src0_ddc, src1_ddc, a, main_stream);
        } else {
            cpyScalarCuda(i32, i32, false, src0_ddc, src1_ddc, a, main_stream);
        }
    } else if (t0 == F32 and t1 == I32) {
        if (contiguous_srcs) {
            cpyScalarContiguousCuda(f32, i32, src0_ddc, src1_ddc, ne, main_stream);
        } else {
            cpyScalarCuda(f32, i32, false, src0_ddc, src1_ddc, a, main_stream);
        }
    } else if (t0 == I32 and t1 == F32) {
        if (contiguous_srcs) {
            cpyScalarContiguousCuda(i32, f32, src0_ddc, src1_ddc, ne, main_stream);
        } else {
            cpyScalarCuda(i32, f32, false, src0_ddc, src1_ddc, a, main_stream);
        }
    } else {
        c.ggml_abort(@src().file.ptr, @intCast(@src().line), "%s: unsupported type combination (%s to %s)\n", "ggml_cuda_cpy", c.ggml_type_name(t0), c.ggml_type_name(t1));
    }
}

/// Ports `ggml_cuda_dup` (cpy.cu:614 @5266f24da), which serves `DUP` and
/// `CONT`.
pub fn dup(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    cpy(ctx, src0, dst);
}
