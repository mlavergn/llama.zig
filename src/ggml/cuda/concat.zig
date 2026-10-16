//! `GGML_OP_CONCAT`: join two tensors along one dimension.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/concat.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/concat.cuh` at v0.3.0 (`c1d0e7a00`). The
//! kernels, `concat_cont` and `concat_non_cont`, stay CUDA.
//!
//! # Launches
//!
//! Only `concat_cont<T, 0>` goes through `ggml_cuda_kernel_launch` (PDL);
//! the `dim == 1` and `dim == 2` contiguous kernels and the non-contiguous
//! one use `<<<>>>`. That asymmetry is upstream's and is kept.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_CONCAT_BLOCK_SIZE` (concat.cuh:3 @c1d0e7a00).
const block_size = 256;

/// A `dim3` member from an `int64_t`: C++ converts modulo 2^32.
fn dimOf(v: i64) u32 {
    return @truncate(@as(u64, @bitCast(v)));
}

/// Ports `concat_cont_cuda` (concat.cu:56 @c1d0e7a00).
fn concatContCuda(comptime T: type, x: ?*const anyopaque, y: ?*const anyopaque, dst: ?*anyopaque, ne00: i64, ne01: i64, ne02: i64, ne0: i64, ne1: i64, ne2: i64, dim: c_int, stream: cu.cudaStream_t) void {
    const n: i64 = ne0 * ne1 * ne2;
    const num_blocks: c_int = @truncate(@divTrunc(n + block_size - 1, block_size));
    const p: kernels.LaunchParams = .{
        .grid = .{ .x = @bitCast(num_blocks) },
        .block = .{ .x = block_size },
        .stream = stream,
    };
    const args = .{ x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2 };

    const t = comptime kernels.typeName(T);
    if (dim == 0) {
        kernels.launchPdl(kernels.get("concat", comptime kernels.tmpl("concat_cont", &.{ t, kernels.int(0) })), p, args);
        return;
    }
    if (dim == 1) {
        kernels.launch(kernels.get("concat", comptime kernels.tmpl("concat_cont", &.{ t, kernels.int(1) })), p, args);
        return;
    }
    kernels.launch(kernels.get("concat", comptime kernels.tmpl("concat_cont", &.{ t, kernels.int(2) })), p, args);
}

/// `ptr + i3 * (nb3 / sizeof(T))`, as `const T *` arithmetic. `i3` is a
/// Zig type name, so the index is `j3`.
fn plane(comptime T: type, p: ?*const anyopaque, j3: i64, nb3: usize) usize {
    const elems: usize = @as(u64, @bitCast(j3)) *% (nb3 / @sizeOf(T));
    return @intFromPtr(p) +% elems *% @sizeOf(T);
}

/// Ports `concat_cuda` (concat.cu:143 @c1d0e7a00) with its
/// `launch_kernel` lambda.
fn concatCuda(comptime T: type, src0: *const Tensor, src1: *const Tensor, dst: *Tensor, dim: c_int, stream: cu.cudaStream_t) void {
    if (dim != 3 and c.ggml_is_contiguous_to_3(src0) and c.ggml_is_contiguous_to_3(src1)) {
        var j3: i64 = 0;
        while (j3 < dst.ne[3]) : (j3 += 1) {
            concatContCuda(
                T,
                @ptrFromInt(plane(T, src0.data, j3, src0.nb[3])),
                @ptrFromInt(plane(T, src1.data, j3, src1.nb[3])),
                @ptrFromInt(plane(T, dst.data, j3, dst.nb[3])),
                @bitCast(@as(u64, c.ggml_row_size(src0.type, src0.ne[0]) / @sizeOf(T))),
                src0.ne[1],
                src0.ne[2],
                @bitCast(@as(u64, c.ggml_row_size(dst.type, dst.ne[0]) / @sizeOf(T))),
                dst.ne[1],
                dst.ne[2],
                dim,
                stream,
            );
        }
    } else if (dim == 3 and c.ggml_is_contiguous(src0) and c.ggml_is_contiguous(src1)) {
        const size0 = c.ggml_nbytes(src0);
        const size1 = c.ggml_nbytes(src1);

        common.cudaCheck(cu.cudaMemcpyAsync(dst.data, src0.data, size0, cu.cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync((char *) dst->data,         src0->data, size0, cudaMemcpyDeviceToDevice, stream)", @src());
        common.cudaCheck(cu.cudaMemcpyAsync(@ptrFromInt(@intFromPtr(dst.data) + size0), src1.data, size1, cu.cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync((char *) dst->data + size0, src1->data, size1, cudaMemcpyDeviceToDevice, stream)", @src());
    } else {
        impl.assert(!c.ggml_is_quantized(src0.type), "!ggml_is_quantized(src0->type)");

        const p: kernels.LaunchParams = .{
            .grid = .{ .x = dimOf(dst.ne[1]), .y = dimOf(dst.ne[2]), .z = dimOf(dst.ne[3]) },
            .block = .{ .x = block_size },
            .stream = stream,
        };
        const args = .{
            @as(?*const anyopaque, src0.data), @as(?*const anyopaque, src1.data), @as(?*anyopaque, dst.data),
            src0.ne[0],                        src0.ne[1],                        src0.ne[2],
            src0.ne[3],                        @as(u64, src0.nb[0]),              @as(u64, src0.nb[1]),
            @as(u64, src0.nb[2]),              @as(u64, src0.nb[3]),              src1.ne[0],
            src1.ne[1],                        src1.ne[2],                        src1.ne[3],
            @as(u64, src1.nb[0]),              @as(u64, src1.nb[1]),              @as(u64, src1.nb[2]),
            @as(u64, src1.nb[3]),              dst.ne[0],                         dst.ne[1],
            dst.ne[2],                         dst.ne[3],                         @as(u64, dst.nb[0]),
            @as(u64, dst.nb[1]),               @as(u64, dst.nb[2]),               @as(u64, dst.nb[3]),
        };
        const t = comptime kernels.typeName(T);
        switch (dim) {
            inline 0...3 => |d| kernels.launch(kernels.get("concat", comptime kernels.tmpl("concat_non_cont", &.{ t, kernels.int(d) })), p, args),
            else => c.ggml_abort(@src().file.ptr, @intCast(@src().line), "Invalid dim: %d", dim),
        }
    }
}

/// Ports `ggml_cuda_op_concat` (concat.cu:197 @c1d0e7a00).
pub fn opConcat(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const stream = ctx.stream();

    const dim: i32 = impl.getOpParamsI32(dst, 0);

    impl.assert(src0.type == src1.type, "src0->type == src1->type");
    impl.assert(dst.type == src0.type, "dst->type  == src0->type");

    if (c.ggml_is_quantized(src0.type)) {
        if (dim == 3) {
            impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
            impl.assert(c.ggml_is_contiguous(src1), "ggml_is_contiguous(src1)");
        } else {
            impl.assert(c.ggml_is_contiguous_to_3(src0), "ggml_is_contiguous_to_3(src0)");
            impl.assert(c.ggml_is_contiguous_to_3(src1), "ggml_is_contiguous_to_3(src1)");
        }
        impl.assert(@rem(src0.ne[0], c.ggml_blck_size(src0.type)) == 0, "src0->ne[0] % ggml_blck_size(src0->type) == 0");
        impl.assert(@rem(src1.ne[0], c.ggml_blck_size(src1.type)) == 0, "src1->ne[0] % ggml_blck_size(src1->type) == 0");

        // if first 3 dimensions are contiguous and ne[0] is multiple of the
        // block size we can concat both tensors as byte tensors
        concatCuda(u8, src0, src1, dst, dim, stream);
    } else {
        impl.assert(c.ggml_blck_size(src0.type) == 1, "ggml_blck_size(src0->type) == 1");

        switch (c.ggml_type_size(src0.type)) {
            1 => concatCuda(u8, src0, src1, dst, dim, stream),
            2 => concatCuda(u16, src0, src1, dst, dim, stream),
            4 => concatCuda(u32, src0, src1, dst, dim, stream),
            8 => concatCuda(u64, src0, src1, dst, dim, stream),
            else => c.ggml_abort(@src().file.ptr, @intCast(@src().line), "Unsupported type size: %zu", c.ggml_type_size(src0.type)),
        }
    }
}
