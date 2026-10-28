//! `GGML_OP_RWKV_WKV6` and `GGML_OP_RWKV_WKV7`: the RWKV v6 and v7 time-mix
//! recurrences.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/wkv.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/wkv.cuh` at v0.4.0 (`5266f24da`). The
//! kernels -- `rwkv_wkv_f32`, `rwkv_wkv7_f32` and
//! `rwkv_wkv7_f32_t1_warp_row` -- stay CUDA. Every launch is `<<<>>>`.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_WKV_BLOCK_SIZE` (wkv.cuh:3 @5266f24da).
const block_size = 64;

/// An `int64_t` into a `dim3` member, which is `unsigned int`.
fn dim(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// An `int64_t` into an `int` parameter.
fn int(x: i64) c_int {
    return @truncate(x);
}

fn srcData(dst: *const Tensor, i: usize) ?*const anyopaque {
    return impl.one(Tensor, dst.src[i]).data;
}

/// Ports `ggml_cuda_op_rwkv_wkv6` (wkv.cu:195 @5266f24da).
pub fn opRwkvWkv6(ctx: *common.Context, dst: *Tensor) void {
    const k_d = srcData(dst, 0);
    const v_d = srcData(dst, 1);
    const r_d = srcData(dst, 2);
    const tf_d = srcData(dst, 3);
    const td_d = srcData(dst, 4);
    const s_d = srcData(dst, 5);

    const B = dst.src[5].*.ne[1];
    const T = dst.src[0].*.ne[2];
    const C = dst.ne[0];
    const H = dst.src[0].*.ne[1];

    const dst_d: ?*anyopaque = dst.data;

    const stream = ctx.stream();

    impl.assert(dst.src[5].*.type == c.GGML_TYPE_F32, "dst->src[5]->type == GGML_TYPE_F32");
    impl.assert(@rem(C, H) == 0, "C % H == 0");
    impl.assert(@divTrunc(C, H) == block_size or @divTrunc(C, H) == block_size * 2, "C / H == CUDA_WKV_BLOCK_SIZE || C / H == CUDA_WKV_BLOCK_SIZE * 2");

    const p: kernels.LaunchParams = .{ .grid = .{ .x = dim(B * H) }, .block = .{ .x = dim(@divTrunc(C, H)) }, .shmem = 0, .stream = stream };
    const args = .{ int(B), int(T), int(C), int(H), k_d, v_d, r_d, tf_d, td_d, s_d, dst_d };
    if (@divTrunc(C, H) == block_size) {
        kernels.launch(kernels.get("wkv", kernels.tmpl("rwkv_wkv_f32", &.{kernels.int(block_size)})), p, args);
    } else {
        kernels.launch(kernels.get("wkv", kernels.tmpl("rwkv_wkv_f32", &.{kernels.int(block_size * 2)})), p, args);
    }
}

/// Ports `ggml_cuda_op_rwkv_wkv7` (wkv.cu:223 @5266f24da).
pub fn opRwkvWkv7(ctx: *common.Context, dst: *Tensor) void {
    const r_d = srcData(dst, 0);
    const w_d = srcData(dst, 1);
    const k_d = srcData(dst, 2);
    const v_d = srcData(dst, 3);
    const a_d = srcData(dst, 4);
    const b_d = srcData(dst, 5);
    const s_d = srcData(dst, 6);

    const B = dst.src[6].*.ne[1];
    const T = dst.src[0].*.ne[2];
    const C = dst.ne[0];
    const H = dst.src[0].*.ne[1];

    const dst_d: ?*anyopaque = dst.data;

    const stream = ctx.stream();

    impl.assert(dst.src[6].*.type == c.GGML_TYPE_F32, "dst->src[6]->type == GGML_TYPE_F32");
    impl.assert(@rem(C, H) == 0, "C % H == 0");
    impl.assert(@divTrunc(C, H) == block_size or @divTrunc(C, H) == block_size * 2, "C / H == CUDA_WKV_BLOCK_SIZE || C / H == CUDA_WKV_BLOCK_SIZE * 2");

    if (@divTrunc(T, B) == 1 and @divTrunc(C, H) == block_size) {
        const rows_per_block = 4;
        kernels.launch(kernels.get("wkv", kernels.tmpl("rwkv_wkv7_f32_t1_warp_row", &.{kernels.int(rows_per_block)})), .{
            .grid = .{ .x = dim(B * H), .y = block_size / rows_per_block },
            .block = .{ .x = common.warp_size, .y = rows_per_block },
            .shmem = 0,
            .stream = stream,
        }, .{ int(T), int(C), int(H), r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d });
    } else {
        const p: kernels.LaunchParams = .{ .grid = .{ .x = dim(B * H) }, .block = .{ .x = dim(@divTrunc(C, H)) }, .shmem = 0, .stream = stream };
        const args = .{ int(B), int(T), int(C), int(H), r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d };
        if (@divTrunc(C, H) == block_size) {
            kernels.launch(kernels.get("wkv", kernels.tmpl("rwkv_wkv7_f32", &.{kernels.int(block_size)})), p, args);
        } else {
            kernels.launch(kernels.get("wkv", kernels.tmpl("rwkv_wkv7_f32", &.{kernels.int(block_size * 2)})), p, args);
        }
    }
}
