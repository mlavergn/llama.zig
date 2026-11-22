//! MMVQ: quantized matrix-vector products, `src1` quantized to `q8_1` first,
//! for batches of at most `MMVQ_MAX_BATCH_SIZE` columns.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/mmvq.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/mmvq.cuh` at v1.0.0 (`b1b770d90`), the host
//! half; the `VDR_*_MMVQ` constants are `vecdotq.cuh`'s. The kernels,
//! `mul_mat_vec_q` and `mul_mat_vec_q_moe`, stay CUDA.
//!
//! `ggml_cuda_op_mul_mat_vec_q`, the row-split entry point, is not ported:
//! nothing in this version of upstream calls it.

const ggml = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `MMVQ_MAX_BATCH_SIZE` (mmvq.cuh:3 @b1b770d90).
pub const max_batch_size = 8;

/// Ports `get_vdr_mmvq` (mmvq.cu:65 @b1b770d90), with the `VDR_*_MMVQ`
/// constants of `vecdotq.cuh`.
fn getVdrMmvq(comptime ty: ggml.impl.ggml_c.enum_ggml_type) comptime_int {
    return switch (ty) {
        ggml.impl.ggml_c.GGML_TYPE_Q1_0 => 1,
        ggml.impl.ggml_c.GGML_TYPE_Q2_0 => 1,
        ggml.impl.ggml_c.GGML_TYPE_Q4_0 => 2,
        ggml.impl.ggml_c.GGML_TYPE_Q4_1 => 2,
        ggml.impl.ggml_c.GGML_TYPE_Q5_0 => 2,
        ggml.impl.ggml_c.GGML_TYPE_Q5_1 => 2,
        ggml.impl.ggml_c.GGML_TYPE_Q8_0 => 2,
        ggml.impl.ggml_c.GGML_TYPE_MXFP4 => 2,
        ggml.impl.ggml_c.GGML_TYPE_NVFP4 => 4,
        ggml.impl.ggml_c.GGML_TYPE_Q2_K => 2,
        ggml.impl.ggml_c.GGML_TYPE_Q3_K => 2,
        ggml.impl.ggml_c.GGML_TYPE_Q4_K => 2,
        ggml.impl.ggml_c.GGML_TYPE_Q5_K => 2,
        ggml.impl.ggml_c.GGML_TYPE_Q6_K => 2,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS => 2,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XS => 2,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_S => 2,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS => 2,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_S => 2,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_NL => 2,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_XS => 4,
        else => 1,
    };
}

/// Ports `mmvq_parameter_table_id` (mmvq.cu:92 @b1b770d90).
const TableId = enum(c_int) { generic = 0, turing, gcn, rdna2, rdna3_0, rdna4, gb10 };

/// Ports the host `get_device_table_id` (mmvq.cu:120 @b1b770d90), NVIDIA
/// arms (RDNA4_M is AMD). Volta takes the Turing table too.
fn getDeviceTableId(x: c_int) TableId {
    const h = ggml.cuda.common.highestCompiledArch(x);
    if (ggml.cuda.common.cc.isNvidia(x) and h >= ggml.cuda.common.cc.volta and h < ggml.cuda.common.cc.ampere) return .turing;
    if (ggml.cuda.common.cc.isNvidia(x) and h == ggml.cuda.common.cc.dgx_spark) return .gb10;
    return .generic;
}

/// Ports `get_mmvq_mmid_max_batch_pascal_older` (mmvq.cu:146 @b1b770d90).
fn mmidMaxBatchPascalOlder(ty: ggml.impl.ggml_c.enum_ggml_type) c_int {
    return switch (ty) {
        ggml.impl.ggml_c.GGML_TYPE_IQ1_S => 6,
        ggml.impl.ggml_c.GGML_TYPE_IQ1_M => 6,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_S => 4,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XS => 5,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS => 5,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_S => 4,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS => 4,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_NL => 6,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_XS => 5,
        ggml.impl.ggml_c.GGML_TYPE_MXFP4 => 4,
        ggml.impl.ggml_c.GGML_TYPE_NVFP4 => 4,
        ggml.impl.ggml_c.GGML_TYPE_Q2_K => 4,
        ggml.impl.ggml_c.GGML_TYPE_Q3_K => 4,
        ggml.impl.ggml_c.GGML_TYPE_Q4_0 => 6,
        ggml.impl.ggml_c.GGML_TYPE_Q4_1 => 6,
        ggml.impl.ggml_c.GGML_TYPE_Q4_K => 5,
        ggml.impl.ggml_c.GGML_TYPE_Q5_0 => 6,
        ggml.impl.ggml_c.GGML_TYPE_Q5_1 => 6,
        ggml.impl.ggml_c.GGML_TYPE_Q5_K => 5,
        ggml.impl.ggml_c.GGML_TYPE_Q6_K => 4,
        ggml.impl.ggml_c.GGML_TYPE_Q8_0 => 4,
        else => max_batch_size,
    };
}

/// Ports `get_mmvq_mmid_max_batch_turing_plus` (mmvq.cu:173 @b1b770d90).
fn mmidMaxBatchTuringPlus(ty: ggml.impl.ggml_c.enum_ggml_type) c_int {
    return switch (ty) {
        ggml.impl.ggml_c.GGML_TYPE_IQ2_S => 7,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_S => 6,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS => 7,
        ggml.impl.ggml_c.GGML_TYPE_MXFP4 => 7,
        ggml.impl.ggml_c.GGML_TYPE_NVFP4 => 8,
        ggml.impl.ggml_c.GGML_TYPE_Q2_K => 7,
        ggml.impl.ggml_c.GGML_TYPE_Q3_K => 5,
        else => max_batch_size,
    };
}

/// Ports `get_mmvq_mmid_max_batch` (mmvq.cu:281 @b1b770d90), NVIDIA arms.
/// The P100 (`cc == GGML_CUDA_CC_PASCAL`) is in the always-MMVQ
/// set, as in the device side's `get_mmvq_mmid_max_batch_for_device`.
/// It decides on the highest compiled architecture,
/// as the device side does.
pub fn getMmidMaxBatch(ty: ggml.impl.ggml_c.enum_ggml_type, x_in: c_int) c_int {
    // NVIDIA: P100, Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (ggml.cuda.common.cc.isNvidia(x_in)) {
        const x = ggml.cuda.common.highestCompiledArch(x_in);
        if (x == ggml.cuda.common.cc.pascal or x == ggml.cuda.common.cc.volta or x >= ggml.cuda.common.cc.ada_lovelace) return max_batch_size;
        if (x >= ggml.cuda.common.cc.turing) return mmidMaxBatchTuringPlus(ty);
        return mmidMaxBatchPascalOlder(ty);
    }
    return max_batch_size;
}

/// Ports `ggml_cuda_should_use_mmvq` (mmvq.cu:315 @b1b770d90), NVIDIA arms.
pub fn shouldUseMmvq(ty: ggml.impl.ggml_c.enum_ggml_type, x: c_int, ne11: i64) bool {
    if (!ggml.impl.ggml_c.ggml_is_quantized(ty)) return false;
    // k-quants cost more to decode and mvq redoes that per column, so MMQ
    // wins sooner. Only list quant-types MMQ supports, others would fall back
    // to cuBLAS.
    if (ggml.cuda.common.cc.isNvidia(x) and x == ggml.cuda.common.cc.ada_lovelace) {
        return switch (ty) { // tuned on RTX 4090
            ggml.impl.ggml_c.GGML_TYPE_Q2_K => ne11 <= 4,
            ggml.impl.ggml_c.GGML_TYPE_Q3_K => ne11 <= 6,
            else => ne11 <= max_batch_size,
        };
    }
    if (ggml.cuda.common.cc.isNvidia(x) and x == ggml.cuda.common.cc.blackwell) {
        return switch (ty) { // tuned on RTX 5090
            ggml.impl.ggml_c.GGML_TYPE_Q2_K, ggml.impl.ggml_c.GGML_TYPE_Q3_K, ggml.impl.ggml_c.GGML_TYPE_Q4_K => ne11 <= 5,
            ggml.impl.ggml_c.GGML_TYPE_Q5_K => ne11 <= 6,
            ggml.impl.ggml_c.GGML_TYPE_Q6_K => ne11 <= 7,
            else => ne11 <= max_batch_size,
        };
    }
    if (ggml.cuda.common.cc.isNvidia(x) and x == ggml.cuda.common.cc.dgx_spark) {
        return switch (ty) { // tuned on DGX Spark GB10
            ggml.impl.ggml_c.GGML_TYPE_Q2_K => ne11 <= 6,
            else => ne11 <= max_batch_size,
        };
    }
    if (ggml.cuda.common.cc.isNvidia(x) and x == ggml.cuda.common.cc.orin) {
        return switch (ty) { // tuned for Jetson Orin
            ggml.impl.ggml_c.GGML_TYPE_Q2_K, ggml.impl.ggml_c.GGML_TYPE_Q3_K, ggml.impl.ggml_c.GGML_TYPE_Q4_K, ggml.impl.ggml_c.GGML_TYPE_Q5_K, ggml.impl.ggml_c.GGML_TYPE_Q6_K => ne11 <= 1,
            else => ne11 <= max_batch_size,
        };
    }
    if (ggml.cuda.common.cc.isNvidia(x) and x == ggml.cuda.common.cc.volta) {
        return switch (ty) {
            ggml.impl.ggml_c.GGML_TYPE_Q2_K => ne11 <= 4,
            ggml.impl.ggml_c.GGML_TYPE_Q3_K => ne11 <= 6,
            ggml.impl.ggml_c.GGML_TYPE_Q4_K => ne11 <= 5,
            ggml.impl.ggml_c.GGML_TYPE_Q5_K => ne11 <= 6,
            ggml.impl.ggml_c.GGML_TYPE_Q6_K => ne11 <= 7,
            else => ne11 <= max_batch_size,
        };
    }
    return ne11 <= max_batch_size;
}

/// Ports `calc_nwarps` (mmvq.cu:449 @b1b770d90), the NVIDIA tables; the
/// AMD tables are unreachable from `getDeviceTableId` here.
fn calcNwarps(ty: ggml.impl.ggml_c.enum_ggml_type, ncols_dst: c_int, table_id: TableId, small_k: bool, halve_iters: bool) c_int {
    switch (table_id) {
        .generic => return switch (ncols_dst) {
            1, 2, 3, 4 => 4,
            5, 6, 7, 8 => 2,
            else => 1,
        },
        .turing => {
            if (ncols_dst == 1) {
                return switch (ty) {
                    ggml.impl.ggml_c.GGML_TYPE_Q2_K, ggml.impl.ggml_c.GGML_TYPE_Q3_K, ggml.impl.ggml_c.GGML_TYPE_Q4_K, ggml.impl.ggml_c.GGML_TYPE_Q5_K, ggml.impl.ggml_c.GGML_TYPE_Q6_K => 2,
                    else => 4,
                };
            }
            return switch (ncols_dst) {
                2, 3, 4 => 4,
                5, 6, 7, 8 => 2,
                else => 1,
            };
        },
        .gb10 => {
            const generic = calcNwarps(ty, ncols_dst, .generic, false, false);
            // Only worth the wider block when it actually retires the K loop
            // in half the trips (observation).
            if (ncols_dst == 1 and !small_k and halve_iters) {
                switch (ty) {
                    ggml.impl.ggml_c.GGML_TYPE_Q4_0,
                    ggml.impl.ggml_c.GGML_TYPE_Q4_1,
                    ggml.impl.ggml_c.GGML_TYPE_Q5_0,
                    ggml.impl.ggml_c.GGML_TYPE_Q5_1,
                    ggml.impl.ggml_c.GGML_TYPE_Q8_0,
                    ggml.impl.ggml_c.GGML_TYPE_Q4_K,
                    ggml.impl.ggml_c.GGML_TYPE_Q5_K,
                    ggml.impl.ggml_c.GGML_TYPE_Q6_K,
                    ggml.impl.ggml_c.GGML_TYPE_IQ4_NL,
                    => return 2 * generic,
                    else => {},
                }
            }
            return generic;
        },
        else => unreachable,
    }
}

/// Ports `mmvq_two_rows_ncols1` (mmvq.cu:579 @b1b770d90): low bit types load
/// few weight bytes per thread, a second row per block shares the q8_1
/// loads and adds memory parallelism.
fn twoRowsNcols1(ty: ggml.impl.ggml_c.enum_ggml_type) bool {
    return switch (ty) {
        ggml.impl.ggml_c.GGML_TYPE_Q1_0, ggml.impl.ggml_c.GGML_TYPE_Q2_K, ggml.impl.ggml_c.GGML_TYPE_Q3_K, ggml.impl.ggml_c.GGML_TYPE_IQ1_S, ggml.impl.ggml_c.GGML_TYPE_IQ1_M => true,
        else => false,
    };
}

/// Ports `calc_rows_per_block` (mmvq.cu:592 @b1b770d90). The kernel computes
/// the same number on the device, from `get_device_table_id`.
fn calcRowsPerBlock(ty: ggml.impl.ggml_c.enum_ggml_type, ncols_dst: c_int, table_id: TableId, small_k: bool, nwarps: c_int) c_int {
    if (table_id == .generic or table_id == .gcn or table_id == .turing or table_id == .gb10) {
        return switch (ncols_dst) {
            1 => {
                if (small_k) return nwarps;
                if (table_id == .gb10) return if (ty == ggml.impl.ggml_c.GGML_TYPE_IQ2_S or ty == ggml.impl.ggml_c.GGML_TYPE_IQ3_S) 2 else 1;
                return if (table_id == .generic and twoRowsNcols1(ty)) 2 else 1;
            },
            2, 3, 4, 5, 6, 7, 8 => 2,
            else => 1,
        };
    }
    return 1;
}

/// Ports `calc_launch_params` (mmvq.cu:1032 @b1b770d90).
fn calcLaunchParams(ty: ggml.impl.ggml_c.enum_ggml_type, ncols_dst: c_int, nrows_x: c_int, nchannels_dst: c_int, nsamples_or_ntokens: c_int, warp_size: c_int, table_id: TableId, small_k: bool, halve_iters: bool) [2]ggml.cuda.kernels.Dim3 {
    const nwarps = calcNwarps(ty, ncols_dst, table_id, small_k, halve_iters);
    const rpb = calcRowsPerBlock(ty, ncols_dst, table_id, small_k, nwarps);
    const nblocks: i64 = @divTrunc(@as(i64, nrows_x) + rpb - 1, rpb);
    return .{
        .{ .x = @truncate(@as(u64, @bitCast(nblocks))), .y = @bitCast(nchannels_dst), .z = @bitCast(nsamples_or_ntokens) },
        .{ .x = @bitCast(warp_size), .y = @bitCast(nwarps), .z = 1 },
    };
}

/// The arguments every `mul_mat_vec_q` launch takes after the kernel's
/// template parameters, at the kernel's parameter types.
const Common = struct {
    vx: ?*const anyopaque,
    vy: ?*const anyopaque,
    ids: ?*const i32,
    fusion: ggml.cuda.common.MmFusionArgsDevice,
    dst: ?*f32,
    ncols_x: u32,
    nchannels_y: ggml.cuda.common.cu.uint3,
    stride_row_x: u32,
    stride_col_y: u32,
    stride_col_dst: u32,
    channel_ratio: ggml.cuda.common.cu.uint3,
    stride_channel_x: u32,
    stride_channel_y: u32,
    stride_channel_dst: u32,
    sample_ratio: ggml.cuda.common.cu.uint3,
    stride_sample_x: u32,
    stride_sample_y: u32,
    stride_sample_dst: u32,
    ids_stride: u32,
};

fn mmvqKernel(comptime ty: ggml.impl.ggml_c.enum_ggml_type, comptime ncols_dst: comptime_int, comptime has_fusion: bool, comptime small_k: bool, comptime halve_iters: bool) ggml.cuda.kernels.Kernel {
    return ggml.cuda.kernels.get("mmvq", ggml.cuda.kernels.tmpl("mul_mat_vec_q", &.{ ggml.cuda.kernels.ggmlType(ty), ggml.cuda.kernels.int(ncols_dst), ggml.cuda.kernels.boolean(has_fusion), ggml.cuda.kernels.boolean(small_k), ggml.cuda.kernels.boolean(halve_iters) }));
}

/// Ports `mul_mat_vec_q_switch_fusion` (mmvq.cu:1044 @b1b770d90).
fn switchFusion(comptime ty: ggml.impl.ggml_c.enum_ggml_type, comptime ncols_dst: comptime_int, comptime small_k: bool, comptime halve_iters: bool, a: *const Common, dims: [2]ggml.cuda.kernels.Dim3, nbytes_shared: c_int, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const f = a.fusion;
    const has_fusion = f.gate != null or f.x_bias != null or f.gate_bias != null or f.x_scale != null or f.gate_scale != null;
    const p: ggml.cuda.kernels.LaunchParams = .{ .grid = dims[0], .block = dims[1], .shmem = @intCast(nbytes_shared), .stream = stream };
    const args = .{
        a.vx,              a.vy,               a.ids,               a.fusion,             a.dst,
        a.ncols_x,         a.nchannels_y,      a.stride_row_x,      a.stride_col_y,       a.stride_col_dst,
        a.channel_ratio,   a.stride_channel_x, a.stride_channel_y,  a.stride_channel_dst, a.sample_ratio,
        a.stride_sample_x, a.stride_sample_y,  a.stride_sample_dst, a.ids_stride,
    };
    if (ncols_dst == 1) {
        if (has_fusion) {
            ggml.cuda.kernels.launchPdl(comptime mmvqKernel(ty, ncols_dst, true, small_k, halve_iters), p, args);
            return;
        }
    }

    ggml.impl.assert(!has_fusion, "!has_fusion && \"fusion only supported for ncols_dst=1\"");

    ggml.cuda.kernels.launchPdl(comptime mmvqKernel(ty, ncols_dst, false, small_k, halve_iters), p, args);
}

/// Ports `mul_mat_vec_q_moe_launch` (mmvq.cu:1076 @b1b770d90). The kernel
/// takes the fusion arguments by value either way; `has_fusion` picks the
/// instantiation that reads them. A shared expert is one more
/// channel of blocks, the last, which the kernel points at its weights.
fn moeLaunch(comptime ty: ggml.impl.ggml_c.enum_ggml_type, a: *const Common, nrows_x: u32, ncols_dst: u32, warp_size: c_int, nchannels_dst: c_int, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const rows_per_block = 2; // 2 gives best perf based on tuning
    const nblocks_rows: i64 = @divTrunc(@as(i64, nrows_x) + rows_per_block - 1, rows_per_block);
    const p: ggml.cuda.kernels.LaunchParams = .{
        .grid = .{ .x = @intCast(nblocks_rows), .y = @bitCast(nchannels_dst + @intFromBool(a.fusion.shared_up != null)) },
        .block = .{ .x = @bitCast(warp_size), .y = ncols_dst },
        .stream = stream,
    };
    const args = .{
        a.vx,               a.vy,                 a.ids,            a.fusion,
        a.dst,              a.ncols_x,            a.nchannels_y,    nrows_x,
        a.stride_row_x,     a.stride_col_y,       a.stride_col_dst, a.stride_channel_x,
        a.stride_channel_y, a.stride_channel_dst, ncols_dst,        a.ids_stride,
    };

    const f = a.fusion;
    const has_fusion = f.gate != null or f.x_bias != null or f.gate_bias != null or
        f.x_scale != null or f.gate_scale != null;

    if (has_fusion) {
        ggml.cuda.kernels.launchPdl(comptime moeKernel(ty, rows_per_block, true), p, args);
    } else {
        ggml.cuda.kernels.launchPdl(comptime moeKernel(ty, rows_per_block, false), p, args);
    }
}

fn moeKernel(comptime ty: ggml.impl.ggml_c.enum_ggml_type, comptime rows_per_block: comptime_int, comptime has_fusion: bool) ggml.cuda.kernels.Kernel {
    return ggml.cuda.kernels.get("mmvq", ggml.cuda.kernels.tmpl("mul_mat_vec_q_moe", &.{ ggml.cuda.kernels.ggmlType(ty), ggml.cuda.kernels.int(rows_per_block), ggml.cuda.kernels.boolean(has_fusion) }));
}

fn contains(list: []const ggml.impl.ggml_c.enum_ggml_type, t: ggml.impl.ggml_c.enum_ggml_type) bool {
    for (list) |x| if (x == t) return true;
    return false;
}

/// Ports `mul_mat_vec_q_switch_ncols_dst` (mmvq.cu:1109 @b1b770d90).
fn switchNcolsDst(
    comptime ty: ggml.impl.ggml_c.enum_ggml_type,
    vx: ?*const anyopaque,
    vy: ?*const anyopaque,
    ids: ?*const i32,
    fusion: ggml.cuda.common.MmFusionArgsDevice,
    dst: ?*f32,
    ncols_x: c_int,
    nrows_x: c_int,
    ncols_dst: c_int,
    stride_row_x: c_int,
    stride_col_y: c_int,
    stride_col_dst: c_int,
    nchannels_x: c_int,
    nchannels_y: c_int,
    nchannels_dst: c_int,
    stride_channel_x: c_int,
    stride_channel_y: c_int,
    stride_channel_dst: c_int,
    nsamples_x: c_int,
    nsamples_dst: c_int,
    stride_sample_x: c_int,
    stride_sample_y: c_int,
    stride_sample_dst: c_int,
    ids_stride: c_int,
    stream: ggml.cuda.common.cu.cudaStream_t,
) void {
    ggml.impl.assert(@rem(ncols_x, @as(c_int, @intCast(ggml.impl.ggml_c.ggml_blck_size(ty)))) == 0, "ncols_x % ggml_blck_size(type) == 0");
    ggml.impl.assert(ncols_dst <= max_batch_size, "ncols_dst <= MMVQ_MAX_BATCH_SIZE");

    const zero3: ggml.cuda.common.cu.uint3 = .{ .x = 0, .y = 0, .z = 0 };
    const nchannels_y_fd = if (ids != null) ggml.cuda.common.initFastdivValues(@bitCast(@as(i64, nchannels_y))) else zero3;
    const channel_ratio_fd = if (ids != null) zero3 else ggml.cuda.common.initFastdivValues(@bitCast(@as(i64, @divTrunc(nchannels_dst, nchannels_x))));
    const sample_ratio_fd = ggml.cuda.common.initFastdivValues(@bitCast(@as(i64, @divTrunc(nsamples_dst, nsamples_x))));

    const device = ggml.cuda.common.getDevice();
    const x = ggml.cuda.common.info().devices[@intCast(device)].cc;
    const warp_size = ggml.cuda.common.info().devices[@intCast(device)].warp_size;
    const table_id = getDeviceTableId(x);

    const has_ids = ids != null;

    // How the K loop divides up at the baseline block width.
    const traits = comptime ggml.cuda.common.typeTraits(ty);
    const vdr = comptime getVdrMmvq(ty);
    const blocks_per_row_x: c_int = @divTrunc(ncols_x, traits.qk);
    const blocks_per_iter_1warp: c_int = @divTrunc(vdr * warp_size, traits.qi);

    const a: Common = .{
        .vx = vx,
        .vy = vy,
        .ids = ids,
        .fusion = fusion,
        .dst = dst,
        .ncols_x = @bitCast(ncols_x),
        .nchannels_y = nchannels_y_fd,
        .stride_row_x = @bitCast(stride_row_x),
        .stride_col_y = @bitCast(stride_col_y),
        .stride_col_dst = @bitCast(stride_col_dst),
        .channel_ratio = channel_ratio_fd,
        .stride_channel_x = @bitCast(stride_channel_x),
        .stride_channel_y = @bitCast(stride_channel_y),
        .stride_channel_dst = @bitCast(stride_channel_dst),
        .sample_ratio = sample_ratio_fd,
        .stride_sample_x = @bitCast(stride_sample_x),
        .stride_sample_y = @bitCast(stride_sample_y),
        .stride_sample_dst = @bitCast(stride_sample_dst),
        .ids_stride = @bitCast(ids_stride),
    };

    if (has_ids and ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        moeLaunch(ty, &a, @bitCast(nrows_x), @bitCast(ncols_dst), warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        1 => {
            const c_ncols_dst = 1;
            // The C's `should_use_small_k` lambda.
            const use_small_k = blk: {
                const nwarps = calcNwarps(ty, c_ncols_dst, table_id, false, false);
                var use = nwarps > 1 and blocks_per_row_x < nwarps * blocks_per_iter_1warp;

                const iq_slow_turing = [_]ggml.impl.ggml_c.enum_ggml_type{ ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS, ggml.impl.ggml_c.GGML_TYPE_IQ3_S };
                const iq_slow_other = [_]ggml.impl.ggml_c.enum_ggml_type{
                    ggml.impl.ggml_c.GGML_TYPE_IQ1_S, ggml.impl.ggml_c.GGML_TYPE_IQ1_M,   ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS, ggml.impl.ggml_c.GGML_TYPE_IQ2_XS,
                    ggml.impl.ggml_c.GGML_TYPE_IQ2_S, ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS, ggml.impl.ggml_c.GGML_TYPE_IQ3_S,   ggml.impl.ggml_c.GGML_TYPE_IQ4_XS,
                };
                const slow_pascal = [_]ggml.impl.ggml_c.enum_ggml_type{ ggml.impl.ggml_c.GGML_TYPE_IQ3_S, ggml.impl.ggml_c.GGML_TYPE_Q2_K, ggml.impl.ggml_c.GGML_TYPE_Q3_K };

                const is_nvidia_turing_plus = ggml.cuda.common.cc.isNvidia(x) and x >= ggml.cuda.common.cc.turing;
                const is_nvidia_pascal_older = ggml.cuda.common.cc.isNvidia(x) and x < ggml.cuda.common.cc.volta;

                if (is_nvidia_turing_plus) {
                    if (ncols_dst == 1 and contains(&iq_slow_turing, ty)) use = false;
                } else if ((ncols_dst == 1 and contains(&iq_slow_other, ty)) or
                    (is_nvidia_pascal_older and contains(&slow_pascal, ty)))
                {
                    use = false;
                }
                break :blk use;
            };

            // The C's `should_halve_iters` lambda.
            const halve = blk: {
                if (table_id != .gb10) break :blk false;
                // Expert rows are gathered per token, so a wider block adds
                // reduction work without reuse.
                if (has_ids) break :blk false;

                const blocks_per_iter = calcNwarps(ty, 1, table_id, false, false) * blocks_per_iter_1warp;
                const iters = @divTrunc(blocks_per_row_x + blocks_per_iter - 1, blocks_per_iter);
                const iters_wide = @divTrunc(blocks_per_row_x + blocks_per_iter * 2 - 1, blocks_per_iter * 2);

                // An odd trip count leaves half the wider block idle for its
                // last iteration.
                const idle = iters_wide * 2 - iters;
                break :blk idle * 8 <= iters_wide * 2;
            };

            // Types the table does not promote would compile a second,
            // identical kernel, so the C launches the unhalved one.
            const c_promoted = comptime calcNwarps(ty, c_ncols_dst, .gb10, false, true) != calcNwarps(ty, c_ncols_dst, .gb10, false, false);

            // A shared expert is one more channel of blocks, the last.
            const nchannels_launch = nchannels_dst + @intFromBool(fusion.shared_up != null);
            if (use_small_k) {
                const dims = calcLaunchParams(ty, c_ncols_dst, nrows_x, nchannels_launch, nsamples_dst, warp_size, table_id, true, false);
                switchFusion(ty, c_ncols_dst, true, false, &a, dims, 0, stream);
            } else if (halve and c_promoted) {
                const dims = calcLaunchParams(ty, c_ncols_dst, nrows_x, nchannels_launch, nsamples_dst, warp_size, table_id, false, true);
                switchFusion(ty, c_ncols_dst, false, true, &a, dims, 0, stream);
            } else {
                const dims = calcLaunchParams(ty, c_ncols_dst, nrows_x, nchannels_launch, nsamples_dst, warp_size, table_id, false, false);
                switchFusion(ty, c_ncols_dst, false, false, &a, dims, 0, stream);
            }
        },
        inline 2, 3, 4, 5, 6, 7, 8 => |n| {
            const dims = calcLaunchParams(ty, n, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id, false, false);
            switchFusion(ty, n, false, false, &a, dims, 0, stream);
        },
        else => ggml.impl.abort("fatal error"),
    }
}

/// Ports `mul_mat_vec_q_switch_type` (mmvq.cu:1300 @b1b770d90).
fn switchType(vx: ?*const anyopaque, type_x: ggml.impl.ggml_c.enum_ggml_type, vy: ?*const anyopaque, ids: ?*const i32, fusion: ggml.cuda.common.MmFusionArgsDevice, dst: ?*f32, ncols_x: c_int, nrows_x: c_int, ncols_dst: c_int, stride_row_x: c_int, stride_col_y: c_int, stride_col_dst: c_int, nchannels_x: c_int, nchannels_y: c_int, nchannels_dst: c_int, stride_channel_x: c_int, stride_channel_y: c_int, stride_channel_dst: c_int, nsamples_x: c_int, nsamples_dst: c_int, stride_sample_x: c_int, stride_sample_y: c_int, stride_sample_dst: c_int, ids_stride: c_int, stream: ggml.cuda.common.cu.cudaStream_t) void {
    switch (type_x) {
        inline ggml.impl.ggml_c.GGML_TYPE_Q1_0,
        ggml.impl.ggml_c.GGML_TYPE_Q2_0,
        ggml.impl.ggml_c.GGML_TYPE_Q4_0,
        ggml.impl.ggml_c.GGML_TYPE_Q4_1,
        ggml.impl.ggml_c.GGML_TYPE_Q5_0,
        ggml.impl.ggml_c.GGML_TYPE_Q5_1,
        ggml.impl.ggml_c.GGML_TYPE_Q8_0,
        ggml.impl.ggml_c.GGML_TYPE_MXFP4,
        ggml.impl.ggml_c.GGML_TYPE_NVFP4,
        ggml.impl.ggml_c.GGML_TYPE_Q2_K,
        ggml.impl.ggml_c.GGML_TYPE_Q3_K,
        ggml.impl.ggml_c.GGML_TYPE_Q4_K,
        ggml.impl.ggml_c.GGML_TYPE_Q5_K,
        ggml.impl.ggml_c.GGML_TYPE_Q6_K,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XS,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_S,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS,
        ggml.impl.ggml_c.GGML_TYPE_IQ1_S,
        ggml.impl.ggml_c.GGML_TYPE_IQ1_M,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_NL,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_XS,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_S,
        => |t| switchNcolsDst(t, vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream),
        else => ggml.impl.abort("fatal error"),
    }
}

fn i32Of(x: i64) c_int {
    return @truncate(x);
}

/// Ports `ggml_cuda_mul_mat_vec_q` (mmvq.cu:1452 @b1b770d90).
pub fn mulMatVecQ(ctx: *ggml.cuda.common.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, ids: ?*const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor, fusion: ?*const ggml.cuda.common.MmFusionArgsHost) void {
    ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "        src1->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "        dst->type  == GGML_TYPE_F32");
    ggml.impl.assert(ids == null or ids.?.type == ggml.impl.ggml_c.GGML_TYPE_I32, "!ids || ids->type  == GGML_TYPE_I32");

    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];
    const ne10 = src1.ne[0];
    const ne11 = src1.ne[1];
    const ne12 = src1.ne[2];
    const ne13 = src1.ne[3];
    const ne1 = dst.ne[1];
    const ne2 = dst.ne[2];
    const ne3 = dst.ne[3];

    const stream = ctx.stream();

    const ts_src0 = ggml.impl.ggml_c.ggml_type_size(src0.type);
    const ts_src1 = ggml.impl.ggml_c.ggml_type_size(src1.type);
    const ts_dst = ggml.impl.ggml_c.ggml_type_size(dst.type);

    ggml.impl.assert(src0.nb[0] == ts_src0, "        nb00       == ts_src0");
    ggml.impl.assert(src1.nb[0] == ts_src1, "        nb10       == ts_src1");
    ggml.impl.assert(dst.nb[0] == ts_dst, "        nb0        == ts_dst");
    ggml.impl.assert(ids == null or ids.?.nb[0] == ggml.impl.ggml_c.ggml_type_size(ids.?.type), "!ids || ids->nb[0] == ggml_type_size(ids->type)");

    ggml.impl.assert(ids == null or ne12 <= max_batch_size, "!ids || ne12 <= MMVQ_MAX_BATCH_SIZE");

    const src1_d: ?*const f32 = @ptrCast(@alignCast(src1.data));
    const ids_d: ?*const i32 = if (ids) |i| @ptrCast(@alignCast(i.data)) else null;
    const dst_d: ?*f32 = @ptrCast(@alignCast(dst.data));

    var fusion_local: ggml.cuda.common.MmFusionArgsDevice = .{};

    if (fusion) |f| {
        const x = ggml.cuda.common.info().devices[@intCast(ggml.cuda.common.getDevice())].cc;
        ggml.impl.assert(ids == null or dst.ne[2] <= getMmidMaxBatch(src0.type, x), " !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)");
        ggml.impl.assert(ids != null or dst.ne[1] == 1, "  ids || dst->ne[1] == 1");
        // Scale fusion is only allowed for NVFP4 currently.
        ggml.impl.assert((f.x_scale == null and f.gate_scale == null) or src0.type == ggml.impl.ggml_c.GGML_TYPE_NVFP4, "(fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4");

        if (f.shared_up) |su| {
            ggml.impl.assert(ids != null and f.gate != null and f.shared_gate != null and f.shared_dst != null, "ids && fusion->gate && fusion->shared_gate && fusion->shared_dst");
            ggml.impl.assert(f.x_bias == null and f.gate_bias == null and f.x_scale == null and f.gate_scale == null, "!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale");
            ggml.impl.assert(ne11 == 1 and ne03 == 1 and ne13 == 1, "ne11 == 1 && ne03 == 1 && ne13 == 1");
            const sg = f.shared_gate.?;
            const sd = f.shared_dst.?;
            ggml.impl.assert(su.type == src0.type and sg.type == src0.type, "fusion->shared_up->type == src0->type && fusion->shared_gate->type == src0->type");
            ggml.impl.assert(ggml.impl.ggml_c.ggml_are_same_shape(su, sg), "ggml_are_same_shape(fusion->shared_up, fusion->shared_gate)");
            ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(su) and ggml.impl.ggml_c.ggml_is_contiguous(sg), "ggml_is_contiguous(fusion->shared_up) && ggml_is_contiguous(fusion->shared_gate)");
            ggml.impl.assert(su.ne[0] == ne00 and su.ne[1] == ne01, "fusion->shared_up->ne[0] == ne00 && fusion->shared_up->ne[1] == ne01");
            ggml.impl.assert(su.nb[1] == src0.nb[1] and ggml.impl.ggml_c.ggml_is_matrix(su), "fusion->shared_up->nb[1] == nb01 && ggml_is_matrix(fusion->shared_up)");
            ggml.impl.assert(sd.type == ggml.impl.ggml_c.GGML_TYPE_F32 and ggml.impl.ggml_c.ggml_is_contiguous(sd), "fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst)");
            ggml.impl.assert(sd.ne[0] == dst.ne[0] and sd.ne[1] == ne2, "fusion->shared_dst->ne[0] == ne0 && fusion->shared_dst->ne[1] == ne2");
            fusion_local.shared_up = su.data;
            fusion_local.shared_gate = sg.data;
            fusion_local.shared_dst = @ptrCast(@alignCast(sd.data));
            fusion_local.shared_stride_col_dst = @truncate(sd.nb[1] / ts_dst);
        }

        if (f.x_bias) |b| {
            ggml.impl.assert(b.type == ggml.impl.ggml_c.GGML_TYPE_F32, "fusion->x_bias->type == GGML_TYPE_F32");
            ggml.impl.assert(b.ne[0] == dst.ne[0], "fusion->x_bias->ne[0] == dst->ne[0]");
            ggml.impl.assert(ids == null or b.ne[1] == src0.ne[2], "!ids || fusion->x_bias->ne[1] == src0->ne[2]");
            fusion_local.x_bias = b.data;
        }
        if (f.gate) |g| {
            ggml.impl.assert(g.type == src0.type and ggml.impl.ggml_c.ggml_are_same_stride(g, src0), "fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0)");
            fusion_local.gate = g.data;
        }
        if (f.gate_bias) |b| {
            ggml.impl.assert(b.type == ggml.impl.ggml_c.GGML_TYPE_F32, "fusion->gate_bias->type == GGML_TYPE_F32");
            ggml.impl.assert(b.ne[0] == dst.ne[0], "fusion->gate_bias->ne[0] == dst->ne[0]");
            ggml.impl.assert(ids == null or b.ne[1] == src0.ne[2], "!ids || fusion->gate_bias->ne[1] == src0->ne[2]");
            fusion_local.gate_bias = b.data;
        }
        if (f.x_scale) |s| {
            ggml.impl.assert(s.type == ggml.impl.ggml_c.GGML_TYPE_F32, "fusion->x_scale->type == GGML_TYPE_F32");
            ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(s), "ggml_is_contiguous(fusion->x_scale)");
            ggml.impl.assert(ggml.impl.ggml_c.ggml_nelements(s) == (if (ids != null) src0.ne[2] else 1), "ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1)");
            fusion_local.x_scale = s.data;
        }
        if (f.gate_scale) |s| {
            ggml.impl.assert(s.type == ggml.impl.ggml_c.GGML_TYPE_F32, "fusion->gate_scale->type == GGML_TYPE_F32");
            ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(s), "ggml_is_contiguous(fusion->gate_scale)");
            ggml.impl.assert(ggml.impl.ggml_c.ggml_nelements(s) == (if (ids != null) src0.ne[2] else 1), "ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1)");
            fusion_local.gate_scale = s.data;
        }
        fusion_local.glu_op = f.glu_op;
        fusion_local.glu_limit = f.glu_limit;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml.impl.ggml_c.ggml_backend_buffer_get_usage(src0.buffer) == ggml.impl.ggml_c.GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_data = ggml.impl.ggml_c.ggml_nbytes(src0);
        const size_alloc = ggml.impl.ggml_c.ggml_backend_buffer_get_alloc_size(src0.buffer, src0);
        if (size_alloc > size_data) {
            ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguously_allocated(src0), "ggml_is_contiguously_allocated(src0)");
            ggml.impl.assert(src0.view_src == null, "!src0->view_src");
            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemsetAsync(@as([*]u8, @ptrCast(src0.data.?)) + size_data, 0, size_alloc - size_data, stream), "cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream)", @src());
        }
    }

    const ne10_padded: i64 = @intCast(stdlib.num.pad(usize, @intCast(ne10), ggml.cuda.common.matrix_row_padding));
    var src1_q8_1 = ggml.cuda.common.PoolAlloc(u8).initAlloc(ctx.pool(), @intCast(@divTrunc(ne13 * ne12 * ne11 * ne10_padded * ggml.cuda.mmq.sizeof_block_q8_1, ggml.impl.ggml_c.QK8_1)));
    defer src1_q8_1.deinit();
    {
        const s11: i64 = @intCast(src1.nb[1] / ts_src1);
        const s12: i64 = @intCast(src1.nb[2] / ts_src1);
        const s13: i64 = @intCast(src1.nb[3] / ts_src1);
        ggml.cuda.quantize.quantizeRowQ8_1(src1_d, src1_q8_1.get(), ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const s01: i64 = @intCast(src0.nb[1] / ts_src0);
    const s11: i64 = @divTrunc(ne10_padded, ggml.impl.ggml_c.QK8_1);
    const s1: i64 = @intCast(dst.nb[1] / ts_dst);
    const s02: i64 = @intCast(src0.nb[2] / ts_src0);
    const s2: i64 = @intCast(dst.nb[2] / ts_dst);
    const s03: i64 = @intCast(src0.nb[3] / ts_src0);
    const s3: i64 = @intCast(dst.nb[3] / ts_dst);

    const s12 = ne11 * s11;
    const s13 = ne12 * s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const has_ids = ids != null;
    const ncols_dst = if (has_ids) ne2 else ne1;
    const nchannels_y = if (has_ids) ne11 else ne12;
    const nchannels_dst = if (has_ids) ne1 else ne2;
    const stride_col_dst = if (has_ids) s2 else s1;
    const stride_col_y = if (has_ids) s12 else s11;
    const stride_channel_dst = if (has_ids) s1 else s2;
    const stride_channel_y = if (has_ids) s11 else s12;

    const ids_stride: i64 = if (ids) |i| @intCast(i.nb[1] / ggml.impl.ggml_c.ggml_type_size(i.type)) else 0;

    switchType(
        src0.data,
        src0.type,
        src1_q8_1.get(),
        ids_d,
        fusion_local,
        dst_d,
        i32Of(ne00),
        i32Of(ne01),
        i32Of(ncols_dst),
        i32Of(s01),
        i32Of(stride_col_y),
        i32Of(stride_col_dst),
        i32Of(ne02),
        i32Of(nchannels_y),
        i32Of(nchannels_dst),
        i32Of(s02),
        i32Of(stride_channel_y),
        i32Of(stride_channel_dst),
        i32Of(ne03),
        i32Of(ne3),
        i32Of(s03),
        i32Of(s13),
        i32Of(s3),
        i32Of(ids_stride),
        stream,
    );
}
