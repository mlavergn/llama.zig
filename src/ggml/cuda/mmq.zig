//! MMQ: quantized matrix multiplication with integer tensor cores (or dp4a),
//! `src1` quantized to `q8_1` (or FP4 on Blackwell) first.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/mmq.cu` and the host half of
//! `llama.cpp/ggml/src/ggml-cuda/mmq.cuh` at v0.4.1 (`b29c606e2`). The
//! configuration tables of `mmq-config-{pascal,ampere,blackwell}.cuh` are
//! generated into `mmq_config_table.zig`. The kernels, `mul_mat_q` and
//! `mul_mat_q_stream_k_fixup`, stay CUDA, one instantiation set per type in
//! `template-instances/mmq-instance-<type>.cu`.
//!
//! # Host-side configuration
//!
//! The host chooses a configuration with `ggml_cuda_mmq_get_config(…, cc)`
//! and the kernel recomputes the same one from `__CUDA_ARCH__`; they agree
//! because `blackwell_mma_available(cc)` and the device's
//! `BLACKWELL_MMA_AVAILABLE` both test the one architecture the fatbins hold.
//! Only the NVIDIA arms are ported.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");
const quantize = @import("quantize.zig");
const mmid = @import("mmid.zig");
const table = @import("mmq_config_table.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;
const q = common.q;

/// Ports `MMQ_DP4A_MAX_BATCH_SIZE` (mmq.cuh:8 @b29c606e2).
const dp4a_max_batch_size = 64;

/// Ports `MMQ_TILE_NE_K` (mmq.cuh:116 @b29c606e2).
const tile_ne_k = 32;

/// Ports `QK8_1_MMQ` (mmq.cuh:24 @b29c606e2).
pub const qk8_1_mmq = 4 * c.QK8_1;

/// Ports `QK_FP4_MMQ` (mmq.cuh:25 @b29c606e2).
pub const qk_fp4_mmq = 2 * qk8_1_mmq;

/// `sizeof(block_q8_1)`: two halves and `QK8_1` bytes (ggml-common.h:269 @b29c606e2).
pub const sizeof_block_q8_1 = 2 * 2 + c.QK8_1;

/// The size of `block_q8_1_mmq` (mmq.cuh:56 @b29c606e2): `QK8_1_MMQ + 4*sizeof(half2)`.
pub const sizeof_block_q8_1_mmq = qk8_1_mmq + 4 * 4;

/// The size of `block_fp4_mmq` (mmq.cuh:58 @b29c606e2), equal to `block_q8_1_mmq`'s.
pub const sizeof_block_fp4_mmq = sizeof_block_q8_1_mmq;

/// Ports `mmq_q8_1_ds_layout` (mmq.cuh:18 @b29c606e2).
pub const Q8_1DsLayout = enum(c_int) { d4 = 0, ds4 = 1, d2s6 = 2 };

/// Ports `mmq_get_q8_1_ds_layout` (mmq.cuh:60 @b29c606e2).
pub fn getQ8_1DsLayout(type_x: c.enum_ggml_type) Q8_1DsLayout {
    return switch (type_x) {
        c.GGML_TYPE_Q1_0, c.GGML_TYPE_Q2_0 => .d4,
        c.GGML_TYPE_Q4_0, c.GGML_TYPE_Q4_1 => .ds4,
        c.GGML_TYPE_Q5_0 => .d4,
        c.GGML_TYPE_Q5_1 => .ds4,
        c.GGML_TYPE_Q8_0 => .d4,
        c.GGML_TYPE_MXFP4 => .d4,
        c.GGML_TYPE_NVFP4 => .d4,
        c.GGML_TYPE_Q2_K => .d2s6,
        c.GGML_TYPE_Q3_K => .d4,
        c.GGML_TYPE_Q4_K, c.GGML_TYPE_Q5_K => .ds4,
        c.GGML_TYPE_Q6_K, c.GGML_TYPE_IQ2_XXS, c.GGML_TYPE_IQ2_XS, c.GGML_TYPE_IQ2_S, c.GGML_TYPE_IQ3_XXS, c.GGML_TYPE_IQ3_S => .d4,
        c.GGML_TYPE_IQ1_S => .ds4,
        c.GGML_TYPE_IQ4_XS, c.GGML_TYPE_IQ4_NL => .d4,
        else => impl.abort("fatal error"),
    };
}

/// Ports `ggml_cuda_mmq_sram_layout` (mmq.cuh:122 @b29c606e2).
pub const SramLayout = enum { q8_0, q8_1, q2_k, q3_k, q6_k, fp4, nvfp4 };

/// Ports `ggml_cuda_mmq_get_sram_stride` (mmq.cuh:132 @b29c606e2): the
/// padded SRAM tile length, in 32-bit elements.
fn getSramStride(layout: SramLayout) c_int {
    return switch (layout) {
        .q8_0 => 2 * tile_ne_k + 2 * tile_ne_k / q.QI8_0 + 4,
        .q8_1 => 2 * tile_ne_k + 2 * tile_ne_k / q.QI8_1 + 4,
        .q2_k => 2 * tile_ne_k + tile_ne_k + 4,
        .q3_k => 2 * tile_ne_k + tile_ne_k / 2 + 4,
        .q6_k => 2 * tile_ne_k + tile_ne_k / q.QI6_K + tile_ne_k / 8 + 7,
        .fp4 => 2 * tile_ne_k + 8 + 4,
        .nvfp4 => 2 * tile_ne_k + tile_ne_k / 2 + 4,
    };
}

/// Ports `ggml_cuda_mmq_config` (mmq.cuh:165 @b29c606e2).
pub const Config = struct {
    type: c.enum_ggml_type,
    nthreads: c_int,
    occupancy: c_int,
    I: c_int,
    J: c_int,
    sram_layout: SramLayout,
    K_vram: c_int,
    stream_k: bool,
    fallback: bool,

    /// Ports `ggml_cuda_mmq_config`'s `use_mma_data_layout` (mmq.cuh:189 @b29c606e2).
    fn useMmaDataLayout(self: Config, x: c_int) bool {
        _ = self;
        return common.turingMmaAvailable(x);
    }
};

fn lookup(t: *const table.Table, ty: c.enum_ggml_type, J: c_int, fallback: bool) Config {
    for (t.cases) |e| {
        if (e.type == ty and e.J == J and e.fallback == fallback) return e;
    }
    if (t.default) |d| return d;
    return lookup(t.fallthrough.?, ty, J, fallback);
}

/// Ports the host `ggml_cuda_mmq_get_config` (mmq.cuh:230 @b29c606e2),
/// NVIDIA arms. The AMD arms -- GCN, CDNA, RDNA2/3/4 -- are taken only for
/// an AMD compute capability, which a CUDA device never has.
pub fn getConfig(ty: c.enum_ggml_type, J: c_int, fallback: bool, x: c_int) Config {
    if (common.blackwellMmaAvailable(x)) return lookup(&table.blackwell, ty, J, fallback);
    if (common.highestCompiledArch(x) >= common.cc.volta) return lookup(&table.ampere, ty, J, fallback);
    if (common.highestCompiledArch(x) >= common.cc.dp4a) return lookup(&table.pascal_dp4a, ty, J, fallback);
    return lookup(&table.pascal_older, ty, J, fallback);
}

/// Ports `ggml_cuda_mmq_get_J_max` (mmq.cuh:372 @b29c606e2).
fn getJMax(ty: c.enum_ggml_type, fallback: bool, x: c_int, ne11: i64) c_int {
    var ret: c_int = @intCast(@min(ne11, 512));
    ret -= @rem(ret, 8);
    while (ret > 0) : (ret -= 8) {
        if (getConfig(ty, ret, fallback, x).type != c.GGML_TYPE_COUNT) return ret;
    }
    return ret;
}

/// Ports `tile_x_sizes` (mmq.cuh:103 @b29c606e2).
const TileXSizes = struct { qs: c_int, dm: c_int, sc: c_int };

/// Ports `mmq_get_dp4a_tile_x_sizes` (mmq.cuh:398 @b29c606e2) with the
/// `MMQ_DP4A_TXS_*` macros, from `MMQ_DP4A_TXS_Q4_0` (mmq.cuh:387 @b29c606e2).
fn getDp4aTileXSizes(ty: c.enum_ggml_type, I: c_int) TileXSizes {
    const K = tile_ne_k;
    const q4_0: TileXSizes = .{ .qs = I * K + I, .dm = @divTrunc(I * K, q.QI4_0) + @divTrunc(I, q.QI4_0), .sc = 0 };
    const q4_1: TileXSizes = .{ .qs = I * K + I, .dm = @divTrunc(I * K, q.QI4_1) + @divTrunc(I, q.QI4_1), .sc = 0 };
    const q8_0: TileXSizes = .{ .qs = I * K * 2 + I, .dm = @divTrunc(I * K * 2, q.QI8_0) + @divTrunc(I, q.QI8_0 / 2), .sc = 0 };
    const q8_0_16: TileXSizes = .{ .qs = I * K * 2 + I, .dm = @divTrunc(I * K * 4, q.QI8_0) + @divTrunc(I, q.QI8_0 / 4), .sc = 0 };
    const q8_1: TileXSizes = .{ .qs = I * K * 2 + I, .dm = @divTrunc(I * K * 2, q.QI8_1) + @divTrunc(I, q.QI8_1 / 2), .sc = 0 };
    const q2_k: TileXSizes = .{ .qs = I * K * 2 + I, .dm = I * K + I, .sc = 0 };
    const q3_k: TileXSizes = .{ .qs = I * K * 2 + I, .dm = I, .sc = @divTrunc(I * K, 8) + @divTrunc(I, 8) };
    const q4_k: TileXSizes = .{ .qs = I * K + I, .dm = @divTrunc(I * K, q.QI4_K), .sc = @divTrunc(I * K, 8) + @divTrunc(I, 8) };
    const q5_k: TileXSizes = .{ .qs = I * K * 2 + I, .dm = @divTrunc(I * K, q.QI5_K) + @divTrunc(I, q.QI5_K), .sc = @divTrunc(I * K, 8) + @divTrunc(I, 8) };
    const q6_k: TileXSizes = .{ .qs = I * K * 2 + I, .dm = @divTrunc(I * K, q.QI6_K) + @divTrunc(I, q.QI6_K), .sc = @divTrunc(I * K, 8) + @divTrunc(I, 8) };
    return switch (ty) {
        c.GGML_TYPE_Q1_0, c.GGML_TYPE_Q2_0 => q8_0,
        c.GGML_TYPE_Q4_0 => q4_0,
        c.GGML_TYPE_Q4_1 => q4_1,
        c.GGML_TYPE_Q5_0 => q8_0,
        c.GGML_TYPE_Q5_1 => q8_1,
        c.GGML_TYPE_Q8_0 => q8_0,
        c.GGML_TYPE_MXFP4 => q8_1,
        c.GGML_TYPE_NVFP4 => q8_0_16,
        c.GGML_TYPE_Q2_K => q2_k,
        c.GGML_TYPE_Q3_K => q3_k,
        c.GGML_TYPE_Q4_K => q4_k,
        c.GGML_TYPE_Q5_K => q5_k,
        c.GGML_TYPE_Q6_K => q6_k,
        c.GGML_TYPE_IQ2_XXS => q8_0,
        c.GGML_TYPE_IQ2_XS, c.GGML_TYPE_IQ2_S => q8_0_16,
        c.GGML_TYPE_IQ3_XXS, c.GGML_TYPE_IQ3_S, c.GGML_TYPE_IQ1_S, c.GGML_TYPE_IQ4_XS, c.GGML_TYPE_IQ4_NL => q8_0,
        else => .{ .qs = 0, .dm = 0, .sc = 0 },
    };
}

/// Ports `ggml_cuda_mmq_get_nbytes_shared_x` (mmq.cuh:427 @b29c606e2).
fn getNbytesSharedX(config: Config, x: c_int) c_int {
    if (config.useMmaDataLayout(x)) {
        return config.I * getSramStride(config.sram_layout) * 4;
    }
    const txs = getDp4aTileXSizes(config.type, config.I);
    return (txs.qs + txs.dm + txs.sc) * 4;
}

/// Ports `mmq_get_nbytes_shared` (mmq.cuh:1388 @b29c606e2).
fn getNbytesShared(config: Config, x: c_int) usize {
    const nbs_ids: usize = @as(usize, @intCast(config.J)) * @sizeOf(c_int);
    const nbs_x: usize = @intCast(getNbytesSharedX(config, x));
    const nbs_y: usize = @as(usize, @intCast(config.J)) * sizeof_block_q8_1_mmq;
    return nbs_ids + nbs_x + impl.pad(nbs_y, @as(usize, @intCast(config.nthreads)) * @sizeOf(c_int));
}

/// Ports `mmq_args` (mmq.cuh:1378 @b29c606e2).
pub const Args = struct {
    x: ?*const anyopaque,
    type_x: c.enum_ggml_type,
    y: ?*const anyopaque,
    ids_dst: ?*const i32,
    expert_bounds: ?*const i32,
    dst: ?*f32,
    y_scale: ?*const f32,
    ncols_x: i64,
    nrows_x: i64,
    ncols_dst: i64,
    stride_row_x: i64,
    ncols_y: i64,
    nrows_dst: i64,
    nchannels_x: i64,
    nchannels_y: i64,
    stride_channel_x: i64,
    stride_channel_y: i64,
    stride_channel_dst: i64,
    nsamples_x: i64,
    nsamples_y: i64,
    stride_sample_x: i64,
    stride_sample_y: i64,
    stride_sample_dst: i64,
    ncols_max: i64,
    /// Value to optimize the tile size against; the launch grid still uses
    /// `ncols_max`.
    ncols_opt: i64,
};

/// The `.cu` stem of a type's template instances.
fn instanceTu(comptime ty: c.enum_ggml_type) []const u8 {
    return switch (ty) {
        c.GGML_TYPE_Q1_0 => "mmq-instance-q1_0",
        c.GGML_TYPE_Q2_0 => "mmq-instance-q2_0",
        c.GGML_TYPE_Q4_0 => "mmq-instance-q4_0",
        c.GGML_TYPE_Q4_1 => "mmq-instance-q4_1",
        c.GGML_TYPE_Q5_0 => "mmq-instance-q5_0",
        c.GGML_TYPE_Q5_1 => "mmq-instance-q5_1",
        c.GGML_TYPE_Q8_0 => "mmq-instance-q8_0",
        c.GGML_TYPE_Q2_K => "mmq-instance-q2_k",
        c.GGML_TYPE_Q3_K => "mmq-instance-q3_k",
        c.GGML_TYPE_Q4_K => "mmq-instance-q4_k",
        c.GGML_TYPE_Q5_K => "mmq-instance-q5_k",
        c.GGML_TYPE_Q6_K => "mmq-instance-q6_k",
        c.GGML_TYPE_IQ1_S => "mmq-instance-iq1_s",
        c.GGML_TYPE_IQ2_XXS => "mmq-instance-iq2_xxs",
        c.GGML_TYPE_IQ2_XS => "mmq-instance-iq2_xs",
        c.GGML_TYPE_IQ2_S => "mmq-instance-iq2_s",
        c.GGML_TYPE_IQ3_XXS => "mmq-instance-iq3_xxs",
        c.GGML_TYPE_IQ3_S => "mmq-instance-iq3_s",
        c.GGML_TYPE_IQ4_XS => "mmq-instance-iq4_xs",
        c.GGML_TYPE_IQ4_NL => "mmq-instance-iq4_nl",
        c.GGML_TYPE_MXFP4 => "mmq-instance-mxfp4",
        c.GGML_TYPE_NVFP4 => "mmq-instance-nvfp4",
        else => @compileError("no MMQ instance for this type"),
    };
}

fn mulMatQKernel(comptime ty: c.enum_ggml_type, comptime J: comptime_int, comptime fallback: bool) kernels.Kernel {
    return kernels.get(instanceTu(ty), kernels.tmpl("mul_mat_q", &.{ kernels.ggmlType(ty), kernels.int(J), kernels.boolean(fallback) }));
}

fn fixupKernel(comptime ty: c.enum_ggml_type, comptime J: comptime_int, comptime fallback: bool) kernels.Kernel {
    return kernels.get(instanceTu(ty), kernels.tmpl("mul_mat_q_stream_k_fixup", &.{ kernels.ggmlType(ty), kernels.int(J), kernels.boolean(fallback) }));
}

fn i32Of(x: i64) c_int {
    return @truncate(x);
}

fn u32Of(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// Ports `launch_mul_mat_q` (mmq.cuh:1396 @b29c606e2).
fn launchMulMatQ(comptime ty: c.enum_ggml_type, comptime J: comptime_int, comptime fallback: bool, ctx: *common.Context, args: *const Args, stream: cu.cudaStream_t) void {
    const id = common.getDevice();
    const dev = common.info().devices[@intCast(id)];
    const x = dev.cc;
    const nsm = dev.nsm;
    const warp_size = dev.warp_size;

    const config = getConfig(ty, J, fallback, x);
    impl.assert(@rem(config.nthreads, warp_size) == 0, "config.nthreads % warp_size == 0");
    const nwarps = @divTrunc(config.nthreads, warp_size);
    const nbytes_shared: c_int = @intCast(getNbytesShared(config, x));

    const block_dims: kernels.Dim3 = .{ .x = @intCast(warp_size), .y = @intCast(nwarps), .z = 1 };

    kernels.setSharedMemoryLimit(comptime mulMatQKernel(ty, J, false), @intCast(nbytes_shared));
    kernels.setSharedMemoryLimit(comptime mulMatQKernel(ty, J, true), @intCast(nbytes_shared));

    const nty: c_int = i32Of(@divTrunc(args.nrows_x + config.I - 1, config.I));
    const ntx: c_int = i32Of(@divTrunc(args.ncols_max + config.J - 1, config.J));
    const ntzw: c_int = i32Of(args.nchannels_y * args.nsamples_y);
    const block_nums_xy_tiling: kernels.Dim3 = .{ .x = @bitCast(nty), .y = @bitCast(ntx), .z = @bitCast(ntzw) };

    impl.assert(@rem(args.nchannels_y, args.nchannels_x) == 0, "args.nchannels_y % args.nchannels_x == 0");
    impl.assert(@rem(args.nsamples_y, args.nsamples_x) == 0, "args.nsamples_y  % args.nsamples_x  == 0");
    const channel_ratio: c_int = i32Of(@divTrunc(args.nchannels_y, args.nchannels_x));
    const sample_ratio: c_int = i32Of(@divTrunc(args.nsamples_y, args.nsamples_x));

    const blocks_per_ne00_fd = common.initFastdivValues(@bitCast(@divTrunc(args.ncols_x, common.typeTraits(ty).qk)));
    const ntx_fd = common.initFastdivValues(@intCast(ntx));
    const nchannels_y_fd = common.initFastdivValues(@bitCast(args.nchannels_y));
    const nsamples_y_fd = common.initFastdivValues(@bitCast(args.nsamples_y));
    const channel_ratio_fd = common.initFastdivValues(@intCast(channel_ratio));
    const sample_ratio_fd = common.initFastdivValues(@intCast(sample_ratio));

    const k = comptime mulMatQKernel(ty, J, fallback);
    const shmem: usize = @intCast(nbytes_shared);

    if (!getConfig(ty, J, fallback, x).stream_k) {
        kernels.launch(k, .{ .grid = block_nums_xy_tiling, .block = block_dims, .shmem = shmem, .stream = stream }, .{
            args.x,                       args.y,                         args.ids_dst,                  args.expert_bounds,
            args.dst,                     @as(?*f32, null),               args.y_scale,                  blocks_per_ne00_fd,
            i32Of(args.nrows_x),          i32Of(args.ncols_dst),          i32Of(args.stride_row_x),      i32Of(args.ncols_y),
            i32Of(args.nrows_dst),        channel_ratio_fd,               nchannels_y_fd,                i32Of(args.stride_channel_x),
            i32Of(args.stride_channel_y), i32Of(args.stride_channel_dst), sample_ratio_fd,               nsamples_y_fd,
            i32Of(args.stride_sample_x),  i32Of(args.stride_sample_y),    i32Of(args.stride_sample_dst), ntx_fd,
        });
        return;
    }

    // For the stream-k kernel it is possible to run it with tiling by
    // setting the number of CUDA blocks equal to the number of tiles.
    const ntiles_dst: c_int = ntx *% nty *% ntzw;
    const tiles_nwaves: c_int = @divTrunc(ntiles_dst + nsm - 1, nsm);
    const tiles_efficiency_percent: c_int = @divTrunc(100 * ntiles_dst, nsm * tiles_nwaves);
    const block_nums_stream_k: kernels.Dim3 = .{
        .x = @bitCast(if (common.cc.isNvidia(x) and tiles_efficiency_percent >= 90) ntiles_dst else nsm),
    };

    // Assert that variable kbc will not overflow.
    // `int * unsigned int` is an unsigned product, wrapping.
    impl.assert(@as(u32, @bitCast(ntiles_dst)) *% blocks_per_ne00_fd.z < (1 << 30), "ntiles_dst * blocks_per_ne00_fd.z < (1 << 30)");

    const fixup_needed = @as(u32, @bitCast(ntiles_dst)) % block_nums_stream_k.x != 0;

    const pool = ctx.poolFor(id);
    var tmp_fixup = common.PoolAlloc(f32).init(pool);
    defer tmp_fixup.deinit();
    if (fixup_needed) {
        _ = tmp_fixup.alloc(@as(usize, block_nums_stream_k.x) * @as(usize, @intCast(config.J * config.I)));
    }

    const block_nums_fixup: kernels.Dim3 = .{ .x = block_nums_stream_k.x, .y = @intCast(@divTrunc(config.I, warp_size)), .z = 1 };
    const block_dims_fixup: kernels.Dim3 = .{ .x = block_dims.x, .y = block_dims.y / 2, .z = block_dims.z };

    const tmp_ptr: ?*f32 = if (tmp_fixup.ptr) |p| &p[0] else null;
    kernels.launch(k, .{ .grid = block_nums_stream_k, .block = block_dims, .shmem = shmem, .stream = stream }, .{
        args.x,                       args.y,                         args.ids_dst,                  args.expert_bounds,
        args.dst,                     tmp_ptr,                        args.y_scale,                  blocks_per_ne00_fd,
        i32Of(args.nrows_x),          i32Of(args.ncols_dst),          i32Of(args.stride_row_x),      i32Of(args.ncols_y),
        i32Of(args.nrows_dst),        channel_ratio_fd,               nchannels_y_fd,                i32Of(args.stride_channel_x),
        i32Of(args.stride_channel_y), i32Of(args.stride_channel_dst), sample_ratio_fd,               nsamples_y_fd,
        i32Of(args.stride_sample_x),  i32Of(args.stride_sample_y),    i32Of(args.stride_sample_dst), ntx_fd,
    });

    if (!fixup_needed) return;

    common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());
    kernels.launch(comptime fixupKernel(ty, J, fallback), .{ .grid = block_nums_fixup, .block = block_dims_fixup, .stream = stream }, .{
        args.ids_dst,       args.expert_bounds,             args.dst,              tmp_ptr,
        blocks_per_ne00_fd, i32Of(args.nrows_x),            i32Of(args.ncols_dst), i32Of(args.nrows_dst),
        nchannels_y_fd,     i32Of(args.stride_channel_dst), nsamples_y_fd,         i32Of(args.stride_sample_dst),
        ntx_fd,
    });
}

/// Ports `mul_mat_q_switch_J` (mmq.cuh:1478 @b29c606e2).
fn mulMatQSwitchJ(comptime ty: c.enum_ggml_type, comptime fallback: bool, ctx: *common.Context, args: *const Args, stream: cu.cudaStream_t) void {
    const id = common.getDevice();
    const x = common.info().devices[@intCast(id)].cc;
    const smpbo = common.info().devices[@intCast(id)].smpbo;

    var J_best: c_int = 0;
    var ntiles_J_best: c_int = std.math.maxInt(c_int);

    var J: c_int = 8;
    while (J <= 128 and ntiles_J_best > 1) : (J += 8) {
        const config = getConfig(ty, J, fallback, x);
        if (config.type == c.GGML_TYPE_COUNT) continue;
        if (getNbytesShared(config, x) > smpbo) continue;

        const ntiles_x: c_int = i32Of(@divTrunc(args.ncols_opt + config.J - 1, config.J));
        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            ntiles_J_best = ntiles_x;
        }
    }

    switch (J_best) {
        inline 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128 => |j| launchMulMatQ(ty, j, fallback, ctx, args, stream),
        else => {
            impl.logError("J_best=%d\n", .{J_best});
            impl.abort("fatal error");
        },
    }
}

/// Ports `mul_mat_q_case` (mmq.cuh:1561 @b29c606e2).
fn mulMatQCase(comptime ty: c.enum_ggml_type, ctx: *common.Context, args: *const Args, stream: cu.cudaStream_t) void {
    if (@rem(args.nrows_x, 128) == 0) {
        mulMatQSwitchJ(ty, false, ctx, args, stream);
    } else {
        mulMatQSwitchJ(ty, true, ctx, args, stream);
    }
}

/// Ports `ggml_cuda_mul_mat_q_switch_type` (mmq.cu:8 @b29c606e2).
fn mulMatQSwitchType(ctx: *common.Context, args: *const Args, stream: cu.cudaStream_t) void {
    switch (args.type_x) {
        inline c.GGML_TYPE_Q1_0,
        c.GGML_TYPE_Q2_0,
        c.GGML_TYPE_Q4_0,
        c.GGML_TYPE_Q4_1,
        c.GGML_TYPE_Q5_0,
        c.GGML_TYPE_Q5_1,
        c.GGML_TYPE_Q8_0,
        c.GGML_TYPE_Q2_K,
        c.GGML_TYPE_Q3_K,
        c.GGML_TYPE_Q4_K,
        c.GGML_TYPE_Q5_K,
        c.GGML_TYPE_Q6_K,
        c.GGML_TYPE_IQ1_S,
        c.GGML_TYPE_IQ2_XXS,
        c.GGML_TYPE_IQ2_XS,
        c.GGML_TYPE_IQ2_S,
        c.GGML_TYPE_IQ3_XXS,
        c.GGML_TYPE_IQ3_S,
        c.GGML_TYPE_IQ4_XS,
        c.GGML_TYPE_IQ4_NL,
        c.GGML_TYPE_MXFP4,
        c.GGML_TYPE_NVFP4,
        => |t| mulMatQCase(t, ctx, args, stream),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_cuda_mul_mat_q` (mmq.cu:85 @b29c606e2).
pub fn mulMatQ(ctx: *common.Context, src0: *const Tensor, src1: *const Tensor, ids: ?*const Tensor, dst: *Tensor) void {
    impl.assert(src1.type == c.GGML_TYPE_F32, "        src1->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "        dst->type  == GGML_TYPE_F32");
    impl.assert(ids == null or ids.?.type == c.GGML_TYPE_I32, "!ids || ids->type  == GGML_TYPE_I32");

    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];
    const ne10 = src1.ne[0];
    const ne11 = src1.ne[1];
    const ne12 = src1.ne[2];
    const ne13 = src1.ne[3];
    const nb11 = src1.nb[1];
    const nb12 = src1.nb[2];
    const ne1 = dst.ne[1];
    const nb1 = dst.nb[1];
    const nb2 = dst.nb[2];

    const stream = ctx.stream();
    const x = common.info().devices[@intCast(common.getDevice())].cc;

    const ts_src0 = c.ggml_type_size(src0.type);
    const ts_src1 = c.ggml_type_size(src1.type);
    const ts_dst = c.ggml_type_size(dst.type);

    impl.assert(src0.nb[0] == ts_src0, "        nb00       == ts_src0");
    impl.assert(src1.nb[0] == ts_src1, "        nb10       == ts_src1");
    impl.assert(dst.nb[0] == ts_dst, "        nb0        == ts_dst");
    impl.assert(ids == null or ids.?.nb[0] == c.ggml_type_size(ids.?.type), "!ids || ids->nb[0] == ggml_type_size(ids->type)");

    const src0_d: ?*const anyopaque = src0.data;
    const src1_d: ?*const f32 = @ptrCast(@alignCast(src1.data));
    const dst_d: ?*f32 = @ptrCast(@alignCast(dst.data));

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (c.ggml_backend_buffer_get_usage(src0.buffer) == c.GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_data = c.ggml_nbytes(src0);
        const size_alloc = c.ggml_backend_buffer_get_alloc_size(src0.buffer, src0);
        if (size_alloc > size_data) {
            impl.assert(c.ggml_is_contiguously_allocated(src0), "ggml_is_contiguously_allocated(src0)");
            impl.assert(src0.view_src == null, "!src0->view_src");
            common.cudaCheck(cu.cudaMemsetAsync(@as([*]u8, @ptrCast(src0.data.?)) + size_data, 0, size_alloc - size_data, stream), "cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream)", @src());
        }
    }

    const ne10_padded: i64 = @intCast(impl.pad(@intCast(ne10), common.matrix_row_padding));

    const s01: i64 = @intCast(src0.nb[1] / ts_src0);
    const s1: i64 = @intCast(dst.nb[1] / ts_dst);
    const s02: i64 = @intCast(src0.nb[2] / ts_src0);
    const s2: i64 = @intCast(dst.nb[2] / ts_dst);
    const s03: i64 = @intCast(src0.nb[3] / ts_src0);
    const s3: i64 = @intCast(dst.nb[3] / ts_dst);

    const fallback = @rem(ne01, 128) != 0;

    const use_native_fp4 = common.blackwellMmaAvailable(x) and (src0.type == c.GGML_TYPE_MXFP4 or src0.type == c.GGML_TYPE_NVFP4);
    const y_block_size: i64 = if (use_native_fp4) sizeof_block_fp4_mmq else sizeof_block_q8_1_mmq;
    const y_values_per_block: i64 = if (use_native_fp4) qk_fp4_mmq else qk8_1_mmq;

    if (ids == null) {
        const nbytes_src1_q8_1: usize = @intCast(@divTrunc(ne13 * ne12 * ne11 * ne10_padded * y_block_size, y_values_per_block) +
            @as(i64, getJMax(src0.type, fallback, x, ne11)) * sizeof_block_q8_1_mmq);
        var src1_q8_1 = common.PoolAlloc(u8).initAlloc(ctx.pool(), nbytes_src1_q8_1);
        defer src1_q8_1.deinit();
        var src1_scale = common.PoolAlloc(f32).init(ctx.pool());
        defer src1_scale.deinit();
        if (src0.type == c.GGML_TYPE_NVFP4 and use_native_fp4) {
            _ = src1_scale.alloc(@intCast(ne13 * ne12 * ne11));
        }

        {
            const s11: i64 = @intCast(src1.nb[1] / ts_src1);
            const s12: i64 = @intCast(src1.nb[2] / ts_src1);
            const s13: i64 = @intCast(src1.nb[3] / ts_src1);
            if (use_native_fp4) {
                const align_float8 = 32;
                const use_aligned_float8 = common.isAligned(src1, align_float8);
                quantize.quantizeMmqFp4(src1_d, null, src1_q8_1.get(), firstOf(f32, src1_scale.ptr), src0.type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
            } else {
                quantize.quantizeMmqQ8_1(src1_d, null, src1_q8_1.get(), src0.type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
            }
            common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());
        }

        // Stride depends on quantization format
        const s12: i64 = if (use_native_fp4)
            @divTrunc(ne11 * ne10_padded * sizeof_block_fp4_mmq, qk_fp4_mmq * @sizeOf(c_int))
        else
            @divTrunc(ne11 * ne10_padded * sizeof_block_q8_1, c.QK8_1 * @sizeOf(c_int));
        const s13 = ne12 * s12;

        const args: Args = .{
            .x = src0_d,
            .type_x = src0.type,
            .y = src1_q8_1.get(),
            .ids_dst = null,
            .expert_bounds = null,
            .dst = dst_d,
            .y_scale = if (src0.type == c.GGML_TYPE_NVFP4 and use_native_fp4) firstOf(f32, src1_scale.ptr) else null,
            .ncols_x = ne00,
            .nrows_x = ne01,
            .ncols_dst = ne1,
            .stride_row_x = s01,
            .ncols_y = ne11,
            .nrows_dst = s1,
            .nchannels_x = ne02,
            .nchannels_y = ne12,
            .stride_channel_x = s02,
            .stride_channel_y = s12,
            .stride_channel_dst = s2,
            .nsamples_x = ne03,
            .nsamples_y = ne13,
            .stride_sample_x = s03,
            .stride_sample_y = s13,
            .stride_sample_dst = s3,
            .ncols_max = ne1,
            .ncols_opt = ne1,
        };
        mulMatQSwitchType(ctx, &args, stream);
        return;
    }

    const id_t = ids.?;
    impl.assert(ne13 == 1, "ne13 == 1");
    impl.assert(nb12 % nb11 == 0, "nb12 % nb11 == 0");
    impl.assert(nb2 % nb1 == 0, "nb2  % nb1  == 0");

    const n_expert_used = id_t.ne[0];
    const ne_get_rows = ne12 * n_expert_used;
    impl.assert(ne1 == n_expert_used, "ne1 == n_expert_used");

    var ids_src1 = common.PoolAlloc(i32).initAlloc(ctx.pool(), @intCast(ne_get_rows));
    defer ids_src1.deinit();
    var ids_dst = common.PoolAlloc(i32).initAlloc(ctx.pool(), @intCast(ne_get_rows));
    defer ids_dst.deinit();
    var expert_bounds = common.PoolAlloc(i32).initAlloc(ctx.pool(), @intCast(ne02 + 1));
    defer expert_bounds.deinit();

    // gate/up activations are broadcast across experts (ne11 == 1):
    // quantize each token once and scatter to its slots. ids_src1 then
    // holds the inverse map (token slot -> compact row).
    const dedup_bcast = ne11 == 1 and n_expert_used > 1;

    {
        impl.assert(id_t.nb[0] == c.ggml_element_size(id_t), "ids->nb[0] == ggml_element_size(ids)");
        const si1: c_int = @intCast(id_t.nb[1] / c.ggml_element_size(id_t));
        const sis1: c_int = @intCast(nb12 / nb11);

        mmid.launchMmIdsHelperDispatch(@ptrCast(@alignCast(id_t.data)), firstOf(i32, ids_src1.ptr), firstOf(i32, ids_dst.ptr), firstOf(i32, expert_bounds.ptr), i32Of(ne02), i32Of(ne12), i32Of(n_expert_used), i32Of(ne11), si1, sis1, dedup_bcast, stream);
        common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());
    }

    const nbytes_src1_q8_1: usize = @intCast(@divTrunc(ne12 * n_expert_used * ne10_padded * y_block_size, y_values_per_block) +
        @as(i64, getJMax(src0.type, fallback, x, ne11)) * sizeof_block_q8_1_mmq);
    var src1_q8_1 = common.PoolAlloc(u8).initAlloc(ctx.pool(), nbytes_src1_q8_1);
    defer src1_q8_1.deinit();
    var src1_scale = common.PoolAlloc(f32).init(ctx.pool());
    defer src1_scale.deinit();
    if (src0.type == c.GGML_TYPE_NVFP4 and use_native_fp4) {
        _ = src1_scale.alloc(@intCast(ne12 * n_expert_used));
    }

    const ne11_flat = ne12 * n_expert_used;
    const ne12_flat: i64 = 1;
    const ne13_flat: i64 = 1;

    {
        const s11: i64 = @intCast(src1.nb[1] / ts_src1);
        const s12: i64 = @intCast(src1.nb[2] / ts_src1);
        const s13: i64 = @intCast(src1.nb[3] / ts_src1);

        if (use_native_fp4) {
            const align_float8 = 32;
            const use_aligned_float8 = common.isAligned(src1, align_float8);
            if (dedup_bcast) {
                quantize.quantizeScatterMmqFp4(src1_d, firstOf(i32, ids_src1.ptr), src1_q8_1.get(), firstOf(f32, src1_scale.ptr), src0.type, use_aligned_float8, ne10, s12, ne10_padded, ne12, ne11_flat, i32Of(n_expert_used), stream);
            } else {
                quantize.quantizeMmqFp4(src1_d, firstOf(i32, ids_src1.ptr), src1_q8_1.get(), firstOf(f32, src1_scale.ptr), src0.type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize.quantizeScatterMmqQ8_1(src1_d, firstOf(i32, ids_src1.ptr), src1_q8_1.get(), src0.type, ne10, s12, ne10_padded, ne12, ne11_flat, i32Of(n_expert_used), stream);
        } else {
            quantize.quantizeMmqQ8_1(src1_d, firstOf(i32, ids_src1.ptr), src1_q8_1.get(), src0.type, ne10, s11, s12, s13, ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());
    }

    const s12: i64 = if (use_native_fp4)
        @divTrunc(ne11 * ne10_padded * sizeof_block_fp4_mmq, qk_fp4_mmq * @sizeOf(c_int))
    else
        @divTrunc(ne11 * ne10_padded * sizeof_block_q8_1, c.QK8_1 * @sizeOf(c_int));
    const s13 = ne12 * s12;

    // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
    // On RDNA3 and RDNA4 it is faster to pick the tile size against this
    // value instead of ne12: the C's `GGML_CUDA_CC_IS_RDNA3_0(cc) ||
    // GGML_CUDA_CC_IS_RDNA4(cc)` arm, which a CUDA device never takes.
    const ncols_opt: i64 = ne12;

    // ne02 is used instead of ne12 because the number of y channels
    // determines the z dimension of the CUDA grid.
    const args: Args = .{
        .x = src0_d,
        .type_x = src0.type,
        .y = src1_q8_1.get(),
        .ids_dst = firstOf(i32, ids_dst.ptr),
        .expert_bounds = firstOf(i32, expert_bounds.ptr),
        .dst = dst_d,
        .y_scale = firstOf(f32, src1_scale.ptr),
        .ncols_x = ne00,
        .nrows_x = ne01,
        .ncols_dst = ne_get_rows,
        .stride_row_x = s01,
        .ncols_y = ne_get_rows,
        .nrows_dst = s1,
        .nchannels_x = ne02,
        .nchannels_y = ne02,
        .stride_channel_x = s02,
        .stride_channel_y = s12,
        .stride_channel_dst = s2,
        .nsamples_x = ne03,
        .nsamples_y = ne13,
        .stride_sample_x = s03,
        .stride_sample_y = s13,
        .stride_sample_dst = s3,
        .ncols_max = ne12,
        .ncols_opt = ncols_opt,
    };

    mulMatQSwitchType(ctx, &args, stream);
}

fn firstOf(comptime T: type, p: ?[*]T) ?*T {
    return if (p) |x| &x[0] else null;
}

/// Ports `ggml_cuda_should_use_mmq` (mmq.cu:266 @b29c606e2), NVIDIA arms.
pub fn shouldUseMmq(ty: c.enum_ggml_type, x: c_int, ne11: i64, n_experts: i64) bool {
    const mmq_supported = switch (ty) {
        c.GGML_TYPE_Q1_0,
        c.GGML_TYPE_Q2_0,
        c.GGML_TYPE_Q4_0,
        c.GGML_TYPE_Q4_1,
        c.GGML_TYPE_Q5_0,
        c.GGML_TYPE_Q5_1,
        c.GGML_TYPE_Q8_0,
        c.GGML_TYPE_Q2_K,
        c.GGML_TYPE_Q3_K,
        c.GGML_TYPE_Q4_K,
        c.GGML_TYPE_Q5_K,
        c.GGML_TYPE_Q6_K,
        c.GGML_TYPE_IQ1_S,
        c.GGML_TYPE_IQ2_XXS,
        c.GGML_TYPE_IQ2_XS,
        c.GGML_TYPE_IQ2_S,
        c.GGML_TYPE_IQ3_XXS,
        c.GGML_TYPE_IQ3_S,
        c.GGML_TYPE_IQ4_XS,
        c.GGML_TYPE_IQ4_NL,
        c.GGML_TYPE_MXFP4,
        c.GGML_TYPE_NVFP4,
        => true,
        else => false,
    };

    if (!mmq_supported) return false;

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back
    // to BLAS otherwise.
    {
        const id = common.getDevice();
        const smpbo = common.info().devices[@intCast(id)].smpbo;
        if (smpbo < 48 * 1024) return false;
    }

    if (common.turingMmaAvailable(x)) return true;

    if (common.highestCompiledArch(x) < common.cc.dp4a) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return x >= common.cc.pascal and n_experts > 0;
    }

    if (common.cc.isNvidia(x)) {
        return !common.fp16MmaHardwareAvailable(x) or ne11 < dp4a_max_batch_size;
    }

    // The AMD and Moore Threads arms are not reachable on a CUDA device.
    return ne11 < dp4a_max_batch_size;
}
