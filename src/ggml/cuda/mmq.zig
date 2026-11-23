//! MMQ: quantized matrix multiplication with integer tensor cores (or dp4a),
//! `src1` quantized to `q8_1` (or FP4 on Blackwell) first.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/mmq.cu` and the host half of
//! `llama.cpp/ggml/src/ggml-cuda/mmq.cuh` at v1.0.0 (`b1b770d90`). The
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
//!
//! # The `src1` precision
//!
//! Every MMQ kernel and launcher takes `prec_src1`, a
//! `ggml_prec`: `GGML_PREC_Q8` quantizes `src1` to `q8_1` as before, and
//! `GGML_PREC_Q4` -- the FP4 types on Blackwell only -- keeps it at FP4 for
//! the native W4A4 instructions. Which one a multiplication takes is
//! `getPrecSrc1`'s answer: `GGML_CUDA_MMQ_PREC` overrides the precision the
//! graph requests (`ggml_prec_get_src(dst, 1)`, or the fused
//! matmuls' `prec_src1` in the fusion args). On Blackwell an FP4 type at `Q8` takes the Ampere
//! configuration, so the native path is no longer implied by the type.

const std = @import("std");
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `MMQ_DP4A_MAX_BATCH_SIZE` (mmq.cuh:10 @b1b770d90).
const dp4a_max_batch_size = 64;

/// Ports `MMQ_TILE_NE_K` (mmq.cuh:119 @b1b770d90).
const tile_ne_k = 32;

/// Ports `QK8_1_MMQ` (mmq.cuh:28 @b1b770d90).
pub const qk8_1_mmq = 4 * ggml.impl.ggml_c.QK8_1;

/// Ports `QK_FP4_MMQ` (mmq.cuh:29 @b1b770d90).
pub const qk_fp4_mmq = 2 * qk8_1_mmq;

/// `sizeof(block_q8_1)`: two halves and `QK8_1` bytes (ggml-common.h:269 @b1b770d90).
pub const sizeof_block_q8_1 = 2 * 2 + ggml.impl.ggml_c.QK8_1;

/// The size of `block_q8_1_mmq` (mmq.cuh:60 @b1b770d90): `QK8_1_MMQ + 4*sizeof(half2)`.
pub const sizeof_block_q8_1_mmq = qk8_1_mmq + 4 * 4;

/// The size of `block_fp4_mmq` (mmq.cuh:62 @b1b770d90), equal to `block_q8_1_mmq`'s.
pub const sizeof_block_fp4_mmq = sizeof_block_q8_1_mmq;

/// Ports `mmq_q8_1_ds_layout` (mmq.cuh:22 @b1b770d90).
pub const Q8_1DsLayout = enum(c_int) { d4 = 0, ds4 = 1, d2s6 = 2 };

/// Ports `mmq_get_q8_1_ds_layout` (mmq.cuh:64 @b1b770d90).
pub fn getQ8_1DsLayout(type_x: ggml.impl.ggml_c.enum_ggml_type) Q8_1DsLayout {
    return switch (type_x) {
        ggml.impl.ggml_c.GGML_TYPE_Q1_0, ggml.impl.ggml_c.GGML_TYPE_Q2_0 => .d4,
        ggml.impl.ggml_c.GGML_TYPE_Q4_0, ggml.impl.ggml_c.GGML_TYPE_Q4_1 => .ds4,
        ggml.impl.ggml_c.GGML_TYPE_Q5_0 => .d4,
        ggml.impl.ggml_c.GGML_TYPE_Q5_1 => .ds4,
        ggml.impl.ggml_c.GGML_TYPE_Q8_0 => .d4,
        ggml.impl.ggml_c.GGML_TYPE_MXFP4 => .d4,
        ggml.impl.ggml_c.GGML_TYPE_NVFP4 => .d4,
        ggml.impl.ggml_c.GGML_TYPE_Q2_K => .d2s6,
        ggml.impl.ggml_c.GGML_TYPE_Q3_K => .d4,
        ggml.impl.ggml_c.GGML_TYPE_Q4_K, ggml.impl.ggml_c.GGML_TYPE_Q5_K => .ds4,
        ggml.impl.ggml_c.GGML_TYPE_Q6_K, ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS, ggml.impl.ggml_c.GGML_TYPE_IQ2_XS, ggml.impl.ggml_c.GGML_TYPE_IQ2_S, ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS, ggml.impl.ggml_c.GGML_TYPE_IQ3_S => .d4,
        ggml.impl.ggml_c.GGML_TYPE_IQ1_S => .ds4,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_XS, ggml.impl.ggml_c.GGML_TYPE_IQ4_NL => .d4,
        else => ggml.impl.abort("fatal error"),
    };
}

/// Ports `ggml_cuda_mmq_sram_layout` (mmq.cuh:125 @b1b770d90).
pub const SramLayout = enum { q8_0, q8_1, q2_k, q3_k, q6_k, fp4, nvfp4 };

/// Ports `ggml_cuda_mmq_get_sram_stride` (mmq.cuh:135 @b1b770d90): the
/// padded SRAM tile length, in 32-bit elements.
fn getSramStride(layout: SramLayout) c_int {
    return switch (layout) {
        .q8_0 => 2 * tile_ne_k + 2 * tile_ne_k / ggml.cuda.common.q.QI8_0 + 4,
        .q8_1 => 2 * tile_ne_k + 2 * tile_ne_k / ggml.cuda.common.q.QI8_1 + 4,
        .q2_k => 2 * tile_ne_k + tile_ne_k + 4,
        .q3_k => 2 * tile_ne_k + tile_ne_k / 2 + 4,
        .q6_k => 2 * tile_ne_k + tile_ne_k / ggml.cuda.common.q.QI6_K + tile_ne_k / 8 + 7,
        .fp4 => 2 * tile_ne_k + 8 + 4,
        .nvfp4 => 2 * tile_ne_k + tile_ne_k / 2 + 4,
    };
}

/// Ports `ggml_cuda_mmq_config` (mmq.cuh:168 @b1b770d90).
pub const Config = struct {
    type: ggml.impl.ggml_c.enum_ggml_type,
    nthreads: c_int,
    occupancy: c_int,
    I: c_int,
    J: c_int,
    sram_layout: SramLayout,
    K_vram: c_int,
    stream_k: bool,
    fallback: bool,

    /// Ports `ggml_cuda_mmq_config`'s `use_mma_data_layout` (mmq.cuh:192 @b1b770d90).
    fn useMmaDataLayout(self: Config, x: c_int) bool {
        _ = self;
        return ggml.cuda.common.turingMmaAvailable(x);
    }
};

fn lookup(t: *const ggml.cuda.mmq_config_table.Table, ty: ggml.impl.ggml_c.enum_ggml_type, J: c_int, fallback: bool) Config {
    for (t.cases) |e| {
        if (e.type == ty and e.J == J and e.fallback == fallback) return e;
    }
    if (t.default) |d| return d;
    return lookup(t.fallthrough.?, ty, J, fallback);
}

/// Ports the host `ggml_cuda_mmq_get_config` (mmq.cuh:233 @b1b770d90),
/// NVIDIA arms. The AMD arms -- GCN, CDNA, RDNA2/3/3.5/4 and RDNA4_M -- are taken only
/// for an AMD compute capability, which a CUDA device never has.
/// `prec_src1` has no default: every caller passes it (#30168).
pub fn getConfig(ty: ggml.impl.ggml_c.enum_ggml_type, J: c_int, fallback: bool, x: c_int, prec_src1: ggml.impl.ggml_c.enum_ggml_prec) Config {
    if (ggml.cuda.common.blackwellMmaAvailable(x)) {
        // only src1 at Q4 uses the native FP4 config, higher precisions keep src1 at Q8_1
        if (prec_src1 != ggml.impl.ggml_c.GGML_PREC_Q4 and (ty == ggml.impl.ggml_c.GGML_TYPE_NVFP4 or ty == ggml.impl.ggml_c.GGML_TYPE_MXFP4)) {
            return lookup(&ggml.cuda.mmq_config_table.ampere, ty, J, fallback);
        }
        return lookup(&ggml.cuda.mmq_config_table.blackwell, ty, J, fallback);
    }
    if (ggml.cuda.common.highestCompiledArch(x) >= ggml.cuda.common.cc.volta) return lookup(&ggml.cuda.mmq_config_table.ampere, ty, J, fallback);
    if (ggml.cuda.common.highestCompiledArch(x) >= ggml.cuda.common.cc.dp4a) return lookup(&ggml.cuda.mmq_config_table.pascal_dp4a, ty, J, fallback);
    return lookup(&ggml.cuda.mmq_config_table.pascal_older, ty, J, fallback);
}

/// Ports `ggml_cuda_mmq_needs_fallback` (mmq.cuh:380 @b1b770d90): whether the rows of
/// `x` need the bounds-checked kernels.
pub fn needsFallback(nrows_x: i64) bool {
    return @rem(nrows_x, 128) != 0;
}

/// Ports `tile_x_sizes` (mmq.cuh:106 @b1b770d90).
const TileXSizes = struct { qs: c_int, dm: c_int, sc: c_int };

/// Ports `mmq_get_dp4a_tile_x_sizes` (mmq.cuh:399 @b1b770d90) with the
/// `MMQ_DP4A_TXS_*` macros, from `MMQ_DP4A_TXS_Q4_0` (mmq.cuh:388 @b1b770d90).
fn getDp4aTileXSizes(ty: ggml.impl.ggml_c.enum_ggml_type, I: c_int) TileXSizes {
    const K = tile_ne_k;
    const q4_0: TileXSizes = .{ .qs = I * K + I, .dm = @divTrunc(I * K, ggml.cuda.common.q.QI4_0) + @divTrunc(I, ggml.cuda.common.q.QI4_0), .sc = 0 };
    const q4_1: TileXSizes = .{ .qs = I * K + I, .dm = @divTrunc(I * K, ggml.cuda.common.q.QI4_1) + @divTrunc(I, ggml.cuda.common.q.QI4_1), .sc = 0 };
    const q8_0: TileXSizes = .{ .qs = I * K * 2 + I, .dm = @divTrunc(I * K * 2, ggml.cuda.common.q.QI8_0) + @divTrunc(I, ggml.cuda.common.q.QI8_0 / 2), .sc = 0 };
    const q8_0_16: TileXSizes = .{ .qs = I * K * 2 + I, .dm = @divTrunc(I * K * 4, ggml.cuda.common.q.QI8_0) + @divTrunc(I, ggml.cuda.common.q.QI8_0 / 4), .sc = 0 };
    const q8_1: TileXSizes = .{ .qs = I * K * 2 + I, .dm = @divTrunc(I * K * 2, ggml.cuda.common.q.QI8_1) + @divTrunc(I, ggml.cuda.common.q.QI8_1 / 2), .sc = 0 };
    const q2_k: TileXSizes = .{ .qs = I * K * 2 + I, .dm = I * K + I, .sc = 0 };
    const q3_k: TileXSizes = .{ .qs = I * K * 2 + I, .dm = I, .sc = @divTrunc(I * K, 8) + @divTrunc(I, 8) };
    const q4_k: TileXSizes = .{ .qs = I * K + I, .dm = @divTrunc(I * K, ggml.cuda.common.q.QI4_K), .sc = @divTrunc(I * K, 8) + @divTrunc(I, 8) };
    const q5_k: TileXSizes = .{ .qs = I * K * 2 + I, .dm = @divTrunc(I * K, ggml.cuda.common.q.QI5_K) + @divTrunc(I, ggml.cuda.common.q.QI5_K), .sc = @divTrunc(I * K, 8) + @divTrunc(I, 8) };
    const q6_k: TileXSizes = .{ .qs = I * K * 2 + I, .dm = @divTrunc(I * K, ggml.cuda.common.q.QI6_K) + @divTrunc(I, ggml.cuda.common.q.QI6_K), .sc = @divTrunc(I * K, 8) + @divTrunc(I, 8) };
    return switch (ty) {
        ggml.impl.ggml_c.GGML_TYPE_Q1_0, ggml.impl.ggml_c.GGML_TYPE_Q2_0 => q8_0,
        ggml.impl.ggml_c.GGML_TYPE_Q4_0 => q4_0,
        ggml.impl.ggml_c.GGML_TYPE_Q4_1 => q4_1,
        ggml.impl.ggml_c.GGML_TYPE_Q5_0 => q8_0,
        ggml.impl.ggml_c.GGML_TYPE_Q5_1 => q8_1,
        ggml.impl.ggml_c.GGML_TYPE_Q8_0 => q8_0,
        ggml.impl.ggml_c.GGML_TYPE_MXFP4 => q8_1,
        ggml.impl.ggml_c.GGML_TYPE_NVFP4 => q8_0_16,
        ggml.impl.ggml_c.GGML_TYPE_Q2_K => q2_k,
        ggml.impl.ggml_c.GGML_TYPE_Q3_K => q3_k,
        ggml.impl.ggml_c.GGML_TYPE_Q4_K => q4_k,
        ggml.impl.ggml_c.GGML_TYPE_Q5_K => q5_k,
        ggml.impl.ggml_c.GGML_TYPE_Q6_K => q6_k,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS => q8_0,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XS, ggml.impl.ggml_c.GGML_TYPE_IQ2_S => q8_0_16,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS, ggml.impl.ggml_c.GGML_TYPE_IQ3_S, ggml.impl.ggml_c.GGML_TYPE_IQ1_S, ggml.impl.ggml_c.GGML_TYPE_IQ4_XS, ggml.impl.ggml_c.GGML_TYPE_IQ4_NL => q8_0,
        else => .{ .qs = 0, .dm = 0, .sc = 0 },
    };
}

/// Ports `ggml_cuda_mmq_get_nbytes_shared_x` (mmq.cuh:428 @b1b770d90).
fn getNbytesSharedX(config: Config, x: c_int) c_int {
    if (config.useMmaDataLayout(x)) {
        return config.I * getSramStride(config.sram_layout) * 4;
    }
    const txs = getDp4aTileXSizes(config.type, config.I);
    return (txs.qs + txs.dm + txs.sc) * 4;
}

/// Ports `mmq_get_nbytes_shared` (mmq.cuh:1457 @b1b770d90).
fn getNbytesShared(config: Config, x: c_int) usize {
    const nbs_ids: usize = @as(usize, @intCast(config.J)) * @sizeOf(c_int);
    const nbs_x: usize = @intCast(getNbytesSharedX(config, x));
    const nbs_y: usize = @as(usize, @intCast(config.J)) * sizeof_block_q8_1_mmq;
    return nbs_ids + nbs_x + stdlib.num.pad(usize, nbs_y, @as(usize, @intCast(config.nthreads)) * @sizeOf(c_int));
}

/// Ports `ggml_cuda_mmq_fusion_supported` (mmq.cuh:439 @b1b770d90): the weight types with a fused gate/up/GLU path.
pub fn fusionSupported(ty: ggml.impl.ggml_c.enum_ggml_type) bool {
    return ty == ggml.impl.ggml_c.GGML_TYPE_Q4_K or ty == ggml.impl.ggml_c.GGML_TYPE_Q5_K or ty == ggml.impl.ggml_c.GGML_TYPE_Q6_K or ty == ggml.impl.ggml_c.GGML_TYPE_Q8_0 or ty == ggml.impl.ggml_c.GGML_TYPE_IQ4_XS or
        ty == ggml.impl.ggml_c.GGML_TYPE_NVFP4;
}

/// Ports `mmq_args` (mmq.cuh:1446 @b1b770d90). It carries the tile specs instead of the expert
/// bounds, and the number of column tiles, `ntj`, instead of the widest
/// column count.
pub const Args = struct {
    x: ?*const anyopaque,
    type_x: ggml.impl.ggml_c.enum_ggml_type,
    y: ?*const anyopaque,
    ids_dst: ?*const i32,
    tile_spec: ?*const ggml.cuda.mmid.TileSpec,
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
    ntj: i64,
    /// Tile width in ne11 (dense) or ne12 (MoE) direction to use for
    /// optimal performance. `ggml_cuda_mul_mat_q` chooses it,
    /// to size `src1`'s padding.
    J_best: c_int,
    /// `gate != null` selects the fused gate/up/GLU path.
    fusion: ggml.cuda.common.MmFusionArgsDevice = .{},
};

/// The `.cu` stem of a type's template instances.
fn instanceTu(comptime ty: ggml.impl.ggml_c.enum_ggml_type) []const u8 {
    return switch (ty) {
        ggml.impl.ggml_c.GGML_TYPE_Q1_0 => "mmq-instance-q1_0",
        ggml.impl.ggml_c.GGML_TYPE_Q2_0 => "mmq-instance-q2_0",
        ggml.impl.ggml_c.GGML_TYPE_Q4_0 => "mmq-instance-q4_0",
        ggml.impl.ggml_c.GGML_TYPE_Q4_1 => "mmq-instance-q4_1",
        ggml.impl.ggml_c.GGML_TYPE_Q5_0 => "mmq-instance-q5_0",
        ggml.impl.ggml_c.GGML_TYPE_Q5_1 => "mmq-instance-q5_1",
        ggml.impl.ggml_c.GGML_TYPE_Q8_0 => "mmq-instance-q8_0",
        ggml.impl.ggml_c.GGML_TYPE_Q2_K => "mmq-instance-q2_k",
        ggml.impl.ggml_c.GGML_TYPE_Q3_K => "mmq-instance-q3_k",
        ggml.impl.ggml_c.GGML_TYPE_Q4_K => "mmq-instance-q4_k",
        ggml.impl.ggml_c.GGML_TYPE_Q5_K => "mmq-instance-q5_k",
        ggml.impl.ggml_c.GGML_TYPE_Q6_K => "mmq-instance-q6_k",
        ggml.impl.ggml_c.GGML_TYPE_IQ1_S => "mmq-instance-iq1_s",
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS => "mmq-instance-iq2_xxs",
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XS => "mmq-instance-iq2_xs",
        ggml.impl.ggml_c.GGML_TYPE_IQ2_S => "mmq-instance-iq2_s",
        ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS => "mmq-instance-iq3_xxs",
        ggml.impl.ggml_c.GGML_TYPE_IQ3_S => "mmq-instance-iq3_s",
        ggml.impl.ggml_c.GGML_TYPE_IQ4_XS => "mmq-instance-iq4_xs",
        ggml.impl.ggml_c.GGML_TYPE_IQ4_NL => "mmq-instance-iq4_nl",
        ggml.impl.ggml_c.GGML_TYPE_MXFP4 => "mmq-instance-mxfp4",
        ggml.impl.ggml_c.GGML_TYPE_NVFP4 => "mmq-instance-nvfp4",
        else => @compileError("no MMQ instance for this type"),
    };
}

fn mulMatQKernel(comptime ty: ggml.impl.ggml_c.enum_ggml_type, comptime J: comptime_int, comptime fallback: bool, comptime prec_src1: ggml.impl.ggml_c.enum_ggml_prec, comptime has_fusion: bool) ggml.cuda.kernels.Kernel {
    return ggml.cuda.kernels.get(instanceTu(ty), ggml.cuda.kernels.tmpl("mul_mat_q", &.{ ggml.cuda.kernels.ggmlType(ty), ggml.cuda.kernels.int(J), ggml.cuda.kernels.boolean(fallback), ggml.cuda.kernels.ggmlPrec(prec_src1), ggml.cuda.kernels.boolean(has_fusion) }));
}

fn fixupKernel(comptime ty: ggml.impl.ggml_c.enum_ggml_type, comptime J: comptime_int, comptime fallback: bool, comptime prec_src1: ggml.impl.ggml_c.enum_ggml_prec) ggml.cuda.kernels.Kernel {
    return ggml.cuda.kernels.get(instanceTu(ty), ggml.cuda.kernels.tmpl("mul_mat_q_stream_k_fixup", &.{ ggml.cuda.kernels.ggmlType(ty), ggml.cuda.kernels.int(J), ggml.cuda.kernels.boolean(fallback), ggml.cuda.kernels.ggmlPrec(prec_src1) }));
}

fn i32Of(x: i64) c_int {
    return @truncate(x);
}

fn u32Of(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// Ports `launch_mul_mat_q` (mmq.cuh:1465 @b1b770d90). With `has_fusion`
/// the kernel always tiles: it needs the whole K range of a tile in one
/// block, and writes the GLU of `I/2` up rows and `I/2` gate rows per tile.
/// Column tiles are innermost, grid x, so that the concurrently
/// running blocks share the same tile of `x` and read it from L2.
fn launchMulMatQ(comptime ty: ggml.impl.ggml_c.enum_ggml_type, comptime J: comptime_int, comptime fallback: bool, comptime prec_src1: ggml.impl.ggml_c.enum_ggml_prec, comptime has_fusion: bool, ctx: *ggml.cuda.common.Context, args: *const Args, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const id = ggml.cuda.common.getDevice();
    const dev = ggml.cuda.common.info().devices[@intCast(id)];
    const x = dev.cc;
    const nsm = dev.nsm;
    const warp_size = dev.warp_size;

    const config = getConfig(ty, J, fallback, x, prec_src1);
    ggml.impl.assert(@rem(config.nthreads, warp_size) == 0, "config.nthreads % warp_size == 0");
    const nwarps = @divTrunc(config.nthreads, warp_size);
    const nbytes_shared: c_int = @intCast(getNbytesShared(config, x));

    const block_dims: ggml.cuda.kernels.Dim3 = .{ .x = @intCast(warp_size), .y = @intCast(nwarps), .z = 1 };

    ggml.cuda.kernels.setSharedMemoryLimit(comptime mulMatQKernel(ty, J, false, prec_src1, has_fusion), @intCast(nbytes_shared));
    if (!has_fusion) {
        ggml.cuda.kernels.setSharedMemoryLimit(comptime mulMatQKernel(ty, J, true, prec_src1, false), @intCast(nbytes_shared));
    }

    const I_dst: c_int = if (has_fusion) @divTrunc(config.I, 2) else config.I; // Rows of dst per tile, see mul_mat_q.
    const nti: c_int = i32Of(@divTrunc(args.nrows_x + I_dst - 1, I_dst));
    const ntj: c_int = i32Of(args.ntj);
    const ntzw: c_int = i32Of(args.nchannels_y * args.nsamples_y);
    // Column tiles innermost so that the concurrently running blocks share the same tile of x and read it from L2.
    const block_nums_xy_tiling: ggml.cuda.kernels.Dim3 = .{ .x = @bitCast(ntj), .y = @bitCast(nti), .z = @bitCast(ntzw) };

    ggml.impl.assert(args.ids_dst != null or @rem(args.nchannels_y, args.nchannels_x) == 0, "args.ids_dst || args.nchannels_y % args.nchannels_x == 0");
    ggml.impl.assert(@rem(args.nsamples_y, args.nsamples_x) == 0, "args.nsamples_y  % args.nsamples_x  == 0");
    const channel_ratio: c_int = if (args.ids_dst != null) 1 else i32Of(@divTrunc(args.nchannels_y, args.nchannels_x));
    const sample_ratio: c_int = i32Of(@divTrunc(args.nsamples_y, args.nsamples_x));

    const blocks_per_ne00_fd = ggml.cuda.common.initFastdivValues(@bitCast(@divTrunc(args.ncols_x, ggml.cuda.common.typeTraits(ty).qk)));
    const ntj_fd = ggml.cuda.common.initFastdivValues(@bitCast(@as(i64, ntj)));
    const nchannels_y_fd = ggml.cuda.common.initFastdivValues(@bitCast(args.nchannels_y));
    const nsamples_y_fd = ggml.cuda.common.initFastdivValues(@bitCast(args.nsamples_y));
    const channel_ratio_fd = ggml.cuda.common.initFastdivValues(@intCast(channel_ratio));
    const sample_ratio_fd = ggml.cuda.common.initFastdivValues(@intCast(sample_ratio));

    const k = comptime mulMatQKernel(ty, J, fallback, prec_src1, has_fusion);
    const shmem: usize = @intCast(nbytes_shared);

    if (has_fusion or !config.stream_k) {
        ggml.impl.assert(nti <= 65535, "nti <= 65535"); // gridDim.y limit
        ggml.cuda.kernels.launch(k, .{ .grid = block_nums_xy_tiling, .block = block_dims, .shmem = shmem, .stream = stream }, .{
            args.x,                       args.fusion.gate,             args.y,                                                      args.ids_dst,
            args.tile_spec,               args.dst,                     @as(?*f32, null),                                            args.y_scale,
            args.fusion.glu_op,           args.fusion.glu_limit,        @as(?*const f32, @ptrCast(@alignCast(args.fusion.x_scale))), @as(?*const f32, @ptrCast(@alignCast(args.fusion.gate_scale))),
            blocks_per_ne00_fd,           i32Of(args.nrows_x),          i32Of(args.ncols_dst),                                       i32Of(args.stride_row_x),
            i32Of(args.ncols_y),          i32Of(args.nrows_dst),        channel_ratio_fd,                                            nchannels_y_fd,
            i32Of(args.stride_channel_x), i32Of(args.stride_channel_y), i32Of(args.stride_channel_dst),                              sample_ratio_fd,
            nsamples_y_fd,                i32Of(args.stride_sample_x),  i32Of(args.stride_sample_y),                                 i32Of(args.stride_sample_dst),
            ntj_fd,
        });
        return;
    }

    // For the stream-k kernel it is possible to run it with tiling by
    // setting the number of CUDA blocks equal to the number of tiles.
    const ntiles_dst: c_int = ntj *% nti *% ntzw;
    const tiles_nwaves: c_int = @divTrunc(ntiles_dst + nsm - 1, nsm);
    const tiles_efficiency_percent: c_int = @divTrunc(100 * ntiles_dst, nsm * tiles_nwaves);

    // Assert that variable kbc will not overflow.
    // `int * unsigned int` is an unsigned product, wrapping.
    ggml.impl.assert(@as(u32, @bitCast(ntiles_dst)) *% blocks_per_ne00_fd.z < (1 << 30), "ntiles_dst * blocks_per_ne00_fd.z < (1 << 30)");

    // Maximum number of CUDA blocks that can do useful work.
    const nwork: c_int = i32Of(@as(i64, ntiles_dst) *% @divTrunc(args.ncols_x + config.K_vram - 1, config.K_vram));
    const block_nums_stream_k: ggml.cuda.kernels.Dim3 = .{
        .x = @bitCast(if (ggml.cuda.common.cc.isNvidia(x) and tiles_efficiency_percent >= 90) ntiles_dst else @min(nsm, nwork)),
    };

    const fixup_needed = @as(u32, @bitCast(ntiles_dst)) % block_nums_stream_k.x != 0;

    const pool = ctx.poolFor(id);
    var tmp_fixup = ggml.cuda.common.PoolAlloc(f32).init(pool);
    defer tmp_fixup.deinit();
    if (fixup_needed) {
        _ = tmp_fixup.alloc(@as(usize, block_nums_stream_k.x) * @as(usize, @intCast(config.J * config.I)));
    }

    const block_nums_fixup: ggml.cuda.kernels.Dim3 = .{ .x = block_nums_stream_k.x, .y = @intCast(@divTrunc(config.I, warp_size)), .z = 1 };
    const block_dims_fixup: ggml.cuda.kernels.Dim3 = .{ .x = block_dims.x, .y = block_dims.y / 2, .z = block_dims.z };

    const tmp_ptr: ?*f32 = if (tmp_fixup.ptr) |p| &p[0] else null;
    ggml.cuda.kernels.launch(k, .{ .grid = block_nums_stream_k, .block = block_dims, .shmem = shmem, .stream = stream }, .{
        args.x,                       args.fusion.gate,             args.y,                                                      args.ids_dst,
        args.tile_spec,               args.dst,                     tmp_ptr,                                                     args.y_scale,
        args.fusion.glu_op,           args.fusion.glu_limit,        @as(?*const f32, @ptrCast(@alignCast(args.fusion.x_scale))), @as(?*const f32, @ptrCast(@alignCast(args.fusion.gate_scale))),
        blocks_per_ne00_fd,           i32Of(args.nrows_x),          i32Of(args.ncols_dst),                                       i32Of(args.stride_row_x),
        i32Of(args.ncols_y),          i32Of(args.nrows_dst),        channel_ratio_fd,                                            nchannels_y_fd,
        i32Of(args.stride_channel_x), i32Of(args.stride_channel_y), i32Of(args.stride_channel_dst),                              sample_ratio_fd,
        nsamples_y_fd,                i32Of(args.stride_sample_x),  i32Of(args.stride_sample_y),                                 i32Of(args.stride_sample_dst),
        ntj_fd,
    });

    if (!fixup_needed) return;

    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGetLastError(), "cudaGetLastError()", @src());
    ggml.cuda.kernels.launch(comptime fixupKernel(ty, J, fallback, prec_src1), .{ .grid = block_nums_fixup, .block = block_dims_fixup, .stream = stream }, .{
        args.ids_dst,       args.tile_spec,                 args.dst,              tmp_ptr,
        blocks_per_ne00_fd, i32Of(args.nrows_x),            i32Of(args.ncols_dst), i32Of(args.nrows_dst),
        nchannels_y_fd,     i32Of(args.stride_channel_dst), nsamples_y_fd,         i32Of(args.stride_sample_dst),
        ntj_fd,
    });
}

/// Ports `mul_mat_q_switch_J` (mmq.cuh:1555 @b1b770d90): the dispatch on
/// `args.J_best`, which the caller chose.
fn mulMatQSwitchJ(comptime ty: ggml.impl.ggml_c.enum_ggml_type, comptime fallback: bool, comptime prec_src1: ggml.impl.ggml_c.enum_ggml_prec, comptime has_fusion: bool, ctx: *ggml.cuda.common.Context, args: *const Args, stream: ggml.cuda.common.cu.cudaStream_t) void {
    switch (args.J_best) {
        inline 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128 => |j| launchMulMatQ(ty, j, fallback, prec_src1, has_fusion, ctx, args, stream),
        else => {
            ggml.impl.logError("J_best=%d\n", .{args.J_best});
            ggml.impl.abort("fatal error");
        },
    }
}

/// Ports `mul_mat_q_switch_fusion` (mmq.cuh:1612 @b1b770d90):
/// a fused kernel exists only without the fallback and for the types
/// `fusionSupported` names.
fn mulMatQSwitchFusion(comptime ty: ggml.impl.ggml_c.enum_ggml_type, comptime fallback: bool, comptime prec_src1: ggml.impl.ggml_c.enum_ggml_prec, ctx: *ggml.cuda.common.Context, args: *const Args, stream: ggml.cuda.common.cu.cudaStream_t) void {
    if (comptime !fallback and fusionSupported(ty)) {
        if (args.fusion.gate != null) {
            mulMatQSwitchJ(ty, fallback, prec_src1, true, ctx, args, stream);
            return;
        }
    }
    ggml.impl.assert(args.fusion.gate == null, "!args.fusion.gate");
    mulMatQSwitchJ(ty, fallback, prec_src1, false, ctx, args, stream);
}

/// Ports `mul_mat_q_case` (mmq.cuh:1624 @b1b770d90). The `GGML_PREC_Q4`
/// instantiation exists for MXFP4 and NVFP4 only (`DECL_MMQ_CASE_W4A4`).
fn mulMatQCase(comptime ty: ggml.impl.ggml_c.enum_ggml_type, comptime prec_src1: ggml.impl.ggml_c.enum_ggml_prec, ctx: *ggml.cuda.common.Context, args: *const Args, stream: ggml.cuda.common.cu.cudaStream_t) void {
    if (needsFallback(args.nrows_x)) {
        mulMatQSwitchFusion(ty, true, prec_src1, ctx, args, stream);
    } else {
        mulMatQSwitchFusion(ty, false, prec_src1, ctx, args, stream);
    }
}

/// Ports `ggml_cuda_mul_mat_q_switch_type` (mmq.cu:8 @b1b770d90). It
/// spells `GGML_PREC_Q8` at every case.
fn mulMatQSwitchType(ctx: *ggml.cuda.common.Context, args: *const Args, stream: ggml.cuda.common.cu.cudaStream_t, prec_src1: ggml.impl.ggml_c.enum_ggml_prec) void {
    switch (args.type_x) {
        // src1 at Q4 uses the native FP4 instructions, which are Blackwell-only
        inline ggml.impl.ggml_c.GGML_TYPE_MXFP4, ggml.impl.ggml_c.GGML_TYPE_NVFP4 => |t| {
            if (prec_src1 == ggml.impl.ggml_c.GGML_PREC_Q4) {
                mulMatQCase(t, ggml.impl.ggml_c.GGML_PREC_Q4, ctx, args, stream);
            } else {
                mulMatQCase(t, ggml.impl.ggml_c.GGML_PREC_Q8, ctx, args, stream);
            }
        },
        inline ggml.impl.ggml_c.GGML_TYPE_Q1_0,
        ggml.impl.ggml_c.GGML_TYPE_Q2_0,
        ggml.impl.ggml_c.GGML_TYPE_Q4_0,
        ggml.impl.ggml_c.GGML_TYPE_Q4_1,
        ggml.impl.ggml_c.GGML_TYPE_Q5_0,
        ggml.impl.ggml_c.GGML_TYPE_Q5_1,
        ggml.impl.ggml_c.GGML_TYPE_Q8_0,
        ggml.impl.ggml_c.GGML_TYPE_Q2_K,
        ggml.impl.ggml_c.GGML_TYPE_Q3_K,
        ggml.impl.ggml_c.GGML_TYPE_Q4_K,
        ggml.impl.ggml_c.GGML_TYPE_Q5_K,
        ggml.impl.ggml_c.GGML_TYPE_Q6_K,
        ggml.impl.ggml_c.GGML_TYPE_IQ1_S,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XS,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_S,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_S,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_XS,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_NL,
        => |t| mulMatQCase(t, ggml.impl.ggml_c.GGML_PREC_Q8, ctx, args, stream),
        else => ggml.impl.abort("fatal error"),
    }
}

/// `ggml_cuda_mmq_get_prec_env`'s `static const` in
/// `ggml_cuda_mmq_get_prec_src1`: read once per process.
const PrecEnv = struct {
    var value: ggml.impl.ggml_c.enum_ggml_prec = ggml.impl.ggml_c.GGML_PREC_UNDEFINED;
    fn run() void {
        value = getPrecEnv();
    }
    const once = ggml.cuda.common.Once(run);
};

/// Ports `ggml_cuda_mmq_get_prec_env` (mmq.cu:94 @b1b770d90):
/// `GGML_CUDA_MMQ_PREC` overrides the `src1` precision the graph
/// asks for; "auto" (or anything unknown, with a warning) keeps it.
fn getPrecEnv() ggml.impl.ggml_c.enum_ggml_prec {
    const env_c = stdlib.env.getenv("GGML_CUDA_MMQ_PREC") orelse return ggml.impl.ggml_c.GGML_PREC_UNDEFINED;
    const env = std.mem.span(env_c);
    const env_cpp = std.heap.c_allocator.allocSentinel(u8, env.len, 0) catch ggml.impl.abort("out of memory");
    defer std.heap.c_allocator.free(env_cpp);
    for (env, 0..) |ch, i| env_cpp[i] = std.ascii.toLower(ch);
    if (std.mem.eql(u8, env_cpp, "q4")) {
        return ggml.impl.ggml_c.GGML_PREC_Q4;
    }
    if (std.mem.eql(u8, env_cpp, "q8")) {
        return ggml.impl.ggml_c.GGML_PREC_Q8;
    }
    if (!std.mem.eql(u8, env_cpp, "auto")) {
        ggml.impl.logWarn("%s: Unknown value for GGML_CUDA_MMQ_PREC: '%s'. Available: 'q4', 'q8', 'auto'.\n", .{ "ggml_cuda_mmq_get_prec_env", env_cpp.ptr });
    }
    return ggml.impl.ggml_c.GGML_PREC_UNDEFINED;
}

/// Ports `ggml_cuda_mmq_get_prec_src1` (mmq.cu:117 @b1b770d90):
/// `src1` is quantized to Q8_1 unless the FP4 types can use 4-bit
/// activations, in which case they default to the native W4A4 instructions
/// on Blackwell. It takes the type and the requested precision
/// rather than reading them off the tensors, so the fused paths can ask.
fn getPrecSrc1(ty: ggml.impl.ggml_c.enum_ggml_type, prec_requested: ggml.impl.ggml_c.enum_ggml_prec, x: c_int) ggml.impl.ggml_c.enum_ggml_prec {
    PrecEnv.once.call();
    const prec_env = PrecEnv.value;

    var prec = prec_env;
    if (prec == ggml.impl.ggml_c.GGML_PREC_UNDEFINED) {
        prec = prec_requested;
    }

    // Q4 only for the FP4 types on Blackwell
    ggml.impl.assert(prec == ggml.impl.ggml_c.GGML_PREC_UNDEFINED or prec == ggml.impl.ggml_c.GGML_PREC_Q8 or prec == ggml.impl.ggml_c.GGML_PREC_Q4, "prec == GGML_PREC_UNDEFINED || prec == GGML_PREC_Q8 || prec == GGML_PREC_Q4");
    const can_use_q4 = (ty == ggml.impl.ggml_c.GGML_TYPE_NVFP4 or ty == ggml.impl.ggml_c.GGML_TYPE_MXFP4) and ggml.cuda.common.blackwellMmaAvailable(x);
    if (prec == ggml.impl.ggml_c.GGML_PREC_Q8 or !can_use_q4) {
        return ggml.impl.ggml_c.GGML_PREC_Q8;
    }
    return ggml.impl.ggml_c.GGML_PREC_Q4;
}

/// Ports `ggml_cuda_mmq_get_J_best` (mmq.cu:135 @b1b770d90):
/// the tile width in ne11 (dense) / ne12 (MoE) direction with the fewest
/// tiles that fits into shared memory, 0 if there is none.
fn getJBest(ty: ggml.impl.ggml_c.enum_ggml_type, ncols: i64, fallback: bool, x: c_int, smpbo: usize, prec_src1: ggml.impl.ggml_c.enum_ggml_prec, nthreads_best: ?*c_int) c_int {
    var J_best: c_int = 0;
    var ntiles_J_best: c_int = std.math.maxInt(c_int);

    var J: c_int = 8;
    while (J <= 128 and ntiles_J_best > 1) : (J += 8) {
        const config = getConfig(ty, J, fallback, x, prec_src1);
        if (config.type == ggml.impl.ggml_c.GGML_TYPE_COUNT) continue;
        if (getNbytesShared(config, x) > smpbo) continue;

        const ntiles_x: c_int = i32Of(@divTrunc(ncols + config.J - 1, config.J));
        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            if (nthreads_best) |p| p.* = config.nthreads;
            ntiles_J_best = ntiles_x;
        }
    }
    return J_best;
}

/// Ports `ggml_cuda_mul_mat_q` (mmq.cu:164 @b1b770d90) (#29953).
/// It chooses the tile width `J_best` itself,
/// and pads `src1`'s quantized copy and `ids_dst` by what one tile of that
/// width can read past the end. With `fusion` (gate/up/GLU) `src0` is the
/// up weight and `dst` the GLU output, with `ids` too for MoE.
/// The fusion may carry NVFP4 weight scales, with or without a
/// gate (without, `dst` is the `MUL` applying the scale), and its
/// `prec_src1`; and the MoE path takes the linearized tile specs.
pub fn mulMatQ(ctx: *ggml.cuda.common.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, ids: ?*const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor, fusion: ?*const ggml.cuda.common.MmFusionArgsHost) void {
    ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "        src1->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "        dst->type  == GGML_TYPE_F32");
    ggml.impl.assert(ids == null or ids.?.type == ggml.impl.ggml_c.GGML_TYPE_I32, "!ids || ids->type  == GGML_TYPE_I32");

    // Fused gate/up/GLU: src0 is the up weight, dst the GLU output.
    // Fused scale (no gate): dst is the MUL that applies the weight scale to the MUL_MAT.
    const gate: ?*const ggml.impl.ggml_c.ggml_tensor = if (fusion) |f| f.gate else null;
    var fusion_device: ggml.cuda.common.MmFusionArgsDevice = .{};
    if (fusion) |f| {
        ggml.impl.assert(f.x_bias == null and f.gate_bias == null, "!fusion->x_bias && !fusion->gate_bias");
        // NVFP4 weight scales (only applied for NVFP4): one value per tensor, or one per expert with ids.
        for ([_]?*const ggml.impl.ggml_c.ggml_tensor{ f.x_scale, f.gate_scale }) |scale_opt| {
            if (scale_opt) |scale| {
                const n: i64 = if (ids != null) src0.ne[2] else 1;
                ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_NVFP4 and scale.type == ggml.impl.ggml_c.GGML_TYPE_F32 and ggml.types.ggml_is_contiguous(scale) and
                    ggml.types.ggml_nelements(scale) == n, "!scale || (src0->type == GGML_TYPE_NVFP4 && scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale) && ggml_nelements(scale) == (ids ? src0->ne[2] : 1))");
            }
        }
        if (gate) |g| {
            ggml.impl.assert((f.x_scale == null) == (f.gate_scale == null), "!fusion->x_scale == !fusion->gate_scale");
            ggml.impl.assert(g.type == src0.type and ggml.types.ggml_are_same_shape(g, src0) and ggml.types.ggml_are_same_stride(g, src0), "gate->type == src0->type && ggml_are_same_shape(gate, src0) && ggml_are_same_stride(gate, src0)");
            fusion_device.gate = g.data;
            fusion_device.gate_scale = if (f.gate_scale) |sc| sc.data else null;
            fusion_device.glu_op = f.glu_op;
            fusion_device.glu_limit = f.glu_limit;
        } else {
            ggml.impl.assert(f.gate_scale == null and f.x_scale != null, "!fusion->gate_scale && fusion->x_scale");
        }
        fusion_device.x_scale = if (f.x_scale) |sc| sc.data else null;
    }

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

    const id = ggml.cuda.common.getDevice();
    const x = ggml.cuda.common.info().devices[@intCast(id)].cc;
    const smpbo = ggml.cuda.common.info().devices[@intCast(id)].smpbo;

    const ts_src0 = ggml.types.ggml_type_size(src0.type);
    const ts_src1 = ggml.types.ggml_type_size(src1.type);
    const ts_dst = ggml.types.ggml_type_size(dst.type);

    ggml.impl.assert(src0.nb[0] == ts_src0, "        nb00       == ts_src0");
    ggml.impl.assert(src1.nb[0] == ts_src1, "        nb10       == ts_src1");
    ggml.impl.assert(dst.nb[0] == ts_dst, "        nb0        == ts_dst");
    ggml.impl.assert(ids == null or ids.?.nb[0] == ggml.types.ggml_type_size(ids.?.type), "!ids || ids->nb[0] == ggml_type_size(ids->type)");

    const src0_d: ?*const anyopaque = src0.data;
    const src1_d: ?*const f32 = @ptrCast(@alignCast(src1.data));
    const dst_d: ?*f32 = @ptrCast(@alignCast(dst.data));

    // If src0 (or the fused gate weight) is a temporary compute buffer, clear any potential padding.
    for ([_]?*const ggml.impl.ggml_c.ggml_tensor{ src0, gate }) |t_opt| {
        const t = t_opt orelse continue;
        if (ggml.backend.ggml_backend_buffer_get_usage(t.buffer) == ggml.impl.ggml_c.GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            const size_data = ggml.types.ggml_nbytes(t);
            const size_alloc = ggml.backend.ggml_backend_buffer_get_alloc_size(t.buffer, t);
            if (size_alloc > size_data) {
                ggml.impl.assert(ggml.types.ggml_is_contiguously_allocated(t), "ggml_is_contiguously_allocated(t)");
                ggml.impl.assert(t.view_src == null, "!t->view_src");
                ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemsetAsync(@as([*]u8, @ptrCast(t.data.?)) + size_data, 0, size_alloc - size_data, stream), "cudaMemsetAsync((char *) t->data + size_data, 0, size_alloc - size_data, stream)", @src());
            }
        }
    }

    const ne10_padded: i64 = @intCast(stdlib.num.pad(usize, @intCast(ne10), ggml.cuda.common.matrix_row_padding));

    const s01: i64 = @intCast(src0.nb[1] / ts_src0);
    const s1: i64 = @intCast(dst.nb[1] / ts_dst);
    const s02: i64 = @intCast(src0.nb[2] / ts_src0);
    const s2: i64 = @intCast(dst.nb[2] / ts_dst);
    const s03: i64 = @intCast(src0.nb[3] / ts_src0);
    const s3: i64 = @intCast(dst.nb[3] / ts_dst);

    const fallback = needsFallback(ne01);

    // With fusion dst is the GLU output or the scale node, not a MUL_MAT; the src1 precision requested by the fused MUL_MATs is in the fusion args.
    const prec_src1: ggml.impl.ggml_c.enum_ggml_prec = getPrecSrc1(src0.type, if (fusion) |f| f.prec_src1 else ggml.ops.ggml_prec_get_src(dst, 1), x);

    const use_native_fp4 = prec_src1 == ggml.impl.ggml_c.GGML_PREC_Q4;
    const y_block_size: i64 = if (use_native_fp4) sizeof_block_fp4_mmq else sizeof_block_q8_1_mmq;
    const y_values_per_block: i64 = if (use_native_fp4) qk_fp4_mmq else qk8_1_mmq;

    var J_best: c_int = 0;
    var nthreads_best: c_int = 0;
    {
        var ncols_opt: i64 = ne11;
        if (ids != null) {
            ncols_opt = ne12;

            // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
            ncols_opt = @divTrunc(ne12 * ids.?.ne[0] + ne02 - 1, ne02);
        }

        J_best = getJBest(src0.type, ncols_opt, fallback, x, smpbo, prec_src1, &nthreads_best);
    }
    ggml.impl.assert(J_best > 0, "J_best > 0");

    // A tile of size J can read in at most J - 1 extra columns.
    // For simplicity, round up the padding of a full tile to a multiple of the number of bytes that nthreads can load in parallel.
    const src1_load_chunk_size: usize = @as(usize, @intCast(nthreads_best)) * @sizeOf(c_int);
    const src1_q8_1_padding: usize = ((@as(usize, @intCast(J_best)) * sizeof_block_q8_1_mmq + src1_load_chunk_size - 1) / src1_load_chunk_size) * src1_load_chunk_size;

    if (ids == null) {
        const ntj: c_int = i32Of(@divTrunc(ne1 + J_best - 1, J_best));

        const nbytes_src1_q8_1: usize = @as(usize, @intCast(@divTrunc(ne13 * ne12 * ne11 * ne10_padded * y_block_size, y_values_per_block))) + src1_q8_1_padding;
        var src1_q8_1 = ggml.cuda.common.PoolAlloc(u8).initAlloc(ctx.pool(), nbytes_src1_q8_1);
        defer src1_q8_1.deinit();
        var src1_scale = ggml.cuda.common.PoolAlloc(f32).init(ctx.pool());
        defer src1_scale.deinit();
        if (src0.type == ggml.impl.ggml_c.GGML_TYPE_NVFP4 and use_native_fp4) {
            _ = src1_scale.alloc(@intCast(ne13 * ne12 * ne11));
        }

        {
            const s11: i64 = @intCast(src1.nb[1] / ts_src1);
            const s12: i64 = @intCast(src1.nb[2] / ts_src1);
            const s13: i64 = @intCast(src1.nb[3] / ts_src1);
            if (use_native_fp4) {
                const align_float8 = 32;
                const use_aligned_float8 = ggml.cuda.common.isAligned(src1, align_float8);
                ggml.cuda.quantize.quantizeMmqFp4(src1_d, null, src1_q8_1.get(), firstOf(f32, src1_scale.ptr), src0.type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
            } else {
                ggml.cuda.quantize.quantizeMmqQ8_1(src1_d, null, src1_q8_1.get(), src0.type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
            }
            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGetLastError(), "cudaGetLastError()", @src());
        }

        // Stride depends on quantization format
        const s12: i64 = if (use_native_fp4)
            @divTrunc(ne11 * ne10_padded * sizeof_block_fp4_mmq, qk_fp4_mmq * @sizeOf(c_int))
        else
            @divTrunc(ne11 * ne10_padded * sizeof_block_q8_1, ggml.impl.ggml_c.QK8_1 * @sizeOf(c_int));
        const s13 = ne12 * s12;

        const args: Args = .{
            .x = src0_d,
            .type_x = src0.type,
            .y = src1_q8_1.get(),
            .ids_dst = null,
            .tile_spec = null,
            .dst = dst_d,
            .y_scale = if (src0.type == ggml.impl.ggml_c.GGML_TYPE_NVFP4 and use_native_fp4) firstOf(f32, src1_scale.ptr) else null,
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
            .ntj = ntj,
            .J_best = J_best,
            .fusion = fusion_device,
        };
        mulMatQSwitchType(ctx, &args, stream, prec_src1);
        return;
    }

    const id_t = ids.?;
    ggml.impl.assert(ne13 == 1, "ne13 == 1");
    ggml.impl.assert(nb12 % nb11 == 0, "nb12 % nb11 == 0");
    ggml.impl.assert(nb2 % nb1 == 0, "nb2  % nb1  == 0");

    const n_expert_used = id_t.ne[0];
    const ne_get_rows = ne12 * n_expert_used;
    ggml.impl.assert(ne1 == n_expert_used, "ne1 == n_expert_used");

    var ids_src1 = ggml.cuda.common.PoolAlloc(i32).initAlloc(ctx.pool(), @intCast(ne_get_rows));
    defer ids_src1.deinit();
    // Needs to be padded for unconditional memory access.
    var ids_dst = ggml.cuda.common.PoolAlloc(i32).initAlloc(ctx.pool(), @intCast(ne_get_rows + J_best - 1));
    defer ids_dst.deinit();

    const ntj: c_int = i32Of(@divTrunc(ne_get_rows + J_best - 1, J_best) + ne02);
    var tile_spec = ggml.cuda.common.PoolAlloc(ggml.cuda.mmid.TileSpec).initAlloc(ctx.pool(), @intCast(ntj));
    defer tile_spec.deinit();

    // gate/up activations are broadcast across experts (ne11 == 1):
    // quantize each token once and scatter to its slots. ids_src1 then
    // holds the inverse map (token slot -> compact row).
    const dedup_bcast = ne11 == 1 and n_expert_used > 1;

    {
        ggml.impl.assert(id_t.nb[0] == ggml.types.ggml_element_size(id_t), "ids->nb[0] == ggml_element_size(ids)");
        const si1: c_int = @intCast(id_t.nb[1] / ggml.types.ggml_element_size(id_t));
        const sis1: c_int = @intCast(nb12 / nb11);

        ggml.cuda.mmid.launchMmIdsHelperDispatch(@ptrCast(@alignCast(id_t.data)), firstOf(i32, ids_src1.ptr), firstOf(i32, ids_dst.ptr), firstOf(ggml.cuda.mmid.TileSpec, tile_spec.ptr), i32Of(ne02), i32Of(ne12), i32Of(n_expert_used), i32Of(ne11), J_best, si1, sis1, dedup_bcast, stream);
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGetLastError(), "cudaGetLastError()", @src());
    }

    const nbytes_src1_q8_1: usize = @as(usize, @intCast(@divTrunc(ne12 * n_expert_used * ne10_padded * y_block_size, y_values_per_block))) + src1_q8_1_padding;
    var src1_q8_1 = ggml.cuda.common.PoolAlloc(u8).initAlloc(ctx.pool(), nbytes_src1_q8_1);
    defer src1_q8_1.deinit();
    var src1_scale = ggml.cuda.common.PoolAlloc(f32).init(ctx.pool());
    defer src1_scale.deinit();
    if (src0.type == ggml.impl.ggml_c.GGML_TYPE_NVFP4 and use_native_fp4) {
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
            const use_aligned_float8 = ggml.cuda.common.isAligned(src1, align_float8);
            if (dedup_bcast) {
                ggml.cuda.quantize.quantizeScatterMmqFp4(src1_d, firstOf(i32, ids_src1.ptr), src1_q8_1.get(), firstOf(f32, src1_scale.ptr), src0.type, use_aligned_float8, ne10, s12, ne10_padded, ne12, ne11_flat, i32Of(n_expert_used), stream);
            } else {
                ggml.cuda.quantize.quantizeMmqFp4(src1_d, firstOf(i32, ids_src1.ptr), src1_q8_1.get(), firstOf(f32, src1_scale.ptr), src0.type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            ggml.cuda.quantize.quantizeScatterMmqQ8_1(src1_d, firstOf(i32, ids_src1.ptr), src1_q8_1.get(), src0.type, ne10, s12, ne10_padded, ne12, ne11_flat, i32Of(n_expert_used), stream);
        } else {
            ggml.cuda.quantize.quantizeMmqQ8_1(src1_d, firstOf(i32, ids_src1.ptr), src1_q8_1.get(), src0.type, ne10, s11, s12, s13, ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGetLastError(), "cudaGetLastError()", @src());
    }

    const s12: i64 = if (use_native_fp4)
        @divTrunc(ne11 * ne10_padded * sizeof_block_fp4_mmq, qk_fp4_mmq * @sizeOf(c_int))
    else
        @divTrunc(ne11 * ne10_padded * sizeof_block_q8_1, ggml.impl.ggml_c.QK8_1 * @sizeOf(c_int));
    const s13 = ne12 * s12;

    // ne02 is used instead of ne12 because the number of y channels
    // determines the z dimension of the CUDA grid.
    const args: Args = .{
        .x = src0_d,
        .type_x = src0.type,
        .y = src1_q8_1.get(),
        .ids_dst = firstOf(i32, ids_dst.ptr),
        .tile_spec = firstOf(ggml.cuda.mmid.TileSpec, tile_spec.ptr),
        .dst = dst_d,
        .y_scale = firstOf(f32, src1_scale.ptr),
        .ncols_x = ne00,
        .nrows_x = ne01,
        .ncols_dst = ne_get_rows,
        .stride_row_x = s01,
        .ncols_y = ne_get_rows,
        .nrows_dst = s1,
        .nchannels_x = ne02,
        .nchannels_y = 1,
        .stride_channel_x = s02,
        .stride_channel_y = s12,
        .stride_channel_dst = s2,
        .nsamples_x = ne03,
        .nsamples_y = ne13,
        .stride_sample_x = s03,
        .stride_sample_y = s13,
        .stride_sample_dst = s3,
        .ntj = ntj,
        .J_best = J_best,
        .fusion = fusion_device,
    };

    mulMatQSwitchType(ctx, &args, stream, prec_src1);
}

fn firstOf(comptime T: type, p: ?[*]T) ?*T {
    return if (p) |x| &x[0] else null;
}

/// Ports `ggml_cuda_should_use_mmq` (mmq.cu:395 @b1b770d90), NVIDIA arms
/// (the RDNA4_M arms are AMD).
/// The `cc == GGML_CUDA_CC_PH1` refusal is a Moore Threads arm, which
/// no CUDA device's compute capability reaches.
pub fn shouldUseMmq(ty: ggml.impl.ggml_c.enum_ggml_type, x: c_int, ne11: i64, n_experts: i64) bool {
    const mmq_supported = switch (ty) {
        ggml.impl.ggml_c.GGML_TYPE_Q1_0,
        ggml.impl.ggml_c.GGML_TYPE_Q2_0,
        ggml.impl.ggml_c.GGML_TYPE_Q4_0,
        ggml.impl.ggml_c.GGML_TYPE_Q4_1,
        ggml.impl.ggml_c.GGML_TYPE_Q5_0,
        ggml.impl.ggml_c.GGML_TYPE_Q5_1,
        ggml.impl.ggml_c.GGML_TYPE_Q8_0,
        ggml.impl.ggml_c.GGML_TYPE_Q2_K,
        ggml.impl.ggml_c.GGML_TYPE_Q3_K,
        ggml.impl.ggml_c.GGML_TYPE_Q4_K,
        ggml.impl.ggml_c.GGML_TYPE_Q5_K,
        ggml.impl.ggml_c.GGML_TYPE_Q6_K,
        ggml.impl.ggml_c.GGML_TYPE_IQ1_S,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XS,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_S,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_S,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_XS,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_NL,
        ggml.impl.ggml_c.GGML_TYPE_MXFP4,
        ggml.impl.ggml_c.GGML_TYPE_NVFP4,
        => true,
        else => false,
    };

    if (!mmq_supported) return false;

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back
    // to BLAS otherwise.
    {
        const id = ggml.cuda.common.getDevice();
        const smpbo = ggml.cuda.common.info().devices[@intCast(id)].smpbo;
        if (smpbo < 48 * 1024) return false;
    }

    if (ggml.cuda.common.turingMmaAvailable(x)) return true;

    if (ggml.cuda.common.highestCompiledArch(x) < ggml.cuda.common.cc.dp4a) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return x >= ggml.cuda.common.cc.pascal and n_experts > 0;
    }

    if (ggml.cuda.common.cc.isNvidia(x)) {
        return !ggml.cuda.common.fp16MmaHardwareAvailable(x) or ne11 < dp4a_max_batch_size;
    }

    // The AMD and Moore Threads arms are not reachable on a CUDA device.
    return ne11 < dp4a_max_batch_size;
}

/// Ports `ggml_cuda_mmq_fusion_is_efficient` (mmq.cu:529 @b1b770d90):
/// the fused kernel needs one block per tile, with no stream-k
/// split, so it is only worth taking if the tiling is efficient. It
/// takes the requested `src1` precision.
pub fn fusionIsEfficient(ty: ggml.impl.ggml_c.enum_ggml_type, prec_requested: ggml.impl.ggml_c.enum_ggml_prec, nrows_x: i64, ncols: i64, nchannels: i64, device: c_int) bool {
    const dev = ggml.cuda.common.info().devices[@intCast(device)];
    const x = dev.cc;
    const nsm: i64 = dev.nsm;
    const smpbo = dev.smpbo;

    // Same tile width as ggml_cuda_mul_mat_q; the fused kernel has no fallback and writes I/2 output rows per tile.
    const fallback = false;
    const prec_src1 = getPrecSrc1(ty, prec_requested, x);
    const J_best = getJBest(ty, ncols, fallback, x, smpbo, prec_src1, null);
    if (J_best == 0) return false;
    const nrows_tile: i64 = @divTrunc(getConfig(ty, J_best, fallback, x, prec_src1).I, 2);
    const ntiles = @divTrunc(ncols + J_best - 1, J_best) * @divTrunc(nrows_x + nrows_tile - 1, nrows_tile) * nchannels;
    // Same condition as launch_mul_mat_q uses to run the stream-k kernel with one block per tile.
    const tiles_nwaves = @divTrunc(ntiles + nsm - 1, nsm);
    const tiles_efficiency_percent = @divTrunc(100 * ntiles, nsm * tiles_nwaves);
    return tiles_efficiency_percent >= 90;
}
