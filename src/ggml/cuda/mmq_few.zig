//! Few-row MMQ: an int8 MMA matrix multiplication for 2 to 24 `src1`
//! columns, faster than MMVQ and MMQ there on Ampere to Blackwell. A CUDA
//! block takes 16 `src0` rows (32 for two column groups on large matrices);
//! its warps split K in units of 256 values and copy raw `src0` rows into a
//! `cp.async` ring, and `src1` is quantized to the MMA fragment layout.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/mmq-few.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/mmq-few.cuh` at v1.0.0 (`b1b770d90`), new in
//! that release (#31533), the host half. The kernel, `mul_mat_q_few`, stays
//! CUDA, as does `quantize.cu`'s `quantize_mmq_few_q8_1`. The arm compiled
//! is the NVIDIA one; the HIP/MUSA stubs are not.
//!
//! The `__host__ __device__` sizing helpers are ported here because the host
//! asks them too: the kernel recomputes the same answers from its template
//! arguments.

const ggml = @import("../module.zig");
/// Ports `MMQ_FEW_MAX_NCG` (mmq-few.cu:17 @b1b770d90): max. number of
/// column groups.
const max_ncg = 3;
/// Ports `MMQ_FEW_NROWS` (mmq-few.cu:18 @b1b770d90): `src0` rows of one MMA
/// tile.
const nrows_tile = 16;
/// Ports `MMQ_FEW_NWARPS` (mmq-few.cu:19 @b1b770d90).
const nwarps = 4;

/// Ports `mmq_few_unit_size_of` (mmq-few.cu:22 @b1b770d90): `src0` bytes
/// per 256 values, `QK_K/qk*bs`.
fn unitSizeOf(ty: ggml.impl.c.enum_ggml_type) c_int {
    return @intCast(@divTrunc(@as(i64, ggml.impl.c.QK_K), ggml.impl.c.ggml_blck_size(ty)) * @as(i64, @intCast(ggml.impl.c.ggml_type_size(ty))));
}

/// Ports `mmq_few_unit_size` (mmq-few.cu:27 @b1b770d90): `src0` bytes per
/// 256 values, 0 for the types without a few-row decoder.
fn unitSize(ty: ggml.impl.c.enum_ggml_type) c_int {
    return switch (ty) {
        ggml.impl.c.GGML_TYPE_Q8_0,
        ggml.impl.c.GGML_TYPE_Q3_K,
        ggml.impl.c.GGML_TYPE_Q4_K,
        ggml.impl.c.GGML_TYPE_Q5_K,
        ggml.impl.c.GGML_TYPE_Q6_K,
        ggml.impl.c.GGML_TYPE_IQ3_S,
        ggml.impl.c.GGML_TYPE_IQ4_NL,
        ggml.impl.c.GGML_TYPE_IQ4_XS,
        => unitSizeOf(ty),
        else => 0,
    };
}

/// Ports `mmq_few_ds_layout` (mmq-few.cu:42 @b1b770d90): Q4_K and Q5_K also
/// need the `src1` sums for their mins.
fn dsLayout(ty: ggml.impl.c.enum_ggml_type) ggml.cuda.mmq.Q8_1DsLayout {
    return if (ty == ggml.impl.c.GGML_TYPE_Q4_K or ty == ggml.impl.c.GGML_TYPE_Q5_K) .ds4 else .d4;
}

/// Ports `mmq_few_nunits_stage` (mmq-few.cu:47 @b1b770d90): units of 256
/// values per ring stage -- 1 for Q6_K, for 3 column groups (fewer free
/// registers) and for 32-row tiles (2 blocks per SM), else 2.
fn nunitsStage(ty: ggml.impl.c.enum_ggml_type, ncg: c_int, nmt: c_int) c_int {
    return if (ty == ggml.impl.c.GGML_TYPE_Q6_K or ncg > 2 or nmt > 1) 1 else 2;
}

/// Ports `mmq_few_nstages` (mmq-few.cu:52 @b1b770d90): a third ring stage
/// only makes Q6_K faster.
fn nstages(ty: ggml.impl.c.enum_ggml_type) c_int {
    return if (ty == ggml.impl.c.GGML_TYPE_Q6_K) 3 else 2;
}

/// Ports `mmq_few_unaligned` (mmq-few.cu:57 @b1b770d90): whether a ring
/// stage's `src0` data fills whole 16-byte chunks.
fn unaligned(ty: ggml.impl.c.enum_ggml_type, nunits_stage: c_int) bool {
    return @rem(nunits_stage * unitSize(ty), 16) != 0;
}

/// Ports `mmq_few_nchunks` (mmq-few.cu:62 @b1b770d90): for unaligned data
/// the max. over the 8 possible start offsets.
fn nchunks(ty: ggml.impl.c.enum_ggml_type, nunits_stage: c_int) c_int {
    const nbytes = nunits_stage * unitSize(ty);
    return if (unaligned(ty, nunits_stage)) @divTrunc(14 + nbytes + 15, 16) else @divTrunc(nbytes, 16);
}

/// Ports `mmq_few_sram_stride` (mmq-few.cu:68 @b1b770d90): unaligned rows are
/// padded to an odd number of 16-byte chunks against bank conflicts.
fn sramStride(ty: ggml.impl.c.enum_ggml_type, nunits_stage: c_int) c_int {
    const n = nchunks(ty, nunits_stage);
    return if (unaligned(ty, nunits_stage) and @rem(n, 2) == 0) 16 * (n + 1) else 16 * n;
}

/// Ports `mmq_few_nbytes_shared` (mmq-few.cu:73 @b1b770d90).
fn nbytesShared(ty: ggml.impl.c.enum_ggml_type, ncg: c_int, nmt: c_int) c_int {
    return nwarps * nstages(ty) * nmt * nrows_tile * sramStride(ty, nunitsStage(ty, ncg, nmt));
}

/// Ports `mmq_few_tall_supported` (mmq-few.cu:78 @b1b770d90): with Q6_K and
/// Q8_0 two CUDA blocks with 32-row tiles would not fit into an SM.
fn tallSupported(ty: ggml.impl.c.enum_ggml_type) bool {
    return ty == ggml.impl.c.GGML_TYPE_Q3_K or ty == ggml.impl.c.GGML_TYPE_Q4_K or ty == ggml.impl.c.GGML_TYPE_Q5_K or
        ty == ggml.impl.c.GGML_TYPE_IQ3_S or ty == ggml.impl.c.GGML_TYPE_IQ4_NL or ty == ggml.impl.c.GGML_TYPE_IQ4_XS;
}

/// Ports `mmq_few_is_constant` (mmq-few.cu:692 @b1b770d90): `ggml.c`'s
/// static `ggml_is_constant` on the tensor a view points to.
fn isConstant(src0: *const ggml.impl.c.ggml_tensor) bool {
    const x_root: *const ggml.impl.c.ggml_tensor = if (src0.view_src) |v| v else src0;
    return x_root.buffer != null and ggml.impl.c.ggml_backend_buffer_get_usage(x_root.buffer) == ggml.impl.c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS and
        (x_root.flags & ggml.impl.c.GGML_TENSOR_FLAG_PARAM) == 0;
}

fn kernel(comptime ty: ggml.impl.c.enum_ggml_type, comptime ncg: comptime_int, comptime nmt: comptime_int) ggml.cuda.kernels.Kernel {
    return ggml.cuda.kernels.get("mmq-few", ggml.cuda.kernels.tmpl("mul_mat_q_few", &.{ ggml.cuda.kernels.ggmlType(ty), ggml.cuda.kernels.int(ncg), ggml.cuda.kernels.int(nmt) }));
}

/// Ports `launch_mul_mat_q_few` (mmq-few.cu:699 @b1b770d90).
fn launch(comptime ty: ggml.impl.c.enum_ggml_type, comptime ncg: comptime_int, comptime nmt: comptime_int, src0: *const ggml.impl.c.ggml_tensor, src1: *const ggml.impl.c.ggml_tensor, dst: *ggml.impl.c.ggml_tensor, src1_q: ?*const anyopaque, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];
    const nb01 = src0.nb[1];
    const nb02 = src0.nb[2];
    const nb03 = src0.nb[3];
    const ne1 = dst.ne[1];
    const nb1 = dst.nb[1];
    const nb2 = dst.nb[2];
    const nb3 = dst.nb[3];
    const ne12 = src1.ne[2];
    const ne13 = src1.ne[3];

    const nunits: c_int = @truncate(@divTrunc(ne00, ggml.impl.c.QK_K));
    ggml.impl.assert(@rem(nunits, nunitsStage(ty, ncg, nmt)) == 0, "nunits % mmq_few_nunits_stage(type, ncg, nmt) == 0");

    const warp_size = ggml.cuda.common.info().devices[@intCast(ggml.cuda.common.getDevice())].warp_size;
    const nbytes_shared = nbytesShared(ty, ncg, nmt);
    const k = comptime kernel(ty, ncg, nmt);
    ggml.cuda.kernels.setSharedMemoryLimit(k, @intCast(nbytes_shared));

    const x_is_constant = isConstant(src0);

    const block_nums: ggml.cuda.kernels.Dim3 = .{
        .x = @truncate(@as(u64, @bitCast(@divTrunc(ne01 + nmt * nrows_tile - 1, nmt * nrows_tile)))),
        .y = @truncate(@as(u64, @bitCast(ne12))),
        .z = @truncate(@as(u64, @bitCast(ne13))),
    };
    const block_dims: ggml.cuda.kernels.Dim3 = .{ .x = @bitCast(warp_size), .y = nwarps, .z = 1 };
    ggml.cuda.kernels.launchPdl(k, .{ .grid = block_nums, .block = block_dims, .shmem = @intCast(nbytes_shared), .stream = stream }, .{
        src0.data,                src1_q,                                                        dst.data,
        nunits,                   @as(c_int, @truncate(ne01)),                                   @as(c_int, @truncate(ne1)),
        @as(i64, @bitCast(nb01)), @as(c_int, @truncate(@as(i64, @bitCast(nb1 / @sizeOf(f32))))), @as(c_int, @truncate(@divTrunc(ne12, ne02))),
        @as(i64, @bitCast(nb02)), @as(c_int, @truncate(@as(i64, @bitCast(nb2 / @sizeOf(f32))))), @as(c_int, @truncate(@divTrunc(ne13, ne03))),
        @as(i64, @bitCast(nb03)), @as(c_int, @truncate(@as(i64, @bitCast(nb3 / @sizeOf(f32))))), x_is_constant,
    });
}

/// Ports `mmq_few_ncg` (mmq-few.cu:722 @b1b770d90).
fn ncgOf(ncols: i64) c_int {
    return @truncate(@divTrunc(ncols + ggml.cuda.quantize.mmq_few_ncols - 1, ggml.cuda.quantize.mmq_few_ncols));
}

/// Ports `mmq_few_nmt` (mmq-few.cu:727 @b1b770d90): with 2 column groups,
/// 32-row tiles halve the `src1` loads per `src0` byte; with 3 they spill
/// registers.
fn nmtOf(ty: ggml.impl.c.enum_ggml_type, ncg: c_int, nrows_x: i64) c_int {
    const nsm = ggml.cuda.common.info().devices[@intCast(ggml.cuda.common.getDevice())].nsm;
    return if (tallSupported(ty) and ncg == 2 and nrows_x >= @as(i64, nsm) * 2 * nrows_tile) 2 else 1;
}

/// Ports `launch_mul_mat_q_few_tall` (mmq-few.cu:733 @b1b770d90).
fn launchTall(comptime ty: ggml.impl.c.enum_ggml_type, src0: *const ggml.impl.c.ggml_tensor, src1: *const ggml.impl.c.ggml_tensor, dst: *ggml.impl.c.ggml_tensor, src1_q: ?*const anyopaque, stream: ggml.cuda.common.cu.cudaStream_t) void {
    if (comptime tallSupported(ty)) {
        if (nmtOf(ty, 2, src0.ne[1]) == 2) {
            launch(ty, 2, 2, src0, src1, dst, src1_q, stream);
            return;
        }
    }
    launch(ty, 2, 1, src0, src1, dst, src1_q, stream);
}

/// Ports `launch_mul_mat_q_few_switch_ncg` (mmq-few.cu:745 @b1b770d90).
fn launchSwitchNcg(comptime ty: ggml.impl.c.enum_ggml_type, src0: *const ggml.impl.c.ggml_tensor, src1: *const ggml.impl.c.ggml_tensor, dst: *ggml.impl.c.ggml_tensor, src1_q: ?*const anyopaque, ncg: c_int, stream: ggml.cuda.common.cu.cudaStream_t) void {
    switch (ncg) {
        1 => launch(ty, 1, 1, src0, src1, dst, src1_q, stream),
        2 => launchTall(ty, src0, src1, dst, src1_q, stream),
        3 => launch(ty, 3, 1, src0, src1, dst, src1_q, stream),
        else => ggml.impl.abort("fatal error"),
    }
}

/// Ports `ggml_cuda_mul_mat_q_few` (mmq-few.cu:762 @b1b770d90).
pub fn mulMatQFew(ctx: *ggml.cuda.common.Context, src0: *const ggml.impl.c.ggml_tensor, src1: *const ggml.impl.c.ggml_tensor, dst: *ggml.impl.c.ggml_tensor) void {
    const ne00 = src0.ne[0];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];
    const ne10 = src1.ne[0];
    const ne11 = src1.ne[1];
    const ne12 = src1.ne[2];
    const ne13 = src1.ne[3];

    ggml.impl.assert(src1.type == ggml.impl.c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");
    ggml.impl.assert(src0.nb[0] == ggml.impl.c.ggml_type_size(src0.type), "nb00 == ggml_type_size(src0->type)");
    ggml.impl.assert(src1.nb[0] == @sizeOf(f32), "nb10 == sizeof(float)");
    ggml.impl.assert(dst.nb[0] == @sizeOf(f32), "nb0  == sizeof(float)");
    ggml.impl.assert(ne11 <= max_ncg * ggml.cuda.quantize.mmq_few_ncols, "ne11 <= MMQ_FEW_MAX_NCG*MMQ_FEW_NCOLS");
    ggml.impl.assert(@rem(ne00, ggml.impl.c.QK_K) == 0, "ne00 % QK_K == 0");
    ggml.impl.assert(@rem(ne12, ne02) == 0 and @rem(ne13, ne03) == 0, "ne12 % ne02 == 0 && ne13 % ne03 == 0");

    const stream = ctx.stream();

    const nunits: c_int = @truncate(@divTrunc(ne00, ggml.impl.c.QK_K));
    const ncg = ncgOf(ne11);

    var src1_q = ggml.cuda.common.PoolAlloc(ggml.cuda.quantize.Int4).initAlloc(ctx.pool(), @intCast(ne12 * ne13 * nunits * ncg * ggml.cuda.quantize.mmq_few_q8_1_int4));
    defer src1_q.deinit();
    const src1_q_ptr: ?*anyopaque = @ptrCast(src1_q.get());
    ggml.cuda.quantize.quantizeMmqFewQ8_1(@ptrCast(@alignCast(src1.data)), src1_q_ptr, dsLayout(src0.type), ne10, @intCast(src1.nb[1] / @sizeOf(f32)), @intCast(src1.nb[2] / @sizeOf(f32)), @intCast(src1.nb[3] / @sizeOf(f32)), ne11, ne12, ne13, stream);

    switch (src0.type) {
        inline ggml.impl.c.GGML_TYPE_Q8_0,
        ggml.impl.c.GGML_TYPE_Q3_K,
        ggml.impl.c.GGML_TYPE_Q4_K,
        ggml.impl.c.GGML_TYPE_Q5_K,
        ggml.impl.c.GGML_TYPE_Q6_K,
        ggml.impl.c.GGML_TYPE_IQ3_S,
        ggml.impl.c.GGML_TYPE_IQ4_NL,
        ggml.impl.c.GGML_TYPE_IQ4_XS,
        => |t| launchSwitchNcg(t, src0, src1, dst, src1_q_ptr, ncg, stream),
        else => ggml.impl.abort("unsupported type for few-row MMQ"),
    }
}

/// Ports `ggml_cuda_mmq_few_min_ncols` (mmq-few.cu:814 @b1b770d90): the min.
/// number of `src1` columns for few-row MMQ, 0 if it is not used for this
/// type, GPU and row length. `GGML_CUDA_FORCE_CUBLAS` is not defined.
fn minNcols(ty: ggml.impl.c.enum_ggml_type, x: c_int, ne00: i64) c_int {
    // NVIDIA Ampere to Blackwell
    if (!ggml.cuda.common.ampereMmaAvailable(x) or x >= ggml.cuda.common.cc.rubin) return 0;
    // on DGX Spark the kernel needs its L2 prefetch, which is only in the DGX Spark device code
    if (x == ggml.cuda.common.cc.dgx_spark and ggml.cuda.common.highestCompiledArch(x) != ggml.cuda.common.cc.dgx_spark) return 0;
    // with fewer than 8 units of 256 values per row the warps idle, MMVQ and MMQ are better there
    if (unitSize(ty) == 0 or ne00 < 8 * ggml.impl.c.QK_K) return 0;
    if (x == ggml.cuda.common.cc.dgx_spark) { // tuned on DGX Spark GB10
        return if (ty == ggml.impl.c.GGML_TYPE_IQ4_XS or ty == ggml.impl.c.GGML_TYPE_IQ4_NL) 8 else 2;
    }
    return switch (ty) { // all GPUs except DGX Spark, tuned on RTX 5090
        ggml.impl.c.GGML_TYPE_Q3_K => 3,
        ggml.impl.c.GGML_TYPE_Q5_K => 3,
        ggml.impl.c.GGML_TYPE_Q4_K => 4,
        ggml.impl.c.GGML_TYPE_IQ4_NL => 4,
        ggml.impl.c.GGML_TYPE_Q6_K => 4,
        ggml.impl.c.GGML_TYPE_Q8_0 => 5,
        ggml.impl.c.GGML_TYPE_IQ3_S => 5,
        ggml.impl.c.GGML_TYPE_IQ4_XS => 6,
        else => 0,
    };
}

/// Ports `ggml_cuda_should_use_mmq_few` (mmq-few.cu:847 @b1b770d90).
pub fn shouldUseMmqFew(src0: *const ggml.impl.c.ggml_tensor, src1: *const ggml.impl.c.ggml_tensor, x: c_int) bool {
    const min_ncols = minNcols(src0.type, x, src0.ne[0]);
    if (min_ncols == 0 or src1.ne[1] < min_ncols or src1.ne[1] > max_ncg * ggml.cuda.quantize.mmq_few_ncols) return false;
    // the row must hold a whole number of ring stages of the launched kernel
    const ncg = ncgOf(src1.ne[1]);
    const nmt = nmtOf(src0.type, ncg, src0.ne[1]);
    const nunits_stage = nunitsStage(src0.type, ncg, nmt);
    if (@rem(src0.ne[0], @as(i64, nunits_stage) * ggml.impl.c.QK_K) != 0) return false;
    // unaligned data needs the alignment of the decoder loads
    const x_alignment: usize = if (unaligned(src0.type, nunits_stage)) (if (ggml.impl.c.ggml_type_size(src0.type) % 4 == 0) 4 else 2) else 16;
    return ggml.cuda.common.isAligned(src0, x_alignment) and ggml.cuda.common.isAligned(src1, 16);
}
