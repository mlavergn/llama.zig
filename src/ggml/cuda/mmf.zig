//! MMF: matrix multiplication of `F32`, `F16` and `BF16` weights with
//! tensor cores, for up to 16 columns (or more, for `MUL_MAT_ID`).
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/mmf.cu` and the host half of
//! `llama.cpp/ggml/src/ggml-cuda/mmf.cuh` at v0.4.0 (`5266f24da`). The
//! kernels, `mul_mat_f` and `mul_mat_f_ids`, stay CUDA, one set per column
//! count in `template-instances/mmf-instance-ncols_<n>.cu`.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");
const mmid = @import("mmid.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `MMF_ROWS_PER_BLOCK` (mmf.cuh:9 @5266f24da).
const rows_per_block_default = 32;

/// Ports `MMF_ROWS_PER_BLOCK_CDNA` (mmf.cuh:10 @5266f24da).
const rows_per_block_cdna = 64;

/// Ports `mmf_get_max_block_size` (mmf.cuh:12 @5266f24da), NVIDIA arm.
fn getMaxBlockSize(x: c_int) i64 {
    _ = x;
    return 256;
}

/// Ports `mmf_get_padding` (mmf.cuh:20 @5266f24da), NVIDIA arm.
fn getPadding(x: c_int) c_int {
    _ = x;
    return 4;
}

/// Ports `mmf_get_rows_per_block` (mmf.cu:5 @5266f24da), NVIDIA arm.
fn getRowsPerBlock(x: c_int) c_int {
    _ = x;
    return rows_per_block_default;
}

/// Ports `mmf_ids_data` (mmf.cuh:36 @5266f24da).
const IdsData = struct {
    ids_src_compact: ?*const i32 = null,
    ids_dst_compact: ?*const i32 = null,
    expert_bounds_dev: ?*const i32 = null,
    n_experts: c_int = 0,
    sis1: c_int = 0,
};

fn i32Of(x: i64) c_int {
    return @truncate(x);
}

fn u32Of(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

fn instanceTu(comptime cols_per_block: comptime_int) []const u8 {
    return std.fmt.comptimePrint("mmf-instance-ncols_{d}", .{cols_per_block});
}

/// The arguments the two kernels share, at the kernels' (all `int`)
/// parameter types.
const Args = struct {
    x: ?*const anyopaque,
    y: ?*const f32,
    ids: ?*const i32,
    dst: ?*f32,
    ncols_x: i64,
    ncols_dst: i64,
    nchannels_dst: i64,
    stride_row: i64,
    stride_col_y: i64,
    stride_col_dst: i64,
    stride_col_id: i64,
    stride_row_id: i64,
    channel_ratio: i64,
    stride_channel_x: i64,
    stride_channel_y: i64,
    stride_channel_dst: i64,
    sample_ratio: i64,
    stride_sample_x: i64,
    stride_sample_y: i64,
    stride_sample_dst: i64,
};

/// Ports `mul_mat_f_switch_ids` (mmf.cuh:573 @5266f24da).
fn switchIds(comptime T: type, comptime rpb: comptime_int, comptime cpb: comptime_int, comptime nwarps: comptime_int, a: *const Args, block_nums: kernels.Dim3, block_dims: kernels.Dim3, nbytes_shared_total: c_int, stream: cu.cudaStream_t, ids_data: ?*const IdsData) void {
    const has_ids_data = ids_data != null and ids_data.?.ids_src_compact != null;
    const tu = comptime instanceTu(cpb);
    const tname = comptime kernels.typeName(T);
    const shmem: usize = @intCast(nbytes_shared_total);

    // Use the compact-ids kernel only for larger tiles; for small
    // ncols_dst (< 16) the normal mul_mat_f path with has_ids=true.
    if (has_ids_data and a.ncols_dst > 16) {
        const d = ids_data.?;
        const max_tiles: c_int = i32Of(@divTrunc(a.ncols_dst + cpb - 1, cpb));
        if (max_tiles == 0) return;
        const block_nums_ids: kernels.Dim3 = .{ .x = block_nums.x, .y = @bitCast(d.n_experts), .z = @bitCast(max_tiles) };

        const sis1_fd: cu.uint3 = if (d.sis1 > 0) common.initFastdivValues(@as(u32, @bitCast(d.sis1))) else .{ .x = 0, .y = 0, .z = 1 };
        const nch_fd = common.initFastdivValues(@as(u32, @truncate(@as(u64, @bitCast(a.nchannels_dst)))));

        kernels.launch(comptime kernels.get(tu, kernels.tmpl("mul_mat_f_ids", &.{ tname, kernels.int(rpb), kernels.int(cpb), kernels.int(nwarps) })), .{
            .grid = block_nums_ids,
            .block = block_dims,
            .shmem = shmem,
            .stream = stream,
        }, .{
            a.x,                    a.y,                       d.ids_src_compact,         d.ids_dst_compact,
            d.expert_bounds_dev,    a.dst,                     i32Of(a.ncols_x),          i32Of(a.ncols_dst),
            i32Of(a.nchannels_dst), i32Of(a.stride_row),       i32Of(a.stride_col_y),     i32Of(a.stride_col_dst),
            i32Of(a.channel_ratio), i32Of(a.stride_channel_x), i32Of(a.stride_channel_y), i32Of(a.stride_channel_dst),
            i32Of(a.sample_ratio),  i32Of(a.stride_sample_x),  i32Of(a.stride_sample_y),  i32Of(a.stride_sample_dst),
            sis1_fd,                nch_fd,
        });
    } else if (a.ids != null) {
        const col_tiles: i64 = @divTrunc(a.ncols_dst + cpb - 1, cpb);
        var block_nums_ids = block_nums;
        block_nums_ids.y *%= u32Of(col_tiles);

        kernels.launch(comptime kernels.get(tu, kernels.tmpl("mul_mat_f", &.{ tname, kernels.int(rpb), kernels.int(cpb), kernels.int(nwarps), kernels.boolean(true) })), .{
            .grid = block_nums_ids,
            .block = block_dims,
            .shmem = shmem,
            .stream = stream,
        }, .{
            a.x,                    a.y,                       a.ids,                     a.dst,
            i32Of(a.ncols_x),       i32Of(a.ncols_dst),        i32Of(a.nchannels_dst),    i32Of(a.stride_row),
            i32Of(a.stride_col_y),  i32Of(a.stride_col_dst),   i32Of(a.stride_col_id),    i32Of(a.stride_row_id),
            i32Of(a.channel_ratio), i32Of(a.stride_channel_x), i32Of(a.stride_channel_y), i32Of(a.stride_channel_dst),
            i32Of(a.sample_ratio),  i32Of(a.stride_sample_x),  i32Of(a.stride_sample_y),  i32Of(a.stride_sample_dst),
        });
    } else {
        kernels.launch(comptime kernels.get(tu, kernels.tmpl("mul_mat_f", &.{ tname, kernels.int(rpb), kernels.int(cpb), kernels.int(nwarps), kernels.boolean(false) })), .{
            .grid = block_nums,
            .block = block_dims,
            .shmem = shmem,
            .stream = stream,
        }, .{
            a.x,                    a.y,                       a.ids,                     a.dst,
            i32Of(a.ncols_x),       @as(c_int, cpb),           i32Of(a.nchannels_dst),    i32Of(a.stride_row),
            i32Of(a.stride_col_y),  i32Of(a.stride_col_dst),   i32Of(a.stride_col_id),    i32Of(a.stride_row_id),
            i32Of(a.channel_ratio), i32Of(a.stride_channel_x), i32Of(a.stride_channel_y), i32Of(a.stride_channel_dst),
            i32Of(a.sample_ratio),  i32Of(a.stride_sample_x),  i32Of(a.stride_sample_y),  i32Of(a.stride_sample_dst),
        });
    }
}

/// Ports `mul_mat_f_cuda` (mmf.cuh:620 @5266f24da).
fn mulMatFCuda(
    comptime T: type,
    comptime rpb: comptime_int,
    comptime cpb: comptime_int,
    x: ?*const anyopaque,
    y: ?*const f32,
    ids: ?*const i32,
    dst: ?*f32,
    ncols_x: i64,
    nrows_x: i64,
    ncols_dst: i64,
    stride_row: i64,
    stride_col_y: i64,
    stride_col_dst: i64,
    stride_col_id: i64,
    stride_row_id: i64,
    nchannels_x: i64,
    nchannels_y: i64,
    nchannels_dst: i64,
    stride_channel_x: i64,
    stride_channel_y: i64,
    stride_channel_dst: i64,
    nsamples_x: i64,
    nsamples_dst: i64,
    stride_sample_x: i64,
    stride_sample_y: i64,
    stride_sample_dst: i64,
    stream: cu.cudaStream_t,
    ids_data: ?*const IdsData,
) void {
    _ = nchannels_y;
    impl.assert(@rem(ncols_x, 2) == 0, "ncols_x      % 2 == 0");
    impl.assert(@rem(stride_row, 2) == 0, "stride_row   % 2 == 0");
    impl.assert(@rem(stride_col_y, 2) == 0, "stride_col_y % 2 == 0");
    impl.assert(ids != null or @rem(nchannels_dst, nchannels_x) == 0, "ids || nchannels_dst % nchannels_x == 0");
    impl.assert(@rem(nsamples_dst, nsamples_x) == 0, "       nsamples_dst  % nsamples_x  == 0");
    const channel_ratio = @divTrunc(nchannels_dst, nchannels_x);
    const sample_ratio = @divTrunc(nsamples_dst, nsamples_x);

    const device = common.getDevice();
    const x_cc = common.info().devices[@intCast(device)].cc;
    const warp_size: i64 = common.info().devices[@intCast(device)].warp_size;

    var nwarps_best: i64 = 1;
    var niter_best: i64 = @divTrunc(ncols_x + warp_size * 2 - 1, warp_size * 2);
    const max_block_size = getMaxBlockSize(x_cc);
    var nwarps: i64 = 2;
    while (nwarps <= @divTrunc(max_block_size, warp_size)) : (nwarps += 1) {
        const niter = @divTrunc(ncols_x + nwarps * warp_size * 2 - 1, nwarps * warp_size * 2);
        if (niter < niter_best) {
            niter_best = niter;
            nwarps_best = nwarps;
        }
    }

    // tile<16, 8, T>::I, tile<32, 8, T>::I and tile<8, 8, T>::I.
    const tile_a_16_i = 16;
    const tile_a_32_i = 32;
    const tile_b_8_i = 8;
    const ws: c_int = @intCast(warp_size);
    const nwb: c_int = @intCast(nwarps_best);
    const nbytes_shared_iter: c_int = nwb * (if (common.voltaMmaAvailable(x_cc)) @as(c_int, tile_a_32_i) else tile_a_16_i) * (ws + getPadding(x_cc)) * 4;
    const nbytes_cols_per_block_pad: c_int = tile_b_8_i;
    const nbytes_shared_combine: c_int = @as(c_int, @intCast(impl.pad(cpb, @intCast(nbytes_cols_per_block_pad)))) * (nwb * rpb + getPadding(x_cc)) * 4;
    const nbytes_shared = @max(nbytes_shared_iter, nbytes_shared_combine);
    const nbytes_slotmap: c_int = if (ids != null) @as(c_int, @intCast(impl.pad(cpb, 16))) * @sizeOf(c_int) else 0;
    const nbytes_shared_total = nbytes_shared + nbytes_slotmap;
    const grid_y: i64 = if (ids != null) nchannels_x else nchannels_dst;

    const block_nums: kernels.Dim3 = .{ .x = u32Of(@divTrunc(nrows_x, rpb)), .y = u32Of(grid_y), .z = u32Of(nsamples_dst) };
    const block_dims: kernels.Dim3 = .{ .x = @intCast(ws), .y = @intCast(nwb), .z = 1 };

    const a: Args = .{
        .x = x,
        .y = y,
        .ids = ids,
        .dst = dst,
        .ncols_x = ncols_x,
        .ncols_dst = ncols_dst,
        .nchannels_dst = nchannels_dst,
        .stride_row = stride_row,
        .stride_col_y = stride_col_y,
        .stride_col_dst = stride_col_dst,
        .stride_col_id = stride_col_id,
        .stride_row_id = stride_row_id,
        .channel_ratio = channel_ratio,
        .stride_channel_x = stride_channel_x,
        .stride_channel_y = stride_channel_y,
        .stride_channel_dst = stride_channel_dst,
        .sample_ratio = sample_ratio,
        .stride_sample_x = stride_sample_x,
        .stride_sample_y = stride_sample_y,
        .stride_sample_dst = stride_sample_dst,
    };

    switch (nwarps_best) {
        inline 1, 2, 3, 4, 5, 6, 7, 8 => |nw| switchIds(T, rpb, cpb, nw, &a, block_nums, block_dims, nbytes_shared_total, stream, ids_data),
        else => impl.abort("fatal error"),
    }
}

/// Ports `mul_mat_f_switch_cols_per_block` (mmf.cuh:734 @5266f24da) and
/// `mul_mat_f_switch_rows_per_block` (mmf.cuh:836 @5266f24da). Only
/// `MMF_ROWS_PER_BLOCK` is reachable on an NVIDIA device.
fn switchColsPerBlock(comptime T: type, rows_per_block: c_int, x: ?*const anyopaque, y: ?*const f32, ids: ?*const i32, dst: ?*f32, ncols_x: i64, nrows_x: i64, ncols_dst: i64, stride_row: i64, stride_col_y: i64, stride_col_dst: i64, stride_col_id: i64, stride_row_id: c_int, nchannels_x: i64, nchannels_y: i64, nchannels_dst: i64, stride_channel_x: i64, stride_channel_y: i64, stride_channel_dst: i64, nsamples_x: i64, nsamples_dst: i64, stride_sample_x: i64, stride_sample_y: i64, stride_sample_dst: i64, stream: cu.cudaStream_t, ids_data: ?*const IdsData) void {
    if (rows_per_block != rows_per_block_default) {
        impl.logError("unsupported rows_per_block: %i\n", .{rows_per_block});
        impl.abort("unsupported rows_per_block");
    }

    const ncols_case: i64 = if (ids != null and ncols_dst > 16) 16 else ncols_dst;

    impl.assert(ids != null or ncols_dst <= 16, "ids || ncols_dst <= 16");

    switch (ncols_case) {
        inline 1...16 => |cpb| mulMatFCuda(T, rows_per_block_default, cpb, x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst, stride_col_id, stride_row_id, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_cuda_mul_mat_f` (mmf.cu:13 @5266f24da).
pub fn mulMatF(ctx: *common.Context, src0: *const Tensor, src1: *const Tensor, ids: ?*const Tensor, dst: *Tensor) void {
    impl.assert(src1.type == c.GGML_TYPE_F32, "        src1->type == GGML_TYPE_F32");
    impl.assert(ids == null or ids.?.type == c.GGML_TYPE_I32, "!ids ||  ids->type == GGML_TYPE_I32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "         dst->type == GGML_TYPE_F32");

    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];
    const ne11 = src1.ne[1];
    const ne12 = src1.ne[2];
    const ne13 = src1.ne[3];
    const ne1 = dst.ne[1];
    const ne2 = dst.ne[2];
    const ne3 = dst.ne[3];

    const ts_src0 = c.ggml_type_size(src0.type);
    const ts_src1 = c.ggml_type_size(src1.type);
    const ts_dst = c.ggml_type_size(dst.type);

    impl.assert(ne13 == ne3, "ne13 == ne3");

    impl.assert(src0.nb[0] == ts_src0, "        nb00       == ts_src0");
    impl.assert(src1.nb[0] == ts_src1, "        nb10       == ts_src1");
    impl.assert(ids == null or ids.?.nb[0] == c.ggml_type_size(ids.?.type), "!ids || ids->nb[0] == ggml_type_size(ids->type)");
    impl.assert(dst.nb[0] == ts_dst, "        nb0        == ts_dst");

    const src1_d: ?*const f32 = @ptrCast(@alignCast(src1.data));
    const ids_d: ?*const i32 = if (ids) |i| @ptrCast(@alignCast(i.data)) else null;
    const dst_d: ?*f32 = @ptrCast(@alignCast(dst.data));

    const s01: i64 = @intCast(src0.nb[1] / ts_src0);
    const s11: i64 = @intCast(src1.nb[1] / ts_src1);
    const s1: i64 = @intCast(dst.nb[1] / ts_dst);
    const s02: i64 = @intCast(src0.nb[2] / ts_src0);
    const s12: i64 = @intCast(src1.nb[2] / ts_src1);
    const s2: i64 = @intCast(dst.nb[2] / ts_dst);
    const s03: i64 = @intCast(src0.nb[3] / ts_src0);
    const s13: i64 = @intCast(src1.nb[3] / ts_src1);
    const s3: i64 = @intCast(dst.nb[3] / ts_dst);

    const ids_s0: i64 = if (ids) |i| @intCast(i.nb[0] / c.ggml_type_size(i.type)) else 0;
    const ids_s1: i64 = if (ids) |i| @intCast(i.nb[1] / c.ggml_type_size(i.type)) else 0;

    var ids_info: IdsData = .{};
    var ids_info_ptr: ?*const IdsData = null;
    var ids_src_compact_dev = common.PoolAlloc(i32){};
    defer ids_src_compact_dev.deinit();
    var ids_dst_compact_dev = common.PoolAlloc(i32){};
    defer ids_dst_compact_dev.deinit();
    var expert_bounds_dev = common.PoolAlloc(i32){};
    defer expert_bounds_dev.deinit();

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const has_ids = ids != null;
    const ncols_dst = if (has_ids) ne2 else ne1;
    const nchannels_dst = if (has_ids) ne1 else ne2;

    const stride_col_dst = if (has_ids) s2 else s1;
    const stride_col_y = if (has_ids) s12 else s11;
    const stride_channel_dst = if (has_ids) s1 else s2;

    var stride_channel_y = if (has_ids) s11 else s12;
    var nchannels_y = if (has_ids) ne11 else ne12;

    //mul_mat_id: handle broadcast
    if (has_ids and nchannels_y == 1) {
        stride_channel_y = 0;
        nchannels_y = ids.?.ne[0];
    }

    if (has_ids and ncols_dst > 16) {
        const n_expert_used = ids.?.ne[0];
        const n_experts = ne02;
        const n_tokens = ne12;
        const ne_get_rows = n_tokens * n_expert_used;

        _ = ids_src_compact_dev.allocFrom(ctx.pool(), @intCast(ne_get_rows));
        _ = ids_dst_compact_dev.allocFrom(ctx.pool(), @intCast(ne_get_rows));
        _ = expert_bounds_dev.allocFrom(ctx.pool(), @intCast(n_experts + 1));

        const si1: c_int = i32Of(ids_s1);
        const sis1: c_int = @intCast(src1.nb[2] / src1.nb[1]);

        impl.assert(sis1 > 0, "sis1 > 0");

        mmid.launchMmIdsHelperDispatch(ids_d, &ids_src_compact_dev.ptr.?[0], &ids_dst_compact_dev.ptr.?[0], &expert_bounds_dev.ptr.?[0], i32Of(n_experts), i32Of(n_tokens), i32Of(n_expert_used), i32Of(ne11), si1, sis1, false, ctx.stream());
        common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());

        ids_info.ids_src_compact = &ids_src_compact_dev.ptr.?[0];
        ids_info.ids_dst_compact = &ids_dst_compact_dev.ptr.?[0];
        ids_info.expert_bounds_dev = &expert_bounds_dev.ptr.?[0];
        ids_info.n_experts = i32Of(n_experts);
        ids_info.sis1 = sis1;
        ids_info_ptr = &ids_info;
    }

    const device = common.getDevice();
    const x_cc = common.info().devices[@intCast(device)].cc;
    const rows_per_block = getRowsPerBlock(x_cc);

    const stream = ctx.stream();
    switch (src0.type) {
        c.GGML_TYPE_F32 => {
            const vals_per_t = 1;
            switchColsPerBlock(f32, rows_per_block, src0.data, src1_d, ids_d, dst_d, @divTrunc(ne00, vals_per_t), ne01, ncols_dst, @divTrunc(s01, vals_per_t), @divTrunc(stride_col_y, vals_per_t), stride_col_dst, ids_s0, i32Of(ids_s1), ne02, nchannels_y, nchannels_dst, @divTrunc(s02, vals_per_t), stride_channel_y, stride_channel_dst, ne03, ne3, @divTrunc(s03, vals_per_t), s13, s3, stream, ids_info_ptr);
        },
        c.GGML_TYPE_F16 => {
            const vals_per_t = 2;
            switchColsPerBlock(kernels.half2, rows_per_block, src0.data, src1_d, ids_d, dst_d, @divTrunc(ne00, vals_per_t), ne01, ncols_dst, @divTrunc(s01, vals_per_t), @divTrunc(stride_col_y, vals_per_t), stride_col_dst, ids_s0, i32Of(ids_s1), ne02, nchannels_y, nchannels_dst, @divTrunc(s02, vals_per_t), stride_channel_y, stride_channel_dst, ne03, ne3, @divTrunc(s03, vals_per_t), s13, s3, stream, ids_info_ptr);
        },
        c.GGML_TYPE_BF16 => {
            const vals_per_t = 2;
            switchColsPerBlock(kernels.bf162, rows_per_block, src0.data, src1_d, ids_d, dst_d, @divTrunc(ne00, vals_per_t), ne01, ncols_dst, @divTrunc(s01, vals_per_t), @divTrunc(stride_col_y, vals_per_t), stride_col_dst, ids_s0, i32Of(ids_s1), ne02, nchannels_y, nchannels_dst, @divTrunc(s02, vals_per_t), stride_channel_y, stride_channel_dst, ne03, ne3, @divTrunc(s03, vals_per_t), s13, s3, stream, ids_info_ptr);
        },
        else => {
            impl.logError("unsupported type: %s\n", .{c.ggml_type_name(src0.type)});
            impl.abort("unsupported type");
        },
    }
}

/// Ports `ggml_cuda_should_use_mmf` (mmf.cu:133 @5266f24da), NVIDIA arms.
pub fn shouldUseMmf(ty: c.enum_ggml_type, x: c_int, warp_size: c_int, src0_ne: *const [4]i64, src0_nb: *const [4]usize, src1_ncols: c_int, mul_mat_id: bool) bool {
    if (c.ggml_is_quantized(ty)) return false;

    const ts = c.ggml_type_size(ty);
    if (@rem(src0_ne[0], @as(i64, warp_size) * @as(i64, @intCast(4 / ts))) != 0) return false;

    if (src0_nb[0] != ts) return false;

    // Pointers not aligned to the size of half2/nv_bfloat162/float2 would
    // result in a crash:
    for (1..c.GGML_MAX_DIMS) |i| {
        if (src0_nb[i] % (2 * ts) != 0) return false;
    }
    if (@rem(src0_ne[1], getRowsPerBlock(x)) != 0) return false;

    if (mul_mat_id) {
        if (src0_ne[1] <= 1024 and src1_ncols > 512) {
            return false;
        } else if (src0_ne[1] > 1024 and src1_ncols > 128) {
            return false;
        }
    } else {
        if (src1_ncols > 16) return false;
    }

    return switch (ty) {
        c.GGML_TYPE_F32 => common.ampereMmaAvailable(x),
        c.GGML_TYPE_F16 => common.voltaMmaAvailable(x) or common.turingMmaAvailable(x),
        c.GGML_TYPE_BF16 => common.ampereMmaAvailable(x),
        else => false,
    };
}
