//! The expert-routing helper `MUL_MAT_ID`'s matrix kernels share: for each
//! expert, the tokens routed to it and where their results go.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/mmid.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/mmid.cuh` at v1.0.0 (`b1b770d90`). The
//! kernel, `mm_ids_helper`, stays CUDA. It writes one
//! `ggml_cuda_mm_id_tile_spec` per column tile of `tile_width` instead of each expert's bounds.

const ggml = @import("../module.zig");
/// Ports `ggml_cuda_mm_id_tile_spec` (mmid.cuh:3 @b1b770d90): the expert and
/// the column range of one tile. The host only allocates them.
pub const TileSpec = extern struct {
    expert: i32,
    col_low: i32,
    col_high: i32,
    padding: i32,

    comptime {
        if (@sizeOf(TileSpec) != 16) @compileError("bad ggml_cuda_mm_id_tile_spec size");
    }
};

/// Ports `launch_mm_ids_helper` (mmid.cu:143 @b1b770d90).
fn launchMmIdsHelper(
    comptime n_expert_used_template: comptime_int,
    ids: ?*const i32,
    ids_src1: ?*i32,
    ids_dst: ?*i32,
    tile_spec: ?*TileSpec,
    n_experts: c_int,
    n_tokens: c_int,
    n_expert_used_var: c_int,
    nchannels_y: c_int,
    tile_width: c_int,
    si1: c_int,
    sis1: c_int,
    write_inverse: bool,
    stream: ggml.cuda.common.cu.cudaStream_t,
) void {
    ggml.impl.assert(n_tokens < (1 << 22), "n_tokens          < (1 << 22) && \"too few bits in mm_ids_helper_store\"");
    ggml.impl.assert(n_expert_used_var < (1 << 10), "n_expert_used_var < (1 << 10) && \"too few bits in mm_ids_helper_store\"");

    const id = ggml.cuda.common.getDevice();
    const warp_size = ggml.cuda.common.info().devices[@intCast(id)].warp_size;
    const smpbo = ggml.cuda.common.info().devices[@intCast(id)].smpbo;
    const k = comptime ggml.cuda.kernels.get("mmid", ggml.cuda.kernels.tmpl("mm_ids_helper", &.{ggml.cuda.kernels.int(n_expert_used_template)}));
    ggml.cuda.kernels.setSharedMemoryLimit(k, smpbo);

    const tile_width_fd = ggml.cuda.common.initFastdivValues(@bitCast(@as(i64, tile_width)));

    // `n_tokens*sizeof(mm_ids_helper_store)`, a 4-byte struct.
    const nbytes_shared: usize = @as(usize, @intCast(n_tokens)) * 4;
    ggml.impl.assert(nbytes_shared <= smpbo, "nbytes_shared <= smpbo");
    ggml.cuda.kernels.launch(k, .{
        .grid = .{ .x = @intCast(n_experts) },
        .block = .{ .x = @intCast(warp_size) },
        .shmem = nbytes_shared,
        .stream = stream,
    }, .{ ids, ids_src1, ids_dst, tile_spec, n_tokens, n_expert_used_var, nchannels_y, tile_width_fd, si1, sis1, write_inverse });
}

/// Ports `ggml_cuda_launch_mm_ids_helper` (mmid.cu:166 @b1b770d90).
pub fn launchMmIdsHelperDispatch(
    ids: ?*const i32,
    ids_src1: ?*i32,
    ids_dst: ?*i32,
    tile_spec: ?*TileSpec,
    n_experts: c_int,
    n_tokens: c_int,
    n_expert_used: c_int,
    nchannels_y: c_int,
    tile_width: c_int,
    si1: c_int,
    sis1: c_int,
    write_inverse: bool,
    stream: ggml.cuda.common.cu.cudaStream_t,
) void {
    switch (n_expert_used) {
        inline 2, 4, 6, 8, 10, 16, 32 => |n| launchMmIdsHelper(n, ids, ids_src1, ids_dst, tile_spec, n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream),
        else => launchMmIdsHelper(0, ids, ids_src1, ids_dst, tile_spec, n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream),
    }
}
