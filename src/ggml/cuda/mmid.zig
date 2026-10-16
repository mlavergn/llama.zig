//! The expert-routing helper `MUL_MAT_ID`'s matrix kernels share: for each
//! expert, the tokens routed to it and where their results go.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/mmid.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/mmid.cuh` at v0.3.0 (`c1d0e7a00`). The
//! kernel, `mm_ids_helper`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const cu = common.cu;

/// Ports `launch_mm_ids_helper` (mmid.cu:124 @c1d0e7a00).
fn launchMmIdsHelper(
    comptime n_expert_used_template: comptime_int,
    ids: ?*const i32,
    ids_src1: ?*i32,
    ids_dst: ?*i32,
    expert_bounds: ?*i32,
    n_experts: c_int,
    n_tokens: c_int,
    n_expert_used_var: c_int,
    nchannels_y: c_int,
    si1: c_int,
    sis1: c_int,
    write_inverse: bool,
    stream: cu.cudaStream_t,
) void {
    impl.assert(n_tokens < (1 << 22), "n_tokens          < (1 << 22) && \"too few bits in mm_ids_helper_store\"");
    impl.assert(n_expert_used_var < (1 << 10), "n_expert_used_var < (1 << 10) && \"too few bits in mm_ids_helper_store\"");

    const id = common.getDevice();
    const warp_size = common.info().devices[@intCast(id)].warp_size;
    const smpbo = common.info().devices[@intCast(id)].smpbo;
    const k = comptime kernels.get("mmid", kernels.tmpl("mm_ids_helper", &.{kernels.int(n_expert_used_template)}));
    kernels.setSharedMemoryLimit(k, smpbo);

    // `n_tokens*sizeof(mm_ids_helper_store)`, a 4-byte struct.
    const nbytes_shared: usize = @as(usize, @intCast(n_tokens)) * 4;
    impl.assert(nbytes_shared <= smpbo, "nbytes_shared <= smpbo");
    kernels.launch(k, .{
        .grid = .{ .x = @intCast(n_experts) },
        .block = .{ .x = @intCast(warp_size) },
        .shmem = nbytes_shared,
        .stream = stream,
    }, .{ ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse });
}

/// Ports `ggml_cuda_launch_mm_ids_helper` (mmid.cu:143 @c1d0e7a00).
pub fn launchMmIdsHelperDispatch(
    ids: ?*const i32,
    ids_src1: ?*i32,
    ids_dst: ?*i32,
    expert_bounds: ?*i32,
    n_experts: c_int,
    n_tokens: c_int,
    n_expert_used: c_int,
    nchannels_y: c_int,
    si1: c_int,
    sis1: c_int,
    write_inverse: bool,
    stream: cu.cudaStream_t,
) void {
    switch (n_expert_used) {
        inline 2, 4, 6, 8, 16, 32 => |n| launchMmIdsHelper(n, ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream),
        else => launchMmIdsHelper(0, ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream),
    }
}
