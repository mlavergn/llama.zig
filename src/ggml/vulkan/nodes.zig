//! One graph node at a time: the synchronisation each node needs against
//! the ones still in flight, the dispatch to its op, and the submission of
//! a finished batch.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.4.1
//! (`b29c606e2`).

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const common = m.common;
const vk = common.vk;
const allocator = common.allocator;
const BackendCtx = m.context.BackendCtx;
const Context = m.submit.Context;
const CGraph = impl.CGraph;
const ops = m.ops;

fn oom() noreturn {
    impl.abort("ggml_vulkan: out of memory");
}

fn nodeAt(cgraph: *const CGraph, i: c_int) *c.ggml_tensor {
    return cgraph.nodes[@intCast(i)].?;
}

/// Ports `ggml_vk_preallocate_buffers` (ggml-vulkan.cpp:15842 @b29c606e2):
/// grows the scratch buffers to the sizes the recorded ops asked for,
/// first submitting and waiting for whatever already uses them.
pub fn preallocateBuffers(ctx: *BackendCtx, subctx: ?*Context) void {
    if (subctx) |sc| {
        // Submit and wait for any pending work before reallocating the buffers
        m.submit.ctxEnd(sc);
        m.submit.submit(sc, null);
        ctx.submit_pending = true;
        m.context.synchronize(ctx);
        impl.assert(ctx.compute_ctx == null, "ctx->compute_ctx.expired()");
        m.submit.ctxBegin(ctx.device, sc);
        ctx.compute_ctx = sc;
    }

    if (ctx.prealloc_x == null or (ctx.prealloc_size_x > 0 and ctx.prealloc_x.?.size < ctx.prealloc_size_x)) {
        m.buffer.logMemory("ggml_vk_preallocate_buffers(x_size: %zu)", .{ctx.prealloc_size_x});
        m.buffer.destroyBuffer(&ctx.prealloc_x);
        ctx.prealloc_x = createOrAbort(ctx, ctx.prealloc_size_x);
    }
    if (ctx.prealloc_y == null or (ctx.prealloc_size_y > 0 and ctx.prealloc_y.?.size < ctx.prealloc_size_y)) {
        m.buffer.logMemory("ggml_vk_preallocate_buffers(y_size: %zu)", .{ctx.prealloc_size_y});
        m.buffer.destroyBuffer(&ctx.prealloc_y);
        ctx.prealloc_y = createOrAbort(ctx, ctx.prealloc_size_y);
        ctx.prealloc_y_last_pipeline_used = null;
        ctx.prealloc_y_last_tensor_used = null;
        ctx.prealloc_y_last_k_padded = false;
    }
    if (ctx.prealloc_split_k == null or (ctx.prealloc_size_split_k > 0 and ctx.prealloc_split_k.?.size < ctx.prealloc_size_split_k)) {
        m.buffer.logMemory("ggml_vk_preallocate_buffers(split_k_size: %zu)", .{ctx.prealloc_size_split_k});
        m.buffer.destroyBuffer(&ctx.prealloc_split_k);
        ctx.prealloc_split_k = createOrAbort(ctx, ctx.prealloc_size_split_k);
    }
    if (ctx.prealloc_add_rms_partials == null or (ctx.prealloc_size_add_rms_partials > 0 and ctx.prealloc_add_rms_partials.?.size < ctx.prealloc_size_add_rms_partials)) {
        // The C++ streams the buffer's shared pointer here, not its size.
        m.buffer.logMemory("ggml_vk_preallocate_buffers(add_partials_size: %p)", .{@as(?*anyopaque, ctx.prealloc_add_rms_partials)});
        m.buffer.destroyBuffer(&ctx.prealloc_add_rms_partials);
        ctx.prealloc_add_rms_partials = createOrAbort(ctx, ctx.prealloc_size_add_rms_partials);
    }
}

/// `ggml_vk_create_buffer_device` where nothing catches its exception.
fn createOrAbort(ctx: *BackendCtx, size: usize) *m.buffer.Buffer {
    return m.buffer.createBufferDevice(ctx.device, size) catch common.abortLastError();
}

/// The dependency check at the top of `ggml_vk_build_graph`: whether a
/// tensor overlaps, in the same buffer, any tensor in `unsynced`.
fn overlapsUnsynced(node: *const c.ggml_tensor, unsynced: []const *const c.ggml_tensor) bool {
    if (unsynced.len == 0) return false;
    const n_base = m.buffer.tensorOffset(node) + node.view_offs;
    const n_size = c.ggml_nbytes(node);
    const a_buf = m.buffer.tensorBuffer(node);
    for (unsynced) |other| {
        if (a_buf != m.buffer.tensorBuffer(other)) continue;
        const o_base = m.buffer.tensorOffset(other) + other.view_offs;
        const o_size = c.ggml_nbytes(other);
        if ((o_base <= n_base and n_base < o_base + o_size) or (n_base <= o_base and o_base < n_base + n_size)) return true;
    }
    return false;
}

/// Ports `ggml_vk_build_graph` (ggml-vulkan.cpp:15984 @b29c606e2): records
/// one node (or fused run) into the compute context, after a barrier if it
/// depends on unsynchronised work; submits the batch if asked.
///
/// Return: whether the node enqueued work.
pub fn buildGraph(ctx: *BackendCtx, cgraph: *CGraph, node_idx: c_int, node_idx_begin: c_int, last_node: bool, almost_ready: bool, submit: bool) bool {
    const node = nodeAt(cgraph, node_idx);
    if (c.ggml_is_empty(node) or impl.opIsEmpty(node.op) or node.buffer == null) return false;
    if ((node.flags & c.GGML_TENSOR_FLAG_COMPUTE) == 0) return false;

    ctx.semaphore_idx = 0;

    const src0 = node.src[0];
    const src1 = node.src[1];
    const src2 = node.src[2];
    const src3 = node.src[3];

    if (node.op == c.GGML_OP_ADD) {
        const next_node_idx = node_idx + 1 + ctx.num_additional_fused_ops;
        if (next_node_idx < cgraph.n_nodes and
            nodeAt(cgraph, next_node_idx).op == c.GGML_OP_RMS_NORM and
            nodeAt(cgraph, next_node_idx).src[0] == nodeAt(cgraph, next_node_idx - 1) and
            c.ggml_nrows(nodeAt(cgraph, next_node_idx)) == 1 and
            ctx.device.add_rms_fusion)
        {
            const size = ops.rmsPartialsSize(ctx, nodeAt(cgraph, node_idx));
            ctx.do_add_rms_partials_offset_calculation = true;
            if (ctx.prealloc_size_add_rms_partials_offset + size <= ctx.prealloc_size_add_rms_partials) ctx.do_add_rms_partials = true;
        }
    }

    const compute_ctx = m.context.getComputeCtx(ctx);

    {
        // A node needs a barrier if it overlaps an unsynchronised node and
        // one of the two writes. Destinations are checked against both
        // lists, sources only against the written one.
        var need_sync = false;
        var i: c_int = 0;
        while (i < ctx.num_additional_fused_ops + 1 and !need_sync) : (i += 1) {
            const cur_node = nodeAt(cgraph, node_idx + i);
            // If the node actually writes to memory, then check if it needs to sync
            if (ctx.fused_ops_write_mask & (@as(c_int, 1) << @intCast(i)) != 0) {
                if (overlapsUnsynced(cur_node, ctx.unsynced_nodes_read.items) or overlapsUnsynced(cur_node, ctx.unsynced_nodes_written.items)) {
                    need_sync = true;
                    break;
                }
            }
            for (0..c.GGML_MAX_SRC) |j| {
                const s = cur_node.src[j] orelse continue;
                if (overlapsUnsynced(s, ctx.unsynced_nodes_written.items)) {
                    need_sync = true;
                    break;
                }
            }
        }

        if (need_sync) {
            if (m.instance.loggers.enable_sync_logger) impl.fprintfStderr("sync\n", .{});
            ctx.unsynced_nodes_written.clearRetainingCapacity();
            ctx.unsynced_nodes_read.clearRetainingCapacity();
            m.context.syncBuffers(ctx, compute_ctx);

            if (m.instance.loggers.perf_logger_enabled and m.instance.loggers.perf_logger_concurrent) {
                ctx.query_node_idx.items[@intCast(ctx.query_idx)] = node_idx;
                m.graph.writeTimestamp(ctx, compute_ctx);
                m.context.syncBuffers(ctx, compute_ctx);
            }
        }
        // Add all fused nodes to the unsynchronized lists.
        i = 0;
        while (i < ctx.num_additional_fused_ops + 1) : (i += 1) {
            const cur_node = nodeAt(cgraph, node_idx + i);
            // Multiple outputs could be written, e.g. in topk_moe.
            if (ctx.fused_ops_write_mask & (@as(c_int, 1) << @intCast(i)) != 0) ctx.unsynced_nodes_written.append(allocator, cur_node) catch oom();
            for (0..c.GGML_MAX_SRC) |j| {
                const s = cur_node.src[j] orelse continue;
                ctx.unsynced_nodes_read.append(allocator, s) catch oom();
            }
        }
    }
    if (m.instance.loggers.enable_sync_logger) syncLogNodes(ctx, cgraph, node_idx);

    // closed explicitly below, and by the defer on the early return
    var dbg = m.submit.DebugLabel.node(compute_ctx, cgraph, node_idx, ctx.num_additional_fused_ops);
    defer dbg.close();

    if (!ops.dispatchNode(ctx, compute_ctx, cgraph, node_idx, node, src0, src1, src2, src3)) return false;

    // the submit path below can end the command buffer, so close the region first
    dbg.close();

    ctx.tensor_ctxs.items[@intCast(node_idx)] = compute_ctx;
    if (submit or last_node) {
        m.submit.ctxEnd(compute_ctx);
        compute_ctx.exit_tensor_idx = if (last_node) node_idx_begin else -1;
        ctx.compute_ctx = null;
        computeForward(ctx, cgraph, node_idx_begin, almost_ready);
    }
    return true;
}

/// The `vk_enable_sync_logger` listing of the nodes being recorded.
fn syncLogNodes(ctx: *BackendCtx, cgraph: *const CGraph, node_idx: c_int) void {
    var i: c_int = 0;
    while (i < ctx.num_additional_fused_ops + 1) : (i += 1) {
        const n = nodeAt(cgraph, node_idx + i);
        impl.fprintfStderr("%d %s %s", .{ node_idx + i, c.ggml_op_name(n.op), @as([*:0]const u8, @ptrCast(&n.name)) });
        if (n.op == c.GGML_OP_GLU) {
            impl.fprintfStderr(" %s %s ", .{ c.ggml_glu_op_name(c.ggml_get_glu_op(n)), @as([*:0]const u8, if (n.src[1] != null) "split" else "single") });
        }
        if (n.op == c.GGML_OP_ROPE) impl.fprintfStderr(" rope mode: %d", .{n.op_params[2]});
        impl.fprintfStderr("\n", .{});
    }
}

/// Ports `ggml_vk_compute_forward` (ggml-vulkan.cpp:16557 @b29c606e2):
/// submits the context node `tensor_idx` was recorded into, unless it was
/// already, and runs the host copies of a finished one.
pub fn computeForward(ctx: *BackendCtx, cgraph: *const CGraph, tensor_idx: c_int, almost_ready: bool) void {
    _ = cgraph;
    const subctx = ctx.tensor_ctxs.items[@intCast(tensor_idx)].?;
    // Only run if ctx hasn't been submitted yet
    if (subctx.seqs.items.len != 0) {
        // Do staging buffer copies
        m.submit.runMemcpys(subctx.in_memcpys.items);
        m.submit.runMemsets(subctx.memsets.items);
        if (ctx.device.serialize_submissions) {
            m.submit.submit(subctx, ctx.fence);
        } else if (almost_ready and !ctx.almost_ready_fence_pending) {
            m.submit.submit(subctx, ctx.almost_ready_fence);
            ctx.almost_ready_fence_pending = true;
        } else {
            m.submit.submit(subctx, null);
        }
        ctx.submit_pending = true;
    }
    if (tensor_idx == subctx.exit_tensor_idx) {
        // Do staging buffer copies
        m.submit.runMemcpys(subctx.out_memcpys.items);
        subctx.in_memcpys.clearRetainingCapacity();
        subctx.out_memcpys.clearRetainingCapacity();
        subctx.memsets.clearRetainingCapacity();
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
