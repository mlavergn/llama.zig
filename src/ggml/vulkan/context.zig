//! The backend context: one per `ggml_backend_vk_init`, holding the
//! device, the command pools and contexts recorded into, the preallocated
//! scratch buffers, the descriptor sets, and the state fusion and
//! synchronisation track between nodes.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.4.1
//! (`b29c606e2`). The C++'s `vk_context_ref`s -- weak pointers to contexts
//! the garbage list owns -- are plain pointers here, nulled where the list
//! is cleared (`graphCleanup`), the only place those contexts die.

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const common = m.common;
const vk = common.vk;
const d = &common.dispatch;
const allocator = common.allocator;
const Device = m.device.Device;
const CommandPool = m.device.CommandPool;
const Buffer = m.buffer.Buffer;
const Pipeline = m.pipeline.Pipeline;
const Context = m.submit.Context;
const Semaphore = m.submit.Semaphore;

fn oom() noreturn {
    impl.abort("ggml_vulkan: out of memory");
}

/// Ports `topk_moe_mode` (ggml-vulkan.cpp:795 @b29c606e2).
/// `count` is `TOPK_MOE_COUNT`, which the C++ uses for "no topk_moe fusion".
pub const TopkMoeMode = enum(u32) { early_softmax, early_softmax_norm, late_softmax, sigmoid_norm_bias, sqrt_softplus_norm_bias, count };

/// Ports `rms_norm_mode` (ggml-vulkan.cpp:804 @b29c606e2): which fusion the
/// RMS_NORM being processed heads. `count` is `RMS_NORM_COUNT`, "no
/// rms_norm fusion".
pub const RmsNormMode = enum(u32) { mul, mul_add, mul_add_mul, mul_rope, mul_rope_view_set_rows, view_set_rows, count };

/// Ports `ggml_vk_garbage_collector` (ggml-vulkan.cpp:2233 @b29c606e2).
/// Semaphores are handed out by pointer and must not move, so each is its
/// own allocation.
pub const GarbageCollector = struct {
    tl_semaphores: std.ArrayList(*Semaphore) = .empty,
    semaphores: std.ArrayList(*Semaphore) = .empty,
    events: std.ArrayList(vk.VkEvent) = .empty,
    contexts: std.ArrayList(*Context) = .empty,
};

/// Ports `ggml_backend_vk_context` (ggml-vulkan.cpp:2506 @b29c606e2).
pub const BackendCtx = struct {
    name: [:0]const u8 = "",
    device: *Device = undefined,

    semaphore_idx: usize = 0,
    event_idx: usize = 0,
    gc: GarbageCollector = .{},
    prealloc_size_x: usize = 0,
    prealloc_size_y: usize = 0,
    prealloc_size_split_k: usize = 0,
    prealloc_size_add_rms_partials: usize = 0,
    prealloc_size_add_rms_partials_offset: usize = 0,
    prealloc_x: ?*Buffer = null,
    prealloc_y: ?*Buffer = null,
    prealloc_split_k: ?*Buffer = null,
    prealloc_add_rms_partials: ?*Buffer = null,
    sync_staging: ?*Buffer = null,
    fence: vk.VkFence = null,
    almost_ready_fence: vk.VkFence = null,
    submit_pending: bool = false,
    almost_ready_fence_pending: bool = false,
    // Set before op_add and unset after op_rms_norm to indicate that the add
    // should write partial sums to accumulate the square of the vector components
    do_add_rms_partials_offset_calculation: bool = false,
    do_add_rms_partials: bool = false,

    last_total_flops: u64 = std.math.maxInt(u64),

    // Cache most recent tensor that was converted into prealloc_y, and what pipeline it used to convert.
    prealloc_y_last_pipeline_used: ?*Pipeline = null,
    prealloc_y_last_tensor_used: ?*const c.ggml_tensor = null,
    // True when the K dimension in prealloc_y is padded.
    prealloc_y_last_k_padded: bool = false,

    // Track which nodes have been used since the last sync, and whether they were written to
    unsynced_nodes_written: std.ArrayList(*const c.ggml_tensor) = .empty,
    unsynced_nodes_read: std.ArrayList(*const c.ggml_tensor) = .empty,
    // Track which prealloc buffers have pending reads that need to be synchronized.
    prealloc_x_need_sync: bool = false,
    prealloc_y_need_sync: bool = false,
    prealloc_split_k_need_sync: bool = false,

    compute_ctx: ?*Context = null,

    transfer_ctx: ?*Context = null,
    transfer_semaphore: Semaphore = .{},
    transfer_semaphore_last_submitted: u64 = 0,

    tensor_ctxs: std.ArrayList(?*Context) = .empty,

    descriptor_pools: std.ArrayList(vk.VkDescriptorPool) = .empty,
    descriptor_sets: std.ArrayList(vk.VkDescriptorSet) = .empty,
    descriptor_set_idx: u32 = 0,
    pipeline_descriptor_set_requirements: u32 = 0,

    compute_cmd_pool: CommandPool = .{},
    transfer_cmd_pool: CommandPool = .{},

    // number of additional consecutive nodes fused with the node being processed
    num_additional_fused_ops: c_int = 0,
    // Bit 'i' means nodes[start_of_fusion + i] writes to memory.
    fused_ops_write_mask: c_int = 0,
    fused_topk_moe_mode: TopkMoeMode = .early_softmax,
    fused_topk_moe_scale: bool = false,
    // QSA indexer gather+add+top_k fused into one radix-select
    fused_topk_qsa: bool = false,
    fused_rms_norm_mode: RmsNormMode = .count,

    // for GGML_VK_PERF_LOGGER
    perf_logger: ?*m.perf.PerfLogger = null,
    query_pool: vk.VkQueryPool = null,
    query_fusion_names: std.ArrayList(?[*:0]const u8) = .empty,
    query_fusion_node_count: std.ArrayList(c_int) = .empty,
    query_nodes: std.ArrayList(?*c.ggml_tensor) = .empty,
    query_node_idx: std.ArrayList(c_int) = .empty,
    num_queries: i32 = 0,
    query_idx: i32 = 0,
};

/// Ports `ggml_vk_init` (ggml-vulkan.cpp:7964 @b29c606e2).
pub fn init(ctx: *BackendCtx, idx: usize) void {
    m.instance.instanceInitOrAbort();
    impl.assert(idx < m.instance.instance.device_indices.items.len, "idx < vk_instance.device_indices.size()");

    ctx.name = std.fmt.allocPrintSentinel(allocator, common.name ++ "{d}", .{idx}, 0) catch oom();
    ctx.device = m.device.getDevice(idx);

    ctx.semaphore_idx = 0;
    ctx.event_idx = 0;
    ctx.prealloc_size_x = 0;
    ctx.prealloc_size_y = 0;
    ctx.prealloc_size_split_k = 0;
    // Fixed size of 1KB, for deterministic behavior
    ctx.prealloc_size_add_rms_partials = 1024;

    var fci = common.structure(vk.VkFenceCreateInfo, vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    common.checkOrAbort(d.vkCreateFence.?(ctx.device.device, &fci, null, &ctx.fence), "vk::Device::createFence");
    common.checkOrAbort(d.vkCreateFence.?(ctx.device.device, &fci, null, &ctx.almost_ready_fence), "vk::Device::createFence");

    ctx.compute_cmd_pool.init(ctx.device, ctx.device.compute_queue.?);
    if (ctx.device.async_use_transfer_queue) {
        ctx.transfer_semaphore.s = createSemaphore(ctx.device, vk.VK_SEMAPHORE_TYPE_TIMELINE);
        ctx.transfer_semaphore.value = 0;
        ctx.transfer_cmd_pool.init(ctx.device, ctx.device.transfer_queue.?);
    }

    if (m.instance.loggers.perf_logger_enabled) ctx.perf_logger = m.perf.PerfLogger.create();
}

fn createSemaphore(device: *Device, ty: vk.VkSemaphoreType) vk.VkSemaphore {
    var tci = common.structure(vk.VkSemaphoreTypeCreateInfo, vk.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO);
    tci.semaphoreType = ty;
    tci.initialValue = 0;
    var ci = common.structure(vk.VkSemaphoreCreateInfo, vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    ci.pNext = &tci;
    var s: vk.VkSemaphore = null;
    common.checkOrAbort(d.vkCreateSemaphore.?(device.device, &ci, null, &s), "vk::Device::createSemaphore");
    return s;
}

/// Ports `ggml_vk_create_context` (ggml-vulkan.cpp:3558 @b29c606e2): a
/// context the garbage list owns until the next graph cleanup.
pub fn createContext(ctx: *BackendCtx, p: *CommandPool) *Context {
    const result = m.submit.createTemporaryContext(p);
    ctx.gc.contexts.append(allocator, result) catch oom();
    return result;
}

/// Ports `ggml_vk_create_binary_semaphore` (ggml-vulkan.cpp:3573 @b29c606e2).
pub fn createBinarySemaphore(ctx: *BackendCtx) *Semaphore {
    const s = allocator.create(Semaphore) catch oom();
    s.* = .{ .s = createSemaphore(ctx.device, vk.VK_SEMAPHORE_TYPE_BINARY), .value = 0 };
    ctx.gc.semaphores.append(allocator, s) catch oom();
    return s;
}

/// Ports `ggml_vk_create_timeline_semaphore` (ggml-vulkan.cpp:3583 @b29c606e2).
pub fn createTimelineSemaphore(ctx: *BackendCtx) *Semaphore {
    if (ctx.semaphore_idx >= ctx.gc.tl_semaphores.items.len) {
        const s = allocator.create(Semaphore) catch oom();
        s.* = .{ .s = createSemaphore(ctx.device, vk.VK_SEMAPHORE_TYPE_TIMELINE), .value = 0 };
        ctx.gc.tl_semaphores.append(allocator, s) catch oom();
    }
    defer ctx.semaphore_idx += 1;
    return ctx.gc.tl_semaphores.items[ctx.semaphore_idx];
}

/// Ports `ggml_vk_create_event` (ggml-vulkan.cpp:3595 @b29c606e2).
pub fn createEvent(ctx: *BackendCtx) vk.VkEvent {
    if (ctx.event_idx >= ctx.gc.events.items.len) {
        var ci = common.structure(vk.VkEventCreateInfo, vk.VK_STRUCTURE_TYPE_EVENT_CREATE_INFO);
        var e: vk.VkEvent = null;
        common.checkOrAbort(d.vkCreateEvent.?(ctx.device.device, &ci, null, &e), "vk::Device::createEvent");
        ctx.gc.events.append(allocator, e) catch oom();
    }
    defer ctx.event_idx += 1;
    return ctx.gc.events.items[ctx.event_idx];
}

/// Ports `ggml_vk_sync_buffers` (ggml-vulkan.cpp:3864 @b29c606e2): a full
/// memory barrier, after which no preallocated buffer has a read pending.
pub fn syncBuffers(ctx: ?*BackendCtx, subctx: *Context) void {
    if (ctx) |cx| {
        cx.prealloc_x_need_sync = false;
        cx.prealloc_y_need_sync = false;
        cx.prealloc_split_k_need_sync = false;
    }
    m.submit.syncBuffersBarrier(subctx);
}

/// Ports `ggml_vk_get_max_buffer_range` (ggml-vulkan.cpp:2812 @b29c606e2).
pub fn getMaxBufferRange(ctx: *const BackendCtx, buf: *const Buffer, offset: u64) u64 {
    return @min(buf.size - offset, ctx.device.properties.limits.maxStorageBufferRange);
}

/// Ports `ggml_vk_subbuffer` (ggml-vulkan.cpp:3860 @b29c606e2).
pub fn subbuffer(ctx: *const BackendCtx, buf: *Buffer, offset: u64) m.buffer.Subbuffer {
    return .{ .buffer = buf, .offset = offset, .size = getMaxBufferRange(ctx, buf, offset) };
}

/// Ports `ggml_vk_tensor_buffer_offset` (ggml-vulkan.cpp:2590 @b29c606e2):
/// a tensor's offset within its Vulkan buffer. `vk_tensor_offset` is
/// relative to `vk_ptr_base`, but mapped host tensors need an offset
/// relative to their own buffer.
fn tensorBufferOffset(ctx: *const BackendCtx, t: *const c.ggml_tensor) usize {
    if (ctx.device.uma) {
        var off: usize = 0;
        if (m.buffer.hostGet(ctx.device, t.data, &off) != null) return off;
    }
    return @intCast(m.buffer.tensorOffset(t) + t.view_offs);
}

/// Ports `ggml_vk_descriptor_offset` (ggml-vulkan.cpp:2603 @b29c606e2): moves
/// the descriptor back until its distance to the tensor is divisible by the
/// tensor type size.
fn descriptorOffset(tensor_offset: usize, alignment: usize, type_size: usize) usize {
    var descriptor_offset = tensor_offset & ~(alignment -% 1);
    while ((tensor_offset -% descriptor_offset) % type_size != 0) {
        impl.assert(descriptor_offset >= alignment, "descriptor_offset >= alignment");
        descriptor_offset -%= alignment;
    }
    return descriptor_offset;
}

/// Ports `get_misalign_bytes` (ggml-vulkan.cpp:2614 @b29c606e2).
pub fn getMisalignBytes(ctx: *const BackendCtx, t: *const c.ggml_tensor) u32 {
    const tensor_offset = tensorBufferOffset(ctx, t);
    const descriptor_offset = descriptorOffset(tensor_offset, ctx.device.properties.limits.minStorageBufferOffsetAlignment, c.ggml_type_size(t.type));
    impl.assert(tensor_offset - descriptor_offset <= std.math.maxInt(u32), "tensor_offset - descriptor_offset <= UINT32_MAX");
    return @intCast(tensor_offset - descriptor_offset);
}

/// Ports `ggml_vk_tensor_subbuffer` (ggml-vulkan.cpp:8283 @b29c606e2): the
/// buffer range holding a tensor, its offset moved back to a storage
/// alignment a whole number of elements before it and its size grown to
/// match.
pub fn tensorSubbuffer(ctx: *const BackendCtx, tensor: *const c.ggml_tensor, allow_misalign: bool) m.buffer.Subbuffer {
    var buffer: ?*Buffer = null;
    var offset: usize = 0;
    if (ctx.device.uma) buffer = m.buffer.hostGet(ctx.device, tensor.data, &offset);
    if (buffer == null) {
        buffer = m.buffer.tensorBuffer(tensor);
        offset = m.buffer.tensorOffset(tensor) + tensor.view_offs;
    }
    impl.assert(buffer != null, "buffer != nullptr");

    var size = c.ggml_nbytes(tensor);
    const descriptor_offset = descriptorOffset(offset, ctx.device.properties.limits.minStorageBufferOffsetAlignment, c.ggml_type_size(tensor.type));
    const misalign_bytes = offset - descriptor_offset;
    // The shader must support misaligned offsets when indexing into the buffer
    impl.assert(allow_misalign or misalign_bytes == 0, "allow_misalign || misalign_bytes == 0");
    offset = descriptor_offset;
    size += misalign_bytes;
    return .{ .buffer = buffer, .offset = offset, .size = size };
}

/// Ports `ggml_vk_ensure_sync_staging_buffer` for a backend context
/// (ggml-vulkan.cpp:8611 @b29c606e2).
pub fn ensureSyncStagingBuffer(ctx: *BackendCtx, size: usize) void {
    if (ctx.sync_staging == null or ctx.sync_staging.?.size < size) {
        m.buffer.logMemory("ggml_vk_ensure_sync_staging_buffer(%zu)", .{size});
        m.buffer.destroyBuffer(&ctx.sync_staging);
        ctx.sync_staging = m.buffer.createBufferCheck(ctx.device, size, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
}

/// Ports `ggml_vk_get_compute_ctx` (ggml-vulkan.cpp:8524 @b29c606e2): the
/// context compute is recorded into, begun on first use; it waits for any
/// transfer submitted since.
pub fn getComputeCtx(ctx: *BackendCtx) *Context {
    const result = ctx.compute_ctx orelse blk: {
        const r = createContext(ctx, &ctx.compute_cmd_pool);
        ctx.compute_ctx = r;
        m.submit.ctxBegin(ctx.device, r);
        break :blk r;
    };
    if (ctx.device.async_use_transfer_queue and ctx.transfer_semaphore_last_submitted < ctx.transfer_semaphore.value) {
        result.s().wait_semaphores.append(allocator, ctx.transfer_semaphore) catch oom();
        ctx.transfer_semaphore_last_submitted = ctx.transfer_semaphore.value;
    }
    return result;
}

/// Ports `ggml_vk_get_transfer_ctx` (ggml-vulkan.cpp:8543 @b29c606e2).
pub fn getTransferCtx(ctx: *BackendCtx) *Context {
    if (ctx.transfer_ctx) |r| return r;
    const r = createContext(ctx, &ctx.transfer_cmd_pool);
    ctx.transfer_ctx = r;
    m.submit.ctxBegin(ctx.device, r);
    return r;
}

/// Ports `ggml_vk_submit_transfer_ctx` (ggml-vulkan.cpp:8560 @b29c606e2):
/// submits pending transfer-queue work, signalling the semaphore the next
/// compute context waits on.
///
/// Return: whether work was submitted.
pub fn submitTransferCtx(ctx: *BackendCtx) bool {
    if (!ctx.device.async_use_transfer_queue) return false;
    const cpy_ctx = ctx.transfer_ctx orelse return false;
    m.submit.ctxEnd(cpy_ctx);
    m.submit.runMemcpys(cpy_ctx.in_memcpys.items);

    ctx.transfer_semaphore.value += 1;
    const last_seq = &cpy_ctx.seqs.items[cpy_ctx.seqs.items.len - 1];
    last_seq.items[last_seq.items.len - 1].signal_semaphores.append(allocator, ctx.transfer_semaphore) catch oom();

    m.submit.submit(cpy_ctx, null);
    ctx.transfer_ctx = null;
    return true;
}

/// Ports `ggml_vk_wait_for_fence` (ggml-vulkan.cpp:2819 @b29c606e2): waits
/// on the almost-ready fence, then spins on the graph's fence.
pub fn waitForFence(ctx: *BackendCtx) void {
    const dev = ctx.device;
    // Use waitForFences while most of the graph executes.
    if (ctx.almost_ready_fence_pending) {
        m.submit.waitAndResetFence(dev, ctx.almost_ready_fence, "almost_ready_fence");
        ctx.almost_ready_fence_pending = false;
    }

    // Spin (w/pause) waiting for the graph to finish executing.
    while (true) {
        const result = d.vkGetFenceStatus.?(dev.device, ctx.fence);
        if (result == vk.VK_ERROR_DEVICE_LOST) {
            m.submit.printDeviceLostInfo(dev);
            impl.logError("ggml_vulkan: getFenceStatus\n", .{});
            common.abortLastError();
        }
        if (result == vk.VK_SUCCESS) break;
        if (result != vk.VK_NOT_READY) {
            impl.logError("ggml_vulkan: error %s\n", .{common.resultString(result).ptr});
            impl.abort("ggml_vulkan: getFenceStatus");
        }
        for (0..100) |_| {
            inline for (0..10) |_| std.atomic.spinLoopHint();
        }
    }
    common.checkOrAbort(d.vkResetFences.?(dev.device, 1, &ctx.fence), "vk::Device::resetFences");
}

/// Ports `ggml_vk_synchronize` (ggml-vulkan.cpp:17185 @b29c606e2): submits
/// the recorded compute context and any transfer, waits for both, and runs
/// the deferred host copies.
pub fn synchronize(ctx: *BackendCtx) void {
    const do_transfer = ctx.compute_ctx != null;

    if (submitTransferCtx(ctx)) ctx.submit_pending = true;

    var compute_ctx: ?*Context = null;
    var cmd_buf: ?*m.device.CommandBuffer = null;
    if (do_transfer) {
        const cc = ctx.compute_ctx.?;
        compute_ctx = cc;
        if (cc.has_s) cmd_buf = cc.s().buffer;
        m.submit.ctxEnd(cc);
        m.submit.runMemcpys(cc.in_memcpys.items);

        if (ctx.device.serialize_submissions) {
            m.submit.submit(cc, ctx.fence);
            m.submit.waitAndResetFence(ctx.device, ctx.fence, "synchronize waitForFences");
        } else {
            m.submit.submit(cc, null);
        }
        ctx.submit_pending = true;
    }

    if (ctx.submit_pending) {
        const h = ctx.device.compute_queue.?.handle;
        if (ctx.device.serialize_submissions) {
            ctx.submit_pending = false;
        } else if (ctx.device.async_use_transfer_queue and ctx.transfer_semaphore_last_submitted < ctx.transfer_semaphore.value) {
            var tl_info = common.structure(vk.VkTimelineSemaphoreSubmitInfo, vk.VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO);
            tl_info.waitSemaphoreValueCount = 1;
            tl_info.pWaitSemaphoreValues = &ctx.transfer_semaphore.value;
            const stage: vk.VkPipelineStageFlags = ctx.device.transfer_queue.?.stage_flags;
            var si = common.structure(vk.VkSubmitInfo, vk.VK_STRUCTURE_TYPE_SUBMIT_INFO);
            si.waitSemaphoreCount = 1;
            si.pWaitSemaphores = &ctx.transfer_semaphore.s;
            si.pWaitDstStageMask = &stage;
            si.pNext = &tl_info;
            m.submit.queueSubmit(h, &.{si}, ctx.fence);
            ctx.transfer_semaphore_last_submitted = ctx.transfer_semaphore.value;
        } else {
            m.submit.queueSubmit(h, &.{}, ctx.fence);
        }
        if (!ctx.device.serialize_submissions) waitForFence(ctx);
        ctx.submit_pending = false;
        if (cmd_buf) |cb| {
            cb.in_use = false;
            common.checkOrAbort(d.vkResetCommandBuffer.?(cb.buf, 0), "vk::CommandBuffer::reset");
        }
    }

    if (do_transfer) {
        m.submit.runMemcpys(compute_ctx.?.out_memcpys.items);
        ctx.compute_ctx = null;
    }
}

/// Ports `ggml_vk_graph_cleanup` (ggml-vulkan.cpp:16608 @b29c606e2): frees
/// what one graph's recording made, and resets the per-graph state.
pub fn graphCleanup(ctx: *BackendCtx) void {
    ctx.prealloc_y_last_pipeline_used = null;
    ctx.prealloc_y_last_tensor_used = null;
    ctx.prealloc_y_last_k_padded = false;

    ctx.unsynced_nodes_written.clearRetainingCapacity();
    ctx.unsynced_nodes_read.clearRetainingCapacity();
    ctx.prealloc_x_need_sync = false;
    ctx.prealloc_y_need_sync = false;
    ctx.prealloc_split_k_need_sync = false;

    m.submit.commandPoolCleanup(ctx.device, &ctx.compute_cmd_pool);
    if (ctx.device.async_use_transfer_queue) m.submit.commandPoolCleanup(ctx.device, &ctx.transfer_cmd_pool);

    for (ctx.gc.semaphores.items) |s| {
        d.vkDestroySemaphore.?(ctx.device.device, s.s, null);
        allocator.destroy(s);
    }
    ctx.gc.semaphores.clearRetainingCapacity();

    for (ctx.gc.tl_semaphores.items) |s| {
        d.vkDestroySemaphore.?(ctx.device.device, s.s, null);
        allocator.destroy(s);
    }
    ctx.gc.tl_semaphores.clearRetainingCapacity();
    ctx.semaphore_idx = 0;

    ctx.event_idx = 0;
    for (ctx.gc.events.items) |e| common.checkOrAbort(d.vkResetEvent.?(ctx.device.device, e), "vk::Device::resetEvent");

    ctx.tensor_ctxs.clearRetainingCapacity();
    // The garbage list held the only strong references: the contexts die
    // here, and every weak reference to one expires.
    for (ctx.gc.contexts.items) |cx| {
        if (ctx.compute_ctx == cx) ctx.compute_ctx = null;
        if (ctx.transfer_ctx == cx) ctx.transfer_ctx = null;
        cx.destroy();
    }
    ctx.gc.contexts.clearRetainingCapacity();
    ctx.pipeline_descriptor_set_requirements = 0;
    ctx.descriptor_set_idx = 0;
}

/// Ports `ggml_vk_cleanup` (ggml-vulkan.cpp:16647 @b29c606e2): everything
/// the backend owns, on free.
pub fn cleanup(ctx: *BackendCtx) void {
    // discard any unsubmitted command buffers
    ctx.compute_ctx = null;
    // wait for any pending command buffers to finish
    synchronize(ctx);

    graphCleanup(ctx);

    m.buffer.destroyBuffer(&ctx.prealloc_x);
    m.buffer.destroyBuffer(&ctx.prealloc_y);
    m.buffer.destroyBuffer(&ctx.prealloc_split_k);
    m.buffer.destroyBuffer(&ctx.prealloc_add_rms_partials);
    m.buffer.destroyBuffer(&ctx.sync_staging);

    ctx.prealloc_y_last_pipeline_used = null;
    ctx.prealloc_y_last_tensor_used = null;
    ctx.prealloc_y_last_k_padded = false;

    ctx.prealloc_size_x = 0;
    ctx.prealloc_size_y = 0;
    ctx.prealloc_size_split_k = 0;

    for (ctx.gc.events.items) |e| d.vkDestroyEvent.?(ctx.device.device, e, null);
    ctx.gc.events.clearRetainingCapacity();

    d.vkDestroyFence.?(ctx.device.device, ctx.fence, null);
    d.vkDestroyFence.?(ctx.device.device, ctx.almost_ready_fence, null);

    for (ctx.descriptor_pools.items) |pool| d.vkDestroyDescriptorPool.?(ctx.device.device, pool, null);
    ctx.descriptor_pools.clearRetainingCapacity();
    ctx.descriptor_sets.clearRetainingCapacity();

    ctx.compute_cmd_pool.destroy(ctx.device.device);
    if (ctx.device.async_use_transfer_queue) {
        d.vkDestroySemaphore.?(ctx.device.device, ctx.transfer_semaphore.s, null);
        ctx.transfer_cmd_pool.destroy(ctx.device.device);
    }
    if (m.instance.loggers.perf_logger_enabled) ctx.perf_logger.?.printTimings(true);
}

/// Ports `ggml_pipeline_request_descriptor_sets` (ggml-vulkan.cpp:3332 @b29c606e2):
/// reserves `n` descriptor sets for a pipeline's dispatches, compiling the
/// pipeline on first use.
pub fn requestDescriptorSets(ctx: *BackendCtx, pipeline: *Pipeline, n: u32) void {
    ctx.pipeline_descriptor_set_requirements += n;
    if (!pipeline.compiled.load(.seq_cst)) m.shaders.loadShaders(ctx.device, pipeline);
    allocateDescriptorSets(ctx);
}

/// Ports `ggml_pipeline_allocate_descriptor_sets` (ggml-vulkan.cpp:3341 @b29c606e2).
pub fn allocateDescriptorSets(ctx: *BackendCtx) void {
    if (ctx.descriptor_sets.items.len >= ctx.pipeline_descriptor_set_requirements) return;

    const device = ctx.device;
    const pool_size: u32 = m.submit.descriptor_pool_size;
    const have: u32 = @intCast(ctx.descriptor_sets.items.len);
    // Grow by 50% to avoid frequent allocations
    const needed = @max(3 * have / 2, ctx.pipeline_descriptor_set_requirements);
    var to_alloc = needed - have;
    var pool_remaining = pool_size - have % pool_size;
    var pool_idx = have / pool_size;

    while (to_alloc > 0) {
        const alloc_count = @min(pool_remaining, to_alloc);
        to_alloc -= alloc_count;
        pool_remaining = pool_size;

        if (pool_idx >= ctx.descriptor_pools.items.len) {
            const dps: vk.VkDescriptorPoolSize = .{ .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = m.pipeline.max_parameter_count * pool_size };
            var dpci = common.structure(vk.VkDescriptorPoolCreateInfo, vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
            dpci.maxSets = pool_size;
            dpci.poolSizeCount = 1;
            dpci.pPoolSizes = &dps;
            var pool: vk.VkDescriptorPool = null;
            common.checkOrAbort(d.vkCreateDescriptorPool.?(device.device, &dpci, null, &pool), "vk::Device::createDescriptorPool");
            ctx.descriptor_pools.append(allocator, pool) catch oom();
        }

        const layouts = allocator.alloc(vk.VkDescriptorSetLayout, alloc_count) catch oom();
        defer allocator.free(layouts);
        @memset(layouts, device.dsl);
        var dsai = common.structure(vk.VkDescriptorSetAllocateInfo, vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
        dsai.descriptorPool = ctx.descriptor_pools.items[pool_idx];
        dsai.descriptorSetCount = alloc_count;
        dsai.pSetLayouts = layouts.ptr;
        const sets = ctx.descriptor_sets.addManyAsSlice(allocator, alloc_count) catch oom();
        common.checkOrAbort(d.vkAllocateDescriptorSets.?(device.device, &dsai, sets.ptr), "vk::Device::allocateDescriptorSets");

        pool_idx += 1;
    }
}

/// Ports `ggml_vk_dispatch_pipeline` (ggml-vulkan.cpp:8450 @b29c606e2):
/// binds the buffers, pushes the constants and records the dispatch,
/// rounding each extent up to whole workgroups.
pub fn dispatchPipeline(ctx: *BackendCtx, subctx: *Context, pipeline: *Pipeline, descriptor_buffer_infos: []const m.buffer.Subbuffer, push_constants: []const u8, elements: [3]u32) void {
    const wg0 = std.math.divCeil(u32, elements[0], pipeline.wg_denoms[0]) catch unreachable;
    const wg1 = std.math.divCeil(u32, elements[1], pipeline.wg_denoms[1]) catch unreachable;
    const wg2 = std.math.divCeil(u32, elements[2], pipeline.wg_denoms[2]) catch unreachable;
    const limits = ctx.device.properties.limits;
    impl.assert(wg0 <= limits.maxComputeWorkGroupCount[0] and wg1 <= limits.maxComputeWorkGroupCount[1] and wg2 <= limits.maxComputeWorkGroupCount[2], "wg0 <= ctx->device->properties.limits.maxComputeWorkGroupCount[0] && wg1 <= ctx->device->properties.limits.maxComputeWorkGroupCount[1] && wg2 <= ctx->device->properties.limits.maxComputeWorkGroupCount[2]");
    impl.assert(ctx.descriptor_set_idx < ctx.descriptor_sets.items.len, "ctx->descriptor_set_idx < ctx->descriptor_sets.size()");
    impl.assert(descriptor_buffer_infos.len <= m.pipeline.max_parameter_count, "descriptor_buffer_infos.size() <= MAX_PARAMETER_COUNT");
    impl.assert(pipeline.parameter_count == descriptor_buffer_infos.len, "pipeline->parameter_count == descriptor_buffer_infos.size()");
    impl.assert(pipeline.push_constant_size == push_constants.len, "pipeline->push_constant_size == push_constant_size(push_constants)");

    const descriptor_set = ctx.descriptor_sets.items[ctx.descriptor_set_idx];
    ctx.descriptor_set_idx += 1;

    var infos: [m.pipeline.max_parameter_count]vk.VkDescriptorBufferInfo = undefined;
    for (descriptor_buffer_infos, 0..) |sb, i| infos[i] = sb.info();
    var wds = common.structure(vk.VkWriteDescriptorSet, vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
    wds.dstSet = descriptor_set;
    wds.dstBinding = 0;
    wds.dstArrayElement = 0;
    wds.descriptorCount = pipeline.parameter_count;
    wds.descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo = &infos;
    d.vkUpdateDescriptorSets.?(ctx.device.device, 1, &wds, 0, null);

    const cmd = subctx.cmd();
    d.vkCmdPushConstants.?(cmd, pipeline.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @intCast(push_constants.len), push_constants.ptr);
    d.vkCmdBindPipeline.?(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    d.vkCmdBindDescriptorSets.?(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &descriptor_set, 0, null);
    {
        var dbg = m.submit.DebugLabel.dispatch(subctx, pipeline.name, wg0, wg1, wg2);
        defer dbg.close();
        d.vkCmdDispatch.?(cmd, wg0, wg1, wg2);
    }
}

/// `ggml_vk_dispatch_pipeline` for one push-constant struct.
pub fn dispatch(ctx: *BackendCtx, subctx: *Context, pipeline: *Pipeline, bufs: []const m.buffer.Subbuffer, pc: anytype, elements: [3]u32) void {
    dispatchPipeline(ctx, subctx, pipeline, bufs, std.mem.asBytes(pc), elements);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
