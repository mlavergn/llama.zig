//! The backend context: one per `ggml_backend_vk_init`, holding the
//! device, the command pools and contexts recorded into, the preallocated
//! scratch buffers, the descriptor sets, and the state fusion and
//! synchronisation track between nodes.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp`,
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-common.h`, and
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-types.h` at v1.0.0
//! (`b1b770d90`). The C++'s `vk_context_ref`s -- weak pointers to contexts the
//! garbage list owns -- are plain pointers here, nulled where the list is
//! cleared (`graphCleanup`), the only place those contexts die.

const std = @import("std");
const ggml = @import("../module.zig");

fn oom() noreturn {
    ggml.impl.abort("ggml_vulkan: out of memory");
}

/// Ports `topk_moe_mode` (ggml-vulkan-types.h:635 @b1b770d90).
/// `count` is `TOPK_MOE_COUNT`, which the C++ uses for "no topk_moe fusion".
pub const TopkMoeMode = enum(u32) { early_softmax, early_softmax_norm, late_softmax, sigmoid_norm_bias, sqrt_softplus_norm_bias, count };

/// Ports `rms_norm_mode` (ggml-vulkan-types.h:644 @b1b770d90): which fusion the
/// RMS_NORM being processed heads. `count` is `RMS_NORM_COUNT`, "no
/// rms_norm fusion".
pub const RmsNormMode = enum(u32) { mul, mul_add, mul_add_mul, mul_rope, mul_rope_view_set_rows, view_set_rows, count };

/// Ports `ggml_vk_garbage_collector` (ggml-vulkan-types.h:1191 @b1b770d90).
/// Semaphores are handed out by pointer and must not move, so each is its
/// own allocation.
pub const GarbageCollector = struct {
    tl_semaphores: std.ArrayList(*ggml.vulkan.submit.Semaphore) = .empty,
    semaphores: std.ArrayList(*ggml.vulkan.submit.Semaphore) = .empty,
    events: std.ArrayList(ggml.vulkan.common.vulkan_c.VkEvent) = .empty,
    contexts: std.ArrayList(*ggml.vulkan.submit.Context) = .empty,
};

/// The bindings last written to one descriptor set: the C++'s
/// `std::vector<vk::DescriptorBufferInfo>`, held inline since a set never
/// takes more than `MAX_PARAMETER_COUNT`.
pub const DescriptorSetBindings = struct {
    len: u32 = 0,
    infos: [ggml.vulkan.pipeline.max_parameter_count]ggml.vulkan.common.vulkan_c.VkDescriptorBufferInfo = undefined,

    fn items(self: *const DescriptorSetBindings) []const ggml.vulkan.common.vulkan_c.VkDescriptorBufferInfo {
        return self.infos[0..self.len];
    }
};

/// Ports `ggml_backend_vk_context` (ggml-vulkan-types.h:1277 @b1b770d90).
pub const BackendCtx = struct {
    name: [:0]const u8 = "",
    device: *ggml.vulkan.device.Device = undefined,

    semaphore_idx: usize = 0,
    binary_semaphore_idx: usize = 0,
    event_idx: usize = 0,
    gc: GarbageCollector = .{},
    prealloc_size_x: usize = 0,
    prealloc_size_y: usize = 0,
    prealloc_size_split_k: usize = 0,
    prealloc_size_add_rms_partials: usize = 0,
    prealloc_size_add_rms_partials_offset: usize = 0,
    prealloc_x: ?*ggml.vulkan.buffer.Buffer = null,
    prealloc_y: ?*ggml.vulkan.buffer.Buffer = null,
    prealloc_split_k: ?*ggml.vulkan.buffer.Buffer = null,
    prealloc_add_rms_partials: ?*ggml.vulkan.buffer.Buffer = null,
    sync_staging: ?*ggml.vulkan.buffer.Buffer = null,
    fence: ggml.vulkan.common.vulkan_c.VkFence = null,
    almost_ready_fence: ggml.vulkan.common.vulkan_c.VkFence = null,
    submit_pending: bool = false,
    almost_ready_fence_pending: bool = false,
    // Set before op_add and unset after op_rms_norm to indicate that the add
    // should write partial sums to accumulate the square of the vector components
    do_add_rms_partials_offset_calculation: bool = false,
    do_add_rms_partials: bool = false,

    last_total_flops: u64 = std.math.maxInt(u64),

    // Cache most recent tensor that was converted into prealloc_y, and what pipeline it used to convert.
    prealloc_y_last_pipeline_used: ?*ggml.vulkan.pipeline.Pipeline = null,
    prealloc_y_last_tensor_used: ?*const ggml.impl.ggml_c.ggml_tensor = null,
    // True when the K dimension in prealloc_y is padded.
    prealloc_y_last_k_padded: bool = false,

    // Track which nodes have been used since the last sync, and whether they were written to
    unsynced_nodes_written: std.ArrayList(*const ggml.impl.ggml_c.ggml_tensor) = .empty,
    unsynced_nodes_read: std.ArrayList(*const ggml.impl.ggml_c.ggml_tensor) = .empty,
    // Track which prealloc buffers have pending reads that need to be synchronized.
    prealloc_x_need_sync: bool = false,
    prealloc_y_need_sync: bool = false,
    prealloc_split_k_need_sync: bool = false,

    compute_ctx: ?*ggml.vulkan.submit.Context = null,

    transfer_ctx: ?*ggml.vulkan.submit.Context = null,
    transfer_semaphore: ggml.vulkan.submit.Semaphore = .{},
    transfer_semaphore_last_submitted: u64 = 0,

    tensor_ctxs: std.ArrayList(?*ggml.vulkan.submit.Context) = .empty,

    descriptor_pools: std.ArrayList(ggml.vulkan.common.vulkan_c.VkDescriptorPool) = .empty,
    descriptor_sets: std.ArrayList(ggml.vulkan.common.vulkan_c.VkDescriptorSet) = .empty,
    // last bindings written to each set; descriptor_sets is append-only so an index always names the same set
    descriptor_set_bindings: std.ArrayList(DescriptorSetBindings) = .empty,
    descriptor_set_bindings_destroy_count: u64 = 0,
    descriptor_set_idx: u32 = 0,
    pipeline_descriptor_set_requirements: u32 = 0,

    compute_cmd_pool: ggml.vulkan.device.CommandPool = .{},
    transfer_cmd_pool: ggml.vulkan.device.CommandPool = .{},

    // number of additional consecutive nodes fused with the node being processed
    num_additional_fused_ops: c_int = 0,
    // Bit 'i' means nodes[start_of_fusion + i] writes to memory.
    fused_ops_write_mask: c_int = 0,
    fused_topk_moe_mode: TopkMoeMode = .early_softmax,
    fused_topk_moe_scale: bool = false,
    // QSA indexer gather+add+top_k fused into one radix-select
    fused_topk_qsa: bool = false,
    fused_hc_post_gate: bool = false,
    fused_rms_norm_mode: RmsNormMode = .count,

    // for GGML_VK_PERF_LOGGER
    perf_logger: ?*ggml.vulkan.perf.PerfLogger = null,
    query_pool: ggml.vulkan.common.vulkan_c.VkQueryPool = null,
    query_fusion_names: std.ArrayList(?[*:0]const u8) = .empty,
    query_fusion_node_count: std.ArrayList(c_int) = .empty,
    query_nodes: std.ArrayList(?*ggml.impl.ggml_c.ggml_tensor) = .empty,
    query_node_idx: std.ArrayList(c_int) = .empty,
    num_queries: i32 = 0,
    query_idx: i32 = 0,
};

/// Ports `ggml_vk_init` (ggml-vulkan.cpp:5749 @b1b770d90).
pub fn init(ctx: *BackendCtx, idx: usize) void {
    ggml.vulkan.instance.instanceInitOrAbort();
    ggml.impl.assert(idx < ggml.vulkan.instance.instance.device_indices.items.len, "idx < vk_instance.device_indices.size()");

    ctx.name = std.fmt.allocPrintSentinel(std.heap.c_allocator, ggml.vulkan.common.name ++ "{d}", .{idx}, 0) catch oom();
    ctx.device = ggml.vulkan.device.getDevice(idx);

    ctx.semaphore_idx = 0;
    ctx.binary_semaphore_idx = 0;
    ctx.event_idx = 0;
    ctx.prealloc_size_x = 0;
    ctx.prealloc_size_y = 0;
    ctx.prealloc_size_split_k = 0;
    // Fixed size of 1KB, for deterministic behavior
    ctx.prealloc_size_add_rms_partials = 1024;

    var fci = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkFenceCreateInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkCreateFence.?(ctx.device.device, &fci, null, &ctx.fence), "vk::Device::createFence");
    ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkCreateFence.?(ctx.device.device, &fci, null, &ctx.almost_ready_fence), "vk::Device::createFence");

    ctx.compute_cmd_pool.init(ctx.device, ctx.device.compute_queue.?);
    if (ctx.device.async_use_transfer_queue) {
        ctx.transfer_semaphore.s = createSemaphore(ctx.device, ggml.vulkan.common.vulkan_c.VK_SEMAPHORE_TYPE_TIMELINE);
        ctx.transfer_semaphore.value = 0;
        ctx.transfer_cmd_pool.init(ctx.device, ctx.device.transfer_queue.?);
    }

    if (ggml.vulkan.instance.loggers.perf_logger_enabled) ctx.perf_logger = ggml.vulkan.perf.PerfLogger.create();
}

fn createSemaphore(device: *ggml.vulkan.device.Device, ty: ggml.vulkan.common.vulkan_c.VkSemaphoreType) ggml.vulkan.common.vulkan_c.VkSemaphore {
    var tci = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkSemaphoreTypeCreateInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO);
    tci.semaphoreType = ty;
    tci.initialValue = 0;
    var ci = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkSemaphoreCreateInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    ci.pNext = &tci;
    var s: ggml.vulkan.common.vulkan_c.VkSemaphore = null;
    ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkCreateSemaphore.?(device.device, &ci, null, &s), "vk::Device::createSemaphore");
    return s;
}

/// Ports `ggml_vk_create_context` (ggml-vulkan.cpp:1077 @b1b770d90): a
/// context the garbage list owns until the next graph cleanup.
pub fn createContext(ctx: *BackendCtx, p: *ggml.vulkan.device.CommandPool) *ggml.vulkan.submit.Context {
    const result = ggml.vulkan.submit.createTemporaryContext(p);
    ctx.gc.contexts.append(std.heap.c_allocator, result) catch oom();
    return result;
}

/// Ports `ggml_vk_create_binary_semaphore` (ggml-vulkan.cpp:1092 @b1b770d90):
/// pooled, and handed out again only after `graphCleanup` resets the index.
pub fn createBinarySemaphore(ctx: *BackendCtx) ggml.vulkan.common.vulkan_c.VkSemaphore {
    if (ctx.binary_semaphore_idx >= ctx.gc.semaphores.items.len) {
        const s = std.heap.c_allocator.create(ggml.vulkan.submit.Semaphore) catch oom();
        s.* = .{ .s = createSemaphore(ctx.device, ggml.vulkan.common.vulkan_c.VK_SEMAPHORE_TYPE_BINARY), .value = 0 };
        ctx.gc.semaphores.append(std.heap.c_allocator, s) catch oom();
    }
    defer ctx.binary_semaphore_idx += 1;
    return ctx.gc.semaphores.items[ctx.binary_semaphore_idx].s;
}

/// Ports `ggml_vk_create_timeline_semaphore` (ggml-vulkan.cpp:1104 @b1b770d90).
pub fn createTimelineSemaphore(ctx: *BackendCtx) *ggml.vulkan.submit.Semaphore {
    if (ctx.semaphore_idx >= ctx.gc.tl_semaphores.items.len) {
        const s = std.heap.c_allocator.create(ggml.vulkan.submit.Semaphore) catch oom();
        s.* = .{ .s = createSemaphore(ctx.device, ggml.vulkan.common.vulkan_c.VK_SEMAPHORE_TYPE_TIMELINE), .value = 0 };
        ctx.gc.tl_semaphores.append(std.heap.c_allocator, s) catch oom();
    }
    defer ctx.semaphore_idx += 1;
    return ctx.gc.tl_semaphores.items[ctx.semaphore_idx];
}

/// Ports `ggml_vk_create_event` (ggml-vulkan.cpp:1116 @b1b770d90).
pub fn createEvent(ctx: *BackendCtx) ggml.vulkan.common.vulkan_c.VkEvent {
    if (ctx.event_idx >= ctx.gc.events.items.len) {
        var ci = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkEventCreateInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_EVENT_CREATE_INFO);
        var e: ggml.vulkan.common.vulkan_c.VkEvent = null;
        ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkCreateEvent.?(ctx.device.device, &ci, null, &e), "vk::Device::createEvent");
        ctx.gc.events.append(std.heap.c_allocator, e) catch oom();
    }
    defer ctx.event_idx += 1;
    return ctx.gc.events.items[ctx.event_idx];
}

/// Ports `ggml_vk_sync_buffers` (ggml-vulkan.cpp:1153 @b1b770d90): a full
/// memory barrier, after which no preallocated buffer has a read pending.
pub fn syncBuffers(ctx: ?*BackendCtx, subctx: *ggml.vulkan.submit.Context) void {
    if (ctx) |cx| {
        cx.prealloc_x_need_sync = false;
        cx.prealloc_y_need_sync = false;
        cx.prealloc_split_k_need_sync = false;
    }
    ggml.vulkan.submit.syncBuffersBarrier(subctx);
}

/// Ports `ggml_vk_get_max_buffer_range` (ggml-vulkan.cpp:335 @b1b770d90).
pub fn getMaxBufferRange(ctx: *const BackendCtx, buf: *const ggml.vulkan.buffer.Buffer, offset: u64) u64 {
    return @min(buf.size - offset, ctx.device.properties.limits.maxStorageBufferRange);
}

/// Ports `ggml_vk_subbuffer` (ggml-vulkan.cpp:1149 @b1b770d90).
pub fn subbuffer(ctx: *const BackendCtx, buf: *ggml.vulkan.buffer.Buffer, offset: u64) ggml.vulkan.buffer.Subbuffer {
    return .{ .buffer = buf, .offset = offset, .size = getMaxBufferRange(ctx, buf, offset) };
}

/// Ports `ggml_vk_tensor_buffer_offset` (ggml-vulkan.cpp:269 @b1b770d90):
/// a tensor's offset within its Vulkan buffer. `vk_tensor_offset` is
/// relative to `vk_ptr_base`, but mapped host tensors need an offset
/// relative to their own buffer.
pub fn tensorBufferOffset(ctx: *const BackendCtx, t: *const ggml.impl.ggml_c.ggml_tensor) usize {
    if (ctx.device.uma) {
        var off: usize = 0;
        if (ggml.vulkan.buffer.hostGet(ctx.device, t.data, &off) != null) return off;
    }
    return @intCast(ggml.vulkan.buffer.tensorOffset(t) + t.view_offs);
}

/// Ports `ggml_vk_descriptor_offset` (ggml-vulkan.cpp:281 @b1b770d90): moves
/// the descriptor back until its distance to the tensor is divisible by the
/// tensor type size.
fn descriptorOffset(tensor_offset: usize, alignment: usize, type_size: usize) usize {
    var descriptor_offset = tensor_offset & ~(alignment -% 1);
    while ((tensor_offset -% descriptor_offset) % type_size != 0) {
        ggml.impl.assert(descriptor_offset >= alignment, "descriptor_offset >= alignment");
        descriptor_offset -%= alignment;
    }
    return descriptor_offset;
}

/// Ports `get_misalign_bytes` (ggml-vulkan.cpp:291 @b1b770d90).
pub fn getMisalignBytes(ctx: *const BackendCtx, t: *const ggml.impl.ggml_c.ggml_tensor) u32 {
    const tensor_offset = tensorBufferOffset(ctx, t);
    const descriptor_offset = descriptorOffset(tensor_offset, ctx.device.properties.limits.minStorageBufferOffsetAlignment, ggml.types.ggml_type_size(t.type));
    ggml.impl.assert(tensor_offset - descriptor_offset <= std.math.maxInt(u32), "tensor_offset - descriptor_offset <= UINT32_MAX");
    return @intCast(tensor_offset - descriptor_offset);
}

/// Ports `ggml_vk_tensor_subbuffer` (ggml-vulkan.cpp:6008 @b1b770d90): the
/// buffer range holding a tensor, its offset moved back to a storage
/// alignment a whole number of elements before it and its size grown to
/// match.
pub fn tensorSubbuffer(ctx: *const BackendCtx, tensor: *const ggml.impl.ggml_c.ggml_tensor, allow_misalign: bool) ggml.vulkan.buffer.Subbuffer {
    var buffer: ?*ggml.vulkan.buffer.Buffer = null;
    var offset: usize = 0;
    if (ctx.device.uma) buffer = ggml.vulkan.buffer.hostGet(ctx.device, tensor.data, &offset);
    if (buffer == null) {
        buffer = ggml.vulkan.buffer.tensorBuffer(tensor);
        offset = ggml.vulkan.buffer.tensorOffset(tensor) + tensor.view_offs;
    }
    ggml.impl.assert(buffer != null, "buffer != nullptr");

    var size = ggml.types.ggml_nbytes(tensor);
    const descriptor_offset = descriptorOffset(offset, ctx.device.properties.limits.minStorageBufferOffsetAlignment, ggml.types.ggml_type_size(tensor.type));
    const misalign_bytes = offset - descriptor_offset;
    // The shader must support misaligned offsets when indexing into the buffer
    ggml.impl.assert(allow_misalign or misalign_bytes == 0, "allow_misalign || misalign_bytes == 0");
    offset = descriptor_offset;
    size += misalign_bytes;
    return .{ .buffer = buffer, .offset = offset, .size = size };
}

/// Ports `ggml_vk_ensure_sync_staging_buffer` for a backend context
/// (ggml-vulkan-buffers.cpp:397 @b1b770d90).
pub fn ensureSyncStagingBuffer(ctx: *BackendCtx, size: usize) void {
    if (ctx.sync_staging == null or ctx.sync_staging.?.size < size) {
        ggml.vulkan.buffer.logMemory("ggml_vk_ensure_sync_staging_buffer(%zu)", .{size});
        ggml.vulkan.buffer.destroyBuffer(&ctx.sync_staging);
        ctx.sync_staging = ggml.vulkan.buffer.createBufferCheck(ctx.device, size, ggml.vulkan.common.vulkan_c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ggml.vulkan.common.vulkan_c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | ggml.vulkan.common.vulkan_c.VK_MEMORY_PROPERTY_HOST_CACHED_BIT, ggml.vulkan.common.vulkan_c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ggml.vulkan.common.vulkan_c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
}

/// Ports `ggml_vk_get_compute_ctx` (ggml-vulkan.cpp:6106 @b1b770d90): the
/// context compute is recorded into, begun on first use; it waits for any
/// transfer submitted since.
pub fn getComputeCtx(ctx: *BackendCtx) *ggml.vulkan.submit.Context {
    const result = ctx.compute_ctx orelse blk: {
        const r = createContext(ctx, &ctx.compute_cmd_pool);
        ctx.compute_ctx = r;
        ggml.vulkan.submit.ctxBegin(ctx.device, r);
        break :blk r;
    };
    if (ctx.device.async_use_transfer_queue and ctx.transfer_semaphore_last_submitted < ctx.transfer_semaphore.value) {
        result.s().wait_semaphores.append(std.heap.c_allocator, ctx.transfer_semaphore) catch oom();
        ctx.transfer_semaphore_last_submitted = ctx.transfer_semaphore.value;
    }
    return result;
}

/// Ports `ggml_vk_get_transfer_ctx` (ggml-vulkan.cpp:6125 @b1b770d90).
pub fn getTransferCtx(ctx: *BackendCtx) *ggml.vulkan.submit.Context {
    if (ctx.transfer_ctx) |r| return r;
    const r = createContext(ctx, &ctx.transfer_cmd_pool);
    ctx.transfer_ctx = r;
    ggml.vulkan.submit.ctxBegin(ctx.device, r);
    return r;
}

/// Ports `ggml_vk_submit_transfer_ctx` (ggml-vulkan.cpp:6139 @b1b770d90):
/// submits pending transfer-queue work, signalling the semaphore the next
/// compute context waits on.
///
/// Return: whether work was submitted.
pub fn submitTransferCtx(ctx: *BackendCtx) bool {
    if (!ctx.device.async_use_transfer_queue) return false;
    const cpy_ctx = ctx.transfer_ctx orelse return false;
    ggml.vulkan.submit.ctxEnd(cpy_ctx);
    ggml.vulkan.submit.runMemcpys(cpy_ctx.in_memcpys.items);

    ctx.transfer_semaphore.value += 1;
    const last_seq = &cpy_ctx.seqs.items[cpy_ctx.seqs.items.len - 1];
    last_seq.items[last_seq.items.len - 1].signal_semaphores.append(std.heap.c_allocator, ctx.transfer_semaphore) catch oom();

    ggml.vulkan.submit.submit(cpy_ctx, null);
    ctx.transfer_ctx = null;
    return true;
}

/// Ports `ggml_vk_wait_for_fence` (ggml-vulkan.cpp:341 @b1b770d90): waits
/// on the almost-ready fence, then spins on the graph's fence.
pub fn waitForFence(ctx: *BackendCtx) void {
    const dev = ctx.device;
    // Use waitForFences while most of the graph executes.
    if (ctx.almost_ready_fence_pending) {
        ggml.vulkan.submit.waitAndResetFence(dev, ctx.almost_ready_fence, "almost_ready_fence");
        ctx.almost_ready_fence_pending = false;
    }

    // Spin (w/pause) waiting for the graph to finish executing.
    while (true) {
        const result = ggml.vulkan.common.dispatch.vkGetFenceStatus.?(dev.device, ctx.fence);
        if (result == ggml.vulkan.common.vulkan_c.VK_ERROR_DEVICE_LOST) {
            ggml.vulkan.submit.printDeviceLostInfo(dev);
            ggml.impl.logError("ggml_vulkan: getFenceStatus\n", .{});
            ggml.vulkan.common.abortLastError();
        }
        if (result == ggml.vulkan.common.vulkan_c.VK_SUCCESS) break;
        if (result != ggml.vulkan.common.vulkan_c.VK_NOT_READY) {
            ggml.impl.logError("ggml_vulkan: error %s\n", .{ggml.vulkan.common.resultString(result).ptr});
            ggml.impl.abort("ggml_vulkan: getFenceStatus");
        }
        for (0..100) |_| {
            inline for (0..10) |_| std.atomic.spinLoopHint();
        }
    }
    ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkResetFences.?(dev.device, 1, &ctx.fence), "vk::Device::resetFences");
}

/// Ports `ggml_vk_synchronize` (ggml-vulkan.cpp:13918 @b1b770d90): submits
/// the recorded compute context and any transfer, waits for both, and runs
/// the deferred host copies.
pub fn synchronize(ctx: *BackendCtx) void {
    const do_transfer = ctx.compute_ctx != null;

    if (submitTransferCtx(ctx)) ctx.submit_pending = true;

    var compute_ctx: ?*ggml.vulkan.submit.Context = null;
    var cmd_buf: ?*ggml.vulkan.device.CommandBuffer = null;
    if (do_transfer) {
        const cc = ctx.compute_ctx.?;
        compute_ctx = cc;
        if (cc.has_s) cmd_buf = cc.s().buffer;
        ggml.vulkan.submit.ctxEnd(cc);
        ggml.vulkan.submit.runMemcpys(cc.in_memcpys.items);

        if (ctx.device.serialize_submissions) {
            ggml.vulkan.submit.submit(cc, ctx.fence);
            ggml.vulkan.submit.waitAndResetFence(ctx.device, ctx.fence, "synchronize waitForFences");
        } else {
            ggml.vulkan.submit.submit(cc, null);
        }
        ctx.submit_pending = true;
    }

    if (ctx.submit_pending) {
        const h = ctx.device.compute_queue.?.handle;
        if (ctx.device.serialize_submissions) {
            ctx.submit_pending = false;
        } else if (ctx.device.async_use_transfer_queue and ctx.transfer_semaphore_last_submitted < ctx.transfer_semaphore.value) {
            var tl_info = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkTimelineSemaphoreSubmitInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO);
            tl_info.waitSemaphoreValueCount = 1;
            tl_info.pWaitSemaphoreValues = &ctx.transfer_semaphore.value;
            const stage: ggml.vulkan.common.vulkan_c.VkPipelineStageFlags = ctx.device.transfer_queue.?.stage_flags;
            var si = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkSubmitInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_SUBMIT_INFO);
            si.waitSemaphoreCount = 1;
            si.pWaitSemaphores = &ctx.transfer_semaphore.s;
            si.pWaitDstStageMask = &stage;
            si.pNext = &tl_info;
            ggml.vulkan.submit.queueSubmit(h, &.{si}, ctx.fence);
            ctx.transfer_semaphore_last_submitted = ctx.transfer_semaphore.value;
        } else {
            ggml.vulkan.submit.queueSubmit(h, &.{}, ctx.fence);
        }
        if (!ctx.device.serialize_submissions) waitForFence(ctx);
        ctx.submit_pending = false;
        if (cmd_buf) |cb| {
            cb.in_use = false;
            ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkResetCommandBuffer.?(cb.buf, 0), "vk::CommandBuffer::reset");
        }
    }

    if (do_transfer) {
        ggml.vulkan.submit.runMemcpys(compute_ctx.?.out_memcpys.items);
        ctx.compute_ctx = null;
    }
}

/// Ports `ggml_vk_graph_cleanup` (ggml-vulkan.cpp:13319 @b1b770d90): frees
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

    ggml.vulkan.submit.commandPoolCleanup(ctx.device, &ctx.compute_cmd_pool);
    if (ctx.device.async_use_transfer_queue) ggml.vulkan.submit.commandPoolCleanup(ctx.device, &ctx.transfer_cmd_pool);

    ctx.binary_semaphore_idx = 0;

    for (ctx.gc.tl_semaphores.items) |s| {
        ggml.vulkan.common.dispatch.vkDestroySemaphore.?(ctx.device.device, s.s, null);
        std.heap.c_allocator.destroy(s);
    }
    ctx.gc.tl_semaphores.clearRetainingCapacity();
    ctx.semaphore_idx = 0;

    ctx.event_idx = 0;
    for (ctx.gc.events.items) |e| ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkResetEvent.?(ctx.device.device, e), "vk::Device::resetEvent");

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

/// Ports `ggml_vk_cleanup` (ggml-vulkan.cpp:13354 @b1b770d90): everything
/// the backend owns, on free.
pub fn cleanup(ctx: *BackendCtx) void {
    // discard any unsubmitted command buffers
    ctx.compute_ctx = null;
    // wait for any pending command buffers to finish
    synchronize(ctx);

    graphCleanup(ctx);

    ggml.vulkan.buffer.destroyBuffer(&ctx.prealloc_x);
    ggml.vulkan.buffer.destroyBuffer(&ctx.prealloc_y);
    ggml.vulkan.buffer.destroyBuffer(&ctx.prealloc_split_k);
    ggml.vulkan.buffer.destroyBuffer(&ctx.prealloc_add_rms_partials);
    ggml.vulkan.buffer.destroyBuffer(&ctx.sync_staging);

    ctx.prealloc_y_last_pipeline_used = null;
    ctx.prealloc_y_last_tensor_used = null;
    ctx.prealloc_y_last_k_padded = false;

    ctx.prealloc_size_x = 0;
    ctx.prealloc_size_y = 0;
    ctx.prealloc_size_split_k = 0;

    for (ctx.gc.events.items) |e| ggml.vulkan.common.dispatch.vkDestroyEvent.?(ctx.device.device, e, null);
    ctx.gc.events.clearRetainingCapacity();

    for (ctx.gc.semaphores.items) |sem| {
        ggml.vulkan.common.dispatch.vkDestroySemaphore.?(ctx.device.device, sem.s, null);
        std.heap.c_allocator.destroy(sem);
    }
    ctx.gc.semaphores.clearRetainingCapacity();

    ggml.vulkan.common.dispatch.vkDestroyFence.?(ctx.device.device, ctx.fence, null);
    ggml.vulkan.common.dispatch.vkDestroyFence.?(ctx.device.device, ctx.almost_ready_fence, null);

    for (ctx.descriptor_pools.items) |pool| ggml.vulkan.common.dispatch.vkDestroyDescriptorPool.?(ctx.device.device, pool, null);
    ctx.descriptor_pools.clearRetainingCapacity();
    ctx.descriptor_sets.clearRetainingCapacity();
    ctx.descriptor_set_bindings.clearRetainingCapacity();

    ctx.compute_cmd_pool.destroy(ctx.device.device);
    if (ctx.device.async_use_transfer_queue) {
        ggml.vulkan.common.dispatch.vkDestroySemaphore.?(ctx.device.device, ctx.transfer_semaphore.s, null);
        ctx.transfer_cmd_pool.destroy(ctx.device.device);
    }
    if (ggml.vulkan.instance.loggers.perf_logger_enabled) ctx.perf_logger.?.printTimings(true);
}

/// Ports `ggml_pipeline_request_descriptor_sets` (ggml-vulkan.cpp:845 @b1b770d90):
/// reserves `n` descriptor sets for a pipeline's dispatches, compiling the
/// pipeline on first use.
pub fn requestDescriptorSets(ctx: *BackendCtx, pipeline: *ggml.vulkan.pipeline.Pipeline, n: u32) void {
    ctx.pipeline_descriptor_set_requirements += n;
    if (!pipeline.compiled.load(.seq_cst)) ggml.vulkan.shaders.loadShaders(ctx.device, pipeline);
    allocateDescriptorSets(ctx);
}

/// Ports `ggml_pipeline_allocate_descriptor_sets` (ggml-vulkan.cpp:854 @b1b770d90).
pub fn allocateDescriptorSets(ctx: *BackendCtx) void {
    if (ctx.descriptor_sets.items.len >= ctx.pipeline_descriptor_set_requirements) return;

    const device = ctx.device;
    const pool_size: u32 = ggml.vulkan.submit.descriptor_pool_size;
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
            const dps: ggml.vulkan.common.vulkan_c.VkDescriptorPoolSize = .{ .type = ggml.vulkan.common.vulkan_c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = ggml.vulkan.pipeline.max_parameter_count * pool_size };
            var dpci = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkDescriptorPoolCreateInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
            dpci.maxSets = pool_size;
            dpci.poolSizeCount = 1;
            dpci.pPoolSizes = &dps;
            var pool: ggml.vulkan.common.vulkan_c.VkDescriptorPool = null;
            ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkCreateDescriptorPool.?(device.device, &dpci, null, &pool), "vk::Device::createDescriptorPool");
            ctx.descriptor_pools.append(std.heap.c_allocator, pool) catch oom();
        }

        const layouts = std.heap.c_allocator.alloc(ggml.vulkan.common.vulkan_c.VkDescriptorSetLayout, alloc_count) catch oom();
        defer std.heap.c_allocator.free(layouts);
        @memset(layouts, device.dsl);
        var dsai = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkDescriptorSetAllocateInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
        dsai.descriptorPool = ctx.descriptor_pools.items[pool_idx];
        dsai.descriptorSetCount = alloc_count;
        dsai.pSetLayouts = layouts.ptr;
        const sets = ctx.descriptor_sets.addManyAsSlice(std.heap.c_allocator, alloc_count) catch oom();
        ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkAllocateDescriptorSets.?(device.device, &dsai, sets.ptr), "vk::Device::allocateDescriptorSets");

        pool_idx += 1;
    }
    // resize: the new sets hold no bindings yet
    const old_len = ctx.descriptor_set_bindings.items.len;
    ctx.descriptor_set_bindings.resize(std.heap.c_allocator, ctx.descriptor_sets.items.len) catch oom();
    if (ctx.descriptor_set_bindings.items.len > old_len) @memset(ctx.descriptor_set_bindings.items[old_len..], .{});
}

/// Ports `ggml_vk_dispatch_pipeline` (ggml-vulkan-common.h:288 @b1b770d90):
/// binds the buffers, pushes the constants and records the dispatch,
/// rounding each extent up to whole workgroups. A set that already holds
/// the same bindings from the last graph is not written again, unless a
/// buffer has been destroyed since (a new one can reuse its handle) or
/// `GGML_VK_DISABLE_DESCRIPTOR_REUSE` is set.
pub fn dispatchPipeline(ctx: *BackendCtx, subctx: *ggml.vulkan.submit.Context, pipeline: *ggml.vulkan.pipeline.Pipeline, descriptor_buffer_infos: []const ggml.vulkan.buffer.Subbuffer, push_constants: []const u8, elements: [3]u32) void {
    const wg0 = std.math.divCeil(u32, elements[0], pipeline.wg_denoms[0]) catch unreachable;
    const wg1 = std.math.divCeil(u32, elements[1], pipeline.wg_denoms[1]) catch unreachable;
    const wg2 = std.math.divCeil(u32, elements[2], pipeline.wg_denoms[2]) catch unreachable;
    const limits = ctx.device.properties.limits;
    ggml.impl.assert(wg0 <= limits.maxComputeWorkGroupCount[0] and wg1 <= limits.maxComputeWorkGroupCount[1] and wg2 <= limits.maxComputeWorkGroupCount[2], "wg0 <= ctx->device->properties.limits.maxComputeWorkGroupCount[0] && wg1 <= ctx->device->properties.limits.maxComputeWorkGroupCount[1] && wg2 <= ctx->device->properties.limits.maxComputeWorkGroupCount[2]");
    ggml.impl.assert(ctx.descriptor_set_idx < ctx.descriptor_sets.items.len, "ctx->descriptor_set_idx < ctx->descriptor_sets.size()");
    ggml.impl.assert(descriptor_buffer_infos.len <= ggml.vulkan.pipeline.max_parameter_count, "descriptor_buffer_infos.size() <= MAX_PARAMETER_COUNT");
    ggml.impl.assert(pipeline.parameter_count == descriptor_buffer_infos.len, "pipeline->parameter_count == descriptor_buffer_infos.size()");
    ggml.impl.assert(pipeline.push_constant_size == push_constants.len, "pipeline->push_constant_size == push_constant_size(push_constants)");

    const descriptor_set_idx = ctx.descriptor_set_idx;
    ctx.descriptor_set_idx += 1;
    const descriptor_set = ctx.descriptor_sets.items[descriptor_set_idx];

    var infos: [ggml.vulkan.pipeline.max_parameter_count]ggml.vulkan.common.vulkan_c.VkDescriptorBufferInfo = undefined;
    for (descriptor_buffer_infos, 0..) |sb, i| infos[i] = sb.info();
    const n = descriptor_buffer_infos.len;

    // a new buffer can get the handle of a destroyed one, so drop all cached bindings after any destroy
    const destroy_count = ctx.device.buffer_destroy_count.load(.acquire);
    if (ctx.descriptor_set_bindings_destroy_count != destroy_count) {
        for (ctx.descriptor_set_bindings.items) |*b| b.len = 0;
        ctx.descriptor_set_bindings_destroy_count = destroy_count;
    }

    // skip the write if this set already holds these bindings from the last graph
    const bindings = &ctx.descriptor_set_bindings.items[descriptor_set_idx];
    var same = !ctx.device.disable_descriptor_reuse and bindings.len == n;
    if (same) {
        for (bindings.items(), infos[0..n]) |prev, info| {
            if (prev.buffer != info.buffer or prev.offset != info.offset or prev.range != info.range) {
                same = false;
                break;
            }
        }
    }
    if (!same) {
        var wds = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkWriteDescriptorSet, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
        wds.dstSet = descriptor_set;
        wds.dstBinding = 0;
        wds.dstArrayElement = 0;
        wds.descriptorCount = pipeline.parameter_count;
        wds.descriptorType = ggml.vulkan.common.vulkan_c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds.pBufferInfo = &infos;
        ggml.vulkan.common.dispatch.vkUpdateDescriptorSets.?(ctx.device.device, 1, &wds, 0, null);
        bindings.len = @intCast(n);
        @memcpy(bindings.infos[0..n], infos[0..n]);
    }

    const cmd = subctx.cmd();
    ggml.vulkan.common.dispatch.vkCmdPushConstants.?(cmd, pipeline.layout, ggml.vulkan.common.vulkan_c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @intCast(push_constants.len), push_constants.ptr);
    ggml.vulkan.common.dispatch.vkCmdBindPipeline.?(cmd, ggml.vulkan.common.vulkan_c.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    ggml.vulkan.common.dispatch.vkCmdBindDescriptorSets.?(cmd, ggml.vulkan.common.vulkan_c.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &descriptor_set, 0, null);
    {
        var dbg = ggml.vulkan.submit.DebugLabel.dispatch(subctx, pipeline.name, wg0, wg1, wg2);
        defer dbg.close();
        ggml.vulkan.common.dispatch.vkCmdDispatch.?(cmd, wg0, wg1, wg2);
    }
}

/// `ggml_vk_dispatch_pipeline` for one push-constant struct.
pub fn dispatch(ctx: *BackendCtx, subctx: *ggml.vulkan.submit.Context, pipeline: *ggml.vulkan.pipeline.Pipeline, bufs: []const ggml.vulkan.buffer.Subbuffer, pc: anytype, elements: [3]u32) void {
    dispatchPipeline(ctx, subctx, pipeline, bufs, std.mem.asBytes(pc), elements);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
