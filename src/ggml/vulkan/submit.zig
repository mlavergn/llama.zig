//! Command buffers, submissions and the contexts that collect them: how
//! recorded work reaches a queue, and the synchronous transfers built on
//! top.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.4.0
//! (`5266f24da`). A `vk_context` is a `std::shared_ptr` in the C++, kept
//! alive either by the backend context's garbage list or by the one caller
//! that made it ("temporary"); here the owner frees it explicitly.

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const common = m.common;
const vk = common.vk;
const d = &common.dispatch;
const allocator = common.allocator;
const Device = m.device.Device;
const CommandBuffer = m.device.CommandBuffer;
const CommandPool = m.device.CommandPool;
const Buffer = m.buffer.Buffer;

/// Ports `VK_DEVICE_DESCRIPTOR_POOL_SIZE` (ggml-vulkan.cpp:187 @5266f24da).
pub const descriptor_pool_size = 256;

/// Ports `vk_semaphore` (ggml-vulkan.cpp:1278 @5266f24da).
pub const Semaphore = struct {
    s: vk.VkSemaphore = null,
    value: u64 = 0,
};

/// Ports `vk_submission` (ggml-vulkan.cpp:1298 @5266f24da).
pub const Submission = struct {
    buffer: ?*CommandBuffer = null,
    wait_semaphores: std.ArrayList(Semaphore) = .empty,
    signal_semaphores: std.ArrayList(Semaphore) = .empty,

    fn deinit(self: *Submission) void {
        self.wait_semaphores.deinit(allocator);
        self.signal_semaphores.deinit(allocator);
    }
};

/// Ports `vk_sequence` (ggml-vulkan.cpp:1304 @5266f24da).
pub const Sequence = std.ArrayList(Submission);

/// Ports `vk_staging_memcpy` (ggml-vulkan.cpp:2106 @5266f24da).
pub const StagingMemcpy = struct { dst: [*]u8, src: [*]const u8, n: usize };

/// Ports `vk_staging_memset` (ggml-vulkan.cpp:2114 @5266f24da).
pub const StagingMemset = struct { dst: [*]u8, val: u32, n: usize };

/// Ports `vk_context_struct` (ggml-vulkan.cpp:2122 @5266f24da).
///
/// `s` points at the last submission of the last sequence; it is an index
/// pair here, since the sequences' storage moves as they grow.
pub const Context = struct {
    seqs: std.ArrayList(Sequence) = .empty,
    has_s: bool = false,
    exit_tensor_idx: c_int = 0,
    in_memcpys: std.ArrayList(StagingMemcpy) = .empty,
    out_memcpys: std.ArrayList(StagingMemcpy) = .empty,
    memsets: std.ArrayList(StagingMemset) = .empty,
    p: ?*CommandPool = null,

    /// `ctx->s`: the submission being recorded.
    pub fn s(self: *Context) *Submission {
        std.debug.assert(self.has_s);
        const seq = &self.seqs.items[self.seqs.items.len - 1];
        return &seq.items[seq.items.len - 1];
    }

    /// `ctx->s->buffer->buf`: the command buffer being recorded.
    pub fn cmd(self: *Context) vk.VkCommandBuffer {
        return self.s().buffer.?.buf;
    }

    fn clearSeqs(self: *Context) void {
        for (self.seqs.items) |*seq| {
            for (seq.items) |*sub| sub.deinit();
            seq.deinit(allocator);
        }
        self.seqs.clearRetainingCapacity();
    }

    /// The shared pointer's destructor.
    pub fn destroy(self: *Context) void {
        self.clearSeqs();
        self.seqs.deinit(allocator);
        self.in_memcpys.deinit(allocator);
        self.out_memcpys.deinit(allocator);
        self.memsets.deinit(allocator);
        allocator.destroy(self);
    }
};

fn oom() noreturn {
    impl.abort("ggml_vulkan: out of memory");
}

/// Ports `ggml_vk_create_temporary_context` (ggml-vulkan.cpp:3410 @5266f24da):
/// a context the caller owns and frees with `Context.destroy`.
pub fn createTemporaryContext(p: *CommandPool) *Context {
    const result = allocator.create(Context) catch oom();
    result.* = .{ .p = p };
    return result;
}

/// Ports `ggml_vk_create_cmd_buffer` (ggml-vulkan.cpp:3227 @5266f24da).
fn createCmdBuffer(device: *Device, p: *CommandPool) *CommandBuffer {
    var info = common.structure(vk.VkCommandBufferAllocateInfo, vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    info.commandPool = p.pool;
    info.level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    var buf: vk.VkCommandBuffer = null;
    common.checkOrAbort(d.vkAllocateCommandBuffers.?(device.device, &info, &buf), "vk::Device::allocateCommandBuffers");
    const cb = allocator.create(CommandBuffer) catch oom();
    cb.* = .{ .buf = buf, .use_counter = 0, .in_use = true };
    p.cmd_buffers.append(allocator, cb) catch oom();
    return cb;
}

/// Ports `ggml_vk_get_or_create_cmd_buffer` (ggml-vulkan.cpp:8224 @5266f24da).
pub fn getOrCreateCmdBuffer(device: *Device, pool: *CommandPool) *CommandBuffer {
    for (pool.cmd_buffers.items) |cb| {
        if (!cb.in_use) {
            cb.use_counter += 1;
            cb.in_use = true;
            return cb;
        }
    }
    return createCmdBuffer(device, pool);
}

/// Ports `ggml_vk_begin_submission` (ggml-vulkan.cpp:8235 @5266f24da).
pub fn beginSubmission(device: *Device, p: *CommandPool, one_time: bool) Submission {
    var sub: Submission = .{};
    sub.buffer = getOrCreateCmdBuffer(device, p);
    var info = common.structure(vk.VkCommandBufferBeginInfo, vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    info.flags = if (one_time) vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT else 0;
    common.checkOrAbort(d.vkBeginCommandBuffer.?(sub.buffer.?.buf, &info), "vk::CommandBuffer::begin");
    return sub;
}

/// Ports `ggml_vk_ctx_end` (ggml-vulkan.cpp:8304 @5266f24da).
pub fn ctxEnd(ctx: *Context) void {
    if (!ctx.has_s) return;
    common.checkOrAbort(d.vkEndCommandBuffer.?(ctx.cmd()), "vk::CommandBuffer::end");
    ctx.has_s = false;
}

/// Ports `ggml_vk_ctx_begin` (ggml-vulkan.cpp:8314 @5266f24da): starts a new
/// sequence of one submission.
pub fn ctxBegin(device: *Device, subctx: *Context) void {
    if (subctx.has_s) ctxEnd(subctx);
    var seq: Sequence = .empty;
    seq.append(allocator, beginSubmission(device, subctx.p.?, true)) catch oom();
    subctx.seqs.append(allocator, seq) catch oom();
    subctx.has_s = true;
}

/// `vk_queue_handle::submit`: a queue submission, serialised by the
/// handle's mutex unless the queue synchronises itself. A lost device
/// prints what is known about it before the C++'s rethrow.
pub fn queueSubmit(h: *m.device.QueueHandle, submits: []const vk.VkSubmitInfo, fence: vk.VkFence) void {
    h.lock();
    const result = d.vkQueueSubmit.?(h.queue, @intCast(submits.len), submits.ptr, fence);
    h.unlock();
    if (result == vk.VK_ERROR_DEVICE_LOST) printDeviceLostInfo(h.device);
    common.checkOrAbort(result, "vk::Queue::submit");
}

/// Ports `ggml_vk_submit` (ggml-vulkan.cpp:3238 @5266f24da): submits every
/// recorded sequence as timeline-semaphore submissions, then forgets them.
pub fn submit(ctx: *Context, fence: vk.VkFence) void {
    const h = ctx.p.?.q.?.handle;
    if (ctx.seqs.items.len == 0) {
        if (fence != null) queueSubmit(h, &.{}, fence);
        return;
    }

    var n: usize = 0;
    for (ctx.seqs.items) |seq| n += seq.items.len;

    const submit_infos = allocator.alloc(vk.VkSubmitInfo, n) catch oom();
    defer allocator.free(submit_infos);
    const tl_infos = allocator.alloc(vk.VkTimelineSemaphoreSubmitInfo, n) catch oom();
    defer allocator.free(tl_infos);
    var scratch: std.ArrayList(u64) = .empty;
    defer scratch.deinit(allocator);
    var sems: std.ArrayList(vk.VkSemaphore) = .empty;
    defer sems.deinit(allocator);
    var stages: std.ArrayList(vk.VkPipelineStageFlags) = .empty;
    defer stages.deinit(allocator);
    // Size the scratch arrays first: pointers into them must not move.
    var total_sems: usize = 0;
    for (ctx.seqs.items) |seq| for (seq.items) |sub| {
        total_sems += sub.wait_semaphores.items.len + sub.signal_semaphores.items.len;
    };
    scratch.ensureTotalCapacity(allocator, total_sems) catch oom();
    sems.ensureTotalCapacity(allocator, total_sems) catch oom();
    stages.ensureTotalCapacity(allocator, total_sems) catch oom();

    const stage_flags = ctx.p.?.q.?.stage_flags;
    var idx: usize = 0;
    for (ctx.seqs.items) |seq| for (seq.items) |*sub| {
        const nw = sub.wait_semaphores.items.len;
        const ns = sub.signal_semaphores.items.len;
        const wait_vals = scratch.items.len;
        const wait_sems = sems.items.len;
        const wait_stages = stages.items.len;
        for (sub.wait_semaphores.items) |sem| {
            stages.appendAssumeCapacity(stage_flags);
            scratch.appendAssumeCapacity(sem.value);
            sems.appendAssumeCapacity(sem.s);
        }
        const signal_vals = scratch.items.len;
        const signal_sems = sems.items.len;
        for (sub.signal_semaphores.items) |sem| {
            scratch.appendAssumeCapacity(sem.value);
            sems.appendAssumeCapacity(sem.s);
        }
        tl_infos[idx] = .{
            .sType = vk.VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreValueCount = @intCast(nw),
            .pWaitSemaphoreValues = scratch.items.ptr + wait_vals,
            .signalSemaphoreValueCount = @intCast(ns),
            .pSignalSemaphoreValues = scratch.items.ptr + signal_vals,
        };
        submit_infos[idx] = .{
            .sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &tl_infos[idx],
            .waitSemaphoreCount = @intCast(nw),
            .pWaitSemaphores = sems.items.ptr + wait_sems,
            .pWaitDstStageMask = stages.items.ptr + wait_stages,
            .commandBufferCount = 1,
            .pCommandBuffers = &sub.buffer.?.buf,
            .signalSemaphoreCount = @intCast(ns),
            .pSignalSemaphores = sems.items.ptr + signal_sems,
        };
        idx += 1;
    };

    queueSubmit(h, submit_infos, fence);
    ctx.clearSeqs();
    ctx.has_s = false;
}

/// `VK_CHECK` (ggml-vulkan.cpp:189 @5266f24da) around a wait: a lost device
/// prints what is known about it; any failure logs and throws, and nothing
/// catches it.
pub fn vkCheck(result: vk.VkResult, comptime msg: [:0]const u8, device: *Device) void {
    if (result == vk.VK_SUCCESS) return;
    if (result == vk.VK_ERROR_DEVICE_LOST) printDeviceLostInfo(device);
    impl.logError("ggml_vulkan: %s error %s\n", .{ @as([*:0]const u8, msg), common.resultString(result).ptr });
    impl.abort("ggml_vulkan: " ++ msg);
}

/// `device->device.waitForFences({ fence }, true, UINT64_MAX)` under
/// `VK_CHECK`, then `resetFences`: the end of every synchronous transfer.
pub fn waitAndResetFence(device: *Device, fence: vk.VkFence, comptime msg: [:0]const u8) void {
    vkCheck(d.vkWaitForFences.?(device.device, 1, &fence, vk.VK_TRUE, std.math.maxInt(u64)), msg, device);
    common.checkOrAbort(d.vkResetFences.?(device.device, 1, &fence), "vk::Device::resetFences");
}

/// Ports `ggml_vk_print_device_fault_info` (ggml-vulkan.cpp:1196 @5266f24da).
pub fn printDeviceFaultInfo(device: *Device) void {
    const pfn = device.pfn_vkGetDeviceFaultInfoEXT orelse return;
    if (!device.device_fault) return;
    var counts = common.structure(vk.VkDeviceFaultCountsEXT, vk.VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT);
    var res = pfn(device.device, &counts, null);
    if (res != vk.VK_SUCCESS) {
        impl.logError("ggml_vulkan: vkGetDeviceFaultInfoEXT (counts) failed: %d\n", .{res});
        return;
    }
    const address_infos = allocator.alloc(vk.VkDeviceFaultAddressInfoEXT, counts.addressInfoCount) catch oom();
    defer allocator.free(address_infos);
    const vendor_infos = allocator.alloc(vk.VkDeviceFaultVendorInfoEXT, counts.vendorInfoCount) catch oom();
    defer allocator.free(vendor_infos);
    @memset(address_infos, std.mem.zeroes(vk.VkDeviceFaultAddressInfoEXT));
    @memset(vendor_infos, std.mem.zeroes(vk.VkDeviceFaultVendorInfoEXT));
    var info = common.structure(vk.VkDeviceFaultInfoEXT, vk.VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT);
    info.pAddressInfos = address_infos.ptr;
    info.pVendorInfos = vendor_infos.ptr;
    res = pfn(device.device, &counts, &info);
    if (res != vk.VK_SUCCESS) {
        impl.logError("ggml_vulkan: vkGetDeviceFaultInfoEXT (info) failed: %d\n", .{res});
        return;
    }
    if (counts.addressInfoCount == 0 and counts.vendorInfoCount == 0 and info.description[0] == 0) return;
    if (info.description[0] != 0) {
        impl.logError("ggml_vulkan: device fault on %s: %s\n", .{ device.name.ptr, @as([*:0]const u8, @ptrCast(&info.description)) });
    }
    for (address_infos, 0..) |a, i| {
        impl.logCont("  address fault %u: type=%d address=0x%llx precision=0x%llx\n", .{ @as(c_uint, @intCast(i)), @as(c_int, @intCast(a.addressType)), @as(c_ulonglong, a.reportedAddress), @as(c_ulonglong, a.addressPrecision) });
    }
    for (vendor_infos, 0..) |v, i| {
        impl.logCont("  vendor fault %u: %s (code=0x%llx data=0x%llx)\n", .{ @as(c_uint, @intCast(i)), @as([*:0]const u8, @ptrCast(&v.description)), @as(c_ulonglong, v.vendorFaultCode), @as(c_ulonglong, v.vendorFaultData) });
    }
}

/// Ports `ggml_vk_print_device_lost_info` (ggml-vulkan.cpp:2247 @5266f24da).
pub fn printDeviceLostInfo(device: *Device) void {
    printDeviceFaultInfo(device);
    if (device.serialize_submissions and device.diag_cgraph != null and device.diag_prev_start >= 0) {
        impl.logError("ggml_vulkan: device lost on %s, likely caused by previous submission (nodes %d to %d):\n", .{ device.name.ptr, device.diag_prev_start, device.diag_prev_end });
        m.graph.printNodeList(device.diag_cgraph.?, device.diag_prev_start, device.diag_prev_end);
    } else {
        impl.logError("ggml_vulkan: device lost on %s\n", .{device.name.ptr});
    }
}

/// Ports `ggml_vk_command_pool_cleanup` (ggml-vulkan.cpp:3446 @5266f24da):
/// resets the pool and marks its buffers reusable.
pub fn commandPoolCleanup(device: *Device, p: *CommandPool) void {
    // Requires command buffers to be done
    common.checkOrAbort(d.vkResetCommandPool.?(device.device, p.pool, 0), "vk::Device::resetCommandPool");
    for (p.cmd_buffers.items) |cb| cb.in_use = false;
}

/// Ports `ggml_vk_queue_command_pools_cleanup` (ggml-vulkan.cpp:3458 @5266f24da).
pub fn queueCommandPoolsCleanup(device: *Device) void {
    // Arbitrary frequency to cleanup/reuse command buffers
    const cleanup_frequency = 10;
    if (device.compute_queue) |q| if (q.cmd_pool.buffersInUse() >= cleanup_frequency) commandPoolCleanup(device, &q.cmd_pool);
    if (device.transfer_queue) |q| if (q.cmd_pool.buffersInUse() >= cleanup_frequency) commandPoolCleanup(device, &q.cmd_pool);
}

/// The memory barrier `ggml_vk_sync_buffers` records (ggml-vulkan.cpp:3708 @5266f24da),
/// without the backend context's bookkeeping, which `context.syncBuffers`
/// adds.
pub fn syncBuffersBarrier(subctx: *Context) void {
    const q = subctx.p.?.q.?;
    const access: vk.VkAccessFlags = if (!q.transfer_only)
        vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT | vk.VK_ACCESS_TRANSFER_READ_BIT | vk.VK_ACCESS_TRANSFER_WRITE_BIT
    else
        vk.VK_ACCESS_TRANSFER_READ_BIT | vk.VK_ACCESS_TRANSFER_WRITE_BIT;
    var barrier = common.structure(vk.VkMemoryBarrier, vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER);
    barrier.srcAccessMask = access;
    barrier.dstAccessMask = access;
    d.vkCmdPipelineBarrier.?(subctx.cmd(), q.stage_flags, q.stage_flags, 0, 1, &barrier, 0, null, 0, null);
}

/// Ports `ggml_vk_reset_event` (ggml-vulkan.cpp:3730 @5266f24da).
pub fn resetEvent(ctx: *Context, event: vk.VkEvent) void {
    d.vkCmdResetEvent.?(ctx.cmd(), event, ctx.p.?.q.?.stage_flags);
}

/// Ports `ggml_vk_set_event` (ggml-vulkan.cpp:3739 @5266f24da).
pub fn setEvent(ctx: *Context, event: vk.VkEvent) void {
    d.vkCmdSetEvent.?(ctx.cmd(), event, ctx.p.?.q.?.stage_flags);
}

/// Ports `ggml_vk_wait_events` (ggml-vulkan.cpp:3748 @5266f24da).
pub fn waitEvents(ctx: *Context, events: []const vk.VkEvent) void {
    if (events.len == 0) return;
    const stage = ctx.p.?.q.?.stage_flags;
    d.vkCmdWaitEvents.?(ctx.cmd(), @intCast(events.len), events.ptr, stage, stage, 0, null, 0, null, 0, null);
}

/// Ports `deferred_memcpy` (ggml-vulkan.cpp:8385 @5266f24da).
pub fn deferredMemcpy(dst: [*]u8, src: [*]const u8, size: usize, memcpys: ?*std.ArrayList(StagingMemcpy)) void {
    if (memcpys) |list| {
        list.append(allocator, .{ .dst = dst, .src = src, .n = size }) catch oom();
    } else {
        @memcpy(dst[0..size], src[0..size]);
    }
}

/// Ports `deferred_memset` (ggml-vulkan.cpp:8393 @5266f24da).
pub fn deferredMemset(dst: [*]u8, val: u32, size: usize, memsets: ?*std.ArrayList(StagingMemset)) void {
    if (memsets) |list| {
        list.append(allocator, .{ .dst = dst, .val = val, .n = size }) catch oom();
    } else {
        @memset(dst[0..size], @truncate(val));
    }
}

/// Runs a context's deferred host copies: `for (auto& cpy : ...) memcpy(...)`.
pub fn runMemcpys(list: []const StagingMemcpy) void {
    for (list) |cpy| @memcpy(cpy.dst[0..cpy.n], cpy.src[0..cpy.n]);
}

/// Runs a context's deferred host fills.
pub fn runMemsets(list: []const StagingMemset) void {
    for (list) |ms| @memset(ms.dst[0..ms.n], @truncate(ms.val));
}

/// The `vk::CommandBuffer::copyBuffer` of a slice list.
pub fn copyBuffer(cmdbuf: vk.VkCommandBuffer, src: vk.VkBuffer, dst: vk.VkBuffer, slices: []const vk.VkBufferCopy) void {
    d.vkCmdCopyBuffer.?(cmdbuf, src, dst, @intCast(slices.len), slices.ptr);
}

/// The C++'s direct `vkCmdCopyBuffer` -- the loader's exported symbol, not
/// the dispatcher's pointer (`NOTES.md`, "Stage 7: the measurement").
pub extern fn vkCmdCopyBuffer(commandBuffer: vk.VkCommandBuffer, srcBuffer: vk.VkBuffer, dstBuffer: vk.VkBuffer, regionCount: u32, pRegions: [*]const vk.VkBufferCopy) void;

// -----------------------------------------------------------------------------
// Synchronous transfers

/// The `std::vector<vk::BufferCopy>` the 2-D transfers build: one slice
/// when both pitches equal the width, one per row otherwise.
fn slices2d(src_off: usize, dst_off: usize, spitch: usize, dpitch: usize, width: usize, height: usize, single: bool) []vk.VkBufferCopy {
    if (single) {
        const out = allocator.alloc(vk.VkBufferCopy, 1) catch oom();
        out[0] = .{ .srcOffset = src_off, .dstOffset = dst_off, .size = width * height };
        return out;
    }
    const out = allocator.alloc(vk.VkBufferCopy, height) catch oom();
    for (out, 0..) |*s, i| s.* = .{ .srcOffset = src_off + i * spitch, .dstOffset = dst_off + i * dpitch, .size = width };
    return out;
}

/// Ports `ggml_vk_ensure_sync_staging_buffer` for a device (ggml-vulkan.cpp:8401 @5266f24da).
pub fn ensureSyncStagingBuffer(device: *Device, size: usize) void {
    if (device.sync_staging == null or device.sync_staging.?.size < size) {
        m.buffer.logMemory("ggml_vk_ensure_sync_staging_buffer(%zu)", .{size});
        m.buffer.destroyBuffer(&device.sync_staging);
        device.sync_staging = m.buffer.createBufferCheck(device, size, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
}

/// Ports `ggml_vk_buffer_write_2d_async` (ggml-vulkan.cpp:8518 @5266f24da).
///
/// Return: false when `src` is not pinned and `sync_staging` is off -- the
/// caller must fall back.
pub fn bufferWrite2dAsync(subctx: *Context, dst: *Buffer, offset: usize, src: [*]const u8, spitch: usize, dpitch: usize, width: usize, height: usize, sync_staging: bool) bool {
    // Check if src is pinned memory
    var buf_offset: usize = 0;
    if (m.buffer.hostGet(dst.device, src, &buf_offset)) |buf| {
        const slices = slices2d(buf_offset, offset, spitch, dpitch, width, height, width == spitch and width == dpitch);
        defer allocator.free(slices);
        syncBuffersBarrier(subctx);
        copyBuffer(subctx.cmd(), buf.buffer, dst.buffer, slices);
        return true;
    }

    if (!sync_staging) return false;

    // Staging buffer required
    const staging_size = width * height;
    ensureSyncStagingBuffer(dst.device, staging_size);
    const staging = dst.device.sync_staging.?;

    const slices = if (width == dpitch) slices2d(0, offset, 0, 0, staging_size, 1, true) else slices2d(0, offset, width, dpitch, width, height, false);
    defer allocator.free(slices);
    syncBuffersBarrier(subctx);
    copyBuffer(subctx.cmd(), staging.buffer, dst.buffer, slices);

    const sptr: [*]u8 = @ptrCast(staging.ptr.?);
    if (width == spitch) {
        deferredMemcpy(sptr, src, staging_size, &subctx.in_memcpys);
    } else {
        for (0..height) |i| deferredMemcpy(sptr + i * width, src + i * spitch, width, &subctx.in_memcpys);
    }
    return true;
}

/// Ports `ggml_vk_buffer_write_async` (ggml-vulkan.cpp:8586 @5266f24da).
pub fn bufferWriteAsync(subctx: *Context, dst: *Buffer, offset: usize, src: [*]const u8, size: usize, sync_staging: bool) bool {
    return bufferWrite2dAsync(subctx, dst, offset, src, size, size, size, 1, sync_staging);
}

/// Ports `ggml_vk_buffer_write_2d` (ggml-vulkan.cpp:8591 @5266f24da).
pub fn bufferWrite2d(dst: *Buffer, offset: usize, src: [*]const u8, spitch: usize, dpitch: usize, width: usize, height: usize) void {
    // Buffer is already mapped
    if (dst.memory_property_flags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0) {
        impl.assert(dst.memory_property_flags & vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT != 0, "dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent");
        const p: [*]u8 = @ptrCast(dst.ptr.?);
        if (width == spitch and width == dpitch) {
            @memcpy(p[offset..][0 .. width * height], src[0 .. width * height]);
        } else {
            for (0..height) |i| @memcpy(p[offset + i * dpitch ..][0..width], src[i * spitch ..][0..width]);
        }
        return;
    }
    const device = dst.device;
    device.lock();
    defer device.unlock();

    const subctx = createTemporaryContext(&device.transfer_queue.?.cmd_pool);
    defer subctx.destroy();
    ctxBegin(device, subctx);
    const ret = bufferWrite2dAsync(subctx, dst, offset, src, spitch, dpitch, width, height, true);
    impl.assert(ret, "ret");
    ctxEnd(subctx);

    runMemcpys(subctx.in_memcpys.items);
    runMemsets(subctx.memsets.items);

    submit(subctx, device.fence);
    waitAndResetFence(device, device.fence, "vk_buffer_write_2d waitForFences");
    queueCommandPoolsCleanup(device);
}

/// Ports `ggml_vk_buffer_write` (ggml-vulkan.cpp:8628 @5266f24da).
pub fn bufferWrite(dst: *Buffer, offset: usize, src: [*]const u8, size: usize) void {
    bufferWrite2d(dst, offset, src, size, size, size, 1);
}

/// Ports `ggml_vk_buffer_read_2d_async` (ggml-vulkan.cpp:8633 @5266f24da).
pub fn bufferRead2dAsync(subctx: *Context, src: *Buffer, offset: usize, dst: [*]u8, spitch: usize, dpitch: usize, width: usize, height: usize, sync_staging: bool) bool {
    impl.assert(width > 0, "width > 0");
    impl.assert(height > 0, "height > 0");

    // Check if dst is pinned memory
    var buf_offset: usize = 0;
    const pinned = m.buffer.hostGet(src.device, dst, &buf_offset);

    if (pinned) |buf| {
        const slices = slices2d(offset, buf_offset, spitch, dpitch, width, height, width == spitch and width == dpitch);
        defer allocator.free(slices);
        syncBuffersBarrier(subctx);
        copyBuffer(subctx.cmd(), src.buffer, buf.buffer, slices);
        return true;
    }

    if (!sync_staging) return false;

    // Fall back to staging buffer
    const staging_size = width * height;
    ensureSyncStagingBuffer(src.device, staging_size);
    const staging = src.device.sync_staging.?;

    const slices = if (width == spitch) slices2d(offset, 0, 0, 0, staging_size, 1, true) else slices2d(offset, 0, spitch, width, width, height, false);
    defer allocator.free(slices);
    syncBuffersBarrier(subctx);
    copyBuffer(subctx.cmd(), src.buffer, staging.buffer, slices);

    const sptr: [*]const u8 = @ptrCast(staging.ptr.?);
    if (width == dpitch) {
        deferredMemcpy(dst, sptr, staging_size, &subctx.out_memcpys);
    } else {
        for (0..height) |i| deferredMemcpy(dst + i * dpitch, sptr + i * width, width, &subctx.out_memcpys);
    }
    return true;
}

/// Ports `ggml_vk_buffer_read_async` (ggml-vulkan.cpp:8708 @5266f24da).
pub fn bufferReadAsync(subctx: *Context, src: *Buffer, offset: usize, dst: [*]u8, size: usize, sync_staging: bool) bool {
    return bufferRead2dAsync(subctx, src, offset, dst, size, size, size, 1, sync_staging);
}

/// Ports `ggml_vk_buffer_read_2d` (ggml-vulkan.cpp:8712 @5266f24da).
pub fn bufferRead2d(src: *Buffer, offset: usize, dst: [*]u8, spitch: usize, dpitch: usize, width: usize, height: usize) void {
    const device = src.device;
    // On a UMA device read through the mapping, after a barrier makes the
    // GPU's writes visible to the host.
    if (src.memory_property_flags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0 and device.uma) {
        impl.assert(src.memory_property_flags & vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT != 0, "src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent");
        device.lock();
        defer device.unlock();
        const subctx = createTemporaryContext(&device.compute_queue.?.cmd_pool);
        defer subctx.destroy();
        ctxBegin(device, subctx);
        var barrier = common.structure(vk.VkMemoryBarrier, vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER);
        barrier.srcAccessMask = vk.VK_ACCESS_SHADER_WRITE_BIT | vk.VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = vk.VK_ACCESS_HOST_READ_BIT;
        d.vkCmdPipelineBarrier.?(subctx.cmd(), vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, null, 0, null);
        ctxEnd(subctx);
        submit(subctx, device.fence);
        waitAndResetFence(device, device.fence, "vk_buffer_read_2d uma waitForFences");
        queueCommandPoolsCleanup(device);

        const p: [*]const u8 = @ptrCast(src.ptr.?);
        if (width == spitch and width == dpitch) {
            @memcpy(dst[0 .. width * height], p[offset..][0 .. width * height]);
        } else {
            for (0..height) |i| @memcpy(dst[i * dpitch ..][0..width], p[offset + i * spitch ..][0..width]);
        }
        return;
    }

    device.lock();
    defer device.unlock();
    const subctx = createTemporaryContext(&device.transfer_queue.?.cmd_pool);
    defer subctx.destroy();
    ctxBegin(device, subctx);
    const ret = bufferRead2dAsync(subctx, src, offset, dst, spitch, dpitch, width, height, true);
    impl.assert(ret, "ret");
    ctxEnd(subctx);

    submit(subctx, device.fence);
    waitAndResetFence(device, device.fence, "vk_buffer_read_2d waitForFences");
    queueCommandPoolsCleanup(device);

    runMemcpys(subctx.out_memcpys.items);
}

/// Ports `ggml_vk_buffer_read` (ggml-vulkan.cpp:8765 @5266f24da).
pub fn bufferRead(src: *Buffer, offset: usize, dst: [*]u8, size: usize) void {
    bufferRead2d(src, offset, dst, size, size, size, 1);
}

/// Ports `ggml_vk_buffer_copy_async` (ggml-vulkan.cpp:8770 @5266f24da).
pub fn bufferCopyAsync(ctx: *Context, dst: *Buffer, dst_offset: usize, src: *Buffer, src_offset: usize, size: usize) void {
    // Make sure both buffers are on same device
    impl.assert(src.device == dst.device, "src->device == dst->device");
    const bc: vk.VkBufferCopy = .{ .srcOffset = src_offset, .dstOffset = dst_offset, .size = size };
    vkCmdCopyBuffer(ctx.cmd(), src.buffer, dst.buffer, 1, @ptrCast(&bc));
}

/// Ports `ggml_vk_buffer_copy` (ggml-vulkan.cpp:8780 @5266f24da).
pub fn bufferCopy(dst: *Buffer, dst_offset: usize, src: *Buffer, src_offset: usize, size: usize) void {
    if (src.device == dst.device) {
        const device = src.device;
        device.lock();
        defer device.unlock();
        // Copy within the device
        const subctx = createTemporaryContext(&device.transfer_queue.?.cmd_pool);
        defer subctx.destroy();
        ctxBegin(device, subctx);
        bufferCopyAsync(subctx, dst, dst_offset, src, src_offset, size);
        ctxEnd(subctx);
        submit(subctx, device.fence);
        waitAndResetFence(device, device.fence, "vk_buffer_copy waitForFences");
        queueCommandPoolsCleanup(device);
    } else {
        // Copy device to device
        ensureSyncStagingBuffer(src.device, size);
        // Copy to src staging buffer
        bufferCopy(src.device.sync_staging.?, 0, src, src_offset, size);
        // Copy to dst buffer
        bufferWrite(dst, dst_offset, @ptrCast(src.device.sync_staging.?.ptr.?), size);
    }
}

/// Ports `ggml_vk_buffer_memset_async` (ggml-vulkan.cpp:8805 @5266f24da).
pub fn bufferMemsetAsync(ctx: *Context, dst: *Buffer, offset: usize, val: u32, size: usize) void {
    if (dst.memory_property_flags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0 and dst.device.uma) {
        deferredMemset(@as([*]u8, @ptrCast(dst.ptr.?)) + offset, val, size, &ctx.memsets);
        return;
    }
    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    d.vkCmdFillBuffer.?(ctx.cmd(), dst.buffer, offset, size, val);
}

/// Ports `ggml_vk_buffer_memset` (ggml-vulkan.cpp:8818 @5266f24da).
pub fn bufferMemset(dst: *Buffer, offset: usize, val: u32, size: usize) void {
    if (dst.memory_property_flags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0 and dst.device.uma) {
        @memset((@as([*]u8, @ptrCast(dst.ptr.?)) + offset)[0..size], @truncate(val));
        return;
    }
    const device = dst.device;
    device.lock();
    defer device.unlock();
    const subctx = createTemporaryContext(&device.transfer_queue.?.cmd_pool);
    defer subctx.destroy();
    ctxBegin(device, subctx);
    d.vkCmdFillBuffer.?(subctx.cmd(), dst.buffer, offset, size, val);
    ctxEnd(subctx);
    submit(subctx, device.fence);
    waitAndResetFence(device, device.fence, "vk_memset waitForFences");
    queueCommandPoolsCleanup(device);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
