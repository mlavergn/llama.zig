//! Command buffers, submissions and the contexts that collect them: how
//! recorded work reaches a queue, and the synchronous transfers built on
//! top.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp`, and
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-types.h` at v1.0.0
//! (`b1b770d90`). A `vk_context` is a `std::shared_ptr` in the C++, kept alive
//! either by the backend context's garbage list or by the one caller that
//! made it ("temporary"); here the owner frees it explicitly.

const std = @import("std");
const ggml = @import("../module.zig");

/// Ports `VK_DEVICE_DESCRIPTOR_POOL_SIZE` (ggml-vulkan-types.h:193 @b1b770d90).
pub const descriptor_pool_size = 256;

/// Ports `vk_semaphore` (ggml-vulkan-types.h:1106 @b1b770d90).
pub const Semaphore = struct {
    s: ggml.vulkan.common.vk.VkSemaphore = null,
    value: u64 = 0,
};

/// Ports `vk_submission` (ggml-vulkan-types.h:1122 @b1b770d90).
pub const Submission = struct {
    buffer: ?*ggml.vulkan.device.CommandBuffer = null,
    wait_semaphores: std.ArrayList(Semaphore) = .empty,
    signal_semaphores: std.ArrayList(Semaphore) = .empty,
    // submit waits until *ready >= ready_value
    ready: ?*const std.atomic.Value(u64) = null,
    ready_value: u64 = 0,

    fn deinit(self: *Submission) void {
        self.wait_semaphores.deinit(std.heap.c_allocator);
        self.signal_semaphores.deinit(std.heap.c_allocator);
    }
};

/// Ports `vk_sequence` (ggml-vulkan-types.h:1131 @b1b770d90).
pub const Sequence = std.ArrayList(Submission);

/// Ports `vk_staging_memcpy` (ggml-vulkan-types.h:1141 @b1b770d90).
pub const StagingMemcpy = struct { dst: [*]u8, src: [*]const u8, n: usize };

/// Ports `vk_staging_memset` (ggml-vulkan-types.h:1149 @b1b770d90).
pub const StagingMemset = struct { dst: [*]u8, val: u32, n: usize };

/// Ports `vk_context_struct` (ggml-vulkan-types.h:1172 @b1b770d90).
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
    // labels open on this context, reopened on each new command buffer
    debug_labels: std.ArrayList([:0]u8) = .empty,
    p: ?*ggml.vulkan.device.CommandPool = null,

    /// `ctx->s`: the submission being recorded.
    pub fn s(self: *Context) *Submission {
        std.debug.assert(self.has_s);
        const seq = &self.seqs.items[self.seqs.items.len - 1];
        return &seq.items[seq.items.len - 1];
    }

    /// `ctx->s->buffer->buf`: the command buffer being recorded.
    pub fn cmd(self: *Context) ggml.vulkan.common.vk.VkCommandBuffer {
        return self.s().buffer.?.buf;
    }

    fn clearSeqs(self: *Context) void {
        for (self.seqs.items) |*seq| {
            for (seq.items) |*sub| sub.deinit();
            seq.deinit(std.heap.c_allocator);
        }
        self.seqs.clearRetainingCapacity();
    }

    /// The shared pointer's destructor.
    pub fn destroy(self: *Context) void {
        self.clearSeqs();
        self.seqs.deinit(std.heap.c_allocator);
        self.in_memcpys.deinit(std.heap.c_allocator);
        self.out_memcpys.deinit(std.heap.c_allocator);
        self.memsets.deinit(std.heap.c_allocator);
        for (self.debug_labels.items) |l| std.heap.c_allocator.free(l);
        self.debug_labels.deinit(std.heap.c_allocator);
        std.heap.c_allocator.destroy(self);
    }
};

fn oom() noreturn {
    ggml.impl.abort("ggml_vulkan: out of memory");
}

/// Ports `ggml_vk_create_temporary_context` (ggml-vulkan.cpp:1085 @b1b770d90):
/// a context the caller owns and frees with `Context.destroy`.
pub fn createTemporaryContext(p: *ggml.vulkan.device.CommandPool) *Context {
    const result = std.heap.c_allocator.create(Context) catch oom();
    result.* = .{ .p = p };
    return result;
}

/// Ports `ggml_vk_create_cmd_buffer` (ggml-vulkan.cpp:893 @b1b770d90).
fn createCmdBuffer(device: *ggml.vulkan.device.Device, p: *ggml.vulkan.device.CommandPool) *ggml.vulkan.device.CommandBuffer {
    var info = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkCommandBufferAllocateInfo, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    info.commandPool = p.pool;
    info.level = ggml.vulkan.common.vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    var buf: ggml.vulkan.common.vk.VkCommandBuffer = null;
    ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkAllocateCommandBuffers.?(device.device, &info, &buf), "vk::Device::allocateCommandBuffers");
    const cb = std.heap.c_allocator.create(ggml.vulkan.device.CommandBuffer) catch oom();
    cb.* = .{ .buf = buf, .use_counter = 0, .in_use = true };
    p.cmd_buffers.append(std.heap.c_allocator, cb) catch oom();
    return cb;
}

/// Ports `ggml_vk_get_or_create_cmd_buffer` (ggml-vulkan.cpp:6036 @b1b770d90).
pub fn getOrCreateCmdBuffer(device: *ggml.vulkan.device.Device, pool: *ggml.vulkan.device.CommandPool) *ggml.vulkan.device.CommandBuffer {
    for (pool.cmd_buffers.items) |cb| {
        if (!cb.in_use) {
            cb.use_counter += 1;
            cb.in_use = true;
            return cb;
        }
    }
    return createCmdBuffer(device, pool);
}

/// Ports `ggml_vk_begin_submission` (ggml-vulkan.cpp:6047 @b1b770d90).
pub fn beginSubmission(device: *ggml.vulkan.device.Device, p: *ggml.vulkan.device.CommandPool, one_time: bool) Submission {
    var sub: Submission = .{};
    sub.buffer = getOrCreateCmdBuffer(device, p);
    var info = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkCommandBufferBeginInfo, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    info.flags = if (one_time) ggml.vulkan.common.vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT else 0;
    ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkBeginCommandBuffer.?(sub.buffer.?.buf, &info), "vk::CommandBuffer::begin");
    return sub;
}

/// Ports `ggml_vk_cmd_label_begin` (ggml-vulkan.cpp:6059 @b1b770d90).
fn cmdLabelBegin(buf: ggml.vulkan.common.vk.VkCommandBuffer, name: [*:0]const u8) void {
    var label = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkDebugUtilsLabelEXT, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT);
    label.pLabelName = name;
    label.color = .{ 1.0, 1.0, 1.0, 1.0 };
    ggml.vulkan.instance.instance.pfn_vkCmdBeginDebugUtilsLabelEXT.?(buf, &label);
}

/// Ports `ggml_vk_debug_label` (ggml-vulkan-types.h:1486 @b1b770d90): a debug
/// region for GPU profilers, a no-op unless `GGML_VK_DEBUG_MARKERS` turned
/// on `VK_EXT_debug_utils`. At most one of `subctx` and `qhandle` is set,
/// depending on the scope the label was opened in; the C++'s destructor is
/// a `defer label.close()` here.
pub const DebugLabel = struct {
    subctx: ?*Context = null,
    qhandle: ?*ggml.vulkan.device.QueueHandle = null,

    /// One region per dispatch, e.g. "matmul_q4_k_f32_f16acc_aligned_m (192,8,1)".
    /// RGP cannot recover the pipeline name on its own, it only has the hash.
    pub fn dispatch(ctx: *Context, pipeline_name: []const u8, wg0: u32, wg1: u32, wg2: u32) DebugLabel {
        var self: DebugLabel = .{};
        if (!ggml.vulkan.instance.instance.debug_utils_support or !ctx.has_s) return self;
        const name = std.fmt.allocPrintSentinel(std.heap.c_allocator, "{s} ({d},{d},{d})", .{ pipeline_name, wg0, wg1, wg2 }, 0) catch oom();
        self.begin(ctx, name);
        return self;
    }

    /// One region per graph node; fused nodes are joined with '+', e.g.
    /// "RMS_NORM+MUL+ROPE Qcur-19".
    pub fn node(ctx: *Context, cgraph: *const ggml.impl.CGraph, node_idx: c_int, n_fused: c_int) DebugLabel {
        var self: DebugLabel = .{};
        if (!ggml.vulkan.instance.instance.debug_utils_support or !ctx.has_s) return self;
        var name: std.ArrayList(u8) = .empty;
        const first: *const ggml.impl.ggml_c.ggml_tensor = cgraph.nodes[@intCast(node_idx)].?;
        name.appendSlice(std.heap.c_allocator, std.mem.span(ggml.impl.ggml_c.ggml_op_name(first.op))) catch oom();
        var i: c_int = 1;
        while (i <= n_fused) : (i += 1) {
            name.append(std.heap.c_allocator, '+') catch oom();
            name.appendSlice(std.heap.c_allocator, std.mem.span(ggml.impl.ggml_c.ggml_op_name(cgraph.nodes[@intCast(node_idx + i)].?.op))) catch oom();
        }
        name.append(std.heap.c_allocator, ' ') catch oom();
        name.appendSlice(std.heap.c_allocator, std.mem.sliceTo(&first.name, 0)) catch oom();
        self.begin(ctx, name.toOwnedSliceSentinel(std.heap.c_allocator, 0) catch oom());
        return self;
    }

    /// One region per graph evaluation, opened on the queue instead of a
    /// command buffer so it spans every submit the evaluation makes.
    pub fn queue(handle: ?*ggml.vulkan.device.QueueHandle, name: [*:0]const u8) DebugLabel {
        var self: DebugLabel = .{};
        const h = handle orelse return self;
        if (!ggml.vulkan.instance.instance.debug_utils_support) return self;
        var label = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkDebugUtilsLabelEXT, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT);
        label.pLabelName = name;
        label.color = .{ 1.0, 1.0, 1.0, 1.0 };
        self.qhandle = h;
        h.lock();
        defer h.unlock();
        ggml.vulkan.instance.instance.pfn_vkQueueBeginDebugUtilsLabelEXT.?(h.queue, &label);
        return self;
    }

    /// Call before the command buffer can end; also what the C++'s
    /// destructor does.
    pub fn close(self: *DebugLabel) void {
        if (self.subctx) |sc| {
            // close on the current command buffer, which may differ from the one begin used
            if (sc.has_s) ggml.vulkan.instance.instance.pfn_vkCmdEndDebugUtilsLabelEXT.?(sc.cmd());
            if (sc.debug_labels.pop()) |l| std.heap.c_allocator.free(l);
            self.subctx = null;
        }
        if (self.qhandle) |h| {
            h.lock();
            defer h.unlock();
            ggml.vulkan.instance.instance.pfn_vkQueueEndDebugUtilsLabelEXT.?(h.queue);
            self.qhandle = null;
        }
    }

    // the constructors check this too, so the name is not built when markers are off
    fn begin(self: *DebugLabel, ctx: *Context, name: [:0]u8) void {
        self.subctx = ctx;
        ctx.debug_labels.append(std.heap.c_allocator, name) catch oom();
        cmdLabelBegin(ctx.cmd(), name.ptr);
    }
};

/// Ports `ggml_vk_ctx_end` (ggml-vulkan.cpp:6066 @b1b770d90).
pub fn ctxEnd(ctx: *Context) void {
    if (!ctx.has_s) return;
    // close open labels so this buffer is balanced; reopened in ggml_vk_ctx_begin
    if (ggml.vulkan.instance.instance.debug_utils_support) {
        for (ctx.debug_labels.items) |_| ggml.vulkan.instance.instance.pfn_vkCmdEndDebugUtilsLabelEXT.?(ctx.cmd());
        // the enclosing per-command-buffer region
        ggml.vulkan.instance.instance.pfn_vkCmdEndDebugUtilsLabelEXT.?(ctx.cmd());
    }
    ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkEndCommandBuffer.?(ctx.cmd()), "vk::CommandBuffer::end");
    ctx.has_s = false;
}

/// Ports `ggml_vk_ctx_begin` (ggml-vulkan.cpp:6085 @b1b770d90): starts a new
/// sequence of one submission.
pub fn ctxBegin(device: *ggml.vulkan.device.Device, subctx: *Context) void {
    if (subctx.has_s) ctxEnd(subctx);
    var seq: Sequence = .empty;
    seq.append(std.heap.c_allocator, beginSubmission(device, subctx.p.?, true)) catch oom();
    subctx.seqs.append(std.heap.c_allocator, seq) catch oom();
    subctx.has_s = true;

    if (ggml.vulkan.instance.instance.debug_utils_support) {
        // outermost region, one per command buffer, so the gaps between submits stand out
        var buf: [32]u8 = undefined;
        const name = std.fmt.bufPrintSentinel(&buf, "submit {d}", .{device.debug_cmdbuf_idx}, 0) catch unreachable;
        device.debug_cmdbuf_idx +%= 1;
        cmdLabelBegin(subctx.cmd(), name.ptr);

        // reopen labels left open when the previous command buffer was submitted
        for (subctx.debug_labels.items) |label| cmdLabelBegin(subctx.cmd(), label.ptr);
    }
}

/// The `submit` of `vk_queue_handle_synchronized` and
/// `vk_queue_handle_unsynchronized` (ggml-vulkan-types.h:337, 345 @b1b770d90): a
/// queue submission, serialised by the handle's mutex unless the queue
/// synchronises itself, and first by the device's submit mutex where the
/// NVIDIA workaround set one. A lost device prints what is known about it
/// before the C++'s rethrow, which nothing catches here.
pub fn queueSubmit(h: *ggml.vulkan.device.QueueHandle, submits: []const ggml.vulkan.common.vk.VkSubmitInfo, fence: ggml.vulkan.common.vk.VkFence) void {
    queueSubmitChecked(h, submits, fence) catch ggml.vulkan.common.abortLastError();
}

/// `queueSubmit` where the C++'s caller catches the throw:
/// `error.VulkanError`, with `common.last_error` set.
pub fn queueSubmitChecked(h: *ggml.vulkan.device.QueueHandle, submits: []const ggml.vulkan.common.vk.VkSubmitInfo, fence: ggml.vulkan.common.vk.VkFence) ggml.vulkan.common.VkError!void {
    // Workaround for NVIDIA driver bug
    if (h.device_submit_mutex) |dm| dm.lock();
    h.lock();
    const result = ggml.vulkan.common.dispatch.vkQueueSubmit.?(h.queue, @intCast(submits.len), submits.ptr, fence);
    h.unlock();
    if (h.device_submit_mutex) |dm| dm.unlock();
    if (result == ggml.vulkan.common.vk.VK_ERROR_DEVICE_LOST) printDeviceLostInfo(h.device);
    return ggml.vulkan.common.check(result, "vk::Queue::submit");
}

/// Ports `ggml_vk_submit` (ggml-vulkan.cpp:904 @b1b770d90): submits every
/// recorded sequence as timeline-semaphore submissions, then forgets them.
pub fn submit(ctx: *Context, fence: ggml.vulkan.common.vk.VkFence) void {
    const h = ctx.p.?.q.?.handle;
    if (ctx.seqs.items.len == 0) {
        if (fence != null) queueSubmit(h, &.{}, fence);
        return;
    }

    var n: usize = 0;
    for (ctx.seqs.items) |seq| n += seq.items.len;

    const submit_infos = std.heap.c_allocator.alloc(ggml.vulkan.common.vk.VkSubmitInfo, n) catch oom();
    defer std.heap.c_allocator.free(submit_infos);
    const tl_infos = std.heap.c_allocator.alloc(ggml.vulkan.common.vk.VkTimelineSemaphoreSubmitInfo, n) catch oom();
    defer std.heap.c_allocator.free(tl_infos);
    var scratch: std.ArrayList(u64) = .empty;
    defer scratch.deinit(std.heap.c_allocator);
    var sems: std.ArrayList(ggml.vulkan.common.vk.VkSemaphore) = .empty;
    defer sems.deinit(std.heap.c_allocator);
    var stages: std.ArrayList(ggml.vulkan.common.vk.VkPipelineStageFlags) = .empty;
    defer stages.deinit(std.heap.c_allocator);
    // Size the scratch arrays first: pointers into them must not move.
    var total_sems: usize = 0;
    for (ctx.seqs.items) |seq| for (seq.items) |sub| {
        total_sems += sub.wait_semaphores.items.len + sub.signal_semaphores.items.len;
    };
    scratch.ensureTotalCapacity(std.heap.c_allocator, total_sems) catch oom();
    sems.ensureTotalCapacity(std.heap.c_allocator, total_sems) catch oom();
    stages.ensureTotalCapacity(std.heap.c_allocator, total_sems) catch oom();

    const stage_flags = ctx.p.?.q.?.stage_flags;
    var idx: usize = 0;
    for (ctx.seqs.items) |seq| for (seq.items) |*sub| {
        if (sub.ready) |ready| {
            while (ready.load(.acquire) < sub.ready_value) ggml.vulkan.common.yield();
        }
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
            .sType = ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreValueCount = @intCast(nw),
            .pWaitSemaphoreValues = scratch.items.ptr + wait_vals,
            .signalSemaphoreValueCount = @intCast(ns),
            .pSignalSemaphoreValues = scratch.items.ptr + signal_vals,
        };
        submit_infos[idx] = .{
            .sType = ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
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

/// `VK_CHECK` (ggml-vulkan-types.h:195 @b1b770d90) around a wait: a lost device
/// prints what is known about it; any failure logs and throws, and nothing
/// catches it.
pub fn vkCheck(result: ggml.vulkan.common.vk.VkResult, comptime msg: [:0]const u8, device: *ggml.vulkan.device.Device) void {
    vkCheckChecked(result, msg, device) catch ggml.impl.abort("ggml_vulkan: " ++ msg);
}

/// `VK_CHECK` where the C++'s caller catches the throw: logs as `vkCheck`
/// does, then returns `error.VulkanError` with `common.last_error` naming
/// `"ggml_vulkan: " msg`, the exception's message.
pub fn vkCheckChecked(result: ggml.vulkan.common.vk.VkResult, comptime msg: [:0]const u8, device: *ggml.vulkan.device.Device) ggml.vulkan.common.VkError!void {
    if (result == ggml.vulkan.common.vk.VK_SUCCESS) return;
    if (result == ggml.vulkan.common.vk.VK_ERROR_DEVICE_LOST) printDeviceLostInfo(device);
    ggml.impl.logError("ggml_vulkan: %s error %s\n", .{ @as([*:0]const u8, msg), ggml.vulkan.common.resultString(result).ptr });
    return ggml.vulkan.common.check(result, "ggml_vulkan: " ++ msg);
}

/// `device->device.waitForFences({ fence }, true, UINT64_MAX)` under
/// `VK_CHECK`, then `resetFences`: the end of every synchronous transfer.
pub fn waitAndResetFence(device: *ggml.vulkan.device.Device, fence: ggml.vulkan.common.vk.VkFence, comptime msg: [:0]const u8) void {
    vkCheck(ggml.vulkan.common.dispatch.vkWaitForFences.?(device.device, 1, &fence, ggml.vulkan.common.vk.VK_TRUE, std.math.maxInt(u64)), msg, device);
    ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkResetFences.?(device.device, 1, &fence), "vk::Device::resetFences");
}

/// Ports `ggml_vk_print_device_fault_info` (ggml-vulkan.cpp:149 @b1b770d90).
pub fn printDeviceFaultInfo(device: *ggml.vulkan.device.Device) void {
    const pfn = device.pfn_vkGetDeviceFaultInfoEXT orelse return;
    if (!device.device_fault) return;
    var counts = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkDeviceFaultCountsEXT, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT);
    var res = pfn(device.device, &counts, null);
    if (res != ggml.vulkan.common.vk.VK_SUCCESS) {
        ggml.impl.logError("ggml_vulkan: vkGetDeviceFaultInfoEXT (counts) failed: %d\n", .{res});
        return;
    }
    const address_infos = std.heap.c_allocator.alloc(ggml.vulkan.common.vk.VkDeviceFaultAddressInfoEXT, counts.addressInfoCount) catch oom();
    defer std.heap.c_allocator.free(address_infos);
    const vendor_infos = std.heap.c_allocator.alloc(ggml.vulkan.common.vk.VkDeviceFaultVendorInfoEXT, counts.vendorInfoCount) catch oom();
    defer std.heap.c_allocator.free(vendor_infos);
    @memset(address_infos, std.mem.zeroes(ggml.vulkan.common.vk.VkDeviceFaultAddressInfoEXT));
    @memset(vendor_infos, std.mem.zeroes(ggml.vulkan.common.vk.VkDeviceFaultVendorInfoEXT));
    const vendor_binary = std.heap.c_allocator.alloc(u8, @intCast(counts.vendorBinarySize)) catch oom();
    defer std.heap.c_allocator.free(vendor_binary);
    @memset(vendor_binary, 0);
    var info = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkDeviceFaultInfoEXT, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT);
    info.pAddressInfos = address_infos.ptr;
    info.pVendorInfos = vendor_infos.ptr;
    info.pVendorBinaryData = if (vendor_binary.len == 0) null else vendor_binary.ptr;
    res = pfn(device.device, &counts, &info);
    if (res != ggml.vulkan.common.vk.VK_SUCCESS) {
        ggml.impl.logError("ggml_vulkan: vkGetDeviceFaultInfoEXT (info) failed: %d\n", .{res});
        return;
    }
    if (counts.addressInfoCount == 0 and counts.vendorInfoCount == 0 and info.description[0] == 0) return;
    if (info.description[0] != 0) {
        ggml.impl.logError("ggml_vulkan: device fault on %s: %s\n", .{ device.name.ptr, @as([*:0]const u8, @ptrCast(&info.description)) });
    }
    for (address_infos, 0..) |a, i| {
        ggml.impl.logCont("  address fault %u: type=%d address=0x%llx precision=0x%llx\n", .{ @as(c_uint, @intCast(i)), @as(c_int, @intCast(a.addressType)), @as(c_ulonglong, a.reportedAddress), @as(c_ulonglong, a.addressPrecision) });
    }
    for (vendor_infos, 0..) |v, i| {
        ggml.impl.logCont("  vendor fault %u: %s (code=0x%llx data=0x%llx)\n", .{ @as(c_uint, @intCast(i)), @as([*:0]const u8, @ptrCast(&v.description)), @as(c_ulonglong, v.vendorFaultCode), @as(c_ulonglong, v.vendorFaultData) });
    }
}

/// Ports `ggml_vk_print_device_lost_info` (ggml-vulkan.cpp:250 @b1b770d90).
pub fn printDeviceLostInfo(device: *ggml.vulkan.device.Device) void {
    printDeviceFaultInfo(device);
    if (device.serialize_submissions and device.diag_cgraph != null and device.diag_prev_start >= 0) {
        ggml.impl.logError("ggml_vulkan: device lost on %s, likely caused by previous submission (nodes %d to %d):\n", .{ device.name.ptr, device.diag_prev_start, device.diag_prev_end });
        ggml.vulkan.graph.printNodeList(device.diag_cgraph.?, device.diag_prev_start, device.diag_prev_end);
    } else {
        ggml.impl.logError("ggml_vulkan: device lost on %s\n", .{device.name.ptr});
    }
}

/// Ports `ggml_vk_command_pool_cleanup` (ggml-vulkan.cpp:1123 @b1b770d90):
/// resets the pool and marks its buffers reusable.
pub fn commandPoolCleanup(device: *ggml.vulkan.device.Device, p: *ggml.vulkan.device.CommandPool) void {
    // Requires command buffers to be done
    ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkResetCommandPool.?(device.device, p.pool, 0), "vk::Device::resetCommandPool");
    for (p.cmd_buffers.items) |cb| cb.in_use = false;
}

/// Ports `ggml_vk_queue_command_pools_cleanup` (ggml-vulkan.cpp:1135 @b1b770d90).
pub fn queueCommandPoolsCleanup(device: *ggml.vulkan.device.Device) void {
    // Arbitrary frequency to cleanup/reuse command buffers
    const cleanup_frequency = 10;
    if (device.compute_queue) |q| if (q.cmd_pool.buffersInUse() >= cleanup_frequency) commandPoolCleanup(device, &q.cmd_pool);
    if (device.transfer_queue) |q| if (q.cmd_pool.buffersInUse() >= cleanup_frequency) commandPoolCleanup(device, &q.cmd_pool);
}

/// The memory barrier `ggml_vk_sync_buffers` records (ggml-vulkan.cpp:1153 @b1b770d90),
/// without the backend context's bookkeeping, which `context.syncBuffers`
/// adds.
pub fn syncBuffersBarrier(subctx: *Context) void {
    const q = subctx.p.?.q.?;
    const access: ggml.vulkan.common.vk.VkAccessFlags = if (!q.transfer_only)
        ggml.vulkan.common.vk.VK_ACCESS_SHADER_READ_BIT | ggml.vulkan.common.vk.VK_ACCESS_SHADER_WRITE_BIT | ggml.vulkan.common.vk.VK_ACCESS_TRANSFER_READ_BIT | ggml.vulkan.common.vk.VK_ACCESS_TRANSFER_WRITE_BIT
    else
        ggml.vulkan.common.vk.VK_ACCESS_TRANSFER_READ_BIT | ggml.vulkan.common.vk.VK_ACCESS_TRANSFER_WRITE_BIT;
    var barrier = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkMemoryBarrier, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER);
    barrier.srcAccessMask = access;
    barrier.dstAccessMask = access;
    ggml.vulkan.common.dispatch.vkCmdPipelineBarrier.?(subctx.cmd(), q.stage_flags, q.stage_flags, 0, 1, &barrier, 0, null, 0, null);
}

/// Ports `ggml_vk_reset_event` (ggml-vulkan.cpp:1175 @b1b770d90).
pub fn resetEvent(ctx: *Context, event: ggml.vulkan.common.vk.VkEvent) void {
    ggml.vulkan.common.dispatch.vkCmdResetEvent.?(ctx.cmd(), event, ctx.p.?.q.?.stage_flags);
}

/// Ports `ggml_vk_set_event` (ggml-vulkan.cpp:1184 @b1b770d90).
pub fn setEvent(ctx: *Context, event: ggml.vulkan.common.vk.VkEvent) void {
    ggml.vulkan.common.dispatch.vkCmdSetEvent.?(ctx.cmd(), event, ctx.p.?.q.?.stage_flags);
}

/// Ports `ggml_vk_wait_events` (ggml-vulkan.cpp:1193 @b1b770d90).
pub fn waitEvents(ctx: *Context, events: []const ggml.vulkan.common.vk.VkEvent) void {
    if (events.len == 0) return;
    const stage = ctx.p.?.q.?.stage_flags;
    ggml.vulkan.common.dispatch.vkCmdWaitEvents.?(ctx.cmd(), @intCast(events.len), events.ptr, stage, stage, 0, null, 0, null, 0, null);
}

/// Ports `deferred_memcpy` (ggml-vulkan.cpp:6164 @b1b770d90).
pub fn deferredMemcpy(dst: [*]u8, src: [*]const u8, size: usize, memcpys: ?*std.ArrayList(StagingMemcpy)) void {
    if (memcpys) |list| {
        list.append(std.heap.c_allocator, .{ .dst = dst, .src = src, .n = size }) catch oom();
    } else {
        @memcpy(dst[0..size], src[0..size]);
    }
}

/// Ports `deferred_memset` (ggml-vulkan.cpp:6172 @b1b770d90).
pub fn deferredMemset(dst: [*]u8, val: u32, size: usize, memsets: ?*std.ArrayList(StagingMemset)) void {
    if (memsets) |list| {
        list.append(std.heap.c_allocator, .{ .dst = dst, .val = val, .n = size }) catch oom();
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
pub fn copyBuffer(cmdbuf: ggml.vulkan.common.vk.VkCommandBuffer, src: ggml.vulkan.common.vk.VkBuffer, dst: ggml.vulkan.common.vk.VkBuffer, slices: []const ggml.vulkan.common.vk.VkBufferCopy) void {
    ggml.vulkan.common.dispatch.vkCmdCopyBuffer.?(cmdbuf, src, dst, @intCast(slices.len), slices.ptr);
}

/// The C++'s direct `vkCmdCopyBuffer` -- the loader's exported symbol, not
/// the dispatcher's pointer (measured with `nm` on the reference).
pub extern fn vkCmdCopyBuffer(commandBuffer: ggml.vulkan.common.vk.VkCommandBuffer, srcBuffer: ggml.vulkan.common.vk.VkBuffer, dstBuffer: ggml.vulkan.common.vk.VkBuffer, regionCount: u32, pRegions: [*]const ggml.vulkan.common.vk.VkBufferCopy) void;

// -----------------------------------------------------------------------------
// Synchronous transfers

/// The `std::vector<vk::BufferCopy>` the 2-D transfers build: one slice
/// when both pitches equal the width, one per row otherwise.
fn slices2d(src_off: usize, dst_off: usize, spitch: usize, dpitch: usize, width: usize, height: usize, single: bool) []ggml.vulkan.common.vk.VkBufferCopy {
    if (single) {
        const out = std.heap.c_allocator.alloc(ggml.vulkan.common.vk.VkBufferCopy, 1) catch oom();
        out[0] = .{ .srcOffset = src_off, .dstOffset = dst_off, .size = width * height };
        return out;
    }
    const out = std.heap.c_allocator.alloc(ggml.vulkan.common.vk.VkBufferCopy, height) catch oom();
    for (out, 0..) |*s, i| s.* = .{ .srcOffset = src_off + i * spitch, .dstOffset = dst_off + i * dpitch, .size = width };
    return out;
}

/// Ports `ggml_vk_ensure_sync_staging_buffer` for a device (ggml-vulkan-buffers.cpp:387 @b1b770d90).
pub fn ensureSyncStagingBuffer(device: *ggml.vulkan.device.Device, size: usize) void {
    if (device.sync_staging == null or device.sync_staging.?.size < size) {
        ggml.vulkan.buffer.logMemory("ggml_vk_ensure_sync_staging_buffer(%zu)", .{size});
        ggml.vulkan.buffer.destroyBuffer(&device.sync_staging);
        device.sync_staging = ggml.vulkan.buffer.createBufferCheck(device, size, ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT, ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
}

/// Ports `ggml_vk_buffer_write_2d_async` (ggml-vulkan-buffers.cpp:531 @b1b770d90).
///
/// Return: false when `src` is not pinned and `sync_staging` is off -- the
/// caller must fall back.
pub fn bufferWrite2dAsync(subctx: *Context, dst: *ggml.vulkan.buffer.Buffer, offset: usize, src: [*]const u8, spitch: usize, dpitch: usize, width: usize, height: usize, sync_staging: bool) bool {
    // Check if src is pinned memory
    var buf_offset: usize = 0;
    if (ggml.vulkan.buffer.hostGet(dst.device, src, &buf_offset)) |buf| {
        const slices = slices2d(buf_offset, offset, spitch, dpitch, width, height, width == spitch and width == dpitch);
        defer std.heap.c_allocator.free(slices);
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
    defer std.heap.c_allocator.free(slices);
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

/// Ports `ggml_vk_buffer_write_async` (ggml-vulkan-buffers.cpp:599 @b1b770d90).
pub fn bufferWriteAsync(subctx: *Context, dst: *ggml.vulkan.buffer.Buffer, offset: usize, src: [*]const u8, size: usize, sync_staging: bool) bool {
    return bufferWrite2dAsync(subctx, dst, offset, src, size, size, size, 1, sync_staging);
}

/// Ports `ggml_vk_begin_sync_ctx` (ggml-vulkan-buffers.cpp:407 @b1b770d90):
/// a temporary context from `pool`, begun. The caller destroys it.
fn beginSyncCtx(device: *ggml.vulkan.device.Device, pool: *ggml.vulkan.device.CommandPool) *Context {
    const subctx = createTemporaryContext(pool);
    ctxBegin(device, subctx);
    return subctx;
}

/// Ports `ggml_vk_submit_sync` (ggml-vulkan-buffers.cpp:413 @b1b770d90): ends
/// `subctx`, runs its staged host writes, submits it and waits, then runs
/// its staged host reads.
fn submitSync(device: *ggml.vulkan.device.Device, subctx: *Context) void {
    ctxEnd(subctx);

    runMemcpys(subctx.in_memcpys.items);
    runMemsets(subctx.memsets.items);

    submit(subctx, device.fence);
    waitAndResetFence(device, device.fence, "vk_submit_sync waitForFences");
    queueCommandPoolsCleanup(device);

    runMemcpys(subctx.out_memcpys.items);
}

/// Ports `ggml_vk_buffer_write_2d` (ggml-vulkan-buffers.cpp:604 @b1b770d90).
pub fn bufferWrite2d(dst: *ggml.vulkan.buffer.Buffer, offset: usize, src: [*]const u8, spitch: usize, dpitch: usize, width: usize, height: usize) void {
    // Buffer is already mapped
    if (dst.memory_property_flags & ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0) {
        ggml.impl.assert(dst.memory_property_flags & ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT != 0, "dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent");
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

    const subctx = beginSyncCtx(device, &device.transfer_queue.?.cmd_pool);
    defer subctx.destroy();
    const ret = bufferWrite2dAsync(subctx, dst, offset, src, spitch, dpitch, width, height, true);
    ggml.impl.assert(ret, "ret");
    submitSync(device, subctx);
}

/// Ports `ggml_vk_buffer_write` (ggml-vulkan-buffers.cpp:627 @b1b770d90).
pub fn bufferWrite(dst: *ggml.vulkan.buffer.Buffer, offset: usize, src: [*]const u8, size: usize) void {
    bufferWrite2d(dst, offset, src, size, size, size, 1);
}

/// Ports `ggml_vk_buffer_read_2d_async` (ggml-vulkan-buffers.cpp:632 @b1b770d90).
pub fn bufferRead2dAsync(subctx: *Context, src: *ggml.vulkan.buffer.Buffer, offset: usize, dst: [*]u8, spitch: usize, dpitch: usize, width: usize, height: usize, sync_staging: bool) bool {
    ggml.impl.assert(width > 0, "width > 0");
    ggml.impl.assert(height > 0, "height > 0");

    // Check if dst is pinned memory
    var buf_offset: usize = 0;
    const pinned = ggml.vulkan.buffer.hostGet(src.device, dst, &buf_offset);

    if (pinned) |buf| {
        const slices = slices2d(offset, buf_offset, spitch, dpitch, width, height, width == spitch and width == dpitch);
        defer std.heap.c_allocator.free(slices);
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
    defer std.heap.c_allocator.free(slices);
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

/// Ports `ggml_vk_buffer_read_async` (ggml-vulkan-buffers.cpp:707 @b1b770d90).
pub fn bufferReadAsync(subctx: *Context, src: *ggml.vulkan.buffer.Buffer, offset: usize, dst: [*]u8, size: usize, sync_staging: bool) bool {
    return bufferRead2dAsync(subctx, src, offset, dst, size, size, size, 1, sync_staging);
}

/// Ports `ggml_vk_buffer_host_read_direct` (ggml-vulkan-buffers.cpp:711 @b1b770d90).
fn bufferHostReadDirect(buf: *const ggml.vulkan.buffer.Buffer) bool {
    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    const slow_host_read = buf.device.vendor_id == ggml.vulkan.common.vendor_id.amd and
        buf.memory_property_flags & ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT == 0;
    return buf.memory_property_flags & ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0 and buf.device.uma and !slow_host_read;
}

/// The host-read barrier both direct read paths record: makes the GPU's
/// shader and transfer writes visible to the host.
fn hostReadBarrier(subctx: *Context) void {
    var barrier = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkMemoryBarrier, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER);
    barrier.srcAccessMask = ggml.vulkan.common.vk.VK_ACCESS_SHADER_WRITE_BIT | ggml.vulkan.common.vk.VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = ggml.vulkan.common.vk.VK_ACCESS_HOST_READ_BIT;
    ggml.vulkan.common.dispatch.vkCmdPipelineBarrier.?(subctx.cmd(), ggml.vulkan.common.vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | ggml.vulkan.common.vk.VK_PIPELINE_STAGE_TRANSFER_BIT, ggml.vulkan.common.vk.VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, null, 0, null);
}

/// Ports `ggml_vk_buffer_read_2d` (ggml-vulkan-buffers.cpp:721 @b1b770d90).
pub fn bufferRead2d(src: *ggml.vulkan.buffer.Buffer, offset: usize, dst: [*]u8, spitch: usize, dpitch: usize, width: usize, height: usize) void {
    const device = src.device;
    if (bufferHostReadDirect(src)) {
        ggml.impl.assert(src.memory_property_flags & ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT != 0, "src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent");
        device.lock();
        defer device.unlock();
        const subctx = beginSyncCtx(device, &device.compute_queue.?.cmd_pool);
        defer subctx.destroy();
        hostReadBarrier(subctx);
        submitSync(device, subctx);

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
    const subctx = beginSyncCtx(device, &device.transfer_queue.?.cmd_pool);
    defer subctx.destroy();
    const ret = bufferRead2dAsync(subctx, src, offset, dst, spitch, dpitch, width, height, true);
    ggml.impl.assert(ret, "ret");
    submitSync(device, subctx);
}

/// Ports `ggml_vk_buffer_read` (ggml-vulkan-buffers.cpp:755 @b1b770d90).
pub fn bufferRead(src: *ggml.vulkan.buffer.Buffer, offset: usize, dst: [*]u8, size: usize) void {
    bufferRead2d(src, offset, dst, size, size, size, 1);
}

/// Ports `ggml_vk_buffer_copy_async` (ggml-vulkan-buffers.cpp:760 @b1b770d90).
pub fn bufferCopyAsync(ctx: *Context, dst: *ggml.vulkan.buffer.Buffer, dst_offset: usize, src: *ggml.vulkan.buffer.Buffer, src_offset: usize, size: usize) void {
    // Make sure both buffers are on same device
    ggml.impl.assert(src.device == dst.device, "src->device == dst->device");
    const bc: ggml.vulkan.common.vk.VkBufferCopy = .{ .srcOffset = src_offset, .dstOffset = dst_offset, .size = size };
    vkCmdCopyBuffer(ctx.cmd(), src.buffer, dst.buffer, 1, @ptrCast(&bc));
}

/// Ports `ggml_vk_buffer_copy` (ggml-vulkan-buffers.cpp:770 @b1b770d90).
pub fn bufferCopy(dst: *ggml.vulkan.buffer.Buffer, dst_offset: usize, src: *ggml.vulkan.buffer.Buffer, src_offset: usize, size: usize) void {
    if (src.device == dst.device) {
        const device = src.device;
        device.lock();
        defer device.unlock();
        // Copy within the device
        const subctx = beginSyncCtx(device, &device.transfer_queue.?.cmd_pool);
        defer subctx.destroy();
        bufferCopyAsync(subctx, dst, dst_offset, src, src_offset, size);
        submitSync(device, subctx);
    } else {
        // Copy device to device
        ensureSyncStagingBuffer(src.device, size);
        // Copy to src staging buffer
        bufferCopy(src.device.sync_staging.?, 0, src, src_offset, size);
        // Copy to dst buffer
        bufferWrite(dst, dst_offset, @ptrCast(src.device.sync_staging.?.ptr.?), size);
    }
}

/// Ports `ggml_vk_buffer_memset_async` (ggml-vulkan-buffers.cpp:790 @b1b770d90).
pub fn bufferMemsetAsync(ctx: *Context, dst: *ggml.vulkan.buffer.Buffer, offset: usize, val: u32, size: usize) void {
    if (dst.memory_property_flags & ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0 and dst.device.uma) {
        deferredMemset(@as([*]u8, @ptrCast(dst.ptr.?)) + offset, val, size, &ctx.memsets);
        return;
    }
    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ggml.vulkan.common.dispatch.vkCmdFillBuffer.?(ctx.cmd(), dst.buffer, offset, size, val);
}

/// Ports `ggml_vk_buffer_memset` (ggml-vulkan-buffers.cpp:803 @b1b770d90).
pub fn bufferMemset(dst: *ggml.vulkan.buffer.Buffer, offset: usize, val: u32, size: usize) void {
    if (dst.memory_property_flags & ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0 and dst.device.uma) {
        @memset((@as([*]u8, @ptrCast(dst.ptr.?)) + offset)[0..size], @truncate(val));
        return;
    }
    const device = dst.device;
    device.lock();
    defer device.unlock();
    const subctx = beginSyncCtx(device, &device.transfer_queue.?.cmd_pool);
    defer subctx.destroy();
    ggml.vulkan.common.dispatch.vkCmdFillBuffer.?(subctx.cmd(), dst.buffer, offset, size, val);
    submitSync(device, subctx);
}

/// Ports `vk_buffer_region` (ggml-vulkan-types.h:1157 @b1b770d90): one
/// device range and the host memory it is written from or read into.
pub const BufferRegion = struct {
    buf: *ggml.vulkan.buffer.Buffer,
    offset: usize,
    host: ?*anyopaque,
    size: usize,
};

/// Ports `vk_buffer_copy_region` (ggml-vulkan-types.h:1164 @b1b770d90).
pub const BufferCopyRegion = struct {
    dst: *ggml.vulkan.buffer.Buffer,
    dst_offset: usize,
    src: *ggml.vulkan.buffer.Buffer,
    src_offset: usize,
    size: usize,
};

fn copyOne(cmd: ggml.vulkan.common.vk.VkCommandBuffer, src: ggml.vulkan.common.vk.VkBuffer, dst: ggml.vulkan.common.vk.VkBuffer, src_offset: usize, dst_offset: usize, size: usize) void {
    const bc: ggml.vulkan.common.vk.VkBufferCopy = .{ .srcOffset = src_offset, .dstOffset = dst_offset, .size = size };
    vkCmdCopyBuffer(cmd, src, dst, 1, @ptrCast(&bc));
}

/// Ports `ggml_vk_staging_size_n` (ggml-vulkan-buffers.cpp:818 @b1b770d90):
/// the staging a batch needs -- every unpinned region's bytes, capped at
/// one suballocation block, but never less than its largest region.
fn stagingSizeN(device: *ggml.vulkan.device.Device, regions: []const BufferRegion) usize {
    var total: usize = 0;
    var max: usize = 0;
    for (regions) |r| {
        var pinned_offset: usize = 0;
        if (ggml.vulkan.buffer.hostGet(device, r.host, &pinned_offset) != null) continue;
        total += r.size;
        max = @max(max, r.size);
    }
    return @max(max, @min(total, @as(usize, @intCast(device.suballocation_block_size))));
}

/// Ports `ggml_vk_buffer_write_n_async` (ggml-vulkan-buffers.cpp:835 @b1b770d90):
/// records host-to-device copies, pinned memory directly and the rest
/// through `staging` from `staging_offset` on.
///
/// Return: the number of regions recorded; it stops when staging is full.
pub fn bufferWriteNAsync(subctx: *Context, regions: []const BufferRegion, staging: ?*ggml.vulkan.buffer.Buffer, staging_offset: ?*usize) usize {
    syncBuffersBarrier(subctx);

    var i: usize = 0;
    while (i < regions.len) : (i += 1) {
        const r = regions[i];

        var pinned_offset: usize = 0;
        const pinned = ggml.vulkan.buffer.hostGet(r.buf.device, r.host, &pinned_offset);

        if (pinned) |p| {
            copyOne(subctx.cmd(), p.buffer, r.buf.buffer, pinned_offset, r.offset, r.size);
            continue;
        }

        const st = staging orelse break;
        if (staging_offset.?.* + r.size > st.size) break;

        copyOne(subctx.cmd(), st.buffer, r.buf.buffer, staging_offset.?.*, r.offset, r.size);
        deferredMemcpy(@as([*]u8, @ptrCast(st.ptr.?)) + staging_offset.?.*, @ptrCast(r.host.?), r.size, &subctx.in_memcpys);
        staging_offset.?.* += r.size;
    }
    return i;
}

/// Ports `ggml_vk_buffer_read_n_async` (ggml-vulkan-buffers.cpp:864 @b1b770d90):
/// `bufferWriteNAsync` the other way.
///
/// Return: the number of regions recorded; it stops when staging is full.
pub fn bufferReadNAsync(subctx: *Context, regions: []const BufferRegion, staging: ?*ggml.vulkan.buffer.Buffer, staging_offset: ?*usize) usize {
    syncBuffersBarrier(subctx);

    var i: usize = 0;
    while (i < regions.len) : (i += 1) {
        const r = regions[i];

        var pinned_offset: usize = 0;
        const pinned = ggml.vulkan.buffer.hostGet(r.buf.device, r.host, &pinned_offset);

        if (pinned) |p| {
            copyOne(subctx.cmd(), r.buf.buffer, p.buffer, r.offset, pinned_offset, r.size);
            continue;
        }

        const st = staging orelse break;
        if (staging_offset.?.* + r.size > st.size) break;

        copyOne(subctx.cmd(), r.buf.buffer, st.buffer, r.offset, staging_offset.?.*, r.size);
        deferredMemcpy(@ptrCast(r.host.?), @as([*]const u8, @ptrCast(st.ptr.?)) + staging_offset.?.*, r.size, &subctx.out_memcpys);
        staging_offset.?.* += r.size;
    }
    return i;
}

/// Ports `ggml_vk_buffer_copy_n_async` (ggml-vulkan-buffers.cpp:892 @b1b770d90).
pub fn bufferCopyNAsync(subctx: *Context, regions: []const BufferCopyRegion) void {
    syncBuffersBarrier(subctx);

    for (regions) |r| {
        ggml.impl.assert(r.src.device == r.dst.device, "r.src->device == r.dst->device");
        copyOne(subctx.cmd(), r.src.buffer, r.dst.buffer, r.src_offset, r.dst_offset, r.size);
    }
}

/// Ports `ggml_vk_buffer_write_n` (ggml-vulkan-buffers.cpp:903 @b1b770d90):
/// many host-to-device writes, mapped buffers by `memcpy` and the rest in as
/// few synchronous submissions as the staging buffer allows.
pub fn bufferWriteN(regions: []const BufferRegion) void {
    var pending: std.ArrayList(BufferRegion) = .empty;
    defer pending.deinit(std.heap.c_allocator);

    for (regions) |r| {
        if (r.size == 0) continue;
        if (r.buf.memory_property_flags & ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0) {
            ggml.impl.assert(r.buf.memory_property_flags & ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT != 0, "r.buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent");
            @memcpy((@as([*]u8, @ptrCast(r.buf.ptr.?)) + r.offset)[0..r.size], @as([*]const u8, @ptrCast(r.host.?))[0..r.size]);
            continue;
        }
        ggml.impl.assert(pending.items.len == 0 or r.buf.device == pending.items[0].buf.device, "pending.empty() || r.buf->device == pending[0].buf->device");
        pending.append(std.heap.c_allocator, r) catch oom();
    }

    if (pending.items.len == 0) return;

    const device = pending.items[0].buf.device;
    device.lock();
    defer device.unlock();
    const staging_size = stagingSizeN(device, pending.items);
    if (staging_size > 0) ensureSyncStagingBuffer(device, staging_size);

    var i: usize = 0;
    while (i < pending.items.len) {
        const subctx = beginSyncCtx(device, &device.transfer_queue.?.cmd_pool);
        defer subctx.destroy();
        var staging_offset: usize = 0;
        const recorded = bufferWriteNAsync(subctx, pending.items[i..], device.sync_staging, &staging_offset);
        ggml.impl.assert(recorded > 0, "recorded > 0");
        submitSync(device, subctx);
        i += recorded;
    }
}

/// Ports `ggml_vk_buffer_read_n` (ggml-vulkan-buffers.cpp:942 @b1b770d90):
/// many device-to-host reads, host-mapped UMA buffers through the mapping
/// after one barrier, the rest staged as `bufferWriteN` stages writes.
pub fn bufferReadN(regions: []const BufferRegion) void {
    var direct: std.ArrayList(BufferRegion) = .empty;
    defer direct.deinit(std.heap.c_allocator);
    var pending: std.ArrayList(BufferRegion) = .empty;
    defer pending.deinit(std.heap.c_allocator);

    for (regions) |r| {
        if (r.size == 0) continue;
        ggml.impl.assert((direct.items.len == 0 and pending.items.len == 0) or r.buf.device == (if (direct.items.len == 0) pending.items[0] else direct.items[0]).buf.device, "(direct.empty() && pending.empty()) || r.buf->device == (direct.empty() ? pending[0] : direct[0]).buf->device");
        if (bufferHostReadDirect(r.buf)) {
            ggml.impl.assert(r.buf.memory_property_flags & ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT != 0, "r.buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent");
            direct.append(std.heap.c_allocator, r) catch oom();
        } else {
            pending.append(std.heap.c_allocator, r) catch oom();
        }
    }

    if (direct.items.len == 0 and pending.items.len == 0) return;

    const device = (if (direct.items.len == 0) pending.items[0] else direct.items[0]).buf.device;
    device.lock();
    defer device.unlock();

    if (direct.items.len != 0) {
        const subctx = beginSyncCtx(device, &device.compute_queue.?.cmd_pool);
        defer subctx.destroy();
        hostReadBarrier(subctx);
        submitSync(device, subctx);

        for (direct.items) |r| {
            @memcpy(@as([*]u8, @ptrCast(r.host.?))[0..r.size], (@as([*]const u8, @ptrCast(r.buf.ptr.?)) + r.offset)[0..r.size]);
        }
    }

    if (pending.items.len == 0) return;

    const staging_size = stagingSizeN(device, pending.items);
    if (staging_size > 0) ensureSyncStagingBuffer(device, staging_size);

    var i: usize = 0;
    while (i < pending.items.len) {
        const subctx = beginSyncCtx(device, &device.transfer_queue.?.cmd_pool);
        defer subctx.destroy();
        var staging_offset: usize = 0;
        const recorded = bufferReadNAsync(subctx, pending.items[i..], device.sync_staging, &staging_offset);
        ggml.impl.assert(recorded > 0, "recorded > 0");
        submitSync(device, subctx);
        i += recorded;
    }
}

/// Ports `ggml_vk_buffer_copy_n` (ggml-vulkan-buffers.cpp:1003 @b1b770d90):
/// many copies, cross-device ones one at a time through staging and the
/// rest in one submission.
pub fn bufferCopyN(regions: []const BufferCopyRegion) void {
    var same_device: std.ArrayList(BufferCopyRegion) = .empty;
    defer same_device.deinit(std.heap.c_allocator);

    for (regions) |r| {
        if (r.size == 0) continue;
        if (r.src.device != r.dst.device) {
            bufferCopy(r.dst, r.dst_offset, r.src, r.src_offset, r.size);
            continue;
        }
        ggml.impl.assert(same_device.items.len == 0 or r.dst.device == same_device.items[0].dst.device, "same_device.empty() || r.dst->device == same_device[0].dst->device");
        same_device.append(std.heap.c_allocator, r) catch oom();
    }

    if (same_device.items.len == 0) return;

    const device = same_device.items[0].dst.device;
    device.lock();
    defer device.unlock();
    const subctx = beginSyncCtx(device, &device.transfer_queue.?.cmd_pool);
    defer subctx.destroy();
    bufferCopyNAsync(subctx, same_device.items);
    submitSync(device, subctx);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
