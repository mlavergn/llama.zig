//! The `ggml_backend` a Vulkan device hands out: its interface table, the
//! asynchronous transfers, events, and the buffer wrapped around host
//! memory.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp`, and
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-types.h` at v1.0.0
//! (`b1b770d90`).

const std = @import("std");
const ggml = @import("../module.zig");

fn oom() noreturn {
    ggml.impl.abort("ggml_vulkan: out of memory");
}

fn backendCtx(backend: ggml.impl.ggml_c.ggml_backend_t) *ggml.vulkan.context.BackendCtx {
    return @ptrCast(@alignCast(backend.*.context));
}

/// Ports `ggml_backend_vk_name` (ggml-vulkan.cpp:13706 @b1b770d90).
fn name(backend: ggml.impl.ggml_c.ggml_backend_t) callconv(.c) [*c]const u8 {
    return backendCtx(backend).name.ptr;
}

/// Ports `ggml_backend_vk_free` (ggml-vulkan.cpp:13712 @b1b770d90).
fn free(backend: ggml.impl.ggml_c.ggml_backend_t) callconv(.c) void {
    const ctx = backendCtx(backend);
    ggml.vulkan.context.cleanup(ctx);
    std.heap.c_allocator.destroy(ctx);
    std.heap.c_allocator.destroy(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend, backend));
}

/// Ports `ggml_backend_vk_get_default_buffer_type` (ggml-vulkan.cpp:13722 @b1b770d90).
fn getDefaultBufferType(backend: ggml.impl.ggml_c.ggml_backend_t) callconv(.c) ggml.impl.ggml_c.ggml_backend_buffer_type_t {
    return &backendCtx(backend).device.buffer_type;
}

/// The staging copies `set_tensor_2d_async` and `get_tensor_2d_async` make
/// when the transfer could not go through a mapping or pinned memory.
fn stagingSlices(off: usize, stride: usize, size: usize, n_copies: usize, to_staging: bool) []ggml.vulkan.common.vk.VkBufferCopy {
    if (size == stride) {
        const out = std.heap.c_allocator.alloc(ggml.vulkan.common.vk.VkBufferCopy, 1) catch oom();
        out[0] = if (to_staging) .{ .srcOffset = off, .dstOffset = 0, .size = size * n_copies } else .{ .srcOffset = 0, .dstOffset = off, .size = size * n_copies };
        return out;
    }
    const out = std.heap.c_allocator.alloc(ggml.vulkan.common.vk.VkBufferCopy, n_copies) catch oom();
    for (out, 0..) |*s, i| {
        s.* = if (to_staging) .{ .srcOffset = off + i * stride, .dstOffset = i * size, .size = size } else .{ .srcOffset = i * size, .dstOffset = off + i * stride, .size = size };
    }
    return out;
}

fn assertBufferType(backend: ggml.impl.ggml_c.ggml_backend_t, tensor: *const ggml.impl.ggml_c.ggml_tensor) void {
    ggml.impl.assert(tensor.buffer.*.buft == getDefaultBufferType(backend) or tensor.buffer.*.buft == ggml.vulkan.buffer.hostBufferType(), "(tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && \"unsupported buffer type\"");
}

/// Ports `ggml_backend_vk_set_tensor_2d_async` (ggml-vulkan.cpp:13728 @b1b770d90).
fn setTensor2dAsync(backend: ggml.impl.ggml_c.ggml_backend_t, tensor_c: [*c]ggml.impl.ggml_c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    const ctx = backendCtx(backend);
    const tensor = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, tensor_c);
    assertBufferType(backend, tensor);
    if (size == 0) return;

    const cpy_ctx = if (ctx.device.async_use_transfer_queue) ggml.vulkan.context.getTransferCtx(ctx) else ggml.vulkan.context.getComputeCtx(ctx);
    const buf = ggml.vulkan.buffer.tensorBuffer(tensor);
    const dst_offset = ggml.vulkan.buffer.tensorOffset(tensor) + tensor.view_offs + offset;
    const src: [*]const u8 = @ptrCast(data.?);

    if (ggml.vulkan.submit.bufferWrite2dAsync(cpy_ctx, buf, dst_offset, src, stride_data, stride_tensor, size, n_copies, false)) return;

    const staging_size = size * n_copies;
    ggml.vulkan.context.ensureSyncStagingBuffer(ctx, staging_size);
    ggml.vulkan.context.syncBuffers(null, cpy_ctx);
    const slices = stagingSlices(dst_offset, stride_tensor, size, n_copies, false);
    defer std.heap.c_allocator.free(slices);
    ggml.vulkan.submit.copyBuffer(cpy_ctx.cmd(), ctx.sync_staging.?.buffer, buf.buffer, slices);

    const sptr: [*]u8 = @ptrCast(ctx.sync_staging.?.ptr.?);
    if (size == stride_data) {
        ggml.vulkan.submit.deferredMemcpy(sptr, src, staging_size, &cpy_ctx.in_memcpys);
    } else {
        for (0..n_copies) |i| ggml.vulkan.submit.deferredMemcpy(sptr + i * size, src + i * stride_data, size, &cpy_ctx.in_memcpys);
    }
    ggml.vulkan.context.synchronize(ctx);
}

/// Ports `ggml_backend_vk_set_tensor_async` (ggml-vulkan.cpp:13786 @b1b770d90).
fn setTensorAsync(backend: ggml.impl.ggml_c.ggml_backend_t, tensor: [*c]ggml.impl.ggml_c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    setTensor2dAsync(backend, tensor, data, offset, size, 1, size, size);
}

/// Ports `ggml_backend_vk_get_tensor_2d_async` (ggml-vulkan.cpp:13791 @b1b770d90).
fn getTensor2dAsync(backend: ggml.impl.ggml_c.ggml_backend_t, tensor_c: [*c]const ggml.impl.ggml_c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    const ctx = backendCtx(backend);
    const tensor = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, @constCast(tensor_c));
    assertBufferType(backend, tensor);
    if (size == 0) return;

    const compute_ctx = ggml.vulkan.context.getComputeCtx(ctx);
    const buf = ggml.vulkan.buffer.tensorBuffer(tensor);
    const src_offset = ggml.vulkan.buffer.tensorOffset(tensor) + tensor.view_offs + offset;
    const dst: [*]u8 = @ptrCast(data.?);

    if (ggml.vulkan.submit.bufferRead2dAsync(compute_ctx, buf, src_offset, dst, stride_tensor, stride_data, size, n_copies, false)) return;

    const staging_size = size * n_copies;
    ggml.vulkan.context.ensureSyncStagingBuffer(ctx, staging_size);
    ggml.vulkan.context.syncBuffers(null, compute_ctx);
    const slices = stagingSlices(src_offset, stride_tensor, size, n_copies, true);
    defer std.heap.c_allocator.free(slices);
    ggml.vulkan.submit.copyBuffer(compute_ctx.cmd(), buf.buffer, ctx.sync_staging.?.buffer, slices);

    const sptr: [*]const u8 = @ptrCast(ctx.sync_staging.?.ptr.?);
    if (size == stride_data) {
        ggml.vulkan.submit.deferredMemcpy(dst, sptr, staging_size, &compute_ctx.out_memcpys);
    } else {
        for (0..n_copies) |i| ggml.vulkan.submit.deferredMemcpy(dst + i * stride_data, sptr + i * size, size, &compute_ctx.out_memcpys);
    }
    ggml.vulkan.context.synchronize(ctx);
}

/// Ports `ggml_backend_vk_get_tensor_async` (ggml-vulkan.cpp:13842 @b1b770d90).
fn getTensorAsync(backend: ggml.impl.ggml_c.ggml_backend_t, tensor: [*c]const ggml.impl.ggml_c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    getTensor2dAsync(backend, tensor, data, offset, size, 1, size, size);
}

/// Ports `ggml_backend_vk_cpy_tensor_async` (ggml-vulkan.cpp:13847 @b1b770d90).
fn cpyTensorAsync(backend_src: ggml.impl.ggml_c.ggml_backend_t, backend_dst: ggml.impl.ggml_c.ggml_backend_t, src_c: [*c]const ggml.impl.ggml_c.ggml_tensor, dst_c: [*c]ggml.impl.ggml_c.ggml_tensor) callconv(.c) bool {
    const ctx = backendCtx(backend_dst);
    const src = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, @constCast(src_c));
    const dst = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst_c);

    // Skip zero-size tensors
    if (ggml.impl.ggml_c.ggml_nbytes(src) == 0) return true;
    if (dst.buffer.*.buft != getDefaultBufferType(backend_dst)) return false;

    const dst_buf = ggml.vulkan.buffer.tensorBuffer(dst);

    if (ggml.vulkan.buffer.bufferIsVk(src.buffer)) {
        const src_buf = ggml.vulkan.buffer.tensorBuffer(src);
        // Async copy only works within the same device
        if (src_buf.device != dst_buf.device) return false;
        const compute_ctx = ggml.vulkan.context.getComputeCtx(ctx);
        ggml.vulkan.submit.bufferCopyAsync(compute_ctx, dst_buf, ggml.vulkan.buffer.tensorOffset(dst) + dst.view_offs, src_buf, ggml.vulkan.buffer.tensorOffset(src) + src.view_offs, ggml.impl.ggml_c.ggml_nbytes(src));
        return true;
    }

    if (ggml.impl.ggml_c.ggml_backend_buffer_is_host(src.buffer)) {
        var pinned_offset: usize = 0;
        if (ggml.vulkan.buffer.hostGet(ctx.device, src.data, &pinned_offset) == null) return false;

        // If the backend is idle, use a CPU copy to avoid GPU synchronization overhead.
        const max_cpu_copy_size: usize = 128 * 1024;
        const src_backend_synchronous = backend_src.*.iface.synchronize == null;
        const transfer_idle = !ctx.device.async_use_transfer_queue or
            ctx.transfer_semaphore_last_submitted == ctx.transfer_semaphore.value;
        const backend_idle = ctx.compute_ctx == null and ctx.transfer_ctx == null and
            !ctx.submit_pending and !ctx.almost_ready_fence_pending and transfer_idle;
        const coherent: ggml.vulkan.common.vk.VkMemoryPropertyFlags = ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const dst_host_coherent = (dst_buf.memory_property_flags & coherent) == coherent;

        if ((backend_src == backend_dst or src_backend_synchronous) and backend_idle and dst_host_coherent and ggml.impl.ggml_c.ggml_nbytes(src) <= max_cpu_copy_size) {
            ggml.vulkan.submit.bufferWrite(dst_buf, ggml.vulkan.buffer.tensorOffset(dst) + dst.view_offs, @ptrCast(src.data.?), ggml.impl.ggml_c.ggml_nbytes(src));
            return true;
        }

        const cpy_ctx = if (ctx.device.async_use_transfer_queue) ggml.vulkan.context.getTransferCtx(ctx) else ggml.vulkan.context.getComputeCtx(ctx);
        return ggml.vulkan.submit.bufferWriteAsync(cpy_ctx, dst_buf, ggml.vulkan.buffer.tensorOffset(dst) + dst.view_offs, @ptrCast(src.data.?), ggml.impl.ggml_c.ggml_nbytes(src), false);
    }
    return false;
}

/// Ports `ggml_backend_vk_synchronize` (ggml-vulkan.cpp:13989 @b1b770d90).
fn synchronize(backend: ggml.impl.ggml_c.ggml_backend_t) callconv(.c) void {
    const ctx = backendCtx(backend);
    ggml.vulkan.context.synchronize(ctx);
    ggml.vulkan.context.graphCleanup(ctx);
}

/// Ports `vk_event` (ggml-vulkan-types.h:1111 @b1b770d90): `vk::Event`s for
/// `event_wait`, a timeline semaphore for `event_synchronize`.
pub const Event = struct {
    events_free: std.ArrayList(ggml.vulkan.common.vk.VkEvent) = .empty, // available for reuse
    events_submitted: std.ArrayList(ggml.vulkan.common.vk.VkEvent) = .empty, // reusable after the next synchronize
    event: ggml.vulkan.common.vk.VkEvent = null,
    has_event: bool = false,
    tl_semaphore: ggml.vulkan.submit.Semaphore = .{},
    cmd_buffer: ?*ggml.vulkan.device.CommandBuffer = null,
    cmd_buffer_use_counter: u64 = 0,
};

fn eventOf(event: ggml.impl.ggml_c.ggml_backend_event_t) *Event {
    return @ptrCast(@alignCast(event.*.context));
}

/// Ports `ggml_backend_vk_event_record` (ggml-vulkan.cpp:15642 @b1b770d90).
fn eventRecord(backend: ggml.impl.ggml_c.ggml_backend_t, event: ggml.impl.ggml_c.ggml_backend_event_t) callconv(.c) void {
    const ctx = backendCtx(backend);
    const vkev = eventOf(event);

    _ = ggml.vulkan.context.submitTransferCtx(ctx);

    const compute_ctx = ggml.vulkan.context.getComputeCtx(ctx);
    const cmd_buf = compute_ctx.s().buffer.?; // retrieve pointer before it gets reset

    if (vkev.has_event) vkev.events_submitted.append(std.heap.c_allocator, vkev.event) catch oom();

    // Grab the next event and record it, create one if necessary
    if (vkev.events_free.pop()) |e| {
        vkev.event = e;
    } else {
        var ci = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkEventCreateInfo, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_EVENT_CREATE_INFO);
        ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkCreateEvent.?(ctx.device.device, &ci, null, &vkev.event), "vk::Device::createEvent");
    }
    vkev.has_event = true;

    ggml.vulkan.submit.setEvent(compute_ctx, vkev.event);

    vkev.tl_semaphore.value += 1;
    compute_ctx.s().signal_semaphores.append(std.heap.c_allocator, vkev.tl_semaphore) catch oom();
    ggml.vulkan.submit.ctxEnd(compute_ctx);

    ggml.vulkan.submit.submit(compute_ctx, null);
    ctx.submit_pending = true;
    vkev.cmd_buffer = cmd_buf;
    vkev.cmd_buffer_use_counter = cmd_buf.use_counter;
    ctx.compute_ctx = null;
}

/// Ports `ggml_backend_vk_event_wait` (ggml-vulkan.cpp:15680 @b1b770d90).
fn eventWait(backend: ggml.impl.ggml_c.ggml_backend_t, event: ggml.impl.ggml_c.ggml_backend_event_t) callconv(.c) void {
    const ctx = backendCtx(backend);
    const vkev = eventOf(event);
    const compute_ctx = ggml.vulkan.context.getComputeCtx(ctx);
    if (vkev.has_event) {
        // Wait for latest event
        ggml.vulkan.submit.waitEvents(compute_ctx, &.{vkev.event});
        if (ctx.device.async_use_transfer_queue) {
            const transfer_ctx = ggml.vulkan.context.getTransferCtx(ctx);
            transfer_ctx.s().wait_semaphores.append(std.heap.c_allocator, vkev.tl_semaphore) catch oom();
        }
    }
}

/// Ports `ggml_backend_vk_device_event_new` (ggml-vulkan.cpp:16573 @b1b770d90).
pub fn deviceEventNew(dev: ggml.impl.ggml_c.ggml_backend_dev_t, dev_idx: usize) ggml.impl.ggml_c.ggml_backend_event_t {
    const device = ggml.vulkan.device.getDevice(dev_idx);
    const vkev = std.heap.c_allocator.create(Event) catch oom();
    // No events initially, they get created on demand
    vkev.* = .{};
    var tci = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkSemaphoreTypeCreateInfo, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO);
    tci.semaphoreType = ggml.vulkan.common.vk.VK_SEMAPHORE_TYPE_TIMELINE;
    var ci = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkSemaphoreCreateInfo, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    ci.pNext = &tci;
    ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkCreateSemaphore.?(device.device, &ci, null, &vkev.tl_semaphore.s), "vk::Device::createSemaphore");
    const ev = std.heap.c_allocator.create(ggml.impl.ggml_c.struct_ggml_backend_event) catch oom();
    ev.* = .{ .device = dev, .context = vkev };
    return ev;
}

/// Ports `ggml_backend_vk_device_event_free` (ggml-vulkan.cpp:16596 @b1b770d90).
pub fn deviceEventFree(dev_idx: usize, event: ggml.impl.ggml_c.ggml_backend_event_t) void {
    const device = ggml.vulkan.device.getDevice(dev_idx);
    const vkev = eventOf(event);
    ggml.vulkan.common.dispatch.vkDestroySemaphore.?(device.device, vkev.tl_semaphore.s, null);
    for (vkev.events_free.items) |e| ggml.vulkan.common.dispatch.vkDestroyEvent.?(device.device, e, null);
    for (vkev.events_submitted.items) |e| ggml.vulkan.common.dispatch.vkDestroyEvent.?(device.device, e, null);
    if (vkev.has_event) ggml.vulkan.common.dispatch.vkDestroyEvent.?(device.device, vkev.event, null);
    vkev.events_free.deinit(std.heap.c_allocator);
    vkev.events_submitted.deinit(std.heap.c_allocator);
    std.heap.c_allocator.destroy(vkev);
    std.heap.c_allocator.destroy(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_event, event));
}

/// Ports `ggml_backend_vk_device_event_synchronize` (ggml-vulkan.cpp:16616 @b1b770d90).
pub fn deviceEventSynchronize(dev_idx: usize, event: ggml.impl.ggml_c.ggml_backend_event_t) void {
    const device = ggml.vulkan.device.getDevice(dev_idx);
    const vkev = eventOf(event);
    // Only do something if the event has actually been used
    if (!vkev.has_event) return;
    var swi = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkSemaphoreWaitInfo, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO);
    swi.semaphoreCount = 1;
    swi.pSemaphores = &vkev.tl_semaphore.s;
    swi.pValues = &vkev.tl_semaphore.value;
    ggml.vulkan.submit.vkCheck(ggml.vulkan.common.dispatch.vkWaitSemaphores.?(device.device, &swi, std.math.maxInt(u64)), "event_synchronize", device);

    // Reset and move submitted events
    for (vkev.events_submitted.items) |e| ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkResetEvent.?(device.device, e), "vk::Device::resetEvent");
    vkev.events_free.appendSlice(std.heap.c_allocator, vkev.events_submitted.items) catch oom();
    vkev.events_submitted.clearRetainingCapacity();

    // Finished using current command buffer so we flag for reuse
    if (vkev.cmd_buffer) |cb| {
        // Only flag for reuse if it hasn't been reused already
        if (vkev.cmd_buffer_use_counter == cb.use_counter) {
            cb.in_use = false;
            ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkResetCommandBuffer.?(cb.buf, 0), "vk::CommandBuffer::reset");
        }
        vkev.cmd_buffer = null;
    }
}

/// Ports `ggml_vk_buffer_from_host_ptr` (ggml-vulkan-buffers.cpp:1048 @b1b770d90):
/// imports host memory as a device buffer, if the device can and the
/// pointer and size are aligned as it needs.
pub fn bufferFromHostPtr(device: *ggml.vulkan.device.Device, ptr: ?*anyopaque, size: usize) ?*ggml.vulkan.buffer.Buffer {
    if (!device.external_memory_host) return null;
    const a = device.min_imported_host_pointer_alignment - 1;
    if (@intFromPtr(ptr) & a != 0) return null;
    if (size & a != 0) return null;
    const property_flags: ggml.vulkan.common.vk.VkMemoryPropertyFlags = ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    return ggml.vulkan.buffer.createBuffer(device, size, &.{property_flags}, ptr) catch {
        ggml.impl.logWarn("ggml_vulkan: Failed ggml_vk_create_buffer (%s: %s)\n", .{ ggml.vulkan.common.last_error.what, ggml.vulkan.common.resultString(ggml.vulkan.common.last_error.result).ptr });
        return null;
    };
}

/// Ports `ggml_backend_vk_device_buffer_from_host_ptr` (ggml-vulkan.cpp:16648 @b1b770d90).
pub fn deviceBufferFromHostPtr(dev: ggml.impl.ggml_c.ggml_backend_dev_t, dev_idx: usize, ptr: ?*anyopaque, size: usize) ggml.impl.ggml_c.ggml_backend_buffer_t {
    const device = ggml.vulkan.device.getDevice(dev_idx);
    const buf = bufferFromHostPtr(device, ptr, size) orelse return null;
    return ggml.vulkan.buffer.wrapBuffer(ggml.impl.ggml_c.ggml_backend_dev_buffer_type(dev), device, buf, device.name, size);
}

/// Ports `ggml_backend_vk_interface` (ggml-vulkan.cpp:15698 @b1b770d90).
const backend_iface: ggml.impl.ggml_c.struct_ggml_backend_i = .{
    .get_name = &name,
    .free = &free,
    .set_tensor_async = &setTensorAsync,
    .get_tensor_async = &getTensorAsync,
    .set_tensor_2d_async = &setTensor2dAsync,
    .get_tensor_2d_async = &getTensor2dAsync,
    .cpy_tensor_async = &cpyTensorAsync,
    .synchronize = &synchronize,
    .graph_plan_create = null,
    .graph_plan_free = null,
    .graph_plan_update = null,
    .graph_plan_compute = null,
    .graph_compute = &ggml.vulkan.graph.graphCompute,
    .event_record = &eventRecord,
    .event_wait = &eventWait,
    .graph_optimize = &ggml.vulkan.graph.graphOptimize,
};

/// Ports `ggml_backend_vk_init` (ggml-vulkan.cpp:15722 @b1b770d90).
pub fn backendInit(dev_num: usize, guid: ggml.impl.ggml_c.ggml_guid_t) ggml.impl.ggml_c.ggml_backend_t {
    const ctx = std.heap.c_allocator.create(ggml.vulkan.context.BackendCtx) catch oom();
    ctx.* = .{};
    ggml.vulkan.context.init(ctx, dev_num);

    const backend = std.heap.c_allocator.create(ggml.impl.ggml_c.struct_ggml_backend) catch oom();
    backend.* = .{
        .guid = guid,
        .iface = backend_iface,
        .device = ggml.impl.ggml_c.ggml_backend_reg_dev_get(ggml.vulkan.backend.reg(), dev_num),
        .context = ctx,
    };
    if (!ctx.device.support_async) backend.iface.get_tensor_async = null;
    return backend;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
