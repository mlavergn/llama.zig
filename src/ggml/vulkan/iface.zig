//! The `ggml_backend` a Vulkan device hands out: its interface table, the
//! asynchronous transfers, events, and the buffer wrapped around host
//! memory.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.3.0
//! (`c1d0e7a00`).

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const common = m.common;
const vk = common.vk;
const d = &common.dispatch;
const allocator = common.allocator;
const BackendCtx = m.context.BackendCtx;
const Context = m.submit.Context;

fn oom() noreturn {
    impl.abort("ggml_vulkan: out of memory");
}

fn backendCtx(backend: c.ggml_backend_t) *BackendCtx {
    return @ptrCast(@alignCast(backend.*.context));
}

/// Ports `ggml_backend_vk_name` (ggml-vulkan.cpp:16243 @c1d0e7a00).
fn name(backend: c.ggml_backend_t) callconv(.c) [*c]const u8 {
    return backendCtx(backend).name.ptr;
}

/// Ports `ggml_backend_vk_free` (ggml-vulkan.cpp:16249 @c1d0e7a00).
fn free(backend: c.ggml_backend_t) callconv(.c) void {
    const ctx = backendCtx(backend);
    m.context.cleanup(ctx);
    allocator.destroy(ctx);
    allocator.destroy(impl.one(c.struct_ggml_backend, backend));
}

/// Ports `ggml_backend_vk_get_default_buffer_type` (ggml-vulkan.cpp:16259 @c1d0e7a00).
fn getDefaultBufferType(backend: c.ggml_backend_t) callconv(.c) c.ggml_backend_buffer_type_t {
    return &backendCtx(backend).device.buffer_type;
}

/// The staging copies `set_tensor_2d_async` and `get_tensor_2d_async` make
/// when the transfer could not go through a mapping or pinned memory.
fn stagingSlices(off: usize, stride: usize, size: usize, n_copies: usize, to_staging: bool) []vk.VkBufferCopy {
    if (size == stride) {
        const out = allocator.alloc(vk.VkBufferCopy, 1) catch oom();
        out[0] = if (to_staging) .{ .srcOffset = off, .dstOffset = 0, .size = size * n_copies } else .{ .srcOffset = 0, .dstOffset = off, .size = size * n_copies };
        return out;
    }
    const out = allocator.alloc(vk.VkBufferCopy, n_copies) catch oom();
    for (out, 0..) |*s, i| {
        s.* = if (to_staging) .{ .srcOffset = off + i * stride, .dstOffset = i * size, .size = size } else .{ .srcOffset = i * size, .dstOffset = off + i * stride, .size = size };
    }
    return out;
}

fn assertBufferType(backend: c.ggml_backend_t, tensor: *const c.ggml_tensor) void {
    impl.assert(tensor.buffer.*.buft == getDefaultBufferType(backend) or tensor.buffer.*.buft == m.buffer.hostBufferType(), "(tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && \"unsupported buffer type\"");
}

/// Ports `ggml_backend_vk_set_tensor_2d_async` (ggml-vulkan.cpp:16265 @c1d0e7a00).
fn setTensor2dAsync(backend: c.ggml_backend_t, tensor_c: [*c]c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    const ctx = backendCtx(backend);
    const tensor = impl.one(c.ggml_tensor, tensor_c);
    assertBufferType(backend, tensor);
    if (size == 0) return;

    const cpy_ctx = if (ctx.device.async_use_transfer_queue) m.context.getTransferCtx(ctx) else m.context.getComputeCtx(ctx);
    const buf = m.buffer.tensorBuffer(tensor);
    const dst_offset = m.buffer.tensorOffset(tensor) + tensor.view_offs + offset;
    const src: [*]const u8 = @ptrCast(data.?);

    if (m.submit.bufferWrite2dAsync(cpy_ctx, buf, dst_offset, src, stride_data, stride_tensor, size, n_copies, false)) return;

    const staging_size = size * n_copies;
    m.context.ensureSyncStagingBuffer(ctx, staging_size);
    m.context.syncBuffers(null, cpy_ctx);
    const slices = stagingSlices(dst_offset, stride_tensor, size, n_copies, false);
    defer allocator.free(slices);
    m.submit.copyBuffer(cpy_ctx.cmd(), ctx.sync_staging.?.buffer, buf.buffer, slices);

    const sptr: [*]u8 = @ptrCast(ctx.sync_staging.?.ptr.?);
    if (size == stride_data) {
        m.submit.deferredMemcpy(sptr, src, staging_size, &cpy_ctx.in_memcpys);
    } else {
        for (0..n_copies) |i| m.submit.deferredMemcpy(sptr + i * size, src + i * stride_data, size, &cpy_ctx.in_memcpys);
    }
    m.context.synchronize(ctx);
}

/// Ports `ggml_backend_vk_set_tensor_async` (ggml-vulkan.cpp:16323 @c1d0e7a00).
fn setTensorAsync(backend: c.ggml_backend_t, tensor: [*c]c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    setTensor2dAsync(backend, tensor, data, offset, size, 1, size, size);
}

/// Ports `ggml_backend_vk_get_tensor_2d_async` (ggml-vulkan.cpp:16328 @c1d0e7a00).
fn getTensor2dAsync(backend: c.ggml_backend_t, tensor_c: [*c]const c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    const ctx = backendCtx(backend);
    const tensor = impl.one(c.ggml_tensor, @constCast(tensor_c));
    assertBufferType(backend, tensor);
    if (size == 0) return;

    const compute_ctx = m.context.getComputeCtx(ctx);
    const buf = m.buffer.tensorBuffer(tensor);
    const src_offset = m.buffer.tensorOffset(tensor) + tensor.view_offs + offset;
    const dst: [*]u8 = @ptrCast(data.?);

    if (m.submit.bufferRead2dAsync(compute_ctx, buf, src_offset, dst, stride_tensor, stride_data, size, n_copies, false)) return;

    const staging_size = size * n_copies;
    m.context.ensureSyncStagingBuffer(ctx, staging_size);
    m.context.syncBuffers(null, compute_ctx);
    const slices = stagingSlices(src_offset, stride_tensor, size, n_copies, true);
    defer allocator.free(slices);
    m.submit.copyBuffer(compute_ctx.cmd(), buf.buffer, ctx.sync_staging.?.buffer, slices);

    const sptr: [*]const u8 = @ptrCast(ctx.sync_staging.?.ptr.?);
    if (size == stride_data) {
        m.submit.deferredMemcpy(dst, sptr, staging_size, &compute_ctx.out_memcpys);
    } else {
        for (0..n_copies) |i| m.submit.deferredMemcpy(dst + i * stride_data, sptr + i * size, size, &compute_ctx.out_memcpys);
    }
    m.context.synchronize(ctx);
}

/// Ports `ggml_backend_vk_get_tensor_async` (ggml-vulkan.cpp:16379 @c1d0e7a00).
fn getTensorAsync(backend: c.ggml_backend_t, tensor: [*c]const c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    getTensor2dAsync(backend, tensor, data, offset, size, 1, size, size);
}

/// Ports `ggml_backend_vk_cpy_tensor_async` (ggml-vulkan.cpp:16384 @c1d0e7a00).
fn cpyTensorAsync(backend_src: c.ggml_backend_t, backend_dst: c.ggml_backend_t, src_c: [*c]const c.ggml_tensor, dst_c: [*c]c.ggml_tensor) callconv(.c) bool {
    _ = backend_src;
    const ctx = backendCtx(backend_dst);
    const src = impl.one(c.ggml_tensor, @constCast(src_c));
    const dst = impl.one(c.ggml_tensor, dst_c);

    // Skip zero-size tensors
    if (c.ggml_nbytes(src) == 0) return true;
    if (dst.buffer.*.buft != getDefaultBufferType(backend_dst)) return false;

    const dst_buf = m.buffer.tensorBuffer(dst);

    if (m.buffer.bufferIsVk(src.buffer)) {
        const src_buf = m.buffer.tensorBuffer(src);
        // Async copy only works within the same device
        if (src_buf.device != dst_buf.device) return false;
        const compute_ctx = m.context.getComputeCtx(ctx);
        m.submit.bufferCopyAsync(compute_ctx, dst_buf, m.buffer.tensorOffset(dst) + dst.view_offs, src_buf, m.buffer.tensorOffset(src) + src.view_offs, c.ggml_nbytes(src));
        return true;
    }

    if (c.ggml_backend_buffer_is_host(src.buffer)) {
        var pinned_offset: usize = 0;
        if (m.buffer.hostGet(ctx.device, src.data, &pinned_offset) == null) return false;
        const cpy_ctx = if (ctx.device.async_use_transfer_queue) m.context.getTransferCtx(ctx) else m.context.getComputeCtx(ctx);
        return m.submit.bufferWriteAsync(cpy_ctx, dst_buf, m.buffer.tensorOffset(dst) + dst.view_offs, @ptrCast(src.data.?), c.ggml_nbytes(src), false);
    }
    return false;
}

/// Ports `ggml_backend_vk_synchronize` (ggml-vulkan.cpp:16511 @c1d0e7a00).
fn synchronize(backend: c.ggml_backend_t) callconv(.c) void {
    const ctx = backendCtx(backend);
    m.context.synchronize(ctx);
    m.context.graphCleanup(ctx);
}

/// Ports `vk_event` (ggml-vulkan.cpp:1251 @c1d0e7a00): `vk::Event`s for
/// `event_wait`, a timeline semaphore for `event_synchronize`.
pub const Event = struct {
    events_free: std.ArrayList(vk.VkEvent) = .empty, // available for reuse
    events_submitted: std.ArrayList(vk.VkEvent) = .empty, // reusable after the next synchronize
    event: vk.VkEvent = null,
    has_event: bool = false,
    tl_semaphore: m.submit.Semaphore = .{},
    cmd_buffer: ?*m.device.CommandBuffer = null,
    cmd_buffer_use_counter: u64 = 0,
};

fn eventOf(event: c.ggml_backend_event_t) *Event {
    return @ptrCast(@alignCast(event.*.context));
}

/// Ports `ggml_backend_vk_event_record` (ggml-vulkan.cpp:17789 @c1d0e7a00).
fn eventRecord(backend: c.ggml_backend_t, event: c.ggml_backend_event_t) callconv(.c) void {
    const ctx = backendCtx(backend);
    const vkev = eventOf(event);

    _ = m.context.submitTransferCtx(ctx);

    const compute_ctx = m.context.getComputeCtx(ctx);
    const cmd_buf = compute_ctx.s().buffer.?; // retrieve pointer before it gets reset

    if (vkev.has_event) vkev.events_submitted.append(allocator, vkev.event) catch oom();

    // Grab the next event and record it, create one if necessary
    if (vkev.events_free.pop()) |e| {
        vkev.event = e;
    } else {
        var ci = common.structure(vk.VkEventCreateInfo, vk.VK_STRUCTURE_TYPE_EVENT_CREATE_INFO);
        common.checkOrAbort(d.vkCreateEvent.?(ctx.device.device, &ci, null, &vkev.event), "vk::Device::createEvent");
    }
    vkev.has_event = true;

    m.submit.setEvent(compute_ctx, vkev.event);

    vkev.tl_semaphore.value += 1;
    compute_ctx.s().signal_semaphores.append(allocator, vkev.tl_semaphore) catch oom();
    m.submit.ctxEnd(compute_ctx);

    m.submit.submit(compute_ctx, null);
    ctx.submit_pending = true;
    vkev.cmd_buffer = cmd_buf;
    vkev.cmd_buffer_use_counter = cmd_buf.use_counter;
    ctx.compute_ctx = null;
}

/// Ports `ggml_backend_vk_event_wait` (ggml-vulkan.cpp:17827 @c1d0e7a00).
fn eventWait(backend: c.ggml_backend_t, event: c.ggml_backend_event_t) callconv(.c) void {
    const ctx = backendCtx(backend);
    const vkev = eventOf(event);
    const compute_ctx = m.context.getComputeCtx(ctx);
    if (vkev.has_event) {
        // Wait for latest event
        m.submit.waitEvents(compute_ctx, &.{vkev.event});
        if (ctx.device.async_use_transfer_queue) {
            const transfer_ctx = m.context.getTransferCtx(ctx);
            transfer_ctx.s().wait_semaphores.append(allocator, vkev.tl_semaphore) catch oom();
        }
    }
}

/// Ports `ggml_backend_vk_device_event_new` (ggml-vulkan.cpp:18665 @c1d0e7a00).
pub fn deviceEventNew(dev: c.ggml_backend_dev_t, dev_idx: usize) c.ggml_backend_event_t {
    const device = m.device.getDevice(dev_idx);
    const vkev = allocator.create(Event) catch oom();
    // No events initially, they get created on demand
    vkev.* = .{};
    var tci = common.structure(vk.VkSemaphoreTypeCreateInfo, vk.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO);
    tci.semaphoreType = vk.VK_SEMAPHORE_TYPE_TIMELINE;
    var ci = common.structure(vk.VkSemaphoreCreateInfo, vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    ci.pNext = &tci;
    common.checkOrAbort(d.vkCreateSemaphore.?(device.device, &ci, null, &vkev.tl_semaphore.s), "vk::Device::createSemaphore");
    const ev = allocator.create(c.struct_ggml_backend_event) catch oom();
    ev.* = .{ .device = dev, .context = vkev };
    return ev;
}

/// Ports `ggml_backend_vk_device_event_free` (ggml-vulkan.cpp:18688 @c1d0e7a00).
pub fn deviceEventFree(dev_idx: usize, event: c.ggml_backend_event_t) void {
    const device = m.device.getDevice(dev_idx);
    const vkev = eventOf(event);
    d.vkDestroySemaphore.?(device.device, vkev.tl_semaphore.s, null);
    for (vkev.events_free.items) |e| d.vkDestroyEvent.?(device.device, e, null);
    for (vkev.events_submitted.items) |e| d.vkDestroyEvent.?(device.device, e, null);
    if (vkev.has_event) d.vkDestroyEvent.?(device.device, vkev.event, null);
    vkev.events_free.deinit(allocator);
    vkev.events_submitted.deinit(allocator);
    allocator.destroy(vkev);
    allocator.destroy(impl.one(c.struct_ggml_backend_event, event));
}

/// Ports `ggml_backend_vk_device_event_synchronize` (ggml-vulkan.cpp:18708 @c1d0e7a00).
pub fn deviceEventSynchronize(dev_idx: usize, event: c.ggml_backend_event_t) void {
    const device = m.device.getDevice(dev_idx);
    const vkev = eventOf(event);
    // Only do something if the event has actually been used
    if (!vkev.has_event) return;
    var swi = common.structure(vk.VkSemaphoreWaitInfo, vk.VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO);
    swi.semaphoreCount = 1;
    swi.pSemaphores = &vkev.tl_semaphore.s;
    swi.pValues = &vkev.tl_semaphore.value;
    m.submit.vkCheck(d.vkWaitSemaphores.?(device.device, &swi, std.math.maxInt(u64)), "event_synchronize", device);

    // Reset and move submitted events
    for (vkev.events_submitted.items) |e| common.checkOrAbort(d.vkResetEvent.?(device.device, e), "vk::Device::resetEvent");
    vkev.events_free.appendSlice(allocator, vkev.events_submitted.items) catch oom();
    vkev.events_submitted.clearRetainingCapacity();

    // Finished using current command buffer so we flag for reuse
    if (vkev.cmd_buffer) |cb| {
        // Only flag for reuse if it hasn't been reused already
        if (vkev.cmd_buffer_use_counter == cb.use_counter) {
            cb.in_use = false;
            common.checkOrAbort(d.vkResetCommandBuffer.?(cb.buf, 0), "vk::CommandBuffer::reset");
        }
        vkev.cmd_buffer = null;
    }
}

/// Ports `ggml_vk_buffer_from_host_ptr` (ggml-vulkan.cpp:18740 @c1d0e7a00):
/// imports host memory as a device buffer, if the device can and the
/// pointer and size are aligned as it needs.
fn bufferFromHostPtr(device: *m.device.Device, ptr: ?*anyopaque, size: usize) ?*m.buffer.Buffer {
    if (!device.external_memory_host) return null;
    const a = device.min_imported_host_pointer_alignment - 1;
    if (@intFromPtr(ptr) & a != 0) return null;
    if (size & a != 0) return null;
    const property_flags: vk.VkMemoryPropertyFlags = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    return m.buffer.createBuffer(device, size, &.{property_flags}, ptr) catch {
        impl.logWarn("ggml_vulkan: Failed ggml_vk_create_buffer (%s: %s)\n", .{ common.last_error.what, common.resultString(common.last_error.result).ptr });
        return null;
    };
}

/// Ports `ggml_backend_vk_device_buffer_from_host_ptr` (ggml-vulkan.cpp:18765 @c1d0e7a00).
pub fn deviceBufferFromHostPtr(dev: c.ggml_backend_dev_t, dev_idx: usize, ptr: ?*anyopaque, size: usize) c.ggml_backend_buffer_t {
    const device = m.device.getDevice(dev_idx);
    const buf = bufferFromHostPtr(device, ptr, size) orelse return null;
    return m.buffer.wrapBuffer(c.ggml_backend_dev_buffer_type(dev), device, buf, device.name, size);
}

/// Ports `ggml_backend_vk_interface` (ggml-vulkan.cpp:17846 @c1d0e7a00).
const backend_iface: c.struct_ggml_backend_i = .{
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
    .graph_compute = &m.graph.graphCompute,
    .event_record = &eventRecord,
    .event_wait = &eventWait,
    .graph_optimize = &m.graph.graphOptimize,
};

/// Ports `ggml_backend_vk_init` (ggml-vulkan.cpp:17870 @c1d0e7a00).
pub fn backendInit(dev_num: usize, guid: c.ggml_guid_t) c.ggml_backend_t {
    const ctx = allocator.create(BackendCtx) catch oom();
    ctx.* = .{};
    m.context.init(ctx, dev_num);

    const backend = allocator.create(c.struct_ggml_backend) catch oom();
    backend.* = .{
        .guid = guid,
        .iface = backend_iface,
        .device = c.ggml_backend_reg_dev_get(m.backend.reg(), dev_num),
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
