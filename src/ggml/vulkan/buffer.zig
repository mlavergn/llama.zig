//! Device memory: `vk_buffer`, how its memory type is chosen, pinned host
//! memory, and the two buffer types the backend exposes to ggml.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.3.0
//! (`c1d0e7a00`). A `vk_buffer` is a `std::shared_ptr` whose destructor
//! frees the memory; here it is reference counted by hand, and only the
//! holders the C++'s lifetimes depend on take a reference.

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const common = m.common;
const vk = common.vk;
const d = &common.dispatch;
const allocator = common.allocator;
const Device = m.device.Device;

fn oom() noreturn {
    impl.abort("ggml_vulkan: out of memory");
}

/// Ports `vk_buffer_struct` (ggml-vulkan.cpp:1211 @c1d0e7a00).
pub const Buffer = struct {
    buffer: vk.VkBuffer = null,
    device_memory: vk.VkDeviceMemory = null,
    memory_property_flags: vk.VkMemoryPropertyFlags = 0,
    ptr: ?*anyopaque = null,
    size: usize = 0,
    bda_addr: vk.VkDeviceAddress = 0,
    device: *Device,
    refs: u32 = 1,

    /// A new owner: copying the `shared_ptr`.
    pub fn ref(self: *Buffer) *Buffer {
        self.refs += 1;
        return self;
    }

    /// An owner gone: the `shared_ptr`'s destructor; the last one frees
    /// the memory, as `~vk_buffer_struct` does.
    pub fn unref(self: *Buffer) void {
        self.refs -= 1;
        if (self.refs != 0) return;
        if (self.size != 0) {
            d.vkFreeMemory.?(self.device.device, self.device_memory, null);
            d.vkDestroyBuffer.?(self.device.device, self.buffer, null);
        }
        allocator.destroy(self);
    }
};

/// Ports `vk_subbuffer` (ggml-vulkan.cpp:1232 @c1d0e7a00). Transient: it
/// does not keep its buffer alive, and nothing in the C++ relies on it to.
pub const Subbuffer = struct {
    buffer: ?*Buffer = null,
    offset: u64 = 0,
    size: u64 = 0,

    /// `operator vk::DescriptorBufferInfo`.
    pub fn info(self: Subbuffer) vk.VkDescriptorBufferInfo {
        return .{ .buffer = self.buffer.?.buffer, .offset = self.offset, .range = self.size };
    }
};

/// One pinned host allocation: `std::tuple<void*, size_t, vk_buffer>`.
pub const PinnedMemory = struct { ptr: [*]u8, size: usize, buf: *Buffer };

// -----------------------------------------------------------------------------
// The memory logger

/// Ports `format_size` (ggml-vulkan.cpp:2079 @c1d0e7a00).
fn formatSize(size: usize, buf: []u8) [:0]const u8 {
    const kib: usize = 1024;
    const mib = kib * 1024;
    const gib = mib * 1024;
    const f: f64 = @floatFromInt(size);
    return (if (size >= gib)
        std.fmt.bufPrintSentinel(buf, "{d:.2} GiB", .{f / @as(f64, @floatFromInt(gib))}, 0)
    else if (size >= mib)
        std.fmt.bufPrintSentinel(buf, "{d:.2} MiB", .{f / @as(f64, @floatFromInt(mib))}, 0)
    else if (size >= kib)
        std.fmt.bufPrintSentinel(buf, "{d:.2} KiB", .{f / @as(f64, @floatFromInt(kib))}, 0)
    else
        std.fmt.bufPrintSentinel(buf, "{d} B", .{size}, 0)) catch unreachable;
}

/// Ports `vk_memory_logger` (ggml-vulkan.cpp:2100 @c1d0e7a00), one per
/// process here where the C++ keeps one per device; its totals are per
/// device in the C++, so they are keyed by device.
const memory_logger = struct {
    var mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
    var allocations: std.AutoHashMapUnmanaged(vk.VkBuffer, usize) = .empty;
    var totals: std.AutoHashMapUnmanaged(*Device, [2]usize) = .empty;

    /// Ports `vk_memory_logger`'s `log_allocation` (ggml-vulkan.cpp:2519 @c1d0e7a00).
    fn logAllocation(buf: *Buffer, size: usize) void {
        if (!m.instance.loggers.memory_logger_enabled) return;
        _ = std.c.pthread_mutex_lock(&mutex);
        defer _ = std.c.pthread_mutex_unlock(&mutex);
        const device = buf.memory_property_flags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0;
        allocations.put(allocator, buf.buffer, size) catch oom();
        const t = totals.getOrPutValue(allocator, buf.device, .{ 0, 0 }) catch oom();
        if (device) t.value_ptr[0] += size else t.value_ptr[1] += size;
        var a: [32]u8 = undefined;
        var b: [32]u8 = undefined;
        var e: [32]u8 = undefined;
        impl.fprintfStderr("ggml_vulkan memory: %s: +%s %s at %p. Total device: %s, total host: %s\n", .{ buf.device.name.ptr, formatSize(size, &a).ptr, @as([*:0]const u8, if (device) "device" else "host"), @as(?*anyopaque, buf.buffer), formatSize(t.value_ptr[0], &b).ptr, formatSize(t.value_ptr[1], &e).ptr });
    }

    /// Ports `vk_memory_logger`'s `log_deallocation` (ggml-vulkan.cpp:2533 @c1d0e7a00).
    fn logDeallocation(buf: *Buffer) void {
        if (buf.size == 0 or !m.instance.loggers.memory_logger_enabled) return;
        _ = std.c.pthread_mutex_lock(&mutex);
        defer _ = std.c.pthread_mutex_unlock(&mutex);
        const device = buf.memory_property_flags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0;
        const kind: [*:0]const u8 = if (device) "device" else "host";
        if (allocations.fetchRemove(buf.buffer)) |kv| {
            const t = totals.getOrPutValue(allocator, buf.device, .{ 0, 0 }) catch oom();
            if (device) t.value_ptr[0] -= kv.value else t.value_ptr[1] -= kv.value;
            var a: [32]u8 = undefined;
            var b: [32]u8 = undefined;
            var e: [32]u8 = undefined;
            impl.fprintfStderr("ggml_vulkan memory: %s: -%s %s at %p. Total device: %s, total host: %s\n", .{ buf.device.name.ptr, formatSize(kv.value, &a).ptr, kind, @as(?*anyopaque, buf.buffer), formatSize(t.value_ptr[0], &b).ptr, formatSize(t.value_ptr[1], &e).ptr });
        } else {
            impl.fprintfStderr("ggml_vulkan memory: ERROR %s: Attempted to deallocate unknown %s memory at %p\n", .{ buf.device.name.ptr, kind, @as(?*anyopaque, buf.buffer) });
        }
    }
};

/// `VK_LOG_MEMORY` (ggml-vulkan.cpp:2077 @c1d0e7a00), for its plain-text uses.
pub fn logMemory(comptime fmt: []const u8, args: anytype) void {
    if (m.instance.loggers.memory_logger_enabled) impl.fprintfStderr("ggml_vulkan memory: " ++ fmt ++ "\n", args);
}

// -----------------------------------------------------------------------------
// Creation

/// Ports `ggml_vk_find_memory_properties` (ggml-vulkan.cpp:3397 @c1d0e7a00).
fn findMemoryProperties(mem_props: *const vk.VkPhysicalDeviceMemoryProperties, mem_req: *const vk.VkMemoryRequirements, flags: vk.VkMemoryPropertyFlags, out: *[vk.VK_MAX_MEMORY_TYPES]u32) usize {
    var n: usize = 0;
    for (0..mem_props.memoryTypeCount) |i| {
        const memory_type = mem_props.memoryTypes[i];
        if (mem_req.memoryTypeBits & (@as(u64, 1) << @intCast(i)) != 0 and
            flags & memory_type.propertyFlags == flags and
            mem_props.memoryHeaps[memory_type.heapIndex].size >= mem_req.size)
        {
            out[n] = @intCast(i);
            n += 1;
        }
    }
    return n;
}

/// `throw vk::OutOfDeviceMemoryError(message)`: records the message the
/// exception's `what()` carries.
fn throwOutOfDeviceMemory(comptime message: [*:0]const u8) common.VkError {
    common.last_error = .{ .result = vk.VK_ERROR_OUT_OF_DEVICE_MEMORY, .what = message };
    return error.VulkanError;
}

/// `vk::PhysicalDevice::getMemoryProperties`.
fn getMemoryProperties(device: *Device) vk.VkPhysicalDeviceMemoryProperties {
    var props: vk.VkPhysicalDeviceMemoryProperties = undefined;
    d.vkGetPhysicalDeviceMemoryProperties.?(device.physical_device, &props);
    return props;
}

/// Ports `ggml_vk_create_buffer` (ggml-vulkan.cpp:3411 @c1d0e7a00): a buffer
/// backed by the first memory type, trying each set of flags in order,
/// that allocates; or, given `import_ptr`, imported host memory.
///
/// Return: the buffer; null where the C++ returns an empty pointer (an
/// import that fails); `error.VulkanError` where it throws.
pub fn createBuffer(device: *Device, size: usize, req_flags_list: []const vk.VkMemoryPropertyFlags, import_ptr: ?*anyopaque) common.VkError!?*Buffer {
    if (size > device.max_buffer_size) {
        return throwOutOfDeviceMemory("Requested buffer size exceeds device buffer size limit");
    }

    const buf = allocator.create(Buffer) catch oom();
    buf.* = .{ .device = device };

    if (size == 0) {
        buf.size = 0;
        return buf;
    }

    var usage_flags: vk.VkBufferUsageFlags = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    var mem_flags: vk.VkMemoryAllocateFlags = 0;
    if (device.buffer_device_address) {
        usage_flags |= vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        mem_flags |= vk.VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    }

    var bci = common.structure(vk.VkBufferCreateInfo, vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    bci.size = size;
    bci.usage = usage_flags;
    bci.sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE;
    var external_memory_bci = common.structure(vk.VkExternalMemoryBufferCreateInfo, vk.VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
    if (import_ptr != null) {
        external_memory_bci.handleTypes = vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        bci.pNext = &external_memory_bci;
    }

    common.check(d.vkCreateBuffer.?(device.device, &bci, null, &buf.buffer), "vk::Device::createBuffer") catch |err| {
        allocator.destroy(buf);
        return err;
    };

    var mem_req: vk.VkMemoryRequirements = undefined;
    d.vkGetBufferMemoryRequirements.?(device.device, buf.buffer, &mem_req);
    const mem_props = getMemoryProperties(device);

    var mem_priority_info = common.structure(vk.VkMemoryPriorityAllocateInfoEXT, vk.VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT);
    mem_priority_info.priority = 1.0;
    var mem_flags_info = common.structure(vk.VkMemoryAllocateFlagsInfo, vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO);
    mem_flags_info.flags = mem_flags;
    if (device.memory_priority) mem_flags_info.pNext = &mem_priority_info;

    if (import_ptr) |ptr| {
        var host_pointer_props = common.structure(vk.VkMemoryHostPointerPropertiesEXT, vk.VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT);
        const r = d.vkGetMemoryHostPointerPropertiesEXT.?(device.device, vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, ptr, &host_pointer_props);
        if (r != vk.VK_SUCCESS) {
            impl.logWarn("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s: %s)\n", .{ @as([*:0]const u8, "vk::Device::getMemoryHostPointerPropertiesEXT"), common.resultString(r).ptr });
            d.vkDestroyBuffer.?(device.device, buf.buffer, null);
            allocator.destroy(buf);
            return null;
        }
        const import_props = getMemoryProperties(device);

        var memory_type_idx: u32 = 0;
        var property_flags = req_flags_list[0];
        while (memory_type_idx < 32) : (memory_type_idx += 1) {
            const bit = @as(u32, 1) << @intCast(memory_type_idx);
            if (host_pointer_props.memoryTypeBits & bit == 0) continue;
            if (mem_req.memoryTypeBits & bit == 0) continue;
            const memory_type = import_props.memoryTypes[memory_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            if (memory_type.propertyFlags & property_flags == property_flags) {
                property_flags = memory_type.propertyFlags;
                break;
            }
        }
        if (memory_type_idx == 32) {
            impl.logWarn("ggml_vulkan: Memory type for host allocation not found\n", .{});
            d.vkDestroyBuffer.?(device.device, buf.buffer, null);
            allocator.destroy(buf);
            return null;
        }

        buf.memory_property_flags = import_props.memoryTypes[memory_type_idx].propertyFlags;
        var import_info = common.structure(vk.VkImportMemoryHostPointerInfoEXT, vk.VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT);
        import_info.handleType = vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        import_info.pHostPointer = ptr;
        import_info.pNext = &mem_flags_info;
        var mai = common.structure(vk.VkMemoryAllocateInfo, vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
        mai.allocationSize = size;
        mai.memoryTypeIndex = memory_type_idx;
        mai.pNext = &import_info;
        // A failed import is swallowed; the null memory is caught below.
        if (d.vkAllocateMemory.?(device.device, &mai, null, &buf.device_memory) != vk.VK_SUCCESS) buf.device_memory = null;
    } else {
        for (req_flags_list, 0..) |req_flags, it| {
            var indices: [vk.VK_MAX_MEMORY_TYPES]u32 = undefined;
            const n = findMemoryProperties(&mem_props, &mem_req, req_flags, &indices);
            if (n == 0) continue;

            var done = false;
            for (indices[0..n], 0..) |mtype, mi| {
                var mai = common.structure(vk.VkMemoryAllocateInfo, vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
                mai.allocationSize = mem_req.size;
                mai.memoryTypeIndex = mtype;
                mai.pNext = &mem_flags_info;
                const r = d.vkAllocateMemory.?(device.device, &mai, null, &buf.device_memory);
                if (r == vk.VK_SUCCESS) {
                    buf.memory_property_flags = mem_props.memoryTypes[mtype].propertyFlags;
                    done = true;
                    break;
                }
                buf.device_memory = null;
                // loop and retry; during last attempt throw the exception
                if (it + 1 == req_flags_list.len and mi + 1 == n) {
                    d.vkDestroyBuffer.?(device.device, buf.buffer, null);
                    allocator.destroy(buf);
                    try common.check(r, "vk::Device::allocateMemory");
                }
            }
            if (done) break;
        }
    }

    if (buf.device_memory == null) {
        d.vkDestroyBuffer.?(device.device, buf.buffer, null);
        allocator.destroy(buf);
        return throwOutOfDeviceMemory("No suitable memory type found");
    }

    buf.ptr = null;
    if (import_ptr != null) {
        buf.ptr = import_ptr;
    } else if (buf.memory_property_flags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0) {
        common.checkOrAbort(d.vkMapMemory.?(device.device, buf.device_memory, 0, vk.VK_WHOLE_SIZE, 0, &buf.ptr), "vk::Device::mapMemory");
    }

    common.checkOrAbort(d.vkBindBufferMemory.?(device.device, buf.buffer, buf.device_memory, 0), "vk::Device::bindBufferMemory");

    buf.size = size;

    if (device.buffer_device_address) {
        var address_info = common.structure(vk.VkBufferDeviceAddressInfo, vk.VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO);
        address_info.buffer = buf.buffer;
        buf.bda_addr = d.vkGetBufferDeviceAddress.?(device.device, &address_info);
    }

    memory_logger.logAllocation(buf, size);
    return buf;
}

/// Ports `ggml_vk_create_buffer_check` (ggml-vulkan.cpp:3568 @c1d0e7a00): a
/// failure prints the size and rethrows, and nothing catches it.
pub fn createBufferCheck(device: *Device, size: usize, req_flags: vk.VkMemoryPropertyFlags, fallback_flags: vk.VkMemoryPropertyFlags) *Buffer {
    return (createBuffer(device, size, &.{ req_flags, fallback_flags }, null) catch {
        impl.fprintfStderr("ggml_vulkan: Memory allocation of size %zu failed.\n", .{size});
        impl.fprintfStderr("ggml_vulkan: %s: %s\n", .{ common.last_error.what, common.resultString(common.last_error.result).ptr });
        impl.abort("ggml_vulkan: memory allocation failed");
    }).?;
}

/// Ports `ggml_vk_create_buffer_device` (ggml-vulkan.cpp:3578 @c1d0e7a00):
/// device memory for a ggml buffer, with the fallbacks the device's
/// settings allow.
///
/// Return: the buffer, or `error.VulkanError` after printing why -- the
/// C++ rethrows to `alloc_buffer`, which returns null.
pub fn createBufferDevice(device: *Device, size: usize) common.VkError!*Buffer {
    const hv = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const hc = vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const dl = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const result = if (device.prefer_host_memory)
        createBuffer(device, size, &.{ hv | hc, dl }, null)
    else if (device.uma)
        // On UMA, prefer host-visible memory so direct tensor borrowing works.
        createBuffer(device, size, &.{ dl | hv | hc, dl, hv | hc }, null)
    else if (device.disable_host_visible_vidmem)
        (if (device.allow_sysmem_fallback) createBuffer(device, size, &.{ dl, hv | hc }, null) else createBuffer(device, size, &.{dl}, null))
    else if (device.allow_sysmem_fallback)
        // use rebar if available, otherwise fallback to device only visible memory
        createBuffer(device, size, &.{ dl | hv | hc, dl, hv | hc }, null)
    else
        createBuffer(device, size, &.{ dl | hv | hc, dl }, null);

    const buf = result catch |err| {
        impl.fprintfStderr("ggml_vulkan: Device memory allocation of size %zu failed.\n", .{size});
        impl.fprintfStderr("ggml_vulkan: %s: %s\n", .{ common.last_error.what, common.resultString(common.last_error.result).ptr });
        return err;
    };
    return buf.?;
}

/// Ports `ggml_vk_destroy_buffer` (ggml-vulkan.cpp:3617 @c1d0e7a00): drops
/// this holder's reference and clears it.
pub fn destroyBuffer(buf: *?*Buffer) void {
    const b = buf.* orelse return;
    memory_logger.logDeallocation(b);
    b.unref();
    buf.* = null;
}

// -----------------------------------------------------------------------------
// Pinned host memory

/// Ports `ggml_vk_host_malloc` (ggml-vulkan.cpp:8018 @c1d0e7a00).
///
/// Return: the mapping, or null if no host-visible memory was found; a
/// failed allocation is `error.VulkanError`, as the C++ throws.
pub fn hostMalloc(device: *Device, size: usize) common.VkError!?*anyopaque {
    logMemory("ggml_vk_host_malloc(%zu)", .{size});
    const buf = (try createBuffer(device, size, &.{
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    }, null)).?;

    if (buf.memory_property_flags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT == 0) {
        impl.fprintfStderr("WARNING: failed to allocate %.2f MB of pinned memory\n", .{@as(f64, @floatFromInt(size)) / 1024.0 / 1024.0});
        // The C++ frees the memory and buffer here and then again in the
        // shared pointer's destructor; the port frees them once.
        buf.unref();
        return null;
    }

    _ = std.c.pthread_rwlock_wrlock(&device.pinned_memory_mutex);
    defer _ = std.c.pthread_rwlock_unlock(&device.pinned_memory_mutex);
    device.pinned_memory.append(allocator, .{ .ptr = @ptrCast(buf.ptr.?), .size = size, .buf = buf }) catch oom();
    return buf.ptr;
}

/// Ports `ggml_vk_host_free` (ggml-vulkan.cpp:8038 @c1d0e7a00).
pub fn hostFree(device: *Device, ptr: ?*anyopaque) void {
    const p: [*]u8 = @ptrCast(ptr orelse return);
    logMemory("ggml_vk_host_free(%p)", .{ptr});
    _ = std.c.pthread_rwlock_wrlock(&device.pinned_memory_mutex);
    defer _ = std.c.pthread_rwlock_unlock(&device.pinned_memory_mutex);

    for (device.pinned_memory.items, 0..) |pm, i| {
        if (@intFromPtr(p) >= @intFromPtr(pm.ptr) and @intFromPtr(p) < @intFromPtr(pm.ptr) + pm.size) {
            var b: ?*Buffer = pm.buf;
            destroyBuffer(&b);
            _ = device.pinned_memory.orderedRemove(i);
            return;
        }
    }
    impl.fprintfStderr("WARNING: failed to free pinned memory: memory not in map\n", .{});
}

/// Ports `ggml_vk_host_get` (ggml-vulkan.cpp:8066 @c1d0e7a00): the pinned
/// buffer holding `ptr`, and the offset of `ptr` in it.
///
/// Return: the buffer, borrowed, or null when `ptr` is not pinned.
pub fn hostGet(device: *Device, ptr: ?*const anyopaque, buf_offset: *usize) ?*Buffer {
    _ = std.c.pthread_rwlock_rdlock(&device.pinned_memory_mutex);
    defer _ = std.c.pthread_rwlock_unlock(&device.pinned_memory_mutex);
    buf_offset.* = 0;
    const p = @intFromPtr(ptr);
    for (device.pinned_memory.items) |pm| {
        const addr = @intFromPtr(pm.ptr);
        if (p >= addr and p < addr + pm.size) {
            buf_offset.* = p - addr;
            return pm.buf;
        }
    }
    return null;
}

// -----------------------------------------------------------------------------
// The device buffer

/// Ports `vk_ptr_base` (ggml-vulkan.cpp:2407 @c1d0e7a00): the fake base
/// address tensors in a Vulkan buffer are placed at.
pub const ptr_base: usize = 0x1000;

/// Ports `vk_tensor_offset` (ggml-vulkan.cpp:2409 @c1d0e7a00).
pub fn tensorOffset(tensor: *const c.ggml_tensor) u64 {
    if (tensor.view_src) |vs| return @intFromPtr(vs.*.data) -% ptr_base;
    return @intFromPtr(tensor.data) -% ptr_base;
}

/// Ports `ggml_backend_vk_buffer_context` (ggml-vulkan.cpp:2503 @c1d0e7a00).
pub const BufferCtx = struct {
    device: *Device,
    dev_buffer: ?*Buffer,
    name: [:0]const u8,
};

/// Ports `ggml_backend_vk_buffer_type_context` (ggml-vulkan.cpp:282 @c1d0e7a00).
pub const BufferTypeCtx = struct {
    name: [:0]const u8,
    device: *Device,

    pub fn create(name: [:0]const u8, device: *Device) *BufferTypeCtx {
        const ctx = allocator.create(BufferTypeCtx) catch oom();
        ctx.* = .{ .name = name, .device = device };
        return ctx;
    }
};

fn bufCtx(buffer: c.ggml_backend_buffer_t) *BufferCtx {
    return @ptrCast(@alignCast(buffer.*.context));
}

/// Ports `ggml_backend_buffer_is_vk` (ggml-vulkan.cpp:15988 @c1d0e7a00).
pub fn bufferIsVk(buffer: c.ggml_backend_buffer_t) bool {
    return isVkBufferType(buffer.*.buft);
}

/// Whether `buft` is a device buffer type of this backend: the C++'s
/// `buft->iface.get_name == ggml_backend_vk_buffer_type_name`.
pub fn isVkBufferType(buft: c.ggml_backend_buffer_type_t) bool {
    return buft.*.iface.get_name == &bufferTypeName;
}

/// The device buffer behind a tensor's ggml buffer.
pub fn tensorBuffer(tensor: *const c.ggml_tensor) *Buffer {
    return bufCtx(tensor.buffer).dev_buffer.?;
}

/// Ports `ggml_backend_vk_buffer_free_buffer` (ggml-vulkan.cpp:15992 @c1d0e7a00).
fn bufferFreeBuffer(buffer: c.ggml_backend_buffer_t) callconv(.c) void {
    logMemory("ggml_backend_vk_buffer_free_buffer()", .{});
    const ctx = bufCtx(buffer);
    destroyBuffer(&ctx.dev_buffer);
    allocator.destroy(ctx);
}

/// Ports `ggml_backend_vk_buffer_get_base` (ggml-vulkan.cpp:15999 @c1d0e7a00).
fn bufferGetBase(_: c.ggml_backend_buffer_t) callconv(.c) ?*anyopaque {
    return @ptrFromInt(ptr_base);
}

/// Ports `ggml_backend_vk_buffer_init_tensor` (ggml-vulkan.cpp:16005 @c1d0e7a00).
fn bufferInitTensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]c.ggml_tensor) callconv(.c) c.enum_ggml_status {
    const t = impl.one(c.ggml_tensor, tensor);
    if (t.view_src) |vs| impl.assert(vs.*.buffer.*.buft == buffer.*.buft, "tensor->view_src->buffer->buft == buffer->buft");
    return c.GGML_STATUS_SUCCESS;
}

/// Ports `ggml_backend_vk_buffer_memset_tensor` (ggml-vulkan.cpp:16013 @c1d0e7a00).
fn bufferMemsetTensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]c.ggml_tensor, value: u8, offset: usize, size: usize) callconv(.c) void {
    const buf = bufCtx(buffer).dev_buffer.?;
    if (size == 0) return;
    const t = impl.one(c.ggml_tensor, tensor);
    const val32 = @as(u32, value) *% 0x01010101;
    m.submit.bufferMemset(buf, tensorOffset(t) + t.view_offs + offset, val32, size);
}

/// Ports `ggml_backend_vk_buffer_set_tensor` (ggml-vulkan.cpp:16026 @c1d0e7a00).
fn bufferSetTensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    const buf = bufCtx(buffer).dev_buffer.?;
    if (size == 0) return;
    const t = impl.one(c.ggml_tensor, tensor);
    m.submit.bufferWrite(buf, tensorOffset(t) + t.view_offs + offset, @ptrCast(data.?), size);
}

/// Ports `ggml_backend_vk_buffer_set_tensor_2d` (ggml-vulkan.cpp:16038 @c1d0e7a00).
fn bufferSetTensor2d(buffer: c.ggml_backend_buffer_t, tensor: [*c]c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    const buf = bufCtx(buffer).dev_buffer.?;
    if (size == 0) return;
    const t = impl.one(c.ggml_tensor, tensor);
    m.submit.bufferWrite2d(buf, tensorOffset(t) + t.view_offs + offset, @ptrCast(data.?), stride_data, stride_tensor, size, n_copies);
}

/// Ports `ggml_backend_vk_buffer_get_tensor` (ggml-vulkan.cpp:16052 @c1d0e7a00).
fn bufferGetTensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]const c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    if (size == 0) return;
    const buf = bufCtx(buffer).dev_buffer.?;
    const t = impl.one(c.ggml_tensor, @constCast(tensor));
    m.submit.bufferRead(buf, tensorOffset(t) + t.view_offs + offset, @ptrCast(data.?), size);
}

/// Ports `ggml_backend_vk_buffer_get_tensor_2d` (ggml-vulkan.cpp:16065 @c1d0e7a00).
fn bufferGetTensor2d(buffer: c.ggml_backend_buffer_t, tensor: [*c]const c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    if (size == 0) return;
    const buf = bufCtx(buffer).dev_buffer.?;
    const t = impl.one(c.ggml_tensor, @constCast(tensor));
    m.submit.bufferRead2d(buf, tensorOffset(t) + t.view_offs + offset, @ptrCast(data.?), stride_tensor, stride_data, size, n_copies);
}

/// Ports `ggml_backend_vk_buffer_cpy_tensor` (ggml-vulkan.cpp:16080 @c1d0e7a00).
fn bufferCpyTensor(_: c.ggml_backend_buffer_t, src: [*c]const c.ggml_tensor, dst: [*c]c.ggml_tensor) callconv(.c) bool {
    if (c.ggml_nbytes(src) == 0) return true;
    const s = impl.one(c.ggml_tensor, @constCast(src));
    const t = impl.one(c.ggml_tensor, dst);
    if (bufferIsVk(s.buffer)) {
        const src_buf = bufCtx(s.buffer).dev_buffer.?;
        const dst_buf = bufCtx(t.buffer).dev_buffer.?;
        m.submit.bufferCopy(dst_buf, tensorOffset(t) + t.view_offs, src_buf, tensorOffset(s) + s.view_offs, c.ggml_nbytes(src));
        return true;
    }
    return false;
}

/// Ports `ggml_backend_vk_buffer_clear` (ggml-vulkan.cpp:16101 @c1d0e7a00).
/// The C++ passes the byte straight through as the fill value.
fn bufferClear(buffer: c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
    m.submit.bufferMemset(bufCtx(buffer).dev_buffer.?, 0, value, buffer.*.size);
}

/// Ports `ggml_backend_vk_buffer_interface` (ggml-vulkan.cpp:16107 @c1d0e7a00).
const buffer_iface: c.struct_ggml_backend_buffer_i = .{
    .free_buffer = &bufferFreeBuffer,
    .get_base = &bufferGetBase,
    .init_tensor = &bufferInitTensor,
    .memset_tensor = &bufferMemsetTensor,
    .set_tensor = &bufferSetTensor,
    .get_tensor = &bufferGetTensor,
    .set_tensor_2d = &bufferSetTensor2d,
    .get_tensor_2d = &bufferGetTensor2d,
    .cpy_tensor = &bufferCpyTensor,
    .clear = &bufferClear,
    .reset = null,
};

fn buftCtx(buft: c.ggml_backend_buffer_type_t) *BufferTypeCtx {
    return @ptrCast(@alignCast(buft.*.context));
}

/// Ports `ggml_backend_vk_buffer_type_name` (ggml-vulkan.cpp:16122 @c1d0e7a00).
fn bufferTypeName(buft: c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    return buftCtx(buft).name.ptr;
}

/// Ports `ggml_backend_vk_buffer_type_alloc_buffer` (ggml-vulkan.cpp:16128 @c1d0e7a00).
fn bufferTypeAllocBuffer(buft: c.ggml_backend_buffer_type_t, size: usize) callconv(.c) c.ggml_backend_buffer_t {
    logMemory("ggml_backend_vk_buffer_type_alloc_buffer(%zu)", .{size});
    const ctx = buftCtx(buft);
    const dev_buffer = createBufferDevice(ctx.device, size) catch return null;
    return wrapBuffer(buft, ctx.device, dev_buffer, ctx.name, size);
}

/// A ggml buffer over a device buffer: `new ggml_backend_vk_buffer_context`
/// and `ggml_backend_buffer_init` with the Vulkan buffer interface.
pub fn wrapBuffer(buft: c.ggml_backend_buffer_type_t, device: *Device, dev_buffer: *Buffer, name: [:0]const u8, size: usize) c.ggml_backend_buffer_t {
    const bufctx = allocator.create(BufferCtx) catch oom();
    bufctx.* = .{ .device = device, .dev_buffer = dev_buffer, .name = name };
    return c.ggml_backend_buffer_init(buft, buffer_iface, bufctx, size);
}

/// Ports `ggml_backend_vk_buffer_type_get_alignment` (ggml-vulkan.cpp:16144 @c1d0e7a00).
fn bufferTypeGetAlignment(buft: c.ggml_backend_buffer_type_t) callconv(.c) usize {
    return @intCast(buftCtx(buft).device.properties.limits.minStorageBufferOffsetAlignment);
}

/// Ports `ggml_backend_vk_buffer_type_get_max_size` (ggml-vulkan.cpp:16149 @c1d0e7a00).
fn bufferTypeGetMaxSize(buft: c.ggml_backend_buffer_type_t) callconv(.c) usize {
    return @intCast(buftCtx(buft).device.suballocation_block_size);
}

/// Ports `ggml_backend_vk_buffer_type_get_alloc_size` (ggml-vulkan.cpp:16154 @c1d0e7a00).
fn bufferTypeGetAllocSize(_: c.ggml_backend_buffer_type_t, tensor: [*c]const c.ggml_tensor) callconv(.c) usize {
    return c.ggml_nbytes(tensor);
}

/// Ports `ggml_backend_vk_buffer_type_interface` (ggml-vulkan.cpp:375 @c1d0e7a00).
pub const buffer_type_iface: c.struct_ggml_backend_buffer_type_i = .{
    .get_name = &bufferTypeName,
    .alloc_buffer = &bufferTypeAllocBuffer,
    .get_alignment = &bufferTypeGetAlignment,
    .get_max_size = &bufferTypeGetMaxSize,
    .get_alloc_size = &bufferTypeGetAllocSize,
    .is_host = null,
};

/// Ports `ggml_backend_vk_buffer_type` (ggml-vulkan.cpp:16160 @c1d0e7a00).
pub fn bufferType(dev_num: usize) callconv(.c) c.ggml_backend_buffer_type_t {
    m.instance.instanceInitOrAbort();
    const dev = m.device.getDevice(dev_num);
    return &dev.buffer_type;
}

// -----------------------------------------------------------------------------
// The pinned host buffer type

/// Ports `ggml_backend_vk_host_buffer_type_name` (ggml-vulkan.cpp:16172 @c1d0e7a00).
fn hostBufferTypeName(_: c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    return common.name ++ "_Host";
}

/// Ports `ggml_backend_vk_host_buffer_free_buffer` (ggml-vulkan.cpp:16178 @c1d0e7a00).
fn hostBufferFreeBuffer(buffer: c.ggml_backend_buffer_t) callconv(.c) void {
    logMemory("ggml_backend_vk_host_buffer_free_buffer()", .{});
    hostFree(m.device.devices[0].?, buffer.*.context);
}

/// Ports `ggml_backend_vk_host_buffer_type_alloc_buffer` (ggml-vulkan.cpp:16183 @c1d0e7a00).
fn hostBufferTypeAllocBuffer(buft: c.ggml_backend_buffer_type_t, size_in: usize) callconv(.c) c.ggml_backend_buffer_t {
    logMemory("ggml_backend_vk_host_buffer_type_alloc_buffer(%zu)", .{size_in});
    const size = size_in + 32; // Behave like the CPU buffer type
    const ptr = hostMalloc(m.device.devices[0].?, size) catch {
        impl.logWarn("ggml_vulkan: Failed to allocate pinned memory (%s: %s)\n", .{ common.last_error.what, common.resultString(common.last_error.result).ptr });
        // fallback to cpu buffer
        return c.ggml_backend_buft_alloc_buffer(c.ggml_backend_cpu_buffer_type(), size);
    };
    const buffer = c.ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer.*.buft = buft;
    buffer.*.iface.free_buffer = &hostBufferFreeBuffer;
    return buffer;
}

/// Ports `ggml_backend_vk_host_buffer_type_get_alignment` (ggml-vulkan.cpp:16205 @c1d0e7a00).
fn hostBufferTypeGetAlignment(_: c.ggml_backend_buffer_type_t) callconv(.c) usize {
    return @intCast(m.device.devices[0].?.properties.limits.minMemoryMapAlignment);
}

/// Ports `ggml_backend_vk_host_buffer_type_get_max_size` (ggml-vulkan.cpp:16211 @c1d0e7a00).
fn hostBufferTypeGetMaxSize(_: c.ggml_backend_buffer_type_t) callconv(.c) usize {
    return @intCast(m.device.devices[0].?.suballocation_block_size);
}

var host_buffer_type: c.struct_ggml_backend_buffer_type = undefined;
var host_buffer_type_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var host_buffer_type_ready = false;

fn initHostBufferType() void {
    host_buffer_type = .{
        .iface = .{
            .get_name = &hostBufferTypeName,
            .alloc_buffer = &hostBufferTypeAllocBuffer,
            .get_alignment = &hostBufferTypeGetAlignment,
            .get_max_size = &hostBufferTypeGetMaxSize,
            .get_alloc_size = c.ggml_backend_cpu_buffer_type().*.iface.get_alloc_size,
            .is_host = c.ggml_backend_cpu_buffer_type().*.iface.is_host,
        },
        .device = c.ggml_backend_reg_dev_get(m.backend.reg(), 0),
        .context = null,
    };
}

/// Ports `ggml_backend_vk_host_buffer_type` (ggml-vulkan.cpp:16219 @c1d0e7a00):
/// a function-local static, initialised on first call.
pub fn hostBufferType() callconv(.c) c.ggml_backend_buffer_type_t {
    _ = std.c.pthread_mutex_lock(&host_buffer_type_mutex);
    if (!host_buffer_type_ready) {
        initHostBufferType();
        host_buffer_type_ready = true;
    }
    _ = std.c.pthread_mutex_unlock(&host_buffer_type_mutex);
    // Make sure device 0 is initialized
    m.instance.instanceInitOrAbort();
    _ = m.device.getDevice(0);
    return &host_buffer_type;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "formatSize matches format_size's units and precision" {
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("1023 B", formatSize(1023, &buf));
    try std.testing.expectEqualStrings("1.00 KiB", formatSize(1024, &buf));
    try std.testing.expectEqualStrings("1.50 MiB", formatSize(3 * 512 * 1024, &buf));
    try std.testing.expectEqualStrings("2.00 GiB", formatSize(2 << 30, &buf));
}
