//! Device memory: `vk_buffer`, how its memory type is chosen, pinned host
//! memory, and the two buffer types the backend exposes to ggml.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp`,
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-debug.cpp`, and
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-types.h` at v1.0.0
//! (`b1b770d90`). A `vk_buffer` is a `std::shared_ptr` whose destructor frees the
//! memory; here it is reference counted by hand, and only the holders the
//! C++'s lifetimes depend on take a reference.

const std = @import("std");
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");
const builtin = @import("builtin");

fn oom() noreturn {
    ggml.impl.abort("ggml_vulkan: out of memory");
}

/// Ports `vk_buffer_struct` (ggml-vulkan-types.h:1073 @b1b770d90).
pub const Buffer = struct {
    buffer: ggml.vulkan.common.vk.VkBuffer = null,
    device_memory: ggml.vulkan.common.vk.VkDeviceMemory = null,
    memory_property_flags: ggml.vulkan.common.vk.VkMemoryPropertyFlags = 0,
    ptr: ?*anyopaque = null,
    size: usize = 0,
    bda_addr: ggml.vulkan.common.vk.VkDeviceAddress = 0,
    device: *ggml.vulkan.device.Device,
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
            // bump before destroying, so a thread that sees the buffer gone also sees the new count
            _ = self.device.buffer_destroy_count.fetchAdd(1, .release);
            ggml.vulkan.common.dispatch.vkFreeMemory.?(self.device.device, self.device_memory, null);
            ggml.vulkan.common.dispatch.vkDestroyBuffer.?(self.device.device, self.buffer, null);
        }
        std.heap.c_allocator.destroy(self);
    }
};

/// Ports `vk_subbuffer` (ggml-vulkan-types.h:1096 @b1b770d90). Transient: it
/// does not keep its buffer alive, and nothing in the C++ relies on it to.
pub const Subbuffer = struct {
    buffer: ?*Buffer = null,
    offset: u64 = 0,
    size: u64 = 0,

    /// `operator vk::DescriptorBufferInfo`.
    pub fn info(self: Subbuffer) ggml.vulkan.common.vk.VkDescriptorBufferInfo {
        return .{ .buffer = self.buffer.?.buffer, .offset = self.offset, .range = self.size };
    }
};

// -----------------------------------------------------------------------------
// The memory logger

/// Ports `format_size` (ggml-vulkan-types.h:1200 @b1b770d90).
///
/// `std::fixed << std::setprecision(2)` formats as `%.2f` does, through the
/// C library, which rounds an exact binary tie to even: 16000 bytes is
/// 15.625 KiB and prints `15.62`. Zig's `{d:.2}` rounds it away, `15.63`,
/// which `vk-backend-diff`'s memory-logger scenarios caught on the GB10.
fn formatSize(size: usize, buf: []u8) [:0]const u8 {
    const kib: usize = 1024;
    const mib = kib * 1024;
    const gib = mib * 1024;
    const f: f64 = @floatFromInt(size);
    const n = if (size >= gib)
        stdlib.libc.snprintf(buf.ptr, buf.len, "%.2f GiB", f / @as(f64, @floatFromInt(gib)))
    else if (size >= mib)
        stdlib.libc.snprintf(buf.ptr, buf.len, "%.2f MiB", f / @as(f64, @floatFromInt(mib)))
    else if (size >= kib)
        stdlib.libc.snprintf(buf.ptr, buf.len, "%.2f KiB", f / @as(f64, @floatFromInt(kib)))
    else
        stdlib.libc.snprintf(buf.ptr, buf.len, "%zu B", size);
    return buf[0..@min(@as(usize, @intCast(n)), buf.len - 1) :0];
}

/// Ports `vk_memory_logger` (ggml-vulkan-types.h:1221 @b1b770d90), one per
/// process here where the C++ keeps one per device; its totals are per
/// device in the C++, so they are keyed by device.
const memory_logger = struct {
    var mutex: stdlib.sync.Mutex = .{};
    var allocations: std.AutoHashMapUnmanaged(ggml.vulkan.common.vk.VkBuffer, usize) = .empty;
    var totals: std.AutoHashMapUnmanaged(*ggml.vulkan.device.Device, [2]usize) = .empty;

    /// Ports `vk_memory_logger`'s `log_allocation` (ggml-vulkan-debug.cpp:67 @b1b770d90).
    fn logAllocation(buf: *Buffer, size: usize) void {
        if (!ggml.vulkan.instance.loggers.memory_logger_enabled) return;
        mutex.lock();
        defer mutex.unlock();
        const device = buf.memory_property_flags & ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0;
        allocations.put(std.heap.c_allocator, buf.buffer, size) catch oom();
        const t = totals.getOrPutValue(std.heap.c_allocator, buf.device, .{ 0, 0 }) catch oom();
        if (device) t.value_ptr[0] += size else t.value_ptr[1] += size;
        var a: [32]u8 = undefined;
        var b: [32]u8 = undefined;
        var e: [32]u8 = undefined;
        ggml.impl.fprintfStderr("ggml_vulkan memory: %s: +%s %s at %p. Total device: %s, total host: %s\n", .{ buf.device.name.ptr, formatSize(size, &a).ptr, @as([*:0]const u8, if (device) "device" else "host"), @as(?*anyopaque, buf.buffer), formatSize(t.value_ptr[0], &b).ptr, formatSize(t.value_ptr[1], &e).ptr });
    }

    /// Ports `vk_memory_logger`'s `log_deallocation` (ggml-vulkan-debug.cpp:81 @b1b770d90).
    fn logDeallocation(buf: *Buffer) void {
        if (buf.size == 0 or !ggml.vulkan.instance.loggers.memory_logger_enabled) return;
        mutex.lock();
        defer mutex.unlock();
        const device = buf.memory_property_flags & ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0;
        const kind: [*:0]const u8 = if (device) "device" else "host";
        if (allocations.fetchRemove(buf.buffer)) |kv| {
            const t = totals.getOrPutValue(std.heap.c_allocator, buf.device, .{ 0, 0 }) catch oom();
            if (device) t.value_ptr[0] -= kv.value else t.value_ptr[1] -= kv.value;
            var a: [32]u8 = undefined;
            var b: [32]u8 = undefined;
            var e: [32]u8 = undefined;
            ggml.impl.fprintfStderr("ggml_vulkan memory: %s: -%s %s at %p. Total device: %s, total host: %s\n", .{ buf.device.name.ptr, formatSize(kv.value, &a).ptr, kind, @as(?*anyopaque, buf.buffer), formatSize(t.value_ptr[0], &b).ptr, formatSize(t.value_ptr[1], &e).ptr });
        } else {
            ggml.impl.fprintfStderr("ggml_vulkan memory: ERROR %s: Attempted to deallocate unknown %s memory at %p\n", .{ buf.device.name.ptr, kind, @as(?*anyopaque, buf.buffer) });
        }
    }
};

/// `VK_LOG_MEMORY` (ggml-vulkan-types.h:1198 @b1b770d90), for its plain-text uses.
pub fn logMemory(comptime fmt: []const u8, args: anytype) void {
    if (ggml.vulkan.instance.loggers.memory_logger_enabled) ggml.impl.fprintfStderr("ggml_vulkan memory: " ++ fmt ++ "\n", args);
}

// -----------------------------------------------------------------------------
// Creation

/// Ports `ggml_vk_find_memory_properties` (ggml-vulkan-buffers.cpp:39 @b1b770d90).
fn findMemoryProperties(mem_props: *const ggml.vulkan.common.vk.VkPhysicalDeviceMemoryProperties, mem_req: *const ggml.vulkan.common.vk.VkMemoryRequirements, flags: ggml.vulkan.common.vk.VkMemoryPropertyFlags, out: *[ggml.vulkan.common.vk.VK_MAX_MEMORY_TYPES]u32) usize {
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
fn throwOutOfDeviceMemory(comptime message: [*:0]const u8) ggml.vulkan.common.VkError {
    ggml.vulkan.common.last_error = .{ .result = ggml.vulkan.common.vk.VK_ERROR_OUT_OF_DEVICE_MEMORY, .what = message };
    return error.VulkanError;
}

/// `vk::PhysicalDevice::getMemoryProperties`.
fn getMemoryProperties(device: *ggml.vulkan.device.Device) ggml.vulkan.common.vk.VkPhysicalDeviceMemoryProperties {
    var props: ggml.vulkan.common.vk.VkPhysicalDeviceMemoryProperties = undefined;
    ggml.vulkan.common.dispatch.vkGetPhysicalDeviceMemoryProperties.?(device.physical_device, &props);
    return props;
}

/// Ports `ggml_vk_create_buffer` (ggml-vulkan-buffers.cpp:53 @b1b770d90): a buffer
/// backed by the first memory type, trying each set of flags in order,
/// that allocates; or, given `import_ptr`, imported host memory.
///
/// Return: the buffer; null where the C++ returns an empty pointer (an
/// import that fails); `error.VulkanError` where it throws.
pub fn createBuffer(device: *ggml.vulkan.device.Device, size: usize, req_flags_list: []const ggml.vulkan.common.vk.VkMemoryPropertyFlags, import_ptr: ?*anyopaque) ggml.vulkan.common.VkError!?*Buffer {
    if (size > device.max_buffer_size) {
        return throwOutOfDeviceMemory("Requested buffer size exceeds device buffer size limit");
    }

    const buf = std.heap.c_allocator.create(Buffer) catch oom();
    buf.* = .{ .device = device };

    if (size == 0) {
        buf.size = 0;
        return buf;
    }

    var usage_flags: ggml.vulkan.common.vk.VkBufferUsageFlags = ggml.vulkan.common.vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | ggml.vulkan.common.vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | ggml.vulkan.common.vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    var mem_flags: ggml.vulkan.common.vk.VkMemoryAllocateFlags = 0;
    if (device.buffer_device_address) {
        usage_flags |= ggml.vulkan.common.vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        mem_flags |= ggml.vulkan.common.vk.VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    }

    var bci = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkBufferCreateInfo, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    bci.size = size;
    bci.usage = usage_flags;
    bci.sharingMode = ggml.vulkan.common.vk.VK_SHARING_MODE_EXCLUSIVE;
    var external_memory_bci = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkExternalMemoryBufferCreateInfo, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
    if (import_ptr != null) {
        external_memory_bci.handleTypes = ggml.vulkan.common.vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        bci.pNext = &external_memory_bci;
    }

    ggml.vulkan.common.check(ggml.vulkan.common.dispatch.vkCreateBuffer.?(device.device, &bci, null, &buf.buffer), "vk::Device::createBuffer") catch |err| {
        std.heap.c_allocator.destroy(buf);
        return err;
    };

    var mem_req: ggml.vulkan.common.vk.VkMemoryRequirements = undefined;
    ggml.vulkan.common.dispatch.vkGetBufferMemoryRequirements.?(device.device, buf.buffer, &mem_req);
    const mem_props = getMemoryProperties(device);

    var mem_priority_info = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkMemoryPriorityAllocateInfoEXT, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT);
    mem_priority_info.priority = 1.0;
    var mem_flags_info = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkMemoryAllocateFlagsInfo, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO);
    mem_flags_info.flags = mem_flags;
    if (device.memory_priority) mem_flags_info.pNext = &mem_priority_info;

    if (import_ptr) |ptr| {
        var host_pointer_props = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkMemoryHostPointerPropertiesEXT, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT);
        const r = ggml.vulkan.common.dispatch.vkGetMemoryHostPointerPropertiesEXT.?(device.device, ggml.vulkan.common.vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, ptr, &host_pointer_props);
        if (r != ggml.vulkan.common.vk.VK_SUCCESS) {
            ggml.impl.logWarn("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s: %s)\n", .{ @as([*:0]const u8, "vk::Device::getMemoryHostPointerPropertiesEXT"), ggml.vulkan.common.resultString(r).ptr });
            ggml.vulkan.common.dispatch.vkDestroyBuffer.?(device.device, buf.buffer, null);
            std.heap.c_allocator.destroy(buf);
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
            ggml.impl.logWarn("ggml_vulkan: Memory type for host allocation not found\n", .{});
            ggml.vulkan.common.dispatch.vkDestroyBuffer.?(device.device, buf.buffer, null);
            std.heap.c_allocator.destroy(buf);
            return null;
        }

        buf.memory_property_flags = import_props.memoryTypes[memory_type_idx].propertyFlags;
        var import_info = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkImportMemoryHostPointerInfoEXT, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT);
        import_info.handleType = ggml.vulkan.common.vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        import_info.pHostPointer = ptr;
        import_info.pNext = &mem_flags_info;
        var mai = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkMemoryAllocateInfo, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
        mai.allocationSize = size;
        mai.memoryTypeIndex = memory_type_idx;
        mai.pNext = &import_info;
        // A failed import is swallowed; the null memory is caught below.
        if (ggml.vulkan.common.dispatch.vkAllocateMemory.?(device.device, &mai, null, &buf.device_memory) != ggml.vulkan.common.vk.VK_SUCCESS) buf.device_memory = null;
    } else {
        for (req_flags_list, 0..) |req_flags, it| {
            var indices: [ggml.vulkan.common.vk.VK_MAX_MEMORY_TYPES]u32 = undefined;
            const n = findMemoryProperties(&mem_props, &mem_req, req_flags, &indices);
            if (n == 0) continue;

            var done = false;
            for (indices[0..n], 0..) |mtype, mi| {
                var mai = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkMemoryAllocateInfo, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
                mai.allocationSize = mem_req.size;
                mai.memoryTypeIndex = mtype;
                mai.pNext = &mem_flags_info;
                const r = ggml.vulkan.common.dispatch.vkAllocateMemory.?(device.device, &mai, null, &buf.device_memory);
                if (r == ggml.vulkan.common.vk.VK_SUCCESS) {
                    buf.memory_property_flags = mem_props.memoryTypes[mtype].propertyFlags;
                    done = true;
                    break;
                }
                buf.device_memory = null;
                // loop and retry; during last attempt throw the exception
                if (it + 1 == req_flags_list.len and mi + 1 == n) {
                    ggml.vulkan.common.dispatch.vkDestroyBuffer.?(device.device, buf.buffer, null);
                    std.heap.c_allocator.destroy(buf);
                    try ggml.vulkan.common.check(r, "vk::Device::allocateMemory");
                }
            }
            if (done) break;
        }
    }

    if (buf.device_memory == null) {
        ggml.vulkan.common.dispatch.vkDestroyBuffer.?(device.device, buf.buffer, null);
        std.heap.c_allocator.destroy(buf);
        return throwOutOfDeviceMemory("No suitable memory type found");
    }

    buf.ptr = null;
    if (import_ptr != null) {
        buf.ptr = import_ptr;
    } else if (buf.memory_property_flags & ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0) {
        ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkMapMemory.?(device.device, buf.device_memory, 0, ggml.vulkan.common.vk.VK_WHOLE_SIZE, 0, &buf.ptr), "vk::Device::mapMemory");
    }

    ggml.vulkan.common.checkOrAbort(ggml.vulkan.common.dispatch.vkBindBufferMemory.?(device.device, buf.buffer, buf.device_memory, 0), "vk::Device::bindBufferMemory");

    buf.size = size;

    if (device.buffer_device_address) {
        var address_info = ggml.vulkan.common.structure(ggml.vulkan.common.vk.VkBufferDeviceAddressInfo, ggml.vulkan.common.vk.VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO);
        address_info.buffer = buf.buffer;
        buf.bda_addr = ggml.vulkan.common.dispatch.vkGetBufferDeviceAddress.?(device.device, &address_info);
    }

    memory_logger.logAllocation(buf, size);
    return buf;
}

/// Ports `ggml_vk_create_buffer_check` (ggml-vulkan-buffers.cpp:210 @b1b770d90): a
/// failure prints the size and rethrows, and nothing catches it.
pub fn createBufferCheck(device: *ggml.vulkan.device.Device, size: usize, req_flags: ggml.vulkan.common.vk.VkMemoryPropertyFlags, fallback_flags: ggml.vulkan.common.vk.VkMemoryPropertyFlags) *Buffer {
    return (createBuffer(device, size, &.{ req_flags, fallback_flags }, null) catch {
        ggml.impl.fprintfStderr("ggml_vulkan: Memory allocation of size %zu failed.\n", .{size});
        ggml.impl.fprintfStderr("ggml_vulkan: %s: %s\n", .{ ggml.vulkan.common.last_error.what, ggml.vulkan.common.resultString(ggml.vulkan.common.last_error.result).ptr });
        ggml.impl.abort("ggml_vulkan: memory allocation failed");
    }).?;
}

/// Ports `ggml_vk_create_buffer_device` (ggml-vulkan-buffers.cpp:220 @b1b770d90):
/// device memory for a ggml buffer, with the fallbacks the device's
/// settings allow.
///
/// Return: the buffer, or `error.VulkanError` after printing why -- the
/// C++ rethrows to `alloc_buffer`, which returns null.
pub fn createBufferDevice(device: *ggml.vulkan.device.Device, size: usize) ggml.vulkan.common.VkError!*Buffer {
    const hv = ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const hc = ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const dl = ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const result = if (device.prefer_host_memory)
        createBuffer(device, size, &.{ hv | hc, dl }, null)
    else if (device.uma)
        // On UMA, prefer host-visible memory so direct tensor borrowing works.
        createBuffer(device, size, &.{ dl | hv | hc, dl, hv | hc }, null)
    else if (!device.rebar or device.disable_host_visible_vidmem)
        (if (device.allow_sysmem_fallback) createBuffer(device, size, &.{ dl, hv | hc }, null) else createBuffer(device, size, &.{dl}, null))
    else if (device.allow_sysmem_fallback)
        // use rebar if available, otherwise fallback to device only visible memory
        createBuffer(device, size, &.{ dl | hv | hc, dl, hv | hc }, null)
    else
        createBuffer(device, size, &.{ dl | hv | hc, dl }, null);

    const buf = result catch |err| {
        ggml.impl.fprintfStderr("ggml_vulkan: Device memory allocation of size %zu failed.\n", .{size});
        ggml.impl.fprintfStderr("ggml_vulkan: %s: %s\n", .{ ggml.vulkan.common.last_error.what, ggml.vulkan.common.resultString(ggml.vulkan.common.last_error.result).ptr });
        return err;
    };
    return buf.?;
}

/// Ports `ggml_vk_destroy_buffer` (ggml-vulkan-buffers.cpp:259 @b1b770d90): drops
/// this holder's reference and clears it.
pub fn destroyBuffer(buf: *?*Buffer) void {
    const b = buf.* orelse return;
    memory_logger.logDeallocation(b);
    b.unref();
    buf.* = null;
}

// -----------------------------------------------------------------------------
// Pinned host memory

/// Ports `portable_aligned_malloc` (ggml-vulkan-buffers.cpp:8 @b1b770d90).
fn portableAlignedMalloc(alignment: usize, size: usize) ?*anyopaque {
    if (builtin.os.tag == .windows) return stdlib.libc._aligned_malloc(size, alignment);
    var ptr: ?*anyopaque = null;
    if (stdlib.libc.posix_memalign(&ptr, alignment, size) != 0) return null;
    return ptr;
}

/// Ports `portable_aligned_free` (ggml-vulkan-buffers.cpp:20 @b1b770d90).
fn portableAlignedFree(ptr: ?*anyopaque) void {
    if (builtin.os.tag == .windows) return stdlib.libc._aligned_free(ptr);
    std.c.free(ptr);
}

/// Ports `vk_host_allocation` (ggml-vulkan-types.h:1369 @b1b770d90): one
/// pinned host allocation and the buffer each device maps it through.
pub const HostAllocation = struct {
    base: ?*anyopaque = null,
    size: usize = 0,
    /// Indexed by `device.idx`; null = not (yet) mapped there. Each holds a
    /// reference.
    views: [ggml.vulkan.common.max_devices]?*Buffer = @splat(null),
};

/// Ports `vk_instance_t`'s members `host_allocations` and
/// `host_allocations_mutex` (ggml-vulkan-types.h:1391, 1392 @b1b770d90), which
/// live here rather than in `instance.zig`'s `Instance`. Every lock the C++ takes on the
/// `std::shared_mutex` is a unique one, so a write lock stands for each.
var host_allocations: std.ArrayList(HostAllocation) = .empty;
var host_allocations_mutex: std.c.pthread_rwlock_t = .{};

/// Ports `ggml_vk_host_allocation_find` (ggml-vulkan-buffers.cpp:272 @b1b770d90).
/// The caller must hold `host_allocations_mutex`.
///
/// Return: the index of the allocation holding `ptr`, or null.
fn hostAllocationFind(ptr: ?*const anyopaque) ?usize {
    const p = @intFromPtr(ptr);
    for (host_allocations.items, 0..) |entry, i| {
        const base = @intFromPtr(entry.base);
        if (p >= base and p < base + entry.size) return i;
    }
    return null;
}

/// Ports `ggml_vk_host_malloc` (ggml-vulkan-buffers.cpp:282 @b1b770d90). With
/// several devices that can all import host memory, the allocation is plain
/// aligned host memory each device maps on first use; otherwise it is a
/// host-visible buffer of device 0.
///
/// Return: the mapping, or null if it could not be allocated or no
/// host-visible memory was found.
pub fn hostMalloc(size: usize) ?*anyopaque {
    logMemory("ggml_vk_host_malloc(%zu)", .{size});

    ggml.vulkan.instance.instanceInitOrAbort();

    if (ggml.vulkan.instance.instance.multi_device) {
        const alignment = @max(ggml.vulkan.instance.instance.host_import_alignment, ggml.impl.tensor_alignment);
        const alloc_size = (size + alignment - 1) / alignment * alignment;

        const ptr = portableAlignedMalloc(alignment, alloc_size) orelse return null;

        _ = std.c.pthread_rwlock_wrlock(&host_allocations_mutex);
        defer _ = std.c.pthread_rwlock_unlock(&host_allocations_mutex);
        host_allocations.append(std.heap.c_allocator, .{ .base = ptr, .size = alloc_size }) catch oom();
        return ptr;
    }

    const dev0 = ggml.vulkan.device.getDevice(0);
    var buf: ?*Buffer = (createBuffer(dev0, size, &.{
        ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    }, null) catch return null).?;

    if (buf.?.memory_property_flags & ggml.vulkan.common.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT == 0) {
        ggml.impl.fprintfStderr("WARNING: failed to allocate %.2f MB of pinned memory\n", .{@as(f64, @floatFromInt(size)) / 1024.0 / 1024.0});
        destroyBuffer(&buf);
        return null;
    }

    var entry: HostAllocation = .{ .base = buf.?.ptr, .size = size };
    entry.views[dev0.idx] = buf;

    _ = std.c.pthread_rwlock_wrlock(&host_allocations_mutex);
    defer _ = std.c.pthread_rwlock_unlock(&host_allocations_mutex);
    host_allocations.append(std.heap.c_allocator, entry) catch oom();
    return entry.base;
}

/// Ports `ggml_vk_host_free` (ggml-vulkan-buffers.cpp:330 @b1b770d90).
pub fn hostFree(ptr: ?*anyopaque) void {
    if (ptr == null) return;
    logMemory("ggml_vk_host_free(%p)", .{ptr});

    var entry: HostAllocation = blk: {
        _ = std.c.pthread_rwlock_wrlock(&host_allocations_mutex);
        defer _ = std.c.pthread_rwlock_unlock(&host_allocations_mutex);
        const index = hostAllocationFind(ptr) orelse {
            ggml.impl.fprintfStderr("WARNING: failed to free pinned memory: memory not in map\n", .{});
            return;
        };
        break :blk host_allocations.orderedRemove(index);
    };

    for (&entry.views) |*view| destroyBuffer(view);
    if (ggml.vulkan.instance.instance.multi_device) portableAlignedFree(entry.base);
}

/// Ports `ggml_vk_host_get` (ggml-vulkan-buffers.cpp:356 @b1b770d90): the
/// buffer through which `device` sees the pinned memory holding `ptr`, and
/// the offset of `ptr` in it. With several devices, a device maps the
/// allocation the first time it asks; a device whose import failed once is
/// not asked again, and stages from then on.
///
/// Return: the buffer, borrowed, or null when `ptr` is not pinned or this
/// device cannot map it.
pub fn hostGet(device: *ggml.vulkan.device.Device, ptr: ?*const anyopaque, buf_offset: *usize) ?*Buffer {
    buf_offset.* = 0;

    _ = std.c.pthread_rwlock_wrlock(&host_allocations_mutex);
    defer _ = std.c.pthread_rwlock_unlock(&host_allocations_mutex);

    const index = hostAllocationFind(ptr) orelse return null;
    const entry = &host_allocations.items[index];

    if (entry.views[device.idx] == null) {
        const inst = &ggml.vulkan.instance.instance;
        const bit = @as(u32, 1) << @intCast(device.idx);
        if (!inst.multi_device or !inst.device_host_buffer_import.items[device.idx] or (inst.host_import_failed & bit) != 0) return null;
        ggml.impl.assert(device.external_memory_host, "device->external_memory_host");
        entry.views[device.idx] = ggml.vulkan.iface.bufferFromHostPtr(device, entry.base, entry.size);
        if (entry.views[device.idx] == null) {
            inst.host_import_failed |= bit;
            ggml.impl.logWarn("ggml_vulkan: %s: host memory import failed, using staging\n", .{device.name.ptr});
            return null;
        }
    }

    buf_offset.* = @intFromPtr(ptr) - @intFromPtr(entry.base);
    return entry.views[device.idx];
}

// -----------------------------------------------------------------------------
// The device buffer

/// Ports `vk_ptr_base` (ggml-vulkan.cpp:260 @b1b770d90): the fake base
/// address tensors in a Vulkan buffer are placed at.
pub const ptr_base: usize = 0x1000;

/// Ports `vk_tensor_offset` (ggml-vulkan.cpp:262 @b1b770d90).
pub fn tensorOffset(tensor: *const ggml.impl.c.ggml_tensor) u64 {
    if (tensor.view_src) |vs| return @intFromPtr(vs.*.data) -% ptr_base;
    return @intFromPtr(tensor.data) -% ptr_base;
}

/// Ports `ggml_backend_vk_buffer_context` (ggml-vulkan-types.h:1354 @b1b770d90).
pub const BufferCtx = struct {
    device: *ggml.vulkan.device.Device,
    dev_buffer: ?*Buffer,
    name: [:0]const u8,
};

/// Ports `ggml_backend_vk_buffer_type_context` (ggml-vulkan-types.h:297 @b1b770d90).
pub const BufferTypeCtx = struct {
    name: [:0]const u8,
    device: *ggml.vulkan.device.Device,

    pub fn create(name: [:0]const u8, device: *ggml.vulkan.device.Device) *BufferTypeCtx {
        const ctx = std.heap.c_allocator.create(BufferTypeCtx) catch oom();
        ctx.* = .{ .name = name, .device = device };
        return ctx;
    }
};

fn bufCtx(buffer: ggml.impl.c.ggml_backend_buffer_t) *BufferCtx {
    return @ptrCast(@alignCast(buffer.*.context));
}

/// Ports `ggml_backend_buffer_is_vk` (ggml-vulkan.cpp:13425 @b1b770d90).
pub fn bufferIsVk(buffer: ggml.impl.c.ggml_backend_buffer_t) bool {
    return isVkBufferType(buffer.*.buft);
}

/// Whether `buft` is a device buffer type of this backend: the C++'s
/// `buft->iface.get_name == ggml_backend_vk_buffer_type_name`.
pub fn isVkBufferType(buft: ggml.impl.c.ggml_backend_buffer_type_t) bool {
    return buft.*.iface.get_name == &bufferTypeName;
}

/// The device buffer behind a tensor's ggml buffer.
pub fn tensorBuffer(tensor: *const ggml.impl.c.ggml_tensor) *Buffer {
    return bufCtx(tensor.buffer).dev_buffer.?;
}

/// The device buffer behind a ggml buffer of this backend: the C++'s
/// `((ggml_backend_vk_buffer_context *) buffer->context)->dev_buffer`.
pub fn devBuffer(buffer: ggml.impl.c.ggml_backend_buffer_t) *Buffer {
    return bufCtx(buffer).dev_buffer.?;
}

/// Ports `ggml_backend_vk_buffer_free_buffer` (ggml-vulkan.cpp:13429 @b1b770d90).
fn bufferFreeBuffer(buffer: ggml.impl.c.ggml_backend_buffer_t) callconv(.c) void {
    logMemory("ggml_backend_vk_buffer_free_buffer()", .{});
    const ctx = bufCtx(buffer);
    destroyBuffer(&ctx.dev_buffer);
    std.heap.c_allocator.destroy(ctx);
}

/// Ports `ggml_backend_vk_buffer_get_base` (ggml-vulkan.cpp:13436 @b1b770d90).
fn bufferGetBase(_: ggml.impl.c.ggml_backend_buffer_t) callconv(.c) ?*anyopaque {
    return @ptrFromInt(ptr_base);
}

/// Ports `ggml_backend_vk_buffer_init_tensor` (ggml-vulkan.cpp:13442 @b1b770d90).
fn bufferInitTensor(buffer: ggml.impl.c.ggml_backend_buffer_t, tensor: [*c]ggml.impl.c.ggml_tensor) callconv(.c) ggml.impl.c.enum_ggml_status {
    const t = ggml.impl.one(ggml.impl.c.ggml_tensor, tensor);
    if (t.view_src) |vs| ggml.impl.assert(vs.*.buffer.*.buft == buffer.*.buft, "tensor->view_src->buffer->buft == buffer->buft");
    return ggml.impl.c.GGML_STATUS_SUCCESS;
}

/// Ports `ggml_backend_vk_buffer_memset_tensor` (ggml-vulkan.cpp:13450 @b1b770d90).
fn bufferMemsetTensor(buffer: ggml.impl.c.ggml_backend_buffer_t, tensor: [*c]ggml.impl.c.ggml_tensor, value: u8, offset: usize, size: usize) callconv(.c) void {
    const buf = bufCtx(buffer).dev_buffer.?;
    if (size == 0) return;
    const t = ggml.impl.one(ggml.impl.c.ggml_tensor, tensor);
    const val32 = @as(u32, value) *% 0x01010101;
    ggml.vulkan.submit.bufferMemset(buf, tensorOffset(t) + t.view_offs + offset, val32, size);
}

/// Ports `ggml_backend_vk_buffer_set_tensor` (ggml-vulkan.cpp:13463 @b1b770d90).
fn bufferSetTensor(buffer: ggml.impl.c.ggml_backend_buffer_t, tensor: [*c]ggml.impl.c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    const buf = bufCtx(buffer).dev_buffer.?;
    if (size == 0) return;
    const t = ggml.impl.one(ggml.impl.c.ggml_tensor, tensor);
    ggml.vulkan.submit.bufferWrite(buf, tensorOffset(t) + t.view_offs + offset, @ptrCast(data.?), size);
}

/// Ports `ggml_backend_vk_buffer_set_tensor_2d` (ggml-vulkan.cpp:13475 @b1b770d90).
fn bufferSetTensor2d(buffer: ggml.impl.c.ggml_backend_buffer_t, tensor: [*c]ggml.impl.c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    const buf = bufCtx(buffer).dev_buffer.?;
    if (size == 0) return;
    const t = ggml.impl.one(ggml.impl.c.ggml_tensor, tensor);
    ggml.vulkan.submit.bufferWrite2d(buf, tensorOffset(t) + t.view_offs + offset, @ptrCast(data.?), stride_data, stride_tensor, size, n_copies);
}

/// Ports `ggml_backend_vk_buffer_get_tensor` (ggml-vulkan.cpp:13489 @b1b770d90).
fn bufferGetTensor(buffer: ggml.impl.c.ggml_backend_buffer_t, tensor: [*c]const ggml.impl.c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    if (size == 0) return;
    const buf = bufCtx(buffer).dev_buffer.?;
    const t = ggml.impl.one(ggml.impl.c.ggml_tensor, @constCast(tensor));
    ggml.vulkan.submit.bufferRead(buf, tensorOffset(t) + t.view_offs + offset, @ptrCast(data.?), size);
}

/// Ports `ggml_backend_vk_buffer_get_tensor_2d` (ggml-vulkan.cpp:13502 @b1b770d90).
fn bufferGetTensor2d(buffer: ggml.impl.c.ggml_backend_buffer_t, tensor: [*c]const ggml.impl.c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    if (size == 0) return;
    const buf = bufCtx(buffer).dev_buffer.?;
    const t = ggml.impl.one(ggml.impl.c.ggml_tensor, @constCast(tensor));
    ggml.vulkan.submit.bufferRead2d(buf, tensorOffset(t) + t.view_offs + offset, @ptrCast(data.?), stride_tensor, stride_data, size, n_copies);
}

/// Ports `ggml_backend_vk_buffer_cpy_tensor` (ggml-vulkan.cpp:13517 @b1b770d90).
fn bufferCpyTensor(_: ggml.impl.c.ggml_backend_buffer_t, src: [*c]const ggml.impl.c.ggml_tensor, dst: [*c]ggml.impl.c.ggml_tensor) callconv(.c) bool {
    if (ggml.impl.c.ggml_nbytes(src) == 0) return true;
    const s = ggml.impl.one(ggml.impl.c.ggml_tensor, @constCast(src));
    const t = ggml.impl.one(ggml.impl.c.ggml_tensor, dst);
    if (bufferIsVk(s.buffer)) {
        const src_buf = bufCtx(s.buffer).dev_buffer.?;
        const dst_buf = bufCtx(t.buffer).dev_buffer.?;
        ggml.vulkan.submit.bufferCopy(dst_buf, tensorOffset(t) + t.view_offs, src_buf, tensorOffset(s) + s.view_offs, ggml.impl.c.ggml_nbytes(src));
        return true;
    }
    return false;
}

/// Ports `ggml_backend_vk_buffer_clear` (ggml-vulkan.cpp:13586 @b1b770d90).
/// The C++ passes the byte straight through as the fill value.
fn bufferClear(buffer: ggml.impl.c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
    ggml.vulkan.submit.bufferMemset(bufCtx(buffer).dev_buffer.?, 0, value, buffer.*.size);
}

/// Ports `ggml_vk_tensor_dev_buffer` (ggml-vulkan.cpp:13538 @b1b770d90): the
/// device buffer behind a tensor or the tensor it views.
fn tensorDevBuffer(tensor: *const ggml.impl.c.ggml_tensor) *Buffer {
    const buffer = if (tensor.view_src) |vs| vs.*.buffer else tensor.buffer;
    return bufCtx(buffer).dev_buffer.?;
}

/// Ports `ggml_backend_vk_buffer_set_tensor_n` (ggml-vulkan.cpp:13543 @b1b770d90).
fn bufferSetTensorN(_: ggml.impl.c.ggml_backend_buffer_t, tensors: [*c]const [*c]ggml.impl.c.ggml_tensor, data: [*c]const ?*const anyopaque, offsets: [*c]const usize, sizes: [*c]const usize, n_tensors: usize) callconv(.c) void {
    const regions = std.heap.c_allocator.alloc(ggml.vulkan.submit.BufferRegion, n_tensors) catch ggml.impl.abort("ggml_vulkan: out of memory");
    defer std.heap.c_allocator.free(regions);
    for (regions, 0..) |*r, i| {
        const t = ggml.impl.one(ggml.impl.c.ggml_tensor, tensors[i]);
        r.* = .{ .buf = tensorDevBuffer(t), .offset = tensorOffset(t) + t.view_offs + offsets[i], .host = @constCast(data[i]), .size = sizes[i] };
    }
    ggml.vulkan.submit.bufferWriteN(regions);
}

/// Ports `ggml_backend_vk_buffer_get_tensor_n` (ggml-vulkan.cpp:13555 @b1b770d90).
fn bufferGetTensorN(_: ggml.impl.c.ggml_backend_buffer_t, tensors: [*c]const [*c]const ggml.impl.c.ggml_tensor, data: [*c]const ?*anyopaque, offsets: [*c]const usize, sizes: [*c]const usize, n_tensors: usize) callconv(.c) void {
    const regions = std.heap.c_allocator.alloc(ggml.vulkan.submit.BufferRegion, n_tensors) catch ggml.impl.abort("ggml_vulkan: out of memory");
    defer std.heap.c_allocator.free(regions);
    for (regions, 0..) |*r, i| {
        const t = ggml.impl.one(ggml.impl.c.ggml_tensor, @constCast(tensors[i]));
        r.* = .{ .buf = tensorDevBuffer(t), .offset = tensorOffset(t) + t.view_offs + offsets[i], .host = data[i], .size = sizes[i] };
    }
    ggml.vulkan.submit.bufferReadN(regions);
}

/// Ports `ggml_backend_vk_buffer_cpy_tensor_n` (ggml-vulkan.cpp:13567 @b1b770d90).
fn bufferCpyTensorN(_: ggml.impl.c.ggml_backend_buffer_t, srcs: [*c]const [*c]const ggml.impl.c.ggml_tensor, dsts: [*c]const [*c]ggml.impl.c.ggml_tensor, n_tensors: usize) callconv(.c) bool {
    for (0..n_tensors) |i| {
        const s = ggml.impl.one(ggml.impl.c.ggml_tensor, @constCast(srcs[i]));
        if (!bufferIsVk(if (s.view_src) |vs| vs.*.buffer else s.buffer)) return false;
    }

    const regions = std.heap.c_allocator.alloc(ggml.vulkan.submit.BufferCopyRegion, n_tensors) catch ggml.impl.abort("ggml_vulkan: out of memory");
    defer std.heap.c_allocator.free(regions);
    for (regions, 0..) |*r, i| {
        const s = ggml.impl.one(ggml.impl.c.ggml_tensor, @constCast(srcs[i]));
        const t = ggml.impl.one(ggml.impl.c.ggml_tensor, dsts[i]);
        r.* = .{ .dst = tensorDevBuffer(t), .dst_offset = tensorOffset(t) + t.view_offs, .src = tensorDevBuffer(s), .src_offset = tensorOffset(s) + s.view_offs, .size = ggml.impl.c.ggml_nbytes(s) };
    }
    ggml.vulkan.submit.bufferCopyN(regions);
    return true;
}

/// Ports `ggml_backend_vk_buffer_interface` (ggml-vulkan-buffers.cpp:1031 @b1b770d90).
const buffer_iface: ggml.impl.c.struct_ggml_backend_buffer_i = .{
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
    .set_tensor_n = &bufferSetTensorN,
    .get_tensor_n = &bufferGetTensorN,
    .cpy_tensor_n = &bufferCpyTensorN,
};

fn buftCtx(buft: ggml.impl.c.ggml_backend_buffer_type_t) *BufferTypeCtx {
    return @ptrCast(@alignCast(buft.*.context));
}

/// Ports `ggml_backend_vk_buffer_type_name` (ggml-vulkan.cpp:13592 @b1b770d90).
fn bufferTypeName(buft: ggml.impl.c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    return buftCtx(buft).name.ptr;
}

/// Ports `ggml_backend_vk_buffer_type_alloc_buffer` (ggml-vulkan.cpp:13598 @b1b770d90).
fn bufferTypeAllocBuffer(buft: ggml.impl.c.ggml_backend_buffer_type_t, size: usize) callconv(.c) ggml.impl.c.ggml_backend_buffer_t {
    logMemory("ggml_backend_vk_buffer_type_alloc_buffer(%zu)", .{size});
    const ctx = buftCtx(buft);
    const dev_buffer = createBufferDevice(ctx.device, size) catch return null;
    return wrapBuffer(buft, ctx.device, dev_buffer, ctx.name, size);
}

/// A ggml buffer over a device buffer: `new ggml_backend_vk_buffer_context`
/// and `ggml_backend_buffer_init` with the Vulkan buffer interface.
pub fn wrapBuffer(buft: ggml.impl.c.ggml_backend_buffer_type_t, device: *ggml.vulkan.device.Device, dev_buffer: *Buffer, name: [:0]const u8, size: usize) ggml.impl.c.ggml_backend_buffer_t {
    const bufctx = std.heap.c_allocator.create(BufferCtx) catch oom();
    bufctx.* = .{ .device = device, .dev_buffer = dev_buffer, .name = name };
    return ggml.impl.c.ggml_backend_buffer_init(buft, buffer_iface, bufctx, size);
}

/// Ports `ggml_backend_vk_buffer_type_get_alignment` (ggml-vulkan.cpp:13614 @b1b770d90).
fn bufferTypeGetAlignment(buft: ggml.impl.c.ggml_backend_buffer_type_t) callconv(.c) usize {
    return @intCast(buftCtx(buft).device.properties.limits.minStorageBufferOffsetAlignment);
}

/// Ports `ggml_backend_vk_buffer_type_get_max_size` (ggml-vulkan.cpp:13619 @b1b770d90).
fn bufferTypeGetMaxSize(buft: ggml.impl.c.ggml_backend_buffer_type_t) callconv(.c) usize {
    return @intCast(buftCtx(buft).device.suballocation_block_size);
}

/// Ports `ggml_backend_vk_buffer_type_get_alloc_size` (ggml-vulkan.cpp:13624 @b1b770d90).
fn bufferTypeGetAllocSize(_: ggml.impl.c.ggml_backend_buffer_type_t, tensor: [*c]const ggml.impl.c.ggml_tensor) callconv(.c) usize {
    return ggml.impl.c.ggml_nbytes(tensor);
}

/// Ports `ggml_backend_vk_buffer_type_interface` (ggml-vulkan-buffers.cpp:28 @b1b770d90).
pub const buffer_type_iface: ggml.impl.c.struct_ggml_backend_buffer_type_i = .{
    .get_name = &bufferTypeName,
    .alloc_buffer = &bufferTypeAllocBuffer,
    .alloc_buffer_n = null,
    .get_alignment = &bufferTypeGetAlignment,
    .get_max_size = &bufferTypeGetMaxSize,
    .get_alloc_size = &bufferTypeGetAllocSize,
    .get_alloc_size_n = null,
    .is_host = null,
};

/// Ports `ggml_backend_vk_buffer_type` (ggml-vulkan.cpp:13630 @b1b770d90).
pub fn bufferType(dev_num: usize) callconv(.c) ggml.impl.c.ggml_backend_buffer_type_t {
    ggml.vulkan.instance.instanceInitOrAbort();
    const dev = ggml.vulkan.device.getDevice(dev_num);
    return &dev.buffer_type;
}

// -----------------------------------------------------------------------------
// The pinned host buffer type

/// Ports `ggml_backend_vk_host_buffer_type_name` (ggml-vulkan.cpp:13640 @b1b770d90).
fn hostBufferTypeName(_: ggml.impl.c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    return ggml.vulkan.common.name ++ "_Host";
}

/// Ports `ggml_backend_vk_host_buffer_free_buffer` (ggml-vulkan.cpp:13646 @b1b770d90).
fn hostBufferFreeBuffer(buffer: ggml.impl.c.ggml_backend_buffer_t) callconv(.c) void {
    logMemory("ggml_backend_vk_host_buffer_free_buffer()", .{});
    hostFree(buffer.*.context);
}

/// Ports `ggml_backend_vk_host_buffer_type_alloc_buffer` (ggml-vulkan.cpp:13651 @b1b770d90).
fn hostBufferTypeAllocBuffer(buft: ggml.impl.c.ggml_backend_buffer_type_t, size_in: usize) callconv(.c) ggml.impl.c.ggml_backend_buffer_t {
    logMemory("ggml_backend_vk_host_buffer_type_alloc_buffer(%zu)", .{size_in});
    const size = size_in + 32; // Behave like the CPU buffer type
    const ptr = hostMalloc(size) orelse {
        ggml.impl.logWarn("ggml_vulkan: Failed to allocate pinned memory\n", .{});
        // fallback to cpu buffer
        return ggml.impl.c.ggml_backend_buft_alloc_buffer(ggml.impl.c.ggml_backend_cpu_buffer_type(), size);
    };
    const buffer = ggml.impl.c.ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer.*.buft = buft;
    buffer.*.iface.free_buffer = &hostBufferFreeBuffer;
    return buffer;
}

/// Ports `ggml_backend_vk_host_buffer_type_get_alignment` (ggml-vulkan.cpp:13671 @b1b770d90).
fn hostBufferTypeGetAlignment(_: ggml.impl.c.ggml_backend_buffer_type_t) callconv(.c) usize {
    return @intCast(ggml.vulkan.device.devices[0].?.properties.limits.minMemoryMapAlignment);
}

/// Ports `ggml_backend_vk_host_buffer_type_get_max_size` (ggml-vulkan.cpp:13677 @b1b770d90).
fn hostBufferTypeGetMaxSize(_: ggml.impl.c.ggml_backend_buffer_type_t) callconv(.c) usize {
    return @intCast(ggml.vulkan.device.devices[0].?.suballocation_block_size);
}

var host_buffer_type: ggml.impl.c.struct_ggml_backend_buffer_type = undefined;
var host_buffer_type_mutex: stdlib.sync.Mutex = .{};
var host_buffer_type_ready = false;

fn initHostBufferType() void {
    host_buffer_type = .{
        .iface = .{
            .get_name = &hostBufferTypeName,
            .alloc_buffer = &hostBufferTypeAllocBuffer,
            .alloc_buffer_n = null,
            .get_alignment = &hostBufferTypeGetAlignment,
            .get_max_size = &hostBufferTypeGetMaxSize,
            .get_alloc_size = ggml.impl.c.ggml_backend_cpu_buffer_type().*.iface.get_alloc_size,
            .get_alloc_size_n = null,
            .is_host = ggml.impl.c.ggml_backend_cpu_buffer_type().*.iface.is_host,
        },
        .device = ggml.impl.c.ggml_backend_reg_dev_get(ggml.vulkan.backend.reg(), 0),
        .context = null,
    };
}

/// Ports `ggml_backend_vk_host_buffer_type` (ggml-vulkan.cpp:13683 @b1b770d90):
/// a function-local static, initialised on first call.
pub fn hostBufferType() callconv(.c) ggml.impl.c.ggml_backend_buffer_type_t {
    host_buffer_type_mutex.lock();
    if (!host_buffer_type_ready) {
        initHostBufferType();
        host_buffer_type_ready = true;
    }
    host_buffer_type_mutex.unlock();
    // Make sure device 0 is initialized
    ggml.vulkan.instance.instanceInitOrAbort();
    _ = ggml.vulkan.device.getDevice(0);
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
    // An exact binary tie rounds to even, as the C library's %.2f does.
    try std.testing.expectEqualStrings("15.62 KiB", formatSize(16000, &buf));
    try std.testing.expectEqualStrings("15.38 KiB", formatSize(15744, &buf));
}
