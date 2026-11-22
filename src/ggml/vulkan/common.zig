//! What every file of the Vulkan host layer shares: the C API, the dynamic
//! dispatch table, `vulkan.hpp`'s error and enumeration conventions, and the
//! backend's constants.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-types.h` and
//! `llama.cpp/ggml/include/ggml-vulkan.h` at v1.0.0 (`b1b770d90`). The C++
//! is written against `vulkan.hpp`, a header-only C++ wrapper over the C API
//! in `vulkan_core.h`; what it adds that changes behaviour is reproduced here
//! -- the dispatcher, exceptions on a failed `VkResult`, and the
//! `VK_INCOMPLETE` retry of every `enumerate*` -- and nothing else.

const std = @import("std");
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");
const builtin = @import("builtin");

/// The Vulkan C API, `vulkan_core.h` at the pinned SDK,
/// translated with `VK_NO_PROTOTYPES`: every command is reached through
/// `dispatch`, as `vulkan.hpp`'s dynamic dispatcher reaches it.
pub const vulkan_c = @import("vulkan_c");

/// Ports `GGML_VK_NAME` (ggml-vulkan.h:10 @b1b770d90).
pub const name = "Vulkan";

/// Ports `GGML_VK_MAX_DEVICES` (ggml-vulkan.h:11 @b1b770d90).
pub const max_devices = 16;

/// The PCI vendor ids the C++ defines for itself. Ports `VK_VENDOR_ID_AMD`,
/// `VK_VENDOR_ID_APPLE`, `VK_VENDOR_ID_INTEL`, `VK_VENDOR_ID_NVIDIA`,
/// `VK_VENDOR_ID_QUALCOMM` and `VK_VENDOR_ID_SAMSUNG`
/// (ggml-vulkan-types.h:181, 183, 185, 187, 189, 191 @b1b770d90).
pub const vendor_id = struct {
    pub const amd = 0x1002;
    pub const apple = 0x106b;
    pub const intel = 0x8086;
    pub const nvidia = 0x10de;
    pub const qualcomm = 0x5143;
    pub const samsung = 0x144d;
};

/// Ports `ggml_vk_external_memory_host_usable` (ggml-vulkan.cpp:4211 @b1b770d90).
pub fn externalMemoryHostUsable(driver_id: vulkan_c.VkDriverId) bool {
    // Disable external_memory_host until
    // https://github.com/KhronosGroup/MoltenVK/pull/2622 is available in the
    // Vulkan SDK.
    return driver_id != vulkan_c.VK_DRIVER_ID_MOLTENVK;
}

/// Ports `ggml_vk_host_buffer_import_usable` (ggml-vulkan.cpp:4217 @b1b770d90).
pub fn hostBufferImportUsable(vendor: u32) bool {
    // amdgpu re-faults all pages of every imported host allocation on each submission
    return vendor != vendor_id.amd;
}

/// Ports `ggml_vk_has_rebar` (ggml-vulkan.cpp:4222 @b1b770d90): whether a
/// memory type that is device-local, host-visible and host-coherent lives on
/// the heap of the first device-local type, i.e. the whole of VRAM is
/// mappable (resizable BAR).
pub fn hasRebar(physical_device: vulkan_c.VkPhysicalDevice) bool {
    var mem_props: vulkan_c.VkPhysicalDeviceMemoryProperties = undefined;
    dispatch.vkGetPhysicalDeviceMemoryProperties.?(physical_device, &mem_props);
    const hvv: vulkan_c.VkMemoryPropertyFlags = vulkan_c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        vulkan_c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        vulkan_c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    var vram_heap: u32 = std.math.maxInt(u32);
    for (mem_props.memoryTypes[0..mem_props.memoryTypeCount]) |t| {
        if (vram_heap == std.math.maxInt(u32) and (t.propertyFlags & vulkan_c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
            vram_heap = t.heapIndex;
        }
        if ((t.propertyFlags & hvv) == hvv and t.heapIndex == vram_heap) return true;
    }
    return false;
}

/// The loader's one entry point every other command is found through. The
/// C++ links it directly (`VULKAN_HPP_DISPATCH_LOADER_DYNAMIC` still names
/// `vkGetInstanceProcAddr` at `init`).
pub extern fn vkGetInstanceProcAddr(instance: vulkan_c.VkInstance, p_name: [*:0]const u8) vulkan_c.PFN_vkVoidFunction;

/// The four commands the C++ calls by their exported names rather than
/// through the dispatcher (measured with `nm` on the reference).
pub extern fn vkGetPhysicalDeviceFeatures2(physical_device: vulkan_c.VkPhysicalDevice, features: *vulkan_c.VkPhysicalDeviceFeatures2) void;

/// Commands that need no instance: `vulkan.hpp`'s `init(vkGetInstanceProcAddr)`
/// loads these with a null instance.
const global_commands = [_][:0]const u8{
    "vkCreateInstance",
    "vkEnumerateInstanceVersion",
    "vkEnumerateInstanceExtensionProperties",
    "vkEnumerateInstanceLayerProperties",
};

/// Ports `vulkan.hpp`'s `DispatchLoaderDynamic`, the C++'s
/// `ggml_vk_default_dispatcher_instance` (ggml-vulkan.cpp:5464 @b1b770d90):
/// one function pointer per command the host layer calls, each field named
/// after its command and typed by its `PFN_`.
///
/// `loadGlobal` is `init(vkGetInstanceProcAddr)`; `loadInstance` is
/// `init(instance)`, which loads every command -- device-level ones
/// included -- through `vkGetInstanceProcAddr`, so a device command calls
/// the loader's trampoline, as the C++'s does. A field the driver does not
/// provide stays null, and calling it is the C++'s null-pointer call.
pub const Dispatch = struct {
    vkCreateInstance: vulkan_c.PFN_vkCreateInstance = null,
    vkEnumerateInstanceVersion: vulkan_c.PFN_vkEnumerateInstanceVersion = null,
    vkEnumerateInstanceExtensionProperties: vulkan_c.PFN_vkEnumerateInstanceExtensionProperties = null,
    vkEnumerateInstanceLayerProperties: vulkan_c.PFN_vkEnumerateInstanceLayerProperties = null,

    vkEnumeratePhysicalDevices: vulkan_c.PFN_vkEnumeratePhysicalDevices = null,
    vkEnumerateDeviceExtensionProperties: vulkan_c.PFN_vkEnumerateDeviceExtensionProperties = null,
    vkGetPhysicalDeviceProperties: vulkan_c.PFN_vkGetPhysicalDeviceProperties = null,
    vkGetPhysicalDeviceProperties2: vulkan_c.PFN_vkGetPhysicalDeviceProperties2 = null,
    vkGetPhysicalDeviceMemoryProperties2: vulkan_c.PFN_vkGetPhysicalDeviceMemoryProperties2 = null,
    vkGetPhysicalDeviceQueueFamilyProperties: vulkan_c.PFN_vkGetPhysicalDeviceQueueFamilyProperties = null,
    vkGetPhysicalDeviceFeatures: vulkan_c.PFN_vkGetPhysicalDeviceFeatures = null,

    vkCreateDevice: vulkan_c.PFN_vkCreateDevice = null,
    vkDestroyDevice: vulkan_c.PFN_vkDestroyDevice = null,
    vkGetDeviceQueue2: vulkan_c.PFN_vkGetDeviceQueue2 = null,
    vkCreateCommandPool: vulkan_c.PFN_vkCreateCommandPool = null,
    vkDestroyCommandPool: vulkan_c.PFN_vkDestroyCommandPool = null,
    vkCreateDescriptorSetLayout: vulkan_c.PFN_vkCreateDescriptorSetLayout = null,
    vkDestroyDescriptorSetLayout: vulkan_c.PFN_vkDestroyDescriptorSetLayout = null,
    vkCreateFence: vulkan_c.PFN_vkCreateFence = null,
    vkDestroyFence: vulkan_c.PFN_vkDestroyFence = null,
    vkCreateShaderModule: vulkan_c.PFN_vkCreateShaderModule = null,
    vkDestroyShaderModule: vulkan_c.PFN_vkDestroyShaderModule = null,
    vkCreatePipelineLayout: vulkan_c.PFN_vkCreatePipelineLayout = null,
    vkDestroyPipelineLayout: vulkan_c.PFN_vkDestroyPipelineLayout = null,
    vkCreateComputePipelines: vulkan_c.PFN_vkCreateComputePipelines = null,
    vkDestroyPipeline: vulkan_c.PFN_vkDestroyPipeline = null,
    vkGetPipelineExecutableStatisticsKHR: vulkan_c.PFN_vkGetPipelineExecutableStatisticsKHR = null,
    vkCreateBuffer: vulkan_c.PFN_vkCreateBuffer = null,
    vkDestroyBuffer: vulkan_c.PFN_vkDestroyBuffer = null,
    vkGetBufferMemoryRequirements: vulkan_c.PFN_vkGetBufferMemoryRequirements = null,
    vkGetPhysicalDeviceMemoryProperties: vulkan_c.PFN_vkGetPhysicalDeviceMemoryProperties = null,
    vkAllocateMemory: vulkan_c.PFN_vkAllocateMemory = null,
    vkFreeMemory: vulkan_c.PFN_vkFreeMemory = null,
    vkMapMemory: vulkan_c.PFN_vkMapMemory = null,
    vkBindBufferMemory: vulkan_c.PFN_vkBindBufferMemory = null,
    vkGetBufferDeviceAddress: vulkan_c.PFN_vkGetBufferDeviceAddress = null,
    vkGetMemoryHostPointerPropertiesEXT: vulkan_c.PFN_vkGetMemoryHostPointerPropertiesEXT = null,
    vkAllocateCommandBuffers: vulkan_c.PFN_vkAllocateCommandBuffers = null,
    vkFreeCommandBuffers: vulkan_c.PFN_vkFreeCommandBuffers = null,
    vkResetCommandBuffer: vulkan_c.PFN_vkResetCommandBuffer = null,
    vkBeginCommandBuffer: vulkan_c.PFN_vkBeginCommandBuffer = null,
    vkEndCommandBuffer: vulkan_c.PFN_vkEndCommandBuffer = null,
    vkQueueSubmit: vulkan_c.PFN_vkQueueSubmit = null,
    vkWaitForFences: vulkan_c.PFN_vkWaitForFences = null,
    vkResetFences: vulkan_c.PFN_vkResetFences = null,
    vkGetFenceStatus: vulkan_c.PFN_vkGetFenceStatus = null,
    vkResetCommandPool: vulkan_c.PFN_vkResetCommandPool = null,
    vkCmdPipelineBarrier: vulkan_c.PFN_vkCmdPipelineBarrier = null,
    vkCmdCopyBuffer: vulkan_c.PFN_vkCmdCopyBuffer = null,
    vkCmdFillBuffer: vulkan_c.PFN_vkCmdFillBuffer = null,
    vkCmdResetEvent: vulkan_c.PFN_vkCmdResetEvent = null,
    vkCmdSetEvent: vulkan_c.PFN_vkCmdSetEvent = null,
    vkCmdWaitEvents: vulkan_c.PFN_vkCmdWaitEvents = null,
    vkCreateSemaphore: vulkan_c.PFN_vkCreateSemaphore = null,
    vkDestroySemaphore: vulkan_c.PFN_vkDestroySemaphore = null,
    vkWaitSemaphores: vulkan_c.PFN_vkWaitSemaphores = null,
    vkSignalSemaphore: vulkan_c.PFN_vkSignalSemaphore = null,
    vkGetSemaphoreCounterValue: vulkan_c.PFN_vkGetSemaphoreCounterValue = null,
    vkGetDeviceProcAddr: vulkan_c.PFN_vkGetDeviceProcAddr = null,
    vkGetPhysicalDeviceExternalSemaphoreProperties: vulkan_c.PFN_vkGetPhysicalDeviceExternalSemaphoreProperties = null,
    vkCreateEvent: vulkan_c.PFN_vkCreateEvent = null,
    vkDestroyEvent: vulkan_c.PFN_vkDestroyEvent = null,
    vkGetEventStatus: vulkan_c.PFN_vkGetEventStatus = null,
    vkResetEvent: vulkan_c.PFN_vkResetEvent = null,
    vkUpdateDescriptorSets: vulkan_c.PFN_vkUpdateDescriptorSets = null,
    vkCmdPushConstants: vulkan_c.PFN_vkCmdPushConstants = null,
    vkCmdBindPipeline: vulkan_c.PFN_vkCmdBindPipeline = null,
    vkCmdBindDescriptorSets: vulkan_c.PFN_vkCmdBindDescriptorSets = null,
    vkCmdDispatch: vulkan_c.PFN_vkCmdDispatch = null,
    vkCreateDescriptorPool: vulkan_c.PFN_vkCreateDescriptorPool = null,
    vkDestroyDescriptorPool: vulkan_c.PFN_vkDestroyDescriptorPool = null,
    vkAllocateDescriptorSets: vulkan_c.PFN_vkAllocateDescriptorSets = null,
    vkResetDescriptorPool: vulkan_c.PFN_vkResetDescriptorPool = null,
    vkCreateQueryPool: vulkan_c.PFN_vkCreateQueryPool = null,
    vkDestroyQueryPool: vulkan_c.PFN_vkDestroyQueryPool = null,
    vkCmdResetQueryPool: vulkan_c.PFN_vkCmdResetQueryPool = null,
    vkResetQueryPool: vulkan_c.PFN_vkResetQueryPool = null,
    vkCmdWriteTimestamp: vulkan_c.PFN_vkCmdWriteTimestamp = null,
    vkGetQueryPoolResults: vulkan_c.PFN_vkGetQueryPoolResults = null,

    /// Loads the global commands. Ports `init(vkGetInstanceProcAddr)`.
    pub fn loadGlobal(self: *Dispatch) void {
        inline for (global_commands) |cmd| {
            @field(self, cmd) = @ptrCast(vkGetInstanceProcAddr(null, cmd));
        }
    }

    /// Loads every command through `instance`. Ports `init(instance)`.
    ///
    /// Parameters:
    /// - `instance`: the instance the commands are looked up through.
    pub fn loadInstance(self: *Dispatch, instance: vulkan_c.VkInstance) void {
        inline for (@typeInfo(Dispatch).@"struct".field_names) |field| {
            @field(self, field) = @ptrCast(vkGetInstanceProcAddr(instance, field));
        }
    }
};

/// The process's one dispatcher, as the C++'s is.
pub var dispatch: Dispatch = .{};

/// A `VkResult` other than success, where `vulkan.hpp` would throw
/// `vk::SystemError`. The C++'s catch sites decide what happens next; each
/// call site here does what its catch does.
pub const VkError = error{VulkanError};

/// The last failed result and the command that returned it, for the
/// message a catch site prints.
pub threadlocal var last_error: struct { result: vulkan_c.VkResult = vulkan_c.VK_SUCCESS, what: [*:0]const u8 = "" } = .{};

/// `vulkan.hpp`'s `resultCheck`: anything but `VK_SUCCESS` throws.
///
/// Parameters:
/// - `result`: what the command returned.
/// - `what`: the command's `vulkan.hpp` name, as the exception's message
///   names it.
///
/// Return: nothing on success; `error.VulkanError` otherwise, with
/// `last_error` set.
pub fn check(result: vulkan_c.VkResult, what: [*:0]const u8) VkError!void {
    if (result == vulkan_c.VK_SUCCESS) return;
    last_error = .{ .result = result, .what = what };
    return error.VulkanError;
}

/// What a `vk::SystemError` nothing catches comes to: the C++ terminates
/// with the exception's message; the port prints it and aborts.
///
/// Parameters:
/// - `result`: what the command returned.
/// - `what`: the command's `vulkan.hpp` name.
pub fn checkOrAbort(result: vulkan_c.VkResult, what: [*:0]const u8) void {
    check(result, what) catch abortLastError();
}

/// Aborts on `last_error`, printing what `vk::SystemError::what()` would.
pub fn abortLastError() noreturn {
    ggml.impl.fprintfStderr("ggml_vulkan: %s: %s\n", .{ last_error.what, resultString(last_error.result).ptr });
    ggml.impl.abort("ggml_vulkan: uncaught vk::SystemError");
}

/// `vk::to_string(vk::Result)`: the enumerant's name without its `VK_`
/// prefix, in `vulkan.hpp`'s camel case, or `invalid ( 0x... )`.
///
/// Return: the name; an unknown value's text lives in a thread-local buffer
/// until the next call.
pub fn resultString(result: vulkan_c.VkResult) [:0]const u8 {
    return switch (result) {
        vulkan_c.VK_SUCCESS => "Success",
        vulkan_c.VK_NOT_READY => "NotReady",
        vulkan_c.VK_TIMEOUT => "Timeout",
        vulkan_c.VK_EVENT_SET => "EventSet",
        vulkan_c.VK_EVENT_RESET => "EventReset",
        vulkan_c.VK_INCOMPLETE => "Incomplete",
        vulkan_c.VK_ERROR_OUT_OF_HOST_MEMORY => "ErrorOutOfHostMemory",
        vulkan_c.VK_ERROR_OUT_OF_DEVICE_MEMORY => "ErrorOutOfDeviceMemory",
        vulkan_c.VK_ERROR_INITIALIZATION_FAILED => "ErrorInitializationFailed",
        vulkan_c.VK_ERROR_DEVICE_LOST => "ErrorDeviceLost",
        vulkan_c.VK_ERROR_MEMORY_MAP_FAILED => "ErrorMemoryMapFailed",
        vulkan_c.VK_ERROR_LAYER_NOT_PRESENT => "ErrorLayerNotPresent",
        vulkan_c.VK_ERROR_EXTENSION_NOT_PRESENT => "ErrorExtensionNotPresent",
        vulkan_c.VK_ERROR_FEATURE_NOT_PRESENT => "ErrorFeatureNotPresent",
        vulkan_c.VK_ERROR_INCOMPATIBLE_DRIVER => "ErrorIncompatibleDriver",
        vulkan_c.VK_ERROR_TOO_MANY_OBJECTS => "ErrorTooManyObjects",
        vulkan_c.VK_ERROR_FORMAT_NOT_SUPPORTED => "ErrorFormatNotSupported",
        vulkan_c.VK_ERROR_FRAGMENTED_POOL => "ErrorFragmentedPool",
        vulkan_c.VK_ERROR_UNKNOWN => "ErrorUnknown",
        vulkan_c.VK_ERROR_OUT_OF_POOL_MEMORY => "ErrorOutOfPoolMemory",
        vulkan_c.VK_ERROR_INVALID_EXTERNAL_HANDLE => "ErrorInvalidExternalHandle",
        vulkan_c.VK_ERROR_FRAGMENTATION => "ErrorFragmentation",
        vulkan_c.VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS => "ErrorInvalidOpaqueCaptureAddress",
        vulkan_c.VK_PIPELINE_COMPILE_REQUIRED => "PipelineCompileRequired",
        vulkan_c.VK_ERROR_VALIDATION_FAILED => "ErrorValidationFailed",
        vulkan_c.VK_ERROR_NOT_PERMITTED => "ErrorNotPermitted",
        else => blk: {
            const S = struct {
                threadlocal var buf: [32]u8 = undefined;
            };
            break :blk std.fmt.bufPrintSentinel(&S.buf, "invalid ( 0x{x} )", .{@as(u32, @bitCast(result))}, 0) catch unreachable;
        },
    };
}

/// `vulkan.hpp`'s two-call enumeration: ask for the count, size the vector,
/// fetch, and start again while the command returns `VK_INCOMPLETE` --
/// which it does when the set grew between the calls.
///
/// Parameters:
/// - `T`: the element type.
/// - `what`: the `vulkan.hpp` name, for `check`.
/// - `f`: the command, bound to its leading arguments.
/// - `args`: those leading arguments.
///
/// Return: the elements, owned by the caller and freed with `allocator`.
pub fn enumerate(comptime T: type, what: [*:0]const u8, f: anytype, args: anytype) VkError![]T {
    while (true) {
        var count: u32 = 0;
        try check(@call(.auto, f, args ++ .{ &count, @as([*c]T, null) }), what);
        const items = std.heap.c_allocator.alloc(T, count) catch ggml.impl.abort("ggml_vulkan: out of memory");
        const result = @call(.auto, f, args ++ .{ &count, items.ptr });
        if (result == vulkan_c.VK_INCOMPLETE) {
            std.heap.c_allocator.free(items);
            continue;
        }
        check(result, what) catch |err| {
            std.heap.c_allocator.free(items);
            return err;
        };
        // The second call may report fewer than the first.
        return std.heap.c_allocator.realloc(items, count) catch items[0..count];
    }
}

/// `enumerate` for an output structure: `vulkan.hpp` default-constructs
/// each element, so each arrives zeroed with its `sType` set.
pub fn enumerateWith(comptime T: type, s_type: vulkan_c.VkStructureType, what: [*:0]const u8, f: anytype, args: anytype) VkError![]T {
    while (true) {
        var count: u32 = 0;
        try check(@call(.auto, f, args ++ .{ &count, @as([*c]T, null) }), what);
        const items = std.heap.c_allocator.alloc(T, count) catch ggml.impl.abort("ggml_vulkan: out of memory");
        for (items) |*item| item.* = structure(T, s_type);
        const result = @call(.auto, f, args ++ .{ &count, items.ptr });
        if (result == vulkan_c.VK_INCOMPLETE) {
            std.heap.c_allocator.free(items);
            continue;
        }
        check(result, what) catch |err| {
            std.heap.c_allocator.free(items);
            return err;
        };
        return std.heap.c_allocator.realloc(items, count) catch items[0..count];
    }
}

/// `vk::Instance::enumeratePhysicalDevices`.
pub fn enumeratePhysicalDevices(instance: vulkan_c.VkInstance) VkError![]vulkan_c.VkPhysicalDevice {
    return enumerate(vulkan_c.VkPhysicalDevice, "vk::Instance::enumeratePhysicalDevices", dispatch.vkEnumeratePhysicalDevices.?, .{instance});
}

/// `vk::PhysicalDevice::enumerateDeviceExtensionProperties`, for no layer.
pub fn enumerateDeviceExtensionProperties(dev: vulkan_c.VkPhysicalDevice) VkError![]vulkan_c.VkExtensionProperties {
    return enumerate(vulkan_c.VkExtensionProperties, "vk::PhysicalDevice::enumerateDeviceExtensionProperties", dispatch.vkEnumerateDeviceExtensionProperties.?, .{ dev, @as([*c]const u8, null) });
}

/// `vk::enumerateInstanceExtensionProperties`, for `layer` or none.
pub fn enumerateInstanceExtensionProperties(layer: ?[*:0]const u8) VkError![]vulkan_c.VkExtensionProperties {
    return enumerate(vulkan_c.VkExtensionProperties, "vk::enumerateInstanceExtensionProperties", dispatch.vkEnumerateInstanceExtensionProperties.?, .{@as([*c]const u8, layer)});
}

/// `vk::enumerateInstanceLayerProperties`.
pub fn enumerateInstanceLayerProperties() VkError![]vulkan_c.VkLayerProperties {
    return enumerate(vulkan_c.VkLayerProperties, "vk::enumerateInstanceLayerProperties", dispatch.vkEnumerateInstanceLayerProperties.?, .{});
}

/// Whether `props` lists the extension `ext_name`: the C++'s
/// `strcmp(name, properties.extensionName) == 0` loop.
pub fn hasExtension(props: []const vulkan_c.VkExtensionProperties, ext_name: []const u8) bool {
    for (props) |p| {
        if (std.mem.eql(u8, std.mem.sliceTo(&p.extensionName, 0), ext_name)) return true;
    }
    return false;
}

/// `vk::PhysicalDevice::getProperties`.
pub fn getProperties(dev: vulkan_c.VkPhysicalDevice) vulkan_c.VkPhysicalDeviceProperties {
    var props: vulkan_c.VkPhysicalDeviceProperties = undefined;
    dispatch.vkGetPhysicalDeviceProperties.?(dev, &props);
    return props;
}

/// A `vulkan.hpp` structure value: zeroed, with its `sType` set, as the
/// C++ wrapper's default constructor leaves it.
pub fn structure(comptime T: type, s_type: vulkan_c.VkStructureType) T {
    var s = std.mem.zeroes(T);
    s.sType = s_type;
    return s;
}

/// Ports `YIELD` (ggml-vulkan-types.h:82 @b1b770d90), the clang arm on ARM,
/// `__yield()`; on x86 the clang arm is `_mm_pause()` -- `zig c++` is clang
/// on Windows too, so `YieldProcessor()` is never compiled -- and elsewhere
/// it is empty.
pub inline fn yield() void {
    switch (builtin.cpu.arch) {
        .x86_64, .x86 => asm volatile ("pause"),
        .aarch64, .aarch64_be, .arm, .armeb, .thumb, .thumbeb => asm volatile ("yield"),
        else => {},
    }
}

/// Whether the environment variable `var_name` is set: the C++'s
/// `getenv(...) != nullptr`.
pub fn envSet(comptime var_name: [:0]const u8) bool {
    return stdlib.env.getenv(var_name) != null;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "host memory is imported everywhere but MoltenVK and amdgpu" {
    try std.testing.expect(externalMemoryHostUsable(vulkan_c.VK_DRIVER_ID_NVIDIA_PROPRIETARY));
    try std.testing.expect(!externalMemoryHostUsable(vulkan_c.VK_DRIVER_ID_MOLTENVK));
    try std.testing.expect(hostBufferImportUsable(vendor_id.nvidia));
    try std.testing.expect(hostBufferImportUsable(vendor_id.intel));
    try std.testing.expect(!hostBufferImportUsable(vendor_id.amd));
}
