//! What every file of the Vulkan host layer shares: the C API, the dynamic
//! dispatch table, `vulkan.hpp`'s error and enumeration conventions, and the
//! backend's constants.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` and
//! `llama.cpp/ggml/include/ggml-vulkan.h` at v0.3.0 (`c1d0e7a00`). The C++
//! is written against `vulkan.hpp`, a header-only C++ wrapper over the C API
//! in `vulkan_core.h`; what it adds that changes behaviour is reproduced here
//! -- the dispatcher, exceptions on a failed `VkResult`, and the
//! `VK_INCOMPLETE` retry of every `enumerate*` -- and nothing else.

const std = @import("std");
const impl = @import("../impl.zig");

/// The Vulkan C API, `vulkan_core.h` at the pinned SDK (PLAN.md, Decision
/// 59), translated with `VK_NO_PROTOTYPES`: every command is reached through
/// `dispatch`, as `vulkan.hpp`'s dynamic dispatcher reaches it.
pub const vk = @import("vulkan_c");

/// The allocator for host-side bookkeeping, libc's heap as everywhere else
/// in ggml.
pub const allocator = std.heap.c_allocator;

/// Ports `GGML_VK_NAME` (ggml-vulkan.h:10 @c1d0e7a00).
pub const name = "Vulkan";

/// Ports `GGML_VK_MAX_DEVICES` (ggml-vulkan.h:11 @c1d0e7a00).
pub const max_devices = 16;

/// The PCI vendor ids the C++ defines for itself. Ports `VK_VENDOR_ID_AMD`,
/// `VK_VENDOR_ID_APPLE`, `VK_VENDOR_ID_INTEL`, `VK_VENDOR_ID_NVIDIA` and
/// `VK_VENDOR_ID_QUALCOMM` (ggml-vulkan.cpp:181, 182, 183, 184, 185 @c1d0e7a00).
pub const vendor_id = struct {
    pub const amd = 0x1002;
    pub const apple = 0x106b;
    pub const intel = 0x8086;
    pub const nvidia = 0x10de;
    pub const qualcomm = 0x5143;
};

/// The loader's one entry point every other command is found through. The
/// C++ links it directly (`VULKAN_HPP_DISPATCH_LOADER_DYNAMIC` still names
/// `vkGetInstanceProcAddr` at `init`).
pub extern fn vkGetInstanceProcAddr(instance: vk.VkInstance, p_name: [*:0]const u8) vk.PFN_vkVoidFunction;

/// The four commands the C++ calls by their exported names rather than
/// through the dispatcher (`NOTES.md`, "Stage 7: the measurement").
pub extern fn vkGetPhysicalDeviceFeatures2(physical_device: vk.VkPhysicalDevice, features: *vk.VkPhysicalDeviceFeatures2) void;

/// Commands that need no instance: `vulkan.hpp`'s `init(vkGetInstanceProcAddr)`
/// loads these with a null instance.
const global_commands = [_][:0]const u8{
    "vkCreateInstance",
    "vkEnumerateInstanceVersion",
    "vkEnumerateInstanceExtensionProperties",
    "vkEnumerateInstanceLayerProperties",
};

/// Ports `vulkan.hpp`'s `DispatchLoaderDynamic`, the C++'s
/// `ggml_vk_default_dispatcher_instance` (ggml-vulkan.cpp:7347 @c1d0e7a00):
/// one function pointer per command the host layer calls, each field named
/// after its command and typed by its `PFN_`.
///
/// `loadGlobal` is `init(vkGetInstanceProcAddr)`; `loadInstance` is
/// `init(instance)`, which loads every command -- device-level ones
/// included -- through `vkGetInstanceProcAddr`, so a device command calls
/// the loader's trampoline, as the C++'s does. A field the driver does not
/// provide stays null, and calling it is the C++'s null-pointer call.
pub const Dispatch = struct {
    vkCreateInstance: vk.PFN_vkCreateInstance = null,
    vkEnumerateInstanceVersion: vk.PFN_vkEnumerateInstanceVersion = null,
    vkEnumerateInstanceExtensionProperties: vk.PFN_vkEnumerateInstanceExtensionProperties = null,
    vkEnumerateInstanceLayerProperties: vk.PFN_vkEnumerateInstanceLayerProperties = null,

    vkEnumeratePhysicalDevices: vk.PFN_vkEnumeratePhysicalDevices = null,
    vkEnumerateDeviceExtensionProperties: vk.PFN_vkEnumerateDeviceExtensionProperties = null,
    vkGetPhysicalDeviceProperties: vk.PFN_vkGetPhysicalDeviceProperties = null,
    vkGetPhysicalDeviceProperties2: vk.PFN_vkGetPhysicalDeviceProperties2 = null,
    vkGetPhysicalDeviceMemoryProperties2: vk.PFN_vkGetPhysicalDeviceMemoryProperties2 = null,
    vkGetPhysicalDeviceQueueFamilyProperties: vk.PFN_vkGetPhysicalDeviceQueueFamilyProperties = null,
    vkGetPhysicalDeviceFeatures: vk.PFN_vkGetPhysicalDeviceFeatures = null,

    vkCreateDevice: vk.PFN_vkCreateDevice = null,
    vkDestroyDevice: vk.PFN_vkDestroyDevice = null,
    vkGetDeviceQueue2: vk.PFN_vkGetDeviceQueue2 = null,
    vkCreateCommandPool: vk.PFN_vkCreateCommandPool = null,
    vkDestroyCommandPool: vk.PFN_vkDestroyCommandPool = null,
    vkCreateDescriptorSetLayout: vk.PFN_vkCreateDescriptorSetLayout = null,
    vkDestroyDescriptorSetLayout: vk.PFN_vkDestroyDescriptorSetLayout = null,
    vkCreateFence: vk.PFN_vkCreateFence = null,
    vkDestroyFence: vk.PFN_vkDestroyFence = null,
    vkCreateShaderModule: vk.PFN_vkCreateShaderModule = null,
    vkDestroyShaderModule: vk.PFN_vkDestroyShaderModule = null,
    vkCreatePipelineLayout: vk.PFN_vkCreatePipelineLayout = null,
    vkDestroyPipelineLayout: vk.PFN_vkDestroyPipelineLayout = null,
    vkCreateComputePipelines: vk.PFN_vkCreateComputePipelines = null,
    vkDestroyPipeline: vk.PFN_vkDestroyPipeline = null,
    vkGetPipelineExecutableStatisticsKHR: vk.PFN_vkGetPipelineExecutableStatisticsKHR = null,
    vkCreateBuffer: vk.PFN_vkCreateBuffer = null,
    vkDestroyBuffer: vk.PFN_vkDestroyBuffer = null,
    vkGetBufferMemoryRequirements: vk.PFN_vkGetBufferMemoryRequirements = null,
    vkGetPhysicalDeviceMemoryProperties: vk.PFN_vkGetPhysicalDeviceMemoryProperties = null,
    vkAllocateMemory: vk.PFN_vkAllocateMemory = null,
    vkFreeMemory: vk.PFN_vkFreeMemory = null,
    vkMapMemory: vk.PFN_vkMapMemory = null,
    vkBindBufferMemory: vk.PFN_vkBindBufferMemory = null,
    vkGetBufferDeviceAddress: vk.PFN_vkGetBufferDeviceAddress = null,
    vkGetMemoryHostPointerPropertiesEXT: vk.PFN_vkGetMemoryHostPointerPropertiesEXT = null,
    vkAllocateCommandBuffers: vk.PFN_vkAllocateCommandBuffers = null,
    vkFreeCommandBuffers: vk.PFN_vkFreeCommandBuffers = null,
    vkResetCommandBuffer: vk.PFN_vkResetCommandBuffer = null,
    vkBeginCommandBuffer: vk.PFN_vkBeginCommandBuffer = null,
    vkEndCommandBuffer: vk.PFN_vkEndCommandBuffer = null,
    vkQueueSubmit: vk.PFN_vkQueueSubmit = null,
    vkWaitForFences: vk.PFN_vkWaitForFences = null,
    vkResetFences: vk.PFN_vkResetFences = null,
    vkGetFenceStatus: vk.PFN_vkGetFenceStatus = null,
    vkResetCommandPool: vk.PFN_vkResetCommandPool = null,
    vkCmdPipelineBarrier: vk.PFN_vkCmdPipelineBarrier = null,
    vkCmdCopyBuffer: vk.PFN_vkCmdCopyBuffer = null,
    vkCmdFillBuffer: vk.PFN_vkCmdFillBuffer = null,
    vkCmdResetEvent: vk.PFN_vkCmdResetEvent = null,
    vkCmdSetEvent: vk.PFN_vkCmdSetEvent = null,
    vkCmdWaitEvents: vk.PFN_vkCmdWaitEvents = null,
    vkCreateSemaphore: vk.PFN_vkCreateSemaphore = null,
    vkDestroySemaphore: vk.PFN_vkDestroySemaphore = null,
    vkWaitSemaphores: vk.PFN_vkWaitSemaphores = null,
    vkCreateEvent: vk.PFN_vkCreateEvent = null,
    vkDestroyEvent: vk.PFN_vkDestroyEvent = null,
    vkGetEventStatus: vk.PFN_vkGetEventStatus = null,
    vkResetEvent: vk.PFN_vkResetEvent = null,
    vkUpdateDescriptorSets: vk.PFN_vkUpdateDescriptorSets = null,
    vkCmdPushConstants: vk.PFN_vkCmdPushConstants = null,
    vkCmdBindPipeline: vk.PFN_vkCmdBindPipeline = null,
    vkCmdBindDescriptorSets: vk.PFN_vkCmdBindDescriptorSets = null,
    vkCmdDispatch: vk.PFN_vkCmdDispatch = null,
    vkCreateDescriptorPool: vk.PFN_vkCreateDescriptorPool = null,
    vkDestroyDescriptorPool: vk.PFN_vkDestroyDescriptorPool = null,
    vkAllocateDescriptorSets: vk.PFN_vkAllocateDescriptorSets = null,
    vkResetDescriptorPool: vk.PFN_vkResetDescriptorPool = null,
    vkCreateQueryPool: vk.PFN_vkCreateQueryPool = null,
    vkDestroyQueryPool: vk.PFN_vkDestroyQueryPool = null,
    vkCmdResetQueryPool: vk.PFN_vkCmdResetQueryPool = null,
    vkResetQueryPool: vk.PFN_vkResetQueryPool = null,
    vkCmdWriteTimestamp: vk.PFN_vkCmdWriteTimestamp = null,
    vkGetQueryPoolResults: vk.PFN_vkGetQueryPoolResults = null,

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
    pub fn loadInstance(self: *Dispatch, instance: vk.VkInstance) void {
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
pub threadlocal var last_error: struct { result: vk.VkResult = vk.VK_SUCCESS, what: [*:0]const u8 = "" } = .{};

/// `vulkan.hpp`'s `resultCheck`: anything but `VK_SUCCESS` throws.
///
/// Parameters:
/// - `result`: what the command returned.
/// - `what`: the command's `vulkan.hpp` name, as the exception's message
///   names it.
///
/// Return: nothing on success; `error.VulkanError` otherwise, with
/// `last_error` set.
pub fn check(result: vk.VkResult, what: [*:0]const u8) VkError!void {
    if (result == vk.VK_SUCCESS) return;
    last_error = .{ .result = result, .what = what };
    return error.VulkanError;
}

/// What a `vk::SystemError` nothing catches comes to: the C++ terminates
/// with the exception's message; the port prints it and aborts.
///
/// Parameters:
/// - `result`: what the command returned.
/// - `what`: the command's `vulkan.hpp` name.
pub fn checkOrAbort(result: vk.VkResult, what: [*:0]const u8) void {
    check(result, what) catch abortLastError();
}

/// Aborts on `last_error`, printing what `vk::SystemError::what()` would.
pub fn abortLastError() noreturn {
    impl.fprintfStderr("ggml_vulkan: %s: %s\n", .{ last_error.what, resultString(last_error.result).ptr });
    impl.abort("ggml_vulkan: uncaught vk::SystemError");
}

/// `vk::to_string(vk::Result)`: the enumerant's name without its `VK_`
/// prefix, in `vulkan.hpp`'s camel case, or `invalid ( 0x... )`.
///
/// Return: the name; an unknown value's text lives in a thread-local buffer
/// until the next call.
pub fn resultString(result: vk.VkResult) [:0]const u8 {
    return switch (result) {
        vk.VK_SUCCESS => "Success",
        vk.VK_NOT_READY => "NotReady",
        vk.VK_TIMEOUT => "Timeout",
        vk.VK_EVENT_SET => "EventSet",
        vk.VK_EVENT_RESET => "EventReset",
        vk.VK_INCOMPLETE => "Incomplete",
        vk.VK_ERROR_OUT_OF_HOST_MEMORY => "ErrorOutOfHostMemory",
        vk.VK_ERROR_OUT_OF_DEVICE_MEMORY => "ErrorOutOfDeviceMemory",
        vk.VK_ERROR_INITIALIZATION_FAILED => "ErrorInitializationFailed",
        vk.VK_ERROR_DEVICE_LOST => "ErrorDeviceLost",
        vk.VK_ERROR_MEMORY_MAP_FAILED => "ErrorMemoryMapFailed",
        vk.VK_ERROR_LAYER_NOT_PRESENT => "ErrorLayerNotPresent",
        vk.VK_ERROR_EXTENSION_NOT_PRESENT => "ErrorExtensionNotPresent",
        vk.VK_ERROR_FEATURE_NOT_PRESENT => "ErrorFeatureNotPresent",
        vk.VK_ERROR_INCOMPATIBLE_DRIVER => "ErrorIncompatibleDriver",
        vk.VK_ERROR_TOO_MANY_OBJECTS => "ErrorTooManyObjects",
        vk.VK_ERROR_FORMAT_NOT_SUPPORTED => "ErrorFormatNotSupported",
        vk.VK_ERROR_FRAGMENTED_POOL => "ErrorFragmentedPool",
        vk.VK_ERROR_UNKNOWN => "ErrorUnknown",
        vk.VK_ERROR_OUT_OF_POOL_MEMORY => "ErrorOutOfPoolMemory",
        vk.VK_ERROR_INVALID_EXTERNAL_HANDLE => "ErrorInvalidExternalHandle",
        vk.VK_ERROR_FRAGMENTATION => "ErrorFragmentation",
        vk.VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS => "ErrorInvalidOpaqueCaptureAddress",
        vk.VK_PIPELINE_COMPILE_REQUIRED => "PipelineCompileRequired",
        vk.VK_ERROR_VALIDATION_FAILED => "ErrorValidationFailed",
        vk.VK_ERROR_NOT_PERMITTED => "ErrorNotPermitted",
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
        const items = allocator.alloc(T, count) catch impl.abort("ggml_vulkan: out of memory");
        const result = @call(.auto, f, args ++ .{ &count, items.ptr });
        if (result == vk.VK_INCOMPLETE) {
            allocator.free(items);
            continue;
        }
        check(result, what) catch |err| {
            allocator.free(items);
            return err;
        };
        // The second call may report fewer than the first.
        return allocator.realloc(items, count) catch items[0..count];
    }
}

/// `enumerate` for an output structure: `vulkan.hpp` default-constructs
/// each element, so each arrives zeroed with its `sType` set.
pub fn enumerateWith(comptime T: type, s_type: vk.VkStructureType, what: [*:0]const u8, f: anytype, args: anytype) VkError![]T {
    while (true) {
        var count: u32 = 0;
        try check(@call(.auto, f, args ++ .{ &count, @as([*c]T, null) }), what);
        const items = allocator.alloc(T, count) catch impl.abort("ggml_vulkan: out of memory");
        for (items) |*item| item.* = structure(T, s_type);
        const result = @call(.auto, f, args ++ .{ &count, items.ptr });
        if (result == vk.VK_INCOMPLETE) {
            allocator.free(items);
            continue;
        }
        check(result, what) catch |err| {
            allocator.free(items);
            return err;
        };
        return allocator.realloc(items, count) catch items[0..count];
    }
}

/// `vk::Instance::enumeratePhysicalDevices`.
pub fn enumeratePhysicalDevices(instance: vk.VkInstance) VkError![]vk.VkPhysicalDevice {
    return enumerate(vk.VkPhysicalDevice, "vk::Instance::enumeratePhysicalDevices", dispatch.vkEnumeratePhysicalDevices.?, .{instance});
}

/// `vk::PhysicalDevice::enumerateDeviceExtensionProperties`, for no layer.
pub fn enumerateDeviceExtensionProperties(dev: vk.VkPhysicalDevice) VkError![]vk.VkExtensionProperties {
    return enumerate(vk.VkExtensionProperties, "vk::PhysicalDevice::enumerateDeviceExtensionProperties", dispatch.vkEnumerateDeviceExtensionProperties.?, .{ dev, @as([*c]const u8, null) });
}

/// `vk::enumerateInstanceExtensionProperties`, for `layer` or none.
pub fn enumerateInstanceExtensionProperties(layer: ?[*:0]const u8) VkError![]vk.VkExtensionProperties {
    return enumerate(vk.VkExtensionProperties, "vk::enumerateInstanceExtensionProperties", dispatch.vkEnumerateInstanceExtensionProperties.?, .{@as([*c]const u8, layer)});
}

/// `vk::enumerateInstanceLayerProperties`.
pub fn enumerateInstanceLayerProperties() VkError![]vk.VkLayerProperties {
    return enumerate(vk.VkLayerProperties, "vk::enumerateInstanceLayerProperties", dispatch.vkEnumerateInstanceLayerProperties.?, .{});
}

/// Whether `props` lists the extension `ext_name`: the C++'s
/// `strcmp(name, properties.extensionName) == 0` loop.
pub fn hasExtension(props: []const vk.VkExtensionProperties, ext_name: []const u8) bool {
    for (props) |p| {
        if (std.mem.eql(u8, std.mem.sliceTo(&p.extensionName, 0), ext_name)) return true;
    }
    return false;
}

/// `vk::PhysicalDevice::getProperties`.
pub fn getProperties(dev: vk.VkPhysicalDevice) vk.VkPhysicalDeviceProperties {
    var props: vk.VkPhysicalDeviceProperties = undefined;
    dispatch.vkGetPhysicalDeviceProperties.?(dev, &props);
    return props;
}

/// A `vulkan.hpp` structure value: zeroed, with its `sType` set, as the
/// C++ wrapper's default constructor leaves it.
pub fn structure(comptime T: type, s_type: vk.VkStructureType) T {
    var s = std.mem.zeroes(T);
    s.sType = s_type;
    return s;
}

/// Whether the environment variable `var_name` is set: the C++'s
/// `getenv(...) != nullptr`.
pub fn envSet(comptime var_name: [:0]const u8) bool {
    return std.c.getenv(var_name) != null;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
