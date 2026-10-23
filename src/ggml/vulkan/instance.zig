//! The Vulkan instance and the physical devices the backend exposes: instance
//! creation, device selection and deduplication, the architecture probe, the
//! `ggml_vulkan: N = ...` line, and the per-device queries `ggml-vulkan.h`
//! answers -- count, description, memory, type and PCI id.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # Exceptions
//!
//! The C++ throws from `ggml_vk_instance_init` -- a failed Vulkan call, or a
//! bad `GGML_VK_VISIBLE_DEVICES` index -- and `ggml_backend_vk_reg` catches
//! everything and returns null. `instanceInit` returns an error union
//! instead, and `backend.zig`'s `reg` returns null on it; every other caller
//! lets the C++'s exception escape, which terminates, so they abort.

const std = @import("std");
const config = @import("config");
const impl = @import("../impl.zig");
const common = @import("common.zig");

const vk = common.vk;
const d = &common.dispatch;
const allocator = common.allocator;

comptime {
    // The C++ carries fallback declarations for headers that predate
    // `VK_NV_cooperative_matrix_decode_vector`, `VK_KHR_shader_bfloat16`,
    // `VK_VALVE_shader_mixed_float_dot_product`,
    // `VK_EXT_shader_ocp_microscaling_types` and `VK_EXT_shader_float8`
    // (ggml-vulkan.cpp:26, 99, 116, 131, 146 @c1d0e7a00). The pinned SDK has
    // all of them (PLAN.md, Decision 59), so the port needs none.
    for (.{
        "VK_NV_cooperative_matrix_decode_vector",  "VK_KHR_shader_bfloat16",
        "VK_VALVE_shader_mixed_float_dot_product", "VK_EXT_shader_ocp_microscaling_types",
        "VK_EXT_shader_float8",                    "VK_NV_cooperative_matrix2",
    }) |ext| {
        if (!@hasDecl(vk, ext)) @compileError("the Vulkan headers predate " ++ ext ++ "; build with the pinned SDK (scripts/vk-toolchain)");
    }
}

/// The shader extensions `glslc` accepted, as the C++'s
/// `GGML_VULKAN_*_GLSLC_SUPPORT` macros (build/llamacpp.zig's `Vulkan`).
pub const glslc = struct {
    pub const coopmat = config.vk_coopmat_glslc;
    pub const coopmat2 = config.vk_coopmat2_glslc;
    pub const coopmat2_decode_vector = config.vk_coopmat2_decode_vector_glslc;
    pub const integer_dot = config.vk_integer_dot_glslc;
    pub const bfloat16 = config.vk_bfloat16_glslc;
    pub const float_e2m1 = config.vk_float_e2m1_glslc;
    pub const float_e4m3 = config.vk_float_e4m3_glslc;
};

// -----------------------------------------------------------------------------
// Architecture

/// Ports `vk_device_architecture` (ggml-vulkan.cpp:392 @c1d0e7a00).
pub const Architecture = enum {
    other,
    amd_gcn,
    amd_rdna1,
    amd_rdna2,
    amd_rdna3,
    intel_xe1,
    intel_xe2,
    nvidia_pre_turing,
    nvidia_turing,
};

/// Ports `get_device_architecture` (ggml-vulkan.cpp:404 @c1d0e7a00).
pub fn getDeviceArchitecture(device: vk.VkPhysicalDevice) common.VkError!Architecture {
    const props = common.getProperties(device);

    if (props.vendorID == common.vendor_id.amd) {
        const ext_props = try common.enumerateDeviceExtensionProperties(device);
        defer allocator.free(ext_props);
        const amd_shader_core_properties = common.hasExtension(ext_props, "VK_AMD_shader_core_properties");
        const integer_dot_product = common.hasExtension(ext_props, "VK_KHR_shader_integer_dot_product");
        const subgroup_size_control = common.hasExtension(ext_props, "VK_EXT_subgroup_size_control");
        if (!amd_shader_core_properties or !integer_dot_product or !subgroup_size_control) return .other;

        var props2 = common.structure(vk.VkPhysicalDeviceProperties2, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
        var shader_core_props_amd = common.structure(vk.VkPhysicalDeviceShaderCorePropertiesAMD, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD);
        var integer_dot_props = common.structure(vk.VkPhysicalDeviceShaderIntegerDotProductProperties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES);
        var subgroup_size_control_props = common.structure(vk.VkPhysicalDeviceSubgroupSizeControlProperties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES);
        props2.pNext = &shader_core_props_amd;
        shader_core_props_amd.pNext = &integer_dot_props;
        integer_dot_props.pNext = &subgroup_size_control_props;
        d.vkGetPhysicalDeviceProperties2.?(device, &props2);

        if (subgroup_size_control_props.maxSubgroupSize == 64 and subgroup_size_control_props.minSubgroupSize == 64) return .amd_gcn;
        if (subgroup_size_control_props.maxSubgroupSize == 64 and subgroup_size_control_props.minSubgroupSize == 32) {
            // RDNA
            if (shader_core_props_amd.wavefrontsPerSimd == 20) return .amd_rdna1;
            if (integer_dot_props.integerDotProduct4x8BitPackedMixedSignednessAccelerated != 0) return .amd_rdna3;
            return .amd_rdna2;
        }
    } else if (props.vendorID == common.vendor_id.intel) {
        const ext_props = try common.enumerateDeviceExtensionProperties(device);
        defer allocator.free(ext_props);
        const subgroup_size_control = common.hasExtension(ext_props, "VK_EXT_subgroup_size_control");
        const integer_dot_product = common.hasExtension(ext_props, "VK_KHR_shader_integer_dot_product");
        if (!subgroup_size_control or !integer_dot_product) return .other;

        var props2 = common.structure(vk.VkPhysicalDeviceProperties2, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
        var subgroup_size_control_props = common.structure(vk.VkPhysicalDeviceSubgroupSizeControlProperties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES);
        var integer_dot_props = common.structure(vk.VkPhysicalDeviceShaderIntegerDotProductProperties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES);
        props2.pNext = &subgroup_size_control_props;
        subgroup_size_control_props.pNext = &integer_dot_props;
        d.vkGetPhysicalDeviceProperties2.?(device, &props2);

        if (subgroup_size_control_props.minSubgroupSize == 16) {
            // Xe2 is SIMD16, earlier Xe and Gen SIMD8: the minimum subgroup
            // size is the SIMD width.
            return .intel_xe2;
        } else if (subgroup_size_control_props.minSubgroupSize == 8 and
            integer_dot_product and integer_dot_props.integerDotProduct4x8BitPackedSignedAccelerated != 0)
        {
            return .intel_xe1;
        }
    } else if (props.vendorID == common.vendor_id.nvidia) {
        const ext_props = try common.enumerateDeviceExtensionProperties(device);
        defer allocator.free(ext_props);
        // "Pre-Turing" is the lack of coopmat support.
        const cooperative_matrix = common.hasExtension(ext_props, "VK_KHR_cooperative_matrix");
        const sm_builtins = common.hasExtension(ext_props, "VK_NV_shader_sm_builtins");
        if (!cooperative_matrix) return .nvidia_pre_turing;

        if (sm_builtins) {
            var props2 = common.structure(vk.VkPhysicalDeviceProperties2, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
            var sm_props = common.structure(vk.VkPhysicalDeviceShaderSMBuiltinsPropertiesNV, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV);
            props2.pNext = &sm_props;
            d.vkGetPhysicalDeviceProperties2.?(device, &props2);
            // Turing has 32, later architectures 48.
            if (sm_props.shaderWarpsPerSM == 32) return .nvidia_turing;
        }
    }
    return .other;
}

/// One entry of `GpuPipelineConfig` (ggml-vulkan.cpp:4055 @c1d0e7a00): a
/// pipeline name and its subgroup size.
const PipelineSubgroupSize = struct { []const u8, u32 };

/// Ports `rdna1_pipelines` (ggml-vulkan.cpp:4070 @c1d0e7a00).
const rdna1_pipelines = [_]PipelineSubgroupSize{
    .{ "soft_max", 64 },        .{ "im2col", 64 },
    .{ "argmax", 64 },          .{ "mul_mat_vec", 64 },
    .{ "mul_mat_vec_f16", 32 }, .{ "mul_mat_vec_f32_f16", 32 },
};

/// Ports `rdna2_pipelines` (ggml-vulkan.cpp:4077 @c1d0e7a00).
const rdna2_pipelines = [_]PipelineSubgroupSize{
    .{ "soft_max", 64 }, .{ "im2col", 64 },
};

/// Ports `RDNA_DEFAULT_SUBGROUP_SIZE` (ggml-vulkan.cpp:4081 @c1d0e7a00).
const rdna_default_subgroup_size = 32;

/// Ports `gpu_pipeline_configs` (ggml-vulkan.cpp:4084 @c1d0e7a00).
const gpu_pipeline_configs = [_]struct { arch: Architecture, pipelines: []const PipelineSubgroupSize, default_subgroup_size: u32 }{
    .{ .arch = .amd_rdna1, .pipelines = &rdna1_pipelines, .default_subgroup_size = rdna_default_subgroup_size },
    .{ .arch = .amd_rdna2, .pipelines = &rdna2_pipelines, .default_subgroup_size = rdna_default_subgroup_size },
};

/// Ports `get_subgroup_size` (ggml-vulkan.cpp:4101 @c1d0e7a00).
///
/// The C++ sorts the map's entries longest first and takes the first one
/// contained in the name. `std::sort` is not stable, so entries of equal
/// length come in an order the standard does not fix -- but no pipeline name
/// contains two keys of the same length, so trying every key of the longest
/// length first gives the same answer.
pub fn getSubgroupSize(pipeline_name: []const u8, arch: Architecture) u32 {
    for (gpu_pipeline_configs) |cfg| {
        if (cfg.arch != arch) continue;
        for (cfg.pipelines) |entry| {
            if (std.mem.eql(u8, entry[0], pipeline_name)) return entry[1];
        }
        var best: ?PipelineSubgroupSize = null;
        for (cfg.pipelines) |entry| {
            if (std.mem.indexOf(u8, pipeline_name, entry[0]) == null) continue;
            if (best == null or entry[0].len > best.?[0].len) best = entry;
        }
        if (best) |b| return b[1];
        return cfg.default_subgroup_size;
    }
    return 0; // If no matching configuration is found
}

/// Ports `ggml_vk_khr_cooperative_matrix_support` (ggml-vulkan.cpp:18939 @c1d0e7a00).
pub fn khrCooperativeMatrixSupport(props: *const vk.VkPhysicalDeviceProperties, driver_props: *const vk.VkPhysicalDeviceDriverProperties, arch: Architecture) bool {
    switch (props.vendorID) {
        common.vendor_id.intel => {
            // Only Xe2/Xe3 and integrated Xe for now: older hardware (Arc
            // A770) regresses.
            return arch == .intel_xe2 or
                (arch == .intel_xe1 and props.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU and
                    driver_props.driverID == vk.VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS);
        },
        common.vendor_id.amd => {
            if (driver_props.driverID == vk.VK_DRIVER_ID_AMD_PROPRIETARY or driver_props.driverID == vk.VK_DRIVER_ID_AMD_OPEN_SOURCE) {
                // The AMD proprietary driver reports support on every GPU.
                return arch == .amd_rdna3;
            }
            return true;
        },
        else => return true,
    }
}

// -----------------------------------------------------------------------------
// The instance

/// Ports `vk_instance_t` (ggml-vulkan.cpp:2553 @c1d0e7a00), without
/// `devices[]`, which belongs to the logical-device layer.
pub const Instance = struct {
    instance: vk.VkInstance = null,

    debug_utils_support: bool = false, // VK_EXT_debug_utils enabled
    pfn_vkSetDebugUtilsObjectNameEXT: vk.PFN_vkSetDebugUtilsObjectNameEXT = null,
    pfn_vkQueueBeginDebugUtilsLabelEXT: vk.PFN_vkQueueBeginDebugUtilsLabelEXT = null,
    pfn_vkQueueEndDebugUtilsLabelEXT: vk.PFN_vkQueueEndDebugUtilsLabelEXT = null,
    pfn_vkCmdBeginDebugUtilsLabelEXT: vk.PFN_vkCmdBeginDebugUtilsLabelEXT = null,
    pfn_vkCmdEndDebugUtilsLabelEXT: vk.PFN_vkCmdEndDebugUtilsLabelEXT = null,
    pfn_vkCmdInsertDebugUtilsLabelEXT: vk.PFN_vkCmdInsertDebugUtilsLabelEXT = null,

    device_indices: std.ArrayList(usize) = .empty,
    device_supports_membudget: std.ArrayList(bool) = .empty,
};

/// Ports `vk_instance_initialized` (ggml-vulkan.cpp:2569 @c1d0e7a00).
pub var initialized = false;
/// Ports `vk_instance` (ggml-vulkan.cpp:2570 @c1d0e7a00).
pub var instance: Instance = .{};

/// The logger switches `ggml_vk_instance_init` reads from the environment.
/// Ports `vk_memory_logger_enabled` (ggml-vulkan.cpp:2075 @c1d0e7a00),
/// `vk_perf_logger_enabled`, `vk_perf_logger_concurrent`,
/// `vk_enable_sync_logger` (ggml-vulkan.cpp:2115, 2116, 2117 @c1d0e7a00),
/// `vk_perf_logger_frequency` and `vk_pipeline_stats_filter`
/// (ggml-vulkan.cpp:2119, 2120 @c1d0e7a00).
pub const loggers = struct {
    pub var memory_logger_enabled = false;
    pub var perf_logger_enabled = false;
    pub var perf_logger_concurrent = false;
    pub var enable_sync_logger = false;
    pub var perf_logger_frequency: u32 = 1;
    pub var pipeline_stats_filter: ?[:0]const u8 = null;
};

/// What `instanceInit` can fail with, where the C++ throws.
pub const InitError = common.VkError || error{ VulkanTooOld, InvalidDeviceIndex, InvalidNumber };

/// Ports `ggml_vk_instance_layer_settings_available` (ggml-vulkan.cpp:18876 @c1d0e7a00).
///
/// False: its body is all behind `GGML_VULKAN_VALIDATE`, which the reference
/// build does not define.
fn layerSettingsAvailable() bool {
    return false;
}

/// Ports `ggml_vk_instance_portability_enumeration_ext_available`
/// (ggml-vulkan.cpp:18894 @c1d0e7a00) and the `__APPLE__` arms that call it,
/// which this build -- Linux and Windows -- never compiles.
fn portabilityEnumerationExtAvailable(instance_extensions: []const vk.VkExtensionProperties) bool {
    _ = instance_extensions;
    return false;
}

/// Ports `ggml_vk_instance_debug_utils_ext_available` (ggml-vulkan.cpp:18910 @c1d0e7a00).
fn debugUtilsExtAvailable(instance_extensions: []const vk.VkExtensionProperties) bool {
    if (common.hasExtension(instance_extensions, "VK_EXT_debug_utils")) return true;
    impl.fprintfStderr("ggml_vulkan: WARNING: Instance extension VK_EXT_debug_utils not found.\n", .{});
    return false;
}

/// Ports `ggml_vk_device_is_supported` (ggml-vulkan.cpp:18925 @c1d0e7a00).
fn deviceIsSupported(vkdev: vk.VkPhysicalDevice) bool {
    var device_features2 = common.structure(vk.VkPhysicalDeviceFeatures2, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
    var vk11_features = common.structure(vk.VkPhysicalDeviceVulkan11Features, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
    device_features2.pNext = &vk11_features;
    common.vkGetPhysicalDeviceFeatures2(vkdev, &device_features2);
    return vk11_features.storageBuffer16BitAccess != 0;
}

/// `std::stoul` into the C++'s `uint32_t`: leading space, an optional sign,
/// decimal digits, anything after them ignored; no digits throws.
fn stoulTruncated(text: []const u8) error{InvalidNumber}!u32 {
    var i: usize = 0;
    while (i < text.len and std.ascii.isWhitespace(text[i])) i += 1;
    var negative = false;
    if (i < text.len and (text[i] == '+' or text[i] == '-')) {
        negative = text[i] == '-';
        i += 1;
    }
    const start = i;
    var value: u64 = 0;
    while (i < text.len and std.ascii.isDigit(text[i])) : (i += 1) {
        value = std.math.mul(u64, value, 10) catch return error.InvalidNumber;
        value = std.math.add(u64, value, text[i] - '0') catch return error.InvalidNumber;
    }
    if (i == start) return error.InvalidNumber;
    if (negative) value = 0 -% value;
    return @truncate(value);
}

/// `std::stringstream >> size_t` over `GGML_VK_VISIBLE_DEVICES` with commas
/// turned into spaces: whitespace-separated unsigned decimal numbers, read
/// until the first token that does not start with one. A number runs into
/// what follows it -- `1a` reads `1`, then stops at `a`.
const VisibleDevices = struct {
    text: []const u8,
    pos: usize = 0,

    fn next(self: *VisibleDevices) ?usize {
        while (self.pos < self.text.len and (std.ascii.isWhitespace(self.text[self.pos]) or self.text[self.pos] == ',')) self.pos += 1;
        var negative = false;
        if (self.pos < self.text.len and (self.text[self.pos] == '+' or self.text[self.pos] == '-')) {
            negative = self.text[self.pos] == '-';
            self.pos += 1;
        }
        const start = self.pos;
        var value: usize = 0;
        var overflow = false;
        while (self.pos < self.text.len and std.ascii.isDigit(self.text[self.pos])) : (self.pos += 1) {
            const mul = @mulWithOverflow(value, 10);
            const add = @addWithOverflow(mul[0], self.text[self.pos] - '0');
            overflow = overflow or mul[1] != 0 or add[1] != 0;
            value = add[0];
        }
        if (self.pos == start or overflow) return null;
        return if (negative) 0 -% value else value;
    }
};

/// The driver priorities `ggml_vk_instance_init` uses to choose between two
/// drivers for one GPU: smaller wins, and a driver not listed is `INT_MAX`.
fn driverPriority(vendor_id: u32, driver_id: vk.VkDriverId) c_int {
    if (driver_id == vk.VK_DRIVER_ID_MESA_DOZEN) return 100;
    switch (vendor_id) {
        common.vendor_id.amd => switch (driver_id) {
            vk.VK_DRIVER_ID_MESA_RADV => return 1,
            vk.VK_DRIVER_ID_AMD_OPEN_SOURCE => return 2,
            vk.VK_DRIVER_ID_AMD_PROPRIETARY => return 3,
            else => {},
        },
        common.vendor_id.intel => switch (driver_id) {
            vk.VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA => return 1,
            vk.VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS => return 2,
            else => {},
        },
        common.vendor_id.nvidia => switch (driver_id) {
            vk.VK_DRIVER_ID_NVIDIA_PROPRIETARY => return 1,
            vk.VK_DRIVER_ID_MESA_NVK => return 2,
            else => {},
        },
        common.vendor_id.qualcomm => switch (driver_id) {
            vk.VK_DRIVER_ID_QUALCOMM_PROPRIETARY => return 1,
            vk.VK_DRIVER_ID_MESA_TURNIP => return 2,
            else => {},
        },
        else => {},
    }
    return std.math.maxInt(c_int);
}

/// A physical device's properties with its driver and ID properties chained,
/// as the deduplication reads them.
const DeviceIdentity = struct {
    props: vk.VkPhysicalDeviceProperties2,
    driver: vk.VkPhysicalDeviceDriverProperties,
    id: vk.VkPhysicalDeviceIDProperties,

    fn get(self: *DeviceIdentity, dev: vk.VkPhysicalDevice) void {
        self.props = common.structure(vk.VkPhysicalDeviceProperties2, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
        self.driver = common.structure(vk.VkPhysicalDeviceDriverProperties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES);
        self.id = common.structure(vk.VkPhysicalDeviceIDProperties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES);
        self.props.pNext = &self.driver;
        self.driver.pNext = &self.id;
        d.vkGetPhysicalDeviceProperties2.?(dev, &self.props);
    }
};

/// Ports `ggml_vk_instance_init` (ggml-vulkan.cpp:7352 @c1d0e7a00).
///
/// Return: nothing; an error where the C++ throws. As in the C++, the
/// instance counts as initialised from the moment it is created, so a
/// failure after that is not retried.
pub fn instanceInit() InitError!void {
    if (initialized) return;

    d.loadGlobal();

    var api_version: u32 = 0;
    try common.check(d.vkEnumerateInstanceVersion.?(&api_version), "vk::enumerateInstanceVersion");
    if (api_version < vk.VK_API_VERSION_1_2) {
        impl.fprintfStderr("ggml_vulkan: Error: Vulkan 1.2 required.\n", .{});
        return error.VulkanTooOld;
    }

    const app_info: vk.VkApplicationInfo = .{
        .sType = vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "ggml-vulkan",
        .applicationVersion = 1,
        .pEngineName = null,
        .engineVersion = 0,
        .apiVersion = api_version,
    };

    const instance_extensions = try common.enumerateInstanceExtensionProperties(null);
    defer allocator.free(instance_extensions);
    const layer_settings = layerSettingsAvailable();
    _ = portabilityEnumerationExtAvailable(instance_extensions);
    const debug_utils_ext = debugUtilsExtAvailable(instance_extensions) and common.envSet("GGML_VK_DEBUG_MARKERS");

    var layers: std.ArrayList([*:0]const u8) = .empty;
    defer layers.deinit(allocator);
    if (layer_settings) layers.append(allocator, "VK_LAYER_KHRONOS_validation") catch impl.abort("ggml_vulkan: out of memory");
    var extensions: std.ArrayList([*:0]const u8) = .empty;
    defer extensions.deinit(allocator);
    if (layer_settings) extensions.append(allocator, "VK_EXT_layer_settings") catch impl.abort("ggml_vulkan: out of memory");
    if (debug_utils_ext) extensions.append(allocator, "VK_EXT_debug_utils") catch impl.abort("ggml_vulkan: out of memory");

    // Chained whether or not the layer is there, as in the C++.
    var enable_best_practice: vk.VkBool32 = @intFromBool(layer_settings);
    const settings = [_]vk.VkLayerSettingEXT{.{
        .pLayerName = "VK_LAYER_KHRONOS_validation",
        .pSettingName = "validate_best_practices",
        .type = vk.VK_LAYER_SETTING_TYPE_BOOL32_EXT,
        .valueCount = 1,
        .pValues = &enable_best_practice,
    }};
    const layer_setting_info: vk.VkLayerSettingsCreateInfoEXT = .{
        .sType = vk.VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT,
        .pNext = null,
        .settingCount = settings.len,
        .pSettings = &settings,
    };
    const instance_create_info: vk.VkInstanceCreateInfo = .{
        .sType = vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = &layer_setting_info,
        .flags = 0,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = @intCast(layers.items.len),
        .ppEnabledLayerNames = @ptrCast(layers.items.ptr),
        .enabledExtensionCount = @intCast(extensions.items.len),
        .ppEnabledExtensionNames = @ptrCast(extensions.items.ptr),
    };

    try common.check(d.vkCreateInstance.?(&instance_create_info, null, &instance.instance), "vk::createInstance");
    initialized = true;

    if (debug_utils_ext) {
        instance.debug_utils_support = true;
        instance.pfn_vkSetDebugUtilsObjectNameEXT = @ptrCast(common.vkGetInstanceProcAddr(instance.instance, "vkSetDebugUtilsObjectNameEXT"));
        instance.pfn_vkQueueBeginDebugUtilsLabelEXT = @ptrCast(common.vkGetInstanceProcAddr(instance.instance, "vkQueueBeginDebugUtilsLabelEXT"));
        instance.pfn_vkQueueEndDebugUtilsLabelEXT = @ptrCast(common.vkGetInstanceProcAddr(instance.instance, "vkQueueEndDebugUtilsLabelEXT"));
        instance.pfn_vkCmdBeginDebugUtilsLabelEXT = @ptrCast(common.vkGetInstanceProcAddr(instance.instance, "vkCmdBeginDebugUtilsLabelEXT"));
        instance.pfn_vkCmdEndDebugUtilsLabelEXT = @ptrCast(common.vkGetInstanceProcAddr(instance.instance, "vkCmdEndDebugUtilsLabelEXT"));
        instance.pfn_vkCmdInsertDebugUtilsLabelEXT = @ptrCast(common.vkGetInstanceProcAddr(instance.instance, "vkCmdInsertDebugUtilsLabelEXT"));
    }

    loggers.perf_logger_enabled = common.envSet("GGML_VK_PERF_LOGGER");
    loggers.perf_logger_concurrent = common.envSet("GGML_VK_PERF_LOGGER_CONCURRENT");
    loggers.enable_sync_logger = common.envSet("GGML_VK_SYNC_LOGGER");
    loggers.memory_logger_enabled = common.envSet("GGML_VK_MEMORY_LOGGER");
    if (std.c.getenv("GGML_VK_PIPELINE_STATS")) |stats| loggers.pipeline_stats_filter = std.mem.span(stats);
    if (std.c.getenv("GGML_VK_PERF_LOGGER_FREQUENCY")) |freq| {
        loggers.perf_logger_frequency = try stoulTruncated(std.mem.span(freq));
    }

    d.loadInstance(instance.instance);

    const devices = try common.enumeratePhysicalDevices(instance.instance);
    defer allocator.free(devices);

    // Emulate behavior of CUDA_VISIBLE_DEVICES for Vulkan
    if (std.c.getenv("GGML_VK_VISIBLE_DEVICES")) |devices_env| {
        const num_available_devices = devices.len;
        var it: VisibleDevices = .{ .text = std.mem.span(devices_env) };
        while (it.next()) |tmp| {
            if (tmp >= num_available_devices) {
                impl.fprintfStderr("ggml_vulkan: Invalid device index %zu in GGML_VK_VISIBLE_DEVICES.\n", .{tmp});
                return error.InvalidDeviceIndex;
            }
            instance.device_indices.append(allocator, tmp) catch impl.abort("ggml_vulkan: out of memory");
        }
    } else {
        // If no vulkan devices are found, return early
        if (devices.len == 0) {
            impl.logInfo("ggml_vulkan: No devices found.\n", .{});
            return;
        }

        // Default to using all dedicated GPUs
        for (devices, 0..) |dev, i| {
            var new: DeviceIdentity = undefined;
            new.get(dev);
            const t = new.props.properties.deviceType;
            if (!((t == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU or t == vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) and deviceIsSupported(dev))) continue;

            // Two physical devices can be one GPU under two drivers (RADV and
            // AMDVLK, say). MoltenVK may report one UUID for distinct GPUs on
            // a multi-GPU card, so two MoltenVK devices are never merged.
            const old_device: ?usize = for (instance.device_indices.items) |k| {
                var old: DeviceIdentity = undefined;
                old.get(devices[k]);
                var same_uuid = std.mem.eql(u8, &old.id.deviceUUID, &new.id.deviceUUID);
                same_uuid = same_uuid or (old.id.deviceLUIDValid != 0 and new.id.deviceLUIDValid != 0 and
                    std.mem.eql(u8, &old.id.deviceLUID, &new.id.deviceLUID));
                const both_molten_vk = new.driver.driverID == vk.VK_DRIVER_ID_MOLTENVK and old.driver.driverID == vk.VK_DRIVER_ID_MOLTENVK;
                if (same_uuid and !both_molten_vk) break k;
            } else null;

            if (old_device == null) {
                instance.device_indices.append(allocator, i) catch impl.abort("ggml_vulkan: out of memory");
                continue;
            }
            // Keep one driver per GPU: splitting layers across both fails.
            var old_props = common.structure(vk.VkPhysicalDeviceProperties2, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
            var old_driver = common.structure(vk.VkPhysicalDeviceDriverProperties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES);
            old_props.pNext = &old_driver;
            d.vkGetPhysicalDeviceProperties2.?(devices[old_device.?], &old_props);

            // The C++ keys its priority map on the *old* device's vendor.
            const vendor = old_props.properties.vendorID;
            const old_priority = driverPriority(vendor, old_driver.driverID);
            const new_priority = driverPriority(vendor, new.driver.driverID);
            if (new_priority < old_priority) {
                // `std::remove` + `erase`: every occurrence, order kept.
                var w: usize = 0;
                for (instance.device_indices.items) |k| {
                    if (k != old_device.?) {
                        instance.device_indices.items[w] = k;
                        w += 1;
                    }
                }
                instance.device_indices.shrinkRetainingCapacity(w);
                instance.device_indices.append(allocator, i) catch impl.abort("ggml_vulkan: out of memory");
            }
        }

        // If no GPUs found, fall back to the first non-CPU device.
        // If only CPU devices are available, return without devices.
        if (instance.device_indices.items.len == 0) {
            for (devices, 0..) |dev, i| {
                if (common.getProperties(dev).deviceType != vk.VK_PHYSICAL_DEVICE_TYPE_CPU) {
                    instance.device_indices.append(allocator, i) catch impl.abort("ggml_vulkan: out of memory");
                    break;
                }
            }
        }

        if (instance.device_indices.items.len == 0) {
            impl.logInfo("ggml_vulkan: No devices found.\n", .{});
            return;
        }
    }
    impl.logDebug("ggml_vulkan: Found %zu Vulkan devices:\n", .{instance.device_indices.items.len});

    for (instance.device_indices.items, 0..) |dev_index, i| {
        const extensionprops = try common.enumerateDeviceExtensionProperties(devices[dev_index]);
        defer allocator.free(extensionprops);
        const membudget_supported = common.hasExtension(extensionprops, vk.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        instance.device_supports_membudget.append(allocator, membudget_supported) catch impl.abort("ggml_vulkan: out of memory");

        try printGpuInfo(i);
    }
}

/// `ggml_vk_instance_init` where the C++ lets its exception escape: anywhere
/// but `ggml_backend_vk_reg`, which catches it.
pub fn instanceInitOrAbort() void {
    instanceInit() catch impl.abort("ggml_vulkan: instance initialisation failed");
}

/// Ports `ggml_vk_print_gpu_info` (ggml-vulkan.cpp:7096 @c1d0e7a00).
fn printGpuInfo(idx: usize) InitError!void {
    impl.assert(idx < instance.device_indices.items.len, "idx < vk_instance.device_indices.size()");
    const dev_num = instance.device_indices.items[idx];
    impl.assert(initialized, "vk_instance_initialized");

    const devices = try common.enumeratePhysicalDevices(instance.instance);
    defer allocator.free(devices);
    if (dev_num >= devices.len) {
        impl.fprintfStderr("ggml_vulkan: Device with index %zu does not exist.\n", .{dev_num});
        return error.InvalidDeviceIndex;
    }

    const physical_device = devices[dev_num];
    const ext_props = try common.enumerateDeviceExtensionProperties(physical_device);
    defer allocator.free(ext_props);

    var fp16_storage = false;
    var fp16_compute = false;
    var coopmat_support = false;
    var coopmat2_support = false;
    var coopmat2_decode_vector_support = false;
    var integer_dot_product = false;
    var bfloat16_support = false;
    var dot2_f16_support = false;
    var ocp_microscaling_extension = false;
    var shader_float8_extension = false;

    for (ext_props) |properties| {
        const ext = std.mem.sliceTo(&properties.extensionName, 0);
        if (std.mem.eql(u8, ext, "VK_KHR_16bit_storage")) {
            fp16_storage = true;
        } else if (std.mem.eql(u8, ext, "VK_KHR_shader_float16_int8")) {
            fp16_compute = true;
        } else if (glslc.coopmat and std.mem.eql(u8, ext, "VK_KHR_cooperative_matrix") and !common.envSet("GGML_VK_DISABLE_COOPMAT")) {
            coopmat_support = true;
        } else if (glslc.coopmat2 and std.mem.eql(u8, ext, "VK_NV_cooperative_matrix2") and !common.envSet("GGML_VK_DISABLE_COOPMAT2")) {
            coopmat2_support = true;
        } else if (std.mem.eql(u8, ext, vk.VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME) and !common.envSet("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
            coopmat2_decode_vector_support = true;
        } else if (glslc.integer_dot and std.mem.eql(u8, ext, "VK_KHR_shader_integer_dot_product") and !common.envSet("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
            integer_dot_product = true;
        } else if (glslc.bfloat16 and std.mem.eql(u8, ext, "VK_KHR_shader_bfloat16") and !common.envSet("GGML_VK_DISABLE_BFLOAT16")) {
            bfloat16_support = true;
        } else if (glslc.float_e2m1 and std.mem.eql(u8, ext, vk.VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME)) {
            ocp_microscaling_extension = true;
        } else if (glslc.float_e4m3 and std.mem.eql(u8, ext, vk.VK_EXT_SHADER_FLOAT8_EXTENSION_NAME)) {
            shader_float8_extension = true;
        } else if (std.mem.eql(u8, ext, "VK_VALVE_shader_mixed_float_dot_product") and !common.envSet("GGML_VK_DISABLE_DOT2")) {
            dot2_f16_support = true;
        }
    }

    const device_architecture = try getDeviceArchitecture(physical_device);

    const force_disable_f16 = common.envSet("GGML_VK_DISABLE_F16");
    var fp16 = !force_disable_f16 and fp16_storage and fp16_compute;

    var props2 = common.structure(vk.VkPhysicalDeviceProperties2, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
    var props3 = common.structure(vk.VkPhysicalDeviceMaintenance3Properties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES);
    var subgroup_props = common.structure(vk.VkPhysicalDeviceSubgroupProperties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES);
    var driver_props = common.structure(vk.VkPhysicalDeviceDriverProperties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES);
    var shader_integer_dot_product_props = common.structure(vk.VkPhysicalDeviceShaderIntegerDotProductProperties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES);
    props2.pNext = &props3;
    props3.pNext = &subgroup_props;
    subgroup_props.pNext = &driver_props;
    if (integer_dot_product) driver_props.pNext = &shader_integer_dot_product_props;
    d.vkGetPhysicalDeviceProperties2.?(physical_device, &props2);

    var device_features2 = common.structure(vk.VkPhysicalDeviceFeatures2, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
    var vk11_features = common.structure(vk.VkPhysicalDeviceVulkan11Features, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
    var vk12_features = common.structure(vk.VkPhysicalDeviceVulkan12Features, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    device_features2.pNext = &vk11_features;
    vk11_features.pNext = &vk12_features;
    // Pointer to the last chain element
    var last_struct: *vk.VkBaseOutStructure = @ptrCast(&vk12_features);

    var coopmat_features = common.structure(vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR);
    if (glslc.coopmat and coopmat_support) chain(&last_struct, &coopmat_features);
    var shader_integer_dot_product_features = common.structure(vk.VkPhysicalDeviceShaderIntegerDotProductFeatures, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES);
    if (integer_dot_product) chain(&last_struct, &shader_integer_dot_product_features);
    var bfloat16_features = common.structure(vk.VkPhysicalDeviceShaderBfloat16FeaturesKHR, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR);
    if (bfloat16_support) chain(&last_struct, &bfloat16_features);
    var coopmat2_features = common.structure(vk.VkPhysicalDeviceCooperativeMatrix2FeaturesNV, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV);
    if (coopmat2_support) chain(&last_struct, &coopmat2_features);
    var coopmat2_decode_vector_features = common.structure(vk.VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV);
    if (coopmat2_decode_vector_support) chain(&last_struct, &coopmat2_decode_vector_features);
    var dot2_features = common.structure(vk.VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES_VALVE);
    if (dot2_f16_support) chain(&last_struct, &dot2_features);
    const fp4_compiled = glslc.float_e2m1 and glslc.float_e4m3;
    var ocp_microscaling_features = common.structure(vk.VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT);
    var shader_float8_features = common.structure(vk.VkPhysicalDeviceShaderFloat8FeaturesEXT, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT);
    if (fp4_compiled) {
        if (ocp_microscaling_extension) chain(&last_struct, &ocp_microscaling_features);
        if (shader_float8_extension) chain(&last_struct, &shader_float8_features);
    }

    common.vkGetPhysicalDeviceFeatures2(physical_device, &device_features2);

    fp16 = fp16 and vk12_features.shaderFloat16 != 0;
    const bf16 = bfloat16_support and bfloat16_features.shaderBFloat16Type != 0;

    const default_subgroup_size = getSubgroupSize("", device_architecture);
    const subgroup_size: usize = if (default_subgroup_size != 0) default_subgroup_size else subgroup_props.subgroupSize;
    const uma = props2.properties.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

    integer_dot_product = integer_dot_product and
        shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated != 0 and
        shader_integer_dot_product_features.shaderIntegerDotProduct != 0;

    coopmat_support = coopmat_support and
        (!glslc.coopmat or coopmat_features.cooperativeMatrix != 0) and
        khrCooperativeMatrixSupport(&props2.properties, &driver_props, device_architecture);

    coopmat2_support = glslc.coopmat2 and coopmat2_support and
        coopmat2_features.cooperativeMatrixWorkgroupScope != 0 and
        coopmat2_features.cooperativeMatrixFlexibleDimensions != 0 and
        coopmat2_features.cooperativeMatrixReductions != 0 and
        coopmat2_features.cooperativeMatrixConversions != 0 and
        coopmat2_features.cooperativeMatrixPerElementOperations != 0 and
        coopmat2_features.cooperativeMatrixTensorAddressing != 0 and
        coopmat2_features.cooperativeMatrixBlockLoads != 0;

    coopmat2_decode_vector_support = glslc.coopmat2_decode_vector and coopmat2_decode_vector_support and
        coopmat2_decode_vector_features.cooperativeMatrixDecodeVector != 0;

    const matrix_cores: [*:0]const u8 = if (coopmat2_support)
        (if (coopmat2_decode_vector_support) "NV_coopmat2v" else "NV_coopmat2")
    else if (coopmat_support) "KHR_coopmat" else "none";

    const dot2_f16 = dot2_f16_support and dot2_features.shaderMixedFloatDotProductFloat16AccFloat32 != 0;
    const fp16_str: [*:0]const u8 = if (fp16) (if (dot2_f16) "dot2" else "1") else "0";
    const fp4 = fp4_compiled and ocp_microscaling_extension and ocp_microscaling_features.shaderFloat4 != 0 and
        shader_float8_extension and shader_float8_features.shaderFloat8 != 0 and
        !common.envSet("GGML_VK_DISABLE_OCP_FP4");

    impl.logDebug("ggml_vulkan: %zu = %s (%s) | uma: %d | fp16: %s | bf16: %d | fp4: %d | warp size: %zu | shared memory: %d | int dot: %d | matrix cores: %s\n", .{
        idx,                                                    @as([*:0]const u8, @ptrCast(&props2.properties.deviceName)),
        @as([*:0]const u8, @ptrCast(&driver_props.driverName)), @as(c_int, @intFromBool(uma)),
        fp16_str,                                               @as(c_int, @intFromBool(bf16)),
        @as(c_int, @intFromBool(fp4)),                          subgroup_size,
        props2.properties.limits.maxComputeSharedMemorySize,    @as(c_int, @intFromBool(integer_dot_product)),
        matrix_cores,
    });

    if (props2.properties.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_CPU) {
        impl.logDebug("ggml_vulkan: Warning: Device type is CPU. This is probably not the device you want.\n", .{});
    }
}

/// Appends `next` to a `pNext` chain whose last element is `last.*`, as the
/// C++'s `last_struct->pNext = ...; last_struct = ...;` pairs do.
fn chain(last: **vk.VkBaseOutStructure, next: anytype) void {
    const node: *vk.VkBaseOutStructure = @ptrCast(next);
    last.*.pNext = node;
    last.* = node;
}

// -----------------------------------------------------------------------------
// Per-device queries

/// Ports `ggml_vk_get_device_count` (ggml-vulkan.cpp:15965 @c1d0e7a00).
pub fn getDeviceCount() c_int {
    instanceInitOrAbort();
    return @intCast(instance.device_indices.items.len);
}

/// Copies `text` into `description` as `snprintf(description, size, "%s",
/// text)` does: at most `size - 1` bytes, then a terminator.
fn copyTruncated(description: [*]u8, description_size: usize, text: []const u8) void {
    if (description_size == 0) return;
    const n = @min(text.len, description_size - 1);
    @memcpy(description[0..n], text[0..n]);
    description[n] = 0;
}

/// Ports `ggml_vk_get_device_description` (ggml-vulkan.cpp:15971 @c1d0e7a00).
///
/// `device` indexes the *physical* devices, not the backend's.
pub fn getPhysicalDeviceDescription(device: usize, description: [*]u8, description_size: usize) void {
    instanceInitOrAbort();
    const devices = common.enumeratePhysicalDevices(instance.instance) catch impl.abort("ggml_vulkan: vk::Instance::enumeratePhysicalDevices failed");
    defer allocator.free(devices);
    const props = common.getProperties(devices[device]);
    copyTruncated(description, description_size, std.mem.sliceTo(&props.deviceName, 0));
}

/// The physical device behind backend device `device`, as the C++'s
/// `vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]]`.
pub fn physicalDevice(device: usize) vk.VkPhysicalDevice {
    const devices = common.enumeratePhysicalDevices(instance.instance) catch impl.abort("ggml_vulkan: vk::Instance::enumeratePhysicalDevices failed");
    defer allocator.free(devices);
    return devices[instance.device_indices.items[device]];
}

/// Ports `ggml_backend_vk_get_device_memory` (ggml-vulkan.cpp:17904 @c1d0e7a00).
pub fn getDeviceMemory(device: c_int, free: *usize, total: *usize) void {
    impl.assert(device < @as(c_int, @intCast(instance.device_indices.items.len)), "device < (int) vk_instance.device_indices.size()");
    impl.assert(device < @as(c_int, @intCast(instance.device_supports_membudget.items.len)), "device < (int) vk_instance.device_supports_membudget.size()");
    const index: usize = @intCast(device);

    const vkdev = physicalDevice(index);
    var budgetprops = common.structure(vk.VkPhysicalDeviceMemoryBudgetPropertiesEXT, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT);
    var memprops = common.structure(vk.VkPhysicalDeviceMemoryProperties2, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2);
    const membudget_supported = instance.device_supports_membudget.items[index];
    const is_integrated_gpu = common.getProperties(vkdev).deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

    if (membudget_supported) memprops.pNext = &budgetprops;
    d.vkGetPhysicalDeviceMemoryProperties2.?(vkdev, &memprops);

    total.* = 0;
    free.* = 0;
    for (0..memprops.memoryProperties.memoryHeapCount) |i| {
        const heap = memprops.memoryProperties.memoryHeaps[i];
        if (is_integrated_gpu or (heap.flags & vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            total.* +%= heap.size;
            if (membudget_supported and i < budgetprops.heapUsage.len) {
                free.* +%= budgetprops.heapBudget[i] -% budgetprops.heapUsage[i];
            } else {
                free.* +%= heap.size;
            }
        }
    }
}

/// Ports `ggml_backend_vk_get_device_type` (ggml-vulkan.cpp:17937 @c1d0e7a00).
pub fn getDeviceType(device_idx: usize) vk.VkPhysicalDeviceType {
    impl.assert(device_idx < instance.device_indices.items.len, "device_idx >= 0 && device_idx < (int) vk_instance.device_indices.size()");
    const device = physicalDevice(device_idx);
    var props = common.structure(vk.VkPhysicalDeviceProperties2, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
    d.vkGetPhysicalDeviceProperties2.?(device, &props);
    return props.properties.deviceType;
}

/// Ports `ggml_backend_vk_get_device_pci_id` (ggml-vulkan.cpp:17948 @c1d0e7a00).
///
/// Return: the bus id, `dddd:bb:dd.f`, or an empty string when the device
/// lacks `VK_EXT_pci_bus_info`.
pub fn getDevicePciId(device_idx: usize, buf: *[16]u8) []const u8 {
    impl.assert(device_idx < instance.device_indices.items.len, "device_idx >= 0 && device_idx < (int) vk_instance.device_indices.size()");
    const device = physicalDevice(device_idx);

    const ext_props = common.enumerateDeviceExtensionProperties(device) catch impl.abort("ggml_vulkan: enumerateDeviceExtensionProperties failed");
    defer allocator.free(ext_props);
    if (!common.hasExtension(ext_props, "VK_EXT_pci_bus_info")) return "";

    var props = common.structure(vk.VkPhysicalDeviceProperties2, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
    var pci_bus_info = common.structure(vk.VkPhysicalDevicePCIBusInfoPropertiesEXT, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT);
    props.pNext = &pci_bus_info;
    d.vkGetPhysicalDeviceProperties2.?(device, &props);

    // The C++ narrows the function to `uint8_t` before printing it.
    const pci_function: u8 = @truncate(pci_bus_info.pciFunction);
    // `char pci_bus_id[16]` and `snprintf`: at most 15 characters.
    var tmp: [64]u8 = undefined;
    const full = std.fmt.bufPrint(&tmp, "{x:0>4}:{x:0>2}:{x:0>2}.{x}", .{ pci_bus_info.pciDomain, pci_bus_info.pciBus, pci_bus_info.pciDevice, pci_function }) catch unreachable;
    const n = @min(full.len, buf.len - 1);
    @memcpy(buf[0..n], full[0..n]);
    return buf[0..n];
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "getSubgroupSize matches the C++'s lookup" {
    try std.testing.expectEqual(@as(u32, 0), getSubgroupSize("", .other));
    try std.testing.expectEqual(@as(u32, 32), getSubgroupSize("", .amd_rdna1));
    try std.testing.expectEqual(@as(u32, 64), getSubgroupSize("soft_max", .amd_rdna1));
    // The longest contained key wins: `mul_mat_vec_f32_f16` over `mul_mat_vec`.
    try std.testing.expectEqual(@as(u32, 32), getSubgroupSize("mul_mat_vec_f32_f16_f32", .amd_rdna1));
    try std.testing.expectEqual(@as(u32, 64), getSubgroupSize("mul_mat_vec_q4_0_f32_f32", .amd_rdna1));
    try std.testing.expectEqual(@as(u32, 32), getSubgroupSize("mul_mat_vec_q4_0_f32_f32", .amd_rdna2));
}

test "VisibleDevices reads as a stringstream does" {
    var it: VisibleDevices = .{ .text = "0, 2 1a 3" };
    try std.testing.expectEqual(@as(?usize, 0), it.next());
    try std.testing.expectEqual(@as(?usize, 2), it.next());
    try std.testing.expectEqual(@as(?usize, 1), it.next());
    try std.testing.expectEqual(@as(?usize, null), it.next());
}
