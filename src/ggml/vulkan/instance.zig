//! The Vulkan instance and the physical devices the backend exposes: instance
//! creation, device selection and deduplication, the architecture probe, the
//! `ggml_vulkan: N = ...` line, and the per-device queries `ggml-vulkan.h`
//! answers -- count, description, memory, type and PCI id.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-debug.cpp`, and
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-types.h` at v1.0.0
//! (`b1b770d90`). Each declaration below names the C++ it replaces and the line it
//! began at.
//!
//! # Exceptions
//!
//! The C++ throws from `ggml_vk_instance_init` -- a failed Vulkan call, or a
//! bad `GGML_VK_VISIBLE_DEVICES` index -- and `ggml_backend_vk_reg` catches
//! everything and returns null. `instanceInit` returns an error union
//! instead, and `backend.zig`'s `reg` returns null on it; every other caller
//! lets the C++'s exception escape, which terminates, so they abort.

const std = @import("std");
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");
const config = @import("config");

comptime {
    // The C++ carries fallback declarations for headers that predate
    // `VK_NV_cooperative_matrix_decode_vector`, `VK_KHR_shader_bfloat16`,
    // `VK_VALVE_shader_mixed_float_dot_product`,
    // `VK_EXT_shader_ocp_microscaling_types` and `VK_EXT_shader_float8`
    // (ggml-vulkan-types.h:28, 97, 114, 129, 144 @b1b770d90). The pinned SDK has
    // all of them, so the port needs none.
    for (.{
        "VK_NV_cooperative_matrix_decode_vector",  "VK_KHR_shader_bfloat16",
        "VK_VALVE_shader_mixed_float_dot_product", "VK_EXT_shader_ocp_microscaling_types",
        "VK_EXT_shader_float8",                    "VK_NV_cooperative_matrix2",
    }) |ext| {
        if (!@hasDecl(ggml.vulkan.common.vulkan_c, ext)) @compileError("the Vulkan headers predate " ++ ext ++ "; build with the pinned SDK (scripts/vk-toolchain)");
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

/// Ports `vk_device_architecture` (ggml-vulkan-types.h:366 @b1b770d90).
/// The ordinals are the C's: the cm1 integer-dot matmul passes one to its
/// shader as the `DEVICE_ARCH` specialization constant.
pub const Architecture = enum(u32) {
    other,
    amd_gcn,
    amd_rdna1,
    amd_rdna2,
    amd_rdna3,
    amd_rdna4,
    intel_xe1,
    intel_xe2,
    nvidia_pre_turing,
    nvidia_turing,
    qualcomm_adreno,
};

/// Ports `get_device_architecture` (ggml-vulkan.cpp:3 @b1b770d90).
pub fn getDeviceArchitecture(device: ggml.vulkan.common.vulkan_c.VkPhysicalDevice) ggml.vulkan.common.VkError!Architecture {
    const props = ggml.vulkan.common.getProperties(device);

    if (props.vendorID == ggml.vulkan.common.vendor_id.amd) {
        const ext_props = try ggml.vulkan.common.enumerateDeviceExtensionProperties(device);
        defer std.heap.c_allocator.free(ext_props);
        const amd_shader_core_properties = ggml.vulkan.common.hasExtension(ext_props, "VK_AMD_shader_core_properties");
        const integer_dot_product = ggml.vulkan.common.hasExtension(ext_props, "VK_KHR_shader_integer_dot_product");
        const subgroup_size_control = ggml.vulkan.common.hasExtension(ext_props, "VK_EXT_subgroup_size_control");
        const shader_float8 = ggml.vulkan.common.hasExtension(ext_props, "VK_EXT_shader_float8");
        if (!amd_shader_core_properties or !integer_dot_product or !subgroup_size_control) return .other;

        var props2 = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceProperties2, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
        var shader_core_props_amd = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceShaderCorePropertiesAMD, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD);
        var integer_dot_props = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceShaderIntegerDotProductProperties, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES);
        var subgroup_size_control_props = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceSubgroupSizeControlProperties, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES);
        props2.pNext = &shader_core_props_amd;
        shader_core_props_amd.pNext = &integer_dot_props;
        integer_dot_props.pNext = &subgroup_size_control_props;
        ggml.vulkan.common.dispatch.vkGetPhysicalDeviceProperties2.?(device, &props2);

        if (subgroup_size_control_props.maxSubgroupSize == 64 and subgroup_size_control_props.minSubgroupSize == 64) return .amd_gcn;
        if (subgroup_size_control_props.maxSubgroupSize == 64 and subgroup_size_control_props.minSubgroupSize == 32) {
            // RDNA
            if (shader_core_props_amd.wavefrontsPerSimd == 20) return .amd_rdna1;
            if (shader_float8) return .amd_rdna4;
            if (integer_dot_props.integerDotProduct4x8BitPackedMixedSignednessAccelerated != 0) return .amd_rdna3;
            return .amd_rdna2;
        }
    } else if (props.vendorID == ggml.vulkan.common.vendor_id.intel) {
        const ext_props = try ggml.vulkan.common.enumerateDeviceExtensionProperties(device);
        defer std.heap.c_allocator.free(ext_props);
        const subgroup_size_control = ggml.vulkan.common.hasExtension(ext_props, "VK_EXT_subgroup_size_control");
        const integer_dot_product = ggml.vulkan.common.hasExtension(ext_props, "VK_KHR_shader_integer_dot_product");
        if (!subgroup_size_control or !integer_dot_product) return .other;

        var props2 = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceProperties2, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
        var subgroup_size_control_props = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceSubgroupSizeControlProperties, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES);
        var integer_dot_props = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceShaderIntegerDotProductProperties, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES);
        props2.pNext = &subgroup_size_control_props;
        subgroup_size_control_props.pNext = &integer_dot_props;
        ggml.vulkan.common.dispatch.vkGetPhysicalDeviceProperties2.?(device, &props2);

        if (subgroup_size_control_props.minSubgroupSize == 16) {
            // Xe2 is SIMD16, earlier Xe and Gen SIMD8: the minimum subgroup
            // size is the SIMD width.
            return .intel_xe2;
        } else if (subgroup_size_control_props.minSubgroupSize == 8 and
            integer_dot_product and integer_dot_props.integerDotProduct4x8BitPackedSignedAccelerated != 0)
        {
            return .intel_xe1;
        }
    } else if (props.vendorID == ggml.vulkan.common.vendor_id.nvidia) {
        const ext_props = try ggml.vulkan.common.enumerateDeviceExtensionProperties(device);
        defer std.heap.c_allocator.free(ext_props);
        // "Pre-Turing" is the lack of coopmat support.
        const cooperative_matrix = ggml.vulkan.common.hasExtension(ext_props, "VK_KHR_cooperative_matrix");
        const sm_builtins = ggml.vulkan.common.hasExtension(ext_props, "VK_NV_shader_sm_builtins");
        if (!cooperative_matrix) return .nvidia_pre_turing;

        if (sm_builtins) {
            var props2 = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceProperties2, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
            var sm_props = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceShaderSMBuiltinsPropertiesNV, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV);
            props2.pNext = &sm_props;
            ggml.vulkan.common.dispatch.vkGetPhysicalDeviceProperties2.?(device, &props2);
            // Turing has 32, later architectures 48.
            if (sm_props.shaderWarpsPerSM == 32) return .nvidia_turing;
        }
    } else if (props.vendorID == ggml.vulkan.common.vendor_id.qualcomm) {
        const ext_props = try ggml.vulkan.common.enumerateDeviceExtensionProperties(device);
        defer std.heap.c_allocator.free(ext_props);
        const cooperative_matrix = ggml.vulkan.common.hasExtension(ext_props, "VK_KHR_cooperative_matrix");
        const cooperative_matrix_conversion = ggml.vulkan.common.hasExtension(ext_props, "VK_QCOM_cooperative_matrix_conversion");
        if (cooperative_matrix and cooperative_matrix_conversion) return .qualcomm_adreno;
    }
    return .other;
}

/// One entry of `GpuPipelineConfig` (ggml-vulkan-types.h:1457 @b1b770d90): a
/// pipeline name and its subgroup size.
const PipelineSubgroupSize = struct { []const u8, u32 };

/// Ports `rdna1_pipelines` (ggml-vulkan.cpp:1797 @b1b770d90).
const rdna1_pipelines = [_]PipelineSubgroupSize{
    .{ "soft_max", 64 },        .{ "im2col", 64 },
    .{ "argmax", 64 },          .{ "mul_mat_vec", 64 },
    .{ "mul_mat_vec_f16", 32 }, .{ "mul_mat_vec_f32_f16", 32 },
};

/// Ports `rdna2_pipelines` (ggml-vulkan.cpp:1803 @b1b770d90).
const rdna2_pipelines = [_]PipelineSubgroupSize{
    .{ "soft_max", 64 }, .{ "im2col", 64 },
};

/// Ports `RDNA_DEFAULT_SUBGROUP_SIZE` (ggml-vulkan-types.h:1471 @b1b770d90).
const rdna_default_subgroup_size = 32;

/// Ports `gpu_pipeline_configs` (ggml-vulkan.cpp:1807 @b1b770d90).
const gpu_pipeline_configs = [_]struct { arch: Architecture, pipelines: []const PipelineSubgroupSize, default_subgroup_size: u32 }{
    .{ .arch = .amd_rdna1, .pipelines = &rdna1_pipelines, .default_subgroup_size = rdna_default_subgroup_size },
    .{ .arch = .amd_rdna2, .pipelines = &rdna2_pipelines, .default_subgroup_size = rdna_default_subgroup_size },
};

/// Ports `get_subgroup_size` (ggml-vulkan.cpp:1824 @b1b770d90).
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

/// Ports `ggml_vk_khr_cooperative_matrix_support` (ggml-vulkan.cpp:16852 @b1b770d90).
pub fn khrCooperativeMatrixSupport(props: *const ggml.vulkan.common.vulkan_c.VkPhysicalDeviceProperties, driver_props: *const ggml.vulkan.common.vulkan_c.VkPhysicalDeviceDriverProperties, arch: Architecture) bool {
    switch (props.vendorID) {
        ggml.vulkan.common.vendor_id.intel => {
            // Only Xe2/Xe3 and integrated Xe for now: older hardware (Arc
            // A770) regresses.
            return arch == .intel_xe2 or
                (arch == .intel_xe1 and props.deviceType == ggml.vulkan.common.vulkan_c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU and
                    driver_props.driverID == ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS);
        },
        ggml.vulkan.common.vendor_id.amd => {
            if (driver_props.driverID == ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_AMD_PROPRIETARY or driver_props.driverID == ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_AMD_OPEN_SOURCE) {
                // The AMD proprietary driver reports support on every GPU.
                return arch == .amd_rdna3 or arch == .amd_rdna4;
            }
            return true;
        },
        // Only Adreno GPUs with hardware matrix cores (Gen 6+).
        ggml.vulkan.common.vendor_id.qualcomm => return arch == .qualcomm_adreno,
        else => return true,
    }
}

// -----------------------------------------------------------------------------
// The instance

/// Ports `vk_instance_t` (ggml-vulkan-types.h:1375 @b1b770d90), without
/// `devices[]`, which belongs to the logical-device layer, or
/// `host_allocations` and its mutex, which belong to the buffer layer
/// (`buffer.zig`'s `host_allocations`).
pub const Instance = struct {
    instance: ggml.vulkan.common.vulkan_c.VkInstance = null,

    debug_utils_support: bool = false, // VK_EXT_debug_utils enabled
    pfn_vkSetDebugUtilsObjectNameEXT: ggml.vulkan.common.vulkan_c.PFN_vkSetDebugUtilsObjectNameEXT = null,
    pfn_vkQueueBeginDebugUtilsLabelEXT: ggml.vulkan.common.vulkan_c.PFN_vkQueueBeginDebugUtilsLabelEXT = null,
    pfn_vkQueueEndDebugUtilsLabelEXT: ggml.vulkan.common.vulkan_c.PFN_vkQueueEndDebugUtilsLabelEXT = null,
    pfn_vkCmdBeginDebugUtilsLabelEXT: ggml.vulkan.common.vulkan_c.PFN_vkCmdBeginDebugUtilsLabelEXT = null,
    pfn_vkCmdEndDebugUtilsLabelEXT: ggml.vulkan.common.vulkan_c.PFN_vkCmdEndDebugUtilsLabelEXT = null,
    pfn_vkCmdInsertDebugUtilsLabelEXT: ggml.vulkan.common.vulkan_c.PFN_vkCmdInsertDebugUtilsLabelEXT = null,

    device_indices: std.ArrayList(usize) = .empty,
    device_supports_membudget: std.ArrayList(bool) = .empty,
    device_host_buffer_import: std.ArrayList(bool) = .empty,

    host_import_alignment: usize = 0,
    host_import_failed: u32 = 0, // bitmask by device->idx, guarded by host_allocations_mutex

    multi_device: bool = false,
    allreduce_disabled: bool = false,
};

/// Ports `vk_instance_initialized` (ggml-vulkan.cpp:331 @b1b770d90).
pub var initialized = false;
/// Ports `vk_instance` (ggml-vulkan.cpp:333 @b1b770d90).
pub var instance: Instance = .{};

/// The logger switches `ggml_vk_instance_init` reads from the environment.
/// Ports `vk_memory_logger_enabled` (ggml-vulkan-debug.cpp:3 @b1b770d90),
/// `vk_perf_logger_enabled`, `vk_perf_logger_concurrent`,
/// `vk_enable_sync_logger` (ggml-vulkan-debug.cpp:5, 7, 9 @b1b770d90),
/// `vk_perf_logger_frequency` and `vk_pipeline_stats_filter`
/// (ggml-vulkan-debug.cpp:11, 13 @b1b770d90).
pub const loggers = struct {
    pub var memory_logger_enabled = false;
    pub var perf_logger_enabled = false;
    pub var perf_logger_concurrent = false;
    pub var enable_sync_logger = false;
    pub var perf_logger_frequency: u32 = 1;
    pub var pipeline_stats_filter: ?[:0]const u8 = null;
};

/// What `instanceInit` can fail with, where the C++ throws.
pub const InitError = ggml.vulkan.common.VkError || error{ VulkanTooOld, InvalidDeviceIndex, InvalidNumber };

/// Ports `ggml_vk_instance_layer_settings_available` (ggml-vulkan.cpp:16789 @b1b770d90).
///
/// False: its body is all behind `GGML_VULKAN_VALIDATE`, which the reference
/// build does not define.
fn layerSettingsAvailable() bool {
    return false;
}

/// Ports `ggml_vk_instance_portability_enumeration_ext_available`
/// (ggml-vulkan.cpp:16808 @b1b770d90) and the `__APPLE__` arms that call it,
/// which this build -- Linux and Windows -- never compiles.
fn portabilityEnumerationExtAvailable(instance_extensions: []const ggml.vulkan.common.vulkan_c.VkExtensionProperties) bool {
    _ = instance_extensions;
    return false;
}

/// Ports `ggml_vk_instance_debug_utils_ext_available` (ggml-vulkan.cpp:16823 @b1b770d90).
fn debugUtilsExtAvailable(instance_extensions: []const ggml.vulkan.common.vulkan_c.VkExtensionProperties) bool {
    if (ggml.vulkan.common.hasExtension(instance_extensions, "VK_EXT_debug_utils")) return true;
    ggml.impl.fprintfStderr("ggml_vulkan: WARNING: Instance extension VK_EXT_debug_utils not found.\n", .{});
    return false;
}

/// Ports `ggml_vk_device_is_supported` (ggml-vulkan.cpp:16838 @b1b770d90).
fn deviceIsSupported(vkdev: ggml.vulkan.common.vulkan_c.VkPhysicalDevice) bool {
    var device_features2 = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceFeatures2, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
    var vk11_features = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceVulkan11Features, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
    device_features2.pNext = &vk11_features;
    ggml.vulkan.common.vkGetPhysicalDeviceFeatures2(vkdev, &device_features2);
    return vk11_features.storageBuffer16BitAccess != 0;
}

/// `std::stoul` into the C++'s `uint32_t`: `strtoul` in base 10, its two
/// throws (no digits, out of range) as an error, the value narrowed.
fn stoulTruncated(text: []const u8) error{InvalidNumber}!u32 {
    const p = stdlib.num.strtoul(text, 10);
    if (p.end == 0 or p.range) return error.InvalidNumber;
    return @truncate(p.value);
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
fn driverPriority(vendor_id: u32, driver_id: ggml.vulkan.common.vulkan_c.VkDriverId) c_int {
    if (driver_id == ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_MESA_DOZEN) return 100;
    switch (vendor_id) {
        ggml.vulkan.common.vendor_id.amd => switch (driver_id) {
            ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_MESA_RADV => return 1,
            ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_AMD_OPEN_SOURCE => return 2,
            ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_AMD_PROPRIETARY => return 3,
            else => {},
        },
        ggml.vulkan.common.vendor_id.intel => switch (driver_id) {
            ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA => return 1,
            ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS => return 2,
            else => {},
        },
        ggml.vulkan.common.vendor_id.nvidia => switch (driver_id) {
            ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_NVIDIA_PROPRIETARY => return 1,
            ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_MESA_NVK => return 2,
            else => {},
        },
        ggml.vulkan.common.vendor_id.qualcomm => switch (driver_id) {
            ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_QUALCOMM_PROPRIETARY => return 1,
            ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_MESA_TURNIP => return 2,
            else => {},
        },
        else => {},
    }
    return std.math.maxInt(c_int);
}

/// A physical device's properties with its driver and ID properties chained,
/// as the deduplication reads them.
const DeviceIdentity = struct {
    props: ggml.vulkan.common.vulkan_c.VkPhysicalDeviceProperties2,
    driver: ggml.vulkan.common.vulkan_c.VkPhysicalDeviceDriverProperties,
    id: ggml.vulkan.common.vulkan_c.VkPhysicalDeviceIDProperties,

    fn get(self: *DeviceIdentity, dev: ggml.vulkan.common.vulkan_c.VkPhysicalDevice) void {
        self.props = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceProperties2, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
        self.driver = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceDriverProperties, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES);
        self.id = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceIDProperties, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES);
        self.props.pNext = &self.driver;
        self.driver.pNext = &self.id;
        ggml.vulkan.common.dispatch.vkGetPhysicalDeviceProperties2.?(dev, &self.props);
    }
};

/// Ports `ggml_vk_instance_init` (ggml-vulkan.cpp:5470 @b1b770d90).
///
/// Return: nothing; an error where the C++ throws. As in the C++, the
/// instance counts as initialised from the moment it is created, so a
/// failure after that is not retried.
pub fn instanceInit() InitError!void {
    if (initialized) return;

    ggml.vulkan.common.dispatch.loadGlobal();

    // vkEnumerateInstanceVersion is Vulkan 1.1. A null value indicated Vulkan 1.0.
    if (ggml.vulkan.common.dispatch.vkEnumerateInstanceVersion == null) {
        ggml.impl.logError("ggml_vulkan: Error: Vulkan 1.2 required.", .{});
        return error.VulkanTooOld;
    }

    var api_version: u32 = 0;
    try ggml.vulkan.common.check(ggml.vulkan.common.dispatch.vkEnumerateInstanceVersion.?(&api_version), "vk::enumerateInstanceVersion");
    if (api_version < ggml.vulkan.common.vulkan_c.VK_API_VERSION_1_2) {
        ggml.impl.logError("ggml_vulkan: Error: Vulkan 1.2 required.", .{});
        return error.VulkanTooOld;
    }

    const app_info: ggml.vulkan.common.vulkan_c.VkApplicationInfo = .{
        .sType = ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "ggml-vulkan",
        .applicationVersion = 1,
        .pEngineName = null,
        .engineVersion = 0,
        .apiVersion = api_version,
    };

    const instance_extensions = try ggml.vulkan.common.enumerateInstanceExtensionProperties(null);
    defer std.heap.c_allocator.free(instance_extensions);
    const layer_settings = layerSettingsAvailable();
    _ = portabilityEnumerationExtAvailable(instance_extensions);
    const debug_utils_ext = debugUtilsExtAvailable(instance_extensions) and ggml.vulkan.common.envSet("GGML_VK_DEBUG_MARKERS");

    var layers: std.ArrayList([*:0]const u8) = .empty;
    defer layers.deinit(std.heap.c_allocator);
    if (layer_settings) layers.append(std.heap.c_allocator, "VK_LAYER_KHRONOS_validation") catch ggml.impl.abort("ggml_vulkan: out of memory");
    var extensions: std.ArrayList([*:0]const u8) = .empty;
    defer extensions.deinit(std.heap.c_allocator);
    if (layer_settings) extensions.append(std.heap.c_allocator, "VK_EXT_layer_settings") catch ggml.impl.abort("ggml_vulkan: out of memory");
    if (debug_utils_ext) extensions.append(std.heap.c_allocator, "VK_EXT_debug_utils") catch ggml.impl.abort("ggml_vulkan: out of memory");

    // Chained whether or not the layer is there, as in the C++.
    var enable_best_practice: ggml.vulkan.common.vulkan_c.VkBool32 = @intFromBool(layer_settings);
    const settings = [_]ggml.vulkan.common.vulkan_c.VkLayerSettingEXT{.{
        .pLayerName = "VK_LAYER_KHRONOS_validation",
        .pSettingName = "validate_best_practices",
        .type = ggml.vulkan.common.vulkan_c.VK_LAYER_SETTING_TYPE_BOOL32_EXT,
        .valueCount = 1,
        .pValues = &enable_best_practice,
    }};
    const layer_setting_info: ggml.vulkan.common.vulkan_c.VkLayerSettingsCreateInfoEXT = .{
        .sType = ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT,
        .pNext = null,
        .settingCount = settings.len,
        .pSettings = &settings,
    };
    const instance_create_info: ggml.vulkan.common.vulkan_c.VkInstanceCreateInfo = .{
        .sType = ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = &layer_setting_info,
        .flags = 0,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = @intCast(layers.items.len),
        .ppEnabledLayerNames = @ptrCast(layers.items.ptr),
        .enabledExtensionCount = @intCast(extensions.items.len),
        .ppEnabledExtensionNames = @ptrCast(extensions.items.ptr),
    };

    try ggml.vulkan.common.check(ggml.vulkan.common.dispatch.vkCreateInstance.?(&instance_create_info, null, &instance.instance), "vk::createInstance");
    initialized = true;

    if (debug_utils_ext) {
        instance.debug_utils_support = true;
        instance.pfn_vkSetDebugUtilsObjectNameEXT = @ptrCast(ggml.vulkan.common.vkGetInstanceProcAddr(instance.instance, "vkSetDebugUtilsObjectNameEXT"));
        instance.pfn_vkQueueBeginDebugUtilsLabelEXT = @ptrCast(ggml.vulkan.common.vkGetInstanceProcAddr(instance.instance, "vkQueueBeginDebugUtilsLabelEXT"));
        instance.pfn_vkQueueEndDebugUtilsLabelEXT = @ptrCast(ggml.vulkan.common.vkGetInstanceProcAddr(instance.instance, "vkQueueEndDebugUtilsLabelEXT"));
        instance.pfn_vkCmdBeginDebugUtilsLabelEXT = @ptrCast(ggml.vulkan.common.vkGetInstanceProcAddr(instance.instance, "vkCmdBeginDebugUtilsLabelEXT"));
        instance.pfn_vkCmdEndDebugUtilsLabelEXT = @ptrCast(ggml.vulkan.common.vkGetInstanceProcAddr(instance.instance, "vkCmdEndDebugUtilsLabelEXT"));
        instance.pfn_vkCmdInsertDebugUtilsLabelEXT = @ptrCast(ggml.vulkan.common.vkGetInstanceProcAddr(instance.instance, "vkCmdInsertDebugUtilsLabelEXT"));
    }

    loggers.perf_logger_enabled = ggml.vulkan.common.envSet("GGML_VK_PERF_LOGGER");
    loggers.perf_logger_concurrent = ggml.vulkan.common.envSet("GGML_VK_PERF_LOGGER_CONCURRENT");
    loggers.enable_sync_logger = ggml.vulkan.common.envSet("GGML_VK_SYNC_LOGGER");
    loggers.memory_logger_enabled = ggml.vulkan.common.envSet("GGML_VK_MEMORY_LOGGER");
    instance.allreduce_disabled = ggml.vulkan.common.envSet("GGML_VK_DISABLE_ALLREDUCE");
    if (stdlib.env.getenv("GGML_VK_PIPELINE_STATS")) |stats| loggers.pipeline_stats_filter = std.mem.span(stats);
    if (stdlib.env.getenv("GGML_VK_PERF_LOGGER_FREQUENCY")) |freq| {
        loggers.perf_logger_frequency = try stoulTruncated(std.mem.span(freq));
    }

    ggml.vulkan.common.dispatch.loadInstance(instance.instance);

    const devices = try ggml.vulkan.common.enumeratePhysicalDevices(instance.instance);
    defer std.heap.c_allocator.free(devices);

    // Emulate behavior of CUDA_VISIBLE_DEVICES for Vulkan
    if (stdlib.env.getenv("GGML_VK_VISIBLE_DEVICES")) |devices_env| {
        const num_available_devices = devices.len;
        var it: VisibleDevices = .{ .text = std.mem.span(devices_env) };
        while (it.next()) |tmp| {
            if (tmp >= num_available_devices) {
                ggml.impl.fprintfStderr("ggml_vulkan: Invalid device index %zu in GGML_VK_VISIBLE_DEVICES.\n", .{tmp});
                return error.InvalidDeviceIndex;
            }
            instance.device_indices.append(std.heap.c_allocator, tmp) catch ggml.impl.abort("ggml_vulkan: out of memory");
        }
    } else {
        // If no vulkan devices are found, return early
        if (devices.len == 0) {
            ggml.impl.logInfo("ggml_vulkan: No devices found.\n", .{});
            return;
        }

        // Default to using all dedicated GPUs
        for (devices, 0..) |dev, i| {
            var new: DeviceIdentity = undefined;
            new.get(dev);
            const t = new.props.properties.deviceType;
            if (!((t == ggml.vulkan.common.vulkan_c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU or t == ggml.vulkan.common.vulkan_c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) and deviceIsSupported(dev))) continue;

            // Two physical devices can be one GPU under two drivers (RADV and
            // AMDVLK, say). MoltenVK may report one UUID for distinct GPUs on
            // a multi-GPU card, so two MoltenVK devices are never merged.
            const old_device: ?usize = for (instance.device_indices.items) |k| {
                var old: DeviceIdentity = undefined;
                old.get(devices[k]);
                var same_uuid = std.mem.eql(u8, &old.id.deviceUUID, &new.id.deviceUUID);
                same_uuid = same_uuid or (old.id.deviceLUIDValid != 0 and new.id.deviceLUIDValid != 0 and
                    std.mem.eql(u8, &old.id.deviceLUID, &new.id.deviceLUID));
                const both_molten_vk = new.driver.driverID == ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_MOLTENVK and old.driver.driverID == ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_MOLTENVK;
                if (same_uuid and !both_molten_vk) break k;
            } else null;

            if (old_device == null) {
                instance.device_indices.append(std.heap.c_allocator, i) catch ggml.impl.abort("ggml_vulkan: out of memory");
                continue;
            }
            // Keep one driver per GPU: splitting layers across both fails.
            var old_props = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceProperties2, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
            var old_driver = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceDriverProperties, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES);
            old_props.pNext = &old_driver;
            ggml.vulkan.common.dispatch.vkGetPhysicalDeviceProperties2.?(devices[old_device.?], &old_props);

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
                instance.device_indices.append(std.heap.c_allocator, i) catch ggml.impl.abort("ggml_vulkan: out of memory");
            }
        }

        // If no GPUs found, fall back to the first non-CPU device.
        // If only CPU devices are available, return without devices.
        if (instance.device_indices.items.len == 0) {
            for (devices, 0..) |dev, i| {
                if (ggml.vulkan.common.getProperties(dev).deviceType != ggml.vulkan.common.vulkan_c.VK_PHYSICAL_DEVICE_TYPE_CPU) {
                    instance.device_indices.append(std.heap.c_allocator, i) catch ggml.impl.abort("ggml_vulkan: out of memory");
                    break;
                }
            }
        }

        if (instance.device_indices.items.len == 0) {
            ggml.impl.logInfo("ggml_vulkan: No devices found.\n", .{});
            return;
        }
    }
    ggml.impl.logDebug("ggml_vulkan: Found %zu Vulkan devices:\n", .{instance.device_indices.items.len});

    var host_buffer_import_count: usize = 0;
    for (instance.device_indices.items, 0..) |dev_index, i| {
        const vkdev = devices[dev_index];
        const extensionprops = try ggml.vulkan.common.enumerateDeviceExtensionProperties(vkdev);
        defer std.heap.c_allocator.free(extensionprops);
        const membudget_supported = ggml.vulkan.common.hasExtension(extensionprops, ggml.vulkan.common.vulkan_c.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        const external_memory_host_supported = ggml.vulkan.common.hasExtension(extensionprops, "VK_EXT_external_memory_host");
        instance.device_supports_membudget.append(std.heap.c_allocator, membudget_supported) catch ggml.impl.abort("ggml_vulkan: out of memory");

        var host_buffer_import = false;
        if (external_memory_host_supported) {
            var props2 = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceProperties2, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
            var driver_props = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceDriverProperties, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES);
            var ext_props = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceExternalMemoryHostPropertiesEXT, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT);
            props2.pNext = &driver_props;
            driver_props.pNext = &ext_props;
            ggml.vulkan.common.dispatch.vkGetPhysicalDeviceProperties2.?(vkdev, &props2);
            host_buffer_import = ggml.vulkan.common.externalMemoryHostUsable(driver_props.driverID) and
                ggml.vulkan.common.hostBufferImportUsable(props2.properties.vendorID);
            if (host_buffer_import) {
                instance.host_import_alignment = @max(instance.host_import_alignment, @as(usize, @intCast(ext_props.minImportedHostPointerAlignment)));
                host_buffer_import_count += 1;
            }
        }
        instance.device_host_buffer_import.append(std.heap.c_allocator, host_buffer_import) catch ggml.impl.abort("ggml_vulkan: out of memory");

        try printGpuInfo(i);
    }

    instance.multi_device = host_buffer_import_count > 1 and instance.device_host_buffer_import.items[0];
}

/// `ggml_vk_instance_init` where the C++ lets its exception escape: anywhere
/// but `ggml_backend_vk_reg`, which catches it.
pub fn instanceInitOrAbort() void {
    instanceInit() catch ggml.impl.abort("ggml_vulkan: instance initialisation failed");
}

/// Ports `ggml_vk_print_gpu_info` (ggml-vulkan.cpp:5217 @b1b770d90).
fn printGpuInfo(idx: usize) InitError!void {
    ggml.impl.assert(idx < instance.device_indices.items.len, "idx < vk_instance.device_indices.size()");
    const dev_num = instance.device_indices.items[idx];
    ggml.impl.assert(initialized, "vk_instance_initialized");

    const devices = try ggml.vulkan.common.enumeratePhysicalDevices(instance.instance);
    defer std.heap.c_allocator.free(devices);
    if (dev_num >= devices.len) {
        ggml.impl.fprintfStderr("ggml_vulkan: Device with index %zu does not exist.\n", .{dev_num});
        return error.InvalidDeviceIndex;
    }

    const physical_device = devices[dev_num];
    const ext_props = try ggml.vulkan.common.enumerateDeviceExtensionProperties(physical_device);
    defer std.heap.c_allocator.free(ext_props);

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
        } else if (glslc.coopmat and std.mem.eql(u8, ext, "VK_KHR_cooperative_matrix") and !ggml.vulkan.common.envSet("GGML_VK_DISABLE_COOPMAT")) {
            coopmat_support = true;
        } else if (glslc.coopmat2 and std.mem.eql(u8, ext, "VK_NV_cooperative_matrix2") and !ggml.vulkan.common.envSet("GGML_VK_DISABLE_COOPMAT2")) {
            coopmat2_support = true;
        } else if (std.mem.eql(u8, ext, ggml.vulkan.common.vulkan_c.VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME) and !ggml.vulkan.common.envSet("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
            coopmat2_decode_vector_support = true;
        } else if (glslc.integer_dot and std.mem.eql(u8, ext, "VK_KHR_shader_integer_dot_product") and !ggml.vulkan.common.envSet("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
            integer_dot_product = true;
        } else if (glslc.bfloat16 and std.mem.eql(u8, ext, "VK_KHR_shader_bfloat16") and !ggml.vulkan.common.envSet("GGML_VK_DISABLE_BFLOAT16")) {
            bfloat16_support = true;
        } else if (glslc.float_e2m1 and std.mem.eql(u8, ext, ggml.vulkan.common.vulkan_c.VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME)) {
            ocp_microscaling_extension = true;
        } else if (glslc.float_e4m3 and std.mem.eql(u8, ext, ggml.vulkan.common.vulkan_c.VK_EXT_SHADER_FLOAT8_EXTENSION_NAME)) {
            shader_float8_extension = true;
        } else if (std.mem.eql(u8, ext, "VK_VALVE_shader_mixed_float_dot_product") and !ggml.vulkan.common.envSet("GGML_VK_DISABLE_DOT2")) {
            dot2_f16_support = true;
        }
    }

    const device_architecture = try getDeviceArchitecture(physical_device);

    const force_disable_f16 = ggml.vulkan.common.envSet("GGML_VK_DISABLE_F16");
    var fp16 = !force_disable_f16 and fp16_storage and fp16_compute;

    var props2 = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceProperties2, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
    var props3 = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceMaintenance3Properties, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES);
    var subgroup_props = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceSubgroupProperties, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES);
    var driver_props = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceDriverProperties, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES);
    var shader_integer_dot_product_props = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceShaderIntegerDotProductProperties, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES);
    props2.pNext = &props3;
    props3.pNext = &subgroup_props;
    subgroup_props.pNext = &driver_props;
    if (integer_dot_product) driver_props.pNext = &shader_integer_dot_product_props;
    ggml.vulkan.common.dispatch.vkGetPhysicalDeviceProperties2.?(physical_device, &props2);

    var device_features2 = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceFeatures2, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
    var vk11_features = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceVulkan11Features, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
    var vk12_features = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceVulkan12Features, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    device_features2.pNext = &vk11_features;
    vk11_features.pNext = &vk12_features;
    // Pointer to the last chain element
    var last_struct: *ggml.vulkan.common.vulkan_c.VkBaseOutStructure = @ptrCast(&vk12_features);

    var coopmat_features = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceCooperativeMatrixFeaturesKHR, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR);
    if (glslc.coopmat and coopmat_support) chain(&last_struct, &coopmat_features);
    var shader_integer_dot_product_features = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceShaderIntegerDotProductFeatures, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES);
    if (integer_dot_product) chain(&last_struct, &shader_integer_dot_product_features);
    var bfloat16_features = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceShaderBfloat16FeaturesKHR, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR);
    if (bfloat16_support) chain(&last_struct, &bfloat16_features);
    var coopmat2_features = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceCooperativeMatrix2FeaturesNV, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV);
    if (coopmat2_support) chain(&last_struct, &coopmat2_features);
    var coopmat2_decode_vector_features = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV);
    if (coopmat2_decode_vector_support) chain(&last_struct, &coopmat2_decode_vector_features);
    var dot2_features = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES_VALVE);
    if (dot2_f16_support) chain(&last_struct, &dot2_features);
    const fp4_compiled = glslc.float_e2m1 and glslc.float_e4m3;
    var ocp_microscaling_features = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT);
    var shader_float8_features = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceShaderFloat8FeaturesEXT, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT);
    if (fp4_compiled) {
        if (ocp_microscaling_extension) chain(&last_struct, &ocp_microscaling_features);
        if (shader_float8_extension) chain(&last_struct, &shader_float8_features);
    }

    ggml.vulkan.common.vkGetPhysicalDeviceFeatures2(physical_device, &device_features2);

    fp16 = fp16 and vk12_features.shaderFloat16 != 0;
    const bf16 = bfloat16_support and bfloat16_features.shaderBFloat16Type != 0;

    const default_subgroup_size = getSubgroupSize("", device_architecture);
    const subgroup_size: usize = if (default_subgroup_size != 0) default_subgroup_size else subgroup_props.subgroupSize;
    const uma = props2.properties.deviceType == ggml.vulkan.common.vulkan_c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    const rebar = ggml.vulkan.common.hasRebar(physical_device);

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
        !ggml.vulkan.common.envSet("GGML_VK_DISABLE_OCP_FP4");

    ggml.impl.logDebug("ggml_vulkan: %zu = %s (%s) | uma: %d | rebar: %d | fp16: %s | bf16: %d | fp4: %d | warp size: %zu | shared memory: %d | int dot: %d | matrix cores: %s\n", .{
        idx,                                                    @as([*:0]const u8, @ptrCast(&props2.properties.deviceName)),
        @as([*:0]const u8, @ptrCast(&driver_props.driverName)), @as(c_int, @intFromBool(uma)),
        @as(c_int, @intFromBool(rebar)),                        fp16_str,
        @as(c_int, @intFromBool(bf16)),                         @as(c_int, @intFromBool(fp4)),
        subgroup_size,                                          props2.properties.limits.maxComputeSharedMemorySize,
        @as(c_int, @intFromBool(integer_dot_product)),          matrix_cores,
    });

    if (props2.properties.deviceType == ggml.vulkan.common.vulkan_c.VK_PHYSICAL_DEVICE_TYPE_CPU) {
        ggml.impl.logDebug("ggml_vulkan: Warning: Device type is CPU. This is probably not the device you want.\n", .{});
    }
}

/// Appends `next` to a `pNext` chain whose last element is `last.*`, as the
/// C++'s `last_struct->pNext = ...; last_struct = ...;` pairs do.
fn chain(last: **ggml.vulkan.common.vulkan_c.VkBaseOutStructure, next: anytype) void {
    const node: *ggml.vulkan.common.vulkan_c.VkBaseOutStructure = @ptrCast(next);
    last.*.pNext = node;
    last.* = node;
}

// -----------------------------------------------------------------------------
// Per-device queries

/// Ports `ggml_vk_get_device_count` (ggml-vulkan.cpp:13408 @b1b770d90).
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

/// Ports `ggml_vk_get_device_description` (ggml-vulkan.cpp:13414 @b1b770d90).
///
/// `device` indexes the *physical* devices, not the backend's.
pub fn getPhysicalDeviceDescription(device: usize, description: [*]u8, description_size: usize) void {
    instanceInitOrAbort();
    const devices = ggml.vulkan.common.enumeratePhysicalDevices(instance.instance) catch ggml.impl.abort("ggml_vulkan: vk::Instance::enumeratePhysicalDevices failed");
    defer std.heap.c_allocator.free(devices);
    const props = ggml.vulkan.common.getProperties(devices[device]);
    copyTruncated(description, description_size, std.mem.sliceTo(&props.deviceName, 0));
}

/// The physical device behind backend device `device`, as the C++'s
/// `vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]]`.
pub fn physicalDevice(device: usize) ggml.vulkan.common.vulkan_c.VkPhysicalDevice {
    const devices = ggml.vulkan.common.enumeratePhysicalDevices(instance.instance) catch ggml.impl.abort("ggml_vulkan: vk::Instance::enumeratePhysicalDevices failed");
    defer std.heap.c_allocator.free(devices);
    return devices[instance.device_indices.items[device]];
}

/// Ports `ggml_backend_vk_get_device_memory` (ggml-vulkan.cpp:15756 @b1b770d90).
pub fn getDeviceMemory(device: c_int, free: *usize, total: *usize) void {
    ggml.impl.assert(device < @as(c_int, @intCast(instance.device_indices.items.len)), "device < (int) vk_instance.device_indices.size()");
    ggml.impl.assert(device < @as(c_int, @intCast(instance.device_supports_membudget.items.len)), "device < (int) vk_instance.device_supports_membudget.size()");
    const index: usize = @intCast(device);

    const vkdev = physicalDevice(index);
    var budgetprops = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceMemoryBudgetPropertiesEXT, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT);
    var memprops = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceMemoryProperties2, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2);
    const membudget_supported = instance.device_supports_membudget.items[index];
    const is_integrated_gpu = ggml.vulkan.common.getProperties(vkdev).deviceType == ggml.vulkan.common.vulkan_c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

    if (membudget_supported) memprops.pNext = &budgetprops;
    ggml.vulkan.common.dispatch.vkGetPhysicalDeviceMemoryProperties2.?(vkdev, &memprops);

    total.* = 0;
    free.* = 0;
    for (0..memprops.memoryProperties.memoryHeapCount) |i| {
        const heap = memprops.memoryProperties.memoryHeaps[i];
        if (is_integrated_gpu or (heap.flags & ggml.vulkan.common.vulkan_c.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            total.* +%= heap.size;
            if (membudget_supported and i < budgetprops.heapUsage.len) {
                // Driver may report heapBudget > heap.size so we add a safeguard here
                // See https://github.com/ggml-org/llama.cpp/issues/29277
                const budget = @min(budgetprops.heapBudget[i], heap.size);
                free.* +%= if (budget > budgetprops.heapUsage[i]) budget - budgetprops.heapUsage[i] else 0;
            } else {
                free.* +%= heap.size;
            }
        }
    }
}

/// Ports `ggml_backend_vk_get_device_type` (ggml-vulkan.cpp:15792 @b1b770d90).
pub fn getDeviceType(device_idx: usize) ggml.vulkan.common.vulkan_c.VkPhysicalDeviceType {
    ggml.impl.assert(device_idx < instance.device_indices.items.len, "device_idx >= 0 && device_idx < (int) vk_instance.device_indices.size()");
    const device = physicalDevice(device_idx);
    var props = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceProperties2, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
    ggml.vulkan.common.dispatch.vkGetPhysicalDeviceProperties2.?(device, &props);
    return props.properties.deviceType;
}

/// Ports `ggml_backend_vk_get_device_pci_id` (ggml-vulkan.cpp:15803 @b1b770d90).
///
/// Return: the bus id, `dddd:bb:dd.f`, or an empty string when the device
/// lacks `VK_EXT_pci_bus_info`.
pub fn getDevicePciId(device_idx: usize, buf: *[16]u8) []const u8 {
    ggml.impl.assert(device_idx < instance.device_indices.items.len, "device_idx >= 0 && device_idx < (int) vk_instance.device_indices.size()");
    const device = physicalDevice(device_idx);

    const ext_props = ggml.vulkan.common.enumerateDeviceExtensionProperties(device) catch ggml.impl.abort("ggml_vulkan: enumerateDeviceExtensionProperties failed");
    defer std.heap.c_allocator.free(ext_props);
    if (!ggml.vulkan.common.hasExtension(ext_props, "VK_EXT_pci_bus_info")) return "";

    var props = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceProperties2, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
    var pci_bus_info = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDevicePCIBusInfoPropertiesEXT, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT);
    props.pNext = &pci_bus_info;
    ggml.vulkan.common.dispatch.vkGetPhysicalDeviceProperties2.?(device, &props);

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

test "Architecture keeps the C enum's ordinals" {
    // mul_mmq_cm1.comp compares its DEVICE_ARCH spec constant with
    // `VK_ARCH_AMD_RDNA4 5u`.
    try std.testing.expectEqual(@as(u32, 5), @backingInt(Architecture.amd_rdna4));
    try std.testing.expectEqual(@as(u32, 4), @backingInt(Architecture.amd_rdna3));
    try std.testing.expectEqual(@as(u32, 10), @backingInt(Architecture.qualcomm_adreno));
}
