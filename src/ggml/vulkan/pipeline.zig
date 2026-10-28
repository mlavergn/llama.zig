//! A compute pipeline: what a shader variant is called, how it is
//! dispatched, and the Vulkan objects behind it -- and the one function
//! that creates those objects from embedded SPIR-V.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.4.0
//! (`5266f24da`). The C++ holds pipelines in `std::shared_ptr`s that live
//! until their device is destroyed; here they are heap records owned by
//! their device, never freed before it.

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const common = m.common;
const vk = common.vk;
const d = &common.dispatch;
const allocator = common.allocator;
const Device = m.device.Device;

/// Ports `MAX_PARAMETER_COUNT` and `MAX_FUSED_ADDS` (ggml-vulkan.cpp:216, 218 @5266f24da).
pub const max_parameter_count = 12;
pub const max_fused_adds = max_parameter_count - 3;

/// Ports `vk_pipeline_struct` (ggml-vulkan.cpp:222 @5266f24da).
///
/// `initialized`, `compile_pending`, `all_pipelines` membership and the
/// `next` link are guarded by the device's `compile_mutex`; `compiled` is
/// read without it, so it is atomic, as in the C++.
pub const Pipeline = struct {
    name: [:0]const u8 = "",
    shader_module: vk.VkShaderModule = null,
    layout: vk.VkPipelineLayout = null,
    pipeline: vk.VkPipeline = null,
    push_constant_size: u32 = 0,
    parameter_count: u32 = 0,
    wg_denoms: [3]u32 = .{ 0, 0, 0 },
    @"align": u32 = 0,
    // true if fields have been set by ggml_vk_create_pipeline
    initialized: bool = false,
    // true while a compile is in flight, used to dedupe concurrent claims.
    compile_pending: bool = false,
    // set to true when the shader has been compiled
    compiled: std.atomic.Value(bool) = .init(false),
    // number of registers used, extracted from pipeline executable properties
    register_count: u32 = 0,
    is_64b_indexing: bool = false,
    // linked list of pipelines for multiple compilation variants.
    next: ?*Pipeline = null,

    /// `std::make_shared<vk_pipeline_struct>()`.
    pub fn create() *Pipeline {
        const p = allocator.create(Pipeline) catch impl.abort("ggml_vulkan: out of memory");
        p.* = .{};
        return p;
    }
};

/// Ports `vk_matmul_pipeline_struct` (ggml-vulkan.cpp:253 @5266f24da): the
/// three tile sizes, each unaligned and aligned.
pub const MatmulPipeline = struct {
    l: ?*Pipeline = null,
    m: ?*Pipeline = null,
    s: ?*Pipeline = null,
    a_l: ?*Pipeline = null,
    a_m: ?*Pipeline = null,
    a_s: ?*Pipeline = null,

    /// Ports `vk_matmul_pipeline_struct`'s `is_empty` (ggml-vulkan.cpp:259 @5266f24da):
    /// true when all unaligned pipelines are null.
    pub fn isEmpty(self: *const MatmulPipeline) bool {
        return self.l == null and self.m == null and self.s == null;
    }

    /// `std::make_shared<vk_matmul_pipeline_struct>()`.
    pub fn create() *MatmulPipeline {
        const p = allocator.create(MatmulPipeline) catch impl.abort("ggml_vulkan: out of memory");
        p.* = .{};
        return p;
    }
};

/// Ports `vk_matmul_pipeline2` (ggml-vulkan.cpp:265 @5266f24da): one matmul
/// pipeline per accumulator precision, both always allocated.
pub const MatmulPipeline2 = struct {
    f32acc: *MatmulPipeline,
    f16acc: *MatmulPipeline,

    pub fn create() MatmulPipeline2 {
        return .{ .f16acc = MatmulPipeline.create(), .f32acc = MatmulPipeline.create() };
    }
};

// SPIR-V passes

const spv = vk;
const op_code_mask: u32 = spv.SpvOpCodeMask;
const word_count_shift: u5 = spv.SpvWordCountShift;

/// Ports `kSpvOpCooperativeMatrixLoadTensorNV`,
/// `kSpvCapabilityCooperativeMatrixDecodeVectorNV` and
/// `kSpvTensorAddressingDecodeVectorFuncBit` (ggml-vulkan.cpp:2709, 2710, 2711 @5266f24da).
const op_cooperative_matrix_load_tensor_nv: u32 = 5367;
const capability_cooperative_matrix_decode_vector_nv: u32 = 5447;
const tensor_addressing_decode_vector_func_bit: u32 = 0x4;

/// An instruction's opcode and word count.
fn opOf(word: u32) u32 {
    return word & op_code_mask;
}
fn wcOf(word: u32) u32 {
    return word >> word_count_shift;
}

/// Whether the literal string operand at `words` reads `s`, as the C++'s
/// `strcmp` over the instruction's words does.
fn literalEq(words: []const u32, s: []const u8) bool {
    const bytes = std.mem.sliceAsBytes(words);
    const len = std.mem.indexOfScalar(u8, bytes, 0) orelse bytes.len;
    return std.mem.eql(u8, bytes[0..len], s);
}

/// Ports `ggml_vk_strip_decode_vector` (ggml-vulkan.cpp:2719 @5266f24da):
/// removes `SPV_NV_cooperative_matrix_decode_vector` from a module so a
/// driver with only `SPV_NV_cooperative_matrix2` can load it.
///
/// Return: the stripped copy, owned by the caller, when the module used the
/// extension; null otherwise.
fn stripDecodeVector(code: []const u32) ?[]u32 {
    const ext = "SPV_NV_cooperative_matrix_decode_vector";
    if (code.len < 5) return null;

    var uses_decode_vector = false;
    var pos: usize = 5;
    while (pos < code.len) {
        const wc = wcOf(code[pos]);
        const op = opOf(code[pos]);
        impl.assert(wc > 0 and pos + wc <= code.len, "wc > 0 && pos + wc <= word_count");
        if (op == spv.SpvOpExtension and wc >= 2 and literalEq(code[pos + 1 .. pos + wc], ext)) {
            uses_decode_vector = true;
            break;
        }
        pos += wc;
    }
    if (!uses_decode_vector) return null;

    var out: std.ArrayList(u32) = .empty;
    out.ensureTotalCapacity(allocator, code.len) catch impl.abort("ggml_vulkan: out of memory");
    var run_start: usize = 0;
    const flush = struct {
        fn run(o: *std.ArrayList(u32), src: []const u32, start: usize, up_to: usize) void {
            if (up_to > start) o.appendSliceAssumeCapacity(src[start..up_to]);
        }
    }.run;

    pos = 5;
    while (pos < code.len) {
        const wc = wcOf(code[pos]);
        const op = opOf(code[pos]);
        impl.assert(wc > 0 and pos + wc <= code.len, "wc > 0 && pos + wc <= word_count");

        if (op == spv.SpvOpExtension and wc >= 2 and literalEq(code[pos + 1 .. pos + wc], ext)) {
            flush(&out, code, run_start, pos);
            pos += wc;
            run_start = pos;
            continue;
        }
        if (op == spv.SpvOpCapability and wc == 2 and code[pos + 1] == capability_cooperative_matrix_decode_vector_nv) {
            flush(&out, code, run_start, pos);
            pos += wc;
            run_start = pos;
            continue;
        }
        if (op == op_cooperative_matrix_load_tensor_nv) {
            impl.assert(wc >= 8, "wc >= 8");
            const mem_mask = code[pos + 6];
            var cur = pos + 7;
            // Each of these MemoryAccess bits (when set) carries one trailing operand.
            cur += @intFromBool(mem_mask & 0x2 != 0); // Aligned
            cur += @intFromBool(mem_mask & 0x8 != 0); // MakePointerAvailable
            cur += @intFromBool(mem_mask & 0x10 != 0); // MakePointerVisible
            cur += @intFromBool(mem_mask & 0x10000 != 0); // AliasScopeINTELMask
            cur += @intFromBool(mem_mask & 0x20000 != 0); // NoAliasINTELMask
            impl.assert(cur < pos + wc, "cur < pos + wc");

            const ta_mask = code[cur];
            if (ta_mask & tensor_addressing_decode_vector_func_bit == 0) {
                pos += wc;
                continue;
            }
            flush(&out, code, run_start, pos);

            const inst_start = out.items.len;
            const pre_n = cur - pos;
            out.appendSliceAssumeCapacity(code[pos .. pos + pre_n]);
            out.appendAssumeCapacity(ta_mask & ~tensor_addressing_decode_vector_func_bit);
            const keep_ta_extras: usize = @as(usize, @intFromBool(ta_mask & 0x1 != 0)) + @intFromBool(ta_mask & 0x2 != 0);
            if (keep_ta_extras != 0) out.appendSliceAssumeCapacity(code[cur + 1 .. cur + 1 + keep_ta_extras]);
            impl.assert(wc == pre_n + 1 + keep_ta_extras + 1, "wc == pre_n + 1 + keep_ta_extras + 1");

            const new_wc = wc - 1;
            out.items[inst_start] = (new_wc << word_count_shift) | op;
            pos += wc;
            run_start = pos;
            continue;
        }
        pos += wc;
    }
    // The header and anything before the first dropped instruction are in
    // the first run: `run_start` began at 0.
    flush(&out, code, run_start, code.len);
    return out.toOwnedSlice(allocator) catch impl.abort("ggml_vulkan: out of memory");
}

/// Ports `ggml_vk_roll_bk_loop` (ggml-vulkan.cpp:2847 @5266f24da): swaps the
/// unroll hint of `mul_mm.comp`'s BK loop for `DontUnroll`.
///
/// Return: the patched copy, owned by the caller, or null when no BK loop
/// was found.
fn rollBkLoop(code: []const u32) ?[]u32 {
    if (code.len < 5) return null;
    const Loop = struct { header: usize, end: usize, control: u32 };
    var loops: std.ArrayList(Loop) = .empty;
    defer loops.deinit(allocator);

    var pos: usize = 5;
    while (pos < code.len) {
        const wc = wcOf(code[pos]);
        const op = opOf(code[pos]);
        if (wc == 0 or pos + wc > code.len) return null;
        if (op == spv.SpvOpLoopMerge and wc >= 4) {
            loops.append(allocator, .{ .header = pos, .end = 0, .control = code[pos + 3] }) catch impl.abort("ggml_vulkan: out of memory");
        }
        if (op == spv.SpvOpLabel and wc >= 2) {
            for (loops.items) |*l| {
                if (l.end == 0 and code[l.header + 1] == code[pos + 1]) l.end = pos;
            }
        }
        pos += wc;
    }

    const encloses = struct {
        fn f(a: Loop, b: Loop) bool {
            return a.header < b.header and b.header < a.end;
        }
    }.f;

    var bk: ?Loop = null;
    for (loops.items) |h| {
        if (h.control != spv.SpvLoopControlUnrollMask) continue;
        var parent: ?Loop = null;
        var has_child = false;
        for (loops.items) |g| {
            if (encloses(g, h) and (parent == null or g.header > parent.?.header)) parent = g;
            if (encloses(h, g)) has_child = true;
        }
        if (parent != null and parent.?.control == spv.SpvLoopControlMaskNone and has_child and
            (bk == null or h.header > bk.?.header))
        {
            bk = h;
        }
    }
    const b = bk orelse return null;
    const out = allocator.dupe(u32, code) catch impl.abort("ggml_vulkan: out of memory");
    out[b.header + 3] = spv.SpvLoopControlDontUnrollMask;
    return out;
}

/// The float-controls patch at the top of `ggml_vk_create_pipeline_func`:
/// enables the FP16 rounding and denorm modes the device supports, so one
/// shader variant serves both.
fn patchFloatControls(device: *const Device, spv_words: []const u32) []u32 {
    var spirv: std.ArrayList(u32) = .empty;
    spirv.appendSlice(allocator, spv_words) catch impl.abort("ggml_vulkan: out of memory");

    var pos: usize = 5; // skip header
    var cap_insert_pos = pos;
    var ext_insert_pos = pos;
    var exec_insert_pos = pos;
    var entry_point_id: u32 = 0;
    while (pos < spirv.items.len) {
        const opcode = opOf(spirv.items[pos]);
        const len = wcOf(spirv.items[pos]);
        if (len == 0) break;
        if (opcode == spv.SpvOpCapability) {
            cap_insert_pos = pos + len;
            ext_insert_pos = pos + len;
        } else if (opcode == spv.SpvOpExtension) {
            ext_insert_pos = pos + len;
        } else if (opcode == spv.SpvOpEntryPoint) {
            entry_point_id = spirv.items[pos + 2];
            exec_insert_pos = pos + len;
        } else if (opcode == spv.SpvOpExecutionMode or opcode == spv.SpvOpExecutionModeId) {
            exec_insert_pos = pos + len;
        } else if (entry_point_id != 0) {
            break;
        }
        pos += len;
    }

    const insert = struct {
        fn f(list: *std.ArrayList(u32), at: usize, words: []const u32) void {
            list.insertSlice(allocator, at, words) catch impl.abort("ggml_vulkan: out of memory");
        }
    }.f;
    const exec_mode_op = (4 << word_count_shift) | @as(u32, spv.SpvOpExecutionMode);
    // Insert from latest position first so earlier indices stay valid.
    if (device.float_controls_rte_fp16) {
        insert(&spirv, exec_insert_pos, &.{ exec_mode_op, entry_point_id, spv.SpvExecutionModeRoundingModeRTE, 16 });
    }
    if (device.float_controls_denorm_preserve_fp16) {
        insert(&spirv, exec_insert_pos, &.{ exec_mode_op, entry_point_id, spv.SpvExecutionModeDenormPreserve, 16 });
    }

    // OpExtension "SPV_KHR_float_controls"
    const ext_str = "SPV_KHR_float_controls";
    const ext_str_words = (ext_str.len + 1 + 3) / 4;
    var extension: [1 + ext_str_words]u32 = @splat(0);
    extension[0] = ((1 + ext_str_words) << word_count_shift) | @as(u32, spv.SpvOpExtension);
    @memcpy(std.mem.sliceAsBytes(extension[1..])[0..ext_str.len], ext_str);
    insert(&spirv, ext_insert_pos, &extension);

    const cap_op = (2 << word_count_shift) | @as(u32, spv.SpvOpCapability);
    if (device.float_controls_rte_fp16) insert(&spirv, cap_insert_pos, &.{ cap_op, spv.SpvCapabilityRoundingModeRTE });
    if (device.float_controls_denorm_preserve_fp16) insert(&spirv, cap_insert_pos, &.{ cap_op, spv.SpvCapabilityDenormPreserve });

    return spirv.toOwnedSlice(allocator) catch impl.abort("ggml_vulkan: out of memory");
}

/// What `ggml_vk_load_shaders` hands to the compile: `CompileTask`
/// (ggml-vulkan.cpp:4231 @5266f24da) without the pipeline.
pub const CompileArgs = struct {
    spv: []const u8,
    entrypoint: [:0]const u8,
    parameter_count: u32,
    wg_denoms: [3]u32,
    specialization_constants: []const u32,
    disable_robustness: bool,
    require_full_subgroups: bool,
    required_subgroup_size: u32,
};

/// Ports `ggml_vk_create_pipeline_func` (ggml-vulkan.cpp:2918 @5266f24da):
/// creates the shader module, layout and compute pipeline for `pipeline`,
/// records it in the device's pipeline list and marks it compiled.
///
/// A failed create throws in the C++, after printing which pipeline, and
/// nothing catches it; the port prints the same and aborts.
pub fn createPipelineFunc(device: *Device, pipeline: *Pipeline, args: CompileArgs) void {
    impl.assert(args.parameter_count > 0, "parameter_count > 0");
    impl.assert(args.parameter_count <= max_parameter_count, "parameter_count <= MAX_PARAMETER_COUNT");
    impl.assert(args.wg_denoms[0] > 0 and args.wg_denoms[1] > 0 and args.wg_denoms[2] > 0, "wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0");

    // The embedded variants start 4-aligned in the blob
    // (build/vulkan_shaders.zig), so they can be read as words in place.
    const spv_words: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, args.spv[0 .. args.spv.len / 4 * 4]));
    var code: []const u32 = spv_words;
    var owned: ?[]u32 = null;
    defer if (owned) |o| allocator.free(o);

    if (device.float_controls_rte_fp16 or device.float_controls_denorm_preserve_fp16) {
        owned = patchFloatControls(device, spv_words);
        code = owned.?;
    }

    if (comptime m.instance.glslc.coopmat2_decode_vector) {
        if (device.coopmat2 and !device.coopmat2_decode_vector) {
            if (stripDecodeVector(code)) |stripped| {
                if (owned) |o| allocator.free(o);
                owned = stripped;
                code = stripped;
            }
        }
    }

    if (comptime vk.VK_HEADER_VERSION >= 287) {
        // Roll the mul_mm BK loop on Asahi Linux. Skip bf16 and the mul_mmq pipelines.
        if (device.driver_id == vk.VK_DRIVER_ID_MESA_HONEYKRISP and
            std.mem.startsWith(u8, pipeline.name, "matmul") and
            std.mem.indexOf(u8, pipeline.name, "bf16") == null and
            std.mem.indexOf(u8, pipeline.name, "q8_1") == null)
        {
            if (rollBkLoop(code)) |rolled| {
                if (owned) |o| allocator.free(o);
                owned = rolled;
                code = rolled;
            }
        }
    }

    var smci = common.structure(vk.VkShaderModuleCreateInfo, vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
    // The C++ hands the unpatched module over by its byte size.
    smci.codeSize = if (owned == null) args.spv.len else code.len * 4;
    smci.pCode = code.ptr;
    common.checkOrAbort(d.vkCreateShaderModule.?(device.device, &smci, null, &pipeline.shader_module), "vk::Device::createShaderModule");

    const pcr: vk.VkPushConstantRange = .{ .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = pipeline.push_constant_size };
    var plci = common.structure(vk.VkPipelineLayoutCreateInfo, vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &device.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    common.checkOrAbort(d.vkCreatePipelineLayout.?(device.device, &plci, null, &pipeline.layout), "vk::Device::createPipelineLayout");

    const entries = allocator.alloc(vk.VkSpecializationMapEntry, args.specialization_constants.len) catch impl.abort("ggml_vulkan: out of memory");
    defer allocator.free(entries);
    for (entries, 0..) |*e, i| {
        e.* = .{ .constantID = @intCast(i), .offset = @intCast(i * @sizeOf(u32)), .size = @sizeOf(u32) };
    }
    const specialization_info: vk.VkSpecializationInfo = .{
        .mapEntryCount = @intCast(entries.len),
        .pMapEntries = entries.ptr,
        .dataSize = args.specialization_constants.len * @sizeOf(u32),
        .pData = args.specialization_constants.ptr,
    };

    var stage_flags: vk.VkPipelineShaderStageCreateFlags = 0;
    if (device.subgroup_require_full_support and args.require_full_subgroups) {
        stage_flags |= vk.VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
    }
    var stage = common.structure(vk.VkPipelineShaderStageCreateInfo, vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
    stage.flags = stage_flags;
    stage.stage = vk.VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = pipeline.shader_module;
    stage.pName = args.entrypoint.ptr;
    stage.pSpecializationInfo = &specialization_info;

    var rss = common.structure(vk.VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT, vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO);
    rss.requiredSubgroupSize = args.required_subgroup_size;
    if (device.subgroup_size_control and args.required_subgroup_size > 0) {
        impl.assert(device.subgroup_min_size <= args.required_subgroup_size and args.required_subgroup_size <= device.subgroup_max_size, "device->subgroup_min_size <= required_subgroup_size && required_subgroup_size <= device->subgroup_max_size");
        stage.pNext = &rss;
    }

    var cpci = common.structure(vk.VkComputePipelineCreateInfo, vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);
    cpci.flags = if (device.pipeline_executable_properties_support) vk.VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR else 0;
    cpci.stage = stage;
    cpci.layout = pipeline.layout;

    var rci = common.structure(vk.VkPipelineRobustnessCreateInfoEXT, vk.VK_STRUCTURE_TYPE_PIPELINE_ROBUSTNESS_CREATE_INFO);
    if (device.pipeline_robustness and args.disable_robustness) {
        rci.storageBuffers = vk.VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED;
        rci.uniformBuffers = vk.VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED;
        cpci.pNext = &rci;
    }

    var flags2 = common.structure(vk.VkPipelineCreateFlags2CreateInfo, vk.VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO);
    if (comptime @hasDecl(vk, "VK_EXT_shader_64bit_indexing")) {
        if (pipeline.is_64b_indexing) {
            flags2.flags = vk.VK_PIPELINE_CREATE_2_64_BIT_INDEXING_BIT_EXT;
            if (device.pipeline_executable_properties_support) {
                flags2.flags |= vk.VK_PIPELINE_CREATE_2_CAPTURE_STATISTICS_BIT_KHR;
            }
            flags2.pNext = cpci.pNext;
            cpci.pNext = &flags2;
        }
    }

    // `vk::Device::createComputePipeline` accepts VK_PIPELINE_COMPILE_REQUIRED
    // as well as success.
    const result = d.vkCreateComputePipelines.?(device.device, null, 1, &cpci, null, &pipeline.pipeline);
    if (result != vk.VK_SUCCESS and result != vk.VK_PIPELINE_COMPILE_REQUIRED) {
        impl.fprintfStderr("ggml_vulkan: Compute pipeline creation failed for %s\n", .{pipeline.name.ptr});
        impl.fprintfStderr("ggml_vulkan: vk::Device::createComputePipeline: %s\n", .{common.resultString(result).ptr});
        impl.abort("ggml_vulkan: compute pipeline creation failed");
    }

    if (m.instance.instance.debug_utils_support) {
        var duoni = common.structure(vk.VkDebugUtilsObjectNameInfoEXT, vk.VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT);
        duoni.objectType = vk.VK_OBJECT_TYPE_PIPELINE;
        duoni.pObjectName = pipeline.name.ptr;
        duoni.objectHandle = @intFromPtr(pipeline.pipeline);
        _ = m.instance.instance.pfn_vkSetDebugUtilsObjectNameEXT.?(device.device, &duoni);
    }

    if (device.pipeline_executable_properties_support) {
        readStatistics(device, pipeline);
    }

    _ = std.c.pthread_mutex_lock(&device.compile_mutex);
    device.all_pipelines.append(allocator, pipeline) catch impl.abort("ggml_vulkan: out of memory");
    pipeline.compiled.store(true, .seq_cst);
    pipeline.compile_pending = false;
    _ = std.c.pthread_mutex_unlock(&device.compile_mutex);
    _ = std.c.pthread_cond_broadcast(&device.compile_cv);
}

/// The pipeline-statistics tail of `ggml_vk_create_pipeline_func`: prints
/// every statistic of a pipeline `GGML_VK_PIPELINE_STATS` names, and keeps
/// the register count NVIDIA reports.
fn readStatistics(device: *Device, pipeline: *Pipeline) void {
    var info = common.structure(vk.VkPipelineExecutableInfoKHR, vk.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR);
    info.pipeline = pipeline.pipeline;
    const stats = common.enumerateWith(vk.VkPipelineExecutableStatisticKHR, vk.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR, "vk::Device::getPipelineExecutableStatisticsKHR", d.vkGetPipelineExecutableStatisticsKHR.?, .{ device.device, &info }) catch
        common.abortLastError();
    defer allocator.free(stats);

    const print_stats = if (m.instance.loggers.pipeline_stats_filter) |filter|
        filter.len != 0 and std.mem.indexOf(u8, pipeline.name, filter) != null
    else
        false;
    if (print_stats) impl.fprintfStderr("ggml_vulkan: pipeline stats for %s:\n", .{pipeline.name.ptr});

    for (stats) |*s| {
        if (print_stats) {
            const sname: [*:0]const u8 = @ptrCast(&s.name);
            switch (s.format) {
                vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR => impl.fprintfStderr("ggml_vulkan:   %s: %s\n", .{ sname, @as([*:0]const u8, if (s.value.b32 != 0) "true" else "false") }),
                vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR => impl.fprintfStderr("ggml_vulkan:   %s: %lld\n", .{ sname, @as(c_longlong, s.value.i64) }),
                vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR => impl.fprintfStderr("ggml_vulkan:   %s: %llu\n", .{ sname, @as(c_ulonglong, s.value.u64) }),
                // `std::cerr << double` is `%g`.
                vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR => impl.fprintfStderr("ggml_vulkan:   %s: %g\n", .{ sname, s.value.f64 }),
                else => impl.fprintfStderr("ggml_vulkan:   %s: \n", .{sname}),
            }
        }
        // "Register Count" is reported by NVIDIA drivers.
        if (std.mem.eql(u8, std.mem.sliceTo(&s.name, 0), "Register Count")) {
            pipeline.register_count = @truncate(s.value.u64);
        }
    }
}

/// Ports `ggml_vk_destroy_pipeline` (ggml-vulkan.cpp:3171 @5266f24da).
pub fn destroyPipeline(device: vk.VkDevice, pipeline: *Pipeline) void {
    d.vkDestroyPipelineLayout.?(device, pipeline.layout, null);
    d.vkDestroyShaderModule.?(device, pipeline.shader_module, null);
    d.vkDestroyPipeline.?(device, pipeline.pipeline, null);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "rollBkLoop takes the last unrolled loop with a child under a hint-free parent" {
    const L = struct {
        fn merge(id: u32, control: u32) [4]u32 {
            return .{ (4 << word_count_shift) | @as(u32, spv.SpvOpLoopMerge), id, 0, control };
        }
        fn label(id: u32) [2]u32 {
            return .{ (2 << word_count_shift) | @as(u32, spv.SpvOpLabel), id };
        }
    };
    // K-tile loop (no hint) { BK loop (unroll) { inner (unroll) } }
    const code = [_]u32{ 0, 0, 0, 0, 0 } ++ L.merge(1, 0) ++ L.merge(2, 1) ++ L.merge(3, 1) ++ L.label(3) ++ L.label(2) ++ L.label(1);
    const out = rollBkLoop(&code).?;
    defer allocator.free(out);
    try std.testing.expectEqual(@as(u32, spv.SpvLoopControlDontUnrollMask), out[5 + 4 + 3]);
    try std.testing.expectEqual(@as(u32, 1), out[5 + 8 + 3]);
}
