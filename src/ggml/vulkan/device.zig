//! The logical device: what `ggml_vk_get_device` learns about a physical
//! device, the `VkDevice` and queues it creates, and every pipeline slot
//! `ggml_vk_load_shaders` fills.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.3.0
//! (`c1d0e7a00`). A device is created once per index and lives for the
//! process, as the C++'s `vk_instance.devices[]` shared pointers do; the
//! C++ destroys them from a static destructor at exit, which the port does
//! not reproduce.

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const common = m.common;
const vk = common.vk;
const d = &common.dispatch;
const allocator = common.allocator;
const Pipeline = m.pipeline.Pipeline;
const MatmulPipeline = m.pipeline.MatmulPipeline;
const MatmulPipeline2 = m.pipeline.MatmulPipeline2;
const Architecture = m.instance.Architecture;
const glslc = m.instance.glslc;

const type_count = c.GGML_TYPE_COUNT;

/// Ports `mul_mat_vec_max_cols` and `p021_max_gqa_ratio` (ggml-vulkan.cpp:389, 390 @c1d0e7a00).
pub const mul_mat_vec_max_cols = 8;
pub const p021_max_gqa_ratio = 8;

/// Ports `dmmv_wg_sizes` (ggml-vulkan.cpp:546 @c1d0e7a00).
pub const DmmvWgSize = enum(u32) { subgroup, large };
pub const dmmv_wg_size_count = 2;

/// Ports `FaCodePath` (ggml-vulkan.cpp:552 @c1d0e7a00).
pub const FaCodePath = enum(u32) { scalar, coopmat1, coopmat2 };

/// Ports `vk_fa_pipeline_state` (ggml-vulkan.cpp:558 @c1d0e7a00). The C++
/// keys a `std::map` on it; only lookup and iteration are needed, so the
/// port hashes it.
pub const FaPipelineState = struct {
    HSK: u32,
    HSV: u32,
    Br: u32,
    Bc: u32,
    D_split: u32,
    row_split: u32,
    shmem_staging: bool,
    path: FaCodePath,
    workgroup_size: u32,
    subgroup_size: u32,
    aligned: bool,
    f32acc: bool,
    flags: u32,
    limit_occupancy_shmem: u32,
    k_type: c.ggml_type,
    v_type: c.ggml_type,
};

/// Ports `vk_conv2d_pipeline_state` (ggml-vulkan.cpp:578 @c1d0e7a00).
pub const Conv2dPipelineState = struct { s0: u32, s1: u32, p0: u32, p1: u32, d0: u32, d1: u32, KW: u32, KH: u32, aligned: u32 };

/// Ports `vk_conv3d_pipeline_state` (ggml-vulkan.cpp:592 @c1d0e7a00).
pub const Conv3dPipelineState = struct { s0: u32, s1: u32, s2: u32, p0: u32, p1: u32, p2: u32, d0: u32, d1: u32, d2: u32, KW: u32, KH: u32, KD: u32, aligned: u32 };

/// Ports `vk_solve_tri_pipeline_state` (ggml-vulkan.cpp:606 @c1d0e7a00).
pub const SolveTriPipelineState = struct { N: u32, K: u32 };

/// A dynamic pipeline map: `std::map<State, vk_pipeline>`.
pub fn PipelineMap(comptime K: type) type {
    return std.AutoArrayHashMapUnmanaged(K, ?*Pipeline);
}

/// Ports `num_argsort_pipelines`, `num_topk_moe_pipelines` and
/// `num_topk_pipelines` (ggml-vulkan.cpp:626, 627, 628 @c1d0e7a00).
pub const num_argsort_pipelines = 11;
pub const num_topk_moe_pipelines = 10;
pub const num_topk_pipelines = 11;

/// Ports `CONV_SHAPE_COUNT` (ggml-vulkan.cpp:529 @c1d0e7a00).
pub const conv_shape_count = 4;

/// Ports `vk_command_buffer` (ggml-vulkan.cpp:289 @c1d0e7a00).
pub const CommandBuffer = struct {
    buf: vk.VkCommandBuffer = null,
    use_counter: u64 = 0,
    in_use: bool = false,
};

/// Ports `vk_command_pool` (ggml-vulkan.cpp:297 @c1d0e7a00). The C++ keeps
/// its buffers in a `std::deque` so pointers to them survive growth; here
/// each is its own allocation.
pub const CommandPool = struct {
    pool: vk.VkCommandPool = null,
    cmd_buffers: std.ArrayList(*CommandBuffer) = .empty,
    q: ?*Queue = null,

    /// Ports `vk_command_pool`'s `init` (ggml-vulkan.cpp:1144 @c1d0e7a00).
    pub fn init(self: *CommandPool, device: *Device, q: *Queue) void {
        self.clearBuffers();
        self.q = q;
        var info = common.structure(vk.VkCommandPoolCreateInfo, vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
        info.flags = vk.VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        info.queueFamilyIndex = q.queue_family_index;
        common.checkOrAbort(d.vkCreateCommandPool.?(device.device, &info, null, &self.pool), "vk::Device::createCommandPool");
    }

    /// Ports `vk_command_pool`'s `destroy` (ggml-vulkan.cpp:1154 @c1d0e7a00).
    pub fn destroy(self: *CommandPool, device: vk.VkDevice) void {
        d.vkDestroyCommandPool.?(device, self.pool, null);
        self.pool = null;
        self.clearBuffers();
    }

    fn clearBuffers(self: *CommandPool) void {
        for (self.cmd_buffers.items) |b| allocator.destroy(b);
        self.cmd_buffers.clearRetainingCapacity();
    }

    /// Ports `vk_command_pool`'s `buffers_in_use` (ggml-vulkan.cpp:308 @c1d0e7a00).
    pub fn buffersInUse(self: *const CommandPool) usize {
        var n: usize = 0;
        for (self.cmd_buffers.items) |b| n += @intFromBool(b.in_use);
        return n;
    }
};

/// Ports `vk_queue_handle` with its two implementations,
/// `vk_queue_handle_synchronized` and `vk_queue_handle_unsynchronized`
/// (ggml-vulkan.cpp:318, 327, 344 @c1d0e7a00): a queue that serialises its
/// own submissions unless the driver synchronises it internally.
pub const QueueHandle = struct {
    queue: vk.VkQueue = null,
    device: *Device,
    synchronized: bool,
    mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,

    pub fn lock(self: *QueueHandle) void {
        if (self.synchronized) _ = std.c.pthread_mutex_lock(&self.mutex);
    }

    pub fn unlock(self: *QueueHandle) void {
        if (self.synchronized) _ = std.c.pthread_mutex_unlock(&self.mutex);
    }
};

/// Ports `vk_queue` (ggml-vulkan.cpp:359 @c1d0e7a00).
pub const Queue = struct {
    queue_family_index: u32 = 0,
    handle: *QueueHandle,
    cmd_pool: CommandPool = .{},
    stage_flags: vk.VkPipelineStageFlags = 0,
    transfer_only: bool = false,
};

/// `std::recursive_mutex`: a mutex the holding thread may lock again.
/// libc++ builds it on a `PTHREAD_MUTEX_RECURSIVE` mutex, whose attribute
/// type and constant differ between C libraries; this is the same contract
/// over a plain one.
pub const RecursiveMutex = struct {
    m: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    owner: std.atomic.Value(std.Thread.Id) = .init(0),
    depth: u32 = 0,

    pub fn lock(self: *RecursiveMutex) void {
        const me = std.Thread.getCurrentId();
        if (self.owner.load(.monotonic) == me) {
            self.depth += 1;
            return;
        }
        _ = std.c.pthread_mutex_lock(&self.m);
        self.owner.store(me, .monotonic);
        self.depth = 1;
    }

    pub fn unlock(self: *RecursiveMutex) void {
        self.depth -= 1;
        if (self.depth != 0) return;
        self.owner.store(0, .monotonic);
        _ = std.c.pthread_mutex_unlock(&self.m);
    }
};

/// Ports `vk_device_struct` (ggml-vulkan.cpp:771 @c1d0e7a00).
pub const Device = struct {
    mutex: RecursiveMutex = .{},
    pinned_memory_mutex: std.c.pthread_rwlock_t = .{},

    // Guards compile_pending, all_pipelines, and the dynamic pipeline maps.
    // Lock order is device->mutex -> compile_mutex, never the reverse.
    compile_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    compile_cv: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER,

    physical_device: vk.VkPhysicalDevice = null,
    properties: vk.VkPhysicalDeviceProperties = std.mem.zeroes(vk.VkPhysicalDeviceProperties),
    name: [:0]const u8 = "",
    max_memory_allocation_size: u64 = 0,
    max_buffer_size: u64 = 0,
    suballocation_block_size: u64 = 0,
    min_imported_host_pointer_alignment: u64 = 0,
    external_memory_host: bool = false,
    fp16: bool = false,
    bf16: bool = false,
    pipeline_robustness: bool = false,
    memory_priority: bool = false,
    device: vk.VkDevice = null,
    vendor_id: u32 = 0,
    driver_id: vk.VkDriverId = 0,
    architecture: Architecture = .other,
    compute_queue: ?*Queue = null,
    transfer_queue: ?*Queue = null,
    single_queue: bool = false,
    support_async: bool = false,
    async_use_transfer_queue: bool = false,
    has_internally_synchronized_queues: bool = false,
    subgroup_size: u32 = 0,
    subgroup_size_log2: u32 = 0,
    shader_core_count: u32 = 0,
    uma: bool = false,
    prefer_host_memory: bool = false,
    float_controls_rte_fp16: bool = false,
    float_controls_denorm_preserve_fp16: bool = false,
    subgroup_basic: bool = false,
    subgroup_arithmetic: bool = false,
    subgroup_shuffle: bool = false,
    subgroup_ballot: bool = false,
    subgroup_clustered: bool = false,
    subgroup_vote: bool = false,
    multi_add: bool = false,
    shader_int64: bool = false,
    buffer_device_address: bool = false,
    vulkan_memory_model: bool = false,

    add_rms_fusion: bool = false,
    partials_binding_alignment: u32 = 0,
    max_nodes_per_submit: u32 = 0,

    shader_64b_indexing: bool = false,

    integer_dot_product: bool = false,
    // 0: default, 1: force mmvq, -1: disable mmvq
    mmvq_mode: i32 = 0,

    subgroup_size_control: bool = false,
    subgroup_min_size: u32 = 0,
    subgroup_max_size: u32 = 0,
    subgroup_require_full_support: bool = false,

    // floor(log2(maxComputeWorkGroupInvocations))
    max_workgroup_size_log2: u32 = 0,

    coopmat_support: bool = false,
    coopmat_acc_f32_support: bool = false,
    coopmat_acc_f16_support: bool = false,
    coopmat_bf16_support: bool = false,
    coopmat_support_16x16x16_f16acc: bool = false,
    coopmat_support_16x16x16_f32acc: bool = false,
    coopmat1_fa_support: bool = false,
    coopmat_m: u32 = 0,
    coopmat_n: u32 = 0,
    coopmat_k: u32 = 0,

    coopmat_int_support: bool = false,
    coopmat_int_m: u32 = 0,
    coopmat_int_n: u32 = 0,
    coopmat_int_k: u32 = 0,

    coopmat2: bool = false,
    coopmat2_bf16_support: bool = false,
    coopmat2_decode_vector: bool = false,

    dot2_f16: bool = false,
    ocp_fp4: bool = false,

    pipeline_executable_properties_support: bool = false,

    device_fault: bool = false,
    pfn_vkGetDeviceFaultInfoEXT: vk.PFN_vkGetDeviceFaultInfoEXT = null,

    serialize_submissions: bool = false,

    diag_cgraph: ?*const impl.CGraph = null,
    diag_prev_start: c_int = -1,
    diag_prev_end: c_int = -1,

    idx: usize = 0,

    mul_mat_l: [type_count]bool = @splat(false),
    mul_mat_m: [type_count]bool = @splat(false),
    mul_mat_s: [type_count]bool = @splat(false),
    mul_mat_id_l: [type_count]bool = @splat(false),
    mul_mat_id_m: [type_count]bool = @splat(false),
    mul_mat_id_s: [type_count]bool = @splat(false),

    // Separate flags for the q8_1 (integer dot) mmq path.
    mul_mat_l_int: [type_count]bool = @splat(false),
    mul_mat_m_int: [type_count]bool = @splat(false),
    mul_mat_s_int: [type_count]bool = @splat(false),
    mul_mat_id_l_int: [type_count]bool = @splat(false),
    mul_mat_id_m_int: [type_count]bool = @splat(false),
    mul_mat_id_s_int: [type_count]bool = @splat(false),

    dsl: vk.VkDescriptorSetLayout = null,

    pipeline_matmul_f32: ?*MatmulPipeline = null,
    pipeline_matmul_f32_f16: ?*MatmulPipeline = null,
    pipeline_matmul_bf16: ?*MatmulPipeline = null,
    pipeline_matmul_f16: MatmulPipeline2,
    pipeline_matmul_f16_f32: MatmulPipeline2,

    pipeline_dequant_mul_mat_mat: [type_count]MatmulPipeline2,
    pipeline_dequant_mul_mat_mat_f16: [type_count]MatmulPipeline2,
    pipeline_dequant_mul_mat_mat_q8_1: [type_count]MatmulPipeline2,

    pipeline_matmul_id_f32: ?*MatmulPipeline = null,
    pipeline_matmul_id_bf16: ?*MatmulPipeline = null,
    pipeline_matmul_id_f16: MatmulPipeline2,
    pipeline_matmul_id_f16_f32: MatmulPipeline2,

    pipeline_dequant_mul_mat_mat_id: [type_count]MatmulPipeline2,
    pipeline_dequant_mul_mat_mat_id_q8_1: [type_count]MatmulPipeline2,

    pipeline_matmul_split_k_reduce: ?*Pipeline = null,
    pipeline_quantize_q8_1_x4: ?*Pipeline = null,

    pipeline_dequant: [type_count]?*Pipeline = @splat(null),
    pipeline_dequant_transpose: [type_count]?*Pipeline = @splat(null),
    pipeline_dequant_mul_mat_vec_f32_f32: [dmmv_wg_size_count][type_count][mul_mat_vec_max_cols]?*Pipeline = @splat(@splat(@splat(null))),
    pipeline_dequant_mul_mat_vec_f16_f32: [dmmv_wg_size_count][type_count][mul_mat_vec_max_cols]?*Pipeline = @splat(@splat(@splat(null))),
    pipeline_dequant_mul_mat_vec_id_f32: [dmmv_wg_size_count][type_count]?*Pipeline = @splat(@splat(null)),

    pipeline_dequant_mul_mat_vec_q8_1_f32: [dmmv_wg_size_count][type_count][mul_mat_vec_max_cols]?*Pipeline = @splat(@splat(@splat(null))),
    pipeline_dequant_mul_mat_vec_id_q8_1_f32: [dmmv_wg_size_count][type_count]?*Pipeline = @splat(@splat(null)),

    pipeline_mul_mat_vec_p021_f16_f32: [p021_max_gqa_ratio]?*Pipeline = @splat(null),
    pipeline_mul_mat_vec_nc_f16_f32: ?*Pipeline = null,
    pipeline_get_rows: [type_count]?*Pipeline = @splat(null),
    pipeline_get_rows_f32: [type_count]?*Pipeline = @splat(null),
    pipeline_get_rows_back_f32: ?*Pipeline = null,
    pipeline_acc_f32: ?*Pipeline = null,
    pipeline_set_f32: ?*Pipeline = null,

    // [src0 0=fp32,1=fp16][src1 0=fp32,1=fp16][dst 0=fp32,1=fp16]
    pipeline_add: [2][2][2]?*Pipeline = @splat(@splat(@splat(null))),
    pipeline_add_norepeat: [2][2][2]?*Pipeline = @splat(@splat(@splat(null))),
    pipeline_sub: [2][2][2]?*Pipeline = @splat(@splat(@splat(null))),
    pipeline_sub_norepeat: [2][2][2]?*Pipeline = @splat(@splat(@splat(null))),
    pipeline_mul: [2][2][2]?*Pipeline = @splat(@splat(@splat(null))),
    pipeline_mul_norepeat: [2][2][2]?*Pipeline = @splat(@splat(@splat(null))),
    pipeline_div: [2][2][2]?*Pipeline = @splat(@splat(@splat(null))),
    pipeline_div_norepeat: [2][2][2]?*Pipeline = @splat(@splat(@splat(null))),
    pipeline_add_rms: [2][2][2]?*Pipeline = @splat(@splat(@splat(null))),
    pipeline_add_rms_norepeat: [2][2][2]?*Pipeline = @splat(@splat(@splat(null))),

    // indexed by num_additional_fused_ops == num_adds - 1
    pipeline_multi_add: [m.pipeline.max_fused_adds]?*Pipeline = @splat(null),
    pipeline_multi_add_rms: [m.pipeline.max_fused_adds]?*Pipeline = @splat(null),

    pipeline_add_id_f32: ?*Pipeline = null,

    pipeline_concat_i8: ?*Pipeline = null,
    pipeline_concat_i16: ?*Pipeline = null,
    pipeline_concat_i32: ?*Pipeline = null,
    pipeline_concat_i64: ?*Pipeline = null,
    pipeline_upscale_nearest_f32: ?*Pipeline = null,
    pipeline_upscale_bilinear_f32: ?*Pipeline = null,
    pipeline_upscale_bicubic_f32: ?*Pipeline = null,
    pipeline_upscale_bilinear_antialias_f32: ?*Pipeline = null,
    pipeline_scale_f32: ?*Pipeline = null,
    pipeline_log: [2]?*Pipeline = @splat(null),
    pipeline_tri: [2]?*Pipeline = @splat(null),
    pipeline_diag: [2]?*Pipeline = @splat(null),
    pipeline_clamp: [2]?*Pipeline = @splat(null),
    pipeline_pad_f32: ?*Pipeline = null,
    pipeline_pad_reflect_1d_f32: ?*Pipeline = null,
    pipeline_roll_f32: ?*Pipeline = null,
    pipeline_repeat_i32: ?*Pipeline = null,
    pipeline_repeat_back_f32: ?*Pipeline = null,
    pipeline_repeat_i16: ?*Pipeline = null,
    pipeline_cpy_f32_f32: ?*Pipeline = null,
    pipeline_cpy_f32_f16: ?*Pipeline = null,
    pipeline_cpy_f16_f16: ?*Pipeline = null,
    pipeline_cpy_f16_f32: ?*Pipeline = null,
    pipeline_cpy_f32_bf16: ?*Pipeline = null,
    pipeline_cpy_bf16_f32: ?*Pipeline = null,
    pipeline_cpy_f32_i32: ?*Pipeline = null,
    pipeline_cpy_i32_f32: ?*Pipeline = null,
    pipeline_contig_cpy_f32_f32: ?*Pipeline = null,
    pipeline_contig_cpy_f32_f16: ?*Pipeline = null,
    pipeline_contig_cpy_f16_f16: ?*Pipeline = null,
    pipeline_contig_cpy_f16_f32: ?*Pipeline = null,
    pipeline_contig_cpy_f32_bf16: ?*Pipeline = null,
    pipeline_contig_cpy_bf16_f32: ?*Pipeline = null,
    pipeline_contig_cpy_f32_i32: ?*Pipeline = null,
    pipeline_contig_cpy_i32_f32: ?*Pipeline = null,
    pipeline_cpy_f32_quant: [type_count]?*Pipeline = @splat(null),
    pipeline_cpy_quant_f32: [type_count]?*Pipeline = @splat(null),
    pipeline_cpy_transpose_16: ?*Pipeline = null,
    pipeline_cpy_transpose_32: ?*Pipeline = null,
    pipeline_cpy_transpose_02_16: ?*Pipeline = null,
    pipeline_cpy_transpose_02_32: ?*Pipeline = null,
    // [src0 0=fp32,1=fp16][dst]
    pipeline_set_rows_i32: [2][type_count]?*Pipeline = @splat(@splat(null)),
    pipeline_set_rows_i64: [2][type_count]?*Pipeline = @splat(@splat(null)),
    pipeline_norm_f32: ?*Pipeline = null,
    pipeline_group_norm_f32: ?*Pipeline = null,
    pipeline_rms_norm_f32: ?*Pipeline = null,
    pipeline_rms_norm_mul_f32: ?*Pipeline = null,
    pipeline_rms_norm_partials_f32: ?*Pipeline = null,
    pipeline_rms_norm_mul_partials_f32: ?*Pipeline = null,
    pipeline_rms_norm_mul_rope_f32_f32: ?*Pipeline = null,
    pipeline_rms_norm_mul_rope_f32_f16: ?*Pipeline = null,
    pipeline_rms_norm_back_f32: ?*Pipeline = null,
    pipeline_l2_norm_f32: ?*Pipeline = null,

    // [src/dst 0=fp32,1=fp16]
    pipeline_exp: [2]?*Pipeline = @splat(null),
    pipeline_expm1: [2]?*Pipeline = @splat(null),
    pipeline_elu: [2]?*Pipeline = @splat(null),
    pipeline_gelu: [2]?*Pipeline = @splat(null),
    pipeline_gelu_erf: [2]?*Pipeline = @splat(null),
    pipeline_gelu_quick: [2]?*Pipeline = @splat(null),
    pipeline_silu: [2]?*Pipeline = @splat(null),
    pipeline_relu: [2]?*Pipeline = @splat(null),
    pipeline_sqr: [2]?*Pipeline = @splat(null),
    pipeline_sqrt: [2]?*Pipeline = @splat(null),
    pipeline_sin: [2]?*Pipeline = @splat(null),
    pipeline_cos: [2]?*Pipeline = @splat(null),
    pipeline_xielu: [2]?*Pipeline = @splat(null),
    pipeline_neg: [2]?*Pipeline = @splat(null),
    pipeline_tanh: [2]?*Pipeline = @splat(null),
    pipeline_sigmoid: [2]?*Pipeline = @splat(null),
    pipeline_hardsigmoid: [2]?*Pipeline = @splat(null),
    pipeline_hardswish: [2]?*Pipeline = @splat(null),
    pipeline_abs: [2]?*Pipeline = @splat(null),
    pipeline_softplus: [2]?*Pipeline = @splat(null),
    pipeline_step: [2]?*Pipeline = @splat(null),
    pipeline_round: [2]?*Pipeline = @splat(null),
    pipeline_ceil: [2]?*Pipeline = @splat(null),
    pipeline_floor: [2]?*Pipeline = @splat(null),
    pipeline_trunc: [2]?*Pipeline = @splat(null),
    pipeline_sgn: [2]?*Pipeline = @splat(null),

    pipeline_add1_f16_f16: ?*Pipeline = null,
    pipeline_add1_f16_f32: ?*Pipeline = null,
    pipeline_add1_f32_f32: ?*Pipeline = null,

    pipeline_arange_f32: ?*Pipeline = null,

    pipeline_fill_f32: ?*Pipeline = null,
    pipeline_fill_f16: ?*Pipeline = null,

    pipeline_geglu: [2]?*Pipeline = @splat(null),
    pipeline_reglu: [2]?*Pipeline = @splat(null),
    pipeline_swiglu: [2]?*Pipeline = @splat(null),
    pipeline_swiglu_oai: [2]?*Pipeline = @splat(null),
    pipeline_geglu_erf: [2]?*Pipeline = @splat(null),
    pipeline_geglu_quick: [2]?*Pipeline = @splat(null),

    pipeline_leaky_relu: [2]?*Pipeline = @splat(null),
    pipeline_silu_back_f32: ?*Pipeline = null,
    pipeline_diag_mask_inf_f32: ?*Pipeline = null,
    pipeline_soft_max_f32: ?*Pipeline = null,
    pipeline_soft_max_f32_f16: ?*Pipeline = null,
    pipeline_soft_max_f32_wg512: ?*Pipeline = null,
    pipeline_soft_max_f32_f16_wg512: ?*Pipeline = null,
    pipeline_soft_max_back_f32: ?*Pipeline = null,

    pipeline_soft_max_large1_f32: ?*Pipeline = null,
    pipeline_soft_max_large1_f32_f16: ?*Pipeline = null,
    pipeline_soft_max_large2_f32: ?*Pipeline = null,
    pipeline_soft_max_large2_f32_f16: ?*Pipeline = null,
    pipeline_soft_max_large3_f32: ?*Pipeline = null,
    pipeline_soft_max_large3_f32_f16: ?*Pipeline = null,

    pipeline_rope_norm_f32: ?*Pipeline = null,
    pipeline_rope_norm_f16: ?*Pipeline = null,
    pipeline_rope_norm_f32_f16: ?*Pipeline = null,
    pipeline_rope_neox_f32: ?*Pipeline = null,
    pipeline_rope_neox_f16: ?*Pipeline = null,
    pipeline_rope_neox_f32_f16: ?*Pipeline = null,
    pipeline_rope_multi_f32: ?*Pipeline = null,
    pipeline_rope_multi_f16: ?*Pipeline = null,
    pipeline_rope_multi_f32_f16: ?*Pipeline = null,
    pipeline_rope_vision_f32: ?*Pipeline = null,
    pipeline_rope_vision_f16: ?*Pipeline = null,
    pipeline_argsort_f32: [num_argsort_pipelines]?*Pipeline = @splat(null),
    pipeline_argsort_large_f32: [num_argsort_pipelines]?*Pipeline = @splat(null),
    pipeline_topk_f32: [num_topk_pipelines]?*Pipeline = @splat(null),
    pipeline_sum_rows_f32: ?*Pipeline = null,
    pipeline_fwht_f32: [4]?*Pipeline = @splat(null),
    pipeline_cumsum_f32: ?*Pipeline = null,
    pipeline_cumsum_small_f32: ?*Pipeline = null,
    pipeline_cumsum_multipass1_f32: ?*Pipeline = null,
    pipeline_cumsum_multipass2_f32: ?*Pipeline = null,
    pipeline_argmax_f32: ?*Pipeline = null,
    pipeline_count_equal_i32: ?*Pipeline = null,
    pipeline_solve_tri_f32: PipelineMap(SolveTriPipelineState) = .empty,
    pipeline_im2col_f32: ?*Pipeline = null,
    pipeline_im2col_f32_f16: ?*Pipeline = null,
    pipeline_im2col_3d_f32: ?*Pipeline = null,
    pipeline_im2col_3d_f32_f16: ?*Pipeline = null,
    pipeline_timestep_embedding_f32: ?*Pipeline = null,
    pipeline_conv_transpose_1d_f32: ?*Pipeline = null,
    pipeline_col2im_1d_f32: ?*Pipeline = null,
    pipeline_col2im_1d_f16: ?*Pipeline = null,
    pipeline_col2im_1d_bf16: ?*Pipeline = null,
    pipeline_out_prod_f32: ?*Pipeline = null,
    pipeline_snake_f32: ?*Pipeline = null,
    pipeline_snake_f16: ?*Pipeline = null,
    pipeline_snake_bf16: ?*Pipeline = null,
    pipeline_pool1d_f32: ?*Pipeline = null,
    pipeline_pool2d_f32: ?*Pipeline = null,
    pipeline_rwkv_wkv6_f32: ?*Pipeline = null,
    pipeline_rwkv_wkv7_f32: ?*Pipeline = null,
    pipeline_gated_linear_attn_f32: ?*Pipeline = null,
    // [size_idx][kda] where size_idx: 0=d16, 1=d32, 2=d64, 3=d128
    pipeline_gated_delta_net: [4][2]?*Pipeline = @splat(@splat(null)),
    pipeline_ssm_scan_f32_d128: ?*Pipeline = null,
    pipeline_ssm_scan_f32_d256: ?*Pipeline = null,
    pipeline_ssm_conv_f32: ?*Pipeline = null,
    pipeline_ssm_conv_silu_f32: ?*Pipeline = null,
    pipeline_ssm_conv_bias_silu_f32: ?*Pipeline = null,
    pipeline_opt_step_adamw_f32: ?*Pipeline = null,
    pipeline_opt_step_sgd_f32: ?*Pipeline = null,
    pipeline_conv2d_f32: [conv_shape_count]PipelineMap(Conv2dPipelineState) = @splat(.empty),
    pipeline_conv2d_f16_f32: [conv_shape_count]PipelineMap(Conv2dPipelineState) = @splat(.empty),
    pipeline_conv_transpose_2d_f32: [conv_shape_count]PipelineMap(Conv2dPipelineState) = @splat(.empty),
    pipeline_conv_transpose_2d_f16_f32: [conv_shape_count]PipelineMap(Conv2dPipelineState) = @splat(.empty),
    pipeline_conv3d_f32: [conv_shape_count]PipelineMap(Conv3dPipelineState) = @splat(.empty),
    pipeline_conv3d_f16_f32: [conv_shape_count]PipelineMap(Conv3dPipelineState) = @splat(.empty),
    pipeline_conv2d_dw_whcn_f32: ?*Pipeline = null,
    pipeline_conv2d_dw_whcn_f16_f32: ?*Pipeline = null,
    pipeline_conv2d_dw_cwhn_f32: ?*Pipeline = null,
    pipeline_conv2d_dw_cwhn_f16_f32: ?*Pipeline = null,

    pipeline_flash_attn_f32_f16: PipelineMap(FaPipelineState) = .empty,

    pipeline_fa_mask_opt: PipelineMap(struct { u32, u32 }) = .empty,

    pipeline_flash_attn_split_k_reduce: ?*Pipeline = null,
    pipeline_count_experts: ?*Pipeline = null,

    // [2] is for whether to take n_experts from spec constant (0) or push constant (1)
    pipeline_topk_moe: [num_topk_moe_pipelines][2]?*Pipeline = @splat(@splat(null)),

    all_pipelines: std.ArrayList(*Pipeline) = .empty,

    pinned_memory: std.ArrayList(m.buffer.PinnedMemory) = .empty,

    fence: vk.VkFence = null,
    sync_staging: ?*m.buffer.Buffer = null,

    buffer_type: c.struct_ggml_backend_buffer_type = undefined,

    disable_fusion: bool = false,
    disable_host_visible_vidmem: bool = false,
    allow_sysmem_fallback: bool = false,
    disable_graph_optimize: bool = false,

    /// Locks the device's recursive mutex, `std::lock_guard<std::recursive_mutex>`.
    pub fn lock(self: *Device) void {
        self.mutex.lock();
    }

    pub fn unlock(self: *Device) void {
        self.mutex.unlock();
    }
};

/// Ports `vk_instance_t`'s `devices` (ggml-vulkan.cpp:2566 @c1d0e7a00).
pub var devices: [common.max_devices]?*Device = @splat(null);

/// `std::make_shared<vk_device_struct>()`: every pipeline slot null, every
/// `vk_matmul_pipeline2` with both its halves allocated.
fn newDevice() *Device {
    const dev = allocator.create(Device) catch impl.abort("ggml_vulkan: out of memory");
    dev.* = .{
        .pipeline_matmul_f16 = .create(),
        .pipeline_matmul_f16_f32 = .create(),
        .pipeline_dequant_mul_mat_mat = undefined,
        .pipeline_dequant_mul_mat_mat_f16 = undefined,
        .pipeline_dequant_mul_mat_mat_q8_1 = undefined,
        .pipeline_matmul_id_f16 = .create(),
        .pipeline_matmul_id_f16_f32 = .create(),
        .pipeline_dequant_mul_mat_mat_id = undefined,
        .pipeline_dequant_mul_mat_mat_id_q8_1 = undefined,
    };
    inline for (.{ "pipeline_dequant_mul_mat_mat", "pipeline_dequant_mul_mat_mat_f16", "pipeline_dequant_mul_mat_mat_q8_1", "pipeline_dequant_mul_mat_mat_id", "pipeline_dequant_mul_mat_mat_id_q8_1" }) |field| {
        for (&@field(dev, field)) |*p| p.* = .create();
    }
    return dev;
}

/// Ports `ggml_vk_find_queue_family_index` (ggml-vulkan.cpp:3240 @c1d0e7a00).
fn findQueueFamilyIndex(props: []const vk.VkQueueFamilyProperties, required: vk.VkQueueFlags, avoid: vk.VkQueueFlags, compute_index: i32, min_num_queues: u32) u32 {
    const qfsize: u32 = @intCast(props.len);
    const other = struct {
        fn f(ci: i32, i: u32) bool {
            return ci < 0 or i != @as(u32, @bitCast(ci));
        }
    }.f;

    // Try with avoid preferences first
    var i: u32 = 0;
    while (i < qfsize) : (i += 1) {
        if (props[i].queueCount >= min_num_queues and other(compute_index, i) and props[i].queueFlags & required != 0 and props[i].queueFlags & avoid == 0) return i;
    }
    // Fall back to only required
    i = 0;
    while (i < qfsize) : (i += 1) {
        if (props[i].queueCount >= min_num_queues and other(compute_index, i) and props[i].queueFlags & required != 0) return i;
    }
    // Fall back to reusing compute queue
    i = 0;
    while (i < qfsize) : (i += 1) {
        if (props[i].queueCount >= min_num_queues and props[i].queueFlags & required != 0) return i;
    }
    // Fall back to ignoring min_num_queries
    i = 0;
    while (i < qfsize) : (i += 1) {
        if (props[i].queueFlags & required != 0) return i;
    }
    // A transfer-capable family need not say so if it has graphics or compute.
    if (compute_index >= 0) return @intCast(compute_index);

    impl.fprintfStderr("ggml_vulkan: No suitable queue family index found.\n", .{});
    for (props) |q| {
        var buf: [256]u8 = undefined;
        impl.fprintfStderr("Queue number: %u flags: %s\n", .{ q.queueCount, queueFlagsString(q.queueFlags, &buf).ptr });
    }
    std.c.abort();
}

/// `vk::to_string(vk::QueueFlags)`: `{ Graphics | Compute | ... }`.
fn queueFlagsString(flags: vk.VkQueueFlags, buf: []u8) [:0]const u8 {
    const names = [_]struct { u32, []const u8 }{
        .{ vk.VK_QUEUE_GRAPHICS_BIT, "Graphics" },
        .{ vk.VK_QUEUE_COMPUTE_BIT, "Compute" },
        .{ vk.VK_QUEUE_TRANSFER_BIT, "Transfer" },
        .{ vk.VK_QUEUE_SPARSE_BINDING_BIT, "SparseBinding" },
        .{ vk.VK_QUEUE_PROTECTED_BIT, "Protected" },
        .{ vk.VK_QUEUE_VIDEO_DECODE_BIT_KHR, "VideoDecodeKHR" },
        .{ vk.VK_QUEUE_VIDEO_ENCODE_BIT_KHR, "VideoEncodeKHR" },
        .{ vk.VK_QUEUE_OPTICAL_FLOW_BIT_NV, "OpticalFlowNV" },
    };
    if (flags == 0) return "{}";
    var w: std.Io.Writer = .fixed(buf[0 .. buf.len - 1]);
    var first = true;
    w.writeAll("{ ") catch {};
    for (names) |n| {
        if (flags & n[0] == 0) continue;
        if (!first) w.writeAll(" | ") catch {};
        w.writeAll(n[1]) catch {};
        first = false;
    }
    w.writeAll(" }") catch {};
    buf[w.end] = 0;
    return buf[0..w.end :0];
}

/// Ports `ggml_vk_create_queue` (ggml-vulkan.cpp:3286 @c1d0e7a00).
fn createQueue(device: *Device, queue_family_index: u32, queue_index: u32, stage_flags: vk.VkPipelineStageFlags, transfer_only: bool) *Queue {
    device.lock();
    defer device.unlock();

    const h = allocator.create(QueueHandle) catch impl.abort("ggml_vulkan: out of memory");
    h.* = .{ .device = device, .synchronized = !device.has_internally_synchronized_queues };
    var info2 = common.structure(vk.VkDeviceQueueInfo2, vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2);
    info2.queueFamilyIndex = queue_family_index;
    info2.queueIndex = queue_index;
    if (device.has_internally_synchronized_queues) info2.flags = vk.VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR;
    d.vkGetDeviceQueue2.?(device.device, &info2, &h.queue);

    const q = allocator.create(Queue) catch impl.abort("ggml_vulkan: out of memory");
    q.* = .{ .queue_family_index = queue_family_index, .handle = h, .transfer_only = transfer_only };
    q.cmd_pool.init(device, q);
    q.stage_flags = stage_flags;
    return q;
}

/// Ports `ggml_vk_create_aliased_queue` (ggml-vulkan.cpp:3316 @c1d0e7a00):
/// a second queue record over the same `VkQueue`, with its own pool.
fn createAliasedQueue(device: *Device, source: *const Queue) *Queue {
    device.lock();
    defer device.unlock();
    const q = allocator.create(Queue) catch impl.abort("ggml_vulkan: out of memory");
    q.* = .{
        .handle = source.handle,
        .queue_family_index = source.queue_family_index,
        .stage_flags = source.stage_flags,
        .transfer_only = source.transfer_only,
    };
    q.cmd_pool.init(device, q);
    return q;
}

/// Ports `ggml_vk_intel_shader_core_count` (ggml-vulkan.cpp:18956 @c1d0e7a00).
fn intelShaderCoreCount(vkdev: vk.VkPhysicalDevice) u32 {
    var props = common.structure(vk.VkPhysicalDeviceProperties2, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
    d.vkGetPhysicalDeviceProperties2.?(vkdev, &props);
    if (props.properties.vendorID != common.vendor_id.intel) return 0;
    return switch (props.properties.deviceID) {
        0x56A6 => 6, // A310
        0x5693, 0x56A5, 0x56B1 => 8, // A370M, A380, Pro A40/A50
        0x5697 => 12, // A530M
        0x5692, 0x56B3 => 16, // A550M, Pro A60
        0x56A2 => 24, // A580
        0x5691, 0x56A1 => 28, // A730M, A750
        0x56A0, 0x5690 => 32, // A770, A770M
        0xE212 => 16, // Pro B50
        0xE20C => 18, // B570
        0xE20B, 0xE211 => 20, // B580, Pro B60
        0xB080 => 12, // PTL Xe3 LPG 2x6 (12 subslices)
        else => 0,
    };
}

/// `std::stoull` / `std::stoul` on an environment value: leading
/// whitespace, an optional sign, then digits in base 10; anything else, or
/// overflow, throws -- and nothing catches it.
fn stoull(comptime var_name: []const u8, s: [*:0]const u8) u64 {
    var end: [*c]u8 = undefined;
    std.c._errno().* = 0;
    const v = strtoull(s, &end, 10);
    if (end == @as([*c]const u8, s)) impl.abort("ggml_vulkan: std::stoull: no conversion (" ++ var_name ++ ")");
    if (std.c._errno().* == @backingInt(std.c.E.RANGE)) impl.abort("ggml_vulkan: std::stoull: out of range (" ++ var_name ++ ")");
    return v;
}
extern fn strtoull(s: [*:0]const u8, end: *[*c]u8, base: c_int) u64;

/// `std::stoul`, which is `strtoul` and on an LP64 target the same width as
/// `stoull`; the C++ then narrows the result to `uint32_t`.
fn stoul(comptime var_name: []const u8, s: [*:0]const u8) u32 {
    return @truncate(stoull(var_name, s));
}

/// Links `next` onto the end of a `pNext` chain at `last`, the C++'s
/// `last_struct->pNext = ...; last_struct = ...;`.
fn link(last: **vk.VkBaseOutStructure, next: anytype) void {
    last.*.pNext = @ptrCast(next);
    last.* = @ptrCast(next);
}

/// Ports `ggml_vk_get_device` (ggml-vulkan.cpp:6161 @c1d0e7a00): creates
/// the logical device for index `idx` on first use, and returns it.
pub fn getDevice(idx: usize) *Device {
    if (devices[idx]) |dev| return dev;

    const device = newDevice();
    devices[idx] = device;

    const dev_num = m.instance.instance.device_indices.items[idx];

    const physical_devices = common.enumeratePhysicalDevices(m.instance.instance.instance) catch common.abortLastError();
    defer allocator.free(physical_devices);
    if (dev_num >= physical_devices.len) {
        impl.fprintfStderr("ggml_vulkan: Device with index %zu does not exist.\n", .{dev_num});
        impl.abort("Device not found");
    }

    device.physical_device = physical_devices[dev_num];
    const ext_props = common.enumerateDeviceExtensionProperties(device.physical_device) catch common.abortLastError();
    defer allocator.free(ext_props);

    device.architecture = m.instance.getDeviceArchitecture(device.physical_device) catch common.abortLastError();

    device.prefer_host_memory = common.envSet("GGML_VK_PREFER_HOST_MEMORY");
    device.disable_host_visible_vidmem = common.envSet("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM");
    device.allow_sysmem_fallback = common.envSet("GGML_VK_ALLOW_SYSMEM_FALLBACK");
    device.disable_graph_optimize = common.envSet("GGML_VK_DISABLE_GRAPH_OPTIMIZE");

    var fp16_storage = false;
    var fp16_compute = false;
    var maintenance4_support = false;
    var sm_builtins = false;
    var amd_shader_core_properties2 = false;
    var pipeline_robustness = false;
    var coopmat2_support = false;
    var coopmat2_decode_vector_support = false;
    var pipeline_executable_properties_support = false;
    var internally_sync_support = false;
    device.coopmat_support = false;
    device.integer_dot_product = false;
    device.shader_64b_indexing = false;
    var bfloat16_support = false;
    var dot2_f16_support = false;
    var ocp_microscaling_extension = false;
    var shader_float8_extension = false;

    // One `else if` chain in the C++: the first matching arm wins, and an
    // arm whose environment veto is set falls through to the next.
    for (ext_props) |*p| {
        const n = std.mem.sliceTo(&p.extensionName, 0);
        const eq = struct {
            fn f(a: []const u8, b: []const u8) bool {
                return std.mem.eql(u8, a, b);
            }
        }.f;
        if (eq("VK_KHR_maintenance4", n)) {
            maintenance4_support = true;
        } else if (eq("VK_KHR_16bit_storage", n)) {
            fp16_storage = true;
        } else if (eq("VK_KHR_shader_float16_int8", n)) {
            fp16_compute = true;
        } else if (eq("VK_NV_shader_sm_builtins", n)) {
            sm_builtins = true;
        } else if (eq("VK_AMD_shader_core_properties2", n)) {
            amd_shader_core_properties2 = true;
        } else if (eq("VK_EXT_pipeline_robustness", n)) {
            pipeline_robustness = true;
        } else if (eq("VK_EXT_subgroup_size_control", n)) {
            device.subgroup_size_control = true;
        } else if (glslc.coopmat and eq("VK_KHR_cooperative_matrix", n) and !common.envSet("GGML_VK_DISABLE_COOPMAT")) {
            device.coopmat_support = true;
            device.coopmat_m = 0;
            device.coopmat_n = 0;
            device.coopmat_k = 0;
        } else if (glslc.coopmat2 and eq("VK_NV_cooperative_matrix2", n) and !common.envSet("GGML_VK_DISABLE_COOPMAT2")) {
            coopmat2_support = true;
        } else if (eq(vk.VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME, n) and !common.envSet("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
            coopmat2_decode_vector_support = true;
        } else if (glslc.integer_dot and eq("VK_KHR_shader_integer_dot_product", n) and !common.envSet("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
            device.integer_dot_product = true;
        } else if (glslc.bfloat16 and eq("VK_KHR_shader_bfloat16", n) and !common.envSet("GGML_VK_DISABLE_BFLOAT16")) {
            bfloat16_support = true;
        } else if (glslc.float_e2m1 and eq(vk.VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, n)) {
            ocp_microscaling_extension = true;
        } else if (glslc.float_e4m3 and eq(vk.VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, n)) {
            shader_float8_extension = true;
        } else if (eq("VK_VALVE_shader_mixed_float_dot_product", n) and !common.envSet("GGML_VK_DISABLE_DOT2")) {
            dot2_f16_support = true;
        } else if (eq("VK_KHR_pipeline_executable_properties", n)) {
            pipeline_executable_properties_support = true;
        } else if (eq("VK_EXT_memory_priority", n) and common.envSet("GGML_VK_ENABLE_MEMORY_PRIORITY")) {
            device.memory_priority = true;
        } else if (eq("VK_EXT_external_memory_host", n)) {
            device.external_memory_host = true;
        } else if (@hasDecl(vk, "VK_EXT_shader_64bit_indexing") and eq("VK_EXT_shader_64bit_indexing", n)) {
            device.shader_64b_indexing = true;
        } else if (eq(vk.VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME, n)) {
            internally_sync_support = true;
        } else if (eq("VK_EXT_device_fault", n)) {
            device.device_fault = true;
        }
    }

    var props2 = common.structure(vk.VkPhysicalDeviceProperties2, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
    var props3 = common.structure(vk.VkPhysicalDeviceMaintenance3Properties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES);
    var props4 = common.structure(vk.VkPhysicalDeviceMaintenance4Properties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES);
    var subgroup_props = common.structure(vk.VkPhysicalDeviceSubgroupProperties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES);
    var driver_props = common.structure(vk.VkPhysicalDeviceDriverProperties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES);
    var sm_props = common.structure(vk.VkPhysicalDeviceShaderSMBuiltinsPropertiesNV, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV);
    var amd_shader_core_properties2_props = common.structure(vk.VkPhysicalDeviceShaderCoreProperties2AMD, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD);
    var vk11_props = common.structure(vk.VkPhysicalDeviceVulkan11Properties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES);
    var vk12_props = common.structure(vk.VkPhysicalDeviceVulkan12Properties, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES);
    var subgroup_size_control_props = common.structure(vk.VkPhysicalDeviceSubgroupSizeControlPropertiesEXT, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES);
    var shader_integer_dot_product_props = common.structure(vk.VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES);
    var external_memory_host_props = common.structure(vk.VkPhysicalDeviceExternalMemoryHostPropertiesEXT, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT);

    props2.pNext = &props3;
    props3.pNext = &subgroup_props;
    subgroup_props.pNext = &driver_props;
    driver_props.pNext = &vk11_props;
    vk11_props.pNext = &vk12_props;

    var last_struct: *vk.VkBaseOutStructure = @ptrCast(&vk12_props);
    if (maintenance4_support) link(&last_struct, &props4);
    if (sm_builtins) link(&last_struct, &sm_props);
    if (amd_shader_core_properties2) link(&last_struct, &amd_shader_core_properties2_props);
    if (device.subgroup_size_control) link(&last_struct, &subgroup_size_control_props);
    var coopmat2_props = common.structure(vk.VkPhysicalDeviceCooperativeMatrix2PropertiesNV, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_PROPERTIES_NV);
    if (coopmat2_support) link(&last_struct, &coopmat2_props);
    if (device.integer_dot_product) link(&last_struct, &shader_integer_dot_product_props);
    if (device.external_memory_host) link(&last_struct, &external_memory_host_props);

    d.vkGetPhysicalDeviceProperties2.?(device.physical_device, &props2);
    device.properties = props2.properties;
    device.vendor_id = device.properties.vendorID;
    device.driver_id = driver_props.driverID;

    if (device.driver_id == vk.VK_DRIVER_ID_MOLTENVK) {
        // Disable external_memory_host until MoltenVK#2622 is in the SDK.
        device.external_memory_host = false;
    }

    // Implementing the async backend interfaces seems broken on older Intel HW.
    device.support_async = (device.vendor_id != common.vendor_id.intel or
        std.mem.indexOf(u8, std.mem.sliceTo(&device.properties.deviceName, 0), "(DG1)") == null) and
        !common.envSet("GGML_VK_DISABLE_ASYNC");
    if (!device.support_async) {
        impl.logDebug("ggml_vulkan: WARNING: Async execution disabled on certain Intel devices.\n", .{});
    }

    if (std.c.getenv("GGML_VK_FORCE_MAX_ALLOCATION_SIZE")) |s| {
        device.max_memory_allocation_size = stoull("GGML_VK_FORCE_MAX_ALLOCATION_SIZE", s);
    } else if (maintenance4_support) {
        device.max_memory_allocation_size = @min(props3.maxMemoryAllocationSize, props4.maxBufferSize);
    } else {
        device.max_memory_allocation_size = props3.maxMemoryAllocationSize;
    }

    if (std.c.getenv("GGML_VK_FORCE_MAX_BUFFER_SIZE")) |s| {
        device.max_buffer_size = stoull("GGML_VK_FORCE_MAX_BUFFER_SIZE", s);
    } else if (maintenance4_support) {
        device.max_buffer_size = props4.maxBufferSize;
    } else {
        device.max_buffer_size = device.max_memory_allocation_size;
    }

    if (std.c.getenv("GGML_VK_SUBALLOCATION_BLOCK_SIZE")) |s| {
        device.suballocation_block_size = stoull("GGML_VK_SUBALLOCATION_BLOCK_SIZE", s);
    } else {
        // Limit batching of allocations to 1GB by default to avoid fragmentation issues
        device.suballocation_block_size = 1024 * 1024 * 1024;
    }
    device.suballocation_block_size = @min(device.suballocation_block_size, device.max_memory_allocation_size);

    device.subgroup_size = subgroup_props.subgroupSize;
    device.subgroup_size_log2 = @intFromFloat(@log2(@as(f32, @floatFromInt(device.subgroup_size))));
    device.uma = device.properties.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    if (sm_builtins) {
        device.shader_core_count = sm_props.shaderSMCount;
    } else if (amd_shader_core_properties2) {
        device.shader_core_count = amd_shader_core_properties2_props.activeComputeUnitCount;
    } else if (device.vendor_id == common.vendor_id.intel) {
        device.shader_core_count = intelShaderCoreCount(device.physical_device);
    } else {
        device.shader_core_count = 0;
    }
    device.float_controls_rte_fp16 = vk12_props.shaderRoundingModeRTEFloat16 != 0;
    device.float_controls_denorm_preserve_fp16 = vk12_props.shaderDenormPreserveFloat16 != 0;

    const compute_stage = vk11_props.subgroupSupportedStages & vk.VK_SHADER_STAGE_COMPUTE_BIT != 0;
    const ops = vk11_props.subgroupSupportedOperations;
    device.subgroup_basic = compute_stage and ops & vk.VK_SUBGROUP_FEATURE_BASIC_BIT != 0;
    device.subgroup_arithmetic = compute_stage and ops & vk.VK_SUBGROUP_FEATURE_ARITHMETIC_BIT != 0;
    // Workaround for subgroup arithmetic failing on MoltenVK with AMD GPUs (issue 15846)
    if (@import("builtin").os.tag.isDarwin() and device.vendor_id == common.vendor_id.amd) device.subgroup_arithmetic = false;
    device.subgroup_shuffle = compute_stage and ops & vk.VK_SUBGROUP_FEATURE_SHUFFLE_BIT != 0;
    if (@import("builtin").os.tag.isDarwin() and device.vendor_id == common.vendor_id.amd) device.subgroup_shuffle = false;
    device.subgroup_clustered = compute_stage and ops & vk.VK_SUBGROUP_FEATURE_CLUSTERED_BIT != 0;
    device.subgroup_ballot = compute_stage and ops & vk.VK_SUBGROUP_FEATURE_BALLOT_BIT != 0;
    device.subgroup_vote = compute_stage and ops & vk.VK_SUBGROUP_FEATURE_VOTE_BIT != 0;

    // Submit at least every 100 nodes, in case there are workloads without as much matmul.
    device.max_nodes_per_submit = 100;
    if (std.c.getenv("GGML_VK_MAX_NODES_PER_SUBMIT")) |s| {
        device.max_nodes_per_submit = @max(stoul("GGML_VK_MAX_NODES_PER_SUBMIT", s), 1);
    }

    const force_disable_f16 = common.envSet("GGML_VK_DISABLE_F16");
    device.fp16 = !force_disable_f16 and fp16_storage and fp16_compute;

    if (!m.instance.khrCooperativeMatrixSupport(&device.properties, &driver_props, device.architecture)) {
        device.coopmat_support = false;
    }

    device.integer_dot_product = device.integer_dot_product and shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated != 0;
    device.min_imported_host_pointer_alignment = external_memory_host_props.minImportedHostPointerAlignment;
    device.max_workgroup_size_log2 = @intFromFloat(@log2(@as(f32, @floatFromInt(device.properties.limits.maxComputeWorkGroupInvocations))));

    var qf_count: u32 = 0;
    d.vkGetPhysicalDeviceQueueFamilyProperties.?(device.physical_device, &qf_count, null);
    const queue_family_props = allocator.alloc(vk.VkQueueFamilyProperties, qf_count) catch impl.abort("ggml_vulkan: out of memory");
    defer allocator.free(queue_family_props);
    d.vkGetPhysicalDeviceQueueFamilyProperties.?(device.physical_device, &qf_count, queue_family_props.ptr);

    // Try to find a non-graphics compute queue and transfer-focused queues
    const allow_graphics_queue = common.envSet("GGML_VK_ALLOW_GRAPHICS_QUEUE");
    const graphics_flag: vk.VkQueueFlags = if (allow_graphics_queue) 0 else vk.VK_QUEUE_GRAPHICS_BIT;
    const compute_queue_family_index = findQueueFamilyIndex(queue_family_props[0..qf_count], vk.VK_QUEUE_COMPUTE_BIT, graphics_flag, -1, 1);
    const transfer_queue_family_index = findQueueFamilyIndex(queue_family_props[0..qf_count], vk.VK_QUEUE_TRANSFER_BIT, @as(vk.VkQueueFlags, vk.VK_QUEUE_COMPUTE_BIT) | graphics_flag, @bitCast(compute_queue_family_index), 1);

    const priorities = [_]f32{ 1.0, 1.0 };
    device.single_queue = compute_queue_family_index == transfer_queue_family_index and queue_family_props[compute_queue_family_index].queueCount == 1;

    var device_queue_create_infos: [2]vk.VkDeviceQueueCreateInfo = undefined;
    var n_queue_create_infos: u32 = 0;
    var device_extensions: std.ArrayList([*:0]const u8) = .empty;
    defer device_extensions.deinit(allocator);
    const addExt = struct {
        fn f(list: *std.ArrayList([*:0]const u8), name: [*:0]const u8) void {
            list.append(allocator, name) catch impl.abort("ggml_vulkan: out of memory");
        }
    }.f;
    var device_features: vk.VkPhysicalDeviceFeatures = undefined;
    d.vkGetPhysicalDeviceFeatures.?(device.physical_device, &device_features);

    var device_features2: vk.VkPhysicalDeviceFeatures2 = .{ .sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = null, .features = device_features };
    // The C++ leaves the feature bits of these two uninitialised; the query
    // writes every one of them.
    var vk11_features = common.structure(vk.VkPhysicalDeviceVulkan11Features, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
    device_features2.pNext = &vk11_features;
    var vk12_features = common.structure(vk.VkPhysicalDeviceVulkan12Features, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    vk11_features.pNext = &vk12_features;

    last_struct = @ptrCast(&vk12_features);

    var internally_synchronized_queues_features = common.structure(vk.VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR);
    if (internally_sync_support) {
        link(&last_struct, &internally_synchronized_queues_features);
        addExt(&device_extensions, vk.VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME);
    }

    var pl_robustness_features = common.structure(vk.VkPhysicalDevicePipelineRobustnessFeaturesEXT, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES);
    if (pipeline_robustness) {
        link(&last_struct, &pl_robustness_features);
        addExt(&device_extensions, "VK_EXT_pipeline_robustness");
    }

    var memory_priority_features = common.structure(vk.VkPhysicalDeviceMemoryPriorityFeaturesEXT, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT);
    if (device.memory_priority) {
        link(&last_struct, &memory_priority_features);
        addExt(&device_extensions, "VK_EXT_memory_priority");
    }

    var subgroup_size_control_features = common.structure(vk.VkPhysicalDeviceSubgroupSizeControlFeaturesEXT, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES);
    if (device.subgroup_size_control) link(&last_struct, &subgroup_size_control_features);

    var coopmat_features = common.structure(vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR);
    if (device.coopmat_support) link(&last_struct, &coopmat_features);

    var coopmat2_features = common.structure(vk.VkPhysicalDeviceCooperativeMatrix2FeaturesNV, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV);
    if (coopmat2_support) {
        link(&last_struct, &coopmat2_features);
        addExt(&device_extensions, "VK_NV_cooperative_matrix2");
    }

    var coopmat2_decode_vector_features = common.structure(vk.VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV);
    if (coopmat2_decode_vector_support) {
        link(&last_struct, &coopmat2_decode_vector_features);
        addExt(&device_extensions, vk.VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME);
    }

    var bfloat16_features = common.structure(vk.VkPhysicalDeviceShaderBfloat16FeaturesKHR, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR);
    if (bfloat16_support) {
        link(&last_struct, &bfloat16_features);
        addExt(&device_extensions, "VK_KHR_shader_bfloat16");
    }

    var ocp_microscaling_features = common.structure(vk.VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT);
    if (ocp_microscaling_extension) {
        link(&last_struct, &ocp_microscaling_features);
        addExt(&device_extensions, vk.VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME);
    }

    var shader_float8_features = common.structure(vk.VkPhysicalDeviceShaderFloat8FeaturesEXT, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT);
    if (shader_float8_extension) {
        link(&last_struct, &shader_float8_features);
        addExt(&device_extensions, vk.VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    }

    var maint4_features = common.structure(vk.VkPhysicalDeviceMaintenance4Features, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES);
    if (maintenance4_support) {
        link(&last_struct, &maint4_features);
        addExt(&device_extensions, "VK_KHR_maintenance4");
    }

    var shader_integer_dot_product_features = common.structure(vk.VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES);
    if (device.integer_dot_product) {
        link(&last_struct, &shader_integer_dot_product_features);
        addExt(&device_extensions, "VK_KHR_shader_integer_dot_product");
    }

    var dot2_features = common.structure(vk.VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES_VALVE);
    if (dot2_f16_support) {
        link(&last_struct, &dot2_features);
        addExt(&device_extensions, "VK_VALVE_shader_mixed_float_dot_product");
    }

    var pep_features = common.structure(vk.VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR);
    if (pipeline_executable_properties_support) {
        link(&last_struct, &pep_features);
        addExt(&device_extensions, "VK_KHR_pipeline_executable_properties");
    }

    if (device.external_memory_host) addExt(&device_extensions, "VK_EXT_external_memory_host");

    var shader_64bit_indexing_features = common.structure(vk.VkPhysicalDeviceShader64BitIndexingFeaturesEXT, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_64_BIT_INDEXING_FEATURES_EXT);
    if (device.shader_64b_indexing) {
        link(&last_struct, &shader_64bit_indexing_features);
        addExt(&device_extensions, "VK_EXT_shader_64bit_indexing");
    }

    var fault_features = common.structure(vk.VkPhysicalDeviceFaultFeaturesEXT, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT);
    if (device.device_fault) {
        link(&last_struct, &fault_features);
        addExt(&device_extensions, "VK_EXT_device_fault");
    }

    common.vkGetPhysicalDeviceFeatures2(device.physical_device, &device_features2);

    device.device_fault = device.device_fault and fault_features.deviceFault != 0;
    device.has_internally_synchronized_queues = internally_synchronized_queues_features.internallySynchronizedQueues != 0;

    // Build queue create infos only after querying whether internally synchronized queues are enabled.
    const queue_flags: vk.VkDeviceQueueCreateFlags = if (device.has_internally_synchronized_queues) vk.VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR else 0;
    const qci = struct {
        fn f(flags: vk.VkDeviceQueueCreateFlags, family: u32, count: u32, prio: [*]const f32) vk.VkDeviceQueueCreateInfo {
            var info = common.structure(vk.VkDeviceQueueCreateInfo, vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
            info.flags = flags;
            info.queueFamilyIndex = family;
            info.queueCount = count;
            info.pQueuePriorities = prio;
            return info;
        }
    }.f;
    if (compute_queue_family_index != transfer_queue_family_index) {
        device_queue_create_infos[0] = qci(queue_flags, compute_queue_family_index, 1, &priorities);
        device_queue_create_infos[1] = qci(queue_flags, transfer_queue_family_index, 1, priorities[1..].ptr);
        n_queue_create_infos = 2;
    } else if (!device.single_queue) {
        device_queue_create_infos[0] = qci(queue_flags, compute_queue_family_index, 2, &priorities);
        n_queue_create_infos = 1;
    } else {
        device_queue_create_infos[0] = qci(queue_flags, compute_queue_family_index, 1, &priorities);
        n_queue_create_infos = 1;
    }

    device.pipeline_executable_properties_support = pipeline_executable_properties_support;
    device.fp16 = device.fp16 and vk12_features.shaderFloat16 != 0;
    device.bf16 = bfloat16_support and bfloat16_features.shaderBFloat16Type != 0;
    device.dot2_f16 = dot2_f16_support and dot2_features.shaderMixedFloatDotProductFloat16AccFloat32 != 0;
    device.ocp_fp4 = ocp_microscaling_extension and ocp_microscaling_features.shaderFloat4 != 0 and
        shader_float8_extension and shader_float8_features.shaderFloat8 != 0 and
        !common.envSet("GGML_VK_DISABLE_OCP_FP4");
    device.pipeline_robustness = pl_robustness_features.pipelineRobustness != 0;
    device.multi_add = vk12_props.shaderRoundingModeRTEFloat16 != 0 and
        device.properties.limits.maxPushConstantsSize >= @sizeOf(m.push.MultiAdd) and
        !common.envSet("GGML_VK_DISABLE_MULTI_ADD");
    device.shader_int64 = device_features2.features.shaderInt64 != 0;
    device.buffer_device_address = vk12_features.bufferDeviceAddress != 0;
    device.vulkan_memory_model = vk12_features.vulkanMemoryModel != 0;

    if (device.subgroup_size_control) {
        device.subgroup_min_size = subgroup_size_control_props.minSubgroupSize;
        device.subgroup_max_size = subgroup_size_control_props.maxSubgroupSize;
        addExt(&device_extensions, "VK_EXT_subgroup_size_control");
    }
    device.subgroup_size_control = device.subgroup_size_control and
        subgroup_size_control_props.requiredSubgroupSizeStages & vk.VK_SHADER_STAGE_COMPUTE_BIT != 0 and
        subgroup_size_control_features.subgroupSizeControl != 0;
    device.subgroup_require_full_support = subgroup_size_control_features.computeFullSubgroups != 0;

    device.coopmat_support = device.coopmat_support and coopmat_features.cooperativeMatrix != 0;
    device.coopmat1_fa_support = device.coopmat_support and device.subgroup_require_full_support;

    if (coopmat2_support) {
        if (comptime glslc.coopmat2) {
            if (coopmat2_features.cooperativeMatrixWorkgroupScope != 0 and
                coopmat2_features.cooperativeMatrixFlexibleDimensions != 0 and
                coopmat2_features.cooperativeMatrixReductions != 0 and
                coopmat2_features.cooperativeMatrixConversions != 0 and
                coopmat2_features.cooperativeMatrixPerElementOperations != 0 and
                coopmat2_features.cooperativeMatrixTensorAddressing != 0 and
                coopmat2_features.cooperativeMatrixBlockLoads != 0 and
                vk12_features.bufferDeviceAddress != 0)
            {
                coopmat2Dimensions(device, &coopmat2_props, coopmat2_decode_vector_support, &coopmat2_decode_vector_features);
            }
        }
    }

    if (vk11_features.storageBuffer16BitAccess == 0) {
        impl.fprintfStderr("ggml_vulkan: device %s%zu does not support 16-bit storage.\n", .{ common.name, idx });
        impl.abort("Unsupported device");
    }

    addExt(&device_extensions, "VK_KHR_16bit_storage");
    if (device.fp16) addExt(&device_extensions, "VK_KHR_shader_float16_int8");

    if (device.coopmat_support) coopmatShapes(device);
    if (device.coopmat_support) addExt(&device_extensions, "VK_KHR_cooperative_matrix");
    if (device.coopmat_bf16_support) addExt(&device_extensions, "VK_KHR_shader_bfloat16");

    device.name = std.fmt.allocPrintSentinel(allocator, common.name ++ "{d}", .{idx}, 0) catch impl.abort("ggml_vulkan: out of memory");

    var device_create_info = common.structure(vk.VkDeviceCreateInfo, vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
    device_create_info.queueCreateInfoCount = n_queue_create_infos;
    device_create_info.pQueueCreateInfos = &device_queue_create_infos;
    device_create_info.enabledExtensionCount = @intCast(device_extensions.items.len);
    device_create_info.ppEnabledExtensionNames = device_extensions.items.ptr;
    device_create_info.pNext = &device_features2;
    common.checkOrAbort(d.vkCreateDevice.?(device.physical_device, &device_create_info, null, &device.device), "vk::PhysicalDevice::createDevice");

    if (device.device_fault) {
        device.pfn_vkGetDeviceFaultInfoEXT = @ptrCast(vkGetDeviceProcAddr(device.device, "vkGetDeviceFaultInfoEXT"));
    }

    // Queues
    device.compute_queue = createQueue(device, compute_queue_family_index, 0, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | vk.VK_PIPELINE_STAGE_TRANSFER_BIT, false);

    // Shaders
    // Disable matmul tile sizes early if performance low or not supported
    for (0..type_count) |i| {
        switch (device.vendor_id) {
            common.vendor_id.amd => {
                device.mul_mat_l[i] = device.coopmat_support and device.driver_id != vk.VK_DRIVER_ID_AMD_PROPRIETARY;
                device.mul_mat_m[i] = true;
                device.mul_mat_s[i] = true;
                device.mul_mat_id_l[i] = false;
                device.mul_mat_id_m[i] = true;
                device.mul_mat_id_s[i] = true;
            },
            common.vendor_id.intel => {
                // Current Windows driver does not expose BF16 support.
                const use_l_warptile = if (i == c.GGML_TYPE_BF16) device.coopmat_bf16_support and device.coopmat_support else device.coopmat_support;
                device.mul_mat_l[i] = use_l_warptile;
                device.mul_mat_id_l[i] = use_l_warptile;
                device.mul_mat_m[i] = true;
                device.mul_mat_s[i] = true;
                device.mul_mat_id_m[i] = true;
                device.mul_mat_id_s[i] = true;
            },
            common.vendor_id.apple => {
                device.mul_mat_l[i] = false;
                device.mul_mat_m[i] = true;
                device.mul_mat_s[i] = false;
                device.mul_mat_id_l[i] = false;
                device.mul_mat_id_m[i] = true;
                device.mul_mat_id_s[i] = false;
            },
            common.vendor_id.qualcomm => {
                device.mul_mat_l[i] = false;
                device.mul_mat_m[i] = true;
                device.mul_mat_s[i] = true;
                device.mul_mat_id_l[i] = false;
                device.mul_mat_id_m[i] = true;
                device.mul_mat_id_s[i] = true;
            },
            else => {
                device.mul_mat_l[i] = true;
                device.mul_mat_m[i] = true;
                device.mul_mat_s[i] = true;
                device.mul_mat_id_l[i] = true;
                device.mul_mat_id_m[i] = true;
                device.mul_mat_id_s[i] = true;
            },
        }

        if (comptime vk.VK_HEADER_VERSION >= 287) {
            // Honeykrisp (Asahi Linux) doesn't report VK_VENDOR_ID_APPLE.
            if (device.driver_id == vk.VK_DRIVER_ID_MESA_HONEYKRISP) {
                device.mul_mat_l[i] = false;
                device.mul_mat_m[i] = true;
                device.mul_mat_s[i] = false;
                device.mul_mat_id_l[i] = false;
                device.mul_mat_id_m[i] = true;
                device.mul_mat_id_s[i] = false;
            }
        }

        device.mul_mat_l_int[i] = device.mul_mat_l[i];
        device.mul_mat_m_int[i] = device.mul_mat_m[i];
        device.mul_mat_s_int[i] = device.mul_mat_s[i];
        device.mul_mat_id_l_int[i] = device.mul_mat_id_l[i];
        device.mul_mat_id_m_int[i] = device.mul_mat_id_m[i];
        device.mul_mat_id_s_int[i] = device.mul_mat_id_s[i];
    }

    var dsl_binding: [m.pipeline.max_parameter_count]vk.VkDescriptorSetLayoutBinding = undefined;
    var dsl_binding_flags: [m.pipeline.max_parameter_count]vk.VkDescriptorBindingFlags = @splat(0);
    for (&dsl_binding, 0..) |*b, i| {
        b.* = .{ .binding = @intCast(i), .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = null };
    }
    var dslbfci = common.structure(vk.VkDescriptorSetLayoutBindingFlagsCreateInfo, vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);
    dslbfci.bindingCount = dsl_binding_flags.len;
    dslbfci.pBindingFlags = &dsl_binding_flags;
    var dslci = common.structure(vk.VkDescriptorSetLayoutCreateInfo, vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
    dslci.bindingCount = dsl_binding.len;
    dslci.pBindings = &dsl_binding;
    dslci.pNext = &dslbfci;
    common.checkOrAbort(d.vkCreateDescriptorSetLayout.?(device.device, &dslci, null, &device.dsl), "vk::Device::createDescriptorSetLayout");

    m.shaders.loadShaders(device, null);

    // Prefer a dedicated transfer queue on AMD dGPUs (non-GCN) when graphics queue use is disabled.
    const prefers_transfer_queue = device.vendor_id == common.vendor_id.amd and
        device.architecture != .amd_gcn and !device.uma and !allow_graphics_queue;

    if (!device.single_queue) {
        const transfer_queue_index: u32 = if (compute_queue_family_index == transfer_queue_family_index) 1 else 0;
        device.transfer_queue = createQueue(device, transfer_queue_family_index, transfer_queue_index, vk.VK_PIPELINE_STAGE_TRANSFER_BIT, true);
        device.async_use_transfer_queue = prefers_transfer_queue or common.envSet("GGML_VK_ASYNC_USE_TRANSFER_QUEUE");
    } else {
        device.transfer_queue = createAliasedQueue(device, device.compute_queue.?);
        device.async_use_transfer_queue = false;
    }

    device.buffer_type = .{
        .iface = m.buffer.buffer_type_iface,
        .device = c.ggml_backend_reg_dev_get(m.backend.reg(), idx),
        .context = m.buffer.BufferTypeCtx.create(device.name, device),
    };

    var fence_info = common.structure(vk.VkFenceCreateInfo, vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    common.checkOrAbort(d.vkCreateFence.?(device.device, &fence_info, null, &device.fence), "vk::Device::createFence");

    device.idx = idx;
    device.serialize_submissions = common.envSet("GGML_VK_SERIALIZE_SUBMISSIONS");
    device.disable_fusion = common.envSet("GGML_VK_DISABLE_FUSION");
    device.add_rms_fusion = !device.disable_fusion and device.subgroup_arithmetic and device.vendor_id != common.vendor_id.intel;
    device.partials_binding_alignment = @max(4, @as(u32, @intCast(device.properties.limits.minStorageBufferOffsetAlignment)));

    device.mmvq_mode = 0;
    if (common.envSet("GGML_VK_DISABLE_MMVQ")) {
        device.mmvq_mode = -1;
    } else if (common.envSet("GGML_VK_FORCE_MMVQ")) {
        device.mmvq_mode = 1;
    }

    return device;
}

extern fn vkGetDeviceProcAddr(device: vk.VkDevice, name: [*:0]const u8) vk.PFN_vkVoidFunction;

/// The coopmat2 flexible-dimension query inside `ggml_vk_get_device`: the
/// device takes the coopmat2 path only if it supports fp16 with both
/// accumulators at workgroup sizes 128 and 256.
fn coopmat2Dimensions(device: *Device, coopmat2_props: *const vk.VkPhysicalDeviceCooperativeMatrix2PropertiesNV, decode_vector_support: bool, decode_vector_features: *const vk.VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV) void {
    const pfn: vk.PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV = @ptrCast(common.vkGetInstanceProcAddr(m.instance.instance.instance, "vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV"));
    var count: u32 = 0;
    _ = pfn.?(device.physical_device, &count, null);
    const flexible_dimensions = allocator.alloc(vk.VkCooperativeMatrixFlexibleDimensionsPropertiesNV, count) catch impl.abort("ggml_vulkan: out of memory");
    defer allocator.free(flexible_dimensions);
    for (flexible_dimensions) |*p| p.* = common.structure(vk.VkCooperativeMatrixFlexibleDimensionsPropertiesNV, vk.VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV);
    _ = pfn.?(device.physical_device, &count, flexible_dimensions.ptr);

    var found_fp16_128 = false;
    var found_fp16_256 = false;
    var found_fp32_128 = false;
    var found_fp32_256 = false;
    var found_bf16_128 = false;
    var found_bf16_256 = false;
    // need to support fp16*fp16 with fp16/fp32 accumulator, for workgroupsize 128
    // with 32x16x16 and 256 with 32x32x16.
    for (flexible_dimensions) |prop| {
        if (prop.saturatingAccumulation != vk.VK_FALSE or prop.scope != vk.VK_SCOPE_WORKGROUP_KHR) continue;
        if (prop.AType == vk.VK_COMPONENT_TYPE_FLOAT16_KHR and prop.BType == vk.VK_COMPONENT_TYPE_FLOAT16_KHR) {
            if (prop.workgroupInvocations == 128 and prop.MGranularity <= 32 and prop.NGranularity <= 16 and prop.KGranularity <= 16) {
                if (prop.CType == vk.VK_COMPONENT_TYPE_FLOAT16_KHR and prop.ResultType == vk.VK_COMPONENT_TYPE_FLOAT16_KHR) found_fp16_128 = true;
                if (prop.CType == vk.VK_COMPONENT_TYPE_FLOAT32_KHR and prop.ResultType == vk.VK_COMPONENT_TYPE_FLOAT32_KHR) found_fp32_128 = true;
            }
            if (prop.workgroupInvocations == 256 and prop.MGranularity <= 32 and prop.NGranularity <= 32 and prop.KGranularity <= 16) {
                if (prop.CType == vk.VK_COMPONENT_TYPE_FLOAT16_KHR and prop.ResultType == vk.VK_COMPONENT_TYPE_FLOAT16_KHR) found_fp16_256 = true;
                if (prop.CType == vk.VK_COMPONENT_TYPE_FLOAT32_KHR and prop.ResultType == vk.VK_COMPONENT_TYPE_FLOAT32_KHR) found_fp32_256 = true;
            }
        }
        if (glslc.bfloat16 and prop.AType == vk.VK_COMPONENT_TYPE_BFLOAT16_KHR and prop.BType == vk.VK_COMPONENT_TYPE_BFLOAT16_KHR and
            prop.CType == vk.VK_COMPONENT_TYPE_FLOAT32_KHR and prop.ResultType == vk.VK_COMPONENT_TYPE_FLOAT32_KHR)
        {
            if (prop.workgroupInvocations == 128 and prop.MGranularity <= 32 and prop.NGranularity <= 16 and prop.KGranularity <= 16) found_bf16_128 = true;
            if (prop.workgroupInvocations == 256 and prop.MGranularity <= 32 and prop.NGranularity <= 32 and prop.KGranularity <= 16) found_bf16_256 = true;
        }
    }
    if (found_fp16_128 and found_fp16_256 and found_fp32_128 and found_fp32_256 and
        coopmat2_props.cooperativeMatrixFlexibleDimensionsMaxDimension >= 512)
    {
        device.coopmat2 = true;
        device.coopmat2_bf16_support = found_bf16_128 and found_bf16_256;
        device.coopmat2_decode_vector = decode_vector_support and decode_vector_features.cooperativeMatrixDecodeVector != 0;
    }
}

/// The KHR coopmat shape query inside `ggml_vk_get_device`: picks the
/// first fp16 subgroup shape and records which accumulators share it.
fn coopmatShapes(device: *Device) void {
    const pfn: vk.PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = @ptrCast(common.vkGetInstanceProcAddr(m.instance.instance.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    var cm_props_num: u32 = 0;
    _ = pfn.?(device.physical_device, &cm_props_num, null);
    const cm_props = allocator.alloc(vk.VkCooperativeMatrixPropertiesKHR, cm_props_num) catch impl.abort("ggml_vulkan: out of memory");
    defer allocator.free(cm_props);
    // The C++ value-initialises the vector, then sets each `sType`.
    for (cm_props) |*p| p.* = common.structure(vk.VkCooperativeMatrixPropertiesKHR, vk.VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR);
    _ = pfn.?(device.physical_device, &cm_props_num, cm_props.ptr);

    for (cm_props) |prop| {
        if (prop.AType == vk.VK_COMPONENT_TYPE_FLOAT16_KHR and prop.BType == vk.VK_COMPONENT_TYPE_FLOAT16_KHR and prop.scope == vk.VK_SCOPE_SUBGROUP_KHR) {
            if (prop.CType == vk.VK_COMPONENT_TYPE_FLOAT32_KHR and prop.ResultType == vk.VK_COMPONENT_TYPE_FLOAT32_KHR) {
                // coopmat sizes not set yet
                if (device.coopmat_m == 0) {
                    device.coopmat_acc_f32_support = true;
                    device.coopmat_m = prop.MSize;
                    device.coopmat_n = prop.NSize;
                    device.coopmat_k = prop.KSize;
                } else if (device.coopmat_m == prop.MSize and device.coopmat_n == prop.NSize and device.coopmat_k == prop.KSize) {
                    // Only enable if shape is identical
                    device.coopmat_acc_f32_support = true;
                }
                if (prop.MSize == 16 and prop.NSize == 16 and prop.KSize == 16) device.coopmat_support_16x16x16_f32acc = true;
            } else if (prop.CType == vk.VK_COMPONENT_TYPE_FLOAT16_KHR and prop.ResultType == vk.VK_COMPONENT_TYPE_FLOAT16_KHR) {
                if (device.coopmat_m == 0) {
                    device.coopmat_acc_f16_support = true;
                    device.coopmat_m = prop.MSize;
                    device.coopmat_n = prop.NSize;
                    device.coopmat_k = prop.KSize;
                } else if (device.coopmat_m == prop.MSize and device.coopmat_n == prop.NSize and device.coopmat_k == prop.KSize) {
                    device.coopmat_acc_f16_support = true;
                }
                if (prop.MSize == 16 and prop.NSize == 16 and prop.KSize == 16) device.coopmat_support_16x16x16_f16acc = true;
            }
        } else if (prop.AType == vk.VK_COMPONENT_TYPE_SINT8_KHR and prop.BType == vk.VK_COMPONENT_TYPE_SINT8_KHR and
            prop.CType == vk.VK_COMPONENT_TYPE_SINT32_KHR and prop.ResultType == vk.VK_COMPONENT_TYPE_SINT32_KHR and
            prop.scope == vk.VK_SCOPE_SUBGROUP_KHR and device.coopmat_int_m == 0)
        {
            device.coopmat_int_support = true;
            device.coopmat_int_m = prop.MSize;
            device.coopmat_int_n = prop.NSize;
            device.coopmat_int_k = prop.KSize;
        }
        if (glslc.bfloat16 and prop.AType == vk.VK_COMPONENT_TYPE_BFLOAT16_KHR and prop.BType == vk.VK_COMPONENT_TYPE_BFLOAT16_KHR and
            prop.CType == vk.VK_COMPONENT_TYPE_FLOAT32_KHR and prop.ResultType == vk.VK_COMPONENT_TYPE_FLOAT32_KHR and
            prop.scope == vk.VK_SCOPE_SUBGROUP_KHR)
        {
            if (device.coopmat_m == 0) {
                device.coopmat_bf16_support = true;
                device.coopmat_m = prop.MSize;
                device.coopmat_n = prop.NSize;
                device.coopmat_k = prop.KSize;
            } else if (device.coopmat_m == prop.MSize and device.coopmat_n == prop.NSize and device.coopmat_k == prop.KSize) {
                device.coopmat_bf16_support = true;
            }
        }
    }

    if (device.coopmat_m == 0 or !device.coopmat_acc_f32_support) {
        // No suitable matmul mode found
        impl.logDebug("ggml_vulkan: WARNING: No suitable matrix core mode found. Disabling matrix cores.\n", .{});
        device.coopmat_support = false;
    }
    if (common.envSet("GGML_VK_DISABLE_BFLOAT16")) device.coopmat_bf16_support = false;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "findQueueFamilyIndex prefers a family without the avoided flags" {
    const props = [_]vk.VkQueueFamilyProperties{
        .{ .queueFlags = vk.VK_QUEUE_GRAPHICS_BIT | vk.VK_QUEUE_COMPUTE_BIT | vk.VK_QUEUE_TRANSFER_BIT, .queueCount = 16, .timestampValidBits = 0, .minImageTransferGranularity = .{ .width = 0, .height = 0, .depth = 0 } },
        .{ .queueFlags = vk.VK_QUEUE_COMPUTE_BIT | vk.VK_QUEUE_TRANSFER_BIT, .queueCount = 8, .timestampValidBits = 0, .minImageTransferGranularity = .{ .width = 0, .height = 0, .depth = 0 } },
        .{ .queueFlags = vk.VK_QUEUE_TRANSFER_BIT, .queueCount = 2, .timestampValidBits = 0, .minImageTransferGranularity = .{ .width = 0, .height = 0, .depth = 0 } },
    };
    const compute = findQueueFamilyIndex(&props, vk.VK_QUEUE_COMPUTE_BIT, vk.VK_QUEUE_GRAPHICS_BIT, -1, 1);
    try std.testing.expectEqual(@as(u32, 1), compute);
    try std.testing.expectEqual(@as(u32, 2), findQueueFamilyIndex(&props, vk.VK_QUEUE_TRANSFER_BIT, vk.VK_QUEUE_COMPUTE_BIT | vk.VK_QUEUE_GRAPHICS_BIT, @intCast(compute), 1));
}
