//! `ggml_vk_load_shaders`: the walk over every pipeline the device can
//! use. Each call records a pipeline's name, layout and dispatch shape the
//! first time; only the pipeline asked for is compiled, by whichever
//! caller claims it first, with the lock dropped.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.3.0
//! (`c1d0e7a00`). The C++ spells each family of pipelines as a macro
//! pasting the shader's variable name together (`NAMELC ## F16ACC ##
//! _cm2_data`); the port pastes the same name at comptime and looks it up
//! in the generated shader table with `@field`, so a misspelt name is a
//! compile error rather than a wrong shader.
//!
//! **The order of the calls is part of the behaviour.** A pipeline slot
//! that two calls name is claimed by the first, with that call's arguments,
//! so the walk keeps the C++'s order -- including where a table loop stands
//! in for a run of macro invocations.

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");
const S = @import("vulkan_shaders");

const c = impl.c;
const common = m.common;
const vk = common.vk;
const allocator = common.allocator;
const Device = m.device.Device;
const Pipeline = m.pipeline.Pipeline;
const MatmulPipeline = m.pipeline.MatmulPipeline;
const MatmulPipeline2 = m.pipeline.MatmulPipeline2;
const glslc = m.instance.glslc;
const push = m.push;

fn oom() noreturn {
    impl.abort("ggml_vulkan: out of memory");
}

/// A warptile: the C++'s `std::vector<uint32_t>` of spec constants.
pub const Tile = struct {
    v: [16]u32 = undefined,
    n: usize = 0,

    pub fn of(vals: anytype) Tile {
        var t: Tile = .{};
        inline for (vals) |x| {
            t.v[t.n] = x;
            t.n += 1;
        }
        return t;
    }

    pub fn slice(self: *const Tile) []const u32 {
        return self.v[0..self.n];
    }

    pub fn at(self: Tile, i: usize) u32 {
        return self.v[i];
    }

    fn push_(self: *Tile, x: u32) void {
        self.v[self.n] = x;
        self.n += 1;
    }

    fn insert(self: *Tile, at_: usize, x: u32) void {
        var i = self.n;
        while (i > at_) : (i -= 1) self.v[i] = self.v[i - 1];
        self.v[at_] = x;
        self.n += 1;
    }
};

/// Ports `ggml_vk_matmul_shmem_support` (ggml-vulkan.cpp:3930 @c1d0e7a00):
/// whether a float matmul tile fits in shared memory.
pub fn matmulShmemSupport(device: *const Device, warptile: Tile, mul_mat_id: bool, src0_type: c.ggml_type) bool {
    const lut_size: u32 = switch (src0_type) {
        // Regular matmul uses the compact uint16_t IQ1 grid.
        c.GGML_TYPE_IQ1_S, c.GGML_TYPE_IQ1_M => 2 * 2048,
        c.GGML_TYPE_IQ2_XXS => 8 * 256,
        c.GGML_TYPE_IQ2_XS => 8 * 512,
        c.GGML_TYPE_IQ2_S => 8 * 1024,
        c.GGML_TYPE_IQ3_XXS => 4 * 256,
        c.GGML_TYPE_IQ3_S => 4 * 512,
        c.GGML_TYPE_IQ4_NL, c.GGML_TYPE_IQ4_XS, c.GGML_TYPE_MXFP4 => 4 * 16,
        // Same kvalues budget as MXFP4 plus ue4m3_fp32_lut[128].
        c.GGML_TYPE_NVFP4 => 4 * 16 + 128 * @sizeOf(f32),
        else => 0,
    };

    // Needs to be kept up to date on shader changes
    const intel_shmem_stride_pad_zero = device.vendor_id == common.vendor_id.intel and device.coopmat_support and
        device.driver_id == vk.VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS;
    const bank_conflict_offset: u32 = if (intel_shmem_stride_pad_zero) 0 else if (device.coopmat_support) 8 else 1;
    const type_size: u32 = if (device.fp16) @sizeOf(c.ggml_fp16_t) else @sizeOf(f32);
    const warps = warptile.at(0) / warptile.at(10);

    const load_bufs = (warptile.at(1) + warptile.at(2)) * (warptile.at(3) + bank_conflict_offset) * type_size;
    const mmid_row_ids: u32 = if (mul_mat_id) warptile.at(2) * 2 * @sizeOf(u16) else 0;
    const coopmat_stage: u32 = if (device.coopmat_support) warptile.at(7) * warptile.at(8) / warps * @sizeOf(f32) else 0;
    const ballots_sh: u32 = if (mul_mat_id) warps * 4 * @sizeOf(u32) else 0;

    const total_size = load_bufs + mmid_row_ids + coopmat_stage + lut_size + ballots_sh;
    return total_size <= device.properties.limits.maxComputeSharedMemorySize;
}

/// Ports `ggml_vk_matmul_int_shmem_support` (ggml-vulkan.cpp:3994 @c1d0e7a00):
/// the same question for the q8_1 integer-dot matmul's layout.
pub fn matmulIntShmemSupport(device: *const Device, warptile: Tile, mul_mat_id: bool, src0_type: c.ggml_type) bool {
    // FLOAT_TYPE in the shader is float16_t with fp16 support, otherwise float.
    const fp_size: u32 = if (device.fp16) 2 else 4;
    const fp_align = fp_size;
    const fp2_size = 2 * fp_size;
    const fp2_align: u32 = if (device.fp16) 4 else 8;

    const Member = struct { size: u32, @"align": u32 };
    const std430Size = struct {
        fn f(members: []const Member) u32 {
            var off: u32 = 0;
            var struct_align: u32 = 1;
            for (members) |mem| {
                off = (off + mem.@"align" - 1) & ~(mem.@"align" - 1);
                off += mem.size;
                struct_align = @max(struct_align, mem.@"align");
            }
            return (off + struct_align - 1) & ~(struct_align - 1);
        }
    }.f;

    const block_a_size: u32 = switch (src0_type) {
        c.GGML_TYPE_Q2_0 => std430Size(&.{ .{ .size = 32, .@"align" = 4 }, .{ .size = fp_size, .@"align" = fp_align } }),
        c.GGML_TYPE_Q4_0 => std430Size(&.{ .{ .size = 16, .@"align" = 4 }, .{ .size = fp_size, .@"align" = fp_align } }),
        c.GGML_TYPE_Q4_1 => std430Size(&.{ .{ .size = 16, .@"align" = 4 }, .{ .size = fp2_size, .@"align" = fp2_align } }),
        c.GGML_TYPE_Q5_0 => std430Size(&.{ .{ .size = 16, .@"align" = 4 }, .{ .size = 4, .@"align" = 4 }, .{ .size = fp_size, .@"align" = fp_align } }),
        c.GGML_TYPE_Q5_1 => std430Size(&.{ .{ .size = 16, .@"align" = 4 }, .{ .size = 4, .@"align" = 4 }, .{ .size = fp2_size, .@"align" = fp2_align } }),
        c.GGML_TYPE_Q8_0 => std430Size(&.{ .{ .size = 32, .@"align" = 4 }, .{ .size = fp_size, .@"align" = fp_align } }),
        c.GGML_TYPE_MXFP4 => std430Size(&.{ .{ .size = 32, .@"align" = 4 }, .{ .size = fp_size, .@"align" = fp_align } }),
        c.GGML_TYPE_Q2_K => std430Size(&.{ .{ .size = 8, .@"align" = 4 }, .{ .size = 2, .@"align" = 2 }, .{ .size = fp2_size, .@"align" = fp2_align } }),
        c.GGML_TYPE_Q3_K, c.GGML_TYPE_Q4_K => std430Size(&.{ .{ .size = 16, .@"align" = 4 }, .{ .size = fp2_size, .@"align" = fp2_align } }),
        c.GGML_TYPE_Q5_K, c.GGML_TYPE_Q6_K => std430Size(&.{ .{ .size = 32, .@"align" = 4 }, .{ .size = fp2_size, .@"align" = fp2_align } }),
        else => return false,
    };

    // block_b_cache: { int32_t qs[8]; FLOAT_TYPEV2 ds; }
    const block_b_size = std430Size(&.{ .{ .size = 32, .@"align" = 4 }, .{ .size = fp2_size, .@"align" = fp2_align } });

    const BM = warptile.at(1);
    const BN = warptile.at(2);
    // mul_mmq.comp: BK_STEP=1 for MUL_MAT_ID, 4 otherwise.
    const BK_STEP: u32 = if (mul_mat_id) 1 else 4;

    const buf_a_size = BM * BK_STEP * block_a_size;
    const buf_b_size = BN * BK_STEP * block_b_size;
    const mmid_row_ids: u32 = if (mul_mat_id) BN * 2 * @sizeOf(u16) else 0;
    const warps = warptile.at(0) / warptile.at(10);
    const ballots_sh: u32 = if (mul_mat_id) warps * 4 * @sizeOf(u32) else 0;

    const total_size = buf_a_size + buf_b_size + mmid_row_ids + ballots_sh;
    return total_size <= device.properties.limits.maxComputeSharedMemorySize;
}

/// Ports `get_fa_spec_constants` (ggml-vulkan.cpp:3905 @c1d0e7a00).
pub fn faSpecConstants(state: m.device.FaPipelineState) Tile {
    const blockBytes = struct {
        fn f(t: c.ggml_type) u32 {
            if (t == c.GGML_TYPE_F32) return 16;
            return @intCast(c.ggml_type_size(t));
        }
    }.f;
    return Tile.of(.{
        state.workgroup_size, // 0 WorkGroupSize
        state.Br, // 1 Br
        state.Bc, // 2 Bc
        state.HSK, // 3 HSK
        state.HSV, // 4 HSV
        @as(u32, @intFromBool(!state.aligned)), // 5 Clamp
        state.D_split, // 6 D_split
        state.row_split, // 7 row_split
        state.subgroup_size, // 8 SubGroupSize
        @as(u32, @intFromBool(state.shmem_staging)), // 9 SHMEM_STAGING
        state.flags, // 10 Flags
        state.limit_occupancy_shmem, // 11 LIMIT_OCCUPANCY_SHMEM
        @as(u32, @intCast(state.k_type)), // 12 FaTypeK
        @as(u32, @intCast(state.v_type)), // 13 FaTypeV
        blockBytes(state.k_type), // 14 FaBlockBytesK
        blockBytes(state.v_type), // 15 FaBlockBytesV
    });
}

/// Ports `ggml_vk_fa_type_needs_shmem` (ggml-vulkan.cpp:4123 @c1d0e7a00).
fn faTypeNeedsShmem(t: c.ggml_type) bool {
    return t == c.GGML_TYPE_IQ4_NL;
}

/// Ports `ggml_vk_fa_scalar_uses_mmq` (ggml-vulkan.cpp:4132 @c1d0e7a00):
/// whether scalar flash attention takes the integer-dot path.
pub fn faScalarUsesMmq(device: *const Device, k_type: c.ggml_type, v_type: c.ggml_type) bool {
    if (comptime !glslc.integer_dot) return false;
    return device.integer_dot_product and device.subgroup_clustered and
        !faTypeNeedsShmem(v_type) and
        (k_type == c.GGML_TYPE_Q4_0 or k_type == c.GGML_TYPE_Q4_1 or
            k_type == c.GGML_TYPE_Q5_0 or k_type == c.GGML_TYPE_Q5_1 or
            k_type == c.GGML_TYPE_Q8_0);
}

/// One `ggml_vk_create_pipeline` call's arguments, after the name.
const Args = struct {
    spv: []const u8,
    entrypoint: [:0]const u8 = "main",
    parameter_count: u32,
    push_constant_size: u32,
    wg_denoms: [3]u32,
    spec: []const u32,
    @"align": u32 = 1,
    disable_robustness: bool = false,
    require_full_subgroups: bool = false,
    required_subgroup_size: u32 = 0,
};

/// The state of one walk: what `ggml_vk_load_shaders` keeps in locals.
const Walk = struct {
    device: *Device,
    requested: ?*Pipeline,
    wait_pipeline: ?*Pipeline = null,
    claimed: ?struct { pipeline: *Pipeline, args: m.pipeline.CompileArgs } = null,

    /// The `ggml_vk_create_pipeline` lambda (ggml-vulkan.cpp:4391 @c1d0e7a00):
    /// records the pipeline's description on first sight, and claims it for
    /// compilation if it is the one requested and nobody has yet.
    fn create(w: *Walk, base: *?*Pipeline, name: []const u8, a: Args) void {
        const device = w.device;
        var required_subgroup_size = a.required_subgroup_size;
        if (!a.require_full_subgroups and required_subgroup_size == 0) {
            required_subgroup_size = m.instance.getSubgroupSize(name, device.architecture);
        }

        const num_pipelines: usize = if (comptime @hasDecl(vk, "VK_EXT_shader_64bit_indexing")) (if (device.shader_64b_indexing) 2 else 1) else 1;
        var ptr = base;
        for (0..num_pipelines) |i| {
            if (ptr.* == null) ptr.* = Pipeline.create();
            const pipeline = ptr.*.?;
            defer ptr = &pipeline.next;
            if (!pipeline.initialized) {
                pipeline.name = allocator.dupeSentinel(u8, name, 0) catch oom();
                pipeline.parameter_count = a.parameter_count;
                pipeline.push_constant_size = a.push_constant_size;
                pipeline.wg_denoms = a.wg_denoms;
                pipeline.@"align" = a.@"align";
                pipeline.initialized = true;
                pipeline.is_64b_indexing = i == 1;
            }

            // We only care about the pipeline this call asked for; the rest
            // are handled by their own request_descriptor_sets calls.
            if (pipeline != w.requested) continue;
            if (pipeline.compiled.load(.seq_cst)) continue;

            w.wait_pipeline = pipeline;
            if (!pipeline.compile_pending) {
                pipeline.compile_pending = true;
                w.claimed = .{ .pipeline = pipeline, .args = .{
                    .spv = a.spv,
                    .entrypoint = a.entrypoint,
                    .parameter_count = a.parameter_count,
                    .wg_denoms = a.wg_denoms,
                    .specialization_constants = allocator.dupe(u32, a.spec) catch oom(),
                    .disable_robustness = a.disable_robustness,
                    .require_full_subgroups = a.require_full_subgroups,
                    .required_subgroup_size = required_subgroup_size,
                } };
            }
        }
    }

    /// The `ggml_vk_create_pipeline2` lambda, for names built at run time.
    fn createFmt(w: *Walk, base: *?*Pipeline, comptime fmt: []const u8, fargs: anytype, a: Args) void {
        var buf: [128]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, fmt, fargs) catch unreachable;
        w.create(base, name, a);
    }
};

/// The per-walk tile tables: the `l_`/`m_`/`s_` warptiles, workgroup
/// denominators and alignments the matmul macros paste names from.
const Tiles = struct {
    l_warptile: Tile = .{},
    m_warptile: Tile = .{},
    s_warptile: Tile = .{},
    l_warptile_id: Tile = .{},
    m_warptile_id: Tile = .{},
    s_warptile_id: Tile = .{},
    l_warptile_mmq: Tile = .{},
    m_warptile_mmq: Tile = .{},
    s_warptile_mmq: Tile = .{},
    l_warptile_mmq_int: Tile = .{},
    m_warptile_mmq_int: Tile = .{},
    s_warptile_mmq_int: Tile = .{},
    l_warptile_mmq_int_k: Tile = .{},
    m_warptile_mmq_int_k: Tile = .{},
    s_warptile_mmq_int_k: Tile = .{},
    l_warptile_mmq_k: Tile = .{},
    m_warptile_mmq_k: Tile = .{},
    s_warptile_mmq_k: Tile = .{},
    l_warptile_mmqid: Tile = .{},
    m_warptile_mmqid: Tile = .{},
    s_warptile_mmqid: Tile = .{},
    l_warptile_mmqid_int: Tile = .{},
    m_warptile_mmqid_int: Tile = .{},
    s_warptile_mmqid_int: Tile = .{},
    l_warptile_mmqid_int_k: Tile = .{},
    m_warptile_mmqid_int_k: Tile = .{},
    s_warptile_mmqid_int_k: Tile = .{},
    l_wg_denoms: [3]u32 = undefined,
    m_wg_denoms: [3]u32 = undefined,
    s_wg_denoms: [3]u32 = undefined,
    l_mmq_wg_denoms: [3]u32 = undefined,
    m_mmq_wg_denoms: [3]u32 = undefined,
    s_mmq_wg_denoms: [3]u32 = undefined,
    l_mmq_wg_denoms_k: [3]u32 = undefined,
    m_mmq_wg_denoms_k: [3]u32 = undefined,
    s_mmq_wg_denoms_k: [3]u32 = undefined,
    l_mmqid_wg_denoms: [3]u32 = undefined,
    m_mmqid_wg_denoms: [3]u32 = undefined,
    s_mmqid_wg_denoms: [3]u32 = undefined,
    l_align: u32 = 0,
    m_align: u32 = 0,
    s_align: u32 = 0,
};

/// The three sizes and their aligned twins, in the macros' call order.
const sizes = [_]struct { size: []const u8, slot: []const u8, aligned: bool }{
    .{ .size = "l", .slot = "l", .aligned = false },
    .{ .size = "m", .slot = "m", .aligned = false },
    .{ .size = "s", .slot = "s", .aligned = false },
    .{ .size = "l", .slot = "a_l", .aligned = true },
    .{ .size = "m", .slot = "a_m", .aligned = true },
    .{ .size = "s", .slot = "a_s", .aligned = true },
};

/// `ggml_vk_mul_mm_spec` (ggml-vulkan.cpp:4589 @c1d0e7a00): the matmul spec
/// constants with ALIGNED, and the Intel Windows shared-memory switches.
fn mulMmSpec(device: *const Device, spec_in: Tile, aligned: bool) Tile {
    var spec = spec_in;
    spec.push_(@intFromBool(aligned)); // constantID=11: ALIGNED
    if (device.vendor_id == common.vendor_id.intel and device.coopmat_support and
        device.driver_id == vk.VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS)
    {
        spec.push_(0); // constantID=12: SHMEM_STRIDE_PAD = 0
        spec.push_(1); // constantID=13: APPLY_SLM_A_RESHAPE = true
    }
    return spec;
}

/// `ggml_vk_mul_mm_cm2_spec` (ggml-vulkan.cpp:4603 @c1d0e7a00).
fn mulMmCm2Spec(spec_in: Tile, aligned: bool, mul_mat_id: bool) Tile {
    var spec = spec_in;
    if (mul_mat_id and spec.n > 5) {
        spec.insert(5, @intFromBool(aligned));
    } else {
        spec.push_(@intFromBool(aligned));
    }
    if (mul_mat_id and spec.n == 6) spec.push_(32);
    return spec;
}

const mul_mat_id_param_count = 5;

/// Which matmul macro a family uses.
const MmKind = enum {
    /// coopmat2's `CREATE_MM`: no tile-size switches, robustness off.
    cm2,
    /// coopmat1's `CREATE_MM`: tile-size switches, full subgroups.
    cm1,
    /// The fp16 scalar `CREATE_MM`: the dot2 variant when the device has it.
    fp16,
    /// The fp16 scalar `CREATE_MM_NODOT2`.
    fp16_nodot2,
    /// The fp32 scalar `CREATE_MM`.
    fp32,
};

/// The six-pipeline matmul macros (`CREATE_MM`, `CREATE_MM_NODOT2`):
/// `{s,m,l} x {unaligned,aligned}`, under the tile-size switches of
/// `ID` for every kind but coopmat2.
fn createMm(w: *Walk, t: *const Tiles, comptime kind: MmKind, ty: c.ggml_type, p: *MatmulPipeline, comptime namelc: []const u8, comptime f16acc: []const u8, comptime wg_denoms: []const u8, comptime warptile: []const u8, push_size: u32, param_count: u32, comptime id: []const u8, req_subgroup_size: u32) void {
    const device = w.device;
    const spv: []const u8 = switch (kind) {
        .cm2 => @field(S, namelc ++ f16acc ++ "_cm2_data"),
        .cm1 => @field(S, namelc ++ f16acc ++ "_cm1_data"),
        .fp16 => if (device.dot2_f16) @field(S, namelc ++ "_dot2" ++ f16acc ++ "_data") else @field(S, namelc ++ f16acc ++ "_data"),
        .fp16_nodot2 => @field(S, namelc ++ f16acc ++ "_data"),
        .fp32 => @field(S, namelc ++ f16acc ++ "_fp32_data"),
    };
    inline for (sizes) |sz| {
        const enabled = kind == .cm2 or @field(device, "mul_mat" ++ id ++ "_" ++ sz.size)[@intCast(ty)];
        if (enabled) {
            const tile = @field(t, sz.size ++ "_" ++ warptile);
            const spec = if (kind == .cm2) mulMmCm2Spec(tile, sz.aligned, param_count == mul_mat_id_param_count) else mulMmSpec(device, tile, sz.aligned);
            const name = namelc ++ f16acc ++ (if (sz.aligned) "_aligned_" else "_") ++ sz.size;
            w.create(&@field(p, sz.slot), name, .{
                .spv = spv,
                .parameter_count = param_count,
                .push_constant_size = push_size,
                .wg_denoms = @field(t, sz.size ++ "_" ++ wg_denoms),
                .spec = spec.slice(),
                .@"align" = if (sz.aligned) @field(t, sz.size ++ "_align") else 1,
                .disable_robustness = kind == .cm2,
                .require_full_subgroups = switch (kind) {
                    .cm2 => false,
                    .cm1 => true,
                    else => req_subgroup_size > 0,
                },
                .required_subgroup_size = switch (kind) {
                    .cm2, .cm1 => 0,
                    else => req_subgroup_size,
                },
            });
        }
    }
}

/// `CREATE_MM2`: both accumulators, f16acc first. coopmat1 asks the device
/// which accumulators it supports.
fn createMm2(w: *Walk, t: *const Tiles, comptime kind: MmKind, ty: c.ggml_type, p: MatmulPipeline2, comptime namelc: []const u8, comptime wg_denoms: []const u8, comptime warptile: []const u8, push_size: u32, param_count: u32, comptime id: []const u8, req_subgroup_size: u32) void {
    if (kind != .cm1 or w.device.coopmat_acc_f16_support) {
        createMm(w, t, kind, ty, p.f16acc, namelc, "_f16acc", wg_denoms, warptile, push_size, param_count, id, req_subgroup_size);
    }
    if (kind != .cm1 or w.device.coopmat_acc_f32_support) {
        createMm(w, t, kind, ty, p.f32acc, namelc, "", wg_denoms, warptile, push_size, param_count, id, req_subgroup_size);
    }
}

/// The q8_1 matmul macro (`CREATE_MMQ`): three sizes, unaligned only,
/// under the integer-dot tile switches. The fp16 path forces full
/// subgroups when given a size; the fp32 path takes the defaults.
fn createMmq(w: *Walk, t: *const Tiles, comptime fp32: bool, ty: c.ggml_type, p: *MatmulPipeline, comptime namelc: []const u8, comptime wg_denoms: []const u8, comptime warptile: []const u8, push_size: u32, param_count: u32, comptime id: []const u8, req_subgroup_size: u32) void {
    const device = w.device;
    const spv: []const u8 = if (fp32) @field(S, namelc ++ "_fp32_data") else @field(S, namelc ++ "_data");
    inline for (sizes[0..3]) |sz| {
        if (@field(device, "mul_mat" ++ id ++ "_" ++ sz.size ++ "_int")[@intCast(ty)]) {
            w.create(&@field(p, sz.slot), namelc ++ "_" ++ sz.size, .{
                .spv = spv,
                .parameter_count = param_count,
                .push_constant_size = push_size,
                .wg_denoms = @field(t, sz.size ++ "_" ++ wg_denoms),
                .spec = @field(t, sz.size ++ "_" ++ warptile).slice(),
                .require_full_subgroups = !fp32 and req_subgroup_size > 0,
                .required_subgroup_size = if (fp32) 0 else req_subgroup_size,
            });
        }
    }
}

/// One row of a quantized-type matmul table: the type, its shader-name
/// fragment, and which tile family it uses.
const MmRow = struct { ty: c.ggml_type, frag: []const u8, k: bool = false };

/// The quantized types of the dequant matmul tables, in the C++'s order.
/// `k` marks the rows that use the `_k` tiles under coopmat2.
const mm_quant_rows = [_]MmRow{
    .{ .ty = c.GGML_TYPE_Q1_0, .frag = "q1_0" },
    .{ .ty = c.GGML_TYPE_Q2_0, .frag = "q2_0" },
    .{ .ty = c.GGML_TYPE_Q4_0, .frag = "q4_0" },
    .{ .ty = c.GGML_TYPE_Q4_1, .frag = "q4_1" },
    .{ .ty = c.GGML_TYPE_Q5_0, .frag = "q5_0" },
    .{ .ty = c.GGML_TYPE_Q5_1, .frag = "q5_1" },
    .{ .ty = c.GGML_TYPE_Q8_0, .frag = "q8_0" },
    .{ .ty = c.GGML_TYPE_Q2_K, .frag = "q2_k", .k = true },
    .{ .ty = c.GGML_TYPE_TQ2_0, .frag = "tq2_0", .k = true },
    .{ .ty = c.GGML_TYPE_Q3_K, .frag = "q3_k", .k = true },
    .{ .ty = c.GGML_TYPE_Q4_K, .frag = "q4_k", .k = true },
    .{ .ty = c.GGML_TYPE_Q5_K, .frag = "q5_k", .k = true },
    .{ .ty = c.GGML_TYPE_Q6_K, .frag = "q6_k", .k = true },
    .{ .ty = c.GGML_TYPE_IQ1_S, .frag = "iq1_s" },
    .{ .ty = c.GGML_TYPE_IQ1_M, .frag = "iq1_m" },
    .{ .ty = c.GGML_TYPE_IQ2_XXS, .frag = "iq2_xxs" },
    .{ .ty = c.GGML_TYPE_IQ2_XS, .frag = "iq2_xs" },
    .{ .ty = c.GGML_TYPE_IQ2_S, .frag = "iq2_s" },
    .{ .ty = c.GGML_TYPE_IQ3_XXS, .frag = "iq3_xxs" },
    .{ .ty = c.GGML_TYPE_IQ3_S, .frag = "iq3_s" },
    .{ .ty = c.GGML_TYPE_IQ4_XS, .frag = "iq4_xs" },
    .{ .ty = c.GGML_TYPE_IQ4_NL, .frag = "iq4_nl" },
};

/// The fp4 rows, which have an `_ocp` variant when the device does OCP fp4.
const mm_fp4_rows = [_]MmRow{
    .{ .ty = c.GGML_TYPE_MXFP4, .frag = "mxfp4" },
    .{ .ty = c.GGML_TYPE_NVFP4, .frag = "nvfp4" },
};

/// The integer-dot (q8_1) matmul rows: the types and whether they take
/// the K-quant tiles.
const mmq_rows = [_]MmRow{
    .{ .ty = c.GGML_TYPE_Q2_0, .frag = "q2_0" },
    .{ .ty = c.GGML_TYPE_Q4_0, .frag = "q4_0" },
    .{ .ty = c.GGML_TYPE_Q4_1, .frag = "q4_1" },
    .{ .ty = c.GGML_TYPE_Q5_0, .frag = "q5_0" },
    .{ .ty = c.GGML_TYPE_Q5_1, .frag = "q5_1" },
    .{ .ty = c.GGML_TYPE_Q8_0, .frag = "q8_0" },
    .{ .ty = c.GGML_TYPE_MXFP4, .frag = "mxfp4" },
    .{ .ty = c.GGML_TYPE_Q2_K, .frag = "q2_k", .k = true },
    .{ .ty = c.GGML_TYPE_Q3_K, .frag = "q3_k", .k = true },
    .{ .ty = c.GGML_TYPE_Q4_K, .frag = "q4_k", .k = true },
    .{ .ty = c.GGML_TYPE_Q5_K, .frag = "q5_k", .k = true },
    .{ .ty = c.GGML_TYPE_Q6_K, .frag = "q6_k", .k = true },
};

/// The integer-dot rows of the fp32 path, which has no MXFP4.
const mmq_rows_fp32 = [_]MmRow{
    mmq_rows[0], mmq_rows[1], mmq_rows[2], mmq_rows[3],  mmq_rows[4],  mmq_rows[5],
    mmq_rows[7], mmq_rows[8], mmq_rows[9], mmq_rows[10], mmq_rows[11],
};

fn ocp(device: *const Device) bool {
    return glslc.float_e2m1 and glslc.float_e4m3 and device.ocp_fp4;
}

/// Ports `ggml_vk_load_shaders` (ggml-vulkan.cpp:4164 @c1d0e7a00).
///
/// Parameters:
/// - `device`: the device whose pipeline slots are walked.
/// - `requested`: the pipeline to compile, or null to only describe them.
pub fn loadShaders(device: *Device, requested: ?*Pipeline) void {
    // some shaders have a minimum subgroup size
    const subgroup_size_8 = @max(device.subgroup_size, 8);
    const subgroup_size_16 = @max(device.subgroup_size, 16);
    const subgroup_size_32 = @max(device.subgroup_size, 32);

    const mul_mat_subgroup_size = if (device.vendor_id == common.vendor_id.intel and device.subgroup_size_control) device.subgroup_min_size else device.subgroup_size;
    const mul_mat_subgroup_size_8 = @max(mul_mat_subgroup_size, 8);
    const mul_mat_subgroup_size_16 = @max(mul_mat_subgroup_size, 16);
    const mul_mat_subgroup_size_32 = @max(mul_mat_subgroup_size, 32);

    const subgroup_min_size_16 = (!device.subgroup_size_control and device.subgroup_size >= 16) or
        (device.subgroup_size_control and device.subgroup_max_size >= 16);

    var t: Tiles = .{};
    var w: Walk = .{ .device = device, .requested = requested };

    // The rest of the walk reads and writes shared device state, so hold
    // the lock until we're done deciding what to compile.
    _ = std.c.pthread_mutex_lock(&device.compile_mutex);

    if (device.coopmat2) {
        // spec constants and tile sizes for non-quant matmul/matmul_id
        t.l_warptile = .of(.{ 256, 128, 256, 64, 1 });
        t.m_warptile = .of(.{ 256, 128, 128, 64, 0 });
        t.s_warptile = .of(.{ 128, 64, 64, 64, 0 });
        t.l_wg_denoms = .{ 128, 256, 1 };
        t.m_wg_denoms = .{ 128, 128, 1 };
        t.s_wg_denoms = .{ 64, 64, 1 };

        // spec constants and tile sizes for quant matmul (non-Qi_K)
        t.l_warptile_mmq = .of(.{ 256, 128, 256, 64, 1 });
        t.m_warptile_mmq = .of(.{ 256, 128, 128, 64, 1 });
        t.s_warptile_mmq = .of(.{ 256, 32, 64, 128, 0 });
        t.l_mmq_wg_denoms = .{ 128, 256, 1 };
        t.m_mmq_wg_denoms = .{ 128, 128, 1 };
        t.s_mmq_wg_denoms = .{ 32, 64, 1 };

        // spec constants and tile sizes for quant matmul (Qi_K)
        t.l_warptile_mmq_k = .of(.{ 256, 128, 256, 64, 1 });
        t.m_warptile_mmq_k = .of(.{ 256, 128, 128, 64, 1 });
        t.s_warptile_mmq_k = .of(.{ 256, 32, 64, 128, 0 });
        t.l_mmq_wg_denoms_k = .{ 128, 256, 1 };
        t.m_mmq_wg_denoms_k = .{ 128, 128, 1 };
        t.s_mmq_wg_denoms_k = .{ 32, 64, 1 };

        // spec constants and tile sizes for quant matmul_id
        const mmqid_bk: u32 = if (device.coopmat2_decode_vector) 64 else 32;
        t.l_warptile_mmqid = .of(.{ 256, 128, 128, mmqid_bk, 1, device.subgroup_size });
        t.m_warptile_mmqid = .of(.{ 256, 128, 64, mmqid_bk, 0, device.subgroup_size });
        t.s_warptile_mmqid = .of(.{ 256, 128, 64, mmqid_bk, 0, device.subgroup_size });
        t.l_mmqid_wg_denoms = .{ 128, 128, 1 };
        t.m_mmqid_wg_denoms = .{ 128, 64, 1 };
        t.s_mmqid_wg_denoms = .{ 128, 64, 1 };

        t.l_align = 128;
        t.m_align = 64;
        t.s_align = 32;
    } else {
        // Matrix cores require different warp group sizes
        const cm = device.coopmat_support;
        const tm_l: u32 = if (cm) device.coopmat_m else 4;
        const tm_m: u32 = if (cm) device.coopmat_m else 4;
        const tm_s: u32 = if (cm) device.coopmat_m else 2;
        const tn_l: u32 = if (cm) device.coopmat_n else 4;
        const tn_m: u32 = if (cm) device.coopmat_n else 2;
        const tn_s: u32 = if (cm) device.coopmat_n else 2;
        const tk_l: u32 = if (cm) device.coopmat_k else 1;
        const tk_m: u32 = if (cm) device.coopmat_k else 1;
        const tk_s: u32 = if (cm) device.coopmat_k else 1;

        const s_warptile_wm: u32 = if (device.subgroup_size == 8) 8 else 32;

        t.l_warptile = .of(.{ 128, 128, 128, 16, subgroup_size_8 * 2, 64, 2, tm_l, tn_l, tk_l, subgroup_size_8 });
        t.m_warptile = .of(.{ 128, 64, 64, 16, subgroup_size_8, 32, 2, tm_m, tn_m, tk_m, subgroup_size_8 });
        t.s_warptile = .of(.{ subgroup_size_32, 32, 32, 16, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_size_8 });

        t.l_warptile_mmq = .of(.{ 128, 128, 128, 32, subgroup_size_8 * 2, 64, 2, tm_l, tn_l, tk_l, subgroup_size_8 });
        t.m_warptile_mmq = .of(.{ 128, 64, 64, 32, subgroup_size_8, 32, 2, tm_m, tn_m, tk_m, subgroup_size_8 });
        t.s_warptile_mmq = .of(.{ subgroup_size_32, 32, 32, 32, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_size_8 });

        // Integer MMQ has a smaller shared memory profile, but heavier register use
        t.l_warptile_mmq_int = .of(.{ 128, 128, 128, 32, subgroup_size_8 * 2, 64, 2, 4, 4, 1, subgroup_size_8 });
        t.m_warptile_mmq_int = .of(.{ 128, 64, 64, 32, subgroup_size_8, 32, 2, 2, 2, 1, subgroup_size_8 });
        t.s_warptile_mmq_int = .of(.{ subgroup_size_32, 32, 32, 32, s_warptile_wm, 32, 2, 2, 1, 1, subgroup_size_8 });

        // K-quants use even more registers, mitigate by setting WMITER to 1
        t.l_warptile_mmq_int_k = .of(.{ 128, 128, 128, 32, subgroup_size_8 * 2, 64, 1, 4, 4, 1, subgroup_size_8 });
        t.m_warptile_mmq_int_k = .of(.{ 128, 64, 64, 32, subgroup_size_8, 32, 1, 2, 2, 1, subgroup_size_8 });
        t.s_warptile_mmq_int_k = .of(.{ subgroup_size_32, 32, 32, 32, s_warptile_wm, 32, 1, 2, 1, 1, subgroup_size_8 });

        t.l_warptile_id = .of(.{ 128, 128, 128, 16, mul_mat_subgroup_size_16 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_subgroup_size_16 });
        t.m_warptile_id = .of(.{ 128, 64, 64, 16, mul_mat_subgroup_size_16, 32, 2, tm_m, tn_m, tk_m, mul_mat_subgroup_size_16 });
        t.s_warptile_id = .of(.{ mul_mat_subgroup_size_16, 32, 32, 16, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_size_16 });

        t.l_warptile_mmqid = .of(.{ 128, 128, 128, 32, mul_mat_subgroup_size_8 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_subgroup_size_8 });
        t.m_warptile_mmqid = .of(.{ 128, 64, 64, 32, mul_mat_subgroup_size_8, 32, 2, tm_m, tn_m, tk_m, mul_mat_subgroup_size_8 });
        t.s_warptile_mmqid = .of(.{ mul_mat_subgroup_size_32, 32, 32, 32, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_size_8 });

        t.l_warptile_mmqid_int = .of(.{ 128, 128, 128, 32, mul_mat_subgroup_size_8 * 2, 64, 2, 4, 4, 1, mul_mat_subgroup_size_8 });
        t.m_warptile_mmqid_int = .of(.{ 128, 64, 64, 32, mul_mat_subgroup_size_8, 32, 2, 2, 2, 1, mul_mat_subgroup_size_8 });
        t.s_warptile_mmqid_int = .of(.{ mul_mat_subgroup_size_32, 32, 32, 32, s_warptile_wm, 32, 2, 2, 1, 1, mul_mat_subgroup_size_8 });

        t.l_warptile_mmqid_int_k = .of(.{ 128, 128, 128, 32, mul_mat_subgroup_size_16 * 2, 64, 1, 4, 4, 1, mul_mat_subgroup_size_16 });
        t.m_warptile_mmqid_int_k = .of(.{ 128, 64, 64, 32, mul_mat_subgroup_size_16, 32, 1, 2, 2, 1, mul_mat_subgroup_size_16 });
        t.s_warptile_mmqid_int_k = .of(.{ mul_mat_subgroup_size_32, 32, 32, 32, s_warptile_wm, 32, 1, 2, 1, 1, mul_mat_subgroup_size_16 });

        // chip specific tuning
        if (device.architecture == .amd_gcn and device.driver_id != vk.VK_DRIVER_ID_AMD_PROPRIETARY) {
            t.m_warptile_mmq = .of(.{ 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 });
            t.m_warptile_mmq_int = t.m_warptile_mmq;
            t.m_warptile_mmqid = .of(.{ 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 });
            t.m_warptile_mmqid_int = t.m_warptile_mmqid;
        } else if (device.vendor_id == common.vendor_id.amd and device.coopmat_support and device.driver_id != vk.VK_DRIVER_ID_AMD_PROPRIETARY) {
            // This is intentionally using tx_m values, slight performance increase
            t.l_warptile = .of(.{ 256, 128, 128, 16, subgroup_size_8, 64, 2, tm_m, tn_m, tk_m, subgroup_size_8 });
            t.l_warptile_mmq = .of(.{ 256, 128, 128, 32, subgroup_size_8, 64, 2, tm_m, tn_m, tk_m, subgroup_size_8 });
            t.l_warptile_mmq_int = t.l_warptile_mmq;
            t.l_warptile_mmq_int_k = .of(.{ 256, 128, 128, 32, subgroup_size_16, 64, 1, 4, 2, 1, subgroup_size_16 });
        } else if (device.vendor_id == common.vendor_id.intel and device.coopmat_support) {
            // Xe2/Xe3 with coopmat enabled - warptile performance tuning
            t.l_warptile = .of(.{ 512, 128, 128, 16, subgroup_size_8, 32, 2, tm_m, tn_m, tk_m, subgroup_size_8 });
            t.l_warptile_mmq = .of(.{ 512, 128, 128, 32, subgroup_size_8, 32, 2, tm_m, tn_m, tk_m, subgroup_size_8 });
        }

        t.l_wg_denoms = .{ 128, 128, 1 };
        t.l_mmq_wg_denoms = t.l_wg_denoms;
        t.m_wg_denoms = .{ 64, 64, 1 };
        t.m_mmq_wg_denoms = t.m_wg_denoms;
        t.s_wg_denoms = .{ 32, 32, 1 };
        t.s_mmq_wg_denoms = t.s_wg_denoms;
        t.l_align = 128;
        t.m_align = 64;
        t.s_align = 32;

        for (0..c.GGML_TYPE_COUNT) |i| {
            const ty: c.ggml_type = @intCast(i);
            // Disable medium and large matrix multiplication if not enough
            // shared memory is available; throw if not enough for any.
            if (!matmulShmemSupport(device, t.s_warptile_mmq, false, ty)) {
                impl.fprintfStderr("ggml_vulkan: Error: Shared memory size too small for matrix multiplication.\n", .{});
                impl.abort("Shared memory size too small for matrix multiplication.");
            } else if (!matmulShmemSupport(device, t.m_warptile_mmq, false, ty)) {
                device.mul_mat_m[i] = false;
                device.mul_mat_l[i] = false;
            } else if (!matmulShmemSupport(device, t.l_warptile_mmq, false, ty)) {
                device.mul_mat_l[i] = false;
            }

            // Disable mul_mat_id if not enough shared memory is available
            if (!matmulShmemSupport(device, t.s_warptile_mmqid, true, ty)) {
                device.mul_mat_id_s[i] = false;
                device.mul_mat_id_m[i] = false;
                device.mul_mat_id_l[i] = false;
            } else if (!matmulShmemSupport(device, t.m_warptile_mmqid, true, ty)) {
                device.mul_mat_id_m[i] = false;
                device.mul_mat_id_l[i] = false;
            } else if (!matmulShmemSupport(device, t.l_warptile_mmqid, true, ty)) {
                device.mul_mat_id_l[i] = false;
            }

            // The q8_1 mmq path has its own (larger) shmem layout.
            const is_k_quant = ty == c.GGML_TYPE_Q2_K or ty == c.GGML_TYPE_Q3_K or
                ty == c.GGML_TYPE_Q4_K or ty == c.GGML_TYPE_Q5_K or ty == c.GGML_TYPE_Q6_K;
            const s_int = if (is_k_quant) t.s_warptile_mmq_int_k else t.s_warptile_mmq_int;
            const m_int = if (is_k_quant) t.m_warptile_mmq_int_k else t.m_warptile_mmq_int;
            const l_int = if (is_k_quant) t.l_warptile_mmq_int_k else t.l_warptile_mmq_int;
            const s_intid = if (is_k_quant) t.s_warptile_mmqid_int_k else t.s_warptile_mmqid_int;
            const m_intid = if (is_k_quant) t.m_warptile_mmqid_int_k else t.m_warptile_mmqid_int;
            const l_intid = if (is_k_quant) t.l_warptile_mmqid_int_k else t.l_warptile_mmqid_int;

            if (!matmulIntShmemSupport(device, s_int, false, ty)) {
                device.mul_mat_s_int[i] = false;
                device.mul_mat_m_int[i] = false;
                device.mul_mat_l_int[i] = false;
            } else if (!matmulIntShmemSupport(device, m_int, false, ty)) {
                device.mul_mat_m_int[i] = false;
                device.mul_mat_l_int[i] = false;
            } else if (!matmulIntShmemSupport(device, l_int, false, ty)) {
                device.mul_mat_l_int[i] = false;
            }

            if (!matmulIntShmemSupport(device, s_intid, true, ty)) {
                device.mul_mat_id_s_int[i] = false;
                device.mul_mat_id_m_int[i] = false;
                device.mul_mat_id_l_int[i] = false;
            } else if (!matmulIntShmemSupport(device, m_intid, true, ty)) {
                device.mul_mat_id_m_int[i] = false;
                device.mul_mat_id_l_int[i] = false;
            } else if (!matmulIntShmemSupport(device, l_intid, true, ty)) {
                device.mul_mat_id_l_int[i] = false;
            }
        }
    }

    if (device.pipeline_matmul_f32 == null) device.pipeline_matmul_f32 = MatmulPipeline.create();
    if (device.pipeline_matmul_f32_f16 == null) device.pipeline_matmul_f32_f16 = MatmulPipeline.create();
    if (device.pipeline_matmul_id_f32 == null) device.pipeline_matmul_id_f32 = MatmulPipeline.create();
    if (device.pipeline_matmul_bf16 == null) device.pipeline_matmul_bf16 = MatmulPipeline.create();
    if (device.pipeline_matmul_id_bf16 == null) device.pipeline_matmul_id_bf16 = MatmulPipeline.create();

    walkFlashAttention(&w);
    walkMatmul(&w, &t, subgroup_min_size_16, mul_mat_subgroup_size, mul_mat_subgroup_size_16, subgroup_size_8, subgroup_size_32);
    walkMulMatVec(&w, subgroup_min_size_16);
    walkRest(&w, subgroup_min_size_16, subgroup_size_16, subgroup_size_32);

    // Compile the claimed pipeline with the lock dropped, or wait for
    // whoever claimed it.
    finish(&w);
}

/// The three flash-attention loops over the pipelines the device's FA
/// cache has asked for: scalar, coopmat1, coopmat2.
fn walkFlashAttention(w: *Walk) void {
    const device = w.device;
    const pc_size: u32 = @sizeOf(push.FlashAttn);

    var it = device.pipeline_flash_attn_f32_f16.iterator();
    while (it.next()) |fa| {
        const st = fa.key_ptr.*;
        if (st.path != .scalar) continue;
        const fa_sgs = st.subgroup_size;
        const fa_ds = st.subgroup_size == 0;
        const bf16_kv = st.k_type == c.GGML_TYPE_BF16;
        const use_mmq = faScalarUsesMmq(device, st.k_type, st.v_type);
        var spv: []const u8 = &.{};
        var name: []const u8 = undefined;
        if (bf16_kv) {
            spv = S.flash_attn_f32_f16_fp32_data;
            name = if (st.aligned) "flash_attn_f32_bf16_aligned" else "flash_attn_f32_bf16";
        } else if (use_mmq) {
            if (comptime glslc.integer_dot) {
                spv = if (device.fp16) (if (st.f32acc) S.flash_attn_f32_f16_int8_data else S.flash_attn_f32_f16_f16acc_int8_data) else S.flash_attn_f32_f16_fp32_int8_data;
            }
            name = if (st.aligned) "flash_attn_f32_f16_aligned" else "flash_attn_f32_f16";
        } else {
            if (device.fp16) {
                if (device.dot2_f16) {
                    spv = if (st.f32acc) S.flash_attn_f32_f16_dot2_data else S.flash_attn_f32_f16_dot2_f16acc_data;
                } else {
                    spv = if (st.f32acc) S.flash_attn_f32_f16_data else S.flash_attn_f32_f16_f16acc_data;
                }
            } else {
                spv = S.flash_attn_f32_f16_fp32_data;
            }
            name = if (st.aligned) "flash_attn_f32_f16_aligned" else "flash_attn_f32_f16";
        }
        const spec = faSpecConstants(st);
        w.create(fa.value_ptr, name, .{ .spv = spv, .parameter_count = 7, .push_constant_size = pc_size, .wg_denoms = .{ st.Br, 1, 1 }, .spec = spec.slice(), .@"align" = if (st.aligned) st.Bc else 1, .disable_robustness = true, .require_full_subgroups = !fa_ds, .required_subgroup_size = if (!fa_ds) fa_sgs else 0 });
    }

    if (comptime glslc.coopmat) {
        if (device.coopmat1_fa_support) {
            it = device.pipeline_flash_attn_f32_f16.iterator();
            while (it.next()) |fa| {
                const st = fa.key_ptr.*;
                if (st.path != .coopmat1) continue;
                const fa_sgs = st.subgroup_size;
                const fa_ds = st.subgroup_size == 0;
                var spv: []const u8 = undefined;
                var name: []const u8 = undefined;
                if (st.k_type == c.GGML_TYPE_BF16) {
                    if (comptime !glslc.bfloat16) continue;
                    if (!device.coopmat_bf16_support) continue;
                    spv = S.flash_attn_f32_f16_bf16_cm1_data;
                    name = if (st.aligned) "flash_attn_f32_bf16_aligned_cm1" else "flash_attn_f32_bf16_cm1";
                } else {
                    spv = if (st.f32acc) S.flash_attn_f32_f16_cm1_data else S.flash_attn_f32_f16_f16acc_cm1_data;
                    name = if (st.aligned) "flash_attn_f32_f16_aligned_cm1" else "flash_attn_f32_f16_cm1";
                }
                const spec = faSpecConstants(st);
                w.create(fa.value_ptr, name, .{ .spv = spv, .parameter_count = 7, .push_constant_size = pc_size, .wg_denoms = .{ st.Br, 1, 1 }, .spec = spec.slice(), .@"align" = if (st.aligned) st.Bc else 1, .disable_robustness = true, .require_full_subgroups = !fa_ds, .required_subgroup_size = if (!fa_ds) fa_sgs else 0 });
            }
        }
    }

    if (comptime glslc.coopmat2) {
        if (device.coopmat2) {
            it = device.pipeline_flash_attn_f32_f16.iterator();
            while (it.next()) |fa| {
                const st = fa.key_ptr.*;
                if (st.path != .coopmat2) continue;
                var spv: []const u8 = undefined;
                var name: []const u8 = undefined;
                if (st.k_type == c.GGML_TYPE_BF16) {
                    if (comptime !glslc.bfloat16) continue;
                    if (!device.coopmat2_bf16_support) continue;
                    spv = S.flash_attn_f32_f16_bf16_cm2_data;
                    name = if (st.aligned) "flash_attn_f32_bf16_aligned_cm2" else "flash_attn_f32_bf16_cm2";
                } else if (st.aligned) {
                    spv = if (st.f32acc) S.flash_attn_f32_f16_cm2_data else S.flash_attn_f32_f16_f16acc_cm2_data;
                    name = if (st.f32acc) "flash_attn_f32_f16_aligned_f32acc_cm2" else "flash_attn_f32_f16_aligned_f16acc_cm2";
                } else {
                    spv = if (st.f32acc) S.flash_attn_f32_f16_cm2_data else S.flash_attn_f32_f16_f16acc_cm2_data;
                    name = if (st.f32acc) "flash_attn_f32_f16_f32acc_cm2" else "flash_attn_f32_f16_f16acc_cm2";
                }
                const spec = faSpecConstants(st);
                w.create(fa.value_ptr, name, .{ .spv = spv, .parameter_count = 7, .push_constant_size = pc_size, .wg_denoms = .{ st.Br, 1, 1 }, .spec = spec.slice(), .@"align" = if (st.aligned) st.Bc else 1, .disable_robustness = true });
            }
        }
    }
}

/// The dequant matmul tables of one path: every quantized row with
/// `CREATE_MM2` (or `CREATE_MM` on `.f32acc` for fp32), then the fp4 rows
/// with their OCP variant.
fn mmQuantRows(w: *Walk, t: *const Tiles, comptime kind: MmKind, pipes: *[c.GGML_TYPE_COUNT]MatmulPipeline2, comptime prefix: []const u8, comptime suffix: []const u8, comptime wg: []const u8, comptime wg_k: []const u8, comptime tile: []const u8, comptime tile_k: []const u8, push_size: u32, param_count: u32, comptime id: []const u8, req_sg: u32) void {
    inline for (mm_quant_rows) |row| {
        const wgd = if (row.k) wg_k else wg;
        const tl = if (row.k) tile_k else tile;
        if (kind == .fp32) {
            createMm(w, t, kind, row.ty, pipes[@intCast(row.ty)].f32acc, prefix ++ row.frag ++ suffix, "", wgd, tl, push_size, param_count, id, req_sg);
        } else {
            createMm2(w, t, kind, row.ty, pipes[@intCast(row.ty)], prefix ++ row.frag ++ suffix, wgd, tl, push_size, param_count, id, req_sg);
        }
    }
    // The scalar paths have no OCP variant.
    const has_ocp = (kind == .cm2 or kind == .cm1) and ocp(w.device);
    inline for (mm_fp4_rows) |row| {
        if (kind == .fp32) {
            createMm(w, t, kind, row.ty, pipes[@intCast(row.ty)].f32acc, prefix ++ row.frag ++ suffix, "", wg, tile, push_size, param_count, id, req_sg);
        } else if (has_ocp) {
            if (comptime glslc.float_e2m1 and glslc.float_e4m3) {
                createMm2(w, t, kind, row.ty, pipes[@intCast(row.ty)], prefix ++ row.frag ++ suffix ++ "_ocp", wg, tile, push_size, param_count, id, req_sg);
            }
        } else {
            createMm2(w, t, kind, row.ty, pipes[@intCast(row.ty)], prefix ++ row.frag ++ suffix, wg, tile, push_size, param_count, id, req_sg);
        }
    }
}

/// The matmul and matmul_id pipelines of whichever path the device takes:
/// coopmat2, coopmat1, fp16 scalar or fp32 scalar; then the bf16 fallback.
fn walkMatmul(w: *Walk, t: *Tiles, subgroup_min_size_16: bool, mul_mat_subgroup_size: u32, mul_mat_subgroup_size_16: u32, subgroup_size_8: u32, subgroup_size_32: u32) void {
    const device = w.device;
    const pc: u32 = @sizeOf(push.MatMat);
    const pcid: u32 = @sizeOf(push.MatMatId);
    const midpc = mul_mat_id_param_count;

    if (glslc.coopmat2 and device.coopmat2) {
        if (comptime glslc.coopmat2) {
            createMm2(w, t, .cm2, c.GGML_TYPE_F16, device.pipeline_matmul_f16, "matmul_f16", "wg_denoms", "warptile", pc, 3, "", 0);
            if (comptime glslc.bfloat16) {
                if (device.coopmat_bf16_support) createMm(w, t, .cm2, c.GGML_TYPE_BF16, device.pipeline_matmul_bf16.?, "matmul_bf16", "", "wg_denoms", "warptile", pc, 3, "", 0);
            }
            mmQuantRows(w, t, .cm2, &device.pipeline_dequant_mul_mat_mat_f16, "matmul_", "_f16", "mmq_wg_denoms", "mmq_wg_denoms_k", "warptile_mmq", "warptile_mmq_k", pc, 3, "", 0);

            impl.assert(device.subgroup_ballot, "device->subgroup_ballot");

            createMm2(w, t, .cm2, c.GGML_TYPE_F16, device.pipeline_matmul_id_f16, "matmul_id_subgroup_f16", "wg_denoms", "warptile", pcid, 5, "", 0);
            if (comptime glslc.bfloat16) {
                if (device.coopmat_bf16_support) createMm(w, t, .cm2, c.GGML_TYPE_BF16, device.pipeline_matmul_id_bf16.?, "matmul_id_subgroup_bf16", "", "wg_denoms", "warptile", pcid, 5, "", 0);
            }
            mmQuantRows(w, t, .cm2, &device.pipeline_dequant_mul_mat_mat_id, "matmul_id_subgroup_", "_f16", "mmqid_wg_denoms", "mmqid_wg_denoms", "warptile_mmqid", "warptile_mmqid", pcid, 5, "", 0);
        }
    } else if (glslc.coopmat and device.coopmat_support) {
        if (comptime glslc.coopmat) {
            createMm(w, t, .cm1, c.GGML_TYPE_F32, device.pipeline_matmul_f32.?, "matmul_f32_f32", "", "wg_denoms", "warptile", pc, 3, "", 0);
            createMm(w, t, .cm1, c.GGML_TYPE_F32, device.pipeline_matmul_f32_f16.?, "matmul_f32_f16", "", "wg_denoms", "warptile", pc, 3, "", 0);
            createMm2(w, t, .cm1, c.GGML_TYPE_F16, device.pipeline_matmul_f16, "matmul_f16", "wg_denoms", "warptile", pc, 3, "", 0);
            createMm2(w, t, .cm1, c.GGML_TYPE_F16, device.pipeline_matmul_f16_f32, "matmul_f16_f32", "wg_denoms", "warptile", pc, 3, "", 0);
            if (comptime glslc.bfloat16) {
                if (device.coopmat_bf16_support) createMm(w, t, .cm1, c.GGML_TYPE_BF16, device.pipeline_matmul_bf16.?, "matmul_bf16", "", "wg_denoms", "warptile", pc, 3, "", 0);
            }
            mmQuantRows(w, t, .cm1, &device.pipeline_dequant_mul_mat_mat, "matmul_", "_f32", "mmq_wg_denoms", "mmq_wg_denoms", "warptile_mmq", "warptile_mmq", pc, 3, "", 0);

            impl.assert(device.subgroup_ballot, "device->subgroup_ballot");

            createMm(w, t, .cm1, c.GGML_TYPE_F32, device.pipeline_matmul_id_f32.?, "matmul_id_subgroup_f32_f32", "", "wg_denoms", "warptile", pcid, midpc, "_id", 0);
            createMm2(w, t, .cm1, c.GGML_TYPE_F16, device.pipeline_matmul_id_f16, "matmul_id_subgroup_f16", "wg_denoms", "warptile", pcid, midpc, "_id", 0);
            createMm2(w, t, .cm1, c.GGML_TYPE_F16, device.pipeline_matmul_id_f16_f32, "matmul_id_subgroup_f16_f32", "wg_denoms", "warptile", pcid, midpc, "_id", 0);
            if (comptime glslc.bfloat16) {
                if (device.coopmat_bf16_support) createMm(w, t, .cm1, c.GGML_TYPE_BF16, device.pipeline_matmul_id_bf16.?, "matmul_id_subgroup_bf16", "", "wg_denoms", "warptile", pcid, midpc, "_id", 0);
            }
            mmQuantRows(w, t, .cm1, &device.pipeline_dequant_mul_mat_mat_id, "matmul_id_subgroup_", "_f32", "mmq_wg_denoms", "mmq_wg_denoms", "warptile_mmq", "warptile_mmq", pcid, midpc, "_id", 0);
        }
    } else if (device.fp16) {
        createMm(w, t, .fp16, c.GGML_TYPE_F32, device.pipeline_matmul_f32.?, "matmul_f32_f32", "", "wg_denoms", "warptile", pc, 3, "", 0);
        createMm(w, t, .fp16, c.GGML_TYPE_F32, device.pipeline_matmul_f32_f16.?, "matmul_f32_f16", "", "wg_denoms", "warptile", pc, 3, "", 0);
        createMm2(w, t, .fp16, c.GGML_TYPE_F16, device.pipeline_matmul_f16, "matmul_f16", "wg_denoms", "warptile", pc, 3, "", 0);
        createMm2(w, t, .fp16, c.GGML_TYPE_F16, device.pipeline_matmul_f16_f32, "matmul_f16_f32", "wg_denoms", "warptile", pc, 3, "", 0);
        createMm(w, t, .fp16_nodot2, c.GGML_TYPE_BF16, device.pipeline_matmul_bf16.?, "matmul_bf16", "", "wg_denoms", "warptile", pc, 3, "", 0);
        mmQuantRows(w, t, .fp16, &device.pipeline_dequant_mul_mat_mat, "matmul_", "_f32", "mmq_wg_denoms", "mmq_wg_denoms", "warptile_mmq", "warptile_mmq", pc, 3, "", 0);

        if (comptime glslc.integer_dot) {
            if (device.integer_dot_product) {
                inline for (mmq_rows) |row| {
                    createMmq(w, t, false, row.ty, device.pipeline_dequant_mul_mat_mat_q8_1[@intCast(row.ty)].f32acc, "matmul_" ++ row.frag ++ "_q8_1", "mmq_wg_denoms", if (row.k) "warptile_mmq_int_k" else "warptile_mmq_int", pc, 3, "", 0);
                }
            }
        }

        if (device.subgroup_ballot and device.subgroup_require_full_support and subgroup_min_size_16) {
            createMm(w, t, .fp16, c.GGML_TYPE_F32, device.pipeline_matmul_id_f32.?, "matmul_id_subgroup_f32_f32", "", "wg_denoms", "warptile_id", pcid, midpc, "_id", mul_mat_subgroup_size_16);
            createMm2(w, t, .fp16, c.GGML_TYPE_F16, device.pipeline_matmul_id_f16, "matmul_id_subgroup_f16", "wg_denoms", "warptile_id", pcid, midpc, "_id", mul_mat_subgroup_size_16);
            createMm2(w, t, .fp16, c.GGML_TYPE_F16, device.pipeline_matmul_id_f16_f32, "matmul_id_subgroup_f16_f32", "wg_denoms", "warptile_id", pcid, midpc, "_id", mul_mat_subgroup_size_16);
            createMm(w, t, .fp16_nodot2, c.GGML_TYPE_BF16, device.pipeline_matmul_id_bf16.?, "matmul_id_subgroup_bf16", "", "wg_denoms", "warptile_id", pcid, midpc, "_id", mul_mat_subgroup_size_16);
            mmQuantRows(w, t, .fp16, &device.pipeline_dequant_mul_mat_mat_id, "matmul_id_subgroup_", "_f32", "mmq_wg_denoms", "mmq_wg_denoms", "warptile_mmqid", "warptile_mmqid", pcid, midpc, "_id", mul_mat_subgroup_size);

            if (comptime glslc.integer_dot) {
                if (device.integer_dot_product) {
                    inline for (mmq_rows) |row| {
                        createMmq(w, t, false, row.ty, device.pipeline_dequant_mul_mat_mat_id_q8_1[@intCast(row.ty)].f32acc, "matmul_id_subgroup_" ++ row.frag ++ "_q8_1", "mmq_wg_denoms", if (row.k) "warptile_mmqid_int_k" else "warptile_mmqid_int", pcid, midpc, "_id", if (row.k) mul_mat_subgroup_size_16 else mul_mat_subgroup_size);
                    }
                }
            }
        } else {
            createMm(w, t, .fp16, c.GGML_TYPE_F32, device.pipeline_matmul_id_f32.?, "matmul_id_f32_f32", "", "wg_denoms", "warptile", pcid, midpc, "_id", 0);
            createMm2(w, t, .fp16, c.GGML_TYPE_F16, device.pipeline_matmul_id_f16, "matmul_id_f16", "wg_denoms", "warptile", pcid, midpc, "_id", 0);
            createMm2(w, t, .fp16, c.GGML_TYPE_F16, device.pipeline_matmul_id_f16_f32, "matmul_id_f16_f32", "wg_denoms", "warptile", pcid, midpc, "_id", 0);
            createMm(w, t, .fp16_nodot2, c.GGML_TYPE_BF16, device.pipeline_matmul_id_bf16.?, "matmul_id_bf16", "", "wg_denoms", "warptile", pcid, midpc, "_id", 0);
            mmQuantRows(w, t, .fp16, &device.pipeline_dequant_mul_mat_mat_id, "matmul_id_", "_f32", "mmq_wg_denoms", "mmq_wg_denoms", "warptile_mmqid", "warptile_mmqid", pcid, midpc, "_id", 0);

            if (comptime glslc.integer_dot) {
                if (device.integer_dot_product) {
                    inline for (mmq_rows) |row| {
                        createMmq(w, t, false, row.ty, device.pipeline_dequant_mul_mat_mat_id_q8_1[@intCast(row.ty)].f32acc, "matmul_id_" ++ row.frag ++ "_q8_1", "mmq_wg_denoms", if (row.k) "warptile_mmqid_int_k" else "warptile_mmqid_int", pcid, midpc, "_id", 0);
                    }
                }
            }
        }
    } else {
        createMm(w, t, .fp32, c.GGML_TYPE_F32, device.pipeline_matmul_f32.?, "matmul_f32_f32", "", "wg_denoms", "warptile", pc, 3, "", 0);
        createMm(w, t, .fp32, c.GGML_TYPE_F32, device.pipeline_matmul_f32_f16.?, "matmul_f32_f16", "", "wg_denoms", "warptile", pc, 3, "", 0);
        createMm(w, t, .fp32, c.GGML_TYPE_F16, device.pipeline_matmul_f16.f32acc, "matmul_f16", "", "wg_denoms", "warptile", pc, 3, "", 0);
        createMm(w, t, .fp32, c.GGML_TYPE_F16, device.pipeline_matmul_f16_f32.f32acc, "matmul_f16_f32", "", "wg_denoms", "warptile", pc, 3, "", 0);
        createMm(w, t, .fp32, c.GGML_TYPE_BF16, device.pipeline_matmul_bf16.?, "matmul_bf16", "", "wg_denoms", "warptile", pc, 3, "", 0);
        mmQuantRows(w, t, .fp32, &device.pipeline_dequant_mul_mat_mat, "matmul_", "_f32", "mmq_wg_denoms", "mmq_wg_denoms", "warptile_mmq", "warptile_mmq", pc, 3, "", 0);

        if (comptime glslc.integer_dot) {
            if (device.integer_dot_product) {
                inline for (mmq_rows_fp32) |row| {
                    createMmq(w, t, true, row.ty, device.pipeline_dequant_mul_mat_mat_q8_1[@intCast(row.ty)].f32acc, "matmul_" ++ row.frag ++ "_q8_1", "mmq_wg_denoms", if (row.k) "warptile_mmq_int_k" else "warptile_mmq_int", pc, 3, "", 0);
                }
            }
        }

        walkMatmulIdFp32(w, t, subgroup_min_size_16, mul_mat_subgroup_size, mul_mat_subgroup_size_16);
    }

    // reusing CREATE_MM from the fp32 path
    if ((device.coopmat2 or device.coopmat_support) and !(glslc.bfloat16 and device.coopmat_bf16_support)) {
        const s_warptile_wm: u32 = if (device.subgroup_size == 8) 8 else 32;

        // use scalar tile sizes
        t.l_warptile = .of(.{ 128, 128, 128, 16, subgroup_size_8 * 2, 64, 2, 4, 4, 1, subgroup_size_8 });
        t.m_warptile = .of(.{ 128, 64, 64, 16, subgroup_size_8, 32, 2, 4, 2, 1, subgroup_size_8 });
        t.s_warptile = .of(.{ subgroup_size_32, 32, 32, 16, s_warptile_wm, 32, 2, 2, 2, 1, subgroup_size_8 });

        t.l_wg_denoms = .{ 128, 128, 1 };
        t.m_wg_denoms = .{ 64, 64, 1 };
        t.s_wg_denoms = .{ 32, 32, 1 };

        createMm(w, t, .fp32, c.GGML_TYPE_BF16, device.pipeline_matmul_bf16.?, "matmul_bf16", "", "wg_denoms", "warptile", pc, 3, "", 0);
        createMm(w, t, .fp32, c.GGML_TYPE_BF16, device.pipeline_matmul_id_bf16.?, "matmul_id_bf16", "", "wg_denoms", "warptile", pcid, midpc, "_id", 0);
    }
}

/// The fp32 path's matmul_id pipelines: the subgroup variants when the
/// device has full subgroups of at least 16, the plain ones otherwise.
fn walkMatmulIdFp32(w: *Walk, t: *const Tiles, subgroup_min_size_16: bool, mul_mat_subgroup_size: u32, mul_mat_subgroup_size_16: u32) void {
    const device = w.device;
    const pcid: u32 = @sizeOf(push.MatMatId);
    const midpc = mul_mat_id_param_count;
    if (device.subgroup_ballot and device.subgroup_require_full_support and subgroup_min_size_16) {
        createMm(w, t, .fp32, c.GGML_TYPE_F32, device.pipeline_matmul_id_f32.?, "matmul_id_subgroup_f32_f32", "", "wg_denoms", "warptile_id", pcid, midpc, "_id", mul_mat_subgroup_size_16);
        createMm(w, t, .fp32, c.GGML_TYPE_F16, device.pipeline_matmul_id_f16.f32acc, "matmul_id_subgroup_f16", "", "wg_denoms", "warptile_id", pcid, midpc, "_id", mul_mat_subgroup_size_16);
        createMm(w, t, .fp32, c.GGML_TYPE_F16, device.pipeline_matmul_id_f16_f32.f32acc, "matmul_id_subgroup_f16_f32", "", "wg_denoms", "warptile_id", pcid, midpc, "_id", mul_mat_subgroup_size_16);
        createMm(w, t, .fp32, c.GGML_TYPE_BF16, device.pipeline_matmul_id_bf16.?, "matmul_id_subgroup_bf16", "", "wg_denoms", "warptile_id", pcid, midpc, "_id", mul_mat_subgroup_size_16);
        mmQuantRows(w, t, .fp32, &device.pipeline_dequant_mul_mat_mat_id, "matmul_id_subgroup_", "_f32", "mmq_wg_denoms", "mmq_wg_denoms", "warptile_mmqid", "warptile_mmqid", pcid, midpc, "_id", mul_mat_subgroup_size);
    } else {
        createMm(w, t, .fp32, c.GGML_TYPE_F32, device.pipeline_matmul_id_f32.?, "matmul_id_f32_f32", "", "wg_denoms", "warptile", pcid, midpc, "_id", 0);
        createMm(w, t, .fp32, c.GGML_TYPE_F16, device.pipeline_matmul_id_f16.f32acc, "matmul_id_f16", "", "wg_denoms", "warptile", pcid, midpc, "_id", 0);
        createMm(w, t, .fp32, c.GGML_TYPE_F16, device.pipeline_matmul_id_f16_f32.f32acc, "matmul_id_f16_f32", "", "wg_denoms", "warptile", pcid, midpc, "_id", 0);
        createMm(w, t, .fp32, c.GGML_TYPE_BF16, device.pipeline_matmul_id_bf16.?, "matmul_id_bf16", "", "wg_denoms", "warptile", pcid, midpc, "_id", 0);
        mmQuantRows(w, t, .fp32, &device.pipeline_dequant_mul_mat_mat_id, "matmul_id_", "_f32", "mmq_wg_denoms", "mmq_wg_denoms", "warptile_mmqid", "warptile_mmqid", pcid, midpc, "_id", 0);
    }
}

/// Ports `shader_reduction_mode` (ggml-vulkan.cpp:618 @c1d0e7a00), the
/// index into a dmmv shader's three variants.
const ReductionMode = enum(usize) { shmem, hybrid, subgroup };

/// How many rows a dmmv shader computes: the C++'s `rm_*` choices.
const Rm = enum { one, two, stdq2, stdq1, kq, iq };

/// One row of the dequant mul_mat_vec tables.
const DmmvRow = struct {
    ty: c.ggml_type,
    frag: []const u8,
    rm: Rm,
    /// Takes the `16` subgroup variables (K- and I-quants).
    sixteen: bool = false,
    /// Robustness off: every quantized row.
    robust_off: bool = true,
    /// Has an `_ocp` variant.
    ocp: bool = false,
};

const dmmv_rows = [_]DmmvRow{
    .{ .ty = c.GGML_TYPE_F32, .frag = "f32", .rm = .one, .robust_off = false },
    .{ .ty = c.GGML_TYPE_F16, .frag = "f16", .rm = .two, .robust_off = false },
    .{ .ty = c.GGML_TYPE_BF16, .frag = "bf16", .rm = .two, .robust_off = false },
    .{ .ty = c.GGML_TYPE_Q1_0, .frag = "q1_0", .rm = .stdq2 },
    .{ .ty = c.GGML_TYPE_Q2_0, .frag = "q2_0", .rm = .stdq2 },
    .{ .ty = c.GGML_TYPE_Q4_0, .frag = "q4_0", .rm = .stdq2 },
    .{ .ty = c.GGML_TYPE_Q4_1, .frag = "q4_1", .rm = .stdq2 },
    .{ .ty = c.GGML_TYPE_Q5_0, .frag = "q5_0", .rm = .stdq2 },
    .{ .ty = c.GGML_TYPE_Q5_1, .frag = "q5_1", .rm = .stdq2 },
    .{ .ty = c.GGML_TYPE_Q8_0, .frag = "q8_0", .rm = .stdq1 },
    .{ .ty = c.GGML_TYPE_Q2_K, .frag = "q2_k", .rm = .kq, .sixteen = true },
    .{ .ty = c.GGML_TYPE_TQ2_0, .frag = "tq2_0", .rm = .kq, .sixteen = true },
    .{ .ty = c.GGML_TYPE_Q3_K, .frag = "q3_k", .rm = .kq, .sixteen = true },
    .{ .ty = c.GGML_TYPE_Q4_K, .frag = "q4_k", .rm = .kq, .sixteen = true },
    .{ .ty = c.GGML_TYPE_Q5_K, .frag = "q5_k", .rm = .kq, .sixteen = true },
    .{ .ty = c.GGML_TYPE_Q6_K, .frag = "q6_k", .rm = .kq, .sixteen = true },
    .{ .ty = c.GGML_TYPE_IQ1_S, .frag = "iq1_s", .rm = .iq, .sixteen = true },
    .{ .ty = c.GGML_TYPE_IQ1_M, .frag = "iq1_m", .rm = .iq, .sixteen = true },
    .{ .ty = c.GGML_TYPE_IQ2_XXS, .frag = "iq2_xxs", .rm = .iq, .sixteen = true },
    .{ .ty = c.GGML_TYPE_IQ2_XS, .frag = "iq2_xs", .rm = .iq, .sixteen = true },
    .{ .ty = c.GGML_TYPE_IQ2_S, .frag = "iq2_s", .rm = .iq, .sixteen = true },
    .{ .ty = c.GGML_TYPE_IQ3_XXS, .frag = "iq3_xxs", .rm = .iq, .sixteen = true },
    .{ .ty = c.GGML_TYPE_IQ3_S, .frag = "iq3_s", .rm = .iq, .sixteen = true },
    .{ .ty = c.GGML_TYPE_IQ4_XS, .frag = "iq4_xs", .rm = .iq, .sixteen = true },
    .{ .ty = c.GGML_TYPE_IQ4_NL, .frag = "iq4_nl", .rm = .iq, .sixteen = true },
    .{ .ty = c.GGML_TYPE_MXFP4, .frag = "mxfp4", .rm = .iq, .sixteen = true, .ocp = true },
    .{ .ty = c.GGML_TYPE_NVFP4, .frag = "nvfp4", .rm = .iq, .sixteen = true, .ocp = true },
};

/// How many rows an integer-dot dmmv shader computes.
const RmInt = enum { kq2, stdq1, stdq2, kq1, iq };

const dmmv_int_rows = [_]struct { ty: c.ggml_type, frag: []const u8, rm: RmInt }{
    .{ .ty = c.GGML_TYPE_Q2_0, .frag = "q2_0", .rm = .kq2 },
    .{ .ty = c.GGML_TYPE_Q4_0, .frag = "q4_0", .rm = .stdq1 },
    .{ .ty = c.GGML_TYPE_Q4_1, .frag = "q4_1", .rm = .stdq1 },
    .{ .ty = c.GGML_TYPE_Q5_0, .frag = "q5_0", .rm = .stdq1 },
    .{ .ty = c.GGML_TYPE_Q5_1, .frag = "q5_1", .rm = .stdq1 },
    .{ .ty = c.GGML_TYPE_Q8_0, .frag = "q8_0", .rm = .stdq1 },
    .{ .ty = c.GGML_TYPE_MXFP4, .frag = "mxfp4", .rm = .stdq2 },
    .{ .ty = c.GGML_TYPE_Q2_K, .frag = "q2_k", .rm = .kq2 },
    .{ .ty = c.GGML_TYPE_Q3_K, .frag = "q3_k", .rm = .kq1 },
    .{ .ty = c.GGML_TYPE_Q4_K, .frag = "q4_k", .rm = .kq1 },
    .{ .ty = c.GGML_TYPE_Q5_K, .frag = "q5_k", .rm = .kq1 },
    .{ .ty = c.GGML_TYPE_Q6_K, .frag = "q6_k", .rm = .kq1 },
    .{ .ty = c.GGML_TYPE_IQ1_S, .frag = "iq1_s", .rm = .iq },
    .{ .ty = c.GGML_TYPE_IQ1_M, .frag = "iq1_m", .rm = .iq },
};

/// A dmmv shader's variant for a reduction mode, with the OCP choice of
/// `OCP_DMMV_DATA`.
fn dmmvData(comptime base: []const u8, comptime has_ocp: bool, device: *const Device, reduc: ReductionMode) []const u8 {
    if (has_ocp and glslc.float_e2m1 and glslc.float_e4m3) {
        if (device.ocp_fp4) return @field(S, base ++ "_ocp_data")[@backingInt(reduc)];
    }
    return @field(S, base ++ "_data")[@backingInt(reduc)];
}

/// The mul_mat_vec pipelines: for each workgroup size, every column count
/// of the f32 and f16 B paths and the integer-dot path, then the id
/// variants.
fn walkMulMatVec(w: *Walk, subgroup_min_size_16: bool) void {
    const device = w.device;
    // the number of rows computed per shader depends on GPU model and quant
    var rm_stdq: u32 = 1;
    var rm_kq: u32 = 2;
    var rm_stdq_int: u32 = 1;
    const rm_kq_int: u32 = 1;
    const rmIqInt = struct {
        fn f(i: u32) u32 {
            return if (i == 0) 8 else 4;
        }
    }.f;
    if (device.vendor_id == common.vendor_id.amd) {
        if (device.architecture == .amd_gcn) {
            rm_stdq = 2;
            rm_kq = 4;
            rm_stdq_int = 4;
        }
    } else if (device.vendor_id == common.vendor_id.intel) {
        rm_stdq = 2;
        rm_stdq_int = 2;
    }
    const rm_iq = 2 * rm_kq;

    const use_subgroups = device.subgroup_arithmetic;
    // Ensure a subgroup size >= 16 is available
    const use_subgroups16 = use_subgroups and subgroup_min_size_16;

    const subgroup_size: u32 = if (device.vendor_id == common.vendor_id.intel and device.subgroup_size_control and device.subgroup_min_size <= 16 and device.subgroup_max_size >= 16) 16 else device.subgroup_size;
    const subgroup_size16 = @max(subgroup_size, 16);

    const force_subgroup_size: u32 = if (use_subgroups) subgroup_size else 0;
    const force_subgroup_size16: u32 = if (use_subgroups16) subgroup_size16 else 0;
    const mul_mat_vec_num_bindings = 5;
    const mul_mat_vec_id_num_bindings = 6;
    const pcv: u32 = @sizeOf(push.MatVec);
    const pcvid: u32 = @sizeOf(push.MatVecId);

    const rowsOf = struct {
        fn f(rm: Rm, stdq: u32, kq: u32, iq: u32) u32 {
            return switch (rm) {
                .one => 1,
                .two => 2,
                .stdq2 => 2 * stdq,
                .stdq1 => 1 * stdq,
                .kq => kq,
                .iq => iq,
            };
        }
    }.f;

    for (0..m.device.dmmv_wg_size_count) |wi| {
        const is_sub = wi == @backingInt(m.device.DmmvWgSize.subgroup);
        const wg_size_subgroup = if (is_sub) subgroup_size else subgroup_size * 4;
        const wg_size_subgroup16 = if (is_sub) subgroup_size16 else subgroup_size16 * 4;

        const reduc: ReductionMode = if (use_subgroups and is_sub) .subgroup else if (use_subgroups and !is_sub) .hybrid else .shmem;
        const reduc16: ReductionMode = if (use_subgroups16 and is_sub) .subgroup else if (use_subgroups16 and !is_sub) .hybrid else .shmem;

        for (0..m.device.mul_mat_vec_max_cols) |i| {
            const cols: u32 = @intCast(i + 1);
            inline for (.{ "f32_f32", "f16_f32" }) |b| {
                const table = if (comptime std.mem.eql(u8, b, "f32_f32")) &device.pipeline_dequant_mul_mat_vec_f32_f32 else &device.pipeline_dequant_mul_mat_vec_f16_f32;
                inline for (dmmv_rows) |row| {
                    const rows = rowsOf(row.rm, rm_stdq, rm_kq, rm_iq);
                    const wgs = if (row.sixteen) wg_size_subgroup16 else wg_size_subgroup;
                    const spec = [_]u32{ wgs, rows, cols };
                    w.create(&table[wi][@intCast(row.ty)][i], "mul_mat_vec_" ++ row.frag ++ "_" ++ b, .{
                        .spv = dmmvData("arr_dmmv_" ++ row.frag ++ "_" ++ b, row.ocp, device, if (row.sixteen) reduc16 else reduc),
                        .parameter_count = mul_mat_vec_num_bindings,
                        .push_constant_size = pcv,
                        .wg_denoms = .{ rows, 1, 1 },
                        .spec = &spec,
                        .disable_robustness = row.robust_off,
                        .require_full_subgroups = if (row.sixteen) use_subgroups16 else use_subgroups,
                        .required_subgroup_size = if (row.sixteen) force_subgroup_size16 else force_subgroup_size,
                    });
                }
            }

            if (comptime glslc.integer_dot) {
                if (device.integer_dot_product) {
                    const subgroup_size_int = if (device.vendor_id == common.vendor_id.intel and device.subgroup_size_control) device.subgroup_min_size else device.subgroup_size;
                    const wg_size_subgroup_int = if (is_sub) subgroup_size_int else subgroup_size_int * 4;
                    inline for (dmmv_int_rows) |row| {
                        const rows: u32 = switch (row.rm) {
                            .kq2 => 2 * rm_kq_int,
                            .stdq1 => 1 * rm_stdq_int,
                            .stdq2 => 2 * rm_stdq_int,
                            .kq1 => 1 * rm_kq_int,
                            .iq => 1 * rmIqInt(@intCast(i)),
                        };
                        const spec = [_]u32{ wg_size_subgroup_int, rows, cols };
                        w.create(&device.pipeline_dequant_mul_mat_vec_q8_1_f32[wi][@intCast(row.ty)][i], "mul_mat_vec_" ++ row.frag ++ "_q8_1_f32", .{
                            .spv = @field(S, "arr_dmmv_" ++ row.frag ++ "_q8_1_f32_data")[@backingInt(reduc)],
                            .parameter_count = mul_mat_vec_num_bindings,
                            .push_constant_size = pcv,
                            .wg_denoms = .{ rows, 1, 1 },
                            .spec = &spec,
                            .disable_robustness = true,
                            .require_full_subgroups = use_subgroups,
                            .required_subgroup_size = subgroup_size_int,
                        });
                    }
                }
            }
        }

        inline for (dmmv_rows) |row| {
            const rows = rowsOf(row.rm, rm_stdq, rm_kq, rm_iq);
            const wgs = if (row.sixteen) wg_size_subgroup16 else wg_size_subgroup;
            const spec = [_]u32{ wgs, rows };
            w.create(&device.pipeline_dequant_mul_mat_vec_id_f32[wi][@intCast(row.ty)], "mul_mat_vec_id_" ++ row.frag ++ "_f32", .{
                .spv = dmmvData("arr_dmmv_id_" ++ row.frag ++ "_f32_f32", row.ocp, device, if (row.sixteen) reduc16 else reduc),
                .parameter_count = mul_mat_vec_id_num_bindings,
                .push_constant_size = pcvid,
                .wg_denoms = .{ rows, 1, 1 },
                .spec = &spec,
                .disable_robustness = row.robust_off,
                .require_full_subgroups = if (row.sixteen) use_subgroups16 else use_subgroups,
                .required_subgroup_size = if (row.sixteen) force_subgroup_size16 else force_subgroup_size,
            });
        }

        if (comptime glslc.integer_dot) {
            if (device.integer_dot_product) {
                const subgroup_size_int = if (device.vendor_id == common.vendor_id.intel and device.subgroup_size_control) device.subgroup_min_size else device.subgroup_size;
                const wg_size_subgroup_int = if (is_sub) subgroup_size_int else subgroup_size_int * 4;
                inline for (dmmv_int_rows) |row| {
                    const rows: u32 = switch (row.rm) {
                        .kq2 => 2 * rm_kq_int,
                        .stdq1 => 1 * rm_stdq_int,
                        .stdq2 => 2 * rm_stdq_int,
                        .kq1 => 1 * rm_kq_int,
                        .iq => 1 * rmIqInt(0),
                    };
                    const spec = [_]u32{ wg_size_subgroup_int, rows };
                    w.create(&device.pipeline_dequant_mul_mat_vec_id_q8_1_f32[wi][@intCast(row.ty)], "mul_mat_vec_id_" ++ row.frag ++ "_q8_1_f32", .{
                        .spv = @field(S, "arr_dmmv_id_" ++ row.frag ++ "_q8_1_f32_data")[@backingInt(reduc)],
                        .parameter_count = mul_mat_vec_id_num_bindings,
                        .push_constant_size = pcvid,
                        .wg_denoms = .{ rows, 1, 1 },
                        .spec = &spec,
                        .disable_robustness = true,
                        .require_full_subgroups = use_subgroups,
                        .required_subgroup_size = subgroup_size_int,
                    });
                }
            }
        }
    }
}

/// A plain `ggml_vk_create_pipeline` call with the macros' defaults.
fn cp(w: *Walk, base: *?*Pipeline, name: []const u8, spv: []const u8, param_count: u32, push_size: usize, wg_denoms: [3]u32, spec: []const u32, al: u32, disable_robustness: bool) void {
    w.create(base, name, .{ .spv = spv, .parameter_count = param_count, .push_constant_size = @intCast(push_size), .wg_denoms = wg_denoms, .spec = spec, .@"align" = al, .disable_robustness = disable_robustness });
}

/// The type rows of `dequant`, `get_rows` and their kin: type, name
/// fragment, and the dequant workgroup denominator.
const dequant_rows = [_]struct { ty: c.ggml_type, frag: []const u8, denom: u32 }{
    .{ .ty = c.GGML_TYPE_Q1_0, .frag = "q1_0", .denom = 256 * 8 },
    .{ .ty = c.GGML_TYPE_Q2_0, .frag = "q2_0", .denom = 256 * 4 },
    .{ .ty = c.GGML_TYPE_Q4_0, .frag = "q4_0", .denom = 256 * 16 },
    .{ .ty = c.GGML_TYPE_Q4_1, .frag = "q4_1", .denom = 256 * 16 },
    .{ .ty = c.GGML_TYPE_Q5_0, .frag = "q5_0", .denom = 256 * 16 },
    .{ .ty = c.GGML_TYPE_Q5_1, .frag = "q5_1", .denom = 256 * 16 },
    .{ .ty = c.GGML_TYPE_Q8_0, .frag = "q8_0", .denom = 256 * 16 },
    .{ .ty = c.GGML_TYPE_Q2_K, .frag = "q2_k", .denom = 256 * 64 },
    .{ .ty = c.GGML_TYPE_TQ2_0, .frag = "tq2_0", .denom = 256 * 64 },
    .{ .ty = c.GGML_TYPE_Q3_K, .frag = "q3_k", .denom = 256 * 64 },
    .{ .ty = c.GGML_TYPE_Q4_K, .frag = "q4_k", .denom = 256 * 32 },
    .{ .ty = c.GGML_TYPE_Q5_K, .frag = "q5_k", .denom = 256 * 64 },
    .{ .ty = c.GGML_TYPE_Q6_K, .frag = "q6_k", .denom = 256 * 64 },
    .{ .ty = c.GGML_TYPE_IQ1_S, .frag = "iq1_s", .denom = 256 * 32 },
    .{ .ty = c.GGML_TYPE_IQ1_M, .frag = "iq1_m", .denom = 256 * 32 },
    .{ .ty = c.GGML_TYPE_IQ2_XXS, .frag = "iq2_xxs", .denom = 256 * 32 },
    .{ .ty = c.GGML_TYPE_IQ2_XS, .frag = "iq2_xs", .denom = 256 * 32 },
    .{ .ty = c.GGML_TYPE_IQ2_S, .frag = "iq2_s", .denom = 256 * 32 },
    .{ .ty = c.GGML_TYPE_IQ3_XXS, .frag = "iq3_xxs", .denom = 256 * 32 },
    .{ .ty = c.GGML_TYPE_IQ3_S, .frag = "iq3_s", .denom = 256 * 32 },
    .{ .ty = c.GGML_TYPE_IQ4_XS, .frag = "iq4_xs", .denom = 256 * 32 },
    .{ .ty = c.GGML_TYPE_IQ4_NL, .frag = "iq4_nl", .denom = 256 * 16 },
    .{ .ty = c.GGML_TYPE_MXFP4, .frag = "mxfp4", .denom = 256 * 16 },
    .{ .ty = c.GGML_TYPE_NVFP4, .frag = "nvfp4", .denom = 256 * 16 },
};

/// The rows `SET_ROWS` creates, in its order.
const set_rows_rows = [_]struct { ty: c.ggml_type, frag: []const u8 }{
    .{ .ty = c.GGML_TYPE_F32, .frag = "f32" },
    .{ .ty = c.GGML_TYPE_F16, .frag = "f16" },
    .{ .ty = c.GGML_TYPE_BF16, .frag = "bf16" },
    .{ .ty = c.GGML_TYPE_Q1_0, .frag = "q1_0" },
    .{ .ty = c.GGML_TYPE_Q2_0, .frag = "q2_0" },
    .{ .ty = c.GGML_TYPE_Q4_0, .frag = "q4_0" },
    .{ .ty = c.GGML_TYPE_Q4_1, .frag = "q4_1" },
    .{ .ty = c.GGML_TYPE_Q5_0, .frag = "q5_0" },
    .{ .ty = c.GGML_TYPE_Q5_1, .frag = "q5_1" },
    .{ .ty = c.GGML_TYPE_Q8_0, .frag = "q8_0" },
    .{ .ty = c.GGML_TYPE_IQ4_NL, .frag = "iq4_nl" },
};

/// `SET_ROWS(src_idx, src, itype)`.
fn setRows(w: *Walk, comptime src_idx: usize, comptime src: []const u8, comptime itype: []const u8) void {
    const table = &@field(w.device, "pipeline_set_rows" ++ itype)[src_idx];
    inline for (set_rows_rows) |row| {
        const name = "set_rows_" ++ src ++ "_" ++ row.frag ++ itype;
        cp(w, &table[@intCast(row.ty)], name, @field(S, name ++ "_data"), 3, @sizeOf(push.Binary), .{ 1, 1, 1 }, &.{1}, 1, true);
    }
}

/// Everything after mul_mat_vec: dequant, get_rows, the reductions, the
/// copies, then one or a few pipelines per remaining op.
fn walkRest(w: *Walk, subgroup_min_size_16: bool, subgroup_size_16: u32, subgroup_size_32: u32) void {
    const device = w.device;
    const pcb = @sizeOf(push.Binary);
    const pcu = @sizeOf(push.Unary);
    const pco = @sizeOf(push.Op);
    const mul_mat_vec_num_bindings = 5;

    // dequant shaders
    cp(w, &device.pipeline_dequant[c.GGML_TYPE_F32], "f32_to_f16", S.dequant_f32_data, 2, 5 * @sizeOf(u32), .{ 256 * 16, 1, 1 }, &.{}, 1, false);
    inline for (dequant_rows) |row| {
        cp(w, &device.pipeline_dequant[@intCast(row.ty)], "dequant_" ++ row.frag, @field(S, "dequant_" ++ row.frag ++ "_data"), 2, 5 * @sizeOf(u32), .{ row.denom, 1, 1 }, &.{}, 1, false);
        if (row.ty == c.GGML_TYPE_Q8_0) {
            cp(w, &device.pipeline_dequant_transpose[c.GGML_TYPE_Q8_0], "dequant_q8_0_transpose", S.dequant_q8_0_transpose_data, 2, 5 * @sizeOf(u32), .{ 256 * 16, 1, 1 }, &.{}, 1, false);
        }
    }

    // get_rows
    inline for (.{ .{ "", &device.pipeline_get_rows }, .{ "_f32", &device.pipeline_get_rows_f32 } }) |v| {
        const sfx = v[0];
        const table = v[1];
        inline for (.{ .{ c.GGML_TYPE_F32, "f32" }, .{ c.GGML_TYPE_F16, "f16" }, .{ c.GGML_TYPE_BF16, "bf16" } }) |f| {
            const name = "get_rows_" ++ f[1] ++ sfx;
            cp(w, &table[f[0]], name, @field(S, name ++ "_data"), 3, pcb, .{ 512, 1, 1 }, &.{}, 1, false);
        }
        inline for (dequant_rows) |row| {
            const name = "get_rows_" ++ row.frag ++ sfx;
            cp(w, &table[@intCast(row.ty)], name, @field(S, name ++ "_data"), 3, pcb, .{ 1024, 1, 1 }, &.{}, 1, false);
        }
        if (sfx.len == 0) {
            cp(w, &table[c.GGML_TYPE_I32], "get_rows_i32", S.get_rows_i32_data, 3, pcb, .{ 1024, 1, 1 }, &.{}, 1, false);
        }
    }
    cp(w, &device.pipeline_get_rows_back_f32, "get_rows_back_f32", S.get_rows_back_f32_data, 3, pcb, .{ 256, 1, 1 }, &.{}, 1, true);

    cp(w, &device.pipeline_matmul_split_k_reduce, "split_k_reduce", S.split_k_reduce_data, 2, 2 * @sizeOf(u32), .{ 256 * 4, 1, 1 }, &.{}, 1, false);
    cp(w, &device.pipeline_flash_attn_split_k_reduce, "fa_split_k_reduce", S.fa_split_k_reduce_data, 3, @sizeOf(push.FlashAttnSplitKReduce), .{ 1, device.subgroup_size, 1 }, &.{device.subgroup_size}, 1, true);

    var mit = device.pipeline_fa_mask_opt.iterator();
    while (mit.next()) |it| {
        const br_bc = it.key_ptr.*;
        const spec = [_]u32{ 128, 128 / device.subgroup_size, br_bc[0], br_bc[1] };
        w.create(it.value_ptr, "fa_mask_opt", .{ .spv = S.fa_mask_opt_data, .parameter_count = 2, .push_constant_size = @sizeOf(push.FlashAttnMaskOpt), .wg_denoms = .{ 1, 1, 1 }, .spec = &spec, .disable_robustness = true, .require_full_subgroups = true, .required_subgroup_size = device.subgroup_size });
    }

    if (device.subgroup_clustered and device.subgroup_require_full_support) {
        w.create(&device.pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", .{ .spv = S.quantize_q8_1_x4_subgroup_data, .parameter_count = 2, .push_constant_size = @sizeOf(push.QuantizeQ81), .wg_denoms = .{ 32 * device.subgroup_size / 8, 1, 1 }, .spec = &.{device.subgroup_size}, .disable_robustness = true, .require_full_subgroups = true });
    } else {
        cp(w, &device.pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", S.quantize_q8_1_x4_data, 2, @sizeOf(push.QuantizeQ81), .{ 32 * device.subgroup_size / 8, 1, 1 }, &.{device.subgroup_size}, 1, false);
    }

    for (0..m.device.p021_max_gqa_ratio) |i| {
        const spec = [_]u32{ device.subgroup_size, @intCast(i + 1) };
        if (device.subgroup_arithmetic and device.subgroup_require_full_support) {
            var buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&buf, "mul_mat_vec_p021_f16_f32{d}", .{i + 1}) catch unreachable;
            w.create(&device.pipeline_mul_mat_vec_p021_f16_f32[i], name, .{ .spv = S.mul_mat_vec_p021_f16_f32_subgroup_add_data, .parameter_count = mul_mat_vec_num_bindings, .push_constant_size = @sizeOf(push.MatVecP021), .wg_denoms = .{ 1, 1, 1 }, .spec = &spec, .disable_robustness = true, .require_full_subgroups = true });
        } else {
            w.createFmt(&device.pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32{d}", .{i + 1}, .{ .spv = S.mul_mat_vec_p021_f16_f32_data, .parameter_count = mul_mat_vec_num_bindings, .push_constant_size = @sizeOf(push.MatVecP021), .wg_denoms = .{ 1, 1, 1 }, .spec = &spec, .disable_robustness = true });
        }
    }
    cp(w, &device.pipeline_mul_mat_vec_nc_f16_f32, "mul_mat_vec_nc_f16_f32", S.mul_mat_vec_nc_f16_f32_data, mul_mat_vec_num_bindings, @sizeOf(push.MatVecNc), .{ 1, 1, 1 }, &.{}, 1, false);

    cp(w, &device.pipeline_norm_f32, "norm_f32", S.norm_f32_data, 2, pcu, .{ 1, 1, 1 }, &.{}, 1, false);
    cp(w, &device.pipeline_group_norm_f32, "group_norm_f32", S.group_norm_f32_data, 2, pco, .{ 1, 1, 1 }, &.{}, 1, false);

    cp(w, &device.pipeline_rms_norm_f32, "rms_norm_f32", S.rms_norm_f32_data, 4, pcb, .{ 1, 1, 1 }, &.{ 0, 0 }, 1, true);
    cp(w, &device.pipeline_rms_norm_mul_f32, "rms_norm_mul_f32", S.rms_norm_f32_data, 4, pcb, .{ 1, 1, 1 }, &.{ 0, 1 }, 1, true);
    cp(w, &device.pipeline_rms_norm_partials_f32, "rms_norm_partials_f32", S.rms_norm_partials_f32_data, 4, pcb, .{ 1, 1, 1 }, &.{ 0, 0 }, 1, true);
    cp(w, &device.pipeline_rms_norm_mul_partials_f32, "rms_norm_mul_partials_f32", S.rms_norm_partials_f32_data, 4, pcb, .{ 1, 1, 1 }, &.{ 0, 1 }, 1, true);

    if (@sizeOf(push.RmsNormMulRope) <= device.properties.limits.maxPushConstantsSize) {
        cp(w, &device.pipeline_rms_norm_mul_rope_f32_f32, "rms_norm_mul_rope_f32_f32", S.rms_norm_mul_rope_f32_f32_data, 7, @sizeOf(push.RmsNormMulRope), .{ 1, 1, 1 }, &.{ 0, 1 }, 1, true);
        cp(w, &device.pipeline_rms_norm_mul_rope_f32_f16, "rms_norm_mul_rope_f32_f16", S.rms_norm_mul_rope_f32_f16_data, 7, @sizeOf(push.RmsNormMulRope), .{ 1, 1, 1 }, &.{ 0, 1 }, 1, true);
    }

    cp(w, &device.pipeline_rms_norm_back_f32, "rms_norm_back_f32", S.rms_norm_back_f32_data, 3, pco, .{ 1, 1, 1 }, &.{}, 1, false);
    cp(w, &device.pipeline_l2_norm_f32, "l2_norm_f32", S.l2_norm_f32_data, 2, pcu, .{ 1, 1, 1 }, &.{}, 1, false);

    inline for (.{ "", "contig_" }) |pre| {
        inline for (.{ "f32_f32", "f32_f16", "f16_f16", "f16_f32", "f32_bf16", "bf16_f32", "i32_f32", "f32_i32" }) |pair| {
            const name = pre ++ "cpy_" ++ pair;
            cp(w, &@field(device, "pipeline_" ++ pre ++ "cpy_" ++ pair), name, @field(S, name ++ "_data"), 2, pcu, .{ 512, 1, 1 }, &.{}, 1, false);
        }
    }

    inline for (.{ "32", "16", "02_32", "02_16" }) |sfx| {
        const name = "cpy_transpose_" ++ sfx;
        cp(w, &@field(device, "pipeline_" ++ name), name, @field(S, name ++ "_data"), 2, pcu, .{ 1, 1, 1 }, &.{}, 1, false);
    }

    inline for (.{ .{ c.GGML_TYPE_Q1_0, "q1_0" }, .{ c.GGML_TYPE_Q2_0, "q2_0" }, .{ c.GGML_TYPE_Q4_0, "q4_0" }, .{ c.GGML_TYPE_Q4_1, "q4_1" }, .{ c.GGML_TYPE_Q5_0, "q5_0" }, .{ c.GGML_TYPE_Q5_1, "q5_1" }, .{ c.GGML_TYPE_Q8_0, "q8_0" }, .{ c.GGML_TYPE_IQ4_NL, "iq4_nl" } }) |q| {
        const name = "cpy_f32_" ++ q[1];
        cp(w, &device.pipeline_cpy_f32_quant[q[0]], name, @field(S, name ++ "_data"), 2, pcu, .{ 32, 1, 1 }, &.{}, 1, false);
    }

    setRows(w, 0, "f32", "_i32");
    setRows(w, 0, "f32", "_i64");
    setRows(w, 1, "f16", "_i32");
    setRows(w, 1, "f16", "_i64");

    inline for (.{ .{ c.GGML_TYPE_Q1_0, "q1_0" }, .{ c.GGML_TYPE_Q2_0, "q2_0" }, .{ c.GGML_TYPE_Q4_0, "q4_0" }, .{ c.GGML_TYPE_Q4_1, "q4_1" }, .{ c.GGML_TYPE_Q5_0, "q5_0" }, .{ c.GGML_TYPE_Q5_1, "q5_1" }, .{ c.GGML_TYPE_Q8_0, "q8_0" }, .{ c.GGML_TYPE_IQ4_NL, "iq4_nl" } }) |q| {
        const name = "cpy_" ++ q[1] ++ "_f32";
        cp(w, &device.pipeline_cpy_quant_f32[q[0]], name, @field(S, name ++ "_data"), 2, pcu, .{ @intCast(c.ggml_blck_size(q[0])), 1, 1 }, &.{}, 1, false);
    }

    // CREATE_BINARY(name, namemod, spec, bindings)
    inline for (.{
        .{ "add", "", 0, 4 },     .{ "add", "_norepeat", 1, 4 },
        .{ "sub", "", 0, 3 },     .{ "sub", "_norepeat", 1, 3 },
        .{ "mul", "", 0, 3 },     .{ "mul", "_norepeat", 1, 3 },
        .{ "div", "", 0, 3 },     .{ "div", "_norepeat", 1, 3 },
        .{ "add_rms", "", 0, 4 }, .{ "add_rms", "_norepeat", 1, 4 },
    }) |b| {
        const table = &@field(device, "pipeline_" ++ b[0] ++ b[1]);
        const data = @field(S, b[0] ++ "_data");
        for (0..2) |s0| for (0..2) |s1| for (0..2) |dd| {
            const sfx = [2][]const u8{ "_f32", "_f16" };
            w.createFmt(&table[s0][s1][dd], "{s}{s}{s}{s}{s}", .{ b[0], sfx[s0], sfx[s1], sfx[dd], b[1] }, .{ .spv = data[s0][s1][dd], .parameter_count = b[3], .push_constant_size = pcb, .wg_denoms = .{ 512, 1, 1 }, .spec = &.{b[2]} });
        };
    }

    if (device.multi_add) {
        for (0..m.pipeline.max_fused_adds) |i| {
            const spec = [_]u32{@intCast(i + 2)};
            w.createFmt(&device.pipeline_multi_add[i], "multi_add_f32_{d}", .{i + 1}, .{ .spv = S.multi_add_f32_data, .parameter_count = m.pipeline.max_parameter_count, .push_constant_size = @sizeOf(push.MultiAdd), .wg_denoms = .{ 512, 1, 1 }, .spec = &spec });
            w.createFmt(&device.pipeline_multi_add_rms[i], "multi_add_rms_f32_{d}", .{i + 1}, .{ .spv = S.multi_add_rms_f32_data, .parameter_count = m.pipeline.max_parameter_count, .push_constant_size = @sizeOf(push.MultiAdd), .wg_denoms = .{ 512, 1, 1 }, .spec = &spec });
        }
    }

    cp(w, &device.pipeline_add_id_f32, "add_id_f32", S.add_id_f32_data, 4, @sizeOf(push.AddId), .{ 1, 1, 1 }, &.{}, 1, false);

    cp(w, &device.pipeline_acc_f32, "acc_f32", S.acc_f32_data, 3, pcb, .{ 512, 1, 1 }, &.{ 0, 1 }, 1, false);
    cp(w, &device.pipeline_set_f32, "set_f32", S.acc_f32_data, 3, pcb, .{ 512, 1, 1 }, &.{ 0, 0 }, 1, false);

    inline for (.{ "i8", "i16", "i32", "i64" }) |ty| {
        cp(w, &@field(device, "pipeline_concat_" ++ ty), "concat_" ++ ty, @field(S, "concat_" ++ ty ++ "_data"), 3, pcb, .{ 512, 1, 1 }, &.{}, 1, false);
    }

    const pcup = @sizeOf(push.Upscale);
    cp(w, &device.pipeline_upscale_nearest_f32, "upscale_f32", S.upscale_f32_data, 2, pcup, .{ 512, 1, 1 }, &.{c.GGML_SCALE_MODE_NEAREST}, 1, false);
    cp(w, &device.pipeline_upscale_bilinear_f32, "upscale_f32", S.upscale_f32_data, 2, pcup, .{ 512, 1, 1 }, &.{c.GGML_SCALE_MODE_BILINEAR}, 1, false);
    cp(w, &device.pipeline_upscale_bicubic_f32, "upscale_f32", S.upscale_f32_data, 2, pcup, .{ 512, 1, 1 }, &.{c.GGML_SCALE_MODE_BICUBIC}, 1, false);
    cp(w, &device.pipeline_upscale_bilinear_antialias_f32, "upscale_f32", S.upscale_f32_data, 2, pcup, .{ 512, 1, 1 }, &.{c.GGML_SCALE_MODE_BILINEAR | c.GGML_SCALE_FLAG_ANTIALIAS}, 1, false);

    cp(w, &device.pipeline_scale_f32, "scale_f32", S.scale_f32_data, 2, pcu, .{ 512, 1, 1 }, &.{}, 1, false);

    inline for (.{ "log", "tri", "diag" }) |op| {
        cp(w, &@field(device, "pipeline_" ++ op)[0], op ++ "_f32", @field(S, op ++ "_f32_data"), 2, pcu, .{ 512, 1, 1 }, &.{}, 1, false);
        cp(w, &@field(device, "pipeline_" ++ op)[1], op ++ "_f16", @field(S, op ++ "_f16_data"), 2, pcu, .{ 512, 1, 1 }, &.{}, 1, false);
    }

    cp(w, &device.pipeline_pad_f32, "pad_f32", S.pad_f32_data, 2, @sizeOf(push.Pad), .{ 512, 1, 1 }, &.{}, 1, false);
    cp(w, &device.pipeline_pad_reflect_1d_f32, "pad_reflect_1d_f32", S.pad_reflect_1d_f32_data, 2, pcu, .{ 512, 1, 1 }, &.{}, 1, false);

    cp(w, &device.pipeline_roll_f32, "roll_f32", S.roll_f32_data, 2, pcu, .{ 512, 1, 1 }, &.{}, 1, false);

    cp(w, &device.pipeline_repeat_i32, "repeat_i32", S.repeat_i32_data, 2, pcu, .{ 512, 1, 1 }, &.{}, 1, false);
    cp(w, &device.pipeline_repeat_back_f32, "repeat_back_f32", S.repeat_back_f32_data, 2, pcu, .{ 512, 1, 1 }, &.{}, 1, false);

    cp(w, &device.pipeline_repeat_i16, "repeat_i16", S.repeat_i16_data, 2, pcu, .{ 512, 1, 1 }, &.{}, 1, false);

    // CREATE_UNARY(name)
    inline for (.{ "elu", "gelu", "gelu_erf", "gelu_quick", "silu", "relu", "sqr", "sqrt", "sin", "cos", "clamp", "leaky_relu", "xielu", "neg", "tanh", "sigmoid", "hardsigmoid", "hardswish", "abs", "softplus", "step", "round", "ceil", "floor", "trunc", "sgn", "exp", "expm1" }) |op| {
        cp(w, &@field(device, "pipeline_" ++ op)[0], op ++ "_f32", @field(S, op ++ "_f32_data"), 2, pcu, .{ 512, 1, 1 }, &.{}, 1, false);
        cp(w, &@field(device, "pipeline_" ++ op)[1], op ++ "_f16", @field(S, op ++ "_f16_data"), 2, pcu, .{ 512, 1, 1 }, &.{}, 1, false);
    }

    inline for (.{ "add1_f16_f16", "add1_f16_f32", "add1_f32_f32" }) |name| {
        cp(w, &@field(device, "pipeline_" ++ name), name, @field(S, name ++ "_data"), 3, pcb, .{ 512, 1, 1 }, &.{}, 1, false);
    }

    cp(w, &device.pipeline_arange_f32, "arange_f32", S.arange_f32_data, 1, pco, .{ 512, 1, 1 }, &.{}, 1, false);

    cp(w, &device.pipeline_fill_f32, "fill_f32", S.fill_f32_data, 1, pco, .{ 512, 1, 1 }, &.{}, 1, false);
    cp(w, &device.pipeline_fill_f16, "fill_f16", S.fill_f16_data, 1, pco, .{ 512, 1, 1 }, &.{}, 1, false);

    // CREATE_GLU(name)
    inline for (.{ "geglu", "reglu", "swiglu", "swiglu_oai", "geglu_erf", "geglu_quick" }) |op| {
        cp(w, &@field(device, "pipeline_" ++ op)[0], op ++ "_f32", @field(S, op ++ "_f32_data"), 3, @sizeOf(push.Glu), .{ 512, 1, 1 }, &.{}, 1, true);
        cp(w, &@field(device, "pipeline_" ++ op)[1], op ++ "_f16", @field(S, op ++ "_f16_data"), 3, @sizeOf(push.Glu), .{ 512, 1, 1 }, &.{}, 1, true);
    }

    cp(w, &device.pipeline_silu_back_f32, "silu_back_f32", S.silu_back_f32_data, 3, pco, .{ 512, 1, 1 }, &.{}, 1, false);

    cp(w, &device.pipeline_diag_mask_inf_f32, "diag_mask_inf_f32", S.diag_mask_inf_f32_data, 2, @sizeOf(push.DiagMask), .{ 1, 512, 1 }, &.{}, 1, true);

    const pcsm = @sizeOf(push.SoftMax);
    const sg = device.subgroup_size;
    cp(w, &device.pipeline_soft_max_f32, "soft_max_f32", S.soft_max_f32_data, 4, pcsm, .{ 1, 1, 1 }, &.{sg}, 1, false);
    cp(w, &device.pipeline_soft_max_f32_wg512, "soft_max_f32_wg512", S.soft_max_f32_data, 4, pcsm, .{ 1, 1, 1 }, &.{512}, 1, false);
    cp(w, &device.pipeline_soft_max_f32_f16, "soft_max_f32_f16", S.soft_max_f32_f16_data, 4, pcsm, .{ 1, 1, 1 }, &.{sg}, 1, false);
    cp(w, &device.pipeline_soft_max_f32_f16_wg512, "soft_max_f32_f16_wg512", S.soft_max_f32_f16_data, 4, pcsm, .{ 1, 1, 1 }, &.{512}, 1, false);
    cp(w, &device.pipeline_soft_max_back_f32, "soft_max_back_f32", S.soft_max_back_f32_data, 3, pco, .{ 1, 1, 1 }, &.{sg}, 1, true);

    inline for (.{ "", "_f16" }) |sfx| {
        inline for (.{ "1", "2", "3" }) |n| {
            const name = "soft_max_large" ++ n ++ "_f32" ++ sfx;
            cp(w, &@field(device, "pipeline_" ++ name), name, @field(S, name ++ "_data"), 6, pcsm, .{ 1, 1, 1 }, &.{ 128, 4 }, 1, true);
        }
    }

    const pcr = @sizeOf(push.Rope);
    inline for (.{ "rope_norm_f32", "rope_neox_f32", "rope_multi_f32", "rope_vision_f32", "rope_norm_f16", "rope_neox_f16", "rope_multi_f16", "rope_vision_f16", "rope_norm_f32_f16", "rope_neox_f32_f16", "rope_multi_f32_f16" }) |name| {
        cp(w, &@field(device, "pipeline_" ++ name), name, @field(S, name ++ "_data"), 5, pcr, .{ 1, 512, 1 }, &.{}, 1, false);
    }

    const pcas = @sizeOf(push.Argsort);
    const shmem = device.properties.limits.maxComputeSharedMemorySize;
    for (0..m.device.num_argsort_pipelines) |iu| {
        const i: u32 = @intCast(iu);
        var block_size: u32 = @as(u32, 1) << @intCast(@min(i, device.max_workgroup_size_log2));
        if (i <= device.max_workgroup_size_log2 and 2 * @sizeOf(c_int) * block_size <= shmem) {
            const ncols_padded_log2 = i;
            const spec = [_]u32{ block_size, ncols_padded_log2 };
            w.createFmt(&device.pipeline_argsort_f32[iu], "argsort_f32_{d}", .{i}, .{ .spv = S.argsort_f32_data, .parameter_count = 3, .push_constant_size = pcas, .wg_denoms = .{ block_size, 1, 1 }, .spec = &spec, .disable_robustness = true });
        }
        const wg_unroll_factor: u32 = if (block_size > 1) 2 else 1;
        block_size /= wg_unroll_factor;
        const spec = [_]u32{ block_size, wg_unroll_factor };
        w.createFmt(&device.pipeline_argsort_large_f32[iu], "argsort_large_f32_{d}", .{i}, .{ .spv = S.argsort_large_f32_data, .parameter_count = 3, .push_constant_size = pcas, .wg_denoms = .{ block_size * wg_unroll_factor, 1, 1 }, .spec = &spec, .disable_robustness = true });
    }

    for (0..m.device.num_topk_pipelines) |iu| {
        const i: u32 = @intCast(iu);
        const block_size: u32 = @as(u32, 1) << @intCast(i);
        const ncols_padded_log2 = i;
        if (i <= device.max_workgroup_size_log2) {
            const nary_shmem = 2 * @sizeOf(c_int) * block_size +
                @sizeOf(c_int) * sg +
                2 * @sizeOf(c_int) +
                2 * (block_size / sg) * @sizeOf(c_int);
            if (device.subgroup_arithmetic and device.subgroup_require_full_support and device.subgroup_shuffle and device.subgroup_ballot and nary_shmem <= shmem) {
                const spec = [_]u32{ block_size, sg, device.subgroup_size_log2 };
                w.createFmt(&device.pipeline_topk_f32[iu], "topk_f32_{d}", .{i}, .{ .spv = S.topk_nary_search_f32_data, .parameter_count = 2, .push_constant_size = @sizeOf(push.Topk), .wg_denoms = .{ block_size, 1, 1 }, .spec = &spec, .disable_robustness = true, .require_full_subgroups = true, .required_subgroup_size = sg });
            } else if (2 * @sizeOf(c_int) * block_size <= shmem) {
                const spec = [_]u32{ block_size, ncols_padded_log2 };
                w.createFmt(&device.pipeline_topk_f32[iu], "topk_f32_{d}", .{i}, .{ .spv = S.topk_argsort_f32_data, .parameter_count = 2, .push_constant_size = @sizeOf(push.Topk), .wg_denoms = .{ block_size, 1, 1 }, .spec = &spec, .disable_robustness = true });
            }
        }
    }

    cp(w, &device.pipeline_argmax_f32, "argmax_f32", S.argmax_f32_data, 2, pco, .{ 1, 1, 1 }, &.{sg}, 1, false);

    cp(w, &device.pipeline_sum_rows_f32, "sum_rows_f32", S.sum_rows_f32_data, 2, @sizeOf(push.SumRows), .{ 1, 1, 1 }, &.{sg}, 1, false);
    // Intel Windows driver in range [32.0.101.8509, 32.0.101.8860) will
    // crash when using fwht kernels so we gate that here
    const can_use_fwht = device.driver_id != vk.VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS or
        !intelWindowsDriverInRange(device.properties.driverVersion, 101, 8509, 101, 8860);
    const fwht_ns = [_]u32{ 64, 128, 256, 512 };
    if (can_use_fwht and device.subgroup_basic and device.subgroup_shuffle) {
        for (fwht_ns, 0..) |n, idx| {
            if (sg <= n) {
                w.create(&device.pipeline_fwht_f32[idx], "fwht_f32", .{ .spv = S.fwht_f32_data, .parameter_count = 2, .push_constant_size = @sizeOf(push.Fwht), .wg_denoms = .{ 1, 1, 1 }, .spec = &.{ sg, n }, .disable_robustness = true, .require_full_subgroups = true, .required_subgroup_size = sg });
            }
        }
    } else if (can_use_fwht) {
        for (fwht_ns, 0..) |n, idx| {
            const block_size = @min(sg, n);
            cp(w, &device.pipeline_fwht_f32[idx], "fwht_shmem_f32", S.fwht_shmem_f32_data, 2, @sizeOf(push.Fwht), .{ 1, 1, 1 }, &.{ block_size, n }, 1, false);
        }
    }

    const cumsum_elem_per_thread: u32 = if (device.vendor_id == common.vendor_id.amd or device.vendor_id == common.vendor_id.intel) 2 else 4;
    const pcsr = @sizeOf(push.SumRows);
    w.create(&device.pipeline_cumsum_f32, "cumsum_f32", .{ .spv = S.cumsum_f32_data, .parameter_count = 2, .push_constant_size = pcsr, .wg_denoms = .{ 1, 1, 1 }, .spec = &.{ 256, sg, cumsum_elem_per_thread }, .disable_robustness = true, .require_full_subgroups = true, .required_subgroup_size = sg });
    w.create(&device.pipeline_cumsum_small_f32, "cumsum_f32", .{ .spv = S.cumsum_f32_data, .parameter_count = 2, .push_constant_size = pcsr, .wg_denoms = .{ 1, 1, 1 }, .spec = &.{ 128, sg, 1 }, .disable_robustness = true, .require_full_subgroups = true, .required_subgroup_size = sg });
    w.create(&device.pipeline_cumsum_multipass1_f32, "cumsum_multipass1_f32", .{ .spv = S.cumsum_multipass1_f32_data, .parameter_count = 3, .push_constant_size = pcsr, .wg_denoms = .{ 256, 1, 1 }, .spec = &.{ 256, sg }, .disable_robustness = true, .require_full_subgroups = true, .required_subgroup_size = sg });
    w.create(&device.pipeline_cumsum_multipass2_f32, "cumsum_multipass2_f32", .{ .spv = S.cumsum_multipass2_f32_data, .parameter_count = 3, .push_constant_size = pcsr, .wg_denoms = .{ 256, 1, 1 }, .spec = &.{ 256, sg }, .disable_robustness = true, .require_full_subgroups = true, .required_subgroup_size = sg });

    cp(w, &device.pipeline_count_equal_i32, "count_equal_i32", S.count_equal_i32_data, 3, pco, .{ 512, 1, 1 }, &.{sg}, 1, false);

    cp(w, &device.pipeline_count_experts, "count_experts", S.count_experts_data, 2, @sizeOf(push.CountExperts), .{ 1, 1, 1 }, &.{}, 1, true);

    var sit = device.pipeline_solve_tri_f32.iterator();
    while (sit.next()) |s| {
        const state = s.key_ptr.*;
        // Max number of rows to load at a time, limited by shared memory
        const batch_N: u32 = @intCast(shmem / ((state.N + state.K) * @sizeOf(f32)));
        // Need at least K invocations, and prefer a minimum of 128
        const block_size = @max(128, @as(u32, 1) << @intFromFloat(@ceil(@log2(@as(f32, @floatFromInt(state.K))))));
        cp(w, s.value_ptr, "solve_tri_f32", S.solve_tri_f32_data, 3, pcb, .{ 1, 1, 1 }, &.{ 0, state.N, state.K, batch_N, block_size }, 1, true);
    }

    const bda = device.shader_int64 and device.buffer_device_address;
    const pcim = @sizeOf(push.Im2col);
    const pcim3 = @sizeOf(push.Im2col3d);
    cp(w, &device.pipeline_im2col_f32, "im2col_f32", if (bda) S.im2col_f32_bda_data else S.im2col_f32_data, 2, pcim, .{ 512, 1, 1 }, &.{sg}, 1, true);
    cp(w, &device.pipeline_im2col_3d_f32, "im2col_3d_f32", if (bda) S.im2col_3d_f32_bda_data else S.im2col_3d_f32_data, 2, pcim3, .{ 512, 1, 1 }, &.{512}, 1, true);
    cp(w, &device.pipeline_im2col_f32_f16, "im2col_f32_f16", if (bda) S.im2col_f32_f16_bda_data else S.im2col_f32_f16_data, 2, pcim, .{ 512, 1, 1 }, &.{sg}, 1, true);
    cp(w, &device.pipeline_im2col_3d_f32_f16, "im2col_3d_f32_f16", if (bda) S.im2col_3d_f32_f16_bda_data else S.im2col_3d_f32_f16_data, 2, pcim3, .{ 512, 1, 1 }, &.{512}, 1, true);

    cp(w, &device.pipeline_timestep_embedding_f32, "timestep_embedding_f32", S.timestep_embedding_f32_data, 2, @sizeOf(push.TimestepEmbedding), .{ 256, 1, 1 }, &.{}, 1, false);

    cp(w, &device.pipeline_conv_transpose_1d_f32, "conv_transpose_1d_f32", S.conv_transpose_1d_f32_data, 3, @sizeOf(push.ConvTranspose1d), .{ 1, 1, 1 }, &.{}, 1, false);
    inline for (.{ "f32", "f16", "bf16" }) |ty| {
        cp(w, &@field(device, "pipeline_col2im_1d_" ++ ty), "col2im_1d_" ++ ty, @field(S, "col2im_1d_" ++ ty ++ "_data"), 2, @sizeOf(push.Col2im1d), .{ 256, 1, 1 }, &.{}, 1, true);
    }

    cp(w, &device.pipeline_out_prod_f32, "out_prod_f32", S.out_prod_f32_data, 3, pcb, .{ 256, 1, 1 }, &.{}, 1, false);

    inline for (.{ "f32", "f16", "bf16" }) |ty| {
        cp(w, &@field(device, "pipeline_snake_" ++ ty), "snake_" ++ ty, @field(S, "snake_" ++ ty ++ "_data"), 4, @sizeOf(push.Snake), .{ 256, 1, 1 }, &.{}, 1, false);
    }

    cp(w, &device.pipeline_pool1d_f32, "pool1d_f32", S.pool1d_f32_data, 2, @sizeOf(push.Pool1d), .{ 512, 1, 1 }, &.{}, 1, false);
    cp(w, &device.pipeline_pool2d_f32, "pool2d_f32", S.pool2d_f32_data, 2, @sizeOf(push.Pool2d), .{ 512, 1, 1 }, &.{}, 1, false);

    cp(w, &device.pipeline_rwkv_wkv6_f32, "rwkv_wkv6_f32", S.rwkv_wkv6_f32_data, 7, @sizeOf(push.RwkvWkv6), .{ 1, 1, 1 }, &.{sg}, 1, false);
    cp(w, &device.pipeline_rwkv_wkv7_f32, "rwkv_wkv7_f32", S.rwkv_wkv7_f32_data, 8, @sizeOf(push.RwkvWkv7), .{ 1, 1, 1 }, &.{sg}, 1, false);
    cp(w, &device.pipeline_gated_linear_attn_f32, "gated_linear_attn_f32", S.gated_linear_attn_f32_data, 6, @sizeOf(push.GatedLinearAttn), .{ 1, 1, 1 }, &.{}, 1, false);

    walkGatedDeltaNet(w);

    const pcss = @sizeOf(push.SsmScan);
    if (device.subgroup_arithmetic and device.subgroup_require_full_support) {
        w.create(&device.pipeline_ssm_scan_f32_d128, "ssm_scan_128_f32", .{ .spv = S.ssm_scan_subgroup_f32_data, .parameter_count = 8, .push_constant_size = pcss, .wg_denoms = .{ 1, 1, 1 }, .spec = &.{ 128, sg }, .disable_robustness = true, .require_full_subgroups = true });
        w.create(&device.pipeline_ssm_scan_f32_d256, "ssm_scan_256_f32", .{ .spv = S.ssm_scan_subgroup_f32_data, .parameter_count = 8, .push_constant_size = pcss, .wg_denoms = .{ 1, 1, 1 }, .spec = &.{ 256, sg }, .disable_robustness = true, .require_full_subgroups = true });
    } else {
        w.create(&device.pipeline_ssm_scan_f32_d128, "ssm_scan_128_f32", .{ .spv = S.ssm_scan_f32_data, .parameter_count = 8, .push_constant_size = pcss, .wg_denoms = .{ 1, 1, 1 }, .spec = &.{ 128, sg, 16 }, .disable_robustness = true, .require_full_subgroups = true });
        w.create(&device.pipeline_ssm_scan_f32_d256, "ssm_scan_256_f32", .{ .spv = S.ssm_scan_f32_data, .parameter_count = 8, .push_constant_size = pcss, .wg_denoms = .{ 1, 1, 1 }, .spec = &.{ 256, sg, 16 }, .disable_robustness = true, .require_full_subgroups = true });
    }

    const pcsc = @sizeOf(push.SsmConv);
    cp(w, &device.pipeline_ssm_conv_f32, "ssm_conv_f32", S.ssm_conv_f32_data, 4, pcsc, .{ 32, 16, 1 }, &.{ 32, 16, 0, 0 }, 1, false);
    cp(w, &device.pipeline_ssm_conv_silu_f32, "ssm_conv_silu_f32", S.ssm_conv_f32_data, 4, pcsc, .{ 32, 16, 1 }, &.{ 32, 16, 0, 1 }, 1, false);
    cp(w, &device.pipeline_ssm_conv_bias_silu_f32, "ssm_conv_bias_silu_f32", S.ssm_conv_f32_data, 4, pcsc, .{ 32, 16, 1 }, &.{ 32, 16, 1, 1 }, 1, false);

    cp(w, &device.pipeline_opt_step_adamw_f32, "opt_step_adamw_f32", S.opt_step_adamw_f32_data, 5, pco, .{ 512, 1, 1 }, &.{}, 1, false);
    cp(w, &device.pipeline_opt_step_sgd_f32, "opt_step_sgd_f32", S.opt_step_sgd_f32_data, 3, pco, .{ 512, 1, 1 }, &.{}, 1, false);

    walkConv(w, subgroup_min_size_16, subgroup_size_16, subgroup_size_32);
}

/// The gated delta net pipelines: per head size, the largest lane count
/// that divides both the size and the subgroup, and the reduction the
/// device can do across those lanes.
fn walkGatedDeltaNet(w: *Walk) void {
    const device = w.device;
    const sg = device.subgroup_size;
    const gdn_sizes = [_]u32{ 16, 32, 64, 128 };
    const gdn_names = [_][2][]const u8{
        .{ "gated_delta_net_f32_d16", "gated_delta_net_f32_d16_kda" },
        .{ "gated_delta_net_f32_d32", "gated_delta_net_f32_d32_kda" },
        .{ "gated_delta_net_f32_d64", "gated_delta_net_f32_d64_kda" },
        .{ "gated_delta_net_f32_d128", "gated_delta_net_f32_d128_kda" },
    };
    for (gdn_sizes, 0..) |S_V, si| {
        impl.assert(S_V > 1 and S_V & (S_V - 1) == 0, "is_pow2(S_V)");

        var lanes_per_column: u32 = if (S_V >= 128 and device.subgroup_clustered) 8 else @min(S_V, sg);

        // gated_delta_net.comp relies on S_V % COLS_PER_WG == 0 and
        // S_V % LANES_PER_COLUMN == 0 to avoid bounds checks.
        while (lanes_per_column > 1) {
            const valid_lanes = sg % lanes_per_column == 0 and S_V % lanes_per_column == 0;
            const cols_per_wg: u32 = if (valid_lanes) sg / lanes_per_column else 0;
            if (valid_lanes and cols_per_wg > 0 and S_V % cols_per_wg == 0) break;
            lanes_per_column >>= 1;
        }

        impl.assert(sg % lanes_per_column == 0, "(device->subgroup_size % lanes_per_column) == 0");
        impl.assert(S_V % lanes_per_column == 0, "(S_V % lanes_per_column) == 0");
        impl.assert(S_V % (sg / lanes_per_column) == 0, "(S_V % (device->subgroup_size / lanes_per_column)) == 0");

        const need_partial_subgroup_reduce = lanes_per_column != 1 and lanes_per_column < sg;
        const use_clustered_reduce = device.subgroup_arithmetic and device.subgroup_clustered and need_partial_subgroup_reduce;
        const use_subgroup_reduce = device.subgroup_arithmetic and !need_partial_subgroup_reduce;
        const use_subgroup_ops = use_clustered_reduce or use_subgroup_reduce;
        const gdn_data: []const u8 = if (use_clustered_reduce) S.gated_delta_net_f32_data else if (use_subgroup_reduce) S.gated_delta_net_f32_nocluster_data else S.gated_delta_net_f32_shmem_data;

        const cols_per_wg = sg / lanes_per_column;
        for (0..2) |kda| {
            const spec = [_]u32{ S_V, @intCast(kda), sg, lanes_per_column };
            w.create(&device.pipeline_gated_delta_net[si][kda], gdn_names[si][kda], .{ .spv = gdn_data, .parameter_count = 7, .push_constant_size = @sizeOf(push.GatedDeltaNet), .wg_denoms = .{ 1, 1, cols_per_wg }, .spec = &spec, .disable_robustness = true, .require_full_subgroups = use_subgroup_ops, .required_subgroup_size = sg });
        }
    }
}

/// Ports `ggml_vk_intel_windows_driver_in_range` (ggml-vulkan.cpp:19000 @c1d0e7a00):
/// whether `lower <= driver_version < upper`, each bound `xxx.yyyy`;
/// always true off Windows.
pub fn intelWindowsDriverInRange(driver_version: u32, lower_major: u32, lower_minor: u32, upper_major: u32, upper_minor: u32) bool {
    if (comptime @import("builtin").os.tag != .windows) return true;
    // Intel Windows encodes xxx.yyyy as [31:14].[13:0].
    const major = driver_version >> 14;
    const minor = driver_version & 0x3fff;
    const ge_lower = major > lower_major or (major == lower_major and minor >= lower_minor);
    const lt_upper = major < upper_major or (major == upper_major and minor < upper_minor);
    return ge_lower and lt_upper;
}

/// The conv2d, conv_transpose_2d and conv3d pipelines per tile shape, then
/// depthwise conv and topk_moe.
fn walkConv(w: *Walk, subgroup_min_size_16: bool, subgroup_size_16: u32, subgroup_size_32: u32) void {
    _ = subgroup_min_size_16;
    _ = subgroup_size_16;
    _ = subgroup_size_32;
    const device = w.device;
    const sg = device.subgroup_size;
    const shmem = device.properties.limits.maxComputeSharedMemorySize;

    for (0..m.device.conv_shape_count) |s| {
        const shape: m.backend.ConvShape = @fromBackingInt(@intCast(s));
        // smaller WG for the small-tile fallback gives more concurrent WGs per SM
        var conv2d_WG_SIZE: u32 = if (shape == .@"64x32") 128 else 256;
        var use_collectives: u32 = 0; // Enables subgroup ops for preventing the re-calculation of indices.
        const conv2d_TS_K: u32 = if (shape == .@"64x32") 4 else 8;
        var conv2d_SHMEM_PAD: u32 = 4;
        var bs = m.backend.conv_block_sizes[s];
        var conv2d_UNROLL = true;

        if (glslc.coopmat2 and device.coopmat2) conv2d_SHMEM_PAD = 8; // 8 float16_t

        if (device.vendor_id == common.vendor_id.intel) {
            conv2d_SHMEM_PAD = 0;
            conv2d_UNROLL = false;
        } else if (device.vendor_id == common.vendor_id.amd) {
            conv2d_SHMEM_PAD = if (device.architecture == .amd_gcn) 1 else 4;
            if (shape == .@"128x128" and device.architecture != .amd_gcn) conv2d_UNROLL = false;
        }

        // Use collectives on pre-Turing NVIDIA GPUs and GCN AMD cards, which had slower integer math.
        const allow_collectives_nv = device.vendor_id != common.vendor_id.nvidia or device.architecture == .nvidia_pre_turing;
        const allow_collectives_amd = device.vendor_id != common.vendor_id.amd or device.architecture == .amd_gcn;
        if (device.subgroup_shuffle and device.vendor_id != common.vendor_id.intel and allow_collectives_nv and allow_collectives_amd) {
            use_collectives = 1;
            // CRS block size should be capped at subgroup size when shuffle is used.
            bs.CRS = @min(sg, bs.CRS);
        }

        // cm1 is used only when cm2 is unavailable; capped at 64x128.
        var conv2d_use_cm1 = glslc.coopmat and !device.coopmat2 and
            device.coopmat_support and device.coopmat_support_16x16x16_f16acc and
            device.subgroup_size_control and (sg == 32 or sg == 64) and shape != .@"128x128";

        const conv2d_cm1_shmem_pad: u32 = 8;
        const shmemReq = struct {
            fn f(b: m.backend.ConvBlockSize, pad: u32, csh_store: bool, fp16_shmem: bool) u32 {
                const elem_size: u32 = if (fp16_shmem) @sizeOf(u16) else @sizeOf(f32);
                const csh_elems: u32 = if (csh_store) b.K * b.NPQ else 0;
                return (b.K * (b.CRS + pad) + b.CRS * (b.NPQ + pad) + csh_elems) * elem_size;
            }
        }.f;

        // cm1 needs Csh for output, so check before applying cm1 params.
        if (conv2d_use_cm1 and shmem < shmemReq(bs, conv2d_cm1_shmem_pad, true, true)) conv2d_use_cm1 = false;

        var conv2d_WM: u32 = 16; // cm1 subgroup tile, ignored otherwise
        var conv2d_WN: u32 = 16;
        if (conv2d_use_cm1) {
            conv2d_SHMEM_PAD = conv2d_cm1_shmem_pad;
            // 16x16x16 fragments; pick WM/WN to keep WG_SIZE at 256
            const sg64 = sg == 64;
            switch (shape) {
                .@"64x32" => {
                    conv2d_WM = if (sg64) 32 else 16;
                    conv2d_WN = 16;
                },
                .@"64x128" => {
                    conv2d_WM = 32;
                    conv2d_WN = if (sg64) 64 else 32;
                },
                .@"32x256" => {
                    conv2d_WM = if (sg64) 16 else 32;
                    conv2d_WN = if (sg64) 128 else 32;
                },
                else => {},
            }
            const warps_M = bs.K / conv2d_WM;
            const warps_N = bs.NPQ / conv2d_WN;
            conv2d_WG_SIZE = warps_M * warps_N * sg;
        }

        // stage the cm2 accumulator through shmem, except on 128x128; cm1
        // always uses the staged path.
        var conv2d_csh_store: u32 = if (device.coopmat2 and shape != .@"128x128") 1 else 0;
        if (conv2d_use_cm1) conv2d_csh_store = 1;

        // shmem is fp16 on cm2/cm1 (matches Csh), fp32 on scalar
        const conv2d_use_fp16_shmem = device.coopmat2 or conv2d_use_cm1;

        // shrink CRS if the non-cm1 config still doesn't fit
        if (shmem < shmemReq(bs, conv2d_SHMEM_PAD, conv2d_csh_store != 0, conv2d_use_fp16_shmem)) {
            impl.assert(!conv2d_use_cm1, "!conv2d_use_cm1");
            bs.CRS = 8;
            if (use_collectives != 0) bs.CRS = @min(sg, bs.CRS);
            conv2d_csh_store = 0;
        }

        const wg_denoms = [3]u32{ bs.K, 1, 1 };
        // cm1 needs a fixed subgroup width to match the WG_SIZE we computed
        const conv2d_required_subgroup_size: u32 = if (conv2d_use_cm1) sg else 0;

        const variant: enum { cm2, cm1, unroll, plain } = if (glslc.coopmat2 and device.coopmat2) .cm2 else if (glslc.coopmat and conv2d_use_cm1) .cm1 else if (conv2d_UNROLL) .unroll else .plain;

        const Conv2dCfg = struct { wg: u32, k: u32, crs: u32, npq: u32, tsk: u32, coll: u32, pad: u32, csh: u32, wm: u32, wn: u32, req: u32, denoms: [3]u32 };
        const cfg: Conv2dCfg = .{ .wg = conv2d_WG_SIZE, .k = bs.K, .crs = bs.CRS, .npq = bs.NPQ, .tsk = conv2d_TS_K, .coll = use_collectives, .pad = conv2d_SHMEM_PAD, .csh = conv2d_csh_store, .wm = conv2d_WM, .wn = conv2d_WN, .req = conv2d_required_subgroup_size, .denoms = wg_denoms };

        // CREATE_CONVS(spv_suffix)
        const createConvs = struct {
            fn f(ww: *Walk, si: usize, k: Conv2dCfg, comptime spv_suffix: []const u8) void {
                const d_ = ww.device;
                inline for (.{ .{ "conv2d", "_f32" }, .{ "conv2d", "_f16_f32" }, .{ "conv_transpose_2d", "_f32" }, .{ "conv_transpose_2d", "_f16_f32" } }) |cv| {
                    var it = @field(d_, "pipeline_" ++ cv[0] ++ cv[1])[si].iterator();
                    while (it.next()) |e| {
                        const st = e.key_ptr.*;
                        const spec = [_]u32{ k.wg, k.k, k.crs, k.npq, k.tsk, k.coll, k.pad, st.s0, st.s1, st.p0, st.p1, st.d0, st.d1, st.KW, st.KH, st.aligned, k.csh, k.wm, k.wn };
                        ww.create(e.value_ptr, cv[0] ++ cv[1], .{ .spv = @field(S, cv[0] ++ cv[1] ++ spv_suffix ++ "_data"), .parameter_count = 3, .push_constant_size = @sizeOf(push.Conv2d), .wg_denoms = k.denoms, .spec = &spec, .disable_robustness = true, .require_full_subgroups = k.coll != 0 or k.req != 0, .required_subgroup_size = k.req });
                    }
                }
            }
        }.f;
        switch (variant) {
            .cm2 => if (comptime glslc.coopmat2) createConvs(w, s, cfg, "_cm2"),
            .cm1 => if (comptime glslc.coopmat) createConvs(w, s, cfg, "_cm1"),
            .unroll => createConvs(w, s, cfg, "_unroll"),
            .plain => createConvs(w, s, cfg, ""),
        }

        // CREATE_CONV3D(type_suffix, spv_suffix)
        const createConv3d = struct {
            fn f(ww: *Walk, si: usize, k: Conv2dCfg, comptime type_suffix: []const u8, comptime spv_suffix: []const u8) void {
                var it = @field(ww.device, "pipeline_conv3d" ++ type_suffix)[si].iterator();
                while (it.next()) |e| {
                    const st = e.key_ptr.*;
                    const spec = [_]u32{ k.wg, k.k, k.crs, k.npq, k.tsk, k.pad, st.s0, st.s1, st.s2, st.p0, st.p1, st.p2, st.d0, st.d1, st.d2, st.KW, st.KH, st.KD, st.aligned, k.csh, k.wm, k.wn };
                    ww.create(e.value_ptr, "conv3d" ++ type_suffix, .{ .spv = @field(S, "conv3d" ++ type_suffix ++ spv_suffix ++ "_data"), .parameter_count = 3, .push_constant_size = @sizeOf(push.Conv3d), .wg_denoms = k.denoms, .spec = &spec, .disable_robustness = true, .require_full_subgroups = k.req != 0, .required_subgroup_size = k.req });
                }
            }
        }.f;
        switch (variant) {
            .cm2 => if (comptime glslc.coopmat2) {
                createConv3d(w, s, cfg, "_f32", "_cm2");
                createConv3d(w, s, cfg, "_f16_f32", "_cm2");
            },
            .cm1 => if (comptime glslc.coopmat) {
                createConv3d(w, s, cfg, "_f32", "_cm1");
                createConv3d(w, s, cfg, "_f16_f32", "_cm1");
            },
            .unroll => {
                createConv3d(w, s, cfg, "_f32", "_unroll");
                createConv3d(w, s, cfg, "_f16_f32", "_unroll");
            },
            .plain => {
                createConv3d(w, s, cfg, "_f32", "");
                createConv3d(w, s, cfg, "_f16_f32", "");
            },
        }
    }

    const pcdw = @sizeOf(push.Conv2dDw);
    inline for (.{ "conv2d_dw_whcn_f32", "conv2d_dw_cwhn_f32", "conv2d_dw_whcn_f16_f32", "conv2d_dw_cwhn_f16_f32" }) |name| {
        cp(w, &@field(device, "pipeline_" ++ name), name, @field(S, name ++ "_data"), 3, pcdw, .{ 512, 1, 1 }, &.{}, 1, false);
    }

    for (0..2) |use_push| {
        for (0..m.device.num_topk_moe_pipelines) |i| {
            const spec = [_]u32{ sg, @as(u32, 1) << @intCast(i), @intCast(use_push) };
            w.createFmt(&device.pipeline_topk_moe[i][use_push], "topk_moe_f32_{d}", .{i}, .{ .spv = S.topk_moe_f32_data, .parameter_count = 4, .push_constant_size = @sizeOf(push.TopkMoe), .wg_denoms = .{ 1, 1, 1 }, .spec = &spec, .disable_robustness = true, .require_full_subgroups = true, .required_subgroup_size = sg });
        }
    }
}

/// The end of `ggml_vk_load_shaders`: drop the lock, compile what this
/// walk claimed, and wait for whoever claimed the pipeline we need.
fn finish(w: *Walk) void {
    const device = w.device;
    // Drop compile_mutex so other threads can walk while we compile.
    _ = std.c.pthread_mutex_unlock(&device.compile_mutex);

    if (w.claimed) |task| {
        m.pipeline.createPipelineFunc(device, task.pipeline, task.args);
        allocator.free(task.args.specialization_constants);
    }

    // Another thread may be compiling the pipeline we need; block on it here.
    if (w.wait_pipeline) |p| {
        _ = std.c.pthread_mutex_lock(&device.compile_mutex);
        while (!p.compiled.load(.seq_cst)) _ = std.c.pthread_cond_wait(&device.compile_cv, &device.compile_mutex);
        _ = std.c.pthread_mutex_unlock(&device.compile_mutex);
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "mulMmCm2Spec inserts ALIGNED at 5 for matmul_id and pads to 7" {
    const a = mulMmCm2Spec(Tile.of(.{ 256, 128, 128, 32, 1, 32 }), true, true);
    try std.testing.expectEqualSlices(u32, &.{ 256, 128, 128, 32, 1, 1, 32 }, a.slice());
    const b = mulMmCm2Spec(Tile.of(.{ 256, 128, 256, 64, 1 }), false, true);
    try std.testing.expectEqualSlices(u32, &.{ 256, 128, 256, 64, 1, 0, 32 }, b.slice());
    const cc = mulMmCm2Spec(Tile.of(.{ 256, 128, 256, 64, 1 }), true, false);
    try std.testing.expectEqualSlices(u32, &.{ 256, 128, 256, 64, 1, 1 }, cc.slice());
}
