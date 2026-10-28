//! Flash attention: the tuning that picks a code path (scalar, coopmat1,
//! coopmat2) and its tile shape, the shared-memory checks behind it, and
//! the dispatch -- grouped-query folding, split-KV with a reduction pass,
//! the mask-skip prepass, and K/V dequantized once into fp16 scratch.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.4.0
//! (`5266f24da`).

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const vk = m.common.vk;
const BackendCtx = m.context.BackendCtx;
const Context = m.submit.Context;
const Device = m.device.Device;
const Tensor = c.ggml_tensor;
const Subbuffer = m.buffer.Subbuffer;
const Pipeline = m.pipeline.Pipeline;
const FaCodePath = m.device.FaCodePath;
const FaPipelineState = m.device.FaPipelineState;
const push = m.push;
const ops = m.ops;
const ceilDiv = ops.ceilDiv;
const roundupPow2 = m.ops_matmul.roundupPow2;
const sub = m.ops_matmul.sub;

extern fn powf(x: f32, y: f32) f32;
extern fn log2f(x: f32) f32;
extern fn floorf(x: f32) f32;

const vendor_id_amd = 0x1002;
const vendor_id_intel = 0x8086;
const vendor_id_nvidia = 0x10de;

/// Ports `vk_fa_tuning_params` (ggml-vulkan.cpp:3764 @5266f24da).
pub const FaTuningParams = struct {
    path: FaCodePath = .scalar,
    workgroup_size: u32 = 0,
    subgroup_size: u32 = 0,
    block_rows: u32 = 0,
    block_cols: u32 = 0,
    d_split: u32 = 0,
    row_split: u32 = 0,
    shmem_staging: bool = false,
    disable_subgroups: bool = false,
    limit_occupancy_shmem: u32 = 0,
};

/// `D ^ (D & (D-1))`: the lowest set bit.
fn lowestBit(d: u32) u32 {
    return d ^ (d & (d -% 1));
}

/// Ports `get_fa_tuning_params_scalar` (ggml-vulkan.cpp:3787 @5266f24da).
fn tuningScalar(dev: *const Device, hsk: u32, hsv: u32, n_rows: u32, n_kv: u32, k_type: c.ggml_type, v_type: c.ggml_type, f32acc: bool) FaTuningParams {
    var r: FaTuningParams = .{ .path = .scalar };

    if (dev.vendor_id == vendor_id_intel) {
        // Disable subgroup use due to performance issues when enforcing subgroup sizes
        r.subgroup_size = 32;
        r.disable_subgroups = true;
    } else if (dev.vendor_id == vendor_id_amd and dev.architecture != .amd_gcn) {
        r.subgroup_size = if (n_rows < 4) 32 else dev.subgroup_size;
    } else {
        r.subgroup_size = dev.subgroup_size;
    }

    // Row split keeps synchronization within subgroups, which avoids barriers
    var row_split_max_hsk: u32 = 64;
    if (dev.vendor_id == vendor_id_amd and dev.architecture != .amd_gcn and !dev.uma) {
        row_split_max_hsk = if (n_rows <= 8) 64 else 128;
    }
    r.row_split = if (n_rows < 4 or hsk <= row_split_max_hsk) 1 else 4;

    if (r.subgroup_size > 32 and (n_rows < 4 or hsk < @as(u32, if (r.row_split == 1) 128 else 64))) {
        r.workgroup_size = r.subgroup_size * 2;
    } else {
        r.workgroup_size = r.subgroup_size * 4;
    }

    const D = hsk | hsv;
    const reduce_block_rows = (D & 8) != 0 or n_kv < 1024 or dev.vendor_id == vendor_id_intel;

    if (n_rows == 1) {
        r.block_rows = 1;
        r.block_cols = 64;
    } else {
        // row_split 1 means higher register use per row, so block size has to be adjusted
        if (r.row_split == 1) {
            r.block_rows = if (n_rows == 2) 2 else if (n_rows <= 4 or reduce_block_rows) 4 else 8;
        } else {
            r.block_rows = if (n_rows <= 4) 4 else if (n_rows <= 8 or reduce_block_rows) 8 else 16;
        }
        r.block_cols = if ((D & 8) != 0) 64 else 32;
    }

    r.d_split = @min(@min(r.subgroup_size, 8), lowestBit(D) / 4);
    r.shmem_staging = dev.vendor_id == vendor_id_nvidia and hsk < 256 and hsv < 256;

    if (!reduce_block_rows and !scalarShmemSupport(dev, &r, hsk, hsv, f32acc, k_type, v_type)) r.block_rows /= 2;

    // On AMD RDNA, small head sizes with a big batch schedule too many
    // subgroups and thrash the cache; an unused shmem buffer limits occupancy.
    if (dev.vendor_id == vendor_id_amd and dev.properties.limits.maxComputeSharedMemorySize == 65536) {
        if (dev.architecture != .amd_gcn and n_rows >= 64 and hsk <= 128) {
            // 30kb target for hsk > 64, 26kb for <= 64 (smaller workgroup); guessed, tested on RDNA2
            r.limit_occupancy_shmem = @as(u32, if (hsk <= 64) 26 else 30) * 1024 / 4 / 4;
        } else if (dev.architecture == .amd_gcn and n_rows <= 8 and hsk >= 256) {
            // GCN, 2 subgroups per SIMD: low-batch FA with a large head size.
            r.limit_occupancy_shmem = @as(u32, if (n_rows < 4) 14 else 26) * 1024 / 4 / 4;
        }
    }
    return r;
}

/// Ports `get_fa_tuning_params_coopmat1` (ggml-vulkan.cpp:3862 @5266f24da).
fn tuningCoopmat1(dev: *const Device, hsk: u32, hsv: u32) FaTuningParams {
    var r: FaTuningParams = .{ .path = .coopmat1 };
    const D = hsk | hsv;
    const coopmat_block_rows: u32 = 16;
    const coopmat_block_cols: u32 = 16;
    const num_subgroups: u32 = 4;

    r.block_rows = coopmat_block_rows;
    r.block_cols = coopmat_block_cols * num_subgroups;
    r.row_split = num_subgroups;
    r.subgroup_size = dev.subgroup_size;
    r.workgroup_size = num_subgroups * r.subgroup_size;
    r.d_split = @min(@min(r.subgroup_size, 8), lowestBit(D) / 4);
    r.shmem_staging = dev.vendor_id == vendor_id_nvidia and hsk < 256 and hsv < 256;
    return r;
}

/// Ports `get_fa_tuning_params_coopmat2` (ggml-vulkan.cpp:3893 @5266f24da).
fn tuningCoopmat2(dev: *const Device, hsk: u32, hsv: u32, n_rows: u32, k_type: c.ggml_type, v_type: c.ggml_type) FaTuningParams {
    var r: FaTuningParams = .{ .path = .coopmat2 };
    const D = hsk | hsv;
    const small_rows = n_rows < 32;
    if (small_rows) {
        r.block_rows = 32;
        r.block_cols = 32;
    } else if (c.ggml_is_quantized(k_type) or c.ggml_is_quantized(v_type) or hsk >= 256 or hsv >= 256) {
        r.block_rows = if (hsk >= 512 or hsv >= 512) 32 else 64;
        r.block_cols = 32;
    } else {
        r.block_rows = 64;
        r.block_cols = 64;
    }
    r.subgroup_size = dev.subgroup_size;
    r.workgroup_size = if (small_rows and (D % 32) == 0) 256 else 128;
    return r;
}

/// Ports `get_fa_tuning_params` (ggml-vulkan.cpp:3921 @5266f24da).
pub fn tuningParams(dev: *const Device, hsk: u32, hsv: u32, n_rows: u32, n_kv: u32, k_type: c.ggml_type, v_type: c.ggml_type, f32acc: bool) FaTuningParams {
    var path: FaCodePath = if (dev.coopmat2) .coopmat2 else if (dev.coopmat1_fa_support) .coopmat1 else .scalar;

    if (path == .coopmat2 and k_type == c.GGML_TYPE_BF16 and !dev.coopmat2_bf16_support) path = .coopmat1;
    if (path == .coopmat1 and k_type == c.GGML_TYPE_BF16 and !dev.coopmat_bf16_support) path = .scalar;
    // Nvidia compiler bug, see https://github.com/ggml-org/llama.cpp/pull/19075#issuecomment-3820716090
    if (path == .coopmat1 and dev.architecture == .nvidia_turing) path = .scalar;

    if (path == .coopmat1) {
        const shape_ok = (f32acc and dev.coopmat_support_16x16x16_f32acc) or
            (!f32acc and dev.coopmat_support_16x16x16_f16acc);
        const params = tuningCoopmat1(dev, hsk, hsv);
        const shmem_ok = coopmatShmemSupport(dev, &params, hsk, hsv, f32acc, k_type);
        if (!shape_ok or !shmem_ok) path = .scalar;
    }

    // scalar is faster than coopmat when N==1
    if (n_rows == 1 and (path == .coopmat1 or path == .coopmat2)) path = .scalar;

    return switch (path) {
        .scalar => tuningScalar(dev, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc),
        .coopmat1 => tuningCoopmat1(dev, hsk, hsv),
        .coopmat2 => tuningCoopmat2(dev, hsk, hsv, n_rows, k_type, v_type),
    };
}

/// Ports `get_fa_pipeline_state` (ggml-vulkan.cpp:3965 @5266f24da).
pub fn pipelineState(dev: *const Device, params: *const FaTuningParams, hsk: u32, hsv: u32, aligned: bool, f32acc: bool, use_mask: bool, use_mask_opt: bool, use_logit_softcap: bool, k_type: c.ggml_type, v_type: c.ggml_type) FaPipelineState {
    const old_amd_windows = dev.vendor_id == vendor_id_amd and dev.driver_id == vk.VK_DRIVER_ID_AMD_PROPRIETARY and
        (dev.architecture == .amd_gcn or dev.architecture == .amd_rdna1 or dev.architecture == .amd_rdna2);
    const flags: u32 = @as(u32, @intFromBool(use_mask_opt)) |
        @as(u32, if (use_mask) 2 else 0) |
        @as(u32, if (use_logit_softcap) 4 else 0) |
        @as(u32, if (old_amd_windows) 8 else 0);
    const subgroup_size: u32 = if (params.disable_subgroups) 0 else params.subgroup_size;
    return .{
        .HSK = hsk,
        .HSV = hsv,
        .Br = params.block_rows,
        .Bc = params.block_cols,
        .D_split = params.d_split,
        .row_split = params.row_split,
        .shmem_staging = params.shmem_staging,
        .path = params.path,
        .workgroup_size = params.workgroup_size,
        .subgroup_size = subgroup_size,
        .aligned = aligned,
        .f32acc = f32acc,
        .flags = flags,
        .limit_occupancy_shmem = params.limit_occupancy_shmem,
        .k_type = k_type,
        .v_type = v_type,
    };
}

/// Ports `ggml_vk_flash_attn_scalar_shmem_support` (ggml-vulkan.cpp:10806 @5266f24da).
/// Must track the shader; it overestimates `tmpsh` slightly.
pub fn scalarShmemSupport(dev: *const Device, params: *const FaTuningParams, hsk: u32, hsv: u32, f32acc: bool, k_type: c.ggml_type, v_type: c.ggml_type) bool {
    _ = f32acc;
    const wg_size = params.workgroup_size;
    const Br = params.block_rows;
    const Bc = params.block_cols;

    // BF16 uses the fp32 shader (FLOAT_TYPE=float)
    const float_type_size: u32 = if (dev.fp16 and k_type != c.GGML_TYPE_BF16) @sizeOf(c.ggml_fp16_t) else @sizeOf(f32);
    const mmq = m.shaders.faScalarUsesMmq(dev, k_type, v_type);

    const tmpsh = wg_size * @sizeOf(f32);
    const tmpshv4 = wg_size * 4 * float_type_size;
    const masksh = Bc * (Br + 1) * float_type_size;
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its table is always allocated.
    const iq_shmem = 16 * float_type_size;

    var Qf: u32 = undefined;
    var kvsh: u32 = undefined;
    var kblocksh_size: u32 = undefined;
    if (mmq) {
        // block_b_cache: int32_t qs[8] + FLOAT_TYPEV2 ds
        const block_b_size = 8 * @sizeOf(i32) + 2 * float_type_size;
        Qf = Br * (hsk / 32) * block_b_size;
        // kvsh uses D = HSV (K goes through kblocksh instead)
        kvsh = if (params.shmem_staging) Bc * (hsv / 4 + 1) * 4 * float_type_size else 4 * float_type_size;
        // A superset block_a_cache fitting every FA quant: int32_t qs[8] + uint32_t qh + FLOAT_TYPEV2 dm.
        const block_a_size = 8 * @sizeOf(i32) + @sizeOf(u32) + 2 * float_type_size;
        kblocksh_size = if (params.shmem_staging) Bc * (hsk / 32) * block_a_size else block_a_size;
    } else {
        Qf = Br * (hsk / 4 + 1) * 4 * float_type_size;
        const D = @max(hsk, hsv);
        kvsh = if (params.shmem_staging) Bc * (D / 4 + 1) * 4 * float_type_size else 4 * float_type_size;
        kblocksh_size = 0;
    }

    const total_size = tmpsh + tmpshv4 + masksh + iq_shmem + Qf + kvsh + kblocksh_size;
    return total_size <= dev.properties.limits.maxComputeSharedMemorySize;
}

/// Ports `ggml_vk_flash_attn_coopmat_shmem_support` (ggml-vulkan.cpp:10857 @5266f24da).
pub fn coopmatShmemSupport(dev: *const Device, params: *const FaTuningParams, hsk: u32, hsv: u32, f32acc: bool, k_type: c.ggml_type) bool {
    const Br = params.block_rows;
    const Bc = params.block_cols;
    const MatBr: u32 = 16;
    const MatBc: u32 = 16;
    const row_split = Bc / MatBc;
    const hsk_pad: u32 = @intCast(roundupPow2(hsk, 16));
    const hsv_pad: u32 = @intCast(roundupPow2(hsv, 16));
    const acctype: u32 = if (f32acc) 4 else 2;
    const f16vec4: u32 = 8;

    const tmpsh = (Bc / MatBc) * @sizeOf(f32);
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its table is always allocated.
    const iq_shmem = 16 * @sizeOf(c.ggml_fp16_t);

    const qstride = hsk_pad / 4 + 2;
    const Qf = Br * qstride * f16vec4;
    const psh_stride = Br / 4 + 2;
    const Psh = Bc * psh_stride * f16vec4;
    const sfshstride = if (hsk <= 128) Br + 8 else Br;
    const sfsh = Bc * sfshstride * acctype;

    const kvshstride = (if (params.shmem_staging) @max(hsk_pad, hsv_pad) else MatBr) / 4 + 2;
    const vsh_stride = MatBc / 4 * row_split;
    const ksh = (if (kvshstride >= vsh_stride) Bc * kvshstride else Bc * vsh_stride) * f16vec4;

    // BF16's PVMat accumulator is f32, so pvsh is vec4 (16 bytes)
    const pvsh_elem_size: u32 = if (k_type == c.GGML_TYPE_BF16) 16 else f16vec4;
    const osh_stride = params.row_split * MatBr / 4;
    const pvsh = MatBc * osh_stride * pvsh_elem_size;
    const slope = Br * acctype;

    const total_size = tmpsh + iq_shmem + Qf + Psh + sfsh + ksh + pvsh + slope;
    return total_size <= dev.properties.limits.maxComputeSharedMemorySize;
}

/// The `std::map` lookup the C++ does under `compile_mutex`: the pipeline
/// for `key`, created empty on first sight for the compile walk to fill.
fn dynamicPipeline(ctx: *BackendCtx, map: anytype, key: anytype) *Pipeline {
    const dev = ctx.device;
    _ = std.c.pthread_mutex_lock(&dev.compile_mutex);
    defer _ = std.c.pthread_mutex_unlock(&dev.compile_mutex);
    const gop = map.getOrPut(m.common.allocator, key) catch impl.abort("ggml_vulkan: out of memory");
    if (!gop.found_existing or gop.value_ptr.* == null) gop.value_ptr.* = Pipeline.create();
    return gop.value_ptr.*.?;
}

/// Whether a K or V tensor is a dense cache: rows contiguous, heads then
/// cells packed with no gaps, and streams too unless there is only one.
fn isDenseKvCache(t: *const Tensor) bool {
    return t.nb[0] == c.ggml_type_size(t.type) and
        t.nb[2] == c.ggml_row_size(t.type, t.ne[0]) and
        t.nb[1] == t.nb[2] * @as(usize, @intCast(t.ne[2])) and
        (t.ne[3] == 1 or t.nb[3] == t.nb[1] * @as(usize, @intCast(t.ne[1])));
}

/// Ports `ggml_vk_flash_attn` (ggml-vulkan.cpp:10905 @5266f24da).
pub fn flashAttn(ctx: *BackendCtx, subctx: *Context, q: *const Tensor, k: *const Tensor, v: *const Tensor, mask: ?*const Tensor, sinks: ?*const Tensor, dst: *Tensor) void {
    const dev = ctx.device;
    const limits = &dev.properties.limits;
    const u = ops.u32Of;

    const nem0: u32 = if (mask) |t| u(t.ne[0]) else 0;
    const nem1: u32 = if (mask) |t| u(t.ne[1]) else 0;
    const nem2: u32 = if (mask) |t| u(t.ne[2]) else 0;
    const nem3: u32 = if (mask) |t| u(t.ne[3]) else 0;

    const HSK = u(k.ne[0]);
    const HSV = u(v.ne[0]);
    var N = u(q.ne[1]);
    const KV = u(k.ne[1]);

    impl.assert(dst.ne[0] == HSV, "ne0 == HSV");
    impl.assert(dst.ne[2] == N, "ne2 == N");
    // input tensor rows must be contiguous
    impl.assert(q.nb[0] == c.ggml_type_size(q.type), "nbq0 == ggml_type_size(q->type)");
    impl.assert(k.nb[0] == c.ggml_type_size(k.type), "nbk0 == ggml_type_size(k->type)");
    impl.assert(v.nb[0] == c.ggml_type_size(v.type), "nbv0 == ggml_type_size(v->type)");
    impl.assert(q.ne[0] == HSK, "neq0 == HSK");
    impl.assert(q.ne[1] == N, "neq1 == N");
    impl.assert(v.ne[1] == k.ne[1], "nev1 == nek1");
    // dst cannot be transposed or permuted
    impl.assert(dst.nb[0] == @sizeOf(f32), "nb0 == sizeof(float)");
    impl.assert(dst.nb[0] <= dst.nb[1], "nb0 <= nb1");
    impl.assert(dst.nb[1] <= dst.nb[2], "nb1 <= nb2");
    impl.assert(dst.nb[2] <= dst.nb[3], "nb2 <= nb3");

    var gqa_ratio: u32 = 1;
    const qk_ratio = u(@divTrunc(q.ne[2], k.ne[2]));
    var workgroups_x = u(q.ne[1]);
    var workgroups_y = u(q.ne[2]);
    const workgroups_z = u(q.ne[3]);

    const f32acc = !dev.fp16 or dst.op_params[3] == c.GGML_PREC_F32 or k.type == c.GGML_TYPE_BF16;

    // Dequantize K/V once into an f16 scratch, reordered so FA can read without a stride.
    const isF = struct {
        fn f(t: c.ggml_type) bool {
            return t == c.GGML_TYPE_F16 or t == c.GGML_TYPE_BF16 or t == c.GGML_TYPE_F32;
        }
    }.f;
    const k_quant = !isF(k.type);
    const v_quant = !isF(v.type);
    const fp16sz: u64 = @sizeOf(c.ggml_fp16_t);
    const use_dequant_kv = k_quant and v_quant and q.ne[1] >= 64 and
        isDenseKvCache(k) and isDenseKvCache(v) and
        @as(u64, @intCast(c.ggml_nelements(k))) * fp16sz <= limits.maxStorageBufferRange and
        @as(u64, @intCast(c.ggml_nelements(v))) * fp16sz <= limits.maxStorageBufferRange and
        dev.pipeline_dequant_transpose[@intCast(k.type)] != null and
        dev.pipeline_dequant_transpose[@intCast(v.type)] != null and
        // coopmat2 path does not benefit from the f16 scratch
        !dev.coopmat2 and
        // Intel Xe1 regresses, see PR 25494
        (dev.vendor_id != vendor_id_intel or (dev.coopmat_support and dev.architecture != .intel_xe1));

    const k_type_eff: c.ggml_type = if (use_dequant_kv) c.GGML_TYPE_F16 else k.type;
    const v_type_eff: c.ggml_type = if (use_dequant_kv) c.GGML_TYPE_F16 else v.type;

    // Scalar and coopmat1 FA can use the "large" size to accommodate gqa;
    // coopmat2 always uses the small one (still large enough for gqa).
    var tuning = tuningParams(dev, HSK, HSV, 512, KV, k_type_eff, v_type_eff, f32acc);
    const max_gqa = @min(tuning.block_rows, 32);

    if (N <= 8 and qk_ratio > 1 and qk_ratio <= max_gqa and
        @as(i64, qk_ratio) * k.ne[2] == q.ne[2] and k.ne[2] == v.ne[2] and nem2 <= 1)
    {
        // Grouped-query attention: N becomes gqa_ratio and the y workgroups
        // shrink to match; the shader indexes Q's dimension 2 instead.
        gqa_ratio = qk_ratio;
        N = gqa_ratio;
        workgroups_y /= gqa_ratio;
    }

    tuning = tuningParams(dev, HSK, HSV, N, KV, k_type_eff, v_type_eff, f32acc);

    const q_stride: u32 = @truncate(q.nb[1] / c.ggml_type_size(q.type));
    var k_stride: u32 = @truncate(k.nb[1] / c.ggml_type_size(k.type));
    var v_stride: u32 = @truncate(v.nb[1] / c.ggml_type_size(v.type));
    // For F32, the shader treats it as a block of size 4 (for vec4 loads)
    if (k.type == c.GGML_TYPE_F32) k_stride /= 4;
    if (v.type == c.GGML_TYPE_F32) v_stride /= 4;

    var nbk2_eff: u32 = @truncate(k.nb[2]);
    var nbk3_eff: u32 = @truncate(k.nb[3]);
    var nbv2_eff: u32 = @truncate(v.nb[2]);
    var nbv3_eff: u32 = @truncate(v.nb[3]);
    if (use_dequant_kv) {
        k_stride = HSK;
        v_stride = HSV;
        nbk2_eff = @truncate(@as(u64, HSK) * KV * fp16sz);
        nbk3_eff = @truncate(@as(u64, HSK) * KV * @as(u64, @intCast(k.ne[2])) * fp16sz);
        nbv2_eff = @truncate(@as(u64, HSV) * KV * fp16sz);
        nbv3_eff = @truncate(@as(u64, HSV) * KV * @as(u64, @intCast(v.ne[2])) * fp16sz);
    }

    const alignment = tuning.block_cols;
    // The "aligned" shader variant forcibly aligns strides, for performance.
    var aligned = (KV % alignment) == 0 and
        (q_stride & 7) == 0 and (k_stride & 7) == 0 and (v_stride & 7) == 0;
    // coopmat2 needs the variant that clamps loads when HSK/HSV aren't sufficiently aligned.
    if (((HSK | HSV) % 16) != 0 and tuning.path == .coopmat2) aligned = false;

    var scale: f32 = @bitCast(dst.op_params[0]);
    const max_bias: f32 = @bitCast(dst.op_params[1]);
    const logit_softcap: f32 = @bitCast(dst.op_params[2]);
    if (logit_softcap != 0) scale /= logit_softcap;

    // Only use mask opt when the mask is fairly large. Not tuned extensively.
    const use_mask_opt = mask != null and nem1 >= 32 and nem0 *% nem1 > 32768 and nem0 >= tuning.block_cols * 16 and
        (dev.architecture != .amd_gcn or HSK > 256 or HSV > 256);

    const state = pipelineState(dev, &tuning, HSK, HSV, aligned, f32acc, mask != null, use_mask_opt, logit_softcap != 0, k_type_eff, v_type_eff);
    const pipeline = dynamicPipeline(ctx, &dev.pipeline_flash_attn_f32_f16, state);

    // Compile early to initialize wg_denoms.
    m.context.requestDescriptorSets(ctx, pipeline, 1);

    var split_kv = KV;
    var split_k: u32 = 1;

    // Intel Alchemist prefers more workgroups
    const multiplier: u32 = if (dev.vendor_id == vendor_id_intel and dev.architecture != .intel_xe2) 2 else 1;
    // A placeholder core count when none is known; split_k is a big help for perf.
    const shader_core_count: u32 = if (dev.shader_core_count != 0) dev.shader_core_count * multiplier else 16;

    const Br = state.Br;
    const Bc = state.Bc;
    impl.assert(Br == pipeline.wg_denoms[0], "Br == pipeline->wg_denoms[0]");
    const Tr = ceilDiv(N, Br);

    // Try to use split_k when KV is large enough to be worth the overhead.
    if (gqa_ratio > 1 and workgroups_x <= Br) {
        split_k = shader_core_count * 2 / (workgroups_x *% workgroups_y *% workgroups_z);
    } else if (gqa_ratio <= 1) {
        const total_wgs_no_split = Tr *% workgroups_y *% workgroups_z;
        if (total_wgs_no_split < shader_core_count * 2) split_k = shader_core_count * 2 / total_wgs_no_split;
    }

    if (split_k > 1) {
        // Split KV evenly into split_k chunks, each a multiple of the
        // alignment, and recompute split_k from that.
        split_kv = @intCast(roundupPow2(@max(1, KV / split_k), alignment));
        split_k = ceilDiv(KV, split_kv);
    }

    const ne1: u64 = @intCast(dst.ne[1]);
    const ne2: u64 = @intCast(dst.ne[2]);
    const ne3: u64 = @intCast(dst.ne[3]);

    // Split-k temporaries: per split and batch, the O matrix (D x ne1),
    // then the per-row m and L values -- matrices first, ordered
    // [HSV, ne1, k, ne2, ne3], then the rows, [ne1, k, ne2, ne3].
    const split_k_size: u64 = if (split_k > 1) (@as(u64, HSV) * ne1 * @sizeOf(f32) + ne1 * @sizeOf(f32) * 2) * split_k * ne2 * ne3 else 0;
    if (split_k_size > limits.maxStorageBufferRange) impl.abort("Requested preallocation size is too large");
    if (ctx.prealloc_size_split_k < split_k_size) {
        ctx.prealloc_size_split_k = split_k_size;
        m.nodes.preallocateBuffers(ctx, subctx);
    }

    const mask_opt_num_dwords = ceilDiv(nem0, 16 * Bc);
    const mask_opt_size: u64 = @sizeOf(u32) * @as(u64, mask_opt_num_dwords) * ceilDiv(nem1, Br) * nem2 * nem3;

    var pipeline_fa_mask_opt: ?*Pipeline = null;
    if (use_mask_opt) {
        pipeline_fa_mask_opt = dynamicPipeline(ctx, &dev.pipeline_fa_mask_opt, .{ Br, Bc });
        m.context.requestDescriptorSets(ctx, pipeline_fa_mask_opt.?, 1);
        if (ctx.prealloc_size_y < mask_opt_size) {
            ctx.prealloc_size_y = mask_opt_size;
            m.nodes.preallocateBuffers(ctx, subctx);
        }
        if (ctx.prealloc_y_need_sync) m.context.syncBuffers(ctx, subctx);
    }

    const n_head_kv = u(q.ne[2]);
    const n_head_log2: u32 = @as(u32, 1) << @intCast(@as(u32, @intFromFloat(floorf(log2f(@floatFromInt(n_head_kv))))));
    const nhl: f32 = @floatFromInt(n_head_log2);
    const m0 = powf(2.0, -max_bias / nhl);
    const m1 = powf(2.0, -(max_bias / 2.0) / nhl);

    const q_buf = m.context.tensorSubbuffer(ctx, q, false);
    var k_buf = m.context.tensorSubbuffer(ctx, k, false);
    var v_buf = m.context.tensorSubbuffer(ctx, v, false);
    const dst_buf = m.context.tensorSubbuffer(ctx, dst, false);
    const mask_buf = if (mask) |t| m.context.tensorSubbuffer(ctx, t, false) else q_buf;
    const sinks_buf = if (sinks) |t| m.context.tensorSubbuffer(ctx, t, false) else q_buf;
    const mask_opt_buf = if (use_mask_opt) m.context.subbuffer(ctx, ctx.prealloc_y.?, 0) else q_buf;

    if (use_dequant_kv) {
        const k_f16_sz = @as(u64, @intCast(c.ggml_nelements(k))) * fp16sz;
        const v_f16_sz = @as(u64, @intCast(c.ggml_nelements(v))) * fp16sz;
        if (ctx.prealloc_size_x < k_f16_sz + v_f16_sz) {
            ctx.prealloc_size_x = k_f16_sz + v_f16_sz;
            m.nodes.preallocateBuffers(ctx, subctx);
        }
        const tr_k = dev.pipeline_dequant_transpose[@intCast(k.type)].?;
        const tr_v = dev.pipeline_dequant_transpose[@intCast(v.type)].?;
        m.context.requestDescriptorSets(ctx, tr_k, 1);
        m.context.requestDescriptorSets(ctx, tr_v, 1);
        if (ctx.prealloc_x_need_sync) m.context.syncBuffers(ctx, subctx);

        const k_dst = sub(ctx.prealloc_x, 0, k_f16_sz);
        const v_dst = sub(ctx.prealloc_x, k_f16_sz, v_f16_sz);
        const k_nel = u(c.ggml_nelements(k));
        const v_nel = u(c.ggml_nelements(v));
        {
            const pc = [5]u32{ HSK, u(k.ne[2]), KV, 0, k_nel };
            m.context.dispatch(ctx, subctx, tr_k, &.{ k_buf, k_dst }, &pc, .{ k_nel, 1, 1 });
        }
        {
            const pc = [5]u32{ HSV, u(v.ne[2]), KV, 0, v_nel };
            m.context.dispatch(ctx, subctx, tr_v, &.{ v_buf, v_dst }, &pc, .{ v_nel, 1, 1 });
        }
        m.context.syncBuffers(ctx, subctx);
        k_buf = k_dst;
        v_buf = v_dst;
    }

    const mask_n_head_log2: u32 = (@as(u32, @intFromBool(sinks != null)) << 24) | n_head_log2;

    if (use_mask_opt) {
        const mk = mask.?;
        const h: u32 = @sizeOf(c.ggml_fp16_t);
        const opt_pc: push.FlashAttnMaskOpt = .{
            .nem0 = nem0,
            .nem1 = nem1,
            .nem2 = nem2,
            .nbm1 = @truncate(mk.nb[1] / h),
            .nbm2 = @truncate(mk.nb[2] / h),
            .nbm3 = @truncate(mk.nb[3] / h),
            .nbd1 = mask_opt_num_dwords,
            .nbd2 = mask_opt_num_dwords *% ceilDiv(nem1, Br),
            .nbd3 = mask_opt_num_dwords *% ceilDiv(nem1, Br) *% nem2,
        };
        m.context.dispatch(ctx, subctx, pipeline_fa_mask_opt.?, &.{ mask_buf, mask_opt_buf }, &opt_pc, .{ mask_opt_num_dwords, ceilDiv(nem1, Br), nem2 *% nem3 });
        m.context.syncBuffers(ctx, subctx);
    }

    const pc: push.FlashAttn = .{
        .N = N,
        .KV = KV,
        .ne1 = @truncate(ne1),
        .ne2 = @truncate(ne2),
        .ne3 = @truncate(ne3),
        .neq2 = u(q.ne[2]),
        .neq3 = u(q.ne[3]),
        .nek2 = u(k.ne[2]),
        .nek3 = u(k.ne[3]),
        .nev2 = u(v.ne[2]),
        .nev3 = u(v.ne[3]),
        .nem1 = nem1,
        .nem2 = nem2,
        .nem3 = nem3,
        .nb01 = q_stride,
        .nb02 = @truncate(q.nb[2]),
        .nb03 = @truncate(q.nb[3]),
        .nb11 = k_stride,
        .nb12 = nbk2_eff,
        .nb13 = nbk3_eff,
        .nb21 = v_stride,
        .nb22 = nbv2_eff,
        .nb23 = nbv3_eff,
        .scale = scale,
        .max_bias = max_bias,
        .logit_softcap = logit_softcap,
        .mask_n_head_log2 = mask_n_head_log2,
        .m0 = m0,
        .m1 = m1,
        .gqa_ratio = gqa_ratio,
        .split_kv = split_kv,
        .k_num = split_k,
    };

    if (split_k > 1) {
        const reduce = dev.pipeline_flash_attn_split_k_reduce.?;
        m.context.requestDescriptorSets(ctx, reduce, 1);
        if (ctx.prealloc_split_k_need_sync) m.context.syncBuffers(ctx, subctx);

        // workgroups_x now counts splits, so cancel the divide by wg_denoms[0].
        var dispatch_x: u32 = undefined;
        if (gqa_ratio > 1) {
            workgroups_x *%= pipeline.wg_denoms[0];
            dispatch_x = split_k *% workgroups_x;
        } else {
            dispatch_x = Tr *% split_k *% pipeline.wg_denoms[0];
        }

        const split_k_buf = m.context.subbuffer(ctx, ctx.prealloc_split_k.?, 0);
        m.context.dispatch(ctx, subctx, pipeline, &.{ q_buf, k_buf, v_buf, mask_buf, sinks_buf, split_k_buf, mask_opt_buf }, &pc, .{ dispatch_x, workgroups_y, workgroups_z });

        m.context.syncBuffers(ctx, subctx);
        const pc2: push.FlashAttnSplitKReduce = .{ .D = HSV, .ne1 = @truncate(ne1), .ne2 = @truncate(ne2), .ne3 = @truncate(ne3), .k_num = split_k, .sinks = @intFromBool(sinks != null) };
        m.context.dispatch(ctx, subctx, reduce, &.{ split_k_buf, sinks_buf, dst_buf }, &pc2, .{ @truncate(ne1), HSV, @truncate(ne2 *% ne3) });
        ctx.prealloc_split_k_need_sync = true;
    } else {
        // With gqa, one actual workgroup per batch: cancel out wg_denoms
        if (gqa_ratio > 1) workgroups_x *%= pipeline.wg_denoms[0];
        m.context.dispatch(ctx, subctx, pipeline, &.{ q_buf, k_buf, v_buf, mask_buf, sinks_buf, dst_buf, mask_opt_buf }, &pc, .{ workgroups_x, workgroups_y, workgroups_z });
    }

    if (use_dequant_kv) ctx.prealloc_x_need_sync = true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
