//! Flash attention: the tuning that picks a code path (scalar, coopmat1,
//! coopmat2) and its tile shape, the shared-memory checks behind it, and
//! the dispatch -- grouped-query folding, split-KV with a reduction pass,
//! the mask-skip prepass, and K/V dequantized once into fp16 scratch.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` and
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-types.h` at v0.7.0
//! (`620d2225c`). Upstream split `ggml-vulkan.cpp` into those files at
//! v0.5.0.

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

/// Ports `vk_fa_tuning_params` (ggml-vulkan-types.h:1350 @620d2225c).
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

/// Ports `get_fa_tuning_params_scalar` (ggml-vulkan.cpp:1200 @620d2225c).
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

/// Ports `get_fa_tuning_params_coopmat1` (ggml-vulkan.cpp:1275 @620d2225c).
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

/// Ports `get_fa_tuning_params_coopmat2` (ggml-vulkan.cpp:1306 @620d2225c).
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

/// Ports `get_fa_tuning_params` (ggml-vulkan.cpp:1334 @620d2225c).
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

/// Ports `get_fa_pipeline_state` (ggml-vulkan.cpp:1378 @620d2225c).
pub fn pipelineState(dev: *const Device, params: *const FaTuningParams, hsk: u32, hsv: u32, aligned: bool, f32acc: bool, use_mask: bool, use_mask_opt: bool, use_logit_softcap: bool, use_sparse: bool, k_type: c.ggml_type, v_type: c.ggml_type) FaPipelineState {
    const old_amd_windows = dev.vendor_id == vendor_id_amd and dev.driver_id == vk.VK_DRIVER_ID_AMD_PROPRIETARY and
        (dev.architecture == .amd_gcn or dev.architecture == .amd_rdna1 or dev.architecture == .amd_rdna2);
    const flags: u32 = @as(u32, @intFromBool(use_mask_opt)) |
        @as(u32, if (use_mask) 2 else 0) |
        @as(u32, if (use_logit_softcap) 4 else 0) |
        @as(u32, if (old_amd_windows) 8 else 0) |
        @as(u32, if (use_sparse) 16 else 0);
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

/// Ports `ggml_vk_flash_attn_scalar_shmem_support` (ggml-vulkan.cpp:7961 @620d2225c).
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

/// Ports `ggml_vk_flash_attn_coopmat_shmem_support` (ggml-vulkan.cpp:8012 @620d2225c).
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

/// Ports `ggml_vk_flash_attn` (ggml-vulkan.cpp:8060 @620d2225c).
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
    const kv_raw = (k.type == c.GGML_TYPE_F16 or k.type == c.GGML_TYPE_BF16) and k.type == v.type;
    // gated to AMD, except RDNA4, which regressed in benchmarks
    const kv_raw_strided = dev.vendor_id == vendor_id_amd and
        dev.architecture != .amd_rdna4 and
        q.ne[1] >= 256 and
        kv_raw and
        (k.nb[1] != @as(u64, HSK) * c.ggml_type_size(k.type) or
            v.nb[1] != @as(u64, HSV) * c.ggml_type_size(v.type)) and
        (HSK % 8) == 0 and (HSV % 8) == 0;
    const kv_scratch_ts: u64 = if (kv_raw_strided) c.ggml_type_size(k.type) else fp16sz;
    const use_dequant_kv = ((k_quant and v_quant) or kv_raw_strided) and q.ne[1] >= 64 and
        isDenseKvCache(k) and isDenseKvCache(v) and
        @as(u64, @intCast(c.ggml_nelements(k))) * kv_scratch_ts <= limits.maxStorageBufferRange and
        @as(u64, @intCast(c.ggml_nelements(v))) * kv_scratch_ts <= limits.maxStorageBufferRange and
        (kv_raw_strided or
            (dev.pipeline_dequant_transpose[@intCast(k.type)] != null and
                dev.pipeline_dequant_transpose[@intCast(v.type)] != null)) and
        // coopmat2 path does not benefit from the f16 scratch
        !dev.coopmat2 and
        // Intel Xe1 regresses, see PR 25494
        (dev.vendor_id != vendor_id_intel or (dev.coopmat_support and dev.architecture != .intel_xe1));

    const k_type_eff: c.ggml_type = if (use_dequant_kv and !kv_raw_strided) c.GGML_TYPE_F16 else k.type;
    const v_type_eff: c.ggml_type = if (use_dequant_kv and !kv_raw_strided) c.GGML_TYPE_F16 else v.type;

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

    var scale: f32 = @bitCast(dst.op_params[0]);
    const max_bias: f32 = @bitCast(dst.op_params[1]);
    const logit_softcap: f32 = @bitCast(dst.op_params[2]);
    if (logit_softcap != 0) scale /= logit_softcap;

    // Sparse mask hint (op_params[4]): compact the <= n_kv_max finite
    // positions and gather only those.
    const n_kv_max: i32 = if (mask != null) dst.op_params[4] else 0;
    const disable_sparse = DisableSparse.get();
    const kv_f16 = k_type_eff == c.GGML_TYPE_F16 and v_type_eff == c.GGML_TYPE_F16;
    // cm2 dense is fast, so it needs a larger reduction to win.
    // With quantized K/V, sparse only breaks even around 16x (measured on RDNA3/RDNA4).
    const min_ratio: i64 = if (tuning.path == .coopmat2) 4 else if (kv_f16) 2 else 16;
    // coopmat2 vector decode requires 8B strides.
    const sparseGatherAligned = struct {
        fn f(t: *const Tensor) bool {
            return (t.type != c.GGML_TYPE_F16 and t.type != c.GGML_TYPE_BF16) or
                (t.nb[1] | t.nb[2] | t.nb[3]) % (4 * @sizeOf(c.ggml_fp16_t)) == 0;
        }
    }.f;
    const use_sparse = !disable_sparse and n_kv_max > 0 and mask != null and
        max_bias == 0.0 and logit_softcap == 0.0 and
        (tuning.path != .coopmat2 or (sparseGatherAligned(k) and sparseGatherAligned(v))) and
        nem0 == KV and
        @as(i64, KV) >= @max(4096, min_ratio * @as(i64, n_kv_max)) and
        (gqa_ratio > 1 or (tuning.path == .scalar and N == 1));

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

    // Only use mask opt when the mask is fairly large. Not tuned extensively.
    var use_mask_opt = mask != null and !use_sparse and nem1 >= 32 and nem0 *% nem1 > 32768 and nem0 >= tuning.block_cols * 16 and
        (dev.architecture != .amd_gcn or HSK > 256 or HSV > 256);

    const state = pipelineState(dev, &tuning, HSK, HSV, aligned, f32acc, mask != null, use_mask_opt, logit_softcap != 0, use_sparse, k_type_eff, v_type_eff);

    // Intel Xe's two-phase decode path: its platforms, shapes and types.
    var xe_fa_opt = false;
    var fa_copy_qstate = false;
    const xe_fa_supported_platform =
        (dev.architecture == .intel_xe2 and dev.properties.deviceID != 0xFD80 and dev.properties.deviceID != 0xFD81) or
        (dev.architecture == .intel_xe1 and dev.coopmat_support and dev.uma);
    const xe_fa_supported_usage = @mod(q.ne[0], 32) == 0 and @mod(v.ne[0], 16) == 0 and q.nb[1] > q.nb[2] and k.nb[1] > k.nb[2] and v.nb[1] > v.nb[2] and mask != null;
    const xe_fa_supported_dtype = q.type == c.GGML_TYPE_F32 and k.type == c.GGML_TYPE_F16 and v.type == c.GGML_TYPE_F16 and (mask != null and mask.?.type == c.GGML_TYPE_F16);
    var xe_fa_pipeline_dual_phases: m.device.XeFaDecodePipelines = .{};
    var size_p: u64 = 0;
    var size_group_max: u64 = 0;

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
    // Sparse: split_kv carries n_kv_max, split_k partitions its blocks for
    // occupancy.
    if (use_sparse) {
        split_kv = @intCast(n_kv_max);
        const total_blocks = ceilDiv(@as(u32, @intCast(n_kv_max)), Bc);
        const base_wgs = (if (gqa_ratio > 1) workgroups_x else Tr) *% workgroups_y *% workgroups_z;
        if (base_wgs < shader_core_count * 2) split_k = shader_core_count * 2 / base_wgs;
        split_k = @max(1, @min(split_k, total_blocks));
        // Match the shader's per-split block count so no split is empty.
        const per_blocks = ceilDiv(total_blocks, split_k);
        split_k = ceilDiv(total_blocks, per_blocks);
    } else if (gqa_ratio > 1 and workgroups_x <= Br) {
        split_k = shader_core_count * 2 / (workgroups_x *% workgroups_y *% workgroups_z);
    } else if (gqa_ratio <= 1) {
        const total_wgs_no_split = Tr *% workgroups_y *% workgroups_z;
        if (total_wgs_no_split < shader_core_count * 2) split_k = shader_core_count * 2 / total_wgs_no_split;
    }

    if (!use_sparse and split_k > 1) {
        // Split KV evenly into split_k chunks, each a multiple of the
        // alignment, and recompute split_k from that.
        split_kv = @intCast(roundupPow2(@max(1, KV / split_k), alignment));
        split_k = ceilDiv(KV, split_kv);
        xe_fa_opt = xe_fa_supported_platform and xe_fa_supported_usage and xe_fa_supported_dtype;
        if (xe_fa_opt) {
            const split_p_size: u64 = 32;
            const nek1: u64 = @intCast(k.ne[1]);
            const max_dim = (nek1 +% split_p_size -% 1) / split_p_size;
            const p_dim = max_dim *% split_p_size;
            // The two-phase pipelines exist only where glslc took coopmat
            // (`fa_decode_ph1`/`ph2` are generated under it); the C++ holds
            // `compile_mutex` for the rest of this block when it does.
            const have_xe_pipelines = comptime m.instance.glslc.coopmat;
            if (have_xe_pipelines) _ = std.c.pthread_mutex_lock(&dev.compile_mutex);
            defer if (have_xe_pipelines) {
                _ = std.c.pthread_mutex_unlock(&dev.compile_mutex);
            };
            if (have_xe_pipelines) {
                const xe_key: m.device.XeFaDecodeKey = .{ u(q.ne[0]), u(v.ne[0]), qk_ratio, u(q.ne[1]) };
                const gop = dev.pipeline_xe_fa_decode_dual_phases.getOrPut(m.common.allocator, xe_key) catch impl.abort("ggml_vulkan: out of memory");
                if (!gop.found_existing) gop.value_ptr.* = .{ .first = Pipeline.create(), .second = Pipeline.create() };
                xe_fa_pipeline_dual_phases = gop.value_ptr.*;
            }
            if (xe_fa_pipeline_dual_phases.first == null or xe_fa_pipeline_dual_phases.second == null) {
                xe_fa_opt = false;
                fa_copy_qstate = false;
            } else {
                const neq1: u64 = @intCast(q.ne[1]);
                const neq2: u64 = @intCast(q.ne[2]);
                const neq3: u64 = @intCast(q.ne[3]);
                size_p = neq1 *% neq2 *% p_dim *% neq3 *% @sizeOf(c.ggml_fp16_t);
                size_group_max = neq1 *% neq2 *% max_dim *% neq3 *% @sizeOf(f32);
                const temp_size = @as(u64, @intCast(c.ggml_nelements(q))) *% @sizeOf(c.ggml_fp16_t) +% size_p +% size_group_max;
                fa_copy_qstate = true;
                if (ctx.prealloc_size_x < temp_size) {
                    ctx.prealloc_size_x = temp_size;
                    m.nodes.preallocateBuffers(ctx, subctx);
                }

                if (ctx.prealloc_x_need_sync) m.context.syncBuffers(ctx, subctx);
            }
        }
    }

    if (xe_fa_opt) use_mask_opt = false;

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

    // Sparse index scratch reuses prealloc_y (mutually exclusive with mask opt).
    const sparse_idx_size: u64 = if (use_sparse) @sizeOf(i32) *% @as(u64, @intCast(n_kv_max)) *% nem1 *% nem2 *% nem3 else 0;
    const sparse_compact_pipeline = if (dev.fa_sparse_compact_use_subgroups) dev.pipeline_fa_sparse_compact_subgroup else dev.pipeline_fa_sparse_compact;
    if (use_sparse) {
        m.context.requestDescriptorSets(ctx, sparse_compact_pipeline.?, 1);
        if (ctx.prealloc_size_y < sparse_idx_size) {
            ctx.prealloc_size_y = sparse_idx_size;
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
    const sparse_buf = if (use_sparse) m.context.subbuffer(ctx, ctx.prealloc_y.?, 0) else q_buf;
    if (use_mask_opt or use_sparse) {
        // the mask opt bits and the sparse index list overwrite a matmul input converted into prealloc_y
        ctx.prealloc_y_last_pipeline_used = null;
        ctx.prealloc_y_last_tensor_used = null;
    }

    if (use_dequant_kv) {
        const k_sz = @as(u64, @intCast(c.ggml_nelements(k))) * kv_scratch_ts;
        const v_sz = @as(u64, @intCast(c.ggml_nelements(v))) * kv_scratch_ts;
        if (ctx.prealloc_size_x < k_sz + v_sz) {
            ctx.prealloc_size_x = k_sz + v_sz;
            m.nodes.preallocateBuffers(ctx, subctx);
        }
        const k_dst = sub(ctx.prealloc_x, 0, k_sz);
        const v_dst = sub(ctx.prealloc_x, k_sz, v_sz);
        const k_nel = u(c.ggml_nelements(k));
        const v_nel = u(c.ggml_nelements(v));
        if (kv_raw_strided) {
            // Copy the strided f16/bf16 K/V once into a contiguous
            // [KV][heads][HS] scratch, laid out as the dequant transpose's.
            const cp_k = m.op_pipeline.getCpyPipeline(ctx, k, null, k.type).?;
            const cp_v = m.op_pipeline.getCpyPipeline(ctx, v, null, v.type).?;
            m.context.requestDescriptorSets(ctx, cp_k, 1);
            m.context.requestDescriptorSets(ctx, cp_v, 1);
            if (ctx.prealloc_x_need_sync) m.context.syncBuffers(ctx, subctx);
            const makePc = struct {
                fn f(hs: u32, nh: u32, kv: u32, nel: u32) push.Unary {
                    var pc: push.Unary = .{};
                    pc.ne = nel;
                    pc.ne00 = hs;
                    pc.ne01 = nh;
                    pc.ne02 = kv;
                    pc.ne03 = nel / (hs *% nh *% kv);
                    pc.nb00 = 1;
                    pc.nb01 = hs;
                    pc.nb02 = hs *% nh;
                    pc.nb03 = hs *% nh *% kv;
                    pc.ne10 = hs;
                    pc.ne11 = nh;
                    pc.ne12 = kv;
                    pc.ne13 = pc.ne03;
                    pc.nb10 = 1;
                    pc.nb11 = hs *% kv;
                    pc.nb12 = hs;
                    pc.nb13 = hs *% kv *% nh;
                    ops.initPushconstFastdiv(&pc);
                    return pc;
                }
            }.f;
            const cpyElems = struct {
                fn f(ne: u32) [3]u32 {
                    if (ne > 262144) return .{ 512, 512, ops.ceilDiv(ne, 262144) };
                    if (ne > 512) return .{ 512, ops.ceilDiv(ne, 512), 1 };
                    return .{ ne, 1, 1 };
                }
            }.f;
            const pc_k = makePc(HSK, u(k.ne[2]), KV, k_nel);
            m.context.dispatch(ctx, subctx, cp_k, &.{ k_buf, k_dst }, &pc_k, cpyElems(k_nel));
            const pc_v = makePc(HSV, u(v.ne[2]), KV, v_nel);
            m.context.dispatch(ctx, subctx, cp_v, &.{ v_buf, v_dst }, &pc_v, cpyElems(v_nel));
        } else {
            const tr_k = dev.pipeline_dequant_transpose[@intCast(k.type)].?;
            const tr_v = dev.pipeline_dequant_transpose[@intCast(v.type)].?;
            m.context.requestDescriptorSets(ctx, tr_k, 1);
            m.context.requestDescriptorSets(ctx, tr_v, 1);
            if (ctx.prealloc_x_need_sync) m.context.syncBuffers(ctx, subctx);
            {
                const pc = [5]u32{ HSK, u(k.ne[2]), KV, 0, k_nel };
                m.context.dispatch(ctx, subctx, tr_k, &.{ k_buf, k_dst }, &pc, .{ k_nel, 1, 1 });
            }
            {
                const pc = [5]u32{ HSV, u(v.ne[2]), KV, 0, v_nel };
                m.context.dispatch(ctx, subctx, tr_v, &.{ v_buf, v_dst }, &pc, .{ v_nel, 1, 1 });
            }
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

    if (use_sparse) {
        const mk = mask.?;
        const h: u32 = @sizeOf(c.ggml_fp16_t);
        const sc_pc: push.FlashAttnSparseCompact = .{
            .KV = KV,
            .nem1 = nem1,
            .nem2 = nem2,
            .nbm1 = @truncate(mk.nb[1] / h),
            .nbm2 = @truncate(mk.nb[2] / h),
            .nbm3 = @truncate(mk.nb[3] / h),
            .n_kv_max = @intCast(n_kv_max),
        };
        m.context.dispatch(ctx, subctx, sparse_compact_pipeline.?, &.{ mask_buf, sparse_buf }, &sc_pc, .{ nem1, nem2, nem3 });
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

    if (xe_fa_opt and split_k > 1) {
        xeFaDecode(ctx, subctx, q, k, v, mask, sinks, dst, .{
            .pipelines = xe_fa_pipeline_dual_phases,
            .qk_ratio = qk_ratio,
            .k_stride = k_stride,
            .v_stride = v_stride,
            .scale = scale,
            .fa_copy_qstate = fa_copy_qstate,
            .size_p = size_p,
            .size_group_max = size_group_max,
            .q_buf = q_buf,
            .k_buf = k_buf,
            .v_buf = v_buf,
            .mask_buf = mask_buf,
            .sinks_buf = sinks_buf,
            .dst_buf = dst_buf,
        });
    } else if (split_k > 1) {
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
        m.context.dispatch(ctx, subctx, pipeline, &.{ q_buf, k_buf, v_buf, mask_buf, sinks_buf, split_k_buf, mask_opt_buf, sparse_buf }, &pc, .{ dispatch_x, workgroups_y, workgroups_z });

        m.context.syncBuffers(ctx, subctx);
        const pc2: push.FlashAttnSplitKReduce = .{ .D = HSV, .ne1 = @truncate(ne1), .ne2 = @truncate(ne2), .ne3 = @truncate(ne3), .k_num = split_k, .sinks = @intFromBool(sinks != null) };
        m.context.dispatch(ctx, subctx, reduce, &.{ split_k_buf, sinks_buf, dst_buf }, &pc2, .{ @truncate(ne1), HSV, @truncate(ne2 *% ne3) });
        ctx.prealloc_split_k_need_sync = true;
    } else {
        // With gqa, one actual workgroup per batch: cancel out wg_denoms
        if (gqa_ratio > 1) workgroups_x *%= pipeline.wg_denoms[0];
        m.context.dispatch(ctx, subctx, pipeline, &.{ q_buf, k_buf, v_buf, mask_buf, sinks_buf, dst_buf, mask_opt_buf, sparse_buf }, &pc, .{ workgroups_x, workgroups_y, workgroups_z });
    }

    if (use_dequant_kv) ctx.prealloc_x_need_sync = true;
    if (use_mask_opt or use_sparse) ctx.prealloc_y_need_sync = true;
}

/// The function-static `disable_sparse` (ggml-vulkan.cpp:8184 @620d2225c):
/// whether `GGML_VK_FA_SPARSE_DISABLE` is set, read once, on first use.
const DisableSparse = struct {
    /// 0 unread, 1 false, 2 true.
    var state: std.atomic.Value(u8) = .init(0);

    fn get() bool {
        const v = state.load(.acquire);
        if (v != 0) return v == 2;
        const set = m.common.envSet("GGML_VK_FA_SPARSE_DISABLE");
        state.store(if (set) 2 else 1, .release);
        return set;
    }
};

/// What `xeFaDecode` takes from `flashAttn`'s locals.
const XeFaArgs = struct {
    pipelines: m.device.XeFaDecodePipelines,
    qk_ratio: u32,
    k_stride: u32,
    v_stride: u32,
    scale: f32,
    fa_copy_qstate: bool,
    size_p: u64,
    size_group_max: u64,
    q_buf: Subbuffer,
    k_buf: Subbuffer,
    v_buf: Subbuffer,
    mask_buf: Subbuffer,
    sinks_buf: Subbuffer,
    dst_buf: Subbuffer,
};

/// The `upper_power_of_2` and `upper_power_of_2` lambdas
/// (ggml-vulkan.cpp:8536, 3178 @620d2225c), one in `ggml_vk_flash_attn` and
/// one in `ggml_vk_load_shaders`, which are the same: the next power of two
/// at or above `in`.
pub fn upperPowerOf2(in: u32) u32 {
    impl.assert(in != 0, "in != 0");
    if (in <= 1) return 1;
    var ret = in - 1;
    ret |= ret >> 1;
    ret |= ret >> 2;
    ret |= ret >> 4;
    ret |= ret >> 8;
    ret |= ret >> 16;
    return ret +% 1;
}

/// The `xe_fa_opt` arm (ggml-vulkan.cpp:8535 @620d2225c) of
/// `ggml_vk_flash_attn`, taken when `split_k > 1`: Intel Xe's two-phase decode -- Q copied
/// to fp16 into `prealloc_x`, phase 1 writing P and each group's max beside
/// it, phase 2 reducing them against V into `dst`.
fn xeFaDecode(ctx: *BackendCtx, subctx: *Context, q: *const Tensor, k: *const Tensor, v: *const Tensor, mask: ?*const Tensor, sinks: ?*const Tensor, dst: *const Tensor, a: XeFaArgs) void {
    const dev = ctx.device;
    const u = ops.u32Of;
    const to_fp16_vk_0 = m.ops_matmul.getToFp16(ctx, q.type);
    const out_dim_per_wg: i64 = if (a.qk_ratio > 16) 8 else 16;
    const x_ne: u64 = @intCast(c.ggml_nelements(q));
    var temp_buf_offset: u64 = 0;
    const head_stride_k: u32 = @truncate(k.nb[2] / c.ggml_type_size(k.type));
    const head_stride_v: u32 = @truncate(v.nb[2] / c.ggml_type_size(v.type));
    const batch_stride_q: u32 = @truncate(q.nb[3] / c.ggml_type_size(q.type));
    const batch_stride_k: u32 = @truncate(k.nb[3] / c.ggml_type_size(k.type));
    const batch_stride_v: u32 = @truncate(v.nb[3] / c.ggml_type_size(v.type));
    const batch_stride_m: u32 = if (mask) |mk| @truncate(mk.nb[3] / c.ggml_type_size(mk.type)) else 0;
    const batch_stride_o: u32 = @truncate(dst.nb[3] / c.ggml_type_size(dst.type));
    const pc_ph1: push.FaXeOpt = .{
        .kv_seq_len = u(k.ne[1]),
        .activation_length = u(q.ne[1]),
        .q_head = u(q.ne[2]),
        .kv_head = u(k.ne[2]),
        .qk_ratio = a.qk_ratio,
        .qk_sub_groups = 1,
        .flag = @intFromBool(sinks != null),
        .nbkv_tok = a.k_stride,
        .nbkv_head = head_stride_k,
        .batch_stride_q = batch_stride_q,
        .batch_stride_k = batch_stride_k,
        .batch_stride_v = batch_stride_v,
        .batch_stride_m = batch_stride_m,
        .batch_stride_o = batch_stride_o,
        .softmax_scale = a.scale,
    };
    var pc_ph2 = pc_ph1;
    pc_ph2.nbkv_tok = a.v_stride;
    pc_ph2.nbkv_head = head_stride_v;
    const q_temp_buf = if (a.fa_copy_qstate) m.context.subbuffer(ctx, ctx.prealloc_x.?, temp_buf_offset) else a.q_buf;
    temp_buf_offset += if (a.fa_copy_qstate) x_ne * @sizeOf(c.ggml_fp16_t) else 0;
    const p_temp_buf = m.context.subbuffer(ctx, ctx.prealloc_x.?, temp_buf_offset);
    temp_buf_offset += a.size_p;
    const max_temp_buf = m.context.subbuffer(ctx, ctx.prealloc_x.?, temp_buf_offset);
    temp_buf_offset += a.size_group_max;
    const xe_native_sub_group_size: u32 = if (dev.architecture == .intel_xe1) 8 else 16;
    const aligned_gqa_ratio = upperPowerOf2(a.qk_ratio);
    const neq1: u32 = u(q.ne[1]);
    const out_per_wg_ph1 = @min(256 / xe_native_sub_group_size, neq1);
    const out_per_wg_ph2 = @min(@max(16 / aligned_gqa_ratio, 1), neq1);
    // int64_t arithmetic in the C++: neq1, nek2 and ne0 are int64_t.
    const ph1_wg: u32 = @truncate(@as(u64, @bitCast(@divTrunc(q.ne[1] + out_per_wg_ph1 - 1, out_per_wg_ph1) * k.ne[2])));
    const ph2_wg: u32 = @truncate(@as(u64, @bitCast(@divTrunc(@divTrunc(q.ne[1] + out_per_wg_ph2 - 1, out_per_wg_ph2) * dst.ne[0], out_dim_per_wg))));
    if (a.fa_copy_qstate) {
        const pc_cpy_fp16 = [5]u32{ u(q.ne[0]), u(q.ne[1]), u(q.ne[2]), u(q.ne[3]), @truncate(x_ne) };
        m.context.syncBuffers(ctx, subctx);
        m.context.requestDescriptorSets(ctx, to_fp16_vk_0.?, 1);
        m.context.dispatch(ctx, subctx, to_fp16_vk_0.?, &.{ a.q_buf, q_temp_buf }, &pc_cpy_fp16, .{ @truncate(x_ne), 1, 1 });
    }

    m.context.syncBuffers(ctx, subctx);
    m.context.requestDescriptorSets(ctx, a.pipelines.first.?, 1);
    m.context.dispatch(ctx, subctx, a.pipelines.first.?, &.{ q_temp_buf, a.k_buf, a.mask_buf, p_temp_buf, max_temp_buf }, &pc_ph1, .{ ph1_wg, u(k.ne[1]), u(q.ne[3]) });

    m.context.syncBuffers(ctx, subctx);
    m.context.requestDescriptorSets(ctx, a.pipelines.second.?, 1);
    m.context.dispatch(ctx, subctx, a.pipelines.second.?, &.{ p_temp_buf, a.v_buf, max_temp_buf, a.sinks_buf, a.dst_buf }, &pc_ph2, .{ ph2_wg, u(v.ne[2]), u(q.ne[3]) });

    ctx.prealloc_x_need_sync = true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "upperPowerOf2 is the smallest power of two at or above its input" {
    var x: u32 = 1;
    while (x <= 1 << 20) : (x += 1) {
        const p = upperPowerOf2(x);
        try std.testing.expect(std.math.isPowerOfTwo(p));
        try std.testing.expect(p >= x and (p == 1 or p / 2 < x));
    }
    try std.testing.expectEqual(@as(u32, 1 << 31), upperPowerOf2((1 << 30) + 1));
    try std.testing.expectEqual(@as(u32, 1 << 31), upperPowerOf2(1 << 31));
    // Past 2^31 the C++'s uint32_t wraps to 0, and so does the port.
    try std.testing.expectEqual(@as(u32, 0), upperPowerOf2((1 << 31) + 1));
}
