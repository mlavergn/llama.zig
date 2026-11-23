//! Flash attention: the tuning that picks a code path (scalar, coopmat1,
//! coopmat2) and its tile shape, the shared-memory checks behind it, and
//! the dispatch -- grouped-query folding, split-KV with a reduction pass,
//! the mask-skip prepass, and K/V dequantized once into fp16 scratch.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` and
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-types.h` at v1.0.0
//! (`b1b770d90`).

const std = @import("std");
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");

const vendor_id_amd = 0x1002;
const vendor_id_intel = 0x8086;
const vendor_id_nvidia = 0x10de;

/// Ports `vk_fa_tuning_params` (ggml-vulkan-types.h:1409 @b1b770d90).
pub const FaTuningParams = struct {
    path: ggml.vulkan.device.FaCodePath = .scalar,
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

/// Ports `get_xe_fa_tuning_params` (ggml-vulkan.cpp:1437 @b1b770d90): the
/// tile shape, workgroup and subgroup sizes, specialization values and
/// shared-memory footprint of the Intel Xe flash-attention kernels -- the
/// prefill kernel when `use_prefill`, the two decode phases otherwise.
pub fn xeFaTuningParams(device: *const ggml.vulkan.device.Device, pipeline_state: ggml.vulkan.device.XeFaPipelineState, use_prefill: bool) ggml.vulkan.device.XeFaTuningParams {
    var params: ggml.vulkan.device.XeFaTuningParams = .{};
    var shmem_sz: usize = 0;
    var sg_sz: u32 = 32;
    var matrix_mnk = [3]u32{ 16, 16, 16 };
    const intel_gpu = device.vendor_id == vendor_id_intel;

    if (device.architecture == .intel_xe1) {
        sg_sz = 8;
        matrix_mnk = .{ 8, 8, 16 };
    } else if (device.architecture == .intel_xe2) {
        sg_sz = 16;
        matrix_mnk = .{ 8, 16, 16 };
    }

    const tm = matrix_mnk[0];
    const tn = matrix_mnk[1];
    const tk = matrix_mnk[2];
    const hsk = pipeline_state.HSK;
    const hsv = pipeline_state.HSV;
    const qk_ratio = pipeline_state.qk_ratio;
    const pv_ratio = pipeline_state.pv_ratio;
    var sg_per_gw: u32 = undefined;
    const fp16: usize = @sizeOf(ggml.impl.ggml_c.ggml_fp16_t);
    const f32sz: usize = @sizeOf(f32);

    params.tm = tm;
    params.tn = tn;
    params.tk = tk;
    ggml.impl.assert(hsv % tm == 0, "hsv % tm == 0");

    if (use_prefill) {
        sg_per_gw = if (sg_sz == 8) 32 else 16;
        const wg_sz = sg_per_gw * sg_sz;
        const output_div: u32 = if (hsk > 256) 8 else if (hsk > 160) 4 else if (hsk > 64) 2 else 1;
        const p_per_iter: u32 = if (intel_gpu) (if (hsk > 256) 16 else 32) else 16;
        const q_in_reg: u32 = if (hsk > 512) hsk - 384 else @min(128, hsk);
        const q_in_shmem = hsk - q_in_reg;
        const wg_row = 256 / output_div;
        var gqa_fold: u32 = if (pipeline_state.multi_head_mask or qk_ratio != pv_ratio) 1 else 16;
        const max_fold_rows: u32 = if (sg_sz <= 16) 16 else 8;

        while (gqa_fold > 1 and (qk_ratio % gqa_fold != 0 or gqa_fold * output_div > max_fold_rows)) {
            gqa_fold = gqa_fold / 2;
        }

        const mask_row = wg_row / gqa_fold;
        const k_col: u32 = if (hsk % 64 == 0 and hsk <= 512 and wg_sz <= 256) 8 else 4;
        const k_loops = ggml.vulkan.ops.ceilDiv(hsk * p_per_iter, k_col * wg_sz);

        const k_in_shmem = hsk <= 576;
        const v_in_shmem = hsk <= 128 and hsv <= 128;
        const emf_in_shmem = wg_row / matrix_mnk[1] != sg_per_gw;
        const mask_in_shmem = mask_row <= 64;
        const shmem_q_sz: usize = @as(usize, q_in_shmem) * wg_row * fp16;
        const shmem_oob_staging_sz: usize = @as(usize, sg_per_gw) * tm * tk * fp16;
        const shmem_mask_sz: usize = if (mask_in_shmem) @as(usize, mask_row) * p_per_iter * fp16 else 0;
        const shmem_k_sz: usize = if (k_in_shmem) @as(usize, k_loops) * k_col * wg_sz * fp16 else shmem_oob_staging_sz;
        const shmem_v_sz: usize = if (v_in_shmem) @as(usize, hsv) * p_per_iter * fp16 else 0;
        const shmem_p_sz: usize = @as(usize, wg_row) * p_per_iter * f32sz;
        const shmem_emf_sz: usize = if (emf_in_shmem) @as(usize, wg_row) * f32sz else 0;
        const shmem_pv_sz: usize = if (intel_gpu) 0 else @as(usize, sg_per_gw) * tm * tn * f32sz;

        // Reuse section sits at tail of shmem, reused with k + mask/kv oob staging/pv staging
        var shmem_reuse_sz = shmem_k_sz + shmem_mask_sz;
        // When v is not in shmem, its staging reuses the k + mask region; size the reuse section to the larger of the two
        shmem_reuse_sz = if (v_in_shmem) shmem_reuse_sz else @max(shmem_reuse_sz, shmem_oob_staging_sz);
        shmem_reuse_sz = @max(shmem_pv_sz, shmem_reuse_sz);

        // Q shmem size check for prologue staging is safe to bypass:
        // 1. Prologue q staging shmem is released right after prologue, so it never overlaps other shmem sections
        // 2. Extreme large hsk one-time q staging is capped within shader (max_q_reg_staging = shmem_size / 2 / Wr)
        shmem_sz = shmem_q_sz + shmem_reuse_sz + shmem_v_sz + shmem_p_sz + shmem_emf_sz;

        var sg_col = hsv / tm;
        var pipeline_flag: u32 = 0;
        sg_col = (sg_col + output_div - 1) / output_div;
        sg_col = sg_col * tm;
        if (pipeline_state.unaligned_seq_len) pipeline_flag |= 1 << 0;
        if (pipeline_state.unaligned_q) pipeline_flag |= 1 << 1;
        if (pipeline_state.partial_row_mask) pipeline_flag |= 1 << 2;
        if (intel_gpu) pipeline_flag |= 1 << 3;

        params.workgroup_size = .{ wg_sz, 0 };
        params.subgroup_size = .{ sg_sz, 0 };
        params.shmem_sz = shmem_sz;
        params.prefill_params.pipeline_flag = pipeline_flag;
        params.prefill_params.wg_row = wg_row;
        params.prefill_params.gqa_fold = gqa_fold;
        params.prefill_params.q_in_reg = q_in_reg;
        params.prefill_params.p_per_iter = p_per_iter;
        params.prefill_params.sg_col = sg_col;
        params.prefill_params.acc_count = if (tm == 8) (if (sg_col % 16 == 0) 2 else 1) else 1;
    } else {
        // Phase 1 shmem size calculation
        const p_chunk_ph1: u32 = 32;
        // Phase 1 k shmem staging for extreme large hsk is capped within shader, bypass that check
        const pmat_row = ggml.vulkan.ops.ceilDiv(qk_ratio, tn);
        const aligned_q_len = upperPowerOf2(pipeline_state.decode_n);
        const group_sz_ph1 = @min(@max(aligned_q_len *% sg_sz, 64), 256);
        const out_per_wg_ph1 = @min(pipeline_state.decode_n, 256 / sg_sz);
        const shmem_k_sz: usize = @as(usize, group_sz_ph1 / sg_sz) * tm * tk * fp16;
        const shmem_mask_sz: usize = @as(usize, out_per_wg_ph1) * p_chunk_ph1 * f32sz;
        var shmem_p_sz: usize = @as(usize, p_chunk_ph1) * pmat_row * tn * f32sz;
        // K shmem is reused with mask/p, calculate the max size
        const shmem_ph1_sz = @max(shmem_k_sz, shmem_mask_sz + shmem_p_sz);

        // Phase 2 shmem size calculation
        const aligned_pv_ratio = upperPowerOf2(pv_ratio);
        const split_p_per_iter_ph2: u32 = if (pv_ratio <= 32) 256 else 128;
        const out_col_per_wg_ph2: u32 = if (intel_gpu) (if (pv_ratio > 16) 8 else 16) else 16;
        const split_p_per_warp: u32 = 16;
        const group_sz_ph2 = (split_p_per_iter_ph2 / split_p_per_warp) * sg_sz;
        sg_per_gw = group_sz_ph2 / sg_sz;
        const out_row_per_wg_ph2 = @min(@max(16 / aligned_pv_ratio, 1), pipeline_state.decode_n);
        const o_row = pv_ratio * out_row_per_wg_ph2;
        const omat_row = ggml.vulkan.ops.ceilDiv(o_row, tn) * tn;
        const global_max_loops = ggml.vulkan.ops.ceilDiv(o_row, sg_per_gw);
        const shmem_global_max_sz: usize = @as(usize, ggml.vulkan.ops.ceilDiv(global_max_loops * sg_per_gw, sg_sz)) * sg_sz * f32sz;

        const size_max = o_row * split_p_per_iter_ph2 / p_chunk_ph1;
        const shmem_group_max_sz: usize = @as(usize, ggml.vulkan.ops.ceilDiv(size_max, group_sz_ph2)) * group_sz_ph2 * f32sz;

        const p_loop_col = group_sz_ph2 * 4;
        const p_loops = ggml.vulkan.ops.ceilDiv(o_row * split_p_per_iter_ph2, p_loop_col);
        shmem_p_sz = @max(@as(usize, p_loop_col) * p_loops * fp16, @as(usize, omat_row) * split_p_per_iter_ph2 * fp16);

        const shmem_lf_sz: usize = @as(usize, ggml.vulkan.ops.ceilDiv(sg_per_gw * p_loops, sg_sz)) * sg_sz * f32sz;
        const shmem_oob_staging_sz: usize = @as(usize, sg_per_gw) * tm * tk * fp16;
        const shmem_o_sz: usize = @as(usize, sg_per_gw) * out_col_per_wg_ph2 * omat_row * f32sz;
        const shmem_reuse_sz = 2 * shmem_group_max_sz + @max(shmem_oob_staging_sz, 2 * shmem_p_sz);

        var shmem_ph2_sz = shmem_lf_sz + shmem_global_max_sz;
        // Shmem of p/group max is reused with output accumulation, calculate the max size
        shmem_ph2_sz += @max(shmem_reuse_sz, shmem_o_sz);
        shmem_sz = @max(shmem_ph1_sz, shmem_ph2_sz);

        params.workgroup_size = .{ group_sz_ph1, group_sz_ph2 };
        params.subgroup_size = .{ sg_sz, sg_sz };
        params.shmem_sz = shmem_sz;
        params.decode_params.p_chunk_ph1 = p_chunk_ph1;
        params.decode_params.out_per_wg_ph1 = out_per_wg_ph1;
        params.decode_params.split_p_per_iter_ph2 = split_p_per_iter_ph2;
        params.decode_params.out_row_per_wg_ph2 = out_row_per_wg_ph2;
        params.decode_params.out_col_per_wg_ph2 = out_col_per_wg_ph2;
        params.decode_params.pipeline_flag_ph2 = if (pipeline_state.unaligned_seq_len) 1 << 0 else 0;
    }

    // The C++ reports a footprint past maxComputeSharedMemorySize through
    // `VK_LOG_DEBUG`, which is empty without `GGML_VULKAN_DEBUG`.
    return params;
}

/// `D ^ (D & (D-1))`: the lowest set bit.
fn lowestBit(d: u32) u32 {
    return d ^ (d & (d -% 1));
}

/// Ports `get_fa_tuning_params_scalar` (ggml-vulkan.cpp:1209 @b1b770d90).
fn tuningScalar(dev: *const ggml.vulkan.device.Device, hsk: u32, hsv: u32, n_rows: u32, n_kv: u32, k_type: ggml.impl.ggml_c.ggml_type, v_type: ggml.impl.ggml_c.ggml_type, f32acc: bool) FaTuningParams {
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

    // With head sizes above 128, 16 rows per workgroup spill registers on
    // Qualcomm and run about 40x slower; at large KV a single dispatch can
    // then exceed the Windows GPU timeout.
    if (dev.vendor_id == ggml.vulkan.common.vendor_id.qualcomm and (hsk > 128 or hsv > 128)) {
        r.block_rows = @min(r.block_rows, 8);
    }

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

/// Ports `get_fa_tuning_params_coopmat1` (ggml-vulkan.cpp:1290 @b1b770d90).
fn tuningCoopmat1(dev: *const ggml.vulkan.device.Device, hsk: u32, hsv: u32) FaTuningParams {
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

/// Ports `get_fa_tuning_params_coopmat2` (ggml-vulkan.cpp:1321 @b1b770d90).
fn tuningCoopmat2(dev: *const ggml.vulkan.device.Device, hsk: u32, hsv: u32, n_rows: u32, k_type: ggml.impl.ggml_c.ggml_type, v_type: ggml.impl.ggml_c.ggml_type) FaTuningParams {
    var r: FaTuningParams = .{ .path = .coopmat2 };
    const D = hsk | hsv;
    const small_rows = n_rows < 32;
    if (small_rows) {
        r.block_rows = 32;
        r.block_cols = 32;
    } else if (ggml.types.ggml_is_quantized(k_type) or ggml.types.ggml_is_quantized(v_type) or hsk >= 256 or hsv >= 256) {
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

/// Ports `get_fa_tuning_params` (ggml-vulkan.cpp:1349 @b1b770d90).
pub fn tuningParams(dev: *const ggml.vulkan.device.Device, hsk: u32, hsv: u32, n_rows: u32, n_kv: u32, k_type: ggml.impl.ggml_c.ggml_type, v_type: ggml.impl.ggml_c.ggml_type, f32acc: bool) FaTuningParams {
    var path: ggml.vulkan.device.FaCodePath = if (dev.coopmat2) .coopmat2 else if (dev.coopmat1_fa_support) .coopmat1 else .scalar;

    if (path == .coopmat2 and k_type == ggml.impl.ggml_c.GGML_TYPE_BF16 and !dev.coopmat2_bf16_support) path = .coopmat1;
    if (path == .coopmat1 and k_type == ggml.impl.ggml_c.GGML_TYPE_BF16 and !dev.coopmat_bf16_support) path = .scalar;
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

/// Ports `get_fa_pipeline_state` (ggml-vulkan.cpp:1393 @b1b770d90).
pub fn pipelineState(dev: *const ggml.vulkan.device.Device, params: *const FaTuningParams, hsk: u32, hsv: u32, aligned: bool, f32acc: bool, use_mask: bool, use_mask_opt: bool, use_logit_softcap: bool, use_sparse: bool, k_type: ggml.impl.ggml_c.ggml_type, v_type: ggml.impl.ggml_c.ggml_type) ggml.vulkan.device.FaPipelineState {
    const old_amd_windows = dev.vendor_id == vendor_id_amd and dev.driver_id == ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_AMD_PROPRIETARY and
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

/// Ports `ggml_vk_flash_attn_scalar_shmem_support` (ggml-vulkan.cpp:8313 @b1b770d90).
/// Must track the shader; it overestimates `tmpsh` slightly.
pub fn scalarShmemSupport(dev: *const ggml.vulkan.device.Device, params: *const FaTuningParams, hsk: u32, hsv: u32, f32acc: bool, k_type: ggml.impl.ggml_c.ggml_type, v_type: ggml.impl.ggml_c.ggml_type) bool {
    _ = f32acc;
    const wg_size = params.workgroup_size;
    const Br = params.block_rows;
    const Bc = params.block_cols;

    // BF16 uses the fp32 shader (FLOAT_TYPE=float)
    const float_type_size: u32 = if (dev.fp16 and k_type != ggml.impl.ggml_c.GGML_TYPE_BF16) @sizeOf(ggml.impl.ggml_c.ggml_fp16_t) else @sizeOf(f32);
    const mmq = ggml.vulkan.shaders.faScalarUsesMmq(dev, k_type, v_type);

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

/// Ports `ggml_vk_flash_attn_coopmat_shmem_support` (ggml-vulkan.cpp:8364 @b1b770d90).
pub fn coopmatShmemSupport(dev: *const ggml.vulkan.device.Device, params: *const FaTuningParams, hsk: u32, hsv: u32, f32acc: bool, k_type: ggml.impl.ggml_c.ggml_type) bool {
    const Br = params.block_rows;
    const Bc = params.block_cols;
    const MatBr: u32 = 16;
    const MatBc: u32 = 16;
    const row_split = Bc / MatBc;
    const hsk_pad: u32 = @intCast(ggml.vulkan.ops_matmul.roundupPow2(hsk, 16));
    const hsv_pad: u32 = @intCast(ggml.vulkan.ops_matmul.roundupPow2(hsv, 16));
    const acctype: u32 = if (f32acc) 4 else 2;
    const f16vec4: u32 = 8;

    const tmpsh = (Bc / MatBc) * @sizeOf(f32);
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its table is always allocated.
    const iq_shmem = 16 * @sizeOf(ggml.impl.ggml_c.ggml_fp16_t);

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
    const pvsh_elem_size: u32 = if (k_type == ggml.impl.ggml_c.GGML_TYPE_BF16) 16 else f16vec4;
    const osh_stride = params.row_split * MatBr / 4;
    const pvsh = MatBc * osh_stride * pvsh_elem_size;
    const slope = Br * acctype;

    const total_size = tmpsh + iq_shmem + Qf + Psh + sfsh + ksh + pvsh + slope;
    return total_size <= dev.properties.limits.maxComputeSharedMemorySize;
}

/// The `std::map` lookup the C++ does under `compile_mutex`: the pipeline
/// for `key`, created empty on first sight for the compile walk to fill.
fn dynamicPipeline(ctx: *ggml.vulkan.context.BackendCtx, map: anytype, key: anytype) *ggml.vulkan.pipeline.Pipeline {
    const dev = ctx.device;
    dev.compile_mutex.lock();
    defer dev.compile_mutex.unlock();
    const gop = map.getOrPut(std.heap.c_allocator, key) catch ggml.impl.abort("ggml_vulkan: out of memory");
    if (!gop.found_existing or gop.value_ptr.* == null) gop.value_ptr.* = ggml.vulkan.pipeline.Pipeline.create();
    return gop.value_ptr.*.?;
}

/// Whether a K or V tensor is a dense cache: rows contiguous, heads then
/// cells packed with no gaps, and streams too unless there is only one.
fn isDenseKvCache(t: *const ggml.impl.ggml_c.ggml_tensor) bool {
    return t.nb[0] == ggml.types.ggml_type_size(t.type) and
        t.nb[2] == ggml.types.ggml_row_size(t.type, t.ne[0]) and
        t.nb[1] == t.nb[2] * @as(usize, @intCast(t.ne[2])) and
        (t.ne[3] == 1 or t.nb[3] == t.nb[1] * @as(usize, @intCast(t.ne[1])));
}

/// Ports `ggml_vk_flash_attn` (ggml-vulkan.cpp:8412 @b1b770d90).
pub fn flashAttn(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, q: *const ggml.impl.ggml_c.ggml_tensor, k: *const ggml.impl.ggml_c.ggml_tensor, v: *const ggml.impl.ggml_c.ggml_tensor, mask: ?*const ggml.impl.ggml_c.ggml_tensor, sinks: ?*const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const dev = ctx.device;
    const limits = &dev.properties.limits;
    const u = ggml.vulkan.ops.u32Of;

    const nem0: u32 = if (mask) |t| u(t.ne[0]) else 0;
    const nem1: u32 = if (mask) |t| u(t.ne[1]) else 0;
    const nem2: u32 = if (mask) |t| u(t.ne[2]) else 0;
    const nem3: u32 = if (mask) |t| u(t.ne[3]) else 0;

    const HSK = u(k.ne[0]);
    const HSV = u(v.ne[0]);
    var N = u(q.ne[1]);
    const KV = u(k.ne[1]);

    ggml.impl.assert(dst.ne[0] == HSV, "ne0 == HSV");
    ggml.impl.assert(dst.ne[2] == N, "ne2 == N");
    // input tensor rows must be contiguous
    ggml.impl.assert(q.nb[0] == ggml.types.ggml_type_size(q.type), "nbq0 == ggml_type_size(q->type)");
    ggml.impl.assert(k.nb[0] == ggml.types.ggml_type_size(k.type), "nbk0 == ggml_type_size(k->type)");
    ggml.impl.assert(v.nb[0] == ggml.types.ggml_type_size(v.type), "nbv0 == ggml_type_size(v->type)");
    ggml.impl.assert(q.ne[0] == HSK, "neq0 == HSK");
    ggml.impl.assert(q.ne[1] == N, "neq1 == N");
    ggml.impl.assert(v.ne[1] == k.ne[1], "nev1 == nek1");
    // dst cannot be transposed or permuted
    ggml.impl.assert(dst.nb[0] == @sizeOf(f32), "nb0 == sizeof(float)");
    ggml.impl.assert(dst.nb[0] <= dst.nb[1], "nb0 <= nb1");
    ggml.impl.assert(dst.nb[1] <= dst.nb[2], "nb1 <= nb2");
    ggml.impl.assert(dst.nb[2] <= dst.nb[3], "nb2 <= nb3");

    var gqa_ratio: u32 = 1;
    const qk_ratio = u(@divTrunc(q.ne[2], k.ne[2]));
    var workgroups_x = u(q.ne[1]);
    var workgroups_y = u(q.ne[2]);
    const workgroups_z = u(q.ne[3]);

    const f32acc = !dev.fp16 or !ggml.vulkan.ops_matmul.precAllowsF16acc(ggml.ops.ggml_prec_get_acc(dst)) or k.type == ggml.impl.ggml_c.GGML_TYPE_BF16;

    // Dequantize K/V once into an f16 scratch, reordered so FA can read without a stride.
    const isF = struct {
        fn f(t: ggml.impl.ggml_c.ggml_type) bool {
            return t == ggml.impl.ggml_c.GGML_TYPE_F16 or t == ggml.impl.ggml_c.GGML_TYPE_BF16 or t == ggml.impl.ggml_c.GGML_TYPE_F32;
        }
    }.f;
    const k_quant = !isF(k.type);
    const v_quant = !isF(v.type);
    const fp16sz: u64 = @sizeOf(ggml.impl.ggml_c.ggml_fp16_t);
    const kv_raw = (k.type == ggml.impl.ggml_c.GGML_TYPE_F16 or k.type == ggml.impl.ggml_c.GGML_TYPE_BF16) and k.type == v.type;
    // gated to AMD, except RDNA4, which regressed in benchmarks
    const kv_raw_strided = dev.vendor_id == vendor_id_amd and
        dev.architecture != .amd_rdna4 and
        q.ne[1] >= 256 and
        kv_raw and
        (k.nb[1] != @as(u64, HSK) * ggml.types.ggml_type_size(k.type) or
            v.nb[1] != @as(u64, HSV) * ggml.types.ggml_type_size(v.type)) and
        (HSK % 8) == 0 and (HSV % 8) == 0;
    const kv_scratch_ts: u64 = if (kv_raw_strided) ggml.types.ggml_type_size(k.type) else fp16sz;
    // K and V share one scratch buffer, so the sum must fit in one allocation
    const dequant_kv_sz: u64 = (@as(u64, @intCast(ggml.types.ggml_nelements(k))) + @as(u64, @intCast(ggml.types.ggml_nelements(v)))) *% kv_scratch_ts;
    const cm2_min_queries: u32 = if (KV >= 8192) 512 else 1024;
    const cm2_min_reuse: u32 = if (KV >= 8192) 4096 else 8192;
    // Query reuse must cover the conversion and synchronization cost.
    const cm2_dequant_kv = KV >= 4096 and q.ne[1] >= cm2_min_queries and @as(u64, @bitCast(q.ne[1])) *% qk_ratio >= cm2_min_reuse;
    const use_dequant_kv = ((k_quant and v_quant) or kv_raw_strided) and q.ne[1] >= 64 and
        isDenseKvCache(k) and isDenseKvCache(v) and
        @as(u64, @intCast(ggml.types.ggml_nelements(k))) * kv_scratch_ts <= limits.maxStorageBufferRange and
        @as(u64, @intCast(ggml.types.ggml_nelements(v))) * kv_scratch_ts <= limits.maxStorageBufferRange and
        dequant_kv_sz <= dev.max_buffer_size and
        dequant_kv_sz <= dev.max_memory_allocation_size and
        (kv_raw_strided or
            (dev.pipeline_dequant_transpose[@intCast(k.type)] != null and
                dev.pipeline_dequant_transpose[@intCast(v.type)] != null)) and
        // Intel Xe1 regresses, see PR 25494
        (dev.vendor_id != vendor_id_intel or (dev.coopmat_support and dev.architecture != .intel_xe1)) and
        (!dev.coopmat2 or cm2_dequant_kv);

    const k_type_eff: ggml.impl.ggml_c.ggml_type = if (use_dequant_kv and !kv_raw_strided) ggml.impl.ggml_c.GGML_TYPE_F16 else k.type;
    const v_type_eff: ggml.impl.ggml_c.ggml_type = if (use_dequant_kv and !kv_raw_strided) ggml.impl.ggml_c.GGML_TYPE_F16 else v.type;

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
    const kv_f16 = k_type_eff == ggml.impl.ggml_c.GGML_TYPE_F16 and v_type_eff == ggml.impl.ggml_c.GGML_TYPE_F16;
    // cm2 dense is fast, so it needs a larger reduction to win.
    // With quantized K/V, sparse only breaks even around 16x (measured on RDNA3/RDNA4).
    const min_ratio: i64 = if (tuning.path == .coopmat2) 4 else if (kv_f16) 2 else 16;
    // coopmat2 vector decode requires 8B strides.
    const sparseGatherAligned = struct {
        fn f(t: *const ggml.impl.ggml_c.ggml_tensor) bool {
            return (t.type != ggml.impl.ggml_c.GGML_TYPE_F16 and t.type != ggml.impl.ggml_c.GGML_TYPE_BF16) or
                (t.nb[1] | t.nb[2] | t.nb[3]) % (4 * @sizeOf(ggml.impl.ggml_c.ggml_fp16_t)) == 0;
        }
    }.f;
    const use_sparse = !disable_sparse and n_kv_max > 0 and mask != null and
        max_bias == 0.0 and logit_softcap == 0.0 and
        (tuning.path != .coopmat2 or (sparseGatherAligned(k) and sparseGatherAligned(v))) and
        nem0 == KV and
        @as(i64, KV) >= @max(4096, min_ratio * @as(i64, n_kv_max)) and
        (gqa_ratio > 1 or (tuning.path == .scalar and N == 1));

    const q_stride: u32 = @truncate(q.nb[1] / ggml.types.ggml_type_size(q.type));
    const q_stride_eff: u32 = if (gqa_ratio > 1) @truncate(q.nb[2] / ggml.types.ggml_type_size(q.type)) else q_stride;
    var k_stride: u32 = @truncate(k.nb[1] / ggml.types.ggml_type_size(k.type));
    var v_stride: u32 = @truncate(v.nb[1] / ggml.types.ggml_type_size(v.type));
    // For F32, the shader treats it as a block of size 4 (for vec4 loads)
    if (k.type == ggml.impl.ggml_c.GGML_TYPE_F32) k_stride /= 4;
    if (v.type == ggml.impl.ggml_c.GGML_TYPE_F32) v_stride /= 4;

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
    // CM2 decoded K/V strides count whole blocks and do not need to be multiples of eight blocks.
    const cm2_decode_k = tuning.path == .coopmat2 and
        k_type_eff != ggml.impl.ggml_c.GGML_TYPE_F16 and k_type_eff != ggml.impl.ggml_c.GGML_TYPE_BF16;
    const cm2_decode_v = tuning.path == .coopmat2 and
        v_type_eff != ggml.impl.ggml_c.GGML_TYPE_F16 and v_type_eff != ggml.impl.ggml_c.GGML_TYPE_BF16;
    var aligned = (KV % alignment) == 0 and (q_stride_eff & 7) == 0 and
        (cm2_decode_k or (k_stride & 7) == 0) and (cm2_decode_v or (v_stride & 7) == 0);
    // coopmat2 needs the variant that clamps loads when HSK/HSV aren't sufficiently aligned.
    if (((HSK | HSV) % 16) != 0 and tuning.path == .coopmat2) aligned = false;

    // Only use mask opt when the mask is fairly large. Not tuned extensively.
    var use_mask_opt = mask != null and !use_sparse and nem1 >= 32 and nem0 *% nem1 > 32768 and nem0 >= tuning.block_cols * 16 and
        (dev.architecture != .amd_gcn or HSK > 256 or HSV > 256);

    const state = pipelineState(dev, &tuning, HSK, HSV, aligned, f32acc, mask != null, use_mask_opt, logit_softcap != 0, use_sparse, k_type_eff, v_type_eff);

    // Intel Xe's flash attention: a prefill kernel, and a two-phase decode.
    const tensorAlignedWith = struct {
        fn f(bctx: *const ggml.vulkan.context.BackendCtx, t: *const ggml.impl.ggml_c.ggml_tensor, in: u32) bool {
            ggml.impl.assert(in != 0, "t != nullptr && in != 0");
            return t.nb[1] % in == 0 and t.nb[2] % in == 0 and t.nb[3] % in == 0 and ggml.vulkan.context.tensorBufferOffset(bctx, t) % in == 0;
        }
    }.f;
    var xe_fa_opt = false;
    var xe_fa_use_2phases = false;
    const xe_fa_supported_platform = dev.xe_fa_support and dev.properties.deviceID != 0xFD80 and dev.properties.deviceID != 0xFD81;
    const xe_fa_supported_usage = @mod(q.ne[0], 32) == 0 and @mod(v.ne[0], 16) == 0 and mask != null and !use_sparse and max_bias == 0.0 and logit_softcap == 0.0 and
        tensorAlignedWith(ctx, k, 16) and tensorAlignedWith(ctx, v, 16);
    const xe_fa_supported_dtype = q.type == ggml.impl.ggml_c.GGML_TYPE_F32 and k.type == ggml.impl.ggml_c.GGML_TYPE_F16 and v.type == ggml.impl.ggml_c.GGML_TYPE_F16 and
        mask != null and mask.?.type == ggml.impl.ggml_c.GGML_TYPE_F16;
    xe_fa_opt = xe_fa_supported_platform and xe_fa_supported_usage and xe_fa_supported_dtype;
    var xe_fa_pipeline_dual_phases: ggml.vulkan.device.XeFaDecodePipelines = .{};
    var xe_fa_pipeline: ?*ggml.vulkan.pipeline.Pipeline = null;
    var size_q_staging: u64 = 0;
    var size_p: u64 = 0;
    var size_group_max: u64 = 0;
    var size_decode_temp: u64 = 0;
    const pv_ratio: u32 = @truncate(@as(u64, @bitCast(@divTrunc(q.ne[2], v.ne[2]))));
    var xe_fa_params: ggml.vulkan.device.XeFaTuningParams = .{};

    var pipeline_: ?*ggml.vulkan.pipeline.Pipeline = null;
    {
        dev.compile_mutex.lock();
        defer dev.compile_mutex.unlock();
        if (xe_fa_opt) {
            const neq1: i64 = q.ne[1];
            const nek1: i64 = k.ne[1];
            if (neq1 > 8 or pv_ratio > 64) {
                const head_stride_q: u32 = @truncate(q.nb[2] / ggml.types.ggml_type_size(q.type));
                const batch_stride_q: u32 = @truncate(q.nb[3] / ggml.types.ggml_type_size(q.type));
                var prefill_state: ggml.vulkan.device.XeFaPipelineState = .{ .HSK = u(q.ne[0]), .HSV = u(v.ne[0]), .qk_ratio = qk_ratio, .pv_ratio = pv_ratio, .decode_n = 0 };
                prefill_state.unaligned_seq_len = @rem(nek1, 32) != 0;
                prefill_state.unaligned_q = head_stride_q % 8 != 0 or q_stride % 8 != 0 or batch_stride_q % 8 != 0;
                prefill_state.partial_row_mask = neq1 > nem1;
                prefill_state.multi_head_mask = nem2 > 1;
                xe_fa_params = xeFaTuningParams(dev, prefill_state, true);
                if (xe_fa_params.shmem_sz <= limits.maxComputeSharedMemorySize) {
                    const gop = dev.pipeline_xe_fa_prefill.getOrPut(std.heap.c_allocator, prefill_state) catch ggml.impl.abort("ggml_vulkan: out of memory");
                    if (!gop.found_existing) gop.value_ptr.* = ggml.vulkan.pipeline.Pipeline.create();
                    xe_fa_pipeline = gop.value_ptr.*;
                } else {
                    xe_fa_opt = false;
                }
            } else {
                var decode_state: ggml.vulkan.device.XeFaPipelineState = .{ .HSK = u(q.ne[0]), .HSV = u(v.ne[0]), .qk_ratio = qk_ratio, .pv_ratio = pv_ratio, .decode_n = @truncate(@as(u64, @bitCast(neq1))) };
                decode_state.unaligned_seq_len = @rem(nek1, 256) != 0;
                xe_fa_params = xeFaTuningParams(dev, decode_state, false);
                const split_p_size = xe_fa_params.decode_params.p_chunk_ph1;
                const max_dim: u64 = @max(@as(u64, @bitCast(@divTrunc(nek1 + split_p_size - 1, split_p_size))), 1);
                const p_dim = max_dim *% split_p_size;
                const q_head_padding: u64 = 16;
                const neq0u: u64 = @bitCast(q.ne[0]);
                const neq1u: u64 = @bitCast(q.ne[1]);
                const neq2u: u64 = @bitCast(q.ne[2]);
                const neq3u: u64 = @bitCast(q.ne[3]);
                size_q_staging = neq0u *% neq1u *% (neq2u *% neq3u +% q_head_padding) *% @sizeOf(ggml.impl.ggml_c.ggml_fp16_t);
                size_p = neq1u *% neq2u *% p_dim *% neq3u *% @sizeOf(ggml.impl.ggml_c.ggml_fp16_t);
                size_group_max = neq1u *% neq2u *% max_dim *% neq3u *% @sizeOf(f32);
                size_decode_temp = size_q_staging +% size_p +% size_group_max;
                const ph1_wg1: u32 = @truncate(@as(u64, @bitCast(@divTrunc(nek1 + xe_fa_params.decode_params.p_chunk_ph1 - 1, xe_fa_params.decode_params.p_chunk_ph1))));
                if (xe_fa_params.shmem_sz <= limits.maxComputeSharedMemorySize and
                    ph1_wg1 <= limits.maxComputeWorkGroupCount[1] and
                    size_decode_temp <= limits.maxStorageBufferRange)
                {
                    const gop = dev.pipeline_xe_fa_decode_dual_phases.getOrPut(std.heap.c_allocator, decode_state) catch ggml.impl.abort("ggml_vulkan: out of memory");
                    if (!gop.found_existing) gop.value_ptr.* = .{ .first = ggml.vulkan.pipeline.Pipeline.create(), .second = ggml.vulkan.pipeline.Pipeline.create() };
                    xe_fa_pipeline_dual_phases = gop.value_ptr.*;
                    xe_fa_use_2phases = true;
                } else {
                    xe_fa_opt = false;
                }
            }
        }

        if (!xe_fa_opt) {
            const gop = dev.pipeline_flash_attn_f32_f16.getOrPut(std.heap.c_allocator, state) catch ggml.impl.abort("ggml_vulkan: out of memory");
            if (!gop.found_existing or gop.value_ptr.* == null) gop.value_ptr.* = ggml.vulkan.pipeline.Pipeline.create();
            pipeline_ = gop.value_ptr.*;
        }
    }

    if (!xe_fa_opt) {
        // Compile early to initialize wg_denoms.
        ggml.vulkan.context.requestDescriptorSets(ctx, pipeline_.?, 1);
    }

    var split_kv = KV;
    var split_k: u32 = 1;

    // Intel Alchemist prefers more workgroups
    const multiplier: u32 = if (dev.vendor_id == vendor_id_intel and dev.architecture != .intel_xe2) 2 else 1;
    // A placeholder core count when none is known; split_k is a big help for perf.
    const shader_core_count: u32 = if (dev.shader_core_count != 0) dev.shader_core_count * multiplier else 16;

    const Br = state.Br;
    const Bc = state.Bc;
    if (!xe_fa_opt) {
        ggml.impl.assert(Br == pipeline_.?.wg_denoms[0], "Br == pipeline->wg_denoms[0]");
    }
    const Tr = ggml.vulkan.ops.ceilDiv(N, Br);

    // Try to use split_k when KV is large enough to be worth the overhead.
    // Sparse: split_kv carries n_kv_max, split_k partitions its blocks for
    // occupancy.
    if (use_sparse) {
        split_kv = @intCast(n_kv_max);
        const total_blocks = ggml.vulkan.ops.ceilDiv(@as(u32, @intCast(n_kv_max)), Bc);
        const base_wgs = (if (gqa_ratio > 1) workgroups_x else Tr) *% workgroups_y *% workgroups_z;
        if (base_wgs < shader_core_count * 2) split_k = shader_core_count * 2 / base_wgs;
        split_k = @max(1, @min(split_k, total_blocks));
        // Match the shader's per-split block count so no split is empty.
        const per_blocks = ggml.vulkan.ops.ceilDiv(total_blocks, split_k);
        split_k = ggml.vulkan.ops.ceilDiv(total_blocks, per_blocks);
    } else if (gqa_ratio > 1 and workgroups_x <= Br) {
        split_k = shader_core_count * 2 / (workgroups_x *% workgroups_y *% workgroups_z);
    } else if (gqa_ratio <= 1) {
        const total_wgs_no_split = Tr *% workgroups_y *% workgroups_z;
        if (total_wgs_no_split < shader_core_count * 2) split_k = shader_core_count * 2 / total_wgs_no_split;
    }

    if (!use_sparse and split_k > 1) {
        // Split KV evenly into split_k chunks, each a multiple of the
        // alignment, and recompute split_k from that.
        split_kv = @intCast(ggml.vulkan.ops_matmul.roundupPow2(@max(1, KV / split_k), alignment));
        split_k = ggml.vulkan.ops.ceilDiv(KV, split_kv);
    }

    if (xe_fa_opt) {
        // Bypass mask_opt and split_k buffer grow when they are not in use
        use_mask_opt = false;
        split_k = 1;
        if (xe_fa_use_2phases) {
            if (ctx.prealloc_size_x < size_decode_temp) {
                ctx.prealloc_size_x = size_decode_temp;
                ggml.vulkan.nodes.preallocateBuffers(ctx, subctx);
            }

            if (ctx.prealloc_x_need_sync) ggml.vulkan.context.syncBuffers(ctx, subctx);
        }
    }

    const ne1: u64 = @intCast(dst.ne[1]);
    const ne2: u64 = @intCast(dst.ne[2]);
    const ne3: u64 = @intCast(dst.ne[3]);

    // Split-k temporaries: per split and batch, the O matrix (D x ne1),
    // then the per-row m and L values -- matrices first, ordered
    // [HSV, ne1, k, ne2, ne3], then the rows, [ne1, k, ne2, ne3].
    const split_k_size: u64 = if (split_k > 1) (@as(u64, HSV) * ne1 * @sizeOf(f32) + ne1 * @sizeOf(f32) * 2) * split_k * ne2 * ne3 else 0;
    if (split_k_size > limits.maxStorageBufferRange) ggml.impl.abort("Requested preallocation size is too large");
    if (ctx.prealloc_size_split_k < split_k_size) {
        ctx.prealloc_size_split_k = split_k_size;
        ggml.vulkan.nodes.preallocateBuffers(ctx, subctx);
    }

    const mask_opt_num_dwords = ggml.vulkan.ops.ceilDiv(nem0, 16 * Bc);
    const mask_opt_size: u64 = @sizeOf(u32) * @as(u64, mask_opt_num_dwords) * ggml.vulkan.ops.ceilDiv(nem1, Br) * nem2 * nem3;

    var pipeline_fa_mask_opt: ?*ggml.vulkan.pipeline.Pipeline = null;
    if (use_mask_opt) {
        pipeline_fa_mask_opt = dynamicPipeline(ctx, &dev.pipeline_fa_mask_opt, .{ Br, Bc });
        ggml.vulkan.context.requestDescriptorSets(ctx, pipeline_fa_mask_opt.?, 1);
        if (ctx.prealloc_size_y < mask_opt_size) {
            ctx.prealloc_size_y = mask_opt_size;
            ggml.vulkan.nodes.preallocateBuffers(ctx, subctx);
        }
        if (ctx.prealloc_y_need_sync) ggml.vulkan.context.syncBuffers(ctx, subctx);
    }

    // Sparse index scratch reuses prealloc_y (mutually exclusive with mask opt).
    const sparse_idx_size: u64 = if (use_sparse) @sizeOf(i32) *% @as(u64, @intCast(n_kv_max)) *% nem1 *% nem2 *% nem3 else 0;
    const sparse_compact_pipeline = if (dev.fa_sparse_compact_use_subgroups) dev.pipeline_fa_sparse_compact_subgroup else dev.pipeline_fa_sparse_compact;
    if (use_sparse) {
        ggml.vulkan.context.requestDescriptorSets(ctx, sparse_compact_pipeline.?, 1);
        if (ctx.prealloc_size_y < sparse_idx_size) {
            ctx.prealloc_size_y = sparse_idx_size;
            ggml.vulkan.nodes.preallocateBuffers(ctx, subctx);
        }
        if (ctx.prealloc_y_need_sync) ggml.vulkan.context.syncBuffers(ctx, subctx);
    }

    const n_head_kv = u(q.ne[2]);
    const n_head_log2: u32 = @as(u32, 1) << @intCast(@as(u32, @intFromFloat(@floor(@as(f32, stdlib.libm.log2fInt(n_head_kv))))));
    const nhl: f32 = @floatFromInt(n_head_log2);
    const m0 = stdlib.libm.powf(2.0, -max_bias / nhl);
    const m1 = stdlib.libm.powf(2.0, -(max_bias / 2.0) / nhl);

    const q_buf = ggml.vulkan.context.tensorSubbuffer(ctx, q, false);
    var k_buf = ggml.vulkan.context.tensorSubbuffer(ctx, k, false);
    var v_buf = ggml.vulkan.context.tensorSubbuffer(ctx, v, false);
    const dst_buf = ggml.vulkan.context.tensorSubbuffer(ctx, dst, false);
    const mask_buf = if (mask) |t| ggml.vulkan.context.tensorSubbuffer(ctx, t, false) else q_buf;
    const sinks_buf = if (sinks) |t| ggml.vulkan.context.tensorSubbuffer(ctx, t, false) else q_buf;
    const mask_opt_buf = if (use_mask_opt) ggml.vulkan.context.subbuffer(ctx, ctx.prealloc_y.?, 0) else q_buf;
    const sparse_buf = if (use_sparse) ggml.vulkan.context.subbuffer(ctx, ctx.prealloc_y.?, 0) else q_buf;
    if (use_mask_opt or use_sparse) {
        // the mask opt bits and the sparse index list overwrite a matmul input converted into prealloc_y
        ctx.prealloc_y_last_pipeline_used = null;
        ctx.prealloc_y_last_tensor_used = null;
    }

    if (use_dequant_kv) {
        const k_sz = @as(u64, @intCast(ggml.types.ggml_nelements(k))) * kv_scratch_ts;
        const v_sz = @as(u64, @intCast(ggml.types.ggml_nelements(v))) * kv_scratch_ts;
        if (ctx.prealloc_size_x < k_sz + v_sz) {
            ctx.prealloc_size_x = k_sz + v_sz;
            ggml.vulkan.nodes.preallocateBuffers(ctx, subctx);
        }
        const k_dst = ggml.vulkan.ops_matmul.sub(ctx.prealloc_x, 0, k_sz);
        const v_dst = ggml.vulkan.ops_matmul.sub(ctx.prealloc_x, k_sz, v_sz);
        const k_nel = u(ggml.types.ggml_nelements(k));
        const v_nel = u(ggml.types.ggml_nelements(v));
        if (kv_raw_strided) {
            // Copy the strided f16/bf16 K/V once into a contiguous
            // [KV][heads][HS] scratch, laid out as the dequant transpose's.
            const cp_k = ggml.vulkan.op_pipeline.getCpyPipeline(ctx, k, null, k.type).?;
            const cp_v = ggml.vulkan.op_pipeline.getCpyPipeline(ctx, v, null, v.type).?;
            ggml.vulkan.context.requestDescriptorSets(ctx, cp_k, 1);
            ggml.vulkan.context.requestDescriptorSets(ctx, cp_v, 1);
            if (ctx.prealloc_x_need_sync) ggml.vulkan.context.syncBuffers(ctx, subctx);
            const makePc = struct {
                fn f(hs: u32, nh: u32, kv: u32, nel: u32) ggml.vulkan.push.Unary {
                    var pc: ggml.vulkan.push.Unary = .{};
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
                    ggml.vulkan.ops.initPushconstFastdiv(&pc);
                    return pc;
                }
            }.f;
            const cpyElems = struct {
                fn f(ne: u32) [3]u32 {
                    if (ne > 262144) return .{ 512, 512, ggml.vulkan.ops.ceilDiv(ne, 262144) };
                    if (ne > 512) return .{ 512, ggml.vulkan.ops.ceilDiv(ne, 512), 1 };
                    return .{ ne, 1, 1 };
                }
            }.f;
            const pc_k = makePc(HSK, u(k.ne[2]), KV, k_nel);
            ggml.vulkan.context.dispatch(ctx, subctx, cp_k, &.{ k_buf, k_dst }, &pc_k, cpyElems(k_nel));
            const pc_v = makePc(HSV, u(v.ne[2]), KV, v_nel);
            ggml.vulkan.context.dispatch(ctx, subctx, cp_v, &.{ v_buf, v_dst }, &pc_v, cpyElems(v_nel));
        } else {
            const tr_k = dev.pipeline_dequant_transpose[@intCast(k.type)].?;
            const tr_v = dev.pipeline_dequant_transpose[@intCast(v.type)].?;
            ggml.vulkan.context.requestDescriptorSets(ctx, tr_k, 1);
            ggml.vulkan.context.requestDescriptorSets(ctx, tr_v, 1);
            if (ctx.prealloc_x_need_sync) ggml.vulkan.context.syncBuffers(ctx, subctx);
            {
                const pc = [5]u32{ HSK, u(k.ne[2]), KV, 0, k_nel };
                ggml.vulkan.context.dispatch(ctx, subctx, tr_k, &.{ k_buf, k_dst }, &pc, .{ k_nel, 1, 1 });
            }
            {
                const pc = [5]u32{ HSV, u(v.ne[2]), KV, 0, v_nel };
                ggml.vulkan.context.dispatch(ctx, subctx, tr_v, &.{ v_buf, v_dst }, &pc, .{ v_nel, 1, 1 });
            }
        }
        ggml.vulkan.context.syncBuffers(ctx, subctx);
        k_buf = k_dst;
        v_buf = v_dst;
    }

    const mask_n_head_log2: u32 = (@as(u32, @intFromBool(sinks != null)) << 24) | n_head_log2;

    if (use_mask_opt) {
        const mk = mask.?;
        const h: u32 = @sizeOf(ggml.impl.ggml_c.ggml_fp16_t);
        const opt_pc: ggml.vulkan.push.FlashAttnMaskOpt = .{
            .nem0 = nem0,
            .nem1 = nem1,
            .nem2 = nem2,
            .nbm1 = @truncate(mk.nb[1] / h),
            .nbm2 = @truncate(mk.nb[2] / h),
            .nbm3 = @truncate(mk.nb[3] / h),
            .nbd1 = mask_opt_num_dwords,
            .nbd2 = mask_opt_num_dwords *% ggml.vulkan.ops.ceilDiv(nem1, Br),
            .nbd3 = mask_opt_num_dwords *% ggml.vulkan.ops.ceilDiv(nem1, Br) *% nem2,
        };
        ggml.vulkan.context.dispatch(ctx, subctx, pipeline_fa_mask_opt.?, &.{ mask_buf, mask_opt_buf }, &opt_pc, .{ mask_opt_num_dwords, ggml.vulkan.ops.ceilDiv(nem1, Br), nem2 *% nem3 });
        ggml.vulkan.context.syncBuffers(ctx, subctx);
    }

    if (use_sparse) {
        const mk = mask.?;
        const h: u32 = @sizeOf(ggml.impl.ggml_c.ggml_fp16_t);
        const sc_pc: ggml.vulkan.push.FlashAttnSparseCompact = .{
            .KV = KV,
            .nem1 = nem1,
            .nem2 = nem2,
            .nbm1 = @truncate(mk.nb[1] / h),
            .nbm2 = @truncate(mk.nb[2] / h),
            .nbm3 = @truncate(mk.nb[3] / h),
            .n_kv_max = @intCast(n_kv_max),
        };
        ggml.vulkan.context.dispatch(ctx, subctx, sparse_compact_pipeline.?, &.{ mask_buf, sparse_buf }, &sc_pc, .{ nem1, nem2, nem3 });
        ggml.vulkan.context.syncBuffers(ctx, subctx);
    }

    const pc: ggml.vulkan.push.FlashAttn = .{
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

    if (xe_fa_opt) {
        xeFa(ctx, subctx, q, k, v, dst, .{
            .use_2phases = xe_fa_use_2phases,
            .pipelines = xe_fa_pipeline_dual_phases,
            .pipeline = xe_fa_pipeline,
            .params = xe_fa_params,
            .q_stride = q_stride,
            .k_stride = k_stride,
            .v_stride = v_stride,
            .nem1 = nem1,
            .nem2 = nem2,
            .nem3 = nem3,
            .sinks = sinks != null,
            .scale = scale,
            .size_q_staging = size_q_staging,
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
        const pipeline = pipeline_.?;
        const reduce = dev.pipeline_flash_attn_split_k_reduce.?;
        ggml.vulkan.context.requestDescriptorSets(ctx, reduce, 1);
        if (ctx.prealloc_split_k_need_sync) ggml.vulkan.context.syncBuffers(ctx, subctx);

        // workgroups_x now counts splits, so cancel the divide by wg_denoms[0].
        var dispatch_x: u32 = undefined;
        if (gqa_ratio > 1) {
            workgroups_x *%= pipeline.wg_denoms[0];
            dispatch_x = split_k *% workgroups_x;
        } else {
            dispatch_x = Tr *% split_k *% pipeline.wg_denoms[0];
        }

        const split_k_buf = ggml.vulkan.context.subbuffer(ctx, ctx.prealloc_split_k.?, 0);
        ggml.vulkan.context.dispatch(ctx, subctx, pipeline, &.{ q_buf, k_buf, v_buf, mask_buf, sinks_buf, split_k_buf, mask_opt_buf, sparse_buf }, &pc, .{ dispatch_x, workgroups_y, workgroups_z });

        ggml.vulkan.context.syncBuffers(ctx, subctx);
        const pc2: ggml.vulkan.push.FlashAttnSplitKReduce = .{ .D = HSV, .ne1 = @truncate(ne1), .ne2 = @truncate(ne2), .ne3 = @truncate(ne3), .k_num = split_k, .sinks = @intFromBool(sinks != null) };
        ggml.vulkan.context.dispatch(ctx, subctx, reduce, &.{ split_k_buf, sinks_buf, dst_buf }, &pc2, .{ @truncate(ne1), HSV, @truncate(ne2 *% ne3) });
        ctx.prealloc_split_k_need_sync = true;
    } else {
        const pipeline = pipeline_.?;
        // With gqa, one actual workgroup per batch: cancel out wg_denoms
        if (gqa_ratio > 1) workgroups_x *%= pipeline.wg_denoms[0];
        ggml.vulkan.context.dispatch(ctx, subctx, pipeline, &.{ q_buf, k_buf, v_buf, mask_buf, sinks_buf, dst_buf, mask_opt_buf, sparse_buf }, &pc, .{ workgroups_x, workgroups_y, workgroups_z });
    }

    if (use_dequant_kv) ctx.prealloc_x_need_sync = true;
    if (use_mask_opt or use_sparse) ctx.prealloc_y_need_sync = true;
}

/// The function-static `disable_sparse` (ggml-vulkan.cpp:8543 @b1b770d90):
/// whether `GGML_VK_FA_SPARSE_DISABLE` is set, read once, on first use.
const DisableSparse = struct {
    /// 0 unread, 1 false, 2 true.
    var state: std.atomic.Value(u8) = .init(0);

    fn get() bool {
        const v = state.load(.acquire);
        if (v != 0) return v == 2;
        const set = ggml.vulkan.common.envSet("GGML_VK_FA_SPARSE_DISABLE");
        state.store(if (set) 2 else 1, .release);
        return set;
    }
};

/// What `xeFa` takes from `flashAttn`'s locals.
const XeFaArgs = struct {
    use_2phases: bool,
    pipelines: ggml.vulkan.device.XeFaDecodePipelines,
    pipeline: ?*ggml.vulkan.pipeline.Pipeline,
    params: ggml.vulkan.device.XeFaTuningParams,
    q_stride: u32,
    k_stride: u32,
    v_stride: u32,
    nem1: u32,
    nem2: u32,
    nem3: u32,
    sinks: bool,
    scale: f32,
    size_q_staging: u64,
    size_p: u64,
    size_group_max: u64,
    q_buf: ggml.vulkan.buffer.Subbuffer,
    k_buf: ggml.vulkan.buffer.Subbuffer,
    v_buf: ggml.vulkan.buffer.Subbuffer,
    mask_buf: ggml.vulkan.buffer.Subbuffer,
    sinks_buf: ggml.vulkan.buffer.Subbuffer,
    dst_buf: ggml.vulkan.buffer.Subbuffer,
};

/// The `upper_power_of_2` lambda (ggml-vulkan.cpp:1540 @b1b770d90) of
/// `get_xe_fa_tuning_params`: the next power of two at or above `in`.
pub fn upperPowerOf2(in: u32) u32 {
    ggml.impl.assert(in != 0, "in != 0");
    if (in <= 1) return 1;
    var ret = in - 1;
    ret |= ret >> 1;
    ret |= ret >> 2;
    ret |= ret >> 4;
    ret |= ret >> 8;
    ret |= ret >> 16;
    return ret +% 1;
}

/// The `xe_fa_opt` arm (ggml-vulkan.cpp:8946 @b1b770d90) of
/// `ggml_vk_flash_attn`: Intel Xe's flash attention. Prefill is one
/// dispatch; decode copies Q to contiguous fp16 in `prealloc_x`, then phase
/// 1 writes P and each group's max beside it and phase 2 reduces them
/// against V into `dst`.
fn xeFa(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, q: *const ggml.impl.ggml_c.ggml_tensor, k: *const ggml.impl.ggml_c.ggml_tensor, v: *const ggml.impl.ggml_c.ggml_tensor, dst: *const ggml.impl.ggml_c.ggml_tensor, a: XeFaArgs) void {
    const u = ggml.vulkan.ops.u32Of;
    const head_stride_q: u32 = @truncate(q.nb[2] / ggml.types.ggml_type_size(q.type));
    const head_stride_k: u32 = @truncate(k.nb[2] / ggml.types.ggml_type_size(k.type));
    const head_stride_v: u32 = @truncate(v.nb[2] / ggml.types.ggml_type_size(v.type));
    const batch_stride_q: u32 = @truncate(q.nb[3] / ggml.types.ggml_type_size(q.type));
    const batch_stride_k: u32 = @truncate(k.nb[3] / ggml.types.ggml_type_size(k.type));
    const batch_stride_v: u32 = @truncate(v.nb[3] / ggml.types.ggml_type_size(v.type));
    const batch_stride_o: u32 = @truncate(dst.nb[3] / ggml.types.ggml_type_size(dst.type));

    var pc_xe_fa: ggml.vulkan.push.XeFa = .{
        .neq1 = u(q.ne[1]),
        .neq2 = u(q.ne[2]),
        .neq3 = u(q.ne[3]),
        .nek1 = u(k.ne[1]),
        .nek2 = u(k.ne[2]),
        .nek3 = u(k.ne[3]),
        .nev2 = u(v.ne[2]),
        .nev3 = u(v.ne[3]),
        .nem1 = a.nem1,
        .nem2 = a.nem2,
        .nem3 = a.nem3,
        .nbq1 = a.q_stride,
        .nbq2 = head_stride_q,
        .nbk1 = a.k_stride,
        .nbk2 = head_stride_k,
        .nbv1 = a.v_stride,
        .nbv2 = head_stride_v,
        .nbq3 = batch_stride_q,
        .nbk3 = batch_stride_k,
        .nbv3 = batch_stride_v,
        .nb3 = batch_stride_o,
        .flag = if (a.sinks) 1 << 16 else 0,
        .scale = a.scale,
    };

    if (a.use_2phases) {
        const to_fp16_vk_0 = ggml.vulkan.op_pipeline.getCpyPipeline(ctx, q, null, ggml.impl.ggml_c.GGML_TYPE_F16).?;
        var temp_buf_offset: u64 = 0;
        const q_temp_buf = ggml.vulkan.context.subbuffer(ctx, ctx.prealloc_x.?, temp_buf_offset);
        temp_buf_offset += a.size_q_staging;
        const p_temp_buf = ggml.vulkan.context.subbuffer(ctx, ctx.prealloc_x.?, temp_buf_offset);
        temp_buf_offset += a.size_p;
        const max_temp_buf = ggml.vulkan.context.subbuffer(ctx, ctx.prealloc_x.?, temp_buf_offset);
        temp_buf_offset += a.size_group_max;
        // int64_t arithmetic in the C++: neq1, nek2 and ne0 are int64_t.
        const dp = a.params.decode_params;
        const ph1_wg: u32 = @truncate(@as(u64, @bitCast(@divTrunc(q.ne[1] + dp.out_per_wg_ph1 - 1, dp.out_per_wg_ph1) * k.ne[2])));
        const ph2_wg: u32 = @truncate(@as(u64, @bitCast(@divTrunc(@divTrunc(q.ne[1] + dp.out_row_per_wg_ph2 - 1, dp.out_row_per_wg_ph2) * dst.ne[0], dp.out_col_per_wg_ph2))));

        pc_xe_fa.nbq1 = @truncate(@as(u64, @bitCast(q.ne[0])));
        pc_xe_fa.nbq2 = @truncate(@as(u64, @bitCast(q.ne[0] * q.ne[1])));
        pc_xe_fa.nbq3 = @truncate(@as(u64, @bitCast(q.ne[0] * q.ne[1] * q.ne[2])));

        ggml.vulkan.context.requestDescriptorSets(ctx, to_fp16_vk_0, 1);
        ggml.vulkan.ops_matmul.cpyToContiguous(ctx, subctx, to_fp16_vk_0, q, a.q_buf, q_temp_buf);

        ggml.vulkan.context.requestDescriptorSets(ctx, a.pipelines.first.?, 1);
        ggml.vulkan.context.dispatch(ctx, subctx, a.pipelines.first.?, &.{ q_temp_buf, a.k_buf, a.mask_buf, p_temp_buf, max_temp_buf }, &pc_xe_fa, .{ ph1_wg, u(k.ne[1]), u(q.ne[3]) });

        ggml.vulkan.context.syncBuffers(ctx, subctx);
        ggml.vulkan.context.requestDescriptorSets(ctx, a.pipelines.second.?, 1);
        ggml.vulkan.context.dispatch(ctx, subctx, a.pipelines.second.?, &.{ p_temp_buf, a.v_buf, max_temp_buf, a.sinks_buf, a.dst_buf }, &pc_xe_fa, .{ ph2_wg, u(v.ne[2]), u(q.ne[3]) });

        ctx.prealloc_x_need_sync = true;
    } else {
        ggml.vulkan.context.requestDescriptorSets(ctx, a.pipeline.?, 1);
        ggml.vulkan.context.dispatch(ctx, subctx, a.pipeline.?, &.{ a.q_buf, a.k_buf, a.v_buf, a.mask_buf, a.sinks_buf, a.dst_buf }, &pc_xe_fa, .{ u(q.ne[2]), u(q.ne[1]), u(q.ne[3]) });
    }
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
