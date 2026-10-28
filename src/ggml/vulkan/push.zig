//! The push constants the Vulkan shaders read.
//!
//! # Provenance
//!
//! Mirrors the `vk_*_push_constants` structs of
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.4.0
//! (`5266f24da`).
//!
//! # Generated, not written
//!
//! A shader reads these by offset. This file is produced by
//! `scripts/gen-vk-push`, which also writes `harness/vk_push_layout.cpp`:
//! the same sizes and offsets as `static_assert`s against the structs the
//! reference compiles. Do not edit it by hand; change the generator and
//! re-run it.

const std = @import("std");

/// Mirrors `vk_mat_mat_push_constants` (ggml-vulkan.cpp:1306 @5266f24da).
pub const MatMat = extern struct {
    M: u32 = std.mem.zeroes(u32),
    N: u32 = std.mem.zeroes(u32),
    K: u32 = std.mem.zeroes(u32),
    stride_a: u32 = std.mem.zeroes(u32),
    stride_b: u32 = std.mem.zeroes(u32),
    stride_d: u32 = std.mem.zeroes(u32),
    batch_stride_a: u32 = std.mem.zeroes(u32),
    batch_stride_b: u32 = std.mem.zeroes(u32),
    batch_stride_d: u32 = std.mem.zeroes(u32),
    base_work_group_z: u32 = std.mem.zeroes(u32),
    num_batches: u32 = std.mem.zeroes(u32),
    k_split: u32 = std.mem.zeroes(u32),
    ne02: u32 = std.mem.zeroes(u32),
    ne12: u32 = std.mem.zeroes(u32),
    broadcast2: u32 = std.mem.zeroes(u32),
    broadcast3: u32 = std.mem.zeroes(u32),
    padded_N: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(MatMat) == 68);
    std.debug.assert(@offsetOf(MatMat, "M") == 0);
    std.debug.assert(@offsetOf(MatMat, "N") == 4);
    std.debug.assert(@offsetOf(MatMat, "K") == 8);
    std.debug.assert(@offsetOf(MatMat, "stride_a") == 12);
    std.debug.assert(@offsetOf(MatMat, "stride_b") == 16);
    std.debug.assert(@offsetOf(MatMat, "stride_d") == 20);
    std.debug.assert(@offsetOf(MatMat, "batch_stride_a") == 24);
    std.debug.assert(@offsetOf(MatMat, "batch_stride_b") == 28);
    std.debug.assert(@offsetOf(MatMat, "batch_stride_d") == 32);
    std.debug.assert(@offsetOf(MatMat, "base_work_group_z") == 36);
    std.debug.assert(@offsetOf(MatMat, "num_batches") == 40);
    std.debug.assert(@offsetOf(MatMat, "k_split") == 44);
    std.debug.assert(@offsetOf(MatMat, "ne02") == 48);
    std.debug.assert(@offsetOf(MatMat, "ne12") == 52);
    std.debug.assert(@offsetOf(MatMat, "broadcast2") == 56);
    std.debug.assert(@offsetOf(MatMat, "broadcast3") == 60);
    std.debug.assert(@offsetOf(MatMat, "padded_N") == 64);
}

/// Mirrors `vk_mat_vec_push_constants` (ggml-vulkan.cpp:1321 @5266f24da).
pub const MatVec = extern struct {
    ncols: u32 = std.mem.zeroes(u32),
    stride_a: u32 = std.mem.zeroes(u32),
    stride_b: u32 = std.mem.zeroes(u32),
    stride_d: u32 = std.mem.zeroes(u32),
    batch_stride_a: u32 = std.mem.zeroes(u32),
    batch_stride_b: u32 = std.mem.zeroes(u32),
    batch_stride_d: u32 = std.mem.zeroes(u32),
    fusion_flags: u32 = std.mem.zeroes(u32),
    base_work_group_y: u32 = std.mem.zeroes(u32),
    ne02: u32 = std.mem.zeroes(u32),
    ne12: u32 = std.mem.zeroes(u32),
    broadcast2: u32 = std.mem.zeroes(u32),
    broadcast3: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(MatVec) == 52);
    std.debug.assert(@offsetOf(MatVec, "ncols") == 0);
    std.debug.assert(@offsetOf(MatVec, "stride_a") == 4);
    std.debug.assert(@offsetOf(MatVec, "stride_b") == 8);
    std.debug.assert(@offsetOf(MatVec, "stride_d") == 12);
    std.debug.assert(@offsetOf(MatVec, "batch_stride_a") == 16);
    std.debug.assert(@offsetOf(MatVec, "batch_stride_b") == 20);
    std.debug.assert(@offsetOf(MatVec, "batch_stride_d") == 24);
    std.debug.assert(@offsetOf(MatVec, "fusion_flags") == 28);
    std.debug.assert(@offsetOf(MatVec, "base_work_group_y") == 32);
    std.debug.assert(@offsetOf(MatVec, "ne02") == 36);
    std.debug.assert(@offsetOf(MatVec, "ne12") == 40);
    std.debug.assert(@offsetOf(MatVec, "broadcast2") == 44);
    std.debug.assert(@offsetOf(MatVec, "broadcast3") == 48);
}

/// Mirrors `vk_mat_vec_p021_push_constants` (ggml-vulkan.cpp:1337 @5266f24da).
pub const MatVecP021 = extern struct {
    ncols_x: u32 = std.mem.zeroes(u32),
    nrows_x: u32 = std.mem.zeroes(u32),
    nchannels_x: u32 = std.mem.zeroes(u32),
    nchannels_y: u32 = std.mem.zeroes(u32),
    b_offset: u32 = std.mem.zeroes(u32),
    d_offset: u32 = std.mem.zeroes(u32),
    fusion_flags: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(MatVecP021) == 28);
    std.debug.assert(@offsetOf(MatVecP021, "ncols_x") == 0);
    std.debug.assert(@offsetOf(MatVecP021, "nrows_x") == 4);
    std.debug.assert(@offsetOf(MatVecP021, "nchannels_x") == 8);
    std.debug.assert(@offsetOf(MatVecP021, "nchannels_y") == 12);
    std.debug.assert(@offsetOf(MatVecP021, "b_offset") == 16);
    std.debug.assert(@offsetOf(MatVecP021, "d_offset") == 20);
    std.debug.assert(@offsetOf(MatVecP021, "fusion_flags") == 24);
}

/// Mirrors `vk_mat_vec_nc_push_constants` (ggml-vulkan.cpp:1347 @5266f24da).
pub const MatVecNc = extern struct {
    ncols_x: u32 = std.mem.zeroes(u32),
    nrows_x: u32 = std.mem.zeroes(u32),
    row_stride_x: u32 = std.mem.zeroes(u32),
    channel_stride_x: u32 = std.mem.zeroes(u32),
    channel_stride_y: u32 = std.mem.zeroes(u32),
    channel_x_divisor: u32 = std.mem.zeroes(u32),
    ne12: u32 = std.mem.zeroes(u32),
    b_offset: u32 = std.mem.zeroes(u32),
    d_offset: u32 = std.mem.zeroes(u32),
    nb03: u32 = std.mem.zeroes(u32),
    nb13: u32 = std.mem.zeroes(u32),
    nb23: u32 = std.mem.zeroes(u32),
    fusion_flags: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(MatVecNc) == 52);
    std.debug.assert(@offsetOf(MatVecNc, "ncols_x") == 0);
    std.debug.assert(@offsetOf(MatVecNc, "nrows_x") == 4);
    std.debug.assert(@offsetOf(MatVecNc, "row_stride_x") == 8);
    std.debug.assert(@offsetOf(MatVecNc, "channel_stride_x") == 12);
    std.debug.assert(@offsetOf(MatVecNc, "channel_stride_y") == 16);
    std.debug.assert(@offsetOf(MatVecNc, "channel_x_divisor") == 20);
    std.debug.assert(@offsetOf(MatVecNc, "ne12") == 24);
    std.debug.assert(@offsetOf(MatVecNc, "b_offset") == 28);
    std.debug.assert(@offsetOf(MatVecNc, "d_offset") == 32);
    std.debug.assert(@offsetOf(MatVecNc, "nb03") == 36);
    std.debug.assert(@offsetOf(MatVecNc, "nb13") == 40);
    std.debug.assert(@offsetOf(MatVecNc, "nb23") == 44);
    std.debug.assert(@offsetOf(MatVecNc, "fusion_flags") == 48);
}

/// Mirrors `vk_mat_mat_id_push_constants` (ggml-vulkan.cpp:1363 @5266f24da).
pub const MatMatId = extern struct {
    M: u32 = std.mem.zeroes(u32),
    N: u32 = std.mem.zeroes(u32),
    K: u32 = std.mem.zeroes(u32),
    stride_a: u32 = std.mem.zeroes(u32),
    stride_b: u32 = std.mem.zeroes(u32),
    stride_d: u32 = std.mem.zeroes(u32),
    batch_stride_a: u32 = std.mem.zeroes(u32),
    batch_stride_b: u32 = std.mem.zeroes(u32),
    batch_stride_d: u32 = std.mem.zeroes(u32),
    nei0: u32 = std.mem.zeroes(u32),
    nei1: u32 = std.mem.zeroes(u32),
    nbi1: u32 = std.mem.zeroes(u32),
    ne11: u32 = std.mem.zeroes(u32),
    n_experts: u32 = std.mem.zeroes(u32),
    hoist_row_ids: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(MatMatId) == 60);
    std.debug.assert(@offsetOf(MatMatId, "M") == 0);
    std.debug.assert(@offsetOf(MatMatId, "N") == 4);
    std.debug.assert(@offsetOf(MatMatId, "K") == 8);
    std.debug.assert(@offsetOf(MatMatId, "stride_a") == 12);
    std.debug.assert(@offsetOf(MatMatId, "stride_b") == 16);
    std.debug.assert(@offsetOf(MatMatId, "stride_d") == 20);
    std.debug.assert(@offsetOf(MatMatId, "batch_stride_a") == 24);
    std.debug.assert(@offsetOf(MatMatId, "batch_stride_b") == 28);
    std.debug.assert(@offsetOf(MatMatId, "batch_stride_d") == 32);
    std.debug.assert(@offsetOf(MatMatId, "nei0") == 36);
    std.debug.assert(@offsetOf(MatMatId, "nei1") == 40);
    std.debug.assert(@offsetOf(MatMatId, "nbi1") == 44);
    std.debug.assert(@offsetOf(MatMatId, "ne11") == 48);
    std.debug.assert(@offsetOf(MatMatId, "n_experts") == 52);
    std.debug.assert(@offsetOf(MatMatId, "hoist_row_ids") == 56);
}

/// Mirrors `vk_mat_vec_id_push_constants` (ggml-vulkan.cpp:1371 @5266f24da).
pub const MatVecId = extern struct {
    ncols: u32 = std.mem.zeroes(u32),
    stride_a: u32 = std.mem.zeroes(u32),
    stride_b: u32 = std.mem.zeroes(u32),
    stride_d: u32 = std.mem.zeroes(u32),
    batch_stride_a: u32 = std.mem.zeroes(u32),
    batch_stride_b: u32 = std.mem.zeroes(u32),
    batch_stride_d: u32 = std.mem.zeroes(u32),
    fusion_flags: u32 = std.mem.zeroes(u32),
    nei0: u32 = std.mem.zeroes(u32),
    ne11: u32 = std.mem.zeroes(u32),
    expert_i1: u32 = std.mem.zeroes(u32),
    nbi1: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(MatVecId) == 48);
    std.debug.assert(@offsetOf(MatVecId, "ncols") == 0);
    std.debug.assert(@offsetOf(MatVecId, "stride_a") == 4);
    std.debug.assert(@offsetOf(MatVecId, "stride_b") == 8);
    std.debug.assert(@offsetOf(MatVecId, "stride_d") == 12);
    std.debug.assert(@offsetOf(MatVecId, "batch_stride_a") == 16);
    std.debug.assert(@offsetOf(MatVecId, "batch_stride_b") == 20);
    std.debug.assert(@offsetOf(MatVecId, "batch_stride_d") == 24);
    std.debug.assert(@offsetOf(MatVecId, "fusion_flags") == 28);
    std.debug.assert(@offsetOf(MatVecId, "nei0") == 32);
    std.debug.assert(@offsetOf(MatVecId, "ne11") == 36);
    std.debug.assert(@offsetOf(MatVecId, "expert_i1") == 40);
    std.debug.assert(@offsetOf(MatVecId, "nbi1") == 44);
}

/// Mirrors `vk_flash_attn_push_constants` (ggml-vulkan.cpp:1386 @5266f24da).
pub const FlashAttn = extern struct {
    N: u32 = std.mem.zeroes(u32),
    KV: u32 = std.mem.zeroes(u32),
    ne1: u32 = std.mem.zeroes(u32),
    ne2: u32 = std.mem.zeroes(u32),
    ne3: u32 = std.mem.zeroes(u32),
    neq2: u32 = std.mem.zeroes(u32),
    neq3: u32 = std.mem.zeroes(u32),
    nek2: u32 = std.mem.zeroes(u32),
    nek3: u32 = std.mem.zeroes(u32),
    nev2: u32 = std.mem.zeroes(u32),
    nev3: u32 = std.mem.zeroes(u32),
    nem1: u32 = std.mem.zeroes(u32),
    nem2: u32 = std.mem.zeroes(u32),
    nem3: u32 = std.mem.zeroes(u32),
    nb01: u32 = std.mem.zeroes(u32),
    nb02: u32 = std.mem.zeroes(u32),
    nb03: u32 = std.mem.zeroes(u32),
    nb11: u32 = std.mem.zeroes(u32),
    nb12: u32 = std.mem.zeroes(u32),
    nb13: u32 = std.mem.zeroes(u32),
    nb21: u32 = std.mem.zeroes(u32),
    nb22: u32 = std.mem.zeroes(u32),
    nb23: u32 = std.mem.zeroes(u32),
    scale: f32 = std.mem.zeroes(f32),
    max_bias: f32 = std.mem.zeroes(f32),
    logit_softcap: f32 = std.mem.zeroes(f32),
    mask_n_head_log2: u32 = std.mem.zeroes(u32),
    m0: f32 = std.mem.zeroes(f32),
    m1: f32 = std.mem.zeroes(f32),
    gqa_ratio: u32 = std.mem.zeroes(u32),
    split_kv: u32 = std.mem.zeroes(u32),
    k_num: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(FlashAttn) == 128);
    std.debug.assert(@offsetOf(FlashAttn, "N") == 0);
    std.debug.assert(@offsetOf(FlashAttn, "KV") == 4);
    std.debug.assert(@offsetOf(FlashAttn, "ne1") == 8);
    std.debug.assert(@offsetOf(FlashAttn, "ne2") == 12);
    std.debug.assert(@offsetOf(FlashAttn, "ne3") == 16);
    std.debug.assert(@offsetOf(FlashAttn, "neq2") == 20);
    std.debug.assert(@offsetOf(FlashAttn, "neq3") == 24);
    std.debug.assert(@offsetOf(FlashAttn, "nek2") == 28);
    std.debug.assert(@offsetOf(FlashAttn, "nek3") == 32);
    std.debug.assert(@offsetOf(FlashAttn, "nev2") == 36);
    std.debug.assert(@offsetOf(FlashAttn, "nev3") == 40);
    std.debug.assert(@offsetOf(FlashAttn, "nem1") == 44);
    std.debug.assert(@offsetOf(FlashAttn, "nem2") == 48);
    std.debug.assert(@offsetOf(FlashAttn, "nem3") == 52);
    std.debug.assert(@offsetOf(FlashAttn, "nb01") == 56);
    std.debug.assert(@offsetOf(FlashAttn, "nb02") == 60);
    std.debug.assert(@offsetOf(FlashAttn, "nb03") == 64);
    std.debug.assert(@offsetOf(FlashAttn, "nb11") == 68);
    std.debug.assert(@offsetOf(FlashAttn, "nb12") == 72);
    std.debug.assert(@offsetOf(FlashAttn, "nb13") == 76);
    std.debug.assert(@offsetOf(FlashAttn, "nb21") == 80);
    std.debug.assert(@offsetOf(FlashAttn, "nb22") == 84);
    std.debug.assert(@offsetOf(FlashAttn, "nb23") == 88);
    std.debug.assert(@offsetOf(FlashAttn, "scale") == 92);
    std.debug.assert(@offsetOf(FlashAttn, "max_bias") == 96);
    std.debug.assert(@offsetOf(FlashAttn, "logit_softcap") == 100);
    std.debug.assert(@offsetOf(FlashAttn, "mask_n_head_log2") == 104);
    std.debug.assert(@offsetOf(FlashAttn, "m0") == 108);
    std.debug.assert(@offsetOf(FlashAttn, "m1") == 112);
    std.debug.assert(@offsetOf(FlashAttn, "gqa_ratio") == 116);
    std.debug.assert(@offsetOf(FlashAttn, "split_kv") == 120);
    std.debug.assert(@offsetOf(FlashAttn, "k_num") == 124);
}

/// Mirrors `vk_op_push_constants` (ggml-vulkan.cpp:1428 @5266f24da).
pub const Op = extern struct {
    KX: u32 = std.mem.zeroes(u32),
    KY: u32 = std.mem.zeroes(u32),
    param1: f32 = std.mem.zeroes(f32),
    param2: f32 = std.mem.zeroes(f32),
    param3: f32 = std.mem.zeroes(f32),
    param4: f32 = std.mem.zeroes(f32),
};
comptime {
    std.debug.assert(@sizeOf(Op) == 24);
    std.debug.assert(@offsetOf(Op, "KX") == 0);
    std.debug.assert(@offsetOf(Op, "KY") == 4);
    std.debug.assert(@offsetOf(Op, "param1") == 8);
    std.debug.assert(@offsetOf(Op, "param2") == 12);
    std.debug.assert(@offsetOf(Op, "param3") == 16);
    std.debug.assert(@offsetOf(Op, "param4") == 20);
}

/// Mirrors `vk_op_fwht_push_constants` (ggml-vulkan.cpp:1437 @5266f24da).
pub const Fwht = extern struct {
    n_rows: u32 = std.mem.zeroes(u32),
    src_offset: u32 = std.mem.zeroes(u32),
    dst_offset: u32 = std.mem.zeroes(u32),
    scale: f32 = std.mem.zeroes(f32),
};
comptime {
    std.debug.assert(@sizeOf(Fwht) == 16);
    std.debug.assert(@offsetOf(Fwht, "n_rows") == 0);
    std.debug.assert(@offsetOf(Fwht, "src_offset") == 4);
    std.debug.assert(@offsetOf(Fwht, "dst_offset") == 8);
    std.debug.assert(@offsetOf(Fwht, "scale") == 12);
}

/// Mirrors `vk_op_count_experts_push_constants` (ggml-vulkan.cpp:1444 @5266f24da).
pub const CountExperts = extern struct {
    ne00: u32 = std.mem.zeroes(u32),
    ne01: u32 = std.mem.zeroes(u32),
    nb00: u32 = std.mem.zeroes(u32),
    nb01: u32 = std.mem.zeroes(u32),
    a_offset: u32 = std.mem.zeroes(u32),
    n_experts: u32 = std.mem.zeroes(u32),
    hoist_row_ids: u32 = std.mem.zeroes(u32),
    ne00mp: u32 = std.mem.zeroes(u32),
    ne00L: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(CountExperts) == 36);
    std.debug.assert(@offsetOf(CountExperts, "ne00") == 0);
    std.debug.assert(@offsetOf(CountExperts, "ne01") == 4);
    std.debug.assert(@offsetOf(CountExperts, "nb00") == 8);
    std.debug.assert(@offsetOf(CountExperts, "nb01") == 12);
    std.debug.assert(@offsetOf(CountExperts, "a_offset") == 16);
    std.debug.assert(@offsetOf(CountExperts, "n_experts") == 20);
    std.debug.assert(@offsetOf(CountExperts, "hoist_row_ids") == 24);
    std.debug.assert(@offsetOf(CountExperts, "ne00mp") == 28);
    std.debug.assert(@offsetOf(CountExperts, "ne00L") == 32);
}

/// Mirrors `vk_op_glu_push_constants` (ggml-vulkan.cpp:1456 @5266f24da).
pub const Glu = extern struct {
    N: u32 = std.mem.zeroes(u32),
    ne00: u32 = std.mem.zeroes(u32),
    ne20: u32 = std.mem.zeroes(u32),
    mode: u32 = std.mem.zeroes(u32),
    alpha: f32 = std.mem.zeroes(f32),
    limit: f32 = std.mem.zeroes(f32),
    nb00: u32 = std.mem.zeroes(u32),
    nb01: u32 = std.mem.zeroes(u32),
    nb02: u32 = std.mem.zeroes(u32),
    nb03: u32 = std.mem.zeroes(u32),
    nb10: u32 = std.mem.zeroes(u32),
    nb11: u32 = std.mem.zeroes(u32),
    nb12: u32 = std.mem.zeroes(u32),
    nb13: u32 = std.mem.zeroes(u32),
    nb20: u32 = std.mem.zeroes(u32),
    nb21: u32 = std.mem.zeroes(u32),
    nb22: u32 = std.mem.zeroes(u32),
    nb23: u32 = std.mem.zeroes(u32),
    ne21: u32 = std.mem.zeroes(u32),
    ne22: u32 = std.mem.zeroes(u32),
    misalign_offsets: u32 = std.mem.zeroes(u32),
    ne2_012mp: u32 = std.mem.zeroes(u32),
    ne2_012L: u32 = std.mem.zeroes(u32),
    ne2_01mp: u32 = std.mem.zeroes(u32),
    ne2_01L: u32 = std.mem.zeroes(u32),
    ne2_0mp: u32 = std.mem.zeroes(u32),
    ne2_0L: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(Glu) == 108);
    std.debug.assert(@offsetOf(Glu, "N") == 0);
    std.debug.assert(@offsetOf(Glu, "ne00") == 4);
    std.debug.assert(@offsetOf(Glu, "ne20") == 8);
    std.debug.assert(@offsetOf(Glu, "mode") == 12);
    std.debug.assert(@offsetOf(Glu, "alpha") == 16);
    std.debug.assert(@offsetOf(Glu, "limit") == 20);
    std.debug.assert(@offsetOf(Glu, "nb00") == 24);
    std.debug.assert(@offsetOf(Glu, "nb01") == 28);
    std.debug.assert(@offsetOf(Glu, "nb02") == 32);
    std.debug.assert(@offsetOf(Glu, "nb03") == 36);
    std.debug.assert(@offsetOf(Glu, "nb10") == 40);
    std.debug.assert(@offsetOf(Glu, "nb11") == 44);
    std.debug.assert(@offsetOf(Glu, "nb12") == 48);
    std.debug.assert(@offsetOf(Glu, "nb13") == 52);
    std.debug.assert(@offsetOf(Glu, "nb20") == 56);
    std.debug.assert(@offsetOf(Glu, "nb21") == 60);
    std.debug.assert(@offsetOf(Glu, "nb22") == 64);
    std.debug.assert(@offsetOf(Glu, "nb23") == 68);
    std.debug.assert(@offsetOf(Glu, "ne21") == 72);
    std.debug.assert(@offsetOf(Glu, "ne22") == 76);
    std.debug.assert(@offsetOf(Glu, "misalign_offsets") == 80);
    std.debug.assert(@offsetOf(Glu, "ne2_012mp") == 84);
    std.debug.assert(@offsetOf(Glu, "ne2_012L") == 88);
    std.debug.assert(@offsetOf(Glu, "ne2_01mp") == 92);
    std.debug.assert(@offsetOf(Glu, "ne2_01L") == 96);
    std.debug.assert(@offsetOf(Glu, "ne2_0mp") == 100);
    std.debug.assert(@offsetOf(Glu, "ne2_0L") == 104);
}

/// Mirrors `vk_op_unary_push_constants` (ggml-vulkan.cpp:1484 @5266f24da).
pub const Unary = extern struct {
    ne: u32 = std.mem.zeroes(u32),
    ne00: u32 = std.mem.zeroes(u32),
    ne01: u32 = std.mem.zeroes(u32),
    ne02: u32 = std.mem.zeroes(u32),
    ne03: u32 = std.mem.zeroes(u32),
    nb00: u32 = std.mem.zeroes(u32),
    nb01: u32 = std.mem.zeroes(u32),
    nb02: u32 = std.mem.zeroes(u32),
    nb03: u32 = std.mem.zeroes(u32),
    ne10: u32 = std.mem.zeroes(u32),
    ne11: u32 = std.mem.zeroes(u32),
    ne12: u32 = std.mem.zeroes(u32),
    ne13: u32 = std.mem.zeroes(u32),
    nb10: u32 = std.mem.zeroes(u32),
    nb11: u32 = std.mem.zeroes(u32),
    nb12: u32 = std.mem.zeroes(u32),
    nb13: u32 = std.mem.zeroes(u32),
    misalign_offsets: u32 = std.mem.zeroes(u32),
    param1: f32 = std.mem.zeroes(f32),
    param2: f32 = std.mem.zeroes(f32),
    param3: f32 = std.mem.zeroes(f32),
    param4: f32 = std.mem.zeroes(f32),
    ne0_012mp: u32 = std.mem.zeroes(u32),
    ne0_01mp: u32 = std.mem.zeroes(u32),
    ne0_0mp: u32 = std.mem.zeroes(u32),
    ne0_Ls: u32 = std.mem.zeroes(u32),
    ne1_012mp: u32 = std.mem.zeroes(u32),
    ne1_01mp: u32 = std.mem.zeroes(u32),
    ne1_0mp: u32 = std.mem.zeroes(u32),
    ne1_Ls: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(Unary) == 120);
    std.debug.assert(@offsetOf(Unary, "ne") == 0);
    std.debug.assert(@offsetOf(Unary, "ne00") == 4);
    std.debug.assert(@offsetOf(Unary, "ne01") == 8);
    std.debug.assert(@offsetOf(Unary, "ne02") == 12);
    std.debug.assert(@offsetOf(Unary, "ne03") == 16);
    std.debug.assert(@offsetOf(Unary, "nb00") == 20);
    std.debug.assert(@offsetOf(Unary, "nb01") == 24);
    std.debug.assert(@offsetOf(Unary, "nb02") == 28);
    std.debug.assert(@offsetOf(Unary, "nb03") == 32);
    std.debug.assert(@offsetOf(Unary, "ne10") == 36);
    std.debug.assert(@offsetOf(Unary, "ne11") == 40);
    std.debug.assert(@offsetOf(Unary, "ne12") == 44);
    std.debug.assert(@offsetOf(Unary, "ne13") == 48);
    std.debug.assert(@offsetOf(Unary, "nb10") == 52);
    std.debug.assert(@offsetOf(Unary, "nb11") == 56);
    std.debug.assert(@offsetOf(Unary, "nb12") == 60);
    std.debug.assert(@offsetOf(Unary, "nb13") == 64);
    std.debug.assert(@offsetOf(Unary, "misalign_offsets") == 68);
    std.debug.assert(@offsetOf(Unary, "param1") == 72);
    std.debug.assert(@offsetOf(Unary, "param2") == 76);
    std.debug.assert(@offsetOf(Unary, "param3") == 80);
    std.debug.assert(@offsetOf(Unary, "param4") == 84);
    std.debug.assert(@offsetOf(Unary, "ne0_012mp") == 88);
    std.debug.assert(@offsetOf(Unary, "ne0_01mp") == 92);
    std.debug.assert(@offsetOf(Unary, "ne0_0mp") == 96);
    std.debug.assert(@offsetOf(Unary, "ne0_Ls") == 100);
    std.debug.assert(@offsetOf(Unary, "ne1_012mp") == 104);
    std.debug.assert(@offsetOf(Unary, "ne1_01mp") == 108);
    std.debug.assert(@offsetOf(Unary, "ne1_0mp") == 112);
    std.debug.assert(@offsetOf(Unary, "ne1_Ls") == 116);
}

/// Mirrors `vk_op_pad_push_constants` (ggml-vulkan.cpp:1526 @5266f24da).
pub const Pad = extern struct {
    ne: u32 = std.mem.zeroes(u32),
    ne00: u32 = std.mem.zeroes(u32),
    ne01: u32 = std.mem.zeroes(u32),
    ne02: u32 = std.mem.zeroes(u32),
    ne03: u32 = std.mem.zeroes(u32),
    nb00: u32 = std.mem.zeroes(u32),
    nb01: u32 = std.mem.zeroes(u32),
    nb02: u32 = std.mem.zeroes(u32),
    nb03: u32 = std.mem.zeroes(u32),
    ne10: u32 = std.mem.zeroes(u32),
    ne11: u32 = std.mem.zeroes(u32),
    ne12: u32 = std.mem.zeroes(u32),
    ne13: u32 = std.mem.zeroes(u32),
    nb10: u32 = std.mem.zeroes(u32),
    nb11: u32 = std.mem.zeroes(u32),
    nb12: u32 = std.mem.zeroes(u32),
    nb13: u32 = std.mem.zeroes(u32),
    misalign_offsets: u32 = std.mem.zeroes(u32),
    circular: u32 = std.mem.zeroes(u32),
    lp0: u32 = std.mem.zeroes(u32),
    rp0: u32 = std.mem.zeroes(u32),
    lp1: u32 = std.mem.zeroes(u32),
    rp1: u32 = std.mem.zeroes(u32),
    lp2: u32 = std.mem.zeroes(u32),
    rp2: u32 = std.mem.zeroes(u32),
    lp3: u32 = std.mem.zeroes(u32),
    rp3: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(Pad) == 108);
    std.debug.assert(@offsetOf(Pad, "ne") == 0);
    std.debug.assert(@offsetOf(Pad, "ne00") == 4);
    std.debug.assert(@offsetOf(Pad, "ne01") == 8);
    std.debug.assert(@offsetOf(Pad, "ne02") == 12);
    std.debug.assert(@offsetOf(Pad, "ne03") == 16);
    std.debug.assert(@offsetOf(Pad, "nb00") == 20);
    std.debug.assert(@offsetOf(Pad, "nb01") == 24);
    std.debug.assert(@offsetOf(Pad, "nb02") == 28);
    std.debug.assert(@offsetOf(Pad, "nb03") == 32);
    std.debug.assert(@offsetOf(Pad, "ne10") == 36);
    std.debug.assert(@offsetOf(Pad, "ne11") == 40);
    std.debug.assert(@offsetOf(Pad, "ne12") == 44);
    std.debug.assert(@offsetOf(Pad, "ne13") == 48);
    std.debug.assert(@offsetOf(Pad, "nb10") == 52);
    std.debug.assert(@offsetOf(Pad, "nb11") == 56);
    std.debug.assert(@offsetOf(Pad, "nb12") == 60);
    std.debug.assert(@offsetOf(Pad, "nb13") == 64);
    std.debug.assert(@offsetOf(Pad, "misalign_offsets") == 68);
    std.debug.assert(@offsetOf(Pad, "circular") == 72);
    std.debug.assert(@offsetOf(Pad, "lp0") == 76);
    std.debug.assert(@offsetOf(Pad, "rp0") == 80);
    std.debug.assert(@offsetOf(Pad, "lp1") == 84);
    std.debug.assert(@offsetOf(Pad, "rp1") == 88);
    std.debug.assert(@offsetOf(Pad, "lp2") == 92);
    std.debug.assert(@offsetOf(Pad, "rp2") == 96);
    std.debug.assert(@offsetOf(Pad, "lp3") == 100);
    std.debug.assert(@offsetOf(Pad, "rp3") == 104);
}

/// Mirrors `vk_op_binary_push_constants` (ggml-vulkan.cpp:1636 @5266f24da).
pub const Binary = extern struct {
    ne: u32 = std.mem.zeroes(u32),
    ne00: u32 = std.mem.zeroes(u32),
    ne01: u32 = std.mem.zeroes(u32),
    ne02: u32 = std.mem.zeroes(u32),
    ne03: u32 = std.mem.zeroes(u32),
    nb00: u32 = std.mem.zeroes(u32),
    nb01: u32 = std.mem.zeroes(u32),
    nb02: u32 = std.mem.zeroes(u32),
    nb03: u32 = std.mem.zeroes(u32),
    ne10: u32 = std.mem.zeroes(u32),
    ne11: u32 = std.mem.zeroes(u32),
    ne12: u32 = std.mem.zeroes(u32),
    ne13: u32 = std.mem.zeroes(u32),
    nb10: u32 = std.mem.zeroes(u32),
    nb11: u32 = std.mem.zeroes(u32),
    nb12: u32 = std.mem.zeroes(u32),
    nb13: u32 = std.mem.zeroes(u32),
    ne20: u32 = std.mem.zeroes(u32),
    ne21: u32 = std.mem.zeroes(u32),
    ne22: u32 = std.mem.zeroes(u32),
    ne23: u32 = std.mem.zeroes(u32),
    nb20: u32 = std.mem.zeroes(u32),
    nb21: u32 = std.mem.zeroes(u32),
    nb22: u32 = std.mem.zeroes(u32),
    nb23: u32 = std.mem.zeroes(u32),
    misalign_offsets: u32 = std.mem.zeroes(u32),
    param1: f32 = std.mem.zeroes(f32),
    param2: f32 = std.mem.zeroes(f32),
    param3: i32 = std.mem.zeroes(i32),
};
comptime {
    std.debug.assert(@sizeOf(Binary) == 116);
    std.debug.assert(@offsetOf(Binary, "ne") == 0);
    std.debug.assert(@offsetOf(Binary, "ne00") == 4);
    std.debug.assert(@offsetOf(Binary, "ne01") == 8);
    std.debug.assert(@offsetOf(Binary, "ne02") == 12);
    std.debug.assert(@offsetOf(Binary, "ne03") == 16);
    std.debug.assert(@offsetOf(Binary, "nb00") == 20);
    std.debug.assert(@offsetOf(Binary, "nb01") == 24);
    std.debug.assert(@offsetOf(Binary, "nb02") == 28);
    std.debug.assert(@offsetOf(Binary, "nb03") == 32);
    std.debug.assert(@offsetOf(Binary, "ne10") == 36);
    std.debug.assert(@offsetOf(Binary, "ne11") == 40);
    std.debug.assert(@offsetOf(Binary, "ne12") == 44);
    std.debug.assert(@offsetOf(Binary, "ne13") == 48);
    std.debug.assert(@offsetOf(Binary, "nb10") == 52);
    std.debug.assert(@offsetOf(Binary, "nb11") == 56);
    std.debug.assert(@offsetOf(Binary, "nb12") == 60);
    std.debug.assert(@offsetOf(Binary, "nb13") == 64);
    std.debug.assert(@offsetOf(Binary, "ne20") == 68);
    std.debug.assert(@offsetOf(Binary, "ne21") == 72);
    std.debug.assert(@offsetOf(Binary, "ne22") == 76);
    std.debug.assert(@offsetOf(Binary, "ne23") == 80);
    std.debug.assert(@offsetOf(Binary, "nb20") == 84);
    std.debug.assert(@offsetOf(Binary, "nb21") == 88);
    std.debug.assert(@offsetOf(Binary, "nb22") == 92);
    std.debug.assert(@offsetOf(Binary, "nb23") == 96);
    std.debug.assert(@offsetOf(Binary, "misalign_offsets") == 100);
    std.debug.assert(@offsetOf(Binary, "param1") == 104);
    std.debug.assert(@offsetOf(Binary, "param2") == 108);
    std.debug.assert(@offsetOf(Binary, "param3") == 112);
}

/// Mirrors `vk_op_concat_push_constants` (ggml-vulkan.cpp:1646 @5266f24da).
pub const Concat = extern struct {
    ne: u32 = std.mem.zeroes(u32),
    ne00: u32 = std.mem.zeroes(u32),
    ne01: u32 = std.mem.zeroes(u32),
    ne02: u32 = std.mem.zeroes(u32),
    ne03: u32 = std.mem.zeroes(u32),
    nb00: u32 = std.mem.zeroes(u32),
    nb01: u32 = std.mem.zeroes(u32),
    nb02: u32 = std.mem.zeroes(u32),
    nb03: u32 = std.mem.zeroes(u32),
    ne10: u32 = std.mem.zeroes(u32),
    ne11: u32 = std.mem.zeroes(u32),
    ne12: u32 = std.mem.zeroes(u32),
    ne13: u32 = std.mem.zeroes(u32),
    nb10: u32 = std.mem.zeroes(u32),
    nb11: u32 = std.mem.zeroes(u32),
    nb12: u32 = std.mem.zeroes(u32),
    nb13: u32 = std.mem.zeroes(u32),
    ne20: u32 = std.mem.zeroes(u32),
    ne21: u32 = std.mem.zeroes(u32),
    ne22: u32 = std.mem.zeroes(u32),
    ne23: u32 = std.mem.zeroes(u32),
    nb20: u32 = std.mem.zeroes(u32),
    nb21: u32 = std.mem.zeroes(u32),
    nb22: u32 = std.mem.zeroes(u32),
    nb23: u32 = std.mem.zeroes(u32),
    misalign_offsets: u32 = std.mem.zeroes(u32),
    param1: f32 = std.mem.zeroes(f32),
    param2: f32 = std.mem.zeroes(f32),
    param3: i32 = std.mem.zeroes(i32),
};
comptime {
    std.debug.assert(@sizeOf(Concat) == 116);
    std.debug.assert(@offsetOf(Concat, "ne") == 0);
    std.debug.assert(@offsetOf(Concat, "ne00") == 4);
    std.debug.assert(@offsetOf(Concat, "ne01") == 8);
    std.debug.assert(@offsetOf(Concat, "ne02") == 12);
    std.debug.assert(@offsetOf(Concat, "ne03") == 16);
    std.debug.assert(@offsetOf(Concat, "nb00") == 20);
    std.debug.assert(@offsetOf(Concat, "nb01") == 24);
    std.debug.assert(@offsetOf(Concat, "nb02") == 28);
    std.debug.assert(@offsetOf(Concat, "nb03") == 32);
    std.debug.assert(@offsetOf(Concat, "ne10") == 36);
    std.debug.assert(@offsetOf(Concat, "ne11") == 40);
    std.debug.assert(@offsetOf(Concat, "ne12") == 44);
    std.debug.assert(@offsetOf(Concat, "ne13") == 48);
    std.debug.assert(@offsetOf(Concat, "nb10") == 52);
    std.debug.assert(@offsetOf(Concat, "nb11") == 56);
    std.debug.assert(@offsetOf(Concat, "nb12") == 60);
    std.debug.assert(@offsetOf(Concat, "nb13") == 64);
    std.debug.assert(@offsetOf(Concat, "ne20") == 68);
    std.debug.assert(@offsetOf(Concat, "ne21") == 72);
    std.debug.assert(@offsetOf(Concat, "ne22") == 76);
    std.debug.assert(@offsetOf(Concat, "ne23") == 80);
    std.debug.assert(@offsetOf(Concat, "nb20") == 84);
    std.debug.assert(@offsetOf(Concat, "nb21") == 88);
    std.debug.assert(@offsetOf(Concat, "nb22") == 92);
    std.debug.assert(@offsetOf(Concat, "nb23") == 96);
    std.debug.assert(@offsetOf(Concat, "misalign_offsets") == 100);
    std.debug.assert(@offsetOf(Concat, "param1") == 104);
    std.debug.assert(@offsetOf(Concat, "param2") == 108);
    std.debug.assert(@offsetOf(Concat, "param3") == 112);
}

/// Mirrors `vk_op_multi_add_push_constants` (ggml-vulkan.cpp:1650 @5266f24da).
pub const MultiAdd = extern struct {
    ne20: u32 = std.mem.zeroes(u32),
    ne21: u32 = std.mem.zeroes(u32),
    ne22: u32 = std.mem.zeroes(u32),
    ne23: u32 = std.mem.zeroes(u32),
    nb: [12][4]u32 = std.mem.zeroes([12][4]u32),
    rms_partials: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(MultiAdd) == 212);
    std.debug.assert(@offsetOf(MultiAdd, "ne20") == 0);
    std.debug.assert(@offsetOf(MultiAdd, "ne21") == 4);
    std.debug.assert(@offsetOf(MultiAdd, "ne22") == 8);
    std.debug.assert(@offsetOf(MultiAdd, "ne23") == 12);
    std.debug.assert(@offsetOf(MultiAdd, "nb") == 16);
    std.debug.assert(@offsetOf(MultiAdd, "rms_partials") == 208);
}

/// Mirrors `vk_op_topk_moe_push_constants` (ggml-vulkan.cpp:1663 @5266f24da).
pub const TopkMoe = extern struct {
    n_rows: u32 = std.mem.zeroes(u32),
    n_experts_push: u32 = std.mem.zeroes(u32),
    n_expert_used: u32 = std.mem.zeroes(u32),
    clamp_min: f32 = std.mem.zeroes(f32),
    clamp_max: f32 = std.mem.zeroes(f32),
    gating_func: u32 = std.mem.zeroes(u32),
    has_bias: u32 = std.mem.zeroes(u32),
    with_norm: u32 = std.mem.zeroes(u32),
    output_scale: f32 = std.mem.zeroes(f32),
    output_bias: f32 = std.mem.zeroes(f32),
};
comptime {
    std.debug.assert(@sizeOf(TopkMoe) == 40);
    std.debug.assert(@offsetOf(TopkMoe, "n_rows") == 0);
    std.debug.assert(@offsetOf(TopkMoe, "n_experts_push") == 4);
    std.debug.assert(@offsetOf(TopkMoe, "n_expert_used") == 8);
    std.debug.assert(@offsetOf(TopkMoe, "clamp_min") == 12);
    std.debug.assert(@offsetOf(TopkMoe, "clamp_max") == 16);
    std.debug.assert(@offsetOf(TopkMoe, "gating_func") == 20);
    std.debug.assert(@offsetOf(TopkMoe, "has_bias") == 24);
    std.debug.assert(@offsetOf(TopkMoe, "with_norm") == 28);
    std.debug.assert(@offsetOf(TopkMoe, "output_scale") == 32);
    std.debug.assert(@offsetOf(TopkMoe, "output_bias") == 36);
}

/// Mirrors `vk_op_add_id_push_constants` (ggml-vulkan.cpp:1676 @5266f24da).
pub const AddId = extern struct {
    ne0: u32 = std.mem.zeroes(u32),
    ne1: u32 = std.mem.zeroes(u32),
    s01: u32 = std.mem.zeroes(u32),
    s02: u32 = std.mem.zeroes(u32),
    s11: u32 = std.mem.zeroes(u32),
    s21: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(AddId) == 24);
    std.debug.assert(@offsetOf(AddId, "ne0") == 0);
    std.debug.assert(@offsetOf(AddId, "ne1") == 4);
    std.debug.assert(@offsetOf(AddId, "s01") == 8);
    std.debug.assert(@offsetOf(AddId, "s02") == 12);
    std.debug.assert(@offsetOf(AddId, "s11") == 16);
    std.debug.assert(@offsetOf(AddId, "s21") == 20);
}

/// Mirrors `vk_op_diag_mask_push_constants` (ggml-vulkan.cpp:1685 @5266f24da).
pub const DiagMask = extern struct {
    ncols: u32 = std.mem.zeroes(u32),
    rows_per_channel: u32 = std.mem.zeroes(u32),
    n_past: i32 = std.mem.zeroes(i32),
};
comptime {
    std.debug.assert(@sizeOf(DiagMask) == 12);
    std.debug.assert(@offsetOf(DiagMask, "ncols") == 0);
    std.debug.assert(@offsetOf(DiagMask, "rows_per_channel") == 4);
    std.debug.assert(@offsetOf(DiagMask, "n_past") == 8);
}

/// Mirrors `vk_op_rope_push_constants` (ggml-vulkan.cpp:1691 @5266f24da).
pub const Rope = extern struct {
    rope_mode: u32 = std.mem.zeroes(u32),
    nrows: u32 = std.mem.zeroes(u32),
    n_dims: u32 = std.mem.zeroes(u32),
    n_offs: u32 = std.mem.zeroes(u32),
    freq_scale: f32 = std.mem.zeroes(f32),
    freq_base: f32 = std.mem.zeroes(f32),
    ext_factor: f32 = std.mem.zeroes(f32),
    attn_factor: f32 = std.mem.zeroes(f32),
    corr_dims: [2]f32 = std.mem.zeroes([2]f32),
    theta_scale: f32 = std.mem.zeroes(f32),
    has_ff: u32 = std.mem.zeroes(u32),
    sections: [4]i32 = std.mem.zeroes([4]i32),
    is_imrope: u32 = std.mem.zeroes(u32),
    is_back: u32 = std.mem.zeroes(u32),
    set_rows_stride: u32 = std.mem.zeroes(u32),
    ne00: u32 = std.mem.zeroes(u32),
    ne01: u32 = std.mem.zeroes(u32),
    ne02: u32 = std.mem.zeroes(u32),
    nb01: u32 = std.mem.zeroes(u32),
    nb02: u32 = std.mem.zeroes(u32),
    nb03: u32 = std.mem.zeroes(u32),
    nb11: u32 = std.mem.zeroes(u32),
    nb12: u32 = std.mem.zeroes(u32),
    nb13: u32 = std.mem.zeroes(u32),
    a_offset: u32 = std.mem.zeroes(u32),
    d_offset: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(Rope) == 120);
    std.debug.assert(@offsetOf(Rope, "rope_mode") == 0);
    std.debug.assert(@offsetOf(Rope, "nrows") == 4);
    std.debug.assert(@offsetOf(Rope, "n_dims") == 8);
    std.debug.assert(@offsetOf(Rope, "n_offs") == 12);
    std.debug.assert(@offsetOf(Rope, "freq_scale") == 16);
    std.debug.assert(@offsetOf(Rope, "freq_base") == 20);
    std.debug.assert(@offsetOf(Rope, "ext_factor") == 24);
    std.debug.assert(@offsetOf(Rope, "attn_factor") == 28);
    std.debug.assert(@offsetOf(Rope, "corr_dims") == 32);
    std.debug.assert(@offsetOf(Rope, "theta_scale") == 40);
    std.debug.assert(@offsetOf(Rope, "has_ff") == 44);
    std.debug.assert(@offsetOf(Rope, "sections") == 48);
    std.debug.assert(@offsetOf(Rope, "is_imrope") == 64);
    std.debug.assert(@offsetOf(Rope, "is_back") == 68);
    std.debug.assert(@offsetOf(Rope, "set_rows_stride") == 72);
    std.debug.assert(@offsetOf(Rope, "ne00") == 76);
    std.debug.assert(@offsetOf(Rope, "ne01") == 80);
    std.debug.assert(@offsetOf(Rope, "ne02") == 84);
    std.debug.assert(@offsetOf(Rope, "nb01") == 88);
    std.debug.assert(@offsetOf(Rope, "nb02") == 92);
    std.debug.assert(@offsetOf(Rope, "nb03") == 96);
    std.debug.assert(@offsetOf(Rope, "nb11") == 100);
    std.debug.assert(@offsetOf(Rope, "nb12") == 104);
    std.debug.assert(@offsetOf(Rope, "nb13") == 108);
    std.debug.assert(@offsetOf(Rope, "a_offset") == 112);
    std.debug.assert(@offsetOf(Rope, "d_offset") == 116);
}

/// Mirrors `vk_op_rms_norm_mul_rope_push_constants` (ggml-vulkan.cpp:1722 @5266f24da).
pub const RmsNormMulRope = extern struct {
    bin: Binary = std.mem.zeroes(Binary),
    rope: Rope = std.mem.zeroes(Rope),
};
comptime {
    std.debug.assert(@sizeOf(RmsNormMulRope) == 236);
    std.debug.assert(@offsetOf(RmsNormMulRope, "bin") == 0);
    std.debug.assert(@offsetOf(RmsNormMulRope, "rope") == 116);
}

/// Mirrors `vk_op_soft_max_push_constants` (ggml-vulkan.cpp:1727 @5266f24da).
pub const SoftMax = extern struct {
    KX: u32 = std.mem.zeroes(u32),
    KY: u32 = std.mem.zeroes(u32),
    ne00: u32 = std.mem.zeroes(u32),
    ne01: u32 = std.mem.zeroes(u32),
    ne02: u32 = std.mem.zeroes(u32),
    ne12: u32 = std.mem.zeroes(u32),
    ne13: u32 = std.mem.zeroes(u32),
    nb11: u32 = std.mem.zeroes(u32),
    nb12: u32 = std.mem.zeroes(u32),
    nb13: u32 = std.mem.zeroes(u32),
    scale: f32 = std.mem.zeroes(f32),
    max_bias: f32 = std.mem.zeroes(f32),
    m0: f32 = std.mem.zeroes(f32),
    m1: f32 = std.mem.zeroes(f32),
    n_head_log2: u32 = std.mem.zeroes(u32),
    nrows_x: u32 = std.mem.zeroes(u32),
    has_sinks: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(SoftMax) == 68);
    std.debug.assert(@offsetOf(SoftMax, "KX") == 0);
    std.debug.assert(@offsetOf(SoftMax, "KY") == 4);
    std.debug.assert(@offsetOf(SoftMax, "ne00") == 8);
    std.debug.assert(@offsetOf(SoftMax, "ne01") == 12);
    std.debug.assert(@offsetOf(SoftMax, "ne02") == 16);
    std.debug.assert(@offsetOf(SoftMax, "ne12") == 20);
    std.debug.assert(@offsetOf(SoftMax, "ne13") == 24);
    std.debug.assert(@offsetOf(SoftMax, "nb11") == 28);
    std.debug.assert(@offsetOf(SoftMax, "nb12") == 32);
    std.debug.assert(@offsetOf(SoftMax, "nb13") == 36);
    std.debug.assert(@offsetOf(SoftMax, "scale") == 40);
    std.debug.assert(@offsetOf(SoftMax, "max_bias") == 44);
    std.debug.assert(@offsetOf(SoftMax, "m0") == 48);
    std.debug.assert(@offsetOf(SoftMax, "m1") == 52);
    std.debug.assert(@offsetOf(SoftMax, "n_head_log2") == 56);
    std.debug.assert(@offsetOf(SoftMax, "nrows_x") == 60);
    std.debug.assert(@offsetOf(SoftMax, "has_sinks") == 64);
}

/// Mirrors `vk_op_argsort_push_constants` (ggml-vulkan.cpp:1747 @5266f24da).
pub const Argsort = extern struct {
    ncols: u32 = std.mem.zeroes(u32),
    ncols_padded: u32 = std.mem.zeroes(u32),
    ncols_padded_log2: u32 = std.mem.zeroes(u32),
    nrows: u32 = std.mem.zeroes(u32),
    order: u32 = std.mem.zeroes(u32),
    outer_start: u32 = std.mem.zeroes(u32),
    outer_end: u32 = std.mem.zeroes(u32),
    inner_start: u32 = std.mem.zeroes(u32),
    inner_end: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(Argsort) == 36);
    std.debug.assert(@offsetOf(Argsort, "ncols") == 0);
    std.debug.assert(@offsetOf(Argsort, "ncols_padded") == 4);
    std.debug.assert(@offsetOf(Argsort, "ncols_padded_log2") == 8);
    std.debug.assert(@offsetOf(Argsort, "nrows") == 12);
    std.debug.assert(@offsetOf(Argsort, "order") == 16);
    std.debug.assert(@offsetOf(Argsort, "outer_start") == 20);
    std.debug.assert(@offsetOf(Argsort, "outer_end") == 24);
    std.debug.assert(@offsetOf(Argsort, "inner_start") == 28);
    std.debug.assert(@offsetOf(Argsort, "inner_end") == 32);
}

/// Mirrors `vk_op_topk_push_constants` (ggml-vulkan.cpp:1759 @5266f24da).
pub const Topk = extern struct {
    orig_ncols: u32 = std.mem.zeroes(u32),
    ncols_input: u32 = std.mem.zeroes(u32),
    ncols_output: u32 = std.mem.zeroes(u32),
    k: u32 = std.mem.zeroes(u32),
    nrows: u32 = std.mem.zeroes(u32),
    first_pass: u32 = std.mem.zeroes(u32),
    last_pass: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(Topk) == 28);
    std.debug.assert(@offsetOf(Topk, "orig_ncols") == 0);
    std.debug.assert(@offsetOf(Topk, "ncols_input") == 4);
    std.debug.assert(@offsetOf(Topk, "ncols_output") == 8);
    std.debug.assert(@offsetOf(Topk, "k") == 12);
    std.debug.assert(@offsetOf(Topk, "nrows") == 16);
    std.debug.assert(@offsetOf(Topk, "first_pass") == 20);
    std.debug.assert(@offsetOf(Topk, "last_pass") == 24);
}

/// Mirrors `vk_op_topk_radix_push_constants` (ggml-vulkan.cpp:1769 @5266f24da).
pub const TopkRadix = extern struct {
    ncols: u32 = std.mem.zeroes(u32),
    k: u32 = std.mem.zeroes(u32),
    nrows: u32 = std.mem.zeroes(u32),
    n_tps: u32 = std.mem.zeroes(u32),
    n_blocks: u32 = std.mem.zeroes(u32),
    n_stream: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(TopkRadix) == 24);
    std.debug.assert(@offsetOf(TopkRadix, "ncols") == 0);
    std.debug.assert(@offsetOf(TopkRadix, "k") == 4);
    std.debug.assert(@offsetOf(TopkRadix, "nrows") == 8);
    std.debug.assert(@offsetOf(TopkRadix, "n_tps") == 12);
    std.debug.assert(@offsetOf(TopkRadix, "n_blocks") == 16);
    std.debug.assert(@offsetOf(TopkRadix, "n_stream") == 20);
}

/// Mirrors `vk_op_im2col_push_constants` (ggml-vulkan.cpp:1778 @5266f24da).
pub const Im2col = extern struct {
    dst_addr: u64 = std.mem.zeroes(u64),
    batch_offset: u32 = std.mem.zeroes(u32),
    offset_delta: u32 = std.mem.zeroes(u32),
    IC: u32 = std.mem.zeroes(u32),
    IW: u32 = std.mem.zeroes(u32),
    IH: u32 = std.mem.zeroes(u32),
    OW: u32 = std.mem.zeroes(u32),
    OH: u32 = std.mem.zeroes(u32),
    KW: u32 = std.mem.zeroes(u32),
    KH: u32 = std.mem.zeroes(u32),
    OH_batch: u32 = std.mem.zeroes(u32),
    CHW: u32 = std.mem.zeroes(u32),
    s0: i32 = std.mem.zeroes(i32),
    s1: i32 = std.mem.zeroes(i32),
    p0: i32 = std.mem.zeroes(i32),
    p1: i32 = std.mem.zeroes(i32),
    d0: i32 = std.mem.zeroes(i32),
    d1: i32 = std.mem.zeroes(i32),
    batch_IC: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(Im2col) == 80);
    std.debug.assert(@offsetOf(Im2col, "dst_addr") == 0);
    std.debug.assert(@offsetOf(Im2col, "batch_offset") == 8);
    std.debug.assert(@offsetOf(Im2col, "offset_delta") == 12);
    std.debug.assert(@offsetOf(Im2col, "IC") == 16);
    std.debug.assert(@offsetOf(Im2col, "IW") == 20);
    std.debug.assert(@offsetOf(Im2col, "IH") == 24);
    std.debug.assert(@offsetOf(Im2col, "OW") == 28);
    std.debug.assert(@offsetOf(Im2col, "OH") == 32);
    std.debug.assert(@offsetOf(Im2col, "KW") == 36);
    std.debug.assert(@offsetOf(Im2col, "KH") == 40);
    std.debug.assert(@offsetOf(Im2col, "OH_batch") == 44);
    std.debug.assert(@offsetOf(Im2col, "CHW") == 48);
    std.debug.assert(@offsetOf(Im2col, "s0") == 52);
    std.debug.assert(@offsetOf(Im2col, "s1") == 56);
    std.debug.assert(@offsetOf(Im2col, "p0") == 60);
    std.debug.assert(@offsetOf(Im2col, "p1") == 64);
    std.debug.assert(@offsetOf(Im2col, "d0") == 68);
    std.debug.assert(@offsetOf(Im2col, "d1") == 72);
    std.debug.assert(@offsetOf(Im2col, "batch_IC") == 76);
}

/// Mirrors `vk_op_im2col_3d_push_constants` (ggml-vulkan.cpp:1793 @5266f24da).
pub const Im2col3d = extern struct {
    dst_addr: u64 = std.mem.zeroes(u64),
    nb10: u32 = std.mem.zeroes(u32),
    nb11: u32 = std.mem.zeroes(u32),
    nb12: u32 = std.mem.zeroes(u32),
    nb13: u32 = std.mem.zeroes(u32),
    s0: u32 = std.mem.zeroes(u32),
    s1: u32 = std.mem.zeroes(u32),
    s2: u32 = std.mem.zeroes(u32),
    p0: u32 = std.mem.zeroes(u32),
    p1: u32 = std.mem.zeroes(u32),
    p2: u32 = std.mem.zeroes(u32),
    d0: u32 = std.mem.zeroes(u32),
    d1: u32 = std.mem.zeroes(u32),
    d2: u32 = std.mem.zeroes(u32),
    IW: u32 = std.mem.zeroes(u32),
    IH: u32 = std.mem.zeroes(u32),
    ID: u32 = std.mem.zeroes(u32),
    IC: u32 = std.mem.zeroes(u32),
    KW: u32 = std.mem.zeroes(u32),
    OH: u32 = std.mem.zeroes(u32),
    KD_KH_KW: u32 = std.mem.zeroes(u32),
    KH_KW: u32 = std.mem.zeroes(u32),
    IC_KD_KH_KW: u32 = std.mem.zeroes(u32),
    N_OD_OH: u32 = std.mem.zeroes(u32),
    OD_OH: u32 = std.mem.zeroes(u32),
    OD_OH_OW_IC_KD_KH_KW: u32 = std.mem.zeroes(u32),
    OH_OW_IC_KD_KH_KW: u32 = std.mem.zeroes(u32),
    OW_IC_KD_KH_KW: u32 = std.mem.zeroes(u32),
    misalign_offsets: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(Im2col3d) == 120);
    std.debug.assert(@offsetOf(Im2col3d, "dst_addr") == 0);
    std.debug.assert(@offsetOf(Im2col3d, "nb10") == 8);
    std.debug.assert(@offsetOf(Im2col3d, "nb11") == 12);
    std.debug.assert(@offsetOf(Im2col3d, "nb12") == 16);
    std.debug.assert(@offsetOf(Im2col3d, "nb13") == 20);
    std.debug.assert(@offsetOf(Im2col3d, "s0") == 24);
    std.debug.assert(@offsetOf(Im2col3d, "s1") == 28);
    std.debug.assert(@offsetOf(Im2col3d, "s2") == 32);
    std.debug.assert(@offsetOf(Im2col3d, "p0") == 36);
    std.debug.assert(@offsetOf(Im2col3d, "p1") == 40);
    std.debug.assert(@offsetOf(Im2col3d, "p2") == 44);
    std.debug.assert(@offsetOf(Im2col3d, "d0") == 48);
    std.debug.assert(@offsetOf(Im2col3d, "d1") == 52);
    std.debug.assert(@offsetOf(Im2col3d, "d2") == 56);
    std.debug.assert(@offsetOf(Im2col3d, "IW") == 60);
    std.debug.assert(@offsetOf(Im2col3d, "IH") == 64);
    std.debug.assert(@offsetOf(Im2col3d, "ID") == 68);
    std.debug.assert(@offsetOf(Im2col3d, "IC") == 72);
    std.debug.assert(@offsetOf(Im2col3d, "KW") == 76);
    std.debug.assert(@offsetOf(Im2col3d, "OH") == 80);
    std.debug.assert(@offsetOf(Im2col3d, "KD_KH_KW") == 84);
    std.debug.assert(@offsetOf(Im2col3d, "KH_KW") == 88);
    std.debug.assert(@offsetOf(Im2col3d, "IC_KD_KH_KW") == 92);
    std.debug.assert(@offsetOf(Im2col3d, "N_OD_OH") == 96);
    std.debug.assert(@offsetOf(Im2col3d, "OD_OH") == 100);
    std.debug.assert(@offsetOf(Im2col3d, "OD_OH_OW_IC_KD_KH_KW") == 104);
    std.debug.assert(@offsetOf(Im2col3d, "OH_OW_IC_KD_KH_KW") == 108);
    std.debug.assert(@offsetOf(Im2col3d, "OW_IC_KD_KH_KW") == 112);
    std.debug.assert(@offsetOf(Im2col3d, "misalign_offsets") == 116);
}

/// Mirrors `vk_op_timestep_embedding_push_constants` (ggml-vulkan.cpp:1825 @5266f24da).
pub const TimestepEmbedding = extern struct {
    nb1: u32 = std.mem.zeroes(u32),
    dim: u32 = std.mem.zeroes(u32),
    max_period: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(TimestepEmbedding) == 12);
    std.debug.assert(@offsetOf(TimestepEmbedding, "nb1") == 0);
    std.debug.assert(@offsetOf(TimestepEmbedding, "dim") == 4);
    std.debug.assert(@offsetOf(TimestepEmbedding, "max_period") == 8);
}

/// Mirrors `vk_op_col2im_1d_push_constants` (ggml-vulkan.cpp:1831 @5266f24da).
pub const Col2im1d = extern struct {
    T_out: u32 = std.mem.zeroes(u32),
    OC: u32 = std.mem.zeroes(u32),
    K_OC: u32 = std.mem.zeroes(u32),
    T_in: u32 = std.mem.zeroes(u32),
    K: u32 = std.mem.zeroes(u32),
    stride: i32 = std.mem.zeroes(i32),
    p0: i32 = std.mem.zeroes(i32),
};
comptime {
    std.debug.assert(@sizeOf(Col2im1d) == 28);
    std.debug.assert(@offsetOf(Col2im1d, "T_out") == 0);
    std.debug.assert(@offsetOf(Col2im1d, "OC") == 4);
    std.debug.assert(@offsetOf(Col2im1d, "K_OC") == 8);
    std.debug.assert(@offsetOf(Col2im1d, "T_in") == 12);
    std.debug.assert(@offsetOf(Col2im1d, "K") == 16);
    std.debug.assert(@offsetOf(Col2im1d, "stride") == 20);
    std.debug.assert(@offsetOf(Col2im1d, "p0") == 24);
}

/// Mirrors `vk_op_conv_transpose_1d_push_constants` (ggml-vulkan.cpp:1841 @5266f24da).
pub const ConvTranspose1d = extern struct {
    Cout: u32 = std.mem.zeroes(u32),
    Cin: u32 = std.mem.zeroes(u32),
    K: u32 = std.mem.zeroes(u32),
    L: u32 = std.mem.zeroes(u32),
    KL: u32 = std.mem.zeroes(u32),
    nb01: u32 = std.mem.zeroes(u32),
    nb02: u32 = std.mem.zeroes(u32),
    nb11: u32 = std.mem.zeroes(u32),
    nb1: u32 = std.mem.zeroes(u32),
    s0: i32 = std.mem.zeroes(i32),
};
comptime {
    std.debug.assert(@sizeOf(ConvTranspose1d) == 40);
    std.debug.assert(@offsetOf(ConvTranspose1d, "Cout") == 0);
    std.debug.assert(@offsetOf(ConvTranspose1d, "Cin") == 4);
    std.debug.assert(@offsetOf(ConvTranspose1d, "K") == 8);
    std.debug.assert(@offsetOf(ConvTranspose1d, "L") == 12);
    std.debug.assert(@offsetOf(ConvTranspose1d, "KL") == 16);
    std.debug.assert(@offsetOf(ConvTranspose1d, "nb01") == 20);
    std.debug.assert(@offsetOf(ConvTranspose1d, "nb02") == 24);
    std.debug.assert(@offsetOf(ConvTranspose1d, "nb11") == 28);
    std.debug.assert(@offsetOf(ConvTranspose1d, "nb1") == 32);
    std.debug.assert(@offsetOf(ConvTranspose1d, "s0") == 36);
}

/// Mirrors `vk_op_snake_push_constants` (ggml-vulkan.cpp:1856 @5266f24da).
pub const Snake = extern struct {
    ne0: u32 = std.mem.zeroes(u32),
    ne1: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(Snake) == 8);
    std.debug.assert(@offsetOf(Snake, "ne0") == 0);
    std.debug.assert(@offsetOf(Snake, "ne1") == 4);
}

/// Mirrors `vk_op_pool1d_push_constants` (ggml-vulkan.cpp:1861 @5266f24da).
pub const Pool1d = extern struct {
    IL: u32 = std.mem.zeroes(u32),
    OL: u32 = std.mem.zeroes(u32),
    OC: u32 = std.mem.zeroes(u32),
    pelements: u32 = std.mem.zeroes(u32),
    op: u32 = std.mem.zeroes(u32),
    k0: i32 = std.mem.zeroes(i32),
    s0: i32 = std.mem.zeroes(i32),
    p0: i32 = std.mem.zeroes(i32),
};
comptime {
    std.debug.assert(@sizeOf(Pool1d) == 32);
    std.debug.assert(@offsetOf(Pool1d, "IL") == 0);
    std.debug.assert(@offsetOf(Pool1d, "OL") == 4);
    std.debug.assert(@offsetOf(Pool1d, "OC") == 8);
    std.debug.assert(@offsetOf(Pool1d, "pelements") == 12);
    std.debug.assert(@offsetOf(Pool1d, "op") == 16);
    std.debug.assert(@offsetOf(Pool1d, "k0") == 20);
    std.debug.assert(@offsetOf(Pool1d, "s0") == 24);
    std.debug.assert(@offsetOf(Pool1d, "p0") == 28);
}

/// Mirrors `vk_op_pool2d_push_constants` (ggml-vulkan.cpp:1872 @5266f24da).
pub const Pool2d = extern struct {
    IW: u32 = std.mem.zeroes(u32),
    IH: u32 = std.mem.zeroes(u32),
    OW: u32 = std.mem.zeroes(u32),
    OH: u32 = std.mem.zeroes(u32),
    OC: u32 = std.mem.zeroes(u32),
    pelements: u32 = std.mem.zeroes(u32),
    op: u32 = std.mem.zeroes(u32),
    k0: i32 = std.mem.zeroes(i32),
    k1: i32 = std.mem.zeroes(i32),
    s0: i32 = std.mem.zeroes(i32),
    s1: i32 = std.mem.zeroes(i32),
    p0: i32 = std.mem.zeroes(i32),
    p1: i32 = std.mem.zeroes(i32),
};
comptime {
    std.debug.assert(@sizeOf(Pool2d) == 52);
    std.debug.assert(@offsetOf(Pool2d, "IW") == 0);
    std.debug.assert(@offsetOf(Pool2d, "IH") == 4);
    std.debug.assert(@offsetOf(Pool2d, "OW") == 8);
    std.debug.assert(@offsetOf(Pool2d, "OH") == 12);
    std.debug.assert(@offsetOf(Pool2d, "OC") == 16);
    std.debug.assert(@offsetOf(Pool2d, "pelements") == 20);
    std.debug.assert(@offsetOf(Pool2d, "op") == 24);
    std.debug.assert(@offsetOf(Pool2d, "k0") == 28);
    std.debug.assert(@offsetOf(Pool2d, "k1") == 32);
    std.debug.assert(@offsetOf(Pool2d, "s0") == 36);
    std.debug.assert(@offsetOf(Pool2d, "s1") == 40);
    std.debug.assert(@offsetOf(Pool2d, "p0") == 44);
    std.debug.assert(@offsetOf(Pool2d, "p1") == 48);
}

/// Mirrors `vk_op_rwkv_wkv6_push_constants` (ggml-vulkan.cpp:1883 @5266f24da).
pub const RwkvWkv6 = extern struct {
    B: u32 = std.mem.zeroes(u32),
    T: u32 = std.mem.zeroes(u32),
    C: u32 = std.mem.zeroes(u32),
    H: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(RwkvWkv6) == 16);
    std.debug.assert(@offsetOf(RwkvWkv6, "B") == 0);
    std.debug.assert(@offsetOf(RwkvWkv6, "T") == 4);
    std.debug.assert(@offsetOf(RwkvWkv6, "C") == 8);
    std.debug.assert(@offsetOf(RwkvWkv6, "H") == 12);
}

/// Mirrors `vk_op_rwkv_wkv7_push_constants` (ggml-vulkan.cpp:1890 @5266f24da).
pub const RwkvWkv7 = extern struct {
    B: u32 = std.mem.zeroes(u32),
    T: u32 = std.mem.zeroes(u32),
    C: u32 = std.mem.zeroes(u32),
    H: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(RwkvWkv7) == 16);
    std.debug.assert(@offsetOf(RwkvWkv7, "B") == 0);
    std.debug.assert(@offsetOf(RwkvWkv7, "T") == 4);
    std.debug.assert(@offsetOf(RwkvWkv7, "C") == 8);
    std.debug.assert(@offsetOf(RwkvWkv7, "H") == 12);
}

/// Mirrors `vk_op_gated_linear_attn_push_constants` (ggml-vulkan.cpp:1896 @5266f24da).
pub const GatedLinearAttn = extern struct {
    B: u32 = std.mem.zeroes(u32),
    T: u32 = std.mem.zeroes(u32),
    C: u32 = std.mem.zeroes(u32),
    H: u32 = std.mem.zeroes(u32),
    scale: f32 = std.mem.zeroes(f32),
};
comptime {
    std.debug.assert(@sizeOf(GatedLinearAttn) == 20);
    std.debug.assert(@offsetOf(GatedLinearAttn, "B") == 0);
    std.debug.assert(@offsetOf(GatedLinearAttn, "T") == 4);
    std.debug.assert(@offsetOf(GatedLinearAttn, "C") == 8);
    std.debug.assert(@offsetOf(GatedLinearAttn, "H") == 12);
    std.debug.assert(@offsetOf(GatedLinearAttn, "scale") == 16);
}

/// Mirrors `vk_op_lightning_indexer_push_constants` (ggml-vulkan.cpp:1903 @5266f24da).
pub const LightningIndexer = extern struct {
    n_kv: u32 = std.mem.zeroes(u32),
    n_heads: u32 = std.mem.zeroes(u32),
    n_tokens: u32 = std.mem.zeroes(u32),
    n_streams: u32 = std.mem.zeroes(u32),
    n_masks: u32 = std.mem.zeroes(u32),
    dispatch_x: u32 = std.mem.zeroes(u32),
    q_nb1: u32 = std.mem.zeroes(u32),
    q_nb2: u32 = std.mem.zeroes(u32),
    q_nb3: u32 = std.mem.zeroes(u32),
    k_nb2: u32 = std.mem.zeroes(u32),
    k_nb3: u32 = std.mem.zeroes(u32),
    w_nb1: u32 = std.mem.zeroes(u32),
    w_nb3: u32 = std.mem.zeroes(u32),
    m_nb1: u32 = std.mem.zeroes(u32),
    m_nb3: u32 = std.mem.zeroes(u32),
    d_nb1: u32 = std.mem.zeroes(u32),
    d_nb3: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(LightningIndexer) == 68);
    std.debug.assert(@offsetOf(LightningIndexer, "n_kv") == 0);
    std.debug.assert(@offsetOf(LightningIndexer, "n_heads") == 4);
    std.debug.assert(@offsetOf(LightningIndexer, "n_tokens") == 8);
    std.debug.assert(@offsetOf(LightningIndexer, "n_streams") == 12);
    std.debug.assert(@offsetOf(LightningIndexer, "n_masks") == 16);
    std.debug.assert(@offsetOf(LightningIndexer, "dispatch_x") == 20);
    std.debug.assert(@offsetOf(LightningIndexer, "q_nb1") == 24);
    std.debug.assert(@offsetOf(LightningIndexer, "q_nb2") == 28);
    std.debug.assert(@offsetOf(LightningIndexer, "q_nb3") == 32);
    std.debug.assert(@offsetOf(LightningIndexer, "k_nb2") == 36);
    std.debug.assert(@offsetOf(LightningIndexer, "k_nb3") == 40);
    std.debug.assert(@offsetOf(LightningIndexer, "w_nb1") == 44);
    std.debug.assert(@offsetOf(LightningIndexer, "w_nb3") == 48);
    std.debug.assert(@offsetOf(LightningIndexer, "m_nb1") == 52);
    std.debug.assert(@offsetOf(LightningIndexer, "m_nb3") == 56);
    std.debug.assert(@offsetOf(LightningIndexer, "d_nb1") == 60);
    std.debug.assert(@offsetOf(LightningIndexer, "d_nb3") == 64);
}

/// Mirrors `vk_op_gated_delta_net_push_constants` (ggml-vulkan.cpp:1923 @5266f24da).
pub const GatedDeltaNet = extern struct {
    H: u32 = std.mem.zeroes(u32),
    n_tokens: u32 = std.mem.zeroes(u32),
    n_seqs: u32 = std.mem.zeroes(u32),
    s_off: u32 = std.mem.zeroes(u32),
    sq1: u32 = std.mem.zeroes(u32),
    sq2: u32 = std.mem.zeroes(u32),
    sq3: u32 = std.mem.zeroes(u32),
    sv1: u32 = std.mem.zeroes(u32),
    sv2: u32 = std.mem.zeroes(u32),
    sv3: u32 = std.mem.zeroes(u32),
    sb1: u32 = std.mem.zeroes(u32),
    sb2: u32 = std.mem.zeroes(u32),
    sb3: u32 = std.mem.zeroes(u32),
    neq1: u32 = std.mem.zeroes(u32),
    rq3: u32 = std.mem.zeroes(u32),
    scale: f32 = std.mem.zeroes(f32),
    K: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(GatedDeltaNet) == 68);
    std.debug.assert(@offsetOf(GatedDeltaNet, "H") == 0);
    std.debug.assert(@offsetOf(GatedDeltaNet, "n_tokens") == 4);
    std.debug.assert(@offsetOf(GatedDeltaNet, "n_seqs") == 8);
    std.debug.assert(@offsetOf(GatedDeltaNet, "s_off") == 12);
    std.debug.assert(@offsetOf(GatedDeltaNet, "sq1") == 16);
    std.debug.assert(@offsetOf(GatedDeltaNet, "sq2") == 20);
    std.debug.assert(@offsetOf(GatedDeltaNet, "sq3") == 24);
    std.debug.assert(@offsetOf(GatedDeltaNet, "sv1") == 28);
    std.debug.assert(@offsetOf(GatedDeltaNet, "sv2") == 32);
    std.debug.assert(@offsetOf(GatedDeltaNet, "sv3") == 36);
    std.debug.assert(@offsetOf(GatedDeltaNet, "sb1") == 40);
    std.debug.assert(@offsetOf(GatedDeltaNet, "sb2") == 44);
    std.debug.assert(@offsetOf(GatedDeltaNet, "sb3") == 48);
    std.debug.assert(@offsetOf(GatedDeltaNet, "neq1") == 52);
    std.debug.assert(@offsetOf(GatedDeltaNet, "rq3") == 56);
    std.debug.assert(@offsetOf(GatedDeltaNet, "scale") == 60);
    std.debug.assert(@offsetOf(GatedDeltaNet, "K") == 64);
}

/// Mirrors `vk_op_ssm_scan_push_constants` (ggml-vulkan.cpp:1936 @5266f24da).
pub const SsmScan = extern struct {
    nb02: u32 = std.mem.zeroes(u32),
    nb03: u32 = std.mem.zeroes(u32),
    nb12: u32 = std.mem.zeroes(u32),
    nb13: u32 = std.mem.zeroes(u32),
    nb21: u32 = std.mem.zeroes(u32),
    nb22: u32 = std.mem.zeroes(u32),
    nb31: u32 = std.mem.zeroes(u32),
    nb42: u32 = std.mem.zeroes(u32),
    nb43: u32 = std.mem.zeroes(u32),
    nb52: u32 = std.mem.zeroes(u32),
    nb53: u32 = std.mem.zeroes(u32),
    s_off: u32 = std.mem.zeroes(u32),
    n_head: u32 = std.mem.zeroes(u32),
    d_head: u32 = std.mem.zeroes(u32),
    n_group: u32 = std.mem.zeroes(u32),
    n_tok: u32 = std.mem.zeroes(u32),
    n_seq: u32 = std.mem.zeroes(u32),
    K: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(SsmScan) == 72);
    std.debug.assert(@offsetOf(SsmScan, "nb02") == 0);
    std.debug.assert(@offsetOf(SsmScan, "nb03") == 4);
    std.debug.assert(@offsetOf(SsmScan, "nb12") == 8);
    std.debug.assert(@offsetOf(SsmScan, "nb13") == 12);
    std.debug.assert(@offsetOf(SsmScan, "nb21") == 16);
    std.debug.assert(@offsetOf(SsmScan, "nb22") == 20);
    std.debug.assert(@offsetOf(SsmScan, "nb31") == 24);
    std.debug.assert(@offsetOf(SsmScan, "nb42") == 28);
    std.debug.assert(@offsetOf(SsmScan, "nb43") == 32);
    std.debug.assert(@offsetOf(SsmScan, "nb52") == 36);
    std.debug.assert(@offsetOf(SsmScan, "nb53") == 40);
    std.debug.assert(@offsetOf(SsmScan, "s_off") == 44);
    std.debug.assert(@offsetOf(SsmScan, "n_head") == 48);
    std.debug.assert(@offsetOf(SsmScan, "d_head") == 52);
    std.debug.assert(@offsetOf(SsmScan, "n_group") == 56);
    std.debug.assert(@offsetOf(SsmScan, "n_tok") == 60);
    std.debug.assert(@offsetOf(SsmScan, "n_seq") == 64);
    std.debug.assert(@offsetOf(SsmScan, "K") == 68);
}

/// Mirrors `vk_op_ssm_conv_push_constants` (ggml-vulkan.cpp:1944 @5266f24da).
pub const SsmConv = extern struct {
    nb01: u32 = std.mem.zeroes(u32),
    nb02: u32 = std.mem.zeroes(u32),
    nb11: u32 = std.mem.zeroes(u32),
    dst_nb0: u32 = std.mem.zeroes(u32),
    dst_nb1: u32 = std.mem.zeroes(u32),
    dst_nb2: u32 = std.mem.zeroes(u32),
    nc: u32 = std.mem.zeroes(u32),
    ncs: u32 = std.mem.zeroes(u32),
    nr: u32 = std.mem.zeroes(u32),
    n_t: u32 = std.mem.zeroes(u32),
    n_s: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(SsmConv) == 44);
    std.debug.assert(@offsetOf(SsmConv, "nb01") == 0);
    std.debug.assert(@offsetOf(SsmConv, "nb02") == 4);
    std.debug.assert(@offsetOf(SsmConv, "nb11") == 8);
    std.debug.assert(@offsetOf(SsmConv, "dst_nb0") == 12);
    std.debug.assert(@offsetOf(SsmConv, "dst_nb1") == 16);
    std.debug.assert(@offsetOf(SsmConv, "dst_nb2") == 20);
    std.debug.assert(@offsetOf(SsmConv, "nc") == 24);
    std.debug.assert(@offsetOf(SsmConv, "ncs") == 28);
    std.debug.assert(@offsetOf(SsmConv, "nr") == 32);
    std.debug.assert(@offsetOf(SsmConv, "n_t") == 36);
    std.debug.assert(@offsetOf(SsmConv, "n_s") == 40);
}

/// Mirrors `vk_op_conv2d_push_constants` (ggml-vulkan.cpp:1951 @5266f24da).
pub const Conv2d = extern struct {
    Cout: u32 = std.mem.zeroes(u32),
    Cin: u32 = std.mem.zeroes(u32),
    N: u32 = std.mem.zeroes(u32),
    W: u32 = std.mem.zeroes(u32),
    H: u32 = std.mem.zeroes(u32),
    OW: u32 = std.mem.zeroes(u32),
    OH: u32 = std.mem.zeroes(u32),
    nb01: u32 = std.mem.zeroes(u32),
    nb02: u32 = std.mem.zeroes(u32),
    nb03: u32 = std.mem.zeroes(u32),
    nb11: u32 = std.mem.zeroes(u32),
    nb12: u32 = std.mem.zeroes(u32),
    nb13: u32 = std.mem.zeroes(u32),
    nb1: u32 = std.mem.zeroes(u32),
    nb2: u32 = std.mem.zeroes(u32),
    nb3: u32 = std.mem.zeroes(u32),
    OWmp: u32 = std.mem.zeroes(u32),
    OWL: u32 = std.mem.zeroes(u32),
    OWOHmp: u32 = std.mem.zeroes(u32),
    OWOHL: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(Conv2d) == 80);
    std.debug.assert(@offsetOf(Conv2d, "Cout") == 0);
    std.debug.assert(@offsetOf(Conv2d, "Cin") == 4);
    std.debug.assert(@offsetOf(Conv2d, "N") == 8);
    std.debug.assert(@offsetOf(Conv2d, "W") == 12);
    std.debug.assert(@offsetOf(Conv2d, "H") == 16);
    std.debug.assert(@offsetOf(Conv2d, "OW") == 20);
    std.debug.assert(@offsetOf(Conv2d, "OH") == 24);
    std.debug.assert(@offsetOf(Conv2d, "nb01") == 28);
    std.debug.assert(@offsetOf(Conv2d, "nb02") == 32);
    std.debug.assert(@offsetOf(Conv2d, "nb03") == 36);
    std.debug.assert(@offsetOf(Conv2d, "nb11") == 40);
    std.debug.assert(@offsetOf(Conv2d, "nb12") == 44);
    std.debug.assert(@offsetOf(Conv2d, "nb13") == 48);
    std.debug.assert(@offsetOf(Conv2d, "nb1") == 52);
    std.debug.assert(@offsetOf(Conv2d, "nb2") == 56);
    std.debug.assert(@offsetOf(Conv2d, "nb3") == 60);
    std.debug.assert(@offsetOf(Conv2d, "OWmp") == 64);
    std.debug.assert(@offsetOf(Conv2d, "OWL") == 68);
    std.debug.assert(@offsetOf(Conv2d, "OWOHmp") == 72);
    std.debug.assert(@offsetOf(Conv2d, "OWOHL") == 76);
}

/// Mirrors `vk_op_conv3d_push_constants` (ggml-vulkan.cpp:1984 @5266f24da).
pub const Conv3d = extern struct {
    OC: u32 = std.mem.zeroes(u32),
    IC: u32 = std.mem.zeroes(u32),
    N: u32 = std.mem.zeroes(u32),
    IW: u32 = std.mem.zeroes(u32),
    IH: u32 = std.mem.zeroes(u32),
    ID: u32 = std.mem.zeroes(u32),
    OW: u32 = std.mem.zeroes(u32),
    OH: u32 = std.mem.zeroes(u32),
    OD: u32 = std.mem.zeroes(u32),
    nb01: u32 = std.mem.zeroes(u32),
    nb02: u32 = std.mem.zeroes(u32),
    nb03: u32 = std.mem.zeroes(u32),
    nb11: u32 = std.mem.zeroes(u32),
    nb12: u32 = std.mem.zeroes(u32),
    nb13: u32 = std.mem.zeroes(u32),
    nb1: u32 = std.mem.zeroes(u32),
    nb2: u32 = std.mem.zeroes(u32),
    nb3: u32 = std.mem.zeroes(u32),
    OWmp: u32 = std.mem.zeroes(u32),
    OWL: u32 = std.mem.zeroes(u32),
    OWOHmp: u32 = std.mem.zeroes(u32),
    OWOHL: u32 = std.mem.zeroes(u32),
    OWOHODmp: u32 = std.mem.zeroes(u32),
    OWOHODL: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(Conv3d) == 96);
    std.debug.assert(@offsetOf(Conv3d, "OC") == 0);
    std.debug.assert(@offsetOf(Conv3d, "IC") == 4);
    std.debug.assert(@offsetOf(Conv3d, "N") == 8);
    std.debug.assert(@offsetOf(Conv3d, "IW") == 12);
    std.debug.assert(@offsetOf(Conv3d, "IH") == 16);
    std.debug.assert(@offsetOf(Conv3d, "ID") == 20);
    std.debug.assert(@offsetOf(Conv3d, "OW") == 24);
    std.debug.assert(@offsetOf(Conv3d, "OH") == 28);
    std.debug.assert(@offsetOf(Conv3d, "OD") == 32);
    std.debug.assert(@offsetOf(Conv3d, "nb01") == 36);
    std.debug.assert(@offsetOf(Conv3d, "nb02") == 40);
    std.debug.assert(@offsetOf(Conv3d, "nb03") == 44);
    std.debug.assert(@offsetOf(Conv3d, "nb11") == 48);
    std.debug.assert(@offsetOf(Conv3d, "nb12") == 52);
    std.debug.assert(@offsetOf(Conv3d, "nb13") == 56);
    std.debug.assert(@offsetOf(Conv3d, "nb1") == 60);
    std.debug.assert(@offsetOf(Conv3d, "nb2") == 64);
    std.debug.assert(@offsetOf(Conv3d, "nb3") == 68);
    std.debug.assert(@offsetOf(Conv3d, "OWmp") == 72);
    std.debug.assert(@offsetOf(Conv3d, "OWL") == 76);
    std.debug.assert(@offsetOf(Conv3d, "OWOHmp") == 80);
    std.debug.assert(@offsetOf(Conv3d, "OWOHL") == 84);
    std.debug.assert(@offsetOf(Conv3d, "OWOHODmp") == 88);
    std.debug.assert(@offsetOf(Conv3d, "OWOHODL") == 92);
}

/// Mirrors `vk_op_conv2d_dw_push_constants` (ggml-vulkan.cpp:2019 @5266f24da).
pub const Conv2dDw = extern struct {
    ne: u32 = std.mem.zeroes(u32),
    batches: u32 = std.mem.zeroes(u32),
    channels: u32 = std.mem.zeroes(u32),
    dst_w: u32 = std.mem.zeroes(u32),
    dst_h: u32 = std.mem.zeroes(u32),
    src_w: u32 = std.mem.zeroes(u32),
    src_h: u32 = std.mem.zeroes(u32),
    knl_w: u32 = std.mem.zeroes(u32),
    knl_h: u32 = std.mem.zeroes(u32),
    stride_x: i32 = std.mem.zeroes(i32),
    stride_y: i32 = std.mem.zeroes(i32),
    pad_x: i32 = std.mem.zeroes(i32),
    pad_y: i32 = std.mem.zeroes(i32),
    dilation_x: i32 = std.mem.zeroes(i32),
    dilation_y: i32 = std.mem.zeroes(i32),
};
comptime {
    std.debug.assert(@sizeOf(Conv2dDw) == 60);
    std.debug.assert(@offsetOf(Conv2dDw, "ne") == 0);
    std.debug.assert(@offsetOf(Conv2dDw, "batches") == 4);
    std.debug.assert(@offsetOf(Conv2dDw, "channels") == 8);
    std.debug.assert(@offsetOf(Conv2dDw, "dst_w") == 12);
    std.debug.assert(@offsetOf(Conv2dDw, "dst_h") == 16);
    std.debug.assert(@offsetOf(Conv2dDw, "src_w") == 20);
    std.debug.assert(@offsetOf(Conv2dDw, "src_h") == 24);
    std.debug.assert(@offsetOf(Conv2dDw, "knl_w") == 28);
    std.debug.assert(@offsetOf(Conv2dDw, "knl_h") == 32);
    std.debug.assert(@offsetOf(Conv2dDw, "stride_x") == 36);
    std.debug.assert(@offsetOf(Conv2dDw, "stride_y") == 40);
    std.debug.assert(@offsetOf(Conv2dDw, "pad_x") == 44);
    std.debug.assert(@offsetOf(Conv2dDw, "pad_y") == 48);
    std.debug.assert(@offsetOf(Conv2dDw, "dilation_x") == 52);
    std.debug.assert(@offsetOf(Conv2dDw, "dilation_y") == 56);
}

/// Mirrors `vk_op_upscale_push_constants` (ggml-vulkan.cpp:2037 @5266f24da).
pub const Upscale = extern struct {
    ne: u32 = std.mem.zeroes(u32),
    a_offset: u32 = std.mem.zeroes(u32),
    d_offset: u32 = std.mem.zeroes(u32),
    ne00: u32 = std.mem.zeroes(u32),
    ne01: u32 = std.mem.zeroes(u32),
    nb00: u32 = std.mem.zeroes(u32),
    nb01: u32 = std.mem.zeroes(u32),
    nb02: u32 = std.mem.zeroes(u32),
    nb03: u32 = std.mem.zeroes(u32),
    ne10: u32 = std.mem.zeroes(u32),
    ne11: u32 = std.mem.zeroes(u32),
    ne12: u32 = std.mem.zeroes(u32),
    ne13: u32 = std.mem.zeroes(u32),
    sf0: f32 = std.mem.zeroes(f32),
    sf1: f32 = std.mem.zeroes(f32),
    sf2: f32 = std.mem.zeroes(f32),
    sf3: f32 = std.mem.zeroes(f32),
    pixel_offset: f32 = std.mem.zeroes(f32),
};
comptime {
    std.debug.assert(@sizeOf(Upscale) == 72);
    std.debug.assert(@offsetOf(Upscale, "ne") == 0);
    std.debug.assert(@offsetOf(Upscale, "a_offset") == 4);
    std.debug.assert(@offsetOf(Upscale, "d_offset") == 8);
    std.debug.assert(@offsetOf(Upscale, "ne00") == 12);
    std.debug.assert(@offsetOf(Upscale, "ne01") == 16);
    std.debug.assert(@offsetOf(Upscale, "nb00") == 20);
    std.debug.assert(@offsetOf(Upscale, "nb01") == 24);
    std.debug.assert(@offsetOf(Upscale, "nb02") == 28);
    std.debug.assert(@offsetOf(Upscale, "nb03") == 32);
    std.debug.assert(@offsetOf(Upscale, "ne10") == 36);
    std.debug.assert(@offsetOf(Upscale, "ne11") == 40);
    std.debug.assert(@offsetOf(Upscale, "ne12") == 44);
    std.debug.assert(@offsetOf(Upscale, "ne13") == 48);
    std.debug.assert(@offsetOf(Upscale, "sf0") == 52);
    std.debug.assert(@offsetOf(Upscale, "sf1") == 56);
    std.debug.assert(@offsetOf(Upscale, "sf2") == 60);
    std.debug.assert(@offsetOf(Upscale, "sf3") == 64);
    std.debug.assert(@offsetOf(Upscale, "pixel_offset") == 68);
}

/// Mirrors `vk_op_sum_rows_push_constants` (ggml-vulkan.cpp:2046 @5266f24da).
pub const SumRows = extern struct {
    n_cols: u32 = std.mem.zeroes(u32),
    ne01: u32 = std.mem.zeroes(u32),
    ne02: u32 = std.mem.zeroes(u32),
    nb01: u32 = std.mem.zeroes(u32),
    nb02: u32 = std.mem.zeroes(u32),
    nb03: u32 = std.mem.zeroes(u32),
    nb11: u32 = std.mem.zeroes(u32),
    nb12: u32 = std.mem.zeroes(u32),
    nb13: u32 = std.mem.zeroes(u32),
    weight: f32 = std.mem.zeroes(f32),
    misalign_offsets: u32 = std.mem.zeroes(u32),
    ne0_12mp: u32 = std.mem.zeroes(u32),
    ne0_12L: u32 = std.mem.zeroes(u32),
    ne0_1mp: u32 = std.mem.zeroes(u32),
    ne0_1L: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(SumRows) == 60);
    std.debug.assert(@offsetOf(SumRows, "n_cols") == 0);
    std.debug.assert(@offsetOf(SumRows, "ne01") == 4);
    std.debug.assert(@offsetOf(SumRows, "ne02") == 8);
    std.debug.assert(@offsetOf(SumRows, "nb01") == 12);
    std.debug.assert(@offsetOf(SumRows, "nb02") == 16);
    std.debug.assert(@offsetOf(SumRows, "nb03") == 20);
    std.debug.assert(@offsetOf(SumRows, "nb11") == 24);
    std.debug.assert(@offsetOf(SumRows, "nb12") == 28);
    std.debug.assert(@offsetOf(SumRows, "nb13") == 32);
    std.debug.assert(@offsetOf(SumRows, "weight") == 36);
    std.debug.assert(@offsetOf(SumRows, "misalign_offsets") == 40);
    std.debug.assert(@offsetOf(SumRows, "ne0_12mp") == 44);
    std.debug.assert(@offsetOf(SumRows, "ne0_12L") == 48);
    std.debug.assert(@offsetOf(SumRows, "ne0_1mp") == 52);
    std.debug.assert(@offsetOf(SumRows, "ne0_1L") == 56);
}

/// Mirrors `vk_quantize_q8_1_push_constants` (ggml-vulkan.cpp:2079 @5266f24da).
pub const QuantizeQ81 = extern struct {
    ne: u32 = std.mem.zeroes(u32),
    num_blocks: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(QuantizeQ81) == 8);
    std.debug.assert(@offsetOf(QuantizeQ81, "ne") == 0);
    std.debug.assert(@offsetOf(QuantizeQ81, "num_blocks") == 4);
}

/// Mirrors `vk_op_flash_attn_split_k_reduce_push_constants` (ggml-vulkan.cpp:2084 @5266f24da).
pub const FlashAttnSplitKReduce = extern struct {
    D: u32 = std.mem.zeroes(u32),
    ne1: u32 = std.mem.zeroes(u32),
    ne2: u32 = std.mem.zeroes(u32),
    ne3: u32 = std.mem.zeroes(u32),
    k_num: u32 = std.mem.zeroes(u32),
    sinks: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(FlashAttnSplitKReduce) == 24);
    std.debug.assert(@offsetOf(FlashAttnSplitKReduce, "D") == 0);
    std.debug.assert(@offsetOf(FlashAttnSplitKReduce, "ne1") == 4);
    std.debug.assert(@offsetOf(FlashAttnSplitKReduce, "ne2") == 8);
    std.debug.assert(@offsetOf(FlashAttnSplitKReduce, "ne3") == 12);
    std.debug.assert(@offsetOf(FlashAttnSplitKReduce, "k_num") == 16);
    std.debug.assert(@offsetOf(FlashAttnSplitKReduce, "sinks") == 20);
}

/// Mirrors `vk_op_flash_attn_mask_opt_push_constants` (ggml-vulkan.cpp:2093 @5266f24da).
pub const FlashAttnMaskOpt = extern struct {
    nem0: u32 = std.mem.zeroes(u32),
    nem1: u32 = std.mem.zeroes(u32),
    nem2: u32 = std.mem.zeroes(u32),
    nbm1: u32 = std.mem.zeroes(u32),
    nbm2: u32 = std.mem.zeroes(u32),
    nbm3: u32 = std.mem.zeroes(u32),
    nbd1: u32 = std.mem.zeroes(u32),
    nbd2: u32 = std.mem.zeroes(u32),
    nbd3: u32 = std.mem.zeroes(u32),
};
comptime {
    std.debug.assert(@sizeOf(FlashAttnMaskOpt) == 36);
    std.debug.assert(@offsetOf(FlashAttnMaskOpt, "nem0") == 0);
    std.debug.assert(@offsetOf(FlashAttnMaskOpt, "nem1") == 4);
    std.debug.assert(@offsetOf(FlashAttnMaskOpt, "nem2") == 8);
    std.debug.assert(@offsetOf(FlashAttnMaskOpt, "nbm1") == 12);
    std.debug.assert(@offsetOf(FlashAttnMaskOpt, "nbm2") == 16);
    std.debug.assert(@offsetOf(FlashAttnMaskOpt, "nbm3") == 20);
    std.debug.assert(@offsetOf(FlashAttnMaskOpt, "nbd1") == 24);
    std.debug.assert(@offsetOf(FlashAttnMaskOpt, "nbd2") == 28);
    std.debug.assert(@offsetOf(FlashAttnMaskOpt, "nbd3") == 32);
}
