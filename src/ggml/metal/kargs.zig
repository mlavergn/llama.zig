//! The argument blocks the Metal kernels read.
//!
//! # Provenance
//!
//! Mirrors the `ggml_metal_kargs_*` structs of
//! `llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h` at v0.3.0
//! (`c1d0e7a00`).
//!
//! # Generated, not written
//!
//! `ggml-metal-ops.cpp` fills one of these per op and hands it to a kernel
//! as a block of bytes, which the kernel reads **by offset**. A field at
//! the wrong offset therefore feeds the kernel a different number, and
//! there is nothing in the ported code that could notice -- no assertion
//! it violates, no pointer it invalidates.
//!
//! So this file is produced by `scripts/gen-kargs` from the header, and
//! `make struct-layout` compares every struct's size and alignment and
//! every field's offset and width against the real header. Do not edit it
//! by hand; change the generator and re-run it.
//!
//! The header cannot simply be imported: only `impl.zig`'s single
//! `cImport` may hold it without creating two incompatible `*ggml_tensor`,
//! and putting it there widens the `c` namespace every ported file sees.
//! See `device_c.zig` for that reasoning in full.

const std = @import("std");

/// Mirrors `ggml_metal_kargs_concat` (ggml-metal-impl.h:168 @c1d0e7a00).
pub const concat = extern struct {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    dim: i32,
};

/// Mirrors `ggml_metal_kargs_unary` (ggml-metal-impl.h:196 @c1d0e7a00).
pub const unary = extern struct {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    slope: f32,
    scale: f32,
    bias: f32,
    val: f32,
    min: f32,
    max: f32,
};

/// Mirrors `ggml_metal_kargs_bin` (ggml-metal-impl.h:221 @c1d0e7a00).
pub const bin = extern struct {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    offs: u64,
    o1: [8]u64,
};

/// Mirrors `ggml_metal_kargs_add_id` (ggml-metal-impl.h:250 @c1d0e7a00).
pub const add_id = extern struct {
    ne0: i64,
    ne1: i64,
    nb01: usize,
    nb02: usize,
    nb11: usize,
    nb21: usize,
};

/// Mirrors `ggml_metal_kargs_repeat` (ggml-metal-impl.h:259 @c1d0e7a00).
pub const repeat = extern struct {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
};

/// Mirrors `ggml_metal_kargs_cpy` (ggml-metal-impl.h:278 @c1d0e7a00).
pub const cpy = extern struct {
    nk0: i64,
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
};

/// Mirrors `ggml_metal_kargs_set` (ggml-metal-impl.h:298 @c1d0e7a00).
pub const set = extern struct {
    ne10: i64,
    ne11: i64,
    ne12: i64,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    offs: u64,
    inplace: bool,
};

/// Mirrors `ggml_metal_kargs_rope` (ggml-metal-impl.h:313 @c1d0e7a00).
pub const rope = extern struct {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    n_past: i32,
    n_dims: i32,
    n_offs: i32,
    n_ctx_orig: i32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    sect_0: i32,
    sect_1: i32,
    sect_2: i32,
    sect_3: i32,
    src2: bool,
    inplace: bool,
};

/// Mirrors `ggml_metal_kargs_flash_attn_ext_kv_f16` (ggml-metal-impl.h:348 @c1d0e7a00).
pub const flash_attn_ext_kv_f16 = extern struct {
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    nblocks: i32,
};

/// Mirrors `ggml_metal_kargs_flash_attn_ext_pad` (ggml-metal-impl.h:360 @c1d0e7a00).
pub const flash_attn_ext_pad = extern struct {
    ne11: i32,
    ne_12_2: i32,
    ne_12_3: i32,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    nb21: u64,
    nb22: u64,
    nb23: u64,
    ne31: i32,
    ne32: i32,
    ne33: i32,
    nb31: u64,
    nb32: u64,
    nb33: u64,
};

/// Mirrors `ggml_metal_kargs_flash_attn_ext_blk` (ggml-metal-impl.h:378 @c1d0e7a00).
pub const flash_attn_ext_blk = extern struct {
    ne01: i32,
    ne30: i32,
    ne31: i32,
    ne32: i32,
    ne33: i32,
    nb31: u64,
    nb32: u64,
    nb33: u64,
};

/// Mirrors `ggml_metal_kargs_flash_attn_ext` (ggml-metal-impl.h:389 @c1d0e7a00).
pub const flash_attn_ext = extern struct {
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne11: i32,
    ne_12_2: i32,
    ne_12_3: i32,
    ns10: i32,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ns20: i32,
    nb21: u64,
    nb22: u64,
    nb23: u64,
    ne31: i32,
    ne32: i32,
    ne33: i32,
    nb31: u64,
    nb32: u64,
    nb33: u64,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    scale: f32,
    max_bias: f32,
    m0: f32,
    m1: f32,
    n_head_log2: i32,
    logit_softcap: f32,
};

/// Mirrors `ggml_metal_kargs_flash_attn_ext_vec` (ggml-metal-impl.h:424 @c1d0e7a00).
pub const flash_attn_ext_vec = extern struct {
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne11: i32,
    ne_12_2: i32,
    ne_12_3: i32,
    ns10: i32,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ns20: i32,
    nb21: u64,
    nb22: u64,
    nb23: u64,
    ne31: i32,
    ne32: i32,
    ne33: i32,
    nb31: u64,
    nb32: u64,
    nb33: u64,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    scale: f32,
    max_bias: f32,
    m0: f32,
    m1: f32,
    n_head_log2: i32,
    logit_softcap: f32,
};

/// Mirrors `ggml_metal_kargs_flash_attn_ext_vec_reduce` (ggml-metal-impl.h:459 @c1d0e7a00).
pub const flash_attn_ext_vec_reduce = extern struct {
    nrows: i32,
};

/// Mirrors `ggml_metal_kargs_mul_mm` (ggml-metal-impl.h:463 @c1d0e7a00).
pub const mul_mm = extern struct {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
};

/// Mirrors `ggml_metal_kargs_mul_mv` (ggml-metal-impl.h:480 @c1d0e7a00).
pub const mul_mv = extern struct {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    nr0: i32,
    r2: i16,
    r3: i16,
};

/// Mirrors `ggml_metal_kargs_mul_mv_ext` (ggml-metal-impl.h:502 @c1d0e7a00).
pub const mul_mv_ext = extern struct {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
};

/// Mirrors `ggml_metal_kargs_mul_mm_id_map0` (ggml-metal-impl.h:523 @c1d0e7a00).
pub const mul_mm_id_map0 = extern struct {
    ne02: i32,
    ne10: i32,
    ne11: i32,
    nb11: u64,
    nb12: u64,
    ne21: i32,
    ne20: i32,
    nb21: u64,
};

/// Mirrors `ggml_metal_kargs_mul_mm_id` (ggml-metal-impl.h:534 @c1d0e7a00).
pub const mul_mm_id = extern struct {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne11: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne20: i32,
    ne21: i32,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
};

/// Mirrors `ggml_metal_kargs_mul_mv_id` (ggml-metal-impl.h:553 @c1d0e7a00).
pub const mul_mv_id = extern struct {
    nei0: i32,
    nei1: i32,
    nbi1: u64,
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    ne0: i32,
    ne1: i32,
    nb1: u64,
    nr0: i32,
};

/// Mirrors `ggml_metal_kargs_norm` (ggml-metal-impl.h:578 @c1d0e7a00).
pub const norm = extern struct {
    ne00: i32,
    ne00_t: i32,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    eps: f32,
    nef1: [3]i32,
    nef2: [3]i32,
    nef3: [3]i32,
    nbf1: [3]u64,
    nbf2: [3]u64,
    nbf3: [3]u64,
};

/// Mirrors `ggml_metal_kargs_l2_norm` (ggml-metal-impl.h:593 @c1d0e7a00).
pub const l2_norm = extern struct {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    eps: f32,
};

/// Mirrors `ggml_metal_kargs_group_norm` (ggml-metal-impl.h:613 @c1d0e7a00).
pub const group_norm = extern struct {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    ngrp: i32,
    eps: f32,
};

/// Mirrors `ggml_metal_kargs_conv_transpose_1d` (ggml-metal-impl.h:624 @c1d0e7a00).
pub const conv_transpose_1d = extern struct {
    IC: i32,
    IL: i32,
    K: i32,
    s0: i32,
    nb0: u64,
    nb1: u64,
};

/// Mirrors `ggml_metal_kargs_col2im_1d` (ggml-metal-impl.h:633 @c1d0e7a00).
pub const col2im_1d = extern struct {
    T_in: i32,
    T_out: i32,
    OC: i32,
    K: i32,
    K_OC: i32,
    s0: i32,
    p0: i32,
};

/// Mirrors `ggml_metal_kargs_snake` (ggml-metal-impl.h:643 @c1d0e7a00).
pub const snake = extern struct {
    T: i32,
    C: i32,
};

/// Mirrors `ggml_metal_kargs_conv_transpose_2d` (ggml-metal-impl.h:648 @c1d0e7a00).
pub const conv_transpose_2d = extern struct {
    IC: i32,
    IH: i32,
    IW: i32,
    KH: i32,
    KW: i32,
    OC: i32,
    s0: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
};

/// Mirrors `ggml_metal_kargs_conv_2d` (ggml-metal-impl.h:661 @c1d0e7a00).
pub const conv_2d = extern struct {
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    IW: i32,
    IH: i32,
    KW: i32,
    KH: i32,
    IC: i32,
    OC: i32,
    OW: i32,
    OH: i32,
    N: i32,
    s0: i32,
    s1: i32,
    p0: i32,
    p1: i32,
    d0: i32,
    d1: i32,
};

/// Mirrors `ggml_metal_kargs_conv_2d_dw` (ggml-metal-impl.h:691 @c1d0e7a00).
pub const conv_2d_dw = extern struct {
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    IW: i32,
    IH: i32,
    KW: i32,
    KH: i32,
    C: i32,
    OW: i32,
    OH: i32,
    N: i32,
    s0: i32,
    s1: i32,
    p0: i32,
    p1: i32,
    d0: i32,
    d1: i32,
};

/// Mirrors `ggml_metal_kargs_im2col` (ggml-metal-impl.h:719 @c1d0e7a00).
pub const im2col = extern struct {
    ofs0: u64,
    ofs1: u64,
    IW: i32,
    IH: i32,
    CHW: i32,
    s0: i32,
    s1: i32,
    p0: i32,
    p1: i32,
    d0: i32,
    d1: i32,
    N: i32,
    KH: i32,
    KW: i32,
    KHW: i32,
};

/// Mirrors `ggml_metal_kargs_conv_3d` (ggml-metal-impl.h:737 @c1d0e7a00).
pub const conv_3d = extern struct {
    IW: i32,
    IH: i32,
    ID: i32,
    OW: i32,
    OH: i32,
    OD: i32,
    KW: i32,
    KH: i32,
    KD: i32,
    s0: i32,
    s1: i32,
    s2: i32,
    p0: i32,
    p1: i32,
    p2: i32,
    d0: i32,
    d1: i32,
    d2: i32,
    IC: i32,
    N: i32,
    OC: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
};

/// Mirrors `ggml_metal_kargs_sum` (ggml-metal-impl.h:786 @c1d0e7a00).
pub const sum = extern struct {
    np: u64,
};

/// Mirrors `ggml_metal_kargs_sum_rows` (ggml-metal-impl.h:790 @c1d0e7a00).
pub const sum_rows = extern struct {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
};

/// Mirrors `ggml_metal_kargs_cumsum_blk` (ggml-metal-impl.h:809 @c1d0e7a00).
pub const cumsum_blk = extern struct {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    net0: i64,
    net1: i64,
    net2: i64,
    net3: i64,
    nbt0: u64,
    nbt1: u64,
    nbt2: u64,
    nbt3: u64,
    outb: bool,
};

/// Mirrors `ggml_metal_kargs_cumsum_add` (ggml-metal-impl.h:829 @c1d0e7a00).
pub const cumsum_add = extern struct {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    net0: i64,
    net1: i64,
    net2: i64,
    net3: i64,
    nbt0: u64,
    nbt1: u64,
    nbt2: u64,
    nbt3: u64,
};

/// Mirrors `ggml_metal_kargs_soft_max` (ggml-metal-impl.h:848 @c1d0e7a00).
pub const soft_max = extern struct {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    scale: f32,
    max_bias: f32,
    m0: f32,
    m1: f32,
    n_head_log2: i32,
};

/// Mirrors `ggml_metal_kargs_ssm_conv` (ggml-metal-impl.h:871 @c1d0e7a00).
pub const ssm_conv = extern struct {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    ne10: i64,
    ne11: i64,
    nb10: u64,
    nb11: u64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
};

/// Mirrors `ggml_metal_kargs_ssm_scan` (ggml-metal-impl.h:890 @c1d0e7a00).
pub const ssm_scan = extern struct {
    d_state: i64,
    d_inner: i64,
    n_head: i64,
    n_group: i64,
    n_seq_tokens: i64,
    n_seqs: i64,
    K: i64,
    s_off: u64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    ns12: u64,
    nb13: u64,
    nb20: u64,
    nb21: u64,
    ns21: u64,
    nb22: u64,
    ne30: i64,
    nb31: u64,
    nb41: u64,
    nb42: u64,
    ns42: u64,
    nb43: u64,
    nb51: u64,
    nb52: u64,
    ns52: u64,
    nb53: u64,
    nb0: u64,
};

/// Mirrors `ggml_metal_kargs_gated_delta_net` (ggml-metal-impl.h:925 @c1d0e7a00).
pub const gated_delta_net = extern struct {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne20: i32,
    ne21: i32,
    ne22: i32,
    ne23: i32,
    nb20: u64,
    nb21: u64,
    nb22: u64,
    nb23: u64,
    ns02: i32,
    ns12: i32,
    ns22: i32,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
};

/// Mirrors `ggml_metal_kargs_solve_tri` (ggml-metal-impl.h:963 @c1d0e7a00).
pub const solve_tri = extern struct {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
};

/// Mirrors `ggml_metal_kargs_get_rows` (ggml-metal-impl.h:990 @c1d0e7a00).
pub const get_rows = extern struct {
    ne00t: i32,
    ne00: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
};

/// Mirrors `ggml_metal_kargs_set_rows` (ggml-metal-impl.h:1005 @c1d0e7a00).
pub const set_rows = extern struct {
    nk0: i32,
    ne01: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
};

/// Mirrors `ggml_metal_kargs_diag` (ggml-metal-impl.h:1021 @c1d0e7a00).
pub const diag = extern struct {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
};

/// Mirrors `ggml_metal_kargs_upscale` (ggml-metal-impl.h:1040 @c1d0e7a00).
pub const upscale = extern struct {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    sf0: f32,
    sf1: f32,
    sf2: f32,
    sf3: f32,
    poffs: f32,
};

/// Mirrors `ggml_metal_kargs_pad` (ggml-metal-impl.h:1064 @c1d0e7a00).
pub const pad = extern struct {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
};

/// Mirrors `ggml_metal_kargs_pad_reflect_1d` (ggml-metal-impl.h:1083 @c1d0e7a00).
pub const pad_reflect_1d = extern struct {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    p0: i32,
    p1: i32,
};

/// Mirrors `ggml_metal_kargs_roll` (ggml-metal-impl.h:1104 @c1d0e7a00).
pub const roll = extern struct {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    s0: i32,
    s1: i32,
    s2: i32,
    s3: i32,
};

/// Mirrors `ggml_metal_kargs_timestep_embedding` (ggml-metal-impl.h:1127 @c1d0e7a00).
pub const timestep_embedding = extern struct {
    nb1: u64,
    dim: c_int,
    max_period: c_int,
};

/// Mirrors `ggml_metal_kargs_tri` (ggml-metal-impl.h:1133 @c1d0e7a00).
pub const tri = extern struct {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
};

/// Mirrors `ggml_metal_kargs_argsort` (ggml-metal-impl.h:1152 @c1d0e7a00).
pub const argsort = extern struct {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    top_k: i32,
};

/// Mirrors `ggml_metal_kargs_argsort_merge` (ggml-metal-impl.h:1168 @c1d0e7a00).
pub const argsort_merge = extern struct {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    top_k: i32,
    len: i32,
};

/// Mirrors `ggml_metal_kargs_fwht` (ggml-metal-impl.h:1185 @c1d0e7a00).
pub const fwht = extern struct {
    nrows: i32,
};

/// Mirrors `ggml_metal_kargs_arange` (ggml-metal-impl.h:1189 @c1d0e7a00).
pub const arange = extern struct {
    ne0: i64,
    start: f32,
    step: f32,
};

/// Mirrors `ggml_metal_kargs_memset` (ggml-metal-impl.h:1195 @c1d0e7a00).
pub const memset = extern struct {
    val: i64,
};

/// Mirrors `ggml_metal_kargs_lightning_indexer` (ggml-metal-impl.h:1199 @c1d0e7a00).
pub const lightning_indexer = extern struct {
    n_kv: i32,
    n_batch: i32,
    mask_ne3: i32,
    nb1: u64,
    nb3: u64,
    nbq1: u64,
    nbq2: u64,
    nbq3: u64,
    nbk2: u64,
    nbk3: u64,
    nbw1: u64,
    nbw3: u64,
    nbm1: u64,
    nbm3: u64,
};

/// Mirrors `ggml_metal_kargs_dsv4_hc_comb` (ggml-metal-impl.h:1216 @c1d0e7a00).
pub const dsv4_hc_comb = extern struct {
    n_tokens: i32,
    n_iter: i32,
    nb_m0: u64,
    nb_m1: u64,
    nb_s0: u64,
    nb_b0: u64,
    nb_d0: u64,
    nb_d1: u64,
    nb_d2: u64,
    eps: f32,
};

/// Mirrors `ggml_metal_kargs_dsv4_hc_pre` (ggml-metal-impl.h:1229 @c1d0e7a00).
pub const dsv4_hc_pre = extern struct {
    n_embd: i32,
    n_tokens: i32,
    nb_x0: u64,
    nb_x1: u64,
    nb_x2: u64,
    nb_w0: u64,
    nb_w1: u64,
    nb_d0: u64,
    nb_d1: u64,
};

/// Mirrors `ggml_metal_kargs_dsv4_hc_post` (ggml-metal-impl.h:1241 @c1d0e7a00).
pub const dsv4_hc_post = extern struct {
    n_embd: i32,
    n_tokens: i32,
    nb_x0: u64,
    nb_x1: u64,
    nb_r0: u64,
    nb_r1: u64,
    nb_r2: u64,
    nb_p0: u64,
    nb_p1: u64,
    nb_c0: u64,
    nb_c1: u64,
    nb_c2: u64,
    nb_d0: u64,
    nb_d1: u64,
    nb_d2: u64,
};

/// Mirrors `ggml_metal_kargs_count_equal` (ggml-metal-impl.h:1259 @c1d0e7a00).
pub const count_equal = extern struct {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
};

/// Mirrors `ggml_metal_kargs_pool_2d` (ggml-metal-impl.h:1274 @c1d0e7a00).
pub const pool_2d = extern struct {
    k0: i32,
    k1: i32,
    s0: i32,
    s1: i32,
    p0: i32,
    p1: i32,
    IH: i64,
    IW: i64,
    OH: i64,
    OW: i64,
    np: i64,
};

/// Mirrors `ggml_metal_kargs_pool_1d` (ggml-metal-impl.h:1288 @c1d0e7a00).
pub const pool_1d = extern struct {
    k0: i32,
    s0: i32,
    p0: i32,
    IW: i64,
    OW: i64,
    np: i64,
};

/// Mirrors `ggml_metal_kargs_argmax` (ggml-metal-impl.h:1297 @c1d0e7a00).
pub const argmax = extern struct {
    ne00: i64,
    nb01: u64,
};

/// Mirrors `ggml_metal_kargs_opt_step_adamw` (ggml-metal-impl.h:1302 @c1d0e7a00).
pub const opt_step_adamw = extern struct {
    np: i64,
};

/// Mirrors `ggml_metal_kargs_opt_step_sgd` (ggml-metal-impl.h:1306 @c1d0e7a00).
pub const opt_step_sgd = extern struct {
    np: i64,
};

/// Mirrors `ggml_metal_kargs_silu_back` (ggml-metal-impl.h:1310 @c1d0e7a00).
pub const silu_back = extern struct {
    ne: i64,
};

// -----------------------------------------------------------------------------
// The gate's view
//
// `harness/kargs_layout.cpp` includes the real header and compares these
// against `offsetof`/`sizeof`. Names travel with the numbers so a
// reordering cannot let two wrong entries cancel.

const all = .{
    concat,
    unary,
    bin,
    add_id,
    repeat,
    cpy,
    set,
    rope,
    flash_attn_ext_kv_f16,
    flash_attn_ext_pad,
    flash_attn_ext_blk,
    flash_attn_ext,
    flash_attn_ext_vec,
    flash_attn_ext_vec_reduce,
    mul_mm,
    mul_mv,
    mul_mv_ext,
    mul_mm_id_map0,
    mul_mm_id,
    mul_mv_id,
    norm,
    l2_norm,
    group_norm,
    conv_transpose_1d,
    col2im_1d,
    snake,
    conv_transpose_2d,
    conv_2d,
    conv_2d_dw,
    im2col,
    conv_3d,
    sum,
    sum_rows,
    cumsum_blk,
    cumsum_add,
    soft_max,
    ssm_conv,
    ssm_scan,
    gated_delta_net,
    solve_tri,
    get_rows,
    set_rows,
    diag,
    upscale,
    pad,
    pad_reflect_1d,
    roll,
    timestep_embedding,
    tri,
    argsort,
    argsort_merge,
    fwht,
    arange,
    memset,
    lightning_indexer,
    dsv4_hc_comb,
    dsv4_hc_pre,
    dsv4_hc_post,
    count_equal,
    pool_2d,
    pool_1d,
    argmax,
    opt_step_adamw,
    opt_step_sgd,
    silu_back,
};

export fn zz_kargs_count() usize {
    return all.len;
}

export fn zz_kargs_name(i: usize) ?[*:0]const u8 {
    @setEvalBranchQuota(200000);
    inline for (all, 0..) |T, k| {
        if (k == i) return @typeName(T)[(comptime std.mem.lastIndexOfScalar(u8, @typeName(T), '.').? + 1)..].ptr;
    }
    return null;
}

export fn zz_kargs_sizeof(i: usize) usize {
    @setEvalBranchQuota(200000);
    inline for (all, 0..) |T, k| {
        if (k == i) return @sizeOf(T);
    }
    return std.math.maxInt(usize);
}

export fn zz_kargs_alignof(i: usize) usize {
    @setEvalBranchQuota(200000);
    inline for (all, 0..) |T, k| {
        if (k == i) return @alignOf(T);
    }
    return std.math.maxInt(usize);
}

export fn zz_kargs_nfields(i: usize) usize {
    @setEvalBranchQuota(200000);
    inline for (all, 0..) |T, k| {
        if (k == i) return @typeInfo(T).@"struct".fields.len;
    }
    return std.math.maxInt(usize);
}

export fn zz_kargs_field_offset(i: usize, f: usize) usize {
    @setEvalBranchQuota(200000);
    inline for (all, 0..) |T, k| {
        if (k == i) {
            inline for (@typeInfo(T).@"struct".fields, 0..) |fld, j| {
                if (j == f) return @offsetOf(T, fld.name);
            }
            return std.math.maxInt(usize);
        }
    }
    return std.math.maxInt(usize);
}

export fn zz_kargs_field_size(i: usize, f: usize) usize {
    @setEvalBranchQuota(200000);
    inline for (all, 0..) |T, k| {
        if (k == i) {
            inline for (@typeInfo(T).@"struct".fields, 0..) |fld, j| {
                if (j == f) return @sizeOf(fld.type);
            }
            return std.math.maxInt(usize);
        }
    }
    return std.math.maxInt(usize);
}

export fn zz_kargs_field_name(i: usize, f: usize) ?[*:0]const u8 {
    @setEvalBranchQuota(200000);
    inline for (all, 0..) |T, k| {
        if (k == i) {
            inline for (@typeInfo(T).@"struct".fields, 0..) |fld, j| {
                if (j == f) return fld.name.ptr;
            }
            return null;
        }
    }
    return null;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "every struct is an extern struct with the expected field count" {
    // A guard on the generator rather than on the layout, which
    // `make struct-layout` owns: if a struct came out empty, or as a
    // non-extern struct whose layout Zig may reorder, that is a generator
    // fault and no offset check would be meaningful.
    inline for (@typeInfo(@This()).@"struct".decls) |d| {
        const T = @field(@This(), d.name);
        if (@TypeOf(T) != type) continue;
        if (@typeInfo(T) != .@"struct") continue;
        try std.testing.expectEqual(std.builtin.Type.ContainerLayout.@"extern", @typeInfo(T).@"struct".layout);
        try std.testing.expect(@typeInfo(T).@"struct".fields.len > 0);
    }
}
