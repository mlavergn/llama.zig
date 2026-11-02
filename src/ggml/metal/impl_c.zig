//! The function-constant indices, op numbers and kernel-shape constants
//! the Metal kernel names and specialisations carry.
//!
//! # Provenance
//!
//! Mirrors the `FC_*`, `OP_*`, `N_*` and `SZ_*` `#define`s of
//! `llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h` at v0.5.0
//! (`7fe450e19`).
//!
//! # Hand-declared, generated, and checked
//!
//! `ggml-metal-impl.h` is shared with the Metal Shading Language, and like
//! the other Metal headers it cannot join `impl.zig`'s single `cImport`
//! without widening the `c` namespace the whole library sees — see
//! `device_c.zig` for that reasoning.
//!
//! These are not types but numbers, and a wrong one is the quietest
//! possible fault in the Metal port. `FC_UNARY + 1` names the slot a
//! Metal function constant is written to, so an off-by-one configures a
//! *different* constant and the kernel does something else; `N_R0_Q4_K`
//! sets how many rows a dot-product kernel handles per thread, so a wrong
//! value reads out of bounds. Nothing in the ported Zig could notice
//! either.
//!
//! So they are **extracted by script, not typed**, and every one is
//! compared against the header by `make struct-layout` — by name and by
//! value, in order, with the count cross-checked so an addition upstream
//! cannot pass as "everything I listed is fine".

const std = @import("std");

/// Mirrors `SZ_SIMDGROUP` (ggml-metal-impl.h:8 @7fe450e19).
pub const SZ_SIMDGROUP: c_int = 16;
/// Mirrors `N_MM_NK` (ggml-metal-impl.h:9 @7fe450e19).
pub const N_MM_NK: c_int = 2;
/// Mirrors `N_MM_BLOCK_X` (ggml-metal-impl.h:12 @7fe450e19).
pub const N_MM_BLOCK_X: c_int = 4;
/// Mirrors `N_MM_BLOCK_Y` (ggml-metal-impl.h:13 @7fe450e19).
pub const N_MM_BLOCK_Y: c_int = 2;
/// Mirrors `N_MM_SIMD_GROUP_X` (ggml-metal-impl.h:14 @7fe450e19).
pub const N_MM_SIMD_GROUP_X: c_int = 2;
/// Mirrors `N_MM_SIMD_GROUP_Y` (ggml-metal-impl.h:15 @7fe450e19).
pub const N_MM_SIMD_GROUP_Y: c_int = 2;
/// Mirrors `N_MM_NPART_AMAX` (ggml-metal-impl.h:17 @7fe450e19): the
/// threadgroups the first `MUL_MAT_ID` amax stage splits `src1` across.
pub const N_MM_NPART_AMAX: c_int = 256;
/// Mirrors `N_R0_Q1_0` (ggml-metal-impl.h:26 @7fe450e19).
pub const N_R0_Q1_0: c_int = 8;
/// Mirrors `N_SG_Q1_0` (ggml-metal-impl.h:27 @7fe450e19).
pub const N_SG_Q1_0: c_int = 2;
/// Mirrors `N_R0_Q2_0` (ggml-metal-impl.h:29 @7fe450e19).
pub const N_R0_Q2_0: c_int = 8;
/// Mirrors `N_SG_Q2_0` (ggml-metal-impl.h:30 @7fe450e19).
pub const N_SG_Q2_0: c_int = 2;
/// Mirrors `N_R0_Q4_0` (ggml-metal-impl.h:32 @7fe450e19).
pub const N_R0_Q4_0: c_int = 4;
/// Mirrors `N_SG_Q4_0` (ggml-metal-impl.h:33 @7fe450e19).
pub const N_SG_Q4_0: c_int = 2;
/// Mirrors `N_R0_Q4_1` (ggml-metal-impl.h:35 @7fe450e19).
pub const N_R0_Q4_1: c_int = 4;
/// Mirrors `N_SG_Q4_1` (ggml-metal-impl.h:36 @7fe450e19).
pub const N_SG_Q4_1: c_int = 2;
/// Mirrors `N_R0_Q5_0` (ggml-metal-impl.h:38 @7fe450e19).
pub const N_R0_Q5_0: c_int = 4;
/// Mirrors `N_SG_Q5_0` (ggml-metal-impl.h:39 @7fe450e19).
pub const N_SG_Q5_0: c_int = 2;
/// Mirrors `N_R0_Q5_1` (ggml-metal-impl.h:41 @7fe450e19).
pub const N_R0_Q5_1: c_int = 4;
/// Mirrors `N_SG_Q5_1` (ggml-metal-impl.h:42 @7fe450e19).
pub const N_SG_Q5_1: c_int = 2;
/// Mirrors `N_R0_Q8_0` (ggml-metal-impl.h:44 @7fe450e19).
pub const N_R0_Q8_0: c_int = 2;
/// Mirrors `N_SG_Q8_0` (ggml-metal-impl.h:45 @7fe450e19).
pub const N_SG_Q8_0: c_int = 4;
/// Mirrors `N_R0_MXFP4` (ggml-metal-impl.h:47 @7fe450e19).
pub const N_R0_MXFP4: c_int = 2;
/// Mirrors `N_SG_MXFP4` (ggml-metal-impl.h:48 @7fe450e19).
pub const N_SG_MXFP4: c_int = 2;
/// Mirrors `N_R0_Q2_K` (ggml-metal-impl.h:50 @7fe450e19).
pub const N_R0_Q2_K: c_int = 4;
/// Mirrors `N_SG_Q2_K` (ggml-metal-impl.h:51 @7fe450e19).
pub const N_SG_Q2_K: c_int = 2;
/// Mirrors `N_R0_Q3_K` (ggml-metal-impl.h:53 @7fe450e19).
pub const N_R0_Q3_K: c_int = 2;
/// Mirrors `N_SG_Q3_K` (ggml-metal-impl.h:54 @7fe450e19).
pub const N_SG_Q3_K: c_int = 2;
/// Mirrors `N_R0_Q4_K` (ggml-metal-impl.h:56 @7fe450e19).
pub const N_R0_Q4_K: c_int = 2;
/// Mirrors `N_SG_Q4_K` (ggml-metal-impl.h:57 @7fe450e19).
pub const N_SG_Q4_K: c_int = 2;
/// Mirrors `N_R0_Q5_K` (ggml-metal-impl.h:59 @7fe450e19).
pub const N_R0_Q5_K: c_int = 1;
/// Mirrors `N_SG_Q5_K` (ggml-metal-impl.h:60 @7fe450e19).
pub const N_SG_Q5_K: c_int = 2;
/// Mirrors `N_R0_Q6_K` (ggml-metal-impl.h:62 @7fe450e19).
pub const N_R0_Q6_K: c_int = 2;
/// Mirrors `N_SG_Q6_K` (ggml-metal-impl.h:63 @7fe450e19).
pub const N_SG_Q6_K: c_int = 2;
/// Mirrors `N_R0_IQ1_S` (ggml-metal-impl.h:65 @7fe450e19).
pub const N_R0_IQ1_S: c_int = 4;
/// Mirrors `N_SG_IQ1_S` (ggml-metal-impl.h:66 @7fe450e19).
pub const N_SG_IQ1_S: c_int = 2;
/// Mirrors `N_R0_IQ1_S_SPLIT` (ggml-metal-impl.h:67 @7fe450e19).
pub const N_R0_IQ1_S_SPLIT: c_int = 8;
/// Mirrors `N_R0_IQ1_M` (ggml-metal-impl.h:69 @7fe450e19).
pub const N_R0_IQ1_M: c_int = 4;
/// Mirrors `N_SG_IQ1_M` (ggml-metal-impl.h:70 @7fe450e19).
pub const N_SG_IQ1_M: c_int = 2;
/// Mirrors `N_R0_IQ1_M_SPLIT` (ggml-metal-impl.h:71 @7fe450e19).
pub const N_R0_IQ1_M_SPLIT: c_int = 8;
/// Mirrors `N_R0_IQ2_XXS` (ggml-metal-impl.h:73 @7fe450e19).
pub const N_R0_IQ2_XXS: c_int = 4;
/// Mirrors `N_SG_IQ2_XXS` (ggml-metal-impl.h:74 @7fe450e19).
pub const N_SG_IQ2_XXS: c_int = 2;
/// Mirrors `N_R0_IQ2_XXS_SPLIT` (ggml-metal-impl.h:75 @7fe450e19).
pub const N_R0_IQ2_XXS_SPLIT: c_int = 8;
/// Mirrors `N_R0_IQ2_XS` (ggml-metal-impl.h:77 @7fe450e19).
pub const N_R0_IQ2_XS: c_int = 4;
/// Mirrors `N_SG_IQ2_XS` (ggml-metal-impl.h:78 @7fe450e19).
pub const N_SG_IQ2_XS: c_int = 2;
/// Mirrors `N_R0_IQ2_XS_SPLIT` (ggml-metal-impl.h:79 @7fe450e19).
pub const N_R0_IQ2_XS_SPLIT: c_int = 8;
/// Mirrors `N_R0_IQ2_S` (ggml-metal-impl.h:81 @7fe450e19).
pub const N_R0_IQ2_S: c_int = 4;
/// Mirrors `N_SG_IQ2_S` (ggml-metal-impl.h:82 @7fe450e19).
pub const N_SG_IQ2_S: c_int = 2;
/// Mirrors `N_R0_IQ2_S_SPLIT` (ggml-metal-impl.h:83 @7fe450e19).
pub const N_R0_IQ2_S_SPLIT: c_int = 8;
/// Mirrors `N_R0_IQ3_XXS` (ggml-metal-impl.h:85 @7fe450e19).
pub const N_R0_IQ3_XXS: c_int = 4;
/// Mirrors `N_SG_IQ3_XXS` (ggml-metal-impl.h:86 @7fe450e19).
pub const N_SG_IQ3_XXS: c_int = 2;
/// Mirrors `N_R0_IQ3_XXS_SPLIT` (ggml-metal-impl.h:87 @7fe450e19).
pub const N_R0_IQ3_XXS_SPLIT: c_int = 8;
/// Mirrors `N_R0_IQ3_S` (ggml-metal-impl.h:89 @7fe450e19).
pub const N_R0_IQ3_S: c_int = 4;
/// Mirrors `N_SG_IQ3_S` (ggml-metal-impl.h:90 @7fe450e19).
pub const N_SG_IQ3_S: c_int = 2;
/// Mirrors `N_R0_IQ3_S_SPLIT` (ggml-metal-impl.h:91 @7fe450e19).
pub const N_R0_IQ3_S_SPLIT: c_int = 8;
/// Mirrors `N_R0_IQ4_NL` (ggml-metal-impl.h:93 @7fe450e19).
pub const N_R0_IQ4_NL: c_int = 2;
/// Mirrors `N_SG_IQ4_NL` (ggml-metal-impl.h:94 @7fe450e19).
pub const N_SG_IQ4_NL: c_int = 2;
/// Mirrors `N_R0_IQ4_XS` (ggml-metal-impl.h:96 @7fe450e19).
pub const N_R0_IQ4_XS: c_int = 2;
/// Mirrors `N_SG_IQ4_XS` (ggml-metal-impl.h:97 @7fe450e19).
pub const N_SG_IQ4_XS: c_int = 2;
/// Mirrors `N_R0_TQ2_0` (ggml-metal-impl.h:99 @7fe450e19).
pub const N_R0_TQ2_0: c_int = 4;
/// Mirrors `N_SG_TQ2_0` (ggml-metal-impl.h:100 @7fe450e19).
pub const N_SG_TQ2_0: c_int = 2;
/// Mirrors `FC_FLASH_ATTN_EXT_PAD` (ggml-metal-impl.h:103 @7fe450e19).
pub const FC_FLASH_ATTN_EXT_PAD: c_int = 100;
/// Mirrors `FC_FLASH_ATTN_EXT_BLK` (ggml-metal-impl.h:104 @7fe450e19).
pub const FC_FLASH_ATTN_EXT_BLK: c_int = 200;
/// Mirrors `FC_FLASH_ATTN_EXT` (ggml-metal-impl.h:105 @7fe450e19).
pub const FC_FLASH_ATTN_EXT: c_int = 300;
/// Mirrors `FC_FLASH_ATTN_EXT_VEC` (ggml-metal-impl.h:106 @7fe450e19).
pub const FC_FLASH_ATTN_EXT_VEC: c_int = 400;
/// Mirrors `FC_FLASH_ATTN_EXT_VEC_REDUCE` (ggml-metal-impl.h:107 @7fe450e19).
pub const FC_FLASH_ATTN_EXT_VEC_REDUCE: c_int = 500;
/// Mirrors `FC_MUL_MV` (ggml-metal-impl.h:108 @7fe450e19).
pub const FC_MUL_MV: c_int = 600;
/// Mirrors `FC_MUL_MM` (ggml-metal-impl.h:109 @7fe450e19).
pub const FC_MUL_MM: c_int = 700;
/// Mirrors `FC_ROPE` (ggml-metal-impl.h:110 @7fe450e19).
pub const FC_ROPE: c_int = 800;
/// Mirrors `FC_SSM_CONV` (ggml-metal-impl.h:111 @7fe450e19).
pub const FC_SSM_CONV: c_int = 900;
/// Mirrors `FC_SOLVE_TRI` (ggml-metal-impl.h:112 @7fe450e19).
pub const FC_SOLVE_TRI: c_int = 1000;
/// Mirrors `FC_COUNT_EQUAL` (ggml-metal-impl.h:113 @7fe450e19).
pub const FC_COUNT_EQUAL: c_int = 1100;
/// Mirrors `FC_UNARY` (ggml-metal-impl.h:114 @7fe450e19).
pub const FC_UNARY: c_int = 1200;
/// Mirrors `FC_BIN` (ggml-metal-impl.h:115 @7fe450e19).
pub const FC_BIN: c_int = 1300;
/// Mirrors `FC_SUM_ROWS` (ggml-metal-impl.h:116 @7fe450e19).
pub const FC_SUM_ROWS: c_int = 1400;
/// Mirrors `FC_UPSCALE` (ggml-metal-impl.h:117 @7fe450e19).
pub const FC_UPSCALE: c_int = 1500;
/// Mirrors `FC_GATED_DELTA_NET` (ggml-metal-impl.h:118 @7fe450e19).
pub const FC_GATED_DELTA_NET: c_int = 1600;
/// Mirrors `FC_NORM` (ggml-metal-impl.h:119 @7fe450e19).
pub const FC_NORM: c_int = 1700;
/// Mirrors `FC_TOPK_MOE` (ggml-metal-impl.h:120 @7fe450e19).
pub const FC_TOPK_MOE: c_int = 1800;
/// Mirrors `FC_MOE_REDUCE` (ggml-metal-impl.h:121 @7fe450e19).
pub const FC_MOE_REDUCE: c_int = 1900;
/// Mirrors `FC_DSV4_HC` (ggml-metal-impl.h:122 @7fe450e19).
pub const FC_DSV4_HC: c_int = 2000;
/// Mirrors `OP_FLASH_ATTN_EXT_NQPSG` (ggml-metal-impl.h:125 @7fe450e19).
pub const OP_FLASH_ATTN_EXT_NQPSG: c_int = 8;
/// Mirrors `OP_FLASH_ATTN_EXT_NCPSG` (ggml-metal-impl.h:126 @7fe450e19).
pub const OP_FLASH_ATTN_EXT_NCPSG: c_int = 64;
/// Mirrors `OP_FLASH_ATTN_EXT_VEC_NQPSG` (ggml-metal-impl.h:128 @7fe450e19).
pub const OP_FLASH_ATTN_EXT_VEC_NQPSG: c_int = 1;
/// Mirrors `OP_FLASH_ATTN_EXT_VEC_NCPSG` (ggml-metal-impl.h:129 @7fe450e19).
pub const OP_FLASH_ATTN_EXT_VEC_NCPSG: c_int = 32;
/// Mirrors `OP_LIGHTNING_INDEXER_DK` (ggml-metal-impl.h:131 @7fe450e19).
pub const OP_LIGHTNING_INDEXER_DK: c_int = 128;
/// Mirrors `OP_LIGHTNING_INDEXER_NH` (ggml-metal-impl.h:132 @7fe450e19).
pub const OP_LIGHTNING_INDEXER_NH: c_int = 64;
/// Mirrors `OP_LIGHTNING_INDEXER_NHPTG` (ggml-metal-impl.h:133 @7fe450e19).
pub const OP_LIGHTNING_INDEXER_NHPTG: c_int = 8;
/// Mirrors `OP_LIGHTNING_INDEXER_NKPSG` (ggml-metal-impl.h:134 @7fe450e19).
pub const OP_LIGHTNING_INDEXER_NKPSG: c_int = 8;
/// Mirrors `OP_LIGHTNING_INDEXER_NSG` (ggml-metal-impl.h:135 @7fe450e19).
pub const OP_LIGHTNING_INDEXER_NSG: c_int = 8;
/// Mirrors `OP_LIGHTNING_INDEXER_NBPTG` (ggml-metal-impl.h:136 @7fe450e19).
pub const OP_LIGHTNING_INDEXER_NBPTG: c_int = 8;
/// Mirrors `OP_UNARY_NUM_SCALE` (ggml-metal-impl.h:138 @7fe450e19).
pub const OP_UNARY_NUM_SCALE: c_int = 10;
/// Mirrors `OP_UNARY_NUM_FILL` (ggml-metal-impl.h:139 @7fe450e19).
pub const OP_UNARY_NUM_FILL: c_int = 11;
/// Mirrors `OP_UNARY_NUM_CLAMP` (ggml-metal-impl.h:140 @7fe450e19).
pub const OP_UNARY_NUM_CLAMP: c_int = 12;
/// Mirrors `OP_UNARY_NUM_SQR` (ggml-metal-impl.h:141 @7fe450e19).
pub const OP_UNARY_NUM_SQR: c_int = 13;
/// Mirrors `OP_UNARY_NUM_SQRT` (ggml-metal-impl.h:142 @7fe450e19).
pub const OP_UNARY_NUM_SQRT: c_int = 14;
/// Mirrors `OP_UNARY_NUM_SIN` (ggml-metal-impl.h:143 @7fe450e19).
pub const OP_UNARY_NUM_SIN: c_int = 15;
/// Mirrors `OP_UNARY_NUM_COS` (ggml-metal-impl.h:144 @7fe450e19).
pub const OP_UNARY_NUM_COS: c_int = 16;
/// Mirrors `OP_UNARY_NUM_LOG` (ggml-metal-impl.h:145 @7fe450e19).
pub const OP_UNARY_NUM_LOG: c_int = 17;
/// Mirrors `OP_UNARY_NUM_LEAKY_RELU` (ggml-metal-impl.h:146 @7fe450e19).
pub const OP_UNARY_NUM_LEAKY_RELU: c_int = 18;
/// Mirrors `OP_UNARY_NUM_TANH` (ggml-metal-impl.h:148 @7fe450e19).
pub const OP_UNARY_NUM_TANH: c_int = 100;
/// Mirrors `OP_UNARY_NUM_RELU` (ggml-metal-impl.h:149 @7fe450e19).
pub const OP_UNARY_NUM_RELU: c_int = 101;
/// Mirrors `OP_UNARY_NUM_SIGMOID` (ggml-metal-impl.h:150 @7fe450e19).
pub const OP_UNARY_NUM_SIGMOID: c_int = 102;
/// Mirrors `OP_UNARY_NUM_GELU` (ggml-metal-impl.h:151 @7fe450e19).
pub const OP_UNARY_NUM_GELU: c_int = 103;
/// Mirrors `OP_UNARY_NUM_GELU_ERF` (ggml-metal-impl.h:152 @7fe450e19).
pub const OP_UNARY_NUM_GELU_ERF: c_int = 104;
/// Mirrors `OP_UNARY_NUM_GELU_QUICK` (ggml-metal-impl.h:153 @7fe450e19).
pub const OP_UNARY_NUM_GELU_QUICK: c_int = 105;
/// Mirrors `OP_UNARY_NUM_SILU` (ggml-metal-impl.h:154 @7fe450e19).
pub const OP_UNARY_NUM_SILU: c_int = 106;
/// Mirrors `OP_UNARY_NUM_ELU` (ggml-metal-impl.h:155 @7fe450e19).
pub const OP_UNARY_NUM_ELU: c_int = 107;
/// Mirrors `OP_UNARY_NUM_NEG` (ggml-metal-impl.h:156 @7fe450e19).
pub const OP_UNARY_NUM_NEG: c_int = 108;
/// Mirrors `OP_UNARY_NUM_ABS` (ggml-metal-impl.h:157 @7fe450e19).
pub const OP_UNARY_NUM_ABS: c_int = 109;
/// Mirrors `OP_UNARY_NUM_SGN` (ggml-metal-impl.h:158 @7fe450e19).
pub const OP_UNARY_NUM_SGN: c_int = 110;
/// Mirrors `OP_UNARY_NUM_STEP` (ggml-metal-impl.h:159 @7fe450e19).
pub const OP_UNARY_NUM_STEP: c_int = 111;
/// Mirrors `OP_UNARY_NUM_HARDSWISH` (ggml-metal-impl.h:160 @7fe450e19).
pub const OP_UNARY_NUM_HARDSWISH: c_int = 112;
/// Mirrors `OP_UNARY_NUM_HARDSIGMOID` (ggml-metal-impl.h:161 @7fe450e19).
pub const OP_UNARY_NUM_HARDSIGMOID: c_int = 113;
/// Mirrors `OP_UNARY_NUM_EXP` (ggml-metal-impl.h:162 @7fe450e19).
pub const OP_UNARY_NUM_EXP: c_int = 114;
/// Mirrors `OP_UNARY_NUM_SOFTPLUS` (ggml-metal-impl.h:163 @7fe450e19).
pub const OP_UNARY_NUM_SOFTPLUS: c_int = 115;
/// Mirrors `OP_UNARY_NUM_EXPM1` (ggml-metal-impl.h:164 @7fe450e19).
pub const OP_UNARY_NUM_EXPM1: c_int = 116;
/// Mirrors `OP_UNARY_NUM_FLOOR` (ggml-metal-impl.h:165 @7fe450e19).
pub const OP_UNARY_NUM_FLOOR: c_int = 117;
/// Mirrors `OP_UNARY_NUM_CEIL` (ggml-metal-impl.h:166 @7fe450e19).
pub const OP_UNARY_NUM_CEIL: c_int = 118;
/// Mirrors `OP_UNARY_NUM_ROUND` (ggml-metal-impl.h:167 @7fe450e19).
pub const OP_UNARY_NUM_ROUND: c_int = 119;
/// Mirrors `OP_UNARY_NUM_TRUNC` (ggml-metal-impl.h:168 @7fe450e19).
pub const OP_UNARY_NUM_TRUNC: c_int = 120;
/// Mirrors `OP_UNARY_NUM_XIELU` (ggml-metal-impl.h:169 @7fe450e19).
pub const OP_UNARY_NUM_XIELU: c_int = 121;
/// Mirrors `OP_SUM_ROWS_NUM_SUM_ROWS` (ggml-metal-impl.h:171 @7fe450e19).
pub const OP_SUM_ROWS_NUM_SUM_ROWS: c_int = 10;
/// Mirrors `OP_SUM_ROWS_NUM_MEAN` (ggml-metal-impl.h:172 @7fe450e19).
pub const OP_SUM_ROWS_NUM_MEAN: c_int = 11;
/// Mirrors `OP_SSM_SCAN_SSD_CS` (ggml-metal-impl.h:174 @7fe450e19): the SSD
/// scan's chunk size.
pub const OP_SSM_SCAN_SSD_CS: c_int = 64;
/// Mirrors `OP_SSM_SCAN_SSD_HD` (ggml-metal-impl.h:175 @7fe450e19): the head
/// dimension the MMA kernel is specialised for.
pub const OP_SSM_SCAN_SSD_HD: c_int = 64;
/// Mirrors `OP_SSM_SCAN_SSD_NSG` (ggml-metal-impl.h:176 @7fe450e19): simdgroups
/// per threadgroup.
pub const OP_SSM_SCAN_SSD_NSG: c_int = 4;
/// Mirrors `N_MM_NK_TOTAL` (ggml-metal-impl.h:10 @7fe450e19).
///
/// The one computed define: the C writes `(SZ_SIMDGROUP * N_MM_NK)`, kept as an
/// expression here so it tracks its operands.
pub const N_MM_NK_TOTAL: c_int = SZ_SIMDGROUP * N_MM_NK;

/// The count the gate cross-checks, so a constant added upstream and not
/// here shows up as a mismatch rather than as nothing.
pub const count: usize = 127;

// -----------------------------------------------------------------------------
// The gate's view
//
// `harness/struct_layout.cpp` includes the real header and compares these
// against its own `#define`s. Name and value travel together so a
// reordering cannot let two wrong entries cancel.

const Entry = struct { name: [*:0]const u8, value: c_long };

const all = [_]Entry{
    .{ .name = "SZ_SIMDGROUP", .value = 16 },
    .{ .name = "N_MM_NK", .value = 2 },
    .{ .name = "N_MM_BLOCK_X", .value = 4 },
    .{ .name = "N_MM_BLOCK_Y", .value = 2 },
    .{ .name = "N_MM_SIMD_GROUP_X", .value = 2 },
    .{ .name = "N_MM_SIMD_GROUP_Y", .value = 2 },
    .{ .name = "N_MM_NPART_AMAX", .value = 256 },
    .{ .name = "N_R0_Q1_0", .value = 8 },
    .{ .name = "N_SG_Q1_0", .value = 2 },
    .{ .name = "N_R0_Q2_0", .value = 8 },
    .{ .name = "N_SG_Q2_0", .value = 2 },
    .{ .name = "N_R0_Q4_0", .value = 4 },
    .{ .name = "N_SG_Q4_0", .value = 2 },
    .{ .name = "N_R0_Q4_1", .value = 4 },
    .{ .name = "N_SG_Q4_1", .value = 2 },
    .{ .name = "N_R0_Q5_0", .value = 4 },
    .{ .name = "N_SG_Q5_0", .value = 2 },
    .{ .name = "N_R0_Q5_1", .value = 4 },
    .{ .name = "N_SG_Q5_1", .value = 2 },
    .{ .name = "N_R0_Q8_0", .value = 2 },
    .{ .name = "N_SG_Q8_0", .value = 4 },
    .{ .name = "N_R0_MXFP4", .value = 2 },
    .{ .name = "N_SG_MXFP4", .value = 2 },
    .{ .name = "N_R0_Q2_K", .value = 4 },
    .{ .name = "N_SG_Q2_K", .value = 2 },
    .{ .name = "N_R0_Q3_K", .value = 2 },
    .{ .name = "N_SG_Q3_K", .value = 2 },
    .{ .name = "N_R0_Q4_K", .value = 2 },
    .{ .name = "N_SG_Q4_K", .value = 2 },
    .{ .name = "N_R0_Q5_K", .value = 1 },
    .{ .name = "N_SG_Q5_K", .value = 2 },
    .{ .name = "N_R0_Q6_K", .value = 2 },
    .{ .name = "N_SG_Q6_K", .value = 2 },
    .{ .name = "N_R0_IQ1_S", .value = 4 },
    .{ .name = "N_SG_IQ1_S", .value = 2 },
    .{ .name = "N_R0_IQ1_S_SPLIT", .value = 8 },
    .{ .name = "N_R0_IQ1_M", .value = 4 },
    .{ .name = "N_SG_IQ1_M", .value = 2 },
    .{ .name = "N_R0_IQ1_M_SPLIT", .value = 8 },
    .{ .name = "N_R0_IQ2_XXS", .value = 4 },
    .{ .name = "N_SG_IQ2_XXS", .value = 2 },
    .{ .name = "N_R0_IQ2_XXS_SPLIT", .value = 8 },
    .{ .name = "N_R0_IQ2_XS", .value = 4 },
    .{ .name = "N_SG_IQ2_XS", .value = 2 },
    .{ .name = "N_R0_IQ2_XS_SPLIT", .value = 8 },
    .{ .name = "N_R0_IQ2_S", .value = 4 },
    .{ .name = "N_SG_IQ2_S", .value = 2 },
    .{ .name = "N_R0_IQ2_S_SPLIT", .value = 8 },
    .{ .name = "N_R0_IQ3_XXS", .value = 4 },
    .{ .name = "N_SG_IQ3_XXS", .value = 2 },
    .{ .name = "N_R0_IQ3_XXS_SPLIT", .value = 8 },
    .{ .name = "N_R0_IQ3_S", .value = 4 },
    .{ .name = "N_SG_IQ3_S", .value = 2 },
    .{ .name = "N_R0_IQ3_S_SPLIT", .value = 8 },
    .{ .name = "N_R0_IQ4_NL", .value = 2 },
    .{ .name = "N_SG_IQ4_NL", .value = 2 },
    .{ .name = "N_R0_IQ4_XS", .value = 2 },
    .{ .name = "N_SG_IQ4_XS", .value = 2 },
    .{ .name = "N_R0_TQ2_0", .value = 4 },
    .{ .name = "N_SG_TQ2_0", .value = 2 },
    .{ .name = "FC_FLASH_ATTN_EXT_PAD", .value = 100 },
    .{ .name = "FC_FLASH_ATTN_EXT_BLK", .value = 200 },
    .{ .name = "FC_FLASH_ATTN_EXT", .value = 300 },
    .{ .name = "FC_FLASH_ATTN_EXT_VEC", .value = 400 },
    .{ .name = "FC_FLASH_ATTN_EXT_VEC_REDUCE", .value = 500 },
    .{ .name = "FC_MUL_MV", .value = 600 },
    .{ .name = "FC_MUL_MM", .value = 700 },
    .{ .name = "FC_ROPE", .value = 800 },
    .{ .name = "FC_SSM_CONV", .value = 900 },
    .{ .name = "FC_SOLVE_TRI", .value = 1000 },
    .{ .name = "FC_COUNT_EQUAL", .value = 1100 },
    .{ .name = "FC_UNARY", .value = 1200 },
    .{ .name = "FC_BIN", .value = 1300 },
    .{ .name = "FC_SUM_ROWS", .value = 1400 },
    .{ .name = "FC_UPSCALE", .value = 1500 },
    .{ .name = "FC_GATED_DELTA_NET", .value = 1600 },
    .{ .name = "FC_NORM", .value = 1700 },
    .{ .name = "FC_TOPK_MOE", .value = 1800 },
    .{ .name = "FC_MOE_REDUCE", .value = 1900 },
    .{ .name = "FC_DSV4_HC", .value = 2000 },
    .{ .name = "OP_FLASH_ATTN_EXT_NQPSG", .value = 8 },
    .{ .name = "OP_FLASH_ATTN_EXT_NCPSG", .value = 64 },
    .{ .name = "OP_FLASH_ATTN_EXT_VEC_NQPSG", .value = 1 },
    .{ .name = "OP_FLASH_ATTN_EXT_VEC_NCPSG", .value = 32 },
    .{ .name = "OP_LIGHTNING_INDEXER_DK", .value = 128 },
    .{ .name = "OP_LIGHTNING_INDEXER_NH", .value = 64 },
    .{ .name = "OP_LIGHTNING_INDEXER_NHPTG", .value = 8 },
    .{ .name = "OP_LIGHTNING_INDEXER_NKPSG", .value = 8 },
    .{ .name = "OP_LIGHTNING_INDEXER_NSG", .value = 8 },
    .{ .name = "OP_LIGHTNING_INDEXER_NBPTG", .value = 8 },
    .{ .name = "OP_UNARY_NUM_SCALE", .value = 10 },
    .{ .name = "OP_UNARY_NUM_FILL", .value = 11 },
    .{ .name = "OP_UNARY_NUM_CLAMP", .value = 12 },
    .{ .name = "OP_UNARY_NUM_SQR", .value = 13 },
    .{ .name = "OP_UNARY_NUM_SQRT", .value = 14 },
    .{ .name = "OP_UNARY_NUM_SIN", .value = 15 },
    .{ .name = "OP_UNARY_NUM_COS", .value = 16 },
    .{ .name = "OP_UNARY_NUM_LOG", .value = 17 },
    .{ .name = "OP_UNARY_NUM_LEAKY_RELU", .value = 18 },
    .{ .name = "OP_UNARY_NUM_TANH", .value = 100 },
    .{ .name = "OP_UNARY_NUM_RELU", .value = 101 },
    .{ .name = "OP_UNARY_NUM_SIGMOID", .value = 102 },
    .{ .name = "OP_UNARY_NUM_GELU", .value = 103 },
    .{ .name = "OP_UNARY_NUM_GELU_ERF", .value = 104 },
    .{ .name = "OP_UNARY_NUM_GELU_QUICK", .value = 105 },
    .{ .name = "OP_UNARY_NUM_SILU", .value = 106 },
    .{ .name = "OP_UNARY_NUM_ELU", .value = 107 },
    .{ .name = "OP_UNARY_NUM_NEG", .value = 108 },
    .{ .name = "OP_UNARY_NUM_ABS", .value = 109 },
    .{ .name = "OP_UNARY_NUM_SGN", .value = 110 },
    .{ .name = "OP_UNARY_NUM_STEP", .value = 111 },
    .{ .name = "OP_UNARY_NUM_HARDSWISH", .value = 112 },
    .{ .name = "OP_UNARY_NUM_HARDSIGMOID", .value = 113 },
    .{ .name = "OP_UNARY_NUM_EXP", .value = 114 },
    .{ .name = "OP_UNARY_NUM_SOFTPLUS", .value = 115 },
    .{ .name = "OP_UNARY_NUM_EXPM1", .value = 116 },
    .{ .name = "OP_UNARY_NUM_FLOOR", .value = 117 },
    .{ .name = "OP_UNARY_NUM_CEIL", .value = 118 },
    .{ .name = "OP_UNARY_NUM_ROUND", .value = 119 },
    .{ .name = "OP_UNARY_NUM_TRUNC", .value = 120 },
    .{ .name = "OP_UNARY_NUM_XIELU", .value = 121 },
    .{ .name = "OP_SUM_ROWS_NUM_SUM_ROWS", .value = 10 },
    .{ .name = "OP_SUM_ROWS_NUM_MEAN", .value = 11 },
    .{ .name = "OP_SSM_SCAN_SSD_CS", .value = OP_SSM_SCAN_SSD_CS },
    .{ .name = "OP_SSM_SCAN_SSD_HD", .value = OP_SSM_SCAN_SSD_HD },
    .{ .name = "OP_SSM_SCAN_SSD_NSG", .value = OP_SSM_SCAN_SSD_NSG },
    .{ .name = "N_MM_NK_TOTAL", .value = N_MM_NK_TOTAL },
};

/// Return: the `i`th constant's value, or `maxInt` past the end.
export fn zz_fc_value(i: usize) c_long {
    inline for (all, 0..) |e, k| {
        if (k == i) return e.value;
    }
    return std.math.maxInt(c_long);
}

/// Return: the `i`th constant's name, NUL-terminated, or null past the end.
export fn zz_fc_name(i: usize) ?[*:0]const u8 {
    inline for (all, 0..) |e, k| {
        if (k == i) return e.name;
    }
    return null;
}

export fn zz_fc_count() usize {
    return count;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "every constant is in the exported table exactly once" {
    try std.testing.expectEqual(count, all.len);
    for (all, 0..) |a, i| {
        for (all[i + 1 ..]) |b| {
            try std.testing.expect(!std.mem.eql(u8, std.mem.span(a.name), std.mem.span(b.name)));
        }
    }
}
