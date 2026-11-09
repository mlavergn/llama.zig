//! The function-constant indices, op numbers and kernel-shape constants
//! the Metal kernel names and specialisations carry.
//!
//! # Provenance
//!
//! Mirrors the `FC_*`, `OP_*`, `N_*` and `SZ_*` `#define`s of
//! `llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h` at v0.8.0
//! (`8642541b2`), and since v0.6.0 its two `GGML_METAL_FWHT_*`.
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

/// Mirrors `SZ_SIMDGROUP` (ggml-metal-impl.h:8 @cd959bae9).
pub const SZ_SIMDGROUP: c_int = 16;
/// Mirrors `N_MM_NK` (ggml-metal-impl.h:9 @cd959bae9).
pub const N_MM_NK: c_int = 2;
/// Mirrors `N_MM_BLOCK_X` (ggml-metal-impl.h:12 @cd959bae9).
pub const N_MM_BLOCK_X: c_int = 4;
/// Mirrors `N_MM_BLOCK_Y` (ggml-metal-impl.h:13 @cd959bae9).
pub const N_MM_BLOCK_Y: c_int = 2;
/// Mirrors `N_MM_SIMD_GROUP_X` (ggml-metal-impl.h:14 @cd959bae9).
pub const N_MM_SIMD_GROUP_X: c_int = 2;
/// Mirrors `N_MM_SIMD_GROUP_Y` (ggml-metal-impl.h:15 @cd959bae9).
pub const N_MM_SIMD_GROUP_Y: c_int = 2;
/// Mirrors `N_ARGMAX_NPART_MAX` (ggml-metal-impl.h:17 @cd959bae9): the most
/// parts the two-stage `ARGMAX` splits a row into.
pub const N_ARGMAX_NPART_MAX: c_int = 256;
/// Mirrors `N_ARGMAX_CHUNK` (ggml-metal-impl.h:18 @cd959bae9): the row
/// length past which `ARGMAX` takes two stages, and the part size up to
/// the limit above.
pub const N_ARGMAX_CHUNK: c_int = 4096;
/// Mirrors `N_R0_Q1_0` (ggml-metal-impl.h:27 @cd959bae9).
pub const N_R0_Q1_0: c_int = 8;
/// Mirrors `N_SG_Q1_0` (ggml-metal-impl.h:28 @cd959bae9).
pub const N_SG_Q1_0: c_int = 2;
/// Mirrors `N_R0_Q2_0` (ggml-metal-impl.h:30 @cd959bae9).
pub const N_R0_Q2_0: c_int = 8;
/// Mirrors `N_SG_Q2_0` (ggml-metal-impl.h:31 @cd959bae9).
pub const N_SG_Q2_0: c_int = 2;
/// Mirrors `N_R0_Q4_0` (ggml-metal-impl.h:33 @cd959bae9).
pub const N_R0_Q4_0: c_int = 4;
/// Mirrors `N_SG_Q4_0` (ggml-metal-impl.h:34 @cd959bae9).
pub const N_SG_Q4_0: c_int = 2;
/// Mirrors `N_R0_Q4_1` (ggml-metal-impl.h:36 @cd959bae9).
pub const N_R0_Q4_1: c_int = 4;
/// Mirrors `N_SG_Q4_1` (ggml-metal-impl.h:37 @cd959bae9).
pub const N_SG_Q4_1: c_int = 2;
/// Mirrors `N_R0_Q5_0` (ggml-metal-impl.h:39 @cd959bae9).
pub const N_R0_Q5_0: c_int = 4;
/// Mirrors `N_SG_Q5_0` (ggml-metal-impl.h:40 @cd959bae9).
pub const N_SG_Q5_0: c_int = 2;
/// Mirrors `N_R0_Q5_1` (ggml-metal-impl.h:42 @cd959bae9).
pub const N_R0_Q5_1: c_int = 4;
/// Mirrors `N_SG_Q5_1` (ggml-metal-impl.h:43 @cd959bae9).
pub const N_SG_Q5_1: c_int = 2;
/// Mirrors `N_R0_Q8_0` (ggml-metal-impl.h:45 @cd959bae9).
pub const N_R0_Q8_0: c_int = 2;
/// Mirrors `N_SG_Q8_0` (ggml-metal-impl.h:46 @cd959bae9).
pub const N_SG_Q8_0: c_int = 4;
/// Mirrors `N_R0_MXFP4` (ggml-metal-impl.h:48 @cd959bae9).
pub const N_R0_MXFP4: c_int = 2;
/// Mirrors `N_SG_MXFP4` (ggml-metal-impl.h:49 @cd959bae9).
pub const N_SG_MXFP4: c_int = 2;
/// Mirrors `N_R0_Q2_K` (ggml-metal-impl.h:51 @cd959bae9).
pub const N_R0_Q2_K: c_int = 4;
/// Mirrors `N_SG_Q2_K` (ggml-metal-impl.h:52 @cd959bae9).
pub const N_SG_Q2_K: c_int = 2;
/// Mirrors `N_R0_Q3_K` (ggml-metal-impl.h:54 @cd959bae9).
pub const N_R0_Q3_K: c_int = 2;
/// Mirrors `N_SG_Q3_K` (ggml-metal-impl.h:55 @cd959bae9).
pub const N_SG_Q3_K: c_int = 2;
/// Mirrors `N_R0_Q4_K` (ggml-metal-impl.h:57 @cd959bae9).
pub const N_R0_Q4_K: c_int = 2;
/// Mirrors `N_SG_Q4_K` (ggml-metal-impl.h:58 @cd959bae9).
pub const N_SG_Q4_K: c_int = 2;
/// Mirrors `N_R0_Q5_K` (ggml-metal-impl.h:60 @cd959bae9).
pub const N_R0_Q5_K: c_int = 1;
/// Mirrors `N_SG_Q5_K` (ggml-metal-impl.h:61 @cd959bae9).
pub const N_SG_Q5_K: c_int = 2;
/// Mirrors `N_R0_Q6_K` (ggml-metal-impl.h:63 @cd959bae9).
pub const N_R0_Q6_K: c_int = 2;
/// Mirrors `N_SG_Q6_K` (ggml-metal-impl.h:64 @cd959bae9).
pub const N_SG_Q6_K: c_int = 2;
/// Mirrors `N_R0_IQ1_S` (ggml-metal-impl.h:66 @cd959bae9).
pub const N_R0_IQ1_S: c_int = 4;
/// Mirrors `N_SG_IQ1_S` (ggml-metal-impl.h:67 @cd959bae9).
pub const N_SG_IQ1_S: c_int = 2;
/// Mirrors `N_R0_IQ1_S_SPLIT` (ggml-metal-impl.h:68 @cd959bae9).
pub const N_R0_IQ1_S_SPLIT: c_int = 8;
/// Mirrors `N_R0_IQ1_M` (ggml-metal-impl.h:70 @cd959bae9).
pub const N_R0_IQ1_M: c_int = 4;
/// Mirrors `N_SG_IQ1_M` (ggml-metal-impl.h:71 @cd959bae9).
pub const N_SG_IQ1_M: c_int = 2;
/// Mirrors `N_R0_IQ1_M_SPLIT` (ggml-metal-impl.h:72 @cd959bae9).
pub const N_R0_IQ1_M_SPLIT: c_int = 8;
/// Mirrors `N_R0_IQ2_XXS` (ggml-metal-impl.h:74 @cd959bae9).
pub const N_R0_IQ2_XXS: c_int = 4;
/// Mirrors `N_SG_IQ2_XXS` (ggml-metal-impl.h:75 @cd959bae9).
pub const N_SG_IQ2_XXS: c_int = 2;
/// Mirrors `N_R0_IQ2_XXS_SPLIT` (ggml-metal-impl.h:76 @cd959bae9).
pub const N_R0_IQ2_XXS_SPLIT: c_int = 8;
/// Mirrors `N_R0_IQ2_XS` (ggml-metal-impl.h:78 @cd959bae9).
pub const N_R0_IQ2_XS: c_int = 4;
/// Mirrors `N_SG_IQ2_XS` (ggml-metal-impl.h:79 @cd959bae9).
pub const N_SG_IQ2_XS: c_int = 2;
/// Mirrors `N_R0_IQ2_XS_SPLIT` (ggml-metal-impl.h:80 @cd959bae9).
pub const N_R0_IQ2_XS_SPLIT: c_int = 8;
/// Mirrors `N_R0_IQ2_S` (ggml-metal-impl.h:82 @cd959bae9).
pub const N_R0_IQ2_S: c_int = 4;
/// Mirrors `N_SG_IQ2_S` (ggml-metal-impl.h:83 @cd959bae9).
pub const N_SG_IQ2_S: c_int = 2;
/// Mirrors `N_R0_IQ2_S_SPLIT` (ggml-metal-impl.h:84 @cd959bae9).
pub const N_R0_IQ2_S_SPLIT: c_int = 8;
/// Mirrors `N_R0_IQ3_XXS` (ggml-metal-impl.h:86 @cd959bae9).
pub const N_R0_IQ3_XXS: c_int = 4;
/// Mirrors `N_SG_IQ3_XXS` (ggml-metal-impl.h:87 @cd959bae9).
pub const N_SG_IQ3_XXS: c_int = 2;
/// Mirrors `N_R0_IQ3_XXS_SPLIT` (ggml-metal-impl.h:88 @cd959bae9).
pub const N_R0_IQ3_XXS_SPLIT: c_int = 8;
/// Mirrors `N_R0_IQ3_S` (ggml-metal-impl.h:90 @cd959bae9).
pub const N_R0_IQ3_S: c_int = 4;
/// Mirrors `N_SG_IQ3_S` (ggml-metal-impl.h:91 @cd959bae9).
pub const N_SG_IQ3_S: c_int = 2;
/// Mirrors `N_R0_IQ3_S_SPLIT` (ggml-metal-impl.h:92 @cd959bae9).
pub const N_R0_IQ3_S_SPLIT: c_int = 8;
/// Mirrors `N_R0_IQ4_NL` (ggml-metal-impl.h:94 @cd959bae9).
pub const N_R0_IQ4_NL: c_int = 2;
/// Mirrors `N_SG_IQ4_NL` (ggml-metal-impl.h:95 @cd959bae9).
pub const N_SG_IQ4_NL: c_int = 2;
/// Mirrors `N_R0_IQ4_XS` (ggml-metal-impl.h:97 @cd959bae9).
pub const N_R0_IQ4_XS: c_int = 2;
/// Mirrors `N_SG_IQ4_XS` (ggml-metal-impl.h:98 @cd959bae9).
pub const N_SG_IQ4_XS: c_int = 2;
/// Mirrors `N_R0_TQ2_0` (ggml-metal-impl.h:100 @cd959bae9).
pub const N_R0_TQ2_0: c_int = 4;
/// Mirrors `N_SG_TQ2_0` (ggml-metal-impl.h:101 @cd959bae9).
pub const N_SG_TQ2_0: c_int = 2;
/// Mirrors `FC_FLASH_ATTN_EXT_PAD` (ggml-metal-impl.h:104 @cd959bae9).
pub const FC_FLASH_ATTN_EXT_PAD: c_int = 100;
/// Mirrors `FC_FLASH_ATTN_EXT_BLK` (ggml-metal-impl.h:105 @cd959bae9).
pub const FC_FLASH_ATTN_EXT_BLK: c_int = 200;
/// Mirrors `FC_FLASH_ATTN_EXT` (ggml-metal-impl.h:106 @cd959bae9).
pub const FC_FLASH_ATTN_EXT: c_int = 300;
/// Mirrors `FC_FLASH_ATTN_EXT_VEC` (ggml-metal-impl.h:107 @cd959bae9).
pub const FC_FLASH_ATTN_EXT_VEC: c_int = 400;
/// Mirrors `FC_FLASH_ATTN_EXT_VEC_REDUCE` (ggml-metal-impl.h:108 @cd959bae9).
pub const FC_FLASH_ATTN_EXT_VEC_REDUCE: c_int = 500;
/// Mirrors `FC_MUL_MV` (ggml-metal-impl.h:109 @cd959bae9).
pub const FC_MUL_MV: c_int = 600;
/// Mirrors `FC_MUL_MM` (ggml-metal-impl.h:110 @cd959bae9).
pub const FC_MUL_MM: c_int = 700;
/// Mirrors `FC_ROPE` (ggml-metal-impl.h:111 @cd959bae9).
pub const FC_ROPE: c_int = 800;
/// Mirrors `FC_SSM_CONV` (ggml-metal-impl.h:112 @cd959bae9).
pub const FC_SSM_CONV: c_int = 900;
/// Mirrors `FC_SOLVE_TRI` (ggml-metal-impl.h:113 @cd959bae9).
pub const FC_SOLVE_TRI: c_int = 1000;
/// Mirrors `FC_COUNT_EQUAL` (ggml-metal-impl.h:114 @cd959bae9).
pub const FC_COUNT_EQUAL: c_int = 1100;
/// Mirrors `FC_UNARY` (ggml-metal-impl.h:115 @cd959bae9).
pub const FC_UNARY: c_int = 1200;
/// Mirrors `FC_BIN` (ggml-metal-impl.h:116 @cd959bae9).
pub const FC_BIN: c_int = 1300;
/// Mirrors `FC_SUM_ROWS` (ggml-metal-impl.h:117 @cd959bae9).
pub const FC_SUM_ROWS: c_int = 1400;
/// Mirrors `FC_UPSCALE` (ggml-metal-impl.h:118 @cd959bae9).
pub const FC_UPSCALE: c_int = 1500;
/// Mirrors `FC_GATED_DELTA_NET` (ggml-metal-impl.h:119 @cd959bae9).
pub const FC_GATED_DELTA_NET: c_int = 1600;
/// Mirrors `FC_NORM` (ggml-metal-impl.h:120 @cd959bae9).
pub const FC_NORM: c_int = 1700;
/// Mirrors `FC_TOPK_MOE` (ggml-metal-impl.h:121 @cd959bae9).
pub const FC_TOPK_MOE: c_int = 1800;
/// Mirrors `FC_MOE_REDUCE` (ggml-metal-impl.h:122 @cd959bae9).
pub const FC_MOE_REDUCE: c_int = 1900;
/// Mirrors `FC_DSV4_HC` (ggml-metal-impl.h:123 @cd959bae9).
pub const FC_DSV4_HC: c_int = 2000;
/// Mirrors `FC_PAD` (ggml-metal-impl.h:124 @cd959bae9).
pub const FC_PAD: c_int = 2100;
/// Mirrors `FC_FLASH_ATTN_EXT_TENSOR` (ggml-metal-impl.h:125 @cd959bae9).
pub const FC_FLASH_ATTN_EXT_TENSOR: c_int = 2200;
/// Mirrors `FC_LIGHTNING_INDEXER` (ggml-metal-impl.h:126 @cd959bae9).
///
/// **The same value as `FC_FLASH_ATTN_EXT_TENSOR`**, upstream's own
/// collision. Harmless there -- the two kernels are in different libraries
/// and each pipeline gets its own constant values -- and reproduced, not
/// corrected, because the slot is what the kernel reads.
pub const FC_LIGHTNING_INDEXER: c_int = 2200;
/// Mirrors `FC_MUL_MV_MMA` (ggml-metal-impl.h:127 @cd959bae9).
pub const FC_MUL_MV_MMA: c_int = 2300;
/// Mirrors `FC_TOP_K_MASK` (ggml-metal-impl.h:128 @cd959bae9).
pub const FC_TOP_K_MASK: c_int = 2400;
/// Mirrors `OP_FLASH_ATTN_EXT_NQPSG` (ggml-metal-impl.h:131 @cd959bae9).
pub const OP_FLASH_ATTN_EXT_NQPSG: c_int = 8;
/// Mirrors `OP_FLASH_ATTN_EXT_NCPSG` (ggml-metal-impl.h:132 @cd959bae9).
pub const OP_FLASH_ATTN_EXT_NCPSG: c_int = 64;
/// Mirrors `OP_FLASH_ATTN_EXT_TENSOR_NQPSG` (ggml-metal-impl.h:134 @cd959bae9).
pub const OP_FLASH_ATTN_EXT_TENSOR_NQPSG: c_int = 32;
/// Mirrors `OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE` (ggml-metal-impl.h:135
/// @cd959bae9): queries per threadgroup for heads of 512 and up.
pub const OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE: c_int = 16;
/// Mirrors `OP_FLASH_ATTN_EXT_TENSOR_NCPSG` (ggml-metal-impl.h:136 @cd959bae9).
pub const OP_FLASH_ATTN_EXT_TENSOR_NCPSG: c_int = 64;
/// Mirrors `OP_FLASH_ATTN_EXT_TENSOR_NSG` (ggml-metal-impl.h:137 @cd959bae9).
pub const OP_FLASH_ATTN_EXT_TENSOR_NSG: c_int = 8;
/// Mirrors `OP_FLASH_ATTN_EXT_VEC_NQPSG` (ggml-metal-impl.h:139 @cd959bae9).
pub const OP_FLASH_ATTN_EXT_VEC_NQPSG: c_int = 1;
/// Mirrors `OP_FLASH_ATTN_EXT_VEC_NCPSG` (ggml-metal-impl.h:140 @cd959bae9).
pub const OP_FLASH_ATTN_EXT_VEC_NCPSG: c_int = 32;
/// Mirrors `OP_LIGHTNING_INDEXER_DK` (ggml-metal-impl.h:142 @cd959bae9).
pub const OP_LIGHTNING_INDEXER_DK: c_int = 128;
/// Mirrors `OP_LIGHTNING_INDEXER_NHPTG` (ggml-metal-impl.h:143 @cd959bae9).
pub const OP_LIGHTNING_INDEXER_NHPTG: c_int = 8;
/// Mirrors `OP_LIGHTNING_INDEXER_NKPSG` (ggml-metal-impl.h:144 @cd959bae9).
pub const OP_LIGHTNING_INDEXER_NKPSG: c_int = 8;
/// Mirrors `OP_LIGHTNING_INDEXER_NSG` (ggml-metal-impl.h:145 @cd959bae9).
pub const OP_LIGHTNING_INDEXER_NSG: c_int = 8;
/// Mirrors `OP_LIGHTNING_INDEXER_NBPTG` (ggml-metal-impl.h:146 @cd959bae9).
pub const OP_LIGHTNING_INDEXER_NBPTG: c_int = 8;
/// Mirrors `OP_UNARY_NUM_SCALE` (ggml-metal-impl.h:148 @cd959bae9).
pub const OP_UNARY_NUM_SCALE: c_int = 10;
/// Mirrors `OP_UNARY_NUM_FILL` (ggml-metal-impl.h:149 @cd959bae9).
pub const OP_UNARY_NUM_FILL: c_int = 11;
/// Mirrors `OP_UNARY_NUM_CLAMP` (ggml-metal-impl.h:150 @cd959bae9).
pub const OP_UNARY_NUM_CLAMP: c_int = 12;
/// Mirrors `OP_UNARY_NUM_SQR` (ggml-metal-impl.h:151 @cd959bae9).
pub const OP_UNARY_NUM_SQR: c_int = 13;
/// Mirrors `OP_UNARY_NUM_SQRT` (ggml-metal-impl.h:152 @cd959bae9).
pub const OP_UNARY_NUM_SQRT: c_int = 14;
/// Mirrors `OP_UNARY_NUM_SIN` (ggml-metal-impl.h:153 @cd959bae9).
pub const OP_UNARY_NUM_SIN: c_int = 15;
/// Mirrors `OP_UNARY_NUM_COS` (ggml-metal-impl.h:154 @cd959bae9).
pub const OP_UNARY_NUM_COS: c_int = 16;
/// Mirrors `OP_UNARY_NUM_LOG` (ggml-metal-impl.h:155 @cd959bae9).
pub const OP_UNARY_NUM_LOG: c_int = 17;
/// Mirrors `OP_UNARY_NUM_LEAKY_RELU` (ggml-metal-impl.h:156 @cd959bae9).
pub const OP_UNARY_NUM_LEAKY_RELU: c_int = 18;
/// Mirrors `OP_UNARY_NUM_TANH` (ggml-metal-impl.h:158 @cd959bae9).
pub const OP_UNARY_NUM_TANH: c_int = 100;
/// Mirrors `OP_UNARY_NUM_RELU` (ggml-metal-impl.h:159 @cd959bae9).
pub const OP_UNARY_NUM_RELU: c_int = 101;
/// Mirrors `OP_UNARY_NUM_SIGMOID` (ggml-metal-impl.h:160 @cd959bae9).
pub const OP_UNARY_NUM_SIGMOID: c_int = 102;
/// Mirrors `OP_UNARY_NUM_GELU` (ggml-metal-impl.h:161 @cd959bae9).
pub const OP_UNARY_NUM_GELU: c_int = 103;
/// Mirrors `OP_UNARY_NUM_GELU_ERF` (ggml-metal-impl.h:162 @cd959bae9).
pub const OP_UNARY_NUM_GELU_ERF: c_int = 104;
/// Mirrors `OP_UNARY_NUM_GELU_QUICK` (ggml-metal-impl.h:163 @cd959bae9).
pub const OP_UNARY_NUM_GELU_QUICK: c_int = 105;
/// Mirrors `OP_UNARY_NUM_SILU` (ggml-metal-impl.h:164 @cd959bae9).
pub const OP_UNARY_NUM_SILU: c_int = 106;
/// Mirrors `OP_UNARY_NUM_ELU` (ggml-metal-impl.h:165 @cd959bae9).
pub const OP_UNARY_NUM_ELU: c_int = 107;
/// Mirrors `OP_UNARY_NUM_NEG` (ggml-metal-impl.h:166 @cd959bae9).
pub const OP_UNARY_NUM_NEG: c_int = 108;
/// Mirrors `OP_UNARY_NUM_ABS` (ggml-metal-impl.h:167 @cd959bae9).
pub const OP_UNARY_NUM_ABS: c_int = 109;
/// Mirrors `OP_UNARY_NUM_SGN` (ggml-metal-impl.h:168 @cd959bae9).
pub const OP_UNARY_NUM_SGN: c_int = 110;
/// Mirrors `OP_UNARY_NUM_STEP` (ggml-metal-impl.h:169 @cd959bae9).
pub const OP_UNARY_NUM_STEP: c_int = 111;
/// Mirrors `OP_UNARY_NUM_HARDSWISH` (ggml-metal-impl.h:170 @cd959bae9).
pub const OP_UNARY_NUM_HARDSWISH: c_int = 112;
/// Mirrors `OP_UNARY_NUM_HARDSIGMOID` (ggml-metal-impl.h:171 @cd959bae9).
pub const OP_UNARY_NUM_HARDSIGMOID: c_int = 113;
/// Mirrors `OP_UNARY_NUM_EXP` (ggml-metal-impl.h:172 @cd959bae9).
pub const OP_UNARY_NUM_EXP: c_int = 114;
/// Mirrors `OP_UNARY_NUM_SOFTPLUS` (ggml-metal-impl.h:173 @cd959bae9).
pub const OP_UNARY_NUM_SOFTPLUS: c_int = 115;
/// Mirrors `OP_UNARY_NUM_EXPM1` (ggml-metal-impl.h:174 @cd959bae9).
pub const OP_UNARY_NUM_EXPM1: c_int = 116;
/// Mirrors `OP_UNARY_NUM_FLOOR` (ggml-metal-impl.h:175 @cd959bae9).
pub const OP_UNARY_NUM_FLOOR: c_int = 117;
/// Mirrors `OP_UNARY_NUM_CEIL` (ggml-metal-impl.h:176 @cd959bae9).
pub const OP_UNARY_NUM_CEIL: c_int = 118;
/// Mirrors `OP_UNARY_NUM_ROUND` (ggml-metal-impl.h:177 @cd959bae9).
pub const OP_UNARY_NUM_ROUND: c_int = 119;
/// Mirrors `OP_UNARY_NUM_TRUNC` (ggml-metal-impl.h:178 @cd959bae9).
pub const OP_UNARY_NUM_TRUNC: c_int = 120;
/// Mirrors `OP_UNARY_NUM_XIELU` (ggml-metal-impl.h:179 @cd959bae9).
pub const OP_UNARY_NUM_XIELU: c_int = 121;
/// Mirrors `OP_SUM_ROWS_NUM_SUM_ROWS` (ggml-metal-impl.h:181 @cd959bae9).
pub const OP_SUM_ROWS_NUM_SUM_ROWS: c_int = 10;
/// Mirrors `OP_SUM_ROWS_NUM_MEAN` (ggml-metal-impl.h:182 @cd959bae9).
pub const OP_SUM_ROWS_NUM_MEAN: c_int = 11;
/// Mirrors `OP_SSM_SCAN_SSD_CS` (ggml-metal-impl.h:184 @cd959bae9): the SSD
/// scan's chunk size.
pub const OP_SSM_SCAN_SSD_CS: c_int = 64;
/// Mirrors `OP_SSM_SCAN_SSD_HD` (ggml-metal-impl.h:185 @cd959bae9): the head
/// dimension the MMA kernel is specialised for.
pub const OP_SSM_SCAN_SSD_HD: c_int = 64;
/// Mirrors `OP_SSM_SCAN_SSD_NSG` (ggml-metal-impl.h:186 @cd959bae9): simdgroups
/// per threadgroup.
pub const OP_SSM_SCAN_SSD_NSG: c_int = 4;
/// Mirrors `GGML_METAL_FWHT_TG_MIN_N` (ggml-metal-impl.h:1265 @cd959bae9):
/// FWHT widths at or above this run the threadgroup kernel, one row per
/// threadgroup of `GGML_METAL_FWHT_TG_NT` threads.
pub const GGML_METAL_FWHT_TG_MIN_N: c_int = 512;
/// Mirrors `GGML_METAL_FWHT_TG_NT` (ggml-metal-impl.h:1266 @cd959bae9).
pub const GGML_METAL_FWHT_TG_NT: c_int = 256;
/// Mirrors `N_MM_NK_TOTAL` (ggml-metal-impl.h:10 @cd959bae9).
///
/// The one computed define: the C writes `(SZ_SIMDGROUP * N_MM_NK)`, kept as an
/// expression here so it tracks its operands.
pub const N_MM_NK_TOTAL: c_int = SZ_SIMDGROUP * N_MM_NK;

/// The count the gate cross-checks, so a constant added upstream and not
/// here shows up as a mismatch rather than as nothing.
pub const count: usize = 138;

// -----------------------------------------------------------------------------
// The gate's view
//
// `harness/struct_layout.cpp` includes the real header and compares these
// against its own `#define`s. Name and value travel together so a
// reordering cannot let two wrong entries cancel. Every value is the
// declaration above it, not a second copy of the number: until v0.6.0 the
// table held literals, so a wrong `pub const` -- the one the encoders
// read -- passed the gate beside a right literal.

const Entry = struct { name: [*:0]const u8, value: c_long };

const all = [_]Entry{
    .{ .name = "SZ_SIMDGROUP", .value = SZ_SIMDGROUP },
    .{ .name = "N_MM_NK", .value = N_MM_NK },
    .{ .name = "N_MM_BLOCK_X", .value = N_MM_BLOCK_X },
    .{ .name = "N_MM_BLOCK_Y", .value = N_MM_BLOCK_Y },
    .{ .name = "N_MM_SIMD_GROUP_X", .value = N_MM_SIMD_GROUP_X },
    .{ .name = "N_MM_SIMD_GROUP_Y", .value = N_MM_SIMD_GROUP_Y },
    .{ .name = "N_ARGMAX_NPART_MAX", .value = N_ARGMAX_NPART_MAX },
    .{ .name = "N_ARGMAX_CHUNK", .value = N_ARGMAX_CHUNK },
    .{ .name = "N_R0_Q1_0", .value = N_R0_Q1_0 },
    .{ .name = "N_SG_Q1_0", .value = N_SG_Q1_0 },
    .{ .name = "N_R0_Q2_0", .value = N_R0_Q2_0 },
    .{ .name = "N_SG_Q2_0", .value = N_SG_Q2_0 },
    .{ .name = "N_R0_Q4_0", .value = N_R0_Q4_0 },
    .{ .name = "N_SG_Q4_0", .value = N_SG_Q4_0 },
    .{ .name = "N_R0_Q4_1", .value = N_R0_Q4_1 },
    .{ .name = "N_SG_Q4_1", .value = N_SG_Q4_1 },
    .{ .name = "N_R0_Q5_0", .value = N_R0_Q5_0 },
    .{ .name = "N_SG_Q5_0", .value = N_SG_Q5_0 },
    .{ .name = "N_R0_Q5_1", .value = N_R0_Q5_1 },
    .{ .name = "N_SG_Q5_1", .value = N_SG_Q5_1 },
    .{ .name = "N_R0_Q8_0", .value = N_R0_Q8_0 },
    .{ .name = "N_SG_Q8_0", .value = N_SG_Q8_0 },
    .{ .name = "N_R0_MXFP4", .value = N_R0_MXFP4 },
    .{ .name = "N_SG_MXFP4", .value = N_SG_MXFP4 },
    .{ .name = "N_R0_Q2_K", .value = N_R0_Q2_K },
    .{ .name = "N_SG_Q2_K", .value = N_SG_Q2_K },
    .{ .name = "N_R0_Q3_K", .value = N_R0_Q3_K },
    .{ .name = "N_SG_Q3_K", .value = N_SG_Q3_K },
    .{ .name = "N_R0_Q4_K", .value = N_R0_Q4_K },
    .{ .name = "N_SG_Q4_K", .value = N_SG_Q4_K },
    .{ .name = "N_R0_Q5_K", .value = N_R0_Q5_K },
    .{ .name = "N_SG_Q5_K", .value = N_SG_Q5_K },
    .{ .name = "N_R0_Q6_K", .value = N_R0_Q6_K },
    .{ .name = "N_SG_Q6_K", .value = N_SG_Q6_K },
    .{ .name = "N_R0_IQ1_S", .value = N_R0_IQ1_S },
    .{ .name = "N_SG_IQ1_S", .value = N_SG_IQ1_S },
    .{ .name = "N_R0_IQ1_S_SPLIT", .value = N_R0_IQ1_S_SPLIT },
    .{ .name = "N_R0_IQ1_M", .value = N_R0_IQ1_M },
    .{ .name = "N_SG_IQ1_M", .value = N_SG_IQ1_M },
    .{ .name = "N_R0_IQ1_M_SPLIT", .value = N_R0_IQ1_M_SPLIT },
    .{ .name = "N_R0_IQ2_XXS", .value = N_R0_IQ2_XXS },
    .{ .name = "N_SG_IQ2_XXS", .value = N_SG_IQ2_XXS },
    .{ .name = "N_R0_IQ2_XXS_SPLIT", .value = N_R0_IQ2_XXS_SPLIT },
    .{ .name = "N_R0_IQ2_XS", .value = N_R0_IQ2_XS },
    .{ .name = "N_SG_IQ2_XS", .value = N_SG_IQ2_XS },
    .{ .name = "N_R0_IQ2_XS_SPLIT", .value = N_R0_IQ2_XS_SPLIT },
    .{ .name = "N_R0_IQ2_S", .value = N_R0_IQ2_S },
    .{ .name = "N_SG_IQ2_S", .value = N_SG_IQ2_S },
    .{ .name = "N_R0_IQ2_S_SPLIT", .value = N_R0_IQ2_S_SPLIT },
    .{ .name = "N_R0_IQ3_XXS", .value = N_R0_IQ3_XXS },
    .{ .name = "N_SG_IQ3_XXS", .value = N_SG_IQ3_XXS },
    .{ .name = "N_R0_IQ3_XXS_SPLIT", .value = N_R0_IQ3_XXS_SPLIT },
    .{ .name = "N_R0_IQ3_S", .value = N_R0_IQ3_S },
    .{ .name = "N_SG_IQ3_S", .value = N_SG_IQ3_S },
    .{ .name = "N_R0_IQ3_S_SPLIT", .value = N_R0_IQ3_S_SPLIT },
    .{ .name = "N_R0_IQ4_NL", .value = N_R0_IQ4_NL },
    .{ .name = "N_SG_IQ4_NL", .value = N_SG_IQ4_NL },
    .{ .name = "N_R0_IQ4_XS", .value = N_R0_IQ4_XS },
    .{ .name = "N_SG_IQ4_XS", .value = N_SG_IQ4_XS },
    .{ .name = "N_R0_TQ2_0", .value = N_R0_TQ2_0 },
    .{ .name = "N_SG_TQ2_0", .value = N_SG_TQ2_0 },
    .{ .name = "FC_FLASH_ATTN_EXT_PAD", .value = FC_FLASH_ATTN_EXT_PAD },
    .{ .name = "FC_FLASH_ATTN_EXT_BLK", .value = FC_FLASH_ATTN_EXT_BLK },
    .{ .name = "FC_FLASH_ATTN_EXT", .value = FC_FLASH_ATTN_EXT },
    .{ .name = "FC_FLASH_ATTN_EXT_VEC", .value = FC_FLASH_ATTN_EXT_VEC },
    .{ .name = "FC_FLASH_ATTN_EXT_VEC_REDUCE", .value = FC_FLASH_ATTN_EXT_VEC_REDUCE },
    .{ .name = "FC_MUL_MV", .value = FC_MUL_MV },
    .{ .name = "FC_MUL_MM", .value = FC_MUL_MM },
    .{ .name = "FC_ROPE", .value = FC_ROPE },
    .{ .name = "FC_SSM_CONV", .value = FC_SSM_CONV },
    .{ .name = "FC_SOLVE_TRI", .value = FC_SOLVE_TRI },
    .{ .name = "FC_COUNT_EQUAL", .value = FC_COUNT_EQUAL },
    .{ .name = "FC_UNARY", .value = FC_UNARY },
    .{ .name = "FC_BIN", .value = FC_BIN },
    .{ .name = "FC_SUM_ROWS", .value = FC_SUM_ROWS },
    .{ .name = "FC_UPSCALE", .value = FC_UPSCALE },
    .{ .name = "FC_GATED_DELTA_NET", .value = FC_GATED_DELTA_NET },
    .{ .name = "FC_NORM", .value = FC_NORM },
    .{ .name = "FC_TOPK_MOE", .value = FC_TOPK_MOE },
    .{ .name = "FC_MOE_REDUCE", .value = FC_MOE_REDUCE },
    .{ .name = "FC_DSV4_HC", .value = FC_DSV4_HC },
    .{ .name = "FC_PAD", .value = FC_PAD },
    .{ .name = "FC_FLASH_ATTN_EXT_TENSOR", .value = FC_FLASH_ATTN_EXT_TENSOR },
    .{ .name = "FC_LIGHTNING_INDEXER", .value = FC_LIGHTNING_INDEXER },
    .{ .name = "FC_MUL_MV_MMA", .value = FC_MUL_MV_MMA },
    .{ .name = "FC_TOP_K_MASK", .value = FC_TOP_K_MASK },
    .{ .name = "OP_FLASH_ATTN_EXT_NQPSG", .value = OP_FLASH_ATTN_EXT_NQPSG },
    .{ .name = "OP_FLASH_ATTN_EXT_NCPSG", .value = OP_FLASH_ATTN_EXT_NCPSG },
    .{ .name = "OP_FLASH_ATTN_EXT_TENSOR_NQPSG", .value = OP_FLASH_ATTN_EXT_TENSOR_NQPSG },
    .{ .name = "OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE", .value = OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE },
    .{ .name = "OP_FLASH_ATTN_EXT_TENSOR_NCPSG", .value = OP_FLASH_ATTN_EXT_TENSOR_NCPSG },
    .{ .name = "OP_FLASH_ATTN_EXT_TENSOR_NSG", .value = OP_FLASH_ATTN_EXT_TENSOR_NSG },
    .{ .name = "OP_FLASH_ATTN_EXT_VEC_NQPSG", .value = OP_FLASH_ATTN_EXT_VEC_NQPSG },
    .{ .name = "OP_FLASH_ATTN_EXT_VEC_NCPSG", .value = OP_FLASH_ATTN_EXT_VEC_NCPSG },
    .{ .name = "OP_LIGHTNING_INDEXER_DK", .value = OP_LIGHTNING_INDEXER_DK },
    .{ .name = "OP_LIGHTNING_INDEXER_NHPTG", .value = OP_LIGHTNING_INDEXER_NHPTG },
    .{ .name = "OP_LIGHTNING_INDEXER_NKPSG", .value = OP_LIGHTNING_INDEXER_NKPSG },
    .{ .name = "OP_LIGHTNING_INDEXER_NSG", .value = OP_LIGHTNING_INDEXER_NSG },
    .{ .name = "OP_LIGHTNING_INDEXER_NBPTG", .value = OP_LIGHTNING_INDEXER_NBPTG },
    .{ .name = "OP_UNARY_NUM_SCALE", .value = OP_UNARY_NUM_SCALE },
    .{ .name = "OP_UNARY_NUM_FILL", .value = OP_UNARY_NUM_FILL },
    .{ .name = "OP_UNARY_NUM_CLAMP", .value = OP_UNARY_NUM_CLAMP },
    .{ .name = "OP_UNARY_NUM_SQR", .value = OP_UNARY_NUM_SQR },
    .{ .name = "OP_UNARY_NUM_SQRT", .value = OP_UNARY_NUM_SQRT },
    .{ .name = "OP_UNARY_NUM_SIN", .value = OP_UNARY_NUM_SIN },
    .{ .name = "OP_UNARY_NUM_COS", .value = OP_UNARY_NUM_COS },
    .{ .name = "OP_UNARY_NUM_LOG", .value = OP_UNARY_NUM_LOG },
    .{ .name = "OP_UNARY_NUM_LEAKY_RELU", .value = OP_UNARY_NUM_LEAKY_RELU },
    .{ .name = "OP_UNARY_NUM_TANH", .value = OP_UNARY_NUM_TANH },
    .{ .name = "OP_UNARY_NUM_RELU", .value = OP_UNARY_NUM_RELU },
    .{ .name = "OP_UNARY_NUM_SIGMOID", .value = OP_UNARY_NUM_SIGMOID },
    .{ .name = "OP_UNARY_NUM_GELU", .value = OP_UNARY_NUM_GELU },
    .{ .name = "OP_UNARY_NUM_GELU_ERF", .value = OP_UNARY_NUM_GELU_ERF },
    .{ .name = "OP_UNARY_NUM_GELU_QUICK", .value = OP_UNARY_NUM_GELU_QUICK },
    .{ .name = "OP_UNARY_NUM_SILU", .value = OP_UNARY_NUM_SILU },
    .{ .name = "OP_UNARY_NUM_ELU", .value = OP_UNARY_NUM_ELU },
    .{ .name = "OP_UNARY_NUM_NEG", .value = OP_UNARY_NUM_NEG },
    .{ .name = "OP_UNARY_NUM_ABS", .value = OP_UNARY_NUM_ABS },
    .{ .name = "OP_UNARY_NUM_SGN", .value = OP_UNARY_NUM_SGN },
    .{ .name = "OP_UNARY_NUM_STEP", .value = OP_UNARY_NUM_STEP },
    .{ .name = "OP_UNARY_NUM_HARDSWISH", .value = OP_UNARY_NUM_HARDSWISH },
    .{ .name = "OP_UNARY_NUM_HARDSIGMOID", .value = OP_UNARY_NUM_HARDSIGMOID },
    .{ .name = "OP_UNARY_NUM_EXP", .value = OP_UNARY_NUM_EXP },
    .{ .name = "OP_UNARY_NUM_SOFTPLUS", .value = OP_UNARY_NUM_SOFTPLUS },
    .{ .name = "OP_UNARY_NUM_EXPM1", .value = OP_UNARY_NUM_EXPM1 },
    .{ .name = "OP_UNARY_NUM_FLOOR", .value = OP_UNARY_NUM_FLOOR },
    .{ .name = "OP_UNARY_NUM_CEIL", .value = OP_UNARY_NUM_CEIL },
    .{ .name = "OP_UNARY_NUM_ROUND", .value = OP_UNARY_NUM_ROUND },
    .{ .name = "OP_UNARY_NUM_TRUNC", .value = OP_UNARY_NUM_TRUNC },
    .{ .name = "OP_UNARY_NUM_XIELU", .value = OP_UNARY_NUM_XIELU },
    .{ .name = "OP_SUM_ROWS_NUM_SUM_ROWS", .value = OP_SUM_ROWS_NUM_SUM_ROWS },
    .{ .name = "OP_SUM_ROWS_NUM_MEAN", .value = OP_SUM_ROWS_NUM_MEAN },
    .{ .name = "OP_SSM_SCAN_SSD_CS", .value = OP_SSM_SCAN_SSD_CS },
    .{ .name = "OP_SSM_SCAN_SSD_HD", .value = OP_SSM_SCAN_SSD_HD },
    .{ .name = "OP_SSM_SCAN_SSD_NSG", .value = OP_SSM_SCAN_SSD_NSG },
    .{ .name = "GGML_METAL_FWHT_TG_MIN_N", .value = GGML_METAL_FWHT_TG_MIN_N },
    .{ .name = "GGML_METAL_FWHT_TG_NT", .value = GGML_METAL_FWHT_TG_NT },
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
