//! The function-constant indices, op numbers and kernel-shape constants
//! the Metal kernel names and specialisations carry.
//!
//! # Provenance
//!
//! Mirrors the `FC_*`, `OP_*`, `N_*` and `SZ_*` `#define`s of
//! `llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h` at v0.3.0
//! (`c1d0e7a00`).
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

/// Mirrors `SZ_SIMDGROUP` (ggml-metal-impl.h:8 @c1d0e7a00).
pub const SZ_SIMDGROUP: c_int = 16;
/// Mirrors `N_MM_NK` (ggml-metal-impl.h:9 @c1d0e7a00).
pub const N_MM_NK: c_int = 2;
/// Mirrors `N_MM_BLOCK_X` (ggml-metal-impl.h:12 @c1d0e7a00).
pub const N_MM_BLOCK_X: c_int = 4;
/// Mirrors `N_MM_BLOCK_Y` (ggml-metal-impl.h:13 @c1d0e7a00).
pub const N_MM_BLOCK_Y: c_int = 2;
/// Mirrors `N_MM_SIMD_GROUP_X` (ggml-metal-impl.h:14 @c1d0e7a00).
pub const N_MM_SIMD_GROUP_X: c_int = 2;
/// Mirrors `N_MM_SIMD_GROUP_Y` (ggml-metal-impl.h:15 @c1d0e7a00).
pub const N_MM_SIMD_GROUP_Y: c_int = 2;
/// Mirrors `N_R0_Q1_0` (ggml-metal-impl.h:24 @c1d0e7a00).
pub const N_R0_Q1_0: c_int = 8;
/// Mirrors `N_SG_Q1_0` (ggml-metal-impl.h:25 @c1d0e7a00).
pub const N_SG_Q1_0: c_int = 2;
/// Mirrors `N_R0_Q2_0` (ggml-metal-impl.h:27 @c1d0e7a00).
pub const N_R0_Q2_0: c_int = 8;
/// Mirrors `N_SG_Q2_0` (ggml-metal-impl.h:28 @c1d0e7a00).
pub const N_SG_Q2_0: c_int = 2;
/// Mirrors `N_R0_Q4_0` (ggml-metal-impl.h:30 @c1d0e7a00).
pub const N_R0_Q4_0: c_int = 4;
/// Mirrors `N_SG_Q4_0` (ggml-metal-impl.h:31 @c1d0e7a00).
pub const N_SG_Q4_0: c_int = 2;
/// Mirrors `N_R0_Q4_1` (ggml-metal-impl.h:33 @c1d0e7a00).
pub const N_R0_Q4_1: c_int = 4;
/// Mirrors `N_SG_Q4_1` (ggml-metal-impl.h:34 @c1d0e7a00).
pub const N_SG_Q4_1: c_int = 2;
/// Mirrors `N_R0_Q5_0` (ggml-metal-impl.h:36 @c1d0e7a00).
pub const N_R0_Q5_0: c_int = 4;
/// Mirrors `N_SG_Q5_0` (ggml-metal-impl.h:37 @c1d0e7a00).
pub const N_SG_Q5_0: c_int = 2;
/// Mirrors `N_R0_Q5_1` (ggml-metal-impl.h:39 @c1d0e7a00).
pub const N_R0_Q5_1: c_int = 4;
/// Mirrors `N_SG_Q5_1` (ggml-metal-impl.h:40 @c1d0e7a00).
pub const N_SG_Q5_1: c_int = 2;
/// Mirrors `N_R0_Q8_0` (ggml-metal-impl.h:42 @c1d0e7a00).
pub const N_R0_Q8_0: c_int = 2;
/// Mirrors `N_SG_Q8_0` (ggml-metal-impl.h:43 @c1d0e7a00).
pub const N_SG_Q8_0: c_int = 4;
/// Mirrors `N_R0_MXFP4` (ggml-metal-impl.h:45 @c1d0e7a00).
pub const N_R0_MXFP4: c_int = 2;
/// Mirrors `N_SG_MXFP4` (ggml-metal-impl.h:46 @c1d0e7a00).
pub const N_SG_MXFP4: c_int = 2;
/// Mirrors `N_R0_Q2_K` (ggml-metal-impl.h:48 @c1d0e7a00).
pub const N_R0_Q2_K: c_int = 4;
/// Mirrors `N_SG_Q2_K` (ggml-metal-impl.h:49 @c1d0e7a00).
pub const N_SG_Q2_K: c_int = 2;
/// Mirrors `N_R0_Q3_K` (ggml-metal-impl.h:51 @c1d0e7a00).
pub const N_R0_Q3_K: c_int = 2;
/// Mirrors `N_SG_Q3_K` (ggml-metal-impl.h:52 @c1d0e7a00).
pub const N_SG_Q3_K: c_int = 2;
/// Mirrors `N_R0_Q4_K` (ggml-metal-impl.h:54 @c1d0e7a00).
pub const N_R0_Q4_K: c_int = 2;
/// Mirrors `N_SG_Q4_K` (ggml-metal-impl.h:55 @c1d0e7a00).
pub const N_SG_Q4_K: c_int = 2;
/// Mirrors `N_R0_Q5_K` (ggml-metal-impl.h:57 @c1d0e7a00).
pub const N_R0_Q5_K: c_int = 1;
/// Mirrors `N_SG_Q5_K` (ggml-metal-impl.h:58 @c1d0e7a00).
pub const N_SG_Q5_K: c_int = 2;
/// Mirrors `N_R0_Q6_K` (ggml-metal-impl.h:60 @c1d0e7a00).
pub const N_R0_Q6_K: c_int = 2;
/// Mirrors `N_SG_Q6_K` (ggml-metal-impl.h:61 @c1d0e7a00).
pub const N_SG_Q6_K: c_int = 2;
/// Mirrors `N_R0_IQ1_S` (ggml-metal-impl.h:63 @c1d0e7a00).
pub const N_R0_IQ1_S: c_int = 4;
/// Mirrors `N_SG_IQ1_S` (ggml-metal-impl.h:64 @c1d0e7a00).
pub const N_SG_IQ1_S: c_int = 2;
/// Mirrors `N_R0_IQ1_M` (ggml-metal-impl.h:66 @c1d0e7a00).
pub const N_R0_IQ1_M: c_int = 4;
/// Mirrors `N_SG_IQ1_M` (ggml-metal-impl.h:67 @c1d0e7a00).
pub const N_SG_IQ1_M: c_int = 2;
/// Mirrors `N_R0_IQ2_XXS` (ggml-metal-impl.h:69 @c1d0e7a00).
pub const N_R0_IQ2_XXS: c_int = 4;
/// Mirrors `N_SG_IQ2_XXS` (ggml-metal-impl.h:70 @c1d0e7a00).
pub const N_SG_IQ2_XXS: c_int = 2;
/// Mirrors `N_R0_IQ2_XS` (ggml-metal-impl.h:72 @c1d0e7a00).
pub const N_R0_IQ2_XS: c_int = 4;
/// Mirrors `N_SG_IQ2_XS` (ggml-metal-impl.h:73 @c1d0e7a00).
pub const N_SG_IQ2_XS: c_int = 2;
/// Mirrors `N_R0_IQ2_S` (ggml-metal-impl.h:75 @c1d0e7a00).
pub const N_R0_IQ2_S: c_int = 4;
/// Mirrors `N_SG_IQ2_S` (ggml-metal-impl.h:76 @c1d0e7a00).
pub const N_SG_IQ2_S: c_int = 2;
/// Mirrors `N_R0_IQ3_XXS` (ggml-metal-impl.h:78 @c1d0e7a00).
pub const N_R0_IQ3_XXS: c_int = 4;
/// Mirrors `N_SG_IQ3_XXS` (ggml-metal-impl.h:79 @c1d0e7a00).
pub const N_SG_IQ3_XXS: c_int = 2;
/// Mirrors `N_R0_IQ3_S` (ggml-metal-impl.h:81 @c1d0e7a00).
pub const N_R0_IQ3_S: c_int = 4;
/// Mirrors `N_SG_IQ3_S` (ggml-metal-impl.h:82 @c1d0e7a00).
pub const N_SG_IQ3_S: c_int = 2;
/// Mirrors `N_R0_IQ4_NL` (ggml-metal-impl.h:84 @c1d0e7a00).
pub const N_R0_IQ4_NL: c_int = 2;
/// Mirrors `N_SG_IQ4_NL` (ggml-metal-impl.h:85 @c1d0e7a00).
pub const N_SG_IQ4_NL: c_int = 2;
/// Mirrors `N_R0_IQ4_XS` (ggml-metal-impl.h:87 @c1d0e7a00).
pub const N_R0_IQ4_XS: c_int = 2;
/// Mirrors `N_SG_IQ4_XS` (ggml-metal-impl.h:88 @c1d0e7a00).
pub const N_SG_IQ4_XS: c_int = 2;
/// Mirrors `N_R0_TQ2_0` (ggml-metal-impl.h:90 @c1d0e7a00).
pub const N_R0_TQ2_0: c_int = 4;
/// Mirrors `N_SG_TQ2_0` (ggml-metal-impl.h:91 @c1d0e7a00).
pub const N_SG_TQ2_0: c_int = 2;
/// Mirrors `FC_FLASH_ATTN_EXT_PAD` (ggml-metal-impl.h:94 @c1d0e7a00).
pub const FC_FLASH_ATTN_EXT_PAD: c_int = 100;
/// Mirrors `FC_FLASH_ATTN_EXT_BLK` (ggml-metal-impl.h:95 @c1d0e7a00).
pub const FC_FLASH_ATTN_EXT_BLK: c_int = 200;
/// Mirrors `FC_FLASH_ATTN_EXT` (ggml-metal-impl.h:96 @c1d0e7a00).
pub const FC_FLASH_ATTN_EXT: c_int = 300;
/// Mirrors `FC_FLASH_ATTN_EXT_VEC` (ggml-metal-impl.h:97 @c1d0e7a00).
pub const FC_FLASH_ATTN_EXT_VEC: c_int = 400;
/// Mirrors `FC_FLASH_ATTN_EXT_VEC_REDUCE` (ggml-metal-impl.h:98 @c1d0e7a00).
pub const FC_FLASH_ATTN_EXT_VEC_REDUCE: c_int = 500;
/// Mirrors `FC_MUL_MV` (ggml-metal-impl.h:99 @c1d0e7a00).
pub const FC_MUL_MV: c_int = 600;
/// Mirrors `FC_MUL_MM` (ggml-metal-impl.h:100 @c1d0e7a00).
pub const FC_MUL_MM: c_int = 700;
/// Mirrors `FC_ROPE` (ggml-metal-impl.h:101 @c1d0e7a00).
pub const FC_ROPE: c_int = 800;
/// Mirrors `FC_SSM_CONV` (ggml-metal-impl.h:102 @c1d0e7a00).
pub const FC_SSM_CONV: c_int = 900;
/// Mirrors `FC_SOLVE_TRI` (ggml-metal-impl.h:103 @c1d0e7a00).
pub const FC_SOLVE_TRI: c_int = 1000;
/// Mirrors `FC_COUNT_EQUAL` (ggml-metal-impl.h:104 @c1d0e7a00).
pub const FC_COUNT_EQUAL: c_int = 1100;
/// Mirrors `FC_UNARY` (ggml-metal-impl.h:105 @c1d0e7a00).
pub const FC_UNARY: c_int = 1200;
/// Mirrors `FC_BIN` (ggml-metal-impl.h:106 @c1d0e7a00).
pub const FC_BIN: c_int = 1300;
/// Mirrors `FC_SUM_ROWS` (ggml-metal-impl.h:107 @c1d0e7a00).
pub const FC_SUM_ROWS: c_int = 1400;
/// Mirrors `FC_UPSCALE` (ggml-metal-impl.h:108 @c1d0e7a00).
pub const FC_UPSCALE: c_int = 1500;
/// Mirrors `FC_GATED_DELTA_NET` (ggml-metal-impl.h:109 @c1d0e7a00).
pub const FC_GATED_DELTA_NET: c_int = 1600;
/// Mirrors `OP_FLASH_ATTN_EXT_NQPSG` (ggml-metal-impl.h:112 @c1d0e7a00).
pub const OP_FLASH_ATTN_EXT_NQPSG: c_int = 8;
/// Mirrors `OP_FLASH_ATTN_EXT_NCPSG` (ggml-metal-impl.h:113 @c1d0e7a00).
pub const OP_FLASH_ATTN_EXT_NCPSG: c_int = 64;
/// Mirrors `OP_FLASH_ATTN_EXT_VEC_NQPSG` (ggml-metal-impl.h:115 @c1d0e7a00).
pub const OP_FLASH_ATTN_EXT_VEC_NQPSG: c_int = 1;
/// Mirrors `OP_FLASH_ATTN_EXT_VEC_NCPSG` (ggml-metal-impl.h:116 @c1d0e7a00).
pub const OP_FLASH_ATTN_EXT_VEC_NCPSG: c_int = 32;
/// Mirrors `OP_LIGHTNING_INDEXER_DK` (ggml-metal-impl.h:118 @c1d0e7a00).
pub const OP_LIGHTNING_INDEXER_DK: c_int = 128;
/// Mirrors `OP_LIGHTNING_INDEXER_NH` (ggml-metal-impl.h:119 @c1d0e7a00).
pub const OP_LIGHTNING_INDEXER_NH: c_int = 64;
/// Mirrors `OP_LIGHTNING_INDEXER_NHPTG` (ggml-metal-impl.h:120 @c1d0e7a00).
pub const OP_LIGHTNING_INDEXER_NHPTG: c_int = 8;
/// Mirrors `OP_LIGHTNING_INDEXER_NKPSG` (ggml-metal-impl.h:121 @c1d0e7a00).
pub const OP_LIGHTNING_INDEXER_NKPSG: c_int = 8;
/// Mirrors `OP_LIGHTNING_INDEXER_NSG` (ggml-metal-impl.h:122 @c1d0e7a00).
pub const OP_LIGHTNING_INDEXER_NSG: c_int = 8;
/// Mirrors `OP_LIGHTNING_INDEXER_NBPTG` (ggml-metal-impl.h:123 @c1d0e7a00).
pub const OP_LIGHTNING_INDEXER_NBPTG: c_int = 8;
/// Mirrors `OP_UNARY_NUM_SCALE` (ggml-metal-impl.h:125 @c1d0e7a00).
pub const OP_UNARY_NUM_SCALE: c_int = 10;
/// Mirrors `OP_UNARY_NUM_FILL` (ggml-metal-impl.h:126 @c1d0e7a00).
pub const OP_UNARY_NUM_FILL: c_int = 11;
/// Mirrors `OP_UNARY_NUM_CLAMP` (ggml-metal-impl.h:127 @c1d0e7a00).
pub const OP_UNARY_NUM_CLAMP: c_int = 12;
/// Mirrors `OP_UNARY_NUM_SQR` (ggml-metal-impl.h:128 @c1d0e7a00).
pub const OP_UNARY_NUM_SQR: c_int = 13;
/// Mirrors `OP_UNARY_NUM_SQRT` (ggml-metal-impl.h:129 @c1d0e7a00).
pub const OP_UNARY_NUM_SQRT: c_int = 14;
/// Mirrors `OP_UNARY_NUM_SIN` (ggml-metal-impl.h:130 @c1d0e7a00).
pub const OP_UNARY_NUM_SIN: c_int = 15;
/// Mirrors `OP_UNARY_NUM_COS` (ggml-metal-impl.h:131 @c1d0e7a00).
pub const OP_UNARY_NUM_COS: c_int = 16;
/// Mirrors `OP_UNARY_NUM_LOG` (ggml-metal-impl.h:132 @c1d0e7a00).
pub const OP_UNARY_NUM_LOG: c_int = 17;
/// Mirrors `OP_UNARY_NUM_LEAKY_RELU` (ggml-metal-impl.h:133 @c1d0e7a00).
pub const OP_UNARY_NUM_LEAKY_RELU: c_int = 18;
/// Mirrors `OP_UNARY_NUM_TANH` (ggml-metal-impl.h:135 @c1d0e7a00).
pub const OP_UNARY_NUM_TANH: c_int = 100;
/// Mirrors `OP_UNARY_NUM_RELU` (ggml-metal-impl.h:136 @c1d0e7a00).
pub const OP_UNARY_NUM_RELU: c_int = 101;
/// Mirrors `OP_UNARY_NUM_SIGMOID` (ggml-metal-impl.h:137 @c1d0e7a00).
pub const OP_UNARY_NUM_SIGMOID: c_int = 102;
/// Mirrors `OP_UNARY_NUM_GELU` (ggml-metal-impl.h:138 @c1d0e7a00).
pub const OP_UNARY_NUM_GELU: c_int = 103;
/// Mirrors `OP_UNARY_NUM_GELU_ERF` (ggml-metal-impl.h:139 @c1d0e7a00).
pub const OP_UNARY_NUM_GELU_ERF: c_int = 104;
/// Mirrors `OP_UNARY_NUM_GELU_QUICK` (ggml-metal-impl.h:140 @c1d0e7a00).
pub const OP_UNARY_NUM_GELU_QUICK: c_int = 105;
/// Mirrors `OP_UNARY_NUM_SILU` (ggml-metal-impl.h:141 @c1d0e7a00).
pub const OP_UNARY_NUM_SILU: c_int = 106;
/// Mirrors `OP_UNARY_NUM_ELU` (ggml-metal-impl.h:142 @c1d0e7a00).
pub const OP_UNARY_NUM_ELU: c_int = 107;
/// Mirrors `OP_UNARY_NUM_NEG` (ggml-metal-impl.h:143 @c1d0e7a00).
pub const OP_UNARY_NUM_NEG: c_int = 108;
/// Mirrors `OP_UNARY_NUM_ABS` (ggml-metal-impl.h:144 @c1d0e7a00).
pub const OP_UNARY_NUM_ABS: c_int = 109;
/// Mirrors `OP_UNARY_NUM_SGN` (ggml-metal-impl.h:145 @c1d0e7a00).
pub const OP_UNARY_NUM_SGN: c_int = 110;
/// Mirrors `OP_UNARY_NUM_STEP` (ggml-metal-impl.h:146 @c1d0e7a00).
pub const OP_UNARY_NUM_STEP: c_int = 111;
/// Mirrors `OP_UNARY_NUM_HARDSWISH` (ggml-metal-impl.h:147 @c1d0e7a00).
pub const OP_UNARY_NUM_HARDSWISH: c_int = 112;
/// Mirrors `OP_UNARY_NUM_HARDSIGMOID` (ggml-metal-impl.h:148 @c1d0e7a00).
pub const OP_UNARY_NUM_HARDSIGMOID: c_int = 113;
/// Mirrors `OP_UNARY_NUM_EXP` (ggml-metal-impl.h:149 @c1d0e7a00).
pub const OP_UNARY_NUM_EXP: c_int = 114;
/// Mirrors `OP_UNARY_NUM_SOFTPLUS` (ggml-metal-impl.h:150 @c1d0e7a00).
pub const OP_UNARY_NUM_SOFTPLUS: c_int = 115;
/// Mirrors `OP_UNARY_NUM_EXPM1` (ggml-metal-impl.h:151 @c1d0e7a00).
pub const OP_UNARY_NUM_EXPM1: c_int = 116;
/// Mirrors `OP_UNARY_NUM_FLOOR` (ggml-metal-impl.h:152 @c1d0e7a00).
pub const OP_UNARY_NUM_FLOOR: c_int = 117;
/// Mirrors `OP_UNARY_NUM_CEIL` (ggml-metal-impl.h:153 @c1d0e7a00).
pub const OP_UNARY_NUM_CEIL: c_int = 118;
/// Mirrors `OP_UNARY_NUM_ROUND` (ggml-metal-impl.h:154 @c1d0e7a00).
pub const OP_UNARY_NUM_ROUND: c_int = 119;
/// Mirrors `OP_UNARY_NUM_TRUNC` (ggml-metal-impl.h:155 @c1d0e7a00).
pub const OP_UNARY_NUM_TRUNC: c_int = 120;
/// Mirrors `OP_UNARY_NUM_XIELU` (ggml-metal-impl.h:156 @c1d0e7a00).
pub const OP_UNARY_NUM_XIELU: c_int = 121;
/// Mirrors `OP_SUM_ROWS_NUM_SUM_ROWS` (ggml-metal-impl.h:158 @c1d0e7a00).
pub const OP_SUM_ROWS_NUM_SUM_ROWS: c_int = 10;
/// Mirrors `OP_SUM_ROWS_NUM_MEAN` (ggml-metal-impl.h:159 @c1d0e7a00).
pub const OP_SUM_ROWS_NUM_MEAN: c_int = 11;
/// Mirrors `N_MM_NK_TOTAL` (ggml-metal-impl.h:10 @c1d0e7a00).
///
/// The one computed define: the C writes `(SZ_SIMDGROUP * N_MM_NK)`, kept as an
/// expression here so it tracks its operands.
pub const N_MM_NK_TOTAL: c_int = SZ_SIMDGROUP * N_MM_NK;

/// The count the gate cross-checks, so a constant added upstream and not
/// here shows up as a mismatch rather than as nothing.
pub const count: usize = 112;

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
    .{ .name = "N_R0_IQ1_M", .value = 4 },
    .{ .name = "N_SG_IQ1_M", .value = 2 },
    .{ .name = "N_R0_IQ2_XXS", .value = 4 },
    .{ .name = "N_SG_IQ2_XXS", .value = 2 },
    .{ .name = "N_R0_IQ2_XS", .value = 4 },
    .{ .name = "N_SG_IQ2_XS", .value = 2 },
    .{ .name = "N_R0_IQ2_S", .value = 4 },
    .{ .name = "N_SG_IQ2_S", .value = 2 },
    .{ .name = "N_R0_IQ3_XXS", .value = 4 },
    .{ .name = "N_SG_IQ3_XXS", .value = 2 },
    .{ .name = "N_R0_IQ3_S", .value = 4 },
    .{ .name = "N_SG_IQ3_S", .value = 2 },
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
