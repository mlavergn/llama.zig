//! x86's `arch-fallback.h` renames for `repack.cpp`: the 28 kernels
//! `arch/x86/repack.cpp` has no version of, so the generic body is the real
//! one.
//!
//! # Provenance
//!
//! **Not a port of a function.** It reproduces
//! `llama.cpp/ggml/src/ggml-cpu/arch-fallback.h` at v1.0.0 (`b1b770d90`),
//! lines 88-119, the `__x86_64__` block, which `#define`s each generic to
//! the bare name before `repack.cpp` compiles. The bodies are the generic
//! ports in `cpu/repack/`; this exports each under the name the C gives it.
//! `arm/fallback.zig` does the same for ARM's eight.

const repack = @import("../module.zig");

comptime {
    @export(&repack.quantize.ggml_quantize_mat_q8_0_4x4_generic, .{ .name = "ggml_quantize_mat_q8_0_4x4" });
    @export(&repack.quantize.ggml_quantize_mat_q8_K_4x4_generic, .{ .name = "ggml_quantize_mat_q8_K_4x4" });

    @export(&repack.q1_0.ggml_gemv_q1_0_4x4_q8_0_generic, .{ .name = "ggml_gemv_q1_0_4x4_q8_0" });
    @export(&repack.q1_0.ggml_gemv_q1_0_4x8_q8_0_generic, .{ .name = "ggml_gemv_q1_0_4x8_q8_0" });
    @export(&repack.q4_0.ggml_gemv_q4_0_4x4_q8_0_generic, .{ .name = "ggml_gemv_q4_0_4x4_q8_0" });
    @export(&repack.q4_0.ggml_gemv_q4_0_4x8_q8_0_generic, .{ .name = "ggml_gemv_q4_0_4x8_q8_0" });
    @export(&repack.q4_k.ggml_gemv_q4_K_8x4_q8_K_generic, .{ .name = "ggml_gemv_q4_K_8x4_q8_K" });
    @export(&repack.q5_k.ggml_gemv_q5_K_8x4_q8_K_generic, .{ .name = "ggml_gemv_q5_K_8x4_q8_K" });
    @export(&repack.q5_k.ggml_gemv_q5_K_8x8_q8_K_generic, .{ .name = "ggml_gemv_q5_K_8x8_q8_K" });
    @export(&repack.q6_k.ggml_gemv_q6_K_8x4_q8_K_generic, .{ .name = "ggml_gemv_q6_K_8x4_q8_K" });
    @export(&repack.q6_k.ggml_gemv_q6_K_8x8_q8_K_generic, .{ .name = "ggml_gemv_q6_K_8x8_q8_K" });
    @export(&repack.nibble4.ggml_gemv_iq4_nl_4x4_q8_0_generic, .{ .name = "ggml_gemv_iq4_nl_4x4_q8_0" });
    @export(&repack.nibble4.ggml_gemv_mxfp4_4x4_q8_0_generic, .{ .name = "ggml_gemv_mxfp4_4x4_q8_0" });
    @export(&repack.q8_0.ggml_gemv_q8_0_4x4_q8_0_generic, .{ .name = "ggml_gemv_q8_0_4x4_q8_0" });
    @export(&repack.q8_0.ggml_gemv_q8_0_4x8_q8_0_generic, .{ .name = "ggml_gemv_q8_0_4x8_q8_0" });

    @export(&repack.q1_0.ggml_gemm_q1_0_4x4_q8_0_generic, .{ .name = "ggml_gemm_q1_0_4x4_q8_0" });
    @export(&repack.q1_0.ggml_gemm_q1_0_4x8_q8_0_generic, .{ .name = "ggml_gemm_q1_0_4x8_q8_0" });
    @export(&repack.q4_0.ggml_gemm_q4_0_4x4_q8_0_generic, .{ .name = "ggml_gemm_q4_0_4x4_q8_0" });
    @export(&repack.q4_0.ggml_gemm_q4_0_4x8_q8_0_generic, .{ .name = "ggml_gemm_q4_0_4x8_q8_0" });
    @export(&repack.q4_k.ggml_gemm_q4_K_8x4_q8_K_generic, .{ .name = "ggml_gemm_q4_K_8x4_q8_K" });
    @export(&repack.q5_k.ggml_gemm_q5_K_8x4_q8_K_generic, .{ .name = "ggml_gemm_q5_K_8x4_q8_K" });
    @export(&repack.q5_k.ggml_gemm_q5_K_8x8_q8_K_generic, .{ .name = "ggml_gemm_q5_K_8x8_q8_K" });
    @export(&repack.q6_k.ggml_gemm_q6_K_8x4_q8_K_generic, .{ .name = "ggml_gemm_q6_K_8x4_q8_K" });
    @export(&repack.q6_k.ggml_gemm_q6_K_8x8_q8_K_generic, .{ .name = "ggml_gemm_q6_K_8x8_q8_K" });
    @export(&repack.nibble4.ggml_gemm_iq4_nl_4x4_q8_0_generic, .{ .name = "ggml_gemm_iq4_nl_4x4_q8_0" });
    @export(&repack.nibble4.ggml_gemm_mxfp4_4x4_q8_0_generic, .{ .name = "ggml_gemm_mxfp4_4x4_q8_0" });
    @export(&repack.q8_0.ggml_gemm_q8_0_4x4_q8_0_generic, .{ .name = "ggml_gemm_q8_0_4x4_q8_0" });
    @export(&repack.q8_0.ggml_gemm_q8_0_4x8_q8_0_generic, .{ .name = "ggml_gemm_q8_0_4x8_q8_0" });
}
