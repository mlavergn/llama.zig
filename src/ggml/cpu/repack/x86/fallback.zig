//! x86's `arch-fallback.h` renames for `repack.cpp`: the 24 kernels
//! `arch/x86/repack.cpp` has no version of, so the generic body is the real
//! one.
//!
//! # Provenance
//!
//! **Not a port of a function.** It reproduces
//! `llama.cpp/ggml/src/ggml-cpu/arch-fallback.h` at v0.3.0 (`c1d0e7a00`),
//! lines 88-111, the `__x86_64__` block, which `#define`s each generic to
//! the bare name before `repack.cpp` compiles. The bodies are the generic
//! ports in `cpu/repack/`; this exports each under the name the C gives it.
//! `arm/fallback.zig` does the same for ARM's eight.

const nibble4 = @import("../nibble4.zig");
const q4_0 = @import("../q4_0.zig");
const q4_k = @import("../q4_k.zig");
const q5_k = @import("../q5_k.zig");
const q6_k = @import("../q6_k.zig");
const q8_0 = @import("../q8_0.zig");
const quantize = @import("../quantize.zig");

comptime {
    @export(&quantize.ggml_quantize_mat_q8_0_4x4_generic, .{ .name = "ggml_quantize_mat_q8_0_4x4" });
    @export(&quantize.ggml_quantize_mat_q8_K_4x4_generic, .{ .name = "ggml_quantize_mat_q8_K_4x4" });

    @export(&q4_0.ggml_gemv_q4_0_4x4_q8_0_generic, .{ .name = "ggml_gemv_q4_0_4x4_q8_0" });
    @export(&q4_0.ggml_gemv_q4_0_4x8_q8_0_generic, .{ .name = "ggml_gemv_q4_0_4x8_q8_0" });
    @export(&q4_k.ggml_gemv_q4_K_8x4_q8_K_generic, .{ .name = "ggml_gemv_q4_K_8x4_q8_K" });
    @export(&q5_k.ggml_gemv_q5_K_8x4_q8_K_generic, .{ .name = "ggml_gemv_q5_K_8x4_q8_K" });
    @export(&q5_k.ggml_gemv_q5_K_8x8_q8_K_generic, .{ .name = "ggml_gemv_q5_K_8x8_q8_K" });
    @export(&q6_k.ggml_gemv_q6_K_8x4_q8_K_generic, .{ .name = "ggml_gemv_q6_K_8x4_q8_K" });
    @export(&q6_k.ggml_gemv_q6_K_8x8_q8_K_generic, .{ .name = "ggml_gemv_q6_K_8x8_q8_K" });
    @export(&nibble4.ggml_gemv_iq4_nl_4x4_q8_0_generic, .{ .name = "ggml_gemv_iq4_nl_4x4_q8_0" });
    @export(&nibble4.ggml_gemv_mxfp4_4x4_q8_0_generic, .{ .name = "ggml_gemv_mxfp4_4x4_q8_0" });
    @export(&q8_0.ggml_gemv_q8_0_4x4_q8_0_generic, .{ .name = "ggml_gemv_q8_0_4x4_q8_0" });
    @export(&q8_0.ggml_gemv_q8_0_4x8_q8_0_generic, .{ .name = "ggml_gemv_q8_0_4x8_q8_0" });

    @export(&q4_0.ggml_gemm_q4_0_4x4_q8_0_generic, .{ .name = "ggml_gemm_q4_0_4x4_q8_0" });
    @export(&q4_0.ggml_gemm_q4_0_4x8_q8_0_generic, .{ .name = "ggml_gemm_q4_0_4x8_q8_0" });
    @export(&q4_k.ggml_gemm_q4_K_8x4_q8_K_generic, .{ .name = "ggml_gemm_q4_K_8x4_q8_K" });
    @export(&q5_k.ggml_gemm_q5_K_8x4_q8_K_generic, .{ .name = "ggml_gemm_q5_K_8x4_q8_K" });
    @export(&q5_k.ggml_gemm_q5_K_8x8_q8_K_generic, .{ .name = "ggml_gemm_q5_K_8x8_q8_K" });
    @export(&q6_k.ggml_gemm_q6_K_8x4_q8_K_generic, .{ .name = "ggml_gemm_q6_K_8x4_q8_K" });
    @export(&q6_k.ggml_gemm_q6_K_8x8_q8_K_generic, .{ .name = "ggml_gemm_q6_K_8x8_q8_K" });
    @export(&nibble4.ggml_gemm_iq4_nl_4x4_q8_0_generic, .{ .name = "ggml_gemm_iq4_nl_4x4_q8_0" });
    @export(&nibble4.ggml_gemm_mxfp4_4x4_q8_0_generic, .{ .name = "ggml_gemm_mxfp4_4x4_q8_0" });
    @export(&q8_0.ggml_gemm_q8_0_4x4_q8_0_generic, .{ .name = "ggml_gemm_q8_0_4x4_q8_0" });
    @export(&q8_0.ggml_gemm_q8_0_4x8_q8_0_generic, .{ .name = "ggml_gemm_q8_0_4x8_q8_0" });
}
