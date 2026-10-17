//! Barrel for the x86 repack kernels.
//!
//! # Provenance
//!
//! **Not a port by itself.** It gathers the port of
//! `llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` at v0.3.0
//! (`c1d0e7a00`), the x86 counterpart of `repack/arm/`, and x86's
//! `arch-fallback.h` renames (`fallback.zig`). It says which of the file's
//! 12 exports are ported.
//!
//! # Not ported yet
//!
//! `PLAN.md`, Stage 5 step 6. Every kernel below is pending, exactly as in
//! `quants/x86/module.zig`: a compile error by default, its generic port
//! under the real name with `-Dx86-incomplete`, for development under QEMU.

const std = @import("std");
const config = @import("config");

const nibble4 = @import("../nibble4.zig");
const q2_k = @import("../q2_k.zig");
const q4_0 = @import("../q4_0.zig");
const q4_k = @import("../q4_k.zig");
const quantize = @import("../quantize.zig");

pub const fallback = @import("fallback.zig");

/// The kernels `arch/x86/repack.cpp` exports at x86-64-v3, each with the
/// generic port that stands in for it under `-Dx86-incomplete`.
const pending = .{
    .{ "ggml_quantize_mat_q8_0_4x8", &quantize.ggml_quantize_mat_q8_0_4x8_generic },
    .{ "ggml_quantize_mat_q8_K_4x8", &quantize.ggml_quantize_mat_q8_K_4x8_generic },
    .{ "ggml_gemv_q4_0_8x8_q8_0", &q4_0.ggml_gemv_q4_0_8x8_q8_0_generic },
    .{ "ggml_gemm_q4_0_8x8_q8_0", &q4_0.ggml_gemm_q4_0_8x8_q8_0_generic },
    .{ "ggml_gemv_q4_K_8x8_q8_K", &q4_k.ggml_gemv_q4_K_8x8_q8_K_generic },
    .{ "ggml_gemm_q4_K_8x8_q8_K", &q4_k.ggml_gemm_q4_K_8x8_q8_K_generic },
    .{ "ggml_gemv_q2_K_8x8_q8_K", &q2_k.ggml_gemv_q2_K_8x8_q8_K_generic },
    .{ "ggml_gemm_q2_K_8x8_q8_K", &q2_k.ggml_gemm_q2_K_8x8_q8_K_generic },
    .{ "ggml_gemv_iq4_nl_8x8_q8_0", &nibble4.ggml_gemv_iq4_nl_8x8_q8_0_generic },
    .{ "ggml_gemm_iq4_nl_8x8_q8_0", &nibble4.ggml_gemm_iq4_nl_8x8_q8_0_generic },
    .{ "ggml_gemv_mxfp4_8x8_q8_0", &nibble4.ggml_gemv_mxfp4_8x8_q8_0_generic },
    .{ "ggml_gemm_mxfp4_8x8_q8_0", &nibble4.ggml_gemm_mxfp4_8x8_q8_0_generic },
};

/// How many of `arch/x86/repack.cpp`'s 12 exports are still pending.
pub const pending_count = pending.len;

comptime {
    _ = fallback;
    for (pending) |p| {
        if (!config.x86_incomplete) {
            @compileError(p[0] ++ ": the x86 kernel is not ported yet (PLAN.md, Stage 5 step 6); -Dx86-incomplete stands the generic port in, for development");
        }
        @export(p[1], .{ .name = p[0] });
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
