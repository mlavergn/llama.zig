//! Barrel for the x86 quant kernels.
//!
//! # Provenance
//!
//! **Not a port by itself.** It gathers the port of
//! `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` at v0.3.0 (`c1d0e7a00`),
//! the x86 counterpart of `quants/arm/`, and says which of that file's 27
//! exports are ported. `arch-fallback.h` adds one more: on x86 the real
//! `ggml_vec_dot_q2_0_q8_0` is `quants.c`'s generic body, which is already
//! ported in `quants/legacy.zig` and is exported under the real name here.
//!
//! # Not ported yet
//!
//! `PLAN.md`, Stage 5 step 6. Every kernel below is pending: by default its
//! name is a compile error, so an x86 build cannot ship one by accident.
//! With `-Dx86-incomplete` -- development only -- each exports its generic
//! port under the real name instead. The generic kernels do not compute what
//! the AVX2 ones do, so each stand-in fails `ops-diff`; that is how the
//! gates, run under QEMU, list what is left.

const std = @import("std");
const config = @import("config");

const legacy = @import("../legacy.zig");
const k = @import("../k.zig");
const iq = @import("../iq.zig");
const ternary = @import("../ternary.zig");
const rows = @import("../rows.zig");

/// The kernels `arch/x86/quants.c` exports, each with the generic port that
/// stands in for it under `-Dx86-incomplete`. A kernel leaves this list when
/// its x86 port is exported from this directory in its place.
const pending = .{
    .{ "ggml_vec_dot_q1_0_q8_0", &legacy.ggml_vec_dot_q1_0_q8_0_generic },
    .{ "ggml_vec_dot_q4_0_q8_0", &legacy.ggml_vec_dot_q4_0_q8_0_generic },
    .{ "ggml_vec_dot_q4_1_q8_1", &legacy.ggml_vec_dot_q4_1_q8_1_generic },
    .{ "ggml_vec_dot_q5_0_q8_0", &legacy.ggml_vec_dot_q5_0_q8_0_generic },
    .{ "ggml_vec_dot_q5_1_q8_1", &legacy.ggml_vec_dot_q5_1_q8_1_generic },
    .{ "ggml_vec_dot_q8_0_q8_0", &legacy.ggml_vec_dot_q8_0_q8_0_generic },
    .{ "ggml_vec_dot_mxfp4_q8_0", &legacy.ggml_vec_dot_mxfp4_q8_0_generic },
    .{ "ggml_vec_dot_nvfp4_q8_0", &legacy.ggml_vec_dot_nvfp4_q8_0_generic },
    .{ "ggml_vec_dot_iq4_nl_q8_0", &legacy.ggml_vec_dot_iq4_nl_q8_0_generic },
    .{ "ggml_vec_dot_q2_K_q8_K", &k.ggml_vec_dot_q2_K_q8_K_generic },
    .{ "ggml_vec_dot_q3_K_q8_K", &k.ggml_vec_dot_q3_K_q8_K_generic },
    .{ "ggml_vec_dot_q4_K_q8_K", &k.ggml_vec_dot_q4_K_q8_K_generic },
    .{ "ggml_vec_dot_q5_K_q8_K", &k.ggml_vec_dot_q5_K_q8_K_generic },
    .{ "ggml_vec_dot_q6_K_q8_K", &k.ggml_vec_dot_q6_K_q8_K_generic },
    .{ "ggml_vec_dot_tq1_0_q8_K", &ternary.ggml_vec_dot_tq1_0_q8_K_generic },
    .{ "ggml_vec_dot_tq2_0_q8_K", &ternary.ggml_vec_dot_tq2_0_q8_K_generic },
    .{ "ggml_vec_dot_iq1_s_q8_K", &iq.ggml_vec_dot_iq1_s_q8_K_generic },
    .{ "ggml_vec_dot_iq1_m_q8_K", &iq.ggml_vec_dot_iq1_m_q8_K_generic },
    .{ "ggml_vec_dot_iq2_xxs_q8_K", &iq.ggml_vec_dot_iq2_xxs_q8_K_generic },
    .{ "ggml_vec_dot_iq2_xs_q8_K", &iq.ggml_vec_dot_iq2_xs_q8_K_generic },
    .{ "ggml_vec_dot_iq2_s_q8_K", &iq.ggml_vec_dot_iq2_s_q8_K_generic },
    .{ "ggml_vec_dot_iq3_xxs_q8_K", &iq.ggml_vec_dot_iq3_xxs_q8_K_generic },
    .{ "ggml_vec_dot_iq3_s_q8_K", &iq.ggml_vec_dot_iq3_s_q8_K_generic },
    .{ "ggml_vec_dot_iq4_xs_q8_K", &iq.ggml_vec_dot_iq4_xs_q8_K_generic },
    .{ "quantize_row_q8_0", &rows.quantize_row_q8_0_generic },
    .{ "quantize_row_q8_1", &rows.quantize_row_q8_1_generic },
    .{ "quantize_row_q8_K", &rows.quantize_row_q8_K_generic },
};

/// How many of `arch/x86/quants.c`'s 27 exports are still pending.
pub const pending_count = pending.len;

comptime {
    // arch-fallback.h:86: x86 takes the generic body under the real name.
    @export(&legacy.ggml_vec_dot_q2_0_q8_0_generic, .{ .name = "ggml_vec_dot_q2_0_q8_0" });

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
