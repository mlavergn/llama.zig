//! Barrel for the ported `ggml-cpu/arch/arm/quants.c`.
//!
//! # Provenance
//!
//! **Not a port.** Scaffolding this project owns. Everything it gathers ports
//! `llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` at v0.4.0 (`5266f24da`).
//!
//! # Only a third of that file compiles here
//!
//! 1,556 of its 4,319 lines survive the preprocessor for the baseline (M1)
//! CPU. The rest is behind `__ARM_FEATURE_SVE`, which no Apple core has, and
//! `__ARM_FEATURE_MATMUL_INT8`, which M2/A15 and later do. The i8mm arms are
//! ported too, in `mmla.zig`, and compile only for a CPU that has it --
//! `-Dcpu=apple_m3`, which the xcframework's macOS slice targets. The SVE
//! arms are not. Each site says which arm it took.
//!
//! - `neon.zig`    — the ACLE intrinsics as Zig vector code
//! - `rows.zig`    — the `q8_0`, `q8_1` and `q8_K` row quantizers
//! - `legacy.zig`  — the block-per-32 dot products, plus mxfp4 and nvfp4
//! - `ternary.zig` — tq1_0 and tq2_0
//! - `k.zig`       — the K-quant super-block dot products
//! - `mmla.zig`    — the two-row i8mm arms of five of the above
//! - `iq.zig`      — the codebook formats
//! - `golden.zig`  — values captured from the C, generated
//!
//! # These kernels are live
//!
//! Unlike `ggml-cpu/quants.c`'s `_generic` twins, every symbol here is on the
//! execution path: `type_traits_cpu` points at them and every quantized
//! `mul_mat` on the CPU goes through one. `test-backend-ops` and token parity
//! do reach them — but with a tolerance and at token granularity, and a
//! last-bit difference survives both. The goldens do not.

const std = @import("std");

comptime {
    _ = @import("rows.zig");
    _ = @import("legacy.zig");
    _ = @import("ternary.zig");
    _ = @import("k.zig");
    _ = @import("mmla.zig");
    _ = @import("iq.zig");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = @import("neon.zig");
    _ = @import("rows.zig");
    _ = @import("legacy.zig");
    _ = @import("ternary.zig");
    _ = @import("k.zig");
    _ = @import("mmla.zig");
    _ = @import("iq.zig");
}
