//! Barrel for the ported `ggml-cpu/quants.c`.
//!
//! # Provenance
//!
//! **Not a port.** Scaffolding this project owns, matching the barrels above
//! it. Everything it gathers ports
//! `llama.cpp/ggml/src/ggml-cpu/quants.c` at v0.3.0 (`c1d0e7a00`), split by
//! format family:
//!
//! - `rows.zig`    — the 20 row quantizers `type_traits_cpu` points at
//! - `legacy.zig`  — the block-per-32 dot products, plus mxfp4 and nvfp4
//! - `k.zig`       — the K-quant super-block dot products
//! - `ternary.zig` — tq1_0 and tq2_0
//! - `iq.zig`      — the codebook formats
//! - `golden.zig`  — values captured from the C, generated
//! - `testing.zig` — the fixture the dot products are checked through
//! - `arm/`        — a second translation unit,
//!                   `ggml-cpu/arch/arm/quants.c`, which supplies the *real*
//!                   entry points these `_generic` ones stand behind
//!
//! # The dot products here are unreachable on this target
//!
//! `arch-fallback.h` renames nothing from `quants.c` on ARM, and
//! `arch/arm/quants.c` supplies every real entry point, so the 25 `_generic`
//! names are exported and never called. `golden.zig` is their only gate. The
//! 20 row quantizers are live: 17 are what the traits table calls, and three
//! are `_generic` fallbacks the NEON file replaces.

const std = @import("std");
const builtin = @import("builtin");

comptime {
    _ = @import("rows.zig");
    _ = @import("legacy.zig");
    _ = @import("k.zig");
    _ = @import("ternary.zig");
    _ = @import("iq.zig");
    // The architecture's own kernels, as the C compiles one `arch/` file:
    // ggml-cpu/arch/arm/quants.c across quants/arm/, arch/x86/quants.c
    // across quants/x86/.
    switch (builtin.cpu.arch) {
        .aarch64 => _ = @import("arm/module.zig"),
        .x86_64 => _ = @import("x86/module.zig"),
        else => @compileError("no quant kernels for " ++ @tagName(builtin.cpu.arch)),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = @import("rows.zig");
    _ = @import("legacy.zig");
    _ = @import("k.zig");
    _ = @import("ternary.zig");
    _ = @import("iq.zig");
    switch (builtin.cpu.arch) {
        .aarch64 => _ = @import("arm/module.zig"),
        .x86_64 => _ = @import("x86/module.zig"),
        else => {},
    }
    _ = @import("testing.zig");
}
