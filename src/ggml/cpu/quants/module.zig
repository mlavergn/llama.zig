//! Barrel for the ported `ggml-cpu/quants.c`.
//!
//! # Provenance
//!
//! **Not a port.** Scaffolding this project owns, matching the barrels above
//! it. Everything it gathers ports
//! `llama.cpp/ggml/src/ggml-cpu/quants.c` at v1.0.0 (`b1b770d90`), split by
//! format family:
//!
//! - `rows.zig`    — the 21 row quantizers `type_traits_cpu` points at
//! - `legacy.zig`  — the block-per-32 dot products, plus mxfp4, nvfp4 and
//!                   the live f8_e4m3 one
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
//! All but `ggml_vec_dot_f8_e4m3_f32` in `legacy.zig`, which is live.
//!
//! `arch-fallback.h` renames nothing from `quants.c` on ARM, and
//! `arch/arm/quants.c` supplies every real entry point, so the 25 `_generic`
//! names are exported and never called. `golden.zig` is their only gate. The
//! 21 row quantizers are live: 18 are what the traits table calls, and three
//! are `_generic` fallbacks the NEON file replaces.

const std = @import("std");
const builtin = @import("builtin");

pub const rows = @import("rows.zig");
pub const legacy = @import("legacy.zig");
pub const k = @import("k.zig");
pub const ternary = @import("ternary.zig");
pub const iq = @import("iq.zig");
pub const testing = @import("testing.zig");

/// The architecture's kernels, as `arch/arm/quants.c` or `arch/x86/quants.c`
/// supplies them.
pub const arch = switch (@import("builtin").cpu.arch) {
    .aarch64 => @import("arm/module.zig"),
    .x86_64 => @import("x86/module.zig"),
    else => @compileError("no quants kernels for " ++ @tagName(@import("builtin").cpu.arch)),
};

pub const golden = @import("golden.zig");

comptime {
    _ = rows;
    _ = legacy;
    _ = k;
    _ = ternary;
    _ = iq;
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
    switch (builtin.cpu.arch) {
        .aarch64 => _ = @import("arm/module.zig"),
        .x86_64 => _ = @import("x86/module.zig"),
        else => {},
    }
    _ = testing;
}
