//! Barrel for the `arch/arm/repack.cpp` port.
//!
//! **Not a port.** It re-exports the pieces so siblings reach each other
//! through one import, per the project's barrel convention.

pub const fallback = @import("fallback.zig");
pub const kscales = @import("kscales.zig");
pub const nibble4 = @import("nibble4.zig");
pub const q4_0 = @import("q4_0.zig");
pub const q4_0_gemm = @import("q4_0_gemm.zig");
pub const q4_k = @import("q4_k.zig");
pub const q5_k = @import("q5_k.zig");
pub const q6_k = @import("q6_k.zig");
pub const q8_0 = @import("q8_0.zig");
pub const quantize = @import("quantize.zig");

comptime {
    _ = fallback;
    _ = kscales;
    _ = nibble4;
    _ = q4_0;
    _ = q4_0_gemm;
    _ = q4_k;
    _ = q5_k;
    _ = q6_k;
    _ = q8_0;
    _ = quantize;
}
