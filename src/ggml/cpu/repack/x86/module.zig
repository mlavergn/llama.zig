//! Barrel for the x86 repack kernels.
//!
//! # Provenance
//!
//! **Not a port by itself.** It gathers the port of
//! `llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` at v0.3.0
//! (`c1d0e7a00`), the x86 counterpart of `repack/arm/`, and x86's
//! `arch-fallback.h` renames (`fallback.zig`): the file's 12 exports at the
//! `__AVX2__` arms x86-64-v3 compiles, and the gemms' `__AVX512BW__ &&
//! __AVX512DQ__` arms x86-64-v4 adds.
//!
//! - `common.zig`   — the file's static helpers and the F16C loads
//! - `wide.zig`     — the gemms' intrinsics at 256 and 512 bits
//! - `quantize.zig` — `ggml_quantize_mat_q8_0_4x8` and `_q8_K_4x8`
//! - `q4_b32.zig`   — the `q4_0`, `iq4_nl` and `mxfp4` gemv and gemm
//! - `q4_k.zig`     — the `q4_K` gemv and gemm
//! - `q2_k.zig`     — the `q2_K` gemv and gemm
//! - `fallback.zig` — the 24 names x86 takes from the generic bodies
//!
//! # The oracle
//!
//! `scripts/x86-repack-diff`: `harness/repack_diff.zig` against the
//! reference C++, on bits, on the Windows laptop at x86-64-v3 and on an
//! AVX-512 machine at v4.

const std = @import("std");

pub const fallback = @import("fallback.zig");

comptime {
    _ = fallback;
    _ = @import("quantize.zig");
    _ = @import("q4_b32.zig");
    _ = @import("q4_k.zig");
    _ = @import("q2_k.zig");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = @import("common.zig");
    _ = @import("wide.zig");
    _ = @import("quantize.zig");
    _ = @import("q4_b32.zig");
    _ = @import("q4_k.zig");
    _ = @import("q2_k.zig");
}
