//! Barrel for the x86 quant kernels.
//!
//! # Provenance
//!
//! **Not a port by itself.** It gathers the port of
//! `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` at v0.8.0 (`8642541b2`),
//! the x86 counterpart of `quants/arm/`: that file's 28 exports, at the
//! `__AVX2__` arms x86-64-v3 compiles. Until v0.7.0 the 28th,
//! `ggml_vec_dot_q2_0_q8_0`, was `quants.c`'s generic body renamed by
//! `arch-fallback.h`, ported again as x86 compiled it, fused; since #30447 it
//! is an AVX2 kernel of its own and the generic is dead here as on ARM.
//!
//! - `avx2.zig`   — the SSE/AVX/AVX2 intrinsics, as Zig
//! - `common.zig` — the file's static helpers
//! - `rows.zig`   — the row quantizers
//! - `legacy.zig` — the block-per-32 dot products, `q1_0` and the fp4 codebooks
//! - `k.zig`      — the K-quant dot products
//! - `ternary.zig` — tq1_0 and tq2_0
//! - `iq.zig`     — the codebook formats
//!
//! # The oracle
//!
//! `scripts/x86-quants-diff`: every one of the 28 exports against the
//! reference C, in one process, on bits, run on the Windows laptop's
//! Haswell. All 28 are ported.

const std = @import("std");

comptime {
    _ = @import("rows.zig");
    _ = @import("legacy.zig");
    _ = @import("k.zig");
    _ = @import("ternary.zig");
    _ = @import("iq.zig");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = @import("avx2.zig");
    _ = @import("common.zig");
    _ = @import("rows.zig");
    _ = @import("legacy.zig");
    _ = @import("k.zig");
    _ = @import("ternary.zig");
    _ = @import("iq.zig");
}
