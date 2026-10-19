//! Barrel for the x86 quant kernels.
//!
//! # Provenance
//!
//! **Not a port by itself.** It gathers the port of
//! `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` at v0.3.0 (`c1d0e7a00`),
//! the x86 counterpart of `quants/arm/`: that file's 27 exports, at the
//! `__AVX2__` arms x86-64-v3 compiles. `arch-fallback.h` adds one more: on x86 the real
//! `ggml_vec_dot_q2_0_q8_0` is `quants.c`'s generic body. `legacy.zig` ports
//! it again as x86 compiles it, fused, rather than exporting the shared
//! generic port, whose goldens are unfused.
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
//! `scripts/x86-quants-diff`: every one of the 27 exports against the
//! reference C, in one process, on bits, run on the Windows laptop's
//! Haswell. All 27 are ported.

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
