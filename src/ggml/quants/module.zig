//! Barrel for the ported `ggml-quants.c`.
//!
//! # Provenance
//!
//! The files here together replace `llama.cpp/ggml/src/ggml-quants.c` (v0.3.0,
//! `c1d0e7a00`), 5,667 lines and 79 exported symbols. Each names the section of
//! that file it covers.
//!
//! **This file is not a port**: it is a barrel, and `testing.zig` and
//! `golden.zig` are test scaffolding rather than translations.
//!
//! # How this file is tested
//!
//! `golden.zig` holds checksums captured from the C by `scripts/quants-golden`.
//! Every quantizer is checked against those rather than against a round trip
//! through its own dequantizer -- a round trip passes happily with both halves
//! wrong in the same way, and this project has produced four such false passes
//! already.

const std = @import("std");

pub const blocks = @import("blocks.zig");
pub const helpers = @import("helpers.zig");
pub const k = @import("k.zig");
pub const ternary = @import("ternary.zig");
pub const iq_dequant = @import("iq_dequant.zig");
pub const iq4 = @import("iq4.zig");
pub const chunks = @import("chunks.zig");
pub const grids = @import("grids.zig");
pub const codebook = @import("codebook.zig");
pub const validate = @import("validate.zig");
pub const iq1 = @import("iq1.zig");
pub const iq2 = @import("iq2.zig");
pub const iq3 = @import("iq3.zig");
pub const legacy = @import("legacy.zig");
pub const golden = @import("golden.zig");
pub const testing = @import("testing.zig");

comptime {
    // Ported translation units export C symbols nothing in Zig references, so
    // they need forcing into the compilation.
    _ = legacy;
    _ = k;
    _ = ternary;
    _ = iq_dequant;
    _ = iq4;
    _ = chunks;
    _ = codebook;
    _ = validate;
    _ = iq1;
    _ = iq2;
    _ = iq3;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = @import("blocks.zig");
    _ = @import("helpers.zig");
    _ = @import("k.zig");
    _ = @import("ternary.zig");
    _ = @import("iq_dequant.zig");
    _ = @import("iq4.zig");
    _ = @import("chunks.zig");
    _ = @import("codebook.zig");
    _ = @import("validate.zig");
    _ = @import("iq1.zig");
    _ = @import("iq_test.zig");
    _ = @import("iq2.zig");
    _ = @import("iq3.zig");
    _ = @import("legacy.zig");
}
