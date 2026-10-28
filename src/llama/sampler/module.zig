//! Barrel for the ported sampler.
//!
//! # Provenance
//!
//! **Not a port.** It gathers the files that together replace
//! `llama.cpp/src/llama-sampler.cpp` (v0.4.0, `5266f24da`):
//!
//! - `core.zig`       — the dispatchers, the shared sort and softmax, the
//!                      empty placeholder, the backend-sampling base
//! - `chain.zig`      — the chain, `llama_sampler_sample`, copy, seed, perf
//! - `basic.zig`      — greedy, dist, top-k/p, min-p, typical, temp,
//!                      temp-ext, XTC, top-n-sigma
//! - `mirostat.zig`   — both mirostats and adaptive-p
//! - `penalties.zig`  — penalties and DRY
//! - `logit_bias.zig` — logit-bias
//! - `infill.zig`     — infill
//! - `grammar.zig`    — the grammar sampler, over `../grammar/`

const std = @import("std");

comptime {
    // Every file here exports C symbols nothing in Zig references, so each
    // is forced into the compilation.
    _ = @import("core.zig");
    _ = @import("chain.zig");
    _ = @import("basic.zig");
    _ = @import("mirostat.zig");
    _ = @import("penalties.zig");
    _ = @import("logit_bias.zig");
    _ = @import("infill.zig");
    _ = @import("grammar.zig");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
