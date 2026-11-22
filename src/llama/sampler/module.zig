//! Barrel for the ported sampler.
//!
//! # Provenance
//!
//! **Not a port.** It gathers the files that together replace
//! `llama.cpp/src/llama-sampler.cpp` (v1.0.0, `b1b770d90`):
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

pub const core = @import("core.zig");
pub const chain = @import("chain.zig");
pub const basic = @import("basic.zig");
pub const mirostat = @import("mirostat.zig");
pub const penalties = @import("penalties.zig");
pub const logit_bias = @import("logit_bias.zig");
pub const infill = @import("infill.zig");
pub const grammar = @import("grammar.zig");

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
