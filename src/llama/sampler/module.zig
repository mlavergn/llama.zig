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

const llama = @import("../module.zig");

/// The sampler as libllama's other files call it: by symbol, at link time.
/// In libllama that is the port's (`chain.zig`, `core.zig`); a harness that
/// compiles those files against upstream's `libllama-ref.a` reaches
/// upstream's sampler instead. Referencing these compiles none of the
/// sampler, so none of its exports reach a harness.
pub const linked = struct {
    pub const chain_get = @extern(*const @TypeOf(llama.impl.c.llama_sampler_chain_get), .{ .name = "llama_sampler_chain_get" });
    pub const chain_n = @extern(*const @TypeOf(llama.impl.c.llama_sampler_chain_n), .{ .name = "llama_sampler_chain_n" });
    pub const name = @extern(*const @TypeOf(llama.impl.c.llama_sampler_name), .{ .name = "llama_sampler_name" });
    /// C++ linkage upstream, so not in `llama.h`: by mangled name.
    pub const backend_begin = @extern(*const fn (*llama.impl.c.llama_sampler) callconv(.c) void, .{ .name = "_Z27llama_sampler_backend_beginP13llama_sampler" });
    pub const backend_n_nodes = @extern(*const fn (*const llama.impl.c.llama_sampler) callconv(.c) u32, .{ .name = "_Z29llama_sampler_backend_n_nodesPK13llama_sampler" });
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
