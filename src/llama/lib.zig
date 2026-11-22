//! libllama's C API: every file that exports a `llama.h` symbol, forced into
//! the compilation.
//!
//! # Provenance
//!
//! **Not a port.** Linking this is what defines the `llama_*` symbols in a
//! binary. `module.zig` forces it when libllama itself is the compilation
//! root, and under test; an executable that compiles the `llama` module in
//! and calls the C API forces it too (`_ = llama.lib`). A harness never does:
//! it links upstream's libllama beside the port, and two definitions of a
//! symbol pick one silently.

const std = @import("std");
const mod = @import("module.zig");

comptime {
    // Every llama.h entry point but the sampler's, from api/.
    _ = @import("exports.zig");
    // llama-sampler.cpp, split across src/llama/sampler/; it reaches
    // llama-grammar.cpp (src/llama/grammar/) and std::regex (regex.zig).
    _ = mod.sampler.core;
    _ = mod.sampler.chain;
    _ = mod.sampler.basic;
    _ = mod.sampler.mirostat;
    _ = mod.sampler.penalties;
    _ = mod.sampler.logit_bias;
    _ = mod.sampler.infill;
    _ = mod.sampler.grammar;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
