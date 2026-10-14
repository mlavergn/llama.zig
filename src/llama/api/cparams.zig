//! `llama.h`'s one entry point in `llama-cparams.cpp`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-cparams.cpp` (v0.3.0, `c1d0e7a00`):
//! `llama_max_parallel_sequences`, its one unmangled export.
//!
//! The function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call it beside the C++ that defines the same name,
//! and `exports.zig` exports it once libllama swaps.

const std = @import("std");
const cparams = @import("../cparams.zig");

/// Ports `llama_max_parallel_sequences` (llama-cparams.cpp:3 @c1d0e7a00).
pub fn llama_max_parallel_sequences() callconv(.c) usize {
    return cparams.maxParallelSequences();
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
