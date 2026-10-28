//! `llama.h`'s one entry point in `llama-cparams.cpp`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-cparams.cpp` (v0.4.0, `5266f24da`):
//! `llama_max_parallel_sequences`, its one unmangled export.
//!
//! The function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call it beside the C++ that defines the same name,
//! and `exports.zig` exports it once libllama swaps.

const std = @import("std");
const cparams = @import("../cparams.zig");

/// Ports `llama_max_parallel_sequences` (llama-cparams.cpp:3 @5266f24da).
pub fn llama_max_parallel_sequences() callconv(.c) usize {
    return cparams.maxParallelSequences();
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
