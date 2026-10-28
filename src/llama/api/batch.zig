//! `llama.h`'s batch entry points.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-batch.cpp` (v0.4.0, `5266f24da`):
//! `llama_batch_get_one`, `llama_batch_init` and `llama_batch_free`, its
//! three unmangled exports. The bodies are `batch.zig`'s.
//!
//! Each function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call these beside the C++ that defines the same
//! names, and `exports.zig` exports them once libllama swaps.

const std = @import("std");
const impl = @import("../impl.zig");
const batch = @import("../batch.zig");

const c = impl.c;

/// Ports `llama_batch_get_one` (llama-batch.cpp:931 @5266f24da).
pub fn llama_batch_get_one(tokens: [*c]c.llama_token, n_tokens: i32) callconv(.c) c.struct_llama_batch {
    return batch.batchGetOne(tokens, n_tokens);
}

/// Ports `llama_batch_init` (llama-batch.cpp:945 @5266f24da).
pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) callconv(.c) c.struct_llama_batch {
    return batch.batchInit(n_tokens, embd, n_seq_max);
}

/// Ports `llama_batch_free` (llama-batch.cpp:975 @5266f24da).
pub fn llama_batch_free(b: c.struct_llama_batch) callconv(.c) void {
    batch.batchFree(b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
