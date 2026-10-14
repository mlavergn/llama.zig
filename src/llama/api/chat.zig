//! `llama.h`'s one entry point in `llama-chat.cpp`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-chat.cpp` (v0.3.0, `c1d0e7a00`):
//! `llama_chat_builtin_templates`, its one unmangled export. The template
//! table is `chat.zig`'s. `llama_chat_apply_template` is defined in
//! `llama.cpp/src/llama.cpp`, so it is in `api/llama.zig`.
//!
//! The function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call it beside the C++ that defines the same name,
//! and `exports.zig` exports it once libllama swaps.

const std = @import("std");
const chat = @import("../chat.zig");

/// Ports `llama_chat_builtin_templates` (llama-chat.cpp:950 @c1d0e7a00).
pub fn llama_chat_builtin_templates(output: [*c][*c]const u8, len: usize) callconv(.c) i32 {
    return chat.builtinTemplates(@ptrCast(output), len);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
