//! libllama's C ABI, imported once.
//!
//! # Provenance
//!
//! **Not a port.** This is the `@cImport` of
//! `llama.cpp/include/llama.h` (v0.3.0, `c1d0e7a00`), and nothing else.
//!
//! # Why it is its own file
//!
//! Each `@cImport` produces its own set of types: two of them in one binary
//! give two incompatible `*llama_model`, and the error says only that
//! `*cimport.struct_llama_model` cannot coerce to `*cimport.struct_llama_model`.
//! `session.zig` and `chat.zig` both need the header, so it is imported here
//! and shared.

/// The header's declarations. `usingnamespace` is gone in Zig 0.16, so this
/// is a named constant that callers reach through rather than a flattened
/// namespace.
pub const api = @cImport({
    @cInclude("llama.h");
});
