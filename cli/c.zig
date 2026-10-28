//! libllama's C ABI, imported once.
//!
//! # Provenance
//!
//! **Not a port.** This is the translation of
//! `llama.cpp/include/llama.h` (v0.4.0, `5266f24da`), and nothing else:
//! `c_import.h`, which `build.zig` translates as the `llama_h` module, since
//! Zig 0.17 has no `@cImport`.
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
pub const api = @import("llama_h");
