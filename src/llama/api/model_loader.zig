//! `llama.h`'s one entry point in `llama-model-loader.cpp`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-model-loader.cpp` (v1.0.0,
//! `b1b770d90`): `llama_ftype_name`, its one unmangled export. The name
//! table is `loader.zig`'s.
//!
//! The function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call it beside the C++ that defines the same name,
//! and `exports.zig` exports it once libllama swaps.

const std = @import("std");
const llama = @import("../module.zig");

/// Ports `llama_ftype_name` (llama-model-loader.cpp:34 @b1b770d90).
pub fn llama_ftype_name(ftype: llama.impl.c.enum_llama_ftype) callconv(.c) [*c]const u8 {
    return llama.loader.ftypeName(ftype).ptr;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
