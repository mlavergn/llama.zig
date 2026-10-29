//! `llama.h`'s one entry point in `llama-model-loader.cpp`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-model-loader.cpp` (v0.4.1,
//! `b29c606e2`): `llama_ftype_name`, its one unmangled export. The name
//! table is `loader.zig`'s.
//!
//! The function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call it beside the C++ that defines the same name,
//! and `exports.zig` exports it once libllama swaps.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");

const c = impl.c;

/// Ports `llama_ftype_name` (llama-model-loader.cpp:33 @b29c606e2).
pub fn llama_ftype_name(ftype: c.enum_llama_ftype) callconv(.c) [*c]const u8 {
    return loader.ftypeName(ftype).ptr;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
