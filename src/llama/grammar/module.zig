//! Barrel for the ported grammar: `llama-grammar.cpp`'s parser and engine.
//!
//! # Provenance
//!
//! **Not a port.** It gathers the files that together replace
//! `llama.cpp/src/llama-grammar.cpp` (v1.0.0, `b1b770d90`).

const std = @import("std");

pub const parser = @import("parser.zig");
pub const grammar = @import("grammar.zig");

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
