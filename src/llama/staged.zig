//! The root the in-process harnesses compile.
//!
//! **Not a port.** libllama swaps once, at `llama.h` (`PLAN.md`, Decision
//! 42), so the files ported ahead of that swap are reached only by their
//! diff harnesses, which link them beside the C++ they replace. Those
//! harnesses cannot root at `module.zig`: it carries the sampler's exports,
//! which `libllama.a` already defines. This root re-exports the staged files
//! and nothing that exports a symbol.

const std = @import("std");

pub const impl = @import("impl.zig");
pub const libcxx = @import("libcxx.zig");
pub const regex = @import("regex.zig");
pub const unicode = @import("unicode.zig");
pub const gguf_meta = @import("gguf_meta.zig");
pub const vocab = @import("vocab/module.zig");

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
