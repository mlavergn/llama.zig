//! Barrel for the ported vocabulary.
//!
//! **Not a port.** Each file it re-exports names its source; together they
//! are `llama.cpp/src/llama-vocab.cpp` at v1.0.0 (`b1b770d90`): the
//! vocabulary itself in `vocab.zig`, one file per tokenizer, and the trie two
//! of them share.

const std = @import("std");
const llama = @import("../module.zig");

pub const vocab = @import("vocab.zig");
pub const trie = @import("trie.zig");
pub const spm = @import("spm.zig");
pub const bpe = @import("bpe.zig");
pub const wpm = @import("wpm.zig");
pub const ugm = @import("ugm.zig");
pub const rwkv = @import("rwkv.zig");
pub const plamo2 = @import("plamo2.zig");

pub const Vocab = vocab.Vocab;
pub const Token = llama.impl.c.llama_token;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
