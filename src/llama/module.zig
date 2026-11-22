//! Barrel for the ported libllama.
//!
//! # Provenance
//!
//! **Not a port.** Everything under `src/llama/` is a port of C++ in the
//! pinned reference checkout, v1.0.0 (`b1b770d90`), or of the libc++ it was
//! compiled against; each file states its sources.
//!
//! # The swap
//!
//! libllama is one strongly connected component of C++ translation units,
//! so it swapped once, at `llama.h`. The sampler
//! and grammar went first, their contract with the rest being C; every
//! other translation unit followed together. `build/llamacpp.zig` roots
//! `libllama.a` here and compiles no C or C++ into it: `exports.zig` exports
//! every function of `llama.h`, and the sampler exports its own.
//!
//! This file is the barrel and exports nothing by itself: the C API is
//! `lib.zig`, forced here only when libllama is the compilation root. A file
//! anywhere in the port can therefore reach this barrel, and the in-process
//! harnesses, which compile the port beside upstream's C++ (`libllama-ref.a`),
//! never pull an export in.

const std = @import("std");

/// What the CLI reaches as Zig: `cli/fit.zig` ports `common/fit.cpp`,
/// which calls libllama through its internal `llama-ext.h` as well as
/// `llama.h`. Declaring these exports nothing.
pub const api = @import("api/module.zig");
pub const impl = @import("impl.zig");
pub const context = @import("context.zig");
/// `std::regex` as libc++ implements it, for `cli/json/schema_to_grammar.zig`
/// and the chat parsers over it.
pub const regex = @import("regex.zig");

pub const sampler = @import("sampler/module.zig");
/// The C API: every file that exports a symbol. Forced below only when
/// libllama is the compilation root or under test; see `lib.zig`.
pub const lib = @import("lib.zig");
pub const unicode = @import("unicode.zig");
pub const arch = @import("arch.zig");
pub const gguf_meta = @import("gguf_meta.zig");
pub const hparams = @import("hparams.zig");
pub const mmap = @import("mmap.zig");
pub const loader = @import("loader.zig");
pub const model = @import("model.zig");
pub const models = @import("models/module.zig");
pub const chat = @import("chat.zig");
pub const batch = @import("batch.zig");
pub const io = @import("io.zig");
pub const memory = @import("memory.zig");
pub const cparams = @import("cparams.zig");
pub const adapter = @import("adapter.zig");
pub const kv_cells = @import("kv_cells.zig");
pub const kv_cache = @import("kv_cache.zig");
pub const kv_cache_dsv4 = @import("kv_cache_dsv4.zig");
pub const memory_recurrent = @import("memory_recurrent.zig");
pub const memory_hybrid = @import("memory_hybrid.zig");
pub const memory_hybrid_iswa = @import("memory_hybrid_iswa.zig");
pub const memory_hybrid_idx = @import("memory_hybrid_idx.zig");
pub const graph = @import("graph.zig");
pub const moe_cache = @import("moe_cache.zig");
pub const context_state = @import("context_state.zig");
pub const meta_split = @import("meta_split.zig");
pub const vocab = @import("vocab/module.zig");
pub const grammar = @import("grammar/module.zig");

pub const context_opt = @import("context_opt.zig");
pub const kv_cache_msa = @import("kv_cache_msa.zig");
pub const log = @import("log.zig");
pub const model_saver = @import("model_saver.zig");
pub const quant = @import("quant.zig");

pub const arch_table = @import("arch_table.zig");
pub const kv_cache_dsa = @import("kv_cache_dsa.zig");
pub const kv_cache_dsa_iswa = @import("kv_cache_dsa_iswa.zig");
pub const kv_cache_iswa = @import("kv_cache_iswa.zig");
pub const model_table = @import("model_table.zig");
pub const unicode_data = @import("unicode_data.zig");
pub const win32 = @import("win32.zig");

comptime {
    if (@import("root") == @This() or @import("builtin").is_test) _ = lib;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = regex;
    _ = unicode;
    _ = arch;
    _ = gguf_meta;
    _ = hparams;
    _ = mmap;
    _ = loader;
    _ = model;
    _ = models;
    _ = chat;
    _ = batch;
    _ = io;
    _ = memory;
    _ = cparams;
    _ = adapter;
    _ = kv_cells;
    _ = kv_cache;
    _ = kv_cache_dsv4;
    _ = memory_recurrent;
    _ = memory_hybrid;
    _ = memory_hybrid_iswa;
    _ = memory_hybrid_idx;
    _ = graph;
    _ = context;
    _ = moe_cache;
    _ = context_state;
    _ = meta_split;
    _ = api;
    _ = vocab;
    _ = grammar;
    _ = impl;
}
