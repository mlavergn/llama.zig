//! The root the in-process harnesses compile.
//!
//! **Not a port.** The in-process harnesses link the port beside upstream's
//! C++ (`libllama-ref.a`) and call both under the same names. They cannot
//! root at `module.zig`: it imports `exports.zig`, which exports every
//! `llama.h` function and the sampler's, and two definitions of a symbol in
//! one link pick one silently. This root re-exports the port's files and
//! nothing that exports a symbol. Never import `exports.zig` from here.

const std = @import("std");

pub const impl = @import("impl.zig");
pub const libcxx = @import("libcxx.zig");
pub const regex = @import("regex.zig");
pub const unicode = @import("unicode.zig");
pub const arch = @import("arch.zig");
pub const arch_table = @import("arch_table.zig");
pub const gguf_meta = @import("gguf_meta.zig");
pub const hparams = @import("hparams.zig");
pub const mmap = @import("mmap.zig");
pub const loader = @import("loader.zig");
pub const model = @import("model.zig");
pub const model_table = @import("model_table.zig");
pub const vocab = @import("vocab/module.zig");
pub const chat = @import("chat.zig");
pub const batch = @import("batch.zig");
pub const models = @import("models/module.zig");
pub const models_qwen35 = models.qwen35;
pub const io = @import("io.zig");
pub const memory = @import("memory.zig");
pub const cparams = @import("cparams.zig");
pub const adapter = @import("adapter.zig");
pub const kv_cells = @import("kv_cells.zig");
pub const kv_cache = @import("kv_cache.zig");
pub const memory_recurrent = @import("memory_recurrent.zig");
pub const memory_hybrid = @import("memory_hybrid.zig");
pub const memory_hybrid_idx = @import("memory_hybrid_idx.zig");
pub const graph = @import("graph.zig");
pub const context = @import("context.zig");
pub const context_state = @import("context_state.zig");
pub const context_opt = @import("context_opt.zig");
pub const meta_split = @import("meta_split.zig");
pub const quant = @import("quant.zig");
pub const model_saver = @import("model_saver.zig");
pub const api = @import("api/module.zig");
pub const log = @import("log.zig");

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
