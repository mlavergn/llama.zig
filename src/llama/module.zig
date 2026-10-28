//! Barrel for the ported libllama.
//!
//! # Provenance
//!
//! **Not a port.** Everything under `src/llama/` is a port of C++ in the
//! pinned reference checkout, v0.4.0 (`5266f24da`), or of the libc++ it was
//! compiled against; each file states its sources.
//!
//! # The swap
//!
//! libllama is one strongly connected component of C++ translation units
//! (`PLAN.md`, Decision 42), so it swapped once, at `llama.h`. The sampler
//! and grammar went first, their contract with the rest being C; every
//! other translation unit followed together. `build/llamacpp.zig` roots
//! `libllama.a` here and compiles no C or C++ into it: `exports.zig` exports
//! every function of `llama.h`, and the sampler exports its own.
//!
//! The in-process harnesses compile the same files through `harness_root.zig`,
//! which exports nothing, and link upstream's C++ as `libllama-ref.a`
//! beside them.

const std = @import("std");

/// What the CLI reaches as Zig: `cli/fit.zig` ports `common/fit.cpp`,
/// which calls libllama through its internal `llama-ext.h` as well as
/// `llama.h`. Declaring these exports nothing.
pub const api = @import("api/module.zig");
pub const impl = @import("impl.zig");
pub const context = @import("context.zig");
/// libc++'s containers and algorithms as the reference links them, for the
/// tools under `tools/` that port code reading their order.
pub const libcxx = @import("libcxx.zig");

comptime {
    // llama-sampler.cpp, split across src/llama/sampler/; it reaches
    // llama-grammar.cpp (src/llama/grammar/) and std::regex (regex.zig).
    _ = @import("sampler/module.zig");
    // Every other llama.h entry point, from api/.
    _ = @import("exports.zig");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = @import("libcxx.zig");
    _ = @import("regex.zig");
    _ = @import("unicode.zig");
    _ = @import("arch.zig");
    _ = @import("gguf_meta.zig");
    _ = @import("hparams.zig");
    _ = @import("mmap.zig");
    _ = @import("loader.zig");
    _ = @import("model.zig");
    _ = @import("models/module.zig");
    _ = @import("chat.zig");
    _ = @import("batch.zig");
    _ = @import("io.zig");
    _ = @import("memory.zig");
    _ = @import("cparams.zig");
    _ = @import("adapter.zig");
    _ = @import("kv_cells.zig");
    _ = @import("kv_cache.zig");
    _ = @import("kv_cache_dsv4.zig");
    _ = @import("memory_recurrent.zig");
    _ = @import("memory_hybrid.zig");
    _ = @import("memory_hybrid_iswa.zig");
    _ = @import("memory_hybrid_idx.zig");
    _ = @import("graph.zig");
    _ = @import("context.zig");
    _ = @import("context_state.zig");
    _ = @import("meta_split.zig");
    _ = @import("api/module.zig");
    _ = @import("vocab/module.zig");
    _ = @import("grammar/parser.zig");
    _ = @import("grammar/grammar.zig");
    _ = @import("impl.zig");
}
