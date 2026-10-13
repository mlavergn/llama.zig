//! Barrel for the ported libllama.
//!
//! # Provenance
//!
//! **Not a port.** Everything under `src/llama/` is a port of C++ in the
//! pinned reference checkout, v0.3.0 (`c1d0e7a00`), or of the libc++ it was
//! compiled against; each file states its sources.
//!
//! # How this is swapped in
//!
//! libllama is one strongly connected component of C++ translation units
//! (`PLAN.md`, Decision 42), so it swaps once, at `llama.h` -- with one
//! exception, which is what this barrel holds today: `llama-sampler.cpp`
//! and `llama-grammar.cpp`, whose contract with the rest of libllama is C.
//! `build/llamacpp.zig` roots the libllama module here and drops those two
//! files from its sources, and the C++ that stays links against these
//! exports.
//!
//! Files ported ahead of the swap and not yet reachable from it -- the
//! vocabulary under `vocab/`, `unicode.zig` and `gguf_meta.zig` -- are
//! imported by the tests below and gated in-process by their own diff
//! harnesses, which compile them through `staged.zig`.

const std = @import("std");

comptime {
    // llama-sampler.cpp, split across src/llama/sampler/; it reaches
    // llama-grammar.cpp (src/llama/grammar/) and std::regex (regex.zig).
    _ = @import("sampler/module.zig");
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
    _ = @import("loader.zig");
    _ = @import("model.zig");
    _ = @import("vocab/module.zig");
    _ = @import("grammar/parser.zig");
    _ = @import("grammar/grammar.zig");
    _ = @import("impl.zig");
}
