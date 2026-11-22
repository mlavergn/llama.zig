//! libllama's C API: every file that exports a `llama.h` symbol, forced into
//! the compilation.
//!
//! # Provenance
//!
//! **Not a port.** Linking this is what defines the `llama_*` symbols in a
//! binary. `module.zig` forces it when libllama itself is the compilation
//! root, and under test; an executable that compiles the `llama` module in
//! and calls the C API forces it too (`_ = llama.lib`). A harness never does:
//! it links upstream's libllama beside the port, and two definitions of a
//! symbol pick one silently.

const std = @import("std");
const mod = @import("module.zig");

comptime {
    // Every llama.h entry point but the sampler's, from api/.
    _ = @import("exports.zig");
    // llama-sampler.cpp, split across src/llama/sampler/: each `llama_*`
    // function there, by its own name, and the two C++-linkage helpers
    // libllama-ref's callers reach by mangled name.
    @setEvalBranchQuota(100_000);
    for (@typeInfo(mod.sampler).@"struct".decl_names) |file_name| {
        const file = @field(mod.sampler, file_name);
        if (@TypeOf(file) != type) continue;
        for (@typeInfo(file).@"struct".decl_names) |name| {
            if (!std.mem.startsWith(u8, name, "llama_")) continue;
            const f = @field(file, name);
            if (@typeInfo(@TypeOf(f)) != .@"fn") continue;
            @export(&f, .{ .name = name });
        }
    }
    // Mangled names from `nm` on the reference `llama-sampler.o`.
    @export(&mod.sampler.chain.backendBegin, .{ .name = "_Z27llama_sampler_backend_beginP13llama_sampler" });
    @export(&mod.sampler.chain.backendNNodes, .{ .name = "_Z29llama_sampler_backend_n_nodesPK13llama_sampler" });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
