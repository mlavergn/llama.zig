//! Exports `llama.h`: the step that makes this library libllama.
//!
//! **Not a port.** Every `llama_*` function in `api/` is a `pub` Zig
//! function with the C signature and no export, so the in-process harnesses
//! can call it beside the C++ that defines the same name. Importing this
//! file exports all of them by name, which is libllama's single swap at
//! `llama.h` (`PLAN.md`, Decision 42). Only `module.zig`, the library root,
//! imports it; `staged.zig`, the harnesses' root, must not.
//!
//! One C++-linkage symbol is exported too: `llama_log_internal`, which
//! `impl.zig`'s `logInfo` and friends call by its mangled name. In the
//! harnesses that name resolves to the C++'s logger; here it resolves to
//! `log.zig`'s, so the port logs through the callback `llama_log_set`
//! installed without either side choosing at build time.

const std = @import("std");
const api = @import("api/module.zig");
const log = @import("log.zig");

/// Entry points `api/` ports from `llama-ext.h`, which declares them with
/// C++ linkage: they are no part of `llama.h`'s C contract, and the
/// reference exports none of them by these names.
const ext_only = [_][]const u8{
    "llama_get_embeddings_layer_inp",
    "llama_get_embeddings_nextn",
    "llama_get_embeddings_nextn_ith",
    "llama_set_embeddings_layer_inp",
    "llama_set_embeddings_nextn",
    "llama_set_nextn_layer_offset",
};

fn isExtOnly(name: []const u8) bool {
    for (ext_only) |e| if (std.mem.eql(u8, e, name)) return true;
    return false;
}

comptime {
    @setEvalBranchQuota(100_000);
    for (@typeInfo(api).@"struct".decl_names) |file_name| {
        const file = @field(api, file_name);
        if (@TypeOf(file) != type) continue;
        for (@typeInfo(file).@"struct".decl_names) |name| {
            if (!std.mem.startsWith(u8, name, "llama_")) continue;
            if (isExtOnly(name)) continue;
            const f = @field(file, name);
            if (@typeInfo(@TypeOf(f)) != .@"fn") continue;
            @export(&f, .{ .name = name });
        }
    }
    @export(&log.logInternal, .{ .name = "_Z18llama_log_internal14ggml_log_levelPKcz" });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
