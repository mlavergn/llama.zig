//! Exports `llama.h`: the step that makes this library libllama.
//!
//! **Not a port.** Every `llama_*` function in `api/` is a `pub` Zig
//! function with the C signature and no export, so the in-process harnesses
//! can call it beside the C++ that defines the same name. Importing this
//! file exports all of them by name, which is libllama's single swap at
//! `llama.h` (`PLAN.md`, Decision 42). Only `module.zig`, the library root,
//! imports it; `harness_root.zig`, the harnesses' root, must not.

const std = @import("std");
const api = @import("api/module.zig");

/// Entry points `api/` ports from `llama-ext.h`, which declares them with
/// C++ linkage: they are no part of `llama.h`'s C contract, and the
/// reference exports none of them by these names.
const ext_only = [_][]const u8{
    "llama_arch_all",
    "llama_arch_from_string",
    "llama_arch_kv_name",
    "llama_arch_name",
    "llama_arch_supports_mixed_batch",
    "llama_arch_supports_save",
    "llama_arch_supports_sm_tensor",
    "llama_batch_ext_add_embd_row",
    "llama_batch_ext_set_decision_order",
    "llama_get_ctx_other",
    "llama_get_embeddings_layer_inp",
    "llama_get_embeddings_nextn",
    "llama_get_embeddings_nextn_ith",
    "llama_get_memory_breakdown",
    "llama_memory_get_buffers",
    "llama_model_decision_label",
    "llama_model_dflash_selector_top_k",
    "llama_model_get_device",
    "llama_model_get_tok_embd",
    "llama_model_get_tok_embd_rows",
    "llama_model_n_devices",
    "llama_model_n_expert",
    "llama_model_n_tokens_delay",
    "llama_model_set_attn_rot",
    "llama_model_target_layer_ids",
    "llama_model_target_layer_ids_n",
    "llama_set_embeddings_layer_inp",
    "llama_set_embeddings_layer_inp_dev",
    "llama_set_embeddings_nextn",
    "llama_set_nextn_layer_offset",
    "llama_supports_embd_row",
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
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
