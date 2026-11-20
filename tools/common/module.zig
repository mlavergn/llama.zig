//! What the ported tools share: the C ABI, `build-info.h`, and the parts of
//! upstream's `common/` they call.
//!
//! # Provenance
//!
//! Barrel for `tools/common/`. `build_info.zig` ports
//! `llama.cpp/common/build-info.cpp.in` (v1.0.0, `b1b770d90`) as
//! `llama.cpp.zmake` fills it in; the rest says what it ports where it does.
//!
//! # One import of the headers
//!
//! `c` is the only `tools_h` every tool reaches: two translations of
//! `llama.h` in one binary give two incompatible `*llama_model`.

const std = @import("std");

pub const c = @import("tools_h");
pub const build_info = @import("build_info.zig");
pub const io = @import("io.zig");
pub const log = @import("log.zig");
pub const strings = @import("strings.zig");
pub const ggml = @import("ggml.zig");
pub const hash_map = @import("hash_map.zig");
pub const imatrix_loader = @import("imatrix_loader.zig");
pub const params = @import("params.zig");
pub const arg = @import("arg.zig");
pub const arg_options = @import("arg_options.zig");
pub const arg_options2 = @import("arg_options2.zig");
pub const arg_options3 = @import("arg_options3.zig");
pub const arg_options4 = @import("arg_options4.zig");
pub const common = @import("common.zig");
pub const cpu = @import("cpu.zig");
pub const init = @import("init.zig");
pub const sampling = @import("sampling.zig");
pub const reasoning_budget = @import("reasoning_budget.zig");
pub const ngram_cache = @import("ngram_cache.zig");
pub const ngram_map = @import("ngram_map.zig");
pub const ngram_mod = @import("ngram_mod.zig");
pub const speculative = @import("speculative.zig");
pub const speculative_draft = @import("speculative_draft.zig");
pub const console = @import("console.zig");
pub const chat = @import("chat.zig");
pub const subproc = @import("subproc.zig");
pub const fs_stat = @import("fs_stat.zig");
pub const file = @import("file.zig");
pub const env = @import("env.zig");
pub const num = @import("num.zig");
pub const clock = @import("clock.zig");

/// `LLM_KV_SPLIT_NO`, `LLM_KV_SPLIT_COUNT`, `LLM_KV_SPLIT_TENSORS_COUNT`
/// (common/common.h:1269, 1270, 1271 @b1b770d90).
pub const kv_split_no = "split.no";
pub const kv_split_count = "split.count";
pub const kv_split_tensors_count = "split.tensors.count";

/// The C's `exit`: what is buffered for stdout and stderr is written first,
/// as C's `exit` flushes stdio and upstream's tools rely on.
pub fn cExit(code: c_int) noreturn {
    io.exitFlushed(code);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
