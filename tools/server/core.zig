//! Barrel for `llama-server-core`: `llama-server`'s port minus its HTTP
//! front, for a consumer that drives `ServerContext` in-process (zigllm).
//!
//! # Provenance
//!
//! **Not a port.** The files it re-exports port parts of
//! `llama.cpp/tools/server/`; each names its own. `routes.zig`, `server.zig`
//! and the router stay out: a consumer posts tasks and reads results the way
//! `routes.handleCompletionsImpl` does, converting Responses, Anthropic and
//! transcription bodies with `convert` and grouping tasks with `decision` as
//! the routes do. `context.zig` (through `realtime.zig`) and `tools.zig` still
//! reach `http.zig`, so the module compiles it, but nothing here starts a
//! listener.

const std = @import("std");

pub const common = @import("common.zig");
pub const context = @import("context.zig");
pub const task = @import("task.zig");
pub const queue = @import("queue.zig");
pub const result = @import("result.zig");
pub const oaicompat = @import("oaicompat.zig");
pub const convert = @import("convert.zig");
pub const decision = @import("decision.zig");
pub const tokens = @import("tokens.zig");
pub const schema = @import("schema.zig");
pub const tools = @import("tools.zig");
pub const mcp = @import("mcp.zig");
pub const subproc = @import("subproc.zig");

/// `tools/common/`: `Params`, argument parsing, logging and `llama.h`.
pub const tools_common = @import("tools_common");
/// The JSON value type the server's data layer speaks.
pub const json = @import("json");
/// `common/chat`'s port: templates, sessions and parsed messages.
pub const common_chat = @import("common_chat");

comptime {
    // libllama and libmtmd, compiled in: their exports are what `c.llama_*`
    // and `c.mtmd_*` link against.
    _ = @import("llama").lib;
    _ = @import("mtmd").lib;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
