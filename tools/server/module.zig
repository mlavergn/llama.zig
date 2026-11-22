//! Barrel for `tools/server/`: `llama-server`'s port.
//!
//! # Provenance
//!
//! **Not a port.** The files it re-exports port parts of
//! `llama.cpp/tools/server/`; each names its own. The HTTP layer
//! (`http.zig`, `httplib/`) and the executable (`main.zig`, `server.zig`)
//! are built from their own roots for now; this is the data layer and the
//! queues.

const std = @import("std");

pub const common = @import("common.zig");
pub const tokens = @import("tokens.zig");
pub const oaicompat = @import("oaicompat.zig");
pub const convert = @import("convert.zig");
pub const task = @import("task.zig");
pub const result = @import("result.zig");
pub const schema = @import("schema.zig");
pub const subproc = @import("subproc.zig");
pub const queue = @import("queue.zig");
pub const decision = @import("decision.zig");

pub const connect = @import("connect.zig");
pub const context = @import("context.zig");
pub const cors_proxy = @import("cors_proxy.zig");
pub const download = @import("download.zig");
pub const file_lock = @import("file_lock.zig");
pub const hf_cache = @import("hf_cache.zig");
pub const http = @import("http.zig");
pub const mcp = @import("mcp.zig");
pub const models = @import("models.zig");
pub const models_sched = @import("models_sched.zig");
pub const preset = @import("preset.zig");
pub const proxy = @import("proxy.zig");
pub const realtime = @import("realtime.zig");
pub const routes = @import("routes.zig");
pub const stream = @import("stream.zig");
pub const tools = @import("tools.zig");

pub const httplib = @import("httplib/module.zig");

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    // libllama and libmtmd, compiled in: their exports are what `c.llama_*`
    // and `c.mtmd_*` link against.
    _ = @import("llama").lib;
    _ = @import("mtmd").lib;
}

const task_mod = @import("task.zig");
const result_mod = @import("result.zig");

const NeverStop = struct {
    pub fn shouldStop(_: NeverStop) bool {
        return false;
    }
};

// The interface queue.zig is generic over, met by the data layer's types.
test "a task and its streamed results go through the queues and the reader's state" {
    const gpa = std.testing.allocator;
    var q = queue.Queue(task_mod.Task).init(gpa);
    defer q.deinit();
    var resp = try queue.Response(*result_mod.Result).init(gpa);
    defer resp.deinit();
    var reader = try queue.ResponseReader(task_mod.Task, *result_mod.Result, task_mod.TaskResultState).init(gpa, &q, &resp, 1);
    defer reader.deinit();

    var t = task_mod.Task.initType(.completion);
    t.id = q.getNewId();
    const a = try t.arenaAlloc(gpa);
    t.params = try task_mod.TaskParams.init(a);
    const id = t.id;
    try reader.postTask(t, false);

    const p = try result_mod.Result.create(gpa, .cmpl_partial);
    p.id = id;
    p.kind.cmpl_partial.content = try .fromText(p.alloc(), "he");
    p.kind.cmpl_partial.n_decoded = 1;
    try resp.send(p);
    const f = try result_mod.Result.create(gpa, .cmpl_final);
    f.id = id;
    f.kind.cmpl_final.content = try .fromText(f.alloc(), "llo");
    try resp.send(f);

    const r1 = (try reader.next(NeverStop{})).?;
    defer r1.destroy(gpa);
    try std.testing.expect(!r1.isStop());
    try std.testing.expectEqualStrings("he", r1.kind.cmpl_partial.oaicompat_msg_diffs.items[0].content_delta);
    const r2 = (try reader.next(NeverStop{})).?;
    defer r2.destroy(gpa);
    try std.testing.expect(r2.isStop());
    try std.testing.expectEqualStrings("hello", r2.kind.cmpl_final.oaicompat_msg.content);
    try std.testing.expect(!reader.hasNext());
}
