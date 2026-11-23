//! `llama-server-core`'s tests, from where a consumer stands: `zig build
//! test-server-core` imports the module by its exported name, as
//! `llama_server`, as zigllm imports it, so only what it exports is
//! reachable and only these tests run. `testing/scripts/server-core-consumer` runs
//! that step from another directory, as a consumer's build runs it.
//!
//! # Provenance
//!
//! **Not a port.** It drives the server the way
//! `llama.cpp/tools/server/server-context.cpp`'s `handle_completions_impl`
//! does, through `routes.zig`'s port of it.

const std = @import("std");
const builtin = @import("builtin");
const llama_server = @import("llama_server");
const llama_h = @import("llama_h");

// `refAllDecls` analyzes declarations, not function bodies: only a function
// whose address is taken is compiled. These are the entry points a
// consumer calls, so a change that breaks one fails here, not in zigllm.
test "the in-process API compiles" {
    const SC = llama_server.context.ServerContext;
    const Reader = llama_server.context.Reader;
    inline for (.{
        &SC.init,                                      &SC.deinit,                                &SC.loadModel,
        &SC.startLoop,                                 &SC.terminate,                             &SC.getLlamaContext,
        &SC.setStateCallback,                          &SC.getMeta,                               &llama_server.context.installChat,
        &Reader.init,                                  &Reader.deinit,                            &Reader.getNewId,
        &Reader.postTask,                              &Reader.postTasks,                         &Reader.hasNext,
        &Reader.stop,                                  &llama_server.task.Task.arenaAlloc,        &llama_server.task.Task.deinit,
        &llama_server.result.Result.toJson,            &llama_server.result.Result.destroy,       &llama_server.tokens.tokenizeInputPrompts,
        &llama_server.schema.evalLlamaCmplSchema,      &llama_server.convert.responsesToChatcmpl, &llama_server.convert.anthropicToOai,
        &llama_server.decision.groupTasks,             &llama_server.tools_common.arg.parse,      &llama_server.tools_common.common.init,
        &llama_server.tools_common.params.Params.init,
    }) |f| std.mem.doNotOptimizeAway(f);
    // POSIX only for now: MCP, the server-side tools, `subproc`'s waiter
    // (a self-pipe), and the chat parser, whose media loading fetches URLs
    // through `download.zig` and httplib's client.
    if (builtin.os.tag != .windows) {
        inline for (.{
            &llama_server.subproc.Waiter.init,                &llama_server.subproc.Waiter.deinit,
            &llama_server.oaicompat.oaicompatChatParamsParse, &llama_server.mcp.Mcp.start,
            &llama_server.mcp.Mcp.listTools,                  &llama_server.mcp.Mcp.callTool,
            &llama_server.mcp.Mcp.deinit,                     &llama_server.mcp.Transport.create,
            &llama_server.tools.Tools.init,                   &llama_server.tools.Tools.setup,
            &llama_server.tools.Tools.deinit,
        }) |f| std.mem.doNotOptimizeAway(f);
    }
}

// Two translations of `llama.h` would give a consumer two incompatible
// `*llama_model`: the exported `llama_h` must be the server core's.
test "llama_h is the translation the server core speaks" {
    try std.testing.expect(llama_h.llama_model == llama_server.tools_common.c.llama_model);
    try std.testing.expect(llama_h.mtmd_context == llama_server.tools_common.c.mtmd_context);
}

/// A one-layer `llama` model with random weights and an SPM vocabulary of
/// `<unk>`, `<s>`, `</s>`, then `<tN>`: enough for `loadModel` and a
/// decode, small enough to write in a test.
const TinyModel = struct {
    const n_vocab = 64;
    const n_embd = 64;
    const n_head = 2;
    const n_ff = 128;
    const n_ctx_train = 256;

    /// Writes the model to `path`.
    ///
    /// Parameters:
    /// - `gpa`: scratch for the token texts, freed before return.
    /// - `path`: where the GGUF goes.
    ///
    /// Return: nothing; `error.OutOfMemory`, or `error.WriteFailed` when
    /// `gguf_write_to_file` fails.
    fn write(gpa: std.mem.Allocator, path: [:0]const u8) !void {
        const meta = llama_server.tools_common.c.gguf_init_empty().?;
        defer llama_server.tools_common.c.gguf_free(meta);
        llama_server.tools_common.c.gguf_set_val_str(meta, "general.architecture", "llama");
        llama_server.tools_common.c.gguf_set_val_u32(meta, "llama.context_length", n_ctx_train);
        llama_server.tools_common.c.gguf_set_val_u32(meta, "llama.embedding_length", n_embd);
        llama_server.tools_common.c.gguf_set_val_u32(meta, "llama.block_count", 1);
        llama_server.tools_common.c.gguf_set_val_u32(meta, "llama.feed_forward_length", n_ff);
        llama_server.tools_common.c.gguf_set_val_u32(meta, "llama.attention.head_count", n_head);
        llama_server.tools_common.c.gguf_set_val_u32(meta, "llama.attention.head_count_kv", n_head);
        llama_server.tools_common.c.gguf_set_val_f32(meta, "llama.attention.layer_norm_rms_epsilon", 1e-5);

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var texts: [n_vocab][*c]const u8 = undefined;
        var scores: [n_vocab]f32 = undefined;
        var types: [n_vocab]i32 = @splat(1); // normal
        texts[0] = "<unk>";
        texts[1] = "<s>";
        texts[2] = "</s>";
        types[0] = 2; // unknown
        types[1] = 3; // control
        types[2] = 3;
        for (3..n_vocab) |i| texts[i] = try std.fmt.allocPrintSentinel(a, "<t{d}>", .{i}, 0);
        for (&scores, 0..) |*s, i| s.* = -@as(f32, @floatFromInt(i));
        llama_server.tools_common.c.gguf_set_val_str(meta, "tokenizer.ggml.model", "llama");
        llama_server.tools_common.c.gguf_set_arr_str(meta, "tokenizer.ggml.tokens", &texts, n_vocab);
        llama_server.tools_common.c.gguf_set_arr_data(meta, "tokenizer.ggml.scores", llama_server.tools_common.c.GGUF_TYPE_FLOAT32, &scores, n_vocab);
        llama_server.tools_common.c.gguf_set_arr_data(meta, "tokenizer.ggml.token_type", llama_server.tools_common.c.GGUF_TYPE_INT32, &types, n_vocab);
        llama_server.tools_common.c.gguf_set_val_u32(meta, "tokenizer.ggml.unknown_token_id", 0);
        llama_server.tools_common.c.gguf_set_val_u32(meta, "tokenizer.ggml.bos_token_id", 1);
        llama_server.tools_common.c.gguf_set_val_u32(meta, "tokenizer.ggml.eos_token_id", 2);

        const tensors = [_]struct { name: [:0]const u8, ne0: i64, ne1: i64 }{
            .{ .name = "token_embd.weight", .ne0 = n_embd, .ne1 = n_vocab },
            .{ .name = "output_norm.weight", .ne0 = n_embd, .ne1 = 1 },
            .{ .name = "output.weight", .ne0 = n_embd, .ne1 = n_vocab },
            .{ .name = "blk.0.attn_norm.weight", .ne0 = n_embd, .ne1 = 1 },
            .{ .name = "blk.0.attn_q.weight", .ne0 = n_embd, .ne1 = n_embd },
            .{ .name = "blk.0.attn_k.weight", .ne0 = n_embd, .ne1 = n_embd },
            .{ .name = "blk.0.attn_v.weight", .ne0 = n_embd, .ne1 = n_embd },
            .{ .name = "blk.0.attn_output.weight", .ne0 = n_embd, .ne1 = n_embd },
            .{ .name = "blk.0.ffn_norm.weight", .ne0 = n_embd, .ne1 = 1 },
            .{ .name = "blk.0.ffn_gate.weight", .ne0 = n_embd, .ne1 = n_ff },
            .{ .name = "blk.0.ffn_up.weight", .ne0 = n_embd, .ne1 = n_ff },
            .{ .name = "blk.0.ffn_down.weight", .ne0 = n_ff, .ne1 = n_embd },
        };
        var n_floats: usize = 0;
        for (tensors) |t| n_floats += @intCast(t.ne0 * t.ne1);
        const g = llama_server.tools_common.c.ggml_init(.{ .mem_size = n_floats * @sizeOf(f32) + tensors.len * llama_server.tools_common.c.ggml_tensor_overhead(), .mem_buffer = null, .no_alloc = false }).?;
        defer llama_server.tools_common.c.ggml_free(g);
        var lcg: u32 = 12345;
        for (tensors) |t| {
            const x = if (t.ne1 == 1) llama_server.tools_common.c.ggml_new_tensor_1d(g, llama_server.tools_common.c.GGML_TYPE_F32, t.ne0) else llama_server.tools_common.c.ggml_new_tensor_2d(g, llama_server.tools_common.c.GGML_TYPE_F32, t.ne0, t.ne1);
            _ = llama_server.tools_common.c.ggml_set_name(x, t.name);
            const norm = std.mem.indexOf(u8, t.name, "norm") != null;
            const data: [*]f32 = @ptrCast(@alignCast(x.*.data));
            for (data[0..@intCast(t.ne0 * t.ne1)]) |*d| {
                lcg = lcg *% 1103515245 +% 12345;
                const r = @as(f32, @floatFromInt((lcg >> 8) & 0xffff)) / 65536.0 - 0.5;
                d.* = if (norm) 1.0 + r * 0.2 else r * 0.5;
            }
            llama_server.tools_common.c.gguf_add_tensor(meta, x);
        }
        if (!llama_server.tools_common.c.gguf_write_to_file(meta, path.ptr, false)) return error.WriteFailed;
    }
};

const NeverStop = struct {
    pub fn shouldStop(_: NeverStop) bool {
        return false;
    }
};

// The path a consumer drives, as `routes.handleCompletionsImpl` does it:
// parse arguments, load, run the loop on its own thread, tokenize, build a
// task from the request body, post it and wait for its result.
test "a completion runs in-process: load, post, wait for the result" {
    const gpa = std.heap.c_allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrintSentinel(gpa, ".zig-cache/tmp/{s}/tiny.gguf", .{&tmp.sub_path}, 0);
    defer gpa.free(path);
    try TinyModel.write(gpa, path);

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    llama_server.tools_common.common.init();
    llama_server.context.installChat();
    var params = try llama_server.tools_common.params.Params.init(a);
    const argv = [_][:0]const u8{ "llama-server-core", "-m", path, "-c", "128", "-np", "1", "-ngl", "0", "--fit", "off", "-t", "2", "--no-warmup", "--log-disable" };
    try std.testing.expect(try llama_server.tools_common.arg.parse(&argv, &params, .server, null));
    llama_server.tools_common.c.llama_backend_init();
    defer llama_server.tools_common.c.llama_backend_free();

    var ctx = try llama_server.context.ServerContext.init();
    defer ctx.deinit();
    try std.testing.expect(try ctx.loadModel(&params));
    const meta = try ctx.getMeta(a);
    try std.testing.expectEqual(@as(i32, TinyModel.n_vocab), meta.model_vocab_n_tokens);

    const loop = try std.Thread.spawn(.{}, struct {
        fn run(s: *llama_server.context.ServerContext) void {
            s.startLoop() catch {};
        }
    }.run, .{&ctx});
    defer {
        ctx.terminate();
        loop.join();
    }

    var reader = try llama_server.context.Reader.init(gpa, &ctx.impl.queue_tasks, &ctx.impl.queue_results, llama_server.context.http_polling_seconds);
    defer reader.deinit();

    // eos is ignored: random weights would otherwise end the reply early
    var data = try llama_server.json.parse(a, "{\"prompt\":[1,5,6,7],\"n_predict\":4,\"ignore_eos\":true,\"temperature\":0}");
    var inputs = try llama_server.tokens.tokenizeInputPrompts(gpa, ctx.impl.vocab, ctx.impl.mctx, try data.at("prompt"), true, true, ctx.impl.init_opt);
    defer {
        for (inputs.items) |*t| t.deinit();
        inputs.deinit(gpa);
    }
    try std.testing.expectEqual(@as(usize, 1), inputs.items.len);

    const Task = llama_server.task.Task;
    var tasks: [1]Task = .{Task.initType(.completion)};
    tasks[0].id = reader.getNewId();
    tasks[0].tokens = inputs.items[0];
    inputs.items[0] = llama_server.tokens.ServerTokens.init(gpa);
    const ta = try tasks[0].arenaAlloc(gpa);
    tasks[0].params = try llama_server.schema.evalLlamaCmplSchema(ta, ctx.impl.vocab, &params, meta.logit_bias_eog, &data);
    try reader.postTasks(&tasks, null, false);

    var all = try reader.waitForAll(NeverStop{});
    defer all.deinit(gpa);
    try std.testing.expect(!all.is_terminated);
    try std.testing.expect(all.err == null);
    try std.testing.expectEqual(@as(usize, 1), all.results.len);
    const final = &all.results[0].?.kind.cmpl_final;
    try std.testing.expectEqual(@as(i32, 4), final.n_prompt_tokens);
    try std.testing.expectEqual(@as(i32, 4), final.n_decoded);
    try std.testing.expectEqual(llama_server.task.StopType.limit, final.stop);
}
