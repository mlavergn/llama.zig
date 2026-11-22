//! `llama-mtmd-cli`: a model with an mmproj, given images or audio, as one
//! turn (`-p` with `--image`) or a conversation.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/mtmd/mtmd-cli.cpp` at v1.0.0 (`b1b770d90`).
//! Each declaration names the C++ it replaces and the line it began at.
//!
//! # What differs
//!
//! - `MTMD_DEBUG_GRAPH` installs upstream's `common_debug_cb_eval`, from
//!   `common/debug.cpp`, which is not ported: with it set this refuses to
//!   run rather than silently not printing the graph.
//! - Chat templates are `cli/common_chat`'s (`common_chat_templates_init`,
//!   `common_chat_format_single`, `common_chat_format_example`), as the
//!   server's: building a template runs its differential
//!   analysis, which logs at debug.
//! - Video: the reference is built without `MTMD_VIDEO`, so there are no
//!   video helpers on either side.

const std = @import("std");
const mtmd = @import("mtmd");
const llama = @import("llama");
const stdlib = @import("stdlib");
const builtin = @import("builtin");
const tools_common = @import("tools_common");
const common_chat = @import("common_chat");

/// libmtmd's port, for `mtmd_get_memory_usage`, which has C++ linkage
/// upstream and so no C symbol to call.

// Ports `g_is_generating`, `g_is_interrupted` (mtmd-cli.cpp:31, 32
// @b1b770d90): volatile, because of signal being an interrupt
var g_is_generating_: bool = false;
var g_is_interrupted_: bool = false;

fn isGenerating() bool {
    return @as(*volatile bool, &g_is_generating_).*;
}
fn setGenerating(v: bool) void {
    @as(*volatile bool, &g_is_generating_).* = v;
}
fn isInterrupted() bool {
    return @as(*volatile bool, &g_is_interrupted_).*;
}

/// Ports `show_additional_info` (mtmd-cli.cpp:40 @b1b770d90).
fn showAdditionalInfo(argv: []const [:0]const u8) void {
    tools_common.log.out("Experimental CLI for multimodal\n\n" ++
        "Usage: %s [options] -m <model> --mmproj <mmproj> --image <image> --audio <audio> -p <prompt>\n\n" ++
        "  -m and --mmproj are required\n" ++
        "  -hf user/repo can replace both -m and --mmproj in most cases\n" ++
        "  --image, --audio and -p are optional, if NOT provided, the CLI will run in chat mode\n" ++
        "  to disable using GPU for mmproj model, add --no-mmproj-offload\n", .{argv[0].ptr});
}

/// Ports `sigint_handler` (mtmd-cli.cpp:53 @b1b770d90).
fn sigintHandler(signo: std.posix.SIG) callconv(.c) void {
    if (signo == std.posix.SIG.INT) {
        if (isGenerating()) {
            setGenerating(false);
        } else {
            tools_common.console.cleanup();
            if (isInterrupted()) std.c._exit(1);
            @as(*volatile bool, &g_is_interrupted_).* = true;
        }
    }
}

/// Ports `inject_test_response_marker` (mtmd-cli.cpp:69 @b1b770d90): only
/// used by upstream's tests.sh to capture the response.
fn injectTestResponseMarker() void {
    if (stdlib.env.get("MTMD_TEST_RESPONSE_MARKER")) |env| tools_common.log.out("%s\n", .{env.ptr});
}

/// Ports `mtmd_cli_context` (mtmd-cli.cpp:76 @b1b770d90).
const CliContext = struct {
    a: std.mem.Allocator,
    ctx_vision: ?*mtmd.mtmd.Context = null,
    mtmd_cap: mtmd.helper.Cap = std.mem.zeroes(mtmd.helper.Cap),
    llama_init: tools_common.init.InitResult,

    model: ?*llama.impl.llama_c.llama_model,
    lctx: ?*llama.impl.llama_c.llama_context,
    vocab: ?*const llama.impl.llama_c.llama_vocab,
    smpl: *tools_common.sampling.Sampler,
    batch: tools_common.common.Batch,
    n_batch: c_int,

    bitmaps: std.ArrayList(*mtmd.mtmd.Bitmap) = .empty,

    init_opt: mtmd.helper.InitOpt = undefined, // mtmd_helper_init_opt_default(), set in `init`
    video_ffmpeg_bin_dir: [:0]const u8 = "",

    // chat template
    tmpls: common_chat.templates.Templates = undefined,
    chat_history: std.ArrayList(common_chat.Msg) = .empty,
    use_jinja: bool = false,

    // support for legacy templates (models not having EOT token)
    antiprompt_tokens: []const llama.impl.llama_c.llama_token = &.{},

    n_threads: c_int = 1,
    n_past: llama.impl.llama_c.llama_pos = 0,

    /// Ports the constructor `mtmd_cli_context` (mtmd-cli.cpp:108 @b1b770d90). Every failure is the
    /// C++'s `exit(1)`, which runs no destructor.
    fn init(self: *CliContext, gpa: std.mem.Allocator, a: std.mem.Allocator, params: *tools_common.params.Params) !void {
        const llama_init = try tools_common.init.initFromParams(gpa, params, false);
        const model = llama_init.model;
        if (model == null or llama_init.context == null) stdlib.stdio.exit(1);
        self.* = .{
            .a = a,
            .llama_init = llama_init,
            .model = model,
            .lctx = llama_init.context,
            .vocab = llama.api.model.llama_model_get_vocab(model),
            .smpl = samplerOrTerminate(params, model),
            .batch = tools_common.common.Batch.initEmpty(a),
            .n_batch = params.n_batch,
            .init_opt = mtmd.helper.mtmd_helper_init_opt_default(),
        };
        self.n_threads = params.cpuparams.n_threads;
        self.batch = tools_common.common.Batch.init(a, self.lctx); // batch for next token generation

        self.initVisionContext(params);

        self.mtmd_cap = mtmd.helper.mtmd_helper_get_capabilities(self.model, self.ctx_vision);
        // a realtime transcription model has no chat, it streams the audio below
        if (!self.mtmd_cap.chat and !self.mtmd_cap.audio_stream) {
            tools_common.log.err("Model does not support chat mode\n", .{});
            tools_common.log.err("Hint: for TTS models, please use llama-tts\n", .{});
            stdlib.stdio.exit(1);
        }

        if (llama.api.model.llama_model_chat_template(model, null) == null and params.chat_template.len == 0) {
            tools_common.log.err("Model does not have chat template.\n", .{});
            tools_common.log.err("  For old llava models, you may need to use '--chat-template vicuna'\n", .{});
            tools_common.log.err("  For MobileVLM models, use '--chat-template deepseek'\n", .{});
            tools_common.log.err("  For Mistral Small 3.1, use '--chat-template mistral-v7'\n", .{});
            stdlib.stdio.exit(1);
        }

        var info = try common_chat.templates.modelInfo(a, @ptrCast(model.?));
        self.tmpls = try common_chat.templates.init(a, &info, params.chat_template, "", "");
        self.use_jinja = params.use_jinja;
        self.chat_history.clearRetainingCapacity();
        var kwargs: std.ArrayList(common_chat.templates.KwArg) = .empty;
        for (params.default_template_kwargs.entries.items) |e| try kwargs.append(a, .{ .key = e.key, .value = e.value });
        const example = try a.dupeSentinel(u8, try common_chat.templates.formatExample(a, &self.tmpls, params.use_jinja, kwargs.items), 0);
        tools_common.log.inf("%s: chat template example:\n%s\n", .{ "mtmd_cli_context", example.ptr });

        // load antiprompt tokens for legacy templates
        if (std.mem.eql(u8, params.chat_template, "vicuna")) {
            self.antiprompt_tokens = try tools_common.common.tokenize(a, self.vocab, "ASSISTANT:", false, true);
        } else if (std.mem.eql(u8, params.chat_template, "deepseek")) {
            self.antiprompt_tokens = try tools_common.common.tokenize(a, self.vocab, "###", false, true);
        }
    }

    /// `common_sampler_init`, whose exceptions escape `main`.
    fn samplerOrTerminate(params: *tools_common.params.Params, model: ?*llama.impl.llama_c.llama_model) *tools_common.sampling.Sampler {
        return tools_common.init.samplerInit(params, model) catch |e| {
            const kind: [*:0]const u8 = if (e == error.InvalidArgument) "std::invalid_argument" else "std::runtime_error";
            tools_common.io.eprintf("libc++abi: terminating due to uncaught exception of type %s: %.*s\n", .{ kind, @as(c_int, @intCast(tools_common.arg.what().len)), tools_common.arg.what().ptr });
            std.process.abort();
        };
    }

    /// Ports the destructor of `mtmd_cli_context` (mtmd-cli.cpp:151 @b1b770d90), then the members in
    /// reverse order of declaration.
    fn deinit(self: *CliContext) void {
        self.smpl.destroy();
        for (self.bitmaps.items) |b| mtmd.mtmd.mtmd_bitmap_free(b);
        self.batch.deinit();
        self.llama_init.deinit();
        if (self.ctx_vision) |v| mtmd.mtmd.mtmd_free(v);
    }

    /// Ports `init_vision_context` (mtmd-cli.cpp:155 @b1b770d90).
    fn initVisionContext(self: *CliContext, params: *tools_common.params.Params) void {
        const mmproj_path = self.a.dupeSentinel(u8, params.mmproj.path, 0) catch stdlib.stdio.exit(1);
        var mparams = mtmd.mtmd.mtmd_context_params_default();
        mparams.use_gpu = params.mmproj_use_gpu;
        mparams.device = params.mmproj_device;
        mparams.print_timings = true;
        mparams.n_threads = params.cpuparams.n_threads;
        mparams.flash_attn_type = params.flash_attn_type;
        mparams.warmup = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;
        {
            // non-causal models need the whole image in one ubatch
            const n_ubatch: c_int = @bitCast(llama.api.context.llama_n_ubatch(self.lctx));
            const MtmdParams = @typeInfo(@TypeOf(mtmd.mtmd.getMemoryUsage)).@"fn".param_types[1].?;
            var mem = mtmd.mtmd.getMemoryUsage(mmproj_path.ptr, @as(*const MtmdParams, @ptrCast(&mparams)).*);
            defer mem.deinit();
            if (mem.use_non_causal and mem.image_max_tokens > n_ubatch) {
                tools_common.log.wrn("%s: cap image_max_tokens (original=%d) to n_ubatch (%d) because model needs non-causal attention on image\n", .{ "init_vision_context", mem.image_max_tokens, n_ubatch });
                tools_common.log.wrn("%s: increase n_ubatch (-ub) to increase vision token budget\n", .{"init_vision_context"});
                mparams.image_max_tokens = n_ubatch;
                mparams.image_min_tokens = @min(mparams.image_min_tokens, n_ubatch);
            }
        }
        if (stdlib.env.get("MTMD_DEBUG_GRAPH") != null) {
            tools_common.log.err("MTMD_DEBUG_GRAPH needs common_debug_cb_eval, which this port does not have\n", .{});
            stdlib.stdio.exit(1);
        }
        self.ctx_vision = mtmd.mtmd.mtmd_init_from_file(mmproj_path.ptr, self.model, mparams);
        if (self.ctx_vision == null) {
            tools_common.log.err("Failed to load vision model from %s\n", .{mmproj_path.ptr});
            stdlib.stdio.exit(1);
        }

        self.video_ffmpeg_bin_dir = self.a.dupeSentinel(u8, params.video_ffmpeg_bin_dir, 0) catch stdlib.stdio.exit(1);
        self.init_opt.video_params.fps_target = params.video_fps;
        self.init_opt.video_params.timestamp_interval_ms = params.video_timestamp_interval_ms;
        self.init_opt.video_params.ffmpeg_bin_dir = if (self.video_ffmpeg_bin_dir.len == 0) null else self.video_ffmpeg_bin_dir.ptr;
    }

    /// Ports `check_antiprompt` (mtmd-cli.cpp:194 @b1b770d90).
    fn checkAntiprompt(self: *const CliContext, generated_tokens: []const llama.impl.llama_c.llama_token) bool {
        if (self.antiprompt_tokens.len == 0 or generated_tokens.len < self.antiprompt_tokens.len) return false;
        return std.mem.eql(llama.impl.llama_c.llama_token, generated_tokens[generated_tokens.len - self.antiprompt_tokens.len ..], self.antiprompt_tokens);
    }

    /// Ports `load_media` (mtmd-cli.cpp:205 @b1b770d90).
    fn loadMedia(self: *CliContext, fname: []const u8) !bool {
        const f = try self.a.dupeSentinel(u8, fname, 0);
        const res = mtmd.helper.mtmd_helper_bitmap_init_from_file(self.ctx_vision.?, f.ptr, false, self.init_opt);
        const bm = res.bitmap orelse return false;
        try self.bitmaps.append(self.a, bm);
        return true;
    }
};

/// Ports `generate_response` (mtmd-cli.cpp:218 @b1b770d90).
fn generateResponse(ctx: *CliContext, n_predict: c_int) !c_int {
    const a = ctx.a;
    var generated_tokens: std.ArrayList(llama.impl.llama_c.llama_token) = .empty;
    var i: c_int = 0;
    while (i < n_predict) : (i += 1) {
        if (i > n_predict or !isGenerating() or isInterrupted()) {
            tools_common.log.out("\n", .{});
            break;
        }

        const token_id = try ctx.smpl.sample(ctx.lctx, -1, false);
        try generated_tokens.append(a, token_id);
        ctx.smpl.accept(token_id, true);

        if (llama.api.vocab.llama_vocab_is_eog(ctx.vocab, token_id) or ctx.checkAntiprompt(generated_tokens.items)) {
            tools_common.log.out("\n", .{});
            break; // end of generation
        }

        tools_common.log.out("%s", .{(try tools_common.common.tokenToPiece(a, ctx.vocab, token_id, true)).ptr});
        stdlib.stdio.stdout().flush();

        if (isInterrupted()) {
            tools_common.log.out("\n", .{});
            break;
        }

        // eval the token
        ctx.batch.clear();
        _ = ctx.batch.add(token_id, ctx.n_past, 0, true);
        ctx.n_past += 1;
        if (llama.api.context.llama_process(ctx.lctx, llama.impl.llama_c.LLAMA_PROCESS_TYPE_DECODE, ctx.batch.get()) != 0) {
            tools_common.log.err("failed to decode token\n", .{});
            return 1;
        }
    }

    const generated_text = try tools_common.common.detokenize(a, ctx.vocab, generated_tokens.items, true);
    try ctx.chat_history.append(a, .{ .role = "assistant", .content = generated_text });

    return 0;
}

/// Ports `run_audio_stream` (mtmd-cli.cpp:262 @b1b770d90): feeds an audio file
/// to a realtime model chunk by chunk, printing the transcript as it is
/// produced.
fn runAudioStream(ctx: *CliContext, fname_in: []const u8, n_predict: c_int) !c_int {
    const a = ctx.a;
    const func = "run_audio_stream";
    const fname = try a.dupeSentinel(u8, fname_in, 0);
    const res = mtmd.helper.mtmd_helper_bitmap_init_from_file(ctx.ctx_vision.?, fname.ptr, false, ctx.init_opt);
    const bmp = res.bitmap orelse return 1; // error is already printed by libmtmd
    defer mtmd.mtmd.mtmd_bitmap_free(bmp);
    if (!mtmd.mtmd.mtmd_bitmap_is_audio(bmp)) {
        tools_common.log.err("%s: %s is not an audio file\n", .{ func, fname.ptr });
        return 1;
    }
    const samples: [*]const f32 = @ptrCast(@alignCast(mtmd.mtmd.mtmd_bitmap_get_data(bmp)));
    const n_samples: usize = mtmd.mtmd.mtmd_bitmap_get_n_bytes(bmp) / @sizeOf(f32);
    const rate = mtmd.mtmd.mtmd_get_audio_sample_rate(ctx.ctx_vision.?);

    const sparams = mtmd.mtmd.mtmd_audio_stream_params_default();
    const stream = mtmd.mtmd.mtmd_audio_stream_init(ctx.ctx_vision.?, sparams) orelse return 1;
    defer mtmd.mtmd.mtmd_audio_stream_free(stream);

    const n_chunk: usize = @as(usize, @intCast(rate)) * @as(usize, @intCast(sparams.step_ms)) / 1000;
    const n_embd: usize = @intCast(llama.api.model.llama_model_n_embd_inp(ctx.model));
    const audio_s = @as(f64, @floatFromInt(n_samples)) / @as(f64, @floatFromInt(rate));

    tools_common.log.out("Transcribing %.1f s of audio, fed in chunks of %d ms\n\n", .{ audio_s, sparams.step_ms });

    var n_fed: usize = 0;
    var n_gen: c_int = 0;
    var tok: llama.impl.llama_c.llama_token = -1; // LLAMA_TOKEN_NULL
    const t_start = llama.impl.llama_c.ggml_time_us();
    var t_enc: i64 = 0; // time spent in the stream (mel + encoder)
    var t_dec: i64 = 0; // time spent in the decoder
    while (!isInterrupted()) {
        const t0 = llama.impl.llama_c.ggml_time_us();
        const n = mtmd.mtmd.mtmd_audio_stream_step(stream, tok);
        t_enc += llama.impl.llama_c.ggml_time_us() - t0;
        if (n < 0) {
            if (n == -2) {
                tools_common.log.err("%s: stream error\n", .{func});
                return 1;
            }
            break; // end of stream
        }
        if (n == 0) {
            if (n_fed < n_samples) {
                const cnt = @min(n_chunk, n_samples - n_fed);
                _ = mtmd.mtmd.mtmd_audio_stream_push(stream, samples + n_fed, cnt);
                n_fed += cnt;
            } else if (mtmd.mtmd.mtmd_audio_stream_flush(stream, true) != 0) {
                break; // already ended
            }
            continue;
        }

        const t1 = llama.impl.llama_c.ggml_time_us();
        const rows: [*]const f32 = mtmd.mtmd.mtmd_audio_stream_get_output_embd(stream).?;
        ctx.batch.clear();
        var i: i32 = 0;
        while (i < n) : (i += 1) {
            var pos = [4]llama.impl.llama_c.llama_pos{ ctx.n_past, ctx.n_past, ctx.n_past, 0 };
            ctx.n_past += 1;
            _ = ctx.batch.addEmbd(.{ .data = rows + @as(usize, @intCast(i)) * n_embd, .n_rows = 1, .n_embd = n_embd }, &pos, 0, i == n - 1);
        }
        if (llama.api.context.llama_process(ctx.lctx, llama.impl.llama_c.LLAMA_PROCESS_TYPE_DECODE, ctx.batch.get()) != 0) {
            tools_common.log.err("%s: failed to decode\n", .{func});
            return 1;
        }

        tok = try ctx.smpl.sample(ctx.lctx, -1, false);
        ctx.smpl.accept(tok, true);
        t_dec += llama.impl.llama_c.ggml_time_us() - t1;
        if (llama.api.vocab.llama_vocab_is_eog(ctx.vocab, tok)) break;
        // the model emits pad and word-boundary tokens between words
        if (!llama.api.vocab.llama_vocab_is_control(ctx.vocab, tok)) {
            tools_common.log.out("%s", .{(try tools_common.common.tokenToPiece(a, ctx.vocab, tok, true)).ptr});
            stdlib.stdio.stdout().flush();
        }
        n_gen += 1;
        if (n_gen >= n_predict) break;
    }

    const t_total = @as(f64, @floatFromInt(llama.impl.llama_c.ggml_time_us() - t_start)) / 1e6;
    tools_common.log.out("\n\n%s: %d tokens, %.2f s audio in %.2f s (%.2fx realtime)\n", .{ func, n_gen, audio_s, t_total, audio_s / t_total });
    tools_common.log.out("%s: encoder %.2f s, decoder %.2f s, %.1f ms per 80 ms step\n", .{ func, @as(f64, @floatFromInt(t_enc)) / 1e6, @as(f64, @floatFromInt(t_dec)) / 1e6, @as(f64, @floatFromInt(t_enc + t_dec)) / 1e3 / @as(f64, @floatFromInt(@max(n_gen, 1))) });
    llama.api.context.llama_perf_context_print(ctx.lctx);

    return 0;
}

/// Ports `chat_add_and_format` (mtmd-cli.cpp:354 @b1b770d90).
fn chatAddAndFormat(ctx: *CliContext, new_msg: common_chat.Msg) ![]const u8 {
    const a = ctx.a;
    tools_common.log.dbg("chat_add_and_format: new_msg.role='%s', new_msg.content='%s'\n", .{ (try a.dupeSentinel(u8, new_msg.role, 0)).ptr, (try a.dupeSentinel(u8, new_msg.content, 0)).ptr });
    const formatted = try common_chat.templates.formatSingle(a, &ctx.tmpls, ctx.chat_history.items, new_msg, std.mem.eql(u8, new_msg.role, "user"), ctx.use_jinja);
    try ctx.chat_history.append(a, new_msg);
    return formatted;
}

/// Ports `eval_message` (mtmd-cli.cpp:364 @b1b770d90).
fn evalMessage(ctx: *CliContext, msg: common_chat.Msg) !c_int {
    const a = ctx.a;
    injectTestResponseMarker();

    const add_bos = ctx.chat_history.items.len == 0;
    const formatted_chat = try chatAddAndFormat(ctx, msg);
    const fmt_prompt = "formatted_chat" ++ ".prompt: %s\n";
    tools_common.log.dbg(fmt_prompt, .{(try a.dupeSentinel(u8, formatted_chat, 0)).ptr});

    if (isInterrupted()) return 0;

    // note: we replace the marker here instead of letting mtmd_tokenize() to do that
    //       because we want to demonstrate how to use mtmd_tokenize_from_parts()

    // split the formatted chat on the media marker to get text segments
    const marker = std.mem.span(mtmd.mtmd.mtmd_default_marker());
    var segments: std.ArrayList([]const u8) = .empty;
    var start: usize = 0;
    while (std.mem.indexOfPos(u8, formatted_chat, start, marker)) |pos| {
        try segments.append(a, formatted_chat[start..pos]);
        start = pos + marker.len;
    }
    try segments.append(a, formatted_chat[start..]);

    const bitmaps_c_ptr = ctx.bitmaps.items;
    if (segments.items.len - 1 != bitmaps_c_ptr.len) {
        tools_common.log.err("Number of media markers (%zu) does not match number of loaded media (%zu)\n", .{ segments.items.len - 1, bitmaps_c_ptr.len });
        return 1;
    }

    // interleave text and media parts
    const texts = try a.alloc(mtmd.mtmd.InputText, segments.items.len);
    var parts: std.ArrayList(mtmd.mtmd.InputPart) = .empty;
    for (segments.items, 0..) |seg, k| {
        texts[k] = .{ .text = seg.ptr, .text_len = seg.len, .add_special = false, .parse_special = true };
        try parts.append(a, .{ .text = &texts[k], .bitmap = null });
        if (k < bitmaps_c_ptr.len) {
            try parts.append(a, .{ .text = null, .bitmap = bitmaps_c_ptr[k] });
        }
    }
    var parts_ptr: std.ArrayList(*const mtmd.mtmd.InputPart) = .empty;
    for (parts.items) |*p| try parts_ptr.append(a, p);

    const chunks = mtmd.mtmd.mtmd_input_chunks_init();
    defer mtmd.mtmd.mtmd_input_chunks_free(chunks);
    // mbatch must not outlive the chunks it references
    var mbatch: ?*mtmd.mtmd.Batch = null;
    defer if (mbatch) |mb| mtmd.mtmd.mtmd_batch_free(mb);
    var res = mtmd.mtmd.mtmd_tokenize_from_parts(ctx.ctx_vision.?, chunks, parts_ptr.items.ptr, parts_ptr.items.len, add_bos);
    if (res != 0) {
        tools_common.log.err("Unable to tokenize prompt, res = %d\n", .{res});
        return 1;
    }

    for (ctx.bitmaps.items) |b| mtmd.mtmd.mtmd_bitmap_free(b);
    ctx.bitmaps.clearRetainingCapacity();

    // batch encode all media chunks, then decode each
    const n_chunks = mtmd.mtmd.mtmd_input_chunks_size(chunks);
    var i: usize = 0;
    while (i < n_chunks) : (i += 1) {
        const chunk = mtmd.mtmd.mtmd_input_chunks_get(chunks, i);
        const chunk_type = mtmd.mtmd.mtmd_input_chunk_get_type(chunk.?);

        if (chunk_type == mtmd.mtmd.ChunkType.text) {
            // decode text chunk
            var new_n_past = ctx.n_past;
            res = mtmd.helper.mtmd_helper_eval_chunk_single(ctx.ctx_vision.?, ctx.lctx, chunk.?, ctx.n_past, 0, ctx.n_batch, i == n_chunks - 1, &new_n_past);
            if (res != 0) {
                tools_common.log.err("Unable to eval text chunk %zu\n", .{i});
                return 1;
            }
            ctx.n_past = new_n_past;
        } else {
            // media chunk: try to get embd from existing batch, or create a new batch
            var embd: ?[*]f32 = null;
            if (mbatch) |mb| {
                embd = mtmd.mtmd.mtmd_batch_get_output_embd(mb, chunk);
                if (embd != null) {
                    tools_common.log.dbg("found embd for media chunk %zu in existing batch\n", .{i});
                } else {
                    tools_common.log.dbg("media chunk %zu not found in existing batch, creating new batch\n", .{i});
                }
            }

            if (embd == null) {
                // create and encode a new batch with as many media chunks as possible
                if (mbatch) |mb| mtmd.mtmd.mtmd_batch_free(mb);
                mbatch = mtmd.mtmd.mtmd_batch_init(ctx.ctx_vision.?);
                res = mtmd.mtmd.mtmd_batch_add_chunk(mbatch.?, chunk.?);
                if (res != 0) _ = llama.impl.llama_c.ggml_abort("mtmd-cli.cpp", 464, "GGML_ASSERT(%s) failed", "res == 0"); // first chunk must always succeed

                var n_added: c_int = 1;
                // add as many subsequent media chunks as possible
                var j = i + 1;
                while (j < n_chunks) : (j += 1) {
                    const next_chunk = mtmd.mtmd.mtmd_input_chunks_get(chunks, j);
                    const next_type = mtmd.mtmd.mtmd_input_chunk_get_type(next_chunk.?);
                    if (next_type == mtmd.mtmd.ChunkType.text) break; // text chunk splits the batch
                    res = mtmd.mtmd.mtmd_batch_add_chunk(mbatch.?, next_chunk.?);
                    if (res != 0) break; // batch full or incompatible
                    n_added += 1;
                }

                const time_start = llama.impl.llama_c.ggml_time_ms();
                tools_common.log.inf("encoding mtmd batch, n_chunks = %d (done = %zu, total = %zu)\n", .{ n_added, i, n_chunks });
                res = mtmd.mtmd.mtmd_batch_encode(mbatch.?);
                if (res != 0) {
                    tools_common.log.err("Failed to encode mtmd batch, res = %d\n", .{res});
                    return 1;
                }
                tools_common.log.inf("mtmd batch encoding done in %d ms\n", .{@as(c_int, @truncate(llama.impl.llama_c.ggml_time_ms() - time_start))});

                embd = mtmd.mtmd.mtmd_batch_get_output_embd(mbatch.?, chunk);
            }

            if (embd == null) _ = llama.impl.llama_c.ggml_abort("mtmd-cli.cpp", 493, "GGML_ASSERT(%s) failed", "embd != nullptr");

            var new_n_past = ctx.n_past;
            res = mtmd.helper.mtmd_helper_decode_image_chunk(ctx.ctx_vision.?, ctx.lctx, chunk.?, embd, ctx.n_past, 0, ctx.n_batch, &new_n_past, null, null);
            if (res != 0) {
                tools_common.log.err("Unable to decode media chunk %zu\n", .{i});
                return 1;
            }
            ctx.n_past = new_n_past;
        }
    }

    tools_common.log.out("\n", .{});

    return 0;
}

fn consoleCleanup() callconv(.c) void {
    tools_common.console.cleanup();
}

/// `common/chat.cpp`'s `LOG_DBG`, `LOG_INF`, `LOG_WRN` and `LOG_ERR`, which
/// `cli/common_chat/log.zig` hands its host as finished lines: into
/// `common_log` at the macro's level, as the server wires them.
fn chatLogSink(level: common_chat.log.Level, text: []const u8) void {
    var buf: [4096]u8 = undefined;
    const z = std.fmt.bufPrintSentinel(&buf, "{s}", .{text}, 0) catch return chatLogSinkHeap(level, text);
    chatLog(level, z.ptr);
}

fn chatLogSinkHeap(level: common_chat.log.Level, text: []const u8) void {
    const z = std.heap.c_allocator.dupeSentinel(u8, text, 0) catch return;
    defer std.heap.c_allocator.free(z);
    chatLog(level, z.ptr);
}

fn chatLog(level: common_chat.log.Level, z: [*:0]const u8) void {
    switch (level) {
        .debug => tools_common.log.dbg("%s", .{z}),
        .info => tools_common.log.inf("%s", .{z}),
        .warn => tools_common.log.wrn("%s", .{z}),
        .err => tools_common.log.err("%s", .{z}),
    }
}

/// `chat.h`'s message helpers' `LOG_WRN` and `LOG_ERR` (`msg.zig`'s own sink).
fn chatMsgLogSink(level: common_chat.msg.Level, text: []const u8) void {
    chatLogSink(if (level == .warn) .warn else .err, text);
}

/// Whether `LOG_DBG` prints at the current verbosity.
fn chatDebugOn() bool {
    return tools_common.log.level_debug <= tools_common.log.verbosity_thold;
}

/// `common_chat_verify_template(tmpl, true)`, for `--chat-template` at parse time.
fn verifyJinjaTemplate(_: std.mem.Allocator, tmpl: []const u8) anyerror!bool {
    return common_chat.templates.verifyTemplate(std.heap.c_allocator, tmpl, true);
}

fn run(init: std.process.Init, a: std.mem.Allocator, argv: []const [:0]const u8) !u8 {
    _ = stdlib.libc.setlocale(stdlib.libc.LC.NUMERIC, "C");

    // cli/common_chat into common_log, as upstream's common/chat.cpp logs
    common_chat.log.sink = chatLogSink;
    common_chat.log.debug_on = chatDebugOn;
    common_chat.msg.log_sink = chatMsgLogSink;
    tools_common.arg.verify_jinja_template = verifyJinjaTemplate;

    llama.impl.llama_c.ggml_time_init();

    var params = try tools_common.params.Params.init(a);

    tools_common.common.init();

    if (!try tools_common.arg.parse(argv, &params, .mtmd, showAdditionalInfo)) return 1;

    mtmd.helper.mtmd_helper_log_set(tools_common.common.defaultLogCallback, null);

    if (params.mmproj.path.len == 0) {
        showAdditionalInfo(argv);
        tools_common.log.err("ERR: Missing --mmproj argument\n", .{});
        return 1;
    }

    llama.impl.llama_c.ggml_backend_load_all();

    var ctx: CliContext = undefined;
    try ctx.init(init.gpa, a, &params);
    defer ctx.deinit();
    tools_common.log.inf("%s: loading model: %s\n", .{ "main", (try a.dupeSentinel(u8, params.model.path, 0)).ptr });

    const is_single_turn = params.prompt.len != 0 and params.image.items.len != 0;

    const n_predict: c_int = if (params.n_predict < 0) std.math.maxInt(c_int) else params.n_predict;

    tools_common.console.init(params.simple_io, params.use_color);
    _ = stdlib.libc.atexit(consoleCleanup);

    // Ctrl+C handling
    if (comptime builtin.os.tag != .windows) {
        var sigint_action: std.posix.Sigaction = .{
            .handler = .{ .handler = sigintHandler },
            .mask = std.posix.sigemptyset(),
            .flags = 0,
        };
        std.posix.sigaction(std.posix.SIG.INT, &sigint_action, null);
    } else {
        _ = win32.SetConsoleCtrlHandler(consoleCtrlHandler, 1);
    }

    if (isInterrupted()) return 130;

    // realtime transcription models (no chat) transcribe while they listen, the others answer after the audio
    const transcribes = ctx.mtmd_cap.audio_stream and !ctx.mtmd_cap.chat;

    if (transcribes) {
        if (params.image.items.len != 1) {
            tools_common.log.err("This model transcribes audio in realtime, pass exactly one --audio file\n", .{});
            return 1;
        }
        return @intCast(try runAudioStream(&ctx, params.image.items[0], n_predict));
    }

    const System = struct {
        /// The `eval_system_prompt_if_present` lambda (mtmd-cli.cpp:582
        /// @b1b770d90).
        fn evalIfPresent(cx: *CliContext, p: *const tools_common.params.Params) !c_int {
            if (p.system_prompt.len == 0) return 0;
            return evalMessage(cx, .{ .role = "system", .content = p.system_prompt });
        }
    };

    tools_common.log.wrn("WARN: This is an experimental CLI for testing multimodal capability.\n", .{});
    tools_common.log.wrn("      For normal use cases, please use the standard llama-cli\n", .{});

    if (try System.evalIfPresent(&ctx, &params) != 0) return 1;

    if (is_single_turn) {
        setGenerating(true);
        const marker = std.mem.span(mtmd.mtmd.mtmd_default_marker());
        if (std.mem.indexOf(u8, params.prompt, marker) == null) {
            for (params.image.items) |_| {
                // most models require the marker before each image
                // ref: https://github.com/ggml-org/llama.cpp/pull/17616
                params.prompt = try std.mem.concat(a, u8, &.{ marker, params.prompt });
            }
        }

        const msg: common_chat.Msg = .{ .role = "user", .content = params.prompt };
        for (params.image.items) |image| {
            if (!try ctx.loadMedia(image)) return 1; // error is already printed by libmtmd
        }
        if (try evalMessage(&ctx, msg) != 0) return 1;
        if (!isInterrupted() and try generateResponse(&ctx, n_predict) != 0) return 1;
    } else {
        tools_common.log.out("\n Running in chat mode, available commands:", .{});
        if (mtmd.mtmd.mtmd_support_vision(ctx.ctx_vision.?)) tools_common.log.out("\n   /image <path>    load an image", .{});
        if (mtmd.mtmd.mtmd_support_audio(ctx.ctx_vision.?)) tools_common.log.out("\n   /audio <path>    load an audio", .{});
        if (ctx.mtmd_cap.video) tools_common.log.out("\n   /video <path>    load a video", .{});
        tools_common.log.out("\n   /clear           clear the chat history", .{});
        tools_common.log.out("\n   /quit or /exit   exit the program", .{});
        tools_common.log.out("\n", .{});

        var content: std.ArrayList(u8) = .empty;
        var line_buf: std.ArrayList(u8) = .empty;

        while (!isInterrupted()) {
            setGenerating(false);
            tools_common.log.out("\n> ", .{});
            tools_common.console.setDisplay(.user_input);
            _ = try tools_common.console.readline(a, &line_buf, false);
            if (isInterrupted()) break;
            tools_common.console.setDisplay(.reset);
            // a submitted line always ends with a newline, an empty read is EOF
            if (line_buf.items.len == 0) {
                break;
            }
            const line = stdlib.strings.strip(line_buf.items);
            if (line.len == 0) continue;
            if (std.mem.eql(u8, line, "/quit") or std.mem.eql(u8, line, "/exit")) break;
            if (std.mem.eql(u8, line, "/clear")) {
                ctx.n_past = 0;
                ctx.chat_history.clearRetainingCapacity();
                llama.api.context.llama_memory_clear(llama.api.context.llama_get_memory(ctx.lctx), true);
                if (try System.evalIfPresent(&ctx, &params) != 0) return 1;
                tools_common.log.out("Chat history cleared\n\n", .{});
                continue;
            }
            setGenerating(true);
            const is_image = std.mem.eql(u8, line, "/image") or std.mem.startsWith(u8, line, "/image ");
            const is_audio = std.mem.eql(u8, line, "/audio") or std.mem.startsWith(u8, line, "/audio ");
            const is_video = std.mem.eql(u8, line, "/video") or std.mem.startsWith(u8, line, "/video ");
            if (is_image or is_audio or is_video) {
                if (line.len < 8) {
                    tools_common.log.err("ERR: Missing media filename\n", .{});
                    continue;
                }
                const media_path = line[7..];
                if (try ctx.loadMedia(media_path)) {
                    const kind: [*:0]const u8 = if (is_image) "image" else if (is_audio) "audio" else "video";
                    tools_common.log.out("%s %s loaded\n", .{ (try a.dupeSentinel(u8, media_path, 0)).ptr, kind });
                    try content.appendSlice(a, marker_);
                }
                // else, error is already printed by libmtmd
                continue;
            } else {
                try content.appendSlice(a, line);
            }
            const msg: common_chat.Msg = .{ .role = "user", .content = try a.dupe(u8, content.items) };
            const ret = try evalMessage(&ctx, msg);
            if (ret != 0) return 1;
            if (isInterrupted()) break;
            if (try generateResponse(&ctx, n_predict) != 0) return 1;
            content.clearRetainingCapacity();
        }
    }
    if (isInterrupted()) tools_common.log.out("\nInterrupted by user\n", .{});
    tools_common.log.out("\n\n", .{});
    llama.api.context.llama_perf_context_print(ctx.lctx);
    return if (isInterrupted()) 130 else 0;
}

/// `mtmd_default_marker()`, which the conversation appends per media file.
const marker_ = "<__media__>";

const win32 = struct {
    extern "kernel32" fn SetConsoleCtrlHandler(handler: ?*const fn (u32) callconv(.winapi) c_int, add: c_int) callconv(.winapi) c_int;
};

/// The Windows arm of the Ctrl+C handling, `console_ctrl_handler`
/// (mtmd-cli.cpp:562 @b1b770d90).
fn consoleCtrlHandler(ctrl_type: u32) callconv(.winapi) c_int {
    const ctrl_c_event: u32 = 0;
    if (ctrl_type == ctrl_c_event) {
        sigintHandler(std.posix.SIG.INT);
        return 1;
    }
    return 0;
}

/// Ports `main` (mtmd-cli.cpp:520 @b1b770d90).
pub fn main(init: std.process.Init) !u8 {
    stdlib.stdio.bufferAsC();
    stdlib.env.set(init.environ_map);
    defer stdlib.stdio.flushAll();
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return run(init, arena, argv);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
