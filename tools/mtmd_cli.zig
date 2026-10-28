//! `llama-mtmd-cli`: a model with an mmproj, given images or audio, as one
//! turn (`-p` with `--image`) or a conversation.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/mtmd/mtmd-cli.cpp` at v0.4.0 (`5266f24da`).
//! Each declaration names the C++ it replaces and the line it began at.
//!
//! # What differs
//!
//! - `MTMD_DEBUG_GRAPH` installs upstream's `common_debug_cb_eval`, from
//!   `common/debug.cpp`, which is not ported: with it set this refuses to
//!   run rather than silently not printing the graph.
//! - `--jinja`: `tools/common/chat.zig` ports the legacy chat-template path,
//!   which is this tool's default; the Jinja applier is not ported.
//! - Video: the reference is built without `MTMD_VIDEO`, so there are no
//!   video helpers on either side.

const std = @import("std");
const builtin = @import("builtin");
const tc = @import("tools_common");
const c = tc.c;
const log = tc.log;
const arg = tc.arg;
const console = tc.console;
const chat = tc.chat;

comptime {
    // libllama and libmtmd, compiled in (see `build/tools.zig`).
    _ = @import("llama");
    _ = @import("mtmd");
}

extern fn setlocale(cat: c_int, locale: [*:0]const u8) ?[*:0]const u8;
extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;
extern fn atexit(f: *const fn () callconv(.c) void) c_int;
extern fn _exit(code: c_int) noreturn;

const lc_numeric: c_int = if (builtin.os.tag.isDarwin()) 4 else if (builtin.os.tag == .windows) 4 else 1;

// Ports `g_is_generating`, `g_is_interrupted` (mtmd-cli.cpp:31, 32
// @5266f24da): volatile, because of signal being an interrupt
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

/// Ports `show_additional_info` (mtmd-cli.cpp:40 @5266f24da).
fn showAdditionalInfo(argv: []const [:0]const u8) void {
    log.out("Experimental CLI for multimodal\n\n" ++
        "Usage: %s [options] -m <model> --mmproj <mmproj> --image <image> --audio <audio> -p <prompt>\n\n" ++
        "  -m and --mmproj are required\n" ++
        "  -hf user/repo can replace both -m and --mmproj in most cases\n" ++
        "  --image, --audio and -p are optional, if NOT provided, the CLI will run in chat mode\n" ++
        "  to disable using GPU for mmproj model, add --no-mmproj-offload\n", .{argv[0].ptr});
}

/// Ports `sigint_handler` (mtmd-cli.cpp:53 @5266f24da).
fn sigintHandler(signo: std.posix.SIG) callconv(.c) void {
    if (signo == std.posix.SIG.INT) {
        if (isGenerating()) {
            setGenerating(false);
        } else {
            console.cleanup();
            if (isInterrupted()) _exit(1);
            @as(*volatile bool, &g_is_interrupted_).* = true;
        }
    }
}

/// Ports `inject_test_response_marker` (mtmd-cli.cpp:69 @5266f24da): only
/// used by upstream's tests.sh to capture the response.
fn injectTestResponseMarker() void {
    if (getenv("MTMD_TEST_RESPONSE_MARKER")) |env| log.out("%s\n", .{env});
}

/// Ports `mtmd_cli_context` (mtmd-cli.cpp:76 @5266f24da).
const CliContext = struct {
    a: std.mem.Allocator,
    ctx_vision: ?*c.mtmd_context = null,
    llama_init: tc.init.InitResult,

    model: ?*c.llama_model,
    lctx: ?*c.llama_context,
    vocab: ?*const c.llama_vocab,
    smpl: *tc.sampling.Sampler,
    batch: c.llama_batch,
    n_batch: c_int,

    bitmaps: std.ArrayList(*c.mtmd_bitmap) = .empty,

    init_opt: c.mtmd_helper_init_opt = undefined, // mtmd_helper_init_opt_default(), set in `init`
    video_ffmpeg_bin_dir: [:0]const u8 = "",

    mbatch: ?*c.mtmd_batch = null,

    // chat template
    tmpls: chat.Templates = undefined,
    chat_history: std.ArrayList(chat.Msg) = .empty,
    use_jinja: bool = false,

    // support for legacy templates (models not having EOT token)
    antiprompt_tokens: []const c.llama_token = &.{},

    n_threads: c_int = 1,
    n_past: c.llama_pos = 0,

    /// Ports the constructor `mtmd_cli_context` (mtmd-cli.cpp:109 @5266f24da). Every failure is the
    /// C++'s `exit(1)`, which runs no destructor.
    fn init(self: *CliContext, gpa: std.mem.Allocator, a: std.mem.Allocator, params: *tc.params.Params) !void {
        const llama_init = try tc.init.initFromParams(gpa, params, false);
        const model = llama_init.model;
        if (model == null or llama_init.context == null) tc.cExit(1);
        self.* = .{
            .a = a,
            .llama_init = llama_init,
            .model = model,
            .lctx = llama_init.context,
            .vocab = c.llama_model_get_vocab(model),
            .smpl = samplerOrTerminate(params, model),
            .batch = undefined,
            .n_batch = params.n_batch,
            .init_opt = c.mtmd_helper_init_opt_default(),
        };
        self.n_threads = params.cpuparams.n_threads;
        self.batch = c.llama_batch_init(1, 0, 1); // batch for next token generation

        self.initVisionContext(params);

        if (!c.mtmd_helper_model_can_chat(self.lctx, self.ctx_vision)) {
            log.err("Model does not support chat mode\n", .{});
            log.err("Hint: for TTS models, please use llama-tts\n", .{});
            tc.cExit(1);
        }

        if (c.llama_model_chat_template(model, null) == null and params.chat_template.len == 0) {
            log.err("Model does not have chat template.\n", .{});
            log.err("  For old llava models, you may need to use '--chat-template vicuna'\n", .{});
            log.err("  For MobileVLM models, use '--chat-template deepseek'\n", .{});
            log.err("  For Mistral Small 3.1, use '--chat-template mistral-v7'\n", .{});
            tc.cExit(1);
        }

        self.tmpls = try chat.Templates.init(a, model, params.chat_template);
        self.use_jinja = params.use_jinja;
        self.chat_history.clearRetainingCapacity();
        const example = try a.dupeSentinel(u8, try self.tmpls.formatExample(a, params.use_jinja), 0);
        log.inf("%s: chat template example:\n%s\n", .{ "mtmd_cli_context", example.ptr });

        // load antiprompt tokens for legacy templates
        if (std.mem.eql(u8, params.chat_template, "vicuna")) {
            self.antiprompt_tokens = try tc.common.tokenize(a, self.vocab, "ASSISTANT:", false, true);
        } else if (std.mem.eql(u8, params.chat_template, "deepseek")) {
            self.antiprompt_tokens = try tc.common.tokenize(a, self.vocab, "###", false, true);
        }
    }

    /// `common_sampler_init`, whose exceptions escape `main`.
    fn samplerOrTerminate(params: *tc.params.Params, model: ?*c.llama_model) *tc.sampling.Sampler {
        return tc.init.samplerInit(params, model) catch |e| {
            const kind: [*:0]const u8 = if (e == error.InvalidArgument) "std::invalid_argument" else "std::runtime_error";
            tc.io.eprintf("libc++abi: terminating due to uncaught exception of type %s: %.*s\n", .{ kind, @as(c_int, @intCast(arg.what().len)), arg.what().ptr });
            std.process.abort();
        };
    }

    /// Ports the destructor of `mtmd_cli_context` (mtmd-cli.cpp:150 @5266f24da), then the members in
    /// reverse order of declaration.
    fn deinit(self: *CliContext) void {
        c.llama_batch_free(self.batch);
        self.smpl.destroy();
        if (self.mbatch) |b| c.mtmd_batch_free(b);
        for (self.bitmaps.items) |b| c.mtmd_bitmap_free(b);
        self.llama_init.deinit();
        if (self.ctx_vision) |v| c.mtmd_free(v);
    }

    /// Ports `init_vision_context` (mtmd-cli.cpp:155 @5266f24da).
    fn initVisionContext(self: *CliContext, params: *tc.params.Params) void {
        const clip_path = self.a.dupeSentinel(u8, params.mmproj.path, 0) catch tc.cExit(1);
        var mparams = c.mtmd_context_params_default();
        mparams.use_gpu = params.mmproj_use_gpu;
        mparams.device = params.mmproj_device;
        mparams.print_timings = true;
        mparams.n_threads = params.cpuparams.n_threads;
        mparams.flash_attn_type = params.flash_attn_type;
        mparams.warmup = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;
        if (getenv("MTMD_DEBUG_GRAPH") != null) {
            log.err("MTMD_DEBUG_GRAPH needs common_debug_cb_eval, which this port does not have\n", .{});
            tc.cExit(1);
        }
        self.ctx_vision = c.mtmd_init_from_file(clip_path.ptr, self.model, mparams);
        if (self.ctx_vision == null) {
            log.err("Failed to load vision model from %s\n", .{clip_path.ptr});
            tc.cExit(1);
        }

        self.video_ffmpeg_bin_dir = self.a.dupeSentinel(u8, params.video_ffmpeg_bin_dir, 0) catch tc.cExit(1);
        self.init_opt.video_params.fps_target = params.video_fps;
        self.init_opt.video_params.timestamp_interval_ms = params.video_timestamp_interval_ms;
        self.init_opt.video_params.ffmpeg_bin_dir = if (self.video_ffmpeg_bin_dir.len == 0) null else self.video_ffmpeg_bin_dir.ptr;
    }

    /// Ports `check_antiprompt` (mtmd-cli.cpp:183 @5266f24da).
    fn checkAntiprompt(self: *const CliContext, generated_tokens: []const c.llama_token) bool {
        if (self.antiprompt_tokens.len == 0 or generated_tokens.len < self.antiprompt_tokens.len) return false;
        return std.mem.eql(c.llama_token, generated_tokens[generated_tokens.len - self.antiprompt_tokens.len ..], self.antiprompt_tokens);
    }

    /// Ports `load_media` (mtmd-cli.cpp:194 @5266f24da).
    fn loadMedia(self: *CliContext, fname: []const u8) !bool {
        const f = try self.a.dupeSentinel(u8, fname, 0);
        const res = c.mtmd_helper_bitmap_init_from_file(self.ctx_vision, f.ptr, false, self.init_opt);
        const bm = res.bitmap orelse return false;
        try self.bitmaps.append(self.a, bm);
        return true;
    }
};

/// Ports `generate_response` (mtmd-cli.cpp:207 @5266f24da).
fn generateResponse(ctx: *CliContext, n_predict: c_int) !c_int {
    const a = ctx.a;
    var generated_tokens: std.ArrayList(c.llama_token) = .empty;
    var i: c_int = 0;
    while (i < n_predict) : (i += 1) {
        if (i > n_predict or !isGenerating() or isInterrupted()) {
            log.out("\n", .{});
            break;
        }

        const token_id = try ctx.smpl.sample(ctx.lctx, -1, false);
        try generated_tokens.append(a, token_id);
        ctx.smpl.accept(token_id, true);

        if (c.llama_vocab_is_eog(ctx.vocab, token_id) or ctx.checkAntiprompt(generated_tokens.items)) {
            log.out("\n", .{});
            break; // end of generation
        }

        log.out("%s", .{(try tc.common.tokenToPiece(a, ctx.vocab, token_id, true)).ptr});
        tc.io.flushStdout();

        if (isInterrupted()) {
            log.out("\n", .{});
            break;
        }

        // eval the token
        tc.common.batchClear(&ctx.batch);
        tc.common.batchAdd(&ctx.batch, token_id, ctx.n_past, &.{0}, true);
        ctx.n_past += 1;
        if (c.llama_decode(ctx.lctx, ctx.batch) != 0) {
            log.err("failed to decode token\n", .{});
            return 1;
        }
    }

    const generated_text = try tc.common.detokenize(a, ctx.vocab, generated_tokens.items, true);
    try ctx.chat_history.append(a, .{ .role = "assistant", .content = generated_text });

    return 0;
}

/// Ports `chat_add_and_format` (mtmd-cli.cpp:250 @5266f24da).
fn chatAddAndFormat(ctx: *CliContext, new_msg: chat.Msg) ![]const u8 {
    const a = ctx.a;
    log.dbg("chat_add_and_format: new_msg.role='%s', new_msg.content='%s'\n", .{ (try a.dupeSentinel(u8, new_msg.role, 0)).ptr, (try a.dupeSentinel(u8, new_msg.content, 0)).ptr });
    const formatted = try ctx.tmpls.formatSingle(a, ctx.chat_history.items, new_msg, std.mem.eql(u8, new_msg.role, "user"), ctx.use_jinja);
    try ctx.chat_history.append(a, new_msg);
    return formatted;
}

/// Ports `eval_message` (mtmd-cli.cpp:260 @5266f24da).
fn evalMessage(ctx: *CliContext, msg: chat.Msg) !c_int {
    const a = ctx.a;
    injectTestResponseMarker();

    const add_bos = ctx.chat_history.items.len == 0;
    const formatted_chat = try chatAddAndFormat(ctx, msg);
    const fmt_prompt = "formatted_chat" ++ ".prompt: %s\n";
    log.dbg(fmt_prompt, .{(try a.dupeSentinel(u8, formatted_chat, 0)).ptr});

    if (isInterrupted()) return 0;

    // note: we replace the marker here instead of letting mtmd_tokenize() to do that
    //       because we want to demonstrate how to use mtmd_tokenize_from_parts()

    // split the formatted chat on the media marker to get text segments
    const marker = std.mem.span(c.mtmd_default_marker());
    var segments: std.ArrayList([]const u8) = .empty;
    var start: usize = 0;
    while (std.mem.indexOfPos(u8, formatted_chat, start, marker)) |pos| {
        try segments.append(a, formatted_chat[start..pos]);
        start = pos + marker.len;
    }
    try segments.append(a, formatted_chat[start..]);

    const bitmaps_c_ptr = ctx.bitmaps.items;
    if (segments.items.len - 1 != bitmaps_c_ptr.len) {
        log.err("Number of media markers (%zu) does not match number of loaded media (%zu)\n", .{ segments.items.len - 1, bitmaps_c_ptr.len });
        return 1;
    }

    // interleave text and media parts
    const texts = try a.alloc(c.mtmd_input_text, segments.items.len);
    var parts: std.ArrayList(c.mtmd_input_part) = .empty;
    for (segments.items, 0..) |seg, k| {
        texts[k] = .{ .text = seg.ptr, .text_len = seg.len, .add_special = false, .parse_special = true };
        try parts.append(a, .{ .text = &texts[k], .bitmap = null });
        if (k < bitmaps_c_ptr.len) {
            try parts.append(a, .{ .text = null, .bitmap = bitmaps_c_ptr[k] });
        }
    }
    var parts_ptr: std.ArrayList([*c]const c.mtmd_input_part) = .empty;
    for (parts.items) |*p| try parts_ptr.append(a, p);

    const chunks = c.mtmd_input_chunks_init();
    defer c.mtmd_input_chunks_free(chunks);
    var res = c.mtmd_tokenize_from_parts(ctx.ctx_vision, chunks, parts_ptr.items.ptr, parts_ptr.items.len, add_bos);
    if (res != 0) {
        log.err("Unable to tokenize prompt, res = %d\n", .{res});
        return 1;
    }

    for (ctx.bitmaps.items) |b| c.mtmd_bitmap_free(b);
    ctx.bitmaps.clearRetainingCapacity();

    // batch encode all media chunks, then decode each
    const n_chunks = c.mtmd_input_chunks_size(chunks);
    var i: usize = 0;
    while (i < n_chunks) : (i += 1) {
        const chunk = c.mtmd_input_chunks_get(chunks, i);
        const chunk_type = c.mtmd_input_chunk_get_type(chunk);

        if (chunk_type == c.MTMD_INPUT_CHUNK_TYPE_TEXT) {
            // decode text chunk
            var new_n_past = ctx.n_past;
            res = c.mtmd_helper_eval_chunk_single(ctx.ctx_vision, ctx.lctx, chunk, ctx.n_past, 0, ctx.n_batch, i == n_chunks - 1, &new_n_past);
            if (res != 0) {
                log.err("Unable to eval text chunk %zu\n", .{i});
                return 1;
            }
            ctx.n_past = new_n_past;
        } else {
            // media chunk: try to get embd from existing batch, or create a new batch
            var embd: ?[*]f32 = null;
            if (ctx.mbatch) |mb| {
                embd = c.mtmd_batch_get_output_embd(mb, chunk);
                if (embd != null) {
                    log.dbg("found embd for media chunk %zu in existing batch\n", .{i});
                } else {
                    log.dbg("media chunk %zu not found in existing batch, creating new batch\n", .{i});
                }
            }

            if (embd == null) {
                // create and encode a new batch with as many media chunks as possible
                if (ctx.mbatch) |mb| c.mtmd_batch_free(mb);
                ctx.mbatch = c.mtmd_batch_init(ctx.ctx_vision);
                res = c.mtmd_batch_add_chunk(ctx.mbatch, chunk);
                if (res != 0) _ = c.ggml_abort("mtmd-cli.cpp", 358, "GGML_ASSERT(%s) failed", "res == 0"); // first chunk must always succeed

                var n_added: c_int = 1;
                // add as many subsequent media chunks as possible
                var j = i + 1;
                while (j < n_chunks) : (j += 1) {
                    const next_chunk = c.mtmd_input_chunks_get(chunks, j);
                    const next_type = c.mtmd_input_chunk_get_type(next_chunk);
                    if (next_type == c.MTMD_INPUT_CHUNK_TYPE_TEXT) break; // text chunk splits the batch
                    res = c.mtmd_batch_add_chunk(ctx.mbatch, next_chunk);
                    if (res != 0) break; // batch full or incompatible
                    n_added += 1;
                }

                const time_start = c.ggml_time_ms();
                log.inf("encoding mtmd batch, n_chunks = %d (done = %zu, total = %zu)\n", .{ n_added, i, n_chunks });
                res = c.mtmd_batch_encode(ctx.mbatch);
                if (res != 0) {
                    log.err("Failed to encode mtmd batch, res = %d\n", .{res});
                    return 1;
                }
                log.inf("mtmd batch encoding done in %d ms\n", .{@as(c_int, @truncate(c.ggml_time_ms() - time_start))});

                embd = c.mtmd_batch_get_output_embd(ctx.mbatch, chunk);
            }

            if (embd == null) _ = c.ggml_abort("mtmd-cli.cpp", 387, "GGML_ASSERT(%s) failed", "embd != nullptr");

            var new_n_past = ctx.n_past;
            res = c.mtmd_helper_decode_image_chunk(ctx.ctx_vision, ctx.lctx, chunk, embd, ctx.n_past, 0, ctx.n_batch, &new_n_past, null, null);
            if (res != 0) {
                log.err("Unable to decode media chunk %zu\n", .{i});
                return 1;
            }
            ctx.n_past = new_n_past;
        }
    }

    log.out("\n", .{});

    return 0;
}

fn consoleCleanup() callconv(.c) void {
    console.cleanup();
}

/// `string_strip` (common/common.cpp): leading and trailing whitespace off.
fn stringStrip(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, " \t\n\r\x0b\x0c");
}

fn run(init: std.process.Init, a: std.mem.Allocator, argv: []const [:0]const u8) !u8 {
    _ = setlocale(lc_numeric, "C");

    c.ggml_time_init();

    var params = try tc.params.Params.init(a);

    tc.common.init();

    if (!try arg.parse(argv, &params, .mtmd, showAdditionalInfo)) return 1;

    c.mtmd_helper_log_set(tc.common.defaultLogCallback, null);

    if (params.mmproj.path.len == 0) {
        showAdditionalInfo(argv);
        log.err("ERR: Missing --mmproj argument\n", .{});
        return 1;
    }

    c.ggml_backend_load_all();

    var ctx: CliContext = undefined;
    try ctx.init(init.gpa, a, &params);
    defer ctx.deinit();
    log.inf("%s: loading model: %s\n", .{ "main", (try a.dupeSentinel(u8, params.model.path, 0)).ptr });

    const is_single_turn = params.prompt.len != 0 and params.image.items.len != 0;

    const n_predict: c_int = if (params.n_predict < 0) std.math.maxInt(c_int) else params.n_predict;

    console.init(params.simple_io, params.use_color);
    _ = atexit(consoleCleanup);

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

    const System = struct {
        /// The `eval_system_prompt_if_present` lambda (mtmd-cli.cpp:465
        /// @5266f24da).
        fn evalIfPresent(cx: *CliContext, p: *const tc.params.Params) !c_int {
            if (p.system_prompt.len == 0) return 0;
            return evalMessage(cx, .{ .role = "system", .content = p.system_prompt });
        }
    };

    log.wrn("WARN: This is an experimental CLI for testing multimodal capability.\n", .{});
    log.wrn("      For normal use cases, please use the standard llama-cli\n", .{});

    if (try System.evalIfPresent(&ctx, &params) != 0) return 1;

    if (is_single_turn) {
        setGenerating(true);
        const marker = std.mem.span(c.mtmd_default_marker());
        if (std.mem.indexOf(u8, params.prompt, marker) == null) {
            for (params.image.items) |_| {
                // most models require the marker before each image
                // ref: https://github.com/ggml-org/llama.cpp/pull/17616
                params.prompt = try std.mem.concat(a, u8, &.{ marker, params.prompt });
            }
        }

        const msg: chat.Msg = .{ .role = "user", .content = params.prompt };
        for (params.image.items) |image| {
            if (!try ctx.loadMedia(image)) return 1; // error is already printed by libmtmd
        }
        if (try evalMessage(&ctx, msg) != 0) return 1;
        if (!isInterrupted() and try generateResponse(&ctx, n_predict) != 0) return 1;
    } else {
        log.out("\n Running in chat mode, available commands:", .{});
        if (c.mtmd_support_vision(ctx.ctx_vision)) log.out("\n   /image <path>    load an image", .{});
        if (c.mtmd_support_audio(ctx.ctx_vision)) log.out("\n   /audio <path>    load an audio", .{});
        if (c.mtmd_helper_support_video(ctx.ctx_vision)) log.out("\n   /video <path>    load a video", .{});
        log.out("\n   /clear           clear the chat history", .{});
        log.out("\n   /quit or /exit   exit the program", .{});
        log.out("\n", .{});

        var content: std.ArrayList(u8) = .empty;
        var line_buf: std.ArrayList(u8) = .empty;

        while (!isInterrupted()) {
            setGenerating(false);
            log.out("\n> ", .{});
            console.setDisplay(.user_input);
            _ = try console.readline(a, &line_buf, false);
            if (isInterrupted()) break;
            console.setDisplay(.reset);
            const line = stringStrip(line_buf.items);
            if (line.len == 0) continue;
            if (std.mem.eql(u8, line, "/quit") or std.mem.eql(u8, line, "/exit")) break;
            if (std.mem.eql(u8, line, "/clear")) {
                ctx.n_past = 0;
                ctx.chat_history.clearRetainingCapacity();
                c.llama_memory_clear(c.llama_get_memory(ctx.lctx), true);
                if (try System.evalIfPresent(&ctx, &params) != 0) return 1;
                log.out("Chat history cleared\n\n", .{});
                continue;
            }
            setGenerating(true);
            const is_image = std.mem.eql(u8, line, "/image") or std.mem.startsWith(u8, line, "/image ");
            const is_audio = std.mem.eql(u8, line, "/audio") or std.mem.startsWith(u8, line, "/audio ");
            const is_video = std.mem.eql(u8, line, "/video") or std.mem.startsWith(u8, line, "/video ");
            if (is_image or is_audio or is_video) {
                if (line.len < 8) {
                    log.err("ERR: Missing media filename\n", .{});
                    continue;
                }
                const media_path = line[7..];
                if (try ctx.loadMedia(media_path)) {
                    const kind: [*:0]const u8 = if (is_image) "image" else if (is_audio) "audio" else "video";
                    log.out("%s %s loaded\n", .{ (try a.dupeSentinel(u8, media_path, 0)).ptr, kind });
                    try content.appendSlice(a, marker_);
                }
                // else, error is already printed by libmtmd
                continue;
            } else {
                try content.appendSlice(a, line);
            }
            const msg: chat.Msg = .{ .role = "user", .content = try a.dupe(u8, content.items) };
            const ret = try evalMessage(&ctx, msg);
            if (ret != 0) return 1;
            if (isInterrupted()) break;
            if (try generateResponse(&ctx, n_predict) != 0) return 1;
            content.clearRetainingCapacity();
        }
    }
    if (isInterrupted()) log.out("\nInterrupted by user\n", .{});
    log.out("\n\n", .{});
    c.llama_perf_context_print(ctx.lctx);
    return if (isInterrupted()) 130 else 0;
}

/// `mtmd_default_marker()`, which the conversation appends per media file.
const marker_ = "<__media__>";

const win32 = struct {
    extern "kernel32" fn SetConsoleCtrlHandler(handler: ?*const fn (u32) callconv(.winapi) c_int, add: c_int) callconv(.winapi) c_int;
};

/// The Windows arm of the Ctrl+C handling, `console_ctrl_handler`
/// (mtmd-cli.cpp:456 @5266f24da).
fn consoleCtrlHandler(ctrl_type: u32) callconv(.winapi) c_int {
    const ctrl_c_event: u32 = 0;
    if (ctrl_type == ctrl_c_event) {
        sigintHandler(std.posix.SIG.INT);
        return 1;
    }
    return 0;
}

/// Ports `main` (mtmd-cli.cpp:414 @5266f24da).
pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return run(init, arena, argv);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
