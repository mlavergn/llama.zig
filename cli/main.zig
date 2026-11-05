//! Entry point for `llamazig`.
//!
//! # Provenance
//!
//! **Not a port.** See `cli/module.zig`.
//!
//! Deliberately thin: it unpacks `std.process.Init`, hands off, and turns a
//! failure into an exit code. Everything with behaviour lives in `args.zig` and
//! `session.zig`, where the tests can reach it without spawning a process.

const std = @import("std");
const builtin = @import("builtin");
const cli = @import("module.zig");

/// Log configuration for the executable.
pub const std_options: std.Options = .{
    .log_level = if (builtin.mode == .debug) .debug else .warn,
};

/// Runs the CLI.
///
/// Parameters:
/// - `init`: process capabilities supplied by the runtime (allocator, IO, args).
///
/// Return: nothing on success. Exits non-zero on a bad command line or a
/// failure to generate, having written the reason to stderr -- Decision 22
/// requires a non-zero exit, so failures are not merely reported.
pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);

    var out_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
    const out = &stdout.interface;

    var err_buf: [1024]u8 = undefined;
    var stderr = std.Io.File.stderr().writerStreaming(init.io, &err_buf);
    const err = &stderr.interface;

    const parsed = try cli.args.parse(argv, out);
    switch (parsed) {
        .fail => |f| {
            try f.report(err);
            try err.flush();
            std.process.exit(1);
        },
        .ok => |parsed_settings| {
            if (parsed_settings.done) {
                try out.flush();
                return;
            }
            var settings = parsed_settings;
            // Upstream reads and converts a grammar while it parses the
            // command line, so a bad one fails before anything loads.
            settings.grammar = resolveGrammar(arena, init.io, settings.grammar_source) catch |e| {
                if (e == error.OutOfMemory) return e;
                try err.print("error while handling argument \"{s}\": {s}\n", .{ settings.grammar_source.flag(), grammar_error.items });
                try err.flush();
                std.process.exit(1);
            };
            // `--chat-template-kwargs` is parsed with the command line upstream.
            const kwargs = cli.conversation.resolveKwargs(arena, settings.kwargOps()) catch |e| {
                if (e == error.OutOfMemory) return e;
                try err.print("error while handling argument \"--chat-template-kwargs\": {s}\n", .{cli.conversation.last_error.items});
                try err.flush();
                std.process.exit(1);
            };
            const color = switch (settings.color) {
                .on => true,
                .off => false,
                .auto => cli.conversation.ttyCanUseColors(init.io, init.environ_map),
            };
            // Upstream is interactive by default and this binary is not, so it says so rather than quietly differing.
            if (!settings.one_shot and !settings.interactive) {
                try err.print(
                    "note: this defaults to one-shot completion; upstream's llama-cli would be\n" ++
                        "      interactive here. Pass -cnv for a conversation, or -st to say you meant this.\n",
                    .{},
                );
                try err.flush();
            }
            const trace = init.environ_map.get("LLAMAZIG_CNV_TRACE") != null;
            run(init.gpa, init.io, &settings, .{ .kwargs = kwargs, .color = color, .trace = trace }, err, out) catch |e| {
                try err.print("error: {s}\n", .{describe(e)});
                try err.flush();
                std.process.exit(1);
            };
        },
    }
}

/// What `main` works out beside the command line, before anything loads.
const RunOptions = struct {
    /// The template kwargs, from `conversation.resolveKwargs`.
    kwargs: []const cli.common_chat.templates.KwArg,
    /// Whether the reasoning is greyed, `--color` resolved.
    color: bool,
    /// Whether `LLAMAZIG_CNV_TRACE` is set, sending the context size and GPU
    /// layers `--fit` left, and each turn's cache report
    /// (`Session.turn_trace`), to `err`.
    trace: bool,
};

/// Loads the model and generates.
///
/// Split from `main` so the failure path above stays a single `catch`.
///
/// Parameters:
/// - `gpa`: allocator for the prompt and token buffers.
/// - `io`: for reading a template or system prompt from a file.
/// - `settings`: parsed command line.
/// - `opts`: what `main` resolved beside it.
/// - `err`: where the loading notice goes.
/// - `out`: destination for the completion.
///
/// Return: nothing on success; propagates the stage that failed.
fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    settings: *const cli.Args,
    opts: RunOptions,
    err: *std.Io.Writer,
    out: *std.Io.Writer,
) !void {
    // Quiet by default: libllama logs load progress at INFO, which upstream's
    // llama-cli hides unless asked. Ours does the same, so a completion is the
    // only thing on stdout and the output is usable in a pipeline.
    cli.session.c.llama_log_set(quietLog, null);

    cli.session.c.llama_backend_init();
    defer cli.session.c.llama_backend_free();

    const prompt = if (settings.escape)
        try cli.args.processEscapes(gpa, settings.prompt)
    else
        try gpa.dupe(u8, settings.prompt);
    defer gpa.free(prompt);

    // Loading a multi-gigabyte model and compiling the Metal pipelines can
    // take many seconds, and libllama's progress logging is suppressed above.
    // Without this the binary is completely silent until the first token,
    // which is indistinguishable from a hang -- upstream shows a spinner for
    // the same reason. On stderr, so a redirected completion stays clean.
    try err.print("loading {s} ...\n", .{settings.model});
    try err.flush();

    var session = try cli.Session.init(gpa, settings.*, err);
    defer session.deinit();
    if (opts.trace) {
        session.turn_trace = err;
        // What `--fit` left, as libllama took it.
        try err.print("[ctx: {d}, gpu layers: {d}]\n", .{ cli.session.c.llama_n_ctx(session.ctx), session.gpu_layers });
        try err.flush();
    }

    if (settings.interactive or settings.useChatTemplate()) {
        var chat = try openChat(gpa, io, settings, opts, &session);
        defer chat.deinit();
        var display: cli.conversation.Display = .{ .out = out, .color = opts.color };
        if (settings.interactive) {
            try runInteractive(io, settings, prompt, &chat, &session, &display, err);
        } else {
            try chat.addUser(prompt);
            _ = try chat.reply(&session, &display, if (settings.display_prompt) out else null);
        }
    } else {
        _ = try session.generate(prompt, out);
    }

    if (settings.show_timings) try session.reportTimings(out);
}

/// Loads the chat templates and the system prompt, as upstream's server does
/// when it loads the model.
///
/// Parameters:
/// - `gpa`: owns the chat.
/// - `io`: for `--chat-template-file` and `--system-prompt-file`.
/// - `settings`: parsed command line; must outlive the chat.
/// - `opts`: the kwargs, which must outlive the chat too.
/// - `session`: a loaded session.
///
/// Return: the chat, released with `deinit`.
fn openChat(
    gpa: std.mem.Allocator,
    io: std.Io,
    settings: *const cli.Args,
    opts: RunOptions,
    session: *cli.Session,
) !cli.conversation.Chat {
    // `--chat-template` beats `--chat-template-file`, which beats the
    // model's own, which a model without one replaces with ChatML.
    const override = try readMaybeFile(gpa, io, settings.chat_template, settings.chat_template_file);
    defer gpa.free(override);
    const system = try readMaybeFile(gpa, io, settings.system_prompt, settings.system_prompt_file);
    defer gpa.free(system);
    return cli.conversation.Chat.init(gpa, settings, opts.kwargs, session.model, override, system);
}

/// Holds a multi-turn conversation on stdin.
///
/// Each line read becomes a user turn. The whole conversation is re-rendered
/// through the chat template every turn rather than appended to, because a
/// template decides for itself where the system prompt goes and how a turn is
/// framed; `Session.generateTurn` keeps the KV prefix that survives, so the
/// re-render costs tokenization rather than decoding.
///
/// `-p` seeds the first turn when given, so `-cnv -p "hello"` answers straight
/// away and then waits for the next line.
///
/// Parameters:
/// - `io`: for reading stdin.
/// - `settings`: parsed command line.
/// - `prompt`: the user's prompt, escape-processed; may be empty.
/// - `chat`: the conversation, holding the system prompt.
/// - `session`: a loaded session.
/// - `display`: where the replies go.
/// - `err`: where the turn marker goes, so a redirected transcript stays clean.
///
/// Return: nothing; ends at end of input.
fn runInteractive(
    io: std.Io,
    settings: *const cli.Args,
    prompt: []const u8,
    chat: *cli.conversation.Chat,
    session: *cli.Session,
    display: *cli.conversation.Display,
    err: *std.Io.Writer,
) !void {
    var stdin_buf: [4096]u8 = undefined;
    var stdin = std.Io.File.stdin().readerStreaming(io, &stdin_buf);

    var seed: ?[]const u8 = if (prompt.len > 0) prompt else null;

    try err.print("conversation mode -- /exit to quit, /clear to reset, /regen to retry\n", .{});
    try err.flush();

    while (true) {
        const line = if (seed) |sp| blk: {
            seed = null;
            break :blk sp;
        } else blk: {
            try err.print("\n> ", .{});
            try err.flush();
            const got = stdin.interface.takeDelimiter('\n') catch |e| switch (e) {
                error.StreamTooLong => return error.TurnTooLong,
                else => return e,
            };
            break :blk got orelse break;
        };

        const turn = std.mem.trim(u8, line, " \t\r\n");
        if (turn.len == 0) continue;

        if (std.mem.startsWith(u8, turn, "/exit")) break;

        if (std.mem.startsWith(u8, turn, "/clear")) {
            try chat.clear();
            try err.print("chat history cleared\n", .{});
            try err.flush();
            continue;
        }

        if (std.mem.startsWith(u8, turn, "/regen")) {
            // Drop the last reply so the turn before it is answered again.
            if (!chat.dropReply()) {
                try err.print("nothing to regenerate\n", .{});
                try err.flush();
                continue;
            }
        } else {
            try chat.addUser(turn);
        }

        _ = try chat.reply(session, display, null);
        // Say when the reply was cut off rather than finished: without this
        // the prompt comes back mid-sentence and the cut looks like the
        // model's. stderr, so a piped transcript holds only the reply.
        switch (session.last_stop) {
            .eog, .word, .rejected => {},
            .n_predict => try err.print("[reply cut off: -n {d} tokens reached]\n", .{settings.n_predict}),
            .context_full => try err.print("[reply cut off: context full; /clear to start over, or raise -c]\n", .{}),
        }
        try err.flush();
    }

    try err.print("\n", .{});
    try err.flush();
}

/// A flag's literal value, or the contents of the file its `-file` twin names.
///
/// Parameters:
/// - `gpa`: allocator for the result.
/// - `io`: for reading `path`.
/// - `literal`: the inline value, or empty.
/// - `path`: the file, or empty.
///
/// Return: the text, owned by the caller. Empty when neither was given.
fn readMaybeFile(
    gpa: std.mem.Allocator,
    io: std.Io,
    literal: []const u8,
    path: []const u8,
) ![]u8 {
    if (literal.len > 0) return gpa.dupe(u8, literal);
    if (path.len > 0) return readFile(gpa, io, path);
    return gpa.alloc(u8, 0);
}

/// The `what()` of the last `resolveGrammar` failure.
var grammar_error: std.ArrayList(u8) = .empty;

/// Resolves `--grammar`, `--grammar-file`, `--json-schema` or
/// `--json-schema-file` to GBNF, as their handlers in upstream's
/// `common_params_parser_init` (common/arg.cpp:1397 @620d2225c) do: a schema is parsed
/// with `common_json::parse` and converted with `json_schema_to_grammar`,
/// whose warnings go to stderr.
///
/// Parameters:
/// - `gpa`: allocates the result; an arena in practice.
/// - `io`: for reading the file forms.
/// - `src`: what the command line gave.
///
/// Return: the grammar, empty for none; on failure an error, with the text
/// upstream's exception would carry in `grammar_error`.
fn resolveGrammar(gpa: std.mem.Allocator, io: std.Io, src: cli.args.GrammarSource) ![]const u8 {
    grammar_error.clearRetainingCapacity();
    const text: []const u8 = switch (src) {
        .none => return "",
        .gbnf => |g| return g,
        .gbnf_file, .schema_file => |path| readFile(gpa, io, path) catch |e| {
            if (e == error.OutOfMemory) return e;
            // the message of `read_file` (common/arg.cpp:67 @620d2225c), newline included
            try grammar_error.print(gpa, "error: failed to open file '{s}'\n", .{path});
            return error.GrammarSource;
        },
        .schema => |sch| sch,
    };
    if (src == .gbnf_file) return text;

    var value = cli.json.parse(gpa, text) catch |e| {
        if (e == error.OutOfMemory) return e;
        try grammar_error.appendSlice(gpa, cli.json.what());
        return error.GrammarSource;
    };
    defer value.deinit(gpa);
    return cli.json.schema_to_grammar.jsonSchemaToGrammar(gpa, &value, false) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        try grammar_error.appendSlice(gpa, cli.json.schema.what());
        return error.GrammarSource;
    };
}

/// Reads a whole file.
///
/// Parameters:
/// - `gpa`: allocator for the contents.
/// - `io`: the I/O implementation.
/// - `path`: what to read.
///
/// Return: the contents, owned by the caller.
fn readFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(4 << 20));
}

/// Swallows libllama's INFO and DEBUG chatter, passing warnings and errors on.
///
/// Parameters:
/// - `level`: libllama log level.
/// - `text`: the message, NUL-terminated.
/// - `user_data`: unused.
///
/// Return: nothing.
fn quietLog(level: c_uint, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = user_data;
    if (level == cli.session.c.GGML_LOG_LEVEL_ERROR or level == cli.session.c.GGML_LOG_LEVEL_WARN) {
        std.debug.print("{s}", .{std.mem.span(text)});
    }
}

/// Turns an error into the sentence a user should read.
///
/// Parameters:
/// - `e`: the error to describe.
///
/// Return: a static string. Deliberately exhaustive over the session errors so
/// adding one forces a decision about what to say.
fn describe(e: anyerror) []const u8 {
    return switch (e) {
        cli.session.Error.ModelLoadFailed => "could not load the model -- check the path and that it is a GGUF file",
        cli.session.Error.NoVocab => "the model has no vocabulary",
        cli.session.Error.ContextFailed => "could not create a context -- try a smaller -c",
        cli.session.Error.SamplerFailed => "could not build the sampler chain",
        cli.session.Error.GrammarFailed => "failed to parse grammar",
        cli.session.Error.TokenizeFailed => "could not tokenize the prompt",
        cli.session.Error.DecodeFailed => "the model failed to decode",
        cli.session.Error.PromptTooLong => "the prompt is longer than the context -- raise -c or shorten it",
        cli.conversation.Error.ChatFailed => cli.conversation.last_error.items,
        cli.conversation.Error.MarkingLost => "the chat template's output could not be matched with its trust marks",
        cli.conversation.Error.EmptyRender => "the chat template rendered nothing",
        error.OutOfMemory => "out of memory",
        else => @errorName(e),
    };
}
