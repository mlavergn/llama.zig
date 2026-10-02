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
    .log_level = if (builtin.mode == .Debug) .debug else .warn,
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
        .ok => |settings| {
            if (settings.done) {
                try out.flush();
                return;
            }
            // Upstream is interactive by default and this binary is not, so it says so rather than quietly differing.
            if (!settings.one_shot and !settings.interactive) {
                try err.print(
                    "note: this defaults to one-shot completion; upstream's llama-cli would be\n" ++
                        "      interactive here. Pass -cnv for a conversation, or -st to say you meant this.\n",
                    .{},
                );
                try err.flush();
            }
            run(init.gpa, init.io, settings, err, out) catch |e| {
                try err.print("error: {s}\n", .{describe(e)});
                try err.flush();
                std.process.exit(1);
            };
        },
    }
}

/// Loads the model and generates.
///
/// Split from `main` so the failure path above stays a single `catch`.
///
/// Parameters:
/// - `gpa`: allocator for the prompt and token buffers.
/// - `io`: for reading a template or system prompt from a file.
/// - `settings`: parsed command line.
/// - `err`: where the loading notice goes.
/// - `out`: destination for the completion.
///
/// Return: nothing on success; propagates the stage that failed.
fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    settings: cli.Args,
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

    var session = try cli.Session.init(gpa, settings);
    defer session.deinit();

    if (settings.interactive) {
        try runInteractive(gpa, io, settings, prompt, &session, err, out);
    } else if (settings.useChatTemplate()) {
        try runChat(gpa, io, settings, prompt, &session, out);
    } else {
        _ = try session.generate(prompt, out);
    }

    if (settings.show_timings) try session.reportTimings(out);
}

/// Generates through the model's chat template.
///
/// Parameters:
/// - `gpa`: allocator for the template, messages and rendered prompt.
/// - `io`: for reading a template or system prompt from a file.
/// - `settings`: parsed command line.
/// - `prompt`: the user's prompt, escape-processed.
/// - `session`: a loaded session.
/// - `out`: destination for the completion.
///
/// Return: nothing; propagates the stage that failed.
fn runChat(
    gpa: std.mem.Allocator,
    io: std.Io,
    settings: cli.Args,
    prompt: []const u8,
    session: *cli.Session,
    out: *std.Io.Writer,
) !void {
    var template = try resolveTemplate(gpa, io, settings, session.model);
    defer template.deinit();

    const system = try readMaybeFile(gpa, io, settings.system_prompt, settings.system_prompt_file);
    defer gpa.free(system);

    // One turn of a conversation: an optional system message, then the
    // prompt as the user's turn. Multi-turn is interactive mode, which this
    // binary does not have.
    var messages: std.ArrayList(cli.chat.Message) = .empty;
    defer messages.deinit(gpa);
    if (system.len > 0) try messages.append(gpa, .{ .role = "system", .content = system });
    try messages.append(gpa, .{ .role = "user", .content = prompt });

    const segments = try template.applySegments(messages.items, .{
        .add_generation_prompt = true,
        .bos_token = tokenText(session, cli.session.c.llama_vocab_bos(session.vocab)),
        .eos_token = tokenText(session, cli.session.c.llama_vocab_eos(session.vocab)),
    });
    defer cli.chat.freeSegments(gpa, segments);

    _ = try session.generateChat(segments, out);
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
/// - `gpa`: allocator for the conversation and the rendered prompts.
/// - `io`: for reading stdin, the template and the system prompt.
/// - `settings`: parsed command line.
/// - `prompt`: the user's prompt, escape-processed; may be empty.
/// - `session`: a loaded session.
/// - `err`: where the turn marker goes, so a redirected transcript stays clean.
/// - `out`: destination for the replies.
///
/// Return: nothing; ends at end of input.
fn runInteractive(
    gpa: std.mem.Allocator,
    io: std.Io,
    settings: cli.Args,
    prompt: []const u8,
    session: *cli.Session,
    err: *std.Io.Writer,
    out: *std.Io.Writer,
) !void {
    var template = try resolveTemplate(gpa, io, settings, session.model);
    defer template.deinit();

    const system = try readMaybeFile(gpa, io, settings.system_prompt, settings.system_prompt_file);
    defer gpa.free(system);

    var history: std.ArrayList(cli.chat.Message) = .empty;
    defer {
        for (history.items) |m| gpa.free(m.content);
        history.deinit(gpa);
    }
    if (system.len > 0) {
        try history.append(gpa, .{ .role = "system", .content = try gpa.dupe(u8, system) });
    }

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
            for (history.items) |m| gpa.free(m.content);
            history.clearRetainingCapacity();
            if (system.len > 0) {
                try history.append(gpa, .{ .role = "system", .content = try gpa.dupe(u8, system) });
            }
            try err.print("chat history cleared\n", .{});
            try err.flush();
            continue;
        }

        if (std.mem.startsWith(u8, turn, "/regen")) {
            // Drop the last reply so the turn before it is answered again.
            const last = history.getLastOrNull();
            if (last == null or !std.mem.eql(u8, last.?.role, "assistant")) {
                try err.print("nothing to regenerate\n", .{});
                try err.flush();
                continue;
            }
            gpa.free(history.pop().?.content);
        } else {
            try history.append(gpa, .{ .role = "user", .content = try gpa.dupe(u8, turn) });
        }

        const segments = try template.applySegments(history.items, .{
            .add_generation_prompt = true,
            .bos_token = tokenText(session, cli.session.c.llama_vocab_bos(session.vocab)),
            .eos_token = tokenText(session, cli.session.c.llama_vocab_eos(session.vocab)),
        });
        defer cli.chat.freeSegments(gpa, segments);

        var reply: std.ArrayList(u8) = .empty;
        defer reply.deinit(gpa);

        _ = try session.generateTurn(segments, out, &reply);
        // Say when the reply was cut off rather than finished: without this
        // the prompt comes back mid-sentence and the cut looks like the
        // model's. stderr, so a piped transcript holds only the reply.
        switch (session.last_stop) {
            .eog => {},
            .n_predict => try err.print("[reply cut off: -n {d} tokens reached]\n", .{settings.n_predict}),
            .context_full => try err.print("[reply cut off: context full; /clear to start over, or raise -c]\n", .{}),
        }
        try err.flush();

        try history.append(gpa, .{ .role = "assistant", .content = try gpa.dupe(u8, reply.items) });
    }

    try err.print("\n", .{});
    try err.flush();
}

/// The template to use, in upstream's order of precedence.
///
/// `--chat-template` beats `--chat-template-file`, which beats the model's
/// own.
///
/// Parameters:
/// - `gpa`: allocator for the template source.
/// - `io`: for `--chat-template-file`.
/// - `settings`: parsed command line.
/// - `model`: the loaded model, for its embedded template.
///
/// Return: the template, released with `deinit`.
fn resolveTemplate(
    gpa: std.mem.Allocator,
    io: std.Io,
    settings: cli.Args,
    model: *cli.session.c.llama_model,
) !cli.chat.Template {
    if (settings.chat_template.len > 0) {
        return cli.chat.Template.init(gpa, settings.chat_template);
    }
    if (settings.chat_template_file.len > 0) {
        const src = try readFile(gpa, io, settings.chat_template_file);
        defer gpa.free(src);
        return cli.chat.Template.init(gpa, src);
    }
    return cli.chat.Template.fromModel(gpa, model, null);
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

/// The text a token spells, for binding `bos_token` and `eos_token`.
///
/// Parameters:
/// - `session`: a loaded session.
/// - `token`: the token to spell, or `LLAMA_TOKEN_NULL`.
///
/// Return: the spelling, or empty when the vocabulary has no such token. Points
/// into libllama's own storage, which outlives the render.
fn tokenText(session: *cli.Session, token: cli.session.c.llama_token) []const u8 {
    if (token == cli.session.c.LLAMA_TOKEN_NULL) return "";
    const raw = cli.session.c.llama_vocab_get_text(session.vocab, token) orelse return "";
    return std.mem.span(raw);
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
        cli.session.Error.TokenizeFailed => "could not tokenize the prompt",
        cli.session.Error.DecodeFailed => "the model failed to decode",
        cli.session.Error.PromptTooLong => "the prompt is longer than the context -- raise -c or shorten it",
        error.OutOfMemory => "out of memory",
        else => @errorName(e),
    };
}
