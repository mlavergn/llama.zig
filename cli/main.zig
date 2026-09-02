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
            // Decision 30: upstream's llama-cli is interactive by default and
            // this binary is not. Saying so is the difference between "the
            // port is incomplete" and "the port silently did something else".
            // On stderr, so it cannot land in a redirected completion.
            if (!settings.one_shot) {
                try err.print(
                    "note: llamazig only does one-shot completion; upstream's llama-cli would be\n" ++
                        "      interactive here. Pass -st (or -no-cnv) to say you meant this.\n",
                    .{},
                );
                try err.flush();
            }
            run(init.gpa, settings, err, out) catch |e| {
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
/// - `settings`: parsed command line.
/// - `err`: where the loading notice goes.
/// - `out`: destination for the completion.
///
/// Return: nothing on success; propagates the stage that failed.
fn run(gpa: std.mem.Allocator, settings: cli.Args, err: *std.Io.Writer, out: *std.Io.Writer) !void {
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

    _ = try session.generate(prompt, out);

    if (settings.show_timings) try session.reportTimings(out);
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
