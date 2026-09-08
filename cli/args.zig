//! Command-line parsing for `llamazig`, mimicking upstream's `llama-cli`.
//!
//! # Provenance
//!
//! **Not a port.** This is our own code. The *flag surface* it accepts is
//! modelled on `llama.cpp/common/arg.cpp` (v0.3.0, `c1d0e7a00`) and the
//! defaults on `llama.cpp/common/common.h`, so a command line written for
//! upstream's `llama-cli` runs unchanged against this binary for the features
//! that are ported. Each option below names the upstream spelling it matches.
//!
//! # Flag compatibility is the contract
//!
//! Decision 20: this is a drop-in replacement for what we support, not a new
//! CLI with its own ideas. So flags keep upstream's names, aliases, argument
//! forms and defaults even where something else would read better. A flag we
//! have not implemented is **refused**, never silently ignored (Decision 22) --
//! silently ignoring `--temp` would make a wrong answer look like a right one.
//!
//! # One-shot only, for now
//!
//! Decision 30 sequences one-shot generation before interactive conversation.
//! Upstream's `llama-cli` is interactive *by default*; this binary is not, so
//! it says so rather than quietly behaving differently -- see `Args.one_shot`.
//!
//! Which flag means "one shot" depends on which upstream binary you came from,
//! and they do not agree:
//!
//! - `tools/cli` (the `llama-cli` this repo builds) takes **`-st`** and
//!   **rejects** `-no-cnv`.
//! - `tools/main` takes **`-no-cnv`** and has no `-st`.
//!
//! Both are accepted here. Refusing either would break a command line that
//! works against one of the two upstream binaries, which is the opposite of
//! what Decision 20 promises.

const std = @import("std");
const upstream = @import("upstream_flags.zig");

/// Sampling and generation settings, plus what to run them on.
///
/// Defaults match `common_params` and `common_params_sampling` in
/// `llama.cpp/common/common.h`, so omitting a flag here behaves as omitting it
/// upstream. Where a default differs, the field says why.
pub const Args = struct {
    /// `-m, --model` -- required; there is no default model.
    model: []const u8,
    /// `-p, --prompt`. Upstream defaults to empty.
    prompt: []const u8 = "",
    /// `-n, --predict, --n-predict`. -1 means "until EOG or context is full".
    n_predict: i32 = -1,
    /// `-c, --ctx-size`. 0 means "whatever the model was trained with".
    n_ctx: i32 = 0,
    /// `-b, --batch-size`.
    n_batch: i32 = 2048,
    /// `-ngl, --gpu-layers, --n-gpu-layers`. -1 is auto.
    n_gpu_layers: i32 = -1,
    /// `-t, --threads`. 0 lets llama.cpp pick.
    n_threads: i32 = 0,
    /// `-s, --seed`. `LLAMA_DEFAULT_SEED` asks for a random one.
    seed: u32 = 0xFFFFFFFF,
    /// `--temp, --temperature`. <= 0 samples greedily.
    temp: f32 = 0.80,
    /// `--top-k`. <= 0 uses the vocabulary size.
    top_k: i32 = 40,
    /// `--top-p`. 1.0 disables.
    top_p: f32 = 0.95,
    /// `--min-p`. 0.0 disables.
    min_p: f32 = 0.05,
    /// `--repeat-last-n`. 0 disables the penalty; -1 uses the context size.
    penalty_last_n: i32 = 64,
    /// `--repeat-penalty`. 1.0 disables.
    penalty_repeat: f32 = 1.00,
    /// `--ignore-eos`.
    ignore_eos: bool = false,
    /// `-e, --escape` / `--no-escape`. Upstream escapes by default.
    escape: bool = true,
    /// `--no-warmup`.
    warmup: bool = true,
    /// Set by `-st`, `--single-turn`, `-no-cnv`, or `--no-conversation`.
    ///
    /// All four say "generate once and exit", which is the only thing this
    /// binary does. They are accepted rather than refused because each is the
    /// correct flag for one of the upstream binaries.
    ///
    /// When *none* of them is given, upstream would have been interactive and
    /// we are not, so `main` warns. Decision 30: say so plainly rather than
    /// behaving differently in silence.
    one_shot: bool = false,
    /// `-cnv`, `--conversation`, `-i`, `--interactive`.
    ///
    /// Multi-turn: each line read from stdin becomes a user turn, the whole
    /// conversation is re-rendered through the chat template, and the reply is
    /// appended to the history. Implies chat mode, since a conversation with
    /// no template is just concatenated text.
    interactive: bool = false,
    /// `--no-display-prompt` inverted. Upstream echoes the prompt by default.
    display_prompt: bool = true,
    /// `--show-timings` / `--no-show-timings`.
    ///
    /// Default true, matching `common_params::show_timings` in
    /// `common/common.h`. Upstream writes the line to stdout, so this does
    /// too -- see the note in `session.zig` about why that is worth knowing.
    show_timings: bool = true,

    /// `--jinja` / `--no-jinja`.
    ///
    /// Upstream defaults this to true and uses it to pick between its Jinja
    /// engine and a legacy one. We have one engine, so it reads as "template
    /// the prompt at all": `--no-jinja` forces raw completion even when a
    /// system prompt or a template was given.
    jinja: bool = true,
    /// `--chat-template TEMPLATE` -- Jinja source, overriding the model's.
    chat_template: []const u8 = "",
    /// `--chat-template-file FNAME` -- the same, read from a file.
    chat_template_file: []const u8 = "",
    /// `-sys, --system-prompt TEXT`.
    system_prompt: []const u8 = "",
    /// `-sysf, --system-prompt-file FNAME`.
    system_prompt_file: []const u8 = "",

    /// Whether to render the prompt through a chat template.
    ///
    /// **Opt-in, where upstream templates by default.** Our default mode is
    /// raw completion, and it is what `make port` and `make ref` diff against
    /// each other; templating by default would silently change the one gate
    /// that covers the whole binary. Any chat flag turns it on. See SPEC
    /// section 6.1.
    chat: bool = false,

    /// Set when parsing consumed the whole command line and the caller should
    /// exit successfully without generating -- `-h` and `--version`.
    done: bool = false,

    /// Whether the prompt should be templated.
    ///
    /// Parameters:
    /// - `self`: parsed arguments.
    ///
    /// Return: true when a chat flag asked for it and `--no-jinja` did not
    /// veto it.
    pub fn useChatTemplate(self: Args) bool {
        return self.chat and self.jinja;
    }
};

/// What went wrong, as distinct from how it is reported.
pub const Error = error{
    MissingModel,
    MissingValue,
    UnknownFlag,
    UnsupportedFlag,
    BadNumber,
    OutOfMemory,
};

/// A parse failure with enough context to explain itself.
///
/// The parser fills this in rather than printing, so the same failures are
/// assertable from a test without capturing stdout.
pub const Failure = struct {
    err: Error,
    /// The offending argument, borrowed from `argv` and valid as long as it is.
    arg: []const u8 = "",

    /// Writes the message a user sees.
    ///
    /// Parameters:
    /// - `self`: the failure to describe.
    /// - `w`: destination.
    ///
    /// Return: nothing; propagates write errors.
    pub fn report(self: Failure, w: *std.Io.Writer) !void {
        switch (self.err) {
            error.MissingModel => try w.print(
                "error: no model given\n" ++
                    "       pass one with -m PATH (or --model PATH)\n",
                .{},
            ),
            error.MissingValue => try w.print(
                "error: {s} needs a value\n",
                .{self.arg},
            ),
            // The distinction Decision 22 is really about: the user's command
            // line is fine and we are behind, which is worth saying plainly.
            error.UnsupportedFlag => try w.print(
                "error: {s} is a llama.cpp flag that llamazig does not support yet\n" ++
                    "       this port is incomplete; run `llamazig --help` for what works today\n",
                .{self.arg},
            ),
            error.UnknownFlag => try w.print(
                "error: unknown flag {s}\n" ++
                    "       run `llamazig --help` for the supported flags\n",
                .{self.arg},
            ),
            error.BadNumber => try w.print(
                "error: {s} is not a valid number\n",
                .{self.arg},
            ),
            error.OutOfMemory => try w.print("error: out of memory\n", .{}),
        }
    }
};

/// Result of parsing: either settings to run, or a failure to report.
pub const Result = union(enum) {
    ok: Args,
    fail: Failure,
};

/// Every flag this binary accepts, with its upstream aliases.
///
/// Kept as data rather than a `switch` so the help text and the "is this
/// supported" check are generated from one source and cannot drift from what
/// the parser actually does.
const Spec = struct {
    names: []const []const u8,
    /// Placeholder shown in help, empty for a flag that takes no value.
    value: []const u8 = "",
    help: []const u8,
};

const specs = [_]Spec{
    .{ .names = &.{ "-h", "--help", "--usage" }, .help = "print this help and exit" },
    .{ .names = &.{"--version"}, .help = "print version and exit" },
    .{ .names = &.{ "-m", "--model" }, .value = "PATH", .help = "model path (required)" },
    .{ .names = &.{ "-p", "--prompt" }, .value = "TEXT", .help = "prompt to complete" },
    .{ .names = &.{ "-n", "--predict", "--n-predict" }, .value = "N", .help = "tokens to predict (-1 = unlimited)" },
    .{ .names = &.{ "-c", "--ctx-size" }, .value = "N", .help = "context size (0 = from model)" },
    .{ .names = &.{ "-b", "--batch-size" }, .value = "N", .help = "logical batch size" },
    .{ .names = &.{ "-ngl", "--gpu-layers", "--n-gpu-layers" }, .value = "N", .help = "layers to offload (-1 = auto)" },
    .{ .names = &.{ "-t", "--threads" }, .value = "N", .help = "threads to use" },
    .{ .names = &.{ "-s", "--seed" }, .value = "N", .help = "RNG seed" },
    .{ .names = &.{ "--temp", "--temperature" }, .value = "F", .help = "temperature (<= 0 = greedy)" },
    .{ .names = &.{"--top-k"}, .value = "N", .help = "top-k sampling" },
    .{ .names = &.{"--top-p"}, .value = "F", .help = "top-p sampling (1.0 = off)" },
    .{ .names = &.{"--min-p"}, .value = "F", .help = "min-p sampling (0.0 = off)" },
    .{ .names = &.{"--repeat-last-n"}, .value = "N", .help = "tokens to penalise (0 = off, -1 = ctx)" },
    .{ .names = &.{"--repeat-penalty"}, .value = "F", .help = "repeat penalty (1.0 = off)" },
    .{ .names = &.{"--ignore-eos"}, .help = "never stop at end-of-generation" },
    .{ .names = &.{ "-e", "--escape" }, .help = "process \\n, \\t, \\\\ in the prompt (default)" },
    .{ .names = &.{"--no-escape"}, .help = "take the prompt literally" },
    .{ .names = &.{"--no-warmup"}, .help = "skip the warmup decode (on by default; timings suffer without it)" },
    .{ .names = &.{"--no-display-prompt"}, .help = "do not echo the prompt before the completion" },
    .{ .names = &.{"--show-timings"}, .help = "print prompt and generation tokens/second (default)" },
    .{ .names = &.{"--no-show-timings"}, .help = "suppress the timings line" },
    .{ .names = &.{"--jinja"}, .help = "render the prompt through the model's chat template" },
    .{ .names = &.{"--no-jinja"}, .help = "raw completion; never apply a chat template" },
    .{ .names = &.{"--chat-template"}, .value = "TEXT", .help = "Jinja template to use instead of the model's" },
    .{ .names = &.{"--chat-template-file"}, .value = "FNAME", .help = "the same, read from a file" },
    .{ .names = &.{ "-sys", "--system-prompt" }, .value = "TEXT", .help = "system message prepended to the conversation" },
    .{ .names = &.{ "-sysf", "--system-prompt-file" }, .value = "FNAME", .help = "the same, read from a file" },
    .{ .names = &.{ "-st", "--single-turn" }, .help = "generate once and exit (the only mode today)" },
    .{ .names = &.{ "-no-cnv", "--no-conversation" }, .help = "same; the spelling tools/main uses" },
    .{ .names = &.{ "-cnv", "--conversation", "-i", "--interactive" }, .help = "multi-turn conversation, reading turns from stdin" },
};

/// Whether `flag` names an option this binary accepts.
fn supported(flag: []const u8) bool {
    for (specs) |s| {
        for (s.names) |n| {
            if (std.mem.eql(u8, n, flag)) return true;
        }
    }
    return false;
}

/// Writes the help text.
///
/// Parameters:
/// - `w`: destination.
///
/// Return: nothing; propagates write errors.
pub fn help(w: *std.Io.Writer) !void {
    try w.print(
        \\usage: llamazig -m PATH [options]
        \\
        \\A Zig port of llama.cpp. Flags match upstream's llama-cli for the
        \\features this port supports; anything else is refused rather than
        \\ignored.
        \\
        \\One-shot by default; -cnv holds a multi-turn conversation on stdin.
        \\
        \\options:
        \\
    , .{});

    for (specs) |s| {
        var buf: [64]u8 = undefined;
        var n: usize = 0;
        for (s.names, 0..) |name, i| {
            if (i > 0) {
                @memcpy(buf[n..][0..2], ", ");
                n += 2;
            }
            @memcpy(buf[n..][0..name.len], name);
            n += name.len;
        }
        if (s.value.len > 0) {
            buf[n] = ' ';
            n += 1;
            @memcpy(buf[n..][0..s.value.len], s.value);
            n += s.value.len;
        }
        try w.print("  {s: <38}{s}\n", .{ buf[0..n], s.help });
    }
}

/// Parses a command line.
///
/// Parameters:
/// - `argv`: the full argument vector, program name included. Slices in the
///   result borrow from it and are valid as long as it is.
/// - `w`: where `--help` and `--version` write. Not used for errors, which are
///   returned rather than printed.
///
/// Return: `.ok` with settings to run, or `.fail` with a reportable failure.
/// `Args.done` is set when the command line asked for output and nothing else.
pub fn parse(argv: []const []const u8, w: *std.Io.Writer) !Result {
    var args: Args = .{ .model = "" };
    var saw_model = false;

    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const a = argv[i];

        // A bare word is not a positional argument: upstream has none for
        // llama-cli, so accepting one would invent behaviour.
        if (a.len == 0 or a[0] != '-') {
            return .{ .fail = .{ .err = error.UnknownFlag, .arg = a } };
        }

        // Reads the value belonging to the current flag.
        const value = struct {
            fn next(argv_: []const []const u8, idx: *usize) ?[]const u8 {
                if (idx.* + 1 >= argv_.len) return null;
                idx.* += 1;
                return argv_[idx.*];
            }
        }.next;

        if (eq(a, &.{ "-h", "--help", "--usage" })) {
            try help(w);
            args.done = true;
            return .{ .ok = args };
        } else if (eq(a, &.{"--version"})) {
            try w.print("llamazig {s}\n", .{version});
            args.done = true;
            return .{ .ok = args };
        } else if (eq(a, &.{ "-m", "--model" })) {
            args.model = value(argv, &i) orelse return missing(a);
            saw_model = true;
        } else if (eq(a, &.{ "-p", "--prompt" })) {
            args.prompt = value(argv, &i) orelse return missing(a);
        } else if (eq(a, &.{ "-n", "--predict", "--n-predict" })) {
            args.n_predict = parseI32(value(argv, &i) orelse return missing(a)) catch return bad(a);
        } else if (eq(a, &.{ "-c", "--ctx-size" })) {
            args.n_ctx = parseI32(value(argv, &i) orelse return missing(a)) catch return bad(a);
        } else if (eq(a, &.{ "-b", "--batch-size" })) {
            args.n_batch = parseI32(value(argv, &i) orelse return missing(a)) catch return bad(a);
        } else if (eq(a, &.{ "-ngl", "--gpu-layers", "--n-gpu-layers" })) {
            args.n_gpu_layers = parseI32(value(argv, &i) orelse return missing(a)) catch return bad(a);
        } else if (eq(a, &.{ "-t", "--threads" })) {
            args.n_threads = parseI32(value(argv, &i) orelse return missing(a)) catch return bad(a);
        } else if (eq(a, &.{ "-s", "--seed" })) {
            const v = value(argv, &i) orelse return missing(a);
            args.seed = std.fmt.parseInt(u32, v, 10) catch return bad(v);
        } else if (eq(a, &.{ "--temp", "--temperature" })) {
            args.temp = parseF32(value(argv, &i) orelse return missing(a)) catch return bad(a);
        } else if (eq(a, &.{"--top-k"})) {
            args.top_k = parseI32(value(argv, &i) orelse return missing(a)) catch return bad(a);
        } else if (eq(a, &.{"--top-p"})) {
            args.top_p = parseF32(value(argv, &i) orelse return missing(a)) catch return bad(a);
        } else if (eq(a, &.{"--min-p"})) {
            args.min_p = parseF32(value(argv, &i) orelse return missing(a)) catch return bad(a);
        } else if (eq(a, &.{"--repeat-last-n"})) {
            args.penalty_last_n = parseI32(value(argv, &i) orelse return missing(a)) catch return bad(a);
        } else if (eq(a, &.{"--repeat-penalty"})) {
            args.penalty_repeat = parseF32(value(argv, &i) orelse return missing(a)) catch return bad(a);
        } else if (eq(a, &.{"--ignore-eos"})) {
            args.ignore_eos = true;
        } else if (eq(a, &.{ "-e", "--escape" })) {
            args.escape = true;
        } else if (eq(a, &.{"--no-escape"})) {
            args.escape = false;
        } else if (eq(a, &.{"--no-warmup"})) {
            args.warmup = false;
        } else if (eq(a, &.{"--no-display-prompt"})) {
            args.display_prompt = false;
        } else if (eq(a, &.{"--show-timings"})) {
            args.show_timings = true;
        } else if (eq(a, &.{"--jinja"})) {
            args.jinja = true;
            args.chat = true;
        } else if (eq(a, &.{"--no-jinja"})) {
            args.jinja = false;
        } else if (eq(a, &.{"--chat-template"})) {
            args.chat_template = value(argv, &i) orelse return missing(a);
            args.chat = true;
        } else if (eq(a, &.{"--chat-template-file"})) {
            args.chat_template_file = value(argv, &i) orelse return missing(a);
            args.chat = true;
        } else if (eq(a, &.{ "-sys", "--system-prompt" })) {
            args.system_prompt = value(argv, &i) orelse return missing(a);
            args.chat = true;
        } else if (eq(a, &.{ "-sysf", "--system-prompt-file" })) {
            args.system_prompt_file = value(argv, &i) orelse return missing(a);
            args.chat = true;
        } else if (eq(a, &.{"--no-show-timings"})) {
            args.show_timings = false;
        } else if (eq(a, &.{ "-cnv", "--conversation", "-i", "--interactive" })) {
            args.interactive = true;
            args.chat = true;
        } else if (eq(a, &.{ "-st", "--single-turn", "-no-cnv", "--no-conversation" })) {
            args.one_shot = true;
        } else if (upstream.contains(a)) {
            // Real flag, we are behind. Distinguished from a typo on purpose.
            return .{ .fail = .{ .err = error.UnsupportedFlag, .arg = a } };
        } else {
            return .{ .fail = .{ .err = error.UnknownFlag, .arg = a } };
        }
    }

    if (!saw_model) return .{ .fail = .{ .err = error.MissingModel } };
    return .{ .ok = args };
}

/// The version this binary reports. Ours, not llama.cpp's: a user asking a
/// llamazig binary what it is should not be told a version of another project.
pub const version = "0.0.1";

fn eq(a: []const u8, names: []const []const u8) bool {
    for (names) |n| {
        if (std.mem.eql(u8, a, n)) return true;
    }
    return false;
}

fn missing(arg: []const u8) Result {
    return .{ .fail = .{ .err = error.MissingValue, .arg = arg } };
}

fn bad(arg: []const u8) Result {
    return .{ .fail = .{ .err = error.BadNumber, .arg = arg } };
}

fn parseI32(s: []const u8) !i32 {
    return std.fmt.parseInt(i32, s, 10);
}

fn parseF32(s: []const u8) !f32 {
    return std.fmt.parseFloat(f32, s);
}

/// Expands the escapes upstream's `-e` handles, in place semantics.
///
/// Upstream escapes by default and `string_process_escapes` handles
/// `\n \t \r \' \" \\ \xNN`. Anything else is left as written, backslash
/// included, which is upstream's behaviour and not an error.
///
/// Parameters:
/// - `allocator`: allocates the result.
/// - `s`: the raw prompt.
///
/// Return: a newly allocated string the caller frees. Never longer than `s`.
pub fn processEscapes(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var out = try allocator.alloc(u8, s.len);
    errdefer allocator.free(out);

    var n: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        // A backslash at the very end escapes nothing and stays as written.
        if (s[i] != '\\' or i + 1 >= s.len) {
            out[n] = s[i];
            n += 1;
            i += 1;
            continue;
        }

        const esc = s[i + 1];
        const simple: ?u8 = switch (esc) {
            'n' => '\n',
            't' => '\t',
            'r' => '\r',
            '\'' => '\'',
            '"' => '"',
            '\\' => '\\',
            else => null,
        };
        if (simple) |ch| {
            out[n] = ch;
            n += 1;
            i += 2;
            continue;
        }

        // \xNN, and only with two hex digits. Anything else is not an escape.
        if (esc == 'x' and i + 3 < s.len) {
            if (std.fmt.parseInt(u8, s[i + 2 ..][0..2], 16)) |byte| {
                out[n] = byte;
                n += 1;
                i += 4;
                continue;
            } else |_| {}
        }

        // Not an escape: keep the backslash and the character after it, which
        // is what upstream's string_process_escapes does.
        out[n] = '\\';
        n += 1;
        i += 1;
    }

    return allocator.realloc(out, n);
}

// -----------------------------------------------------------------------------
// Unit Tests

const testing = std.testing;

test {
    testing.refAllDecls(@This());
}

/// Parses into a `Result` with a throwaway writer, for the tests below.
fn parseForTest(argv: []const []const u8) !Result {
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    return parse(argv, &w);
}

test "a minimal command line yields upstream's defaults" {
    const r = try parseForTest(&.{ "llamazig", "-m", "model.gguf" });
    const a = r.ok;
    try testing.expectEqualStrings("model.gguf", a.model);
    // These are `common_params` / `common_params_sampling` values, not ours.
    // If upstream's defaults move, this test is how we find out.
    try testing.expectEqual(@as(i32, -1), a.n_predict);
    try testing.expectEqual(@as(i32, 0), a.n_ctx);
    try testing.expectEqual(@as(i32, 2048), a.n_batch);
    try testing.expectEqual(@as(f32, 0.80), a.temp);
    try testing.expectEqual(@as(i32, 40), a.top_k);
    try testing.expectEqual(@as(f32, 0.95), a.top_p);
    try testing.expectEqual(@as(f32, 0.05), a.min_p);
    try testing.expectEqual(@as(i32, 64), a.penalty_last_n);
    try testing.expectEqual(@as(f32, 1.00), a.penalty_repeat);
    try testing.expect(a.escape);
    try testing.expect(a.display_prompt);
    try testing.expect(a.show_timings);
}

test "every alias of a flag reaches the same field" {
    for ([_][]const u8{ "-n", "--predict", "--n-predict" }) |flag| {
        const r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", flag, "7" });
        try testing.expectEqual(@as(i32, 7), r.ok.n_predict);
    }
    for ([_][]const u8{ "-ngl", "--gpu-layers", "--n-gpu-layers" }) |flag| {
        const r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", flag, "33" });
        try testing.expectEqual(@as(i32, 33), r.ok.n_gpu_layers);
    }
    for ([_][]const u8{ "--temp", "--temperature" }) |flag| {
        const r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", flag, "0" });
        try testing.expectEqual(@as(f32, 0.0), r.ok.temp);
    }
}

test "a real upstream flag is refused differently from a typo" {
    // The distinction Decision 22 exists for: one of these is the user's
    // mistake, the other is ours.
    const known = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "--mirostat", "2" });
    try testing.expectEqual(Error.UnsupportedFlag, known.fail.err);

    const typo = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "--mirostatt" });
    try testing.expectEqual(Error.UnknownFlag, typo.fail.err);
}

test "every spelling of interactive mode is accepted, and implies chat" {
    // A conversation with no chat template is just concatenated text, so the
    // flag turns templating on rather than leaving it to be asked for twice.
    for ([_][]const u8{ "-cnv", "--conversation", "-i", "--interactive" }) |flag| {
        const r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", flag });
        try testing.expect(r.ok.interactive);
        try testing.expect(r.ok.useChatTemplate());
    }
}

test "interactive spellings we do not implement are still refused" {
    // -if starts interactive *and* takes a file; -mli is multiline input. Both
    // change what a turn is, so accepting them would answer a different
    // question from the one asked.
    for ([_][]const u8{ "-if", "-mli" }) |flag| {
        const r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", flag });
        try testing.expectEqual(Error.UnsupportedFlag, r.fail.err);
    }
}

test "every upstream spelling of one-shot is accepted" {
    // The two upstream binaries disagree: `tools/cli` takes -st and rejects
    // -no-cnv, `tools/main` takes -no-cnv and has no -st. Refusing either
    // would break a command line that works against one of them.
    for ([_][]const u8{ "-st", "--single-turn", "-no-cnv", "--no-conversation" }) |flag| {
        const r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", flag });
        try testing.expect(r.ok.one_shot);
    }
    // Absent, it stays false -- that is what makes `main` warn.
    const bare = try parseForTest(&.{ "llamazig", "-m", "m.gguf" });
    try testing.expect(!bare.ok.one_shot);
}

test "no flag is both supported and reported as unsupported" {
    // `upstream_flags.zig` is a diagnostic list, never a parser input. If a
    // flag appeared in both, `parse` would accept it and the error path would
    // be unreachable -- or worse, the other way around.
    for (specs) |s| {
        for (s.names) |n| {
            try testing.expect(supported(n));
        }
    }
    // Everything we support that upstream also has must spell it identically,
    // which is the whole of "flag-compatible". Ours-only flags are allowed but
    // there should be none today.
    for (specs) |s| {
        for (s.names) |n| {
            if (!upstream.contains(n)) {
                std.debug.print("flag {s} is not an upstream spelling\n", .{n});
                return error.NotUpstreamSpelling;
            }
        }
    }
}

test "a missing value is caught rather than swallowing the next flag" {
    const r = try parseForTest(&.{ "llamazig", "-m" });
    try testing.expectEqual(Error.MissingValue, r.fail.err);
    try testing.expectEqualStrings("-m", r.fail.arg);
}

test "a model is required" {
    const r = try parseForTest(&.{ "llamazig", "-p", "hello" });
    try testing.expectEqual(Error.MissingModel, r.fail.err);
}

test "a bare word is refused rather than taken as a positional" {
    // Upstream's llama-cli has no positional arguments, so accepting one would
    // invent behaviour that then has to be supported forever.
    const r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "stray" });
    try testing.expectEqual(Error.UnknownFlag, r.fail.err);
}

test "help and version consume the command line and ask to exit" {
    for ([_][]const u8{ "-h", "--help", "--usage", "--version" }) |flag| {
        const r = try parseForTest(&.{ "llamazig", flag });
        try testing.expect(r.ok.done);
    }
    // ...even without a model, which would otherwise be an error.
    const r = try parseForTest(&.{ "llamazig", "--help" });
    try testing.expect(r.ok.done);
}

test "escape processing matches what upstream's -e handles" {
    const cases = [_]struct { in: []const u8, want: []const u8 }{
        .{ .in = "a\\nb", .want = "a\nb" },
        .{ .in = "a\\tb", .want = "a\tb" },
        .{ .in = "a\\\\b", .want = "a\\b" },
        .{ .in = "a\\x41b", .want = "aAb" },
        // Not an escape: upstream leaves the backslash in place.
        .{ .in = "a\\qb", .want = "a\\qb" },
        // A trailing backslash has nothing to escape.
        .{ .in = "ab\\", .want = "ab\\" },
        // \x needs two hex digits; otherwise it stays literal.
        .{ .in = "a\\xZZ", .want = "a\\xZZ" },
        .{ .in = "no escapes", .want = "no escapes" },
    };
    for (cases) |case| {
        const got = try processEscapes(testing.allocator, case.in);
        defer testing.allocator.free(got);
        try testing.expectEqualStrings(case.want, got);
    }
}

test "escape processing releases its buffer on every path" {
    // Called twice so the leak check has something to prove: the first
    // allocation must be gone before the second is made.
    const a = try processEscapes(testing.allocator, "one\\ntwo");
    testing.allocator.free(a);
    const b = try processEscapes(testing.allocator, "three\\tfour");
    testing.allocator.free(b);
}

test "help lists every supported flag" {
    var buf: [8192]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try help(&w);
    const text = w.buffered();
    for (specs) |s| {
        for (s.names) |n| {
            if (std.mem.indexOf(u8, text, n) == null) {
                std.debug.print("help omits {s}\n", .{n});
                return error.FlagMissingFromHelp;
            }
        }
    }
}

test "timings default on, and both spellings of the switch work" {
    // Upstream's `common_params::show_timings` defaults to true, so a bare
    // command line prints the line.
    const on = try parseForTest(&.{ "llamazig", "-m", "m.gguf" });
    try testing.expect(on.ok.show_timings);

    const off = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "--no-show-timings" });
    try testing.expect(!off.ok.show_timings);

    // Later flags win, as with every other option here.
    const back_on = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "--no-show-timings", "--show-timings" });
    try testing.expect(back_on.ok.show_timings);
}
