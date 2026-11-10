//! Command-line parsing for `llamazig`, mimicking upstream's `llama-cli`.
//!
//! # Provenance
//!
//! **Not a port.** This is our own code. The *flag surface* it accepts is
//! modelled on `llama.cpp/common/arg.cpp` (v0.9.0, `cd959bae9`) and the
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
const common_chat = @import("common_chat");

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
    /// `-c, --ctx-size`. 0 means "whatever the model was trained with",
    /// which `--fit` may then shrink to fit free memory -- unless `-c 0` was
    /// given explicitly; see `fit_min_ctx`.
    n_ctx: i32 = 0,
    /// `-b, --batch-size`.
    n_batch: i32 = 2048,
    /// `-ngl, --gpu-layers, --n-gpu-layers`. -1 is auto.
    n_gpu_layers: i32 = -1,
    /// `-fit, --fit [on|off]`. Fit the context size and, on a GPU, the
    /// layers and tensors given to it to free memory before loading
    /// (`fit.zig`), as upstream does by default.
    fit: bool = true,
    /// `-fitt, --fit-target MiB0,MiB1,...`, in bytes: what `--fit` leaves
    /// free on each device. One value applies to every device; upstream's
    /// default is 1 GiB.
    fit_target: [max_devices]usize = @splat(1024 * 1024 * 1024),
    /// `-fitc, --fit-ctx N`: the least context `--fit` shrinks to. An
    /// explicit `-c 0` sets it to `maxInt(u32)`, which forbids shrinking:
    /// upstream's `-c` handler sets `fit_params_min_ctx` so
    /// (common/arg.cpp:1595 @cd959bae9). The later of the two flags wins.
    fit_min_ctx: u32 = 4096,
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

    /// `--grammar`, `--grammar-file`, `-j, --json-schema` and
    /// `-jf, --json-schema-file`: where the grammar constraining generation
    /// comes from. Each sets upstream's `params.sampling.grammar` outright
    /// in its handler in `common_params_parser_init` (common/arg.cpp:1344 @cd959bae9),
    /// so the last one given wins.
    grammar_source: GrammarSource = .none,
    /// The GBNF `grammar_source` resolves to, filled in by `main.zig` before
    /// the model loads -- upstream converts a schema while it parses the
    /// command line -- and empty for none.
    grammar: []const u8 = "",

    /// `--reasoning-format FORMAT`: which parser the template's reply is read
    /// with -- `none` leaves the thinking in the answer, `deepseek` and
    /// `auto` take it out, `deepseek-legacy` too. `common_params`' default.
    reasoning_format: common_chat.ReasoningFormat = .deepseek,
    /// `-rea, --reasoning on|off|auto`: upstream's `enable_reasoning`, 1, 0
    /// or -1. `on` and `off` also set the `enable_thinking` kwarg.
    enable_reasoning: i8 = -1,
    /// `--reasoning-budget N`: tokens the model may think for; -1 is
    /// unlimited, 0 closes the block at once.
    reasoning_budget: i32 = -1,
    /// `--reasoning-budget-message MESSAGE`: written before the closing tag
    /// when the budget forces it.
    reasoning_budget_message: []const u8 = "",
    /// `--skip-chat-parsing` / `--no-skip-chat-parsing`: render with the
    /// template but read every reply as plain content -- upstream's
    /// `force_pure_content_parser`.
    skip_chat_parsing: bool = false,
    /// What `--chat-template-kwargs`, `--reasoning on|off`,
    /// `--reasoning-effort` and `--reasoning-preserve` did to upstream's
    /// `default_template_kwargs`, in command-line order, which
    /// `conversation.Kwargs` replays: a later write to a key wins, as the
    /// `std::map`'s assignment does. The JSON is parsed there, before the model
    /// loads, as upstream parses it with the command line.
    kwarg_ops: [max_kwarg_ops]KwargOp = undefined,
    n_kwarg_ops: usize = 0,
    /// `-co, --color on|off|auto`: grey reasoning, as upstream's console
    /// writes it. `auto` asks `tty_can_use_colors`, in `main.zig`.
    color: Color = .auto,

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

    /// The kwarg writes, oldest first.
    pub fn kwargOps(self: *const Args) []const KwargOp {
        return self.kwarg_ops[0..self.n_kwarg_ops];
    }
};

/// How many kwarg writes the command line may make; a longer one is refused
/// rather than truncated.
pub const max_kwarg_ops = 32;

/// One write to upstream's `default_template_kwargs`, which maps a name to
/// the JSON text of its value. Borrowed from `argv`, except the literals.
pub const KwargOp = union(enum) {
    /// `--chat-template-kwargs JSON`: every member of the object.
    object: []const u8,
    /// `name = value`, the value already JSON text (`--reasoning on` writes
    /// `enable_thinking` = `true`).
    set: struct { key: []const u8, value: []const u8 },
    /// `--reasoning-effort LEVEL`: `reasoning_effort` = the level as a JSON
    /// string, or `default`, which erases it.
    effort: []const u8,
};

/// `--color`'s three answers.
pub const Color = enum { auto, on, off };

/// Where a grammar comes from, as the four flags that set one say. A value is
/// borrowed from `argv`.
pub const GrammarSource = union(enum) {
    none,
    /// `--grammar GRAMMAR`: GBNF, as given (`COMMON_GRAMMAR_TYPE_USER`).
    gbnf: []const u8,
    /// `--grammar-file FNAME`: GBNF, read from the file.
    gbnf_file: []const u8,
    /// `-j, --json-schema SCHEMA`: a JSON schema, converted
    /// (`COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT`).
    schema: []const u8,
    /// `-jf, --json-schema-file FILE`: the same, read from the file.
    schema_file: []const u8,

    /// The flag that set it, as upstream names it in an error.
    pub fn flag(self: GrammarSource) []const u8 {
        return switch (self) {
            .none => "",
            .gbnf => "--grammar",
            .gbnf_file => "--grammar-file",
            .schema => "--json-schema",
            .schema_file => "--json-schema-file",
        };
    }
};

/// What went wrong, as distinct from how it is reported.
pub const Error = error{
    MissingModel,
    MissingValue,
    UnknownFlag,
    UnsupportedFlag,
    BadNumber,
    /// A value the flag's handler throws on: `report` writes upstream's
    /// `error while handling argument` line with its `what()`.
    BadArgument,
    /// More than `max_kwarg_ops` template kwarg writes.
    TooManyKwargs,
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
    /// For `BadArgument`, the value the flag was given.
    value: []const u8 = "",

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
            error.BadArgument => {
                try w.print("error while handling argument \"{s}\": ", .{self.arg});
                try writeWhat(w, self.arg, self.value);
                try w.print("\n", .{});
            },
            error.TooManyKwargs => try w.print(
                "error: {s}: more than {d} template kwarg flags\n",
                .{ self.arg, max_kwarg_ops },
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
    .{ .names = &.{ "-fit", "--fit" }, .value = "on|off", .help = "fit unset context and layers to free memory (default: on)" },
    .{ .names = &.{ "-fitt", "--fit-target" }, .value = "MiB0,MiB1,...", .help = "memory --fit leaves free per device (default: 1024)" },
    .{ .names = &.{ "-fitc", "--fit-ctx" }, .value = "N", .help = "least context --fit shrinks to (default: 4096)" },
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
    .{ .names = &.{"--grammar"}, .value = "GRAMMAR", .help = "BNF-like grammar to constrain generations" },
    .{ .names = &.{"--grammar-file"}, .value = "FNAME", .help = "file to read grammar from" },
    .{ .names = &.{ "-j", "--json-schema" }, .value = "SCHEMA", .help = "JSON schema to constrain generations" },
    .{ .names = &.{ "-jf", "--json-schema-file" }, .value = "FILE", .help = "file containing a JSON schema to constrain generations" },
    .{ .names = &.{"--reasoning-format"}, .value = "FORMAT", .help = "how thinking is read off a reply: none, auto, deepseek, deepseek-legacy (default: deepseek)" },
    .{ .names = &.{ "-rea", "--reasoning" }, .value = "on|off|auto", .help = "use reasoning in the chat (default: auto, from the template)" },
    .{ .names = &.{"--reasoning-budget"}, .value = "N", .help = "token budget for thinking: -1 unlimited, 0 immediate end (default: -1)" },
    .{ .names = &.{"--reasoning-budget-message"}, .value = "MESSAGE", .help = "written before the end-of-thinking tag when the budget is spent" },
    .{ .names = &.{"--reasoning-effort"}, .value = "LEVEL", .help = "reasoning effort given to the template, or 'default'" },
    .{ .names = &.{ "--reasoning-preserve", "--no-reasoning-preserve" }, .help = "keep earlier turns' reasoning in the history (default: on)" },
    .{ .names = &.{ "--skip-chat-parsing", "--no-skip-chat-parsing" }, .help = "read every reply as plain content, thinking included (default: off)" },
    .{ .names = &.{"--chat-template-kwargs"}, .value = "STRING", .help = "JSON object of extra template variables" },
    .{ .names = &.{ "-co", "--color" }, .value = "on|off|auto", .help = "grey the reasoning (default: auto, when a terminal)" },
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
            // The full context, asked for explicitly: `--fit` must not shrink it.
            if (args.n_ctx == 0) args.fit_min_ctx = std.math.maxInt(u32);
        } else if (eq(a, &.{ "-fit", "--fit" })) {
            const v = value(argv, &i) orelse return missing(a);
            if (isTruthy(v)) {
                args.fit = true;
            } else if (isFalsey(v)) {
                args.fit = false;
            } else return bad(v);
        } else if (eq(a, &.{ "-fitt", "--fit-target" })) {
            const v = value(argv, &i) orelse return missing(a);
            parseFitTarget(v, &args.fit_target) catch return bad(v);
        } else if (eq(a, &.{ "-fitc", "--fit-ctx" })) {
            args.fit_min_ctx = @bitCast(parseI32(value(argv, &i) orelse return missing(a)) catch return bad(a));
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
        } else if (eq(a, &.{"--grammar"})) {
            args.grammar_source = .{ .gbnf = value(argv, &i) orelse return missing(a) };
        } else if (eq(a, &.{"--grammar-file"})) {
            args.grammar_source = .{ .gbnf_file = value(argv, &i) orelse return missing(a) };
        } else if (eq(a, &.{ "-j", "--json-schema" })) {
            args.grammar_source = .{ .schema = value(argv, &i) orelse return missing(a) };
        } else if (eq(a, &.{ "-jf", "--json-schema-file" })) {
            args.grammar_source = .{ .schema_file = value(argv, &i) orelse return missing(a) };
        } else if (eq(a, &.{"--reasoning-format"})) {
            const v = value(argv, &i) orelse return missing(a);
            args.reasoning_format = common_chat.ReasoningFormat.fromName(v) catch return badArg(a, v);
            args.chat = true;
        } else if (eq(a, &.{ "-rea", "--reasoning" })) {
            const v = value(argv, &i) orelse return missing(a);
            if (isTruthy(v)) {
                args.enable_reasoning = 1;
                if (!pushKwarg(&args, .{ .set = .{ .key = "enable_thinking", .value = "true" } })) return tooMany(a);
            } else if (isFalsey(v)) {
                args.enable_reasoning = 0;
                if (!pushKwarg(&args, .{ .set = .{ .key = "enable_thinking", .value = "false" } })) return tooMany(a);
            } else if (isAutoy(v)) {
                args.enable_reasoning = -1;
            } else return badArg(a, v);
            args.chat = true;
        } else if (eq(a, &.{"--reasoning-budget"})) {
            const v = value(argv, &i) orelse return missing(a);
            args.reasoning_budget = parseI32(v) catch return bad(a);
            if (args.reasoning_budget < -1) return badArg(a, v);
            args.chat = true;
        } else if (eq(a, &.{"--reasoning-budget-message"})) {
            args.reasoning_budget_message = value(argv, &i) orelse return missing(a);
            args.chat = true;
        } else if (eq(a, &.{"--reasoning-effort"})) {
            const v = value(argv, &i) orelse return missing(a);
            if (!pushKwarg(&args, .{ .effort = v })) return tooMany(a);
            args.chat = true;
        } else if (eq(a, &.{ "--reasoning-preserve", "--no-reasoning-preserve" })) {
            const on = std.mem.eql(u8, a, "--reasoning-preserve");
            if (!pushKwarg(&args, .{ .set = .{ .key = "preserve_reasoning", .value = if (on) "true" else "false" } })) return tooMany(a);
            args.chat = true;
        } else if (eq(a, &.{ "--skip-chat-parsing", "--no-skip-chat-parsing" })) {
            args.skip_chat_parsing = std.mem.eql(u8, a, "--skip-chat-parsing");
            args.chat = true;
        } else if (eq(a, &.{"--chat-template-kwargs"})) {
            const v = value(argv, &i) orelse return missing(a);
            if (!pushKwarg(&args, .{ .object = v })) return tooMany(a);
            args.chat = true;
        } else if (eq(a, &.{ "-co", "--color" })) {
            const v = value(argv, &i) orelse return missing(a);
            args.color = if (isTruthy(v)) .on else if (isFalsey(v)) .off else if (isAutoy(v)) .auto else return badArg(a, v);
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

fn tooMany(arg: []const u8) Result {
    return .{ .fail = .{ .err = error.TooManyKwargs, .arg = arg } };
}

fn badArg(arg: []const u8, v: []const u8) Result {
    return .{ .fail = .{ .err = error.BadArgument, .arg = arg, .value = v } };
}

/// Records a kwarg write; false when the command line has made too many.
/// The caller reports that as `tooMany`.
fn pushKwarg(args: *Args, op: KwargOp) bool {
    if (args.n_kwarg_ops == max_kwarg_ops) return false;
    args.kwarg_ops[args.n_kwarg_ops] = op;
    args.n_kwarg_ops += 1;
    return true;
}

/// The `what()` of the exception the flag's handler in
/// `common_params_parser_init` throws on `v` (common/arg.cpp @cd959bae9):
/// `common_reasoning_format_from_name`'s, `--reasoning`'s and `--color`'s
/// own, and `--reasoning-budget`'s "invalid value".
fn writeWhat(w: *std.Io.Writer, flag: []const u8, v: []const u8) !void {
    if (std.mem.eql(u8, flag, "--reasoning-format")) {
        try w.print("Unknown reasoning format: {s}", .{v});
    } else if (std.mem.eql(u8, flag, "--reasoning-budget")) {
        try w.print("invalid value", .{});
    } else {
        // `--reasoning` and `--color`; the handler's message carries its own newline.
        const name = if (eq(flag, &.{ "-rea", "--reasoning" })) "--reasoning" else "--color";
        try w.print("error: unknown value for {s}: '{s}'\n", .{ name, v });
    }
}

/// `llama_max_devices` (src/llama.cpp:86 @cd959bae9), which sizes upstream's
/// `fit_params_target`.
pub const max_devices = 16;

/// Ports `is_truthy` (arg.cpp:1284 @cd959bae9), in `common_arg_utils`.
fn isTruthy(v: []const u8) bool {
    return eq(v, &.{ "on", "enabled", "true", "1" });
}

/// Ports `is_falsey` (arg.cpp:1288 @cd959bae9), in `common_arg_utils`.
fn isFalsey(v: []const u8) bool {
    return eq(v, &.{ "off", "disabled", "false", "0" });
}

/// Ports `is_autoy` (arg.cpp:1292 @cd959bae9), in `common_arg_utils`.
fn isAutoy(v: []const u8) bool {
    return eq(v, &.{ "auto", "-1" });
}

/// Parses `-fitt`'s value as upstream's handler for it in `common/arg.cpp`
/// does: split on runs of `,` and `/`, as `std::sregex_token_iterator`
/// over `[,/]+` does -- a leading separator gives an empty first value, a
/// trailing one none -- each value read by `std::stoull` and taken as MiB.
/// One value applies to every device; more set the first devices.
///
/// Parameters:
/// - `v`: the flag's value.
/// - `target`: updated in place.
///
/// Return: `error.BadNumber` for a value `std::stoull` rejects, or for
/// `max_devices` values or more.
fn parseFitTarget(v: []const u8, target: *[max_devices]usize) !void {
    // With no separator at all the iterator yields the whole value once,
    // empty or not; `std::stoull("")` then throws.
    if (v.len == 0) return error.BadNumber;
    var parts: [max_devices]u64 = undefined;
    var n: usize = 0;
    var rest = v;
    while (rest.len > 0) {
        const end = std.mem.indexOfAny(u8, rest, ",/") orelse rest.len;
        if (n == max_devices) return error.BadNumber;
        parts[n] = try stoull(rest[0..end]);
        n += 1;
        rest = rest[end..];
        while (rest.len > 0 and (rest[0] == ',' or rest[0] == '/')) rest = rest[1..];
    }
    if (n >= max_devices) return error.BadNumber;
    const mib = 1024 * 1024;
    if (n == 1) {
        target.* = @splat(parts[0] *% mib);
        return;
    }
    for (parts[0..n], 0..) |p, k| target[k] = p *% mib;
}

/// `std::stoull(s)` in base 10: leading white space, an optional sign (a
/// minus negates modulo 2^64), at least one digit, and whatever follows the
/// digits ignored.
fn stoull(s: []const u8) !u64 {
    var k: usize = 0;
    while (k < s.len and std.ascii.isWhitespace(s[k])) k += 1;
    var negative = false;
    if (k < s.len and (s[k] == '+' or s[k] == '-')) {
        negative = s[k] == '-';
        k += 1;
    }
    const start = k;
    var r: u64 = 0;
    while (k < s.len and std.ascii.isDigit(s[k])) : (k += 1) {
        r = std.math.mul(u64, r, 10) catch return error.BadNumber;
        r = std.math.add(u64, r, s[k] - '0') catch return error.BadNumber;
    }
    if (k == start) return error.BadNumber;
    return if (negative) 0 -% r else r;
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
    var buf: [16384]u8 = undefined;
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
    try testing.expect(a.fit);
    try testing.expectEqual(@as(u32, 4096), a.fit_min_ctx);
    try testing.expectEqual(@as(usize, 1024 * 1024 * 1024), a.fit_target[max_devices - 1]);
}

test "max_devices is libllama's" {
    try testing.expectEqual(@as(usize, max_devices), @import("c.zig").api.llama_max_devices());
}

test "--fit takes upstream's truthy and falsey words and nothing else" {
    for ([_][]const u8{ "on", "enabled", "true", "1" }) |v| {
        const r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "-fit", "off", "--fit", v });
        try testing.expect(r.ok.fit);
    }
    for ([_][]const u8{ "off", "disabled", "false", "0" }) |v| {
        const r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "-fit", v });
        try testing.expect(!r.ok.fit);
    }
    const r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "-fit", "yes" });
    try testing.expectEqual(error.BadNumber, r.fail.err);
}

test "an explicit -c 0 forbids --fit from shrinking, and the later flag wins" {
    var r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "-c", "0" });
    try testing.expectEqual(@as(u32, std.math.maxInt(u32)), r.ok.fit_min_ctx);
    r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "-c", "0", "-fitc", "2048" });
    try testing.expectEqual(@as(u32, 2048), r.ok.fit_min_ctx);
    r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "--fit-ctx", "2048", "-c", "0" });
    try testing.expectEqual(@as(u32, std.math.maxInt(u32)), r.ok.fit_min_ctx);
    r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "-c", "512" });
    try testing.expectEqual(@as(u32, 4096), r.ok.fit_min_ctx);
    r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "-fitc", "-1" });
    try testing.expectEqual(@as(u32, std.math.maxInt(u32)), r.ok.fit_min_ctx);
}

test "--fit-target splits as upstream's regex does and reads values as std::stoull" {
    const mib: usize = 1024 * 1024;
    var r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "-fitt", "512" });
    for (r.ok.fit_target) |t| try testing.expectEqual(512 * mib, t);
    r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "--fit-target", " +256,,/128x," });
    try testing.expectEqual(256 * mib, r.ok.fit_target[0]);
    try testing.expectEqual(128 * mib, r.ok.fit_target[1]);
    try testing.expectEqual(1024 * mib, r.ok.fit_target[2]);
    r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "-fitt", "-1" });
    try testing.expectEqual(@as(usize, 0) -% mib, r.ok.fit_target[0]);
    for ([_][]const u8{ "", ",512", "x", "512,y", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16", "99999999999999999999" }) |v| {
        r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "-fitt", v });
        try testing.expectEqual(error.BadNumber, r.fail.err);
    }
    r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "-fitt", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" });
    try testing.expectEqual(15 * mib, r.ok.fit_target[14]);
    try testing.expectEqual(1024 * mib, r.ok.fit_target[15]);
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

test "v0.6.0's new flags are real upstream flags this CLI does not take" {
    // `--spec-draft-sampling` (no speculative decoding here), `--rpc` (no RPC
    // backend) and `--nextn` (llama-imatrix's) are refused as ours to add,
    // not as typos: cli/upstream_flags.zig lists them at cd959bae9.
    for ([_][]const u8{ "--spec-draft-sampling", "--rpc", "--nextn" }) |f| {
        const r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", f, "x" });
        try testing.expectEqual(Error.UnsupportedFlag, r.fail.err);
    }
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
    var buf: [16384]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try help(&w);
    const text = w.buffered();
    for (specs) |s| {
        for (s.names) |n| {
            if (std.mem.find(u8, text, n) == null) {
                std.debug.print("help omits {s}\n", .{n});
                return error.FlagMissingFromHelp;
            }
        }
    }
}

test "the last grammar flag wins, whichever kind it is" {
    const r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "--grammar", "root ::= \"a\"", "-j", "{}" });
    try testing.expectEqualStrings("{}", r.ok.grammar_source.schema);
    const f = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "-jf", "s.json", "--grammar-file", "g.gbnf" });
    try testing.expectEqualStrings("g.gbnf", f.ok.grammar_source.gbnf_file);
    try testing.expectEqualStrings("--grammar-file", f.ok.grammar_source.flag());
    const m = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "--json-schema" });
    try testing.expectEqual(Error.MissingValue, m.fail.err);
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

test "the reasoning flags take upstream's values, imply chat, and refuse the rest" {
    const r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "--reasoning-format", "none", "-rea", "off", "--reasoning-budget", "0", "--skip-chat-parsing" });
    try testing.expectEqual(common_chat.ReasoningFormat.none, r.ok.reasoning_format);
    try testing.expectEqual(@as(i8, 0), r.ok.enable_reasoning);
    try testing.expectEqual(@as(i32, 0), r.ok.reasoning_budget);
    try testing.expect(r.ok.skip_chat_parsing);
    try testing.expect(r.ok.useChatTemplate());
    // `--reasoning off` writes the kwarg as well as the switch.
    try testing.expectEqualStrings("enable_thinking", r.ok.kwargOps()[0].set.key);
    try testing.expectEqualStrings("false", r.ok.kwargOps()[0].set.value);

    const defaults = try parseForTest(&.{ "llamazig", "-m", "m.gguf" });
    try testing.expectEqual(common_chat.ReasoningFormat.deepseek, defaults.ok.reasoning_format);
    try testing.expectEqual(@as(i8, -1), defaults.ok.enable_reasoning);
    try testing.expectEqual(Color.auto, defaults.ok.color);

    for ([_][2][]const u8{
        .{ "--reasoning-format", "bogus" },
        .{ "--reasoning", "maybe" },
        .{ "--reasoning-budget", "-2" },
        .{ "--color", "sometimes" },
    }) |fv| {
        const f = try parseForTest(&.{ "llamazig", "-m", "m.gguf", fv[0], fv[1] });
        try testing.expectEqual(Error.BadArgument, f.fail.err);
    }
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    const f = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "--reasoning", "maybe" });
    try f.fail.report(&w);
    try testing.expectEqualStrings("error while handling argument \"--reasoning\": error: unknown value for --reasoning: 'maybe'\n\n", w.buffered());
}

test "the kwarg flags are kept in command-line order" {
    const r = try parseForTest(&.{ "llamazig", "-m", "m.gguf", "--chat-template-kwargs", "{\"a\":1}", "--reasoning-effort", "high", "--no-reasoning-preserve", "--reasoning", "auto" });
    const ops = r.ok.kwargOps();
    try testing.expectEqual(@as(usize, 3), ops.len);
    try testing.expectEqualStrings("{\"a\":1}", ops[0].object);
    try testing.expectEqualStrings("high", ops[1].effort);
    try testing.expectEqualStrings("false", ops[2].set.value);
}
