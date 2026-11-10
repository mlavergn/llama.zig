//! One generation's chat state: the conversation applied through the
//! templates once, and the model's reply fed back in pieces and parsed
//! after each, the last good message kept.
//!
//! # Provenance
//!
//! Ports `common_chat_session_params`, `common_chat_session` and its
//! members from `llama.cpp/common/chat.h` and `llama.cpp/common/chat.cpp`,
//! and `common_sampling_add_preserved_tokens` and
//! `common_sampling_add_grammar_triggers` from
//! `llama.cpp/common/sampling.cpp` (llama.cpp v0.9.0, `cd959bae9`), which
//! the session's constructor calls. Those two live here because
//! `tools_common` does not import this module; `tools/common/sampling.zig`
//! has its own over its own sampling parameters. Each declaration names
//! what it ports and the line it began at.
//!
//! # Ownership
//!
//! A `Session` owns everything it holds: what the templates produced, in
//! an arena of its own, and the reply and the message, through the
//! allocator it was made with. `deinit` frees it all; `clone` is the C++
//! copy.
//!
//! # libllama
//!
//! The vocabulary is reached through libllama's port (`input.zig`), never a
//! `llama_*` C symbol.

const std = @import("std");
const peg = @import("peg");
const exception = @import("exception.zig");
const msg_mod = @import("msg.zig");
const params_mod = @import("params.zig");
const prompt_mod = @import("prompt.zig");
const input_mod = @import("input.zig");
const parse_mod = @import("parse.zig");
const templates_mod = @import("templates.zig");
const util = @import("util.zig");

const Allocator = std.mem.Allocator;
const Error = exception.Error;
const Msg = msg_mod.Msg;
const Input = input_mod.Input;
const Prompt = prompt_mod.Prompt;
const GrammarTrigger = params_mod.GrammarTrigger;

/// Ports `common_sampling_add_preserved_tokens` (sampling.cpp:1055
/// @cd959bae9): each string that is a single token in the vocabulary added
/// to `preserved`, a `std::set` kept sorted and unique.
///
/// Parameters:
/// - `vocab`: a `llama_vocab *`, only passed through.
/// - `preserved`: grown with `a`.
pub fn addPreservedTokens(a: Allocator, preserved: *std.ArrayList(i32), vocab: *const anyopaque, tokens: []const []const u8) Error!void {
    for (tokens) |t| {
        const ids = try input_mod.commonTokenize(a, vocab, t, false, true);
        defer a.free(ids);
        if (ids.len == 1) try setInsert(a, preserved, ids[0]);
    }
}

/// `std::set<llama_token>::insert`.
fn setInsert(a: Allocator, set: *std.ArrayList(i32), id: i32) Error!void {
    const at = std.sort.lowerBound(i32, set.items, id, struct {
        fn order(x: i32, y: i32) std.math.Order {
            return std.math.order(x, y);
        }
    }.order);
    if (at < set.items.len and set.items[at] == id) return;
    try set.insert(a, at, id);
}

/// Ports `common_sampling_add_grammar_triggers` (sampling.cpp:1065
/// @cd959bae9): the triggers added to `out`, a trigger word that is a
/// single token becoming a token trigger -- which must then be one of
/// `preserved`.
///
/// Parameters:
/// - `vocab`: a `llama_vocab *`, only passed through.
/// - `grammar_lazy`: a lazy grammar needs a trigger.
///
/// Return: `error.RuntimeError` for a single-token word that is not
/// preserved, or for a lazy grammar left without triggers.
pub fn addGrammarTriggers(a: Allocator, out: *std.ArrayList(GrammarTrigger), preserved: []const i32, grammar_lazy: bool, vocab: *const anyopaque, triggers: []const GrammarTrigger) Error!void {
    for (triggers) |trigger| {
        if (trigger.type == .word) {
            const word = trigger.value;
            const ids = try input_mod.commonTokenize(a, vocab, word, false, true);
            defer a.free(ids);
            if (ids.len == 1) {
                const token = ids[0];
                if (std.mem.indexOfScalar(i32, preserved, token) == null) {
                    return exception.throw(error.RuntimeError, "Grammar trigger word should be marked as preserved token: {s}", .{word});
                }
                try out.append(a, .{ .type = .token, .value = word, .token = token });
            } else {
                try out.append(a, .{ .type = .word, .value = word });
            }
        } else {
            try out.append(a, trigger);
        }
    }
    if (grammar_lazy and out.items.len == 0) {
        return exception.throw(error.RuntimeError, "Error: no triggers set for lazy grammar!", .{});
    }
}

/// Ports `common_chat_session_params` (chat.h:397 @cd959bae9).
pub const SessionParams = struct {
    echo: bool = false, // include the assistant prefill in the output when continuing a message
    debug: bool = false, // enable debug output for the PEG parser
};

/// Ports `common_chat_session` (chat.h:402 @cd959bae9).
pub const Session = struct {
    gpa: Allocator,
    /// Everything the templates produced; never freed singly.
    arena: *std.heap.ArenaAllocator,

    prompt_parts: Prompt = .{},
    grammar_text: []const u8 = "",
    grammar_lazy: bool = false,
    grammar_triggers: std.ArrayList(GrammarTrigger) = .empty,
    /// A `std::set<llama_token>`: sorted, unique.
    preserved_tokens: std.ArrayList(i32) = .empty,
    stops: std.ArrayList([]const u8) = .empty,
    generation_prompt_text: []const u8 = "",
    thinking_start: []const u8 = "",
    thinking_ends: std.ArrayList([]const u8) = .empty,

    /// `parser_params`: its format, generation prompt and parser, kept
    /// apart so the session can move -- `parserParams` points at them.
    parser_format: msg_mod.Format = .content_only,
    parser_generation_prompt: Input,
    parser_debug: bool = false,
    parser_arena: ?peg.Arena = null,

    delimiters: msg_mod.MsgDelimiters = .{},
    input: Input,
    result: Msg = .{},
    templated: bool = false,
    finished: bool = false,

    /// Ports `common_chat_session` (chat.h:404 @cd959bae9), the default
    /// constructor: no template; feeding it parses content only.
    ///
    /// Return: owned by `gpa`; release with `deinit`.
    pub fn initEmpty(gpa: Allocator) Error!Session {
        const arena = try gpa.create(std.heap.ArenaAllocator);
        arena.* = .init(gpa);
        var self: Session = .{
            .gpa = gpa,
            .arena = arena,
            .parser_generation_prompt = .init(arena.allocator()),
            .input = .init(gpa),
        };
        errdefer self.deinit();
        try Msg.setField(gpa, &self.result.role, "assistant");
        return self;
    }

    /// Ports `common_chat_session` (chat.cpp:1868 @cd959bae9), the
    /// constructor: the inputs applied through `tmpls`; with a vocabulary,
    /// the generation prompt tokenized, the preserved tokens and grammar
    /// triggers resolved and the message delimiters tokenized. When continuing a message and not
    /// echoing it, the prefill is parsed so it is not emitted as part of
    /// the first delta.
    ///
    /// Parameters:
    /// - `vocab`: a `llama_vocab *` from whichever `llama.h` import the
    ///   caller has, or null.
    ///
    /// Return: owned by `gpa`; release with `deinit`.
    pub fn init(gpa: Allocator, tmpls: *const templates_mod.Templates, vocab: ?*const anyopaque, inputs: *const templates_mod.TemplatesInputs, params: SessionParams) Error!Session {
        var self = try initEmpty(gpa);
        errdefer self.deinit();
        const a = self.arena.allocator();

        const applied = try templates_mod.apply(a, tmpls, inputs);

        self.templated = true;
        self.prompt_parts = applied.prompt;

        self.grammar_text = applied.grammar;
        self.grammar_lazy = applied.grammar_lazy;
        self.stops = applied.additional_stops;
        self.generation_prompt_text = applied.generation_prompt;
        self.thinking_start = applied.thinking_start_tag;
        self.thinking_ends = applied.thinking_end_tags;

        self.parser_format = applied.format;
        self.parser_generation_prompt = if (vocab) |v|
            try input_mod.tokenize(a, v, applied.generation_prompt)
        else
            try Input.fromText(a, applied.generation_prompt);
        self.parser_debug = params.debug;
        self.parser_arena = applied.parser;

        self.delimiters = applied.message_delimiters;

        if (vocab) |v| {
            // `common_params_sampling resolved` with `grammar_lazy` set
            try addPreservedTokens(a, &self.preserved_tokens, v, applied.preserved_tokens.items);
            try addGrammarTriggers(a, &self.grammar_triggers, self.preserved_tokens.items, applied.grammar_lazy, v, applied.grammar_triggers.items);

            const Tok = struct {
                fn run(vv: *const anyopaque, al: Allocator, text: []const u8) anyerror!std.ArrayList(i32) {
                    return .fromOwnedSlice(try input_mod.commonTokenize(al, vv, text, false, true));
                }
            };
            self.delimiters.tokenize(a, v, Tok.run) catch |e| return switch (e) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.RuntimeError,
            };
        } else {
            self.grammar_triggers = applied.grammar_triggers;
        }

        if (inputs.continue_final_message != .none and !params.echo) {
            // start from the prefill so it is not emitted as part of the first delta
            const m = try parse_mod.parse(gpa, &self.input, true, &self.parserParams());
            self.result.deinit(gpa);
            self.result = m;
        }
        return self;
    }

    pub fn deinit(self: *Session) void {
        self.input.deinit();
        self.result.deinit(self.gpa);
        self.arena.deinit();
        self.gpa.destroy(self.arena);
        self.* = undefined;
    }

    /// The C++ copy.
    ///
    /// Return: owned by `gpa`; release with `deinit`.
    pub fn clone(self: *const Session, gpa: Allocator) Error!Session {
        var out = try initEmpty(gpa);
        errdefer out.deinit();
        const a = out.arena.allocator();
        for (self.prompt_parts.parts.items) |part| {
            try out.prompt_parts.parts.append(a, .{ .text = try a.dupe(u8, part.text), .is_input = part.is_input });
        }
        out.grammar_text = try a.dupe(u8, self.grammar_text);
        out.grammar_lazy = self.grammar_lazy;
        for (self.grammar_triggers.items) |t| {
            try out.grammar_triggers.append(a, .{ .type = t.type, .value = try a.dupe(u8, t.value), .token = t.token });
        }
        try out.preserved_tokens.appendSlice(a, self.preserved_tokens.items);
        for (self.stops.items) |s| try out.stops.append(a, try a.dupe(u8, s));
        out.generation_prompt_text = try a.dupe(u8, self.generation_prompt_text);
        out.thinking_start = try a.dupe(u8, self.thinking_start);
        for (self.thinking_ends.items) |s| try out.thinking_ends.append(a, try a.dupe(u8, s));
        out.parser_format = self.parser_format;
        out.parser_generation_prompt = try self.parser_generation_prompt.clone(a);
        out.parser_debug = self.parser_debug;
        if (self.parser_arena) |*p| out.parser_arena = p.clone(a) catch |e| return util.fromPeg(e);
        for (self.delimiters.delimiters.items) |d| {
            try out.delimiters.delimiters.append(a, .{
                .role = d.role,
                .delimiter = try a.dupe(u8, d.delimiter),
                .tokens = .fromOwnedSlice(try a.dupe(i32, d.tokens.items)),
            });
        }
        out.input.deinit();
        out.input = try self.input.clone(gpa);
        out.result.deinit(gpa);
        out.result = try self.result.clone(gpa);
        out.templated = self.templated;
        out.finished = self.finished;
        return out;
    }

    /// `parser_params`, pointing into this session: valid while it stays
    /// where it is and unchanged.
    pub fn parserParams(self: *const Session) parse_mod.ParserParams {
        return .{
            .format = self.parser_format,
            .generation_prompt = &self.parser_generation_prompt,
            .debug = self.parser_debug,
            .parser = if (self.parser_arena) |*p| p else null,
        };
    }

    /// Ports `prompt` (chat.h:411 @cd959bae9).
    pub fn prompt(self: *const Session) *const Prompt {
        return &self.prompt_parts;
    }

    /// Ports `format` (chat.h:412 @cd959bae9).
    pub fn format(self: *const Session) msg_mod.Format {
        return self.parser_format;
    }

    /// Ports `msg` (chat.h:413 @cd959bae9).
    pub fn msg(self: *const Session) *const Msg {
        return &self.result;
    }

    /// Ports `parser` (chat.h:414 @cd959bae9): null for the empty arena.
    pub fn parser(self: *const Session) ?*const peg.Arena {
        return if (self.parser_arena) |*p| p else null;
    }

    /// Ports `grammar` (chat.h:416 @cd959bae9).
    pub fn grammar(self: *const Session) []const u8 {
        return self.grammar_text;
    }

    /// Ports `generation_prompt` (chat.h:417 @cd959bae9).
    pub fn generationPrompt(self: *const Session) []const u8 {
        return self.generation_prompt_text;
    }

    /// Ports `thinking_start_tag` (chat.h:418 @cd959bae9).
    pub fn thinkingStartTag(self: *const Session) []const u8 {
        return self.thinking_start;
    }

    /// Ports `thinking_end_tags` (chat.h:419 @cd959bae9).
    pub fn thinkingEndTags(self: *const Session) []const []const u8 {
        return self.thinking_ends.items;
    }

    /// Ports `additional_stops` (chat.h:420 @cd959bae9).
    pub fn additionalStops(self: *const Session) []const []const u8 {
        return self.stops.items;
    }

    /// Ports `message_delimiters` (chat.h:422 @cd959bae9).
    pub fn messageDelimiters(self: *const Session) *const msg_mod.MsgDelimiters {
        return &self.delimiters;
    }

    /// Ports `has_template` (chat.h:426 @cd959bae9).
    pub fn hasTemplate(self: *const Session) bool {
        return self.templated;
    }

    /// Ports `apply_sampling` (chat.cpp:1912 @cd959bae9): the grammar,
    /// triggers, preserved tokens and generation prompt into a caller's
    /// `common_params_sampling`, nothing without a template.
    ///
    /// Parameters:
    /// - `a`: grows `sampling`'s lists and owns the strings copied in.
    /// - `sampling`: a pointer to a struct with `grammar_type`, `grammar`,
    ///   `grammar_lazy`, `generation_prompt`, `preserved_tokens` (sorted,
    ///   unique) and `grammar_triggers`, as `tools_common`'s `Sampling`.
    pub fn applySampling(self: *const Session, a: Allocator, sampling: anytype) Error!void {
        if (!self.templated) return;
        const S = @typeInfo(@TypeOf(sampling)).pointer.child;
        if (self.grammar_text.len != 0) {
            sampling.grammar_type = .tool_calls;
            sampling.grammar = try a.dupe(u8, self.grammar_text);
        }
        sampling.grammar_lazy = self.grammar_lazy;
        sampling.generation_prompt = try a.dupe(u8, self.generation_prompt_text);
        for (self.preserved_tokens.items) |id| try setInsert(a, &sampling.preserved_tokens, id);
        const T = @typeInfo(@FieldType(S, "grammar_triggers").Slice).pointer.child;
        const TT = @FieldType(T, "type");
        for (self.grammar_triggers.items) |t| {
            try sampling.grammar_triggers.append(a, .{
                .type = @as(TT, @fromBackingInt(@backingInt(t.type))),
                .value = try a.dupe(u8, t.value),
                .token = t.token,
            });
        }
    }

    /// Ports `feed` (chat.cpp:1925 @cd959bae9): `chunk` appended to the
    /// reply, which is parsed as partial; the message is replaced only by a
    /// parse that is not empty.
    ///
    /// Return: the message, borrowed until the next `feed` or `finish`.
    pub fn feed(self: *Session, chunk: *const Input) Error!*const Msg {
        util.ggmlAssert(!self.finished, "!finished && \"feed() after finish()\"");
        try self.input.appendInput(chunk);
        return self.update(true);
    }

    /// Ports `finish` (chat.cpp:1935 @cd959bae9): the last `chunk`, and the
    /// whole reply parsed as complete.
    ///
    /// Return: the message, borrowed until `deinit`.
    pub fn finish(self: *Session, chunk: ?*const Input) Error!*const Msg {
        util.ggmlAssert(!self.finished, "!finished && \"finish() called twice\"");
        self.finished = true;
        if (chunk) |c| try self.input.appendInput(c);
        return self.update(false);
    }

    fn update(self: *Session, is_partial: bool) Error!*const Msg {
        var m = try parse_mod.parse(self.gpa, &self.input, is_partial, &self.parserParams());
        if (!m.empty()) {
            self.result.deinit(self.gpa);
            self.result = m;
        } else {
            m.deinit(self.gpa);
        }
        return &self.result;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "an empty session parses what it is fed as content, and keeps the last message" {
    const a = std.testing.allocator;
    var s = try Session.initEmpty(a);
    defer s.deinit();
    try std.testing.expectEqualStrings("assistant", s.msg().role);
    var chunk = try Input.fromText(a, "Hel");
    defer chunk.deinit();
    _ = try s.feed(&chunk);
    var c2 = try Input.fromText(a, "lo");
    defer c2.deinit();
    var copy = try s.clone(a);
    defer copy.deinit();
    const m = try s.finish(&c2);
    try std.testing.expectEqualStrings("Hello", m.content);
    try std.testing.expectEqualStrings("Hel", copy.msg().content);
}

test "the preserved tokens are a sorted set" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var set: std.ArrayList(i32) = .empty;
    for ([_]i32{ 5, 1, 5, 3 }) |id| try setInsert(a, &set, id);
    try std.testing.expectEqualSlices(i32, &.{ 1, 3, 5 }, set.items);
}

test "applySampling copies the grammar fields into a sampling struct" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const Trigger = struct { type: enum(c_int) { token, word, pattern, pattern_full, _ }, value: []const u8 = "", token: i32 = -1 };
    const Sampling = struct {
        grammar_type: enum { none, user, output_format, tool_calls } = .none,
        grammar: []const u8 = "",
        grammar_lazy: bool = false,
        generation_prompt: []const u8 = "",
        preserved_tokens: std.ArrayList(i32) = .empty,
        grammar_triggers: std.ArrayList(Trigger) = .empty,
    };
    var s = try Session.initEmpty(std.testing.allocator);
    defer s.deinit();
    var sampling: Sampling = .{};
    try s.applySampling(a, &sampling);
    try std.testing.expectEqual(.none, sampling.grammar_type);
    s.templated = true;
    s.grammar_text = "root ::= x";
    s.grammar_lazy = true;
    try s.preserved_tokens.append(s.arena.allocator(), 7);
    try s.grammar_triggers.append(s.arena.allocator(), .{ .type = .word, .value = "<t>" });
    try sampling.preserved_tokens.append(a, 9);
    try s.applySampling(a, &sampling);
    try std.testing.expectEqual(.tool_calls, sampling.grammar_type);
    try std.testing.expectEqualStrings("root ::= x", sampling.grammar);
    try std.testing.expectEqualSlices(i32, &.{ 7, 9 }, sampling.preserved_tokens.items);
    try std.testing.expectEqual(@as(usize, 1), sampling.grammar_triggers.items.len);
    try std.testing.expectEqualStrings("<t>", sampling.grammar_triggers.items[0].value);
}
