//! The request-field table: each JSON field a completion request may carry,
//! its aliases and limits, and how it sets a task's parameters.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server-schema.cpp` and
//! `llama.cpp/tools/server/server-schema.h` (namespace `server_schema`),
//! and `common_sampler_types_from_names` and
//! `common_sampler_types_from_chars` from `llama.cpp/common/sampling.cpp`
//! (llama.cpp v0.7.0, `620d2225c`). Each declaration names what it ports and
//! the line it began at.
//!
//! # The field classes
//!
//! `field_num<T>`, `field_str`, `field_bool`, `field_json` and
//! `field_nested` are one `Field` with a `Kind`; `field_num<T>`'s four
//! instantiations (`int32_t`, `int64_t`, `uint32_t`, `float`) are `Num(T)`.
//! A `set_handler` lambda is a `Handler`: it captures the parameters and
//! `params_base` by reference, so both live in the `EvalContext` it gets.
//!
//! # Where a field's own evaluation is gated on the log level
//!
//! `SRV_DBG` and `SRV_TRC` evaluate their arguments only when the message
//! would print. The JSON schema's `dump(2)` can throw (on invalid UTF-8),
//! so whether a request is refused depends on the log verbosity, upstream
//! and here alike.
//!
//! # The chat fields
//!
//! Since v0.7.0 a chat request's format, grammar, triggers and parser live
//! in its chat session, which `Task.applyChatSession` applies; the schema
//! keeps `reasoning_format`, and `continue_final_message` and `echo` only
//! for their type checks.

const std = @import("std");
const json = @import("json");
const chat = @import("common_chat");
const tools = @import("tools_common");
const common = @import("common.zig");
const task = @import("task.zig");
const oaicompat = @import("oaicompat.zig");

const c = tools.c;
const log = tools.log;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Error = common.Error;
const Token = common.Token;
const throw = common.throw;
const TaskParams = task.TaskParams;
const SamplerType = task.SamplerType;
const GrammarTrigger = common.GrammarTrigger;

/// Ports `field_eval_context` (server-schema.h:18 @620d2225c), with what the
/// handlers' lambdas capture.
pub const EvalContext = struct {
    /// the task's arena: everything a handler sets is allocated here
    a: Allocator,
    params: *TaskParams,
    params_base: *const tools.params.Params,
    vocab: ?*const c.llama_vocab = null,
    logit_bias_eog: ?[]const c.llama_logit_bias = null,
};

/// Ports `field_handler` (server-schema.h:25 @620d2225c).
pub const Handler = *const fn (ctx: *EvalContext, data: *const Value) Error!void;

/// Ports `field_num` (server-schema.h:48 @620d2225c): a number, clamped to
/// `[min, max]`, or refused outside it when the limits are hard.
pub fn Num(comptime T: type) type {
    return struct {
        val: *T,
        min: T = if (@typeInfo(T) == .float) -std.math.floatMax(T) else std.math.minInt(T),
        max: T = if (@typeInfo(T) == .float) std.math.floatMax(T) else std.math.maxInt(T),
        is_hard_limit: bool = false, // if true, throw error if the value is invalid
    };
}

/// The field classes (see the file's doc comment).
pub const Kind = union(enum) {
    num_i32: Num(i32),
    num_i64: Num(i64),
    num_u32: Num(u32),
    num_f32: Num(f32),
    str,
    boolean: *bool,
    json,
    nested: []const Field,
};

/// Ports `field` (server-schema.h:27 @620d2225c): a request member, its
/// aliases in the order they are tried, and how it is read.
pub const Field = struct {
    name: []const []const u8,
    desc: []const u8 = "",
    kind: Kind,
    custom_handler: ?Handler = null,
};

/// `std::to_string(x)`: `%d`, `%lld`, `%u` or `%f`.
fn toString(buf: *[512]u8, x: anytype) []const u8 {
    switch (@typeInfo(@TypeOf(x))) {
        .float => {
            const n = snprintf(buf, buf.len, "%f", @as(f64, @floatCast(x)));
            return buf[0..@intCast(n)];
        },
        else => return std.fmt.bufPrint(buf, "{d}", .{x}) catch |err| switch (err) {
            error.NoSpaceLeft => unreachable,
        },
    }
}

extern fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;

// -----------------------------------------------------------------------------
// eval() implementations

/// Ports `handle_with_catch` (server-schema.cpp:518 @620d2225c): any
/// exception a field throws becomes `std::invalid_argument("Field '<name>':
/// <what>")`.
fn handleWithCatch(name: []const u8, r: Error!void) Error!void {
    r catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            const w = try std.heap.c_allocator.dupe(u8, common.what(e));
            defer std.heap.c_allocator.free(w);
            // `string_format` takes `%s`: the text stops at its first NUL
            const cut = if (std.mem.indexOfScalar(u8, w, 0)) |nul| w[0..nul] else w;
            const n = if (std.mem.indexOfScalar(u8, name, 0)) |nul| name[0..nul] else name;
            return throw(error.InvalidArgument, "Field '{s}': {s}", .{ n, cut });
        },
    };
}

/// Ports `has_value` (server-schema.cpp:527 @620d2225c): a null value is
/// absent, so a client can send null for the server's default.
fn hasValue(data: *const Value, n: []const u8) bool {
    return data.contains(n) and !(data.at(n) catch |err| switch (err) {
        error.OutOfMemory, error.JsonError => unreachable,
    }).isNull();
}

/// `std::max(min, std::min(max, v))`, NaN included.
fn clamp(comptime T: type, min: T, max: T, v: T) T {
    const lo = if (v < max) v else max;
    return if (min < lo) lo else min;
}

/// Ports `field_num`'s `eval` (server-schema.cpp:532 @620d2225c), the template's.
fn evalNum(comptime T: type, f: *const Field, num: Num(T), ctx: *EvalContext, data: *const Value) Error!void {
    for (f.name) |n| {
        if (hasValue(data, n)) {
            try handleWithCatch(n, blk: {
                if (f.custom_handler) |h| {
                    break :blk h(ctx, data);
                } else if (!num.is_hard_limit) {
                    const v = (data.at(n) catch |err| switch (err) {
                        error.OutOfMemory, error.JsonError => unreachable,
                    }).get(T) catch |e| break :blk e;
                    num.val.* = clamp(T, num.min, num.max, v);
                    break :blk {};
                } else {
                    const tmp = (data.at(n) catch |err| switch (err) {
                        error.OutOfMemory, error.JsonError => unreachable,
                    }).get(T) catch |e| break :blk e;
                    if (tmp < num.min or tmp > num.max) {
                        var b1: [512]u8 = undefined;
                        var b2: [512]u8 = undefined;
                        var b3: [512]u8 = undefined;
                        break :blk throw(error.InvalidArgument, "Value must be between {s} <= value <= {s}, but got {s}", .{ toString(&b1, num.min), toString(&b2, num.max), toString(&b3, tmp) });
                    }
                    num.val.* = tmp;
                    break :blk {};
                }
            });
            return;
        }
    }
}

/// Ports `field_str`'s `eval` (server-schema.cpp:553 @620d2225c) and
/// `field_json`'s `eval` (server-schema.cpp:580 @620d2225c), which are the same.
fn evalHandled(f: *const Field, ctx: *EvalContext, data: *const Value, comptime line: c_int) Error!void {
    common.ggmlAssert(f.custom_handler != null, "server-schema.cpp", line, "custom_handler");
    for (f.name) |n| {
        if (hasValue(data, n)) {
            try handleWithCatch(n, f.custom_handler.?(ctx, data));
            return;
        }
    }
}

/// Ports `field_bool`'s `eval` (server-schema.cpp:565 @620d2225c).
fn evalBool(f: *const Field, val: *bool, ctx: *EvalContext, data: *const Value) Error!void {
    for (f.name) |n| {
        if (hasValue(data, n)) {
            try handleWithCatch(n, blk: {
                if (f.custom_handler) |h| {
                    break :blk h(ctx, data);
                } else {
                    val.* = (data.at(n) catch |err| switch (err) {
                        error.OutOfMemory, error.JsonError => unreachable,
                    }).get(bool) catch |e| break :blk e;
                    break :blk {};
                }
            });
            return;
        }
    }
}

/// Ports `field_nested`'s `eval` (server-schema.cpp:592 @620d2225c): the
/// subfields, evaluated on the member's object.
fn evalNested(f: *const Field, subfields: []const Field, ctx: *EvalContext, data: *const Value) Error!void {
    for (f.name) |n| {
        if (data.contains(n) and (data.at(n) catch |err| switch (err) {
            error.OutOfMemory, error.JsonError => unreachable,
        }).isObject()) {
            const sub = data.at(n) catch |err| switch (err) {
                error.OutOfMemory, error.JsonError => unreachable,
            };
            for (subfields) |*sf| {
                try eval(sf, ctx, sub);
            }
            return;
        }
    }
}

/// `f->eval(ctx, data)`, the virtual dispatched.
pub fn eval(f: *const Field, ctx: *EvalContext, data: *const Value) Error!void {
    switch (f.kind) {
        .num_i32 => |n| try evalNum(i32, f, n, ctx, data),
        .num_i64 => |n| try evalNum(i64, f, n, ctx, data),
        .num_u32 => |n| try evalNum(u32, f, n, ctx, data),
        .num_f32 => |n| try evalNum(f32, f, n, ctx, data),
        .str => try evalHandled(f, ctx, data, 554),
        .json => try evalHandled(f, ctx, data, 581),
        .boolean => |v| try evalBool(f, v, ctx, data),
        .nested => |subs| try evalNested(f, subs, ctx, data),
    }
}

// -----------------------------------------------------------------------------
// the handlers

/// `common_tokenize(ctx.vocab, text, add_special, parse_special)` after
/// `GGML_ASSERT(ctx.vocab != nullptr)`.
fn tokenizeIn(ctx: *EvalContext, comptime line: c_int, text: []const u8, add_special: bool, parse_special: bool) Error![]Token {
    common.ggmlAssert(ctx.vocab != null, "server-schema.cpp", line, "ctx.vocab != nullptr");
    return common.tokenize(ctx.a, ctx.vocab, text, add_special, parse_special);
}

fn hResponseFields(ctx: *EvalContext, data: *const Value) Error!void {
    const fields = try common.jsonValueStrings(ctx.a, data, "response_fields");
    const out = try ctx.a.alloc([]const u8, fields.len);
    for (fields, 0..) |f, i| out[i] = try ctx.a.dupe(u8, f);
    ctx.params.response_fields = out;
}

fn hDryBase(ctx: *EvalContext, data: *const Value) Error!void {
    const v = try (try data.at("dry_base")).get(f32);
    ctx.params.sampling.dry_base = if (v < 1.0) ctx.params_base.sampling.dry_base else v;
}

fn hLora(ctx: *EvalContext, data: *const Value) Error!void {
    const lora = try data.at("lora");
    if (!lora.isArray()) {
        return throw(error.RuntimeError, "Error: 'lora' must be an array of objects with 'id' and 'scale' fields", .{});
    }
    ctx.params.lora = try common.parseLoraRequest(ctx.a, lora);
}

fn hDrySequenceBreakers(ctx: *EvalContext, data: *const Value) Error!void {
    const br = try common.jsonValueStrings(ctx.a, data, "dry_sequence_breakers");
    const out = try ctx.a.alloc([]const u8, br.len);
    for (br, 0..) |b, i| out[i] = try ctx.a.dupe(u8, b);
    ctx.params.sampling.dry_sequence_breakers = .fromOwnedSlice(out);
    if (ctx.params.sampling.dry_sequence_breakers.items.len == 0) {
        return throw(error.RuntimeError, "Error: dry_sequence_breakers must be a non-empty array of strings", .{});
    }
}

fn hJsonSchema(ctx: *EvalContext, data: *const Value) Error!void {
    const params = ctx.params;
    const a = ctx.a;
    if (data.contains("json_schema") and !data.contains("grammar")) {
        convertSchema(ctx, data) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return common.exception.rethrow(error.RuntimeError, e, "\"json_schema\": ", ""),
        };
    } else {
        const grammar_str = common.jsonValue(data, "grammar", []const u8, "");
        if (grammar_str.len != 0) {
            // grammar_type key is set by the server when converting chat template grammars
            const grammar_type = common.jsonValue(data, "grammar_type", []const u8, "");
            if (std.mem.eql(u8, grammar_type, "tool_calls")) {
                params.sampling.grammar_type = .tool_calls;
                params.sampling.grammar = try a.dupe(u8, grammar_str);
            } else {
                // explicit grammar from the user (API field "grammar")
                params.sampling.grammar_type = .user;
                params.sampling.grammar = try a.dupe(u8, grammar_str);
            }
            if (log.verbosity_thold >= log.level_debug) {
                const gt = common.dupZ(grammar_type);
                defer common.freeZ(gt);
                const gv = common.dupZ(task.grammarValue(&params.sampling));
                defer common.freeZ(gv);
                common.srv(.dbg, "operator()", "Grammar (%s): %s\n", .{ gt, gv });
            }
        }
    }
}

/// The `try` block of the `json_schema` handler.
fn convertSchema(ctx: *EvalContext, data: *const Value) Error!void {
    const a = ctx.a;
    var schema = try common.jsonValueJson(a, data, "json_schema", Value.emptyObject());
    if (schema.isObject() and schema.empty()) {
        // an empty schema means any object
        try schema.set(a, "type", try Value.initString(a, "object"));
    }
    if (log.verbosity_thold >= log.level_debug) {
        const d = try json.dump(a, &schema, 2);
        const z = common.dupZ(d);
        defer common.freeZ(z);
        common.srv(.dbg, "operator()", "JSON schema: %s\n", .{z});
    }
    const grammar_str = try oaicompat.schemaToGrammar(a, &schema);
    if (log.verbosity_thold >= log.level_debug) {
        const z = common.dupZ(grammar_str);
        defer common.freeZ(z);
        common.srv(.dbg, "operator()", "Converted grammar: %s\n", .{z});
    }
    ctx.params.sampling.grammar_type = .output_format;
    ctx.params.sampling.grammar = grammar_str;
}

fn hReasoningFormat(ctx: *EvalContext, data: *const Value) Error!void {
    ctx.params.reasoning_format = try chat.ReasoningFormat.fromName(try (try data.at("reasoning_format")).get([]const u8));
}

fn hContinueFinalMessage(_: *EvalContext, data: *const Value) Error!void {
    _ = try chat.msg.continuationParse(try data.at("continue_final_message"));
}

fn hEcho(_: *EvalContext, data: *const Value) Error!void {
    _ = try (try data.at("echo")).get(bool);
}

fn hPreservedTokens(ctx: *EvalContext, data: *const Value) Error!void {
    const tokens = try (try data.at("preserved_tokens")).getSlice(ctx.a, []const u8);
    // `common_sampling_add_preserved_tokens` (sampling.cpp:1055 @620d2225c)
    common.ggmlAssert(ctx.vocab != null, "sampling.cpp", 1056, "vocab != nullptr");
    try chat.session.addPreservedTokens(ctx.a, &ctx.params.sampling.preserved_tokens, ctx.vocab.?, tokens);
}

fn hGrammarTriggers(ctx: *EvalContext, data: *const Value) Error!void {
    var triggers: std.ArrayList(tools.params.GrammarTrigger) = .empty;
    var it = (try data.at("grammar_triggers")).items();
    while (it.next()) |t| {
        const ct = try GrammarTrigger.fromJson(ctx.a, t.value);
        try triggers.append(ctx.a, .{ .type = try triggerType(ct.type), .value = ct.value, .token = ct.token });
    }
    try addGrammarTriggers(ctx, triggers.items);
}

/// Ports `common_sampling_add_grammar_triggers` (sampling.cpp:1065
/// @620d2225c) over `tools_common`'s sampling parameters, whose trigger
/// type, unlike `common_chat`'s, holds any integer a request sends:
/// `cli/common_chat/session.zig`'s `addGrammarTriggers` is the same
/// function over that module's types.
fn addGrammarTriggers(ctx: *EvalContext, triggers: []const tools.params.GrammarTrigger) Error!void {
    common.ggmlAssert(ctx.vocab != null, "sampling.cpp", 1066, "vocab != nullptr");
    const s = &ctx.params.sampling;
    for (triggers) |trigger| {
        if (trigger.type == .word) {
            const word = trigger.value;
            const ids = try common.tokenize(ctx.a, ctx.vocab, word, false, true);
            if (ids.len == 1) {
                const token = ids[0];
                if (std.mem.indexOfScalar(Token, s.preserved_tokens.items, token) == null) {
                    return throw(error.RuntimeError, "Grammar trigger word should be marked as preserved token: {s}", .{word});
                }
                try s.grammar_triggers.append(ctx.a, .{ .type = .token, .value = word, .token = token });
            } else {
                try s.grammar_triggers.append(ctx.a, .{ .type = .word, .value = word });
            }
        } else {
            try s.grammar_triggers.append(ctx.a, trigger);
        }
    }
    if (s.grammar_lazy and s.grammar_triggers.items.len == 0) {
        return throw(error.RuntimeError, "Error: no triggers set for lazy grammar!", .{});
    }
}

/// `(common_grammar_trigger_type) in.at("type").get<int>()`: the C++ keeps
/// any integer, `tools/common/params.zig`'s `GrammarTriggerType` only the
/// four it names; another one is refused until that enum is open (see
/// `NOTES.md`, "Stage 10 step 3a").
fn triggerType(t: c_int) Error!tools.params.GrammarTriggerType {
    const T = tools.params.GrammarTriggerType;
    if (@typeInfo(T).@"enum".mode == .nonexhaustive) return @fromBackingInt(@intCast(t));
    inline for (@typeInfo(T).@"enum".field_values) |v| {
        if (t == v) return @fromBackingInt(@intCast(v));
    }
    return throw(error.RuntimeError, "grammar trigger type {d} is not representable here", .{t});
}

fn hReasoningBudgetStartTag(ctx: *EvalContext, data: *const Value) Error!void {
    const toks = try tokenizeIn(ctx, 342, try (try data.at("reasoning_budget_start_tag")).get([]const u8), false, true);
    ctx.params.sampling.reasoning_budget_start = .fromOwnedSlice(toks);
}

fn hReasoningBudgetEndTags(ctx: *EvalContext, data: *const Value) Error!void {
    common.ggmlAssert(ctx.vocab != null, "server-schema.cpp", 342, "ctx.vocab != nullptr");
    const s = &ctx.params.sampling;
    s.reasoning_budget_end = .empty;
    if (data.contains("reasoning_budget_end_tags")) {
        var it = (try data.at("reasoning_budget_end_tags")).items();
        while (it.next()) |t| {
            const tag = try t.value.get([]const u8);
            if (tag.len != 0) {
                try s.reasoning_budget_end.append(ctx.a, .fromOwnedSlice(try tokenizeIn(ctx, 350, tag, false, true)));
            }
        }
    } else if (data.contains("reasoning_budget_end_tag")) {
        const tag = try (try data.at("reasoning_budget_end_tag")).get([]const u8);
        if (tag.len != 0) {
            try s.reasoning_budget_end.append(ctx.a, .fromOwnedSlice(try tokenizeIn(ctx, 350, tag, false, true)));
        }
    }
}

fn hReasoningBudgetMessage(ctx: *EvalContext, data: *const Value) Error!void {
    common.ggmlAssert(ctx.vocab != null, "server-schema.cpp", 370, "ctx.vocab != nullptr");
    const s = &ctx.params.sampling;
    if (s.reasoning_budget_end.items.len != 0) {
        var end_tag: std.ArrayList(Token) = .fromOwnedSlice(try ctx.a.dupe(Token, s.reasoning_budget_end.items[0].items));
        const message = common.jsonValue(data, "reasoning_budget_message", []const u8, "");
        if (message.len != 0) {
            const message_tokens = try tokenizeIn(ctx, 370, message, false, true);
            try end_tag.insertSlice(ctx.a, 0, message_tokens);
        }
        s.reasoning_budget_forced = end_tag;
    }
}

extern fn strtol(s: [*:0]const u8, end: *[*:0]const u8, base: c_int) c_long;

/// The `parse_bias` lambda: a number, or `false` to ban the token.
fn parseBias(v: *const Value, bias: *f32) Error!bool {
    if (v.isNumber()) {
        bias.* = try v.get(f32);
        return true;
    }
    if (v.isBoolean() and !(try v.get(bool))) {
        bias.* = -std.math.inf(f32);
        return true;
    }
    return false;
}

fn hLogitBias(ctx: *EvalContext, data: *const Value) Error!void {
    common.ggmlAssert(ctx.vocab != null, "server-schema.cpp", 385, "ctx.vocab != nullptr");
    const s = &ctx.params.sampling;
    s.logit_bias = .empty;
    const logit_bias = try data.at("logit_bias");
    const n_vocab = c.llama_vocab_n_tokens(ctx.vocab);
    if (logit_bias.isArray()) {
        for (logit_bias.array.items) |*el| {
            if (!el.isArray() or el.size() != 2) continue;
            var bias: f32 = 0;
            if (!try parseBias(try el.atIndex(1), &bias)) continue;
            const e0 = try el.atIndex(0);
            if (e0.isNumberInteger()) {
                const tok = try e0.get(Token);
                if (tok >= 0 and tok < n_vocab) try s.logit_bias.append(ctx.a, .{ .token = tok, .bias = bias });
            } else if (e0.isString()) {
                for (try common.tokenize(ctx.a, ctx.vocab, try e0.get([]const u8), false, false)) |tok|
                    try s.logit_bias.append(ctx.a, .{ .token = tok, .bias = bias });
            }
        }
    } else if (logit_bias.isObject()) {
        var it = logit_bias.items();
        while (it.next()) |el| {
            var bias: f32 = 0;
            if (!try parseBias(el.value, &bias)) continue;
            const key = try ctx.a.dupeSentinel(u8, el.key, 0);
            var end: [*:0]const u8 = key.ptr;
            const tok: Token = @truncate(strtol(key.ptr, &end, 10));
            if (end[0] == 0) {
                if (tok >= 0 and tok < n_vocab) try s.logit_bias.append(ctx.a, .{ .token = tok, .bias = bias });
            } else {
                for (try common.tokenize(ctx.a, ctx.vocab, el.key, false, false)) |t|
                    try s.logit_bias.append(ctx.a, .{ .token = t, .bias = bias });
            }
        }
    }
}

fn hIgnoreEos(ctx: *EvalContext, data: *const Value) Error!void {
    common.ggmlAssert(ctx.logit_bias_eog != null, "server-schema.cpp", 426, "ctx.logit_bias_eog != nullptr");
    const s = &ctx.params.sampling;
    s.ignore_eos = try (try data.at("ignore_eos")).get(bool);
    if (s.ignore_eos) {
        if (ctx.logit_bias_eog) |eog| try s.logit_bias.appendSlice(ctx.a, eog);
    }
}

fn hStop(ctx: *EvalContext, data: *const Value) Error!void {
    var antiprompt: std.ArrayList([]const u8) = .empty;
    const stop = try data.at("stop");
    if (stop.isArray()) {
        for (stop.array.items) |*word| {
            if (!word.empty()) try antiprompt.append(ctx.a, try ctx.a.dupe(u8, try word.get([]const u8)));
        }
    } else if (stop.isString()) {
        try antiprompt.append(ctx.a, try ctx.a.dupe(u8, try stop.get([]const u8)));
    }
    ctx.params.antiprompt = antiprompt.items;
    // fall back to CLI defaults if the request provided no effective stop strings
    if (ctx.params.antiprompt.len == 0) {
        ctx.params.antiprompt = try dupeStrings(ctx.a, ctx.params_base.antiprompt.items);
    }
}

fn hSamplers(ctx: *EvalContext, data: *const Value) Error!void {
    const samplers = try data.at("samplers");
    if (samplers.isArray()) {
        const names = try samplers.getSlice(ctx.a, []const u8);
        ctx.params.sampling.samplers = try samplerTypesFromNames(ctx.a, names);
    } else if (samplers.isString()) {
        ctx.params.sampling.samplers = try samplerTypesFromChars(ctx.a, try samplers.get([]const u8));
    }
}

fn dupeStrings(a: Allocator, xs: []const []const u8) Error![]const []const u8 {
    const out = try a.alloc([]const u8, xs.len);
    for (xs, 0..) |x, i| out[i] = try a.dupe(u8, x);
    return out;
}

/// Ports `common_sampler_types_from_names` (sampling.cpp:963 @620d2225c): each
/// name, lower-cased, by its canonical spelling or an alias; an unknown
/// one is warned about and skipped.
pub fn samplerTypesFromNames(a: Allocator, names: []const []const u8) Error!std.ArrayList(SamplerType) {
    const Entry = struct { []const u8, SamplerType };
    const map = [_]Entry{
        // canonical
        .{ "dry", .dry },                 .{ "top_k", .top_k },           .{ "top_p", .top_p },
        .{ "top_n_sigma", .top_n_sigma }, .{ "typ_p", .typical_p },       .{ "min_p", .min_p },
        .{ "temperature", .temperature }, .{ "xtc", .xtc },               .{ "infill", .infill },
        .{ "penalties", .penalties },     .{ "adaptive_p", .adaptive_p },
        // kebab-case and no-dash aliases of the names with an underscore
        .{ "top-k", .top_k },
        .{ "topk", .top_k },              .{ "top-p", .top_p },           .{ "topp", .top_p },
        .{ "top-n-sigma", .top_n_sigma }, .{ "topnsigma", .top_n_sigma }, .{ "typ-p", .typical_p },
        .{ "typp", .typical_p },          .{ "min-p", .min_p },           .{ "minp", .min_p },
        .{ "adaptive-p", .adaptive_p },   .{ "adaptivep", .adaptive_p },
        // misc. aliases
         .{ "nucleus", .top_p },
        .{ "temp", .temperature },        .{ "typ", .typical_p },
    };

    var samplers: std.ArrayList(SamplerType) = .empty;
    try samplers.ensureTotalCapacity(a, names.len);

    for (names) |name| {
        const name_lower = try a.dupeSentinel(u8, name, 0);
        for (name_lower) |*ch| ch.* = std.ascii.toLower(ch.*);
        const found = for (map) |e| {
            if (std.mem.eql(u8, e[0], name_lower)) break e[1];
        } else null;
        if (found) |t| {
            try samplers.append(a, t);
            continue;
        }
        log.wrn("%s: unable to match sampler by name '%s'\n", .{ "common_sampler_types_from_names", name_lower.ptr });
    }

    return samplers;
}

/// Ports `common_sampler_types_from_chars` (sampling.cpp:1025 @620d2225c).
pub fn samplerTypesFromChars(a: Allocator, chars: []const u8) Error!std.ArrayList(SamplerType) {
    const all = [_]SamplerType{ .dry, .top_k, .typical_p, .top_p, .top_n_sigma, .min_p, .temperature, .xtc, .infill, .penalties, .adaptive_p };
    var samplers: std.ArrayList(SamplerType) = .empty;
    try samplers.ensureTotalCapacity(a, chars.len);
    for (chars) |ch| {
        const found = for (all) |t| {
            if (t.toChr() == ch) break t;
        } else null;
        if (found) |t| {
            try samplers.append(a, t);
        } else {
            log.wrn("%s: unable to match sampler by char '%c'\n", .{ "common_sampler_types_from_chars", @as(c_int, ch) });
        }
    }
    return samplers;
}

// -----------------------------------------------------------------------------
// the table

/// Ports `make_llama_cmpl_schema` (server-schema.cpp:11 @620d2225c): the
/// llama.cpp-specific completion schema, every field pointing into
/// `params`.
///
/// Return: owned by `a`.
pub fn makeLlamaCmplSchema(a: Allocator, params_base: *const tools.params.Params, params: *TaskParams) Error![]const Field {
    const s = &params.sampling;
    const i32max = std.math.maxInt(i32);
    const stream_options = try a.dupe(Field, &.{
        .{ .name = &.{"include_usage"}, .desc = "Whether to include usage information in the stream", .kind = .{ .boolean = &params.include_usage } },
    });
    const fields = [_]Field{
        .{ .name = &.{"verbose"}, .desc = "Include __verbose field in the response with additional debug information", .kind = .{ .boolean = &params.verbose } },
        .{ .name = &.{"timings_per_token"}, .desc = "Include prompt processing and text generation speed information in each response", .kind = .{ .boolean = &params.timings_per_token } },
        .{ .name = &.{"stream"}, .desc = "Allows receiving each predicted token in real-time instead of waiting for the completion to finish", .kind = .{ .boolean = &params.stream } },
        .{ .name = &.{"stream_options"}, .desc = "Additional options for streaming responses", .kind = .{ .nested = stream_options } },
        .{ .name = &.{"cache_prompt"}, .desc = "Re-use KV cache from a previous request if possible. This way the common prefix does not have to be re-processed, only the suffix that differs between the requests", .kind = .{ .boolean = &params.cache_prompt } },
        .{ .name = &.{"return_tokens"}, .desc = "Return the raw generated token ids in the `tokens` field", .kind = .{ .boolean = &params.return_tokens } },
        .{ .name = &.{"return_progress"}, .desc = "Include prompt processing progress events in stream mode", .kind = .{ .boolean = &params.return_progress } },
        .{ .name = &.{"sse_ping_interval"}, .desc = "Interval in seconds between SSE comment pings emitted while the stream stays silent, -1 disables pings", .kind = .{ .num_i32 = .{ .val = &params.sse_ping_interval, .min = -1, .max = i32max, .is_hard_limit = true } } },
        .{ .name = &.{ "n_predict", "max_completion_tokens", "max_tokens" }, .desc = "Set the maximum number of tokens to predict. When 0, no tokens will be generated but the prompt is evaluated into the cache", .kind = .{ .num_i32 = .{ .val = &params.n_predict, .min = -1, .max = i32max, .is_hard_limit = true } } },
        .{ .name = &.{"n_indent"}, .desc = "Specify the minimum line indentation for the generated text in number of whitespace characters. Useful for code completion tasks", .kind = .{ .num_i32 = .{ .val = &params.n_indent, .min = 0, .max = i32max, .is_hard_limit = true } } },
        .{ .name = &.{"n_keep"}, .desc = "Specify the number of tokens from the initial prompt to retain when context size is exceeded. Use -1 to retain all tokens from the prompt", .kind = .{ .num_i32 = .{ .val = &params.n_keep, .min = -1, .max = i32max, .is_hard_limit = true } } },
        .{ .name = &.{"n_discard"}, .desc = "Number of tokens after n_keep that may be discarded when shifting context (0 = half context)", .kind = .{ .num_i32 = .{ .val = &params.n_discard, .min = 0, .max = i32max, .is_hard_limit = true } } },
        .{ .name = &.{ "n_cmpl", "n" }, .desc = "Number of completions to generate. If the input has multiple prompts, total outputs will be N prompts times n_cmpl", .kind = .{ .num_i32 = .{ .val = &params.n_cmpl, .min = 1, .max = params_base.n_parallel, .is_hard_limit = true } } },
        .{ .name = &.{"n_cache_reuse"}, .desc = "Min chunk size to attempt reusing from the cache via KV shifting. See --cache-reuse arg", .kind = .{ .num_i32 = .{ .val = &params.n_cache_reuse, .min = 0, .max = i32max, .is_hard_limit = true } } },
        .{ .name = &.{"t_max_predict_ms"}, .desc = "Set a time limit in milliseconds for the prediction phase. The timeout triggers if generation exceeds this time (measured since the first token) and a newline has been generated. Useful for FIM applications", .kind = .{ .num_i64 = .{ .val = &params.t_max_predict_ms, .min = -1, .max = std.math.maxInt(i64), .is_hard_limit = true } } },
        .{ .name = &.{"response_fields"}, .desc = "A list of response fields to return. Missing fields are omitted without error. Fields with a slash are unnested (e.g. generation_settings/n_predict moves n_predict to the root)", .kind = .json, .custom_handler = hResponseFields },

        //
        // Sampling params
        //

        .{ .name = &.{"top_k"}, .desc = "Limit the next token selection to the K most probable tokens (0 = disabled)", .kind = .{ .num_i32 = .{ .val = &s.top_k, .min = 0, .max = i32max } } },
        .{ .name = &.{"top_p"}, .desc = "Limit the next token selection to a subset of tokens with cumulative probability above threshold P (1.0 = disabled)", .kind = .{ .num_f32 = .{ .val = &s.top_p, .min = 0.0, .max = 1.0 } } },
        .{ .name = &.{"min_p"}, .desc = "The minimum probability for a token to be considered, relative to the probability of the most likely token (0 = disabled)", .kind = .{ .num_f32 = .{ .val = &s.min_p, .min = 0.0, .max = 1.0 } } },
        .{ .name = &.{"top_n_sigma"}, .desc = "Keep tokens within n standard deviations of the top token logit (< 0 = disabled)", .kind = .{ .num_f32 = .{ .val = &s.top_n_sigma } } },
        .{ .name = &.{"xtc_probability"}, .desc = "Set the chance for token removal via XTC sampler (0 = disabled)", .kind = .{ .num_f32 = .{ .val = &s.xtc_probability, .min = 0.0, .max = 1.0 } } },
        .{ .name = &.{"xtc_threshold"}, .desc = "Set a minimum probability threshold for tokens to be removed via XTC sampler (> 0.5 disables XTC)", .kind = .{ .num_f32 = .{ .val = &s.xtc_threshold, .min = 0.0, .max = 1.0 } } },
        .{ .name = &.{"typical_p"}, .desc = "Enable locally typical sampling with parameter p (1.0 = disabled)", .kind = .{ .num_f32 = .{ .val = &s.typ_p } } },
        .{ .name = &.{"temperature"}, .desc = "Adjust the randomness of the generated text (0 = greedy)", .kind = .{ .num_f32 = .{ .val = &s.temp, .min = 0.0, .max = std.math.inf(f32) } } },
        .{ .name = &.{"dynatemp_range"}, .desc = "Dynamic temperature range. The final temperature will be in [temperature - range, temperature + range] (0 = disabled)", .kind = .{ .num_f32 = .{ .val = &s.dynatemp_range } } },
        .{ .name = &.{"dynatemp_exponent"}, .desc = "Dynamic temperature exponent, controls how entropy maps to temperature", .kind = .{ .num_f32 = .{ .val = &s.dynatemp_exponent } } },
        .{ .name = &.{"repeat_last_n"}, .desc = "Last n tokens to consider for penalizing repetition (0 = disabled)", .kind = .{ .num_i32 = .{ .val = &s.penalty_last_n, .min = 0, .max = i32max, .is_hard_limit = true } } },
        .{ .name = &.{"repeat_penalty"}, .desc = "Control the repetition of token sequences in the generated text (1.0 = disabled)", .kind = .{ .num_f32 = .{ .val = &s.penalty_repeat } } },
        .{ .name = &.{"frequency_penalty"}, .desc = "Repeat alpha frequency penalty (0 = disabled)", .kind = .{ .num_f32 = .{ .val = &s.penalty_freq } } },
        .{ .name = &.{"presence_penalty"}, .desc = "Repeat alpha presence penalty (0 = disabled)", .kind = .{ .num_f32 = .{ .val = &s.penalty_present } } },
        .{ .name = &.{"dry_multiplier"}, .desc = "Set the DRY (Don't Repeat Yourself) repetition penalty multiplier (0 = disabled)", .kind = .{ .num_f32 = .{ .val = &s.dry_multiplier } } },
        .{ .name = &.{"dry_base"}, .desc = "Set the DRY repetition penalty base value (must be >= 1.0, any values < 1.0 will be replaced with the default value)", .kind = .{ .num_f32 = .{ .val = &s.dry_base } }, .custom_handler = hDryBase },
        .{ .name = &.{"dry_allowed_length"}, .desc = "Tokens that extend repetition beyond this length receive exponentially increasing penalty: multiplier * base ^ (sequence_length - allowed_length)", .kind = .{ .num_i32 = .{ .val = &s.dry_allowed_length, .min = 0, .max = i32max, .is_hard_limit = true } } },
        .{ .name = &.{"dry_penalty_last_n"}, .desc = "How many tokens to scan for repetitions (0 = disabled)", .kind = .{ .num_i32 = .{ .val = &s.dry_penalty_last_n, .min = 0, .max = i32max, .is_hard_limit = true } } },
        .{ .name = &.{"mirostat"}, .desc = "Enable Mirostat sampling, controlling perplexity during text generation (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", .kind = .{ .num_i32 = .{ .val = &s.mirostat, .min = 0, .max = 2 } } },
        .{ .name = &.{"mirostat_tau"}, .desc = "Set the Mirostat target entropy, parameter tau", .kind = .{ .num_f32 = .{ .val = &s.mirostat_tau } } },
        .{ .name = &.{"mirostat_eta"}, .desc = "Set the Mirostat learning rate, parameter eta", .kind = .{ .num_f32 = .{ .val = &s.mirostat_eta } } },
        .{ .name = &.{"adaptive_target"}, .desc = "Adaptive sampling target entropy (valid range 0.0 to 1.0; negative = disabled)", .kind = .{ .num_f32 = .{ .val = &s.adaptive_target, .min = -std.math.floatMax(f32), .max = 1.0 } } },
        .{ .name = &.{"adaptive_decay"}, .desc = "EMA decay for adaptive sampling; history approximates 1/(1-decay) tokens", .kind = .{ .num_f32 = .{ .val = &s.adaptive_decay, .min = 0.0, .max = 0.99, .is_hard_limit = true } } },
        // seed is uint32_t; field_num uses int32_t so use a handler
        .{ .name = &.{"seed"}, .desc = "Set the random number generator (RNG) seed (-1 = random)", .kind = .{ .num_u32 = .{ .val = &s.seed } } },
        .{ .name = &.{ "n_probs", "logprobs" }, .desc = "If greater than 0, output the probabilities of top N tokens for each generated token", .kind = .{ .num_i32 = .{ .val = &s.n_probs } } },
        .{ .name = &.{"min_keep"}, .desc = "If greater than 0, force samplers to return at least N possible tokens", .kind = .{ .num_i32 = .{ .val = &s.min_keep, .min = 0, .max = i32max, .is_hard_limit = true } } },
        .{ .name = &.{"backend_sampling"}, .desc = "Use backend sampling instead of llama.cpp sampling", .kind = .{ .boolean = &s.backend_sampling } },
        .{ .name = &.{"post_sampling_probs"}, .desc = "Return probabilities of top n_probs tokens after applying the sampling chain", .kind = .{ .boolean = &params.post_sampling_probs } },

        // Speculative decoding params: disabled upstream (`#if 0`)

        .{ .name = &.{"lora"}, .desc = "A list of LoRA adapters to apply to this request. Each entry must have `id` and `scale` fields. Adapters not listed default to scale 0.0", .kind = .json, .custom_handler = hLora },
        // sequence breakers for DRY
        .{ .name = &.{"dry_sequence_breakers"}, .desc = "Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted", .kind = .json, .custom_handler = hDrySequenceBreakers },
        // handle both "json_schema" and "grammar"
        .{ .name = &.{ "json_schema", "grammar" }, .desc = "Set a JSON schema (json_schema) or GBNF grammar string (grammar) for constrained generation. json_schema takes precedence if both are provided", .kind = .json, .custom_handler = hJsonSchema },
        .{ .name = &.{"grammar_lazy"}, .desc = "Whether to apply grammar constraints lazily, only when triggered (instead of at every step)", .kind = .{ .boolean = &s.grammar_lazy } },

        //
        // Chat parser params
        //

        .{ .name = &.{"reasoning_format"}, .desc = "Reasoning format for chain-of-thought models", .kind = .str, .custom_handler = hReasoningFormat },
        .{ .name = &.{"continue_final_message"}, .desc = "Whether to continue the final message of the chat template", .kind = .json, .custom_handler = hContinueFinalMessage },
        .{ .name = &.{"echo"}, .desc = "Whether to include the continued assistant message in the output", .kind = .json, .custom_handler = hEcho },

        //
        // Token-level fields (require vocab)
        //

        .{ .name = &.{"preserved_tokens"}, .desc = "List of token strings that must not be split during tokenization", .kind = .json, .custom_handler = hPreservedTokens },
        .{ .name = &.{"grammar_triggers"}, .desc = "List of strings or patterns that trigger grammar-constrained generation", .kind = .json, .custom_handler = hGrammarTriggers },
        .{ .name = &.{"reasoning_control"}, .desc = "Create the budget sampler on demand so reasoning can be ended at runtime", .kind = .{ .boolean = &s.reasoning_control } },
        .{ .name = &.{"reasoning_budget_tokens"}, .desc = "Number of tokens in the reasoning budget (-1 = disabled)", .kind = .{ .num_i32 = .{ .val = &s.reasoning_budget_tokens, .min = -1, .max = i32max, .is_hard_limit = true } } },
        .{ .name = &.{"reasoning_budget_start_tag"}, .desc = "Token string marking the start of the reasoning budget section", .kind = .str, .custom_handler = hReasoningBudgetStartTag },
        .{ .name = &.{ "reasoning_budget_end_tags", "reasoning_budget_end_tag" }, .desc = "Token strings marking the end of the reasoning budget section; the first is forced when the budget expires", .kind = .json, .custom_handler = hReasoningBudgetEndTags },
        .{ .name = &.{"reasoning_budget_message"}, .desc = "Message to prepend to the reasoning budget end tag when forcing it", .kind = .str, .custom_handler = hReasoningBudgetMessage },
        .{ .name = &.{"logit_bias"}, .desc = "Modify the likelihood of specific tokens. Accepts an array of [token, bias] pairs or an object mapping token to bias. Use false as bias to ban a token", .kind = .json, .custom_handler = hLogitBias },
        .{ .name = &.{"ignore_eos"}, .desc = "Ignore the end-of-sequence token and continue generating", .kind = .{ .boolean = &s.ignore_eos }, .custom_handler = hIgnoreEos },
        .{ .name = &.{"stop"}, .desc = "Specify stopping strings. Generation stops when one is produced, and the string is not included in the output", .kind = .json, .custom_handler = hStop },
        .{ .name = &.{"samplers"}, .desc = "The order in which samplers are applied. An array of sampler type names, or a single string of sampler chars", .kind = .json, .custom_handler = hSamplers },
    };
    return a.dupe(Field, &fields);
}

/// Ports `eval_llama_cmpl_schema` (server-schema.cpp:467 @620d2225c): a
/// request's parameters -- the command line's defaults, every field the
/// request carries applied over them in the table's order, then the
/// derived ones.
///
/// Parameters:
/// - `a`: an arena; the parameters' strings and lists are allocated here.
///
/// Return: the parameters, allocated in `a`.
pub fn evalLlamaCmplSchema(
    a: Allocator,
    vocab: ?*const c.llama_vocab,
    params_base: *const tools.params.Params,
    logit_bias_eog: []const c.llama_logit_bias,
    data: *const Value,
) Error!TaskParams {
    var params: TaskParams = .{};

    // Sampling parameter defaults are loaded from the global server context (but individual requests can still override them)
    params.sampling = try task.deepClone(a, params_base.sampling);
    params.speculative = params_base.speculative;
    params.speculative.types = .fromOwnedSlice(try a.dupe(task.SpeculativeType, params_base.speculative.types.items));
    params.n_keep = params_base.n_keep;
    params.n_predict = params_base.n_predict;
    params.n_cache_reuse = params_base.n_cache_reuse;
    params.cache_prompt = params_base.cache_prompt;
    params.antiprompt = try dupeStrings(a, params_base.antiprompt.items);
    params.sse_ping_interval = params_base.sse_ping_interval;

    // enabling this will output extra debug information in the HTTP responses from the server
    params.verbose = params_base.verbosity > 9;

    params.reasoning_format = @fromBackingInt(@intCast(@backingInt(params_base.reasoning_format)));

    // create context and schema
    var ctx: EvalContext = .{ .a = a, .params = &params, .params_base = params_base };
    ctx.vocab = vocab;
    ctx.logit_bias_eog = logit_bias_eog;

    const schema = try makeLlamaCmplSchema(a, params_base, &params);

    // eval all fields in the schema
    for (schema) |*f| {
        try eval(f, &ctx, data);
    }

    // debugging
    if (log.verbosity_thold >= log.level_debug) {
        const budget = params.sampling.reasoning_budget_tokens;
        const gp = common.dupZ(params.sampling.generation_prompt);
        defer common.freeZ(gp);
        common.srv(.dbg, "eval_llama_cmpl_schema", "reasoning budget: tokens=%d, generation_prompt='%s', start=%zu toks, end=%zu seqs, forced=%zu toks\n", .{
            budget,                                            gp,
            params.sampling.reasoning_budget_start.items.len,  params.sampling.reasoning_budget_end.items.len,
            params.sampling.reasoning_budget_forced.items.len,
        });
    }

    return params;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a hard limit refuses with the C++'s text" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var base = try tools.params.Params.init(a);
    base.n_parallel = 4;
    var data = try json.parse(a, "{\"n_cmpl\":9,\"top_p\":7}");
    const r = evalLlamaCmplSchema(a, null, &base, &.{}, &data);
    try std.testing.expectError(error.InvalidArgument, r);
    try std.testing.expectEqualStrings("Field 'n_cmpl': Value must be between 1 <= value <= 4, but got 9", common.what(error.InvalidArgument));
}

test "a soft limit clamps" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var base = try tools.params.Params.init(a);
    base.n_parallel = 4;
    var data = try json.parse(a, "{\"top_p\":7,\"n\":2}");
    const p = try evalLlamaCmplSchema(a, null, &base, &.{}, &data);
    try std.testing.expectEqual(@as(f32, 1.0), p.sampling.top_p);
    try std.testing.expectEqual(@as(i32, 2), p.n_cmpl);
}
