//! Gemma 4: `<|channel>thought ... <channel|>` reasoning and
//! `<|tool_call>call:name{key:value,...}<tool_call|>` tool calls, with
//! strings quoted `<|"|>`, their arguments shaped by each tool's parameter
//! schema.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/gemma4.cpp` (llama.cpp v1.0.0,
//! `b1b770d90`): `depth_guard`, `gemma4_value_rules`,
//! `gemma4_schema_parser` and `common_chat_params_init_gemma4`. The parser
//! is built in the order clang evaluates the C++ expressions, left to right,
//! since a parser's id is its position.

const std = @import("std");
const json = @import("json");
const peg = @import("peg");
const exception = @import("../exception.zig");
const chat_peg = @import("../chat_peg.zig");
const msg_mod = @import("../msg.zig");
const params_mod = @import("../params.zig");
const template = @import("../template.zig");
const util = @import("../util.zig");
const log = @import("../log.zig");
const parsers = @import("module.zig");

const Allocator = std.mem.Allocator;
const Value = json.Value;
const Node = json.schema.Node;
const Error = exception.Error;
const ChatParams = params_mod.ChatParams;
const ChatBuilder = chat_peg.ChatBuilder;
const Mapper = chat_peg.Mapper;
const ParserId = peg.ParserId;
const GenerationParams = template.GenerationParams;

// -----------------------------------------------------------------------------
// Gemma 4 value parsers

fn gemma4Dict(b: *peg.Builder) peg.Error!ParserId {
    const ws = try b.space();
    const m = try b.ref("gemma4-dict-kv");
    const members = try b.sequence(&.{ m, try b.zeroOrMore(try b.sequence(&.{ try b.literal(","), ws, m })) });
    return b.tag(Mapper.GEMMA4_DICT, try b.sequence(&.{
        try b.literal("{"),                                                                            ws,
        try b.choice(&.{ try b.literal("}"), try b.sequence(&.{ members, ws, try b.literal("}") }) }),
    }));
}

fn gemma4Array(b: *peg.Builder) peg.Error!ParserId {
    const ws = try b.space();
    const value = try b.ref("gemma4-value");
    const elements = try b.sequence(&.{ value, try b.zeroOrMore(try b.sequence(&.{ try b.literal(","), ws, value })) });
    return b.tag(Mapper.GEMMA4_ARRAY, try b.sequence(&.{
        try b.literal("["),                                                                             ws,
        try b.choice(&.{ try b.literal("]"), try b.sequence(&.{ elements, ws, try b.literal("]") }) }),
    }));
}

fn gemma4Value(b: *peg.Builder) peg.Error!ParserId {
    return b.choice(&.{
        try b.ref("gemma4-string"), try b.ref("gemma4-dict"), try b.ref("gemma4-array"),
        try b.ref("gemma4-number"), try b.ref("gemma4-bool"), try b.ref("gemma4-null"),
    });
}

/// `p.literal(q) + p.tag(DICT_KEY_NAME, p.chars(cls, 1, -1)) + p.literal(q)`.
fn quotedKeyName(b: *peg.Builder, quote: []const u8, cls: []const u8) peg.Error!ParserId {
    const open = try b.literal(quote);
    const name = try b.tag(Mapper.GEMMA4_DICT_KEY_NAME, try b.chars(cls, 1, -1));
    return b.plus(try b.plus(open, name), try b.literal(quote));
}

/// Ports `gemma4_value_rules` (parsers/gemma4.cpp:25 @b1b770d90): the
/// rules of Gemma 4's value syntax. The tags are what `gemma4_to_json` in
/// the mapper reads, the rules may wrap them freely.
fn valueRules(b: *peg.Builder) peg.Error!void {
    {
        const open = try b.literal("<|\"|>");
        const content = try b.tag(Mapper.GEMMA4_STRING_CONTENT, try b.until("<|\"|>"));
        const body = try b.plus(try b.plus(open, content), try b.literal("<|\"|>"));
        _ = try b.rule("gemma4-string", try b.tag(Mapper.GEMMA4_STRING, body), false);
    }

    _ = try b.rule("gemma4-bool", try b.tag(Mapper.GEMMA4_BOOL, try b.jsonBool()), false);

    _ = try b.rule("gemma4-null", try b.tag(Mapper.GEMMA4_NULL, try b.jsonNull()), false);

    _ = try b.rule("gemma4-number", try b.tag(Mapper.GEMMA4_NUMBER, try b.jsonNumber()), false);

    // Keys are bare in the template, but Gemma 4 occasionally wraps them in quotes
    {
        const delim = try b.literal("<|\"|>");
        const delim_name = try b.tag(Mapper.GEMMA4_DICT_KEY_NAME, try b.until("<|\"|>"));
        const alt0 = try b.plus(try b.plus(delim, delim_name), try b.literal("<|\"|>"));
        const alt1 = try quotedKeyName(b, "\"", "[^\"]");
        const alt2 = try quotedKeyName(b, "'", "[^']");
        const alt3 = try b.tag(Mapper.GEMMA4_DICT_KEY_NAME, try b.chars("[^:}]", 1, -1));
        const key = try b.plus(try b.choice(&.{ alt0, alt1, alt2, alt3 }), try b.literal(":"));
        _ = try b.rule("gemma4-dict-key", try b.tag(Mapper.GEMMA4_DICT_KEY, key), false);
    }

    {
        const kv = try b.plus(try b.plus(try b.ref("gemma4-dict-key"), try b.space()), try b.ref("gemma4-value"));
        _ = try b.rule("gemma4-dict-kv", try b.tag(Mapper.GEMMA4_DICT_KV, kv), false);
    }

    _ = try b.ruleFn("gemma4-dict", b, gemma4Dict, false);
    _ = try b.ruleFn("gemma4-array", b, gemma4Array, false);
    _ = try b.ruleFn("gemma4-value", b, gemma4Value, false);
}

// -----------------------------------------------------------------------------
// The schema parser

/// Ports `gemma4_schema_parser` (parsers/gemma4.cpp:75 @b1b770d90): Gemma 4
/// uses different delimiters and cannot leverage the
/// json-schema-to-grammar implementation. This implements a subset to
/// enforce properties.
const SchemaParser = struct {
    a: Allocator,
    b: *peg.Builder,
    root: []const u8,
    depth: c_int = 0,
    /// `refs_`: each `$ref` target's rule name, keyed by the target
    refs: std.AutoHashMapUnmanaged(*const Node, []const u8) = .empty,

    fn visitRoot(self: *SchemaParser, node: *const Node) peg.Error!ParserId {
        return self.visit(node, self.root);
    }

    /// Ports `visit` (parsers/gemma4.cpp:88 @b1b770d90), the overload taking
    /// a name; its `depth_guard` (parsers/gemma4.cpp:14 @b1b770d90) throws
    /// past `COMMON_RECURSIVE_LIMIT`.
    fn visit(self: *SchemaParser, node: *const Node, name: []const u8) peg.Error!ParserId {
        self.depth += 1;
        defer self.depth -= 1;
        if (self.depth > json.schema.recursive_limit) {
            return peg.errors.raise(error.InvalidArgument, &.{"JSON schema is nested more than " ++ json.schema.recursive_limit_text ++ " levels deep"});
        }
        return switch (node.*) {
            .ref => |*r| self.visitRef(r),
            .any_of => |children| self.visitAnyOf(node, children, name),
            .all_of => |children| self.visitAllOf(children, name),
            .array => |*arr| self.visitArray(arr, name),
            .tuple => |items| self.visitTuple(items, name),
            .object => |*obj| self.visitObject(obj, name),
            else => self.b.ref("gemma4-value"),
        };
    }

    /// Ports `shaped` (parsers/gemma4.cpp:102 @b1b770d90): whether a schema
    /// contributes a shape of its own, as opposed to accepting any value.
    fn shaped(node: *const Node) bool {
        return switch (node.*) {
            .ref => |r| r.target != null,
            .any_of => |children| for (children) |child| {
                if (shaped(child)) break true;
            } else false,
            .all_of => |children| children.len == 1 and shaped(children[0]),
            .array, .tuple, .object => true,
            else => false,
        };
    }

    /// `p_.tag(DICT_KEY_NAME, p_.literal(key))`, the lambda `name` in `key`.
    fn keyName(self: *SchemaParser, k: []const u8) peg.Error!ParserId {
        return self.b.tag(Mapper.GEMMA4_DICT_KEY_NAME, try self.b.literal(k));
    }

    /// `lit + name() + lit`.
    fn quoted(self: *SchemaParser, quote: []const u8, k: []const u8) peg.Error!ParserId {
        const open = try self.b.literal(quote);
        const name = try self.keyName(k);
        return self.b.plus(try self.b.plus(open, name), try self.b.literal(quote));
    }

    /// Ports `key` (parsers/gemma4.cpp:128 @b1b770d90): the template writes
    /// keys bare. Models keep that for identifiers and quote anything else,
    /// with single or double quotes or the string delimiter, so those are
    /// accepted as well.
    fn key(self: *SchemaParser, k: []const u8) peg.Error!ParserId {
        const b = self.b;
        var identifier = k.len != 0 and !std.ascii.isDigit(k[0]);
        for (k) |c| {
            if (!std.ascii.isAlphanumeric(c) and c != '_') identifier = false;
        }
        if (identifier) {
            return b.tag(Mapper.GEMMA4_DICT_KEY, try b.plus(try self.keyName(k), try b.literal(":")));
        }

        var alts = try b.choiceEmpty();
        if (k.len != 0 and std.mem.indexOfAny(u8, k, ":}") == null and std.mem.indexOfScalar(u8, "'\"<", k[0]) == null) {
            // Check if key is unambiguous
            try b.orAssign(&alts, try self.keyName(k));
        }
        // Gemma 4 may emit keys in its string delimiters or quotes
        try b.orAssign(&alts, try self.quoted("<|\"|>", k));
        if (std.mem.indexOfScalar(u8, k, '\'') == null) {
            try b.orAssign(&alts, try self.quoted("'", k));
        }
        if (std.mem.indexOfScalar(u8, k, '"') == null) {
            try b.orAssign(&alts, try self.quoted("\"", k));
        }
        return b.tag(Mapper.GEMMA4_DICT_KEY, try b.plus(alts, try b.literal(":")));
    }

    const RefRule = struct {
        self: *SchemaParser,
        target: *const Node,
        name: []const u8,

        fn build(ctx: *const RefRule) peg.Error!ParserId {
            return ctx.self.visit(ctx.target, ctx.name);
        }
    };

    /// Ports `visit_ref` (parsers/gemma4.cpp:157 @b1b770d90): a target
    /// reached from several places gets one rule, named by the order it was
    /// first seen.
    fn visitRef(self: *SchemaParser, r: *const json.schema.Ref) peg.Error!ParserId {
        const target = r.target orelse return self.b.ref("gemma4-value");
        if (self.refs.get(target)) |rule| return self.b.ref(rule);
        const rule = try std.fmt.allocPrint(self.a, "{s}-ref-{d}", .{ self.root, self.refs.count() });
        try self.refs.put(self.a, target, rule);
        const ctx: RefRule = .{ .self = self, .target = target, .name = try std.mem.concat(self.a, u8, &.{ rule, "-value" }) };
        return self.b.ruleFn(rule, &ctx, RefRule.build, false);
    }

    /// Ports `visit_any_of` (parsers/gemma4.cpp:172 @b1b770d90): shaped
    /// alternatives are tried first, then anything else collapses into one
    /// generic value.
    fn visitAnyOf(self: *SchemaParser, node: *const Node, children: []const *Node, name: []const u8) peg.Error!ParserId {
        const b = self.b;
        if (!shaped(node)) return b.ref("gemma4-value");
        if (children.len == 1) return self.visit(children[0], name);
        var alts = try b.choiceEmpty();
        var loose = false;
        for (children, 0..) |child, i| {
            if (shaped(child)) {
                try b.orAssign(&alts, try self.visit(child, try std.fmt.allocPrint(self.a, "{s}-{d}", .{ name, i })));
            } else {
                loose = true;
            }
        }
        if (loose) try b.orAssign(&alts, try b.ref("gemma4-value"));
        return b.rule(name, alts, false);
    }

    /// Ports `visit_all_of` (parsers/gemma4.cpp:194 @b1b770d90).
    fn visitAllOf(self: *SchemaParser, children: []const *Node, name: []const u8) peg.Error!ParserId {
        if (children.len == 1) return self.visit(children[0], name);
        return self.b.ref("gemma4-value");
    }

    /// Ports `visit_array` (parsers/gemma4.cpp:201 @b1b770d90).
    fn visitArray(self: *SchemaParser, arr: *const json.schema.Array, name: []const u8) peg.Error!ParserId {
        const b = self.b;
        const ws = try b.space();
        const item = if (shaped(arr.items))
            try self.visit(arr.items, try std.mem.concat(self.a, u8, &.{ name, "-item" }))
        else
            try b.ref("gemma4-value");
        const elements = try b.plus(item, try b.zeroOrMore(try b.plus(try b.plus(try b.literal(","), ws), item)));
        const head = try b.plus(try b.literal("["), ws);
        const empty = try b.literal("]");
        const full = try b.plus(try b.plus(elements, ws), try b.literal("]"));
        return b.rule(name, try b.tag(Mapper.GEMMA4_ARRAY, try b.plus(head, try b.orr(empty, full))), false);
    }

    /// Ports `visit_tuple` (parsers/gemma4.cpp:209 @b1b770d90).
    fn visitTuple(self: *SchemaParser, items: []const *Node, name: []const u8) peg.Error!ParserId {
        const b = self.b;
        const ws = try b.space();
        var seq = try b.sequenceEmpty();
        for (items, 0..) |item, i| {
            if (i > 0) {
                seq = try b.plus(seq, try b.literal(","));
                seq = try b.plus(seq, ws);
            }
            seq = try b.plus(seq, try self.visit(item, try std.fmt.allocPrint(self.a, "{s}-{d}", .{ name, i })));
        }
        var body = try b.plus(try b.literal("["), ws);
        body = try b.plus(body, seq);
        body = try b.plus(body, ws);
        body = try b.plus(body, try b.literal("]"));
        return b.rule(name, try b.tag(Mapper.GEMMA4_ARRAY, body), false);
    }

    /// `parser + *rest`, or the parser alone without `rest`: the lambda
    /// `then_rest` in `visit_object`.
    fn thenRest(b: *peg.Builder, rest: ?ParserId, parser: ParserId) peg.Error!ParserId {
        return if (rest) |r| b.plus(parser, r) else parser;
    }

    /// `p_.literal(",") + ws + kv`.
    fn commaKv(b: *peg.Builder, ws: ParserId, kv: ParserId) peg.Error!ParserId {
        return b.plus(try b.plus(try b.literal(","), ws), kv);
    }

    fn lowerLess(a: Allocator, x: []const u8, y: []const u8) Allocator.Error!bool {
        const lx = try std.ascii.allocLowerString(a, x);
        const ly = try std.ascii.allocLowerString(a, y);
        return std.mem.order(u8, lx, ly) == .lt;
    }

    /// Ports `visit_object` (parsers/gemma4.cpp:221 @b1b770d90).
    fn visitObject(self: *SchemaParser, obj: *const json.schema.Object, name: []const u8) peg.Error!ParserId {
        const b = self.b;
        const ws = try b.space();

        const props = try self.a.alloc(*const json.schema.Property, obj.properties.len);
        for (obj.properties, 0..) |*prop, i| props[i] = prop;

        // The Gemma 4 jinja template sorts properties, so our rules need to match.
        // std::stable_sort on the ASCII-lowered names: an insertion sort is stable.
        var i_sort: usize = 1;
        while (i_sort < props.len) : (i_sort += 1) {
            const x = props[i_sort];
            var j = i_sort;
            while (j > 0 and try lowerLess(self.a, x.name, props[j - 1].name)) : (j -= 1) props[j] = props[j - 1];
            props[j] = x;
        }

        if (props.len == 0) {
            if (obj.additional_properties != null) return b.ref("gemma4-dict");
            const body = try b.plus(try b.plus(try b.literal("{"), ws), try b.literal("}"));
            return b.rule(name, try b.tag(Mapper.GEMMA4_DICT, body), false);
        }

        var rest: ?ParserId = null;
        var first: ?ParserId = null;
        if (obj.additional_properties != null) {
            const kv = try b.ref("gemma4-dict-kv");
            rest = try b.zeroOrMore(try commaKv(b, ws, kv));
            first = try b.optional(try b.plus(kv, rest.?));
        }

        var i = props.len;
        while (i > 0) {
            i -= 1;
            const prop = props[i];
            const k = try self.key(prop.name);
            const kws = try b.plus(k, ws);
            const value = try self.visit(prop.schema, try std.fmt.allocPrint(self.a, "{s}-{d}", .{ name, i }));
            const kv = try b.tag(Mapper.GEMMA4_DICT_KV, try b.plus(kws, value));

            var rest_i = try thenRest(b, rest, try commaKv(b, ws, kv));
            var first_i = try thenRest(b, rest, kv);
            if (!prop.required) {
                rest_i = try thenRest(b, rest, try b.optional(try commaKv(b, ws, kv)));
                first_i = if (first) |f| try b.orr(first_i, f) else try b.optional(first_i);
            }

            if (i > 0) {
                rest = try b.rule(try std.fmt.allocPrint(self.a, "{s}-rest-{d}", .{ name, i }), rest_i, false);
            }
            first = first_i;
        }

        var body = try b.plus(try b.literal("{"), ws);
        body = try b.plus(body, first.?);
        body = try b.plus(body, ws);
        body = try b.plus(body, try b.literal("}"));
        return b.rule(name, try b.tag(Mapper.GEMMA4_DICT, body), false);
    }
};

// -----------------------------------------------------------------------------
// The parser

const Build = struct {
    a: Allocator,
    inputs: *const GenerationParams,
    has_tools: bool,
    has_response_format: bool,
    extract_reasoning: bool,
    cb: *ChatBuilder = undefined,
    tool_choice: ParserId = undefined,
    /// the `what()` of the last throw `schemaArgs` caught
    what: []const u8 = "",

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        self.cb = p;
        const b = &p.b;
        const start = try b.rule("start", try b.optional(try b.literal("<|turn>model\n")), false);

        if (self.extract_reasoning) {
            const x = try b.plus(try b.plus(try b.literal("<|channel>thought"), try b.space()), try p.reasoning(try b.until("<channel|>")));
            _ = try b.rule("thought", try b.plus(x, try b.literal("<channel|>")), false);
        } else {
            const x = try b.plus(try b.plus(try b.literal("<|channel>thought"), try b.space()), try b.until("<channel|>"));
            _ = try b.rule("thought", try p.content(try b.plus(x, try b.literal("<channel|>"))), false);
        }

        const consume_empty_channels = try b.gbnf(try b.zeroOrMore(try b.plus(try b.literal("<|channel>"), try b.negate(try b.literal("thought")))), "");
        const peeked = try b.plus(try b.plus(try b.peek(try b.literal("<|channel>")), consume_empty_channels), try b.ref("thought"));
        const thought = try b.orr(peeked, try b.negate(try b.literal("<|channel>")));

        if (self.has_response_format) {
            const head = try b.shl(try b.literal("```json"), try p.content(try b.schemaJson(try b.json(), "response-format-schema", &self.inputs.json_schema, false)));
            const response_format = try b.shl(head, try b.literal("```"));
            return b.plus(try b.plus(start, try b.optional(thought)), response_format);
        }

        if (self.has_tools and self.inputs.tool_choice != .none) {
            try valueRules(b);

            self.tool_choice = try b.choiceEmpty();

            parsers.foreachFunction(&self.inputs.tools, self, tool) catch |e| return util.toPeg(e);

            const min: c_int = if (self.inputs.tool_choice == .required) 1 else 0;
            const max: c_int = if (self.inputs.parallel_tool_calls) -1 else 1;
            const tool_call = try b.triggerRule("tool-call", try b.repeat(try b.plusLit(try b.litPlus("<|tool_call>call:", self.tool_choice), "<tool_call|>"), min, max));

            if (self.inputs.tool_choice == .required) {
                return b.plus(try b.plus(start, thought), tool_call);
            }

            const scan_to_toolcall = try b.rule("scan-to-toolcall", try b.until("<|tool_call>"), false);
            const content = try b.rule("content", try p.content(try b.untilOneOf(&.{ "<|channel>", "<channel|>", "<|tool_call>" })), false);
            const message = try b.rule("message", try b.plus(thought, content), false);
            return b.plus(try b.plus(try b.plus(start, try b.zeroOrMore(message)), scan_to_toolcall), tool_call);
        }

        // Gemma 4 may emit an extra <|channel>thought\n<channel|> at the end of the content. It may
        // also emit a single trailing <channel|> token. Consume all complete reasoning blocks and
        // then stop at the first unmatched <channel|> token.
        const content = try b.rule("content", try p.content(try b.untilOneOf(&.{ "<|channel>", "<channel|>" })), false);
        const message = try b.rule("message", try b.plus(thought, content), false);
        return b.plus(start, try b.oneOrMore(message));
    }

    fn tool(self: *Build, tool_index: usize, t: *Value) Error!void {
        const function = t.at("function") catch |e| return util.fromJson(e);
        const name = try parsers.functionName(t);

        // Arguments follow the shape of the parameter schema when it is an object,
        // see gemma4_schema_parser, and are a generic dict otherwise.
        var args = self.cb.b.ref("gemma4-dict") catch |e| return util.fromPeg(e);
        if (self.schemaArgs(tool_index, function)) |shaped| {
            if (shaped) |x| args = x;
        } else |e| {
            if (e == error.OutOfMemory) return error.OutOfMemory;
            log.wrn("{s}: tool {s} uses the generic argument parser: {s}\n", .{ "operator()", name, self.what });
        }
        self.toolPeg(tool_index, name, args) catch |e| return util.fromPeg(e);
    }

    /// The `try` block of the tool lambda: the arguments shaped by the
    /// parameter schema, or null when it is not an object. A throw sets
    /// `self.what`, the `e.what()` the `catch` logs.
    fn schemaArgs(self: *Build, tool_index: usize, function: *const Value) anyerror!?ParserId {
        var params = msg_mod.toolParameters(self.a, function) catch |e| {
            self.what = exception.what(e);
            return e;
        };
        _ = &params;
        var doc = json.schema.fromJsonSchema(self.a, &params) catch |e| {
            self.what = if (e == error.JsonError) json.what() else json.schema.what();
            return e;
        };
        defer doc.deinit();
        if (doc.root.* != .object) return null;
        var sp: SchemaParser = .{
            .a = self.a,
            .b = &self.cb.b,
            .root = try parsers.indexName(self.a, "tool-", tool_index, "-args"),
        };
        return sp.visitRoot(doc.root) catch |e| {
            self.what = try self.a.dupe(u8, peg.what(e));
            return e;
        };
    }

    fn toolPeg(self: *Build, tool_index: usize, name: []const u8, args: ParserId) chat_peg.Error!void {
        const p = self.cb;
        const b = &p.b;
        const open = try p.toolOpen(try b.plus(try p.toolName(try b.literal(name)), try b.peek(try b.literal("{"))));
        const t = try p.tool(try b.sequence(&.{ open, try p.toolArgs(args) }));
        try b.orAssign(&self.tool_choice, try b.rule(try parsers.indexName(self.a, "tool-", tool_index, ""), t, false));
    }
};

/// Ports `common_chat_params_init_gemma4` (parsers/gemma4.cpp:283 @b1b770d90).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, null, null, null);
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, null, null, null);

    if (inputs.add_generation_prompt and std.mem.endsWith(u8, try data.prompt.str(a), "<turn|>\n")) {
        // This may happen if the model generates content + tool_call, the
        // template does not add the model's next turn and confuses the model
        // from emitting its proper reasoning token sequence.
        data.generation_prompt = "<|turn>model\n";
        try data.prompt.append(a, data.generation_prompt, false);
    }

    try data.message_delimiters.add(a, .user, "<|turn>user");
    try data.message_delimiters.add(a, .assistant, "<|turn>model");

    data.format = .peg_gemma4;
    data.supports_thinking = true;
    data.thinking_start_tag = "<|channel>thought";
    try data.thinking_end_tags.append(a, "<channel|>");

    try data.preserved_tokens.appendSlice(a, &.{ "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|turn>" });

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;

        const head: []const u8 = if (std.mem.endsWith(u8, try data.prompt.str(a), "<turn|>\n")) "<|turn>model\n" else "";
        data.generation_prompt = try std.mem.concat(a, u8, &.{ head, "<|channel>thought\n", m.reasoning_content });
        if (inputs.continue_final_message == .content) {
            data.generation_prompt = try std.mem.concat(a, u8, &.{ data.generation_prompt, "<channel|>", try m.renderContent(a, "\n\n") });
        }

        try data.prompt.append(a, data.generation_prompt, false);
    }

    const has_tools = parsers.hasTools(&inputs.tools);
    const has_response_format = !inputs.json_schema.isNull() and inputs.json_schema.isObject();
    const include_grammar = has_response_format or (has_tools and inputs.tool_choice != .none);
    const extract_reasoning = inputs.reasoning_format != .none;

    var build: Build = .{
        .a = a,
        .inputs = inputs,
        .has_tools = has_tools,
        .has_response_format = has_response_format,
        .extract_reasoning = extract_reasoning,
    };
    data.parser = chat_peg.buildChatPegParser(a, &build, Build.parser) catch |e| return util.fromPeg(e);
    const arena = &data.parser.?;

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format or (has_tools and inputs.tool_choice == .required));
        data.grammar = try parsers.grammar(a, arena, data.grammar_lazy);
        try data.addWordTrigger(a, "<|tool_call>");
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "google-gemma-4-31B-it.jinja applies with a tool and builds a parser and a grammar" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "llama.cpp/models/templates/google-gemma-4-31B-it.jinja", a, .limited(1 << 20));
    var t = try template.Template.init(std.testing.allocator, src, "<s>", "</s>");
    defer t.deinit();
    var p: GenerationParams = .{ .add_generation_prompt = true };
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    p.tools = try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]");
    const data = try init(a, &t, &p);
    try std.testing.expectEqual(.peg_gemma4, data.format);
    try std.testing.expect(data.prompt.parts.items.len > 0);
    try std.testing.expect(data.parser != null);
    try std.testing.expect(data.grammar.len > 0);
}
