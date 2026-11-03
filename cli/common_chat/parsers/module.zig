//! Barrel for `cli/common_chat/parsers/`: the hand-written parsers for the
//! templates the auto-parser cannot read, and the helpers they share.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/parsers.h` and
//! `llama.cpp/common/parsers/parsers.cpp` (llama.cpp v0.6.0, `d81235049`):
//! `foreach_function` and `foreach_parameter`. Each per-model file names its
//! own source; `common_chat_try_specialized_template` in `../templates.zig`
//! chooses between them.
//!
//! # The shape of one
//!
//! Every parser is `init(a, tmpl, inputs) Error!ChatParams` -- upstream's
//! `common_chat_params_init_<model>` -- over an arena `a` that owns
//! everything it returns. The template is rendered through
//! `template.directApplyImpl` and `template.generationPromptImpl`, the
//! parser built with `chat_peg.buildChatPegParser` and saved with
//! `peg.serialize`, the grammar exported with `peg.gbnf.grammar`.

const std = @import("std");
const json = @import("json");
const peg = @import("peg");
const exception = @import("../exception.zig");
const msg_mod = @import("../msg.zig");
const util = @import("../util.zig");
const log = @import("../log.zig");

const Allocator = std.mem.Allocator;
const Value = json.Value;
const Error = exception.Error;

pub const cohere2moe = @import("cohere2moe.zig");
pub const deepseek = @import("deepseek.zig");
pub const functionary_v3_2 = @import("functionary_v3_2.zig");
pub const gemma4 = @import("gemma4.zig");
pub const gigachat_v3 = @import("gigachat_v3.zig");
pub const gpt_oss = @import("gpt_oss.zig");
pub const kimi_k2 = @import("kimi_k2.zig");
pub const kimi_k3 = @import("kimi_k3.zig");
pub const lfm2 = @import("lfm2.zig");
pub const ling3 = @import("ling3.zig");
pub const llm_jp_harmony = @import("llm_jp_harmony.zig");
pub const minicpm5 = @import("minicpm5.zig");
pub const minimax_m3 = @import("minimax_m3.zig");
pub const ministral3 = @import("ministral3.zig");
pub const muse_glimmer = @import("muse_glimmer.zig");
pub const qwen3_coder = @import("qwen3_coder.zig");

/// Ports `foreach_function` (parsers.cpp:5 @d81235049): `f` over each
/// function tool of an OpenAI-style `tools`, logging and skipping the rest.
/// Iterates as a range-`for` over a `common_json` does: an array's
/// elements, an object's values, a scalar once, null never.
pub fn foreachFunction(tools: *const Value, ctx: anytype, comptime f: fn (@TypeOf(ctx), tool: *Value) Error!void) Error!void {
    var it = @constCast(tools).items();
    while (it.next()) |item| {
        const tool = item.value;
        const is_function = tool.contains("type") and blk: {
            const t = tool.at("type") catch |err| switch (err) {
                error.OutOfMemory, error.JsonError => unreachable,
            };
            break :blk t.isString() and std.mem.eql(u8, t.string, "function");
        };
        if (!is_function or !tool.contains("function")) {
            if (log.debugEnabled() or log.sink != null) {
                var arena_state = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
                defer arena_state.deinit();
                const dumped = json.dump(arena_state.allocator(), tool, 2) catch |e| return util.fromJson(e);
                log.inf("Skipping tool without function: {s}", .{dumped});
            }
            continue;
        }
        try f(ctx, tool);
    }
}

/// Ports `foreach_parameter` (parsers.cpp:15 @d81235049): `f` over each
/// property of a function tool's parameters schema, in schema order, with
/// the document that owns them -- nothing when the schema is not an object.
///
/// Parameters:
/// - `a`: an arena; the parameters' JSON is its.
/// - `function`: the tool's `function` member.
/// - `f`: called with the property and its document, which `f` may retain
///   (`peg.Builder.schemaDoc` does); released here after the last call.
pub fn foreachParameter(a: Allocator, function: *const Value, ctx: anytype, comptime f: fn (@TypeOf(ctx), prop: *const json.schema.Property, doc: *peg.SharedDoc) Error!void) Error!void {
    const params = try msg_mod.toolParameters(a, function);
    const doc = peg.SharedDoc.fromJson(a, &params) catch |e| return util.fromPeg(e);
    defer doc.release();
    switch (doc.doc.root.*) {
        .object => |o| for (o.properties) |*prop| try f(ctx, prop, doc),
        else => {},
    }
}

/// `parser.save()`: the arena, serialized as `common_chat_params`'s `parser`
/// carries it.
///
/// Return: owned by `a`.
pub fn save(a: Allocator, arena: *const peg.Arena) Error![]u8 {
    return peg.serialize.save(a, arena) catch |e| util.fromPeg(e);
}

/// `build_grammar([&](b) { parser.build_grammar(b, lazy); })`.
///
/// Return: owned by `a`.
pub fn grammar(a: Allocator, arena: *const peg.Arena, lazy: bool) Error![]u8 {
    return peg.gbnf.grammar(a, arena, lazy, .{}) catch |e| util.fromPeg(e);
}

/// `std::string name = tool.at("function").at("name")`: the conversion
/// throws nlohmann's type error when the name is not a string.
///
/// Return: borrowed from `tool`.
pub fn functionName(tool: *const Value) Error![]const u8 {
    const f = tool.at("function") catch |e| return util.fromJson(e);
    const n = f.at("name") catch |e| return util.fromJson(e);
    return n.get([]const u8) catch |e| util.fromJson(e);
}

/// `inputs.tools.is_array() && !inputs.tools.empty()`.
pub fn hasTools(tools: *const Value) bool {
    return tools.isArray() and !tools.empty();
}

/// `inputs.json_schema.is_object() && !inputs.json_schema.empty()`.
pub fn hasResponseFormat(schema: *const Value) bool {
    return schema.isObject() and !schema.empty();
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
