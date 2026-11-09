//! Reading a model's reply back into a message: reasoning, content and tool
//! calls, through the PEG parser `templates.apply` built for it.
//!
//! # Provenance
//!
//! Ports `common_chat_parser_params`, `common_chat_parse` and
//! `common_chat_peg_parse` from `llama.cpp/common/chat.h` and
//! `llama.cpp/common/chat.cpp` (llama.cpp v0.8.0, `8642541b2`). Each
//! declaration names what it ports and the line it began at.

const std = @import("std");
const json = @import("json");
const peg = @import("peg");
const exception = @import("exception.zig");
const msg_mod = @import("msg.zig");
const chat_peg = @import("chat_peg.zig");
const util = @import("util.zig");
const log = @import("log.zig");
const input_mod = @import("input.zig");

const Allocator = std.mem.Allocator;
const Error = exception.Error;
const Msg = msg_mod.Msg;
const Input = input_mod.Input;

/// Ports `common_chat_parser_params` (chat.h:347 @cd959bae9): per-message
/// parsing syntax, derived from `common_chat_params`.
pub const ParserParams = struct {
    format: msg_mod.Format = .content_only,
    /// Borrowed.
    generation_prompt: ?*const Input = null,
    debug: bool = false, // Enable debug output for PEG parser
    /// `common_peg_arena parser`: null for the empty arena. Borrowed.
    parser: ?*const peg.Arena = null,
};

/// Ports `common_chat_parse` (chat.cpp:1781 @cd959bae9).
pub fn parse(gpa: Allocator, input: *const Input, is_partial: bool, params: *const ParserParams) Error!Msg {
    return pegParse(gpa, params.parser, input, is_partial, params);
}

fn pureContent(_: void, p: *chat_peg.ChatBuilder) chat_peg.Error!peg.ParserId {
    return p.b.plus(try p.content(try p.b.rest()), try p.b.end());
}

fn mapperKind(format: msg_mod.Format) chat_peg.Mapper.Kind {
    return switch (format) {
        .peg_gemma4 => .gemma4,
        .peg_minimax_m3 => .minimax_m3,
        else => .base,
    };
}

/// Ports `common_chat_peg_parse` (chat.cpp:1787 @cd959bae9): the reply,
/// with the generation prompt in front, parsed leniently and mapped to an
/// assistant message. A partial reply that fails past its start still maps
/// what it matched; anything else that fails throws.
///
/// Parameters:
/// - `gpa`: owns the message; free it with `Msg.deinit`.
/// - `src_parser`: the parser; null or empty for content only.
///
/// Return: the message. `error.RuntimeError` when the reply does not match.
pub fn pegParse(gpa: Allocator, src_parser: ?*const peg.Arena, input: *const Input, is_partial: bool, params: *const ParserParams) Error!Msg {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    const empty = if (src_parser) |p| p.empty() else true;
    // upstream builds this once, a function-local static; building it again
    // each time gives the same arena
    var built: ?peg.Arena = null;
    defer if (built) |*b| b.deinit();
    if (empty) {
        built = chat_peg.buildChatPegParser(a, {}, pureContent) catch |e| return util.fromPeg(e);
        log.dbg("No parser definition detected, assuming pure content parser.", .{});
    }
    const parser: *const peg.Arena = if (built) |*b| b else src_parser.?;

    var effective_input = try input.clone(a);
    if (params.generation_prompt) |gp| try effective_input.prependInput(gp);

    var flags: u32 = peg.Flags.lenient;
    if (!is_partial) flags |= peg.Flags.complete;
    if (params.debug) flags |= peg.Flags.debug;

    var ctx = peg.ParseContext.initTokens(a, effective_input.text.items, effective_input.tokens.items, flags) catch |e| return util.fromPeg(e);
    defer ctx.deinit();
    const result = parser.parse(&ctx, 0) catch |e| return util.fromPeg(e);

    if (result.fail()) {
        // During partial parsing, return partial results if any AST nodes were captured
        // This allows streaming to work correctly for formats like FUNC_MARKDOWN_CODE_BLOCK
        if (is_partial and result.end > 0) {
            // Try to extract any partial results from what was successfully parsed
            const msg = try mapMessage(gpa, &ctx, &result, params.format);
            if (ctx.isDebug()) {
                const dumped = ctx.ast.dump(a) catch |e| return util.fromPeg(e);
                peg.errors.eprint("\nAST for partial parse (fail):\n{s}\n", .{dumped}) catch |e| return util.fromPeg(e);
            }
            return msg;
        }
        log.wrn("{s}: unparsed {s} output: {s}\n", .{ "common_chat_peg_parse", params.format.name(), try util.substrFrom(ctx.input, result.end) });
        log.dbg("{s}: full {s} output triggering error:\n=== BEGIN ===\n{s}\n=== END ===\n", .{ "common_chat_peg_parse", params.format.name(), ctx.input });
        return exception.throw(error.RuntimeError, "The model produced output that does not match the expected {s} format", .{params.format.name()});
    }

    var msg = try mapMessage(gpa, &ctx, &result, params.format);
    errdefer msg.deinit(gpa);

    if (ctx.isDebug()) {
        const dumped = ctx.ast.dump(a) catch |e| return util.fromPeg(e);
        peg.errors.eprint("\nAST for {s} parse:\n{s}\n", .{ if (is_partial) "partial" else "full", dumped }) catch |e| return util.fromPeg(e);
    }

    if (!is_partial and log.debugEnabled()) {
        var arr = try msg_mod.msgsToJsonOaicompat(a, (&msg)[0..1], false);
        const dumped = json.dump(a, arr.atIndex(0) catch |e| return util.fromJson(e), -1) catch |e| return util.fromJson(e);
        log.dbg("Parsed message: {s}\n", .{dumped});
    }
    return msg;
}

/// The message a mapper of the format's kind reads off the AST.
fn mapMessage(gpa: Allocator, ctx: *const peg.ParseContext, result: *const peg.ParseResult, format: msg_mod.Format) Error!Msg {
    var msg: Msg = .{};
    errdefer msg.deinit(gpa);
    try Msg.setField(gpa, &msg.role, "assistant");
    var mapper = chat_peg.Mapper.init(gpa, &msg, mapperKind(format));
    defer mapper.deinit();
    mapper.fromAst(&ctx.ast, result) catch |e| return util.fromPeg(e);
    return msg;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "without a parser the reply is all content" {
    const a = std.testing.allocator;
    var gp = try Input.fromText(a, "<a>");
    defer gp.deinit();
    var in = try Input.fromText(a, "hello");
    defer in.deinit();
    const params: ParserParams = .{ .generation_prompt = &gp };
    var msg = try parse(a, &in, false, &params);
    defer msg.deinit(a);
    try std.testing.expectEqualStrings("assistant", msg.role);
    try std.testing.expectEqualStrings("<a>hello", msg.content);
}
