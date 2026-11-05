//! Barrel for `cli/peg/`: upstream's PEG parser -- the combinators, the
//! arena they build, partial parsing over a stream, the AST and GBNF
//! export. Upstream dropped the arena's JSON form (`save` and `load`) at
//! v0.7.0; the gates keep a serializer of their own,
//! `harness/peg_serialize.zig`.
//!
//! # Provenance
//!
//! **Not a port.** The files it re-exports port
//! `llama.cpp/common/peg-parser.h` and `llama.cpp/common/peg-parser.cpp`,
//! over `cli/json/`'s `common_trie` and JSON schema converter; each names
//! its own.
//!
//! It imports `json` and `llama` (for `regex.zig` and `libcxx.zig`) by
//! module name, so whatever builds it -- the server, a gate -- supplies
//! both.
//!
//! # Using it
//!
//! ```zig
//! var b = try peg.Builder.init(gpa);
//! defer b.deinit();
//! b.setRoot(try b.plus(try b.literal("<a>"), try b.json()));
//! var arena = try b.build();                       // common_peg_arena
//! defer arena.deinit();
//! var ctx = try peg.ParseContext.init(gpa, text, peg.Flags.lenient);
//! defer ctx.deinit();
//! const r = try arena.parse(&ctx, 0);              // r.type, r.end, r.nodes
//! const g = try peg.gbnf.grammar(gpa, &arena, false, .{});
//! ```
//!
//! Every fallible call returns `peg.Error`; `peg.what(e)` is the `what()`
//! of the exception upstream throws. `make peg-diff` holds all of it to
//! upstream's `peg-parser.cpp` and `chat-peg-parser.cpp`.

const std = @import("std");

pub const errors = @import("errors.zig");
pub const arena = @import("arena.zig");
pub const builder = @import("builder.zig");
pub const gbnf = @import("gbnf.zig");

pub const Error = errors.Error;
pub const what = errors.what;

pub const ParserId = arena.ParserId;
pub const invalid_parser_id = arena.invalid_parser_id;
pub const AstId = arena.AstId;
pub const invalid_ast_id = arena.invalid_ast_id;
pub const ResultType = arena.ResultType;
pub const Flags = arena.Flags;
pub const InvalidUtf8 = arena.InvalidUtf8;
pub const AstNode = arena.AstNode;
pub const AstArena = arena.AstArena;
pub const ParseResult = arena.ParseResult;
pub const ParseContext = arena.ParseContext;
pub const Parser = arena.Parser;
pub const SharedDoc = arena.SharedDoc;
pub const Arena = arena.Arena;
pub const Builder = builder.Builder;
pub const buildPegParser = builder.buildPegParser;
pub const GrammarBuilder = gbnf.GrammarBuilder;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = errors;
    _ = arena;
    _ = builder;
    _ = gbnf;
}
