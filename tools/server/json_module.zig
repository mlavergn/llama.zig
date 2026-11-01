//! The `json` module the server's Stage 9 imports name: `cli/json/`, reached
//! through `cli_fit`.
//!
//! # Provenance
//!
//! **Not a port.** `cli/common_chat/` imports `json` by name, and
//! `cli/fit.zig` -- which every tool reaches through `tools_common` --
//! already compiles `cli/json/` into its own module by a relative import. A
//! file may belong to one module only, so a second module rooted at
//! `cli/json/module.zig` cannot be built beside `cli_fit`; and two copies
//! would give two incompatible `json.Value`s. This file is that `json`
//! module, re-exporting `cli_fit`'s, so there is exactly one.

const std = @import("std");
const j = @import("cli_fit").json;

pub const errors = j.errors;
pub const value = j.value;
pub const parser = j.parser;
pub const serializer = j.serializer;
pub const dtoa = j.dtoa;
pub const writer = j.writer;
pub const trie = j.trie;
pub const schema = j.schema;
pub const schema_to_grammar = j.schema_to_grammar;

pub const Error = j.Error;
pub const what = j.what;

pub const Value = j.Value;
pub const Type = j.Type;
pub const Map = j.Map;
pub const Member = j.Member;

pub const parse = j.parse;
pub const parseNoThrow = j.parseNoThrow;

pub const ErrorHandler = j.ErrorHandler;
pub const DumpOptions = j.DumpOptions;
pub const dump = j.dump;
pub const dumpSafe = j.dumpSafe;
pub const dumpOptions = j.dumpOptions;
pub const dumpTo = j.dumpTo;

pub const Object = j.Object;
pub const writeEscaped = j.writeEscaped;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
