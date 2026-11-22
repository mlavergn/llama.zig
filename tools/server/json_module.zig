//! The `json` module the server's chat imports name: `cli/json/`, reached
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
const cli_fit = @import("cli_fit");

pub const errors = cli_fit.json.errors;
pub const value = cli_fit.json.value;
pub const parser = cli_fit.json.parser;
pub const serializer = cli_fit.json.serializer;
pub const dtoa = cli_fit.json.dtoa;
pub const writer = cli_fit.json.writer;
pub const trie = cli_fit.json.trie;
pub const schema = cli_fit.json.schema;
pub const schema_to_grammar = cli_fit.json.schema_to_grammar;

pub const Error = cli_fit.json.Error;
pub const what = cli_fit.json.what;

pub const Value = cli_fit.json.Value;
pub const Type = cli_fit.json.Type;
pub const Map = cli_fit.json.Map;
pub const Member = cli_fit.json.Member;

pub const parse = cli_fit.json.parse;
pub const parseNoThrow = cli_fit.json.parseNoThrow;

pub const ErrorHandler = cli_fit.json.ErrorHandler;
pub const DumpOptions = cli_fit.json.DumpOptions;
pub const dump = cli_fit.json.dump;
pub const dumpSafe = cli_fit.json.dumpSafe;
pub const dumpOptions = cli_fit.json.dumpOptions;
pub const dumpTo = cli_fit.json.dumpTo;

pub const Object = cli_fit.json.Object;
pub const writeEscaped = cli_fit.json.writeEscaped;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
