//! Barrel for `cli/json/`: upstream's `common_json`, an insertion-ordered
//! JSON value with nlohmann's parser and serializer.
//!
//! # Provenance
//!
//! **Not a port.** The files it re-exports port `llama.cpp/common/json.h`,
//! `llama.cpp/common/json.cpp` and the parts of
//! `llama.cpp/vendor/nlohmann/json.hpp` they reach; each names its own.
//!
//! # Using it
//!
//! ```zig
//! var v = try json.parse(gpa, text);          // common_json::parse
//! defer v.deinit(gpa);
//! const name = try (try v.at("name")).get([]const u8);
//! try v.set(gpa, "n", try json.Value.from(gpa, 3));
//! const s = try json.dump(gpa, &v, 2);         // common_json::dump(2)
//! ```
//!
//! Every fallible call returns `json.Error`; on `error.JsonError`,
//! `json.what()` is the `what()` of the `common_json_error` upstream throws.
//! `make json-diff` holds all of it to upstream's `json.cpp` over the
//! vendored nlohmann, byte for byte.

const std = @import("std");

pub const errors = @import("errors.zig");
pub const value = @import("value.zig");
pub const parser = @import("parser.zig");
pub const serializer = @import("serializer.zig");
pub const dtoa = @import("dtoa.zig");
pub const writer = @import("writer.zig");

pub const Error = errors.Error;
pub const what = errors.what;

pub const Value = value.Value;
pub const Type = value.Type;
pub const Map = value.Map;
pub const Member = value.Member;

pub const parse = parser.parse;
pub const parseNoThrow = parser.parseNoThrow;

pub const ErrorHandler = serializer.ErrorHandler;
pub const DumpOptions = serializer.Options;
pub const dump = serializer.dump;
pub const dumpSafe = serializer.dumpSafe;
pub const dumpOptions = serializer.dumpOptions;
pub const dumpTo = serializer.dumpTo;

/// The `--log-jsonl` builder, kept for its callers in `fit.zig` and
/// `tools/common/log.zig`.
pub const Object = writer.Object;
pub const writeEscaped = writer.writeEscaped;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = errors;
    _ = value;
    _ = parser;
    _ = serializer;
    _ = dtoa;
    _ = writer;
}

test "a parsed document dumps back at every indent" {
    const a = std.testing.allocator;
    const text = "{\"b\":[1,-2,3.5,true,null,\"\\u00e9\"],\"a\":{}}";
    const compact = "{\"b\":[1,-2,3.5,true,null,\"\xc3\xa9\"],\"a\":{}}";
    var v = try parse(a, text);
    defer v.deinit(a);
    const c = try dump(a, &v, -1);
    defer a.free(c);
    try std.testing.expectEqualStrings(compact, c);
    const asc = try dumpOptions(a, &v, .{ .indent = 0, .ensure_ascii = true });
    defer a.free(asc);
    try std.testing.expectEqualStrings("{\n\"b\": [\n1,\n-2,\n3.5,\ntrue,\nnull,\n\"\\u00e9\"\n],\n\"a\": {}\n}", asc);
}
