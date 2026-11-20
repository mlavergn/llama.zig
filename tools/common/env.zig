//! The process environment, as upstream's tools read it with `getenv`.
//!
//! # Provenance
//!
//! **Not a port.** On POSIX std hands the environment to `main` alone, in
//! `std.process.Init`; every tool's `main`, and the server's, passes it to
//! `set` before anything reads it. The tools never change their own
//! environment, so what `main` was given is what `getenv` would read.
//!
//! `llama-server-core`'s consumer calls `set` the same way. One that does
//! not gets libc's `getenv`: std exposes no POSIX environment outside
//! `main`, so that fallback is the C library's until it does.

const std = @import("std");

/// The environment `set` copied, each value stored with a terminating NUL so
/// `get` can hand out a C string. An `Environ.Map`, so a Windows lookup
/// ignores case as `getenv` does there.
var copy: ?std.process.Environ.Map = null;
var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);

/// Takes the environment `main` was given.
///
/// Parameters:
/// - `map`: `std.process.Init.environ_map`; read here, not kept.
pub fn set(map: *const std.process.Environ.Map) void {
    const a = arena_state.allocator();
    var m: std.process.Environ.Map = .init(a);
    var it = map.array_hash_map.iterator();
    while (it.next()) |e| {
        const v = a.dupeSentinel(u8, e.value_ptr.*, 0) catch return;
        m.put(e.key_ptr.*, v[0 .. v.len + 1]) catch return;
    }
    copy = m;
}

extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

/// `getenv(name)`.
///
/// Return: the value, valid for the process's life, or null when `name` is
/// not set.
pub fn get(name: [:0]const u8) ?[:0]const u8 {
    if (copy) |*m| {
        const v = m.get(name) orelse return null;
        return v[0 .. v.len - 1 :0];
    }
    return std.mem.span(getenv(name.ptr) orelse return null);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "set copies the environment and get answers from it, as C strings" {
    var map: std.process.Environ.Map = .init(std.testing.allocator);
    defer map.deinit();
    try map.put("LLAMA_TEST_A", "one");
    try map.put("LLAMA_TEST_EMPTY", "");
    set(&map);
    defer copy = null;
    const a = get("LLAMA_TEST_A") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("one", a);
    try std.testing.expectEqual(@as(u8, 0), a.ptr[a.len]);
    try std.testing.expectEqualStrings("", get("LLAMA_TEST_EMPTY").?);
    try std.testing.expect(get("LLAMA_TEST_MISSING") == null);
}
