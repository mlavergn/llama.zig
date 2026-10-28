//! The byte trie the UGM and RWKV tokenizers match tokens with.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-vocab.cpp` at v0.4.0 (`5266f24da`).
//!
//! The C++ keeps each node's children in a `std::map<char, naive_trie>` and
//! never iterates it, so the order of children is unobservable. Here every
//! edge lives in one hash map keyed by (node, byte), which answers the same
//! lookups.

const std = @import("std");
const impl = @import("../impl.zig");

const allocator = impl.allocator;

fn oom() noreturn {
    impl.abort("llama-vocab.cpp", 0, "out of memory");
}

/// Ports `naive_trie` (llama-vocab.cpp:28 @5266f24da).
pub const Trie = struct {
    pub const Node = struct {
        has_value: bool = false,
        value: i32 = 0,
    };

    /// A node's index; the root is 0.
    pub const Index = u32;
    pub const root: Index = 0;

    nodes: std.ArrayList(Node) = .empty,
    edges: std.AutoHashMapUnmanaged(u64, Index) = .empty,

    /// Return: a trie holding only its root.
    pub fn init() Trie {
        var t: Trie = .{};
        t.nodes.append(allocator, .{}) catch oom();
        return t;
    }

    /// Return: nothing.
    pub fn deinit(self: *Trie) void {
        self.nodes.deinit(allocator);
        self.edges.deinit(allocator);
    }

    fn edgeKey(node: Index, c: u8) u64 {
        return (@as(u64, node) << 8) | c;
    }

    /// Ports `naive_trie`'s `insert`: walks `key` from the root, creating
    /// nodes, and sets the last one's value. A repeated key overwrites.
    ///
    /// Parameters:
    /// - `key`: the bytes.
    /// - `value`: the value to store.
    ///
    /// Return: nothing.
    pub fn insert(self: *Trie, key: []const u8, value: i32) void {
        var node: Index = root;
        for (key) |c| {
            const r = self.edges.getOrPut(allocator, edgeKey(node, c)) catch oom();
            if (!r.found_existing) {
                self.nodes.append(allocator, .{}) catch oom();
                r.value_ptr.* = @intCast(self.nodes.items.len - 1);
            }
            node = r.value_ptr.*;
        }
        self.nodes.items[node].has_value = true;
        self.nodes.items[node].value = value;
    }

    /// Ports `naive_trie`'s `traverse`: the child on byte `c`.
    ///
    /// Return: the child, or null.
    pub fn traverse(self: *const Trie, node: Index, c: u8) ?Index {
        return self.edges.get(edgeKey(node, c));
    }

    /// Return: the node's value fields, borrowed until the next insert.
    pub fn get(self: *const Trie, node: Index) Node {
        return self.nodes.items[node];
    }

    /// Ports `naive_trie`'s `get_longest_prefix`: how far `key` can be walked
    /// from the root -- the deepest *path*, not the deepest stored value.
    ///
    /// Return: the length walked.
    pub fn longestPrefix(self: *const Trie, key: []const u8) usize {
        var node: Index = root;
        var offset: usize = 0;
        while (offset < key.len) : (offset += 1) {
            node = self.traverse(node, key[offset]) orelse break;
        }
        return offset;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the longest prefix is the longest path, valued or not" {
    var t = Trie.init();
    defer t.deinit();
    t.insert("abcd", 1);
    t.insert("ab", 2);
    try std.testing.expectEqual(@as(usize, 3), t.longestPrefix("abcx"));
    try std.testing.expectEqual(@as(usize, 0), t.longestPrefix("x"));
    const a = t.traverse(Trie.root, 'a').?;
    const b = t.traverse(a, 'b').?;
    try std.testing.expect(t.get(b).has_value);
    try std.testing.expectEqual(@as(i32, 2), t.get(b).value);
}
