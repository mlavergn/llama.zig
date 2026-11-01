//! The C++ standard library calls the tools' parsing leans on, as libc++
//! answers them.
//!
//! # Provenance
//!
//! **Not a port of a file.** `std::stoi` is libc++'s `string.cpp`:
//! `strtol` in base 10, then `invalid_argument` if nothing converted and
//! `out_of_range` if `errno` says so or the value leaves `int` -- with the
//! messages libc++ throws, which upstream prints through `e.what()`.
//! `HashMap` is libc++'s `std::unordered_map`, for its iteration order.

const std = @import("std");
const libcxx = @import("llama").libcxx;

extern fn strtol(s: [*:0]const u8, end: *?[*:0]const u8, base: c_int) c_long;

/// What `std::stoi` returns or throws.
pub const Stoi = union(enum) {
    ok: c_int,
    /// `std::invalid_argument`, `what()` "stoi: no conversion".
    no_conversion,
    /// `std::out_of_range`, `what()` "stoi: out of range".
    out_of_range,

    /// `e.what()` for the two throws.
    pub fn what(self: Stoi) [*:0]const u8 {
        return switch (self) {
            .ok => "",
            .no_conversion => "stoi: no conversion",
            .out_of_range => "stoi: out of range",
        };
    }
};

/// `std::stoi(s)`.
pub fn stoi(s: [:0]const u8) Stoi {
    var end: ?[*:0]const u8 = null;
    std.c._errno().* = 0;
    const v = strtol(s.ptr, &end, 10);
    if (end == s.ptr) return .no_conversion;
    if (std.c._errno().* == @backingInt(std.c.E.RANGE) or v < std.math.minInt(c_int) or v > std.math.maxInt(c_int)) return .out_of_range;
    return .{ .ok = @intCast(v) };
}

// -----------------------------------------------------------------------------
// std::unordered_map<K, V, Hash, Eq>

/// `std::unordered_map<K, V, Hash, Eq>` as libc++ 22 lays it out
/// (`__hash_table`), for its iteration order -- the generic form of
/// `src/llama/libcxx.zig`'s `StringMap`, with the hash and equality as
/// parameters.
///
/// One singly linked list of every node behind a sentinel; each bucket
/// points at the node *before* its group; a new key goes at the head of its
/// bucket's group, or at the head of the whole list when its bucket is
/// empty; the bucket count grows to `max(2 * bc + !pow2, ceil((size + 1) /
/// max_load_factor))`, rounded by `__rehash` to a prime, and `__do_rehash`
/// regroups in list order.
///
/// **Copies keep the structure.** libc++ 22's copy constructor and copy
/// assignment (`__copy_construct`) allocate the source's bucket count and
/// copy the list in order, so `clone` reproduces both -- except that a copy
/// of an empty map has no buckets, as the copy constructor leaves it.
///
/// A value type with `deinit(*V) void` and `clone(*const V, Allocator) !V`
/// declarations (another `HashMap`) is owned: `deinit` and `clone` recurse
/// into it, as the C++ destroys and copies a nested map.
///
/// Parameters:
/// - `K`: the key, copied in.
/// - `V`: the value.
/// - `hashFn`: the C++ `Hash`, returning `size_t`.
/// - `eqlFn`: the C++ `Pred`.
///
/// Return: the map type.
pub fn HashMap(comptime K: type, comptime V: type, comptime hashFn: fn (K) u64, comptime eqlFn: fn (K, K) bool) type {
    return struct {
        const Self = @This();
        const none = std.math.maxInt(u32);
        /// The sentinel, `__first_node_`.
        const head: u32 = 0;
        const owns = @typeInfo(V) == .@"struct" and @hasDecl(V, "deinit") and @hasDecl(V, "clone");

        const Node = struct {
            next: u32,
            hash: u64,
            key: K,
            value: V,
        };

        allocator: std.mem.Allocator,
        /// Node 0 is the sentinel; its `next` is the first element.
        nodes: std.ArrayList(Node) = .empty,
        /// `__bucket_list_`: the node before each bucket's group, or none.
        buckets: []u32 = &.{},
        len: usize = 0,
        max_load_factor: f32 = 1.0,

        /// The default constructor: no buckets.
        ///
        /// Parameters:
        /// - `allocator`: owns the nodes and buckets.
        ///
        /// Return: an empty map.
        pub fn init(allocator: std.mem.Allocator) error{OutOfMemory}!Self {
            var m: Self = .{ .allocator = allocator };
            try m.nodes.append(allocator, .{ .next = none, .hash = 0, .key = undefined, .value = undefined });
            return m;
        }

        /// Frees the map, and every owned value in it.
        pub fn deinit(self: *Self) void {
            if (owns) {
                for (self.nodes.items[1..]) |*n| n.value.deinit();
            }
            self.nodes.deinit(self.allocator);
            self.allocator.free(self.buckets);
        }

        /// The copy constructor (`__hash_table(const __hash_table&)` and
        /// `__copy_construct`): the source's bucket count and list order,
        /// owned values copied the same way.
        ///
        /// Parameters:
        /// - `allocator`: owns the copy.
        ///
        /// Return: the copy, freed with `deinit`.
        pub fn clone(self: *const Self, allocator: std.mem.Allocator) error{OutOfMemory}!Self {
            var m = try Self.init(allocator);
            errdefer m.deinit();
            m.max_load_factor = self.max_load_factor;
            if (self.len == 0) return m;
            const bc = self.buckets.len;
            m.buckets = try allocator.alloc(u32, bc);
            @memset(m.buckets, none);
            try m.nodes.ensureTotalCapacityPrecise(allocator, self.len + 1);
            var own: u32 = head;
            var current: usize = std.math.maxInt(usize);
            var it = self.nodes.items[head].next;
            while (it != none) : (it = self.nodes.items[it].next) {
                const src = &self.nodes.items[it];
                const value = if (owns) try src.value.clone(allocator) else src.value;
                m.nodes.appendAssumeCapacity(.{ .next = none, .hash = src.hash, .key = src.key, .value = value });
                const idx: u32 = @intCast(m.nodes.items.len - 1);
                const chash = constrain(src.hash, bc);
                if (chash != current) {
                    m.buckets[chash] = own;
                    current = chash;
                }
                m.nodes.items[own].next = idx;
                own = idx;
                m.len += 1;
            }
            return m;
        }

        fn constrain(h: u64, bc: usize) usize {
            if ((bc & (bc - 1)) == 0) return @intCast(h & (bc - 1));
            return @intCast(if (h < bc) h else h % bc);
        }

        fn isHashPow2(bc: usize) bool {
            return bc > 2 and (bc & (bc - 1)) == 0;
        }

        fn nextHashPow2(n: usize) usize {
            if (n < 2) return n;
            return @as(usize, 1) << @intCast(@bitSizeOf(usize) - @clz(n - 1));
        }

        /// `bucket_count()`.
        pub fn bucketCount(self: *const Self) usize {
            return self.buckets.len;
        }

        /// `size()`.
        pub fn count(self: *const Self) usize {
            return self.len;
        }

        fn findNode(self: *const Self, key: K, hash: u64) ?u32 {
            const bc = self.buckets.len;
            if (bc == 0) return null;
            const chash = constrain(hash, bc);
            var nd = self.buckets[chash];
            if (nd == none) return null;
            nd = self.nodes.items[nd].next;
            while (nd != none) : (nd = self.nodes.items[nd].next) {
                const h = self.nodes.items[nd].hash;
                if (!(h == hash or constrain(h, bc) == chash)) break;
                if (h == hash and eqlFn(self.nodes.items[nd].key, key)) return nd;
            }
            return null;
        }

        /// `find(key)`.
        ///
        /// Return: a pointer to the value, valid until the next insertion,
        /// or null.
        pub fn get(self: *const Self, key: K) ?*V {
            const nd = self.findNode(key, hashFn(key)) orelse return null;
            return &self.nodes.items[nd].value;
        }

        /// `emplace(key, value)`: inserts unless the key is there.
        ///
        /// Parameters:
        /// - `key`: the key.
        /// - `value`: moved in when inserted; left to the caller otherwise.
        ///
        /// Return: the slot, valid until the next insertion, and whether
        /// `value` was inserted.
        pub fn emplace(self: *Self, key: K, value: V) error{OutOfMemory}!struct { value: *V, inserted: bool } {
            const hash = hashFn(key);
            if (self.findNode(key, hash)) |nd| return .{ .value = &self.nodes.items[nd].value, .inserted = false };

            try self.nodes.append(self.allocator, .{ .next = none, .hash = hash, .key = key, .value = value });
            const h: u32 = @intCast(self.nodes.items.len - 1);

            var bc = self.buckets.len;
            const size1: f32 = @floatFromInt(self.len + 1);
            if (size1 > @as(f32, @floatFromInt(bc)) * self.max_load_factor) {
                const grow = 2 * bc + @intFromBool(!isHashPow2(bc));
                const need: usize = @intFromFloat(@ceil(size1 / self.max_load_factor));
                try self.rehash(@max(grow, need));
                bc = self.buckets.len;
            }
            const chash = constrain(hash, bc);
            const nodes = self.nodes.items;
            const pn = self.buckets[chash];
            if (pn == none) {
                nodes[h].next = nodes[head].next;
                nodes[head].next = h;
                self.buckets[chash] = head;
                if (nodes[h].next != none) self.buckets[constrain(nodes[nodes[h].next].hash, bc)] = h;
            } else {
                nodes[h].next = nodes[pn].next;
                nodes[pn].next = h;
            }
            self.len += 1;
            return .{ .value = &nodes[h].value, .inserted = true };
        }

        /// `operator[]`: the value slot for `key`, inserted with `default`
        /// if absent.
        ///
        /// Return: the slot, valid until the next insertion, and whether it
        /// was there already (`default` is then the caller's).
        pub fn getOrPut(self: *Self, key: K, default: V) error{OutOfMemory}!struct { value: *V, found: bool } {
            const r = try self.emplace(key, default);
            return .{ .value = r.value, .found = !r.inserted };
        }

        /// Ports `__hash_table::__rehash<true>`.
        fn rehash(self: *Self, n_in: usize) error{OutOfMemory}!void {
            var n = n_in;
            if (n == 1) {
                n = 2;
            } else if ((n & (n - 1)) != 0) {
                n = libcxx.nextPrime(n);
            }
            const bc = self.buckets.len;
            if (n > bc) {
                try self.doRehash(n);
            } else if (n < bc) {
                const need: usize = @intFromFloat(@ceil(@as(f32, @floatFromInt(self.len)) / self.max_load_factor));
                n = @max(n, if (isHashPow2(bc)) nextHashPow2(need) else libcxx.nextPrime(need));
                if (n < bc) try self.doRehash(n);
            }
        }

        /// Ports `__hash_table::__do_rehash<true>`.
        fn doRehash(self: *Self, bucket_count: usize) error{OutOfMemory}!void {
            self.allocator.free(self.buckets);
            self.buckets = &.{};
            if (bucket_count == 0) return;
            self.buckets = try self.allocator.alloc(u32, bucket_count);
            @memset(self.buckets, none);

            const nodes = self.nodes.items;
            var pp: u32 = head;
            var cp = nodes[pp].next;
            if (cp == none) return;

            var chash = constrain(nodes[cp].hash, bucket_count);
            self.buckets[chash] = pp;
            var phash = chash;
            pp = cp;
            cp = nodes[cp].next;
            while (cp != none) : (cp = nodes[pp].next) {
                chash = constrain(nodes[cp].hash, bucket_count);
                if (chash == phash) {
                    pp = cp;
                } else if (self.buckets[chash] == none) {
                    self.buckets[chash] = pp;
                    pp = cp;
                    phash = chash;
                } else {
                    const np = cp;
                    nodes[pp].next = nodes[np].next;
                    nodes[np].next = nodes[self.buckets[chash]].next;
                    nodes[self.buckets[chash]].next = cp;
                }
            }
        }

        /// One entry, as iteration yields it.
        pub const Entry = struct { key: K, value: *V };

        /// Iterates in libc++'s order: the list from the sentinel.
        pub const Iterator = struct {
            map: *const Self,
            at: u32,

            /// Return: the next entry, or null at the end.
            pub fn next(self: *Iterator) ?Entry {
                if (self.at == none) return null;
                const i = self.at;
                self.at = self.map.nodes.items[i].next;
                return .{ .key = self.map.nodes.items[i].key, .value = &self.map.nodes.items[i].value };
            }
        };

        /// Return: an iterator over every entry, valid until the next
        /// insertion.
        pub fn iterator(self: *const Self) Iterator {
            return .{ .map = self, .at = self.nodes.items[head].next };
        }
    };
}

/// `std::hash<int>`: the value, sign-extended to `size_t`.
pub fn hashInt(v: i32) u64 {
    return @bitCast(@as(i64, v));
}

/// `std::equal_to<int>`.
pub fn eqlInt(a: i32, b: i32) bool {
    return a == b;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "stoi converts a prefix and throws as libc++ does" {
    try std.testing.expectEqual(Stoi{ .ok = 12 }, stoi(" 12abc"));
    try std.testing.expectEqual(Stoi.no_conversion, stoi("abc"));
    try std.testing.expectEqual(Stoi.out_of_range, stoi("99999999999"));
}

test "the hash map iterates in libc++'s order, and a copy keeps it" {
    const M = HashMap(i32, i32, hashInt, eqlInt);
    var m = try M.init(std.testing.allocator);
    defer m.deinit();
    // derived by hand from __emplace_unique_key_args and __do_rehash:
    // 2 buckets, then 5 at the third key, then 11 at the sixth
    for ([_]i32{ 1, 2, 3, 8, 13, 4 }) |k| _ = try m.emplace(k, k * 10);
    try std.testing.expect(!(try m.emplace(8, 0)).inserted);
    try std.testing.expectEqual(@as(usize, 11), m.bucketCount());
    const want = [_]i32{ 4, 2, 13, 8, 3, 1 };
    var it = m.iterator();
    for (want) |k| try std.testing.expectEqual(k, it.next().?.key);
    try std.testing.expect(it.next() == null);
    try std.testing.expectEqual(@as(i32, 130), m.get(13).?.*);

    var copy = try m.clone(std.testing.allocator);
    defer copy.deinit();
    try std.testing.expectEqual(m.bucketCount(), copy.bucketCount());
    var it2 = copy.iterator();
    for (want) |k| try std.testing.expectEqual(k, it2.next().?.key);
    for (want) |k| try std.testing.expectEqual(k * 10, copy.get(k).?.*);
    // a copy goes on growing as the original would
    _ = try copy.emplace(24, 0);
    _ = try m.emplace(24, 0);
    var a = m.iterator();
    var b = copy.iterator();
    while (a.next()) |e| try std.testing.expectEqual(e.key, b.next().?.key);

    // a copy of an emptied-out map has no buckets, as the copy constructor
    // leaves it
    var empty = try M.init(std.testing.allocator);
    defer empty.deinit();
    try empty.doRehash(7);
    var e2 = try empty.clone(std.testing.allocator);
    defer e2.deinit();
    try std.testing.expectEqual(@as(usize, 0), e2.bucketCount());
}

test "a hash map of hash maps deep-copies and frees its values" {
    const Inner = HashMap(i32, i32, hashInt, eqlInt);
    const Outer = HashMap(i32, Inner, hashInt, eqlInt);
    var o = try Outer.init(std.testing.allocator);
    defer o.deinit();
    var in = try Inner.init(std.testing.allocator);
    _ = try in.emplace(7, 1);
    _ = try o.emplace(3, in);
    var copy = try o.clone(std.testing.allocator);
    defer copy.deinit();
    copy.get(3).?.get(7).?.* = 5;
    try std.testing.expectEqual(@as(i32, 1), o.get(3).?.get(7).?.*);
}
