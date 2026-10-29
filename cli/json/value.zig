//! `common_json`: a JSON value whose objects keep their keys in insertion
//! order.
//!
//! # Provenance
//!
//! Ports `common_json` from `llama.cpp/common/json.h` and
//! `llama.cpp/common/json.cpp`, and the parts of nlohmann's `ordered_json`
//! (`basic_json<ordered_map>`, `llama.cpp/vendor/nlohmann/json.hpp`, 3.12.0)
//! that `common_json` reaches: the value model, `ordered_map`, the accessors
//! and their exceptions, `operator==`, and `from_json` for the `get<T>()`
//! types `json.cpp` instantiates. llama.cpp v0.4.1, `b29c606e2`. Each
//! declaration names what it ports and the line it began at.
//!
//! # Shape
//!
//! `common_json` is a pimpl over an `ordered_json`; here the value *is* the
//! tree, a tagged union. `binary` is left out: nothing in `common_json` can
//! make one. Containers are unmanaged -- every call that allocates takes the
//! allocator, and `deinit` frees with the one that built the tree -- because
//! a value is a node of a tree, and storing an allocator in every node would
//! cost a word per element for nothing. The owner of the root frees it.
//!
//! A call that takes a `Value` argument takes ownership of it, **even when it
//! fails**, as the C++ moves its argument in before anything can throw.
//!
//! Errors are `errors.Error`; on `error.JsonError`, `errors.what()` holds the
//! exact `what()` text upstream's `common_json_error` carries.

const std = @import("std");
const builtin = @import("builtin");
const errors = @import("errors.zig");

const Error = errors.Error;
const fail = errors.fail;

/// `value_t`, in nlohmann's order. `binary` is not representable.
pub const Type = enum {
    null,
    object,
    array,
    string,
    boolean,
    number_integer,
    number_unsigned,
    number_float,
    discarded,
};

/// One member of an object: the pair `ordered_map` holds.
pub const Member = struct {
    key: []u8,
    value: Value,
};

/// Ports `ordered_map` (json.hpp:19746 @b29c606e2): a vector of pairs,
/// searched linearly, kept in insertion order.
pub const Map = struct {
    members: std.ArrayList(Member) = .empty,

    /// Frees every key and value.
    ///
    /// Parameters:
    /// - `gpa`: the allocator the map was built with.
    pub fn deinit(self: *Map, gpa: std.mem.Allocator) void {
        for (self.members.items) |*m| {
            gpa.free(m.key);
            m.value.deinit(gpa);
        }
        self.members.deinit(gpa);
        self.* = .{};
    }

    /// The index of `key`, or null.
    ///
    /// Parameters:
    /// - `key`: compared bytewise.
    ///
    /// Return: the member's index, or null if absent.
    pub fn find(self: *const Map, key: []const u8) ?usize {
        for (self.members.items, 0..) |m, i| {
            if (std.mem.eql(u8, m.key, key)) return i;
        }
        return null;
    }

    /// The value at `key`, or null.
    ///
    /// Parameters:
    /// - `key`: compared bytewise.
    ///
    /// Return: borrowed; valid until the map next changes.
    pub fn get(self: *const Map, key: []const u8) ?*Value {
        const i = self.find(key) orelse return null;
        return &self.members.items[i].value;
    }

    /// Ports `ordered_map`'s `emplace` (json.hpp:19771 @b29c606e2) as
    /// `operator[]` calls it, with a null value: the existing member if
    /// `key` is there, otherwise a new one at the end.
    ///
    /// Parameters:
    /// - `gpa`: the map's allocator; `key` is copied with it.
    /// - `key`: the member's name.
    ///
    /// Return: the member's value, borrowed until the map next changes.
    pub fn emplace(self: *Map, gpa: std.mem.Allocator, key: []const u8) Error!*Value {
        if (self.get(key)) |v| return v;
        try self.members.ensureUnusedCapacity(gpa, 1);
        const k = try gpa.dupe(u8, key);
        self.members.appendAssumeCapacity(.{ .key = k, .value = .null });
        return &self.members.items[self.members.items.len - 1].value;
    }

    /// Ports `ordered_map`'s key `erase` (json.hpp:19879 @b29c606e2):
    /// removes `key`, shifting the members after it down.
    ///
    /// Parameters:
    /// - `gpa`: the map's allocator.
    /// - `key`: the member to drop.
    ///
    /// Return: 1 if it was there, 0 if not.
    pub fn erase(self: *Map, gpa: std.mem.Allocator, key: []const u8) usize {
        const i = self.find(key) orelse return 0;
        var m = self.members.orderedRemove(i);
        gpa.free(m.key);
        m.value.deinit(gpa);
        return 1;
    }
};

/// The array type: `std::vector<basic_json>`.
pub const Array = std.ArrayList(Value);

/// One JSON value: `common_json`, which is an `ordered_json`.
pub const Value = union(Type) {
    null,
    object: Map,
    array: Array,
    string: []u8,
    boolean: bool,
    number_integer: i64,
    number_unsigned: u64,
    number_float: f64,
    discarded,

    // -------------------------------------------------------------------------
    // construction

    /// Ports `common_json`'s `object` (json.cpp:240 @b29c606e2): `{}`.
    pub fn emptyObject() Value {
        return .{ .object = .{} };
    }

    /// Ports `common_json`'s `array` (json.cpp:222 @b29c606e2): `[]`.
    pub fn emptyArray() Value {
        return .{ .array = .empty };
    }

    /// A string value, copied.
    ///
    /// Parameters:
    /// - `gpa`: the tree's allocator.
    /// - `s`: any bytes; need not be valid UTF-8 (`dump` decides).
    ///
    /// Return: the value, owned by the caller.
    pub fn initString(gpa: std.mem.Allocator, s: []const u8) Error!Value {
        return .{ .string = try gpa.dupe(u8, s) };
    }

    /// Ports `common_json_value`'s constructors and `to_json`
    /// (json.cpp:46 @b29c606e2): a value from a Zig one, as `common_json(x)`
    /// makes it. A signed integer is `number_integer` and an unsigned one
    /// `number_unsigned`, whatever its value; a float widens to `double`; a
    /// string slice is copied; `null` and an optional that is null are null;
    /// a `Value` is moved in.
    ///
    /// Parameters:
    /// - `gpa`: the tree's allocator.
    /// - `x`: a bool, integer, float, string slice, null, optional or `Value`.
    ///
    /// Return: the value, owned by the caller.
    pub fn from(gpa: std.mem.Allocator, x: anytype) Error!Value {
        const T = @TypeOf(x);
        if (T == Value) return x;
        switch (@typeInfo(T)) {
            .null => return .null,
            .bool => return .{ .boolean = x },
            .comptime_int => return .{ .number_integer = x },
            .int => |info| return if (info.signedness == .signed)
                .{ .number_integer = @intCast(x) }
            else
                .{ .number_unsigned = @intCast(x) },
            .comptime_float, .float => return .{ .number_float = @floatCast(x) },
            .optional => return if (x) |v| from(gpa, v) else .null,
            else => {},
        }
        // anything that coerces to a byte slice: string literals, arrays, slices
        const s: []const u8 = x;
        return initString(gpa, s);
    }

    /// Frees the tree.
    ///
    /// Parameters:
    /// - `gpa`: the allocator the tree was built with.
    pub fn deinit(self: *Value, gpa: std.mem.Allocator) void {
        switch (self.*) {
            .object => |*m| m.deinit(gpa),
            .array => |*a| {
                for (a.items) |*v| v.deinit(gpa);
                a.deinit(gpa);
            },
            .string => |s| gpa.free(s),
            else => {},
        }
        self.* = .null;
    }

    /// A deep copy: `common_json`'s copy constructor (json.cpp:167 @b29c606e2).
    ///
    /// Parameters:
    /// - `gpa`: the copy's allocator.
    ///
    /// Return: the copy, owned by the caller.
    pub fn clone(self: *const Value, gpa: std.mem.Allocator) Error!Value {
        switch (self.*) {
            .object => |m| {
                var out: Map = .{};
                errdefer out.deinit(gpa);
                try out.members.ensureTotalCapacity(gpa, m.members.items.len);
                for (m.members.items) |mem| {
                    const k = try gpa.dupe(u8, mem.key);
                    const v = mem.value.clone(gpa) catch |e| {
                        gpa.free(k);
                        return e;
                    };
                    out.members.appendAssumeCapacity(.{ .key = k, .value = v });
                }
                return .{ .object = out };
            },
            .array => |a| {
                var out: Array = .empty;
                errdefer {
                    for (out.items) |*v| v.deinit(gpa);
                    out.deinit(gpa);
                }
                try out.ensureTotalCapacity(gpa, a.items.len);
                for (a.items) |*v| out.appendAssumeCapacity(try v.clone(gpa));
                return .{ .array = out };
            },
            .string => |s| return .{ .string = try gpa.dupe(u8, s) },
            else => return self.*,
        }
    }

    // -------------------------------------------------------------------------
    // type

    /// Ports `type_name` (json.hpp:24200 @b29c606e2): what an exception calls
    /// this value's type.
    pub fn typeName(self: *const Value) []const u8 {
        return switch (self.*) {
            .null => "null",
            .object => "object",
            .array => "array",
            .string => "string",
            .boolean => "boolean",
            .discarded => "discarded",
            .number_integer, .number_unsigned, .number_float => "number",
        };
    }

    /// Ports `common_json`'s `is_null` (json.cpp:254 @b29c606e2).
    pub fn isNull(self: *const Value) bool {
        return self.* == .null;
    }
    /// Ports `common_json`'s `is_object` (json.cpp:255 @b29c606e2).
    pub fn isObject(self: *const Value) bool {
        return self.* == .object;
    }
    /// Ports `common_json`'s `is_array` (json.cpp:256 @b29c606e2).
    pub fn isArray(self: *const Value) bool {
        return self.* == .array;
    }
    /// Ports `common_json`'s `is_string` (json.cpp:257 @b29c606e2).
    pub fn isString(self: *const Value) bool {
        return self.* == .string;
    }
    /// Ports `common_json`'s `is_boolean` (json.cpp:258 @b29c606e2).
    pub fn isBoolean(self: *const Value) bool {
        return self.* == .boolean;
    }
    /// Ports `common_json`'s `is_number` (json.cpp:259 @b29c606e2).
    pub fn isNumber(self: *const Value) bool {
        return self.isNumberInteger() or self.isNumberFloat();
    }
    /// Ports `common_json`'s `is_number_integer` (json.cpp:260 @b29c606e2): signed
    /// or unsigned.
    pub fn isNumberInteger(self: *const Value) bool {
        return self.* == .number_integer or self.* == .number_unsigned;
    }
    /// Ports `common_json`'s `is_number_float` (json.cpp:261 @b29c606e2).
    pub fn isNumberFloat(self: *const Value) bool {
        return self.* == .number_float;
    }
    /// Ports `common_json`'s `is_discarded` (json.cpp:218 @b29c606e2): what
    /// `parseNoThrow` gives for invalid text.
    pub fn isDiscarded(self: *const Value) bool {
        return self.* == .discarded;
    }

    // -------------------------------------------------------------------------
    // capacity

    /// Ports `size` (json.hpp:22986 @b29c606e2): 0 for null, the element
    /// count of a container, 1 for anything else.
    pub fn size(self: *const Value) usize {
        return switch (self.*) {
            .null => 0,
            .array => |a| a.items.len,
            .object => |m| m.members.items.len,
            else => 1,
        };
    }

    /// Ports `empty` (json.hpp:22947 @b29c606e2): true for null and an empty
    /// container, false for anything else.
    pub fn empty(self: *const Value) bool {
        return switch (self.*) {
            .null => true,
            .array => |a| a.items.len == 0,
            .object => |m| m.members.items.len == 0,
            else => false,
        };
    }

    /// Ports `contains` (json.hpp:22770 @b29c606e2): false for anything but
    /// an object.
    pub fn contains(self: *const Value, key: []const u8) bool {
        return switch (self.*) {
            .object => |*m| m.find(key) != null,
            else => false,
        };
    }

    /// Ports `count` (json.hpp:22752 @b29c606e2): 1 if `key` is a member, 0
    /// otherwise.
    pub fn count(self: *const Value, key: []const u8) usize {
        return @intFromBool(self.contains(key));
    }

    // -------------------------------------------------------------------------
    // comparison

    /// Ports nlohmann's equality `operator` (json.hpp:23724 @b29c606e2),
    /// which `common_json`'s equality `operator` (json.cpp:270 @b29c606e2) calls: integers
    /// and floats compare by value, a signed and an unsigned integer by the
    /// unsigned one's bits as `int64_t`, NaN equals nothing, a discarded
    /// value equals nothing, and objects compare member by member **in
    /// order**, since `ordered_map` is a vector.
    pub fn eql(lhs: *const Value, rhs: *const Value) bool {
        const lt = std.meta.activeTag(lhs.*);
        const rt = std.meta.activeTag(rhs.*);
        if (lt == rt) {
            return switch (lhs.*) {
                .array => |a| blk: {
                    const b = rhs.array;
                    if (a.items.len != b.items.len) break :blk false;
                    for (a.items, b.items) |*x, *y| if (!x.eql(y)) break :blk false;
                    break :blk true;
                },
                .object => |m| blk: {
                    const n = rhs.object;
                    if (m.members.items.len != n.members.items.len) break :blk false;
                    for (m.members.items, n.members.items) |*x, *y| {
                        if (!std.mem.eql(u8, x.key, y.key) or !x.value.eql(&y.value)) break :blk false;
                    }
                    break :blk true;
                },
                .null => true,
                .string => |s| std.mem.eql(u8, s, rhs.string),
                .boolean => |b| b == rhs.boolean,
                .number_integer => |x| x == rhs.number_integer,
                .number_unsigned => |x| x == rhs.number_unsigned,
                .number_float => |x| x == rhs.number_float,
                .discarded => false,
            };
        }
        return switch (lt) {
            .number_integer => switch (rt) {
                .number_float => @as(f64, @floatFromInt(lhs.number_integer)) == rhs.number_float,
                .number_unsigned => lhs.number_integer == @as(i64, @bitCast(rhs.number_unsigned)),
                else => false,
            },
            .number_float => switch (rt) {
                .number_integer => lhs.number_float == @as(f64, @floatFromInt(rhs.number_integer)),
                .number_unsigned => lhs.number_float == @as(f64, @floatFromInt(rhs.number_unsigned)),
                else => false,
            },
            .number_unsigned => switch (rt) {
                .number_float => @as(f64, @floatFromInt(lhs.number_unsigned)) == rhs.number_float,
                .number_integer => @as(i64, @bitCast(lhs.number_unsigned)) == rhs.number_integer,
                else => false,
            },
            else => false,
        };
    }

    // -------------------------------------------------------------------------
    // element access

    /// Ports `common_json`'s key `at` (json.cpp:282 @b29c606e2)
    /// over `at` (json.hpp:22025 @b29c606e2): the member `key`, or an error
    /// if this is not an object or has no such member.
    ///
    /// Return: borrowed; valid until this value next changes.
    pub fn at(self: *const Value, key: []const u8) Error!*Value {
        switch (self.*) {
            .object => |*m| return m.get(key) orelse fail(.out_of_range, 403, "key '{s}' not found", .{key}),
            else => return fail(.type_error, 304, "cannot use at() with {s}", .{self.typeName()}),
        }
    }

    /// Ports `common_json`'s index `at` (json.cpp:284 @b29c606e2) over `at`
    /// (json.hpp:21979 @b29c606e2): element `idx`, bounds-checked.
    ///
    /// Return: borrowed; valid until this value next changes.
    pub fn atIndex(self: *const Value, idx: usize) Error!*Value {
        switch (self.*) {
            .array => |*a| {
                if (idx >= a.items.len) return fail(.out_of_range, 401, "array index {d} is out of range", .{idx});
                return &a.items[idx];
            },
            else => return fail(.type_error, 304, "cannot use at() with {s}", .{self.typeName()}),
        }
    }

    /// Ports `common_json`'s key `operator[]`, `common_json` (json.cpp:287 @b29c606e2)
    /// over `operator` `[]` (json.hpp:22160 @b29c606e2): the member `key`,
    /// added as null if absent; a null value becomes an object first.
    ///
    /// Return: borrowed; valid until this value next changes.
    pub fn index(self: *Value, gpa: std.mem.Allocator, key: []const u8) Error!*Value {
        if (self.* == .null) self.* = emptyObject();
        switch (self.*) {
            .object => |*m| return m.emplace(gpa, key),
            else => return fail(.type_error, 305, "cannot use operator[] with a string argument with {s}", .{self.typeName()}),
        }
    }

    /// Ports `common_json`'s index `operator[]`, `common_json` (json.cpp:289 @b29c606e2) over
    /// `operator` `[]` (json.hpp:22101 @b29c606e2): element `idx`, the array
    /// padded with nulls up to it; a null value becomes an array first.
    ///
    /// Return: borrowed; valid until this value next changes.
    pub fn indexAt(self: *Value, gpa: std.mem.Allocator, idx: usize) Error!*Value {
        if (self.* == .null) self.* = emptyArray();
        switch (self.*) {
            .array => |*a| {
                if (idx >= a.items.len) try a.appendNTimes(gpa, .null, idx + 1 - a.items.len);
                return &a.items[idx];
            },
            else => return fail(.type_error, 305, "cannot use operator[] with a numeric argument with {s}", .{self.typeName()}),
        }
    }

    /// Ports `common_json`'s `front` (json.cpp:292 @b29c606e2) over `front`
    /// (json.hpp:22435 @b29c606e2): the first element or member, or the
    /// value itself if it is not a container. Null throws, uncaught by
    /// `guard`, the same text.
    ///
    /// An empty container is `JSON_ASSERT`, which upstream defines as
    /// `GGML_ASSERT`, and aborts there and here.
    ///
    /// Return: borrowed; valid until this value next changes.
    pub fn front(self: *Value) Error!*Value {
        return self.end(true);
    }

    /// Ports `common_json`'s `back` (json.cpp:294 @b29c606e2) over `back`
    /// (json.hpp:22449 @b29c606e2): as `front`, at the other end.
    pub fn back(self: *Value) Error!*Value {
        return self.end(false);
    }

    fn end(self: *Value, first: bool) Error!*Value {
        switch (self.*) {
            .null => return fail(.invalid_iterator, 214, "cannot get value", .{}),
            .array => |*a| {
                if (a.items.len == 0) jsonAssert("m_it.array_iterator != m_object->m_data.m_value.array->end()");
                return &a.items[if (first) 0 else a.items.len - 1];
            },
            .object => |*m| {
                if (m.members.items.len == 0) jsonAssert("m_it.object_iterator != m_object->m_data.m_value.object->end()");
                return &m.members.items[if (first) 0 else m.members.items.len - 1].value;
            },
            else => return self,
        }
    }

    // -------------------------------------------------------------------------
    // modifiers

    /// Ports `clear` (json.hpp:23068 @b29c606e2), which
    /// `common_json`'s `clear` (json.cpp:297 @b29c606e2) calls: a number becomes
    /// 0 of its own kind, a boolean false, a string or container empty;
    /// null and discarded stay.
    pub fn clear(self: *Value, gpa: std.mem.Allocator) void {
        switch (self.*) {
            .number_integer => self.number_integer = 0,
            .number_unsigned => self.number_unsigned = 0,
            .number_float => self.number_float = 0.0,
            .boolean => self.boolean = false,
            .string => |s| {
                gpa.free(s);
                self.string = &.{};
            },
            .array => |*a| {
                for (a.items) |*v| v.deinit(gpa);
                a.clearRetainingCapacity();
            },
            .object => |*m| {
                for (m.members.items) |*mem| {
                    gpa.free(mem.key);
                    mem.value.deinit(gpa);
                }
                m.members.clearRetainingCapacity();
            },
            .null, .discarded => {},
        }
    }

    /// Ports `common_json`'s key `erase` (json.cpp:301 @b29c606e2)
    /// over `erase_internal` (json.hpp:22611 @b29c606e2): drops `key` if it
    /// is a member; an error if this is not an object.
    pub fn erase(self: *Value, gpa: std.mem.Allocator, key: []const u8) Error!void {
        switch (self.*) {
            .object => |*m| _ = m.erase(gpa, key),
            else => return fail(.type_error, 307, "cannot use erase() with {s}", .{self.typeName()}),
        }
    }

    /// Ports `common_json`'s index `erase` (json.cpp:305 @b29c606e2) over
    /// the index `erase` (json.hpp:22663 @b29c606e2).
    pub fn eraseIndex(self: *Value, gpa: std.mem.Allocator, idx: usize) Error!void {
        switch (self.*) {
            .array => |*a| {
                if (idx >= a.items.len) return fail(.out_of_range, 401, "array index {d} is out of range", .{idx});
                var v = a.orderedRemove(idx);
                v.deinit(gpa);
            },
            else => return fail(.type_error, 307, "cannot use erase() with {s}", .{self.typeName()}),
        }
    }

    /// Ports `common_json`'s `assign` (json.cpp:309 @b29c606e2) and
    /// `operator=`: replaces this value with `v`.
    ///
    /// Parameters:
    /// - `gpa`: the tree's allocator.
    /// - `v`: moved in.
    pub fn assign(self: *Value, gpa: std.mem.Allocator, v: Value) void {
        self.deinit(gpa);
        self.* = v;
    }

    /// Ports `common_json`'s `set` (json.cpp:313 @b29c606e2): `self[key] = v`,
    /// so an existing member keeps its place and takes the new value.
    ///
    /// Parameters:
    /// - `gpa`: the tree's allocator.
    /// - `key`: the member's name.
    /// - `v`: moved in, even on error.
    pub fn set(self: *Value, gpa: std.mem.Allocator, key: []const u8, v: Value) Error!void {
        var owned = v;
        const slot = self.index(gpa, key) catch |e| {
            owned.deinit(gpa);
            return e;
        };
        slot.assign(gpa, owned);
    }

    /// Ports `common_json`'s `push_back` (json.cpp:317 @b29c606e2) over
    /// `push_back` (json.hpp:23129 @b29c606e2): appends to an array; a null
    /// value becomes an array first.
    ///
    /// Parameters:
    /// - `gpa`: the tree's allocator.
    /// - `v`: moved in, even on error.
    pub fn pushBack(self: *Value, gpa: std.mem.Allocator, v: Value) Error!void {
        var owned = v;
        if (self.* != .null and self.* != .array) {
            owned.deinit(gpa);
            return fail(.type_error, 308, "cannot use push_back() with {s}", .{self.typeName()});
        }
        if (self.* == .null) self.* = emptyArray();
        self.array.append(gpa, owned) catch |e| {
            owned.deinit(gpa);
            return e;
        };
    }

    /// Ports `common_json`'s `insert` (json.cpp:331 @b29c606e2) over `insert`
    /// (json.hpp:23373 @b29c606e2): appends a copy of every element of
    /// `vals`. Only an array can be inserted into; inserting it into itself
    /// is an error; and `vals` that is not an array inserts nothing, because
    /// the C++ then reads two default `array_iterator`s, an empty range.
    ///
    /// Parameters:
    /// - `gpa`: the tree's allocator.
    /// - `vals`: copied from; may be any value.
    pub fn insert(self: *Value, gpa: std.mem.Allocator, vals: *const Value) Error!void {
        if (self.* != .array) return fail(.type_error, 309, "cannot use insert() with {s}", .{self.typeName()});
        if (vals == self) return fail(.invalid_iterator, 211, "passed iterators may not belong to container", .{});
        if (vals.* != .array) return;
        const src = vals.array.items;
        try self.array.ensureUnusedCapacity(gpa, src.len);
        var added: usize = 0;
        errdefer for (self.array.items[self.array.items.len - added ..]) |*v| v.deinit(gpa);
        for (src) |*v| {
            self.array.appendAssumeCapacity(try v.clone(gpa));
            added += 1;
        }
    }

    // -------------------------------------------------------------------------
    // get<T>()

    /// Ports `common_json`'s `get<T>()`, `COMMON_JSON_GET` (json.cpp:405 @b29c606e2) for a scalar
    /// `T`: `bool`, `[]const u8` (borrowed), or any integer or float type.
    ///
    /// The numeric rules are nlohmann's two `from_json` overloads.
    /// `get_arithmetic_value` (json.hpp:4857 @b29c606e2) serves
    /// `number_integer_t`, `number_unsigned_t` and `number_float_t` --
    /// `int64_t`, `uint64_t`, `double` -- and refuses a boolean; the other
    /// arithmetic `from_json` (json.hpp:5194 @b29c606e2) serves every other
    /// type and takes `true` as 1. Which C++ types are `int64_t` depends on
    /// the platform (`long` on Linux, `long long` on Darwin and Windows), so
    /// `usize`, `isize`, `c_long` and `c_longlong` follow the platform: see
    /// `isExactNumber`. Conversions are `static_cast`'s; a float out of an
    /// integer's range saturates, as AArch64's `fcvtz*` does.
    ///
    /// Return: the value; a string is borrowed until this value next changes.
    pub fn get(self: *const Value, comptime T: type) Error!T {
        if (T == bool) {
            if (self.* != .boolean) return fail(.type_error, 302, "type must be boolean, but is {s}", .{self.typeName()});
            return self.boolean;
        }
        if (T == []const u8) {
            if (self.* != .string) return fail(.type_error, 302, "type must be string, but is {s}", .{self.typeName()});
            return self.string;
        }
        switch (self.*) {
            .number_unsigned => |x| return castNumber(T, x),
            .number_integer => |x| return castNumber(T, x),
            .number_float => |x| return castNumber(T, x),
            .boolean => |b| if (!isExactNumber(T)) return castNumber(T, @as(u8, @intFromBool(b))),
            else => {},
        }
        return fail(.type_error, 302, "type must be number, but is {s}", .{self.typeName()});
    }

    /// Ports `get<std::vector<T>>()`, `COMMON_JSON_GET` (json.cpp:417 @b29c606e2)
    /// over the array `from_json_array_impl` (json.hpp:5048 @b29c606e2): every element
    /// through `get(T)`.
    ///
    /// Parameters:
    /// - `gpa`: allocates the result.
    /// - `T`: as `get` takes it.
    ///
    /// Return: owned by the caller; strings in it are borrowed from this value.
    pub fn getSlice(self: *const Value, gpa: std.mem.Allocator, comptime T: type) Error![]T {
        if (self.* != .array) return fail(.type_error, 302, "type must be array, but is {s}", .{self.typeName()});
        const out = try gpa.alloc(T, self.array.items.len);
        errdefer gpa.free(out);
        for (self.array.items, 0..) |*v, i| out[i] = try v.get(T);
        return out;
    }

    /// Ports `get<std::set<std::string>>()`, `COMMON_JSON_GET` (json.cpp:419 @b29c606e2):
    /// the array's strings, sorted bytewise as `std::string` compares, with
    /// duplicates dropped.
    ///
    /// Parameters:
    /// - `gpa`: allocates the result.
    ///
    /// Return: owned by the caller; the strings are borrowed from this value.
    pub fn getStringSet(self: *const Value, gpa: std.mem.Allocator) Error![][]const u8 {
        const all = try self.getSlice(gpa, []const u8);
        errdefer gpa.free(all);
        std.mem.sort([]const u8, all, {}, lessBytes);
        var n: usize = 0;
        for (all) |s| {
            if (n > 0 and std.mem.eql(u8, all[n - 1], s)) continue;
            all[n] = s;
            n += 1;
        }
        return gpa.realloc(all, n);
    }

    /// One entry of `getMap`.
    pub fn MapEntry(comptime T: type) type {
        return struct { key: []const u8, value: T };
    }

    /// Ports `get<std::unordered_map<std::string, T>>()`, `COMMON_JSON_GET`
    /// (json.cpp:422 @b29c606e2) over the object `from_json`
    /// (json.hpp:5162 @b29c606e2): every member's value through `get(T)`,
    /// in the object's order. (The C++ map's own order is its hash table's;
    /// a caller that iterates it must not depend on it.)
    ///
    /// Parameters:
    /// - `gpa`: allocates the result.
    /// - `T`: as `get` takes it.
    ///
    /// Return: owned by the caller; keys are borrowed from this value.
    pub fn getMap(self: *const Value, gpa: std.mem.Allocator, comptime T: type) Error![]MapEntry(T) {
        if (self.* != .object) return fail(.type_error, 302, "type must be object, but is {s}", .{self.typeName()});
        const ms = self.object.members.items;
        const out = try gpa.alloc(MapEntry(T), ms.len);
        errdefer gpa.free(out);
        for (ms, 0..) |*m, i| out[i] = .{ .key = m.key, .value = try m.value.get(T) };
        return out;
    }

    /// Ports the template `value` (json.h:227 @b29c606e2), `value(key, def)`:
    /// `contains(key) ? at(key).get<T>() : def`. Not nlohmann's `value()`:
    /// on anything but an object this gives `def` rather than throwing.
    ///
    /// Return: as `get`.
    pub fn value(self: *const Value, key: []const u8, comptime T: type, def: T) Error!T {
        if (!self.contains(key)) return def;
        return (try self.at(key)).get(T);
    }

    // -------------------------------------------------------------------------
    // iteration

    /// One step of `items()`: the key and the value.
    pub const Item = struct {
        /// the member's name; the decimal index for an array; "" otherwise.
        /// Borrowed until the next `next()` or until the value changes.
        key: []const u8,
        value: *Value,
    };

    /// Ports `common_json::items_view` and its iterator's `entry`
    /// (json.cpp:377 @b29c606e2): an object's members in insertion order, an
    /// array's elements keyed by index, a plain value once with key "",
    /// null never.
    pub const Items = struct {
        node: *Value,
        idx: usize = 0,
        n: usize,
        key_buf: [20]u8 = undefined,

        /// Return: the next item, or null at the end.
        pub fn next(self: *Items) ?Item {
            if (self.idx >= self.n) return null;
            const i = self.idx;
            self.idx += 1;
            switch (self.node.*) {
                .object => |*m| return .{ .key = m.members.items[i].key, .value = &m.members.items[i].value },
                .array => |*a| {
                    const k = std.fmt.bufPrint(&self.key_buf, "{d}", .{i}) catch unreachable;
                    return .{ .key = k, .value = &a.items[i] };
                },
                else => return .{ .key = "", .value = self.node },
            }
        }
    };

    /// Ports `common_json`'s `items` (json.cpp:394 @b29c606e2).
    ///
    /// Return: an iterator over this value; invalid once the value changes.
    pub fn items(self: *Value) Items {
        return .{ .node = self, .n = self.size() };
    }

    /// Ports `common_json::iterator`'s `key` (json.cpp:364 @b29c606e2): the
    /// name of member `i`. Anything but an object is an error, as the C++
    /// iterator's `key()` is.
    ///
    /// Return: borrowed until the value changes.
    pub fn keyAt(self: *const Value, i: usize) Error![]const u8 {
        switch (self.*) {
            .object => |*m| return m.members.items[i].key,
            else => return fail(.invalid_iterator, 207, "cannot use key() for non-object iterators", .{}),
        }
    }
};

/// `JSON_ASSERT`, which `json.cpp` defines as `GGML_ASSERT`.
fn jsonAssert(comptime cond: []const u8) noreturn {
    std.debug.print("{s}:{d}: GGML_ASSERT({s}) failed\n", .{ "json.hpp", 0, cond });
    std.process.abort();
}

fn lessBytes(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

/// Whether `T` is `int64_t`, `uint64_t` or `double` in the C++ on this
/// target -- `number_integer_t`, `number_unsigned_t`, `number_float_t` --
/// so `get<T>()` refuses a boolean. On Darwin and Windows `int64_t` is
/// `long long`, on Linux `long`; `size_t` is `unsigned long` everywhere but
/// Windows, so it is `uint64_t` on Linux and Windows and not on Darwin.
pub fn isExactNumber(comptime T: type) bool {
    if (T == i64 or T == u64 or T == f64) return true;
    const darwin = builtin.os.tag.isDarwin();
    const windows = builtin.os.tag == .windows;
    if (T == usize or T == isize) return !darwin;
    if (T == c_long or T == c_ulong) return !darwin and !windows and @bitSizeOf(c_long) == 64;
    if (T == c_longlong or T == c_ulonglong) return darwin or windows;
    return false;
}

/// `static_cast<T>(x)` as AArch64 clang compiles it: integers truncate to
/// the narrower width, a float converts with round-to-nearest, and a float
/// out of an integer's range saturates (`fcvtzs`/`fcvtzu` at 32 or 64 bits,
/// NaN to 0).
fn castNumber(comptime T: type, x: anytype) T {
    const X = @TypeOf(x);
    switch (@typeInfo(T)) {
        .float => return switch (@typeInfo(X)) {
            .float => @floatCast(x),
            else => @floatFromInt(x),
        },
        .int => |ti| {
            switch (@typeInfo(X)) {
                .float => {
                    const W = if (ti.bits <= 32)
                        @Int(ti.signedness, 32)
                    else
                        @Int(ti.signedness, 64);
                    const wide = saturate(W, x);
                    const U = @Int(.unsigned, @bitSizeOf(W));
                    const bits: U = @bitCast(wide);
                    return @bitCast(@as(@Int(.unsigned, ti.bits), @truncate(bits)));
                },
                .int => |xi| {
                    const UX = @Int(.unsigned, xi.bits);
                    const ux: UX = @bitCast(x);
                    if (ti.bits <= xi.bits) {
                        return @bitCast(@as(@Int(.unsigned, ti.bits), @truncate(ux)));
                    }
                    // widen: sign-extend a signed source, zero-extend an unsigned one
                    const wide: @Int(xi.signedness, ti.bits) = x;
                    return @bitCast(wide);
                },
                else => @compileError("not a number"),
            }
        },
        else => @compileError("get(T) takes bool, []const u8, or a number type"),
    }
}

fn saturate(comptime W: type, x: f64) W {
    if (std.math.isNan(x)) return 0;
    const lo: f64 = @floatFromInt(std.math.minInt(W));
    const hi: f64 = @floatFromInt(std.math.maxInt(W)); // rounds up to 2^n
    if (x <= lo) return std.math.minInt(W);
    if (x >= hi) return std.math.maxInt(W);
    return @intFromFloat(x);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

const testing = std.testing;

test "operator[] and set keep the first position and the last value" {
    const a = testing.allocator;
    var v: Value = .null;
    defer v.deinit(a);
    try v.set(a, "b", try Value.from(a, 1));
    try v.set(a, "a", try Value.from(a, "x"));
    try v.set(a, "b", try Value.from(a, @as(u64, 7)));
    try testing.expectEqual(@as(usize, 2), v.size());
    try testing.expectEqualStrings("b", v.object.members.items[0].key);
    try testing.expectEqual(@as(u64, 7), v.object.members.items[0].value.number_unsigned);
    try testing.expectError(error.JsonError, v.atIndex(0));
    try testing.expectEqualStrings("[json.exception.type_error.304] cannot use at() with object", errors.what());
    try testing.expectError(error.JsonError, v.at("zz"));
    try testing.expectEqualStrings("[json.exception.out_of_range.403] key 'zz' not found", errors.what());
    try v.erase(a, "b");
    try testing.expectEqual(@as(usize, 1), v.size());
}

test "get follows from_json's two numeric overloads" {
    const t: Value = .{ .boolean = true };
    try testing.expectEqual(@as(i32, 1), try t.get(i32));
    try testing.expectError(error.JsonError, t.get(i64));
    try testing.expectEqualStrings("[json.exception.type_error.302] type must be number, but is boolean", errors.what());
    const f: Value = .{ .number_float = 3.0e10 };
    try testing.expectEqual(@as(i32, std.math.maxInt(i32)), try f.get(i32));
    try testing.expectEqual(@as(i64, 30000000000), try f.get(i64));
    const u: Value = .{ .number_unsigned = std.math.maxInt(u64) };
    try testing.expectEqual(@as(i64, -1), try u.get(i64));
    try testing.expect(u.eql(&.{ .number_integer = -1 }));
}

test "insert copies an array, refuses itself, and ignores a non-array" {
    const a = testing.allocator;
    var x = Value.emptyArray();
    defer x.deinit(a);
    var y = Value.emptyArray();
    defer y.deinit(a);
    try y.pushBack(a, try Value.from(a, "s"));
    try x.insert(a, &y);
    try x.insert(a, &y);
    try testing.expectEqual(@as(usize, 2), x.size());
    try testing.expectError(error.JsonError, x.insert(a, &x));
    const o = Value.emptyObject();
    try x.insert(a, &o);
    try testing.expectEqual(@as(usize, 2), x.size());
    var c = try x.clone(a);
    defer c.deinit(a);
    try testing.expect(c.eql(&x));
}
