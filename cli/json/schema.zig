//! `common_chat_schema`: a JSON schema, normalised into the subset
//! `json_schema_to_grammar` converts.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/json-schema.h` and
//! `llama.cpp/common/json-schema.cpp` (llama.cpp v0.7.0, `620d2225c`).
//! Each declaration names what it ports and the line it began at.
//!
//! # Shape
//!
//! The C++ is a class hierarchy, one subclass of `common_chat_schema` per
//! `node_kind`, owned through `std::unique_ptr`; here a node is a tagged
//! union, and every node, string and copied JSON value of a document lives
//! in the document's arena, so `Document.deinit` frees the lot. Nodes are
//! never moved once built, which is what lets a `$ref` node point at its
//! target as the C++'s raw `target` pointer does.
//!
//! # Errors
//!
//! Where the C++ throws, this returns an error and leaves the exception's
//! `what()` in a per-thread buffer, `what()`, cut at its first NUL byte as
//! `std::runtime_error::what()` is. `error.RuntimeError` is a
//! `std::runtime_error` (`fail`'s "JSON schema error at ..."),
//! `error.InvalidArgument` a `std::invalid_argument`, and `error.JsonError`
//! a `common_json_error`, which is a `std::runtime_error` too. The three
//! are kept apart because `json_schema_to_grammar` catches the
//! `runtime_error`s and not the `invalid_argument`, and the gate compares
//! which one escaped.

const std = @import("std");
const json = @import("module.zig");

const Value = json.Value;

/// Every fallible call here and in `schema_to_grammar.zig`. See the file
/// comment for which C++ exception each is.
pub const Error = error{ RuntimeError, InvalidArgument, JsonError, OutOfMemory };

threadlocal var what_buf: std.ArrayList(u8) = .empty;

/// The `what()` of the last error returned on this thread, for every error
/// but `OutOfMemory`.
///
/// Return: borrowed; valid until the next failing call on this thread.
pub fn what() []const u8 {
    return what_buf.items;
}

/// Sets `what()` and returns `err`. The text is cut at its first NUL, as
/// `what()` -- a `const char *` -- reads it.
///
/// Parameters:
/// - `err`: the error to return.
/// - `parts`: concatenated into the text.
///
/// Return: `err`, or `error.OutOfMemory` if the text could not be kept.
pub fn raise(err: Error, parts: []const []const u8) Error {
    // built apart from `what_buf`, since a part may be `what()` itself, as
    // when `json_schema_to_grammar` wraps an error it caught
    const a = std.heap.smp_allocator;
    var text: std.ArrayList(u8) = .empty;
    for (parts) |p| text.appendSlice(a, p) catch {
        text.deinit(a);
        return error.OutOfMemory;
    };
    if (std.mem.indexOfScalar(u8, text.items, 0)) |nul| text.shrinkRetainingCapacity(nul);
    what_buf.deinit(a);
    what_buf = text;
    return err;
}

/// A `cli/json/` error as one of ours: `error.JsonError` carries
/// `json.what()` over.
///
/// Return: the mapped error.
pub fn fromJson(e: json.Error) Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.JsonError => raise(error.JsonError, &.{json.what()}),
    };
}

/// Ports `node_kind` (json-schema.h:15 @620d2225c), a member of
/// `common_chat_schema`, in its order.
pub const Kind = enum {
    any,
    ref,
    any_of,
    all_of,
    @"const",
    @"enum",
    null,
    boolean,
    number,
    integer,
    string,
    array,
    tuple,
    object,
};

/// Ports `value_type` (json-schema.h:32 @620d2225c), a member of
/// `common_chat_schema`, in its order: the bit each takes in a `TypeSet`.
pub const ValueType = enum(u5) {
    null,
    boolean,
    number,
    integer,
    string,
    array,
    object,
};

/// Ports `string_format` (json-schema.h:42 @620d2225c), a member of
/// `common_chat_schema`.
pub const StringFormat = enum {
    none,
    uuid,
    date,
    time,
    date_time,
};

/// Ports `type_set` (json-schema.h:50 @620d2225c), a member of
/// `common_chat_schema`: a bit mask over `ValueType`.
pub const TypeSet = struct {
    mask: u32 = 0,

    /// Return: the set of the given types.
    pub fn of(types: []const ValueType) TypeSet {
        var s: TypeSet = .{};
        for (types) |t| s.add(t);
        return s;
    }

    /// Ports `all` (json-schema.h:61 @620d2225c).
    pub fn all() TypeSet {
        return of(&.{ .null, .boolean, .number, .integer, .string, .array, .object });
    }

    /// Ports `add` (json-schema.h:65 @620d2225c).
    pub fn add(self: *TypeSet, t: ValueType) void {
        self.mask |= @as(u32, 1) << @backingInt(t);
    }

    /// Ports `has` (json-schema.h:67 @620d2225c).
    pub fn has(self: TypeSet, t: ValueType) bool {
        return self.mask & (@as(u32, 1) << @backingInt(t)) != 0;
    }

    /// Ports `is_only` (json-schema.h:68 @620d2225c).
    pub fn isOnly(self: TypeSet, t: ValueType) bool {
        return self.mask == @as(u32, 1) << @backingInt(t);
    }

    /// Ports `empty` (json-schema.h:69 @620d2225c).
    pub fn empty(self: TypeSet) bool {
        return self.mask == 0;
    }
};

/// Ports `common_chat_schema_ref` (json-schema.h:97 @620d2225c): only
/// references into the same document.
pub const Ref = struct {
    ref: []const u8,
    /// set once every `$ref` is built; owned by the document's `refs`
    target: ?*const Node = null,
};

/// Ports `common_chat_schema_integer` (json-schema.h:146 @620d2225c):
/// inclusive bounds, `exclusiveMinimum`/`exclusiveMaximum` folded in.
pub const Integer = struct {
    /// `minInt(i64)` for unbounded
    minimum: i64 = std.math.minInt(i64),
    /// `maxInt(i64)` for unbounded
    maximum: i64 = std.math.maxInt(i64),
};

/// Ports `common_chat_schema_string` (json-schema.h:153 @620d2225c).
pub const String = struct {
    /// empty when absent
    pattern: []const u8 = "",
    format: StringFormat = .none,
    min_length: c_int = 0,
    /// -1 for unbounded
    max_length: c_int = -1,
};

/// Ports `common_chat_schema_array` (json-schema.h:162 @620d2225c).
pub const Array = struct {
    /// an `any` node when `items` is absent
    items: *Node,
    min_items: c_int = 0,
    /// -1 for unbounded
    max_items: c_int = -1,
};

/// Ports `common_chat_schema_property` (json-schema.h:176 @620d2225c).
pub const Property = struct {
    name: []const u8,
    schema: *Node,
    required: bool = false,
};

/// Ports `common_chat_schema_object` (json-schema.h:182 @620d2225c).
pub const Object = struct {
    /// in schema order
    properties: []Property = &.{},
    /// null when not allowed
    additional_properties: ?*Node = null,
};

/// Ports `common_chat_schema` (json-schema.h:14 @620d2225c) and its
/// subclasses: one node of a schema.
pub const Node = union(Kind) {
    /// `common_chat_schema_any` (json-schema.h:92 @620d2225c)
    any,
    ref: Ref,
    /// `common_chat_schema_any_of` (json-schema.h:107 @620d2225c): oneOf,
    /// anyOf, or a `type` array expanded to one alternative per type
    any_of: []*Node,
    /// `common_chat_schema_all_of` (json-schema.h:113 @620d2225c)
    all_of: []*Node,
    /// `common_chat_schema_const` (json-schema.h:119 @620d2225c)
    @"const": Value,
    /// `common_chat_schema_enum` (json-schema.h:127 @620d2225c)
    @"enum": []Value,
    null,
    boolean,
    number,
    integer: Integer,
    string: String,
    array: Array,
    /// `common_chat_schema_tuple` (json-schema.h:170 @620d2225c)
    tuple: []*Node,
    object: Object,

    /// The node's `kind()`.
    pub fn kind(self: *const Node) Kind {
        return self.*;
    }

    /// Ports `value_types` (json-schema.cpp:424 @620d2225c): the JSON types a
    /// value matching this node may have.
    pub fn valueTypes(self: *const Node) TypeSet {
        return valueTypesImpl(self, null);
    }

    /// Ports `may_be_string` (json-schema.cpp:478 @620d2225c): whether a
    /// matching value may be a string, through any branch.
    pub fn mayBeString(self: *const Node) bool {
        return mayBeStringImpl(self, null);
    }
};

/// One entry of `Document.refs`.
pub const RefEntry = struct {
    key: []const u8,
    node: *Node,
};

/// Ports `common_chat_schema_document` (json-schema.h:189 @620d2225c).
pub const Document = struct {
    arena: std.heap.ArenaAllocator,
    root: *Node,
    /// every `$ref` target, by its `$ref` string, in `std::map` order
    /// (bytewise)
    refs: []RefEntry,

    /// Frees every node.
    pub fn deinit(self: *Document) void {
        self.arena.deinit();
    }

    /// `refs.at(key)`, or null when absent.
    pub fn findRef(self: *const Document, key: []const u8) ?*Node {
        for (self.refs) |e| {
            if (std.mem.eql(u8, e.key, key)) return e.node;
        }
        return null;
    }
};

/// Ports `common_chat_schema_from_json` (json-schema.cpp:340 @620d2225c):
/// builds the schema, or fails when it falls outside the supported subset.
///
/// Parameters:
/// - `gpa`: backs the document's arena.
/// - `schema`: the schema; read, never kept.
///
/// Return: the document, freed with `deinit`; `error.RuntimeError` with
/// `what()` set when the schema is outside the subset.
pub fn fromJsonSchema(gpa: std.mem.Allocator, schema: *const Value) Error!Document {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    var b: Builder = .{ .a = arena.allocator(), .root = schema };
    const root = try b.buildNode(schema, "#");

    // `build` (json-schema.cpp:328 @620d2225c): the targets move into the
    // document, then every ref node gets its target.
    const entries = try b.a.alloc(RefEntry, b.refs.count());
    var it = b.refs.iterator();
    var n: usize = 0;
    while (it.next()) |e| : (n += 1) {
        entries[n] = .{ .key = e.key_ptr.*, .node = e.value_ptr.*.? };
    }
    std.mem.sort(RefEntry, entries, {}, struct {
        fn lt(_: void, x: RefEntry, y: RefEntry) bool {
            return std.mem.order(u8, x.key, y.key) == .lt;
        }
    }.lt);
    for (b.pending.items) |node| {
        node.ref.target = b.refs.get(node.ref.ref).?.?;
    }
    return .{ .arena = arena, .root = root, .refs = entries };
}

/// Ports `common_chat_schema_builder` (json-schema.cpp:12 @620d2225c).
const Builder = struct {
    a: std.mem.Allocator,
    root: *const Value,
    /// the targets built here; a key maps to null while its target is being
    /// built, so a cycle back to it stops there
    refs: std.StringHashMapUnmanaged(?*Node) = .empty,
    /// ref nodes, which get their target once every `$ref` is built
    pending: std.ArrayList(*Node) = .empty,

    /// Ports `fail` (json-schema.cpp:22 @620d2225c).
    fn fail(path: []const u8, msg: []const []const u8) Error {
        const a = std.heap.smp_allocator;
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(a);
        for (msg) |m| text.appendSlice(a, m) catch return error.OutOfMemory;
        return raise(error.RuntimeError, &.{ "JSON schema error at ", path, ": ", text.items });
    }

    fn newNode(self: *Builder, n: Node) Error!*Node {
        const p = try self.a.create(Node);
        p.* = n;
        return p;
    }

    fn cat(self: *Builder, parts: []const []const u8) Error![]u8 {
        return std.mem.concat(self.a, u8, parts);
    }

    fn index(self: *Builder, path: []const u8, sep: []const u8, i: usize) Error![]u8 {
        var buf: [24]u8 = undefined;
        const s = std.fmt.bufPrint(&buf, "{d}", .{i}) catch |err| switch (err) {
            error.NoSpaceLeft => unreachable,
        };
        return self.cat(&.{ path, sep, s });
    }

    /// Ports `get_count` (json-schema.cpp:26 @620d2225c): a non-negative
    /// `int`, read as `get<int>()` reads it -- an integer out of range
    /// truncates first.
    fn getCount(schema: *const Value, key: []const u8, path: []const u8, def: c_int) Error!c_int {
        if (!schema.contains(key)) return def;
        const v = schema.at(key) catch |e| return fromJson(e);
        if (!v.isNumberInteger() or (v.get(c_int) catch |e| return fromJson(e)) < 0) {
            return fail(path, &.{ key, " must be a non-negative integer" });
        }
        return v.get(c_int) catch |e| return fromJson(e);
    }

    /// Ports `get_bound` (json-schema.cpp:38 @620d2225c): a fractional
    /// bound rounds inwards. `(int64_t)` of a double out of range is
    /// undefined in C++; this saturates, as AArch64's `fcvtzs` does.
    fn getBound(schema: *const Value, key: []const u8, path: []const u8, round_up: bool) Error!i64 {
        const v = schema.at(key) catch |e| return fromJson(e);
        if (v.isNumberInteger()) return v.get(i64) catch |e| return fromJson(e);
        if (!v.isNumber()) return fail(path, &.{ key, " must be a number" });
        const d = v.get(f64) catch |e| return fromJson(e);
        const r: Value = .{ .number_float = if (round_up) @ceil(d) else @floor(d) };
        return r.get(i64) catch |e| return fromJson(e);
    }

    /// Ports `get_format` (json-schema.cpp:50 @620d2225c).
    fn getFormat(schema: *const Value, path: []const u8) Error!StringFormat {
        if (!schema.contains("format")) return .none;
        const v = schema.at("format") catch |e| return fromJson(e);
        if (!v.isString()) return fail(path, &.{"format must be a string"});
        const f = v.string;
        if (std.mem.eql(u8, f, "date")) return .date;
        if (std.mem.eql(u8, f, "time")) return .time;
        if (std.mem.eql(u8, f, "date-time")) return .date_time;
        if (std.mem.eql(u8, f, "uuid") or
            (f.len == 5 and std.mem.startsWith(u8, f, "uuid") and f[4] >= '1' and f[4] <= '5'))
        {
            return .uuid;
        }
        return .none;
    }

    /// Ports `resolve_ref` (json-schema.cpp:74 @620d2225c): walks the
    /// pointer's segments as written -- no `~0`/`~1` unescaping -- an array
    /// index read by `std::stoull`.
    fn resolveRef(self: *Builder, ref: []const u8, path: []const u8) Error!*const Value {
        var target: *const Value = self.root;
        // `string_split(ref.substr(1), "/")`; the first token is always
        // the empty one before the leading '/'
        var tokens = std.mem.splitSequence(u8, ref[1..], "/");
        _ = tokens.next();
        while (tokens.next()) |sel| {
            if (target.isObject() and target.contains(sel)) {
                target = target.at(sel) catch |e| return fromJson(e);
            } else if (target.isArray()) {
                const idx = stoull(sel) orelse target.size();
                if (idx >= target.size()) {
                    return fail(path, &.{ "cannot resolve $ref ", ref, ", ", sel, " is out of range" });
                }
                target = &target.array.items[@intCast(idx)];
            } else {
                return fail(path, &.{ "cannot resolve $ref ", ref, ", ", sel, " not found" });
            }
        }
        return target;
    }

    /// Ports `build_ref` (json-schema.cpp:99 @620d2225c).
    fn buildRef(self: *Builder, value: *const Value, path: []const u8) Error!*Node {
        if (!value.isString()) return fail(path, &.{"$ref must be a string"});
        const ref = try self.a.dupe(u8, value.string);
        if (!std.mem.startsWith(u8, ref, "#/")) {
            return fail(path, &.{ "unsupported $ref ", ref, ", only references into the same document are supported" });
        }
        if (!self.refs.contains(ref)) {
            // reserve the key first, so that a cycle back to this $ref stops here
            try self.refs.put(self.a, ref, null);
            const target = try self.resolveRef(ref, path);
            const built = try self.buildNode(target, ref);
            self.refs.getPtr(ref).?.* = built;
        }
        const node = try self.newNode(.{ .ref = .{ .ref = ref } });
        try self.pending.append(self.a, node);
        return node;
    }

    /// Ports `build_alternatives` (json-schema.cpp:118 @620d2225c), for
    /// both `any_of` and `all_of`.
    fn buildAlternatives(self: *Builder, comptime k: Kind, alts: *const Value, path: []const u8) Error!*Node {
        if (!alts.isArray()) return fail(path, &.{"must be an array of schemas"});
        if (alts.empty()) return fail(path, &.{"must not be empty"});
        const children = try self.a.alloc(*Node, alts.array.items.len);
        for (alts.array.items, 0..) |*alt, i| {
            children[i] = try self.buildNode(alt, try self.index(path, "/", i));
        }
        return self.newNode(@unionInit(Node, @tagName(k), children));
    }

    /// Ports `build_object` (json-schema.cpp:133 @620d2225c).
    fn buildObject(self: *Builder, schema: *const Value, path: []const u8) Error!*Node {
        var obj: Object = .{};

        var required: std.ArrayList([]const u8) = .empty;
        if (schema.contains("required")) {
            const req = schema.at("required") catch |e| return fromJson(e);
            if (req.isArray()) {
                for (req.array.items) |*name| {
                    if (name.isString()) try required.append(self.a, name.string);
                }
            }
        }

        if (schema.contains("properties")) {
            const properties = schema.at("properties") catch |e| return fromJson(e);
            if (!properties.isObject()) return fail(path, &.{"properties must be an object"});
            const members = properties.object.members.items;
            const props = try self.a.alloc(Property, members.len);
            for (members, 0..) |*m, i| {
                const name = try self.a.dupe(u8, m.key);
                const node = try self.buildNode(&m.value, try self.cat(&.{ path, "/properties/", name }));
                var is_required = false;
                for (required.items) |r| {
                    if (std.mem.eql(u8, r, name)) is_required = true;
                }
                props[i] = .{ .name = name, .schema = node, .required = is_required };
            }
            obj.properties = props;
        }

        if (schema.contains("additionalProperties")) {
            const additional = schema.at("additionalProperties") catch |e| return fromJson(e);
            if (additional.isBoolean()) {
                if (additional.boolean) obj.additional_properties = try self.newNode(.any);
            } else if (additional.isObject()) {
                obj.additional_properties = try self.buildNode(additional, try self.cat(&.{ path, "/additionalProperties" }));
            } else {
                return fail(path, &.{"additionalProperties must be a boolean or a schema"});
            }
        } else if (!schema.contains("properties")) {
            // {"type": "object"} on its own accepts any object
            obj.additional_properties = try self.newNode(.any);
        }

        return self.newNode(.{ .object = obj });
    }

    /// Ports `build_array` (json-schema.cpp:174 @620d2225c).
    fn buildArray(self: *Builder, schema: *const Value, path: []const u8) Error!*Node {
        var arr: Array = undefined;
        if (schema.contains("items") or schema.contains("prefixItems")) {
            // "items" wins when both are present; a schema instead of an
            // array is the item schema
            const key: []const u8 = if (schema.contains("items")) "items" else "prefixItems";
            const items = schema.at(key) catch |e| return fromJson(e);
            if (items.isArray()) {
                const tuple = try self.a.alloc(*Node, items.array.items.len);
                for (items.array.items, 0..) |*item, i| {
                    tuple[i] = try self.buildNode(item, try self.index(try self.cat(&.{ path, "/", key }), "/", i));
                }
                return self.newNode(.{ .tuple = tuple });
            }
            arr = .{ .items = try self.buildNode(items, try self.cat(&.{ path, "/", key })) };
        } else {
            arr = .{ .items = try self.newNode(.any) };
        }
        arr.min_items = try getCount(schema, "minItems", path, 0);
        arr.max_items = try getCount(schema, "maxItems", path, -1);
        return self.newNode(.{ .array = arr });
    }

    /// Ports `build_string` (json-schema.cpp:197 @620d2225c).
    fn buildString(self: *Builder, schema: *const Value, path: []const u8) Error!*Node {
        var s: String = .{};
        if (schema.contains("pattern")) {
            const pattern = schema.at("pattern") catch |e| return fromJson(e);
            if (!pattern.isString()) return fail(path, &.{"pattern must be a string"});
            s.pattern = try self.a.dupe(u8, pattern.string);
        }
        s.format = try getFormat(schema, path);
        s.min_length = try getCount(schema, "minLength", path, 0);
        s.max_length = try getCount(schema, "maxLength", path, -1);
        return self.newNode(.{ .string = s });
    }

    /// Ports `build_integer` (json-schema.cpp:212 @620d2225c). The `+ 1`
    /// and `- 1` on a saturated bound overflow in the C++, which wraps.
    fn buildInteger(self: *Builder, schema: *const Value, path: []const u8) Error!*Node {
        var n: Integer = .{};
        if (schema.contains("minimum")) {
            n.minimum = try getBound(schema, "minimum", path, true);
        } else if (schema.contains("exclusiveMinimum")) {
            n.minimum = (try getBound(schema, "exclusiveMinimum", path, false)) +% 1;
        }
        if (schema.contains("maximum")) {
            n.maximum = try getBound(schema, "maximum", path, false);
        } else if (schema.contains("exclusiveMaximum")) {
            n.maximum = (try getBound(schema, "exclusiveMaximum", path, true)) -% 1;
        }
        return self.newNode(.{ .integer = n });
    }

    /// Ports `build_node` (json-schema.cpp:227 @620d2225c).
    fn buildNode(self: *Builder, schema: *const Value, path: []const u8) Error!*Node {
        if (!schema.isObject()) return fail(path, &.{"schema must be an object"});
        if (schema.contains("$ref")) {
            return self.buildRef(schema.at("$ref") catch |e| return fromJson(e), path);
        }
        if (schema.contains("oneOf") or schema.contains("anyOf")) {
            const key: []const u8 = if (schema.contains("oneOf")) "oneOf" else "anyOf";
            return self.buildAlternatives(.any_of, schema.at(key) catch |e| return fromJson(e), try self.cat(&.{ path, "/", key }));
        }

        const null_value: Value = .null;
        const type_v: *const Value = if (schema.contains("type"))
            schema.at("type") catch |e| return fromJson(e)
        else
            &null_value;
        if (type_v.isArray()) {
            // {"type": ["a", "b"], ...} is {"anyOf": [{"type": "a", ...}, {"type": "b", ...}]}
            if (type_v.empty()) return fail(path, &.{"type must not be empty"});
            const children = try self.a.alloc(*Node, type_v.array.items.len);
            for (type_v.array.items, 0..) |*t, i| {
                var alt = schema.clone(self.a) catch |e| return fromJson(e);
                alt.set(self.a, "type", t.clone(self.a) catch |e| return fromJson(e)) catch |e| return fromJson(e);
                children[i] = try self.buildNode(&alt, try self.index(path, "/type/", i));
            }
            return self.newNode(.{ .any_of = children });
        }
        if (schema.contains("const")) {
            const v = schema.at("const") catch |e| return fromJson(e);
            return self.newNode(.{ .@"const" = v.clone(self.a) catch |e| return fromJson(e) });
        }
        if (schema.contains("enum")) {
            const values = schema.at("enum") catch |e| return fromJson(e);
            if (!values.isArray() or values.empty()) return fail(path, &.{"enum must be a non-empty array"});
            const vs = try self.a.alloc(Value, values.array.items.len);
            for (values.array.items, 0..) |*v, i| vs[i] = v.clone(self.a) catch |e| return fromJson(e);
            return self.newNode(.{ .@"enum" = vs });
        }
        if (!type_v.isNull() and !type_v.isString()) {
            return fail(path, &.{"type must be a string or an array of strings"});
        }

        const type_name: []const u8 = if (type_v.isString()) type_v.string else "";
        const true_value: Value = .{ .boolean = true };
        const has_properties = schema.contains("properties") or
            (schema.contains("additionalProperties") and
                !(schema.at("additionalProperties") catch |e| return fromJson(e)).eql(&true_value));

        if (type_name.len == 0) {
            // without a type the structural keywords decide, in the same
            // order as the converter
            if (has_properties) return self.buildObject(schema, path);
            if (schema.contains("allOf")) {
                return self.buildAlternatives(.all_of, schema.at("allOf") catch |e| return fromJson(e), try self.cat(&.{ path, "/allOf" }));
            }
            if (schema.contains("items") or schema.contains("prefixItems")) return self.buildArray(schema, path);
            if (schema.contains("pattern") or schema.contains("minLength") or schema.contains("maxLength") or
                try getFormat(schema, path) != .none)
            {
                return self.buildString(schema, path);
            }
            return self.newNode(.any);
        }
        if (std.mem.eql(u8, type_name, "object")) {
            if (!has_properties and schema.contains("allOf")) {
                return self.buildAlternatives(.all_of, schema.at("allOf") catch |e| return fromJson(e), try self.cat(&.{ path, "/allOf" }));
            }
            return self.buildObject(schema, path);
        }
        if (std.mem.eql(u8, type_name, "string")) {
            if (schema.contains("allOf")) {
                return self.buildAlternatives(.all_of, schema.at("allOf") catch |e| return fromJson(e), try self.cat(&.{ path, "/allOf" }));
            }
            return self.buildString(schema, path);
        }
        if (std.mem.eql(u8, type_name, "array")) return self.buildArray(schema, path);
        if (std.mem.eql(u8, type_name, "integer")) return self.buildInteger(schema, path);
        if (std.mem.eql(u8, type_name, "number")) return self.newNode(.number);
        if (std.mem.eql(u8, type_name, "boolean")) return self.newNode(.boolean);
        if (std.mem.eql(u8, type_name, "null")) return self.newNode(.null);
        return fail(path, &.{ "unrecognized type ", type_name });
    }
};

/// `std::stoull(s)` in base 10, as `resolve_ref` uses it: leading white
/// space, an optional sign (a minus negates modulo 2^64), at least one
/// digit, and whatever follows ignored.
///
/// Return: the value, or null where `stoull` throws (`invalid_argument`
/// for no digits, `out_of_range` past 2^64 - 1) -- `resolve_ref` catches
/// both as `std::logic_error`.
pub fn stoull(s: []const u8) ?u64 {
    var k: usize = 0;
    while (k < s.len and isCSpace(s[k])) k += 1;
    var negative = false;
    if (k < s.len and (s[k] == '+' or s[k] == '-')) {
        negative = s[k] == '-';
        k += 1;
    }
    const start = k;
    var r: u64 = 0;
    var overflow = false;
    while (k < s.len and s[k] >= '0' and s[k] <= '9') : (k += 1) {
        const m = @mulWithOverflow(r, 10);
        const ad = @addWithOverflow(m[0], s[k] - '0');
        if (m[1] != 0 or ad[1] != 0) overflow = true;
        r = ad[0];
    }
    if (k == start or overflow) return null;
    return if (negative) 0 -% r else r;
}

/// `isspace` in the `"C"` locale, which `strtoull` and `strtol` skip.
pub fn isCSpace(ch: u8) bool {
    return ch == ' ' or (ch >= '\t' and ch <= '\r');
}

/// Ports `json_type` (json-schema.cpp:346 @620d2225c).
fn jsonType(v: *const Value) ValueType {
    if (v.isNull()) return .null;
    if (v.isBoolean()) return .boolean;
    if (v.isNumberInteger()) return .integer;
    if (v.isNumber()) return .number;
    if (v.isString()) return .string;
    if (v.isArray()) return .array;
    return .object;
}

/// The `visited` set of `value_types_impl` and `may_be_string_impl`. A node
/// is inserted before the recursion into it and erased after, so the set is
/// always the chain of targets on the stack, kept here as a list on it.
const Visited = struct {
    node: *const Node,
    up: ?*const Visited,

    fn has(self: ?*const Visited, n: *const Node) bool {
        var v = self;
        while (v) |x| : (v = x.up) {
            if (x.node == n) return true;
        }
        return false;
    }
};

/// Ports `value_types_impl` (json-schema.cpp:368 @620d2225c).
fn valueTypesImpl(s: *const Node, visited: ?*const Visited) TypeSet {
    switch (s.*) {
        .any => return TypeSet.all(),
        .null => return TypeSet.of(&.{.null}),
        .boolean => return TypeSet.of(&.{.boolean}),
        .number => return TypeSet.of(&.{ .number, .integer }),
        .integer => return TypeSet.of(&.{.integer}),
        .string => return TypeSet.of(&.{.string}),
        .array, .tuple => return TypeSet.of(&.{.array}),
        .object => return TypeSet.of(&.{.object}),
        .@"const" => |*v| return TypeSet.of(&.{jsonType(v)}),
        .@"enum" => |vs| {
            var types: TypeSet = .{};
            for (vs) |*v| types.add(jsonType(v));
            return types;
        },
        .ref => |r| {
            // a cycle contributes no type, to be safe
            const target = r.target orelse return .{};
            if (Visited.has(visited, target)) return .{};
            const here: Visited = .{ .node = target, .up = visited };
            return valueTypesImpl(target, &here);
        },
        .any_of => |children| {
            var types: TypeSet = .{};
            for (children) |child| types.mask |= valueTypesImpl(child, visited).mask;
            return types;
        },
        .all_of => |children| {
            var types = TypeSet.all();
            for (children) |child| types.mask &= valueTypesImpl(child, visited).mask;
            return types;
        },
    }
}

/// Ports `may_be_string_impl` (json-schema.cpp:429 @620d2225c).
fn mayBeStringImpl(s: *const Node, visited: ?*const Visited) bool {
    switch (s.*) {
        .string => return true,
        .@"const" => |*v| return v.isString(),
        .@"enum" => |vs| {
            for (vs) |*v| {
                if (v.isString()) return true;
            }
            return false;
        },
        .ref => |r| {
            // a cycle is taken as not a string, to be safe
            const target = r.target orelse return false;
            if (Visited.has(visited, target)) return false;
            const here: Visited = .{ .node = target, .up = visited };
            return mayBeStringImpl(target, &here);
        },
        .any_of => |children| {
            for (children) |child| {
                if (mayBeStringImpl(child, visited)) return true;
            }
            return false;
        },
        .all_of => |children| {
            // every child must allow a string, an any child constrains nothing
            var any_string = false;
            for (children) |child| {
                if (child.* == .any) continue;
                if (!mayBeStringImpl(child, visited)) return false;
                any_string = true;
            }
            return any_string;
        },
        else => return false,
    }
}

/// Ports `kind_name` (json-schema.cpp:483 @620d2225c).
pub fn kindName(k: Kind) []const u8 {
    return switch (k) {
        .any => "any",
        .ref => "ref",
        .any_of => "anyOf",
        .all_of => "allOf",
        .@"const" => "const",
        .@"enum" => "enum",
        .null => "null",
        .boolean => "boolean",
        .number => "number",
        .integer => "integer",
        .string => "string",
        .array => "array",
        .tuple => "tuple",
        .object => "object",
    };
}

/// Ports `type_name` (json-schema.cpp:503 @620d2225c), a member of
/// `common_chat_schema`.
pub fn typeName(t: ValueType) []const u8 {
    return switch (t) {
        .null => "null",
        .boolean => "boolean",
        .number => "number",
        .integer => "integer",
        .string => "string",
        .array => "array",
        .object => "object",
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a recursive $ref resolves to its own target" {
    const a = std.testing.allocator;
    var v = try json.parse(a,
        \\{"$ref": "#/$defs/n", "$defs": {"n": {"type": "object",
        \\ "properties": {"next": {"$ref": "#/$defs/n"}, "s": {"type": ["string", "null"]}}}}}
    );
    defer v.deinit(a);
    var doc = try fromJsonSchema(a, &v);
    defer doc.deinit();
    try std.testing.expectEqual(Kind.ref, doc.root.kind());
    const target = doc.root.ref.target.?;
    try std.testing.expectEqual(Kind.object, target.kind());
    try std.testing.expectEqual(target, target.object.properties[0].schema.ref.target.?);
    try std.testing.expect(doc.root.valueTypes().isOnly(.object));
    try std.testing.expect(target.object.properties[1].schema.mayBeString());
    try std.testing.expectEqual(@as(usize, 1), doc.refs.len);
}

test "an unsupported schema names its path" {
    const a = std.testing.allocator;
    var v = try json.parse(a, "{\"properties\": {\"x\": {\"type\": \"date\"}}}");
    defer v.deinit(a);
    try std.testing.expectError(error.RuntimeError, fromJsonSchema(a, &v));
    try std.testing.expectEqualStrings("JSON schema error at #/properties/x: unrecognized type date", what());
}
