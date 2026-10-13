//! GGUF metadata read the way `llama_model_loader` reads it.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-model-loader.cpp` -- `GGUFMeta::GKV`, the typed
//!   getter with its override handling, `llama_model_loader::get_key`, and
//!   the override map the constructor builds
//!
//! All at v0.3.0 (`c1d0e7a00`).
//!
//! # Scope
//!
//! The getter for the types `llama-vocab.cpp` reads. Key names come from
//! `arch.zig`, which carries every `llm_kv`.
//!
//! # Strings stop at the first NUL
//!
//! `gguf_get_val_str` hands the C++ a `const char *`, which it wraps in a
//! `std::string`, so a value is read up to its first NUL byte whatever its
//! stored length. The same holds for `gguf_get_arr_str`. `cString` does
//! that here.

const std = @import("std");
const impl = @import("impl.zig");
const arch = @import("arch.zig");

const c = impl.c;
const allocator = impl.allocator;

/// The metadata keys, generated from `llm_kv`; see `arch.zig`.
pub const Kv = arch.Kv;

/// A string value as the C++ sees it through a `const char *`: up to the
/// first NUL.
pub fn cString(p: [*c]const u8) []const u8 {
    return std.mem.span(@as([*:0]const u8, @ptrCast(p)));
}

/// The metadata of one GGUF file, with the caller's overrides.
pub const Meta = struct {
    ctx: *const c.gguf_context,
    /// The architecture whose name `%s` keys take.
    arch: arch.Arch,
    /// `kv_overrides`, keyed by the override's key.
    overrides: std.StringHashMapUnmanaged(c.llama_model_kv_override) = .empty,

    /// Ports the override half of `llama_model_loader`'s constructor
    /// (llama-model-loader.cpp:527 @c1d0e7a00): every override up to the
    /// one with an empty key, the first of any repeated key kept, as
    /// `unordered_map`'s `insert` keeps it.
    ///
    /// Parameters:
    /// - `ctx`: the metadata; borrowed for the `Meta`'s lifetime.
    /// - `which`: the architecture keys are named for.
    /// - `overrides`: the caller's list, or null.
    ///
    /// Return: the reader; free it with `deinit`.
    pub fn init(ctx: *const c.gguf_context, which: arch.Arch, overrides: ?[*]const c.llama_model_kv_override) Meta {
        var m: Meta = .{ .ctx = ctx, .arch = which };
        if (overrides) |list| {
            var i: usize = 0;
            while (list[i].key[0] != 0) : (i += 1) {
                const key = cString(&list[i].key);
                const r = m.overrides.getOrPut(allocator, key) catch impl.abort("gguf_meta", 0, "out of memory");
                if (!r.found_existing) r.value_ptr.* = list[i];
            }
        }
        return m;
    }

    /// Return: nothing.
    pub fn deinit(self: *Meta) void {
        self.overrides.deinit(allocator);
    }

    /// The key's name, as `LLM_KV(arch)(kv)` formats it; see `arch.kvName`.
    ///
    /// Return: the name.
    pub fn name(self: *const Meta, kv: Kv) arch.Name {
        return arch.kvName(self.arch, kv, null);
    }

    /// `gguf_find_key` on a key's name.
    ///
    /// Return: the key's index, or -1.
    pub fn find(self: *const Meta, kv: Kv) i64 {
        const n = self.name(kv);
        return self.findNamed(n.slice());
    }

    /// `gguf_find_key` on a name.
    ///
    /// Return: the key's index, or -1.
    pub fn findNamed(self: *const Meta, key: []const u8) i64 {
        var zbuf: [arch.max_name + 1]u8 = undefined;
        if (key.len > arch.max_name) return -1;
        @memcpy(zbuf[0..key.len], key);
        zbuf[key.len] = 0;
        return c.gguf_find_key(self.ctx, &zbuf);
    }

    /// Ports `llama_model_loader`'s `get_key` (llama-model-loader.cpp:415
    /// @c1d0e7a00) through `GGUFMeta::GKV<T>`'s `set` (llama-model-loader.cpp:255
    /// @c1d0e7a00): an override of the right type wins, logged; otherwise the
    /// stored value, which must be exactly the type `T` maps to.
    ///
    /// Parameters:
    /// - `T`: `bool`, an integer of 8 to 64 bits, `c_int`, `f32`, `f64`, or
    ///   `[]const u8` (`std::string`).
    /// - `kv`: the key.
    /// - `required`: whether a missing key throws.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the value, null when absent and not required, or
    /// `error.Thrown` -- a wrong stored type or a missing required key. A
    /// string borrows from the metadata or the override.
    pub fn getKey(self: *const Meta, comptime T: type, kv: Kv, required: bool, exc: *impl.Exception) impl.Thrown!?T {
        const n = self.name(kv);
        return self.getKeyNamed(T, n.slice(), required, exc);
    }

    /// `getKey` on a name rather than an `llm_kv`, as the `std::string`
    /// overload of `get_key` takes one.
    pub fn getKeyNamed(self: *const Meta, comptime T: type, key: []const u8, required: bool, exc: *impl.Exception) impl.Thrown!?T {
        const ovrd = self.overrides.getPtr(key);
        if (tryOverride(T, ovrd)) |v| return v;

        const k = self.findNamed(key);
        if (k < 0) {
            if (required) return exc.throw(.runtime_error, "key not found in model: {s}", .{key});
            return null;
        }
        return try getKv(self.ctx, T, k, exc);
    }

    /// Ports the `llama_pooling_type` specialization of `get_key`
    /// (llama-model-loader.cpp:441 @c1d0e7a00): a `uint32_t` read, and
    /// `LLAMA_POOLING_TYPE_UNSPECIFIED` when the key is absent.
    ///
    /// Return: the pooling type, or `error.Thrown`.
    pub fn getPoolingType(self: *const Meta, kv: Kv, required: bool, exc: *impl.Exception) impl.Thrown!c_int {
        const v = (try self.getKey(u32, kv, required, exc)) orelse return c.LLAMA_POOLING_TYPE_UNSPECIFIED;
        return @bitCast(v);
    }

    /// Ports `llama_model_loader`'s `get_arr_n` (llama-model-loader.cpp:276
    /// @c1d0e7a00): the length of an array-typed key.
    ///
    /// Return: the length, null when absent and not required, or
    /// `error.Thrown` -- a missing required key or a key that is not an array.
    pub fn getArrN(self: *const Meta, kv: Kv, required: bool, exc: *impl.Exception) impl.Thrown!?u32 {
        const n = self.name(kv);
        const key = n.slice();
        const k = self.findNamed(key);
        if (k < 0) {
            if (required) return exc.throw(.runtime_error, "key not found in model: {s}", .{key});
            return null;
        }
        const info = try arrayInfo(self.ctx, k, exc);
        return @truncate(info.length);
    }

    /// Ports the `std::array` overload of `llama_model_loader`'s `get_arr`
    /// (llama-model-loader.cpp:350 @c1d0e7a00): an array key into `result`,
    /// whose length is `N_MAX`. A boolean array converts to 0 and 1.
    ///
    /// Parameters:
    /// - `T`: `i32`, `u32`, `c_int` or `f32`.
    /// - `key`: the key's name.
    /// - `result`: receives the elements; the rest is left alone.
    /// - `required`: whether a missing key throws.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: whether the key was read, or `error.Thrown`.
    pub fn getArrInto(self: *const Meta, comptime T: type, key: []const u8, result: []T, required: bool, exc: *impl.Exception) impl.Thrown!bool {
        const k = self.findNamed(key);
        if (k < 0 or c.gguf_get_kv_type(self.ctx, k) != c.GGUF_TYPE_ARRAY) {
            if (required) return exc.throw(.runtime_error, "array key not found in model: {s}", .{key});
            return false;
        }
        const info = try arrayInfo(self.ctx, k, exc);
        const type_ok = switch (info.gt) {
            c.GGUF_TYPE_BOOL, c.GGUF_TYPE_UINT32, c.GGUF_TYPE_INT32 => isInt32(T),
            c.GGUF_TYPE_FLOAT32 => T == f32,
            c.GGUF_TYPE_STRING => false,
            else => return exc.throw(.runtime_error, "{s} is not a string/float32/uint32/int32 array", .{key}),
        };
        if (!type_ok) return exc.throw(.runtime_error, "{s} has wrong array element type {s}", .{ key, cString(c.gguf_type_name(info.gt)) });
        if (info.length > result.len) {
            return exc.throw(.runtime_error, "array length {d} for key {s} exceeds max {d}", .{
                @as(u32, @truncate(info.length)), key, @as(u32, @truncate(result.len)),
            });
        }
        if (info.gt == c.GGUF_TYPE_BOOL) {
            const values: [*]const i8 = @ptrCast(info.data.?);
            for (result[0..info.length], values[0..info.length]) |*r, x| r.* = @intFromBool(x != 0);
        } else {
            const values: [*]align(1) const T = @ptrCast(info.data.?);
            for (result[0..info.length], values[0..info.length]) |*r, x| r.* = x;
        }
        return true;
    }

    /// `getArrInto` on an `llm_kv`.
    pub fn getArr(self: *const Meta, comptime T: type, kv: Kv, result: []T, required: bool, exc: *impl.Exception) impl.Thrown!bool {
        const n = self.name(kv);
        return self.getArrInto(T, n.slice(), result, required, exc);
    }

    /// Ports the `std::vector` overload of `llama_model_loader`'s `get_arr`
    /// (llama-model-loader.cpp:305 @c1d0e7a00): an array key as a new list.
    /// Unlike the `std::array` overload, a boolean array is refused.
    ///
    /// Parameters:
    /// - `T`: `i32`, `u32`, `f32`, or `[]const u8` for a string array,
    ///   whose elements borrow from the metadata.
    /// - `kv`: the key.
    /// - `required`: whether a missing key throws.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the elements, owned by the caller and freed with `impl.allocator`;
    /// null when absent and not required; or `error.Thrown`.
    pub fn getArrList(self: *const Meta, comptime T: type, kv: Kv, required: bool, exc: *impl.Exception) impl.Thrown!?[]T {
        const n = self.name(kv);
        const key = n.slice();
        const k = self.findNamed(key);
        if (k < 0 or c.gguf_get_kv_type(self.ctx, k) != c.GGUF_TYPE_ARRAY) {
            if (required) return exc.throw(.runtime_error, "array key not found in model: {s}", .{key});
            return null;
        }
        const info = try arrayInfo(self.ctx, k, exc);
        const type_ok = switch (info.gt) {
            c.GGUF_TYPE_UINT32, c.GGUF_TYPE_INT32 => T != []const u8 and isInt32(T),
            c.GGUF_TYPE_FLOAT32 => T == f32,
            c.GGUF_TYPE_STRING => T == []const u8,
            else => return exc.throw(.runtime_error, "{s} is not a string/float32/uint32/int32 array", .{key}),
        };
        if (!type_ok) return exc.throw(.runtime_error, "{s} has wrong array element type {s}", .{ key, cString(c.gguf_type_name(info.gt)) });
        const out = allocator.alloc(T, info.length) catch impl.abort("gguf_meta", 0, "out of memory");
        if (T == []const u8) {
            for (out, 0..) |*r, i| r.* = cString(c.gguf_get_arr_str(self.ctx, k, i));
        } else {
            const values: [*]align(1) const T = @ptrCast(info.data.?);
            for (out, values[0..info.length]) |*r, x| r.* = x;
        }
        return out;
    }

    /// Ports the `std::array` form of `llama_model_loader`'s `get_key_or_arr`
    /// (llama-model-loader.cpp:454 @c1d0e7a00): an array of exactly `n`
    /// elements, or one value repeated `n` times.
    ///
    /// Parameters:
    /// - `T`: `u32`, `c_int` or `f32`.
    /// - `kv`: the key.
    /// - `result`: receives the elements; its length is `N_MAX`.
    /// - `n`: how many elements.
    /// - `required`: whether a missing key throws.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: whether the key was read, or `error.Thrown`.
    pub fn getKeyOrArr(self: *const Meta, comptime T: type, kv: Kv, result: []T, n: u32, required: bool, exc: *impl.Exception) impl.Thrown!bool {
        const nm = self.name(kv);
        const key = nm.slice();
        const k = self.findNamed(key);
        if (k < 0) {
            if (required) return exc.throw(.runtime_error, "key not found in model: {s}", .{key});
            return false;
        }
        if (n > result.len) {
            return exc.throw(.runtime_error, "n > N_MAX: {d} > {d} for key {s}", .{ n, @as(u32, @truncate(result.len)), key });
        }
        if (c.gguf_get_kv_type(self.ctx, k) == c.GGUF_TYPE_ARRAY) {
            const info = try arrayInfo(self.ctx, k, exc);
            if (n != info.length) {
                return exc.throw(.runtime_error, "key {s} has wrong array length; expected {d}, got {d}", .{
                    key, n, @as(u32, @truncate(info.length)),
                });
            }
            return self.getArrInto(T, key, result, required, exc);
        }
        const value = (try self.getKeyNamed(T, key, required, exc)) orelse return false;
        for (result[0..n]) |*r| r.* = value;
        return true;
    }

    /// Ports the scalar form of `llama_model_loader`'s `get_key_or_arr`
    /// (llama-model-loader.cpp:498 @c1d0e7a00): a `uint32_t`, refusing an
    /// array -- with a throw only when the key is required.
    ///
    /// Return: the value, null when absent or an array and not required, or
    /// `error.Thrown`.
    pub fn getKeyOrArrScalar(self: *const Meta, kv: Kv, required: bool, exc: *impl.Exception) impl.Thrown!?u32 {
        const nm = self.name(kv);
        const key = nm.slice();
        const id = self.findNamed(key);
        if (id < 0) {
            if (required) return exc.throw(.runtime_error, "key not found in model: {s}", .{key});
            return null;
        }
        if (c.gguf_get_kv_type(self.ctx, id) == c.GGUF_TYPE_ARRAY) {
            if (required) return exc.throw(.runtime_error, "expected scalar, found array for key: {s}", .{key});
            return null;
        }
        return self.getKeyNamed(u32, key, required, exc);
    }
};

fn isInt32(comptime T: type) bool {
    return T == i32 or T == u32 or T == c_int;
}

/// Ports `GGUFMeta`'s `ArrayInfo` (llama-model-loader.cpp:140 @c1d0e7a00).
const ArrayInfo = struct {
    gt: c.enum_gguf_type,
    length: usize,
    data: ?*const anyopaque,
};

/// `GGUFMeta::GKV<ArrayInfo>::get_kv`: the array's element type, length and
/// data, or a throw when the key is not an array.
fn arrayInfo(ctx: *const c.gguf_context, k: i64, exc: *impl.Exception) impl.Thrown!ArrayInfo {
    try checkType(ctx, k, c.GGUF_TYPE_ARRAY, exc);
    const arr_type = c.gguf_get_arr_type(ctx, k);
    return .{
        .gt = arr_type,
        .length = c.gguf_get_arr_n(ctx, k),
        .data = if (arr_type == c.GGUF_TYPE_STRING) null else c.gguf_get_arr_data(ctx, k),
    };
}

/// The storage type of `T`: `c_int` is `int32_t`.
fn Stored(comptime T: type) type {
    return if (T == c_int) i32 else T;
}

fn ggufType(comptime T: type) c.enum_gguf_type {
    return switch (Stored(T)) {
        bool => c.GGUF_TYPE_BOOL,
        u8 => c.GGUF_TYPE_UINT8,
        u16 => c.GGUF_TYPE_UINT16,
        u32 => c.GGUF_TYPE_UINT32,
        u64 => c.GGUF_TYPE_UINT64,
        i8 => c.GGUF_TYPE_INT8,
        i16 => c.GGUF_TYPE_INT16,
        i32 => c.GGUF_TYPE_INT32,
        i64 => c.GGUF_TYPE_INT64,
        f32 => c.GGUF_TYPE_FLOAT32,
        f64 => c.GGUF_TYPE_FLOAT64,
        []const u8 => c.GGUF_TYPE_STRING,
        else => @compileError("unsupported metadata type"),
    };
}

fn checkType(ctx: *const c.gguf_context, k: i64, want: c.enum_gguf_type, exc: *impl.Exception) impl.Thrown!void {
    const kt = c.gguf_get_kv_type(ctx, k);
    if (kt != want) {
        return exc.throw(.runtime_error, "key {s} has wrong type {s} but expected type {s}", .{
            cString(c.gguf_get_key(ctx, k)), cString(c.gguf_type_name(kt)), cString(c.gguf_type_name(want)),
        });
    }
}

/// Ports `GGUFMeta::GKV<T>`'s `get_kv` (llama-model-loader.cpp:164 @c1d0e7a00).
fn getKv(ctx: *const c.gguf_context, comptime T: type, k: i64, exc: *impl.Exception) impl.Thrown!T {
    try checkType(ctx, k, ggufType(T), exc);
    return switch (Stored(T)) {
        bool => c.gguf_get_val_bool(ctx, k),
        u8 => c.gguf_get_val_u8(ctx, k),
        u16 => c.gguf_get_val_u16(ctx, k),
        u32 => c.gguf_get_val_u32(ctx, k),
        u64 => c.gguf_get_val_u64(ctx, k),
        i8 => c.gguf_get_val_i8(ctx, k),
        i16 => c.gguf_get_val_i16(ctx, k),
        i32 => c.gguf_get_val_i32(ctx, k),
        i64 => c.gguf_get_val_i64(ctx, k),
        f32 => c.gguf_get_val_f32(ctx, k),
        f64 => c.gguf_get_val_f64(ctx, k),
        []const u8 => cString(c.gguf_get_val_str(ctx, k)),
        else => unreachable,
    };
}

fn overrideTypeStr(ty: c.enum_llama_model_kv_override_type) [:0]const u8 {
    return switch (ty) {
        c.LLAMA_KV_OVERRIDE_TYPE_BOOL => "bool",
        c.LLAMA_KV_OVERRIDE_TYPE_INT => "int",
        c.LLAMA_KV_OVERRIDE_TYPE_FLOAT => "float",
        c.LLAMA_KV_OVERRIDE_TYPE_STR => "str",
        else => "unknown",
    };
}

/// Ports `GGUFMeta::GKV`'s `validate_override` (llama-model-loader.cpp:184
/// @c1d0e7a00): an override of the expected type is logged and taken; one
/// of another type is warned about and ignored.
fn validateOverride(expected: c.enum_llama_model_kv_override_type, ovrd: ?*const c.llama_model_kv_override) bool {
    const o = ovrd orelse return false;
    if (o.tag == expected) {
        impl.logInfo("%s: Using metadata override (%5s) '%s' = ", .{ "validate_override", overrideTypeStr(o.tag).ptr, &o.key });
        switch (o.tag) {
            c.LLAMA_KV_OVERRIDE_TYPE_BOOL => impl.logInfo("%s\n", .{@as([*:0]const u8, if (o.unnamed_0.val_bool) "true" else "false")}),
            c.LLAMA_KV_OVERRIDE_TYPE_INT => impl.logInfo("%lld\n", .{@as(c_longlong, o.unnamed_0.val_i64)}),
            c.LLAMA_KV_OVERRIDE_TYPE_FLOAT => impl.logInfo("%.6f\n", .{o.unnamed_0.val_f64}),
            c.LLAMA_KV_OVERRIDE_TYPE_STR => impl.logInfo("%s\n", .{&o.unnamed_0.val_str}),
            else => {},
        }
        return true;
    }
    impl.logWarn("%s: Warning: Bad metadata override type for key '%s', expected %s but got %s\n", .{
        "validate_override", &o.key, overrideTypeStr(expected).ptr, overrideTypeStr(o.tag).ptr,
    });
    return false;
}

/// Ports the four `GGUFMeta::GKV::try_override` overloads -- bool, integer,
/// floating point, string: `try_override`, `try_override`, `try_override`,
/// `try_override` (llama-model-loader.cpp:217, 227, 237, 247 @c1d0e7a00).
/// An integer override is the `int64_t` narrowed to `T` and a float one the
/// `double` rounded to `T`, as the C++'s assignments convert them.
fn tryOverride(comptime T: type, ovrd: ?*const c.llama_model_kv_override) ?T {
    switch (T) {
        bool => if (validateOverride(c.LLAMA_KV_OVERRIDE_TYPE_BOOL, ovrd)) return ovrd.?.unnamed_0.val_bool,
        []const u8 => if (validateOverride(c.LLAMA_KV_OVERRIDE_TYPE_STR, ovrd)) return cString(&ovrd.?.unnamed_0.val_str),
        f32, f64 => if (validateOverride(c.LLAMA_KV_OVERRIDE_TYPE_FLOAT, ovrd)) return @floatCast(ovrd.?.unnamed_0.val_f64),
        else => {
            const U = @Int(.unsigned, @bitSizeOf(T));
            if (validateOverride(c.LLAMA_KV_OVERRIDE_TYPE_INT, ovrd)) {
                return @bitCast(@as(U, @truncate(@as(u64, @bitCast(ovrd.?.unnamed_0.val_i64)))));
            }
        },
    }
    return null;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "an arch-prefixed key takes the architecture name" {
    const m: Meta = .{ .ctx = undefined, .arch = .qwen35 };
    try std.testing.expectEqualStrings("qwen35.vocab_size", m.name(.vocab_size).slice());
    try std.testing.expectEqualStrings("tokenizer.ggml.model", m.name(.tokenizer_model).slice());
}
