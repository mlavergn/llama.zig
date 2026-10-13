//! GGUF metadata read the way `llama_model_loader` reads it.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-model-loader.cpp` -- `GGUFMeta::GKV`, the typed
//!   getter with its override handling, `llama_model_loader::get_key`, and
//!   the override map the constructor builds
//! - `llama.cpp/src/llama-arch.cpp` -- the `LLM_KV_NAMES` entries for the
//!   keys read here, and `LLM_KV::operator()`
//!
//! All at v0.3.0 (`c1d0e7a00`).
//!
//! # Scope
//!
//! The keys `llama-vocab.cpp` reads, which is what is ported so far. The
//! model loader's port extends `Kv` with the rest of `LLM_KV_NAMES`; the
//! getter does not change.
//!
//! # Strings stop at the first NUL
//!
//! `gguf_get_val_str` hands the C++ a `const char *`, which it wraps in a
//! `std::string`, so a value is read up to its first NUL byte whatever its
//! stored length. The same holds for `gguf_get_arr_str`. `cString` does
//! that here.

const std = @import("std");
const impl = @import("impl.zig");

const c = impl.c;
const allocator = impl.allocator;

/// The `llm_kv` values read so far, with their `LLM_KV_NAMES` formats.
pub const Kv = enum {
    general_architecture,
    general_name,
    vocab_size,
    tokenizer_model,
    tokenizer_pre,
    tokenizer_list,
    tokenizer_token_type,
    tokenizer_token_type_count,
    tokenizer_scores,
    tokenizer_merges,
    tokenizer_bos_id,
    tokenizer_eos_id,
    tokenizer_eot_id,
    tokenizer_eom_id,
    tokenizer_unk_id,
    tokenizer_sep_id,
    tokenizer_pad_id,
    tokenizer_mask_id,
    tokenizer_add_bos,
    tokenizer_add_eos,
    tokenizer_add_sep,
    tokenizer_add_prefix,
    tokenizer_remove_extra_ws,
    tokenizer_precompiled_charsmap,
    tokenizer_normalizer_lowercase,
    tokenizer_normalizer_strip_accents,
    tokenizer_fim_pre_id,
    tokenizer_fim_suf_id,
    tokenizer_fim_mid_id,
    tokenizer_fim_pad_id,
    tokenizer_fim_rep_id,
    tokenizer_fim_sep_id,
    tokenizer_suppress_tokens,
    tokenizer_prefix_id,
    tokenizer_suffix_id,
    tokenizer_middle_id,

    /// Ports the `LLM_KV_NAMES` entries (llama-arch.cpp:159 @c1d0e7a00).
    fn format(self: Kv) []const u8 {
        return switch (self) {
            .general_architecture => "general.architecture",
            .general_name => "general.name",
            .vocab_size => "%s.vocab_size",
            .tokenizer_model => "tokenizer.ggml.model",
            .tokenizer_pre => "tokenizer.ggml.pre",
            .tokenizer_list => "tokenizer.ggml.tokens",
            .tokenizer_token_type => "tokenizer.ggml.token_type",
            .tokenizer_token_type_count => "tokenizer.ggml.token_type_count",
            .tokenizer_scores => "tokenizer.ggml.scores",
            .tokenizer_merges => "tokenizer.ggml.merges",
            .tokenizer_bos_id => "tokenizer.ggml.bos_token_id",
            .tokenizer_eos_id => "tokenizer.ggml.eos_token_id",
            .tokenizer_eot_id => "tokenizer.ggml.eot_token_id",
            .tokenizer_eom_id => "tokenizer.ggml.eom_token_id",
            .tokenizer_unk_id => "tokenizer.ggml.unknown_token_id",
            .tokenizer_sep_id => "tokenizer.ggml.seperator_token_id",
            .tokenizer_pad_id => "tokenizer.ggml.padding_token_id",
            .tokenizer_mask_id => "tokenizer.ggml.mask_token_id",
            .tokenizer_add_bos => "tokenizer.ggml.add_bos_token",
            .tokenizer_add_eos => "tokenizer.ggml.add_eos_token",
            .tokenizer_add_sep => "tokenizer.ggml.add_sep_token",
            .tokenizer_add_prefix => "tokenizer.ggml.add_space_prefix",
            .tokenizer_remove_extra_ws => "tokenizer.ggml.remove_extra_whitespaces",
            .tokenizer_precompiled_charsmap => "tokenizer.ggml.precompiled_charsmap",
            .tokenizer_normalizer_lowercase => "tokenizer.ggml.normalizer.lowercase",
            .tokenizer_normalizer_strip_accents => "tokenizer.ggml.normalizer.strip_accents",
            .tokenizer_fim_pre_id => "tokenizer.ggml.fim_pre_token_id",
            .tokenizer_fim_suf_id => "tokenizer.ggml.fim_suf_token_id",
            .tokenizer_fim_mid_id => "tokenizer.ggml.fim_mid_token_id",
            .tokenizer_fim_pad_id => "tokenizer.ggml.fim_pad_token_id",
            .tokenizer_fim_rep_id => "tokenizer.ggml.fim_rep_token_id",
            .tokenizer_fim_sep_id => "tokenizer.ggml.fim_sep_token_id",
            .tokenizer_suppress_tokens => "tokenizer.ggml.suppress_tokens",
            .tokenizer_prefix_id => "tokenizer.ggml.prefix_token_id",
            .tokenizer_suffix_id => "tokenizer.ggml.suffix_token_id",
            .tokenizer_middle_id => "tokenizer.ggml.middle_token_id",
        };
    }
};

/// A string value as the C++ sees it through a `const char *`: up to the
/// first NUL.
pub fn cString(p: [*c]const u8) []const u8 {
    return std.mem.span(@as([*:0]const u8, @ptrCast(p)));
}

/// The metadata of one GGUF file, with the caller's overrides.
pub const Meta = struct {
    ctx: *const c.gguf_context,
    /// `LLM_ARCH_NAMES.at(arch)`: the architecture name `%s` keys take.
    arch_name: []const u8,
    /// `kv_overrides`, keyed by the override's key.
    overrides: std.StringHashMapUnmanaged(c.llama_model_kv_override) = .empty,

    /// Ports the override half of `llama_model_loader`'s constructor
    /// (llama-model-loader.cpp:527 @c1d0e7a00): every override up to the
    /// one with an empty key, the first of any repeated key kept, as
    /// `unordered_map`'s `insert` keeps it.
    ///
    /// Parameters:
    /// - `ctx`: the metadata; borrowed for the `Meta`'s lifetime.
    /// - `arch_name`: the architecture's name; borrowed likewise.
    /// - `overrides`: the caller's list, or null.
    ///
    /// Return: the reader; free it with `deinit`.
    pub fn init(ctx: *const c.gguf_context, arch_name: []const u8, overrides: ?[*]const c.llama_model_kv_override) Meta {
        var m: Meta = .{ .ctx = ctx, .arch_name = arch_name };
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

    /// Ports the call operator of `LLM_KV` (llama-arch.cpp:923 @c1d0e7a00):
    /// the format with the architecture name substituted, with no suffix.
    ///
    /// Parameters:
    /// - `kv`: the key.
    /// - `buf`: receives the name.
    ///
    /// Return: the name, borrowed from `buf`.
    pub fn name(self: *const Meta, kv: Kv, buf: []u8) []const u8 {
        const f = kv.format();
        if (std.mem.find(u8, f, "%s")) |at| {
            return std.mem.print(buf, "{s}{s}{s}", .{ f[0..at], self.arch_name, f[at + 2 ..] }) catch buf[0..0];
        }
        return f;
    }

    /// `gguf_find_key` on a key's name.
    ///
    /// Return: the key's index, or -1.
    pub fn find(self: *const Meta, kv: Kv) i64 {
        var buf: [256]u8 = undefined;
        var zbuf: [257]u8 = undefined;
        const n = self.name(kv, &buf);
        @memcpy(zbuf[0..n.len], n);
        zbuf[n.len] = 0;
        return c.gguf_find_key(self.ctx, &zbuf);
    }

    /// Ports `llama_model_loader`'s `get_key` (llama-model-loader.cpp:415
    /// @c1d0e7a00) through `GGUFMeta::GKV<T>`'s `set` (llama-model-loader.cpp:255
    /// @c1d0e7a00): an override of the right type wins, logged; otherwise the
    /// stored value, which must be exactly the type `T` maps to.
    ///
    /// Parameters:
    /// - `T`: `bool`, `u32` or `[]const u8` (`std`'s `string`).
    /// - `kv`: the key.
    /// - `required`: whether a missing key throws.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the value, null when absent and not required, or
    /// `error.Thrown` -- a wrong stored type or a missing required key. A
    /// string borrows from the metadata or the override.
    pub fn getKey(self: *const Meta, comptime T: type, kv: Kv, required: bool, exc: *impl.Exception) impl.Thrown!?T {
        var buf: [256]u8 = undefined;
        const key = self.name(kv, &buf);
        const ovrd = self.overrides.getPtr(key);
        if (tryOverride(T, ovrd)) |v| return v;

        const k = self.find(kv);
        if (k < 0) {
            if (required) return exc.throw(.runtime_error, "key not found in model: {s}", .{key});
            return null;
        }
        return try getKv(self.ctx, T, k, exc);
    }
};

fn ggufType(comptime T: type) c.enum_gguf_type {
    return switch (T) {
        bool => c.GGUF_TYPE_BOOL,
        u32 => c.GGUF_TYPE_UINT32,
        []const u8 => c.GGUF_TYPE_STRING,
        else => @compileError("unsupported metadata type"),
    };
}

/// Ports `GGUFMeta::GKV<T>`'s `get_kv` (llama-model-loader.cpp:164 @c1d0e7a00).
fn getKv(ctx: *const c.gguf_context, comptime T: type, k: i64, exc: *impl.Exception) impl.Thrown!T {
    const kt = c.gguf_get_kv_type(ctx, k);
    if (kt != ggufType(T)) {
        return exc.throw(.runtime_error, "key {s} has wrong type {s} but expected type {s}", .{
            cString(c.gguf_get_key(ctx, k)), cString(c.gguf_type_name(kt)), cString(c.gguf_type_name(ggufType(T))),
        });
    }
    return switch (T) {
        bool => c.gguf_get_val_bool(ctx, k),
        u32 => c.gguf_get_val_u32(ctx, k),
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

/// Ports the `GGUFMeta::GKV::try_override` overloads for the types read
/// here, one each for bool, integers and strings: `try_override`,
/// `try_override`, `try_override` (llama-model-loader.cpp:217, 227, 247
/// @c1d0e7a00). An integer override is the `int64_t` narrowed to `T`, as
/// the C++'s assignment narrows it.
fn tryOverride(comptime T: type, ovrd: ?*const c.llama_model_kv_override) ?T {
    switch (T) {
        bool => if (validateOverride(c.LLAMA_KV_OVERRIDE_TYPE_BOOL, ovrd)) return ovrd.?.unnamed_0.val_bool,
        u32 => if (validateOverride(c.LLAMA_KV_OVERRIDE_TYPE_INT, ovrd)) return @truncate(@as(u64, @bitCast(ovrd.?.unnamed_0.val_i64))),
        []const u8 => if (validateOverride(c.LLAMA_KV_OVERRIDE_TYPE_STR, ovrd)) return cString(&ovrd.?.unnamed_0.val_str),
        else => @compileError("unsupported metadata type"),
    }
    return null;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "an arch-prefixed key takes the architecture name" {
    var buf: [64]u8 = undefined;
    const m: Meta = .{ .ctx = undefined, .arch_name = "qwen35" };
    try std.testing.expectEqualStrings("qwen35.vocab_size", m.name(.vocab_size, &buf));
    try std.testing.expectEqualStrings("tokenizer.ggml.model", m.name(.tokenizer_model, &buf));
}
