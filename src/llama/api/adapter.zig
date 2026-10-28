//! `llama.h`'s adapter entry points: LoRA adapters and control vectors.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-adapter.cpp` -- `llama_adapter_lora_init`, the
//!   metadata accessors, `llama_adapter_lora_free` and the aLoRA invocation
//!   tokens
//! - `llama.cpp/src/llama-context.cpp` -- `llama_set_adapters_lora` and
//!   `llama_set_adapter_cvec`
//!
//! All at v0.4.0 (`5266f24da`).
//!
//! Each function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call these beside the C++ that defines the same
//! names, and `exports.zig` exports them once libllama swaps.

const std = @import("std");
const impl = @import("../impl.zig");
const adapter = @import("../adapter.zig");
const ctx_api = @import("context.zig");

const c = impl.c;
const Lora = adapter.Lora;

/// The port's adapter behind a `llama_adapter_lora *`.
pub fn loraOf(a: ?*const c.struct_llama_adapter_lora) *Lora {
    return @ptrCast(@alignCast(@constCast(a.?)));
}

/// The metadata entry `i` steps into the map's iteration order, as
/// `std::advance(begin(), i)` reaches it.
fn metaAt(a: *Lora, i: usize) struct { key: []const u8, value: [:0]u8 } {
    var it = a.gguf_kv.iterator();
    var k: usize = 0;
    while (it.next()) |e| : (k += 1) {
        if (k == i) return .{ .key = e.key, .value = e.value.* };
    }
    unreachable;
}

/// `snprintf(buf, buf_size, "%s", s)` for a string that is not
/// NUL-terminated where it is stored.
fn printStr(buf: [*c]u8, buf_size: usize, s: []const u8) i32 {
    return c.snprintf(buf, buf_size, "%.*s", @as(c_int, @intCast(s.len)), s.ptr);
}

/// Ports `llama_adapter_lora_init` (llama-adapter.cpp:423 @5266f24da).
pub fn llama_adapter_lora_init(model: ?*c.struct_llama_model, path_lora: [*c]const u8) callconv(.c) ?*c.struct_llama_adapter_lora {
    const a = adapter.loraInit(ctx_api.modelOf(model), @ptrCast(path_lora)) orelse return null;
    return @ptrCast(a);
}

/// Ports `llama_adapter_meta_val_str` (llama-adapter.cpp:438 @5266f24da).
///
/// Return: the length `snprintf` reports, or -1 with `buf` emptied when the
/// key is absent.
pub fn llama_adapter_meta_val_str(a: ?*const c.struct_llama_adapter_lora, key: [*c]const u8, buf: [*c]u8, buf_size: usize) callconv(.c) i32 {
    const k: [*:0]const u8 = @ptrCast(key);
    const v = loraOf(a).gguf_kv.get(std.mem.span(k)) orelse {
        if (buf_size > 0) buf[0] = 0;
        return -1;
    };
    return c.snprintf(buf, buf_size, "%s", v.*.ptr);
}

/// Ports `llama_adapter_meta_count` (llama-adapter.cpp:449 @5266f24da).
pub fn llama_adapter_meta_count(a: ?*const c.struct_llama_adapter_lora) callconv(.c) i32 {
    return @truncate(@as(i64, @intCast(loraOf(a).gguf_kv.count())));
}

/// Ports `llama_adapter_meta_key_by_index` (llama-adapter.cpp:453 @5266f24da).
///
/// Return: the length `snprintf` reports, or -1 with `buf` emptied when `i`
/// is out of range.
pub fn llama_adapter_meta_key_by_index(a: ?*const c.struct_llama_adapter_lora, i: i32, buf: [*c]u8, buf_size: usize) callconv(.c) i32 {
    const l = loraOf(a);
    if (i < 0 or i >= llama_adapter_meta_count(a)) {
        if (buf_size > 0) buf[0] = 0;
        return -1;
    }
    return printStr(buf, buf_size, metaAt(l, @intCast(i)).key);
}

/// Ports `llama_adapter_meta_val_str_by_index` (llama-adapter.cpp:465 @5266f24da).
///
/// Return: the length `snprintf` reports, or -1 with `buf` emptied when `i`
/// is out of range.
pub fn llama_adapter_meta_val_str_by_index(a: ?*const c.struct_llama_adapter_lora, i: i32, buf: [*c]u8, buf_size: usize) callconv(.c) i32 {
    const l = loraOf(a);
    if (i < 0 or i >= llama_adapter_meta_count(a)) {
        if (buf_size > 0) buf[0] = 0;
        return -1;
    }
    return c.snprintf(buf, buf_size, "%s", metaAt(l, @intCast(i)).value.ptr);
}

/// Ports `llama_adapter_lora_free` (llama-adapter.cpp:477 @5266f24da).
pub fn llama_adapter_lora_free(a: ?*c.struct_llama_adapter_lora) callconv(.c) void {
    adapter.loraFree(if (a == null) null else loraOf(a));
}

/// Ports `llama_adapter_get_alora_n_invocation_tokens` (llama-adapter.cpp:490
/// @5266f24da).
pub fn llama_adapter_get_alora_n_invocation_tokens(a: ?*const c.struct_llama_adapter_lora) callconv(.c) u64 {
    if (a == null) return 0;
    return loraOf(a).alora_invocation_tokens.items.len;
}

/// Ports `llama_adapter_get_alora_invocation_tokens` (llama-adapter.cpp:497
/// @5266f24da).
///
/// Return: the tokens, borrowed from the adapter; null when the vector
/// never allocated, as an empty `std::vector`'s `data()` is.
pub fn llama_adapter_get_alora_invocation_tokens(a: ?*const c.struct_llama_adapter_lora) callconv(.c) [*c]const c.llama_token {
    impl.assert(a != null, "llama-adapter.cpp", 498, "adapter");
    const toks = &loraOf(a).alora_invocation_tokens;
    if (toks.capacity == 0) return null;
    return toks.items.ptr;
}

/// Ports `llama_set_adapters_lora` (llama-context.cpp:3986 @5266f24da).
///
/// Return: 0.
pub fn llama_set_adapters_lora(ctx: ?*c.struct_llama_context, adapters: [*c]?*c.struct_llama_adapter_lora, n_adapters: usize, scales: [*c]f32) callconv(.c) i32 {
    if (adapters == null or scales == null) {
        impl.assert(n_adapters == 0, "llama-context.cpp", 3992, "n_adapters == 0 && \"invalid llama_set_adapters_lora call\"");
    }
    ctx_api.ctxOf(ctx).setAdaptersLora(@ptrCast(adapters), n_adapters, scales);
    return 0;
}

/// Ports `llama_set_adapter_cvec` (llama-context.cpp:4000 @5266f24da).
///
/// Return: 0 when the control vector was applied, -1 otherwise.
pub fn llama_set_adapter_cvec(ctx: ?*c.struct_llama_context, data: [*c]const f32, len: usize, n_embd: i32, il_start: i32, il_end: i32) callconv(.c) i32 {
    const res = ctx_api.ctxOf(ctx).setAdapterCvec(data, len, n_embd, il_start, il_end);
    return if (res) 0 else -1;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
