//! Architecture, metadata-key and tensor names.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-arch.cpp` -- `LLM_KV::operator()`,
//!   `LLM_TN_IMPL::str`, `llm_arch_all`, `llm_arch_name`,
//!   `llm_arch_from_string`, `llm_tensor_info_for` and the five
//!   architecture predicates
//! - `llama.cpp/src/llama-arch.h` -- `LLM_KV`, `LLM_TN_IMPL`
//!
//! All at v0.4.1 (`b29c606e2`). The four tables those functions read are
//! generated into `arch_table.zig`.
//!
//! # Formatting
//!
//! The C++ formats a name with `vsnprintf`: a key's format takes the
//! architecture name for its one `%s`, a tensor's takes the block and
//! expert indices for up to two `%d`. `scripts/gen-arch` refuses a table
//! entry with any other conversion, so substituting those two by hand here
//! is the whole of `vsnprintf` these strings can reach.

const std = @import("std");
const impl = @import("impl.zig");
const table = @import("arch_table.zig");

pub const Arch = table.Arch;
pub const Kv = table.Kv;
pub const Tensor = table.Tensor;
pub const TensorLayer = table.TensorLayer;
pub const TensorInfo = table.TensorInfo;

/// Longest name either formatter can produce with a `suffix` of
/// `max_suffix` bytes: every format and architecture name is shorter than
/// 64, and an `i32` prints in 11.
pub const max_name = 64 + 64 + 2 * 11 + 1 + max_suffix;
/// The longest suffix a name may carry; upstream's are `weight`, `bias`
/// and the like.
pub const max_suffix = 64;

comptime {
    for (table.arch_names) |n| if (n.?.len >= 64) @compileError("arch name too long");
    for (table.kv_names) |n| if (n.?.len >= 64) @compileError("key format too long");
    for (table.tensor_names) |n| if (n) |s| if (s.len >= 64) @compileError("tensor format too long");
}

/// A formatted name, held inline so a caller needs no allocator.
pub const Name = struct {
    buf: [max_name]u8 = undefined,
    len: usize = 0,

    /// Return: the name, borrowed from `self`.
    pub fn slice(self: *const Name) []const u8 {
        return self.buf[0..self.len];
    }

    fn append(self: *Name, s: []const u8) void {
        @memcpy(self.buf[self.len..][0..s.len], s);
        self.len += s.len;
    }
};

/// Ports `LLM_KV` (llama-arch.h:717 @b29c606e2) and its call operator,
/// `LLM_KV::operator()` (llama-arch.cpp:995 @b29c606e2): the key's format
/// with the architecture name for its `%s`, then `.suffix` if there is one.
///
/// Every `llm_arch` and every `llm_kv` is in its map, so the two `at`
/// calls cannot throw for a value of these enums.
///
/// Parameters:
/// - `arch`: the architecture whose name the key takes.
/// - `kv`: the key.
/// - `suffix`: appended after a `.`, or null; at most `max_suffix` bytes.
///
/// Return: the key's name.
pub fn kvName(arch: Arch, kv: Kv, suffix: ?[]const u8) Name {
    var out: Name = .{};
    const f = table.kv_names[@intCast(@backingInt(kv))].?;
    const a = table.arch_names[@intCast(@backingInt(arch))].?;
    if (std.mem.find(u8, f, "%s")) |at| {
        out.append(f[0..at]);
        out.append(a);
        out.append(f[at + 2 ..]);
    } else {
        out.append(f);
    }
    if (suffix) |s| {
        std.debug.assert(s.len <= max_suffix);
        out.append(".");
        out.append(s);
    }
    return out;
}

/// Ports `LLM_TN_IMPL`'s `str` (llama-arch.cpp:1009 @b29c606e2): the tensor's
/// format with `bid` for its first `%d` and `xid` for its second, then
/// `.suffix` if there is one. A tensor with no name aborts, as the C++ does.
///
/// Parameters:
/// - `tensor`: the tensor.
/// - `suffix`: appended after a `.`, or null; at most `max_suffix` bytes.
/// - `bid`: the block index, -1 when there is none.
/// - `xid`: the expert index, -1 when there is none.
///
/// Return: the tensor's name.
pub fn tensorName(tensor: Tensor, suffix: ?[]const u8, bid: i32, xid: i32) Name {
    const f = table.tensor_names[@intCast(@backingInt(tensor))] orelse
        impl.c.ggml_abort("llama-arch.cpp", 1011, "unknown tensor name for tensor id %d", @as(c_int, @backingInt(tensor)));
    var out: Name = .{};
    const args = [2]i32{ bid, xid };
    var next: usize = 0;
    var i: usize = 0;
    while (i < f.len) {
        if (f[i] == '%' and i + 1 < f.len and f[i + 1] == 'd') {
            var num: [16]u8 = undefined;
            out.append(std.mem.print(&num, "{d}", .{args[next]}) catch unreachable);
            next += 1;
            i += 2;
        } else {
            out.append(f[i .. i + 1]);
            i += 1;
        }
    }
    if (suffix) |s| {
        std.debug.assert(s.len <= max_suffix);
        out.append(".");
        out.append(s);
    }
    return out;
}

/// Ports `llm_arch_all` (llama-arch.cpp:1023 @b29c606e2): every
/// architecture, in the map's key order.
///
/// Return: the list, static.
pub fn archAll() []const Arch {
    return comptime std.enums.values(Arch);
}

/// Ports `llm_arch_name` (llama-arch.cpp:1032 @b29c606e2).
///
/// Return: the architecture's name, static.
pub fn archName(arch: Arch) [:0]const u8 {
    return table.arch_names[@intCast(@backingInt(arch))] orelse "unknown";
}

/// Ports `llm_arch_from_string` (llama-arch.cpp:1040 @b29c606e2): the first
/// architecture in key order whose name is `name`.
///
/// Return: the architecture, or `.unknown`.
pub fn archFromString(name: []const u8) Arch {
    for (table.arch_names, 0..) |n, i| {
        if (n) |s| if (std.mem.eql(u8, s, name)) return @fromBackingInt(@intCast(i));
    }
    return .unknown;
}

/// Ports `llm_tensor_info_for` (llama-arch.cpp:1050 @b29c606e2).
///
/// Parameters:
/// - `tensor`: the tensor.
/// - `exc`: receives what is thrown.
///
/// Return: the tensor's info, or `error.Thrown` with `std::out_of_range`
/// for one `LLM_TENSOR_INFOS` lacks, as `map::at` throws.
pub fn tensorInfoFor(tensor: Tensor, exc: *impl.Exception) impl.Thrown!TensorInfo {
    return table.tensor_infos[@intCast(@backingInt(tensor))] orelse
        exc.throw(.out_of_range, "map::at:  key not found", .{});
}

/// Ports `llm_arch_is_recurrent` (llama-arch.cpp:1054 @b29c606e2).
pub fn isRecurrent(arch: Arch) bool {
    return switch (arch) {
        .mamba, .mamba2, .rwkv6, .rwkv6qwen2, .rwkv7, .arwkv7 => true,
        else => false,
    };
}

/// Ports `llm_arch_is_hybrid` (llama-arch.cpp:1068 @b29c606e2).
pub fn isHybrid(arch: Arch) bool {
    return switch (arch) {
        .jamba,
        .falcon_h1,
        .plamo2,
        .granite_hybrid,
        .lfm2,
        .lfm2moe,
        .nemotron_h,
        .nemotron_h_moe,
        .qwen3next,
        .kimi_linear,
        .bailingmoe3,
        .kimi_k3,
        .qwen35,
        .qwen35moe,
        .qwen4exp,
        .deepseek4,
        .minimax_01,
        => true,
        else => false,
    };
}

/// Ports `llm_arch_is_diffusion` (llama-arch.cpp:1093 @b29c606e2).
pub fn isDiffusion(arch: Arch) bool {
    return switch (arch) {
        .dream, .llada, .llada_moe, .rnd1 => true,
        else => false,
    };
}

/// Ports `llm_arch_supports_rs_rollback` (llama-arch.cpp:1105 @b29c606e2).
pub fn supportsRsRollback(arch: Arch) bool {
    return switch (arch) {
        .kimi_k3, .qwen35, .qwen35moe, .qwen4exp, .deepseek4, .nemotron_h, .nemotron_h_moe, .lfm2, .lfm2moe, .bailingmoe3 => true,
        else => false,
    };
}

/// Ports `llm_arch_supports_sm_tensor` (llama-arch.cpp:1123 @b29c606e2).
pub fn supportsSmTensor(arch: Arch) bool {
    return switch (arch) {
        .grok,
        .mpt,
        .plamo2,
        .minicpm3,
        .gemma3n,
        .mamba,
        .mamba2,
        .jamba,
        .falcon_h1,
        .olmo2,
        .olmoe,
        .deepseek2,
        .deepseek32,
        .hy_v4,
        .dots3note,
        .glm_dsa,
        .bitnet,
        .t5,
        .nemotron_h,
        .nemotron_h_moe,
        .granite_hybrid,
        .minimax_01,
        .minimax_m2,
        .minimax_m3,
        .mistral4,
        .kimi_linear,
        .bailingmoe3,
        .kimi_k3,
        .qwen3tts,
        .qwen4exp, // TODO: fix test-llama-archs
        => false,
        else => true,
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a key takes the architecture name and an optional suffix" {
    try std.testing.expectEqualStrings("qwen35.context_length", kvName(.qwen35, .context_length, null).slice());
    try std.testing.expectEqualStrings("general.architecture", kvName(.qwen35, .general_architecture, null).slice());
    try std.testing.expectEqualStrings("llama.context_length.x", kvName(.llama, .context_length, "x").slice());
}

test "a tensor name takes the block and expert indices" {
    try std.testing.expectEqualStrings("token_embd.weight", tensorName(.token_embd, "weight", -1, -1).slice());
    try std.testing.expectEqualStrings("blk.3.attn_norm.weight", tensorName(.attn_norm, "weight", 3, -1).slice());
    try std.testing.expectEqualStrings("blk.-1.attn_norm", tensorName(.attn_norm, null, -1, -1).slice());
}

test "an architecture round-trips through its name" {
    for (archAll()) |a| {
        if (a == .unknown) continue;
        try std.testing.expectEqual(a, archFromString(archName(a)));
    }
    try std.testing.expectEqual(Arch.unknown, archFromString("no-such-arch"));
}

test "a tensor without an info throws out_of_range" {
    var exc: impl.Exception = .{};
    try std.testing.expectError(error.Thrown, tensorInfoFor(.attn_rot_embd, &exc));
    try std.testing.expectEqualStrings("map::at:  key not found", exc.what());
}
