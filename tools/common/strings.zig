//! The string helpers of `common/common.cpp` the tools call.
//!
//! # Provenance
//!
//! Ports `string_remove_suffix`, `string_parse_kv_override` and the
//! `std::string` specialisation of `string_split` from
//! `llama.cpp/common/common.cpp` and `common/common.h` (v0.3.0, `c1d0e7a00`).
//! Each declaration names what it ports and the line it began at.

const std = @import("std");
const c = @import("tools_h");
const log = @import("log.zig");

extern fn atol(s: [*:0]const u8) c_long;
extern fn atof(s: [*:0]const u8) f64;

/// Ports `string_remove_suffix` (common/common.h:833 @c1d0e7a00), returning the
/// shortened string where the C shortens in place and returns true.
pub fn removeSuffix(str: []const u8, suffix: []const u8) ?[]const u8 {
    if (!std.mem.endsWith(u8, str, suffix)) return null;
    return str[0 .. str.len - suffix.len];
}

/// Ports `string_split` (common/common.h:801 @c1d0e7a00): the parts
/// between each `delim`, including empty ones and the last. The parts
/// borrow `str`; the caller frees the list.
pub fn split(allocator: std.mem.Allocator, str: []const u8, delim: u8) !std.ArrayList([]const u8) {
    var parts: std.ArrayList([]const u8) = .empty;
    errdefer parts.deinit(allocator);
    var it = std.mem.splitScalar(u8, str, delim);
    while (it.next()) |part| try parts.append(allocator, part);
    return parts;
}

/// Ports `string_parse_kv_override` (common/common.cpp:682 @c1d0e7a00).
///
/// Return: false, having logged why, for a malformed override.
pub fn parseKvOverride(data: [:0]const u8, overrides: *std.ArrayList(c.llama_model_kv_override), allocator: std.mem.Allocator) !bool {
    const func = "string_parse_kv_override";
    const sep_idx = std.mem.indexOfScalar(u8, data, '=') orelse {
        log.comErr(func, "%s: malformed KV override '%s'\n", .{ func, data.ptr });
        return false;
    };
    if (sep_idx >= 128) {
        log.comErr(func, "%s: malformed KV override '%s'\n", .{ func, data.ptr });
        return false;
    }
    var kvo: c.llama_model_kv_override = std.mem.zeroes(c.llama_model_kv_override);
    @memcpy(kvo.key[0..sep_idx], data[0..sep_idx]);
    kvo.key[sep_idx] = 0;
    const sep = data[sep_idx + 1 ..];
    if (std.mem.startsWith(u8, sep, "int:")) {
        kvo.tag = c.LLAMA_KV_OVERRIDE_TYPE_INT;
        kvo.unnamed_0.val_i64 = atol(sep[4..].ptr);
    } else if (std.mem.startsWith(u8, sep, "float:")) {
        kvo.tag = c.LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        kvo.unnamed_0.val_f64 = atof(sep[6..].ptr);
    } else if (std.mem.startsWith(u8, sep, "bool:")) {
        kvo.tag = c.LLAMA_KV_OVERRIDE_TYPE_BOOL;
        const v = sep[5..];
        if (std.mem.eql(u8, v, "true")) {
            kvo.unnamed_0.val_bool = true;
        } else if (std.mem.eql(u8, v, "false")) {
            kvo.unnamed_0.val_bool = false;
        } else {
            log.comErr(func, "%s: invalid boolean value for KV override '%s'\n", .{ func, data.ptr });
            return false;
        }
    } else if (std.mem.startsWith(u8, sep, "str:")) {
        kvo.tag = c.LLAMA_KV_OVERRIDE_TYPE_STR;
        const v = sep[4..];
        if (v.len > 127) {
            log.comErr(func, "%s: malformed KV override '%s', value cannot exceed 127 chars\n", .{ func, data.ptr });
            return false;
        }
        @memcpy(kvo.unnamed_0.val_str[0..v.len], v);
        kvo.unnamed_0.val_str[127] = 0;
    } else {
        log.comErr(func, "%s: invalid type for KV override '%s'\n", .{ func, data.ptr });
        return false;
    }
    try overrides.append(allocator, kvo);
    return true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
