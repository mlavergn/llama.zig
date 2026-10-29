//! What every mtmd file shares: the C namespace, the image types, the
//! logger and the small string and GGUF utilities.
//!
//! # Provenance
//!
//! Ports the parts of `llama.cpp/tools/mtmd/clip-impl.h` and
//! `llama.cpp/tools/mtmd/clip.h` that are not name tables (those are
//! `names.zig`, generated) at v0.4.1 (`b29c606e2`). Each declaration names
//! what it ports and the line it began at.
//!
//! # One C namespace
//!
//! `c` is libllama's own (`llama_c`: `llama.h`, `ggml.h`, `ggml-backend.h`,
//! `ggml-cpu.h`, `gguf.h`), not a second import: two imports of `llama.h`
//! give two incompatible `*llama_model`, and mtmd hands its models to
//! libllama.

const std = @import("std");
const builtin = @import("builtin");

pub const c = @import("llama").impl.c;

/// Every allocation mtmd makes, as the C++'s `new` and `std::vector` do:
/// the C heap, so a buffer handed out through the C ABI can be freed there.
pub const gpa = std.heap.c_allocator;

/// `GGML_ASSERT`, kept in every build mode as upstream keeps it.
pub fn ggmlAssert(ok: bool, comptime file: [*:0]const u8, comptime line: c_int, comptime expr: [*:0]const u8) void {
    if (!ok) _ = c.ggml_abort(file, line, "GGML_ASSERT(%s) failed", expr);
}

/// `GGML_ABORT(msg)`.
pub fn abort(comptime file: [*:0]const u8, comptime line: c_int, comptime msg: [*:0]const u8) noreturn {
    _ = c.ggml_abort(file, line, msg);
    unreachable;
}

/// An exception escaping `main`: libc++abi's message, then `abort`.
pub fn abortUncaught(comptime kind: [*:0]const u8, what_: [*:0]const u8) noreturn {
    const f = cStderr();
    _ = fputs("libc++abi: terminating due to uncaught exception of type ", f);
    _ = fputs(kind, f);
    _ = fputs(": ", f);
    _ = fputs(what_, f);
    _ = fputs("\n", f);
    _ = fflush(f);
    std.process.abort();
}

/// Ports `clip_image_size` (clip.h:17 @b29c606e2).
pub const ImageSize = extern struct {
    width: c_int = 0,
    height: c_int = 0,

    pub fn eql(a: ImageSize, b: ImageSize) bool {
        return a.width == b.width and a.height == b.height;
    }

    /// `clip_image_size::area`, with its overflow guard.
    pub fn area(self: ImageSize) c_int {
        ggmlAssert(self.width >= 0 and self.width <= 46000, "clip.h", 28, "width  >= 0 && width  <= 46000");
        ggmlAssert(self.height >= 0 and self.height <= 46000, "clip.h", 29, "height >= 0 && height <= 46000");
        return self.width * self.height;
    }
};

/// Ports `clip_modality` (clip.h:37 @b29c606e2).
pub const Modality = enum(c_int) { vision, audio, gen_audio };

/// Ports `clip_flash_attn_type` (clip.h:43 @b29c606e2).
pub const FlashAttnType = enum(c_int) { auto = -1, disabled = 0, enabled = 1 };

/// Ports `clip_gen_process_type` (clip.h:90 @b29c606e2).
pub const GenProcessType = enum(c_int) { unknown, gen_code, gen_wav };

/// Ports `clip_image_u8` (clip-impl.h:584 @b29c606e2): RGB, eight bits a
/// channel. An empty buffer marks a placeholder, which reads as black and
/// ignores writes.
pub const ImageU8 = struct {
    buf: std.ArrayList(u8) = .empty,
    nx: c_int = 0,
    ny: c_int = 0,

    pub fn deinit(self: *ImageU8) void {
        self.buf.deinit(gpa);
    }

    pub fn clone(self: *const ImageU8) !ImageU8 {
        var out: ImageU8 = .{ .nx = self.nx, .ny = self.ny };
        try out.buf.appendSlice(gpa, self.buf.items);
        return out;
    }

    pub fn getSize(self: *const ImageU8) ImageSize {
        return .{ .width = self.nx, .height = self.ny };
    }

    pub fn setSize(self: *ImageU8, size: ImageSize, is_placeholder: bool) !void {
        self.nx = size.width;
        self.ny = size.height;
        if (is_placeholder) {
            self.buf.clearRetainingCapacity();
        } else {
            const n = @as(usize, @intCast(self.nx)) * @as(usize, @intCast(self.ny)) * 3;
            const old = self.buf.items.len;
            try self.buf.resize(gpa, n);
            if (n > old) @memset(self.buf.items[old..], 0);
        }
    }

    pub fn cpyBuf(self: *ImageU8, new_buf: []const u8) !void {
        self.buf.clearRetainingCapacity();
        try self.buf.appendSlice(gpa, new_buf);
    }

    pub fn isPlaceholder(self: *const ImageU8) bool {
        return self.buf.items.len == 0;
    }

    /// `get_ro_buf`; the C++ throws on a placeholder, which no caller reaches.
    pub fn roBuf(self: *const ImageU8) []const u8 {
        return self.buf.items;
    }

    pub fn getPixel(self: *const ImageU8, x: c_int, y: c_int) [3]u8 {
        if (self.isPlaceholder()) return .{ 0, 0, 0 };
        const idx = (@as(usize, @intCast(y)) * @as(usize, @intCast(self.nx)) + @as(usize, @intCast(x))) * 3;
        return .{ self.buf.items[idx], self.buf.items[idx + 1], self.buf.items[idx + 2] };
    }

    pub fn setPixel(self: *ImageU8, x: c_int, y: c_int, rgb: [3]u8) void {
        if (self.isPlaceholder()) return;
        const idx = (@as(usize, @intCast(y)) * @as(usize, @intCast(self.nx)) + @as(usize, @intCast(x))) * 3;
        self.buf.items[idx] = rgb[0];
        self.buf.items[idx + 1] = rgb[1];
        self.buf.items[idx + 2] = rgb[2];
    }

    pub fn nElements(self: *const ImageU8) usize {
        return @as(usize, @intCast(self.nx)) * @as(usize, @intCast(self.ny)) * 3;
    }
};

/// Ports `anyres_info` (clip-impl.h:670 @b29c606e2), a member of `clip_image_f32`.
pub const AnyresInfo = struct {
    grid_x: c_int = 0, // tiles per row, 0 means the image is not tiled
    grid_y: c_int = 0,
    orig_nx: c_int = 0, // size of the source image, used to drop the padding tokens
    orig_ny: c_int = 0,

    pub fn isTiled(self: AnyresInfo) bool {
        return self.grid_x > 0 and self.grid_y > 0;
    }
};

/// Ports `clip_image_f32` (clip-impl.h:657 @b29c606e2). Images are RGB
/// interleaved; audio uses one channel, `nx` frames by `ny` mel bins.
pub const ImageF32 = struct {
    /// marks the global view in e.g. DeepSeek-OCR models
    add_viewsep: bool = false,
    /// appends a learned newline (or EOI) token after the image
    add_newline: bool = false,
    /// deepseek4v: number of leading IMAGE_PAD embeddings, aligns IMAGE_START to the LLM compressor ratio
    /// depends on the chunk position, set at tokenize time (see mtmd_tokenizer::add_media)
    lead_pad: i32 = 0,
    anyres: AnyresInfo = .{},
    buf: std.ArrayList(f32) = .empty,
    nx_: c_int = 0,
    ny_: c_int = 0,

    pub fn deinit(self: *ImageF32) void {
        self.buf.deinit(gpa);
    }

    pub fn clone(self: *const ImageF32) !ImageF32 {
        var out = self.*;
        out.buf = .empty;
        try out.buf.appendSlice(gpa, self.buf.items);
        return out;
    }

    pub fn getSize(self: *const ImageF32) ImageSize {
        return .{ .width = self.nx_, .height = self.ny_ };
    }

    pub fn nx(self: *const ImageF32) c_int {
        return self.nx_;
    }

    pub fn ny(self: *const ImageF32) c_int {
        return self.ny_;
    }

    pub fn setSize(self: *ImageF32, size: ImageSize, is_placeholder: bool, is_audio: bool) !void {
        self.nx_ = size.width;
        self.ny_ = size.height;
        if (is_placeholder) {
            self.buf.clearRetainingCapacity();
        } else {
            const px = @as(usize, @intCast(self.nx_)) * @as(usize, @intCast(self.ny_));
            const n = if (is_audio) px else px * 3;
            const old = self.buf.items.len;
            try self.buf.resize(gpa, n);
            if (n > old) @memset(self.buf.items[old..], 0);
        }
    }

    pub fn cpyBuf(self: *ImageF32, new_buf: []const f32) !void {
        self.buf.clearRetainingCapacity();
        try self.buf.appendSlice(gpa, new_buf);
    }

    /// Ports `from_u8` (clip-impl.h:707 @b29c606e2), a member of `clip_image_f32`.
    pub fn fromU8(self: *ImageF32, img: *const ImageU8) !void {
        const size = img.getSize();
        self.nx_ = size.width;
        self.ny_ = size.height;
        if (img.isPlaceholder()) {
            self.buf.clearRetainingCapacity();
            return; // no-op
        }
        try self.buf.resize(gpa, img.nElements());
        for (img.roBuf(), 0..) |v, i| self.buf.items[i] = @as(f32, @floatFromInt(v)) / 255.0;
    }

    pub fn nElements(self: *const ImageF32) usize {
        return self.nPixels() * 3;
    }

    fn nPixels(self: *const ImageF32) usize {
        return @as(usize, @intCast(self.nx_)) * @as(usize, @intCast(self.ny_));
    }

    /// Ports `normalize` (clip-impl.h:726 @b29c606e2), a member of `clip_image_f32`.
    pub fn normalize(self: *ImageF32, mean: *const [3]f32, std_: *const [3]f32) void {
        if (self.isPlaceholder()) return; // no-op
        const b = self.buf.items;
        for (0..self.nPixels()) |i| {
            b[i * 3 + 0] = (b[i * 3 + 0] - mean[0]) / std_[0];
            b[i * 3 + 1] = (b[i * 3 + 1] - mean[1]) / std_[1];
            b[i * 3 + 2] = (b[i * 3 + 2] - mean[2]) / std_[2];
        }
    }

    pub fn roBuf(self: *const ImageF32) []const f32 {
        return self.buf.items;
    }

    pub fn isPlaceholder(self: *const ImageF32) bool {
        return self.buf.items.len == 0;
    }
};

/// Ports `clip_anyres_unpad` (clip-impl.h:765 @b29c606e2): the token area
/// kept after removing the padding the anyres resize added.
pub fn anyresUnpad(cur_w: c_int, cur_h: c_int, orig_w: c_int, orig_h: c_int) struct { off_x: c_int, off_y: c_int, out_w: c_int, out_h: c_int } {
    var r: @TypeOf(anyresUnpad(0, 0, 1, 1)) = .{ .off_x = 0, .off_y = 0, .out_w = cur_w, .out_h = cur_h };
    const fo = @as(f32, @floatFromInt(orig_w)) / @as(f32, @floatFromInt(orig_h));
    const fc = @as(f32, @floatFromInt(cur_w)) / @as(f32, @floatFromInt(cur_h));
    if (fo > fc) {
        const new_h: c_int = @intFromFloat(@floor(@as(f64, @floatFromInt(orig_h)) * @as(f64, @floatFromInt(cur_w)) / @as(f64, @floatFromInt(orig_w)) + 1e-7));
        r.off_y = @divTrunc(cur_h - new_h, 2);
        r.out_h = cur_h - 2 * r.off_y;
    } else {
        const new_w: c_int = @intFromFloat(@floor(@as(f64, @floatFromInt(orig_w)) * @as(f64, @floatFromInt(cur_h)) / @as(f64, @floatFromInt(orig_h)) + 1e-7));
        r.off_x = @divTrunc(cur_w - new_w, 2);
        r.out_w = cur_w - 2 * r.off_x;
    }
    return r;
}

/// Ports `dsv4_block_layout` (clip-impl.h:783 @b29c606e2): deepseek4v's
/// layout of the LLM token block built from the aligner grid.
pub const Dsv4BlockLayout = struct {
    rows: c_int, // grid rows, padded to an even count
    row_len: c_int, // grid width + 1 newline
    pad_last: c_int, // trailing pads
    n_out: c_int, // total block size, including lead pads and the start/end sentinels
};

/// Ports `dsv4_get_block_layout` (clip-impl.h:789 @b29c606e2).
pub fn dsv4GetBlockLayout(n_llm_w: c_int, n_llm_h: c_int, lead_pad: c_int) Dsv4BlockLayout {
    var bl: Dsv4BlockLayout = undefined;
    bl.rows = n_llm_h + @rem(n_llm_h, 2);
    bl.row_len = n_llm_w + 1;
    bl.pad_last = @rem(@divTrunc(bl.rows, 2) * bl.row_len, 2) * 2;
    bl.n_out = lead_pad + 1 + bl.rows * bl.row_len + bl.pad_last + 1;
    return bl;
}

/// Ports `clip_image_f32_batch` (clip-impl.h:854 @b29c606e2).
pub const ImageF32Batch = struct {
    entries: std.ArrayList(ImageF32) = .empty,
    is_audio: bool = false,

    pub fn deinit(self: *ImageF32Batch) void {
        for (self.entries.items) |*e| e.deinit();
        self.entries.deinit(gpa);
    }

    pub fn clone(self: *const ImageF32Batch) !ImageF32Batch {
        var out: ImageF32Batch = .{ .is_audio = self.is_audio };
        try out.entries.ensureTotalCapacity(gpa, self.entries.items.len);
        for (self.entries.items) |*e| out.entries.appendAssumeCapacity(try e.clone());
        return out;
    }
};

// -----------------------------------------------------------------------------
// logging

/// Ports `clip_log_callback_default` (clip-impl.h:802 @b29c606e2).
pub fn logCallbackDefault(level: c.enum_ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = level;
    _ = user_data;
    const stderr_ = cStderr();
    _ = fputs(text, stderr_);
    _ = fflush(stderr_);
}

/// Ports `clip_logger_state` (clip-impl.h:809 @b29c606e2) and
/// `g_logger_state` (mtmd/clip.cpp:30 @b29c606e2).
pub const LoggerState = struct {
    log_callback: c.ggml_log_callback = logCallbackDefault,
    log_callback_user_data: ?*anyopaque = null,
};

pub var g_logger_state: LoggerState = .{};

extern fn fputs(s: [*c]const u8, f: ?*anyopaque) c_int;
extern fn fflush(f: ?*anyopaque) c_int;
extern fn snprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, ...) c_int;

/// The C library's `stderr`.
pub fn cStderr() ?*anyopaque {
    if (comptime builtin.os.tag.isDarwin()) return @extern(*?*anyopaque, .{ .name = "__stderrp" }).*;
    if (comptime builtin.os.tag == .windows) {
        const f = @extern(*const fn (c_uint) callconv(.c) ?*anyopaque, .{ .name = "__acrt_iob_func" });
        return f(2);
    }
    return @extern(*?*anyopaque, .{ .name = "stderr" }).*;
}

/// Ports `clip_log_internal_v` (clip-impl.h:816 @b29c606e2): format into
/// 128 bytes, or a heap buffer when that is too short.
pub fn logInternal(level: c.enum_ggml_log_level, comptime fmt: [*:0]const u8, args: anytype) void {
    var buffer: [128]u8 = undefined;
    const len = @call(.auto, snprintf, .{ &buffer, buffer.len, fmt } ++ args);
    const cb = g_logger_state.log_callback orelse return;
    if (len < 128) {
        cb(level, @ptrCast(&buffer), g_logger_state.log_callback_user_data);
    } else {
        const n: usize = @intCast(len);
        const buffer2 = gpa.alloc(u8, n + 1) catch return;
        defer gpa.free(buffer2);
        _ = @call(.auto, snprintf, .{ buffer2.ptr, n + 1, fmt } ++ args);
        buffer2[n] = 0;
        cb(level, @ptrCast(buffer2.ptr), g_logger_state.log_callback_user_data);
    }
}

/// `LOG_DBG` and `LOG_TRC` (clip-impl.h:844, 843 @b29c606e2).
pub fn logDbg(comptime fmt: [*:0]const u8, args: anytype) void {
    logInternal(c.GGML_LOG_LEVEL_DEBUG, fmt, args);
}

/// `LOG_INF` (clip-impl.h:845 @b29c606e2).
pub fn logInf(comptime fmt: [*:0]const u8, args: anytype) void {
    logInternal(c.GGML_LOG_LEVEL_INFO, fmt, args);
}

/// `LOG_WRN` (clip-impl.h:846 @b29c606e2).
pub fn logWrn(comptime fmt: [*:0]const u8, args: anytype) void {
    logInternal(c.GGML_LOG_LEVEL_WARN, fmt, args);
}

/// `LOG_ERR` (clip-impl.h:847 @b29c606e2).
pub fn logErr(comptime fmt: [*:0]const u8, args: anytype) void {
    logInternal(c.GGML_LOG_LEVEL_ERROR, fmt, args);
}

/// `LOG_CNT` (clip-impl.h:848 @b29c606e2).
pub fn logCnt(comptime fmt: [*:0]const u8, args: anytype) void {
    logInternal(c.GGML_LOG_LEVEL_CONT, fmt, args);
}

// -----------------------------------------------------------------------------
// string utilities

/// Ports `string_format` (clip-impl.h:897 @b29c606e2), with the C's
/// `snprintf`. The caller owns the string.
pub fn stringFormat(a: std.mem.Allocator, comptime fmt: [*:0]const u8, args: anytype) ![:0]u8 {
    const size = @call(.auto, snprintf, .{ @as(?[*]u8, null), 0, fmt } ++ args);
    ggmlAssert(size >= 0, "clip-impl.h", 903, "size >= 0 && size < INT_MAX");
    const buf = try a.allocSentinel(u8, @intCast(size), 0);
    _ = @call(.auto, snprintf, .{ buf.ptr, @as(usize, @intCast(size)) + 1, fmt } ++ args);
    return buf;
}

/// Ports `string_replace_all` (clip-impl.h:912 @b29c606e2).
pub fn stringReplaceAll(a: std.mem.Allocator, s: []const u8, search: []const u8, replace: []const u8) ![]u8 {
    if (search.len == 0) return a.dupe(u8, s);
    var out: std.ArrayList(u8) = .empty;
    var last: usize = 0;
    while (std.mem.indexOfPos(u8, s, last, search)) |pos| {
        try out.appendSlice(a, s[last..pos]);
        try out.appendSlice(a, replace);
        last = pos + search.len;
    }
    try out.appendSlice(a, s[last..]);
    return out.toOwnedSlice(a);
}

/// Ports `string_split_str` (clip-impl.h:930 @b29c606e2).
pub fn stringSplitStr(a: std.mem.Allocator, s: []const u8, delimiter: []const u8) !std.ArrayList([]const u8) {
    var tokens: std.ArrayList([]const u8) = .empty;
    var rest = s;
    while (std.mem.indexOf(u8, rest, delimiter)) |pos| {
        try tokens.append(a, rest[0..pos]);
        rest = rest[pos + delimiter.len ..];
    }
    try tokens.append(a, rest);
    return tokens;
}

// -----------------------------------------------------------------------------
// gguf utilities

/// Ports `gguf_data_to_str` (clip-impl.h:960 @b29c606e2): `std::to_string`
/// per type, `%f` for the floats.
pub fn ggufDataToStr(a: std.mem.Allocator, t: c.enum_gguf_type, data: ?*const anyopaque, i: usize) ![:0]u8 {
    return switch (t) {
        c.GGUF_TYPE_UINT8 => stringFormat(a, "%u", .{@as(c_uint, @as([*]const u8, @ptrCast(data))[i])}),
        c.GGUF_TYPE_INT8 => stringFormat(a, "%d", .{@as(c_int, @as([*]const i8, @ptrCast(data))[i])}),
        c.GGUF_TYPE_UINT16 => stringFormat(a, "%u", .{@as(c_uint, @as([*]align(1) const u16, @ptrCast(data))[i])}),
        c.GGUF_TYPE_INT16 => stringFormat(a, "%d", .{@as(c_int, @as([*]align(1) const i16, @ptrCast(data))[i])}),
        c.GGUF_TYPE_UINT32 => stringFormat(a, "%u", .{@as([*]align(1) const u32, @ptrCast(data))[i]}),
        c.GGUF_TYPE_INT32 => stringFormat(a, "%d", .{@as([*]align(1) const i32, @ptrCast(data))[i]}),
        c.GGUF_TYPE_UINT64 => stringFormat(a, "%llu", .{@as(c_ulonglong, @as([*]align(1) const u64, @ptrCast(data))[i])}),
        c.GGUF_TYPE_INT64 => stringFormat(a, "%lld", .{@as(c_longlong, @as([*]align(1) const i64, @ptrCast(data))[i])}),
        c.GGUF_TYPE_FLOAT32 => stringFormat(a, "%f", .{@as(f64, @as([*]align(1) const f32, @ptrCast(data))[i])}),
        c.GGUF_TYPE_FLOAT64 => stringFormat(a, "%f", .{@as([*]align(1) const f64, @ptrCast(data))[i]}),
        c.GGUF_TYPE_BOOL => a.dupeSentinel(u8, if (@as([*]const i8, @ptrCast(data))[i] != 0) "true" else "false", 0),
        else => stringFormat(a, "unknown type %d", .{@as(c_int, @intCast(t))}),
    };
}

/// Ports `gguf_kv_to_str` (clip-impl.h:977 @b29c606e2).
pub fn ggufKvToStr(a: std.mem.Allocator, ctx_gguf: ?*const c.gguf_context, i: i64) ![:0]u8 {
    const t = c.gguf_get_kv_type(ctx_gguf, i);
    switch (t) {
        c.GGUF_TYPE_STRING => return a.dupeSentinel(u8, std.mem.span(c.gguf_get_val_str(ctx_gguf, i)), 0),
        c.GGUF_TYPE_ARRAY => {
            const arr_type = c.gguf_get_arr_type(ctx_gguf, i);
            const arr_n: c_int = @intCast(c.gguf_get_arr_n(ctx_gguf, i));
            const data: ?*const anyopaque = if (arr_type == c.GGUF_TYPE_STRING) null else c.gguf_get_arr_data(ctx_gguf, i);
            var ss: std.ArrayList(u8) = .empty;
            try ss.append(a, '[');
            var j: c_int = 0;
            while (j < arr_n) : (j += 1) {
                if (arr_type == c.GGUF_TYPE_STRING) {
                    const v0 = std.mem.span(c.gguf_get_arr_str(ctx_gguf, i, @intCast(j)));
                    // escape quotes
                    const v1 = try stringReplaceAll(a, v0, "\\", "\\\\");
                    const v2 = try stringReplaceAll(a, v1, "\"", "\\\"");
                    try ss.append(a, '"');
                    try ss.appendSlice(a, v2);
                    try ss.append(a, '"');
                } else if (arr_type == c.GGUF_TYPE_ARRAY) {
                    try ss.appendSlice(a, "???");
                } else {
                    try ss.appendSlice(a, try ggufDataToStr(a, arr_type, data, @intCast(j)));
                }
                if (j < arr_n - 1) try ss.appendSlice(a, ", ");
            }
            try ss.append(a, ']');
            return a.dupeSentinel(u8, ss.items, 0);
        },
        else => return ggufDataToStr(a, t, c.gguf_get_val_data(ctx_gguf, i), 0),
    }
}

// -----------------------------------------------------------------------------
// testing

/// `std.testing.refAllDecls`, descending into nested structs: a method is
/// compiled only when referenced, and Zig 0.17 dropped the recursive form,
/// so a test of a file of types would otherwise compile none of their bodies.
/// Re-exported namespaces are skipped: the C import (whose untranslatable
/// macros only fail when touched) and other files, which test themselves.
pub fn refAllDeep(comptime Ty: type) void {
    comptime refDeep(Ty, 0);
}

fn refDeep(comptime Ty: type, comptime depth: usize) void {
    @setEvalBranchQuota(1000000);
    if (depth > 4) return;
    for (@typeInfo(Ty).@"struct".decl_names) |name| {
        if (isNamespaceName(name)) continue;
        if (@TypeOf(@field(Ty, name)) == type) {
            const Inner = @field(Ty, name);
            // only this module's own types: std's refer back to themselves
            if (@typeInfo(Inner) == .@"struct" and !std.mem.startsWith(u8, @typeName(Inner), "std.")) refDeep(Inner, depth + 1);
        } else {
            _ = &@field(Ty, name);
        }
    }
}

fn isNamespaceName(comptime name: []const u8) bool {
    const skip = [_][]const u8{ "c", "std", "builtin", "impl", "names", "model", "model_mod", "graph", "clip", "image", "audio", "vendor", "llama" };
    for (skip) |n| {
        if (std.mem.eql(u8, n, name)) return true;
    }
    return false;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    refAllDeep(@This());
}
