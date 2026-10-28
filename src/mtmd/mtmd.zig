//! libmtmd's public API: the context, bitmaps, tokenizing a prompt with
//! media markers into chunks, encoding them, and audio generation.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/mtmd.cpp`, the C types of
//! `llama.cpp/tools/mtmd/mtmd.h` and `mtmd_input_part` of
//! `llama.cpp/tools/mtmd/mtmd-internal.h`, at v0.4.0 (`5266f24da`). Each
//! declaration names what it ports and the line it began at.
//!
//! # The C ABI
//!
//! Every `mtmd_*` function is a `pub` Zig function with the C signature and
//! no export, as `src/llama/api/` is: `exports.zig` exports them, and the
//! in-process harnesses call them beside upstream's libmtmd, which defines
//! the same names. libllama is reached through the port's `api`, never the
//! C symbols, for the same reason.
//!
//! `mtmd.h`'s structs are declared here by hand rather than imported:
//! importing `mtmd.h` would import `llama.h` a second time, and two imports
//! give two incompatible `*llama_model`.
//!
//! # Not ported
//!
//! The debugging entry points (`mtmd_debug_encode_image` and its three
//! siblings, mtmd.cpp:2531-2631) take `std::vector`s, have C++ linkage and
//! no caller outside upstream's `tools/mtmd/debug/`.

const std = @import("std");
const impl = @import("impl.zig");
const names = @import("names.zig");
const clip = @import("clip.zig");
const image = @import("image.zig");
const audio = @import("audio.zig");
const lapi = @import("llama").api;
const c = impl.c;
const gpa = impl.gpa;

const Token = c.llama_token;
const token_null: Token = -1; // LLAMA_TOKEN_NULL

extern fn snprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, ...) c_int;
extern fn free(p: ?*anyopaque) void;

/// `MTMD_SERIALIZATION_VERSION` (mtmd.cpp:30 @5266f24da).
const serialization_version: u64 = 2;

// -----------------------------------------------------------------------------
// mtmd.h

/// Ports `mtmd_input_chunk_type` (mtmd.h:54 @5266f24da).
pub const ChunkType = enum(c_int) { text, image, audio, _ };
const chunk_type_count: u32 = 3;

/// Ports `mtmd_input_text` (mtmd.h:69 @5266f24da).
pub const InputText = extern struct {
    text: ?[*]const u8,
    text_len: usize,
    add_special: bool,
    parse_special: bool,
};

/// Ports `mtmd_input_part` (mtmd.h:76 @5266f24da).
pub const InputPart = extern struct {
    // only text or bitmap can be set, not both
    text: ?*const InputText,
    bitmap: ?*const Bitmap,
};

pub const ProgressCallback = ?*const fn (f32, ?*anyopaque) callconv(.c) bool;

/// Ports `mtmd_context_params` (mtmd.h:97 @5266f24da).
pub const ContextParams = extern struct {
    use_gpu: bool,
    device: c.ggml_backend_dev_t,
    print_timings: bool,
    n_threads: c_int,
    image_marker: ?[*:0]const u8, // deprecated, use media_marker instead
    media_marker: ?[*:0]const u8,
    flash_attn_type: c.enum_llama_flash_attn_type,
    warmup: bool, // whether to run a warmup encode pass after initialization
    image_min_tokens: c_int,
    image_max_tokens: c_int,
    cb_eval: c.ggml_backend_sched_eval_callback,
    cb_eval_user_data: ?*anyopaque,
    batch_max_tokens: i32,
    progress_callback: ProgressCallback,
    progress_callback_user_data: ?*anyopaque,
};

/// Ports `mtmd_bitmap_lazy_callback` (mtmd.h:208 @5266f24da).
pub const LazyCallback = ?*const fn (usize, ?*anyopaque, *?*Bitmap, *?[*:0]u8) callconv(.c) c_int;

/// Ports `mtmd_decoder_pos` (mtmd.h:274 @5266f24da).
pub const DecoderPos = extern struct { t: u32, x: u32, y: u32, z: u32 };

/// Ports `mtmd_caps` (mtmd.h:360 @5266f24da).
pub const Caps = extern struct { inp_vision: bool, inp_audio: bool };

/// Ports `mtmd_gen_audio_type` (mtmd.h:370 @5266f24da).
pub const GenAudioType = enum(c_int) { none, qwen3tts, pockettts };

/// Ports `mtmd_gen_audio_info` (mtmd.h:376 @5266f24da).
pub const GenAudioInfo = extern struct {
    type: GenAudioType,
    sample_rate: i32,
    model_variant: ?[*:0]const u8,
};

/// Ports `mtmd_gen_process_type` (mtmd.h:385 @5266f24da).
pub const GenProcessType = enum(c_int) { gen_code, gen_wav, _ };

/// Ports `mtmd_gen_inp` (mtmd.h:392 @5266f24da).
pub const GenInp = extern struct {
    type: GenProcessType,
    code0: i32,
    embd: ?[*]f32,
    top_k: i32,
    top_p: f32,
    seed: u32,
    temp: f32,
    codes: ?[*]i32,
    n_codes: usize,
    feats: ?[*]const f32,
    n_feats: usize,
    state_data: ?[*]const u8,
    state_size: usize,
};

/// Ports `mtmd_gen_out` (mtmd.h:413 @5266f24da).
pub const GenOut = extern struct {
    codes: ?[*]const i32,
    n_codes: usize,
    feats: ?[*]const f32,
    n_feats: usize,
    embd: ?[*]const f32,
    is_eos: bool,
    audio: ?[*]const f32,
    n_samples: usize,
    state_data: ?[*]const u8,
    state_size: usize,
};

// -----------------------------------------------------------------------------
// helpers

/// A `std::string` that hands out `c_str()`: the text and a NUL past it.
pub const Str = struct {
    buf: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Str) void {
        self.buf.deinit(gpa);
    }

    pub fn slice(self: *const Str) []const u8 {
        return if (self.buf.items.len == 0) "" else self.buf.items[0 .. self.buf.items.len - 1];
    }

    pub fn cStr(self: *const Str) [*:0]const u8 {
        return if (self.buf.items.len == 0) "" else @ptrCast(self.buf.items.ptr);
    }

    pub fn set(self: *Str, s: []const u8) !void {
        self.buf.clearRetainingCapacity();
        if (s.len == 0) return;
        try self.buf.ensureTotalCapacity(gpa, s.len + 1);
        self.buf.appendSliceAssumeCapacity(s);
        self.buf.appendAssumeCapacity(0);
    }

    pub fn clone(self: *const Str) !Str {
        var out: Str = .{};
        try out.set(self.slice());
        return out;
    }
};

/// `std::vector::data()`: null for a vector that never allocated, as
/// libc++ returns it.
fn dataPtr(comptime T: type, list: *const std.ArrayList(T)) ?[*]T {
    return if (list.capacity == 0) null else list.items.ptr;
}

/// `std::vector::resize`: new elements are zero, old ones kept.
fn resizeZero(comptime T: type, list: *std.ArrayList(T), n: usize) !void {
    const old = list.items.len;
    try list.resize(gpa, n);
    if (n > old) @memset(list.items[old..], std.mem.zeroes(T));
}

/// `e.what()` of an error caught where the C++ catches `std::exception`.
fn whatOf(e: anyerror) [*:0]const u8 {
    return if (e == error.OutOfMemory) "std::bad_alloc" else clip.what().ptr;
}

const Error = clip.Error;

/// `size_t += int`: the int converts to `size_t` first.
fn addInt(a: usize, b: c_int) usize {
    return a +% @as(usize, @bitCast(@as(isize, b)));
}

// -----------------------------------------------------------------------------
// serialization

/// Ports `mtmd_serialization` (mtmd.cpp:32 @5266f24da): a byte buffer
/// written and read in native byte order.
const Ser = struct {
    version: u64 = serialization_version,
    data: std.ArrayList(u8) = .empty,
    read_pos: usize = 0, // cursor used when reading

    /// for writing
    fn initWrite(version: u64) !Ser {
        var self: Ser = .{ .version = version };
        try self.write(u64, version);
        return self;
    }

    /// for reading
    fn initRead(version: u64, buf: []const u8) Error!Ser {
        var self: Ser = .{};
        errdefer self.deinit();
        try self.data.appendSlice(gpa, buf);
        const ver_in = try self.read(u64);
        if (ver_in != version) return clip.throwStr("version mismatch");
        self.version = ver_in;
        return self;
    }

    fn deinit(self: *Ser) void {
        self.data.deinit(gpa);
    }

    fn write(self: *Ser, comptime T: type, value: T) !void {
        if (T == bool) return self.write(u8, if (value) 1 else 0);
        try self.data.appendSlice(gpa, std.mem.asBytes(&value));
    }

    fn read(self: *Ser, comptime T: type) Error!T {
        if (T == bool) return (try self.read(u8)) != 0;
        if (self.read_pos + @sizeOf(T) > self.data.items.len) return clip.throwStr("read OOB");
        var value: T = undefined;
        @memcpy(std.mem.asBytes(&value), self.data.items[self.read_pos..][0..@sizeOf(T)]);
        self.read_pos += @sizeOf(T);
        return value;
    }

    fn writeString(self: *Ser, s: []const u8) !void {
        try self.write(u64, s.len);
        try self.data.appendSlice(gpa, s);
    }

    fn readString(self: *Ser, out: *Str) Error!void {
        const len = try self.read(u64);
        if (self.read_pos +% len > self.data.items.len) return clip.throwStr("read_string OOB");
        try out.set(self.data.items[self.read_pos..][0..@intCast(len)]);
        self.read_pos += @intCast(len);
    }

    fn remaining(self: *const Ser) usize {
        return self.data.items.len - self.read_pos;
    }
};

/// Ports `serialize` (mtmd.cpp:103 @5266f24da), a member of `clip_image_f32`. `buf` is
/// intentionally not serialized: a loaded image is always a placeholder.
fn serializeImage(img: *const impl.ImageF32, ser: *Ser) !void {
    try ser.write(bool, img.add_viewsep);
    try ser.write(bool, img.add_newline);
    try ser.write(i32, img.lead_pad);
    try ser.write(i32, img.nx_);
    try ser.write(i32, img.ny_);
}

/// Ports `deserialize` (mtmd.cpp:112 @5266f24da), a member of `clip_image_f32`.
fn deserializeImage(img: *impl.ImageF32, ser: *Ser) Error!void {
    img.add_viewsep = try ser.read(bool);
    img.add_newline = try ser.read(bool);
    img.lead_pad = try ser.read(i32);
    img.nx_ = try ser.read(i32);
    img.ny_ = try ser.read(i32);
    img.buf.clearRetainingCapacity(); // always a placeholder after loading
}

/// Ports `serialize` (mtmd.cpp:121 @5266f24da), a member of `clip_image_f32_batch`.
fn serializeBatch(b: *const impl.ImageF32Batch, ser: *Ser) !void {
    try ser.write(bool, b.is_audio);
    try ser.write(u64, b.entries.items.len);
    for (b.entries.items) |*e| try serializeImage(e, ser);
}

/// Ports `deserialize` (mtmd.cpp:129 @5266f24da), a member of `clip_image_f32_batch`.
fn deserializeBatch(b: *impl.ImageF32Batch, ser: *Ser) Error!void {
    b.is_audio = try ser.read(bool);
    const n = try ser.read(u64);
    const min_entry_bytes = @sizeOf(u8) * 2 + @sizeOf(i32) * 2;
    if (n > ser.remaining() / min_entry_bytes) return clip.throwStr("entries count exceeds buffer size");
    for (b.entries.items) |*e| e.deinit();
    b.entries.clearRetainingCapacity();
    try b.entries.ensureTotalCapacity(gpa, @intCast(n));
    var i: u64 = 0;
    while (i < n) : (i += 1) {
        var entry: impl.ImageF32 = .{};
        try deserializeImage(&entry, ser);
        b.entries.appendAssumeCapacity(entry);
    }
}

// -----------------------------------------------------------------------------
// bitmaps, tokens and chunks

/// Ports `mtmd_bitmap` (mtmd.cpp:150 @5266f24da). An image is RGB
/// interleaved, `nx * ny * 3` bytes; audio is `nx` F32 samples and `ny = 1`.
/// No data marks a placeholder.
pub const Bitmap = struct {
    nx: u32 = 0,
    ny: u32 = 0,
    id: Str = .{}, // optional user-defined id, for ex: can be set to image hash, useful for KV cache tracking
    is_audio: bool = false, // true if the bitmap is audio
    mergeable: bool = false, // [QWEN_VIDEO] set only on frames of the same video

    // lazy-loaded bitmap
    lazy_callback: LazyCallback = null,
    lazy_user_data: ?*anyopaque = null,

    data: std.ArrayList(u8) = .empty,

    /// The image constructor `mtmd_bitmap` (mtmd.cpp:161 @5266f24da).
    fn createImage(data: ?[*]const u8, nx: u32, ny: u32) !*Bitmap {
        const self = try gpa.create(Bitmap);
        self.* = .{ .nx = nx, .ny = ny, .is_audio = false };
        if (data) |d| try self.data.appendSlice(gpa, d[0 .. @as(usize, nx) * ny * 3]);
        return self;
    }

    /// The audio constructor `mtmd_bitmap` (mtmd.cpp:170 @5266f24da).
    fn createAudio(data: ?[*]const u8, n_samples: u32) !*Bitmap {
        const self = try gpa.create(Bitmap);
        self.* = .{ .nx = n_samples, .ny = 1, .is_audio = true };
        if (data) |d| try self.data.appendSlice(gpa, d[0 .. @as(usize, n_samples) * @sizeOf(f32)]);
        return self;
    }

    fn destroy(self: *Bitmap) void {
        self.id.deinit();
        self.data.deinit(gpa);
        gpa.destroy(self);
    }

    pub fn isPlaceholder(self: *const Bitmap) bool {
        return self.data.items.len == 0;
    }

    /// `can_merge_with` (mtmd.cpp:191 @5266f24da): [QWEN_VIDEO] can
    /// (temporal) merge if both are images with same size.
    fn canMergeWith(self: *const Bitmap, other: *const Bitmap) bool {
        return self.mergeable and other.mergeable and
            !self.is_audio and !other.is_audio and
            self.nx == other.nx and self.ny == other.ny;
    }
};

/// Ports `mtmd_pos_type` (mtmd.cpp:203 @5266f24da): position indexing for
/// the decoder model.
pub const PosType = enum(u32) {
    normal, // number of positions equals to number of tokens
    mrope, // qwen-vl mrope style, each image takes max(t,h,w) position indexes
    hunyuanvl, // HunyuanVL mrope + BOI/EOI/newline layout with XD-RoPE dim-3
    _,
};
const pos_type_count: u32 = 3;

/// Ports `mtmd_image_tokens` (mtmd.cpp:210 @5266f24da).
pub const ImageTokens = struct {
    nx: u32 = 0, // number of tokens in x direction
    ny: u32 = 0, // number of tokens in y direction
    pos: PosType = .normal,
    image_idx: u32 = 0, // 0-based position of this image among image chunks in the prompt(used by pos == MTMD_POS_TYPE_HUNYUANVL)
    n_temporal_merge: u32 = 1, // for qwen-vl style temporal merge
    batch_f32: impl.ImageF32Batch = .{}, // preprocessed image patches
    id: Str = .{}, // optional user-defined ID, useful for KV cache tracking

    fn destroy(self: *ImageTokens) void {
        self.batch_f32.deinit();
        self.id.deinit();
        gpa.destroy(self);
    }

    /// `n_tokens` (mtmd.cpp:216 @5266f24da).
    pub fn nTokens(self: *const ImageTokens) u32 {
        if (self.pos == .hunyuanvl) {
            // [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
            return (self.nx +% 1) *% self.ny +% 2;
        }
        var nz: u32 = @truncate(self.batch_f32.entries.items.len);
        if (self.n_temporal_merge > 1) {
            // [QWEN_VIDEO] this logic is quite ugly, it's mostly to make qwen-vl temporal merge work, can be improved in the future
            if (nz % self.n_temporal_merge != 0) {
                nz = nz / self.n_temporal_merge + 1;
            } else {
                nz = nz / self.n_temporal_merge;
            }
        }
        return self.nx *% self.ny *% nz;
    }

    /// true if one of entries in batch_f32 is a placeholder
    fn isPlaceholder(self: *const ImageTokens) bool {
        for (self.batch_f32.entries.items) |*e| if (e.isPlaceholder()) return true;
        return false;
    }

    fn canBatchWith(self: *const ImageTokens, other: *const ImageTokens) bool {
        return self.nx == other.nx and self.ny == other.ny and self.pos == other.pos;
    }

    fn cloneAlloc(self: *const ImageTokens) !*ImageTokens {
        const out = try gpa.create(ImageTokens);
        out.* = .{
            .nx = self.nx,
            .ny = self.ny,
            .pos = self.pos,
            .image_idx = self.image_idx,
            .n_temporal_merge = self.n_temporal_merge,
        };
        out.batch_f32 = try self.batch_f32.clone();
        out.id = try self.id.clone();
        return out;
    }

    /// `serialize` (mtmd.cpp:262 @5266f24da).
    fn serialize(self: *const ImageTokens, ser: *Ser) !void {
        try ser.write(u32, self.nx);
        try ser.write(u32, self.ny);
        try ser.write(u32, @backingInt(self.pos));
        try ser.write(u32, self.image_idx);
        try ser.write(u32, self.n_temporal_merge);
        try ser.writeString(self.id.slice());
        try serializeBatch(&self.batch_f32, ser);
    }

    /// `deserialize` (mtmd.cpp:272 @5266f24da).
    fn deserialize(self: *ImageTokens, ser: *Ser) Error!void {
        self.nx = try ser.read(u32);
        self.ny = try ser.read(u32);
        const pos_raw = try ser.read(u32);
        if (pos_raw >= pos_type_count) return clip.throwStr("invalid pos type");
        self.pos = @fromBackingInt(@intCast(pos_raw));
        self.image_idx = try ser.read(u32);
        self.n_temporal_merge = try ser.read(u32);
        try ser.readString(&self.id);
        try deserializeBatch(&self.batch_f32, ser);
    }
};

/// Ports `mtmd_audio_tokens` (mtmd.cpp:288 @5266f24da).
pub const AudioTokens = struct {
    n_tokens: u32 = 0, // number of tokens
    batch_f32: impl.ImageF32Batch = .{}, // preprocessed image patches
    id: Str = .{}, // optional user-defined ID, useful for KV cache tracking

    fn destroy(self: *AudioTokens) void {
        self.batch_f32.deinit();
        self.id.deinit();
        gpa.destroy(self);
    }

    /// true if one of entries in batch_f32 is a placeholder
    fn isPlaceholder(self: *const AudioTokens) bool {
        for (self.batch_f32.entries.items) |*e| if (e.isPlaceholder()) return true;
        return false;
    }

    fn cloneAlloc(self: *const AudioTokens) !*AudioTokens {
        const out = try gpa.create(AudioTokens);
        out.* = .{ .n_tokens = self.n_tokens };
        out.batch_f32 = try self.batch_f32.clone();
        out.id = try self.id.clone();
        return out;
    }

    /// `serialize` (mtmd.cpp:311 @5266f24da).
    fn serialize(self: *const AudioTokens, ser: *Ser) !void {
        try ser.write(u32, self.n_tokens);
        try ser.writeString(self.id.slice());
        try serializeBatch(&self.batch_f32, ser);
    }

    /// `deserialize` (mtmd.cpp:317 @5266f24da).
    fn deserialize(self: *AudioTokens, ser: *Ser) Error!void {
        self.n_tokens = try ser.read(u32);
        try ser.readString(&self.id);
        try deserializeBatch(&self.batch_f32, ser);
    }
};

/// Ports `mtmd_input_chunk` (mtmd.cpp:325 @5266f24da).
pub const InputChunk = struct {
    type: ChunkType = .text,
    tokens_text: std.ArrayList(Token) = .empty,
    tokens_image: ?*ImageTokens = null,
    tokens_audio: ?*AudioTokens = null,

    fn deinit(self: *InputChunk) void {
        self.tokens_text.deinit(gpa);
        if (self.tokens_image) |t| t.destroy();
        if (self.tokens_audio) |t| t.destroy();
        self.* = .{};
    }

    /// `can_batch_with` (mtmd.cpp:331 @5266f24da).
    fn canBatchWith(self: *const InputChunk, other: *const InputChunk) bool {
        if (self.type != other.type) return false;
        if (self.tokens_image != null and other.tokens_image != null) {
            return self.tokens_image.?.canBatchWith(other.tokens_image.?);
        }
        // TODO: allow batching audio chunks of the same size
        return false;
    }

    /// `is_placeholder` (mtmd.cpp:345 @5266f24da).
    fn isPlaceholder(self: *const InputChunk) bool {
        if (self.type == .image) {
            return self.tokens_image != null and self.tokens_image.?.isPlaceholder();
        } else if (self.type == .audio) {
            return self.tokens_audio != null and self.tokens_audio.?.isPlaceholder();
        }
        return false;
    }

    /// `serialize` (mtmd.cpp:354 @5266f24da).
    fn serialize(self: *const InputChunk, ser: *Ser) !void {
        try ser.write(u32, @bitCast(@backingInt(self.type)));

        try ser.write(u64, self.tokens_text.items.len);
        for (self.tokens_text.items) |tok| try ser.write(i32, tok);

        try ser.write(bool, self.tokens_image != null);
        if (self.tokens_image) |t| try t.serialize(ser);

        try ser.write(bool, self.tokens_audio != null);
        if (self.tokens_audio) |t| try t.serialize(ser);
    }

    /// `deserialize` (mtmd.cpp:373 @5266f24da).
    fn deserialize(self: *InputChunk, ser: *Ser) Error!void {
        const type_raw = try ser.read(u32);
        if (type_raw >= chunk_type_count) return clip.throwStr("invalid chunk type");
        self.type = @fromBackingInt(@intCast(@as(c_int, @intCast(type_raw))));

        const n_tokens_text = try ser.read(u64);
        // reject before resize() so a tiny corrupted/malicious buffer can't force a huge allocation
        if (n_tokens_text > ser.remaining() / @sizeOf(i32)) return clip.throwStr("tokens_text length exceeds buffer size");
        try resizeZero(Token, &self.tokens_text, @intCast(n_tokens_text));
        for (self.tokens_text.items) |*t| t.* = try ser.read(i32);

        if (try ser.read(bool)) {
            const t = try gpa.create(ImageTokens);
            t.* = .{};
            if (self.tokens_image) |old| old.destroy();
            self.tokens_image = t;
            try t.deserialize(ser);
        } else if (self.tokens_image) |old| {
            old.destroy();
            self.tokens_image = null;
        }

        if (try ser.read(bool)) {
            const t = try gpa.create(AudioTokens);
            t.* = .{};
            if (self.tokens_audio) |old| old.destroy();
            self.tokens_audio = t;
            try t.deserialize(ser);
        } else if (self.tokens_audio) |old| {
            old.destroy();
            self.tokens_audio = null;
        }

        // catch buffers where the declared type doesn't match which payload is actually present,
        // so a mismatched chunk can't slip through and null-deref/abort later in an accessor
        if (self.type == .image and self.tokens_image == null) return clip.throwStr("type is IMAGE but tokens_image is missing");
        if (self.type == .audio and self.tokens_audio == null) return clip.throwStr("type is AUDIO but tokens_audio is missing");
    }
};

/// Ports `mtmd_input_chunks` (mtmd.cpp:415 @5266f24da).
pub const InputChunks = struct {
    entries: std.ArrayList(InputChunk) = .empty,

    fn clear(self: *InputChunks) void {
        for (self.entries.items) |*e| e.deinit();
        self.entries.clearRetainingCapacity();
    }

    fn deinit(self: *InputChunks) void {
        self.clear();
        self.entries.deinit(gpa);
    }
};

/// Ports `mtmd_batch` (mtmd.cpp:419 @5266f24da).
pub const Batch = struct {
    ctx: *Context,
    entries: std.ArrayList(*const InputChunk) = .empty,
    output_embd: std.ArrayList(f32) = .empty, // aggregated output embedding for the whole batch

    fn nTokens(self: *const Batch) i32 {
        var n: i32 = 0;
        for (self.entries.items) |chunk| {
            n = @truncate(@as(isize, n) +% @as(isize, @bitCast(mtmd_input_chunk_get_n_tokens(chunk))));
        }
        return n;
    }
};

/// Ports `mtmd_slice_tmpl` (mtmd.cpp:435 @5266f24da): where some llava-uhd
/// models put special tokens around image embeddings.
const SliceTmpl = enum { none, minicpmv_2_5, minicpmv_2_6, llama4, idefics3, lfm2, step3vl };

/// Ports `mtmd_default_marker` (mtmd.cpp:445 @5266f24da).
pub fn mtmd_default_marker() callconv(.c) [*:0]const u8 {
    return "<__media__>";
}

/// Ports `mtmd_get_clip_flash_attn_type` (mtmd.cpp:449 @5266f24da).
fn clipFlashAttnType(t: c.enum_llama_flash_attn_type) impl.FlashAttnType {
    return switch (t) {
        c.LLAMA_FLASH_ATTN_TYPE_AUTO => .auto,
        c.LLAMA_FLASH_ATTN_TYPE_DISABLED => .disabled,
        c.LLAMA_FLASH_ATTN_TYPE_ENABLED => .enabled,
        else => .auto,
    };
}

/// Ports `mtmd_context_params_default` (mtmd.cpp:458 @5266f24da).
pub fn mtmd_context_params_default() callconv(.c) ContextParams {
    return .{
        .use_gpu = true,
        .device = null,
        .print_timings = true,
        .n_threads = 4,
        .image_marker = null,
        .media_marker = mtmd_default_marker(),
        .flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_AUTO,
        .warmup = true,
        .image_min_tokens = -1,
        .image_max_tokens = -1,
        .cb_eval = null,
        .cb_eval_user_data = null,
        .batch_max_tokens = 1024,
        .progress_callback = null,
        .progress_callback_user_data = null,
    };
}

// -----------------------------------------------------------------------------
// the context

/// Ports `mtmd_context` (mtmd.cpp:479 @5266f24da).
pub const Context = struct {
    ctx_v: ?*clip.Ctx = null, // vision
    ctx_a: ?*clip.Ctx = null, // audio
    out_embd: std.ArrayList(f32) = .empty, // image embedding vector

    // generation context
    ctx_gen_a: ?*clip.Ctx = null, // audio
    gen_out_codes: std.ArrayList(i32) = .empty, // this frame's 16 sampled codes (GEN_CODE)
    gen_out_feats: std.ArrayList(f32) = .empty, // this frame's continuous features, if any (GEN_CODE)
    gen_out_embd: std.ArrayList(f32) = .empty, // next-step hidden state fed back to backbone (GEN_CODE)
    gen_out_audio: std.ArrayList(f32) = .empty, // decoded PCM samples for the current frame (GEN_WAV)
    gen_out_state: std.ArrayList(u8) = .empty, // state to feed into the next GEN_WAV call

    print_timings: bool = false,
    n_threads: c_int = 0,
    media_marker: Str = .{},
    n_embd_text: c_int = -1, // -1 means llm context not provided, skip checking this
    vocab: ?*const c.struct_llama_vocab = null, // can be nullptr if text_model is not provided
    // The C++ leaves this indeterminate without a text model; `normal` is
    // what fresh heap memory gives.
    pos_type: PosType = .normal,

    // these are not token, but strings used to mark the beginning and end of image/audio embeddings
    img_beg: []const u8 = "",
    img_end: []const u8 = "",
    aud_beg: []const u8 = "",
    aud_end: []const u8 = "",

    // for llava-uhd style models, we need special tokens in-between slices
    // minicpmv calls them "slices", llama 4 calls them "tiles"
    slice_tmpl: SliceTmpl = .none,
    tok_ov_img_start: std.ArrayList(Token) = .empty, // overview image
    tok_ov_img_end: std.ArrayList(Token) = .empty, // overview image
    tok_slices_start: std.ArrayList(Token) = .empty, // start of all slices
    tok_slices_end: std.ArrayList(Token) = .empty, // end of all slices
    tok_sli_img_start: std.ArrayList(Token) = .empty, // single slice start
    tok_sli_img_end: std.ArrayList(Token) = .empty, // single slice end
    tok_sli_img_mid: std.ArrayList(Token) = .empty, // between 2 slices
    tok_row_end: std.ArrayList(Token) = .empty, // end of row
    tok_row_end_trail: bool = false,
    ov_img_first: bool = false,

    // string template for slice image delimiters with row/col (idefics3)
    sli_img_start_tmpl: ?[*:0]const u8 = null,

    audio_preproc: ?audio.Preprocessor = null,
    image_preproc: ?image.Preprocessor = null,

    // batching
    batch_max_tokens: i32 = 0,

    /// The constructor `mtmd_context` (mtmd.cpp:530 @5266f24da). A throw leaks the clip
    /// contexts as the C++'s does -- its destructor never runs for an
    /// object whose constructor threw, and the pointers are raw -- so that
    /// freeing them logs nothing upstream does not.
    fn create(mmproj_fname: [*:0]const u8, text_model: ?*const c.struct_llama_model, ctx_params: *const ContextParams, no_alloc: bool) Error!*Context {
        const self = try gpa.create(Context);
        self.* = .{};
        errdefer self.deinitMembers();
        errdefer gpa.destroy(self);

        self.print_timings = ctx_params.print_timings;
        self.n_threads = ctx_params.n_threads;
        // `std::string(nullptr)` is undefined; an absent marker reads as empty.
        try self.media_marker.set(if (ctx_params.media_marker) |m| std.mem.span(m) else "");
        self.n_embd_text = if (text_model != null) lapi.model.llama_model_n_embd_inp(text_model) else -1;
        self.vocab = if (text_model != null) lapi.model.llama_model_get_vocab(text_model) else null;
        self.batch_max_tokens = ctx_params.batch_max_tokens;

        if (ctx_params.image_marker != null) {
            return clip.throwStr("custom image_marker is not supported anymore, use media_marker instead");
        }

        if (self.media_marker.slice().len == 0) {
            return clip.throwStr("media_marker must not be empty");
        }

        if (text_model != null) {
            const decoder_rope_type = lapi.model.llama_model_rope_type(text_model);
            switch (decoder_rope_type) {
                c.LLAMA_ROPE_TYPE_NONE, c.LLAMA_ROPE_TYPE_NORM, c.LLAMA_ROPE_TYPE_NEOX => self.pos_type = .normal,
                c.LLAMA_ROPE_TYPE_MROPE, c.LLAMA_ROPE_TYPE_IMROPE => self.pos_type = .mrope,
                else => return clip.throwErr("unsupported decoder rope type: %d\n", .{@as(c_int, decoder_rope_type)}),
            }
        }

        const ctx_clip_params: clip.ContextParams = .{
            .use_gpu = ctx_params.use_gpu,
            .device = ctx_params.device,
            .flash_attn_type = clipFlashAttnType(ctx_params.flash_attn_type),
            .image_min_tokens = ctx_params.image_min_tokens,
            .image_max_tokens = ctx_params.image_max_tokens,
            .warmup = ctx_params.warmup,
            .cb_eval = ctx_params.cb_eval,
            .cb_eval_user_data = ctx_params.cb_eval_user_data,
            .no_alloc = no_alloc,
            .progress_callback = ctx_params.progress_callback,
            .progress_callback_user_data = ctx_params.progress_callback_user_data,
        };

        const res = clip.init(std.mem.span(mmproj_fname), ctx_clip_params);
        self.ctx_v = res.ctx_v;
        self.ctx_a = res.ctx_a;
        self.ctx_gen_a = res.ctx_gen_a;
        if (self.ctx_v == null and self.ctx_a == null) {
            return clip.throwErr("Failed to load CLIP model from %s\n", .{mmproj_fname});
        }

        // if both vision and audio mmproj are present, we need to validate their n_embd
        if (self.ctx_v != null and self.ctx_a != null) {
            const n_embd_v = clip.nMmprojEmbd(self.ctx_v.?);
            const n_embd_a = clip.nMmprojEmbd(self.ctx_a.?);
            if (n_embd_v != n_embd_a) {
                return clip.throwErr("mismatch between vision and audio mmproj (n_embd_v = %d, n_embd_a = %d)\n", .{ n_embd_v, n_embd_a });
            }
        }

        // since we already validate n_embd of vision and audio mmproj,
        // we can safely assume that they are the same
        const n_embd_clip = clip.nMmprojEmbd(self.ctx_v orelse self.ctx_a.?);
        if (self.n_embd_text > 0 and self.n_embd_text != n_embd_clip) {
            return clip.throwErr("mismatch between text model (n_embd = %d) and mmproj (n_embd = %d)\n" ++
                "hint: you may be using wrong mmproj\n", .{ self.n_embd_text, n_embd_clip });
        }
        if (self.ctx_gen_a) |g| {
            const n_embd_gen = clip.nMmprojEmbd(g);
            if (self.n_embd_text > 0 and self.n_embd_text != n_embd_gen) {
                return clip.throwErr("mismatch between text model (n_embd = %d) and gen-audio mmproj (n_embd = %d)\n" ++
                    "hint: you may be using wrong mmproj\n", .{ self.n_embd_text, n_embd_gen });
            }
        }
        if (self.ctx_v != null) try self.initVision();
        if (self.ctx_a != null) try self.initAudio();
        return self;
    }

    fn setTokens(list: *std.ArrayList(Token), toks: []const Token) !void {
        list.clearRetainingCapacity();
        try list.appendSlice(gpa, toks);
    }

    /// Ports `init_vision` (mtmd.cpp:627 @5266f24da).
    fn initVision(self: *Context) Error!void {
        impl.ggmlAssert(self.ctx_v != null, "mtmd.cpp", 628, "ctx_v != nullptr");
        self.image_preproc = null;

        const ctx_v = self.ctx_v.?;
        const proj = clip.getProjectorType(ctx_v);
        const hparams = clip.getHparams(ctx_v);
        var kind: image.Kind = undefined;

        switch (proj) {
            .mlp, .mlp_norm, .ldp, .ldpv2, .cogvlm, .janus_pro, .glm_edge => {
                const has_pinpoints = hparams.image_res_candidates.items.len != 0;
                kind = if (has_pinpoints) .llava_uhd else .fixed_size;
            },
            .minicpmv => {
                const minicpmv_version = hparams.minicpmv_version;
                if (minicpmv_version == 2) {
                    // minicpmv 2.5 format:
                    // <image> (overview) </image><slice><image> (slice) </image><image> (slice) </image>\n ... </slice>
                    self.slice_tmpl = .minicpmv_2_5;
                    try setTokens(&self.tok_ov_img_start, &.{self.lookupToken("<image>")});
                    try setTokens(&self.tok_ov_img_end, &.{self.lookupToken("</image>")});
                    try setTokens(&self.tok_slices_start, &.{self.lookupToken("<slice>")});
                    try setTokens(&self.tok_slices_end, &.{self.lookupToken("</slice>")});
                    try setTokens(&self.tok_sli_img_start, self.tok_ov_img_start.items);
                    try setTokens(&self.tok_sli_img_end, self.tok_ov_img_end.items);
                    try setTokens(&self.tok_row_end, &.{self.lookupToken("\n")});
                    self.tok_row_end_trail = false; // no trailing end-of-row token
                    self.ov_img_first = true;
                } else if (minicpmv_version == 3 or minicpmv_version == 4 or minicpmv_version == 5 or minicpmv_version == 6 or minicpmv_version == 100045) {
                    // minicpmv 2.6 format:
                    // <image> (overview) </image><slice> (slice) </slice><slice> (slice) </slice>\n ...
                    self.slice_tmpl = .minicpmv_2_6;
                    try setTokens(&self.tok_ov_img_start, &.{self.lookupToken("<image>")});
                    try setTokens(&self.tok_ov_img_end, &.{self.lookupToken("</image>")});
                    try setTokens(&self.tok_sli_img_start, &.{self.lookupToken("<slice>")});
                    try setTokens(&self.tok_sli_img_end, &.{self.lookupToken("</slice>")});
                    try setTokens(&self.tok_row_end, &.{self.lookupToken("\n")});
                    self.tok_row_end_trail = false; // no trailing end-of-row token
                    self.ov_img_first = true;
                } else if (minicpmv_version != 0) {
                    return clip.throwErr("unsupported minicpmv version: %d\n", .{minicpmv_version});
                }
                kind = .llava_uhd;
            },
            .minicpmv4_6 => {
                self.slice_tmpl = .minicpmv_2_6;
                try setTokens(&self.tok_ov_img_start, &.{self.lookupToken("<image>")});
                try setTokens(&self.tok_ov_img_end, &.{self.lookupToken("</image>")});
                try setTokens(&self.tok_sli_img_start, &.{self.lookupToken("<slice>")});
                try setTokens(&self.tok_sli_img_end, &.{self.lookupToken("</slice>")});
                try setTokens(&self.tok_row_end, &.{self.lookupToken("\n")});
                self.tok_row_end_trail = false; // no trailing end-of-row token
                self.ov_img_first = true;
                kind = .minicpmv;
            },
            .qwen2vl, .qwen25vl, .qwen3vl, .mimovl => {
                // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                self.img_beg = "<|vision_start|>";
                self.img_end = "<|vision_end|>";
                kind = .dyn_size;
            },
            .minimax_m3 => {
                // ]<]start of image[>[ ... (image embeddings) ... ]<]end of image[>[
                self.img_beg = "]<]start of image[>[";
                self.img_end = "]<]end of image[>[";
                kind = .dyn_size;
            },
            .muse_glimmer => {
                self.img_beg = "<|image_start|>";
                self.img_end = "<|image_end|>";
                kind = .muse_glimmer;
            },
            .youtuvl => {
                // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                self.img_beg = "<|vision_start|>";
                self.img_end = "<|vision_end|>";
                kind = .youtuvl;
            },
            .yasa2 => {
                self.img_beg = "<image>";
                self.img_end = "</image>";
                // Currently only supports single-tile preprocessing (see mtmd.cpp:728).
                kind = .fixed_size;
            },
            .gemma3, .gemma3nv => {
                // <start_of_image> ... (image embeddings) ... <end_of_image>
                self.img_beg = "<start_of_image>";
                self.img_end = "<end_of_image>";
                kind = .fixed_size;
            },
            .idefics3 => {
                self.slice_tmpl = .idefics3;
                try setTokens(&self.tok_ov_img_start, &.{ self.lookupToken("\n\n"), self.lookupToken("<fake_token_around_image>"), self.lookupToken("<global-img>") });
                try setTokens(&self.tok_ov_img_end, &.{self.lookupToken("<fake_token_around_image>")});
                try setTokens(&self.tok_row_end, &.{self.lookupToken("\n")});
                self.sli_img_start_tmpl = "<fake_token_around_image><row_%d_col_%d>";
                kind = .idefics3;
            },
            .pixtral => {
                self.img_end = "[IMG_END]";
                kind = .dyn_size;
            },
            .phi4 => {
                // Phi-4 uses media marker insertion only. Keep image boundary text empty.
                kind = .dyn_size;
            },
            .llama4 => {
                self.img_beg = "<|image_start|>";
                self.img_end = "<|image_end|>";
                impl.logWrn("%s: llama 4 vision is known to have degraded quality:\n" ++
                    "    https://github.com/ggml-org/llama.cpp/pull/13282\n", .{"init_vision"});
                kind = .llava_uhd;
                self.ov_img_first = false;
            },
            .step3vl => {
                // Step3 format:
                //   <patch_start> (patch) <patch_end> [<patch_newline>]
                //   ... (all patch rows)
                //   <im_start> (overview) <im_end>
                self.slice_tmpl = .step3vl;
                try setTokens(&self.tok_ov_img_start, &.{self.lookupToken("<im_start>")});
                try setTokens(&self.tok_ov_img_end, &.{self.lookupToken("<im_end>")});
                try setTokens(&self.tok_sli_img_start, &.{self.lookupToken("<patch_start>")});
                try setTokens(&self.tok_sli_img_end, &.{self.lookupToken("<patch_end>")});
                try setTokens(&self.tok_row_end, &.{self.lookupToken("<patch_newline>")});
                self.tok_row_end_trail = false;
                self.ov_img_first = false; // patches first, overview last
                kind = .step3vl;
            },
            .internvl => {
                // <img> ... (image embeddings) ... </img>
                self.img_beg = "<img>";
                self.img_end = "</img>";
                kind = .internvl;
                self.ov_img_first = false;
            },
            .kimivl => {
                // <|media_start|> ... (image embeddings) ... <|media_end|>
                self.img_beg = "<|media_start|>";
                self.img_end = "<|media_end|>";
                kind = .dyn_size;
            },
            .kimik25 => {
                // GLM-5.2-V reuses the Kimi-K2.5 vision encoder and projector, but marks
                // images with its own tokens, so decide based on the text model vocab
                if (self.lookupToken("<|begin_of_image|>") != token_null) {
                    self.img_beg = "<|begin_of_image|>";
                    self.img_end = "<|end_of_image|>";
                } else {
                    self.img_beg = "<|media_begin|>";
                    self.img_end = "<|media_end|>";
                }
                kind = .dyn_size;
            },
            .lightonocr => {
                // <|im_start|> ... (image embeddings) ... <|im_end|>
                self.img_beg = "<|im_start|>";
                self.img_end = "<|im_end|>";
                kind = .longest_edge;
            },
            .deepseek4v => {
                // no vocab tokens are added; the start/end/newline markers are learned embeddings emitted by the encoder
                kind = .deepseek4v;
            },
            .dots_ocr, .dots3note_v => {
                // <|img|> ... (image embeddings) ... <|endofimg|>
                self.img_beg = "<|img|>";
                self.img_end = "<|endofimg|>";
                kind = .dyn_size;
            },
            .nemotron_v2_vl => kind = .fixed_size,
            .lfm2 => {
                // multi-tile:
                //   <|image_start|>
                //     <|img_row_1_col_1|> (tile) <|img_row_1_col_2|> (tile) ...
                //     <|img_thumbnail|> (thumbnail)
                //   <|image_end|>
                // single-tile:
                //   <|image_start|> (image) <|image_end|>
                self.img_beg = "<|image_start|>";
                self.img_end = "<|image_end|>";
                self.slice_tmpl = .lfm2;
                self.sli_img_start_tmpl = "<|img_row_%d_col_%d|>";
                try setTokens(&self.tok_ov_img_start, &.{self.lookupToken("<|img_thumbnail|>")});
                self.ov_img_first = false;
                kind = .lfm2;
            },
            .glm4v => {
                // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
                self.img_beg = "<|begin_of_image|>";
                self.img_end = "<|end_of_image|>";
                kind = .dyn_size;
            },
            .paddleocr => {
                // <|IMAGE_START|> ... (image embeddings) ... <|IMAGE_END|>
                self.img_beg = "<|IMAGE_START|>";
                self.img_end = "<|IMAGE_END|>";
                kind = .dyn_size;
            },
            .gemma4v, .gemma4uv => {
                // <|image> ... (image embeddings) ... <image|>
                self.img_beg = "<|image>";
                self.img_end = "<image|>";
                kind = .dyn_size;
            },
            .deepseekocr, .deepseekocr2 => {
                self.img_end = "\n"; // prevent empty batch on llama-server
                kind = .deepseekocr;
                self.ov_img_first = false;
            },
            .hunyuanvl => {
                // note: these use fullwidth ｜ (U+FF5C) and ▁ (U+2581) to match the tokenizer vocabulary
                self.img_beg = "<｜hy_place▁holder▁no▁100｜>";
                self.img_end = "<｜hy_place▁holder▁no▁101｜>";
                kind = .dyn_size;
            },
            .exaone4_5 => {
                // <vision> ... (image embeddings) ... </vision>
                self.img_beg = "<vision>";
                self.img_end = "</vision>";
                kind = .dyn_size;
            },
            .granite4_vision => {
                // ... (image embeddings) \n ...
                self.img_beg = "";
                self.img_end = "\n";
                kind = .granite;
            },
            else => return clip.throwErr("%s: unexpected vision projector type %d\n", .{ "init_vision", @as(c_int, @backingInt(proj)) }),
        }
        self.image_preproc = image.Preprocessor.init(kind, hparams, proj);
    }

    /// Ports `init_audio` (mtmd.cpp:920 @5266f24da).
    fn initAudio(self: *Context) Error!void {
        impl.ggmlAssert(self.ctx_a != null, "mtmd.cpp", 921, "ctx_a != nullptr");
        if (self.audio_preproc) |*p| p.deinit();
        self.audio_preproc = null;

        const proj = clip.getProjectorType(self.ctx_a.?);

        impl.logWrn("%s: audio input is in experimental stage and may have reduced quality:\n" ++
            "    https://github.com/ggml-org/llama.cpp/discussions/13759\n", .{"init_audio"});

        // set preprocessor
        const kind: audio.Kind = switch (proj) {
            .qwen2a, .qwen25o => blk: {
                // <|audio_bos|> ... (embeddings) ... <|audio_eos|>
                self.aud_beg = "<|audio_bos|>";
                self.aud_end = "<|audio_eos|>";
                break :blk .whisper;
            },
            .qwen3a => blk: {
                self.aud_beg = "<|audio_start|>";
                self.aud_end = "<|audio_end|>";
                break :blk .qwen3a;
            },
            .voxtral => blk: {
                // [BEGIN_AUDIO] ... (embeddings) ...
                self.aud_beg = "[BEGIN_AUDIO]";
                break :blk .whisper;
            },
            .music_flamingo => blk: {
                // <sound> ... (embeddings) ...
                self.aud_beg = "<sound>";
                break :blk .whisper;
            },
            .ultravox, .glma, .meralion => .whisper,
            .lfm2a => .conformer,
            .granite_speech => .granite_speech,
            .gemma4a => blk: {
                self.aud_beg = "<|audio>";
                self.aud_end = "<audio|>";
                break :blk .gemma4a;
            },
            .parakeet => .parakeet,
            .gemma4ua => blk: {
                self.aud_beg = "<|audio>";
                self.aud_end = "<audio|>";
                break :blk .gemma4ua;
            },
            .dots3note_a => blk: {
                // <|audio_comp_start|> ... (embeddings) ... <|audio_comp_end|>
                self.aud_beg = "<|audio_comp_start|>";
                self.aud_end = "<|audio_comp_end|>";
                break :blk .dots3note;
            },
            .mimo_audio => blk: {
                self.aud_beg = "<|mimo_audio_start|>";
                self.aud_end = "<|mimo_audio_end|>";
                break :blk .mimo_audio;
            },
            .qwen3tts_spkenc => .qwen3tts_spk,
            .pockettts_spkenc => .pockettts,
            else => return clip.throwErr("%s: unexpected audio projector type %d\n", .{ "init_audio", @as(c_int, @backingInt(proj)) }),
        };

        // initialize audio preprocessor
        self.audio_preproc = audio.Preprocessor.init(kind, clip.getHparams(self.ctx_a.?));
        try self.audio_preproc.?.initialize();
    }

    /// Ports `get_clip_ctx` (mtmd.cpp:1018 @5266f24da): the clip context
    /// for a chunk's type.
    fn getClipCtx(self: *const Context, chunk: *const InputChunk) ?*clip.Ctx {
        if (chunk.type == .image) return self.ctx_v;
        if (chunk.type == .audio) return self.ctx_a;
        impl.abort("mtmd.cpp", 1024, "unknown chunk type");
    }

    fn projTypeV(self: *const Context) names.ProjectorType {
        return if (self.ctx_v) |v| clip.getProjectorType(v) else .unknown;
    }

    fn projTypeA(self: *const Context) names.ProjectorType {
        return if (self.ctx_a) |a| clip.getProjectorType(a) else .unknown;
    }

    /// Ports `n_embd_out` (mtmd.cpp:1035 @5266f24da).
    fn nEmbdOut(self: *const Context) Error!i64 {
        if (self.ctx_v) |v| return clip.nMmprojEmbd(v);
        if (self.ctx_a) |a| return clip.nMmprojEmbd(a);
        return clip.throwStr("no CLIP model loaded");
    }

    /// Ports `mtmd_context` (mtmd.cpp:1045 @5266f24da), its destructor.
    fn destroy(self: *Context) void {
        clip.free(self.ctx_a);
        clip.free(self.ctx_v);
        clip.free(self.ctx_gen_a);
        self.deinitMembers();
        gpa.destroy(self);
    }

    fn deinitMembers(self: *Context) void {
        if (self.audio_preproc) |*p| p.deinit();
        self.out_embd.deinit(gpa);
        self.gen_out_codes.deinit(gpa);
        self.gen_out_feats.deinit(gpa);
        self.gen_out_embd.deinit(gpa);
        self.gen_out_audio.deinit(gpa);
        self.gen_out_state.deinit(gpa);
        self.media_marker.deinit();
        inline for (.{ "tok_ov_img_start", "tok_ov_img_end", "tok_slices_start", "tok_slices_end", "tok_sli_img_start", "tok_sli_img_end", "tok_sli_img_mid", "tok_row_end" }) |f| {
            @field(self, f).deinit(gpa);
        }
    }

    /// Ports `lookup_token` (mtmd.cpp:1052 @5266f24da): the first token
    /// whose special piece is `token_text`.
    fn lookupToken(self: *const Context, token_text: []const u8) Token {
        // TODO @ngxson : this case is currently hit by mtmd_get_memory_usage
        const vocab = self.vocab orelse return token_null;
        const n_vocab = lapi.vocab.llama_vocab_n_tokens(vocab);
        var piece: std.ArrayList(u8) = .empty;
        defer piece.deinit(gpa);
        var i: Token = 0;
        while (i < n_vocab) : (i += 1) {
            tokenToPiece(vocab, i, true, &piece) catch return token_null;
            if (std.mem.eql(u8, piece.items, token_text)) return i;
        }
        return token_null;
    }

    /// Ports `token_to_piece` (mtmd.cpp:1067 @5266f24da). The first try
    /// uses libc++'s short-string capacity, 22 bytes.
    fn tokenToPiece(vocab: *const c.struct_llama_vocab, token: Token, special: bool, piece: *std.ArrayList(u8)) !void {
        try resizeZero(u8, piece, 22); // using string internal cache
        const n_chars = lapi.vocab.llama_token_to_piece(vocab, token, piece.items.ptr, @intCast(piece.items.len), 0, special);
        if (n_chars < 0) {
            try resizeZero(u8, piece, @intCast(-n_chars));
            const check = lapi.vocab.llama_token_to_piece(vocab, token, piece.items.ptr, @intCast(piece.items.len), 0, special);
            impl.ggmlAssert(check == -n_chars, "mtmd.cpp", 1077, "check == -n_chars");
        } else {
            piece.shrinkRetainingCapacity(@intCast(n_chars));
        }
    }
};

/// Ports `mtmd_init_from_file` (mtmd.cpp:1085 @5266f24da).
pub fn mtmd_init_from_file(mmproj_fname: [*:0]const u8, text_model: ?*const c.struct_llama_model, ctx_params: ContextParams) callconv(.c) ?*Context {
    return Context.create(mmproj_fname, text_model, &ctx_params, false) catch |e| {
        impl.logErr("%s: error: %s\n", .{ "mtmd_init_from_file", whatOf(e) });
        return null;
    };
}

/// Ports `mtmd_free` (mtmd.cpp:1096 @5266f24da).
pub fn mtmd_free(ctx: ?*Context) callconv(.c) void {
    if (ctx) |x| x.destroy();
}

// -----------------------------------------------------------------------------
// tokenizer

/// Ports `mtmd_internal_part` (mtmd-internal.h:13 @5266f24da): a bitmap, or
/// text when it is null.
pub const InternalPart = struct {
    text: []const u8,
    bitmap: ?*const Bitmap,
    // only used for text parts
    parse_special: bool = false,
};

/// Ports `mtmd_group_mergeable_bitmaps` (mtmd.cpp:1100 @5266f24da):
/// [QWEN_VIDEO] merged parts are erased from `parts`, so one group always
/// maps to one part. The caller owns the groups.
pub fn groupMergeableBitmaps(parts: *std.ArrayList(InternalPart), n_merge: c_int) !std.ArrayList(std.ArrayList(*const Bitmap)) {
    var output: std.ArrayList(std.ArrayList(*const Bitmap)) = .empty;
    errdefer freeGroups(&output);
    var i: usize = 0;
    while (i < parts.items.len) : (i += 1) {
        const bm = parts.items[i].bitmap orelse continue; // text part
        const has_next = n_merge > 1 and i + 1 < parts.items.len and parts.items[i + 1].bitmap != null;
        var group: std.ArrayList(*const Bitmap) = .empty;
        errdefer group.deinit(gpa);
        if (has_next and bm.canMergeWith(parts.items[i + 1].bitmap.?)) {
            impl.logDbg("%s: merging 2 frames at part index %zu and %zu\n", .{ "mtmd_group_mergeable_bitmaps", i, i + 1 });
            try group.appendSlice(gpa, &.{ bm, parts.items[i + 1].bitmap.? });
            try output.append(gpa, group);
            _ = parts.orderedRemove(i + 1);
            continue;
        }
        impl.logDbg("%s: no merging for part index %zu\n", .{ "mtmd_group_mergeable_bitmaps", i });
        try group.append(gpa, bm);
        try output.append(gpa, group);
    }
    return output;
}

pub fn freeGroups(groups: *std.ArrayList(std.ArrayList(*const Bitmap))) void {
    for (groups.items) |*g| g.deinit(gpa);
    groups.deinit(gpa);
}

/// Ports `mtmd_tokenizer` (mtmd.cpp:1119 @5266f24da).
const Tokenizer = struct {
    ctx: *const Context,

    input_text: []const u8, // note: can contain null bytes
    add_special: bool,
    parse_special: bool,
    vocab: ?*const c.struct_llama_vocab,

    parts: std.ArrayList(InternalPart) = .empty,
    // these will be freed when the tokenizer finishes
    bm_from_lazy: std.ArrayList(*Bitmap) = .empty,
    text_from_lazy: std.ArrayList([*:0]u8) = .empty,

    cur: InputChunks = .{},
    n_images_added: u32 = 0, // 0-based index assigned to the next image chunk

    /// The constructor `mtmd_tokenizer` (mtmd.cpp:1143 @5266f24da) from a
    /// marked prompt.
    fn init(self: *Tokenizer, ctx: *const Context, text: *const InputText, bmps: []const *const Bitmap) Error!void {
        self.* = .{
            .ctx = ctx,
            .add_special = text.add_special,
            .parse_special = text.parse_special,
            .vocab = ctx.vocab,
            .input_text = if (text.text_len == 0) "" else text.text.?[0..text.text_len],
        };

        const marker = ctx.media_marker.slice();
        var i_bm: usize = 0;
        // split_text: "a <__media__> b <__media__> c" --> "a", "<__media__>", "b", "<__media__>", "c"
        if (self.input_text.len != 0) {
            const input = self.input_text;
            var start: usize = 0;
            while (std.mem.indexOfPos(u8, input, start, marker)) |pos| {
                if (pos > start) try self.parts.append(gpa, .{ .text = input[start..pos], .bitmap = null, .parse_special = self.parse_special });
                if (i_bm >= bmps.len) {
                    return clip.throwErr("number of media markers in text (%zu) exceeds number of bitmaps (%zu)", .{ i_bm + 1, bmps.len });
                }
                try self.parts.append(gpa, .{ .text = "", .bitmap = bmps[i_bm] });
                i_bm += 1;
                start = pos + marker.len;
            }
            if (start < input.len) try self.parts.append(gpa, .{ .text = input[start..], .bitmap = null, .parse_special = self.parse_special });
        }

        var n_markers: usize = 0;
        for (self.parts.items) |p| {
            if (p.bitmap != null) n_markers += 1;
        }
        if (n_markers != bmps.len) {
            return clip.throwErr("number of media markers in text (%zu) does not match number of bitmaps (%zu)", .{ n_markers, bmps.len });
        }

        try self.expandLazyBitmaps();
    }

    /// The constructor `mtmd_tokenizer` (mtmd.cpp:1180 @5266f24da) from a
    /// list of parts. Each part has exactly one of text and bitmap set,
    /// which `mtmd_tokenize_from_parts` has checked.
    fn initParts(self: *Tokenizer, ctx: *const Context, input_parts: []const *const InputPart, add_special: bool) Error!void {
        self.* = .{
            .ctx = ctx,
            .add_special = add_special,
            .parse_special = true, // only used for text returned by lazy bitmaps
            .vocab = ctx.vocab,
            .input_text = "",
        };

        for (input_parts) |p| {
            if (p.text) |t| {
                const txt: []const u8 = if (t.text_len == 0) "" else t.text.?[0..t.text_len];
                try self.parts.append(gpa, .{ .text = txt, .bitmap = null, .parse_special = t.parse_special });
            } else {
                try self.parts.append(gpa, .{ .text = "", .bitmap = p.bitmap });
            }
        }

        try self.expandLazyBitmaps();
    }

    /// The destructor of `mtmd_tokenizer` (mtmd.cpp:1136 @5266f24da).
    fn deinit(self: *Tokenizer) void {
        for (self.bm_from_lazy.items) |b| b.destroy();
        self.bm_from_lazy.deinit(gpa);
        for (self.text_from_lazy.items) |s| free(s);
        self.text_from_lazy.deinit(gpa);
        self.parts.deinit(gpa);
        self.cur.deinit();
    }

    /// Ports `expand_lazy_bitmaps` (mtmd.cpp:1200 @5266f24da).
    fn expandLazyBitmaps(self: *Tokenizer) Error!void {
        var expanded: std.ArrayList(InternalPart) = .empty;
        errdefer expanded.deinit(gpa);
        try expanded.ensureTotalCapacity(gpa, self.parts.items.len);
        for (self.parts.items) |p| {
            if (p.bitmap != null and p.bitmap.?.lazy_callback != null) {
                const bm = p.bitmap.?;
                impl.logDbg("%s: expanding lazy bitmap\n", .{"expand_lazy_bitmaps"});
                var i: usize = 0;
                while (true) : (i += 1) {
                    var out_str: ?[*:0]u8 = null;
                    var out_bm: ?*Bitmap = null;
                    const res = bm.lazy_callback.?(i, bm.lazy_user_data, &out_bm, &out_str);
                    if (out_bm != null and out_str != null) {
                        return clip.throwStr("lazy callback cannot return both bitmap and text");
                    }
                    if (res == 0) {
                        // OK, append the returned chunk; lazy part is not yet added
                        if (out_bm) |ob| {
                            try self.bm_from_lazy.append(gpa, ob); // remember to free it later
                            try expanded.append(gpa, .{ .text = "", .bitmap = ob });
                            impl.logDbg("%s: lazy callback returned bitmap with dimensions %d x %d\n", .{ "expand_lazy_bitmaps", ob.nx, ob.ny });
                        } else if (out_str) |os| {
                            try self.text_from_lazy.append(gpa, os); // remember to free it later
                            try expanded.append(gpa, .{ .text = std.mem.span(os), .bitmap = null, .parse_special = self.parse_special });
                            impl.logDbg("%s: lazy callback returned text: %s\n", .{ "expand_lazy_bitmaps", os });
                        }
                    } else if (res == -1) {
                        // EOF: lazy part removes itself (not added to expanded)
                        break;
                    } else if (res == -2) {
                        // error
                        return clip.throwStr("lazy callback returned error");
                    }
                }
            } else {
                try expanded.append(gpa, p);
            }
        }
        self.parts.deinit(gpa);
        self.parts = expanded;
    }

    /// Ports `tokenize` (mtmd.cpp:1242 @5266f24da).
    fn tokenize(self: *Tokenizer, output: *InputChunks) Error!i32 {
        self.cur.clear();

        // [QWEN_VIDEO] handle frame merging for models that support it (i.e. qwen-vl)
        var n_merge_frames: c_int = 1;
        if (self.ctx.ctx_v) |v| {
            n_merge_frames = clip.modelNTemporalMerge(v);
            impl.ggmlAssert(n_merge_frames <= 2, "mtmd.cpp", 1249, "n_merge_frames <= 2 && \"we only support merging maximum 2 images for now; open an issue if this model supports merging more\"");
        }

        var merged_bitmaps = try groupMergeableBitmaps(&self.parts, n_merge_frames);
        defer freeGroups(&merged_bitmaps);

        var i_bm: usize = 0;
        for (self.parts.items) |p| {
            if (p.bitmap != null) {
                if (i_bm >= merged_bitmaps.items.len) {
                    impl.logErr("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n", .{ "tokenize", merged_bitmaps.items.len, self.parts.items.len -% 1 });
                    return 1;
                }
                const bmps = merged_bitmaps.items[i_bm].items;
                i_bm += 1;
                const res = try self.addMedia(bmps);
                if (res != 0) return res;
            } else {
                try self.addTextStr(p.text, p.parse_special);
            }
        }

        if (self.vocab) |vocab| {
            if (self.add_special and lapi.vocab.llama_vocab_get_add_bos(vocab)) {
                // if first chunk is text, we add BOS token to first text chunk
                // otherwise, create a new text chunk with BOS token
                const bos = lapi.vocab.llama_vocab_bos(vocab);
                if (self.cur.entries.items.len != 0 and self.cur.entries.items[0].type == .text) {
                    try self.cur.entries.items[0].tokens_text.insert(gpa, 0, bos);
                } else {
                    var bos_chunk: InputChunk = .{ .type = .text };
                    try bos_chunk.tokens_text.append(gpa, bos);
                    try self.cur.entries.insert(gpa, 0, bos_chunk);
                }
            }

            if (self.add_special and lapi.vocab.llama_vocab_get_add_eos(vocab)) {
                // if last chunk is text, we add EOS token to it
                try self.addTextTokens(&.{lapi.vocab.llama_vocab_eos(vocab)});
            }
        }

        if (i_bm != merged_bitmaps.items.len) {
            impl.logErr("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n", .{ "tokenize", merged_bitmaps.items.len, self.parts.items.len -% 1 });
            return 1;
        }

        output.deinit();
        output.* = self.cur;
        self.cur = .{};
        return 0;
    }

    /// Ports `add_text` (mtmd.cpp:1308 @5266f24da), the string overload.
    fn addTextStr(self: *Tokenizer, txt: []const u8, parse_special: bool) Error!void {
        const vocab = self.vocab orelse return clip.throwStr("llama_vocab is not provided");
        impl.logDbg("%s: %.*s\n", .{ "add_text", @as(c_int, @intCast(txt.len)), txt.ptr });
        var tokens = try tokenizeTextInternal(vocab, txt, false, parse_special);
        defer tokens.deinit(gpa);
        try self.addTextTokens(tokens.items);
    }

    /// Ports `add_text` (mtmd.cpp:1317 @5266f24da), the tokens overload.
    fn addTextTokens(self: *Tokenizer, tokens: []const Token) !void {
        if (tokens.len == 0) return;
        // if last entry is also a text chunk, add tokens to it instead of creating new chunk
        const entries = self.cur.entries.items;
        if (entries.len != 0 and entries[entries.len - 1].type == .text) {
            try entries[entries.len - 1].tokens_text.appendSlice(gpa, tokens);
        } else {
            var chunk: InputChunk = .{ .type = .text };
            errdefer chunk.deinit();
            try chunk.tokens_text.appendSlice(gpa, tokens);
            try self.cur.entries.append(gpa, chunk);
        }
    }

    fn pushChunk(self: *Tokenizer, chunk: InputChunk) !void {
        var ch = chunk;
        errdefer ch.deinit();
        try self.cur.entries.append(gpa, ch);
    }

    /// Ports `add_media` (mtmd.cpp:1338 @5266f24da).
    fn addMedia(self: *Tokenizer, bitmaps: []const *const Bitmap) Error!i32 {
        impl.ggmlAssert(bitmaps.len != 0, "mtmd.cpp", 1339, "!bitmaps.empty()");
        const ctx = self.ctx;

        // note: only one type of media is supported per call, caller should enforce this
        const is_vision = !bitmaps[0].is_audio;

        if (is_vision) {
            // handle image
            const ctx_v = ctx.ctx_v orelse {
                impl.logErr("%s: error: model does not support vision input\n", .{"add_media"});
                return 2;
            };

            if (ctx.img_beg.len != 0) try self.addTextStr(ctx.img_beg, true); // add image begin token

            var preproc_out: image.PreprocOut = .{};
            defer preproc_out.deinit();

            for (bitmaps) |bmp| {
                // sanity check
                impl.ggmlAssert(!bmp.is_audio, "mtmd.cpp", 1362, "!bmp->is_audio");
                impl.ggmlAssert(ctx.image_preproc != null, "mtmd.cpp", 1363, "ctx->image_preproc != nullptr");
                if (bmp.nx <= 0 or bmp.ny <= 0) {
                    impl.logErr("%s: error: invalid bitmap dimensions: nx = %d, ny = %d\n", .{ "add_media", bmp.nx, bmp.ny });
                    return 2;
                }

                // convert mtmd_bitmap to clip_image_u8
                var img_u8: impl.ImageU8 = .{};
                defer img_u8.deinit();
                try img_u8.setSize(.{ .width = @bitCast(bmp.nx), .height = @bitCast(bmp.ny) }, bmp.isPlaceholder());
                try img_u8.cpyBuf(bmp.data.items);

                // preprocess image
                var tmp_preproc_out = try ctx.image_preproc.?.preprocess(&img_u8);
                defer tmp_preproc_out.deinit();

                // move entries and grid dimensions to the "global" preproc_out
                try preproc_out.entries.appendSlice(gpa, tmp_preproc_out.entries.items);
                tmp_preproc_out.entries.clearRetainingCapacity();

                // for llava-uhd style, we need to handle grid too
                if ((tmp_preproc_out.grid_x > 0 and tmp_preproc_out.grid_y > 0) or tmp_preproc_out.hasOverview()) {
                    impl.ggmlAssert(bitmaps.len == 1, "mtmd.cpp", 1389, "bitmaps.size() == 1");
                    preproc_out.grid_x = tmp_preproc_out.grid_x;
                    preproc_out.grid_y = tmp_preproc_out.grid_y;
                    preproc_out.overview.deinit();
                    preproc_out.overview = tmp_preproc_out.overview;
                    tmp_preproc_out.overview = .{};
                }
            }

            impl.logDbg("%s: preproc_out has %zu entries, grid_x = %d, grid_y = %d, has_overview = %d\n", .{
                "add_media",                                         preproc_out.entries.items.len, preproc_out.grid_x, preproc_out.grid_y,
                @as(c_int, @intFromBool(preproc_out.hasOverview())),
            });

            // handle llava-uhd style preprocessing
            // (output either a grid, or overview-only)
            const has_tiling_grid = (preproc_out.grid_x > 0 and preproc_out.grid_y > 0) or preproc_out.hasOverview();

            if (has_tiling_grid) {
                // [QWEN_VIDEO] we do not support "frame merging" for llama-uhd style, so no batching for now
                impl.ggmlAssert(bitmaps.len == 1, "mtmd.cpp", 1407, "bitmaps.size() == 1");

                const n_col = preproc_out.grid_x;
                const n_row = preproc_out.grid_y;

                // split batch into chunks of single images
                var chunks = try self.splitBatchToChunk(&preproc_out, &bitmaps[0].id);
                defer {
                    for (chunks.items) |*ch| ch.deinit();
                    chunks.deinit(gpa);
                }
                impl.ggmlAssert(chunks.items.len > 0, "mtmd.cpp", 1414, "chunks.size() > 0");

                // split_batch_to_chunk must always put the overview image first
                var ov_chunk = chunks.orderedRemove(0);
                defer ov_chunk.deinit();

                // add overview image (first)
                if (ctx.ov_img_first) {
                    try self.addTextTokens(ctx.tok_ov_img_start.items);
                    try self.pushChunk(ov_chunk);
                    ov_chunk = .{};
                    try self.addTextTokens(ctx.tok_ov_img_end.items);
                }

                // add slices (or tiles)
                if (chunks.items.len != 0) {
                    impl.logDbg("%s: adding %d slices (%d rows x %d cols)\n", .{ "add_media", @as(c_int, @intCast(chunks.items.len)), n_row, n_col });
                    impl.ggmlAssert(@as(c_int, @intCast(chunks.items.len)) == n_row * n_col, "mtmd.cpp", 1432, "(int)chunks.size() == n_row * n_col");
                    try self.addTextTokens(ctx.tok_slices_start.items);
                    var y: c_int = 0;
                    while (y < n_row) : (y += 1) {
                        var x: c_int = 0;
                        while (x < n_col) : (x += 1) {
                            const is_last_in_row = (x == n_col - 1);
                            if (ctx.tok_sli_img_start.items.len != 0) {
                                try self.addTextTokens(ctx.tok_sli_img_start.items);
                            } else if (ctx.sli_img_start_tmpl) |tmpl| {
                                // If using a template to preceed a slice image
                                const sz: usize = @intCast(snprintf(null, 0, tmpl, y + 1, x + 1) + 1);
                                const buf = try gpa.alloc(u8, sz);
                                defer gpa.free(buf);
                                _ = snprintf(buf.ptr, sz, tmpl, y + 1, x + 1);
                                try self.addTextStr(buf[0 .. sz - 1], true);
                            }

                            const curr_chunk = &chunks.items[@intCast(y * n_col + x)];
                            const curr_batch = &curr_chunk.tokens_image.?.batch_f32;
                            if (curr_batch.entries.items.len != 1) {
                                return clip.throwErr("%s: expect 1 image in batch_f32", .{"add_media"});
                            }

                            impl.logDbg("%s: adding slice image at row %d col %d\n", .{ "add_media", y, x });
                            try self.pushChunk(curr_chunk.*);
                            curr_chunk.* = .{};

                            try self.addTextTokens(ctx.tok_sli_img_end.items);
                            if (!is_last_in_row) try self.addTextTokens(ctx.tok_sli_img_mid.items);
                        }
                        if (y != n_row - 1 or ctx.tok_row_end_trail) try self.addTextTokens(ctx.tok_row_end.items);
                    }
                    try self.addTextTokens(ctx.tok_slices_end.items);
                }

                // add overview image (last)
                if (!ctx.ov_img_first) {
                    try self.addTextTokens(ctx.tok_ov_img_start.items);
                    try self.pushChunk(ov_chunk);
                    ov_chunk = .{};
                    try self.addTextTokens(ctx.tok_ov_img_end.items);
                }
            } else {
                if (preproc_out.entries.items.len == 0) {
                    impl.logErr("%s: no image tokens produced by preprocessor (ref: https://github.com/ggml-org/llama.cpp/pull/24769)\n", .{"add_media"});
                    return 2;
                }

                if (ctx.projTypeV() == .deepseek4v) {
                    // the text model perceives input in blocks of N tokens (N = COMPRESS_PAD_TO = 4, same as the CSA compress ratio)
                    // image need to be aligned to block size, while adding IMAGE_PAD embeddings to the beginning
                    const alignment: i32 = 4;
                    var n_past: usize = 0;
                    for (self.cur.entries.items) |*e| n_past +%= mtmd_input_chunk_get_n_tokens(e);
                    preproc_out.entries.items[0].lead_pad = alignment - 1 - @as(i32, @intCast(n_past % alignment));
                }

                var n_tokens: usize = 0;
                for (preproc_out.entries.items) |*e| {
                    n_tokens = addInt(n_tokens, clip.nOutputTokens(ctx_v, e));
                    if (clip.modelNTemporalMerge(ctx_v) == 2) {
                        // [QWEN_VIDEO] pair input is merged to the same embd, so only count as one image
                        break;
                    }
                }

                const image_tokens = try gpa.create(ImageTokens);
                image_tokens.* = .{};
                errdefer image_tokens.destroy();

                // [QWEN_VIDEO] improve this in the future
                image_tokens.n_temporal_merge = @bitCast(clip.modelNTemporalMerge(ctx_v));

                if (mtmd_decode_use_mrope(ctx)) {
                    // for Qwen2VL, we need this information for M-RoPE decoding positions
                    image_tokens.nx = @bitCast(clip.nOutputTokensX(ctx_v, &preproc_out.entries.items[0]));
                    image_tokens.ny = @bitCast(clip.nOutputTokensY(ctx_v, &preproc_out.entries.items[0]));
                } else {
                    // other models, we only need the total number of tokens
                    image_tokens.nx = @truncate(n_tokens);
                    image_tokens.ny = 1;
                }
                image_tokens.pos = ctx.pos_type;
                // HunyuanVL wraps the image grid with BOI/EOI and adds one newline per row,
                // and uses XD-RoPE (dim-3 = image index).
                if (ctx.projTypeV() == .hunyuanvl) {
                    image_tokens.pos = .hunyuanvl;
                    image_tokens.image_idx = self.n_images_added;
                    impl.ggmlAssert(n_tokens == @as(usize, image_tokens.nTokens()), "mtmd.cpp", 1523, "n_tokens == (size_t)image_tokens->n_tokens()");
                }

                image_tokens.batch_f32.is_audio = false;
                image_tokens.batch_f32.entries = preproc_out.entries;
                preproc_out.entries = .empty;
                // do NOT use preproc_out from this point on, it's moved

                try image_tokens.id.set(bitmaps[0].id.slice()); // optional

                impl.logDbg("image_tokens->nx = %d\n", .{image_tokens.nx});
                impl.logDbg("image_tokens->ny = %d\n", .{image_tokens.ny});
                impl.logDbg("batch_f32 size = %d\n", .{@as(c_int, @intCast(image_tokens.batch_f32.entries.items.len))});

                try self.pushChunk(.{ .type = .image, .tokens_image = image_tokens });
            }

            if (ctx.img_end.len != 0) try self.addTextStr(ctx.img_end, true); // add image end token

            // advance image-chunk counter so the next image gets the next XD-RoPE dim-3 slot
            self.n_images_added += 1;
        } else {
            // handle audio

            impl.ggmlAssert(bitmaps.len == 1, "mtmd.cpp", 1557, "bitmaps.size() == 1"); // no batching support for now
            const bitmap = bitmaps[0];

            const ctx_a = ctx.ctx_a orelse {
                impl.logErr("%s: error: model does not support audio input\n", .{"add_media"});
                return 2;
            };

            if (bitmap.nx == 0) {
                impl.logErr("%s: error: empty audio data\n", .{"add_media"});
                return 2;
            }

            if (ctx.aud_beg.len != 0) try self.addTextStr(ctx.aud_beg, true); // add audio begin token

            // sanity check
            impl.ggmlAssert(ctx.audio_preproc != null, "mtmd.cpp", 1575, "ctx->audio_preproc != nullptr");

            // preprocess audio
            var mel_spec_chunks: std.ArrayList(audio.Mel) = .empty;
            defer {
                for (mel_spec_chunks.items) |*m| m.deinit();
                mel_spec_chunks.deinit(gpa);
            }
            {
                var dummy: std.ArrayList(f32) = .empty;
                defer dummy.deinit(gpa);
                var samples: []const f32 = &.{};
                if (bitmap.isPlaceholder()) {
                    // TODO @ngxson : skip underlay processing if bitmap is placeholder
                    impl.ggmlAssert(bitmap.ny == 1, "mtmd.cpp", 1585, "bitmap->ny == 1");
                    try resizeZero(f32, &dummy, bitmap.nx);
                    samples = dummy.items;
                } else {
                    const buf = bitmap.data.items;
                    impl.ggmlAssert(buf.len > @sizeOf(f32), "mtmd.cpp", 1592, "buf.size() > sizeof(float)");
                    impl.ggmlAssert(buf.len % @sizeOf(f32) == 0, "mtmd.cpp", 1593, "buf.size() % sizeof(float) == 0");
                    samples = @as([*]const f32, @ptrCast(@alignCast(buf.ptr)))[0 .. buf.len / @sizeOf(f32)];
                }
                const ok = try ctx.audio_preproc.?.preprocess(samples, &mel_spec_chunks);
                if (!ok) {
                    impl.logErr("Unable to preprocess audio\n", .{});
                    return 2;
                }
            }

            // consider each mel_spec as a separate audio chunk
            for (mel_spec_chunks.items) |*mel_spec| {
                const is_placeholder = mel_spec.data.items.len == 0;

                // Validate dimensions fit in clip_image_size (int)
                impl.ggmlAssert(mel_spec.n_len <= std.math.maxInt(i32) and mel_spec.n_len >= 0, "mtmd.cpp", 1611, "mel_spec.n_len <= INT32_MAX && mel_spec.n_len >= 0");
                impl.ggmlAssert(mel_spec.n_mel <= std.math.maxInt(i32) and mel_spec.n_mel >= 0, "mtmd.cpp", 1612, "mel_spec.n_mel <= INT32_MAX && mel_spec.n_mel >= 0");
                var mel_f32: impl.ImageF32 = .{};
                errdefer mel_f32.deinit();
                try mel_f32.setSize(.{ .width = @intCast(mel_spec.n_len), .height = @intCast(mel_spec.n_mel) }, is_placeholder, true);
                try mel_f32.cpyBuf(mel_spec.data.items);

                const n_tokens = addInt(0, clip.nOutputTokens(ctx_a, &mel_f32));

                const audio_tokens = try gpa.create(AudioTokens);
                audio_tokens.* = .{};
                errdefer audio_tokens.destroy();
                audio_tokens.batch_f32.is_audio = true;
                try audio_tokens.batch_f32.entries.append(gpa, mel_f32);
                audio_tokens.n_tokens = @truncate(n_tokens);
                try audio_tokens.id.set(bitmap.id.slice()); // optional

                impl.logDbg("audio_tokens->n_tokens = %d\n", .{audio_tokens.n_tokens});

                try self.pushChunk(.{ .type = .audio, .tokens_audio = audio_tokens });
            }

            if (ctx.aud_end.len != 0) try self.addTextStr(ctx.aud_end, true); // add audio end token
        }

        return 0;
    }

    /// Ports `split_batch_to_chunk` (mtmd.cpp:1649 @5266f24da): the
    /// overview, then each slice, as image chunks of one image each.
    /// `preproc_out`'s images are moved out.
    fn splitBatchToChunk(self: *Tokenizer, preproc_out: *image.PreprocOut, id: *const Str) Error!std.ArrayList(InputChunk) {
        var chunks: std.ArrayList(InputChunk) = .empty;
        errdefer {
            for (chunks.items) |*ch| ch.deinit();
            chunks.deinit(gpa);
        }

        // overview image first
        const overview = &preproc_out.overview;
        if (overview.nx() == 0 or overview.ny() == 0) {
            return clip.throwErr("%s: invalid overview image for llava-uhd style preprocessing\n", .{"split_batch_to_chunk"});
        }
        try self.processChunk(&chunks, overview, id);

        // then, process slices
        for (preproc_out.entries.items) |*entry| {
            if (entry.nx() == 0 or entry.ny() == 0) {
                return clip.throwErr("%s: invalid image slice for llava-uhd style preprocessing\n", .{"split_batch_to_chunk"});
            }
            try self.processChunk(&chunks, entry, id);
        }

        return chunks;
    }

    /// `split_batch_to_chunk`'s `process_chunk` lambda (mtmd.cpp:1652
    /// @5266f24da). `img` is moved out.
    fn processChunk(self: *Tokenizer, chunks: *std.ArrayList(InputChunk), img: *impl.ImageF32, id: *const Str) Error!void {
        const image_tokens = try gpa.create(ImageTokens);
        image_tokens.* = .{};
        errdefer image_tokens.destroy();
        image_tokens.nx = @bitCast(clip.nOutputTokens(self.ctx.ctx_v.?, img));
        image_tokens.ny = 1;
        try image_tokens.batch_f32.entries.append(gpa, img.*);
        img.* = .{};
        try image_tokens.id.set(id.slice());

        impl.ggmlAssert(image_tokens.nx > 0, "mtmd.cpp", 1659, "image_tokens->nx > 0");

        try chunks.append(gpa, .{ .type = .image, .tokens_image = image_tokens });
    }

    /// Ports `mtmd_tokenize_text_internal` (mtmd.cpp:1710 @5266f24da),
    /// copied upstream from `common_tokenize`. The caller owns the tokens.
    fn tokenizeTextInternal(vocab: *const c.struct_llama_vocab, text: []const u8, add_special: bool, parse_special: bool) Error!std.ArrayList(Token) {
        // upper limit for the number of tokens
        var n_tokens: c_int = @truncate(@as(isize, @intCast(text.len)) + 2 * @as(isize, @intFromBool(add_special)));
        var result: std.ArrayList(Token) = .empty;
        errdefer result.deinit(gpa);
        try resizeZero(Token, &result, @intCast(n_tokens));
        n_tokens = lapi.vocab.llama_tokenize(vocab, text.ptr, @truncate(@as(isize, @intCast(text.len))), result.items.ptr, @intCast(result.items.len), add_special, parse_special);
        if (n_tokens == std.math.minInt(i32)) {
            return clip.throwStr("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
        }
        if (n_tokens < 0) {
            try resizeZero(Token, &result, @intCast(-n_tokens));
            const check = lapi.vocab.llama_tokenize(vocab, text.ptr, @truncate(@as(isize, @intCast(text.len))), result.items.ptr, @intCast(result.items.len), add_special, parse_special);
            impl.ggmlAssert(check == -n_tokens, "mtmd.cpp", 1728, "check == -n_tokens");
        } else {
            result.shrinkRetainingCapacity(@intCast(n_tokens));
        }
        return result;
    }
};

/// Ports `mtmd_tokenize` (mtmd.cpp:1736 @5266f24da).
pub fn mtmd_tokenize(ctx: *const Context, output: *InputChunks, text: *const InputText, bitmaps: ?[*]const *const Bitmap, n_bitmaps: usize) callconv(.c) i32 {
    var tokenizer: Tokenizer = undefined;
    const bmps: []const *const Bitmap = if (n_bitmaps == 0) &.{} else bitmaps.?[0..n_bitmaps];
    tokenizer.init(ctx, text, bmps) catch |e| {
        tokenizer.deinit();
        impl.logErr("%s: error: %s\n", .{ "mtmd_tokenize", whatOf(e) });
        return 2;
    };
    defer tokenizer.deinit();
    return tokenizer.tokenize(output) catch |e| {
        impl.logErr("%s: error: %s\n", .{ "mtmd_tokenize", whatOf(e) });
        return 2;
    };
}

/// Ports `mtmd_tokenize_from_parts` (mtmd.cpp:1750 @5266f24da): as
/// `mtmd_tokenize`, from a list of text and media parts in place of a
/// marked prompt. Returns 1 when a part has both or neither of text and
/// bitmap set.
pub fn mtmd_tokenize_from_parts(ctx: *const Context, output: *InputChunks, parts: ?[*]const *const InputPart, n_parts: usize, add_special: bool) callconv(.c) i32 {
    const ps: []const *const InputPart = if (n_parts == 0) &.{} else parts.?[0..n_parts];
    for (ps, 0..) |p, i| {
        if ((p.text == null) == (p.bitmap == null)) {
            impl.logErr("%s: part %zu must have either text or bitmap set, not both\n", .{ "mtmd_tokenize_from_parts", i });
            return 1;
        }
        if (p.text != null and p.text.?.text == null) {
            impl.logErr("%s: part %zu has null text pointer\n", .{ "mtmd_tokenize_from_parts", i });
            return 1;
        }
    }
    var tokenizer: Tokenizer = undefined;
    tokenizer.initParts(ctx, ps, add_special) catch |e| {
        tokenizer.deinit();
        impl.logErr("%s: error: %s\n", .{ "mtmd_tokenize_from_parts", whatOf(e) });
        return 2;
    };
    defer tokenizer.deinit();
    return tokenizer.tokenize(output) catch |e| {
        impl.logErr("%s: error: %s\n", .{ "mtmd_tokenize_from_parts", whatOf(e) });
        return 2;
    };
}

// -----------------------------------------------------------------------------
// encoding

/// Ports `mtmd_encode_impl` (mtmd.cpp:1774 @5266f24da).
fn encodeImpl(ctx: *Context, image_tokens: *const ImageTokens, out_embd: *std.ArrayList(f32)) Error!i32 {
    const ctx_clip = ctx.ctx_v orelse {
        impl.logErr("%s: this API does not support non-vision input, please use mtmd_encode_chunk instead\n", .{"mtmd_encode_impl"});
        return 1;
    };

    const n_embd_out: c_int = @truncate(try ctx.nEmbdOut());
    const n_tokens_out = image_tokens.nTokens();
    try resizeZero(f32, out_embd, @as(usize, @bitCast(@as(isize, n_embd_out))) *% n_tokens_out);

    if (image_tokens.isPlaceholder()) {
        impl.logErr("%s: image tokens batch is placeholder\n", .{"mtmd_encode_impl"});
        return 1;
    }

    const ok = try clip.imageBatchEncode(ctx_clip, ctx.n_threads, &image_tokens.batch_f32, out_embd);
    return if (ok) 0 else 1;
}

/// Ports `mtmd_encode_chunk_impl` (mtmd.cpp:1799 @5266f24da).
fn encodeChunkImpl(ctx: *Context, chunk: *const InputChunk, out_embd: *std.ArrayList(f32)) Error!i32 {
    if (chunk.type == .text) {
        impl.logWrn("mtmd_encode_chunk has no effect for text chunks\n", .{});
        return 0;
    } else if (chunk.type == .image) {
        if (ctx.ctx_v == null) {
            impl.logErr("%s: model does not support vision input\n", .{"mtmd_encode_chunk_impl"});
            return 1;
        }
        const t = chunk.tokens_image orelse {
            impl.logErr("%s: image tokens are null\n", .{"mtmd_encode_chunk_impl"});
            return 1;
        };
        if (t.isPlaceholder()) {
            impl.logErr("%s: image tokens batch is placeholder\n", .{"mtmd_encode_chunk_impl"});
            return 1;
        }
        return encodeImpl(ctx, t, out_embd);
    } else if (chunk.type == .audio) {
        const ctx_a = ctx.ctx_a orelse {
            impl.logErr("%s: model does not support audio input\n", .{"mtmd_encode_chunk_impl"});
            return 1;
        };
        const t = chunk.tokens_audio orelse {
            impl.logErr("%s: audio tokens are null\n", .{"mtmd_encode_chunk_impl"});
            return 1;
        };
        if (t.isPlaceholder()) {
            impl.logErr("%s: audio tokens batch is placeholder\n", .{"mtmd_encode_chunk_impl"});
            return 1;
        }
        const n_mmproj_embd: c_int = @truncate(try ctx.nEmbdOut());
        try resizeZero(f32, out_embd, @as(usize, t.n_tokens) *% @as(usize, @bitCast(@as(isize, n_mmproj_embd))));
        const ok = try clip.imageBatchEncode(ctx_a, ctx.n_threads, &t.batch_f32, out_embd);
        return if (ok) 0 else 1;
    }

    impl.logErr("%s: unknown chunk type %d\n", .{ "mtmd_encode_chunk_impl", @as(c_int, @backingInt(chunk.type)) });
    return 1;
}

/// Ports `mtmd_encode_chunk` (mtmd.cpp:1844 @5266f24da): the non-batching
/// version.
pub fn mtmd_encode_chunk(ctx: *Context, chunk: *const InputChunk) callconv(.c) i32 {
    return encodeChunkImpl(ctx, chunk, &ctx.out_embd) catch |e| {
        impl.logErr("%s: error: %s\n", .{ "mtmd_encode_chunk", whatOf(e) });
        return 1;
    };
}

/// Ports `mtmd_encode` (mtmd.cpp:1854 @5266f24da).
pub fn mtmd_encode(ctx: *Context, image_tokens: *const ImageTokens) callconv(.c) i32 {
    return encodeImpl(ctx, image_tokens, &ctx.out_embd) catch |e| {
        impl.logErr("%s: error: %s\n", .{ "mtmd_encode", whatOf(e) });
        return 1;
    };
}

/// Ports `mtmd_get_output_embd` (mtmd.cpp:1863 @5266f24da).
pub fn mtmd_get_output_embd(ctx: *Context) callconv(.c) ?[*]f32 {
    return dataPtr(f32, &ctx.out_embd);
}

// -----------------------------------------------------------------------------
// audio generation

/// Ports `mtmd_gen_audio_get_info` (mtmd.cpp:1871 @5266f24da).
pub fn mtmd_gen_audio_get_info(ctx: *const Context) callconv(.c) GenAudioInfo {
    var info: GenAudioInfo = .{ .type = .none, .sample_rate = 0, .model_variant = "" };
    const g = ctx.ctx_gen_a orelse {
        info.type = .none;
        return info;
    };
    // `Loader.getString` keeps a NUL past the string's end.
    info.model_variant = @ptrCast(clip.getHparams(g).gen_model_variant.ptr);
    switch (clip.getProjectorType(g)) {
        .qwen3tts_gen => {
            info.type = .qwen3tts;
            info.sample_rate = 24000;
        },
        .pockettts_gen => {
            info.type = .pockettts;
            info.sample_rate = 24000;
        },
        else => info.type = .none,
    }
    return info;
}

/// Ports `mtmd_gen_inp_default` (mtmd.cpp:1895 @5266f24da).
pub fn mtmd_gen_inp_default(ctx: *const Context) callconv(.c) GenInp {
    var inp = std.mem.zeroes(GenInp);
    inp.type = .gen_code;
    inp.seed = std.math.maxInt(u32);
    const g = ctx.ctx_gen_a orelse return inp;

    switch (clip.getProjectorType(g)) {
        .qwen3tts_gen => {
            // https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/blob/main/generation_config.json
            inp.top_k = 50;
            inp.top_p = 1.0;
            inp.temp = 0.9; // TODO: handle this on graph
        },
        .pockettts_gen => {
            // https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/default_parameters.py
            inp.top_k = 50;
            inp.top_p = 1.0;
            inp.temp = 0.7;
        },
        else => {},
    }
    return inp;
}

/// `vector = std::move(other)`: `dst` takes `src`'s buffer.
fn moveList(comptime T: type, dst: *std.ArrayList(T), src: *std.ArrayList(T)) void {
    dst.deinit(gpa);
    dst.* = src.*;
    src.* = .empty;
}

/// Ports `mtmd_gen_audio_process_impl` (mtmd.cpp:1922 @5266f24da).
fn genAudioProcessImpl(ctx: *Context, inp: *const GenInp, out: *GenOut) Error!i32 {
    const ctx_clip = ctx.ctx_gen_a orelse {
        impl.logErr("%s: model does not support audio generation\n", .{"mtmd_gen_audio_process_impl"});
        return 1;
    };

    out.* = std.mem.zeroes(GenOut);

    if (inp.type == .gen_code) {
        const n_embd: usize = @bitCast(@as(isize, clip.nMmprojEmbd(ctx_clip)));

        var batch: impl.ImageF32Batch = .{ .is_audio = true };
        defer batch.deinit();
        {
            var hidden_state: impl.ImageF32 = .{};
            errdefer hidden_state.deinit();
            try hidden_state.setSize(.{ .width = @intCast(n_embd), .height = 1 }, false, true);
            try hidden_state.cpyBuf(inp.embd.?[0..n_embd]);
            try batch.entries.append(gpa, hidden_state);
        }

        var out_embd: std.ArrayList(f32) = .empty;
        defer out_embd.deinit(gpa);
        try resizeZero(f32, &out_embd, n_embd);
        var out_codes: std.ArrayList(i32) = .empty;
        defer out_codes.deinit(gpa);
        var out_feats: std.ArrayList(f32) = .empty;
        defer out_feats.deinit(gpa);
        var is_eos = false;

        var params: clip.EncodeParams = .{
            .imgs = &batch,
            .n_threads = ctx.n_threads,
            .gen_process = .gen_code,
            .out_embd = &out_embd,
            .out_codes = &out_codes,
            .out_feats = &out_feats,
            .code0 = inp.code0,
            .top_k = inp.top_k,
            .top_p = inp.top_p,
            .seed = inp.seed,
            .temp = inp.temp,
            .out_is_eos = &is_eos,
        };

        if (!try clip.encode(ctx_clip, &params)) {
            impl.logErr("%s: clip_encode failed (gen_code)\n", .{"mtmd_gen_audio_process_impl"});
            return 1;
        }

        moveList(f32, &ctx.gen_out_embd, &out_embd);
        moveList(i32, &ctx.gen_out_codes, &out_codes);
        moveList(f32, &ctx.gen_out_feats, &out_feats);

        out.embd = dataPtr(f32, &ctx.gen_out_embd);
        out.codes = dataPtr(i32, &ctx.gen_out_codes);
        out.n_codes = ctx.gen_out_codes.items.len;
        out.feats = dataPtr(f32, &ctx.gen_out_feats);
        out.n_feats = ctx.gen_out_feats.items.len;
        out.is_eos = is_eos;
        return 0;
    }

    // MTMD_GEN_PROCESS_TYPE_GEN_WAV
    const has_codes = inp.codes != null and inp.n_codes > 0;
    const has_feats = inp.feats != null and inp.n_feats > 0;
    if (has_codes == has_feats) {
        impl.logErr("%s: gen_wav requires exactly one of codes or feats\n", .{"mtmd_gen_audio_process_impl"});
        return 1;
    }
    var in_codes: std.ArrayList(i32) = .empty;
    defer in_codes.deinit(gpa);
    var in_feats: std.ArrayList(f32) = .empty;
    defer in_feats.deinit(gpa);
    if (has_codes) {
        try in_codes.appendSlice(gpa, inp.codes.?[0..inp.n_codes]);
    } else {
        try in_feats.appendSlice(gpa, inp.feats.?[0..inp.n_feats]);
    }
    var in_state: std.ArrayList(u8) = .empty;
    defer in_state.deinit(gpa);
    if (inp.state_data) |sd| try in_state.appendSlice(gpa, sd[0..inp.state_size]);

    // gen_wav has no hidden-state input, the batch entry is an unused placeholder
    var batch: impl.ImageF32Batch = .{ .is_audio = true };
    defer batch.deinit();
    {
        var dummy: impl.ImageF32 = .{};
        errdefer dummy.deinit();
        try dummy.setSize(.{ .width = 1, .height = 1 }, false, true);
        try dummy.cpyBuf(&.{0.0});
        try batch.entries.append(gpa, dummy);
    }

    var params: clip.EncodeParams = .{
        .imgs = &batch,
        .n_threads = ctx.n_threads,
        .gen_process = .gen_wav,
        // gen_wav draws no randomness, but keep the seed so it does not reseed mid-generation
        .seed = inp.seed,
        .codes = if (has_codes) &in_codes else null,
        .feats = if (has_feats) &in_feats else null,
        .out_audio = &ctx.gen_out_audio,
        .state_in = if (inp.state_data != null) &in_state else null,
        .state_out = &ctx.gen_out_state,
    };

    if (!try clip.encode(ctx_clip, &params)) {
        impl.logErr("%s: clip_encode failed (code2wav)\n", .{"mtmd_gen_audio_process_impl"});
        return 1;
    }

    out.audio = dataPtr(f32, &ctx.gen_out_audio);
    out.n_samples = ctx.gen_out_audio.items.len;
    out.state_data = dataPtr(u8, &ctx.gen_out_state);
    out.state_size = ctx.gen_out_state.items.len;

    return 0;
}

/// Ports `mtmd_gen_audio_process` (mtmd.cpp:2033 @5266f24da).
pub fn mtmd_gen_audio_process(ctx: *Context, inp: *const GenInp, out: *GenOut) callconv(.c) i32 {
    return genAudioProcessImpl(ctx, inp, out) catch |e| {
        impl.logErr("%s: error: %s\n", .{ "mtmd_gen_audio_process", whatOf(e) });
        return 1;
    };
}

// -----------------------------------------------------------------------------
// batches

/// Ports `mtmd_batch_init` (mtmd.cpp:2042 @5266f24da).
pub fn mtmd_batch_init(ctx: *Context) callconv(.c) ?*Batch {
    const b = gpa.create(Batch) catch impl.abortUncaught("std::bad_alloc", "std::bad_alloc");
    b.* = .{ .ctx = ctx };
    return b;
}

/// Ports `mtmd_batch_free` (mtmd.cpp:2046 @5266f24da).
pub fn mtmd_batch_free(batch: ?*Batch) callconv(.c) void {
    if (batch) |b| {
        b.entries.deinit(gpa);
        b.output_embd.deinit(gpa);
        gpa.destroy(b);
    }
}

/// Ports `mtmd_batch_add_chunk` (mtmd.cpp:2052 @5266f24da).
pub fn mtmd_batch_add_chunk(batch: *Batch, chunk: *const InputChunk) callconv(.c) i32 {
    if (chunk.type == .text) {
        impl.logErr("%s: text chunk is not supported in batch\n", .{"mtmd_batch_add_chunk"});
        return 1;
    }

    const ctx = batch.ctx.getClipCtx(chunk) orelse {
        impl.logErr("%s: model does not support input chunk type %d\n", .{ "mtmd_batch_add_chunk", @as(c_int, @backingInt(chunk.type)) });
        return 1;
    };

    if (batch.entries.items.len == 0) {
        // batch must have at least one chunk
        batch.entries.append(gpa, chunk) catch impl.abortUncaught("std::bad_alloc", "std::bad_alloc");
        return 0;
    }

    if (!clip.supportBatch(ctx)) {
        // if no batching support, batch can only have one single chunk
        return 2; // "batch too large" error code
    }

    const new_n_tokens: i32 = batch.nTokens() +% @as(i32, @truncate(@as(isize, @bitCast(mtmd_input_chunk_get_n_tokens(chunk)))));
    if (new_n_tokens > batch.ctx.batch_max_tokens) {
        return 2; // "batch too large" error code
    }

    const first_chunk = batch.entries.items[0];
    if (first_chunk.canBatchWith(chunk)) {
        batch.entries.append(gpa, chunk) catch impl.abortUncaught("std::bad_alloc", "std::bad_alloc");
        return 0;
    }

    return 3; // "cannot batch" error code
}

/// Ports `mtmd_batch_encode_impl` (mtmd.cpp:2089 @5266f24da).
fn batchEncodeImpl(batch: *Batch) Error!i32 {
    if (batch.entries.items.len == 0) {
        impl.logErr("%s: batch is empty\n", .{"mtmd_batch_encode_impl"});
        return 1;
    }
    for (batch.entries.items) |chunk| {
        if (chunk.isPlaceholder()) {
            impl.logErr("%s: chunk is placeholder\n", .{"mtmd_batch_encode_impl"});
            return 1;
        }
    }

    // represent the whole batch as one single chunk
    const batch_chunk = try copyChunk(batch.entries.items[0]);
    defer {
        batch_chunk.deinit();
        gpa.destroy(batch_chunk);
    }
    if (batch_chunk.tokens_image) |ti| {
        const b0_f32 = &ti.batch_f32;
        // copy all entries from other chunks into the first chunk's batch_f32
        // note: skip first entry because it's already in batch_chunk
        for (batch.entries.items[1..]) |chunk| {
            impl.ggmlAssert(chunk.tokens_image != null, "mtmd.cpp", 2109, "chunk->tokens_image");
            var b1_f32 = try chunk.tokens_image.?.batch_f32.clone();
            defer b1_f32.entries.deinit(gpa);
            try b0_f32.entries.appendSlice(gpa, b1_f32.entries.items);
        }
    } else if (batch_chunk.tokens_audio) |ta| {
        const b0_f32 = &ta.batch_f32;
        for (batch.entries.items[1..]) |chunk| {
            impl.ggmlAssert(chunk.tokens_audio != null, "mtmd.cpp", 2121, "chunk->tokens_audio");
            var b1_f32 = try chunk.tokens_audio.?.batch_f32.clone();
            defer b1_f32.entries.deinit(gpa);
            try b0_f32.entries.appendSlice(gpa, b1_f32.entries.items);
        }
    } else {
        impl.logErr("%s: unsupported chunk type\n", .{"mtmd_batch_encode_impl"});
        return 1;
    }

    impl.logDbg("%s: encoding batch with %zu entries and total %zu tokens\n", .{ "mtmd_batch_encode_impl", batch.entries.items.len, mtmd_input_chunk_get_n_tokens(batch_chunk) });
    return encodeChunkImpl(batch.ctx, batch_chunk, &batch.output_embd);
}

/// Ports `mtmd_batch_encode` (mtmd.cpp:2141 @5266f24da).
pub fn mtmd_batch_encode(batch: *Batch) callconv(.c) i32 {
    return batchEncodeImpl(batch) catch |e| {
        impl.logErr("%s: error: %s\n", .{ "mtmd_batch_encode", whatOf(e) });
        return 1;
    };
}

/// Ports `mtmd_batch_get_output_embd` (mtmd.cpp:2150 @5266f24da).
pub fn mtmd_batch_get_output_embd(batch: *Batch, chunk: ?*const InputChunk) callconv(.c) ?[*]f32 {
    if (batch.output_embd.items.len == 0) {
        impl.logErr("%s: batch has not been encoded yet\n", .{"mtmd_batch_get_output_embd"});
        return null;
    }
    var offset: usize = 0;
    const n_embd: usize = @bitCast(@as(isize, @intCast(batch.ctx.nEmbdOut() catch |e| impl.abortUncaught("std::runtime_error", whatOf(e)))));
    for (batch.entries.items) |cc| {
        const offset_prev = offset;
        const n_tokens = mtmd_input_chunk_get_n_tokens(cc);
        offset +%= n_tokens *% n_embd;
        impl.ggmlAssert(offset_prev < batch.output_embd.items.len, "mtmd.cpp", 2161, "offset_prev <  batch->output_embd.size()");
        impl.ggmlAssert(offset <= batch.output_embd.items.len, "mtmd.cpp", 2162, "offset      <= batch->output_embd.size()");
        if (cc == chunk) return batch.output_embd.items.ptr + offset_prev;
    }
    return null; // not found
}

/// Ports `mtmd_decode_use_non_causal` (mtmd.cpp:2170 @5266f24da).
pub fn mtmd_decode_use_non_causal(ctx: *const Context, chunk: ?*const InputChunk) callconv(.c) bool {
    var proj_type = ctx.projTypeV();
    if (chunk != null and chunk.?.type == .audio) proj_type = ctx.projTypeA();
    return switch (proj_type) {
        // E2B (n_embd = 1536) and E4B (n_embd = 2560) always use causal
        .gemma4v => ctx.n_embd_text != 1536 and ctx.n_embd_text != 2560,
        .gemma4uv, .gemma3, .deepseek4v => true,
        else => false,
    };
}

/// Ports `mtmd_decode_use_mrope` (mtmd.cpp:2188 @5266f24da).
pub fn mtmd_decode_use_mrope(ctx: *const Context) callconv(.c) bool {
    return ctx.pos_type == .mrope;
}

/// Ports `mtmd_support_vision` (mtmd.cpp:2192 @5266f24da).
pub fn mtmd_support_vision(ctx: *const Context) callconv(.c) bool {
    return ctx.ctx_v != null;
}

/// Ports `mtmd_support_audio` (mtmd.cpp:2196 @5266f24da).
pub fn mtmd_support_audio(ctx: *const Context) callconv(.c) bool {
    return ctx.ctx_a != null;
}

/// Ports `mtmd_get_audio_sample_rate` (mtmd.cpp:2200 @5266f24da).
pub fn mtmd_get_audio_sample_rate(ctx: *const Context) callconv(.c) c_int {
    const a = ctx.ctx_a orelse return -1;
    return clip.getHparams(a).audio_sample_rate;
}

/// Ports `mtmd_get_marker` (mtmd.cpp:2207 @5266f24da).
pub fn mtmd_get_marker(ctx: *const Context) callconv(.c) [*:0]const u8 {
    return ctx.media_marker.cStr();
}

// -----------------------------------------------------------------------------
// public API functions: mtmd_bitmap

fn oom() noreturn {
    impl.abortUncaught("std::bad_alloc", "std::bad_alloc");
}

/// Ports `mtmd_bitmap_init` (mtmd.cpp:2217 @5266f24da).
pub fn mtmd_bitmap_init(nx: u32, ny: u32, data: ?[*]const u8) callconv(.c) *Bitmap {
    return Bitmap.createImage(data, nx, ny) catch oom();
}

/// Ports `mtmd_bitmap_init_from_audio` (mtmd.cpp:2224 @5266f24da). The
/// count is truncated to the bitmap's `uint32_t` as the C++'s constructor
/// truncates it.
pub fn mtmd_bitmap_init_from_audio(n_samples: usize, data: ?[*]const f32) callconv(.c) *Bitmap {
    const bitmap = Bitmap.createAudio(@ptrCast(data), @truncate(n_samples)) catch oom();
    impl.ggmlAssert(bitmap.is_audio, "mtmd.cpp", 2227, "bitmap->is_audio");
    if (!bitmap.isPlaceholder()) {
        impl.ggmlAssert(bitmap.data.items.len == n_samples *% @sizeOf(f32), "mtmd.cpp", 2229, "bitmap->get_ro_buf().size() == n_samples * sizeof(float)");
    }
    return bitmap;
}

/// Ports `mtmd_bitmap_get_nx` (mtmd.cpp:2234 @5266f24da).
pub fn mtmd_bitmap_get_nx(bitmap: *const Bitmap) callconv(.c) u32 {
    return bitmap.nx;
}

/// Ports `mtmd_bitmap_get_ny` (mtmd.cpp:2238 @5266f24da).
pub fn mtmd_bitmap_get_ny(bitmap: *const Bitmap) callconv(.c) u32 {
    return bitmap.ny;
}

/// Ports `mtmd_bitmap_get_data` (mtmd.cpp:2242 @5266f24da).
pub fn mtmd_bitmap_get_data(bitmap: *const Bitmap) callconv(.c) ?[*]const u8 {
    if (bitmap.isPlaceholder()) return null;
    return bitmap.data.items.ptr;
}

/// Ports `mtmd_bitmap_get_n_bytes` (mtmd.cpp:2249 @5266f24da).
pub fn mtmd_bitmap_get_n_bytes(bitmap: *const Bitmap) callconv(.c) usize {
    if (bitmap.isPlaceholder()) return 0;
    return bitmap.data.items.len;
}

/// Ports `mtmd_bitmap_is_audio` (mtmd.cpp:2256 @5266f24da).
pub fn mtmd_bitmap_is_audio(bitmap: *const Bitmap) callconv(.c) bool {
    return bitmap.is_audio;
}

/// Ports `mtmd_bitmap_get_id` (mtmd.cpp:2260 @5266f24da).
pub fn mtmd_bitmap_get_id(bitmap: *const Bitmap) callconv(.c) [*:0]const u8 {
    return bitmap.id.cStr();
}

/// Ports `mtmd_bitmap_set_id` (mtmd.cpp:2264 @5266f24da).
pub fn mtmd_bitmap_set_id(bitmap: *Bitmap, id: ?[*:0]const u8) callconv(.c) void {
    bitmap.id.set(if (id) |s| std.mem.span(s) else "") catch oom();
}

/// Ports `mtmd_bitmap_set_mergeable` (mtmd.cpp:2272 @5266f24da).
pub fn mtmd_bitmap_set_mergeable(bitmap: *Bitmap, mergeable: bool) callconv(.c) void {
    bitmap.mergeable = mergeable;
}

/// Ports `mtmd_bitmap_init_lazy` (mtmd.cpp:2276 @5266f24da).
pub fn mtmd_bitmap_init_lazy(ctx: ?*const Context, id: ?[*:0]const u8, user_data: ?*anyopaque, callback: LazyCallback) callconv(.c) *Bitmap {
    _ = ctx; // reserved for future use
    const bitmap = Bitmap.createImage(null, 0, 0) catch oom();
    bitmap.lazy_callback = callback;
    bitmap.lazy_user_data = user_data;
    mtmd_bitmap_set_id(bitmap, id);
    return bitmap;
}

/// Ports `mtmd_bitmap_free` (mtmd.cpp:2288 @5266f24da).
pub fn mtmd_bitmap_free(bitmap: ?*Bitmap) callconv(.c) void {
    if (bitmap) |b| b.destroy();
}

// -----------------------------------------------------------------------------
// mtmd_input_chunks

/// Ports `mtmd_input_chunks_init` (mtmd.cpp:2296 @5266f24da).
pub fn mtmd_input_chunks_init() callconv(.c) *InputChunks {
    const chunks = gpa.create(InputChunks) catch oom();
    chunks.* = .{};
    return chunks;
}

/// Ports `mtmd_input_chunks_size` (mtmd.cpp:2300 @5266f24da).
pub fn mtmd_input_chunks_size(chunks: *const InputChunks) callconv(.c) usize {
    return chunks.entries.items.len;
}

/// Ports `mtmd_input_chunks_get` (mtmd.cpp:2304 @5266f24da).
pub fn mtmd_input_chunks_get(chunks: *const InputChunks, idx: usize) callconv(.c) ?*const InputChunk {
    if (idx >= chunks.entries.items.len) return null;
    return &chunks.entries.items[idx];
}

/// Ports `mtmd_input_chunks_free` (mtmd.cpp:2311 @5266f24da).
pub fn mtmd_input_chunks_free(chunks: ?*InputChunks) callconv(.c) void {
    if (chunks) |ch| {
        ch.deinit();
        gpa.destroy(ch);
    }
}

// -----------------------------------------------------------------------------
// mtmd_input_chunk

/// Ports `mtmd_input_chunk_get_type` (mtmd.cpp:2319 @5266f24da).
pub fn mtmd_input_chunk_get_type(chunk: *const InputChunk) callconv(.c) ChunkType {
    return chunk.type;
}

/// Ports `mtmd_input_chunk_get_tokens_text` (mtmd.cpp:2323 @5266f24da).
pub fn mtmd_input_chunk_get_tokens_text(chunk: *const InputChunk, n_tokens_output: *usize) callconv(.c) ?[*]const Token {
    if (chunk.type == .text) {
        n_tokens_output.* = chunk.tokens_text.items.len;
        return dataPtr(Token, &chunk.tokens_text);
    }
    n_tokens_output.* = 0;
    return null;
}

/// Ports `mtmd_input_chunk_get_tokens_image` (mtmd.cpp:2332 @5266f24da).
pub fn mtmd_input_chunk_get_tokens_image(chunk: *const InputChunk) callconv(.c) ?*const ImageTokens {
    if (chunk.type == .image) return chunk.tokens_image;
    return null;
}

/// Ports `mtmd_input_chunk_get_n_tokens` (mtmd.cpp:2339 @5266f24da).
pub fn mtmd_input_chunk_get_n_tokens(chunk: *const InputChunk) callconv(.c) usize {
    return switch (chunk.type) {
        .text => chunk.tokens_text.items.len,
        .image => mtmd_image_tokens_get_n_tokens(chunk.tokens_image.?),
        .audio => chunk.tokens_audio.?.n_tokens,
        _ => impl.abort("mtmd.cpp", 2347, "invalid chunk type"),
    };
}

/// Ports `mtmd_input_chunk_get_n_pos` (mtmd.cpp:2351 @5266f24da).
pub fn mtmd_input_chunk_get_n_pos(chunk: *const InputChunk) callconv(.c) c.llama_pos {
    return switch (chunk.type) {
        .text => @truncate(@as(isize, @intCast(chunk.tokens_text.items.len))),
        .image => mtmd_image_tokens_get_n_pos(chunk.tokens_image.?),
        .audio => @bitCast(chunk.tokens_audio.?.n_tokens),
        _ => impl.abort("mtmd.cpp", 2359, "invalid chunk type"),
    };
}

/// Ports `mtmd_input_chunk_get_id` (mtmd.cpp:2363 @5266f24da).
pub fn mtmd_input_chunk_get_id(chunk: *const InputChunk) callconv(.c) ?[*:0]const u8 {
    if (chunk.type == .image) return chunk.tokens_image.?.id.cStr();
    if (chunk.type == .audio) return chunk.tokens_audio.?.id.cStr();
    return null;
}

fn copyChunk(chunk: *const InputChunk) !*InputChunk {
    const copy = try gpa.create(InputChunk);
    copy.* = .{ .type = chunk.type };
    errdefer {
        copy.deinit();
        gpa.destroy(copy);
    }
    try copy.tokens_text.appendSlice(gpa, chunk.tokens_text.items);
    // copy the image tokens
    if (chunk.tokens_image) |t| copy.tokens_image = try t.cloneAlloc();
    // copy the audio tokens
    if (chunk.tokens_audio) |t| copy.tokens_audio = try t.cloneAlloc();
    return copy;
}

/// Ports `mtmd_input_chunk_copy` (mtmd.cpp:2372 @5266f24da).
pub fn mtmd_input_chunk_copy(chunk: *const InputChunk) callconv(.c) *InputChunk {
    return copyChunk(chunk) catch oom();
}

/// Ports `mtmd_input_chunk_free` (mtmd.cpp:2392 @5266f24da).
pub fn mtmd_input_chunk_free(chunk: ?*InputChunk) callconv(.c) void {
    if (chunk) |ch| {
        ch.deinit();
        gpa.destroy(ch);
    }
}

/// Ports `mtmd_input_chunk_save_impl` (mtmd.cpp:2399 @5266f24da): 0 on
/// success. The caller owns `out_buf`.
fn chunkSaveImpl(chunk: *const InputChunk, out_buf: *std.ArrayList(u8)) i32 {
    var ser = Ser.initWrite(serialization_version) catch |e| {
        impl.logErr("%s: %s\n", .{ "mtmd_input_chunk_save_impl", whatOf(e) });
        return -1;
    };
    chunk.serialize(&ser) catch |e| {
        ser.deinit();
        impl.logErr("%s: %s\n", .{ "mtmd_input_chunk_save_impl", whatOf(e) });
        return -1;
    };
    moveList(u8, out_buf, &ser.data);
    return 0;
}

/// Ports `mtmd_input_chunk_get_placeholder` (mtmd.cpp:2411 @5266f24da):
/// the chunk without its image or audio data.
pub fn mtmd_input_chunk_get_placeholder(chunk: *const InputChunk) callconv(.c) ?*InputChunk {
    // this is hacky, but still faster than copy the whole batch data
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    if (chunkSaveImpl(chunk, &buf) != 0) return null;
    return mtmd_input_chunk_load(buf.items.ptr, buf.items.len);
}

/// Ports `mtmd_input_chunk_save` (mtmd.cpp:2420 @5266f24da).
pub fn mtmd_input_chunk_save(chunk: *const InputChunk, out_buf: ?[*]u8, out_len: usize, expected_out_len: ?*usize) callconv(.c) i32 {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    if (chunkSaveImpl(chunk, &buf) != 0) return -1;
    if (expected_out_len) |p| p.* = buf.items.len;
    // caller is only querying the required size
    const dst = out_buf orelse return 0;
    if (out_len < buf.items.len) {
        impl.logErr("%s: out_buf is too small, need %zu bytes, got %zu\n", .{ "mtmd_input_chunk_save", buf.items.len, out_len });
        return -1;
    }
    @memcpy(dst[0..buf.items.len], buf.items);
    return 0;
}

/// Ports `mtmd_input_chunk_load` (mtmd.cpp:2440 @5266f24da).
pub fn mtmd_input_chunk_load(buf: ?[*]const u8, len: usize) callconv(.c) ?*InputChunk {
    return chunkLoad(if (len == 0) &.{} else buf.?[0..len]) catch |e| {
        impl.logErr("%s: %s\n", .{ "mtmd_input_chunk_load", whatOf(e) });
        return null;
    };
}

fn chunkLoad(buf: []const u8) Error!*InputChunk {
    var ser = try Ser.initRead(serialization_version, buf);
    defer ser.deinit();
    const chunk = try gpa.create(InputChunk);
    chunk.* = .{};
    errdefer {
        chunk.deinit();
        gpa.destroy(chunk);
    }
    try chunk.deserialize(&ser);
    return chunk;
}

// -----------------------------------------------------------------------------
// mtmd_image_tokens

/// Ports `mtmd_image_tokens_get_n_tokens` (mtmd.cpp:2454 @5266f24da).
pub fn mtmd_image_tokens_get_n_tokens(image_tokens: *const ImageTokens) callconv(.c) usize {
    return image_tokens.nTokens();
}

/// Ports `mtmd_image_tokens_get_nx` (mtmd.cpp:2458 @5266f24da).
pub fn mtmd_image_tokens_get_nx(image_tokens: *const ImageTokens) callconv(.c) usize {
    return image_tokens.nx;
}

/// Ports `mtmd_image_tokens_get_ny` (mtmd.cpp:2462 @5266f24da).
pub fn mtmd_image_tokens_get_ny(image_tokens: *const ImageTokens) callconv(.c) usize {
    return image_tokens.ny;
}

/// Ports `mtmd_image_tokens_get_decoder_pos` (mtmd.cpp:2466 @5266f24da).
/// The arithmetic is the C++'s: `llama_pos + size_t` in `size_t`,
/// truncated into each `uint32_t`.
pub fn mtmd_image_tokens_get_decoder_pos(image_tokens: *const ImageTokens, pos_0: c.llama_pos, i: usize) callconv(.c) DecoderPos {
    var pos: DecoderPos = undefined;
    const p0: usize = @bitCast(@as(isize, pos_0));
    const p0u: u32 = @bitCast(pos_0);
    switch (image_tokens.pos) {
        .mrope => {
            pos.t = p0u;
            pos.x = @truncate(p0 +% (i % image_tokens.nx));
            pos.y = @truncate(p0 +% (i / image_tokens.nx));
            pos.z = 0; // unused for now
        },
        .normal => {
            const v: u32 = @truncate(p0 +% i);
            pos = .{ .t = v, .x = v, .y = v, .z = v };
        },
        .hunyuanvl => {
            // HunyuanVL layout: [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
            // Total = 1 + ny*(nx+1) + 1. BOI and EOI use sequential positions in every dim;
            // content and row-newline tokens use (row, col) with XD-RoPE dim-3 = image_idx.
            const nx = image_tokens.nx;
            const n_total = image_tokens.nTokens();
            const v: u32 = @truncate(p0 +% i);
            if (i == 0) {
                // BOI
                pos = .{ .t = v, .x = v, .y = v, .z = v };
            } else if (i == @as(usize, n_total -% 1)) {
                // EOI
                pos = .{ .t = v, .x = v, .y = v, .z = v };
            } else {
                // content token at (row, col), or the trailing newline of a row (col == nx)
                //   section 0 = sequential, section 1 = w(col), section 2 = h(row), section 3 = image_count.
                // set_position_mrope_2d writes .y -> section 1 and .x -> section 2
                const offset: u32 = @as(u32, @truncate(i)) -% 1;
                const row = offset / (nx +% 1);
                const col = offset % (nx +% 1);
                pos.t = v;
                pos.x = row;
                pos.y = col;
                pos.z = image_tokens.image_idx;
            }
        },
        _ => impl.abort("mtmd.cpp", 2516, "invalid position type"),
    }
    return pos;
}

/// Ports `mtmd_image_tokens_get_id` (mtmd.cpp:2521 @5266f24da).
pub fn mtmd_image_tokens_get_id(image_tokens: *const ImageTokens) callconv(.c) [*:0]const u8 {
    return image_tokens.id.cStr();
}

/// Ports `mtmd_image_tokens_get_n_pos` (mtmd.cpp:2525 @5266f24da).
pub fn mtmd_image_tokens_get_n_pos(image_tokens: *const ImageTokens) callconv(.c) c.llama_pos {
    return switch (image_tokens.pos) {
        .mrope => @bitCast(@max(image_tokens.nx, image_tokens.ny)),
        .normal => @bitCast(image_tokens.nTokens()),
        // HunyuanVL: the sequential (dim-0) position advances by the full token count
        // (includes BOI/EOI and row newline tokens), not by max(nx, ny)
        .hunyuanvl => @bitCast(image_tokens.nTokens()),
        _ => impl.abort("mtmd.cpp", 2536, "invalid position type"),
    };
}

// -----------------------------------------------------------------------------
// test function

/// Ports `mtmd_test_create_input_chunks` (mtmd.cpp:2542 @5266f24da).
pub fn mtmd_test_create_input_chunks() callconv(.c) ?*InputChunks {
    const chunks = mtmd_input_chunks_init();

    // create a text chunk
    var chunk_text: InputChunk = .{ .type = .text };
    chunk_text.tokens_text.appendSlice(gpa, &.{ 1, 2, 3, 4, 5 }) catch oom();
    chunks.entries.append(gpa, chunk_text) catch oom();

    // create an image chunk
    const image_tokens = gpa.create(ImageTokens) catch oom();
    image_tokens.* = .{ .nx = 4, .ny = 4 };
    image_tokens.batch_f32.entries.appendNTimes(gpa, .{}, 16) catch oom();
    image_tokens.id.set("image_1") catch oom();
    chunks.entries.append(gpa, .{ .type = .image, .tokens_image = image_tokens }) catch oom();

    return chunks;
}

/// Ports `mtmd_log_set` (mtmd.cpp:2575 @5266f24da).
pub fn mtmd_log_set(log_callback: c.ggml_log_callback, user_data: ?*anyopaque) callconv(.c) void {
    impl.g_logger_state.log_callback = log_callback orelse impl.logCallbackDefault;
    impl.g_logger_state.log_callback_user_data = user_data;
}

/// Ports `mtmd_get_cap_from_file` (mtmd.cpp:2580 @5266f24da).
pub fn mtmd_get_cap_from_file(fname: [*:0]const u8) callconv(.c) Caps {
    const tmp = clip.getCap(std.mem.span(fname)) catch |e| {
        impl.logErr("%s: failed to get capabilities from file '%s': %s\n", .{ "mtmd_get_cap_from_file", fname, whatOf(e) });
        return .{ .inp_vision = false, .inp_audio = false };
    };
    return .{ .inp_audio = tmp.has_audio, .inp_vision = tmp.has_vision };
}

fn stubLogCallback(level: c.enum_ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = level;
    _ = text;
    _ = user_data;
    // do nothing
}

/// Ports `mtmd_get_memory_usage` (mtmd.cpp:2703 @5266f24da): weights and
/// compute per device, measured without allocating. The C++ returns a
/// `std::map` and has C++ linkage, so this is for Zig callers only; the
/// caller owns the map, empty on failure.
pub fn getMemoryUsage(mmproj_fname: [*:0]const u8, ctx_params_in: ContextParams) clip.DevSizeMap {
    var ctx_params = ctx_params_in;
    const saved_log_callback = impl.g_logger_state.log_callback;
    const saved_log_user_data = impl.g_logger_state.log_callback_user_data;

    ctx_params.progress_callback = null;

    mtmd_log_set(stubLogCallback, null); // suppress logging
    const ctx = Context.create(mmproj_fname, null, &ctx_params, true) catch |e| {
        mtmd_log_set(saved_log_callback, saved_log_user_data); // restore log callback
        impl.logErr("%s: error: %s\n", .{ "mtmd_get_memory_usage", whatOf(e) });
        return .{};
    };
    defer ctx.destroy();
    mtmd_log_set(saved_log_callback, saved_log_user_data); // restore log callback
    var total_mem: clip.DevSizeMap = .{};
    inline for (.{ ctx.ctx_v, ctx.ctx_a }) |cc| {
        if (cc) |x| {
            var m = clip.getMemUsage(x) catch oom();
            defer m.deinit();
            for (m.items.items) |e| total_mem.add(e.dev, e.size) catch oom();
        }
    }
    return total_mem;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a chunk saves and loads as a placeholder" {
    const chunks = mtmd_test_create_input_chunks().?;
    defer mtmd_input_chunks_free(chunks);
    const img = mtmd_input_chunks_get(chunks, 1).?;
    var need: usize = 0;
    try std.testing.expectEqual(@as(i32, 0), mtmd_input_chunk_save(img, null, 0, &need));
    const buf = try gpa.alloc(u8, need);
    defer gpa.free(buf);
    try std.testing.expectEqual(@as(i32, 0), mtmd_input_chunk_save(img, buf.ptr, buf.len, null));
    const loaded = mtmd_input_chunk_load(buf.ptr, buf.len).?;
    defer mtmd_input_chunk_free(loaded);
    try std.testing.expectEqual(ChunkType.image, mtmd_input_chunk_get_type(loaded));
    try std.testing.expectEqual(@as(usize, 16 * 16), mtmd_input_chunk_get_n_tokens(loaded));
    try std.testing.expectEqualStrings("image_1", std.mem.span(mtmd_input_chunk_get_id(loaded).?));
    // a truncated buffer is refused, not read past; its log line is silenced
    mtmd_log_set(stubLogCallback, null);
    defer mtmd_log_set(null, null);
    try std.testing.expectEqual(@as(?*InputChunk, null), mtmd_input_chunk_load(buf.ptr, buf.len - 1));
}

test "decoder positions" {
    var t: ImageTokens = .{ .nx = 3, .ny = 2, .pos = .mrope };
    const p = mtmd_image_tokens_get_decoder_pos(&t, 10, 4);
    try std.testing.expectEqual(DecoderPos{ .t = 10, .x = 11, .y = 11, .z = 0 }, p);
    try std.testing.expectEqual(@as(c.llama_pos, 3), mtmd_image_tokens_get_n_pos(&t));
}
