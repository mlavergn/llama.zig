//! libmtmd's public API: the context, bitmaps, tokenizing a prompt with
//! media markers into chunks, encoding them, and audio generation.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/mtmd.cpp`, the C types of
//! `llama.cpp/tools/mtmd/mtmd.h` and `mtmd_input_part` of
//! `llama.cpp/tools/mtmd/mtmd-internal.h`, at v0.9.0 (`cd959bae9`). Each
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
const graph = @import("graph.zig");
const lapi = @import("llama").api;
const LlamaException = @import("llama").impl.Exception;
const c = impl.c;
const gpa = impl.gpa;

const Token = c.llama_token;
const token_null: Token = -1; // LLAMA_TOKEN_NULL

extern fn snprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, ...) c_int;
extern fn free(p: ?*anyopaque) void;
extern fn expf(x: f32) f32;
extern fn sinf(x: f32) f32;
extern fn cosf(x: f32) f32;
extern fn lround(x: f64) c_long;
// chatterbox's host voice encoder and perceiver (cbx_ref_cond)
extern fn tanhf(x: f32) f32;
extern fn exp(x: f64) f64;

/// `MTMD_SERIALIZATION_VERSION` (mtmd.cpp:34 @cd959bae9).
const serialization_version: u64 = 4;

/// `MTMD_SERIALIZATION_VERSION_MIN` (mtmd.cpp:37 @cd959bae9): the oldest
/// compat version that can be loaded.
const serialization_version_min: u64 = 2;

// -----------------------------------------------------------------------------
// mtmd.h

/// Ports `mtmd_input_chunk_type` (mtmd.h:54 @cd959bae9).
pub const ChunkType = enum(c_int) { text, image, audio, _ };
const chunk_type_count: u32 = 3;

/// Ports `mtmd_input_text` (mtmd.h:78 @cd959bae9).
pub const InputText = extern struct {
    text: ?[*]const u8,
    text_len: usize,
    add_special: bool,
    parse_special: bool,
};

/// Ports `mtmd_input_part` (mtmd.h:85 @cd959bae9).
pub const InputPart = extern struct {
    // only text or bitmap can be set, not both
    text: ?*const InputText,
    bitmap: ?*const Bitmap,
};

pub const ProgressCallback = ?*const fn (f32, ?*anyopaque) callconv(.c) bool;

/// Ports `mtmd_context_params` (mtmd.h:108 @cd959bae9).
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

/// Ports `mtmd_bitmap_lazy_callback` (mtmd.h:240 @cd959bae9).
pub const LazyCallback = ?*const fn (usize, ?*anyopaque, *?*Bitmap, *?[*:0]u8) callconv(.c) c_int;

/// Ports `mtmd_decoder_pos` (mtmd.h:306 @cd959bae9).
pub const DecoderPos = extern struct { t: u32, x: u32, y: u32, z: u32 };

/// Ports `mtmd_caps` (mtmd.h:392 @cd959bae9).
pub const Caps = extern struct { inp_vision: bool, inp_audio: bool, gen_audio: bool };

/// Ports `mtmd_gen_audio_type` (mtmd.h:463 @cd959bae9).
pub const GenAudioType = enum(c_int) { none, qwen3tts, pockettts, soprano, lfm3a, chatterbox };

/// Ports `mtmd_gen_audio_info` (mtmd.h:472 @cd959bae9).
pub const GenAudioInfo = extern struct {
    type: GenAudioType,
    sample_rate: i32, // in Hz, for example 24000 for qwen3tts
    model_variant: ?[*:0]const u8, // name of the weight variant, can be nullptr if not applicable
    n_voices: i32, // predefined voices, see mtmd_gen_audio_voice_name()
};

/// Ports `mtmd_gen_process_type` (mtmd.h:491 @cd959bae9).
pub const GenProcessType = enum(c_int) { gen_code, gen_wav, _ };

/// Ports `mtmd_gen_inp` (mtmd.h:499 @cd959bae9).
pub const GenInp = extern struct {
    type: GenProcessType,
    code0: i32,
    embd: ?[*]f32,
    top_k: i32,
    top_p: f32,
    temp: f32,
    penalty_repeat: f32, // repetition penalty of the sampled codes, <= 1 is off
    penalty_last_n: i32, // frames the repetition penalty looks back on
    codes: ?[*]i32,
    n_codes: usize,
    feats: ?[*]const f32,
    n_feats: usize,
};

/// Ports `mtmd_gen_out` (mtmd.h:520 @cd959bae9).
pub const GenOut = extern struct {
    codes: ?[*]const i32,
    n_codes: usize,
    feats: ?[*]const f32,
    n_feats: usize,
    embd: ?[*]const f32,
    is_eos: bool,
    audio: ?[*]const f32,
    n_samples: usize,
};

/// Ports `mtmd_gen_state_params` (mtmd.h:537 @cd959bae9).
pub const GenStateParams = extern struct { seed: u32 };

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

/// Ports `mtmd_serialization` (mtmd.cpp:39 @cd959bae9): a byte buffer
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
        if (ver_in < serialization_version_min or ver_in > version) return clip.throwStr("version mismatch");
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

/// Ports `serialize` (mtmd.cpp:110 @cd959bae9), a member of `clip_image_f32`. `buf` is
/// intentionally not serialized: a loaded image is always a placeholder.
fn serializeImage(img: *const impl.ImageF32, ser: *Ser) !void {
    try ser.write(bool, img.add_viewsep);
    try ser.write(bool, img.add_newline);
    try ser.write(i32, img.lead_pad);
    try ser.write(i32, @backingInt(img.suffix_type));
    try ser.write(i32, img.anyres.grid_x);
    try ser.write(i32, img.anyres.grid_y);
    try ser.write(i32, img.anyres.orig_nx);
    try ser.write(i32, img.anyres.orig_ny);
    try ser.write(i32, img.nx_);
    try ser.write(i32, img.ny_);
}

/// Ports `deserialize` (mtmd.cpp:124 @cd959bae9), a member of `clip_image_f32`.
fn deserializeImage(img: *impl.ImageF32, ser: *Ser) Error!void {
    img.add_viewsep = try ser.read(bool);
    img.add_newline = try ser.read(bool);
    img.lead_pad = try ser.read(i32);
    if (ser.version >= 3) {
        const suffix_raw = try ser.read(i32);
        if (suffix_raw < 0 or suffix_raw >= @backingInt(impl.SuffixType.count)) return clip.throwStr("invalid suffix type");
        img.suffix_type = @fromBackingInt(suffix_raw);
        img.anyres.grid_x = try ser.read(i32);
        img.anyres.grid_y = try ser.read(i32);
        img.anyres.orig_nx = try ser.read(i32);
        img.anyres.orig_ny = try ser.read(i32);
    }
    img.nx_ = try ser.read(i32);
    img.ny_ = try ser.read(i32);
    img.buf.clearRetainingCapacity(); // always a placeholder after loading
}

/// Ports `serialize` (mtmd.cpp:144 @cd959bae9), a member of `clip_image_f32_batch`.
fn serializeBatch(b: *const impl.ImageF32Batch, ser: *Ser) !void {
    try ser.write(bool, b.is_audio);
    try ser.write(u64, b.entries.items.len);
    for (b.entries.items) |*e| try serializeImage(e, ser);
}

/// Ports `deserialize` (mtmd.cpp:152 @cd959bae9), a member of `clip_image_f32_batch`.
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

/// A `std::shared_ptr<const clip_gen_ref>`: shared by the voice bitmap, the
/// track it tokenizes to, and every copy of either; `refs` counts the owners.
pub const SharedGenRef = struct {
    refs: std.atomic.Value(usize) = .init(1),
    ref: clip.GenRef = .{},

    /// `std::make_shared<const clip_gen_ref>(r)`.
    fn make(r: *const clip.GenRef) !*SharedGenRef {
        const self = try gpa.create(SharedGenRef);
        self.* = .{};
        errdefer gpa.destroy(self);
        self.ref = try r.clone();
        return self;
    }

    fn retain(self: *SharedGenRef) *SharedGenRef {
        _ = self.refs.fetchAdd(1, .monotonic);
        return self;
    }

    fn release(self: *SharedGenRef) void {
        if (self.refs.fetchSub(1, .acq_rel) != 1) return;
        self.ref.deinit();
        gpa.destroy(self);
    }
};

/// Ports `mtmd_bitmap` (mtmd.cpp:173 @cd959bae9). An image is RGB
/// interleaved, `nx * ny * 3` bytes; audio is `nx` F32 samples and `ny = 1`;
/// a voice is `ny` F32 rows of `nx`. No data marks a placeholder.
pub const Bitmap = struct {
    nx: u32 = 0,
    ny: u32 = 0,
    id: Str = .{}, // optional user-defined id, for ex: can be set to image hash, useful for KV cache tracking
    is_audio: bool = false, // true if the bitmap is audio
    is_voice: bool = false,
    gen_ref: ?*SharedGenRef = null, // gen-side reference of a predefined voice, if any
    mergeable: bool = false, // [QWEN_VIDEO] set only on frames of the same video
    is_video: bool = false, // see mtmd_bitmap_set_is_video()

    // lazy-loaded bitmap
    lazy_callback: LazyCallback = null,
    lazy_user_data: ?*anyopaque = null,

    data: std.ArrayList(u8) = .empty,

    /// The image constructor `mtmd_bitmap` (mtmd.cpp:187 @cd959bae9).
    fn createImage(data: ?[*]const u8, nx: u32, ny: u32) !*Bitmap {
        const self = try gpa.create(Bitmap);
        self.* = .{ .nx = nx, .ny = ny, .is_audio = false };
        if (data) |d| try self.data.appendSlice(gpa, d[0 .. @as(usize, nx) * ny * 3]);
        return self;
    }

    /// The audio constructor `mtmd_bitmap` (mtmd.cpp:196 @cd959bae9).
    fn createAudio(data: ?[*]const u8, n_samples: u32) !*Bitmap {
        const self = try gpa.create(Bitmap);
        self.* = .{ .nx = n_samples, .ny = 1, .is_audio = true };
        if (data) |d| try self.data.appendSlice(gpa, d[0 .. @as(usize, n_samples) * @sizeOf(f32)]);
        return self;
    }

    /// Ports `init_voice` (mtmd.cpp:205 @cd959bae9), a member of `mtmd_bitmap`.
    fn initVoice(name: []const u8, voice: *const clip.GenVoice, n_embd: u32) !*Bitmap {
        const rows = voice.rows.items;
        const n_rows: u32 = @intCast(rows.len / n_embd);
        const self = try createImage(null, n_embd, n_rows);
        errdefer self.destroy();
        self.is_voice = true;
        if (voice.ref.tokens.items.len != 0) self.gen_ref = try SharedGenRef.make(&voice.ref);
        try self.id.set(name);
        try self.data.appendSlice(gpa, std.mem.sliceAsBytes(rows[0 .. @as(usize, n_embd) * n_rows]));
        return self;
    }

    fn destroy(self: *Bitmap) void {
        if (self.gen_ref) |g| g.release();
        self.id.deinit();
        self.data.deinit(gpa);
        gpa.destroy(self);
    }

    pub fn isPlaceholder(self: *const Bitmap) bool {
        return self.data.items.len == 0;
    }

    /// `is_video_frame` (mtmd.cpp:230 @cd959bae9).
    fn isVideoFrame(self: *const Bitmap) bool {
        return self.mergeable and self.is_video;
    }

    /// `can_merge_with` (mtmd.cpp:234 @cd959bae9): [QWEN_VIDEO] can
    /// (temporal) merge if both are images with same size.
    fn canMergeWith(self: *const Bitmap, other: *const Bitmap) bool {
        return self.mergeable and other.mergeable and
            !self.is_audio and !other.is_audio and
            self.nx == other.nx and self.ny == other.ny;
    }
};

/// Ports `mtmd_pos_type` (mtmd.cpp:246 @cd959bae9): position indexing for
/// the decoder model.
pub const PosType = enum(u32) {
    normal, // number of positions equals to number of tokens
    mrope, // qwen-vl mrope style, each image takes max(t,h,w) position indexes
    hunyuanvl, // HunyuanVL mrope + BOI/EOI/newline layout with XD-RoPE dim-3
    sensenova_u1, // U1 uses zero-based H/W and one temporal position per image
    canvas, // MiniCPM-V 4.7: overview + slices in one chunk, sharing one 2D canvas (see ImageTokens.canvasTileGrid)
    _,
};
const pos_type_count: u32 = 5;

/// Ports `mtmd_image_tokens` (mtmd.cpp:256 @cd959bae9).
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

    /// `n_tokens` (mtmd.cpp:262 @cd959bae9).
    pub fn nTokens(self: *const ImageTokens) u32 {
        if (self.pos == .hunyuanvl) {
            // [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
            return (self.nx +% 1) *% self.ny +% 2;
        }
        if (self.pos == .canvas) {
            var n: u32 = 0;
            for (self.batch_f32.entries.items, 0..) |*e, k| {
                const g = self.canvasTileGrid(k);
                n +%= g[0] *% g[1] +% @as(u32, @intCast(impl.suffixRows(e.suffix_type).len));
            }
            return n;
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

    /// `canvas_tile_grid` (mtmd.cpp:302 @cd959bae9): for `.canvas`, the
    /// entries are [overview, slices row by row] and nx/ny is the token grid
    /// of the last entry; the token grid (w, h) of entry `k`, scaled from its
    /// pixel size.
    fn canvasTileGrid(self: *const ImageTokens, k: usize) [2]u32 {
        const entries = self.batch_f32.entries.items;
        const ref = &entries[entries.len - 1];
        const e = &entries[k];
        return .{
            (@as(u32, @bitCast(e.nx())) *% self.nx) / @as(u32, @bitCast(ref.nx())),
            (@as(u32, @bitCast(e.ny())) *% self.ny) / @as(u32, @bitCast(ref.ny())),
        };
    }

    /// `can_batch_with` (mtmd.cpp:308 @cd959bae9): a canvas chunk holds a
    /// whole image group, its layout is not given by nx/ny alone.
    fn canBatchWith(self: *const ImageTokens, other: *const ImageTokens) bool {
        return self.nx == other.nx and self.ny == other.ny and self.pos == other.pos and self.pos != .canvas;
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

    /// `serialize` (mtmd.cpp:325 @cd959bae9).
    fn serialize(self: *const ImageTokens, ser: *Ser) !void {
        try ser.write(u32, self.nx);
        try ser.write(u32, self.ny);
        try ser.write(u32, @backingInt(self.pos));
        try ser.write(u32, self.image_idx);
        try ser.write(u32, self.n_temporal_merge);
        try ser.writeString(self.id.slice());
        try serializeBatch(&self.batch_f32, ser);
    }

    /// `deserialize` (mtmd.cpp:335 @cd959bae9).
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

/// Ports `mtmd_audio_track` (mtmd.cpp:353 @cd959bae9): an audio track split
/// across chunks (e.g. the soundtrack of a video): it is encoded once, as one
/// clip, and every chunk takes its rows of it. Shared as the C++'s `shared_ptr`
/// shares it, `refs` counting the owners.
pub const AudioTrack = struct {
    refs: std.atomic.Value(usize) = .init(1),
    batch_f32: impl.ImageF32Batch = .{}, // the mel of the whole track
    n_tokens: u32 = 0,
    embd: std.ArrayList(f32) = .empty, // filled by the first encode
    gen_ref: ?*SharedGenRef = null, // gen-side reference of a predefined voice, if any

    fn ref(self: *AudioTrack) *AudioTrack {
        _ = self.refs.fetchAdd(1, .monotonic);
        return self;
    }

    fn unref(self: *AudioTrack) void {
        if (self.refs.fetchSub(1, .acq_rel) != 1) return;
        if (self.gen_ref) |g| g.release();
        self.batch_f32.deinit();
        self.embd.deinit(gpa);
        gpa.destroy(self);
    }
};

/// Ports `mtmd_audio_tokens` (mtmd.cpp:360 @cd959bae9).
pub const AudioTokens = struct {
    n_tokens: u32 = 0, // number of tokens
    batch_f32: impl.ImageF32Batch = .{}, // preprocessed image patches
    id: Str = .{}, // optional user-defined ID, useful for KV cache tracking

    // a slice of a track: the rows [row_begin, row_begin + n_tokens) of it, batch_f32 is unused
    track: ?*AudioTrack = null,
    row_begin: u32 = 0,

    fn destroy(self: *AudioTokens) void {
        if (self.track) |t| t.unref();
        self.batch_f32.deinit();
        self.id.deinit();
        gpa.destroy(self);
    }

    /// true if one of entries in batch_f32 is a placeholder
    fn isPlaceholder(self: *const AudioTokens) bool {
        if (self.track) |t| {
            if (t.batch_f32.entries.items.len == 0) return t.embd.items.len == 0; // a voice: its rows are not serialized
        }
        const batch = if (self.track) |t| &t.batch_f32 else &self.batch_f32;
        for (batch.entries.items) |*e| if (e.isPlaceholder()) return true;
        return false;
    }

    fn cloneAlloc(self: *const AudioTokens) !*AudioTokens {
        const out = try gpa.create(AudioTokens);
        out.* = .{ .n_tokens = self.n_tokens, .row_begin = self.row_begin };
        out.batch_f32 = try self.batch_f32.clone();
        out.id = try self.id.clone();
        if (self.track) |t| out.track = t.ref();
        return out;
    }

    /// `serialize` (mtmd.cpp:392 @cd959bae9).
    fn serialize(self: *const AudioTokens, ser: *Ser) !void {
        try ser.write(u32, self.n_tokens);
        try ser.writeString(self.id.slice());
        try serializeBatch(&self.batch_f32, ser);
        // a slice keeps its own copy of the track
        try ser.write(bool, self.track != null);
        if (self.track) |t| {
            try ser.write(u32, self.row_begin);
            try ser.write(u32, t.n_tokens);
            try serializeBatch(&t.batch_f32, ser);
        }
    }

    /// `deserialize` (mtmd.cpp:405 @cd959bae9).
    fn deserialize(self: *AudioTokens, ser: *Ser) Error!void {
        self.n_tokens = try ser.read(u32);
        try ser.readString(&self.id);
        try deserializeBatch(&self.batch_f32, ser);
        if (self.track) |t| t.unref();
        self.track = null;
        if (ser.version >= 4 and try ser.read(bool)) {
            self.row_begin = try ser.read(u32);
            const t = try gpa.create(AudioTrack);
            t.* = .{};
            self.track = t;
            t.n_tokens = try ser.read(u32);
            try deserializeBatch(&t.batch_f32, ser);
        }
    }
};

/// Ports `mtmd_input_chunk` (mtmd.cpp:420 @cd959bae9).
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

    /// `can_batch_with` (mtmd.cpp:426 @cd959bae9).
    fn canBatchWith(self: *const InputChunk, other: *const InputChunk) bool {
        if (self.type != other.type) return false;
        if (self.tokens_image != null and other.tokens_image != null) {
            return self.tokens_image.?.canBatchWith(other.tokens_image.?);
        }
        // TODO: allow batching audio chunks of the same size
        return false;
    }

    /// `is_placeholder` (mtmd.cpp:440 @cd959bae9).
    fn isPlaceholder(self: *const InputChunk) bool {
        if (self.type == .image) {
            return self.tokens_image != null and self.tokens_image.?.isPlaceholder();
        } else if (self.type == .audio) {
            return self.tokens_audio != null and self.tokens_audio.?.isPlaceholder();
        }
        return false;
    }

    /// `serialize` (mtmd.cpp:449 @cd959bae9).
    fn serialize(self: *const InputChunk, ser: *Ser) !void {
        try ser.write(u32, @bitCast(@backingInt(self.type)));

        try ser.write(u64, self.tokens_text.items.len);
        for (self.tokens_text.items) |tok| try ser.write(i32, tok);

        try ser.write(bool, self.tokens_image != null);
        if (self.tokens_image) |t| try t.serialize(ser);

        try ser.write(bool, self.tokens_audio != null);
        if (self.tokens_audio) |t| try t.serialize(ser);
    }

    /// `deserialize` (mtmd.cpp:468 @cd959bae9).
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

/// Ports `mtmd_input_chunks` (mtmd.cpp:510 @cd959bae9).
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

/// Ports `mtmd_batch` (mtmd.cpp:514 @cd959bae9).
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

/// Ports `mtmd_slice_tmpl` (mtmd.cpp:530 @cd959bae9): where some llava-uhd
/// models put special tokens around image embeddings.
const SliceTmpl = enum { none, minicpmv_2_5, minicpmv_2_6, llama4, idefics3, lfm2, step3vl };

/// Ports `mtmd_default_marker` (mtmd.cpp:540 @cd959bae9).
pub fn mtmd_default_marker() callconv(.c) [*:0]const u8 {
    return "<__media__>";
}

/// Ports `mtmd_get_clip_flash_attn_type` (mtmd.cpp:544 @cd959bae9).
fn clipFlashAttnType(t: c.enum_llama_flash_attn_type) impl.FlashAttnType {
    return switch (t) {
        c.LLAMA_FLASH_ATTN_TYPE_AUTO => .auto,
        c.LLAMA_FLASH_ATTN_TYPE_DISABLED => .disabled,
        c.LLAMA_FLASH_ATTN_TYPE_ENABLED => .enabled,
        else => .auto,
    };
}

/// Ports `mtmd_context_params_default` (mtmd.cpp:553 @cd959bae9).
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

/// Ports `cbx_is_mtl` (mtmd.cpp:575 @cd959bae9): chatterbox multilingual
/// has a perceiver in its conditioning encoder (speaker encoding context).
fn cbxIsMtl(ctx_a: *clip.Ctx) bool {
    return clip.cbxReadTensor(ctx_a, "a.cenc.perceiver.pre_attention_query", null, 0) > 0;
}

/// Ports `mtmd_context` (mtmd.cpp:579 @cd959bae9).
pub const Context = struct {
    ctx_v: ?*clip.Ctx = null, // vision
    ctx_a: ?*clip.Ctx = null, // audio
    out_embd: std.ArrayList(f32) = .empty, // image embedding vector

    // generation context
    ctx_gen_a: ?*clip.Ctx = null, // audio

    // gen-side reference data of the last encoded chunk (chatterbox), see mtmd_gen_state_set_ref()
    gen_ref: clip.GenRef = .{},

    print_timings: bool = false,
    n_threads: c_int = 0,
    media_marker: Str = .{},
    n_embd_text: c_int = -1, // -1 means llm context not provided, skip checking this
    vocab: ?*const c.struct_llama_vocab = null, // can be nullptr if text_model is not provided
    text_model: ?*const c.struct_llama_model = null, // for the token embeddings of mtmd_audio_stream
    // The C++ leaves this indeterminate without a text model; `normal` is
    // what fresh heap memory gives.
    pos_type: PosType = .normal,

    // these are not token, but strings used to mark the beginning and end of image/audio embeddings
    img_beg: []const u8 = "",
    img_end: []const u8 = "",
    aud_beg: []const u8 = "",
    aud_end: []const u8 = "",
    vid_beg: []const u8 = "", // wraps a whole video (lazy bitmap with is_video)
    vid_end: []const u8 = "",

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

    // MiniCPM-V 4.6/4.7 prepends an <image_id>N</image_id> tag before <image>
    use_image_id: bool = false,

    // string template for slice image delimiters with row/col (idefics3)
    sli_img_start_tmpl: ?[*:0]const u8 = null,

    audio_preproc: ?audio.Preprocessor = null,
    image_preproc: ?image.Preprocessor = null,

    // batching
    batch_max_tokens: i32 = 0,

    /// The constructor `mtmd_context` (mtmd.cpp:634 @cd959bae9). A throw leaks the clip
    /// contexts as the C++'s does -- its destructor never runs for an
    /// object whose constructor threw, and the pointers are raw -- so that
    /// freeing them logs nothing upstream does not.
    fn create(mmproj_fname: [*:0]const u8, text_model: ?*const c.struct_llama_model, ctx_params: *const ContextParams, no_alloc: bool) Error!*Context {
        const self = try gpa.create(Context);
        self.* = .{};
        // errdefers run last-registered first: the members go before the
        // object that holds them
        errdefer gpa.destroy(self);
        errdefer self.deinitMembers();

        self.print_timings = ctx_params.print_timings;
        self.n_threads = ctx_params.n_threads;
        // `std::string(nullptr)` is undefined; an absent marker reads as empty.
        try self.media_marker.set(if (ctx_params.media_marker) |m| std.mem.span(m) else "");
        self.n_embd_text = if (text_model != null) lapi.model.llama_model_n_embd_inp(text_model) else -1;
        self.vocab = if (text_model != null) lapi.model.llama_model_get_vocab(text_model) else null;
        self.text_model = text_model;
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
        // soprano and pocket-tts without voice cloning have no encoder
        if (self.ctx_v == null and self.ctx_a == null and self.ctx_gen_a == null) {
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
        // a gen-only mmproj is validated below
        const n_embd_clip = if (self.ctx_v != null or self.ctx_a != null) clip.nMmprojEmbd(self.ctx_v orelse self.ctx_a.?) else self.n_embd_text;
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

    /// Ports `init_vision` (mtmd.cpp:734 @cd959bae9).
    fn initVision(self: *Context) Error!void {
        impl.ggmlAssert(self.ctx_v != null, "mtmd.cpp", 735, "ctx_v != nullptr");
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
            .minicpmv4_6, .minicpmv4_7, .minicpm_ocr_1_0 => {
                self.slice_tmpl = .minicpmv_2_6;
                try setTokens(&self.tok_ov_img_start, &.{self.lookupToken("<image>")});
                try setTokens(&self.tok_ov_img_end, &.{self.lookupToken("</image>")});
                try setTokens(&self.tok_sli_img_start, &.{self.lookupToken("<slice>")});
                try setTokens(&self.tok_sli_img_end, &.{self.lookupToken("</slice>")});
                try setTokens(&self.tok_row_end, &.{self.lookupToken("\n")});
                self.tok_row_end_trail = false; // no trailing end-of-row token
                self.ov_img_first = true;
                self.use_image_id = true;
                kind = if (proj == .minicpm_ocr_1_0) .minicpm_ocr_1_0 else .minicpmv;
            },
            .qwen3vl => {
                // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                self.img_beg = "<|vision_start|>";
                self.img_end = "<|vision_end|>";
                kind = .qwen3vl;
            },
            .qwen2vl, .qwen25vl, .ling3vl, .mimovl => {
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
            .sensenova_u1 => {
                self.img_beg = "<img>";
                self.img_end = "</img>";
                kind = .dyn_size;
            },
            .muse_glimmer => {
                self.img_beg = "<|image_start|>";
                self.img_end = "<|image_end|>";
                kind = .muse_glimmer;
            },
            .lfm3vl => {
                // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                self.img_beg = "<|vision_start|>";
                self.img_end = "<|vision_end|>";
                self.vid_beg = self.img_beg;
                self.vid_end = self.img_end;
                kind = .lfm3vl;
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
                // https://github.com/huggingface/transformers/blob/1cd110c6cb6a6237614130c470e9a902dbc1a4bd/docs/source/en/model_doc/pixtral.md
                self.img_end = "[IMG_END]";
                kind = if (hparams.image_longest_edge > 0) .longest_edge else .dyn_size;
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
            .cohere2v => {
                // <|START_OF_IMG|> (tile embeddings) <|IMG_LINE_BREAK|> ... <|END_OF_IMG|>
                self.img_beg = "<|START_OF_IMG|>";
                self.img_end = "<|END_OF_IMG|>";
                try setTokens(&self.tok_sli_img_end, &.{self.lookupToken("<|IMG_LINE_BREAK|>")});
                try setTokens(&self.tok_ov_img_end, self.tok_sli_img_end.items);
                self.ov_img_first = false;
                kind = .cohere2v;
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
            .deepseek41v => {
                // as deepseek4v: markers come from the encoder, the block is decoded causally
                kind = .deepseek41v;
            },
            .dots_ocr, .dots3note_v => {
                // <|img|> ... (image embeddings) ... <|endofimg|>
                self.img_beg = "<|img|>";
                self.img_end = "<|endofimg|>";
                kind = .dyn_size;
            },
            .nemotron_v2_vl => kind = .fixed_size,
            .nemotron_omni => {
                self.img_beg = "<img>";
                self.img_end = "</img>";
                kind = .dyn_size;
            },
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
            .d1omni_v => {
                // same tiles and thumbnail as lfm2, without separator tokens
                kind = .lfm2;
            },
            .glm4v => {
                // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
                self.img_beg = "<|begin_of_image|>";
                self.img_end = "<|end_of_image|>";
                kind = .dyn_size;
            },
            .glm5v => {
                // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
                self.img_beg = "<|begin_of_image|>";
                self.img_end = "<|end_of_image|>";
                kind = .glm5v;
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

    /// Ports `init_audio` (mtmd.cpp:1093 @cd959bae9).
    fn initAudio(self: *Context) Error!void {
        impl.ggmlAssert(self.ctx_a != null, "mtmd.cpp", 1094, "ctx_a != nullptr");
        if (self.audio_preproc) |*p| p.deinit();
        self.audio_preproc = null;

        const proj = clip.getProjectorType(self.ctx_a.?);

        impl.logWrn("%s: audio input is in experimental stage and may have reduced quality:\n" ++
            "    https://github.com/ggml-org/llama.cpp/discussions/13759\n", .{"init_audio"});

        // set preprocessor
        const kind: ?audio.Kind = switch (proj) {
            .qwen2a, .qwen25o => blk: {
                // <|audio_bos|> ... (embeddings) ... <|audio_eos|>
                self.aud_beg = "<|audio_bos|>";
                self.aud_end = "<|audio_eos|>";
                break :blk .whisper;
            },
            .muse_spark_audio => blk: {
                self.aud_beg = "<|soft_audio_start|>";
                self.aud_end = "<|soft_audio_end|>";
                break :blk .muse_spark_audio;
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
            // audio and text share positions, only usable via mtmd_audio_stream
            .voxtral_rt => null,
            .ultravox, .glma, .meralion => .whisper,
            .lfm2a => .conformer,
            .d1omni_a => .d1omni,
            .lfm3a => blk: {
                // LFM3-Omni wraps the audio in the image span, LFM3-Audio (no vision tower) places it bare
                if (self.ctx_v != null) {
                    self.aud_beg = self.img_beg;
                    self.aud_end = self.img_end;
                }
                break :blk .lfm3a;
            },
            .granite_speech => .granite_speech,
            .gemma4a => blk: {
                self.aud_beg = "<|audio>";
                self.aud_end = "<audio|>";
                break :blk .gemma4a;
            },
            .parakeet => .parakeet,
            .nemotron_transcribe => .nemotron_transcribe,
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
            .chatterbox_spkenc => .chatterbox_ref,
            else => return clip.throwErr("%s: unexpected audio projector type %d\n", .{ "init_audio", @as(c_int, @backingInt(proj)) }),
        };

        // initialize audio preprocessor
        if (kind) |k| {
            self.audio_preproc = audio.Preprocessor.init(k, clip.getHparams(self.ctx_a.?));
            // `mtmd_audio_preprocessor_chatterbox_ref(ctx_a, cbx_is_mtl(ctx_a))`
            if (k == .chatterbox_ref) self.audio_preproc.?.is_mtl = cbxIsMtl(self.ctx_a.?);
            try self.audio_preproc.?.initialize();
        }
    }

    /// Ports `get_clip_ctx` (mtmd.cpp:1223 @cd959bae9): the clip context
    /// for a chunk's type.
    fn getClipCtx(self: *const Context, chunk: *const InputChunk) ?*clip.Ctx {
        if (chunk.type == .image) return self.ctx_v;
        if (chunk.type == .audio) return self.ctx_a;
        impl.abort("mtmd.cpp", 1229, "unknown chunk type");
    }

    fn projTypeV(self: *const Context) names.ProjectorType {
        return if (self.ctx_v) |v| clip.getProjectorType(v) else .unknown;
    }

    fn projTypeA(self: *const Context) names.ProjectorType {
        return if (self.ctx_a) |a| clip.getProjectorType(a) else .unknown;
    }

    /// Ports `video_has_soundtrack` (mtmd.cpp:1241 @cd959bae9): [LFM3_OMNI] the
    /// audio of a video is its soundtrack, see `MTMD_VIDEO_TEMPLATE_LFM3_OMNI`.
    fn videoHasSoundtrack(self: *const Context) bool {
        return self.projTypeV() == .lfm3vl and self.projTypeA() == .lfm3a;
    }

    /// Ports `n_embd_out` (mtmd.cpp:1245 @cd959bae9).
    fn nEmbdOut(self: *const Context) Error!i64 {
        if (self.ctx_v) |v| return clip.nMmprojEmbd(v);
        if (self.ctx_a) |a| return clip.nMmprojEmbd(a);
        return clip.throwStr("no CLIP model loaded");
    }

    /// Ports `mtmd_context` (mtmd.cpp:1255 @cd959bae9), its destructor.
    fn destroy(self: *Context) void {
        clip.free(self.ctx_a);
        clip.free(self.ctx_v);
        clip.free(self.ctx_gen_a);
        self.deinitMembers();
        gpa.destroy(self);
    }

    fn deinitMembers(self: *Context) void {
        if (self.audio_preproc) |*p| p.deinit();
        self.gen_ref.deinit();
        self.out_embd.deinit(gpa);
        self.media_marker.deinit();
        inline for (.{ "tok_ov_img_start", "tok_ov_img_end", "tok_slices_start", "tok_slices_end", "tok_sli_img_start", "tok_sli_img_end", "tok_sli_img_mid", "tok_row_end" }) |f| {
            @field(self, f).deinit(gpa);
        }
    }

    /// Ports `lookup_token` (mtmd.cpp:1262 @cd959bae9): the first token
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

    /// Ports `token_to_piece` (mtmd.cpp:1277 @cd959bae9). The first try
    /// uses libc++'s short-string capacity, 22 bytes.
    fn tokenToPiece(vocab: *const c.struct_llama_vocab, token: Token, special: bool, piece: *std.ArrayList(u8)) !void {
        try resizeZero(u8, piece, 22); // using string internal cache
        const n_chars = lapi.vocab.llama_token_to_piece(vocab, token, piece.items.ptr, @intCast(piece.items.len), 0, special);
        if (n_chars < 0) {
            try resizeZero(u8, piece, @intCast(-n_chars));
            const check = lapi.vocab.llama_token_to_piece(vocab, token, piece.items.ptr, @intCast(piece.items.len), 0, special);
            impl.ggmlAssert(check == -n_chars, "mtmd.cpp", 1287, "check == -n_chars");
        } else {
            piece.shrinkRetainingCapacity(@intCast(n_chars));
        }
    }
};

/// Ports `mtmd_init_from_file` (mtmd.cpp:1295 @cd959bae9).
pub fn mtmd_init_from_file(mmproj_fname: [*:0]const u8, text_model: ?*const c.struct_llama_model, ctx_params: ContextParams) callconv(.c) ?*Context {
    return Context.create(mmproj_fname, text_model, &ctx_params, false) catch |e| {
        impl.logErr("%s: error: %s\n", .{ "mtmd_init_from_file", whatOf(e) });
        return null;
    };
}

/// Ports `mtmd_free` (mtmd.cpp:1306 @cd959bae9).
pub fn mtmd_free(ctx: ?*Context) callconv(.c) void {
    if (ctx) |x| x.destroy();
}

// -----------------------------------------------------------------------------
// tokenizer

/// Ports `mtmd_internal_part` (mtmd-internal.h:13 @cd959bae9): a bitmap, or
/// text when it is null.
pub const InternalPart = struct {
    text: []const u8,
    bitmap: ?*const Bitmap,
    // only used for text parts
    parse_special: bool = false,
};

/// Ports `mtmd_group_mergeable_bitmaps` (mtmd.cpp:1310 @cd959bae9):
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

/// Ports `soundtrack` (mtmd.cpp:1344 @cd959bae9), a member of `mtmd_tokenizer`:
/// [LFM3_OMNI] the audio pieces of a video are joined back and encoded as one
/// clip, each piece takes its rows.
const Soundtrack = struct {
    pieces: std.ArrayList(*const Bitmap) = .empty,
    track: ?*AudioTrack = null, // built on the first piece
    n_samples: usize = 0, // samples of the pieces added so far
    n_assigned: u32 = 0, // rows of the pieces added so far

    fn deinit(self: *Soundtrack) void {
        self.pieces.deinit(gpa);
        if (self.track) |t| t.unref();
    }
};

/// Ports `mtmd_tokenizer` (mtmd.cpp:1329 @cd959bae9).
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

    soundtracks: std.ArrayList(Soundtrack) = .empty,
    soundtrack_of: std.AutoHashMapUnmanaged(*const Bitmap, usize) = .empty, // piece -> index in soundtracks

    cur: InputChunks = .{},
    n_images_added: u32 = 0, // 0-based index assigned to the next image chunk

    /// The constructor `mtmd_tokenizer` (mtmd.cpp:1363 @cd959bae9) from a
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

    /// The constructor `mtmd_tokenizer` (mtmd.cpp:1400 @cd959bae9) from a
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

    /// The destructor of `mtmd_tokenizer` (mtmd.cpp:1356 @cd959bae9).
    fn deinit(self: *Tokenizer) void {
        for (self.bm_from_lazy.items) |b| b.destroy();
        self.bm_from_lazy.deinit(gpa);
        for (self.text_from_lazy.items) |s| free(s);
        self.text_from_lazy.deinit(gpa);
        for (self.soundtracks.items) |*st| st.deinit();
        self.soundtracks.deinit(gpa);
        self.soundtrack_of.deinit(gpa);
        self.parts.deinit(gpa);
        self.cur.deinit();
    }

    /// Ports `expand_lazy_bitmaps` (mtmd.cpp:1420 @cd959bae9).
    fn expandLazyBitmaps(self: *Tokenizer) Error!void {
        var expanded: std.ArrayList(InternalPart) = .empty;
        errdefer expanded.deinit(gpa);
        try expanded.ensureTotalCapacity(gpa, self.parts.items.len);
        for (self.parts.items) |p| {
            if (p.bitmap != null and p.bitmap.?.lazy_callback != null) {
                const bm = p.bitmap.?;
                impl.logDbg("%s: expanding lazy bitmap\n", .{"expand_lazy_bitmaps"});
                const is_video = bm.is_video;
                if (is_video and self.ctx.vid_beg.len != 0) {
                    try expanded.append(gpa, .{ .text = self.ctx.vid_beg, .bitmap = null, .parse_special = true });
                }
                var st: Soundtrack = .{};
                errdefer st.deinit();
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
                            if (is_video and ob.is_audio and self.ctx.videoHasSoundtrack()) {
                                try st.pieces.append(gpa, ob);
                            }
                            impl.logDbg("%s: lazy callback returned bitmap with dimensions %d x %d\n", .{ "expand_lazy_bitmaps", ob.nx, ob.ny });
                        } else if (out_str) |os| {
                            try self.text_from_lazy.append(gpa, os); // remember to free it later
                            try expanded.append(gpa, .{ .text = std.mem.span(os), .bitmap = null, .parse_special = self.parse_special });
                            impl.logDbg("%s: lazy callback returned text: %s\n", .{ "expand_lazy_bitmaps", os });
                        }
                    } else if (res == -1) {
                        // EOF: lazy part removes itself (not added to expanded)
                        if (is_video and self.ctx.vid_end.len != 0) {
                            try expanded.append(gpa, .{ .text = self.ctx.vid_end, .bitmap = null, .parse_special = true });
                        }
                        break;
                    } else if (res == -2) {
                        // error
                        return clip.throwStr("lazy callback returned error");
                    }
                }
                if (st.pieces.items.len != 0) {
                    for (st.pieces.items) |piece| try self.soundtrack_of.put(gpa, piece, self.soundtracks.items.len);
                    try self.soundtracks.append(gpa, st);
                } else {
                    st.deinit();
                }
            } else {
                try expanded.append(gpa, p);
            }
        }
        self.parts.deinit(gpa);
        self.parts = expanded;
    }

    /// Ports `tokenize` (mtmd.cpp:1479 @cd959bae9).
    fn tokenize(self: *Tokenizer, output: *InputChunks) Error!i32 {
        self.cur.clear();

        // [QWEN_VIDEO] handle frame merging for models that support it (i.e. qwen-vl)
        var n_merge_frames: c_int = 1;
        if (self.ctx.ctx_v) |v| {
            n_merge_frames = clip.modelNTemporalMerge(v);
            impl.ggmlAssert(n_merge_frames <= 2, "mtmd.cpp", 1486, "n_merge_frames <= 2 && \"we only support merging maximum 2 images for now; open an issue if this model supports merging more\"");
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

    /// Ports `add_text` (mtmd.cpp:1545 @cd959bae9), the string overload.
    fn addTextStr(self: *Tokenizer, txt: []const u8, parse_special: bool) Error!void {
        const vocab = self.vocab orelse return clip.throwStr("llama_vocab is not provided");
        impl.logDbg("%s: %.*s\n", .{ "add_text", @as(c_int, @intCast(txt.len)), txt.ptr });
        var tokens = try tokenizeTextInternal(vocab, txt, false, parse_special);
        defer tokens.deinit(gpa);
        try self.addTextTokens(tokens.items);
    }

    /// Ports `add_text` (mtmd.cpp:1554 @cd959bae9), the tokens overload.
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

    /// Ports `add_media` (mtmd.cpp:1575 @cd959bae9).
    fn addMedia(self: *Tokenizer, bitmaps: []const *const Bitmap) Error!i32 {
        impl.ggmlAssert(bitmaps.len != 0, "mtmd.cpp", 1576, "!bitmaps.empty()");
        const ctx = self.ctx;

        // note: only one type of media is supported per call, caller should enforce this
        if (bitmaps[0].is_voice) {
            impl.ggmlAssert(bitmaps.len == 1, "mtmd.cpp", 1580, "bitmaps.size() == 1");
            return self.addVoice(bitmaps[0]);
        }
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
                impl.ggmlAssert(!bmp.is_audio, "mtmd.cpp", 1603, "!bmp->is_audio");
                impl.ggmlAssert(ctx.image_preproc != null, "mtmd.cpp", 1604, "ctx->image_preproc != nullptr");
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
                var tmp_preproc_out = if (bmp.isVideoFrame()) try ctx.image_preproc.?.preprocessVideoFrame(&img_u8) else try ctx.image_preproc.?.preprocess(&img_u8);
                defer tmp_preproc_out.deinit();

                // move entries and grid dimensions to the "global" preproc_out
                try preproc_out.entries.appendSlice(gpa, tmp_preproc_out.entries.items);
                tmp_preproc_out.entries.clearRetainingCapacity();

                // for llava-uhd style, we need to handle grid too
                if ((tmp_preproc_out.grid_x > 0 and tmp_preproc_out.grid_y > 0) or tmp_preproc_out.hasOverview()) {
                    impl.ggmlAssert(bitmaps.len == 1, "mtmd.cpp", 1632, "bitmaps.size() == 1");
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

            if (has_tiling_grid and ctx.projTypeV() == .minicpmv4_7) {
                impl.ggmlAssert(bitmaps.len == 1, "mtmd.cpp", 1649, "bitmaps.size() == 1");
                if (ctx.use_image_id) try self.addImageId();
                try self.addTextTokens(ctx.tok_ov_img_start.items);
                // the separators after <image> are appended by clip, see add_canvas_chunk()
                try self.addCanvasChunk(&preproc_out, &bitmaps[0].id);
            } else if (has_tiling_grid) {
                // [QWEN_VIDEO] we do not support "frame merging" for llama-uhd style, so no batching for now
                impl.ggmlAssert(bitmaps.len == 1, "mtmd.cpp", 1658, "bitmaps.size() == 1");

                const n_col = preproc_out.grid_x;
                const n_row = preproc_out.grid_y;

                // split batch into chunks of single images
                var chunks = try self.splitBatchToChunk(&preproc_out, &bitmaps[0].id);
                defer {
                    for (chunks.items) |*ch| ch.deinit();
                    chunks.deinit(gpa);
                }
                impl.ggmlAssert(chunks.items.len > 0, "mtmd.cpp", 1665, "chunks.size() > 0");

                // split_batch_to_chunk must always put the overview image first
                var ov_chunk = chunks.orderedRemove(0);
                defer ov_chunk.deinit();

                // add overview image (first)
                if (ctx.ov_img_first) {
                    if (ctx.use_image_id) try self.addImageId();
                    try self.addTextTokens(ctx.tok_ov_img_start.items);
                    try self.pushChunk(ov_chunk);
                    ov_chunk = .{};
                    try self.addTextTokens(ctx.tok_ov_img_end.items);
                }

                // add slices (or tiles)
                if (chunks.items.len != 0) {
                    impl.logDbg("%s: adding %d slices (%d rows x %d cols)\n", .{ "add_media", @as(c_int, @intCast(chunks.items.len)), n_row, n_col });
                    impl.ggmlAssert(@as(c_int, @intCast(chunks.items.len)) == n_row * n_col, "mtmd.cpp", 1686, "(int)chunks.size() == n_row * n_col");
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
                // U1 uses M-RoPE, but unlike Qwen its image H/W coordinates are
                // zero-based and the whole image occupies one temporal position.
                if (ctx.projTypeV() == .sensenova_u1) {
                    image_tokens.pos = .sensenova_u1;
                }
                // HunyuanVL wraps the image grid with BOI/EOI and adds one newline per row,
                // and uses XD-RoPE (dim-3 = image index).
                else if (ctx.projTypeV() == .hunyuanvl) {
                    image_tokens.pos = .hunyuanvl;
                    image_tokens.image_idx = self.n_images_added;
                    impl.ggmlAssert(n_tokens == @as(usize, image_tokens.nTokens()), "mtmd.cpp", 1782, "n_tokens == (size_t)image_tokens->n_tokens()");
                }

                image_tokens.batch_f32.is_audio = false;
                image_tokens.batch_f32.is_video = bitmaps[0].isVideoFrame();
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

            impl.ggmlAssert(bitmaps.len == 1, "mtmd.cpp", 1817, "bitmaps.size() == 1"); // no batching support for now
            const bitmap = bitmaps[0];

            const ctx_a = ctx.ctx_a orelse {
                impl.logErr("%s: error: model does not support audio input\n", .{"add_media"});
                return 2;
            };

            if (self.soundtrack_of.get(bitmap)) |idx| {
                return self.addSoundtrackPiece(&self.soundtracks.items[idx], bitmap);
            }

            if (bitmap.nx == 0) {
                impl.logErr("%s: error: empty audio data\n", .{"add_media"});
                return 2;
            }

            if (ctx.aud_beg.len != 0) try self.addTextStr(ctx.aud_beg, true); // add audio begin token

            // sanity check
            impl.ggmlAssert(ctx.audio_preproc != null, "mtmd.cpp", 1839, "ctx->audio_preproc != nullptr");

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
                    impl.ggmlAssert(bitmap.ny == 1, "mtmd.cpp", 1849, "bitmap->ny == 1");
                    try resizeZero(f32, &dummy, bitmap.nx);
                    samples = dummy.items;
                } else {
                    const buf = bitmap.data.items;
                    impl.ggmlAssert(buf.len > @sizeOf(f32), "mtmd.cpp", 1856, "buf.size() > sizeof(float)");
                    impl.ggmlAssert(buf.len % @sizeOf(f32) == 0, "mtmd.cpp", 1857, "buf.size() % sizeof(float) == 0");
                    samples = @as([*]const f32, @ptrCast(@alignCast(buf.ptr)))[0 .. buf.len / @sizeOf(f32)];
                }
                if (ctx.audio_preproc == null) {
                    impl.logErr("%s: this model takes audio input only via mtmd_audio_stream\n", .{"add_media"});
                    return 2;
                }
                const ok = try ctx.audio_preproc.?.preprocess(samples, &mel_spec_chunks);
                if (!ok) {
                    impl.logErr("Unable to preprocess audio\n", .{});
                    return 2;
                }
            }

            // one chunk for the five entries of a chatterbox clip, one token per talker conditioning row
            if (clip.getProjectorType(ctx_a) == .chatterbox_spkenc) {
                impl.ggmlAssert(mel_spec_chunks.items.len == 5, "mtmd.cpp", 1875, "mel_spec_chunks.size() == 5");
                // mtl: spkr, perceiver query rows, emotion; turbo: spkr, t3 speech tokens
                const n_query: usize = clip.cbxReadTensor(ctx_a, "a.cenc.perceiver.pre_attention_query", null, 0) / @as(usize, @intCast(try ctx.nEmbdOut()));
                const n_tokens: usize = if (cbxIsMtl(ctx_a)) 2 + n_query else 1 + @as(usize, @bitCast(mel_spec_chunks.items[2].n_len)) / 4;

                const audio_tokens = try gpa.create(AudioTokens);
                audio_tokens.* = .{};
                errdefer audio_tokens.destroy();
                audio_tokens.batch_f32.is_audio = true;
                for (mel_spec_chunks.items) |*mel_spec| {
                    impl.ggmlAssert(mel_spec.n_len <= std.math.maxInt(i32) and mel_spec.n_len >= 0, "mtmd.cpp", 1883, "mel_spec.n_len <= INT32_MAX && mel_spec.n_len >= 0");
                    impl.ggmlAssert(mel_spec.n_mel <= std.math.maxInt(i32) and mel_spec.n_mel >= 0, "mtmd.cpp", 1884, "mel_spec.n_mel <= INT32_MAX && mel_spec.n_mel >= 0");
                    var mel_f32: impl.ImageF32 = .{};
                    errdefer mel_f32.deinit();
                    try mel_f32.setSize(.{ .width = @intCast(mel_spec.n_len), .height = @intCast(mel_spec.n_mel) }, mel_spec.data.items.len == 0, true);
                    try mel_f32.cpyBuf(mel_spec.data.items);
                    try audio_tokens.batch_f32.entries.append(gpa, mel_f32);
                }
                audio_tokens.n_tokens = @truncate(n_tokens);
                try audio_tokens.id.set(bitmap.id.slice());

                try self.pushChunk(.{ .type = .audio, .tokens_audio = audio_tokens });
            } else
            // consider each mel_spec as a separate audio chunk
            for (mel_spec_chunks.items) |*mel_spec| {
                const is_placeholder = mel_spec.data.items.len == 0;

                // Validate dimensions fit in clip_image_size (int)
                impl.ggmlAssert(mel_spec.n_len <= std.math.maxInt(i32) and mel_spec.n_len >= 0, "mtmd.cpp", 1911, "mel_spec.n_len <= INT32_MAX && mel_spec.n_len >= 0");
                impl.ggmlAssert(mel_spec.n_mel <= std.math.maxInt(i32) and mel_spec.n_mel >= 0, "mtmd.cpp", 1912, "mel_spec.n_mel <= INT32_MAX && mel_spec.n_mel >= 0");
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

    /// Ports `add_voice` (mtmd.cpp:1949 @cd959bae9): a voice is a filled
    /// track, its rows are the embeddings.
    fn addVoice(self: *Tokenizer, bitmap: *const Bitmap) Error!i32 {
        if (bitmap.ny == 0) {
            impl.logErr("%s: error: voice \"%s\" holds no embeddings\n", .{ "add_voice", bitmap.id.cStr() });
            return 2;
        }
        const track = try gpa.create(AudioTrack);
        track.* = .{};
        defer track.unref();
        track.n_tokens = bitmap.ny;
        const buf = bitmap.data.items;
        const rows: [*]const f32 = @ptrCast(@alignCast(buf.ptr));
        try track.embd.appendSlice(gpa, rows[0 .. buf.len / @sizeOf(f32)]);
        if (bitmap.gen_ref) |g| track.gen_ref = g.retain();

        const audio_tokens = try gpa.create(AudioTokens);
        audio_tokens.* = .{};
        errdefer audio_tokens.destroy();
        audio_tokens.n_tokens = bitmap.ny;
        try audio_tokens.id.set(bitmap.id.slice());
        audio_tokens.track = track.ref();

        try self.pushChunk(.{ .type = .audio, .tokens_audio = audio_tokens });
        return 0;
    }

    /// Ports `add_soundtrack_piece` (mtmd.cpp:1975 @cd959bae9): a piece of a
    /// soundtrack goes in without the audio markers, it lies inside the span of
    /// its video.
    fn addSoundtrackPiece(self: *Tokenizer, st: *Soundtrack, bitmap: *const Bitmap) Error!i32 {
        const ctx = self.ctx;
        if (st.track == null) {
            var samples: std.ArrayList(f32) = .empty;
            defer samples.deinit(gpa);
            for (st.pieces.items) |piece| {
                if (piece.isPlaceholder()) {
                    try samples.appendNTimes(gpa, 0.0, piece.nx);
                } else {
                    const data: [*]const f32 = @ptrCast(@alignCast(piece.data.items.ptr));
                    try samples.appendSlice(gpa, data[0..piece.nx]);
                }
            }
            var mels: std.ArrayList(audio.Mel) = .empty;
            defer {
                for (mels.items) |*m| m.deinit();
                mels.deinit(gpa);
            }
            if (ctx.audio_preproc == null or !try ctx.audio_preproc.?.preprocess(samples.items, &mels) or mels.items.len != 1) {
                impl.logErr("%s: unable to preprocess the soundtrack\n", .{"add_soundtrack_piece"});
                return 2;
            }
            var mel_f32: impl.ImageF32 = .{};
            errdefer mel_f32.deinit();
            try mel_f32.setSize(.{ .width = @intCast(mels.items[0].n_len), .height = @intCast(mels.items[0].n_mel) }, mels.items[0].data.items.len == 0, true);
            try mel_f32.cpyBuf(mels.items[0].data.items);
            const t = try gpa.create(AudioTrack);
            t.* = .{};
            st.track = t;
            t.n_tokens = @bitCast(clip.nOutputTokens(ctx.ctx_a.?, &mel_f32));
            t.batch_f32.is_audio = true;
            try t.batch_f32.entries.append(gpa, mel_f32);
        }

        // rows of [begin, end): round(sample / samples per row), the last piece takes the rest of the track
        const samples_per_row: f64 = 8.0 * @as(f64, @floatFromInt(clip.getHparams(ctx.ctx_a.?).audio_hop_len)); // one row per 8 mel frames
        const n_tokens = st.track.?.n_tokens;
        st.n_samples += bitmap.nx;
        const begin = st.n_assigned;
        const end: u32 = if (bitmap == st.pieces.items[st.pieces.items.len - 1])
            n_tokens
        else
            std.math.clamp(@as(u32, @intFromFloat(nearbyint(@as(f64, @floatFromInt(st.n_samples)) / samples_per_row))), begin, n_tokens);
        st.n_assigned = end;
        if (end == begin) return 0;

        const audio_tokens = try gpa.create(AudioTokens);
        audio_tokens.* = .{};
        errdefer audio_tokens.destroy();
        audio_tokens.n_tokens = end - begin;
        audio_tokens.track = st.track.?.ref();
        audio_tokens.row_begin = begin;
        try audio_tokens.id.set(bitmap.id.slice());
        try self.pushChunk(.{ .type = .audio, .tokens_audio = audio_tokens });
        return 0;
    }

    /// `add_text("<image_id>" + std::to_string(n_images_added) + "</image_id>", true)`,
    /// as `add_media` writes it for MiniCPM-V 4.6/4.7.
    fn addImageId(self: *Tokenizer) Error!void {
        var buf: [64]u8 = undefined;
        const n = snprintf(&buf, buf.len, "<image_id>%u</image_id>", self.n_images_added);
        try self.addTextStr(buf[0..@intCast(n)], true);
    }

    /// Ports `add_canvas_chunk` (mtmd.cpp:2028 @cd959bae9): MiniCPM-V 4.7
    /// puts the overview and all slices in one chunk, and clip appends the
    /// separators after each tile:
    ///   [ov] </image><slice> [S00] </slice><slice> [S01] </slice>\n<slice> [S10] </slice><slice> [S11] </slice>
    /// `preproc_out`'s images are moved out.
    fn addCanvasChunk(self: *Tokenizer, preproc_out: *image.PreprocOut, id: *const Str) Error!void {
        const ctx_v = self.ctx.ctx_v.?;
        const n_col = preproc_out.grid_x;
        const n_row = preproc_out.grid_y;
        const slices = preproc_out.entries.items;
        impl.ggmlAssert(preproc_out.hasOverview(), "mtmd.cpp", 2032, "preproc_out.has_overview()");
        impl.ggmlAssert(@as(c_int, @intCast(slices.len)) == n_col * n_row, "mtmd.cpp", 2033, "(int) slices.size() == n_col * n_row");

        const ov = &preproc_out.overview;
        ov.suffix_type = .minicpmv_ov;
        if (slices.len != 0) {
            ov.suffix_type = .minicpmv_ov_slice;
            ov.anyres.grid_x = n_col;
            ov.anyres.grid_y = n_row;
        }
        var y: c_int = 0;
        while (y < n_row) : (y += 1) {
            var x: c_int = 0;
            while (x < n_col) : (x += 1) {
                const suffix = &slices[@intCast(y * n_col + x)].suffix_type;
                if (y == n_row - 1 and x == n_col - 1) {
                    suffix.* = .minicpmv_last;
                } else if (x == n_col - 1) {
                    suffix.* = .minicpmv_row_end;
                } else {
                    suffix.* = .minicpmv_slice;
                }
            }
        }

        const image_tokens = try gpa.create(ImageTokens);
        image_tokens.* = .{};
        errdefer image_tokens.destroy();
        image_tokens.pos = .canvas;
        try image_tokens.id.set(id.slice());
        const entries = &image_tokens.batch_f32.entries;
        try entries.ensureTotalCapacity(gpa, 1 + slices.len);
        entries.appendAssumeCapacity(ov.*);
        ov.* = .{};
        for (slices) |s| entries.appendAssumeCapacity(s);
        preproc_out.entries.clearRetainingCapacity();
        // token grid of the last entry, the grids of the other entries are scaled from it
        const last = &entries.items[entries.items.len - 1];
        image_tokens.nx = @bitCast(clip.nOutputTokensX(ctx_v, last));
        image_tokens.ny = @bitCast(clip.nOutputTokensY(ctx_v, last));

        var n_tokens: usize = 0;
        for (entries.items) |*entry| n_tokens = addInt(n_tokens, clip.nOutputTokens(ctx_v, entry));
        impl.ggmlAssert(n_tokens == image_tokens.nTokens(), "mtmd.cpp", 2071, "n_tokens == image_tokens->n_tokens()");

        try self.pushChunk(.{ .type = .image, .tokens_image = image_tokens });
    }

    /// Ports `split_batch_to_chunk` (mtmd.cpp:2082 @cd959bae9): the
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

    /// `split_batch_to_chunk`'s `process_chunk` lambda (mtmd.cpp:2085
    /// @cd959bae9). `img` is moved out.
    fn processChunk(self: *Tokenizer, chunks: *std.ArrayList(InputChunk), img: *impl.ImageF32, id: *const Str) Error!void {
        const image_tokens = try gpa.create(ImageTokens);
        image_tokens.* = .{};
        errdefer image_tokens.destroy();
        image_tokens.nx = @bitCast(clip.nOutputTokens(self.ctx.ctx_v.?, img));
        image_tokens.ny = 1;
        try image_tokens.batch_f32.entries.append(gpa, img.*);
        img.* = .{};
        try image_tokens.id.set(id.slice());

        impl.ggmlAssert(image_tokens.nx > 0, "mtmd.cpp", 2092, "image_tokens->nx > 0");

        try chunks.append(gpa, .{ .type = .image, .tokens_image = image_tokens });
    }

    /// Ports `mtmd_tokenize_text_internal` (mtmd.cpp:2143 @cd959bae9),
    /// copied upstream from `common_tokenize`. The caller owns the tokens.
    fn tokenizeTextInternal(vocab: *const c.struct_llama_vocab, text: []const u8, add_special: bool, parse_special: bool) Error!std.ArrayList(Token) {
        // upper limit for the number of tokens
        var n_tokens: c_int = @truncate(@as(isize, @intCast(text.len)) + 2 * @as(isize, @intFromBool(add_special)));
        var result: std.ArrayList(Token) = .empty;
        errdefer result.deinit(gpa);
        try resizeZero(Token, &result, @intCast(n_tokens));
        // libllama's throw (a vocabulary with no byte token for a byte) is
        // `error.Thrown` here, rethrown for `mtmd_tokenize`'s catch, as the
        // C++'s exception reaches it.
        var exc: LlamaException = .{};
        n_tokens = lapi.vocab.tokenize(vocab, text, result.items, add_special, parse_special, &exc) catch
            return clip.throwStr(exc.what());
        if (n_tokens == std.math.minInt(i32)) {
            return clip.throwStr("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
        }
        if (n_tokens < 0) {
            try resizeZero(Token, &result, @intCast(-n_tokens));
            const check = lapi.vocab.tokenize(vocab, text, result.items, add_special, parse_special, &exc) catch
                return clip.throwStr(exc.what());
            impl.ggmlAssert(check == -n_tokens, "mtmd.cpp", 2161, "check == -n_tokens");
        } else {
            result.shrinkRetainingCapacity(@intCast(n_tokens));
        }
        return result;
    }
};

/// Ports `mtmd_tokenize` (mtmd.cpp:2169 @cd959bae9).
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

/// Ports `mtmd_tokenize_from_parts` (mtmd.cpp:2183 @cd959bae9): as
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

/// Ports `mtmd_encode_impl` (mtmd.cpp:2207 @cd959bae9).
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

    if (image_tokens.pos == .canvas) {
        // the tiles differ in size, encode them one by one
        var offset: usize = 0;
        for (image_tokens.batch_f32.entries.items) |*entry| {
            var one: impl.ImageF32Batch = .{};
            defer one.deinit();
            try one.entries.append(gpa, try entry.clone());
            var embd: std.ArrayList(f32) = .empty;
            defer embd.deinit(gpa);
            try resizeZero(f32, &embd, @as(usize, @bitCast(@as(isize, n_embd_out))) *% @as(usize, @bitCast(@as(isize, clip.nOutputTokens(ctx_clip, entry)))));
            if (!try clip.imageBatchEncode(ctx_clip, ctx.n_threads, &one, &embd)) {
                return 1;
            }
            impl.ggmlAssert(offset + embd.items.len <= out_embd.items.len, "mtmd.cpp", 2233, "offset + embd.size() <= out_embd.size()");
            @memcpy(out_embd.items[offset..][0..embd.items.len], embd.items);
            offset += embd.items.len;
        }
        impl.ggmlAssert(offset == out_embd.items.len, "mtmd.cpp", 2237, "offset == out_embd.size()");
        return 0;
    }

    const ok = try clip.imageBatchEncode(ctx_clip, ctx.n_threads, &image_tokens.batch_f32, out_embd);
    return if (ok) 0 else 1;
}

/// Ports `mtmd_encode_chunk_impl` (mtmd.cpp:2252 @cd959bae9).
fn encodeChunkImpl(ctx: *Context, chunk: *const InputChunk, out_embd: *std.ArrayList(f32)) Error!i32 {
    ctx.gen_ref.deinit(); // see mtmd_gen_state_set_ref()
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
        const t = chunk.tokens_audio orelse {
            impl.logErr("%s: audio tokens are null\n", .{"mtmd_encode_chunk_impl"});
            return 1;
        };
        if (t.isPlaceholder()) {
            impl.logErr("%s: audio tokens batch is placeholder\n", .{"mtmd_encode_chunk_impl"});
            return 1;
        }
        // a filled track (e.g. a voice) needs no audio encoder
        if (ctx.ctx_a == null and (t.track == null or t.track.?.embd.items.len == 0)) {
            impl.logErr("%s: model does not support audio input\n", .{"mtmd_encode_chunk_impl"});
            return 1;
        }
        if (t.track) |track| {
            if (track.embd.items.len == 0) {
                try resizeZero(f32, &track.embd, @as(usize, track.n_tokens) *% @as(usize, @bitCast(@as(isize, @truncate(try ctx.nEmbdOut())))));
                if (!try clip.imageBatchEncode(ctx.ctx_a.?, ctx.n_threads, &track.batch_f32, &track.embd)) {
                    track.embd.clearRetainingCapacity();
                    return 1;
                }
            }
            const ne = track.embd.items.len / track.n_tokens;
            const off = @as(usize, t.row_begin) * ne;
            out_embd.clearRetainingCapacity();
            try out_embd.appendSlice(gpa, track.embd.items[off .. off + @as(usize, t.n_tokens) * ne]);
            if (track.gen_ref) |g| try ctx.gen_ref.assign(&g.ref);
            return 0;
        }
        if (clip.getProjectorType(ctx.ctx_a.?) == .chatterbox_spkenc) {
            return cbxRefEncode(ctx, t, out_embd);
        }
        const ctx_a = ctx.ctx_a.?;
        try resizeZero(f32, out_embd, @as(usize, t.n_tokens) *% @as(usize, @bitCast(@as(isize, @truncate(try ctx.nEmbdOut())))));
        const ok = try clip.imageBatchEncode(ctx_a, ctx.n_threads, &t.batch_f32, out_embd);
        return if (ok) 0 else 1;
    }

    impl.logErr("%s: unknown chunk type %d\n", .{ "mtmd_encode_chunk_impl", @as(c_int, @backingInt(chunk.type)) });
    return 1;
}

/// Ports `mtmd_encode_chunk` (mtmd.cpp:2319 @cd959bae9): the non-batching
/// version.
pub fn mtmd_encode_chunk(ctx: *Context, chunk: *const InputChunk) callconv(.c) i32 {
    return encodeChunkImpl(ctx, chunk, &ctx.out_embd) catch |e| {
        impl.logErr("%s: error: %s\n", .{ "mtmd_encode_chunk", whatOf(e) });
        return 1;
    };
}

/// Ports `mtmd_encode` (mtmd.cpp:2329 @cd959bae9).
pub fn mtmd_encode(ctx: *Context, image_tokens: *const ImageTokens) callconv(.c) i32 {
    return encodeImpl(ctx, image_tokens, &ctx.out_embd) catch |e| {
        impl.logErr("%s: error: %s\n", .{ "mtmd_encode", whatOf(e) });
        return 1;
    };
}

/// Ports `mtmd_get_output_embd` (mtmd.cpp:2338 @cd959bae9).
pub fn mtmd_get_output_embd(ctx: *Context) callconv(.c) ?[*]f32 {
    return dataPtr(f32, &ctx.out_embd);
}

// -----------------------------------------------------------------------------
// audio generation

/// Ports `mtmd_gen_audio_get_info` (mtmd.cpp:2346 @cd959bae9).
pub fn mtmd_gen_audio_get_info(ctx: *const Context) callconv(.c) GenAudioInfo {
    var info: GenAudioInfo = .{ .type = .none, .sample_rate = 0, .model_variant = "", .n_voices = 0 };
    const g = ctx.ctx_gen_a orelse {
        info.type = .none;
        return info;
    };
    // `Loader.getString` keeps a NUL past the string's end.
    info.model_variant = @ptrCast(clip.getHparams(g).gen_model_variant.ptr);
    info.n_voices = @intCast(clip.genGetVoices(g).count());
    switch (clip.getProjectorType(g)) {
        .qwen3tts_gen => {
            info.type = .qwen3tts;
            info.sample_rate = 24000;
        },
        .pockettts_gen => {
            info.type = .pockettts;
            info.sample_rate = 24000;
        },
        .soprano => {
            info.type = .soprano;
            info.sample_rate = 32000;
        },
        .lfm3a_gen => {
            info.type = .lfm3a;
            info.sample_rate = 24000;
        },
        .chatterbox => {
            info.type = .chatterbox;
            info.sample_rate = 24000;
        },
        else => info.type = .none,
    }
    return info;
}

/// Ports `mtmd_gen_audio_read_tensor` (mtmd.cpp:2383 @cd959bae9): a named
/// tensor of the gen audio model as F32, returns the element count (0 if not
/// found); `out` may be null to query the size.
pub fn mtmd_gen_audio_read_tensor(ctx: *Context, name: [*:0]const u8, out: ?[*]f32, n_max: usize) callconv(.c) usize {
    const g = ctx.ctx_gen_a orelse return 0;
    return clip.cbxReadTensor(g, std.mem.span(name), out, n_max);
}

// -----------------------------------------------------------------------------
// chatterbox speaker-clip cloning: the five entries of a CHATTERBOX_SPKENC
// chunk to talker conditioning rows and the flow decoder's reference

/// A one-entry audio batch holding a copy of `mel`, as `cbx_ref_xvec` and
/// `cbx_ref_tokenize` build it. The caller owns the batch.
fn cbxOneEntryBatch(mel: *const impl.ImageF32) Error!impl.ImageF32Batch {
    var batch: impl.ImageF32Batch = .{};
    errdefer batch.deinit();
    batch.is_audio = true;
    {
        var mel_img: impl.ImageF32 = .{};
        errdefer mel_img.deinit();
        try mel_img.setSize(mel.getSize(), false, true);
        try mel_img.cpyBuf(mel.buf.items);
        try batch.entries.append(gpa, mel_img);
    }
    return batch;
}

/// Ports `cbx_ref_xvec` (mtmd.cpp:2391 @cd959bae9): the raw 192-dim
/// CAMPPlus x-vector of the fbank entry.
fn cbxRefXvec(ctx: *Context, fbank: *const impl.ImageF32, xvec: *std.ArrayList(f32)) Error!bool {
    var batch = try cbxOneEntryBatch(fbank);
    defer batch.deinit();
    try resizeZero(f32, xvec, 192);
    if (!try clip.imageBatchEncode(ctx.ctx_a.?, ctx.n_threads, &batch, xvec)) {
        impl.logErr("%s: speaker encoder failed\n", .{"cbx_ref_xvec"});
        return false;
    }
    return true;
}

/// Ports `cbx_ref_tokenize` (mtmd.cpp:2407 @cd959bae9): the s3 speech
/// tokens of a log-mel entry; `rows` optionally receives their speech
/// embedding rows.
fn cbxRefTokenize(ctx: *Context, mel: *const impl.ImageF32, codes: *std.ArrayList(i32), rows: ?*std.ArrayList(f32)) Error!i32 {
    var batch = try cbxOneEntryBatch(mel);
    defer batch.deinit();

    var out_codes: std.ArrayList(i32) = .empty;
    defer out_codes.deinit(gpa);
    var out_embd: std.ArrayList(f32) = .empty;
    defer out_embd.deinit(gpa);

    var params: clip.EncodeParams = .{
        .imgs = &batch,
        .n_threads = ctx.n_threads,
        .gen_process = .tokenize,
        .out_codes = &out_codes,
        .out_code_embd = if (rows != null) &out_embd else null,
    };

    if (!try clip.encode(ctx.ctx_gen_a.?, &params)) {
        impl.logErr("%s: clip_encode failed (tokenize)\n", .{"cbx_ref_tokenize"});
        return 1;
    }

    codes.deinit(gpa);
    codes.* = out_codes;
    out_codes = .empty;
    if (rows) |r| {
        r.deinit(gpa);
        r.* = out_embd;
        out_embd = .empty;
    }
    return 0;
}

/// A lambda of `cbx_ref_cond`, its `read_t` (mtmd.cpp:2444 @cd959bae9): a
/// chatterbox host tensor as F32, or null if it is missing or not
/// `n_expect` long (0 accepts any non-empty size). The slice is allocated
/// from `a`.
fn cbxReadT(a: std.mem.Allocator, ctx_clip: *clip.Ctx, name: []const u8, n_expect: usize) Error!?[]f32 {
    const n = clip.cbxReadTensor(ctx_clip, name, null, 0);
    if (n == 0 or (n_expect > 0 and n != n_expect)) {
        return null;
    }
    const v = try a.alloc(f32, n);
    @memset(v, 0.0);
    return if (clip.cbxReadTensor(ctx_clip, name, v.ptr, n) == n) v else null;
}

/// `sum += (x[i] - mean) * (x[i] - mean)` over a `float` row into a
/// `double`, as the reference `mtmd.o` computes `cbx_ref_cond`'s
/// `layer_norm` variance: the loop vectorizer's strict in-order reduction,
/// products rounded in whole groups of four and only the `n % 4` past the
/// last group fused (read off its disassembly at `-O2`, baseline CPU; the
/// same shape as `audio.zig`'s `strictSquareSum`).
fn cbxStrictSquareSum(row: []const f32, mean: f64) f64 {
    var sum: f64 = 0.0;
    const split = row.len - row.len % 4;
    for (row[0..split]) |v| {
        const d: f64 = @as(f64, v) - mean;
        sum += d * d;
    }
    for (row[split..]) |v| {
        const d: f64 = @as(f64, v) - mean;
        sum = @mulAdd(f64, d, d, sum);
    }
    return sum;
}

/// The perceiver's weights: the locals of `cbx_ref_cond`'s perceiver arm,
/// which opens on its `query` (mtmd.cpp:2565 @cd959bae9), that its lambdas
/// capture.
const CbxPerceiver = struct {
    a: std.mem.Allocator,
    n_e: usize,
    n_head: usize,
    d_head: usize,
    ln_w: []const f32,
    ln_b: []const f32,
    wq: []const f32,
    bq: []const f32,
    wk: []const f32,
    bk: []const f32,
    wv: []const f32,
    bv: []const f32,
    wo: []const f32,
    bo: []const f32,

    /// The `layer_norm` lambda (mtmd.cpp:2585 @cd959bae9).
    fn layerNorm(self: *const CbxPerceiver, in: []const f32, out: []f32) void {
        const n_e = self.n_e;
        var mean: f64 = 0.0;
        for (in[0..n_e]) |v| mean += v;
        mean /= @as(f64, @floatFromInt(n_e));
        const variance = cbxStrictSquareSum(in[0..n_e], mean);
        const sd = @sqrt(variance / @as(f64, @floatFromInt(n_e)) + 1e-5);
        for (0..n_e) |i| {
            out[i] = @floatCast(@mulAdd(f64, (@as(f64, in[i]) - mean) / sd, self.ln_w[i], self.ln_b[i]));
        }
    }

    /// The `linear` lambda (mtmd.cpp:2599 @cd959bae9). The products of two
    /// `float`s in `double` are exact, so whether clang fuses the sum
    /// cannot change a bit.
    fn linear(self: *const CbxPerceiver, w: []const f32, b: []const f32, in: []const f32, n_rows: usize) Error![]f32 {
        const n_e = self.n_e;
        const out = try self.a.alloc(f32, n_rows * n_e);
        for (0..n_rows) |r| {
            for (0..n_e) |o| {
                var acc: f64 = b[o];
                const wr = w[o * n_e ..][0..n_e];
                const ir = in[r * n_e ..][0..n_e];
                for (wr, ir) |wv, iv| acc += @as(f64, wv) * @as(f64, iv);
                out[r * n_e + o] = @floatCast(acc);
            }
        }
        return out;
    }

    /// The `attn_block` lambda (mtmd.cpp:2614 @cd959bae9): a pre-norm
    /// multi-head attention of `x1` over `x2`, plus the residual.
    fn attnBlock(self: *const CbxPerceiver, x1: []const f32, n1: usize, x2: []const f32, n2: usize) Error![]f32 {
        const n_e = self.n_e;
        const d_head = self.d_head;
        const nx1 = try self.a.alloc(f32, n1 * n_e);
        const nx2 = try self.a.alloc(f32, n2 * n_e);
        for (0..n1) |r| self.layerNorm(x1[r * n_e ..][0..n_e], nx1[r * n_e ..][0..n_e]);
        for (0..n2) |r| self.layerNorm(x2[r * n_e ..][0..n_e], nx2[r * n_e ..][0..n_e]);
        const q = try self.linear(self.wq, self.bq, nx1, n1);
        const k = try self.linear(self.wk, self.bk, nx2, n2);
        const v = try self.linear(self.wv, self.bv, nx2, n2);

        const ctxt = try self.a.alloc(f32, n1 * n_e);
        @memset(ctxt, 0.0);
        const sc = try self.a.alloc(f64, n2);
        @memset(sc, 0.0);
        for (0..self.n_head) |hd| {
            const off = hd * d_head;
            for (0..n1) |t| {
                var mx: f64 = -1e30;
                for (0..n2) |s| {
                    var acc: f64 = 0.0;
                    for (0..d_head) |i| {
                        acc += @as(f64, q[t * n_e + off + i]) * @as(f64, k[s * n_e + off + i]);
                    }
                    sc[s] = acc / @sqrt(@as(f64, @floatFromInt(d_head)));
                    mx = if (mx < sc[s]) sc[s] else mx;
                }
                var sum: f64 = 0.0;
                for (sc) |*e| {
                    e.* = exp(e.* - mx);
                    sum += e.*;
                }
                for (0..d_head) |i| {
                    // strided over v: clang leaves it scalar and fused (it
                    // vectorizes only a unit stride, n_e == 1)
                    var acc: f64 = 0.0;
                    for (0..n2) |s| {
                        acc = @mulAdd(f64, sc[s], @as(f64, v[s * n_e + off + i]), acc);
                    }
                    ctxt[t * n_e + off + i] = @floatCast(acc / sum);
                }
            }
        }
        const out = try self.linear(self.wo, self.bo, ctxt, n1);
        for (out, x1[0..out.len]) |*o, x| o.* += x;
        return out;
    }
};

/// Ports `cbx_ref_cond` (mtmd.cpp:2440 @cd959bae9): the talker
/// conditioning rows -- the speaker row, then on mtl the perceiver block
/// and the emotion row.
fn cbxRefCond(ctx: *Context, ve_mel: *const impl.ImageF32, pse: []const f32, out_rows: *std.ArrayList(f32)) Error!i32 {
    const func = "cbx_ref_cond";
    const ctx_clip = ctx.ctx_a.?;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // voice encoder (embeds_from_wavs): 3-layer lstm per 160-frame partial, the normalized partial embeddings are averaged
    const mel = ve_mel.buf.items;
    const n_mel: usize = 40;
    const n_partial: c_int = 160;
    const n_h: usize = 256;
    const step: c_int = @intCast(lround((@as(f64, 16000.0) / 1.3) / @as(f64, n_partial))); // reference rate 1.3
    if (ve_mel.nx() < n_partial or mel.len < @as(usize, @intCast(ve_mel.nx())) * n_mel) {
        impl.logErr("%s: voice encoder mel too short\n", .{func});
        return 1;
    }
    const n_wins: usize = @intCast(@divTrunc(ve_mel.nx() - n_partial, step) + 1);

    var w_ih: [3][]const f32 = undefined;
    var w_hh: [3][]const f32 = undefined;
    var b_ih: [3][]const f32 = undefined;
    var b_hh: [3][]const f32 = undefined;
    for (0..3) |l| {
        const n_in: usize = if (l == 0) n_mel else n_h;
        var nb: [64]u8 = undefined;
        w_ih[l] = (try cbxReadT(a, ctx_clip, std.fmt.bufPrint(&nb, "a.ve.lstm.weight_ih_l{d}", .{l}) catch unreachable, 4 * n_h * n_in)) orelse {
            impl.logErr("%s: model has no voice encoder\n", .{func});
            return 1;
        };
        w_hh[l] = (try cbxReadT(a, ctx_clip, std.fmt.bufPrint(&nb, "a.ve.lstm.weight_hh_l{d}", .{l}) catch unreachable, 4 * n_h * n_h)) orelse {
            impl.logErr("%s: model has no voice encoder\n", .{func});
            return 1;
        };
        b_ih[l] = (try cbxReadT(a, ctx_clip, std.fmt.bufPrint(&nb, "a.ve.lstm.bias_ih_l{d}", .{l}) catch unreachable, 4 * n_h)) orelse {
            impl.logErr("%s: model has no voice encoder\n", .{func});
            return 1;
        };
        b_hh[l] = (try cbxReadT(a, ctx_clip, std.fmt.bufPrint(&nb, "a.ve.lstm.bias_hh_l{d}", .{l}) catch unreachable, 4 * n_h)) orelse {
            impl.logErr("%s: model has no voice encoder\n", .{func});
            return 1;
        };
    }
    const w_proj = (try cbxReadT(a, ctx_clip, "a.ve.proj.weight", n_h * n_h)) orelse {
        impl.logErr("%s: model has no voice encoder projection\n", .{func});
        return 1;
    };
    const b_proj = (try cbxReadT(a, ctx_clip, "a.ve.proj.bias", n_h)) orelse {
        impl.logErr("%s: model has no voice encoder projection\n", .{func});
        return 1;
    };

    const ve = try a.alloc(f32, n_h);
    @memset(ve, 0.0);
    const h = try a.alloc(f32, 3 * n_h);
    const cs = try a.alloc(f32, 3 * n_h);
    const x = try a.alloc(f32, n_h);
    const g = try a.alloc(f32, 4 * n_h);
    @memset(x, 0.0);
    @memset(g, 0.0);
    const e = try a.alloc(f32, n_h);
    for (0..n_wins) |p| {
        @memset(h, 0.0);
        @memset(cs, 0.0);
        for (0..@intCast(n_partial)) |t| {
            var in: []const f32 = mel[(p * @as(usize, @intCast(step)) + t) * n_mel ..][0..n_mel];
            for (0..3) |l| {
                const hl = h[l * n_h ..][0..n_h];
                const cl = cs[l * n_h ..][0..n_h];
                const n_in = in.len;
                for (0..4 * n_h) |j| {
                    // a `float` sum, widened; the products of two `float`s
                    // in `double` are exact, fused or not
                    var acc: f64 = b_ih[l][j] + b_hh[l][j];
                    const wi = w_ih[l][j * n_in ..][0..n_in];
                    for (wi, in) |wv, iv| acc += @as(f64, wv) * @as(f64, iv);
                    const wh = w_hh[l][j * n_h ..][0..n_h];
                    for (wh, hl) |wv, hv| acc += @as(f64, wv) * @as(f64, hv);
                    g[j] = @floatCast(acc);
                }
                // torch gate order: input, forget, cell, output
                for (0..n_h) |i| {
                    const gi = 1.0 / (1.0 + expf(-g[i]));
                    const gf = 1.0 / (1.0 + expf(-g[i + n_h]));
                    const gc = tanhf(g[i + 2 * n_h]);
                    const go = 1.0 / (1.0 + expf(-g[i + 3 * n_h]));
                    cl[i] = @mulAdd(f32, gf, cl[i], gi * gc);
                    x[i] = go * tanhf(cl[i]);
                }
                @memcpy(hl, x);
                in = hl;
            }
        }
        // projected, relu'd, normalized partial embedding
        var norm: f64 = 0.0;
        const hl = h[2 * n_h ..][0..n_h];
        for (0..n_h) |o| {
            var acc: f64 = b_proj[o];
            const w = w_proj[o * n_h ..][0..n_h];
            for (w, hl) |wv, hv| acc += @as(f64, wv) * @as(f64, hv);
            e[o] = @floatCast(if (acc < 0.0) 0.0 else acc);
            norm += @as(f64, e[o]) * @as(f64, e[o]);
        }
        norm = @sqrt(norm);
        for (0..n_h) |o| {
            ve[o] += @as(f32, @floatCast(@as(f64, e[o]) / norm));
        }
    }
    {
        var norm: f64 = 0.0;
        for (ve) |v| norm += @as(f64, v) * @as(f64, v);
        norm = @sqrt(norm);
        for (ve) |*v| v.* = @floatCast(@as(f64, v.*) / norm);
    }

    // speaker projection row
    const n_e: usize = @intCast(try ctx.nEmbdOut());
    const w_spkr = try cbxReadT(a, ctx_clip, "a.cenc.spkr_enc.weight", n_e * n_h);
    const b_spkr = if (w_spkr != null) try cbxReadT(a, ctx_clip, "a.cenc.spkr_enc.bias", n_e) else null;
    if (w_spkr == null or b_spkr == null) {
        impl.logErr("%s: model has no speaker conditioning projection\n", .{func});
        return 1;
    }
    var rows: std.ArrayList(f32) = .empty;
    defer rows.deinit(gpa);
    try rows.resize(gpa, n_e);
    for (0..n_e) |o| {
        var acc: f64 = b_spkr.?[o];
        const w = w_spkr.?[o * n_h ..][0..n_h];
        for (w, ve) |wv, vv| acc += @as(f64, wv) * @as(f64, vv);
        rows.items[o] = @floatCast(acc);
    }

    // the perceiver runs its shared attention block as cross then self attention
    if (try cbxReadT(a, ctx_clip, "a.cenc.perceiver.pre_attention_query", 0)) |query| {
        const n_head: usize = 4;
        const n_ee = n_e * n_e;
        if (pse.len == 0 or pse.len % n_e != 0 or query.len % n_e != 0 or n_e % n_head != 0) {
            impl.logErr("%s: reference speech embeddings required for the perceiver\n", .{func});
            return 1;
        }
        const names_ = [_]struct { []const u8, usize }{
            .{ "a.cenc.perceiver.attn.norm.weight", n_e },
            .{ "a.cenc.perceiver.attn.norm.bias", n_e },
            .{ "a.cenc.perceiver.attn.to_q.weight", n_ee },
            .{ "a.cenc.perceiver.attn.to_q.bias", n_e },
            .{ "a.cenc.perceiver.attn.to_k.weight", n_ee },
            .{ "a.cenc.perceiver.attn.to_k.bias", n_e },
            .{ "a.cenc.perceiver.attn.to_v.weight", n_ee },
            .{ "a.cenc.perceiver.attn.to_v.bias", n_e },
            .{ "a.cenc.perceiver.attn.proj_out.weight", n_ee },
            .{ "a.cenc.perceiver.attn.proj_out.bias", n_e },
            .{ "a.cenc.emotion_adv_fc.weight", n_e },
        };
        var t: [names_.len][]const f32 = undefined;
        for (names_, 0..) |nm, i| {
            t[i] = (try cbxReadT(a, ctx_clip, nm[0], nm[1])) orelse {
                impl.logErr("%s: model has an incomplete perceiver\n", .{func});
                return 1;
            };
        }
        const pc: CbxPerceiver = .{
            .a = a,
            .n_e = n_e,
            .n_head = n_head,
            .d_head = n_e / n_head,
            .ln_w = t[0],
            .ln_b = t[1],
            .wq = t[2],
            .bq = t[3],
            .wk = t[4],
            .bk = t[5],
            .wv = t[6],
            .bv = t[7],
            .wo = t[8],
            .bo = t[9],
        };
        const emo = t[10];

        const n_q = query.len / n_e;
        const pre = try pc.attnBlock(query, n_q, pse, pse.len / n_e);
        const p32 = try pc.attnBlock(pre, n_q, pre, n_q);

        try rows.appendSlice(gpa, p32);
        const exaggeration: f32 = 0.5; // reference default
        for (0..n_e) |i| {
            try rows.append(gpa, emo[i] * exaggeration);
        }
    }

    out_rows.deinit(gpa);
    out_rows.* = rows;
    rows = .empty;
    return 0;
}

/// Ports `cbx_ref_encode` (mtmd.cpp:2679 @cd959bae9): the talker
/// conditioning rows go to `out_embd`, the flow decoder reference to
/// `ctx.gen_ref`.
fn cbxRefEncode(ctx: *Context, audio_tokens: *const AudioTokens, out_embd: *std.ArrayList(f32)) Error!i32 {
    const func = "cbx_ref_encode";
    if (ctx.ctx_gen_a == null) {
        impl.logErr("%s: reference encoding needs the audio generation context\n", .{func});
        return 1;
    }
    const entries = audio_tokens.batch_f32.entries.items;
    impl.ggmlAssert(entries.len == 5, "mtmd.cpp", 2685, "entries.size() == 5");

    var xvec: std.ArrayList(f32) = .empty;
    defer xvec.deinit(gpa);
    if (!try cbxRefXvec(ctx, &entries[0], &xvec)) {
        return 1;
    }

    var flow_codes: std.ArrayList(i32) = .empty;
    defer flow_codes.deinit(gpa);
    var t3_codes: std.ArrayList(i32) = .empty;
    defer t3_codes.deinit(gpa);
    var t3_rows: std.ArrayList(f32) = .empty;
    defer t3_rows.deinit(gpa);
    if (try cbxRefTokenize(ctx, &entries[1], &flow_codes, null) != 0 or
        try cbxRefTokenize(ctx, &entries[2], &t3_codes, &t3_rows) != 0)
    {
        return 1;
    }

    const nx3: usize = @intCast(entries[3].nx());
    const n_feat = nx3 * 80;
    if (nx3 != 2 * flow_codes.items.len) {
        impl.logErr("%s: reference mel length %d does not match %zu codes\n", .{ func, entries[3].nx(), flow_codes.items.len });
        return 1;
    }

    // [spkr, perceiver block] on mtl, [spkr, reference speech rows] on turbo
    const mtl = cbxIsMtl(ctx.ctx_a.?);
    var pse: std.ArrayList(f32) = .empty;
    defer pse.deinit(gpa);
    if (mtl) {
        // `pse = std::move(t3_rows)`: t3_rows is left empty
        pse = t3_rows;
        t3_rows = .empty;
    }
    var rows: std.ArrayList(f32) = .empty;
    defer rows.deinit(gpa);
    if (try cbxRefCond(ctx, &entries[4], pse.items, &rows) != 0) {
        return 1;
    }
    if (!mtl) {
        try rows.appendSlice(gpa, t3_rows.items);
    }
    if (rows.items.len != @as(usize, audio_tokens.n_tokens) * @as(usize, @intCast(try ctx.nEmbdOut()))) {
        impl.logErr("%s: unexpected number of conditioning rows\n", .{func});
        return 1;
    }

    ctx.gen_ref.tokens.deinit(gpa);
    ctx.gen_ref.tokens = flow_codes;
    flow_codes = .empty;
    ctx.gen_ref.feat.clearRetainingCapacity();
    try ctx.gen_ref.feat.appendSlice(gpa, entries[3].buf.items[0..n_feat]);
    ctx.gen_ref.spk.deinit(gpa);
    ctx.gen_ref.spk = xvec;
    xvec = .empty;

    out_embd.deinit(gpa);
    out_embd.* = rows;
    rows = .empty;
    return 0;
}

/// Ports `mtmd_gen_audio_voice_name` (mtmd.cpp:2732 @cd959bae9): the name of
/// the idx-th predefined voice, null when idx is out of range.
pub fn mtmd_gen_audio_voice_name(ctx: *const Context, idx: i32) callconv(.c) ?[*:0]const u8 {
    const g = ctx.ctx_gen_a orelse return null;
    if (idx < 0) return null;
    const voices = clip.genGetVoices(g);
    return if (idx < @as(i32, @intCast(voices.count()))) voices.entries.items[@intCast(idx)].name.ptr else null;
}

/// Ports `mtmd_gen_audio_get_voice` (mtmd.cpp:2740 @cd959bae9): the
/// predefined voice with this name as a voice bitmap, to pass as speaker
/// reference; null if there is no such voice, the caller must free it.
pub fn mtmd_gen_audio_get_voice(ctx: *const Context, name: ?[*:0]const u8) callconv(.c) ?*Bitmap {
    const g = ctx.ctx_gen_a orelse return null;
    const n = std.mem.span(name orelse return null);
    // the name is matched case-insensitively, as the lfm3a "tts" chat template lowers it
    for (clip.genGetVoices(g).entries.items) |*e| {
        if (e.name.len != n.len) continue;
        var eq = true;
        for (e.name, n) |a, b| {
            if (std.ascii.toLower(a) != std.ascii.toLower(b)) {
                eq = false;
                break;
            }
        }
        if (!eq) continue;
        return Bitmap.initVoice(e.name, &e.voice, @intCast(clip.nMmprojEmbd(g))) catch oom();
    }
    return null;
}

/// Ports `mtmd_gen_inp_default` (mtmd.cpp:2756 @cd959bae9).
pub fn mtmd_gen_inp_default(ctx: *const Context) callconv(.c) GenInp {
    var inp = std.mem.zeroes(GenInp);
    inp.type = .gen_code;
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
        .soprano => inp.type = .gen_wav,
        .lfm3a_gen => {
            // MaskGIT sampling runs in the helper, with the settings of the checkpoint
            const hparams = clip.getHparams(g);
            inp.top_k = hparams.gen_top_k;
            inp.top_p = 1.0;
            inp.temp = hparams.gen_temp;
            inp.penalty_repeat = hparams.gen_penalty_repeat;
            inp.penalty_last_n = hparams.gen_penalty_last_n;
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

/// Ports `mtmd_gen_state` (mtmd.cpp:2795 @cd959bae9).
pub const GenState = struct {
    ctx_clip: ?*clip.Ctx = null,
    clip_state: ?*clip.GenState = null,

    out_codes: std.ArrayList(i32) = .empty, // this frame's 16 sampled codes (GEN_CODE)
    out_feats: std.ArrayList(f32) = .empty, // this frame's continuous features, if any (GEN_CODE)
    out_embd: std.ArrayList(f32) = .empty, // next-step hidden state fed back to backbone (GEN_CODE)
    out_audio: std.ArrayList(f32) = .empty, // decoded PCM samples for the current frame (GEN_WAV)

    // voice reference for GEN_WAV, from a speaker clip or a predefined voice (chatterbox)
    ref: clip.GenRef = .{},

    // lfm3a: the codec graph yields STFT frames, the ISTFT runs on the host
    istft: ?audio.StreamingIstft = null,

    fn destroy(self: *GenState) void {
        if (self.istft) |*i| i.deinit();
        self.ref.deinit();
        self.out_audio.deinit(gpa);
        self.out_embd.deinit(gpa);
        self.out_feats.deinit(gpa);
        self.out_codes.deinit(gpa);
        clip.genStateFree(self.clip_state);
        gpa.destroy(self);
    }
};

/// Ports `mtmd_gen_state_params_default` (mtmd.cpp:2815 @cd959bae9).
pub fn mtmd_gen_state_params_default() callconv(.c) GenStateParams {
    return .{ .seed = std.math.maxInt(u32) };
}

/// `mtmd_gen_state_init`'s body, with the C++'s exceptions as errors.
fn genStateInitImpl(ctx: *Context, params: GenStateParams) Error!?*GenState {
    const g = ctx.ctx_gen_a orelse {
        impl.logErr("%s: model does not support audio generation\n", .{"mtmd_gen_state_init"});
        return null;
    };
    const st = try gpa.create(GenState);
    st.* = .{};
    errdefer st.destroy();
    st.ctx_clip = g;
    st.clip_state = try clip.genStateInit(g, params.seed);
    if (st.clip_state == null) {
        st.destroy();
        return null;
    }
    if (clip.getProjectorType(g) == .lfm3a_gen) {
        const hparams = clip.getHparams(g);
        st.istft = try audio.StreamingIstft.init(hparams.audio_n_fft, hparams.audio_hop_len);
    }
    return st;
}

/// Ports `mtmd_gen_state_init` (mtmd.cpp:2821 @cd959bae9).
pub fn mtmd_gen_state_init(ctx: *Context, params: GenStateParams) callconv(.c) ?*GenState {
    return genStateInitImpl(ctx, params) catch impl.abortUncaught("std::bad_alloc", "std::bad_alloc");
}

/// Ports `mtmd_gen_state_free` (mtmd.cpp:2839 @cd959bae9).
pub fn mtmd_gen_state_free(state: ?*GenState) callconv(.c) void {
    if (state) |st| st.destroy();
}

/// Ports `mtmd_gen_state_reset` (mtmd.cpp:2843 @cd959bae9): clears the decoder state
/// (cold start) and keeps the RNG stream.
pub fn mtmd_gen_state_reset(state: *GenState) callconv(.c) void {
    clip.genStateReset(state.clip_state.?);
    if (state.istft) |*i| {
        const hparams = clip.getHparams(state.ctx_clip.?);
        i.deinit();
        i.* = audio.StreamingIstft.init(hparams.audio_n_fft, hparams.audio_hop_len) catch impl.abortUncaught("std::bad_alloc", "std::bad_alloc");
    }
}

/// Ports `mtmd_gen_state_set_ref` (mtmd.cpp:2913 @cd959bae9): copies the
/// gen-side reference data of the last `mtmd_encode_chunk()` call into the
/// state (speaker clip or predefined voice); non-zero if the last encode
/// produced no such data.
pub fn mtmd_gen_state_set_ref(ctx: *Context, state: ?*GenState) callconv(.c) i32 {
    const st = state orelse {
        impl.logErr("%s: invalid gen state\n", .{"mtmd_gen_state_set_ref"});
        return 1;
    };
    if (st.ctx_clip != ctx.ctx_gen_a) {
        impl.logErr("%s: invalid gen state\n", .{"mtmd_gen_state_set_ref"});
        return 1;
    }
    if (ctx.gen_ref.tokens.items.len == 0) {
        impl.logErr("%s: the last encode produced no reference data\n", .{"mtmd_gen_state_set_ref"});
        return 1;
    }
    st.ref.assign(&ctx.gen_ref) catch oom();
    return 0;
}

/// Ports `mtmd_gen_wav_lfm3a` (mtmd.cpp:2852 @cd959bae9): lfm3a GEN_WAV, the ISTFT of the
/// codec output; `n_codes == 0` flushes and resets the state.
fn genWavLfm3a(ctx: *Context, state: *GenState, inp: *const GenInp, out: *GenOut) Error!i32 {
    const ctx_clip = ctx.ctx_gen_a.?;
    const hparams = clip.getHparams(ctx_clip);

    const n_bins: usize = @intCast(@divTrunc(hparams.audio_n_fft, 2) + 1);
    const n_trim: usize = @intCast(@divTrunc(hparams.audio_n_fft - hparams.audio_hop_len, 2));

    const pcm_out = &state.out_audio;
    pcm_out.clearRetainingCapacity();

    if (inp.codes == null or inp.n_codes == 0) {
        // the tail beyond the last frame is padding, like the centered ISTFT of the reference
        var tail = try state.istft.?.flush();
        moveList(f32, pcm_out, &tail);
        pcm_out.shrinkRetainingCapacity(pcm_out.items.len - @min(pcm_out.items.len, n_trim));
        mtmd_gen_state_reset(state);
    } else {
        var in_codes: std.ArrayList(i32) = .empty;
        defer in_codes.deinit(gpa);
        try in_codes.appendSlice(gpa, inp.codes.?[0..inp.n_codes]);
        var feats: std.ArrayList(f32) = .empty;
        defer feats.deinit(gpa);

        // no hidden-state input, the batch entry is an unused placeholder
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
            .codes = &in_codes,
            .out_feats = &feats,
            .gen_state = state.clip_state,
        };
        if (!try clip.encode(ctx_clip, &params)) {
            impl.logErr("%s: clip_encode failed (codec)\n", .{"mtmd_gen_wav_lfm3a"});
            return 1;
        }

        // [2 * n_bins, n_stft_frames]: log magnitude, then phase
        impl.ggmlAssert(feats.items.len % (2 * n_bins) == 0, "mtmd.cpp", 2893, "feats.size() % (2 * (size_t) n_bins) == 0");
        const n_frames = feats.items.len / (2 * n_bins);
        const spec = try gpa.alloc(f32, 2 * n_bins);
        defer gpa.free(spec);
        for (0..n_frames) |f| {
            const fr = feats.items[f * 2 * n_bins ..];
            for (0..n_bins) |j| {
                const mag = expf(fr[j]);
                // clang vectorizes this loop by two, and only there folds the
                // sin/cos pair into __sincosf_stret: the odd last bin is two
                // calls, as `mtmd.o`'s disassembly reads
                const phase = fr[n_bins + j];
                const sc: [2]f32 = if (j < n_bins & ~@as(usize, 1)) audio.sinCos(phase) else .{ sinf(phase), cosf(phase) };
                spec[2 * j + 0] = mag * sc[1];
                spec[2 * j + 1] = mag * sc[0];
            }
            var pcm = try state.istft.?.processFrame(spec.ptr);
            defer pcm.deinit(gpa);
            try pcm_out.appendSlice(gpa, pcm.items);
        }
    }

    out.audio = dataPtr(f32, pcm_out);
    out.n_samples = pcm_out.items.len;
    return 0;
}

/// Ports `mtmd_gen_audio_process_impl` (mtmd.cpp:2926 @cd959bae9).
fn genAudioProcessImpl(ctx: *Context, state: ?*GenState, inp: *const GenInp, out: *GenOut) Error!i32 {
    const ctx_clip = ctx.ctx_gen_a orelse {
        impl.logErr("%s: model does not support audio generation\n", .{"mtmd_gen_audio_process_impl"});
        return 1;
    };
    if (state == null or state.?.ctx_clip != ctx_clip) {
        impl.logErr("%s: invalid gen state\n", .{"mtmd_gen_audio_process_impl"});
        return 1;
    }
    const st = state.?;

    out.* = std.mem.zeroes(GenOut);

    if (clip.getProjectorType(ctx_clip) == .soprano) {
        const n_embd: usize = @bitCast(@as(isize, clip.nMmprojEmbd(ctx_clip)));
        if (inp.type != .gen_wav or inp.feats == null or inp.n_feats < 2 * n_embd or inp.n_feats % n_embd != 0 or inp.codes != null) {
            impl.logErr("%s: soprano requires at least 2 frames of features, without codes\n", .{"mtmd_gen_audio_process_impl"});
            return 1;
        }
    }

    if (inp.type == .gen_wav and clip.getProjectorType(ctx_clip) == .lfm3a_gen) {
        return genWavLfm3a(ctx, st, inp, out);
    }

    if (inp.type == .gen_code) {
        const n_embd: usize = @bitCast(@as(isize, clip.nMmprojEmbd(ctx_clip)));

        // the hidden state may be omitted when the graph does not use it
        var batch: impl.ImageF32Batch = .{ .is_audio = true };
        defer batch.deinit();
        {
            var hidden_state: impl.ImageF32 = .{};
            errdefer hidden_state.deinit();
            try hidden_state.setSize(.{ .width = @intCast(n_embd), .height = 1 }, false, true);
            if (inp.embd) |embd| {
                try hidden_state.cpyBuf(embd[0..n_embd]);
            } else {
                const zeros = try gpa.alloc(f32, n_embd);
                defer gpa.free(zeros);
                @memset(zeros, 0.0);
                try hidden_state.cpyBuf(zeros);
            }
            try batch.entries.append(gpa, hidden_state);
        }

        var in_codes: std.ArrayList(i32) = .empty;
        defer in_codes.deinit(gpa);
        if (inp.codes) |codes| try in_codes.appendSlice(gpa, codes[0..inp.n_codes]);

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
            .temp = inp.temp,
            .codes = if (inp.codes != null) &in_codes else null,
            .out_is_eos = &is_eos,
            .gen_state = st.clip_state,
        };

        if (!try clip.encode(ctx_clip, &params)) {
            impl.logErr("%s: clip_encode failed (gen_code)\n", .{"mtmd_gen_audio_process_impl"});
            return 1;
        }

        moveList(f32, &st.out_embd, &out_embd);
        moveList(i32, &st.out_codes, &out_codes);
        moveList(f32, &st.out_feats, &out_feats);

        out.embd = dataPtr(f32, &st.out_embd);
        out.codes = dataPtr(i32, &st.out_codes);
        out.n_codes = st.out_codes.items.len;
        out.feats = dataPtr(f32, &st.out_feats);
        out.n_feats = st.out_feats.items.len;
        out.is_eos = is_eos;
        return 0;
    }

    // MTMD_GEN_PROCESS_TYPE_GEN_WAV
    if (clip.getProjectorType(ctx_clip) == .chatterbox) {
        if (inp.codes == null or inp.n_codes == 0) {
            impl.logErr("%s: codes required for gen_wav\n", .{"mtmd_gen_audio_process_impl"});
            return 1;
        }
        if (st.ref.tokens.items.len == 0) {
            impl.logErr("%s: gen_wav needs a voice reference, see mtmd_gen_state_set_ref()\n", .{"mtmd_gen_audio_process_impl"});
            return 1;
        }
        var cbx_codes: std.ArrayList(i32) = .empty;
        defer cbx_codes.deinit(gpa);
        try cbx_codes.appendSlice(gpa, inp.codes.?[0..inp.n_codes]);

        // the batch entry is unused, present to satisfy the encode interface
        var cbx_batch: impl.ImageF32Batch = .{ .is_audio = true };
        defer cbx_batch.deinit();
        {
            var dummy: impl.ImageF32 = .{};
            errdefer dummy.deinit();
            try dummy.setSize(.{ .width = 1, .height = 1 }, false, true);
            try dummy.cpyBuf(&.{0.0});
            try cbx_batch.entries.append(gpa, dummy);
        }

        var cbx_params: clip.EncodeParams = .{
            .imgs = &cbx_batch,
            .n_threads = ctx.n_threads,
            .gen_process = .tts,
            .codes = &cbx_codes,
            .out_audio = &st.out_audio,
            .ref = &st.ref,
            .gen_state = st.clip_state,
        };

        if (!try clip.encode(ctx_clip, &cbx_params)) {
            impl.logErr("%s: clip_encode failed (gen_wav)\n", .{"mtmd_gen_audio_process_impl"});
            return 1;
        }

        out.audio = dataPtr(f32, &st.out_audio);
        out.n_samples = st.out_audio.items.len;
        return 0;
    }

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
        .codes = if (has_codes) &in_codes else null,
        .feats = if (has_feats) &in_feats else null,
        .out_audio = &st.out_audio,
        .gen_state = st.clip_state,
    };

    if (!try clip.encode(ctx_clip, &params)) {
        impl.logErr("%s: clip_encode failed (code2wav)\n", .{"mtmd_gen_audio_process_impl"});
        return 1;
    }

    out.audio = dataPtr(f32, &st.out_audio);
    out.n_samples = st.out_audio.items.len;

    return 0;
}

/// Ports `mtmd_gen_audio_process` (mtmd.cpp:3089 @cd959bae9).
pub fn mtmd_gen_audio_process(ctx: *Context, state: ?*GenState, inp: *const GenInp, out: *GenOut) callconv(.c) i32 {
    return genAudioProcessImpl(ctx, state, inp, out) catch |e| {
        impl.logErr("%s: error: %s\n", .{ "mtmd_gen_audio_process", whatOf(e) });
        return 1;
    };
}

// -----------------------------------------------------------------------------
// batches

/// Ports `mtmd_batch_init` (mtmd.cpp:3098 @cd959bae9).
pub fn mtmd_batch_init(ctx: *Context) callconv(.c) ?*Batch {
    const b = gpa.create(Batch) catch impl.abortUncaught("std::bad_alloc", "std::bad_alloc");
    b.* = .{ .ctx = ctx };
    return b;
}

/// Ports `mtmd_batch_free` (mtmd.cpp:3102 @cd959bae9).
pub fn mtmd_batch_free(batch: ?*Batch) callconv(.c) void {
    if (batch) |b| {
        b.entries.deinit(gpa);
        b.output_embd.deinit(gpa);
        gpa.destroy(b);
    }
}

/// Ports `mtmd_batch_add_chunk` (mtmd.cpp:3108 @cd959bae9).
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

/// Ports `mtmd_batch_encode_impl` (mtmd.cpp:3145 @cd959bae9).
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
            impl.ggmlAssert(chunk.tokens_image != null, "mtmd.cpp", 3165, "chunk->tokens_image");
            var b1_f32 = try chunk.tokens_image.?.batch_f32.clone();
            defer b1_f32.entries.deinit(gpa);
            try b0_f32.entries.appendSlice(gpa, b1_f32.entries.items);
        }
    } else if (batch_chunk.tokens_audio) |ta| {
        const b0_f32 = &ta.batch_f32;
        for (batch.entries.items[1..]) |chunk| {
            impl.ggmlAssert(chunk.tokens_audio != null, "mtmd.cpp", 3177, "chunk->tokens_audio");
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

/// Ports `mtmd_batch_encode` (mtmd.cpp:3197 @cd959bae9).
pub fn mtmd_batch_encode(batch: *Batch) callconv(.c) i32 {
    return batchEncodeImpl(batch) catch |e| {
        impl.logErr("%s: error: %s\n", .{ "mtmd_batch_encode", whatOf(e) });
        return 1;
    };
}

/// Ports `mtmd_batch_get_output_embd` (mtmd.cpp:3206 @cd959bae9).
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
        impl.ggmlAssert(offset_prev < batch.output_embd.items.len, "mtmd.cpp", 3217, "offset_prev <  batch->output_embd.size()");
        impl.ggmlAssert(offset <= batch.output_embd.items.len, "mtmd.cpp", 3218, "offset      <= batch->output_embd.size()");
        if (cc == chunk) return batch.output_embd.items.ptr + offset_prev;
    }
    return null; // not found
}

/// Ports `mtmd_decode_use_non_causal` (mtmd.cpp:3226 @cd959bae9).
pub fn mtmd_decode_use_non_causal(ctx: *const Context, chunk: ?*const InputChunk) callconv(.c) bool {
    var proj_type = ctx.projTypeV();
    if (chunk != null and chunk.?.type == .audio) proj_type = ctx.projTypeA();
    return switch (proj_type) {
        .gemma4v => blk: {
            // E2B (n_embd = 1536) and E4B (n_embd = 2560) always use causal
            // note: use mmproj n_embd, because text model may not be provided (e.g. mtmd_get_memory_usage)
            const n_embd = clip.nMmprojEmbd(ctx.ctx_v.?);
            break :blk n_embd != 1536 and n_embd != 2560;
        },
        .gemma4uv, .gemma3, .sensenova_u1, .deepseek4v => true,
        else => false,
    };
}

/// Ports `mtmd_decode_use_mrope` (mtmd.cpp:3249 @cd959bae9).
pub fn mtmd_decode_use_mrope(ctx: *const Context) callconv(.c) bool {
    return ctx.pos_type == .mrope;
}

/// Ports `mtmd_support_vision` (mtmd.cpp:3253 @cd959bae9).
pub fn mtmd_support_vision(ctx: *const Context) callconv(.c) bool {
    return ctx.ctx_v != null;
}

/// Ports `mtmd_support_audio` (mtmd.cpp:3257 @cd959bae9).
pub fn mtmd_support_audio(ctx: *const Context) callconv(.c) bool {
    return ctx.ctx_a != null;
}

/// Ports `mtmd_get_audio_sample_rate` (mtmd.cpp:3261 @cd959bae9).
pub fn mtmd_get_audio_sample_rate(ctx: *const Context) callconv(.c) c_int {
    const a = ctx.ctx_a orelse return -1;
    return clip.getHparams(a).audio_sample_rate;
}

/// Ports `mtmd_video_template` (mtmd.h:62 @cd959bae9): how the frames and timestamps of a
/// video are laid out in the prompt.
pub const VideoTemplate = enum(c_int) {
    default, // "Video:" prefix, then a "[XmY.YYs]" timestamp every N ms
    lfm3, // "<X.X seconds>" before every group of 2 frames
    lfm3_omni, // same as LFM3, then the soundtrack of each group, as an audio bitmap after its frames
    qwen3vl, // "<X.X seconds>" before every group of 2 frames, no markers around the whole video
    _,
};

/// Ports `mtmd_video_config` (mtmd.h:169 @cd959bae9): the recommended video input for
/// the current model, given the source video. Fields set to 0 mean "no
/// preference, use the caller's default".
pub const VideoConfig = extern struct {
    tmpl: VideoTemplate,
    fps: f32, // frame sampling rate
    n_frames: i32, // number of frames to sample
    frame_width: i32, // size to resize the frames to
    frame_height: i32,
};

/// Ports `mtmd_video_config_init` (mtmd.cpp:3268 @cd959bae9), declared in
/// `mtmd-internal.h`: `mtmd_get_video_config()` for a given template, without
/// a context.
pub fn videoConfigInit(tmpl: VideoTemplate, preproc: ?*const image.Preprocessor, src_width: i32, src_height: i32, src_n_frames: i32, src_fps: f32) VideoConfig {
    var cfg: VideoConfig = .{ .tmpl = tmpl, .fps = 0.0, .n_frames = 0, .frame_width = 0, .frame_height = 0 };
    if (tmpl == .default) return cfg;

    // matches Qwen3VLVideoProcessor and Lfm3VlVideoProcessor: 2 fps sampling with a [4, 768] frame clamp
    // LFM3-Omni samples at most 384 frames and hears the soundtrack
    cfg.fps = 2.0;
    if (src_n_frames <= 0 or src_fps <= 0.0) return cfg;

    const n_f: f64 = @as(f64, @floatFromInt(src_n_frames)) / @as(f64, src_fps) * @as(f64, cfg.fps);
    var n: i32 = @intFromFloat(n_f);
    n = std.math.clamp(n, 4, if (tmpl == .lfm3_omni) @as(i32, 384) else 768);
    n = @min(n, src_n_frames);
    cfg.n_frames = n;

    var size: impl.ImageSize = .{ .width = 0, .height = 0 };
    if (preproc != null and preproc.?.getVideoFrameSize(n, src_width, src_height, &size)) {
        cfg.frame_width = size.width;
        cfg.frame_height = size.height;
    }
    return cfg;
}

/// Ports `mtmd_get_video_config` (mtmd.cpp:3296 @cd959bae9): `src_n_frames` and
/// `src_fps` can be <= 0 if unknown.
pub fn mtmd_get_video_config(ctx: *const Context, src_width: i32, src_height: i32, src_n_frames: i32, src_fps: f32) callconv(.c) VideoConfig {
    var tmpl: VideoTemplate = .default;
    switch (if (ctx.ctx_v) |v| clip.getProjectorType(v) else names.ProjectorType.unknown) {
        .lfm3vl => tmpl = if (ctx.videoHasSoundtrack()) .lfm3_omni else .lfm3,
        .qwen3vl => tmpl = .qwen3vl,
        else => {},
    }
    return videoConfigInit(tmpl, if (ctx.image_preproc) |*p| p else null, src_width, src_height, src_n_frames, src_fps);
}

/// Ports `mtmd_get_marker` (mtmd.cpp:3313 @cd959bae9).
pub fn mtmd_get_marker(ctx: *const Context) callconv(.c) [*:0]const u8 {
    return ctx.media_marker.cStr();
}

// -----------------------------------------------------------------------------
// public API functions: mtmd_bitmap

fn oom() noreturn {
    impl.abortUncaught("std::bad_alloc", "std::bad_alloc");
}

/// Ports `mtmd_bitmap_init` (mtmd.cpp:3323 @cd959bae9).
pub fn mtmd_bitmap_init(nx: u32, ny: u32, data: ?[*]const u8) callconv(.c) *Bitmap {
    return Bitmap.createImage(data, nx, ny) catch oom();
}

/// Ports `mtmd_bitmap_init_from_audio` (mtmd.cpp:3330 @cd959bae9). The
/// count is truncated to the bitmap's `uint32_t` as the C++'s constructor
/// truncates it.
pub fn mtmd_bitmap_init_from_audio(n_samples: usize, data: ?[*]const f32) callconv(.c) *Bitmap {
    const bitmap = Bitmap.createAudio(@ptrCast(data), @truncate(n_samples)) catch oom();
    impl.ggmlAssert(bitmap.is_audio, "mtmd.cpp", 3333, "bitmap->is_audio");
    if (!bitmap.isPlaceholder()) {
        impl.ggmlAssert(bitmap.data.items.len == n_samples *% @sizeOf(f32), "mtmd.cpp", 3335, "bitmap->get_ro_buf().size() == n_samples * sizeof(float)");
    }
    return bitmap;
}

/// Ports `mtmd_bitmap_get_nx` (mtmd.cpp:3340 @cd959bae9).
pub fn mtmd_bitmap_get_nx(bitmap: *const Bitmap) callconv(.c) u32 {
    return bitmap.nx;
}

/// Ports `mtmd_bitmap_get_ny` (mtmd.cpp:3344 @cd959bae9).
pub fn mtmd_bitmap_get_ny(bitmap: *const Bitmap) callconv(.c) u32 {
    return bitmap.ny;
}

/// Ports `mtmd_bitmap_get_data` (mtmd.cpp:3348 @cd959bae9).
pub fn mtmd_bitmap_get_data(bitmap: *const Bitmap) callconv(.c) ?[*]const u8 {
    if (bitmap.isPlaceholder()) return null;
    return bitmap.data.items.ptr;
}

/// Ports `mtmd_bitmap_get_n_bytes` (mtmd.cpp:3355 @cd959bae9).
pub fn mtmd_bitmap_get_n_bytes(bitmap: *const Bitmap) callconv(.c) usize {
    if (bitmap.isPlaceholder()) return 0;
    return bitmap.data.items.len;
}

/// Ports `mtmd_bitmap_is_audio` (mtmd.cpp:3362 @cd959bae9).
pub fn mtmd_bitmap_is_audio(bitmap: *const Bitmap) callconv(.c) bool {
    return bitmap.is_audio;
}

/// Ports `mtmd_bitmap_is_voice` (mtmd.cpp:3366 @cd959bae9).
pub fn mtmd_bitmap_is_voice(bitmap: *const Bitmap) callconv(.c) bool {
    return bitmap.is_voice;
}

/// Ports `mtmd_bitmap_get_id` (mtmd.cpp:3370 @cd959bae9).
pub fn mtmd_bitmap_get_id(bitmap: *const Bitmap) callconv(.c) [*:0]const u8 {
    return bitmap.id.cStr();
}

/// Ports `mtmd_bitmap_set_id` (mtmd.cpp:3374 @cd959bae9).
pub fn mtmd_bitmap_set_id(bitmap: *Bitmap, id: ?[*:0]const u8) callconv(.c) void {
    bitmap.id.set(if (id) |s| std.mem.span(s) else "") catch oom();
}

/// Ports `mtmd_bitmap_set_mergeable` (mtmd.cpp:3382 @cd959bae9).
pub fn mtmd_bitmap_set_mergeable(bitmap: *Bitmap, mergeable: bool) callconv(.c) void {
    bitmap.mergeable = mergeable;
}

/// Ports `mtmd_bitmap_set_is_video` (mtmd.cpp:3386 @cd959bae9). On a frame: video frames
/// must be both mergeable and is_video. On a lazy bitmap: the expanded chunks
/// are one video, wrapped by the model's video begin/end tokens.
pub fn mtmd_bitmap_set_is_video(bitmap: *Bitmap, is_video: bool) callconv(.c) void {
    bitmap.is_video = is_video;
}

/// Ports `mtmd_bitmap_init_lazy` (mtmd.cpp:3390 @cd959bae9).
pub fn mtmd_bitmap_init_lazy(ctx: ?*const Context, id: ?[*:0]const u8, user_data: ?*anyopaque, callback: LazyCallback) callconv(.c) *Bitmap {
    _ = ctx; // reserved for future use
    const bitmap = Bitmap.createImage(null, 0, 0) catch oom();
    bitmap.lazy_callback = callback;
    bitmap.lazy_user_data = user_data;
    mtmd_bitmap_set_id(bitmap, id);
    return bitmap;
}

/// Ports `mtmd_bitmap_free` (mtmd.cpp:3402 @cd959bae9).
pub fn mtmd_bitmap_free(bitmap: ?*Bitmap) callconv(.c) void {
    if (bitmap) |b| b.destroy();
}

// -----------------------------------------------------------------------------
// mtmd_input_chunks

/// Ports `mtmd_input_chunks_init` (mtmd.cpp:3410 @cd959bae9).
pub fn mtmd_input_chunks_init() callconv(.c) *InputChunks {
    const chunks = gpa.create(InputChunks) catch oom();
    chunks.* = .{};
    return chunks;
}

/// Ports `mtmd_input_chunks_size` (mtmd.cpp:3414 @cd959bae9).
pub fn mtmd_input_chunks_size(chunks: *const InputChunks) callconv(.c) usize {
    return chunks.entries.items.len;
}

/// Ports `mtmd_input_chunks_get` (mtmd.cpp:3418 @cd959bae9).
pub fn mtmd_input_chunks_get(chunks: *const InputChunks, idx: usize) callconv(.c) ?*const InputChunk {
    if (idx >= chunks.entries.items.len) return null;
    return &chunks.entries.items[idx];
}

/// Ports `mtmd_input_chunks_free` (mtmd.cpp:3425 @cd959bae9).
pub fn mtmd_input_chunks_free(chunks: ?*InputChunks) callconv(.c) void {
    if (chunks) |ch| {
        ch.deinit();
        gpa.destroy(ch);
    }
}

// -----------------------------------------------------------------------------
// mtmd_input_chunk

/// Ports `mtmd_input_chunk_get_type` (mtmd.cpp:3433 @cd959bae9).
pub fn mtmd_input_chunk_get_type(chunk: *const InputChunk) callconv(.c) ChunkType {
    return chunk.type;
}

/// Ports `mtmd_input_chunk_get_tokens_text` (mtmd.cpp:3437 @cd959bae9).
pub fn mtmd_input_chunk_get_tokens_text(chunk: *const InputChunk, n_tokens_output: *usize) callconv(.c) ?[*]const Token {
    if (chunk.type == .text) {
        n_tokens_output.* = chunk.tokens_text.items.len;
        return dataPtr(Token, &chunk.tokens_text);
    }
    n_tokens_output.* = 0;
    return null;
}

/// Ports `mtmd_input_chunk_get_tokens_image` (mtmd.cpp:3446 @cd959bae9).
pub fn mtmd_input_chunk_get_tokens_image(chunk: *const InputChunk) callconv(.c) ?*const ImageTokens {
    if (chunk.type == .image) return chunk.tokens_image;
    return null;
}

/// Ports `mtmd_input_chunk_get_n_tokens` (mtmd.cpp:3453 @cd959bae9).
pub fn mtmd_input_chunk_get_n_tokens(chunk: *const InputChunk) callconv(.c) usize {
    return switch (chunk.type) {
        .text => chunk.tokens_text.items.len,
        .image => mtmd_image_tokens_get_n_tokens(chunk.tokens_image.?),
        .audio => chunk.tokens_audio.?.n_tokens,
        _ => impl.abort("mtmd.cpp", 3461, "invalid chunk type"),
    };
}

/// Ports `mtmd_input_chunk_get_n_pos` (mtmd.cpp:3465 @cd959bae9).
pub fn mtmd_input_chunk_get_n_pos(chunk: *const InputChunk) callconv(.c) c.llama_pos {
    return switch (chunk.type) {
        .text => @truncate(@as(isize, @intCast(chunk.tokens_text.items.len))),
        .image => mtmd_image_tokens_get_n_pos(chunk.tokens_image.?),
        .audio => @bitCast(chunk.tokens_audio.?.n_tokens),
        _ => impl.abort("mtmd.cpp", 3473, "invalid chunk type"),
    };
}

/// Ports `mtmd_input_chunk_get_id` (mtmd.cpp:3477 @cd959bae9).
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

/// Ports `mtmd_input_chunk_copy` (mtmd.cpp:3486 @cd959bae9).
pub fn mtmd_input_chunk_copy(chunk: *const InputChunk) callconv(.c) *InputChunk {
    return copyChunk(chunk) catch oom();
}

/// Ports `mtmd_input_chunk_free` (mtmd.cpp:3506 @cd959bae9).
pub fn mtmd_input_chunk_free(chunk: ?*InputChunk) callconv(.c) void {
    if (chunk) |ch| {
        ch.deinit();
        gpa.destroy(ch);
    }
}

/// Ports `mtmd_input_chunk_save_impl` (mtmd.cpp:3513 @cd959bae9): 0 on
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

/// Ports `mtmd_input_chunk_get_placeholder` (mtmd.cpp:3525 @cd959bae9):
/// the chunk without its image or audio data.
pub fn mtmd_input_chunk_get_placeholder(chunk: *const InputChunk) callconv(.c) ?*InputChunk {
    // this is hacky, but still faster than copy the whole batch data
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    if (chunkSaveImpl(chunk, &buf) != 0) return null;
    return mtmd_input_chunk_load(buf.items.ptr, buf.items.len);
}

/// Ports `mtmd_input_chunk_save` (mtmd.cpp:3534 @cd959bae9).
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

/// Ports `mtmd_input_chunk_load` (mtmd.cpp:3554 @cd959bae9).
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

/// Ports `mtmd_image_tokens_get_n_tokens` (mtmd.cpp:3568 @cd959bae9).
pub fn mtmd_image_tokens_get_n_tokens(image_tokens: *const ImageTokens) callconv(.c) usize {
    return image_tokens.nTokens();
}

/// Ports `mtmd_image_tokens_get_nx` (mtmd.cpp:3572 @cd959bae9).
pub fn mtmd_image_tokens_get_nx(image_tokens: *const ImageTokens) callconv(.c) usize {
    return image_tokens.nx;
}

/// Ports `mtmd_image_tokens_get_ny` (mtmd.cpp:3576 @cd959bae9).
pub fn mtmd_image_tokens_get_ny(image_tokens: *const ImageTokens) callconv(.c) usize {
    return image_tokens.ny;
}

extern "c" fn nearbyint(x: f64) f64;

/// Ports `mtmd_canvas_scale` (mtmd.cpp:3581 @cd959bae9): a tile coordinate
/// mapped onto the canvas as the reference maps it,
/// `round(linspace(0, canvas - 1, grid))`, with `round()` breaking ties to
/// even.
fn canvasScale(coord: u32, grid: u32, canvas: u32) u32 {
    if (grid <= 1 or canvas <= 1) return 0;
    const v = @as(f64, @floatFromInt(coord)) * @as(f64, @floatFromInt(canvas - 1)) / @as(f64, @floatFromInt(grid - 1));
    const r: u32 = @intFromFloat(nearbyint(v));
    return @min(r, canvas - 1);
}

/// Ports `mtmd_canvas_decoder_pos` (mtmd.cpp:3592 @cd959bae9): for
/// `.canvas`, every tile shares the `<image>` token before the chunk as
/// origin; the overview is stretched over the whole canvas, each slice
/// fills its own cell, and the time component is the origin, in slot z. A
/// tile takes one position in slot t (the KV cache position), the
/// separators after it take one position each.
fn canvasDecoderPos(image_tokens: *const ImageTokens, pos_0: c.llama_pos, i_in: usize) DecoderPos {
    const entries = image_tokens.batch_f32.entries.items;
    const grid = entries[0].anyres;
    const nx = image_tokens.nx;
    const ny = image_tokens.ny;
    const gx: u32 = @bitCast(grid.grid_x);
    const gy: u32 = @bitCast(grid.grid_y);
    const canvas_w: u32 = if (grid.isTiled()) gx *% nx else nx;
    const canvas_h: u32 = if (grid.isTiled()) gy *% ny else ny;
    const base: u32 = @bitCast(pos_0 -% 1);

    var i = i_in;
    var pos: DecoderPos = undefined;
    var t: u32 = @bitCast(pos_0);
    for (entries, 0..) |*e, k| {
        const g = image_tokens.canvasTileGrid(k);
        const gw = g[0];
        const gh = g[1];
        if (i < @as(usize, gw *% gh)) {
            const row: u32 = @truncate(i / gw);
            const col: u32 = @truncate(i % gw);
            var h: u32 = undefined;
            var w: u32 = undefined;
            if (k == 0) {
                h = canvasScale(row, gh, canvas_h);
                w = canvasScale(col, gw, canvas_w);
            } else {
                const s: u32 = @truncate(k - 1);
                h = (s / gx) *% ny +% row;
                w = (s % gx) *% nx +% col;
            }
            pos.t = t;
            pos.x = base +% w;
            pos.y = base +% h;
            pos.z = base;
            return pos;
        }
        i -= gw *% gh;

        const n_sep = impl.suffixRows(e.suffix_type).len;
        if (i < n_sep) {
            const p: u32 = @truncate(@as(usize, t) +% 1 +% i);
            pos = .{ .t = p, .x = p, .y = p, .z = p };
            return pos;
        }
        i -= n_sep;
        t = @truncate(@as(usize, t) +% 1 +% n_sep);
    }
    impl.abort("mtmd.cpp", 3638, "token index out of range");
}

/// Ports `mtmd_image_tokens_get_decoder_pos` (mtmd.cpp:3641 @cd959bae9).
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
        .sensenova_u1 => {
            pos.t = p0u;
            pos.x = @truncate(i % image_tokens.nx);
            pos.y = @truncate(i / image_tokens.nx);
            pos.z = 0;
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
        .canvas => pos = canvasDecoderPos(image_tokens, pos_0, i),
        _ => impl.abort("mtmd.cpp", 3702, "invalid position type"),
    }
    return pos;
}

/// Ports `mtmd_image_tokens_get_id` (mtmd.cpp:3707 @cd959bae9).
pub fn mtmd_image_tokens_get_id(image_tokens: *const ImageTokens) callconv(.c) [*:0]const u8 {
    return image_tokens.id.cStr();
}

/// Ports `mtmd_image_tokens_get_n_pos` (mtmd.cpp:3711 @cd959bae9).
pub fn mtmd_image_tokens_get_n_pos(image_tokens: *const ImageTokens) callconv(.c) c.llama_pos {
    return switch (image_tokens.pos) {
        .mrope => @bitCast(@max(image_tokens.nx, image_tokens.ny)),
        .normal => @bitCast(image_tokens.nTokens()),
        .sensenova_u1 => 1,
        // HunyuanVL: the sequential (dim-0) position advances by the full token count
        // (includes BOI/EOI and row newline tokens), not by max(nx, ny)
        .hunyuanvl => @bitCast(image_tokens.nTokens()),
        .canvas => blk: {
            // one position per tile, plus one per separator
            var n_pos: c.llama_pos = 0;
            for (image_tokens.batch_f32.entries.items) |*entry| {
                n_pos +%= 1 +% @as(c.llama_pos, @intCast(impl.suffixRows(entry.suffix_type).len));
            }
            break :blk n_pos;
        },
        _ => impl.abort("mtmd.cpp", 3733, "invalid position type"),
    };
}

// -----------------------------------------------------------------------------
// test function

/// Ports `mtmd_test_create_input_chunks` (mtmd.cpp:3739 @cd959bae9).
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

/// Ports `mtmd_log_set` (mtmd.cpp:3772 @cd959bae9).
pub fn mtmd_log_set(log_callback: c.ggml_log_callback, user_data: ?*anyopaque) callconv(.c) void {
    impl.g_logger_state.log_callback = log_callback orelse impl.logCallbackDefault;
    impl.g_logger_state.log_callback_user_data = user_data;
}

/// Ports `mtmd_get_cap_from_file` (mtmd.cpp:3777 @cd959bae9).
pub fn mtmd_get_cap_from_file(fname: [*:0]const u8) callconv(.c) Caps {
    const tmp = clip.getCap(std.mem.span(fname)) catch |e| {
        impl.logErr("%s: failed to get capabilities from file '%s': %s\n", .{ "mtmd_get_cap_from_file", fname, whatOf(e) });
        return .{ .inp_vision = false, .inp_audio = false, .gen_audio = false };
    };
    return .{ .inp_audio = tmp.has_audio, .inp_vision = tmp.has_vision, .gen_audio = tmp.has_gen_audio };
}

fn stubLogCallback(level: c.enum_ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = level;
    _ = text;
    _ = user_data;
    // do nothing
}

/// Ports `mtmd_memory_usage` (mtmd.h:576 @cd959bae9), new at v0.6.0: what
/// `mtmd_get_memory_usage` measured. The caller owns it and calls `deinit`.
pub const MemoryUsage = struct {
    backend_mem_usage: clip.DevSizeMap = .{},
    /// for models that use non-causal attention, max_tokens must not exceed n_ubatch of llama_context
    image_max_tokens: c_int,
    use_non_causal: bool,

    pub fn deinit(self: *MemoryUsage) void {
        self.backend_mem_usage.deinit();
    }
};

/// Ports `mtmd_get_memory_usage` (mtmd.cpp:3901 @cd959bae9): weights and
/// compute per device, measured without allocating, and since v0.6.0 the
/// image token budget and whether the model decodes images non-causally.
/// The C++ returns a struct holding a `std::map` and has C++ linkage, so
/// this is for Zig callers only; the caller owns the result, `{{}, -1,
/// false}` on failure.
pub fn getMemoryUsage(mmproj_fname: [*:0]const u8, ctx_params_in: ContextParams) MemoryUsage {
    var ctx_params = ctx_params_in;
    const saved_log_callback = impl.g_logger_state.log_callback;
    const saved_log_user_data = impl.g_logger_state.log_callback_user_data;

    ctx_params.progress_callback = null;

    mtmd_log_set(stubLogCallback, null); // suppress logging
    const ctx = Context.create(mmproj_fname, null, &ctx_params, true) catch |e| {
        mtmd_log_set(saved_log_callback, saved_log_user_data); // restore log callback
        impl.logErr("%s: error: %s\n", .{ "mtmd_get_memory_usage", whatOf(e) });
        return .{ .image_max_tokens = -1, .use_non_causal = false };
    };
    defer ctx.destroy();
    mtmd_log_set(saved_log_callback, saved_log_user_data); // restore log callback
    var total_mem: clip.DevSizeMap = .{};
    inline for (.{ ctx.ctx_v, ctx.ctx_a, ctx.ctx_gen_a }) |cc| {
        if (cc) |x| {
            var m = clip.getMemUsage(x) catch oom();
            defer m.deinit();
            for (m.items.items) |e| total_mem.add(e.dev, e.size) catch oom();
        }
    }
    return .{
        .backend_mem_usage = total_mem,
        .image_max_tokens = if (ctx.ctx_v) |v| clip.getImageMaxTokens(v) else -1,
        .use_non_causal = if (ctx.ctx_v != null) mtmd_decode_use_non_causal(ctx, null) else false,
    };
}

// -----------------------------------------------------------------------------
// realtime audio input

/// `std::vector::erase(begin, begin + n)`.
fn eraseFront(list: *std.ArrayList(f32), n: usize) void {
    const rest = list.items.len - n;
    std.mem.copyForwards(f32, list.items[0..rest], list.items[n..]);
    list.shrinkRetainingCapacity(rest);
}

/// Ports `mtmd_audio_stream_params` (mtmd.h:413 @cd959bae9).
pub const AudioStreamParams = extern struct {
    step_ms: i32, // audio encoded per encoder call, rounded up to the model step (default: 320)

    // tokens that open the stream, they go out with the first rows (default: none)
    // with the prompt of its "asr" chat template, LFM3-Audio transcribes live
    prompt: ?[*]const Token,
    n_prompt: usize,
};

/// Ports `mtmd_audio_stream_voxtral_rt` (mtmd.cpp:3990 @cd959bae9): Voxtral
/// Realtime, a row is an audio embedding plus the embedding of the previous
/// token.
const StreamVoxtralRt = struct {
    enc_state: ?*clip.EncState = null,

    n_fft: c_int = 0,
    n_frames_max: c_int = 0, // encoder frames per clip_encode() call
    n_right_pad: c_int = 0, // silence tokens added by flush()
    log_mel_max: f32 = 0.0, // global_log_mel_max in params.json

    prefix: std.ArrayList(Token) = .empty, // text tokens paired with the first rows

    cache: audio.Cache = .{},

    mel_ctx: std.ArrayList(f32) = .empty, // last 2 mel frames, the left context of the conv stem
    tok_embd: std.ArrayList(f32) = .empty,

    fn deinit(self: *StreamVoxtralRt) void {
        clip.encStateFree(self.enc_state);
        self.prefix.deinit(gpa);
        self.cache.deinit();
        self.mel_ctx.deinit(gpa);
        self.tok_embd.deinit(gpa);
    }
};

/// Ports `mtmd_audio_stream_lfm3a` (mtmd.cpp:4017 @cd959bae9): LFM3-Audio, the rows
/// are plain audio embeddings of a chat turn, the model answers after the
/// audio. Live ASR (with a prompt), trained layout:
/// `<prompt> [A x 12] text<|text_end|> ... [A x r] <|mixed_end|> text<|text_end|>`.
/// Training samples are max 30 s: restart the sequence at a silent chunk after
/// 15 s, always after 30 s; the encoder keeps its state.
const StreamLfm3a = struct {
    win: c_int = 0, // samples per mel frame
    enc_state: ?*clip.EncState = null,

    prompt: std.ArrayList(Token) = .empty, // empty: the rows of a chat turn
    tok_text_end: Token = token_null,
    tok_mixed_end: Token = token_null,
    soft_seq_rows: i64 = 0,
    max_seq_rows: i64 = 0,

    started: bool = false,
    done: bool = false, // the text of the final chunk is complete
    in_chunk: bool = false, // the text of the last chunk is being sampled
    final_chunk: bool = false,
    chunk_has_text: bool = false,
    n_chunk_text: i32 = 0,
    n_chunk_rows: i64 = 0,
    n_seq_rows: i64 = 0, // audio rows since the sequence started
    closing: std.ArrayList(Token) = .empty, // the tokens that close the last chunk, they go in before the next one
    tok_embd: std.ArrayList(f32) = .empty,
    // the audio markers of the model (LFM3-Omni), as rows around the audio
    beg_rows: std.ArrayList(f32) = .empty,
    end_rows: std.ArrayList(f32) = .empty,
    beg_done: bool = false,
    end_done: bool = false,

    // live ASR
    const chunk_rows: i64 = 12;
    const max_chunk_tokens: i32 = 48; // text tokens per chunk, then the chunk is closed
    const max_final_tokens: i32 = 96;

    fn deinit(self: *StreamLfm3a) void {
        clip.encStateFree(self.enc_state);
        self.prompt.deinit(gpa);
        self.closing.deinit(gpa);
        self.tok_embd.deinit(gpa);
        self.beg_rows.deinit(gpa);
        self.end_rows.deinit(gpa);
    }
};

/// Ports `mtmd_audio_stream` (mtmd.cpp:3945 @cd959bae9): the model-independent
/// part of a stream; the C++'s two subclasses are the arms of `model`.
pub const AudioStream = struct {
    ctx: *Context,
    ctx_clip: *clip.Ctx,

    // model constants
    n_mel: c_int = 0,
    hop: c_int = 0,
    n_embd: c_int = 0, // decoder embedding size
    frames_per_tok: c_int = 0, // mel frames per decoder row
    samples_per_tok: c_int = 0,
    n_step_mel: c_int = 0, // mel frames per encoder call

    // samples not yet turned into mel frames, buf[0] is at absolute index buf_pos
    buf: std.ArrayList(f32) = .empty,
    buf_pos: i64 = 0,
    n_samples: i64 = 0, // total samples pushed, including padding
    n_frames: i64 = 0, // mel frames computed so far

    mel: std.ArrayList(f32) = .empty, // pending mel frames, frame-major
    embd: std.ArrayList(f32) = .empty, // pending audio embeddings
    embd_off: usize = 0, // rows of embd already consumed
    out: std.ArrayList(f32) = .empty, // rows of the last step

    n_rows: i64 = 0, // rows produced so far
    n_rows_done: i64 = 0, // audio rows whose text has been sampled, see mtmd_audio_stream_get_n_rows()
    n_rows_flushed: i64 = 0, // audio rows up to the end of the last flush
    pending_flush: bool = false,
    ended: bool = false,
    seq_start: bool = false, // the rows of the last step start a new sequence

    model: union(enum) {
        voxtral_rt: StreamVoxtralRt,
        lfm3a: StreamLfm3a,
    },

    fn destroy(self: *AudioStream) void {
        switch (self.model) {
            inline else => |*m| m.deinit(),
        }
        self.buf.deinit(gpa);
        self.mel.deinit(gpa);
        self.embd.deinit(gpa);
        self.out.deinit(gpa);
        gpa.destroy(self);
    }

    /// Ports `add_silence` (mtmd.cpp:3983 @cd959bae9).
    fn addSilence(self: *AudioStream, n: i64) Error!void {
        try self.buf.appendNTimes(gpa, 0.0, @intCast(n));
        self.n_samples += n;
    }

    /// `int(embd.size() / n_embd) - embd_off`: the rows not consumed yet.
    fn nAvail(self: *const AudioStream) i64 {
        return @as(i64, @intCast(self.embd.items.len / @as(usize, @intCast(self.n_embd)))) - @as(i64, @intCast(self.embd_off));
    }

    /// `n_avail` rows, dropping the consumed ones once in a while.
    fn dropConsumed(self: *AudioStream) void {
        if (self.embd_off >= 1024) {
            eraseFront(&self.embd, self.embd_off * @as(usize, @intCast(self.n_embd)));
            self.embd_off = 0;
        }
    }

    /// One encoder call's batch, and its result appended to `embd`.
    fn encodeMel(self: *AudioStream, data: []const f32, nx: c_int, n_out: usize, enc_state: ?*clip.EncState) Error!bool {
        var img: impl.ImageF32 = .{};
        errdefer img.deinit();
        try img.setSize(.{ .width = nx, .height = self.n_mel }, false, true);
        try img.cpyBuf(data);

        var batch: impl.ImageF32Batch = .{ .is_audio = true };
        defer batch.deinit();
        try batch.entries.append(gpa, img);

        var res: std.ArrayList(f32) = .empty;
        defer res.deinit(gpa);
        try resizeZero(f32, &res, @as(usize, @intCast(self.n_embd)) * n_out);
        var params: clip.EncodeParams = .{
            .n_threads = self.ctx.n_threads,
            .imgs = &batch,
            .out_embd = &res,
            .enc_state = enc_state,
        };
        if (!try clip.encode(self.ctx_clip, &params)) {
            impl.logErr("%s: failed to encode audio\n", .{"mtmd_audio_stream_encode"});
            return false;
        }
        try self.embd.appendSlice(gpa, res.items);
        return true;
    }

    // ---- Voxtral Realtime ----

    /// Ports `init` (mtmd.cpp:4141 @cd959bae9), a member of `mtmd_audio_stream_voxtral_rt`.
    fn voxtralInit(self: *AudioStream, params: *const AudioStreamParams) Error!bool {
        const m = &self.model.voxtral_rt;
        const hparams = clip.getHparams(self.ctx_clip);
        const vocab = self.ctx.vocab;
        const func = "init";

        m.n_fft = hparams.audio_n_fft;

        // one decoder row per 80 ms of audio: the conv stem has stride 2, then frames are stacked
        self.frames_per_tok = 2 * hparams.proj_stack_factor;
        self.samples_per_tok = self.frames_per_tok * self.hop;
        m.n_frames_max = @divTrunc(hparams.warmup_audio_size - 2, 2);
        m.log_mel_max = 1.5;

        // the encoder works in whole decoder rows
        var n_step: c_int = @truncate(@divTrunc(@divTrunc(@as(i64, params.step_ms) * hparams.audio_sample_rate, 1000), self.hop));
        n_step = @max(n_step, self.frames_per_tok);
        n_step = @truncate(graph.clipAlign(n_step, self.frames_per_tok));
        self.n_step_mel = @min(n_step, 2 * m.n_frames_max);

        if (params.n_prompt > 0) {
            impl.logErr("%s: Voxtral Realtime takes no prompt\n", .{func});
            return false;
        }

        // prefix: BOS then one pad token per silence token and per delay token
        // the output lags behind the audio by n_delay rows, the text model is conditioned on the same value
        const n_delay: c_int = @bitCast(lapi.model.llama_model_n_tokens_delay(self.ctx.text_model));
        const n_left_pad: c_int = 32;
        m.n_right_pad = n_delay + 1 + 10;
        {
            const pad_str = "[STREAMING_PAD]";
            var pad: [8]Token = undefined;
            const n_pad = lapi.vocab.llama_tokenize(vocab, pad_str, pad_str.len, &pad, pad.len, false, true);
            if (n_pad != 1) {
                impl.logErr("%s: cannot find the %s token\n", .{ func, @as([*:0]const u8, pad_str) });
                return false;
            }
            try m.prefix.append(gpa, lapi.vocab.llama_vocab_bos(vocab));
            try m.prefix.appendNTimes(gpa, pad[0], @intCast(n_left_pad + n_delay));
        }

        try m.cache.fillSinCosTable(@bitCast(m.n_fft));
        try m.cache.fillHannWindow(@bitCast(hparams.audio_window_len), true);
        try m.cache.fillMelFilterbankMatrix(self.n_mel, m.n_fft, hparams.audio_sample_rate, 0.0, -1.0, true, 1.0, false);

        m.enc_state = try clip.encStateInit(self.ctx_clip, m.n_frames_max);
        if (m.enc_state == null) return false;

        // the first mel frame is centered on the first sample, so its window starts before the audio
        const half: usize = @intCast(@divTrunc(m.n_fft, 2));
        try self.buf.appendNTimes(gpa, 0.0, half);
        self.buf_pos = @divTrunc(-m.n_fft, 2);
        try m.mel_ctx.appendNTimes(gpa, 0.0, @intCast(2 * self.n_mel));

        // silence before the audio
        try self.addSilence(@as(i64, n_left_pad) * self.samples_per_tok);

        return true;
    }

    /// Ports `flush` (mtmd.cpp:4201 @cd959bae9), a member of `mtmd_audio_stream_voxtral_rt`.
    fn voxtralFlush(self: *AudioStream, end: bool) Error!i32 {
        const m = &self.model.voxtral_rt;
        // align to a whole row, then add the silence the model expects after the audio
        const spt: i64 = self.samples_per_tok;
        const n_align = @rem(spt - @rem(self.n_samples, spt), spt);
        try self.addSilence(n_align + @as(i64, m.n_right_pad) * spt);
        // one row per samples_per_tok, the left padding included
        self.n_rows_flushed = @divTrunc(self.n_samples, spt);
        self.pending_flush = true;
        self.ended = end;
        return 0;
    }

    /// Ports `mtmd_audio_stream_compute_mel` (mtmd.cpp:4213 @cd959bae9), its
    /// Voxtral overload: mel frame k is centered on sample k * hop, with `force`
    /// the samples past the end are taken as silence.
    fn voxtralComputeMel(self: *AudioStream, force: bool) Error!void {
        const m = &self.model.voxtral_rt;
        const n_fft: usize = @intCast(m.n_fft);
        const nm: usize = @intCast(self.n_mel);
        const half: i64 = @divTrunc(m.n_fft, 2);
        const window = try gpa.alloc(f32, n_fft);
        defer gpa.free(window);
        const frame = try gpa.alloc(f32, nm);
        defer gpa.free(frame);

        while (true) {
            const start = self.n_frames * self.hop - half;
            if (start + m.n_fft > self.n_samples and !(force and start + half < self.n_samples)) break;
            for (0..n_fft) |j| {
                const idx = start + @as(i64, @intCast(j)) - self.buf_pos;
                window[j] = if (idx < @as(i64, @intCast(self.buf.items.len))) self.buf.items[@intCast(idx)] else 0.0;
            }
            try audio.logMelFrame(&m.cache, window.ptr, m.n_fft, self.n_mel, frame.ptr);
            for (0..nm) |i| {
                const v = stdMaxF32(frame[i], m.log_mel_max - 8.0);
                try self.mel.append(gpa, (v + 4.0) / 4.0);
            }
            self.n_frames += 1;
        }

        // drop the samples no frame will need again
        const keep_from = self.n_frames * self.hop - half;
        if (keep_from > self.buf_pos) {
            eraseFront(&self.buf, @intCast(keep_from - self.buf_pos));
            self.buf_pos = keep_from;
        }
    }

    /// Ports `mtmd_audio_stream_encode` (mtmd.cpp:4243 @cd959bae9), its Voxtral
    /// overload.
    fn voxtralEncode(self: *AudioStream, force: bool) Error!bool {
        const m = &self.model.voxtral_rt;
        const nm: usize = @intCast(self.n_mel);
        while (true) {
            const n_avail: c_int = @intCast(self.mel.items.len / nm);
            var n = n_avail - @rem(n_avail, self.frames_per_tok);
            if (!force) {
                if (n_avail < self.n_step_mel) break;
                n = @min(n, self.n_step_mel);
            }
            n = @min(n, 2 * m.n_frames_max);
            if (n <= 0) break;

            // 2 context frames, then the new frames, stored mel-major
            const n_in: usize = @intCast(2 + n);
            const nu: usize = @intCast(n);
            const data = try gpa.alloc(f32, nm * n_in);
            defer gpa.free(data);
            for (0..nm) |mi| {
                data[mi * n_in + 0] = m.mel_ctx.items[mi];
                data[mi * n_in + 1] = m.mel_ctx.items[nm + mi];
                for (0..nu) |t| data[mi * n_in + 2 + t] = self.mel.items[t * nm + mi];
            }

            if (!try self.encodeMel(data, @intCast(n_in), nu / @as(usize, @intCast(self.frames_per_tok)), m.enc_state)) return false;

            m.mel_ctx.clearRetainingCapacity();
            try m.mel_ctx.appendSlice(gpa, self.mel.items[(nu - 2) * nm .. nu * nm]);
            eraseFront(&self.mel, nu * nm);
        }
        return true;
    }

    /// Ports `mtmd_audio_stream_add_rows` (mtmd.cpp:4295 @cd959bae9): out rows =
    /// audio embeddings + token embeddings.
    fn voxtralAddRows(self: *AudioStream, tokens: [*]const Token, n: usize) Error!bool {
        const m = &self.model.voxtral_rt;
        const n_embd: usize = @intCast(self.n_embd);
        try resizeZero(f32, &m.tok_embd, n * n_embd);
        if (lapi.model.llama_model_get_tok_embd_rows(self.ctx.text_model, tokens, @intCast(n), m.tok_embd.items.ptr) == 0) {
            impl.logErr("%s: invalid token\n", .{"mtmd_audio_stream_add_rows"});
            return false;
        }
        const audio_rows = self.embd.items[self.embd_off * n_embd ..];
        for (0..n * n_embd) |i| try self.out.append(gpa, audio_rows[i] + m.tok_embd.items[i]);
        self.embd_off += n;
        self.n_rows += @intCast(n);
        return true;
    }

    /// Ports `step` (mtmd.cpp:4310 @cd959bae9), a member of `mtmd_audio_stream_voxtral_rt`.
    fn voxtralStep(self: *AudioStream, last_token_in: Token) Error!i32 {
        const m = &self.model.voxtral_rt;
        var last_token = last_token_in;
        const force = self.pending_flush;
        self.pending_flush = false;

        // last_token was sampled from the last row, the text of every row so far is out
        self.n_rows_done = self.n_rows;

        try self.voxtralComputeMel(force);
        if (!try self.voxtralEncode(force)) return -2;

        // drop the consumed rows once in a while
        self.dropConsumed();

        const n_avail = self.nAvail();
        self.out.clearRetainingCapacity();

        if (self.n_rows == 0) {
            // the prefix rows go out together, the caller samples from the last one
            const n_prefix: i64 = @intCast(m.prefix.items.len);
            if (n_avail < n_prefix) return if (self.ended) -1 else 0;
            if (!try self.voxtralAddRows(m.prefix.items.ptr, m.prefix.items.len)) return -2;
            return @intCast(n_prefix);
        }

        if (n_avail == 0) return if (self.ended) -1 else 0;
        if (last_token == token_null) {
            impl.logErr("%s: the token sampled from the previous row is required\n", .{"step"});
            return -2;
        }
        if (!try self.voxtralAddRows(@ptrCast(&last_token), 1)) return -2;
        return 1;
    }

    // ---- LFM3-Audio ----

    /// The `find_token` lambda of `mtmd_audio_stream_lfm3a::init`.
    fn lfm3aFindToken(vocab: ?*const c.struct_llama_vocab, piece: [:0]const u8) Token {
        var tok: [2]Token = undefined;
        const n = lapi.vocab.llama_tokenize(vocab, piece.ptr, @intCast(piece.len), &tok, 2, false, true);
        return if (n == 1) tok[0] else token_null;
    }

    /// The `marker_rows` lambda of `mtmd_audio_stream_lfm3a::init`: the markers
    /// the tokenizer puts around an audio clip go around the stream as well.
    fn lfm3aMarkerRows(self: *AudioStream, text: []const u8, rows: *std.ArrayList(f32)) Error!bool {
        if (text.len == 0) return true;
        const vocab = lapi.model.llama_model_get_vocab(self.ctx.text_model);
        const tokens = try gpa.alloc(Token, text.len + 4);
        defer gpa.free(tokens);
        const n = lapi.vocab.llama_tokenize(vocab, text.ptr, @intCast(text.len), tokens.ptr, @intCast(tokens.len), false, true);
        if (n <= 0) return false;
        try resizeZero(f32, rows, @as(usize, @intCast(n)) * @as(usize, @intCast(self.n_embd)));
        return lapi.model.llama_model_get_tok_embd_rows(self.ctx.text_model, tokens.ptr, @bitCast(n), rows.items.ptr) != 0;
    }

    /// Ports `init` (mtmd.cpp:4360 @cd959bae9), a member of `mtmd_audio_stream_lfm3a`.
    fn lfm3aInit(self: *AudioStream, params: *const AudioStreamParams) Error!bool {
        const m = &self.model.lfm3a;
        const hparams = clip.getHparams(self.ctx_clip);

        // one row per 8 mel frames (three stride-2 convs), mel frames are not centered
        m.win = hparams.audio_window_len;
        self.frames_per_tok = 8;
        self.samples_per_tok = self.frames_per_tok * self.hop;

        const n_step: c_int = @truncate(@divTrunc(@divTrunc(@as(i64, params.step_ms) * hparams.audio_sample_rate, 1000), self.hop));
        self.n_step_mel = @truncate(graph.clipAlign(@max(n_step, self.frames_per_tok), self.frames_per_tok));

        if (params.n_prompt > 0) {
            // live ASR, a chunk per encoder call
            self.n_step_mel = @intCast(StreamLfm3a.chunk_rows * self.frames_per_tok);
            try m.prompt.appendSlice(gpa, params.prompt.?[0..params.n_prompt]);

            const vocab = lapi.model.llama_model_get_vocab(self.ctx.text_model);
            m.tok_text_end = lfm3aFindToken(vocab, "<|text_end|>");
            m.tok_mixed_end = lfm3aFindToken(vocab, "<|mixed_end|>");
            if (m.tok_text_end == token_null or m.tok_mixed_end == token_null) {
                impl.logErr("%s: the vocab lacks the tokens of LFM3-Audio live ASR\n", .{"init"});
                return false;
            }
            const sr: f64 = @floatFromInt(hparams.audio_sample_rate);
            const spt: f64 = @floatFromInt(self.samples_per_tok);
            m.soft_seq_rows = lround(15.0 * sr / spt);
            m.max_seq_rows = lround(30.0 * sr / spt);
        }

        m.enc_state = try clip.encStateInit(self.ctx_clip, self.n_step_mel);
        if (m.enc_state == null) return false;

        // the markers the tokenizer puts around an audio clip go around the stream as well
        return (try self.lfm3aMarkerRows(self.ctx.aud_beg, &m.beg_rows)) and (try self.lfm3aMarkerRows(self.ctx.aud_end, &m.end_rows));
    }

    /// Ports `flush` (mtmd.cpp:4417 @cd959bae9), a member of `mtmd_audio_stream_lfm3a`. `end`: pad
    /// the tail like `mtmd_audio_preprocessor_lfm3a`, so the rows match encoding
    /// the clip at once; otherwise only whole rows are encoded, a padded partial
    /// row would change the encoder state. Live ASR: pad the last chunk with
    /// silence, so its text comes out.
    fn lfm3aFlush(self: *AudioStream, end: bool) Error!i32 {
        const m = &self.model.lfm3a;
        const win: i64 = m.win;
        const hop: i64 = self.hop;
        if (end) {
            const n_padded = if (self.n_samples <= win) win else self.n_samples + @rem(hop - @rem(self.n_samples - win, hop), hop);
            try self.addSilence(n_padded - self.n_samples);
        } else if (m.prompt.items.len != 0 and self.n_samples > 0) {
            const frames_per_chunk = StreamLfm3a.chunk_rows * self.frames_per_tok;
            const n_frames_audio = if (self.n_samples <= win) 1 else @divTrunc(self.n_samples - win + hop - 1, hop) + 1;
            const n_frames_padded = @divTrunc(n_frames_audio + frames_per_chunk - 1, frames_per_chunk) * frames_per_chunk;
            try self.addSilence(win + (n_frames_padded - 1) * hop - self.n_samples);
        }
        self.n_rows_flushed = @divTrunc(if (self.n_samples >= win) @divTrunc(self.n_samples - win, hop) + 1 else 0, self.frames_per_tok);
        self.pending_flush = true;
        self.ended = end;
        return 0;
    }

    /// Ports `mtmd_audio_stream_compute_mel` (mtmd.cpp:4434 @cd959bae9), its LFM3
    /// overload: mel frame k covers samples [k * hop, k * hop + win), computed
    /// with the offline preprocessor.
    fn lfm3aComputeMel(self: *AudioStream) Error!bool {
        const m = &self.model.lfm3a;
        const win: i64 = m.win;
        const hop: i64 = self.hop;
        const n_total: i64 = if (self.n_samples >= win) @divTrunc(self.n_samples - win, hop) + 1 else 0;
        const n_new = n_total - self.n_frames;
        if (n_new <= 0) return true;
        const start: usize = @intCast(self.n_frames * hop - self.buf_pos);
        const len: usize = @intCast(win + (n_new - 1) * hop);
        var mel_chunks: std.ArrayList(audio.Mel) = .empty;
        defer {
            for (mel_chunks.items) |*x| x.deinit();
            mel_chunks.deinit(gpa);
        }
        if (!try self.ctx.audio_preproc.?.preprocess(self.buf.items[start..][0..len], &mel_chunks) or mel_chunks.items.len != 1) {
            impl.logErr("%s: unable to preprocess audio\n", .{"mtmd_audio_stream_compute_mel"});
            return false;
        }
        const mel = &mel_chunks.items[0];
        impl.ggmlAssert(mel.n_len == n_new and mel.n_mel == self.n_mel, "mtmd.cpp", 4448, "mel.n_len == n_new && mel.n_mel == s->n_mel");
        const nl: usize = @intCast(mel.n_len);
        for (0..nl) |t| {
            for (0..@as(usize, @intCast(self.n_mel))) |mi| try self.mel.append(gpa, mel.data.items[mi * nl + t]);
        }
        self.n_frames = n_total;

        const keep_from = self.n_frames * hop;
        if (keep_from > self.buf_pos) {
            eraseFront(&self.buf, @intCast(keep_from - self.buf_pos));
            self.buf_pos = keep_from;
        }
        return true;
    }

    /// Ports `mtmd_audio_stream_encode` (mtmd.cpp:4465 @cd959bae9), its LFM3 overload:
    /// whole rows only, except for the last call of an ended stream; the encoder
    /// state carries the rest.
    fn lfm3aEncode(self: *AudioStream, flush: bool) Error!bool {
        const m = &self.model.lfm3a;
        const nm: usize = @intCast(self.n_mel);
        const n_avail: c_int = @intCast(self.mel.items.len / nm);
        var n = n_avail - @rem(n_avail, self.frames_per_tok);
        if (flush and self.ended) {
            n = n_avail;
        } else if (!flush) {
            n -= @rem(n, self.n_step_mel);
        }
        if (n <= 0) return true;

        const nu: usize = @intCast(n);
        const data = try gpa.alloc(f32, nm * nu);
        defer gpa.free(data);
        for (0..nm) |mi| {
            for (0..nu) |t| data[mi * nu + t] = self.mel.items[t * nm + mi];
        }

        var probe: impl.ImageF32 = .{};
        defer probe.deinit();
        try probe.setSize(.{ .width = n, .height = self.n_mel }, false, true);
        const n_rows: usize = @intCast(clip.nOutputTokens(self.ctx_clip, &probe));

        if (!try self.encodeMel(data, n, n_rows, m.enc_state)) return false;
        eraseFront(&self.mel, nu * nm);
        return true;
    }

    /// Ports `step` (mtmd.cpp:4507 @cd959bae9), a member of `mtmd_audio_stream_lfm3a`.
    fn lfm3aStep(self: *AudioStream, last_token: Token) Error!i32 {
        const m = &self.model.lfm3a;
        const flush = self.pending_flush;
        self.pending_flush = false;

        if (!try self.lfm3aComputeMel() or !try self.lfm3aEncode(flush)) return -2;
        if (m.prompt.items.len != 0) return self.lfm3aStepLive(last_token);

        // rows carry no text, all of them go out at once and last_token is not used
        const n_embd: usize = @intCast(self.n_embd);
        const n_avail = self.nAvail();
        self.out.clearRetainingCapacity();
        if (n_avail > 0 or self.ended) {
            if (!m.beg_done) {
                m.beg_done = true;
                try self.out.appendSlice(gpa, m.beg_rows.items);
            }
            try self.out.appendSlice(gpa, self.embd.items[self.embd_off * n_embd ..]);
            self.embd.clearRetainingCapacity();
            self.embd_off = 0;
            if (self.ended and !m.end_done) {
                m.end_done = true;
                try self.out.appendSlice(gpa, m.end_rows.items);
            }
        }
        const n_out: i64 = @intCast(self.out.items.len / n_embd);
        if (n_out == 0) return if (self.ended) -1 else 0;
        self.n_rows += n_out;
        self.n_rows_done += n_avail; // the markers are not audio rows
        return @intCast(n_out);
    }

    /// Ports `add_tokens` (mtmd.cpp:4543 @cd959bae9), a member of `mtmd_audio_stream_lfm3a`.
    fn lfm3aAddTokens(self: *AudioStream, tokens: [*]const Token, n: usize) Error!bool {
        if (n == 0) return true;
        const m = &self.model.lfm3a;
        try resizeZero(f32, &m.tok_embd, n * @as(usize, @intCast(self.n_embd)));
        if (lapi.model.llama_model_get_tok_embd_rows(self.ctx.text_model, tokens, @intCast(n), m.tok_embd.items.ptr) == 0) {
            impl.logErr("%s: invalid token\n", .{"add_tokens"});
            return false;
        }
        try self.out.appendSlice(gpa, m.tok_embd.items);
        return true;
    }

    /// Ports `step_live` (mtmd.cpp:4557 @cd959bae9), a member of `mtmd_audio_stream_lfm3a`:
    /// `last_token` continues the text of the chunk, or <|text_end|> closes it
    /// and the next 12 rows go out (the rest at the end).
    fn lfm3aStepLive(self: *AudioStream, last_token_in: Token) Error!i32 {
        const m = &self.model.lfm3a;
        var last_token = last_token_in;
        self.out.clearRetainingCapacity();
        self.seq_start = false;

        // drop the consumed rows once in a while
        self.dropConsumed();

        if (m.in_chunk) {
            if (last_token == token_null) {
                impl.logErr("%s: the token sampled from the previous row is required\n", .{"step_live"});
                return -2;
            }
            if (last_token != m.tok_text_end) {
                m.chunk_has_text = true;
                if (m.n_chunk_text + 1 < (if (m.final_chunk) StreamLfm3a.max_final_tokens else StreamLfm3a.max_chunk_tokens)) {
                    m.n_chunk_text += 1;
                    return if (try self.lfm3aAddTokens(@ptrCast(&last_token), 1)) 1 else -2;
                }
                // close a runaway chunk like the model should have
                try m.closing.append(gpa, last_token);
            }
            // the text of the chunk is complete, its closing tokens go in before the next chunk
            m.in_chunk = false;
            self.n_rows_done += m.n_chunk_rows;
            if (m.final_chunk) m.done = true;
            try m.closing.append(gpa, m.tok_text_end);
        }
        if (m.done) return -1;

        const n_avail = self.nAvail();
        const drained = self.ended and self.mel.items.len == 0;
        const final = drained and n_avail <= StreamLfm3a.chunk_rows;
        if (!final and n_avail < StreamLfm3a.chunk_rows) return 0;
        if (final and n_avail == 0 and !m.started) {
            m.done = true; // no audio at all
            return -1;
        }

        if (!m.started or m.n_seq_rows >= m.max_seq_rows or (m.n_seq_rows >= m.soft_seq_rows and !m.chunk_has_text)) {
            // the closing tokens go away with the sequence
            m.closing.clearRetainingCapacity();
            try m.closing.appendSlice(gpa, m.prompt.items);
            self.seq_start = true;
            m.n_seq_rows = 0;
        }
        const n_feed: i64 = if (final) n_avail else StreamLfm3a.chunk_rows;
        if (!try self.lfm3aAddTokens(m.closing.items.ptr, m.closing.items.len)) return -2;
        m.closing.clearRetainingCapacity();
        const n_embd: usize = @intCast(self.n_embd);
        const rows = self.embd.items[self.embd_off * n_embd ..];
        try self.out.appendSlice(gpa, rows[0 .. @as(usize, @intCast(n_feed)) * n_embd]);
        self.embd_off += @intCast(n_feed);
        if (final and !try self.lfm3aAddTokens(@ptrCast(&m.tok_mixed_end), 1)) return -2;

        m.started = true;
        m.in_chunk = true;
        m.final_chunk = final;
        m.chunk_has_text = false;
        m.n_chunk_text = 0;
        m.n_chunk_rows = n_feed;
        m.n_seq_rows += n_feed;
        self.n_rows += n_feed;
        return @intCast(self.out.items.len / n_embd);
    }
};

/// `std::max(float, float)`: `(a < b) ? b : a`, which decides where a NaN goes.
fn stdMaxF32(a: f32, b: f32) f32 {
    return if (a < b) b else a;
}

/// Ports `mtmd_support_audio_stream` (mtmd.cpp:4060 @cd959bae9).
pub fn mtmd_support_audio_stream(ctx: *const Context) callconv(.c) bool {
    const a = ctx.ctx_a orelse return false;
    if (ctx.text_model == null) return false;
    const proj = clip.getProjectorType(a);
    return proj == .voxtral_rt or proj == .lfm3a;
}

/// Ports `mtmd_audio_stream_params_default` (mtmd.cpp:4068 @cd959bae9).
pub fn mtmd_audio_stream_params_default() callconv(.c) AudioStreamParams {
    return .{ .step_ms = 320, .prompt = null, .n_prompt = 0 };
}

/// `mtmd_audio_stream_init`'s body, with the C++'s exceptions as errors.
fn audioStreamInitImpl(ctx: *Context, params: *const AudioStreamParams) Error!?*AudioStream {
    if (!mtmd_support_audio_stream(ctx)) {
        impl.logErr("%s: the model does not support streaming audio input\n", .{"mtmd_audio_stream_init"});
        return null;
    }
    const ctx_a = ctx.ctx_a.?;
    const hparams = clip.getHparams(ctx_a);

    const s = try gpa.create(AudioStream);
    s.* = .{
        .ctx = ctx,
        .ctx_clip = ctx_a,
        .model = switch (clip.getProjectorType(ctx_a)) {
            .voxtral_rt => .{ .voxtral_rt = .{} },
            .lfm3a => .{ .lfm3a = .{} },
            else => impl.abort("mtmd.cpp", 4087, "unsupported projector"),
        },
    };
    errdefer s.destroy();
    s.n_mel = hparams.n_mel_bins;
    s.hop = hparams.audio_hop_len;
    s.n_embd = clip.nMmprojEmbd(ctx_a);
    if (s.n_embd != ctx.n_embd_text) {
        impl.logErr("%s: mmproj embedding size %d does not match the text model (%d)\n", .{ "mtmd_audio_stream_init", s.n_embd, ctx.n_embd_text });
        s.destroy();
        return null;
    }
    const ok = switch (s.model) {
        .voxtral_rt => try s.voxtralInit(params),
        .lfm3a => try s.lfm3aInit(params),
    };
    if (!ok) {
        s.destroy();
        return null;
    }
    return s;
}

/// Ports `mtmd_audio_stream_init` (mtmd.cpp:4076 @cd959bae9): null on failure.
pub fn mtmd_audio_stream_init(ctx: *Context, params: AudioStreamParams) callconv(.c) ?*AudioStream {
    return audioStreamInitImpl(ctx, &params) catch |e| impl.abortUncaught("std::runtime_error", whatOf(e));
}

/// Ports `mtmd_audio_stream_free` (mtmd.cpp:4104 @cd959bae9).
pub fn mtmd_audio_stream_free(stream: ?*AudioStream) callconv(.c) void {
    if (stream) |s| s.destroy();
}

/// Ports `mtmd_audio_stream_push` (mtmd.cpp:4108 @cd959bae9): `samples` are PCM F32
/// at `mtmd_get_audio_sample_rate()`. Returns 0 on success, 1 if the stream has
/// ended.
pub fn mtmd_audio_stream_push(s: *AudioStream, samples: ?[*]const f32, n_samples: usize) callconv(.c) i32 {
    if (s.ended) return 1;
    if (n_samples != 0) s.buf.appendSlice(gpa, samples.?[0..n_samples]) catch oom();
    s.n_samples += @intCast(n_samples);
    return 0;
}

/// Ports `mtmd_audio_stream_flush` (mtmd.cpp:4117 @cd959bae9): pads the stream so
/// that all pending rows can be produced. Returns 0 on success, 1 if the stream
/// has ended.
pub fn mtmd_audio_stream_flush(s: *AudioStream, end: bool) callconv(.c) i32 {
    if (s.ended) return 1;
    return (switch (s.model) {
        .voxtral_rt => s.voxtralFlush(end),
        .lfm3a => s.lfm3aFlush(end),
    }) catch |e| impl.abortUncaught("std::runtime_error", whatOf(e));
}

/// Ports `mtmd_audio_stream_step` (mtmd.cpp:4124 @cd959bae9): the next decoder rows.
/// `last_token` is the token sampled from the previous row, `LLAMA_TOKEN_NULL`
/// before the first sample. `> 0` rows ready, 0 more audio is needed, -1 end of
/// stream, -2 error.
pub fn mtmd_audio_stream_step(s: *AudioStream, last_token: Token) callconv(.c) i32 {
    return (switch (s.model) {
        .voxtral_rt => s.voxtralStep(last_token),
        .lfm3a => s.lfm3aStep(last_token),
    }) catch |e| impl.abortUncaught("std::runtime_error", whatOf(e));
}

/// Ports `mtmd_audio_stream_get_output_embd` (mtmd.cpp:4128 @cd959bae9).
pub fn mtmd_audio_stream_get_output_embd(s: *const AudioStream) callconv(.c) ?[*]const f32 {
    return dataPtr(f32, &s.out);
}

/// Ports `mtmd_audio_stream_is_seq_start` (mtmd.cpp:4132 @cd959bae9).
pub fn mtmd_audio_stream_is_seq_start(s: *const AudioStream) callconv(.c) bool {
    return s.seq_start;
}

/// Ports `mtmd_audio_stream_get_n_rows` (mtmd.cpp:4632 @cd959bae9): audio rows whose
/// text has been sampled, prefix rows included.
pub fn mtmd_audio_stream_get_n_rows(s: *const AudioStream) callconv(.c) i64 {
    return s.n_rows_done;
}

/// Ports `mtmd_audio_stream_get_n_rows_flushed` (mtmd.cpp:4636 @cd959bae9).
pub fn mtmd_audio_stream_get_n_rows_flushed(s: *const AudioStream) callconv(.c) i64 {
    return s.n_rows_flushed;
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
