//! libmtmd's helpers: loading a bitmap from a file or buffer, evaluating
//! chunks into a llama context, and the helpers' own logger.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/mtmd-helper.cpp` and
//! `llama.cpp/tools/mtmd/mtmd-helper-common.h`, plus the C types of
//! `llama.cpp/tools/mtmd/mtmd-helper.h`, at v0.4.0 (`5266f24da`). Each
//! declaration names what it ports and the line it began at.
//!
//! # The build ported
//!
//! The reference builds without `MTMD_VIDEO`, as `llama.cpp.zmake` does, so
//! the video helpers are their `#else` arms: they log or assert that video
//! is not compiled in. The vendored decoders upstream compiles into this
//! file -- stb_image, miniaudio, the SHA-256 of `vendor/hash` -- are
//! `vendor/` and `std.crypto`.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("impl.zig");
const mtmd = @import("mtmd.zig");
const vendor = @import("vendor/module.zig");
const lapi = @import("llama").api;
const c = impl.c;
const gpa = impl.gpa;

const Context = mtmd.Context;
const Token = c.llama_token;

extern fn vsnprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, args: *anyopaque) c_int;
extern fn snprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, ...) c_int;
extern fn fputs(s: [*c]const u8, f: ?*anyopaque) c_int;
extern fn fflush(f: ?*anyopaque) c_int;
extern fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern fn fseek(f: ?*anyopaque, off: c_long, whence: c_int) c_int;
extern fn ftell(f: ?*anyopaque) c_long;
extern fn fread(p: ?*anyopaque, size: usize, n: usize, f: ?*anyopaque) usize;
extern fn fclose(f: ?*anyopaque) c_int;
extern fn strerror(e: c_int) [*:0]const u8;

const is_windows = builtin.os.tag == .windows;
const win32 = struct {
    extern "kernel32" fn MultiByteToWideChar(code_page: c_uint, flags: u32, mbs: [*]const u8, mbs_len: c_int, wcs: ?[*]u16, wcs_len: c_int) callconv(.winapi) c_int;
    extern "c" fn _wfopen(path: [*:0]const u16, mode: [*:0]const u16) ?*anyopaque;
};

const seek_set: c_int = 0;
const seek_end: c_int = 2;

// -----------------------------------------------------------------------------
// mtmd-helper.h

/// Ports `mtmd_helper_bitmap_wrapper` (mtmd-helper.h:51 @5266f24da).
pub const BitmapWrapper = extern struct {
    bitmap: ?*mtmd.Bitmap,
    video_ctx: ?*anyopaque,
};

/// Ports `mtmd_helper_post_decode_callback` (mtmd-helper.h:120 @5266f24da).
pub const PostDecodeCallback = ?*const fn (c.llama_batch, ?*anyopaque) callconv(.c) i32;

/// Ports `mtmd_helper_video_info` (mtmd-helper.h:145 @5266f24da).
pub const VideoInfo = extern struct {
    width: u32,
    height: u32,
    fps: f32,
    n_frames: i32,
};

/// Ports `mtmd_helper_video_init_params` (mtmd-helper.h:26 @5266f24da).
pub const VideoInitParams = extern struct {
    fps_target: f32, // desired output fps; <= 0 means use the video's native fps, defaulted to 4.0f
    ffmpeg_bin_dir: ?[*:0]const u8, // directory containing ffmpeg/ffprobe binaries; NULL means search PATH
    timestamp_interval_ms: i64, // interval for adding timestamp as text chunk (example: "[10m50.5s]"); <= 0 means no timestamp, defaulted to 5000ms
};

/// Ports `mtmd_helper_init_opt` (mtmd-helper.h:36 @5266f24da): the options
/// of `mtmd_helper_bitmap_init_from_*()`.
pub const InitOpt = extern struct {
    video_params: VideoInitParams,
};

// -----------------------------------------------------------------------------
// mtmd-helper-common.h: logging

fn defaultCallback(level: c.enum_ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = level;
    _ = user_data;
    const f = impl.cStderr();
    _ = fputs(text, f);
    _ = fflush(f);
}

/// Ports `mtmd_helper_logger` (mtmd-helper-common.h:20 @5266f24da): the
/// helpers' logger, separate from libmtmd's.
pub const Logger = struct {
    default_callback: c.ggml_log_callback = defaultCallback,
    log_callback: c.ggml_log_callback = defaultCallback,
    log_callback_user_data: ?*anyopaque = null,

    /// `log` (mtmd-helper-common.h:51 @5266f24da): format into 128 bytes,
    /// or a heap buffer when that is too short.
    pub fn log(self: *const Logger, level: c.enum_ggml_log_level, comptime fmt: [*:0]const u8, args: anytype) void {
        var buffer: [128]u8 = undefined;
        const len = @call(.auto, snprintf, .{ &buffer, buffer.len, fmt } ++ args);
        const cb = self.log_callback orelse return;
        if (len < 128) {
            cb(level, @ptrCast(&buffer), self.log_callback_user_data);
        } else {
            const n: usize = @intCast(len);
            const buffer2 = std.heap.c_allocator.alloc(u8, n + 1) catch return;
            defer std.heap.c_allocator.free(buffer2);
            @memset(buffer2, 0);
            _ = @call(.auto, snprintf, .{ buffer2.ptr, n + 1, fmt } ++ args);
            buffer2[n] = 0;
            cb(level, @ptrCast(buffer2.ptr), self.log_callback_user_data);
        }
    }
};

/// `g_logger` (mtmd-helper-common.h:60 @5266f24da): one instance for both
/// helper translation units.
pub var g_logger: Logger = .{};

pub fn logDbg(comptime fmt: [*:0]const u8, args: anytype) void {
    g_logger.log(c.GGML_LOG_LEVEL_DEBUG, fmt, args);
}
pub fn logInf(comptime fmt: [*:0]const u8, args: anytype) void {
    g_logger.log(c.GGML_LOG_LEVEL_INFO, fmt, args);
}
pub fn logWrn(comptime fmt: [*:0]const u8, args: anytype) void {
    g_logger.log(c.GGML_LOG_LEVEL_WARN, fmt, args);
}
pub fn logErr(comptime fmt: [*:0]const u8, args: anytype) void {
    g_logger.log(c.GGML_LOG_LEVEL_ERROR, fmt, args);
}

// -----------------------------------------------------------------------------
// mtmd-helper-common.h: the embedding batch

/// Ports `decode_embd_batch` (mtmd-helper-common.h:73 @5266f24da): a
/// `llama_batch` over precomputed embeddings.
pub const EmbdBatch = struct {
    n_pos_per_embd: c_int,
    n_mmproj_embd: c_int,
    pos: std.ArrayList(c.llama_pos) = .empty,
    pos_view: std.ArrayList(c.llama_pos) = .empty, // used by mrope
    n_seq_id: std.ArrayList(i32) = .empty,
    seq_id_0: std.ArrayList(c.llama_seq_id) = .empty,
    seq_ids: std.ArrayList(?[*]c.llama_seq_id) = .empty,
    logits: std.ArrayList(i8) = .empty,
    batch: c.llama_batch = undefined,

    /// The constructor `decode_embd_batch` (mtmd-helper-common.h:83 @5266f24da). Heap-allocated:
    /// `batch` points into the struct's own vectors.
    pub fn create(embd: ?[*]f32, n_tokens: i32, n_pos_per_embd: c_int, n_mmproj_embd: c_int) *EmbdBatch {
        impl.ggmlAssert(n_tokens > 0 and n_pos_per_embd > 0 and n_mmproj_embd > 0, "mtmd-helper-common.h", 84, "n_tokens > 0 && n_pos_per_embd > 0 && n_mmproj_embd > 0");
        const self = gpa.create(EmbdBatch) catch oom();
        self.* = .{ .n_pos_per_embd = n_pos_per_embd, .n_mmproj_embd = n_mmproj_embd };
        const n: usize = @intCast(n_tokens);
        resizeZero(c.llama_pos, &self.pos, n * @as(usize, @intCast(n_pos_per_embd)));
        resizeZero(i32, &self.n_seq_id, n);
        self.seq_ids.appendNTimes(gpa, null, n + 1) catch oom();
        resizeZero(i8, &self.logits, n);
        resizeZero(c.llama_seq_id, &self.seq_id_0, 1);
        self.seq_ids.items[n] = null;
        self.batch = .{
            .n_tokens = n_tokens,
            .token = null,
            .embd = embd,
            .pos = self.pos.items.ptr,
            .n_seq_id = self.n_seq_id.items.ptr,
            .seq_id = @ptrCast(self.seq_ids.items.ptr),
            .logits = self.logits.items.ptr,
        };
        return self;
    }

    pub fn destroy(self: *EmbdBatch) void {
        self.pos.deinit(gpa);
        self.pos_view.deinit(gpa);
        self.n_seq_id.deinit(gpa);
        self.seq_id_0.deinit(gpa);
        self.seq_ids.deinit(gpa);
        self.logits.deinit(gpa);
        gpa.destroy(self);
    }

    /// `set_position_normal` (mtmd-helper-common.h:102 @5266f24da).
    pub fn setPositionNormal(self: *EmbdBatch, pos_0: c.llama_pos, seq_id: c.llama_seq_id) void {
        self.seq_id_0.items[0] = seq_id;
        var i: i32 = 0;
        while (i < self.batch.n_tokens) : (i += 1) {
            const k: usize = @intCast(i);
            self.batch.pos[k] = pos_0 +% i;
            self.batch.n_seq_id[k] = 1;
            self.batch.seq_id[k] = self.seq_id_0.items.ptr;
            self.batch.logits[k] = 0;
        }
    }

    /// `set_position_mrope_2d` (mtmd-helper-common.h:113 @5266f24da): M-RoPE
    /// for image.
    pub fn setPositionMrope2d(self: *EmbdBatch, rel_pos: []const mtmd.DecoderPos, seq_id: c.llama_seq_id) void {
        impl.ggmlAssert(self.n_pos_per_embd == 4, "mtmd-helper-common.h", 114, "n_pos_per_embd == 4");
        impl.ggmlAssert(rel_pos.len != 0 and @as(i32, @truncate(@as(isize, @intCast(rel_pos.len)))) == self.batch.n_tokens, "mtmd-helper-common.h", 115, "!rel_pos.empty() && (int32_t)rel_pos.size() == batch.n_tokens");
        self.seq_id_0.items[0] = seq_id;
        const n_tokens: usize = @intCast(self.batch.n_tokens);
        for (0..n_tokens) |idx| {
            self.pos.items[idx] = @bitCast(rel_pos[idx].t);
            self.pos.items[idx + n_tokens] = @bitCast(rel_pos[idx].y);
            self.pos.items[idx + n_tokens * 2] = @bitCast(rel_pos[idx].x);
            self.pos.items[idx + n_tokens * 3] = @bitCast(rel_pos[idx].z);
        }
        for (0..n_tokens) |i| {
            self.batch.n_seq_id[i] = 1;
            self.batch.seq_id[i] = self.seq_id_0.items.ptr;
            self.batch.logits[i] = 0;
        }
    }

    /// `set_position_mrope_1d` (mtmd-helper-common.h:133 @5266f24da): M-RoPE
    /// for audio.
    pub fn setPositionMrope1d(self: *EmbdBatch, pos_0: c.llama_pos, seq_id: c.llama_seq_id) void {
        impl.ggmlAssert(self.n_pos_per_embd == 4, "mtmd-helper-common.h", 134, "n_pos_per_embd == 4");
        self.seq_id_0.items[0] = seq_id;
        const n_tokens: usize = @intCast(self.batch.n_tokens);
        for (0..n_tokens) |idx| {
            const p = pos_0 +% @as(i32, @intCast(idx));
            self.pos.items[idx] = p;
            self.pos.items[idx + n_tokens] = p;
            self.pos.items[idx + n_tokens * 2] = p;
            self.pos.items[idx + n_tokens * 3] = p;
        }
        for (0..n_tokens) |i| {
            self.batch.n_seq_id[i] = 1;
            self.batch.seq_id[i] = self.seq_id_0.items.ptr;
            self.batch.logits[i] = 0;
        }
    }

    /// `get_view` (mtmd-helper-common.h:151 @5266f24da): `n_tokens` tokens
    /// from `offset`; an M-RoPE view copies its positions into `pos_view`.
    pub fn getView(self: *EmbdBatch, offset: c_int, n_tokens: c_int) c.llama_batch {
        impl.ggmlAssert(offset >= 0 and n_tokens > 0 and offset + n_tokens <= self.batch.n_tokens, "mtmd-helper-common.h", 152, "offset >= 0 && n_tokens > 0 && offset + n_tokens <= batch.n_tokens");
        const off: usize = @intCast(offset);
        const n: usize = @intCast(n_tokens);
        var pos_ptr: [*]c.llama_pos = undefined;
        self.pos_view.clearRetainingCapacity();
        self.pos_view.ensureTotalCapacity(gpa, n * @as(usize, @intCast(self.n_pos_per_embd))) catch oom();
        if (self.n_pos_per_embd > 1) {
            // mrope
            // for example, with layout of src: 1234...1234...1234...1234...
            //       offset 2 will give us dst: 34...34...34...34...
            var i: usize = 0;
            while (i < @as(usize, @intCast(self.n_pos_per_embd))) : (i += 1) {
                const src_idx = i * @as(usize, @intCast(self.batch.n_tokens)) + off;
                self.pos_view.appendSliceAssumeCapacity(self.pos.items[src_idx..][0..n]);
            }
            pos_ptr = self.pos_view.items.ptr;
        } else {
            // normal
            pos_ptr = self.pos.items.ptr + off;
        }
        return .{
            .n_tokens = n_tokens,
            .token = null,
            .embd = if (self.batch.embd) |e| e + off * @as(usize, @intCast(self.n_mmproj_embd)) else null,
            .pos = pos_ptr,
            .n_seq_id = self.batch.n_seq_id + off,
            .seq_id = self.batch.seq_id + off,
            .logits = self.batch.logits + off,
        };
    }
};

fn oom() noreturn {
    impl.abortUncaught("std::bad_alloc", "std::bad_alloc");
}

fn resizeZero(comptime T: type, list: *std.ArrayList(T), n: usize) void {
    const old = list.items.len;
    list.resize(gpa, n) catch oom();
    if (n > old) @memset(list.items[old..], std.mem.zeroes(T));
}

// -----------------------------------------------------------------------------
// mtmd-helper.cpp

/// Ports `mtmd_helper_log_set` (mtmd-helper.cpp:56 @5266f24da): the
/// helpers' logger and libmtmd's together.
pub fn mtmd_helper_log_set(log_callback_in: c.ggml_log_callback, user_data: ?*anyopaque) callconv(.c) void {
    const log_callback = log_callback_in orelse g_logger.default_callback;
    g_logger.log_callback = log_callback;
    g_logger.log_callback_user_data = user_data;
    mtmd.mtmd_log_set(log_callback, user_data);
}

/// Ports `mtmd_helper_get_n_tokens` (mtmd-helper.cpp:69 @5266f24da).
pub fn mtmd_helper_get_n_tokens(chunks: *const mtmd.InputChunks) callconv(.c) usize {
    var n_tokens: usize = 0;
    var i: usize = 0;
    while (i < mtmd.mtmd_input_chunks_size(chunks)) : (i += 1) {
        const chunk = mtmd.mtmd_input_chunks_get(chunks, i).?;
        n_tokens +%= mtmd.mtmd_input_chunk_get_n_tokens(chunk);
    }
    return n_tokens;
}

/// Ports `mtmd_helper_get_n_pos` (mtmd-helper.cpp:78 @5266f24da).
pub fn mtmd_helper_get_n_pos(chunks: *const mtmd.InputChunks) callconv(.c) c.llama_pos {
    var n_pos: c.llama_pos = 0;
    var i: usize = 0;
    while (i < mtmd.mtmd_input_chunks_size(chunks)) : (i += 1) {
        const chunk = mtmd.mtmd_input_chunks_get(chunks, i).?;
        n_pos +%= mtmd.mtmd_input_chunk_get_n_pos(chunk);
    }
    return n_pos;
}

/// Ports `mtmd_helper_image_get_decoder_pos` (mtmd-helper.cpp:87 @5266f24da).
pub fn mtmd_helper_image_get_decoder_pos(chunks: *const mtmd.ImageTokens, pos_0: c.llama_pos, out_pos: [*]mtmd.DecoderPos) callconv(.c) void {
    const n_tokens = mtmd.mtmd_image_tokens_get_n_tokens(chunks);
    for (0..n_tokens) |i| out_pos[i] = mtmd.mtmd_image_tokens_get_decoder_pos(chunks, pos_0, i);
}

/// Ports `mtmd_helper_decode_image_chunk` (mtmd-helper.cpp:118 @5266f24da):
/// decodes media embeddings already computed, in batches of `n_batch`.
/// `scope_non_causal` (mtmd-helper.cpp:95 @5266f24da) is the `defer`.
pub fn mtmd_helper_decode_image_chunk(
    ctx: *Context,
    lctx: ?*c.struct_llama_context,
    chunk: *const mtmd.InputChunk,
    encoded_embd: ?[*]f32,
    n_past_in: c.llama_pos,
    seq_id: c.llama_seq_id,
    n_batch: i32,
    new_n_past: *c.llama_pos,
    callback: PostDecodeCallback,
    user_data: ?*anyopaque,
) callconv(.c) i32 {
    var n_past = n_past_in;
    impl.ggmlAssert(n_batch > 0, "mtmd-helper.cpp", 129, "n_batch > 0");
    const chunk_type = mtmd.mtmd_input_chunk_get_type(chunk);
    const name: [*:0]const u8 = if (chunk_type == .image) "image" else "audio";
    if (chunk_type == .text) {
        logErr("failed to decode chunk: input chunk not of image/audio type\n", .{});
        return -1;
    }

    const model = lapi.context.llama_get_model(lctx);
    const n_mmproj_embd = lapi.model.llama_model_n_embd_inp(model);
    const n_pos_per_embd: c_int = if (mtmd.mtmd_decode_use_mrope(ctx)) 4 else 1;

    const n_tokens: i32 = @truncate(@as(isize, @bitCast(mtmd.mtmd_input_chunk_get_n_tokens(chunk))));
    var i_batch: i32 = 0;
    const n_img_batches: i32 = @divTrunc(n_tokens + n_batch - 1, n_batch);
    const batch_embd = EmbdBatch.create(encoded_embd, n_tokens, n_pos_per_embd, n_mmproj_embd);
    defer batch_embd.destroy();

    if (mtmd.mtmd_decode_use_mrope(ctx)) {
        if (chunk_type == .image) {
            const image_tokens = mtmd.mtmd_input_chunk_get_tokens_image(chunk) orelse {
                logErr("failed to decode chunk: image tokens are null\n", .{});
                return -1;
            };
            const n_img_tokens = mtmd.mtmd_image_tokens_get_n_tokens(image_tokens);
            const rel_pos = gpa.alloc(mtmd.DecoderPos, n_img_tokens) catch oom();
            defer gpa.free(rel_pos);
            @memset(rel_pos, std.mem.zeroes(mtmd.DecoderPos));
            mtmd_helper_image_get_decoder_pos(image_tokens, n_past, rel_pos.ptr);
            batch_embd.setPositionMrope2d(rel_pos, seq_id);
        } else if (chunk_type == .audio) {
            batch_embd.setPositionMrope1d(n_past, seq_id);
        } else {
            impl.abort("mtmd-helper.cpp", 160, "invalid chunk type for M-RoPE");
        }
    } else {
        batch_embd.setPositionNormal(n_past, seq_id);
    }

    const use_non_causal = mtmd.mtmd_decode_use_non_causal(ctx, chunk);
    // TODO @ngxson : need to make sure only one image is processed at a time, and n_ubatch must be enough to hold the image
    if (use_non_causal) lapi.context.llama_set_causal_attn(lctx, false);
    defer if (use_non_causal) lapi.context.llama_set_causal_attn(lctx, true);

    while (i_batch < n_img_batches) { // split into batches
        const pos_offset = i_batch * n_batch;
        const n_tokens_batch = @min(n_batch, n_tokens - pos_offset);
        const batch_embd_view = batch_embd.getView(pos_offset, n_tokens_batch);

        logInf("decoding %s batch %d/%d, n_tokens_batch = %d\n", .{ name, i_batch + 1, n_img_batches, n_tokens_batch });

        const t1 = c.ggml_time_ms();
        var ret = lapi.context.llama_decode(lctx, batch_embd_view);
        if (ret != 0) {
            logErr("failed to decode %s\n", .{name});
            return ret;
        }

        if (callback) |cb| {
            ret = cb(batch_embd_view, user_data);
            if (ret != 0) {
                logErr("post-decode callback failed\n", .{});
                return ret;
            }
        }

        logInf("%s decoded (batch %d/%d) in %lld ms\n", .{ name, i_batch + 1, n_img_batches, @as(c_longlong, c.ggml_time_ms() - t1) });

        i_batch += 1;
    }

    n_past +%= mtmd.mtmd_input_chunk_get_n_pos(chunk);
    new_n_past.* = n_past;

    return 0;
}

/// Ports `mtmd_helper_eval_chunk_single` (mtmd-helper.cpp:202 @5266f24da).
pub fn mtmd_helper_eval_chunk_single(
    ctx: *Context,
    lctx: ?*c.struct_llama_context,
    chunk: *const mtmd.InputChunk,
    n_past_in: c.llama_pos,
    seq_id: c.llama_seq_id,
    n_batch: i32,
    logits_last: bool,
    new_n_past: *c.llama_pos,
) callconv(.c) i32 {
    var n_past = n_past_in;
    impl.ggmlAssert(n_batch > 0, "mtmd-helper.cpp", 210, "n_batch > 0");
    var ret: i32 = undefined;
    var text_batch = lapi.batch.llama_batch_init(n_batch, 0, 1);
    const chunk_type = mtmd.mtmd_input_chunk_get_type(chunk);

    if (chunk_type == .text) {
        var n_tokens: usize = 0;
        const tokens = mtmd.mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        var i: usize = 0;
        while (i < n_tokens) { // split into batches
            text_batch.n_tokens = 0; // clear the batch
            while (i < n_tokens and text_batch.n_tokens < n_batch) : (i += 1) {
                const j: usize = @intCast(text_batch.n_tokens);
                text_batch.token[j] = tokens.?[i];
                text_batch.pos[j] = n_past;
                n_past +%= 1;
                text_batch.n_seq_id[j] = 1;
                text_batch.seq_id[j][0] = seq_id;
                text_batch.logits[j] = 0;

                text_batch.n_tokens += 1;
            }
            const is_last_token = (i == n_tokens);
            if (logits_last and is_last_token) {
                text_batch.logits[@intCast(text_batch.n_tokens - 1)] = 1;
            }
            ret = lapi.context.llama_decode(lctx, text_batch);
            if (ret != 0) {
                logErr("failed to decode text\n", .{});
                lapi.batch.llama_batch_free(text_batch);
                return ret;
            }
            new_n_past.* +%= text_batch.n_tokens;
        }
    } else if (chunk_type == .image or chunk_type == .audio) {
        const name: [*:0]const u8 = if (chunk_type == .image) "image" else "audio";
        const t0 = c.ggml_time_ms();

        logInf("encoding %s slice...\n", .{name});

        ret = mtmd.mtmd_encode_chunk(ctx, chunk);
        if (ret != 0) {
            logErr("failed to encode %s slice\n", .{name});
            lapi.batch.llama_batch_free(text_batch);
            return ret;
        }

        logInf("%s slice encoded in %lld ms\n", .{ name, @as(c_longlong, c.ggml_time_ms() - t0) });

        const embd = mtmd.mtmd_get_output_embd(ctx);
        ret = mtmd_helper_decode_image_chunk(ctx, lctx, chunk, embd, n_past, seq_id, n_batch, new_n_past, null, null);
        if (ret != 0) {
            logErr("failed to decode %s\n", .{name});
            lapi.batch.llama_batch_free(text_batch);
            return ret;
        }
    } else {
        impl.abort("mtmd-helper.cpp", 268, "chunk type not supported");
    }

    lapi.batch.llama_batch_free(text_batch);
    return 0;
}

/// Ports `mtmd_helper_eval_chunks` (mtmd-helper.cpp:275 @5266f24da).
pub fn mtmd_helper_eval_chunks(
    ctx: *Context,
    lctx: ?*c.struct_llama_context,
    chunks: *const mtmd.InputChunks,
    n_past_in: c.llama_pos,
    seq_id: c.llama_seq_id,
    n_batch: i32,
    logits_last: bool,
    new_n_past: *c.llama_pos,
) callconv(.c) i32 {
    var n_past = n_past_in;
    const n_chunks = mtmd.mtmd_input_chunks_size(chunks);
    if (n_chunks == 0) {
        logWrn("no chunks to eval\n", .{});
        return 0;
    }

    for (0..n_chunks) |i| {
        const chunk_logits_last = (i == n_chunks - 1) and logits_last;
        const chunk = mtmd.mtmd_input_chunks_get(chunks, i).?;

        const res = mtmd_helper_eval_chunk_single(ctx, lctx, chunk, n_past, seq_id, n_batch, chunk_logits_last, &n_past);
        if (res != 0) {
            logErr("failed to eval chunk %zu\n", .{i});
            return res;
        }
        new_n_past.* = n_past;
    }

    return 0;
}

/// Ports `is_audio_file` (mtmd-helper.cpp:306 @5266f24da), a member of `audio_helpers`.
fn isAudioFile(buf: []const u8) bool {
    if (buf.len < 12) return false;

    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    const is_wav = std.mem.eql(u8, buf[0..4], "RIFF") and std.mem.eql(u8, buf[8..12], "WAVE");
    const is_mp3 = buf.len >= 3 and (std.mem.eql(u8, buf[0..3], "ID3") or
        // Check for MPEG sync word (simplified check)
        (buf[0] == 0xFF and (buf[1] & 0xE0) == 0xE0));
    const is_flac = std.mem.eql(u8, buf[0..4], "fLaC");

    return is_wav or is_mp3 or is_flac;
}

/// `decode_audio_from_buf` (mtmd-helper.cpp:325 @5266f24da), a member of `audio_helpers`,
/// which is miniaudio's decoder, in `vendor/`.
fn decodeAudioFromBuf(buf: []const u8, target_sample_rate: c_int, out: *std.ArrayList(f32)) bool {
    return vendor.miniaudio.decodeAudioFromBuf(gpa, buf, target_sample_rate, out);
}

/// SHA-256 of `buf` in lowercase hex: `hash_sha256_hex` (vendor/hash/hash.cpp:19
/// @5266f24da), the standard digest.
fn sha256Hex(buf: []const u8) [64:0]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(buf, &digest, .{});
    var out: [64:0]u8 = undefined;
    const hex = "0123456789abcdef";
    for (digest, 0..) |b, i| {
        out[2 * i] = hex[b >> 4];
        out[2 * i + 1] = hex[b & 0xf];
    }
    out[64] = 0;
    return out;
}

/// Ports `mtmd_helper_bitmap_init_from_buf` (mtmd-helper.cpp:376 @5266f24da):
/// audio by its magic bytes, else an image stb_image can read. The ID is
/// the buffer's SHA-256 unless `placeholder`. `opt` is only read by the
/// video paths, which this build does not compile.
pub fn mtmd_helper_bitmap_init_from_buf(ctx: *const Context, buf_ptr: ?[*]const u8, len: usize, placeholder: bool, opt: InitOpt) callconv(.c) BitmapWrapper {
    _ = opt; // only used by video code paths
    const buf: []const u8 = if (len == 0) &.{} else buf_ptr.?[0..len];
    // calculate the hash if needed
    var id_buf: [64:0]u8 = undefined;
    var id: ?[*:0]const u8 = null;

    if (!placeholder) {
        // use sha256 to prevent cache poisoning
        id_buf = sha256Hex(buf);
        id = &id_buf;
    }

    if (isAudioFile(buf)) {
        var pcmf32: std.ArrayList(f32) = .empty;
        defer pcmf32.deinit(gpa);
        const sample_rate = mtmd.mtmd_get_audio_sample_rate(ctx);
        if (sample_rate < 0) {
            logErr("This model does not support audio input\n", .{});
            return .{ .bitmap = null, .video_ctx = null };
        }
        if (!decodeAudioFromBuf(buf, sample_rate, &pcmf32)) {
            logErr("Unable to read WAV audio file from buffer\n", .{});
            return .{ .bitmap = null, .video_ctx = null };
        }
        const result = mtmd.mtmd_bitmap_init_from_audio(pcmf32.items.len, if (placeholder) null else pcmf32.items.ptr);
        mtmd.mtmd_bitmap_set_id(result, id);
        return .{ .bitmap = result, .video_ctx = null };
    }

    // otherwise, we assume it's an image
    switch (vendor.stb_image.loadFromMemory(gpa, buf, 3)) {
        .ok => |img| {
            defer gpa.free(img.pixels);
            const result = mtmd.mtmd_bitmap_init(@bitCast(img.x), @bitCast(img.y), if (placeholder) null else img.pixels.ptr);
            mtmd.mtmd_bitmap_set_id(result, id);
            return .{ .bitmap = result, .video_ctx = null };
        },
        // otherwise, fallthrough to video decoding (if supported)
        .failed => {},
    }

    logErr("%s: failed to decode buffer as either image or audio (video support not compiled in)\n", .{"mtmd_helper_bitmap_init_from_buf"});
    return .{ .bitmap = null, .video_ctx = null };
}

/// Ports `mtmd_helper_bitmap_init_from_file` (mtmd-helper.cpp:462 @5266f24da).
pub fn mtmd_helper_bitmap_init_from_file(ctx: *const Context, fname: [*:0]const u8, placeholder: bool, opt: InitOpt) callconv(.c) BitmapWrapper {
    const f = openFile(fname) orelse return .{ .bitmap = null, .video_ctx = null };

    _ = fseek(f, 0, seek_end);
    const file_size = ftell(f);
    _ = fseek(f, 0, seek_set);
    if (file_size < 0) {
        logErr("Failed to get file size of %s\n", .{fname});
        _ = fclose(f);
        return .{ .bitmap = null, .video_ctx = null };
    }
    const buf = gpa.alloc(u8, @intCast(file_size)) catch oom();
    defer gpa.free(buf);
    @memset(buf, 0);

    const n_read = fread(buf.ptr, 1, buf.len, f);
    _ = fclose(f);
    if (n_read != buf.len) {
        logErr("Failed to read entire file %s", .{fname});
        return .{ .bitmap = null, .video_ctx = null };
    }

    return mtmd_helper_bitmap_init_from_buf(ctx, buf.ptr, buf.len, placeholder, opt);
}

/// `fopen(fname, "rb")`, or on Windows `_wfopen` of the UTF-16 name. Logs
/// and returns null where the C++ does.
fn openFile(fname: [*:0]const u8) ?*anyopaque {
    if (comptime is_windows) {
        const cp_utf8: c_uint = 65001;
        var wlen = win32.MultiByteToWideChar(cp_utf8, 0, fname, -1, null, 0);
        if (wlen == 0) {
            logErr("Unable to convert filename to UTF-16: %s\n", .{fname});
            return null;
        }
        const wfname = gpa.alloc(u16, @intCast(wlen)) catch oom();
        defer gpa.free(wfname);
        @memset(wfname, 0);
        wlen = win32.MultiByteToWideChar(cp_utf8, 0, fname, -1, wfname.ptr, wlen);
        if (wlen == 0) {
            logErr("Unable to convert filename to UTF-16: %s\n", .{fname});
            return null;
        }
        const f = win32._wfopen(@ptrCast(wfname.ptr), std.unicode.utf8ToUtf16LeStringLiteral("rb"));
        if (f == null) logErr("Unable to open file %s: %s\n", .{ fname, strerror(std.c._errno().*) });
        return f;
    }
    const f = fopen(fname, "rb");
    if (f == null) logErr("Unable to open file %s: %s\n", .{ fname, strerror(std.c._errno().*) });
    return f;
}

/// Ports `mtmd_helper_support_video` (mtmd-helper.cpp:507 @5266f24da): video
/// is not compiled in.
pub fn mtmd_helper_support_video(ctx: ?*const Context) callconv(.c) bool {
    _ = ctx;
    return false;
}

/// Ports `mtmd_helper_video_init_params_default` (mtmd-helper.cpp:854
/// @5266f24da).
pub fn mtmd_helper_video_init_params_default() callconv(.c) VideoInitParams {
    return .{ .fps_target = 4.0, .ffmpeg_bin_dir = null, .timestamp_interval_ms = 5000 };
}

/// Ports `mtmd_helper_init_opt_default` (mtmd-helper.cpp:862 @5266f24da).
pub fn mtmd_helper_init_opt_default() callconv(.c) InitOpt {
    return .{ .video_params = mtmd_helper_video_init_params_default() };
}

/// Ports `mtmd_helper_video_init` (mtmd-helper.cpp:915 @5266f24da), its
/// `#else` arm.
pub fn mtmd_helper_video_init(mctx: ?*const Context, path: ?[*:0]const u8, params: VideoInitParams) callconv(.c) ?*anyopaque {
    _ = mctx;
    _ = path;
    _ = params;
    logErr("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", .{"mtmd_helper_video_init"});
    return null;
}

/// Ports `mtmd_helper_video_init_from_buf` (mtmd-helper.cpp:950 @5266f24da),
/// its `#else` arm.
pub fn mtmd_helper_video_init_from_buf(mctx: ?*const Context, buf: ?[*]const u8, len: usize, params: VideoInitParams) callconv(.c) ?*anyopaque {
    _ = mctx;
    _ = buf;
    _ = len;
    _ = params;
    logErr("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", .{"mtmd_helper_video_init_from_buf"});
    return null;
}

/// Ports `mtmd_helper_video_free` (mtmd-helper.cpp:986 @5266f24da), its
/// `#else` arm.
pub fn mtmd_helper_video_free(ctx: ?*anyopaque) callconv(.c) void {
    _ = ctx;
    logErr("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", .{"mtmd_helper_video_free"});
}

/// Ports `mtmd_helper_video_get_info` (mtmd-helper.cpp:997 @5266f24da), its
/// `#else` arm.
pub fn mtmd_helper_video_get_info(ctx: ?*const anyopaque) callconv(.c) VideoInfo {
    _ = ctx;
    impl.ggmlAssert(false, "mtmd-helper.cpp", 1002, "false && \"video is not supported in this build (MTMD_VIDEO is set to OFF)\"");
    unreachable;
}

/// Ports `mtmd_helper_video_read_next` (mtmd-helper.cpp:1006 @5266f24da), its
/// `#else` arm.
pub fn mtmd_helper_video_read_next(ctx: ?*anyopaque, out_bitmap: ?*?*mtmd.Bitmap, out_text: ?*?[*:0]u8) callconv(.c) i32 {
    _ = ctx;
    _ = out_bitmap;
    _ = out_text;
    impl.ggmlAssert(false, "mtmd-helper.cpp", 1015, "false && \"video is not supported in this build (MTMD_VIDEO is set to OFF)\"");
    unreachable;
}

/// Ports `mtmd_helper_model_can_chat` (mtmd-helper.cpp:1019 @5266f24da): a
/// TTS-only model, which has no chat template, cannot chat.
pub fn mtmd_helper_model_can_chat(lctx: ?*const c.struct_llama_context, mctx: ?*const Context) callconv(.c) bool {
    const m = mctx orelse return true;

    const model = lapi.context.llama_get_model(lctx);
    const tmpl = lapi.model.llama_model_chat_template(model, null);
    const info = mtmd.mtmd_gen_audio_get_info(m);

    // tts-only model cannot be used for chat (no chat template)
    const is_tts_only = info.type != .none and tmpl == null;

    return !is_tts_only;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the SHA-256 id is the standard digest in lowercase hex" {
    const h = sha256Hex("abc");
    try std.testing.expectEqualStrings("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", &h);
}

test "audio magic bytes" {
    try std.testing.expect(isAudioFile("RIFF\x00\x00\x00\x00WAVEfmt "));
    try std.testing.expect(isAudioFile("ID3\x04\x00\x00\x00\x00\x00\x00\x00\x00"));
    try std.testing.expect(isAudioFile("fLaC\x00\x00\x00\x00\x00\x00\x00\x00"));
    try std.testing.expect(!isAudioFile("RIFF\x00\x00\x00\x00WEBPVP8 "));
    try std.testing.expect(!isAudioFile("fLaC"));
}
