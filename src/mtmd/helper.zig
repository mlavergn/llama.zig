//! libmtmd's helpers: loading a bitmap from a file or buffer, evaluating
//! chunks into a llama context, and the helpers' own logger.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/mtmd-helper.cpp` and
//! `llama.cpp/tools/mtmd/mtmd-helper-common.h`, plus the C types of
//! `llama.cpp/tools/mtmd/mtmd-helper.h`, at v1.0.0 (`b1b770d90`). Each
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
const mod = @import("module.zig");
const llama = @import("llama");
const stdlib = @import("stdlib");
const builtin = @import("builtin");

const is_windows = builtin.os.tag == .windows;
const win32 = struct {
    extern "kernel32" fn MultiByteToWideChar(code_page: c_uint, flags: u32, mbs: [*]const u8, mbs_len: c_int, wcs: ?[*]u16, wcs_len: c_int) callconv(.winapi) c_int;
};

const seek_set: c_int = 0;
const seek_end: c_int = 2;

// -----------------------------------------------------------------------------
// mtmd-helper.h

/// Ports `mtmd_helper_bitmap_wrapper` (mtmd-helper.h:49 @b1b770d90).
pub const BitmapWrapper = extern struct {
    bitmap: ?*mod.mtmd.Bitmap,
    video_ctx: ?*anyopaque,
};

/// Ports `mtmd_helper_embd_batch` (mtmd-helper.h:132 @b1b770d90): one decoded sub-batch of embeddings, passed to
/// `mtmd_helper_post_decode_callback`.
pub const EmbdBatchView = extern struct {
    n_tokens: i32,
    embd: ?[*]const f32, // [n_tokens, n_embd]
    n_embd: i32,
    pos: ?[*]const llama.impl.llama_c.llama_pos, // [n_pos, n_tokens], section-major
    n_pos: i32, // 4 for M-RoPE models, 1 otherwise
    seq_id: llama.impl.llama_c.llama_seq_id,
};

/// Ports `mtmd_helper_post_decode_callback` (mtmd-helper.h:141 @b1b770d90):
/// it takes the sub-batch's view, not a `llama_batch`.
pub const PostDecodeCallback = ?*const fn (?*const EmbdBatchView, ?*anyopaque) callconv(.c) i32;

/// Ports `mtmd_helper_video_info` (mtmd-helper.h:166 @b1b770d90).
pub const VideoInfo = extern struct {
    width: u32,
    height: u32,
    fps: f32,
    n_frames: i32,
};

/// Ports `mtmd_helper_video_init_params` (mtmd-helper.h:26 @b1b770d90).
pub const VideoInitParams = extern struct {
    fps_target: f32, // desired output fps; <= 0 means use the video's native fps, defaulted to 4.0f
    ffmpeg_bin_dir: ?[*:0]const u8, // directory containing ffmpeg/ffprobe binaries; NULL means search PATH
    timestamp_interval_ms: i64, // interval for adding timestamp as text chunk (example: "[10m50.5s]"); <= 0 means no timestamp, defaulted to 5000ms
    no_soundtrack: bool, // frames only, also for models that hear the soundtrack of a video (LFM3-Omni)
};

/// Ports `mtmd_helper_cap` (mtmd-helper.h:200 @b1b770d90).
pub const Cap = extern struct {
    chat: bool, // false for tts-only models (no chat template) and realtime transcription models (token delay)

    // input capabilities
    vision: bool,
    audio: bool,
    video: bool, // also depending on MTMD_VIDEO build option

    // realtime capabilities
    audio_stream: bool, // realtime transcription, see mtmd_audio_stream_init()

    // output capabilities
    text_gen: bool, // false for an audio-only model (tts-only, no chat template)
    audio_gen: bool,
    audio_gen_spk_ref: bool, // accepts a speaker reference audio for voice cloning
};

/// Ports `mtmd_helper_init_opt` (mtmd-helper.h:37 @b1b770d90): the options
/// of `mtmd_helper_bitmap_init_from_*()`.
pub const InitOpt = extern struct {
    video_params: VideoInitParams,
};

// -----------------------------------------------------------------------------
// mtmd-helper-common.h: logging

fn defaultCallback(level: llama.impl.llama_c.enum_ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = level;
    _ = user_data;
    const f = stdlib.stdio.cStream(.stderr);
    _ = stdlib.libc.fputs(text, f);
    _ = stdlib.libc.fflush(f);
}

/// Ports `mtmd_helper_logger` (mtmd-helper-common.h:21 @b1b770d90): the
/// helpers' logger, separate from libmtmd's.
pub const Logger = struct {
    default_callback: llama.impl.llama_c.ggml_log_callback = defaultCallback,
    log_callback: llama.impl.llama_c.ggml_log_callback = defaultCallback,
    log_callback_user_data: ?*anyopaque = null,

    /// `log` (mtmd-helper-common.h:52 @b1b770d90): format into 128 bytes,
    /// or a heap buffer when that is too short.
    pub fn log(self: *const Logger, level: llama.impl.llama_c.enum_ggml_log_level, comptime fmt: [*:0]const u8, args: anytype) void {
        var buffer: [128]u8 = undefined;
        const len = @call(.auto, stdlib.libc.snprintf, .{ &buffer, buffer.len, fmt } ++ args);
        const cb = self.log_callback orelse return;
        if (len < 128) {
            cb(level, @ptrCast(&buffer), self.log_callback_user_data);
        } else {
            const n: usize = @intCast(len);
            const buffer2 = std.heap.c_allocator.alloc(u8, n + 1) catch return;
            defer std.heap.c_allocator.free(buffer2);
            @memset(buffer2, 0);
            _ = @call(.auto, stdlib.libc.snprintf, .{ buffer2.ptr, n + 1, fmt } ++ args);
            buffer2[n] = 0;
            cb(level, @ptrCast(buffer2.ptr), self.log_callback_user_data);
        }
    }
};

/// `g_logger` (mtmd-helper-common.h:61 @b1b770d90): one instance for both
/// helper translation units.
pub var g_logger: Logger = .{};

pub fn logDbg(comptime fmt: [*:0]const u8, args: anytype) void {
    g_logger.log(llama.impl.llama_c.GGML_LOG_LEVEL_DEBUG, fmt, args);
}
pub fn logInf(comptime fmt: [*:0]const u8, args: anytype) void {
    g_logger.log(llama.impl.llama_c.GGML_LOG_LEVEL_INFO, fmt, args);
}
pub fn logWrn(comptime fmt: [*:0]const u8, args: anytype) void {
    g_logger.log(llama.impl.llama_c.GGML_LOG_LEVEL_WARN, fmt, args);
}
pub fn logErr(comptime fmt: [*:0]const u8, args: anytype) void {
    g_logger.log(llama.impl.llama_c.GGML_LOG_LEVEL_ERROR, fmt, args);
}

// -----------------------------------------------------------------------------
// mtmd-helper-common.h: the embedding batch

/// Ports `decode_embd_batch` (mtmd-helper-common.h:74 @b1b770d90): precomputed
/// embeddings and their positions, rendered into a `llama_batch_ext` one
/// sub-batch at a time.
pub const EmbdBatch = struct {
    n_pos_per_embd: c_int,
    n_mmproj_embd: c_int,
    n_tokens: i32,
    embd: ?[*]const f32, // [n_tokens, n_mmproj_embd], not owned
    pos: std.ArrayList(llama.impl.llama_c.llama_pos) = .empty, // [n_pos_per_embd, n_tokens], section-major
    pos_view: std.ArrayList(llama.impl.llama_c.llama_pos) = .empty, // sliced positions of the last get_view()
    logits: std.ArrayList(i8) = .empty,
    seq_id: llama.impl.llama_c.llama_seq_id = 0,

    batch: ?*llama.impl.llama_c.struct_llama_batch_ext = null, // rendered sub-batch, see render()

    /// The constructor `decode_embd_batch` (mtmd-helper-common.h:86 @b1b770d90).
    /// Heap-allocated, as the callers hold it by pointer.
    pub fn create(embd: ?[*]const f32, n_tokens: i32, n_pos_per_embd: c_int, n_mmproj_embd: c_int) *EmbdBatch {
        mod.impl.ggmlAssert(n_tokens > 0 and n_pos_per_embd > 0 and n_mmproj_embd > 0, "mtmd-helper-common.h", 88, "n_tokens > 0 && n_pos_per_embd > 0 && n_mmproj_embd > 0");
        const self = std.heap.c_allocator.create(EmbdBatch) catch oom();
        self.* = .{ .n_pos_per_embd = n_pos_per_embd, .n_mmproj_embd = n_mmproj_embd, .n_tokens = n_tokens, .embd = embd };
        const n: usize = @intCast(n_tokens);
        resizeZero(llama.impl.llama_c.llama_pos, &self.pos, n * @as(usize, @intCast(n_pos_per_embd)));
        resizeZero(i8, &self.logits, n);
        return self;
    }

    /// The destructor: `batch`'s `llama_batch_ext_ptr` frees it.
    pub fn destroy(self: *EmbdBatch) void {
        if (self.batch) |b| llama.api.batch.llama_batch_ext_free(b);
        self.pos.deinit(std.heap.c_allocator);
        self.pos_view.deinit(std.heap.c_allocator);
        self.logits.deinit(std.heap.c_allocator);
        std.heap.c_allocator.destroy(self);
    }

    /// `set_position_normal` (mtmd-helper-common.h:93 @b1b770d90).
    pub fn setPositionNormal(self: *EmbdBatch, pos_0: llama.impl.llama_c.llama_pos, seq_id: llama.impl.llama_c.llama_seq_id) void {
        self.seq_id = seq_id;
        var i: i32 = 0;
        while (i < self.n_tokens) : (i += 1) {
            self.pos.items[@intCast(i)] = pos_0 +% i;
        }
    }

    /// `set_position_mrope_2d` (mtmd-helper-common.h:101 @b1b770d90): M-RoPE
    /// for image.
    pub fn setPositionMrope2d(self: *EmbdBatch, rel_pos: []const mod.mtmd.DecoderPos, seq_id: llama.impl.llama_c.llama_seq_id) void {
        mod.impl.ggmlAssert(self.n_pos_per_embd == 4, "mtmd-helper-common.h", 102, "n_pos_per_embd == 4");
        mod.impl.ggmlAssert(rel_pos.len != 0 and @as(i32, @truncate(@as(isize, @intCast(rel_pos.len)))) == self.n_tokens, "mtmd-helper-common.h", 103, "!rel_pos.empty() && (int32_t)rel_pos.size() == n_tokens");
        self.seq_id = seq_id;
        const n: usize = @intCast(self.n_tokens);
        for (0..n) |idx| {
            self.pos.items[idx] = @bitCast(rel_pos[idx].t);
            self.pos.items[idx + n] = @bitCast(rel_pos[idx].y);
            self.pos.items[idx + n * 2] = @bitCast(rel_pos[idx].x);
            self.pos.items[idx + n * 3] = @bitCast(rel_pos[idx].z);
        }
    }

    /// `set_position_mrope_1d` (mtmd-helper-common.h:116 @b1b770d90): M-RoPE
    /// for audio.
    pub fn setPositionMrope1d(self: *EmbdBatch, pos_0: llama.impl.llama_c.llama_pos, seq_id: llama.impl.llama_c.llama_seq_id) void {
        mod.impl.ggmlAssert(self.n_pos_per_embd == 4, "mtmd-helper-common.h", 117, "n_pos_per_embd == 4");
        self.seq_id = seq_id;
        const n: usize = @intCast(self.n_tokens);
        for (0..n) |idx| {
            const p = pos_0 +% @as(i32, @intCast(idx));
            self.pos.items[idx] = p;
            self.pos.items[idx + n] = p;
            self.pos.items[idx + n * 2] = p;
            self.pos.items[idx + n * 3] = p;
        }
    }

    /// `get_view` (mtmd-helper-common.h:130 @b1b770d90): describes the
    /// entries `[offset, offset + n)` with section-major positions, copied
    /// into `pos_view`; valid until the next call.
    pub fn getView(self: *EmbdBatch, offset: c_int, n: c_int) EmbdBatchView {
        mod.impl.ggmlAssert(offset >= 0 and n > 0 and offset + n <= self.n_tokens, "mtmd-helper-common.h", 131, "offset >= 0 && n > 0 && offset + n <= n_tokens");
        const off: usize = @intCast(offset);
        const nn: usize = @intCast(n);
        self.pos_view.clearRetainingCapacity();
        self.pos_view.ensureTotalCapacity(std.heap.c_allocator, nn * @as(usize, @intCast(self.n_pos_per_embd))) catch oom();
        var j: usize = 0;
        while (j < @as(usize, @intCast(self.n_pos_per_embd))) : (j += 1) {
            const src = j * @as(usize, @intCast(self.n_tokens)) + off;
            self.pos_view.appendSliceAssumeCapacity(self.pos.items[src..][0..nn]);
        }
        return .{
            .n_tokens = n,
            .embd = if (self.embd) |e| e + off * @as(usize, @intCast(self.n_mmproj_embd)) else null,
            .n_embd = self.n_mmproj_embd,
            .pos = self.pos_view.items.ptr,
            .n_pos = self.n_pos_per_embd,
            .seq_id = self.seq_id,
        };
    }

    /// `render` (mtmd-helper-common.h:149 @b1b770d90): the entries
    /// `[offset, offset + n)` as a batch this object owns, ready for
    /// `llama_process`.
    pub fn render(self: *EmbdBatch, lctx: ?*llama.impl.llama_c.struct_llama_context, offset: c_int, n: c_int) ?*llama.impl.llama_c.struct_llama_batch_ext {
        mod.impl.ggmlAssert(offset >= 0 and n > 0 and offset + n <= self.n_tokens, "mtmd-helper-common.h", 150, "offset >= 0 && n > 0 && offset + n <= n_tokens");
        if (self.batch == null) {
            self.batch = llama.api.batch.llama_batch_ext_init(lctx);
        }
        llama.api.batch.llama_batch_ext_clear(self.batch);
        const n_tokens: usize = @intCast(self.n_tokens);
        var i: usize = @intCast(offset);
        while (i < @as(usize, @intCast(offset + n))) : (i += 1) {
            const e: llama.impl.llama_c.struct_llama_embd = .{
                .data = if (self.embd) |p| p + i * @as(usize, @intCast(self.n_mmproj_embd)) else null,
                .n_rows = 1,
                .n_embd = @intCast(self.n_mmproj_embd),
            };
            const idx = llama.api.batch.llama_batch_ext_add_embd(self.batch, self.seq_id, e);
            mod.impl.ggmlAssert(idx >= 0, "mtmd-helper-common.h", 158, "idx >= 0");

            var p = [_]llama.impl.llama_c.llama_pos{ 0, 0, 0, 0 }; // GGML_MROPE_SECTIONS
            var j: usize = 0;
            while (j < @as(usize, @intCast(self.n_pos_per_embd))) : (j += 1) {
                p[j] = self.pos.items[j * n_tokens + i];
            }
            _ = llama.api.batch.llama_batch_ext_set_pos(self.batch, idx, &p);

            if (self.logits.items[i] != 0) {
                _ = llama.api.batch.llama_batch_ext_set_output_logits(self.batch, idx, true);
            }
        }
        return self.batch;
    }
};

fn oom() noreturn {
    mod.impl.abortUncaught("std::bad_alloc", "std::bad_alloc");
}

fn resizeZero(comptime T: type, list: *std.ArrayList(T), n: usize) void {
    const old = list.items.len;
    list.resize(std.heap.c_allocator, n) catch oom();
    if (n > old) @memset(list.items[old..], std.mem.zeroes(T));
}

// -----------------------------------------------------------------------------
// mtmd-helper.cpp

/// Ports `mtmd_helper_log_set` (mtmd-helper.cpp:58 @b1b770d90): the
/// helpers' logger and libmtmd's together.
pub fn mtmd_helper_log_set(log_callback_in: llama.impl.llama_c.ggml_log_callback, user_data: ?*anyopaque) callconv(.c) void {
    const log_callback = log_callback_in orelse g_logger.default_callback;
    g_logger.log_callback = log_callback;
    g_logger.log_callback_user_data = user_data;
    mod.mtmd.mtmd_log_set(log_callback, user_data);
}

/// Ports `mtmd_helper_get_n_tokens` (mtmd-helper.cpp:71 @b1b770d90).
pub fn mtmd_helper_get_n_tokens(chunks: *const mod.mtmd.InputChunks) callconv(.c) usize {
    var n_tokens: usize = 0;
    var i: usize = 0;
    while (i < mod.mtmd.mtmd_input_chunks_size(chunks)) : (i += 1) {
        const chunk = mod.mtmd.mtmd_input_chunks_get(chunks, i).?;
        n_tokens +%= mod.mtmd.mtmd_input_chunk_get_n_tokens(chunk);
    }
    return n_tokens;
}

/// Ports `mtmd_helper_get_n_pos` (mtmd-helper.cpp:80 @b1b770d90).
pub fn mtmd_helper_get_n_pos(chunks: *const mod.mtmd.InputChunks) callconv(.c) llama.impl.llama_c.llama_pos {
    var n_pos: llama.impl.llama_c.llama_pos = 0;
    var i: usize = 0;
    while (i < mod.mtmd.mtmd_input_chunks_size(chunks)) : (i += 1) {
        const chunk = mod.mtmd.mtmd_input_chunks_get(chunks, i).?;
        n_pos +%= mod.mtmd.mtmd_input_chunk_get_n_pos(chunk);
    }
    return n_pos;
}

/// Ports `mtmd_helper_image_get_decoder_pos` (mtmd-helper.cpp:89 @b1b770d90).
pub fn mtmd_helper_image_get_decoder_pos(chunks: *const mod.mtmd.ImageTokens, pos_0: llama.impl.llama_c.llama_pos, out_pos: [*]mod.mtmd.DecoderPos) callconv(.c) void {
    const n_tokens = mod.mtmd.mtmd_image_tokens_get_n_tokens(chunks);
    for (0..n_tokens) |i| out_pos[i] = mod.mtmd.mtmd_image_tokens_get_decoder_pos(chunks, pos_0, i);
}

/// Ports `mtmd_helper_decode_image_chunk` (mtmd-helper.cpp:120 @b1b770d90):
/// decodes media embeddings already computed, in batches of `n_batch`.
/// `scope_non_causal` (mtmd-helper.cpp:97 @b1b770d90) is the `defer`.
pub fn mtmd_helper_decode_image_chunk(
    ctx: *mod.mtmd.Context,
    lctx: ?*llama.impl.llama_c.struct_llama_context,
    chunk: *const mod.mtmd.InputChunk,
    encoded_embd: ?[*]f32,
    n_past_in: llama.impl.llama_c.llama_pos,
    seq_id: llama.impl.llama_c.llama_seq_id,
    n_batch: i32,
    new_n_past: *llama.impl.llama_c.llama_pos,
    callback: PostDecodeCallback,
    user_data: ?*anyopaque,
) callconv(.c) i32 {
    var n_past = n_past_in;
    mod.impl.ggmlAssert(n_batch > 0, "mtmd-helper.cpp", 131, "n_batch > 0");
    const chunk_type = mod.mtmd.mtmd_input_chunk_get_type(chunk);
    const name: [*:0]const u8 = if (chunk_type == .image) "image" else "audio";
    if (chunk_type == .text) {
        logErr("failed to decode chunk: input chunk not of image/audio type\n", .{});
        return -1;
    }

    const model = llama.api.context.llama_get_model(lctx);
    const n_mmproj_embd = llama.api.model.llama_model_n_embd_inp(model);
    const n_pos_per_embd: c_int = if (mod.mtmd.mtmd_decode_use_mrope(ctx)) 4 else 1;

    const n_tokens: i32 = @truncate(@as(isize, @bitCast(mod.mtmd.mtmd_input_chunk_get_n_tokens(chunk))));
    var i_batch: i32 = 0;
    const n_img_batches: i32 = @divTrunc(n_tokens + n_batch - 1, n_batch);
    const batch_embd = EmbdBatch.create(encoded_embd, n_tokens, n_pos_per_embd, n_mmproj_embd);
    defer batch_embd.destroy();

    if (mod.mtmd.mtmd_decode_use_mrope(ctx)) {
        if (chunk_type == .image) {
            const image_tokens = mod.mtmd.mtmd_input_chunk_get_tokens_image(chunk) orelse {
                logErr("failed to decode chunk: image tokens are null\n", .{});
                return -1;
            };
            const n_img_tokens = mod.mtmd.mtmd_image_tokens_get_n_tokens(image_tokens);
            const rel_pos = std.heap.c_allocator.alloc(mod.mtmd.DecoderPos, n_img_tokens) catch oom();
            defer std.heap.c_allocator.free(rel_pos);
            @memset(rel_pos, std.mem.zeroes(mod.mtmd.DecoderPos));
            mtmd_helper_image_get_decoder_pos(image_tokens, n_past, rel_pos.ptr);
            batch_embd.setPositionMrope2d(rel_pos, seq_id);
        } else if (chunk_type == .audio) {
            batch_embd.setPositionMrope1d(n_past, seq_id);
        } else {
            mod.impl.abort("mtmd-helper.cpp", 162, "invalid chunk type for M-RoPE");
        }
    } else {
        batch_embd.setPositionNormal(n_past, seq_id);
    }

    const use_non_causal = mod.mtmd.mtmd_decode_use_non_causal(ctx, chunk);
    // TODO @ngxson : need to make sure only one image is processed at a time, and n_ubatch must be enough to hold the image
    if (use_non_causal) llama.api.context.llama_set_causal_attn(lctx, false);
    defer if (use_non_causal) llama.api.context.llama_set_causal_attn(lctx, true);

    while (i_batch < n_img_batches) { // split into batches
        const pos_offset = i_batch * n_batch;
        const n_tokens_batch = @min(n_batch, n_tokens - pos_offset);

        logInf("decoding %s batch %d/%d, n_tokens_batch = %d\n", .{ name, i_batch + 1, n_img_batches, n_tokens_batch });

        const t1 = llama.impl.llama_c.ggml_time_ms();
        var ret = llama.api.context.llama_process(lctx, llama.impl.llama_c.LLAMA_PROCESS_TYPE_DECODE, batch_embd.render(lctx, pos_offset, n_tokens_batch));
        if (ret != 0) {
            logErr("failed to decode %s\n", .{name});
            return ret;
        }

        if (callback) |cb| {
            const view = batch_embd.getView(pos_offset, n_tokens_batch);
            ret = cb(&view, user_data);
            if (ret != 0) {
                logErr("post-decode callback failed\n", .{});
                return ret;
            }
        }

        logInf("%s decoded (batch %d/%d) in %lld ms\n", .{ name, i_batch + 1, n_img_batches, @as(c_longlong, llama.impl.llama_c.ggml_time_ms() - t1) });

        i_batch += 1;
    }

    n_past +%= mod.mtmd.mtmd_input_chunk_get_n_pos(chunk);
    new_n_past.* = n_past;

    return 0;
}

/// Ports `mtmd_helper_eval_chunk_single` (mtmd-helper.cpp:204 @b1b770d90):
/// a text chunk goes through a `llama_batch_ext`.
pub fn mtmd_helper_eval_chunk_single(
    ctx: *mod.mtmd.Context,
    lctx: ?*llama.impl.llama_c.struct_llama_context,
    chunk: *const mod.mtmd.InputChunk,
    n_past_in: llama.impl.llama_c.llama_pos,
    seq_id: llama.impl.llama_c.llama_seq_id,
    n_batch: i32,
    logits_last: bool,
    new_n_past: *llama.impl.llama_c.llama_pos,
) callconv(.c) i32 {
    var n_past = n_past_in;
    mod.impl.ggmlAssert(n_batch > 0, "mtmd-helper.cpp", 212, "n_batch > 0");
    var ret: i32 = undefined;
    const chunk_type = mod.mtmd.mtmd_input_chunk_get_type(chunk);

    if (chunk_type == .text) {
        var n_tokens: usize = 0;
        const tokens = mod.mtmd.mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        // LOG_INF("decoding text chunk, n_tokens = %zu\n", n_tokens);
        const text_batch = llama.api.batch.llama_batch_ext_init(lctx);
        defer llama.api.batch.llama_batch_ext_free(text_batch);
        var i: usize = 0;
        while (i < n_tokens) { // split into batches
            llama.api.batch.llama_batch_ext_clear(text_batch);
            var n_added: i32 = 0;
            var idx: i32 = -1;
            while (i < n_tokens and n_added < n_batch) : (i += 1) {
                idx = llama.api.batch.llama_batch_ext_add_token(text_batch, seq_id, tokens.?[i]);
                mod.impl.ggmlAssert(idx >= 0, "mtmd-helper.cpp", 228, "idx >= 0");
                const pos: llama.impl.llama_c.llama_pos = n_past;
                n_past +%= 1;
                _ = llama.api.batch.llama_batch_ext_set_pos(text_batch, idx, &pos);
                n_added += 1;
            }
            const is_last_token = (i == n_tokens);
            if (logits_last and is_last_token) {
                _ = llama.api.batch.llama_batch_ext_set_output_logits(text_batch, idx, true);
            }
            ret = llama.api.context.llama_process(lctx, llama.impl.llama_c.LLAMA_PROCESS_TYPE_DECODE, text_batch);
            if (ret != 0) {
                logErr("failed to decode text\n", .{});
                return ret;
            }
            new_n_past.* +%= n_added;
        }
    } else if (chunk_type == .image or chunk_type == .audio) {
        const name: [*:0]const u8 = if (chunk_type == .image) "image" else "audio";
        const t0 = llama.impl.llama_c.ggml_time_ms();

        logInf("encoding %s slice...\n", .{name});

        ret = mod.mtmd.mtmd_encode_chunk(ctx, chunk);
        if (ret != 0) {
            logErr("failed to encode %s slice\n", .{name});
            return ret;
        }

        logInf("%s slice encoded in %lld ms\n", .{ name, @as(c_longlong, llama.impl.llama_c.ggml_time_ms() - t0) });

        const embd = mod.mtmd.mtmd_get_output_embd(ctx);
        ret = mtmd_helper_decode_image_chunk(ctx, lctx, chunk, embd, n_past, seq_id, n_batch, new_n_past, null, null);
        if (ret != 0) {
            logErr("failed to decode %s\n", .{name});
            return ret;
        }
    } else {
        mod.impl.abort("mtmd-helper.cpp", 266, "chunk type not supported");
    }

    return 0;
}

/// Ports `mtmd_helper_eval_chunks` (mtmd-helper.cpp:272 @b1b770d90).
pub fn mtmd_helper_eval_chunks(
    ctx: *mod.mtmd.Context,
    lctx: ?*llama.impl.llama_c.struct_llama_context,
    chunks: *const mod.mtmd.InputChunks,
    n_past_in: llama.impl.llama_c.llama_pos,
    seq_id: llama.impl.llama_c.llama_seq_id,
    n_batch: i32,
    logits_last: bool,
    new_n_past: *llama.impl.llama_c.llama_pos,
) callconv(.c) i32 {
    var n_past = n_past_in;
    const n_chunks = mod.mtmd.mtmd_input_chunks_size(chunks);
    if (n_chunks == 0) {
        logWrn("no chunks to eval\n", .{});
        return 0;
    }

    for (0..n_chunks) |i| {
        const chunk_logits_last = (i == n_chunks - 1) and logits_last;
        const chunk = mod.mtmd.mtmd_input_chunks_get(chunks, i).?;

        const res = mtmd_helper_eval_chunk_single(ctx, lctx, chunk, n_past, seq_id, n_batch, chunk_logits_last, &n_past);
        if (res != 0) {
            logErr("failed to eval chunk %zu\n", .{i});
            return res;
        }
        new_n_past.* = n_past;
    }

    return 0;
}

/// Ports `is_audio_file` (mtmd-helper.cpp:303 @b1b770d90), a member of `audio_helpers`.
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

/// `decode_audio_from_buf` (mtmd-helper.cpp:322 @b1b770d90), a member of `audio_helpers`,
/// which is miniaudio's decoder, in `vendor/`.
fn decodeAudioFromBuf(buf: []const u8, target_sample_rate: c_int, out: *std.ArrayList(f32)) bool {
    return mod.vendor.miniaudio.decodeAudioFromBuf(std.heap.c_allocator, buf, target_sample_rate, out);
}

/// Ports `mtmd_helper_resampler_init` (mtmd-helper.cpp:369 @b1b770d90).
pub fn mtmd_helper_resampler_init(rate_in: i32, rate_out: i32) callconv(.c) ?*mod.vendor.miniaudio.HelperResampler {
    if (rate_in <= 0 or rate_out <= 0) {
        logErr("%s: invalid sample rate %d -> %d\n", .{ "mtmd_helper_resampler_init", rate_in, rate_out });
        return null;
    }
    return mod.vendor.miniaudio.HelperResampler.create(std.heap.c_allocator, @intCast(rate_in), @intCast(rate_out)) catch oom() orelse {
        // unreachable with both rates positive, as `ma_resampler_init` is
        logErr("%s: failed to initialize the resampler %d -> %d\n", .{ "mtmd_helper_resampler_init", rate_in, rate_out });
        return null;
    };
}

/// Ports `mtmd_helper_resampler_free` (mtmd-helper.cpp:390 @b1b770d90).
pub fn mtmd_helper_resampler_free(ctx: ?*mod.vendor.miniaudio.HelperResampler) callconv(.c) void {
    const r = ctx orelse return;
    r.destroy(std.heap.c_allocator);
}

/// Ports `mtmd_helper_resampler_process` (mtmd-helper.cpp:400 @b1b770d90):
/// `*out` stays valid until the next call; returns the samples written.
pub fn mtmd_helper_resampler_process(ctx: *mod.vendor.miniaudio.HelperResampler, in: ?[*]const f32, n_in: usize, out: *?[*]const f32) callconv(.c) i64 {
    return ctx.process(std.heap.c_allocator, in, n_in, out) catch oom();
}

/// SHA-256 of `buf` in lowercase hex: `hash_sha256_hex` (vendor/hash/hash.cpp:19
/// @b1b770d90), the standard digest.
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

/// Ports `mtmd_helper_bitmap_init_from_buf` (mtmd-helper.cpp:442 @b1b770d90):
/// audio by its magic bytes, else an image stb_image can read. The ID is
/// the buffer's SHA-256 unless `placeholder`. `opt` is only read by the
/// video paths, which this build does not compile. `ctx` may be null, as
/// the C's pointer may: only audio reads it, and a buffer that is neither
/// audio nor an image fails without it (the server's decision API passes
/// none for a vocab-only model).
pub fn mtmd_helper_bitmap_init_from_buf(ctx: ?*const mod.mtmd.Context, buf_ptr: ?[*]const u8, len: usize, placeholder: bool, opt: InitOpt) callconv(.c) BitmapWrapper {
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
        defer pcmf32.deinit(std.heap.c_allocator);
        const sample_rate = mod.mtmd.mtmd_get_audio_sample_rate(ctx.?);
        if (sample_rate < 0) {
            logErr("This model does not support audio input\n", .{});
            return .{ .bitmap = null, .video_ctx = null };
        }
        if (!decodeAudioFromBuf(buf, sample_rate, &pcmf32)) {
            logErr("Unable to read WAV audio file from buffer\n", .{});
            return .{ .bitmap = null, .video_ctx = null };
        }
        const result = mod.mtmd.mtmd_bitmap_init_from_audio(pcmf32.items.len, if (placeholder) null else pcmf32.items.ptr);
        mod.mtmd.mtmd_bitmap_set_id(result, id);
        return .{ .bitmap = result, .video_ctx = null };
    }

    // otherwise, we assume it's an image
    switch (mod.vendor.stb_image.loadFromMemory(std.heap.c_allocator, buf, 3)) {
        .ok => |img| {
            defer std.heap.c_allocator.free(img.pixels);
            const result = mod.mtmd.mtmd_bitmap_init(@bitCast(img.x), @bitCast(img.y), if (placeholder) null else img.pixels.ptr);
            mod.mtmd.mtmd_bitmap_set_id(result, id);
            return .{ .bitmap = result, .video_ctx = null };
        },
        // otherwise, fallthrough to video decoding (if supported)
        .failed => {},
    }

    logErr("%s: failed to decode buffer as either image or audio (video support not compiled in)\n", .{"mtmd_helper_bitmap_init_from_buf"});
    return .{ .bitmap = null, .video_ctx = null };
}

/// Ports `mtmd_helper_bitmap_init_from_file` (mtmd-helper.cpp:530 @b1b770d90).
pub fn mtmd_helper_bitmap_init_from_file(ctx: *const mod.mtmd.Context, fname: [*:0]const u8, placeholder: bool, opt: InitOpt) callconv(.c) BitmapWrapper {
    const f = openFile(fname) orelse return .{ .bitmap = null, .video_ctx = null };

    _ = stdlib.libc.fseek(f, 0, seek_end);
    const file_size = stdlib.libc.ftell(f);
    _ = stdlib.libc.fseek(f, 0, seek_set);
    if (file_size < 0) {
        logErr("Failed to get file size of %s\n", .{fname});
        _ = stdlib.libc.fclose(f);
        return .{ .bitmap = null, .video_ctx = null };
    }
    const buf = std.heap.c_allocator.alloc(u8, @intCast(file_size)) catch oom();
    defer std.heap.c_allocator.free(buf);
    @memset(buf, 0);

    const n_read = stdlib.libc.fread(buf.ptr, 1, buf.len, f);
    _ = stdlib.libc.fclose(f);
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
        const wfname = std.heap.c_allocator.alloc(u16, @intCast(wlen)) catch oom();
        defer std.heap.c_allocator.free(wfname);
        @memset(wfname, 0);
        wlen = win32.MultiByteToWideChar(cp_utf8, 0, fname, -1, wfname.ptr, wlen);
        if (wlen == 0) {
            logErr("Unable to convert filename to UTF-16: %s\n", .{fname});
            return null;
        }
        const f = stdlib.libc._wfopen(@ptrCast(wfname.ptr), std.unicode.utf8ToUtf16LeStringLiteral("rb"));
        if (f == null) logErr("Unable to open file %s: %s\n", .{ fname, stdlib.errno.strerror(std.c._errno().*) });
        return f;
    }
    const f = stdlib.libc.fopen(fname, "rb");
    if (f == null) logErr("Unable to open file %s: %s\n", .{ fname, stdlib.errno.strerror(std.c._errno().*) });
    return f;
}

/// Ports `mtmd_helper_video_init_params_default` (mtmd-helper.cpp:1133
/// @b1b770d90).
pub fn mtmd_helper_video_init_params_default() callconv(.c) VideoInitParams {
    return .{ .fps_target = 4.0, .ffmpeg_bin_dir = null, .timestamp_interval_ms = 5000, .no_soundtrack = false };
}

/// Ports `mtmd_helper_init_opt_default` (mtmd-helper.cpp:1142 @b1b770d90).
pub fn mtmd_helper_init_opt_default() callconv(.c) InitOpt {
    return .{ .video_params = mtmd_helper_video_init_params_default() };
}

/// Ports `mtmd_helper_video_init` (mtmd-helper.cpp:1200 @b1b770d90), its
/// `#else` arm.
pub fn mtmd_helper_video_init(mctx: ?*const mod.mtmd.Context, path: ?[*:0]const u8, params: VideoInitParams) callconv(.c) ?*anyopaque {
    _ = mctx;
    _ = path;
    _ = params;
    logErr("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", .{"mtmd_helper_video_init"});
    return null;
}

/// Ports `mtmd_helper_video_init_from_buf` (mtmd-helper.cpp:1239 @b1b770d90),
/// its `#else` arm.
pub fn mtmd_helper_video_init_from_buf(mctx: ?*const mod.mtmd.Context, buf: ?[*]const u8, len: usize, params: VideoInitParams) callconv(.c) ?*anyopaque {
    _ = mctx;
    _ = buf;
    _ = len;
    _ = params;
    logErr("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", .{"mtmd_helper_video_init_from_buf"});
    return null;
}

/// Ports `mtmd_helper_video_free` (mtmd-helper.cpp:1279 @b1b770d90), its
/// `#else` arm.
pub fn mtmd_helper_video_free(ctx: ?*anyopaque) callconv(.c) void {
    _ = ctx;
    logErr("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", .{"mtmd_helper_video_free"});
}

/// Ports `mtmd_helper_video_get_info` (mtmd-helper.cpp:1290 @b1b770d90), its
/// `#else` arm.
pub fn mtmd_helper_video_get_info(ctx: ?*const anyopaque) callconv(.c) VideoInfo {
    _ = ctx;
    mod.impl.ggmlAssert(false, "mtmd-helper.cpp", 1295, "false && \"video is not supported in this build (MTMD_VIDEO is set to OFF)\"");
    unreachable;
}

/// Ports `mtmd_helper_video_read_next` (mtmd-helper.cpp:1299 @b1b770d90), its
/// `#else` arm.
pub fn mtmd_helper_video_read_next(ctx: ?*anyopaque, out_bitmap: ?*?*mod.mtmd.Bitmap, out_text: ?*?[*:0]u8) callconv(.c) i32 {
    _ = ctx;
    _ = out_bitmap;
    _ = out_text;
    mod.impl.ggmlAssert(false, "mtmd-helper.cpp", 1308, "false && \"video is not supported in this build (MTMD_VIDEO is set to OFF)\"");
    unreachable;
}

/// Ports `mtmd_helper_get_capabilities` (mtmd-helper.cpp:1312 @b1b770d90): `mctx` can be
/// null (text-only model). Video is not compiled in.
pub fn mtmd_helper_get_capabilities(model: ?*const llama.impl.llama_c.struct_llama_model, mctx: ?*const mod.mtmd.Context) callconv(.c) Cap {
    var cap = std.mem.zeroes(Cap);
    cap.chat = true;
    cap.text_gen = true;
    const m = mctx orelse return cap;

    cap.vision = mod.mtmd.mtmd_support_vision(m);
    cap.audio = mod.mtmd.mtmd_support_audio(m);
    cap.audio_stream = mod.mtmd.mtmd_support_audio_stream(m);
    const gen_audio_type = mod.mtmd.mtmd_gen_audio_get_info(m).type;
    cap.audio_gen = gen_audio_type != .none;
    cap.audio_gen_spk_ref = gen_audio_type == .qwen3tts or gen_audio_type == .pockettts;

    // a tts model without a chat template only outputs audio, it cannot be used for chat
    // neither can a realtime transcription model (e.g. Voxtral Realtime): its text follows the audio by a token delay
    cap.text_gen = !cap.audio_gen or llama.api.model.llama_model_chat_template(model, null) != null;
    cap.chat = cap.text_gen and llama.api.model.llama_model_n_tokens_delay(model) == 0;

    return cap;
}

/// Ports `mtmd_helper_model_can_chat` (mtmd-helper.cpp:1339 @b1b770d90).
pub fn mtmd_helper_model_can_chat(lctx: ?*const llama.impl.llama_c.struct_llama_context, mctx: ?*const mod.mtmd.Context) callconv(.c) bool {
    return mtmd_helper_get_capabilities(llama.api.context.llama_get_model(lctx), mctx).chat;
}

/// Ports `mtmd_helper_support_video` (mtmd-helper.cpp:1343 @b1b770d90): video
/// is not compiled in.
pub fn mtmd_helper_support_video(ctx: ?*const mod.mtmd.Context) callconv(.c) bool {
    _ = ctx;
    return false;
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
