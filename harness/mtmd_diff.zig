//! `make mtmd-diff`: libmtmd's port against upstream's, through `mtmd.h`
//! and `mtmd-helper.h`, on every mmproj fixture.
//!
//! **Not a port.** This is test harness. It links `src/mtmd/harness_root.zig`
//! (which exports nothing) beside `libmtmd-ref.a` and calls one side per
//! process -- `ref` or `port` -- through a table of C function pointers with
//! the same types, so both sides run the same script and print the same
//! transcript, and `scripts/mtmd-diff` diffs the two. A side links its own
//! libllama too: the reference libmtmd calls `libllama-ref.a`'s `llama_*`,
//! the port calls `src/llama/`'s through its `api`, and the harness loads
//! the text model through the side's own.
//!
//! The transcript is everything that crosses the C boundary: every log line
//! of libmtmd, its helpers, libllama and ggml; with `nodes`, every graph
//! node the encoder computes (`cb_eval`: name, op, type, shape and a hash of
//! its data); `mtmd_init_from_file`'s result and every capability getter;
//! the chunks `mtmd_tokenize` makes -- type, tokens, counts, ids, every
//! decoder position, the serialized bytes; every encoded embedding on bits,
//! through `mtmd_encode_chunk` and the batch API; the logits after
//! `mtmd_helper_eval_chunks`; each media chunk through
//! `mtmd_helper_decode_image_chunk` with its post-decode callback's views;
//! `mtmd_get_memory_usage`'s result (through `harness/mtmd_mem_ref.cpp` on
//! the reference side); and for a TTS fixture the generation helper's
//! hidden states and output WAV. Since v0.8.0 also `mtmd_gen_audio_process`
//! through a state, the helpers in streaming mode and a continued sequence,
//! `mtmd_audio_stream` fed PCM in pieces with every step's rows hashed, and a
//! lazy video through the tokenizer with its soundtrack.
//!
//! Usage: mtmd_diff <ref|port> <cpu|gpu> <nodes|fused> <tmp-dir>
//!                  <projector> <mmproj> <text-model> <modality> <inputs>

const std = @import("std");
const harness_root = @import("harness_root");
const mport = @import("mtmd_port");
const c = @import("llama_c");
const mtmd_c = @import("mtmd_c");

const M = mport.mtmd;
const H = mport.helper;
const G = mport.helper_gen;

// -----------------------------------------------------------------------------
// struct layout: the port's hand-declared mtmd.h structs against the C

fn checkLayout(comptime Port: type, comptime C: type) void {
    if (@sizeOf(Port) != @sizeOf(C)) @compileError("size of " ++ @typeName(Port));
    if (@alignOf(Port) != @alignOf(C)) @compileError("align of " ++ @typeName(Port));
    const pi = @typeInfo(Port).@"struct";
    const ci = @typeInfo(C).@"struct";
    if (pi.field_names.len != ci.field_names.len) @compileError("field count of " ++ @typeName(Port));
    for (pi.field_names, ci.field_names, pi.field_types, ci.field_types) |p, q, pt, qt| {
        if (!std.mem.eql(u8, p, q)) @compileError("field " ++ p ++ " vs " ++ q);
        if (@offsetOf(Port, p) != @offsetOf(C, q)) @compileError("offset of " ++ p);
        if (@sizeOf(pt) != @sizeOf(qt)) @compileError("size of field " ++ p);
    }
}

comptime {
    checkLayout(M.InputText, mtmd_c.struct_mtmd_input_text);
    checkLayout(M.InputPart, mtmd_c.struct_mtmd_input_part);
    checkLayout(M.ContextParams, mtmd_c.struct_mtmd_context_params);
    checkLayout(M.DecoderPos, mtmd_c.struct_mtmd_decoder_pos);
    checkLayout(M.Caps, mtmd_c.struct_mtmd_caps);
    checkLayout(M.GenAudioInfo, mtmd_c.struct_mtmd_gen_audio_info);
    checkLayout(M.GenInp, mtmd_c.struct_mtmd_gen_inp);
    checkLayout(M.GenOut, mtmd_c.struct_mtmd_gen_out);
    checkLayout(H.BitmapWrapper, mtmd_c.struct_mtmd_helper_bitmap_wrapper);
    checkLayout(H.VideoInfo, mtmd_c.struct_mtmd_helper_video_info);
    checkLayout(H.VideoInitParams, mtmd_c.struct_mtmd_helper_video_init_params);
    checkLayout(H.InitOpt, mtmd_c.struct_mtmd_helper_init_opt);
    checkLayout(G.Inp, mtmd_c.struct_mtmd_helper_gen_audio_inp);
    checkLayout(H.EmbdBatchView, mtmd_c.struct_mtmd_helper_embd_batch);
    checkLayout(H.Cap, mtmd_c.struct_mtmd_helper_cap);
    checkLayout(M.GenStateParams, mtmd_c.struct_mtmd_gen_state_params);
    checkLayout(M.AudioStreamParams, mtmd_c.struct_mtmd_audio_stream_params);
    checkLayout(M.VideoConfig, mtmd_c.struct_mtmd_video_config);
}

// -----------------------------------------------------------------------------
// the two sides

const P = *anyopaque;
const OptP = ?*anyopaque;

/// Every entry point the script calls, typed as the C declares it. A side
/// is this table filled from its library.
const Api = struct {
    mtmd_context_params_default: *const fn () callconv(.c) M.ContextParams,
    mtmd_init_from_file: *const fn ([*:0]const u8, OptP, M.ContextParams) callconv(.c) OptP,
    mtmd_free: *const fn (OptP) callconv(.c) void,
    mtmd_support_vision: *const fn (P) callconv(.c) bool,
    mtmd_support_audio: *const fn (P) callconv(.c) bool,
    mtmd_get_audio_sample_rate: *const fn (P) callconv(.c) c_int,
    mtmd_decode_use_mrope: *const fn (P) callconv(.c) bool,
    mtmd_decode_use_non_causal: *const fn (P, OptP) callconv(.c) bool,
    mtmd_get_marker: *const fn (P) callconv(.c) [*:0]const u8,
    mtmd_gen_audio_get_info: *const fn (P) callconv(.c) M.GenAudioInfo,
    mtmd_gen_inp_default: *const fn (P) callconv(.c) M.GenInp,
    mtmd_bitmap_init: *const fn (u32, u32, ?[*]const u8) callconv(.c) P,
    mtmd_bitmap_init_from_audio: *const fn (usize, ?[*]const f32) callconv(.c) P,
    mtmd_bitmap_free: *const fn (OptP) callconv(.c) void,
    mtmd_bitmap_set_id: *const fn (P, ?[*:0]const u8) callconv(.c) void,
    mtmd_bitmap_set_mergeable: *const fn (P, bool) callconv(.c) void,
    mtmd_bitmap_get_id: *const fn (P) callconv(.c) [*:0]const u8,
    mtmd_bitmap_get_nx: *const fn (P) callconv(.c) u32,
    mtmd_bitmap_get_ny: *const fn (P) callconv(.c) u32,
    mtmd_bitmap_get_n_bytes: *const fn (P) callconv(.c) usize,
    mtmd_bitmap_is_audio: *const fn (P) callconv(.c) bool,
    mtmd_input_chunks_init: *const fn () callconv(.c) P,
    mtmd_input_chunks_free: *const fn (OptP) callconv(.c) void,
    mtmd_input_chunks_size: *const fn (P) callconv(.c) usize,
    mtmd_input_chunks_get: *const fn (P, usize) callconv(.c) OptP,
    mtmd_tokenize: *const fn (P, P, *const M.InputText, ?[*]const P, usize) callconv(.c) i32,
    mtmd_tokenize_from_parts: *const fn (P, P, ?[*]const *const M.InputPart, usize, bool) callconv(.c) i32,
    mtmd_input_chunk_get_type: *const fn (P) callconv(.c) c_int,
    mtmd_input_chunk_get_tokens_text: *const fn (P, *usize) callconv(.c) ?[*]const c.llama_token,
    mtmd_input_chunk_get_tokens_image: *const fn (P) callconv(.c) OptP,
    mtmd_input_chunk_get_n_tokens: *const fn (P) callconv(.c) usize,
    mtmd_input_chunk_get_n_pos: *const fn (P) callconv(.c) c.llama_pos,
    mtmd_input_chunk_get_id: *const fn (P) callconv(.c) ?[*:0]const u8,
    mtmd_input_chunk_copy: *const fn (P) callconv(.c) P,
    mtmd_input_chunk_free: *const fn (OptP) callconv(.c) void,
    mtmd_input_chunk_get_placeholder: *const fn (P) callconv(.c) OptP,
    mtmd_input_chunk_save: *const fn (P, ?[*]u8, usize, ?*usize) callconv(.c) i32,
    mtmd_input_chunk_load: *const fn (?[*]const u8, usize) callconv(.c) OptP,
    mtmd_image_tokens_get_n_tokens: *const fn (P) callconv(.c) usize,
    mtmd_image_tokens_get_nx: *const fn (P) callconv(.c) usize,
    mtmd_image_tokens_get_ny: *const fn (P) callconv(.c) usize,
    mtmd_image_tokens_get_id: *const fn (P) callconv(.c) [*:0]const u8,
    mtmd_image_tokens_get_n_pos: *const fn (P) callconv(.c) c.llama_pos,
    mtmd_image_tokens_get_decoder_pos: *const fn (P, c.llama_pos, usize) callconv(.c) M.DecoderPos,
    mtmd_encode_chunk: *const fn (P, P) callconv(.c) i32,
    mtmd_get_output_embd: *const fn (P) callconv(.c) ?[*]f32,
    mtmd_batch_init: *const fn (P) callconv(.c) P,
    mtmd_batch_free: *const fn (OptP) callconv(.c) void,
    mtmd_batch_add_chunk: *const fn (P, P) callconv(.c) i32,
    mtmd_batch_encode: *const fn (P) callconv(.c) i32,
    mtmd_batch_get_output_embd: *const fn (P, OptP) callconv(.c) ?[*]f32,
    mtmd_get_cap_from_file: *const fn ([*:0]const u8) callconv(.c) M.Caps,
    mtmd_test_create_input_chunks: *const fn () callconv(.c) OptP,
    mtmd_helper_log_set: *const fn (c.ggml_log_callback, OptP) callconv(.c) void,
    mtmd_helper_init_opt_default: *const fn () callconv(.c) H.InitOpt,
    mtmd_helper_bitmap_init_from_file: *const fn (P, [*:0]const u8, bool, H.InitOpt) callconv(.c) H.BitmapWrapper,
    mtmd_helper_bitmap_init_from_buf: *const fn (P, ?[*]const u8, usize, bool, H.InitOpt) callconv(.c) H.BitmapWrapper,
    mtmd_helper_get_n_tokens: *const fn (P) callconv(.c) usize,
    mtmd_helper_get_n_pos: *const fn (P) callconv(.c) c.llama_pos,
    mtmd_helper_eval_chunks: *const fn (P, OptP, P, c.llama_pos, c.llama_seq_id, i32, bool, *c.llama_pos) callconv(.c) i32,
    mtmd_helper_decode_image_chunk: *const fn (P, OptP, P, ?[*]f32, c.llama_pos, c.llama_seq_id, i32, *c.llama_pos, H.PostDecodeCallback, OptP) callconv(.c) i32,
    mtmd_helper_model_can_chat: *const fn (OptP, OptP) callconv(.c) bool,
    mtmd_helper_gen_audio_init: *const fn (OptP, P) callconv(.c) P,
    mtmd_helper_gen_audio_free: *const fn (OptP) callconv(.c) void,
    mtmd_helper_gen_audio_set_input: *const fn (P, *const G.Inp) callconv(.c) i32,
    mtmd_helper_gen_audio_step_prompt: *const fn (P, i32) callconv(.c) i32,
    mtmd_helper_gen_audio_step_gen: *const fn (P, c.llama_token, ?[*]const f32, *?[*]const f32, ?*bool) callconv(.c) i32,
    mtmd_helper_gen_audio_get_output: *const fn (P, *i32, *?[*]const u8, *usize, ?*i64) callconv(.c) i32,
    mtmd_helper_gen_audio_flush: *const fn (P) callconv(.c) i32,
    mtmd_helper_gen_audio_inp_default: *const fn () callconv(.c) G.Inp,
    mtmd_helper_gen_audio_next_is_text: *const fn (P) callconv(.c) bool,
    mtmd_helper_gen_audio_get_logit_bias: *const fn (P, *usize) callconv(.c) ?[*]const c.llama_logit_bias,
    mtmd_helper_audio_stream_get_logit_bias: *const fn (P, ?*const c.llama_model, ?[*]c.llama_logit_bias, i32) callconv(.c) i32,
    mtmd_helper_get_capabilities: *const fn (?*const c.llama_model, OptP) callconv(.c) H.Cap,
    mtmd_helper_support_video: *const fn (OptP) callconv(.c) bool,
    mtmd_gen_state_params_default: *const fn () callconv(.c) M.GenStateParams,
    mtmd_gen_state_init: *const fn (P, M.GenStateParams) callconv(.c) OptP,
    mtmd_gen_state_free: *const fn (OptP) callconv(.c) void,
    mtmd_gen_state_reset: *const fn (P) callconv(.c) void,
    mtmd_gen_audio_process: *const fn (P, OptP, *const M.GenInp, *M.GenOut) callconv(.c) i32,
    mtmd_support_audio_stream: *const fn (P) callconv(.c) bool,
    mtmd_audio_stream_params_default: *const fn () callconv(.c) M.AudioStreamParams,
    mtmd_audio_stream_init: *const fn (P, M.AudioStreamParams) callconv(.c) OptP,
    mtmd_audio_stream_free: *const fn (OptP) callconv(.c) void,
    mtmd_audio_stream_push: *const fn (P, ?[*]const f32, usize) callconv(.c) i32,
    mtmd_audio_stream_flush: *const fn (P, bool) callconv(.c) i32,
    mtmd_audio_stream_step: *const fn (P, c.llama_token) callconv(.c) i32,
    mtmd_audio_stream_get_output_embd: *const fn (P) callconv(.c) ?[*]const f32,
    mtmd_audio_stream_is_seq_start: *const fn (P) callconv(.c) bool,
    mtmd_audio_stream_get_n_rows: *const fn (P) callconv(.c) i64,
    mtmd_audio_stream_get_n_rows_flushed: *const fn (P) callconv(.c) i64,
    mtmd_get_video_config: *const fn (P, i32, i32, i32, f32) callconv(.c) M.VideoConfig,
    mtmd_bitmap_set_is_video: *const fn (P, bool) callconv(.c) void,
    mtmd_bitmap_init_lazy: *const fn (OptP, ?[*:0]const u8, OptP, ?*const anyopaque) callconv(.c) P,

    llama_log_set: *const fn (c.ggml_log_callback, OptP) callconv(.c) void,
    llama_model_default_params: *const fn () callconv(.c) c.llama_model_params,
    llama_model_load_from_file: *const fn ([*:0]const u8, c.llama_model_params) callconv(.c) ?*c.llama_model,
    llama_model_free: *const fn (?*c.llama_model) callconv(.c) void,
    llama_context_default_params: *const fn () callconv(.c) c.llama_context_params,
    llama_init_from_model: *const fn (?*c.llama_model, c.llama_context_params) callconv(.c) ?*c.llama_context,
    llama_free: *const fn (?*c.llama_context) callconv(.c) void,
    llama_get_logits_ith: *const fn (?*c.llama_context, i32) callconv(.c) ?[*]f32,
    llama_get_embeddings_ith: *const fn (?*c.llama_context, i32) callconv(.c) ?[*]f32,
    llama_get_memory: *const fn (?*const c.llama_context) callconv(.c) c.llama_memory_t,
    llama_memory_clear: *const fn (c.llama_memory_t, bool) callconv(.c) void,
    llama_model_n_embd: *const fn (?*const c.llama_model) callconv(.c) i32,
    llama_model_n_embd_inp: *const fn (?*const c.llama_model) callconv(.c) i32,
    llama_model_get_vocab: *const fn (?*const c.llama_model) callconv(.c) ?*const c.llama_vocab,
    llama_vocab_n_tokens: *const fn (?*const c.llama_vocab) callconv(.c) i32,
    llama_vocab_get_text: *const fn (?*const c.llama_vocab, c.llama_token) callconv(.c) [*c]const u8,
};

/// The port's definition of `name`: libmtmd's three files, then libllama's
/// `api/`.
fn portFn(comptime name: []const u8) *const anyopaque {
    inline for (.{ M, H, G }) |f| {
        if (@hasDecl(f, name)) return @ptrCast(&@field(f, name));
    }
    const api = harness_root.api;
    inline for (@typeInfo(api).@"struct".decl_names) |file_name| {
        const file = @field(api, file_name);
        if (@TypeOf(file) == type and @hasDecl(file, name)) return @ptrCast(&@field(file, name));
    }
    @compileError("the port defines no " ++ name);
}

fn makeApi(comptime port: bool) Api {
    var a: Api = undefined;
    const info = @typeInfo(Api).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(a, name) = if (port) @ptrCast(@alignCast(portFn(name))) else @extern(T, .{ .name = name });
    }
    return a;
}

const ref_api = makeApi(false);
const port_api = makeApi(true);

// -----------------------------------------------------------------------------
// the transcript

var out_buf: [1 << 16]u8 = undefined;
var out_file: std.Io.File = undefined;
var out_writer: std.Io.File.Writer = undefined;
var g_io: std.Io = undefined;

fn w() *std.Io.Writer {
    return &out_writer.interface;
}

fn print(comptime fmt: []const u8, args: anytype) void {
    w().print(fmt, args) catch {};
}

fn flush() void {
    w().flush() catch {};
}

fn logCb(level: c.enum_ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    print("[log {d}] {s}", .{ level, std.mem.span(@as([*:0]const u8, @ptrCast(text))) });
    flush();
}

fn hashBytes(b: []const u8) u64 {
    return std.hash.Wyhash.hash(0, b);
}

fn hashF32(p: ?[*]const f32, n: usize) u64 {
    if (n == 0) return 0;
    const q = p orelse return 1;
    return hashBytes(std.mem.sliceAsBytes(q[0..n]));
}

var node_mode = false;

/// Every node the encoder computes: what it is and a hash of what it holds.
fn cbEval(t: [*c]c.struct_ggml_tensor, ask: bool, _: ?*anyopaque) callconv(.c) bool {
    if (ask) return true;
    const tensor: *c.struct_ggml_tensor = t;
    const nbytes = c.ggml_nbytes(tensor);
    var h: u64 = 0;
    if (nbytes > 0 and tensor.buffer != null) {
        const buf = std.heap.c_allocator.alloc(u8, nbytes) catch return true;
        defer std.heap.c_allocator.free(buf);
        c.ggml_backend_tensor_get(tensor, buf.ptr, 0, nbytes);
        h = hashBytes(buf);
    }
    print("node {s} {s} {s} [{d},{d},{d},{d}] {x:0>16}\n", .{
        std.mem.sliceTo(&tensor.name, 0), std.mem.span(c.ggml_op_desc(tensor)), std.mem.span(c.ggml_type_name(tensor.type)),
        tensor.ne[0],                     tensor.ne[1],                         tensor.ne[2],
        tensor.ne[3],                     h,
    });
    return true;
}

// -----------------------------------------------------------------------------
// inputs

/// xorshift32: the same bytes on both sides, from a seed.
const Rng = struct {
    s: u32,
    fn next(self: *Rng) u32 {
        var x = self.s;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.s = x;
        return x;
    }
};

fn makeImage(a: std.mem.Allocator, nx: u32, ny: u32) ![]u8 {
    const px = try a.alloc(u8, @as(usize, nx) * ny * 3);
    var r: Rng = .{ .s = nx *% 7919 +% ny *% 104729 +% 1 };
    for (px, 0..) |*p, i| {
        // smooth gradients plus noise, so resizes and crops see structure
        const x = (i / 3) % nx;
        const y = (i / 3) / nx;
        const base: u32 = @intCast((x * 255 / @max(nx, 1) + y * 255 / @max(ny, 1)) / 2);
        p.* = @truncate(base +% (r.next() % 48));
    }
    return px;
}

fn makeAudio(a: std.mem.Allocator, n: usize) ![]f32 {
    const s = try a.alloc(f32, n);
    for (s, 0..) |*v, i| {
        const t: f32 = @floatFromInt(i);
        v.* = 0.3 * @sin(0.031 * t) + 0.1 * @sin(0.29 * t) + 0.02 * @sin(1.7 * t);
    }
    return s;
}

/// A 16-bit stereo WAV at 22050 Hz, so loading it mixes down and resamples.
fn writeWav(a: std.mem.Allocator, path: []const u8, n_frames: usize) !void {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    const rate: u32 = 22050;
    const data_sz: u32 = @intCast(n_frames * 4);
    try buf.appendSlice(a, "RIFF");
    try buf.appendSlice(a, std.mem.asBytes(&(36 + data_sz)));
    try buf.appendSlice(a, "WAVEfmt ");
    try buf.appendSlice(a, std.mem.asBytes(&@as(u32, 16)));
    try buf.appendSlice(a, std.mem.asBytes(&@as(u16, 1)));
    try buf.appendSlice(a, std.mem.asBytes(&@as(u16, 2)));
    try buf.appendSlice(a, std.mem.asBytes(&rate));
    try buf.appendSlice(a, std.mem.asBytes(&(rate * 4)));
    try buf.appendSlice(a, std.mem.asBytes(&@as(u16, 4)));
    try buf.appendSlice(a, std.mem.asBytes(&@as(u16, 16)));
    try buf.appendSlice(a, "data");
    try buf.appendSlice(a, std.mem.asBytes(&data_sz));
    for (0..n_frames) |i| {
        const t: f32 = @floatFromInt(i);
        const l: i16 = @intFromFloat(9000.0 * @sin(0.05 * t));
        const r: i16 = @intFromFloat(7000.0 * @sin(0.013 * t + 1.0));
        try buf.appendSlice(a, std.mem.asBytes(&l));
        try buf.appendSlice(a, std.mem.asBytes(&r));
    }
    try std.Io.Dir.cwd().writeFile(g_io, .{ .sub_path = path, .data = buf.items });
}

// -----------------------------------------------------------------------------
// the script

const Ctx = struct {
    a: std.mem.Allocator,
    api: *const Api,
    mctx: P,
    model: *c.llama_model,
    lctx: ?*c.llama_context,
};

fn dumpChunks(x: *const Ctx, chunks: P) void {
    const api = x.api;
    const n = api.mtmd_input_chunks_size(chunks);
    print("chunks {d} n_tokens {d} n_pos {d}\n", .{ n, api.mtmd_helper_get_n_tokens(chunks), api.mtmd_helper_get_n_pos(chunks) });
    for (0..n) |i| {
        const ch = api.mtmd_input_chunks_get(chunks, i).?;
        const ty = api.mtmd_input_chunk_get_type(ch);
        const id = api.mtmd_input_chunk_get_id(ch);
        print("  chunk {d} type {d} n_tokens {d} n_pos {d} id {s}\n", .{ i, ty, api.mtmd_input_chunk_get_n_tokens(ch), api.mtmd_input_chunk_get_n_pos(ch), if (id) |s| std.mem.span(s) else "(null)" });
        var nt: usize = 0;
        const toks = api.mtmd_input_chunk_get_tokens_text(ch, &nt);
        if (ty == 0) {
            print("    tokens", .{});
            for (0..nt) |k| print(" {d}", .{toks.?[k]});
            print("\n", .{});
        }
        if (api.mtmd_input_chunk_get_tokens_image(ch)) |it| {
            const ntok = api.mtmd_image_tokens_get_n_tokens(it);
            var h = std.hash.Wyhash.init(0);
            for (0..ntok) |k| {
                const pos = api.mtmd_image_tokens_get_decoder_pos(it, 5, k);
                h.update(std.mem.asBytes(&pos));
            }
            print("    image n_tokens {d} nx {d} ny {d} n_pos {d} id {s} pos {x:0>16}\n", .{ ntok, api.mtmd_image_tokens_get_nx(it), api.mtmd_image_tokens_get_ny(it), api.mtmd_image_tokens_get_n_pos(it), std.mem.span(api.mtmd_image_tokens_get_id(it)), h.final() });
        }
        // the serialized form: everything but the pixels
        var need: usize = 0;
        const rc = api.mtmd_input_chunk_save(ch, null, 0, &need);
        const buf = x.a.alloc(u8, need) catch return;
        defer x.a.free(buf);
        const rc2 = api.mtmd_input_chunk_save(ch, buf.ptr, buf.len, null);
        print("    save {d} {d} len {d} {x:0>16}\n", .{ rc, rc2, need, hashBytes(buf) });
        if (api.mtmd_input_chunk_load(buf.ptr, buf.len)) |l| {
            print("    load n_tokens {d} type {d}\n", .{ api.mtmd_input_chunk_get_n_tokens(l), api.mtmd_input_chunk_get_type(l) });
            api.mtmd_input_chunk_free(l);
        } else print("    load null\n", .{});
        if (buf.len > 8) {
            // a truncated buffer must be refused
            print("    load-truncated {}\n", .{api.mtmd_input_chunk_load(buf.ptr, buf.len - 3) == null});
        }
    }
}

fn encodeChunks(x: *const Ctx, chunks: P) void {
    const api = x.api;
    const n = api.mtmd_input_chunks_size(chunks);
    const n_embd: usize = @intCast(x.api.llama_model_n_embd_inp(x.model));
    for (0..n) |i| {
        const ch = api.mtmd_input_chunks_get(chunks, i).?;
        if (api.mtmd_input_chunk_get_type(ch) == 0) continue;
        const rc = api.mtmd_encode_chunk(x.mctx, ch);
        const nt = api.mtmd_input_chunk_get_n_tokens(ch);
        const e = api.mtmd_get_output_embd(x.mctx);
        print("encode {d} rc {d} embd {x:0>16}", .{ i, rc, if (rc == 0) hashF32(e, nt * n_embd) else 0 });
        if (rc == 0 and nt * n_embd >= 4) {
            print(" [{x:0>8} {x:0>8} {x:0>8} {x:0>8}]", .{ @as(u32, @bitCast(e.?[0])), @as(u32, @bitCast(e.?[1])), @as(u32, @bitCast(e.?[2])), @as(u32, @bitCast(e.?[nt * n_embd - 1])) });
        }
        print("\n", .{});
        flush();

        // a copy encodes the same; a placeholder refuses to
        const cp = api.mtmd_input_chunk_copy(ch);
        const rc_cp = api.mtmd_encode_chunk(x.mctx, cp);
        print("encode-copy {d} rc {d} embd {x:0>16}\n", .{ i, rc_cp, if (rc_cp == 0) hashF32(api.mtmd_get_output_embd(x.mctx), nt * n_embd) else 0 });
        api.mtmd_input_chunk_free(cp);
        if (api.mtmd_input_chunk_get_placeholder(ch)) |ph| {
            print("encode-placeholder {d} rc {d}\n", .{ i, api.mtmd_encode_chunk(x.mctx, ph) });
            api.mtmd_input_chunk_free(ph);
        }
    }

    // the batch API, over every media chunk in turn
    const batch = api.mtmd_batch_init(x.mctx);
    defer api.mtmd_batch_free(batch);
    var added: std.ArrayList(P) = .empty;
    defer added.deinit(x.a);
    for (0..n) |i| {
        const ch = api.mtmd_input_chunks_get(chunks, i).?;
        const rc = api.mtmd_batch_add_chunk(batch, ch);
        print("batch-add {d} rc {d}\n", .{ i, rc });
        if (rc == 0) added.append(x.a, ch) catch {};
    }
    if (added.items.len != 0) {
        const rc = api.mtmd_batch_encode(batch);
        print("batch-encode rc {d}\n", .{rc});
        if (rc == 0) for (added.items, 0..) |ch, k| {
            const e = api.mtmd_batch_get_output_embd(batch, ch);
            print("batch-embd {d} {x:0>16}\n", .{ k, hashF32(e, api.mtmd_input_chunk_get_n_tokens(ch) * n_embd) });
        };
    }
}

fn evalChunks(x: *const Ctx, chunks: P) void {
    const api = x.api;
    const lctx = x.lctx orelse return;
    var n_past: c.llama_pos = 0;
    const rc = api.mtmd_helper_eval_chunks(x.mctx, lctx, chunks, 0, 0, 512, true, &n_past);
    print("eval rc {d} n_past {d}", .{ rc, n_past });
    if (rc == 0) {
        const n_vocab: usize = @intCast(x.api.llama_vocab_n_tokens(x.api.llama_model_get_vocab(x.model)));
        print(" logits {x:0>16}", .{hashF32(x.api.llama_get_logits_ith(lctx, -1), n_vocab)});
    }
    print("\n", .{});
}

/// The post-decode callback (v0.6.0: a `mtmd_helper_embd_batch` view, not
/// a `llama_batch`): every field of each sub-batch's view, its embeddings
/// and its section-major positions by hash.
fn postDecode(v: ?*const H.EmbdBatchView, _: ?*anyopaque) callconv(.c) i32 {
    const b = v.?;
    const n: usize = @intCast(b.n_tokens);
    const ne: usize = @intCast(b.n_embd);
    const np: usize = @intCast(b.n_pos);
    var h: u64 = 0;
    if (b.pos) |p| h = hashBytes(std.mem.sliceAsBytes(p[0 .. n * np]));
    print("  post-decode n_tokens {d} n_embd {d} n_pos {d} seq {d} embd {x:0>16} pos {x:0>16}\n", .{ b.n_tokens, b.n_embd, b.n_pos, b.seq_id, hashF32(b.embd, n * ne), h });
    return 0;
}

/// `mtmd_helper_decode_image_chunk` on each media chunk, from an empty
/// cache at position 3, in sub-batches of 5 so the views start mid-chunk,
/// with the post-decode callback; then the logits of its last token.
fn decodeMediaChunks(x: *const Ctx, chunks: P) void {
    const api = x.api;
    const lctx = x.lctx orelse return;
    const n = api.mtmd_input_chunks_size(chunks);
    for (0..n) |i| {
        const ch = api.mtmd_input_chunks_get(chunks, i).?;
        if (api.mtmd_input_chunk_get_type(ch) == 0) continue;
        if (api.mtmd_encode_chunk(x.mctx, ch) != 0) continue;
        api.llama_memory_clear(api.llama_get_memory(lctx), true);
        var new_n_past: c.llama_pos = -1;
        print("decode-chunk {d}\n", .{i});
        const rc = api.mtmd_helper_decode_image_chunk(x.mctx, lctx, ch, api.mtmd_get_output_embd(x.mctx), 3, 0, 5, &new_n_past, postDecode, null);
        print("decode-chunk {d} rc {d} n_past {d}\n", .{ i, rc, new_n_past });
        flush();
    }
}

/// `mtmd_get_memory_usage` (C++ linkage upstream, so the reference is
/// reached through `harness/mtmd_mem_ref.cpp`): per device, the image token
/// budget and whether the image decodes non-causally (v0.6.0).
extern fn ref_mtmd_get_memory_usage(fname: [*:0]const u8, params: M.ContextParams, devs: [*]c.ggml_backend_dev_t, sizes: [*]usize, cap: c_int, image_max_tokens: *c_int, use_non_causal: *bool) c_int;

fn memoryUsage(port: bool, fname: [*:0]const u8, params: M.ContextParams) void {
    var devs: [16]c.ggml_backend_dev_t = undefined;
    var sizes: [16]usize = undefined;
    var n: usize = 0;
    var imt: c_int = 0;
    var nc = false;
    if (port) {
        var r = M.getMemoryUsage(fname, params);
        defer r.deinit();
        for (r.backend_mem_usage.items.items) |e| {
            if (n < devs.len) {
                devs[n] = e.dev;
                sizes[n] = e.size;
            }
            n += 1;
        }
        imt = r.image_max_tokens;
        nc = r.use_non_causal;
    } else {
        n = @intCast(ref_mtmd_get_memory_usage(fname, params, &devs, &sizes, devs.len, &imt, &nc));
    }
    print("memory-usage n {d} image_max_tokens {d} non_causal {}", .{ n, imt, nc });
    for (0..@min(n, devs.len)) |k| print(" {s}={d}", .{ std.mem.span(c.ggml_backend_dev_name(devs[k])), sizes[k] });
    print("\n", .{});
    flush();
}

fn runMedia(x: *const Ctx, tmp: []const u8, inputs: []const u8, vision: bool, audio: bool) !void {
    const api = x.api;
    var bitmaps: std.ArrayList(P) = .empty;
    defer {
        for (bitmaps.items) |b| api.mtmd_bitmap_free(b);
        bitmaps.deinit(x.a);
    }
    var placeholders: std.ArrayList(P) = .empty;
    defer {
        for (placeholders.items) |b| api.mtmd_bitmap_free(b);
        placeholders.deinit(x.a);
    }
    // v0.7.0: the frames of one video, mergeable, each with its placeholder
    var frames: std.ArrayList(P) = .empty;
    defer {
        for (frames.items) |b| api.mtmd_bitmap_free(b);
        frames.deinit(x.a);
    }
    var frame_placeholders: std.ArrayList(P) = .empty;
    defer {
        for (frame_placeholders.items) |b| api.mtmd_bitmap_free(b);
        frame_placeholders.deinit(x.a);
    }

    var groups = std.mem.splitScalar(u8, inputs, ';');
    while (groups.next()) |g| {
        const colon = std.mem.indexOfScalar(u8, g, ':') orelse continue;
        const kind = g[0..colon];
        var items = std.mem.splitScalar(u8, g[colon + 1 ..], ',');
        while (items.next()) |it| {
            if (std.mem.eql(u8, kind, "img")) {
                const xpos = std.mem.indexOfScalar(u8, it, 'x') orelse continue;
                const nx = try std.fmt.parseInt(u32, it[0..xpos], 10);
                const ny = try std.fmt.parseInt(u32, it[xpos + 1 ..], 10);
                const px = try makeImage(x.a, nx, ny);
                defer x.a.free(px);
                const b = api.mtmd_bitmap_init(nx, ny, px.ptr);
                var idb: [32]u8 = undefined;
                api.mtmd_bitmap_set_id(b, (std.fmt.bufPrintSentinel(&idb, "img{d}x{d}", .{ nx, ny }, 0) catch unreachable).ptr);
                try bitmaps.append(x.a, b);
                try placeholders.append(x.a, api.mtmd_bitmap_init(nx, ny, null));
            } else if (std.mem.eql(u8, kind, "vid")) {
                // WxHxN: N frames, each its own image, all mergeable
                var dims = std.mem.splitScalar(u8, it, 'x');
                const nx = try std.fmt.parseInt(u32, dims.next() orelse continue, 10);
                const ny = try std.fmt.parseInt(u32, dims.next() orelse continue, 10);
                const nf = try std.fmt.parseInt(u32, dims.next() orelse continue, 10);
                for (0..nf) |k| {
                    const px = try makeImage(x.a, nx, ny);
                    defer x.a.free(px);
                    // a different picture per frame, the same size
                    for (px, 0..) |*p, j| p.* +%= @truncate(k * 37 + j % 5);
                    const b = api.mtmd_bitmap_init(nx, ny, px.ptr);
                    api.mtmd_bitmap_set_id(b, "video");
                    api.mtmd_bitmap_set_mergeable(b, true);
                    api.mtmd_bitmap_set_is_video(b, true);
                    try frames.append(x.a, b);
                    const ph = api.mtmd_bitmap_init(nx, ny, null);
                    api.mtmd_bitmap_set_mergeable(ph, true);
                    api.mtmd_bitmap_set_is_video(ph, true);
                    try frame_placeholders.append(x.a, ph);
                }
            } else if (std.mem.eql(u8, kind, "aud")) {
                const n = try std.fmt.parseInt(usize, it, 10);
                const s = try makeAudio(x.a, n);
                defer x.a.free(s);
                try bitmaps.append(x.a, api.mtmd_bitmap_init_from_audio(n, s.ptr));
                try placeholders.append(x.a, api.mtmd_bitmap_init_from_audio(n, null));
            }
        }
    }

    // through the helpers: a real JPEG, a WAV that needs mixing down and resampling
    const opt = api.mtmd_helper_init_opt_default();
    print("init_opt fps {d} bin_dir_null {} ts {d}\n", .{ opt.video_params.fps_target, opt.video_params.ffmpeg_bin_dir == null, opt.video_params.timestamp_interval_ms });
    if (vision) {
        const wr = api.mtmd_helper_bitmap_init_from_file(x.mctx, "llama.cpp/tools/mtmd/test-1.jpeg", false, opt);
        if (wr.bitmap) |b| {
            print("file jpeg {d}x{d} id {s}\n", .{ api.mtmd_bitmap_get_nx(b), api.mtmd_bitmap_get_ny(b), std.mem.span(api.mtmd_bitmap_get_id(b)) });
            try bitmaps.append(x.a, b);
        } else print("file jpeg null\n", .{});
    }
    if (audio) {
        const path = try std.fmt.allocPrintSentinel(x.a, "{s}/in.wav", .{tmp}, 0);
        defer x.a.free(path);
        try writeWav(x.a, path, 30000);
        const wr = api.mtmd_helper_bitmap_init_from_file(x.mctx, path.ptr, false, opt);
        if (wr.bitmap) |b| {
            print("file wav n {d} bytes {d} id {s}\n", .{ api.mtmd_bitmap_get_nx(b), api.mtmd_bitmap_get_n_bytes(b), std.mem.span(api.mtmd_bitmap_get_id(b)) });
            try bitmaps.append(x.a, b);
        } else print("file wav null\n", .{});
    }
    // a buffer neither decoder reads
    const bad = api.mtmd_helper_bitmap_init_from_buf(x.mctx, "not an image or audio file", 26, false, opt);
    print("buf bad {}\n", .{bad.bitmap == null});

    try tokenizeAndRun(x, bitmaps.items, true);
    try tokenizeAndRun(x, placeholders.items, false);
    // the frames' markers side by side, so the tokenizer merges them
    if (frames.items.len != 0) {
        print("video {d} frames\n", .{frames.items.len});
        try tokenizeAndRunAdjacent(x, frames.items, true, true);
        try tokenizeAndRunAdjacent(x, frame_placeholders.items, false, true);
        // mergeable alone does not make a video frame (v0.8.0)
        for (frames.items) |b| api.mtmd_bitmap_set_is_video(b, false);
        print("merged images {d}\n", .{frames.items.len});
        try tokenizeAndRunAdjacent(x, frames.items, true, true);
    }
    try tokenizeParts(x, placeholders.items, true);
    try tokenizeParts(x, placeholders.items, false);

    // more markers than bitmaps, and fewer
    if (bitmaps.items.len != 0) {
        const ch = api.mtmd_input_chunks_init();
        defer api.mtmd_input_chunks_free(ch);
        const t1 = "a <__media__> b <__media__>";
        const it1: M.InputText = .{ .text = t1, .text_len = t1.len, .add_special = true, .parse_special = true };
        print("tokenize-mismatch {d}\n", .{api.mtmd_tokenize(x.mctx, ch, &it1, bitmaps.items.ptr, 1)});
        const t2 = "no marker here";
        const it2: M.InputText = .{ .text = t2, .text_len = t2.len, .add_special = true, .parse_special = true };
        print("tokenize-mismatch {d}\n", .{api.mtmd_tokenize(x.mctx, ch, &it2, bitmaps.items.ptr, 1)});

        // parts with both, neither, and a null text pointer
        const both: M.InputPart = .{ .text = &it1, .bitmap = @ptrCast(@alignCast(bitmaps.items[0])) };
        const neither: M.InputPart = .{ .text = null, .bitmap = null };
        const null_txt: M.InputText = .{ .text = null, .text_len = 0, .add_special = false, .parse_special = true };
        const null_part: M.InputPart = .{ .text = &null_txt, .bitmap = null };
        for ([_]*const M.InputPart{ &both, &neither, &null_part }) |bp| {
            const one = [_]*const M.InputPart{bp};
            print("tokenize-parts-bad {d}\n", .{api.mtmd_tokenize_from_parts(x.mctx, ch, &one, one.len, true)});
        }
    }
}

/// `mtmd_tokenize_from_parts` over the same text and media `tokenizeAndRun`
/// marks up, the marker text left in a part as plain text, and the middle
/// text part with `parse_special` off.
fn tokenizeParts(x: *const Ctx, bitmaps: []const P, add_special: bool) !void {
    const api = x.api;
    if (bitmaps.len == 0) return;
    var texts: std.ArrayList(M.InputText) = .empty;
    defer texts.deinit(x.a);
    var strs: std.ArrayList([]u8) = .empty;
    defer {
        for (strs.items) |t| x.a.free(t);
        strs.deinit(x.a);
    }
    try texts.ensureTotalCapacity(x.a, bitmaps.len + 1);
    var parts: std.ArrayList(M.InputPart) = .empty;
    defer parts.deinit(x.a);
    for (0..bitmaps.len + 1) |i| {
        const t = if (i == 0)
            try x.a.dupe(u8, "Look: <__media__> <|im_start|>")
        else if (i == bitmaps.len)
            try x.a.dupe(u8, "describe them.")
        else
            try std.fmt.allocPrint(x.a, " item {d} <|im_end|>\n", .{i});
        try strs.append(x.a, t);
        texts.appendAssumeCapacity(.{ .text = t.ptr, .text_len = t.len, .add_special = true, .parse_special = i != 1 });
        try parts.append(x.a, .{ .text = &texts.items[i], .bitmap = null });
        if (i < bitmaps.len) try parts.append(x.a, .{ .text = null, .bitmap = @ptrCast(@alignCast(bitmaps[i])) });
    }
    var ptrs: std.ArrayList(*const M.InputPart) = .empty;
    defer ptrs.deinit(x.a);
    for (parts.items) |*p| try ptrs.append(x.a, p);
    const chunks = api.mtmd_input_chunks_init();
    defer api.mtmd_input_chunks_free(chunks);
    const rc = api.mtmd_tokenize_from_parts(x.mctx, chunks, ptrs.items.ptr, ptrs.items.len, add_special);
    print("tokenize-parts add_special {} rc {d}\n", .{ add_special, rc });
    flush();
    if (rc != 0) return;
    dumpChunks(x, chunks);
    flush();
}

fn tokenizeAndRun(x: *const Ctx, bitmaps: []const P, run: bool) !void {
    return tokenizeAndRunAdjacent(x, bitmaps, run, false);
}

/// `tokenizeAndRun`, with `adjacent` the markers written with no text
/// between them, as a video's frames are.
fn tokenizeAndRunAdjacent(x: *const Ctx, bitmaps: []const P, run: bool, adjacent: bool) !void {
    const api = x.api;
    if (bitmaps.len == 0) return;
    var prompt: std.ArrayList(u8) = .empty;
    defer prompt.deinit(x.a);
    try prompt.appendSlice(x.a, "Look:");
    for (0..bitmaps.len) |i| {
        if (adjacent) {
            try prompt.appendSlice(x.a, "<__media__>");
        } else try prompt.print(x.a, " item {d} <__media__>\n", .{i});
    }
    try prompt.appendSlice(x.a, "describe them.");
    const text: M.InputText = .{ .text = prompt.items.ptr, .text_len = prompt.items.len, .add_special = true, .parse_special = true };
    const chunks = api.mtmd_input_chunks_init();
    defer api.mtmd_input_chunks_free(chunks);
    const rc = api.mtmd_tokenize(x.mctx, chunks, &text, bitmaps.ptr, bitmaps.len);
    print("tokenize {s} rc {d}\n", .{ if (run) "data" else "placeholder", rc });
    flush();
    if (rc != 0) return;
    dumpChunks(x, chunks);
    flush();
    if (!run) return;
    encodeChunks(x, chunks);
    flush();
    evalChunks(x, chunks);
    flush();
    decodeMediaChunks(x, chunks);
    flush();
}

fn runTts(x: *const Ctx, modality: []const u8, steps: c_int) void {
    const pocket = std.mem.eql(u8, modality, "tts-pocket");
    const lfm3a = std.mem.startsWith(u8, modality, "tts-lfm3a");
    runTtsPrompt(x, modality, steps, if (lfm3a) "hello<|audio_start|>" else "hello there, test.");
    // pocket: two sentences past its 50-token chunk, run until the first
    // chunk's budget ends it, so finish_chunk re-prompts the second
    if (pocket) runTtsPrompt(x, modality, 400, "one two three four five six seven. eight nine ten eleven twelve.");
    if (lfm3a) {
        // a prompt the "tts" chat template did not write, and a sequence continued
        runTtsPrompt(x, modality, steps, "no mode token");
        runLfm3aContinue(x);
    } else {
        // only LFM3-Audio continues a sequence
        const lctx = x.lctx orelse return;
        const gen = x.api.mtmd_helper_gen_audio_init(lctx, x.mctx);
        defer x.api.mtmd_helper_gen_audio_free(gen);
        var inp = x.api.mtmd_helper_gen_audio_inp_default();
        inp.n_past = 3;
        print("tts continue refused {d}\n", .{x.api.mtmd_helper_gen_audio_set_input(gen, &inp)});
    }
    runGenApi(x, modality);
    runTtsStream(x, modality, .pcm, steps);
    runTtsStream(x, modality, .wav, steps);
}

fn runTtsPrompt(x: *const Ctx, modality: []const u8, steps: c_int, prompt: [:0]const u8) void {
    const qwen = std.mem.eql(u8, modality, "tts-qwen3");
    const soprano = std.mem.eql(u8, modality, "tts-soprano");
    const api = x.api;
    const lctx = x.lctx orelse return;
    const gen = api.mtmd_helper_gen_audio_init(lctx, x.mctx);
    defer api.mtmd_helper_gen_audio_free(gen);
    const spk = makeAudio(x.a, 24000) catch return;
    defer x.a.free(spk);
    const spk_bm = api.mtmd_bitmap_init_from_audio(spk.len, spk.ptr);
    defer api.mtmd_bitmap_free(spk_bm);
    api.llama_memory_clear(api.llama_get_memory(lctx), true);
    var inp = api.mtmd_helper_gen_audio_inp_default();
    inp.prompt = prompt.ptr;
    inp.prompt_len = prompt.len;
    inp.speaker_ref = if (soprano) null else @ptrCast(@alignCast(spk_bm));
    inp.lang = "";
    inp.top_k = 8;
    inp.seed = 42;
    var rc = api.mtmd_helper_gen_audio_set_input(gen, &inp);
    print("tts set_input {d}\n", .{rc});
    if (rc != 0) return;
    while (true) {
        // a few tokens a step, so the prompt takes several batches and the
        // last batch's output row sits past its start
        const r = api.mtmd_helper_gen_audio_step_prompt(gen, 5);
        print("tts step_prompt {d}\n", .{r});
        if (r <= 0) break;
    }
    const vocab = api.llama_model_get_vocab(x.model);
    var codec_0: c.llama_token = -1;
    if (qwen) {
        var t: c.llama_token = 0;
        while (t < api.llama_vocab_n_tokens(vocab)) : (t += 1) {
            if (std.mem.eql(u8, std.mem.span(@as([*:0]const u8, @ptrCast(api.llama_vocab_get_text(vocab, t)))), "<|codec_0|>")) {
                codec_0 = t;
                break;
            }
        }
    }
    const n_embd: usize = @intCast(x.api.llama_model_n_embd(x.model));
    var h: ?[*]const f32 = api.llama_get_embeddings_ith(lctx, -1);
    var s: c_int = 0;
    while (s < steps) : (s += 1) {
        const sampled: c.llama_token = if (qwen) codec_0 + 7 * (s + 1) else if (soprano) 10 + s else -1;
        var h_next: ?[*]const f32 = null;
        var stop = false;
        rc = api.mtmd_helper_gen_audio_step_gen(gen, sampled, h, &h_next, &stop);
        print("tts step_gen {d} rc {d} stop {} h {x:0>16}\n", .{ s, rc, stop, hashF32(h_next, n_embd) });
        flush();
        if (rc != 0 or h_next == null or stop) break;
        h = h_next;
    }
    var sr: i32 = 0;
    var data: ?[*]const u8 = null;
    var len: usize = 0;
    var n_samples: i64 = 0;
    rc = api.mtmd_helper_gen_audio_get_output(gen, &sr, &data, &len, &n_samples);
    print("tts output rc {d} sr {d} len {d} samples {d} wav {x:0>16}\n", .{ rc, sr, len, n_samples, if (data) |d| hashBytes(d[0..len]) else 0 });
}

// -----------------------------------------------------------------------------
// v0.8.0: audio streams, gen state and the soundtrack of a video

/// The token whose text is `text`, -1 if the vocabulary has none.
fn tokenOf(x: *const Ctx, text: []const u8) c.llama_token {
    const api = x.api;
    const vocab = api.llama_model_get_vocab(x.model);
    var t: c.llama_token = 0;
    while (t < api.llama_vocab_n_tokens(vocab)) : (t += 1) {
        if (std.mem.eql(u8, std.mem.span(@as([*:0]const u8, @ptrCast(api.llama_vocab_get_text(vocab, t)))), text)) return t;
    }
    return -1;
}

/// One audio stream's script.
const StreamScript = struct {
    label: []const u8,
    n: usize, // samples, 0 for none
    piece: usize = 0, // samples a push, growing by `grow`; 0 pushes all at once
    grow: usize = 0,
    step_ms: i32 = 320,
    prompted: bool = false, // open with a prompt: LFM3-Audio's live ASR
    text_len: usize = 0, // text tokens of a chunk before <|text_end|>
    mid_flush: usize = 0, // a flush(false) after this many samples
    no_end: bool = false, // leave the stream open, no flush(true)
    bad_token: ?c.llama_token = null, // pass this after the first rows
};

/// Feeds PCM to a stream in pieces and prints every step: its return value,
/// `is_seq_start`, both row counters and a hash of every output row. Nothing
/// is decoded: the tokens are a script, so a step's rows depend only on the
/// stream, and each step is compared, not only the whole.
fn runStream(x: *const Ctx, sc: StreamScript) void {
    const api = x.api;
    var sp = api.mtmd_audio_stream_params_default();
    sp.step_ms = sc.step_ms;
    const prompt = [_]c.llama_token{ 11, 12, 13, 14 };
    if (sc.prompted) {
        sp.prompt = &prompt;
        sp.n_prompt = prompt.len;
    }
    print("stream {s} n {d} step_ms {d} prompted {}\n", .{ sc.label, sc.n, sc.step_ms, sc.prompted });
    const s = api.mtmd_audio_stream_init(x.mctx, sp) orelse {
        print("stream init null\n", .{});
        return;
    };
    defer api.mtmd_audio_stream_free(s);
    const pcm = makeAudio(x.a, sc.n) catch return;
    defer x.a.free(pcm);
    const n_embd: usize = @intCast(api.llama_model_n_embd_inp(x.model));
    const text_end = tokenOf(x, "<|text_end|>");

    var fed: usize = 0;
    var piece = sc.piece;
    var flushed = false;
    var mid_done = sc.mid_flush == 0;
    var tok: c.llama_token = -1;
    var n_text: usize = 0;
    var bad_sent = false;
    var it: usize = 0;
    while (it < 2_000_000) : (it += 1) {
        const r = api.mtmd_audio_stream_step(s, tok);
        const rows: usize = if (r > 0) @intCast(r) else 0;
        print("step {d} r {d} seq {} rows {d} flushed {d} h {x:0>16}\n", .{
            it,                                          r,
            api.mtmd_audio_stream_is_seq_start(s),       api.mtmd_audio_stream_get_n_rows(s),
            api.mtmd_audio_stream_get_n_rows_flushed(s), hashF32(api.mtmd_audio_stream_get_output_embd(s), rows * n_embd),
        });
        flush();
        if (r < 0) break;
        if (r == 0) {
            if (fed < sc.n) {
                const cnt = if (sc.piece == 0) sc.n - fed else @min(piece, sc.n - fed);
                print("push {d} rc {d}\n", .{ cnt, api.mtmd_audio_stream_push(s, pcm.ptr + fed, cnt) });
                fed += cnt;
                piece += sc.grow;
                if (!mid_done and fed >= sc.mid_flush) {
                    mid_done = true;
                    print("flush mid rc {d}\n", .{api.mtmd_audio_stream_flush(s, false)});
                }
            } else if (!flushed and !sc.no_end) {
                print("flush end rc {d}\n", .{api.mtmd_audio_stream_flush(s, true)});
                flushed = true;
            } else if (flushed) {
                print("stream stuck\n", .{});
                break;
            } else {
                break;
            }
            continue;
        }
        if (sc.bad_token) |b| {
            if (!bad_sent) {
                bad_sent = true;
                tok = b;
                continue;
            }
        }
        // the next token: a chunk of text, then <|text_end|> where the vocabulary has one
        if (text_end >= 0 and n_text >= sc.text_len) {
            tok = text_end;
            n_text = 0;
        } else {
            tok = 20 + @as(c.llama_token, @intCast(it % 7));
            n_text += 1;
        }
    }
    // an ended stream refuses more
    const one = [_]f32{0.5};
    print("after push {d} flush {d} step {d}\n", .{ api.mtmd_audio_stream_push(s, &one, 1), api.mtmd_audio_stream_flush(s, false), api.mtmd_audio_stream_step(s, tok) });
}

/// Every script a streaming model runs, `sizes` from the manifest among them.
fn runStreams(x: *const Ctx, sizes: []const u8) !void {
    const api = x.api;
    const dp = api.mtmd_audio_stream_params_default();
    print("stream support {} default step_ms {d} prompt_null {} n_prompt {d}\n", .{ api.mtmd_support_audio_stream(x.mctx), dp.step_ms, dp.prompt == null, dp.n_prompt });
    var bias: [8]c.llama_logit_bias = undefined;
    const n_bias = api.mtmd_helper_audio_stream_get_logit_bias(x.mctx, x.model, &bias, bias.len);
    print("stream bias {d}\n", .{n_bias});
    for (bias[0..@intCast(@min(n_bias, 8))]) |b| print("bias {d} {x}\n", .{ b.token, @as(u32, @bitCast(b.bias)) });
    print("stream bias short {d}\n", .{api.mtmd_helper_audio_stream_get_logit_bias(x.mctx, x.model, &bias, 2)});
    print("stream bias none {d}\n", .{api.mtmd_helper_audio_stream_get_logit_bias(x.mctx, x.model, null, 0)});
    if (!api.mtmd_support_audio_stream(x.mctx)) {
        runStream(x, .{ .label = "unsupported", .n = 100 });
        return;
    }
    const live = tokenOf(x, "<|text_end|>") >= 0;
    var it = std.mem.splitScalar(u8, sizes, ',');
    while (it.next()) |sz| {
        const n = try std.fmt.parseInt(usize, sz, 10);
        runStream(x, .{ .label = "uneven", .n = n, .piece = 1000, .grow = 317 });
        runStream(x, .{ .label = "whole", .n = n });
        if (live) runStream(x, .{ .label = "live", .n = n, .piece = 1000, .grow = 317, .prompted = true, .text_len = 2 });
    }
    runStream(x, .{ .label = "step 1 ms", .n = 9000, .piece = 700, .step_ms = 1 });
    runStream(x, .{ .label = "step 1 s", .n = 40000, .piece = 3000, .step_ms = 1000 });
    runStream(x, .{ .label = "step 100 s", .n = 40000, .piece = 3000, .step_ms = 100000 });
    runStream(x, .{ .label = "mid flush", .n = 40000, .piece = 2000, .mid_flush = 12000 });
    runStream(x, .{ .label = "open", .n = 20000, .piece = 2000, .no_end = true });
    runStream(x, .{ .label = "bad token", .n = 40000, .piece = 4000, .bad_token = 100000 });
    runStream(x, .{ .label = "null token", .n = 40000, .piece = 4000, .bad_token = -1 });
    if (live) {
        runStream(x, .{ .label = "live mid flush", .n = 40000, .piece = 2000, .prompted = true, .text_len = 1, .mid_flush = 11000 });
        runStream(x, .{ .label = "live empty chunks", .n = 260000, .piece = 4000, .prompted = true });
        runStream(x, .{ .label = "live 30 s", .n = 500000, .piece = 8000, .prompted = true, .text_len = 1 });
        runStream(x, .{ .label = "live runaway", .n = 30000, .piece = 4000, .prompted = true, .text_len = 200 });
        runStream(x, .{ .label = "live bad token", .n = 30000, .piece = 4000, .prompted = true, .bad_token = 100000 });
        runStream(x, .{ .label = "live null token", .n = 30000, .piece = 4000, .prompted = true, .bad_token = -1 });
        runStream(x, .{ .label = "live none", .n = 0, .prompted = true });
    } else {
        // the ring wraps, and past 1024 rows the consumed ones are dropped
        runStream(x, .{ .label = "long", .n = 1400000, .piece = 16000 });
        const s = api.mtmd_audio_stream_params_default();
        var bad = s;
        const p = [_]c.llama_token{1};
        bad.prompt = &p;
        bad.n_prompt = 1;
        print("prompted voxtral {}\n", .{api.mtmd_audio_stream_init(x.mctx, bad) == null});
    }
}

/// `mtmd_gen_audio_process` through a state of its own: the output of every
/// call, the state's reset, and what it refuses.
fn runGenApi(x: *const Ctx, modality: []const u8) void {
    const api = x.api;
    const dp = api.mtmd_gen_state_params_default();
    print("gen state default seed {d}\n", .{dp.seed});
    var sp = dp;
    sp.seed = 7;
    const st = api.mtmd_gen_state_init(x.mctx, sp) orelse {
        print("gen state null\n", .{});
        return;
    };
    defer api.mtmd_gen_state_free(st);
    const n_embd: usize = @intCast(api.llama_model_n_embd(x.model));
    const h = x.a.alloc(f32, n_embd) catch return;
    defer x.a.free(h);
    for (h, 0..) |*v, i| v.* = 0.5 * @sin(0.13 * @as(f32, @floatFromInt(i)));

    const Show = struct {
        fn out(a: *const Api, label: []const u8, rc: i32, o: *const M.GenOut, ne: usize) void {
            _ = a;
            print("gen {s} rc {d} codes {d} {x:0>16} feats {d} {x:0>16} embd {x:0>16} eos {} audio {d} {x:0>16}\n", .{
                label,                                                                    rc,        o.n_codes,
                if (o.codes) |p| hashBytes(std.mem.sliceAsBytes(p[0..o.n_codes])) else 0, o.n_feats, hashF32(o.feats, o.n_feats),
                hashF32(o.embd, if (o.embd != null) ne else 0),                           o.is_eos,  o.n_samples,
                hashF32(o.audio, o.n_samples),
            });
            flush();
        }
    };

    var inp = api.mtmd_gen_inp_default(x.mctx);
    print("gen inp type {d} top_k {d} top_p {x} temp {x} penalty {x} last_n {d}\n", .{ @backingInt(inp.type), inp.top_k, @as(u32, @bitCast(inp.top_p)), @as(u32, @bitCast(inp.temp)), @as(u32, @bitCast(inp.penalty_repeat)), inp.penalty_last_n });
    var o = std.mem.zeroes(M.GenOut);
    Show.out(api, "no state", api.mtmd_gen_audio_process(x.mctx, null, &inp, &o), &o, n_embd);

    if (std.mem.eql(u8, modality, "tts-soprano")) {
        const feats = x.a.alloc(f32, n_embd * 5) catch return;
        defer x.a.free(feats);
        for (feats, 0..) |*v, i| v.* = 0.3 * @sin(0.07 * @as(f32, @floatFromInt(i)));
        inp.type = .gen_wav;
        Show.out(api, "soprano none", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        inp.feats = feats.ptr;
        inp.n_feats = n_embd;
        Show.out(api, "soprano one frame", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        inp.n_feats = n_embd * 2 + 1;
        Show.out(api, "soprano ragged", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        inp.n_feats = n_embd * 2;
        Show.out(api, "soprano 2", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        inp.n_feats = n_embd * 5;
        Show.out(api, "soprano 5", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        inp.type = .gen_code;
        Show.out(api, "soprano gen_code", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        return;
    }
    if (std.mem.startsWith(u8, modality, "tts-lfm3a")) {
        var codes: [12]i32 = @splat(-1);
        inp.type = .gen_code;
        inp.embd = h.ptr;
        inp.codes = &codes;
        inp.n_codes = codes.len;
        Show.out(api, "code masked", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        codes[3] = 2;
        codes[7] = 5;
        Show.out(api, "code partial", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        inp.embd = null;
        Show.out(api, "code no h", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        for (&codes, 0..) |*v, i| v.* = @intCast(i % 7);
        Show.out(api, "code known", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        codes[0] = 100000;
        Show.out(api, "code range", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        codes[0] = 1;

        inp = api.mtmd_gen_inp_default(x.mctx);
        inp.type = .gen_wav;
        inp.codes = &codes;
        inp.n_codes = codes.len;
        Show.out(api, "wav 1", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        codes[4] = 3;
        Show.out(api, "wav 2", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        var two: [24]i32 = undefined;
        for (&two, 0..) |*v, i| v.* = @intCast((i * 3) % 7);
        inp.codes = &two;
        inp.n_codes = two.len;
        Show.out(api, "wav 2 frames", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        two[5] = 100000;
        Show.out(api, "wav range", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        two[5] = 1;
        inp.codes = null;
        inp.n_codes = 0;
        Show.out(api, "wav flush", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        inp.codes = &codes;
        inp.n_codes = codes.len;
        Show.out(api, "wav again", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        api.mtmd_gen_state_reset(st);
        Show.out(api, "wav reset", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
        return;
    }
    // the older pipelines, with gen_wav's refusal
    inp.type = .gen_wav;
    Show.out(api, "wav none", api.mtmd_gen_audio_process(x.mctx, st, &inp, &o), &o, n_embd);
}

/// The pipelines in streaming mode: every step's new audio, then the flush.
fn runTtsStream(x: *const Ctx, modality: []const u8, out_type: G.OutType, steps: c_int) void {
    const api = x.api;
    const lctx = x.lctx orelse return;
    const gen = api.mtmd_helper_gen_audio_init(lctx, x.mctx);
    defer api.mtmd_helper_gen_audio_free(gen);
    api.llama_memory_clear(api.llama_get_memory(lctx), true);
    const qwen = std.mem.eql(u8, modality, "tts-qwen3");
    const soprano = std.mem.eql(u8, modality, "tts-soprano");
    const lfm3a = std.mem.startsWith(u8, modality, "tts-lfm3a");
    const prompt: [:0]const u8 = if (lfm3a) "hello<|audio_start|>" else "hello there, test.";
    var inp = api.mtmd_helper_gen_audio_inp_default();
    print("gen inp default top_k {d} top_p {x} seed {d} out {d}\n", .{ inp.top_k, @as(u32, @bitCast(inp.top_p)), inp.seed, @backingInt(inp.out_type) });
    inp.stream = true;
    inp.prompt = prompt.ptr;
    inp.prompt_len = prompt.len;
    inp.lang = "";
    inp.top_k = 8;
    inp.seed = 42;
    inp.out_type = out_type;
    var rc = api.mtmd_helper_gen_audio_set_input(gen, &inp);
    print("stream tts set_input {d}\n", .{rc});
    if (rc != 0) return;
    while (true) {
        const r = api.mtmd_helper_gen_audio_step_prompt(gen, 5);
        if (r <= 0) break;
    }
    const n_embd: usize = @intCast(api.llama_model_n_embd(x.model));
    const codec_0 = tokenOf(x, "<|codec_0|>");
    var h: ?[*]const f32 = api.llama_get_embeddings_ith(lctx, -1);
    const Out = struct {
        fn show(a: *const Api, g: P, label: []const u8) void {
            var sr: i32 = 0;
            var data: ?[*]const u8 = null;
            var len: usize = 0;
            var ns: i64 = 0;
            const r = a.mtmd_helper_gen_audio_get_output(g, &sr, &data, &len, &ns);
            print("stream tts {s} rc {d} sr {d} len {d} samples {d} data {x:0>16}\n", .{ label, r, sr, len, ns, if (data != null and len != 0) hashBytes(data.?[0..len]) else 0 });
            flush();
        }
    };
    var s: c_int = 0;
    while (s < steps) : (s += 1) {
        const sampled: c.llama_token = if (qwen) codec_0 + 7 * (s + 1) else if (soprano) 10 + s else -1;
        var h_next: ?[*]const f32 = null;
        var stop = false;
        rc = api.mtmd_helper_gen_audio_step_gen(gen, sampled, h, &h_next, &stop);
        print("stream tts step {d} rc {d} stop {} h {x:0>16}\n", .{ s, rc, stop, hashF32(h_next, n_embd) });
        Out.show(api, gen, "step");
        if (rc != 0 or h_next == null or stop) break;
        h = h_next;
    }
    print("stream tts flush {d}\n", .{api.mtmd_helper_gen_audio_flush(gen)});
    Out.show(api, gen, "flushed");
    Out.show(api, gen, "again");
}

/// LFM3-Audio continuing a sequence: each out mode, the model's own switching
/// driven by a script of sampled tokens.
fn runLfm3aContinue(x: *const Ctx) void {
    const api = x.api;
    const lctx = x.lctx orelse return;
    const gen = api.mtmd_helper_gen_audio_init(lctx, x.mctx);
    defer api.mtmd_helper_gen_audio_free(gen);
    const n_embd: usize = @intCast(api.llama_model_n_embd(x.model));
    const t_end = tokenOf(x, "<|text_end|>");
    const t_im_end = tokenOf(x, "<|im_end|>");
    const t_audio = tokenOf(x, "<|audio_start|>");
    const t_mixed = tokenOf(x, "<|mixed_start|>");
    const modes = [_]G.OutMode{ .auto, .auto, .text, .audio, .mixed };
    const scripts = [_][]const c.llama_token{
        &.{ 30, 31, t_audio, 0, 0, 0, 0, 30, t_end },
        &.{ 30, t_mixed, 31, 32, 33, 34, 35, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, t_end, 0, 0, t_im_end },
        &.{ 30, 31, 32, 33, t_im_end },
        &.{ 0, 0, 0, 0, 0, 0 },
        &.{ 30, 31, 32, 33, 34, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 37, t_end, 0, 0, 0, 38 },
    };
    for (modes, scripts) |mode, script| {
        api.llama_memory_clear(api.llama_get_memory(lctx), true);
        const first: [:0]const u8 = "hello<|audio_start|>";
        var inp = api.mtmd_helper_gen_audio_inp_default();
        inp.prompt = first.ptr;
        inp.prompt_len = first.len;
        inp.seed = 11;
        inp.out_type = .pcm;
        var rc = api.mtmd_helper_gen_audio_set_input(gen, &inp);
        // a token a step, to count the positions the sequence holds
        var n_past: i32 = 0;
        while (rc == 0) {
            const r = api.mtmd_helper_gen_audio_step_prompt(gen, 1);
            if (r < 0) break;
            n_past += 1;
            if (r == 0) break;
        }
        const next: [:0]const u8 = "<|im_start|>assistant\n";
        inp.prompt = next.ptr;
        inp.prompt_len = next.len;
        inp.n_past = n_past;
        inp.out_mode = mode;
        inp.stream = true;
        rc = api.mtmd_helper_gen_audio_set_input(gen, &inp);
        print("continue mode {d} set_input {d}\n", .{ @backingInt(mode), rc });
        if (rc != 0) continue;
        var n_bias: usize = 0;
        const bias = api.mtmd_helper_gen_audio_get_logit_bias(gen, &n_bias);
        print("continue bias {d}", .{n_bias});
        if (bias) |b| for (b[0..n_bias]) |e| print(" {d}:{x}", .{ e.token, @as(u32, @bitCast(e.bias)) });
        print("\n", .{});
        while (true) {
            const r = api.mtmd_helper_gen_audio_step_prompt(gen, 512);
            print("continue prompt {d}\n", .{r});
            if (r <= 0) break;
        }
        var h: ?[*]const f32 = api.llama_get_embeddings_ith(lctx, -1);
        for (script, 0..) |tok, si| {
            const is_text = api.mtmd_helper_gen_audio_next_is_text(gen);
            var h_next: ?[*]const f32 = null;
            var stop = false;
            rc = api.mtmd_helper_gen_audio_step_gen(gen, tok, h, &h_next, &stop);
            print("continue step {d} text {} rc {d} stop {} h {x:0>16}\n", .{ si, is_text, rc, stop, hashF32(h_next, n_embd) });
            flush();
            if (rc != 0 or stop or h_next == null) break;
            h = h_next;
        }
        var sr: i32 = 0;
        var data: ?[*]const u8 = null;
        var len: usize = 0;
        var ns: i64 = 0;
        print("continue flush {d}\n", .{api.mtmd_helper_gen_audio_flush(gen)});
        rc = api.mtmd_helper_gen_audio_get_output(gen, &sr, &data, &len, &ns);
        print("continue out rc {d} len {d} samples {d} data {x:0>16}\n", .{ rc, len, ns, if (data != null and len != 0) hashBytes(data.?[0..len]) else 0 });
    }
    // a pipeline that cannot continue a sequence refuses it, see runTts
}

/// What `lazyVideo` serves.
const LazyVideo = struct {
    x: *const Ctx,
    n_groups: usize,
    placeholder: bool,
    audio_rows: usize, // samples of each piece
};

fn dupC(s: []const u8) ?[*:0]u8 {
    const m = std.c.malloc(s.len + 1) orelse return null;
    const p: [*]u8 = @ptrCast(m);
    @memcpy(p[0..s.len], s);
    p[s.len] = 0;
    return @ptrCast(p);
}

/// A video as `mtmd_helper_video_read_next` serves LFM3-Omni's: per group a
/// timestamp, two frames, then the group's piece of the soundtrack.
fn lazyVideo(i: usize, ud: ?*anyopaque, out_bm: *?*anyopaque, out_str: *?[*:0]u8) callconv(.c) c_int {
    const v: *const LazyVideo = @ptrCast(@alignCast(ud.?));
    const api = v.x.api;
    const group = i / 4;
    if (group >= v.n_groups) return -1;
    switch (i % 4) {
        0 => {
            var buf: [32]u8 = undefined;
            const t = std.fmt.bufPrint(&buf, "<{d}.0 seconds>", .{group}) catch return -2;
            out_str.* = dupC(t);
        },
        1, 2 => {
            const px = makeImage(v.x.a, 64, 48) catch return -2;
            defer v.x.a.free(px);
            const b = api.mtmd_bitmap_init(64, 48, px.ptr);
            api.mtmd_bitmap_set_mergeable(b, true);
            api.mtmd_bitmap_set_is_video(b, true);
            out_bm.* = b;
        },
        else => {
            if (!api.mtmd_support_audio(v.x.mctx)) return 0; // nothing in this slot, the video is silent
            const n = v.audio_rows + group * 513;
            const s = makeAudio(v.x.a, n) catch return -2;
            defer v.x.a.free(s);
            const b = api.mtmd_bitmap_init_from_audio(n, if (v.placeholder) null else s.ptr);
            var idbuf: [32]u8 = undefined;
            const id = std.fmt.bufPrintSentinel(&idbuf, "video+audio{d}", .{group}, 0) catch return -2;
            api.mtmd_bitmap_set_id(b, id.ptr);
            out_bm.* = b;
        },
    }
    return 0;
}

/// A lazy video bitmap through the tokenizer: the video markers, frames and,
/// on LFM3-Omni, the pieces of one soundtrack.
fn runVideo(x: *const Ctx) !void {
    const api = x.api;
    for ([_]struct { w: i32, h: i32, n: i32, f: f32 }{
        .{ .w = 64, .h = 48, .n = 0, .f = 0 },      .{ .w = 64, .h = 48, .n = 30, .f = 30 },
        .{ .w = 640, .h = 480, .n = 300, .f = 25 }, .{ .w = 1920, .h = 1080, .n = 90000, .f = 30 },
        .{ .w = 33, .h = 17, .n = 5, .f = 60 },     .{ .w = 100, .h = 100, .n = 2, .f = 1 },
    }) |v| {
        const cfg = api.mtmd_get_video_config(x.mctx, v.w, v.h, v.n, v.f);
        print("video config {d}x{d} n {d} fps {d}: tmpl {d} fps {x} n {d} size {d}x{d}\n", .{ v.w, v.h, v.n, v.f, @backingInt(cfg.tmpl), @as(u32, @bitCast(cfg.fps)), cfg.n_frames, cfg.frame_width, cfg.frame_height });
    }
    if (!api.mtmd_support_vision(x.mctx)) return;
    for ([_]bool{ false, true }) |placeholder| {
        var lv: LazyVideo = .{ .x = x, .n_groups = 3, .placeholder = placeholder, .audio_rows = 6000 };
        const bm = api.mtmd_bitmap_init_lazy(x.mctx, "video", &lv, @ptrCast(&lazyVideo));
        defer api.mtmd_bitmap_free(bm);
        api.mtmd_bitmap_set_is_video(bm, true);
        print("lazy video placeholder {}\n", .{placeholder});
        const bms = [_]P{bm};
        try tokenizeAndRun(x, &bms, !placeholder);
    }
}

pub fn main(init: std.process.Init) !u8 {
    const a = std.heap.c_allocator;
    g_io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 10) {
        std.debug.print("usage: mtmd_diff <ref|port> <cpu|gpu> <nodes|fused> <tmp> <projector> <mmproj> <text-model> <modality> <inputs>\n", .{});
        return 2;
    }
    const api: *const Api = if (std.mem.eql(u8, args[1], "port")) &port_api else &ref_api;
    const gpu = std.mem.eql(u8, args[2], "gpu");
    node_mode = std.mem.eql(u8, args[3], "nodes");
    const tmp = args[4];
    const mmproj = args[6];
    const text_model = args[7];
    const modality = args[8];
    const inputs = args[9];

    out_file = std.Io.File.stdout();
    out_writer = out_file.writer(g_io, &out_buf);
    defer flush();

    c.ggml_log_set(logCb, null);
    api.llama_log_set(logCb, null);
    api.mtmd_helper_log_set(logCb, null); // and libmtmd's

    print("projector {s} modality {s}\n", .{ args[5], modality });
    const cap = api.mtmd_get_cap_from_file(mmproj.ptr);
    print("cap vision {} audio {} gen {}\n", .{ cap.inp_vision, cap.inp_audio, cap.gen_audio });
    const cap_missing = api.mtmd_get_cap_from_file("/nonexistent.gguf");
    print("cap missing vision {} audio {} gen {}\n", .{ cap_missing.inp_vision, cap_missing.inp_audio, cap_missing.gen_audio });

    // the test chunks, as test-mtmd-c-api.c reads them
    if (api.mtmd_test_create_input_chunks()) |tc| {
        print("test chunks {d} {d}\n", .{ api.mtmd_input_chunks_size(tc), api.mtmd_helper_get_n_tokens(tc) });
        api.mtmd_input_chunks_free(tc);
    }
    flush();

    var mp = api.llama_model_default_params();
    mp.n_gpu_layers = if (gpu) 999 else 0;
    if (!gpu) mp.devices = null;
    const model = api.llama_model_load_from_file(text_model.ptr, mp) orelse {
        print("text model failed\n", .{});
        return 0;
    };
    defer api.llama_model_free(model);

    var cp = api.llama_context_default_params();
    cp.n_ctx = 8192;
    cp.n_batch = 2048;
    cp.n_ubatch = 2048;
    cp.embeddings = std.mem.startsWith(u8, modality, "tts");
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    const lctx = api.llama_init_from_model(model, cp);
    defer api.llama_free(lctx);
    flush();

    var mparams = api.mtmd_context_params_default();
    mparams.use_gpu = gpu;
    mparams.n_threads = 4;
    if (node_mode) mparams.cb_eval = cbEval;
    const mctx = api.mtmd_init_from_file(mmproj.ptr, model, mparams) orelse {
        print("init null\n", .{});
        return 0;
    };
    defer api.mtmd_free(mctx);
    print("init ok vision {} audio {} rate {d} mrope {} non_causal {} marker {s}\n", .{
        api.mtmd_support_vision(mctx),   api.mtmd_support_audio(mctx),               api.mtmd_get_audio_sample_rate(mctx),
        api.mtmd_decode_use_mrope(mctx), api.mtmd_decode_use_non_causal(mctx, null), std.mem.span(api.mtmd_get_marker(mctx)),
    });
    memoryUsage(api == &port_api, mmproj.ptr, mparams);
    // with a custom budget, which the token cap reads
    var mparams_cap = mparams;
    mparams_cap.image_max_tokens = 40;
    mparams_cap.image_min_tokens = 16;
    memoryUsage(api == &port_api, mmproj.ptr, mparams_cap);
    memoryUsage(api == &port_api, "/nonexistent.gguf", mparams);
    const info = api.mtmd_gen_audio_get_info(mctx);
    const gi = api.mtmd_gen_inp_default(mctx);
    print("gen type {d} rate {d} variant {s} default top_k {d} top_p {x} temp {x}\n", .{ @backingInt(info.type), info.sample_rate, if (info.model_variant) |v| std.mem.span(v) else "(null)", gi.top_k, @as(u32, @bitCast(gi.top_p)), @as(u32, @bitCast(gi.temp)) });
    print("can_chat {}\n", .{api.mtmd_helper_model_can_chat(lctx, mctx)});
    const hc = api.mtmd_helper_get_capabilities(model, mctx);
    const hc0 = api.mtmd_helper_get_capabilities(model, null);
    print("helper cap chat {} vision {} audio {} video {} stream {} text {} gen {} spk {} support_video {} null chat {} text {}\n", .{
        hc.chat,                             hc.vision, hc.audio,     hc.video, hc.audio_stream, hc.text_gen, hc.audio_gen, hc.audio_gen_spk_ref,
        api.mtmd_helper_support_video(mctx), hc0.chat,  hc0.text_gen,
    });
    flush();

    const ctx: Ctx = .{ .a = a, .api = api, .mctx = mctx, .model = model, .lctx = lctx };
    if (std.mem.startsWith(u8, modality, "tts")) {
        var steps: c_int = 3;
        if (std.mem.indexOf(u8, inputs, "tts:")) |i| steps = std.fmt.parseInt(c_int, std.mem.sliceTo(inputs[i + 4 ..], ';'), 10) catch 3;
        runTts(&ctx, modality, steps);
    } else {
        // a stream-only model takes no media through the tokenizer
        const stream_only = std.mem.startsWith(u8, inputs, "strm:");
        if (!stream_only) try runMedia(&ctx, tmp, inputs, api.mtmd_support_vision(mctx), api.mtmd_support_audio(mctx));
        try runVideo(&ctx);
        if (std.mem.indexOf(u8, inputs, "strm:")) |i| {
            try runStreams(&ctx, std.mem.sliceTo(inputs[i + 5 ..], ';'));
        } else {
            print("stream init on a non-stream model\n", .{});
            runStream(&ctx, .{ .label = "unsupported", .n = 100 });
        }
    }
    print("done\n", .{});
    return 0;
}
