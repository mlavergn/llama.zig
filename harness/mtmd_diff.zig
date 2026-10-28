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
//! `mtmd_helper_eval_chunks`; and for a TTS fixture the generation helper's
//! hidden states and output WAV.
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
    mtmd_helper_model_can_chat: *const fn (OptP, OptP) callconv(.c) bool,
    mtmd_helper_gen_audio_init: *const fn (OptP, P) callconv(.c) P,
    mtmd_helper_gen_audio_free: *const fn (OptP) callconv(.c) void,
    mtmd_helper_gen_audio_set_input: *const fn (P, *const G.Inp) callconv(.c) i32,
    mtmd_helper_gen_audio_step_prompt: *const fn (P, i32) callconv(.c) i32,
    mtmd_helper_gen_audio_step_gen: *const fn (P, c.llama_token, ?[*]const f32, *?[*]const f32, ?*bool) callconv(.c) i32,
    mtmd_helper_gen_audio_get_output: *const fn (P, *i32, *?[*]const u8, *usize, ?*i64) callconv(.c) i32,

    llama_log_set: *const fn (c.ggml_log_callback, OptP) callconv(.c) void,
    llama_model_default_params: *const fn () callconv(.c) c.llama_model_params,
    llama_model_load_from_file: *const fn ([*:0]const u8, c.llama_model_params) callconv(.c) ?*c.llama_model,
    llama_model_free: *const fn (?*c.llama_model) callconv(.c) void,
    llama_context_default_params: *const fn () callconv(.c) c.llama_context_params,
    llama_init_from_model: *const fn (?*c.llama_model, c.llama_context_params) callconv(.c) ?*c.llama_context,
    llama_free: *const fn (?*c.llama_context) callconv(.c) void,
    llama_get_logits_ith: *const fn (?*c.llama_context, i32) callconv(.c) ?[*]f32,
    llama_get_embeddings_ith: *const fn (?*c.llama_context, i32) callconv(.c) ?[*]f32,
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
    const api = x.api;
    if (bitmaps.len == 0) return;
    var prompt: std.ArrayList(u8) = .empty;
    defer prompt.deinit(x.a);
    try prompt.appendSlice(x.a, "Look:");
    for (0..bitmaps.len) |i| try prompt.print(x.a, " item {d} <__media__>\n", .{i});
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
}

fn runTts(x: *const Ctx, qwen: bool, steps: c_int) void {
    const api = x.api;
    const lctx = x.lctx orelse return;
    const gen = api.mtmd_helper_gen_audio_init(lctx, x.mctx);
    defer api.mtmd_helper_gen_audio_free(gen);
    const spk = makeAudio(x.a, 24000) catch return;
    defer x.a.free(spk);
    const spk_bm = api.mtmd_bitmap_init_from_audio(spk.len, spk.ptr);
    defer api.mtmd_bitmap_free(spk_bm);
    const prompt = "hello there, test.";
    var inp: G.Inp = .{ .seq_id = 0, .prompt = prompt, .prompt_len = prompt.len, .speaker_ref = @ptrCast(@alignCast(spk_bm)), .lang = "", .top_k = 8, .top_p = 1.0, .seed = 42, .out_type = .wav };
    var rc = api.mtmd_helper_gen_audio_set_input(gen, &inp);
    print("tts set_input {d}\n", .{rc});
    if (rc != 0) return;
    while (true) {
        const r = api.mtmd_helper_gen_audio_step_prompt(gen, 16);
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
        const sampled: c.llama_token = if (qwen) codec_0 + 7 * (s + 1) else -1;
        var h_next: ?[*]const f32 = null;
        var stop = false;
        rc = api.mtmd_helper_gen_audio_step_gen(gen, sampled, h, &h_next, &stop);
        print("tts step_gen {d} rc {d} stop {} h {x:0>16}\n", .{ s, rc, stop, hashF32(h_next, n_embd) });
        flush();
        if (rc != 0 or h_next == null) break;
        h = h_next;
    }
    var sr: i32 = 0;
    var data: ?[*]const u8 = null;
    var len: usize = 0;
    var n_samples: i64 = 0;
    rc = api.mtmd_helper_gen_audio_get_output(gen, &sr, &data, &len, &n_samples);
    print("tts output rc {d} sr {d} len {d} samples {d} wav {x:0>16}\n", .{ rc, sr, len, n_samples, if (data) |d| hashBytes(d[0..len]) else 0 });
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
    print("cap vision {} audio {}\n", .{ cap.inp_vision, cap.inp_audio });
    const cap_missing = api.mtmd_get_cap_from_file("/nonexistent.gguf");
    print("cap missing vision {} audio {}\n", .{ cap_missing.inp_vision, cap_missing.inp_audio });

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
    const info = api.mtmd_gen_audio_get_info(mctx);
    const gi = api.mtmd_gen_inp_default(mctx);
    print("gen type {d} rate {d} variant {s} default top_k {d} top_p {x} temp {x} seed {d}\n", .{ @backingInt(info.type), info.sample_rate, if (info.model_variant) |v| std.mem.span(v) else "(null)", gi.top_k, @as(u32, @bitCast(gi.top_p)), @as(u32, @bitCast(gi.temp)), gi.seed });
    print("can_chat {}\n", .{api.mtmd_helper_model_can_chat(lctx, mctx)});
    flush();

    const ctx: Ctx = .{ .a = a, .api = api, .mctx = mctx, .model = model, .lctx = lctx };
    if (std.mem.startsWith(u8, modality, "tts")) {
        var steps: c_int = 3;
        if (std.mem.indexOf(u8, inputs, "tts:")) |i| steps = std.fmt.parseInt(c_int, std.mem.sliceTo(inputs[i + 4 ..], ';'), 10) catch 3;
        runTts(&ctx, std.mem.eql(u8, modality, "tts-qwen3"), steps);
    } else {
        try runMedia(&ctx, tmp, inputs, api.mtmd_support_vision(mctx), api.mtmd_support_audio(mctx));
    }
    print("done\n", .{});
    return 0;
}
