//! libmtmd's audio-generation helpers: the Qwen3-TTS, pocket-tts, Soprano and
//! LFM3-Audio pipelines that drive a backbone and the mmproj's generator
//! frame by frame.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/mtmd-helper-gen.cpp` and the gen-audio types
//! of `llama.cpp/tools/mtmd/mtmd-helper.h`, at v0.8.0 (`8642541b2`). Each
//! declaration names what it ports and the line it began at.
//!
//! The C++'s abstract `mtmd_gen_audio_pipeline` and its four subclasses are a
//! tagged union, `Pipeline`, dispatched where the C++ calls a virtual.

const std = @import("std");
const impl = @import("impl.zig");
const mtmd = @import("mtmd.zig");
const helper = @import("helper.zig");
const clip = @import("clip.zig");
const llama = @import("llama");
const libcxx = llama.libcxx;
const lapi = llama.api;
const c = impl.c;
const gpa = impl.gpa;

const Context = mtmd.Context;
const Token = c.llama_token;
const token_null: Token = -1; // LLAMA_TOKEN_NULL
const EmbdBatch = helper.EmbdBatch;
const logInf = helper.logInf;
const logWrn = helper.logWrn;
const logErr = helper.logErr;

extern fn exp(x: f64) f64;

// -----------------------------------------------------------------------------
// mtmd-helper.h

/// Ports `mtmd_helper_gen_audio_outtype` (mtmd-helper.h:223 @8642541b2).
pub const OutType = enum(c_int) {
    pcm, // raw PCM
    wav, // WAV PCM 16-bit LE, mono
    _,
};

/// Ports `mtmd_helper_gen_audio_out_mode` (mtmd-helper.h:228 @8642541b2): what a
/// response that continues a sequence (`n_past > 0`) outputs.
pub const OutMode = enum(c_int) {
    auto, // the model switches between text and audio itself
    text,
    audio,
    mixed, // interleaved text and audio
    _,
};

/// Ports `mtmd_helper_gen_audio_inp` (mtmd-helper.h:234 @8642541b2).
pub const Inp = extern struct {
    stream: bool, // if true, output() must be called after each step_gen()
    seq_id: c.llama_seq_id,

    // the text to speak; for a model with a "tts" chat template (e.g. LFM3-Audio), the prompt rendered from it
    prompt: ?[*]const u8,
    prompt_len: usize,

    // > 0: continue a sequence of n_past positions with raw text (LFM3-Audio only)
    n_past: i32,
    out_mode: OutMode, // n_past > 0 only (default: auto)

    speaker_ref: ?*mtmd.Bitmap, // optional, can be NULL
    lang: ?[*:0]const u8, // optional, can be NULL
    voice: ?[*:0]const u8, // optional, can be NULL, model-dependent
    top_k: i32,
    top_p: f32,
    seed: u32, // UINT32_MAX for random (default: random)
    out_type: OutType,
};

// -----------------------------------------------------------------------------
// shared

fn oom() noreturn {
    impl.abortUncaught("std::bad_alloc", "std::bad_alloc");
}

/// Ports `tts_lang_codes` and `tts_resolve_lang` (mtmd-helper-gen.cpp:27,
/// 40 @8642541b2): a `--tts-lang` code to the language name the
/// `codec_language` special tokens use, anything else unchanged.
fn ttsResolveLang(lang: []const u8) []const u8 {
    const codes = [_][2][]const u8{
        .{ "zh", "chinese" },  .{ "en", "english" },    .{ "de", "german" },
        .{ "it", "italian" },  .{ "pt", "portuguese" }, .{ "es", "spanish" },
        .{ "ja", "japanese" }, .{ "ko", "korean" },     .{ "fr", "french" },
        .{ "ru", "russian" },
    };
    for (codes) |e| if (std.mem.eql(u8, e[0], lang)) return e[1];
    return lang;
}

/// Ports `find_special_token` (mtmd-helper-gen.cpp:45 @8642541b2): the
/// first token whose text is `piece`.
fn findSpecialToken(vocab: ?*const c.struct_llama_vocab, piece: []const u8) Token {
    const n = lapi.vocab.llama_vocab_n_tokens(vocab);
    var t: Token = 0;
    while (t < n) : (t += 1) {
        const text = lapi.vocab.llama_vocab_get_text(vocab, t);
        if (std.mem.eql(u8, piece, std.mem.span(@as([*:0]const u8, @ptrCast(text))))) return t;
    }
    return token_null;
}

/// `std::min` and `std::max` as libc++ defines them, which decides where a
/// NaN goes.
fn stdMin(a: f32, b: f32) f32 {
    return if (b < a) b else a;
}
fn stdMax(a: f32, b: f32) f32 {
    return if (a < b) b else a;
}

/// The `buf.insert` of `put_bytes` (mtmd-helper-gen.cpp:55 @8642541b2).
fn putBytes(b: *std.ArrayList(u8), bytes: []const u8) void {
    b.appendSlice(gpa, bytes) catch oom();
}

/// Ports `write_wav16_header` (mtmd-helper-gen.cpp:61 @8642541b2): `data_sz ==
/// UINT32_MAX` writes the "unknown length" sentinel (streaming), as ffmpeg
/// does on a pipe.
fn writeWav16Header(buf: *std.ArrayList(u8), data_sz: u32, rate: i32) void {
    const riff_sz: u32 = if (data_sz == std.math.maxInt(u32)) std.math.maxInt(u32) else 36 +% data_sz;
    const fmt_sz: u32 = 16;
    const byte_rate: u32 = @as(u32, @bitCast(rate)) *% 2;
    const fmt: u16 = 1;
    const ch: u16 = 1;
    const alignment: u16 = 2;
    const bits: u16 = 16;
    const rate32: u32 = @bitCast(rate);
    putBytes(buf, "RIFF");
    putBytes(buf, std.mem.asBytes(&riff_sz));
    putBytes(buf, "WAVE");
    putBytes(buf, "fmt ");
    putBytes(buf, std.mem.asBytes(&fmt_sz));
    putBytes(buf, std.mem.asBytes(&fmt));
    putBytes(buf, std.mem.asBytes(&ch));
    putBytes(buf, std.mem.asBytes(&rate32));
    putBytes(buf, std.mem.asBytes(&byte_rate));
    putBytes(buf, std.mem.asBytes(&alignment));
    putBytes(buf, std.mem.asBytes(&bits));
    putBytes(buf, "data");
    putBytes(buf, std.mem.asBytes(&data_sz));
}

/// Ports `append_wav16_pcm` (mtmd-helper-gen.cpp:73 @8642541b2).
fn appendWav16Pcm(buf: *std.ArrayList(u8), pcm: []const f32) void {
    buf.ensureUnusedCapacity(gpa, pcm.len * 2) catch oom();
    for (pcm) |v| {
        const s: i16 = @intFromFloat(stdMax(-1.0, stdMin(1.0, v)) * 32767.0);
        buf.appendSliceAssumeCapacity(std.mem.asBytes(&s));
    }
}

/// Ports `write_wav16` (mtmd-helper-gen.cpp:80 @8642541b2): 16-bit mono WAV.
fn writeWav16(buf: *std.ArrayList(u8), pcm: []const f32, rate: i32) bool {
    // RIFF chunk sizes are 32-bit; refuse to emit a file with a truncated header
    if (pcm.len > (@as(usize, std.math.maxInt(u32)) - 36) / 2) return false;
    writeWav16Header(buf, @truncate(pcm.len * 2), rate);
    appendWav16Pcm(buf, pcm);
    return true;
}

/// What a streaming `get_output` remembers between calls.
const StreamOut = struct {
    stream: bool = false,
    pcm_sent: usize = 0, // samples already returned by get_output()
    wav_header_sent: bool = false,

    fn reset(self: *StreamOut) void {
        self.* = .{};
    }
};

/// Ports `mtmd_gen_audio_pipeline` (mtmd-helper-gen.cpp:90 @8642541b2): what
/// every pipeline holds.
const Base = struct {
    lctx: ?*c.struct_llama_context,
    mctx: *Context,
    model: ?*const c.struct_llama_model,
    vocab: ?*const c.struct_llama_vocab,
    n_embd: c_int,
    info: mtmd.GenAudioInfo,
    gen_state: ?*mtmd.GenState = null,
    logit_bias: std.ArrayList(c.llama_logit_bias) = .empty, // see mtmd_helper_gen_audio_get_logit_bias()

    fn init(lctx: ?*c.struct_llama_context, mctx: *Context) Base {
        const model = lapi.context.llama_get_model(lctx);
        return .{
            .lctx = lctx,
            .mctx = mctx,
            .model = model,
            .vocab = lapi.model.llama_model_get_vocab(model),
            .n_embd = lapi.model.llama_model_n_embd(model),
            .info = mtmd.mtmd_gen_audio_get_info(mctx),
        };
    }

    fn deinit(self: *Base) void {
        self.dropGenState();
        self.logit_bias.deinit(gpa);
    }

    /// `gen_state.reset()`.
    fn dropGenState(self: *Base) void {
        mtmd.mtmd_gen_state_free(self.gen_state);
        self.gen_state = null;
    }

    /// Ports `init_gen_state` (mtmd-helper-gen.cpp:116 @8642541b2): one state per
    /// generation, so a fixed seed always gives the same output.
    fn initGenState(self: *Base, seed: u32) bool {
        var params = mtmd.mtmd_gen_state_params_default();
        params.seed = seed;
        self.dropGenState();
        self.gen_state = mtmd.mtmd_gen_state_init(self.mctx, params);
        if (self.gen_state == null) {
            logErr("mtmd_helper_gen_audio: failed to create the gen state\n", .{});
            return false;
        }
        return true;
    }

    fn nE(self: *const Base) usize {
        return @intCast(self.n_embd);
    }

    /// The token embedding matrix, F32: the shared half of `ensure_cache` and
    /// `ensure_cache` (mtmd-helper-gen.cpp:395, 786 @8642541b2), one per pipeline.
    fn loadTokEmbd(self: *const Base, tok_embd: *std.ArrayList(f32)) bool {
        const n_tok_embd = lapi.model.llama_model_get_tok_embd(self.model, null);
        if (n_tok_embd == 0) {
            logErr("mtmd_helper_gen_audio: model has no token embeddings\n", .{});
            return false;
        }
        tok_embd.resize(gpa, n_tok_embd) catch oom();
        @memset(tok_embd.items, 0);
        if (lapi.model.llama_model_get_tok_embd(self.model, tok_embd.items.ptr) != n_tok_embd) {
            logErr("mtmd_helper_gen_audio: token embedding copy failed\n", .{});
            return false;
        }
        return true;
    }

    /// Ports `encode_speaker` and `encode_speaker` (mtmd-helper-gen.cpp:432,
    /// 1000 @8642541b2), one per pipeline: the reference audio through the mmproj's audio encoder.
    /// They differ only in their messages.
    fn encodeSpeaker(self: *const Base, bitmap: *mtmd.Bitmap, out: *std.ArrayList(f32), comptime no_encoder: [*:0]const u8, comptime encode_failed: [*:0]const u8) bool {
        if (!mtmd.mtmd_support_audio(self.mctx)) {
            logErr(no_encoder, .{});
            return false;
        }
        const marker = std.mem.span(mtmd.mtmd_get_marker(self.mctx));
        const text: mtmd.InputText = .{ .text = marker.ptr, .text_len = marker.len, .add_special = false, .parse_special = true };
        const chunks = mtmd.mtmd_input_chunks_init();
        const bptr: [1]*const mtmd.Bitmap = .{bitmap};
        var ok = mtmd.mtmd_tokenize(self.mctx, chunks, &text, &bptr, 1) == 0;
        if (ok) {
            ok = false;
            var i: usize = 0;
            while (i < mtmd.mtmd_input_chunks_size(chunks)) : (i += 1) {
                const chunk = mtmd.mtmd_input_chunks_get(chunks, i).?;
                if (mtmd.mtmd_input_chunk_get_type(chunk) != .audio) continue;
                if (mtmd.mtmd_encode_chunk(self.mctx, chunk) != 0) {
                    logErr(encode_failed, .{});
                    break;
                }
                const embd = mtmd.mtmd_get_output_embd(self.mctx);
                const n = @as(usize, @intCast(lapi.model.llama_model_n_embd_inp(self.model))) *% mtmd.mtmd_input_chunk_get_n_tokens(chunk);
                out.clearRetainingCapacity();
                if (n != 0) out.appendSlice(gpa, embd.?[0..n]) catch oom();
                ok = true;
                break;
            }
        }
        mtmd.mtmd_input_chunks_free(chunks);
        return ok;
    }

    /// Ports `step_prompt` and `step_prompt` (mtmd-helper-gen.cpp:256, 650
    /// @8642541b2), the same in both pipelines: at most `n_batch` tokens of the prompt batch.
    fn stepPrompt(self: *const Base, st: *PromptState, pos: *c_int, n_batch: i32) i32 {
        impl.ggmlAssert(n_batch > 0, "mtmd-helper-gen.cpp", 257, "n_batch > 0");
        if (st.prompt_pos >= st.n_prompt) return 0;
        const n_tokens_batch = @min(n_batch, st.n_prompt - st.prompt_pos);

        const is_last_batch = (st.prompt_pos + n_tokens_batch) == st.n_prompt;
        if (is_last_batch) st.prompt_batch.?.logits.items[@intCast(st.prompt_pos + n_tokens_batch - 1)] = 1;

        if (lapi.context.llama_process(self.lctx, c.LLAMA_PROCESS_TYPE_DECODE, st.prompt_batch.?.render(self.lctx, st.prompt_pos, n_tokens_batch)) != 0) {
            logErr("mtmd_helper_gen_audio: prompt decode failed\n", .{});
            return -1;
        }

        pos.* += n_tokens_batch;
        st.prompt_pos += n_tokens_batch;

        if (st.prompt_pos >= st.n_prompt) {
            // prompt fully processed, its embedding buffer is no longer needed
            st.dropBatch();
            st.prompt_embd_buf.clearRetainingCapacity();
            return 0;
        }
        return st.n_prompt - st.prompt_pos;
    }

    /// Ports `get_output`, `get_output` and `get_output` (mtmd-helper-gen.cpp:105,
    /// 339, 730 @8642541b2), one per pipeline but soprano's, after their flush:
    /// everything, or with `so.stream` only what is new since the last call.
    fn getOutput(self: *const Base, audio_pcm: *const std.ArrayList(f32), out_type: OutType, out_buf: *std.ArrayList(u8), so: *StreamOut, out_data: *?[*]const u8, out_data_len: *usize, out_n_samples: ?*i64) i32 {
        if (!so.stream) {
            if (out_n_samples) |p| p.* = @intCast(audio_pcm.items.len);
            if (out_type == .pcm) {
                out_data.* = if (audio_pcm.capacity == 0) null else @ptrCast(audio_pcm.items.ptr);
                out_data_len.* = audio_pcm.items.len * @sizeOf(f32);
                return 0;
            }
            out_buf.clearRetainingCapacity();
            if (!writeWav16(out_buf, audio_pcm.items, self.info.sample_rate)) {
                logErr("mtmd_helper_gen_audio: output too large for WAV\n", .{});
                return 1;
            }
            out_data.* = if (out_buf.capacity == 0) null else out_buf.items.ptr;
            out_data_len.* = out_buf.items.len;
            return 0;
        }

        // streaming: only return audio produced since the previous call
        const n_new = audio_pcm.items.len - so.pcm_sent;
        if (out_n_samples) |p| p.* = @intCast(n_new);

        if (out_type == .pcm) {
            out_data.* = if (audio_pcm.capacity == 0) null else @ptrCast(audio_pcm.items.ptr + so.pcm_sent);
            out_data_len.* = n_new * @sizeOf(f32);
            so.pcm_sent = audio_pcm.items.len;
            return 0;
        }

        out_buf.clearRetainingCapacity();
        if (!so.wav_header_sent) {
            writeWav16Header(out_buf, std.math.maxInt(u32), self.info.sample_rate);
            so.wav_header_sent = true;
        }
        appendWav16Pcm(out_buf, audio_pcm.items[so.pcm_sent..]);
        so.pcm_sent = audio_pcm.items.len;
        out_data.* = if (out_buf.capacity == 0) null else out_buf.items.ptr;
        out_data_len.* = out_buf.items.len;
        return 0;
    }

    /// `h_state_buf.assign(he, he + n_embd)` from the last output row.
    fn takeHState(self: *const Base, h_state_buf: *std.ArrayList(f32)) [*]const f32 {
        const he = lapi.context.llama_get_embeddings_ith(self.lctx, -1);
        h_state_buf.clearRetainingCapacity();
        h_state_buf.appendSlice(gpa, he.?[0..self.nE()]) catch oom();
        return h_state_buf.items.ptr;
    }
};

/// The prompt-decode state both pipelines keep.
const PromptState = struct {
    prompt_embd_buf: std.ArrayList(f32) = .empty,
    prompt_batch: ?*EmbdBatch = null,
    n_prompt: c_int = 0,
    prompt_pos: c_int = 0,

    fn dropBatch(self: *PromptState) void {
        if (self.prompt_batch) |b| b.destroy();
        self.prompt_batch = null;
    }

    fn reset(self: *PromptState) void {
        self.prompt_embd_buf.clearRetainingCapacity();
        self.dropBatch();
        self.n_prompt = 0;
        self.prompt_pos = 0;
    }

    fn deinit(self: *PromptState) void {
        self.dropBatch();
        self.prompt_embd_buf.deinit(gpa);
    }
};

/// `vec.assign(p, p + n)`.
fn assign(comptime T: type, list: *std.ArrayList(T), p: ?[*]const T, n: usize) void {
    list.clearRetainingCapacity();
    if (n != 0) list.appendSlice(gpa, p.?[0..n]) catch oom();
}

// -----------------------------------------------------------------------------
// Qwen3-TTS

/// Ports `qwen3tts_gen_audio_pipeline` (mtmd-helper-gen.cpp:138 @8642541b2):
/// the backbone samples codec 0, the code predictor gives the other 15
/// codebooks, then code2wav decodes them to PCM.
const Qwen3tts = struct {
    base: Base,

    // vocab specials fixed across the whole session, looked up once
    specials_ok: bool = false,
    codec_0: Token = token_null,
    codec_bos: Token = token_null,
    codec_eos: Token = token_null,
    codec_pad: Token = token_null,
    c_think: Token = token_null,
    c_think_b: Token = token_null,
    c_think_e: Token = token_null,
    tts_pad: Token = token_null,
    tts_bos: Token = token_null,
    tts_eos: Token = token_null,
    tok_embd: std.ArrayList(f32) = .empty, // whole token embedding matrix, n_vocab * n_embd

    // must match hparams.wav_tfm_swa hardcoded in clip.cpp
    window_frames: usize = 72,

    // per-generation state, cleared by reset()
    seq_id: c.llama_seq_id = 0,
    mrope: bool = false,
    pos: c_int = 0,
    prompt: PromptState = .{},
    top_k: i32 = 50,
    top_p: f32 = 1.0,
    codes_buf: std.ArrayList(i32) = .empty,
    audio_pcm: std.ArrayList(f32) = .empty,
    overlay: std.ArrayList(f32) = .empty,
    h_state_buf: std.ArrayList(f32) = .empty,
    out_type: OutType = .wav,
    out_buf: std.ArrayList(u8) = .empty,
    so: StreamOut = .{},

    fn deinit(self: *Qwen3tts) void {
        self.base.deinit();
        self.tok_embd.deinit(gpa);
        self.prompt.deinit();
        self.codes_buf.deinit(gpa);
        self.audio_pcm.deinit(gpa);
        self.overlay.deinit(gpa);
        self.h_state_buf.deinit(gpa);
        self.out_buf.deinit(gpa);
    }

    /// `reset` (mtmd-helper-gen.cpp:142 @8642541b2).
    fn reset(self: *Qwen3tts) void {
        self.seq_id = 0;
        self.pos = 0;
        self.codes_buf.clearRetainingCapacity();
        self.base.dropGenState();
        self.audio_pcm.clearRetainingCapacity();
        self.overlay.clearRetainingCapacity();
        self.h_state_buf.clearRetainingCapacity();
        self.out_buf.clearRetainingCapacity();
        self.prompt.reset();
        self.so.reset();
    }

    fn row(self: *const Qwen3tts, t: Token) []const f32 {
        const n_e = self.base.nE();
        const k: usize = @intCast(t);
        return self.tok_embd.items[k * n_e .. (k + 1) * n_e];
    }

    /// `sum_row` and `sum_vec` (mtmd-helper-gen.cpp:187, 192 @8642541b2),
    /// appended to `dst`.
    fn pushSum(self: *const Qwen3tts, dst: *std.ArrayList(f32), a: Token, vb: []const f32) void {
        const va = self.row(a);
        dst.ensureUnusedCapacity(gpa, va.len) catch oom();
        for (va, 0..) |x, i| dst.appendAssumeCapacity(x + vb[i]);
    }

    /// `set_input` (mtmd-helper-gen.cpp:160 @8642541b2).
    fn setInput(self: *Qwen3tts, inp: *const Inp) i32 {
        self.reset();
        self.seq_id = inp.seq_id;

        if (!self.ensureCache()) return 1;

        const lang_in: []const u8 = if (inp.lang != null and inp.lang.?[0] != 0) std.mem.span(inp.lang.?) else "english";
        const lang = ttsResolveLang(lang_in);
        const piece = std.mem.concat(gpa, u8, &.{ "<|codec_language_", lang, "|>" }) catch oom();
        defer gpa.free(piece);
        const c_lang = findSpecialToken(self.base.vocab, piece);
        if (c_lang == token_null) {
            const lang_z = gpa.dupeSentinel(u8, lang, 0) catch oom();
            defer gpa.free(lang_z);
            logErr("mtmd_helper_gen_audio: unknown language '%s'\n", .{lang_z.ptr});
            return 1;
        }

        var speaker_embd: std.ArrayList(f32) = .empty;
        defer speaker_embd.deinit(gpa);
        if (inp.speaker_ref) |sr| {
            if (!self.base.encodeSpeaker(sr, &speaker_embd, "mtmd_helper_gen_audio: mmproj has no speaker/audio encoder\n", "mtmd_helper_gen_audio: speaker encode failed\n")) return 1;
        }

        const n_e = self.base.nE();

        // upstream chat wrap, then slices: [0:3] role, [3:-5] utterance body
        const prompt_text: []const u8 = if (inp.prompt_len == 0) "" else inp.prompt.?[0..inp.prompt_len];
        const full = std.mem.concat(gpa, u8, &.{ "<|im_start|>assistant\n", prompt_text, "<|im_end|>\n<|im_start|>assistant\n" }) catch oom();
        defer gpa.free(full);
        const ids = gpa.alloc(Token, full.len + 16) catch oom();
        defer gpa.free(ids);
        @memset(ids, 0);
        const n_ids = lapi.vocab.llama_tokenize(self.base.vocab, full.ptr, @truncate(@as(isize, @intCast(full.len))), ids.ptr, @truncate(@as(isize, @intCast(ids.len))), false, true);
        if (n_ids < 8) {
            logErr("mtmd_helper_gen_audio: tokenization failed\n", .{});
            return 1;
        }

        // the prompt rows, flattened straight into the embedding buffer
        const buf = &self.prompt.prompt_embd_buf;
        buf.clearRetainingCapacity();
        var n_rows: c_int = 0;
        for (0..3) |i| {
            buf.appendSlice(gpa, self.row(ids[i])) catch oom();
            n_rows += 1;
        }
        for ([_]Token{ self.c_think, self.c_think_b, c_lang, self.c_think_e }) |t| {
            self.pushSum(buf, self.tts_pad, self.row(t));
            n_rows += 1;
        }
        if (speaker_embd.items.len != 0) {
            self.pushSum(buf, self.tts_pad, speaker_embd.items);
            n_rows += 1;
        }
        self.pushSum(buf, self.tts_bos, self.row(self.codec_pad));
        n_rows += 1;
        var i: c_int = 3;
        while (i < n_ids - 5) : (i += 1) {
            self.pushSum(buf, ids[@intCast(i)], self.row(self.codec_pad));
            n_rows += 1;
        }
        self.pushSum(buf, self.tts_eos, self.row(self.codec_pad));
        self.pushSum(buf, self.tts_pad, self.row(self.codec_bos));
        n_rows += 2;

        self.prompt.n_prompt = n_rows;
        std.debug.assert(buf.items.len == @as(usize, @intCast(n_rows)) * n_e);

        // the talker uses the qwen3vl interleaved mrope, all sections are equal for a text/codec stream
        const rope = lapi.model.llama_model_rope_type(self.base.model);
        self.mrope = rope == c.LLAMA_ROPE_TYPE_MROPE or rope == c.LLAMA_ROPE_TYPE_IMROPE;
        const n_pos_per_embd: c_int = if (self.mrope) 4 else 1;

        self.prompt.dropBatch();
        self.prompt.prompt_batch = EmbdBatch.create(buf.items.ptr, n_rows, n_pos_per_embd, self.base.n_embd);
        if (self.mrope) self.prompt.prompt_batch.?.setPositionMrope1d(0, self.seq_id) else self.prompt.prompt_batch.?.setPositionNormal(0, self.seq_id);
        self.prompt.prompt_pos = 0;

        self.pos = 0;
        const def = mtmd.mtmd_gen_inp_default(self.base.mctx);
        self.top_k = if (inp.top_k > 0) inp.top_k else def.top_k;
        self.top_p = if (inp.top_p > 0) inp.top_p else def.top_p;
        self.out_type = inp.out_type;
        self.so.stream = inp.stream;
        if (!self.base.initGenState(inp.seed)) return 1;

        // the prompt above holds the whole text stream up to tts_eos, so every generated
        // frame adds tts_pad on top of the codes embedding
        assign(f32, &self.overlay, self.row(self.tts_pad).ptr, n_e);

        return 0;
    }

    /// `step_gen` (mtmd-helper-gen.cpp:285 @8642541b2).
    fn stepGen(self: *Qwen3tts, sampled: Token, h_state_in: ?[*]const f32, h_state_out: *?[*]const f32, out_stop: *bool) i32 {
        if (sampled == token_null) {
            logErr("mtmd_helper_gen_audio: qwen3tts requires a token sampled from the backbone\n", .{});
            return 1;
        }

        // backbone signals end-of-speech with a token, no frame for this step
        if (sampled == self.codec_eos or lapi.vocab.llama_vocab_is_eog(self.base.vocab, sampled)) {
            out_stop.* = true;
            h_state_out.* = null;
            return 0;
        }

        var inp = mtmd.mtmd_gen_inp_default(self.base.mctx);
        inp.type = .gen_code;
        inp.code0 = sampled -% self.codec_0;
        inp.embd = @constCast(h_state_in);
        inp.top_k = self.top_k;
        inp.top_p = self.top_p;
        var out = std.mem.zeroes(mtmd.GenOut);
        if (mtmd.mtmd_gen_audio_process(self.base.mctx, self.base.gen_state, &inp, &out) != 0) {
            logErr("mtmd_helper_gen_audio: gen_code process failed\n", .{});
            return 1;
        }

        if (out.n_codes != 0) self.codes_buf.appendSlice(gpa, out.codes.?[0..out.n_codes]) catch oom();
        if (out.n_codes > 0 and self.codes_buf.items.len / out.n_codes >= self.window_frames) {
            if (!self.flushGenWav()) return 1;
        }

        const n_e = self.base.nE();
        const fb = gpa.alloc(f32, n_e) catch oom();
        defer gpa.free(fb);
        @memcpy(fb, out.embd.?[0..n_e]);
        for (fb, 0..) |*x, k| x.* += self.overlay.items[k];

        const n_pos_per_embd: c_int = if (self.mrope) 4 else 1;
        const batch_embd = EmbdBatch.create(fb.ptr, 1, n_pos_per_embd, self.base.n_embd);
        defer batch_embd.destroy();
        if (self.mrope) batch_embd.setPositionMrope1d(self.pos, self.seq_id) else batch_embd.setPositionNormal(self.pos, self.seq_id);
        batch_embd.logits.items[0] = 1;
        self.pos += 1;

        if (lapi.context.llama_process(self.base.lctx, c.LLAMA_PROCESS_TYPE_DECODE, batch_embd.render(self.base.lctx, 0, 1)) != 0) {
            logErr("mtmd_helper_gen_audio: decode failed\n", .{});
            return 1;
        }

        h_state_out.* = self.base.takeHState(&self.h_state_buf);
        return 0;
    }

    /// `get_output` (mtmd-helper-gen.cpp:105 @8642541b2).
    fn getOutput(self: *Qwen3tts, out_sample_rate: *i32, out_data: *?[*]const u8, out_data_len: *usize, out_n_samples: ?*i64) i32 {
        out_sample_rate.* = self.base.info.sample_rate;
        // one-shot call: force out whatever's left, regardless of window_frames
        if (!self.so.stream and !self.flushGenWav()) return 1;
        return self.base.getOutput(&self.audio_pcm, self.out_type, &self.out_buf, &self.so, out_data, out_data_len, out_n_samples);
    }

    /// `flush` (mtmd-helper-gen.cpp:390 @8642541b2).
    fn flush(self: *Qwen3tts) i32 {
        return if (self.flushGenWav()) 0 else 1;
    }

    /// `ensure_cache` (mtmd-helper-gen.cpp:395 @8642541b2).
    fn ensureCache(self: *Qwen3tts) bool {
        if (self.specials_ok) return true;
        const v = self.base.vocab;
        self.codec_0 = findSpecialToken(v, "<|codec_0|>");
        self.codec_bos = findSpecialToken(v, "<|codec_bos|>");
        self.codec_eos = findSpecialToken(v, "<|codec_eos_token|>");
        self.codec_pad = findSpecialToken(v, "<|codec_pad|>");
        self.c_think = findSpecialToken(v, "<|codec_think|>");
        self.c_think_b = findSpecialToken(v, "<|codec_think_bos|>");
        self.c_think_e = findSpecialToken(v, "<|codec_think_eos|>");
        self.tts_pad = findSpecialToken(v, "<tts_pad>");
        self.tts_bos = findSpecialToken(v, "<tts_text_bos>");
        self.tts_eos = findSpecialToken(v, "<tts_text_eod>");
        for ([_]Token{ self.codec_0, self.codec_bos, self.codec_eos, self.codec_pad, self.c_think, self.c_think_b, self.c_think_e, self.tts_pad, self.tts_bos, self.tts_eos }) |t| {
            if (t == token_null) {
                logErr("mtmd_helper_gen_audio: missing a required special token in vocab\n", .{});
                return false;
            }
        }
        if (!self.base.loadTokEmbd(&self.tok_embd)) return false;
        self.specials_ok = true;
        return true;
    }

    /// `flush_gen_wav` (mtmd-helper-gen.cpp:465 @8642541b2): one GEN_WAV
    /// call over the buffered codes, state carried across batches.
    fn flushGenWav(self: *Qwen3tts) bool {
        if (self.codes_buf.items.len == 0) return true;
        var inp = mtmd.mtmd_gen_inp_default(self.base.mctx);
        inp.type = .gen_wav;
        inp.codes = self.codes_buf.items.ptr;
        inp.n_codes = self.codes_buf.items.len;
        var out = std.mem.zeroes(mtmd.GenOut);
        if (mtmd.mtmd_gen_audio_process(self.base.mctx, self.base.gen_state, &inp, &out) != 0) {
            logErr("mtmd_helper_gen_audio: gen_wav process failed\n", .{});
            return false;
        }
        if (out.n_samples != 0) self.audio_pcm.appendSlice(gpa, out.audio.?[0..out.n_samples]) catch oom();
        self.codes_buf.clearRetainingCapacity();
        return true;
    }
};

// -----------------------------------------------------------------------------
// pocket-tts

/// Ports `pockettts_pack_settings` (mtmd-helper-gen.cpp:525 @8642541b2):
/// settings that live only in the reference's per-pack yaml.
const PackSettings = struct {
    temp: f32 = 0.0,
    frames_after_eos: c_int = 0,
    pad_short_text: bool = false,
};

/// Ports `pockettts_pack` (mtmd-helper-gen.cpp:531 @8642541b2).
fn pocketttsPack(variant: ?[*:0]const u8) PackSettings {
    const packs = [_]struct { []const u8, PackSettings }{
        .{ "english", .{ .temp = 0.3, .frames_after_eos = 0, .pad_short_text = false } },
        .{ "english_2026-01", .{ .temp = 0.7, .frames_after_eos = 0, .pad_short_text = true } },
        .{ "english_2026-04", .{ .temp = 0.3, .frames_after_eos = 0, .pad_short_text = false } },
        .{ "french_24l", .{ .temp = 0.7, .frames_after_eos = 8, .pad_short_text = false } },
    };
    const v: [*:0]const u8 = variant orelse "";
    for (packs) |p| if (std.mem.eql(u8, p[0], std.mem.span(v))) return p[1];
    logWrn("mtmd_helper_gen_audio: no tuned settings for pocket-tts variant \"%s\"\n", .{v});
    return .{};
}

const TokenList = std.ArrayList(Token);

fn freeLists(lists: *std.ArrayList(TokenList)) void {
    for (lists.items) |*l| l.deinit(gpa);
    lists.deinit(gpa);
}

/// Ports `pockettts_gen_audio_pipeline` (mtmd-helper-gen.cpp:549
/// @8642541b2): the backbone emits no token; the flow net turns each hidden
/// state into a latent, and the end-of-speech head lives in the mmproj.
const Pockettts = struct {
    base: Base,

    pack: PackSettings = .{},
    specials_ok: bool = false,
    bos_before_voice: Token = token_null,
    audio_bos: Token = token_null,
    tok_embd: std.ArrayList(f32) = .empty,

    seq_id: c.llama_seq_id = 0,
    pos: c_int = 0,
    prompt: PromptState = .{},
    seed: u32 = std.math.maxInt(u32),
    // end-of-speech is latched, then a few more frames are generated as tail padding
    step_idx: c_int = 0,
    eos_step: c_int = -1,
    frames_after_eos: c_int = 3,
    chunks: std.ArrayList(TokenList) = .empty,
    chunk_idx: usize = 0,
    n_voice_pos: c_int = 0, // KV positions held by the voice conditioning
    chunk_budget: c_int = 0,

    // latents are decoded a window at a time, the decoder state bridges the windows
    window_frames: usize = 8,
    feats_buf: std.ArrayList(f32) = .empty,
    audio_pcm: std.ArrayList(f32) = .empty,
    h_state_buf: std.ArrayList(f32) = .empty,
    out_type: OutType = .wav,
    out_buf: std.ArrayList(u8) = .empty,
    so: StreamOut = .{},

    const max_chunk_tokens: c_int = 50; // MAX_TOKEN_PER_CHUNK in the reference
    const frame_rate: f64 = 12.5;

    fn deinit(self: *Pockettts) void {
        self.base.deinit();
        self.tok_embd.deinit(gpa);
        self.prompt.deinit();
        freeLists(&self.chunks);
        self.feats_buf.deinit(gpa);
        self.audio_pcm.deinit(gpa);
        self.h_state_buf.deinit(gpa);
        self.out_buf.deinit(gpa);
    }

    /// `reset` (mtmd-helper-gen.cpp:553 @8642541b2).
    fn reset(self: *Pockettts) void {
        self.seq_id = 0;
        self.pos = 0;
        self.feats_buf.clearRetainingCapacity();
        self.base.dropGenState();
        self.audio_pcm.clearRetainingCapacity();
        self.h_state_buf.clearRetainingCapacity();
        self.out_buf.clearRetainingCapacity();
        self.prompt.reset();
        self.step_idx = 0;
        self.eos_step = -1;
        for (self.chunks.items) |*l| l.deinit(gpa);
        self.chunks.clearRetainingCapacity();
        self.chunk_idx = 0;
        self.n_voice_pos = 0;
        self.chunk_budget = 0;
        self.so.reset();
    }

    /// `set_input` (mtmd-helper-gen.cpp:576 @8642541b2).
    fn setInput(self: *Pockettts, inp: *const Inp) i32 {
        self.reset();
        self.seq_id = inp.seq_id;

        if (!self.ensureCache()) return 1;

        var voice: std.ArrayList(f32) = .empty;
        defer voice.deinit(gpa);
        if (inp.speaker_ref) |sr| {
            if (!self.base.encodeSpeaker(sr, &voice, "mtmd_helper_gen_audio: mmproj has no voice encoder\n", "mtmd_helper_gen_audio: voice encode failed\n")) return 1;
        }

        self.pack = pocketttsPack(self.base.info.model_variant);

        const prompt_text: []const u8 = if (inp.prompt_len == 0) "" else inp.prompt.?[0..inp.prompt_len];
        const text = prepareText(prompt_text, self.pack.pad_short_text);
        defer gpa.free(text);
        if (text.len == 0) {
            logErr("mtmd_helper_gen_audio: empty prompt\n", .{});
            return 1;
        }

        var ids: TokenList = .empty;
        defer ids.deinit(gpa);
        ids.appendNTimes(gpa, 0, text.len + 16) catch oom();
        const n_ids = lapi.vocab.llama_tokenize(self.base.vocab, text.ptr, @truncate(@as(isize, @intCast(text.len))), ids.items.ptr, @truncate(@as(isize, @intCast(ids.items.len))), false, false);
        if (n_ids <= 0) {
            logErr("mtmd_helper_gen_audio: tokenization failed\n", .{});
            return 1;
        }
        ids.shrinkRetainingCapacity(@intCast(n_ids));

        // long inputs degrade badly, so each chunk restarts from the voice conditioning
        // see split_into_best_sentences() in the reference
        freeLists(&self.chunks);
        self.chunks = self.splitChunks(ids.items);
        self.chunk_idx = 0;
        if (self.chunks.items.len > 1) {
            logInf("mtmd_helper_gen_audio: %d tokens split into %zu chunks\n", .{ n_ids, self.chunks.items.len });
        }

        const n_e = self.base.nE();
        const buf = &self.prompt.prompt_embd_buf;

        // sequence order is voice, then text, then the audio BOS that starts generation
        if (voice.items.len != 0) {
            impl.ggmlAssert(voice.items.len % n_e == 0, "mtmd-helper-gen.cpp", 621, "voice.size() % (size_t) n_e == 0");
            if (self.bos_before_voice != token_null) self.pushEmbdRow(buf, self.bos_before_voice);
            buf.appendSlice(gpa, voice.items) catch oom();
        }
        // every later chunk rewinds to here and re-prompts, so the voice stays primed
        self.n_voice_pos = @intCast(buf.items.len / n_e);

        for (self.chunks.items[0].items) |t| self.pushEmbdRow(buf, t);
        self.pushEmbdRow(buf, self.audio_bos);
        self.armChunkBudget(0);

        self.prompt.n_prompt = @intCast(buf.items.len / n_e);
        self.prompt.dropBatch();
        self.prompt.prompt_batch = EmbdBatch.create(buf.items.ptr, self.prompt.n_prompt, 1, self.base.n_embd);
        self.prompt.prompt_batch.?.setPositionNormal(0, self.seq_id);
        self.prompt.prompt_pos = 0;

        self.out_type = inp.out_type;
        self.so.stream = inp.stream;
        if (!self.base.initGenState(inp.seed)) return 1;

        return 0;
    }

    /// `step_gen` (mtmd-helper-gen.cpp:677 @8642541b2). `sampled` is unused:
    /// the backbone output is continuous.
    fn stepGen(self: *Pockettts, h_state_in: ?[*]const f32, h_state_out: *?[*]const f32, out_stop: *bool) i32 {
        var inp = mtmd.mtmd_gen_inp_default(self.base.mctx);
        inp.type = .gen_code;
        inp.embd = @constCast(h_state_in);
        if (self.pack.temp > 0.0) inp.temp = self.pack.temp;
        var out = std.mem.zeroes(mtmd.GenOut);
        if (mtmd.mtmd_gen_audio_process(self.base.mctx, self.base.gen_state, &inp, &out) != 0) {
            logErr("mtmd_helper_gen_audio: flow decode failed\n", .{});
            return 1;
        }
        if (out.is_eos and self.eos_step < 0) self.eos_step = self.step_idx;
        // the frame of the stopping step is discarded, matching _autoregressive_generation().
        // the budget is the reference's fallback for a chunk whose eos head never fires
        const chunk_done = (self.eos_step >= 0 and self.step_idx >= self.eos_step + self.frames_after_eos) or
            self.step_idx >= self.chunk_budget;
        if (chunk_done) {
            if (self.eos_step < 0) {
                logWrn("mtmd_helper_gen_audio: chunk %zu hit its budget without end-of-speech\n", .{self.chunk_idx});
            }
            return self.finishChunk(h_state_out, out_stop);
        }

        if (out.n_feats != 0) self.feats_buf.appendSlice(gpa, out.feats.?[0..out.n_feats]) catch oom();
        self.step_idx += 1;
        if (out.n_feats > 0 and self.feats_buf.items.len / out.n_feats >= self.window_frames) {
            if (!self.flushGenWav()) return 1;
        }

        const batch_embd = EmbdBatch.create(out.embd, 1, 1, self.base.n_embd);
        defer batch_embd.destroy();
        batch_embd.setPositionNormal(self.pos, self.seq_id);
        batch_embd.logits.items[0] = 1;
        self.pos += 1;

        if (lapi.context.llama_process(self.base.lctx, c.LLAMA_PROCESS_TYPE_DECODE, batch_embd.render(self.base.lctx, 0, 1)) != 0) {
            logErr("mtmd_helper_gen_audio: decode failed\n", .{});
            return 1;
        }

        h_state_out.* = self.base.takeHState(&self.h_state_buf);
        return 0;
    }

    /// `get_output` (mtmd-helper-gen.cpp:339 @8642541b2).
    fn getOutput(self: *Pockettts, out_sample_rate: *i32, out_data: *?[*]const u8, out_data_len: *usize, out_n_samples: ?*i64) i32 {
        out_sample_rate.* = self.base.info.sample_rate;
        // one-shot call: force out whatever's left, regardless of window_frames
        if (!self.so.stream and !self.flushGenWav()) return 1;
        return self.base.getOutput(&self.audio_pcm, self.out_type, &self.out_buf, &self.so, out_data, out_data_len, out_n_samples);
    }

    /// `flush` (mtmd-helper-gen.cpp:781 @8642541b2).
    fn flush(self: *Pockettts) i32 {
        return if (self.flushGenWav()) 0 else 1;
    }

    /// `ensure_cache` (mtmd-helper-gen.cpp:786 @8642541b2).
    fn ensureCache(self: *Pockettts) bool {
        if (self.specials_ok) return true;
        // bos_before_voice is optional, some packs do not insert it
        self.bos_before_voice = findSpecialToken(self.base.vocab, "<|bos_before_voice|>");
        self.audio_bos = findSpecialToken(self.base.vocab, "<|audio_bos|>");
        if (self.audio_bos == token_null) {
            logErr("mtmd_helper_gen_audio: missing <|audio_bos|> in vocab\n", .{});
            return false;
        }
        if (!self.base.loadTokEmbd(&self.tok_embd)) return false;
        const n_tok_embd: u32 = @intCast(self.tok_embd.items.len);
        impl.ggmlAssert(self.base.n_embd > 0 and n_tok_embd % @as(u32, @bitCast(self.base.n_embd)) == 0, "mtmd-helper-gen.cpp", 807, "n_embd > 0 && n_tok_embd % (uint32_t) n_embd == 0");
        self.specials_ok = true;
        return true;
    }

    /// `push_embd_row` (mtmd-helper-gen.cpp:813 @8642541b2): the table can
    /// be shorter than the vocab, so the row lookup is bounded.
    fn pushEmbdRow(self: *const Pockettts, dst: *std.ArrayList(f32), t: Token) void {
        const n_e = self.base.nE();
        const n_rows = self.tok_embd.items.len / n_e;
        impl.ggmlAssert(t >= 0 and @as(usize, @intCast(t)) < n_rows, "mtmd-helper-gen.cpp", 815, "t >= 0 && (size_t) t < n_rows");
        const k: usize = @intCast(t);
        dst.appendSlice(gpa, self.tok_embd.items[k * n_e .. (k + 1) * n_e]) catch oom();
    }

    /// `punct_ids` (mtmd-helper-gen.cpp:823 @8642541b2): the token ids of
    /// the pieces the reference splits on, without the dummy prefix.
    fn punctIds(self: *const Pockettts, s: [:0]const u8) TokenList {
        var ids: [16]Token = @splat(0);
        const n = lapi.vocab.llama_tokenize(self.base.vocab, s.ptr, @intCast(s.len), &ids, ids.len, false, false);
        var out: TokenList = .empty;
        if (n <= 1) return out;
        out.appendSlice(gpa, ids[1..@intCast(n)]) catch oom();
        return out;
    }

    /// `split_on` (mtmd-helper-gen.cpp:833 @8642541b2): cut after runs of
    /// boundary tokens, so punctuation stays with the sentence it ends.
    fn splitOn(ids: []const Token, boundary: []const Token) std.ArrayList(TokenList) {
        var out: std.ArrayList(TokenList) = .empty;
        var start: usize = 0;
        var prev_was_boundary = false;
        for (ids, 0..) |id, i| {
            const is_boundary = std.mem.indexOfScalar(Token, boundary, id) != null;
            if (!is_boundary and prev_was_boundary) {
                var seg: TokenList = .empty;
                seg.appendSlice(gpa, ids[start..i]) catch oom();
                out.append(gpa, seg) catch oom();
                start = i;
            }
            prev_was_boundary = is_boundary;
        }
        var last: TokenList = .empty;
        last.appendSlice(gpa, ids[start..]) catch oom();
        out.append(gpa, last) catch oom();
        return out;
    }

    /// `split_chunks` (mtmd-helper-gen.cpp:850 @8642541b2).
    fn splitChunks(self: *const Pockettts, ids: []const Token) std.ArrayList(TokenList) {
        var out: std.ArrayList(TokenList) = .empty;
        if (@as(c_int, @intCast(ids.len)) <= max_chunk_tokens) {
            var one: TokenList = .empty;
            one.appendSlice(gpa, ids) catch oom();
            out.append(gpa, one) catch oom();
            return out;
        }
        var eos_punct = self.punctIds(".!...?");
        defer eos_punct.deinit(gpa);
        var mid_punct = self.punctIds(",;:");
        defer mid_punct.deinit(gpa);

        // oversized sentences are split again on weaker punctuation, else words get skipped
        var segments: std.ArrayList(TokenList) = .empty;
        defer freeLists(&segments);
        var by_eos = splitOn(ids, eos_punct.items);
        defer by_eos.deinit(gpa); // its lists move into segments
        for (by_eos.items) |seg| {
            if (@as(c_int, @intCast(seg.items.len)) <= max_chunk_tokens) {
                segments.append(gpa, seg) catch oom();
                continue;
            }
            var sub = splitOn(seg.items, mid_punct.items);
            defer sub.deinit(gpa);
            if (sub.items.len > 1) {
                var moved = seg;
                moved.deinit(gpa);
                segments.appendSlice(gpa, sub.items) catch oom();
            } else {
                for (sub.items) |*s| s.deinit(gpa);
                segments.append(gpa, seg) catch oom();
            }
        }

        for (segments.items) |*seg| {
            if (seg.items.len == 0) continue;
            if (out.items.len != 0 and @as(c_int, @intCast(out.items[out.items.len - 1].items.len + seg.items.len)) <= max_chunk_tokens) {
                out.items[out.items.len - 1].appendSlice(gpa, seg.items) catch oom();
            } else {
                out.append(gpa, seg.*) catch oom();
                seg.* = .empty; // moved
            }
        }
        if (out.items.len == 0) {
            var one: TokenList = .empty;
            one.appendSlice(gpa, ids) catch oom();
            out.append(gpa, one) catch oom();
        }
        for (out.items) |ch| {
            if (@as(c_int, @intCast(ch.items.len)) > max_chunk_tokens) {
                logWrn("mtmd_helper_gen_audio: chunk of %zu tokens exceeds the %d token budget, " ++
                    "generation may skip words\n", .{ ch.items.len, max_chunk_tokens });
            }
        }
        return out;
    }

    /// `arm_chunk_budget` (mtmd-helper-gen.cpp:898 @8642541b2):
    /// `_estimate_max_gen_len()` plus the per-chunk tail guess, in frames.
    fn armChunkBudget(self: *Pockettts, idx: usize) void {
        const n_tok: c_int = @intCast(self.chunks.items[idx].items.len);
        self.chunk_budget = @intFromFloat(@ceil((@as(f64, @floatFromInt(n_tok)) / 3.0 + 2.0) * frame_rate));
        // the pack may pin the tail, else the reference guesses it from the word count
        self.frames_after_eos = if (self.pack.frames_after_eos > 0) self.pack.frames_after_eos else if (n_tok <= 6) 5 else 3;
        self.step_idx = 0;
        self.eos_step = -1;
    }

    /// `finish_chunk` (mtmd-helper-gen.cpp:908 @8642541b2): ends the chunk
    /// and re-prompts the next on top of the voice.
    fn finishChunk(self: *Pockettts, h_state_out: *?[*]const f32, out_stop: *bool) i32 {
        if (!self.flushGenWav()) return 1;
        // the decoder restarts too, the next chunk's audio is not continuous with this one
        mtmd.mtmd_gen_state_reset(self.base.gen_state.?);

        if (self.chunk_idx + 1 >= self.chunks.items.len) {
            out_stop.* = true;
            h_state_out.* = null;
            return 0;
        }
        self.chunk_idx += 1;

        // drop this chunk's text and audio, keep the voice conditioning
        _ = lapi.context.llama_memory_seq_rm(lapi.context.llama_get_memory(self.base.lctx), self.seq_id, self.n_voice_pos, -1);
        self.pos = self.n_voice_pos;

        const n_e = self.base.nE();
        const buf = &self.prompt.prompt_embd_buf;
        buf.clearRetainingCapacity();
        for (self.chunks.items[self.chunk_idx].items) |t| self.pushEmbdRow(buf, t);
        self.pushEmbdRow(buf, self.audio_bos);
        self.armChunkBudget(self.chunk_idx);

        const n_rows: c_int = @intCast(buf.items.len / n_e);
        impl.ggmlAssert(n_rows > 0, "mtmd-helper-gen.cpp", 935, "n_rows > 0");
        const batch = EmbdBatch.create(buf.items.ptr, n_rows, 1, self.base.n_embd);
        defer batch.destroy();
        batch.setPositionNormal(self.pos, self.seq_id);
        batch.logits.items[@intCast(n_rows - 1)] = 1;
        if (lapi.context.llama_process(self.base.lctx, c.LLAMA_PROCESS_TYPE_DECODE, batch.render(self.base.lctx, 0, n_rows)) != 0) {
            logErr("mtmd_helper_gen_audio: chunk prompt decode failed\n", .{});
            return 1;
        }
        self.pos += n_rows;
        buf.clearRetainingCapacity();

        h_state_out.* = self.base.takeHState(&self.h_state_buf);
        out_stop.* = false;
        return 0;
    }

    /// `prepare_text` (mtmd-helper-gen.cpp:954 @8642541b2): the reference's
    /// `prepare_text_prompt()` normalization. The caller owns the text.
    fn prepareText(in: []const u8, pad_short: bool) []u8 {
        var s: std.ArrayList(u8) = .empty;
        s.ensureTotalCapacity(gpa, in.len + 1) catch oom();
        for (in) |ch| {
            s.appendAssumeCapacity(switch (ch) {
                '\n', '\r' => ' ',
                ';' => ',',
                else => ch,
            });
        }
        const b = std.mem.indexOfNone(u8, s.items, " ") orelse {
            s.deinit(gpa);
            return gpa.alloc(u8, 0) catch oom();
        };
        const e = std.mem.lastIndexOfNone(u8, s.items, " ").?;
        var out: std.ArrayList(u8) = .empty;
        out.ensureTotalCapacity(gpa, e - b + 2 + 8) catch oom();
        out.appendSliceAssumeCapacity(s.items[b .. e + 1]);
        s.deinit(gpa);
        if (out.items[0] >= 'a' and out.items[0] <= 'z') out.items[0] = out.items[0] - 'a' + 'A';
        const last = out.items[out.items.len - 1];
        if (std.ascii.isAlphanumeric(last)) out.appendAssumeCapacity('.'); // std::isalnum, "C" locale
        if (pad_short and countWords(out.items) < 5) {
            out.insertSlice(gpa, 0, "        ") catch oom(); // std::string(8, ' ')
        }
        return out.toOwnedSlice(gpa) catch oom();
    }

    /// `count_words` (mtmd-helper-gen.cpp:985 @8642541b2).
    fn countWords(s: []const u8) c_int {
        var n: c_int = 0;
        var in_word = false;
        for (s) |ch| {
            if (ch == ' ') {
                in_word = false;
            } else if (!in_word) {
                in_word = true;
                n += 1;
            }
        }
        return n;
    }

    /// `flush_gen_wav` (mtmd-helper-gen.cpp:1033 @8642541b2): decodes the
    /// buffered latents, the mimi decoder state carrying over.
    fn flushGenWav(self: *Pockettts) bool {
        if (self.feats_buf.items.len == 0) return true;
        var inp = mtmd.mtmd_gen_inp_default(self.base.mctx);
        inp.type = .gen_wav;
        inp.feats = self.feats_buf.items.ptr;
        inp.n_feats = self.feats_buf.items.len;
        var out = std.mem.zeroes(mtmd.GenOut);
        if (mtmd.mtmd_gen_audio_process(self.base.mctx, self.base.gen_state, &inp, &out) != 0) {
            logErr("mtmd_helper_gen_audio: mimi decode failed\n", .{});
            return false;
        }
        if (out.n_samples != 0) self.audio_pcm.appendSlice(gpa, out.audio.?[0..out.n_samples]) catch oom();
        self.feats_buf.clearRetainingCapacity();
        return true;
    }
};

// -----------------------------------------------------------------------------
// Soprano

/// Ports `soprano_gen_audio_pipeline` (mtmd-helper-gen.cpp:1087
/// @8642541b2): uses the hidden state that predicts each non-EOS token,
/// including the prompt's last state.
const Soprano = struct {
    base: Base,

    seq_id: c.llama_seq_id = 0,
    pos: c.llama_pos = 0,
    prompt_pos: c_int = 0,
    prompt: std.ArrayList(Token) = .empty,
    features: std.ArrayList(f32) = .empty,
    h_next: std.ArrayList(f32) = .empty,
    audio: std.ArrayList(f32) = .empty,
    wav: std.ArrayList(u8) = .empty,
    out_type: OutType = .wav,
    stream: bool = false,
    decoded: bool = false,
    sent: bool = false,

    fn deinit(self: *Soprano) void {
        self.base.deinit();
        self.prompt.deinit(gpa);
        self.features.deinit(gpa);
        self.h_next.deinit(gpa);
        self.audio.deinit(gpa);
        self.wav.deinit(gpa);
    }

    /// `reset` (mtmd-helper-gen.cpp:1091 @8642541b2).
    fn reset(self: *Soprano) void {
        self.prompt.clearRetainingCapacity();
        self.features.clearRetainingCapacity();
        self.h_next.clearRetainingCapacity();
        self.audio.clearRetainingCapacity();
        self.wav.clearRetainingCapacity();
        self.pos = 0;
        self.prompt_pos = 0;
        self.decoded = false;
        self.sent = false;
    }

    /// `set_input` (mtmd-helper-gen.cpp:1103 @8642541b2).
    fn setInput(self: *Soprano, inp: *const Inp) i32 {
        self.reset();
        const lctx = self.base.lctx;
        const vocab = self.base.vocab;
        if (inp.speaker_ref != null or inp.prompt == null or inp.prompt_len == 0 or inp.prompt_len > std.math.maxInt(i32) or
            inp.seq_id < 0 or @as(u32, @bitCast(inp.seq_id)) >= lapi.context.llama_n_seq_max(lctx))
        {
            logErr("mtmd_helper_gen_audio: soprano requires text, without a speaker reference\n", .{});
            return 1;
        }
        self.seq_id = inp.seq_id;
        self.out_type = inp.out_type;
        self.stream = inp.stream;
        const text = findSpecialToken(vocab, "[TEXT]");
        const start = findSpecialToken(vocab, "[START]");
        const stop = findSpecialToken(vocab, "[STOP]");
        if (text == token_null or start == token_null or stop == token_null) {
            logErr("mtmd_helper_gen_audio: incompatible soprano vocabulary\n", .{});
            return 1;
        }
        const prompt_len: i32 = @intCast(inp.prompt_len);
        const required = lapi.vocab.llama_tokenize(vocab, inp.prompt, prompt_len, null, 0, false, false);
        if (required >= 0 or @as(u32, @bitCast(-required + 3)) >= lapi.context.llama_n_ctx_seq(lctx)) {
            logErr("mtmd_helper_gen_audio: soprano prompt does not fit in the context\n", .{});
            return 1;
        }
        const n = -required;
        self.prompt.resize(gpa, @intCast(n + 3)) catch oom();
        self.prompt.items[0] = stop;
        self.prompt.items[1] = text;
        if (lapi.vocab.llama_tokenize(vocab, inp.prompt, prompt_len, self.prompt.items.ptr + 2, n, false, false) != n) return 1;
        self.prompt.items[self.prompt.items.len - 1] = start;
        if (!lapi.context.llama_memory_seq_rm(lapi.context.llama_get_memory(lctx), self.seq_id, 0, -1)) {
            logErr("mtmd_helper_gen_audio: cannot reset soprano sequence\n", .{});
            return 1;
        }
        if (!self.base.initGenState(inp.seed)) return 1;
        return 0;
    }

    /// `step_prompt` (mtmd-helper-gen.cpp:1143 @8642541b2).
    fn stepPrompt(self: *Soprano, n_batch: i32) i32 {
        if (n_batch <= 0 or self.prompt.items.len == 0) return -1;
        const n_prompt: c_int = @intCast(self.prompt.items.len);
        const n = @min(n_batch, n_prompt - self.prompt_pos);
        if (n == 0) return 0;
        if (!self.decode(self.prompt.items.ptr + @as(usize, @intCast(self.prompt_pos)), n)) return -1;
        self.prompt_pos += n;
        return n_prompt - self.prompt_pos;
    }

    /// `step_gen` (mtmd-helper-gen.cpp:1158 @8642541b2).
    fn stepGen(self: *Soprano, sampled_in: Token, h_state_in: ?[*]const f32, h_state_out: *?[*]const f32, out_stop: *bool) i32 {
        const lctx = self.base.lctx;
        const vocab = self.base.vocab;
        const n_e = self.base.nE();
        var sampled = sampled_in;
        h_state_out.* = null;
        out_stop.* = false;
        if (sampled < 0 or sampled >= lapi.vocab.llama_vocab_n_tokens(vocab) or h_state_in == null or self.prompt_pos != @as(c_int, @intCast(self.prompt.items.len)) or self.prompt.items.len == 0) return 1;
        if (lapi.vocab.llama_vocab_is_eog(vocab, sampled) or @as(u32, @bitCast(self.pos)) >= lapi.context.llama_n_ctx_seq(lctx)) {
            out_stop.* = true;
            return 0;
        }
        self.features.appendSlice(gpa, h_state_in.?[0..n_e]) catch oom();
        if (!self.decode(@ptrCast(&sampled), 1)) {
            self.features.shrinkRetainingCapacity(self.features.items.len - n_e);
            return 1;
        }
        const h = lapi.context.llama_get_embeddings_ith(lctx, -1) orelse return 1;
        assign(f32, &self.h_next, h, n_e);
        h_state_out.* = self.h_next.items.ptr;
        return 0;
    }

    /// `get_output` (mtmd-helper-gen.cpp:1182 @8642541b2).
    fn getOutput(self: *Soprano, sample_rate: *i32, data: *?[*]const u8, data_len: *usize, n_samples: ?*i64) i32 {
        sample_rate.* = self.base.info.sample_rate;
        data_len.* = 0;
        if (n_samples) |p| p.* = 0;
        if (self.stream) {
            // the decoder needs the whole utterance, so the audio comes out in one chunk after flush()
            if (!self.decoded or self.sent) return 0;
            self.sent = true;
        } else if (!self.decodeAudio()) {
            return 1;
        }
        if (n_samples) |p| p.* = @intCast(self.audio.items.len);
        if (self.out_type == .pcm) {
            data.* = if (self.audio.capacity == 0) null else @ptrCast(self.audio.items.ptr);
            data_len.* = self.audio.items.len * @sizeOf(f32);
        } else {
            self.wav.clearRetainingCapacity();
            if (!writeWav16(&self.wav, self.audio.items, self.base.info.sample_rate)) return 1;
            data.* = if (self.wav.capacity == 0) null else self.wav.items.ptr;
            data_len.* = self.wav.items.len;
        }
        return 0;
    }

    /// `flush` (mtmd-helper-gen.cpp:1211 @8642541b2): the whole utterance is
    /// decoded at once.
    fn flush(self: *Soprano) i32 {
        return if (self.decodeAudio()) 0 else 1;
    }

    /// `decode_audio` (mtmd-helper-gen.cpp:1216 @8642541b2).
    fn decodeAudio(self: *Soprano) bool {
        if (self.decoded) return true;
        if (self.features.items.len < 2 * self.base.nE()) {
            logErr("mtmd_helper_gen_audio: soprano needs at least two generated frames\n", .{});
            return false;
        }
        var inp = mtmd.mtmd_gen_inp_default(self.base.mctx);
        inp.type = .gen_wav;
        inp.feats = self.features.items.ptr;
        inp.n_feats = self.features.items.len;
        var out = std.mem.zeroes(mtmd.GenOut);
        if (mtmd.mtmd_gen_audio_process(self.base.mctx, self.base.gen_state, &inp, &out) != 0) return false;
        assign(f32, &self.audio, out.audio, out.n_samples);
        self.decoded = true;
        return true;
    }

    /// `decode` (mtmd-helper-gen.cpp:1237 @8642541b2).
    fn decode(self: *Soprano, tokens: [*]const Token, n: c_int) bool {
        var batch = lapi.batch.llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        var i: c_int = 0;
        while (i < n) : (i += 1) {
            const k: usize = @intCast(i);
            batch.token[k] = tokens[k];
            batch.pos[k] = self.pos + i;
            batch.n_seq_id[k] = 1;
            batch.seq_id[k][0] = self.seq_id;
            batch.logits[k] = @intFromBool(i == n - 1);
        }
        const ret = lapi.context.llama_decode(self.base.lctx, batch);
        lapi.batch.llama_batch_free(batch);
        if (ret != 0) {
            logErr("mtmd_helper_gen_audio: soprano backbone decode failed\n", .{});
            return false;
        }
        self.pos += n;
        return true;
    }
};

// -----------------------------------------------------------------------------
// LFM3-Audio

/// `__independent_bits_engine<mt19937, uint64_t>`, as `src/ggml/opt.zig`
/// ports it: the bits `std::shuffle` draws.
const IndependentBits = struct {
    e: *libcxx.Mt19937,
    n: usize,
    n0: usize,
    w0: usize,
    y0: u64,
    y1: u64,
    mask0: u32,
    mask1: u32,

    const rp: u64 = 1 << 32;
    const m: usize = 32;

    fn init(e: *libcxx.Mt19937, w: usize) IndependentBits {
        var self: IndependentBits = undefined;
        self.e = e;
        self.n = w / m + @intFromBool(w % m != 0);
        self.w0 = w / self.n;
        self.y0 = (rp >> @intCast(self.w0)) << @intCast(self.w0);
        if (rp - self.y0 > self.y0 / @as(u64, self.n)) {
            self.n += 1;
            self.w0 = w / self.n;
            self.y0 = (rp >> @intCast(self.w0)) << @intCast(self.w0);
        }
        self.n0 = self.n - w % self.n;
        self.y1 = (rp >> @intCast(self.w0 + 1)) << @intCast(self.w0 + 1);
        self.mask0 = if (self.w0 > 0) ~@as(u32, 0) >> @intCast(32 - self.w0) else 0;
        self.mask1 = if (self.w0 < 31) ~@as(u32, 0) >> @intCast(32 - (self.w0 + 1)) else ~@as(u32, 0);
        return self;
    }

    fn next(self: *const IndependentBits) u64 {
        var sp: u64 = 0;
        var k: usize = 0;
        while (k < self.n0) : (k += 1) {
            var u: u32 = undefined;
            while (true) {
                u = self.e.next();
                if (@as(u64, u) < self.y0) break;
            }
            sp <<= @intCast(self.w0);
            sp +%= u & self.mask0;
        }
        while (k < self.n) : (k += 1) {
            var u: u32 = undefined;
            while (true) {
                u = self.e.next();
                if (@as(u64, u) < self.y1) break;
            }
            sp <<= @intCast(self.w0 + 1);
            sp +%= u & self.mask1;
        }
        return sp;
    }
};

/// `uniform_int_distribution<ptrdiff_t>` over `[0, b]`, `b` at most `n_cb`.
fn uniformInt(g: *libcxx.Mt19937, b: u64) u64 {
    const r: u64 = b + 1;
    if (r == 1) return 0;
    var w: usize = 64 - @clz(r) - 1;
    if ((r & (@as(u64, std.math.maxInt(u64)) >> @intCast(64 - w))) != 0) w += 1;
    const e = IndependentBits.init(g, w);
    while (true) {
        const u = e.next();
        if (u < r) return u;
    }
}

/// `std::shuffle` (shuffle.h), the loop `src/ggml/opt.zig` ports for its
/// dataset: which draws are made is libc++'s, and the seed reproduces them.
fn shuffle(items: []c_int, g: *libcxx.Mt19937) void {
    if (items.len <= 1) return;
    var d: u64 = items.len - 1;
    var first: usize = 0;
    const last: usize = items.len - 1;
    while (first < last) : ({
        first += 1;
        d -= 1;
    }) {
        const i = uniformInt(g, d);
        if (i != 0) std.mem.swap(c_int, &items[first], &items[first + @as(usize, @intCast(i))]);
    }
}

/// `std::nth_element(..., std::greater<float>())` then `sorted[k]`: only the
/// k-th largest value is read, which any selection gives.
fn nthLargest(sorted: []f32, k: usize) f32 {
    std.mem.sort(f32, sorted, {}, std.sort.desc(f32));
    return sorted[k];
}

/// Ports `lfm3a_gen_audio_pipeline` (mtmd-helper-gen.cpp:1273 @8642541b2): LFM3-Audio
/// TTS, see generate() in the reference: each step samples a frame of codes by
/// MaskGIT and feeds its embedding back, the end-of-audio code stops.
const Lfm3a = struct {
    base: Base,

    tok_audio_start: Token = token_null,
    tok_mixed_start: Token = token_null,
    tok_text_end: Token = token_null,
    tok_im_end: Token = token_null,

    continuing: bool = false, // the prompt continues a sequence, the response follows the model's switching
    mode: GenMode = .audio,
    slots_left: c_int = 0,
    speaking: bool = false, // interleaving: the current block is audio
    text_done: bool = false, // interleaving: <|text_end|> was emitted, audio until the end-of-audio frame

    seq_id: c.llama_seq_id = 0,
    pos: c_int = 0,
    prompt_tokens: std.ArrayList(Token) = .empty,
    prompt_pos: c_int = 0,

    temp: f32 = 0.0,
    top_k: i32 = 0,
    rep_penalty: f32 = 1.0,
    rep_window: i32 = 0, // frames
    rng: libcxx.Mt19937 = undefined,
    logits: std.ArrayList(f32) = .empty,
    fb_buf: std.ArrayList(f32) = .empty,
    hist: std.ArrayList(i32) = .empty, // frame-major codes of the frames so far

    pending_tail: bool = false, // the ISTFT holds back the overlap of the last frame
    audio_pcm: std.ArrayList(f32) = .empty,
    h_state_buf: std.ArrayList(f32) = .empty,
    out_type: OutType = .wav,
    out_buf: std.ArrayList(u8) = .empty,
    so: StreamOut = .{},

    // checked by the converter
    /// `n_cb` (mtmd-helper-gen.cpp:1525 @8642541b2).
    const n_cb: usize = 12;
    /// `schedule` (mtmd-helper-gen.cpp:1526 @8642541b2): codebooks revealed per depthformer pass.
    const schedule = [_]c_int{ 4, 4, 4 };
    /// `n_text_block` (mtmd-helper-gen.cpp:1682 @8642541b2): interleaved_n_text.
    const n_text_block: c_int = 6;
    /// `n_audio_block` (mtmd-helper-gen.cpp:1683 @8642541b2): interleaved_n_audio.
    const n_audio_block: c_int = 12;

    /// generate() modes: text, audio until end of audio, interleaved until <|text_end|>
    const GenMode = enum { text, audio, mixed };

    fn deinit(self: *Lfm3a) void {
        self.base.deinit();
        self.prompt_tokens.deinit(gpa);
        self.logits.deinit(gpa);
        self.fb_buf.deinit(gpa);
        self.hist.deinit(gpa);
        self.audio_pcm.deinit(gpa);
        self.h_state_buf.deinit(gpa);
        self.out_buf.deinit(gpa);
    }

    /// `reset` (mtmd-helper-gen.cpp:1277 @8642541b2). The C++ leaves `stream`
    /// alone, `set_input` sets it next.
    fn reset(self: *Lfm3a) void {
        self.seq_id = 0;
        self.pos = 0;
        self.continuing = false;
        self.mode = .audio;
        self.slots_left = 0;
        self.speaking = false;
        self.text_done = false;
        self.prompt_tokens.clearRetainingCapacity();
        self.prompt_pos = 0;
        self.hist.clearRetainingCapacity();
        self.base.dropGenState();
        self.pending_tail = false;
        self.so.pcm_sent = 0;
        self.so.wav_header_sent = false;
        self.audio_pcm.clearRetainingCapacity();
        self.h_state_buf.clearRetainingCapacity();
        self.out_buf.clearRetainingCapacity();
    }

    /// `set_input` (mtmd-helper-gen.cpp:1297 @8642541b2).
    fn setInput(self: *Lfm3a, inp: *const Inp) i32 {
        self.reset();
        self.seq_id = inp.seq_id;

        if (inp.speaker_ref != null) {
            logWrn("mtmd_helper_gen_audio: the model has no voice conditioning, the speaker file is ignored\n", .{});
        }

        // the prompt is rendered by the caller: the "tts" chat template of the model, or the next turn of a conversation
        self.continuing = inp.n_past > 0;
        self.pos = if (self.continuing) inp.n_past else 0;
        const full: []const u8 = if (inp.prompt_len == 0) "" else inp.prompt.?[0..inp.prompt_len];
        self.prompt_tokens.resize(gpa, full.len + 16) catch oom();
        const n_ids = lapi.vocab.llama_tokenize(self.base.vocab, full.ptr, @intCast(full.len), self.prompt_tokens.items.ptr, @intCast(self.prompt_tokens.items.len), false, true);
        if (n_ids <= 0) {
            logErr("mtmd_helper_gen_audio: tokenization failed\n", .{});
            return 1;
        }
        self.prompt_tokens.shrinkRetainingCapacity(@intCast(n_ids));

        if (!self.findTokens()) return 1;
        // a continued response opens with the token of the asked mode, a text-only one cannot switch to audio
        if (self.continuing) {
            switch (inp.out_mode) {
                .audio => self.prompt_tokens.append(gpa, self.tok_audio_start) catch oom(),
                .mixed => self.prompt_tokens.append(gpa, self.tok_mixed_start) catch oom(),
                .text => {
                    self.base.logit_bias.clearRetainingCapacity();
                    self.base.logit_bias.append(gpa, .{ .token = self.tok_audio_start, .bias = -std.math.inf(f32) }) catch oom();
                    self.base.logit_bias.append(gpa, .{ .token = self.tok_mixed_start, .bias = -std.math.inf(f32) }) catch oom();
                },
                else => {},
            }
        }
        // the last prompt token picks the mode, see generate() of the reference
        const last = self.prompt_tokens.items[self.prompt_tokens.items.len - 1];
        self.mode = if (last == self.tok_audio_start) .audio else if (last == self.tok_mixed_start) .mixed else .text;
        self.slots_left = n_text_block;
        if (!self.continuing and self.mode != .audio) {
            logErr("mtmd_helper_gen_audio: the prompt must come from the \"tts\" chat template of the model\n", .{});
            return 1;
        }

        // the sampling defaults of the model, the caller's top-k/top-p are not used
        const def = mtmd.mtmd_gen_inp_default(self.base.mctx);
        self.temp = def.temp;
        self.top_k = def.top_k;
        self.rep_penalty = def.penalty_repeat;
        self.rep_window = def.penalty_last_n;

        self.out_type = inp.out_type;
        self.so.stream = inp.stream;
        self.rng = libcxx.Mt19937.init(if (inp.seed == std.math.maxInt(u32)) clip.randomDevice() else inp.seed);
        return if (self.base.initGenState(inp.seed)) 0 else 1;
    }

    /// `step_prompt` (mtmd-helper-gen.cpp:1354 @8642541b2).
    fn stepPrompt(self: *Lfm3a, n_batch: i32) i32 {
        impl.ggmlAssert(n_batch > 0, "mtmd-helper-gen.cpp", 1355, "n_batch > 0");
        const n_prompt: c_int = @intCast(self.prompt_tokens.items.len);
        if (self.prompt_pos >= n_prompt) return 0;
        const n = @min(n_batch, n_prompt - self.prompt_pos);
        var batch = lapi.batch.llama_batch_init(n, 0, 1);
        var i: c_int = 0;
        while (i < n) : (i += 1) {
            const k: usize = @intCast(i);
            batch.token[k] = self.prompt_tokens.items[@intCast(self.prompt_pos + i)];
            batch.pos[k] = self.pos + i;
            batch.n_seq_id[k] = 1;
            batch.seq_id[k][0] = self.seq_id;
            batch.logits[k] = @intFromBool(self.prompt_pos + i == n_prompt - 1);
        }
        batch.n_tokens = n;
        const ret = lapi.context.llama_decode(self.base.lctx, batch);
        lapi.batch.llama_batch_free(batch);
        if (ret != 0) {
            logErr("mtmd_helper_gen_audio: prompt decode failed\n", .{});
            return -1;
        }
        self.pos += n;
        self.prompt_pos += n;
        return n_prompt - self.prompt_pos;
    }

    /// `next_is_text` (mtmd-helper-gen.cpp:1385 @8642541b2).
    fn nextIsText(self: *const Lfm3a) bool {
        return self.mode == .text or (self.mode == .mixed and !self.speaking);
    }

    /// `step_gen` (mtmd-helper-gen.cpp:1390 @8642541b2): one generate() step, a text
    /// token from the caller or an audio frame from the hidden state.
    fn stepGen(self: *Lfm3a, sampled_in: Token, h_state_in: ?[*]const f32, h_state_out: *?[*]const f32, out_stop: *bool) i32 {
        var sampled = sampled_in;
        const interleaving = self.mode == .mixed;
        if (self.nextIsText()) {
            // <|im_end|> ends the response in any text slot, <|text_end|> outside an interleaved block too
            if (sampled == self.tok_im_end or (!interleaving and sampled == self.tok_text_end)) {
                out_stop.* = true;
                h_state_out.* = null;
                return 0;
            }
            // the switching tokens are fed back like any text token
            if (!interleaving) {
                if (sampled == self.tok_audio_start) {
                    self.mode = .audio;
                } else if (sampled == self.tok_mixed_start) {
                    self.mode = .mixed;
                    self.slots_left = n_text_block;
                    self.speaking = false;
                    self.text_done = false;
                }
            } else if (sampled == self.tok_text_end) {
                self.text_done = true;
                self.speaking = true;
            }
            if (!self.decodeToken(&sampled)) return 1;
        } else {
            var codes: [n_cb]i32 = undefined;
            var fb: ?[*]const f32 = null;
            var is_eos = false;
            if (!self.sampleFrame(h_state_in, &codes, &fb, &is_eos)) return 1;
            if (is_eos) {
                if (!self.continuing) {
                    // TTS: the end of audio ends the generation
                    out_stop.* = true;
                    h_state_out.* = null;
                    return 0;
                }
                // the segment ends, the codec tail goes out and the response continues in text
                if (self.flush() != 0) return 1;
                self.mode = .text;
            } else {
                self.hist.appendSlice(gpa, &codes) catch oom();
                if (!self.decodeWav(&codes, codes.len)) return 1;
            }
            const batch_embd = EmbdBatch.create(fb, 1, 1, self.base.n_embd);
            defer batch_embd.destroy();
            batch_embd.setPositionNormal(self.pos, self.seq_id);
            batch_embd.logits.items[0] = 1;
            if (lapi.context.llama_process(self.base.lctx, c.LLAMA_PROCESS_TYPE_DECODE, batch_embd.render(self.base.lctx, 0, 1)) != 0) {
                logErr("mtmd_helper_gen_audio: decode failed\n", .{});
                return 1;
            }
            self.pos += 1;
        }

        // a slot emitted while interleaving counts down the block, the switching token itself does not
        if (interleaving and !self.text_done) {
            self.slots_left -= 1;
            if (self.slots_left == 0) {
                self.speaking = !self.speaking;
                self.slots_left = if (self.speaking) n_audio_block else n_text_block;
            }
        }

        h_state_out.* = self.base.takeHState(&self.h_state_buf);
        return 0;
    }

    /// `get_output` (mtmd-helper-gen.cpp:1464 @8642541b2). Streaming: every frame
    /// is decoded as it is sampled.
    fn getOutput(self: *Lfm3a, out_sample_rate: *i32, out_data: *?[*]const u8, out_data_len: *usize, out_n_samples: ?*i64) i32 {
        out_sample_rate.* = self.base.info.sample_rate;
        if (!self.so.stream and self.flush() != 0) return 1;
        return self.base.getOutput(&self.audio_pcm, self.out_type, &self.out_buf, &self.so, out_data, out_data_len, out_n_samples);
    }

    /// `flush` (mtmd-helper-gen.cpp:1513 @8642541b2): the ISTFT holds back the
    /// overlap of the last frame.
    fn flush(self: *Lfm3a) i32 {
        if (self.pending_tail) {
            if (!self.decodeWav(null, 0)) return 1;
            self.pending_tail = false;
        }
        return 0;
    }

    /// `sample_frame` (mtmd-helper-gen.cpp:1530 @8642541b2): a random order of the
    /// codebooks, revealed by the schedule; `fb` is the feedback embedding of the
    /// codes. At the end of audio every code is the end code, its feedback
    /// embedding is only built when continuing.
    fn sampleFrame(self: *Lfm3a, h_state: ?[*]const f32, codes: *[n_cb]i32, fb: *?[*]const f32, is_eos: *bool) bool {
        var order: [n_cb]c_int = undefined;
        for (&order, 0..) |*o, i| o.* = @intCast(i);
        shuffle(&order, &self.rng);

        @memset(codes, -1);
        var inp = mtmd.mtmd_gen_inp_default(self.base.mctx);
        inp.type = .gen_code;
        inp.embd = @constCast(h_state);
        inp.codes = codes;
        inp.n_codes = n_cb;
        var out = std.mem.zeroes(mtmd.GenOut);

        var k: usize = 0;
        for (schedule) |n_pick| {
            if (mtmd.mtmd_gen_audio_process(self.base.mctx, self.base.gen_state, &inp, &out) != 0 or out.n_feats == 0 or out.n_feats % n_cb != 0) {
                logErr("mtmd_helper_gen_audio: depthformer pass failed\n", .{});
                return false;
            }
            const n_vocab = out.n_feats / n_cb;
            assign(f32, &self.logits, out.feats, out.n_feats);
            var j: c_int = 0;
            while (j < n_pick) : ({
                j += 1;
                k += 1;
            }) {
                const i: usize = @intCast(order[k]);
                const l = self.logits.items[i * n_vocab ..][0..n_vocab];
                self.applyRepPenalty(l, i);
                codes[i] = self.sample(l);
            }
        }

        // the last code of every codebook ends the audio
        const eos: i32 = @intCast(self.logits.items.len / n_cb - 1);
        is_eos.* = std.mem.indexOfScalar(i32, codes, eos) != null;
        if (is_eos.*) {
            @memset(codes, eos);
            if (!self.continuing) {
                fb.* = null;
                return true;
            }
        }

        // all codes known: only the feedback embedding
        if (mtmd.mtmd_gen_audio_process(self.base.mctx, self.base.gen_state, &inp, &out) != 0) {
            logErr("mtmd_helper_gen_audio: codebook embedding failed\n", .{});
            return false;
        }
        assign(f32, &self.fb_buf, out.embd, self.base.nE());
        fb.* = self.fb_buf.items.ptr;
        return true;
    }

    /// `apply_rep_penalty` (mtmd-helper-gen.cpp:1583 @8642541b2): the CTRL penalty
    /// over the codes codebook `cb` took in the last `rep_window` frames.
    fn applyRepPenalty(self: *const Lfm3a, l: []f32, cb: usize) void {
        if (self.rep_penalty <= 1.0 or self.rep_window <= 0) return;
        const n_hist = self.hist.items.len / n_cb;
        const n_win = @min(n_hist, @as(usize, @intCast(self.rep_window)));
        const seen = gpa.alloc(bool, l.len) catch oom();
        defer gpa.free(seen);
        @memset(seen, false);
        var f = n_hist - n_win;
        while (f < n_hist) : (f += 1) {
            const cc = self.hist.items[f * n_cb + cb];
            if (cc >= 0 and @as(usize, @intCast(cc)) < l.len and !seen[@intCast(cc)]) {
                const i: usize = @intCast(cc);
                seen[i] = true;
                l[i] = if (l[i] > 0.0) l[i] / self.rep_penalty else l[i] * self.rep_penalty;
            }
        }
    }

    /// `sample` (mtmd-helper-gen.cpp:1600 @8642541b2).
    fn sample(self: *Lfm3a, l: []const f32) i32 {
        const n_vocab = l.len;
        if (self.temp <= 0.0) return @intCast(maxElement(l));
        // top-k keeps the codes scoring at least the k-th best, ties included
        var lmin = -std.math.inf(f32);
        if (self.top_k > 0 and @as(usize, @intCast(self.top_k)) < n_vocab) {
            const sorted = gpa.dupe(f32, l) catch oom();
            defer gpa.free(sorted);
            lmin = nthLargest(sorted, @intCast(self.top_k - 1));
        }
        const lmax = l[maxElement(l)];
        const prob = gpa.alloc(f64, n_vocab) catch oom();
        defer gpa.free(prob);
        var sum: f64 = 0.0;
        const temp: f64 = self.temp;
        for (0..n_vocab) |v| {
            prob[v] = if (l[v] < lmin) 0.0 else exp(@as(f64, l[v] - lmax) / temp);
            sum += prob[v];
        }
        var r = libcxx.canonicalF64(&self.rng) * sum;
        for (0..n_vocab) |v| {
            r -= prob[v];
            if (r <= 0.0) return @intCast(v);
        }
        return @intCast(n_vocab - 1);
    }

    /// `std::max_element`: the first of the largest.
    fn maxElement(l: []const f32) usize {
        var best: usize = 0;
        for (l, 0..) |x, i| {
            if (l[best] < x) best = i;
        }
        return best;
    }

    /// `decode_wav` (mtmd-helper-gen.cpp:1629 @8642541b2): `n_codes == 0` flushes the
    /// ISTFT tail.
    fn decodeWav(self: *Lfm3a, codes: ?[*]const i32, n_codes: usize) bool {
        var inp = mtmd.mtmd_gen_inp_default(self.base.mctx);
        inp.type = .gen_wav;
        inp.codes = @constCast(codes);
        inp.n_codes = n_codes;
        var out = std.mem.zeroes(mtmd.GenOut);
        if (mtmd.mtmd_gen_audio_process(self.base.mctx, self.base.gen_state, &inp, &out) != 0) {
            logErr("mtmd_helper_gen_audio: codec decode failed\n", .{});
            return false;
        }
        if (out.n_samples != 0) self.audio_pcm.appendSlice(gpa, out.audio.?[0..out.n_samples]) catch oom();
        self.pending_tail = n_codes > 0;
        return true;
    }

    /// `find_tokens` (mtmd-helper-gen.cpp:1644 @8642541b2).
    fn findTokens(self: *Lfm3a) bool {
        const v = self.base.vocab;
        self.tok_audio_start = findSpecialToken(v, "<|audio_start|>");
        self.tok_mixed_start = findSpecialToken(v, "<|mixed_start|>");
        self.tok_text_end = findSpecialToken(v, "<|text_end|>");
        self.tok_im_end = findSpecialToken(v, "<|im_end|>");
        if (self.tok_audio_start == token_null or self.tok_mixed_start == token_null or
            self.tok_text_end == token_null or self.tok_im_end == token_null)
        {
            logErr("mtmd_helper_gen_audio: missing LFM3-Audio control tokens in vocab\n", .{});
            return false;
        }
        return true;
    }

    /// `decode` (mtmd-helper-gen.cpp:1657 @8642541b2): over `llama_batch_get_one(&sampled, 1)`.
    fn decodeToken(self: *Lfm3a, token: *Token) bool {
        var batch = lapi.batch.llama_batch_get_one(token, 1);
        var position: c.llama_pos = self.pos;
        var n_seq: i32 = 1;
        var seq: c.llama_seq_id = self.seq_id;
        var seq_ptr: [*c]c.llama_seq_id = &seq;
        var logits: i8 = 1;
        batch.pos = &position;
        batch.n_seq_id = &n_seq;
        batch.seq_id = &seq_ptr;
        batch.logits = &logits;
        if (lapi.context.llama_decode(self.base.lctx, batch) != 0) {
            logErr("mtmd_helper_gen_audio: decode failed\n", .{});
            return false;
        }
        self.pos += batch.n_tokens;
        return true;
    }
};

/// The C++'s `mtmd_gen_audio_pipeline` hierarchy, one arm a subclass.
const Pipeline = union(enum) {
    qwen3tts: Qwen3tts,
    pockettts: Pockettts,
    soprano: Soprano,
    lfm3a: Lfm3a,

    fn deinit(self: *Pipeline) void {
        switch (self.*) {
            inline else => |*p| p.deinit(),
        }
    }
};

/// Ports `mtmd_helper_gen_audio` (mtmd-helper-gen.cpp:1735 @8642541b2).
pub const GenAudio = struct {
    pipeline: ?Pipeline = null,
};

/// Ports `make_pipeline` (mtmd-helper-gen.cpp:1720 @8642541b2).
fn makePipeline(lctx: ?*c.struct_llama_context, mctx: *Context) ?Pipeline {
    return switch (mtmd.mtmd_gen_audio_get_info(mctx).type) {
        .qwen3tts => .{ .qwen3tts = .{ .base = Base.init(lctx, mctx) } },
        .pockettts => .{ .pockettts = .{ .base = Base.init(lctx, mctx) } },
        .soprano => .{ .soprano = .{ .base = Base.init(lctx, mctx) } },
        .lfm3a => .{ .lfm3a = .{ .base = Base.init(lctx, mctx) } },
        else => null,
    };
}

/// Ports `mtmd_helper_gen_audio_init` (mtmd-helper-gen.cpp:1739 @8642541b2).
pub fn mtmd_helper_gen_audio_init(lctx: ?*c.struct_llama_context, mctx: *Context) callconv(.c) *GenAudio {
    const ctx = gpa.create(GenAudio) catch oom();
    ctx.* = .{ .pipeline = makePipeline(lctx, mctx) };
    return ctx;
}

/// Ports `mtmd_helper_gen_audio_free` (mtmd-helper-gen.cpp:1745 @8642541b2).
pub fn mtmd_helper_gen_audio_free(ctx: ?*GenAudio) callconv(.c) void {
    const x = ctx orelse return;
    if (x.pipeline) |*p| p.deinit();
    gpa.destroy(x);
}

/// Ports `mtmd_helper_gen_audio_reset` (mtmd-helper-gen.cpp:1749 @8642541b2).
pub fn mtmd_helper_gen_audio_reset(ctx: ?*GenAudio) callconv(.c) void {
    const x = ctx orelse return;
    if (x.pipeline) |*p| switch (p.*) {
        inline else => |*q| q.reset(),
    };
}

/// Ports `mtmd_helper_gen_audio_inp_default` (mtmd-helper-gen.cpp:1755 @8642541b2).
pub fn mtmd_helper_gen_audio_inp_default() callconv(.c) Inp {
    var inp = std.mem.zeroes(Inp);
    inp.top_k = 50;
    inp.top_p = 1.0;
    inp.seed = std.math.maxInt(u32); // random
    inp.out_type = .wav;
    return inp;
}

/// Ports `mtmd_helper_gen_audio_set_input` (mtmd-helper-gen.cpp:1764
/// @8642541b2).
pub fn mtmd_helper_gen_audio_set_input(ctx: *GenAudio, inp: *const Inp) callconv(.c) i32 {
    const p = if (ctx.pipeline) |*p| p else {
        logErr("mtmd_helper_gen_audio: unsupported or missing gen-audio pipeline\n", .{});
        return 1;
    };
    if (inp.n_past > 0 and p.* != .lfm3a) {
        logErr("mtmd_helper_gen_audio: the pipeline cannot continue a sequence\n", .{});
        return 1;
    }
    return switch (p.*) {
        inline else => |*q| blk: {
            q.base.logit_bias.clearRetainingCapacity();
            break :blk q.setInput(inp);
        },
    };
}

/// Ports `mtmd_helper_gen_audio_step_prompt` (mtmd-helper-gen.cpp:1777
/// @8642541b2): >0 prompt tokens remaining, 0 done, <0 an error.
pub fn mtmd_helper_gen_audio_step_prompt(ctx: *GenAudio, n_batch: i32) callconv(.c) i32 {
    const p = if (ctx.pipeline) |*p| p else return -1;
    return switch (p.*) {
        .soprano => |*q| q.stepPrompt(n_batch),
        .lfm3a => |*q| q.stepPrompt(n_batch),
        inline else => |*q| q.base.stepPrompt(&q.prompt, &q.pos, n_batch),
    };
}

/// Ports `mtmd_helper_gen_audio_next_is_text` (mtmd-helper-gen.cpp:1784 @8642541b2).
pub fn mtmd_helper_gen_audio_next_is_text(ctx: *GenAudio) callconv(.c) bool {
    const p = if (ctx.pipeline) |*p| p else return false;
    return switch (p.*) {
        .lfm3a => |*q| q.nextIsText(),
        else => false,
    };
}

/// Ports `mtmd_helper_gen_audio_get_logit_bias` (mtmd-helper-gen.cpp:1788 @8642541b2).
pub fn mtmd_helper_gen_audio_get_logit_bias(ctx: *GenAudio, n_bias: *usize) callconv(.c) ?[*]const c.llama_logit_bias {
    const p = if (ctx.pipeline) |*p| p else {
        n_bias.* = 0;
        return null;
    };
    const items = switch (p.*) {
        inline else => |*q| q.base.logit_bias.items,
    };
    n_bias.* = items.len;
    return if (items.len != 0) items.ptr else null;
}

/// Ports `mtmd_helper_audio_stream_get_logit_bias` (mtmd-helper-gen.cpp:1793 @8642541b2):
/// the bias for the text a caller samples while it transcribes an audio stream
/// opened with a prompt. Writes up to `n_max` entries to `out`.
pub fn mtmd_helper_audio_stream_get_logit_bias(mctx: *const Context, model: ?*const c.struct_llama_model, out: ?[*]c.llama_logit_bias, n_max: i32) callconv(.c) i32 {
    var bias: [6]c.llama_logit_bias = undefined;
    var n: usize = 0;
    if (mtmd.mtmd_gen_audio_get_info(mctx).type == .lfm3a) {
        // LFM3-Audio live ASR: only <|text_end|> ends the text of a chunk, the stream adds the final <|mixed_end|>
        const vocab = lapi.model.llama_model_get_vocab(model);
        for ([_][]const u8{ "<|mixed_end|>", "<|mixed_start|>", "<|audio_start|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>" }) |piece| {
            const tok = findSpecialToken(vocab, piece);
            if (tok != token_null) {
                bias[n] = .{ .token = tok, .bias = -std.math.inf(f32) };
                n += 1;
            }
        }
    }
    var i: i32 = 0;
    while (i < n_max and i < @as(i32, @intCast(n))) : (i += 1) out.?[@intCast(i)] = bias[@intCast(i)];
    return @intCast(n);
}

/// Ports `mtmd_helper_gen_audio_step_gen` (mtmd-helper-gen.cpp:1812
/// @8642541b2).
pub fn mtmd_helper_gen_audio_step_gen(ctx: *GenAudio, sampled: Token, h_state_in: ?[*]const f32, h_state_out: *?[*]const f32, out_stop: ?*bool) callconv(.c) i32 {
    const p = if (ctx.pipeline) |*p| p else return 1;
    var stop = false;
    const ret = switch (p.*) {
        .qwen3tts => |*q| q.stepGen(sampled, h_state_in, h_state_out, &stop),
        .pockettts => |*q| q.stepGen(h_state_in, h_state_out, &stop),
        .soprano => |*q| q.stepGen(sampled, h_state_in, h_state_out, &stop),
        .lfm3a => |*q| q.stepGen(sampled, h_state_in, h_state_out, &stop),
    };
    if (out_stop) |s| s.* = stop;
    return ret;
}

/// Ports `mtmd_helper_gen_audio_get_output` (mtmd-helper-gen.cpp:1826
/// @8642541b2).
pub fn mtmd_helper_gen_audio_get_output(ctx: *GenAudio, out_sample_rate: *i32, out_data: *?[*]const u8, out_data_len: *usize, out_n_samples: ?*i64) callconv(.c) i32 {
    const p = if (ctx.pipeline) |*p| p else return 1;
    return switch (p.*) {
        inline else => |*q| q.getOutput(out_sample_rate, out_data, out_data_len, out_n_samples),
    };
}

/// Ports `mtmd_helper_gen_audio_flush` (mtmd-helper-gen.cpp:1834 @8642541b2): forces any
/// buffered codes through code2wav now, regardless of `window_frames`.
pub fn mtmd_helper_gen_audio_flush(ctx: *GenAudio) callconv(.c) i32 {
    const p = if (ctx.pipeline) |*p| p else return 1;
    return switch (p.*) {
        inline else => |*q| q.flush(),
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "prepare_text normalizes as the reference does" {
    const t = Pockettts.prepareText("\n hello;\rworld \n", false);
    defer gpa.free(t);
    try std.testing.expectEqualStrings("Hello, world.", t);
    const p = Pockettts.prepareText("hi there", true);
    defer gpa.free(p);
    try std.testing.expectEqualStrings("        Hi there.", p);
    const e = Pockettts.prepareText("   ", false);
    defer gpa.free(e);
    try std.testing.expectEqual(@as(usize, 0), e.len);
}

test "split_on keeps punctuation with its sentence" {
    var out = Pockettts.splitOn(&.{ 1, 9, 9, 2, 3, 9 }, &.{9});
    defer freeLists(&out);
    try std.testing.expectEqual(@as(usize, 2), out.items.len);
    try std.testing.expectEqualSlices(Token, &.{ 1, 9, 9 }, out.items[0].items);
    try std.testing.expectEqualSlices(Token, &.{ 2, 3, 9 }, out.items[1].items);
}

test "write_wav16 clamps as std::min and std::max do" {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    try std.testing.expect(writeWav16(&buf, &.{ 2.0, -2.0, std.math.nan(f32), 0.5 }, 24000));
    try std.testing.expectEqual(@as(usize, 44 + 8), buf.items.len);
    const s = std.mem.bytesAsSlice(i16, buf.items[44..]);
    try std.testing.expectEqual(@as(i16, 32767), s[0]);
    try std.testing.expectEqual(@as(i16, -32767), s[1]);
    try std.testing.expectEqual(@as(i16, 32767), s[2]); // NaN: std::min(1, NaN) is 1
    try std.testing.expectEqual(@as(i16, 16383), s[3]);
}
