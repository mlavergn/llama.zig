//! libmtmd's audio-generation helpers: the Qwen3-TTS and pocket-tts
//! pipelines that drive a backbone and the mmproj's generator frame by frame.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/mtmd-helper-gen.cpp` and the gen-audio types
//! of `llama.cpp/tools/mtmd/mtmd-helper.h`, at v0.4.1 (`b29c606e2`). Each
//! declaration names what it ports and the line it began at.
//!
//! The C++'s abstract `mtmd_gen_audio_pipeline` and its two subclasses are a
//! tagged union, `Pipeline`, dispatched where the C++ calls a virtual.

const std = @import("std");
const impl = @import("impl.zig");
const mtmd = @import("mtmd.zig");
const helper = @import("helper.zig");
const lapi = @import("llama").api;
const c = impl.c;
const gpa = impl.gpa;

const Context = mtmd.Context;
const Token = c.llama_token;
const token_null: Token = -1; // LLAMA_TOKEN_NULL
const EmbdBatch = helper.EmbdBatch;
const logInf = helper.logInf;
const logWrn = helper.logWrn;
const logErr = helper.logErr;

// -----------------------------------------------------------------------------
// mtmd-helper.h

/// Ports `mtmd_helper_gen_audio_outtype` (mtmd-helper.h:192 @b29c606e2).
pub const OutType = enum(c_int) {
    pcm, // raw PCM
    wav, // WAV PCM 16-bit LE, mono
    _,
};

/// Ports `mtmd_helper_gen_audio_inp` (mtmd-helper.h:196 @b29c606e2).
pub const Inp = extern struct {
    seq_id: c.llama_seq_id,
    prompt: ?[*]const u8,
    prompt_len: usize,
    speaker_ref: ?*mtmd.Bitmap, // optional, can be NULL
    lang: ?[*:0]const u8, // optional, can be NULL
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

/// Ports `tts_lang_codes` and `tts_resolve_lang` (mtmd-helper-gen.cpp:25,
/// 38 @b29c606e2): a `--tts-lang` code to the language name the
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

/// Ports `find_special_token` (mtmd-helper-gen.cpp:43 @b29c606e2): the
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

/// Ports `write_wav16` (mtmd-helper-gen.cpp:53 @b29c606e2): 16-bit mono WAV.
fn writeWav16(buf: *std.ArrayList(u8), pcm: []const f32, rate: i32) bool {
    // RIFF chunk sizes are 32-bit; refuse to emit a file with a truncated header
    if (pcm.len > (@as(usize, std.math.maxInt(u32)) - 36) / 2) return false;
    const data_sz: u32 = @truncate(pcm.len * 2);
    const riff_sz: u32 = 36 +% data_sz;
    const fmt_sz: u32 = 16;
    const byte_rate: u32 = @as(u32, @bitCast(rate)) *% 2;
    const fmt: u16 = 1;
    const ch: u16 = 1;
    const alignment: u16 = 2;
    const bits: u16 = 16;
    const rate32: u32 = @bitCast(rate);
    const put = struct {
        fn f(b: *std.ArrayList(u8), bytes: []const u8) void {
            b.appendSlice(gpa, bytes) catch oom();
        }
    }.f;
    put(buf, "RIFF");
    put(buf, std.mem.asBytes(&riff_sz));
    put(buf, "WAVE");
    put(buf, "fmt ");
    put(buf, std.mem.asBytes(&fmt_sz));
    put(buf, std.mem.asBytes(&fmt));
    put(buf, std.mem.asBytes(&ch));
    put(buf, std.mem.asBytes(&rate32));
    put(buf, std.mem.asBytes(&byte_rate));
    put(buf, std.mem.asBytes(&alignment));
    put(buf, std.mem.asBytes(&bits));
    put(buf, "data");
    put(buf, std.mem.asBytes(&data_sz));
    buf.ensureUnusedCapacity(gpa, pcm.len * 2) catch oom();
    for (pcm) |v| {
        const s: i16 = @intFromFloat(stdMax(-1.0, stdMin(1.0, v)) * 32767.0);
        buf.appendSliceAssumeCapacity(std.mem.asBytes(&s));
    }
    return true;
}

/// Ports `mtmd_gen_audio_pipeline` (mtmd-helper-gen.cpp:79 @b29c606e2): what
/// both pipelines hold.
const Base = struct {
    lctx: ?*c.struct_llama_context,
    mctx: *Context,
    model: ?*const c.struct_llama_model,
    vocab: ?*const c.struct_llama_vocab,
    n_embd: c_int,
    info: mtmd.GenAudioInfo,

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

    fn nE(self: *const Base) usize {
        return @intCast(self.n_embd);
    }

    /// The token embedding matrix, F32: the shared half of `ensure_cache` and
    /// `ensure_cache` (mtmd-helper-gen.cpp:331, 693 @b29c606e2), one per pipeline.
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

    /// Ports `encode_speaker` and `encode_speaker` (mtmd-helper-gen.cpp:368,
    /// 907 @b29c606e2), one per pipeline: the reference audio through the mmproj's audio encoder.
    /// They differ only in their messages.
    fn encodeSpeaker(self: *const Base, bitmap: *mtmd.Bitmap, out: *std.ArrayList(f32), comptime no_encoder: [*:0]const u8, comptime encode_failed: [*:0]const u8) bool {
        if (!mtmd.mtmd_support_audio(self.mctx)) {
            logErr(no_encoder, .{});
            return false;
        }
        const marker = std.mem.span(mtmd.mtmd_default_marker());
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

    /// Ports `step_prompt` and `step_prompt` (mtmd-helper-gen.cpp:219, 583
    /// @b29c606e2), the same in both pipelines: at most `n_batch` tokens of the prompt batch.
    fn stepPrompt(self: *const Base, st: *PromptState, pos: *c_int, n_batch: i32) i32 {
        impl.ggmlAssert(n_batch > 0, "mtmd-helper-gen.cpp", 220, "n_batch > 0");
        if (st.prompt_pos >= st.n_prompt) return 0;
        const n_tokens_batch = @min(n_batch, st.n_prompt - st.prompt_pos);
        const batch_view = st.prompt_batch.?.getView(st.prompt_pos, n_tokens_batch);

        const is_last_batch = (st.prompt_pos + n_tokens_batch) == st.n_prompt;
        if (is_last_batch) batch_view.logits[@intCast(n_tokens_batch - 1)] = 1;

        if (lapi.context.llama_decode(self.lctx, batch_view) != 0) {
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

    /// Ports `get_output` and `get_output` (mtmd-helper-gen.cpp:304, 666
    /// @b29c606e2), one per pipeline, after their flush.
    fn getOutput(self: *const Base, audio_pcm: *const std.ArrayList(f32), out_type: OutType, out_buf: *std.ArrayList(u8), out_sample_rate: *i32, out_data: *?[*]const u8, out_data_len: *usize, out_n_samples: ?*i64) i32 {
        out_sample_rate.* = self.info.sample_rate;
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

/// Ports `qwen3tts_gen_audio_pipeline` (mtmd-helper-gen.cpp:107 @b29c606e2):
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
    seed: u32 = std.math.maxInt(u32),
    codes_buf: std.ArrayList(i32) = .empty,
    c2w_state: std.ArrayList(u8) = .empty,
    audio_pcm: std.ArrayList(f32) = .empty,
    overlay: std.ArrayList(f32) = .empty,
    h_state_buf: std.ArrayList(f32) = .empty,
    out_type: OutType = .wav,
    out_buf: std.ArrayList(u8) = .empty,

    fn deinit(self: *Qwen3tts) void {
        self.tok_embd.deinit(gpa);
        self.prompt.deinit();
        self.codes_buf.deinit(gpa);
        self.c2w_state.deinit(gpa);
        self.audio_pcm.deinit(gpa);
        self.overlay.deinit(gpa);
        self.h_state_buf.deinit(gpa);
        self.out_buf.deinit(gpa);
    }

    /// `reset` (mtmd-helper-gen.cpp:111 @b29c606e2).
    fn reset(self: *Qwen3tts) void {
        self.seq_id = 0;
        self.pos = 0;
        self.codes_buf.clearRetainingCapacity();
        self.c2w_state.clearRetainingCapacity();
        self.audio_pcm.clearRetainingCapacity();
        self.overlay.clearRetainingCapacity();
        self.h_state_buf.clearRetainingCapacity();
        self.out_buf.clearRetainingCapacity();
        self.prompt.reset();
    }

    fn row(self: *const Qwen3tts, t: Token) []const f32 {
        const n_e = self.base.nE();
        const k: usize = @intCast(t);
        return self.tok_embd.items[k * n_e .. (k + 1) * n_e];
    }

    /// `sum_row` and `sum_vec` (mtmd-helper-gen.cpp:153, 158 @b29c606e2),
    /// appended to `dst`.
    fn pushSum(self: *const Qwen3tts, dst: *std.ArrayList(f32), a: Token, vb: []const f32) void {
        const va = self.row(a);
        dst.ensureUnusedCapacity(gpa, va.len) catch oom();
        for (va, 0..) |x, i| dst.appendAssumeCapacity(x + vb[i]);
    }

    /// `set_input` (mtmd-helper-gen.cpp:126 @b29c606e2).
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
        self.seed = inp.seed;
        self.out_type = inp.out_type;

        // the prompt above holds the whole text stream up to tts_eos, so every generated
        // frame adds tts_pad on top of the codes embedding
        assign(f32, &self.overlay, self.row(self.tts_pad).ptr, n_e);

        return 0;
    }

    /// `step_gen` (mtmd-helper-gen.cpp:249 @b29c606e2).
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
        inp.seed = self.seed;
        var out = std.mem.zeroes(mtmd.GenOut);
        if (mtmd.mtmd_gen_audio_process(self.base.mctx, &inp, &out) != 0) {
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
        batch_embd.batch.logits[0] = 1;
        self.pos += 1;

        if (lapi.context.llama_decode(self.base.lctx, batch_embd.batch) != 0) {
            logErr("mtmd_helper_gen_audio: decode failed\n", .{});
            return 1;
        }

        h_state_out.* = self.base.takeHState(&self.h_state_buf);
        return 0;
    }

    /// `get_output` (mtmd-helper-gen.cpp:304 @b29c606e2).
    fn getOutput(self: *Qwen3tts, out_sample_rate: *i32, out_data: *?[*]const u8, out_data_len: *usize, out_n_samples: ?*i64) i32 {
        if (!self.flushGenWav()) return 1;
        return self.base.getOutput(&self.audio_pcm, self.out_type, &self.out_buf, out_sample_rate, out_data, out_data_len, out_n_samples);
    }

    /// `ensure_cache` (mtmd-helper-gen.cpp:331 @b29c606e2).
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

    /// `flush_gen_wav` (mtmd-helper-gen.cpp:401 @b29c606e2): one GEN_WAV
    /// call over the buffered codes, state carried across batches.
    fn flushGenWav(self: *Qwen3tts) bool {
        if (self.codes_buf.items.len == 0) return true;
        var inp = mtmd.mtmd_gen_inp_default(self.base.mctx);
        inp.type = .gen_wav;
        inp.codes = self.codes_buf.items.ptr;
        inp.n_codes = self.codes_buf.items.len;
        inp.seed = self.seed; // same seed as gen_code, else clip reseeds mid-generation
        inp.state_data = if (self.c2w_state.items.len == 0) null else self.c2w_state.items.ptr;
        inp.state_size = self.c2w_state.items.len;
        var out = std.mem.zeroes(mtmd.GenOut);
        if (mtmd.mtmd_gen_audio_process(self.base.mctx, &inp, &out) != 0) {
            logErr("mtmd_helper_gen_audio: gen_wav process failed\n", .{});
            return false;
        }
        if (out.n_samples != 0) self.audio_pcm.appendSlice(gpa, out.audio.?[0..out.n_samples]) catch oom();
        assign(u8, &self.c2w_state, out.state_data, out.state_size);
        self.codes_buf.clearRetainingCapacity();
        return true;
    }
};

// -----------------------------------------------------------------------------
// pocket-tts

/// Ports `pockettts_pack_settings` (mtmd-helper-gen.cpp:464 @b29c606e2):
/// settings that live only in the reference's per-pack yaml.
const PackSettings = struct {
    temp: f32 = 0.0,
    frames_after_eos: c_int = 0,
    pad_short_text: bool = false,
};

/// Ports `pockettts_pack` (mtmd-helper-gen.cpp:470 @b29c606e2).
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

/// Ports `pockettts_gen_audio_pipeline` (mtmd-helper-gen.cpp:488
/// @b29c606e2): the backbone emits no token; the flow net turns each hidden
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
    dec_state: std.ArrayList(u8) = .empty,
    audio_pcm: std.ArrayList(f32) = .empty,
    h_state_buf: std.ArrayList(f32) = .empty,
    out_type: OutType = .wav,
    out_buf: std.ArrayList(u8) = .empty,

    const max_chunk_tokens: c_int = 50; // MAX_TOKEN_PER_CHUNK in the reference
    const frame_rate: f64 = 12.5;

    fn deinit(self: *Pockettts) void {
        self.tok_embd.deinit(gpa);
        self.prompt.deinit();
        freeLists(&self.chunks);
        self.feats_buf.deinit(gpa);
        self.dec_state.deinit(gpa);
        self.audio_pcm.deinit(gpa);
        self.h_state_buf.deinit(gpa);
        self.out_buf.deinit(gpa);
    }

    /// `reset` (mtmd-helper-gen.cpp:492 @b29c606e2).
    fn reset(self: *Pockettts) void {
        self.seq_id = 0;
        self.pos = 0;
        self.feats_buf.clearRetainingCapacity();
        self.dec_state.clearRetainingCapacity();
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
    }

    /// `set_input` (mtmd-helper-gen.cpp:512 @b29c606e2).
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
            impl.ggmlAssert(voice.items.len % n_e == 0, "mtmd-helper-gen.cpp", 557, "voice.size() % (size_t) n_e == 0");
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

        self.seed = inp.seed;
        self.out_type = inp.out_type;

        return 0;
    }

    /// `step_gen` (mtmd-helper-gen.cpp:611 @b29c606e2). `sampled` is unused:
    /// the backbone output is continuous.
    fn stepGen(self: *Pockettts, h_state_in: ?[*]const f32, h_state_out: *?[*]const f32, out_stop: *bool) i32 {
        var inp = mtmd.mtmd_gen_inp_default(self.base.mctx);
        inp.type = .gen_code;
        inp.embd = @constCast(h_state_in);
        // clip only reseeds when the seed changes, so pass the same one on every step
        inp.seed = self.seed;
        if (self.pack.temp > 0.0) inp.temp = self.pack.temp;
        var out = std.mem.zeroes(mtmd.GenOut);
        if (mtmd.mtmd_gen_audio_process(self.base.mctx, &inp, &out) != 0) {
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

        const batch_embd = EmbdBatch.create(@constCast(out.embd), 1, 1, self.base.n_embd);
        defer batch_embd.destroy();
        batch_embd.setPositionNormal(self.pos, self.seq_id);
        batch_embd.batch.logits[0] = 1;
        self.pos += 1;

        if (lapi.context.llama_decode(self.base.lctx, batch_embd.batch) != 0) {
            logErr("mtmd_helper_gen_audio: decode failed\n", .{});
            return 1;
        }

        h_state_out.* = self.base.takeHState(&self.h_state_buf);
        return 0;
    }

    /// `get_output` (mtmd-helper-gen.cpp:666 @b29c606e2).
    fn getOutput(self: *Pockettts, out_sample_rate: *i32, out_data: *?[*]const u8, out_data_len: *usize, out_n_samples: ?*i64) i32 {
        if (!self.flushGenWav()) return 1;
        return self.base.getOutput(&self.audio_pcm, self.out_type, &self.out_buf, out_sample_rate, out_data, out_data_len, out_n_samples);
    }

    /// `ensure_cache` (mtmd-helper-gen.cpp:693 @b29c606e2).
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
        impl.ggmlAssert(self.base.n_embd > 0 and n_tok_embd % @as(u32, @bitCast(self.base.n_embd)) == 0, "mtmd-helper-gen.cpp", 714, "n_embd > 0 && n_tok_embd % (uint32_t) n_embd == 0");
        self.specials_ok = true;
        return true;
    }

    /// `push_embd_row` (mtmd-helper-gen.cpp:720 @b29c606e2): the table can
    /// be shorter than the vocab, so the row lookup is bounded.
    fn pushEmbdRow(self: *const Pockettts, dst: *std.ArrayList(f32), t: Token) void {
        const n_e = self.base.nE();
        const n_rows = self.tok_embd.items.len / n_e;
        impl.ggmlAssert(t >= 0 and @as(usize, @intCast(t)) < n_rows, "mtmd-helper-gen.cpp", 722, "t >= 0 && (size_t) t < n_rows");
        const k: usize = @intCast(t);
        dst.appendSlice(gpa, self.tok_embd.items[k * n_e .. (k + 1) * n_e]) catch oom();
    }

    /// `punct_ids` (mtmd-helper-gen.cpp:730 @b29c606e2): the token ids of
    /// the pieces the reference splits on, without the dummy prefix.
    fn punctIds(self: *const Pockettts, s: [:0]const u8) TokenList {
        var ids: [16]Token = @splat(0);
        const n = lapi.vocab.llama_tokenize(self.base.vocab, s.ptr, @intCast(s.len), &ids, ids.len, false, false);
        var out: TokenList = .empty;
        if (n <= 1) return out;
        out.appendSlice(gpa, ids[1..@intCast(n)]) catch oom();
        return out;
    }

    /// `split_on` (mtmd-helper-gen.cpp:740 @b29c606e2): cut after runs of
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

    /// `split_chunks` (mtmd-helper-gen.cpp:757 @b29c606e2).
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

    /// `arm_chunk_budget` (mtmd-helper-gen.cpp:805 @b29c606e2):
    /// `_estimate_max_gen_len()` plus the per-chunk tail guess, in frames.
    fn armChunkBudget(self: *Pockettts, idx: usize) void {
        const n_tok: c_int = @intCast(self.chunks.items[idx].items.len);
        self.chunk_budget = @intFromFloat(@ceil((@as(f64, @floatFromInt(n_tok)) / 3.0 + 2.0) * frame_rate));
        // the pack may pin the tail, else the reference guesses it from the word count
        self.frames_after_eos = if (self.pack.frames_after_eos > 0) self.pack.frames_after_eos else if (n_tok <= 6) 5 else 3;
        self.step_idx = 0;
        self.eos_step = -1;
    }

    /// `finish_chunk` (mtmd-helper-gen.cpp:815 @b29c606e2): ends the chunk
    /// and re-prompts the next on top of the voice.
    fn finishChunk(self: *Pockettts, h_state_out: *?[*]const f32, out_stop: *bool) i32 {
        if (!self.flushGenWav()) return 1;
        // the decoder restarts too, the next chunk's audio is not continuous with this one
        self.dec_state.clearRetainingCapacity();

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
        impl.ggmlAssert(n_rows > 0, "mtmd-helper-gen.cpp", 842, "n_rows > 0");
        const batch = EmbdBatch.create(buf.items.ptr, n_rows, 1, self.base.n_embd);
        defer batch.destroy();
        batch.setPositionNormal(self.pos, self.seq_id);
        batch.batch.logits[@intCast(n_rows - 1)] = 1;
        if (lapi.context.llama_decode(self.base.lctx, batch.batch) != 0) {
            logErr("mtmd_helper_gen_audio: chunk prompt decode failed\n", .{});
            return 1;
        }
        self.pos += n_rows;
        buf.clearRetainingCapacity();

        h_state_out.* = self.base.takeHState(&self.h_state_buf);
        out_stop.* = false;
        return 0;
    }

    /// `prepare_text` (mtmd-helper-gen.cpp:861 @b29c606e2): the reference's
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

    /// `count_words` (mtmd-helper-gen.cpp:892 @b29c606e2).
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

    /// `flush_gen_wav` (mtmd-helper-gen.cpp:940 @b29c606e2): decodes the
    /// buffered latents, the mimi decoder state carrying over.
    fn flushGenWav(self: *Pockettts) bool {
        if (self.feats_buf.items.len == 0) return true;
        var inp = mtmd.mtmd_gen_inp_default(self.base.mctx);
        inp.type = .gen_wav;
        inp.feats = self.feats_buf.items.ptr;
        inp.n_feats = self.feats_buf.items.len;
        inp.seed = self.seed;
        inp.state_data = if (self.dec_state.items.len == 0) null else self.dec_state.items.ptr;
        inp.state_size = self.dec_state.items.len;
        var out = std.mem.zeroes(mtmd.GenOut);
        if (mtmd.mtmd_gen_audio_process(self.base.mctx, &inp, &out) != 0) {
            logErr("mtmd_helper_gen_audio: mimi decode failed\n", .{});
            return false;
        }
        if (out.n_samples != 0) self.audio_pcm.appendSlice(gpa, out.audio.?[0..out.n_samples]) catch oom();
        assign(u8, &self.dec_state, out.state_data, out.state_size);
        self.feats_buf.clearRetainingCapacity();
        return true;
    }
};

/// The C++'s `mtmd_gen_audio_pipeline` hierarchy, one arm a subclass.
const Pipeline = union(enum) {
    qwen3tts: Qwen3tts,
    pockettts: Pockettts,

    fn deinit(self: *Pipeline) void {
        switch (self.*) {
            inline else => |*p| p.deinit(),
        }
    }
};

/// Ports `mtmd_helper_gen_audio` (mtmd-helper-gen.cpp:1007 @b29c606e2).
pub const GenAudio = struct {
    pipeline: ?Pipeline = null,
};

/// Ports `make_pipeline` (mtmd-helper-gen.cpp:996 @b29c606e2).
fn makePipeline(lctx: ?*c.struct_llama_context, mctx: *Context) ?Pipeline {
    return switch (mtmd.mtmd_gen_audio_get_info(mctx).type) {
        .qwen3tts => .{ .qwen3tts = .{ .base = Base.init(lctx, mctx) } },
        .pockettts => .{ .pockettts = .{ .base = Base.init(lctx, mctx) } },
        else => null,
    };
}

/// Ports `mtmd_helper_gen_audio_init` (mtmd-helper-gen.cpp:1011 @b29c606e2).
pub fn mtmd_helper_gen_audio_init(lctx: ?*c.struct_llama_context, mctx: *Context) callconv(.c) *GenAudio {
    const ctx = gpa.create(GenAudio) catch oom();
    ctx.* = .{ .pipeline = makePipeline(lctx, mctx) };
    return ctx;
}

/// Ports `mtmd_helper_gen_audio_free` (mtmd-helper-gen.cpp:1017 @b29c606e2).
pub fn mtmd_helper_gen_audio_free(ctx: ?*GenAudio) callconv(.c) void {
    const x = ctx orelse return;
    if (x.pipeline) |*p| p.deinit();
    gpa.destroy(x);
}

/// Ports `mtmd_helper_gen_audio_reset` (mtmd-helper-gen.cpp:1021 @b29c606e2).
pub fn mtmd_helper_gen_audio_reset(ctx: *GenAudio) callconv(.c) void {
    if (ctx.pipeline) |*p| switch (p.*) {
        inline else => |*q| q.reset(),
    };
}

/// Ports `mtmd_helper_gen_audio_set_input` (mtmd-helper-gen.cpp:1027
/// @b29c606e2).
pub fn mtmd_helper_gen_audio_set_input(ctx: *GenAudio, inp: *const Inp) callconv(.c) i32 {
    const p = if (ctx.pipeline) |*p| p else {
        logErr("mtmd_helper_gen_audio: unsupported or missing gen-audio pipeline\n", .{});
        return 1;
    };
    return switch (p.*) {
        inline else => |*q| q.setInput(inp),
    };
}

/// Ports `mtmd_helper_gen_audio_step_prompt` (mtmd-helper-gen.cpp:1035
/// @b29c606e2): >0 prompt tokens remaining, 0 done, <0 an error.
pub fn mtmd_helper_gen_audio_step_prompt(ctx: *GenAudio, n_batch: i32) callconv(.c) i32 {
    const p = if (ctx.pipeline) |*p| p else return -1;
    return switch (p.*) {
        inline else => |*q| q.base.stepPrompt(&q.prompt, &q.pos, n_batch),
    };
}

/// Ports `mtmd_helper_gen_audio_step_gen` (mtmd-helper-gen.cpp:1042
/// @b29c606e2).
pub fn mtmd_helper_gen_audio_step_gen(ctx: *GenAudio, sampled: Token, h_state_in: ?[*]const f32, h_state_out: *?[*]const f32, out_stop: ?*bool) callconv(.c) i32 {
    const p = if (ctx.pipeline) |*p| p else return 1;
    var stop = false;
    const ret = switch (p.*) {
        .qwen3tts => |*q| q.stepGen(sampled, h_state_in, h_state_out, &stop),
        .pockettts => |*q| q.stepGen(h_state_in, h_state_out, &stop),
    };
    if (out_stop) |s| s.* = stop;
    return ret;
}

/// Ports `mtmd_helper_gen_audio_get_output` (mtmd-helper-gen.cpp:1056
/// @b29c606e2).
pub fn mtmd_helper_gen_audio_get_output(ctx: *GenAudio, out_sample_rate: *i32, out_data: *?[*]const u8, out_data_len: *usize, out_n_samples: ?*i64) callconv(.c) i32 {
    const p = if (ctx.pipeline) |*p| p else return 1;
    return switch (p.*) {
        inline else => |*q| q.getOutput(out_sample_rate, out_data, out_data_len, out_n_samples),
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
