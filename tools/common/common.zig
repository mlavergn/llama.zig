//! `common_init` and the token helpers of `common/common.cpp`.
//!
//! # Provenance
//!
//! Ports `common_init`, `common_tokenize`, `common_token_to_piece`,
//! `common_batch`, the typed decision models and the rest of what the tools
//! reach from `llama.cpp/common/common.cpp` and `llama.cpp/common/common.h`,
//! and `common_log_get_verbosity` and `common_log_default_callback` from
//! `common/log.cpp` (v0.6.0, `d81235049`). Each declaration names what it
//! ports and the line it began at.

const std = @import("std");
const builtin = @import("builtin");
const c = @import("tools_h");
const log = @import("log.zig");
const arg = @import("arg.zig");
const Params = @import("params.zig").Params;
const llama = @import("llama");

extern fn SetConsoleOutputCP(cp: c_uint) callconv(.winapi) c_int;
extern fn SetConsoleCP(cp: c_uint) callconv(.winapi) c_int;

/// Ports `common_log_get_verbosity` (log.cpp:532 @d81235049), which v0.4.0
/// exports from `log.h`: the `common_log` verbosity a ggml log level maps to.
pub fn logGetVerbosity(level: c.ggml_log_level) c_int {
    return switch (level) {
        c.GGML_LOG_LEVEL_DEBUG => log.level_debug,
        c.GGML_LOG_LEVEL_INFO => log.level_trace,
        c.GGML_LOG_LEVEL_WARN => log.level_warn,
        c.GGML_LOG_LEVEL_ERROR => log.level_error,
        c.GGML_LOG_LEVEL_CONT => log.level_trace,
        else => log.level_output,
    };
}

/// Ports `common_log_default_callback` (log.cpp:545 @d81235049): libllama's
/// messages through `common_log`, at `common_log_get_verbosity`'s level.
pub fn defaultLogCallback(level: c.ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    if (logGetVerbosity(level) <= log.verbosity_thold) log.add(@fromBackingInt(@intCast(level)), "%s", .{text});
}

/// Ports `common_init` (common/common.cpp:388 @d81235049).
pub fn init() void {
    if (comptime builtin.os.tag == .windows) {
        _ = SetConsoleOutputCP(65001);
        _ = SetConsoleCP(65001);
    }

    log.setPrefix(true);
    log.setTimestamps(true);

    c.llama_log_set(defaultLogCallback, null);
}

/// Ports `common_tokenize` (common/common.cpp:1813 @d81235049).
///
/// Return: the tokens, the allocator's.
pub fn tokenize(a: std.mem.Allocator, vocab: ?*const c.llama_vocab, text: []const u8, add_special: bool, parse_special: bool) ![]c.llama_token {
    // upper limit for the number of tokens
    var n_tokens: c_int = @intCast(text.len + 2 * @as(usize, @intFromBool(add_special)));
    var result = try a.alloc(c.llama_token, @intCast(n_tokens));
    n_tokens = c.llama_tokenize(vocab, text.ptr, @intCast(text.len), result.ptr, @intCast(result.len), add_special, parse_special);
    if (n_tokens == std.math.minInt(i32)) {
        return arg.throwStr("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
    }
    if (n_tokens < 0) {
        result = try a.realloc(result, @intCast(-n_tokens));
        const check = c.llama_tokenize(vocab, text.ptr, @intCast(text.len), result.ptr, @intCast(result.len), add_special, parse_special);
        std.debug.assert(check == -n_tokens);
        return result;
    }
    return result[0..@intCast(n_tokens)];
}

/// Ports `common_token_to_piece` (common/common.cpp:1841 @d81235049): the first
/// try fits libc++'s short-string buffer, 22 bytes.
pub fn tokenToPiece(a: std.mem.Allocator, vocab: ?*const c.llama_vocab, token: c.llama_token, special: bool) ![:0]u8 {
    var piece = try a.alloc(u8, 22 + 1);
    const n_chars = c.llama_token_to_piece(vocab, token, piece.ptr, 22, 0, special);
    if (n_chars < 0) {
        piece = try a.realloc(piece, @intCast(-n_chars + 1));
        const check = c.llama_token_to_piece(vocab, token, piece.ptr, -n_chars, 0, special);
        std.debug.assert(check == -n_chars);
        piece[@intCast(-n_chars)] = 0;
        return piece[0..@intCast(-n_chars) :0];
    }
    piece[@intCast(n_chars)] = 0;
    return piece[0..@intCast(n_chars) :0];
}

/// Ports `common_detokenize` (common/common.cpp:1863 @d81235049): the first try
/// fits libc++'s short-string buffer, 22 bytes, or the token count.
///
/// Return: the text, the allocator's.
pub fn detokenize(a: std.mem.Allocator, vocab: ?*const c.llama_vocab, tokens: []const c.llama_token, special: bool) ![]u8 {
    var text = try a.alloc(u8, @max(22, tokens.len));
    var n_chars = c.llama_detokenize(vocab, tokens.ptr, @intCast(tokens.len), text.ptr, @intCast(text.len), false, special);
    if (n_chars < 0) {
        text = try a.realloc(text, @intCast(-n_chars));
        n_chars = c.llama_detokenize(vocab, tokens.ptr, @intCast(tokens.len), text.ptr, @intCast(text.len), false, special);
        // whitespace trimming is performed after per-token detokenization
        if (n_chars > @as(i32, @intCast(text.len))) _ = c.ggml_abort("common.cpp", 1870, "GGML_ASSERT(%s) failed", "n_chars <= (int32_t)text.size()");
    }
    return text[0..@intCast(n_chars)];
}

/// Ports `common_model_params_to_llama` (common/common.cpp:1649 @d81235049).
pub fn modelParamsToLlama(p: *Params) c.llama_model_params {
    var mparams = c.llama_model_default_params();

    if (p.devices.items.len != 0) mparams.devices = p.devices.items.ptr;

    mparams.n_gpu_layers = p.n_gpu_layers;
    mparams.main_gpu = p.main_gpu;
    mparams.split_mode = p.split_mode;
    mparams.load_mode = p.load_mode;
    mparams.lazy_mode = p.lazy_mode;
    mparams.tensor_split = &p.tensor_split;
    mparams.check_tensors = p.check_tensors;
    mparams.use_extra_bufts = !p.no_extra_bufts;
    mparams.no_host = p.no_host;

    if (p.kv_overrides.items.len == 0) {
        mparams.kv_overrides = null;
    } else {
        std.debug.assert(p.kv_overrides.items[p.kv_overrides.items.len - 1].key[0] == 0);
        mparams.kv_overrides = p.kv_overrides.items.ptr;
    }

    if (p.tensor_buft_overrides.items.len == 0) {
        mparams.tensor_buft_overrides = null;
    } else {
        std.debug.assert(p.tensor_buft_overrides.items[p.tensor_buft_overrides.items.len - 1].pattern == null);
        mparams.tensor_buft_overrides = p.tensor_buft_overrides.items.ptr;
    }

    mparams.progress_callback = p.load_progress_callback;
    mparams.progress_callback_user_data = p.load_progress_callback_user_data;
    mparams.no_alloc = p.no_alloc;
    mparams.load_mtp = p.load_mtp or for (p.speculative.types.items) |t| {
        if (t == .draft_mtp) break true;
    } else false;

    return mparams;
}

/// Ports `common_context_params_to_llama` (common/common.cpp:1688 @d81235049).
pub fn contextParamsToLlama(p: *const Params) c.llama_context_params {
    var cparams = c.llama_context_default_params();

    cparams.n_ctx = @bitCast(p.n_ctx);
    cparams.n_seq_max = @bitCast(p.n_parallel);
    cparams.n_rs_seq = p.speculative.needNRsSeq();
    cparams.n_outputs_max = @intCast(@max(p.n_outputs_max, 0));
    cparams.n_outputs_max_per_seq = @intCast(@max(p.n_outputs_max_per_seq, 0));
    cparams.n_batch = @bitCast(p.n_batch);
    cparams.n_ubatch = @bitCast(p.n_ubatch);
    cparams.n_threads = p.cpuparams.n_threads;
    cparams.n_threads_batch = if (p.cpuparams_batch.n_threads == -1) p.cpuparams.n_threads else p.cpuparams_batch.n_threads;
    cparams.embeddings = p.embedding;
    cparams.rope_scaling_type = p.rope_scaling_type;
    cparams.rope_freq_base = p.rope_freq_base;
    cparams.rope_freq_scale = p.rope_freq_scale;
    cparams.yarn_ext_factor = p.yarn_ext_factor;
    cparams.yarn_attn_factor = p.yarn_attn_factor;
    cparams.yarn_beta_fast = p.yarn_beta_fast;
    cparams.yarn_beta_slow = p.yarn_beta_slow;
    cparams.yarn_orig_ctx = @bitCast(p.yarn_orig_ctx);
    cparams.pooling_type = p.pooling_type;
    cparams.attention_type = p.attention_type;
    cparams.flash_attn_type = p.flash_attn_type;
    cparams.cb_eval = p.cb_eval;
    cparams.cb_eval_user_data = p.cb_eval_user_data;
    cparams.offload_kqv = !p.no_kv_offload;
    cparams.no_perf = p.no_perf;
    cparams.op_offload = !p.no_op_offload;
    cparams.swa_full = p.swa_full;
    cparams.kv_unified = p.kv_unified;

    cparams.type_k = p.cache_type_k;
    cparams.type_v = p.cache_type_v;

    return cparams;
}

/// Ports `common_batch` (common/common.h:1061 @d81235049): a wrapper around
/// `llama_batch_ext` holding its entries, which may exceed `n_batch`; render
/// them with `get` or `getSubBatch`, and decode them in chunks.
pub const Batch = struct {
    /// Ports `token` (common/common.h:1062 @d81235049), a member of
    /// `common_batch`.
    pub const Token = struct {
        id: c.llama_token,
        pos: [c.GGML_MROPE_SECTIONS]c.llama_pos, // only pos[0] is used for text tokens
        seq_id: c.llama_seq_id, // the first sequence id, see addSeq()
        output: bool,
        embd: c.llama_embd, // non-owning view of the data passed to addEmbd()/setEmbd(), data == null if none
        seq_ids_extra: std.ArrayList(c.llama_seq_id) = .empty, // see addSeq()
        decision_order: i32 = 0, // see llama_batch_ext_set_decision_order()
    };

    a: std.mem.Allocator,
    tokens: std.ArrayList(Token) = .empty, // mirror of the entries, tokens[i] describes batch index i
    batch: ?*c.llama_batch_ext = null,

    n_pos: i32 = 1, // positions per embedding entry, GGML_MROPE_SECTIONS for MROPE/IMROPE

    /// A batch with no context, which `getSubBatch` refuses: upstream's
    /// default constructor.
    pub fn initEmpty(a: std.mem.Allocator) Batch {
        return .{ .a = a };
    }

    /// Ports `common_batch` (common/common.cpp:2145 @d81235049), the
    /// constructor over a context.
    pub fn init(a: std.mem.Allocator, ctx: ?*c.llama_context) Batch {
        const rope_type = c.llama_model_rope_type(c.llama_get_model(ctx));
        return .{
            .a = a,
            .batch = c.llama_batch_ext_init(ctx),
            .n_pos = if (rope_type == c.LLAMA_ROPE_TYPE_MROPE or rope_type == c.LLAMA_ROPE_TYPE_IMROPE) c.GGML_MROPE_SECTIONS else 1,
        };
    }

    /// `llama_batch_ext_ptr`'s deleter and the vectors' destructors.
    pub fn deinit(self: *Batch) void {
        self.clear();
        self.tokens.deinit(self.a);
        if (self.batch) |b| c.llama_batch_ext_free(b);
        self.batch = null;
    }

    pub fn get(self: *Batch) ?*c.llama_batch_ext {
        return self.getSubBatch(0, self.size());
    }

    /// content type of the batch, all entries carry the same combination
    pub fn hasToken(self: *const Batch) bool {
        return self.tokens.items.len != 0 and self.tokens.items[0].id != c.LLAMA_TOKEN_NULL;
    }
    pub fn hasEmbd(self: *const Batch) bool {
        return self.tokens.items.len != 0 and self.tokens.items[0].embd.data != null;
    }

    pub fn size(self: *const Batch) i32 {
        return @intCast(self.tokens.items.len);
    }

    /// Ports `clear` (common/common.cpp:2150 @d81235049).
    pub fn clear(self: *Batch) void {
        for (self.tokens.items) |*t| t.seq_ids_extra.deinit(self.a);
        self.tokens.clearRetainingCapacity();
    }

    /// Ports `add` (common/common.cpp:2154 @d81235049): the batch index.
    pub fn add(self: *Batch, id: c.llama_token, pos: c.llama_pos, seq_id: c.llama_seq_id, output: bool) i32 {
        self.tokens.append(self.a, .{ .id = id, .pos = .{ pos, 0, 0, 0 }, .seq_id = seq_id, .output = output, .embd = .{ .data = null, .n_rows = 0, .n_embd = 0 } }) catch oom();
        return self.size() - 1;
    }

    /// Ports `add` (common/common.cpp:2159 @d81235049), the overload with
    /// the entry shared by all of `seq_ids`, which must not be empty.
    pub fn addSeqs(self: *Batch, id: c.llama_token, pos: c.llama_pos, seq_ids: []const c.llama_seq_id, output: bool) i32 {
        if (seq_ids.len == 0) _ = c.ggml_abort("common.cpp", 2160, "GGML_ASSERT(%s) failed", "!seq_ids.empty()");

        const idx = self.add(id, pos, seq_ids[0], output);
        for (seq_ids[1..]) |s| _ = self.addSeq(idx, s);
        return idx;
    }

    /// Ports `add_seq` (common/common.cpp:2169 @d81235049): the entry at
    /// `idx` in another sequence too; `tokens[idx].seq_id` keeps the first.
    pub fn addSeq(self: *Batch, idx: i32, seq_id: c.llama_seq_id) bool {
        if (idx < 0 or idx >= self.size()) return false;
        self.tokens.items[@intCast(idx)].seq_ids_extra.append(self.a, seq_id) catch oom();
        return true;
    }

    /// Ports `set_output` (common/common.cpp:2177 @d81235049).
    pub fn setOutput(self: *Batch, idx: i32, value: bool) bool {
        if (idx < 0 or idx >= self.size()) return false;
        self.tokens.items[@intCast(idx)].output = value;
        return true;
    }

    /// Ports `set_embd` (common/common.cpp:2185 @d81235049): a token
    /// embedding for the entry at `idx`, once.
    pub fn setEmbd(self: *Batch, idx: i32, embd: c.llama_embd) bool {
        if (idx < 0 or idx >= self.size() or self.tokens.items[@intCast(idx)].embd.data != null) return false;
        self.tokens.items[@intCast(idx)].embd = embd;
        return true;
    }

    /// Ports `add_embd` (common/common.cpp:2193 @d81235049): an
    /// embedding-only entry; `pos` points to `n_pos` positions.
    pub fn addEmbd(self: *Batch, embd: c.llama_embd, pos: [*]const c.llama_pos, seq_id: c.llama_seq_id, output: bool) i32 {
        var t: Token = .{ .id = c.LLAMA_TOKEN_NULL, .pos = .{ 0, 0, 0, 0 }, .seq_id = seq_id, .output = output, .embd = embd };
        for (0..@intCast(self.n_pos)) |j| t.pos[j] = pos[j];
        self.tokens.append(self.a, t) catch oom();
        return self.size() - 1;
    }

    /// Ports `get_sub_batch` (common/common.cpp:2202 @d81235049): entries
    /// `[off, off + n)` rendered into the `llama_batch_ext`, which the next
    /// call overwrites.
    pub fn getSubBatch(self: *Batch, off: i32, n: i32) ?*c.llama_batch_ext {
        const res = self.batch orelse {
            _ = c.ggml_abort("common.cpp", 2203, "GGML_ASSERT(%s) failed", "batch && \"common_batch was not initialized with a context\"");
            unreachable;
        };
        if (!(off >= 0 and n >= 0 and off + n <= self.size())) _ = c.ggml_abort("common.cpp", 2204, "GGML_ASSERT(%s) failed", "off >= 0 && n >= 0 && off + n <= size()");

        c.llama_batch_ext_clear(res);

        const func = "get_sub_batch";
        var i = off;
        while (i < off + n) : (i += 1) {
            const t = &self.tokens.items[@intCast(i)];

            var idx: i32 = undefined;
            if (t.id != c.LLAMA_TOKEN_NULL) {
                idx = c.llama_batch_ext_add_token(res, t.seq_id, t.id);
                if (idx < 0) {
                    _ = c.ggml_abort("common.cpp", 2216, "%s: failed to add token %d at index %d (error %d, n = %d)\n", func, t.id, i, idx, n);
                }
                _ = c.llama_batch_ext_set_pos(res, idx, &t.pos);
                if (t.embd.data != null and !c.llama_batch_ext_set_embd_token(res, idx, t.embd)) {
                    _ = c.ggml_abort("common.cpp", 2220, "%s: failed to set the embedding of token %d at index %d\n", func, t.id, i);
                }
            } else {
                idx = c.llama_batch_ext_add_embd(res, t.seq_id, t.embd);
                if (idx < 0) {
                    _ = c.ggml_abort("common.cpp", 2225, "%s: failed to add embedding at index %d (error %d, n = %d)\n", func, i, idx, n);
                }
                _ = c.llama_batch_ext_set_pos(res, idx, &t.pos);
            }
            if (idx != i - off) _ = c.ggml_abort("common.cpp", 2229, "GGML_ASSERT(%s) failed", "idx == i - off");

            for (t.seq_ids_extra.items) |seq_id| {
                if (!c.llama_batch_ext_add_seq(res, idx, seq_id)) {
                    _ = c.ggml_abort("common.cpp", 2233, "%s: failed to add seq %d to the entry at index %d\n", func, seq_id, i);
                }
            }
            if (t.output) {
                _ = c.llama_batch_ext_set_output_logits(res, idx, true);
            }
            if (t.decision_order != 0) {
                _ = llama.api.batch.llama_batch_ext_set_decision_order(@ptrCast(res), idx, @fromBackingInt(@intCast(@as(c_uint, @bitCast(t.decision_order)))));
            }
        }

        return res;
    }
};

/// Ports `common_batch_get_one` (common/common.cpp:2247 @d81235049): a
/// single-sequence batch of `tokens`, positions continuing from the memory,
/// output on the last.
pub fn batchGetOne(a: std.mem.Allocator, ctx: ?*c.llama_context, tokens: []const c.llama_token) Batch {
    var batch = Batch.init(a, ctx);

    const mem = c.llama_get_memory(ctx);
    var pos: c.llama_pos = c.llama_memory_seq_pos_max(mem, 0) + 1; // -1 + 1 == 0 when the memory is empty

    for (tokens, 0..) |t, i| {
        const output = i == tokens.len - 1;
        _ = batch.add(t, pos, 0, output);
        pos += 1;
    }

    return batch;
}

/// Ports `common_replay_last_token` (common/common.cpp:2134 @d81235049).
pub fn replayLastToken(a: std.mem.Allocator, ctx: ?*c.llama_context, last_token: c.llama_token, pos: i32) bool {
    var batch = Batch.init(a, ctx);
    defer batch.deinit();
    _ = batch.add(last_token, pos, 0, true);

    if (c.llama_process(ctx, c.LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        log.err("%s: failed to replay last token\n", .{@as([*:0]const u8, "common_replay_last_token")});
        return false;
    }
    return true;
}

fn oom() noreturn {
    _ = c.ggml_abort("common.cpp", 0, "out of memory");
    unreachable;
}

// -----------------------------------------------------------------------------
// typed decision models

/// Ports `common_decision_type` (common/common.h:955 @d81235049): typed
/// decision models, see `<arch>.decision.type` in the model metadata.
pub const DecisionType = enum(c_int) {
    none, // not a decision model
    openjev, // logits of one label token per option, read at the last prompt token
    lev, // same as openjev, noul is read from a rating scale
    kev, // dot product of the hidden states of the last token and of one end token per option
    nimble, // same as openjev, the prompt lists all the questions of the request
    laya, // score of one marker token per option, read from the embeddings output
    clef, // all questions in one prompt, score of option i read from the embeddings output at row i
    unknown, // a decision model of a type that is not supported
};

/// Ports `COMMON_DECISION_TYPE_NAMES` (common/common.cpp:1165 @d81235049),
/// in the `std::map`'s order, which is the enum's.
const decision_type_names = [_]struct { DecisionType, []const u8 }{
    .{ .openjev, "openjev" },
    .{ .lev, "lev" },
    .{ .kev, "kev" },
    .{ .nimble, "nimble" },
    .{ .laya, "laya" },
    .{ .clef, "clef" },
};

/// Ports `common_decision_type_from_string` (common/common.cpp:1174 @d81235049).
fn decisionTypeFromString(str: []const u8) DecisionType {
    for (decision_type_names) |pair| {
        if (std.mem.eql(u8, pair[1], str)) return pair[0];
    }
    return .unknown;
}

/// Ports `common_get_decision_type` (common/common.cpp:1183 @d81235049),
/// the overload over a loaded model.
pub fn getDecisionType(model: ?*const c.llama_model) DecisionType {
    var buf: [64]u8 = undefined;
    if (c.llama_model_meta_val_str(model, "general.architecture", &buf, buf.len) < 0) {
        return .none;
    }
    var key_buf: [80]u8 = undefined;
    const arch = std.mem.sliceTo(&buf, 0);
    const key = std.fmt.bufPrintSentinel(&key_buf, "{s}.decision.type", .{arch}, 0) catch return .none;
    if (c.llama_model_meta_val_str(model, key.ptr, &buf, buf.len) < 0) {
        return .none;
    }
    return decisionTypeFromString(std.mem.sliceTo(&buf, 0));
}

/// Ports `common_get_decision_type` (common/common.cpp:1195 @d81235049),
/// the overload reading a GGUF file without loading the model: `unknown`
/// for a missing, unreadable or invalid file.
pub fn getDecisionTypeFile(fname: []const u8) DecisionType {
    const gguf_params: c.gguf_init_params = .{ .no_alloc = true, .ctx = null };

    const fname_z = std.heap.c_allocator.dupeSentinel(u8, fname, 0) catch oom();
    defer std.heap.c_allocator.free(fname_z);
    const gguf_ctx = c.gguf_init_from_file(fname_z.ptr, gguf_params) orelse
        return .unknown; // missing or unreadable file
    defer c.gguf_free(gguf_ctx);

    const arch_id = c.gguf_find_key(gguf_ctx, "general.architecture");
    if (arch_id < 0) {
        return .unknown; // no architecture in the metadata
    }
    if (c.gguf_get_kv_type(gguf_ctx, arch_id) != c.GGUF_TYPE_STRING) {
        return .unknown; // malformed metadata
    }
    const arch = std.mem.span(c.gguf_get_val_str(gguf_ctx, arch_id));
    if (arch.len == 0) {
        return .unknown;
    }

    const key = std.fmt.allocPrintSentinel(std.heap.c_allocator, "{s}.decision.type", .{arch}, 0) catch oom();
    defer std.heap.c_allocator.free(key);
    const type_id = c.gguf_find_key(gguf_ctx, key.ptr);
    if (type_id < 0) {
        return .none;
    }
    if (c.gguf_get_kv_type(gguf_ctx, type_id) != c.GGUF_TYPE_STRING) {
        return .unknown; // malformed metadata
    }
    return decisionTypeFromString(std.mem.span(c.gguf_get_val_str(gguf_ctx, type_id)));
}

/// `fit.Log.sink` for the tools: `cli/fit.zig`'s messages through
/// `common_log`, at the level and verbosity of the `LOG_*` macro the C uses.
pub fn fitLogSink(verbosity: i32, text: [*:0]const u8) void {
    const level: log.Level = switch (verbosity) {
        log.level_error => .err,
        log.level_warn => .warn,
        else => .info,
    };
    if (verbosity <= log.verbosity_thold) log.add(level, "%s", .{text});
}

/// `fit.Log.json` for the tools: `LOG_JSON` (log.h:142 @d81235049), a
/// structured record through `common_log`, written only under `--log-jsonl`.
pub fn fitLogJson(kind: [*:0]const u8, data: []const u8) void {
    log.addJson(kind, data);
}

// -----------------------------------------------------------------------------
// what llama-server reaches in common.cpp

/// `COM_TRC`/`COM_ERR`'s `cmn  %12.*s: ` prefix with the caller's name.
fn comLog(comptime level: enum { trc, inf, err }, comptime func: [*:0]const u8, comptime fmt: []const u8, args: anytype) void {
    const f = "cmn  %12.*s: " ++ fmt;
    const pre = .{ @as(c_int, 12), func };
    switch (level) {
        .trc => log.trc(f, pre ++ args),
        .inf => log.inf(f, pre ++ args),
        .err => log.err(f, pre ++ args),
    }
}

/// `std::thread::hardware_concurrency()`.
fn hardwareConcurrency() usize {
    return std.Thread.getCpuCount() catch 0;
}

/// Ports `common_params_get_system_info` (common/common.cpp:424 @d81235049).
///
/// Return: owned by `a`.
pub fn paramsGetSystemInfo(a: std.mem.Allocator, params: *const Params) ![:0]u8 {
    var os: std.ArrayList(u8) = .empty;
    try os.print(a, "system_info: n_threads = {d}", .{params.cpuparams.n_threads});
    if (params.cpuparams_batch.n_threads != -1) {
        try os.print(a, " (n_threads_batch = {d})", .{params.cpuparams_batch.n_threads});
    }
    try os.print(a, " / {d} | {s}", .{ hardwareConcurrency(), c.llama_print_system_info() });
    return a.dupeSentinel(u8, os.items, 0);
}

/// Ports `common_params_print_info` (common/common.cpp:400 @d81235049): the
/// build, the verbosity, the devices and the system, at the trace level but
/// for the verbosity line.
pub fn paramsPrintInfo(a: std.mem.Allocator, params: *const Params, print_devices: bool) !void {
    const func = "common_params_print_info";
    const bi = @import("build_info.zig");
    // `NDEBUG`, which `zig c++` defines in ReleaseFast and ReleaseSmall only
    const build_type: [*:0]const u8 = if (std.debug.runtime_safety) " (debug)" else "";
    comLog(.trc, func, "%s: build %d (%s) with %s for %s%s\n", .{ func, bi.build_number, bi.commit.ptr, bi.compiler.ptr, bi.build_target.ptr, build_type });

    const verbosity = log.verbosity_thold;
    comLog(.inf, func, "%s: verbosity = %d (adjust with the `-lv N` CLI arg)\n", .{ func, verbosity });

    // device enumeration creates a primary context on CUDA backends, skip it when the caller does not own any device
    if (print_devices and verbosity >= log.level_trace) {
        comLog(.trc, func, "%s", .{@as([*:0]const u8, "device_info:\n")});
        var i: usize = 0;
        while (i < c.ggml_backend_dev_count()) : (i += 1) {
            const dev = c.ggml_backend_dev_get(i);
            var free: usize = 0;
            var total: usize = 0;
            c.ggml_backend_dev_memory(dev, &free, &total);
            comLog(.trc, func, "  - %-8s: %s (%zu MiB, %zu MiB free)\n", .{ c.ggml_backend_dev_name(dev), c.ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024 });
        }
    }
    if (verbosity >= log.level_trace) {
        comLog(.trc, func, "%s\n", .{(try paramsGetSystemInfo(a, params)).ptr});
    }
}

/// Ports `common_context_seq_rm_type` (common/common.h:1032 @d81235049).
pub const SeqRmType = enum(c_int) {
    no = 0, // cannot seq_rm at all
    part = 1, // can seq_rm partial sequences
    full = 2, // can seq_rm full sequences only
    rs = 3, // can seq_rm partial sequences, bounded by n_rs_seq
};

/// Ports `common_context_can_seq_rm` (common/common.cpp:1549 @d81235049):
/// whether the context can remove part of a sequence, found by trying.
/// Clears the context's memory.
pub fn contextCanSeqRm(ctx: ?*c.llama_context) SeqRmType {
    const func = "common_context_can_seq_rm";
    const mem = c.llama_get_memory(ctx) orelse return .no;

    if (c.llama_n_rs_seq(ctx) > 0) {
        comLog(.trc, func, "%s", .{@as([*:0]const u8, "the context supports bounded partial sequence removal\n")});
        return .rs;
    }

    var res: SeqRmType = .part;

    c.llama_memory_clear(mem, true);

    // eval 2 tokens to check if the context is compatible
    var tmp = [_]c.llama_token{ 0, 0 };

    done: {
        const ret = blk: {
            var batch = batchGetOne(std.heap.c_allocator, ctx, &tmp);
            defer batch.deinit();
            break :blk c.llama_process(ctx, c.LLAMA_PROCESS_TYPE_DECODE, batch.get());
        };
        if (ret != 0) {
            comLog(.err, func, "llama_process() failed: %d\n", .{ret});
            res = .no;
            break :done;
        }

        // try to remove the last tokens
        if (!c.llama_memory_seq_rm(mem, 0, 1, -1)) {
            comLog(.trc, func, "%s", .{@as([*:0]const u8, "the context does not support partial sequence removal\n")});
            res = .full;
            break :done;
        }
    }

    c.llama_memory_clear(mem, true);
    c.llama_synchronize(ctx);

    return res;
}

/// Ports `common_context_seq_rm` (common/common.cpp:1594 @d81235049): aborts
/// when the memory refuses.
fn contextSeqRm(ctx: ?*c.llama_context, seq_id: c.llama_seq_id, p0: c.llama_pos, p1: c.llama_pos) void {
    const mem = c.llama_get_memory(ctx);
    if (!c.llama_memory_seq_rm(mem, seq_id, p0, p1)) {
        _ = c.ggml_abort("common.cpp", 1597, "failed to remove sequence %d with p0=%d, p1=%d\n", seq_id, p0, p1);
    }
}

/// Ports `common_memory` (common/common.h:1043 @d81235049): the target's
/// and the draft's memory, edited together.
pub const Memory = struct {
    ctx_tgt: ?*c.llama_context = null,
    ctx_dft: ?*c.llama_context = null,

    /// Ports `common_memory`'s `init` (common/common.cpp:1611 @d81235049).
    pub fn init(self: *Memory, ctx_tgt: ?*c.llama_context, ctx_dft: ?*c.llama_context) void {
        self.ctx_tgt = ctx_tgt;
        self.ctx_dft = ctx_dft;
    }

    /// Ports `common_memory`'s `seq_rm` (common/common.cpp:1616 @d81235049):
    /// aborts on failure.
    pub fn seqRm(self: *const Memory, seq_id: c.llama_seq_id, p0: c.llama_pos, p1: c.llama_pos) void {
        contextSeqRm(self.ctx_tgt, seq_id, p0, p1);
        if (self.ctx_dft) |d| contextSeqRm(d, seq_id, p0, p1);
    }

    /// Ports `common_memory`'s `seq_cp` (common/common.cpp:1623 @d81235049).
    pub fn seqCp(self: *const Memory, seq_id_src: c.llama_seq_id, seq_id_dst: c.llama_seq_id, p0: c.llama_pos, p1: c.llama_pos) void {
        c.llama_memory_seq_cp(c.llama_get_memory(self.ctx_tgt), seq_id_src, seq_id_dst, p0, p1);
        if (self.ctx_dft) |d| c.llama_memory_seq_cp(c.llama_get_memory(d), seq_id_src, seq_id_dst, p0, p1);
    }

    /// Ports `common_memory`'s `seq_add` (common/common.cpp:1630 @d81235049).
    pub fn seqAdd(self: *const Memory, seq_id: c.llama_seq_id, p0: c.llama_pos, p1: c.llama_pos, delta: c.llama_pos) void {
        c.llama_memory_seq_add(c.llama_get_memory(self.ctx_tgt), seq_id, p0, p1, delta);
        if (self.ctx_dft) |d| c.llama_memory_seq_add(c.llama_get_memory(d), seq_id, p0, p1, delta);
    }
};

/// Ports `common_set_adapter_lora` (common/common.cpp:1637 @d81235049): the
/// adapters and their scales onto the context.
pub fn setAdapterLora(a: std.mem.Allocator, ctx: ?*c.llama_context, lora: []const @import("params.zig").LoraInfo) !void {
    const loras = try a.alloc(?*c.llama_adapter_lora, lora.len);
    defer a.free(loras);
    const scales = try a.alloc(f32, lora.len);
    defer a.free(scales);
    for (lora, 0..) |la, i| {
        loras[i] = la.ptr;
        scales[i] = la.scale;
    }
    // an empty `std::vector`'s `data()` is null, and libllama logs it
    _ = c.llama_set_adapters_lora(ctx, if (lora.len == 0) null else loras.ptr, lora.len, if (lora.len == 0) null else scales.ptr);
}

extern fn pow(x: f64, y: f64) f64;

/// Ports `common_embd_normalize` (common/common.cpp:1883 @d81235049).
pub fn embdNormalize(inp: [*]const f32, out: [*]f32, n: c_int, embd_norm: c_int) void {
    var sum: f64 = 0.0;
    const len: usize = @intCast(@max(n, 0));

    switch (embd_norm) {
        -1 => { // no normalisation
            sum = 1.0;
        },
        0 => { // max absolute
            for (0..len) |i| {
                if (sum < @abs(inp[i])) sum = @abs(inp[i]);
            }
            sum /= 32760.0; // make an int16 range
        },
        2 => { // euclidean
            for (0..len) |i| {
                // `inp[i] * inp[i]` is a float product, widened for the sum
                sum += @as(f64, inp[i] * inp[i]);
            }
            sum = @sqrt(sum);
        },
        else => { // p-norm (euclidean is p-norm p=2)
            for (0..len) |i| {
                sum += pow(@abs(@as(f64, inp[i])), @floatFromInt(embd_norm));
            }
            sum = pow(sum, 1.0 / @as(f64, @floatFromInt(embd_norm)));
        },
    }

    const norm: f32 = if (sum > 0.0) @floatCast(1.0 / sum) else 0.0;

    for (0..len) |i| out[i] = inp[i] * norm;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
