//! `common_init` and the token helpers of `common/common.cpp`.
//!
//! # Provenance
//!
//! Ports `common_init`, `common_tokenize` and `common_token_to_piece` from
//! `llama.cpp/common/common.cpp`, and `common_get_verbosity` and
//! `common_log_default_callback` from `common/log.cpp` (v0.3.0,
//! `c1d0e7a00`). Each declaration names what it ports and the line it began
//! at.

const std = @import("std");
const builtin = @import("builtin");
const c = @import("tools_h");
const log = @import("log.zig");
const arg = @import("arg.zig");
const Params = @import("params.zig").Params;

extern fn SetConsoleOutputCP(cp: c_uint) callconv(.winapi) c_int;
extern fn SetConsoleCP(cp: c_uint) callconv(.winapi) c_int;

/// Ports `common_get_verbosity` (log.cpp:441 @c1d0e7a00).
fn verbosityOf(level: c.ggml_log_level) c_int {
    return switch (level) {
        c.GGML_LOG_LEVEL_DEBUG => log.level_debug,
        c.GGML_LOG_LEVEL_INFO => log.level_trace,
        c.GGML_LOG_LEVEL_WARN => log.level_warn,
        c.GGML_LOG_LEVEL_ERROR => log.level_error,
        c.GGML_LOG_LEVEL_CONT => log.level_trace,
        else => log.level_output,
    };
}

/// Ports `common_log_default_callback` (log.cpp:454 @c1d0e7a00): libllama's
/// messages through `common_log`, at `common_get_verbosity`'s level.
pub fn defaultLogCallback(level: c.ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    if (verbosityOf(level) <= log.verbosity_thold) log.add(@fromBackingInt(@intCast(level)), "%s", .{text});
}

/// Ports `common_init` (common/common.cpp:385 @c1d0e7a00).
pub fn init() void {
    if (comptime builtin.os.tag == .windows) {
        _ = SetConsoleOutputCP(65001);
        _ = SetConsoleCP(65001);
    }

    log.setPrefix(true);
    log.setTimestamps(true);

    c.llama_log_set(defaultLogCallback, null);
}

/// Ports `common_tokenize` (common/common.cpp:1870 @c1d0e7a00).
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

/// Ports `common_token_to_piece` (common/common.cpp:1898 @c1d0e7a00): the first
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

/// Ports `common_detokenize` (common/common.cpp:1920 @c1d0e7a00): the first try
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
        if (n_chars > @as(i32, @intCast(text.len))) _ = c.ggml_abort("common.cpp", 1927, "GGML_ASSERT(%s) failed", "n_chars <= (int32_t)text.size()");
    }
    return text[0..@intCast(n_chars)];
}

/// Ports `common_model_params_to_llama` (common/common.cpp:1680 @c1d0e7a00).
pub fn modelParamsToLlama(p: *Params) c.llama_model_params {
    var mparams = c.llama_model_default_params();

    if (p.devices.items.len != 0) mparams.devices = p.devices.items.ptr;

    mparams.n_gpu_layers = p.n_gpu_layers;
    mparams.main_gpu = p.main_gpu;
    mparams.split_mode = p.split_mode;
    mparams.load_mode = p.load_mode;
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

    mparams.progress_callback = null;
    mparams.progress_callback_user_data = null;
    mparams.no_alloc = p.no_alloc;
    // no speculative types are set by any tool ported: `load_mtp` is false
    mparams.load_mtp = false;

    return mparams;
}

/// Ports `common_context_params_to_llama` (common/common.cpp:1718 @c1d0e7a00).
pub fn contextParamsToLlama(p: *const Params) c.llama_context_params {
    var cparams = c.llama_context_default_params();

    cparams.n_ctx = @bitCast(p.n_ctx);
    cparams.n_seq_max = @bitCast(p.n_parallel);
    cparams.n_rs_seq = 0; // `speculative.need_n_rs_seq()`: no speculative types
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

/// Ports `common_batch_clear` (common/common.cpp:1833 @c1d0e7a00).
pub fn batchClear(batch: *c.llama_batch) void {
    batch.n_tokens = 0;
}

/// Ports `common_batch_add` (common/common.cpp:1837 @c1d0e7a00).
pub fn batchAdd(batch: *c.llama_batch, id: c.llama_token, pos: c.llama_pos, seq_ids: []const c.llama_seq_id, logits: bool) void {
    const n: usize = @intCast(batch.n_tokens);
    if (batch.seq_id[n] == null) _ = c.ggml_abort("common.cpp", 1843, "GGML_ASSERT(%s) failed", "batch.seq_id[batch.n_tokens] && \"llama_batch size exceeded\"");

    batch.token[n] = id;
    batch.pos[n] = pos;
    batch.n_seq_id[n] = @intCast(seq_ids.len);
    for (seq_ids, 0..) |s, i| batch.seq_id[n][i] = s;
    batch.logits[n] = @intFromBool(logits);

    batch.n_tokens += 1;
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

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
