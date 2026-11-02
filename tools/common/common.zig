//! `common_init` and the token helpers of `common/common.cpp`.
//!
//! # Provenance
//!
//! Ports `common_init`, `common_tokenize` and `common_token_to_piece` from
//! `llama.cpp/common/common.cpp`, and `common_log_get_verbosity` and
//! `common_log_default_callback` from `common/log.cpp` (v0.5.0,
//! `7fe450e19`). Each declaration names what it ports and the line it began
//! at.

const std = @import("std");
const builtin = @import("builtin");
const c = @import("tools_h");
const log = @import("log.zig");
const arg = @import("arg.zig");
const Params = @import("params.zig").Params;

extern fn SetConsoleOutputCP(cp: c_uint) callconv(.winapi) c_int;
extern fn SetConsoleCP(cp: c_uint) callconv(.winapi) c_int;

/// Ports `common_log_get_verbosity` (log.cpp:529 @7fe450e19), which v0.4.0
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

/// Ports `common_log_default_callback` (log.cpp:542 @7fe450e19): libllama's
/// messages through `common_log`, at `common_log_get_verbosity`'s level.
pub fn defaultLogCallback(level: c.ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    if (logGetVerbosity(level) <= log.verbosity_thold) log.add(@fromBackingInt(@intCast(level)), "%s", .{text});
}

/// Ports `common_init` (common/common.cpp:385 @7fe450e19).
pub fn init() void {
    if (comptime builtin.os.tag == .windows) {
        _ = SetConsoleOutputCP(65001);
        _ = SetConsoleCP(65001);
    }

    log.setPrefix(true);
    log.setTimestamps(true);

    c.llama_log_set(defaultLogCallback, null);
}

/// Ports `common_tokenize` (common/common.cpp:1870 @7fe450e19).
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

/// Ports `common_token_to_piece` (common/common.cpp:1898 @7fe450e19): the first
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

/// Ports `common_detokenize` (common/common.cpp:1920 @7fe450e19): the first try
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

/// Ports `common_model_params_to_llama` (common/common.cpp:1679 @7fe450e19).
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
    mparams.load_mtp = for (p.speculative.types.items) |t| {
        if (t == .draft_mtp) break true;
    } else false;

    return mparams;
}

/// Ports `common_context_params_to_llama` (common/common.cpp:1718 @7fe450e19).
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

/// Ports `common_batch_clear` (common/common.cpp:1833 @7fe450e19).
pub fn batchClear(batch: *c.llama_batch) void {
    batch.n_tokens = 0;
}

/// Ports `common_batch_add` (common/common.cpp:1837 @7fe450e19).
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

/// `fit.Log.json` for the tools: `LOG_JSON` (log.h:141 @7fe450e19), a
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

/// Ports `common_params_get_system_info` (common/common.cpp:421 @7fe450e19).
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

/// Ports `common_params_print_info` (common/common.cpp:397 @7fe450e19): the
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

/// Ports `common_context_seq_rm_type` (common/common.h:988 @7fe450e19).
pub const SeqRmType = enum(c_int) {
    no = 0, // cannot seq_rm at all
    part = 1, // can seq_rm partial sequences
    full = 2, // can seq_rm full sequences only
    rs = 3, // can seq_rm partial sequences, bounded by n_rs_seq
};

/// Ports `common_context_can_seq_rm` (common/common.cpp:1583 @7fe450e19):
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
        const ret = c.llama_decode(ctx, c.llama_batch_get_one(&tmp, tmp.len));
        if (ret != 0) {
            comLog(.err, func, "llama_decode() failed: %d\n", .{ret});
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

/// Ports `common_context_seq_rm` (common/common.cpp:1624 @7fe450e19): aborts
/// when the memory refuses.
fn contextSeqRm(ctx: ?*c.llama_context, seq_id: c.llama_seq_id, p0: c.llama_pos, p1: c.llama_pos) void {
    const mem = c.llama_get_memory(ctx);
    if (!c.llama_memory_seq_rm(mem, seq_id, p0, p1)) {
        _ = c.ggml_abort("common.cpp", 1626, "failed to remove sequence %d with p0=%d, p1=%d\n", seq_id, p0, p1);
    }
}

/// Ports `common_memory` (common/common.h:999 @7fe450e19): the target's
/// and the draft's memory, edited together.
pub const Memory = struct {
    ctx_tgt: ?*c.llama_context = null,
    ctx_dft: ?*c.llama_context = null,

    /// Ports `common_memory`'s `init` (common/common.cpp:1641 @7fe450e19).
    pub fn init(self: *Memory, ctx_tgt: ?*c.llama_context, ctx_dft: ?*c.llama_context) void {
        self.ctx_tgt = ctx_tgt;
        self.ctx_dft = ctx_dft;
    }

    /// Ports `common_memory`'s `seq_rm` (common/common.cpp:1646 @7fe450e19):
    /// aborts on failure.
    pub fn seqRm(self: *const Memory, seq_id: c.llama_seq_id, p0: c.llama_pos, p1: c.llama_pos) void {
        contextSeqRm(self.ctx_tgt, seq_id, p0, p1);
        if (self.ctx_dft) |d| contextSeqRm(d, seq_id, p0, p1);
    }

    /// Ports `common_memory`'s `seq_cp` (common/common.cpp:1653 @7fe450e19).
    pub fn seqCp(self: *const Memory, seq_id_src: c.llama_seq_id, seq_id_dst: c.llama_seq_id, p0: c.llama_pos, p1: c.llama_pos) void {
        c.llama_memory_seq_cp(c.llama_get_memory(self.ctx_tgt), seq_id_src, seq_id_dst, p0, p1);
        if (self.ctx_dft) |d| c.llama_memory_seq_cp(c.llama_get_memory(d), seq_id_src, seq_id_dst, p0, p1);
    }

    /// Ports `common_memory`'s `seq_add` (common/common.cpp:1660 @7fe450e19).
    pub fn seqAdd(self: *const Memory, seq_id: c.llama_seq_id, p0: c.llama_pos, p1: c.llama_pos, delta: c.llama_pos) void {
        c.llama_memory_seq_add(c.llama_get_memory(self.ctx_tgt), seq_id, p0, p1, delta);
        if (self.ctx_dft) |d| c.llama_memory_seq_add(c.llama_get_memory(d), seq_id, p0, p1, delta);
    }
};

/// Ports `common_set_adapter_lora` (common/common.cpp:1667 @7fe450e19): the
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

/// Ports `common_embd_normalize` (common/common.cpp:1940 @7fe450e19).
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
