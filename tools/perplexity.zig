//! `llama-perplexity`: perplexity over a text, KL divergence against a base
//! model's saved log-probabilities, and the HellaSwag, Winogrande and
//! multiple-choice scores.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/perplexity/perplexity.cpp` and
//! `tools/perplexity/main.cpp` at v0.4.0 (`5266f24da`). Each declaration
//! names the C++ it replaces and the line it began at.
//!
//! # Threads
//!
//! Upstream computes log-probabilities on `hardware_concurrency()` threads
//! off a shared counter. Where each result lands in its own slot the port
//! does the same with `std.Thread`; where the threads add partial sums into
//! a total as they finish -- the perplexity's `nll`, the KL divergence's
//! sums -- upstream's last bits depend on scheduling, and the port adds the
//! partial sums in thread order.
//!
//! # Random task selection
//!
//! HellaSwag draws tasks with `std::uniform_int_distribution<size_t>` over
//! `std::mt19937(1)`; `uniformIndex` is libc++'s algorithm for it, the
//! independent-bits engine. Winogrande and multiple choice scale a raw draw.

const std = @import("std");
const tc = @import("tools_common");
const fit = @import("cli_fit");
const libcxx = @import("llama").libcxx;
const c = tc.c;
const io = tc.io;
const log = tc.log;
const arg = tc.arg;
const Params = tc.params.Params;

comptime {
    // libllama, compiled in (see `build/tools.zig`).
    _ = @import("llama");
}

const libm = struct {
    extern "c" fn expf(x: f32) f32;
    extern "c" fn logf(x: f32) f32;
    extern "c" fn exp(x: f64) f64;
    extern "c" fn log(x: f64) f64;
    extern "c" fn sqrt(x: f64) f64;
    extern "c" fn sqrtf(x: f32) f32;
};

const gpa = std.heap.c_allocator;

/// `GGML_ASSERT`, which upstream keeps in every build mode.
fn ggmlAssert(ok: bool, comptime line: c_int, comptime expr: [*:0]const u8) void {
    if (!ok) _ = c.ggml_abort("perplexity.cpp", line, "GGML_ASSERT(%s) failed", expr);
}

fn cint(v: anytype) c_int {
    return @intCast(v);
}

fn hardwareConcurrency() usize {
    return std.Thread.getCpuCount() catch 0;
}

/// Ports `softmax` (perplexity.cpp:40 @5266f24da).
fn softmax(a: std.mem.Allocator, logits: []const f32) ![]f32 {
    const probs = try a.alloc(f32, logits.len);
    var max_logit = logits[0];
    for (logits) |v| max_logit = @max(max_logit, v);
    var sum_exp: f64 = 0.0;
    for (logits, 0..) |l, i| {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const logit = l - max_logit;
        const exp_logit = libm.expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (probs) |*p| p.* = @floatCast(@as(f64, p.*) / sum_exp);
    return probs;
}

/// Ports `results_log_softmax` and the first `log_softmax` (perplexity.cpp:34, 60 @5266f24da).
const LogSoftmax = struct { log_softmax: f64, logit: f32, prob: f32 };

fn logSoftmax(n_vocab: usize, logits: [*]const f32, tok: usize) LogSoftmax {
    var max_logit = logits[0];
    for (1..n_vocab) |i| max_logit = @max(max_logit, logits[i]);
    var sum_exp: f64 = 0.0;
    for (0..n_vocab) |i| sum_exp += libm.expf(logits[i] - max_logit);
    return .{
        .log_softmax = @as(f64, logits[tok] - max_logit) - libm.log(sum_exp),
        .logit = logits[tok],
        .prob = libm.expf(logits[tok] - max_logit) / @as(f32, @floatCast(sum_exp)),
    };
}

/// Ports `nearest_int` (perplexity.cpp:72 @5266f24da).
fn nearestInt(fval: f32) c_int {
    const val = fval + 12582912.0;
    const i: c_int = @bitCast(val);
    return (i & 0x007fffff) - 0x00400000;
}

/// Ports the second `log_softmax` (perplexity.cpp:79 @5266f24da): the
/// log-probabilities quantized to 16 bits behind a scale and a minimum.
fn logSoftmaxQ(n_vocab: usize, logits: [*]const f32, log_prob_in: [*]u16, tok: usize) f64 {
    var max_logit = logits[0];
    var min_logit = logits[0];
    for (1..n_vocab) |i| {
        max_logit = @max(max_logit, logits[i]);
        min_logit = @min(min_logit, logits[i]);
    }
    min_logit = @max(min_logit, max_logit - 16);
    var sum_exp: f64 = 0.0;
    for (0..n_vocab) |i| sum_exp += libm.expf(logits[i] - max_logit);
    const log_sum_exp: f32 = @floatCast(libm.log(sum_exp));
    const min_log_prob = min_logit - max_logit - log_sum_exp;
    const scale = (max_logit - min_logit) / 65535.0;
    const d: *[2]f32 = @ptrCast(@alignCast(log_prob_in));
    d[0] = scale;
    d[1] = min_log_prob;
    const log_prob = log_prob_in + 4;
    if (scale != 0) {
        const inv_scale = 1 / scale;
        for (0..n_vocab) |i| {
            log_prob[i] = if (logits[i] > min_logit) @truncate(@as(c_uint, @bitCast(nearestInt(inv_scale * (logits[i] - min_logit))))) else 0;
        }
    } else {
        @memset(log_prob[0..n_vocab], 0);
    }
    return max_logit + log_sum_exp - logits[tok];
}

/// What the threads of the first two overloads of `process_logits` and
/// `process_logits` share (perplexity.cpp:109, 144 @5266f24da).
const NllWork = struct {
    n_vocab: usize,
    logits: [*]const f32,
    tokens: [*]const c.llama_token,
    n_token: usize,
    /// The first form: per-token logit and probability.
    logit_history: ?[*]f32 = null,
    prob_history: ?[*]f32 = null,
    /// The second form: quantized log-probabilities, `nv` per token.
    log_probs: ?[*]u16 = null,
    nv: usize = 0,
    counter: std.atomic.Value(usize) = .init(0),

    fn compute(self: *NllWork, sums: *[2]f64) void {
        var local_nll: f64 = 0;
        var local_nll2: f64 = 0;
        while (true) {
            const i = self.counter.fetchAdd(1, .seq_cst);
            if (i >= self.n_token) break;
            const tok: usize = @intCast(self.tokens[i + 1]);
            if (self.log_probs) |lp| {
                const v = logSoftmaxQ(self.n_vocab, self.logits + i * self.n_vocab, lp + i * self.nv, tok);
                local_nll += v;
                local_nll2 = @mulAdd(f64, v, v, local_nll2);
            } else {
                const results = logSoftmax(self.n_vocab, self.logits + i * self.n_vocab, tok);
                const v = -results.log_softmax;
                local_nll += v;
                local_nll2 = @mulAdd(f64, v, v, local_nll2);
                self.logit_history.?[i] = results.logit;
                self.prob_history.?[i] = results.prob;
            }
        }
        sums.* = .{ local_nll, local_nll2 };
    }

    fn run(self: *NllWork, n_workers: usize, nll: *f64, nll2: *f64) !void {
        const sums = try gpa.alloc([2]f64, n_workers + 1);
        defer gpa.free(sums);
        const threads = try gpa.alloc(std.Thread, n_workers);
        defer gpa.free(threads);
        for (threads, 0..) |*w, k| w.* = try std.Thread.spawn(.{}, compute, .{ self, &sums[k] });
        self.compute(&sums[n_workers]);
        for (threads) |w| w.join();
        for (sums) |s| {
            nll.* += s[0];
            nll2.* += s[1];
        }
    }
};

/// Ports `kl_divergence_result` (perplexity.cpp:175 @5266f24da).
const KldResult = struct {
    sum_nll: f64 = 0.0,
    sum_nll2: f64 = 0.0,
    sum_nll_base: f64 = 0.0,
    sum_nll_base2: f64 = 0.0,
    sum_nll_nll_base: f64 = 0.0,
    sum_kld: f64 = 0.0,
    sum_kld2: f64 = 0.0,
    sum_p_diff: f64 = 0.0,
    sum_p_diff2: f64 = 0.0,
    sum_p_diff4: f64 = 0.0,
    max_p_diff: f32 = 0.0,
    n_same_top: usize = 0,
    count: usize = 0,
};

/// Ports the third `log_softmax` (perplexity.cpp:191 @5266f24da).
fn logSoftmaxKld(n_vocab: usize, logits: [*]const f32, base_log_prob_in: [*]const u16, tok: usize, kld: *KldResult) struct { f64, f32 } {
    var max_logit = logits[0];
    var imax: usize = 0;
    for (1..n_vocab) |i| {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            imax = i;
        }
    }
    var sum_exp: f64 = 0.0;
    for (0..n_vocab) |i| sum_exp += libm.expf(logits[i] - max_logit);
    const log_sum_exp: f32 = @floatCast(libm.log(sum_exp));
    const d: *const [2]f32 = @ptrCast(@alignCast(base_log_prob_in));
    const scale = d[0];
    const min_log_prob = d[1];
    const base_log_prob = base_log_prob_in + 4;

    const nll = max_logit + log_sum_exp - logits[tok];
    kld.sum_nll += nll;
    kld.sum_nll2 += nll * nll;

    const nll_base = -@mulAdd(f32, scale, @floatFromInt(base_log_prob[tok]), min_log_prob);
    kld.sum_nll_base += nll_base;
    kld.sum_nll_base2 += nll_base * nll_base;

    kld.sum_nll_nll_base += nll * nll_base;

    max_logit += log_sum_exp;
    var sum: f64 = 0;
    var imax_base: isize = -1;
    var p_log_base_max: f32 = 0;
    for (0..n_vocab) |i| {
        const p_log_base = @mulAdd(f32, scale, @floatFromInt(base_log_prob[i]), min_log_prob);
        if (i == 0 or p_log_base > p_log_base_max) {
            p_log_base_max = p_log_base;
            imax_base = @intCast(i);
        }
        if (p_log_base > -16.0) {
            const p_base = libm.expf(p_log_base);
            sum += p_base * (p_log_base - logits[i] + max_logit);
        }
    }
    kld.sum_kld += sum;
    kld.sum_kld2 = @mulAdd(f64, sum, sum, kld.sum_kld2);
    kld.count += 1;
    if (@as(isize, @intCast(imax)) == imax_base) kld.n_same_top += 1;

    const p_base = libm.expf(-nll_base);
    const p = libm.expf(-nll);
    const p_diff = p - p_base;
    kld.sum_p_diff += p_diff;
    const p_diff2: f64 = p_diff * p_diff;
    kld.sum_p_diff2 += p_diff2;
    kld.sum_p_diff4 = @mulAdd(f64, p_diff2, p_diff2, kld.sum_p_diff4);
    kld.max_p_diff = @max(kld.max_p_diff, @abs(p_diff));

    return .{ sum, p_diff };
}

/// What the third `process_logits`' threads share (perplexity.cpp:254 @5266f24da).
const KldWork = struct {
    n_vocab: usize,
    logits: [*]const f32,
    tokens: [*]const c.llama_token,
    n_token: usize,
    base_log_probs: [*]const u16,
    nv: usize,
    kld_values: [*]f32,
    p_diff_values: [*]f32,
    counter: std.atomic.Value(usize) = .init(0),

    fn compute(self: *KldWork, local: *KldResult) void {
        local.* = .{};
        while (true) {
            const i = self.counter.fetchAdd(1, .seq_cst);
            if (i >= self.n_token) break;
            const v = logSoftmaxKld(self.n_vocab, self.logits + i * self.n_vocab, self.base_log_probs + i * self.nv, @intCast(self.tokens[i + 1]), local);
            self.kld_values[i] = @floatCast(v[0]);
            self.p_diff_values[i] = v[1];
        }
    }

    fn run(self: *KldWork, n_workers: usize, kld: *KldResult) !void {
        const locals = try gpa.alloc(KldResult, n_workers + 1);
        defer gpa.free(locals);
        const threads = try gpa.alloc(std.Thread, n_workers);
        defer gpa.free(threads);
        for (threads, 0..) |*w, k| w.* = try std.Thread.spawn(.{}, compute, .{ self, &locals[k] });
        self.compute(&locals[n_workers]);
        for (threads) |w| w.join();
        for (locals) |l| {
            kld.sum_nll += l.sum_nll;
            kld.sum_nll2 += l.sum_nll2;
            kld.sum_nll_base += l.sum_nll_base;
            kld.sum_nll_base2 += l.sum_nll_base2;
            kld.sum_nll_nll_base += l.sum_nll_nll_base;
            kld.sum_kld += l.sum_kld;
            kld.sum_kld2 += l.sum_kld2;
            kld.sum_p_diff += l.sum_p_diff;
            kld.sum_p_diff2 += l.sum_p_diff2;
            kld.sum_p_diff4 += l.sum_p_diff4;
            kld.n_same_top += l.n_same_top;
            kld.max_p_diff = @max(kld.max_p_diff, l.max_p_diff);
            kld.count += l.count;
        }
    }
};

fn nowNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

/// The ETA line every pass-timed loop prints after its first pass.
fn printEta(func: [*:0]const u8, t_start: i128, scale: f32) void {
    const t_end = nowNs();
    const t_total: f32 = @floatCast(@as(f64, @floatFromInt(t_end - t_start)) / 1e9);
    log.inf("%s: %.2f seconds per pass - ETA ", .{ func, @as(f64, t_total) });
    var total_seconds: c_int = @intFromFloat(t_total * scale);
    if (total_seconds >= 60 * 60) {
        log.out("%d hours ", .{@divTrunc(total_seconds, 60 * 60)});
        total_seconds = @rem(total_seconds, 60 * 60);
    }
    log.out("%.2f minutes\n", .{@as(f64, @floatFromInt(total_seconds)) / 60.0});
}

/// Ports `perplexity_v2` (perplexity.cpp:296 @5266f24da): the strided form.
fn perplexityV2(a: std.mem.Allocator, ctx: ?*c.llama_context, params: *const Params) !void {
    const func = "perplexity_v2";
    const model = c.llama_get_model(ctx);
    const vocab = c.llama_model_get_vocab(model);

    const add_bos = c.llama_vocab_get_add_bos(vocab);
    ggmlAssert(!c.llama_vocab_get_add_eos(vocab), 306, "!llama_vocab_get_add_eos(vocab)");

    log.inf("%s: tokenizing the input ..\n", .{func});

    const tokens = try tc.common.tokenize(a, vocab, params.prompt, true, false);

    const n_ctx: c_int = @intCast(c.llama_n_ctx(ctx));

    if (cint(tokens.len) < 2 * n_ctx) {
        log.err("%s: you need at least %d tokens to evaluate perplexity with a context of %d\n", .{ func, 2 * n_ctx, n_ctx });
        log.err("%s: the data file you provided tokenizes to only %zu tokens\n", .{ func, tokens.len });
        return;
    }

    const logit_history = try a.alloc(f32, tokens.len);
    @memset(logit_history, 0);
    const prob_history = try a.alloc(f32, tokens.len);
    @memset(prob_history, 0);

    if (params.ppl_stride <= 0) {
        log.err("%s: stride is %d but must be greater than zero!\n", .{ func, params.ppl_stride });
        return;
    }

    const calc_chunk = n_ctx;

    log.inf("%s: have %zu tokens. Calculation chunk = %d\n", .{ func, tokens.len, calc_chunk });

    if (cint(tokens.len) <= calc_chunk) {
        log.err("%s: there are only %zu tokens, this is not enough for a context size of %d and stride %d\n", .{ func, tokens.len, n_ctx, params.ppl_stride });
        return;
    }

    const n_chunk_max: c_int = @intCast(@divTrunc(@as(i64, @intCast(tokens.len)) - calc_chunk + params.ppl_stride - 1, params.ppl_stride));

    const n_chunk = if (params.n_chunks < 0) n_chunk_max else @min(params.n_chunks, n_chunk_max);
    const n_batch = params.n_batch;

    const n_vocab: usize = @intCast(c.llama_vocab_n_tokens(vocab));

    var count: c_int = 0;
    var nll: f64 = 0.0;

    const n_seq: c_int = @max(1, @divTrunc(n_batch, n_ctx));
    log.inf("%s: computing over %d chunks, n_ctx=%d, batch_size=%d, n_seq=%d\n", .{ func, n_chunk, n_ctx, n_batch, n_seq });

    var logits: std.ArrayList(f32) = .empty;
    defer logits.deinit(gpa);

    var i: c_int = 0;
    while (i < n_chunk) : (i += 1) {
        const start = i * params.ppl_stride;
        const end = start + calc_chunk;

        const num_batches = @divTrunc(calc_chunk + n_batch - 1, n_batch);

        logits.clearRetainingCapacity();

        const t_start = nowNs();

        // clear the KV cache
        c.llama_memory_clear(c.llama_get_memory(ctx), true);

        var batch = c.llama_batch_init(n_batch, 0, 1);

        var j: c_int = 0;
        while (j < num_batches) : (j += 1) {
            const batch_start = start + j * n_batch;
            const batch_size = @min(end - batch_start, n_batch);

            tc.common.batchClear(&batch);
            var k: c_int = 0;
            while (k < batch_size) : (k += 1) {
                tc.common.batchAdd(&batch, tokens[@intCast(batch_start + k)], j * n_batch + k, &.{0}, true);
            }

            if (c.llama_decode(ctx, batch) != 0) {
                c.llama_batch_free(batch);
                return;
            }

            // save original token and restore it after eval
            const token_org = tokens[@intCast(batch_start)];

            // add BOS token for the first batch of each chunk
            if (add_bos and j == 0) tokens[@intCast(batch_start)] = c.llama_vocab_bos(vocab);

            const batch_logits = c.llama_get_logits(ctx);
            try logits.appendSlice(gpa, batch_logits[0 .. @as(usize, @intCast(batch_size)) * n_vocab]);

            if (j == 0) tokens[@intCast(batch_start)] = token_org;
        }

        c.llama_batch_free(batch);

        if (i == 0) printEta(func, t_start, @floatFromInt(n_chunk));

        j = n_ctx - params.ppl_stride - 1;
        while (j < n_ctx - 1) : (j += 1) {
            // Calculate probability of next token, given the previous ones.
            const ju: usize = @intCast(j);
            const tok_logits = logits.items[ju * n_vocab .. (ju + 1) * n_vocab];
            const next: usize = @intCast(start + j + 1);

            const probs = try softmax(a, tok_logits);
            const prob = probs[@intCast(tokens[next])];
            a.free(probs);
            logit_history[next] = tok_logits[@intCast(tokens[next])];
            prob_history[next] = prob;

            nll += -libm.logf(prob);
            count += 1;
        }
        // perplexity is e^(average negative log-likelihood)
        if (params.ppl_output_type == 0) {
            log.out("[%d]%.4lf,", .{ i + 1, libm.exp(nll / @as(f64, @floatFromInt(count))) });
        } else {
            log.out("%8d  %.4lf\n", .{ i * params.ppl_stride, libm.exp(nll / @as(f64, @floatFromInt(count))) });
        }
    }
    log.out("\n", .{});
}

/// Ports `perplexity` (perplexity.cpp:444 @5266f24da).
fn perplexity(a: std.mem.Allocator, ctx: ?*c.llama_context, params: *const Params, n_ctx: i32) !void {
    if (params.ppl_stride > 0) return perplexityV2(a, ctx, params);

    const func = "perplexity";
    const model = c.llama_get_model(ctx);
    const vocab = c.llama_model_get_vocab(model);

    const add_bos = c.llama_vocab_get_add_bos(vocab);
    ggmlAssert(!c.llama_vocab_get_add_eos(vocab), 458, "!llama_vocab_get_add_eos(vocab)");

    var logits_stream: ?*std.c.FILE = null;
    defer if (logits_stream) |f| {
        _ = std.c.fclose(f);
    };
    const W = struct {
        fn bytes(f: ?*std.c.FILE, b: []const u8) void {
            if (f) |ff| _ = std.c.fwrite(b.ptr, 1, b.len, ff);
        }
    };
    if (params.logits_file.len != 0) {
        const z = try a.dupeSentinel(u8, params.logits_file, 0);
        logits_stream = std.c.fopen(z.ptr, "wb");
        if (logits_stream == null) {
            log.err("%s: failed to open %s for writing\n", .{ func, z.ptr });
            return;
        }
        log.inf("%s: saving all logits to %s\n", .{ func, z.ptr });
        W.bytes(logits_stream, "_logits_");
        W.bytes(logits_stream, std.mem.asBytes(&n_ctx));
    }

    const tim1 = nowNs();
    log.inf("%s: tokenizing the input ..\n", .{func});

    const tokens = try tc.common.tokenize(a, vocab, params.prompt, true, false);

    const tim2 = nowNs();
    log.inf("%s: tokenization took %g ms\n", .{ func, 1e-3 * @as(f64, @floatFromInt(@divTrunc(tim2 - tim1, 1000))) });

    if (cint(tokens.len) < 2 * n_ctx) {
        log.err("%s: you need at least %d tokens to evaluate perplexity with a context of %d\n", .{ func, 2 * n_ctx, n_ctx });
        log.err("%s: the data file you provided tokenizes to only %zu tokens\n", .{ func, tokens.len });
        return;
    }

    const logit_history = try a.alloc(f32, tokens.len);
    @memset(logit_history, 0);
    const prob_history = try a.alloc(f32, tokens.len);
    @memset(prob_history, 0);

    const n_chunk_max: c_int = @intCast(tokens.len / @as(usize, @intCast(n_ctx)));

    const n_chunk = if (params.n_chunks < 0) n_chunk_max else @min(params.n_chunks, n_chunk_max);
    const n_batch = params.n_batch;

    const n_vocab: c_int = c.llama_vocab_n_tokens(vocab);
    const nvu: usize = @intCast(n_vocab);

    var count: c_int = 0;
    var nll: f64 = 0.0;
    var nll2: f64 = 0.0;

    const num_batches = @divTrunc(n_ctx + n_batch - 1, n_batch);
    const n_seq: c_int = @max(1, @divTrunc(n_batch, n_ctx));

    ggmlAssert(n_batch < n_ctx or @rem(n_batch, n_ctx) == 0, 507, "n_batch < n_ctx || n_batch % n_ctx == 0");
    ggmlAssert(params.n_ctx == n_seq * n_ctx, 508, "params.n_ctx == n_seq * n_ctx");

    var batch = c.llama_batch_init(@min(n_batch, n_ctx * n_seq), 0, 1);

    var logits: std.ArrayList(f32) = .empty;
    defer logits.deinit(gpa);
    if (num_batches > 1) try logits.ensureTotalCapacity(gpa, @as(usize, @intCast(n_ctx)) * nvu);

    log.inf("%s: calculating perplexity over %d chunks, n_ctx=%d, batch_size=%d, n_seq=%d\n", .{ func, n_chunk, n_ctx, n_batch, n_seq });

    const n_workers = hardwareConcurrency() - 1;

    var log_probs: []u16 = &.{};
    const nv: usize = @intCast(2 * @divTrunc(n_vocab + 1, 2) + 4);
    if (params.logits_file.len != 0) {
        W.bytes(logits_stream, std.mem.asBytes(&n_vocab));
        W.bytes(logits_stream, std.mem.asBytes(&n_chunk));
        W.bytes(logits_stream, std.mem.sliceAsBytes(tokens[0..@intCast(n_chunk * n_ctx)]));
        log_probs = try gpa.alloc(u16, @as(usize, @intCast(n_ctx)) * nv);
        @memset(log_probs, 0);
    }
    defer if (log_probs.len != 0) gpa.free(log_probs);

    // calculate the perplexity over the last half of the window (so the
    // model always has some context to predict the token)
    const first = @divTrunc(n_ctx, 2);

    var i: c_int = 0;
    while (i < n_chunk) : (i += n_seq) {
        const start = i * n_ctx;
        const end = start + n_ctx;

        const n_seq_batch = @min(n_seq, n_chunk - i);

        const t_start = nowNs();

        // clear the KV cache
        c.llama_memory_clear(c.llama_get_memory(ctx), true);

        var j: c_int = 0;
        while (j < num_batches) : (j += 1) {
            const batch_start = start + j * n_batch;
            const batch_size = @min(end - batch_start, n_batch);

            var n_outputs: c_int = 0;

            batch.n_tokens = 0;
            var seq: c_int = 0;
            while (seq < n_seq_batch) : (seq += 1) {
                const seq_start: usize = @intCast(batch_start + seq * n_ctx);

                // save original token and restore it after decode
                const token_org = tokens[seq_start];

                // add BOS token for the first batch of each chunk
                if (add_bos and j == 0) tokens[seq_start] = c.llama_vocab_bos(vocab);

                var k: c_int = 0;
                while (k < batch_size) : (k += 1) {
                    const idx: usize = @intCast(seq * n_ctx + k);
                    batch.token[idx] = tokens[seq_start + @as(usize, @intCast(k))];
                    batch.pos[idx] = j * n_batch + k;
                    batch.n_seq_id[idx] = 1;
                    batch.seq_id[idx][0] = seq;
                    batch.logits[idx] = if (batch.pos[idx] >= first) 1 else 0;

                    n_outputs += @intFromBool(batch.logits[idx] != 0);
                }
                batch.n_tokens += batch_size;

                // restore the original token in case it was set to BOS
                tokens[seq_start] = token_org;
            }

            if (c.llama_decode(ctx, batch) != 0) {
                log.inf("%s : failed to decode\n", .{func});
                return;
            }

            if (num_batches > 1 and n_outputs > 0) {
                const batch_logits = c.llama_get_logits(ctx);
                try logits.appendSlice(gpa, batch_logits[0 .. @as(usize, @intCast(n_outputs)) * nvu]);
            }
        }

        if (i == 0) {
            c.llama_synchronize(ctx);
            printEta(func, t_start, @as(f32, @floatFromInt(n_chunk)) / @as(f32, @floatFromInt(n_seq)));
        }

        var seq: c_int = 0;
        while (seq < n_seq_batch) : (seq += 1) {
            const all_logits: [*]const f32 = if (num_batches > 1) logits.items.ptr else c.llama_get_logits_ith(ctx, seq * n_ctx + first);

            const base: usize = @intCast(start + seq * n_ctx + first);
            const n_token: usize = @intCast(n_ctx - 1 - first);
            if (params.logits_file.len != 0) {
                var work: NllWork = .{ .n_vocab = nvu, .logits = all_logits, .tokens = tokens.ptr + base, .n_token = n_token, .log_probs = log_probs.ptr, .nv = nv };
                try work.run(n_workers, &nll, &nll2);
                W.bytes(logits_stream, std.mem.sliceAsBytes(log_probs[0 .. n_token * nv]));
            } else {
                var work: NllWork = .{ .n_vocab = nvu, .logits = all_logits, .tokens = tokens.ptr + base, .n_token = n_token, .logit_history = logit_history.ptr + base, .prob_history = prob_history.ptr + base };
                try work.run(n_workers, &nll, &nll2);
            }
            count += n_ctx - first - 1;

            // perplexity is e^(average negative log-likelihood)
            const cnt: f64 = @floatFromInt(count);
            if (params.ppl_output_type == 0) {
                log.out("[%d]%.4lf,", .{ i + seq + 1, libm.exp(nll / cnt) });
            } else {
                const av = nll / cnt;
                var av2 = @mulAdd(f64, -av, av, nll2 / cnt);
                if (av2 > 0) av2 = libm.sqrt(av2 / @as(f64, @floatFromInt(count - 1)));
                log.out("%8d  %.4lf  %4lf  %4lf\n", .{ i * n_ctx, libm.exp(nll / cnt), av, av2 });
            }
        }

        logits.clearRetainingCapacity();
    }
    log.out("\n", .{});

    nll2 /= @floatFromInt(count);
    nll /= @floatFromInt(count);
    const ppl = libm.exp(nll);
    nll2 = @mulAdd(f64, -nll, nll, nll2);
    if (nll2 > 0) {
        nll2 = libm.sqrt(nll2 / @as(f64, @floatFromInt(count - 1)));
        log.inf("Final estimate: PPL = %.4lf +/- %.5lf\n", .{ ppl, nll2 * ppl });
    } else {
        log.err("Unexpected negative standard deviation of log(prob)\n", .{});
    }

    c.llama_batch_free(batch);
}

/// Ports `decode_helper` (perplexity.cpp:664 @5266f24da).
fn decodeHelper(ctx: ?*c.llama_context, batch: *c.llama_batch, batch_logits: []f32, n_batch: c_int, n_vocab: usize) bool {
    var prev_outputs: usize = 0;
    var i: c_int = 0;
    while (i < batch.n_tokens) : (i += n_batch) {
        const n_tokens = @min(n_batch, batch.n_tokens - i);
        const iu: usize = @intCast(i);

        const batch_view: c.llama_batch = .{
            .n_tokens = n_tokens,
            .token = batch.token + iu,
            .embd = null,
            .pos = batch.pos + iu,
            .n_seq_id = batch.n_seq_id + iu,
            .seq_id = batch.seq_id + iu,
            .logits = batch.logits + iu,
        };

        const ret = c.llama_decode(ctx, batch_view);
        if (ret != 0) {
            log.err("failed to decode the batch, n_batch = %d, ret = %d\n", .{ n_batch, ret });
            return false;
        }

        var n_outputs: usize = 0;
        for (0..@intCast(n_tokens)) |k| n_outputs += @intFromBool(batch_view.logits[k] != 0);

        const src = c.llama_get_logits(ctx);
        @memcpy(batch_logits[prev_outputs * n_vocab ..][0 .. n_outputs * n_vocab], src[0 .. n_outputs * n_vocab]);

        prev_outputs += n_outputs;
    }
    return true;
}

/// `K_TOKEN_CHUNK` (perplexity.cpp:698 @5266f24da).
const k_token_chunk = 4;

const EvalPair = struct { i_logits: usize, token: c.llama_token };

/// What `compute_logprobs`' threads share (perplexity.cpp:700 @5266f24da).
const LogprobWork = struct {
    batch_logits: [*]const f32,
    n_vocab: usize,
    eval_pairs: []const EvalPair,
    eval_results: []f32,
    counter: std.atomic.Value(usize) = .init(0),

    fn compute(self: *LogprobWork) void {
        var local_logprobs: [k_token_chunk]f32 = undefined;
        while (true) {
            const first = self.counter.fetchAdd(k_token_chunk, .monotonic);
            if (first >= self.eval_results.len) break;
            const last = @min(first + k_token_chunk, self.eval_results.len);
            for (first..last) |i| {
                const logits = self.batch_logits + self.eval_pairs[i].i_logits * self.n_vocab;
                var max_logit = logits[0];
                for (1..self.n_vocab) |j| max_logit = @max(max_logit, logits[j]);
                var sum_p: f32 = 0.0;
                for (0..self.n_vocab) |j| sum_p += libm.expf(logits[j] - max_logit);
                local_logprobs[i - first] = logits[@intCast(self.eval_pairs[i].token)] - max_logit - libm.logf(sum_p);
            }
            @memcpy(self.eval_results[first..last], local_logprobs[0 .. last - first]);
        }
    }
};

/// Ports `compute_logprobs` (perplexity.cpp:700 @5266f24da).
fn computeLogprobs(a: std.mem.Allocator, batch_logits: []const f32, n_vocab: usize, n_workers: usize, eval_pairs: []const EvalPair, eval_results: *std.ArrayList(f32)) !void {
    if (eval_results.items.len != eval_pairs.len) try eval_results.resize(a, eval_pairs.len);
    if (eval_pairs.len == 0) return;

    const max_threads = @min((eval_pairs.len + k_token_chunk - 1) / k_token_chunk, n_workers);

    var work: LogprobWork = .{ .batch_logits = batch_logits.ptr, .n_vocab = n_vocab, .eval_pairs = eval_pairs, .eval_results = eval_results.items };
    const threads = try gpa.alloc(std.Thread, max_threads);
    defer gpa.free(threads);
    for (threads) |*w| w.* = try std.Thread.spawn(.{}, LogprobWork.compute, .{&work});
    for (threads) |w| w.join();
}

/// `std::getline(stream, line, '\n')` over a string: every line, and no
/// empty last one after a final newline.
fn getLines(a: std.mem.Allocator, s: []const u8) !std.ArrayList([]const u8) {
    var lines: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, s, '\n');
    while (it.next()) |line| {
        if (it.index == null and line.len == 0) break;
        try lines.append(a, line);
    }
    return lines;
}

/// The C++ throwing `std::stoi` out of `main`: libc++ reports it and aborts.
fn stoiOrTerminate(a: std.mem.Allocator, s: []const u8) !c_int {
    return switch (tc.cxx.stoi(try a.dupeSentinel(u8, s, 0))) {
        .ok => |v| v,
        .no_conversion => {
            io.eprintf("libc++abi: terminating due to uncaught exception of type std::invalid_argument: stoi: no conversion\n", .{});
            std.process.abort();
        },
        .out_of_range => {
            io.eprintf("libc++abi: terminating due to uncaught exception of type std::out_of_range: stoi: out of range\n", .{});
            std.process.abort();
        },
    };
}

/// libc++'s `std::uniform_int_distribution<size_t>(lo, hi)(g)` over
/// `std::mt19937`: `__independent_bits_engine` drawing `w` bits, rejecting
/// past the range (`<__random/uniform_int_distribution.h>`).
fn uniformIndex(g: *libcxx.Mt19937, lo: usize, hi: usize) usize {
    const rp: u64 = @as(u64, hi) -% @as(u64, lo) +% 1;
    if (rp == 1) return lo;
    const dt: usize = 64;
    var w: usize = if (rp == 0) dt else dt - @clz(rp) - 1;
    if (rp != 0 and (rp & (@as(u64, std.math.maxInt(u64)) >> @intCast(dt - w))) != 0) w += 1;

    // the engine: 32-bit draws into a 64-bit working type, R = 2^32
    const r: u64 = 1 << 32;
    const m: usize = 32;
    const wdt: usize = 64;
    const edt: usize = 32;
    var n = w / m + @intFromBool(w % m != 0);
    var w0 = w / n;
    var y0: u64 = if (w0 < wdt) (r >> @intCast(w0)) << @intCast(w0) else 0;
    if (r - y0 > y0 / n) {
        n += 1;
        w0 = w / n;
        y0 = if (w0 < wdt) (r >> @intCast(w0)) << @intCast(w0) else 0;
    }
    const n0 = n - w % n;
    const y1: u64 = if (w0 < wdt - 1) (r >> @intCast(w0 + 1)) << @intCast(w0 + 1) else 0;
    const mask0: u32 = if (w0 > 0) @as(u32, 0xffffffff) >> @intCast(edt - w0) else 0;
    const mask1: u32 = if (w0 < edt - 1) @as(u32, 0xffffffff) >> @intCast(edt - (w0 + 1)) else 0xffffffff;

    while (true) {
        var sp: u64 = 0;
        for (0..n0) |_| {
            var u: u32 = undefined;
            while (true) {
                u = g.next();
                if (u < y0) break;
            }
            sp = if (w0 < 64) sp << @intCast(w0) else 0;
            sp +%= u & mask0;
        }
        for (n0..n) |_| {
            var u: u32 = undefined;
            while (true) {
                u = g.next();
                if (u < y1) break;
            }
            sp = if (w0 < 63) sp << @intCast(w0 + 1) else 0;
            sp +%= u & mask1;
        }
        if (rp == 0) return @intCast(sp);
        if (sp < rp) return @intCast(sp + lo);
    }
}

/// Mirrors `hellaswag_score`'s `hs_data_t` (perplexity.cpp:796 @5266f24da).
const HsData = struct {
    context: []const u8 = "",
    gold_ending_idx: usize = 0,
    ending: [4][]const u8 = .{ "", "", "", "" },
    ending_logprob_count: [4]usize = .{ 0, 0, 0, 0 },
    ending_logprob: [4]f64 = .{ 0, 0, 0, 0 },

    i_logits: usize = 0, // starting index of logits in the llama_batch
    common_prefix: usize = 0, // max number of initial tokens that are the same in all sentences
    required_tokens: usize = 0, // needed number of tokens to evaluate all 4 endings
    seq_tokens: [4][]c.llama_token = .{ &.{}, &.{}, &.{}, &.{} },
};

/// `seq[k]`, or a token no real one equals when `k` is past the end: the
/// C++ reads past a shorter sequence there.
fn at(seq: []const c.llama_token, k: usize) c.llama_token {
    return if (k < seq.len) seq[k] else -1;
}

/// Ports `hellaswag_score` (perplexity.cpp:744 @5266f24da).
fn hellaswagScore(a: std.mem.Allocator, ctx: ?*c.llama_context, params: *const Params) !void {
    const func = "hellaswag_score";
    const model = c.llama_get_model(ctx);
    const vocab = c.llama_model_get_vocab(model);

    // Datafile layout, 6 lines per task: the context, the gold ending's
    // index, and the four endings.
    var prompt_lines = try getLines(a, params.prompt);

    if (prompt_lines.items.len % 6 != 0) {
        log.err("%s : number of lines in prompt not a multiple of 6.\n", .{func});
        return;
    }

    var hs_task_count = prompt_lines.items.len / 6;
    log.inf("%s : loaded %zu tasks from prompt.\n", .{ func, hs_task_count });

    const is_spm = c.llama_vocab_type(vocab) == c.LLAMA_VOCAB_TYPE_SPM;
    log.inf("================================= is_spm = %d\n", .{@as(c_int, @intFromBool(is_spm))});

    // The tasks should be randomized so the score stabilizes quickly.
    const randomize_tasks = true;

    // Number of tasks to use when computing the score
    if (params.hellaswag_tasks < hs_task_count) hs_task_count = params.hellaswag_tasks;

    // The random seed should not impact the final result if the computation is done over enough tasks, so kept hardcoded for now
    var rng = libcxx.Mt19937.init(1);

    log.inf("%s : selecting %zu %s tasks.\n", .{ func, hs_task_count, @as([*:0]const u8, if (randomize_tasks) "randomized" else "the first") });

    // Select and read data from prompt lines
    const hs_data = try a.alloc(HsData, hs_task_count);
    for (hs_data) |*hs_cur| {
        hs_cur.* = .{};
        // Select a random example of those left in the prompt
        const idx = uniformIndex(&rng, 0, prompt_lines.items.len / 6 - 1);

        hs_cur.context = prompt_lines.items[idx * 6];
        hs_cur.gold_ending_idx = @bitCast(@as(isize, try stoiOrTerminate(a, prompt_lines.items[idx * 6 + 1])));
        for (0..4) |j| {
            hs_cur.ending[j] = prompt_lines.items[idx * 6 + 2 + j];
            const text = try std.mem.concat(a, u8, &.{ hs_cur.context, " ", hs_cur.ending[j] });
            hs_cur.seq_tokens[j] = try tc.common.tokenize(a, vocab, text, true, false);
        }

        // determine the common prefix of the endings
        hs_cur.common_prefix = 0;
        for (0..hs_cur.seq_tokens[0].len) |k| {
            const t0 = hs_cur.seq_tokens[0][k];
            if (t0 != at(hs_cur.seq_tokens[1], k) or t0 != at(hs_cur.seq_tokens[2], k) or t0 != at(hs_cur.seq_tokens[3], k)) break;
            hs_cur.common_prefix += 1;
        }
        hs_cur.required_tokens = hs_cur.common_prefix +%
            hs_cur.seq_tokens[0].len -% hs_cur.common_prefix +%
            hs_cur.seq_tokens[1].len -% hs_cur.common_prefix +%
            hs_cur.seq_tokens[2].len -% hs_cur.common_prefix +%
            hs_cur.seq_tokens[3].len -% hs_cur.common_prefix;

        // Delete the selected random example from the prompt
        try prompt_lines.replaceRange(a, idx * 6, 6, &.{});
    }

    log.inf("%s : calculating hellaswag score over selected tasks.\n", .{func});

    log.out("\ntask\tacc_norm\t95%% confidence interval\n", .{});

    var acc: f64 = 0.0;

    const n_ctx: c_int = @intCast(c.llama_n_ctx(ctx));
    const n_batch = params.n_batch;

    const n_vocab: usize = @intCast(c.llama_vocab_n_tokens(vocab));

    const max_tasks_per_batch = 32;
    const max_seq: c_int = @min(4 * max_tasks_per_batch, @as(c_int, @intCast(c.llama_n_seq_max(ctx))));

    var batch = c.llama_batch_init(n_ctx, 0, 4);

    const tok_logits = try a.alloc(f32, n_vocab);
    // TODO: this could be made smaller; it's currently the worst-case size
    const batch_logits = try gpa.alloc(f32, @as(usize, @intCast(n_ctx)) * n_vocab);
    defer gpa.free(batch_logits);
    @memset(batch_logits, 0);

    var eval_pairs: std.ArrayList(EvalPair) = .empty;
    var eval_results: std.ArrayList(f32) = .empty;
    const n_workers = hardwareConcurrency();

    var j0: usize = 0;
    while (j0 < hs_task_count) : (j0 += 1) {
        var n_cur: c_int = 0;

        var j1 = j0;
        var i_logits: usize = 0; // this tells us how many logits were needed before this point in the batch

        tc.common.batchClear(&batch);

        // batch as much tasks as possible into the available context
        while (n_cur + @as(c_int, @intCast(hs_data[j1].required_tokens)) <= n_ctx) {
            const hs_cur = &hs_data[j1];
            var n_logits: usize = 0;

            const s0: c_int = @intCast(4 * (j1 - j0));
            if (s0 + 4 > max_seq) break;

            for (0..hs_cur.common_prefix) |i| {
                tc.common.batchAdd(&batch, hs_cur.seq_tokens[0][i], @intCast(i), &.{ s0 + 0, s0 + 1, s0 + 2, s0 + 3 }, false);
            }
            batch.logits[@intCast(batch.n_tokens - 1)] = 1; // we need logits for the last token of the common prefix
            n_logits += 1;

            for (0..4) |s| {
                const seq_tokens_size = hs_cur.seq_tokens[s].len;
                var i = hs_cur.common_prefix;
                while (i < seq_tokens_size) : (i += 1) {
                    const needs_logits = i < seq_tokens_size - 1;
                    tc.common.batchAdd(&batch, hs_cur.seq_tokens[s][i], @intCast(i), &.{s0 + @as(c_int, @intCast(s))}, needs_logits);
                    n_logits += @intFromBool(needs_logits);
                }
            }

            hs_cur.i_logits = i_logits;
            i_logits += n_logits;

            n_cur += @intCast(hs_data[j1].required_tokens);
            j1 += 1;
            if (j1 == hs_task_count) break;
        }

        if (j0 == j1) {
            log.err("%s : task %zu does not fit in the context window (requires %zu tokens)\n", .{ func, j0, hs_data[j0].required_tokens });
            return;
        }

        c.llama_memory_clear(c.llama_get_memory(ctx), true);

        // decode all tasks [j0, j1)
        if (!decodeHelper(ctx, &batch, batch_logits, n_batch, n_vocab)) {
            log.err("%s: llama_decode() failed\n", .{func});
            return;
        }

        // Compute log-probs in parallel
        // First we collect all tasks
        eval_pairs.clearRetainingCapacity();
        for (j0..j1) |i| {
            const hs_cur = &hs_data[i];
            var li: usize = 1; // skip the last logit of the common prefix (computed separately below)
            for (0..4) |s| {
                var j = hs_cur.common_prefix;
                while (j < hs_cur.seq_tokens[s].len -% 1) : (j += 1) {
                    try eval_pairs.append(a, .{ .i_logits = hs_cur.i_logits + li, .token = hs_cur.seq_tokens[s][j + 1] });
                    li += 1;
                }
            }
        }
        // Then we do the actual calculation
        try computeLogprobs(a, batch_logits, n_vocab, n_workers, eval_pairs.items, &eval_results);

        var ir: usize = 0;

        // compute the logprobs for each ending of the decoded tasks
        for (j0..j1) |i| {
            const hs_cur = &hs_data[i];

            // get the logits of the last token of the common prefix
            @memcpy(tok_logits, batch_logits[hs_cur.i_logits * n_vocab ..][0..n_vocab]);

            const first_probs = try softmax(a, tok_logits);

            for (0..4) |s| {
                hs_cur.ending_logprob_count[s] = 1;
                hs_cur.ending_logprob[s] = libm.logf(first_probs[@intCast(hs_cur.seq_tokens[s][hs_cur.common_prefix])]);
                var j = hs_cur.common_prefix;
                while (j < hs_cur.seq_tokens[s].len -% 1) : (j += 1) {
                    hs_cur.ending_logprob[s] += eval_results.items[ir];
                    ir += 1;
                    hs_cur.ending_logprob_count[s] += 1;
                }
                hs_cur.ending_logprob[s] /= @floatFromInt(hs_cur.ending_logprob_count[s]);
            }
            a.free(first_probs);

            // Find the ending with maximum logprob
            var ending_logprob_max_idx: usize = 0;
            var ending_logprob_max_val = hs_cur.ending_logprob[0];
            for (1..4) |s| {
                if (hs_cur.ending_logprob[s] > ending_logprob_max_val) {
                    ending_logprob_max_idx = s;
                    ending_logprob_max_val = hs_cur.ending_logprob[s];
                }
            }

            // If the gold ending got the maximum logprobe add one accuracy point
            if (ending_logprob_max_idx == hs_cur.gold_ending_idx) acc += 1.0;

            const n: f64 = @floatFromInt(i + 1);
            const freq = acc / n;

            const za = 1.95996398454;

            // Wilson score interval, more accurate
            const z = za * za / n;
            const cnf = z * libm.sqrt(n * @mulAdd(f64, 4.0 * freq, 1 - freq, z)) / (za + za);
            const lo = (@mulAdd(f64, z, 0.5, freq) - cnf) / (1.0 + z);
            const hi = (@mulAdd(f64, z, 0.5, freq) + cnf) / (1.0 + z);

            // Print the accumulated accuracy mean x 100 and confidence interval
            log.out("%zu\t%3.8lf%%\t[%3.4lf%%, %3.4lf%%]\n", .{ i + 1, freq * 100.0, lo * 100.0, hi * 100.0 });
        }

        j0 = j1 - 1;
    }

    c.llama_batch_free(batch);

    log.out("\n", .{});
}

/// Mirrors `winogrande_entry` (perplexity.cpp:1017 @5266f24da).
const WinograndeEntry = struct {
    first: []const u8 = "",
    second: []const u8 = "",
    choices: [2][]const u8 = .{ "", "" },
    answer: c_int = 0,

    i_logits: usize = 0,
    common_prefix: usize = 0,
    required_tokens: usize = 0,
    n_base1: usize = 0, // number of tokens for context + choice 1
    n_base2: usize = 0, // number of tokens for context + choice 2
    seq_tokens: [2][]c.llama_token = .{ &.{}, &.{} },
};

/// `std::string::substr(pos, n)`: `n` past the end takes the rest.
fn substr(s: []const u8, pos: usize, n: usize) []const u8 {
    return s[pos..][0..@min(n, s.len - pos)];
}

/// `std::istringstream(s) >> i`: skip whitespace, an optional sign, digits;
/// fails with no digit or past `int`.
fn streamInt(s: []const u8) ?c_int {
    var i: usize = 0;
    while (i < s.len and std.ascii.isWhitespace(s[i])) i += 1;
    const st = i;
    if (i < s.len and (s[i] == '+' or s[i] == '-')) i += 1;
    const ds = i;
    while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    if (i == ds) return null;
    return std.fmt.parseInt(c_int, s[st..i], 10) catch null;
}

/// Ports `load_winogrande_from_csv` (perplexity.cpp:1031 @5266f24da).
fn loadWinograndeFromCsv(a: std.mem.Allocator, prompt: []const u8) !std.ArrayList(WinograndeEntry) {
    const func = "load_winogrande_from_csv";
    var result: std.ArrayList(WinograndeEntry) = .empty;
    var pos: usize = 0;
    var comma_pos: [4]usize = undefined;
    while (true) {
        // `std::getline` then `if (in.fail() || in.eof()) break;`: a last
        // line without its newline sets eof, and is dropped
        const nl = std.mem.indexOfScalarPos(u8, prompt, pos, '\n') orelse break;
        const line = prompt[pos..nl];
        pos = nl + 1;

        var ipos: usize = 0;
        var quote_open = false;
        for (line, 0..) |ch, i| {
            if (!quote_open) {
                if (ch == ',') {
                    comma_pos[ipos] = i;
                    ipos += 1;
                    if (ipos == 4) break;
                } else if (ch == '"') {
                    quote_open = true;
                }
            } else if (ch == '"') {
                quote_open = false;
            }
        }
        const zl = try a.dupeSentinel(u8, line, 0);
        if (ipos != 4) {
            log.err("%s: failed to find comma separators in <%s>\n", .{ func, zl.ptr });
            continue;
        }
        const sentence = if (line[comma_pos[0] + 1] == '"')
            substr(line, comma_pos[0] + 2, comma_pos[1] -% comma_pos[0] -% 3)
        else
            substr(line, comma_pos[0] + 1, comma_pos[1] - comma_pos[0] - 1);
        const choice1 = substr(line, comma_pos[1] + 1, comma_pos[2] - comma_pos[1] - 1);
        const choice2 = substr(line, comma_pos[2] + 1, comma_pos[3] - comma_pos[2] - 1);
        const answer = substr(line, comma_pos[3] + 1, line.len - comma_pos[3] - 1);
        const where = std.mem.indexOfScalar(u8, sentence, '_') orelse {
            log.err("%s: no _ in <%s>\n", .{ func, (try a.dupeSentinel(u8, sentence, 0)).ptr });
            continue;
        };
        // `answer.c_str()`: the stream stops at an embedded NUL
        const i_answer = streamInt(std.mem.sliceTo(answer, 0));
        if (i_answer == null or i_answer.? < 1 or i_answer.? > 2) {
            log.err("%s: failed to parse answer <%s>\n", .{ func, (try a.dupeSentinel(u8, answer, 0)).ptr });
            continue;
        }
        try result.append(a, .{
            .first = sentence[0..where],
            .second = sentence[where + 1 ..],
            .choices = .{ choice1, choice2 },
            .answer = i_answer.?,
        });
    }
    return result;
}

/// Ports `winogrande_score` (perplexity.cpp:1101 @5266f24da).
fn winograndeScore(a: std.mem.Allocator, ctx: ?*c.llama_context, params: *const Params) !void {
    const func = "winogrande_score";
    const model = c.llama_get_model(ctx);
    const vocab = c.llama_model_get_vocab(model);

    const k_min_trailing_ctx = 3;

    var data = try loadWinograndeFromCsv(a, params.prompt);
    if (data.items.len == 0) {
        log.err("%s: no tasks\n", .{func});
        return;
    }

    log.inf("%s : loaded %zu tasks from prompt.\n", .{ func, data.items.len });

    if (params.winogrande_tasks > 0 and params.winogrande_tasks < data.items.len) {
        log.inf("%s : selecting %zu random tasks\n", .{ func, params.winogrande_tasks });
        var rng = libcxx.Mt19937.init(1);
        var aux: std.ArrayList(c_int) = .empty;
        for (0..data.items.len) |i| try aux.append(a, @intCast(i));
        const scale: f32 = 1 / (1.0 + @as(f32, @floatFromInt(@as(u32, std.math.maxInt(u32)))));
        var selected: std.ArrayList(WinograndeEntry) = .empty;
        for (0..params.winogrande_tasks) |_| {
            const j: usize = @intFromFloat(scale * @as(f32, @floatFromInt(rng.next())) * @as(f32, @floatFromInt(aux.items.len)));
            try selected.append(a, data.items[@intCast(aux.items[j])]);
            aux.items[j] = aux.items[aux.items.len - 1];
            _ = aux.pop();
        }
        data = selected;
    }

    log.inf("%s : tokenizing selected tasks\n", .{func});

    for (data.items) |*task| {
        task.seq_tokens[0] = try tc.common.tokenize(a, vocab, try std.mem.concat(a, u8, &.{ task.first, task.choices[0], task.second }), true, false);
        task.seq_tokens[1] = try tc.common.tokenize(a, vocab, try std.mem.concat(a, u8, &.{ task.first, task.choices[1], task.second }), true, false);

        task.common_prefix = 0;
        for (0..task.seq_tokens[0].len) |k| {
            if (task.seq_tokens[0][k] != at(task.seq_tokens[1], k)) break;
            task.common_prefix += 1;
        }

        // TODO: the last token of each of the sequences don't need to be evaluated
        task.required_tokens = task.common_prefix +%
            task.seq_tokens[0].len -% task.common_prefix +%
            task.seq_tokens[1].len -% task.common_prefix;

        task.n_base1 = (try tc.common.tokenize(a, vocab, try std.mem.concat(a, u8, &.{ task.first, task.choices[0] }), true, false)).len;
        task.n_base2 = (try tc.common.tokenize(a, vocab, try std.mem.concat(a, u8, &.{ task.first, task.choices[1] }), true, false)).len;
    }

    log.inf("%s : calculating winogrande score over selected tasks.\n", .{func});

    const n_ctx: c_int = @intCast(c.llama_n_ctx(ctx));
    const n_batch = params.n_batch;

    const n_vocab: usize = @intCast(c.llama_vocab_n_tokens(vocab));

    const max_tasks_per_batch = 128;
    const max_seq: c_int = @min(2 * max_tasks_per_batch, @as(c_int, @intCast(c.llama_n_seq_max(ctx))));

    var batch = c.llama_batch_init(n_ctx, 0, 2);

    const batch_logits = try gpa.alloc(f32, @as(usize, @intCast(n_ctx)) * n_vocab);
    defer gpa.free(batch_logits);
    @memset(batch_logits, 0);

    var eval_pairs: std.ArrayList(EvalPair) = .empty;
    var eval_results: std.ArrayList(f32) = .empty;
    const n_workers = hardwareConcurrency();

    var n_correct: c_int = 0;
    var n_done: c_int = 0;

    const tasks = data.items;
    var j0: usize = 0;
    while (j0 < tasks.len) : (j0 += 1) {
        var n_cur: c_int = 0;

        var j1 = j0;
        var i_logits: usize = 0;

        tc.common.batchClear(&batch);

        while (n_cur + @as(c_int, @intCast(tasks[j1].required_tokens)) <= n_ctx) {
            var n_logits: usize = 0;
            const s0: c_int = @intCast(2 * (j1 - j0));
            if (s0 + 2 > max_seq) break;

            for (0..tasks[j1].common_prefix) |i| {
                tc.common.batchAdd(&batch, tasks[j1].seq_tokens[0][i], @intCast(i), &.{ s0 + 0, s0 + 1 }, false);
            }
            batch.logits[@intCast(batch.n_tokens - 1)] = 1;
            n_logits += 1;

            for (0..2) |s| {
                // TODO: end before the last token, no need to predict past the end of the sequences
                var i = tasks[j1].common_prefix;
                while (i < tasks[j1].seq_tokens[s].len) : (i += 1) {
                    tc.common.batchAdd(&batch, tasks[j1].seq_tokens[s][i], @intCast(i), &.{s0 + @as(c_int, @intCast(s))}, true);
                    n_logits += 1;
                }
            }

            tasks[j1].i_logits = i_logits;
            i_logits += n_logits;

            n_cur += @intCast(tasks[j1].required_tokens);
            j1 += 1;
            if (j1 == tasks.len) break;
        }

        if (j0 == j1) {
            log.err("%s : task %zu does not fit in the context window (requires %zu tokens)\n", .{ func, j0, tasks[j0].required_tokens });
            return;
        }

        c.llama_memory_clear(c.llama_get_memory(ctx), true);

        // decode all tasks [j0, j1)
        if (!decodeHelper(ctx, &batch, batch_logits, n_batch, n_vocab)) {
            log.err("%s: llama_decode() failed\n", .{func});
            return;
        }

        eval_pairs.clearRetainingCapacity();
        for (j0..j1) |i| {
            const task = &tasks[i];

            const skip_choice = task.seq_tokens[0].len -% task.common_prefix > k_min_trailing_ctx and
                task.seq_tokens[1].len -% task.common_prefix > k_min_trailing_ctx;

            const n_base1 = if (skip_choice) task.n_base1 else task.common_prefix;
            const last_1st: usize = if (task.seq_tokens[0].len -% n_base1 > 1) 1 else 0;
            var li = n_base1 -% task.common_prefix;
            var j = n_base1 -% 1;
            while (j < task.seq_tokens[0].len -% 1 -% last_1st) : (j += 1) {
                try eval_pairs.append(a, .{ .i_logits = task.i_logits +% li, .token = task.seq_tokens[0][j + 1] });
                li +%= 1;
            }
            const n_base2 = if (skip_choice) task.n_base2 else task.common_prefix;
            const last_2nd: usize = if (task.seq_tokens[1].len -% n_base2 > 1) 1 else 0;
            // FIXME: this uses the wrong first logits when not skipping the choice word
            li = task.seq_tokens[0].len -% task.common_prefix +% n_base2 -% task.common_prefix;
            j = n_base2 -% 1;
            while (j < task.seq_tokens[1].len -% 1 -% last_2nd) : (j += 1) {
                try eval_pairs.append(a, .{ .i_logits = task.i_logits +% li, .token = task.seq_tokens[1][j + 1] });
                li +%= 1;
            }
        }
        try computeLogprobs(a, batch_logits, n_vocab, n_workers, eval_pairs.items, &eval_results);

        var ir: usize = 0;
        for (j0..j1) |i| {
            const task = &tasks[i];

            const skip_choice = task.seq_tokens[0].len -% task.common_prefix > k_min_trailing_ctx and
                task.seq_tokens[1].len -% task.common_prefix > k_min_trailing_ctx;

            var score_1st: f32 = 0;
            const n_base1 = if (skip_choice) task.n_base1 else task.common_prefix;
            const last_1st: usize = if (task.seq_tokens[0].len -% n_base1 > 1) 1 else 0;
            var j = n_base1 -% 1;
            while (j < task.seq_tokens[0].len -% 1 -% last_1st) : (j += 1) {
                score_1st += eval_results.items[ir];
                ir += 1;
            }
            score_1st /= @floatFromInt(task.seq_tokens[0].len -% n_base1 -% last_1st);

            var score_2nd: f32 = 0;
            const n_base2 = if (skip_choice) task.n_base2 else task.common_prefix;
            const last_2nd: usize = if (task.seq_tokens[1].len -% n_base2 > 1) 1 else 0;
            j = n_base2 -% 1;
            while (j < task.seq_tokens[1].len -% 1 -% last_2nd) : (j += 1) {
                score_2nd += eval_results.items[ir];
                ir += 1;
            }
            score_2nd /= @floatFromInt(task.seq_tokens[1].len -% n_base2 -% last_2nd);

            const result: c_int = if (score_1st > score_2nd) 1 else 2;

            if (result == task.answer) n_correct += 1;
            n_done += 1;

            // print the accumulated accuracy mean x 100
            log.out("%zu\t%.4lf\t%10.6f  %10.6f  %d  %d\n", .{ i + 1, 100.0 * @as(f64, @floatFromInt(n_correct)) / @as(f64, @floatFromInt(n_done)), @as(f64, score_1st), @as(f64, score_2nd), result, task.answer });
        }

        j0 = j1 - 1;
    }

    log.out("\n", .{});

    if (n_done < 100) return;

    const p: f32 = 1.0 * @as(f32, @floatFromInt(n_correct)) / @as(f32, @floatFromInt(n_done));
    const sigma: f32 = 100.0 * libm.sqrtf(p * (1 - p) / @as(f32, @floatFromInt(n_done - 1)));

    log.inf("Final Winogrande score(%d tasks): %.4lf +/- %.4lf\n", .{ n_done, @as(f64, 100 * p), @as(f64, sigma) });
}

/// A `std::istringstream` over binary data, as far as `read` and `seekg`
/// go: a short read takes what is there and sets the fail bit.
const ByteStream = struct {
    data: []const u8,
    pos: usize = 0,
    failed: bool = false,

    fn read(self: *ByteStream, dst: []u8) void {
        if (self.failed) return;
        const n = @min(dst.len, self.data.len - self.pos);
        @memcpy(dst[0..n], self.data[self.pos..][0..n]);
        self.pos += n;
        if (n < dst.len) self.failed = true;
    }

    fn readInt(self: *ByteStream, comptime T: type) T {
        var v: T = 0;
        self.read(std.mem.asBytes(&v));
        return v;
    }

    fn seekg(self: *ByteStream, p: usize) void {
        if (self.failed) return;
        if (p > self.data.len) self.failed = true else self.pos = p;
    }
};

/// Ports `deserialize_string` (perplexity.cpp:1303 @5266f24da).
fn deserializeString(a: std.mem.Allocator, in: *ByteStream) !?[]const u8 {
    const size = in.readInt(u32);
    if (in.failed) return null;
    const str = try a.alloc(u8, size);
    @memset(str, 0);
    in.read(str);
    if (in.failed) return null;
    return str;
}

/// Mirrors `multiple_choice_answers` (perplexity.cpp:1312 @5266f24da).
const McAnswers = struct {
    answers: [][]const u8 = &.{},
    labels: []c_int = &.{},

    fn deserialize(self: *McAnswers, a: std.mem.Allocator, in: *ByteStream) !bool {
        const n = in.readInt(u32);
        if (in.failed or n > 100) return false; // 100 as max. number of answers should be good enough for any practical purpose
        self.answers = try a.alloc([]const u8, n);
        for (self.answers) |*s| s.* = "";
        self.labels = try a.alloc(c_int, n);
        @memset(self.labels, 0);
        for (self.answers) |*ans| {
            ans.* = (try deserializeString(a, in)) orelse return false;
        }
        in.read(std.mem.sliceAsBytes(self.labels));
        return !in.failed;
    }
};

/// Mirrors `multiple_choice_task` (perplexity.cpp:1329 @5266f24da).
const McTask = struct {
    question: []const u8 = "", // the question (or context that needs to be continued)
    mc1: McAnswers = .{}, // possible answers (continuations) with a single correct answer
    mc2: McAnswers = .{}, // possible answers (continuations) with multiple correct answers - not handled yet

    // For evaluation
    i_logits: usize = 0,
    common_prefix: usize = 0,
    required_tokens: usize = 0,
    seq_tokens: std.ArrayList([]c.llama_token) = .empty,
    log_probs: []f32 = &.{},

    fn deserialize(self: *McTask, a: std.mem.Allocator, in: *ByteStream) !bool {
        self.question = (try deserializeString(a, in)) orelse return false;
        return try self.mc1.deserialize(a, in) and try self.mc2.deserialize(a, in);
    }
};

/// Ports `multiple_choice_prepare_one_task` (perplexity.cpp:1346 @5266f24da).
fn mcPrepareOneTask(a: std.mem.Allocator, vocab: ?*const c.llama_vocab, task: *McTask, log_error: bool) !bool {
    const func = "multiple_choice_prepare_one_task";
    if (task.question.len == 0 or task.mc1.answers.len == 0) {
        if (log_error) log.err("%s: found bad task with empty question and/or answers\n", .{func});
        return false;
    }
    try task.seq_tokens.ensureTotalCapacity(a, task.mc1.answers.len);
    for (task.mc1.answers) |answer| {
        if (answer.len == 0) {
            if (log_error) log.err("%s: found empty answer\n", .{func});
            return false;
        }
        try task.seq_tokens.append(a, try tc.common.tokenize(a, vocab, try std.mem.concat(a, u8, &.{ task.question, " ", answer }), true, false));
    }
    var min_len = task.seq_tokens.items[0].len;
    for (task.seq_tokens.items) |seq| min_len = @min(min_len, seq.len);
    task.common_prefix = 0;
    for (0..min_len) |k| {
        const token = task.seq_tokens.items[0][k];
        var all_same = true;
        for (task.seq_tokens.items[1..]) |seq| {
            if (seq[k] != token) {
                all_same = false;
                break;
            }
        }
        if (!all_same) break;
        task.common_prefix += 1;
    }
    task.required_tokens = task.common_prefix;
    for (task.seq_tokens.items) |seq| task.required_tokens += seq.len - task.common_prefix;
    return true;
}

/// What `multiple_choice_score`'s preparation threads share.
const PrepareWork = struct {
    vocab: ?*const c.llama_vocab,
    tasks: []McTask,
    counter: std.atomic.Value(c_int) = .init(0),
    n_bad: std.atomic.Value(c_int) = .init(0),

    fn prepare(self: *PrepareWork) void {
        const num_tasks: c_int = @intCast(self.tasks.len);
        var n_bad_local: c_int = 0;
        // each thread tokenizes into its own arena: the shared one is not
        // thread-safe
        var arena: std.heap.ArenaAllocator = .init(gpa);
        while (true) {
            const first = self.counter.fetchAdd(k_token_chunk, .seq_cst);
            if (first >= num_tasks) {
                if (n_bad_local > 0) _ = self.n_bad.fetchAdd(n_bad_local, .seq_cst);
                break;
            }
            const last = @min(first + k_token_chunk, num_tasks);
            var i = first;
            while (i < last) : (i += 1) {
                const ok = mcPrepareOneTask(arena.allocator(), self.vocab, &self.tasks[@intCast(i)], false) catch false;
                if (!ok) n_bad_local += 1;
            }
        }
        // the tokens live as long as the tasks: the arena is not freed
    }
};

/// Ports `multiple_choice_score` (perplexity.cpp:1405 @5266f24da).
fn multipleChoiceScore(a: std.mem.Allocator, ctx: ?*c.llama_context, params: *const Params) !void {
    const func = "multiple_choice_score";
    const model = c.llama_get_model(ctx);
    const vocab = c.llama_model_get_vocab(model);

    var strstream: ByteStream = .{ .data = params.prompt };
    var n_task = strstream.readInt(u32);
    if (strstream.failed or n_task == 0) {
        log.err("%s: no tasks\n", .{func});
        return;
    }
    log.inf("%s: there are %u tasks in prompt\n", .{ func, n_task });
    const task_pos = try a.alloc(u32, n_task);
    @memset(task_pos, 0);
    strstream.read(std.mem.sliceAsBytes(task_pos));
    if (strstream.failed) {
        log.err("%s: failed to read task positions from prompt\n", .{func});
        return;
    }

    var tasks: []McTask = undefined;
    if (params.multiple_choice_tasks == 0 or params.multiple_choice_tasks >= n_task) {
        // Use all tasks
        tasks = try a.alloc(McTask, n_task);
        for (tasks) |*t| t.* = .{};
        log.inf("%s: reading tasks", .{func});
        const n_dot: c_int = @max(@as(c_int, @intCast(n_task / 100)), 1);
        var i: c_int = 0;
        for (tasks) |*task| {
            i += 1;
            if (!try task.deserialize(a, &strstream)) {
                log.err("%s: failed to read task %d of %u\n", .{ func, i, n_task });
                return;
            }
            if (@rem(i, n_dot) == 0) log.out(".", .{});
        }
        log.out("done\n", .{});
    } else {
        log.inf("%s: selecting %zu random tasks from %u tasks available\n", .{ func, params.multiple_choice_tasks, n_task });
        var rng = libcxx.Mt19937.init(1);
        var aux: std.ArrayList(c_int) = .empty;
        for (0..n_task) |i| try aux.append(a, @intCast(i));
        const scale: f32 = 1.0 / (1.0 + @as(f32, @floatFromInt(@as(u32, std.math.maxInt(u32)))));
        tasks = try a.alloc(McTask, params.multiple_choice_tasks);
        for (tasks) |*t| t.* = .{};
        for (tasks) |*task| {
            const j: usize = @intFromFloat(scale * @as(f32, @floatFromInt(rng.next())) * @as(f32, @floatFromInt(aux.items.len)));
            const idx = aux.items[j];
            aux.items[j] = aux.items[aux.items.len - 1];
            _ = aux.pop();
            strstream.seekg(task_pos[@intCast(idx)]);
            if (!try task.deserialize(a, &strstream)) {
                log.err("%s: failed to read task %d at position %u\n", .{ func, idx, task_pos[@intCast(idx)] });
                return;
            }
        }
        n_task = @intCast(params.multiple_choice_tasks);
    }

    log.inf("%s: preparing task data", .{func});
    if (n_task > 500) {
        log.out("...", .{});
        var work: PrepareWork = .{ .vocab = vocab, .tasks = tasks };
        var max_thread = hardwareConcurrency();
        max_thread = @min(max_thread, (tasks.len + k_token_chunk - 1) / k_token_chunk);
        const threads = try gpa.alloc(std.Thread, max_thread - 1);
        defer gpa.free(threads);
        for (threads) |*w| w.* = try std.Thread.spawn(.{}, PrepareWork.prepare, .{&work});
        work.prepare();
        for (threads) |w| w.join();
        log.out("done\n", .{});
        const nbad = work.n_bad.load(.seq_cst);
        if (nbad > 0) {
            log.err("%s: found %d malformed tasks\n", .{ func, nbad });
            return;
        }
    } else {
        const n_dot: c_int = @max(@as(c_int, @intCast(n_task / 100)), 1);
        var i_task: c_int = 0;
        for (tasks) |*task| {
            i_task += 1;
            if (!try mcPrepareOneTask(a, vocab, task, true)) return;
            if (@rem(i_task, n_dot) == 0) log.out(".", .{});
        }
        log.out("done\n", .{});
    }

    log.inf("%s : calculating TruthfulQA score over %zu tasks.\n", .{ func, tasks.len });

    log.out("\ntask\tacc_norm\n", .{});

    const n_ctx: c_int = @intCast(c.llama_n_ctx(ctx));
    const n_batch = params.n_batch;

    const n_vocab: usize = @intCast(c.llama_vocab_n_tokens(vocab));

    const max_tasks_per_batch = 32;
    const max_seq: c_int = @min(4 * max_tasks_per_batch, @as(c_int, @intCast(c.llama_n_seq_max(ctx))));

    var batch = c.llama_batch_init(n_ctx, 0, max_seq);

    const tok_logits = try a.alloc(f32, n_vocab);
    const batch_logits = try gpa.alloc(f32, @as(usize, @intCast(n_ctx)) * n_vocab);
    defer gpa.free(batch_logits);
    @memset(batch_logits, 0);

    var eval_pairs: std.ArrayList(EvalPair) = .empty;
    var eval_results: std.ArrayList(f32) = .empty;
    const n_workers = hardwareConcurrency();
    var batch_indeces: std.ArrayList(c.llama_seq_id) = .empty;

    var n_done: c_int = 0;
    var n_correct: c_int = 0;
    var n_tot_answers: c_int = 0;

    var j0: usize = 0;
    while (j0 < tasks.len) : (j0 += 1) {
        var n_cur: c_int = 0;

        var j1 = j0;
        var i_logits: usize = 0; // this tells us how many logits were needed before this point in the batch

        tc.common.batchClear(&batch);

        // batch as much tasks as possible into the available context
        var s0: c_int = 0;
        while (n_cur + @as(c_int, @intCast(tasks[j1].required_tokens)) <= n_ctx) {
            const cur_task = &tasks[j1];
            var n_logits: usize = 0;

            const num_answers: c_int = @intCast(cur_task.seq_tokens.items.len);
            if (s0 + num_answers > max_seq) {
                if (s0 == 0) {
                    log.err("%s : task %zu requires a higher -np|--parallel value (at least %d)\n", .{ func, j0, num_answers });
                    return;
                }
                break;
            }

            try batch_indeces.resize(a, @intCast(num_answers));
            for (batch_indeces.items, 0..) |*b, s| b.* = s0 + @as(c_int, @intCast(s));

            for (0..cur_task.common_prefix) |i| {
                tc.common.batchAdd(&batch, cur_task.seq_tokens.items[0][i], @intCast(i), batch_indeces.items, false);
            }
            batch.logits[@intCast(batch.n_tokens - 1)] = 1; // we need logits for the last token of the common prefix
            n_logits += 1;

            for (cur_task.seq_tokens.items, 0..) |seq, s| {
                const seq_tokens_size = seq.len;
                var i = cur_task.common_prefix;
                while (i < seq_tokens_size) : (i += 1) {
                    const needs_logits = i < seq_tokens_size - 1;
                    tc.common.batchAdd(&batch, seq[i], @intCast(i), &.{s0 + @as(c_int, @intCast(s))}, needs_logits);
                    n_logits += @intFromBool(needs_logits);
                }
            }

            s0 += num_answers;

            cur_task.i_logits = i_logits;
            i_logits += n_logits;

            n_cur += @intCast(cur_task.required_tokens);
            j1 += 1;
            if (j1 == tasks.len) break;
        }

        if (j0 == j1) {
            log.err("%s : task %zu does not fit in the context window (requires %zu tokens)\n", .{ func, j0, tasks[j0].required_tokens });
            return;
        }

        c.llama_memory_clear(c.llama_get_memory(ctx), true);

        // decode all tasks [j0, j1)
        if (!decodeHelper(ctx, &batch, batch_logits, n_batch, n_vocab)) {
            log.err("%s: llama_decode() failed\n", .{func});
            return;
        }

        // Compute log-probs in parallel
        // First we collect all tasks
        eval_pairs.clearRetainingCapacity();
        for (j0..j1) |i| {
            const cur_task = &tasks[i];
            var li: usize = 1; // skip the last logit of the common prefix (computed separately below)
            for (cur_task.seq_tokens.items) |seq| {
                var j = cur_task.common_prefix;
                while (j < seq.len - 1) : (j += 1) {
                    try eval_pairs.append(a, .{ .i_logits = cur_task.i_logits + li, .token = seq[j + 1] });
                    li += 1;
                }
            }
        }
        // Then we do the actual calculation
        try computeLogprobs(a, batch_logits, n_vocab, n_workers, eval_pairs.items, &eval_results);

        var ir: usize = 0;

        // compute the logprobs for each ending of the decoded tasks
        for (j0..j1) |i| {
            const cur_task = &tasks[i];

            // get the logits of the last token of the common prefix
            @memcpy(tok_logits, batch_logits[cur_task.i_logits * n_vocab ..][0..n_vocab]);

            const first_probs = try softmax(a, tok_logits);

            cur_task.log_probs = try a.alloc(f32, cur_task.seq_tokens.items.len);
            for (cur_task.seq_tokens.items, 0..) |seq, s| {
                var count: usize = 1;
                var log_prob = libm.logf(first_probs[@intCast(seq[cur_task.common_prefix])]);
                var j = cur_task.common_prefix;
                while (j < seq.len - 1) : (j += 1) {
                    count += 1;
                    log_prob += eval_results.items[ir];
                    ir += 1;
                }
                cur_task.log_probs[s] = log_prob / @as(f32, @floatFromInt(count));
            }
            a.free(first_probs);

            // Find the ending with maximum logprob
            var logprob_max_idx: usize = 0;
            var logprob_max_val = cur_task.log_probs[0];
            for (1..cur_task.log_probs.len) |s| {
                if (cur_task.log_probs[s] > logprob_max_val) {
                    logprob_max_val = cur_task.log_probs[s];
                    logprob_max_idx = s;
                }
            }

            n_tot_answers += @intCast(cur_task.log_probs.len);
            if (cur_task.mc1.labels[logprob_max_idx] == 1) n_correct += 1;
            n_done += 1;

            // Print the accumulated accuracy mean x 100
            log.out("%d\t%.8lf\n", .{ n_done, 100.0 * @as(f64, @floatFromInt(n_correct)) / @as(f64, @floatFromInt(n_done)) });
        }

        j0 = j1 - 1;
    }

    c.llama_batch_free(batch);

    if (n_done < 100 and (params.multiple_choice_tasks != 0 and params.multiple_choice_tasks < n_task)) return;

    var p: f32 = 1.0 * @as(f32, @floatFromInt(n_correct)) / @as(f32, @floatFromInt(n_done));
    var sigma = libm.sqrtf(p * (1 - p) / @as(f32, @floatFromInt(n_done - 1)));
    log.out("\n", .{});
    log.inf("Final result: %.4f +/- %.4f\n", .{ @as(f64, 100.0 * p), @as(f64, 100.0 * sigma) });
    p = 1.0 * @as(f32, @floatFromInt(n_done)) / @as(f32, @floatFromInt(n_tot_answers));
    sigma = libm.sqrtf(p * (1 - p) / @as(f32, @floatFromInt(n_done - 1)));
    log.inf("Random chance: %.4f +/- %.4f\n", .{ @as(f64, 100.0 * p), @as(f64, 100.0 * sigma) });

    log.inf("\n", .{});
}

/// `kl_divergence`'s `mean_and_uncertainty` (perplexity.cpp:1770 @5266f24da).
fn meanAndUncertainty(sum: f64, sum2: f64, count: usize) struct { f64, f64 } {
    if (count < 1) return .{ 0.0, 0.0 };
    const n: f64 = @floatFromInt(count);
    const f = sum / n;
    var df = @mulAdd(f64, -f, f, sum2 / n);
    df = if (df > 0 and count > 10) libm.sqrt(df / @as(f64, @floatFromInt(count - 1))) else 0.0;
    return .{ f, df };
}

/// `kl_divergence`'s `covariance` (perplexity.cpp:1779 @5266f24da).
fn covariance(suma: f64, sumb: f64, sumab: f64, count: usize) f64 {
    if (count < 10) return 0.0;
    const n: f64 = @floatFromInt(count);
    var v = @mulAdd(f64, -(suma / n), sumb / n, sumab / n);
    v /= @floatFromInt(count - 1);
    return v;
}

/// `kl_divergence`'s `percentile` (perplexity.cpp:1953 @5266f24da).
fn percentile(values: []const f32, fraction: f32) f32 {
    if (fraction <= 0) return values[0];
    if (fraction >= 1) return values[values.len - 1];
    var p = fraction * @as(f32, @floatFromInt(values.len - 1));
    const ip: usize = @intFromFloat(p);
    p -= @floatFromInt(ip);
    return @mulAdd(f32, 1 - p, values[ip], p * values[@min(ip + 1, values.len - 1)]);
}

fn lessF32(_: void, x: f32, y: f32) bool {
    return x < y;
}

/// `sqrt(a*a + b*b - 2.0*cov)` as clang contracts it.
fn sqrtUnc(x: f64, y: f64, cov: f64) f64 {
    return libm.sqrt(@mulAdd(f64, -2.0, cov, @mulAdd(f64, x, x, y * y)));
}

/// Ports `kl_divergence` (perplexity.cpp:1695 @5266f24da).
fn klDivergence(a: std.mem.Allocator, ctx: ?*c.llama_context, params: *const Params) !void {
    const func = "kl_divergence";
    const model = c.llama_get_model(ctx);
    const vocab = c.llama_model_get_vocab(model);

    if (params.logits_file.len == 0) {
        log.err("%s: you must provide a name of a file containing the log probabilities of the base model\n", .{func});
        return;
    }
    const fname = try a.dupeSentinel(u8, params.logits_file, 0);
    const f = std.c.fopen(fname.ptr, "rb") orelse {
        log.err("%s: failed to open %s\n", .{ func, fname.ptr });
        return;
    };
    defer _ = std.c.fclose(f);
    const In = struct {
        file: *std.c.FILE,
        failed: bool = false,
        fn read(self: *@This(), dst: []u8) bool {
            if (self.failed) return false;
            if (std.c.fread(dst.ptr, 1, dst.len, self.file) != dst.len) self.failed = true;
            return !self.failed;
        }
    };
    var in: In = .{ .file = f };
    {
        var check: [8]u8 = undefined;
        if (!in.read(&check) or !std.mem.eql(u8, &check, "_logits_")) {
            log.err("%s: %s does not look like a file containing log-probabilities\n", .{ func, fname.ptr });
            return;
        }
    }

    var n_ctx: u32 = 0;
    _ = in.read(std.mem.asBytes(&n_ctx));
    if (n_ctx > c.llama_n_ctx(ctx)) {
        log.err("%s: %s has been computed with %u, while the current context is %d. Increase it with -c and retry\n", .{ func, fname.ptr, n_ctx, params.n_ctx });
    }

    var n_vocab: c_int = 0;
    var n_chunk: c_int = 0;
    _ = in.read(std.mem.asBytes(&n_vocab));
    _ = in.read(std.mem.asBytes(&n_chunk));
    if (in.failed) {
        log.err("%s: failed reading n_vocab, n_chunk from %s\n", .{ func, fname.ptr });
        return;
    }
    if (n_vocab != c.llama_vocab_n_tokens(vocab)) {
        log.err("%s: inconsistent vocabulary (%d vs %d)\n", .{ func, n_vocab, c.llama_vocab_n_tokens(vocab) });
    }

    const tokens = try gpa.alloc(c.llama_token, @as(usize, n_ctx) * @as(usize, @intCast(n_chunk)));
    defer gpa.free(tokens);
    @memset(tokens, 0);
    if (!in.read(std.mem.sliceAsBytes(tokens))) {
        log.err("%s: failed reading evaluation tokens from %s\n", .{ func, fname.ptr });
        return;
    }

    const n_batch = params.n_batch;
    const n_ctx_i: c_int = @intCast(n_ctx);
    const num_batches = @divTrunc(n_ctx_i + n_batch - 1, n_batch);
    // Calculate n_seq based on the logits file's n_ctx, but cap it at what the context supports
    const n_seq_max: c_int = @intCast(c.llama_n_seq_max(ctx));
    var n_seq: c_int = @max(1, @divTrunc(n_batch, n_ctx_i));
    if (n_seq > n_seq_max) {
        log.wrn("%s: calculated n_seq=%d exceeds context's n_seq_max=%d, capping at %d\n", .{ func, n_seq, n_seq_max, n_seq_max });
        n_seq = n_seq_max;
    }
    const nv: usize = @intCast(2 * @divTrunc(n_vocab + 1, 2) + 4);
    const add_bos = c.llama_vocab_get_add_bos(vocab);
    ggmlAssert(!c.llama_vocab_get_add_eos(vocab), 1754, "!llama_vocab_get_add_eos(vocab)");

    var batch = c.llama_batch_init(@min(n_batch, n_ctx_i * n_seq), 0, 1);

    const n_tok_chunk: usize = n_ctx - 1 - n_ctx / 2;
    const log_probs_uint16 = try gpa.alloc(u16, n_tok_chunk * nv);
    defer gpa.free(log_probs_uint16);
    @memset(log_probs_uint16, 0);
    const kld_values = try gpa.alloc(f32, n_tok_chunk * @as(usize, @intCast(n_chunk)));
    defer gpa.free(kld_values);
    @memset(kld_values, 0);
    const p_diff_values = try gpa.alloc(f32, n_tok_chunk * @as(usize, @intCast(n_chunk)));
    defer gpa.free(p_diff_values);
    @memset(p_diff_values, 0);
    const nvu: usize = @intCast(n_vocab);
    var logits: std.ArrayList(f32) = .empty;
    defer logits.deinit(gpa);
    if (num_batches > 1) try logits.ensureTotalCapacity(gpa, @as(usize, n_ctx) * nvu);

    log.inf("%s: computing over %d chunks, n_ctx=%u, batch_size=%d, n_seq=%d\n", .{ func, n_chunk, n_ctx, n_batch, n_seq });

    const n_workers = hardwareConcurrency() - 1;

    var kld: KldResult = .{};
    var kld_off: usize = 0;

    const first = n_ctx_i >> 1;

    var i: c_int = 0;
    while (i < n_chunk) : (i += n_seq) {
        const start = i * n_ctx_i;
        const end = start + n_ctx_i;

        const n_seq_batch = @min(n_seq, n_chunk - i);

        const t_start = nowNs();

        // clear the KV cache
        c.llama_memory_clear(c.llama_get_memory(ctx), true);

        var j: c_int = 0;
        while (j < num_batches) : (j += 1) {
            const batch_start = start + j * n_batch;
            const batch_size = @min(end - batch_start, n_batch);

            var n_outputs: c_int = 0;

            tc.common.batchClear(&batch);
            var seq: c_int = 0;
            while (seq < n_seq_batch) : (seq += 1) {
                const seq_start: usize = @intCast(batch_start + seq * n_ctx_i);

                // save original token and restore it after eval
                const token_org = tokens[seq_start];

                // add BOS token for the first batch of each chunk
                if (add_bos and j == 0) tokens[seq_start] = c.llama_vocab_bos(vocab);

                var k: c_int = 0;
                while (k < batch_size) : (k += 1) {
                    const pos = j * n_batch + k;
                    const need_logits = pos >= first;
                    tc.common.batchAdd(&batch, tokens[seq_start + @as(usize, @intCast(k))], pos, &.{seq}, need_logits);
                    n_outputs += @intFromBool(need_logits);
                }

                // restore the original token in case it was set to BOS
                tokens[seq_start] = token_org;
            }

            if (c.llama_decode(ctx, batch) != 0) {
                log.err("%s : failed to decode\n", .{func});
                c.llama_batch_free(batch);
                return;
            }

            if (num_batches > 1 and n_outputs > 0) {
                const batch_logits = c.llama_get_logits(ctx);
                try logits.appendSlice(gpa, batch_logits[0 .. @as(usize, @intCast(n_outputs)) * nvu]);
            }
        }

        if (i == 0) {
            c.llama_synchronize(ctx);
            printEta(func, t_start, @as(f32, @floatFromInt(n_chunk)) / @as(f32, @floatFromInt(n_seq)));
            log.out("\n", .{});
            log.out("chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p\n", .{});
        }

        // Read log probs for each sequence in the batch
        var seq: c_int = 0;
        while (seq < n_seq_batch) : (seq += 1) {
            if (!in.read(std.mem.sliceAsBytes(log_probs_uint16))) {
                log.err("%s: failed reading log-probs for chunk %d\n", .{ func, i + seq });
                c.llama_batch_free(batch);
                return;
            }

            const all_logits: [*]const f32 = if (num_batches > 1) logits.items.ptr else c.llama_get_logits_ith(ctx, seq * n_ctx_i + first);

            var work: KldWork = .{
                .n_vocab = nvu,
                .logits = all_logits,
                .tokens = tokens.ptr + @as(usize, @intCast(start + seq * n_ctx_i + first)),
                .n_token = n_tok_chunk,
                .base_log_probs = log_probs_uint16.ptr,
                .nv = nv,
                .kld_values = kld_values.ptr + kld_off,
                .p_diff_values = p_diff_values.ptr + kld_off,
            };
            try work.run(n_workers, &kld);
            kld_off += n_tok_chunk;

            log.out("%4d", .{i + seq + 1});

            const log_ppl = meanAndUncertainty(kld.sum_nll, kld.sum_nll2, kld.count);
            const ppl_val = libm.exp(log_ppl[0]);
            const ppl_unc = ppl_val * log_ppl[1];
            log.out("    %9.4lf ± %9.4lf", .{ ppl_val, ppl_unc });

            const log_ppl_base = meanAndUncertainty(kld.sum_nll_base, kld.sum_nll_base2, kld.count);
            const log_ppl_cov = covariance(kld.sum_nll, kld.sum_nll_base, kld.sum_nll_nll_base, kld.count);
            const log_ppl_ratio_val = log_ppl[0] - log_ppl_base[0];
            const log_ppl_ratio_unc = sqrtUnc(log_ppl[1], log_ppl_base[1], log_ppl_cov);
            log.out("    %10.5lf ± %10.5lf", .{ log_ppl_ratio_val, log_ppl_ratio_unc });

            const kl_div = meanAndUncertainty(kld.sum_kld, kld.sum_kld2, kld.count);
            log.out("    %10.5lf ± %10.5lf", .{ kl_div[0], kl_div[1] });

            const p_diff_mse = meanAndUncertainty(kld.sum_p_diff2, kld.sum_p_diff4, kld.count);
            const p_diff_rms_val = libm.sqrt(p_diff_mse[0]);
            const p_diff_rms_unc = 0.5 / p_diff_rms_val * p_diff_mse[1];
            log.out("    %6.3lf ± %6.3lf %%", .{ 100.0 * p_diff_rms_val, 100.0 * p_diff_rms_unc });

            const p_top_val = 1.0 * @as(f64, @floatFromInt(kld.n_same_top)) / @as(f64, @floatFromInt(kld.count));
            const p_top_unc = libm.sqrt(p_top_val * (1 - p_top_val) / @as(f64, @floatFromInt(kld.count -% 1)));
            log.out("    %6.3lf ± %6.3lf %%", .{ 100.0 * p_top_val, 100.0 * p_top_unc });

            log.out("\n", .{});
        }

        logits.clearRetainingCapacity();
    }

    c.llama_batch_free(batch);
    log.out("\n", .{});

    if (kld.count < 100) return; // we do not wish to do statistics on so few values

    libcxx.sort(f32, kld_values, {}, lessF32);
    libcxx.sort(f32, p_diff_values, {}, lessF32);

    log.out("====== Perplexity statistics ======\n", .{});

    const log_ppl = meanAndUncertainty(kld.sum_nll, kld.sum_nll2, kld.count);
    const ppl_val = libm.exp(log_ppl[0]);
    const ppl_unc = ppl_val * log_ppl[1];
    log.out("Mean PPL(Q)                   : %10.6lf ± %10.6lf\n", .{ ppl_val, ppl_unc });

    const log_ppl_base = meanAndUncertainty(kld.sum_nll_base, kld.sum_nll_base2, kld.count);
    const ppl_base_val = libm.exp(log_ppl_base[0]);
    const ppl_base_unc = ppl_base_val * log_ppl_base[1];
    log.out("Mean PPL(base)                : %10.6lf ± %10.6lf\n", .{ ppl_base_val, ppl_base_unc });

    const log_ppl_cov = covariance(kld.sum_nll, kld.sum_nll_base, kld.sum_nll_nll_base, kld.count);
    const log_ppl_cor = log_ppl_cov / (log_ppl[1] * log_ppl_base[1]);
    log.out("Cor(ln(PPL(Q)), ln(PPL(base))): %6.2lf%%\n", .{100.0 * log_ppl_cor});

    const log_ppl_ratio_val = log_ppl[0] - log_ppl_base[0];
    const log_ppl_ratio_unc = sqrtUnc(log_ppl[1], log_ppl_base[1], log_ppl_cov);
    log.out("Mean ln(PPL(Q)/PPL(base))     : %10.6lf ± %10.6lf\n", .{ log_ppl_ratio_val, log_ppl_ratio_unc });

    const ppl_ratio_val = libm.exp(log_ppl_ratio_val);
    const ppl_ratio_unc = ppl_ratio_val * log_ppl_ratio_unc;
    log.out("Mean PPL(Q)/PPL(base)         : %10.6lf ± %10.6lf\n", .{ ppl_ratio_val, ppl_ratio_unc });

    const ppl_cov = ppl_val * ppl_base_val * log_ppl_cov;
    const ppl_diff_val = ppl_val - ppl_base_val;
    const ppl_diff_unc = sqrtUnc(ppl_unc, ppl_base_unc, ppl_cov);
    log.out("Mean PPL(Q)-PPL(base)         : %10.6lf ± %10.6lf\n", .{ ppl_diff_val, ppl_diff_unc });

    log.out("\n", .{});

    log.out("====== KL divergence statistics ======\n", .{});
    const kl_div = meanAndUncertainty(kld.sum_kld, kld.sum_kld2, kld.count);
    log.out("Mean    KLD: %10.6lf ± %10.6lf\n", .{ kl_div[0], kl_div[1] });
    const kn = kld_values.len;
    const kld_median: f32 = if (kn % 2 == 0) 0.5 * (kld_values[kn / 2] + kld_values[kn / 2 - 1]) else kld_values[kn / 2];

    const D = struct {
        fn d(x: f32) f64 {
            return x;
        }
    };
    log.out("Maximum KLD: %10.6f\n", .{D.d(kld_values[kn - 1])});
    log.out("99.9%%   KLD: %10.6f\n", .{D.d(percentile(kld_values, 0.999))});
    log.out("99.0%%   KLD: %10.6f\n", .{D.d(percentile(kld_values, 0.990))});
    log.out("95.0%%   KLD: %10.6f\n", .{D.d(percentile(kld_values, 0.950))});
    log.out("90.0%%   KLD: %10.6f\n", .{D.d(percentile(kld_values, 0.900))});
    log.out("Median  KLD: %10.6f\n", .{D.d(kld_median)});
    log.out("10.0%%   KLD: %10.6f\n", .{D.d(percentile(kld_values, 0.100))});
    log.out(" 5.0%%   KLD: %10.6f\n", .{D.d(percentile(kld_values, 0.050))});
    log.out(" 1.0%%   KLD: %10.6f\n", .{D.d(percentile(kld_values, 0.010))});
    log.out(" 0.1%%   KLD: %10.6f\n", .{D.d(percentile(kld_values, 0.001))});
    log.out("Minimum KLD: %10.6f\n", .{D.d(kld_values[0])});

    log.out("\n", .{});

    log.out("====== Token probability statistics ======\n", .{});

    const p_diff = meanAndUncertainty(kld.sum_p_diff, kld.sum_p_diff2, kld.count);
    log.out("Mean    Δp: %6.3lf ± %5.3lf %%\n", .{ 100.0 * p_diff[0], 100.0 * p_diff[1] });

    const pn = p_diff_values.len;
    const p_diff_median: f32 = if (pn % 2 == 0) 0.5 * (p_diff_values[pn / 2] + p_diff_values[pn / 2 - 1]) else p_diff_values[pn / 2];

    log.out("Maximum Δp: %6.3lf%%\n", .{100.0 * D.d(p_diff_values[pn - 1])});
    log.out("99.9%%   Δp: %6.3lf%%\n", .{100.0 * D.d(percentile(p_diff_values, 0.999))});
    log.out("99.0%%   Δp: %6.3lf%%\n", .{100.0 * D.d(percentile(p_diff_values, 0.990))});
    log.out("95.0%%   Δp: %6.3lf%%\n", .{100.0 * D.d(percentile(p_diff_values, 0.950))});
    log.out("90.0%%   Δp: %6.3lf%%\n", .{100.0 * D.d(percentile(p_diff_values, 0.900))});
    log.out("75.0%%   Δp: %6.3lf%%\n", .{100.0 * D.d(percentile(p_diff_values, 0.750))});
    log.out("Median  Δp: %6.3lf%%\n", .{100.0 * D.d(p_diff_median)});
    log.out("25.0%%   Δp: %6.3lf%%\n", .{100.0 * D.d(percentile(p_diff_values, 0.250))});
    log.out("10.0%%   Δp: %6.3lf%%\n", .{100.0 * D.d(percentile(p_diff_values, 0.100))});
    log.out(" 5.0%%   Δp: %6.3lf%%\n", .{100.0 * D.d(percentile(p_diff_values, 0.050))});
    log.out(" 1.0%%   Δp: %6.3lf%%\n", .{100.0 * D.d(percentile(p_diff_values, 0.010))});
    log.out(" 0.1%%   Δp: %6.3lf%%\n", .{100.0 * D.d(percentile(p_diff_values, 0.001))});
    log.out("Minimum Δp: %6.3lf%%\n", .{100.0 * D.d(p_diff_values[0])});

    const p_diff_mse = meanAndUncertainty(kld.sum_p_diff2, kld.sum_p_diff4, kld.count);

    const p_diff_rms_val = libm.sqrt(p_diff_mse[0]);
    const p_diff_rms_unc = 0.5 / p_diff_rms_val * p_diff_mse[1];
    log.out("RMS Δp    : %6.3lf ± %5.3lf %%\n", .{ 100.0 * p_diff_rms_val, 100.0 * p_diff_rms_unc });

    const same_top_p = 1.0 * @as(f64, @floatFromInt(kld.n_same_top)) / @as(f64, @floatFromInt(kld.count));
    log.out("Same top p: %6.3lf ± %5.3lf %%\n", .{ 100.0 * same_top_p, 100.0 * libm.sqrt(same_top_p * (1.0 - same_top_p) / @as(f64, @floatFromInt(kld.count - 1))) });
}

/// Ports `common_params_get_system_info` (common/common.cpp:421 @5266f24da).
fn systemInfo(a: std.mem.Allocator, params: *const Params) ![:0]u8 {
    var os: std.ArrayList(u8) = .empty;
    try os.print(a, "system_info: n_threads = {d}", .{params.cpuparams.n_threads});
    if (params.cpuparams_batch.n_threads != -1) try os.print(a, " (n_threads_batch = {d})", .{params.cpuparams_batch.n_threads});
    try os.print(a, " / {d} | {s}", .{ hardwareConcurrency(), c.llama_print_system_info() });
    return a.dupeSentinel(u8, os.items, 0);
}

/// Ports `llama_perplexity` (perplexity.cpp:2011 @5266f24da).
fn run(init: std.process.Init, a: std.mem.Allocator, argv: []const [:0]const u8) !u8 {
    const func = "llama_perplexity";
    var params = try Params.init(a);

    params.n_ctx = 512;
    params.escape = false;

    tc.common.init();

    if (!try arg.parse(argv, &params, .perplexity, null)) return 1;

    const n_ctx = params.n_ctx;
    if (n_ctx <= 0) {
        log.err("%s: perplexity tool requires '--ctx-size' > 0\n", .{func});
        return 1;
    }

    if (params.hellaswag or params.winogrande or params.multiple_choice) {
        params.n_parallel = @max(4, params.n_parallel);
        params.kv_unified = true;
    } else { // Perplexity & KL divergence
        params.n_parallel = @max(1, @divTrunc(params.n_batch, n_ctx));
    }
    params.n_ctx = params.n_parallel * n_ctx;
    params.n_batch = @min(params.n_batch, params.n_ctx);

    if (params.ppl_stride > 0) {
        log.inf("Will perform strided perplexity calculation -> adjusting context size from %d to %d\n", .{ params.n_ctx, params.n_ctx + @divTrunc(params.ppl_stride, 2) });
        params.n_ctx += @divTrunc(params.ppl_stride, 2);
    }

    c.llama_backend_init();
    c.llama_numa_init(params.numa);

    // load the model and apply lora adapter, if any
    var llama_init = try tc.init.initFromParams(init.gpa, &params, false);
    defer llama_init.deinit();

    const model = llama_init.model;
    const ctx = llama_init.context;

    if (model == null) {
        log.err("%s: unable to load model\n", .{func});
        return 1;
    }

    if (ctx == null) {
        log.err("%s: failed to create context\n", .{func});
        return 1;
    }

    const n_ctx_train = c.llama_model_n_ctx_train(model);

    if (params.n_ctx > n_ctx_train) {
        log.wrn("%s: model was trained on only %d context tokens (%d specified)\n", .{ func, n_ctx_train, params.n_ctx });
    }

    // print system information
    {
        log.inf("\n", .{});
        log.inf("%s\n", .{(try systemInfo(a, &params)).ptr});
    }

    if (params.hellaswag) {
        try hellaswagScore(a, ctx, &params);
    } else if (params.winogrande) {
        try winograndeScore(a, ctx, &params);
    } else if (params.multiple_choice) {
        try multipleChoiceScore(a, ctx, &params);
    } else if (params.kl_divergence) {
        try klDivergence(a, ctx, &params);
    } else {
        try perplexity(a, ctx, &params, n_ctx);
    }

    log.out("\n", .{});
    c.llama_perf_context_print(ctx);
    try fit.memoryBreakdownPrint(init.gpa, .{ .verbosity = std.math.maxInt(i32), .sink = tc.common.fitLogSink }, @ptrCast(ctx));

    c.llama_backend_free();

    return 0;
}

/// Ports `main` (tools/perplexity/main.cpp:3 @5266f24da).
pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return run(init, arena, argv);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
