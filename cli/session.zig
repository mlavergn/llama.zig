//! Loading a model and generating from it.
//!
//! # Provenance
//!
//! **Not a port.** Our own code, written against libllama's C ABI. It follows
//! the shape of `llama.cpp/tools/cli/cli-context.cpp` (v0.3.0, `c1d0e7a00`) --
//! load, tokenize, build a sampler chain, decode in a loop -- but shares no
//! code with it and is far smaller, because it does one-shot completion only.
//!
//! The sampler chain in particular mirrors `common_sampler_init` in
//! `llama.cpp/common/sampling.cpp`: **order matters**, and getting it wrong
//! changes what the model says without failing anything.
//!
//! # Lifetimes
//!
//! Everything libllama hands back is owned by libllama and freed through it, in
//! reverse order of acquisition. `deinit` does that, so a caller only has to
//! remember `defer session.deinit()`.

const std = @import("std");
const args_mod = @import("args.zig");

pub const c = @cImport({
    @cInclude("llama.h");
});

/// Failures that are worth telling apart from each other.
///
/// libllama signals almost everything as a null pointer or a non-zero int, so
/// these exist to name the stage that failed rather than surfacing one opaque
/// error for a bad path, an out-of-memory, and a decode fault alike.
pub const Error = error{
    ModelLoadFailed,
    NoVocab,
    ContextFailed,
    SamplerFailed,
    TokenizeFailed,
    DecodeFailed,
    PromptTooLong,
};

/// A loaded model and everything needed to generate from it.
pub const Session = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    model: *c.llama_model,
    vocab: *const c.llama_vocab,
    ctx: *c.llama_context,
    sampler: *c.llama_sampler,
    /// Borrowed from the caller's `Args`; not owned.
    settings: args_mod.Args,

    /// Loads a model and builds the context and sampler chain.
    ///
    /// Parameters:
    /// - `allocator`: allocator for token buffers this session makes.
    /// - `settings`: parsed command line. Borrowed, and must outlive the
    ///   session, since `model` points into it.
    ///
    /// Return: an initialized session the caller releases with `deinit`.
    /// Surfaces the stage that failed as a distinct error.
    pub fn init(allocator: std.mem.Allocator, settings: args_mod.Args) !Self {
        var model_params = c.llama_model_default_params();
        if (settings.n_gpu_layers >= 0) model_params.n_gpu_layers = settings.n_gpu_layers;

        const model_path = try allocator.dupeZ(u8, settings.model);
        defer allocator.free(model_path);

        const model = c.llama_model_load_from_file(model_path.ptr, model_params) orelse
            return Error.ModelLoadFailed;
        errdefer c.llama_model_free(model);

        const vocab = c.llama_model_get_vocab(model) orelse return Error.NoVocab;

        var ctx_params = c.llama_context_default_params();
        if (settings.n_ctx > 0) ctx_params.n_ctx = @intCast(settings.n_ctx);
        if (settings.n_batch > 0) ctx_params.n_batch = @intCast(settings.n_batch);
        if (settings.n_threads > 0) {
            ctx_params.n_threads = settings.n_threads;
            ctx_params.n_threads_batch = settings.n_threads;
        }
        // `llama_context_default_params` sets `no_perf = true`, so the
        // counters `timings()` reads are not collected unless this is turned
        // off. Upstream splits collection (`--perf`) from display
        // (`--show-timings`); we do not take `--perf`, so the one flag drives
        // both -- there is no reason to pay for counters we will not print.
        ctx_params.no_perf = !settings.show_timings;

        const ctx = c.llama_init_from_model(model, ctx_params) orelse return Error.ContextFailed;
        errdefer c.llama_free(ctx);

        const sampler = try buildSampler(settings, vocab);
        errdefer c.llama_sampler_free(sampler);

        if (settings.warmup) warmUp(model, ctx, vocab, settings.n_batch);

        return .{
            .allocator = allocator,
            .model = model,
            .vocab = vocab,
            .ctx = ctx,
            .sampler = sampler,
            .settings = settings,
        };
    }

    /// Releases the model, context and sampler.
    ///
    /// Parameters:
    /// - `self`: the session to tear down.
    ///
    /// Return: nothing. Frees in reverse order of acquisition, which libllama
    /// requires: the context holds a reference to the model.
    pub fn deinit(self: *Self) void {
        c.llama_sampler_free(self.sampler);
        c.llama_free(self.ctx);
        c.llama_model_free(self.model);
    }

    /// Tokenizes the prompt.
    ///
    /// Parameters:
    /// - `self`: the session.
    /// - `prompt`: text to tokenize.
    ///
    /// Return: a newly allocated token slice the caller frees.
    pub fn tokenize(self: *Self, prompt: []const u8) ![]c.llama_token {
        // Negative return means "too small, and this is how many I need", so
        // the first call sizes the buffer rather than guessing.
        const probe = c.llama_tokenize(self.vocab, prompt.ptr, @intCast(prompt.len), null, 0, true, true);
        const n_max: usize = @intCast(if (probe < 0) -probe else probe);

        const tokens = try self.allocator.alloc(c.llama_token, n_max);
        errdefer self.allocator.free(tokens);

        const n = c.llama_tokenize(
            self.vocab,
            prompt.ptr,
            @intCast(prompt.len),
            tokens.ptr,
            @intCast(tokens.len),
            true, // add_special: the BOS the model expects
            true, // parse_special: honour control tokens written in the prompt
        );
        if (n < 0) return Error.TokenizeFailed;

        return self.allocator.realloc(tokens, @intCast(n));
    }

    /// Generates a completion and streams it to `w`.
    ///
    /// Parameters:
    /// - `self`: the session.
    /// - `prompt`: the prompt, already escape-processed.
    /// - `w`: destination; flushed after every token so output appears as it
    ///   is produced rather than at exit.
    ///
    /// Return: the number of tokens generated. Stops at end-of-generation, at
    /// `n_predict`, or when the context is full.
    pub fn generate(self: *Self, prompt: []const u8, w: *std.Io.Writer) !usize {
        const tokens = try self.tokenize(prompt);
        defer self.allocator.free(tokens);

        const n_ctx = c.llama_n_ctx(self.ctx);
        if (tokens.len >= n_ctx) return Error.PromptTooLong;

        if (self.settings.display_prompt) {
            try w.print("{s}", .{prompt});
            try w.flush();
        }

        // `llama_batch_get_one` borrows rather than copies, and the batch built
        // at the end of one iteration is not read until the decode at the top
        // of the next -- so the sampled token must outlive the loop body.
        var id: c.llama_token = undefined;
        var batch = c.llama_batch_get_one(tokens.ptr, @intCast(tokens.len));

        var generated: usize = 0;
        var used: usize = tokens.len;
        while (self.settings.n_predict < 0 or generated < @as(usize, @intCast(self.settings.n_predict))) {
            if (c.llama_decode(self.ctx, batch) != 0) return Error.DecodeFailed;

            id = c.llama_sampler_sample(self.sampler, self.ctx, -1);
            if (!self.settings.ignore_eos and c.llama_vocab_is_eog(self.vocab, id)) break;

            var piece: [256]u8 = undefined;
            const n = c.llama_token_to_piece(self.vocab, id, &piece, piece.len, 0, true);
            if (n > 0) {
                try w.print("{s}", .{piece[0..@intCast(n)]});
                try w.flush();
            }

            generated += 1;
            used += 1;
            // Stopping here rather than letting llama_decode fail: running out
            // of context is a normal end to a generation, not an error.
            if (used >= n_ctx) break;

            batch = c.llama_batch_get_one(&id, 1);
        }

        try w.print("\n", .{});
        try w.flush();
        return generated;
    }

    /// Reads the context's performance counters.
    ///
    /// Parameters:
    /// - `self`: the session, after `generate` has run.
    ///
    /// Return: the two throughputs. A zero elapsed time yields zero rather than an
    /// infinity, which happens for real when a run is short enough that the timer
    /// does not tick.
    pub fn timings(self: *Self) Timings {
        const d = c.llama_perf_context(self.ctx);
        return .{
            .prompt_tokens = d.n_p_eval,
            // 1e3 / t_ms * n, as `llama_perf_context_print` computes it
            // (llama-context.cpp:4156 @c1d0e7a00).
            .prompt_per_second = if (d.t_p_eval_ms > 0)
                1e3 / d.t_p_eval_ms * @as(f64, @floatFromInt(d.n_p_eval))
            else
                0,
            .generated_tokens = d.n_eval,
            .generated_per_second = if (d.t_eval_ms > 0)
                1e3 / d.t_eval_ms * @as(f64, @floatFromInt(d.n_eval))
            else
                0,
        };
    }

    /// Writes the timings line.
    ///
    /// The format matches `cli-context.cpp:651` so a script that scrapes
    /// upstream's output also reads ours.
    ///
    /// **This goes to stdout**, as upstream's does. That is worth stating because
    /// it means a redirect captures it alongside the completion. `--no-show-timings`
    /// is how a caller gets clean output, and `scripts/parity-cli` passes exactly
    /// that.
    ///
    /// Parameters:
    /// - `self`: the session, after `generate` has run.
    /// - `w`: destination.
    ///
    /// Return: nothing; propagates write errors.
    pub fn reportTimings(self: *Self, w: *std.Io.Writer) !void {
        const t = self.timings();
        try w.print("\n[ Prompt: {d:.1} t/s | Generation: {d:.1} t/s ]\n", .{
            t.prompt_per_second,
            t.generated_per_second,
        });
        try w.flush();
    }
};

/// Prompt and generation throughput, in tokens per second.
///
/// Two separate numbers because they measure different work: the prompt is
/// evaluated in one batched pass over many tokens, generation one token at a
/// time. Prompt throughput is normally far higher, and averaging them together
/// would hide both.
pub const Timings = struct {
    /// Tokens in the prompt, and how fast they were evaluated.
    prompt_tokens: i32,
    prompt_per_second: f64,
    /// Tokens generated, and how fast.
    generated_tokens: i32,
    generated_per_second: f64,
};

/// Ports the warmup in `common_init_from_params` (common/common.cpp:1436 @c1d0e7a00).
///
/// Runs one throwaway decode so the one-time costs -- Metal pipeline
/// compilation, buffer allocation, first-touch page faults -- happen *before*
/// anything is measured.
///
/// **`llama_perf_context_reset` at the end is the point.** Without it the
/// warmup's own time lands in the counters `timings()` reads, and the reported
/// prompt throughput is worse than the truth rather than better. Warming up and
/// then not resetting would be worse than not warming up at all.
///
/// Measured effect, Qwen3.5-2B Q4_K_M on this machine, six runs each:
///
/// | Prompt | without warmup | with warmup |
/// | --- | --- | --- |
/// | 5 tokens | ~400 t/s, 1.8x spread | ~540 t/s, 1.6x spread |
/// | ~150 tokens | 3538-4498 t/s, first run an outlier | 4537-4703 t/s, 3.7% spread |
///
/// So warmup raises the mean and, more usefully, removes the first-run
/// outlier. It does **not** rescue a 5-token prompt: at that size the eval is
/// a few milliseconds and the rate stays too noisy to compare. That is a
/// property of the measurement, not something warmup can fix.
///
/// Parameters:
/// - `model`: the loaded model.
/// - `ctx`: its context.
/// - `vocab`: the model's vocabulary, for the BOS and EOS tokens.
/// - `n_batch`: caps the warmup batch, as upstream does.
///
/// Return: nothing. Decode failures are ignored: a warmup is an optimisation,
/// and a model that cannot decode this will fail loudly on the real prompt a
/// moment later with a better error.
fn warmUp(model: *c.llama_model, ctx: *c.llama_context, vocab: *const c.llama_vocab, n_batch: i32) void {
    var tmp: [2]c.llama_token = undefined;
    var n: usize = 0;

    // Some models (e.g. T5) have no BOS, and Qwen has no BOS either -- hence
    // the fallback to token 0 rather than assuming one exists.
    const bos = c.llama_vocab_bos(vocab);
    const eos = c.llama_vocab_eos(vocab);
    if (bos != c.LLAMA_TOKEN_NULL) {
        tmp[n] = bos;
        n += 1;
    }
    if (eos != c.LLAMA_TOKEN_NULL) {
        tmp[n] = eos;
        n += 1;
    }
    if (n == 0) {
        tmp[0] = 0;
        n = 1;
    }

    if (c.llama_model_has_encoder(model)) {
        _ = c.llama_encode(ctx, c.llama_batch_get_one(&tmp, @intCast(n)));
        var start = c.llama_model_decoder_start_token(model);
        if (start == c.LLAMA_TOKEN_NULL) start = bos;
        tmp[0] = start;
        n = 1;
    }
    if (c.llama_model_has_decoder(model)) {
        const count = @min(n, @as(usize, @intCast(@max(n_batch, 1))));
        _ = c.llama_decode(ctx, c.llama_batch_get_one(&tmp, @intCast(count)));
    }

    // Drop the warmup tokens from the KV cache, or they would prefix the real
    // prompt.
    c.llama_memory_clear(c.llama_get_memory(ctx), true);
    // Wait for the GPU before resetting, or the work still in flight lands in
    // the counters after the reset.
    c.llama_synchronize(ctx);
    c.llama_perf_context_reset(ctx);

    // Upstream also re-seeds its samplers here. Ours is untouched by the
    // warmup -- nothing above samples -- so its RNG is still at the seeded
    // state and there is nothing to restore.
}

/// Builds the sampler chain.
///
/// **Order is the behaviour.** This mirrors `common_sampler_init` in
/// `llama.cpp/common/sampling.cpp`, whose order comes from the default
/// `params.samplers` list in `common/common.h`: penalties, then the truncating
/// samplers, then temperature, then the distribution draw. Reordering these
/// silently changes what the model says, and nothing in our test suite would
/// catch it -- so it is written to match upstream rather than tidied.
///
/// Every sampler is added unconditionally, as upstream does. Each one no-ops
/// at its disabling value (`top_p` 1.0, `min_p` 0.0, `penalty_repeat` 1.0), so
/// skipping them would only risk diverging.
///
/// **There is deliberately no greedy short-circuit.** `--temp 0` looks like it
/// should mean "add `llama_sampler_init_greedy` and skip the rest", and that
/// is wrong: `llama_sampler_temp_impl` (llama-sampler.cpp:265 @c1d0e7a00) already does the
/// argmax when `temp <= 0`, and it does it *after* the penalty sampler has
/// altered the logits. Short-circuiting would make `--temp 0 --repeat-penalty
/// 1.1` pick a different token from upstream, on a path where both look
/// plausible.
///
/// Parameters:
/// - `settings`: parsed command line.
/// - `vocab`: the model's vocabulary; the penalty sampler needs its size.
///
/// Return: the chain, owned by the caller.
fn buildSampler(settings: args_mod.Args, vocab: *const c.llama_vocab) !*c.llama_sampler {
    var params = c.llama_sampler_chain_default_params();
    // Upstream measures its own sampler timings; we have no use for them.
    params.no_perf = true;

    const chain = c.llama_sampler_chain_init(params) orelse return Error.SamplerFailed;
    errdefer c.llama_sampler_free(chain);

    // `min_keep` is 0 upstream, not 1. It is the floor on how many candidates a
    // truncating sampler must leave; 0 means "no floor", and passing 1 would
    // quietly change what top-p and min-p do at their edges.
    const min_keep: usize = 0;

    c.llama_sampler_chain_add(chain, c.llama_sampler_init_penalties(
        c.llama_vocab_n_tokens(vocab),
        settings.penalty_last_n,
        settings.penalty_repeat,
        0.0, // penalty_freq: --frequency-penalty is not exposed yet
        0.0, // penalty_present: --presence-penalty is not exposed yet
    ));
    c.llama_sampler_chain_add(chain, c.llama_sampler_init_top_k(settings.top_k));
    c.llama_sampler_chain_add(chain, c.llama_sampler_init_top_p(settings.top_p, min_keep));
    c.llama_sampler_chain_add(chain, c.llama_sampler_init_min_p(settings.min_p, min_keep));
    // temp_ext, not temp: upstream uses the extended form with dynamic
    // temperature disabled (range 0.0, exponent 1.0), and the two are only
    // equivalent while those defaults hold.
    c.llama_sampler_chain_add(chain, c.llama_sampler_init_temp_ext(settings.temp, 0.0, 1.0));
    c.llama_sampler_chain_add(chain, c.llama_sampler_init_dist(settings.seed));

    return chain;
}
