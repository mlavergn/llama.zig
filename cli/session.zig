//! Loading a model and generating from it.
//!
//! # Provenance
//!
//! **Not a port.** Our own code, written against libllama's C ABI. It follows
//! the shape of `llama.cpp/tools/cli/cli-context.cpp` (v0.9.0, `cd959bae9`) --
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
const chat = @import("chat.zig");
const fit = @import("cli_fit");

/// libllama's C ABI. Re-exported so callers can reach it as
/// `cli.session.c`; the import itself lives in `c.zig`, because two
/// `@cImport`s of one header produce two incompatible sets of types.
pub const c = @import("c.zig").api;

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
    /// The grammar did not parse: upstream's "failed to parse grammar".
    GrammarFailed,
    TokenizeFailed,
    DecodeFailed,
    PromptTooLong,
};

/// Why the last generation ended.
///
/// Only `eog` is the model finishing. The other two cut a reply off, and a
/// conversation has to say so: otherwise the prompt just comes back mid-answer
/// and the cut reads as the model's own.
pub const Stop = enum {
    /// The model emitted an end-of-generation token.
    eog,
    /// `-n` tokens were generated.
    n_predict,
    /// The context filled up.
    context_full,
    /// A stop string the chat template's parser asked for appeared in the
    /// reply (`common_chat_params::additional_stops`): a natural end, as
    /// upstream's `STOP_TYPE_WORD` is.
    word,
    /// The reply's consumer refused it -- a chat reply that does not parse.
    rejected,
};

/// What a chat turn generates with, beyond the session's own sampler: the
/// parts of upstream's server task that `common_chat_templates_apply` fills
/// in for a chat request (`oaicompat_chat_params_parse`,
/// tools/server/server-common.cpp:1348 @cd959bae9). Everything is borrowed
/// for the call.
pub const TurnOptions = struct {
    /// Tokens printed with their special spelling; every other control token
    /// prints as nothing, as the server's `accept_special_token` decides
    /// (tools/server/server-context.cpp:4813 @cd959bae9).
    preserved: []const c.llama_token = &.{},
    /// Stop strings: the reply ends where one appears, and it is not sent.
    stops: []const []const u8 = &.{},
    /// The reasoning budget sampler, applied before the chain and accepting
    /// every generated token first, as `common_sampler` holds its `rbudget`.
    /// Owned by the caller.
    rbudget: ?*c.llama_sampler = null,
};

/// Where a chat turn's reply goes, a piece at a time: text the server would
/// have sent, after its stop-string and UTF-8 hold-back.
pub const Sink = struct {
    ctx: *anyopaque,
    /// Takes the text sent since the last call; false stops generating.
    text: *const fn (ctx: *anyopaque, text: []const u8) anyerror!bool,
};

/// A loaded model and everything needed to generate from it.
pub const Session = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    model: *c.llama_model,
    vocab: *const c.llama_vocab,
    ctx: *c.llama_context,
    sampler: *c.llama_sampler,
    /// The grammar sampler, kept apart from the chain as upstream's
    /// `common_sampler` keeps its `grmr`: null without `--grammar` or
    /// `--json-schema`. See `sampleNext`.
    grammar: ?*c.llama_sampler = null,
    /// The candidate array `sampleNext` builds from the logits when a
    /// grammar is set: `common_sampler`'s `cur`.
    candidates: std.ArrayList(c.llama_token_data) = .empty,
    /// The CPU backend's persistent threadpool, attached to `ctx`.
    ///
    /// Without one, every `ggml_graph_compute` builds a disposable pool --
    /// `n_threads - 1` `pthread_create`s and joins -- and a fully offloaded
    /// model still runs a CPU split every token (the token-embedding
    /// `GET_ROWS`, whose weights stay in host memory). Measured on a GB10 with
    /// Qwen3.5-2B: decode 7.59 ms/token without a pool, 7.03 with one, 6.75 at
    /// one thread. Upstream's CLI attaches one in `common_threadpools`'s `init`
    /// (common/common.cpp:1819 @cd959bae9). Null if creation failed, which leaves
    /// libllama's per-call pools, as upstream does.
    threadpool: ?*c.struct_ggml_threadpool = null,
    /// Borrowed from the caller's `Args`; not owned.
    settings: args_mod.Args,
    /// Tokens currently held in the KV cache, in order.
    ///
    /// `generateTurn` diffs the next turn's prompt against this to find how
    /// much of the cache it can keep, so a conversation does not re-decode its
    /// whole history every turn.
    decoded: std.ArrayList(c.llama_token) = .empty,
    /// Why the most recent `generate` or `generateTurn` ended.
    last_stop: Stop = .eog,
    /// Where `generateTurn` reports how much of the cache each turn kept;
    /// null, the default, reports nothing.
    ///
    /// Set from `LLAMAZIG_CNV_TRACE` by `main.zig`. Prefix reuse rarely
    /// shows in what a greedy decode says -- measured, a turn decoded from
    /// scratch picks the same tokens in four of `scripts/cnv-diff`'s six
    /// Qwen3-0.6B scenarios -- so this line is how that gate tells a
    /// conversation that keeps its prefix from one that silently re-decodes
    /// everything.
    turn_trace: ?*std.Io.Writer = null,
    /// The `n_gpu_layers` the model was loaded with, after `--fit`: what
    /// `main.zig` reports under `LLAMAZIG_CNV_TRACE` beside the context size,
    /// so `scripts/cnv-diff` sees what the fit decided reach libllama.
    gpu_layers: i32,
    /// What `llama_model_params` points into while the model loads, kept for
    /// the model's lifetime since libllama keeps the parameters. Heap
    /// allocated: the session is returned by value.
    load: *LoadBuffers,

    /// The buffers upstream's `common_params` lends `llama_model_params`,
    /// which `--fit` writes a layer assignment into.
    const LoadBuffers = struct {
        /// `common_params::tensor_split`, 128 zeros until `--fit` writes it.
        tensor_split: [128]f32 = @splat(0),
        /// `common_params::tensor_buft_overrides`, padded with empty entries
        /// to `llama_max_tensor_buft_overrides` (common/arg.cpp:931
        /// @cd959bae9).
        tensor_buft_overrides: []c.llama_model_tensor_buft_override,
        /// The regexes the overrides point at.
        patterns: fit.Patterns,
    };

    /// Loads a model and builds the context and sampler chain, first
    /// fitting the parameters left unset to free memory unless `--fit off`,
    /// as upstream's `common_init_result` does.
    ///
    /// Parameters:
    /// - `allocator`: allocator for token buffers this session makes.
    /// - `settings`: parsed command line. Borrowed, and must outlive the
    ///   session, since `model` points into it.
    /// - `err`: where `--fit` reports an error, as upstream's log does at
    ///   `llama-cli`'s verbosity; null drops it.
    ///
    /// Return: an initialized session the caller releases with `deinit`.
    /// Surfaces the stage that failed as a distinct error.
    pub fn init(allocator: std.mem.Allocator, settings: args_mod.Args, err: ?*std.Io.Writer) !Self {
        const load = try allocator.create(LoadBuffers);
        errdefer allocator.destroy(load);
        load.* = .{
            .tensor_buft_overrides = try allocator.alloc(c.llama_model_tensor_buft_override, c.llama_max_tensor_buft_overrides()),
            .patterns = try .init(allocator),
        };
        errdefer allocator.free(load.tensor_buft_overrides);
        errdefer load.patterns.deinit();
        @memset(load.tensor_buft_overrides, .{ .pattern = null, .buft = null });

        // `common_model_params_to_llama` (common/common.cpp:1711 @cd959bae9).
        // `load_mode` and v0.4.0's `lazy_mode` stay libllama's defaults,
        // `AUTO` both, which are `common_params`' too: `--load-mode` and
        // `--lazy-mode` are not among the flags this CLI takes.
        var model_params = c.llama_model_default_params();
        model_params.n_gpu_layers = settings.n_gpu_layers;
        model_params.tensor_split = &load.tensor_split;
        model_params.tensor_buft_overrides = load.tensor_buft_overrides.ptr;

        const model_path = try allocator.dupeSentinel(u8, settings.model, 0);
        defer allocator.free(model_path);

        var ctx_params = c.llama_context_default_params();
        // 0 is passed through, not skipped: libllama reads it as "the
        // context the model was trained with", which is what `-c 0` means
        // upstream and what `Args` documents. Skipping it left libllama's own
        // default of 512 tokens, and a conversation's history and reply had
        // to fit in that -- replies were cut off mid-answer. `--fit` below
        // then shrinks it where that would not fit in memory.
        if (settings.n_ctx >= 0) ctx_params.n_ctx = @intCast(settings.n_ctx);
        if (settings.n_batch > 0) ctx_params.n_batch = @intCast(settings.n_batch);
        // `server_output_limits` (tools/server/server-context.cpp:48
        // @cd959bae9), which upstream's `llama-cli` reaches through the
        // server it drives: one sequence and no draft model, so one output
        // in a batch and one per sequence -- all this binary ever reads.
        // libllama's default of 0 sizes the output buffer for the whole
        // batch: 1.2 GB of logits for Qwen3's 151,936-token vocabulary at
        // `-b 2048`, which `--fit` then measured and shrank the context for
        // -- measured on the UNO Q, 15,616 tokens against upstream's 18,176.
        ctx_params.n_outputs_max = 1;
        ctx_params.n_outputs_max_per_seq = 1;
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

        if (settings.fit) {
            // The status is not consulted, as upstream does not: a failed fit
            // leaves the parameters as far as it got, and loading goes ahead.
            _ = try fit.fitParams(
                allocator,
                &load.patterns,
                .{ .verbosity = fit.Log.level_error, .out = err },
                model_path.ptr,
                @ptrCast(&model_params),
                @ptrCast(&ctx_params),
                &load.tensor_split,
                @ptrCast(load.tensor_buft_overrides.ptr),
                &settings.fit_target,
                settings.fit_min_ctx,
                null,
                c.GGML_LOG_LEVEL_ERROR,
            );
        }

        const model = c.llama_model_load_from_file(model_path.ptr, model_params) orelse
            return Error.ModelLoadFailed;
        errdefer c.llama_model_free(model);

        const vocab = c.llama_model_get_vocab(model) orelse return Error.NoVocab;

        const ctx = c.llama_init_from_model(model, ctx_params) orelse return Error.ContextFailed;
        errdefer c.llama_free(ctx);

        // One pool serves prompt and generation alike, because `n_threads`
        // and `n_threads_batch` are always equal here; upstream makes a
        // second only when the two differ. `ggml_threadpool_params_init`
        // supplies upstream's defaults, polling level 50 included.
        var tpp: c.struct_ggml_threadpool_params = undefined;
        c.ggml_threadpool_params_init(&tpp, c.llama_n_threads(ctx));
        const threadpool = c.ggml_threadpool_new(&tpp);
        if (threadpool) |tp| c.llama_attach_threadpool(ctx, tp, null);
        errdefer if (threadpool) |tp| c.ggml_threadpool_free(tp);

        const sampler = try buildSampler(settings, vocab);
        errdefer c.llama_sampler_free(sampler);

        // `common_sampler_init` (common/sampling.cpp:191 @cd959bae9): a grammar
        // that does not parse fails the sampler, and with it the run. The
        // output-format prefill that follows there needs the chat template's
        // generation prompt, which a raw completion does not have.
        var grammar: ?*c.llama_sampler = null;
        if (settings.grammar.len != 0) {
            const text = try allocator.dupeSentinel(u8, settings.grammar, 0);
            defer allocator.free(text);
            grammar = c.llama_sampler_init_grammar(vocab, text.ptr, "root") orelse return Error.GrammarFailed;
        }
        errdefer if (grammar) |g| c.llama_sampler_free(g);

        if (settings.warmup) warmUp(model, ctx, vocab, settings.n_batch);

        return .{
            .allocator = allocator,
            .model = model,
            .vocab = vocab,
            .ctx = ctx,
            .sampler = sampler,
            .grammar = grammar,
            .threadpool = threadpool,
            .settings = settings,
            .gpu_layers = model_params.n_gpu_layers,
            .load = load,
        };
    }

    /// Releases the model, context, threadpool and sampler.
    ///
    /// Parameters:
    /// - `self`: the session to tear down.
    ///
    /// Return: nothing. Frees in reverse order of acquisition, which libllama
    /// requires: the context holds a reference to the model.
    pub fn deinit(self: *Self) void {
        self.decoded.deinit(self.allocator);
        self.candidates.deinit(self.allocator);
        if (self.grammar) |g| c.llama_sampler_free(g);
        c.llama_sampler_free(self.sampler);
        c.llama_free(self.ctx);
        // After the context: it holds the pool until it is freed.
        if (self.threadpool) |tp| c.ggml_threadpool_free(tp);
        c.llama_model_free(self.model);
        self.load.patterns.deinit();
        self.allocator.free(self.load.tensor_buft_overrides);
        self.allocator.destroy(self.load);
    }

    /// Tokenizes the prompt.
    ///
    /// Parameters:
    /// - `self`: the session.
    /// - `prompt`: text to tokenize.
    ///
    /// Return: a newly allocated token slice the caller frees.
    pub fn tokenize(self: *Self, prompt: []const u8) ![]c.llama_token {
        return self.tokenizeText(prompt, .{});
    }

    /// How a run of text should be tokenized.
    pub const TokenizeOptions = struct {
        /// Whether libllama prepends the model's BOS.
        ///
        /// False for chat prompts: the template already writes whatever
        /// opening the model expects, and a second BOS is a different prompt.
        add_special: bool = true,
        /// Whether control-token spellings in the text become control tokens.
        ///
        /// False for anything derived from message content. See the header of
        /// `chat.zig`: this is the flag the whole trust scheme exists to set
        /// correctly.
        parse_special: bool = true,
    };

    /// Tokenizes one run of text.
    ///
    /// Parameters:
    /// - `self`: the session.
    /// - `text`: the text to tokenize.
    /// - `opts`: how to treat specials.
    ///
    /// Return: the tokens, owned by the caller.
    pub fn tokenizeText(
        self: *Self,
        text: []const u8,
        opts: TokenizeOptions,
    ) ![]c.llama_token {
        // Negative return means "too small, and this is how many I need", so
        // the first call sizes the buffer rather than guessing.
        const probe = c.llama_tokenize(
            self.vocab,
            text.ptr,
            @intCast(text.len),
            null,
            0,
            opts.add_special,
            opts.parse_special,
        );
        const n_max: usize = @intCast(if (probe < 0) -probe else probe);

        const tokens = try self.allocator.alloc(c.llama_token, n_max);
        errdefer self.allocator.free(tokens);

        const n = c.llama_tokenize(
            self.vocab,
            text.ptr,
            @intCast(text.len),
            tokens.ptr,
            @intCast(tokens.len),
            opts.add_special,
            opts.parse_special,
        );
        if (n < 0) return Error.TokenizeFailed;

        return self.allocator.realloc(tokens, @intCast(n));
    }

    /// Tokenizes a rendered chat prompt, run by run.
    ///
    /// **This is where the trust marking is spent.** Each segment is
    /// tokenized on its own so that control tokens are honoured only in the
    /// runs the template itself wrote; a `<|im_start|>` that arrived inside a
    /// user message tokenizes as its literal characters instead.
    ///
    /// `add_special` is false throughout -- the template writes the model's
    /// opening itself, and only the first segment could have taken a BOS
    /// anyway.
    ///
    /// Parameters:
    /// - `self`: the session.
    /// - `segments`: the rendered prompt, from `chat.Template.applySegments`.
    ///
    /// Return: the tokens, owned by the caller.
    pub fn tokenizeSegments(self: *Self, segments: []const chat.Segment) ![]c.llama_token {
        var out: std.ArrayList(c.llama_token) = .empty;
        errdefer out.deinit(self.allocator);

        for (segments) |seg| {
            if (seg.text.len == 0) continue;
            const part = try self.tokenizeText(seg.text, .{
                .add_special = false,
                .parse_special = seg.trusted,
            });
            defer self.allocator.free(part);
            try out.appendSlice(self.allocator, part);
        }

        return out.toOwnedSlice(self.allocator);
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
    /// `n_predict`, or when the context is full; `last_stop` says which.
    pub fn generate(self: *Self, prompt: []const u8, w: *std.Io.Writer) !usize {
        const tokens = try self.tokenize(prompt);
        defer self.allocator.free(tokens);
        return self.generateTokens(tokens, prompt, w);
    }

    /// Samples the next token from the last decode's logits and accepts it.
    ///
    /// Without a grammar or a reasoning budget this is `llama_sampler_sample`
    /// over the chain. With either it is upstream's `common_sampler_sample`
    /// with `grammar_first` false, as its callers pass it
    /// (common/sampling.cpp:606 @cd959bae9), then `common_sampler_accept`
    /// (common/sampling.cpp:476 @cd959bae9): the budget, when there is one,
    /// edits the logits first; the chain picks a token; if the grammar
    /// rejects it, the logits are reloaded and the budget and grammar applied
    /// before the chain picks again -- which draws from the RNG a second time
    /// -- and the token is accepted by the budget, the grammar and the chain,
    /// in that order.
    ///
    /// Parameters:
    /// - `rbudget`: the turn's reasoning budget, or null.
    ///
    /// Return: the token.
    fn sampleNext(self: *Self, rbudget: ?*c.llama_sampler) !c.llama_token {
        const grmr = self.grammar;
        if (grmr == null and rbudget == null) return c.llama_sampler_sample(self.sampler, self.ctx, -1);

        try self.setLogits();
        var cur_p = self.candidateArray();
        if (rbudget) |rb| c.llama_sampler_apply(rb, &cur_p);
        c.llama_sampler_apply(self.sampler, &cur_p);
        var id = cur_p.data[@intCast(cur_p.selected)].id;

        if (grmr) |g| {
            // grammar-based rejection sampling
            var single: c.llama_token_data = .{ .id = id, .logit = 1.0, .p = 0.0 };
            var single_arr: c.llama_token_data_array = .{ .data = &single, .size = 1, .selected = -1, .sorted = false };
            c.llama_sampler_apply(g, &single_arr);
            if (single_arr.data[0].logit == -std.math.inf(f32)) {
                try self.setLogits();
                cur_p = self.candidateArray();
                if (rbudget) |rb| c.llama_sampler_apply(rb, &cur_p);
                c.llama_sampler_apply(g, &cur_p);
                c.llama_sampler_apply(self.sampler, &cur_p);
                if (cur_p.selected == -1) return Error.SamplerFailed;
                id = cur_p.data[@intCast(cur_p.selected)].id;
            }
        }

        if (rbudget) |rb| c.llama_sampler_accept(rb, id);
        if (grmr) |g| c.llama_sampler_accept(g, id);
        c.llama_sampler_accept(self.sampler, id);
        return id;
    }

    /// `common_sampler`'s `set_logits` (common/sampling.cpp:134 @cd959bae9), on
    /// its CPU path: one candidate per vocabulary entry, from the last row of
    /// logits.
    fn setLogits(self: *Self) !void {
        c.llama_synchronize(self.ctx);
        const logits = c.llama_get_logits_ith(self.ctx, -1) orelse return Error.DecodeFailed;
        const n_vocab: usize = @intCast(c.llama_vocab_n_tokens(self.vocab));
        try self.candidates.resize(self.allocator, n_vocab);
        for (self.candidates.items, 0..) |*t, i| {
            t.* = .{ .id = @intCast(i), .logit = logits[i], .p = 0.0 };
        }
    }

    fn candidateArray(self: *Self) c.llama_token_data_array {
        return .{ .data = self.candidates.items.ptr, .size = self.candidates.items.len, .selected = -1, .sorted = false };
    }

    /// Starts a new turn's grammar over: upstream's CLI sends each turn to its
    /// server as a new task, whose sampler -- grammar included -- is built
    /// afresh. The chain is left as it is, as it always has been here.
    pub fn resetGrammar(self: *Self) void {
        if (self.grammar) |g| c.llama_sampler_reset(g);
    }

    /// The decode loop, shared by the raw and chat paths.
    ///
    /// Parameters:
    /// - `self`: the session.
    /// - `prompt_tokens`: the prompt, already tokenized.
    /// - `display`: text to echo first, when `--no-display-prompt` was not
    ///   given.
    /// - `w`: destination; flushed after every token.
    ///
    /// Return: the number of tokens generated.
    pub fn generateTokens(
        self: *Self,
        prompt_tokens: []c.llama_token,
        display: []const u8,
        w: *std.Io.Writer,
    ) !usize {
        const tokens = prompt_tokens;
        const prompt = display;

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
        self.last_stop = .n_predict;
        while (self.settings.n_predict < 0 or generated < @as(usize, @intCast(self.settings.n_predict))) {
            if (c.llama_decode(self.ctx, batch) != 0) return Error.DecodeFailed;

            id = try self.sampleNext(null);
            if (!self.settings.ignore_eos and c.llama_vocab_is_eog(self.vocab, id)) {
                self.last_stop = .eog;
                break;
            }

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
            if (used >= n_ctx) {
                self.last_stop = .context_full;
                break;
            }

            batch = c.llama_batch_get_one(&id, 1);
        }

        try w.print("\n", .{});
        try w.flush();
        return generated;
    }

    /// Generates one assistant turn, keeping whatever KV prefix still applies.
    ///
    /// The next turn's prompt is the whole conversation re-rendered, so most
    /// of it is already decoded. Rather than assume the new render extends the
    /// old one -- which depends on what the template does -- this compares the
    /// two token sequences and keeps their common prefix. A template that
    /// rewrites earlier turns simply gets a shorter prefix and more work, not
    /// a wrong answer.
    ///
    /// The reply goes to `sink` as upstream's server sends it
    /// (`process_token`, tools/server/server-context.cpp:2137 @cd959bae9): each
    /// token's piece, special only for a preserved token, joins the text; while
    /// the text ends inside a UTF-8 sequence nothing is sent; a stop string
    /// ends the reply where it begins and is not sent; and text that could be
    /// the start of one is held back until it is not. What is held back when
    /// the reply ends is never sent, as a streamed reply's final chunk carries
    /// no text.
    ///
    /// Parameters:
    /// - `self`: the session.
    /// - `segments`: the whole conversation, rendered and trust-marked.
    /// - `opts`: the chat request's stops, preserved tokens and budget.
    /// - `sink`: receives the reply.
    ///
    /// Return: the number of tokens generated; `last_stop` says why it ended.
    pub fn generateTurn(
        self: *Self,
        segments: []const chat.Segment,
        opts: TurnOptions,
        sink: Sink,
    ) !usize {
        const tokens = try self.tokenizeSegments(segments);
        defer self.allocator.free(tokens);
        self.resetGrammar();

        const n_ctx = c.llama_n_ctx(self.ctx);
        if (tokens.len >= n_ctx) return Error.PromptTooLong;

        var keep: usize = 0;
        while (keep < tokens.len and keep < self.decoded.items.len and
            tokens[keep] == self.decoded.items[keep]) : (keep += 1)
        {}

        // An empty suffix would decode nothing and leave no logits to sample.
        if (keep == tokens.len and keep > 0) keep -= 1;

        if (keep < self.decoded.items.len) {
            const mem = c.llama_get_memory(self.ctx);
            // Recurrent and hybrid models cannot drop a range from the middle;
            // `seq_rm` says so rather than failing loudly, so a discarded
            // result means decoding the next turn on top of a stale cache.
            if (keep == 0 or !c.llama_memory_seq_rm(mem, 0, @intCast(keep), -1)) {
                c.llama_memory_clear(mem, true);
                keep = 0;
            }
            self.decoded.shrinkRetainingCapacity(keep);
        }

        const suffix = tokens[self.decoded.items.len..];
        if (self.turn_trace) |t| {
            try t.print("[turn: kept {d}, decoding {d}]\n", .{ self.decoded.items.len, suffix.len });
            try t.flush();
        }
        try self.decoded.appendSlice(self.allocator, suffix);

        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(self.allocator);
        var n_sent: usize = 0;

        var id: c.llama_token = undefined;
        var batch = c.llama_batch_get_one(@constCast(suffix.ptr), @intCast(suffix.len));

        var generated: usize = 0;
        self.last_stop = .n_predict;
        while (self.settings.n_predict < 0 or generated < @as(usize, @intCast(self.settings.n_predict))) {
            if (c.llama_decode(self.ctx, batch) != 0) return Error.DecodeFailed;

            id = try self.sampleNext(opts.rbudget);
            if (!self.settings.ignore_eos and c.llama_vocab_is_eog(self.vocab, id)) {
                self.last_stop = .eog;
                break;
            }
            generated += 1;

            const special = std.mem.findScalar(c.llama_token, opts.preserved, id) != null;
            var piece: [256]u8 = undefined;
            const n = c.llama_token_to_piece(self.vocab, id, &piece, piece.len, 0, special);
            const token_str = piece[0..@intCast(@max(n, 0))];
            try text.appendSlice(self.allocator, token_str);

            var stopped = false;
            // Nothing is sent while the text ends inside a character.
            if (validUtf8Len(text.items) == text.items.len) {
                const pos = @min(n_sent, text.items.len);
                const str_test = text.items[pos..];
                var send = true;
                if (findStop(opts.stops, str_test, token_str.len, true)) |stop_pos| {
                    text.shrinkRetainingCapacity(pos + stop_pos);
                    stopped = true;
                } else if (findStop(opts.stops, str_test, token_str.len, false) != null) {
                    send = false;
                }
                if (send and text.items.len > n_sent) {
                    const fresh = text.items[n_sent..];
                    n_sent = text.items.len;
                    if (!try sink.text(sink.ctx, fresh)) {
                        self.last_stop = .rejected;
                        break;
                    }
                }
            }
            if (stopped) {
                self.last_stop = .word;
                break;
            }

            // The last token of a reply cut short is never decoded, so it is
            // not part of the cache the next turn compares against.
            if (self.settings.n_predict >= 0 and generated >= @as(usize, @intCast(self.settings.n_predict))) break;
            // Stopping here rather than letting llama_decode fail: running out
            // of context is a normal end to a generation, not an error.
            if (self.decoded.items.len + 1 >= n_ctx) {
                self.last_stop = .context_full;
                break;
            }
            try self.decoded.append(self.allocator, id);
            batch = c.llama_batch_get_one(&id, 1);
        }
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
            // (llama-context.cpp:4927 @cd959bae9).
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

/// Ports `validate_utf8` (tools/server/server-common.cpp:1027 @cd959bae9): the
/// length of `text` without a multi-byte character cut off at its end.
pub fn validUtf8Len(text: []const u8) usize {
    const len = text.len;
    if (len == 0) return 0;
    var i: usize = 1;
    while (i <= 4 and i <= len) : (i += 1) {
        const ch = text[len - i];
        if ((ch & 0xE0) == 0xC0) {
            if (i < 2) return len - i;
        } else if ((ch & 0xF0) == 0xE0) {
            if (i < 3) return len - i;
        } else if ((ch & 0xF8) == 0xF0) {
            if (i < 4) return len - i;
        }
    }
    return len;
}

/// Ports `server_slot`'s `find_stopping_strings` (tools/server/server-context.cpp:685 @cd959bae9): where the first stop string begins in `text`. A full stop
/// looks only at the end the last token could have completed; otherwise this
/// finds a stop string's beginning at the very end of `text`
/// (`string_find_partial_stop`, common/common.h:880 @cd959bae9).
///
/// Return: the offset, or null.
pub fn findStop(stops: []const []const u8, text: []const u8, last_token_size: usize, is_full_stop: bool) ?usize {
    var stop_pos: ?usize = null;
    for (stops) |word| {
        const pos: ?usize = if (is_full_stop) blk: {
            const tmp = word.len + last_token_size;
            const from_pos = if (text.len > tmp) text.len - tmp else 0;
            break :blk std.mem.findPos(u8, text, from_pos, word);
        } else partialStop(text, word);
        if (pos) |p| {
            if (stop_pos == null or p < stop_pos.?) stop_pos = p;
        }
    }
    return stop_pos;
}

/// Ports `string_find_partial_stop` (common/common.h:880 @cd959bae9).
fn partialStop(str: []const u8, stop: []const u8) ?usize {
    if (str.len == 0 or stop.len == 0) return null;
    const last_char = str[str.len - 1];
    var len = @min(str.len, stop.len);
    while (len > 0) : (len -= 1) {
        if (stop[len - 1] == last_char and std.mem.endsWith(u8, str, stop[0..len])) return str.len - len;
    }
    return null;
}

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

/// Ports the warmup in `common_init_from_params` (common/common.cpp:1457 @cd959bae9).
///
/// Upstream decodes through `llama_process` and gives every warmup token a
/// zero state embedding (`common_batch_set_zero_state`, which a context
/// without a state input ignores); this decodes the classic batch, which
/// carries none. A context with a state input is a drafter's, and
/// `llama-cli` never warms one up.
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
/// is wrong: `llama_sampler_temp_impl` (llama-sampler.cpp:266 @cd959bae9) already does the
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
    // `common_sampler_init` (common/sampling.cpp:191 @cd959bae9): greedy
    // selection for a temperature-zero chain -- the list ends in the
    // temperature, dynamic temperature is off and no probabilities are
    // asked for -- and a draw from the distribution otherwise.
    const greedy = settings.temp == 0.0;
    c.llama_sampler_chain_add(chain, if (greedy) c.llama_sampler_init_greedy() else c.llama_sampler_init_dist(settings.seed));

    return chain;
}
