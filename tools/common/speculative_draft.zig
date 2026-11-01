//! The speculative decoders that run a second model: a plain draft model,
//! EAGLE3, DFlash and DSpark, and multi-token prediction.
//!
//! # Provenance
//!
//! Ports `common_speculative_impl_draft_simple`,
//! `common_speculative_impl_draft_eagle3`,
//! `common_speculative_impl_draft_dflash` and
//! `common_speculative_impl_draft_mtp` from
//! `llama.cpp/common/speculative.cpp` (v0.4.1, `b29c606e2`). Each
//! declaration names what it ports and the line it began at; the rest of the
//! file is `speculative.zig`.
//!
//! # Batches
//!
//! EAGLE3 and MTP feed their decoder tokens *and* embeddings in one batch;
//! `llama_batch_init` allocates only one of the two, so the token array is a
//! separate `malloc`, freed before `llama_batch_free`, as upstream does.

const std = @import("std");
const c = @import("tools_h");
const S = @import("speculative.zig");
const params_mod = @import("params.zig");
const sampling = @import("sampling.zig");
const common = @import("common.zig");
const log = @import("log.zig");
const arg = @import("arg.zig");

const ext = S.ext;
const Base = S.Base;
const DraftParams = S.DraftParams;
const spcDbg = S.spcDbg;
const spcTrc = S.spcTrc;
const spcWrn = S.spcWrn;
const spcErr = S.spcErr;

const gpa = std.heap.c_allocator;

fn assertC(ok: bool, comptime line: c_int, comptime expr: [*:0]const u8) void {
    if (!ok) _ = c.ggml_abort("speculative.cpp", line, "GGML_ASSERT(%s) failed", expr);
}

fn oom() noreturn {
    _ = c.ggml_abort("speculative.cpp", 0, "out of memory");
    unreachable;
}

/// The draft samplers' parameters: top-k 10 and nothing else, with
/// performance metrics on.
fn draftSampler(model: ?*const c.llama_model) *sampling.Sampler {
    var sparams: params_mod.Sampling = .{};
    sparams.no_perf = false;
    sparams.top_k = 10;
    var types = [_]params_mod.SamplerType{.top_k};
    // `sampling.init` reads the list and keeps nothing of it
    sparams.samplers = .fromOwnedSlice(&types);
    return sampling.init(model, &sparams) catch |e| switch (e) {
        error.OutOfMemory => oom(),
        // top-k over default penalties cannot throw
        else => unreachable,
    };
}

/// `common_token_to_piece(ctx, id)`, as a NUL-terminated string the caller
/// frees with `freePiece`.
fn piece(ctx: ?*c.llama_context, id: c.llama_token) [:0]u8 {
    return common.tokenToPiece(gpa, c.llama_model_get_vocab(c.llama_get_model(ctx)), id, true) catch oom();
}

fn freePiece(p: [:0]u8) void {
    gpa.free(p[0 .. p.len + 1]);
}

/// The three best candidates, as every drafter logs them.
fn logCandidates(comptime func: [*:0]const u8, ctx: ?*c.llama_context, seq_id: c.llama_seq_id, pos: c_int, cur_p: *const c.llama_token_data_array) void {
    var k: usize = 0;
    while (k < @min(3, cur_p.size)) : (k += 1) {
        if (log.verbosity_thold < log.level_debug) break;
        const p = piece(ctx, cur_p.data[k].id);
        defer freePiece(p);
        spcDbg(func, " - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n", .{ seq_id, @as(c_int, @intCast(k)), pos, cur_p.data[k].id, @as(f64, cur_p.data[k].p), p.ptr });
    }
}

/// `LOG_DBG` of the same, for DFlash, which logs without the `spec` prefix.
fn logCandidatesPlain(ctx: ?*c.llama_context, seq_id: c.llama_seq_id, pos: c_int, cur_p: *const c.llama_token_data_array) void {
    var k: usize = 0;
    while (k < @min(3, cur_p.size)) : (k += 1) {
        if (log.verbosity_thold < log.level_debug) break;
        const p = piece(ctx, cur_p.data[k].id);
        defer freePiece(p);
        log.dbg(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n", .{ seq_id, @as(c_int, @intCast(k)), pos, cur_p.data[k].id, @as(f64, cur_p.data[k].p), p.ptr });
    }
}

fn rowCopy(dst: [*]f32, src: [*]const f32, n: usize) void {
    if (n == 0) return;
    @memcpy(dst[0..n], src[0..n]);
}

/// One backend top-k chain per sequence on the draft context, or null
/// where `llama_set_sampler` refuses it.
fn backendChains(ctx_dft: ?*c.llama_context, n_seq: u32, on: bool) []?*c.llama_sampler {
    const chains = gpa.alloc(?*c.llama_sampler, n_seq) catch oom();
    @memset(chains, null);
    if (!on) return chains;
    for (0..n_seq) |s| {
        const seq_id: c.llama_seq_id = @intCast(s);
        var chain = c.llama_sampler_chain_init(c.llama_sampler_chain_default_params());
        c.llama_sampler_chain_add(chain, c.llama_sampler_init_top_k(10));

        if (!c.llama_set_sampler(ctx_dft, seq_id, chain)) {
            spcWrn("common_speculative_impl", "backend offload failed for seq_id=%d; using CPU sampler\n", .{seq_id});
            c.llama_sampler_free(chain);
            chain = null;
        }
        chains[s] = chain;
    }
    return chains;
}

/// The destructors' loop: detach and free every backend chain.
fn freeBackendChains(ctx_dft: ?*c.llama_context, chains: []?*c.llama_sampler) void {
    for (chains, 0..) |chain, s| {
        if (chain == null) continue;
        if (ctx_dft != null) _ = c.llama_set_sampler(ctx_dft, @intCast(s), null);
        c.llama_sampler_free(chain);
    }
    gpa.free(chains);
}

fn freeSamplers(smpls: []*sampling.Sampler) void {
    for (smpls) |s| s.destroy();
    gpa.free(smpls);
}

/// A batch of `n` tokens with `embd`-wide embeddings and a token array
/// beside them.
fn tokenAndEmbdBatch(n: i32, embd: i32) c.llama_batch {
    var batch = c.llama_batch_init(n, embd, 1);
    // llama_batch_init allocates only one of token/embd; the decoder needs both.
    batch.token = @ptrCast(@alignCast(c.malloc(@sizeOf(c.llama_token) * @as(usize, @intCast(n)))));
    return batch;
}

fn freeTokenAndEmbdBatch(batch: *c.llama_batch) void {
    if (batch.token != null) {
        c.free(batch.token);
        batch.token = null;
    }
    c.llama_batch_free(batch.*);
}

// -----------------------------------------------------------------------------
// draft-simple

/// Ports `common_speculative_impl_draft_simple` (speculative.cpp:179
/// @b29c606e2): a standalone draft model, decoding its own guesses.
pub const DraftSimple = struct {
    params: params_mod.SpeculativeDraft,
    batch: c.llama_batch,
    smpls: []*sampling.Sampler,

    /// Ports `common_speculative_impl_draft_simple` (speculative.cpp:186
    /// @b29c606e2), the constructor.
    pub fn init(base: *Base, params: *const params_mod.Speculative) S.InitError!DraftSimple {
        const func = "common_speculative_impl_draft_simple";
        const p = params.draft;
        const ctx_dft = p.ctx_dft;
        const ctx_tgt = p.ctx_tgt;

        if (ctx_dft == null) return S.throwRuntime("draft-simple requires a draft context");

        spcTrc(func, "%s", .{@as([*:0]const u8, "adding speculative implementation 'draft-simple'\n")});
        spcTrc(func, "- n_max=%d, n_min=%d, p_min=%f\n", .{ p.n_max, p.n_min, @as(f64, p.p_min) });
        const devs = S.devicesStr(p.devices.items);
        defer gpa.free(devs);
        spcTrc(func, "- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n", .{
            p.n_gpu_layers,
            c.ggml_type_name(p.cache_type_k),
            c.ggml_type_name(p.cache_type_v),
            @as([*:0]const u8, if (ctx_tgt != null) "yes" else "no"),
            @as([*:0]const u8, if (ctx_dft != null) "yes" else "no"),
            devs.ptr,
        });

        const batch = c.llama_batch_init(@intCast(c.llama_n_batch(ctx_dft)), 0, 1);

        const smpls = gpa.alloc(*sampling.Sampler, base.n_seq) catch oom();
        for (smpls) |*s| s.* = draftSampler(c.llama_get_model(ctx_dft));

        var self: DraftSimple = .{ .params = p, .batch = batch, .smpls = smpls };

        const vocab_cmpt = S.areCompatible(c.llama_get_model(ctx_tgt), c.llama_get_model(ctx_dft));
        spcDbg(func, "vocab_cmpt = %d\n", .{@as(c_int, @intFromBool(vocab_cmpt))});

        if (!vocab_cmpt) {
            spcErr(func, "%s", .{@as([*:0]const u8, "the target and draft vocabs are not compatible\n")});
            self.deinit();
            return S.throwRuntime("draft model vocab type must match target model to use speculation");
        }

        if (base.n_seq != c.llama_n_seq_max(ctx_dft)) {
            spcErr(func, "n_seq mismatch: %d != %d\n", .{ base.n_seq, c.llama_n_seq_max(ctx_dft) });
            self.deinit();
            return S.throwRuntime("the draft model number of sequences is incompatible with the speculative n_seq");
        }
        return self;
    }

    /// Ports the destructor of `common_speculative_impl_draft_simple` (speculative.cpp:254 @b29c606e2).
    pub fn deinit(self: *DraftSimple) void {
        c.llama_batch_free(self.batch);
        freeSamplers(self.smpls);
    }

    /// Ports `begin` (speculative.cpp:258 @b29c606e2): nothing to do.
    pub fn begin(_: *DraftSimple, _: *Base, _: c.llama_seq_id, _: []const c.llama_token) void {
        // noop
    }

    /// Ports `process` (speculative.cpp:262 @b29c606e2): the target's batch
    /// through the draft model, without outputs.
    pub fn process(self: *DraftSimple, _: *Base, batch: c.llama_batch) bool {
        const ctx_dft = self.params.ctx_dft;

        var batch_dft = batch;
        batch_dft.logits = null;

        const ret = c.llama_decode(ctx_dft, batch_dft);

        if (ret != 0) {
            spcErr("process", "failed to decode draft batch, ret = %d\n", .{ret});
            return false;
        }

        return true;
    }

    /// Ports `draft` (speculative.cpp:279 @b29c606e2): every drafting
    /// sequence decodes `id_last`, then its own best guess, until it runs
    /// out of confidence or room.
    pub fn draft(self: *DraftSimple, base: *Base, dparams: []DraftParams) void {
        const ctx_dft = self.params.ctx_dft;
        const n_seq = base.n_seq;

        common.batchClear(&self.batch);

        // keep track of which sequences are still drafting
        var n_drafting: c_int = 0;
        const drafting = gpa.alloc(bool, n_seq) catch oom();
        defer gpa.free(drafting);
        @memset(drafting, false);

        for (0..n_seq) |s| {
            const seq_id: c.llama_seq_id = @intCast(s);
            const dp = &dparams[s];

            if (!dp.drafting) continue;

            n_drafting += 1;
            drafting[s] = true;
            self.smpls[s].reset();

            common.batchAdd(&self.batch, dp.id_last, dp.pos0, &.{seq_id}, true);
        }

        var ret = c.llama_decode(ctx_dft, self.batch);
        if (ret != 0) {
            spcErr("draft", "llama_decode returned %d\n", .{ret});
            return;
        }

        var i: c_int = 0;

        while (n_drafting > 0) {
            var i_batch: c_int = 0;

            common.batchClear(&self.batch);

            for (0..n_seq) |s| {
                const seq_id: c.llama_seq_id = @intCast(s);
                if (!drafting[s]) continue;

                const smpl = self.smpls[s];

                _ = smpl.sample(ctx_dft, i_batch, true) catch oom();
                i_batch += 1;

                const cur_p = smpl.getCandidates(true);

                logCandidates("draft", ctx_dft, seq_id, i, cur_p);

                // add drafted token for each sequence
                const id = cur_p.data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p.data[0].p < self.params.p_min) {
                    drafting[s] = false;
                    n_drafting -= 1;
                    continue;
                }

                smpl.accept(id, true);

                const dp = &dparams[s];
                const result = dp.result.?;

                result.append(gpa, id) catch oom();

                if (self.params.n_max <= @as(i32, @intCast(result.items.len)) or
                    (dp.n_max > 0 and dp.n_max <= @as(i32, @intCast(result.items.len))))
                {
                    drafting[s] = false;
                    n_drafting -= 1;
                    continue;
                }

                common.batchAdd(&self.batch, id, dp.pos0 + i + 1, &.{seq_id}, true);
            }

            if (self.batch.n_tokens == 0) break;

            // evaluate the drafted tokens on the draft model
            ret = c.llama_decode(ctx_dft, self.batch);
            if (ret != 0) {
                spcErr("draft", "llama_decode[%d] returned %d\n", .{ i, ret });
                break;
            }

            i += 1;
        }

        for (dparams) |*dp| {
            if (!dp.drafting) continue;

            if (dp.result.?.items.len < @as(usize, @bitCast(@as(isize, self.params.n_min)))) dp.result.?.clearRetainingCapacity();
        }
    }

    /// Ports `accept` (speculative.cpp:386 @b29c606e2): nothing to do.
    pub fn accept(_: *DraftSimple, _: *Base, _: c.llama_seq_id, _: u16, _: bool) void {
        // noop
    }
};

// -----------------------------------------------------------------------------
// draft-eagle3

/// Ports `common_speculative_impl_draft_eagle3` (speculative.cpp:426
/// @b29c606e2). See the C++'s comment for the input convention and the
/// deferred boundary: at memory position P the decoder reads the pair
/// (token[P+1], g[P]), so the last position of every `process` waits for the
/// next call -- or `draft`'s seed step -- to bring its token.
pub const Eagle3 = struct {
    params: params_mod.SpeculativeDraft,
    batch: c.llama_batch,

    smpls: []*sampling.Sampler,

    // backend sampler chain per seq, attached to ctx_dft
    backend_chains: []?*c.llama_sampler,

    n_embd_dec: i32 = 0, // draft hidden size
    n_embd_enc: i32 = 0, // target_layer_ids_n * target_hidden_size
    n_embd_tgt: i32 = 0, // target model hidden size
    n_layer_tgt: i32 = 0, // target model layer count

    target_layer_ids: [*c]const i32 = null, // model_dft's extract layer indices
    target_layer_ids_n: u32 = 0,

    // [per-seq] deferred boundary state
    pending_g_last: [][]f32,
    pending_pos_last: []c.llama_pos,

    // [per-seq] snapshot of the most recent process()'s encoder output
    verify_g: []std.ArrayList(f32), // [n_seq][n_rows * n_embd_dec]
    verify_pos_first: []c.llama_pos, // [n_seq] — pos of verify_g[seq][0]
    verify_g_rows: []i32, // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    features_buf: std.ArrayList(f32) = .empty,
    g_embd_buf: std.ArrayList(f32) = .empty,

    /// Ports `common_speculative_impl_draft_eagle3` (speculative.cpp:456
    /// @b29c606e2), the constructor.
    pub fn init(base: *Base, params: *const params_mod.Speculative) S.InitError!Eagle3 {
        const func = "common_speculative_impl_draft_eagle3";
        const n_seq = base.n_seq;
        spcTrc(func, "%s", .{@as([*:0]const u8, "adding speculative implementation 'draft-eagle3'\n")});
        spcTrc(func, "- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", .{ params.draft.n_max, params.draft.n_min, @as(f64, params.draft.p_min), @as(c_int, @intFromBool(params.draft.backend_sampling)) });

        const p = params.draft;
        const ctx_tgt = p.ctx_tgt;
        const ctx_dft = p.ctx_dft;
        assertC(ctx_tgt != null and ctx_dft != null, 465, "ctx_tgt && ctx_dft && \"EAGLE3 requires ctx_tgt and ctx_dft to be set\"");

        const model_dft = c.llama_get_model(ctx_dft);
        const model_tgt = c.llama_get_model(ctx_tgt);

        const target_layer_ids = ext.modelTargetLayerIds(model_dft);
        const target_layer_ids_n = ext.modelTargetLayerIdsN(model_dft);
        if (target_layer_ids_n != 3) {
            return S.throwRuntimeFmt("draft model is not eagle3 (expected 3 extract layers, got %u)", .{target_layer_ids_n});
        }

        const n_embd_tgt = c.llama_model_n_embd(model_tgt);
        const n_embd_dec = c.llama_model_n_embd(model_dft);
        const n_embd_enc: i32 = @as(i32, @intCast(target_layer_ids_n)) * n_embd_tgt;
        const n_layer_tgt = c.llama_model_n_layer(model_tgt);

        const n_b: i32 = @intCast(c.llama_n_batch(ctx_dft));
        const batch = tokenAndEmbdBatch(n_b, n_embd_dec);

        const smpls = gpa.alloc(*sampling.Sampler, n_seq) catch oom();
        for (smpls) |*s| s.* = draftSampler(c.llama_get_model(ctx_dft));

        // offload draft sampling to the backend
        const chains = backendChains(ctx_dft, n_seq, p.backend_sampling);

        // turn on extraction of the target layers' hidden states
        for (0..target_layer_ids_n) |k| {
            const lid = target_layer_ids[k];
            if (lid < n_layer_tgt) {
                ext.setEmbeddingsLayerInp(ctx_tgt, @intCast(lid), true);
            } else if (lid == n_layer_tgt) {
                ext.setEmbeddingsNextn(ctx_tgt, true, false);
            } else {
                _ = c.ggml_abort("speculative.cpp", 520, "EAGLE3: target layer id %d exceeds target n_layer %d", lid, n_layer_tgt);
            }
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        ext.setEmbeddingsNextn(ctx_dft, true, true);

        const pending_g_last = gpa.alloc([]f32, n_seq) catch oom();
        for (pending_g_last) |*g| {
            g.* = gpa.alloc(f32, @intCast(n_embd_dec)) catch oom();
            @memset(g.*, 0.0);
        }
        const pending_pos_last = gpa.alloc(c.llama_pos, n_seq) catch oom();
        @memset(pending_pos_last, -1);
        const verify_g = gpa.alloc(std.ArrayList(f32), n_seq) catch oom();
        @memset(verify_g, .empty);
        const verify_pos_first = gpa.alloc(c.llama_pos, n_seq) catch oom();
        @memset(verify_pos_first, -1);
        const verify_g_rows = gpa.alloc(i32, n_seq) catch oom();
        @memset(verify_g_rows, 0);

        return .{
            .params = p,
            .batch = batch,
            .smpls = smpls,
            .backend_chains = chains,
            .n_embd_dec = n_embd_dec,
            .n_embd_enc = n_embd_enc,
            .n_embd_tgt = n_embd_tgt,
            .n_layer_tgt = n_layer_tgt,
            .target_layer_ids = target_layer_ids,
            .target_layer_ids_n = target_layer_ids_n,
            .pending_g_last = pending_g_last,
            .pending_pos_last = pending_pos_last,
            .verify_g = verify_g,
            .verify_pos_first = verify_pos_first,
            .verify_g_rows = verify_g_rows,
        };
    }

    /// Ports the destructor of `common_speculative_impl_draft_eagle3` (speculative.cpp:536 @b29c606e2).
    pub fn deinit(self: *Eagle3) void {
        freeBackendChains(self.params.ctx_dft, self.backend_chains);
        freeTokenAndEmbdBatch(&self.batch);
        freeSamplers(self.smpls);
        for (self.pending_g_last) |g| gpa.free(g);
        gpa.free(self.pending_g_last);
        gpa.free(self.pending_pos_last);
        for (self.verify_g) |*g| g.deinit(gpa);
        gpa.free(self.verify_g);
        gpa.free(self.verify_pos_first);
        gpa.free(self.verify_g_rows);
        self.features_buf.deinit(gpa);
        self.g_embd_buf.deinit(gpa);
    }

    /// Ports `begin` (speculative.cpp:556 @b29c606e2): warns when the
    /// draft context is missing more than the one deferred position.
    pub fn begin(self: *Eagle3, _: *Base, seq_id: c.llama_seq_id, prompt: []const c.llama_token) void {
        const n: i32 = @intCast(prompt.len);
        if (n <= 0) return;
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        const pos_max = c.llama_memory_seq_pos_max(c.llama_get_memory(self.params.ctx_dft), seq_id);
        if (pos_max < n - 2) {
            spcWrn("begin", "ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. " ++
                "Drafts may degrade.\n", .{ pos_max, n - 2 });
        }
    }

    /// Ports `process` (speculative.cpp:572 @b29c606e2): the target's
    /// features through the encoder, then the decoder over this batch's
    /// pairs, the last position deferred.
    pub fn process(self: *Eagle3, base: *Base, batch_in: c.llama_batch) bool {
        if (batch_in.n_tokens <= 0) return true;

        if (batch_in.token == null or batch_in.embd != null) return true;

        const n_seq = base.n_seq;
        const n_tokens: usize = @intCast(batch_in.n_tokens);
        const n_embd_dec: usize = @intCast(self.n_embd_dec);
        const n_embd_enc: usize = @intCast(self.n_embd_enc);
        const n_embd_tgt: usize = @intCast(self.n_embd_tgt);

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        const i_batch_beg = gpa.alloc(i32, n_seq) catch oom();
        defer gpa.free(i_batch_beg);
        const i_batch_end = gpa.alloc(i32, n_seq) catch oom();
        defer gpa.free(i_batch_end);
        @memset(i_batch_beg, -1);
        @memset(i_batch_end, -1);
        for (0..n_tokens) |k| {
            assertC(batch_in.n_seq_id[k] == 1, 587, "batch_in.n_seq_id[k] == 1");
            const seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 or seq_id >= @as(c.llama_seq_id, @intCast(n_seq))) continue;
            const s: usize = @intCast(seq_id);
            i_batch_end[s] = @intCast(k);
            if (i_batch_beg[s] < 0) i_batch_beg[s] = @intCast(k);
        }

        const ctx_tgt = self.params.ctx_tgt;
        const ctx_dft = self.params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        const old_len = self.features_buf.items.len;
        self.features_buf.resize(gpa, n_tokens * n_embd_enc) catch oom();
        if (self.features_buf.items.len > old_len) @memset(self.features_buf.items[old_len..], 0.0);

        for (0..self.target_layer_ids_n) |k| {
            const lid = self.target_layer_ids[k];
            const layer: ?[*]const f32 = if (lid < self.n_layer_tgt)
                ext.getEmbeddingsLayerInp(ctx_tgt, @intCast(lid))
            else
                ext.getEmbeddingsNextn(ctx_tgt);
            if (layer == null) {
                _ = c.ggml_abort("speculative.cpp", 613, "EAGLE3: target layer %d input not extracted.", lid);
            }
            for (0..n_tokens) |i| {
                const dst = self.features_buf.items.ptr + i * n_embd_enc + k * n_embd_tgt;
                const src = layer.? + i * n_embd_tgt;
                rowCopy(dst, src, n_embd_tgt);
            }
        }

        self.g_embd_buf.resize(gpa, n_tokens * n_embd_dec) catch oom();

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const n_ubatch_dft: usize = @intCast(c.llama_n_ubatch(ctx_dft));
        var i: usize = 0;
        while (i < n_tokens) : (i += n_ubatch_dft) {
            const n_chunk = @min(n_ubatch_dft, n_tokens - i);

            const enc_batch: c.llama_batch = .{
                .n_tokens = @intCast(n_chunk),
                .token = null,
                .embd = self.features_buf.items.ptr + i * n_embd_enc,
                .pos = null,
                .n_seq_id = null,
                .seq_id = null,
                .logits = null,
            };
            const rc = c.llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                spcErr("process", "llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n", .{ rc, @as(c_int, @intCast(n_chunk)), @as(c_int, @intCast(i)) });
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const g_embd_chunk = ext.getEmbeddingsNextn(ctx_dft);
            assertC(g_embd_chunk != null, 652, "g_embd_chunk && \"EAGLE3 encoder produced no output.\"");
            rowCopy(self.g_embd_buf.items.ptr + i * n_embd_dec, g_embd_chunk.?, n_chunk * n_embd_dec);
        }

        const g_embd = self.g_embd_buf.items.ptr;

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        common.batchClear(&self.batch);

        for (0..n_seq) |s| {
            const seq_id: c.llama_seq_id = @intCast(s);
            const beg = i_batch_beg[s];
            const end = i_batch_end[s];
            if (beg < 0 or end < 0) continue;
            const ub: usize = @intCast(beg);
            const ue: usize = @intCast(end);

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            const pending_pos = self.pending_pos_last[s];
            if (pending_pos >= 0 and pending_pos + 1 == batch_in.pos[ub]) {
                const dft_pos_max = c.llama_memory_seq_pos_max(c.llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common.batchAdd(&self.batch, batch_in.token[ub], pending_pos, &.{seq_id}, false);
                    rowCopy(self.batch.embd + @as(usize, @intCast(self.batch.n_tokens - 1)) * n_embd_dec, self.pending_g_last[s].ptr, n_embd_dec);
                }
            }

            var k = ub;
            while (k < ue) : (k += 1) {
                common.batchAdd(&self.batch, batch_in.token[k + 1], batch_in.pos[k], &.{seq_id}, false);
                rowCopy(self.batch.embd + @as(usize, @intCast(self.batch.n_tokens - 1)) * n_embd_dec, g_embd + k * n_embd_dec, n_embd_dec);
            }

            // refresh deferred state
            const n_rows: usize = ue - ub + 1;
            self.verify_pos_first[s] = batch_in.pos[ub];
            self.pending_pos_last[s] = batch_in.pos[ue];
            self.verify_g_rows[s] = @intCast(n_rows);
            const vg = &self.verify_g[s];
            const vold = vg.items.len;
            vg.resize(gpa, n_rows * n_embd_dec) catch oom();
            if (vg.items.len > vold) @memset(vg.items[vold..], 0.0);
            rowCopy(vg.items.ptr, g_embd + ub * n_embd_dec, n_embd_dec * n_rows);
            rowCopy(self.pending_g_last[s].ptr, g_embd + ue * n_embd_dec, n_embd_dec);
        }

        if (self.batch.n_tokens > 0) {
            const rc = c.llama_decode(ctx_dft, self.batch);
            if (rc != 0) {
                spcErr("process", "llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n", .{ rc, self.batch.n_tokens, batch_in.pos[0] });
                return false;
            }
        }

        return true;
    }

    /// Ports `draft` (speculative.cpp:721 @b29c606e2): the seed step
    /// completes the deferred pair with `id_last`, then each step feeds back
    /// its own pre-norm state.
    pub fn draft(self: *Eagle3, base: *Base, dparams: []DraftParams) void {
        const ctx_dft = self.params.ctx_dft;
        const n_seq = base.n_seq;
        const n_embd_dec: usize = @intCast(self.n_embd_dec);

        common.batchClear(&self.batch);

        // keep track of which sequences are still drafting
        var n_drafting: c_int = 0;
        const drafting = gpa.alloc(bool, n_seq) catch oom();
        defer gpa.free(drafting);
        @memset(drafting, false);

        for (0..n_seq) |s| {
            const seq_id: c.llama_seq_id = @intCast(s);
            const dp = &dparams[s];

            if (!dp.drafting) continue;
            if (self.pending_pos_last[s] < 0) continue;

            n_drafting += 1;
            drafting[s] = true;
            self.smpls[s].reset();

            _ = c.llama_memory_seq_rm(c.llama_get_memory(ctx_dft), seq_id, self.pending_pos_last[s], -1);

            common.batchAdd(&self.batch, dp.id_last, self.pending_pos_last[s], &.{seq_id}, true);
            rowCopy(self.batch.embd + @as(usize, @intCast(self.batch.n_tokens - 1)) * n_embd_dec, self.pending_g_last[s].ptr, n_embd_dec);
        }

        if (self.batch.n_tokens == 0) return;

        var ret = c.llama_decode(ctx_dft, self.batch);
        if (ret != 0) {
            spcErr("draft", "llama_decode returned %d\n", .{ret});
            return;
        }

        var i: c_int = 0;

        while (n_drafting > 0) {
            var i_batch: c_int = 0;

            common.batchClear(&self.batch);

            for (0..n_seq) |s| {
                const seq_id: c.llama_seq_id = @intCast(s);
                if (!drafting[s]) continue;

                const smpl = self.smpls[s];

                _ = smpl.sample(ctx_dft, i_batch, true) catch oom();
                // pre-norm hidden state of this position becomes g_embd for the next step
                const prenorm = ext.getEmbeddingsNextnIth(ctx_dft, i_batch);
                i_batch += 1;

                const cur_p = smpl.getCandidates(true);

                logCandidates("draft", ctx_dft, seq_id, i, cur_p);

                const id = cur_p.data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p.data[0].p < self.params.p_min) {
                    drafting[s] = false;
                    n_drafting -= 1;
                    continue;
                }

                smpl.accept(id, true);

                const dp = &dparams[s];
                const result = dp.result.?;

                result.append(gpa, id) catch oom();

                if (self.params.n_max <= @as(i32, @intCast(result.items.len))) {
                    drafting[s] = false;
                    n_drafting -= 1;
                    continue;
                }

                common.batchAdd(&self.batch, id, self.pending_pos_last[s] + (i + 1), &.{seq_id}, true);
                rowCopy(self.batch.embd + @as(usize, @intCast(self.batch.n_tokens - 1)) * n_embd_dec, prenorm.?, n_embd_dec);
            }

            if (self.batch.n_tokens == 0) break;

            ret = c.llama_decode(ctx_dft, self.batch);
            if (ret != 0) {
                spcErr("draft", "llama_decode[%d] returned %d\n", .{ i, ret });
                break;
            }

            i += 1;
        }

        for (0..n_seq) |s| {
            const dp = &dparams[s];
            if (!dp.drafting) continue;

            if (dp.result.?.items.len < @as(usize, @bitCast(@as(isize, self.params.n_min)))) dp.result.?.clearRetainingCapacity();
        }
    }

    /// Ports `accept` (speculative.cpp:848 @b29c606e2): the deferred
    /// boundary moves to the first position the target did not accept.
    pub fn accept(self: *Eagle3, base: *Base, seq_id: c.llama_seq_id, n_accepted: u16, _: bool) void {
        if (seq_id < 0 or seq_id >= @as(c.llama_seq_id, @intCast(base.n_seq))) return;
        const s: usize = @intCast(seq_id);

        const n_rows = self.verify_g_rows[s];
        if (n_rows <= 0) return;

        const i_g: i32 = @min(@as(i32, n_accepted), n_rows - 1);
        self.pending_pos_last[s] = self.verify_pos_first[s] + i_g;
        const n_embd_dec: usize = @intCast(self.n_embd_dec);
        rowCopy(self.pending_g_last[s].ptr, self.verify_g[s].items.ptr + @as(usize, @intCast(i_g)) * n_embd_dec, n_embd_dec);
    }

    /// Ports `need_boundary_stash` (speculative.cpp:867 @b29c606e2): only a
    /// recurrent or hybrid target drops the deferred row on restore.
    fn needBoundaryStash(self: *const Eagle3) bool {
        const model_tgt = c.llama_get_model(self.params.ctx_tgt);
        return c.llama_model_is_recurrent(model_tgt) or c.llama_model_is_hybrid(model_tgt);
    }

    /// Ports `get_state` (speculative.cpp:872 @b29c606e2): the deferred
    /// position and its row, as bytes.
    pub fn getState(self: *const Eagle3, base: *const Base, seq_id: c.llama_seq_id, data: *std.ArrayList(u8)) bool {
        if (!self.needBoundaryStash()) return false;
        if (seq_id < 0 or seq_id >= @as(c.llama_seq_id, @intCast(base.n_seq)) or self.pending_pos_last[@intCast(seq_id)] < 0) return false;

        const s: usize = @intCast(seq_id);
        const pos = self.pending_pos_last[s];
        const g = self.pending_g_last[s];

        data.resize(gpa, @sizeOf(c.llama_pos) + g.len * @sizeOf(f32)) catch oom();
        @memcpy(data.items[0..@sizeOf(c.llama_pos)], std.mem.asBytes(&pos));
        @memcpy(data.items[@sizeOf(c.llama_pos)..], std.mem.sliceAsBytes(g));
        return true;
    }

    /// Ports `set_state` (speculative.cpp:889 @b29c606e2).
    pub fn setState(self: *Eagle3, base: *const Base, seq_id: c.llama_seq_id, data: []const u8) void {
        if (!self.needBoundaryStash()) return;
        if (seq_id < 0 or seq_id >= @as(c.llama_seq_id, @intCast(base.n_seq))) return;
        const n_embd_dec: usize = @intCast(self.n_embd_dec);
        if (data.len != @sizeOf(c.llama_pos) + n_embd_dec * @sizeOf(f32)) return;

        const s: usize = @intCast(seq_id);
        var pos: c.llama_pos = -1;
        @memcpy(std.mem.asBytes(&pos), data[0..@sizeOf(c.llama_pos)]);

        self.pending_pos_last[s] = pos;
        if (self.pending_g_last[s].len != n_embd_dec) {
            self.pending_g_last[s] = gpa.realloc(self.pending_g_last[s], n_embd_dec) catch oom();
        }
        @memcpy(std.mem.sliceAsBytes(self.pending_g_last[s]), data[@sizeOf(c.llama_pos)..]);
    }
};

// -----------------------------------------------------------------------------
// draft-dflash, draft-dspark

/// Ports `common_speculative_impl_draft_dflash` (speculative.cpp:910
/// @b29c606e2): block-diffusion drafting with the target's features
/// injected into the draft's KV cache. DSpark is the same driver with a
/// Markov head, an anchor-first block and an optional confidence head.
pub const DFlash = struct {
    params: params_mod.SpeculativeDraft,

    batch: c.llama_batch, // noise tokens
    batch_inject: c.llama_batch, // target features for KV cache injection

    smpls: []*sampling.Sampler,

    // backend sampler chain per seq, attached to ctx_dft
    backend_chains: []?*c.llama_sampler,

    n_embd_dec: i32 = 0, // draft hidden size
    n_embd_enc: i32 = 0, // target_layer_ids_n * target_hidden_size
    n_embd_tgt: i32 = 0, // target model hidden size

    block_size: i32 = 0,
    mask_token_id: c.llama_token = 0,

    is_dflash2: bool = false,
    is_mrope: bool = false,
    selector_top_k: i32 = 0,

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    is_dspark: bool,

    // dspark speculators
    sample_from_anchor: bool = true,

    // block-internal attention
    causal_attn: bool = false,

    target_layer_ids: [*c]const i32 = null, // model_dft's extract layer indices
    target_layer_ids_n: u32 = 0,

    /// Ports `common_speculative_impl_draft_dflash` (speculative.cpp:944
    /// @b29c606e2), the constructor.
    pub fn init(base: *Base, params: *const params_mod.Speculative, kind: params_mod.SpeculativeType) S.InitError!DFlash {
        const func = "common_speculative_impl_draft_dflash";
        var p = params.draft;
        const n_seq = base.n_seq;
        const is_dspark = kind == .draft_dspark;
        const ctx_tgt = p.ctx_tgt;
        const ctx_dft = p.ctx_dft;
        assertC(ctx_tgt != null and ctx_dft != null, 953, "ctx_tgt && ctx_dft && \"DFlash requires ctx_tgt and ctx_dft to be set\"");

        const model_dft = c.llama_get_model(ctx_dft);
        const model_tgt = c.llama_get_model(ctx_tgt);

        const target_layer_ids = ext.modelTargetLayerIds(model_dft);
        const target_layer_ids_n = ext.modelTargetLayerIdsN(model_dft);
        assertC(target_layer_ids_n > 0, 960, "target_layer_ids_n > 0 && \"DFlash model has no target_layer_ids\"");

        const n_embd_tgt = c.llama_model_n_embd(model_tgt);
        const n_embd_dec = c.llama_model_n_embd(model_dft);
        const n_embd_enc: i32 = @as(i32, @intCast(target_layer_ids_n)) * n_embd_tgt;

        // read the trained block size from the dflash.block_size metadata key
        var block_size: i32 = 16;
        var sample_from_anchor = true;
        var causal_attn = false;
        {
            var buf = std.mem.zeroes([32]u8);
            if (c.llama_model_meta_val_str(model_dft, "dflash.block_size", &buf, buf.len) >= 0) {
                block_size = c.atoi(@ptrCast(&buf));
            }
            if (c.llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", &buf, buf.len) >= 0) {
                sample_from_anchor = std.mem.eql(u8, std.mem.sliceTo(&buf, 0), "true");
            }
            if (c.llama_model_meta_val_str(model_dft, "dflash.attention.causal", &buf, buf.len) >= 0) {
                causal_attn = std.mem.eql(u8, std.mem.sliceTo(&buf, 0), "true");
            }
        }

        const selector_top_k = ext.modelDflashSelectorTopK(model_dft);
        const is_dflash2 = selector_top_k > 0;
        const mask_token_id = c.llama_vocab_mask(c.llama_model_get_vocab(model_dft));

        if (is_dspark and p.p_min > 0.0) {
            var buf = std.mem.zeroes([16]u8);
            const has_conf =
                c.llama_model_meta_val_str(model_dft, "dflash.has_confidence_head", &buf, buf.len) < 0 or
                std.mem.eql(u8, std.mem.sliceTo(&buf, 0), "true");
            if (!has_conf) {
                return S.throwRuntime("DSpark draft has no confidence head: please set --spec-draft-p-min 0");
            }
        }

        const type_str = S.typeToStr(kind);
        log.inf("%s: adding speculative implementation '%s'\n", .{ @as([*:0]const u8, func), type_str.ptr });
        log.inf("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", .{ @as([*:0]const u8, func), p.n_max, p.n_min, @as(f64, p.p_min) });
        log.inf("%s: - block_size=%d, mask_token_id=%d, n_extract=%u, sample_from_anchor=%s\n", .{ @as([*:0]const u8, func), block_size, mask_token_id, target_layer_ids_n, @as([*:0]const u8, if (sample_from_anchor) "true" else "false") });

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
        // block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
        const n_draft_max: i32 = if (is_dspark and sample_from_anchor) block_size else block_size -% 1;
        if (p.n_max > n_draft_max or p.n_min > n_draft_max) {
            log.wrn("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n", .{ @as([*:0]const u8, func), p.n_max, p.n_min, block_size, n_draft_max });
            p.n_max = @min(p.n_max, n_draft_max);
            p.n_min = @min(p.n_min, n_draft_max);
        }
        base.n_max = p.n_max;

        const batch = c.llama_batch_init(@intCast(c.llama_n_batch(ctx_dft)), 0, @intCast(n_seq));
        var batch_inject = c.llama_batch_init(@intCast(c.llama_n_ubatch(ctx_dft)), n_embd_enc, @intCast(n_seq));

        // embd batches on an M-RoPE draft need 4 position rows per token
        const is_mrope = c.llama_model_rope_type(model_dft) == c.LLAMA_ROPE_TYPE_MROPE;
        if (is_mrope) {
            c.free(batch_inject.pos);
            batch_inject.pos = @ptrCast(@alignCast(c.malloc(@sizeOf(c.llama_pos) * 4 * @as(usize, c.llama_n_batch(ctx_dft)))));
        }

        const smpls = gpa.alloc(*sampling.Sampler, n_seq) catch oom();
        for (smpls) |*s| s.* = draftSampler(model_dft);

        // offload draft sampling to the backend
        const chains = backendChains(ctx_dft, n_seq, p.backend_sampling and !is_dflash2);

        // turn on extraction of the target layers' input embeddings
        for (0..target_layer_ids_n) |k| ext.setEmbeddingsLayerInp(ctx_tgt, @intCast(target_layer_ids[k]), true);

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        ext.setEmbeddingsNextn(ctx_dft, true, !is_dflash2);
        c.llama_set_causal_attn(ctx_dft, causal_attn); // DFlash needs non-causal attention unless the model says otherwise

        return .{
            .params = p,
            .batch = batch,
            .batch_inject = batch_inject,
            .smpls = smpls,
            .backend_chains = chains,
            .n_embd_dec = n_embd_dec,
            .n_embd_enc = n_embd_enc,
            .n_embd_tgt = n_embd_tgt,
            .block_size = block_size,
            .mask_token_id = mask_token_id,
            .is_dflash2 = is_dflash2,
            .is_mrope = is_mrope,
            .selector_top_k = selector_top_k,
            .is_dspark = is_dspark,
            .sample_from_anchor = sample_from_anchor,
            .causal_attn = causal_attn,
            .target_layer_ids = target_layer_ids,
            .target_layer_ids_n = target_layer_ids_n,
        };
    }

    /// Ports the destructor of `common_speculative_impl_draft_dflash` (speculative.cpp:1055 @b29c606e2).
    pub fn deinit(self: *DFlash) void {
        freeBackendChains(self.params.ctx_dft, self.backend_chains);
        c.llama_batch_free(self.batch);
        c.llama_batch_free(self.batch_inject);
        freeSamplers(self.smpls);
    }

    /// Ports `begin` (speculative.cpp:1072 @b29c606e2).
    pub fn begin(self: *DFlash, base: *Base, seq_id: c.llama_seq_id, prompt: []const c.llama_token) void {
        if (seq_id < 0 or seq_id >= @as(c.llama_seq_id, @intCast(base.n_seq))) return;

        const n: i32 = @intCast(prompt.len);
        if (n <= 0) return;

        const pos_max = c.llama_memory_seq_pos_max(c.llama_get_memory(self.params.ctx_dft), seq_id);
        if (pos_max < n - 1) {
            log.wrn("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. " ++
                "Drafts may degrade.\n", .{ @as([*:0]const u8, "begin"), pos_max, n - 1 });
        }
    }

    /// Ports `process` (speculative.cpp:1090 @b29c606e2): each sequence's
    /// target features, chunked by the draft's micro-batch, injected into
    /// the draft's KV cache at the target positions.
    pub fn process(self: *DFlash, base: *Base, batch_in: c.llama_batch) bool {
        if (batch_in.n_tokens <= 0) return true;

        // Target prefill may contain token IDs or multimodal embeddings. Both
        // produce the target-layer features used to seed the draft KV cache, so
        // embeddings are injected too, except the pinned ones skipped below.
        const has_tokens = batch_in.token != null;
        const has_embeddings = batch_in.embd != null;
        if (has_tokens == has_embeddings) return true;

        const n_seq = base.n_seq;
        const n_tokens: usize = @intCast(batch_in.n_tokens);
        const n_embd_enc: usize = @intCast(self.n_embd_enc);
        const n_embd_tgt: usize = @intCast(self.n_embd_tgt);

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        const i_batch_beg = gpa.alloc(i32, n_seq) catch oom();
        defer gpa.free(i_batch_beg);
        const i_batch_end = gpa.alloc(i32, n_seq) catch oom();
        defer gpa.free(i_batch_end);
        @memset(i_batch_beg, -1);
        @memset(i_batch_end, -1);
        for (0..n_tokens) |k| {
            assertC(batch_in.n_seq_id[k] == 1, 1111, "batch_in.n_seq_id[k] == 1");
            const seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 or seq_id >= @as(c.llama_seq_id, @intCast(n_seq))) continue;
            const s: usize = @intCast(seq_id);
            i_batch_end[s] = @intCast(k);
            if (i_batch_beg[s] < 0) i_batch_beg[s] = @intCast(k);
        }

        const ctx_tgt = self.params.ctx_tgt;
        const ctx_dft = self.params.ctx_dft;

        const n_ubatch: i32 = @intCast(c.llama_n_ubatch(ctx_dft));

        for (0..n_seq) |s| {
            const seq_id: c.llama_seq_id = @intCast(s);
            if (i_batch_beg[s] < 0) continue;
            const ub: usize = @intCast(i_batch_beg[s]);
            const n_rows: i32 = i_batch_end[s] - i_batch_beg[s] + 1;

            // an M-RoPE image pins all its rows to one position, so a windowed draft
            // cache cannot free cells for it - skip it, the draft can jump over the gap
            const pos_pinned = batch_in.pos[ub] == batch_in.pos[@intCast(i_batch_end[s])];
            if (has_embeddings and n_rows > 1 and pos_pinned) continue;

            var offset: i32 = 0;
            while (offset < n_rows) : (offset += n_ubatch) {
                const n_chunk: i32 = @min(n_ubatch, n_rows - offset);
                const nc: usize = @intCast(n_chunk);
                const uo: usize = @intCast(offset);

                // gather target features per extract layer; the fused decode encodes and
                // injects them into the K/V cache at the target positions
                self.batch_inject.n_tokens = n_chunk;
                for (0..self.target_layer_ids_n) |k| {
                    const layer = ext.getEmbeddingsLayerInp(ctx_tgt, @intCast(self.target_layer_ids[k]));
                    if (layer == null) {
                        _ = c.ggml_abort("speculative.cpp", 1149, "DFlash: target layer %d input not extracted.", self.target_layer_ids[k]);
                    }
                    for (0..nc) |i| {
                        const dst = self.batch_inject.embd + i * n_embd_enc + k * n_embd_tgt;
                        const src = layer.? + (ub + uo + i) * n_embd_tgt;
                        rowCopy(dst, src, n_embd_tgt);
                    }
                }

                for (0..nc) |i| {
                    const p = batch_in.pos[ub + uo + i];
                    self.batch_inject.pos[i] = p;
                    if (self.is_mrope) {
                        self.batch_inject.pos[1 * nc + i] = p;
                        self.batch_inject.pos[2 * nc + i] = p;
                        self.batch_inject.pos[3 * nc + i] = 0;
                    }
                    self.batch_inject.n_seq_id[i] = 1;
                    self.batch_inject.seq_id[i][0] = seq_id;
                    self.batch_inject.logits[i] = 0;
                }
                const rc = c.llama_decode(ctx_dft, self.batch_inject);
                if (rc != 0) {
                    log.err("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n", .{ @as([*:0]const u8, "process"), rc, n_chunk, offset });
                    return false;
                }
            }
        }

        return true;
    }

    /// Ports `draft` (speculative.cpp:1182 @b29c606e2): one noise block per
    /// drafting sequence, all decoded at once, then each block read out --
    /// through the selector lattice (DFlash2), the confidence head (DSpark)
    /// or the samplers.
    pub fn draft(self: *DFlash, base: *Base, dparams: []DraftParams) void {
        const ctx_dft = self.params.ctx_dft;
        const n_seq = base.n_seq;

        common.batchClear(&self.batch);

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        const i_block_beg = gpa.alloc(i32, n_seq) catch oom();
        defer gpa.free(i_block_beg);
        const n_block = gpa.alloc(i32, n_seq) catch oom();
        defer gpa.free(n_block);
        @memset(i_block_beg, -1);
        @memset(n_block, 0);

        for (0..n_seq) |s| {
            const seq_id: c.llama_seq_id = @intCast(s);
            const dp = &dparams[s];
            if (!dp.drafting) continue;

            self.smpls[s].reset();

            const n: i32 = dp.pos0;

            const n_draft = self.params.n_max;

            const n_block_tokens: i32 = n_draft + @as(i32, if (self.is_dspark and self.sample_from_anchor) 0 else 1);
            i_block_beg[s] = self.batch.n_tokens;
            n_block[s] = n_block_tokens;
            var i: i32 = 0;
            while (i < n_block_tokens) : (i += 1) {
                common.batchAdd(&self.batch, if (i == 0) dp.id_last else self.mask_token_id, n + i, &.{seq_id}, !self.is_dflash2);
            }
        }

        if (self.batch.n_tokens == 0) return;

        // decode all sequence's noise block in a single batch
        const ret = c.llama_decode(ctx_dft, self.batch);
        if (ret != 0) {
            log.wrn("%s: llama_decode returned %d\n", .{ @as([*:0]const u8, "draft"), ret });
            return;
        }

        const n_embd_dec: usize = @intCast(self.n_embd_dec);
        const top_k: usize = @intCast(@max(self.selector_top_k, 0));

        for (0..n_seq) |s| {
            const seq_id: c.llama_seq_id = @intCast(s);
            if (i_block_beg[s] < 0) continue;
            const dp = &dparams[s];

            const beg = i_block_beg[s];
            const n_block_tokens = n_block[s];

            const smpl = self.smpls[s];

            const result = dp.result.?;

            if (self.is_dflash2) {
                const lattice = ext.getEmbeddingsNextn(ctx_dft);
                assertC(lattice != null, 1234, "lattice && \"DFlash2 selector produced no lattice\"");

                var predecessor: usize = 0;
                var i: i32 = 1;
                while (i < n_block_tokens) : (i += 1) {
                    const row = lattice.? + @as(usize, @intCast(beg + i)) * n_embd_dec;
                    const scores = row + top_k + predecessor * top_k;

                    // std::max_element: the first of the largest
                    var best: usize = 0;
                    for (1..top_k) |k| {
                        if (scores[best] < scores[k]) best = k;
                    }
                    predecessor = best;
                    if (self.params.p_min > 0.0) {
                        // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max))
                        var sum: f32 = 0.0;
                        for (0..top_k) |k| sum += @exp(scores[k] - scores[predecessor]);
                        if (1.0 / sum < self.params.p_min) break;
                    }
                    result.append(gpa, @intFromFloat(row[predecessor])) catch oom();
                }

                if (result.items.len < @as(usize, @bitCast(@as(isize, self.params.n_min)))) result.clearRetainingCapacity();
                continue;
            }

            if (self.is_dspark) {
                // DSpark: read from the first draft slot, truncate below the confidence threshold
                const conf: ?[*]const f32 = if (self.params.p_min > 0.0) ext.getEmbeddingsNextn(ctx_dft) else null;
                // bonus-anchor drafts read the mask positions only, like DFlash
                const i_draft_beg: i32 = if (self.sample_from_anchor) 0 else 1;
                var i = i_draft_beg;
                while (i < n_block_tokens) : (i += 1) {
                    const idx = beg + i;

                    if (conf != null and conf.?[@as(usize, @intCast(idx)) * n_embd_dec] < self.params.p_min) break;

                    _ = smpl.sample(ctx_dft, idx, true) catch oom();

                    const cur_p = smpl.getCandidates(true);

                    logCandidatesPlain(ctx_dft, seq_id, i, cur_p);

                    const id = cur_p.data[0].id;

                    smpl.accept(id, true);

                    result.append(gpa, id) catch oom();
                }
            } else {
                // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
                var i: i32 = 1;
                while (i < n_block_tokens) : (i += 1) {
                    _ = smpl.sample(ctx_dft, beg + i, true) catch oom();

                    const cur_p = smpl.getCandidates(true);

                    logCandidatesPlain(ctx_dft, seq_id, i - 1, cur_p);

                    const id = cur_p.data[0].id;

                    if (cur_p.data[0].p < self.params.p_min) break;

                    smpl.accept(id, true);

                    result.append(gpa, id) catch oom();
                }
            }

            if (result.items.len < @as(usize, @bitCast(@as(isize, self.params.n_min)))) result.clearRetainingCapacity();
        }
    }

    /// Ports `accept` (speculative.cpp:1325 @b29c606e2): nothing to do.
    pub fn accept(_: *DFlash, _: *Base, _: c.llama_seq_id, _: u16, _: bool) void {
        // noop
    }
};

// -----------------------------------------------------------------------------
// draft-mtp

/// Ports `common_speculative_impl_draft_mtp` (speculative.cpp:1330
/// @b29c606e2): multi-token prediction heads, in one of three modes -- a
/// shared KV cache with every head in one graph (Gemma 4), chained heads
/// one per draft step (Step 3.5), or a single head (Qwen 3.5).
pub const Mtp = struct {
    params: params_mod.SpeculativeDraft, // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    batch: c.llama_batch,

    smpls: []*sampling.Sampler,

    // backend sampler chain per seq, attached to ctx_dft
    backend_chains: []?*c.llama_sampler,

    n_embd: i32 = 0,

    n_mtp_layers: i32 = 1,
    is_mem_shared: bool = false, // gemma4
    chain_heads: bool = false, // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    pending_h: [][]f32, // [n_seq][n_embd]

    i_batch_beg: []i32,
    i_batch_end: []i32,

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    verify_h: []std.ArrayList(f32),
    verify_h_rows: []i32,

    i_last: []c_int,
    chain_h: []std.ArrayList(f32),

    /// Ports `common_speculative_impl_draft_mtp` (speculative.cpp:1366
    /// @b29c606e2), the constructor.
    pub fn init(base: *Base, params: *const params_mod.Speculative) S.InitError!Mtp {
        const func = "common_speculative_impl_draft_mtp";
        var p = params.draft;
        const n_seq = base.n_seq;
        const ctx_tgt = p.ctx_tgt;
        const ctx_dft = p.ctx_dft;
        assertC(ctx_tgt != null and ctx_dft != null, 1372, "ctx_tgt && ctx_dft && \"MTP requires ctx_tgt and ctx_dft to be set\"");

        const n_embd = c.llama_model_n_embd_out(c.llama_get_model(ctx_dft));
        assertC(n_embd == c.llama_model_n_embd_out(c.llama_get_model(ctx_tgt)), 1375, "n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) && \"MTP input row width must match the target h_nextn width\"");
        const n_mtp_layers: i32 = @max(1, c.llama_model_n_layer_nextn(c.llama_get_model(ctx_dft)));

        spcTrc(func, "%s", .{@as([*:0]const u8, "adding speculative implementation 'draft-mtp'\n")});
        spcTrc(func, "- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", .{ p.n_max, p.n_min, @as(f64, p.p_min), n_embd, @as(c_int, @intFromBool(p.backend_sampling)) });
        const devs = S.devicesStr(p.devices.items);
        defer gpa.free(devs);
        spcTrc(func, "- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n", .{
            p.n_gpu_layers,
            c.ggml_type_name(p.cache_type_k),
            c.ggml_type_name(p.cache_type_v),
            @as([*:0]const u8, if (ctx_tgt != null) "yes" else "no"),
            @as([*:0]const u8, if (ctx_dft != null) "yes" else "no"),
            devs.ptr,
        });

        const n_b: i32 = @intCast(c.llama_n_batch(ctx_dft));
        const batch = tokenAndEmbdBatch(n_b, n_embd);

        const smpls = gpa.alloc(*sampling.Sampler, n_seq) catch oom();
        for (smpls) |*s| s.* = draftSampler(c.llama_get_model(ctx_dft));

        // offload draft sampling to the backend
        const chains = backendChains(ctx_dft, n_seq, p.backend_sampling);

        ext.setEmbeddingsNextn(ctx_tgt, true, false);
        ext.setEmbeddingsNextn(ctx_dft, true, true);

        const is_mem_shared = ext.getCtxOther(ctx_dft) == ctx_tgt;
        const chain_heads = n_mtp_layers > 1 and !is_mem_shared;

        const chain_h = gpa.alloc(std.ArrayList(f32), if (chain_heads) n_seq else 0) catch oom();
        if (chain_heads) {
            p.n_max = @min(p.n_max, n_mtp_layers);

            for (chain_h) |*ch| {
                ch.* = .empty;
                ch.ensureTotalCapacity(gpa, @as(usize, @intCast(p.n_max + 1)) * @as(usize, @intCast(n_embd))) catch oom();
            }
        }
        base.n_max = p.n_max;

        const pending_h = gpa.alloc([]f32, n_seq) catch oom();
        for (pending_h) |*h| {
            h.* = gpa.alloc(f32, @intCast(n_embd)) catch oom();
            @memset(h.*, 0.0);
        }

        const i_last = gpa.alloc(c_int, n_seq) catch oom();
        @memset(i_last, -1);
        const i_batch_beg = gpa.alloc(i32, n_seq) catch oom();
        @memset(i_batch_beg, -1);
        const i_batch_end = gpa.alloc(i32, n_seq) catch oom();
        @memset(i_batch_end, -1);

        const verify_h = gpa.alloc(std.ArrayList(f32), n_seq) catch oom();
        @memset(verify_h, .empty);
        const verify_h_rows = gpa.alloc(i32, n_seq) catch oom();
        @memset(verify_h_rows, 0);

        return .{
            .params = p,
            .batch = batch,
            .smpls = smpls,
            .backend_chains = chains,
            .n_embd = n_embd,
            .n_mtp_layers = n_mtp_layers,
            .is_mem_shared = is_mem_shared,
            .chain_heads = chain_heads,
            .pending_h = pending_h,
            .i_batch_beg = i_batch_beg,
            .i_batch_end = i_batch_end,
            .verify_h = verify_h,
            .verify_h_rows = verify_h_rows,
            .i_last = i_last,
            .chain_h = chain_h,
        };
    }

    /// Ports the destructor of `common_speculative_impl_draft_mtp` (speculative.cpp:1446 @b29c606e2).
    pub fn deinit(self: *Mtp) void {
        freeBackendChains(self.params.ctx_dft, self.backend_chains);
        freeTokenAndEmbdBatch(&self.batch);
        freeSamplers(self.smpls);
        for (self.pending_h) |h| gpa.free(h);
        gpa.free(self.pending_h);
        gpa.free(self.i_batch_beg);
        gpa.free(self.i_batch_end);
        for (self.verify_h) |*h| h.deinit(gpa);
        gpa.free(self.verify_h);
        gpa.free(self.verify_h_rows);
        gpa.free(self.i_last);
        for (self.chain_h) |*h| h.deinit(gpa);
        gpa.free(self.chain_h);
    }

    /// Ports `begin` (speculative.cpp:1466 @b29c606e2).
    pub fn begin(self: *Mtp, _: *Base, seq_id: c.llama_seq_id, prompt: []const c.llama_token) void {
        const n: i32 = @intCast(prompt.len);
        if (n <= 0) return;

        const pos_max = c.llama_memory_seq_pos_max(c.llama_get_memory(self.params.ctx_dft), seq_id);

        if (pos_max < n - 1 and !self.is_mem_shared) {
            spcWrn("begin", "ctx_dft pos_max=%d < N-1=%d - " ++
                "process() hook may not have run on every prefill ubatch " ++
                "(need_embd / logits=1 on every prompt position?). " ++
                "Drafts may degrade.\n", .{ pos_max, n - 1 });
        }
    }

    /// Ports `process` (speculative.cpp:1484 @b29c606e2): the catch-up
    /// decode of the target's batch under every head, the hidden states
    /// shifted one position right, and the rows the next `accept` chooses
    /// from.
    pub fn process(self: *Mtp, base: *Base, batch_in: c.llama_batch) bool {
        if (batch_in.n_tokens <= 0) return true;

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == null or batch_in.embd != null) return true;

        const n_seq = base.n_seq;
        const n_tokens: usize = @intCast(batch_in.n_tokens);
        const n_embd: usize = @intCast(self.n_embd);

        // remember the first and last batch index for each sequence
        @memset(self.i_batch_beg, -1);
        @memset(self.i_batch_end, -1);

        for (0..n_tokens) |k| {
            for (0..n_seq) |s| {
                assertC(batch_in.n_seq_id[k] == 1, 1502, "batch_in.n_seq_id[k] == 1");

                if (batch_in.seq_id[k][0] == @as(c.llama_seq_id, @intCast(s))) {
                    self.i_batch_end[s] = @intCast(k);
                    if (self.i_batch_beg[s] < 0) self.i_batch_beg[s] = @intCast(k);
                }
            }
        }

        const ctx_tgt = self.params.ctx_tgt;
        const ctx_dft = self.params.ctx_dft;

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!self.is_mem_shared) {
            common.batchClear(&self.batch);

            for (0..n_tokens) |k| {
                common.batchAdd(&self.batch, batch_in.token[k], batch_in.pos[k], &.{batch_in.seq_id[k][0]}, false);
            }

            // shift the tgt embeddings to the right by one position
            // assumes that the tokens in the batch are sequential for each sequence
            {
                const h_tgt = ext.getEmbeddingsNextn(ctx_tgt);
                if (n_tokens > 1) rowCopy(self.batch.embd + 1 * n_embd, h_tgt.?, n_embd * (n_tokens - 1));
            }

            // fill the pending embeddings from a previous run
            for (0..n_seq) |s| {
                if (self.i_batch_beg[s] < 0) continue;
                rowCopy(self.batch.embd + @as(usize, @intCast(self.i_batch_beg[s])) * n_embd, self.pending_h[s].ptr, n_embd);
            }

            const mem_dft = c.llama_get_memory(ctx_dft);

            var ok = true;
            var head: i32 = 0;
            while (head < self.n_mtp_layers) : (head += 1) {
                if (self.chain_heads) {
                    for (0..n_seq) |s| {
                        if (self.i_batch_beg[s] < 0) continue;
                        _ = c.llama_memory_seq_rm(mem_dft, @intCast(s), batch_in.pos[@intCast(self.i_batch_beg[s])], -1);
                    }
                    ext.setNextnLayerOffset(ctx_dft, head);
                }

                const rc = c.llama_decode(ctx_dft, self.batch);
                if (rc != 0) {
                    spcErr("process", "llama_decode(ctx_dft) head=%d failed rc=%d (pos=%d)\n", .{ head, rc, batch_in.pos[0] });
                    ok = false;
                    break;
                }
            }

            if (self.chain_heads) ext.setNextnLayerOffset(ctx_dft, 0); // restore default for non-draft decodes
            if (!ok) return false;
        }

        for (0..n_seq) |s| {
            if (self.i_batch_end[s] < 0) continue;

            const n_rows: usize = @intCast(self.i_batch_end[s] - self.i_batch_beg[s] + 1);
            self.verify_h_rows[s] = @intCast(n_rows);
            self.verify_h[s].resize(gpa, n_rows * n_embd) catch oom();

            for (0..n_rows) |i| {
                const h = ext.getEmbeddingsNextnIth(ctx_tgt, self.i_batch_beg[s] + @as(i32, @intCast(i)));
                rowCopy(self.verify_h[s].items.ptr + i * n_embd, h.?, n_embd);
            }

            rowCopy(self.pending_h[s].ptr, self.verify_h[s].items.ptr + (n_rows - 1) * n_embd, n_embd);
        }

        return true;
    }

    /// Ports `draft` (speculative.cpp:1602 @b29c606e2).
    pub fn draft(self: *Mtp, base: *Base, dparams: []DraftParams) void {
        const ctx_dft = self.params.ctx_dft;
        const n_seq = base.n_seq;
        const n_embd: usize = @intCast(self.n_embd);

        common.batchClear(&self.batch);

        // keep track of which sequences are still drafting
        var n_drafting: c_int = 0;
        const drafting = gpa.alloc(bool, n_seq) catch oom();
        defer gpa.free(drafting);
        @memset(drafting, false);

        for (0..n_seq) |s| {
            const seq_id: c.llama_seq_id = @intCast(s);
            const dp = &dparams[s];

            if (!dp.drafting) continue;

            n_drafting += 1;
            drafting[s] = true;
            self.smpls[s].reset();

            common.batchAdd(&self.batch, dp.id_last, dp.pos0, &.{seq_id}, true);
            rowCopy(self.batch.embd + @as(usize, @intCast(self.batch.n_tokens - 1)) * n_embd, self.pending_h[s].ptr, n_embd);

            self.i_last[s] = self.batch.n_tokens - 1;

            if (self.chain_heads) {
                self.chain_h[s].clearRetainingCapacity();
                self.chain_h[s].appendSlice(gpa, self.pending_h[s]) catch oom();
            }
        }

        var i: c_int = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. reset the draft region and select head i so it rebuilds its
            // own layer's KV there.
            if (self.chain_heads) {
                const mem_dft = c.llama_get_memory(ctx_dft);
                for (0..n_seq) |s| {
                    if (drafting[s]) _ = c.llama_memory_seq_rm(mem_dft, @intCast(s), dparams[s].pos0, -1);
                }
                ext.setNextnLayerOffset(ctx_dft, i);
            }

            const ret = c.llama_decode(ctx_dft, self.batch);
            if (ret != 0) {
                spcErr("draft", "llama_decode[%d] returned %d\n", .{ i, ret });
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            common.batchClear(&self.batch);

            for (0..n_seq) |s| {
                const seq_id: c.llama_seq_id = @intCast(s);
                if (!drafting[s]) continue;

                const smpl = self.smpls[s];

                _ = smpl.sample(ctx_dft, self.i_last[s], true) catch oom();
                const h_row = ext.getEmbeddingsNextnIth(ctx_dft, self.i_last[s]);

                const cur_p = smpl.getCandidates(true);

                logCandidates("draft", ctx_dft, seq_id, i, cur_p);

                // add drafted token for each sequence
                const id = cur_p.data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p.data[0].p < self.params.p_min) {
                    drafting[s] = false;
                    n_drafting -= 1;
                    continue;
                }

                smpl.accept(id, true);

                const dp = &dparams[s];
                const result = dp.result.?;

                result.append(gpa, id) catch oom();

                if (self.params.n_max <= @as(i32, @intCast(result.items.len))) {
                    drafting[s] = false;
                    n_drafting -= 1;
                    continue;
                }

                if (self.chain_heads) {
                    self.chain_h[s].appendSlice(gpa, h_row.?[0..n_embd]) catch oom();

                    const n_rows: usize = result.items.len + 1; // id_last + tokens drafted so far
                    for (0..n_rows) |t| {
                        const tok = if (t == 0) dp.id_last else result.items[t - 1];
                        common.batchAdd(&self.batch, tok, dp.pos0 + @as(c.llama_pos, @intCast(t)), &.{seq_id}, t == n_rows - 1);
                        rowCopy(self.batch.embd + @as(usize, @intCast(self.batch.n_tokens - 1)) * n_embd, self.chain_h[s].items.ptr + t * n_embd, n_embd);
                    }
                } else if (self.is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    common.batchAdd(&self.batch, id, dp.pos0, &.{seq_id}, true);
                    rowCopy(self.batch.embd + @as(usize, @intCast(self.batch.n_tokens - 1)) * n_embd, h_row.?, n_embd);
                } else {
                    common.batchAdd(&self.batch, id, dp.pos0 + i + 1, &.{seq_id}, true);
                    rowCopy(self.batch.embd + @as(usize, @intCast(self.batch.n_tokens - 1)) * n_embd, h_row.?, n_embd);
                }

                self.i_last[s] = self.batch.n_tokens - 1;
            }

            if (self.batch.n_tokens == 0) break;

            i += 1;
        }

        if (self.chain_heads) ext.setNextnLayerOffset(ctx_dft, 0); // restore default for non-draft decodes

        for (0..n_seq) |s| {
            const dp = &dparams[s];
            if (!dp.drafting) continue;

            if (dp.result.?.items.len < @as(usize, @bitCast(@as(isize, self.params.n_min)))) dp.result.?.clearRetainingCapacity();
        }
    }

    /// Ports `accept` (speculative.cpp:1753 @b29c606e2): the hidden row of
    /// the first position the target did not accept becomes pending.
    pub fn accept(self: *Mtp, base: *Base, seq_id: c.llama_seq_id, n_accepted: u16, _: bool) void {
        if (seq_id < 0 or seq_id >= @as(c.llama_seq_id, @intCast(base.n_seq))) return;
        const s: usize = @intCast(seq_id);

        const n_rows = self.verify_h_rows[s];
        if (n_rows <= 0) return;

        const i_h: i32 = @min(@as(i32, n_accepted), n_rows - 1);
        const n_embd: usize = @intCast(self.n_embd);
        rowCopy(self.pending_h[s].ptr, self.verify_h[s].items.ptr + @as(usize, @intCast(i_h)) * n_embd, n_embd);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
