//! The speculative decoders that run a second model: a plain draft model,
//! EAGLE3, DFlash and DSpark, and multi-token prediction.
//!
//! # Provenance
//!
//! Ports `common_speculative_impl_draft_simple`,
//! `common_speculative_impl_draft_eagle3`,
//! `common_speculative_impl_draft_dflash` and
//! `common_speculative_impl_draft_mtp` from
//! `llama.cpp/common/speculative.cpp` (v0.9.0, `cd959bae9`), with
//! `spec_retune`. Each declaration names what it ports and the line it
//! began at; the rest of the file is `speculative.zig`.
//!
//! # Batches
//!
//! Every batch is a `common.Batch` over the draft context. An embedding set
//! on an entry is a *view*, read when the batch is rendered: EAGLE3's bridge
//! row points at `pending_g_last`, which the same `process` then overwrites
//! before decoding, and the decoder reads the new row -- as upstream's does.

const std = @import("std");
const c = @import("tools_h");
const S = @import("speculative.zig");
const params_mod = @import("params.zig");
const sampling = @import("sampling.zig");
const libcxx = @import("llama").libcxx;
const common = @import("common.zig");
const log = @import("log.zig");

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

/// What `spec_retune` remembers of a draft sampler's parameters: the C++
/// keeps a whole `common_params_sampling` and reads these two.
pub const SmplCfg = struct { temp: f32, seed: u32 };

/// Ports `spec_retune` (speculative.cpp:37 @cd959bae9): rebuilds
/// `seq_id`'s draft sampler at the target's temperature, since rejection
/// weighs q against p and both have to sample alike. Only temp and seed
/// carry over; the draft keeps its own top_k.
fn specRetune(smpls: []*sampling.Sampler, cfg: *std.ArrayList(SmplCfg), model: ?*const c.llama_model, seq_id: c.llama_seq_id, temp: f32, seed: u32) void {
    if (cfg.items.len != smpls.len) {
        const n_old = cfg.items.len;
        // `resize` default-constructs; the temperature is what is read
        cfg.resize(gpa, smpls.len) catch oom();

        // the initial sampler has no temperature, so no request may match the cache and skip a rebuild
        for (cfg.items[n_old..]) |*e| e.* = .{ .temp = std.math.nan(f32), .seed = c.LLAMA_DEFAULT_SEED };
    }

    const cur = &cfg.items[@intCast(seq_id)];

    if (cur.temp == temp and cur.seed == seed) return;

    cur.temp = temp;
    cur.seed = seed;

    var sparams: params_mod.Sampling = .{};
    sparams.no_perf = false;
    sparams.top_k = 10;
    sparams.temp = cur.temp;
    // must be explicit, the default reseeds at random; mixed so it differs from the target's
    sparams.seed = if (cur.seed == c.LLAMA_DEFAULT_SEED) cur.seed else cur.seed ^ 0x85ebca6b;
    var types = [_]params_mod.SamplerType{ .top_k, .temperature };
    // `sampling.init` reads the list and keeps nothing of it
    sparams.samplers = .fromOwnedSlice(&types);

    const smpl = sampling.init(model, &sparams) catch |e| switch (e) {
        error.OutOfMemory => oom(),
        // top-k and a temperature over default penalties cannot throw
        else => unreachable,
    };
    smpls[@intCast(seq_id)].destroy();
    smpls[@intCast(seq_id)] = smpl;
}

/// The candidates `cur_p` holds, appended to `q` as one list -- the C++'s
/// `emplace_back(cur_p->data, cur_p->data + cur_p->size)`.
fn appendCandidates(q: *S.Candidates, cur_p: *const c.llama_token_data_array) void {
    var e: std.ArrayList(c.llama_token_data) = .empty;
    e.appendSlice(gpa, cur_p.data[0..cur_p.size]) catch oom();
    q.append(gpa, e) catch oom();
}

/// `{ data, n_rows, n_embd }`.
fn embdView(data: [*]const f32, n_rows: usize, n_embd: usize) c.llama_embd {
    return .{ .data = data, .n_rows = n_rows, .n_embd = n_embd };
}

// -----------------------------------------------------------------------------
// draft-simple

/// Ports `common_speculative_impl_draft_simple` (speculative.cpp:220
/// @cd959bae9): a standalone draft model, decoding its own guesses.
pub const DraftSimple = struct {
    params: params_mod.SpeculativeDraft,
    batch: common.Batch,

    // zero row at the draft input width, stands in for target embeddings the draft cannot read
    zeros: []f32,
    zeros_warned: bool = false, // the substitution is reported once

    smpls: []*sampling.Sampler,

    smpls_cfg: std.ArrayList(SmplCfg) = .empty,

    /// Ports `common_speculative_impl_draft_simple` (speculative.cpp:233
    /// @cd959bae9), the constructor.
    pub fn init(base: *Base, params: *const params_mod.Speculative) S.InitError!DraftSimple {
        const func = "common_speculative_impl_draft_simple";
        const p = params.draft;
        const ctx_dft = p.ctx_dft;
        const ctx_tgt = p.ctx_tgt;

        if (ctx_dft == null) return S.throwRuntime("draft-simple requires a draft context");

        const zeros = gpa.alloc(f32, @intCast(c.llama_model_n_embd_inp(c.llama_get_model(ctx_dft)))) catch oom();
        @memset(zeros, 0.0);

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

        const batch = common.Batch.init(gpa, ctx_dft);

        const smpls = gpa.alloc(*sampling.Sampler, base.n_seq) catch oom();
        for (smpls) |*s| s.* = draftSampler(c.llama_get_model(ctx_dft));

        var self: DraftSimple = .{ .params = p, .batch = batch, .zeros = zeros, .smpls = smpls };

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

    /// The members' destructors: upstream's is the default since v0.6.0.
    pub fn deinit(self: *DraftSimple) void {
        self.batch.deinit();
        gpa.free(self.zeros);
        freeSamplers(self.smpls);
        self.smpls_cfg.deinit(gpa);
    }

    /// Ports `begin` (speculative.cpp:301 @cd959bae9): the sequence's
    /// sampler is reset here rather than per round, or two identical
    /// requests differ.
    pub fn begin(self: *DraftSimple, _: *Base, seq_id: c.llama_seq_id, _: []const c.llama_token) void {
        self.smpls[@intCast(seq_id)].reset();
    }

    /// Ports `process` (speculative.cpp:306 @cd959bae9): the target's batch
    /// copied to one the draft context owns, only the last entry an output;
    /// an embedding entry the draft cannot read becomes a zero row.
    pub fn process(self: *DraftSimple, _: *Base, batch_in: *const common.Batch) bool {
        const ctx_dft = self.params.ctx_dft;

        // copy the entries to a batch owned by the draft context, only the last token is output
        self.batch.clear();
        const n_tokens = batch_in.size();
        var k: i32 = 0;
        while (k < n_tokens) : (k += 1) {
            const t = &batch_in.tokens.items[@intCast(k)];
            const output = k == n_tokens - 1;
            if (t.id != c.LLAMA_TOKEN_NULL) {
                const idx = self.batch.add(t.id, t.pos[0], t.seq_id, output);
                if (t.embd.data != null) {
                    _ = self.batch.setEmbd(idx, t.embd);
                }
            } else {
                // mtmd input is projected by the target encoder, a draft with a different width cannot read it
                // it gets zeros instead, keeping its positions contiguous
                // ref: https://github.com/ggml-org/llama.cpp/pull/29385#discussion_r4124743243
                const n_embd = t.embd.n_rows * t.embd.n_embd;
                const same_width = n_embd == self.zeros.len;
                if (!same_width and !self.zeros_warned) {
                    spcWrn("process", "target embeddings of size %zu do not fit the draft input width %zu, " ++
                        "the draft receives zero rows for them and drafts after multimodal input will be poor\n", .{ n_embd, self.zeros.len });
                    self.zeros_warned = true;
                }
                const embd = if (same_width) t.embd else embdView(self.zeros.ptr, 1, self.zeros.len);
                _ = self.batch.addEmbd(embd, &t.pos, t.seq_id, output);
            }
        }

        if (self.batch.size() == 0) return true;

        const ret = c.llama_process(ctx_dft, c.LLAMA_PROCESS_TYPE_DECODE, self.batch.get());

        if (ret != 0) {
            spcErr("process", "failed to decode draft batch, ret = %d\n", .{ret});
            return false;
        }

        return true;
    }

    /// Ports `draft` (speculative.cpp:352 @cd959bae9): every drafting
    /// sequence decodes `id_last`, then its own guess -- the best one, or a
    /// sample with its candidates kept when the caller verifies by
    /// rejection -- until it runs out of confidence or room.
    pub fn draft(self: *DraftSimple, base: *Base, dparams: []DraftParams) void {
        const ctx_dft = self.params.ctx_dft;
        const n_seq = base.n_seq;

        self.batch.clear();

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
            // greedy drafting leaves no candidates behind, so the verifier falls back to sample-and-match
            if (!self.params.probabilistic) {
                dp.result_q = null;
            }

            // result_q is only set when the caller wants rejection, so it also gates the retune
            if (dp.result_q != null) {
                specRetune(self.smpls, &self.smpls_cfg, c.llama_get_model(ctx_dft), seq_id, dp.temp, dp.seed);
            }

            // a reset reseeds the chain, which breaks probabilistic drafting
            if (dp.result_q == null) {
                self.smpls[s].reset();
            }

            _ = self.batch.add(dp.id_last, dp.pos0, seq_id, true);
        }

        var ret = c.llama_process(ctx_dft, c.LLAMA_PROCESS_TYPE_DECODE, self.batch.get());
        if (ret != 0) {
            spcErr("draft", "llama_process returned %d\n", .{ret});
            return;
        }

        var i: c_int = 0;

        while (n_drafting > 0) {
            var i_batch: c_int = 0;

            self.batch.clear();

            for (0..n_seq) |s| {
                const seq_id: c.llama_seq_id = @intCast(s);
                if (!drafting[s]) continue;

                const smpl = self.smpls[s];

                const id_sampled = smpl.sample(ctx_dft, i_batch, true) catch oom();
                i_batch += 1;

                const cur_p = smpl.getCandidates(true);

                logCandidates("draft", ctx_dft, seq_id, i, cur_p);

                // add drafted token for each sequence
                const id = if (dparams[s].result_q != null) id_sampled else cur_p.data[0].id;

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

                if (dp.result_q) |q| appendCandidates(q, cur_p);

                if (self.params.n_max <= @as(i32, @intCast(result.items.len)) or
                    (dp.n_max > 0 and dp.n_max <= @as(i32, @intCast(result.items.len))))
                {
                    drafting[s] = false;
                    n_drafting -= 1;
                    continue;
                }

                _ = self.batch.add(id, dp.pos0 + i + 1, seq_id, true);
            }

            if (self.batch.size() == 0) break;

            // evaluate the drafted tokens on the draft model
            ret = c.llama_process(ctx_dft, c.LLAMA_PROCESS_TYPE_DECODE, self.batch.get());
            if (ret != 0) {
                spcErr("draft", "llama_process[%d] returned %d\n", .{ i, ret });
                break;
            }

            i += 1;
        }

        for (dparams) |*dp| {
            if (!dp.drafting) continue;

            if (dp.result.?.items.len < @as(usize, @bitCast(@as(isize, self.params.n_min)))) dp.result.?.clearRetainingCapacity();
        }
    }

    /// Ports `accept` (speculative.cpp:476 @cd959bae9): nothing to do.
    pub fn accept(_: *DraftSimple, _: *Base, _: c.llama_seq_id, _: u16, _: bool) void {
        // noop
    }
};

// -----------------------------------------------------------------------------
// draft-eagle3

/// Ports `common_speculative_impl_draft_eagle3` (speculative.cpp:516
/// @cd959bae9). See the C++'s comment for the input convention and the
/// deferred boundary: at memory position P the decoder reads the pair
/// (token[P+1], g[P]), so the last position of every `process` waits for the
/// next call -- or `draft`'s seed step -- to bring its token.
pub const Eagle3 = struct {
    params: params_mod.SpeculativeDraft,
    batch: common.Batch, // decoder input, (token, g_embd) pairs
    batch_enc: common.Batch, // encoder input, built from the extracted target features

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

    /// Ports `common_speculative_impl_draft_eagle3` (speculative.cpp:547
    /// @cd959bae9), the constructor.
    pub fn init(base: *Base, params: *const params_mod.Speculative) S.InitError!Eagle3 {
        const func = "common_speculative_impl_draft_eagle3";
        const n_seq = base.n_seq;
        spcTrc(func, "%s", .{@as([*:0]const u8, "adding speculative implementation 'draft-eagle3'\n")});
        spcTrc(func, "- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", .{ params.draft.n_max, params.draft.n_min, @as(f64, params.draft.p_min), @as(c_int, @intFromBool(params.draft.backend_sampling)) });

        const p = params.draft;
        const ctx_tgt = p.ctx_tgt;
        const ctx_dft = p.ctx_dft;
        assertC(ctx_tgt != null and ctx_dft != null, 556, "ctx_tgt && ctx_dft && \"EAGLE3 requires ctx_tgt and ctx_dft to be set\"");

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

        const batch = common.Batch.init(gpa, ctx_dft);
        const batch_enc = common.Batch.init(gpa, ctx_dft);

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
                _ = c.ggml_abort("speculative.cpp", 608, "EAGLE3: target layer id %d exceeds target n_layer %d", lid, n_layer_tgt);
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
            .batch_enc = batch_enc,
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

    /// Ports the destructor of `common_speculative_impl_draft_eagle3` (speculative.cpp:624 @cd959bae9).
    pub fn deinit(self: *Eagle3) void {
        freeBackendChains(self.params.ctx_dft, self.backend_chains);
        self.batch.deinit();
        self.batch_enc.deinit();
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

    /// Ports `begin` (speculative.cpp:638 @cd959bae9): warns when the
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

    /// Ports `process` (speculative.cpp:654 @cd959bae9): the target's
    /// features through the encoder, then the decoder over this batch's
    /// pairs, the last position deferred.
    pub fn process(self: *Eagle3, base: *Base, batch_in: *const common.Batch) bool {
        if (batch_in.size() <= 0) return true;

        if (!batch_in.hasToken() or batch_in.hasEmbd()) return true;

        const n_seq = base.n_seq;
        const n_tokens: usize = @intCast(batch_in.size());
        const tok = batch_in.tokens.items;
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
            const seq_id = tok[k].seq_id;
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
                _ = c.ggml_abort("speculative.cpp", 694, "EAGLE3: target layer %d input not extracted.", lid);
            }
            for (0..n_tokens) |i| {
                const dst = self.features_buf.items.ptr + i * n_embd_enc + k * n_embd_tgt;
                const src = layer.? + i * n_embd_tgt;
                rowCopy(dst, src, n_embd_tgt);
            }
        }

        self.g_embd_buf.resize(gpa, n_tokens * n_embd_dec) catch oom();

        // llama_process() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const n_ubatch_dft: usize = @intCast(c.llama_n_ubatch(ctx_dft));
        var i: usize = 0;
        while (i < n_tokens) : (i += n_ubatch_dft) {
            const n_chunk = @min(n_ubatch_dft, n_tokens - i);

            // the per-token encoder does not use positions, generate placeholder ones from the memory state
            self.batch_enc.clear();
            var pos: c.llama_pos = c.llama_memory_seq_pos_max(c.llama_get_memory(ctx_dft), 0) + 1;
            for (0..n_chunk) |j| {
                // the C++ passes `&pos`, from which `add_embd` reads `n_pos`
                // positions: one, the draft being no M-RoPE model
                const pos_arr = [_]c.llama_pos{ pos, 0, 0, 0 };
                _ = self.batch_enc.addEmbd(embdView(self.features_buf.items.ptr + (i + j) * n_embd_enc, 1, n_embd_enc), &pos_arr, 0, true);
                pos += 1;
            }

            const rc = c.llama_process(ctx_dft, c.LLAMA_PROCESS_TYPE_ENCODE, self.batch_enc.get());
            if (rc != 0) {
                spcErr("process", "llama_process(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n", .{ rc, @as(c_int, @intCast(n_chunk)), @as(c_int, @intCast(i)) });
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const g_embd_chunk = ext.getEmbeddingsNextn(ctx_dft);
            assertC(g_embd_chunk != null, 728, "g_embd_chunk && \"EAGLE3 encoder produced no output.\"");
            rowCopy(self.g_embd_buf.items.ptr + i * n_embd_dec, g_embd_chunk.?, n_chunk * n_embd_dec);
        }

        const g_embd = self.g_embd_buf.items.ptr;

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        self.batch.clear();

        for (0..n_seq) |s| {
            const seq_id: c.llama_seq_id = @intCast(s);
            const beg = i_batch_beg[s];
            const end = i_batch_end[s];
            if (beg < 0 or end < 0) continue;
            const ub: usize = @intCast(beg);
            const ue: usize = @intCast(end);

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            const pending_pos = self.pending_pos_last[s];
            if (pending_pos >= 0 and pending_pos + 1 == tok[ub].pos[0]) {
                const dft_pos_max = c.llama_memory_seq_pos_max(c.llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    const idx = self.batch.add(tok[ub].id, pending_pos, seq_id, false);
                    _ = self.batch.setEmbd(idx, embdView(self.pending_g_last[s].ptr, 1, n_embd_dec));
                }
            }

            var k = ub;
            while (k < ue) : (k += 1) {
                const idx = self.batch.add(tok[k + 1].id, tok[k].pos[0], seq_id, false);
                _ = self.batch.setEmbd(idx, embdView(g_embd + k * n_embd_dec, 1, n_embd_dec));
            }

            // refresh deferred state
            const n_rows: usize = ue - ub + 1;
            self.verify_pos_first[s] = tok[ub].pos[0];
            self.pending_pos_last[s] = tok[ue].pos[0];
            self.verify_g_rows[s] = @intCast(n_rows);
            const vg = &self.verify_g[s];
            const vold = vg.items.len;
            vg.resize(gpa, n_rows * n_embd_dec) catch oom();
            if (vg.items.len > vold) @memset(vg.items[vold..], 0.0);
            rowCopy(vg.items.ptr, g_embd + ub * n_embd_dec, n_embd_dec * n_rows);
            rowCopy(self.pending_g_last[s].ptr, g_embd + ue * n_embd_dec, n_embd_dec);
        }

        if (self.batch.size() > 0) {
            const rc = c.llama_process(ctx_dft, c.LLAMA_PROCESS_TYPE_DECODE, self.batch.get());
            if (rc != 0) {
                spcErr("process", "llama_process(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n", .{ rc, self.batch.size(), tok[0].pos[0] });
                return false;
            }
        }

        return true;
    }

    /// Ports `draft` (speculative.cpp:799 @cd959bae9): the seed step
    /// completes the deferred pair with `id_last`, then each step feeds back
    /// its own pre-norm state.
    pub fn draft(self: *Eagle3, base: *Base, dparams: []DraftParams) void {
        const ctx_dft = self.params.ctx_dft;
        const n_seq = base.n_seq;
        const n_embd_dec: usize = @intCast(self.n_embd_dec);

        self.batch.clear();

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

            const idx = self.batch.add(dp.id_last, self.pending_pos_last[s], seq_id, true);
            _ = self.batch.setEmbd(idx, embdView(self.pending_g_last[s].ptr, 1, n_embd_dec));
        }

        if (self.batch.size() == 0) return;

        var ret = c.llama_process(ctx_dft, c.LLAMA_PROCESS_TYPE_DECODE, self.batch.get());
        if (ret != 0) {
            spcErr("draft", "llama_process returned %d\n", .{ret});
            return;
        }

        var i: c_int = 0;

        while (n_drafting > 0) {
            var i_batch: c_int = 0;

            self.batch.clear();

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

                const idx = self.batch.add(id, self.pending_pos_last[s] + (i + 1), seq_id, true);
                _ = self.batch.setEmbd(idx, embdView(prenorm.?, 1, n_embd_dec));
            }

            if (self.batch.size() == 0) break;

            ret = c.llama_process(ctx_dft, c.LLAMA_PROCESS_TYPE_DECODE, self.batch.get());
            if (ret != 0) {
                spcErr("draft", "llama_process[%d] returned %d\n", .{ i, ret });
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

    /// Ports `accept` (speculative.cpp:922 @cd959bae9): the deferred
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

    /// Ports `need_boundary_stash` (speculative.cpp:941 @cd959bae9): only a
    /// recurrent or hybrid target drops the deferred row on restore.
    fn needBoundaryStash(self: *const Eagle3) bool {
        const model_tgt = c.llama_get_model(self.params.ctx_tgt);
        return c.llama_model_is_recurrent(model_tgt) or c.llama_model_is_hybrid(model_tgt);
    }

    /// Ports `get_state` (speculative.cpp:946 @cd959bae9): the deferred
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

    /// Ports `set_state` (speculative.cpp:963 @cd959bae9).
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

/// `std::random_device{}()`, as `src/mtmd/clip.zig`'s `randomDevice`:
/// libc++ reads `arc4random` on Apple platforms, `rand_s` on Windows and
/// `getentropy` elsewhere; either way the seed is not reproducible.
fn randomDevice() u32 {
    var v: u32 = 0;
    switch (comptime @import("builtin").os.tag) {
        .macos, .ios, .tvos, .watchos, .visionos => std.c.arc4random_buf(@ptrCast(&v), @sizeOf(u32)),
        .windows => {
            const rand_s = @extern(*const fn (*c_uint) callconv(.c) c_int, .{ .name = "rand_s" });
            var r: c_uint = 0;
            _ = rand_s(&r);
            v = r;
        },
        else => {
            const getentropy = @extern(*const fn (*anyopaque, usize) callconv(.c) c_int, .{ .name = "getentropy" });
            _ = getentropy(@ptrCast(&v), @sizeOf(u32));
        },
    }
    return v;
}

/// Ports `common_speculative_impl_draft_dflash` (speculative.cpp:984
/// @cd959bae9): block-diffusion drafting with the target's features
/// injected into the draft's KV cache. DSpark is the same driver with a
/// Markov head, an anchor-first block and an optional confidence head;
/// LiliCorr (`lilicorr.candidate_topk` in the draft's metadata) walks a
/// candidate lattice as DFlash2 does, its head reading the draft's own
/// projected embeddings, kept per sequence, as a state input.
pub const DFlash = struct {
    params: params_mod.SpeculativeDraft,

    batch: common.Batch, // noise tokens
    batch_inject: common.Batch, // target features for KV cache injection

    features_buf: std.ArrayList(f32) = .empty, // [n_chunk, n_embd_enc] gathered target features

    smpls: []*sampling.Sampler,

    // backend sampler chain per seq, attached to ctx_dft
    backend_chains: []?*c.llama_sampler,

    n_embd_dec: i32 = 0, // draft hidden size
    n_embd_enc: i32 = 0, // target_layer_ids_n * target_hidden_size
    n_embd_tgt: i32 = 0, // target model hidden size

    block_size: i32 = 0,
    mask_token_id: c.llama_token = 0,

    is_lilicorr: bool = false,
    lilicorr_top_k: i32 = 0,
    lilicorr_rows: usize = 0,
    lilicorr_embd: []f32 = &.{},
    lilicorr_pos: []c.llama_pos = &.{},

    is_dflash2: bool = false,
    is_mrope: bool = false,
    selector_top_k: i32 = 0,

    // DFlash2 probabilistic drafting: one generator per sequence and the seed it was last seeded with (-1: reseed)
    dflash2_rng: []libcxx.Mt19937 = &.{},
    dflash2_seed: []i64 = &.{},

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    is_dspark: bool,

    // dspark speculators
    sample_from_anchor: bool = true,

    // block-internal attention
    causal_attn: bool = false,

    target_layer_ids: [*c]const i32 = null, // model_dft's extract layer indices
    target_layer_ids_n: u32 = 0,

    // inject target features from the target's device buffer instead of host copies (env LLAMA_DFLASH_DEV=0 disables)
    use_dev_features: bool = false,

    /// Ports `common_speculative_impl_draft_dflash` (speculative.cpp:1033
    /// @cd959bae9), the constructor.
    pub fn init(base: *Base, params: *const params_mod.Speculative, kind: params_mod.SpeculativeType) S.InitError!DFlash {
        const func = "common_speculative_impl_draft_dflash";
        var p = params.draft;
        const n_seq = base.n_seq;
        const is_dspark = kind == .draft_dspark;
        const ctx_tgt = p.ctx_tgt;
        const ctx_dft = p.ctx_dft;
        assertC(ctx_tgt != null and ctx_dft != null, 1041, "ctx_tgt && ctx_dft && \"DFlash requires ctx_tgt and ctx_dft to be set\"");

        const model_dft = c.llama_get_model(ctx_dft);
        const model_tgt = c.llama_get_model(ctx_tgt);

        const target_layer_ids = ext.modelTargetLayerIds(model_dft);
        const target_layer_ids_n = ext.modelTargetLayerIdsN(model_dft);
        assertC(target_layer_ids_n > 0, 1048, "target_layer_ids_n > 0 && \"DFlash model has no target_layer_ids\"");

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
        const dflash2_rng = gpa.alloc(libcxx.Mt19937, n_seq) catch oom();
        const dflash2_seed = gpa.alloc(i64, n_seq) catch oom();
        @memset(dflash2_seed, -1);
        var lilicorr_meta = std.mem.zeroes([32]u8);
        const lilicorr_top_k: i32 = if (c.llama_model_meta_val_str(model_dft, "lilicorr.candidate_topk", &lilicorr_meta, lilicorr_meta.len) >= 0)
            c.atoi(@ptrCast(&lilicorr_meta))
        else
            0;
        const is_lilicorr = lilicorr_top_k > 0;
        var lilicorr_rows: usize = 0;
        var lilicorr_embd: []f32 = &.{};
        var lilicorr_pos: []c.llama_pos = &.{};
        if (is_lilicorr) {
            lilicorr_rows = @as(usize, @intCast(block_size)) + 1;
            const lilicorr_slots = n_seq * lilicorr_rows;
            lilicorr_embd = gpa.alloc(f32, lilicorr_slots * @as(usize, @intCast(n_embd_dec))) catch oom();
            @memset(lilicorr_embd, 0.0);
            lilicorr_pos = gpa.alloc(c.llama_pos, lilicorr_slots) catch oom();
            @memset(lilicorr_pos, -1);
        }
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

        const batch = common.Batch.init(gpa, ctx_dft);
        const batch_inject = common.Batch.init(gpa, ctx_dft);

        // embd batches on an M-RoPE draft carry 4 position rows per token
        const is_mrope = c.llama_model_rope_type(model_dft) == c.LLAMA_ROPE_TYPE_MROPE;

        const smpls = gpa.alloc(*sampling.Sampler, n_seq) catch oom();
        for (smpls) |*s| s.* = draftSampler(model_dft);

        // offload draft sampling to the backend
        const chains = backendChains(ctx_dft, n_seq, p.backend_sampling and !is_dflash2 and !is_lilicorr);

        // turn on extraction of the target layers' input embeddings
        for (0..target_layer_ids_n) |k| ext.setEmbeddingsLayerInp(ctx_tgt, @intCast(target_layer_ids[k]), true);

        // the device buffer keeps layers in ascending id order, the fc input expects target_layer_ids order
        var use_dev_features = false;
        {
            const env = c.getenv("LLAMA_DFLASH_DEV");
            const ascending = for (1..target_layer_ids_n) |k| {
                if (target_layer_ids[k - 1] >= target_layer_ids[k]) break false;
            } else true;

            use_dev_features = !is_lilicorr and ascending and (env == null or c.atoi(env) != 0);
            if (use_dev_features) {
                ext.setEmbeddingsLayerInpDev(ctx_tgt, true);
                use_dev_features = ext.supportsEmbdRow(ctx_dft);
                if (!use_dev_features) {
                    ext.setEmbeddingsLayerInpDev(ctx_tgt, false);
                }
            }
            log.inf("%s: DFlash target features via %s\n", .{ @as([*:0]const u8, func), @as([*:0]const u8, if (use_dev_features) "device buffer" else "host copies") });
        }

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        ext.setEmbeddingsNextn(ctx_dft, true, !is_dflash2 and !is_lilicorr);
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
            .is_lilicorr = is_lilicorr,
            .lilicorr_top_k = lilicorr_top_k,
            .lilicorr_rows = lilicorr_rows,
            .lilicorr_embd = lilicorr_embd,
            .lilicorr_pos = lilicorr_pos,
            .is_dflash2 = is_dflash2,
            .dflash2_rng = dflash2_rng,
            .dflash2_seed = dflash2_seed,
            .is_mrope = is_mrope,
            .selector_top_k = selector_top_k,
            .is_dspark = is_dspark,
            .sample_from_anchor = sample_from_anchor,
            .causal_attn = causal_attn,
            .target_layer_ids = target_layer_ids,
            .target_layer_ids_n = target_layer_ids_n,
            .use_dev_features = use_dev_features,
        };
    }

    /// Ports the destructor of `common_speculative_impl_draft_dflash` (speculative.cpp:1169 @cd959bae9).
    pub fn deinit(self: *DFlash) void {
        freeBackendChains(self.params.ctx_dft, self.backend_chains);
        self.batch.deinit();
        self.batch_inject.deinit();
        self.features_buf.deinit(gpa);
        gpa.free(self.lilicorr_embd);
        gpa.free(self.lilicorr_pos);
        gpa.free(self.dflash2_rng);
        gpa.free(self.dflash2_seed);
        freeSamplers(self.smpls);
    }

    /// Ports `begin` (speculative.cpp:1183 @cd959bae9).
    pub fn begin(self: *DFlash, base: *Base, seq_id: c.llama_seq_id, prompt: []const c.llama_token) void {
        if (seq_id < 0 or seq_id >= @as(c.llama_seq_id, @intCast(base.n_seq))) return;

        // reseed per request rather than per round, or two identical requests differ
        self.dflash2_seed[@intCast(seq_id)] = -1;

        const n: i32 = @intCast(prompt.len);
        if (n <= 0) return;

        const pos_max = c.llama_memory_seq_pos_max(c.llama_get_memory(self.params.ctx_dft), seq_id);
        if (pos_max < n - 1) {
            log.wrn("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. " ++
                "Drafts may degrade.\n", .{ @as([*:0]const u8, "begin"), pos_max, n - 1 });
        }
    }

    /// Ports `process` (speculative.cpp:1204 @cd959bae9): each sequence's
    /// target features, chunked by the draft's micro-batch, injected into
    /// the draft's KV cache at the target positions; LiliCorr also keeps the
    /// draft's projected embeddings of the last `block_size + 1` positions.
    pub fn process(self: *DFlash, base: *Base, batch_in: *const common.Batch) bool {
        if (batch_in.size() <= 0) return true;

        // Target prefill may contain token IDs or multimodal embeddings. Both
        // produce the target-layer features used to seed the draft KV cache, so
        // embeddings are injected too, except the pinned ones skipped below.
        // TODO: revisit after https://github.com/ggml-org/llama.cpp/pull/24669 is merged
        const has_tokens = batch_in.hasToken();
        const has_embeddings = batch_in.hasEmbd();
        if (has_tokens == has_embeddings) return true;

        const n_seq = base.n_seq;
        const n_tokens: usize = @intCast(batch_in.size());
        const tok = batch_in.tokens.items;
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
            const seq_id = tok[k].seq_id;
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
            const pos_pinned = tok[ub].pos[0] == tok[@intCast(i_batch_end[s])].pos[0];
            if (has_embeddings and n_rows > 1 and pos_pinned) continue;

            var offset: i32 = 0;
            while (offset < n_rows) : (offset += n_ubatch) {
                const n_chunk: i32 = @min(n_ubatch, n_rows - offset);
                const nc: usize = @intCast(n_chunk);
                const uo: usize = @intCast(offset);

                if (self.use_dev_features) {
                    // the features are the target batch rows in the target's device buffer
                    self.batch_inject.clear();
                    for (0..nc) |i| {
                        const row: i32 = @intCast(ub + uo + i);
                        const p = tok[ub + uo + i].pos[0];
                        const pos_arr = [4]c.llama_pos{ p, p, p, 0 };
                        _ = self.batch_inject.addEmbdRow(row, &pos_arr, seq_id, false);
                    }

                    const rc = c.llama_process(ctx_dft, c.LLAMA_PROCESS_TYPE_DECODE, self.batch_inject.get());
                    if (rc != 0) {
                        log.err("%s: llama_process(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n", .{ @as([*:0]const u8, "process"), rc, n_chunk, offset });
                        return false;
                    }
                    continue;
                }

                // gather target features per extract layer; the fused decode encodes and
                // injects them into the K/V cache at the target positions
                self.features_buf.resize(gpa, nc * n_embd_enc) catch oom();
                for (0..self.target_layer_ids_n) |k| {
                    const layer = ext.getEmbeddingsLayerInp(ctx_tgt, @intCast(self.target_layer_ids[k]));
                    if (layer == null) {
                        _ = c.ggml_abort("speculative.cpp", 1281, "DFlash: target layer %d input not extracted.", self.target_layer_ids[k]);
                    }
                    for (0..nc) |i| {
                        const dst = self.features_buf.items.ptr + i * n_embd_enc + k * n_embd_tgt;
                        const src = layer.? + (ub + uo + i) * n_embd_tgt;
                        rowCopy(dst, src, n_embd_tgt);
                    }
                }

                self.batch_inject.clear();
                for (0..nc) |i| {
                    const p = tok[ub + uo + i].pos[0];
                    const pos_arr = [4]c.llama_pos{ p, p, p, 0 };
                    _ = self.batch_inject.addEmbd(embdView(self.features_buf.items.ptr + i * n_embd_enc, 1, n_embd_enc), &pos_arr, seq_id, false);
                }
                const rc = c.llama_process(ctx_dft, c.LLAMA_PROCESS_TYPE_DECODE, self.batch_inject.get());
                if (rc != 0) {
                    log.err("%s: llama_process(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n", .{ @as([*:0]const u8, "process"), rc, n_chunk, offset });
                    return false;
                }
                if (self.is_lilicorr) {
                    // Ordinary DFlash can write to the KV cache, but LiliCorr needs to store embeddings across sequences.
                    if (ext.getEmbeddingsNextn(ctx_dft)) |projected| {
                        const n_embd_dec: usize = @intCast(self.n_embd_dec);
                        const seq_offset = s * self.lilicorr_rows;
                        const inj = self.batch_inject.tokens.items;
                        const pos_min: c.llama_pos = @max(0, inj[inj.len - 1].pos[0] - self.block_size);
                        for (0..nc) |i| {
                            const pos = inj[i].pos[0];
                            if (pos < pos_min) {
                                continue;
                            }
                            const slot = seq_offset + @as(usize, @intCast(pos)) % self.lilicorr_rows;
                            @memcpy(self.lilicorr_embd[slot * n_embd_dec ..][0..n_embd_dec], projected[i * n_embd_dec ..][0..n_embd_dec]);
                            self.lilicorr_pos[slot] = pos;
                        }
                    }
                }
            }
        }

        return true;
    }

    /// Ports `draft` (speculative.cpp:1326 @cd959bae9): one noise block per
    /// drafting sequence, all decoded at once, then each block read out --
    /// through the candidate lattice (DFlash2, LiliCorr), the confidence
    /// head (DSpark) or the samplers.
    pub fn draft(self: *DFlash, base: *Base, dparams: []DraftParams) void {
        const ctx_dft = self.params.ctx_dft;
        const n_seq = base.n_seq;

        self.batch.clear();

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

            const n_block_tokens: i32 = if (self.is_lilicorr) self.block_size else n_draft + @as(i32, if (self.is_dspark and self.sample_from_anchor) 0 else 1);
            i_block_beg[s] = self.batch.size();
            n_block[s] = n_block_tokens;
            var i: i32 = 0;
            while (i < n_block_tokens) : (i += 1) {
                const idx = self.batch.add(if (i == 0) dp.id_last else self.mask_token_id, n + i, seq_id, !self.is_dflash2 and !self.is_lilicorr);
                if (self.is_lilicorr) {
                    // For LiliCorr drafting batch, set state embeddings for use in the associated transformer head.
                    const pos: c.llama_pos = dp.pos0 - 1;
                    // `(size_t) pos`: sign-extended, as the C++ converts it
                    const slot = s * self.lilicorr_rows + @as(usize, @bitCast(@as(isize, pos))) % self.lilicorr_rows;
                    const n_embd_dec: usize = @intCast(self.n_embd_dec);
                    _ = self.batch.setEmbdState(idx, embdView(self.lilicorr_embd.ptr + slot * n_embd_dec, 1, n_embd_dec));
                }
            }
        }

        if (self.batch.size() == 0) return;

        // decode all sequence's noise block in a single batch
        const ret = c.llama_process(ctx_dft, c.LLAMA_PROCESS_TYPE_DECODE, self.batch.get());
        if (ret != 0) {
            log.wrn("%s: llama_process returned %d\n", .{ @as([*:0]const u8, "draft"), ret });
            return;
        }

        const n_embd_dec: usize = @intCast(self.n_embd_dec);

        for (0..n_seq) |s| {
            const seq_id: c.llama_seq_id = @intCast(s);
            if (i_block_beg[s] < 0) continue;
            const dp = &dparams[s];

            const beg = i_block_beg[s];
            const n_block_tokens = n_block[s];

            const smpl = self.smpls[s];

            const result = dp.result.?;

            if (self.is_dflash2 or self.is_lilicorr) {
                // Both DFlash and LiliCorr greedily walk the candidate lattice.
                const lattice = ext.getEmbeddingsNextn(ctx_dft);
                assertC(lattice != null, 1388, "lattice && \"DFlash2 selector produced no lattice\"");

                // DFlash2 can instead sample each step and hand q to the verifier for rejection sampling
                const sample = self.is_dflash2 and self.params.probabilistic and dp.result_q != null;

                // greedy keeps temperature 1, sampling uses the target's
                const temp: f32 = if (sample) dp.temp else 1.0;

                const seq: usize = @intCast(seq_id);
                if (sample and self.dflash2_seed[seq] != @as(i64, dp.seed)) {
                    self.dflash2_seed[seq] = dp.seed;
                    // the default seed means random; mixed so it differs from the target's, as in spec_retune
                    self.dflash2_rng[seq].seed(if (dp.seed == c.LLAMA_DEFAULT_SEED) randomDevice() else dp.seed ^ 0x85ebca6b);
                }

                var probs: std.ArrayList(f32) = .empty;
                defer probs.deinit(gpa);
                var sorted_idx: std.ArrayList(i32) = .empty;
                defer sorted_idx.deinit(gpa);

                var predecessor: usize = 0;
                var i: i32 = 1;
                while (i < (if (self.is_dflash2) n_block_tokens else self.params.n_max + 1)) : (i += 1) {
                    const row = lattice.? + @as(usize, @intCast(beg + i)) * n_embd_dec;
                    const top_k: usize = @intCast(if (self.is_dflash2) self.selector_top_k else self.lilicorr_top_k);
                    const scores = row + top_k + predecessor * top_k;

                    // std::max_element: the first of the largest
                    var best: usize = 0;
                    for (1..top_k) |k| {
                        if (scores[best] < scores[k]) best = k;
                    }
                    predecessor = best;
                    if (self.params.p_min > 0.0 or sample) {
                        // softmax(scores / temp) at the argmax, i.e. 1 / sum(exp((s_k - s_max) / temp))
                        probs.resize(gpa, top_k) catch oom();
                        var sum: f32 = 0.0;
                        for (0..top_k) |k| {
                            probs.items[k] = @exp((scores[k] - scores[predecessor]) / temp);
                            sum += probs.items[k];
                        }
                        if (1.0 / sum < self.params.p_min) break;
                        for (0..top_k) |k| probs.items[k] /= sum;
                    }
                    if (sample) {
                        // ggml_top_k leaves candidates unordered, so walk them by token id to keep a seed's draft stable
                        sorted_idx.resize(gpa, top_k) catch oom();
                        for (0..top_k) |k| sorted_idx.items[k] = @intCast(k);
                        libcxx.sort(i32, sorted_idx.items, row, struct {
                            fn lt(r: [*]const f32, x: i32, y: i32) bool {
                                return r[@intCast(x)] < r[@intCast(y)];
                            }
                        }.lt);

                        // a round-off overshoot keeps the argmax, which has q > 0
                        var u = libcxx.uniformF64(&self.dflash2_rng[seq]);
                        for (sorted_idx.items) |k| {
                            u -= probs.items[@intCast(k)];
                            if (u < 0.0) {
                                predecessor = @intCast(k);
                                break;
                            }
                        }

                        const q = dp.result_q.?.addOne(gpa) catch oom();
                        q.* = .empty;
                        q.ensureTotalCapacity(gpa, top_k) catch oom();
                        for (0..top_k) |k| {
                            q.appendAssumeCapacity(.{ .id = @intFromFloat(row[k]), .logit = scores[k], .p = probs.items[k] });
                        }
                    }
                    result.append(gpa, @intFromFloat(row[predecessor])) catch oom();
                }

                if (result.items.len < @as(usize, @bitCast(@as(isize, self.params.n_min)))) {
                    result.clearRetainingCapacity();
                    if (sample) S.truncateCandidates(dp.result_q.?, 0);
                }
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

    /// Ports `accept` (speculative.cpp:1523 @cd959bae9): nothing to do.
    pub fn accept(_: *DFlash, _: *Base, _: c.llama_seq_id, _: u16, _: bool) void {
        // noop
    }
};

// -----------------------------------------------------------------------------
// draft-mtp

/// Ports `common_speculative_impl_draft_mtp` (speculative.cpp:1528
/// @cd959bae9): multi-token prediction heads, in one of three modes -- a
/// shared KV cache with every head in one graph (Gemma 4), chained heads
/// one per draft step (Step 3.5), or a single head (Qwen 3.5).
pub const Mtp = struct {
    params: params_mod.SpeculativeDraft, // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    batch: common.Batch,

    smpls: []*sampling.Sampler,

    smpls_cfg: std.ArrayList(SmplCfg) = .empty,

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

    /// Ports `common_speculative_impl_draft_mtp` (speculative.cpp:1566
    /// @cd959bae9), the constructor.
    pub fn init(base: *Base, params: *const params_mod.Speculative) S.InitError!Mtp {
        const func = "common_speculative_impl_draft_mtp";
        var p = params.draft;
        const n_seq = base.n_seq;
        const ctx_tgt = p.ctx_tgt;
        const ctx_dft = p.ctx_dft;
        assertC(ctx_tgt != null and ctx_dft != null, 1572, "ctx_tgt && ctx_dft && \"MTP requires ctx_tgt and ctx_dft to be set\"");

        const n_embd = c.llama_model_n_embd_out(c.llama_get_model(ctx_dft));
        assertC(n_embd == c.llama_model_n_embd_out(c.llama_get_model(ctx_tgt)), 1575, "n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) && \"MTP input row width must match the target h_nextn width\"");
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

        const batch = common.Batch.init(gpa, ctx_dft);

        const smpls = gpa.alloc(*sampling.Sampler, n_seq) catch oom();
        for (smpls) |*s| s.* = draftSampler(c.llama_get_model(ctx_dft));

        // offload draft sampling to the backend
        const chains = backendChains(ctx_dft, n_seq, p.backend_sampling);

        const is_mem_shared = ext.getCtxOther(ctx_dft) == ctx_tgt;
        const chain_heads = n_mtp_layers > 1 and !is_mem_shared;

        ext.setEmbeddingsNextn(ctx_tgt, true, is_mem_shared);
        ext.setEmbeddingsNextn(ctx_dft, true, true);

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

    /// Ports the destructor of `common_speculative_impl_draft_mtp` (speculative.cpp:1642 @cd959bae9).
    pub fn deinit(self: *Mtp) void {
        freeBackendChains(self.params.ctx_dft, self.backend_chains);
        self.batch.deinit();
        freeSamplers(self.smpls);
        self.smpls_cfg.deinit(gpa);
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

    /// Ports `begin` (speculative.cpp:1656 @cd959bae9): the sequence's
    /// sampler is reset here rather than per round, or two identical
    /// requests differ.
    pub fn begin(self: *Mtp, _: *Base, seq_id: c.llama_seq_id, prompt: []const c.llama_token) void {
        self.smpls[@intCast(seq_id)].reset();

        const n: i32 = @intCast(prompt.len);
        if (n <= 0) return;

        const pos_max = c.llama_memory_seq_pos_max(c.llama_get_memory(self.params.ctx_dft), seq_id);

        if (pos_max < n - 1 and !self.is_mem_shared) {
            spcWrn("begin", "ctx_dft pos_max=%d < N-1=%d - " ++
                "process() hook may not have run on every prefill ubatch " ++
                "(need_embd / output flag on every prompt position?). " ++
                "Drafts may degrade.\n", .{ pos_max, n - 1 });
        }
    }

    /// Ports `process` (speculative.cpp:1677 @cd959bae9): the catch-up
    /// decode of the target's batch under every head, the hidden states
    /// shifted one position right as each entry's state input, and the rows
    /// the next `accept` chooses from. Vision tokens, which carry an
    /// embedding instead of an id, are caught up too.
    pub fn process(self: *Mtp, base: *Base, batch_in: *const common.Batch) bool {
        if (batch_in.size() <= 0) return true;

        if (!batch_in.hasToken() and !batch_in.hasEmbd()) return true;

        const n_seq = base.n_seq;
        const n_tokens: usize = @intCast(batch_in.size());
        const tok = batch_in.tokens.items;
        const n_embd: usize = @intCast(self.n_embd);

        // remember the first and last batch index for each sequence
        @memset(self.i_batch_beg, -1);
        @memset(self.i_batch_end, -1);

        for (0..n_tokens) |k| {
            for (0..n_seq) |s| {
                if (tok[k].seq_id == @as(c.llama_seq_id, @intCast(s))) {
                    self.i_batch_end[s] = @intCast(k);
                    if (self.i_batch_beg[s] < 0) self.i_batch_beg[s] = @intCast(k);
                }
            }
        }

        const ctx_tgt = self.params.ctx_tgt;
        const ctx_dft = self.params.ctx_dft;

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!self.is_mem_shared) {
            self.batch.clear();

            // pair each token with the tgt embedding shifted right by one position, and
            // the first token of each sequence with the pending embedding from a previous run
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            const h_tgt = ext.getEmbeddingsNextn(ctx_tgt);

            for (0..n_tokens) |k| {
                const t = &tok[k];

                const seq_id = t.seq_id;
                const s: usize = @intCast(seq_id);

                // vision tokens carry an embedding instead of an id
                const idx = if (t.id != c.LLAMA_TOKEN_NULL)
                    self.batch.add(t.id, t.pos[0], seq_id, false)
                else
                    self.batch.addEmbd(t.embd, &t.pos, seq_id, false);

                const h_row: [*]const f32 = if (@as(i64, @intCast(k)) == self.i_batch_beg[s])
                    self.pending_h[s].ptr
                else
                    h_tgt.? + (k - 1) * n_embd;

                _ = self.batch.setEmbdState(idx, embdView(h_row, 1, n_embd));
            }

            const mem_dft = c.llama_get_memory(ctx_dft);

            var ok = true;
            var head: i32 = 0;
            while (head < self.n_mtp_layers) : (head += 1) {
                if (self.chain_heads) {
                    for (0..n_seq) |s| {
                        if (self.i_batch_beg[s] < 0) continue;
                        _ = c.llama_memory_seq_rm(mem_dft, @intCast(s), tok[@intCast(self.i_batch_beg[s])].pos[0], -1);
                    }
                    ext.setNextnLayerOffset(ctx_dft, head);
                }

                const rc = c.llama_process(ctx_dft, c.LLAMA_PROCESS_TYPE_DECODE, self.batch.get());
                if (rc != 0) {
                    spcErr("process", "llama_process(ctx_dft) head=%d failed rc=%d (pos=%d)\n", .{ head, rc, tok[0].pos[0] });
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
            if (self.is_mem_shared and !tok[@intCast(self.i_batch_end[s])].output) {
                self.verify_h_rows[s] = 0;
                continue;
            }

            const all_rows_output = for (tok[@intCast(self.i_batch_beg[s])..@intCast(self.i_batch_end[s] + 1)]) |t| {
                if (!t.output) break false;
            } else true;
            const h_rows: usize = if (self.is_mem_shared and !all_rows_output) 1 else n_rows;
            const h_offset: i32 = if (self.is_mem_shared and !all_rows_output) self.i_batch_end[s] else self.i_batch_beg[s];

            self.verify_h_rows[s] = @intCast(h_rows);
            self.verify_h[s].resize(gpa, h_rows * n_embd) catch oom();

            for (0..h_rows) |i| {
                const h = ext.getEmbeddingsNextnIth(ctx_tgt, h_offset + @as(i32, @intCast(i)));
                rowCopy(self.verify_h[s].items.ptr + i * n_embd, h.?, n_embd);
            }

            rowCopy(self.pending_h[s].ptr, self.verify_h[s].items.ptr + (h_rows - 1) * n_embd, n_embd);
        }

        return true;
    }

    /// Ports `draft` (speculative.cpp:1802 @cd959bae9): the best guess of
    /// each head, or a sample with its candidates kept when the caller
    /// verifies by rejection.
    pub fn draft(self: *Mtp, base: *Base, dparams: []DraftParams) void {
        const ctx_dft = self.params.ctx_dft;
        const n_seq = base.n_seq;
        const n_embd: usize = @intCast(self.n_embd);

        self.batch.clear();

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
            // greedy drafting leaves no candidates behind, so the verifier falls back to sample-and-match
            if (!self.params.probabilistic) {
                dp.result_q = null;
            }

            // result_q is only set when the caller wants rejection, so it also gates the retune
            if (dp.result_q != null) {
                specRetune(self.smpls, &self.smpls_cfg, c.llama_get_model(ctx_dft), seq_id, dp.temp, dp.seed);
            }

            // a reset reseeds the chain, which breaks probabilistic drafting
            if (dp.result_q == null) {
                self.smpls[s].reset();
            }

            const idx = self.batch.add(dp.id_last, dp.pos0, seq_id, true);
            _ = self.batch.setEmbdState(idx, embdView(self.pending_h[s].ptr, 1, n_embd));

            self.i_last[s] = idx;

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

            const ret = c.llama_process(ctx_dft, c.LLAMA_PROCESS_TYPE_DECODE, self.batch.get());
            if (ret != 0) {
                spcErr("draft", "llama_process[%d] returned %d\n", .{ i, ret });
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            self.batch.clear();

            for (0..n_seq) |s| {
                const seq_id: c.llama_seq_id = @intCast(s);
                if (!drafting[s]) continue;

                const smpl = self.smpls[s];

                const id_sampled = smpl.sample(ctx_dft, self.i_last[s], true) catch oom();
                const h_row = ext.getEmbeddingsNextnIth(ctx_dft, self.i_last[s]);

                const cur_p = smpl.getCandidates(true);

                logCandidates("draft", ctx_dft, seq_id, i, cur_p);

                // add drafted token for each sequence
                const id = if (dparams[s].result_q != null) id_sampled else cur_p.data[0].id;

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

                if (dp.result_q) |q| appendCandidates(q, cur_p);

                if (self.params.n_max <= @as(i32, @intCast(result.items.len))) {
                    drafting[s] = false;
                    n_drafting -= 1;
                    continue;
                }

                if (self.chain_heads) {
                    self.chain_h[s].appendSlice(gpa, h_row.?[0..n_embd]) catch oom();

                    const n_rows: usize = result.items.len + 1; // id_last + tokens drafted so far
                    for (0..n_rows) |t| {
                        const tk = if (t == 0) dp.id_last else result.items[t - 1];
                        const idx = self.batch.add(tk, dp.pos0 + @as(c.llama_pos, @intCast(t)), seq_id, t == n_rows - 1);
                        _ = self.batch.setEmbdState(idx, embdView(self.chain_h[s].items.ptr + t * n_embd, 1, n_embd));
                        self.i_last[s] = idx;
                    }
                } else if (self.is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    const idx = self.batch.add(id, dp.pos0, seq_id, true);
                    _ = self.batch.setEmbdState(idx, embdView(h_row.?, 1, n_embd));
                    self.i_last[s] = idx;
                } else {
                    const idx = self.batch.add(id, dp.pos0 + i + 1, seq_id, true);
                    _ = self.batch.setEmbdState(idx, embdView(h_row.?, 1, n_embd));
                    self.i_last[s] = idx;
                }
            }

            if (self.batch.size() == 0) break;

            i += 1;
        }

        if (self.chain_heads) ext.setNextnLayerOffset(ctx_dft, 0); // restore default for non-draft decodes

        for (0..n_seq) |s| {
            const dp = &dparams[s];
            if (!dp.drafting) continue;

            if (dp.result.?.items.len < @as(usize, @bitCast(@as(isize, self.params.n_min)))) dp.result.?.clearRetainingCapacity();
        }
    }

    /// Ports `accept` (speculative.cpp:1968 @cd959bae9): the hidden row of
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

    /// Ports `get_state` (speculative.cpp:1983 @cd959bae9): the pending
    /// hidden row, as bytes.
    pub fn getState(self: *const Mtp, base: *const Base, seq_id: c.llama_seq_id, data: *std.ArrayList(u8)) bool {
        if (seq_id < 0 or seq_id >= @as(c.llama_seq_id, @intCast(base.n_seq))) return false;

        const row_bytes: usize = @as(usize, @intCast(self.n_embd)) * @sizeOf(f32);
        data.resize(gpa, row_bytes) catch oom();
        @memcpy(data.items, std.mem.sliceAsBytes(self.pending_h[@intCast(seq_id)])[0..row_bytes]);
        return true;
    }

    /// Ports `set_state` (speculative.cpp:1994 @cd959bae9).
    pub fn setState(self: *Mtp, base: *const Base, seq_id: c.llama_seq_id, data: []const u8) void {
        if (seq_id < 0 or seq_id >= @as(c.llama_seq_id, @intCast(base.n_seq))) return;

        const row_bytes: usize = @as(usize, @intCast(self.n_embd)) * @sizeOf(f32);
        if (data.len != row_bytes) return;
        @memcpy(std.mem.sliceAsBytes(self.pending_h[@intCast(seq_id)])[0..row_bytes], data);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
