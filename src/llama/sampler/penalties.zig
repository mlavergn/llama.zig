//! The repetition samplers: penalties and DRY.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-sampler.cpp` (v0.4.0, `5266f24da`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # Hash containers whose order is not observable
//!
//! Both samplers keep `std::unordered_map`s, and DRY an
//! `std::unordered_multimap`, whose iteration order libc++ leaves to its
//! hashing. None of it reaches the output, which is what lets them be Zig
//! hash maps:
//!
//! - penalties' `token_count` is only looked up, except in
//!   `backend_set_input`, which sorts what it reads by token id first.
//! - DRY's `dry_max_token_repeat` is only looked up and assigned.
//! - DRY's `dry_processed_breakers` is read through `equal_range`, but every
//!   reader is order-independent: the restart search keeps the longest
//!   match, the single-token-breaker test asks whether any tail is empty, and
//!   the duplicate check asks whether any tail is equal.
//!
//! # Arithmetic read off the disassembly
//!
//! - penalties, `float(count) * penalty_freq + float(count > 0) *
//!   penalty_present`: `fmadd s6, s6, s1, s7` -- the **left** product is
//!   fused with the rounded right one, the same shape `CLAUDE.md` records for
//!   `xielu` and the repack epilogue.
//! - DRY, `dry_multiplier * std::pow(dry_base, repeat_exp)`: `std::pow` of a
//!   `float` and an `int` is the `double` overload, so it is libm's `pow`,
//!   multiplied in `double` and rounded once to `float`.

const std = @import("std");
const impl = @import("../impl.zig");
const libcxx = @import("../libcxx.zig");
const core = @import("core.zig");

const c = impl.c;
const Sampler = core.Sampler;
const SamplerI = core.SamplerI;
const TokenDataArray = core.TokenDataArray;
const Token = core.Token;
const Backend = core.Backend;
const src = core.src;

// -----------------------------------------------------------------------------
// penalties

/// Ports `llama_sampler_penalties` (llama-sampler.cpp:2856 @5266f24da).
const Penalties = struct {
    base: Backend,

    n_vocab: i32,
    penalty_last_n: i32,
    penalty_repeat: f32,
    penalty_freq: f32,
    penalty_present: f32,

    prev: core.RingBuffer(Token),

    /// How often each token occurs in `prev`.
    token_count: std.AutoHashMapUnmanaged(Token, c_int),

    // The current backend graph's inputs.
    inp_token_ids: ?*c.struct_ggml_tensor = null,
    inp_counts: ?*c.struct_ggml_tensor = null,

    n_max: i32 = 0,
    has_candidates: bool = false,

    host_token_ids: std.ArrayList(i32) = .empty,
    host_counts: std.ArrayList(i32) = .empty,

    /// Ports `llama_sampler_penalties`'s `copy_state` (llama-sampler.cpp:2879 @5266f24da).
    fn copyState(self: *Penalties, other: *const Penalties) void {
        self.prev.assign(&other.prev);
        self.copyCounts(other);
    }

    fn copyCounts(self: *Penalties, other: *const Penalties) void {
        self.token_count.deinit(impl.allocator);
        self.token_count = other.token_count.clone(impl.allocator) catch impl.abort(src, 2882, "out of memory");
    }

    /// Ports `llama_sampler_penalties`'s static `is_disabled` (llama-sampler.cpp:2885 @5266f24da).
    fn isDisabledParams(penalty_last_n: i32, penalty_repeat: f32, penalty_freq: f32, penalty_present: f32) bool {
        return penalty_last_n == 0 or
            (penalty_repeat == 1.0 and penalty_freq == 0.0 and penalty_present == 0.0);
    }

    /// Ports `llama_sampler_penalties`'s member `is_disabled` (llama-sampler.cpp:2894 @5266f24da).
    fn isDisabled(self: *const Penalties) bool {
        return isDisabledParams(self.penalty_last_n, self.penalty_repeat, self.penalty_freq, self.penalty_present);
    }
};

/// Ports `llama_sampler_penalties_name` (llama-sampler.cpp:2914 @5266f24da).
fn penaltiesName(smpl: [*c]const Sampler) callconv(.c) [*c]const u8 {
    return core.ctxOf(Penalties, smpl).base.getName();
}

/// Ports `llama_sampler_penalties_accept` (llama-sampler.cpp:2919 @5266f24da).
///
/// Counts the token in, and the token falling out of the window out.
fn penaltiesAccept(smpl: [*c]Sampler, token: Token) callconv(.c) void {
    const ctx = core.ctxOf(Penalties, smpl);
    if (ctx.penalty_last_n == 0) return;

    const gop = ctx.token_count.getOrPut(impl.allocator, token) catch impl.abort(src, 2925, "out of memory");
    if (!gop.found_existing) gop.value_ptr.* = 0;
    gop.value_ptr.* += 1;

    if (ctx.prev.size() >= @as(usize, @intCast(ctx.penalty_last_n))) {
        const old = ctx.prev.front();

        // `token_count[old]--` default-constructs a missing entry first.
        const og = ctx.token_count.getOrPut(impl.allocator, old) catch impl.abort(src, 2931, "out of memory");
        if (!og.found_existing) og.value_ptr.* = 0;
        og.value_ptr.* -= 1;
        if (og.value_ptr.* == 0) {
            _ = ctx.token_count.remove(old);
        }
    }

    ctx.prev.pushBack(token);
}

/// Ports `llama_sampler_penalties_apply` (llama-sampler.cpp:2950 @5266f24da).
///
/// The repeat penalty multiplies a non-positive logit and divides a positive
/// one; the frequency and presence terms are fused -- see the file header.
fn penaltiesApply(smpl: [*c]Sampler, cur_p: [*c]TokenDataArray) callconv(.c) void {
    const ctx = core.ctxOf(Penalties, smpl);

    if (ctx.isDisabled()) return;

    for (core.items(cur_p)) |*d| {
        const count = ctx.token_count.get(d.id) orelse continue;

        if (d.logit <= 0) {
            d.logit *= ctx.penalty_repeat;
        } else {
            d.logit /= ctx.penalty_repeat;
        }

        const present: f32 = @as(f32, if (count > 0) 1.0 else 0.0) * ctx.penalty_present;
        d.logit -= @mulAdd(f32, @floatFromInt(count), ctx.penalty_freq, present);
    }

    cur_p.*.sorted = false;
}

/// Ports `llama_sampler_penalties_reset` (llama-sampler.cpp:2982 @5266f24da).
fn penaltiesReset(smpl: [*c]Sampler) callconv(.c) void {
    const ctx = core.ctxOf(Penalties, smpl);
    ctx.prev.clear();
    ctx.token_count.clearRetainingCapacity();
}

/// Ports `llama_sampler_penalties_clone` (llama-sampler.cpp:2988 @5266f24da).
fn penaltiesClone(smpl: [*c]const Sampler) callconv(.c) [*c]Sampler {
    const ctx = core.ctxOf(Penalties, smpl);
    const result = llama_sampler_init_penalties(ctx.n_vocab, ctx.penalty_last_n, ctx.penalty_repeat, ctx.penalty_freq, ctx.penalty_present);

    const result_ctx = core.ctxOf(Penalties, result);
    result_ctx.prev.assign(&ctx.prev);
    result_ctx.copyCounts(ctx);

    return result;
}

/// Ports `llama_sampler_penalties_free` (llama-sampler.cpp:3008 @5266f24da).
fn penaltiesFree(smpl: [*c]Sampler) callconv(.c) void {
    const ctx = core.ctxOf(Penalties, smpl);
    ctx.prev.deinit();
    ctx.token_count.deinit(impl.allocator);
    ctx.host_token_ids.deinit(impl.allocator);
    ctx.host_counts.deinit(impl.allocator);
    core.destroy(Penalties, ctx);
}

/// Ports `llama_sampler_penalties_backend_init` (llama-sampler.cpp:3012 @5266f24da).
///
/// Declines the backend when a sequence can have more than one output: the
/// counts would have to advance between outputs inside one graph.
fn penaltiesBackendInit(smpl: [*c]Sampler, buft: c.ggml_backend_buffer_type_t, n_outputs_max_per_seq: u32) callconv(.c) bool {
    const sctx = core.ctxOf(Penalties, smpl);

    if (n_outputs_max_per_seq > 1) {
        sctx.base.initBackend(false);
        return false;
    }

    const res = core.backendSupport(smpl, buft);

    sctx.base.initBackend(res);

    return res;
}

/// Ports `llama_sampler_penalties_backend_apply` (llama-sampler.cpp:3030 @5266f24da).
fn penaltiesBackendApply(smpl: [*c]Sampler, ctx: ?*c.struct_ggml_context, gf: ?*c.struct_ggml_cgraph, data: [*c]c.struct_llama_sampler_data) callconv(.c) void {
    _ = gf;

    const sctx = core.ctxOf(Penalties, smpl);

    if (sctx.isDisabled()) return;

    impl.assert(sctx.n_vocab > 0, src, 3043, "sctx->n_vocab > 0");

    sctx.has_candidates = data.*.candidates != null;
    sctx.n_max = libcxx.min(i32, sctx.penalty_last_n, sctx.n_vocab);

    sctx.inp_token_ids = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, sctx.n_max);
    _ = c.ggml_set_name(sctx.inp_token_ids, "penalties_token_ids");
    c.ggml_set_input(sctx.inp_token_ids);

    sctx.inp_counts = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, sctx.n_max);
    _ = c.ggml_set_name(sctx.inp_counts, "penalties_counts");
    c.ggml_set_input(sctx.inp_counts);

    if (@as(i32, @intCast(sctx.host_token_ids.items.len)) != sctx.n_max) {
        const n: usize = @intCast(sctx.n_max);
        sctx.host_token_ids.resize(impl.allocator, n) catch impl.abort(src, 3057, "out of memory");
        @memset(sctx.host_token_ids.items, 0);
        sctx.host_counts.resize(impl.allocator, n) catch impl.abort(src, 3058, "out of memory");
        @memset(sctx.host_counts.items, 0);
    }

    const logits = c.ggml_reshape_1d(ctx, data.*.logits, c.ggml_nelements(data.*.logits));
    var gathered = logits;
    var counts_f32 = c.ggml_cast(ctx, sctx.inp_counts, c.GGML_TYPE_F32);

    if (sctx.has_candidates) {
        const candidates = c.ggml_reshape_1d(ctx, data.*.candidates, c.ggml_nelements(data.*.candidates));
        const n_candidates = candidates[0].ne[0];
        impl.assert(n_candidates == c.ggml_nelements(logits), src, 3070, "n_candidates == ggml_nelements(logits)");

        var counts_rows = c.ggml_fill(ctx, c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, 1, sctx.n_vocab), 0.0);
        const scatter_rows = c.ggml_reshape_2d(ctx, counts_f32, 1, sctx.n_max);
        counts_rows = c.ggml_set_rows(ctx, counts_rows, scatter_rows, sctx.inp_token_ids);
        counts_f32 = c.ggml_get_rows(ctx, counts_rows, candidates);
        counts_f32 = c.ggml_reshape_1d(ctx, counts_f32, n_candidates);
    } else {
        const logits_rows = c.ggml_reshape_2d(ctx, logits, 1, c.ggml_nelements(logits));
        gathered = c.ggml_get_rows(ctx, logits_rows, sctx.inp_token_ids);
        gathered = c.ggml_reshape_1d(ctx, gathered, sctx.n_max);
    }

    const active_mask = c.ggml_step(ctx, counts_f32);
    const inactive_mask = c.ggml_sub(ctx, c.ggml_fill(ctx, active_mask, 1.0), active_mask);

    var penalized = gathered;

    if (sctx.penalty_repeat != 1.0) {
        const pos_mask = c.ggml_step(ctx, penalized);
        const neg_mask = c.ggml_sub(ctx, c.ggml_fill(ctx, pos_mask, 1.0), pos_mask);

        const pos_scale = c.ggml_scale(ctx, pos_mask, 1.0 / sctx.penalty_repeat);
        const neg_scale = c.ggml_scale(ctx, neg_mask, sctx.penalty_repeat);
        var repeat_scale = c.ggml_add(ctx, pos_scale, neg_scale);

        // Inactive entries scale by 1, so a -inf from top-p is not 0 * -inf.
        repeat_scale = c.ggml_mul(ctx, repeat_scale, active_mask);
        repeat_scale = c.ggml_add(ctx, repeat_scale, inactive_mask);
        penalized = c.ggml_mul(ctx, gathered, repeat_scale);
    }

    if (sctx.penalty_freq != 0.0) {
        const penalty_freq = c.ggml_scale(ctx, counts_f32, sctx.penalty_freq);
        penalized = c.ggml_sub(ctx, penalized, penalty_freq);
    }

    if (sctx.penalty_present != 0.0) {
        const penalty_present = c.ggml_scale(ctx, active_mask, sctx.penalty_present);
        penalized = c.ggml_sub(ctx, penalized, penalty_present);
    }

    if (sctx.has_candidates) {
        data.*.logits = penalized;
    } else {
        var logits_rows = c.ggml_reshape_2d(ctx, logits, 1, c.ggml_nelements(logits));
        const scatter_rows = c.ggml_reshape_2d(ctx, penalized, 1, sctx.n_max);
        logits_rows = c.ggml_set_rows(ctx, logits_rows, scatter_rows, sctx.inp_token_ids);
        data.*.logits = c.ggml_reshape_1d(ctx, logits_rows, c.ggml_nelements(logits));
    }
}

const IdCount = struct { id: i32, count: i32 };

/// The comparator of the `sort` (llama-sampler.cpp:3150 @5266f24da) in
/// `llama_sampler_penalties_backend_set_input`.
fn lessId(_: void, a: IdCount, b: IdCount) bool {
    return a.id < b.id;
}

/// Ports `llama_sampler_penalties_backend_set_input` (llama-sampler.cpp:3123 @5266f24da).
///
/// Writes the active ids, sorted, and pads to `n_max` with a filler id that
/// is not active, so `ggml_set_rows` never sees a duplicate index.
fn penaltiesBackendSetInput(smpl: [*c]Sampler) callconv(.c) void {
    const sctx = core.ctxOf(Penalties, smpl);

    if (sctx.inp_token_ids == null or sctx.inp_counts == null or sctx.n_max <= 0 or sctx.n_vocab <= 0) return;

    if (sctx.isDisabled()) return;

    var n_active: i32 = 0;

    var it = sctx.token_count.iterator();
    while (it.next()) |kv| {
        impl.assert(n_active < sctx.n_max, src, 3138, "n_active < sctx->n_max");
        sctx.host_token_ids.items[@intCast(n_active)] = kv.key_ptr.*;
        sctx.host_counts.items[@intCast(n_active)] = kv.value_ptr.*;
        n_active += 1;
    }

    const entries = impl.allocator.alloc(IdCount, @intCast(n_active)) catch impl.abort(src, 3146, "out of memory");
    defer impl.allocator.free(entries);
    for (entries, 0..) |*e, i| e.* = .{ .id = sctx.host_token_ids.items[i], .count = sctx.host_counts.items[i] };
    libcxx.sort(IdCount, entries, {}, lessId);
    for (entries, 0..) |e, i| {
        sctx.host_token_ids.items[i] = e.id;
        sctx.host_counts.items[i] = e.count;
    }

    var filler: i32 = 0;
    if (n_active < sctx.n_max) {
        while (sctx.token_count.contains(filler)) filler += 1;
        impl.assert(filler < sctx.n_vocab, src, 3166, "filler < sctx->n_vocab");
    }

    var i: usize = @intCast(n_active);
    while (i < @as(usize, @intCast(sctx.n_max))) : (i += 1) {
        sctx.host_token_ids.items[i] = filler;
        sctx.host_counts.items[i] = 0;
    }

    c.ggml_backend_tensor_set(sctx.inp_token_ids, sctx.host_token_ids.items.ptr, 0, @as(usize, @intCast(sctx.n_max)) * @sizeOf(i32));
    c.ggml_backend_tensor_set(sctx.inp_counts, sctx.host_counts.items.ptr, 0, @as(usize, @intCast(sctx.n_max)) * @sizeOf(i32));
}

/// Ports `llama_sampler_penalties_backend_reset` (llama-sampler.cpp:3182 @5266f24da).
fn penaltiesBackendReset(smpl: [*c]Sampler) callconv(.c) void {
    const sctx = core.ctxOf(Penalties, smpl);
    sctx.inp_token_ids = null;
    sctx.inp_counts = null;
}

fn penaltiesCopyState(src_smpl: [*c]const Sampler, dst_smpl: [*c]Sampler) callconv(.c) void {
    core.ctxOf(Penalties, dst_smpl).copyState(core.ctxOf(Penalties, src_smpl));
}

/// Ports `llama_sampler_penalties_i` (llama-sampler.cpp:3188 @5266f24da).
var penalties_iface: SamplerI = .{
    .name = penaltiesName,
    .accept = penaltiesAccept,
    .apply = penaltiesApply,
    .reset = penaltiesReset,
    .clone = penaltiesClone,
    .free = penaltiesFree,
    .backend_init = penaltiesBackendInit,
    .backend_accept = null,
    .backend_apply = penaltiesBackendApply,
    .backend_set_input = penaltiesBackendSetInput,
    .backend_reset = penaltiesBackendReset,
    .copy_state = penaltiesCopyState,
};

/// Ports `llama_sampler_init_penalties` (llama-sampler.cpp:3203 @5266f24da).
///
/// Parameters:
/// - `n_vocab`: the vocabulary size; the backend graph needs it.
/// - `penalty_last_n_in`: the window; negative counts as 0, which disables it.
/// - `penalty_repeat`: the repeat factor; 1 disables it.
/// - `penalty_freq`: subtracted per occurrence.
/// - `penalty_present`: subtracted once for any occurrence.
///
/// Return: the sampler, or the `?penalties` placeholder when disabled.
pub export fn llama_sampler_init_penalties(n_vocab: i32, penalty_last_n_in: i32, penalty_repeat: f32, penalty_freq: f32, penalty_present: f32) callconv(.c) [*c]Sampler {
    const penalty_last_n = libcxx.max(i32, penalty_last_n_in, 0);

    if (Penalties.isDisabledParams(penalty_last_n, penalty_repeat, penalty_freq, penalty_present)) {
        return core.initEmpty("?penalties");
    }

    return core.llama_sampler_init(&penalties_iface, core.create(Penalties, .{
        .base = Backend.init("penalties"),
        .n_vocab = n_vocab,
        .penalty_last_n = penalty_last_n,
        .penalty_repeat = penalty_repeat,
        .penalty_freq = penalty_freq,
        .penalty_present = penalty_present,
        .prev = core.RingBuffer(Token).init(@intCast(penalty_last_n)),
        .token_count = .empty,
    }));
}

// -----------------------------------------------------------------------------
// DRY

/// `dry_processed_breakers`: each head token's restart tails. A tail is
/// owned; the empty tail marks a single-token breaker.
const Breakers = std.AutoHashMapUnmanaged(Token, std.ArrayList(std.ArrayList(Token)));

fn freeBreakers(b: *Breakers) void {
    var it = b.valueIterator();
    while (it.next()) |tails| {
        for (tails.items) |*t| t.deinit(impl.allocator);
        tails.deinit(impl.allocator);
    }
    b.deinit(impl.allocator);
}

fn cloneBreakers(b: *const Breakers) Breakers {
    var out: Breakers = .empty;
    var it = b.iterator();
    while (it.next()) |kv| {
        var tails: std.ArrayList(std.ArrayList(Token)) = .empty;
        for (kv.value_ptr.items) |t| {
            const copy = std.ArrayList(Token).initCapacity(impl.allocator, t.items.len) catch impl.abort(src, 3611, "out of memory");
            var cp = copy;
            cp.appendSliceAssumeCapacity(t.items);
            tails.append(impl.allocator, cp) catch impl.abort(src, 3611, "out of memory");
        }
        out.put(impl.allocator, kv.key_ptr.*, tails) catch impl.abort(src, 3611, "out of memory");
    }
    return out;
}

/// Ports `std::unordered_multimap::emplace` into `dry_processed_breakers`.
fn addBreaker(b: *Breakers, head: Token, tail: []const Token) void {
    const gop = b.getOrPut(impl.allocator, head) catch impl.abort(src, 3334, "out of memory");
    if (!gop.found_existing) gop.value_ptr.* = .empty;
    var t: std.ArrayList(Token) = .empty;
    t.appendSlice(impl.allocator, tail) catch impl.abort(src, 3334, "out of memory");
    gop.value_ptr.append(impl.allocator, t) catch impl.abort(src, 3334, "out of memory");
}

/// Ports `llama_sampler_dry` (llama-sampler.cpp:3317 @5266f24da).
const Dry = struct {
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: i32,
    dry_penalty_last_n: i32,

    dry_processed_breakers: Breakers,
    dry_repeat_count: std.ArrayList(c_int),
    dry_max_token_repeat: std.AutoHashMapUnmanaged(Token, c_int),
    last_tokens: core.RingBuffer(Token),
};

/// `vocab.detokenize({token_id}, true)`, through the C API: no special
/// tokens removed, special tokens rendered.
fn detokenizeOne(vocab: ?*const c.struct_llama_vocab, token: Token, buf: *std.ArrayList(u8)) []const u8 {
    // Mirrors the vector overload of `llama_vocab`'s `detokenize`
    // (llama-vocab.cpp:4171 @5266f24da): start at max(capacity, n_tokens).
    var text_len: usize = @max(buf.capacity, 1);
    buf.resize(impl.allocator, text_len) catch impl.abort(src, 3332, "out of memory");
    var n_chars = c.llama_detokenize(vocab, &token, 1, buf.items.ptr, @intCast(text_len), false, true);
    if (n_chars < 0) {
        text_len = @intCast(-n_chars);
        buf.resize(impl.allocator, text_len) catch impl.abort(src, 3332, "out of memory");
        n_chars = c.llama_detokenize(vocab, &token, 1, buf.items.ptr, @intCast(text_len), false, true);
        impl.assert(n_chars <= @as(i32, @intCast(text_len)), "llama-vocab.cpp", 4178, "n_chars <= (int32_t)text.size()");
    }
    return buf.items[0..@intCast(n_chars)];
}

/// `vocab.tokenize(str, false, false)`, through the C API.
fn tokenizeNoSpecial(vocab: ?*const c.struct_llama_vocab, text: []const u8, out: *std.ArrayList(Token)) void {
    // `llama_vocab::tokenize(std::string...)` returns the whole vector; the C
    // API reports the size needed when the buffer is short.
    out.resize(impl.allocator, @max(text.len + 2, 8)) catch impl.abort(src, 3349, "out of memory");
    var n = c.llama_tokenize(vocab, text.ptr, @intCast(text.len), out.items.ptr, @intCast(out.items.len), false, false);
    if (n < 0) {
        out.resize(impl.allocator, @intCast(-n)) catch impl.abort(src, 3349, "out of memory");
        n = c.llama_tokenize(vocab, text.ptr, @intCast(text.len), out.items.ptr, @intCast(out.items.len), false, false);
    }
    out.shrinkRetainingCapacity(@intCast(n));
}

/// Ports `get_overlapping_token_sequences` (llama-sampler.cpp:3330 @5266f24da).
///
/// For a breaker string, records every token whose text contains it -- a
/// single-token breaker -- and every token whose text ends with a prefix of
/// it, with the tokenization of the rest as the tail to match.
fn getOverlappingTokenSequences(vocab: ?*const c.struct_llama_vocab, str: []const u8, token_sequences: *Breakers, max_tail_len: i32) void {
    var word_buf: std.ArrayList(u8) = .empty;
    defer word_buf.deinit(impl.allocator);
    var tokenization: std.ArrayList(Token) = .empty;
    defer tokenization.deinit(impl.allocator);

    const n_tokens = c.llama_vocab_n_tokens(vocab);
    var token_id: Token = 0;
    while (token_id < n_tokens) : (token_id += 1) {
        const word = detokenizeOne(vocab, token_id, &word_buf);
        if (std.mem.find(u8, word, str) != null) {
            addBreaker(token_sequences, token_id, &.{});
        } else {
            const word_len = word.len;
            const str_len = str.len;
            var from: usize = 0;
            while (std.mem.findScalarPos(u8, word, from, str[0])) |pos| {
                from = pos + 1;
                var match = true;
                var i: usize = 1;
                while (i < str_len and i + pos < word_len) : (i += 1) {
                    if (word[pos + i] != str[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    tokenizeNoSpecial(vocab, str[i..], &tokenization);
                    if (max_tail_len >= 0 and tokenization.items.len > @as(usize, @intCast(max_tail_len))) {
                        tokenization.shrinkRetainingCapacity(@intCast(max_tail_len));
                    }

                    var found = false;
                    if (token_sequences.get(token_id)) |tails| {
                        for (tails.items) |t| {
                            if (std.mem.eql(Token, t.items, tokenization.items)) {
                                found = true;
                                break;
                            }
                        }
                    }
                    if (!found) {
                        addBreaker(token_sequences, token_id, tokenization.items);
                    }
                }
            }
        }
    }
}

/// Ports `llama_sampler_dry_name` (llama-sampler.cpp:3372 @5266f24da).
fn dryName(smpl: [*c]const Sampler) callconv(.c) [*c]const u8 {
    _ = smpl;
    return "dry";
}

/// Ports `llama_sampler_dry_accept` (llama-sampler.cpp:3376 @5266f24da).
fn dryAccept(smpl: [*c]Sampler, token: Token) callconv(.c) void {
    const ctx = core.ctxOf(Dry, smpl);
    if (ctx.dry_multiplier == 0.0 or ctx.dry_base < 1.0 or ctx.dry_penalty_last_n == 0) return;

    ctx.last_tokens.pushBack(token);
}

/// Ports `llama_sampler_dry_apply` (llama-sampler.cpp:3386 @5266f24da).
///
/// Penalises the tokens that would extend a repeat of the recent context,
/// by `multiplier * base^(length - allowed)`. Restart sequences cap how far
/// back a repeat can reach.
fn dryApply(smpl: [*c]Sampler, cur_p: [*c]TokenDataArray) callconv(.c) void {
    const ctx = core.ctxOf(Dry, smpl);

    if (ctx.dry_multiplier == 0.0 or ctx.dry_base < 1.0 or ctx.dry_penalty_last_n == 0) return;

    const last_n_repeat: c_int = libcxx.min(c_int, @intCast(ctx.last_tokens.size()), ctx.dry_penalty_last_n);

    if (last_n_repeat <= ctx.dry_allowed_length) return;

    ctx.dry_repeat_count.resize(impl.allocator, @intCast(last_n_repeat)) catch impl.abort(src, 3399, "out of memory");
    @memset(ctx.dry_repeat_count.items, 0);
    ctx.dry_max_token_repeat.clearRetainingCapacity();

    // Step 1: the nearest restart sequence limits how long a repeat can be.
    var rep_limit: c_int = last_n_repeat;
    var i: c_int = 0;
    while (i < last_n_repeat) : (i += 1) {
        const token = ctx.last_tokens.rat(@intCast(i));
        const tails = ctx.dry_processed_breakers.get(token) orelse continue;
        var longest_match: c_int = -1;
        for (tails.items) |tail| {
            const seq_len: c_int = @intCast(tail.items.len);
            if (seq_len > longest_match and seq_len <= i) {
                var match = true;
                var offset: c_int = 0;
                while (offset < seq_len) : (offset += 1) {
                    if (tail.items[@intCast(offset)] != ctx.last_tokens.rat(@intCast(i - offset - 1))) {
                        match = false;
                        break;
                    }
                }
                if (match) longest_match = seq_len;
            }
        }
        if (longest_match >= 0) {
            rep_limit = i - longest_match;
            break;
        }
    }
    if (rep_limit < ctx.dry_allowed_length) return;

    // Step 2: the reverse Z-algorithm, giving each position the length of the
    // suffix of the context that also ends there.
    {
        const last = last_n_repeat - 1;

        var rt: c_int = 0;
        var lt: c_int = 0;

        var k: c_int = 1;
        while (k < last_n_repeat) : (k += 1) {
            if (k > rt) {
                var n: c_int = 0;
                while (n + k < last_n_repeat and ctx.last_tokens.rat(@intCast(n)) == ctx.last_tokens.rat(@intCast(n + k))) {
                    n += 1;
                }
                ctx.dry_repeat_count.items[@intCast(last - k)] = libcxx.min(c_int, n, rep_limit);
                if (n > 0) {
                    lt = k;
                    rt = k + n - 1;
                }
            } else {
                const p = k - lt;
                const right_part_len = rt - k + 1;

                if (ctx.dry_repeat_count.items[@intCast(last - p)] < right_part_len) {
                    const n = libcxx.min(c_int, ctx.dry_repeat_count.items[@intCast(last - p)], rep_limit);
                    ctx.dry_repeat_count.items[@intCast(last - k)] = n;
                } else {
                    var ii = rt + 1;
                    while (ii < last_n_repeat and ctx.last_tokens.rat(@intCast(ii)) == ctx.last_tokens.rat(@intCast(ii - k))) {
                        ii += 1;
                    }

                    const n = libcxx.min(c_int, ii - k, rep_limit);
                    ctx.dry_repeat_count.items[@intCast(last - k)] = n;
                    lt = k;
                    rt = ii - 1;
                }
            }
        }
    }

    // Step 3: the longest repeat each next token would extend.
    i = 0;
    while (i < last_n_repeat - 1) : (i += 1) {
        const repeat_len = ctx.dry_repeat_count.items[@intCast(i)];
        if (repeat_len >= ctx.dry_allowed_length) {
            const token = ctx.last_tokens.rat(@intCast(last_n_repeat - 2 - i));
            const gop = ctx.dry_max_token_repeat.getOrPut(impl.allocator, token) catch impl.abort(src, 3549, "out of memory");
            if (!gop.found_existing or gop.value_ptr.* < repeat_len) {
                gop.value_ptr.* = repeat_len;
            }
        }
    }

    // Step 4: the penalties, with the exponent clamped below float overflow.
    const FLOAT_MAX_LOG: f32 = 88.7228391;
    var max_exponent: c_int = 0;
    if (ctx.dry_base > 1.000001) {
        max_exponent = core.cvtzs(FLOAT_MAX_LOG / impl.logf(ctx.dry_base));
    }

    for (core.items(cur_p)) |*d| {
        const af = ctx.dry_max_token_repeat.get(d.id) orelse continue;

        var is_single_token_breaker = false;
        if (ctx.dry_processed_breakers.get(d.id)) |tails| {
            for (tails.items) |t| {
                if (t.items.len == 0) {
                    is_single_token_breaker = true;
                    break;
                }
            }
        }

        if (!is_single_token_breaker) {
            var repeat_exp = af - ctx.dry_allowed_length;
            if (max_exponent > 0 and repeat_exp > max_exponent) {
                repeat_exp = max_exponent;
            }
            const penalty: f32 = @floatCast(@as(f64, ctx.dry_multiplier) * impl.pow(ctx.dry_base, @floatFromInt(repeat_exp)));
            d.logit -= penalty;
        }
    }

    cur_p.*.sorted = false;
}

/// Ports `llama_sampler_dry_reset` (llama-sampler.cpp:3593 @5266f24da).
fn dryReset(smpl: [*c]Sampler) callconv(.c) void {
    const ctx = core.ctxOf(Dry, smpl);
    ctx.last_tokens.clear();
    ctx.dry_repeat_count.clearRetainingCapacity();
    ctx.dry_max_token_repeat.clearRetainingCapacity();
}

/// Ports `llama_sampler_dry_clone` (llama-sampler.cpp:3600 @5266f24da).
///
/// The C++ builds the clone through `llama_sampler_init_dry` with a dummy,
/// empty `llama_vocab` and no breakers, only to overwrite every field. The
/// constructor's sole use of the vocabulary is breaker processing, which no
/// breakers skips, so the clone here is built directly.
fn dryClone(smpl: [*c]const Sampler) callconv(.c) [*c]Sampler {
    const ctx = core.ctxOf(Dry, smpl);

    const result = initDry(ctx.dry_multiplier, ctx.dry_base, ctx.dry_allowed_length, ctx.dry_penalty_last_n, .empty);

    const result_ctx = core.ctxOf(Dry, result);
    freeBreakers(&result_ctx.dry_processed_breakers);
    result_ctx.dry_processed_breakers = cloneBreakers(&ctx.dry_processed_breakers);
    result_ctx.dry_repeat_count.clearRetainingCapacity();
    result_ctx.dry_repeat_count.appendSlice(impl.allocator, ctx.dry_repeat_count.items) catch impl.abort(src, 3612, "out of memory");
    result_ctx.dry_max_token_repeat.deinit(impl.allocator);
    result_ctx.dry_max_token_repeat = ctx.dry_max_token_repeat.clone(impl.allocator) catch impl.abort(src, 3613, "out of memory");
    result_ctx.last_tokens.assign(&ctx.last_tokens);

    return result;
}

/// Ports `llama_sampler_dry_free` (llama-sampler.cpp:3620 @5266f24da).
fn dryFree(smpl: [*c]Sampler) callconv(.c) void {
    const ctx = core.ctxOf(Dry, smpl);
    freeBreakers(&ctx.dry_processed_breakers);
    ctx.dry_repeat_count.deinit(impl.allocator);
    ctx.dry_max_token_repeat.deinit(impl.allocator);
    ctx.last_tokens.deinit();
    core.destroy(Dry, ctx);
}

/// Ports `llama_sampler_dry_i` (llama-sampler.cpp:3624 @5266f24da).
var dry_iface: SamplerI = .{
    .name = dryName,
    .accept = dryAccept,
    .apply = dryApply,
    .reset = dryReset,
    .clone = dryClone,
    .free = dryFree,
};

/// The enabled half of `llama_sampler_init_dry`, given processed breakers.
fn initDry(dry_multiplier: f32, dry_base: f32, dry_allowed_length: i32, dry_penalty_last_n: i32, breakers: Breakers) [*c]Sampler {
    var repeat_count: std.ArrayList(c_int) = .empty;
    repeat_count.appendNTimes(impl.allocator, 0, @intCast(dry_penalty_last_n)) catch impl.abort(src, 3682, "out of memory");
    return core.llama_sampler_init(&dry_iface, core.create(Dry, .{
        .dry_multiplier = dry_multiplier,
        .dry_base = dry_base,
        .dry_allowed_length = dry_allowed_length,
        .dry_penalty_last_n = dry_penalty_last_n,
        .dry_processed_breakers = breakers,
        .dry_repeat_count = repeat_count,
        .dry_max_token_repeat = .empty,
        .last_tokens = core.RingBuffer(Token).init(@intCast(dry_penalty_last_n)),
    }));
}

/// Ports `llama_sampler_init_dry` (llama-sampler.cpp:3639 @5266f24da).
///
/// Each breaker is truncated to 40 bytes and matched against every token's
/// text; tails are cut to 20 tokens.
///
/// Parameters:
/// - `vocab`: the vocabulary the breakers are matched against.
/// - `dry_multiplier`: the penalty scale; 0 disables it.
/// - `dry_base`: the penalty base; below 1 disables it.
/// - `dry_allowed_length`: repeats up to this long go unpenalised.
/// - `dry_penalty_last_n_in`: the window; negative counts as 0, which
///   disables it.
/// - `seq_breakers`: the restart strings; null and empty entries are skipped
///   with a warning.
/// - `num_breakers`: how many.
///
/// Return: the sampler, or the `?dry` placeholder when disabled.
pub export fn llama_sampler_init_dry(
    vocab: ?*const c.struct_llama_vocab,
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: i32,
    dry_penalty_last_n_in: i32,
    seq_breakers: [*c]const [*c]const u8,
    num_breakers: usize,
) callconv(.c) [*c]Sampler {
    const dry_penalty_last_n = libcxx.max(i32, dry_penalty_last_n_in, 0);
    var processed_breakers: Breakers = .empty;
    const MAX_CHAR_LEN = 40;
    const MAX_SEQ_LEN = 20;

    const dry_enabled = dry_multiplier != 0.0 and dry_base >= 1.0 and dry_penalty_last_n != 0;

    if (!dry_enabled) return core.initEmpty("?dry");

    if (dry_enabled and seq_breakers != null and num_breakers > 0) {
        for (0..num_breakers) |i| {
            if (seq_breakers[i] == null or std.mem.len(seq_breakers[i]) == 0) {
                impl.logWarn("skipping null or empty DRY sequence breaker at index %zu\n", .{i});
                continue;
            }

            var sequence_break: []const u8 = std.mem.span(seq_breakers[i]);

            if (sequence_break.len > MAX_CHAR_LEN) {
                impl.logWarn("truncating DRY sequence breaker to %d characters\n", .{@as(c_int, MAX_CHAR_LEN)});
                sequence_break = sequence_break[0..MAX_CHAR_LEN];
            }

            getOverlappingTokenSequences(vocab, sequence_break, &processed_breakers, MAX_SEQ_LEN);
        }
    }

    return initDry(dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, processed_breakers);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
