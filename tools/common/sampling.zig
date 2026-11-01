//! `common_sampler`: the sampler a tool or the server draws its tokens from
//! -- the chain, the grammar beside it, the reasoning budget, and the tokens
//! it has accepted.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/sampling.cpp` (v0.4.1, `b29c606e2`): the ring
//! buffer, `common_sampler` and every function the tools and the server call,
//! with `regex_escape` from `llama.cpp/common/common.cpp`. Each declaration
//! names the C++ it replaces and the line it began at.
//!
//! Not ported: `common_perf_print`, which no ported tool or the server
//! calls, and `llama_sampler_init_llg` (no llguidance in the reference
//! build: a `%llguidance` grammar aborts, as upstream's does).
//!
//! # Exceptions
//!
//! `common_sampler_init` throws `std::invalid_argument` and
//! `std::runtime_error`; here they are `error.InvalidArgument` and
//! `error.Runtime` with the text in `arg.what()`. The grammar prefill's
//! `catch` cannot be reproduced: the port's libllama aborts where the C++
//! grammar throws, so the C++'s `LOG_ERR` before its rethrow never prints
//! here, and both end in an uncaught exception's abort.
//!
//! # Parameters
//!
//! The C++ keeps a copy of `common_params_sampling`; only `grammar_lazy`
//! and `no_perf` are ever read back, so only those are kept.

const std = @import("std");
const c = @import("tools_h");
const params_mod = @import("params.zig");
const log = @import("log.zig");
const arg = @import("arg.zig");
const common = @import("common.zig");
const reasoning_budget = @import("reasoning_budget.zig");
const libcxx = @import("llama").libcxx;

const SamplingParams = params_mod.Sampling;
const SamplerType = params_mod.SamplerType;

const gpa = std.heap.c_allocator;

fn assertC(ok: bool, comptime line: c_int, comptime expr: [*:0]const u8) void {
    if (!ok) _ = c.ggml_abort("sampling.cpp", line, "GGML_ASSERT(%s) failed", expr);
}

fn oom() noreturn {
    _ = c.ggml_abort("sampling.cpp", 0, "out of memory");
    unreachable;
}

/// A C++ `throw std::runtime_error(msg)` nothing catches.
fn throwUncaught(comptime line: c_int, msg: [*:0]const u8) noreturn {
    _ = c.ggml_abort("sampling.cpp", line, "terminating due to uncaught exception of type std::runtime_error: %s", msg);
    unreachable;
}

/// Ports `ring_buffer` (sampling.cpp:21 @b29c606e2): a fixed-capacity deque.
pub fn RingBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        capacity: usize = 0,
        sz: usize = 0,
        first: usize = 0,
        pos: usize = 0,
        data: []T = &.{},

        pub fn init(a: std.mem.Allocator, cap: usize) !Self {
            const data = try a.alloc(T, cap);
            @memset(data, std.mem.zeroes(T));
            return .{ .capacity = cap, .data = data };
        }

        pub fn deinit(self: *Self, a: std.mem.Allocator) void {
            a.free(self.data);
        }

        /// A copy, as the C++ copy constructor makes one.
        pub fn clone(self: *const Self, a: std.mem.Allocator) !Self {
            var r = self.*;
            r.data = try a.dupe(T, self.data);
            return r;
        }

        /// Ports `front` (sampling.cpp:24 @b29c606e2), a member of `ring_buffer`.
        pub fn front(self: *const Self) T {
            if (self.sz == 0) throwUncaught(26, "ring buffer is empty");
            return self.data[self.first];
        }

        /// Ports `push_back` (sampling.cpp:52 @b29c606e2), a member of
        /// `ring_buffer`: the oldest element goes when full.
        pub fn pushBack(self: *Self, value: T) void {
            if (self.sz == self.capacity) {
                // advance the start when buffer is full
                self.first = (self.first + 1) % self.capacity;
            } else {
                self.sz += 1;
            }
            self.data[self.pos] = value;
            self.pos = (self.pos + 1) % self.capacity;
        }

        /// Ports `pop_front` (sampling.cpp:63 @b29c606e2), a member of `ring_buffer`.
        pub fn popFront(self: *Self) T {
            if (self.sz == 0) throwUncaught(65, "ring buffer is empty");
            const value = self.data[self.first];
            self.first = (self.first + 1) % self.capacity;
            self.sz -= 1;
            return value;
        }

        /// Ports `rat` (sampling.cpp:73 @b29c606e2), a member of
        /// `ring_buffer`: the `i`th element from the back.
        pub fn rat(self: *const Self, i: usize) T {
            if (i >= self.sz) throwUncaught(75, "ring buffer: index out of bounds");
            return self.data[(self.first + self.sz - i - 1) % self.capacity];
        }

        /// Ports `to_vector` (sampling.cpp:80 @b29c606e2), a member of `ring_buffer`.
        ///
        /// Return: the elements oldest first, the allocator's.
        pub fn toVector(self: *const Self, a: std.mem.Allocator) ![]T {
            const result = try a.alloc(T, self.sz);
            for (0..self.sz) |i| result[i] = self.data[(self.first + i) % self.capacity];
            return result;
        }

        /// Ports `clear` (sampling.cpp:89 @b29c606e2), a member of
        /// `ring_buffer`: here only reset the status of the buffer.
        pub fn clear(self: *Self) void {
            self.sz = 0;
            self.first = 0;
            self.pos = 0;
        }

        /// Ports `empty` (sampling.cpp:96 @b29c606e2), a member of `ring_buffer`.
        pub fn empty(self: *const Self) bool {
            return self.sz == 0;
        }

        /// Ports `size` (sampling.cpp:100 @b29c606e2), a member of `ring_buffer`.
        pub fn size(self: *const Self) usize {
            return self.sz;
        }
    };
}

/// Ports `print` (sampling.cpp:171 @b29c606e2), a member of
/// `common_params_sampling`: the parameters as the tools and the server log
/// them.
///
/// Return: the text, the allocator's.
pub fn paramsPrint(a: std.mem.Allocator, p: *const SamplingParams) ![]u8 {
    var result: [1024]u8 = undefined;
    const n = snprintf(&result, result.len, "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n" ++
        "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n" ++
        "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n" ++
        "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f", p.penalty_last_n, @as(f64, p.penalty_repeat), @as(f64, p.penalty_freq), @as(f64, p.penalty_present), @as(f64, p.dry_multiplier), @as(f64, p.dry_base), p.dry_allowed_length, p.dry_penalty_last_n, p.top_k, @as(f64, p.top_p), @as(f64, p.min_p), @as(f64, p.xtc_probability), @as(f64, p.xtc_threshold), @as(f64, p.typ_p), @as(f64, p.top_n_sigma), @as(f64, p.temp), p.mirostat, @as(f64, p.mirostat_eta), @as(f64, p.mirostat_tau), @as(f64, p.adaptive_target), @as(f64, p.adaptive_decay));
    return a.dupe(u8, result[0..@min(@as(usize, @intCast(n)), result.len - 1)]);
}

extern fn snprintf(buf: [*]u8, size: usize, format: [*:0]const u8, ...) c_int;

/// Ports `regex_escape` (common/common.cpp:524 @b29c606e2): every one of
/// `.^$|()*+?[]{}\` behind a backslash. `std::regex_replace` over that one
/// character class does exactly this, byte by byte.
fn regexEscape(a: std.mem.Allocator, s: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    for (s) |ch| {
        if (std.mem.indexOfScalar(u8, ".^$|()*+?[]{}\\", ch) != null) try out.append(a, '\\');
        try out.append(a, ch);
    }
    return out.toOwnedSlice(a);
}

/// Ports `common_grammar_needs_prefill` (common/common.h:218 @b29c606e2):
/// only output-format and tool-call grammars are prefilled.
pub fn grammarNeedsPrefill(t: params_mod.GrammarType) bool {
    return t == .output_format or t == .tool_calls;
}

/// What `common_sampler_init` throws.
pub const InitError = error{ InvalidArgument, Runtime, OutOfMemory, Throw };

/// Ports `common_sampler` (sampling.cpp:111 @b29c606e2).
pub const Sampler = struct {
    allocator: std.mem.Allocator,
    grammar_lazy: bool = false,
    no_perf: bool = false,
    grmr: ?*c.llama_sampler = null,
    rbudget: ?*c.llama_sampler = null,
    chain: *c.llama_sampler,
    prev: RingBuffer(c.llama_token),
    cur: std.ArrayList(c.llama_token_data) = .empty,
    cur_p: c.llama_token_data_array = .{ .data = null, .size = 0, .selected = -1, .sorted = false },
    t_total_us: i64 = 0,

    /// Ports `tm` (sampling.cpp:164 @b29c606e2), a member of
    /// `common_sampler`: a `common_time_meas` over `t_total_us`. End it
    /// with `tmEnd`.
    fn tm(self: *const Sampler) i64 {
        return if (self.no_perf) -1 else c.ggml_time_us();
    }

    /// The `common_time_meas` destructor (common/common.cpp:68 @b29c606e2).
    fn tmEnd(self: *Sampler, t_start_us: i64) void {
        if (t_start_us >= 0) self.t_total_us += c.ggml_time_us() - t_start_us;
    }

    /// Ports `common_sampler_free` (sampling.cpp:440 @b29c606e2).
    pub fn destroy(self: *Sampler) void {
        c.llama_sampler_free(self.grmr);
        c.llama_sampler_free(self.rbudget);
        c.llama_sampler_free(self.chain);
        self.prev.deinit(self.allocator);
        self.cur.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Ports `common_sampler_get` (sampling.cpp:586 @b29c606e2): the chain.
    pub fn get(self: *const Sampler) *c.llama_sampler {
        return self.chain;
    }

    /// Ports `reset` (sampling.cpp:124 @b29c606e2), a member of
    /// `common_sampler`, which `common_sampler_reset` (sampling.cpp:501
    /// @b29c606e2) calls.
    pub fn reset(self: *Sampler) void {
        self.prev.clear();
        c.llama_sampler_reset(self.chain);
    }

    /// Ports `set_logits` (sampling.cpp:130 @b29c606e2), a member of
    /// `common_sampler`: the candidates, from a backend sampler's output or
    /// the raw logits.
    fn setLogits(self: *Sampler, ctx: ?*c.llama_context, idx: i32) void {
        const sampled_probs = c.llama_get_sampled_probs_ith(ctx, idx);
        const sampled_logits = c.llama_get_sampled_logits_ith(ctx, idx);
        const sampled_ids = c.llama_get_sampled_candidates_ith(ctx, idx);

        const model = c.llama_get_model(ctx);
        const vocab = c.llama_model_get_vocab(model);

        const n_vocab = c.llama_vocab_n_tokens(vocab);

        const a = self.allocator;
        if (sampled_probs != null) {
            const n = c.llama_get_sampled_probs_count_ith(ctx, idx);
            self.cur.resize(a, n) catch oom();
            for (0..n) |i| self.cur.items[i] = .{ .id = sampled_ids[i], .logit = sampled_logits[i], .p = sampled_probs[i] };
        } else if (sampled_logits != null) {
            const n = c.llama_get_sampled_logits_count_ith(ctx, idx);
            self.cur.resize(a, n) catch oom();
            for (0..n) |i| self.cur.items[i] = .{ .id = sampled_ids[i], .logit = sampled_logits[i], .p = 0.0 };
        } else {
            const logits = c.llama_get_logits_ith(ctx, idx);
            assertC(logits != null, 154, "logits != nullptr");
            self.cur.resize(a, @intCast(n_vocab)) catch oom();
            var token_id: c.llama_token = 0;
            while (token_id < n_vocab) : (token_id += 1) {
                const k: usize = @intCast(token_id);
                self.cur.items[k] = .{ .id = token_id, .logit = logits[k], .p = 0.0 };
            }
        }

        self.cur_p = .{ .data = self.cur.items.ptr, .size = self.cur.items.len, .selected = -1, .sorted = false };
    }

    /// Ports `grammar_should_apply` (sampling.cpp:452 @b29c606e2).
    fn grammarShouldApply(self: *const Sampler) bool {
        if (self.grmr == null) return false;
        if (self.rbudget == null) return true;
        if (self.grammar_lazy) {
            // if grammar is lazy, only apply when reasoning budget is not active
            const state = reasoning_budget.getState(self.rbudget);
            return state == .idle or state == .done;
        }
        return true;
    }

    /// Ports `common_sampler_accept` (sampling.cpp:467 @b29c606e2): with
    /// `is_generated`, the token goes to the chain, the reasoning budget and
    /// the grammar; otherwise to the chain alone.
    pub fn accept(self: *Sampler, token: c.llama_token, is_generated: bool) void {
        const t0 = self.tm();
        defer self.tmEnd(t0);

        // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
        const accept_grammar = is_generated and self.grammarShouldApply();

        if (self.rbudget != null and is_generated) {
            c.llama_sampler_accept(self.rbudget, token);

            // if done, replay end sequence which may contain a grammar trigger
            const is_done = reasoning_budget.getState(self.rbudget) == .done;
            if (self.grmr != null and !accept_grammar and is_done) {
                if (reasoning_budget.getEndMatch(self.rbudget)) |end_seq| {
                    for (end_seq) |end_token| c.llama_sampler_accept(self.grmr, end_token);
                }
            }
        }

        if (self.grmr != null and accept_grammar) c.llama_sampler_accept(self.grmr, token);

        c.llama_sampler_accept(self.chain, token);

        self.prev.pushBack(token);
    }

    /// Ports `common_sampler_clone` (sampling.cpp:509 @b29c606e2).
    ///
    /// Return: a new sampler with this one's state, freed with `destroy`.
    pub fn clone(self: *const Sampler) *Sampler {
        const a = self.allocator;
        const r = a.create(Sampler) catch oom();
        r.* = .{
            .allocator = a,
            .grammar_lazy = self.grammar_lazy,
            .no_perf = self.no_perf,
            .grmr = c.llama_sampler_clone(self.grmr),
            .rbudget = c.llama_sampler_clone(self.rbudget),
            .chain = c.llama_sampler_clone(self.chain).?,
            .prev = self.prev.clone(a) catch oom(),
            .cur = .empty,
            .cur_p = self.cur_p,
        };
        r.cur.appendSlice(a, self.cur.items) catch oom();
        // the copied `cur_p` still points into this sampler's `cur`, as the
        // C++'s member-wise copy leaves it
        return r;
    }

    /// Ports `common_sampler_copy` (sampling.cpp:521 @b29c606e2): `src`'s
    /// state into `dst`, which must have the same grammar and budget shape.
    pub fn copy(src: ?*const Sampler, dst: ?*Sampler) void {
        if (src == null or dst == null or @intFromPtr(src.?) == @intFromPtr(dst.?)) return;
        const s = src.?;
        const d = dst.?;

        assertC((s.grmr == null) == (d.grmr == null), 526, "(src->grmr == nullptr) == (dst->grmr == nullptr)");
        assertC((s.rbudget == null) == (d.rbudget == null), 527, "(src->rbudget == nullptr) == (dst->rbudget == nullptr)");

        c.llama_sampler_copy(s.grmr, d.grmr);
        c.llama_sampler_copy(s.rbudget, d.rbudget);
        c.llama_sampler_copy(s.chain, d.chain);

        d.grammar_lazy = s.grammar_lazy;
        d.no_perf = s.no_perf;
        d.prev.deinit(d.allocator);
        d.prev = s.prev.clone(d.allocator) catch oom();
        d.cur.clearRetainingCapacity();
        d.cur.appendSlice(d.allocator, s.cur.items) catch oom();
        d.cur_p = s.cur_p;
        d.cur_p.data = if (s.cur_p.data != null) d.cur.items.ptr else null; // re-point to dst's buffer
        d.t_total_us = s.t_total_us;
    }

    /// Ports `common_sampler_sample` (sampling.cpp:594 @b29c606e2):
    /// grammar-based rejection sampling over the chain, after the reasoning
    /// budget.
    ///
    /// Return: the token, which `cur_p` records as selected.
    pub fn sample(self: *Sampler, ctx: ?*c.llama_context, idx: i32, grammar_first: bool) !c.llama_token {
        c.llama_synchronize(ctx);

        // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
        const t0 = self.tm();
        defer self.tmEnd(t0);

        self.setLogits(ctx, idx);

        // Check if a backend sampler has already sampled a token in which case we
        // return that token id directly.
        {
            const id = c.llama_get_sampled_token_ith(ctx, idx);
            if (id != c.LLAMA_TOKEN_NULL) {
                log.dbg("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", .{ @as([*:0]const u8, "common_sampler_sample"), id });

                assertC(self.grmr == null, 617, "!gsmpl->grmr    && \"using grammar in combination with backend sampling is not supported\"");
                assertC(self.rbudget == null, 618, "!gsmpl->rbudget && \"using reasoning budget in combination with backend sampling is not supported\"");

                for (0..self.cur_p.size) |i| {
                    if (self.cur_p.data[i].id == id) {
                        self.cur_p.selected = @intCast(i);
                        break;
                    }
                }
                return id;
            }
        }

        // apply reasoning budget first
        c.llama_sampler_apply(self.rbudget, &self.cur_p);

        if (grammar_first and self.grammarShouldApply()) c.llama_sampler_apply(self.grmr, &self.cur_p);

        c.llama_sampler_apply(self.chain, &self.cur_p);

        var id = self.cur_p.data[@intCast(self.cur_p.selected)].id;

        if (grammar_first or !self.grammarShouldApply()) return id;

        // check if it the sampled token fits the grammar (grammar-based rejection sampling)
        {
            var single_token_data: c.llama_token_data = .{ .id = id, .logit = 1.0, .p = 0.0 };
            var single_token_data_array: c.llama_token_data_array = .{ .data = &single_token_data, .size = 1, .selected = -1, .sorted = false };

            c.llama_sampler_apply(self.grmr, &single_token_data_array);

            const is_valid = single_token_data_array.data[0].logit != -std.math.inf(f32);
            if (is_valid) return id;
        }

        // resampling:
        // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
        self.setLogits(ctx, idx);

        c.llama_sampler_apply(self.rbudget, &self.cur_p);

        if (self.grammarShouldApply()) c.llama_sampler_apply(self.grmr, &self.cur_p);

        c.llama_sampler_apply(self.chain, &self.cur_p);

        assertC(self.cur_p.selected != -1, 671, "cur_p.selected != -1 && \"no selected token during sampling - check your sampling configuration\"");

        id = self.cur_p.data[@intCast(self.cur_p.selected)].id;

        return id;
    }

    /// Ports `common_sampler_sample_and_accept_n` (sampling.cpp:678
    /// @b29c606e2), the overload taking output indices: samples and accepts
    /// at each index while the samples agree with `draft`.
    ///
    /// Parameters:
    /// - `idxs`: one more output index than `draft` has tokens.
    /// - `draft`: the drafted tokens.
    ///
    /// Return: at least one token, at most `idxs.len`; the allocator's.
    pub fn sampleAndAcceptNIdxs(self: *Sampler, a: std.mem.Allocator, ctx: ?*c.llama_context, idxs: []const i32, draft: []const c.llama_token, grammar_first: bool) ![]c.llama_token {
        assertC(idxs.len == draft.len + 1, 679, "idxs.size() == draft.size() + 1 && \"idxs.size() must be draft.size() + 1\"");

        var result: std.ArrayList(c.llama_token) = .empty;
        try result.ensureTotalCapacityPrecise(a, idxs.len);

        var i: usize = 0;
        while (i < draft.len) : (i += 1) {
            const id = try self.sample(ctx, idxs[i], grammar_first);

            self.accept(id, true);

            result.appendAssumeCapacity(id);

            if (draft[i] != id) break;
        }

        if (i == draft.len) {
            const id = try self.sample(ctx, idxs[i], grammar_first);

            self.accept(id, true);

            result.appendAssumeCapacity(id);
        }

        return result.toOwnedSlice(a);
    }

    /// Ports `common_sampler_sample_and_accept_n` (sampling.cpp:708
    /// @b29c606e2), the overload that assumes `idxs == [0, 1, ...,
    /// draft.size()]`.
    ///
    /// Return: the accepted tokens, the allocator's.
    pub fn sampleAndAcceptN(self: *Sampler, a: std.mem.Allocator, ctx: ?*c.llama_context, draft: []const c.llama_token, grammar_first: bool) ![]c.llama_token {
        const idxs = try a.alloc(i32, draft.len + 1);
        defer a.free(idxs);
        for (idxs, 0..) |*x, i| x.* = @intCast(i);

        return self.sampleAndAcceptNIdxs(a, ctx, idxs, draft, grammar_first);
    }

    /// Ports `common_sampler_get_seed` (sampling.cpp:717 @b29c606e2).
    pub fn getSeed(self: *const Sampler) u32 {
        return c.llama_sampler_get_seed(self.chain);
    }

    /// Ports `common_sampler_reasoning_budget_force` (sampling.cpp:721
    /// @b29c606e2): the reasoning budget, if any, begins forcing its end
    /// sequence now.
    ///
    /// Return: whether it did.
    pub fn reasoningBudgetForce(self: *Sampler) bool {
        return reasoning_budget.force(self.rbudget);
    }

    /// Ports `common_sampler_get_candidates` (sampling.cpp:731 @b29c606e2):
    /// the current candidates, sorted by descending probability with
    /// libc++'s `std::sort` when `do_sort` and not already sorted.
    ///
    /// Return: the sampler's own array, valid until the next sample.
    pub fn getCandidates(self: *Sampler, do_sort: bool) *c.llama_token_data_array {
        const t0 = self.tm();
        defer self.tmEnd(t0);

        const res = &self.cur_p;

        if (do_sort and !res.sorted) {
            // remember the selected token before sorting
            const id = res.data[@intCast(res.selected)].id;

            libcxx.sort(c.llama_token_data, res.data[0..res.size], {}, struct {
                fn gt(_: void, x: c.llama_token_data, y: c.llama_token_data) bool {
                    return x.p > y.p;
                }
            }.gt);

            // restore the selected token after sorting
            for (0..res.size) |i| {
                if (res.data[i].id == id) {
                    res.selected = @intCast(i);
                    break;
                }
            }

            res.sorted = true;
        }

        return res;
    }

    /// Ports `common_sampler_last` (sampling.cpp:758 @b29c606e2): the last
    /// accepted token.
    pub fn last(self: *const Sampler) c.llama_token {
        return self.prev.rat(0);
    }

    /// Ports `common_sampler_print` (sampling.cpp:762 @b29c606e2): the chain
    /// as `logits -> name -> name ...`.
    ///
    /// Return: the text, the allocator's.
    pub fn print(self: *const Sampler, a: std.mem.Allocator) ![]u8 {
        var result: std.ArrayList(u8) = .empty;
        try result.appendSlice(a, "logits ");

        var i: c_int = 0;
        while (i < c.llama_sampler_chain_n(self.chain)) : (i += 1) {
            const smpl = c.llama_sampler_chain_get(self.chain, i);
            try result.appendSlice(a, "-> ");
            try result.appendSlice(a, std.mem.span(c.llama_sampler_name(smpl)));
            try result.append(a, ' ');
        }

        return result.toOwnedSlice(a);
    }

    /// Ports `common_sampler_prev_str` (sampling.cpp:774 @b29c606e2): the
    /// last `n` accepted tokens' text.
    ///
    /// Return: the text, the allocator's.
    pub fn prevStr(self: *const Sampler, a: std.mem.Allocator, ctx_main: ?*c.llama_context, n_in: i32) ![]u8 {
        const n = @min(n_in, @as(i32, @intCast(self.prev.size())));

        if (n <= 0) return a.alloc(u8, 0);

        var result: std.ArrayList(u8) = .empty;
        try result.ensureTotalCapacity(a, @intCast(8 * n)); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

        const vocab = c.llama_model_get_vocab(c.llama_get_model(ctx_main));
        var i: i32 = n - 1;
        while (i >= 0) : (i -= 1) {
            const id = self.prev.rat(@intCast(i));

            assertC(id != c.LLAMA_TOKEN_NULL, 787, "id != LLAMA_TOKEN_NULL && \"null token in the sampling history - should not happen\"");

            const piece = try common.tokenToPiece(a, vocab, id, true);
            defer a.free(piece[0 .. piece.len + 1]);
            try result.appendSlice(a, piece);
        }

        return result.toOwnedSlice(a);
    }
};

/// Ports `common_sampler_init` (sampling.cpp:187 @b29c606e2): the grammar,
/// the reasoning budget and the chain `params` describes.
///
/// Parameters:
/// - `model`: the model sampled from.
/// - `params`: the parameters; `backend_sampling` is turned off when a
///   grammar or a reasoning budget rules it out, as the C++ does.
///
/// Return: the sampler, which the caller owns (`Sampler.destroy`).
pub fn init(model: ?*const c.llama_model, params: *SamplingParams) InitError!*Sampler {
    if (!std.math.isFinite(params.penalty_repeat) or params.penalty_repeat <= 0.0 or !std.math.isFinite(1.0 / params.penalty_repeat)) {
        _ = arg.throwStr("penalty_repeat must be finite and greater than 0") catch {};
        return error.InvalidArgument;
    }
    if (!std.math.isFinite(params.penalty_freq)) {
        _ = arg.throwStr("penalty_freq must be finite") catch {};
        return error.InvalidArgument;
    }
    if (!std.math.isFinite(params.penalty_present)) {
        _ = arg.throwStr("penalty_present must be finite") catch {};
        return error.InvalidArgument;
    }

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const vocab = c.llama_model_get_vocab(model);
    var lparams = c.llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    var grmr: ?*c.llama_sampler = null;
    var rbudget: ?*c.llama_sampler = null;
    const chain = c.llama_sampler_chain_init(lparams);

    var samplers: std.ArrayList(*c.llama_sampler) = .empty;

    const grammar_str = params.grammar;
    if (std.mem.startsWith(u8, grammar_str, "%llguidance")) {
        _ = c.ggml_abort("sampling.cpp", 217, "llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
    } else {
        var trigger_patterns: std.ArrayList([:0]const u8) = .empty;
        var trigger_tokens: std.ArrayList(c.llama_token) = .empty;
        for (params.grammar_triggers.items) |trigger| {
            switch (trigger.type) {
                .word => {
                    const word = trigger.value;
                    try trigger_patterns.append(arena, try arena.dupeSentinel(u8, try regexEscape(arena, word), 0));
                },
                .pattern => try trigger_patterns.append(arena, try arena.dupeSentinel(u8, trigger.value, 0)),
                .pattern_full => {
                    const pattern = trigger.value;
                    var anchored: []const u8 = "^$";
                    if (pattern.len != 0) {
                        anchored = try std.mem.concat(arena, u8, &.{
                            if (pattern[0] != '^') "^" else "",
                            pattern,
                            if (pattern[pattern.len - 1] != '$') "$" else "",
                        });
                    }
                    try trigger_patterns.append(arena, try arena.dupeSentinel(u8, anchored, 0));
                },
                .token => {
                    const token = trigger.token;
                    try trigger_tokens.append(arena, token);
                },
                _ => _ = c.ggml_abort("sampling.cpp", 254, "GGML_ASSERT(%s) failed", "false && \"unknown trigger type\""),
            }
        }

        var trigger_patterns_c: std.ArrayList([*c]const u8) = .empty;
        try trigger_patterns_c.ensureTotalCapacity(arena, trigger_patterns.items.len);
        for (trigger_patterns.items) |regex| trigger_patterns_c.appendAssumeCapacity(regex.ptr);

        if (grammar_str.len != 0) {
            const g = try arena.dupeSentinel(u8, grammar_str, 0);
            if (params.grammar_lazy) {
                grmr = c.llama_sampler_init_grammar_lazy_patterns(vocab, g.ptr, "root", trigger_patterns_c.items.ptr, trigger_patterns_c.items.len, trigger_tokens.items.ptr, trigger_tokens.items.len);
            } else {
                grmr = c.llama_sampler_init_grammar(vocab, g.ptr, "root");
            }
        }
    }
    if (grmr == null and grammar_str.len != 0) {
        c.llama_sampler_free(chain);
        _ = arg.throwStr("failed to parse grammar") catch {};
        return error.Runtime;
    }

    // Compute prefill tokens from the generation prompt
    var prefill_tokens: std.ArrayList(c.llama_token) = .empty;
    if (params.generation_prompt.len != 0) {
        assertC(vocab != null, 281, "vocab != nullptr");
        const tokens = try common.tokenize(arena, vocab, params.generation_prompt, false, true);
        for (tokens, 0..) |tok, i| {
            const piece = try common.tokenToPiece(arena, vocab, tok, true);
            if (i == 0 and isspace(piece[0]) and !isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            log.dbg("%s: prefill token: %d = %s\n", .{ @as([*:0]const u8, "common_sampler_init"), tok, piece.ptr });
            try prefill_tokens.append(arena, tok);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr != null and !params.grammar_lazy and grammarNeedsPrefill(params.grammar_type)) {
        for (prefill_tokens.items) |token| {
            c.llama_sampler_accept(grmr, token);
            log.dbg("%s: grammar accepted prefill token (%d)\n", .{ @as([*:0]const u8, "common_sampler_init"), token });
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (params.reasoning_budget_start.items.len != 0 and params.reasoning_budget_end.items.len != 0 and (params.grammar_lazy or params.reasoning_budget_tokens >= 0 or params.reasoning_control)) {
        const ends = try arena.alloc([]const c.llama_token, params.reasoning_budget_end.items.len);
        for (params.reasoning_budget_end.items, ends) |e, *d| d.* = e.items;
        rbudget = try reasoning_budget.init(vocab, &.{params.reasoning_budget_start.items}, ends, params.reasoning_budget_forced.items, if (params.reasoning_budget_tokens < 0) std.math.maxInt(i32) else params.reasoning_budget_tokens, .idle);

        for (prefill_tokens.items) |token| {
            c.llama_sampler_accept(rbudget, token);
            log.dbg("%s: reasoning-budget accepted prefill token (%d)\n", .{ @as([*:0]const u8, "common_sampler_init"), token });
        }
    }

    // logit bias: user biases + model suppress tokens (-INFINITY)
    {
        var merged: std.ArrayList(c.llama_logit_bias) = .empty;
        try merged.appendSlice(arena, params.logit_bias.items);

        var n_suppress: i32 = 0;
        const suppress = c.llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        var i: i32 = 0;
        while (i < n_suppress) : (i += 1) try merged.append(arena, .{ .token = suppress[@intCast(i)], .bias = -std.math.inf(f32) });

        if (merged.items.len != 0) {
            try samplers.append(arena, c.llama_sampler_init_logit_bias(c.llama_vocab_n_tokens(vocab), @intCast(merged.items.len), merged.items.ptr).?);
        }
    }

    if (params.mirostat == 0) {
        var use_adaptive_p = false; // see below

        for (params.samplers.items) |cnstr| {
            const s: ?*c.llama_sampler = switch (cnstr) {
                .dry => blk: {
                    var c_breakers: std.ArrayList([*c]const u8) = .empty;
                    try c_breakers.ensureTotalCapacity(arena, params.dry_sequence_breakers.items.len);
                    for (params.dry_sequence_breakers.items) |str| c_breakers.appendAssumeCapacity((try arena.dupeSentinel(u8, str, 0)).ptr);
                    break :blk c.llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.items.ptr, c_breakers.items.len);
                },
                .top_k => c.llama_sampler_init_top_k(params.top_k),
                .top_p => c.llama_sampler_init_top_p(params.top_p, @intCast(params.min_keep)),
                .top_n_sigma => c.llama_sampler_init_top_n_sigma(params.top_n_sigma),
                .min_p => c.llama_sampler_init_min_p(params.min_p, @intCast(params.min_keep)),
                .xtc => c.llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, @intCast(params.min_keep), params.seed),
                .typical_p => c.llama_sampler_init_typical(params.typ_p, @intCast(params.min_keep)),
                .temperature => c.llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent),
                .infill => c.llama_sampler_init_infill(vocab),
                .penalties => c.llama_sampler_init_penalties(c.llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present),
                .adaptive_p => blk: {
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break :blk null;
                },
                else => blk: {
                    assertC(false, 391, "false && \"unknown sampler type\"");
                    break :blk null;
                },
            };
            if (s) |smpl| try samplers.append(arena, smpl);
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            try samplers.append(arena, c.llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed).?);
        } else {
            // default: sample from distribution
            try samplers.append(arena, c.llama_sampler_init_dist(params.seed).?);
        }
    } else if (params.mirostat == 1) {
        try samplers.append(arena, c.llama_sampler_init_temp(params.temp).?);
        try samplers.append(arena, c.llama_sampler_init_mirostat(c.llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100).?);
    } else if (params.mirostat == 2) {
        try samplers.append(arena, c.llama_sampler_init_temp(params.temp).?);
        try samplers.append(arena, c.llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta).?);
    } else {
        assertC(false, 408, "false && \"unknown mirostat version\"");
    }

    for (samplers.items) |smpl| c.llama_sampler_chain_add(chain, smpl);

    if (grmr != null and params.backend_sampling) {
        log.wrn("%s: backend sampling is not compatible with grammar, disabling\n", .{@as([*:0]const u8, "common_sampler_init")});

        params.backend_sampling = false;
    }

    if (rbudget != null and params.backend_sampling) {
        log.wrn("%s: backend sampling is not compatible with reasoning budget, disabling\n", .{@as([*:0]const u8, "common_sampler_init")});

        params.backend_sampling = false;
    }

    const result = try gpa.create(Sampler);
    result.* = .{
        .allocator = gpa,
        .grammar_lazy = params.grammar_lazy,
        .no_perf = params.no_perf,
        .grmr = grmr,
        .rbudget = rbudget,
        .chain = chain.?,
        .prev = try RingBuffer(c.llama_token).init(gpa, @intCast(@max(32, params.n_prev))),
    };
    return result;
}

fn isspace(ch: u8) bool {
    return ch == ' ' or (ch >= '\t' and ch <= '\r');
}

/// Ports `common_sampler_free` (sampling.cpp:440 @b29c606e2): null-safe.
pub fn free(gsmpl: ?*Sampler) void {
    if (gsmpl) |s| s.destroy();
}

/// Ports `common_sampler_types_from_names` (sampling.cpp:829 @b29c606e2):
/// each name, lower-cased, through the canonical names, their kebab-case and
/// no-dash forms, and three aliases; an unknown one is logged and skipped.
///
/// Return: the types, the allocator's.
pub fn typesFromNames(a: std.mem.Allocator, names: []const []const u8) ![]SamplerType {
    const Entry = struct { []const u8, SamplerType };
    const map = [_]Entry{
        // canonical
        .{ "dry", .dry },                 .{ "top_k", .top_k },           .{ "top_p", .top_p },
        .{ "top_n_sigma", .top_n_sigma }, .{ "typ_p", .typical_p },       .{ "min_p", .min_p },
        .{ "temperature", .temperature }, .{ "xtc", .xtc },               .{ "infill", .infill },
        .{ "penalties", .penalties },     .{ "adaptive_p", .adaptive_p },
        // kebab-case and no-dash aliases of the names with an underscore
        .{ "top-k", .top_k },
        .{ "topk", .top_k },              .{ "top-p", .top_p },           .{ "topp", .top_p },
        .{ "top-n-sigma", .top_n_sigma }, .{ "topnsigma", .top_n_sigma }, .{ "typ-p", .typical_p },
        .{ "typp", .typical_p },          .{ "min-p", .min_p },           .{ "minp", .min_p },
        .{ "adaptive-p", .adaptive_p },   .{ "adaptivep", .adaptive_p },
        // misc. aliases
         .{ "nucleus", .top_p },
        .{ "temp", .temperature },        .{ "typ", .typical_p },
    };

    var samplers: std.ArrayList(SamplerType) = .empty;
    try samplers.ensureTotalCapacity(a, names.len);
    for (names) |nm| {
        const name_lower = try a.dupeSentinel(u8, nm, 0);
        defer a.free(name_lower);
        for (name_lower) |*ch| ch.* = std.ascii.toLower(ch.*);
        const found = for (map) |e| {
            if (std.mem.eql(u8, e[0], name_lower)) break e[1];
        } else null;
        if (found) |t| {
            samplers.appendAssumeCapacity(t);
            continue;
        }
        log.wrn("%s: unable to match sampler by name '%s'\n", .{ @as([*:0]const u8, "common_sampler_types_from_names"), name_lower.ptr });
    }
    return samplers.toOwnedSlice(a);
}

/// Ports `common_sampler_types_from_chars` (sampling.cpp:891 @b29c606e2).
///
/// Return: the types, the allocator's.
pub fn typesFromChars(a: std.mem.Allocator, chars: []const u8) ![]SamplerType {
    const all = [_]SamplerType{ .dry, .top_k, .typical_p, .top_p, .top_n_sigma, .min_p, .temperature, .xtc, .infill, .penalties, .adaptive_p };
    var samplers: std.ArrayList(SamplerType) = .empty;
    try samplers.ensureTotalCapacity(a, chars.len);
    for (chars) |ch| {
        const found = for (all) |t| {
            if (t.toChr() == ch) break t;
        } else null;
        if (found) |t| {
            samplers.appendAssumeCapacity(t);
        } else {
            log.wrn("%s: unable to match sampler by char '%c'\n", .{ @as([*:0]const u8, "common_sampler_types_from_chars"), @as(c_int, ch) });
        }
    }
    return samplers.toOwnedSlice(a);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the ring buffer drops its oldest element when full" {
    var rb = try RingBuffer(i32).init(std.testing.allocator, 3);
    defer rb.deinit(std.testing.allocator);
    for ([_]i32{ 1, 2, 3, 4 }) |v| rb.pushBack(v);
    try std.testing.expectEqual(@as(usize, 3), rb.sz);
    try std.testing.expectEqual(@as(usize, 1), rb.first);
    try std.testing.expectEqual(@as(i32, 4), rb.data[0]);
    try std.testing.expectEqual(@as(i32, 4), rb.rat(0));
    try std.testing.expectEqual(@as(i32, 2), rb.rat(2));
    try std.testing.expectEqual(@as(i32, 2), rb.popFront());
}

test "regexEscape escapes the class regex_escape replaces" {
    const r = try regexEscape(std.testing.allocator, "a.b(c)[d]{e}|f*g+h?^$\\");
    defer std.testing.allocator.free(r);
    try std.testing.expectEqualStrings("a\\.b\\(c\\)\\[d\\]\\{e\\}\\|f\\*g\\+h\\?\\^\\$\\\\", r);
}
