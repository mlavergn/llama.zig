//! `common_sampler`: the sampler a tool draws its tokens from -- the chain,
//! the grammar beside it, and the tokens it has accepted.
//!
//! # Provenance
//!
//! Ports `ring_buffer`, `common_sampler`, `common_sampler_free`,
//! `grammar_should_apply`, `common_sampler_accept`, `common_sampler_reset`
//! and `common_sampler_sample` from `llama.cpp/common/sampling.cpp` at
//! v0.4.1 (`b29c606e2`). The chain itself is built by `init.zig`'s port of
//! `common_sampler_init`. Each declaration names the C++ it replaces and the
//! line it began at.
//!
//! # The reasoning budget
//!
//! `common_sampler_init` builds its reasoning-budget sampler only when
//! `reasoning_budget_start` is set, which only the chat front ends do; no
//! ported tool reaches one, so `rbudget` is always null here and every
//! `llama_sampler_apply(rbudget, …)` the C++ makes is the no-op libllama
//! makes of a null sampler.

const std = @import("std");
const c = @import("tools_h");

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

        /// `push_back`: the oldest element goes when full.
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

        /// `clear`: here only reset the status of the buffer.
        pub fn clear(self: *Self) void {
            self.sz = 0;
            self.first = 0;
            self.pos = 0;
        }
    };
}

/// Ports `common_sampler` (sampling.cpp:111 @b29c606e2).
pub const Sampler = struct {
    allocator: std.mem.Allocator,
    grammar_lazy: bool = false,
    grmr: ?*c.llama_sampler = null,
    chain: *c.llama_sampler,
    prev: RingBuffer(c.llama_token),
    cur: std.ArrayList(c.llama_token_data) = .empty,
    cur_p: c.llama_token_data_array = .{ .data = null, .size = 0, .selected = -1, .sorted = false },

    /// The tail of `common_sampler_init`, which news the `common_sampler`
    /// (sampling.cpp:427 @b29c606e2) around a chain and grammar already built.
    ///
    /// Parameters:
    /// - `allocator`: owns the ring buffer and the candidate list.
    /// - `n_prev`: `params.n_prev`; the buffer holds at least 32.
    pub fn create(allocator: std.mem.Allocator, chain: *c.llama_sampler, grmr: ?*c.llama_sampler, n_prev: i32, grammar_lazy: bool) !*Sampler {
        const self = try allocator.create(Sampler);
        self.* = .{
            .allocator = allocator,
            .grammar_lazy = grammar_lazy,
            .grmr = grmr,
            .chain = chain,
            .prev = try RingBuffer(c.llama_token).init(allocator, @intCast(@max(32, n_prev))),
        };
        return self;
    }

    /// Ports `common_sampler_free` (sampling.cpp:440 @b29c606e2).
    pub fn destroy(self: *Sampler) void {
        c.llama_sampler_free(self.grmr);
        c.llama_sampler_free(self.chain);
        self.prev.deinit(self.allocator);
        self.cur.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// `common_sampler_get`: the chain.
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
    fn setLogits(self: *Sampler, ctx: ?*c.llama_context, idx: i32) !void {
        const sampled_probs = c.llama_get_sampled_probs_ith(ctx, idx);
        const sampled_logits = c.llama_get_sampled_logits_ith(ctx, idx);
        const sampled_ids = c.llama_get_sampled_candidates_ith(ctx, idx);

        const model = c.llama_get_model(ctx);
        const vocab = c.llama_model_get_vocab(model);

        const n_vocab = c.llama_vocab_n_tokens(vocab);

        const a = self.allocator;
        if (sampled_probs != null) {
            const n = c.llama_get_sampled_probs_count_ith(ctx, idx);
            try self.cur.resize(a, n);
            for (0..n) |i| self.cur.items[i] = .{ .id = sampled_ids[i], .logit = sampled_logits[i], .p = sampled_probs[i] };
        } else if (sampled_logits != null) {
            const n = c.llama_get_sampled_logits_count_ith(ctx, idx);
            try self.cur.resize(a, n);
            for (0..n) |i| self.cur.items[i] = .{ .id = sampled_ids[i], .logit = sampled_logits[i], .p = 0.0 };
        } else {
            const logits = c.llama_get_logits_ith(ctx, idx);
            if (logits == null) _ = c.ggml_abort("sampling.cpp", 154, "GGML_ASSERT(%s) failed", "logits != nullptr");
            try self.cur.resize(a, @intCast(n_vocab));
            var token_id: c.llama_token = 0;
            while (token_id < n_vocab) : (token_id += 1) {
                const k: usize = @intCast(token_id);
                self.cur.items[k] = .{ .id = token_id, .logit = logits[k], .p = 0.0 };
            }
        }

        self.cur_p = .{ .data = self.cur.items.ptr, .size = self.cur.items.len, .selected = -1, .sorted = false };
    }

    /// Ports `grammar_should_apply` (sampling.cpp:452 @b29c606e2): with no
    /// reasoning budget, whenever there is a grammar.
    fn grammarShouldApply(self: *const Sampler) bool {
        return self.grmr != null;
    }

    /// Ports `common_sampler_accept` (sampling.cpp:467 @b29c606e2).
    pub fn accept(self: *Sampler, token: c.llama_token, is_generated: bool) void {
        const accept_grammar = is_generated and self.grammarShouldApply();
        if (self.grmr != null and accept_grammar) c.llama_sampler_accept(self.grmr, token);
        c.llama_sampler_accept(self.chain, token);
        self.prev.pushBack(token);
    }

    /// Ports `common_sampler_sample` (sampling.cpp:594 @b29c606e2):
    /// grammar-based rejection sampling over the chain.
    ///
    /// Return: the token, which `cur_p` records as selected.
    pub fn sample(self: *Sampler, ctx: ?*c.llama_context, idx: i32, grammar_first: bool) !c.llama_token {
        c.llama_synchronize(ctx);

        try self.setLogits(ctx, idx);

        // Check if a backend sampler has already sampled a token in which case we
        // return that token id directly.
        {
            const id = c.llama_get_sampled_token_ith(ctx, idx);
            if (id != c.LLAMA_TOKEN_NULL) {
                if (self.grmr != null) _ = c.ggml_abort("sampling.cpp", 617, "GGML_ASSERT(%s) failed", "!gsmpl->grmr    && \"using grammar in combination with backend sampling is not supported\"");
                for (0..self.cur_p.size) |i| {
                    if (self.cur_p.data[i].id == id) {
                        self.cur_p.selected = @intCast(i);
                        break;
                    }
                }
                return id;
            }
        }

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
        try self.setLogits(ctx, idx);

        if (self.grammarShouldApply()) c.llama_sampler_apply(self.grmr, &self.cur_p);

        c.llama_sampler_apply(self.chain, &self.cur_p);

        if (self.cur_p.selected == -1) _ = c.ggml_abort("sampling.cpp", 671, "GGML_ASSERT(%s) failed", "cur_p.selected != -1 && \"no selected token during sampling - check your sampling configuration\"");

        id = self.cur_p.data[@intCast(self.cur_p.selected)].id;

        return id;
    }
};

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
}
