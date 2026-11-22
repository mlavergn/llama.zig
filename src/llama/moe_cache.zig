//! A device cache for the experts of MoE layers kept in host memory.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-moe-cache.h` -- `llama_moe_cache`
//! - `llama.cpp/src/llama-moe-cache.cpp` -- `moe_cache_lru`, the three
//!   layer helpers and `llama_moe_cache::impl`, whose members `MoeCache`
//!   holds directly (the C++'s pimpl is only a compile firewall)
//!
//! Both at v1.0.0 (`b1b770d90`).
//!
//! # How it works
//!
//! Each cached layer has a slot map, a host tensor of `n_expert` slots. The
//! graph looks up the slots of the selected experts with a `GET_ROWS` from
//! the slot map; the slot map is a host weight, so the scheduler copies it
//! to the device with the copy callback, which is where `copy` plans the
//! LRU, uploads the missing experts into the device banks and only then
//! sends the slot map. With a CPU share, part of the misses is not
//! uploaded: those experts read zero slots on the device and are computed
//! from the host experts on the CPU, selected through the layer's CPU map.
//!
//! # Containers
//!
//! The C++'s `bindings` and `layer_of` are `std::unordered_map`s that are
//! only ever looked up, so a hash map of any order stands in for them.

const std = @import("std");
const mod = @import("module.zig");
const stdlib = @import("stdlib");

fn oom() noreturn {
    mod.impl.abort("llama-moe-cache.cpp", 0, "out of memory");
}

/// Ports `moe_cache_lru` (llama-moe-cache.cpp:25 @b1b770d90): ClassLRU, one
/// LRU list per frequency class, over the experts of a group of layers.
///
/// `slot_map[il][e]` is the slot of expert `e` of layer `il`, `n_slots` or
/// later if not cached. The CRF (combined recency and frequency) of an
/// expert counts its uses, decayed with a half-life of `crf_half_life`
/// tokens and kept across evictions; its class is `floor(log2(CRF))`, at
/// most `n_class - 1`, and each slot is in the LRU list of its expert's
/// class as of its last use. The victim is an empty slot, or among the
/// least recently used slot of each list that the current `plan` does not
/// select, the one with the largest age/2^class.
pub const Lru = struct {
    /// floor(log2(1/(1 - 2^(-1/crf_half_life)))) = 7, the class of the
    /// largest CRF for a half-life of 128, plus one.
    pub const n_class = 8;
    /// In tokens.
    pub const crf_half_life: f32 = 128.0;

    n_expert: i32 = 0,
    n_slots: i32 = 0,
    /// The slot of the experts that are not cached.
    empty: i32 = 0,

    /// `[n_layer]`: the data of the slot maps; the experts at slot
    /// `n_slots` or later are not cached, and the experts computed on the
    /// CPU read zero slots after the cached ones on the device.
    slot_map: []?[*]i32 = &.{},
    /// `[n_slots]`: the key of the expert in the slot, -1 when empty.
    key_of: []i32 = &.{},

    /// One LRU list per class, the head the least recently used end;
    /// `[n_slots]`, -1 at the ends.
    prev: []i32 = &.{},
    next: []i32 = &.{},
    head: [n_class]i32 = @splat(-1),
    tail: [n_class]i32 = @splat(-1),
    /// `[n_slots]`: the class list of the slot.
    list_of: []i8 = &.{},
    /// The slots that hold an expert.
    n_filled: i32 = 0,

    /// `[n_layer*n_expert]`.
    crf: []f32 = &.{},
    /// `[n_layer*n_expert]`: the layer clock at the last use, -1 if unused.
    last_use: []i64 = &.{},
    /// `[n_layer]`: the tokens seen by the layer, as `plan` adds them.
    clock: []i64 = &.{},

    /// `[n_expert]`: `seen[e] == seen_gen` if the current `plan` selects `e`.
    seen: []u32 = &.{},
    seen_gen: u32 = 0,
    /// The distinct selected experts.
    uniq: std.ArrayList(i32) = .empty,
    misses: std.ArrayList(i32) = .empty,
    /// `[n_tokens]`: the zero slots each token uses.
    token_zero: std.ArrayList(u64) = .empty,

    pub const Fill = struct {
        expert: i32,
        slot: i32,
    };

    pub fn deinit(self: *Lru) void {
        std.heap.c_allocator.free(self.slot_map);
        std.heap.c_allocator.free(self.key_of);
        std.heap.c_allocator.free(self.prev);
        std.heap.c_allocator.free(self.next);
        std.heap.c_allocator.free(self.list_of);
        std.heap.c_allocator.free(self.crf);
        std.heap.c_allocator.free(self.last_use);
        std.heap.c_allocator.free(self.clock);
        std.heap.c_allocator.free(self.seen);
        self.uniq.deinit(std.heap.c_allocator);
        self.misses.deinit(std.heap.c_allocator);
        self.token_zero.deinit(std.heap.c_allocator);
    }

    /// Ports `moe_cache_lru`'s `init` (llama-moe-cache.cpp:56 @b1b770d90).
    pub fn init(self: *Lru, n_layer: usize, n_expert: i32, n_slots: i32) void {
        mod.impl.assert(n_class == @as(c_int, @intFromFloat(stdlib.libm.log2f(1.0 / (1.0 - stdlib.libm.exp2f(-1.0 / crf_half_life))))) + 1, "llama-moe-cache.cpp", 57, "n_class == (int) std::log2(1.0f/(1.0f - std::exp2(-1.0f/crf_half_life))) + 1");
        self.n_expert = n_expert;
        self.n_slots = n_slots;
        self.empty = n_slots;
        self.slot_map = std.heap.c_allocator.alloc(?[*]i32, n_layer) catch oom();
        @memset(self.slot_map, null);
        const ns: usize = @intCast(n_slots);
        self.key_of = std.heap.c_allocator.alloc(i32, ns) catch oom();
        @memset(self.key_of, -1);
        self.prev = std.heap.c_allocator.alloc(i32, ns) catch oom();
        self.next = std.heap.c_allocator.alloc(i32, ns) catch oom();
        var s: i32 = 0;
        while (s < n_slots) : (s += 1) {
            self.prev[@intCast(s)] = s - 1;
            self.next[@intCast(s)] = if (s + 1 < n_slots) s + 1 else -1;
        }
        self.head = @splat(-1);
        self.tail = @splat(-1);
        self.head[0] = 0;
        self.tail[0] = n_slots - 1;
        self.list_of = std.heap.c_allocator.alloc(i8, ns) catch oom();
        @memset(self.list_of, 0);
        const nk = n_layer * @as(usize, @intCast(n_expert));
        self.crf = std.heap.c_allocator.alloc(f32, nk) catch oom();
        @memset(self.crf, 0.0);
        self.last_use = std.heap.c_allocator.alloc(i64, nk) catch oom();
        @memset(self.last_use, -1);
        self.clock = std.heap.c_allocator.alloc(i64, n_layer) catch oom();
        @memset(self.clock, 0);
        self.seen = std.heap.c_allocator.alloc(u32, @intCast(n_expert)) catch oom();
        @memset(self.seen, 0);
    }

    /// Ports `moe_cache_lru`'s `move` (llama-moe-cache.cpp:83 @b1b770d90):
    /// moves slot `s` to the most recently used end of list `cl`.
    fn move(self: *Lru, s: i32, cl: usize) void {
        const su: usize = @intCast(s);
        const from: usize = @intCast(self.list_of[su]);
        if (from == cl and s == self.tail[from]) {
            return;
        }
        if (self.prev[su] >= 0) {
            self.next[@intCast(self.prev[su])] = self.next[su];
        } else {
            self.head[from] = self.next[su];
        }
        if (self.next[su] >= 0) {
            self.prev[@intCast(self.next[su])] = self.prev[su];
        } else {
            self.tail[from] = self.prev[su];
        }

        self.prev[su] = self.tail[cl];
        self.next[su] = -1;
        if (self.tail[cl] >= 0) {
            self.next[@intCast(self.tail[cl])] = s;
        } else {
            self.head[cl] = s;
        }
        self.tail[cl] = s;
        self.list_of[su] = @intCast(cl);
    }

    /// Ports `moe_cache_lru`'s `make_key`, `key_layer` and `key_expert`
    /// (llama-moe-cache.cpp:111, 112, 113 @b1b770d90): a key indexes an
    /// expert of a layer in `crf` and `last_use`.
    fn makeKey(self: *const Lru, il: i32, e: i32) i32 {
        return il * self.n_expert + e;
    }
    fn keyLayer(self: *const Lru, key: i32) i32 {
        return @divTrunc(key, self.n_expert);
    }
    fn keyExpert(self: *const Lru, key: i32) i32 {
        return @rem(key, self.n_expert);
    }

    /// Ports `moe_cache_lru`'s `class_of` (llama-moe-cache.cpp:116
    /// @b1b770d90): for a key after `use`, its CRF is at least 1, so the
    /// log2 is non-negative and the cast is the floor.
    fn classOf(self: *const Lru, key: i32) usize {
        const cl: c_int = @intFromFloat(stdlib.libm.log2f(self.crf[@intCast(key)]));
        return @intCast(@min(n_class - 1, cl));
    }

    /// Ports `moe_cache_lru`'s `decayed_crf` (llama-moe-cache.cpp:121
    /// @b1b770d90): the CRF of a key, decayed by the time since its last use.
    fn decayedCrf(self: *const Lru, il: i32, key: i32) f32 {
        const age: i64 = self.clock[@intCast(il)] - self.last_use[@intCast(key)];
        return self.crf[@intCast(key)] * stdlib.libm.exp2f(@as(f32, @floatFromInt(-age)) / crf_half_life);
    }

    /// Ports `moe_cache_lru`'s `use` (llama-moe-cache.cpp:126 @b1b770d90):
    /// an unused key gets 1, its CRF being 0 before its first use.
    fn use(self: *Lru, il: i32, key: i32) void {
        self.crf[@intCast(key)] = self.decayedCrf(il, key) + 1.0;
        self.last_use[@intCast(key)] = self.clock[@intCast(il)];
    }

    /// Ports `moe_cache_lru`'s `is_selected` (llama-moe-cache.cpp:132
    /// @b1b770d90): the experts this `plan` selects must not be evicted,
    /// nor the slots it filled already.
    fn isSelected(self: *const Lru, il: i32, key: i32) bool {
        if (key < 0) {
            return false; // unused slot
        }
        if (self.keyLayer(key) != il) {
            return false; // seen is per expert index, an expert of another layer can have the same index
        }
        return self.seen[@intCast(self.keyExpert(key))] == self.seen_gen;
    }

    /// Ports `moe_cache_lru`'s `candidate` (llama-moe-cache.cpp:143
    /// @b1b770d90): the least recently used slot of list `cl` that this
    /// `plan` does not select, -1 if none.
    fn candidate(self: *const Lru, cl: usize, il: i32) i32 {
        var s = self.head[cl];
        while (s >= 0) : (s = self.next[@intCast(s)]) {
            if (!self.isSelected(il, self.key_of[@intCast(s)])) {
                return s;
            }
        }
        return -1;
    }

    /// Ports `moe_cache_lru`'s `victim` (llama-moe-cache.cpp:152 @b1b770d90).
    fn victim(self: *const Lru, il: i32) i32 {
        var res: i32 = -1;
        var res_score: f64 = -1.0;
        for (0..n_class) |cl| {
            const s = self.candidate(cl, il);
            if (s < 0) {
                continue;
            }
            const key = self.key_of[@intCast(s)];
            if (key < 0) {
                return s; // unused slot
            }
            const age: i64 = self.clock[@intCast(self.keyLayer(key))] - self.last_use[@intCast(key)];
            const score = std.math.ldexp(@as(f64, @floatFromInt(age)), -@as(i32, @intCast(cl))); // age/2^class
            if (score > res_score) {
                res = s;
                res_score = score;
            }
        }
        if (res < 0) {
            mod.impl.abort("llama-moe-cache.cpp", 172, "no MoE cache victim");
        }
        return res;
    }

    /// Ports `moe_cache_lru`'s `plan` (llama-moe-cache.cpp:187 @b1b770d90):
    /// gives a slot to each expert `ids` (`n_tokens` tokens, `n_used`
    /// each) selects in layer `il`. `n_cpu.nCpu(n)` of the `n` misses are
    /// left to the CPU and read a zero slot on the device, uncached; a zero
    /// slot is used at most once per token, and a miss that does not get
    /// one is cached instead.
    ///
    /// Return: false when the ids select more distinct experts than there
    /// are slots.
    pub fn plan(self: *Lru, il: i32, ids: []const i32, n_tokens: i64, n_used: i32, n_zero: i32, n_cpu: anytype, fills: *std.ArrayList(Fill), cpu: *std.ArrayList(Fill), n_hit: *usize) bool {
        fills.clearRetainingCapacity();
        cpu.clearRetainingCapacity();
        n_hit.* = 0;

        self.seen_gen +%= 1;
        if (self.seen_gen == 0) {
            @memset(self.seen, 0);
            self.seen_gen = 1;
        }
        self.uniq.clearRetainingCapacity();
        for (ids) |id| {
            mod.impl.assert(id >= 0 and id < self.n_expert, "llama-moe-cache.cpp", 199, "ids[i] >= 0 && ids[i] < n_expert");
            if (self.seen[@intCast(id)] != self.seen_gen) {
                self.seen[@intCast(id)] = self.seen_gen;
                self.uniq.append(std.heap.c_allocator, id) catch oom();
            }
        }
        if (self.uniq.items.len > @as(usize, @intCast(self.n_slots))) {
            return false;
        }

        const slots = self.slot_map[@intCast(il)].?;

        // update the CRF of all selected experts and promote hits before
        // choosing victims for misses
        self.misses.clearRetainingCapacity();
        for (self.uniq.items) |e| {
            const key = self.makeKey(il, e);
            self.use(il, key);
            if (slots[@intCast(e)] < self.n_slots) {
                self.move(slots[@intCast(e)], self.classOf(key));
                n_hit.* += 1;
            } else {
                self.misses.append(std.heap.c_allocator, e) catch oom();
            }
        }

        // the misses with the lowest CRF are left to the CPU, a miss that
        // keeps being used gains CRF and enters the cache; all misses are
        // used at the same clock, so their CRF compare without decay
        const n_c = @min(n_cpu.nCpu(self.misses.items.len), self.misses.items.len);
        if (n_c > 0) {
            const ByCrf = struct {
                lru: *const Lru,
                il: i32,
                fn lt(ctx: @This(), a_: i32, b_: i32) bool {
                    const crf_a = ctx.lru.crf[@intCast(ctx.lru.makeKey(ctx.il, a_))];
                    const crf_b = ctx.lru.crf[@intCast(ctx.lru.makeKey(ctx.il, b_))];
                    return if (crf_a != crf_b) crf_a < crf_b else a_ < b_;
                }
            };
            std.mem.sort(i32, self.misses.items, ByCrf{ .lru = self, .il = il }, ByCrf.lt);
            mod.impl.assert(@as(usize, @intCast(n_tokens * n_used)) == ids.len, "llama-moe-cache.cpp", 233, "(size_t) n_tokens*n_used == n_ids");
            const nt: usize = @intCast(n_tokens);
            const nu: usize = @intCast(n_used);
            self.token_zero.resize(std.heap.c_allocator, nt) catch oom();
            @memset(self.token_zero.items, 0);
            const misses = self.misses.items;
            var n_keep: usize = n_c;
            var i: usize = 0;
            while (i < n_keep) {
                const e = misses[i];
                var used: u64 = 0;
                for (0..nt) |it| {
                    for (0..nu) |k| {
                        if (ids[it * nu + k] == e) {
                            used |= self.token_zero.items[it];
                        }
                    }
                }
                var z: i32 = 0;
                while (z < 64 and ((used >> @intCast(z)) & 1) != 0) {
                    z += 1;
                }
                if (z >= n_zero) {
                    n_keep -= 1;
                    std.mem.swap(i32, &misses[i], &misses[n_keep]);
                    continue;
                }
                for (0..nt) |it| {
                    for (0..nu) |k| {
                        if (ids[it * nu + k] == e) {
                            self.token_zero.items[it] |= @as(u64, 1) << @intCast(z);
                        }
                    }
                }
                slots[@intCast(e)] = self.n_slots + z;
                cpu.append(std.heap.c_allocator, .{ .expert = e, .slot = self.n_slots + z }) catch oom();
                i += 1;
            }
            self.misses.replaceRange(std.heap.c_allocator, 0, n_keep, &.{}) catch oom();
        }

        // fill the misses in expert order: victim() takes unused slots, and
        // slots filled by one plan(), in list order, so consecutive experts
        // often get consecutive slots and one upload
        std.mem.sort(i32, self.misses.items, {}, std.sort.asc(i32));
        for (self.misses.items) |e| {
            const s = self.victim(il);
            const su: usize = @intCast(s);
            if (self.key_of[su] >= 0) {
                const k = self.key_of[su];
                self.slot_map[@intCast(self.keyLayer(k))].?[@intCast(self.keyExpert(k))] = self.empty;
            } else {
                self.n_filled += 1;
            }
            const key = self.makeKey(il, e);
            self.key_of[su] = key;
            slots[@intCast(e)] = s;
            self.move(s, self.classOf(key));
            fills.append(std.heap.c_allocator, .{ .expert = e, .slot = s }) catch oom();
        }
        self.clock[@intCast(il)] += n_tokens;
        return true;
    }
};

/// The `n_cpu` lambda of `impl::copy` (llama-moe-cache.cpp:808 @b1b770d90):
/// part of the misses is computed on the CPU once the cache is full, and
/// the carry keeps the share right on average.
pub const CpuShare = struct {
    /// The share in 1/65536.
    share_q: u32,
    /// The fraction of a miss carried over to the next plan, in 1/65536.
    acc: *u32,

    pub fn nCpu(self: CpuShare, n_miss: usize) usize {
        const acc: u64 = @as(u64, n_miss) * self.share_q + self.acc.*;
        self.acc.* = @truncate(acc & 0xffff);
        return @intCast(acc >> 16);
    }
};

/// Ports `llama_moe_cache_layer_experts` (llama-moe-cache.cpp:290
/// @b1b770d90): gate, up, down, or gate_up, down.
///
/// Return: the tensors, at most four, in `out`'s first slots.
fn layerExperts(layer: *const mod.model.Layer, out: *[4]*mod.impl.llama_c.ggml_tensor) usize {
    var n: usize = 0;
    for ([_]?*mod.impl.llama_c.ggml_tensor{ layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) |t_| {
        if (t_) |t| {
            out[n] = t;
            n += 1;
        }
    }
    return n;
}

/// Ports `llama_moe_cache_same_layout` (llama-moe-cache.cpp:300 @b1b770d90).
fn sameLayout(a: []const *mod.impl.llama_c.ggml_tensor, b: []const *mod.impl.llama_c.ggml_tensor) bool {
    if (a.len != b.len) {
        return false;
    }
    for (a, b) |x, y| {
        if (x.type != y.type or !mod.impl.llama_c.ggml_are_same_shape(x, y) or x.nb[2] != y.nb[2]) {
            return false;
        }
    }
    return true;
}

/// Ports `llama_moe_cache_is_host_weight` (llama-moe-cache.cpp:312 @b1b770d90).
fn isHostWeight(t: *const mod.impl.llama_c.ggml_tensor) bool {
    return t.buffer != null and
        mod.impl.llama_c.ggml_backend_buffer_get_usage(t.buffer) == mod.impl.llama_c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS and
        mod.impl.llama_c.ggml_backend_buffer_is_host(t.buffer);
}

/// Ports `llama_moe_cache` (llama-moe-cache.h:13 @b1b770d90) and its `impl`
/// (llama-moe-cache.cpp:320 @b1b770d90): keeps recently and repeatedly used
/// experts of host-resident MoE layers in a device buffer.
pub const MoeCache = struct {
    /// A GPU with its own budget and banks; it caches the layers assigned
    /// to it.
    const Device = struct {
        backend: mod.impl.llama_c.ggml_backend_t,
        buft: mod.impl.llama_c.ggml_backend_buffer_type_t,
        /// The host experts of the layers it caches.
        host_bytes: usize = 0,
        /// Its share of the budget.
        split: f64 = 0.0,

        /// The banks and their views.
        ctx: ?*mod.impl.llama_c.ggml_context = null,
        buf: mod.impl.llama_c.ggml_backend_buffer_t = null,
        buf_size: usize = 0,
    };

    /// Layers of one device with one expert tensor layout share the banks
    /// and the LRU of a group.
    const Group = struct {
        /// The device.
        id: i32,
        /// The expert tensors of the first layer.
        ref: std.ArrayList(*mod.impl.llama_c.ggml_tensor) = .empty,
        layers: std.ArrayList(i32) = .empty,
        /// The device storage of every slot, one per expert tensor.
        banks: std.ArrayList(*mod.impl.llama_c.ggml_tensor) = .empty,
        host_bytes: usize = 0,
        n_slots: i32 = 0,
        lru: Lru = .{},
    };

    const Layer = struct {
        /// -1 when the layer is not cached.
        ig: i32 = -1,
        /// I32 `[1, n_expert]`, in host memory.
        slot_map: ?*mod.impl.llama_c.ggml_tensor = null,
        /// Looked up in place of `slot_map` by the graphs with a CPU part,
        /// if `cpu_share > 0`; its data is not used.
        slot_map_cpu: ?*mod.impl.llama_c.ggml_tensor = null,
        /// I32 `[1, n_expert]`, in host memory, if `cpu_share > 0`.
        cpu_map: ?*mod.impl.llama_c.ggml_tensor = null,
        /// The experts set in `cpu_map`.
        cpu_experts: std.ArrayList(i32) = .empty,
        /// The fraction of a miss carried over to the next plan, in 1/65536.
        cpu_acc: u32 = 0,
        /// The host expert tensors, in the order of the banks.
        experts: [4]*mod.impl.llama_c.ggml_tensor = undefined,
        n_experts: usize = 0,
    };

    const Binding = struct {
        il: i32,
        /// The index of the bank.
        ip: i32,
        /// The view of the bank used in place of the host experts.
        cached: *mod.impl.llama_c.ggml_tensor,
    };

    const Stats = struct {
        hits: usize = 0,
        misses: usize = 0,
        /// The misses computed on the CPU.
        cpu: usize = 0,
        bytes: usize = 0,
    };

    /// A slot map's layer, and whether the graph that looks it up has a
    /// CPU part.
    const LayerOf = struct { il: i32, has_cpu: bool };

    const max_batch: i64 = 32;

    n_expert_used: i32,

    /// Up to 8 tokens per ubatch.
    stats_small: Stats = .{},
    stats_large: Stats = .{},
    /// The experts copied from the cache for large batches.
    stats_copy: Stats = .{},

    devices: std.ArrayList(Device) = .empty,
    groups: std.ArrayList(Group) = .empty,
    layers: []Layer = &.{},
    /// The host experts to the cached experts.
    bindings: std.AutoHashMapUnmanaged(*const mod.impl.llama_c.ggml_tensor, Binding) = .empty,
    /// A slot map to its layer.
    layer_of: std.AutoHashMapUnmanaged(*const mod.impl.llama_c.ggml_tensor, LayerOf) = .empty,

    ids: std.ArrayList(i32) = .empty,
    fills: std.ArrayList(Lru.Fill) = .empty,
    cpu: std.ArrayList(Lru.Fill) = .empty,

    cpu_share: f32 = 0.0,
    /// `cpu_share` in 1/65536, an integer so that a simulator can
    /// reproduce the rounding.
    cpu_share_q: u32 = 0,
    /// The zero slots per group.
    n_zero: i32 = 0,

    /// `LLAMA_MOE_CACHE_TRACE`: the selected experts of each slot lookup,
    /// for a simulator of the cache.
    trace: ?*std.c.FILE = null,

    /// The slot maps, in host memory.
    ctx_host: ?*mod.impl.llama_c.ggml_context = null,
    buf_host: mod.impl.llama_c.ggml_backend_buffer_t = null,
    buf_host_size: usize = 0,

    /// The views `copyExperts` makes.
    ctx_views: ?*mod.impl.llama_c.ggml_context = null,

    /// Ports the constructor of `llama_moe_cache` (llama-moe-cache.cpp:875
    /// @b1b770d90) and of its `impl` (llama-moe-cache.cpp:401 @b1b770d90).
    ///
    /// Parameters:
    /// - `model`: the model; borrowed for the cache's lifetime.
    /// - `backends`, `bufts`: every backend of the context and its compute
    ///   buffer type; each GPU gets its own cache of `size` bytes, split
    ///   like the layers, for the layers assigned to it.
    /// - `size`: the budget in bytes.
    /// - `cpu_share`: the share of the misses computed on the CPU instead
    ///   of uploaded, clamped to [0, 1].
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the cache, the caller's to `destroy`, or `error.Thrown`.
    pub fn create(model: *const mod.model.Model, no_alloc: bool, backends: []const mod.impl.llama_c.ggml_backend_t, bufts: []const mod.impl.llama_c.ggml_backend_buffer_type_t, size: usize, cpu_share: f32, exc: *mod.impl.Exception) mod.impl.Thrown!*MoeCache {
        // `__func__` in the constructor of `struct impl`
        const func = "impl";
        const self = std.heap.c_allocator.create(MoeCache) catch oom();
        self.* = .{ .n_expert_used = @intCast(model.hparams.nExpertUsedMax()), .cpu_share = std.math.clamp(cpu_share, 0.0, 1.0) };
        errdefer self.destroy();
        self.layers = std.heap.c_allocator.alloc(Layer, model.layers.len) catch oom();
        @memset(self.layers, .{});

        for (backends, bufts) |be, buft| {
            const dev_type = mod.impl.llama_c.ggml_backend_dev_type(mod.impl.llama_c.ggml_backend_get_device(be));
            if (dev_type == mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_GPU or dev_type == mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_IGPU) {
                self.devices.append(std.heap.c_allocator, .{ .backend = be, .buft = buft }) catch oom();
            }
        }
        if (self.devices.items.len == 0) {
            return exc.throw(.runtime_error, "MoE cache requires a GPU backend", .{});
        }
        if (model.params.split_mode == mod.impl.llama_c.LLAMA_SPLIT_MODE_TENSOR) {
            return exc.throw(.runtime_error, "MoE cache does not support tensor parallelism", .{});
        }
        if (model.hparams.n_expert == 0 or self.n_expert_used == 0) {
            return exc.throw(.runtime_error, "MoE cache requires a MoE model", .{});
        }

        // only cache the layers that keep all of their experts in host
        // memory, on the device the layer is assigned to
        for (model.layers, 0..) |*mlayer, il| {
            var experts: [4]*mod.impl.llama_c.ggml_tensor = undefined;
            const n = layerExperts(mlayer, &experts);
            if (n == 0) continue;
            var all_host = true;
            for (experts[0..n]) |t| all_host = all_host and isHostWeight(t);
            if (!all_host) continue;

            var id_opt: ?usize = null;
            for (self.devices.items, 0..) |d, i| {
                if (mod.impl.llama_c.ggml_backend_get_device(d.backend) == model.devLayer(il)) {
                    id_opt = i;
                    break;
                }
            }
            const id = id_opt orelse continue;

            var g_opt: ?*Group = null;
            for (self.groups.items) |*g| {
                if (g.id == @as(i32, @intCast(id)) and sameLayout(g.ref.items, experts[0..n])) {
                    g_opt = g;
                    break;
                }
            }
            const g = g_opt orelse blk: {
                self.groups.append(std.heap.c_allocator, .{ .id = @intCast(id) }) catch oom();
                const ng = &self.groups.items[self.groups.items.len - 1];
                ng.ref.appendSlice(std.heap.c_allocator, experts[0..n]) catch oom();
                break :blk ng;
            };
            g.layers.append(std.heap.c_allocator, @intCast(il)) catch oom();
            for (experts[0..n]) |t| {
                g.host_bytes += mod.impl.llama_c.ggml_nbytes(t);
                self.devices.items[id].host_bytes += mod.impl.llama_c.ggml_nbytes(t);
            }
        }
        if (self.groups.items.len == 0) {
            mod.impl.logWarn("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", .{func});
            return self;
        }

        // with misses on the CPU, the n_zero slots after the cached ones stay
        // zero: the device computes these misses as zero
        self.n_zero = if (self.cpu_share > 0.0) @min(self.n_expert_used, 64) else 0;

        // the budget is split among the devices with host experts like the
        // layers, by the tensor split or by default by free memory
        const tensor_split = model.params.tensor_split;
        var split_by_free = tensor_split == null;
        if (!split_by_free) {
            split_by_free = true;
            for (tensor_split[0..model.nDevices()]) |x| {
                if (x != 0.0) split_by_free = false;
            }
        }
        var split_sum: f64 = 0.0;
        for (self.devices.items) |*d| {
            if (d.host_bytes == 0) {
                continue;
            }
            const dev = mod.impl.llama_c.ggml_backend_get_device(d.backend);
            if (split_by_free) {
                var free: usize = 0;
                var total: usize = 0;
                mod.impl.llama_c.ggml_backend_dev_memory(dev, &free, &total);
                d.split = @floatFromInt(free);
            } else {
                var at: ?usize = null;
                for (model.devices.items, 0..) |ld, i| {
                    if (ld.dev == dev) {
                        at = i;
                        break;
                    }
                }
                mod.impl.assert(at != null, "llama-moe-cache.cpp", 479, "it != model.devices.end()");
                d.split = tensor_split[at.?];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (self.devices.items) |*d| {
                d.split = if (d.host_bytes > 0) 1.0 else 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so
        // each group caches the same fraction of its experts
        self.cpu_share_q = @intFromFloat(@round(self.cpu_share * 65536.0));
        if (stdlib.env.getenv("LLAMA_MOE_CACHE_TRACE")) |path| {
            if (path[0] != 0) {
                self.trace = std.c.fopen(path, "wb");
                if (self.trace == null) {
                    return exc.throw(.runtime_error, "failed to open the MoE cache trace {s}", .{std.mem.span(path)});
                }
                self.writeTraceHeader(split_sum);
            }
        }
        const n_tensors = std.heap.c_allocator.alloc(usize, self.devices.items.len) catch oom();
        defer std.heap.c_allocator.free(n_tensors);
        @memset(n_tensors, 0);
        var n_tensors_host: usize = 0;
        for (self.groups.items) |*g| {
            const d = &self.devices.items[@intCast(g.id)];
            const n_expert: i32 = @intCast(g.ref.items[0].ne[2]);
            const budget: usize = @intFromFloat(@as(f64, @floatFromInt(size)) * d.split / split_sum * @as(f64, @floatFromInt(g.host_bytes)) / @as(f64, @floatFromInt(d.host_bytes)));
            const max_slots: i32 = @intCast(g.layers.items.len * @as(usize, @intCast(n_expert)));
            while (g.n_slots < max_slots and self.allocSize(g, g.n_slots + 1) <= budget) {
                g.n_slots += 1;
            }
            if (g.n_slots < self.n_expert_used) {
                mod.impl.logWarn("%s: MoE cache budget is too small for %zu layers, they are not cached\n", .{ func, g.layers.items.len });
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.len, n_expert, g.n_slots);
            n_tensors[@intCast(g.id)] += g.ref.items.len * (1 + g.layers.items.len);
            n_tensors_host += g.layers.items.len * @as(usize, if (self.n_zero > 0) 3 else 1); // slot maps and CPU map
        }
        if (n_tensors_host == 0) {
            return exc.throw(.runtime_error, "MoE cache is too small to hold the experts of one token", .{});
        }

        for (self.devices.items, 0..) |*d, id| {
            if (n_tensors[id] > 0) {
                d.ctx = try initCtx(n_tensors[id], exc);
            }
        }
        self.ctx_host = try initCtx(n_tensors_host, exc);
        self.ctx_views = try initCtx(2, exc);

        const buft_host = mod.impl.llama_c.ggml_backend_cpu_buffer_type();
        const alignment_host = mod.impl.llama_c.ggml_backend_buft_get_alignment(buft_host);

        for (self.groups.items, 0..) |*g, ig| {
            if (g.n_slots == 0) {
                continue;
            }
            const ctx = self.devices.items[@intCast(g.id)].ctx;
            for (g.ref.items) |t| {
                const bank: *mod.impl.llama_c.ggml_tensor = mod.impl.llama_c.ggml_new_tensor_3d(ctx, t.type, t.ne[0], t.ne[1], g.n_slots + self.n_zero + 1);
                mod.impl.assert(bank.nb[2] == t.nb[2], "llama-moe-cache.cpp", 555, "bank->nb[2] == t->nb[2]");
                _ = mod.impl.llama_c.ggml_format_name(bank, "moe_cache.%zu.%s", ig, mod.impl.llama_c.ggml_get_name(t));
                g.banks.append(std.heap.c_allocator, bank) catch oom();
            }
            for (g.layers.items) |il| {
                const l = &self.layers[@intCast(il)];
                l.ig = @intCast(ig);
                l.n_experts = layerExperts(&model.layers[@intCast(il)], &l.experts);
                for (l.experts[0..l.n_experts], 0..) |e, ip| {
                    const bank = g.banks.items[ip];
                    const cached: *mod.impl.llama_c.ggml_tensor = mod.impl.llama_c.ggml_view_3d(ctx, bank, bank.ne[0], bank.ne[1], g.n_slots + self.n_zero, bank.nb[1], bank.nb[2], 0);
                    _ = mod.impl.llama_c.ggml_format_name(cached, "moe_cache.%s", mod.impl.llama_c.ggml_get_name(e));
                    self.bindings.put(std.heap.c_allocator, e, .{ .il = il, .ip = @intCast(ip), .cached = cached }) catch oom();
                }
                const slot_map: *mod.impl.llama_c.ggml_tensor = mod.impl.llama_c.ggml_new_tensor_2d(self.ctx_host, mod.impl.llama_c.GGML_TYPE_I32, 1, g.ref.items[0].ne[2]);
                _ = mod.impl.llama_c.ggml_format_name(slot_map, "moe_cache.slot_map-%d", il);
                l.slot_map = slot_map;
                self.layer_of.put(std.heap.c_allocator, slot_map, .{ .il = il, .has_cpu = false }) catch oom();
                self.buf_host_size += stdlib.num.pad(usize, mod.impl.llama_c.ggml_nbytes(slot_map), alignment_host);
                if (self.n_zero > 0) {
                    const slot_map_cpu: *mod.impl.llama_c.ggml_tensor = mod.impl.llama_c.ggml_new_tensor_2d(self.ctx_host, mod.impl.llama_c.GGML_TYPE_I32, 1, g.ref.items[0].ne[2]);
                    _ = mod.impl.llama_c.ggml_format_name(slot_map_cpu, "moe_cache.slot_map_cpu-%d", il);
                    l.slot_map_cpu = slot_map_cpu;
                    self.layer_of.put(std.heap.c_allocator, slot_map_cpu, .{ .il = il, .has_cpu = true }) catch oom();
                    self.buf_host_size += stdlib.num.pad(usize, mod.impl.llama_c.ggml_nbytes(slot_map_cpu), alignment_host);
                    const cpu_map: *mod.impl.llama_c.ggml_tensor = mod.impl.llama_c.ggml_new_tensor_2d(self.ctx_host, mod.impl.llama_c.GGML_TYPE_I32, 1, g.ref.items[0].ne[2]);
                    _ = mod.impl.llama_c.ggml_format_name(cpu_map, "moe_cache.cpu_map-%d", il);
                    l.cpu_map = cpu_map;
                    self.buf_host_size += stdlib.num.pad(usize, mod.impl.llama_c.ggml_nbytes(cpu_map), alignment_host);
                }
            }
            self.devices.items[@intCast(g.id)].buf_size += self.allocSize(g, g.n_slots);
        }

        // with no_alloc, dummy buffers are attached so the memory use can
        // still be measured, see llama_context::memory_breakdown
        for (self.devices.items) |*d| {
            const ctx = d.ctx orelse continue;
            d.buf = try mod.memory.Bufs.allocBuf(ctx, d.buft, no_alloc, "MoE cache", exc);
            if (!no_alloc) {
                mod.impl.llama_c.ggml_backend_buffer_clear(d.buf, 0);
                d.buf_size = mod.impl.llama_c.ggml_backend_buffer_get_size(d.buf);
            }
        }
        self.buf_host = try mod.memory.Bufs.allocBuf(self.ctx_host.?, buft_host, no_alloc, "MoE cache slot maps", exc);
        if (!no_alloc) {
            mod.impl.llama_c.ggml_backend_buffer_clear(self.buf_host, 0xff); // the CPU maps are -1
            self.buf_host_size = mod.impl.llama_c.ggml_backend_buffer_get_size(self.buf_host);

            for (self.groups.items) |*g| {
                for (g.layers.items) |il| {
                    if (self.layers[@intCast(il)].slot_map) |sm| {
                        const slots: [*]i32 = @ptrCast(@alignCast(sm.data));
                        @memset(slots[0..@intCast(sm.ne[1])], g.lru.empty);
                        g.lru.slot_map[@intCast(il)] = slots;
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the
        // slot maps are copied with the copy callback
        for (self.devices.items) |*d| {
            if (d.buf != null) {
                mod.impl.llama_c.ggml_backend_buffer_set_usage(d.buf, mod.impl.llama_c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        mod.impl.llama_c.ggml_backend_buffer_set_usage(self.buf_host, mod.impl.llama_c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (self.devices.items, 0..) |*d, id| {
            if (d.host_bytes == 0) {
                continue;
            }
            mod.impl.logInfo("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts, %.0f%% of the misses on the CPU\n", .{
                func,
                mod.impl.llama_c.ggml_backend_buft_name(d.buft),
                @as(f64, @floatFromInt(d.buf_size)) / 1024.0 / 1024.0,
                @as(f64, @floatFromInt(d.host_bytes)) / 1024.0 / 1024.0,
                100.0 * @as(f64, self.cpu_share),
            });
            for (self.groups.items) |*g| {
                if (g.id == @as(i32, @intCast(id))) {
                    const n_all: f64 = @floatFromInt(g.layers.items.len * @as(usize, @intCast(g.ref.items[0].ne[2])));
                    mod.impl.logInfo("%s: %2zu layers, %s: %5d slots (%.1f%%, %.1f tokens worth)\n", .{
                        func,
                        g.layers.items.len,
                        mod.impl.llama_c.ggml_type_name(g.ref.items[g.ref.items.len - 1].type),
                        g.n_slots,
                        100.0 * @as(f64, @floatFromInt(g.n_slots)) / n_all,
                        @as(f64, @floatFromInt(g.n_slots)) / @as(f64, @floatFromInt(g.layers.items.len * @as(usize, @intCast(self.n_expert_used)))),
                    });
                }
            }
        }

        return self;
    }

    /// The `init_ctx` lambda of the constructor (llama-moe-cache.cpp:524
    /// @b1b770d90).
    fn initCtx(n_tensors: usize, exc: *mod.impl.Exception) mod.impl.Thrown!*mod.impl.llama_c.ggml_context {
        return mod.impl.llama_c.ggml_init(.{
            .mem_size = n_tensors * mod.impl.llama_c.ggml_tensor_overhead(),
            .mem_buffer = null,
            .no_alloc = true,
        }) orelse return exc.throw(.runtime_error, "failed to create the MoE cache context", .{});
    }

    /// The `alloc_size` lambda of the constructor (llama-moe-cache.cpp:453
    /// @b1b770d90): the `n_zero` zero slots after the cached ones, and one
    /// extra slot at the end, as CUDA's MMQ can read past the last expert.
    fn allocSize(self: *const MoeCache, g: *const Group, n_slots: i32) usize {
        const alignment = mod.impl.llama_c.ggml_backend_buft_get_alignment(self.devices.items[@intCast(g.id)].buft);
        var res: usize = 0;
        for (g.ref.items) |t| {
            res += stdlib.num.pad(usize, t.nb[2] * @as(usize, @intCast(n_slots + self.n_zero + 1)), alignment);
        }
        return res;
    }

    fn traceWrite(self: *const MoeCache, bytes: []const u8) void {
        _ = std.c.fwrite(bytes.ptr, 1, bytes.len, self.trace.?);
    }

    /// Ports `impl`'s `write_trace_header` (llama-moe-cache.cpp:646
    /// @b1b770d90). The format, little endian: "MCT1", then u32 n_layer,
    /// n_expert_used, n_group; per group u32 device, n_expert, n_layers,
    /// the layers (u32), f64 split, split_sum, u64 group host bytes, device
    /// host bytes, alignment, u32 n_tensors and the bytes of one expert of
    /// each tensor (u64); then one record per slot lookup, `writeTrace`'s.
    fn writeTraceHeader(self: *const MoeCache, split_sum: f64) void {
        const W = struct {
            fn u32_(m: *const MoeCache, v: u32) void {
                m.traceWrite(std.mem.asBytes(&v));
            }
            fn u64_(m: *const MoeCache, v: u64) void {
                m.traceWrite(std.mem.asBytes(&v));
            }
            fn f64_(m: *const MoeCache, v: f64) void {
                m.traceWrite(std.mem.asBytes(&v));
            }
        };
        self.traceWrite("MCT1");
        W.u32_(self, @truncate(self.layers.len));
        W.u32_(self, @bitCast(self.n_expert_used));
        W.u32_(self, @truncate(self.groups.items.len));
        for (self.groups.items) |*g| {
            const d = &self.devices.items[@intCast(g.id)];
            W.u32_(self, @bitCast(g.id));
            W.u32_(self, @truncate(@as(u64, @bitCast(g.ref.items[0].ne[2]))));
            W.u32_(self, @truncate(g.layers.items.len));
            for (g.layers.items) |il| {
                W.u32_(self, @bitCast(il));
            }
            W.f64_(self, d.split);
            W.f64_(self, split_sum);
            W.u64_(self, g.host_bytes);
            W.u64_(self, d.host_bytes);
            W.u64_(self, mod.impl.llama_c.ggml_backend_buft_get_alignment(d.buft));
            W.u32_(self, @truncate(g.ref.items.len));
            for (g.ref.items) |t| {
                W.u64_(self, t.nb[2]);
            }
        }
    }

    /// Ports `impl`'s `write_trace` (llama-moe-cache.cpp:674 @b1b770d90):
    /// i32 layer, experts per token, n_ids, then the ids (i32).
    fn writeTrace(self: *const MoeCache, il: i32, n_used: i32) void {
        const rec = [3]i32{ il, n_used, @truncate(@as(i64, @intCast(self.ids.items.len))) };
        self.traceWrite(std.mem.sliceAsBytes(&rec));
        self.traceWrite(std.mem.sliceAsBytes(self.ids.items));
    }

    /// Ports the destructor of `llama_moe_cache`'s `impl`
    /// (llama-moe-cache.cpp:680 @b1b770d90): logs the statistics, closes the
    /// trace, then frees the members.
    ///
    /// Return: nothing.
    pub fn destroy(self: *MoeCache) void {
        self.logStats();
        if (self.trace) |f| _ = std.c.fclose(f);

        if (self.ctx_views) |ctx| mod.impl.llama_c.ggml_free(ctx);
        if (self.buf_host != null) mod.impl.llama_c.ggml_backend_buffer_free(self.buf_host);
        if (self.ctx_host) |ctx| mod.impl.llama_c.ggml_free(ctx);
        self.cpu.deinit(std.heap.c_allocator);
        self.fills.deinit(std.heap.c_allocator);
        self.ids.deinit(std.heap.c_allocator);
        self.layer_of.deinit(std.heap.c_allocator);
        self.bindings.deinit(std.heap.c_allocator);
        for (self.layers) |*l| l.cpu_experts.deinit(std.heap.c_allocator);
        std.heap.c_allocator.free(self.layers);
        for (self.groups.items) |*g| {
            g.lru.deinit();
            g.banks.deinit(std.heap.c_allocator);
            g.layers.deinit(std.heap.c_allocator);
            g.ref.deinit(std.heap.c_allocator);
        }
        self.groups.deinit(std.heap.c_allocator);
        for (self.devices.items) |*d| {
            if (d.buf != null) mod.impl.llama_c.ggml_backend_buffer_free(d.buf);
            if (d.ctx) |ctx| mod.impl.llama_c.ggml_free(ctx);
        }
        self.devices.deinit(std.heap.c_allocator);
        std.heap.c_allocator.destroy(self);
    }

    /// Ports `llama_moe_cache`'s `backend` (llama-moe-cache.cpp:881
    /// @b1b770d90).
    ///
    /// Return: the device that caches layer `il`.
    pub fn backend(self: *const MoeCache, il: i32) mod.impl.llama_c.ggml_backend_t {
        return self.devices.items[@intCast(self.groups.items[@intCast(self.layers[@intCast(il)].ig)].id)].backend;
    }

    /// Ports `llama_moe_cache`'s `get_slot_map` (llama-moe-cache.cpp:885
    /// @b1b770d90), through the impl's `get_slot_map` (llama-moe-cache.cpp:687
    /// @b1b770d90).
    /// Large batches use most experts of a layer, so they gain little from
    /// the cache and would evict the experts used in generation. With
    /// `cpu_part`, the graph computes the misses in the layer's CPU map.
    ///
    /// Return: the slot map of layer `il`, when its experts can be read from
    /// the cache for `n_tokens` tokens, else null.
    pub fn getSlotMap(self: *const MoeCache, il: i32, n_tokens: i64, n_expert_used: i64, cpu_part: bool) ?*mod.impl.llama_c.ggml_tensor {
        if (il < 0 or il >= @as(i32, @intCast(self.layers.len)) or self.layers[@intCast(il)].ig < 0) {
            return null;
        }
        const l = &self.layers[@intCast(il)];

        if (n_tokens == 0 or n_tokens > max_batch or @min(n_tokens * n_expert_used, l.slot_map.?.ne[1]) > self.groups.items[@intCast(l.ig)].n_slots) {
            return null;
        }
        mod.impl.assert(!cpu_part or l.slot_map_cpu != null, "llama-moe-cache.cpp", 697, "!cpu_part || l.slot_map_cpu != nullptr");
        return if (cpu_part) l.slot_map_cpu else l.slot_map;
    }

    /// Ports `llama_moe_cache`'s `get_cpu_map` (llama-moe-cache.cpp:893
    /// @b1b770d90), through the impl's `get_cpu_map` (llama-moe-cache.cpp:701
    /// @b1b770d90).
    ///
    /// Return: the experts of layer `il` computed on the CPU, I32
    /// `[1, n_expert]` in host memory, `e` for these experts and -1 for the
    /// others; null if no misses of the layer are computed on the CPU for
    /// ubatches of `n_tokens` tokens.
    pub fn getCpuMap(self: *const MoeCache, il: i32, n_tokens: i64) ?*mod.impl.llama_c.ggml_tensor {
        if (il < 0 or il >= @as(i32, @intCast(self.layers.len))) {
            return null;
        }
        const cpu_map = self.layers[@intCast(il)].cpu_map orelse return null;
        // the scheduler would run the CPU part on a device that offloads
        // such batches from host weights
        var probe = std.mem.zeroes(mod.impl.llama_c.ggml_tensor);
        probe.op = mod.impl.llama_c.GGML_OP_MUL_MAT_ID;
        probe.type = mod.impl.llama_c.GGML_TYPE_F32;
        probe.ne[0] = 1;
        probe.ne[1] = self.n_expert_used;
        probe.ne[2] = n_tokens;
        probe.ne[3] = 1;
        for (self.devices.items) |d| {
            if (mod.impl.llama_c.ggml_backend_offload_op(d.backend, &probe)) {
                return null;
            }
        }
        return cpu_map;
    }

    /// Ports `llama_moe_cache`'s `get_experts` (llama-moe-cache.cpp:889
    /// @b1b770d90).
    ///
    /// Return: the experts of `w` in the cache, or null when `w` is not
    /// cached.
    pub fn getExperts(self: *const MoeCache, w: ?*const mod.impl.llama_c.ggml_tensor) ?*mod.impl.llama_c.ggml_tensor {
        const b = self.bindings.get(w orelse return null) orelse return null;
        return b.cached;
    }

    /// Ports `llama_moe_cache`'s `copy_experts` (llama-moe-cache.cpp:901
    /// @b1b770d90), through the impl's `copy_experts` (llama-moe-cache.cpp:726
    /// @b1b770d90):
    /// for large batches, copies the experts of `w` that are in the cache,
    /// from expert `e` up to `last`, to `dst`, the copy of `w`. Large
    /// batches only read the cache, so the experts used in generation stay.
    ///
    /// Return: the number of experts copied, 0 when expert `e` is not
    /// cached.
    pub fn copyExperts(self: *MoeCache, backend_: mod.impl.llama_c.ggml_backend_t, w: *const mod.impl.llama_c.ggml_tensor, dst: *mod.impl.llama_c.ggml_tensor, e: i64, last: i64) i64 {
        const b = self.bindings.get(w) orelse return 0;
        const g = &self.groups.items[@intCast(self.layers[@intCast(b.il)].ig)];
        if (backend_ != self.devices.items[@intCast(g.id)].backend) {
            return 0;
        }

        const slots = g.lru.slot_map[@intCast(b.il)] orelse return 0;
        if (slots[@intCast(e)] >= g.n_slots) {
            return 0;
        }
        var n: i64 = 1;
        while (e + n <= last and slots[@intCast(e + n)] < g.n_slots and slots[@intCast(e + n)] == slots[@intCast(e)] + n) {
            n += 1;
        }

        const bank = g.banks.items[@intCast(b.ip)];
        mod.impl.llama_c.ggml_reset(self.ctx_views);
        const src_view: *mod.impl.llama_c.ggml_tensor = mod.impl.llama_c.ggml_view_3d(self.ctx_views, bank, bank.ne[0], bank.ne[1], n, bank.nb[1], bank.nb[2], @as(usize, @intCast(slots[@intCast(e)])) * bank.nb[2]);
        const dst_view: *mod.impl.llama_c.ggml_tensor = mod.impl.llama_c.ggml_view_3d(self.ctx_views, dst, dst.ne[0], dst.ne[1], n, dst.nb[1], dst.nb[2], @as(usize, @intCast(e)) * dst.nb[2]);
        _ = mod.impl.llama_c.ggml_backend_view_init(src_view);
        _ = mod.impl.llama_c.ggml_backend_view_init(dst_view);
        mod.impl.llama_c.ggml_backend_tensor_copy_async(backend_, backend_, src_view, dst_view);

        self.stats_copy.hits += @intCast(n);
        self.stats_copy.bytes += mod.impl.llama_c.ggml_nbytes(src_view);

        return n;
    }

    /// Ports `llama_moe_cache`'s `copy` (llama-moe-cache.cpp:897 @b1b770d90),
    /// through the impl's `copy` (llama-moe-cache.cpp:761 @b1b770d90): the
    /// scheduler's copy callback for a slot map.
    ///
    /// Return: false when `src` is not a slot map.
    pub fn copy(self: *MoeCache, backend_: mod.impl.llama_c.ggml_backend_t, src: *const mod.impl.llama_c.ggml_tensor, dst: *mod.impl.llama_c.ggml_tensor, graph: *mod.impl.llama_c.ggml_cgraph) bool {
        const lo = self.layer_of.get(src) orelse return false;
        const il = lo.il;
        const has_cpu = lo.has_cpu;
        const l = &self.layers[@intCast(il)];
        const g = &self.groups.items[@intCast(l.ig)];

        mod.impl.assert(backend_ == self.devices.items[@intCast(g.id)].backend, "llama-moe-cache.cpp", 771, "backend == devices[g.id].backend");

        // the get_rows that looks up the slots of the selected experts
        const n_nodes = mod.impl.llama_c.ggml_graph_n_nodes(graph);
        var lookup: ?*const mod.impl.llama_c.ggml_tensor = null;
        var i: c_int = 0;
        while (i < n_nodes and lookup == null) : (i += 1) {
            const node: *const mod.impl.llama_c.ggml_tensor = mod.impl.llama_c.ggml_graph_node(graph, i);
            if (node.op == mod.impl.llama_c.GGML_OP_GET_ROWS and node.src[0] == dst) {
                lookup = node;
            }
        }
        mod.impl.assert(lookup != null, "llama-moe-cache.cpp", 782, "lookup != nullptr");

        // the selected experts must be computed in an earlier split: the
        // scheduler starts a new split at the lookup because it reads a host
        // weight, but only when the split already has inputs
        const sel: *const mod.impl.llama_c.ggml_tensor = lookup.?.src[1];
        i = 0;
        while (i < n_nodes) : (i += 1) {
            const node: *const mod.impl.llama_c.ggml_tensor = mod.impl.llama_c.ggml_graph_node(graph, i);
            if (node == sel or node == @as(?*const mod.impl.llama_c.ggml_tensor, sel.view_src)) {
                mod.impl.llama_c.ggml_abort("llama-moe-cache.cpp", 790, "the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        mod.impl.assert(mod.impl.llama_c.ggml_is_contiguous(sel), "llama-moe-cache.cpp", 793, "ggml_is_contiguous(sel)");
        mod.impl.assert(sel.op == mod.impl.llama_c.GGML_OP_RESHAPE, "llama-moe-cache.cpp", 794, "sel->op == GGML_OP_RESHAPE"); // of the [n_expert_used, n_tokens] ids, see build_moe_cache_slots

        const sel_src: *const mod.impl.llama_c.ggml_tensor = sel.src[0].?;
        const n_used: i32 = @truncate(sel_src.ne[0]);

        self.ids.resize(std.heap.c_allocator, @intCast(mod.impl.llama_c.ggml_nelements(sel))) catch oom();
        mod.impl.llama_c.ggml_backend_tensor_get_async(backend_, sel, if (self.ids.items.len == 0) null else self.ids.items.ptr, 0, mod.impl.llama_c.ggml_nbytes(sel));
        mod.impl.llama_c.ggml_backend_synchronize(backend_);

        if (self.trace != null) {
            self.writeTrace(il, n_used);
        }

        // part of the misses is computed on the CPU once the cache is full,
        // the carry keeps the share right on average
        const share_q: u32 = if (has_cpu and g.lru.n_filled == g.lru.n_slots) self.cpu_share_q else 0;
        var n_hit: usize = 0;
        if (!g.lru.plan(il, self.ids.items, sel_src.ne[1], n_used, self.n_zero, CpuShare{ .share_q = share_q, .acc = &l.cpu_acc }, &self.fills, &self.cpu, &n_hit)) {
            mod.impl.llama_c.ggml_abort("llama-moe-cache.cpp", 815, "the MoE cache is too small for the experts selected in layer %d", il);
        }
        if (has_cpu) {
            const cpu_map: [*]i32 = @ptrCast(@alignCast(l.cpu_map.?.data));
            for (l.cpu_experts.items) |e| {
                cpu_map[@intCast(e)] = -1;
            }
            l.cpu_experts.clearRetainingCapacity();
            for (self.cpu.items) |f| {
                cpu_map[@intCast(f.expert)] = f.expert;
                l.cpu_experts.append(std.heap.c_allocator, f.expert) catch oom();
            }
        }

        // upload the missing experts; consecutive experts going to
        // consecutive slots are uploaded together
        var bytes: usize = 0;
        const fills = self.fills.items;
        for (l.experts[0..l.n_experts], 0..) |w, ip| {
            const bank = g.banks.items[ip];
            const expert_size = w.nb[2];
            var k: usize = 0;
            while (k < fills.len) {
                var n: usize = 1;
                while (k + n < fills.len and fills[k + n].expert == fills[k].expert + @as(i32, @intCast(n)) and fills[k + n].slot == fills[k].slot + @as(i32, @intCast(n))) {
                    n += 1;
                }
                const base: [*]const u8 = @ptrCast(w.data.?);
                mod.impl.llama_c.ggml_backend_tensor_set_async(backend_, bank, base + @as(usize, @intCast(fills[k].expert)) * expert_size, @as(usize, @intCast(fills[k].slot)) * expert_size, n * expert_size);
                bytes += n * expert_size;
                k += n;
            }
        }

        const st = if (self.ids.items.len <= @as(usize, 8) * @as(usize, @intCast(self.n_expert_used))) &self.stats_small else &self.stats_large;
        st.hits += n_hit;
        st.misses += fills.len + self.cpu.items.len;
        st.cpu += self.cpu.items.len;
        st.bytes += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        mod.impl.llama_c.ggml_backend_tensor_set_async(backend_, dst, l.slot_map.?.data, 0, mod.impl.llama_c.ggml_nbytes(l.slot_map.?));

        return true;
    }

    /// Ports `impl`'s `log_stats` (llama-moe-cache.cpp:858 @b1b770d90).
    fn logStats(self: *const MoeCache) void {
        logOne("ubatch <= 8", &self.stats_small);
        logOne("ubatch  > 8", &self.stats_large);
        if (self.stats_copy.hits > 0) {
            mod.impl.logInfo("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", .{ self.stats_copy.hits, @as(f64, @floatFromInt(self.stats_copy.bytes)) / 1024.0 / 1024.0 });
        }
    }

    /// The `log` lambda of `log_stats`.
    fn logOne(name: [*:0]const u8, st: *const Stats) void {
        const n = st.hits + st.misses;
        if (n == 0) {
            return;
        }
        mod.impl.logInfo("llama_moe_cache: %s: hits = %zu, misses = %zu (%zu on the CPU), hit rate = %.2f%%, uploaded = %.2f MiB\n", .{
            name,
            st.hits,
            st.misses,
            st.cpu,
            100.0 * @as(f64, @floatFromInt(st.hits)) / @as(f64, @floatFromInt(n)),
            @as(f64, @floatFromInt(st.bytes)) / 1024.0 / 1024.0,
        });
    }

    /// Ports `llama_moe_cache`'s `memory_breakdown` (llama-moe-cache.cpp:905
    /// @b1b770d90): bytes per buffer type, in the `std::map`'s pointer order.
    ///
    /// Return: the entries, the caller's to free with `impl.allocator`.
    pub fn memoryBreakdown(self: *const MoeCache) []mod.model.BuftSize {
        var res: std.ArrayList(mod.model.BuftSize) = .empty;
        for (self.devices.items) |d| {
            if (d.buf != null) {
                mod.model.addBuftSize(&res, mod.impl.llama_c.ggml_backend_buffer_get_type(d.buf), d.buf_size);
            }
        }
        if (self.buf_host != null) {
            mod.model.addBuftSize(&res, mod.impl.llama_c.ggml_backend_buffer_get_type(self.buf_host), self.buf_host_size);
        }
        return res.toOwnedSlice(std.heap.c_allocator) catch oom();
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(MoeCache);
}
