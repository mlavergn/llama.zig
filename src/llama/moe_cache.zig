//! A device cache for the experts of MoE layers kept in host memory.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-moe-cache.h` -- `llama_moe_cache`
//! - `llama.cpp/src/llama-moe-cache.cpp` -- `moe_cache_slru`, the three
//!   layer helpers and `llama_moe_cache::impl`, whose members `MoeCache`
//!   holds directly (the C++'s pimpl is only a compile firewall)
//!
//! Both at v0.8.0 (`8642541b2`).
//!
//! # How it works
//!
//! Each cached layer has a slot map, a host tensor of `n_expert` slots. The
//! graph looks up the slots of the selected experts with a `GET_ROWS` from
//! the slot map; the slot map is a host weight, so the scheduler copies it
//! to the device with the copy callback, which is where `copy` plans the
//! LRU, uploads the missing experts into the device banks and only then
//! sends the slot map.
//!
//! # Containers
//!
//! The C++'s `bindings` and `layer_of` are `std::unordered_map`s that are
//! only ever looked up, so a hash map of any order stands in for them.

const std = @import("std");
const impl = @import("impl.zig");
const memory = @import("memory.zig");
const model_mod = @import("model.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Tensor = c.ggml_tensor;
const Model = model_mod.Model;

fn oom() noreturn {
    impl.abort("llama-moe-cache.cpp", 0, "out of memory");
}

/// `GGML_PAD` over `size_t`.
fn pad(x: usize, n: usize) usize {
    return (x + n - 1) & ~(n - 1);
}

/// Ports `moe_cache_slru` (llama-moe-cache.cpp:17 @cd959bae9): the SLRU of
/// the experts of a group of layers; the slot of each expert is kept in the
/// slot map of its layer.
pub const Slru = struct {
    n_expert: i32 = 0,
    n_slots: i32 = 0,

    /// `[n_layer]`: the data of the slot maps, -1 where the expert is not
    /// cached.
    slot_map: []?[*]i32 = &.{},
    /// `[n_slots]`: `il*n_expert + expert`, -1 when empty.
    key_of: []i32 = &.{},

    /// The probation and protected LRU lists, in that order; hits promote
    /// slots to the protected half.
    prev: []i32 = &.{},
    next: []i32 = &.{},
    head: [2]i32 = .{ -1, -1 },
    tail: [2]i32 = .{ -1, -1 },
    is_protected: []bool = &.{},
    n_protected: i32 = 0,

    /// `[n_expert]`.
    seen: []u32 = &.{},
    seen_gen: u32 = 0,
    uniq: std.ArrayList(i32) = .empty,

    pub const Fill = struct {
        expert: i32,
        slot: i32,
    };

    pub fn deinit(self: *Slru) void {
        allocator.free(self.slot_map);
        allocator.free(self.key_of);
        allocator.free(self.prev);
        allocator.free(self.next);
        allocator.free(self.is_protected);
        allocator.free(self.seen);
        self.uniq.deinit(allocator);
    }

    /// Ports `moe_cache_slru`'s `init` (llama-moe-cache.cpp:36 @cd959bae9).
    pub fn init(self: *Slru, n_layer: usize, n_expert: i32, n_slots: i32) void {
        self.n_expert = n_expert;
        self.n_slots = n_slots;
        self.slot_map = allocator.alloc(?[*]i32, n_layer) catch oom();
        @memset(self.slot_map, null);
        const ns: usize = @intCast(n_slots);
        self.key_of = allocator.alloc(i32, ns) catch oom();
        @memset(self.key_of, -1);
        self.prev = allocator.alloc(i32, ns) catch oom();
        self.next = allocator.alloc(i32, ns) catch oom();
        var s: i32 = 0;
        while (s < n_slots) : (s += 1) {
            self.prev[@intCast(s)] = s - 1;
            self.next[@intCast(s)] = if (s + 1 < n_slots) s + 1 else -1;
        }
        self.head[0] = 0;
        self.tail[0] = n_slots - 1;
        self.is_protected = allocator.alloc(bool, ns) catch oom();
        @memset(self.is_protected, false);
        self.seen = allocator.alloc(u32, @intCast(n_expert)) catch oom();
        @memset(self.seen, 0);
    }

    /// Ports `moe_cache_slru`'s `move` (llama-moe-cache.cpp:54 @cd959bae9):
    /// moves slot `s` to the most recently used end of a segment.
    fn move(self: *Slru, s: i32, protect: bool) void {
        const su: usize = @intCast(s);
        const from: usize = @intFromBool(self.is_protected[su]);
        const to: usize = @intFromBool(protect);
        if (from == to and s == self.tail[from]) {
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

        self.prev[su] = self.tail[to];
        self.next[su] = -1;
        if (self.tail[to] >= 0) {
            self.next[@intCast(self.tail[to])] = s;
        } else {
            self.head[to] = s;
        }
        self.tail[to] = s;
        self.is_protected[su] = protect;
        self.n_protected += @as(i32, @intCast(to)) - @as(i32, @intCast(from));
    }

    /// Ports `moe_cache_slru`'s `touch` (llama-moe-cache.cpp:82 @cd959bae9):
    /// promotes slot `s`, keeping the protected half at most half the slots.
    fn touch(self: *Slru, s: i32) void {
        self.move(s, true);
        if (self.n_protected > @divTrunc(self.n_slots, 2)) {
            self.move(self.head[1], false);
        }
    }

    /// Ports `moe_cache_slru`'s `victim` (llama-moe-cache.cpp:89 @cd959bae9):
    /// the slot a miss in layer `il` takes, probation first, keeping every
    /// expert selected by this request, including earlier fills.
    fn victim(self: *const Slru, il: i32) i32 {
        for (0..2) |segment| {
            var s = self.head[segment];
            while (s >= 0) : (s = self.next[@intCast(s)]) {
                const key = self.key_of[@intCast(s)];
                if (key < 0 or @divTrunc(key, self.n_expert) != il or self.seen[@intCast(@rem(key, self.n_expert))] != self.seen_gen) {
                    return s;
                }
            }
        }
        impl.abort("llama-moe-cache.cpp", 99, "no SLRU victim");
    }

    /// Ports `moe_cache_slru`'s `plan` (llama-moe-cache.cpp:109 @cd959bae9):
    /// gives a slot to each expert `ids` selects in layer `il`, evicting
    /// from probation first.
    ///
    /// Return: false when the ids select more distinct experts than there
    /// are slots.
    pub fn plan(self: *Slru, il: i32, ids: []const i32, fills: *std.ArrayList(Fill), n_hit: *usize) bool {
        fills.clearRetainingCapacity();
        n_hit.* = 0;

        self.seen_gen +%= 1;
        if (self.seen_gen == 0) {
            @memset(self.seen, 0);
            self.seen_gen = 1;
        }
        self.uniq.clearRetainingCapacity();
        for (ids) |id| {
            impl.assert(id >= 0 and id < self.n_expert, "llama-moe-cache.cpp", 119, "ids[i] >= 0 && ids[i] < n_expert");
            if (self.seen[@intCast(id)] != self.seen_gen) {
                self.seen[@intCast(id)] = self.seen_gen;
                self.uniq.append(allocator, id) catch oom();
            }
        }
        if (self.uniq.items.len > @as(usize, @intCast(self.n_slots))) {
            return false;
        }

        const slots = self.slot_map[@intCast(il)].?;

        // promote hits before choosing victims for misses
        for (self.uniq.items) |e| {
            if (slots[@intCast(e)] >= 0) {
                self.touch(slots[@intCast(e)]);
                n_hit.* += 1;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std.mem.sort(i32, self.uniq.items, {}, std.sort.asc(i32));
        for (self.uniq.items) |e| {
            if (slots[@intCast(e)] >= 0) {
                continue;
            }
            const s = self.victim(il);
            const su: usize = @intCast(s);
            if (self.key_of[su] >= 0) {
                const k = self.key_of[su];
                self.slot_map[@intCast(@divTrunc(k, self.n_expert))].?[@intCast(@rem(k, self.n_expert))] = -1;
            }
            self.key_of[su] = il * self.n_expert + e;
            slots[@intCast(e)] = s;
            self.move(s, false);
            fills.append(allocator, .{ .expert = e, .slot = s }) catch oom();
        }
        return true;
    }
};

/// Ports `llama_moe_cache_layer_experts` (llama-moe-cache.cpp:158
/// @cd959bae9): gate, up, down, or gate_up, down.
///
/// Return: the tensors, at most four, in `out`'s first slots.
fn layerExperts(layer: *const model_mod.Layer, out: *[4]*Tensor) usize {
    var n: usize = 0;
    for ([_]?*Tensor{ layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) |t_| {
        if (t_) |t| {
            out[n] = t;
            n += 1;
        }
    }
    return n;
}

/// Ports `llama_moe_cache_same_layout` (llama-moe-cache.cpp:168 @cd959bae9).
fn sameLayout(a: []const *Tensor, b: []const *Tensor) bool {
    if (a.len != b.len) {
        return false;
    }
    for (a, b) |x, y| {
        if (x.type != y.type or !c.ggml_are_same_shape(x, y) or x.nb[2] != y.nb[2]) {
            return false;
        }
    }
    return true;
}

/// Ports `llama_moe_cache_is_host_weight` (llama-moe-cache.cpp:180 @cd959bae9).
fn isHostWeight(t: *const Tensor) bool {
    return t.buffer != null and
        c.ggml_backend_buffer_get_usage(t.buffer) == c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS and
        c.ggml_backend_buffer_is_host(t.buffer);
}

/// Ports `llama_moe_cache` (llama-moe-cache.h:13 @cd959bae9) and its `impl`
/// (llama-moe-cache.cpp:188 @cd959bae9): keeps recently and repeatedly used
/// experts of host-resident MoE layers in a device buffer.
pub const MoeCache = struct {
    /// A GPU with its own budget and banks; it caches the layers assigned
    /// to it.
    const Device = struct {
        backend: c.ggml_backend_t,
        buft: c.ggml_backend_buffer_type_t,
        /// The host experts of the layers it caches.
        host_bytes: usize = 0,
        /// Its share of the budget.
        split: f64 = 0.0,

        /// The banks and their views.
        ctx: ?*c.ggml_context = null,
        buf: c.ggml_backend_buffer_t = null,
        buf_size: usize = 0,
    };

    /// Layers of one device with one expert tensor layout share the banks
    /// and the LRU of a group.
    const Group = struct {
        /// The device.
        id: i32,
        /// The expert tensors of the first layer.
        ref: std.ArrayList(*Tensor) = .empty,
        layers: std.ArrayList(i32) = .empty,
        /// The device storage of every slot, one per expert tensor.
        banks: std.ArrayList(*Tensor) = .empty,
        host_bytes: usize = 0,
        n_slots: i32 = 0,
        slru: Slru = .{},
    };

    const Layer = struct {
        /// -1 when the layer is not cached.
        ig: i32 = -1,
        /// I32 `[1, n_expert]`, in host memory.
        slot_map: ?*Tensor = null,
        /// The host expert tensors, in the order of the banks.
        experts: [4]*Tensor = undefined,
        n_experts: usize = 0,
    };

    const Binding = struct {
        il: i32,
        /// The index of the bank.
        ip: i32,
        /// The view of the bank used in place of the host experts.
        cached: *Tensor,
    };

    const Stats = struct {
        hits: usize = 0,
        misses: usize = 0,
        bytes: usize = 0,
    };

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
    bindings: std.AutoHashMapUnmanaged(*const Tensor, Binding) = .empty,
    /// A slot map to its layer.
    layer_of: std.AutoHashMapUnmanaged(*const Tensor, i32) = .empty,

    ids: std.ArrayList(i32) = .empty,
    fills: std.ArrayList(Slru.Fill) = .empty,

    /// The slot maps, in host memory.
    ctx_host: ?*c.ggml_context = null,
    buf_host: c.ggml_backend_buffer_t = null,
    buf_host_size: usize = 0,

    /// The views `copyExperts` makes.
    ctx_views: ?*c.ggml_context = null,

    /// Ports the constructor of `llama_moe_cache` (llama-moe-cache.cpp:619
    /// @cd959bae9) and of its `impl` (llama-moe-cache.cpp:256 @cd959bae9).
    ///
    /// Parameters:
    /// - `model`: the model; borrowed for the cache's lifetime.
    /// - `backends`, `bufts`: every backend of the context and its compute
    ///   buffer type; each GPU gets its own cache of `size` bytes, split
    ///   like the layers, for the layers assigned to it.
    /// - `size`: the budget in bytes.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the cache, the caller's to `destroy`, or `error.Thrown`.
    pub fn create(model: *const Model, no_alloc: bool, backends: []const c.ggml_backend_t, bufts: []const c.ggml_backend_buffer_type_t, size: usize, exc: *Exception) Thrown!*MoeCache {
        // `__func__` in the constructor of `struct impl`
        const func = "impl";
        const self = allocator.create(MoeCache) catch oom();
        self.* = .{ .n_expert_used = @intCast(model.hparams.nExpertUsedMax()) };
        errdefer self.destroy();
        self.layers = allocator.alloc(Layer, model.layers.len) catch oom();
        @memset(self.layers, .{});

        for (backends, bufts) |be, buft| {
            const dev_type = c.ggml_backend_dev_type(c.ggml_backend_get_device(be));
            if (dev_type == c.GGML_BACKEND_DEVICE_TYPE_GPU or dev_type == c.GGML_BACKEND_DEVICE_TYPE_IGPU) {
                self.devices.append(allocator, .{ .backend = be, .buft = buft }) catch oom();
            }
        }
        if (self.devices.items.len == 0) {
            return exc.throw(.runtime_error, "MoE cache requires a GPU backend", .{});
        }
        if (model.params.split_mode == c.LLAMA_SPLIT_MODE_TENSOR) {
            return exc.throw(.runtime_error, "MoE cache does not support tensor parallelism", .{});
        }
        if (model.hparams.n_expert == 0 or self.n_expert_used == 0) {
            return exc.throw(.runtime_error, "MoE cache requires a MoE model", .{});
        }

        // only cache the layers that keep all of their experts in host
        // memory, on the device the layer is assigned to
        for (model.layers, 0..) |*mlayer, il| {
            var experts: [4]*Tensor = undefined;
            const n = layerExperts(mlayer, &experts);
            if (n == 0) continue;
            var all_host = true;
            for (experts[0..n]) |t| all_host = all_host and isHostWeight(t);
            if (!all_host) continue;

            var id_opt: ?usize = null;
            for (self.devices.items, 0..) |d, i| {
                if (c.ggml_backend_get_device(d.backend) == model.devLayer(il)) {
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
                self.groups.append(allocator, .{ .id = @intCast(id) }) catch oom();
                const ng = &self.groups.items[self.groups.items.len - 1];
                ng.ref.appendSlice(allocator, experts[0..n]) catch oom();
                break :blk ng;
            };
            g.layers.append(allocator, @intCast(il)) catch oom();
            for (experts[0..n]) |t| {
                g.host_bytes += c.ggml_nbytes(t);
                self.devices.items[id].host_bytes += c.ggml_nbytes(t);
            }
        }
        if (self.groups.items.len == 0) {
            impl.logWarn("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", .{func});
            return self;
        }

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
            const dev = c.ggml_backend_get_device(d.backend);
            if (split_by_free) {
                var free: usize = 0;
                var total: usize = 0;
                c.ggml_backend_dev_memory(dev, &free, &total);
                d.split = @floatFromInt(free);
            } else {
                var at: ?usize = null;
                for (model.devices.items, 0..) |ld, i| {
                    if (ld.dev == dev) {
                        at = i;
                        break;
                    }
                }
                impl.assert(at != null, "llama-moe-cache.cpp", 332, "it != model.devices.end()");
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
        const n_tensors = allocator.alloc(usize, self.devices.items.len) catch oom();
        defer allocator.free(n_tensors);
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
                impl.logWarn("%s: MoE cache budget is too small for %zu layers, they are not cached\n", .{ func, g.layers.items.len });
                g.n_slots = 0;
                continue;
            }
            g.slru.init(model.layers.len, n_expert, g.n_slots);
            n_tensors[@intCast(g.id)] += g.ref.items.len * (1 + g.layers.items.len);
            n_tensors_host += g.layers.items.len;
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

        const buft_host = c.ggml_backend_cpu_buffer_type();
        const alignment_host = c.ggml_backend_buft_get_alignment(buft_host);

        for (self.groups.items, 0..) |*g, ig| {
            if (g.n_slots == 0) {
                continue;
            }
            const ctx = self.devices.items[@intCast(g.id)].ctx;
            for (g.ref.items) |t| {
                const bank: *Tensor = c.ggml_new_tensor_3d(ctx, t.type, t.ne[0], t.ne[1], g.n_slots + 1);
                impl.assert(bank.nb[2] == t.nb[2], "llama-moe-cache.cpp", 400, "bank->nb[2] == t->nb[2]");
                _ = c.ggml_format_name(bank, "moe_cache.%zu.%s", ig, c.ggml_get_name(t));
                g.banks.append(allocator, bank) catch oom();
            }
            for (g.layers.items) |il| {
                const l = &self.layers[@intCast(il)];
                l.ig = @intCast(ig);
                l.n_experts = layerExperts(&model.layers[@intCast(il)], &l.experts);
                for (l.experts[0..l.n_experts], 0..) |e, ip| {
                    const bank = g.banks.items[ip];
                    const cached: *Tensor = c.ggml_view_3d(ctx, bank, bank.ne[0], bank.ne[1], g.n_slots, bank.nb[1], bank.nb[2], 0);
                    _ = c.ggml_format_name(cached, "moe_cache.%s", c.ggml_get_name(e));
                    self.bindings.put(allocator, e, .{ .il = il, .ip = @intCast(ip), .cached = cached }) catch oom();
                }
                const slot_map: *Tensor = c.ggml_new_tensor_2d(self.ctx_host, c.GGML_TYPE_I32, 1, g.ref.items[0].ne[2]);
                _ = c.ggml_format_name(slot_map, "moe_cache.slot_map-%d", il);
                l.slot_map = slot_map;
                self.layer_of.put(allocator, slot_map, il) catch oom();
                self.buf_host_size += pad(c.ggml_nbytes(slot_map), alignment_host);
            }
            self.devices.items[@intCast(g.id)].buf_size += self.allocSize(g, g.n_slots);
        }

        // with no_alloc, dummy buffers are attached so the memory use can
        // still be measured, see llama_context::memory_breakdown
        for (self.devices.items) |*d| {
            const ctx = d.ctx orelse continue;
            d.buf = try memory.Bufs.allocBuf(ctx, d.buft, no_alloc, "MoE cache", exc);
            if (!no_alloc) {
                c.ggml_backend_buffer_clear(d.buf, 0);
                d.buf_size = c.ggml_backend_buffer_get_size(d.buf);
            }
        }
        self.buf_host = try memory.Bufs.allocBuf(self.ctx_host.?, buft_host, no_alloc, "MoE cache slot maps", exc);
        if (!no_alloc) {
            c.ggml_backend_buffer_clear(self.buf_host, 0xff); // all slots are -1
            self.buf_host_size = c.ggml_backend_buffer_get_size(self.buf_host);

            for (self.groups.items) |*g| {
                for (g.layers.items) |il| {
                    if (self.layers[@intCast(il)].slot_map) |sm| {
                        g.slru.slot_map[@intCast(il)] = @ptrCast(@alignCast(sm.data));
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the
        // slot maps are copied with the copy callback
        for (self.devices.items) |*d| {
            if (d.buf != null) {
                c.ggml_backend_buffer_set_usage(d.buf, c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        c.ggml_backend_buffer_set_usage(self.buf_host, c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (self.devices.items, 0..) |*d, id| {
            if (d.host_bytes == 0) {
                continue;
            }
            impl.logInfo("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", .{
                func,
                c.ggml_backend_buft_name(d.buft),
                @as(f64, @floatFromInt(d.buf_size)) / 1024.0 / 1024.0,
                @as(f64, @floatFromInt(d.host_bytes)) / 1024.0 / 1024.0,
            });
            for (self.groups.items) |*g| {
                if (g.id == @as(i32, @intCast(id))) {
                    const n_all: f64 = @floatFromInt(g.layers.items.len * @as(usize, @intCast(g.ref.items[0].ne[2])));
                    impl.logInfo("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", .{
                        func,
                        g.layers.items.len,
                        c.ggml_type_name(g.ref.items[g.ref.items.len - 1].type),
                        g.n_slots,
                        100.0 * @as(f64, @floatFromInt(g.n_slots)) / n_all,
                    });
                }
            }
        }

        return self;
    }

    /// The `init_ctx` lambda of the constructor (llama-moe-cache.cpp:369
    /// @cd959bae9).
    fn initCtx(n_tensors: usize, exc: *Exception) Thrown!*c.ggml_context {
        return c.ggml_init(.{
            .mem_size = n_tensors * c.ggml_tensor_overhead(),
            .mem_buffer = null,
            .no_alloc = true,
        }) orelse return exc.throw(.runtime_error, "failed to create the MoE cache context", .{});
    }

    /// The `alloc_size` lambda of the constructor (llama-moe-cache.cpp:306
    /// @cd959bae9): one extra slot at the end, as CUDA's MMQ can read past
    /// the last expert.
    fn allocSize(self: *const MoeCache, g: *const Group, n_slots: i32) usize {
        const alignment = c.ggml_backend_buft_get_alignment(self.devices.items[@intCast(g.id)].buft);
        var res: usize = 0;
        for (g.ref.items) |t| {
            res += pad(t.nb[2] * @as(usize, @intCast(n_slots + 1)), alignment);
        }
        return res;
    }

    /// Ports the destructor of `llama_moe_cache`'s `impl`
    /// (llama-moe-cache.cpp:475 @cd959bae9): logs the statistics, then frees
    /// the members.
    ///
    /// Return: nothing.
    pub fn destroy(self: *MoeCache) void {
        self.logStats();

        if (self.ctx_views) |ctx| c.ggml_free(ctx);
        if (self.buf_host != null) c.ggml_backend_buffer_free(self.buf_host);
        if (self.ctx_host) |ctx| c.ggml_free(ctx);
        self.fills.deinit(allocator);
        self.ids.deinit(allocator);
        self.layer_of.deinit(allocator);
        self.bindings.deinit(allocator);
        allocator.free(self.layers);
        for (self.groups.items) |*g| {
            g.slru.deinit();
            g.banks.deinit(allocator);
            g.layers.deinit(allocator);
            g.ref.deinit(allocator);
        }
        self.groups.deinit(allocator);
        for (self.devices.items) |*d| {
            if (d.buf != null) c.ggml_backend_buffer_free(d.buf);
            if (d.ctx) |ctx| c.ggml_free(ctx);
        }
        self.devices.deinit(allocator);
        allocator.destroy(self);
    }

    /// Ports `llama_moe_cache`'s `backend` (llama-moe-cache.cpp:625
    /// @cd959bae9).
    ///
    /// Return: the device that caches layer `il`.
    pub fn backend(self: *const MoeCache, il: i32) c.ggml_backend_t {
        return self.devices.items[@intCast(self.groups.items[@intCast(self.layers[@intCast(il)].ig)].id)].backend;
    }

    /// Ports `llama_moe_cache`'s `get_slot_map` (llama-moe-cache.cpp:629
    /// @cd959bae9), through the impl's `get_slot_map` (llama-moe-cache.cpp:479
    /// @cd959bae9).
    /// Large batches use most experts of a layer, so they gain little from
    /// the cache and would evict the experts used in generation.
    ///
    /// Return: the slot map of layer `il`, when its experts can be read from
    /// the cache for `n_tokens` tokens, else null.
    pub fn getSlotMap(self: *const MoeCache, il: i32, n_tokens: i64, n_expert_used: i64) ?*Tensor {
        if (il < 0 or il >= @as(i32, @intCast(self.layers.len)) or self.layers[@intCast(il)].ig < 0) {
            return null;
        }
        const l = &self.layers[@intCast(il)];

        if (n_tokens == 0 or n_tokens > max_batch or @min(n_tokens * n_expert_used, l.slot_map.?.ne[1]) > self.groups.items[@intCast(l.ig)].n_slots) {
            return null;
        }
        return l.slot_map;
    }

    /// Ports `llama_moe_cache`'s `get_experts` (llama-moe-cache.cpp:633
    /// @cd959bae9).
    ///
    /// Return: the experts of `w` in the cache, or null when `w` is not
    /// cached.
    pub fn getExperts(self: *const MoeCache, w: ?*const Tensor) ?*Tensor {
        const b = self.bindings.get(w orelse return null) orelse return null;
        return b.cached;
    }

    /// Ports `llama_moe_cache`'s `copy_experts` (llama-moe-cache.cpp:641
    /// @cd959bae9), through the impl's `copy_experts` (llama-moe-cache.cpp:497
    /// @cd959bae9):
    /// for large batches, copies the experts of `w` that are in the cache,
    /// from expert `e` up to `last`, to `dst`, the copy of `w`. Large
    /// batches only read the cache, so the experts used in generation stay.
    ///
    /// Return: the number of experts copied, 0 when expert `e` is not
    /// cached.
    pub fn copyExperts(self: *MoeCache, backend_: c.ggml_backend_t, w: *const Tensor, dst: *Tensor, e: i64, last: i64) i64 {
        const b = self.bindings.get(w) orelse return 0;
        const g = &self.groups.items[@intCast(self.layers[@intCast(b.il)].ig)];
        if (backend_ != self.devices.items[@intCast(g.id)].backend) {
            return 0;
        }

        const slots = g.slru.slot_map[@intCast(b.il)] orelse return 0;
        if (slots[@intCast(e)] < 0) {
            return 0;
        }
        var n: i64 = 1;
        while (e + n <= last and slots[@intCast(e + n)] == slots[@intCast(e)] + n) {
            n += 1;
        }

        const bank = g.banks.items[@intCast(b.ip)];
        c.ggml_reset(self.ctx_views);
        const src_view: *Tensor = c.ggml_view_3d(self.ctx_views, bank, bank.ne[0], bank.ne[1], n, bank.nb[1], bank.nb[2], @as(usize, @intCast(slots[@intCast(e)])) * bank.nb[2]);
        const dst_view: *Tensor = c.ggml_view_3d(self.ctx_views, dst, dst.ne[0], dst.ne[1], n, dst.nb[1], dst.nb[2], @as(usize, @intCast(e)) * dst.nb[2]);
        _ = c.ggml_backend_view_init(src_view);
        _ = c.ggml_backend_view_init(dst_view);
        c.ggml_backend_tensor_copy_async(backend_, backend_, src_view, dst_view);

        self.stats_copy.hits += @intCast(n);
        self.stats_copy.bytes += c.ggml_nbytes(src_view);

        return n;
    }

    /// Ports `llama_moe_cache`'s `copy` (llama-moe-cache.cpp:637 @cd959bae9),
    /// through the impl's `copy` (llama-moe-cache.cpp:532 @cd959bae9): the
    /// scheduler's copy callback for a slot map.
    ///
    /// Return: false when `src` is not a slot map.
    pub fn copy(self: *MoeCache, backend_: c.ggml_backend_t, src: *const Tensor, dst: *Tensor, graph: *c.ggml_cgraph) bool {
        const il = self.layer_of.get(src) orelse return false;
        const l = &self.layers[@intCast(il)];
        const g = &self.groups.items[@intCast(l.ig)];

        impl.assert(backend_ == self.devices.items[@intCast(g.id)].backend, "llama-moe-cache.cpp", 541, "backend == devices[g.id].backend");

        // the get_rows that looks up the slots of the selected experts
        const n_nodes = c.ggml_graph_n_nodes(graph);
        var lookup: ?*const Tensor = null;
        var i: c_int = 0;
        while (i < n_nodes and lookup == null) : (i += 1) {
            const node: *const Tensor = c.ggml_graph_node(graph, i);
            if (node.op == c.GGML_OP_GET_ROWS and node.src[0] == dst) {
                lookup = node;
            }
        }
        impl.assert(lookup != null, "llama-moe-cache.cpp", 552, "lookup != nullptr");

        // the selected experts must be computed in an earlier split: the
        // scheduler starts a new split at the lookup because it reads a host
        // weight, but only when the split already has inputs
        const sel: *const Tensor = lookup.?.src[1];
        i = 0;
        while (i < n_nodes) : (i += 1) {
            const node: *const Tensor = c.ggml_graph_node(graph, i);
            if (node == sel or node == @as(?*const Tensor, sel.view_src)) {
                c.ggml_abort("llama-moe-cache.cpp", 560, "the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        impl.assert(c.ggml_is_contiguous(sel), "llama-moe-cache.cpp", 563, "ggml_is_contiguous(sel)");

        self.ids.resize(allocator, @intCast(c.ggml_nelements(sel))) catch oom();
        c.ggml_backend_tensor_get_async(backend_, sel, if (self.ids.items.len == 0) null else self.ids.items.ptr, 0, c.ggml_nbytes(sel));
        c.ggml_backend_synchronize(backend_);

        var n_hit: usize = 0;
        if (!g.slru.plan(il, self.ids.items, &self.fills, &n_hit)) {
            c.ggml_abort("llama-moe-cache.cpp", 571, "the MoE cache is too small for the experts selected in layer %d", il);
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
                c.ggml_backend_tensor_set_async(backend_, bank, base + @as(usize, @intCast(fills[k].expert)) * expert_size, @as(usize, @intCast(fills[k].slot)) * expert_size, n * expert_size);
                bytes += n * expert_size;
                k += n;
            }
        }

        const st = if (self.ids.items.len <= @as(usize, 8) * @as(usize, @intCast(self.n_expert_used))) &self.stats_small else &self.stats_large;
        st.hits += n_hit;
        st.misses += fills.len;
        st.bytes += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        c.ggml_backend_tensor_set_async(backend_, dst, src.data, 0, c.ggml_nbytes(src));

        return true;
    }

    /// Ports `impl`'s `log_stats` (llama-moe-cache.cpp:602 @cd959bae9).
    fn logStats(self: *const MoeCache) void {
        logOne("ubatch <= 8", &self.stats_small);
        logOne("ubatch  > 8", &self.stats_large);
        if (self.stats_copy.hits > 0) {
            impl.logInfo("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", .{ self.stats_copy.hits, @as(f64, @floatFromInt(self.stats_copy.bytes)) / 1024.0 / 1024.0 });
        }
    }

    /// The `log` lambda of `log_stats`.
    fn logOne(name: [*:0]const u8, st: *const Stats) void {
        const n = st.hits + st.misses;
        if (n == 0) {
            return;
        }
        impl.logInfo("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n", .{
            name,
            st.hits,
            st.misses,
            100.0 * @as(f64, @floatFromInt(st.hits)) / @as(f64, @floatFromInt(n)),
            @as(f64, @floatFromInt(st.bytes)) / 1024.0 / 1024.0,
        });
    }

    /// Ports `llama_moe_cache`'s `memory_breakdown` (llama-moe-cache.cpp:645
    /// @cd959bae9): bytes per buffer type, in the `std::map`'s pointer order.
    ///
    /// Return: the entries, the caller's to free with `impl.allocator`.
    pub fn memoryBreakdown(self: *const MoeCache) []model_mod.BuftSize {
        var res: std.ArrayList(model_mod.BuftSize) = .empty;
        for (self.devices.items) |d| {
            if (d.buf != null) {
                model_mod.addBuftSize(&res, c.ggml_backend_buffer_get_type(d.buf), d.buf_size);
            }
        }
        if (self.buf_host != null) {
            model_mod.addBuftSize(&res, c.ggml_backend_buffer_get_type(self.buf_host), self.buf_host_size);
        }
        return res.toOwnedSlice(allocator) catch oom();
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(MoeCache);
}

test "the SLRU evicts from probation first, promotes hits and merges sorted misses" {
    var slru: Slru = .{};
    defer slru.deinit();
    slru.init(2, 4, 3);
    var maps: [2][4]i32 = .{ @splat(-1), @splat(-1) };
    slru.slot_map[0] = &maps[0];
    slru.slot_map[1] = &maps[1];

    var fills: std.ArrayList(Slru.Fill) = .empty;
    defer fills.deinit(allocator);
    var n_hit: usize = 0;

    // two misses in layer 0, sorted, to the first free slots
    try std.testing.expect(slru.plan(0, &.{ 3, 1, 3 }, &fills, &n_hit));
    try std.testing.expectEqual(@as(usize, 0), n_hit);
    try std.testing.expectEqual(@as(usize, 2), fills.items.len);
    try std.testing.expectEqual(Slru.Fill{ .expert = 1, .slot = 0 }, fills.items[0]);
    try std.testing.expectEqual(Slru.Fill{ .expert = 3, .slot = 1 }, fills.items[1]);
    try std.testing.expectEqual(@as(i32, 0), slru.n_protected);

    // a hit and a miss: the hit is protected, the miss takes the last free slot
    try std.testing.expect(slru.plan(0, &.{ 1, 2 }, &fills, &n_hit));
    try std.testing.expectEqual(@as(usize, 1), n_hit);
    try std.testing.expectEqual(Slru.Fill{ .expert = 2, .slot = 2 }, fills.items[0]);
    try std.testing.expectEqual(@as(i32, 1), slru.n_protected);

    // layer 1 evicts expert 3 of layer 0, the oldest on probation
    try std.testing.expect(slru.plan(1, &.{0}, &fills, &n_hit));
    try std.testing.expectEqual(Slru.Fill{ .expert = 0, .slot = 1 }, fills.items[0]);
    try std.testing.expectEqual(@as(i32, -1), maps[0][3]);

    // a second hit overflows the protected half, whose oldest goes back
    try std.testing.expect(slru.plan(0, &.{ 1, 2 }, &fills, &n_hit));
    try std.testing.expectEqual(@as(usize, 2), n_hit);
    try std.testing.expectEqual(@as(i32, 1), slru.n_protected);
    try std.testing.expect(!slru.is_protected[0]);
    try std.testing.expect(slru.is_protected[2]);

    // a miss never evicts what the same request selected: expert 1 is on
    // probation, so without that the older slot 1 aside it would go
    try std.testing.expect(slru.plan(0, &.{ 1, 3 }, &fills, &n_hit));
    try std.testing.expectEqual(@as(usize, 1), n_hit);
    try std.testing.expectEqual(@as(usize, 1), fills.items.len);
    try std.testing.expectEqual(@as(i32, 3), fills.items[0].expert);
    try std.testing.expect(maps[0][1] >= 0);

    // more distinct experts than slots
    try std.testing.expect(!slru.plan(0, &.{ 0, 1, 2, 3 }, &fills, &n_hit));
}
