//! The sampler API's core: the dispatchers, the helpers every sampler shares,
//! the empty placeholder, and the backend-sampling base.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-sampler.cpp` (v1.0.0, `b1b770d90`).
//! Each declaration below names the C++ it replaces and the line it began at.
//! This file and its siblings export the same C symbols with the same
//! signatures, so `llama-context.cpp` and `llama-graph.cpp`, which stay C++
//! until libllama's single swap, link against them unchanged.
//!
//! # Allocation
//!
//! Every sampler and context is allocated with `malloc` through
//! `impl.allocator`, standing in for the C++'s `new`. A `llama_sampler`
//! crosses the C ABI and is freed by whichever entry point the caller
//! chooses, so it cannot belong to a Zig allocator instance.
//!
//! # What is reproduced from libc++, not just from the C++
//!
//! The partial sorts call `libcxx.zig`, which reproduces libc++'s `std::sort`
//! and `std::partial_sort` exactly, because the order they leave tied logits
//! in decides which candidate a cut keeps. See that file and
//! `testing/scripts/libcxx-diff`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// The C++ file every citation and assertion in this directory names.
pub const src = "llama-sampler.cpp";

/// Recovers a sampler's context as its concrete type.
///
/// Parameters:
/// - `T`: the context type the sampler was created with.
/// - `smpl`: the sampler.
///
/// Return: the context, borrowed for the sampler's lifetime.
pub inline fn ctxOf(comptime T: type, smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) *T {
    return @ptrCast(@alignCast(smpl.*.ctx.?));
}

/// Allocates a context, as the C++'s `new T { ... }` does.
///
/// Parameters:
/// - `value`: the initial value.
///
/// Return: the heap copy, freed with `destroy`. Allocation failure aborts,
/// as an uncaught `std::bad_alloc` would.
pub fn create(comptime T: type, value: T) *T {
    const p = std.heap.c_allocator.create(T) catch llama.impl.abort(src, 0, "out of memory");
    p.* = value;
    return p;
}

/// Frees a context allocated with `create`.
///
/// Parameters:
/// - `p`: the context.
///
/// Return: nothing.
pub fn destroy(comptime T: type, p: *T) void {
    std.heap.c_allocator.destroy(p);
}

/// The candidate array's live elements as a slice.
///
/// Parameters:
/// - `cur_p`: the array.
///
/// Return: a slice over `data[0..size]`, borrowed from the array.
pub inline fn items(cur_p: *const llama.impl.llama_c.llama_token_data_array) []llama.impl.llama_c.llama_token_data {
    if (cur_p.size == 0) return &.{};
    return cur_p.data[0..cur_p.size];
}

/// Mirrors arm64's `fcvtzs`, which is what clang emits for `int(float)`.
///
/// The C++ conversion is undefined outside `int`'s range; on this target it
/// truncates toward zero and saturates, and NaN converts to 0. Zig's
/// `@intFromFloat` would trap or be undefined there, and `-inf` logits reach
/// this conversion in the partial sort's bucketing.
///
/// Parameters:
/// - `x`: the float.
///
/// Return: the saturated, truncated integer.
pub fn cvtzs(x: f32) i32 {
    if (std.math.isNan(x)) return 0;
    if (x >= 2147483648.0) return std.math.maxInt(i32);
    if (x <= -2147483648.0) return std.math.minInt(i32);
    return @intFromFloat(x);
}

// -----------------------------------------------------------------------------
// ring_buffer

/// Ports `ring_buffer` (llama-sampler.cpp:26 @b1b770d90).
///
/// A fixed-capacity FIFO. The C++ throws `std::runtime_error` from an empty
/// `front`/`back`/`pop_front` and from a push into zero capacity; nothing
/// catches those, so they terminate, and here they abort with the same text.
pub fn RingBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        capacity: usize = 0,
        sz: usize = 0,
        first: usize = 0,
        pos: usize = 0,
        data: []T = &.{},

        /// Ports the constructor of `ring_buffer` (llama-sampler.cpp:27 @b1b770d90).
        ///
        /// Parameters:
        /// - `cap`: the capacity.
        ///
        /// Return: an empty buffer owning `cap` slots, value-initialised.
        pub fn init(cap: usize) Self {
            const d = std.heap.c_allocator.alloc(T, cap) catch llama.impl.abort(src, 26, "out of memory");
            @memset(d, std.mem.zeroes(T));
            return .{ .capacity = cap, .data = d };
        }

        /// Frees the slots.
        ///
        /// Return: nothing.
        pub fn deinit(self: *Self) void {
            std.heap.c_allocator.free(self.data);
            self.* = .{};
        }

        /// Copies `other` into `self`, as the C++'s implicit copy assignment
        /// does: every field, and the slots.
        ///
        /// Parameters:
        /// - `other`: the buffer to copy.
        ///
        /// Return: nothing.
        pub fn assign(self: *Self, other: *const Self) void {
            if (self.data.len != other.data.len) {
                std.heap.c_allocator.free(self.data);
                self.data = std.heap.c_allocator.alloc(T, other.data.len) catch llama.impl.abort(src, 25, "out of memory");
            }
            @memcpy(self.data, other.data);
            self.capacity = other.capacity;
            self.sz = other.sz;
            self.first = other.first;
            self.pos = other.pos;
        }

        /// Ports `ring_buffer`'s `front` (llama-sampler.cpp:29 @b1b770d90).
        ///
        /// Return: the oldest element.
        pub fn front(self: *const Self) T {
            if (self.sz == 0) llama.impl.throwRuntime(src, 30, "ring buffer is empty");
            return self.data[self.first];
        }

        /// Ports `ring_buffer`'s `push_back` (llama-sampler.cpp:57 @b1b770d90).
        ///
        /// Parameters:
        /// - `value`: the element; the oldest is overwritten when full.
        ///
        /// Return: nothing.
        pub fn pushBack(self: *Self, value: T) void {
            if (self.capacity == 0) llama.impl.throwRuntime(src, 58, "ring buffer: capacity is zero");
            if (self.sz == self.capacity) {
                self.first = (self.first + 1) % self.capacity;
            } else {
                self.sz += 1;
            }
            self.data[self.pos] = value;
            self.pos = (self.pos + 1) % self.capacity;
        }

        /// Ports `ring_buffer`'s `rat` (llama-sampler.cpp:96 @b1b770d90): the
        /// `i`-th element counting back from the newest.
        ///
        /// Parameters:
        /// - `i`: 0 for the newest.
        ///
        /// Return: the element.
        pub fn rat(self: *const Self, i: usize) T {
            if (i >= self.sz) llama.impl.throwRuntime(src, 97, "ring buffer: index out of bounds");
            return self.data[(self.first + self.sz - i - 1) % self.capacity];
        }

        /// Ports `ring_buffer`'s `clear` (llama-sampler.cpp:112 @b1b770d90).
        ///
        /// Return: nothing.
        pub fn clear(self: *Self) void {
            self.sz = 0;
            self.first = 0;
            self.pos = 0;
        }

        /// Ports `ring_buffer`'s `size` (llama-sampler.cpp:123 @b1b770d90).
        ///
        /// Return: the number of elements.
        pub fn size(self: *const Self) usize {
            return self.sz;
        }
    };
}

// -----------------------------------------------------------------------------
// Sorting and normalisation

fn greaterLogit(_: void, a: llama.impl.llama_c.llama_token_data, b: llama.impl.llama_c.llama_token_data) bool {
    return a.logit > b.logit;
}

/// Ports `llama_token_data_array_partial_sort` (llama-sampler.cpp:136 @b1b770d90).
///
/// Buckets the logits over [-10, 10] into 128 bins, keeps the top bins until
/// they hold at least `npartial` candidates, sorts every full bin and
/// partially sorts the last. Writes into `res`; `cur` is not modified.
///
/// **The bucket index is fused.** clang emits `fmadd` for
/// `bucket_scale * val + bucket_inter` -- read off the reference object's
/// disassembly, `fmadd s0, s0, s9, s8` -- so a logit near a bin edge lands
/// in the bin the fused result picks.
///
/// Parameters:
/// - `cur`: the candidates.
/// - `npartial`: how many to sort into the head; at most `cur.size`.
/// - `res`: overwritten with the kept candidates, best first.
///
/// Return: nothing.
pub fn partialSortInto(cur: *const llama.impl.llama_c.llama_token_data_array, npartial: c_int, res: *std.ArrayList(llama.impl.llama_c.llama_token_data)) void {
    const nbuckets = 128;
    const bucket_low: f32 = -10.0;
    const bucket_high: f32 = 10.0;
    const bucket_scale: f32 = @as(f32, nbuckets) / (bucket_high - bucket_low);
    const bucket_inter: f32 = -bucket_low * bucket_scale;

    const n: usize = cur.size;
    const bucket_idx = std.heap.c_allocator.alloc(c_int, n) catch llama.impl.abort(src, 146, "out of memory");
    defer std.heap.c_allocator.free(bucket_idx);
    var histo: [nbuckets]c_int = @splat(0);

    for (0..n) |i| {
        const val = cur.data[i].logit;
        var ib = cvtzs(@mulAdd(f32, val, bucket_scale, bucket_inter));
        ib = stdlib.libcxx.max(c_int, 0, stdlib.libcxx.min(c_int, nbuckets - 1, ib));
        bucket_idx[i] = ib;
        histo[@intCast(ib)] += 1;
    }

    var nhave: c_int = 0;
    var ib: c_int = nbuckets - 1;
    while (ib >= 0) : (ib -= 1) {
        nhave += histo[@intCast(ib)];
        if (nhave >= npartial) break;
    }
    // Every caller passes at most `cur.size`, so the loop always breaks; the
    // C++ would index `histo[-1]` otherwise.
    llama.impl.assert(ib >= 0, src, 163, "ib >= 0");

    res.resize(std.heap.c_allocator, @intCast(nhave)) catch llama.impl.abort(src, 168, "out of memory");
    const out = res.items;

    var bucket_ptrs: [nbuckets]usize = undefined;
    {
        var ptr: usize = 0;
        var j: c_int = nbuckets - 1;
        while (j >= ib) : (j -= 1) {
            bucket_ptrs[@intCast(nbuckets - 1 - j)] = ptr;
            ptr += @intCast(histo[@intCast(j)]);
        }
    }
    for (0..n) |i| {
        const j = bucket_idx[i];
        if (j >= ib) {
            const slot = &bucket_ptrs[@intCast(nbuckets - 1 - j)];
            out[slot.*] = cur.data[i];
            slot.* += 1;
        }
    }

    var ptr: usize = 0;
    var ndone: c_int = 0;
    var j: c_int = nbuckets - 1;
    while (j > ib) : (j -= 1) {
        const h: usize = @intCast(histo[@intCast(j)]);
        stdlib.libcxx.sort(llama.impl.llama_c.llama_token_data, out[ptr .. ptr + h], {}, greaterLogit);
        ptr += h;
        ndone += histo[@intCast(j)];
    }
    const last: usize = @intCast(histo[@intCast(ib)]);
    stdlib.libcxx.partialSort(llama.impl.llama_c.llama_token_data, out[ptr .. ptr + last], @intCast(npartial - ndone), {}, greaterLogit);
}

/// Ports `llama_token_data_array_partial_sort_inplace` (llama-sampler.cpp:194 @b1b770d90).
///
/// Reduces `cur_p` to its top `npartial` candidates, sorted.
///
/// Parameters:
/// - `cur_p`: the candidates; resized and marked sorted.
/// - `npartial`: how many to keep; at most `cur_p.size`.
///
/// Return: nothing.
pub fn partialSortInplace(cur_p: *llama.impl.llama_c.llama_token_data_array, npartial: c_int) void {
    if (npartial <= 128) {
        stdlib.libcxx.partialSort(llama.impl.llama_c.llama_token_data, items(cur_p), @intCast(npartial), {}, greaterLogit);
        cur_p.size = @intCast(npartial);
        cur_p.sorted = true;
        return;
    }

    var tmp: std.ArrayList(llama.impl.llama_c.llama_token_data) = .empty;
    defer tmp.deinit(std.heap.c_allocator);

    partialSortInto(cur_p, npartial, &tmp);

    @memcpy(cur_p.data[0..@intCast(npartial)], tmp.items[0..@intCast(npartial)]);

    cur_p.size = @intCast(npartial);
    cur_p.sorted = true;
}

/// The `p` fields of a candidate array, as libc++'s `discrete_distribution`
/// reads them through `probs_iterator`.
const ProbWeights = struct {
    data: [*c]const llama.impl.llama_c.llama_token_data,
    len: usize,

    pub fn get(self: ProbWeights, i: usize) f64 {
        return self.data[i].p;
    }
};

/// Ports `llama_sample_dist` (llama-sampler.cpp:218 @b1b770d90).
///
/// A fresh `std::discrete_distribution<int>` over the candidates' `p`, drawn
/// once. libc++'s algorithm is reproduced in `libcxx.discreteDraw`.
///
/// Parameters:
/// - `cur_p`: the candidates, with probabilities set.
/// - `rng`: the engine; advanced twice.
///
/// Return: the drawn index.
pub fn sampleDist(cur_p: *const llama.impl.llama_c.llama_token_data_array, rng: *stdlib.libcxx.Mt19937) c_int {
    const w: ProbWeights = .{ .data = cur_p.data, .len = cur_p.size };
    const idx = stdlib.libcxx.discreteDraw(std.heap.c_allocator, w, rng) catch llama.impl.abort(src, 244, "out of memory");
    return @intCast(idx);
}

/// Ports `llama_sampler_temp_impl` (llama-sampler.cpp:266 @b1b770d90).
///
/// Scales the logits by `1/temp`, or at `temp <= 0` keeps only the first
/// maximum and sets every other logit to -inf.
///
/// Parameters:
/// - `cur_p`: the candidates.
/// - `temp`: the temperature.
///
/// Return: nothing.
pub fn tempImpl(cur_p: *llama.impl.llama_c.llama_token_data_array, temp: f32) void {
    if (cur_p.size == 0) return;

    if (temp <= 0.0) {
        var max_i: usize = 0;
        var max_l = cur_p.data[0].logit;

        for (1..cur_p.size) |i| {
            if (cur_p.data[i].logit > max_l) {
                cur_p.data[max_i].logit = -std.math.inf(f32);
                max_i = i;
                max_l = cur_p.data[i].logit;
            } else {
                cur_p.data[i].logit = -std.math.inf(f32);
            }
        }
        return;
    }

    for (items(cur_p)) |*d| d.logit /= temp;
}

/// Ports `llama_sampler_softmax_impl` (llama-sampler.cpp:294 @b1b770d90).
///
/// Parameters:
/// - `cur_p`: the candidates; at least one.
/// - `do_sort`: sort descending first, when not already sorted.
///
/// Return: nothing.
pub fn softmaxImpl(cur_p: *llama.impl.llama_c.llama_token_data_array, do_sort: bool) void {
    llama.impl.assert(cur_p.size > 0, src, 295, "cur_p->size > 0");

    if (do_sort and !cur_p.sorted) {
        partialSortInplace(cur_p, @intCast(cur_p.size));
    }

    var max_l = cur_p.data[0].logit;
    if (!cur_p.sorted) {
        for (1..cur_p.size) |i| max_l = stdlib.libcxx.max(f32, max_l, cur_p.data[i].logit);
    }

    var cum_sum: f32 = 0.0;
    for (items(cur_p)) |*d| {
        const p = stdlib.libm.expf(d.logit - max_l);
        d.p = p;
        cum_sum += p;
    }

    for (items(cur_p)) |*d| d.p /= cum_sum;
}

/// Ports `llama_sampler_top_k_impl` (llama-sampler.cpp:322 @b1b770d90).
///
/// Parameters:
/// - `cur_p`: the candidates.
/// - `k_in`: how many to keep; at most `cur_p.size` are, and none are cut at
///   `k <= 0`.
///
/// Return: nothing.
pub fn topKImpl(cur_p: *llama.impl.llama_c.llama_token_data_array, k_in: i32) void {
    if (k_in <= 0) return;

    const k = stdlib.libcxx.min(c_int, k_in, @intCast(cur_p.size));

    if (!cur_p.sorted) {
        partialSortInplace(cur_p, k);
    }

    cur_p.size = @intCast(k);
}

/// Ports `get_rng_seed` (llama-sampler.cpp:341 @b1b770d90).
///
/// `LLAMA_DEFAULT_SEED` asks for a random seed. The C++ asks
/// `std::random_device().entropy()` once, in a function-local `static`; zero
/// means the device is not a true RNG and the seed comes from
/// `std::chrono::system_clock` instead. What both of those are depends on the
/// libc++ the reference links:
///
/// - **Apple and the BSDs** are `_LIBCPP_USING_ARC4_RANDOM`: `entropy()` is
///   always 32, so the device branch is taken, and it is `arc4random`.
/// - **Windows** is `_LIBCPP_USING_WIN32_RANDOM`, whose `entropy()` is
///   always 0, so the seed is always the clock and `rand_s` is never asked.
/// - **Everywhere else, Linux included**, is `_LIBCPP_USING_DEV_RANDOM`:
///   see `RandomDevice`. A kernel or container that refuses the entropy ioctl
///   makes `entropy()` 0, and the seed is the clock.
///
/// Parameters:
/// - `seed`: the requested seed.
///
/// Return: `seed`, or a random one for `LLAMA_DEFAULT_SEED`.
pub fn getRngSeed(seed: u32) u32 {
    if (seed == llama.impl.llama_c.LLAMA_DEFAULT_SEED) {
        // use system clock if std::random_device is not a true RNG
        if (randomDeviceIsPrng()) return @truncate(@as(u64, @bitCast(systemClockCount())));

        var rd = stdlib.libcxx.RandomDevice.open() catch |e| rdFail(e);
        defer rd.close();
        return rd.next() catch |e| rdFail(e);
    }
    return seed;
}

/// The C++'s `static bool is_rd_prng`, initialised on first use. A C++
/// function-local static is initialised once, under a guard; this caches the
/// answer in an atomic, so two threads racing the first call each open the
/// device once and agree -- the only difference, and not one a caller can
/// see.
var rd_prng_state = std.atomic.Value(u8).init(0); // 0 unknown, 1 false, 2 true

fn randomDeviceIsPrng() bool {
    switch (rd_prng_state.load(.acquire)) {
        1 => return false,
        2 => return true,
        else => {},
    }
    var rd = stdlib.libcxx.RandomDevice.open() catch |e| rdFail(e);
    defer rd.close();
    const is_prng = rd.entropy() == 0;
    rd_prng_state.store(if (is_prng) 2 else 1, .release);
    return is_prng;
}

/// A `std::system_error` from the device: nothing in `get_rng_seed`'s callers
/// catches it, so it terminates -- here, through `ggml_abort` with libc++'s
/// message.
fn rdFail(err: stdlib.libcxx.RandomDevice.Error) noreturn {
    llama.impl.llama_c.ggml_abort("llama-sampler.cpp", 341, "terminating due to uncaught exception of type std::system_error: %s: %s", stdlib.libcxx.RandomDevice.what(err).ptr, stdlib.errno.strerror(std.c._errno().*));
}

/// `std::chrono::system_clock::now().time_since_epoch().count()` in libc++
/// on Linux: `clock_gettime(CLOCK_REALTIME)` in **microseconds**, libc++'s
/// `system_clock::duration`. On Windows libc++ reads
/// `GetSystemTimePreciseAsFileTime`, 100 ns ticks since 1601, and truncates
/// to microseconds since 1970 (`chrono.cpp`'s `__libcpp_system_clock_now`).
fn systemClockCount() i64 {
    // std reads `RtlGetSystemTimePrecise` on Windows, the call under
    // `GetSystemTimePreciseAsFileTime`, and truncates to microseconds alike
    return stdlib.clock.realtimeUs();
}

// -----------------------------------------------------------------------------
// The llama_sampler API

/// Ports `llama_sampler_init` (llama-sampler.cpp:356 @b1b770d90).
///
/// Parameters:
/// - `iface`: the vtable; borrowed, must outlive the sampler.
/// - `ctx`: the sampler's state; owned by the sampler.
///
/// Return: the sampler, freed with `llama_sampler_free`.
pub fn llama_sampler_init(iface: [*c]llama.impl.llama_c.struct_llama_sampler_i, ctx: llama.impl.llama_c.llama_sampler_context_t) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    return create(llama.impl.llama_c.struct_llama_sampler, .{ .iface = iface, .ctx = ctx });
}

/// Ports `llama_sampler_name` (llama-sampler.cpp:365 @b1b770d90).
///
/// Return: the sampler's name, borrowed from the sampler.
pub fn llama_sampler_name(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    if (smpl.*.iface == null) {
        return "(null)";
    }
    return smpl.*.iface.*.name.?(smpl);
}

/// Ports `llama_sampler_accept` (llama-sampler.cpp:373 @b1b770d90).
///
/// Return: nothing.
pub fn llama_sampler_accept(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, token: llama.impl.llama_c.llama_token) callconv(.c) void {
    if (smpl == null) return;

    if (smpl.*.iface.*.accept) |f| f(smpl, token);
}

/// Ports `llama_sampler_apply` (llama-sampler.cpp:383 @b1b770d90).
///
/// Return: nothing.
pub fn llama_sampler_apply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, cur_p: [*c]llama.impl.llama_c.llama_token_data_array) callconv(.c) void {
    if (smpl == null) return;

    llama.impl.assert(smpl.*.iface.*.apply != null, src, 388, "smpl->iface->apply");
    smpl.*.iface.*.apply.?(smpl, cur_p);
}

/// Ports `llama_sampler_reset` (llama-sampler.cpp:392 @b1b770d90).
///
/// Return: nothing.
pub fn llama_sampler_reset(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    if (smpl == null) return;

    if (smpl.*.iface.*.reset) |f| f(smpl);
}

/// Ports `llama_sampler_clone` (llama-sampler.cpp:402 @b1b770d90).
///
/// Return: a new sampler with the same state, or null for a null input.
pub fn llama_sampler_clone(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    if (smpl == null) return null;

    if (smpl.*.iface.*.clone) |f| return f(smpl);

    if (smpl.*.ctx == null) {
        return llama_sampler_init(smpl.*.iface, null);
    }

    llama.impl.abort(src, 418, "the sampler does not support cloning");
}

/// Ports `llama_sampler_free` (llama-sampler.cpp:421 @b1b770d90).
///
/// Return: nothing.
pub fn llama_sampler_free(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    if (smpl == null) return;

    if (smpl.*.iface.*.free) |f| f(smpl);

    destroy(llama.impl.llama_c.struct_llama_sampler, smpl);
}

// -----------------------------------------------------------------------------
// The empty sampler

/// Ports `llama_sampler_empty` (llama-sampler.cpp:435 @b1b770d90).
///
/// The placeholder each `init_*` returns for its disabled configuration --
/// `top_k(0)`, `top_p(1.0)` and so on -- named `?top-k` and the like. It does
/// nothing, and it claims backend support so it never breaks a backend chain.
const Empty = struct {
    name: [*c]const u8,
};

/// Ports `llama_sampler_empty_name` (llama-sampler.cpp:441 @b1b770d90).
fn emptyName(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    return ctxOf(Empty, smpl).name;
}

/// Ports `llama_sampler_empty_accept` (llama-sampler.cpp:446 @b1b770d90).
fn emptyAccept(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, token: llama.impl.llama_c.llama_token) callconv(.c) void {
    _ = smpl;
    _ = token;
}

/// Ports `llama_sampler_empty_apply` (llama-sampler.cpp:451 @b1b770d90).
fn emptyApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, cur_p: [*c]llama.impl.llama_c.llama_token_data_array) callconv(.c) void {
    _ = smpl;
    _ = cur_p;
}

/// Ports `llama_sampler_empty_reset` (llama-sampler.cpp:456 @b1b770d90).
fn emptyReset(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    _ = smpl;
}

/// Ports `llama_sampler_empty_clone` (llama-sampler.cpp:460 @b1b770d90).
fn emptyClone(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    return initEmpty(ctxOf(Empty, smpl).name);
}

/// Ports `llama_sampler_empty_free` (llama-sampler.cpp:465 @b1b770d90).
fn emptyFree(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    destroy(Empty, ctxOf(Empty, smpl));
}

/// Ports `llama_sampler_empty_backend_init` (llama-sampler.cpp:469 @b1b770d90).
fn emptyBackendInit(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, buft: llama.impl.llama_c.ggml_backend_buffer_type_t, n_outputs_max_per_seq: u32) callconv(.c) bool {
    _ = smpl;
    _ = buft;
    _ = n_outputs_max_per_seq;
    return true;
}

/// Ports `llama_sampler_empty_backend_accept` (llama-sampler.cpp:480 @b1b770d90).
fn emptyBackendAccept(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, ctx: ?*llama.impl.llama_c.struct_ggml_context, gf: ?*llama.impl.llama_c.struct_ggml_cgraph, selected: [*c]llama.impl.llama_c.struct_ggml_tensor) callconv(.c) void {
    _ = smpl;
    _ = ctx;
    _ = gf;
    _ = selected;
}

/// Ports `llama_sampler_empty_backend_apply` (llama-sampler.cpp:491 @b1b770d90).
fn emptyBackendApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, ctx: ?*llama.impl.llama_c.struct_ggml_context, gf: ?*llama.impl.llama_c.struct_ggml_cgraph, data: [*c]llama.impl.llama_c.struct_llama_sampler_data) callconv(.c) void {
    _ = smpl;
    _ = ctx;
    _ = gf;
    _ = data;
}

/// Ports `llama_sampler_empty_backend_set_input` (llama-sampler.cpp:502 @b1b770d90).
fn emptyBackendSetInput(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    _ = smpl;
}

/// Ports `llama_sampler_empty_i` (llama-sampler.cpp:506 @b1b770d90).
var empty_iface: llama.impl.llama_c.struct_llama_sampler_i = .{
    .name = emptyName,
    .accept = emptyAccept,
    .apply = emptyApply,
    .reset = emptyReset,
    .clone = emptyClone,
    .free = emptyFree,
    .backend_init = emptyBackendInit,
    .backend_accept = emptyBackendAccept,
    .backend_apply = emptyBackendApply,
    .backend_set_input = emptyBackendSetInput,
    .backend_reset = null,
    .copy_state = null,
};

/// Ports `llama_sampler_init_empty` (llama-sampler.cpp:521 @b1b770d90).
///
/// Parameters:
/// - `name`: the placeholder's name; borrowed, usually a literal.
///
/// Return: the sampler.
pub fn initEmpty(name: [*c]const u8) [*c]llama.impl.llama_c.struct_llama_sampler {
    return llama_sampler_init(&empty_iface, create(Empty, .{ .name = name }));
}

// -----------------------------------------------------------------------------
// The backend-sampling base

/// Ports `llama_sampler_backend` (llama-sampler.cpp:535 @b1b770d90).
///
/// Embedded first in every sampler that has a backend form. Before
/// `backend_init` its name is the bare name; after, `+name` when the device
/// supports every op the sampler needs and `-name` when not.
pub const Backend = struct {
    name: [:0]const u8,
    name_ext: [64]u8 = undefined,
    is_init: bool = false,
    support: bool = false,

    /// Ports the constructor of `llama_sampler_backend` (llama-sampler.cpp:536 @b1b770d90).
    ///
    /// Parameters:
    /// - `name`: the sampler's name; a literal.
    ///
    /// Return: the base, not yet initialised for a backend.
    pub fn init(name: [:0]const u8) Backend {
        return .{ .name = name };
    }

    /// Ports `llama_sampler_backend`'s `get_name` (llama-sampler.cpp:538 @b1b770d90).
    ///
    /// Return: the name, borrowed until the next call.
    pub fn getName(self: *Backend) [*c]const u8 {
        if (!self.is_init) return self.name.ptr;

        self.name_ext[0] = if (self.support) '+' else '-';
        @memcpy(self.name_ext[1 .. 1 + self.name.len], self.name);
        self.name_ext[1 + self.name.len] = 0;
        return &self.name_ext;
    }

    /// Ports `llama_sampler_backend`'s `init` (llama-sampler.cpp:552 @b1b770d90).
    ///
    /// Parameters:
    /// - `support`: whether the device runs every op the sampler needs.
    ///
    /// Return: nothing.
    pub fn initBackend(self: *Backend, support: bool) void {
        llama.impl.assert(self.is_init == false, src, 553, "this->is_init == false");
        self.is_init = true;
        self.support = support;
    }
};

/// The graph `llama_sampler_backend_probe_graph` builds.
pub const Probe = struct {
    ctx: ?*llama.impl.llama_c.struct_ggml_context,
    gf: ?*llama.impl.llama_c.struct_ggml_cgraph,

    /// Frees the probe's context, as `ggml_context_ptr`'s destructor does.
    ///
    /// Return: nothing.
    pub fn deinit(self: Probe) void {
        llama.impl.llama_c.ggml_free(self.ctx);
    }
};

/// Ports `llama_sampler_backend_probe_graph` (llama-sampler.cpp:584 @b1b770d90).
///
/// Builds the sampler's backend graph over a dummy logits tensor, without
/// allocating, so the ops can be checked against a device and counted.
///
/// Parameters:
/// - `sampler`: the sampler; its backend state is reset before and after.
/// - `n_candidates`: the dummy logits' length.
/// - `max_nodes`: the graph's capacity.
/// - `with_candidates`: whether to supply a candidate-id tensor too.
///
/// Return: the probe; free with `deinit`.
pub fn probeGraph(sampler: [*c]llama.impl.llama_c.struct_llama_sampler, n_candidates: i64, max_nodes: u32, with_candidates: bool) Probe {
    const params: llama.impl.llama_c.struct_ggml_init_params = .{
        .mem_size = @as(usize, max_nodes) * llama.impl.llama_c.ggml_tensor_overhead() + llama.impl.llama_c.ggml_graph_overhead_custom(max_nodes, false),
        .mem_buffer = null,
        .no_alloc = true,
    };

    const ctx = llama.impl.llama_c.ggml_init(params) orelse llama.impl.throwRuntime(src, 596, "failed to create ggml context");

    const gf = llama.impl.llama_c.ggml_new_graph_custom(ctx, max_nodes, false);

    var data: llama.impl.llama_c.struct_llama_sampler_data = .{
        .logits = llama.impl.llama_c.ggml_new_tensor_1d(ctx, llama.impl.llama_c.GGML_TYPE_F32, n_candidates),
        .probs = null,
        .sampled = null,
        .candidates = if (with_candidates) llama.impl.llama_c.ggml_new_tensor_1d(ctx, llama.impl.llama_c.GGML_TYPE_I32, n_candidates) else null,
    };

    if (sampler.*.iface.*.backend_reset) |f| f(sampler);
    sampler.*.iface.*.backend_apply.?(sampler, ctx, gf, &data);

    for ([_][*c]llama.impl.llama_c.struct_ggml_tensor{ data.logits, data.probs, data.sampled, data.candidates }) |output| {
        if (output != null) llama.impl.llama_c.ggml_build_forward_expand(gf, output);
    }

    if (sampler.*.iface.*.backend_reset) |f| f(sampler);

    return .{ .ctx = ctx, .gf = gf };
}

/// Ports `llama_sampler_backend_probe_n_nodes` (llama-sampler.cpp:628 @b1b770d90).
///
/// Parameters:
/// - `probe`: a built probe graph.
///
/// Return: the larger of its node count and its tensor count.
pub fn probeNNodes(probe: Probe) u32 {
    var n_tensors: u32 = 0;
    var tensor = llama.impl.llama_c.ggml_get_first_tensor(probe.ctx);
    while (tensor != null) : (tensor = llama.impl.llama_c.ggml_get_next_tensor(probe.ctx, tensor)) {
        n_tensors += 1;
    }
    return stdlib.libcxx.max(u32, @intCast(llama.impl.llama_c.ggml_graph_n_nodes(probe.gf)), n_tensors);
}

/// Ports `llama_sampler_backend_support` (llama-sampler.cpp:639 @b1b770d90).
///
/// Parameters:
/// - `smpl`: the sampler.
/// - `buft`: the buffer type the graph would run from.
///
/// Return: whether the device supports every op in the sampler's graph; the
/// CPU, which has no device here, always does.
pub fn backendSupport(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, buft: llama.impl.llama_c.ggml_backend_buffer_type_t) bool {
    const device = llama.impl.llama_c.ggml_backend_buft_get_device(buft);
    if (device == null) {
        return true;
    }

    const probe = probeGraph(smpl, 128 * 1024, llama.impl.llama_c.GGML_DEFAULT_GRAPH_SIZE, true);
    defer probe.deinit();

    var i: c_int = 0;
    while (i < llama.impl.llama_c.ggml_graph_n_nodes(probe.gf)) : (i += 1) {
        const op = llama.impl.llama_c.ggml_graph_node(probe.gf, i);

        if (!llama.impl.llama_c.ggml_backend_dev_supports_op(device, op)) {
            llama.impl.logWarn("%s: device '%s' does not have support for op %s needed for sampler '%s'\n", .{
                "llama_sampler_backend_support", llama.impl.llama_c.ggml_backend_dev_name(device), llama.impl.llama_c.ggml_op_name(op.*.op), smpl.*.iface.*.name.?(smpl),
            });
            return false;
        }
    }

    return true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "cvtzs saturates and maps NaN to zero, as fcvtzs does" {
    try std.testing.expectEqual(@as(i32, 0), cvtzs(std.math.nan(f32)));
    try std.testing.expectEqual(std.math.minInt(i32), cvtzs(-std.math.inf(f32)));
    try std.testing.expectEqual(std.math.maxInt(i32), cvtzs(std.math.inf(f32)));
    try std.testing.expectEqual(@as(i32, -3), cvtzs(-3.9));
    try std.testing.expectEqual(@as(i32, 3), cvtzs(3.9));
}

test "the ring buffer overwrites its oldest element when full" {
    var rb = RingBuffer(i32).init(3);
    defer rb.deinit();
    for ([_]i32{ 1, 2, 3, 4 }) |v| rb.pushBack(v);
    try std.testing.expectEqual(@as(usize, 3), rb.size());
    try std.testing.expectEqual(@as(i32, 2), rb.front());
    try std.testing.expectEqual(@as(i32, 4), rb.rat(0));
    try std.testing.expectEqual(@as(i32, 2), rb.rat(2));
}
