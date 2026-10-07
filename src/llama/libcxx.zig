//! The libc++ algorithms whose results the sampler depends on, reproduced.
//!
//! # Provenance
//!
//! Ported from Apple's libc++ 22.1.6 (`_LIBCPP_VERSION 220106`), the headers
//! in the macOS 27 SDK at `usr/include/c++/v1/`:
//!
//! - `__algorithm/sort.h`, `partial_sort.h`, `make_heap.h`, `sift_down.h`,
//!   `pop_heap.h`, `push_heap.h`, `sort_heap.h`, `upper_bound.h`
//! - `__random/mersenne_twister_engine.h`, `generate_canonical.h`,
//!   `uniform_real_distribution.h`, `discrete_distribution.h`
//!
//! **Not upstream llama.cpp code**, so there is no pinned commit to cite, and
//! the citations below name the header and the template rather than a line.
//! The SDK is not a repository `port-links` can read. What pins this file is
//! the oracle instead: `scripts/libcxx-diff` runs every function here against
//! the real template, instantiated from the same headers, in one process.
//!
//! # Why reproduce a standard library
//!
//! `llama-sampler.cpp` and `llama-grammar.cpp` are compiled against these
//! headers -- `build/llamacpp.zig` passes `-nostdinc++` and the SDK's
//! `c++/v1` -- and three things in them are left to the library by the C++
//! standard:
//!
//! - **The order of equal elements after `std::sort` and
//!   `std::partial_sort`.** The samplers sort candidates on logit, ties are
//!   common in a quantized model's output, and the order decides which tied
//!   token survives a top-k cut or is drawn by `dist`. Any correct sort would
//!   pass a test that checks sortedness; only this one matches.
//! - **`std::discrete_distribution`**, whose sampling algorithm the standard
//!   does not specify. libc++ normalises, keeps the first n-1 partial sums,
//!   and does an `upper_bound` on one `uniform_real_distribution<double>`
//!   draw.
//! - **`std::generate_canonical`**, whose formula the standard does give but
//!   whose arithmetic type and draw count follow from template arguments:
//!   two engine draws for `double`, **one** for `float`, which can return
//!   exactly 1.0. Reproduced, not fixed.
//!
//! `std::mt19937` is fully specified by the standard and is here because
//! the samplers need one, not because it varies.
//!
//! The classic (non-branchless) introsort is the only one reproduced: libc++
//! picks the branchless variant only for arithmetic element types compared
//! with `std::less` or `std::greater`, and every sort in the sampler and
//! grammar passes a lambda.

const std = @import("std");

// -----------------------------------------------------------------------------
// Sorting

/// Mirrors `std::is_arithmetic`: integers, floats and `bool`.
fn isArithmetic(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int, .float, .bool, .comptime_int, .comptime_float => true,
        else => false,
    };
}

/// Builds the sort and heap routines for one element type and comparator.
///
/// Mirrors libc++'s templates with `_ClassicAlgPolicy`, positions as indices
/// into one slice in place of iterators. `lessThan(ctx, a, b)` is the C++
/// comparator `comp(a, b)`; arguments are passed in the order the C++ passes
/// them, which matters when the comparator is not a strict weak ordering.
///
/// Parameters:
/// - `T`: the element type.
/// - `Context`: the comparator's state, passed by value.
/// - `lessThan`: the comparator.
///
/// Return: a namespace of sorting functions over `[]T`.
pub fn Sorter(comptime T: type, comptime Context: type, comptime lessThan: fn (Context, T, T) bool) type {
    return struct {
        const Self = @This();

        items: []T,
        ctx: Context,

        inline fn less(self: Self, a: usize, b: usize) bool {
            return lessThan(self.ctx, self.items[a], self.items[b]);
        }

        inline fn swap(self: Self, a: usize, b: usize) void {
            std.mem.swap(T, &self.items[a], &self.items[b]);
        }

        /// Ports `std::__sort3`, the `!__use_branchless_sort` overload (sort.h).
        pub fn sort3(self: Self, x: usize, y: usize, z: usize) bool {
            if (!self.less(y, x)) {
                if (!self.less(z, y)) return false;
                self.swap(y, z);
                if (self.less(y, x)) self.swap(x, y);
                return true;
            }
            if (self.less(z, y)) {
                self.swap(x, z);
                return true;
            }
            self.swap(x, y);
            if (self.less(z, y)) self.swap(y, z);
            return true;
        }

        /// Ports `std::__sort4`, the `!__use_branchless_sort` overload (sort.h).
        pub fn sort4(self: Self, x1: usize, x2: usize, x3: usize, x4: usize) void {
            _ = self.sort3(x1, x2, x3);
            if (self.less(x4, x3)) {
                self.swap(x3, x4);
                if (self.less(x3, x2)) {
                    self.swap(x2, x3);
                    if (self.less(x2, x1)) self.swap(x1, x2);
                }
            }
        }

        /// Ports `std::__sort5`, the `!__use_branchless_sort` overload (sort.h).
        pub fn sort5(self: Self, x1: usize, x2: usize, x3: usize, x4: usize, x5: usize) void {
            self.sort4(x1, x2, x3, x4);
            if (self.less(x5, x4)) {
                self.swap(x4, x5);
                if (self.less(x4, x3)) {
                    self.swap(x3, x4);
                    if (self.less(x3, x2)) {
                        self.swap(x2, x3);
                        if (self.less(x2, x1)) self.swap(x1, x2);
                    }
                }
            }
        }

        /// Ports `std::__insertion_sort` (sort.h).
        pub fn insertionSort(self: Self, first: usize, last: usize) void {
            if (first == last) return;
            var i = first + 1;
            while (i != last) : (i += 1) {
                var j = i - 1;
                if (self.less(i, j)) {
                    const t = self.items[i];
                    var k = j;
                    j = i;
                    while (true) {
                        self.items[j] = self.items[k];
                        j = k;
                        if (j == first) break;
                        k -= 1;
                        if (!lessThan(self.ctx, t, self.items[k])) break;
                    }
                    self.items[j] = t;
                }
            }
        }

        /// Ports `std::__insertion_sort_unguarded` (sort.h).
        ///
        /// Relies on the element before `first` being no greater than any in
        /// the range, which the caller guarantees by only using it on a
        /// partition that is not leftmost.
        pub fn insertionSortUnguarded(self: Self, first: usize, last: usize) void {
            if (first == last) return;
            var i = first + 1;
            while (i != last) : (i += 1) {
                var j = i - 1;
                if (self.less(i, j)) {
                    const t = self.items[i];
                    var k = j;
                    j = i;
                    while (true) {
                        self.items[j] = self.items[k];
                        j = k;
                        k -= 1;
                        if (!lessThan(self.ctx, t, self.items[k])) break;
                    }
                    self.items[j] = t;
                }
            }
        }

        /// Ports `std::__insertion_sort_incomplete` (sort.h).
        ///
        /// Return: true when the range ended up sorted, false when it gave up
        /// after eight moves.
        pub fn insertionSortIncomplete(self: Self, first: usize, last: usize) bool {
            switch (last - first) {
                0, 1 => return true,
                2 => {
                    if (self.less(last - 1, first)) self.swap(first, last - 1);
                    return true;
                },
                3 => {
                    _ = self.sort3(first, first + 1, last - 1);
                    return true;
                },
                4 => {
                    self.sort4(first, first + 1, first + 2, last - 1);
                    return true;
                },
                5 => {
                    self.sort5(first, first + 1, first + 2, first + 3, last - 1);
                    return true;
                },
                else => {},
            }
            var j = first + 2;
            _ = self.sort3(first, first + 1, j);
            const limit: u32 = 8;
            var count: u32 = 0;
            var i = j + 1;
            while (i != last) : (i += 1) {
                if (self.less(i, j)) {
                    const t = self.items[i];
                    var k = j;
                    j = i;
                    while (true) {
                        self.items[j] = self.items[k];
                        j = k;
                        if (j == first) break;
                        k -= 1;
                        if (!lessThan(self.ctx, t, self.items[k])) break;
                    }
                    self.items[j] = t;
                    count += 1;
                    if (count == limit) return i + 1 == last;
                }
                j = i;
            }
            return true;
        }

        const Partition = struct { pivot: usize, already_partitioned: bool };

        /// Ports `std::__partition_with_equals_on_right` (sort.h).
        pub fn partitionEqualsOnRight(self: Self, first_in: usize, last_in: usize) Partition {
            var first = first_in;
            var last = last_in;
            const begin = first;
            const pivot = self.items[first];
            while (true) {
                first += 1;
                if (!lessThan(self.ctx, self.items[first], pivot)) break;
            }
            if (begin == first - 1) {
                while (first < last) {
                    last -= 1;
                    if (lessThan(self.ctx, self.items[last], pivot)) break;
                }
            } else {
                while (true) {
                    last -= 1;
                    if (lessThan(self.ctx, self.items[last], pivot)) break;
                }
            }
            const already = first >= last;
            while (first < last) {
                self.swap(first, last);
                while (true) {
                    first += 1;
                    if (!lessThan(self.ctx, self.items[first], pivot)) break;
                }
                while (true) {
                    last -= 1;
                    if (lessThan(self.ctx, self.items[last], pivot)) break;
                }
            }
            const pivot_pos = first - 1;
            if (begin != pivot_pos) self.items[begin] = self.items[pivot_pos];
            self.items[pivot_pos] = pivot;
            return .{ .pivot = pivot_pos, .already_partitioned = already };
        }

        /// Ports `std::__partition_with_equals_on_left` (sort.h).
        ///
        /// Return: the new start of the range still to sort.
        pub fn partitionEqualsOnLeft(self: Self, first_in: usize, last_in: usize) usize {
            var first = first_in;
            var last = last_in;
            const begin = first;
            const pivot = self.items[first];
            if (lessThan(self.ctx, pivot, self.items[last - 1])) {
                while (true) {
                    first += 1;
                    if (lessThan(self.ctx, pivot, self.items[first])) break;
                }
            } else {
                while (true) {
                    first += 1;
                    if (!(first < last)) break;
                    if (lessThan(self.ctx, pivot, self.items[first])) break;
                }
            }
            if (first < last) {
                while (true) {
                    last -= 1;
                    if (!lessThan(self.ctx, pivot, self.items[last])) break;
                }
            }
            while (first < last) {
                self.swap(first, last);
                while (true) {
                    first += 1;
                    if (lessThan(self.ctx, pivot, self.items[first])) break;
                }
                while (true) {
                    last -= 1;
                    if (!lessThan(self.ctx, pivot, self.items[last])) break;
                }
            }
            const pivot_pos = first - 1;
            if (begin != pivot_pos) self.items[begin] = self.items[pivot_pos];
            self.items[pivot_pos] = pivot;
            return first;
        }

        /// Ports `std::__introsort` with `_UseBitSetPartition = false` (sort.h).
        pub fn introsort(self: Self, first_in: usize, last_in: usize, depth_in: usize, leftmost_in: bool) void {
            const limit = 24;
            const ninther_threshold = 128;
            var first = first_in;
            var last = last_in;
            var depth = depth_in;
            var leftmost = leftmost_in;
            while (true) {
                const len = last - first;
                switch (len) {
                    0, 1 => return,
                    2 => {
                        if (self.less(last - 1, first)) self.swap(first, last - 1);
                        return;
                    },
                    3 => {
                        _ = self.sort3(first, first + 1, last - 1);
                        return;
                    },
                    4 => {
                        self.sort4(first, first + 1, first + 2, last - 1);
                        return;
                    },
                    5 => {
                        self.sort5(first, first + 1, first + 2, first + 3, last - 1);
                        return;
                    },
                    else => {},
                }
                if (len < limit) {
                    if (leftmost) {
                        self.insertionSort(first, last);
                    } else {
                        self.insertionSortUnguarded(first, last);
                    }
                    return;
                }
                if (depth == 0) {
                    self.partialSortRange(first, last, last);
                    return;
                }
                depth -= 1;
                {
                    const half = len / 2;
                    if (len > ninther_threshold) {
                        _ = self.sort3(first, first + half, last - 1);
                        _ = self.sort3(first + 1, first + (half - 1), last - 2);
                        _ = self.sort3(first + 2, first + (half + 1), last - 3);
                        _ = self.sort3(first + (half - 1), first + half, first + (half + 1));
                        self.swap(first, first + half);
                    } else {
                        _ = self.sort3(first + half, first, last - 1);
                    }
                }
                if (!leftmost and !self.less(first - 1, first)) {
                    first = self.partitionEqualsOnLeft(first, last);
                    continue;
                }
                const ret = self.partitionEqualsOnRight(first, last);
                var i = ret.pivot;
                if (ret.already_partitioned) {
                    const fs = self.insertionSortIncomplete(first, i);
                    if (self.insertionSortIncomplete(i + 1, last)) {
                        if (fs) return;
                        last = i;
                        continue;
                    } else if (fs) {
                        i += 1;
                        first = i;
                        continue;
                    }
                }
                self.introsort(first, i, depth, leftmost);
                leftmost = false;
                i += 1;
                first = i;
            }
        }

        /// Ports `std::__sift_down` (sift_down.h). `first` is the heap's
        /// base; `start` and `len` are relative to it.
        ///
        /// With `assume_both_children`, every parent visited is taken to have
        /// a right child: the larger child is picked by adding the
        /// comparison's result, with no bounds test. `make_heap` makes that
        /// true by sifting over an even-trimmed length.
        pub fn siftDown(self: Self, comptime assume_both_children: bool, first: usize, len: isize, start_in: isize) void {
            var start = start_in;
            var child = start;
            if (len < 2 or @divTrunc(len - 2, 2) < child) return;
            child = 2 * child + 1;
            if (assume_both_children) {
                child += @intFromBool(self.lessAt(first, child, child + 1));
            } else if (child + 1 < len and self.lessAt(first, child, child + 1)) child += 1;
            if (self.lessAt(first, child, start)) return;
            const top = self.items[first + @as(usize, @intCast(start))];
            while (true) {
                self.items[first + @as(usize, @intCast(start))] = self.items[first + @as(usize, @intCast(child))];
                start = child;
                if (@divTrunc(len - 2, 2) < child) break;
                child = 2 * child + 1;
                if (assume_both_children) {
                    child += @intFromBool(self.lessAt(first, child, child + 1));
                } else if (child + 1 < len and self.lessAt(first, child, child + 1)) child += 1;
                if (lessThan(self.ctx, self.items[first + @as(usize, @intCast(child))], top)) break;
            }
            self.items[first + @as(usize, @intCast(start))] = top;
        }

        inline fn lessAt(self: Self, first: usize, a: isize, b: isize) bool {
            return self.less(first + @as(usize, @intCast(a)), first + @as(usize, @intCast(b)));
        }

        /// Ports `std::__floyd_sift_down` (sift_down.h).
        ///
        /// Return: the index the hole ended at.
        pub fn floydSiftDown(self: Self, first: usize, len: isize) usize {
            var hole = first;
            var child_i = first;
            var child: isize = 0;
            while (true) {
                child_i += @intCast(child + 1);
                child = 2 * child + 1;
                if (child + 1 < len and self.less(child_i, child_i + 1)) {
                    child_i += 1;
                    child += 1;
                }
                self.items[hole] = self.items[child_i];
                hole = child_i;
                if (child > @divTrunc(len - 2, 2)) return hole;
            }
        }

        /// Ports `std::__sift_up` (push_heap.h). `last` is one past the
        /// element to sift; `len_in` the heap length up to it.
        pub fn siftUp(self: Self, first: usize, last_in: usize, len_in: isize) void {
            if (len_in > 1) {
                var len = @divTrunc(len_in - 2, 2);
                var ptr = first + @as(usize, @intCast(len));
                var last = last_in - 1;
                if (self.less(ptr, last)) {
                    const t = self.items[last];
                    while (true) {
                        self.items[last] = self.items[ptr];
                        last = ptr;
                        if (len == 0) break;
                        len = @divTrunc(len - 1, 2);
                        ptr = first + @as(usize, @intCast(len));
                        if (!lessThan(self.ctx, self.items[ptr], t)) break;
                    }
                    self.items[last] = t;
                }
            }
        }

        /// Ports `std::__pop_heap` (pop_heap.h).
        pub fn popHeap(self: Self, first: usize, last_in: usize, len: isize) void {
            if (len > 1) {
                const top = self.items[first];
                var hole = self.floydSiftDown(first, len);
                const last = last_in - 1;
                if (hole == last) {
                    self.items[hole] = top;
                } else {
                    self.items[hole] = self.items[last];
                    hole += 1;
                    self.items[last] = top;
                    self.siftUp(first, hole, @intCast(hole - first));
                }
            }
        }

        /// Ports `std::__make_heap` (make_heap.h).
        ///
        /// **libc++ heapifies arithmetic types differently.**
        /// `__assume_both_children` is `is_arithmetic<value_type>`: for those
        /// it sifts over the length rounded down to odd -- so every parent
        /// has two children -- and then sifts the last element up. A
        /// `llama_token_data` takes the general path; typical's `size_t`
        /// indices take this one. Both arrangements are valid heaps, and they
        /// differ, so a heap sort of tied keys leaves them in different
        /// orders. Found by `make sampler-diff`, at 32000 candidates, where
        /// introsort's depth limit sends typical's index sort to the heap.
        pub fn makeHeap(self: Self, first: usize, last: usize) void {
            const assume_both_children = comptime isArithmetic(T);
            const n: isize = @intCast(last - first);
            const sift_down_n: isize = if (assume_both_children) (if ((n & 1) != 0) n else n - 1) else n;
            if (n > 1) {
                var start = @divTrunc(sift_down_n - 2, 2);
                while (start >= 0) : (start -= 1) {
                    self.siftDown(assume_both_children, first, sift_down_n, start);
                }
                if (assume_both_children) self.siftUp(first, last, n);
            }
        }

        /// Ports `std::__sort_heap` (sort_heap.h).
        pub fn sortHeap(self: Self, first: usize, last_in: usize) void {
            var last = last_in;
            var n: isize = @intCast(last - first);
            while (n > 1) : ({
                last -= 1;
                n -= 1;
            }) {
                self.popHeap(first, last, n);
            }
        }

        /// Ports `std::__partial_sort_impl` (partial_sort.h).
        pub fn partialSortRange(self: Self, first: usize, middle: usize, last: usize) void {
            if (first == middle) return;
            self.makeHeap(first, middle);
            const len: isize = @intCast(middle - first);
            var i = middle;
            while (i != last) : (i += 1) {
                if (self.less(i, first)) {
                    self.swap(i, first);
                    self.siftDown(false, first, len, 0);
                }
            }
            self.sortHeap(first, middle);
        }
    };
}

/// Sorts `items` exactly as libc++'s `std::sort` does.
///
/// Ports `std::sort` through `std::__sort_dispatch` (sort.h): introsort with
/// a depth limit of twice the floor of log2 of the length.
///
/// Parameters:
/// - `T`: the element type.
/// - `items`: the range, sorted in place.
/// - `context`: passed to every comparison.
/// - `lessThan`: the comparator, `comp(a, b)` in the C++.
///
/// Return: nothing.
pub fn sort(comptime T: type, items: []T, context: anytype, comptime lessThan: fn (@TypeOf(context), T, T) bool) void {
    if (items.len == 0) return;
    const S = Sorter(T, @TypeOf(context), lessThan);
    const s: S = .{ .items = items, .ctx = context };
    const depth: usize = 2 * @as(usize, std.math.log2_int(usize, items.len));
    s.introsort(0, items.len, depth, true);
}

/// Sorts the smallest `middle` elements of `items` into place, exactly as
/// libc++'s `std::partial_sort` does; the rest are left in heap-sift order.
///
/// Ports `std::partial_sort` (partial_sort.h).
///
/// Parameters:
/// - `T`: the element type.
/// - `items`: the whole range.
/// - `middle`: how many to sort; at most `items.len`.
/// - `context`: passed to every comparison.
/// - `lessThan`: the comparator.
///
/// Return: nothing.
pub fn partialSort(comptime T: type, items: []T, middle: usize, context: anytype, comptime lessThan: fn (@TypeOf(context), T, T) bool) void {
    const S = Sorter(T, @TypeOf(context), lessThan);
    const s: S = .{ .items = items, .ctx = context };
    s.partialSortRange(0, middle, items.len);
}

// -----------------------------------------------------------------------------
// Random numbers

/// `std::mt19937`, reproduced from libc++'s incremental form.
///
/// Ports `mersenne_twister_engine<uint_fast32_t, 32, 624, 397, 31,
/// 0x9908b0df, 11, 0xffffffff, 7, 0x9d2c5680, 15, 0xefc60000, 18,
/// 1812433253>` (mersenne_twister_engine.h). `uint_fast32_t` is `uint32_t` on
/// Darwin, so the result type is `u32`.
///
/// A plain value: copying it copies the state, as assigning one
/// `std::mt19937` to another does.
pub const Mt19937 = struct {
    const n = 624;
    const m = 397;

    x: [n]u32,
    i: usize,

    /// The engine's `default_seed`, 5489.
    pub const default_seed: u32 = 5489;

    /// Constructs an engine seeded with `sd`, as `std::mt19937(sd)` does.
    ///
    /// Parameters:
    /// - `sd`: the seed.
    ///
    /// Return: the seeded engine.
    pub fn init(sd: u32) Mt19937 {
        var e: Mt19937 = undefined;
        e.seed(sd);
        return e;
    }

    /// Ports `mersenne_twister_engine::seed(result_type)`.
    ///
    /// Parameters:
    /// - `sd`: the seed.
    ///
    /// Return: nothing.
    pub fn seed(self: *Mt19937, sd: u32) void {
        self.x[0] = sd;
        for (1..n) |k| {
            const prev = self.x[k - 1];
            self.x[k] = 1812433253 *% (prev ^ (prev >> 30)) +% @as(u32, @truncate(k));
        }
        self.i = 0;
    }

    /// Ports `mersenne_twister_engine::operator()()`.
    ///
    /// Return: the next 32-bit output.
    pub fn next(self: *Mt19937) u32 {
        const j = (self.i + 1) % n;
        const mask: u32 = (@as(u32, 1) << 31) - 1;
        const yp = (self.x[self.i] & ~mask) | (self.x[j] & mask);
        const k = (self.i + m) % n;
        self.x[self.i] = self.x[k] ^ (yp >> 1) ^ (0x9908b0df *% (yp & 1));
        var z = self.x[self.i] ^ ((self.x[self.i] >> 11) & 0xffffffff);
        self.i = j;
        z ^= (z << 7) & 0x9d2c5680;
        z ^= (z << 15) & 0xefc60000;
        return z ^ (z >> 18);
    }
};

/// Ports `std::generate_canonical<double, 53>` over an `Mt19937`
/// (generate_canonical.h).
///
/// Two draws: `k = 53 / 32 + (53 % 32 != 0)`. The first draw is the low word.
///
/// Parameters:
/// - `g`: the engine; advanced twice.
///
/// Return: a double in [0, 1).
pub fn canonicalF64(g: *Mt19937) f64 {
    const rp: f64 = @as(f64, @floatFromInt(@as(u32, 0xffffffff))) + 1.0;
    var base: f64 = rp;
    var sp: f64 = @floatFromInt(g.next());
    // `__sp += (__g() - _URNG::min()) * __base;` -- the product is exact
    // (32 bits times a power of two), so contraction cannot move it.
    sp += @as(f64, @floatFromInt(g.next())) * base;
    base *= rp;
    return sp / base;
}

/// Ports `std::generate_canonical<float, 24>` over an `Mt19937`
/// (generate_canonical.h).
///
/// **One** draw: `k = 24 / 32 + (24 % 32 != 0) = 1`. The draw is rounded to
/// `float` and divided by 2^32, so any draw within 128 of the top returns
/// exactly 1.0 -- outside `[0, 1)`. That is libc++'s behaviour and the
/// sampler that uses it (XTC) inherits it; reproduced, not fixed.
///
/// Parameters:
/// - `g`: the engine; advanced once.
///
/// Return: a float in [0, 1].
pub fn canonicalF32(g: *Mt19937) f32 {
    const rp: f32 = @as(f32, @floatFromInt(@as(u32, 0xffffffff))) + 1.0;
    const sp: f32 = @floatFromInt(g.next());
    return sp / rp;
}

/// Ports `std::uniform_real_distribution<double>(0, 1)` (uniform_real_distribution.h).
///
/// `(b - a) * canonical + a` with `a = 0`, `b = 1` is the canonical value
/// itself, fused or not: `1 * x` and `x + 0` are both exact for `x >= 0`.
///
/// Parameters:
/// - `g`: the engine; advanced twice.
///
/// Return: a double in [0, 1).
pub fn uniformF64(g: *Mt19937) f64 {
    return canonicalF64(g);
}

/// Ports `std::uniform_real_distribution<float>(0, 1)` (uniform_real_distribution.h).
///
/// Parameters:
/// - `g`: the engine; advanced once.
///
/// Return: a float in [0, 1].
pub fn uniformF32(g: *Mt19937) f32 {
    return canonicalF32(g);
}

/// Ports `std::discrete_distribution<int>` constructed over `weights` and
/// drawn once (discrete_distribution.h).
///
/// libc++'s `param_type::__init` sums the weights as doubles, divides each by
/// the sum, and keeps the partial sums of all but the last; `operator()`
/// draws one `uniform_real_distribution<double>` and returns the
/// `upper_bound` index. One weight, or none, leaves no partial sums and
/// always returns 0.
///
/// Parameters:
/// - `allocator`: for the partial sums; freed before returning.
/// - `weights`: the weights, read as `float` and widened, as the C++ reads
///   them through an iterator over `llama_token_data::p`.
/// - `g`: the engine; advanced twice.
///
/// Return: the drawn index.
pub fn discreteDraw(allocator: std.mem.Allocator, weights: anytype, g: *Mt19937) error{OutOfMemory}!usize {
    const count = weights.len;
    var cum: []f64 = &.{};
    defer allocator.free(cum);

    if (count > 1) {
        var p = try allocator.alloc(f64, count);
        defer allocator.free(p);
        var s: f64 = 0.0;
        for (0..count) |k| {
            p[k] = weights.get(k);
            s += p[k];
        }
        for (0..count) |k| p[k] /= s;
        cum = try allocator.alloc(f64, count - 1);
        var acc: f64 = p[0];
        cum[0] = acc;
        for (1..count - 1) |k| {
            acc = acc + p[k];
            cum[k] = acc;
        }
    }

    const u = uniformF64(g);
    return upperBound(cum, u);
}

/// Ports `std::upper_bound` with `std::__less<>` over doubles (upper_bound.h).
///
/// Parameters:
/// - `items`: a sorted range.
/// - `value`: the value searched for.
///
/// Return: the index of the first element greater than `value`.
pub fn upperBound(items: []const f64, value: f64) usize {
    var first: usize = 0;
    var len: usize = items.len;
    while (len != 0) {
        const half = len / 2;
        const mid = first + half;
        if (value < items[mid]) {
            len = half;
        } else {
            first = mid + 1;
            len -= half + 1;
        }
    }
    return first;
}

/// Ports `std::max(a, b)` for floats: `(a < b) ? b : a`.
///
/// Not `@max`, which returns the non-NaN operand. `std::max` returns `a`
/// whenever the comparison is false, NaN included.
///
/// Return: the larger operand, by libc++'s rule.
pub inline fn max(comptime T: type, a: T, b: T) T {
    return if (a < b) b else a;
}

/// Ports `std::min(a, b)`: `(b < a) ? b : a`.
///
/// Return: the smaller operand, by libc++'s rule.
pub inline fn min(comptime T: type, a: T, b: T) T {
    return if (b < a) b else a;
}

// -----------------------------------------------------------------------------
// std::priority_queue

/// Ports `std::priority_queue<T, std::vector<T>, Compare>`: `push` is
/// `push_back` then `std::push_heap`, `pop` is `std::pop_heap` then
/// `pop_back` (push_heap.h, pop_heap.h). The top is the element no other
/// compares greater than; among equals, which one comes out is decided by
/// these two algorithms and nothing else, so they are libc++'s own.
pub fn PriorityQueue(comptime T: type, comptime Context: type, comptime lessThan: fn (Context, T, T) bool) type {
    return struct {
        const Self = @This();
        const S = Sorter(T, Context, lessThan);

        items: std.ArrayList(T) = .empty,
        ctx: Context,

        /// Return: nothing.
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.items.deinit(allocator);
        }

        /// `empty()`.
        pub fn empty(self: *const Self) bool {
            return self.items.items.len == 0;
        }

        /// `push(v)`.
        ///
        /// Return: nothing, or `error.OutOfMemory`.
        pub fn push(self: *Self, allocator: std.mem.Allocator, v: T) error{OutOfMemory}!void {
            try self.items.append(allocator, v);
            const n = self.items.items.len;
            (S{ .items = self.items.items, .ctx = self.ctx }).siftUp(0, n, @intCast(n));
        }

        /// `top()` then `pop()`, as llama-vocab.cpp's `pop_move` does.
        ///
        /// Return: the element that was on top.
        pub fn pop(self: *Self) T {
            const top = self.items.items[0];
            const n = self.items.items.len;
            (S{ .items = self.items.items, .ctx = self.ctx }).popHeap(0, n, @intCast(n));
            _ = self.items.pop();
            return top;
        }

        /// Empties the queue, keeping its storage.
        pub fn clear(self: *Self) void {
            self.items.clearRetainingCapacity();
        }
    };
}

// -----------------------------------------------------------------------------
// std::hash<std::string>

const k0: u64 = 0xc3a5c85c97cb3127;
const k1: u64 = 0xb492b66fbe98f273;
const k2: u64 = 0x9ae16a3b2f90404f;
const k3: u64 = 0xc949d7c7509e6557;

inline fn load64(s: []const u8, at: usize) u64 {
    return std.mem.readInt(u64, s[at..][0..8], .little);
}

inline fn load32(s: []const u8, at: usize) u32 {
    return std.mem.readInt(u32, s[at..][0..4], .little);
}

inline fn rotate(v: u64, shift: u6) u64 {
    return if (shift == 0) v else std.math.rotr(u64, v, shift);
}

inline fn shiftMix(v: u64) u64 {
    return v ^ (v >> 47);
}

fn hashLen16(u: u64, v: u64) u64 {
    const mul: u64 = 0x9ddfea08eb382d69;
    var a = (u ^ v) *% mul;
    a ^= (a >> 47);
    var b = (v ^ a) *% mul;
    b ^= (b >> 47);
    b *%= mul;
    return b;
}

fn hashLen0To16(s: []const u8) u64 {
    const len: u64 = s.len;
    if (len > 8) {
        const a = load64(s, 0);
        const b = load64(s, s.len - 8);
        return hashLen16(a, std.math.rotr(u64, b +% len, @as(u6, @intCast(len)))) ^ b;
    }
    if (len >= 4) {
        const a = load32(s, 0);
        const b = load32(s, s.len - 4);
        // Without `_LIBCPP_ABI_FIX_CITYHASH_IMPLEMENTATION`, which Apple's
        // dylib does not define -- measured by `libcxx-diff`: `__a << 3`
        // is a 32-bit shift and loses the top three bits.
        return hashLen16(len +% @as(u64, a << 3), b);
    }
    if (len > 0) {
        const a: u32 = s[0];
        const b: u32 = s[s.len >> 1];
        const c: u32 = s[s.len - 1];
        const y: u32 = a +% (b << 8);
        const z: u32 = @as(u32, @intCast(len)) +% (c << 2);
        return shiftMix(@as(u64, y) *% k2 ^ @as(u64, z) *% k3) *% k2;
    }
    return k2;
}

fn hashLen17To32(s: []const u8) u64 {
    const len: u64 = s.len;
    const a = load64(s, 0) *% k1;
    const b = load64(s, 8);
    const c = load64(s, s.len - 8) *% k2;
    const d = load64(s, s.len - 16) *% k0;
    return hashLen16(rotate(a -% b, 43) +% rotate(c, 30) +% d, a +% rotate(b ^ k3, 20) -% c +% len);
}

fn weakHashLen32WithSeeds(w: u64, x: u64, y: u64, z: u64, a_in: u64, b_in: u64) [2]u64 {
    var a = a_in +% w;
    var b = rotate(b_in +% a +% z, 21);
    const c = a;
    a +%= x;
    a +%= y;
    b +%= rotate(a, 44);
    return .{ a +% z, b +% c };
}

fn weakHashLen32WithSeedsAt(s: []const u8, at: usize, a: u64, b: u64) [2]u64 {
    return weakHashLen32WithSeeds(load64(s, at), load64(s, at + 8), load64(s, at + 16), load64(s, at + 24), a, b);
}

fn hashLen33To64(s: []const u8) u64 {
    const len: u64 = s.len;
    var z = load64(s, 24);
    var a = load64(s, 0) +% (len +% load64(s, s.len - 16)) *% k0;
    var b = rotate(a +% z, 52);
    var c = rotate(a, 37);
    a +%= load64(s, 8);
    c +%= rotate(a, 7);
    a +%= load64(s, 16);
    const vf = a +% z;
    const vs = b +% rotate(a, 31) +% c;
    a = load64(s, 16) +% load64(s, s.len - 32);
    z +%= load64(s, s.len - 8);
    b = rotate(a +% z, 52);
    c = rotate(a, 37);
    a +%= load64(s, s.len - 24);
    c +%= rotate(a, 7);
    a +%= load64(s, s.len - 16);
    const wf = a +% z;
    const ws = b +% rotate(a, 31) +% c;
    const r = shiftMix((vf +% ws) *% k2 +% (wf +% vs) *% k0);
    return shiftMix(r *% k0 +% vs) *% k2;
}

/// Ports `std::hash<std::string>`: `__hash_memory`, which is
/// `__murmur2_or_cityhash<size_t, 64>` -- CityHash64 (__functional/hash.h).
///
/// Parameters:
/// - `s`: the bytes.
///
/// Return: the hash.
pub fn hashString(s: []const u8) u64 {
    if (s.len <= 32) {
        if (s.len <= 16) return hashLen0To16(s);
        return hashLen17To32(s);
    } else if (s.len <= 64) {
        return hashLen33To64(s);
    }

    var len: u64 = s.len;
    var x = load64(s, s.len - 40);
    var y = load64(s, s.len - 16) +% load64(s, s.len - 56);
    var z = hashLen16(load64(s, s.len - 48) +% len, load64(s, s.len - 24));
    var v = weakHashLen32WithSeedsAt(s, s.len - 64, len, z);
    var w = weakHashLen32WithSeedsAt(s, s.len - 32, y +% k1, x);
    x = x *% k1 +% load64(s, 0);

    len = (len - 1) & ~@as(u64, 63);
    var at: usize = 0;
    while (true) {
        x = rotate(x +% y +% v[0] +% load64(s, at + 8), 37) *% k1;
        y = rotate(y +% v[1] +% load64(s, at + 48), 42) *% k1;
        x ^= w[1];
        y +%= v[0] +% load64(s, at + 40);
        z = rotate(z +% w[0], 33) *% k1;
        v = weakHashLen32WithSeedsAt(s, at, v[1] *% k1, x +% w[0]);
        w = weakHashLen32WithSeedsAt(s, at + 32, z +% w[1], y +% load64(s, at + 16));
        std.mem.swap(u64, &z, &x);
        at += 64;
        len -= 64;
        if (len == 0) break;
    }
    return hashLen16(hashLen16(v[0], w[0]) +% shiftMix(y) *% k1 +% z, hashLen16(v[1], w[1]) +% x);
}

/// `std::__next_prime`, which lives in the libc++ dylib: the smallest
/// prime not below `n`, and 0 for 0 -- its table of small primes starts
/// at 0. Measured against the dylib by `libcxx-diff`.
pub fn nextPrime(n: usize) usize {
    if (n == 0) return 0;
    if (n <= 2) return 2;
    var c = n | 1;
    while (true) : (c += 2) {
        var prime = true;
        var d: usize = 3;
        while (d * d <= c) : (d += 2) {
            if (c % d == 0) {
                prime = false;
                break;
            }
        }
        if (prime) return c;
    }
}

// -----------------------------------------------------------------------------
// std::unordered_map<std::string, V>

/// Ports `std::unordered_map<std::string, V>` as libc++ lays it out
/// (__hash_table): one singly linked list of every node behind a sentinel,
/// each bucket pointing at the node *before* its group, a prime bucket
/// count grown as `max(2 * bc + !pow2, ceil((size + 1) / max_load_factor))`,
/// a new node inserted at the head of its bucket's group -- or at the head
/// of the whole list when its bucket was empty -- and `__do_rehash`
/// regrouping in list order.
///
/// **Its iteration order is the point.** Lookups would work on any hash map;
/// `llama-vocab.cpp` also iterates `token_to_id`, and the first token whose
/// text matches decides which id becomes EOT and which tokens get their
/// attributes changed. Only this layout reproduces that order.
///
/// Keys are copied in, as the C++ copies them.
pub fn StringMap(comptime V: type) type {
    return struct {
        const Self = @This();
        const none = std.math.maxInt(u32);
        /// The sentinel, `__first_node_`.
        const head: u32 = 0;

        const Node = struct {
            next: u32,
            hash: u64,
            key_off: u32,
            key_len: u32,
            value: V,
        };

        allocator: std.mem.Allocator,
        /// Node 0 is the sentinel; its `next` is the first element.
        nodes: std.ArrayList(Node) = .empty,
        keys: std.ArrayList(u8) = .empty,
        /// `__bucket_list_`: the node before each bucket's group, or none.
        buckets: []u32 = &.{},
        len: usize = 0,
        max_load_factor: f32 = 1.0,

        /// Return: an empty map; `allocator` owns its storage.
        pub fn init(allocator: std.mem.Allocator) error{OutOfMemory}!Self {
            var m: Self = .{ .allocator = allocator };
            try m.nodes.append(allocator, .{ .next = none, .hash = 0, .key_off = 0, .key_len = 0, .value = undefined });
            return m;
        }

        /// Return: nothing.
        pub fn deinit(self: *Self) void {
            self.nodes.deinit(self.allocator);
            self.keys.deinit(self.allocator);
            self.allocator.free(self.buckets);
        }

        fn constrain(h: u64, bc: usize) usize {
            if ((bc & (bc - 1)) == 0) return @intCast(h & (bc - 1));
            return @intCast(if (h < bc) h else h % bc);
        }

        fn isHashPow2(bc: usize) bool {
            return bc > 2 and (bc & (bc - 1)) == 0;
        }

        fn nextHashPow2(n: usize) usize {
            if (n < 2) return n;
            return @as(usize, 1) << @intCast(@bitSizeOf(usize) - @clz(n - 1));
        }

        fn keyOf(self: *const Self, i: u32) []const u8 {
            const n = self.nodes.items[i];
            return self.keys.items[n.key_off..][0..n.key_len];
        }

        /// `bucket_count()`.
        pub fn bucketCount(self: *const Self) usize {
            return self.buckets.len;
        }

        /// `size()`.
        pub fn count(self: *const Self) usize {
            return self.len;
        }

        fn findNode(self: *const Self, key: []const u8, hash: u64) ?u32 {
            const bc = self.buckets.len;
            if (bc == 0) return null;
            const chash = constrain(hash, bc);
            var nd = self.buckets[chash];
            if (nd == none) return null;
            nd = self.nodes.items[nd].next;
            while (nd != none) : (nd = self.nodes.items[nd].next) {
                const h = self.nodes.items[nd].hash;
                if (!(h == hash or constrain(h, bc) == chash)) break;
                if (h == hash and std.mem.eql(u8, self.keyOf(nd), key)) return nd;
            }
            return null;
        }

        /// `find(key)`.
        ///
        /// Return: a pointer to the value, valid until the next insertion,
        /// or null.
        pub fn get(self: *const Self, key: []const u8) ?*V {
            if (self.len == 0) return null;
            const nd = self.findNode(key, hashString(key)) orelse return null;
            return &self.nodes.items[nd].value;
        }

        /// `operator[]`, and `emplace` when the key is new: the value slot
        /// for `key`, inserted with `default` if absent.
        ///
        /// Return: the slot and whether it was there already, valid until
        /// the next insertion.
        pub fn getOrPut(self: *Self, key: []const u8, default: V) error{OutOfMemory}!struct { value: *V, found: bool } {
            const hash = hashString(key);
            if (self.findNode(key, hash)) |nd| return .{ .value = &self.nodes.items[nd].value, .found = true };

            const key_off: u32 = @intCast(self.keys.items.len);
            try self.keys.appendSlice(self.allocator, key);
            try self.nodes.append(self.allocator, .{ .next = none, .hash = hash, .key_off = key_off, .key_len = @intCast(key.len), .value = default });
            const h: u32 = @intCast(self.nodes.items.len - 1);

            var bc = self.buckets.len;
            const size1: f32 = @floatFromInt(self.len + 1);
            if (size1 > @as(f32, @floatFromInt(bc)) * self.max_load_factor) {
                const grow = 2 * bc + @intFromBool(!isHashPow2(bc));
                const need: usize = @intFromFloat(@ceil(size1 / self.max_load_factor));
                try self.rehash(@max(grow, need));
                bc = self.buckets.len;
            }
            const chash = constrain(hash, bc);
            const nodes = self.nodes.items;
            const pn = self.buckets[chash];
            if (pn == none) {
                nodes[h].next = nodes[head].next;
                nodes[head].next = h;
                self.buckets[chash] = head;
                if (nodes[h].next != none) self.buckets[constrain(nodes[nodes[h].next].hash, bc)] = h;
            } else {
                nodes[h].next = nodes[pn].next;
                nodes[pn].next = h;
            }
            self.len += 1;
            return .{ .value = &nodes[h].value, .found = false };
        }

        /// Ports `__hash_table::__rehash<true>`.
        fn rehash(self: *Self, n_in: usize) error{OutOfMemory}!void {
            var n = n_in;
            if (n == 1) {
                n = 2;
            } else if ((n & (n - 1)) != 0) {
                n = nextPrime(n);
            }
            const bc = self.buckets.len;
            if (n > bc) {
                try self.doRehash(n);
            } else if (n < bc) {
                const need: usize = @intFromFloat(@ceil(@as(f32, @floatFromInt(self.len)) / self.max_load_factor));
                n = @max(n, if (isHashPow2(bc)) nextHashPow2(need) else nextPrime(need));
                if (n < bc) try self.doRehash(n);
            }
        }

        /// Ports `__hash_table::__do_rehash<true>`.
        fn doRehash(self: *Self, bucket_count: usize) error{OutOfMemory}!void {
            self.allocator.free(self.buckets);
            self.buckets = &.{};
            if (bucket_count == 0) return;
            self.buckets = try self.allocator.alloc(u32, bucket_count);
            @memset(self.buckets, none);

            const nodes = self.nodes.items;
            var pp: u32 = head;
            var cp = nodes[pp].next;
            if (cp == none) return;

            var chash = constrain(nodes[cp].hash, bucket_count);
            self.buckets[chash] = pp;
            var phash = chash;
            pp = cp;
            cp = nodes[cp].next;
            while (cp != none) : (cp = nodes[pp].next) {
                chash = constrain(nodes[cp].hash, bucket_count);
                if (chash == phash) {
                    pp = cp;
                } else if (self.buckets[chash] == none) {
                    self.buckets[chash] = pp;
                    pp = cp;
                    phash = chash;
                } else {
                    const np = cp;
                    nodes[pp].next = nodes[np].next;
                    nodes[np].next = nodes[self.buckets[chash]].next;
                    nodes[self.buckets[chash]].next = cp;
                }
            }
        }

        /// Iterates in libc++'s order: the list from the sentinel.
        pub const Iterator = struct {
            map: *const Self,
            at: u32,

            /// Return: the next entry, or null at the end.
            pub fn next(self: *Iterator) ?struct { key: []const u8, value: *V } {
                if (self.at == none) return null;
                const i = self.at;
                self.at = self.map.nodes.items[i].next;
                return .{ .key = self.map.keyOf(i), .value = &self.map.nodes.items[i].value };
            }
        };

        /// Return: an iterator over every entry, valid until the next
        /// insertion.
        pub fn iterator(self: *const Self) Iterator {
            return .{ .map = self, .at = self.nodes.items[head].next };
        }
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

fn lessU32(_: void, a: u32, b: u32) bool {
    return a < b;
}

test "sort orders every length through each introsort branch" {
    // Lengths below 6 take the sorting networks, below 24 insertion sort,
    // above 128 the ninther; 2000 forces several partitions.
    var prng = std.Random.DefaultPrng.init(1);
    const r = prng.random();
    var buf: [2000]u32 = undefined;
    for ([_]usize{ 0, 1, 2, 3, 4, 5, 6, 23, 24, 25, 127, 128, 129, 300, 2000 }) |n| {
        for (buf[0..n]) |*v| v.* = r.intRangeLessThan(u32, 0, 50);
        sort(u32, buf[0..n], {}, lessU32);
        try std.testing.expect(std.sort.isSorted(u32, buf[0..n], {}, lessU32));
    }
}

test "partial sort leaves the head sorted and smallest" {
    var prng = std.Random.DefaultPrng.init(2);
    const r = prng.random();
    var buf: [500]u32 = undefined;
    for (&buf) |*v| v.* = r.intRangeLessThan(u32, 0, 1000);
    var copy = buf;
    partialSort(u32, &buf, 40, {}, lessU32);
    std.mem.sort(u32, &copy, {}, lessU32);
    try std.testing.expectEqualSlices(u32, copy[0..40], buf[0..40]);
}

test "mt19937 matches the standard's 10000th output" {
    // [rand.predef]: the 10000th consecutive invocation of a default-
    // constructed std::mt19937 produces 4123659995.
    var g = Mt19937.init(Mt19937.default_seed);
    var v: u32 = 0;
    for (0..10000) |_| v = g.next();
    try std.testing.expectEqual(@as(u32, 4123659995), v);
}

test "upper bound finds the first greater element" {
    const xs = [_]f64{ 0.1, 0.3, 0.3, 0.9 };
    try std.testing.expectEqual(@as(usize, 0), upperBound(&xs, 0.0));
    try std.testing.expectEqual(@as(usize, 3), upperBound(&xs, 0.3));
    try std.testing.expectEqual(@as(usize, 4), upperBound(&xs, 0.95));
    try std.testing.expectEqual(@as(usize, 0), upperBound(&.{}, 0.5));
}

test "std::max and std::min keep the first operand when unordered" {
    try std.testing.expect(std.math.isNan(max(f32, std.math.nan(f32), 1.0)));
    try std.testing.expectEqual(@as(f32, 1.0), max(f32, 1.0, std.math.nan(f32)));
    try std.testing.expectEqual(@as(f32, 2.0), min(f32, 2.0, std.math.nan(f32)));
}
