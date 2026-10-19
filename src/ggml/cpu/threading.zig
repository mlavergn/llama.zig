//! The threadpool: starting workers, handing them a graph, and getting them
//! back in step between nodes.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (v0.3.0, `c1d0e7a00`),
//! the sections at lines 418 (threading defs), 540 (NUMA), 2711 (threadpool
//! lifecycle) and 3065 (graph compute). Each declaration names the C it
//! replaces and the line it began at.
//!
//! # The threading defs
//!
//! The C carries a page of `#define ggml_mutex_lock(m)` shims so one body can
//! drive pthreads on Unix and SRW locks on Windows, plus its own atomic
//! wrappers for MSVC. The shims are ported as the inline functions `lock`,
//! `lockShared`, `condWait` and the rest, both arms each; `std.atomic.Value`
//! replaces the atomic wrappers. The `_shared` variants are the same calls as
//! the plain ones on pthreads, and a reader's hold on Windows -- which is why
//! they stay distinct here.
//!
//! What does survive exactly is the *protocol* -- which counter is bumped with
//! which ordering, and in what sequence. `ggml_barrier` in particular is
//! load-bearing and subtle: the memory orderings are the C's, down to the
//! relaxed loads that bracket the two sequentially-consistent fences.
//!
//! # OpenMP is not compiled in
//!
//! Roughly a third of the C's threading code sits behind `GGML_USE_OPENMP`,
//! which `build/llamacpp.zig` does not define. Only the native-thread arms are
//! ported. That is also why `ggml_compute_state` keeps `thrd`, `last_graph`
//! and `pending`: under OpenMP the C drops those three fields.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("../impl.zig");
const win32 = @import("../win32.zig");
const types = @import("../types.zig");
const graph = @import("../graph.zig");
const context = @import("../context.zig");
const defs = @import("defs.zig");
const plan = @import("plan.zig");
const forward = @import("forward.zig");
const features = @import("features.zig");
const c = impl.c;

const Tensor = defs.Tensor;
const CGraph = defs.CGraph;
const Threadpool = defs.Threadpool;
const ComputeState = defs.ComputeState;
const ComputeParams = defs.ComputeParams;

/// `ggml.c`'s aligned allocator, ported in `src/ggml/runtime.zig`.
extern fn ggml_aligned_malloc(size: usize) ?*anyopaque;
extern fn ggml_aligned_free(ptr: ?*anyopaque, size: usize) void;

/// `ggml.c`'s default threadpool parameters, ported in `src/ggml/runtime.zig`.
extern fn ggml_threadpool_params_default(n_threads: c_int) c.struct_ggml_threadpool_params;

/// The fused RMS-norm kernel, still C++ in `ggml-cpu/ops.cpp`.
extern fn ggml_compute_forward_rms_norm_mul_fused(
    params: *const ComputeParams,
    dst_rms_norm: *Tensor,
    dst_mul: *Tensor,
) void;

// -----------------------------------------------------------------------------
// The barrier
//
// Every node boundary passes through here, so this is the hottest
// synchronisation in the backend.

/// Ports `ggml_barrier` (ggml-cpu.c:575 @c1d0e7a00).
///
/// A sense-reversing barrier built from two counters. Threads bump
/// `n_barrier`; the last one in resets it and bumps `n_barrier_passed`, which
/// the others are spinning on.
///
/// The orderings are the C's and are not incidental. The two `fetchAdd`s are
/// sequentially consistent because they *are* the fences that publish the
/// preceding node's writes; the loads around them are relaxed because the
/// value they read is only a ticket, and the fence at the bottom is what makes
/// the next node's reads safe.
///
/// Parameters:
/// - `tp`: the threadpool whose threads are synchronising.
pub export fn ggml_barrier(tp: *Threadpool) void {
    const n_threads = tp.n_graph.load(.monotonic) & defs.n_threads_mask;
    if (n_threads == 1) return;

    const n_passed = tp.n_barrier_passed.load(.monotonic);

    // Enter the barrier (full seq-cst fence).
    const n_barrier = tp.n_barrier.fetchAdd(1, .seq_cst);

    if (n_barrier == n_threads - 1) {
        // Last thread in: reset the entry counter and release the rest.
        tp.n_barrier.store(0, .monotonic);
        _ = tp.n_barrier_passed.fetchAdd(1, .seq_cst);
        return;
    }

    while (tp.n_barrier_passed.load(.monotonic) == n_passed) {
        defs.cpuRelax();
    }

    // Exit the barrier. The C wants a standalone `atomic_thread_fence` here
    // and falls back to a no-op read-modify-write under thread-sanitizer,
    // which does not support one. Zig has no `@fence` at all as of 0.16, so
    // the port takes the C's own fallback: adding zero seq-cst carries the
    // same ordering.
    _ = tp.n_barrier_passed.fetchAdd(0, .seq_cst);
}

/// Ports `ggml_threadpool_chunk_set` (ggml-cpu.c:613 @c1d0e7a00).
///
/// Parameters:
/// - `tp`: the threadpool.
/// - `value`: the next chunk index to hand out.
pub export fn ggml_threadpool_chunk_set(tp: *Threadpool, value: c_int) void {
    tp.current_chunk.store(value, .monotonic);
}

/// Ports `ggml_threadpool_chunk_add` (ggml-cpu.c:617 @c1d0e7a00).
///
/// Parameters:
/// - `tp`: the threadpool.
/// - `value`: how far to advance the chunk cursor.
///
/// Return: the cursor's value before the add, which is the chunk the caller
/// has just claimed.
pub export fn ggml_threadpool_chunk_add(tp: *Threadpool, value: c_int) c_int {
    return tp.current_chunk.fetchAdd(value, .monotonic);
}

// -----------------------------------------------------------------------------
// NUMA
//
// Linux-with-glibc only in the C (`__gnu_linux__`): every other platform takes
// the `#else` that does nothing. `ggml_is_numa` is read on the `mul_mat`
// chunking path, so all of it is ported, both arms.

/// `__linux__`, which selects the C's Linux arms of thread affinity and
/// priority.
const is_linux = builtin.os.tag == .linux;

/// `__gnu_linux__`, which selects the C's NUMA arm: Linux with glibc. Bionic
/// cannot set affinity, the C notes, so Android takes the `#else`.
const is_gnu_linux = is_linux and builtin.abi.isGnu();

/// `GGML_NUMA_MAX_NODES` and `GGML_NUMA_MAX_CPUS` (ggml-cpu.c:544, 545 @c1d0e7a00).
const numa_max_nodes = 8;
const numa_max_cpus = 512;

/// glibc's `cpu_set_t`: 1024 bits in `unsigned long` words, and the
/// `CPU_SET_S` that writes it. Sized with `CPU_ALLOC_SIZE` where the C uses
/// `CPU_ALLOC`; `numa_max_cpus` bounds every such set, so a fixed array of
/// the full size stands in for the heap allocation.
const CpuSet = extern struct {
    bits: [1024 / @bitSizeOf(c_ulong)]c_ulong = @splat(0),

    fn set(self: *CpuSet, cpu: usize, setsize: usize) void {
        const word = cpu / @bitSizeOf(c_ulong);
        if (cpu / 8 < setsize) self.bits[word] |= @as(c_ulong, 1) << @intCast(cpu % @bitSizeOf(c_ulong));
    }

    /// `CPU_ALLOC_SIZE(count)`: whole words, in bytes.
    fn allocSize(count: usize) usize {
        return (count + @bitSizeOf(c_ulong) - 1) / @bitSizeOf(c_ulong) * @sizeOf(c_ulong);
    }
};

/// The libc calls only the Linux arms make.
const linux_c = if (is_linux) struct {
    extern "c" fn pthread_setaffinity_np(thread: std.c.pthread_t, size: usize, set: *const CpuSet) c_int;
    extern "c" fn pthread_getaffinity_np(thread: std.c.pthread_t, size: usize, set: *CpuSet) c_int;
    extern "c" fn getcpu(cpu: *c_uint, node: *c_uint) c_int;
    extern "c" fn statx(dirfd: c_int, path: [*:0]const u8, flags: c_int, mask: std.os.linux.STATX, buf: *std.os.linux.Statx) c_int;
    extern "c" fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;
    extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
    extern "c" fn fgets(buf: [*]u8, size: c_int, f: *anyopaque) ?[*]u8;
    extern "c" fn fclose(f: *anyopaque) c_int;
    extern "c" fn strncmp(a: [*]const u8, b: [*:0]const u8, n: usize) c_int;
} else struct {};

extern fn strerror(err: c_int) [*:0]const u8;

/// Ports `struct ggml_numa_node` (ggml-cpu.c:547 @c1d0e7a00).
const NumaNode = struct {
    cpus: [numa_max_cpus]u32 = @splat(0),
    n_cpus: u32 = 0,
};

/// Ports `struct ggml_numa_nodes` and the `g_state` that holds it
/// (ggml-cpu.c:552, 573 @c1d0e7a00). Off glibc Linux nothing writes it, and
/// `n_nodes` stays 0.
var numa: struct {
    numa_strategy: c.enum_ggml_numa_strategy = 0,
    nodes: [numa_max_nodes]NumaNode = @splat(.{}),
    n_nodes: u32 = 0,
    total_cpus: u32 = 0,
    current_node: u32 = 0,
    cpuset: CpuSet = .{},
} = .{};

/// The C's `__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33)`: the
/// glibc this build links has a `getcpu` wrapper.
const glibc_has_getcpu = is_gnu_linux and
    builtin.os.version_range.linux.glibc.order(.{ .major = 2, .minor = 34, .patch = 0 }) != .lt;

/// `stat(path, &st) == 0`, asked through `statx` because Zig 0.17 declares
/// no `struct stat` for Linux -- the move `llama/mmap.zig` makes too.
fn pathExists(path: [*:0]const u8) bool {
    var st: std.os.linux.Statx = undefined;
    return linux_c.statx(std.os.linux.AT.FDCWD, path, 0, std.os.linux.STATX.BASIC_STATS, &st) == 0;
}

/// Ports `ggml_get_numa_affinity` (ggml-cpu.c:622 @c1d0e7a00), the
/// `__gnu_linux__` arm: the calling thread's affinity, which under
/// `numactl` is the set it was started with.
fn getNumaAffinity() CpuSet {
    var cpuset: CpuSet = .{};
    _ = linux_c.pthread_getaffinity_np(pthread_self(), @sizeOf(CpuSet), &cpuset);
    return cpuset;
}

/// Ports `ggml_numa_init` (ggml-cpu.c:636 @c1d0e7a00), both arms.
///
/// On glibc Linux: enumerate `/sys/devices/system/node/node*` and
/// `/sys/devices/system/cpu/cpu*`, find the calling CPU's node with
/// `getcpu`, list each node's CPUs, and warn if the kernel's automatic NUMA
/// balancing is on. Anything that fails leaves `n_nodes` at 0, so NUMA stays
/// off. Elsewhere the C's `#else` is `UNUSED(numa_flag)` and a TODO.
///
/// Parameters:
/// - `numa_flag`: the strategy `set_numa_thread_affinity` will apply.
pub export fn ggml_numa_init(numa_flag: c.enum_ggml_numa_strategy) void {
    if (numa.n_nodes > 0) {
        impl.fprintfStderr("ggml_numa_init: NUMA already initialized\n", .{});
        return;
    }
    if (comptime !is_gnu_linux) return;

    var path: [256]u8 = undefined;

    numa.numa_strategy = numa_flag;
    numa.cpuset = getNumaAffinity();

    // enumerate nodes
    while (numa.n_nodes < numa_max_nodes) {
        const rv = linux_c.snprintf(&path, path.len, "/sys/devices/system/node/node%u", numa.n_nodes);
        impl.assert(rv > 0 and @as(c_uint, @intCast(rv)) < path.len, "rv > 0 && (unsigned)rv < sizeof(path)");
        if (!pathExists(@ptrCast(&path))) break;
        numa.n_nodes += 1;
    }

    // enumerate CPUs
    while (numa.total_cpus < numa_max_cpus) {
        const rv = linux_c.snprintf(&path, path.len, "/sys/devices/system/cpu/cpu%u", numa.total_cpus);
        impl.assert(rv > 0 and @as(c_uint, @intCast(rv)) < path.len, "rv > 0 && (unsigned)rv < sizeof(path)");
        if (!pathExists(@ptrCast(&path))) break;
        numa.total_cpus += 1;
    }

    // figure out which node we're on. The C calls glibc's `getcpu` from 2.34
    // and the raw syscall before it; both fill the same two words.
    var current_cpu: c_uint = 0;
    const getcpu_ret: c_int = if (comptime glibc_has_getcpu)
        linux_c.getcpu(&current_cpu, &numa.current_node)
    else getcpu: {
        var cpu: usize = 0;
        var node: usize = 0;
        const r = std.os.linux.getcpu(&cpu, &node);
        current_cpu = @truncate(cpu);
        numa.current_node = @truncate(node);
        break :getcpu if (std.os.linux.errno(r) == .SUCCESS) 0 else -1;
    };

    if (numa.n_nodes < 1 or numa.total_cpus < 1 or getcpu_ret != 0) {
        numa.n_nodes = 0;
        return;
    }

    for (numa.nodes[0..numa.n_nodes], 0..) |*node, n| {
        node.n_cpus = 0;
        var cpu: c_uint = 0;
        while (cpu < numa.total_cpus) : (cpu += 1) {
            const rv = linux_c.snprintf(&path, path.len, "/sys/devices/system/node/node%u/cpu%u", @as(c_uint, @intCast(n)), cpu);
            impl.assert(rv > 0 and @as(c_uint, @intCast(rv)) < path.len, "rv > 0 && (unsigned)rv < sizeof(path)");
            if (pathExists(@ptrCast(&path))) {
                node.cpus[node.n_cpus] = cpu;
                node.n_cpus += 1;
            }
        }
    }

    if (ggml_is_numa()) {
        if (linux_c.fopen("/proc/sys/kernel/numa_balancing", "r")) |fptr| {
            var buf: [42]u8 = undefined;
            if (linux_c.fgets(&buf, buf.len, fptr) != null and linux_c.strncmp(&buf, "0\n", buf.len) != 0) {
                impl.logWarn("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n", .{});
            }
            _ = linux_c.fclose(fptr);
        }
    }
}

/// Ports `ggml_is_numa` (ggml-cpu.c:724 @c1d0e7a00).
///
/// Return: true when more than one NUMA node was found. Always false off
/// glibc Linux, which is what keeps `mul_mat`'s chunk-by-thread fallback off.
pub export fn ggml_is_numa() bool {
    return numa.n_nodes > 1;
}

/// Ports `set_numa_thread_affinity`, `set_numa_thread_affinity`
/// (ggml-cpu.c:2148, 2216 @c1d0e7a00): the `__gnu_linux__` arm and the
/// `#else` no-op.
///
/// Pins worker `thread_n` to the CPUs of one node -- round-robin over the
/// nodes for DISTRIBUTE, the calling thread's node for ISOLATE -- or, for
/// NUMACTL, to the set the process was started with.
fn setNumaThreadAffinity(thread_n: c_int) void {
    if (comptime !is_gnu_linux) return;
    if (!ggml_is_numa()) return;

    const setsize = CpuSet.allocSize(numa.total_cpus);
    const node_num: u32 = switch (numa.numa_strategy) {
        // run thread on node_num thread_n / (threads per node)
        c.GGML_NUMA_STRATEGY_DISTRIBUTE => @as(u32, @intCast(thread_n)) % numa.n_nodes,
        // run thread on current_node
        c.GGML_NUMA_STRATEGY_ISOLATE => numa.current_node,
        // use the cpuset that numactl gave us
        c.GGML_NUMA_STRATEGY_NUMACTL => {
            const rv = linux_c.pthread_setaffinity_np(pthread_self(), setsize, &numa.cpuset);
            if (rv != 0) impl.fprintfStderr("warning: pthread_setaffinity_np() failed: %s\n", .{strerror(rv)});
            return;
        },
        else => return,
    };

    const node = &numa.nodes[node_num];
    var cpus: CpuSet = .{};
    for (node.cpus[0..node.n_cpus]) |cpu| cpus.set(cpu, setsize);

    const rv = linux_c.pthread_setaffinity_np(pthread_self(), setsize, &cpus);
    if (rv != 0) impl.fprintfStderr("warning: pthread_setaffinity_np() failed: %s\n", .{strerror(rv)});
}

/// Ports `clear_numa_thread_affinity`, `clear_numa_thread_affinity`
/// (ggml-cpu.c:2193, 2217 @c1d0e7a00): the `__gnu_linux__` arm, every CPU
/// allowed again, and the `#else` no-op.
fn clearNumaThreadAffinity() void {
    if (comptime !is_gnu_linux) return;
    if (!ggml_is_numa()) return;

    const setsize = CpuSet.allocSize(numa.total_cpus);
    var cpus: CpuSet = .{};
    var i: usize = 0;
    while (i < numa.total_cpus) : (i += 1) cpus.set(i, setsize);

    const rv = linux_c.pthread_setaffinity_np(pthread_self(), setsize, &cpus);
    if (rv != 0) impl.fprintfStderr("warning: pthread_setaffinity_np() failed: %s\n", .{strerror(rv)});
}

// -----------------------------------------------------------------------------
// Thread placement and priority

/// `SCHED_OTHER`, `SCHED_FIFO` and `SCHED_BATCH`, which the two platforms
/// number differently. macOS has no `SCHED_BATCH`; only the Linux arm uses it.
const sched_other: c_int = if (is_linux) 0 else 1;
const sched_fifo: c_int = if (is_linux) 1 else 4;
const sched_batch: c_int = 3;

/// `struct sched_param`: one `int` on glibc, one `int` and four opaque bytes
/// on macOS.
const SchedParam = if (is_linux) extern struct {
    sched_priority: c_int,
} else extern struct {
    sched_priority: c_int,
    __opaque: [4]u8 = @splat(0),
};

extern fn pthread_self() std.c.pthread_t;
extern fn pthread_setschedparam(thread: std.c.pthread_t, policy: c_int, param: *const SchedParam) c_int;

/// What `ggml_mutex_init` and `ggml_cond_init` expand to in the pthread arm
/// (ggml-cpu.c:452, 469 @c1d0e7a00): these two calls with a null attribute.
extern fn pthread_mutex_init(mutex: *std.c.pthread_mutex_t, attr: ?*const anyopaque) c_int;
extern fn pthread_cond_init(cond: *std.c.pthread_cond_t, attr: ?*const anyopaque) c_int;

/// `_WIN32`, which selects the C's SRW-lock threading defs and the Windows
/// arms of thread affinity and priority.
const is_windows = builtin.os.tag == .windows;

/// Ports `ggml_thread_apply_affinity`, `ggml_thread_apply_affinity`,
/// `ggml_thread_apply_affinity` (ggml-cpu.c:2497, 2579, 2614 @c1d0e7a00): the
/// Windows, Apple and Linux arms.
///
/// Apple platforms expose no way to pin a thread to a core, so the C's Apple
/// arm is a no-op that reports success. The Linux arm pins the calling
/// thread to every CPU set in `mask`; its warning prints the **address** of
/// the mask, cast to `unsigned long long`, which is what the C passes. The
/// Windows arm packs the first 64 entries into one affinity mask and warns,
/// without failing, if any CPU past them is set.
fn threadApplyAffinity(mask: []const bool) bool {
    if (comptime is_windows) return threadApplyAffinityWindows(mask);
    if (comptime !is_linux) return true;

    var cpuset: CpuSet = .{};
    for (mask[0..c.GGML_MAX_N_THREADS], 0..) |bit, i| {
        if (bit) cpuset.set(i, @sizeOf(CpuSet));
    }

    const err = linux_c.pthread_setaffinity_np(pthread_self(), @sizeOf(CpuSet), &cpuset);
    if (err != 0) {
        impl.fprintfStderr("warn: failed to set affinity mask 0x%llx : %s (%d)\n", .{ @as(c_ulonglong, @intFromPtr(mask.ptr)), strerror(err), err });
        return false;
    }
    return true;
}

/// The Windows arm of `ggml_thread_apply_affinity` (ggml-cpu.c:2497 @c1d0e7a00).
fn threadApplyAffinityWindows(mask: []const bool) bool {
    const h = win32.GetCurrentThread();
    var bitmask: u64 = 0;

    comptime std.debug.assert(c.GGML_MAX_N_THREADS >= 64);

    var i: u6 = 0;
    while (true) : (i += 1) {
        if (mask[i]) bitmask |= @as(u64, 1) << i;
        if (i == 63) break;
    }

    for (mask[64..c.GGML_MAX_N_THREADS]) |bit| {
        if (bit) {
            impl.fprintfStderr("warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n", .{});
            break;
        }
    }

    const m = win32.SetThreadAffinityMask(h, bitmask);
    return m != 0;
}

/// The Windows arm of `ggml_thread_apply_priority` (ggml-cpu.c:2531 @c1d0e7a00).
///
/// Every level but LOW first opts the thread out of power throttling --
/// Windows 11 otherwise parks cores and crowds the workers onto the first few
/// -- and NORMAL then keeps the inherited priority.
fn threadApplyPriorityWindows(prio: i32) bool {
    const p: c_int = switch (prio) {
        c.GGML_SCHED_PRIO_LOW => win32.THREAD_PRIORITY_BELOW_NORMAL,
        c.GGML_SCHED_PRIO_NORMAL => win32.THREAD_PRIORITY_NORMAL,
        c.GGML_SCHED_PRIO_MEDIUM => win32.THREAD_PRIORITY_ABOVE_NORMAL,
        c.GGML_SCHED_PRIO_HIGH => win32.THREAD_PRIORITY_HIGHEST,
        c.GGML_SCHED_PRIO_REALTIME => win32.THREAD_PRIORITY_TIME_CRITICAL,
        else => win32.THREAD_PRIORITY_NORMAL,
    };

    if (prio != c.GGML_SCHED_PRIO_LOW) {
        // `_WIN32_WINNT` is MinGW's default, 0xa00, so the C's `>= 0x0602`
        // guard is open.
        var t: win32.THREAD_POWER_THROTTLING_STATE = .{
            .Version = win32.THREAD_POWER_THROTTLING_CURRENT_VERSION,
            .ControlMask = win32.THREAD_POWER_THROTTLING_EXECUTION_SPEED,
            .StateMask = 0,
        };
        if (win32.SetThreadInformation(win32.GetCurrentThread(), win32.ThreadPowerThrottling, &t, @sizeOf(win32.THREAD_POWER_THROTTLING_STATE)) == 0) {
            impl.logDebug("failed to disable thread power throttling %d : (%d)\n", .{ prio, @as(c_int, @bitCast(win32.GetLastError())) });
            return false;
        }
    }

    if (prio == c.GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (win32.SetThreadPriority(win32.GetCurrentThread(), p) == 0) {
        impl.fprintfStderr("warn: failed to set thread priority %d : (%d)\n", .{ prio, @as(c_int, @bitCast(win32.GetLastError())) });
        return false;
    }

    return true;
}

/// Ports `ggml_thread_apply_priority`, `ggml_thread_apply_priority`
/// (ggml-cpu.c:2585, 2643 @c1d0e7a00): the Apple and the Linux arms; Windows
/// is `threadApplyPriorityWindows`. They differ only in `LOW`, which
/// is `SCHED_BATCH` on Linux and `SCHED_OTHER` on Apple, where the C notes
/// there is no way to ask for less than normal -- and in the numbering of
/// the policies.
///
/// Parameters:
/// - `prio`: one of `GGML_SCHED_PRIO_*`.
///
/// Return: true on success, or when `prio` is NORMAL and nothing is done.
fn threadApplyPriority(prio: i32) bool {
    if (comptime is_windows) return threadApplyPriorityWindows(prio);

    var p: SchedParam = .{ .sched_priority = 0 };
    var policy: c_int = sched_other;

    switch (prio) {
        c.GGML_SCHED_PRIO_LOW => {
            policy = if (is_linux) sched_batch else sched_other;
            p.sched_priority = 0;
        },
        c.GGML_SCHED_PRIO_NORMAL => {
            policy = sched_other;
            p.sched_priority = 0;
        },
        c.GGML_SCHED_PRIO_MEDIUM => {
            policy = sched_fifo;
            p.sched_priority = 40;
        },
        c.GGML_SCHED_PRIO_HIGH => {
            policy = sched_fifo;
            p.sched_priority = 80;
        },
        c.GGML_SCHED_PRIO_REALTIME => {
            policy = sched_fifo;
            p.sched_priority = 90;
        },
        else => {},
    }

    // Keep the inherited policy and priority.
    if (prio == c.GGML_SCHED_PRIO_NORMAL) return true;

    const err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        impl.fprintfStderr("warn: failed to set thread priority %d : %s (%d)\n", .{ prio, strerror(err), err });
        return false;
    }

    return true;
}

/// Ports `ggml_thread_cpumask_is_valid` (ggml-cpu.c:2682 @c1d0e7a00).
fn cpumaskIsValid(mask: []const bool) bool {
    for (mask) |bit| {
        if (bit) return true;
    }
    return false;
}

/// Ports `ggml_thread_cpumask_next` (ggml-cpu.c:2689 @c1d0e7a00).
///
/// Hands each worker its own core when `strict` is set, walking the global
/// mask from where the last worker stopped. Otherwise every worker gets the
/// whole mask and the scheduler decides.
///
/// Parameters:
/// - `global_mask`: the caller's requested placement.
/// - `local_mask`: filled in with this worker's placement.
/// - `strict`: one core per worker rather than the whole mask.
/// - `iter`: cursor into `global_mask`, advanced past the core taken.
fn cpumaskNext(
    global_mask: *const [c.GGML_MAX_N_THREADS]bool,
    local_mask: *[c.GGML_MAX_N_THREADS]bool,
    strict: bool,
    iter: *i32,
) void {
    if (!strict) {
        local_mask.* = global_mask.*;
        return;
    }

    local_mask.* = @splat(false);
    const base_idx = iter.*;
    for (0..c.GGML_MAX_N_THREADS) |i| {
        var idx = base_idx + @as(i32, @intCast(i));
        // A cheaper modulo, per the C.
        if (idx >= c.GGML_MAX_N_THREADS) idx -= c.GGML_MAX_N_THREADS;

        if (global_mask[@intCast(idx)]) {
            local_mask[@intCast(idx)] = true;
            iter.* = idx + 1;
            return;
        }
    }
}

// -----------------------------------------------------------------------------
// Threadpool lifecycle

/// Ports `ggml_mutex_lock`, `ggml_mutex_lock` (ggml-cpu.c:434, 454 @c1d0e7a00).
inline fn lock(threadpool: *Threadpool) void {
    if (comptime is_windows) return win32.AcquireSRWLockExclusive(&threadpool.mutex);
    _ = std.c.pthread_mutex_lock(&threadpool.mutex);
}

/// Ports `ggml_mutex_unlock`, `ggml_mutex_unlock` (ggml-cpu.c:435, 455 @c1d0e7a00).
inline fn unlock(threadpool: *Threadpool) void {
    if (comptime is_windows) return win32.ReleaseSRWLockExclusive(&threadpool.mutex);
    _ = std.c.pthread_mutex_unlock(&threadpool.mutex);
}

/// Ports `ggml_mutex_lock_shared`, `ggml_mutex_lock_shared` (ggml-cpu.c:436,
/// 456 @c1d0e7a00). The same call as `lock` on pthreads; a reader's hold on
/// the SRW lock on Windows, which is what the C's two waits take.
inline fn lockShared(threadpool: *Threadpool) void {
    if (comptime is_windows) return win32.AcquireSRWLockShared(&threadpool.mutex);
    _ = std.c.pthread_mutex_lock(&threadpool.mutex);
}

/// Ports `ggml_mutex_unlock_shared`, `ggml_mutex_unlock_shared`
/// (ggml-cpu.c:437, 457 @c1d0e7a00).
inline fn unlockShared(threadpool: *Threadpool) void {
    if (comptime is_windows) return win32.ReleaseSRWLockShared(&threadpool.mutex);
    _ = std.c.pthread_mutex_unlock(&threadpool.mutex);
}

/// Ports `ggml_cond_wait`, `ggml_cond_wait` (ggml-cpu.c:441, 471 @c1d0e7a00).
/// On Windows the wait releases and retakes the lock in shared mode, matching
/// `lockShared` around it.
inline fn condWait(threadpool: *Threadpool) void {
    if (comptime is_windows) {
        _ = win32.SleepConditionVariableSRW(&threadpool.cond, &threadpool.mutex, win32.INFINITE, win32.CONDITION_VARIABLE_LOCKMODE_SHARED);
        return;
    }
    _ = std.c.pthread_cond_wait(&threadpool.cond, &threadpool.mutex);
}

/// Ports `ggml_cond_broadcast`, `ggml_cond_broadcast` (ggml-cpu.c:442, 472 @c1d0e7a00).
inline fn condBroadcast(threadpool: *Threadpool) void {
    if (comptime is_windows) return win32.WakeAllConditionVariable(&threadpool.cond);
    _ = std.c.pthread_cond_broadcast(&threadpool.cond);
}

/// Ports `ggml_threadpool_new_impl` (ggml-cpu.c:3278 @c1d0e7a00).
///
/// Parameters:
/// - `tpp`: thread count, placement, priority, polling level, paused state.
/// - `cgraph`, `cplan`: the first graph to run, or null for an idle pool.
///
/// Return: the pool, owned by the caller; free it with `ggml_threadpool_free`.
fn threadpoolNewImpl(
    tpp: *c.struct_ggml_threadpool_params,
    cgraph: ?*const CGraph,
    cplan: ?*const c.ggml_cplan,
) *Threadpool {
    const threadpool: *Threadpool = @ptrCast(@alignCast(ggml_aligned_malloc(@sizeOf(Threadpool)).?));

    threadpool.* = .{
        .mutex = undefined,
        .cond = undefined,
        .cgraph = cgraph,
        .cplan = cplan,
        .n_graph = .init(0),
        .n_barrier = .init(0),
        .n_barrier_passed = .init(0),
        .current_chunk = .init(0),
        .stop = .init(false),
        .pause = .init(tpp.paused),
        .abort = .init(-1),
        .workers = null,
        .n_threads = tpp.n_threads,
        .poll = tpp.poll,
        .prio = tpp.prio,
        .ec = c.GGML_STATUS_SUCCESS,
    };

    const n: usize = @intCast(tpp.n_threads);
    const workers_size = @sizeOf(ComputeState) * n;
    const workers: [*]ComputeState = @ptrCast(@alignCast(ggml_aligned_malloc(workers_size).?));

    for (0..n) |j| {
        workers[j] = .{
            .thrd = undefined,
            .last_graph = 0,
            .pending = false,
            .cpumask = @splat(false),
            .threadpool = threadpool,
            .ith = @intCast(j),
        };
    }

    threadpool.workers = workers;

    // `ggml_mutex_init`, `ggml_cond_init`, `ggml_mutex_init`, `ggml_cond_init`
    // (ggml-cpu.c:432, 439, 452, 469 @c1d0e7a00): both arms.
    if (comptime is_windows) {
        win32.InitializeSRWLock(&threadpool.mutex);
        win32.InitializeConditionVariable(&threadpool.cond);
    } else {
        _ = pthread_mutex_init(&threadpool.mutex, null);
        _ = pthread_cond_init(&threadpool.cond, null);
    }

    // Spin up the workers and place them. The main thread is placed last, so
    // it lands on the higher-numbered cores.
    var cpumask_iter: i32 = 0;

    for (1..n) |j| {
        cpumaskNext(&tpp.cpumask, &workers[j].cpumask, tpp.strict_cpu, &cpumask_iter);
        workers[j].thrd = std.Thread.spawn(.{}, secondaryThread, .{&workers[j]}) catch
            impl.abort("rc == 0");
    }

    cpumaskNext(&tpp.cpumask, &workers[0].cpumask, tpp.strict_cpu, &cpumask_iter);

    if (!threadpool.pause.load(.monotonic)) {
        // Place the main thread now; otherwise `resume` will do it.
        _ = threadApplyPriority(threadpool.prio);
        if (cpumaskIsValid(&workers[0].cpumask)) {
            _ = threadApplyAffinity(&workers[0].cpumask);
        }
    }

    return threadpool;
}

/// Ports `ggml_threadpool_new` (ggml-cpu.c:3351 @c1d0e7a00).
///
/// Parameters:
/// - `tpp`: the pool's configuration.
///
/// Return: an idle pool, with no graph attached yet.
pub export fn ggml_threadpool_new(tpp: *c.struct_ggml_threadpool_params) *Threadpool {
    return threadpoolNewImpl(tpp, null, null);
}

/// Ports `ggml_threadpool_free` (ggml-cpu.c:2711 @c1d0e7a00).
///
/// Stops the workers, joins them, and releases both allocations.
///
/// Parameters:
/// - `threadpool_opt`: the pool, or null, which the C tolerates.
pub export fn ggml_threadpool_free(threadpool_opt: ?*Threadpool) void {
    const threadpool = threadpool_opt orelse return;

    const n: usize = @intCast(threadpool.n_threads);
    const workers = threadpool.workers.?;

    lock(threadpool);
    threadpool.stop.store(true, .monotonic);
    threadpool.pause.store(false, .monotonic);
    condBroadcast(threadpool);
    unlock(threadpool);

    for (1..n) |j| workers[j].thrd.join();

    // `ggml_mutex_destroy`, `ggml_cond_destroy`, `ggml_mutex_destroy`,
    // `ggml_cond_destroy` (ggml-cpu.c:433, 440, 453, 470 @c1d0e7a00): empty on
    // Windows.
    if (comptime !is_windows) {
        _ = std.c.pthread_mutex_destroy(&threadpool.mutex);
        _ = std.c.pthread_cond_destroy(&threadpool.cond);
    }

    ggml_aligned_free(workers, @sizeOf(ComputeState) * n);
    ggml_aligned_free(threadpool, @sizeOf(Threadpool));
}

/// Ports `ggml_threadpool_pause_locked` (ggml-cpu.c:2744 @c1d0e7a00).
///
/// The caller must hold `threadpool.mutex`.
fn pauseLocked(threadpool: *Threadpool) void {
    impl.printDebug("Pausing threadpool\n", .{});
    threadpool.pause.store(true, .monotonic);
    condBroadcast(threadpool);
}

/// Ports `ggml_threadpool_resume_locked` (ggml-cpu.c:2750 @c1d0e7a00).
///
/// The caller must hold `threadpool.mutex`.
fn resumeLocked(threadpool: *Threadpool) void {
    impl.printDebug("Resuming threadpool\n", .{});
    threadpool.pause.store(false, .monotonic);
    condBroadcast(threadpool);
}

/// Ports `ggml_threadpool_pause` (ggml-cpu.c:2757 @c1d0e7a00).
pub export fn ggml_threadpool_pause(threadpool: *Threadpool) void {
    lock(threadpool);
    defer unlock(threadpool);

    if (!threadpool.pause.load(.monotonic)) pauseLocked(threadpool);
}

/// Ports `ggml_threadpool_resume` (ggml-cpu.c:2769 @c1d0e7a00).
pub export fn ggml_threadpool_resume(threadpool: *Threadpool) void {
    lock(threadpool);
    defer unlock(threadpool);

    if (threadpool.pause.load(.monotonic)) resumeLocked(threadpool);
}

// -----------------------------------------------------------------------------
// Running a graph

/// Ports `ggml_cpu_try_fuse_ops` (ggml-cpu.c:3031 @c1d0e7a00).
///
/// Parameters:
/// - `cgraph`: the graph being computed.
/// - `node_n`: index of the node about to run.
/// - `params`: the thread's compute parameters.
/// - `cplan`: the plan, consulted for its reference-only flag.
///
/// Return: how many *extra* nodes the fused kernel consumed, or 0 when
/// nothing fused. The C returns `int` and documents it as ">= 1 or 0", but
/// the caller adds it to `node_n` after the loop's own increment, so 1 here
/// means two nodes were computed.
fn tryFuseOps(
    cgraph: *const CGraph,
    node_n: c_int,
    params: *const ComputeParams,
    cplan: *const c.ggml_cplan,
) c_int {
    if (features.disable_fusion or cplan.use_ref) return 0;

    const node = cgraph.nodes[@intCast(node_n)].?;

    if (node.op == c.GGML_OP_RMS_NORM) {
        const fuse_ops = [_]c.enum_ggml_op{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL };
        if (graph.canFuse(cgraph, node_n, &fuse_ops)) {
            const mul_node = cgraph.nodes[@intCast(node_n + 1)].?;
            const mul_w = if (mul_node.src[0] == node)
                impl.one(Tensor, mul_node.src[1])
            else
                impl.one(Tensor, mul_node.src[0]);

            if (impl.one(Tensor, node.src[0]).type == c.GGML_TYPE_F32 and
                mul_node.type == c.GGML_TYPE_F32 and
                mul_w.type == c.GGML_TYPE_F32 and
                mul_w.ne[0] == node.ne[0] and
                mul_w.nb[0] == @sizeOf(f32))
            {
                ggml_compute_forward_rms_norm_mul_fused(params, node, mul_node);
                return 1;
            }
        }
    }

    return 0;
}

/// Ports `ggml_graph_compute_thread` (ggml-cpu.c:3065 @c1d0e7a00).
///
/// The body every thread runs for one graph, main thread included. Walks the
/// nodes in order, computing this thread's share of each and rendezvousing at
/// the barrier between them.
///
/// Parameters:
/// - `state`: this thread's slot in the pool.
fn graphComputeThread(state: *ComputeState) void {
    const tp = state.threadpool;

    const cgraph = tp.cgraph.?;
    const cplan = tp.cplan.?;

    setNumaThreadAffinity(state.ith);

    const params = ComputeParams{
        .ith = state.ith,
        .nth = tp.n_graph.load(.monotonic) & defs.n_threads_mask,
        .wsize = cplan.work_size,
        .wdata = cplan.work_data,
        .threadpool = tp,
        .use_ref = cplan.use_ref,
    };

    var node_n: c_int = 0;
    while (node_n < cgraph.n_nodes and tp.abort.load(.monotonic) != node_n) : (node_n += 1) {
        const node = cgraph.nodes[@intCast(node_n)].?;

        if (impl.opIsEmpty(node.op)) continue;
        if ((node.flags & c.GGML_TENSOR_FLAG_COMPUTE) == 0) continue;

        // The C notes this belongs in `ggml_graph_plan`, so the decision is
        // made once rather than per thread per graph.
        const n_fused = tryFuseOps(cgraph, node_n, &params, cplan);
        if (n_fused > 0) {
            node_n += n_fused;
        } else {
            forward.computeForward(&params, node);
        }

        if (state.ith == 0) {
            if (cplan.abort_callback) |cb| {
                if (cb(cplan.abort_callback_data)) {
                    tp.abort.store(node_n + 1, .monotonic);
                    tp.ec = c.GGML_STATUS_ABORTED;
                }
            }
        }

        if (node_n + 1 < cgraph.n_nodes) ggml_barrier(tp);
    }

    ggml_barrier(tp);
}

/// Ports `ggml_graph_compute_thread_ready` (ggml-cpu.c:3144 @c1d0e7a00).
///
/// Return: true when the polling or sleeping loop should exit; `state.pending`
/// says whether that is because there is work.
fn threadReady(state: *ComputeState) bool {
    const threadpool = state.threadpool;

    if (state.pending or threadpool.stop.load(.monotonic) or threadpool.pause.load(.monotonic)) {
        return true;
    }

    const n_graph = threadpool.n_graph.load(.monotonic);
    const n_threads = n_graph & defs.n_threads_mask;
    if (n_graph != state.last_graph) {
        // A graph running fewer threads than the pool has leaves the tail
        // idle, so `pending` is not simply true here.
        state.pending = state.ith < n_threads;
        state.last_graph = n_graph;
        return true;
    }

    return false;
}

/// Ports `ggml_graph_compute_thread_sync` (ggml-cpu.c:3162 @c1d0e7a00).
///
/// The fence that pairs with the kickoff's sequentially-consistent store, for
/// a thread that saw the new graph by polling rather than by waking.
///
/// Adding zero rather than fencing, for the reason given in `ggml_barrier`.
/// This is also exactly the C's thread-sanitizer fallback, on the same field.
fn threadSync(state: *ComputeState) void {
    _ = state.threadpool.n_graph.fetchAdd(0, .seq_cst);
}

/// Ports `ggml_graph_compute_poll_for_work` (ggml-cpu.c:3172 @c1d0e7a00).
///
/// Return: whether work arrived before the polling budget ran out.
fn pollForWork(state: *ComputeState) bool {
    const threadpool = state.threadpool;

    // The C's note: this makes 0..100 a reasonable range for the polling
    // level across modern processors.
    const n_rounds: u64 = 1024 * 128 * @as(u64, threadpool.poll);

    var i: u64 = 0;
    while (!threadReady(state) and i < n_rounds) : (i += 1) {
        defs.cpuRelax();
    }

    return state.pending;
}

/// Ports `ggml_graph_compute_check_for_work` (ggml-cpu.c:3187 @c1d0e7a00).
///
/// Polls first, then sleeps on the condition variable. The hybrid is why the
/// kickoff always takes the mutex.
fn checkForWork(state: *ComputeState) bool {
    const threadpool = state.threadpool;

    if (pollForWork(state)) {
        threadSync(state);
        return state.pending;
    }

    lockShared(threadpool);
    while (!threadReady(state)) {
        impl.printDebug("thread #%d waiting for work (sleeping)\n", .{state.ith});
        condWait(threadpool);
    }
    unlockShared(threadpool);

    return state.pending;
}

/// Ports `ggml_graph_compute_secondary_thread` (ggml-cpu.c:3206 @c1d0e7a00).
///
/// The worker loop: sleep while paused, exit on stop, otherwise wait for a
/// graph and run this thread's share of it.
fn secondaryThread(state: *ComputeState) void {
    const threadpool = state.threadpool;

    _ = threadApplyPriority(threadpool.prio);
    if (cpumaskIsValid(&state.cpumask)) {
        _ = threadApplyAffinity(&state.cpumask);
    }

    while (true) {
        while (threadpool.pause.load(.monotonic)) {
            impl.printDebug("thread #%d inside pause loop\n", .{state.ith});
            lockShared(threadpool);
            if (threadpool.pause.load(.monotonic)) {
                condWait(threadpool);
            }
            impl.printDebug("thread #%d resuming after wait\n", .{state.ith});
            unlockShared(threadpool);
        }

        // Checked after the wait, not before it.
        if (threadpool.stop.load(.monotonic)) break;

        // Only the main thread dispatches work.
        _ = checkForWork(state);
        if (state.pending) {
            state.pending = false;
            graphComputeThread(state);
        }
    }
}

/// Ports `ggml_graph_compute_kickoff` (ggml-cpu.c:3244 @c1d0e7a00).
///
/// Publishes a new graph to the workers. The mutex is taken unconditionally
/// because the workers are doing a hybrid poll-then-wait and either half might
/// be the one that notices.
///
/// Parameters:
/// - `threadpool`: the pool.
/// - `n_threads`: how many of its threads this graph will use.
fn kickoff(threadpool: *Threadpool, n_threads: c_int) void {
    lock(threadpool);

    // Bump the graph counter and stamp in the active thread count.
    var n_graph = threadpool.n_graph.load(.monotonic) >> defs.n_threads_bits;
    n_graph = ((n_graph + 1) << defs.n_threads_bits) | (n_threads & defs.n_threads_mask);

    impl.printDebug("compute-kickoff: n_threads %d n_graph %d\n", .{ n_threads, n_graph });

    // Sequentially consistent because of the polling threads: `threadSync`
    // pairs with this store.
    threadpool.n_graph.store(n_graph, .seq_cst);

    if (threadpool.pause.load(.monotonic)) {
        // Bring the main thread's priority and placement up to the pool's
        // settings before resuming.
        _ = threadApplyPriority(threadpool.prio);
        if (cpumaskIsValid(&threadpool.workers.?[0].cpumask)) {
            _ = threadApplyAffinity(&threadpool.workers.?[0].cpumask);
        }
        // `resume` does the broadcast.
        resumeLocked(threadpool);
    } else {
        condBroadcast(threadpool);
    }

    unlock(threadpool);
}

/// Ports `ggml_graph_compute` (ggml-cpu.c:3355 @c1d0e7a00).
///
/// Parameters:
/// - `cgraph`: the graph to evaluate.
/// - `cplan`: the plan from `ggml_graph_plan`, with `work_data` allocated.
///
/// Return: the graph's status; `GGML_STATUS_ABORTED` if the abort callback
/// asked to stop.
pub export fn ggml_graph_compute(cgraph: *const CGraph, cplan: *c.ggml_cplan) c.enum_ggml_status {
    features.ggml_cpu_init();

    impl.assert(cplan.n_threads > 0, "cplan->n_threads > 0");
    impl.assert(
        cplan.work_size == 0 or cplan.work_data != null,
        "cplan->work_size == 0 || cplan->work_data != NULL",
    );

    var n_threads = cplan.n_threads;
    var threadpool: *Threadpool = undefined;

    var disposable_threadpool = false;

    if (cplan.threadpool) |tp| {
        threadpool = @ptrCast(@alignCast(tp));
        // Reset what needs resetting. No worker is looking at these yet.
        threadpool.cgraph = cgraph;
        threadpool.cplan = cplan;
        threadpool.current_chunk.store(0, .monotonic);
        threadpool.abort.store(-1, .monotonic);
        threadpool.ec = c.GGML_STATUS_SUCCESS;
    } else {
        disposable_threadpool = true;
        var ttp = ggml_threadpool_params_default(n_threads);
        threadpool = threadpoolNewImpl(&ttp, cgraph, cplan);
    }

    if (n_threads > threadpool.n_threads) {
        impl.logWarn(
            "cplan requested more threads (%d) than available (%d)\n",
            .{ n_threads, threadpool.n_threads },
        );
        n_threads = threadpool.n_threads;
    }

    kickoff(threadpool, n_threads);

    // The calling thread is a worker too.
    graphComputeThread(&threadpool.workers.?[0]);

    // Do not leave affinity set on the main thread.
    clearNumaThreadAffinity();

    const ret = threadpool.ec;

    if (disposable_threadpool) ggml_threadpool_free(threadpool);

    return ret;
}

/// Ports `ggml_graph_compute_with_ctx` (ggml-cpu.c:3432 @c1d0e7a00).
///
/// Plans the graph, takes the work buffer out of `ctx`, and runs it.
///
/// Parameters:
/// - `ctx`: context to allocate the work buffer in; it owns the buffer.
/// - `cgraph`: the graph to evaluate.
/// - `n_threads`: thread budget.
///
/// Return: the graph's status.
pub export fn ggml_graph_compute_with_ctx(
    ctx: *context.Context,
    cgraph: *const CGraph,
    n_threads: c_int,
) c.enum_ggml_status {
    var cplan = plan.ggml_graph_plan(cgraph, n_threads, null);
    cplan.work_data = @ptrCast(context.ggml_new_buffer(ctx, cplan.work_size));
    return ggml_graph_compute(cgraph, &cplan);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a single-threaded barrier returns without touching the counters" {
    // The early return is what makes the one-thread case free, and it reads
    // `n_graph`'s low bits to find out. A wrong mask here would spin forever.
    var tp: Threadpool = undefined;
    tp.n_graph = .init(1);
    tp.n_barrier = .init(0);
    tp.n_barrier_passed = .init(0);

    ggml_barrier(&tp);

    try std.testing.expectEqual(@as(c_int, 0), tp.n_barrier.load(.monotonic));
    try std.testing.expectEqual(@as(c_int, 0), tp.n_barrier_passed.load(.monotonic));
}

test "the graph counter packs a count and a generation" {
    // `kickoff` shifts the generation up and ORs the thread count in; every
    // reader masks it back out. The two have to agree on the split.
    const n_graph: c_int = ((3 + 1) << defs.n_threads_bits) | (5 & defs.n_threads_mask);

    try std.testing.expectEqual(@as(c_int, 5), n_graph & defs.n_threads_mask);
    try std.testing.expectEqual(@as(c_int, 4), n_graph >> defs.n_threads_bits);
}

test "the chunk cursor hands out each index once" {
    var tp: Threadpool = undefined;
    tp.current_chunk = .init(0);

    ggml_threadpool_chunk_set(&tp, 4);
    try std.testing.expectEqual(@as(c_int, 4), ggml_threadpool_chunk_add(&tp, 1));
    try std.testing.expectEqual(@as(c_int, 5), ggml_threadpool_chunk_add(&tp, 1));
    try std.testing.expectEqual(@as(c_int, 6), tp.current_chunk.load(.monotonic));
}

test "cpumask_next hands out one core each in strict mode" {
    var global: [c.GGML_MAX_N_THREADS]bool = @splat(false);
    global[2] = true;
    global[5] = true;

    var local: [c.GGML_MAX_N_THREADS]bool = @splat(false);
    var iter: i32 = 0;

    cpumaskNext(&global, &local, true, &iter);
    try std.testing.expect(local[2]);
    try std.testing.expect(!local[5]);
    try std.testing.expectEqual(@as(i32, 3), iter);

    cpumaskNext(&global, &local, true, &iter);
    try std.testing.expect(!local[2]);
    try std.testing.expect(local[5]);
    try std.testing.expectEqual(@as(i32, 6), iter);

    // Past the last set bit it wraps, so a third worker returns to core 2.
    cpumaskNext(&global, &local, true, &iter);
    try std.testing.expect(local[2]);
}

test "cpumask_next copies the whole mask when not strict" {
    var global: [c.GGML_MAX_N_THREADS]bool = @splat(false);
    global[2] = true;
    global[5] = true;

    var local: [c.GGML_MAX_N_THREADS]bool = @splat(false);
    var iter: i32 = 0;

    cpumaskNext(&global, &local, false, &iter);
    try std.testing.expect(local[2]);
    try std.testing.expect(local[5]);
    try std.testing.expectEqual(@as(i32, 0), iter);
}

test "an all-zero cpumask is not valid, which is what leaves placement alone" {
    var mask: [c.GGML_MAX_N_THREADS]bool = @splat(false);
    try std.testing.expect(!cpumaskIsValid(&mask));

    mask[7] = true;
    try std.testing.expect(cpumaskIsValid(&mask));
}

test "a threadpool starts, pauses, resumes and frees" {
    var tpp = ggml_threadpool_params_default(2);
    const tp = ggml_threadpool_new(&tpp);

    try std.testing.expectEqual(@as(c_int, 2), tp.n_threads);
    try std.testing.expect(tp.workers != null);

    ggml_threadpool_pause(tp);
    try std.testing.expect(tp.pause.load(.monotonic));

    ggml_threadpool_resume(tp);
    try std.testing.expect(!tp.pause.load(.monotonic));

    ggml_threadpool_free(tp);

    // The C tolerates a null pool here rather than asserting.
    ggml_threadpool_free(null);
}

test "NUMA stays off, which keeps mul_mat's chunking on" {
    try std.testing.expect(!ggml_is_numa());

    // Calling init on a platform with no NUMA support changes nothing, and
    // may be called more than once.
    ggml_numa_init(c.GGML_NUMA_STRATEGY_DISABLED);
    try std.testing.expect(!ggml_is_numa());
}
