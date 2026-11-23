//! The CPU backend's `ggml_backend` and `ggml_backend_device` interfaces.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp` at v1.0.0
//! (`b1b770d90`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # Not to be confused with `ggml-cpu.c`
//!
//! That file is already ported, across `cpu/{forward,plan,threading,…}.zig`.
//! This is the C++ one beside it: the backend registration and device
//! interface, where that one is the compute engine.
//!
//! # The extra-buffer list
//!
//! The C's `ggml_backend_cpu_get_extra_buffer_types()` returns
//! `std::vector<ggml_backend_buffer_type_t> &`, built once by a lambda. It
//! is the only C++-linkage symbol crossing `ggml-cpu/`, and it disappears
//! here: the list is a static array filled on first use, handed out as a
//! slice. `cpu/extra.zig` consumes it.
//!
//! # `new` without `delete`
//!
//! Several objects the C `new`s are never freed — the device context, the
//! registration. Those are process-lifetime singletons and the C leaks them
//! deliberately; the port allocates them the same way rather than inventing
//! an ownership model the C does not have.

const std = @import("std");
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");
const builtin = @import("builtin");

/// The C `new`s these and mostly never frees them. libc's allocator is what
/// a C entry point can reach without being handed one — the same reasoning
/// as `backend_reg.zig` and `backend.zig`.

// -----------------------------------------------------------------------------
// The extra buffer types

/// Storage for `ggml_backend_cpu_get_extra_buffer_types` (ggml-cpu.cpp:42
/// @b1b770d90).
///
/// Only the repack entry compiles on this target: the AMX, SpaceMIT and
/// KleidiAI arms are behind `__AMX_INT8__`, `GGML_USE_CPU_RISCV64_SPACEMIT`
/// and `GGML_USE_CPU_KLEIDIAI`, none of which this build defines.
var extra_bufts: [2]ggml.impl.ggml_c.ggml_backend_buffer_type_t = .{ null, null };
var extra_bufts_n: usize = 0;

/// The C++ magic-static guard, written out — the same shape
/// `backend_reg.zig` uses for `get_reg`. Every call after the first is a
/// single acquire load.
var extra_bufts_ready = std.atomic.Value(bool).init(false);
var extra_bufts_mutex: stdlib.sync.Mutex = .{};

fn ensureExtraBufts() void {
    if (extra_bufts_ready.load(.acquire)) return;
    extra_bufts_mutex.lock();
    defer extra_bufts_mutex.unlock();
    if (extra_bufts_ready.load(.acquire)) return;

    const buft = ggml.cpu.repack.ggml_backend_cpu_repack_buffer_type();
    if (buft != null) {
        extra_bufts[extra_bufts_n] = buft;
        extra_bufts_n += 1;
    }
    extra_bufts_ready.store(true, .release);
}

/// Ports `ggml_backend_cpu_get_extra_buffer_types` (ggml-cpu.cpp:42
/// @b1b770d90).
///
/// Return: the extra buffer types, borrowed for the life of the process.
pub fn getExtraBufferTypes() []const ggml.impl.ggml_c.ggml_backend_buffer_type_t {
    ensureExtraBufts();
    return extra_bufts[0..extra_bufts_n];
}

/// Ports `ggml_backend_cpu_device_get_extra_buffers_type` (ggml-cpu.cpp:76
/// @b1b770d90): the same list, null-terminated, as the proc-address export
/// hands to `llama.cpp`.
fn deviceGetExtraBuffersType(device: ggml.impl.ggml_c.ggml_backend_dev_t) callconv(.c) [*c]ggml.impl.ggml_c.ggml_backend_buffer_type_t {
    _ = device;
    ensureExtraBufts();
    // `extra_bufts` has a trailing null slot that `extra_bufts_n` excludes,
    // which is the `bufts.push_back(nullptr)` the C does into a second
    // vector.
    return &extra_bufts;
}

/// Ports `ggml_backend_cpu_is_extra_buffer_type` (ggml-cpu.cpp:88
/// @b1b770d90).
fn isExtraBufferType(buft: ggml.impl.ggml_c.ggml_backend_buffer_type_t) bool {
    for (getExtraBufferTypes()) |e| {
        if (e == buft) return true;
    }
    return false;
}

// -----------------------------------------------------------------------------
// The backend (stream)

/// Ports `struct ggml_backend_cpu_context` (ggml-cpu.cpp:99 @b1b770d90).
const Context = struct {
    n_threads: c_int,
    threadpool: ?*ggml.cpu.defs.Threadpool,
    work_data: ?[*]u8,
    work_size: usize,
    abort_callback: ggml.impl.ggml_c.ggml_abort_callback,
    abort_callback_data: ?*anyopaque,
    /// Take the reference path, skipping fused and vectorised kernels.
    use_ref: bool,
};

/// Ports `ggml_backend_cpu_get_name` (ggml-cpu.cpp:112 @b1b770d90).
fn getName(backend: ggml.impl.ggml_c.ggml_backend_t) callconv(.c) [*c]const u8 {
    _ = backend;
    return "CPU";
}

/// Ports `ggml_backend_cpu_free` (ggml-cpu.cpp:118 @b1b770d90).
fn backendFree(backend: ggml.impl.ggml_c.ggml_backend_t) callconv(.c) void {
    const b = ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend, backend);
    const ctx: *Context = @ptrCast(@alignCast(b.context.?));
    if (ctx.work_data) |w| std.heap.c_allocator.free(w[0..ctx.work_size]);
    std.heap.c_allocator.destroy(ctx);
    std.heap.c_allocator.destroy(b);
}

/// Ports `struct ggml_backend_plan_cpu` (ggml-cpu.cpp:125 @b1b770d90).
const PlanCpu = struct {
    cplan: ggml.impl.ggml_c.struct_ggml_cplan,
    cgraph: ggml.impl.CGraph,
};

/// Ports `ggml_backend_cpu_graph_plan_create` (ggml-cpu.cpp:130 @b1b770d90).
fn graphPlanCreate(backend: ggml.impl.ggml_c.ggml_backend_t, cgraph: *const ggml.impl.CGraph) callconv(.c) ggml.impl.ggml_c.ggml_backend_graph_plan_t {
    const b = ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend, backend);
    const ctx: *Context = @ptrCast(@alignCast(b.context.?));

    const plan = std.heap.c_allocator.create(PlanCpu) catch return null;

    plan.cplan = ggml.cpu.plan.ggml_graph_plan(@ptrCast(cgraph), ctx.n_threads, ctx.threadpool);
    // The C's own comment: "FIXME: deep copy".
    plan.cgraph = cgraph.*;

    if (plan.cplan.work_size > 0) {
        const buf = std.heap.c_allocator.alloc(u8, plan.cplan.work_size) catch {
            std.heap.c_allocator.destroy(plan);
            return null;
        };
        plan.cplan.work_data = buf.ptr;
    }

    plan.cplan.abort_callback = ctx.abort_callback;
    plan.cplan.abort_callback_data = ctx.abort_callback_data;
    plan.cplan.use_ref = ctx.use_ref;

    return plan;
}

/// Ports `ggml_backend_cpu_graph_plan_free` (ggml-cpu.cpp:153 @b1b770d90).
fn graphPlanFree(backend: ggml.impl.ggml_c.ggml_backend_t, plan_in: ggml.impl.ggml_c.ggml_backend_graph_plan_t) callconv(.c) void {
    _ = backend;
    const plan: *PlanCpu = @ptrCast(@alignCast(plan_in.?));
    if (plan.cplan.work_data) |w| std.heap.c_allocator.free(w[0..plan.cplan.work_size]);
    std.heap.c_allocator.destroy(plan);
}

/// Ports `ggml_backend_cpu_graph_plan_compute` (ggml-cpu.cpp:162 @b1b770d90).
fn graphPlanCompute(backend: ggml.impl.ggml_c.ggml_backend_t, plan_in: ggml.impl.ggml_c.ggml_backend_graph_plan_t) callconv(.c) ggml.impl.ggml_c.enum_ggml_status {
    _ = backend;
    const plan: *PlanCpu = @ptrCast(@alignCast(plan_in.?));
    return ggml.cpu.threading.ggml_graph_compute(@ptrCast(&plan.cgraph), &plan.cplan);
}

/// Ports `ggml_backend_cpu_graph_compute` (ggml-cpu.cpp:170 @b1b770d90).
fn graphCompute(backend: ggml.impl.ggml_c.ggml_backend_t, cgraph: *ggml.impl.CGraph) callconv(.c) ggml.impl.ggml_c.enum_ggml_status {
    const b = ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend, backend);
    const ctx: *Context = @ptrCast(@alignCast(b.context.?));

    var cplan = ggml.cpu.plan.ggml_graph_plan(@ptrCast(cgraph), ctx.n_threads, ctx.threadpool);

    if (ctx.work_size < cplan.work_size) {
        if (ctx.work_data) |w| std.heap.c_allocator.free(w[0..ctx.work_size]);
        const buf = std.heap.c_allocator.alloc(u8, cplan.work_size) catch {
            ctx.work_data = null;
            ctx.work_size = 0;
            return ggml.impl.ggml_c.GGML_STATUS_ALLOC_FAILED;
        };
        ctx.work_data = buf.ptr;
        ctx.work_size = cplan.work_size;
    }
    cplan.work_data = ctx.work_data;

    cplan.abort_callback = ctx.abort_callback;
    cplan.abort_callback_data = ctx.abort_callback_data;
    cplan.use_ref = ctx.use_ref;

    return ggml.cpu.threading.ggml_graph_compute(@ptrCast(cgraph), &cplan);
}

/// Ports `ggml_backend_cpu_i` (ggml-cpu.cpp:193 @b1b770d90).
const backend_i = ggml.impl.ggml_c.struct_ggml_backend_i{
    .get_name = getName,
    .free = backendFree,
    .set_tensor_async = null,
    .get_tensor_async = null,
    .set_tensor_2d_async = null,
    .get_tensor_2d_async = null,
    .cpy_tensor_async = null,
    .synchronize = null,
    .graph_plan_create = @ptrCast(&graphPlanCreate),
    .graph_plan_free = graphPlanFree,
    .graph_plan_update = null,
    .graph_plan_compute = graphPlanCompute,
    .graph_compute = @ptrCast(&graphCompute),
    .event_record = null,
    .event_wait = null,
    .graph_optimize = null,
};

/// Ports `ggml_backend_cpu_guid` (ggml-cpu.cpp:212 @b1b770d90).
var cpu_guid: ggml.impl.ggml_c.ggml_guid = .{ 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };

fn guid() ggml.impl.ggml_c.ggml_guid_t {
    return &cpu_guid;
}

/// Ports `ggml_backend_cpu_init` (ggml-cpu.cpp:217 @b1b770d90).
///
/// Return: a new CPU backend, or null if allocation failed. The caller owns
/// it and frees it with `ggml_backend_free`.
pub fn ggml_backend_cpu_init() callconv(.c) ggml.impl.ggml_c.ggml_backend_t {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml.cpu.features.ggml_cpu_init();

    const ctx = std.heap.c_allocator.create(Context) catch return null;
    ctx.* = .{
        .n_threads = ggml.impl.ggml_c.GGML_DEFAULT_N_THREADS,
        .threadpool = null,
        .work_data = null,
        .work_size = 0,
        .abort_callback = null,
        .abort_callback_data = null,
        .use_ref = false,
    };

    const backend = std.heap.c_allocator.create(ggml.impl.ggml_c.struct_ggml_backend) catch {
        std.heap.c_allocator.destroy(ctx);
        return null;
    };
    backend.* = .{
        .guid = guid(),
        .iface = backend_i,
        .device = ggml.backend.ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        .context = ctx,
    };
    return backend;
}

/// Ports `ggml_backend_is_cpu` (ggml-cpu.cpp:249 @b1b770d90).
pub fn ggml_backend_is_cpu(backend: ggml.impl.ggml_c.ggml_backend_t) callconv(.c) bool {
    return backend != null and ggml.runtime.ggml_guid_matches(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend, backend).guid, guid());
}

inline fn contextOf(backend_cpu: ggml.impl.ggml_c.ggml_backend_t) *Context {
    ggml.impl.assert(ggml_backend_is_cpu(backend_cpu), "ggml_backend_is_cpu(backend_cpu)");
    return @ptrCast(@alignCast(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend, backend_cpu).context.?));
}

/// Ports `ggml_backend_cpu_set_n_threads` (ggml-cpu.cpp:253 @b1b770d90).
pub fn ggml_backend_cpu_set_n_threads(backend_cpu: ggml.impl.ggml_c.ggml_backend_t, n_threads: c_int) callconv(.c) void {
    contextOf(backend_cpu).n_threads = n_threads;
}

/// Ports `ggml_backend_cpu_set_threadpool` (ggml-cpu.cpp:260 @b1b770d90).
pub fn ggml_backend_cpu_set_threadpool(backend_cpu: ggml.impl.ggml_c.ggml_backend_t, threadpool: ggml.impl.ggml_c.ggml_threadpool_t) callconv(.c) void {
    const ctx = contextOf(backend_cpu);
    const tp: ?*ggml.cpu.defs.Threadpool = @ptrCast(@alignCast(threadpool));
    if (ctx.threadpool != null and ctx.threadpool != tp) {
        // already had a different threadpool, pause/suspend it before switching
        ggml.cpu.threading.ggml_threadpool_pause(ctx.threadpool.?);
    }
    ctx.threadpool = tp;
}

/// Ports `ggml_backend_cpu_set_abort_callback` (ggml-cpu.cpp:272 @b1b770d90).
pub fn ggml_backend_cpu_set_abort_callback(backend_cpu: ggml.impl.ggml_c.ggml_backend_t, abort_callback: ggml.impl.ggml_c.ggml_abort_callback, abort_callback_data: ?*anyopaque) callconv(.c) void {
    const ctx = contextOf(backend_cpu);
    ctx.abort_callback = abort_callback;
    ctx.abort_callback_data = abort_callback_data;
}

/// Ports `ggml_backend_cpu_set_use_ref` (ggml-cpu.cpp:280 @b1b770d90).
pub fn ggml_backend_cpu_set_use_ref(backend_cpu: ggml.impl.ggml_c.ggml_backend_t, use_ref: bool) callconv(.c) void {
    contextOf(backend_cpu).use_ref = use_ref;
}

// -----------------------------------------------------------------------------
// The device

/// Ports `struct ggml_backend_cpu_device_context` (ggml-cpu.cpp:289
/// @b1b770d90), the `__APPLE__`, `__linux__` and `_WIN32` arms of its
/// constructor.
///
/// The C holds a `std::string` and resizes it around `sysctlbyname`; here it
/// is a fixed buffer, because the brand string is a short, bounded value and
/// the C never frees its string either.
const DeviceContext = struct {
    description: [256]u8 = undefined,
    description_len: usize = 0,

    fn init(self: *DeviceContext) void {
        const fallback = "CPU";
        @memcpy(self.description[0..fallback.len], fallback);
        self.description[fallback.len] = 0;
        self.description_len = fallback.len;

        if (comptime builtin.os.tag == .linux) return self.initLinux();
        if (comptime builtin.os.tag == .windows) return self.initWindows();
        // The C's `#ifdef __APPLE__`: iOS reads the brand string too.
        if (!builtin.os.tag.isDarwin()) return;
        var len: usize = self.description.len;
        if (std.c.sysctlbyname("machdep.cpu.brand_string", &self.description, &len, null, 0) == 0 and len > 0) {
            // `len` counts the NUL the call wrote.
            self.description_len = len - 1;
            self.description[self.description_len] = 0;
        }
    }

    /// The `_WIN32` arm of `ggml_backend_cpu_device_context`
    /// (ggml-cpu.cpp:292 @b1b770d90): the first processor's
    /// `ProcessorNameString` from the registry, cut at its first NUL.
    ///
    /// The C resizes "CPU" to the size the first query reports before the
    /// second fills it, so a second query that fails leaves "CPU" padded with
    /// NULs -- or cut short, for a size under 3 -- which reads the same as
    /// stopping at the first NUL here.
    fn initWindows(self: *DeviceContext) void {
        var key: std.os.windows.HKEY = undefined;
        if (ggml.win32.RegOpenKeyExA(ggml.win32.HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, ggml.win32.KEY_READ, &key) != ggml.win32.ERROR_SUCCESS) return;
        defer _ = ggml.win32.RegCloseKey(key);

        var cpu_brand_size: std.os.windows.DWORD = 0;
        if (ggml.win32.RegQueryValueExA(key, "ProcessorNameString", null, null, null, &cpu_brand_size) != ggml.win32.ERROR_SUCCESS) return;

        // `description.resize(cpu_brand_size)`.
        const n = @min(cpu_brand_size, self.description.len - 1);
        if (n < self.description_len) self.description_len = n;
        @memset(self.description[self.description_len .. n + 1], 0);
        self.description_len = n;

        var size: std.os.windows.DWORD = @intCast(n);
        if (ggml.win32.RegQueryValueExA(key, "ProcessorNameString", null, null, &self.description, &size) == ggml.win32.ERROR_SUCCESS) {
            self.description[n] = 0;
        }
        self.description_len = std.mem.sliceTo(&self.description, 0).len;
    }

    /// The C's `#elif defined(__linux__)` arm: the first `model name` line of
    /// `/proc/cpuinfo`, trimmed of whitespace. On a CPU whose cpuinfo has no
    /// such line -- most ARM kernels -- the description stays "CPU".
    fn initLinux(self: *DeviceContext) void {
        const f = stdlib.libc.fopen("/proc/cpuinfo", "r") orelse return;
        defer _ = stdlib.libc.fclose(f);
        var buf: [1024]u8 = undefined;
        while (stdlib.libc.fgets(&buf, buf.len, f) != null) {
            const line = std.mem.sliceTo(&buf, 0);
            if (!std.mem.startsWith(u8, line, "model name")) continue;
            const colon = std.mem.findScalar(u8, line, ':') orelse continue;
            // `std::isspace` in the "C" locale.
            const p = std.mem.trim(u8, line[colon + 1 ..], " \t\n\x0b\x0c\r");
            const n = @min(p.len, self.description.len - 1);
            @memmove(self.description[0..n], p[0..n]);
            self.description[n] = 0;
            self.description_len = n;
            break;
        }
    }
};

/// `_SC_PHYS_PAGES` and `_SC_PAGE_SIZE`: Darwin's values, and glibc's.
const sc_phys_pages: c_int = if (builtin.os.tag.isDarwin()) 200 else 85;
const sc_page_size: c_int = if (builtin.os.tag.isDarwin()) 29 else 30;

fn deviceGetName(dev: ggml.impl.ggml_c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    _ = dev;
    return "CPU";
}

/// Ports `ggml_backend_cpu_device_get_description` (ggml-cpu.cpp:359
/// @b1b770d90).
fn deviceGetDescription(dev: ggml.impl.ggml_c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    const d = ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_device, dev);
    const ctx: *DeviceContext = @ptrCast(@alignCast(d.context.?));
    return &ctx.description;
}

/// Ports `ggml_backend_cpu_device_get_memory` (ggml-cpu.cpp:365 @b1b770d90),
/// both arms. Windows reports the physical memory available as free; every
/// other platform reports all of it.
fn deviceGetMemory(dev: ggml.impl.ggml_c.ggml_backend_dev_t, free: *usize, total: *usize) callconv(.c) void {
    _ = dev;
    if (comptime builtin.os.tag == .windows) {
        var status: ggml.win32.MEMORYSTATUSEX = undefined;
        status.dwLength = @sizeOf(ggml.win32.MEMORYSTATUSEX);
        _ = ggml.win32.GlobalMemoryStatusEx(&status);
        total.* = status.ullTotalPhys;
        free.* = status.ullAvailPhys;
        return;
    }
    const pages = std.c.sysconf(sc_phys_pages);
    const page_size = std.c.sysconf(sc_page_size);
    total.* = @intCast(pages * page_size);
    // "free" system memory is ill-defined, for practical purposes assume
    // that all of it is free:
    free.* = total.*;
}

fn deviceGetType(dev: ggml.impl.ggml_c.ggml_backend_dev_t) callconv(.c) ggml.impl.ggml_c.enum_ggml_backend_dev_type {
    _ = dev;
    return ggml.impl.ggml_c.GGML_BACKEND_DEVICE_TYPE_CPU;
}

/// Ports `ggml_backend_cpu_device_get_props` (ggml-cpu.cpp:390 @b1b770d90).
fn deviceGetProps(dev: ggml.impl.ggml_c.ggml_backend_dev_t, props: *ggml.impl.ggml_c.struct_ggml_backend_dev_props) callconv(.c) void {
    props.name = deviceGetName(dev);
    props.description = deviceGetDescription(dev);
    props.type = deviceGetType(dev);
    deviceGetMemory(dev, &props.memory_free, &props.memory_total);
    props.caps = .{
        .async = false,
        .host_buffer = false,
        .buffer_from_host_ptr = true,
        .events = false,
        .mmap_support = true,
    };
}

fn deviceInitBackend(dev: ggml.impl.ggml_c.ggml_backend_dev_t, params: [*c]const u8) callconv(.c) ggml.impl.ggml_c.ggml_backend_t {
    _ = dev;
    _ = params;
    return ggml_backend_cpu_init();
}

fn deviceGetBufferType(dev: ggml.impl.ggml_c.ggml_backend_dev_t) callconv(.c) ggml.impl.ggml_c.ggml_backend_buffer_type_t {
    _ = dev;
    return ggml.backend.ggml_backend_cpu_buffer_type();
}

fn deviceBufferFromHostPtr(dev: ggml.impl.ggml_c.ggml_backend_dev_t, ptr: ?*anyopaque, size: usize, max_tensor_size: usize) callconv(.c) ggml.impl.ggml_c.ggml_backend_buffer_t {
    _ = dev;
    _ = max_tensor_size;
    return ggml.backend.ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

/// Ports `ggml_backend_cpu_device_supports_op` (ggml-cpu.cpp:424 @b1b770d90).
/// `op->src[i]` narrowed, or null.
///
/// Ports nothing by itself — the C writes `const struct ggml_tensor * src0
/// = op->src[0];`, which is a plain copy that may be null. This is that,
/// plus the `[*c]` narrowing Zig 0.16 needs before any `.ne[i]` on it.
inline fn srcOf(p: [*c]ggml.impl.ggml_c.ggml_tensor) ?*const ggml.impl.ggml_c.ggml_tensor {
    return if (p == null) null else ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, p);
}

fn deviceSupportsOp(dev: ggml.impl.ggml_c.ggml_backend_dev_t, op: *const ggml.impl.ggml_c.ggml_tensor) callconv(.c) bool {
    // The C copies the two raw pointers up front and lets each arm below
    // assume its own are non-null. Both must stay **optional** here: a
    // leaf weight reaches this with `op == GGML_OP_NONE` and no sources at
    // all, and narrowing a null `[*c]` to `*T` traps in a safe build.
    // Measured — eagerly narrowing them aborted `llama_init_from_model` in
    // Debug while passing every ReleaseFast gate, where the cast is
    // unchecked. `make parity-cli` is what caught it.
    //
    // Narrowed rather than left as `[*c]` because Zig 0.16 types
    // `p.*.ne[i]` on a C pointer as the whole array and indexes in steps of
    // its size.
    const src0 = srcOf(op.src[0]);
    const src1 = srcOf(op.src[1]);

    if (op.op == ggml.impl.ggml_c.GGML_OP_NONE or op.op == ggml.impl.ggml_c.GGML_OP_RESHAPE or op.op == ggml.impl.ggml_c.GGML_OP_VIEW or
        op.op == ggml.impl.ggml_c.GGML_OP_PERMUTE or op.op == ggml.impl.ggml_c.GGML_OP_TRANSPOSE)
    {
        return true;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to
    // reduce overhead, increase if necessary
    for (0..4) |i| {
        const s = op.src[i] orelse continue;
        const buffer = s.*.buffer orelse continue;
        if (isExtraBufferType(buffer.*.buft)) {
            const buf_extra: *const ggml.cpu.extra.ExtraBufferType = @ptrCast(@alignCast(buffer.*.buft.*.context.?));
            return buf_extra.vtable.supports_op(buf_extra, dev, op);
        }
    }

    switch (op.op) {
        ggml.impl.ggml_c.GGML_OP_CPY, ggml.impl.ggml_c.GGML_OP_SET_ROWS => {
            // missing type_traits.from_float
            return op.type != ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS and op.type != ggml.impl.ggml_c.GGML_TYPE_IQ3_S and
                op.type != ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS and op.type != ggml.impl.ggml_c.GGML_TYPE_IQ2_XS and
                op.type != ggml.impl.ggml_c.GGML_TYPE_IQ2_S and op.type != ggml.impl.ggml_c.GGML_TYPE_IQ1_S and
                op.type != ggml.impl.ggml_c.GGML_TYPE_IQ1_M;
        },
        ggml.impl.ggml_c.GGML_OP_MUL_MAT => {
            if (ggml.impl.getOpParamsI32(op, 1) == ggml.impl.ggml_c.GGML_HINT_SRC0_IS_HADAMARD and
                src0.?.type == ggml.impl.ggml_c.GGML_TYPE_F32 and op.type == ggml.impl.ggml_c.GGML_TYPE_F32)
            {
                return src1.?.type == ggml.impl.ggml_c.GGML_TYPE_F32 or src1.?.type == ggml.impl.ggml_c.GGML_TYPE_F16;
            }
            // F16 or BF16 in src1 is widened into the F32 work buffer
            return src1.?.type == ggml.impl.ggml_c.GGML_TYPE_F32 or
                src1.?.type == ggml.cpu.traits.ggml_get_type_traits_cpu(src0.?.type).*.vec_dot_type or
                ((src1.?.type == ggml.impl.ggml_c.GGML_TYPE_F16 or src1.?.type == ggml.impl.ggml_c.GGML_TYPE_BF16) and
                    ggml.cpu.traits.ggml_get_type_traits_cpu(src0.?.type).*.vec_dot_type == ggml.impl.ggml_c.GGML_TYPE_F32);
        },
        ggml.impl.ggml_c.GGML_OP_SOFT_MAX_BACK => {
            if (src0.?.type != ggml.impl.ggml_c.GGML_TYPE_F32 or src1.?.type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;
            var max_bias: f32 = 0.0;
            const params: [*]const f32 = @ptrCast(&op.op_params);
            @memcpy(@as([*]u8, @ptrCast(&max_bias))[0..4], @as([*]const u8, @ptrCast(params + 1))[0..4]);
            return max_bias == 0.0;
        },
        ggml.impl.ggml_c.GGML_OP_IM2COL_BACK => {
            return src0.?.type == ggml.impl.ggml_c.GGML_TYPE_F32 and
                (src1.?.type == ggml.impl.ggml_c.GGML_TYPE_F32 or src1.?.type == ggml.impl.ggml_c.GGML_TYPE_F16);
        },
        ggml.impl.ggml_c.GGML_OP_GET_ROWS_BACK => {
            return src0.?.type == ggml.impl.ggml_c.GGML_TYPE_F32 or src0.?.type == ggml.impl.ggml_c.GGML_TYPE_F16;
        },
        ggml.impl.ggml_c.GGML_OP_OUT_PROD => {
            return (src0.?.type == ggml.impl.ggml_c.GGML_TYPE_F32 or
                ((src0.?.type == ggml.impl.ggml_c.GGML_TYPE_F16 or ggml.types.ggml_is_quantized(src0.?.type)) and
                    src0.?.ne[2] == src1.?.ne[2] and src0.?.ne[3] == src1.?.ne[3])) and
                src1.?.type == ggml.impl.ggml_c.GGML_TYPE_F32 and op.type == ggml.impl.ggml_c.GGML_TYPE_F32;
        },
        ggml.impl.ggml_c.GGML_OP_CONV_2D => return ggml.types.ggml_is_contiguous(src0.?),
        ggml.impl.ggml_c.GGML_OP_SSM_SCAN => {
            return ggml.impl.getOpParamsI32(op, 0) == 1 or ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, op.src[3]).ne[0] == 1;
        },
        else => return true,
    }
}

/// Ports `ggml_backend_cpu_device_supports_buft` (ggml-cpu.cpp:490
/// @b1b770d90).
fn deviceSupportsBuft(dev: ggml.impl.ggml_c.ggml_backend_dev_t, buft: ggml.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) bool {
    _ = dev;
    return ggml.backend.ggml_backend_buft_is_host(buft) or isExtraBufferType(buft);
}

/// Ports `ggml_backend_cpu_device_i` (ggml-cpu.cpp:495 @b1b770d90).
const device_i = ggml.impl.ggml_c.struct_ggml_backend_device_i{
    .get_name = deviceGetName,
    .get_description = deviceGetDescription,
    .get_memory = @ptrCast(&deviceGetMemory),
    .get_type = deviceGetType,
    .get_props = @ptrCast(&deviceGetProps),
    .init_backend = deviceInitBackend,
    .get_buffer_type = deviceGetBufferType,
    .get_host_buffer_type = null,
    .buffer_from_host_ptr = deviceBufferFromHostPtr,
    .supports_op = @ptrCast(&deviceSupportsOp),
    .supports_buft = deviceSupportsBuft,
    .offload_op = null,
    .event_new = null,
    .event_free = null,
    .event_synchronize = null,
};

// -----------------------------------------------------------------------------
// The registration

fn regGetName(reg: ggml.impl.ggml_c.ggml_backend_reg_t) callconv(.c) [*c]const u8 {
    _ = reg;
    return "CPU";
}

fn regGetDeviceCount(reg: ggml.impl.ggml_c.ggml_backend_reg_t) callconv(.c) usize {
    _ = reg;
    return 1;
}

var device_ctx: DeviceContext = .{};
var cpu_device: ggml.impl.ggml_c.struct_ggml_backend_device = undefined;
var device_ready = std.atomic.Value(bool).init(false);
var device_mutex: stdlib.sync.Mutex = .{};

fn ensureDevice(reg: ggml.impl.ggml_c.ggml_backend_reg_t) void {
    if (device_ready.load(.acquire)) return;
    device_mutex.lock();
    defer device_mutex.unlock();
    if (device_ready.load(.acquire)) return;
    device_ctx.init();
    cpu_device = .{ .iface = device_i, .reg = reg, .context = &device_ctx };
    device_ready.store(true, .release);
}

/// Ports `ggml_backend_cpu_reg_get_device` (ggml-cpu.cpp:527 @b1b770d90).
///
/// The C's two `static` locals are initialised on first call; `std.once`
/// is the same guarantee, and the device's `reg` back-pointer is captured
/// the same way.
fn regGetDevice(reg: ggml.impl.ggml_c.ggml_backend_reg_t, index: usize) callconv(.c) ggml.impl.ggml_c.ggml_backend_dev_t {
    ggml.impl.assert(index == 0, "index == 0");
    ensureDevice(reg);
    return &cpu_device;
}

/// Ports `ggml_backend_cpu_get_features` (ggml-cpu.cpp:542 @b1b770d90).
///
/// Only the entries whose predicate can be true on this target are kept:
/// the x86, RISC-V, POWER, s390x and WASM arms are all compiled out. SVE,
/// SME and MATMUL_INT8 stay because they are runtime checks on ARM even
/// though this build does not select them.
var features: [10]ggml.impl.ggml_c.ggml_backend_feature = undefined;
var features_n: usize = 0;
var features_ready = std.atomic.Value(bool).init(false);
var features_mutex: stdlib.sync.Mutex = .{};

fn pushFeature(name: [*c]const u8, value: [*c]const u8) void {
    features[features_n] = .{ .name = name, .value = value };
    features_n += 1;
}

fn ensureFeatures() void {
    if (features_ready.load(.acquire)) return;
    features_mutex.lock();
    defer features_mutex.unlock();
    if (features_ready.load(.acquire)) return;
    ggml.cpu.features.ggml_cpu_init();
    if (ggml.cpu.features.ggml_cpu_has_neon() != 0) pushFeature("NEON", "1");
    if (ggml.cpu.features.ggml_cpu_has_arm_fma() != 0) pushFeature("ARM_FMA", "1");
    if (ggml.cpu.features.ggml_cpu_has_fp16_va() != 0) pushFeature("FP16_VA", "1");
    if (ggml.cpu.features.ggml_cpu_has_matmul_int8() != 0) pushFeature("MATMUL_INT8", "1");
    if (ggml.cpu.features.ggml_cpu_has_sve() != 0) pushFeature("SVE", "1");
    if (ggml.cpu.features.ggml_cpu_has_dotprod() != 0) pushFeature("DOTPROD", "1");
    if (ggml.cpu.features.ggml_cpu_has_sme() != 0) pushFeature("SME", "1");
    if (ggml.cpu.features.ggml_cpu_has_llamafile() != 0) pushFeature("LLAMAFILE", "1");
    pushFeature("ACCELERATE", "1");
    pushFeature("REPACK", "1");
    pushFeature(null, null);
    features_ready.store(true, .release);
}

fn regGetFeatures(reg: ggml.impl.ggml_c.ggml_backend_reg_t) callconv(.c) [*c]ggml.impl.ggml_c.ggml_backend_feature {
    _ = reg;
    ensureFeatures();
    return &features;
}

/// Ports `ggml_backend_cpu_get_proc_address` (ggml-cpu.cpp:659 @b1b770d90).
fn regGetProcAddress(reg: ggml.impl.ggml_c.ggml_backend_reg_t, name: [*c]const u8) callconv(.c) ?*anyopaque {
    _ = reg;
    const n = std.mem.span(@as([*:0]const u8, @ptrCast(name)));
    const table = .{
        .{ "ggml_backend_set_n_threads", @as(?*anyopaque, @ptrCast(@constCast(&ggml_backend_cpu_set_n_threads))) },
        .{ "ggml_backend_dev_get_extra_bufts", @as(?*anyopaque, @ptrCast(@constCast(&deviceGetExtraBuffersType))) },
        .{ "ggml_backend_get_features", @as(?*anyopaque, @ptrCast(@constCast(&regGetFeatures))) },
        .{ "ggml_backend_set_abort_callback", @as(?*anyopaque, @ptrCast(@constCast(&ggml_backend_cpu_set_abort_callback))) },
        .{ "ggml_backend_cpu_numa_init", @as(?*anyopaque, @ptrCast(@constCast(&ggml.cpu.threading.ggml_numa_init))) },
        .{ "ggml_backend_cpu_is_numa", @as(?*anyopaque, @ptrCast(@constCast(&ggml.cpu.threading.ggml_is_numa))) },
        .{ "ggml_backend_cpu_set_use_ref", @as(?*anyopaque, @ptrCast(@constCast(&ggml_backend_cpu_set_use_ref))) },
        // threadpool - the C's own TODO: move to ggml-base
        .{ "ggml_threadpool_new", @as(?*anyopaque, @ptrCast(@constCast(&ggml.cpu.threading.ggml_threadpool_new))) },
        .{ "ggml_threadpool_free", @as(?*anyopaque, @ptrCast(@constCast(&ggml.cpu.threading.ggml_threadpool_free))) },
        .{ "ggml_backend_cpu_set_threadpool", @as(?*anyopaque, @ptrCast(@constCast(&ggml_backend_cpu_set_threadpool))) },
    };
    inline for (table) |entry| {
        if (std.mem.eql(u8, n, entry[0])) return entry[1];
    }
    return null;
}

/// Ports `ggml_backend_cpu_reg_i` (ggml-cpu.cpp:700 @b1b770d90).
const reg_i = ggml.impl.ggml_c.struct_ggml_backend_reg_i{
    .get_name = regGetName,
    .get_device_count = regGetDeviceCount,
    .get_device = regGetDevice,
    .get_proc_address = regGetProcAddress,
};

var cpu_reg = ggml.impl.ggml_c.struct_ggml_backend_reg{
    .api_version = ggml.impl.ggml_c.GGML_BACKEND_API_VERSION,
    .iface = reg_i,
    .context = null,
};

/// Ports `ggml_backend_cpu_reg` (ggml-cpu.cpp:707 @b1b770d90).
///
/// Return: the CPU registration, a process-lifetime singleton.
pub fn ggml_backend_cpu_reg() callconv(.c) ggml.impl.ggml_c.ggml_backend_reg_t {
    // init CPU feature detection
    ggml.cpu.features.ggml_cpu_init();
    return &cpu_reg;
}
