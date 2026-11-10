//! The Vulkan backend's all-reduce: the collective the meta backend asks
//! for through `ggml_backend_reg_get_proc_address` when a tensor split runs
//! over several Vulkan devices.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-comm.cpp` at
//! v0.9.0 (`cd959bae9`), new in that release. Each declaration below names
//! the C++ it replaces and the line it began at.
//!
//! # The contract
//!
//! The file exports no C symbol: its three entry points have C++ linkage
//! and hidden visibility, and `ggml_backend_vk_reg_get_proc_address`
//! (`backend.zig`) is their only caller, by address. So they are plain
//! `callconv(.c)` functions here, as the meta backend's
//! `ggml_backend_comm_*_t` types call them, and exported by nobody.
//!
//! # Two ways to wait on a peer
//!
//! Each round, every device copies its partial into its own slot of a host
//! buffer imported into every peer, signals, then waits for all its peers
//! before adding their slots into its tensor. A device cannot wait on
//! another device's semaphore, so a host thread carries the signal across:
//! `host_bridge` polls each device's `done` timeline and signals the peers'
//! `go`; `sync_fd`, on Mesa drivers with `VK_KHR_external_semaphore_fd`,
//! exports each device's binary semaphore as a sync file and imports it into
//! every peer, so the wait resolves in the kernel.
//!
//! # Not on Windows
//!
//! The C++ is `#if !defined(_WIN32)` throughout, and the registry returns
//! none of the three there; `supported` is that condition.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const common = m.common;
const vk = common.vk;
const d = &common.dispatch;
const allocator = common.allocator;
const BackendCtx = m.context.BackendCtx;
const Context = m.submit.Context;
const Tensor = c.ggml_tensor;

/// `#if !defined(_WIN32)` around the whole of `ggml-vulkan-comm.cpp`.
pub const supported = builtin.os.tag != .windows;

extern "c" fn posix_memalign(memptr: *?*anyopaque, alignment: usize, size: usize) c_int;
extern "c" fn atoi(s: [*:0]const u8) c_int;
extern "c" fn fcntl(fd: c_int, cmd: c_int, ...) c_int;
extern "c" fn pthread_setname_np(thread: std.c.pthread_t, name: [*:0]const u8) c_int;

fn oom() noreturn {
    impl.abort("ggml_vulkan: out of memory");
}

/// `std::chrono::steady_clock::now()`, in nanoseconds.
fn now() i64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i64, @intCast(ts.sec)) * std.time.ns_per_s + @as(i64, @intCast(ts.nsec));
}

/// Ports `ggml_vk_comm_f16_min_size` (ggml-vulkan-comm.cpp:10 @cd959bae9).
const f16_min_size: usize = 1024 * 1024;

/// Ports `ggml_vk_comm_needs_f32` (ggml-vulkan-comm.cpp:14 @cd959bae9):
/// follows the ops the meta backend may delay a reduction across.
fn needsF32(tensor: ?*const Tensor, depth: c_int) bool {
    const t = tensor orelse return false;
    if (depth > 64) return false;
    return switch (t.op) {
        c.GGML_OP_MUL_MAT, c.GGML_OP_MUL_MAT_ID => impl.getOpParamsI32(t, 0) == c.GGML_PREC_F32,
        c.GGML_OP_ADD => needsF32(t.src[0], depth + 1) or needsF32(t.src[1], depth + 1),
        c.GGML_OP_MUL, c.GGML_OP_ADD_ID => needsF32(t.src[0], depth + 1),
        c.GGML_OP_VIEW => needsF32(t.src[0], depth + 1),
        else => false,
    };
}

/// Ports `vk_comm_sync` (ggml-vulkan-comm.cpp:36 @cd959bae9).
const Sync = enum {
    // device i signals done[i], a host thread signals go[i] once all peers of i are done
    host_bridge,
    // a host thread exchanges SYNC_FD semaphores between peers, the wait resolves in the kernel
    sync_fd,
};

/// Ports `vk_comm_device` (ggml-vulkan-comm.cpp:41 @cd959bae9).
const CommDevice = struct {
    backend: c.ggml_backend_t = null,
    ctx: *BackendCtx = undefined,
    host_ptr: ?*anyopaque = null,
    host_buf: []c.ggml_backend_buffer_t = &.{}, // host_ptr imported into each peer

    sfd: struct {
        get_fd: vk.PFN_vkGetSemaphoreFdKHR = null,
        import_fd: vk.PFN_vkImportSemaphoreFdKHR = null,
        sem: vk.VkSemaphore = null,
        imp: []vk.VkSemaphore = &.{}, // [peer], pooled, reused only after ggml_vk_graph_cleanup
    } = .{},

    bridge: struct {
        done: vk.VkSemaphore = null,
        go: vk.VkSemaphore = null,
    } = .{},
};

/// Ports `ggml_backend_vk_comm_context` (ggml-vulkan-comm.cpp:60 @cd959bae9).
const CommCtx = struct {
    devs: []CommDevice = &.{},
    mode: Sync = .host_bridge,
    @"align": usize = 4096,
    cap: usize = 0,
    round: u64 = 0,
    tctx: ?*c.ggml_context = null,
    slot: *Tensor = undefined,
    disabled: bool = false,

    thread: ?std.Thread = null,
    spin: i64 = 200_000, // nanoseconds
    submitted: std.atomic.Value(u64) = .init(0),
    imported: std.atomic.Value(u64) = .init(0),
    sleeping: std.atomic.Value(bool) = .init(false),
    stop: std.atomic.Value(bool) = .init(false),
    mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    cv: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER,
};

/// Ports `ggml_vk_comm_free_buffers` (ggml-vulkan-comm.cpp:80 @cd959bae9).
fn freeBuffers(comm: *CommCtx) void {
    for (comm.devs) |*dv| {
        for (dv.host_buf) |*buf| {
            c.ggml_backend_buffer_free(buf.*);
            buf.* = null;
        }
        std.c.free(dv.host_ptr);
        dv.host_ptr = null;
    }
    comm.cap = 0;
}

/// Ports `ggml_backend_vk_comm_ensure` (ggml-vulkan-comm.cpp:92 @cd959bae9).
fn ensure(comm: *CommCtx, nbytes: usize) bool {
    if (comm.disabled) return false;
    if (nbytes <= comm.cap) return true;
    const fail = struct {
        fn f(cm: *CommCtx) bool {
            impl.logWarn("ggml_vulkan: shared host buffer allocation failed, using generic all-reduce\n", .{});
            cm.disabled = true;
            return false;
        }
    }.f;
    for (comm.devs) |dv| c.ggml_backend_synchronize(dv.backend);
    freeBuffers(comm);

    const n = comm.devs.len;
    const cap = impl.pad(nbytes, comm.@"align");
    for (0..n) |i| {
        const dv = &comm.devs[i];
        if (posix_memalign(&dv.host_ptr, comm.@"align", 2 * cap) != 0) {
            dv.host_ptr = null;
            return fail(comm);
        }
        for (0..n) |j| {
            dv.host_buf[j] = m.backend.deviceBufferFromHostPtr(c.ggml_backend_get_device(comm.devs[j].backend), dv.host_ptr, 2 * cap, 2 * cap);
            if (dv.host_buf[j] == null) return fail(comm);
        }
    }
    comm.cap = cap;
    return true;
}

/// Ports `ggml_vk_comm_set_slot` (ggml-vulkan-comm.cpp:129 @cd959bae9).
fn setSlot(slot: *Tensor, src: *const Tensor, ty: c.ggml_type, buf: c.ggml_backend_buffer_t, offs: usize) void {
    slot.type = ty;
    slot.nb[0] = c.ggml_type_size(ty);
    for (0..c.GGML_MAX_DIMS) |dim| {
        slot.ne[dim] = src.ne[dim];
        if (dim > 0) slot.nb[dim] = slot.nb[dim - 1] *% @as(usize, @bitCast(slot.ne[dim - 1]));
    }
    slot.buffer = buf;
    slot.data = @as([*]u8, @ptrCast(c.ggml_backend_buffer_get_base(buf))) + offs;
}

/// Ports `ggml_vk_comm_submit` (ggml-vulkan-comm.cpp:142 @cd959bae9).
fn submit(ctx: *BackendCtx, cx: *Context, signal: vk.VkSemaphore, value: u64) void {
    m.submit.ctxEnd(cx);
    m.submit.runMemcpys(cx.in_memcpys.items);
    m.submit.runMemsets(cx.memsets.items);
    cx.in_memcpys.clearRetainingCapacity();
    cx.memsets.clearRetainingCapacity();
    if (signal != null) {
        const seq = &cx.seqs.items[cx.seqs.items.len - 1];
        seq.items[seq.items.len - 1].signal_semaphores.append(allocator, .{ .s = signal, .value = value }) catch oom();
    }
    m.submit.submit(cx, null);
    ctx.submit_pending = true;
    ctx.compute_ctx = null;
}

/// Ports `ggml_vk_comm_notify_submitted` (ggml-vulkan-comm.cpp:160 @cd959bae9).
fn notifySubmitted(comm: *CommCtx, round: u64) void {
    comm.submitted.store(round, .seq_cst);
    if (comm.sleeping.load(.seq_cst)) {
        _ = std.c.pthread_mutex_lock(&comm.mutex);
        defer _ = std.c.pthread_mutex_unlock(&comm.mutex);
        _ = std.c.pthread_cond_signal(&comm.cv);
    }
}

/// Ports `ggml_vk_comm_wait_submitted` (ggml-vulkan-comm.cpp:169 @cd959bae9).
///
/// Return: false on stop.
fn waitSubmitted(comm: *CommCtx, r: u64) bool {
    const t0 = now();
    while (comm.submitted.load(.seq_cst) < r) {
        if (comm.stop.load(.seq_cst)) return false;
        if (now() - t0 < comm.spin) {
            common.yield();
            continue;
        }
        comm.sleeping.store(true, .seq_cst);
        if (comm.submitted.load(.seq_cst) < r) {
            _ = std.c.pthread_mutex_lock(&comm.mutex);
            while (!(comm.submitted.load(.seq_cst) >= r or comm.stop.load(.seq_cst))) {
                _ = std.c.pthread_cond_wait(&comm.cv, &comm.mutex);
            }
            _ = std.c.pthread_mutex_unlock(&comm.mutex);
        }
        comm.sleeping.store(false, .seq_cst);
    }
    return true;
}

/// `device.getSemaphoreCounterValue(sem)`, which throws on failure.
fn semaphoreCounterValue(device: vk.VkDevice, sem: vk.VkSemaphore) common.VkError!u64 {
    var value: u64 = 0;
    try common.check(d.vkGetSemaphoreCounterValue.?(device, sem, &value), "vk::Device::getSemaphoreCounterValue");
    return value;
}

/// `device.signalSemaphore({ sem, value })`, which throws on failure.
fn signalSemaphore(device: vk.VkDevice, sem: vk.VkSemaphore, value: u64) common.VkError!void {
    var si = common.structure(vk.VkSemaphoreSignalInfo, vk.VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO);
    si.semaphore = sem;
    si.value = value;
    try common.check(d.vkSignalSemaphore.?(device, &si), "vk::Device::signalSemaphore");
}

/// Ports `ggml_vk_comm_bridge_release` (ggml-vulkan-comm.cpp:190 @cd959bae9):
/// signals go[i] once all peers of i are done.
///
/// Return: false on stop; an error where a `vk::SystemError` is thrown.
fn bridgeRelease(comm: *CommCtx, r: u64, seen: []u64, sent: []bool) common.VkError!bool {
    const n = comm.devs.len;
    @memset(sent, false);
    var left = n;
    const t0 = now();
    while (true) {
        var pending = n;
        for (0..n) |i| {
            if (sent[i]) continue;
            var ready = true;
            var j: usize = 0;
            while (j < n and ready) : (j += 1) {
                if (j != i and seen[j] < r) {
                    const p = &comm.devs[j];
                    seen[j] = try semaphoreCounterValue(p.ctx.device.device, p.bridge.done);
                    ready = seen[j] >= r;
                    pending = if (ready) pending else j;
                }
            }
            if (ready) {
                const dv = &comm.devs[i];
                try signalSemaphore(dv.ctx.device.device, dv.bridge.go, r);
                sent[i] = true;
                left -= 1;
            }
        }
        if (left == 0) return true;
        if (now() - t0 < comm.spin) {
            common.yield();
            continue;
        }
        const p = &comm.devs[pending];
        var wi = common.structure(vk.VkSemaphoreWaitInfo, vk.VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO);
        wi.semaphoreCount = 1;
        wi.pSemaphores = &p.bridge.done;
        wi.pValues = &r;
        while (true) {
            // `waitSemaphores` returns eSuccess or eTimeout and throws on anything else.
            const result = d.vkWaitSemaphores.?(p.ctx.device.device, &wi, 100_000_000);
            if (result == vk.VK_SUCCESS) break;
            if (result != vk.VK_TIMEOUT) try common.check(result, "vk::Device::waitSemaphores");
            if (comm.stop.load(.seq_cst)) return false;
        }
    }
}

/// Ports `ggml_vk_comm_bridge_thread` (ggml-vulkan-comm.cpp:234 @cd959bae9).
fn bridgeThread(comm: *CommCtx) void {
    if (builtin.os.tag == .linux) _ = pthread_setname_np(std.c.pthread_self(), "ggml-vk-comm");
    const seen = allocator.alloc(u64, comm.devs.len) catch oom();
    defer allocator.free(seen);
    @memset(seen, 0);
    const sent = allocator.alloc(bool, comm.devs.len) catch oom();
    defer allocator.free(sent);
    var r: u64 = 1;
    while (true) : (r += 1) {
        const released = if (!waitSubmitted(comm, r)) false else bridgeRelease(comm, r, seen, sent) catch {
            // `e.what()` of a `vk::SystemError`: its message, then the code's.
            var what: [256]u8 = undefined;
            const text = std.fmt.bufPrintSentinel(&what, "{s}: {s}", .{ std.mem.span(common.last_error.what), common.resultString(common.last_error.result) }, 0) catch "vk::SystemError";
            c.ggml_abort(@src().file.ptr, @intCast(@src().line), "ggml_vulkan: all-reduce bridge failed: %s", text.ptr);
            unreachable;
        };
        if (!released) return;
    }
}

fn createTimelineSemaphore(device: vk.VkDevice) common.VkError!vk.VkSemaphore {
    var stci = common.structure(vk.VkSemaphoreTypeCreateInfo, vk.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO);
    stci.semaphoreType = vk.VK_SEMAPHORE_TYPE_TIMELINE;
    stci.initialValue = 0;
    var sci = common.structure(vk.VkSemaphoreCreateInfo, vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    sci.pNext = &stci;
    var s: vk.VkSemaphore = null;
    try common.check(d.vkCreateSemaphore.?(device, &sci, null, &s), "vk::Device::createSemaphore");
    return s;
}

fn spawnThread(comm: *CommCtx, comptime f: fn (*CommCtx) void) void {
    // A failed `std::thread` throws `std::system_error`, which nothing catches.
    comm.thread = std.Thread.spawn(.{}, f, .{comm}) catch impl.abort("ggml_vulkan: std::thread failed");
}

/// Ports `ggml_vk_comm_bridge_init` (ggml-vulkan-comm.cpp:251 @cd959bae9).
fn bridgeInit(comm: *CommCtx) common.VkError!void {
    for (comm.devs) |*dv| {
        dv.bridge.done = try createTimelineSemaphore(dv.ctx.device.device);
        dv.bridge.go = try createTimelineSemaphore(dv.ctx.device.device);
    }
    spawnThread(comm, bridgeThread);
}

/// Ports `ggml_vk_comm_bridge_free` (ggml-vulkan-comm.cpp:262 @cd959bae9).
fn bridgeFree(dv: *CommDevice) void {
    if (dv.bridge.done != null) d.vkDestroySemaphore.?(dv.ctx.device.device, dv.bridge.done, null);
    if (dv.bridge.go != null) d.vkDestroySemaphore.?(dv.ctx.device.device, dv.bridge.go, null);
}

/// Ports `ggml_vk_comm_barrier_to_host` (ggml-vulkan-comm.cpp:271 @cd959bae9).
fn barrierToHost(cx: *Context) void {
    var to_host = common.structure(vk.VkMemoryBarrier, vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER);
    to_host.srcAccessMask = vk.VK_ACCESS_TRANSFER_WRITE_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT;
    to_host.dstAccessMask = vk.VK_ACCESS_HOST_READ_BIT;
    d.vkCmdPipelineBarrier.?(cx.cmd(), cx.p.?.q.?.stage_flags, vk.VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0, null, 0, null);
}

/// Ports `ggml_vk_comm_barrier_from_host` (ggml-vulkan-comm.cpp:277 @cd959bae9):
/// the peers' slots were made available to the host before the wait
/// resolved, and the submit predates that.
fn barrierFromHost(cx: *Context) void {
    var from_host = common.structure(vk.VkMemoryBarrier, vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER);
    from_host.srcAccessMask = vk.VK_ACCESS_HOST_WRITE_BIT;
    from_host.dstAccessMask = vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_TRANSFER_READ_BIT;
    d.vkCmdPipelineBarrier.?(cx.cmd(), vk.VK_PIPELINE_STAGE_HOST_BIT, cx.p.?.q.?.stage_flags, 0, 1, &from_host, 0, null, 0, null);
}

/// Ports `ggml_vk_comm_bridge_signal` (ggml-vulkan-comm.cpp:282 @cd959bae9).
fn bridgeSignal(comm: *CommCtx, i: usize, cx: *Context, round: u64) void {
    const dv = &comm.devs[i];
    barrierToHost(cx);
    submit(dv.ctx, cx, dv.bridge.done, round);
}

/// Ports `ggml_vk_comm_bridge_wait` (ggml-vulkan-comm.cpp:288 @cd959bae9).
fn bridgeWait(comm: *CommCtx, i: usize, cx: *Context, round: u64) void {
    cx.s().wait_semaphores.append(allocator, .{ .s = comm.devs[i].bridge.go, .value = round }) catch oom();
    barrierFromHost(cx);
}

/// Ports `ggml_vk_comm_sfd_supported` (ggml-vulkan-comm.cpp:293 @cd959bae9).
fn sfdSupported(dv: *CommDevice) bool {
    const dev = dv.ctx.device;
    const mesa = dev.driver_id == vk.VK_DRIVER_ID_MESA_RADV or
        dev.driver_id == vk.VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA or
        dev.driver_id == vk.VK_DRIVER_ID_MESA_NVK;
    if (!mesa or !dev.external_semaphore) return false;
    dv.sfd.get_fd = @ptrCast(d.vkGetDeviceProcAddr.?(dev.device, "vkGetSemaphoreFdKHR"));
    dv.sfd.import_fd = @ptrCast(d.vkGetDeviceProcAddr.?(dev.device, "vkImportSemaphoreFdKHR"));
    if (dv.sfd.get_fd == null or dv.sfd.import_fd == null) return false;
    var esi = common.structure(vk.VkPhysicalDeviceExternalSemaphoreInfo, vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO);
    esi.handleType = vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    var esp = common.structure(vk.VkExternalSemaphoreProperties, vk.VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES);
    d.vkGetPhysicalDeviceExternalSemaphoreProperties.?(dev.physical_device, &esi, &esp);
    const need: vk.VkExternalSemaphoreFeatureFlags = vk.VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT | vk.VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
    return (esp.externalSemaphoreFeatures & need) == need and
        (esp.compatibleHandleTypes & vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT) != 0;
}

/// Ports `ggml_vk_comm_sfd_enable_submit_thread` (ggml-vulkan-comm.cpp:315 @cd959bae9):
/// Mesa moves a queue to its submit thread on the first wait-before-signal,
/// taking the CS ioctls off this thread.
fn sfdEnableSubmitThread(dev: *m.device.Device) common.VkError!void {
    const sem = try createTimelineSemaphore(dev.device);
    var fci = common.structure(vk.VkFenceCreateInfo, vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    var fence: vk.VkFence = null;
    try common.check(d.vkCreateFence.?(dev.device, &fci, null, &fence), "vk::Device::createFence");
    const one: u64 = 1;
    var tsi = common.structure(vk.VkTimelineSemaphoreSubmitInfo, vk.VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO);
    tsi.waitSemaphoreValueCount = 1;
    tsi.pWaitSemaphoreValues = &one;
    const stage: vk.VkPipelineStageFlags = vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    var si = common.structure(vk.VkSubmitInfo, vk.VK_STRUCTURE_TYPE_SUBMIT_INFO);
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &sem;
    si.pWaitDstStageMask = &stage;
    si.pNext = &tsi;
    try m.submit.queueSubmitChecked(dev.compute_queue.?.handle, &.{si}, fence);
    try signalSemaphore(dev.device, sem, 1);
    try m.submit.vkCheckChecked(d.vkWaitForFences.?(dev.device, 1, &fence, vk.VK_TRUE, std.math.maxInt(u64)), "ggml_vk_comm_sfd_enable_submit_thread", dev);
    d.vkDestroyFence.?(dev.device, fence, null);
    d.vkDestroySemaphore.?(dev.device, sem, null);
}

/// Ports `ggml_vk_comm_sfd_export` (ggml-vulkan-comm.cpp:334 @cd959bae9).
/// Requires a submitted signal on sem; the export resets it.
fn sfdExport(dv: *CommDevice) c_int {
    var fd: c_int = -1;
    var gi = common.structure(vk.VkSemaphoreGetFdInfoKHR, vk.VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR);
    gi.semaphore = dv.sfd.sem;
    gi.handleType = vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    const result = dv.sfd.get_fd.?(dv.ctx.device.device, &gi, &fd);
    if (result != vk.VK_SUCCESS) {
        c.ggml_abort(@src().file.ptr, @intCast(@src().line), "ggml_vulkan: vkGetSemaphoreFdKHR failed: %d", @as(c_int, result));
        unreachable;
    }
    return fd;
}

/// Ports `ggml_vk_comm_sfd_import` (ggml-vulkan-comm.cpp:348 @cd959bae9).
/// An fd of -1 imports an already signalled payload.
fn sfdImport(dv: *CommDevice, sem: vk.VkSemaphore, fd: c_int) void {
    var isi = common.structure(vk.VkImportSemaphoreFdInfoKHR, vk.VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR);
    isi.semaphore = sem;
    isi.flags = vk.VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
    isi.handleType = vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    isi.fd = fd;
    const result = dv.sfd.import_fd.?(dv.ctx.device.device, &isi);
    if (result != vk.VK_SUCCESS) {
        c.ggml_abort(@src().file.ptr, @intCast(@src().line), "ggml_vulkan: vkImportSemaphoreFdKHR failed: %d", @as(c_int, result));
        unreachable;
    }
}

/// Ports `ggml_vk_comm_sfd_exchange` (ggml-vulkan-comm.cpp:361 @cd959bae9).
fn sfdExchange(comm: *CommCtx, r: u64) void {
    const n = comm.devs.len;
    for (0..n) |j| {
        const fd = sfdExport(&comm.devs[j]);
        const last = if (j == n - 1) n - 2 else n - 1;
        for (0..n) |i| {
            if (i == j) continue;
            const imp_fd = if (fd < 0 or i == last) fd else fcntl(fd, std.c.F.DUPFD_CLOEXEC, @as(c_int, 0));
            impl.assert(imp_fd >= 0 or fd < 0, "imp_fd >= 0 || fd < 0");
            sfdImport(&comm.devs[i], comm.devs[i].sfd.imp[j], imp_fd);
        }
    }
    comm.imported.store(r, .release);
}

/// Ports `ggml_vk_comm_sfd_thread` (ggml-vulkan-comm.cpp:378 @cd959bae9).
fn sfdThread(comm: *CommCtx) void {
    if (builtin.os.tag == .linux) _ = pthread_setname_np(std.c.pthread_self(), "ggml-vk-comm");
    var r: u64 = 1;
    while (waitSubmitted(comm, r)) : (r += 1) sfdExchange(comm, r);
}

/// Ports `ggml_vk_comm_sfd_init` (ggml-vulkan-comm.cpp:387 @cd959bae9).
fn sfdInit(comm: *CommCtx) common.VkError!void {
    var esci = common.structure(vk.VkExportSemaphoreCreateInfo, vk.VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO);
    esci.handleTypes = vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    var sci = common.structure(vk.VkSemaphoreCreateInfo, vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    sci.pNext = &esci;
    for (comm.devs) |*dv| {
        try common.check(d.vkCreateSemaphore.?(dv.ctx.device.device, &sci, null, &dv.sfd.sem), "vk::Device::createSemaphore");
        dv.sfd.imp = allocator.alloc(vk.VkSemaphore, comm.devs.len) catch oom();
        @memset(dv.sfd.imp, null);
        try sfdEnableSubmitThread(dv.ctx.device);
    }
    spawnThread(comm, sfdThread);
}

/// Ports `ggml_vk_comm_sfd_free` (ggml-vulkan-comm.cpp:399 @cd959bae9).
fn sfdFree(dv: *CommDevice) void {
    if (dv.sfd.sem != null) d.vkDestroySemaphore.?(dv.ctx.device.device, dv.sfd.sem, null);
}

/// Ports `ggml_vk_comm_sfd_signal` (ggml-vulkan-comm.cpp:405 @cd959bae9).
fn sfdSignal(comm: *CommCtx, i: usize, cx: *Context) void {
    const dv = &comm.devs[i];
    barrierToHost(cx);
    submit(dv.ctx, cx, dv.sfd.sem, 0);
    for (0..comm.devs.len) |j| {
        dv.sfd.imp[j] = if (j == i) null else m.context.createBinarySemaphore(dv.ctx);
    }
}

/// Ports `ggml_vk_comm_sfd_wait` (ggml-vulkan-comm.cpp:414 @cd959bae9).
fn sfdWait(comm: *CommCtx, i: usize, cx: *Context, round: u64) void {
    for (0..comm.devs.len) |j| {
        if (j != i) cx.s().wait_semaphores.append(allocator, .{ .s = comm.devs[i].sfd.imp[j], .value = 0 }) catch oom();
    }
    cx.s().ready = &comm.imported;
    cx.s().ready_value = round;
    barrierFromHost(cx);
}

/// Ports `ggml_backend_vk_comm_free` (ggml-vulkan-comm.cpp:425 @cd959bae9).
pub fn commFree(comm_ctx: ?*anyopaque) callconv(.c) void {
    const comm: *CommCtx = @ptrCast(@alignCast(comm_ctx.?));
    // pending reductions need the thread
    for (comm.devs) |dv| c.ggml_backend_synchronize(dv.backend);
    if (comm.thread) |t| {
        {
            _ = std.c.pthread_mutex_lock(&comm.mutex);
            defer _ = std.c.pthread_mutex_unlock(&comm.mutex);
            comm.stop.store(true, .seq_cst);
        }
        _ = std.c.pthread_cond_broadcast(&comm.cv);
        t.join();
        comm.thread = null;
    }
    for (comm.devs) |*dv| {
        sfdFree(dv);
        bridgeFree(dv);
    }
    freeBuffers(comm);
    if (comm.tctx) |tctx| c.ggml_free(tctx);
    for (comm.devs) |*dv| {
        allocator.free(dv.host_buf);
        allocator.free(dv.sfd.imp);
    }
    allocator.free(comm.devs);
    allocator.destroy(comm);
}

/// Ports `ggml_backend_vk_comm_init` (ggml-vulkan-comm.cpp:450 @cd959bae9).
pub fn commInit(backends: [*c]c.ggml_backend_t, n_backends: usize) callconv(.c) ?*anyopaque {
    if (m.instance.instance.allreduce_disabled or n_backends < 2) return null;
    for (0..n_backends) |i| {
        if (!m.backend.isVk(backends[i])) return null;
    }
    const comm = allocator.create(CommCtx) catch oom();
    comm.* = .{};
    comm.devs = allocator.alloc(CommDevice, n_backends) catch oom();
    for (0..n_backends) |i| {
        const dv = &comm.devs[i];
        dv.* = .{};
        dv.backend = backends[i];
        dv.ctx = @ptrCast(@alignCast(backends[i].*.context));
        dv.host_buf = allocator.alloc(c.ggml_backend_buffer_t, n_backends) catch oom();
        @memset(dv.host_buf, null);
    }

    for (0..n_backends) |i| {
        const dev = comm.devs[i].ctx.device;
        comm.@"align" = @max(comm.@"align", @as(usize, @intCast(dev.min_imported_host_pointer_alignment)));
        if (!dev.external_memory_host) {
            impl.logWarn("ggml_vulkan: device %zu: VK_EXT_external_memory_host missing, using generic all-reduce\n", .{i});
            commFree(comm);
            return null;
        }
        if (dev.serialize_submissions) {
            impl.logWarn("ggml_vulkan: device %zu: GGML_VK_SERIALIZE_SUBMISSIONS set, using generic all-reduce\n", .{i});
            commFree(comm);
            return null;
        }
    }

    // `std::all_of`, which stops at the first device that answers no.
    const all_sfd = for (comm.devs) |*dv| {
        if (!sfdSupported(dv)) break false;
    } else true;
    if (all_sfd) comm.mode = .sync_fd;

    if (std.c.getenv("GGML_VK_ALLREDUCE_SPIN_US")) |spin| {
        comm.spin = @as(i64, @max(0, atoi(spin))) * std.time.ns_per_us;
    }
    const ip: c.ggml_init_params = .{ .mem_size = c.ggml_tensor_overhead(), .mem_buffer = null, .no_alloc = true };
    comm.tctx = c.ggml_init(ip);
    comm.slot = c.ggml_new_tensor_1d(comm.tctx, c.GGML_TYPE_F32, 1);

    const inited = if (comm.mode == .sync_fd) sfdInit(comm) else bridgeInit(comm);
    inited catch {
        // `e.what()` of the `vk::SystemError`: its message, then the code's.
        impl.logWarn("ggml_vulkan: semaphore initialization failed: %s: %s, using generic all-reduce\n", .{ common.last_error.what, common.resultString(common.last_error.result).ptr });
        commFree(comm);
        return null;
    };
    return comm;
}

/// Ports `ggml_backend_vk_comm_allreduce_tensor` (ggml-vulkan-comm.cpp:510 @cd959bae9).
///
/// A: partial to own host slot (round & 1), signal; B: wait on peers' A,
/// add their slots; all A precede any B. Slot reuse: A(k) follows own
/// B(k-1), which waited on peers' A(k-1), submitted after their B(k-2), the
/// last reader.
pub fn commAllreduceTensor(comm_ctx: ?*anyopaque, tensors: [*c][*c]Tensor) callconv(.c) bool {
    const comm: *CommCtx = @ptrCast(@alignCast(comm_ctx.?));
    const n = comm.devs.len;

    const ne = c.ggml_nelements(tensors[0]);
    for (0..n) |i| {
        const t = impl.one(Tensor, tensors[i]);
        if (t.type != c.GGML_TYPE_F32 or c.ggml_nelements(t) != ne or !c.ggml_is_contiguous(t) or !m.buffer.bufferIsVk(t.buffer)) {
            return false;
        }
        const ctx = comm.devs[i].ctx;
        if (ctx.compute_ctx) |cc| if (cc.out_memcpys.items.len != 0) return false;
    }
    if (ne == 0) return true;
    const nbytes = c.ggml_nbytes(tensors[0]);
    if (!ensure(comm, nbytes)) return false;

    const f32_wire = nbytes < f16_min_size or needsF32(tensors[0], 0);
    const wire: c.ggml_type = if (f32_wire) c.GGML_TYPE_F32 else c.GGML_TYPE_F16;
    comm.round += 1;
    const round = comm.round;
    const slot_offs = (round & 1) * comm.cap;
    const slot_size = @as(usize, @intCast(ne)) * c.ggml_type_size(wire);

    for (0..n) |i| {
        const dv = &comm.devs[i];
        const ctx = dv.ctx;
        const t = impl.one(Tensor, tensors[i]);
        const tbuf = m.buffer.tensorBuffer(t);
        const hbuf = m.buffer.devBuffer(dv.host_buf[i]);
        const toffs = m.buffer.tensorOffset(t) + t.view_offs;

        const cx = m.context.getComputeCtx(ctx);
        m.context.syncBuffers(ctx, cx);
        if (t.flags & c.GGML_TENSOR_FLAG_COMPUTE == 0) {
            d.vkCmdFillBuffer.?(cx.cmd(), hbuf.buffer, slot_offs, impl.pad(slot_size, 4), 0);
            d.vkCmdFillBuffer.?(cx.cmd(), tbuf.buffer, toffs, nbytes, 0);
        } else if (wire == c.GGML_TYPE_F32) {
            m.submit.bufferCopyAsync(cx, hbuf, slot_offs, tbuf, toffs, nbytes);
        } else {
            setSlot(comm.slot, t, wire, dv.host_buf[i], slot_offs);
            m.ops_basic.cpy(ctx, cx, t, comm.slot);
        }
        if (comm.mode == .sync_fd) {
            sfdSignal(comm, i, cx);
        } else {
            bridgeSignal(comm, i, cx, round);
        }
    }
    notifySubmitted(comm, round);

    for (0..n) |i| {
        const ctx = comm.devs[i].ctx;
        const cx = m.context.getComputeCtx(ctx);
        ctx.do_add_rms_partials = false;
        if (comm.mode == .sync_fd) {
            sfdWait(comm, i, cx, round);
        } else {
            bridgeWait(comm, i, cx, round);
        }
        m.context.syncBuffers(ctx, cx);
        const t = impl.one(Tensor, tensors[i]);
        for (0..n) |j| {
            if (j == i) continue;
            setSlot(comm.slot, t, wire, comm.devs[j].host_buf[i], slot_offs);
            m.ops_basic.add(ctx, cx, t, comm.slot, t);
            m.context.syncBuffers(ctx, cx);
        }
        // small reductions go out with the next graph's first submit
        if (wire != c.GGML_TYPE_F32 or m.instance.loggers.perf_logger_enabled) {
            submit(ctx, cx, null, 0);
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the wire precision follows the ops a reduction may be delayed across" {
    var mm = std.mem.zeroes(Tensor);
    mm.op = c.GGML_OP_MUL_MAT;
    var add = std.mem.zeroes(Tensor);
    add.op = c.GGML_OP_ADD;
    add.src[1] = &mm;
    var view = std.mem.zeroes(Tensor);
    view.op = c.GGML_OP_VIEW;
    view.src[0] = &add;
    try std.testing.expect(!needsF32(&view, 0));
    mm.op_params[0] = c.GGML_PREC_F32;
    try std.testing.expect(needsF32(&view, 0));
    // MUL follows only its first source.
    var mul = std.mem.zeroes(Tensor);
    mul.op = c.GGML_OP_MUL;
    mul.src[1] = &mm;
    try std.testing.expect(!needsF32(&mul, 0));
    try std.testing.expect(!needsF32(null, 0));
    // Past depth 64 the walk gives up.
    try std.testing.expect(!needsF32(&view, 64));
}
