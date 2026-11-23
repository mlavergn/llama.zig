//! The Vulkan backend's all-reduce: the collective the meta backend asks
//! for through `ggml_backend_reg_get_proc_address` when a tensor split runs
//! over several Vulkan devices.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-comm.cpp` at
//! v1.0.0 (`b1b770d90`); new at v0.9.0. Each declaration below names
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
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");
const builtin = @import("builtin");

/// `#if !defined(_WIN32)` around the whole of `ggml-vulkan-comm.cpp`.
pub const supported = builtin.os.tag != .windows;

fn oom() noreturn {
    ggml.impl.abort("ggml_vulkan: out of memory");
}

/// Ports `ggml_vk_comm_f16_min_size` (ggml-vulkan-comm.cpp:10 @b1b770d90).
const f16_min_size: usize = 1024 * 1024;

/// Ports `ggml_vk_comm_needs_f32` (ggml-vulkan-comm.cpp:14 @b1b770d90):
/// follows the ops the meta backend may delay a reduction across.
fn needsF32(tensor: ?*const ggml.impl.ggml_c.ggml_tensor, depth: c_int) bool {
    const t = tensor orelse return false;
    if (depth > 64) return false;
    return switch (t.op) {
        ggml.impl.ggml_c.GGML_OP_MUL_MAT, ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID => ggml.ops.ggml_prec_get_acc(t) == ggml.impl.ggml_c.GGML_PREC_BF16 or ggml.ops.ggml_prec_get_acc(t) == ggml.impl.ggml_c.GGML_PREC_F32,
        ggml.impl.ggml_c.GGML_OP_ADD => needsF32(t.src[0], depth + 1) or needsF32(t.src[1], depth + 1),
        ggml.impl.ggml_c.GGML_OP_MUL, ggml.impl.ggml_c.GGML_OP_ADD_ID => needsF32(t.src[0], depth + 1),
        ggml.impl.ggml_c.GGML_OP_VIEW => needsF32(t.src[0], depth + 1),
        else => false,
    };
}

/// Ports `vk_comm_sync` (ggml-vulkan-comm.cpp:36 @b1b770d90).
const Sync = enum {
    // device i signals done[i], a host thread signals go[i] once all peers of i are done
    host_bridge,
    // a host thread exchanges SYNC_FD semaphores between peers, the wait resolves in the kernel
    sync_fd,
};

/// Ports `vk_comm_device` (ggml-vulkan-comm.cpp:41 @b1b770d90).
const CommDevice = struct {
    backend: ggml.impl.ggml_c.ggml_backend_t = null,
    ctx: *ggml.vulkan.context.BackendCtx = undefined,
    host_ptr: ?*anyopaque = null,
    host_buf: []ggml.impl.ggml_c.ggml_backend_buffer_t = &.{}, // host_ptr imported into each peer

    sfd: struct {
        get_fd: ggml.vulkan.common.vulkan_c.PFN_vkGetSemaphoreFdKHR = null,
        import_fd: ggml.vulkan.common.vulkan_c.PFN_vkImportSemaphoreFdKHR = null,
        sem: ggml.vulkan.common.vulkan_c.VkSemaphore = null,
        imp: []ggml.vulkan.common.vulkan_c.VkSemaphore = &.{}, // [peer], pooled, reused only after ggml_vk_graph_cleanup
    } = .{},

    bridge: struct {
        done: ggml.vulkan.common.vulkan_c.VkSemaphore = null,
        go: ggml.vulkan.common.vulkan_c.VkSemaphore = null,
    } = .{},
};

/// Ports `ggml_backend_vk_comm_context` (ggml-vulkan-comm.cpp:60 @b1b770d90).
const CommCtx = struct {
    devs: []CommDevice = &.{},
    mode: Sync = .host_bridge,
    @"align": usize = 4096,
    cap: usize = 0,
    round: u64 = 0,
    tctx: ?*ggml.context.Context = null,
    slot: *ggml.impl.ggml_c.ggml_tensor = undefined,
    disabled: bool = false,

    thread: ?std.Thread = null,
    spin: i64 = 200_000, // nanoseconds
    submitted: std.atomic.Value(u64) = .init(0),
    imported: std.atomic.Value(u64) = .init(0),
    sleeping: std.atomic.Value(bool) = .init(false),
    stop: std.atomic.Value(bool) = .init(false),
    mutex: stdlib.sync.Mutex = .{},
    cv: stdlib.sync.Condition = .{},
};

/// Ports `ggml_vk_comm_free_buffers` (ggml-vulkan-comm.cpp:80 @b1b770d90).
fn freeBuffers(comm: *CommCtx) void {
    for (comm.devs) |*dv| {
        for (dv.host_buf) |*buf| {
            ggml.backend.ggml_backend_buffer_free(buf.*);
            buf.* = null;
        }
        std.c.free(dv.host_ptr);
        dv.host_ptr = null;
    }
    comm.cap = 0;
}

/// Ports `ggml_backend_vk_comm_ensure` (ggml-vulkan-comm.cpp:92 @b1b770d90).
fn ensure(comm: *CommCtx, nbytes: usize) bool {
    if (comm.disabled) return false;
    if (nbytes <= comm.cap) return true;
    const fail = struct {
        fn f(cm: *CommCtx) bool {
            ggml.impl.logWarn("ggml_vulkan: shared host buffer allocation failed, using generic all-reduce\n", .{});
            cm.disabled = true;
            return false;
        }
    }.f;
    for (comm.devs) |dv| ggml.backend.ggml_backend_synchronize(dv.backend);
    freeBuffers(comm);

    const n = comm.devs.len;
    const cap = stdlib.num.pad(usize, nbytes, comm.@"align");
    for (0..n) |i| {
        const dv = &comm.devs[i];
        if (std.c.posix_memalign(&dv.host_ptr, comm.@"align", 2 * cap) != 0) {
            dv.host_ptr = null;
            return fail(comm);
        }
        for (0..n) |j| {
            dv.host_buf[j] = ggml.vulkan.backend.deviceBufferFromHostPtr(ggml.backend.ggml_backend_get_device(comm.devs[j].backend), dv.host_ptr, 2 * cap, 2 * cap);
            if (dv.host_buf[j] == null) return fail(comm);
        }
    }
    comm.cap = cap;
    return true;
}

/// Ports `ggml_vk_comm_set_slot` (ggml-vulkan-comm.cpp:129 @b1b770d90).
fn setSlot(slot: *ggml.impl.ggml_c.ggml_tensor, src: *const ggml.impl.ggml_c.ggml_tensor, ty: ggml.impl.ggml_c.ggml_type, buf: ggml.impl.ggml_c.ggml_backend_buffer_t, offs: usize) void {
    slot.type = ty;
    slot.nb[0] = ggml.types.ggml_type_size(ty);
    for (0..ggml.impl.ggml_c.GGML_MAX_DIMS) |dim| {
        slot.ne[dim] = src.ne[dim];
        if (dim > 0) slot.nb[dim] = slot.nb[dim - 1] *% @as(usize, @bitCast(slot.ne[dim - 1]));
    }
    slot.buffer = buf;
    slot.data = @as([*]u8, @ptrCast(ggml.backend.ggml_backend_buffer_get_base(buf))) + offs;
}

/// Ports `ggml_vk_comm_submit` (ggml-vulkan-comm.cpp:142 @b1b770d90).
fn submit(ctx: *ggml.vulkan.context.BackendCtx, cx: *ggml.vulkan.submit.Context, signal: ggml.vulkan.common.vulkan_c.VkSemaphore, value: u64) void {
    ggml.vulkan.submit.ctxEnd(cx);
    ggml.vulkan.submit.runMemcpys(cx.in_memcpys.items);
    ggml.vulkan.submit.runMemsets(cx.memsets.items);
    cx.in_memcpys.clearRetainingCapacity();
    cx.memsets.clearRetainingCapacity();
    if (signal != null) {
        const seq = &cx.seqs.items[cx.seqs.items.len - 1];
        seq.items[seq.items.len - 1].signal_semaphores.append(std.heap.c_allocator, .{ .s = signal, .value = value }) catch oom();
    }
    ggml.vulkan.submit.submit(cx, null);
    ctx.submit_pending = true;
    ctx.compute_ctx = null;
}

/// Ports `ggml_vk_comm_notify_submitted` (ggml-vulkan-comm.cpp:160 @b1b770d90).
fn notifySubmitted(comm: *CommCtx, round: u64) void {
    comm.submitted.store(round, .seq_cst);
    if (comm.sleeping.load(.seq_cst)) {
        comm.mutex.lock();
        defer comm.mutex.unlock();
        comm.cv.signal();
    }
}

/// Ports `ggml_vk_comm_wait_submitted` (ggml-vulkan-comm.cpp:169 @b1b770d90).
///
/// Return: false on stop.
fn waitSubmitted(comm: *CommCtx, r: u64) bool {
    const t0 = stdlib.clock.steadyNs();
    while (comm.submitted.load(.seq_cst) < r) {
        if (comm.stop.load(.seq_cst)) return false;
        if (stdlib.clock.steadyNs() - t0 < comm.spin) {
            ggml.vulkan.common.yield();
            continue;
        }
        comm.sleeping.store(true, .seq_cst);
        if (comm.submitted.load(.seq_cst) < r) {
            comm.mutex.lock();
            while (!(comm.submitted.load(.seq_cst) >= r or comm.stop.load(.seq_cst))) {
                comm.cv.wait(&comm.mutex);
            }
            comm.mutex.unlock();
        }
        comm.sleeping.store(false, .seq_cst);
    }
    return true;
}

/// `device.getSemaphoreCounterValue(sem)`, which throws on failure.
fn semaphoreCounterValue(device: ggml.vulkan.common.vulkan_c.VkDevice, sem: ggml.vulkan.common.vulkan_c.VkSemaphore) ggml.vulkan.common.VkError!u64 {
    var value: u64 = 0;
    try ggml.vulkan.common.check(ggml.vulkan.common.dispatch.vkGetSemaphoreCounterValue.?(device, sem, &value), "vk::Device::getSemaphoreCounterValue");
    return value;
}

/// `device.signalSemaphore({ sem, value })`, which throws on failure.
fn signalSemaphore(device: ggml.vulkan.common.vulkan_c.VkDevice, sem: ggml.vulkan.common.vulkan_c.VkSemaphore, value: u64) ggml.vulkan.common.VkError!void {
    var si = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkSemaphoreSignalInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO);
    si.semaphore = sem;
    si.value = value;
    try ggml.vulkan.common.check(ggml.vulkan.common.dispatch.vkSignalSemaphore.?(device, &si), "vk::Device::signalSemaphore");
}

/// Ports `ggml_vk_comm_bridge_release` (ggml-vulkan-comm.cpp:190 @b1b770d90):
/// signals go[i] once all peers of i are done.
///
/// Return: false on stop; an error where a `vk::SystemError` is thrown.
fn bridgeRelease(comm: *CommCtx, r: u64, seen: []u64, sent: []bool) ggml.vulkan.common.VkError!bool {
    const n = comm.devs.len;
    @memset(sent, false);
    var left = n;
    const t0 = stdlib.clock.steadyNs();
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
        if (stdlib.clock.steadyNs() - t0 < comm.spin) {
            ggml.vulkan.common.yield();
            continue;
        }
        const p = &comm.devs[pending];
        var wi = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkSemaphoreWaitInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO);
        wi.semaphoreCount = 1;
        wi.pSemaphores = &p.bridge.done;
        wi.pValues = &r;
        while (true) {
            // `waitSemaphores` returns eSuccess or eTimeout and throws on anything else.
            const result = ggml.vulkan.common.dispatch.vkWaitSemaphores.?(p.ctx.device.device, &wi, 100_000_000);
            if (result == ggml.vulkan.common.vulkan_c.VK_SUCCESS) break;
            if (result != ggml.vulkan.common.vulkan_c.VK_TIMEOUT) try ggml.vulkan.common.check(result, "vk::Device::waitSemaphores");
            if (comm.stop.load(.seq_cst)) return false;
        }
    }
}

/// Ports `ggml_vk_comm_bridge_thread` (ggml-vulkan-comm.cpp:234 @b1b770d90).
fn bridgeThread(comm: *CommCtx) void {
    if (builtin.os.tag == .linux) _ = std.c.pthread_setname_np(std.c.pthread_self(), "ggml-vk-comm");
    const seen = std.heap.c_allocator.alloc(u64, comm.devs.len) catch oom();
    defer std.heap.c_allocator.free(seen);
    @memset(seen, 0);
    const sent = std.heap.c_allocator.alloc(bool, comm.devs.len) catch oom();
    defer std.heap.c_allocator.free(sent);
    var r: u64 = 1;
    while (true) : (r += 1) {
        const released = if (!waitSubmitted(comm, r)) false else bridgeRelease(comm, r, seen, sent) catch {
            // `e.what()` of a `vk::SystemError`: its message, then the code's.
            var what: [256]u8 = undefined;
            const text = std.fmt.bufPrintSentinel(&what, "{s}: {s}", .{ std.mem.span(ggml.vulkan.common.last_error.what), ggml.vulkan.common.resultString(ggml.vulkan.common.last_error.result) }, 0) catch "vk::SystemError";
            ggml.impl.ggml_c.ggml_abort(@src().file.ptr, @intCast(@src().line), "ggml_vulkan: all-reduce bridge failed: %s", text.ptr);
            unreachable;
        };
        if (!released) return;
    }
}

fn createTimelineSemaphore(device: ggml.vulkan.common.vulkan_c.VkDevice) ggml.vulkan.common.VkError!ggml.vulkan.common.vulkan_c.VkSemaphore {
    var stci = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkSemaphoreTypeCreateInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO);
    stci.semaphoreType = ggml.vulkan.common.vulkan_c.VK_SEMAPHORE_TYPE_TIMELINE;
    stci.initialValue = 0;
    var sci = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkSemaphoreCreateInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    sci.pNext = &stci;
    var s: ggml.vulkan.common.vulkan_c.VkSemaphore = null;
    try ggml.vulkan.common.check(ggml.vulkan.common.dispatch.vkCreateSemaphore.?(device, &sci, null, &s), "vk::Device::createSemaphore");
    return s;
}

fn spawnThread(comm: *CommCtx, comptime f: fn (*CommCtx) void) void {
    // A failed `std::thread` throws `std::system_error`, which nothing catches.
    comm.thread = std.Thread.spawn(.{}, f, .{comm}) catch ggml.impl.abort("ggml_vulkan: std::thread failed");
}

/// Ports `ggml_vk_comm_bridge_init` (ggml-vulkan-comm.cpp:251 @b1b770d90).
fn bridgeInit(comm: *CommCtx) ggml.vulkan.common.VkError!void {
    for (comm.devs) |*dv| {
        dv.bridge.done = try createTimelineSemaphore(dv.ctx.device.device);
        dv.bridge.go = try createTimelineSemaphore(dv.ctx.device.device);
    }
    spawnThread(comm, bridgeThread);
}

/// Ports `ggml_vk_comm_bridge_free` (ggml-vulkan-comm.cpp:262 @b1b770d90).
fn bridgeFree(dv: *CommDevice) void {
    if (dv.bridge.done != null) ggml.vulkan.common.dispatch.vkDestroySemaphore.?(dv.ctx.device.device, dv.bridge.done, null);
    if (dv.bridge.go != null) ggml.vulkan.common.dispatch.vkDestroySemaphore.?(dv.ctx.device.device, dv.bridge.go, null);
}

/// Ports `ggml_vk_comm_barrier_to_host` (ggml-vulkan-comm.cpp:271 @b1b770d90).
fn barrierToHost(cx: *ggml.vulkan.submit.Context) void {
    var to_host = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkMemoryBarrier, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_MEMORY_BARRIER);
    to_host.srcAccessMask = ggml.vulkan.common.vulkan_c.VK_ACCESS_TRANSFER_WRITE_BIT | ggml.vulkan.common.vulkan_c.VK_ACCESS_SHADER_WRITE_BIT;
    to_host.dstAccessMask = ggml.vulkan.common.vulkan_c.VK_ACCESS_HOST_READ_BIT;
    ggml.vulkan.common.dispatch.vkCmdPipelineBarrier.?(cx.cmd(), cx.p.?.q.?.stage_flags, ggml.vulkan.common.vulkan_c.VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0, null, 0, null);
}

/// Ports `ggml_vk_comm_barrier_from_host` (ggml-vulkan-comm.cpp:277 @b1b770d90):
/// the peers' slots were made available to the host before the wait
/// resolved, and the submit predates that.
fn barrierFromHost(cx: *ggml.vulkan.submit.Context) void {
    var from_host = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkMemoryBarrier, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_MEMORY_BARRIER);
    from_host.srcAccessMask = ggml.vulkan.common.vulkan_c.VK_ACCESS_HOST_WRITE_BIT;
    from_host.dstAccessMask = ggml.vulkan.common.vulkan_c.VK_ACCESS_SHADER_READ_BIT | ggml.vulkan.common.vulkan_c.VK_ACCESS_TRANSFER_READ_BIT;
    ggml.vulkan.common.dispatch.vkCmdPipelineBarrier.?(cx.cmd(), ggml.vulkan.common.vulkan_c.VK_PIPELINE_STAGE_HOST_BIT, cx.p.?.q.?.stage_flags, 0, 1, &from_host, 0, null, 0, null);
}

/// Ports `ggml_vk_comm_bridge_signal` (ggml-vulkan-comm.cpp:282 @b1b770d90).
fn bridgeSignal(comm: *CommCtx, i: usize, cx: *ggml.vulkan.submit.Context, round: u64) void {
    const dv = &comm.devs[i];
    barrierToHost(cx);
    submit(dv.ctx, cx, dv.bridge.done, round);
}

/// Ports `ggml_vk_comm_bridge_wait` (ggml-vulkan-comm.cpp:288 @b1b770d90).
fn bridgeWait(comm: *CommCtx, i: usize, cx: *ggml.vulkan.submit.Context, round: u64) void {
    cx.s().wait_semaphores.append(std.heap.c_allocator, .{ .s = comm.devs[i].bridge.go, .value = round }) catch oom();
    barrierFromHost(cx);
}

/// Ports `ggml_vk_comm_sfd_supported` (ggml-vulkan-comm.cpp:293 @b1b770d90).
fn sfdSupported(dv: *CommDevice) bool {
    const dev = dv.ctx.device;
    const mesa = dev.driver_id == ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_MESA_RADV or
        dev.driver_id == ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA or
        dev.driver_id == ggml.vulkan.common.vulkan_c.VK_DRIVER_ID_MESA_NVK;
    if (!mesa or !dev.external_semaphore) return false;
    dv.sfd.get_fd = @ptrCast(ggml.vulkan.common.dispatch.vkGetDeviceProcAddr.?(dev.device, "vkGetSemaphoreFdKHR"));
    dv.sfd.import_fd = @ptrCast(ggml.vulkan.common.dispatch.vkGetDeviceProcAddr.?(dev.device, "vkImportSemaphoreFdKHR"));
    if (dv.sfd.get_fd == null or dv.sfd.import_fd == null) return false;
    var esi = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkPhysicalDeviceExternalSemaphoreInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO);
    esi.handleType = ggml.vulkan.common.vulkan_c.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    var esp = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkExternalSemaphoreProperties, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES);
    ggml.vulkan.common.dispatch.vkGetPhysicalDeviceExternalSemaphoreProperties.?(dev.physical_device, &esi, &esp);
    const need: ggml.vulkan.common.vulkan_c.VkExternalSemaphoreFeatureFlags = ggml.vulkan.common.vulkan_c.VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT | ggml.vulkan.common.vulkan_c.VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
    return (esp.externalSemaphoreFeatures & need) == need and
        (esp.compatibleHandleTypes & ggml.vulkan.common.vulkan_c.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT) != 0;
}

/// Ports `ggml_vk_comm_sfd_enable_submit_thread` (ggml-vulkan-comm.cpp:315 @b1b770d90):
/// Mesa moves a queue to its submit thread on the first wait-before-signal,
/// taking the CS ioctls off this thread.
fn sfdEnableSubmitThread(dev: *ggml.vulkan.device.Device) ggml.vulkan.common.VkError!void {
    const sem = try createTimelineSemaphore(dev.device);
    var fci = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkFenceCreateInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    var fence: ggml.vulkan.common.vulkan_c.VkFence = null;
    try ggml.vulkan.common.check(ggml.vulkan.common.dispatch.vkCreateFence.?(dev.device, &fci, null, &fence), "vk::Device::createFence");
    const one: u64 = 1;
    var tsi = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkTimelineSemaphoreSubmitInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO);
    tsi.waitSemaphoreValueCount = 1;
    tsi.pWaitSemaphoreValues = &one;
    const stage: ggml.vulkan.common.vulkan_c.VkPipelineStageFlags = ggml.vulkan.common.vulkan_c.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    var si = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkSubmitInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_SUBMIT_INFO);
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &sem;
    si.pWaitDstStageMask = &stage;
    si.pNext = &tsi;
    try ggml.vulkan.submit.queueSubmitChecked(dev.compute_queue.?.handle, &.{si}, fence);
    try signalSemaphore(dev.device, sem, 1);
    try ggml.vulkan.submit.vkCheckChecked(ggml.vulkan.common.dispatch.vkWaitForFences.?(dev.device, 1, &fence, ggml.vulkan.common.vulkan_c.VK_TRUE, std.math.maxInt(u64)), "ggml_vk_comm_sfd_enable_submit_thread", dev);
    ggml.vulkan.common.dispatch.vkDestroyFence.?(dev.device, fence, null);
    ggml.vulkan.common.dispatch.vkDestroySemaphore.?(dev.device, sem, null);
}

/// Ports `ggml_vk_comm_sfd_export` (ggml-vulkan-comm.cpp:334 @b1b770d90).
/// Requires a submitted signal on sem; the export resets it.
fn sfdExport(dv: *CommDevice) c_int {
    var fd: c_int = -1;
    var gi = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkSemaphoreGetFdInfoKHR, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR);
    gi.semaphore = dv.sfd.sem;
    gi.handleType = ggml.vulkan.common.vulkan_c.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    const result = dv.sfd.get_fd.?(dv.ctx.device.device, &gi, &fd);
    if (result != ggml.vulkan.common.vulkan_c.VK_SUCCESS) {
        ggml.impl.ggml_c.ggml_abort(@src().file.ptr, @intCast(@src().line), "ggml_vulkan: vkGetSemaphoreFdKHR failed: %d", @as(c_int, result));
        unreachable;
    }
    return fd;
}

/// Ports `ggml_vk_comm_sfd_import` (ggml-vulkan-comm.cpp:348 @b1b770d90).
/// An fd of -1 imports an already signalled payload.
fn sfdImport(dv: *CommDevice, sem: ggml.vulkan.common.vulkan_c.VkSemaphore, fd: c_int) void {
    var isi = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkImportSemaphoreFdInfoKHR, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR);
    isi.semaphore = sem;
    isi.flags = ggml.vulkan.common.vulkan_c.VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
    isi.handleType = ggml.vulkan.common.vulkan_c.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    isi.fd = fd;
    const result = dv.sfd.import_fd.?(dv.ctx.device.device, &isi);
    if (result != ggml.vulkan.common.vulkan_c.VK_SUCCESS) {
        ggml.impl.ggml_c.ggml_abort(@src().file.ptr, @intCast(@src().line), "ggml_vulkan: vkImportSemaphoreFdKHR failed: %d", @as(c_int, result));
        unreachable;
    }
}

/// Ports `ggml_vk_comm_sfd_exchange` (ggml-vulkan-comm.cpp:361 @b1b770d90).
fn sfdExchange(comm: *CommCtx, r: u64) void {
    const n = comm.devs.len;
    for (0..n) |j| {
        const fd = sfdExport(&comm.devs[j]);
        const last = if (j == n - 1) n - 2 else n - 1;
        for (0..n) |i| {
            if (i == j) continue;
            const imp_fd = if (fd < 0 or i == last) fd else std.c.fcntl(fd, std.c.F.DUPFD_CLOEXEC, @as(c_int, 0));
            ggml.impl.assert(imp_fd >= 0 or fd < 0, "imp_fd >= 0 || fd < 0");
            sfdImport(&comm.devs[i], comm.devs[i].sfd.imp[j], imp_fd);
        }
    }
    comm.imported.store(r, .release);
}

/// Ports `ggml_vk_comm_sfd_thread` (ggml-vulkan-comm.cpp:378 @b1b770d90).
fn sfdThread(comm: *CommCtx) void {
    if (builtin.os.tag == .linux) _ = std.c.pthread_setname_np(std.c.pthread_self(), "ggml-vk-comm");
    var r: u64 = 1;
    while (waitSubmitted(comm, r)) : (r += 1) sfdExchange(comm, r);
}

/// Ports `ggml_vk_comm_sfd_init` (ggml-vulkan-comm.cpp:387 @b1b770d90).
fn sfdInit(comm: *CommCtx) ggml.vulkan.common.VkError!void {
    var esci = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkExportSemaphoreCreateInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO);
    esci.handleTypes = ggml.vulkan.common.vulkan_c.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    var sci = ggml.vulkan.common.structure(ggml.vulkan.common.vulkan_c.VkSemaphoreCreateInfo, ggml.vulkan.common.vulkan_c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    sci.pNext = &esci;
    for (comm.devs) |*dv| {
        try ggml.vulkan.common.check(ggml.vulkan.common.dispatch.vkCreateSemaphore.?(dv.ctx.device.device, &sci, null, &dv.sfd.sem), "vk::Device::createSemaphore");
        dv.sfd.imp = std.heap.c_allocator.alloc(ggml.vulkan.common.vulkan_c.VkSemaphore, comm.devs.len) catch oom();
        @memset(dv.sfd.imp, null);
        try sfdEnableSubmitThread(dv.ctx.device);
    }
    spawnThread(comm, sfdThread);
}

/// Ports `ggml_vk_comm_sfd_free` (ggml-vulkan-comm.cpp:399 @b1b770d90).
fn sfdFree(dv: *CommDevice) void {
    if (dv.sfd.sem != null) ggml.vulkan.common.dispatch.vkDestroySemaphore.?(dv.ctx.device.device, dv.sfd.sem, null);
}

/// Ports `ggml_vk_comm_sfd_signal` (ggml-vulkan-comm.cpp:405 @b1b770d90).
fn sfdSignal(comm: *CommCtx, i: usize, cx: *ggml.vulkan.submit.Context) void {
    const dv = &comm.devs[i];
    barrierToHost(cx);
    submit(dv.ctx, cx, dv.sfd.sem, 0);
    for (0..comm.devs.len) |j| {
        dv.sfd.imp[j] = if (j == i) null else ggml.vulkan.context.createBinarySemaphore(dv.ctx);
    }
}

/// Ports `ggml_vk_comm_sfd_wait` (ggml-vulkan-comm.cpp:414 @b1b770d90).
fn sfdWait(comm: *CommCtx, i: usize, cx: *ggml.vulkan.submit.Context, round: u64) void {
    for (0..comm.devs.len) |j| {
        if (j != i) cx.s().wait_semaphores.append(std.heap.c_allocator, .{ .s = comm.devs[i].sfd.imp[j], .value = 0 }) catch oom();
    }
    cx.s().ready = &comm.imported;
    cx.s().ready_value = round;
    barrierFromHost(cx);
}

/// Ports `ggml_backend_vk_comm_free` (ggml-vulkan-comm.cpp:425 @b1b770d90).
pub fn commFree(comm_ctx: ?*anyopaque) callconv(.c) void {
    const comm: *CommCtx = @ptrCast(@alignCast(comm_ctx.?));
    // pending reductions need the thread
    for (comm.devs) |dv| ggml.backend.ggml_backend_synchronize(dv.backend);
    if (comm.thread) |t| {
        {
            comm.mutex.lock();
            defer comm.mutex.unlock();
            comm.stop.store(true, .seq_cst);
        }
        comm.cv.broadcast();
        t.join();
        comm.thread = null;
    }
    for (comm.devs) |*dv| {
        sfdFree(dv);
        bridgeFree(dv);
    }
    freeBuffers(comm);
    if (comm.tctx) |tctx| ggml.context.ggml_free(tctx);
    for (comm.devs) |*dv| {
        std.heap.c_allocator.free(dv.host_buf);
        std.heap.c_allocator.free(dv.sfd.imp);
    }
    std.heap.c_allocator.free(comm.devs);
    std.heap.c_allocator.destroy(comm);
}

/// Ports `ggml_backend_vk_comm_init` (ggml-vulkan-comm.cpp:450 @b1b770d90).
pub fn commInit(backends: [*c]ggml.impl.ggml_c.ggml_backend_t, n_backends: usize) callconv(.c) ?*anyopaque {
    if (ggml.vulkan.instance.instance.allreduce_disabled or n_backends < 2) return null;
    for (0..n_backends) |i| {
        if (!ggml.vulkan.backend.isVk(backends[i])) return null;
    }
    const comm = std.heap.c_allocator.create(CommCtx) catch oom();
    comm.* = .{};
    comm.devs = std.heap.c_allocator.alloc(CommDevice, n_backends) catch oom();
    for (0..n_backends) |i| {
        const dv = &comm.devs[i];
        dv.* = .{};
        dv.backend = backends[i];
        dv.ctx = @ptrCast(@alignCast(backends[i].*.context));
        dv.host_buf = std.heap.c_allocator.alloc(ggml.impl.ggml_c.ggml_backend_buffer_t, n_backends) catch oom();
        @memset(dv.host_buf, null);
    }

    for (0..n_backends) |i| {
        const dev = comm.devs[i].ctx.device;
        comm.@"align" = @max(comm.@"align", @as(usize, @intCast(dev.min_imported_host_pointer_alignment)));
        if (!dev.external_memory_host) {
            ggml.impl.logWarn("ggml_vulkan: device %zu: VK_EXT_external_memory_host missing, using generic all-reduce\n", .{i});
            commFree(comm);
            return null;
        }
        if (dev.serialize_submissions) {
            ggml.impl.logWarn("ggml_vulkan: device %zu: GGML_VK_SERIALIZE_SUBMISSIONS set, using generic all-reduce\n", .{i});
            commFree(comm);
            return null;
        }
    }

    // `std::all_of`, which stops at the first device that answers no.
    const all_sfd = for (comm.devs) |*dv| {
        if (!sfdSupported(dv)) break false;
    } else true;
    if (all_sfd) comm.mode = .sync_fd;

    if (stdlib.env.getenv("GGML_VK_ALLREDUCE_SPIN_US")) |spin| {
        comm.spin = @as(i64, @max(0, stdlib.num.atoiC(spin))) * std.time.ns_per_us;
    }
    const ip: ggml.impl.ggml_c.ggml_init_params = .{ .mem_size = ggml.types.ggml_tensor_overhead(), .mem_buffer = null, .no_alloc = true };
    comm.tctx = ggml.context.ggml_init(ip);
    comm.slot = ggml.context.ggml_new_tensor_1d(comm.tctx.?, ggml.impl.ggml_c.GGML_TYPE_F32, 1);

    const inited = if (comm.mode == .sync_fd) sfdInit(comm) else bridgeInit(comm);
    inited catch {
        // `e.what()` of the `vk::SystemError`: its message, then the code's.
        ggml.impl.logWarn("ggml_vulkan: semaphore initialization failed: %s: %s, using generic all-reduce\n", .{ ggml.vulkan.common.last_error.what, ggml.vulkan.common.resultString(ggml.vulkan.common.last_error.result).ptr });
        commFree(comm);
        return null;
    };
    return comm;
}

/// Ports `ggml_backend_vk_comm_allreduce_tensor` (ggml-vulkan-comm.cpp:510 @b1b770d90).
///
/// A: partial to own host slot (round & 1), signal; B: wait on peers' A,
/// add their slots; all A precede any B. Slot reuse: A(k) follows own
/// B(k-1), which waited on peers' A(k-1), submitted after their B(k-2), the
/// last reader.
pub fn commAllreduceTensor(comm_ctx: ?*anyopaque, tensors: [*c][*c]ggml.impl.ggml_c.ggml_tensor) callconv(.c) bool {
    const comm: *CommCtx = @ptrCast(@alignCast(comm_ctx.?));
    const n = comm.devs.len;

    const ne = ggml.types.ggml_nelements(tensors[0]);
    for (0..n) |i| {
        const t = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, tensors[i]);
        if (t.type != ggml.impl.ggml_c.GGML_TYPE_F32 or ggml.types.ggml_nelements(t) != ne or !ggml.types.ggml_is_contiguous(t) or !ggml.vulkan.buffer.bufferIsVk(t.buffer)) {
            return false;
        }
        const ctx = comm.devs[i].ctx;
        if (ctx.compute_ctx) |cc| if (cc.out_memcpys.items.len != 0) return false;
    }
    if (ne == 0) return true;
    const nbytes = ggml.types.ggml_nbytes(tensors[0]);
    if (!ensure(comm, nbytes)) return false;

    const f32_wire = nbytes < f16_min_size or needsF32(tensors[0], 0);
    const wire: ggml.impl.ggml_c.ggml_type = if (f32_wire) ggml.impl.ggml_c.GGML_TYPE_F32 else ggml.impl.ggml_c.GGML_TYPE_F16;
    comm.round += 1;
    const round = comm.round;
    const slot_offs = (round & 1) * comm.cap;
    const slot_size = @as(usize, @intCast(ne)) * ggml.types.ggml_type_size(wire);

    for (0..n) |i| {
        const dv = &comm.devs[i];
        const ctx = dv.ctx;
        const t = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, tensors[i]);
        const tbuf = ggml.vulkan.buffer.tensorBuffer(t);
        const hbuf = ggml.vulkan.buffer.devBuffer(dv.host_buf[i]);
        const toffs = ggml.vulkan.buffer.tensorOffset(t) + t.view_offs;

        const cx = ggml.vulkan.context.getComputeCtx(ctx);
        ggml.vulkan.context.syncBuffers(ctx, cx);
        if (t.flags & ggml.impl.ggml_c.GGML_TENSOR_FLAG_COMPUTE == 0) {
            ggml.vulkan.common.dispatch.vkCmdFillBuffer.?(cx.cmd(), hbuf.buffer, slot_offs, stdlib.num.pad(usize, slot_size, 4), 0);
            ggml.vulkan.common.dispatch.vkCmdFillBuffer.?(cx.cmd(), tbuf.buffer, toffs, nbytes, 0);
        } else if (wire == ggml.impl.ggml_c.GGML_TYPE_F32) {
            ggml.vulkan.submit.bufferCopyAsync(cx, hbuf, slot_offs, tbuf, toffs, nbytes);
        } else {
            setSlot(comm.slot, t, wire, dv.host_buf[i], slot_offs);
            ggml.vulkan.ops_basic.cpy(ctx, cx, t, comm.slot);
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
        const cx = ggml.vulkan.context.getComputeCtx(ctx);
        ctx.do_add_rms_partials = false;
        if (comm.mode == .sync_fd) {
            sfdWait(comm, i, cx, round);
        } else {
            bridgeWait(comm, i, cx, round);
        }
        ggml.vulkan.context.syncBuffers(ctx, cx);
        const t = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, tensors[i]);
        for (0..n) |j| {
            if (j == i) continue;
            setSlot(comm.slot, t, wire, comm.devs[j].host_buf[i], slot_offs);
            ggml.vulkan.ops_basic.add(ctx, cx, t, comm.slot, t);
            ggml.vulkan.context.syncBuffers(ctx, cx);
        }
        // small reductions go out with the next graph's first submit
        if (wire != ggml.impl.ggml_c.GGML_TYPE_F32 or ggml.vulkan.instance.loggers.perf_logger_enabled) {
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
    var mm = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    mm.op = ggml.impl.ggml_c.GGML_OP_MUL_MAT;
    var add = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    add.op = ggml.impl.ggml_c.GGML_OP_ADD;
    add.src[1] = &mm;
    var view = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    view.op = ggml.impl.ggml_c.GGML_OP_VIEW;
    view.src[0] = &add;
    try std.testing.expect(!needsF32(&view, 0));
    mm.op_params[0] = ggml.impl.ggml_c.GGML_PREC_F32;
    try std.testing.expect(needsF32(&view, 0));
    // A bf16 accumulator wants the f32 wire too.
    mm.op_params[0] = ggml.impl.ggml_c.GGML_PREC_BF16;
    try std.testing.expect(needsF32(&view, 0));
    mm.op_params[0] = ggml.impl.ggml_c.GGML_PREC_F16;
    try std.testing.expect(!needsF32(&view, 0));
    mm.op_params[0] = ggml.impl.ggml_c.GGML_PREC_F32;
    // MUL follows only its first source.
    var mul = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    mul.op = ggml.impl.ggml_c.GGML_OP_MUL;
    mul.src[1] = &mm;
    try std.testing.expect(!needsF32(&mul, 0));
    try std.testing.expect(!needsF32(null, 0));
    // Past depth 64 the walk gives up.
    try std.testing.expect(!needsF32(&view, 64));
}
