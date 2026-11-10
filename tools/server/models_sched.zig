//! When the router loads and unloads model instances: the queue of requests
//! waiting for a `--models-max` slot, and the device memory accounting of
//! `--models-memory-margin`, which measures each model in a child and unloads
//! idle models until a new one fits.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server-models-sched.cpp` and
//! `llama.cpp/tools/server/server-models-sched.h` (v0.9.0, `cd959bae9`). Each
//! declaration names what it ports and the line it began at.
//!
//! # Its shape
//!
//! Every method of `Sched` that takes a `std::unique_lock & lk` in the C++
//! expects `models.mutex` held here, as `check_lock` asserts there. A
//! `server_model_memory` is a value: its strings live in the allocator it was
//! built with, an instance's arena for what `Meta` holds. `measure_proc` is
//! the C++'s `shared_ptr` as a `SharedProc` reference.

const std = @import("std");
const tc = @import("tools_common");
const json = @import("json");
const common = @import("common.zig");
const context = @import("context.zig");
const models_mod = @import("models.zig");

const Value = json.Value;
const Allocator = std.mem.Allocator;
const Models = models_mod.Models;
const SharedProc = models_mod.SharedProc;
const srv = common.srv;
const gpa = context.gpa;
const MemoryBreakdownRow = @import("cli_fit").MemoryBreakdownRow;

/// Ports `MEASURE_TIMEOUT` (server-models-sched.cpp:9 @cd959bae9): seconds,
/// then the measure child is killed.
const measure_timeout = 300;

fn zs(a: Allocator, s: []const u8) Allocator.Error![*:0]const u8 {
    return (try a.dupeSentinel(u8, s, 0)).ptr;
}

const mib: usize = 1024 * 1024;

// -----------------------------------------------------------------------------
// server_model_memory

/// Ports `device_t` (server-models-sched.h:20 @cd959bae9), a member of
/// `server_model_memory`.
pub const Device = struct {
    kind: []const u8 = "", // "device", "host" or "buffer_type"
    name: []const u8 = "", // ggml device name, "Host" for host memory
    total: usize = 0, // device memory, 0 unless kind == "device"
    free: usize = 0, // free device memory when this was measured, 0 unless kind == "device"
    model: usize = 0, // held by the instance, by kind of buffer
    context: usize = 0,
    compute: usize = 0,

    /// Ports `is_device` (server-models-sched.h:29 @cd959bae9).
    pub fn isDevice(self: *const Device) bool {
        return std.mem.eql(u8, self.kind, "device");
    }

    /// Ports `self` (server-models-sched.h:30 @cd959bae9).
    pub fn selfBytes(self: *const Device) usize {
        return self.model +% self.context +% self.compute;
    }
};

/// Ports `server_model_memory` (server-models-sched.h:19 @cd959bae9): the
/// memory of a model instance per device, in bytes; from a measure child or
/// from the running instance.
pub const ServerModelMemory = struct {
    n_ctx: u32 = 0, // context size this was measured at
    devices: []const Device = &.{},

    /// Ports `server_model_memory` (server-models-sched.cpp:15 @cd959bae9),
    /// the constructor over breakdown rows.
    pub fn fromRows(a: Allocator, n_ctx: u32, rows: []const MemoryBreakdownRow) Allocator.Error!ServerModelMemory {
        const devices = try a.alloc(Device, rows.len);
        for (rows, devices) |*row, *d| {
            d.* = .{
                .kind = switch (row.kind) {
                    .device => "device",
                    .host => "host",
                    .buffer_type => "buffer_type",
                },
                .name = try a.dupe(u8, row.name),
                .total = @bitCast(row.total),
                .free = @bitCast(row.free),
                .model = row.model,
                .context = row.context,
                .compute = row.compute,
            };
        }
        return .{ .n_ctx = n_ctx, .devices = devices };
    }

    /// Ports `to_json` (server-models-sched.cpp:33 @cd959bae9), a member of
    /// `server_model_memory`.
    pub fn toJson(self: *const ServerModelMemory, a: Allocator) common.Error!Value {
        var devs = Value.emptyArray();
        for (self.devices) |*d| {
            var jd = common.Obj.init(a);
            try jd.put("kind", d.kind);
            try jd.put("name", d.name);
            try jd.put("model", d.model);
            try jd.put("context", d.context);
            try jd.put("compute", d.compute);
            if (d.isDevice()) {
                try jd.put("total", d.total);
                try jd.put("free", d.free);
            }
            try devs.pushBack(a, jd.done());
        }
        var o = common.Obj.init(a);
        try o.put("n_ctx", self.n_ctx);
        try o.put("devices", devs);
        return o.done();
    }

    /// Ports `from_json` (server-models-sched.cpp:55 @cd959bae9), a member of
    /// `server_model_memory`: throws on a malformed input.
    pub fn fromJson(a: Allocator, j: *const Value) common.Error!ServerModelMemory {
        var mem: ServerModelMemory = .{};
        mem.n_ctx = try (try j.at("n_ctx")).get(u32);
        const jdevs = try j.at("devices");
        var devices: std.ArrayList(Device) = .empty;
        if (jdevs.isArray()) {
            for (jdevs.array.items) |*jd| {
                var d: Device = .{};
                d.kind = try a.dupe(u8, try (try jd.at("kind")).get([]const u8));
                d.name = try a.dupe(u8, try (try jd.at("name")).get([]const u8));
                d.total = common.jsonValue(jd, "total", usize, 0);
                d.free = common.jsonValue(jd, "free", usize, 0);
                d.model = try (try jd.at("model")).get(usize);
                d.context = try (try jd.at("context")).get(usize);
                d.compute = try (try jd.at("compute")).get(usize);
                try devices.append(a, d);
            }
        } else if (!jdevs.isNull()) {
            // a range-for over a scalar visits the scalar, whose `.at` throws;
            // no child sends an object here
            _ = try jdevs.at("kind");
        }
        mem.devices = devices.items;
        return mem;
    }

    /// The C++ copy: every string deep-copied into `a`.
    pub fn clone(self: *const ServerModelMemory, a: Allocator) Allocator.Error!ServerModelMemory {
        const devices = try a.alloc(Device, self.devices.len);
        for (self.devices, devices) |*s, *d| {
            d.* = s.*;
            d.kind = try a.dupe(u8, s.kind);
            d.name = try a.dupe(u8, s.name);
        }
        return .{ .n_ctx = self.n_ctx, .devices = devices };
    }
};

/// Ports `server_model_memory_status` (server-models-sched.h:44 @cd959bae9):
/// the memory measurement state of a model.
pub const MemoryStatus = enum {
    unmeasured, // queued for a measure child
    measuring, // its measure child is running
    measured,
    failed, // measure failed, no memory limit for this model
};

/// Ports `server_model_memory_status_to_string` (server-models-sched.h:51
/// @cd959bae9).
fn memoryStatusToString(status: MemoryStatus) []const u8 {
    return switch (status) {
        .unmeasured => "unmeasured",
        .measuring => "measuring",
        .measured => "measured",
        .failed => "failed",
    };
}

/// Ports `server_model_memory_info` (server-models-sched.h:61 @cd959bae9).
pub const MemoryInfo = struct {
    status: MemoryStatus = .unmeasured,
    n_attempts: c_int = 0, // to limit the retries
    min: ?ServerModelMemory = null, // at the smallest context --fit may pick
    max: ?ServerModelMemory = null, // at the largest context --fit picks, a snapshot
    effective: ?ServerModelMemory = null, // what the running instance allocated, reset when it exits

    /// Ports `is_settled` (server-models-sched.h:68 @cd959bae9).
    pub fn isSettled(self: *const MemoryInfo) bool {
        return self.status == .measured or self.status == .failed;
    }

    /// Ports `to_json` (server-models-sched.cpp:72 @cd959bae9), a member of
    /// `server_model_memory_info`.
    pub fn toJson(self: *const MemoryInfo, a: Allocator) common.Error!Value {
        var o = common.Obj.init(a);
        try o.put("status", memoryStatusToString(self.status));
        try o.put("min", if (self.min) |*m| try m.toJson(a) else @as(Value, .null));
        try o.put("max", if (self.max) |*m| try m.toJson(a) else @as(Value, .null));
        try o.put("effective", if (self.effective) |*m| try m.toJson(a) else @as(Value, .null));
        return o.done();
    }

    /// The C++ copy.
    pub fn clone(self: *const MemoryInfo, a: Allocator) Allocator.Error!MemoryInfo {
        var out = self.*;
        if (self.min) |*m| out.min = try m.clone(a);
        if (self.max) |*m| out.max = try m.clone(a);
        if (self.effective) |*m| out.effective = try m.clone(a);
        return out;
    }
};

// -----------------------------------------------------------------------------
// server_models_sched

/// Ports `server_models_sched` (server-models-sched.h:76 @cd959bae9): the
/// router's request queue (`--models-max`) and memory accounting
/// (`--models-memory-margin`). The router never touches a GPU: it tracks what
/// each instance holds, and child reports only calibrate the rest, which works
/// for per-process (Metal) and system-wide (CUDA) free memory.
pub const Sched = struct {
    /// Ports `entry_t` (server-models-sched.h:143 @cd959bae9).
    const Entry = struct {
        model_id: []const u8,
        n_waiters: c_int, // requests waiting for this model
        loading: bool, // one of the waiters is doing the load right now
    };

    /// Ports `device_memory_t` (server-models-sched.h:149 @cd959bae9).
    const DeviceMemory = struct {
        total: i64 = 0, // bytes
        external: i64 = 0, // bytes not tracked by the router: other processes, backend contexts
    };

    /// `std::map<std::string, size_t>` from one device to the bytes held on
    /// it, the deficit's type as well: sorted by name.
    const Bytes = struct { dev: []const u8, bytes: usize };

    models: *Models,

    // requests waiting for a slot
    queue: std.ArrayList(Entry) = .empty,

    // one measure child runs at a time
    measure_queue: std.ArrayList([]const u8) = .empty,
    measuring: []const u8 = "", // model whose measure child is running, empty when idle
    measure_proc: ?*SharedProc = null,

    device_memory: std.StringHashMapUnmanaged(DeviceMemory) = .empty, // by device name, empty until the first child reports
    held: std.StringHashMapUnmanaged(std.ArrayList(Bytes)) = .empty, // model -> device -> bytes counted for its instance

    //
    // request queue
    //

    /// Ports `has_capacity` (server-models-sched.cpp:85 @cd959bae9), a member
    /// of `server_models_sched`.
    pub fn hasCapacity(self: *Sched) bool {
        return self.models.base_params.models_max <= 0 or
            self.countRunning() < @as(usize, @intCast(self.models.base_params.models_max));
    }

    /// Ports `pick_victim` (server-models-sched.cpp:91 @cd959bae9), a member
    /// of `server_models_sched`.
    ///
    /// Return: "" if no model can be given up; borrowed from the mapping.
    pub fn pickVictim(self: *Sched) []const u8 {
        var victim: []const u8 = "";
        var victim_last_used: i64 = 0;
        for (self.models.mapping.items) |m| {
            // a busy model is mid-request, one still coming up has no request to finish
            if (m.req_count != 0 or !m.meta.isReadyOrSleep()) {
                continue;
            }
            // already on its way out, or a queued request wants it
            if (self.models.stopping_models.contains(m.meta.name) or self.find(m.meta.name) != null) {
                continue;
            }
            if (victim.len == 0 or m.meta.last_used < victim_last_used) {
                victim = m.meta.name;
                victim_last_used = m.meta.last_used;
            }
        }
        return victim;
    }

    /// Ports `join` (server-models-sched.cpp:112 @cd959bae9), a member of
    /// `server_models_sched`: requests wanting the same model share one
    /// entry, so they all need only one slot and all get unblocked by the
    /// single load that entry performs.
    pub fn join(self: *Sched, model_id: []const u8) !void {
        const z = try zs(gpa, model_id);
        defer gpa.free(std.mem.span(z));
        if (self.find(model_id)) |e| {
            e.n_waiters += 1;
            srv(.inf, "join", "request for name=%s joined the queue, %d waiting\n", .{ z, e.n_waiters });
            return;
        }
        try self.queue.append(gpa, .{ .model_id = try gpa.dupe(u8, model_id), .n_waiters = 1, .loading = false });
        srv(.inf, "join", "request for name=%s queued at position %zu\n", .{ z, self.queue.items.len });
    }

    /// Ports `leave` (server-models-sched.cpp:124 @cd959bae9), a member of
    /// `server_models_sched`.
    pub fn leave(self: *Sched, model_id: []const u8) void {
        for (self.queue.items, 0..) |*e, i| {
            if (std.mem.eql(u8, e.model_id, model_id)) {
                e.n_waiters -= 1;
                if (e.n_waiters <= 0) {
                    gpa.free(e.model_id);
                    _ = self.queue.orderedRemove(i); // last one waiting for this model went away
                }
                return;
            }
        }
    }

    /// Ports `queue_empty` (server-models-sched.cpp:136 @cd959bae9), a member
    /// of `server_models_sched`.
    pub fn queueEmpty(self: *Sched) bool {
        return self.queue.items.len == 0;
    }

    /// Ports `try_claim` (server-models-sched.cpp:141 @cd959bae9), a member
    /// of `server_models_sched`: true if it is this model's turn to load, and
    /// nobody is loading it yet.
    pub fn tryClaim(self: *Sched, model_id: []const u8) bool {
        if (self.queue.items.len == 0 or !std.mem.eql(u8, self.queue.items[0].model_id, model_id) or self.queue.items[0].loading) {
            return false;
        }
        if (!self.hasCapacity()) {
            return false;
        }
        // over the memory margin with every running model busy: wait for one to go idle
        if (self.memoryEnabled() and self.limitsExceeded(model_id, false) and self.pickVictim().len == 0) {
            return false;
        }
        self.queue.items[0].loading = true;
        return true;
    }

    /// Ports `claim_done` (server-models-sched.cpp:157 @cd959bae9), a member
    /// of `server_models_sched`: on failure the entry is back in line; on
    /// success it stays until its waiters leave, so the model coming up is
    /// never picked as a victim before they use it.
    pub fn claimDone(self: *Sched, model_id: []const u8, ok: bool) void {
        if (ok) {
            return;
        }
        for (self.queue.items) |*e| {
            if (std.mem.eql(u8, e.model_id, model_id)) {
                e.loading = false;
                return;
            }
        }
    }

    /// Ports `tick` (server-models-sched.cpp:170 @cd959bae9), a member of
    /// `server_models_sched`: evict idle models while queued requests
    /// outnumber the slots that are free or being freed. Never blocks, so it
    /// is safe from any thread.
    pub fn tick(self: *Sched) void {
        const models = self.models;
        if (models.base_params.models_max <= 0 or self.queue.items.len == 0) {
            return;
        }
        var n_running: c_int = 0;
        var n_stopping: c_int = 0;
        for (models.mapping.items) |m| {
            if (m.meta.isRunning()) {
                n_running += 1;
                if (models.stopping_models.contains(m.meta.name)) {
                    n_stopping += 1;
                }
            }
        }
        var n_needed: c_int = 0;
        var n_claimed: c_int = 0; // claimed the slot, but load() has not spawned yet
        for (self.queue.items) |e| {
            if (!e.loading) {
                n_needed += 1;
                continue;
            }
            if (models.find(e.model_id)) |it| {
                if (!it.meta.isRunning()) n_claimed += 1;
            }
        }
        var n_free = models.base_params.models_max - n_running + n_stopping - n_claimed;
        while (n_free < n_needed) {
            const victim = self.pickVictim();
            if (victim.len == 0) {
                return; // all remaining models are busy, wait for a request to end
            }
            const z = zs(gpa, victim) catch return;
            defer gpa.free(std.mem.span(z));
            srv(.inf, "tick", "evicting idle LRU name=%s for a queued request\n", .{z});
            models.requestStop(victim, true);
            n_free += 1;
        }
    }

    /// Ports `find` (server-models-sched.cpp:209 @cd959bae9), a member of
    /// `server_models_sched`.
    fn find(self: *Sched, model_id: []const u8) ?*Entry {
        for (self.queue.items) |*e| {
            if (std.mem.eql(u8, e.model_id, model_id)) return e;
        }
        return null;
    }

    /// Ports `count_running` (server-models-sched.cpp:222 @cd959bae9), a
    /// member of `server_models_sched`.
    fn countRunning(self: *Sched) usize {
        var count: usize = 0;
        for (self.models.mapping.items) |m| {
            if (m.meta.isRunning()) count += 1;
        }
        return count;
    }

    //
    // loading a model
    //

    /// Ports `limits_exceeded` (server-models-sched.cpp:236 @cd959bae9), a
    /// member of `server_models_sched`: true if loading `name` now exceeds
    /// `models_max` or the memory margin.
    pub fn limitsExceeded(self: *Sched, name: []const u8, verbose: bool) bool {
        if (!self.hasCapacity()) {
            return true;
        }
        if (!self.memoryEnabled()) {
            return false;
        }
        var missing: std.ArrayList(Bytes) = .empty;
        defer missing.deinit(gpa);
        self.deficit(name, &missing) catch return false;
        if (missing.items.len == 0) {
            return false;
        }
        var any_running = false;
        for (self.models.mapping.items) |m| {
            if (m.meta.isRunning()) {
                any_running = true;
                break;
            }
        }
        for (missing.items) |e| {
            if (!verbose) {
                break;
            }
            const nz = zs(gpa, name) catch break;
            defer gpa.free(std.mem.span(nz));
            const dz = zs(gpa, e.dev) catch break;
            defer gpa.free(std.mem.span(dz));
            if (any_running) {
                srv(.dbg, "limits_exceeded", "model name=%s is short by %zu MiB on %s\n", .{ nz, e.bytes / mib, dz });
            } else {
                // nothing to unload: min is pessimistic, let --fit in the instance try
                srv(.wrn, "limits_exceeded", "model name=%s is projected to be short by %zu MiB on %s with nothing else loaded, loading anyway\n", .{ nz, e.bytes / mib, dz });
            }
        }
        return any_running;
    }

    /// `measured` in `make_room`.
    fn isMeasured(self: *Sched, name: []const u8) bool {
        const it = self.models.find(name) orelse return true;
        return it.meta.memory.isSettled();
    }

    /// Ports `make_room` (server-models-sched.cpp:269 @cd959bae9), a member
    /// of `server_models_sched`: wait for the measure of `name`, then unload
    /// idle LRU models until it fits. Called without `models.mutex`.
    pub fn makeRoom(self: *Sched, name: []const u8) !void {
        const models = self.models;
        if (self.memoryEnabled()) {
            // the measure result decides what to unload
            models.mutex.lock();
            defer models.mutex.unlock();
            try self.enqueueMeasure(name, true);
            if (!self.isMeasured(name)) {
                const z = try zs(gpa, name);
                defer gpa.free(std.mem.span(z));
                srv(.inf, "make_room", "waiting for the memory measurement of model name=%s\n", .{z});
                while (!self.isMeasured(name)) models.cv.wait(&models.mutex);
            }
        }

        while (true) {
            var lru_model_name: []const u8 = "";
            defer if (lru_model_name.len != 0) gpa.free(lru_model_name);
            {
                models.mutex.lock();
                defer models.mutex.unlock();
                if (!self.limitsExceeded(name, true)) {
                    return;
                }
                const v = self.pickVictim();
                if (v.len != 0) lru_model_name = try gpa.dupe(u8, v);
            }
            if (lru_model_name.len == 0) {
                return; // every running model is busy, load() reports the limit
            }
            const z = try zs(gpa, lru_model_name);
            defer gpa.free(std.mem.span(z));
            srv(.inf, "make_room", "limits reached, removing LRU name=%s\n", .{z});
            try models.unload(lru_model_name);
            // wait for unload to complete
            {
                models.mutex.lock();
                defer models.mutex.unlock();
                while ((try models.getOrCreate(lru_model_name)).meta.status != .unloaded) {
                    models.cv.wait(&models.mutex);
                }
            }
        }
    }

    //
    // memory accounting
    //

    /// Ports `memory_enabled` (server-models-sched.cpp:312 @cd959bae9), a
    /// member of `server_models_sched`.
    pub fn memoryEnabled(self: *const Sched) bool {
        return self.models.base_params.models_memory_margin > 0;
    }

    /// Ports `measure_unmeasured` (server-models-sched.cpp:316 @cd959bae9), a
    /// member of `server_models_sched`: queue a measure child for every model
    /// not measured yet.
    pub fn measureUnmeasured(self: *Sched) !void {
        for (self.models.mapping.items) |inst| {
            try self.enqueueMeasure(inst.meta.name, false);
        }
    }

    /// Ports `on_spawned` (server-models-sched.cpp:323 @cd959bae9), a member
    /// of `server_models_sched`: count a new instance at its min until it
    /// reports.
    pub fn onSpawned(self: *Sched, name: []const u8) !void {
        try self.reserve(name);
    }

    /// Ports `on_ready` (server-models-sched.cpp:328 @cd959bae9), a member of
    /// `server_models_sched`: `memory` is what the instance allocated, null
    /// on wake up from sleep.
    pub fn onReady(self: *Sched, name: []const u8, memory: ?*const ServerModelMemory) !void {
        if (!self.memoryEnabled()) {
            return;
        }
        const it = self.models.find(name) orelse return;
        const mem = memory orelse {
            try self.reserve(name); // woke up from sleep
            return;
        };
        // use the real usage, not the estimate
        const info = &it.meta.memory;
        info.effective = try mem.clone(it.alloc());
        self.release(name);
        try self.reserve(name);
        try self.calibrate(mem, name);
        if (info.status == .failed and info.n_attempts < 2) {
            // the model was maybe not downloaded at measure time, retry
            info.status = .unmeasured;
            try self.enqueueMeasure(name, false);
        }
    }

    /// Ports `on_sleeping` (server-models-sched.cpp:354 @cd959bae9), a member
    /// of `server_models_sched`: a sleeping instance frees its model and
    /// context.
    pub fn onSleeping(self: *Sched, name: []const u8) void {
        self.release(name);
    }

    /// Ports `on_unloaded` (server-models-sched.cpp:359 @cd959bae9), a member
    /// of `server_models_sched`.
    pub fn onUnloaded(self: *Sched, name: []const u8) void {
        self.release(name);
        if (self.models.find(name)) |it| {
            it.meta.memory.effective = null;
        }
    }

    /// Ports `cancel_measurements` (server-models-sched.cpp:368 @cd959bae9),
    /// a member of `server_models_sched`: stop the measure child and drop the
    /// queue, for shutdown.
    pub fn cancelMeasurements(self: *Sched) void {
        for (self.measure_queue.items) |n| gpa.free(n);
        self.measure_queue.clearRetainingCapacity();
        if (self.measure_proc) |proc| {
            const z = zs(gpa, self.measuring) catch "";
            srv(.inf, "cancel_measurements", "stopping measure process for model name=%s\n", .{z});
            proc.proc.terminate();
            // the monitor reaps it and clears measuring
        }
    }

    /// Ports `is_measuring` (server-models-sched.cpp:378 @cd959bae9), a
    /// member of `server_models_sched`.
    pub fn isMeasuring(self: *const Sched) bool {
        return self.measuring.len != 0;
    }

    /// Ports `enqueue_measure` (server-models-sched.cpp:383 @cd959bae9), a
    /// member of `server_models_sched`.
    fn enqueueMeasure(self: *Sched, name: []const u8, front: bool) !void {
        if (!self.memoryEnabled()) {
            return;
        }
        const it = self.models.find(name) orelse return;
        if (it.meta.memory.status != .unmeasured) {
            return;
        }
        for (self.measure_queue.items, 0..) |n, i| {
            if (std.mem.eql(u8, n, name)) {
                if (!front) {
                    return; // already queued
                }
                gpa.free(n);
                _ = self.measure_queue.orderedRemove(i);
                break;
            }
        }
        const owned = try gpa.dupe(u8, name);
        if (front) {
            try self.measure_queue.insert(gpa, 0, owned);
        } else {
            try self.measure_queue.append(gpa, owned);
        }
        try self.measureNext();
    }

    /// Ports `measure_next` (server-models-sched.cpp:406 @cd959bae9), a
    /// member of `server_models_sched`.
    fn measureNext(self: *Sched) !void {
        const models = self.models;
        if (self.measuring.len != 0) {
            return; // one measure child at a time
        }
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        while (self.measure_queue.items.len != 0) {
            const name = self.measure_queue.orderedRemove(0);
            var name_owned = true;
            defer if (name_owned) gpa.free(name);
            const it = models.find(name) orelse continue; // removed meanwhile
            if (it.meta.memory.status != .unmeasured) {
                continue; // measured meanwhile
            }
            const meta = &it.meta;
            const z = try zs(sa, name);

            // same args as the instance, --offline: never download
            var child_args: std.ArrayList([]const u8) = .empty;
            try child_args.appendSlice(sa, meta.args);
            try child_args.append(sa, "--offline");
            var child_env: std.ArrayList([]const u8) = .empty;
            try child_env.appendSlice(sa, models.base_env);
            try child_env.append(sa, try std.fmt.allocPrint(sa, "{s}={d}", .{ models_mod.env_router_port, models.base_params.port }));
            try child_env.append(sa, try std.mem.concat(sa, u8, &.{ models_mod.env_child_role, "=", models_mod.roleTypeToStr(.measure) }));

            const proc = try SharedProc.create();
            if (!proc.proc.sproc.create(child_args.items, tc.subproc.option_no_window, child_env.items, null)) {
                proc.release();
                srv(.err, "measure_next", "failed to spawn measure process for model name=%s\n", .{z});
                meta.memory.status = .failed;
                continue;
            }
            srv(.inf, "measure_next", "measuring memory for model name=%s\n", .{z});
            meta.memory.status = .measuring;
            meta.memory.n_attempts += 1;
            self.measuring = name;
            name_owned = false;
            self.measure_proc = proc;
            try models.watchChild(name, proc, .measure, 0, measure_timeout);
            return;
        }
    }

    /// Ports `on_measure_exit` (server-models-sched.cpp:442 @cd959bae9), a
    /// member of `server_models_sched`: called from the monitor thread.
    pub fn onMeasureExit(self: *Sched, name: []const u8, proc: *SharedProc, exit_code: c_int) void {
        const models = self.models;
        models.mutex.lock();
        defer models.mutex.unlock();
        if (self.measure_proc != proc) {
            return; // not the child we are waiting for
        }
        proc.release();
        self.measure_proc = null;
        gpa.free(self.measuring);
        self.measuring = "";
        if (models.find(name)) |it| {
            if (it.meta.memory.status == .measuring) {
                // exited without a result: crashed, or killed after the timeout
                const z = zs(gpa, name) catch "";
                srv(.wrn, "on_measure_exit", "measure process for model name=%s exited with status %d before reporting, memory limits will not apply to it\n", .{ z, exit_code });
                it.meta.memory.status = .failed;
                self.notifyMemory(name, &it.meta.memory);
            }
        }
        models.cv.notifyAll();
        self.measureNext() catch |err| switch (err) {
            else => {},
        };
    }

    /// `models.notify_sse("model_memory", name, info.to_json())`.
    fn notifyMemory(self: *Sched, name: []const u8, info: *const MemoryInfo) void {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const v = info.toJson(scratch.allocator()) catch return;
        self.models.notifySse("model_memory", name, &v);
    }

    /// Ports `handle_measured` (server-models-sched.cpp:461 @cd959bae9), a
    /// member of `server_models_sched`: called from the monitor thread.
    pub fn handleMeasured(self: *Sched, name: []const u8, payload: *const Value) void {
        const func = "handle_measured";
        const models = self.models;
        models.mutex.lock();
        defer models.mutex.unlock();
        const it = models.find(name) orelse return;
        const info = &it.meta.memory;
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        const nz = zs(sa, name) catch return;
        const result = common.jsonValue(payload, "result", []const u8, "");

        var ok = std.mem.eql(u8, result, "measured");
        if (ok) {
            self.parseMeasured(it.alloc(), info, payload) catch |e| {
                srv(.err, func, "malformed memory measurement for model name=%s: %s\n", .{ nz, zs(sa, context.whatOf(e)) catch "" });
                info.min = null;
                info.max = null;
                ok = false;
            };
        } else {
            srv(.wrn, func, "measuring memory for model name=%s failed: %s\n", .{ nz, zs(sa, common.jsonValue(payload, "error", []const u8, "unknown error")) catch "" });
        }

        if (ok) {
            info.status = .measured;
            self.calibrate(&info.max.?, "") catch |e| switch (e) { // latest view of the devices
                error.OutOfMemory => {},
            };
            const mn = &info.min.?;
            const mx = &info.max.?;
            var i: usize = 0;
            while (i < mn.devices.len and i < mx.devices.len) : (i += 1) {
                if (!mn.devices[i].isDevice() or !std.mem.eql(u8, mn.devices[i].name, mx.devices[i].name)) {
                    continue;
                }
                srv(.inf, func, "model name=%s needs on %s: %zu MiB at n_ctx=%u (min), %zu MiB at n_ctx=%u (max), %zu MiB free\n", .{
                    nz,
                    zs(sa, mn.devices[i].name) catch "",
                    mn.devices[i].selfBytes() / mib,
                    mn.n_ctx,
                    mx.devices[i].selfBytes() / mib,
                    mx.n_ctx,
                    mx.devices[i].free / mib,
                });
            }
        } else {
            info.status = .failed;
        }
        self.notifyMemory(name, info);
        models.cv.notifyAll();
    }

    fn parseMeasured(_: *Sched, a: Allocator, info: *MemoryInfo, payload: *const Value) common.Error!void {
        info.min = try ServerModelMemory.fromJson(a, try payload.at("min"));
        info.max = try ServerModelMemory.fromJson(a, try payload.at("max"));
    }

    /// Ports `calibrate` (server-models-sched.cpp:508 @cd959bae9), a member
    /// of `server_models_sched`: update `device_memory` from a child report;
    /// `reporter` is the instance that sent it, empty for a measure child.
    fn calibrate(self: *Sched, mem: *const ServerModelMemory, reporter: []const u8) !void {
        for (mem.devices) |*d| {
            if (!d.isDevice() or d.total == 0) {
                continue;
            }
            // other instances are tracked, so they are not external (CUDA counts them in free, Metal does not)
            var tracked: i64 = 0;
            var it = self.held.iterator();
            while (it.next()) |e| {
                if (std.mem.eql(u8, e.key_ptr.*, reporter)) {
                    continue;
                }
                for (e.value_ptr.items) |h| {
                    if (std.mem.eql(u8, h.dev, d.name)) {
                        tracked += @bitCast(h.bytes);
                    }
                }
            }
            // a measure child allocates nothing, its free does not include self
            const self_bytes: i64 = if (reporter.len == 0) 0 else @bitCast(d.selfBytes());
            const gop = try self.device_memory.getOrPut(gpa, d.name);
            if (!gop.found_existing) {
                gop.key_ptr.* = try gpa.dupe(u8, d.name);
                gop.value_ptr.* = .{};
            }
            const dm = gop.value_ptr;
            dm.total = @bitCast(d.total);
            dm.external = @max(0, dm.total -% @as(i64, @bitCast(d.free)) -% self_bytes -% tracked);
            const z = try zs(gpa, d.name);
            defer gpa.free(std.mem.span(z));
            srv(.dbg, "calibrate", "device %s: total=%lld MiB, external=%lld MiB, available=%lld MiB\n", .{
                z,
                @as(c_longlong, @divTrunc(dm.total, 1024 * 1024)),
                @as(c_longlong, @divTrunc(dm.external, 1024 * 1024)),
                @as(c_longlong, @divTrunc(self.available(d.name), 1024 * 1024)),
            });
        }
    }

    /// Ports `reserve` (server-models-sched.cpp:534 @cd959bae9), a member of
    /// `server_models_sched`: count the instance's memory, effective, else
    /// min.
    fn reserve(self: *Sched, name: []const u8) !void {
        const it = self.models.find(name) orelse return;
        if (self.held.contains(name)) {
            return;
        }
        const info = &it.meta.memory;
        // prefer real usage over the estimate
        const mem: *const ServerModelMemory = if (info.effective) |*e| e else if (info.min) |*m| m else return;
        var devs: std.ArrayList(Bytes) = .empty;
        for (mem.devices) |*d| {
            if (d.isDevice() and d.total > 0) {
                var found = false;
                for (devs.items) |*x| {
                    if (std.mem.eql(u8, x.dev, d.name)) {
                        x.bytes = d.selfBytes();
                        found = true;
                    }
                }
                if (!found) try devs.append(gpa, .{ .dev = try gpa.dupe(u8, d.name), .bytes = d.selfBytes() });
            }
        }
        try self.held.put(gpa, try gpa.dupe(u8, name), devs);
    }

    /// Ports `release` (server-models-sched.cpp:553 @cd959bae9), a member of
    /// `server_models_sched`.
    fn release(self: *Sched, name: []const u8) void {
        const kv = self.held.fetchRemove(name) orelse return;
        gpa.free(kv.key);
        var devs = kv.value;
        for (devs.items) |x| gpa.free(x.dev);
        devs.deinit(gpa);
    }

    /// Ports `available` (server-models-sched.cpp:557 @cd959bae9), a member
    /// of `server_models_sched`: total - external - what the instances hold.
    fn available(self: *const Sched, dev: []const u8) i64 {
        const dm = self.device_memory.get(dev) orelse {
            return std.math.maxInt(i64); // never reported, nothing to check against
        };
        var n_held: i64 = 0;
        var it = self.held.iterator();
        while (it.next()) |e| {
            for (e.value_ptr.items) |h| {
                if (std.mem.eql(u8, h.dev, dev)) {
                    n_held += @bitCast(h.bytes);
                }
            }
        }
        return dm.total - dm.external - n_held;
    }

    /// Ports `deficit` (server-models-sched.cpp:572 @cd959bae9), a member of
    /// `server_models_sched`: the bytes missing per device to load `name` at
    /// its min, in device name order, empty if it fits.
    fn deficit(self: *const Sched, name: []const u8, res: *std.ArrayList(Bytes)) Allocator.Error!void {
        const it = self.models.find(name) orelse return;
        const mn = if (it.meta.memory.min) |*m| m else return;
        const margin: i64 = @as(i64, self.models.base_params.models_memory_margin) * 1024 * 1024;
        for (mn.devices) |*d| {
            // host memory is assumed unlimited, same as --fit
            if (!d.isDevice() or d.total == 0) {
                continue;
            }
            if (!self.device_memory.contains(d.name)) {
                continue;
            }
            const avail = self.available(d.name) - margin;
            const need: i64 = @bitCast(d.selfBytes());
            if (need > avail) {
                const bytes: usize = @bitCast(need - avail);
                var i: usize = 0;
                while (i < res.items.len and std.mem.order(u8, res.items[i].dev, d.name) == .lt) i += 1;
                if (i < res.items.len and std.mem.eql(u8, res.items[i].dev, d.name)) {
                    res.items[i].bytes = bytes;
                } else {
                    try res.insert(gpa, i, .{ .dev = d.name, .bytes = bytes });
                }
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a device's memory goes through its JSON form" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const mem: ServerModelMemory = .{ .n_ctx = 512, .devices = &.{
        .{ .kind = "device", .name = "MTL0", .total = 100, .free = 90, .model = 1, .context = 2, .compute = 3 },
        .{ .kind = "host", .name = "Host", .model = 4 },
    } };
    var v = try mem.toJson(a);
    const back = try ServerModelMemory.fromJson(a, &v);
    try std.testing.expectEqual(@as(u32, 512), back.n_ctx);
    try std.testing.expectEqual(@as(usize, 2), back.devices.len);
    try std.testing.expectEqual(@as(usize, 6), back.devices[0].selfBytes());
    try std.testing.expectEqual(@as(usize, 100), back.devices[0].total);
    try std.testing.expectEqual(@as(usize, 0), back.devices[1].total);
}
