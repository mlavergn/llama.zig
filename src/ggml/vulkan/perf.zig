//! The per-op GPU timings `GGML_VK_PERF_LOGGER` prints.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` and
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-types.h` at v1.0.0
//! (`b1b770d90`). The C++ keys `std::map`s by name and prints them in that order;
//! the port keeps the entries unordered and sorts the names when it prints,
//! which is the same order.

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const allocator = m.common.allocator;

fn oom() noreturn {
    impl.abort("ggml_vulkan: out of memory");
}

/// Ports `vk_perf_logger` (ggml-vulkan-types.h:1236 @b1b770d90).
pub const PerfLogger = struct {
    timings: std.StringHashMapUnmanaged(std.ArrayList(u64)) = .empty,
    flops: std.StringHashMapUnmanaged(std.ArrayList(u64)) = .empty,
    print_count: u32 = 0,

    pub fn create() *PerfLogger {
        const p = allocator.create(PerfLogger) catch oom();
        p.* = .{};
        return p;
    }

    fn clearMap(map: *std.StringHashMapUnmanaged(std.ArrayList(u64))) void {
        var it = map.iterator();
        while (it.next()) |e| {
            allocator.free(e.key_ptr.*);
            e.value_ptr.deinit(allocator);
        }
        map.clearRetainingCapacity();
    }

    /// Ports `vk_perf_logger`'s `print_timings` (ggml-vulkan.cpp:17011 @b1b770d90).
    pub fn printTimings(self: *PerfLogger, force: bool) void {
        if (self.timings.count() == 0) return;
        self.print_count += 1;
        if (self.print_count % m.instance.loggers.perf_logger_frequency != 0 and !force) return;
        self.print_count = 0;

        const names = allocator.alloc([]const u8, self.timings.count()) catch oom();
        defer allocator.free(names);
        var it = self.timings.keyIterator();
        var n: usize = 0;
        while (it.next()) |k| : (n += 1) names[n] = k.*;
        std.mem.sort([]const u8, names, {}, struct {
            fn lt(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.lt);

        var total_all_op_times: u64 = 0;
        impl.fprintfStderr("----------------\nVulkan Timings:\n", .{});
        for (names) |name| {
            const times = self.timings.get(name).?.items;
            var total_op_times: u64 = 0;
            for (times) |t| total_op_times +%= t;
            const line_fmt = "%.*s: %zu x %g us = %g us";
            impl.fprintfStderr(line_fmt, .{ @as(c_int, @intCast(name.len)), name.ptr, times.len, @as(f64, @floatFromInt(total_op_times / times.len)) / 1000.0, @as(f64, @floatFromInt(total_op_times)) / 1000.0 });

            // If there are as many flops entries as timing entries, log the flops/s.
            if (self.flops.get(name)) |fl| {
                if (fl.items.len == times.len) {
                    var total_op_flops: u64 = 0;
                    for (fl.items) |f| total_op_flops +%= f;
                    impl.fprintfStderr(" (%g GFLOPS/s)", .{(@as(f64, @floatFromInt(total_op_flops)) / (1000.0 * 1000.0 * 1000.0)) / (@as(f64, @floatFromInt(total_op_times)) / (1000.0 * 1000.0 * 1000.0))});
                }
            }
            total_all_op_times +%= total_op_times;
            impl.fprintfStderr("\n", .{});
        }
        impl.fprintfStderr("Total time: %g us.\n", .{@as(f64, @floatFromInt(total_all_op_times)) / 1000.0});

        clearMap(&self.timings);
        clearMap(&self.flops);
    }

    /// Ports `vk_perf_logger`'s `get_node_fusion_name` (ggml-vulkan.cpp:17056 @b1b770d90):
    /// the label a node's timing is filed under, with its flops.
    fn nodeFusionName(w: *std.Io.Writer, node: *const c.ggml_tensor, fusion_name: ?[*:0]const u8, n_flops: *u64) void {
        n_flops.* = m.graph.getNodeFlops(node);
        if (fusion_name) |f| w.print("{s} ", .{std.mem.span(f)}) catch {};
        const op_name = std.mem.span(c.ggml_op_name(node.op));
        if (node.op == c.GGML_OP_UNARY) {
            w.writeAll(std.mem.span(c.ggml_unary_op_name(c.ggml_get_unary_op(node)))) catch {};
            return;
        }
        if (node.op == c.GGML_OP_MUL_MAT or node.op == c.GGML_OP_MUL_MAT_ID) {
            const mm: u64 = @bitCast(node.ne[0]);
            const n: u64 = @bitCast(node.ne[1]);
            const k: u64 = @bitCast(node.src[1].*.ne[0]);
            const batch: u64 = @bitCast(node.ne[2] *% node.ne[3]);
            w.writeAll(op_name) catch {};
            if ((node.op == c.GGML_OP_MUL_MAT and n <= m.device.mul_mat_vec_max_cols) or
                (node.op == c.GGML_OP_MUL_MAT_ID and node.src[2].*.ne[1] == 1))
            {
                w.writeAll("_VEC") catch {};
            }
            w.print(" {s} m={d} n={d} k={d}", .{ std.mem.span(c.ggml_type_name(node.src[0].*.type)), mm, n, k }) catch {};
            if (node.op == c.GGML_OP_MUL_MAT_ID) w.print(" n_expert={d}", .{node.src[0].*.ne[2]}) catch {};
            if (batch > 1) w.print(" batch={d}", .{batch}) catch {};
            return;
        }
        if (node.op == c.GGML_OP_CONV_2D or node.op == c.GGML_OP_CONV_TRANSPOSE_2D) {
            const knl = node.src[0].*;
            const cout: u64 = @bitCast(node.ne[2]);
            const size_K: u64 = @bitCast(node.src[1].*.ne[2] *% knl.ne[0] *% knl.ne[1]);
            const size_N: u64 = @bitCast(node.ne[3] *% node.ne[0] *% node.ne[1]);
            w.print("{s} M=Cout={d}, K=Cin*KW*KH={d}, N=N*OW*OH={d}", .{ op_name, cout, size_K, size_N }) catch {};
            return;
        }
        if (node.op == c.GGML_OP_RMS_NORM) {
            w.print("{s}({d},{d},{d},{d})", .{ op_name, node.ne[0], node.ne[1], node.ne[2], node.ne[3] }) catch {};
            return;
        }
        if (node.op == c.GGML_OP_FLASH_ATTN_EXT) {
            const q = node.src[0].*;
            const k = node.src[1].*;
            const v = node.src[2].*;
            const mk: ?*const c.ggml_tensor = node.src[3];
            const me = if (mk) |t| t.ne else [4]i64{ 0, 0, 0, 0 };
            w.print("{s} dst({d},{d},{d},{d}),  q({d},{d},{d},{d}),  k({d},{d},{d},{d}),  v({d},{d},{d},{d}),  m({d},{d},{d},{d})", .{
                op_name, node.ne[0], node.ne[1], node.ne[2], node.ne[3],
                q.ne[0], q.ne[1],    q.ne[2],    q.ne[3],    k.ne[0],
                k.ne[1], k.ne[2],    k.ne[3],    v.ne[0],    v.ne[1],
                v.ne[2], v.ne[3],    me[0],      me[1],      me[2],
                me[3],
            }) catch {};
            return;
        }
        if (node.op == c.GGML_OP_TOP_K) {
            const s = node.src[0].*;
            w.print("{s} K={d} ({d},{d},{d},{d})", .{ op_name, node.ne[0], s.ne[0], s.ne[1], s.ne[2], s.ne[3] }) catch {};
            return;
        }
        w.writeAll(op_name) catch {};
    }

    fn file(map: *std.StringHashMapUnmanaged(std.ArrayList(u64)), name: []const u8, value: u64) void {
        const gop = map.getOrPut(allocator, name) catch oom();
        if (!gop.found_existing) {
            gop.key_ptr.* = allocator.dupe(u8, name) catch oom();
            gop.value_ptr.* = .empty;
        }
        gop.value_ptr.append(allocator, value) catch oom();
    }

    /// Ports `vk_perf_logger`'s `log_timing` (ggml-vulkan-types.h:1253 @b1b770d90)
    /// for a fused run of nodes; one node is the first overload.
    pub fn logTiming(self: *PerfLogger, nodes: []const *c.ggml_tensor, names: []const ?[*:0]const u8, time: u64) void {
        var buf: [4096]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        var total_flops: u64 = 0;
        for (nodes, 0..) |node, i| {
            var n_flops: u64 = 0;
            nodeFusionName(&w, node, names[i], &n_flops);
            total_flops +%= n_flops;
            if (i != nodes.len - 1) w.writeAll(", ") catch {};
        }
        const name = w.buffered();
        if (total_flops != 0) file(&self.flops, name, total_flops);
        file(&self.timings, name, time);
    }
};

// -----------------------------------------------------------------------------
// The test-fusion interface (`tests/test-fusion.cpp`), reached through
// `ggml_backend_reg_get_proc_address`

const Device = m.device.Device;

/// Ports `ggml_vk_count_fusion` (ggml-vulkan-debug.cpp:15 @b1b770d90): one
/// more use of the fusion `label`, when the stats are on.
pub fn countFusion(device: *Device, label: []const u8) void {
    if (!device.fusion_stats_enabled.load(.monotonic)) return;
    device.fusion_mutex.lock();
    defer device.fusion_mutex.unlock();
    // `device->fusion_counts[label]++`, kept in the map's order.
    const items = device.fusion_counts.items;
    var lo: usize = 0;
    var hi: usize = items.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        switch (std.mem.order(u8, items[mid].label, label)) {
            .lt => lo = mid + 1,
            .gt => hi = mid,
            .eq => {
                items[mid].count += 1;
                return;
            },
        }
    }
    const owned = allocator.dupeSentinel(u8, label, 0) catch oom();
    device.fusion_counts.insert(allocator, lo, .{ .label = owned, .count = 1 }) catch oom();
}

/// Ports `ggml_backend_vk_fusion_stats_init` (ggml-vulkan-debug.cpp:28 @b1b770d90).
pub fn fusionStatsInit(finfo: ?*anyopaque) callconv(.c) void {
    const device: *Device = @ptrCast(@alignCast(finfo.?));
    device.fusion_stats_enabled.store(true, .monotonic);
}

/// Ports `ggml_backend_vk_fusion_stats_reset` (ggml-vulkan-debug.cpp:33 @b1b770d90).
pub fn fusionStatsReset(finfo: ?*anyopaque) callconv(.c) void {
    const device: *Device = @ptrCast(@alignCast(finfo.?));
    device.fusion_mutex.lock();
    defer device.fusion_mutex.unlock();
    // Keep labels and their order stable between fused and unfused runs.
    for (device.fusion_counts.items) |*e| e.count = 0;
}

/// Ports `ggml_backend_vk_fusion_stats_get` (ggml-vulkan-debug.cpp:42 @b1b770d90):
/// the label count when `labels` is null, else up to `n` labels and counts
/// in label order.
pub fn fusionStatsGet(finfo: ?*anyopaque, labels: [*c][*c]const u8, counts: [*c]u64, n: c_int) callconv(.c) c_int {
    const device: *Device = @ptrCast(@alignCast(finfo.?));
    device.fusion_mutex.lock();
    defer device.fusion_mutex.unlock();
    if (labels == null) return @intCast(device.fusion_counts.items.len);
    var i: c_int = 0;
    for (device.fusion_counts.items) |e| {
        if (i >= n) break;
        const k: usize = @intCast(i);
        labels[k] = e.label.ptr;
        if (counts != null) counts[k] = e.count;
        i += 1;
    }
    return i;
}

/// Ports `ggml_backend_vk_fusion_set_enabled` (ggml-vulkan-debug.cpp:62 @b1b770d90).
pub fn fusionSetEnabled(finfo: ?*anyopaque, enabled: bool) callconv(.c) void {
    const device: *Device = @ptrCast(@alignCast(finfo.?));
    device.disable_fusion = !enabled;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
