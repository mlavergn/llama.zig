//! Fits a model's and a context's unset parameters to free memory before
//! anything is loaded, as upstream's `llama-cli` does by default (`--fit
//! on`).
//!
//! # Provenance
//!
//! Ported from `llama.cpp/common/fit.cpp` and `llama.cpp/common/fit.h`
//! (v0.4.0, `5266f24da`), the whole file, and from the two structs of
//! `llama.cpp/src/llama-ext.h` it measures with.
//!
//! # Why the CLI needs it
//!
//! `-c 0`, the default, asks for the context the model was trained with --
//! 40,960 tokens for Qwen3, 262,144 for Qwen3.5. Upstream then measures what
//! that would cost on each device (a `no_alloc` load of the model and a
//! context), and where it would leave less than a margin free (1 GiB by
//! default) it shrinks the context, down to 4096, and on GPUs moves layers
//! and tensors back to system memory. Without it a 4.6 GB KV cache was asked
//! of a 3.6 GB board: on Linux the allocation succeeds under overcommit, and
//! the process is killed when the cache is first cleared.
//!
//! # How it reaches libllama
//!
//! `fit.cpp` includes `../src/llama-ext.h`, libllama's internal header: the
//! memory breakdown it reads is a `std::map` with no C signature. This file
//! does the same thing the Zig way, importing libllama's own module as
//! `llama` and calling `api/`'s Zig functions directly -- the CLI compiles
//! libllama into itself rather than linking its archive, so every call here
//! reaches the same code the `llama.h` exports do. `harness/fit_diff.zig`
//! binds `llama` to `harness_root.zig` instead and runs this beside the
//! C++.
//!
//! # Logging
//!
//! Upstream writes through `common_log`, at a verbosity threshold the CLI
//! sets to `LOG_LEVEL_ERROR`: only the error line reaches the terminal, the
//! trace and the warning do not. `Log` is that threshold and that output,
//! and formats every line with `snprintf` and the C's own format string, as
//! libllama's logger does -- Zig's `{d:6}` prints a `+` on a signed integer
//! and rounds 0.125 to two places as 0.13, where `printf` gives 0.12.

const std = @import("std");
const llama = @import("llama");

const api = llama.api;
const c = llama.impl.c;

extern "c" fn snprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, ...) c_int;

const MiB: i64 = 1024 * 1024;

/// `__func__` inside a C++ lambda.
const lambda: [*:0]const u8 = "operator()";

/// `common_log_get_verbosity_thold()` and where `common_log` writes: the
/// part of upstream's logger this file's output depends on.
pub const Log = struct {
    /// `LOG_LEVEL_TRACE` (log.h:25 @5266f24da).
    pub const level_trace: i32 = 4;
    /// `LOG_LEVEL_WARN` (log.h:27 @5266f24da).
    pub const level_warn: i32 = 2;
    /// `LOG_LEVEL_ERROR` (log.h:28 @5266f24da).
    pub const level_error: i32 = 1;

    /// Messages whose verbosity is above this are dropped. `llama-cli`
    /// sets `LOG_LEVEL_ERROR` (tools/cli/cli.cpp:36 @5266f24da).
    verbosity: i32 = level_error,
    /// Where messages go; null drops them all.
    out: ?*std.Io.Writer = null,
    /// `common_log_set_prefix`: a level letter before each message, which
    /// the harness turns on so the level is part of what it compares.
    prefix: bool = false,
    /// Where messages go instead of `out`, with the verbosity they were
    /// logged at: the tools hand them to their own `common_log`, which keeps
    /// its threshold, prefix and timestamps.
    sink: ?*const fn (verbosity: i32, text: [*:0]const u8) void = null,

    /// `LOG_TRC` (log.h:117 @5266f24da).
    fn trc(self: Log, format: [*:0]const u8, args: anytype) void {
        self.emit(level_trace, "I ", format, args);
    }

    /// `LOG_WRN` (log.h:119 @5266f24da).
    fn wrn(self: Log, format: [*:0]const u8, args: anytype) void {
        self.emit(level_warn, "W ", format, args);
    }

    /// `LOG_ERR` (log.h:120 @5266f24da).
    fn err(self: Log, format: [*:0]const u8, args: anytype) void {
        self.emit(level_error, "E ", format, args);
    }

    /// Formats one message as `common_log_add` does and writes it, flushed,
    /// as `common_log_entry::print` does. A failed write is dropped, as the
    /// C's `fprintf` result is.
    fn emit(self: Log, verbosity: i32, letter: []const u8, format: [*:0]const u8, args: anytype) void {
        if (verbosity > self.verbosity) return;
        if (self.out == null and self.sink == null) return;
        var stack: [1024]u8 = undefined;
        const n: usize = @intCast(@call(.auto, snprintf, .{ &stack, stack.len, format } ++ args));
        var heap: ?[]u8 = null;
        defer if (heap) |h| std.heap.c_allocator.free(h);
        const text = if (n < stack.len) stack[0..n] else blk: {
            const h = std.heap.c_allocator.alloc(u8, n + 1) catch return;
            heap = h;
            _ = @call(.auto, snprintf, .{ h.ptr, n + 1, format } ++ args);
            break :blk h[0..n];
        };
        if (self.sink) |sink| {
            const z = std.heap.c_allocator.dupeSentinel(u8, text, 0) catch return;
            defer std.heap.c_allocator.free(z);
            sink(verbosity, z.ptr);
            return;
        }
        const out = self.out.?;
        if (self.prefix) out.writeAll(letter) catch return;
        out.writeAll(text) catch return;
        out.flush() catch return;
    }
};

/// Ports `common_params_fit_status` (fit.h:8 @5266f24da).
pub const Status = enum(c_int) {
    /// Found allocations that are projected to fit.
    success = 0,
    /// Could not find allocations that are projected to fit.
    failure = 1,
    /// A hard error occurred, e.g. no model at the path.
    @"error" = 2,
};

/// Ports `common_fit_extra_model` (fit.h:17 @5266f24da): a second model
/// sharing the main model's devices, as a draft model does. Its context
/// follows the main one's, so it is measured again whenever that changes.
pub const ExtraModel = struct {
    path_model: [*:0]const u8,
    mparams: *c.llama_model_params,
    cparams: *c.llama_context_params,
    /// The weights are already counted in the main model, as for an MTP
    /// context.
    shares_model: bool,
};

/// Ports `llama_memory_breakdown_data` (llama-ext.h:67 @5266f24da).
pub const MemoryBreakdownData = struct {
    model: usize = 0,
    context: usize = 0,
    compute: usize = 0,

    fn total(self: MemoryBreakdownData) usize {
        return self.model +% self.context +% self.compute;
    }
};

/// Ports `llama_device_memory_data` (llama-ext.h:77 @5266f24da).
pub const LlamaDeviceMemoryData = struct {
    total: i64 = 0,
    free: i64 = 0,
    mb: MemoryBreakdownData = .{},
};

/// Ports `common_device_memory_data` (fit.h:48 @5266f24da).
pub const DeviceMemoryData = struct {
    total: i64,
    free: i64,
    model: usize,
    context: usize,
    compute: usize,
};

/// Ports `common_layer_fraction_t` (fit.cpp:17 @5266f24da): which part of a
/// layer's tensors overflows to the next device or system memory.
const LayerFraction = enum(c_int) {
    none = 0,
    /// Attention only stays.
    attn = 1,
    /// Attention and up stay.
    up = 2,
    /// Attention, up and gate stay.
    gate = 3,
    /// Everything but the sparse MoE weights stays.
    moe = 4,
};

/// The tensor-override patterns `common_params_fit_impl` hands to
/// `llama_model_params`, one per layer and fraction. The C keeps them in
/// function-local `static std::array<std::string, 1000>`s, alive for the
/// process; here the caller owns them, and they must outlive the model load
/// that reads them.
pub const Patterns = struct {
    allocator: std.mem.Allocator,
    map: std.AutoHashMapUnmanaged(u32, [:0]u8) = .empty,

    /// Return: an empty cache. Reserved for the lifecycle convention.
    pub fn init(allocator: std.mem.Allocator) !Patterns {
        return .{ .allocator = allocator };
    }

    /// Frees every pattern; overrides that point at them are invalid after.
    pub fn deinit(self: *Patterns) void {
        var it = self.map.valueIterator();
        while (it.next()) |p| self.allocator.free(p.*);
        self.map.deinit(self.allocator);
    }
};

/// Ports `common_params_fit_exception` (fit.cpp:25 @5266f24da) as
/// `error.Fit`, and the `std::runtime_error`s it derives from as
/// `error.Runtime`; the message is `Fitter.what`.
const Failure = error{ Fit, Runtime, OutOfMemory };

/// One `common_fit_params` call: its scratch allocator, its log, and the
/// message of the exception in flight.
const Fitter = struct {
    arena: std.mem.Allocator,
    patterns: *Patterns,
    log: Log,
    what: [:0]const u8 = "",

    fn fail(self: *Fitter, kind: Failure, comptime format: []const u8, args: anytype) Failure {
        self.what = try std.fmt.allocPrintSentinel(self.arena, format, args, 0);
        return kind;
    }
};

/// What `common_get_device_memory_data_impl` installs as libllama's logger
/// while it measures: messages below `min_level` are demoted to debug.
const LogFilter = struct {
    original_callback: c.ggml_log_callback,
    original_user_data: ?*anyopaque,
    min_level: c.enum_ggml_log_level,

    /// Ports the lambda `common_get_device_memory_data_impl` passes to
    /// `llama_log_set` (fit.cpp:49 @5266f24da).
    fn callback(level: c.enum_ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
        const ud: *const LogFilter = @ptrCast(@alignCast(user_data));
        const level_eff = if (level >= ud.min_level) level else c.GGML_LOG_LEVEL_DEBUG;
        ud.original_callback.?(level_eff, text, ud.original_user_data);
    }
};

/// Ports `common_get_device_memory_data_impl` (fit.cpp:29 @5266f24da):
/// loads the model and a context with `no_alloc` and reads what each device,
/// and the host, would hold.
///
/// Return: one entry per model device and a last one for the host, from
/// `f.arena`; `devs` and the `hp_*` outputs are set as in the C.
fn deviceMemoryDataImpl(
    f: *Fitter,
    path_model: [*:0]const u8,
    mparams: *const c.llama_model_params,
    cparams: *const c.llama_context_params,
    devs: *std.ArrayList(c.ggml_backend_dev_t),
    hp_ngl: *u32,
    hp_n_ctx_train: *u32,
    hp_n_expert: *u32,
    log_level: c.enum_ggml_log_level,
) Failure![]LlamaDeviceMemoryData {
    var ud: LogFilter = .{ .original_callback = null, .original_user_data = null, .min_level = log_level };
    api.impl.llama_log_get(&ud.original_callback, &ud.original_user_data);
    api.impl.llama_log_set(LogFilter.callback, &ud);
    defer api.impl.llama_log_set(ud.original_callback, ud.original_user_data);

    var mparams_copy = mparams.*;
    mparams_copy.no_alloc = true;
    mparams_copy.load_mode = c.LLAMA_LOAD_MODE_NONE;

    const model = api.llama.llama_model_load_from_file(path_model, mparams_copy) orelse
        return f.fail(error.Runtime, "failed to load model", .{});
    defer api.model.llama_model_free(model);

    const ctx = api.context.llama_init_from_model(model, cparams.*) orelse
        return f.fail(error.Runtime, "failed to create llama_context from model", .{});
    defer api.context.llama_free(ctx);

    const nd: usize = @intCast(api.model.llama_model_n_devices(model));
    const ret = try f.arena.alloc(LlamaDeviceMemoryData, nd + 1);
    @memset(ret, .{});
    const host = &ret[nd];

    const memory_breakdown = try api.context.llama_get_memory_breakdown(ctx, f.arena);
    for (memory_breakdown) |e| {
        if (c.ggml_backend_buft_is_host(e.buft)) {
            host.mb.model +%= e.model;
            host.mb.context +%= e.context;
            host.mb.compute +%= e.compute;
            continue;
        }
        const dev = c.ggml_backend_buft_get_device(e.buft) orelse continue;
        for (0..nd) |i| {
            if (dev == api.model.llama_model_get_device(model, @intCast(i))) {
                ret[i].mb.model +%= e.model;
                ret[i].mb.context +%= e.context;
                ret[i].mb.compute +%= e.compute;
                break;
            }
        }
    }

    {
        // The C throws here with the context, the model and its own logger
        // still in place, the logger's user data on a frame about to unwind.
        // The defers above release them instead. Unreachable in practice:
        // the CPU backend is always registered.
        const cpu_dev = c.ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_CPU) orelse
            return f.fail(error.Runtime, "no CPU backend found", .{});
        var free: usize = 0;
        var total: usize = 0;
        c.ggml_backend_dev_memory(cpu_dev, &free, &total);
        host.free = @bitCast(free);
        host.total = @bitCast(total);
    }
    for (0..nd) |i| {
        const dev = api.model.llama_model_get_device(model, @intCast(i));

        var free: usize = 0;
        var total: usize = 0;
        c.ggml_backend_dev_memory(dev, &free, &total);

        // Some non-GPU accelerator backends, such as BLAS, report 0/0 and
        // rely on the host-memory fallback. A GPU-like backend keeps 0/0, so
        // that nothing is assigned to a device with an unknown budget.
        if (free == 0 and total == 0) {
            const t = c.ggml_backend_dev_type(dev);
            if (t == c.GGML_BACKEND_DEVICE_TYPE_GPU or t == c.GGML_BACKEND_DEVICE_TYPE_IGPU) {
                f.log.wrn("%s: device %s did not report memory; --fit will not use it\n", .{ @as([*:0]const u8, "common_get_device_memory_data_impl"), c.ggml_backend_dev_name(dev) });
            } else {
                free = @bitCast(host.free);
                total = @bitCast(host.total);
            }
        }
        ret[i].free = @bitCast(free);
        ret[i].total = @bitCast(total);
    }

    devs.clearRetainingCapacity();
    for (0..nd) |i| try devs.append(f.arena, api.model.llama_model_get_device(model, @intCast(i)));

    hp_ngl.* = @bitCast(api.model.llama_model_n_layer(model));
    if (mparams.load_mtp) hp_ngl.* +%= @bitCast(api.model.llama_model_n_layer_nextn(model));
    hp_n_ctx_train.* = @bitCast(api.model.llama_model_n_ctx_train(model));
    hp_n_expert.* = @bitCast(api.model.llama_model_n_expert(model));

    try memoryBreakdownPrint(f.arena, f.log, ctx);

    return ret;
}

/// Ports `common_get_device_memory_data` (fit.cpp:155 @5266f24da): the
/// memory each device and the host would hold under `mparams` and
/// `cparams`, without loading anything.
///
/// Parameters:
/// - `gpa`: allocates the result and the scratch.
/// - `log`: where the trace and warnings go.
/// - `path_model`: the model file.
/// - `mparams`, `cparams`: the parameters to measure under.
/// - `devs`: set to the model's devices, allocated with `gpa`.
/// - `hp_ngl`, `hp_n_ctx_train`, `hp_n_expert`: set from the model.
/// - `log_level`: libllama's messages below this go to the debug level.
///
/// Return: one entry per device and a last one for the host, the caller's
/// to free with `gpa`; `error.Runtime` where the C throws, with the message
/// written to `log` at the error level since there is no exception to carry
/// it.
pub fn getDeviceMemoryData(
    gpa: std.mem.Allocator,
    log: Log,
    path_model: [*:0]const u8,
    mparams: *const c.llama_model_params,
    cparams: *const c.llama_context_params,
    devs: *std.ArrayList(c.ggml_backend_dev_t),
    hp_ngl: *u32,
    hp_n_ctx_train: *u32,
    hp_n_expert: *u32,
    log_level: c.enum_ggml_log_level,
) ![]DeviceMemoryData {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var patterns: Patterns = try .init(gpa);
    defer patterns.deinit();
    var f: Fitter = .{ .arena = arena_state.allocator(), .patterns = &patterns, .log = log };

    var arena_devs: std.ArrayList(c.ggml_backend_dev_t) = .empty;
    const impl_ret = deviceMemoryDataImpl(&f, path_model, mparams, cparams, &arena_devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level) catch |e| switch (e) {
        error.Fit, error.Runtime => {
            log.err("%s\n", .{f.what.ptr});
            return error.Runtime;
        },
        error.OutOfMemory => return error.OutOfMemory,
    };
    devs.clearRetainingCapacity();
    try devs.appendSlice(gpa, arena_devs.items);

    const ret = try gpa.alloc(DeviceMemoryData, impl_ret.len);
    for (impl_ret, ret) |s, *d| d.* = .{
        .total = s.total,
        .free = s.free,
        .model = s.mb.model,
        .context = s.mb.context,
        .compute = s.mb.compute,
    };
    return ret;
}

/// Ports `ngl_t` (fit.cpp:531 @5266f24da): how many layers a device gets,
/// and how many of them only partly.
const Ngl = struct {
    /// Total layers.
    n_layer: u32 = 0,
    /// Partial layers, <= `n_layer`.
    n_part: u32 = 0,
    /// What overflows in the first partial layer; every further one
    /// overflows `moe`.
    overflow_type: LayerFraction = .moe,

    fn nFull(self: Ngl) u32 {
        cAssert(self.n_layer >= self.n_part);
        return self.n_layer -% self.n_part;
    }
};

/// `common_params_fit_impl`'s state, shared by the lambdas the C captures by
/// reference.
const FitImpl = struct {
    f: *Fitter,
    path_model: [*:0]const u8,
    mparams: *c.llama_model_params,
    cparams: *c.llama_context_params,
    tensor_split: ?[*]f32,
    tensor_buft_overrides: ?[*]c.llama_model_tensor_buft_override,
    extra: ?*const ExtraModel,
    log_level: c.enum_ggml_log_level,

    devs: std.ArrayList(c.ggml_backend_dev_t) = .empty,
    hp_ngl: u32 = 0,
    hp_nct: u32 = 0,
    hp_nex: u32 = 0,
    nd: usize = 0,
    ntbo: usize = 0,

    /// Memory of the extra model, laid out on the main model's devices.
    dmds_extra: []LlamaDeviceMemoryData = &.{},
    /// The context that memory was measured at.
    n_ctx_extra: u32 = 0,

    fn measure(self: *FitImpl, mparams: *const c.llama_model_params) Failure![]LlamaDeviceMemoryData {
        return deviceMemoryDataImpl(self.f, self.path_model, mparams, self.cparams, &self.devs, &self.hp_ngl, &self.hp_nct, &self.hp_nex, self.log_level);
    }

    /// Ports the lambda `add_extra_memory` (fit.cpp:204 @5266f24da): the
    /// extra model competes for the same memory, so it is added to every
    /// measurement, and measured again whenever its context changes. Its
    /// messages name `operator()`, which is what `__func__` is inside a C++
    /// lambda.
    fn addExtraMemory(self: *FitImpl, dmds: []LlamaDeviceMemoryData) Failure!void {
        const extra = self.extra orelse return;

        if (self.dmds_extra.len == 0 or self.n_ctx_extra != self.cparams.n_ctx) {
            var devs_extra: std.ArrayList(c.ggml_backend_dev_t) = .empty;
            var ngl_extra: u32 = 0;
            var nct_extra: u32 = 0;
            var nex_extra: u32 = 0;

            extra.cparams.n_ctx = self.cparams.n_ctx;

            self.f.log.trc("%s: getting device memory data for the extra model at a context size of %u:\n", .{ lambda, self.cparams.n_ctx });

            const measured = deviceMemoryDataImpl(self.f, extra.path_model, extra.mparams, extra.cparams, &devs_extra, &ngl_extra, &nct_extra, &nex_extra, self.log_level) catch |e| switch (e) {
                error.Fit, error.Runtime => {
                    // The extra model is optional: fit the main model alone
                    // rather than give up.
                    self.f.log.wrn("%s: failed to measure the memory of the extra model, fitting without it: %s\n", .{ lambda, self.f.what.ptr });
                    self.dmds_extra = try self.f.arena.alloc(LlamaDeviceMemoryData, self.devs.items.len + 1);
                    @memset(self.dmds_extra, .{});
                    self.n_ctx_extra = self.cparams.n_ctx;
                    return;
                },
                error.OutOfMemory => return error.OutOfMemory,
            };

            self.dmds_extra = try self.f.arena.alloc(LlamaDeviceMemoryData, self.devs.items.len + 1);
            @memset(self.dmds_extra, .{});
            self.dmds_extra[self.dmds_extra.len - 1].mb = measured[measured.len - 1].mb;
            for (devs_extra.items, 0..) |dev_extra, je| {
                for (self.devs.items, 0..) |dev, id| {
                    if (dev_extra == dev) {
                        self.dmds_extra[id].mb.model +%= measured[je].mb.model;
                        self.dmds_extra[id].mb.context +%= measured[je].mb.context;
                        self.dmds_extra[id].mb.compute +%= measured[je].mb.compute;
                        break;
                    }
                }
            }
            if (extra.shares_model) {
                for (self.dmds_extra) |*dmd| dmd.mb.model = 0;
            }

            self.n_ctx_extra = self.cparams.n_ctx;
        }

        for (dmds, 0..) |*dmd, id| {
            dmd.mb.model +%= self.dmds_extra[id].mb.model;
            dmd.mb.context +%= self.dmds_extra[id].mb.context;
            dmd.mb.compute +%= self.dmds_extra[id].mb.compute;
        }
    }

    /// Ports the lambda `get_overflow_pattern` (fit.cpp:492 @5266f24da):
    /// the regex matching one layer's tensors that overflow for a fraction.
    fn overflowPattern(self: *FitImpl, il: usize, lf: LayerFraction) Failure![*:0]const u8 {
        const n_strings = 1000;
        if (il >= n_strings) return self.f.fail(error.Runtime, "at most {d} model layers are supported", .{n_strings});
        const key: u32 = @as(u32, @intCast(@backingInt(lf))) * n_strings + @as(u32, @intCast(il));
        const patterns = self.f.patterns;
        if (patterns.map.get(key)) |p| return p.ptr;
        const p = switch (lf) {
            .attn => try std.fmt.allocPrintSentinel(patterns.allocator, "blk\\.{d}\\.ffn_(gate|up|gate_up|down).*", .{il}, 0),
            .up => try std.fmt.allocPrintSentinel(patterns.allocator, "blk\\.{d}\\.ffn_(gate|gate_up|down).*", .{il}, 0),
            .gate => try std.fmt.allocPrintSentinel(patterns.allocator, "blk\\.{d}\\.ffn_down.*", .{il}, 0),
            .moe => try std.fmt.allocPrintSentinel(patterns.allocator, "blk\\.{d}\\.ffn_(up|down|gate_up|gate)_(ch|)exps", .{il}, 0),
            .none => std.process.abort(), // GGML_ABORT("fatal error")
        };
        errdefer patterns.allocator.free(p);
        try patterns.map.put(patterns.allocator, key, p);
        return p.ptr;
    }

    /// Ports the lambda `set_ngl_tensor_split_tbo` (fit.cpp:547 @5266f24da):
    /// writes a layer assignment into `mparams` through the caller's
    /// `tensor_split` and override buffers.
    fn setNglTensorSplitTbo(self: *FitImpl, ngl_per_device: []const Ngl, overflow_bufts: []const c.ggml_backend_buffer_type_t, mparams: *c.llama_model_params) Failure!void {
        const nd = self.nd;
        var n_gpu_layers: u32 = 0;
        for (0..nd) |id| {
            n_gpu_layers +%= ngl_per_device[id].n_layer;
            if (nd > 1) self.tensor_split.?[id] = @floatFromInt(ngl_per_device[id].n_layer);
        }
        mparams.n_gpu_layers = @bitCast(n_gpu_layers);
        cAssert(n_gpu_layers <= self.hp_ngl +% 1);
        var il0: u32 = self.hp_ngl +% 1 -% n_gpu_layers; // start index for tensor buft overrides

        mparams.tensor_split = self.tensor_split;

        const tbo = self.tensor_buft_overrides.?;
        var itbo: usize = 0;
        for (0..nd) |id| {
            il0 +%= ngl_per_device[id].nFull();
            var il = il0;
            while (il < il0 +% ngl_per_device[id].n_part) : (il += 1) {
                if (itbo + 1 >= self.ntbo) {
                    tbo[itbo].pattern = null;
                    tbo[itbo].buft = null;
                    itbo += 1;
                    mparams.tensor_buft_overrides = tbo;
                    return self.f.fail(error.Fit, "llama_max_tensor_buft_overrides() == {d} is insufficient for model", .{self.ntbo});
                }
                tbo[itbo].pattern = try self.overflowPattern(il, if (il == il0) ngl_per_device[id].overflow_type else .moe);
                tbo[itbo].buft = if (il == il0) overflow_bufts[id] else c.ggml_backend_cpu_buffer_type();
                itbo += 1;
            }
            il0 +%= ngl_per_device[id].n_part;
        }
        tbo[itbo].pattern = null;
        tbo[itbo].buft = null;
        itbo += 1;
        mparams.tensor_buft_overrides = tbo;
    }

    /// Ports the lambda `get_memory_for_layers` (fit.cpp:588 @5266f24da): the
    /// memory each device would use for a layer assignment.
    fn memoryForLayers(self: *FitImpl, func_name: [*:0]const u8, ngl_per_device: []const Ngl, overflow_bufts: []const c.ggml_backend_buffer_type_t) Failure![]i64 {
        var mparams_copy = self.mparams.*;
        try self.setNglTensorSplitTbo(ngl_per_device, overflow_bufts, &mparams_copy);

        const dmd_nl = try self.measure(&mparams_copy);
        try self.addExtraMemory(dmd_nl);

        self.f.log.trc("%s: memory for test allocation by device:\n", .{func_name});
        for (0..self.nd) |id| {
            const n = ngl_per_device[id];
            self.f.log.trc(
                "%s: id=%zu, n_layer=%2u, n_part=%2u, overflow_type=%d, mem=%6lld MiB\n",
                .{ func_name, id, n.n_layer, n.n_part, @as(c_int, @backingInt(n.overflow_type)), divMiB(dmd_nl[id].mb.total()) },
            );
        }

        const ret = try self.f.arena.alloc(i64, self.nd);
        for (ret, 0..) |*r, id| r.* = @bitCast(dmd_nl[id].mb.total());
        return ret;
    }
};

/// Upstream's `assert`, as the reference compiles it: out, under `NDEBUG`
/// (upstream's Release build defines it, and so does `zig c++ -O2`). A
/// failed one changes nothing, and the unsigned arithmetic after it wraps as
/// the C's does -- measured: `gpu3-moe-40-10-30` in `scripts/fit-diff` fails
/// `n_full`'s on both sides and agrees on everything after. A
/// `std.debug.assert` there would be undefined behaviour in ReleaseFast.
fn cAssert(ok: bool) void {
    _ = ok;
}

/// `size / MiB` with the size first converted to `int64_t`, as each `%6"
/// PRId64` argument the C passes is.
fn divMiB(size: usize) i64 {
    return @divTrunc(@as(i64, @bitCast(size)), MiB);
}

/// Ports `common_params_fit_impl` (fit.cpp:178 @5266f24da).
fn fitImpl(
    f: *Fitter,
    path_model: [*:0]const u8,
    mparams: *c.llama_model_params,
    cparams: *c.llama_context_params,
    tensor_split: ?[*]f32,
    tensor_buft_overrides: ?[*]c.llama_model_tensor_buft_override,
    margins_s: [*]const usize,
    n_ctx_min: u32,
    extra: ?*const ExtraModel,
    log_level: c.enum_ggml_log_level,
) Failure!void {
    const func: [*:0]const u8 = "common_params_fit_impl";
    const log = f.log;
    if (mparams.split_mode == c.LLAMA_SPLIT_MODE_TENSOR) {
        return f.fail(error.Fit, "llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort", .{});
    }
    const default_mparams = api.model.llama_model_default_params();

    var s: FitImpl = .{
        .f = f,
        .path_model = path_model,
        .mparams = mparams,
        .cparams = cparams,
        .tensor_split = tensor_split,
        .tensor_buft_overrides = tensor_buft_overrides,
        .extra = extra,
        .log_level = log_level,
    };

    // With a non-unified KV cache n_streams has to be taken into account:
    // where memory can hold more than the model's trained context, n_ctx
    // must grow to hold enough streams.
    const n_streams: u32 = if (cparams.kv_unified) 1 else @max(1, cparams.n_seq_max);
    const n_ctx_auto = cparams.n_ctx == 0;

    // step 1: get data for default parameters and check whether any changes
    // are necessary in the first place

    log.trc("%s: getting device memory data for initial parameters:\n", .{func});
    var dmds_full = try s.measure(mparams);

    // Saturate instead of overflowing; this also keeps n_ctx_min's
    // UINT32_MAX sentinel.
    const n_ctx_max: u32 = @intCast(@min(@as(u64, s.hp_nct) * n_streams, std.math.maxInt(u32)));
    const n_ctx_min_total: u32 = @intCast(@min(@as(u64, n_ctx_min) * n_streams, std.math.maxInt(u32)));

    // llama_context would use only hp_nct in total for n_ctx == 0: resolve
    // the context before measuring anything else.
    if (n_ctx_auto) {
        cparams.n_ctx = n_ctx_max;
        if (n_streams > 1) {
            log.trc("%s: context size unset and KV cache not unified -> using %u for %u sequences:\n", .{ func, n_ctx_max, n_streams });
            dmds_full = try s.measure(mparams);
        }
    }
    try s.addExtraMemory(dmds_full);

    const nd = s.devs.items.len; // number of devices
    s.nd = nd;
    const devs = s.devs.items;

    // int64_t rather than size_t, to handle deficits.
    const margins = try f.arena.alloc(i64, @max(nd, 1));
    for (margins, 0..) |*m, id| m.* = @bitCast(margins_s[id]);

    const dev_names = try f.arena.alloc([:0]u8, nd);
    {
        var max_length: usize = 0;
        for (devs, 0..) |dev, id| {
            dev_names[id] = try std.fmt.allocPrintSentinel(f.arena, "{s} ({s})", .{ std.mem.span(c.ggml_backend_dev_name(dev)), std.mem.span(c.ggml_backend_dev_description(dev)) }, 0);
            max_length = @max(max_length, dev_names[id].len);
        }
        for (dev_names) |*dn| {
            const padded = try f.arena.allocSentinel(u8, max_length, 0);
            @memcpy(padded[0..dn.len], dn.*);
            @memset(padded[dn.len..], ' ');
            dn.* = padded;
        }
    }

    var sum_free: i64 = 0;
    var sum_projected_free: i64 = 0;
    var sum_projected_used: i64 = 0;
    var sum_projected_model: i64 = 0;
    const projected_free_per_device = try f.arena.alloc(i64, nd);

    if (nd == 0) {
        sum_projected_used = @bitCast(dmds_full[dmds_full.len - 1].mb.total());
        sum_free = dmds_full[dmds_full.len - 1].total;
        sum_projected_free = sum_free - sum_projected_used;
        log.trc("%s: projected to use %lld MiB of host memory vs. %lld MiB of total host memory\n", .{ func, @divTrunc(sum_projected_used, MiB), @divTrunc(sum_free, MiB) });
        if (sum_projected_free >= margins[0]) {
            log.trc("%s: will leave %lld >= %lld MiB of system memory, no changes needed\n", .{ func, @divTrunc(sum_projected_free, MiB), @divTrunc(margins[0], MiB) });
            return;
        }
    } else {
        if (nd > 1) log.trc("%s: projected memory use with initial parameters [MiB]:\n", .{func});
        for (0..nd) |id| {
            const dmd = dmds_full[id];

            const projected_used: i64 = @bitCast(dmd.mb.total());
            const projected_free = dmd.free - projected_used;
            projected_free_per_device[id] = projected_free;

            sum_free += dmd.free;
            sum_projected_used += projected_used;
            sum_projected_free += projected_free;
            sum_projected_model += @as(i64, @bitCast(dmd.mb.model));

            if (nd > 1) {
                log.trc("%s:   - %s: %6lld total, %6lld used, %6lld free vs. target of %6lld\n", .{
                    func,                           dev_names[id].ptr,              @divTrunc(dmd.total, MiB),
                    @divTrunc(projected_used, MiB), @divTrunc(projected_free, MiB), @divTrunc(margins[id], MiB),
                });
            }
        }
        cAssert(sum_free >= 0 and sum_projected_used >= 0);
        log.trc("%s: projected to use %lld MiB of device memory vs. %lld MiB of free device memory\n", .{ func, @divTrunc(sum_projected_used, MiB), @divTrunc(sum_free, MiB) });
        if (nd == 1) {
            if (projected_free_per_device[0] >= margins[0]) {
                log.trc("%s: will leave %lld >= %lld MiB of free device memory, no changes needed\n", .{ func, @divTrunc(projected_free_per_device[0], MiB), @divTrunc(margins[0], MiB) });
                return;
            }
        } else {
            var changes_needed = false;
            for (0..nd) |id| {
                if (projected_free_per_device[id] < margins[id]) {
                    changes_needed = true;
                    break;
                }
            }
            if (!changes_needed) {
                log.trc("%s: targets for free memory can be met on all devices, no changes needed\n", .{func});
                return;
            }
        }
    }

    // step 2: try reducing memory use by reducing the context size

    {
        var global_surplus = sum_projected_free;
        if (nd == 0) {
            global_surplus -= margins[0];
        } else {
            for (0..nd) |id| global_surplus -= margins[id];
        }
        if (global_surplus < 0) {
            if (nd <= 1) {
                log.trc("%s: cannot meet free memory target of %lld MiB, need to reduce device memory by %lld MiB\n", .{ func, @divTrunc(margins[0], MiB), @divTrunc(-global_surplus, MiB) });
            } else {
                log.trc("%s: cannot meet free memory targets on all devices, need to use %lld MiB less in total\n", .{ func, @divTrunc(-global_surplus, MiB) });
            }
            if (n_ctx_auto) {
                if (n_ctx_max > n_ctx_min_total) {
                    var sum_used_target = sum_free;
                    if (nd == 0) {
                        sum_used_target -= margins[0];
                    } else {
                        for (0..nd) |id| sum_used_target -= margins[id];
                    }
                    if (nd > 1) {
                        // For several devices, be more conservative about how
                        // much context can fit: a dense model assigns only
                        // whole layers to a device and a MoE model whole
                        // tensors (estimated at <= 1/3 of a layer), expected
                        // to waste half a layer or tensor per device on
                        // average; use slightly more than that average.
                        const model_per_layer = @divTrunc(sum_projected_model, @as(i64, @min(@as(u32, @bitCast(mparams.n_gpu_layers)), s.hp_ngl)));
                        // `(nd + 1) * model_per_layer / k` is size_t
                        // arithmetic in the C, subtracted as such.
                        const less = (@as(u64, nd) + 1) *% @as(u64, @bitCast(model_per_layer)) / @as(u64, if (s.hp_nex == 0) 2 else 6);
                        sum_used_target = @bitCast(@as(u64, @bitCast(sum_used_target)) -% less);
                    }

                    var sum_projected_used_min_ctx: i64 = 0;
                    cparams.n_ctx = n_ctx_min_total;
                    const dmds_min_ctx = try s.measure(mparams);
                    try s.addExtraMemory(dmds_min_ctx);
                    if (nd == 0) {
                        sum_projected_used_min_ctx = @bitCast(dmds_min_ctx[dmds_min_ctx.len - 1].mb.total());
                    } else {
                        for (0..nd) |id| sum_projected_used_min_ctx += @as(i64, @bitCast(dmds_min_ctx[id].mb.total()));
                    }
                    if (sum_used_target > sum_projected_used_min_ctx) {
                        // linear interpolation between minimum and maximum context size:
                        const step = @divTrunc(@as(i64, n_ctx_max - n_ctx_min_total) * (sum_used_target - sum_projected_used_min_ctx), sum_projected_used - sum_projected_used_min_ctx);
                        cparams.n_ctx = @truncate(@as(u64, @bitCast(@as(i64, cparams.n_ctx) + step)));
                        // Round down for the CUDA backend, and keep it
                        // divisible by the number of streams.
                        const alignment: u32 = 256 *% n_streams;
                        cparams.n_ctx = @max(cparams.n_ctx -% cparams.n_ctx % alignment, n_ctx_min_total);

                        const bytes_per_ctx = @divTrunc(sum_projected_used - sum_projected_used_min_ctx, @as(i64, n_ctx_max - n_ctx_min_total));
                        const memory_reduction = @as(i64, n_ctx_max -% cparams.n_ctx) * bytes_per_ctx;
                        log.trc("%s: context size reduced from %u to %u -> need %lld MiB less memory in total\n", .{ func, n_ctx_max, cparams.n_ctx, @divTrunc(memory_reduction, MiB) });
                        if (nd <= 1) {
                            log.trc("%s: entire model can be fit by reducing context\n", .{func});
                            return;
                        }
                        log.trc("%s: entire model should be fit across devices by reducing context\n", .{func});
                    } else {
                        const memory_reduction = sum_projected_used - sum_projected_used_min_ctx;
                        log.trc("%s: context size reduced from %u to %u -> need %lld MiB less memory in total\n", .{ func, n_ctx_max, cparams.n_ctx, @divTrunc(memory_reduction, MiB) });
                    }
                } else {
                    if (n_ctx_min == std.math.maxInt(u32)) {
                        log.trc("%s: user has requested full context size of %u -> no change\n", .{ func, n_ctx_max });
                    } else {
                        log.trc("%s: default model context size is %u which is <= the min. context size of %u -> no change\n", .{ func, n_ctx_max, n_ctx_min_total });
                    }
                }
            } else {
                log.trc("%s: context size set by user to %u -> no change\n", .{ func, cparams.n_ctx });
            }
        }
    }
    if (nd == 0) {
        return f.fail(error.Fit, "was unable to fit model into system memory by reducing context, abort", .{});
    }

    if (mparams.n_gpu_layers != default_mparams.n_gpu_layers) {
        return f.fail(error.Fit, "n_gpu_layers already set by user to {d}, abort", .{mparams.n_gpu_layers});
    }
    if (nd > 1) {
        if (tensor_split == null) {
            return f.fail(error.Fit, "did not provide a buffer to write the tensor_split to, abort", .{});
        }
        if (mparams.tensor_split != null) {
            for (0..nd) |id| {
                if (mparams.tensor_split[id] != 0.0) return f.fail(error.Fit, "model_params::tensor_split already set by user, abort", .{});
            }
        }
        if (mparams.split_mode == c.LLAMA_SPLIT_MODE_ROW) {
            return f.fail(error.Fit, "changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort", .{});
        }
    }
    const tbo = tensor_buft_overrides orelse
        return f.fail(error.Fit, "did not provide buffer to set tensor_buft_overrides, abort", .{});
    if (mparams.tensor_buft_overrides != null) {
        if (mparams.tensor_buft_overrides[0].pattern != null or mparams.tensor_buft_overrides[0].buft != null) {
            return f.fail(error.Fit, "model_params::tensor_buft_overrides already set by user, abort", .{});
        }
    }

    // step 3: iteratively fill the back to front with "dense" layers
    //   - for a dense model simply fill full layers, giving each device a
    //     contiguous slice of the model
    //   - for a MoE model, the same but with all MoE tensors in system memory

    s.ntbo = api.llama.llama_max_tensor_buft_overrides();

    var global_surplus_cpu_moe: i64 = 0;
    if (s.hp_nex > 0) {
        const pattern_moe_all = "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps"; // matches all MoE tensors
        const cpu_buft = c.ggml_backend_cpu_buffer_type();
        tbo[0] = .{ .pattern = pattern_moe_all, .buft = cpu_buft };
        tbo[1] = .{ .pattern = null, .buft = null };
        mparams.tensor_buft_overrides = tbo;

        log.trc("%s: getting device memory data with all MoE tensors moved to system memory:\n", .{func});
        const dmds_cpu_moe = try s.measure(mparams);
        try s.addExtraMemory(dmds_cpu_moe);

        for (0..nd) |id| {
            global_surplus_cpu_moe += dmds_cpu_moe[id].free;
            global_surplus_cpu_moe -= @as(i64, @bitCast(dmds_cpu_moe[id].mb.total())) + margins[id];
        }

        if (global_surplus_cpu_moe > 0) {
            log.trc("%s: with only dense weights in device memory there is a total surplus of %lld MiB\n", .{ func, @divTrunc(global_surplus_cpu_moe, MiB) });
        } else {
            log.trc("%s: with only dense weights in device memory there is still a total deficit of %lld MiB\n", .{ func, @divTrunc(-global_surplus_cpu_moe, MiB) });
        }

        // reset
        tbo[0] = .{ .pattern = null, .buft = null };
        mparams.tensor_buft_overrides = tbo;
    }

    const targets = try f.arena.alloc(i64, nd); // maximum acceptable memory use per device
    for (0..nd) |id| {
        targets[id] = dmds_full[id].free - margins[id];
        log.trc("%s: id=%zu, target=%lld MiB\n", .{ func, id, @divTrunc(targets[id], MiB) });
    }

    // Which buffer type the first partial layer of each device overflows to.
    var overflow_bufts = try f.arena.alloc(c.ggml_backend_buffer_type_t, nd);
    for (overflow_bufts) |*b| b.* = c.ggml_backend_cpu_buffer_type();

    var ngl_per_device = try f.arena.alloc(Ngl, nd);
    @memset(ngl_per_device, .{});
    var mem = try s.memoryForLayers(func, ngl_per_device, overflow_bufts);

    // Optimize the number of layers per device by false position:
    //   - ngl_per_device has 0 layers for each device, the lower bound
    //   - try a "high" configuration giving a device all unassigned layers
    //   - interpolate memory per layer between low and high to guess where
    //     it meets the target
    //   - measure the guess and replace the low or the high bound with it
    //   - stop at a difference of one layer and keep the lower bound, which
    //     just still fits
    //   - the last device has the output layer, which cannot be partial
    if (s.hp_nex == 0) {
        log.trc("%s: filling dense layers back-to-front:\n", .{func});
    } else {
        log.trc("%s: filling dense-only layers back-to-front:\n", .{func});
    }
    var id_signed: isize = @as(isize, @intCast(nd)) - 1;
    while (id_signed >= 0) : (id_signed -= 1) {
        const id: usize = @intCast(id_signed);
        var n_unassigned: u32 = s.hp_ngl +% 1;
        for (id + 1..nd) |jd| {
            cAssert(n_unassigned >= ngl_per_device[jd].n_layer);
            n_unassigned -%= ngl_per_device[jd].n_layer;
        }

        var ngl_per_device_high = try f.arena.dupe(Ngl, ngl_per_device);
        ngl_per_device_high[id].n_layer = n_unassigned;
        if (s.hp_nex > 0) {
            ngl_per_device_high[id].n_part = if (id < nd - 1) ngl_per_device_high[id].n_layer else ngl_per_device_high[id].n_layer -% 1;
        }
        if (ngl_per_device_high[id].n_layer > 0) {
            var mem_high = try s.memoryForLayers(func, ngl_per_device_high, overflow_bufts);
            if (mem_high[id] > targets[id]) {
                cAssert(ngl_per_device_high[id].n_layer > ngl_per_device[id].n_layer);
                var delta: u32 = ngl_per_device_high[id].n_layer -% ngl_per_device[id].n_layer;
                log.trc("%s: start filling device %u, delta=%u\n", .{ func, @as(u32, @intCast(id)), delta });
                while (delta > 1) {
                    var step_size: u32 = @truncate(@as(u64, @bitCast(@divTrunc(@as(i64, delta) * (targets[id] - mem[id]), mem_high[id] - mem[id]))));
                    step_size = @max(step_size, 1);
                    step_size = @min(step_size, delta - 1);

                    const ngl_per_device_test = try f.arena.dupe(Ngl, ngl_per_device);
                    ngl_per_device_test[id].n_layer +%= step_size;
                    if (s.hp_nex != 0) {
                        // the first layer is the output layer, which must always be full
                        ngl_per_device_test[id].n_part +%= if (id == nd - 1 and ngl_per_device_test[id].n_part == 0) step_size -% 1 else step_size;
                    }
                    const mem_test = try s.memoryForLayers(func, ngl_per_device_test, overflow_bufts);

                    if (mem_test[id] <= targets[id]) {
                        ngl_per_device = ngl_per_device_test;
                        mem = mem_test;
                        log.trc("%s: set ngl_per_device[%d].n_layer=%u\n", .{ func, @as(c_int, @intCast(id)), ngl_per_device[id].n_layer });
                    } else {
                        ngl_per_device_high = ngl_per_device_test;
                        mem_high = mem_test;
                        log.trc("%s: set ngl_per_device_high[%d].n_layer=%u\n", .{ func, @as(c_int, @intCast(id)), ngl_per_device_high[id].n_layer });
                    }
                    delta = ngl_per_device_high[id].n_layer -% ngl_per_device[id].n_layer;
                }
            } else {
                cAssert(ngl_per_device_high[id].n_layer == n_unassigned);
                ngl_per_device = ngl_per_device_high;
                mem = mem_high;
                log.trc("%s: set ngl_per_device[%d].n_layer=%u\n", .{ func, @as(c_int, @intCast(id)), ngl_per_device[id].n_layer });
            }
        }

        const projected_margin = dmds_full[id].free - mem[id];
        log.trc("%s:   - %s: %2u layers, %6lld MiB used, %6lld MiB free\n", .{ func, dev_names[id].ptr, ngl_per_device[id].n_layer, @divTrunc(mem[id], MiB), @divTrunc(projected_margin, MiB) });
    }
    if (s.hp_nex == 0 or global_surplus_cpu_moe <= 0) {
        try s.setNglTensorSplitTbo(ngl_per_device, overflow_bufts, mparams);
        return;
    }

    // step 4: for a MoE model where all dense tensors fit, convert the
    // dense-only layers at the back to full layers at the front until every
    // device is full -- the same procedure as for the dense-only layers, but
    // front-to-back -- and try to fit at least part of one more layer, to
    // waste less of a "small" GPU with e.g. 24 GiB.

    var id_dense_start: usize = nd;
    id_signed = @as(isize, @intCast(nd)) - 1;
    while (id_signed >= 0) : (id_signed -= 1) {
        const id: usize = @intCast(id_signed);
        if (ngl_per_device[id].n_layer > 0) {
            id_dense_start = id;
            continue;
        }
        break;
    }
    cAssert(id_dense_start < nd);

    log.trc("%s: converting dense-only layers to full layers and filling them front-to-back with overflow to next device/system memory:\n", .{func});
    var id: usize = 0;
    while (id <= id_dense_start and id_dense_start < nd) : (id += 1) {
        var ngl_per_device_high = try f.arena.dupe(Ngl, ngl_per_device);
        for (id_dense_start..nd) |jd| {
            const n_layer_move = if (jd < nd - 1) ngl_per_device_high[jd].n_layer else ngl_per_device_high[jd].n_layer -% 1;
            ngl_per_device_high[id].n_layer +%= n_layer_move;
            ngl_per_device_high[jd].n_layer -%= n_layer_move;
            ngl_per_device_high[jd].n_part = 0;
        }
        var id_dense_start_high: usize = nd - 1;
        var mem_high = try s.memoryForLayers(func, ngl_per_device_high, overflow_bufts);

        if (mem_high[id] > targets[id]) {
            cAssert(ngl_per_device_high[id].nFull() >= ngl_per_device[id].nFull());
            var delta: u32 = ngl_per_device_high[id].nFull() -% ngl_per_device[id].nFull();
            while (delta > 1) {
                var step_size: u32 = @truncate(@as(u64, @bitCast(@divTrunc(@as(i64, delta) * (targets[id] - mem[id]), mem_high[id] - mem[id]))));
                step_size = @max(step_size, 1);
                step_size = @min(step_size, delta - 1);

                const ngl_per_device_test = try f.arena.dupe(Ngl, ngl_per_device);
                var id_dense_start_test = id_dense_start;
                var n_converted_test: u32 = 0;
                while (id_dense_start_test < nd) : (id_dense_start_test += 1) {
                    const n_convert_jd = @min(step_size -% n_converted_test, ngl_per_device_test[id_dense_start_test].n_part);
                    ngl_per_device_test[id_dense_start_test].n_layer -%= n_convert_jd;
                    ngl_per_device_test[id_dense_start_test].n_part -%= n_convert_jd;
                    ngl_per_device_test[id].n_layer +%= n_convert_jd;
                    n_converted_test +%= n_convert_jd;

                    if (ngl_per_device_test[id_dense_start_test].n_part > 0) break;
                }
                const mem_test = try s.memoryForLayers(func, ngl_per_device_test, overflow_bufts);

                if (mem_test[id] <= targets[id]) {
                    ngl_per_device = ngl_per_device_test;
                    mem = mem_test;
                    id_dense_start = id_dense_start_test;
                    log.trc("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%u, %u), id_dense_start=%zu\n", .{ func, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start });
                } else {
                    ngl_per_device_high = ngl_per_device_test;
                    mem_high = mem_test;
                    id_dense_start_high = id_dense_start_test;
                    log.trc("%s: set ngl_per_device_high[%zu].(n_layer, n_part)=(%u, %u), id_dense_start_high=%zu\n", .{ func, id, ngl_per_device_high[id].n_layer, ngl_per_device_high[id].n_part, id_dense_start_high });
                }
                cAssert(ngl_per_device_high[id].nFull() >= ngl_per_device[id].nFull());
                delta = ngl_per_device_high[id].nFull() -% ngl_per_device[id].nFull();
            }
        } else {
            ngl_per_device = ngl_per_device_high;
            mem = mem_high;
            id_dense_start = id_dense_start_high;
            log.trc("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%u, %u), id_dense_start=%zu\n", .{ func, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start });
        }

        // try to fit at least part of one more layer
        if (ngl_per_device[id_dense_start].n_layer > @as(u32, if (id < nd - 1) 0 else 1)) {
            const ngl_per_device_test = try f.arena.dupe(Ngl, ngl_per_device);
            var id_dense_start_test = id_dense_start;
            ngl_per_device_test[id_dense_start_test].n_layer -%= 1;
            ngl_per_device_test[id_dense_start_test].n_part -%= 1;
            ngl_per_device_test[id].n_layer +%= 1;
            ngl_per_device_test[id].n_part +%= 1;
            if (ngl_per_device_test[id_dense_start_test].n_part == 0) id_dense_start_test += 1;
            ngl_per_device_test[id].overflow_type = .up;
            const overflow_bufts_test = try f.arena.dupe(c.ggml_backend_buffer_type_t, overflow_bufts);
            if (id < nd - 1) overflow_bufts_test[id] = c.ggml_backend_dev_buffer_type(devs[id + 1]);
            log.trc("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_UP\n", .{func});
            var mem_test = try s.memoryForLayers(func, ngl_per_device_test, overflow_bufts_test);
            if (mem_test[id] < targets[id] and (id + 1 == nd or mem_test[id + 1] < targets[id + 1])) {
                ngl_per_device = try f.arena.dupe(Ngl, ngl_per_device_test);
                overflow_bufts = overflow_bufts_test;
                mem = mem_test;
                id_dense_start = id_dense_start_test;
                log.trc("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%u, %u, UP), id_dense_start=%zu\n", .{ func, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start });

                ngl_per_device_test[id].overflow_type = .gate;
                log.trc("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_GATE\n", .{func});
                mem_test = try s.memoryForLayers(func, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] and (id + 1 == nd or mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = try f.arena.dupe(Ngl, ngl_per_device_test);
                    overflow_bufts = overflow_bufts_test;
                    mem = mem_test;
                    id_dense_start = id_dense_start_test;
                    log.trc("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%u, %u, GATE), id_dense_start=%zu\n", .{ func, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start });
                }
            } else {
                ngl_per_device_test[id].overflow_type = .attn;
                log.trc("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_ATTN\n", .{func});
                mem_test = try s.memoryForLayers(func, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] and (id + 1 == nd or mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = try f.arena.dupe(Ngl, ngl_per_device_test);
                    overflow_bufts = overflow_bufts_test;
                    mem = mem_test;
                    id_dense_start = id_dense_start_test;
                    log.trc("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%u, %u, ATTN), id_dense_start=%zu\n", .{ func, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start });
                }
            }
        }

        const projected_margin = dmds_full[id].free - mem[id];
        log.trc("%s:   - %s: %2u layers (%2u overflowing), %6lld MiB used, %6lld MiB free\n", .{ func, dev_names[id].ptr, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, @divTrunc(mem[id], MiB), @divTrunc(projected_margin, MiB) });
    }

    // Report the devices the conversion from dense-only to full layers left
    // alone. `id_dense_start` may have reached `nd`, where the C's loop runs
    // no times; a Zig range starting past its end is illegal.
    var jd = id_dense_start + 1;
    while (jd < nd) : (jd += 1) {
        const projected_margin = dmds_full[jd].free - mem[jd];
        log.trc("%s:   - %s: %2u layers (%2u overflowing), %6lld MiB used, %6lld MiB free\n", .{ func, dev_names[jd].ptr, ngl_per_device[jd].n_layer, ngl_per_device[jd].n_part, @divTrunc(mem[jd], MiB), @divTrunc(projected_margin, MiB) });
    }

    try s.setNglTensorSplitTbo(ngl_per_device, overflow_bufts, mparams);
}

/// Ports `common_fit_params` (fit.cpp:878 @5266f24da): fits `mparams` and
/// `cparams` to free device memory, system memory taken as unlimited where
/// there is a device. Only parameters still equal to
/// `llama_model_default_params`'s are changed, and the context size only
/// when it is 0.
///
/// Parameters:
/// - `gpa`: allocates the scratch, freed before return.
/// - `patterns`: holds the override patterns written into
///   `tensor_buft_overrides`; keep it until the model is loaded.
/// - `log`: where the trace, the warning and the error go.
/// - `path_model`: the model file.
/// - `mparams`, `cparams`: changed in place.
/// - `tensor_split`: where a split is written, `llama_max_devices()`
///   entries; may be null with one device or none.
/// - `tensor_buft_overrides`: where overrides are written,
///   `llama_max_tensor_buft_overrides()` entries.
/// - `margins`: bytes to leave free per device; one entry is read when
///   there is no device.
/// - `n_ctx_min`: the least context to shrink to; `maxInt(u32)` forbids
///   shrinking, as an explicit `-c 0` does.
/// - `extra`: a model fitted alongside, or null.
/// - `log_level`: libllama's messages below this go to the debug level.
///
/// Return: the C's status; `error.OutOfMemory` where it would throw
/// `std::bad_alloc`.
pub fn fitParams(
    gpa: std.mem.Allocator,
    patterns: *Patterns,
    log: Log,
    path_model: [*:0]const u8,
    mparams: *c.llama_model_params,
    cparams: *c.llama_context_params,
    tensor_split: ?[*]f32,
    tensor_buft_overrides: ?[*]c.llama_model_tensor_buft_override,
    margins: [*]const usize,
    n_ctx_min: u32,
    extra: ?*const ExtraModel,
    log_level: c.enum_ggml_log_level,
) error{OutOfMemory}!Status {
    const func: [*:0]const u8 = "common_fit_params";
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var f: Fitter = .{ .arena = arena_state.allocator(), .patterns = patterns, .log = log };

    const t0_us = api.llama.llama_time_us();
    var status: Status = .success;
    if (fitImpl(&f, path_model, mparams, cparams, tensor_split, tensor_buft_overrides, margins, n_ctx_min, extra, log_level)) {
        log.trc("%s: successfully fit params to free device memory\n", .{func});
    } else |e| switch (e) {
        error.Fit => {
            log.wrn("%s: failed to fit params to free device memory: %s\n", .{ func, f.what.ptr });
            status = .failure;
        },
        error.Runtime => {
            log.err("%s: encountered an error while trying to fit params to free device memory: %s\n", .{ func, f.what.ptr });
            status = .@"error";
        },
        error.OutOfMemory => return error.OutOfMemory,
    }
    const t1_us = api.llama.llama_time_us();
    log.trc("%s: fitting params to free memory took %.2f seconds\n", .{ func, @as(f64, @floatFromInt(t1_us - t0_us)) * 1e-6 });
    return status;
}

/// Ports `common_memory_breakdown_print` (fit.cpp:905 @5266f24da): a table
/// of what each device, the host and any other buffer type holds, written at
/// the trace level.
///
/// Parameters:
/// - `gpa`: allocates the table, freed before return.
/// - `log`: where the table goes.
/// - `ctx`: the context to break down.
///
/// Return: nothing; `error.OutOfMemory` where the C would throw.
pub fn memoryBreakdownPrint(gpa: std.mem.Allocator, log: Log, ctx: ?*const c.struct_llama_context) !void {
    const func: [*:0]const u8 = "common_memory_breakdown_print";
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const model = api.context.llama_get_model(ctx);

    const devices = try arena.alloc(c.ggml_backend_dev_t, @intCast(api.model.llama_model_n_devices(model)));
    for (devices, 0..) |*d, i| d.* = api.model.llama_model_get_device(model, @intCast(i));

    const memory_breakdown = try api.context.llama_get_memory_breakdown(ctx, arena);

    const Row = [9][]const u8;
    var table_data: std.ArrayList(Row) = .empty;
    const template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const template_gpu = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const template_other = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    try table_data.append(arena, .{ template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted" });

    const mib: usize = 1024 * 1024;
    const desc_prefixes_strip = [_][]const u8{ "NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct " };

    // Buffer types already reported, so none is counted twice.
    var seen_buffer_types: std.ArrayList(c.ggml_backend_buffer_type_t) = .empty;

    // Running breakdown for each device and for the host.
    const mb_dev = try arena.alloc(MemoryBreakdownData, devices.len);
    @memset(mb_dev, .{});
    var mb_host: MemoryBreakdownData = .{};

    for (memory_breakdown) |e| {
        if (c.ggml_backend_buft_is_host(e.buft)) {
            mb_host.model +%= e.model;
            mb_host.context +%= e.context;
            mb_host.compute +%= e.compute;
            try seen_buffer_types.append(arena, e.buft);
            continue;
        }
        if (c.ggml_backend_buft_get_device(e.buft)) |dev| {
            for (devices, 0..) |d, i_dev| {
                if (d == dev) {
                    mb_dev[i_dev].model +%= e.model;
                    mb_dev[i_dev].context +%= e.context;
                    mb_dev[i_dev].compute +%= e.compute;
                    try seen_buffer_types.append(arena, e.buft);
                    break;
                }
            }
        }
    }

    // each device
    for (devices, mb_dev) |dev, mb| {
        const name = std.mem.span(c.ggml_backend_dev_name(dev));
        var desc: []const u8 = std.mem.span(c.ggml_backend_dev_description(dev));
        for (desc_prefixes_strip) |prefix| {
            if (std.mem.startsWith(u8, desc, prefix)) desc = desc[prefix.len..];
        }

        var free: usize = 0;
        var total: usize = 0;
        c.ggml_backend_dev_memory(dev, &free, &total);

        const self_size = mb.model +% mb.context +% mb.compute;
        const unaccounted = @as(i64, @bitCast(total)) - @as(i64, @bitCast(free)) - @as(i64, @bitCast(self_size));

        try table_data.append(arena, .{
            template_gpu,
            try std.fmt.allocPrint(arena, "  - {s} ({s})", .{ name, desc }),
            try std.fmt.allocPrint(arena, "{d}", .{total / mib}),
            try std.fmt.allocPrint(arena, "{d}", .{free / mib}),
            try std.fmt.allocPrint(arena, "{d}", .{self_size / mib}),
            try std.fmt.allocPrint(arena, "{d}", .{mb.model / mib}),
            try std.fmt.allocPrint(arena, "{d}", .{mb.context / mib}),
            try std.fmt.allocPrint(arena, "{d}", .{mb.compute / mib}),
            try std.fmt.allocPrint(arena, "{d}", .{@divTrunc(unaccounted, @as(i64, mib))}),
        });
    }

    // the host
    {
        const self_size = mb_host.model +% mb_host.context +% mb_host.compute;
        try table_data.append(arena, .{
            template_other,
            "  - Host",
            "", // total
            "", // free
            try std.fmt.allocPrint(arena, "{d}", .{self_size / mib}),
            try std.fmt.allocPrint(arena, "{d}", .{mb_host.model / mib}),
            try std.fmt.allocPrint(arena, "{d}", .{mb_host.context / mib}),
            try std.fmt.allocPrint(arena, "{d}", .{mb_host.compute / mib}),
            "", // unaccounted
        });
    }

    // every remaining buffer type
    for (memory_breakdown) |e| {
        if (std.mem.indexOfScalar(c.ggml_backend_buffer_type_t, seen_buffer_types.items, e.buft) != null) continue;
        const name = std.mem.span(c.ggml_backend_buft_name(e.buft));
        const self_size = e.model +% e.context +% e.compute;
        try table_data.append(arena, .{
            template_other,
            try std.fmt.allocPrint(arena, "  - {s}", .{name}),
            "", // total
            "", // free
            try std.fmt.allocPrint(arena, "{d}", .{self_size / mib}),
            try std.fmt.allocPrint(arena, "{d}", .{e.model / mib}),
            try std.fmt.allocPrint(arena, "{d}", .{e.context / mib}),
            try std.fmt.allocPrint(arena, "{d}", .{e.compute / mib}),
            "", // unaccounted
        });
        try seen_buffer_types.append(arena, e.buft);
    }

    // Pad each column to its widest cell: the name column on the right,
    // the numbers on the left.
    for (1..9) |j| {
        var max_len: usize = 0;
        for (table_data.items) |td| max_len = @max(max_len, td[j].len);
        for (table_data.items) |*td| {
            const padded = try arena.alloc(u8, max_len);
            const pad = max_len - td[j].len;
            if (j == 1) {
                @memcpy(padded[0..td[j].len], td[j]);
                @memset(padded[td[j].len..], ' ');
            } else {
                @memset(padded[0..pad], ' ');
                @memcpy(padded[pad..], td[j]);
            }
            td[j] = padded;
        }
    }
    for (table_data.items) |td| {
        var cells: [9][:0]const u8 = undefined;
        for (td, 0..) |cell, j| cells[j] = try arena.dupeSentinel(u8, cell, 0);
        log.trc(cells[0].ptr, .{ func, cells[1].ptr, cells[2].ptr, cells[3].ptr, cells[4].ptr, cells[5].ptr, cells[6].ptr, cells[7].ptr, cells[8].ptr });
    }
}

/// Ports `common_fit_print` (fit.cpp:1047 @5266f24da): one line per device
/// and one for the host -- name, then model, context and compute in MiB --
/// for `llama-fit-params`.
///
/// Parameters:
/// - `gpa`: allocates the scratch, freed before return.
/// - `log`: where the measurement's trace goes.
/// - `out`: where the C's `printf` writes, stdout.
/// - `path_model`, `mparams`, `cparams`: what to measure.
///
/// Return: nothing; write errors, and `error.Runtime` where the C would
/// throw out of the measurement.
/// `e.what()` of the exception `fitPrint` last threw, which upstream lets
/// escape `main`: `llama-fit-params` prints it as libc++ does.
pub var print_what: [256:0]u8 = @splat(0);

pub fn fitPrint(
    gpa: std.mem.Allocator,
    log: Log,
    out: *std.Io.Writer,
    path_model: [*:0]const u8,
    mparams: *c.llama_model_params,
    cparams: *c.llama_context_params,
) !void {
    var devs: std.ArrayList(c.ggml_backend_dev_t) = .empty;
    defer devs.deinit(gpa);
    var hp_ngl: u32 = 0; // hparams.n_gpu_layers
    var hp_nct: u32 = 0; // hparams.n_ctx_train
    var hp_nex: u32 = 0; // hparams.n_expert

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var patterns: Patterns = try .init(gpa);
    defer patterns.deinit();
    var f: Fitter = .{ .arena = arena_state.allocator(), .patterns = &patterns, .log = log };
    var arena_devs: std.ArrayList(c.ggml_backend_dev_t) = .empty;
    const dmd = deviceMemoryDataImpl(&f, path_model, mparams, cparams, &arena_devs, &hp_ngl, &hp_nct, &hp_nex, c.GGML_LOG_LEVEL_ERROR) catch |e| switch (e) {
        error.Fit, error.Runtime => {
            const w = std.mem.span(f.what.ptr);
            const n = @min(w.len, print_what.len - 1);
            @memcpy(print_what[0..n], w[0..n]);
            print_what[n] = 0;
            return error.Runtime;
        },
        error.OutOfMemory => return error.OutOfMemory,
    };
    try devs.appendSlice(gpa, arena_devs.items);
    if (dmd.len != devs.items.len + 1) std.process.abort(); // GGML_ASSERT

    for (devs.items, 0..) |dev, id| {
        try out.print("{s} ", .{std.mem.span(c.ggml_backend_dev_name(dev))});
        try out.print("{d} ", .{dmd[id].mb.model / 1024 / 1024});
        try out.print("{d} ", .{dmd[id].mb.context / 1024 / 1024});
        try out.print("{d} ", .{dmd[id].mb.compute / 1024 / 1024});
        try out.print("\n", .{});
    }

    const host = dmd[dmd.len - 1];
    try out.print("Host ", .{});
    try out.print("{d} ", .{host.mb.model / 1024 / 1024});
    try out.print("{d} ", .{host.mb.context / 1024 / 1024});
    try out.print("{d} ", .{host.mb.compute / 1024 / 1024});
    try out.print("\n", .{});
    try out.flush();
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the log drops what is above its verbosity and keeps the C's formatting" {
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    const log: Log = .{ .verbosity = Log.level_warn, .out = &buf.writer, .prefix = true };
    log.trc("%s\n", .{"dropped"});
    log.wrn("%s: %6lld|%.2f\n", .{ "w", @as(i64, 42), @as(f64, 0.125) });
    log.err("%2u\n", .{@as(u32, 7)});
    try std.testing.expectEqualStrings("W w:     42|0.12\nE  7\n", buf.written());
}

test "override patterns are cached per layer and fraction" {
    var patterns: Patterns = try .init(std.testing.allocator);
    defer patterns.deinit();
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    var f: Fitter = .{ .arena = arena_state.allocator(), .patterns = &patterns, .log = .{} };
    var s: FitImpl = undefined;
    s.f = &f;
    const a = try s.overflowPattern(3, .up);
    try std.testing.expectEqualStrings("blk\\.3\\.ffn_(gate|gate_up|down).*", std.mem.span(a));
    try std.testing.expectEqual(a, try s.overflowPattern(3, .up));
    try std.testing.expectEqualStrings("blk\\.3\\.ffn_(up|down|gate_up|gate)_(ch|)exps", std.mem.span(try s.overflowPattern(3, .moe)));
    try std.testing.expectError(error.Runtime, s.overflowPattern(1000, .attn));
    try std.testing.expectEqualStrings("at most 1000 model layers are supported", f.what);
}
