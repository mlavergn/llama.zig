//! Compares `cli/fit.zig` with upstream's `common/fit.cpp`.
//!
//! **Not a port.** The harness behind `scripts/fit-diff`. Each case runs
//! `common_fit_params` twice in one process on the same inputs -- once
//! through the C++, compiled unmodified over `libllama-ref.a`
//! (`harness/fit_ref.cpp`), once through `fit.zig` over the port -- and each
//! side writes a transcript that must be identical:
//!
//! - the status, the context size, the extra model's context size and
//!   `n_gpu_layers`;
//! - where `tensor_split` and `tensor_buft_overrides` point -- the buffer
//!   handed in, the user's own, or null -- and every entry of both buffers:
//!   each split value, each override's pattern and buffer type;
//! - **every line `fit.cpp` logs**, at the trace level, with its level
//!   letter: what was measured on each device, every step of the search, the
//!   memory breakdown tables, warnings and errors;
//! - every libllama message that passed through the log filter `fit.cpp`
//!   installs, with the level it arrived at.
//!
//! The fit decides from what each device reports free, so the devices are
//! `harness/fit_dev.c`'s fakes with chosen memory -- set relative to what
//! each model was measured to need, so the cases land in the arms they are
//! meant for -- plus the CPU alone (`devices = { NULL }`), where the margin is
//! set relative to the host's total memory, and the real Metal device.
//!
//! **Every arm must be reached.** After the cases, each phrase in
//! `required` must have appeared in some reference transcript: a case list
//! that drifts out of an arm fails the gate rather than passing it by
//! testing nothing.
//!
//! Usage: fit_diff <qwen35.gguf> <qwen3.gguf> <moe.gguf> <dense.gguf> [--case <substring>]
//!   FIT_DIFF_OUT=<dir> keeps both transcripts of every case;
//!   FIT_DIFF_TMP=<dir> holds the reference's log file;
//!   FIT_DIFF_SIDE=ref|port runs one side alone and prints its transcript.

const std = @import("std");
const harness_root = @import("harness_root");
const fit = @import("fit");

const c = harness_root.impl.c;
const api = harness_root.api;

const gpa = std.heap.c_allocator;
const MiB: usize = 1024 * 1024;

extern fn fit_ref_log_setup(path: [*:0]const u8, verbosity: c_int) void;
extern fn fit_ref_log_flush() void;
extern fn fit_ref_params(
    path_model: [*:0]const u8,
    mparams: *c.llama_model_params,
    cparams: *c.llama_context_params,
    tensor_split: ?[*]f32,
    tensor_buft_overrides: ?[*]c.llama_model_tensor_buft_override,
    margins: [*]usize,
    n_ctx_min: u32,
    extra_path: ?[*:0]const u8,
    extra_mparams: ?*c.llama_model_params,
    extra_cparams: ?*c.llama_context_params,
    extra_shares_model: bool,
    log_level: c_int,
) c_int;
extern fn fit_ref_device_memory(
    path_model: [*:0]const u8,
    mparams: *const c.llama_model_params,
    cparams: *const c.llama_context_params,
    out: [*]i64,
    cap: c_int,
    devs: [*]c.ggml_backend_dev_t,
    n_devs: *c_int,
    hp: *[3]u32,
    log_level: c_int,
) c_int;
extern fn fit_ref_print(path_model: [*:0]const u8, mparams: *c.llama_model_params, cparams: *c.llama_context_params, path: [*:0]const u8) c_int;
extern fn fit_dev_set(i: c_int, free: usize, total: usize, typ: c_int) c.ggml_backend_dev_t;

// -----------------------------------------------------------------------------
// Transcripts

/// libllama's messages for the side running, as `level|text`.
var llama_log: std.ArrayList(u8) = .empty;

fn logCb(level: c.enum_ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    llama_log.print(gpa, "{d}|{s}", .{ level, std.mem.span(@as([*:0]const u8, @ptrCast(text))) }) catch unreachable;
}

/// `text` with every clock reading -- "took 1.20 ms", "took 0.03 seconds"
/// -- replaced by `<t>`: `fit.cpp` times itself, and libllama times its
/// reserve.
fn maskTiming(text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var k: usize = 0;
    while (std.mem.indexOfPos(u8, text, k, "took ")) |at| {
        var e = at + "took ".len;
        while (e < text.len and (std.ascii.isDigit(text[e]) or text[e] == '.')) e += 1;
        try out.appendSlice(gpa, text[k .. at + "took ".len]);
        if (e > at + "took ".len) try out.appendSlice(gpa, "<t>");
        k = e;
    }
    try out.appendSlice(gpa, text[k..]);
    return out.toOwnedSlice(gpa);
}

var ref_log_path: [:0]const u8 = undefined;
var ref_log_offset: usize = 0;

/// What the reference's `common_log` wrote since the last call.
fn refLogSince() ![]u8 {
    fit_ref_log_flush();
    const f = std.c.fopen(ref_log_path.ptr, "rb") orelse return error.NoLog;
    defer _ = std.c.fclose(f);
    var all: std.ArrayList(u8) = .empty;
    defer all.deinit(gpa);
    var buf: [65536]u8 = undefined;
    while (true) {
        const n = std.c.fread(&buf, 1, buf.len, f);
        if (n == 0) break;
        try all.appendSlice(gpa, buf[0..n]);
    }
    const out = try gpa.dupe(u8, all.items[@min(ref_log_offset, all.items.len)..]);
    ref_log_offset = all.items.len;
    return out;
}

// -----------------------------------------------------------------------------
// Cases

/// A fake device: free memory as the margin plus a fraction of what the
/// model was measured to need on one device, total twice the free.
const Dev = struct {
    frac: f64,
    /// 0 GPU, 1 IGPU, 2 ACCEL.
    typ: c_int = 0,
    /// Report 0 free of 0 total, whatever `frac` says.
    zero: bool = false,
    /// Instead of `frac`: exactly what this many layers use at the case's
    /// context, so the false-position search meets its target with
    /// equality -- the one place `<=` and `<` part.
    tie_layers: u32 = 0,
};

const ModelKey = enum { qwen35, qwen3, moe, dense, missing };

const Margin = union(enum) {
    /// The same for every device, in MiB.
    mib: usize,
    /// One per device, in MiB.
    per_dev: []const usize,
    /// CPU only: the margin leaves `frac` of the way from the model's use at
    /// the least context to its use at the full one; above 1 it fits as is,
    /// below 0 nothing fits.
    host_frac: f64,
};

const Extra = struct {
    model: ModelKey,
    shares_model: bool = false,
};

const Case = struct {
    name: []const u8,
    model: ModelKey,
    /// Fake devices; empty means the default ones (Metal here) unless
    /// `cpu_only`.
    devs: []const Dev = &.{},
    cpu_only: bool = false,
    margin: Margin = .{ .mib = 1024 },
    n_ctx: u32 = 0,
    n_ctx_min: u32 = 4096,
    n_seq_max: u32 = 1,
    kv_unified: bool = false,
    n_gpu_layers: i32 = -1,
    split_mode: c.enum_llama_split_mode = c.LLAMA_SPLIT_MODE_LAYER,
    user_split: bool = false,
    user_tbo: bool = false,
    null_split_buf: bool = false,
    null_tbo_buf: bool = false,
    load_mtp: bool = false,
    extra: ?Extra = null,
    extra_missing: bool = false,
    log_level: c_int = c.GGML_LOG_LEVEL_ERROR,
    /// Also compare `common_get_device_memory_data` and `common_fit_print`.
    memory_data: bool = false,
};

fn d(frac: f64) Dev {
    return .{ .frac = frac };
}

const cases = [_]Case{
    // The default devices: Metal, which has room for all of these.
    .{ .name = "metal-qwen35", .model = .qwen35, .memory_data = true },
    .{ .name = "metal-qwen3", .model = .qwen3 },
    .{ .name = "metal-moe", .model = .moe },
    .{ .name = "metal-qwen3-ctx-set", .model = .qwen3, .n_ctx = 8192 },

    // The CPU alone: system memory, the context reduced.
    .{ .name = "cpu-qwen3-fits", .model = .qwen3, .cpu_only = true, .memory_data = true },
    .{ .name = "cpu-qwen3-reduce-mid", .model = .qwen3, .cpu_only = true, .margin = .{ .host_frac = 0.5 } },
    .{ .name = "cpu-qwen3-reduce-low", .model = .qwen3, .cpu_only = true, .margin = .{ .host_frac = 0.05 } },
    .{ .name = "cpu-qwen3-reduce-none-fits", .model = .qwen3, .cpu_only = true, .margin = .{ .host_frac = -0.5 } },
    .{ .name = "cpu-qwen35-reduce", .model = .qwen35, .cpu_only = true, .margin = .{ .host_frac = 0.3 } },
    .{ .name = "cpu-qwen35-reduce-streams", .model = .qwen35, .cpu_only = true, .margin = .{ .host_frac = 0.3 }, .n_seq_max = 2 },
    .{ .name = "cpu-qwen35-reduce-unified", .model = .qwen35, .cpu_only = true, .margin = .{ .host_frac = 0.3 }, .n_seq_max = 2, .kv_unified = true },
    .{ .name = "cpu-qwen3-ctx-set", .model = .qwen3, .cpu_only = true, .margin = .{ .host_frac = 0.5 }, .n_ctx = 16384 },
    .{ .name = "cpu-qwen3-min-max", .model = .qwen3, .cpu_only = true, .margin = .{ .host_frac = 0.5 }, .n_ctx_min = std.math.maxInt(u32) },
    .{ .name = "cpu-qwen3-min-above-train", .model = .qwen3, .cpu_only = true, .margin = .{ .host_frac = 0.5 }, .n_ctx_min = 65536 },
    .{ .name = "cpu-qwen3-min-small", .model = .qwen3, .cpu_only = true, .margin = .{ .host_frac = 0.2 }, .n_ctx_min = 512 },
    .{ .name = "cpu-qwen35-mtp", .model = .qwen35, .cpu_only = true, .margin = .{ .host_frac = 0.3 }, .load_mtp = true },
    .{ .name = "cpu-qwen35-debug-level", .model = .qwen35, .cpu_only = true, .margin = .{ .host_frac = 0.3 }, .log_level = c.GGML_LOG_LEVEL_DEBUG },

    // One fake GPU.
    .{ .name = "gpu1-qwen3-fits", .model = .qwen3, .devs = &.{d(1.5)}, .memory_data = true },
    .{ .name = "gpu1-qwen3-reduce-ctx", .model = .qwen3, .devs = &.{d(0.9)} },
    .{ .name = "gpu1-qwen3-layers-70", .model = .qwen3, .devs = &.{d(0.7)} },
    .{ .name = "gpu1-qwen3-layers-40", .model = .qwen3, .devs = &.{d(0.4)} },
    .{ .name = "gpu1-qwen3-layers-10", .model = .qwen3, .devs = &.{d(0.1)} },
    .{ .name = "gpu1-qwen3-layers-0", .model = .qwen3, .devs = &.{d(0.0)} },
    .{ .name = "gpu1-qwen35-layers-50", .model = .qwen35, .devs = &.{d(0.5)} },
    .{ .name = "gpu1-qwen35-layers-25-ctx-set", .model = .qwen35, .devs = &.{d(0.25)}, .n_ctx = 4096 },
    .{ .name = "gpu1-dense-layers-50", .model = .dense, .devs = &.{d(0.5)} },
    .{ .name = "gpu1-moe-fits", .model = .moe, .devs = &.{d(1.5)} },
    .{ .name = "gpu1-moe-80", .model = .moe, .devs = &.{d(0.8)} },
    .{ .name = "gpu1-moe-60", .model = .moe, .devs = &.{d(0.6)} },
    .{ .name = "gpu1-moe-40", .model = .moe, .devs = &.{d(0.4)} },
    .{ .name = "gpu1-moe-25", .model = .moe, .devs = &.{d(0.25)} },
    .{ .name = "gpu1-moe-10", .model = .moe, .devs = &.{d(0.1)} },
    .{ .name = "gpu1-moe-02", .model = .moe, .devs = &.{d(0.02)} },
    // A device holding exactly L layers at a context the user set, so step 3
    // runs and its search compares a measurement equal to its target.
    .{ .name = "gpu1-qwen3-tie-7", .model = .qwen3, .devs = &.{.{ .frac = 0, .tie_layers = 7 }}, .n_ctx = 4096 },
    .{ .name = "gpu1-qwen3-tie-14", .model = .qwen3, .devs = &.{.{ .frac = 0, .tie_layers = 14 }}, .n_ctx = 4096 },
    .{ .name = "gpu1-qwen3-tie-21", .model = .qwen3, .devs = &.{.{ .frac = 0, .tie_layers = 21 }}, .n_ctx = 4096 },
    .{ .name = "gpu1-qwen35-tie-12", .model = .qwen35, .devs = &.{.{ .frac = 0, .tie_layers = 12 }}, .n_ctx = 4096 },
    .{ .name = "gpu1-igpu", .model = .qwen3, .devs = &.{.{ .frac = 0.5, .typ = 1 }} },
    .{ .name = "gpu1-zero-gpu", .model = .qwen3, .devs = &.{.{ .frac = 0.5, .zero = true }} },
    .{ .name = "gpu1-zero-accel", .model = .qwen3, .devs = &.{.{ .frac = 0.5, .typ = 2, .zero = true }} },

    // Two and three fake GPUs.
    .{ .name = "gpu2-qwen3-fits", .model = .qwen3, .devs = &.{ d(1.0), d(1.0) }, .memory_data = true },
    .{ .name = "gpu2-qwen3-60-60", .model = .qwen3, .devs = &.{ d(0.6), d(0.6) } },
    .{ .name = "gpu2-qwen3-30-50", .model = .qwen3, .devs = &.{ d(0.3), d(0.5) } },
    .{ .name = "gpu2-qwen3-50-20", .model = .qwen3, .devs = &.{ d(0.5), d(0.2) } },
    .{ .name = "gpu2-qwen3-margins", .model = .qwen3, .devs = &.{ d(0.6), d(0.6) }, .margin = .{ .per_dev = &.{ 512, 2048 } } },
    .{ .name = "gpu2-qwen35-40-40", .model = .qwen35, .devs = &.{ d(0.4), d(0.4) } },
    .{ .name = "gpu2-moe-50-50", .model = .moe, .devs = &.{ d(0.5), d(0.5) } },
    .{ .name = "gpu2-moe-30-40", .model = .moe, .devs = &.{ d(0.3), d(0.4) } },
    .{ .name = "gpu2-moe-60-20", .model = .moe, .devs = &.{ d(0.6), d(0.2) } },
    .{ .name = "gpu2-moe-15-15", .model = .moe, .devs = &.{ d(0.15), d(0.15) } },
    .{ .name = "gpu3-qwen3-30-30-30", .model = .qwen3, .devs = &.{ d(0.3), d(0.3), d(0.3) } },
    .{ .name = "gpu3-moe-25-25-25", .model = .moe, .devs = &.{ d(0.25), d(0.25), d(0.25) } },
    .{ .name = "gpu3-moe-40-10-30", .model = .moe, .devs = &.{ d(0.4), d(0.1), d(0.3) } },

    // What the fit refuses to change. Each refusal comes after the context
    // reduction, so the device is too small even for the least context.
    .{ .name = "refuse-tensor-split-mode", .model = .qwen3, .devs = &.{d(0.5)}, .split_mode = c.LLAMA_SPLIT_MODE_TENSOR },
    .{ .name = "refuse-ngl-set", .model = .qwen3, .devs = &.{d(0.05)}, .n_gpu_layers = 99 },
    .{ .name = "refuse-ngl-set-fits", .model = .qwen3, .devs = &.{d(1.5)}, .n_gpu_layers = 99 },
    .{ .name = "refuse-user-split", .model = .qwen3, .devs = &.{ d(0.4), d(0.4) }, .user_split = true },
    .{ .name = "refuse-row-split", .model = .qwen3, .devs = &.{ d(0.4), d(0.4) }, .split_mode = c.LLAMA_SPLIT_MODE_ROW },
    .{ .name = "refuse-null-split-buf", .model = .qwen3, .devs = &.{ d(0.4), d(0.4) }, .null_split_buf = true },
    .{ .name = "refuse-null-tbo-buf", .model = .qwen3, .devs = &.{d(0.05)}, .null_tbo_buf = true },
    .{ .name = "refuse-user-tbo", .model = .qwen3, .devs = &.{d(0.05)}, .user_tbo = true },
    .{ .name = "null-split-buf-one-dev", .model = .qwen3, .devs = &.{d(0.5)}, .null_split_buf = true },
    .{ .name = "error-missing-model", .model = .missing, .memory_data = true },

    // A second model alongside.
    .{ .name = "extra-draft-cpu", .model = .qwen35, .cpu_only = true, .margin = .{ .host_frac = 0.3 }, .extra = .{ .model = .qwen3 } },
    .{ .name = "extra-mtp-cpu", .model = .qwen35, .cpu_only = true, .margin = .{ .host_frac = 0.3 }, .extra = .{ .model = .qwen35, .shares_model = true } },
    .{ .name = "extra-draft-gpu1", .model = .qwen3, .devs = &.{d(0.5)}, .extra = .{ .model = .dense } },
    .{ .name = "extra-draft-gpu2-moe", .model = .moe, .devs = &.{ d(0.4), d(0.4) }, .extra = .{ .model = .dense } },
    .{ .name = "extra-missing", .model = .qwen3, .devs = &.{d(0.5)}, .extra = .{ .model = .missing } },
};

/// One phrase per arm of `fit.cpp`; each must appear in some reference
/// transcript.
const required = [_][]const u8{
    "no changes needed",
    "projected memory use with initial parameters",
    "targets for free memory can be met on all devices",
    "context size reduced from",
    "entire model can be fit by reducing context",
    "entire model should be fit across devices by reducing context",
    "user has requested full context size",
    "which is <= the min. context size",
    "context size set by user to",
    "context size unset and KV cache not unified",
    "was unable to fit model into system memory",
    "filling dense layers back-to-front",
    "filling dense-only layers back-to-front",
    "start filling device",
    "set ngl_per_device_high[",
    "with only dense weights in device memory there is a total surplus",
    "with only dense weights in device memory there is still a total deficit",
    "converting dense-only layers to full layers",
    "set ngl_per_device[0].(n_layer, n_part)=",
    "set ngl_per_device_high[0].(n_layer, n_part)=",
    "overflow_type=LAYER_FRACTION_UP",
    "(n_layer, n_part, overflow_type)=",
    "overflow_type=LAYER_FRACTION_GATE",
    "overflow_type=LAYER_FRACTION_ATTN",
    "did not report memory; --fit will not use it",
    "getting device memory data for the extra model",
    "failed to measure the memory of the extra model",
    "not implemented for SPLIT_MODE_TENSOR",
    "n_gpu_layers already set by user",
    "model_params::tensor_split already set by user",
    "LLAMA_SPLIT_MODE_ROW not implemented",
    "did not provide a buffer to write the tensor_split to",
    "did not provide buffer to set tensor_buft_overrides",
    "model_params::tensor_buft_overrides already set by user",
    "encountered an error while trying to fit params",
    "successfully fit params to free device memory",
    "memory breakdown [MiB]",
    "overflow_type=4, mem=",
};

// -----------------------------------------------------------------------------
// Running a case

var model_paths: std.EnumArray(ModelKey, [:0]const u8) = undefined;

/// Per model: what one device holding all of it uses, and the host's use at
/// the least and the full context.
const Measured = struct { one_dev: usize, host_min: usize, host_full: usize, host_total: usize };
var measured: std.EnumArray(ModelKey, Measured) = undefined;

const Inputs = struct {
    devs: [8]c.ggml_backend_dev_t = @splat(null),
    mp: c.llama_model_params,
    cp: c.llama_context_params,
    emp: c.llama_model_params,
    ecp: c.llama_context_params,
    split_buf: [128]f32 = @splat(0),
    user_split: [128]f32 = @splat(0),
    tbo_buf: []c.llama_model_tensor_buft_override,
    user_tbo: [2]c.llama_model_tensor_buft_override,
    margins: [16]usize = @splat(1024 * MiB),

    fn init(cs: Case) !Inputs {
        var in: Inputs = .{
            .mp = api.model.llama_model_default_params(),
            .cp = api.context.llama_context_default_params(),
            .emp = api.model.llama_model_default_params(),
            .ecp = api.context.llama_context_default_params(),
            .tbo_buf = try gpa.alloc(c.llama_model_tensor_buft_override, api.llama.llama_max_tensor_buft_overrides()),
            .user_tbo = .{ .{ .pattern = "blk\\.0\\..*", .buft = c.ggml_backend_cpu_buffer_type() }, .{ .pattern = null, .buft = null } },
        };
        @memset(in.tbo_buf, .{ .pattern = null, .buft = null });
        const m = measured.get(cs.model);
        if (cs.devs.len > 0) {
            for (cs.devs, 0..) |dv, i| {
                const margin = switch (cs.margin) {
                    .mib => |v| v * MiB,
                    .per_dev => |v| v[i] * MiB,
                    .host_frac => unreachable,
                };
                const free: usize = margin + if (dv.tie_layers > 0)
                    try layersUse(cs.model, dv.tie_layers, cs.n_ctx)
                else
                    @as(usize, @intFromFloat(dv.frac * @as(f64, @floatFromInt(m.one_dev))));
                in.devs[i] = if (dv.zero) fit_dev_set(@intCast(i), 0, 0, dv.typ) else fit_dev_set(@intCast(i), free, 2 * free, dv.typ);
            }
            in.mp.devices = &in.devs;
            in.emp.devices = &in.devs;
        } else if (cs.cpu_only) {
            in.mp.devices = &in.devs; // { NULL }: no device but the CPU
            in.emp.devices = &in.devs;
        }
        switch (cs.margin) {
            .mib => |v| in.margins = @splat(v * MiB),
            .per_dev => |v| for (v, 0..) |x, i| {
                in.margins[i] = x * MiB;
            },
            .host_frac => |f| {
                const target = @as(f64, @floatFromInt(m.host_min)) + f * @as(f64, @floatFromInt(m.host_full -| m.host_min));
                in.margins = @splat(m.host_total -| @as(usize, @intFromFloat(@max(target, 0))));
            },
        }
        in.mp.n_gpu_layers = cs.n_gpu_layers;
        in.mp.split_mode = cs.split_mode;
        in.mp.load_mtp = cs.load_mtp;
        in.cp.n_ctx = cs.n_ctx;
        in.cp.n_seq_max = cs.n_seq_max;
        in.cp.kv_unified = cs.kv_unified;
        in.mp.tensor_split = &in.split_buf;
        in.mp.tensor_buft_overrides = in.tbo_buf.ptr;
        return in;
    }

    /// Pointers into `self` set after it reached its final address.
    fn settle(self: *Inputs, cs: Case) void {
        if (cs.devs.len > 0 or cs.cpu_only) {
            self.mp.devices = &self.devs;
            self.emp.devices = &self.devs;
        }
        self.mp.tensor_split = if (cs.user_split) blk: {
            self.user_split[0] = 1;
            self.user_split[1] = 3;
            break :blk &self.user_split;
        } else &self.split_buf;
        self.mp.tensor_buft_overrides = if (cs.user_tbo) &self.user_tbo else self.tbo_buf.ptr;
    }
};

fn pathOf(k: ModelKey) [:0]const u8 {
    return model_paths.get(k);
}

/// Writes what the fit left in `in` to `w`.
fn record(w: *std.Io.Writer, in: *const Inputs, status: c_int) !void {
    try w.print("status {d}\n", .{status});
    try w.print("n_ctx {d} extra_n_ctx {d}\n", .{ in.cp.n_ctx, in.ecp.n_ctx });
    try w.print("n_gpu_layers {d}\n", .{in.mp.n_gpu_layers});
    const ts: ?*const anyopaque = @ptrCast(in.mp.tensor_split);
    try w.print("tensor_split -> {s}\n", .{if (ts == null) "null" else if (ts == @as(*const anyopaque, @ptrCast(&in.split_buf))) "buf" else if (ts == @as(*const anyopaque, @ptrCast(&in.user_split))) "user" else "other"});
    for (in.split_buf, 0..) |v, i| if (v != 0) try w.print("  split[{d}] = {d}\n", .{ i, v });
    const tb: ?*const anyopaque = @ptrCast(in.mp.tensor_buft_overrides);
    try w.print("tensor_buft_overrides -> {s}\n", .{if (tb == null) "null" else if (tb == @as(*const anyopaque, @ptrCast(in.tbo_buf.ptr))) "buf" else if (tb == @as(*const anyopaque, @ptrCast(&in.user_tbo))) "user" else "other"});
    for (in.tbo_buf, 0..) |o, i| {
        if (o.pattern == null and o.buft == null) continue;
        const pat: []const u8 = if (o.pattern != null) std.mem.span(@as([*:0]const u8, @ptrCast(o.pattern))) else "(null)";
        const buft: []const u8 = if (o.buft != null) std.mem.span(c.ggml_backend_buft_name(o.buft)) else "(null)";
        try w.print("  tbo[{d}] = {s} -> {s}\n", .{ i, pat, buft });
    }
}

fn runRef(cs: Case, in: *Inputs) !c_int {
    c.llama_log_set(logCb, null);
    const extra_path: ?[*:0]const u8 = if (cs.extra) |e| pathOf(e.model).ptr else null;
    return fit_ref_params(pathOf(cs.model).ptr, &in.mp, &in.cp, if (cs.null_split_buf) null else &in.split_buf, if (cs.null_tbo_buf) null else in.tbo_buf.ptr, &in.margins, cs.n_ctx_min, extra_path, if (cs.extra != null) &in.emp else null, if (cs.extra != null) &in.ecp else null, if (cs.extra) |e| e.shares_model else false, cs.log_level);
}

fn runPort(cs: Case, in: *Inputs, log: fit.Log) !c_int {
    api.impl.llama_log_set(logCb, null);
    var patterns: fit.Patterns = try .init(gpa);
    // The patterns the overrides point at are compared before this frees them.
    defer patterns.deinit();
    var extra: fit.ExtraModel = undefined;
    if (cs.extra) |e| extra = .{ .path_model = pathOf(e.model).ptr, .mparams = &in.emp, .cparams = &in.ecp, .shares_model = e.shares_model };
    const status = try fit.fitParams(gpa, &patterns, log, pathOf(cs.model).ptr, &in.mp, &in.cp, if (cs.null_split_buf) null else &in.split_buf, if (cs.null_tbo_buf) null else in.tbo_buf.ptr, &in.margins, cs.n_ctx_min, if (cs.extra != null) &extra else null, @intCast(cs.log_level));
    var w: std.Io.Writer.Allocating = .init(gpa);
    defer w.deinit();
    try record(&w.writer, in, @backingInt(status));
    port_record = try gpa.dupe(u8, w.written());
    return @backingInt(status);
}

var port_record: []u8 = &.{};

/// One side's whole transcript of a case.
fn transcript(cs: Case, side: enum { ref, port }) ![]u8 {
    var in = try Inputs.init(cs);
    defer gpa.free(in.tbo_buf);
    in.settle(cs);
    llama_log.clearRetainingCapacity();

    var w: std.Io.Writer.Allocating = .init(gpa);
    defer w.deinit();
    var fit_log_w: std.Io.Writer.Allocating = .init(gpa);
    defer fit_log_w.deinit();

    switch (side) {
        .ref => {
            _ = try refLogSince();
            const status = try runRef(cs, &in);
            try record(&w.writer, &in, status);
            const text = try refLogSince();
            defer gpa.free(text);
            try fit_log_w.writer.writeAll(text);
        },
        .port => {
            const log: fit.Log = .{ .verbosity = fit.Log.level_trace, .out = &fit_log_w.writer, .prefix = true };
            _ = try runPort(cs, &in, log);
            defer gpa.free(port_record);
            try w.writer.writeAll(port_record);
        },
    }

    if (cs.memory_data) {
        // On fresh inputs: the port's override patterns are freed with its
        // cache once the fit is recorded.
        var fresh = try Inputs.init(cs);
        defer gpa.free(fresh.tbo_buf);
        fresh.settle(cs);
        try memoryData(&w.writer, cs, &fresh, side == .ref, &fit_log_w.writer);
    }

    try w.writer.writeAll("--- fit log\n");
    const masked = try maskTiming(fit_log_w.written());
    defer gpa.free(masked);
    try w.writer.writeAll(masked);
    try w.writer.writeAll("\n--- llama log\n");
    const masked_llama = try maskTiming(llama_log.items);
    defer gpa.free(masked_llama);
    try w.writer.writeAll(masked_llama);
    return gpa.dupe(u8, w.written());
}

/// `common_get_device_memory_data` and `common_fit_print` on the case's
/// inputs.
fn memoryData(w: *std.Io.Writer, cs: Case, in: *Inputs, ref: bool, fit_log: *std.Io.Writer) !void {
    var tmp_path_buf: [512]u8 = undefined;
    if (ref) {
        var out: [5 * 16]i64 = undefined;
        var devs: [16]c.ggml_backend_dev_t = undefined;
        var n_devs: c_int = 0;
        var hp: [3]u32 = .{ 0, 0, 0 };
        const n = fit_ref_device_memory(pathOf(cs.model).ptr, &in.mp, &in.cp, &out, 16, &devs, &n_devs, &hp, cs.log_level);
        try w.print("memory_data {d} devs {d} hp {d} {d} {d}\n", .{ n, if (n < 0) 0 else n_devs, hp[0], hp[1], hp[2] });
        if (n > 0) for (0..@intCast(n)) |i| try w.print("  {d} {d} {d} {d} {d}\n", .{ out[5 * i], out[5 * i + 1], out[5 * i + 2], out[5 * i + 3], out[5 * i + 4] });
        if (n > 0) for (devs[0..@intCast(n_devs)]) |dv| try w.print("  dev {s}\n", .{std.mem.span(c.ggml_backend_dev_name(dv))});
        const text = try refLogSince();
        defer gpa.free(text);
        try fit_log.writeAll(text);

        const p = try std.fmt.bufPrintSentinel(&tmp_path_buf, "{s}.print", .{ref_log_path}, 0);
        const rc = fit_ref_print(pathOf(cs.model).ptr, &in.mp, &in.cp, p.ptr);
        try w.print("fit_print {d}\n", .{rc});
        const f = std.c.fopen(p.ptr, "rb") orelse return error.NoPrint;
        defer _ = std.c.fclose(f);
        var buf: [4096]u8 = undefined;
        const got = std.c.fread(&buf, 1, buf.len, f);
        if (rc == 0) try w.writeAll(buf[0..got]);
        const text2 = try refLogSince();
        defer gpa.free(text2);
        try fit_log.writeAll(text2);
    } else {
        const log: fit.Log = .{ .verbosity = fit.Log.level_trace, .out = fit_log, .prefix = true };
        var devs: std.ArrayList(c.ggml_backend_dev_t) = .empty;
        defer devs.deinit(gpa);
        var hp: [3]u32 = .{ 0, 0, 0 };
        if (fit.getDeviceMemoryData(gpa, log, pathOf(cs.model).ptr, &in.mp, &in.cp, &devs, &hp[0], &hp[1], &hp[2], @intCast(cs.log_level))) |v| {
            defer gpa.free(v);
            try w.print("memory_data {d} devs {d} hp {d} {d} {d}\n", .{ v.len, devs.items.len, hp[0], hp[1], hp[2] });
            for (v) |e| try w.print("  {d} {d} {d} {d} {d}\n", .{ e.total, e.free, @as(i64, @bitCast(e.model)), @as(i64, @bitCast(e.context)), @as(i64, @bitCast(e.compute)) });
            for (devs.items) |dv| try w.print("  dev {s}\n", .{std.mem.span(c.ggml_backend_dev_name(dv))});
        } else |e| switch (e) {
            error.Runtime => try w.print("memory_data -1 devs 0 hp {d} {d} {d}\n", .{ hp[0], hp[1], hp[2] }),
            else => return e,
        }

        var pw: std.Io.Writer.Allocating = .init(gpa);
        defer pw.deinit();
        if (fit.fitPrint(gpa, log, &pw.writer, pathOf(cs.model).ptr, &in.mp, &in.cp)) {
            try w.print("fit_print 0\n", .{});
            try w.writeAll(pw.written());
        } else |e| switch (e) {
            error.Runtime => try w.print("fit_print -1\n", .{}),
            else => return e,
        }
    }
}

/// What one fake device holds with the last `layers` layers on it, at
/// `n_ctx`: the measurement `get_memory_for_layers` makes for that many
/// layers, through `n_gpu_layers` -- the same model parameters, since a
/// dense model on one device needs no override.
fn layersUse(k: ModelKey, layers: u32, n_ctx: u32) !usize {
    var hp: [3]u32 = undefined;
    var devs: std.ArrayList(c.ggml_backend_dev_t) = .empty;
    defer devs.deinit(gpa);
    var dev_list: [2]c.ggml_backend_dev_t = .{ fit_dev_set(0, 1 << 40, 1 << 41, 0), null };
    var mp = api.model.llama_model_default_params();
    var cp = api.context.llama_context_default_params();
    cp.n_ctx = n_ctx;
    mp.devices = &dev_list;
    mp.n_gpu_layers = @intCast(layers);
    const v = try fit.getDeviceMemoryData(gpa, .{}, pathOf(k).ptr, &mp, &cp, &devs, &hp[0], &hp[1], &hp[2], c.GGML_LOG_LEVEL_ERROR);
    defer gpa.free(v);
    return v[0].model + v[0].context + v[0].compute;
}

/// Fills `measured` for one model from the port: one fake device holding
/// everything, and the CPU alone at the least and the full context.
fn measure(k: ModelKey) !void {
    if (k == .missing) {
        measured.set(k, .{ .one_dev = 1024 * MiB, .host_min = 0, .host_full = 0, .host_total = 0 });
        return;
    }
    const log: fit.Log = .{};
    var hp: [3]u32 = undefined;
    var devs: std.ArrayList(c.ggml_backend_dev_t) = .empty;
    defer devs.deinit(gpa);

    var dev_list: [2]c.ggml_backend_dev_t = .{ fit_dev_set(0, 1 << 40, 1 << 41, 0), null };
    var mp = api.model.llama_model_default_params();
    var cp = api.context.llama_context_default_params();
    // The cases' own context, the trained one: their fractions mean what
    // they say only against it.
    cp.n_ctx = 0;
    mp.devices = &dev_list;
    const one = try fit.getDeviceMemoryData(gpa, log, pathOf(k).ptr, &mp, &cp, &devs, &hp[0], &hp[1], &hp[2], c.GGML_LOG_LEVEL_ERROR);
    defer gpa.free(one);
    const one_dev: usize = @as(usize, @bitCast(one[0].model)) + @as(usize, @bitCast(one[0].context)) + @as(usize, @bitCast(one[0].compute));

    var none: [1]c.ggml_backend_dev_t = .{null};
    mp.devices = &none;
    cp.n_ctx = 4096;
    const lo = try fit.getDeviceMemoryData(gpa, log, pathOf(k).ptr, &mp, &cp, &devs, &hp[0], &hp[1], &hp[2], c.GGML_LOG_LEVEL_ERROR);
    defer gpa.free(lo);
    cp.n_ctx = 0;
    const hi = try fit.getDeviceMemoryData(gpa, log, pathOf(k).ptr, &mp, &cp, &devs, &hp[0], &hp[1], &hp[2], c.GGML_LOG_LEVEL_ERROR);
    defer gpa.free(hi);
    const host = struct {
        fn total(e: fit.DeviceMemoryData) usize {
            return e.model + e.context + e.compute;
        }
    }.total;
    measured.set(k, .{
        .one_dev = one_dev,
        .host_min = host(lo[lo.len - 1]),
        .host_full = host(hi[hi.len - 1]),
        .host_total = @bitCast(hi[hi.len - 1].total),
    });
    std.debug.print("{s}: one device {d} MiB, host {d}..{d} MiB of {d} MiB, {d} layers, n_ctx_train {d}, {d} experts\n", .{
        @tagName(k),                      one_dev / MiB, measured.get(k).host_min / MiB, measured.get(k).host_full / MiB,
        measured.get(k).host_total / MiB, hp[0],         hp[1],                          hp[2],
    });
}

fn firstDifference(a: []const u8, b: []const u8) void {
    var ia = std.mem.splitScalar(u8, a, '\n');
    var ib = std.mem.splitScalar(u8, b, '\n');
    var line: usize = 1;
    var shown: usize = 0;
    while (true) : (line += 1) {
        const la = ia.next();
        const lb = ib.next();
        if (la == null and lb == null) return;
        if (la != null and lb != null and std.mem.eql(u8, la.?, lb.?)) continue;
        std.debug.print("    line {d}:\n      ref:  {s}\n      port: {s}\n", .{ line, la orelse "(end)", lb orelse "(end)" });
        shown += 1;
        if (shown == 6) return;
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 5) {
        std.debug.print("usage: fit_diff <qwen35.gguf> <qwen3.gguf> <moe.gguf> <dense.gguf> [--case <substring>]\n", .{});
        std.process.exit(2);
    }
    model_paths.set(.qwen35, try gpa.dupeSentinel(u8, args[1], 0));
    model_paths.set(.qwen3, try gpa.dupeSentinel(u8, args[2], 0));
    model_paths.set(.moe, try gpa.dupeSentinel(u8, args[3], 0));
    model_paths.set(.dense, try gpa.dupeSentinel(u8, args[4], 0));
    model_paths.set(.missing, "/nonexistent/fit-diff-missing.gguf");
    var only: ?[]const u8 = null;
    var i: usize = 5;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--case") and i + 1 < args.len) {
            only = args[i + 1];
            i += 1;
        }
    }
    const out_dir: ?[]const u8 = if (std.c.getenv("FIT_DIFF_OUT")) |o| std.mem.span(o) else null;
    const tmp_dir: []const u8 = if (std.c.getenv("FIT_DIFF_TMP")) |t| std.mem.span(t) else "/tmp";
    ref_log_path = try std.fmt.allocPrintSentinel(gpa, "{s}/fit-ref.log", .{tmp_dir}, 0);

    c.llama_log_set(logCb, null);
    api.impl.llama_log_set(logCb, null);
    c.llama_backend_init();
    fit_ref_log_setup(ref_log_path.ptr, fit.Log.level_trace);

    for (std.enums.values(ModelKey)) |k| try measure(k);

    // The first Metal initialisation in a process logs what later ones do
    // not; one throwaway fit on each side puts both past it.
    {
        const warm: Case = .{ .name = "warm-up", .model = .qwen3 };
        gpa.free(try transcript(warm, .ref));
        gpa.free(try transcript(warm, .port));
    }

    var seen: [required.len]bool = @splat(false);
    var failures: usize = 0;
    var ran: usize = 0;
    for (cases) |cs| {
        if (only) |o| if (std.mem.indexOf(u8, cs.name, o) == null) continue;
        ran += 1;
        // FIT_DIFF_SIDE=ref|port runs one side alone, for a case that crashes.
        if (std.c.getenv("FIT_DIFF_SIDE")) |one| {
            const t = try transcript(cs, if (std.mem.eql(u8, std.mem.span(one), "port")) .port else .ref);
            defer gpa.free(t);
            std.debug.print("{s}\n", .{t});
            continue;
        }
        const a = try transcript(cs, .ref);
        defer gpa.free(a);
        const b = try transcript(cs, .port);
        defer gpa.free(b);
        for (required, 0..) |phrase, k| {
            if (std.mem.indexOf(u8, a, phrase) != null) seen[k] = true;
        }
        if (out_dir) |dir| {
            var pbuf: [512]u8 = undefined;
            for ([_][]const u8{ "ref", "port" }, [_][]const u8{ a, b }) |side, text| {
                const p = try std.fmt.bufPrintSentinel(&pbuf, "{s}/{s}.{s}", .{ dir, cs.name, side }, 0);
                const f = std.c.fopen(p.ptr, "wb") orelse continue;
                _ = std.c.fwrite(text.ptr, 1, text.len, f);
                _ = std.c.fclose(f);
            }
        }
        const lines = std.mem.count(u8, a, "\n");
        if (std.mem.eql(u8, a, b)) {
            std.debug.print("  {s:<34} identical ({d} lines)\n", .{ cs.name, lines });
        } else {
            std.debug.print("  {s:<34} DIFFERS\n", .{cs.name});
            firstDifference(a, b);
            failures += 1;
        }
    }

    var missing_arms: usize = 0;
    if (only == null) {
        for (required, seen) |phrase, s| {
            if (!s) {
                std.debug.print("  arm never reached: \"{s}\"\n", .{phrase});
                missing_arms += 1;
            }
        }
    }
    if (failures != 0 or missing_arms != 0) {
        std.debug.print("FAIL: {d} of {d} cases differ, {d} arms never reached\n", .{ failures, ran, missing_arms });
        std.process.exit(1);
    }
    std.debug.print("PASS: {d} cases identical, every arm of fit.cpp reached\n", .{ran});
}
