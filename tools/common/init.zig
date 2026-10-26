//! `common_init_from_params`: load the model and context a tool's command
//! line describes.
//!
//! # Provenance
//!
//! Ports `common_init_result`'s constructor, `common_init_from_params`,
//! `common_init_sampler_from_model`, `common_control_vector_load`,
//! `common_set_adapter_lora`, `common_threadpools::init`,
//! `ggml_threadpool_params_from_cpu_params` and `set_process_priority` from
//! `llama.cpp/common/common.cpp`, and `common_sampler_init` from
//! `common/sampling.cpp` (v0.3.0, `c1d0e7a00`). Each declaration names what
//! it ports and the line it began at.
//!
//! # The sampler, for the tools
//!
//! `common_sampler_init` builds a chain whether or not a tool samples. The
//! ported tools never do, and set none of what the full routine reads beyond
//! the chain -- grammar triggers, a generation prompt, a reasoning budget --
//! so those arms are not ported; the chain itself, its grammar and logit
//! biases are, so a bad `--grammar` fails as upstream's does.

const std = @import("std");
const builtin = @import("builtin");
const c = @import("tools_h");
const params_mod = @import("params.zig");
const log = @import("log.zig");
const common = @import("common.zig");
const arg = @import("arg.zig");
const fit = @import("cli_fit");
const sampling = @import("sampling.zig");

const Params = params_mod.Params;

/// `COM_*`'s `cmn  %12.*s: ` prefix.
fn comPrefix(comptime fmt: []const u8) [:0]const u8 {
    return "cmn  %12.*s: " ++ fmt;
}

fn comTrc(comptime func: [*:0]const u8, comptime fmt: []const u8, args: anytype) void {
    log.trc(comPrefix(fmt), .{ @as(c_int, 12), func } ++ args);
}

fn comInf(comptime func: [*:0]const u8, comptime fmt: []const u8, args: anytype) void {
    log.inf(comPrefix(fmt), .{ @as(c_int, 12), func } ++ args);
}

fn comWrn(comptime func: [*:0]const u8, comptime fmt: []const u8, args: anytype) void {
    log.wrn(comPrefix(fmt), .{ @as(c_int, 12), func } ++ args);
}

fn comErr(comptime func: [*:0]const u8, comptime fmt: []const u8, args: anytype) void {
    log.err(comPrefix(fmt), .{ @as(c_int, 12), func } ++ args);
}

/// Ports `common_init_result` (common/common.h:912 @c1d0e7a00).
pub const InitResult = struct {
    model: ?*c.llama_model = null,
    context: ?*c.llama_context = null,
    lora: std.ArrayList(*c.llama_adapter_lora) = .empty,
    samplers: std.ArrayList(*sampling.Sampler) = .empty,
    threadpool: ?*c.ggml_threadpool = null,
    threadpool_batch: ?*c.ggml_threadpool = null,

    /// Ports `reset_samplers` (common/common.cpp:1426 @c1d0e7a00), a member of `common_init_result`.
    fn resetSamplers(self: *InitResult) void {
        for (self.samplers.items) |s| c.llama_sampler_reset(s.get());
    }

    /// Ports `sampler` (common/common.cpp:1419 @c1d0e7a00), a member of
    /// `common_init_result`.
    ///
    /// Return: the sampler for `seq_id`, or null past the last; owned by
    /// the result.
    pub fn sampler(self: *const InitResult, seq_id: c.llama_seq_id) ?*sampling.Sampler {
        if (seq_id < 0 or seq_id >= @as(c_int, @intCast(self.samplers.items.len))) return null;
        return self.samplers.items[@intCast(seq_id)];
    }

    /// The C++ destructor's order: context, model, adapters, samplers, pools.
    pub fn deinit(self: *InitResult) void {
        for (self.samplers.items) |s| s.destroy();
        if (self.context) |ctx| c.llama_free(ctx);
        if (self.model) |m| c.llama_model_free(m);
        if (self.threadpool) |t| c.ggml_threadpool_free(t);
        if (self.threadpool_batch) |t| c.ggml_threadpool_free(t);
    }
};

/// Ports `common_init_sampler_from_model` (common/common.cpp:1213 @c1d0e7a00): the
/// model's recommended sampling, where the command line set none.
fn initSamplerFromModel(p: *Params, model: ?*const c.llama_model) !void {
    const sp = &p.sampling;
    const config = sp.user_sampling_config;
    const Cfg = params_mod.SamplingConfig;

    const Get = struct {
        fn int32(m: ?*const c.llama_model, cfg: u64, key: c.llama_model_meta_key, dst: *i32, user: u64) void {
            if (cfg & user != 0) return;
            var buf: [64]u8 = @splat(0);
            if (c.llama_model_meta_val_str(m, c.llama_model_meta_key_str(key), &buf, buf.len) > 0) {
                var end: ?[*:0]u8 = null;
                const v = std.c.strtol(@ptrCast(&buf), @ptrCast(&end), 10);
                if (end != null and end != @as([*:0]u8, @ptrCast(&buf))) dst.* = @truncate(v);
            }
        }
        fn float(m: ?*const c.llama_model, cfg: u64, key: c.llama_model_meta_key, dst: *f32, user: u64) void {
            if (cfg & user != 0) return;
            var buf: [128]u8 = @splat(0);
            if (c.llama_model_meta_val_str(m, c.llama_model_meta_key_str(key), &buf, buf.len) > 0) {
                var end: ?[*:0]u8 = null;
                const v = strtof(@ptrCast(&buf), &end);
                if (end != null and end != @as([*:0]u8, @ptrCast(&buf))) dst.* = v;
            }
        }
    };

    // Sampling sequence
    if (config & Cfg.samplers == 0) {
        var buf: [512]u8 = @splat(0);
        if (c.llama_model_meta_val_str(model, c.llama_model_meta_key_str(c.LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE), &buf, buf.len) > 0) {
            const s = std.mem.sliceTo(&buf, 0);
            var names: std.ArrayList([]const u8) = .empty;
            var it = std.mem.splitScalar(u8, s, ';');
            while (it.next()) |n| try names.append(p.arena, n);
            if (names.items.len != 0) sp.samplers = try @import("arg_options.zig").samplerTypesFromNamesPub(p, names.items);
        }
    }

    Get.int32(model, config, c.LLAMA_MODEL_META_KEY_SAMPLING_TOP_K, &sp.top_k, Cfg.top_k);
    Get.float(model, config, c.LLAMA_MODEL_META_KEY_SAMPLING_TOP_P, &sp.top_p, Cfg.top_p);
    Get.float(model, config, c.LLAMA_MODEL_META_KEY_SAMPLING_MIN_P, &sp.min_p, Cfg.min_p);
    Get.float(model, config, c.LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY, &sp.xtc_probability, Cfg.xtc_probability);
    Get.float(model, config, c.LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD, &sp.xtc_threshold, Cfg.xtc_threshold);
    Get.float(model, config, c.LLAMA_MODEL_META_KEY_SAMPLING_TEMP, &sp.temp, Cfg.temp);
    Get.int32(model, config, c.LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N, &sp.penalty_last_n, Cfg.penalty_last_n);
    Get.float(model, config, c.LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT, &sp.penalty_repeat, Cfg.penalty_repeat);
    Get.int32(model, config, c.LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT, &sp.mirostat, Cfg.mirostat);
    Get.float(model, config, c.LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU, &sp.mirostat_tau, Cfg.mirostat_tau);
    Get.float(model, config, c.LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA, &sp.mirostat_eta, Cfg.mirostat_eta);
}

extern fn strtof(s: [*:0]const u8, end: *?[*:0]u8) f32;

/// What `common_sampler_init` throws; `common_init_from_params` does not
/// catch it, so it escapes `main`.
pub const SamplerError = error{ InvalidArgument, Runtime };

/// Ports `common_sampler_init` (sampling.cpp:187 @c1d0e7a00), the chain
/// and its grammar; see the file header for the arms left out.
///
/// Return: the sampler, which the caller owns (`Sampler.destroy`).
pub fn samplerInit(p: *Params, model: ?*const c.llama_model) !*sampling.Sampler {
    const sp = &p.sampling;
    if (!std.math.isFinite(sp.penalty_repeat) or sp.penalty_repeat <= 0.0 or !std.math.isFinite(1.0 / sp.penalty_repeat)) {
        _ = arg.throwStr("penalty_repeat must be finite and greater than 0") catch {};
        return error.InvalidArgument;
    }
    if (!std.math.isFinite(sp.penalty_freq)) {
        _ = arg.throwStr("penalty_freq must be finite") catch {};
        return error.InvalidArgument;
    }
    if (!std.math.isFinite(sp.penalty_present)) {
        _ = arg.throwStr("penalty_present must be finite") catch {};
        return error.InvalidArgument;
    }
    const vocab = c.llama_model_get_vocab(model);
    var lparams = c.llama_sampler_chain_default_params();
    lparams.no_perf = sp.no_perf;

    var grmr: ?*c.llama_sampler = null;
    const chain = c.llama_sampler_chain_init(lparams);

    const grammar_str = sp.grammar;
    if (std.mem.startsWith(u8, grammar_str, "%llguidance")) {
        _ = c.ggml_abort("sampling.cpp", 217, "llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
    } else if (grammar_str.len != 0) {
        const g = try p.arena.dupeSentinel(u8, grammar_str, 0);
        grmr = c.llama_sampler_init_grammar(vocab, g.ptr, "root");
    }
    if (grmr == null and grammar_str.len != 0) {
        _ = arg.throwStr("failed to parse grammar") catch {};
        return error.Runtime;
    }

    var samplers: std.ArrayList(*c.llama_sampler) = .empty;

    // logit bias: user biases + model suppress tokens (-INFINITY)
    {
        var merged: std.ArrayList(c.llama_logit_bias) = .empty;
        try merged.appendSlice(p.arena, sp.logit_bias.items);

        var n_suppress: i32 = 0;
        const suppress = c.llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        var i: i32 = 0;
        while (i < n_suppress) : (i += 1) try merged.append(p.arena, .{ .token = suppress[@intCast(i)], .bias = -std.math.inf(f32) });

        if (merged.items.len != 0) {
            try samplers.append(p.arena, c.llama_sampler_init_logit_bias(c.llama_vocab_n_tokens(vocab), @intCast(merged.items.len), merged.items.ptr).?);
        }
    }

    if (sp.mirostat == 0) {
        var use_adaptive_p = false; // see below
        for (sp.samplers.items) |cnstr| {
            const s: ?*c.llama_sampler = switch (cnstr) {
                .dry => blk: {
                    var breakers: std.ArrayList([*:0]const u8) = .empty;
                    for (sp.dry_sequence_breakers.items) |b| try breakers.append(p.arena, (try p.arena.dupeSentinel(u8, b, 0)).ptr);
                    break :blk c.llama_sampler_init_dry(vocab, sp.dry_multiplier, sp.dry_base, sp.dry_allowed_length, sp.dry_penalty_last_n, @ptrCast(breakers.items.ptr), breakers.items.len);
                },
                .top_k => c.llama_sampler_init_top_k(sp.top_k),
                .top_p => c.llama_sampler_init_top_p(sp.top_p, @intCast(sp.min_keep)),
                .top_n_sigma => c.llama_sampler_init_top_n_sigma(sp.top_n_sigma),
                .min_p => c.llama_sampler_init_min_p(sp.min_p, @intCast(sp.min_keep)),
                .xtc => c.llama_sampler_init_xtc(sp.xtc_probability, sp.xtc_threshold, @intCast(sp.min_keep), sp.seed),
                .typical_p => c.llama_sampler_init_typical(sp.typ_p, @intCast(sp.min_keep)),
                .temperature => c.llama_sampler_init_temp_ext(sp.temp, sp.dynatemp_range, sp.dynatemp_exponent),
                .infill => c.llama_sampler_init_infill(vocab),
                .penalties => c.llama_sampler_init_penalties(c.llama_vocab_n_tokens(vocab), sp.penalty_last_n, sp.penalty_repeat, sp.penalty_freq, sp.penalty_present),
                .adaptive_p => blk: {
                    // the `adaptive-p` sampler selects a single token, like
                    // `dist`; it goes at the end of the chain
                    use_adaptive_p = true;
                    break :blk null;
                },
                else => c.ggml_abort("sampling.cpp", 391, "GGML_ASSERT(%s) failed", "false && \"unknown sampler type\""),
            };
            if (s) |smpl| try samplers.append(p.arena, smpl);
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            try samplers.append(p.arena, c.llama_sampler_init_adaptive_p(sp.adaptive_target, sp.adaptive_decay, sp.seed).?);
        } else {
            // default: sample from distribution
            try samplers.append(p.arena, c.llama_sampler_init_dist(sp.seed).?);
        }
    } else if (sp.mirostat == 1) {
        try samplers.append(p.arena, c.llama_sampler_init_temp(sp.temp).?);
        try samplers.append(p.arena, c.llama_sampler_init_mirostat(c.llama_vocab_n_tokens(vocab), sp.seed, sp.mirostat_tau, sp.mirostat_eta, 100).?);
    } else if (sp.mirostat == 2) {
        try samplers.append(p.arena, c.llama_sampler_init_temp(sp.temp).?);
        try samplers.append(p.arena, c.llama_sampler_init_mirostat_v2(sp.seed, sp.mirostat_tau, sp.mirostat_eta).?);
    } else {
        _ = c.ggml_abort("sampling.cpp", 408, "GGML_ASSERT(%s) failed", "false && \"unknown mirostat version\"");
    }

    for (samplers.items) |smpl| c.llama_sampler_chain_add(chain, smpl);

    if (grmr != null and sp.backend_sampling) {
        log.wrn("%s: backend sampling is not compatible with grammar, disabling\n", .{"common_sampler_init"});
        sp.backend_sampling = false;
    }
    return sampling.Sampler.create(std.heap.c_allocator, chain.?, grmr, sp.n_prev, sp.grammar_lazy);
}

/// Ports `ggml_threadpool_params_from_cpu_params` (common/common.cpp:1761 @c1d0e7a00).
fn threadpoolParams(cp: *const params_mod.CpuParams) c.ggml_threadpool_params {
    var tpp: c.ggml_threadpool_params = undefined;
    c.ggml_threadpool_params_init(&tpp, cp.n_threads); // setup the defaults
    if (cp.mask_valid) @memcpy(tpp.cpumask[0..c.GGML_MAX_N_THREADS], cp.cpumask[0..]);
    tpp.prio = cp.priority;
    tpp.poll = cp.poll;
    tpp.strict_cpu = cp.strict_cpu;
    return tpp;
}

/// Ports `init` (common/common.cpp:1785 @c1d0e7a00), a member of `common_threadpools`.
fn threadpoolsInit(res: *InitResult, ctx: ?*c.llama_context, p: *const Params) void {
    const func = "init";
    comInf(func, "llama threadpool init, n_threads = %d\n", .{p.cpuparams.n_threads});

    const cpu_dev = c.ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == null) {
        comWrn(func, "%s", .{@as([*:0]const u8, "no CPU backend found\n")});
        return;
    }

    const tpp_batch = threadpoolParams(&p.cpuparams_batch);
    var tpp = threadpoolParams(&p.cpuparams);

    // each pool needs to match the respective n_threads exactly
    if (!c.ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        res.threadpool_batch = c.ggml_threadpool_new(@constCast(&tpp_batch));
        if (res.threadpool_batch == null) {
            comWrn(func, "batch threadpool create failed : n_threads %d\n", .{tpp_batch.n_threads});
            return;
        }
        // start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    res.threadpool = c.ggml_threadpool_new(&tpp);
    if (res.threadpool == null) {
        comWrn(func, "threadpool create failed : n_threads %d\n", .{tpp.n_threads});
        if (res.threadpool_batch) |t| c.ggml_threadpool_free(t);
        res.threadpool_batch = null;
        return;
    }

    c.llama_attach_threadpool(ctx, res.threadpool, res.threadpool_batch);
}

extern fn setpriority(which: c_int, who: c_uint, prio: c_int) c_int;
extern fn strerror(e: c_int) [*:0]const u8;

/// Ports `set_process_priority` (common/common.cpp:260 @c1d0e7a00), the POSIX arm;
/// Windows has its own, through `SetPriorityClass`.
pub fn setProcessPriority(prio: c.ggml_sched_priority) bool {
    if (prio == c.GGML_SCHED_PRIO_NORMAL) return true;
    if (comptime builtin.os.tag == .windows) {
        const W = struct {
            extern "kernel32" fn SetPriorityClass(h: ?*anyopaque, p: u32) callconv(.winapi) c_int;
            extern "kernel32" fn GetCurrentProcess() callconv(.winapi) ?*anyopaque;
            extern "kernel32" fn GetLastError() callconv(.winapi) u32;
        };
        const pc: u32 = switch (prio) {
            c.GGML_SCHED_PRIO_LOW => 0x4000, // BELOW_NORMAL_PRIORITY_CLASS
            c.GGML_SCHED_PRIO_MEDIUM => 0x8000, // ABOVE_NORMAL_PRIORITY_CLASS
            c.GGML_SCHED_PRIO_HIGH => 0x80, // HIGH_PRIORITY_CLASS
            c.GGML_SCHED_PRIO_REALTIME => 0x100, // REALTIME_PRIORITY_CLASS
            else => 0x20, // NORMAL_PRIORITY_CLASS
        };
        if (W.SetPriorityClass(W.GetCurrentProcess(), pc) == 0) {
            comWrn("set_process_priority", "failed to set process priority class %d : (%d)\n", .{ @as(c_int, @intCast(prio)), @as(c_int, @intCast(W.GetLastError())) });
            return false;
        }
        return true;
    }
    const v: c_int = switch (prio) {
        c.GGML_SCHED_PRIO_LOW => 5,
        c.GGML_SCHED_PRIO_MEDIUM => -5,
        c.GGML_SCHED_PRIO_HIGH => -10,
        c.GGML_SCHED_PRIO_REALTIME => -20,
        else => 0,
    };
    if (setpriority(0, 0, v) != 0) {
        const e = std.c._errno().*;
        comWrn("set_process_priority", "failed to set process priority %d : %s (%d)\n", .{ @as(c_int, @intCast(prio)), strerror(e), e });
        return false;
    }
    return true;
}

/// Ports `common_control_vector_data` (common/common.h:1081 @c1d0e7a00).
const ControlVectorData = struct { n_embd: i32, data: std.ArrayList(f32) };

/// Ports `common_control_vector_load_one` (common/common.cpp:2002 @c1d0e7a00).
fn controlVectorLoadOne(a: std.mem.Allocator, info: params_mod.ControlVectorLoadInfo) !ControlVectorData {
    const func = "common_control_vector_load_one";
    var result: ControlVectorData = .{ .n_embd = -1, .data = .empty };

    var ctx: ?*c.ggml_context = null;
    const meta: c.gguf_init_params = .{ .no_alloc = false, .ctx = &ctx };
    const ctx_gguf = c.gguf_init_from_file(info.fname.ptr, meta) orelse {
        comErr(func, "failed to load control vector file from %s\n", .{info.fname.ptr});
        return result;
    };

    const n_tensors: i32 = @truncate(c.gguf_get_n_tensors(ctx_gguf));
    if (n_tensors == 0) comWrn(func, "no direction tensors found in %s\n", .{info.fname.ptr});

    var i: i32 = 0;
    while (i < n_tensors) : (i += 1) {
        const name = std.mem.span(c.gguf_get_tensor_name(ctx_gguf, i));

        var layer_idx: c_int = -1;

        // split on '.'
        if (std.mem.indexOfScalar(u8, name, '.')) |dotpos| {
            if (std.mem.eql(u8, name[0..dotpos], "direction")) {
                const z = try a.dupeSentinel(u8, name[dotpos + 1 ..], 0);
                layer_idx = arg.stoi(z) catch -1;
            }
        }
        if (layer_idx < 0) {
            comErr(func, "invalid/unparsable direction tensor layer index in %s\n", .{info.fname.ptr});
            result.n_embd = -1;
            break;
        } else if (layer_idx == 0) {
            comErr(func, "invalid (zero) direction tensor layer index in %s\n", .{info.fname.ptr});
            result.n_embd = -1;
            break;
        }

        const tensor = c.ggml_get_tensor(ctx, name.ptr);
        if (tensor.*.type != c.GGML_TYPE_F32) {
            comErr(func, "invalid (non-F32) direction tensor type in %s\n", .{info.fname.ptr});
            result.n_embd = -1;
            break;
        }
        if (c.ggml_n_dims(tensor) != 1) {
            comErr(func, "invalid (non-1D) direction tensor shape in %s\n", .{info.fname.ptr});
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result.n_embd = @intCast(c.ggml_nelements(tensor));
        } else if (c.ggml_nelements(tensor) != result.n_embd) {
            comErr(func, "direction tensor in %s does not match previous dimensions\n", .{info.fname.ptr});
            result.n_embd = -1;
            break;
        }

        // extend if necessary - do not store data for layer 0 (it's not used)
        const need: usize = @intCast(result.n_embd * layer_idx);
        if (result.data.items.len < need) try result.data.appendNTimes(a, 0.0, need - result.data.items.len);

        const src: [*]const f32 = @ptrCast(@alignCast(tensor.*.data));
        const dst = result.data.items[@intCast(result.n_embd * (layer_idx - 1))..]; // layer 1 at [0]
        for (0..@intCast(result.n_embd)) |j| {
            // allows multiple directions for same layer in same file
            dst[j] = @mulAdd(f32, src[j], info.strength, dst[j]);
        }
    }

    if (result.n_embd == -1) {
        comWrn(func, "skipping %s due to invalid direction tensors\n", .{info.fname.ptr});
        result.data.clearRetainingCapacity();
    }

    c.gguf_free(ctx_gguf);
    c.ggml_free(ctx);
    return result;
}

/// Ports `common_control_vector_load` (common/common.cpp:2087 @c1d0e7a00).
fn controlVectorLoad(a: std.mem.Allocator, infos: []const params_mod.ControlVectorLoadInfo) !ControlVectorData {
    const func = "common_control_vector_load";
    var result: ControlVectorData = .{ .n_embd = -1, .data = .empty };

    for (infos) |info| {
        const cur = try controlVectorLoadOne(a, info);

        if (cur.n_embd == -1) {
            result.n_embd = -1;
            break;
        }
        if (result.n_embd != -1 and result.n_embd != cur.n_embd) {
            comErr(func, "control vectors in %s does not match previous dimensions\n", .{info.fname.ptr});
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result = cur;
        } else {
            if (result.data.items.len < cur.data.items.len) try result.data.appendNTimes(a, 0.0, cur.data.items.len - result.data.items.len); // extend if necessary
            for (cur.data.items, 0..) |v, i| result.data.items[i] += v;
        }
    }

    if (result.n_embd == -1) {
        comErr(func, "%s", .{@as([*:0]const u8, "no valid control vector files passed\n")});
        result.data.clearRetainingCapacity();
    }
    return result;
}

/// Ports `common_init_result` (common/common.cpp:1290 @c1d0e7a00), a member of `common_init_result`.
fn initResult(gpa: std.mem.Allocator, p: *Params, model_only: bool, res: *InitResult) !void {
    const a = p.arena;
    var mparams = common.modelParamsToLlama(p);
    var cparams = common.contextParamsToLlama(p);

    const model_path = try a.dupeSentinel(u8, p.model.path, 0);

    if (p.fit_params) {
        comTrc("common_init_result", "%s", .{@as([*:0]const u8, "fitting params to device memory ...\n")});
        comTrc("common_init_result", "%s", .{@as([*:0]const u8, "(for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on)\n")});

        // no draft model and no MTP in any tool ported: no extra model
        var patterns = try fit.Patterns.init(a);
        _ = try fit.fitParams(
            gpa,
            &patterns,
            .{ .verbosity = std.math.maxInt(i32), .sink = common.fitLogSink },
            model_path.ptr,
            @ptrCast(&mparams),
            @ptrCast(&cparams),
            &p.tensor_split,
            @ptrCast(p.tensor_buft_overrides.items.ptr),
            p.fit_params_target.items.ptr,
            @bitCast(p.fit_params_min_ctx),
            null,
            if (p.verbosity >= log.level_debug) c.GGML_LOG_LEVEL_DEBUG else c.GGML_LOG_LEVEL_ERROR,
        );
    }

    const model = c.llama_model_load_from_file(model_path.ptr, mparams) orelse return;
    res.model = model;

    if (model_only) return;

    const vocab = c.llama_model_get_vocab(model);

    // load and optionally apply lora adapters
    for (p.lora_adapters.items) |*la| {
        const lora = c.llama_adapter_lora_init(model, la.path.ptr) orelse {
            comErr("common_init_result", "failed to load lora adapter '%s'\n", .{la.path.ptr});
            return;
        };
        var buf: [1024]u8 = @splat(0);
        la.ptr = lora;
        _ = c.llama_adapter_meta_val_str(lora, "adapter.lora.task_name", &buf, buf.len);
        la.task_name = try a.dupe(u8, std.mem.sliceTo(&buf, 0));
        _ = c.llama_adapter_meta_val_str(lora, "adapter.lora.prompt_prefix", &buf, buf.len);
        la.prompt_prefix = try a.dupe(u8, std.mem.sliceTo(&buf, 0));
        try res.lora.append(a, lora); // copy to list of loaded adapters
    }

    // updates params.sampling
    try initSamplerFromModel(p, model);

    if (p.sampling.ignore_eos and c.llama_vocab_eos(vocab) == c.LLAMA_TOKEN_NULL) {
        comWrn("common_init_result", "%s", .{@as([*:0]const u8, "vocab does not have an EOS token, ignoring --ignore-eos\n")});
        p.sampling.ignore_eos = false;
    }

    // initialize once
    var logit_bias_eog: std.ArrayList(c.llama_logit_bias) = .empty;
    var i: c.llama_token = 0;
    while (i < c.llama_vocab_n_tokens(vocab)) : (i += 1) {
        if (c.llama_vocab_is_eog(vocab, i)) {
            comTrc("common_init_result", "added %s logit bias = %f\n", .{ (try common.tokenToPiece(a, vocab, i, true)).ptr, -std.math.inf(f64) });
            try logit_bias_eog.append(a, .{ .token = i, .bias = -std.math.inf(f32) });
        }
    }

    if (p.sampling.ignore_eos) {
        // add EOG biases to the active set of logit biases
        try p.sampling.logit_bias.appendSlice(a, logit_bias_eog.items);
    }

    // init the backend samplers as part of the context creation
    var seq_config: std.ArrayList(c.llama_sampler_seq_config) = .empty;
    var k: u32 = 0;
    while (k < cparams.n_seq_max) : (k += 1) {
        const s = samplerInit(p, model) catch |e| switch (e) {
            error.InvalidArgument => {
                io_.eprintf("libc++abi: terminating due to uncaught exception of type std::invalid_argument: %.*s\n", .{ @as(c_int, @intCast(arg.what().len)), arg.what().ptr });
                std.process.abort();
            },
            error.Runtime => {
                io_.eprintf("libc++abi: terminating due to uncaught exception of type std::runtime_error: %.*s\n", .{ @as(c_int, @intCast(arg.what().len)), arg.what().ptr });
                std.process.abort();
            },
            else => |other| return other,
        };
        try res.samplers.append(a, s);
        try seq_config.append(a, .{ .seq_id = @intCast(k), .sampler = s.get() });
    }

    if (p.sampling.backend_sampling) {
        cparams.samplers = seq_config.items.ptr;
        cparams.n_samplers = seq_config.items.len;
    }

    const lctx = c.llama_init_from_model(model, cparams) orelse {
        comErr("common_init_result", "failed to create context with model '%s'\n", .{model_path.ptr});
        return;
    };
    res.context = lctx;

    _ = setProcessPriority(p.cpuparams.priority);

    threadpoolsInit(res, lctx, p);
}

const io_ = @import("io.zig");

/// Ports `common_init_from_params` (common/common.cpp:1436 @c1d0e7a00).
///
/// Return: the model and context, either null where loading failed, having
/// logged why. The caller owns the result.
pub fn initFromParams(gpa: std.mem.Allocator, p: *Params, model_only: bool) !InitResult {
    const func = "common_init_from_params";
    var res: InitResult = .{};
    try initResult(gpa, p, model_only, &res);

    const model_path = try p.arena.dupeSentinel(u8, p.model.path, 0);
    const model = res.model orelse {
        comErr(func, "failed to load model '%s'\n", .{model_path.ptr});
        return res;
    };

    if (model_only) return res;

    const lctx = res.context orelse {
        comErr(func, "failed to create context with model '%s'\n", .{model_path.ptr});
        return res;
    };

    const vocab = c.llama_model_get_vocab(model);

    if (p.ctx_shift and !c.llama_memory_can_shift(c.llama_get_memory(lctx))) {
        comWrn(func, "%s", .{@as([*:0]const u8, "KV cache shifting is not supported for this context, disabling KV cache shifting\n")});
        p.ctx_shift = false;
    }

    if (p.control_vectors.items.len != 0) {
        if (p.control_vector_layer_start <= 0) p.control_vector_layer_start = 1;
        if (p.control_vector_layer_end <= 0) p.control_vector_layer_end = c.llama_model_n_layer(model);

        const cvec = try controlVectorLoad(p.arena, p.control_vectors.items);
        if (cvec.n_embd == -1) return res;

        const err = c.llama_set_adapter_cvec(lctx, cvec.data.items.ptr, cvec.data.items.len, cvec.n_embd, p.control_vector_layer_start, p.control_vector_layer_end);
        if (err != 0) return res;
    }

    if (c.llama_pooling_type(lctx) == c.LLAMA_POOLING_TYPE_RANK) {
        var ok = true;

        if (c.llama_vocab_bos(vocab) == c.LLAMA_TOKEN_NULL) {
            comWrn(func, "%s", .{@as([*:0]const u8, "vocab does not have a  BOS token, reranking will not work\n")});
            ok = false;
        }

        const has_eos = c.llama_vocab_eos(vocab) != c.LLAMA_TOKEN_NULL;
        const has_sep = c.llama_vocab_sep(vocab) != c.LLAMA_TOKEN_NULL;
        const has_rerank_prompt = c.llama_model_chat_template(model, "rerank") != null;

        if (!has_eos and !has_sep and !has_rerank_prompt) {
            comWrn(func, "%s", .{@as([*:0]const u8, "vocab does not have an EOS token, SEP token, or rerank prompt. Reranking will not work\n")});
            ok = false;
        } else if (!has_eos) {
            comWrn(func, "%s", .{@as([*:0]const u8, "vocab does not have an EOS token, using SEP token as fallback\n")});
        }

        if (!ok) return res;
    }

    if (!p.lora_init_without_apply) {
        // `common_set_adapter_lora` (common/common.cpp:1668 @c1d0e7a00)
        var loras: std.ArrayList(?*c.llama_adapter_lora) = .empty;
        var scales: std.ArrayList(f32) = .empty;
        for (p.lora_adapters.items) |la| {
            try loras.append(p.arena, la.ptr);
            try scales.append(p.arena, la.scale);
        }
        // an empty `std::vector`'s `data()` is null, and libllama logs it
        _ = c.llama_set_adapters_lora(
            lctx,
            if (loras.items.len == 0) null else loras.items.ptr,
            loras.items.len,
            if (scales.items.len == 0) null else scales.items.ptr,
        );
    }

    if (p.warmup) {
        comTrc(func, "%s", .{@as([*:0]const u8, "warming up the model with an empty run - please wait ... (--no-warmup to disable)\n")});

        var tmp: std.ArrayList(c.llama_token) = .empty;
        const bos = c.llama_vocab_bos(vocab);
        const eos = c.llama_vocab_eos(vocab);

        // some models (e.g. T5) don't have a BOS token
        if (bos != c.LLAMA_TOKEN_NULL) try tmp.append(p.arena, bos);
        if (eos != c.LLAMA_TOKEN_NULL) try tmp.append(p.arena, eos);
        if (tmp.items.len == 0) try tmp.append(p.arena, 0);

        if (c.llama_model_has_encoder(model)) {
            _ = c.llama_encode(lctx, c.llama_batch_get_one(tmp.items.ptr, @intCast(tmp.items.len)));
            var decoder_start_token_id = c.llama_model_decoder_start_token(model);
            if (decoder_start_token_id == c.LLAMA_TOKEN_NULL) decoder_start_token_id = bos;
            tmp.clearRetainingCapacity();
            try tmp.append(p.arena, decoder_start_token_id);
        }
        if (c.llama_model_has_decoder(model)) {
            _ = c.llama_decode(lctx, c.llama_batch_get_one(tmp.items.ptr, @intCast(@min(tmp.items.len, @as(usize, @intCast(p.n_batch))))));
        }
        c.llama_memory_clear(c.llama_get_memory(lctx), true);
        c.llama_synchronize(lctx);
        c.llama_perf_context_reset(lctx);

        // reset samplers to reset RNG state after warmup to the seeded state
        res.resetSamplers();
    }

    return res;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
