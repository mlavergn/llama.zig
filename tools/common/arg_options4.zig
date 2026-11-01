//! The option table, part four: from `--spec-draft-hf` to the end
//! -- arg.cpp's lines 3961 to 4709: the speculative decoder's options, the removed
//! ones that still parse, TTS, `--check`, and the server's model presets.
//!
//! # Provenance
//!
//! Ports that stretch of `common_params_parser_init` (arg.cpp:1391 @b29c606e2)
//! from `llama.cpp/common/arg.cpp` (v0.4.1, `b29c606e2`), as
//! `arg_options.zig` describes, with `arg_removed` from the same file and
//! the speculative type names from `llama.cpp/common/speculative.cpp`:
//! `common_speculative_type_from_name_map`,
//! `common_speculative_type_name_str`, `common_speculative_all_types_str`
//! and `common_speculative_types_from_names`. Each names what it ports and
//! the line it began at.
//!
//! The presets (`--fim-qwen-7b-default` and the rest) only set parameters;
//! the `-hf` repository they name is downloaded later, by the server, which
//! is where this port refuses it.

const std = @import("std");
const c = @import("tools_h");
const arg = @import("arg.zig");
const params_mod = @import("params.zig");
const log = @import("log.zig");
const io = @import("io.zig");
const options1 = @import("arg_options.zig");
const options2 = @import("arg_options2.zig");
const strings = @import("strings.zig");

const Params = params_mod.Params;
const SpeculativeType = params_mod.SpeculativeType;
const Arg = arg.Arg;
const Context = arg.Context;
const throw = arg.throw;
const throwStr = arg.throwStr;
const S = [:0]const u8;

const kvCacheTypeFromStr = options2.kvCacheTypeFromStr;

/// The examples nearly every speculative option takes.
const spec_examples = [_]params_mod.Example{ .speculative, .server, .cli };

fn tf(b: bool) [*:0]const u8 {
    return if (b) "true" else "false";
}

/// Ports `arg_removed` (arg.cpp:343 @b29c606e2).
fn argRemoved(comptime msg: []const u8) error{Throw} {
    return throwStr("the argument has been removed. " ++ msg);
}

/// Ports `common_speculative_type_from_name_map` (speculative.cpp:33 @b29c606e2),
/// in the `std::map`'s key order.
const type_from_name_map = [_]struct { []const u8, SpeculativeType }{
    .{ "draft-dflash", .draft_dflash }, .{ "draft-dspark", .draft_dspark },
    .{ "draft-eagle3", .draft_eagle3 }, .{ "draft-mtp", .draft_mtp },
    .{ "draft-simple", .draft_simple }, .{ "ngram-cache", .ngram_cache },
    .{ "ngram-map-k", .ngram_map_k },   .{ "ngram-map-k4v", .ngram_map_k4v },
    .{ "ngram-mod", .ngram_mod },       .{ "ngram-simple", .ngram_simple },
    .{ "none", .none },
};

/// Ports `common_speculative_type_name_str` (speculative.cpp:2222 @b29c606e2).
fn typeNameStr(a: std.mem.Allocator, types: []const SpeculativeType) ![:0]const u8 {
    var result: std.ArrayList(u8) = .empty;
    for (types, 0..) |t, i| {
        if (i > 0) try result.append(a, ',');
        try result.appendSlice(a, t.toStr());
    }
    return result.toOwnedSliceSentinel(a, 0);
}

/// Ports `common_speculative_all_types_str` (speculative.cpp:2234 @b29c606e2).
fn allTypesStr(a: std.mem.Allocator) ![:0]const u8 {
    var types: [@backingInt(SpeculativeType.count)]SpeculativeType = undefined;
    for (&types, 0..) |*t, i| t.* = @fromBackingInt(@intCast(i));
    return typeNameStr(a, &types);
}

/// Ports `common_speculative_types_from_names` (speculative.cpp:2263 @b29c606e2).
fn typesFromNames(a: std.mem.Allocator, names: []const []const u8) !std.ArrayList(SpeculativeType) {
    var types: std.ArrayList(SpeculativeType) = .empty;
    for (names) |name| {
        const found = for (type_from_name_map) |e| {
            if (std.mem.eql(u8, e[0], name)) break e[1];
        } else null;
        if (found) |t| {
            if (t == .none) {
                var only: std.ArrayList(SpeculativeType) = .empty;
                try only.append(a, .none);
                return only;
            }
            try types.append(a, t);
            continue;
        }
        var buf: [4096]u8 = undefined;
        return throwStr(std.fmt.bufPrint(&buf, "unknown speculative type: {s}", .{name}) catch "unknown speculative type");
    }
    return types;
}

extern fn strtod(s: [*:0]const u8, end: *?[*:0]const u8) f64;

/// `std::stod(text, &pos)`, as libc++'s `as_float_helper` throws.
///
/// Return: the value and how many bytes it read.
fn stod(a: std.mem.Allocator, text: []const u8) !struct { f64, usize } {
    const s = try a.dupeSentinel(u8, text, 0);
    var end: ?[*:0]const u8 = null;
    std.c._errno().* = 0;
    const v = strtod(s.ptr, &end);
    if (std.c._errno().* == @backingInt(std.c.E.RANGE)) return throwStr("stod: out of range");
    if (end == s.ptr) return throwStr("stod: no conversion");
    return .{ v, @intFromPtr(end.?) - @intFromPtr(s.ptr) };
}

/// The handler of the `--spec-ngram-*-size-n` and `-size-m` options: a
/// size in [1, 1024], or `msg`.
fn ngramSize(value: c_int, comptime msg: []const u8) !u16 {
    if (value < 1 or value > 1024) return throwStr(msg);
    return @intCast(value);
}

/// The handler of the `--spec-ngram-*-min-hits` options: at least 1.
fn ngramMinHits(value: c_int) !u16 {
    if (value < 1) return throwStr("ngram min hits must be at least 1");
    return @truncate(@as(c_uint, @bitCast(value)));
}

/// The table, part four.
pub fn add(ctx: *Context) !void {
    const p = ctx.params;
    const kv_types = try options2.allKvCacheTypes(p.arena);

    //
    // speculative parameters
    //

    // arg.cpp:3961
    try ctx.add((Arg{ .args = &.{ "--spec-draft-hf", "-hfd", "-hfrd", "--hf-repo-draft" }, .value_hint = "<user>/<model>[:quant]", .help = "Same as --hf-repo, but for the draft model (default: unused)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.speculative.draft.mparams.hf_repo = try arg.dupe(q, value);
        }
    }.f } }).setSpec().setExamples(&spec_examples).setEnv("LLAMA_ARG_SPEC_DRAFT_HF_REPO"));
    // arg.cpp:3968
    try ctx.add((Arg{ .args = &.{ "--spec-draft-threads", "-td", "--threads-draft" }, .value_hint = "N", .help = "number of threads to use during generation (default: same as --threads)", .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.draft.cpuparams.n_threads = value;
            if (q.speculative.draft.cpuparams.n_threads <= 0) q.speculative.draft.cpuparams.n_threads = @intCast(std.Thread.getCpuCount() catch 0);
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:3978
    try ctx.add((Arg{ .args = &.{ "--spec-draft-threads-batch", "-tbd", "--threads-batch-draft" }, .value_hint = "N", .help = "number of threads to use during batch and prompt processing (default: same as --threads-draft)", .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.draft.cpuparams_batch.n_threads = value;
            if (q.speculative.draft.cpuparams_batch.n_threads <= 0) q.speculative.draft.cpuparams_batch.n_threads = @intCast(std.Thread.getCpuCount() catch 0);
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:3988
    try ctx.add((Arg{ .args = &.{ "--spec-draft-cpu-mask", "-Cd", "--cpu-mask-draft" }, .value_hint = "M", .help = "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)", .handler = .{ .string = struct {
        fn f(q: *Params, mask: S) !void {
            q.speculative.draft.cpuparams.mask_valid = true;
            if (!options1.parseCpuMask(mask, &q.speculative.draft.cpuparams.cpumask)) return throwStr("invalid cpumask");
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:3998
    try ctx.add((Arg{ .args = &.{ "--spec-draft-cpu-range", "-Crd", "--cpu-range-draft" }, .value_hint = "lo-hi", .help = "Ranges of CPUs for affinity. Complements --cpu-mask-draft", .handler = .{ .string = struct {
        fn f(q: *Params, range: S) !void {
            q.speculative.draft.cpuparams.mask_valid = true;
            if (!try options1.parseCpuRange(q.arena, range, &q.speculative.draft.cpuparams.cpumask)) return throwStr("invalid range");
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4008
    try ctx.add((Arg{ .args = &.{ "--spec-draft-cpu-strict", "--cpu-strict-draft" }, .value_hint = "<0|1>", .help = "Use strict CPU placement for draft model (default: same as --cpu-strict)", .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.draft.cpuparams.strict_cpu = value != 0;
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4015
    try ctx.add((Arg{ .args = &.{ "--spec-draft-prio", "--prio-draft" }, .value_hint = "N", .help = try ctx.fmt("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", .{@as(c_int, @intCast(p.speculative.draft.cpuparams.priority))}), .handler = .{ .int = struct {
        fn f(q: *Params, prio: c_int) !void {
            if (prio < 0 or prio > 3) return throwStr("invalid value");
            q.speculative.draft.cpuparams.priority = @intCast(prio);
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4025
    try ctx.add((Arg{ .args = &.{ "--spec-draft-poll", "--poll-draft" }, .value_hint = "<0|1>", .help = "Use polling to wait for draft model work (default: same as --poll)", .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.draft.cpuparams.poll = @bitCast(value);
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4032
    try ctx.add((Arg{ .args = &.{ "--spec-draft-cpu-mask-batch", "-Cbd", "--cpu-mask-batch-draft" }, .value_hint = "M", .help = "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)", .handler = .{ .string = struct {
        fn f(q: *Params, mask: S) !void {
            q.speculative.draft.cpuparams_batch.mask_valid = true;
            if (!options1.parseCpuMask(mask, &q.speculative.draft.cpuparams_batch.cpumask)) return throwStr("invalid cpumask");
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4042
    try ctx.add((Arg{ .args = &.{ "--spec-draft-cpu-range-batch", "-Crbd", "--cpu-range-batch-draft" }, .value_hint = "lo-hi", .help = "Ranges of CPUs for affinity. Complements --cpu-mask-draft-batch)", .handler = .{ .string = struct {
        fn f(q: *Params, range: S) !void {
            q.speculative.draft.cpuparams_batch.mask_valid = true;
            if (!try options1.parseCpuRange(q.arena, range, &q.speculative.draft.cpuparams_batch.cpumask)) return throwStr("invalid cpumask");
        }
    }.f } }).setSpec().setExamples(&.{.speculative}));
    // arg.cpp:4052
    try ctx.add((Arg{ .args = &.{ "--spec-draft-cpu-strict-batch", "--cpu-strict-batch-draft" }, .value_hint = "<0|1>", .help = "Use strict CPU placement for draft model (default: --cpu-strict-draft)", .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.draft.cpuparams_batch.strict_cpu = value != 0;
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4059
    try ctx.add((Arg{ .args = &.{ "--spec-draft-prio-batch", "--prio-batch-draft" }, .value_hint = "N", .help = try ctx.fmt("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", .{@as(c_int, @intCast(p.speculative.draft.cpuparams_batch.priority))}), .handler = .{ .int = struct {
        fn f(q: *Params, prio: c_int) !void {
            if (prio < 0 or prio > 3) return throwStr("invalid value");
            q.speculative.draft.cpuparams_batch.priority = @intCast(prio);
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4069
    try ctx.add((Arg{ .args = &.{ "--spec-draft-poll-batch", "--poll-batch-draft" }, .value_hint = "<0|1>", .help = "Use polling to wait for draft model work (default: --poll-draft)", .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.draft.cpuparams_batch.poll = @bitCast(value);
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4076
    try ctx.add((Arg{ .args = &.{ "--spec-draft-type-k", "-ctkd", "--cache-type-k-draft" }, .value_hint = "TYPE", .help = try ctx.fmt("KV cache data type for K for the draft model\nallowed values: %s\n(default: %s)", .{ kv_types.ptr, c.ggml_type_name(p.speculative.draft.cache_type_k) }), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.speculative.draft.cache_type_k = try kvCacheTypeFromStr(value);
        }
    }.f } }).setEnv("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K"));
    // arg.cpp:4089
    try ctx.add((Arg{ .args = &.{ "--spec-draft-type-v", "-ctvd", "--cache-type-v-draft" }, .value_hint = "TYPE", .help = try ctx.fmt("KV cache data type for V for the draft model\nallowed values: %s\n(default: %s)", .{ kv_types.ptr, c.ggml_type_name(p.speculative.draft.cache_type_v) }), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.speculative.draft.cache_type_v = try kvCacheTypeFromStr(value);
        }
    }.f } }).setEnv("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V"));
    // arg.cpp:4102
    try ctx.add((Arg{ .args = &.{ "--spec-draft-override-tensor", "-otd", "--override-tensor-draft" }, .value_hint = "<tensor name pattern>=<buffer type>,...", .help = "override tensor buffer type for draft model", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            try options2.parseTensorBufferOverrides(q, value, &q.speculative.draft.tensor_buft_overrides);
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4108
    try ctx.add((Arg{
        .args = &.{ "--spec-draft-cpu-moe", "-cmoed", "--cpu-moe-draft" },
        .help = "keep all Mixture of Experts (MoE) weights in the CPU for the draft model",
        .handler = .{
            .void_ = struct {
                fn f(q: *Params) !void {
                    // `llm_ffn_exps_cpu_override` (common/common.h:1139 @b29c606e2)
                    try q.speculative.draft.tensor_buft_overrides.append(q.arena, .{ .pattern = options2.llm_ffn_exps_regex, .buft = c.ggml_backend_cpu_buffer_type() });
                }
            }.f,
        },
    }).setSpec().setExamples(&spec_examples).setEnv("LLAMA_ARG_SPEC_DRAFT_CPU_MOE"));
    // arg.cpp:4115
    try ctx.add((Arg{ .args = &.{ "--spec-draft-n-cpu-moe", "--spec-draft-ncmoe", "-ncmoed", "--n-cpu-moe-draft" }, .value_hint = "N", .help = "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU for the draft model", .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            if (value < 0) return throwStr("invalid value");
            try options2.llmAddNCpuFfnOverrides(q, value, options2.llm_ffn_exps_regex, &q.speculative.draft.tensor_buft_overrides);
        }
    }.f } }).setSpec().setExamples(&spec_examples).setEnv("LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE"));
    // arg.cpp:4126
    try ctx.add((Arg{ .args = &.{"--spec-draft-n-max"}, .value_hint = "N", .help = try ctx.fmt("number of tokens to draft for speculative decoding (default: %d)", .{p.speculative.draft.n_max}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            if (value < 0) return throwStr("invalid value");
            q.speculative.draft.n_max = value;
        }
    }.f } }).setSpec().setExamples(&.{ .speculative, .lookup, .server, .cli }).setEnv("LLAMA_ARG_SPEC_DRAFT_N_MAX"));
    // arg.cpp:4136
    try ctx.add((Arg{ .args = &.{"--spec-draft-n-min"}, .value_hint = "N", .help = try ctx.fmt("minimum number of draft tokens to use for speculative decoding (default: %d)", .{p.speculative.draft.n_min}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.draft.n_min = value;
        }
    }.f } }).setSpec().setExamples(&.{ .speculative, .lookup, .server, .cli }).setEnv("LLAMA_ARG_SPEC_DRAFT_N_MIN"));
    // arg.cpp:4143
    try ctx.add((Arg{ .args = &.{"--spec-synth-len"}, .value_hint = "L", .help = "target mean synthetic acceptance length, including the target token (benchmarking only)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const text = strings.strip(value);
            const length, const pos = try stod(q.arena, text);
            if (pos != text.len or length == -1.0) return throwStr("invalid value");
            q.speculative.synth_len = length;
        }
    }.f } }).setSpec().setExamples(&.{ .server, .cli }).setEnv("LLAMA_ARG_SPEC_SYNTH_LEN"));
    // arg.cpp:4156
    try ctx.add((Arg{ .args = &.{"--spec-synth-rates"}, .value_hint = "P0,P1,...", .help = "comma-separated unconditional per-position synthetic acceptance probabilities (benchmarking only)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const values = try strings.split(q.arena, value, ',');
            var rates: std.ArrayList(f64) = .empty;
            for (values.items) |raw| {
                const text = strings.strip(raw);
                const rate, const pos = try stod(q.arena, text);
                if (pos != text.len) return throwStr("invalid value");
                try rates.append(q.arena, rate);
            }
            q.speculative.synth_rates = rates;
        }
    }.f } }).setSpec().setExamples(&.{ .server, .cli }).setEnv("LLAMA_ARG_SPEC_SYNTH_RATES"));
    // arg.cpp:4176
    try ctx.add((Arg{ .args = &.{ "--spec-draft-p-split", "--draft-p-split" }, .value_hint = "P", .help = try ctx.fmt("speculative decoding split probability (default: %.2f)", .{@as(f64, p.speculative.draft.p_split)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.speculative.draft.p_split = try arg.stof(value);
        }
    }.f } }).setSpec().setExamples(&spec_examples).setEnv("LLAMA_ARG_SPEC_DRAFT_P_SPLIT"));
    // arg.cpp:4183
    try ctx.add((Arg{ .args = &.{ "--spec-draft-p-min", "--draft-p-min" }, .value_hint = "P", .help = try ctx.fmt("minimum speculative decoding probability (greedy) (default: %.2f)", .{@as(f64, p.speculative.draft.p_min)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.speculative.draft.p_min = try arg.stof(value);
        }
    }.f } }).setSpec().setExamples(&spec_examples).setEnv("LLAMA_ARG_SPEC_DRAFT_P_MIN"));
    // arg.cpp:4190
    try ctx.add((Arg{ .args = &.{"--spec-draft-backend-sampling"}, .args_neg = &.{"--no-spec-draft-backend-sampling"}, .help = try ctx.fmt("offload draft sampling to the backend (default: %s)", .{@as([*:0]const u8, if (p.speculative.draft.backend_sampling) "enabled" else "disabled")}), .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.speculative.draft.backend_sampling = value;
        }
    }.f } }).setSpec().setExamples(&spec_examples).setEnv("LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING"));
    // arg.cpp:4199
    try ctx.add((Arg{ .args = &.{ "--spec-draft-device", "-devd", "--device-draft" }, .value_hint = "<dev1,dev2,..>", .help = "comma-separated list of devices to use for offloading the draft model (none = don't offload, default: follows --device)\nuse --list-devices to see a list of available devices", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.speculative.draft.devices = try options2.parseDeviceList(q, value);
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4208
    std.debug.assert(p.speculative.draft.n_gpu_layers < 0); // string_format would need to be extended for a default >= 0
    try ctx.add((Arg{ .args = &.{ "--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft" }, .value_hint = "N", .help = try ctx.fmt("max. number of draft model layers to store in VRAM, either an exact number, 'auto', or 'all' (default: %s)", .{@as([*:0]const u8, if (p.speculative.draft.n_gpu_layers == -1) "auto" else "all")}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (std.mem.eql(u8, value, "auto")) {
                q.speculative.draft.n_gpu_layers = -1;
            } else if (std.mem.eql(u8, value, "all")) {
                q.speculative.draft.n_gpu_layers = -2;
            } else {
                q.speculative.draft.n_gpu_layers = try arg.stoi(value);
            }
            if (!c.llama_supports_gpu_offload()) {
                io.eprintf("warning: no usable GPU found, --gpu-layers-draft option will be ignored\n", .{});
                io.eprintf("warning: one possible reason is that llama.cpp was compiled without GPU support\n", .{});
                io.eprintf("warning: consult docs/build.md for compilation instructions\n", .{});
            }
        }
    }.f } }).setSpec().setExamples(&spec_examples).setEnv("LLAMA_ARG_N_GPU_LAYERS_DRAFT"));
    // arg.cpp:4227
    try ctx.add((Arg{
        .args = &.{ "--spec-draft-model", "-md", "--model-draft" },
        .value_hint = "FNAME",
        .help = "draft model for speculative decoding (default: unused)",
        .handler = .{
            .string = struct {
                fn f(q: *Params, value: S) !void {
                    q.speculative.draft.mparams.path = try arg.dupe(q, value);
                    q.speculative.draft.mparams.hf_file = q.speculative.draft.mparams.path; // will be used if --spec-draft-hf is set
                }
            }.f,
        },
    }).setSpec().setExamples(&spec_examples).setEnv("LLAMA_ARG_SPEC_DRAFT_MODEL"));
    // arg.cpp:4235
    try ctx.add((Arg{ .args = &.{"--spec-type"}, .value_hint = try allTypesStr(p.arena), .help = try ctx.fmt("comma-separated list of types of speculative decoding to use (default: %s)\n", .{(try typeNameStr(p.arena, p.speculative.types.items)).ptr}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const types_str = try strings.split(q.arena, value, ',');
            const types = try typesFromNames(q.arena, types_str.items);
            try q.speculative.types.appendSlice(q.arena, types.items);
        }
    }.f } }).setSpec().setExamples(&spec_examples).setEnv("LLAMA_ARG_SPEC_TYPE"));
    // arg.cpp:4245
    try ctx.add((Arg{ .args = &.{"--spec-ngram-mod-n-min"}, .value_hint = "N", .help = try ctx.fmt("minimum number of ngram tokens to use for ngram-based speculative decoding (default: %d)", .{p.speculative.ngram_mod.n_min}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            if (value < 0 or value > 1024) return throwStr("ngram n-min must be between 0 and 1024 inclusive");
            q.speculative.ngram_mod.n_min = value;
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4255
    try ctx.add((Arg{ .args = &.{"--spec-ngram-mod-n-max"}, .value_hint = "N", .help = try ctx.fmt("maximum number of ngram tokens to use for ngram-based speculative decoding (default: %d)", .{p.speculative.ngram_mod.n_max}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            if (value < 0 or value > 1024) return throwStr("ngram n-max must be between 0 and 1024 inclusive");
            q.speculative.ngram_mod.n_max = value;
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4265
    try ctx.add((Arg{ .args = &.{"--spec-ngram-mod-n-match"}, .value_hint = "N", .help = try ctx.fmt("ngram-mod lookup length (default: %d)", .{p.speculative.ngram_mod.n_match}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            if (value < 1 or value > 1024) return throwStr("ngram size N must be between 1 and 1024 inclusive");
            q.speculative.ngram_mod.n_match = value;
        }
    }.f } }).setSpec().setExamples(&spec_examples));

    // arg.cpp:4276
    try ctx.add((Arg{ .args = &.{"--spec-ngram-simple-size-n"}, .value_hint = "N", .help = try ctx.fmt("ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: %d)", .{@as(c_int, p.speculative.ngram_simple.size_n)}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.ngram_simple.size_n = try ngramSize(value, "ngram size N must be between 1 and 1024 inclusive");
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4286
    try ctx.add((Arg{ .args = &.{"--spec-ngram-simple-size-m"}, .value_hint = "N", .help = try ctx.fmt("ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: %d)", .{@as(c_int, p.speculative.ngram_simple.size_m)}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.ngram_simple.size_m = try ngramSize(value, "ngram size M must be between 1 and 1024 inclusive");
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4296
    try ctx.add((Arg{ .args = &.{"--spec-ngram-simple-min-hits"}, .value_hint = "N", .help = try ctx.fmt("minimum hits for ngram-simple speculative decoding (default: %d)", .{@as(c_int, p.speculative.ngram_simple.min_hits)}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.ngram_simple.min_hits = try ngramMinHits(value);
        }
    }.f } }).setSpec().setExamples(&spec_examples));

    // arg.cpp:4307
    try ctx.add((Arg{ .args = &.{"--spec-ngram-map-k-size-n"}, .value_hint = "N", .help = try ctx.fmt("ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: %d)", .{@as(c_int, p.speculative.ngram_map_k.size_n)}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.ngram_map_k.size_n = try ngramSize(value, "ngram size N must be between 1 and 1024 inclusive");
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4317
    try ctx.add((Arg{ .args = &.{"--spec-ngram-map-k-size-m"}, .value_hint = "N", .help = try ctx.fmt("ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: %d)", .{@as(c_int, p.speculative.ngram_map_k.size_m)}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.ngram_map_k.size_m = try ngramSize(value, "ngram size M must be between 1 and 1024 inclusive");
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4327
    try ctx.add((Arg{ .args = &.{"--spec-ngram-map-k-min-hits"}, .value_hint = "N", .help = try ctx.fmt("minimum hits for ngram-map-k speculative decoding (default: %d)", .{@as(c_int, p.speculative.ngram_map_k.min_hits)}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.ngram_map_k.min_hits = try ngramMinHits(value);
        }
    }.f } }).setSpec().setExamples(&spec_examples));

    // arg.cpp:4338
    try ctx.add((Arg{ .args = &.{"--spec-ngram-map-k4v-size-n"}, .value_hint = "N", .help = try ctx.fmt("ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: %d)", .{@as(c_int, p.speculative.ngram_map_k4v.size_n)}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.ngram_map_k4v.size_n = try ngramSize(value, "ngram size N must be between 1 and 1024 inclusive");
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4348
    try ctx.add((Arg{ .args = &.{"--spec-ngram-map-k4v-size-m"}, .value_hint = "N", .help = try ctx.fmt("ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: %d)", .{@as(c_int, p.speculative.ngram_map_k4v.size_m)}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.ngram_map_k4v.size_m = try ngramSize(value, "ngram size M must be between 1 and 1024 inclusive");
        }
    }.f } }).setSpec().setExamples(&spec_examples));
    // arg.cpp:4358
    try ctx.add((Arg{ .args = &.{"--spec-ngram-map-k4v-min-hits"}, .value_hint = "N", .help = try ctx.fmt("minimum hits for ngram-map-k4v speculative decoding (default: %d)", .{@as(c_int, p.speculative.ngram_map_k4v.min_hits)}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.speculative.ngram_map_k4v.min_hits = try ngramMinHits(value);
        }
    }.f } }).setSpec().setExamples(&spec_examples));

    //
    // removed params
    //

    // arg.cpp:4373
    try ctx.add((Arg{ .args = &.{ "--draft", "--draft-n", "--draft-max" }, .value_hint = "N", .help = "the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max", .handler = .{ .int = struct {
        fn f(_: *Params, _: c_int) !void {
            return argRemoved("use --spec-draft-n-max or --spec-ngram-mod-n-max");
        }
    }.f } }).setSpec().setExamples(&.{ .speculative, .lookup, .server, .cli }).setEnv("LLAMA_ARG_DRAFT_MAX"));
    // arg.cpp:4380
    try ctx.add((Arg{ .args = &.{ "--draft-min", "--draft-n-min" }, .value_hint = "N", .help = "the argument has been removed. use --spec-draft-n-min or --spec-ngram-mod-n-min", .handler = .{ .int = struct {
        fn f(_: *Params, _: c_int) !void {
            return argRemoved("use --spec-draft-n-min or --spec-ngram-mod-n-min");
        }
    }.f } }).setSpec().setExamples(&.{ .speculative, .lookup, .server, .cli }).setEnv("LLAMA_ARG_DRAFT_MIN"));
    // arg.cpp:4387
    try ctx.add((Arg{ .args = &.{"--spec-ngram-size-n"}, .value_hint = "N", .help = "the argument has been removed. use the respective --spec-ngram-*-size-n or --spec-ngram-mod-n-match", .handler = .{ .int = struct {
        fn f(_: *Params, _: c_int) !void {
            return argRemoved("use the respective --spec-ngram-*-size-n");
        }
    }.f } }).setSpec().setExamples(&.{.server}));
    // arg.cpp:4394
    try ctx.add((Arg{ .args = &.{"--spec-ngram-size-m"}, .value_hint = "N", .help = "the argument has been removed. use the respective --spec-ngram-*-size-m", .handler = .{ .int = struct {
        fn f(_: *Params, _: c_int) !void {
            return argRemoved("use the respective --spec-ngram-*-size-m");
        }
    }.f } }).setSpec().setExamples(&.{.server}));
    // arg.cpp:4401
    try ctx.add((Arg{ .args = &.{"--spec-ngram-min-hits"}, .value_hint = "N", .help = "the argument has been removed. use the respective --spec-ngram-*-min-hits", .handler = .{ .int = struct {
        fn f(_: *Params, _: c_int) !void {
            return argRemoved("use the respective --spec-ngram-*-min-hits");
        }
    }.f } }).setSpec().setExamples(&.{.server}));

    //
    // TTS params
    //

    // arg.cpp:4413
    try ctx.add((Arg{ .args = &.{"--tts-lang"}, .value_hint = "FNAME", .help = "language (ISO 639-1) for audio generation\nsee tts/README.md for per-model usage notes", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.tts_lang = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.tts}));
    // arg.cpp:4421
    try ctx.add((Arg{ .args = &.{"--tts-speaker-file"}, .value_hint = "FNAME", .help = "speaker file path for audio generation", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.tts_speaker_file = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.tts}));
    // arg.cpp:4515
    try ctx.add((Arg{ .args = &.{"--check"}, .help = try ctx.fmt("check rather than generate results (default: %s)", .{tf(p.check)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.check = true;
        }
    }.f } }).setExamples(&.{.results}));

    // presets

    // arg.cpp:4546
    try ctx.add((Arg{ .args = &.{"--embd-gemma-default"}, .help = "use default EmbeddingGemma model (note: can download weights from the internet)", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.model.hf_repo = "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF";
            q.model.hf_file = "embeddinggemma-300M-qat-Q4_0.gguf";
            q.port = 8011;
            q.n_ubatch = 2048;
            q.n_batch = 2048;
            q.n_parallel = 32;
            q.n_ctx = 2048 * q.n_parallel;
            q.verbose_prompt = true;
            q.embedding = true;
        }
    }.f } }).setExamples(&.{ .embedding, .server }));
    // arg.cpp:4562
    try ctx.add((Arg{ .args = &.{"--fim-qwen-1.5b-default"}, .help = "use default Qwen 2.5 Coder 1.5B (note: can download weights from the internet)", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            fimPreset(q, "ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF", "qwen2.5-coder-1.5b-q8_0.gguf");
        }
    }.f } }).setExamples(&.{.server}));
    // arg.cpp:4576
    try ctx.add((Arg{ .args = &.{"--fim-qwen-3b-default"}, .help = "use default Qwen 2.5 Coder 3B (note: can download weights from the internet)", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            fimPreset(q, "ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF", "qwen2.5-coder-3b-q8_0.gguf");
        }
    }.f } }).setExamples(&.{.server}));
    // arg.cpp:4590
    try ctx.add((Arg{ .args = &.{"--fim-qwen-7b-default"}, .help = "use default Qwen 2.5 Coder 7B (note: can download weights from the internet)", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            fimPreset(q, "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF", "qwen2.5-coder-7b-q8_0.gguf");
        }
    }.f } }).setExamples(&.{.server}));
    // arg.cpp:4604
    try ctx.add((Arg{ .args = &.{"--fim-qwen-7b-spec"}, .help = "use Qwen 2.5 Coder 7B + 0.5B draft for speculative decoding (note: can download weights from the internet)", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.speculative.draft.mparams.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
            q.speculative.draft.mparams.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf";
            fimPreset(q, "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF", "qwen2.5-coder-7b-q8_0.gguf");
        }
    }.f } }).setExamples(&.{.server}));
    // arg.cpp:4620
    try ctx.add((Arg{ .args = &.{"--fim-qwen-14b-spec"}, .help = "use Qwen 2.5 Coder 14B + 0.5B draft for speculative decoding (note: can download weights from the internet)", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.speculative.draft.mparams.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
            q.speculative.draft.mparams.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf";
            fimPreset(q, "ggml-org/Qwen2.5-Coder-14B-Q8_0-GGUF", "qwen2.5-coder-14b-q8_0.gguf");
        }
    }.f } }).setExamples(&.{.server}));
    // arg.cpp:4636
    try ctx.add((Arg{ .args = &.{"--fim-qwen-30b-default"}, .help = "use default Qwen 3 Coder 30B A3B Instruct (note: can download weights from the internet)", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            fimPreset(q, "ggml-org/Qwen3-Coder-30B-A3B-Instruct-Q8_0-GGUF", "qwen3-coder-30b-a3b-instruct-q8_0.gguf");
        }
    }.f } }).setExamples(&.{.server}));
    // arg.cpp:4650
    try ctx.add((Arg{ .args = &.{"--gpt-oss-20b-default"}, .help = "use gpt-oss-20b (note: can download weights from the internet)", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.model.hf_repo = "ggml-org/gpt-oss-20b-GGUF";
            q.model.hf_file = "gpt-oss-20b-mxfp4.gguf";
            gptOssPreset(q);
        }
    }.f } }).setExamples(&.{ .server, .cli }));
    // arg.cpp:4669
    try ctx.add((Arg{ .args = &.{"--gpt-oss-120b-default"}, .help = "use gpt-oss-120b (note: can download weights from the internet)", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.model.hf_repo = "ggml-org/gpt-oss-120b-GGUF";
            gptOssPreset(q);
        }
    }.f } }).setExamples(&.{ .server, .cli }));
    // arg.cpp:4687
    try ctx.add((Arg{ .args = &.{"--vision-gemma-4b-default"}, .help = "use Gemma 3 4B QAT (note: can download weights from the internet)", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.model.hf_repo = "ggml-org/gemma-3-4b-it-qat-GGUF";
            visionPreset(q);
        }
    }.f } }).setExamples(&.{ .server, .cli }));
    // arg.cpp:4698
    try ctx.add((Arg{ .args = &.{"--vision-gemma-12b-default"}, .help = "use Gemma 3 12B QAT (note: can download weights from the internet)", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.model.hf_repo = "ggml-org/gemma-3-12b-it-qat-GGUF";
            visionPreset(q);
        }
    }.f } }).setExamples(&.{ .server, .cli }));
    // arg.cpp:4709
    try ctx.add((Arg{ .args = &.{"--spec-default"}, .help = "enable default speculative decoding config", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            try q.speculative.types.append(q.arena, .ngram_mod);
            q.speculative.ngram_mod.n_match = 24;
            q.speculative.ngram_mod.n_min = 48;
            q.speculative.ngram_mod.n_max = 64;
        }
    }.f } }).setExamples(&.{ .server, .cli }));
}

/// What the six `--fim-qwen-*` presets share, from arg.cpp's line 4562 on:
/// the repository and file, then the same port, batch and context.
fn fimPreset(q: *Params, repo: []const u8, file: []const u8) void {
    q.model.hf_repo = repo;
    q.model.hf_file = file;
    q.port = 8012;
    q.n_ubatch = 1024;
    q.n_batch = 1024;
    q.n_ctx = 0;
    q.n_cache_reuse = 256;
}

/// What the two `--gpt-oss-*` presets, arg.cpp's lines 4650 and 4669,
/// share after the repository.
fn gptOssPreset(q: *Params) void {
    q.port = 8013;
    q.n_ubatch = 2048;
    q.n_batch = 32768;
    q.n_parallel = 2;
    q.n_ctx = 131072 * q.n_parallel;
    q.sampling.temp = 1.0;
    q.sampling.top_p = 1.0;
    q.sampling.top_k = 0;
    q.sampling.min_p = 0.01;
    q.use_jinja = true;
}

/// What the two `--vision-gemma-*` presets, arg.cpp's lines 4687 and 4698,
/// share after the repository.
fn visionPreset(q: *Params) void {
    q.port = 8014;
    q.n_ctx = 0;
    q.use_jinja = true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the speculative type names are upstream's, in its order" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    try std.testing.expectEqualStrings(
        "none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache",
        try allTypesStr(arena.allocator()),
    );
}
